//! kernel_emit — generate the AOT kernel sources (design §4/§17, M3).
//!
//! Emits every lane witness kernel and every component's fused constraint
//! kernel(s) as `.cu` files into the stwo repo's
//! `crates/backend-cuda-kernels/cuda/generated/`, plus `aot_manifest.json`
//! (kind, label, kernel name, cache key). The source text and cache keys are
//! BYTE-IDENTICAL to what the NVRTC lane compiles at prove time (same codegen)
//! — build.rs compiles these offline (nvcc -O3, per-arch cubins) and the
//! runtime consults the embedded table before NVRTC, so a cache-key miss IS
//! the drift check.
//!
//! Witness kernels come from the lane registry (`all_lane_recordings`,
//! fixture-independent — recorded programs are straight-line). Constraint
//! kernels need component EVALUATORS, built from a fixture matrix chosen for
//! union coverage (all-opcodes + all-builtins on Canonical; pedersen on
//! CanonicalSmall for the narrow-windows/w9 family). The lowering hoists all
//! statement constants into parameters, so any statement emits the same
//! kernels; a component absent from every fixture FAILS the run (loud gap).
//!
//! Usage: kernel_emit [--stwo-root <path>] [--max-instrs N] [--check]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::relations::CommonLookupElements;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::channel::Blake2sChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_backend_cuda::aot;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, state};
use stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Per-kernel instruction cap for AOT constraint lowering. MUST MATCH the
/// runtime lowering cap (`DEFAULT_MAX_KERNEL_INSTRS` in stwo's jit/mod.rs =
/// 2048): the semantic hash depends on the split, so a mismatched cap means the
/// runtime's keys never hit the pack (silent NVRTC fallback — the drift check
/// firing on everything). Manifest steps that override
/// STWO_JIT_MAX_KERNEL_INSTRS must use this value too; on AOT builds the 512
/// sm_90 override is obsolete (no load-time ptxas). Raising both caps together
/// (bigger fused kernels, fewer launches) is a measured follow-up.
const DEFAULT_AOT_MAX_INSTRS: usize = 2048;

struct Emitted {
    kind: &'static str,
    label: String,
    kernel: aot::EmittedKernel,
}

fn run_fixture(
    program: &str,
    variant: PreProcessedTraceVariant,
    out: &mut Vec<Emitted>,
    covered: &mut BTreeMap<String, bool>,
    max_instrs: usize,
) {
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    eprintln!("kernel_emit: fixture {program} ({variant:?})");
    let compiled = get_compiled_cairo_program_path(program);
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run_and_adapt fixture");

    let state::IngestOutput {
        preprocessed_trace,
        generator,
        proof_plan,
    } = phases::ingest::run(input, variant, None);
    let state::WitnessOutput {
        trace: _trace,
        claim,
        interaction_generator,
        device,
    } = phases::witness::run::<SimdBackend>(
        generator,
        Arc::new(
            CAIRO_SCHEDULE
                .artifact_plan()
                .expect("valid Cairo schedule"),
        ),
        proof_plan,
        None,
        None,
    );
    // Any elements work: the lowering hoists them into parameters; hashes are
    // statement-independent.
    let elements = CommonLookupElements::draw(&mut Blake2sChannel::default());
    let (_evals, interaction_claim) =
        phases::interaction::run(interaction_generator, &device, &elements);
    let components = CairoComponents::new(
        &claim,
        &elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );

    macro_rules! emit_fields {
        ($( $field:ident ),+ $(,)?) => {
            $(
                if let Some(c) = &components.$field {
                    covered.insert(stringify!($field).to_string(), true);
                    emit_component(stringify!($field), c, out, max_instrs);
                } else {
                    covered.entry(stringify!($field).to_string()).or_insert(false);
                }
            )+
        };
    }
    fn emit_component<E: stwo_constraint_framework::FrameworkEval>(
        field: &str,
        component: &stwo_constraint_framework::FrameworkComponent<E>,
        out: &mut Vec<Emitted>,
        max_instrs: usize,
    ) {
        let kernels = aot::constraint_kernel_sources(
            component.evaluator(),
            3,
            component.claimed_sum(),
            component.evaluator().log_size(),
            max_instrs,
        )
        .unwrap_or_else(|| panic!("constraint lowering failed for {field}"));
        for k in kernels {
            out.push(Emitted {
                kind: "constraint",
                label: field.to_string(),
                kernel: k,
            });
        }
    }

    emit_fields!(
        add_opcode,
        add_opcode_small,
        add_ap_opcode,
        assert_eq_opcode,
        assert_eq_opcode_imm,
        assert_eq_opcode_double_deref,
        blake_compress_opcode,
        call_opcode_abs,
        call_opcode_rel_imm,
        generic_opcode,
        jnz_opcode_non_taken,
        jnz_opcode_taken,
        jump_opcode_abs,
        jump_opcode_double_deref,
        jump_opcode_rel,
        jump_opcode_rel_imm,
        mul_opcode,
        mul_opcode_small,
        qm_31_add_mul_opcode,
        ret_opcode,
        verify_instruction,
        blake_round,
        blake_g,
        blake_round_sigma,
        triple_xor_32,
        verify_bitwise_xor_12,
        add_mod_builtin,
        bitwise_builtin,
        mul_mod_builtin,
        pedersen_builtin,
        pedersen_builtin_narrow_windows,
        poseidon_builtin,
        range_check96_builtin,
        range_check_builtin,
        ec_op_builtin,
        partial_ec_mul_generic,
        pedersen_aggregator_window_bits_18,
        partial_ec_mul_window_bits_18,
        pedersen_aggregator_window_bits_9,
        partial_ec_mul_window_bits_9,
        pedersen_points_table_window_bits_18,
        pedersen_points_table_window_bits_9,
        poseidon_aggregator,
        poseidon_3_partial_rounds_chain,
        poseidon_full_round_chain,
        cube_252,
        poseidon_round_keys,
        range_check_252_width_27,
        memory_address_to_id,
        memory_id_to_small,
        range_check_6,
        range_check_8,
        range_check_11,
        range_check_12,
        range_check_18,
        range_check_20,
        range_check_4_3,
        range_check_4_4,
        range_check_9_9,
        range_check_7_2_5,
        range_check_3_6_6_3,
        range_check_4_4_4_4,
        range_check_3_3_3_3_3,
        verify_bitwise_xor_4,
        verify_bitwise_xor_7,
        verify_bitwise_xor_8,
        verify_bitwise_xor_9,
    );
    // The multi-component memory_id_to_big family: one Eval type, one semantic
    // hash — emit from the first (deduped by cache key on write anyway).
    if let Some(c) = components.memory_id_to_big.first() {
        covered.insert("memory_id_to_big".to_string(), true);
        emit_component("memory_id_to_big", c, out, max_instrs);
    } else {
        covered
            .entry("memory_id_to_big".to_string())
            .or_insert(false);
    }
}

fn main() -> ExitCode {
    let stwo_root = PathBuf::from(
        arg("--stwo-root")
            .or_else(|| std::env::var("STWO_ROOT").ok())
            .unwrap_or_else(|| "../stwo".to_string()),
    );
    let out_dir = stwo_root.join("crates/backend-cuda-kernels/cuda/generated");
    let check = std::env::args().any(|a| a == "--check");
    let max_instrs = arg("--max-instrs")
        .map(|v| v.parse::<usize>().expect("--max-instrs <N>"))
        .unwrap_or(DEFAULT_AOT_MAX_INSTRS);

    let mut out: Vec<Emitted> = Vec::new();

    // Witness kernels: fixture-independent (straight-line recorded programs).
    for (label, program) in all_lane_recordings() {
        let kernel = aot::witness_kernel_source(&program)
            .unwrap_or_else(|| panic!("witness codegen failed for {label}"));
        out.push(Emitted {
            kind: "witness",
            label: label.to_string(),
            kernel,
        });
    }

    // Constraint kernels: fixture matrix for union coverage.
    let mut covered: BTreeMap<String, bool> = BTreeMap::new();
    run_fixture(
        "test_prove_verify_all_opcode_components",
        PreProcessedTraceVariant::Canonical,
        &mut out,
        &mut covered,
        max_instrs,
    );
    run_fixture(
        "test_prove_verify_all_builtins",
        PreProcessedTraceVariant::Canonical,
        &mut out,
        &mut covered,
        max_instrs,
    );
    run_fixture(
        "test_prove_verify_pedersen_builtin",
        PreProcessedTraceVariant::CanonicalSmall,
        &mut out,
        &mut covered,
        max_instrs,
    );
    let missing: Vec<&String> = covered
        .iter()
        .filter(|(_, &c)| !c)
        .map(|(k, _)| k)
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "kernel_emit: components NOT covered by any fixture (extend the matrix): \
             {missing:?}"
        );
        return ExitCode::FAILURE;
    }

    // Dedup by cache key (split parts share keys across statements/components
    // never; memory_id_to_big multi-instances and repeated fixture components do).
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    let mut manifest: Vec<serde_json::Value> = Vec::new();
    let mut seen: std::collections::BTreeSet<u64> = Default::default();
    for e in &out {
        if !seen.insert(e.kernel.cache_key) {
            continue;
        }
        let file = format!("{}_{}_{:016x}.cu", e.kind, e.label, e.kernel.cache_key);
        files.insert(file.clone(), e.kernel.source.clone());
        manifest.push(serde_json::json!({
            "kind": e.kind,
            "label": e.label,
            "kernel_name": e.kernel.kernel_name,
            "cache_key": format!("{:016x}", e.kernel.cache_key),
            "semantic_hash": format!("{:016x}", e.kernel.semantic_hash),
            "file": file,
        }));
    }
    manifest.sort_by_key(|m| {
        (
            m["kind"].as_str().unwrap().to_string(),
            m["label"].as_str().unwrap().to_string(),
            m["cache_key"].as_str().unwrap().to_string(),
        )
    });
    files.insert(
        "aot_manifest.json".to_string(),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    );

    if check {
        let mut drift = 0;
        for (name, content) in &files {
            let on_disk = std::fs::read_to_string(out_dir.join(name)).unwrap_or_default();
            if on_disk != *content {
                eprintln!("kernel_emit --check: DRIFT {name}");
                drift += 1;
            }
        }
        // Stale extras count as drift too.
        if let Ok(entries) = std::fs::read_dir(&out_dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !files.contains_key(&name) {
                    eprintln!("kernel_emit --check: STALE {name}");
                    drift += 1;
                }
            }
        }
        if drift == 0 {
            println!("kernel_emit --check: OK ({} kernels)", files.len() - 1);
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    } else {
        std::fs::create_dir_all(&out_dir).expect("create generated dir");
        // Remove stale files so deletions propagate.
        if let Ok(entries) = std::fs::read_dir(&out_dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !files.contains_key(&name) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        for (name, content) in &files {
            std::fs::write(out_dir.join(name), content).expect("write generated file");
        }
        println!(
            "kernel_emit: wrote {} kernels + manifest to {}",
            files.len() - 1,
            out_dir.display()
        );
        ExitCode::SUCCESS
    }
}
