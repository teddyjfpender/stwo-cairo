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
//! statement constants into parameters, so structurally identical evaluator
//! recordings emit the same kernels; a component absent from every fixture
//! FAILS the run (loud gap).
//!
//! Usage: kernel_emit [--stwo-root <path>] [--max-instrs N]
//!                    [--max-live-u32-lanes N]
//!                    [--input-bincode <adapted-input>]... [--check]
//!        kernel_emit --witness-only --output-dir <witness-lab-path> [--check]
//! Set `STWO_KERNEL_EMIT_COMPONENT_STATS=1` to print one JSON record per
//! concrete constraint component. This is intentionally diagnostic-only: it
//! exposes the exact row geometry and split count without changing the pack.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::relations::CommonLookupElements;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::channel::Blake2sChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_backend_cuda::aot;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::composition_plan::plan_cairo_composition;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, state};
use stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings;

#[path = "kernel_emit/staged_output.rs"]
mod staged_output;

use staged_output::StagedOutput;

fn arg(name: &str) -> Option<String> {
    args(name).into_iter().next()
}

fn args(name: &str) -> Vec<String> {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .filter(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .collect()
}

fn witness_only_incompatibility(cli_args: &[String]) -> Option<&'static str> {
    [
        "--input-bincode",
        "--shape-report-bincode",
        "--max-instrs",
        "--max-live-u32-lanes",
        "--stwo-root",
    ]
    .into_iter()
    .find(|option| cli_args.iter().any(|arg| arg == option))
}

fn admit_witness_output_dir(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    let entries = std::fs::read_dir(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if entries.is_empty() {
        return Ok(());
    }

    let manifest_path = path.join("aot_manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "non-empty witness output requires {}: {error}",
            manifest_path.display()
        )
    })?;
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest)
        .map_err(|error| format!("decode {}: {error}", manifest_path.display()))?;
    if manifest
        .iter()
        .any(|entry| entry.get("kind").and_then(|kind| kind.as_str()) != Some("witness"))
    {
        return Err(format!(
            "{} is not a witness-only manifest",
            manifest_path.display()
        ));
    }
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name != "aot_manifest.json" && !(name.starts_with("witness_") && name.ends_with(".cu")) {
            return Err(format!(
                "unexpected non-witness artifact {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

/// Per-kernel instruction cap for AOT constraint lowering. MUST MATCH the runtime
/// lowering cap (`DEFAULT_MAX_KERNEL_INSTRS` in stwo's jit/mod.rs = 2048). The
/// lowerer also applies its compiled live-u32-lane cap; both values are sealed into
/// AOT metadata and the pack hash so a stale policy fails runtime admission.
const DEFAULT_AOT_MAX_INSTRS: usize = 2048;

fn stage_kernel(
    output: &mut StagedOutput,
    kind: &str,
    label: String,
    kernel: aot::EmittedKernel,
) -> Result<(), String> {
    output.stage_kernel(
        kind,
        label,
        kernel.kernel_name,
        kernel.cache_key,
        kernel.semantic_hash,
        kernel.source,
    )?;
    Ok(())
}

fn print_shape_report(input_path: &str) -> ExitCode {
    use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
    use stwo_cairo_prover::witness::proof_shape::RowResolution;

    let bytes = std::fs::read(input_path)
        .unwrap_or_else(|error| panic!("read adapted input {input_path}: {error}"));
    let input = bincode::deserialize(&bytes)
        .unwrap_or_else(|error| panic!("decode adapted input {input_path}: {error}"));
    let state::IngestOutput { proof_plan, .. } =
        phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
    let exact = proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .expect("shape report requires an exact strict-resident proof plan");
    let components = exact
        .proof_shape()
        .components()
        .iter()
        .filter_map(|component| match &component.rows {
            RowResolution::Absent => None,
            RowResolution::Resolved(parts) => Some(serde_json::json!({
                "component": component.id,
                "parts": parts.iter().map(|part| serde_json::json!({
                    "part": format!("{:?}", part.part),
                    "n_real_rows": part.n_real_rows,
                    "padded_rows": part.padded_rows,
                    "trace_log_size": part.padded_rows.ilog2(),
                })).collect::<Vec<_>>(),
                "total_padded_rows": parts.iter().map(|part| part.padded_rows).sum::<u64>(),
            })),
            unresolved => panic!(
                "strict-resident component {} remained unresolved: {unresolved:?}",
                component.id
            ),
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": "stwo.kernel-emit.shape-report.v1",
            "input": input_path,
            "components": components,
        }))
        .expect("serialize shape report")
    );
    ExitCode::SUCCESS
}

fn run_fixture(
    program: &str,
    variant: PreProcessedTraceVariant,
    output: &mut StagedOutput,
    covered: &mut BTreeMap<String, bool>,
    max_instrs: usize,
    max_live_u32_lanes: usize,
) -> Result<(), String> {
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

    run_input(
        input,
        variant,
        output,
        covered,
        max_instrs,
        max_live_u32_lanes,
    )
}

fn run_input(
    input: ProverInput,
    variant: PreProcessedTraceVariant,
    output: &mut StagedOutput,
    covered: &mut BTreeMap<String, bool>,
    max_instrs: usize,
    max_live_u32_lanes: usize,
) -> Result<(), String> {
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
    let preprocessed_ids = preprocessed_trace.ids();
    let components = CairoComponents::new(&claim, &elements, &interaction_claim, &preprocessed_ids);

    macro_rules! emit_fields {
        ($( $field:ident ),+ $(,)?) => {
            $(
                if let Some(c) = &components.$field {
                    covered.insert(stringify!($field).to_string(), true);
                    emit_component(
                        stringify!($field),
                        c,
                        output,
                        max_instrs,
                        max_live_u32_lanes,
                    )?;
                } else {
                    covered.entry(stringify!($field).to_string()).or_insert(false);
                }
            )+
        };
    }
    fn emit_component<E: stwo_constraint_framework::FrameworkEval>(
        field: &str,
        component: &stwo_constraint_framework::FrameworkComponent<E>,
        output: &mut StagedOutput,
        max_instrs: usize,
        max_live_u32_lanes: usize,
    ) -> Result<(), String> {
        let kernels = aot::constraint_kernel_sources_with_live_cap(
            component.evaluator(),
            3,
            component.claimed_sum(),
            component.evaluator().log_size(),
            max_instrs,
            max_live_u32_lanes,
        )
        .unwrap_or_else(|| panic!("constraint lowering failed for {field}"));
        if std::env::var_os("STWO_KERNEL_EMIT_COMPONENT_STATS").is_some() {
            eprintln!(
                "kernel_emit-component: {}",
                serde_json::json!({
                    "component": field,
                    "trace_log_size": component.evaluator().log_size(),
                    "evaluation_log_size": component.evaluator().max_constraint_log_degree_bound(),
                    "parts": kernels.len(),
                    "cache_keys": kernels
                        .iter()
                        .map(|kernel| format!("{:016x}", kernel.cache_key))
                        .collect::<Vec<_>>(),
                })
            );
        }
        for k in kernels {
            stage_kernel(output, "constraint", field.to_string(), k)?;
        }
        Ok(())
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
        emit_component(
            "memory_id_to_big",
            c,
            output,
            max_instrs,
            max_live_u32_lanes,
        )?;
    } else {
        covered
            .entry("memory_id_to_big".to_string())
            .or_insert(false);
    }

    // Wave sources must come from the exact canonical composition planner.
    // The manual fixture traversal above is intentionally only a semantic-key
    // coverage set: its order differs and it collapses repeated instances.
    if max_live_u32_lanes == aot::constraint_split_max_live_u32_lanes() {
        let plan = plan_cairo_composition(
            &claim,
            &elements,
            &interaction_claim,
            &preprocessed_ids,
            max_instrs,
        )
        .map_err(|error| format!("canonical composition-wave lowering: {error}"))?;
        for wave in plan.wave_kernels {
            stage_kernel(
                output,
                "constraint",
                format!("wave_log_{}", wave.evaluation_log_size),
                aot::EmittedKernel {
                    kernel_name: wave.kernel_name,
                    cache_key: wave.cache_key,
                    semantic_hash: wave.semantic_hash,
                    source: wave.source,
                },
            )?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli_args = std::env::args().collect::<Vec<_>>();
    let witness_only = cli_args.iter().any(|arg| arg == "--witness-only");
    if witness_only {
        if let Some(option) = witness_only_incompatibility(&cli_args) {
            eprintln!("kernel_emit: --witness-only is incompatible with {option}");
            return ExitCode::FAILURE;
        }
    } else if cli_args.iter().any(|arg| arg == "--output-dir") {
        eprintln!("kernel_emit: --output-dir requires --witness-only");
        return ExitCode::FAILURE;
    }
    if let Some(input_path) = arg("--shape-report-bincode") {
        return print_shape_report(&input_path);
    }
    let out_dir = if witness_only {
        let output_dirs = args("--output-dir");
        if output_dirs.len() != 1 || output_dirs[0].starts_with("--") {
            eprintln!("kernel_emit: --witness-only requires exactly one --output-dir <path>");
            return ExitCode::FAILURE;
        }
        let out_dir = PathBuf::from(&output_dirs[0]);
        if let Err(error) = admit_witness_output_dir(&out_dir) {
            eprintln!("kernel_emit: refusing witness output: {error}");
            return ExitCode::FAILURE;
        }
        out_dir
    } else {
        let stwo_root = PathBuf::from(
            arg("--stwo-root")
                .or_else(|| std::env::var("STWO_ROOT").ok())
                .unwrap_or_else(|| "../stwo".to_string()),
        );
        stwo_root.join("crates/backend-cuda-kernels/cuda/generated")
    };
    let check = std::env::args().any(|a| a == "--check");
    let max_instrs = arg("--max-instrs")
        .map(|v| v.parse::<usize>().expect("--max-instrs <N>"))
        .unwrap_or(DEFAULT_AOT_MAX_INSTRS);
    let max_live_u32_lanes = arg("--max-live-u32-lanes")
        .map(|v| v.parse::<usize>().expect("--max-live-u32-lanes <N>"))
        .unwrap_or_else(aot::constraint_split_max_live_u32_lanes);

    let mut output = match StagedOutput::new(&out_dir) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("kernel_emit: {error}");
            return ExitCode::FAILURE;
        }
    };
    let generation = (|| -> Result<(), String> {
        // Witness kernels are fixture-independent straight-line recordings.
        for (label, program) in all_lane_recordings() {
            let kernel = aot::witness_kernel_source(&program)
                .ok_or_else(|| format!("witness codegen failed for {label}"))?;
            stage_kernel(&mut output, "witness", label.to_string(), kernel)?;
        }

        let mut covered: BTreeMap<String, bool> = BTreeMap::new();
        if witness_only {
            return Ok(());
        }
        // Constraint kernels: fixture matrix for union coverage.
        run_fixture(
            "test_prove_verify_all_opcode_components",
            PreProcessedTraceVariant::Canonical,
            &mut output,
            &mut covered,
            max_instrs,
            max_live_u32_lanes,
        )?;
        run_fixture(
            "test_prove_verify_all_builtins",
            PreProcessedTraceVariant::Canonical,
            &mut output,
            &mut covered,
            max_instrs,
            max_live_u32_lanes,
        )?;
        run_fixture(
            "test_prove_verify_pedersen_builtin",
            PreProcessedTraceVariant::CanonicalSmall,
            &mut output,
            &mut covered,
            max_instrs,
            max_live_u32_lanes,
        )?;
        // This strict parity fixture contributes small-shape variants absent at SN scale.
        run_fixture(
            "test_prove_verify_poseidon_builtin",
            PreProcessedTraceVariant::CanonicalWithoutPedersen,
            &mut output,
            &mut covered,
            max_instrs,
            max_live_u32_lanes,
        )?;
        run_fixture(
            "test_prove_verify_sn2_profile",
            PreProcessedTraceVariant::Canonical,
            &mut output,
            &mut covered,
            max_instrs,
            max_live_u32_lanes,
        )?;
        for input_path in args("--input-bincode") {
            eprintln!("kernel_emit: adapted input {input_path} (Canonical)");
            let bytes = std::fs::read(&input_path)
                .map_err(|error| format!("read adapted input {input_path}: {error}"))?;
            let input = bincode::deserialize(&bytes)
                .map_err(|error| format!("decode adapted input {input_path}: {error}"))?;
            run_input(
                input,
                PreProcessedTraceVariant::Canonical,
                &mut output,
                &mut covered,
                max_instrs,
                max_live_u32_lanes,
            )?;
        }
        let missing = covered
            .iter()
            .filter(|(_, &covered)| !covered)
            .map(|(component, _)| component.as_str())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "components NOT covered by any fixture (extend the matrix): {missing:?}"
            ))
        }
    })();
    if let Err(error) = generation {
        eprintln!("kernel_emit: {error}");
        return ExitCode::FAILURE;
    }

    let mut manifest = output
        .kernels()
        .iter()
        .map(|entry| {
            serde_json::json!({
                "kind": entry.kind,
                "label": entry.label,
                "kernel_name": entry.kernel_name,
                "cache_key": format!("{:016x}", entry.cache_key),
                "semantic_hash": format!("{:016x}", entry.semantic_hash),
                "file": entry.file,
            })
        })
        .collect::<Vec<_>>();
    manifest.sort_by_key(|m| {
        (
            m["kind"].as_str().unwrap().to_string(),
            m["label"].as_str().unwrap().to_string(),
            m["cache_key"].as_str().unwrap().to_string(),
        )
    });
    let metadata = (|| -> Result<(), String> {
        output.stage_metadata(
            "aot_manifest.json",
            serde_json::to_string_pretty(&manifest).unwrap() + "\n",
        )?;
        if !witness_only {
            output.stage_metadata("aot_constraint_max_instrs.txt", format!("{max_instrs}\n"))?;
            output.stage_metadata(
                "aot_constraint_max_live_u32_lanes.txt",
                format!("{max_live_u32_lanes}\n"),
            )?;
        }
        Ok(())
    })();
    if let Err(error) = metadata {
        eprintln!("kernel_emit: {error}");
        return ExitCode::FAILURE;
    }
    let (kernel_count, metadata_count) = output.file_counts();

    if check {
        let diff = match output.diff() {
            Ok(diff) => diff,
            Err(error) => {
                eprintln!("kernel_emit --check: {error}");
                return ExitCode::FAILURE;
            }
        };
        for name in &diff.changed {
            eprintln!("kernel_emit --check: DRIFT {name}");
        }
        for name in &diff.stale {
            eprintln!("kernel_emit --check: STALE {name}");
        }
        if diff.is_clean() {
            println!(
                "kernel_emit --check: OK ({kernel_count} kernels + {metadata_count} metadata)"
            );
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    } else {
        match output.promote_if_changed() {
            Ok(summary) => {
                println!(
                    "kernel_emit: {} changed, {} unchanged, {} stale generated files removed \
                     ({kernel_count} kernels + {metadata_count} metadata) in {}",
                    summary.changed,
                    summary.unchanged,
                    summary.stale,
                    out_dir.display()
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("kernel_emit: output promotion failed: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{admit_witness_output_dir, witness_only_incompatibility};

    #[test]
    fn witness_only_rejects_constraint_and_shape_inputs() {
        for option in [
            "--input-bincode",
            "--shape-report-bincode",
            "--max-instrs",
            "--max-live-u32-lanes",
            "--stwo-root",
        ] {
            let args = vec!["kernel_emit".to_string(), option.to_string()];
            assert_eq!(witness_only_incompatibility(&args), Some(option));
        }
        let compatible =
            ["kernel_emit", "--witness-only", "--check", "--output-dir"].map(str::to_string);
        assert_eq!(witness_only_incompatibility(&compatible), None);
    }

    #[test]
    fn witness_output_rejects_a_production_constraint_manifest() {
        let path = std::env::temp_dir().join(format!(
            "stwo-kernel-emit-witness-admission-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path).unwrap();
        std::fs::write(
            path.join("aot_manifest.json"),
            r#"[{"kind":"constraint","file":"constraint_x.cu"}]"#,
        )
        .unwrap();
        std::fs::write(path.join("constraint_x.cu"), "source").unwrap();
        let error = admit_witness_output_dir(&path).unwrap_err();
        assert!(error.contains("not a witness-only manifest"));
        std::fs::remove_dir_all(path).unwrap();
    }
}
