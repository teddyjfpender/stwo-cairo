//! SN-scale resident-arena preflight — host-only, no CUDA required.
//!
//! De-risks the first strict resident attempt on a large PIE by running the
//! EXACT planning pipeline the resident session runs
//! (`resident_session::plan_resident_preflight`, which mirrors
//! `with_resident_session_from_generator` up to workspace materialization):
//! ingest -> strict_resident_exact -> strict witness coverage -> planned claim
//! -> recorded witness inputs (require_resolved) -> Graph-A multiplicity plan
//! -> protocol/arena plan. It then prints one JSON record with the component
//! coverage, multiplicity gaps/blockers, arena words/bytes, per-epoch high
//! water, slot counts, transcript segments, and a PASS/FAIL verdict against a
//! VRAM budget. Every failure is the exact fail-closed error the H100 session
//! would raise.
//!
//! Usage (exactly one input source; the binary is registered under the
//! `emit-tools` feature, same as kernel_emit):
//!   arena_preflight --input-bincode <path>   adapted ProverInput serialized
//!                                            with bincode (the same format
//!                                            kernel_emit's --input-bincode arm
//!                                            reads and STWO_DUMP_INPUT emits,
//!                                            e.g. /workspace/bench_inputs/SN_PIE_2.adapted.bin)
//!   arena_preflight --fixture <name>         compiled-program fixture under
//!                                            test_data/<name>/compiled.json,
//!                                            run through the VM + adapter
//!                                            in-process (dev_utils run_and_adapt)
//! Options:
//!   --vram-budget-gb <f64>   budget in GiB the arena must fit under (default 79)
//!   --preprocessed <canonical|canonical-without-pedersen>
//!                            preprocessed-trace variant override. Default is
//!                            auto-detected from the adapted input: canonical iff
//!                            the run has a pedersen builtin segment (the witness
//!                            generator fail-closes with "Missing pedersen points"
//!                            otherwise), canonical-without-pedersen when it does
//!                            not (matches the gpu_bench --program path).
//!
//! The PCS configuration is pinned to the secure benchmark configuration
//! (pow_bits=26, FriConfig(0, 1, 70, 3)) — the same "do not change" config in
//! gpu_bench. Exit code 0 iff the verdict is PASS.

use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::arena_plan::ProofEpoch;
use stwo_cairo_gpu_prover::phases;
use stwo_cairo_gpu_prover::resident_session::{
    plan_resident_preflight, ResidentPreflightError, ResidentPreflightReport,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Budget in bytes for a GiB budget flag value.
fn budget_bytes_of(vram_budget_gb: f64) -> usize {
    (vram_budget_gb * GIB) as usize
}

fn parse_vram_budget_gb(value: Option<&str>) -> Result<f64, String> {
    let value = value.unwrap_or("79");
    let budget = value
        .parse::<f64>()
        .map_err(|error| format!("--vram-budget-gb must be an f64: {error}"))?;
    if !budget.is_finite() || budget <= 0.0 {
        return Err("--vram-budget-gb must be finite and greater than zero".to_owned());
    }
    Ok(budget)
}

/// The PASS verdict: full capture-safe coverage, no multiplicity coverage gaps
/// or feed blockers, and the arena fits the VRAM budget.
fn verdict(
    capture_safe_ok: bool,
    coverage_gaps: usize,
    blockers: usize,
    arena_bytes: usize,
    budget_bytes: usize,
) -> bool {
    capture_safe_ok && coverage_gaps == 0 && blockers == 0 && arena_bytes <= budget_bytes
}

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(current) = args.next() {
        if current == name {
            return args.next();
        }
    }
    None
}

fn fail(stage: &str, error: String) -> ExitCode {
    let record = serde_json::json!({
        "pass": false,
        "stage": stage,
        "error": error,
    });
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
    ExitCode::FAILURE
}

fn load_input(
    variant_override: Option<&str>,
) -> Result<(ProverInput, PreProcessedTraceVariant, String), String> {
    let input_path = arg("--input-bincode");
    let fixture = arg("--fixture");
    let (input, source) = match (input_path, fixture) {
        (Some(_), Some(_)) => {
            return Err(
                "--input-bincode and --fixture are mutually exclusive; pass exactly one".to_owned(),
            )
        }
        (None, None) => {
            return Err(
                "provide exactly one of --input-bincode <adapted.bin> or --fixture \
                 <test_data name>"
                    .to_owned(),
            )
        }
        (Some(path), None) => {
            let bytes =
                std::fs::read(&path).map_err(|error| format!("failed to read {path}: {error}"))?;
            let input: ProverInput = bincode::deserialize(&bytes).map_err(|error| {
                format!(
                    "failed to bincode-deserialize {path} as an adapted ProverInput \
                     (expected the STWO_DUMP_INPUT format): {error}"
                )
            })?;
            (input, path)
        }
        (None, Some(name)) => (load_fixture(&name)?, name),
    };
    let variant = match variant_override {
        None => {
            // The pedersen witness generator fail-closes ("Missing pedersen
            // points in the preprocessed trace.") if the run has a pedersen
            // builtin segment but the variant carries no pedersen points, so
            // pick the variant from the adapted input itself.
            if input.builtin_segments.pedersen_builtin.is_some() {
                PreProcessedTraceVariant::Canonical
            } else {
                PreProcessedTraceVariant::CanonicalWithoutPedersen
            }
        }
        Some("canonical") => PreProcessedTraceVariant::Canonical,
        Some("canonical-without-pedersen") => PreProcessedTraceVariant::CanonicalWithoutPedersen,
        Some(other) => {
            return Err(format!(
                "--preprocessed must be canonical or canonical-without-pedersen, got {other}"
            ))
        }
    };
    Ok((input, variant, source))
}

fn load_fixture(name: &str) -> Result<ProverInput, String> {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    let path = get_compiled_cairo_program_path(name);
    if !path.exists() {
        return Err(format!(
            "fixture {name} not found at {} (run scripts/fetch_large_files.sh?)",
            path.display()
        ));
    }
    run_and_adapt(&path, ProgramType::Json, LayoutName::all_cairo_stwo, None)
        .map_err(|error| format!("VM run + adapt failed for fixture {name}: {error:?}"))
}

/// Exact pre-witness rows of the device-compacted consumers, sealed into the
/// ingest plan by the host derivation (`n_real`/`padded` per label). These are
/// the row counts the compact-finalize kernel will enforce on device.
fn compacted_consumer_rows(
    plan: &stwo_cairo_gpu_prover::plan::ProofPlan,
) -> Vec<serde_json::Value> {
    use stwo_cairo_prover::witness::jit_prove_backend::recorded_input_compaction_geometry;
    use stwo_cairo_prover::witness::proof_shape::RowResolution;

    plan.proof_shape()
        .components()
        .iter()
        .filter(|component| {
            component.is_present() && recorded_input_compaction_geometry(component.id).is_some()
        })
        .map(|component| {
            let RowResolution::Resolved(parts) = &component.rows else {
                panic!(
                    "compacted consumer {} is not sealed exact in the ingest plan",
                    component.id
                );
            };
            serde_json::json!({
                "component": component.id,
                "n_real_rows": parts[0].n_real_rows,
                "padded_rows": parts[0].padded_rows,
            })
        })
        .collect()
}

fn report_json(
    report: &ResidentPreflightReport,
    compacted_rows: Vec<serde_json::Value>,
    source: &str,
    vram_budget_gb: f64,
) -> serde_json::Value {
    let arena = &report.arena;
    let total_words = arena.total_words();
    let total_bytes = total_words * WORD_BYTES;
    let mut slot_capacity = BTreeMap::new();
    for binding in arena.bindings() {
        let capacity = slot_capacity.entry(binding.physical).or_insert(0usize);
        *capacity = (*capacity).max(binding.len_words);
    }
    let peak_by_epoch: Vec<serde_json::Value> = ProofEpoch::ALL
        .iter()
        .map(|&epoch| {
            let words = arena.high_water_words(epoch);
            let logical_words = arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.lifetime.contains(epoch))
                .try_fold(0usize, |total, buffer| total.checked_add(buffer.len_words))
                .expect("logical epoch words overflow");
            assert!(
                logical_words <= words,
                "logical epoch occupancy exceeds physical high-water"
            );
            let mut seen = BTreeSet::new();
            let mut by_purpose_words = BTreeMap::<String, usize>::new();
            let mut logical_by_purpose_words = BTreeMap::<String, usize>::new();
            for buffer in arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.lifetime.contains(epoch))
            {
                *logical_by_purpose_words
                    .entry(format!("{:?}", buffer.purpose))
                    .or_default() += buffer.len_words;
                let binding = arena.binding(buffer.id).unwrap();
                if seen.insert(binding.physical) {
                    *by_purpose_words
                        .entry(format!("{:?}", buffer.purpose))
                        .or_default() += slot_capacity[&binding.physical];
                }
            }
            assert_eq!(
                by_purpose_words.values().sum::<usize>(),
                words,
                "per-purpose physical-slot attribution must partition the epoch high-water"
            );
            let by_purpose_bytes = by_purpose_words
                .into_iter()
                .map(|(purpose, words)| (purpose, words * WORD_BYTES))
                .collect::<BTreeMap<_, _>>();
            let logical_by_purpose_bytes = logical_by_purpose_words
                .into_iter()
                .map(|(purpose, words)| (purpose, words * WORD_BYTES))
                .collect::<BTreeMap<_, _>>();
            serde_json::json!({
                "epoch": format!("{epoch:?}"),
                "words": words,
                "bytes": words * WORD_BYTES,
                "logical_live_bytes": logical_words * WORD_BYTES,
                "slot_slack_bytes": (words - logical_words) * WORD_BYTES,
                "by_purpose_bytes": by_purpose_bytes,
                "logical_by_purpose_bytes": logical_by_purpose_bytes,
            })
        })
        .collect();
    let mut physical_slots: Vec<u32> = arena
        .bindings()
        .iter()
        .map(|binding| binding.physical.0)
        .collect();
    physical_slots.sort_unstable();
    physical_slots.dedup();

    let coverage_gaps: Vec<String> = report
        .multiplicities
        .coverage_gaps
        .iter()
        .map(|gap| format!("{gap:?}"))
        .collect();
    let blockers: Vec<String> = report
        .multiplicities
        .blockers
        .iter()
        .map(|blocker| format!("{blocker:?}"))
        .collect();

    let budget_bytes = budget_bytes_of(vram_budget_gb);
    let capture_safe_ok = report.capture_safe_components.len() == report.present_components.len();
    let pass = verdict(
        capture_safe_ok,
        coverage_gaps.len(),
        blockers.len(),
        total_bytes,
        budget_bytes,
    );

    serde_json::json!({
        "pass": pass,
        "source": source,
        "present_components": report.present_components.len(),
        "capture_safe_components": report.capture_safe_components.len(),
        "capture_safe_coverage_ok": capture_safe_ok,
        "recorded_witness_lanes": report.recorded_lanes.len(),
        "compacted_consumer_rows": compacted_rows,
        "multiplicity_coverage_gaps": coverage_gaps,
        "multiplicity_feed_blockers": blockers,
        "arena": {
            "total_words": total_words,
            "total_bytes": total_bytes,
            "total_gib": (total_bytes as f64) / GIB,
            "physical_slots": physical_slots.len(),
            "logical_buffers": arena.logical_buffers().len(),
            "peak_by_epoch": peak_by_epoch,
        },
        "transcript_segments": report.transcript_segments,
        "manifest_policy": format!("{:?}", report.manifest_policy),
        "runtime_policy": {
            "commit_mode": format!("{:?}", report.protocol_policy.commit_mode),
            "direct_composition_retention_mode": format!(
                "{:?}", report.protocol_policy.direct_composition_retention_mode
            ),
            "quotient_numerator_source_policy": format!(
                "{:?}", report.protocol_policy.quotient_numerator_source_policy
            ),
            "retained_lde_budget_bytes": report.protocol_policy.retained_lde_budget_bytes,
            "interpolation_mode": format!("{:?}", report.interpolation_mode),
            "relation_launch_mode": format!("{:?}", report.arena.relation().launch_mode),
        },
        "vram_budget_gib": vram_budget_gb,
        "vram_budget_bytes": budget_bytes,
        "vram_fit": total_bytes <= budget_bytes,
        "caveat": "arena bytes only; excludes the shared twiddle tree, CUDA context, \
                   and allocator overhead",
    })
}

fn main() -> ExitCode {
    let budget_arg = arg("--vram-budget-gb");
    let vram_budget_gb = match parse_vram_budget_gb(budget_arg.as_deref()) {
        Ok(value) => value,
        Err(error) => return fail("args", error),
    };
    let variant_override = arg("--preprocessed");
    let (input, variant, source) = match load_input(variant_override.as_deref()) {
        Ok(loaded) => loaded,
        Err(error) => return fail("load_input", error),
    };

    // The secure benchmark configuration (gpu_bench `prover_params`; do not change).
    let pcs = PcsConfig {
        pow_bits: 26,
        fri_config: FriConfig::new(0, 1, 70, 3),
        lifting_log_size: None,
    };

    let ingest = phases::ingest::run(input, variant, None);
    let compacted_rows = compacted_consumer_rows(&ingest.proof_plan);
    let report = match plan_resident_preflight(
        &ingest.generator,
        &ingest.proof_plan,
        &ingest.preprocessed_trace,
        pcs,
        false,
    ) {
        Ok(report) => report,
        Err(ResidentPreflightError::Session(error)) => {
            return fail("resident_session_plan", format!("{error:?}"))
        }
        Err(ResidentPreflightError::Multiplicity(error)) => {
            return fail("graph_a_multiplicity_plan", format!("{error:?}"))
        }
    };

    let record = report_json(&report, compacted_rows, &source, vram_budget_gb);
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
    if record["pass"].as_bool() == Some(true) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::{budget_bytes_of, parse_vram_budget_gb, verdict, GIB, WORD_BYTES};

    #[test]
    fn budget_bytes_is_gib_scaled() {
        assert_eq!(budget_bytes_of(1.0), 1024 * 1024 * 1024);
        assert_eq!(budget_bytes_of(79.0), 79 * 1024 * 1024 * 1024);
        assert_eq!(budget_bytes_of(0.5), 512 * 1024 * 1024);
    }

    #[test]
    fn vram_budget_requires_a_positive_finite_number() {
        assert_eq!(parse_vram_budget_gb(None), Ok(79.0));
        assert_eq!(parse_vram_budget_gb(Some("76")), Ok(76.0));
        assert_eq!(budget_bytes_of(76.0), 76 * 1024 * 1024 * 1024);
        for invalid in ["0", "-1", "NaN", "inf"] {
            assert!(
                parse_vram_budget_gb(Some(invalid)).is_err(),
                "accepted invalid budget {invalid}"
            );
        }
    }

    #[test]
    fn word_bytes_and_gib_are_the_arena_units() {
        assert_eq!(WORD_BYTES, 4);
        assert_eq!(GIB, (1u64 << 30) as f64);
    }

    #[test]
    fn verdict_requires_every_gate() {
        // All green, at the budget boundary (inclusive).
        assert!(verdict(true, 0, 0, 100, 100));
        // Each individual failure flips the verdict.
        assert!(!verdict(false, 0, 0, 100, 100));
        assert!(!verdict(true, 1, 0, 100, 100));
        assert!(!verdict(true, 0, 1, 100, 100));
        assert!(!verdict(true, 0, 0, 101, 100));
    }
}
