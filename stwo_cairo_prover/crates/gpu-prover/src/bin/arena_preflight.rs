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
//! water, slot counts, transcript segments, and a fail-closed physical-memory
//! admission verdict against a VRAM budget. Arena-only fit remains diagnostic:
//! the process cannot PASS until every non-arena allocation is in the physical
//! ledger. Every planning failure is the exact fail-closed error the H100
//! session would raise.
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
//! Both input forms require `--aot-manifest <path>` so host admission proves
//! every exact constraint and witness launch key is present before pod contact.
//! Options:
//!   --aot-manifest <path>     required generated/aot_manifest.json whose exact
//!                             semantic keys must cover this statement
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
use stwo_cairo_gpu_prover::memory_ledger::PhysicalMemoryLedger;
use stwo_cairo_gpu_prover::phases;
use stwo_cairo_gpu_prover::resident_session::{
    plan_resident_preflight, ResidentPreflightError, ResidentPreflightReport,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AotKernelOccurrence {
    kind: &'static str,
    component: String,
    instance: usize,
    kernel: usize,
    kernel_name: String,
    semantic_hash: u64,
    cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AotManifestKernel {
    kind: String,
    kernel_name: String,
    semantic_hash: u64,
}

struct AotCoverage {
    manifest_path: String,
    manifest_blake3: String,
    manifest_entries: usize,
    required: Vec<AotKernelOccurrence>,
    missing: Vec<(AotKernelOccurrence, &'static str)>,
}

impl AotKernelOccurrence {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind,
            "component": self.component,
            "instance": self.instance,
            "kernel": self.kernel,
            "kernel_name": self.kernel_name,
            "semantic_hash": format!("{:016x}", self.semantic_hash),
            "cache_key": format!("{:016x}", self.cache_key),
        })
    }
}

impl AotCoverage {
    fn passed(&self) -> bool {
        self.missing.is_empty()
    }

    fn json(&self) -> serde_json::Value {
        let unique_keys = self
            .required
            .iter()
            .map(|kernel| kernel.cache_key)
            .collect::<BTreeSet<_>>();
        let required_occurrences = self
            .required
            .iter()
            .map(AotKernelOccurrence::json)
            .collect::<Vec<_>>();
        let unique_key_values = unique_keys
            .iter()
            .map(|key| format!("{key:016x}"))
            .collect::<Vec<_>>();
        serde_json::json!({
            "pass": self.passed(),
            "manifest": self.manifest_path,
            "manifest_blake3": self.manifest_blake3,
            "manifest_entries": self.manifest_entries,
            "required_occurrences_blake3": blake3_hex(&serde_json::to_vec(&required_occurrences).unwrap()),
            "required_unique_keys_blake3": blake3_hex(&serde_json::to_vec(&unique_key_values).unwrap()),
            "required_occurrences": required_occurrences,
            "required_occurrence_count": self.required.len(),
            "required_unique_key_count": unique_keys.len(),
            "missing_occurrences": self.missing.iter().map(|(kernel, reason)| {
                let mut value = kernel.json();
                value.as_object_mut().unwrap().insert(
                    "reason".to_owned(), serde_json::Value::String((*reason).to_owned())
                );
                value
            }).collect::<Vec<_>>(),
            "missing_occurrence_count": self.missing.len(),
        })
    }
}

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

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

/// The arena-planning verdict: full capture-safe coverage, no multiplicity
/// coverage gaps or feed blockers, exact AOT coverage, and arena-only fit.
/// Full process admission additionally requires a complete physical ledger.
fn verdict(
    capture_safe_ok: bool,
    coverage_gaps: usize,
    blockers: usize,
    arena_bytes: usize,
    budget_bytes: usize,
    aot_coverage_ok: bool,
) -> bool {
    capture_safe_ok
        && coverage_gaps == 0
        && blockers == 0
        && arena_bytes <= budget_bytes
        && aot_coverage_ok
}

fn admission_verdict(
    planning_pass: bool,
    physical_admission_complete: bool,
    physical_admission_pass: bool,
) -> bool {
    planning_pass && physical_admission_complete && physical_admission_pass
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

fn lower_hex_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    let value = value
        .as_str()
        .ok_or_else(|| format!("AOT manifest {field} must be a string"))?;
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "AOT manifest {field} must be 16 lowercase hex digits"
        ));
    }
    u64::from_str_radix(value, 16).map_err(|error| format!("invalid AOT manifest {field}: {error}"))
}

fn load_aot_manifest(path: &str) -> Result<(BTreeMap<u64, AotManifestKernel>, String), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read AOT manifest {path}: {error}"))?;
    let manifest_blake3 = blake3_hex(&bytes);
    let entries: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse AOT manifest {path}: {error}"))?;
    let entries = entries
        .as_array()
        .ok_or_else(|| "AOT manifest root must be an array".to_owned())?;
    let mut manifest = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let entry = entry
            .as_object()
            .ok_or_else(|| format!("AOT manifest entry {index} must be an object"))?;
        let string = |field: &str| -> Result<String, String> {
            entry
                .get(field)
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("AOT manifest entry {index} has invalid {field}"))
        };
        let cache_key = lower_hex_u64(
            entry
                .get("cache_key")
                .ok_or_else(|| format!("AOT manifest entry {index} lacks cache_key"))?,
            "cache_key",
        )?;
        string("label")?;
        let kernel = AotManifestKernel {
            kind: string("kind")?,
            kernel_name: string("kernel_name")?,
            semantic_hash: lower_hex_u64(
                entry
                    .get("semantic_hash")
                    .ok_or_else(|| format!("AOT manifest entry {index} lacks semantic_hash"))?,
                "semantic_hash",
            )?,
        };
        if !matches!(kernel.kind.as_str(), "constraint" | "witness") {
            return Err(format!("AOT manifest entry {index} has invalid kind"));
        }
        if manifest.insert(cache_key, kernel).is_some() {
            return Err(format!("AOT manifest repeats cache key {cache_key:016x}"));
        }
    }
    if manifest.is_empty() {
        return Err("AOT manifest must not be empty".to_owned());
    }
    Ok((manifest, manifest_blake3))
}

fn required_aot_kernels(report: &ResidentPreflightReport) -> Vec<AotKernelOccurrence> {
    let mut required = Vec::new();
    for component in &report.arena.composition().plan.components {
        for (kernel, part) in component.kernels.iter().enumerate() {
            required.push(AotKernelOccurrence {
                kind: "constraint",
                component: component.component.to_owned(),
                instance: component.instance,
                kernel,
                kernel_name: part.kernel_name.clone(),
                semantic_hash: part.semantic_hash,
                cache_key: part.cache_key,
            });
        }
    }
    let mut witness_instances = BTreeMap::<&str, usize>::new();
    for component in &report.arena.witness().components {
        let instance = witness_instances.entry(component.component).or_default();
        let semantic_hash = component.program.semantic_hash();
        required.push(AotKernelOccurrence {
            kind: "witness",
            component: component.component.to_owned(),
            instance: *instance,
            kernel: 0,
            kernel_name: stwo_backend_cuda::jit_witness::codegen::witness_kernel_name(
                semantic_hash,
            ),
            semantic_hash,
            cache_key: stwo_backend_cuda::jit_witness::codegen::witness_jit_cache_key(
                semantic_hash,
            ),
        });
        *instance += 1;
    }
    required.sort();
    required
}

fn missing_aot_kernels(
    required: &[AotKernelOccurrence],
    manifest: &BTreeMap<u64, AotManifestKernel>,
) -> Vec<(AotKernelOccurrence, &'static str)> {
    required
        .iter()
        .filter_map(|kernel| match manifest.get(&kernel.cache_key) {
            None => Some((kernel.clone(), "missing_key")),
            Some(entry)
                if entry.kind != kernel.kind
                    || entry.kernel_name != kernel.kernel_name
                    || entry.semantic_hash != kernel.semantic_hash =>
            {
                Some((kernel.clone(), "identity_mismatch"))
            }
            Some(_) => None,
        })
        .collect()
}

fn aot_coverage(report: &ResidentPreflightReport, path: &str) -> Result<AotCoverage, String> {
    let (manifest, manifest_blake3) = load_aot_manifest(path)?;
    let required = required_aot_kernels(report);
    let missing = missing_aot_kernels(&required, &manifest);
    Ok(AotCoverage {
        manifest_path: path.to_owned(),
        manifest_blake3,
        manifest_entries: manifest.len(),
        required,
        missing,
    })
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
    aot_coverage: &AotCoverage,
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
    let physical_memory = PhysicalMemoryLedger::json(arena, budget_bytes)
        .expect("physical memory ledger must reconcile with the validated arena");
    let physical_admission_complete = physical_memory["admission_complete"]
        .as_bool()
        .unwrap_or(false);
    let physical_admission_pass = physical_memory["admission_pass"].as_bool().unwrap_or(false);
    let capture_safe_ok = report.capture_safe_components.len() == report.present_components.len();
    let planning_pass = verdict(
        capture_safe_ok,
        coverage_gaps.len(),
        blockers.len(),
        total_bytes,
        budget_bytes,
        aot_coverage.passed(),
    );
    let pass = admission_verdict(
        planning_pass,
        physical_admission_complete,
        physical_admission_pass,
    );

    serde_json::json!({
        "pass": pass,
        "planning_pass": planning_pass,
        "source": source,
        "present_components": report.present_components.len(),
        "capture_safe_components": report.capture_safe_components.len(),
        "capture_safe_coverage_ok": capture_safe_ok,
        "recorded_witness_lanes": report.recorded_lanes.len(),
        "aot_coverage": aot_coverage.json(),
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
        "physical_memory": physical_memory,
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
        "arena_vram_fit": total_bytes <= budget_bytes,
        "vram_fit": physical_admission_complete && physical_admission_pass,
        "caveat": "vram_fit fails closed until the physical ledger includes the CUDA context, \
                   modules, graph metadata, allocator slack, profiling overhead, and safety reserve",
    })
}

fn main() -> ExitCode {
    let budget_arg = arg("--vram-budget-gb");
    let vram_budget_gb = match parse_vram_budget_gb(budget_arg.as_deref()) {
        Ok(value) => value,
        Err(error) => return fail("args", error),
    };
    let variant_override = arg("--preprocessed");
    let aot_manifest = match arg("--aot-manifest") {
        Some(path) => path,
        None => return fail("args", "--aot-manifest <path> is required".to_owned()),
    };
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

    let aot_coverage = match aot_coverage(&report, &aot_manifest) {
        Ok(coverage) => coverage,
        Err(error) => return fail("aot_manifest", error),
    };

    let record = report_json(
        &report,
        &aot_coverage,
        compacted_rows,
        &source,
        vram_budget_gb,
    );
    println!("{}", serde_json::to_string_pretty(&record).unwrap());
    if record["pass"].as_bool() == Some(true) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
#[path = "arena_preflight_tests.rs"]
mod tests;
