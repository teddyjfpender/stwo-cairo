//! SN-scale resident-arena preflight — host-only, no CUDA required.
//! gpu-lab-cohesion-review: one CLI owns exact admission planning and its single JSON contract;
//! splitting that contract before Stage 3 stabilizes would obscure field-to-plan reconciliation.
//!
//! De-risks the first strict resident attempt on a large PIE by running the
//! EXACT planning pipeline the resident session runs
//! (`resident_session::plan_resident_preflight`, which mirrors
//! `with_resident_pre_witness_session` up to workspace materialization):
//! ingest -> strict_resident_exact -> strict witness coverage -> planned claim
//! -> recorded witness inputs (require_resolved) -> Graph-A multiplicity plan
//! -> protocol/arena plan. It then prints one JSON record with the component
//! coverage, multiplicity gaps/blockers, arena words/bytes, per-epoch high
//! water, range-view counts, transcript segments, and a fail-closed physical-memory
//! admission verdict against a VRAM budget. Arena-allocation fit remains diagnostic:
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
//!   --resident-backend <legacy-resident|replacement-v1>
//!                            immutable resident generation (default legacy-resident)
//!   --quotient-topology-fixture-output <path>  write the exact versioned numerator topology
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
#[path = "../arena_preflight_cli.rs"]
mod arena_preflight_cli;
#[path = "arena_preflight/commitment_receipt.rs"]
mod arena_preflight_commitment_receipt;
#[path = "arena_preflight/composition_wave_receipt.rs"]
mod arena_preflight_composition_wave_receipt;
#[path = "../arena_preflight_hybrid.rs"]
mod arena_preflight_hybrid;
#[path = "arena_preflight/ntt_lde_receipt.rs"]
mod arena_preflight_ntt_lde_receipt;
#[path = "../arena_preflight_staged.rs"]
mod arena_preflight_staged;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::process::ExitCode;

use arena_preflight_cli::{
    arg, budget_bytes_of, parse_resident_backend, parse_vram_budget_gb, runtime_policy_json,
};
use arena_preflight_commitment_receipt::{
    dynamic_commitment_leaf_program_receipts, validate_dynamic_commitment_leaf_programs,
};
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::arena_plan::{ProofEpoch, ProtocolIdentity, ResidentBackend};
use stwo_cairo_gpu_prover::memory_ledger::{PhysicalMemoryLedger, ARENA_IDLE_DEFINITION};
use stwo_cairo_gpu_prover::phases;
use stwo_cairo_gpu_prover::protocol_plan::ProtocolPlanPolicy;
use stwo_cairo_gpu_prover::resident_session::{
    plan_raw_resident_preflight, plan_resident_preflight_for, ResidentPreflightError,
    ResidentPreflightReport,
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

fn topology_digest_hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn protocol_key_hex(key: u64) -> String {
    format!("{key:016x}")
}

fn validate_selected_policy(
    selected_backend: ResidentBackend,
    policy: ProtocolPlanPolicy,
) -> Result<(), String> {
    if policy.resident_backend != selected_backend {
        return Err(format!(
            "selected resident backend {} planned as {}",
            selected_backend.cli_name(),
            policy.resident_backend.cli_name()
        ));
    }
    if selected_backend == ResidentBackend::ReplacementV1
        && policy
            != ProtocolPlanPolicy::replacement_v1(
                policy.kernel_manifest_hash,
                policy.composition_max_kernel_instrs,
            )
    {
        return Err("replacement-v1 policy drifted from its immutable tuple".to_owned());
    }
    Ok(())
}

fn validate_protocol_identity(
    policy: ProtocolPlanPolicy,
    identity: ProtocolIdentity,
) -> Result<(), String> {
    macro_rules! require_policy_field {
        ($field:ident) => {
            if identity.$field != policy.$field {
                return Err(format!(
                    "arena protocol identity field {} drifted from the selected policy",
                    stringify!($field)
                ));
            }
        };
    }

    require_policy_field!(channel_tag);
    require_policy_field!(kernel_manifest_hash);
    require_policy_field!(decommit_strategy);
    require_policy_field!(interpolation_mode);
    require_policy_field!(blake2s_interior_fused);
    require_policy_field!(composition_launch_mode);
    require_policy_field!(relation_tail_mode);
    require_policy_field!(fri_fold_launch_mode);
    require_policy_field!(witness_feed_launch_mode);
    require_policy_field!(resident_backend);
    require_policy_field!(dynamic_commitment_leaf_schedule);
    require_policy_field!(quotient_numerator_schedule);
    require_policy_field!(quotient_numerator_source_policy);
    require_policy_field!(commit_mode);
    require_policy_field!(direct_composition_retention_mode);
    Ok(())
}

fn validate_preflight_identity(
    selected_backend: ResidentBackend,
    policy: ProtocolPlanPolicy,
    identity: ProtocolIdentity,
    commitment_memory_policies: impl IntoIterator<Item = (u32, u32)>,
) -> Result<(), String> {
    validate_selected_policy(selected_backend, policy)?;
    validate_protocol_identity(policy, identity)?;
    if commitment_memory_policies
        .into_iter()
        .any(|(bottom, tail)| {
            bottom != policy.unretained_bottom_layers || tail != policy.max_fused_tail_levels
        })
    {
        return Err("arena commitment memory policy drifted from the selected policy".to_owned());
    }
    if identity.retained_evaluation_union_bytes > policy.retained_lde_budget_bytes {
        return Err(format!(
            "arena retained-evaluation union {} exceeds selected policy budget {}",
            identity.retained_evaluation_union_bytes, policy.retained_lde_budget_bytes
        ));
    }
    Ok(())
}

/// The arena-planning verdict: full capture-safe coverage, no multiplicity
/// coverage gaps or feed blockers, exact AOT coverage, and arena-allocation fit.
/// Full process admission additionally requires a complete physical ledger.
fn verdict(
    capture_safe_ok: bool,
    coverage_gaps: usize,
    blockers: usize,
    arena_allocation_bytes: usize,
    budget_bytes: usize,
    aot_coverage_ok: bool,
) -> bool {
    capture_safe_ok
        && coverage_gaps == 0
        && blockers == 0
        && arena_allocation_bytes <= budget_bytes
        && aot_coverage_ok
}

fn admission_verdict(
    planning_pass: bool,
    physical_admission_complete: bool,
    physical_admission_pass: bool,
) -> bool {
    planning_pass && physical_admission_complete && physical_admission_pass
}

fn arena_compatibility_aliases_match(arena: &serde_json::Value) -> bool {
    [
        ("total_words", "allocation_words"),
        ("total_bytes", "allocation_bytes"),
        ("total_gib", "allocation_gib"),
        ("logical_buffers", "logical_buffer_count"),
    ]
    .into_iter()
    .all(|(legacy, current)| arena.get(legacy).is_some() && arena.get(legacy) == arena.get(current))
}

fn preflight_fit_alias_matches(record: &serde_json::Value) -> bool {
    record.get("arena_vram_fit").is_some()
        && record.get("arena_vram_fit") == record.get("arena_allocation_vram_fit")
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

fn required_aot_kernels(
    report: &ResidentPreflightReport,
) -> Result<Vec<AotKernelOccurrence>, String> {
    let mut required = required_composition_aot_kernels(
        report.protocol_policy.resident_backend,
        &report.arena.composition().plan,
    )?;
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
    Ok(required)
}

fn required_composition_aot_kernels(
    backend: ResidentBackend,
    plan: &stwo_cairo_gpu_prover::composition_plan::CompositionPlan,
) -> Result<Vec<AotKernelOccurrence>, String> {
    let mut required = Vec::new();
    match backend {
        ResidentBackend::LegacyResident => {
            for component in &plan.components {
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
        }
        ResidentBackend::ReplacementV1 => {
            let canonical =
                stwo_cairo_gpu_prover::composition_wave::CompositionWaveProgram::from_plan(plan)
                    .map_err(|error| {
                        format!("canonical composition wave program failed: {error}")
                    })?;
            if plan.wave_kernels.len() != canonical.waves().len() {
                return Err("replacement composition wave/log ownership is incomplete".to_owned());
            }
            for (wave_index, (wave, canonical_wave)) in
                plan.wave_kernels.iter().zip(canonical.waves()).enumerate()
            {
                let canonical_parts = canonical_wave
                    .part_ordinals
                    .iter()
                    .map(|&ordinal| {
                        let part = &canonical.parts()[ordinal];
                        Ok(stwo_backend_cuda::aot::CompositionWaveKernelPartIdentity {
                            semantic_hash: part.semantic_hash,
                            coefficient_start: u32::try_from(part.coefficient_start)
                                .map_err(|_| "composition wave coefficient start overflow")?,
                            coefficient_end: u32::try_from(part.coefficient_end)
                                .map_err(|_| "composition wave coefficient end overflow")?,
                        })
                    })
                    .collect::<Result<Vec<_>, &str>>()
                    .map_err(str::to_owned)?;
                if wave.evaluation_log_size != canonical_wave.evaluation_log_size
                    || wave.parts != canonical_parts
                {
                    return Err(format!(
                        "composition wave {wave_index} canonical parts drifted"
                    ));
                }
                let identity = stwo_backend_cuda::aot::composition_wave_kernel_identity(
                    wave.evaluation_log_size,
                    &wave.parts,
                )
                .ok_or_else(|| format!("composition wave {wave_index} identity is invalid"))?;
                if identity.part_count != wave.parts.len()
                    || identity.kernel_name != wave.kernel_name
                    || identity.semantic_hash != wave.semantic_hash
                    || identity.cache_key != wave.cache_key
                {
                    return Err(format!("composition wave {wave_index} identity drifted"));
                }
                required.push(AotKernelOccurrence {
                    kind: "constraint",
                    component: format!("composition_wave_log_{}", wave.evaluation_log_size),
                    instance: wave_index,
                    kernel: 0,
                    kernel_name: wave.kernel_name.clone(),
                    semantic_hash: wave.semantic_hash,
                    cache_key: wave.cache_key,
                });
            }
        }
    }
    Ok(required)
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

#[cfg(test)]
mod composition_aot_coverage_tests {
    use stwo::core::fields::m31::BaseField;
    use stwo_backend_cuda::aot::CompositionWaveKernelPartIdentity;
    use stwo_cairo_gpu_prover::composition_plan::{
        CompositionComponentPlan, CompositionKernelPart, CompositionPlan, CompositionWaveKernelPlan,
    };

    use super::*;

    fn plan() -> CompositionPlan {
        let parts = vec![
            CompositionWaveKernelPartIdentity {
                semantic_hash: 13,
                coefficient_start: 0,
                coefficient_end: 1,
            },
            CompositionWaveKernelPartIdentity {
                semantic_hash: 23,
                coefficient_start: 1,
                coefficient_end: 2,
            },
        ];
        let wave = stwo_backend_cuda::aot::composition_wave_kernel_identity(8, &parts).unwrap();
        CompositionPlan {
            max_kernel_instrs: 192,
            total_constraints: 2,
            max_evaluation_log_size: 8,
            components: vec![CompositionComponentPlan {
                component: "component",
                instance: 3,
                trace_locations: Vec::new(),
                preprocessed_column_indices: Vec::new(),
                trace_log_size: 7,
                evaluation_log_size: 8,
                n_constraints: 2,
                random_coefficient_offset: 0,
                denominator_inverses: vec![BaseField::from(1)],
                base_param_values: Vec::new(),
                ext_param_values: Vec::new(),
                ext_param_sources: Vec::new(),
                kernels: vec![
                    CompositionKernelPart {
                        kernel_name: "part_kernel".to_owned(),
                        cache_key: 11,
                        semantic_hash: 13,
                        source: "part_source".to_owned(),
                        rc_base: 0,
                    },
                    CompositionKernelPart {
                        kernel_name: "part_kernel_1".to_owned(),
                        cache_key: 21,
                        semantic_hash: 23,
                        source: "part_source_1".to_owned(),
                        rc_base: 1,
                    },
                ],
            }],
            wave_kernels: vec![CompositionWaveKernelPlan {
                evaluation_log_size: 8,
                parts,
                kernel_name: wave.kernel_name,
                cache_key: wave.cache_key,
                semantic_hash: wave.semantic_hash,
                source: "wave_source".to_owned(),
            }],
        }
    }

    #[test]
    fn selected_backend_requires_only_its_composition_kernel_family() {
        let plan = plan();
        let legacy =
            required_composition_aot_kernels(ResidentBackend::LegacyResident, &plan).unwrap();
        assert_eq!(legacy.len(), 2);
        assert_eq!(legacy[0].kernel_name, "part_kernel");

        let replacement =
            required_composition_aot_kernels(ResidentBackend::ReplacementV1, &plan).unwrap();
        assert_eq!(replacement.len(), 1);
        assert_eq!(replacement[0].component, "composition_wave_log_8");
        assert_eq!(replacement[0].instance, 0);
        assert_eq!(replacement[0].kernel_name, plan.wave_kernels[0].kernel_name);
        assert_eq!(replacement[0].cache_key, plan.wave_kernels[0].cache_key);
    }

    #[test]
    fn missing_or_mutated_selected_wave_fails_manifest_coverage() {
        let plan = plan();
        let required =
            required_composition_aot_kernels(ResidentBackend::ReplacementV1, &plan).unwrap();
        let wave = &plan.wave_kernels[0];
        let mut manifest = BTreeMap::new();
        manifest.insert(
            wave.cache_key,
            AotManifestKernel {
                kind: "constraint".to_owned(),
                kernel_name: wave.kernel_name.clone(),
                semantic_hash: wave.semantic_hash,
            },
        );
        assert!(missing_aot_kernels(&required, &manifest).is_empty());

        manifest.get_mut(&wave.cache_key).unwrap().semantic_hash ^= 1;
        assert_eq!(
            missing_aot_kernels(&required, &manifest)[0].1,
            "identity_mismatch"
        );
        manifest.clear();
        assert_eq!(
            missing_aot_kernels(&required, &manifest)[0].1,
            "missing_key"
        );
    }

    #[test]
    fn empty_or_identity_mutated_wave_fails_before_manifest_lookup() {
        let mut empty = plan();
        empty.wave_kernels.clear();
        assert!(required_composition_aot_kernels(ResidentBackend::ReplacementV1, &empty).is_err());

        let mut drifted = plan();
        drifted.wave_kernels[0].cache_key ^= 1;
        assert!(
            required_composition_aot_kernels(ResidentBackend::ReplacementV1, &drifted).is_err()
        );

        let reseal = |plan: &mut CompositionPlan| {
            let identity = stwo_backend_cuda::aot::composition_wave_kernel_identity(
                plan.wave_kernels[0].evaluation_log_size,
                &plan.wave_kernels[0].parts,
            )
            .unwrap();
            plan.wave_kernels[0].kernel_name = identity.kernel_name;
            plan.wave_kernels[0].cache_key = identity.cache_key;
            plan.wave_kernels[0].semantic_hash = identity.semantic_hash;
        };
        let mut omitted = plan();
        omitted.wave_kernels[0].parts.pop();
        reseal(&mut omitted);
        assert!(
            required_composition_aot_kernels(ResidentBackend::ReplacementV1, &omitted).is_err()
        );

        let mut reordered = plan();
        reordered.wave_kernels[0].parts.swap(0, 1);
        // Reversed proof-global spans are rejected even before resealing; make
        // the mutation self-consistent as a different ordered program.
        reordered.wave_kernels[0].parts[0].coefficient_start = 0;
        reordered.wave_kernels[0].parts[0].coefficient_end = 1;
        reordered.wave_kernels[0].parts[1].coefficient_start = 1;
        reordered.wave_kernels[0].parts[1].coefficient_end = 2;
        reseal(&mut reordered);
        assert!(
            required_composition_aot_kernels(ResidentBackend::ReplacementV1, &reordered).is_err()
        );
    }
}

fn aot_coverage(report: &ResidentPreflightReport, path: &str) -> Result<AotCoverage, String> {
    let (manifest, manifest_blake3) = load_aot_manifest(path)?;
    let required = required_aot_kernels(report)?;
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

/// Exact producer-to-count-table work retained by the standalone witness-feed
/// pass.  This is deliberately compiler-derived: benchmark reviewers can rank
/// fusion candidates without trying to recover descriptor geometry from CUDA
/// grid sizes in an Nsight trace.
fn recorded_multiplicity_feeds(report: &ResidentPreflightReport) -> Vec<serde_json::Value> {
    report
        .multiplicities
        .feeds
        .iter()
        .map(|feed| {
            let descriptors = feed
                .descriptors
                .chunks_exact(stwo_backend_cuda::WITNESS_FEED_DESCRIPTOR_WORDS)
                .map(|entry| {
                    serde_json::json!({
                        "source_word": entry[0],
                        "tuple_words": entry[1],
                        "tuple_bits": &entry[2..7],
                        "relation_index": entry[7],
                        "table_words": entry[8],
                        "lut_index": entry[9],
                        "destination_index": entry[10],
                        "kind": entry[11],
                        "aux_0": entry[12],
                        "aux_1": entry[13],
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "producer": feed.producer,
                "row_count": feed.row_count,
                "sub_words_per_row": feed.sub_words_per_row,
                "source_words": feed.requirements.source_words,
                "descriptor_count": feed.requirements.descriptor_count,
                "descriptors": descriptors,
                "lut_families": feed.lut_families,
                "destination_components": feed.destination_components,
                "multiplicity_words": feed.requirements.multiplicity_words,
            })
        })
        .collect()
}

fn report_json(
    report: &ResidentPreflightReport,
    selected_backend: ResidentBackend,
    aot_coverage: &AotCoverage,
    composition_wave_current_plan: serde_json::Value,
    ntt_lde_direct_slab_frontier: serde_json::Value,
    compacted_rows: Vec<serde_json::Value>,
    source: &str,
    vram_budget_gb: f64,
) -> serde_json::Value {
    let arena = &report.arena;
    assert_eq!(report.protocol_policy.resident_backend, selected_backend);
    assert_eq!(
        report.protocol_policy.quotient_numerator_schedule,
        arena.quotient_numerator().schedule
    );
    assert_eq!(
        report.protocol_policy.interpolation_mode,
        report.interpolation_mode
    );
    assert!(arena
        .commitments()
        .iter()
        .all(|commitment| commitment.interpolation_mode == report.interpolation_mode));
    let dynamic_commitment_leaf_programs =
        dynamic_commitment_leaf_program_receipts(arena.commitments());
    let bytes_of_words = |words: usize| {
        words
            .checked_mul(WORD_BYTES)
            .expect("arena report byte size overflow")
    };
    let allocation_words = arena.total_words();
    let allocation_bytes = bytes_of_words(allocation_words);
    let live_by_epoch: Vec<serde_json::Value> = ProofEpoch::ALL
        .iter()
        .map(|&epoch| {
            let range_live_words = arena.high_water_words(epoch);
            let logical_words = arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.lifetime.contains(epoch))
                .try_fold(0usize, |total, buffer| total.checked_add(buffer.len_words))
                .expect("logical epoch words overflow");
            assert_eq!(
                range_live_words, logical_words,
                "range-live words must reconcile exactly with logical live words"
            );
            let arena_idle_words = allocation_words
                .checked_sub(range_live_words)
                .expect("range-live words exceed the arena allocation");
            let mut range_live_by_purpose_words = BTreeMap::<String, usize>::new();
            for buffer in arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.lifetime.contains(epoch))
            {
                let words = range_live_by_purpose_words
                    .entry(format!("{:?}", buffer.purpose))
                    .or_default();
                *words = words
                    .checked_add(buffer.len_words)
                    .expect("per-purpose range-live words overflow");
            }
            assert_eq!(
                range_live_by_purpose_words.values().sum::<usize>(),
                range_live_words,
                "per-purpose range-live attribution must partition live words"
            );
            let range_live_by_purpose_bytes = range_live_by_purpose_words
                .into_iter()
                .map(|(purpose, words)| (purpose, bytes_of_words(words)))
                .collect::<BTreeMap<_, _>>();
            serde_json::json!({
                "epoch": format!("{epoch:?}"),
                "logical_live_bytes": bytes_of_words(logical_words),
                "range_live_words": range_live_words,
                "range_live_bytes": bytes_of_words(range_live_words),
                "range_live_reconciles_logical": true,
                "arena_idle_bytes": bytes_of_words(arena_idle_words),
                "range_live_by_purpose_bytes": range_live_by_purpose_bytes,
            })
        })
        .collect();

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
    let recorded_multiplicity_feeds = recorded_multiplicity_feeds(report);

    let numerator_groups = &arena.quotient_numerator().requirements.groups;
    let single_write_groups = numerator_groups
        .iter()
        .enumerate()
        .map(|(group, requirements)| {
            serde_json::json!({
                "group": group,
                "log_size": requirements.log_size,
                "output_rows": requirements.value_words,
                "coefficient_source_count": requirements.coefficient_source_count,
                "eligible": requirements.coefficient_source_count == 0,
            })
        })
        .collect::<Vec<_>>();
    let single_write_eligible_groups = numerator_groups
        .iter()
        .filter(|group| group.coefficient_source_count == 0)
        .count();
    let single_write_eligible_rows = numerator_groups
        .iter()
        .filter(|group| group.coefficient_source_count == 0)
        .map(|group| group.value_words)
        .sum::<usize>();
    let hybrid_traffic = arena_preflight_hybrid::json(arena.quotient_numerator());
    let quotient_combine = arena.quotient().requirements.combine_pass_bytes;

    let budget_bytes = budget_bytes_of(vram_budget_gb);
    let physical_memory =
        PhysicalMemoryLedger::json_with_inputs(arena, budget_bytes, &Default::default())
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
        allocation_bytes,
        budget_bytes,
        aot_coverage.passed(),
    );
    let pass = admission_verdict(
        planning_pass,
        physical_admission_complete,
        physical_admission_pass,
    );

    let record = serde_json::json!({
        "pass": pass,
        "planning_pass": planning_pass,
        "source": source,
        "selected_resident_backend": selected_backend.cli_name(),
        "shape_executable_topology_digest": topology_digest_hex(
            &report.shape_executable_topology_digest
        ),
        "protocol_key": protocol_key_hex(arena.protocol_key),
        "dynamic_commitment_leaf_programs": dynamic_commitment_leaf_programs,
        "present_components": report.present_components.len(),
        "capture_safe_components": report.capture_safe_components.len(),
        "capture_safe_coverage_ok": capture_safe_ok,
        "recorded_witness_lanes": report.recorded_lanes.len(),
        "recorded_multiplicity_feeds": recorded_multiplicity_feeds,
        "aot_coverage": aot_coverage.json(),
        "compacted_consumer_rows": compacted_rows,
        "multiplicity_coverage_gaps": coverage_gaps,
        "multiplicity_feed_blockers": blockers,
        "arena": {
            "allocation_count": 1,
            "allocation_words": allocation_words,
            "allocation_bytes": allocation_bytes,
            "allocation_gib": (allocation_bytes as f64) / GIB,
            "total_words": allocation_words,
            "total_bytes": allocation_bytes,
            "total_gib": (allocation_bytes as f64) / GIB,
            "deprecated_compatibility_aliases": {
                "total_words": "allocation_words",
                "total_bytes": "allocation_bytes",
                "total_gib": "allocation_gib",
                "logical_buffers": "logical_buffer_count",
                "reason": "preserve public arena size/count fields during range-arena migration",
                "removal_condition": "remove after all consumers read arena.allocation_* and logical_buffer_count",
            },
            "whole_slot_comparator_words": arena.whole_slot_total_words(),
            "whole_slot_comparator_bytes": bytes_of_words(arena.whole_slot_total_words()),
            "raw_peak_words": arena.raw_peak_words(),
            "raw_peak_bytes": bytes_of_words(arena.raw_peak_words()),
            "excess_over_raw_peak_words": arena.excess_over_raw_peak_words(),
            "excess_over_raw_peak_bytes": bytes_of_words(arena.excess_over_raw_peak_words()),
            "range_view_count": arena.range_view_count(),
            "aggregate_range_view_words": arena.range_view_words(),
            "aggregate_range_view_bytes": bytes_of_words(arena.range_view_words()),
            "process_owned_pedersen_evaluation_bytes": arena.process_owned_pedersen_evaluation_bytes(),
            "requires_registered_pedersen_table": arena.requires_registered_pedersen_table(),
            "logical_buffer_count": arena.logical_buffers().len(),
            "logical_buffers": arena.logical_buffers().len(),
            "arena_idle_definition": ARENA_IDLE_DEFINITION,
            "live_by_epoch": live_by_epoch,
        },
        "physical_memory": physical_memory,
        "quotient_numerator_single_write": {
            "eligible_groups": single_write_eligible_groups,
            "ineligible_groups": numerator_groups.len() - single_write_eligible_groups,
            "eligible_output_rows": single_write_eligible_rows,
            "groups": single_write_groups,
            "hybrid_traffic_model": hybrid_traffic,
        },
        "quotient_numerator_staged_single_write": arena_preflight_staged::json(arena),
        "composition_wave_current_plan": composition_wave_current_plan,
        "ntt_lde_direct_slab_frontier": ntt_lde_direct_slab_frontier,
        "quotient_combine_pass_byte_model": {
            "rows": quotient_combine.rows,
            "samples": quotient_combine.samples,
            "denominator_inversions": quotient_combine.denominator_inversions,
            "denominator_global_passes": quotient_combine.denominator_global_passes,
            "eliminated_scratch_bytes": quotient_combine.eliminated_scratch_bytes,
            "eliminated_logical_traffic_bytes": quotient_combine.eliminated_logical_traffic_bytes,
            "output_write_bytes": quotient_combine.output_write_bytes,
            "scope": "compiler-derived logical requests; not measured HBM traffic",
        },
        "transcript_segments": report.transcript_segments,
        "manifest_policy": format!("{:?}", report.manifest_policy),
        "runtime_policy": runtime_policy_json(
            report.protocol_policy,
            report.arena.relation().launch_mode,
        ),
        "vram_budget_gib": vram_budget_gb,
        "vram_budget_bytes": budget_bytes,
        "arena_allocation_vram_fit": allocation_bytes <= budget_bytes,
        "arena_vram_fit": allocation_bytes <= budget_bytes,
        "deprecated_compatibility_aliases": {
            "arena_vram_fit": "arena_allocation_vram_fit",
            "reason": "preserve the public preflight JSON contract during range-arena migration",
            "removal_condition": "remove after all external consumers read arena_allocation_vram_fit",
        },
        "vram_fit": physical_admission_complete && physical_admission_pass,
        "caveat": "vram_fit fails closed until the physical ledger includes the CUDA context, \
                   modules, graph metadata, allocator slack, profiling overhead, and safety reserve",
    });
    assert!(
        arena_compatibility_aliases_match(&record["arena"]),
        "deprecated arena aliases drifted from current fields"
    );
    assert!(
        preflight_fit_alias_matches(&record),
        "deprecated arena VRAM-fit alias drifted from allocation field"
    );
    record
}

fn main() -> ExitCode {
    let resident_backend = match parse_resident_backend(std::env::args()) {
        Ok(value) => value,
        Err(error) => return fail("args", error),
    };
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
    let planned = match resident_backend {
        ResidentBackend::LegacyResident => {
            let ingest = phases::ingest::run(input, variant, None);
            let compacted_rows = compacted_consumer_rows(&ingest.proof_plan);
            plan_resident_preflight_for(
                &ingest.generator,
                &ingest.proof_plan,
                &ingest.preprocessed_trace,
                pcs,
                false,
                resident_backend,
            )
            .map(|report| (compacted_rows, report))
        }
        ResidentBackend::ReplacementV1 => {
            let ingest = match phases::ingest::run_replacement(input, variant, None) {
                Ok(ingest) => ingest,
                Err(error) => return fail("replacement_ingest", error.to_string()),
            };
            let compacted_rows = compacted_consumer_rows(&ingest.proof_plan);
            plan_raw_resident_preflight(
                &ingest.input,
                &ingest.proof_plan,
                &ingest.preprocessed_trace,
                pcs,
                false,
            )
            .map(|report| (compacted_rows, report))
        }
    };
    let (compacted_rows, report) = match planned {
        Ok(report) => report,
        Err(ResidentPreflightError::Session(error)) => {
            return fail("resident_session_plan", format!("{error:?}"))
        }
        Err(ResidentPreflightError::Multiplicity(error)) => {
            return fail("graph_a_multiplicity_plan", format!("{error:?}"))
        }
    };

    if let Err(error) = validate_preflight_identity(
        resident_backend,
        report.protocol_policy,
        report.arena.protocol_identity(),
        report.arena.commitments().iter().map(|commitment| {
            (
                commitment.config.unretained_bottom_layers,
                commitment.config.max_fused_tail_levels,
            )
        }),
    ) {
        return fail("protocol_identity", error);
    }
    if let Err(error) =
        validate_dynamic_commitment_leaf_programs(resident_backend, report.arena.commitments())
    {
        return fail("dynamic_commitment_leaf_programs", error);
    }
    let composition_wave_current_plan =
        match arena_preflight_composition_wave_receipt::json(&report.arena.composition().plan) {
            Ok(receipt) => receipt,
            Err(error) => return fail("composition_wave_current_plan", error),
        };
    let ntt_lde_direct_slab_frontier = match resident_backend {
        ResidentBackend::ReplacementV1 => {
            match arena_preflight_ntt_lde_receipt::json(
                &report.arena,
                report.composition_slab_counterfactual.as_ref(),
            ) {
                Ok(receipt) => receipt,
                Err(error) => return fail("ntt_lde_direct_slab_frontier", error),
            }
        }
        ResidentBackend::LegacyResident => serde_json::json!({
            "status": "not-applicable",
            "reason": "direct-slab frontier requires the immutable replacement-v1 ownership policy",
        }),
    };

    let aot_coverage = match aot_coverage(&report, &aot_manifest) {
        Ok(coverage) => coverage,
        Err(error) => return fail("aot_manifest", error),
    };
    if let Err(error) = arena_preflight_hybrid::export_requested(report.arena.quotient_numerator())
    {
        return fail("quotient_topology_fixture", error);
    }

    let record = report_json(
        &report,
        resident_backend,
        &aot_coverage,
        composition_wave_current_plan,
        ntt_lde_direct_slab_frontier,
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
#[path = "../arena_preflight_tests.rs"]
mod tests;
