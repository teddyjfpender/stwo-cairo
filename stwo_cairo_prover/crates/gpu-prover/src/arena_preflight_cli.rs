//! Small, host-only CLI and policy-reporting helpers for `arena_preflight`.

use stwo_backend_cuda::{InterpolationLaunchMode, ProgressiveCommitMode, RelationLaunchMode};
use stwo_cairo_gpu_prover::arena_plan::{
    DecommitStrategy, QuotientNumeratorSchedule, QuotientNumeratorSourcePolicy, ResidentBackend,
};
use stwo_cairo_gpu_prover::direct_composition_retention::DirectCompositionRetentionMode;
use stwo_cairo_gpu_prover::protocol_plan::ProtocolPlanPolicy;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(current) = args.next() {
        if current == name {
            return args.next();
        }
    }
    None
}

pub fn budget_bytes_of(vram_budget_gb: f64) -> usize {
    (vram_budget_gb * GIB) as usize
}

pub fn parse_vram_budget_gb(value: Option<&str>) -> Result<f64, String> {
    let value = value.unwrap_or("79");
    let budget = value
        .parse::<f64>()
        .map_err(|error| format!("--vram-budget-gb must be an f64: {error}"))?;
    if !budget.is_finite() || budget <= 0.0 {
        return Err("--vram-budget-gb must be finite and greater than zero".to_owned());
    }
    Ok(budget)
}

pub fn parse_resident_backend<I, S>(args: I) -> Result<ResidentBackend, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let mut selected = None;
    while let Some(argument) = args.next() {
        let argument = argument.as_ref();
        if argument.starts_with("--resident-backend=") {
            return Err(
                "--resident-backend requires a separate value: legacy-resident or replacement-v1"
                    .to_owned(),
            );
        }
        if argument != "--resident-backend" {
            continue;
        }
        if selected.is_some() {
            return Err("--resident-backend may be passed only once".to_owned());
        }
        let value = args
            .next()
            .ok_or_else(|| "--resident-backend requires a value".to_owned())?;
        let value = value.as_ref();
        if value.starts_with("--") {
            return Err("--resident-backend requires a value".to_owned());
        }
        selected = Some(match value {
            "legacy-resident" => ResidentBackend::LegacyResident,
            "replacement-v1" => ResidentBackend::ReplacementV1,
            other => {
                return Err(format!(
                    "--resident-backend must be legacy-resident or replacement-v1, got {other}"
                ))
            }
        });
    }
    Ok(selected.unwrap_or_default())
}

pub fn runtime_policy_json(
    policy: ProtocolPlanPolicy,
    relation_launch_mode: RelationLaunchMode,
) -> serde_json::Value {
    serde_json::json!({
        "resident_backend": policy.resident_backend.cli_name(),
        "quotient_numerator_schedule": numerator_schedule_name(policy.quotient_numerator_schedule),
        "channel_tag": format!("{:016x}", policy.channel_tag),
        "kernel_manifest_hash": format!("{:016x}", policy.kernel_manifest_hash),
        "composition_max_kernel_instrs": policy.composition_max_kernel_instrs,
        "decommit_strategy": decommit_strategy_name(policy.decommit_strategy),
        "retained_lde_budget_bytes": policy.retained_lde_budget_bytes,
        "unretained_bottom_layers": policy.unretained_bottom_layers,
        "max_fused_tail_levels": policy.max_fused_tail_levels,
        "commit_mode": commit_mode_name(policy.commit_mode),
        "direct_composition_retention_mode": direct_retention_name(
            policy.direct_composition_retention_mode
        ),
        "quotient_numerator_source_policy": numerator_source_name(
            policy.quotient_numerator_source_policy
        ),
        "interpolation_mode": interpolation_mode_name(policy.interpolation_mode),
        "relation_launch_mode": relation_launch_mode_name(relation_launch_mode),
    })
}

fn numerator_schedule_name(value: QuotientNumeratorSchedule) -> &'static str {
    match value {
        QuotientNumeratorSchedule::LegacyBatches => "legacy-batches",
        QuotientNumeratorSchedule::HybridSingleWrite => "hybrid-single-write",
    }
}

fn decommit_strategy_name(value: DecommitStrategy) -> &'static str {
    match value {
        DecommitStrategy::RetainAllLde => "retain-all-lde",
        DecommitStrategy::RecomputeQueriedLde => "recompute-queried-lde",
        DecommitStrategy::HybridByGroup => "hybrid-by-group",
    }
}

fn commit_mode_name(value: ProgressiveCommitMode) -> &'static str {
    match value {
        ProgressiveCommitMode::FullLifting => "full-lifting",
        ProgressiveCommitMode::DomainProgressive => "domain-progressive",
    }
}

fn direct_retention_name(value: DirectCompositionRetentionMode) -> &'static str {
    match value {
        DirectCompositionRetentionMode::Disabled => "disabled",
        DirectCompositionRetentionMode::ExactNative => "exact-native",
    }
}

fn numerator_source_name(value: QuotientNumeratorSourcePolicy) -> &'static str {
    match value {
        QuotientNumeratorSourcePolicy::CoefficientsOnly => "coefficients-only",
        QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations => "reuse-retained-evaluations",
    }
}

fn interpolation_mode_name(value: InterpolationLaunchMode) -> &'static str {
    match value {
        InterpolationLaunchMode::StageWiseCopyThenInPlace => "stage-wise-copy-then-in-place",
        InterpolationLaunchMode::StageFusedOutOfPlace => "stage-fused-out-of-place",
    }
}

fn relation_launch_mode_name(value: RelationLaunchMode) -> &'static str {
    match value {
        RelationLaunchMode::ThreeStage => "three-stage",
        RelationLaunchMode::Fused => "fused",
    }
}
