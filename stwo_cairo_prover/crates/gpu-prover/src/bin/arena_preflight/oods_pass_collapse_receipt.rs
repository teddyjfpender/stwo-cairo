//! Address-free receipt for the OODS program selected by the exact arena plan.

use serde_json::{json, Value};
use stwo_backend_cuda::{
    OodsPassCollapseCohortRejection, OodsPassCollapseProgram, OodsSourceKind,
    OodsWorkspaceRequirements,
};
use stwo_cairo_gpu_prover::arena_plan::ProofArenaPlan;

pub(super) fn json(arena: &ProofArenaPlan) -> Value {
    let oods = arena.oods();
    let backend = arena.protocol_identity().resident_backend;
    let Some(program) = &oods.pass_collapse else {
        return json!({
            "schema": "stwo.oods-pass-collapse-selection.v1",
            "resident_backend": backend.cli_name(),
            "plan_selected": false,
            "production_constructor": "PreparedOodsGraph::prepare_mixed",
            "observed_by_preflight": {
                "runtime_prepared": false,
                "graph_captured": false,
                "executed": false,
            },
            "claim_scope": "address-free plan selection only; this preflight did not prepare, capture, or execute CUDA",
            "program": Value::Null,
        });
    };

    json!({
        "schema": "stwo.oods-pass-collapse-selection.v1",
        "resident_backend": backend.cli_name(),
        "plan_selected": true,
        "production_constructor": "PreparedOodsGraph::prepare_mixed_pass_collapsed",
        "ordinary_requirements_match_plan": program.ordinary_requirements() == &oods.requirements,
        "observed_by_preflight": {
            "runtime_prepared": false,
            "graph_captured": false,
            "executed": false,
        },
        "claim_scope": "address-free plan selection only; this preflight did not prepare, capture, or execute CUDA",
        "program": program_json(program),
    })
}

fn program_json(program: &OodsPassCollapseProgram) -> Value {
    let identity = program.identity();
    let receipt = program.receipt();
    json!({
        "identity": {
            "config": {
                "lifting_log_size": identity.config.lifting_log_size,
                "mask_log_size": identity.config.mask_log_size,
            },
            "column_ranges": identity.column_ranges.iter().map(|range| json!({
                "source_log_size": range.source_log_size,
                "evaluation_log_size": range.evaluation_log_size,
                "first_sample": range.first_sample,
                "sample_count": range.sample_count,
            })).collect::<Vec<_>>(),
            "coefficient_groups": identity.coefficient_groups.iter().map(|group| json!({
                "log_size": group.log_size,
                "descriptor_offset": group.descriptor_offset,
                "factor_offset_words": group.factor_offset_words,
                "sample_count": group.sample_count,
                "first_pass_blocks": group.first_pass_blocks,
            })).collect::<Vec<_>>(),
            "evaluation_groups": identity.evaluation_groups.iter().map(|group| json!({
                "log_size": group.log_size,
                "offset_point": point_json(group.offset_point.x.0, group.offset_point.y.0),
                "descriptor_offset": group.descriptor_offset,
                "factor_offset_words": group.factor_offset_words,
                "sample_count": group.sample_count,
                "reduction_blocks": group.reduction_blocks,
            })).collect::<Vec<_>>(),
            "canonical_samples": identity.canonical_samples.iter().map(|sample| json!({
                "source_kind": source_kind(sample.source_kind),
                "source_log_size": sample.source_log_size,
                "evaluation_log_size": sample.evaluation_log_size,
                "column_index": sample.column_index,
                "mask_index": sample.mask_index,
                "offset_point": point_json(sample.offset_point.x.0, sample.offset_point.y.0),
                "output_index": sample.output_index,
            })).collect::<Vec<_>>(),
        },
        "ordinary_workspace": workspace_json(program.ordinary_requirements()),
        "collapsed_workspace": workspace_json(program.collapsed_requirements()),
        "groups": receipt.groups.iter().map(|group| json!({
            "log_size": group.log_size,
            "offset_point": point_json(group.offset_point.x.0, group.offset_point.y.0),
            "descriptor_offset": group.descriptor_offset,
            "sample_count": group.sample_count,
            "domain_rows": group.domain_rows,
            "legacy_weight_kernel_launches": group.legacy_weight_kernel_launches,
            "legacy_weight_logical_bytes": group.legacy_weight_logical_bytes,
            "collapsed_weight_logical_bytes": group.collapsed_weight_logical_bytes,
            "logical_bytes_removed": group.logical_bytes_removed,
        })).collect::<Vec<_>>(),
        "same_log_cohorts": receipt.same_log_cohorts.iter().map(|cohort| json!({
            "log_size": cohort.log_size,
            "first_group": cohort.first_group,
            "group_count": cohort.group_count,
            "sample_count": cohort.sample_count,
            "domain_rows": cohort.domain_rows,
            "full_cohort_weight_bytes": cohort.full_cohort_weight_bytes,
            "full_cohort_fusion_admitted": cohort.full_cohort_fusion_admitted,
            "max_groups_per_launch": cohort.max_groups_per_launch,
            "legacy_weight_kernel_launches": cohort.legacy_weight_kernel_launches,
            "collapsed_weight_kernel_launches": cohort.collapsed_weight_kernel_launches,
            "logical_bytes_removed": cohort.logical_bytes_removed,
            "batches": cohort.batches.iter().map(|batch| json!({
                "first_group": batch.first_group,
                "group_count": batch.group_count,
                "log_size": batch.log_size,
                "weight_words": batch.weight_words,
            })).collect::<Vec<_>>(),
            "full_cohort_rejection": cohort.full_cohort_rejection.as_ref().map(rejection_json),
        })).collect::<Vec<_>>(),
        "launches": {
            "unchanged_coefficient_kernel_launches": receipt.unchanged_coefficient_kernel_launches,
            "unchanged_evaluation_kernel_launches": receipt.unchanged_evaluation_kernel_launches,
            "legacy_weight_kernel_launches": receipt.legacy_weight_kernel_launches,
            "collapsed_weight_kernel_launches": receipt.collapsed_weight_kernel_launches,
            "kernel_launches_removed": receipt.kernel_launches_removed,
            "legacy_total_kernel_launches": receipt.legacy_total_kernel_launches,
            "collapsed_total_kernel_launches": receipt.collapsed_total_kernel_launches,
        },
        "traffic": {
            "legacy_weight_logical_bytes": receipt.legacy_weight_logical_bytes,
            "collapsed_weight_logical_bytes": receipt.collapsed_weight_logical_bytes,
            "logical_bytes_removed": receipt.logical_bytes_removed,
            "byte_scope": "compiler-derived logical global-memory requests; not measured HBM traffic",
        },
        "workspace": {
            "legacy_bytes": receipt.legacy_workspace_bytes,
            "collapsed_bytes": receipt.collapsed_workspace_bytes,
            "bytes_removed": receipt.workspace_bytes_removed,
            "retained_weight_bytes": receipt.retained_weight_bytes,
            "scope": "program requirement delta; the arena may retain ordinary-capacity backing slots for cross-mode stability",
        },
        "scale_recomputation": {
            "legacy_evaluations": receipt.legacy_scale_evaluations,
            "collapsed_evaluations": receipt.collapsed_scale_evaluations,
            "additional_evaluations": receipt.additional_scale_evaluations,
            "legacy_secure_squares": receipt.legacy_scale_secure_squares,
            "collapsed_secure_squares": receipt.collapsed_scale_secure_squares,
        },
        "resources": {
            "large_kernel_dynamic_shared_bytes": receipt.large_kernel_dynamic_shared_bytes,
            "cuda_default_dynamic_shared_limit_bytes": receipt.cuda_default_dynamic_shared_limit_bytes,
            "dynamic_shared_admitted": receipt.dynamic_shared_admitted,
            "scope": "program-bound static shared-memory contract; register and spill counts require loaded-artifact runtime attestation",
        },
    })
}

fn workspace_json(requirements: &OodsWorkspaceRequirements) -> Value {
    json!({
        "sample_count": requirements.sample_count,
        "source_pointer_words": requirements.source_pointer_words,
        "offset_point_words": requirements.offset_point_words,
        "fold_count_words": requirements.fold_count_words,
        "output_index_words": requirements.output_index_words,
        "factor_words": requirements.factor_words,
        "scratch_a_words": requirements.scratch_a_words,
        "scratch_b_words": requirements.scratch_b_words,
        "sample_point_words": requirements.sample_point_words,
        "sampled_value_words": requirements.sampled_value_words,
        "evaluation_point_words": requirements.evaluation_point_words,
        "barycentric_numerator_words": requirements.barycentric_numerator_words,
        "barycentric_weight_words": requirements.barycentric_weight_words,
        "barycentric_scale_words": requirements.barycentric_scale_words,
        "barycentric_partial_words": requirements.barycentric_partial_words,
    })
}

fn rejection_json(rejection: &OodsPassCollapseCohortRejection) -> Value {
    match rejection {
        OodsPassCollapseCohortRejection::WorkspaceCapacity {
            required_weight_words,
            available_weight_words,
        } => json!({
            "kind": "workspace-capacity",
            "required_weight_words": required_weight_words,
            "available_weight_words": available_weight_words,
        }),
        OodsPassCollapseCohortRejection::CudaGridY { group_count, limit } => json!({
            "kind": "cuda-grid-y",
            "group_count": group_count,
            "limit": limit,
        }),
    }
}

fn source_kind(kind: OodsSourceKind) -> &'static str {
    match kind {
        OodsSourceKind::Coefficients => "coefficients",
        OodsSourceKind::Evaluations => "evaluations",
    }
}

fn point_json(x: u32, y: u32) -> Value {
    json!({"x": x, "y": y})
}
