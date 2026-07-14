//! Exact host-side accounting for the disabled hybrid numerator schedule.

use stwo_backend_cuda::quotient_numerator_hybrid_plan;
use stwo_cairo_gpu_prover::arena_plan::PlannedQuotientNumeratorWorkspace;

pub fn json(workspace: &PlannedQuotientNumeratorWorkspace) -> serde_json::Value {
    let topologies = workspace
        .columns
        .iter()
        .map(|column| column.topology.clone())
        .collect::<Vec<_>>();
    let plan = match quotient_numerator_hybrid_plan(workspace.config, &topologies) {
        Ok(plan) => plan,
        Err(error) => {
            return serde_json::json!({
                "eligible": false,
                "error": error.to_string(),
                "scope": "modeled logical output traffic; not HBM or runtime",
            });
        }
    };
    if plan.requirements() != &workspace.requirements {
        return serde_json::json!({
            "eligible": false,
            "error": "hybrid and resident workspace requirements differ",
            "scope": "modeled logical output traffic; not HBM or runtime",
        });
    }

    let report = plan.report();
    let Some(saved) = report
        .legacy_logical_output_bytes
        .checked_sub(report.hybrid_logical_output_bytes)
    else {
        return serde_json::json!({
            "eligible": false,
            "error": "hybrid traffic exceeds its legacy comparator",
            "scope": "modeled logical output traffic; not HBM or runtime",
        });
    };
    serde_json::json!({
        "eligible": true,
        "eligible_groups": report.eligible_group_count,
        "legacy_groups": report.legacy_group_count,
        "eligible_output_rows": report.eligible_output_rows,
        "legacy_output_rows": report.legacy_output_rows,
        "legacy_batches": report.legacy_batch_count,
        "legacy_logical_output_bytes": report.legacy_logical_output_bytes,
        "hybrid_logical_output_bytes": report.hybrid_logical_output_bytes,
        "saved_logical_output_bytes": saved,
        "reduction_fraction": saved as f64 / report.legacy_logical_output_bytes as f64,
        "traffic_reduction_ratio": report.legacy_logical_output_bytes as f64
            / report.hybrid_logical_output_bytes as f64,
        "packed_term_words": plan.packed_terms().len(),
        "packed_group_offset_words": plan.packed_group_offsets().len(),
        "scope": "modeled logical output traffic; not HBM or runtime",
    })
}
