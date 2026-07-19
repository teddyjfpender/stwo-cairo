//! Exact replacement-v1 quotient-staging receipt for host preflight evidence.

use serde_json::{json, Value};
use stwo_backend_cuda::QuotientNumeratorStagingRole;
use stwo_cairo_gpu_prover::arena_plan::{
    ArenaBinding, BufferPurpose, PlannedStagedQuotientOverflow, ProofArenaPlan, ProofEpoch,
    QuotientNumeratorSchedule,
};

pub(super) fn production_constructor(schedule: QuotientNumeratorSchedule) -> &'static str {
    match schedule {
        QuotientNumeratorSchedule::StagedPackedSingleWrite => {
            "prepare_staged_packed_single_write"
        }
        QuotientNumeratorSchedule::StagedRunSumOrPacked => {
            "prepare_staged_group_direct; prepare_staged_packed_single_write only after successful direct preparation without a run-sum receipt; errors fail closed"
        }
        QuotientNumeratorSchedule::LegacyBatches
        | QuotientNumeratorSchedule::HybridSingleWrite => {
            unreachable!("a staged manifest requires a staged numerator schedule")
        }
    }
}

pub(super) fn production_selection(schedule: QuotientNumeratorSchedule) -> Value {
    match schedule {
        QuotientNumeratorSchedule::StagedPackedSingleWrite => json!({
            "initial_constructor": "prepare_staged_packed_single_write",
            "selection_time": "preflight-plan",
            "selected_runtime_constructor": "prepare_staged_packed_single_write",
            "retain_direct_when": null,
            "fallback_constructor": null,
            "fallback_when": null,
            "missing_receipt_policy": "not-applicable",
            "incomplete_receipt_policy": "not-applicable",
            "error_policy": "fail-closed",
        }),
        QuotientNumeratorSchedule::StagedRunSumOrPacked => json!({
            "initial_constructor": "prepare_staged_group_direct",
            "selection_time": "runtime-preparation",
            "selected_runtime_constructor": null,
            "retain_direct_when": "complete-sealed-run-sum-receipt",
            "fallback_constructor": "prepare_staged_packed_single_write",
            "fallback_when": "direct-preparation-succeeded-without-run-sum-receipt",
            "missing_receipt_policy": "prepare-staged-packed-single-write",
            "incomplete_receipt_policy": "fail-closed",
            "error_policy": "fail-closed-no-packed-fallback",
        }),
        QuotientNumeratorSchedule::LegacyBatches | QuotientNumeratorSchedule::HybridSingleWrite => {
            unreachable!("a staged manifest requires a staged numerator schedule")
        }
    }
}

fn binding_json(arena: &ProofArenaPlan, binding: ArenaBinding) -> Value {
    let logical = arena
        .logical_buffers()
        .iter()
        .find(|buffer| buffer.id == binding.logical)
        .expect("validated arena binding must name one logical buffer");
    let slot = arena
        .layout()
        .slot(binding.physical)
        .expect("validated arena binding must name one physical range");
    json!({
        "logical_id": binding.logical.0,
        "physical_id": binding.physical.0,
        "offset_words": slot.offset_words,
        "physical_extent_words": slot.len_words,
        "logical_extent_words": binding.len_words,
        "purpose": format!("{:?}", logical.purpose),
        "ordinal": logical.ordinal,
        "lifetime": {
            "first": format!("{:?}", logical.lifetime.first),
            "last": format!("{:?}", logical.lifetime.last),
        },
    })
}

fn overflow_role_json(
    arena: &ProofArenaPlan,
    role_index: usize,
    role: &PlannedStagedQuotientOverflow,
) -> Value {
    let capacity_words = role.staging.len_words;
    assert_eq!(role.released_slab.physical, role.staging.physical);
    assert_eq!(role.released_slab.len_words, capacity_words);
    assert!(role.used_words <= capacity_words);
    json!({
        "role_index": role_index,
        "commitment": format!("{:?}", role.commitment),
        "capacity_words": capacity_words,
        "used_words": role.used_words,
        "margin_words": capacity_words - role.used_words,
        "released_slab": binding_json(arena, role.released_slab),
        "quotient_staging": binding_json(arena, role.staging),
        "exact_physical_alias": true,
    })
}

pub(super) fn json(arena: &ProofArenaPlan) -> Value {
    let workspace = arena.quotient_numerator();
    let schedule = workspace.schedule;
    let Some(plan) = workspace.staged_single_write.as_ref() else {
        return json!({
            "enabled": false,
            "planned_schedule": schedule.cli_name(),
            "reason": "selected resident backend has no coefficient-inclusive staged manifest",
        });
    };
    let report = plan.report();
    let requirements = plan.requirements();
    let expected_packed_output_rows = requirements
        .groups
        .iter()
        .try_fold(0u64, |total, group| {
            total.checked_add(group.value_words as u64)
        })
        .expect("validated staged row count must fit u64");
    let expected_useful_row_terms = requirements
        .groups
        .iter()
        .enumerate()
        .try_fold(0u64, |total, (group, requirements)| {
            let group_terms =
                u64::from(plan.group_offsets()[group + 1] - plan.group_offsets()[group]);
            total.checked_add(
                (requirements.value_words as u64)
                    .checked_mul(group_terms)
                    .expect("validated staged row-term product must fit u64"),
            )
        })
        .expect("validated staged row-term total must fit u64");
    let packed_output_rows = plan.packed_output_rows();
    assert_eq!(packed_output_rows, expected_packed_output_rows);
    assert_eq!(packed_output_rows, report.output_rows as u64);
    assert_eq!(report.useful_row_terms, expected_useful_row_terms);
    assert_eq!(
        report.rectangular_row_term_capacity,
        (requirements.max_output_size as u64)
            .checked_mul(requirements.term_count as u64)
            .expect("validated rectangular row-term capacity must fit u64")
    );
    assert_ne!(report.rectangular_launch_rows, 0);
    let inactive_rectangular_launch_percent = 100.0
        * report.inactive_rectangular_launch_rows as f64
        / report.rectangular_launch_rows as f64;
    let overflow_role_words = plan.overflow_role_words();
    assert_eq!(overflow_role_words.len(), workspace.staged_overflows.len());
    for (used_words, role) in overflow_role_words.iter().zip(&workspace.staged_overflows) {
        assert_eq!(*used_words, role.used_words);
    }

    let primary_binding = workspace.slots.lde_tile.map(|primary_physical| {
        arena
            .bindings()
            .iter()
            .copied()
            .find(|binding| {
                if binding.physical != primary_physical {
                    return false;
                }
                arena.logical_buffers().iter().any(|buffer| {
                    buffer.id == binding.logical
                        && buffer.purpose == BufferPurpose::QuotientNumeratorLdeTile
                        && buffer.ordinal == 0
                        && buffer.lifetime.first == ProofEpoch::Quotient
                        && buffer.lifetime.last == ProofEpoch::Quotient
                })
            })
            .expect("staged manifest primary role must bind the quotient LDE tile")
    });
    assert_eq!(
        primary_binding.is_some(),
        !plan.coefficient_ldes().is_empty()
    );
    let primary_capacity_words = primary_binding.map_or(0, |binding| binding.len_words);
    assert!(report.primary_staging_words <= primary_capacity_words);

    let role_binding = |role: QuotientNumeratorStagingRole| match role {
        QuotientNumeratorStagingRole::Primary => {
            primary_binding.expect("a primary-staged LDE requires the sealed factor-32 arena role")
        }
        QuotientNumeratorStagingRole::Overflow(index) => {
            workspace.staged_overflows[usize::from(index)].staging
        }
    };
    let staged_ldes = plan
        .coefficient_ldes()
        .iter()
        .copied()
        .map(|lde| {
            let binding = role_binding(lde.staging_role());
            assert!(lde.role_end_words() <= binding.len_words);
            json!({
                "column": lde.column(),
                "evaluation_log_size": lde.evaluation_log_size(),
                "len_words": lde.len_words(),
                "global_offset_words": lde.offset_words(),
                "role": match lde.staging_role() {
                    QuotientNumeratorStagingRole::Primary => "primary".to_owned(),
                    QuotientNumeratorStagingRole::Overflow(index) => format!("overflow_{index}"),
                },
                "role_offset_words": lde.role_offset_words(),
                "role_end_words": lde.role_end_words(),
                "physical_id": binding.physical.0,
                "whole_lde_within_one_role": true,
            })
        })
        .collect::<Vec<_>>();
    let overflow_roles = workspace
        .staged_overflows
        .iter()
        .enumerate()
        .map(|(index, role)| overflow_role_json(arena, index, role))
        .collect::<Vec<_>>();

    json!({
        "enabled": true,
        "planned_schedule": schedule.cli_name(),
        "production_constructor": production_constructor(schedule),
        "production_selection": production_selection(schedule),
        "arena_total_words": arena.total_words(),
        "arena_raw_peak_words": arena.raw_peak_words(),
        "quotient_high_water_words": arena.high_water_words(ProofEpoch::Quotient),
        "group_count": report.group_count,
        "term_count": report.term_count,
        "source_count": report.source_count,
        "coefficient_source_count": report.coefficient_source_count,
        "factor32_batch_count": report.factor32_batch_count,
        "factor32_accumulation_passes": report.factor32_accumulation_passes,
        "factor32_total_output_passes": report.factor32_total_output_passes,
        "candidate_output_passes": report.candidate_output_passes,
        "output_rows": report.output_rows,
        "coefficient_output_rows": report.coefficient_output_rows,
        "factor32_logical_output_bytes": report.factor32_logical_output_bytes,
        "candidate_logical_output_bytes": report.candidate_logical_output_bytes,
        "logical_output_bytes_saved": report.logical_output_bytes_saved,
        "packed_output_rows": packed_output_rows,
        "fallback_packed_geometry_scope": match schedule {
            QuotientNumeratorSchedule::StagedPackedSingleWrite =>
                "selected packed execution model; no adaptive runtime selection",
            QuotientNumeratorSchedule::StagedRunSumOrPacked =>
                "fallback model only; selected runtime execution requires the prepared receipt",
            QuotientNumeratorSchedule::LegacyBatches
            | QuotientNumeratorSchedule::HybridSingleWrite =>
                unreachable!("a staged manifest requires a staged numerator schedule"),
        },
        "fallback_packed_output_passes": report.candidate_output_passes,
        "fallback_packed_output_rows": packed_output_rows,
        "fallback_packed_coefficient_output_rows": report.coefficient_output_rows,
        "fallback_packed_logical_output_bytes": report.candidate_logical_output_bytes,
        "fallback_packed_logical_output_bytes_saved_vs_factor32":
            report.logical_output_bytes_saved,
        "fallback_packed_rectangular_launch_rows": report.rectangular_launch_rows,
        "fallback_packed_inactive_rectangular_launch_rows":
            report.inactive_rectangular_launch_rows,
        "fallback_packed_inactive_rectangular_launch_percent":
            inactive_rectangular_launch_percent,
        "fallback_packed_inactive_rectangular_launch_ratio": {
            "numerator": report.inactive_rectangular_launch_rows,
            "denominator": report.rectangular_launch_rows,
        },
        "fallback_packed_useful_row_terms": report.useful_row_terms,
        "fallback_packed_rectangular_row_term_capacity":
            report.rectangular_row_term_capacity,
        "fallback_packed_rectangular_inactive_rows_return_before_term_loop": true,
        "fallback_packed_rectangular_row_term_capacity_scope":
            "shape upper bound only; inactive rectangular rows do not execute descriptors",
        "fallback_packed_binary_search_comparisons_per_row_max":
            report.packed_binary_search_comparisons_per_row_max,
        "fallback_packed_binary_search_comparisons_max":
            report.packed_binary_search_comparisons_max,
        "deprecated_packed_geometry_aliases_retained": true,
        "rectangular_launch_rows": report.rectangular_launch_rows,
        "inactive_rectangular_launch_rows": report.inactive_rectangular_launch_rows,
        "inactive_rectangular_launch_percent": inactive_rectangular_launch_percent,
        "inactive_rectangular_launch_ratio": {
            "numerator": report.inactive_rectangular_launch_rows,
            "denominator": report.rectangular_launch_rows,
        },
        "useful_row_terms": report.useful_row_terms,
        "rectangular_row_term_capacity": report.rectangular_row_term_capacity,
        "rectangular_inactive_rows_return_before_term_loop": true,
        "rectangular_row_term_capacity_scope":
            "shape upper bound only; inactive rectangular rows do not execute descriptors",
        "packed_binary_search_comparisons_per_row_max":
            report.packed_binary_search_comparisons_per_row_max,
        "packed_binary_search_comparisons_max": report.packed_binary_search_comparisons_max,
        "total_staging_words": report.total_staging_words,
        "factor32_staging_words": report.factor32_staging_words,
        "primary_role": primary_binding.map(|binding| json!({
                "capacity_words": primary_capacity_words,
                "used_words": report.primary_staging_words,
                "margin_words": primary_capacity_words - report.primary_staging_words,
                "binding": binding_json(arena, binding),
            })),
        "overflow_staging_words": report.overflow_staging_words,
        "overflow_staging_role_count": report.overflow_staging_role_count,
        "max_overflow_staging_role_words": report.max_overflow_staging_role_words,
        "incremental_staging_words_over_factor32":
            report.incremental_staging_words_over_factor32,
        "overflow_roles": overflow_roles,
        "staged_ldes": staged_ldes,
    })
}
