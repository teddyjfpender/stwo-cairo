use serde_json::json;
use stwo_backend_cuda::PreparedNumeratorSchedule;
use stwo_cairo_gpu_prover::arena_plan::{QuotientNumeratorSchedule, QuotientNumeratorSourcePolicy};
use stwo_cairo_gpu_prover::direct_composition_retention::DirectCompositionRetentionMode;
use stwo_cairo_gpu_prover::shape_executable::ShapeExecutableMaterialization;
use stwo_cairo_gpu_prover::ResidentSessionTelemetry;
use stwo_cairo_gpu_prover::WorkspaceMaterialization;

pub(crate) fn operational_safety_reserve_bytes(
    cli_value: Option<String>,
) -> Option<core::num::NonZeroUsize> {
    let value =
        cli_value.or_else(|| std::env::var("STWO_GPU_OPERATIONAL_SAFETY_RESERVE_BYTES").ok());
    parse_operational_safety_reserve_bytes(value.as_deref())
        .unwrap_or_else(|error| panic!("{error}"))
}

fn parse_operational_safety_reserve_bytes(
    value: Option<&str>,
) -> Result<Option<core::num::NonZeroUsize>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let bytes = value.parse::<usize>().map_err(|error| {
        format!("--operational-safety-reserve-bytes must be an integer: {error}")
    })?;
    core::num::NonZeroUsize::new(bytes)
        .map(Some)
        .ok_or_else(|| "--operational-safety-reserve-bytes must be greater than zero".to_owned())
}

pub(crate) fn gpu_native_session_context(
    telemetry: Option<&ResidentSessionTelemetry>,
    gpu_native: bool,
) -> serde_json::Value {
    let Some(telemetry) = telemetry.filter(|_| gpu_native) else {
        return json!({
            "gpu_graph_a_setup_gate_passed": null,
            "gpu_resident_backend": null,
            "gpu_dynamic_commitment_leaf_schedule": null,
            "gpu_protocol_key": null,
            "gpu_arena_words": null,
            "gpu_shape_executable_topology_digest": null,
            "gpu_shape_executable_materialization": null,
            "gpu_shape_executable_cache_hits": null,
            "gpu_shape_executable_cache_misses": null,
            "gpu_shape_executable_cache_compilations": null,
            "gpu_shape_executable_cache_topology_key_constructions": null,
            "gpu_shape_executable_cache_replacement_handle_lock_ns": null,
            "gpu_shape_executable_cache_replacement_handle_lock_ops": null,
            "gpu_shape_executable_cache_source_generation_passes": null,
            "gpu_shape_executable_cache_binding_recipe_compilations": null,
            "gpu_shape_executable_cache_capacity_rejections": null,
            "gpu_workspace_materialization": null,
            "gpu_planned_numerator_schedule": null,
            "gpu_prepared_numerator_schedule": null,
            "gpu_prepared_numerator_eligible_groups": null,
            "gpu_prepared_numerator_legacy_groups": null,
            "gpu_policy_kernel_manifest_hash": null,
            "gpu_policy_retained_lde_budget_bytes": null,
            "gpu_policy_commit_mode": null,
            "gpu_policy_direct_composition_retention": null,
            "gpu_policy_numerator_source": null,
            "gpu_policy_interpolation_mode": null,
            "gpu_policy_blake2s_interior_fused": null,
            "gpu_policy_composition_launch_mode": null,
            "gpu_policy_relation_tail_mode": null,
            "gpu_policy_fri_fold_launch_mode": null,
            "gpu_policy_witness_feed_launch_mode": null,
            "gpu_setup_base_migration_copies": null,
            "gpu_setup_lookup_host_copies": null,
            "gpu_setup_legacy_witness_fallbacks": null,
            "gpu_execution_tables_ingest_compact_h2d_bytes": null,
            "gpu_execution_tables_ingest_compact_h2d_copies": null,
            "gpu_execution_tables_ingest_descriptor_h2d_bytes": null,
            "gpu_execution_tables_ingest_descriptor_h2d_copies": null,
            "gpu_execution_tables_ingest_syncs": null,
            "gpu_witness_ingest_components": null,
            "gpu_witness_ingest_h2d_bytes": null,
            "gpu_witness_ingest_h2d_copies": null,
            "gpu_witness_ingest_syncs": null,
            "gpu_transcript_segments": null,
            "gpu_physical_allocator_pool_checkpoint": null,
            "gpu_physical_memory_checkpoint_error": null,
            "gpu_physical_memory_rows": null,
            "gpu_physical_missing_allocation_ids": null,
            "gpu_physical_inputs_complete": null,
        });
    };
    resident_session_telemetry_json(telemetry)
}

pub(crate) fn resident_session_telemetry_json(
    telemetry: &ResidentSessionTelemetry,
) -> serde_json::Value {
    let execution_tables = telemetry.execution_tables_ingest;
    let allocator_pool_checkpoint = telemetry.allocator_pool_checkpoint.map(|checkpoint| {
        json!({
            "isolated_used_bytes": checkpoint.isolated_used_bytes(),
            "isolated_reserved_bytes": checkpoint.isolated_reserved_bytes(),
            "isolated_attributed_bytes": checkpoint.isolated_attributed_bytes(),
            "default_used_bytes": checkpoint.default_used_bytes(),
            "default_reserved_bytes": checkpoint.default_reserved_bytes(),
            "default_attributed_bytes": checkpoint.default_attributed_bytes(),
            "net_slack_bytes": checkpoint.net_slack_bytes(),
        })
    });
    let missing_physical_ids = telemetry.physical_memory_inputs.missing_allocation_ids();
    let policy = telemetry.protocol_policy;
    let host = telemetry.host_preparation;
    let host_cache = host.and_then(|value| value.replacement_host_cache);
    let topology_digest = telemetry.shape_executable_topology_digest.map(|digest| {
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    });
    let (prepared_schedule, eligible_groups, legacy_groups) =
        match telemetry.prepared_numerator_schedule {
            Some(PreparedNumeratorSchedule::LegacyBatches) => (Some("legacy-batches"), None, None),
            Some(PreparedNumeratorSchedule::SingleWriteCandidate) => {
                (Some("single-write"), None, Some(0))
            }
            Some(PreparedNumeratorSchedule::HybridCandidate {
                eligible_groups,
                legacy_groups,
            }) => (
                Some("hybrid-single-write"),
                Some(eligible_groups),
                Some(legacy_groups),
            ),
            Some(PreparedNumeratorSchedule::StagedPackedSingleWrite { .. }) => {
                (Some("staged-packed-single-write"), None, Some(0))
            }
            None => (None, None, None),
        };
    json!({
        "gpu_graph_a_setup_gate_passed": telemetry.require_strict_graph_a().is_ok(),
        "gpu_host_preparation_total_ns": host.map(|value| value.total_ns),
        "gpu_host_preparation_ingest_ns": host.map(|value| value.ingest_ns),
        "gpu_host_preparation_session_ns": host.map(|value| value.session_ns),
        "gpu_host_claim_generator_constructions": host.map(|value| value.claim_generator_constructions),
        "gpu_host_structural_prover_input_clones": host.map(|value| value.ownership.prover_input_clones),
        "gpu_host_structural_memory_slab_clones": host.map(|value| value.ownership.memory_slab_clones),
        "gpu_host_structural_casm_slab_clones": host.map(|value| value.ownership.casm_slab_clones),
        "gpu_host_structural_execution_memory_arc_clones": host.map(|value| value.ownership.execution_memory_arc_clones),
        "gpu_host_structural_recorded_program_arc_clones": host.map(|value| value.ownership.recorded_program_arc_clones),
        "gpu_host_plan_cache_materialization": host_cache.map(|value| value.materialization.as_str()),
        "gpu_host_plan_cache_identity_ns": host_cache.map(|value| value.identity_ns),
        "gpu_host_plan_cache_select_ns": host_cache.map(|value| value.select_ns),
        "gpu_host_plan_cache_hits": host_cache.map(|value| value.telemetry.hits),
        "gpu_host_plan_cache_misses": host_cache.map(|value| value.telemetry.misses),
        "gpu_host_plan_cache_compilations": host_cache.map(|value| value.telemetry.compilations),
        "gpu_host_plan_cache_evictions": host_cache.map(|value| value.telemetry.evictions),
        "gpu_host_plan_cache_collisions": host_cache.map(|value| value.telemetry.collisions),
        "gpu_resident_backend": policy.map(|value| value.resident_backend.cli_name()),
        "gpu_dynamic_commitment_leaf_schedule": policy.map(|value| value.dynamic_commitment_leaf_schedule.cli_name()),
        "gpu_protocol_key": telemetry.workspace_key.map(|value| value.protocol_key),
        "gpu_arena_words": telemetry.arena_words,
        "gpu_shape_executable_topology_digest": topology_digest,
        "gpu_shape_executable_materialization": telemetry.shape_executable_materialization.map(shape_materialization_name),
        "gpu_shape_executable_cache_hits": telemetry.shape_executable_cache.hits,
        "gpu_shape_executable_cache_misses": telemetry.shape_executable_cache.misses,
        "gpu_shape_executable_cache_compilations": telemetry.shape_executable_cache.compilations,
        "gpu_shape_executable_cache_topology_key_constructions": telemetry.shape_executable_cache.topology_key_constructions,
        "gpu_shape_executable_cache_replacement_handle_lock_ns": telemetry.shape_executable_cache.replacement_handle_lock_ns,
        "gpu_shape_executable_cache_replacement_handle_lock_ops": telemetry.shape_executable_cache.replacement_handle_lock_ops,
        "gpu_shape_executable_cache_source_generation_passes": telemetry.shape_executable_cache.source_generation_passes,
        "gpu_shape_executable_cache_binding_recipe_compilations": telemetry.shape_executable_cache.binding_recipe_compilations,
        "gpu_shape_executable_cache_capacity_rejections": telemetry.shape_executable_cache.capacity_rejections,
        "gpu_workspace_materialization": telemetry.workspace_materialization.map(workspace_materialization_name),
        "gpu_planned_numerator_schedule": policy.map(|value| numerator_schedule_name(value.quotient_numerator_schedule)),
        "gpu_prepared_numerator_schedule": prepared_schedule,
        "gpu_prepared_numerator_eligible_groups": eligible_groups,
        "gpu_prepared_numerator_legacy_groups": legacy_groups,
        "gpu_policy_kernel_manifest_hash": policy.map(|value| value.kernel_manifest_hash),
        "gpu_policy_retained_lde_budget_bytes": policy.map(|value| value.retained_lde_budget_bytes),
        "gpu_policy_commit_mode": policy.map(|value| match value.commit_mode {
            stwo_backend_cuda::ProgressiveCommitMode::FullLifting => "full-lifting",
            stwo_backend_cuda::ProgressiveCommitMode::DomainProgressive => "domain-progressive",
        }),
        "gpu_policy_direct_composition_retention": policy.map(|value| match value.direct_composition_retention_mode {
            DirectCompositionRetentionMode::Disabled => "disabled",
            DirectCompositionRetentionMode::ExactNative => "exact-native",
        }),
        "gpu_policy_numerator_source": policy.map(|value| match value.quotient_numerator_source_policy {
            QuotientNumeratorSourcePolicy::CoefficientsOnly => "coefficients-only",
            QuotientNumeratorSourcePolicy::ReuseRetainedEvaluations => "reuse-retained-evaluations",
        }),
        "gpu_policy_interpolation_mode": policy.map(|value| match value.interpolation_mode {
            stwo_backend_cuda::InterpolationLaunchMode::StageWiseCopyThenInPlace => "stage-wise-copy-then-in-place",
            stwo_backend_cuda::InterpolationLaunchMode::StageFusedOutOfPlace => "stage-fused-out-of-place",
        }),
        "gpu_policy_blake2s_interior_fused": policy.map(|value| value.blake2s_interior_fused),
        "gpu_policy_composition_launch_mode": policy.map(|value| match value.composition_launch_mode {
            stwo_cairo_gpu_prover::CompositionLaunchMode::Serial => "serial",
            stwo_cairo_gpu_prover::CompositionLaunchMode::Wide => "wide",
        }),
        "gpu_policy_relation_tail_mode": policy.map(|value| match value.relation_tail_mode {
            stwo_backend_cuda::RelationTailMode::Segmented => "segmented",
            stwo_backend_cuda::RelationTailMode::Scan => "scan",
        }),
        "gpu_policy_fri_fold_launch_mode": policy.map(|value| match value.fri_fold_launch_mode {
            stwo_backend_cuda::FriFoldLaunchMode::PerFold => "per-fold",
            stwo_backend_cuda::FriFoldLaunchMode::FusedTriple => "fused-triple",
        }),
        "gpu_policy_witness_feed_launch_mode": policy.map(|value| match value.witness_feed_launch_mode {
            stwo_backend_cuda::WitnessFeedLaunchMode::GlobalAtomics => "global-atomics",
            stwo_backend_cuda::WitnessFeedLaunchMode::Privatized => "privatized",
        }),
        "gpu_setup_base_migration_copies": telemetry.base.migrated_base_columns,
        "gpu_setup_lookup_host_copies": telemetry.lookups.host_copies,
        "gpu_setup_legacy_witness_fallbacks": telemetry.witness.host_fallbacks.len(),
        "gpu_execution_tables_ingest_compact_h2d_bytes": execution_tables.map(|value| value.compact_h2d_bytes),
        "gpu_execution_tables_ingest_compact_h2d_copies": execution_tables.map(|value| value.compact_h2d_copies),
        "gpu_execution_tables_ingest_descriptor_h2d_bytes": execution_tables.map(|value| value.descriptor_h2d_bytes),
        "gpu_execution_tables_ingest_descriptor_h2d_copies": execution_tables.map(|value| value.descriptor_h2d_copies),
        "gpu_execution_tables_ingest_syncs": execution_tables.map(|value| value.sync_calls),
        "gpu_witness_ingest_components": telemetry.recorded_witness_ingest.components,
        "gpu_witness_ingest_h2d_bytes": telemetry.recorded_witness_ingest.h2d_bytes,
        "gpu_witness_ingest_h2d_copies": telemetry.recorded_witness_ingest.h2d_copies,
        "gpu_witness_ingest_syncs": telemetry.recorded_witness_ingest.sync_calls,
        "gpu_transcript_segments": telemetry.transcript_segments,
        "gpu_physical_allocator_pool_checkpoint": allocator_pool_checkpoint,
        "gpu_physical_memory_checkpoint_error": telemetry.physical_memory_checkpoint_error.as_deref(),
        "gpu_physical_memory_rows": telemetry.physical_memory_inputs.rows_json(),
        "gpu_physical_inputs_complete": missing_physical_ids.is_empty(),
        "gpu_physical_missing_allocation_ids": missing_physical_ids,
    })
}

fn numerator_schedule_name(schedule: QuotientNumeratorSchedule) -> &'static str {
    match schedule {
        QuotientNumeratorSchedule::LegacyBatches => "legacy-batches",
        QuotientNumeratorSchedule::HybridSingleWrite => "hybrid-single-write",
        QuotientNumeratorSchedule::StagedPackedSingleWrite => "staged-packed-single-write",
    }
}

fn shape_materialization_name(value: ShapeExecutableMaterialization) -> &'static str {
    match value {
        ShapeExecutableMaterialization::Compiled => "compiled",
        ShapeExecutableMaterialization::Reused => "reused",
    }
}

fn workspace_materialization_name(value: WorkspaceMaterialization) -> &'static str {
    match value {
        WorkspaceMaterialization::Materialized => "materialized",
        WorkspaceMaterialization::Reused => "reused",
    }
}

#[cfg(test)]
mod tests {
    use stwo_cairo_gpu_prover::memory_ledger::{AllocatorPoolCheckpoint, PhysicalMemoryInputs};
    use stwo_cairo_gpu_prover::protocol_plan::ProtocolPlanPolicy;
    use stwo_cairo_gpu_prover::shape_executable::{
        ShapeExecutableCacheTelemetry, ShapeExecutableMaterialization,
    };
    use stwo_cairo_gpu_prover::workspace_cache::WorkspaceKey;
    use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

    use super::*;

    #[test]
    fn operational_reserve_policy_is_explicit_and_non_zero() {
        assert_eq!(parse_operational_safety_reserve_bytes(None), Ok(None));
        assert_eq!(
            parse_operational_safety_reserve_bytes(Some("4096"))
                .unwrap()
                .map(core::num::NonZeroUsize::get),
            Some(4096)
        );
        assert!(parse_operational_safety_reserve_bytes(Some("0")).is_err());
        assert!(parse_operational_safety_reserve_bytes(Some("1.5")).is_err());
    }

    #[test]
    fn native_pool_checkpoint_and_partial_rows_are_machine_readable() {
        let checkpoint = AllocatorPoolCheckpoint::try_new(100, 128, 96, 20, 32, 16).unwrap();
        let physical_memory_inputs = PhysicalMemoryInputs::default()
            .with_allocator_pool_checkpoint(checkpoint)
            .unwrap()
            .with_operational_safety_reserve(core::num::NonZeroUsize::new(64).unwrap())
            .unwrap();
        let telemetry = ResidentSessionTelemetry {
            allocator_pool_checkpoint: Some(checkpoint),
            physical_memory_inputs,
            ..ResidentSessionTelemetry::default()
        };
        let value = resident_session_telemetry_json(&telemetry);
        assert_eq!(
            value["gpu_physical_allocator_pool_checkpoint"]["net_slack_bytes"],
            48
        );
        assert_eq!(value["gpu_physical_inputs_complete"], false);
        assert_eq!(
            value["gpu_physical_missing_allocation_ids"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(
            value["gpu_physical_memory_rows"].as_array().unwrap().len(),
            6
        );
    }

    #[test]
    fn replacement_policy_and_actual_schedule_are_machine_readable() {
        let telemetry = ResidentSessionTelemetry {
            protocol_policy: Some(ProtocolPlanPolicy::replacement_v1(0x1234, 2048)),
            prepared_numerator_schedule: Some(PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 50_331_088,
            }),
            workspace_key: Some(WorkspaceKey::new(ProofShapeKey(0), 0x5678)),
            arena_words: 123,
            shape_executable_topology_digest: Some([0xab; 32]),
            shape_executable_materialization: Some(ShapeExecutableMaterialization::Compiled),
            shape_executable_cache: ShapeExecutableCacheTelemetry {
                topology_key_constructions: 7,
                replacement_handle_lock_ns: 89,
                replacement_handle_lock_ops: 5,
                ..ShapeExecutableCacheTelemetry::default()
            },
            ..ResidentSessionTelemetry::default()
        };
        let value = resident_session_telemetry_json(&telemetry);
        assert_eq!(value["gpu_resident_backend"], "replacement-v1");
        assert_eq!(
            value["gpu_dynamic_commitment_leaf_schedule"],
            "retained-domain-cooperative"
        );
        assert_eq!(value["gpu_protocol_key"], 0x5678);
        assert_eq!(
            value["gpu_planned_numerator_schedule"],
            "staged-packed-single-write"
        );
        assert_eq!(
            value["gpu_prepared_numerator_schedule"],
            "staged-packed-single-write"
        );
        assert!(value["gpu_prepared_numerator_eligible_groups"].is_null());
        assert_eq!(value["gpu_prepared_numerator_legacy_groups"], 0);
        assert_eq!(value["gpu_policy_commit_mode"], "domain-progressive");
        assert_eq!(
            value["gpu_policy_interpolation_mode"],
            "stage-fused-out-of-place"
        );
        assert_eq!(value["gpu_policy_blake2s_interior_fused"], false);
        assert_eq!(value["gpu_policy_composition_launch_mode"], "serial");
        assert_eq!(value["gpu_policy_relation_tail_mode"], "segmented");
        assert_eq!(value["gpu_policy_fri_fold_launch_mode"], "per-fold");
        assert_eq!(
            value["gpu_policy_witness_feed_launch_mode"],
            "global-atomics"
        );
        assert_eq!(
            value["gpu_shape_executable_topology_digest"],
            "abababababababababababababababababababababababababababababababab"
        );
        assert_eq!(
            value["gpu_shape_executable_cache_topology_key_constructions"],
            7
        );
        assert_eq!(
            value["gpu_shape_executable_cache_replacement_handle_lock_ns"],
            89
        );
        assert_eq!(
            value["gpu_shape_executable_cache_replacement_handle_lock_ops"],
            5
        );

        let null_schema = gpu_native_session_context(None, false);
        for key in [
            "gpu_dynamic_commitment_leaf_schedule",
            "gpu_shape_executable_cache_topology_key_constructions",
            "gpu_shape_executable_cache_replacement_handle_lock_ns",
            "gpu_shape_executable_cache_replacement_handle_lock_ops",
        ] {
            assert!(null_schema[key].is_null(), "{key}");
        }
    }
}
