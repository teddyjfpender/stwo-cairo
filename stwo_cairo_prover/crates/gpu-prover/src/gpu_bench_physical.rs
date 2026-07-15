use serde_json::json;
use stwo_backend_cuda::PreparedNumeratorSchedule;
use stwo_cairo_gpu_prover::arena_plan::{QuotientNumeratorSchedule, QuotientNumeratorSourcePolicy};
use stwo_cairo_gpu_prover::direct_composition_retention::DirectCompositionRetentionMode;
use stwo_cairo_gpu_prover::graphs::GraphSegment;
use stwo_cairo_gpu_prover::shape_executable::ShapeExecutableMaterialization;
use stwo_cairo_gpu_prover::transcript_plan::CairoTranscriptSegment;
use stwo_cairo_gpu_prover::{
    PreparedRuntimeMaterialization, ResidentSessionTelemetry, WorkspaceMaterialization,
};

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
            "gpu_prepared_runtime_materialization": null,
            "gpu_prepared_runtime_capture_ready_at_entry": null,
            "gpu_prepared_runtime_capture_ready_at_exit": null,
            "gpu_statement_refresh_present": null,
            "gpu_statement_refresh_public_memory_seed_h2d_bytes": null,
            "gpu_statement_refresh_public_memory_seed_h2d_copies": null,
            "gpu_statement_refresh_public_memory_seed_syncs": null,
            "gpu_statement_refresh_composition_h2d_bytes": null,
            "gpu_statement_refresh_composition_h2d_copies": null,
            "gpu_statement_refresh_composition_syncs": null,
            "gpu_planned_numerator_schedule": null,
            "gpu_prepared_numerator_schedule": null,
            "gpu_prepared_numerator_eligible_groups": null,
            "gpu_prepared_numerator_legacy_groups": null,
            "gpu_prepared_numerator_packed_output_rows": null,
            "gpu_trace_commit_direct_commitments": null,
            "gpu_trace_commit_separate_interpolation_graph_invocations": null,
            "gpu_trace_commit_separate_interpolation_kernel_launches": null,
            "gpu_trace_commit_terminal_fused_commitments": null,
            "gpu_trace_commit_terminal_materialized_commitments": null,
            "gpu_trace_commit_terminal_fixed16_batches": null,
            "gpu_trace_commit_terminal_materialized_batches": null,
            "gpu_trace_commit_terminal_net_device_bytes_removed": null,
            "gpu_trace_commit_terminal_net_cuda_launches_removed": null,
            "gpu_composition_direct_retained_evaluations": null,
            "gpu_composition_direct_split_graphs": null,
            "gpu_composition_precomputed_compact_commitments": null,
            "gpu_composition_coefficient_commit_paths": null,
            "gpu_composition_part_count": null,
            "gpu_composition_wave_count": null,
            "gpu_composition_replay_launch_mode": null,
            "gpu_composition_replay_wave_launches": null,
            "gpu_composition_split_source_image_bytes": null,
            "gpu_composition_split_retained_image_bytes": null,
            "gpu_composition_split_launch_mode": null,
            "gpu_composition_split_executed_logical_bytes": null,
            "gpu_composition_split_executed_kernel_launches": null,
            "gpu_composition_split_current_logical_bytes": null,
            "gpu_composition_split_terminal_fallback_logical_bytes": null,
            "gpu_composition_split_fused_logical_bytes": null,
            "gpu_composition_split_current_kernel_launches": null,
            "gpu_composition_split_terminal_fallback_kernel_launches": null,
            "gpu_composition_split_fused_kernel_launches": null,
            "gpu_composition_split_current_d2d_nodes": null,
            "gpu_composition_split_fused_d2d_nodes": null,
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
            "gpu_graph_replay_intervals": null,
            "gpu_graph_replay_interval_total_ns": null,
            "gpu_graph_replay_interval_count": null,
            "gpu_graph_replay_semantic_count": null,
            "gpu_graph_replay_timing_scope": null,
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
    let trace_commit_inputs = telemetry.trace_commit_inputs;
    let runtime_materialization = telemetry.prepared_runtime_materialization;
    let statement_refresh = telemetry.statement_refresh;
    let composition_commit = telemetry.composition_commit;
    let composition_traffic = composition_commit.and_then(|value| value.split_traffic);
    let composition_execution = composition_commit.and_then(|value| value.execution_receipt);
    let composition_replay = composition_commit.and_then(|value| value.replay_receipt);
    let graph_timing = telemetry.graph_replay_timing.as_ref();
    let graph_timing_rows = graph_timing.map(|report| {
        report
            .intervals
            .iter()
            .map(|timing| {
                json!({
                    "graph_segment": graph_segment_name(timing.segment),
                    "transcript_segments": timing
                        .transcript_segments
                        .iter()
                        .copied()
                        .map(transcript_segment_name)
                        .collect::<Vec<_>>(),
                    "kernel_nodes": timing.kernel_nodes,
                    "interval_elapsed_ns": timing.elapsed_ns,
                })
            })
            .collect::<Vec<_>>()
    });
    let graph_timing_semantic_count = graph_timing.map(|report| {
        report
            .intervals
            .iter()
            .map(|timing| timing.transcript_segments.len())
            .sum::<usize>()
    });
    let host = telemetry.host_preparation;
    let host_cache = host.and_then(|value| value.replacement_host_cache);
    let topology_digest = telemetry.shape_executable_topology_digest.map(|digest| {
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    });
    let (prepared_schedule, eligible_groups, legacy_groups, packed_output_rows) =
        match telemetry.prepared_numerator_schedule {
            Some(PreparedNumeratorSchedule::LegacyBatches) => {
                (Some("legacy-batches"), None, None, None)
            }
            Some(PreparedNumeratorSchedule::SingleWriteCandidate) => {
                (Some("single-write"), None, Some(0), None)
            }
            Some(PreparedNumeratorSchedule::HybridCandidate {
                eligible_groups,
                legacy_groups,
            }) => (
                Some("hybrid-single-write"),
                Some(eligible_groups),
                Some(legacy_groups),
                None,
            ),
            Some(PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows }) => (
                Some("staged-packed-single-write"),
                None,
                Some(0),
                Some(packed_output_rows),
            ),
            None => (None, None, None, None),
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
        "gpu_prepared_runtime_materialization": runtime_materialization.map(prepared_runtime_materialization_name),
        "gpu_prepared_runtime_capture_ready_at_entry": telemetry.prepared_runtime_capture_ready_at_entry,
        "gpu_prepared_runtime_capture_ready_at_exit": telemetry.prepared_runtime_capture_ready_at_exit,
        "gpu_statement_refresh_present": runtime_materialization.map(|_| statement_refresh.is_some()),
        "gpu_statement_refresh_public_memory_seed_h2d_bytes": statement_refresh.map(|value| value.public_memory_seed_h2d_bytes),
        "gpu_statement_refresh_public_memory_seed_h2d_copies": statement_refresh.map(|value| value.public_memory_seed_h2d_copies),
        "gpu_statement_refresh_public_memory_seed_syncs": statement_refresh.map(|value| value.public_memory_seed_sync_calls),
        "gpu_statement_refresh_composition_h2d_bytes": statement_refresh.map(|value| value.composition.h2d_bytes),
        "gpu_statement_refresh_composition_h2d_copies": statement_refresh.map(|value| value.composition.h2d_copies),
        "gpu_statement_refresh_composition_syncs": statement_refresh.map(|value| value.composition.sync_calls),
        "gpu_planned_numerator_schedule": policy.map(|value| numerator_schedule_name(value.quotient_numerator_schedule)),
        "gpu_prepared_numerator_schedule": prepared_schedule,
        "gpu_prepared_numerator_eligible_groups": eligible_groups,
        "gpu_prepared_numerator_legacy_groups": legacy_groups,
        "gpu_prepared_numerator_packed_output_rows": packed_output_rows,
        "gpu_trace_commit_direct_commitments": trace_commit_inputs.map(|value| value.direct_commitments),
        "gpu_trace_commit_separate_interpolation_graph_invocations": trace_commit_inputs.map(|value| value.separate_interpolation_graph_invocations),
        "gpu_trace_commit_separate_interpolation_kernel_launches": trace_commit_inputs.map(|value| value.separate_interpolation_kernel_launches),
        "gpu_trace_commit_terminal_fused_commitments": trace_commit_inputs.map(|value| value.terminal_fused_commitments),
        "gpu_trace_commit_terminal_materialized_commitments": trace_commit_inputs.map(|value| value.terminal_materialized_commitments),
        "gpu_trace_commit_terminal_fixed16_batches": trace_commit_inputs.map(|value| value.terminal_fixed16_batches),
        "gpu_trace_commit_terminal_materialized_batches": trace_commit_inputs.map(|value| value.terminal_materialized_batches),
        "gpu_trace_commit_terminal_net_device_bytes_removed": trace_commit_inputs.map(|value| value.terminal_net_device_bytes_removed),
        "gpu_trace_commit_terminal_net_cuda_launches_removed": trace_commit_inputs.map(|value| value.terminal_net_cuda_launches_removed),
        "gpu_composition_direct_retained_evaluations": composition_commit.map(|value| value.direct_retained_evaluations),
        "gpu_composition_direct_split_graphs": composition_commit.map(|value| value.direct_split_graphs),
        "gpu_composition_precomputed_compact_commitments": composition_commit.map(|value| value.precomputed_compact_commitments),
        "gpu_composition_coefficient_commit_paths": composition_commit.map(|value| value.coefficient_commit_paths),
        "gpu_composition_part_count": composition_execution.map(|value| value.part_count),
        "gpu_composition_wave_count": composition_execution.map(|value| value.wave_count),
        "gpu_composition_replay_launch_mode": composition_replay.map(|value| composition_launch_mode_name(value.mode)),
        "gpu_composition_replay_wave_launches": composition_replay.map(|value| value.wave_launches),
        "gpu_composition_split_source_image_bytes": composition_traffic.map(|value| value.source_image_bytes),
        "gpu_composition_split_retained_image_bytes": composition_traffic.map(|value| value.retained_image_bytes),
        "gpu_composition_split_launch_mode": composition_commit.and_then(|value| value.split_launch_mode).map(composition_split_launch_mode_name),
        "gpu_composition_split_executed_logical_bytes": composition_commit.and_then(|value| value.split_executed_logical_bytes()),
        "gpu_composition_split_executed_kernel_launches": composition_commit.and_then(|value| value.split_executed_kernel_launches()),
        "gpu_composition_split_current_logical_bytes": composition_traffic.map(|value| value.current_logical_bytes),
        "gpu_composition_split_terminal_fallback_logical_bytes": composition_traffic.map(|value| value.terminal_fallback_logical_bytes),
        "gpu_composition_split_fused_logical_bytes": composition_traffic.map(|value| value.fused_logical_bytes),
        "gpu_composition_split_current_kernel_launches": composition_traffic.map(|value| value.current_kernel_launches),
        "gpu_composition_split_terminal_fallback_kernel_launches": composition_traffic.map(|value| value.terminal_fallback_kernel_launches),
        "gpu_composition_split_fused_kernel_launches": composition_traffic.map(|value| value.fused_kernel_launches),
        "gpu_composition_split_current_d2d_nodes": composition_traffic.map(|value| value.current_d2d_nodes),
        "gpu_composition_split_fused_d2d_nodes": composition_traffic.map(|value| value.fused_d2d_nodes),
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
            stwo_cairo_gpu_prover::CompositionLaunchMode::Wave => "wave",
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
        "gpu_graph_replay_intervals": graph_timing_rows,
        "gpu_graph_replay_interval_total_ns": graph_timing.map(|report| report.total_ns),
        "gpu_graph_replay_interval_count": graph_timing.map(|report| report.intervals.len()),
        "gpu_graph_replay_semantic_count": graph_timing_semantic_count,
        "gpu_graph_replay_timing_scope": graph_timing.map(|_| "main-stream-start-through-final-graph-marker; includes-host-submit-gaps; excludes-proof-bundle-readback"),
        "gpu_physical_allocator_pool_checkpoint": allocator_pool_checkpoint,
        "gpu_physical_memory_checkpoint_error": telemetry.physical_memory_checkpoint_error.as_deref(),
        "gpu_physical_memory_rows": telemetry.physical_memory_inputs.rows_json(),
        "gpu_physical_inputs_complete": missing_physical_ids.is_empty(),
        "gpu_physical_missing_allocation_ids": missing_physical_ids,
    })
}

fn graph_segment_name(segment: GraphSegment) -> String {
    match segment {
        GraphSegment::IngestWitnessBaseCommit => "ingest-witness-base-commit".to_owned(),
        GraphSegment::InteractionCommit => "interaction-commit".to_owned(),
        GraphSegment::CompositionQuotientCommit => "composition-quotient-commit".to_owned(),
        GraphSegment::OodsEvaluation => "oods-evaluation".to_owned(),
        GraphSegment::FriLayer(layer) => format!("fri-layer-{layer}"),
        GraphSegment::OodsQueriesDecommitAssemble => "oods-queries-decommit-assemble".to_owned(),
    }
}

fn transcript_segment_name(segment: CairoTranscriptSegment) -> String {
    match segment {
        CairoTranscriptSegment::BootstrapThroughBase => "bootstrap-through-base".to_owned(),
        CairoTranscriptSegment::InteractionPowAndLookup => "interaction-pow-and-lookup".to_owned(),
        CairoTranscriptSegment::InteractionAndComposition => {
            "interaction-and-composition".to_owned()
        }
        CairoTranscriptSegment::CompositionAndOods => "composition-and-oods".to_owned(),
        CairoTranscriptSegment::OodsAndQuotient => "oods-and-quotient".to_owned(),
        CairoTranscriptSegment::FriLayer(layer) => format!("fri-layer-{layer}"),
        CairoTranscriptSegment::FriLastLayer => "fri-last-layer".to_owned(),
        CairoTranscriptSegment::QueryPowAndPositions => "query-pow-and-positions".to_owned(),
    }
}

fn composition_launch_mode_name(
    mode: stwo_cairo_gpu_prover::CompositionLaunchMode,
) -> &'static str {
    match mode {
        stwo_cairo_gpu_prover::CompositionLaunchMode::Serial => "serial",
        stwo_cairo_gpu_prover::CompositionLaunchMode::Wide => "wide",
        stwo_cairo_gpu_prover::CompositionLaunchMode::Wave => "wave",
    }
}

fn numerator_schedule_name(schedule: QuotientNumeratorSchedule) -> &'static str {
    match schedule {
        QuotientNumeratorSchedule::LegacyBatches => "legacy-batches",
        QuotientNumeratorSchedule::HybridSingleWrite => "hybrid-single-write",
        QuotientNumeratorSchedule::StagedPackedSingleWrite => "staged-packed-single-write",
    }
}

fn composition_split_launch_mode_name(
    mode: stwo_backend_cuda::CompositionSplitLaunchMode,
) -> &'static str {
    match mode {
        stwo_backend_cuda::CompositionSplitLaunchMode::FusedFirstForward => "fused-first-forward",
        stwo_backend_cuda::CompositionSplitLaunchMode::TerminalFallback => "terminal-fallback",
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

fn prepared_runtime_materialization_name(value: PreparedRuntimeMaterialization) -> &'static str {
    match value {
        PreparedRuntimeMaterialization::Materialized => "materialized",
        PreparedRuntimeMaterialization::Reused => "reused",
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
        assert!(value["gpu_trace_commit_direct_commitments"].is_null());
        assert!(value["gpu_trace_commit_separate_interpolation_graph_invocations"].is_null());
        assert!(value["gpu_trace_commit_separate_interpolation_kernel_launches"].is_null());
        assert!(value["gpu_trace_commit_terminal_fused_commitments"].is_null());
        assert!(value["gpu_trace_commit_terminal_net_device_bytes_removed"].is_null());
        assert!(value["gpu_composition_split_launch_mode"].is_null());
        assert!(value["gpu_composition_split_executed_logical_bytes"].is_null());
        assert!(value["gpu_composition_split_executed_kernel_launches"].is_null());
    }

    #[test]
    fn legacy_trace_commit_counts_remain_machine_readable() {
        let telemetry = ResidentSessionTelemetry {
            trace_commit_inputs: Some(
                stwo_cairo_gpu_prover::resident_runtime::ResidentTraceCommitInputTelemetry {
                    direct_commitments: 0,
                    separate_interpolation_graph_invocations: 2,
                    separate_interpolation_kernel_launches: 36,
                    terminal_fused_commitments: 0,
                    terminal_materialized_commitments: 0,
                    terminal_fixed16_batches: 0,
                    terminal_materialized_batches: 0,
                    terminal_net_device_bytes_removed: 0,
                    terminal_net_cuda_launches_removed: 0,
                },
            ),
            ..ResidentSessionTelemetry::default()
        };
        let value = resident_session_telemetry_json(&telemetry);
        assert_eq!(value["gpu_trace_commit_direct_commitments"], 0);
        assert_eq!(
            value["gpu_trace_commit_separate_interpolation_graph_invocations"],
            2
        );
        assert_eq!(
            value["gpu_trace_commit_separate_interpolation_kernel_launches"],
            36
        );
        assert_eq!(value["gpu_trace_commit_terminal_fused_commitments"], 0);
        assert_eq!(
            value["gpu_trace_commit_terminal_materialized_commitments"],
            0
        );
    }

    #[test]
    fn persistent_runtime_and_direct_composition_are_machine_readable() {
        let terminal_traffic = stwo_backend_cuda::CompositionSplitProgram::compile(24)
            .unwrap()
            .traffic();
        let telemetry = ResidentSessionTelemetry {
            workspace_materialization: Some(WorkspaceMaterialization::Reused),
            prepared_runtime_materialization: Some(PreparedRuntimeMaterialization::Reused),
            prepared_runtime_capture_ready_at_entry: Some(true),
            prepared_runtime_capture_ready_at_exit: Some(true),
            statement_refresh: Some(
                stwo_cairo_gpu_prover::resident_runtime::ResidentStatementRefreshTelemetry {
                    public_memory_seed_h2d_bytes: 64,
                    public_memory_seed_h2d_copies: 1,
                    public_memory_seed_sync_calls: 1,
                    composition: stwo_cairo_gpu_prover::prepared_composition::CompositionBindingRefreshTelemetry {
                            h2d_bytes: 32,
                            h2d_copies: 2,
                            sync_calls: 1,
                        },
                },
            ),
            composition_commit: Some(
                stwo_cairo_gpu_prover::resident_runtime::ResidentCompositionCommitTelemetry {
                    direct_retained_evaluations: true,
                    direct_split_graphs: 1,
                    precomputed_compact_commitments: 1,
                    coefficient_commit_paths: 0,
                    split_launch_mode: Some(
                        stwo_backend_cuda::CompositionSplitLaunchMode::TerminalFallback,
                    ),
                    split_traffic: Some(terminal_traffic),
                    execution_receipt: Some(
                        stwo_cairo_gpu_prover::prepared_composition::CompositionExecutionReceipt {
                            part_count: 153,
                            wave_count: 18,
                        },
                    ),
                    replay_receipt: Some(
                        stwo_cairo_gpu_prover::prepared_composition::CompositionReplayReceipt {
                            mode: stwo_cairo_gpu_prover::CompositionLaunchMode::Wave,
                            wave_launches: 18,
                        },
                    ),
                },
            ),
            graph_replay_timing: Some(
                stwo_cairo_gpu_prover::resident_runtime::ResidentGraphReplayTimingReport {
                    intervals: vec![
                        stwo_cairo_gpu_prover::resident_runtime::ResidentGraphReplayIntervalTiming {
                            segment: GraphSegment::IngestWitnessBaseCommit,
                            transcript_segments: vec![
                                CairoTranscriptSegment::BootstrapThroughBase,
                                CairoTranscriptSegment::InteractionPowAndLookup,
                            ],
                            kernel_nodes: 700,
                            elapsed_ns: 11_000_000,
                        },
                        stwo_cairo_gpu_prover::resident_runtime::ResidentGraphReplayIntervalTiming {
                            segment: GraphSegment::FriLayer(8),
                            transcript_segments: vec![],
                            kernel_nodes: 12,
                            elapsed_ns: 3_000_000,
                        },
                    ],
                    total_ns: 14_000_000,
                },
            ),
            ..ResidentSessionTelemetry::default()
        };
        let mut cold = telemetry.clone();
        cold.workspace_materialization = Some(WorkspaceMaterialization::Materialized);
        cold.prepared_runtime_materialization = Some(PreparedRuntimeMaterialization::Materialized);
        cold.prepared_runtime_capture_ready_at_entry = Some(false);
        cold.statement_refresh = None;
        let cold_value = resident_session_telemetry_json(&cold);
        assert_eq!(
            cold_value["gpu_prepared_runtime_capture_ready_at_entry"],
            false
        );
        assert_eq!(
            cold_value["gpu_prepared_runtime_capture_ready_at_exit"],
            true
        );

        let value = resident_session_telemetry_json(&telemetry);
        assert_eq!(value["gpu_prepared_runtime_materialization"], "reused");
        assert_eq!(value["gpu_prepared_runtime_capture_ready_at_entry"], true);
        assert_eq!(value["gpu_prepared_runtime_capture_ready_at_exit"], true);
        assert_eq!(value["gpu_statement_refresh_present"], true);
        assert_eq!(
            value["gpu_statement_refresh_public_memory_seed_h2d_bytes"],
            64
        );
        assert_eq!(value["gpu_statement_refresh_composition_h2d_copies"], 2);
        assert_eq!(value["gpu_composition_direct_split_graphs"], 1);
        assert_eq!(value["gpu_composition_precomputed_compact_commitments"], 1);
        assert_eq!(value["gpu_composition_coefficient_commit_paths"], 0);
        assert_eq!(value["gpu_composition_part_count"], 153);
        assert_eq!(value["gpu_composition_wave_count"], 18);
        assert_eq!(value["gpu_composition_replay_launch_mode"], "wave");
        assert_eq!(value["gpu_composition_replay_wave_launches"], 18);
        assert_eq!(value["gpu_graph_replay_interval_total_ns"], 14_000_000);
        assert_eq!(value["gpu_graph_replay_interval_count"], 2);
        assert_eq!(value["gpu_graph_replay_semantic_count"], 2);
        assert_eq!(
            value["gpu_graph_replay_timing_scope"],
            "main-stream-start-through-final-graph-marker; includes-host-submit-gaps; excludes-proof-bundle-readback"
        );
        assert_eq!(
            value["gpu_graph_replay_intervals"][0]["graph_segment"],
            "ingest-witness-base-commit"
        );
        assert_eq!(
            value["gpu_graph_replay_intervals"][0]["transcript_segments"],
            json!(["bootstrap-through-base", "interaction-pow-and-lookup"])
        );
        assert_eq!(
            value["gpu_graph_replay_intervals"][1]["graph_segment"],
            "fri-layer-8"
        );
        assert_eq!(
            value["gpu_composition_split_launch_mode"],
            "terminal-fallback"
        );
        assert_eq!(
            value["gpu_composition_split_executed_logical_bytes"],
            terminal_traffic.terminal_fallback_logical_bytes
        );
        assert_eq!(
            value["gpu_composition_split_executed_kernel_launches"],
            terminal_traffic.terminal_fallback_kernel_launches
        );
        assert_eq!(
            value["gpu_composition_split_current_logical_bytes"],
            terminal_traffic.current_logical_bytes
        );
        assert_eq!(
            value["gpu_composition_split_terminal_fallback_logical_bytes"],
            terminal_traffic.terminal_fallback_logical_bytes
        );
        assert_eq!(
            value["gpu_composition_split_fused_logical_bytes"],
            terminal_traffic.fused_logical_bytes
        );
        assert_eq!(value["gpu_composition_split_current_d2d_nodes"], 8);
        assert_eq!(value["gpu_composition_split_fused_d2d_nodes"], 0);

        let fused_traffic = stwo_backend_cuda::CompositionSplitProgram::compile(25)
            .unwrap()
            .traffic();
        let mut fused = telemetry;
        let composition = fused.composition_commit.as_mut().unwrap();
        composition.split_launch_mode =
            Some(stwo_backend_cuda::CompositionSplitLaunchMode::FusedFirstForward);
        composition.split_traffic = Some(fused_traffic);
        let fused_value = resident_session_telemetry_json(&fused);
        assert_eq!(
            fused_value["gpu_composition_split_launch_mode"],
            "fused-first-forward"
        );
        assert_eq!(
            fused_value["gpu_composition_split_executed_logical_bytes"],
            fused_traffic.fused_logical_bytes
        );
        assert_eq!(
            fused_value["gpu_composition_split_executed_kernel_launches"],
            fused_traffic.fused_kernel_launches
        );
    }

    #[test]
    fn replacement_policy_and_actual_schedule_are_machine_readable() {
        let telemetry = ResidentSessionTelemetry {
            protocol_policy: Some(ProtocolPlanPolicy::replacement_v1(0x1234, 2048)),
            prepared_numerator_schedule: Some(PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 50_331_088,
            }),
            trace_commit_inputs: Some(
                stwo_cairo_gpu_prover::resident_runtime::ResidentTraceCommitInputTelemetry {
                    direct_commitments: 2,
                    separate_interpolation_graph_invocations: 0,
                    separate_interpolation_kernel_launches: 0,
                    terminal_fused_commitments: 2,
                    terminal_materialized_commitments: 0,
                    terminal_fixed16_batches: 7,
                    terminal_materialized_batches: 3,
                    terminal_net_device_bytes_removed: 1_073_741_824,
                    terminal_net_cuda_launches_removed: 4,
                },
            ),
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
            "retained-domain-compact-h8"
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
        assert_eq!(
            value["gpu_prepared_numerator_packed_output_rows"],
            50_331_088
        );
        assert_eq!(value["gpu_trace_commit_direct_commitments"], 2);
        assert_eq!(
            value["gpu_trace_commit_separate_interpolation_graph_invocations"],
            0
        );
        assert_eq!(
            value["gpu_trace_commit_separate_interpolation_kernel_launches"],
            0
        );
        assert_eq!(value["gpu_trace_commit_terminal_fused_commitments"], 2);
        assert_eq!(value["gpu_trace_commit_terminal_fixed16_batches"], 7);
        assert_eq!(value["gpu_trace_commit_terminal_materialized_batches"], 3);
        assert_eq!(
            value["gpu_trace_commit_terminal_net_device_bytes_removed"],
            1_073_741_824u64
        );
        assert_eq!(
            value["gpu_trace_commit_terminal_net_cuda_launches_removed"],
            4
        );
        assert_eq!(value["gpu_policy_commit_mode"], "domain-progressive");
        assert_eq!(
            value["gpu_policy_interpolation_mode"],
            "stage-fused-out-of-place"
        );
        assert_eq!(value["gpu_policy_blake2s_interior_fused"], false);
        assert_eq!(value["gpu_policy_composition_launch_mode"], "wave");
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
            "gpu_composition_part_count",
            "gpu_composition_wave_count",
            "gpu_prepared_numerator_packed_output_rows",
        ] {
            assert!(null_schema[key].is_null(), "{key}");
        }
    }
}
