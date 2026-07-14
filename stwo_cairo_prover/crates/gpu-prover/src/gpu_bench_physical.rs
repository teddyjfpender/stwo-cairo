use serde_json::json;
use stwo_cairo_gpu_prover::ResidentSessionTelemetry;

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
    json!({
        "gpu_graph_a_setup_gate_passed": telemetry.require_strict_graph_a().is_ok(),
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

#[cfg(test)]
mod tests {
    use stwo_cairo_gpu_prover::memory_ledger::{AllocatorPoolCheckpoint, PhysicalMemoryInputs};

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
}
