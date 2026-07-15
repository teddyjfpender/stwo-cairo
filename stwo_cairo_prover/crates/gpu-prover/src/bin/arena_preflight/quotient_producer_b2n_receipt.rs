use serde_json::{json, Value};
use stwo_backend_cuda::QuotientProducerB2nReceipt;
use stwo_cairo_gpu_prover::arena_plan::ProofArenaPlan;

pub(crate) fn json(arena: &ProofArenaPlan) -> Value {
    let selection = arena.quotient_producer_b2n_selection_receipt();
    json!({
        "schema": "stwo.quotient-producer-b2n-selection.v2",
        "resident_backend": selection.resident_backend.cli_name(),
        "production_selected": selection.production_selected,
        "program": selection.program.map(program_json),
    })
}

fn program_json(receipt: QuotientProducerB2nReceipt) -> Value {
    json!({
        "schedule": {
            "lifting_log_size": receipt.schedule.lifting_log_size,
            "subdomain_log_size": receipt.schedule.subdomain_log_size,
            "sample_count": receipt.schedule.sample_count,
            "producer_stages": receipt.schedule.producer_stages,
            "continuation_intervals": receipt.schedule.continuation_intervals,
        },
        "resource_policy": {
            "required_sm_arch": receipt.resources.required_sm_arch,
            "architecture_registers_per_sm": receipt.resources.registers_per_sm,
            "producer": {
                "launch_threads": receipt.resources.producer.launch_threads,
                "required_blocks_per_sm": receipt.resources.producer.required_blocks_per_sm,
                "max_registers_per_thread": receipt.resources.producer.max_registers_per_thread,
                "max_local_bytes": receipt.resources.producer.max_local_bytes,
                "max_static_shared_bytes": receipt.resources.producer.max_static_shared_bytes,
            },
            "continuation": {
                "launch_threads": receipt.resources.continuation.launch_threads,
                "required_blocks_per_sm": receipt.resources.continuation.required_blocks_per_sm,
                "max_registers_per_thread": receipt.resources.continuation.max_registers_per_thread,
                "max_local_bytes": receipt.resources.continuation.max_local_bytes,
                "max_static_shared_bytes": receipt.resources.continuation.max_static_shared_bytes,
            },
        },
        "traffic": {
            "coordinate_image_bytes": receipt.traffic.coordinate_image_bytes,
            "unchanged_partial_read_bytes": receipt.traffic.unchanged_partial_read_bytes,
            "denominator_factors": receipt.traffic.denominator_factors,
            "batch_inverse_calls": receipt.traffic.batch_inverse_calls,
            "fallback_logical_bytes": receipt.traffic.fallback_logical_bytes,
            "fused_logical_bytes": receipt.traffic.fused_logical_bytes,
            "eliminated_logical_bytes": receipt.traffic.eliminated_logical_bytes,
            "fallback_kernel_launches": receipt.traffic.fallback_kernel_launches,
            "fused_kernel_launches": receipt.traffic.fused_kernel_launches,
            "eliminated_kernel_launches": receipt.traffic.eliminated_kernel_launches,
        },
    })
}
