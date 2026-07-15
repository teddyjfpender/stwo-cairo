use serde_json::{json, Value};
use stwo_backend_cuda::QuotientProducerB2nReceipt;
use stwo_cairo_gpu_prover::arena_plan::ProofArenaPlan;

pub(crate) fn json(arena: &ProofArenaPlan) -> Value {
    let selection = arena.quotient_producer_b2n_selection_receipt();
    json!({
        "schema": "stwo.quotient-producer-b2n-selection.v1",
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
        "resources": {
            "sm_arch": receipt.resources.sm_arch,
            "cuda_toolkit_major": receipt.resources.cuda_toolkit_major,
            "cuda_toolkit_minor": receipt.resources.cuda_toolkit_minor,
            "launch_threads": receipt.resources.launch_threads,
            "min_blocks_per_sm": receipt.resources.min_blocks_per_sm,
            "ptxas_registers_per_thread": receipt.resources.ptxas_registers_per_thread,
            "max_registers_per_thread": receipt.resources.max_registers_per_thread,
            "ptxas_stack_bytes": receipt.resources.ptxas_stack_bytes,
            "ptxas_spill_store_bytes": receipt.resources.ptxas_spill_store_bytes,
            "ptxas_spill_load_bytes": receipt.resources.ptxas_spill_load_bytes,
            "static_shared_bytes": receipt.resources.static_shared_bytes,
            "zero_spills_required": receipt.resources.zero_spills_required,
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
