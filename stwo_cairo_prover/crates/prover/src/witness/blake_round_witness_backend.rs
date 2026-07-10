//! Backend-specific witness generation for the `blake_round` component —
//! witness-on-GPU W3 (the BLAKE family, see `gpu_benchmarks/WITNESS_ON_GPU.md`
//! round-8 addendum: `blake_g + blake_round` are ~23% of the SN-PIE base-write
//! share and the most GPU-natural port).
//!
//! [`BlakeRoundWitness`] gives the component two interchangeable paths behind the
//! `STWO_CUDA_BLAKE_ROUND_WITNESS` kill switch, mirroring [`crate::witness::
//! blake_g_witness_backend::BlakeGWitness`]:
//!
//! - **`SimdBackend` (host)**: delegates to the generated writer in `components/blake_round.rs`.
//!   Byte-identical to the pre-hook flow by construction (it IS that flow).
//! - **`CudaBackend` (device)**: the device lane writes the 212 base-trace columns on device and,
//!   per the documented design seam, emits `blake_g`'s `[u32; 6]` input words STRAIGHT INTO A
//!   DEVICE BUFFER that [`crate::witness::blake_g_witness_backend`] consumes in place of its
//!   host-built H2D upload (device-to-device component feeding). Its sub-feeds
//!   (`memory_address_to_id`, `memory_id_to_big` — device lanes already exist — plus
//!   `range_check_7_2_5`, `blake_round_sigma`) accumulate through the same device count-table
//!   accessors the memory/rc lanes use.
//!
//! ## Status (LOCAL-ONLY session)
//!
//! The 212-column / 3008-line device kernel port and the device-to-device
//! `blake_g` feed are **pod-validation-gated**: they land only behind the
//! `STWO_CUDA_WITNESS_VERIFY` differential (device-vs-host column byte-compare)
//! and the Cairo e2e proof byte-equality, neither of which can run on the CUDA-stub
//! Mac build. Until those gates pass on a pod, [`CudaBackend`] runs the **host
//! writer and uploads** — byte-identical to today's committed behaviour, so the
//! default flow is unchanged and the switch defaults OFF. The trait, the call-site
//! wiring, and the kill switch are in place so the pod session drops in the device
//! kernel behind [`device_lane_enabled`] without further plumbing (at which point
//! the associated interaction-generator type generalises to a device/host enum
//! exactly as `blake_g` did).

use cairo_air::components::blake_round::{Claim as BlakeRoundClaim, N_TRACE_COLUMNS};
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::FromSimdColumns;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::CudaBackend;

use crate::witness::components::{
    blake_g, blake_round, blake_round_sigma, memory_address_to_id, memory_id_to_big,
    range_check_7_2_5,
};
use crate::witness::exec_context::WitnessExecContext;

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// Committed base-trace column count (cairo-air blake_round `N_TRACE_COLUMNS`).
pub const BR_N_TRACE: usize = N_TRACE_COLUMNS; // 212

/// Backend hook for the `blake_round` witness (base trace + sub-component feeds).
///
/// The interaction generator stays the concrete host type for both backends: the
/// interaction trace is finalised through the shared raw-logup path downstream, so
/// only the base-trace write needs a backend split today. When the device lane
/// lands, this associated type generalises to a device/host enum (the `blake_g`
/// precedent) — an additive change local to this module and the call site.
pub trait BlakeRoundWitness: FromSimdColumns {
    /// Writes the blake_round base trace on `Self` and feeds its five
    /// sub-components. Trace/claim bytes must be identical to the host writer's.
    /// `jit_memory`: the prover-input memory the D′ witness-JIT lane resolves its
    /// device execution tables from. `None` (or the Simd backend) → host writer.
    #[allow(clippy::too_many_arguments)]
    fn write_trace(
        exec_context: &WitnessExecContext,
        gen: blake_round::ClaimGenerator,
        blake_round_sigma_state: &blake_round_sigma::ClaimGenerator,
        memory_address_to_id_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big::ClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5::ClaimGenerator,
        blake_g_state: &blake_g::ClaimGenerator,
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        BlakeRoundClaim,
        blake_round::InteractionClaimGenerator,
    );
}

impl BlakeRoundWitness for SimdBackend {
    fn write_trace(
        _exec_context: &WitnessExecContext,
        gen: blake_round::ClaimGenerator,
        blake_round_sigma_state: &blake_round_sigma::ClaimGenerator,
        memory_address_to_id_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big::ClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5::ClaimGenerator,
        blake_g_state: &blake_g::ClaimGenerator,
        _jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        BlakeRoundClaim,
        blake_round::InteractionClaimGenerator,
    ) {
        let (trace, claim, interaction_gen) = gen.write_trace(
            blake_round_sigma_state,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
            blake_g_state,
        );
        (trace.to_evals(), claim, interaction_gen)
    }
}

impl BlakeRoundWitness for CudaBackend {
    fn write_trace(
        exec_context: &WitnessExecContext,
        gen: blake_round::ClaimGenerator,
        blake_round_sigma_state: &blake_round_sigma::ClaimGenerator,
        memory_address_to_id_state: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big_state: &memory_id_to_big::ClaimGenerator,
        range_check_7_2_5_state: &range_check_7_2_5::ClaimGenerator,
        blake_g_state: &blake_g::ClaimGenerator,
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        BlakeRoundClaim,
        blake_round::InteractionClaimGenerator,
    ) {
        // D′ witness-JIT lane: the recorded program — 8 blake_g + 1 sigma
        // computed deduces — launched as a JIT kernel on the slot-layout inputs
        // `[chain, round, 16 raw message words, mp | enabler | iota]`. Gated by
        // `STWO_CUDA_WITNESS_JIT_PROVE(_BLAKE_ROUND)`; any unavailability falls
        // back to the host writer below (this block only READS `gen`).
        //
        // Source edge: blake_compress_opcode emits ten exact 19-word BlakeRound
        // inputs per padded producer row. Consume those columns in place; the
        // generator intentionally remains empty on this path. This upstream edge
        // still carries its recovery mirror until it receives the same
        // conformance certification as the two downstream resident edges.
        if let Some(mem) = jit_memory {
            if let Some(edge) = exec_context.take_edge("blake_compress_opcode", "blake_round") {
                let edge_plan = edge.plan;
                assert_eq!(
                    edge_plan.words_per_instance, 19,
                    "blake_round edge ABI requires nineteen words per instance"
                );
                assert_eq!(
                    edge_plan.n_instances, 10,
                    "blake_compress_opcode must emit ten Blake rounds per row"
                );
                let n_real = edge_plan.n_instances as usize * edge.n_rows;
                let padded = std::cmp::max(
                    n_real.next_power_of_two(),
                    stwo::prover::backend::simd::m31::N_LANES,
                );
                let host_tail = vec![
                    (0..padded).map(|row| u32::from(row < n_real)).collect(),
                    (0..padded).map(|row| row as u32).collect(),
                ];
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_7_2_5_state" => range_check_7_2_5_state.input_to_row_lut(),
                        "blake_round_sigma_state" => blake_round_sigma_state.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_7_2_5_state" => range_check_7_2_5_state.add_count_tables(counts),
                    "blake_round_sigma_state" => blake_round_sigma_state.add_count_tables(counts),
                    "memory_address_to_id_state" => {
                        memory_address_to_id_state.add_count_tables(counts)
                    }
                    "memory_id_to_big_state" => memory_id_to_big_state.add_big_count_tables(counts),
                    "memory_id_to_big_state#small" => {
                        memory_id_to_big_state.add_small_count_tables(counts)
                    }
                    other => panic!("unexpected count family {other}"),
                };
                let sizes = |family: &'static str| match family {
                    "memory_address_to_id_state" => {
                        Some((memory_address_to_id_state.table_size(), 0))
                    }
                    "memory_id_to_big_state" => Some((
                        memory_id_to_big_state.big_table_size(),
                        memory_id_to_big_state.small_table_size(),
                    )),
                    _ => None,
                };
                let plan = crate::witness::jit_prove_backend::DeviceFeedPlan {
                    layout: blake_round::SUB_FEED_LAYOUT,
                    lut_for: &lut_for,
                    merge: &merge,
                    sizes: &sizes,
                    require: false,
                };
                let launched = stwo_backend_cuda::exec_tables::witness_edge_gather(
                    &edge.buffer,
                    edge.n_rows,
                    edge_plan.word_base as usize,
                    edge_plan.words_per_instance as usize,
                    edge_plan.n_instances as usize,
                    padded,
                )
                .and_then(|device_cols| {
                    crate::witness::jit_prove_backend::builtin_cuda_write_trace_from::<
                        crate::witness::jit_prove_backend::BlakeRoundLane,
                    >(
                        exec_context,
                        crate::witness::jit_prove_backend::BuiltinInputs::Edge {
                            device_cols,
                            host_tail,
                        },
                        n_real,
                        mem,
                        Some(plan),
                        Some(
                            crate::witness::jit_prove_backend::DeviceEdgeTarget::fail_closed(
                                "blake_g",
                                "blake_g_state",
                            ),
                        ),
                        |sub_flat, n_padded, skip| {
                            blake_round::feed_sub_inputs_from_flat(
                                sub_flat,
                                n_padded,
                                blake_round_sigma_state,
                                memory_address_to_id_state,
                                memory_id_to_big_state,
                                range_check_7_2_5_state,
                                blake_g_state,
                                skip,
                            );
                        },
                    )
                });
                if let Some(out) = launched {
                    return out;
                }
                eprintln!(
                    "jit_prove[blake_round]: blake_compress_opcode device edge failed - \
                     rebuilding inputs from the stashed host flat"
                );
                crate::witness::jit_prove_backend::feed_blake_round_inputs_from_blake_compress_flat(
                    edge.host_flat.as_deref().expect(
                        "recoverable blake_compress_opcode -> blake_round edge lost its host mirror",
                    ),
                    edge.n_rows,
                    &gen,
                );
            }
        }
        if let Some(mem) = jit_memory {
            if let Ok(inputs) = <crate::witness::jit_prove_backend::BlakeRoundLane as crate::witness::jit_prove_backend::BuiltinLaneSpec>::input_columns(&gen) {
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_7_2_5_state" => range_check_7_2_5_state.input_to_row_lut(),
                        "blake_round_sigma_state" => blake_round_sigma_state.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_7_2_5_state" => range_check_7_2_5_state.add_count_tables(counts),
                    "blake_round_sigma_state" => blake_round_sigma_state.add_count_tables(counts),
                    "memory_address_to_id_state" => {
                        memory_address_to_id_state.add_count_tables(counts)
                    }
                    "memory_id_to_big_state" => memory_id_to_big_state.add_big_count_tables(counts),
                    "memory_id_to_big_state#small" => {
                        memory_id_to_big_state.add_small_count_tables(counts)
                    }
                    other => panic!("unexpected count family {other}"),
                };
                let sizes = |family: &'static str| match family {
                    "memory_address_to_id_state" => {
                        Some((memory_address_to_id_state.table_size(), 0))
                    }
                    "memory_id_to_big_state" => Some((
                        memory_id_to_big_state.big_table_size(),
                        memory_id_to_big_state.small_table_size(),
                    )),
                    _ => None,
                };
                let plan = crate::witness::jit_prove_backend::DeviceFeedPlan {
                    layout: blake_round::SUB_FEED_LAYOUT,
                    lut_for: &lut_for,
                    merge: &merge,
                    sizes: &sizes,
                    require: false,
                };
                let launched = crate::witness::jit_prove_backend::builtin_cuda_write_trace_from::<
                    crate::witness::jit_prove_backend::BlakeRoundLane,
                >(
                    exec_context,
                    crate::witness::jit_prove_backend::BuiltinInputs::HostCols(&inputs.columns),
                    inputs.n_real,
                    mem,
                    Some(plan),
                    Some(
                        crate::witness::jit_prove_backend::DeviceEdgeTarget::fail_closed(
                            "blake_g",
                            "blake_g_state",
                        ),
                    ),
                    |sub_flat, n_padded, skip| {
                        blake_round::feed_sub_inputs_from_flat(
                            sub_flat,
                            n_padded,
                            blake_round_sigma_state,
                            memory_address_to_id_state,
                            memory_id_to_big_state,
                            range_check_7_2_5_state,
                            blake_g_state,
                            skip,
                        );
                    },
                );
                if let Some(out) = launched {
                    return out;
                }
            }
        }
        if device_lane_enabled() {
            // The hand-ported 212-col kernel design is superseded by the JIT lane
            // above; until the JIT lane passes the pod gates, the switch still
            // falls back to the host writer so it never silently ships an
            // unvalidated witness.
            warn_device_lane_pending();
        }
        let (trace, claim, interaction_gen) = gen.write_trace(
            blake_round_sigma_state,
            memory_address_to_id_state,
            memory_id_to_big_state,
            range_check_7_2_5_state,
            blake_g_state,
        );
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

/// Whether the `blake_round` device lane is requested. Default: enabled iff the
/// CUDA kernels are built AND the switch is not `0`. The device path itself is
/// pod-gated (see module docs), so today an enabled switch still falls back to the
/// host writer with a one-time notice.
pub fn device_lane_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_BLAKE_ROUND_WITNESS").as_deref() != Ok("0")
}

fn warn_device_lane_pending() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "STWO_CUDA_BLAKE_ROUND_WITNESS: device lane requested but the blake_round \
             device kernel is pod-validation-gated; falling back to the host writer \
             (byte-identical). Set STWO_CUDA_BLAKE_ROUND_WITNESS=0 to silence."
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn br_n_trace_matches_air() {
        assert_eq!(BR_N_TRACE, 212);
    }

    /// The device-resident xor witness lane (see `stwo`'s
    /// `backend/blake_witness.rs::xor_mult_columns`) treats a device count table
    /// with layout `counts[rel * table_size + row]` as the per-relation
    /// multiplicity columns directly. This is byte-identical to the host merge
    /// (`AtomicMultiplicityColumn::add_at` into a zeroed column, u32 `fetch_add`
    /// wrap, no field reduction) precisely because the slice IS the column. This
    /// test pins that layout invariant so a device-lane regression is caught on the
    /// CUDA-stub host build, where the device slice itself cannot run.
    #[test]
    fn xor_count_table_slice_is_the_multiplicity_column() {
        let n_relations = 3usize;
        let table_size = 8usize;
        // Synthetic count table, distinct per (rel,row).
        let counts: Vec<u32> = (0..n_relations * table_size)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();

        for rel in 0..n_relations {
            // Reference: fold the count table into a fresh atomic multiplicity
            // column the way the host `add_count_tables` accessor does.
            let column: Vec<u32> = (0..table_size)
                .map(|row| {
                    let mut acc = 0u32;
                    let count = counts[rel * table_size + row];
                    if count != 0 {
                        acc = acc.wrapping_add(count);
                    }
                    acc
                })
                .collect();

            // Device lane: the relation's contiguous slice, verbatim.
            let slice = &counts[rel * table_size..(rel + 1) * table_size];

            assert_eq!(
                column.as_slice(),
                slice,
                "count-table slice for relation {rel} must equal the multiplicity column",
            );
        }
    }
}
