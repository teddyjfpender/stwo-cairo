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
        if let Some(mem) = jit_memory {
            use stwo::prover::backend::simd::m31::N_LANES;
            let packed: Vec<blake_round::PackedInputType> =
                gen.packed_inputs.lock().unwrap().clone();
            let remainder_empty = gen.remainder_inputs.lock().unwrap().is_empty();
            if !packed.is_empty() && remainder_empty {
                // Replicate the host preamble EXACTLY: n_rows counts every lane of
                // the pre-pad packed rows; padding repeats the first PACKED row.
                let n_vec_rows = packed.len();
                let n_real = n_vec_rows * N_LANES;
                let packed_size = n_vec_rows.next_power_of_two();
                let size = packed_size * N_LANES;
                let mut padded = packed;
                padded.resize(packed_size, *padded.first().unwrap());
                let mut cols: Vec<Vec<u32>> = vec![Vec::with_capacity(size); 21];
                for p in &padded {
                    let chain = p.0.to_array();
                    let round = p.1.to_array();
                    let mp = p.2 .1.to_array();
                    for l in 0..N_LANES {
                        cols[0].push(chain[l].0);
                        cols[1].push(round[l].0);
                        for (wi, w) in p.2 .0.iter().enumerate() {
                            cols[2 + wi].push(w.simd.as_array()[l]);
                        }
                        cols[18].push(mp[l].0);
                    }
                }
                cols[19] = (0..size).map(|r| u32::from(r < n_real)).collect();
                cols[20] = (0..size).map(|r| r as u32).collect();
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_7_2_5_state" => range_check_7_2_5_state.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_7_2_5_state" => range_check_7_2_5_state.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let plan = crate::witness::jit_prove_backend::DeviceFeedPlan {
                    layout: blake_round::SUB_FEED_LAYOUT,
                    lut_for: &lut_for,
                    merge: &merge,
                    require: false,
                };
                let launched = crate::witness::jit_prove_backend::builtin_cuda_write_trace::<
                    crate::witness::jit_prove_backend::BlakeRoundLane,
                >(
                    &cols,
                    n_real,
                    mem,
                    Some(plan),
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
