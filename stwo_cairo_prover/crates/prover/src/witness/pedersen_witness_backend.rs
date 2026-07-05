//! Backend-specific witness generation for the Pedersen family — witness-on-GPU
//! W3 phase 2 (the partial_ec_mul / pedersen_aggregator cohort, ~27% of the
//! SN-PIE base-write share; see `gpu_benchmarks/WITNESS_ON_GPU.md` round-8
//! addendum and `gpu_benchmarks/ROAD_TO_10MHZ.md` P1).
//!
//! This module establishes the backend seam for the three targeted components,
//! each independently toggleable so the integration agent can bisect with the
//! kill switch:
//!
//! - [`PartialEcMulGenericWitness`]      — `partial_ec_mul_generic`
//! - [`PartialEcMulWindowBits18Witness`] — `partial_ec_mul_window_bits_18`
//! - [`PedersenAggregatorWindowBits18Witness`] — `pedersen_aggregator_window_bits_18`
//!
//! Each trait gives the component two interchangeable base-trace paths:
//!
//! - **`SimdBackend` (host)**: delegates to the generated writer verbatim — byte-identical to the
//!   pre-hook flow by construction (it IS that flow).
//! - **`CudaBackend` (device)**: gated behind the kill switch (default OFF). When the lane is
//!   disabled — the default until the pod differential passes — it takes the host path bridged via
//!   `from_simd_evals`, byte-identical to today.
//!
//! # Why this lane is DEFAULT OFF (soundness-first)
//!
//! Unlike blake_g (53 columns of 32-bit ops) and memory_id_to_big (a mechanical
//! 9-bit limb split), the partial_ec_mul base-trace writers are full fp256 EC-mul
//! GADGET evaluations — 600+ trace columns of 256-bit modular arithmetic with
//! per-limb range-check decompositions feeding ~18 range-check sub-component
//! families across 150+ bespoke interaction columns. That device base-trace
//! kernel (using `stwo cuda/ec_ops.cuh` + the `pedersen_table_init.cu` fp256
//! primitives) is the irreducible hardware-completion item: it must be
//! transcribed from the generated writer and qualified against the
//! `STWO_CUDA_WITNESS_VERIFY` differential ON A REAL GPU before the lane may be
//! enabled. It is NOT byte-identical by construction, so per CLAUDE.md's
//! soundness-first contract the switch defaults OFF until the pod gates pass.
//!
//! The reusable device interaction/finalize primitives that the completed lane
//! will consume are already shipped in `stwo-backend-cuda`'s `pedersen_witness`
//! module (generalized `pair_logup` / `multi_logup` + the shared
//! `finalize_device_raw_logup`), verifiable by structural equivalence to the
//! proven blake/memory kernels. The interaction trace itself already runs on
//! device via the W1 raw-logup finalize (`finalize_raw_logup`), which this lane
//! leaves in place.
//!
//! # Kill switch and per-lane toggles
//!
//! - Master: `STWO_CUDA_PEDERSEN_WITNESS=1` opts the family in (absent or `0` → OFF). This is the
//!   single kill switch.
//! - Per lane (only consulted once the master is on; set to `0` to kill just that lane):
//!   `STWO_CUDA_PEDERSEN_WITNESS_PARTIAL_EC_MUL_GENERIC`,
//!   `STWO_CUDA_PEDERSEN_WITNESS_PARTIAL_EC_MUL_W18`, `STWO_CUDA_PEDERSEN_WITNESS_AGGREGATOR_W18`.

use cairo_air::components::partial_ec_mul_generic::Claim as PartialEcMulGenericClaim;
use cairo_air::components::partial_ec_mul_window_bits_18::Claim as PartialEcMulW18Claim;
use cairo_air::components::pedersen_aggregator_window_bits_18::Claim as PedersenAggregatorW18Claim;
use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::simd::m31::N_LANES;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::FromSimdColumns;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::CudaBackend;

use crate::witness::components::{
    memory_id_to_big, partial_ec_mul_generic, partial_ec_mul_window_bits_18,
    pedersen_aggregator_window_bits_18, pedersen_points_table_window_bits_18, range_check_20,
    range_check_8, range_check_9_9,
};

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// The pedersen-family device lanes, each independently toggleable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PedersenLane {
    PartialEcMulGeneric,
    PartialEcMulWindowBits18,
    AggregatorWindowBits18,
}

impl PedersenLane {
    /// The per-lane opt-out environment variable name.
    fn env_var(self) -> &'static str {
        match self {
            PedersenLane::PartialEcMulGeneric => {
                "STWO_CUDA_PEDERSEN_WITNESS_PARTIAL_EC_MUL_GENERIC"
            }
            PedersenLane::PartialEcMulWindowBits18 => {
                "STWO_CUDA_PEDERSEN_WITNESS_PARTIAL_EC_MUL_W18"
            }
            PedersenLane::AggregatorWindowBits18 => "STWO_CUDA_PEDERSEN_WITNESS_AGGREGATOR_W18",
        }
    }

    fn human(self) -> &'static str {
        match self {
            PedersenLane::PartialEcMulGeneric => "partial_ec_mul_generic",
            PedersenLane::PartialEcMulWindowBits18 => "partial_ec_mul_window_bits_18",
            PedersenLane::AggregatorWindowBits18 => "pedersen_aggregator_window_bits_18",
        }
    }
}

/// The master kill switch (`STWO_CUDA_PEDERSEN_WITNESS`).
const MASTER_ENV: &str = "STWO_CUDA_PEDERSEN_WITNESS";

/// Pure toggle logic (unit-tested): the family opts in only on an explicit
/// master `"1"`, and each lane is then on unless individually set to `"0"`.
/// Default (both `None`) is OFF, satisfying the soundness-first "default OFF
/// until pod gates pass" contract.
fn lane_enabled_from(master: Option<&str>, lane: Option<&str>) -> bool {
    master == Some("1") && lane != Some("0")
}

/// Whether the device base-trace path should run for `lane`: requires the CUDA
/// kernels to be built AND the switches to opt in. Always `false` in a stub /
/// no-CUDA build, so the SimdBackend flow is untouched.
fn device_lane_enabled(lane: PedersenLane) -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && lane_enabled_from(
            std::env::var(MASTER_ENV).ok().as_deref(),
            std::env::var(lane.env_var()).ok().as_deref(),
        )
}

/// Emits a one-time notice (per lane) that the device base-trace kernel is not
/// yet validated, so the host path is used despite the switch being on. Keeps
/// the operator honest about what the toggle currently does.
fn warn_device_pending(lane: PedersenLane) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: [AtomicBool; 3] = [
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
    ];
    let idx = match lane {
        PedersenLane::PartialEcMulGeneric => 0,
        PedersenLane::PartialEcMulWindowBits18 => 1,
        PedersenLane::AggregatorWindowBits18 => 2,
    };
    if !WARNED[idx].swap(true, Ordering::Relaxed) {
        eprintln!(
            "STWO_CUDA_PEDERSEN_WITNESS: device base-trace kernel for {} is pending pod \
             validation (STWO_CUDA_WITNESS_VERIFY); using the host path. See \
             witness/pedersen_witness_backend.rs.",
            lane.human()
        );
    }
}

/// Backend hook for the `partial_ec_mul_generic` base-trace write.
pub trait PartialEcMulGenericWitness: FromSimdColumns {
    fn write_trace(
        gen: partial_ec_mul_generic::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulGenericClaim,
        partial_ec_mul_generic::InteractionClaimGenerator,
    );
}

/// Backend hook for the `partial_ec_mul_window_bits_18` base-trace write.
pub trait PartialEcMulWindowBits18Witness: FromSimdColumns {
    fn write_trace(
        gen: partial_ec_mul_window_bits_18::ClaimGenerator,
        pedersen_points_table: &pedersen_points_table_window_bits_18::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulW18Claim,
        partial_ec_mul_window_bits_18::InteractionClaimGenerator,
    );
}

/// Backend hook for the `pedersen_aggregator_window_bits_18` base-trace write.
pub trait PedersenAggregatorWindowBits18Witness: FromSimdColumns {
    /// `jit_memory`: the prover-input memory the D′ witness-JIT lane resolves its
    /// device execution tables from. `None` (or the Simd backend) → host writer.
    fn write_trace(
        gen: pedersen_aggregator_window_bits_18::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        partial_ec_mul_window_bits_18: &partial_ec_mul_window_bits_18::ClaimGenerator,
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        PedersenAggregatorW18Claim,
        pedersen_aggregator_window_bits_18::InteractionClaimGenerator,
    );
}

// --- SimdBackend: the pre-hook flow verbatim (byte-identical by construction). ---

impl PartialEcMulGenericWitness for SimdBackend {
    fn write_trace(
        gen: partial_ec_mul_generic::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulGenericClaim,
        partial_ec_mul_generic::InteractionClaimGenerator,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(range_check_8, range_check_9_9, range_check_20);
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

impl PartialEcMulWindowBits18Witness for SimdBackend {
    fn write_trace(
        gen: partial_ec_mul_window_bits_18::ClaimGenerator,
        pedersen_points_table: &pedersen_points_table_window_bits_18::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulW18Claim,
        partial_ec_mul_window_bits_18::InteractionClaimGenerator,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(pedersen_points_table, range_check_9_9, range_check_20);
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

impl PedersenAggregatorWindowBits18Witness for SimdBackend {
    fn write_trace(
        gen: pedersen_aggregator_window_bits_18::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        partial_ec_mul_window_bits_18: &partial_ec_mul_window_bits_18::ClaimGenerator,
        _jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        PedersenAggregatorW18Claim,
        pedersen_aggregator_window_bits_18::InteractionClaimGenerator,
    ) {
        let (trace, claim, interaction_gen) = gen.write_trace(
            memory_id_to_big,
            range_check_8,
            partial_ec_mul_window_bits_18,
        );
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

// --- CudaBackend: kill-switch gated; default OFF -> host path (byte-identical).
// The device base-trace branch is the pod-completion item (see module docs). ---

impl PartialEcMulGenericWitness for CudaBackend {
    fn write_trace(
        gen: partial_ec_mul_generic::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulGenericClaim,
        partial_ec_mul_generic::InteractionClaimGenerator,
    ) {
        if device_lane_enabled(PedersenLane::PartialEcMulGeneric) {
            warn_device_pending(PedersenLane::PartialEcMulGeneric);
        }
        let (trace, claim, interaction_gen) =
            gen.write_trace(range_check_8, range_check_9_9, range_check_20);
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

impl PartialEcMulWindowBits18Witness for CudaBackend {
    fn write_trace(
        gen: partial_ec_mul_window_bits_18::ClaimGenerator,
        pedersen_points_table: &pedersen_points_table_window_bits_18::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        range_check_20: &range_check_20::ClaimGenerator,
    ) -> (
        Evals<Self>,
        PartialEcMulW18Claim,
        partial_ec_mul_window_bits_18::InteractionClaimGenerator,
    ) {
        if device_lane_enabled(PedersenLane::PartialEcMulWindowBits18) {
            warn_device_pending(PedersenLane::PartialEcMulWindowBits18);
        }
        let (trace, claim, interaction_gen) =
            gen.write_trace(pedersen_points_table, range_check_9_9, range_check_20);
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

impl PedersenAggregatorWindowBits18Witness for CudaBackend {
    fn write_trace(
        gen: pedersen_aggregator_window_bits_18::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        range_check_8: &range_check_8::ClaimGenerator,
        partial_ec_mul_window_bits_18: &partial_ec_mul_window_bits_18::ClaimGenerator,
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        PedersenAggregatorW18Claim,
        pedersen_aggregator_window_bits_18::InteractionClaimGenerator,
    ) {
        // D′ witness-JIT lane: the recorded program — 28 computed EC-round
        // deduces on the fp256 device functions — launched as a JIT kernel on
        // the slot-layout inputs `[in0, in1, in2 | enabler | iota | mults]`.
        // Gated by `STWO_CUDA_WITNESS_JIT_PROVE(_PEDERSEN_AGGREGATOR_WINDOW_
        // BITS_18)`; any unavailability falls back to the host writer below
        // (this block only READS `gen`, so the fallback stays valid).
        if let Some(mem) = jit_memory {
            use std::sync::atomic::Ordering;
            let mut inputs_mults = gen
                .mults
                .iter()
                .map(|entry| {
                    (
                        *entry.key(),
                        BaseField::from_u32_unchecked(entry.value().load(Ordering::Relaxed)),
                    )
                })
                .collect::<Vec<_>>();
            inputs_mults.sort_by_key(|(input, _)| input.0);
            let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let n_real = inputs.len();
            if n_real > 0 {
                let size = std::cmp::max(n_real.next_power_of_two(), N_LANES);
                inputs.resize(size, *inputs.first().unwrap());
                mults.resize(size, BaseField::from_u32_unchecked(0));
                let cols: Vec<Vec<u32>> = vec![
                    inputs.iter().map(|i| i.0[0].0).collect(),
                    inputs.iter().map(|i| i.0[1].0).collect(),
                    inputs.iter().map(|i| i.1 .0).collect(),
                    (0..size).map(|r| u32::from(r < n_real)).collect(),
                    (0..size).map(|r| r as u32).collect(),
                    mults.iter().map(|m| m.0).collect(),
                ];
                let lut_for = |family: &'static str| -> Vec<u32> {
                    panic!("aggregator count feed needs no LUT, got {family}")
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_8_state" => range_check_8.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let plan = crate::witness::jit_prove_backend::DeviceFeedPlan {
                    layout: pedersen_aggregator_window_bits_18::SUB_FEED_LAYOUT,
                    lut_for: &lut_for,
                    merge: &merge,
                };
                let launched = crate::witness::jit_prove_backend::builtin_cuda_write_trace::<
                    crate::witness::jit_prove_backend::PedersenAggregatorW18Lane,
                >(
                    &cols,
                    n_real,
                    mem,
                    Some(plan),
                    |sub_flat, n_padded, skip| {
                        pedersen_aggregator_window_bits_18::feed_sub_inputs_from_flat(
                            sub_flat,
                            n_padded,
                            memory_id_to_big,
                            range_check_8,
                            partial_ec_mul_window_bits_18,
                            skip,
                        );
                    },
                );
                if let Some(out) = launched {
                    return out;
                }
            }
        }
        if device_lane_enabled(PedersenLane::AggregatorWindowBits18) {
            warn_device_pending(PedersenLane::AggregatorWindowBits18);
        }
        let (trace, claim, interaction_gen) = gen.write_trace(
            memory_id_to_big,
            range_check_8,
            partial_ec_mul_window_bits_18,
        );
        (
            Self::from_simd_evals(trace.to_evals()),
            claim,
            interaction_gen,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_off_by_default() {
        // Absent master and lane -> OFF (the soundness-first default).
        assert!(!lane_enabled_from(None, None));
        // Lane env alone (no master opt-in) -> OFF.
        assert!(!lane_enabled_from(None, Some("1")));
    }

    #[test]
    fn master_kill_switch() {
        // Master must be exactly "1" to opt in.
        assert!(lane_enabled_from(Some("1"), None));
        assert!(!lane_enabled_from(Some("0"), None));
        assert!(!lane_enabled_from(Some(""), None));
        assert!(!lane_enabled_from(Some("true"), None));
    }

    #[test]
    fn per_lane_opt_out() {
        // With the master on, a lane is on unless explicitly "0".
        assert!(lane_enabled_from(Some("1"), Some("1")));
        assert!(lane_enabled_from(Some("1"), Some("anything")));
        assert!(!lane_enabled_from(Some("1"), Some("0")));
    }

    #[test]
    fn lane_env_names_are_distinct() {
        let names = [
            PedersenLane::PartialEcMulGeneric.env_var(),
            PedersenLane::PartialEcMulWindowBits18.env_var(),
            PedersenLane::AggregatorWindowBits18.env_var(),
        ];
        assert_eq!(
            names[0],
            "STWO_CUDA_PEDERSEN_WITNESS_PARTIAL_EC_MUL_GENERIC"
        );
        assert_ne!(names[0], names[1]);
        assert_ne!(names[1], names[2]);
        assert_ne!(names[0], names[2]);
    }
}
