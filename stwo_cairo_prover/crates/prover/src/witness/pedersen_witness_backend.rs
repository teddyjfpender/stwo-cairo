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
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
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
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
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
        _jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
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
        _jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
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
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        PartialEcMulGenericClaim,
        partial_ec_mul_generic::InteractionClaimGenerator,
    ) {
        // D′ witness-JIT lane: 17,000-instr recorded body (inline fp256 felt
        // deduces), slot columns `[in.0, in.1, W27 x10, 2x28, 2x28, in.2.3 |
        // enabler | iota]`. ALL its relations are count-style — the device feed
        // is REQUIRED; any unavailability falls back to the host writer below.
        if let Some(mem) = jit_memory {
            use stwo::prover::backend::simd::m31::N_LANES;
            let packed: Vec<partial_ec_mul_generic::PackedInputType> =
                gen.packed_inputs.lock().unwrap().clone();
            let remainder_empty = gen.remainder_inputs.lock().unwrap().is_empty();
            if !packed.is_empty() && remainder_empty {
                let n_vec_rows = packed.len();
                let n_real = n_vec_rows * N_LANES;
                let packed_size = n_vec_rows.next_power_of_two();
                let size = packed_size * N_LANES;
                let mut padded = packed;
                padded.resize(packed_size, *padded.first().unwrap());
                let mut cols: Vec<Vec<u32>> = vec![Vec::with_capacity(size); 127];
                for p in &padded {
                    let a0 = p.0.to_array();
                    let a1 = p.1.to_array();
                    let tail = p.2 .3.to_array();
                    for l in 0..N_LANES {
                        cols[0].push(a0[l].0);
                        cols[1].push(a1[l].0);
                        for i in 0..10 {
                            cols[2 + i].push(p.2 .0.get_m31(i).to_array()[l].0);
                        }
                        for (fi, f) in p.2 .1.iter().enumerate() {
                            for i in 0..28 {
                                cols[12 + fi * 28 + i].push(f.get_m31(i).to_array()[l].0);
                            }
                        }
                        for (fi, f) in p.2 .2.iter().enumerate() {
                            for i in 0..28 {
                                cols[68 + fi * 28 + i].push(f.get_m31(i).to_array()[l].0);
                            }
                        }
                        cols[124].push(tail[l].0);
                    }
                }
                cols[125] = (0..size).map(|r| u32::from(r < n_real)).collect();
                cols[126] = (0..size).map(|r| r as u32).collect();
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_9_9_state" => range_check_9_9.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_8_state" => range_check_8.add_count_tables(counts),
                    "range_check_9_9_state" => range_check_9_9.add_count_tables(counts),
                    "range_check_20_state" => range_check_20.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let launched = crate::witness::jit_prove_backend::all_count_builtin_write_trace::<
                    crate::witness::jit_prove_backend::PartialEcMulGenericLane,
                >(
                    &cols,
                    n_real,
                    mem,
                    partial_ec_mul_generic::SUB_FEED_LAYOUT,
                    &lut_for,
                    &merge,
                );
                if let Some(out) = launched {
                    return out;
                }
            }
        }
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
        jit_memory: Option<&std::sync::Arc<stwo_cairo_adapter::memory::Memory>>,
    ) -> (
        Evals<Self>,
        PartialEcMulW18Claim,
        partial_ec_mul_window_bits_18::InteractionClaimGenerator,
    ) {
        // D′ witness-JIT lane: the W18 EC-round body (points-table + felt
        // deduces on the device pedersen table), slot columns `[in.0, in.1,
        // 14 windows, acc0 x28, acc1 x28 | enabler | iota]`. ALL relations are
        // count-style (points table included) — device feed REQUIRED.
        // B3 edge consumer: the aggregator's lane stashed its device sub buffer
        // (and skipped the host w18 feed). Prefer the device gather; any failure
        // rebuilds the inputs on CPU from the stashed HOST flat (the edge-gate
        // math) so feeds stay exactly-once with no pair rerun.
        if let Some(mem) = jit_memory {
            if let Some((sub_dev, sub_host, prod_rows)) =
                crate::witness::jit_prove_backend::take_edge("partial_ec_mul_window_bits_18_state")
            {
                use stwo::prover::backend::simd::m31::N_LANES;
                let n_real = 28 * prod_rows;
                let padded = std::cmp::max(n_real.next_power_of_two(), N_LANES);
                let host_tail: Vec<Vec<u32>> = vec![
                    (0..padded).map(|r| u32::from(r < n_real)).collect(),
                    (0..padded).map(|r| r as u32).collect(),
                ];
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_9_9_state" => range_check_9_9.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "pedersen_points_table_window_bits_18_state" => {
                        pedersen_points_table.add_count_tables(counts)
                    }
                    "range_check_9_9_state" => range_check_9_9.add_count_tables(counts),
                    "range_check_20_state" => range_check_20.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let plan = crate::witness::jit_prove_backend::DeviceFeedPlan {
                    layout: partial_ec_mul_window_bits_18::SUB_FEED_LAYOUT,
                    lut_for: &lut_for,
                    merge: &merge,
                    // Memory families stay host-fed at this seam until sized.
                    sizes: &|_| None,
                    require: true,
                };
                let launched = stwo_backend_cuda::exec_tables::witness_edge_gather(
                    &sub_dev, prod_rows, 7, 72, 28, padded,
                )
                .and_then(|device_cols| {
                    crate::witness::jit_prove_backend::builtin_cuda_write_trace_from::<
                        crate::witness::jit_prove_backend::PartialEcMulW18Lane,
                    >(
                        crate::witness::jit_prove_backend::BuiltinInputs::Edge {
                            device_cols,
                            host_tail: host_tail.clone(),
                        },
                        n_real,
                        mem,
                        Some(plan),
                        None,
                        |_s, _n, fed| debug_assert!(!fed.is_empty()),
                    )
                });
                match launched {
                    Some(out) => return out,
                    None => {
                        // Recoverable: rebuild the generator's inputs from the
                        // stashed HOST flat, then fall through to the normal
                        // paths (device host-cols lane, then the host writer).
                        eprintln!(
                            "jit_prove[partial_ec_mul_window_bits_18]: device edge failed — \
                             rebuilding inputs from the stashed host flat"
                        );
                        pedersen_aggregator_window_bits_18::feed_w18_inputs_from_flat(
                            &sub_host, prod_rows, &gen,
                        );
                    }
                }
            }
        }
        if let Some(mem) = jit_memory {
            use stwo::prover::backend::simd::m31::N_LANES;
            let packed: Vec<partial_ec_mul_window_bits_18::PackedInputType> =
                gen.packed_inputs.lock().unwrap().clone();
            let remainder_empty = gen.remainder_inputs.lock().unwrap().is_empty();
            if !packed.is_empty() && remainder_empty {
                let n_vec_rows = packed.len();
                let n_real = n_vec_rows * N_LANES;
                let packed_size = n_vec_rows.next_power_of_two();
                let size = packed_size * N_LANES;
                let mut padded = packed;
                padded.resize(packed_size, *padded.first().unwrap());
                let mut cols: Vec<Vec<u32>> = vec![Vec::with_capacity(size); 74];
                for p in &padded {
                    let a0 = p.0.to_array();
                    let a1 = p.1.to_array();
                    for l in 0..N_LANES {
                        cols[0].push(a0[l].0);
                        cols[1].push(a1[l].0);
                        for (wi, w) in p.2 .0.iter().enumerate() {
                            cols[2 + wi].push(w.to_array()[l].0);
                        }
                        for (fi, f) in p.2 .1.iter().enumerate() {
                            for i in 0..28 {
                                cols[16 + fi * 28 + i].push(f.get_m31(i).to_array()[l].0);
                            }
                        }
                    }
                }
                cols[72] = (0..size).map(|r| u32::from(r < n_real)).collect();
                cols[73] = (0..size).map(|r| r as u32).collect();
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_9_9_state" => range_check_9_9.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "pedersen_points_table_window_bits_18_state" => {
                        pedersen_points_table.add_count_tables(counts)
                    }
                    "range_check_9_9_state" => range_check_9_9.add_count_tables(counts),
                    "range_check_20_state" => range_check_20.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let launched = crate::witness::jit_prove_backend::all_count_builtin_write_trace::<
                    crate::witness::jit_prove_backend::PartialEcMulW18Lane,
                >(
                    &cols,
                    n_real,
                    mem,
                    partial_ec_mul_window_bits_18::SUB_FEED_LAYOUT,
                    &lut_for,
                    &merge,
                );
                if let Some(out) = launched {
                    return out;
                }
            }
        }
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
                    // Memory families stay host-fed at this seam until sized.
                    sizes: &|_| None,
                    require: false,
                };
                let launched = crate::witness::jit_prove_backend::builtin_cuda_write_trace_from::<
                    crate::witness::jit_prove_backend::PedersenAggregatorW18Lane,
                >(
                    crate::witness::jit_prove_backend::BuiltinInputs::HostCols(&cols),
                    n_real,
                    mem,
                    Some(plan),
                    Some("partial_ec_mul_window_bits_18_state"),
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
