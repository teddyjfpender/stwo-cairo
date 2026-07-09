//! Backend-specific witness generation for the `blake_g` component — witness-on-GPU
//! W3 phase 2 (the BLAKE g-function family, see `gpu_benchmarks/WITNESS_ON_GPU.md`
//! round-8 addendum; blake_g is the largest single base-write share and the most
//! GPU-natural port — 32-bit modular arithmetic + xor-rotations).
//!
//! [`BlakeGWitness`] gives the component two interchangeable paths:
//!
//! - **`SimdBackend` (host)**: delegates to the generated writer in `components/blake_g.rs` plus
//!   the raw-logup SIMD finalize — byte-identical to the pre-hook flow by construction (it IS that
//!   flow).
//! - **`CudaBackend` (device)**: the 6 input words upload once; the 53 base-trace columns (plus 20
//!   auxiliary operand columns), the five `verify_bitwise_xor_*` multiplicity feeds, and the 9
//!   logup interaction columns are all born on device via the `stwo-backend-cuda` `blake_witness`
//!   kernels. Only the xor count tables come back to the host (merged into the xor generators'
//!   atomic multiplicities — order-independent adds, byte-equal by construction).
//!
//! Load-bearing contracts of the device path:
//! 1. The base-trace kernel replicates `blake_g.rs::write_trace_simd` formula-for-formula
//!    (triple-sums mod 2^32, split-16 low parts, xor-rotations r16/r12/r8/r7).
//! 2. The xor feed mirrors each family's `add_input`: `xor_8/4/7/9` are dense `input_to_row` LUTs
//!    (keyed by `(a << n) | b`), `xor_12` is the EXPANDED table (closed-form `column_index = (ah <<
//!    EXPAND_BITS) + bh`, `row_index = (al << LIMB_BITS) + bl`). Padding rows feed their operands
//!    exactly like the host.
//! 3. The committed base columns are D2D clones of the device buffers; the originals stay in the
//!    interaction state so the commit never aliases the interaction inputs.
//!
//! Gates: `STWO_CUDA_WITNESS_VERIFY=1` runs the host writer on cloned inputs and
//! byte-compares every base-trace column and the 9 finalized interaction columns +
//! claimed sum (panicking on mismatch). The xor multiplicity feed is validated by the
//! Cairo e2e proof byte-equality (the `verify_bitwise_xor_*` component proofs are
//! identical only if the fed multiplicities match). The device lane falls back to the
//! host path when the CUDA kernels are not built or `STWO_CUDA_BLAKE_WITNESS=0`.

use cairo_air::components::blake_g::{Claim as BlakeGClaim, N_TRACE_COLUMNS};
use cairo_air::components::verify_bitwise_xor_12::{
    EXPAND_BITS as XOR12_EXPAND_BITS, LIMB_BITS as XOR12_LIMB_BITS, LOG_SIZE as XOR12_LOG_SIZE,
    N_MULT_COLUMNS as XOR12_N_MULT,
};
use cairo_air::components::verify_bitwise_xor_4::LOG_SIZE as XOR4_LOG_SIZE;
use cairo_air::components::verify_bitwise_xor_7::LOG_SIZE as XOR7_LOG_SIZE;
use cairo_air::components::verify_bitwise_xor_8::LOG_SIZE as XOR8_LOG_SIZE;
use cairo_air::components::verify_bitwise_xor_9::LOG_SIZE as XOR9_LOG_SIZE;
use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{blake_witness as device, BaseFieldVec, CudaBackend};
use stwo_constraint_framework::LogupFinalizeBackend;

use crate::witness::components::{
    blake_g, verify_bitwise_xor_12, verify_bitwise_xor_4, verify_bitwise_xor_7,
    verify_bitwise_xor_8, verify_bitwise_xor_9,
};
use crate::witness::exec_context::WitnessExecContext;
use crate::witness::prelude::Mutex;

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// Number of committed base-trace columns (cairo-air blake_g N_TRACE_COLUMNS).
const BG_N_TRACE: usize = N_TRACE_COLUMNS; // 53

// The blake_g relation ids, hardcoded in the generated `write_trace_simd`
// (`M31_112558620` etc.). Kept here as the device kernel params; the differential
// gate and the e2e proof byte-equality catch any AIR-regeneration drift.
const REL_XOR8: u32 = 112558620;
const REL_XOR8B: u32 = 521092554;
const REL_XOR12: u32 = 648362599;
const REL_XOR4: u32 = 45448144;
const REL_XOR7: u32 = 62225763;
const REL_XOR9: u32 = 95781001;
const REL_BLAKE_G: u32 = 1139985212;

/// Backend hook for the `blake_g` witness (base trace + xor feeds + finalized logup
/// interaction trace). See the module docs.
pub trait BlakeGWitness: FromSimdColumns + LogupFinalizeBackend {
    /// Backend-resident state carried from the base-trace write to the interaction
    /// write (replaces the component's host lookup-data flow).
    type InteractionGen: Send;

    /// Writes the blake_g base trace on `Self`, feeds the five verify_bitwise_xor
    /// multiplicity families, and returns the claim plus the interaction state.
    /// Trace/claim bytes must be identical to the host writer's.
    fn write_trace(
        exec_context: &WitnessExecContext,
        gen: blake_g::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        xor12: &verify_bitwise_xor_12::ClaimGenerator,
        xor4: &verify_bitwise_xor_4::ClaimGenerator,
        xor7: &verify_bitwise_xor_7::ClaimGenerator,
        xor9: &verify_bitwise_xor_9::ClaimGenerator,
    ) -> (Evals<Self>, BlakeGClaim, Self::InteractionGen);

    /// Writes and FINALIZES the logup interaction trace on `Self`. Returns the 9
    /// finalized interaction columns and the claimed sum.
    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Evals<Self>, SecureField);
}

impl BlakeGWitness for SimdBackend {
    type InteractionGen = blake_g::InteractionClaimGenerator;

    fn write_trace(
        _exec_context: &WitnessExecContext,
        gen: blake_g::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        xor12: &verify_bitwise_xor_12::ClaimGenerator,
        xor4: &verify_bitwise_xor_4::ClaimGenerator,
        xor7: &verify_bitwise_xor_7::ClaimGenerator,
        xor9: &verify_bitwise_xor_9::ClaimGenerator,
    ) -> (Evals<Self>, BlakeGClaim, Self::InteractionGen) {
        let (trace, claim, interaction_gen) = gen.write_trace(xor8, xor12, xor4, xor7, xor9);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Evals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(elements);
        raw.finalize_on_simd()
    }
}

/// Device-resident interaction state for `blake_g`: the 73 base+aux columns.
pub struct DeviceBlakeGWitness {
    cols: Vec<BaseFieldVec>,
    log_size: u32,
    /// Host lookup-data replica, present only under `STWO_CUDA_WITNESS_VERIFY=1`.
    verify_host: Option<blake_g::InteractionClaimGenerator>,
}

/// CudaBackend interaction state: device lane, or the host fallback.
pub enum CudaBlakeGInteractionGen {
    Device(DeviceBlakeGWitness),
    Host(blake_g::InteractionClaimGenerator),
}

impl BlakeGWitness for CudaBackend {
    type InteractionGen = CudaBlakeGInteractionGen;

    fn write_trace(
        exec_context: &WitnessExecContext,
        gen: blake_g::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        xor12: &verify_bitwise_xor_12::ClaimGenerator,
        xor4: &verify_bitwise_xor_4::ClaimGenerator,
        xor7: &verify_bitwise_xor_7::ClaimGenerator,
        xor9: &verify_bitwise_xor_9::ClaimGenerator,
    ) -> (Evals<Self>, BlakeGClaim, Self::InteractionGen) {
        if !device_lane_enabled() {
            // The blake_round producer may have stashed the edge and skipped
            // the host blake_g feed; rebuild the inputs from the stashed HOST
            // flat before the host writer runs (exactly-once feeds).
            if let Some(edge) = exec_context.take_edge("blake_round", "blake_g") {
                crate::witness::components::blake_round::feed_blake_g_inputs_from_flat(
                    &edge.host_flat,
                    edge.n_rows,
                    &gen,
                );
            }
            let (trace, claim, interaction_gen) = gen.write_trace(xor8, xor12, xor4, xor7, xor9);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaBlakeGInteractionGen::Host(interaction_gen),
            );
        }

        // B3 edge consumer: blake_round's lane stashed its device sub buffer
        // (and skipped the host blake_g feed). Build the row-major input buffer
        // straight from it; any failure rebuilds the generator's inputs on CPU
        // from the stashed HOST flat and falls through to the host-built path.
        let mut edge_inputs: Option<(stwo_backend_cuda::BaseFieldVec, usize)> = None;
        if let Some(edge) = exec_context.take_edge("blake_round", "blake_g") {
            let plan = edge.plan;
            assert_eq!(
                plan.words_per_instance, 6,
                "blake_g edge ABI requires six words per instance"
            );
            let n_rows_edge = plan.n_instances as usize * edge.n_rows;
            let column_length = std::cmp::max(n_rows_edge.next_power_of_two(), N_LANES);
            let out = stwo_backend_cuda::BaseFieldVec::new_zeroes(
                column_length * plan.words_per_instance as usize,
            );
            let rc = unsafe {
                stwo_backend_cuda_kernels::raw::stwo_blake_g_inputs_from_sub(
                    edge.buffer.device_ptr,
                    edge.n_rows as u32,
                    plan.word_base,
                    plan.n_instances,
                    column_length as u32,
                    out.device_ptr.cast_mut(),
                )
            };
            if rc == 0 {
                edge_inputs = Some((out, n_rows_edge));
            } else {
                eprintln!(
                    "jit_prove[blake_g]: device edge interleave failed — rebuilding                      inputs from the stashed host flat"
                );
                crate::witness::components::blake_round::feed_blake_g_inputs_from_flat(
                    &edge.host_flat,
                    edge.n_rows,
                    &gen,
                );
            }
        }
        if let Some((inputs_dev, n_rows)) = edge_inputs {
            let column_length = inputs_dev.len() / 6;
            let log_size = column_length.ilog2();
            let cols = device::write_trace(&inputs_dev, n_rows, column_length);
            feed_xor_counts(&cols, column_length, xor8, xor12, xor4, xor7, xor9);
            let domain = CanonicCoset::new(log_size).circle_domain();
            let trace: Evals<Self> = cols[..BG_N_TRACE]
                .iter()
                .map(|c| CircleEvaluation::new(domain, c.clone()))
                .collect();
            let claim = BlakeGClaim { log_size };
            return (
                trace,
                claim,
                CudaBlakeGInteractionGen::Device(DeviceBlakeGWitness {
                    cols,
                    log_size,
                    verify_host: None,
                }),
            );
        }

        let packed_inputs = gen.packed_inputs.into_inner().unwrap();
        assert!(!packed_inputs.is_empty());
        assert!(gen.remainder_inputs.into_inner().unwrap().is_empty());
        let n_vec_rows = packed_inputs.len();
        let n_rows = n_vec_rows * N_LANES;
        let packed_size = n_vec_rows.next_power_of_two();
        let log_size = packed_size.ilog2() + LOG_N_LANES;
        let column_length = packed_size * N_LANES;

        let verify = witness_verify_enabled();
        let host_inputs = verify.then(|| packed_inputs.clone());

        // Pad exactly like the host writer (replicate the first input).
        let mut padded = packed_inputs;
        padded.resize(packed_size, *padded.first().unwrap());

        // Row-major input buffer: for each padded packed row, each lane, the 6 raw
        // u32 words (row = packed_row * N_LANES + lane — the trace's flat layout).
        let mut words: Vec<BaseField> = Vec::with_capacity(column_length * 6);
        for packed in &padded {
            let arrs: [[u32; N_LANES]; 6] = std::array::from_fn(|j| packed[j].simd.to_array());
            for lane in 0..N_LANES {
                for arr in &arrs {
                    words.push(BaseField::from_u32_unchecked(arr[lane]));
                }
            }
        }
        let inputs_dev = BaseFieldVec::from_vec(words);
        let cols = device::write_trace(&inputs_dev, n_rows, column_length);

        // In verify mode the host writer feeds the real xor generators (below); in
        // production the device count tables are merged instead.
        if !verify {
            feed_xor_counts(&cols, column_length, xor8, xor12, xor4, xor7, xor9);
        }

        // Committed base trace: D2D clones of cols[0..53] (originals stay in state).
        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace: Evals<Self> = cols[..BG_N_TRACE]
            .iter()
            .map(|c| CircleEvaluation::new(domain, c.clone()))
            .collect();
        let claim = BlakeGClaim { log_size };

        let verify_host = host_inputs.map(|inp| {
            let clone_gen = blake_g::ClaimGenerator {
                packed_inputs: Mutex::new(inp),
                remainder_inputs: Mutex::new(vec![]),
            };
            let (host_trace, _host_claim, host_ig) =
                clone_gen.write_trace(xor8, xor12, xor4, xor7, xor9);
            verify_trace_columns(&cols[..BG_N_TRACE], host_trace.to_evals());
            host_ig
        });

        (
            trace,
            claim,
            CudaBlakeGInteractionGen::Device(DeviceBlakeGWitness {
                cols,
                log_size,
                verify_host,
            }),
        )
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Evals<Self>, SecureField) {
        match gen {
            CudaBlakeGInteractionGen::Host(interaction_gen) => {
                let (raw, _build_claim) = interaction_gen.write_interaction_trace(elements);
                <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw)
            }
            CudaBlakeGInteractionGen::Device(state) => {
                let z = elements.z();
                let alphas = elements.alpha_powers();
                let len = 1usize << state.log_size;
                let c = |i: usize| &state.cols[i];
                let pair = |a0, b0, x0, a1, b1, x1, r0, r1| {
                    device::pair_logup(
                        c(a0),
                        c(b0),
                        c(x0),
                        c(a1),
                        c(b1),
                        c(x1),
                        r0,
                        r1,
                        len,
                        alphas,
                        z,
                    )
                };
                let columns = vec![
                    pair(53, 55, 18, 14, 16, 19, REL_XOR8, REL_XOR8),
                    pair(54, 56, 20, 15, 17, 21, REL_XOR8B, REL_XOR8B),
                    pair(57, 59, 28, 24, 26, 29, REL_XOR12, REL_XOR4),
                    pair(58, 60, 30, 25, 27, 31, REL_XOR12, REL_XOR4),
                    pair(61, 63, 38, 34, 36, 39, REL_XOR8, REL_XOR8),
                    pair(62, 64, 40, 35, 37, 41, REL_XOR8B, REL_XOR8B),
                    pair(65, 67, 48, 44, 46, 49, REL_XOR7, REL_XOR9),
                    pair(66, 68, 50, 45, 47, 51, REL_XOR7, REL_XOR9),
                    {
                        let vals: Vec<&BaseFieldVec> = [
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 32, 33, 69, 70, 42, 43, 71, 72,
                        ]
                        .into_iter()
                        .map(c)
                        .collect();
                        device::final_logup(&vals, c(52), REL_BLAKE_G, len, alphas, z)
                    },
                ];
                let (trace, claimed_sum) =
                    device::finalize_device_raw_logup(state.log_size, columns);

                if let Some(host_ig) = state.verify_host {
                    verify_interaction(host_ig, elements, &trace, claimed_sum);
                }
                (trace, claimed_sum)
            }
        }
    }
}

fn device_lane_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_BLAKE_WITNESS").as_deref() != Ok("0")
}

fn witness_verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

/// Feeds the five verify_bitwise_xor families from the device operand columns.
#[allow(clippy::too_many_arguments)]
fn feed_xor_counts(
    cols: &[BaseFieldVec],
    column_length: usize,
    xor8: &verify_bitwise_xor_8::ClaimGenerator,
    xor12: &verify_bitwise_xor_12::ClaimGenerator,
    xor4: &verify_bitwise_xor_4::ClaimGenerator,
    xor7: &verify_bitwise_xor_7::ClaimGenerator,
    xor9: &verify_bitwise_xor_9::ClaimGenerator,
) {
    let c = |i: usize| &cols[i];

    // xor_8: 2 relations (8 and 8_b). a/b operands (xor result unused for count).
    let a8 = [c(53), c(14), c(61), c(34), c(54), c(15), c(62), c(35)];
    let b8 = [c(55), c(16), c(63), c(36), c(56), c(17), c(64), c(37)];
    let rel8 = [0u32, 0, 0, 0, 1, 1, 1, 1];
    let counts8 = device::xor_count(
        &a8,
        &b8,
        &rel8,
        column_length,
        8,
        &xor8.input_to_row_lut(),
        2,
        1 << XOR8_LOG_SIZE,
    );
    xor8.add_count_tables(&counts8);

    // xor_4.
    let counts4 = device::xor_count(
        &[c(24), c(25)],
        &[c(26), c(27)],
        &[0, 0],
        column_length,
        4,
        &xor4.input_to_row_lut(),
        1,
        1 << XOR4_LOG_SIZE,
    );
    xor4.add_count_tables(&counts4);

    // xor_7.
    let counts7 = device::xor_count(
        &[c(65), c(66)],
        &[c(67), c(68)],
        &[0, 0],
        column_length,
        7,
        &xor7.input_to_row_lut(),
        1,
        1 << XOR7_LOG_SIZE,
    );
    xor7.add_count_tables(&counts7);

    // xor_9.
    let counts9 = device::xor_count(
        &[c(44), c(45)],
        &[c(46), c(47)],
        &[0, 0],
        column_length,
        9,
        &xor9.input_to_row_lut(),
        1,
        1 << XOR9_LOG_SIZE,
    );
    xor9.add_count_tables(&counts9);

    // xor_12: expanded table, closed-form indexing.
    let counts12 = device::xor12_count(
        &[c(57), c(58)],
        &[c(59), c(60)],
        column_length,
        XOR12_LIMB_BITS,
        XOR12_EXPAND_BITS,
        XOR12_N_MULT,
        1 << XOR12_LOG_SIZE,
    );
    xor12.add_count_tables(&counts12);
}

// --- STWO_CUDA_WITNESS_VERIFY differential legs (qualification harness). ---

fn verify_trace_columns(device_cols: &[BaseFieldVec], host_trace: Evals<SimdBackend>) {
    assert_eq!(
        device_cols.len(),
        host_trace.len(),
        "STWO_CUDA_WITNESS_VERIFY: blake_g trace column count mismatch"
    );
    let mut ok = true;
    for (col_idx, (device_col, host_col)) in device_cols.iter().zip(&host_trace).enumerate() {
        ok &= compare_values(
            &format!("blake_g trace col {col_idx}"),
            &host_col.values.to_cpu(),
            &device_col.to_cpu(),
        );
    }
    if !ok {
        panic!("STWO_CUDA_WITNESS_VERIFY: blake_g trace mismatch (report above)");
    }
}

fn verify_interaction(
    host_gen: blake_g::InteractionClaimGenerator,
    elements: &CommonLookupElements,
    device_trace: &Evals<CudaBackend>,
    device_sum: SecureField,
) {
    let (raw, _build_claim) = host_gen.write_interaction_trace(elements);
    let (host_trace, host_sum) = raw.finalize_on_simd();
    let mut ok = host_trace.len() == device_trace.len();
    if !ok {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY mismatch: blake_g interaction column count host {} vs device {}",
            host_trace.len(),
            device_trace.len()
        );
    }
    for (col_idx, (host_col, device_col)) in host_trace.iter().zip(device_trace).enumerate() {
        ok &= compare_values(
            &format!("blake_g interaction col {col_idx}"),
            &host_col.values.to_cpu(),
            &device_col.values.to_cpu(),
        );
    }
    if host_sum != device_sum {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY mismatch: blake_g claimed sum host {host_sum:?} device {device_sum:?}"
        );
        ok = false;
    }
    if !ok {
        panic!("STWO_CUDA_WITNESS_VERIFY: blake_g interaction mismatch (report above)");
    }
}

fn compare_values(label: &str, host: &[BaseField], device: &[BaseField]) -> bool {
    if host == device {
        return true;
    }
    if host.len() != device.len() {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY mismatch: {label}: length host {} vs device {}",
            host.len(),
            device.len()
        );
        return false;
    }
    let n = host.iter().zip(device).filter(|(h, d)| h != d).count();
    let first = host.iter().zip(device).position(|(h, d)| h != d).unwrap();
    eprintln!(
        "STWO_CUDA_WITNESS_VERIFY mismatch: {label}: {n} rows differ; first at row \
         {first}: host {} device {}",
        host[first], device[first]
    );
    false
}
