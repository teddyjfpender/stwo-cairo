//! Backend-specific witness generation for the `memory_id_to_big` component —
//! witness-on-GPU P1 phase 1 (see `gpu_benchmarks/WITNESS_ON_GPU.md` W3 and
//! `gpu_benchmarks/ROAD_TO_10MHZ.md` P1).
//!
//! [`MemoryIdToBigWitness`] gives the component two interchangeable paths:
//!
//! - **`SimdBackend` (host)**: delegates to the existing generated writers in
//!   `components/memory_id_to_big.rs` plus the raw-logup SIMD finalize — byte-identical to the
//!   pre-hook flow by construction (it IS that flow).
//! - **`CudaBackend` (device)**: the dedup'd f252 / u128 value tables upload once; the 28/8 9-bit
//!   limb columns, the rc_9_9 multiplicity counts, and the logup interaction columns are all born
//!   on device via the `stwo-backend-cuda` `memory_witness` kernels. Only the 8 rc count tables
//!   come back to the host (merged into the rc_9_9 generator's atomic multiplicities —
//!   order-independent adds, so byte-equality is preserved by construction).
//!
//! Load-bearing contracts of the device path (from the adversarial kernel review):
//!
//! 1. The rc_9_9 input -> row LUT is the inversion of the ACTUAL rc generator's `input_to_row` map
//!    (a table-layout mapping, never a closed form).
//! 2. Segmentation mirrors `gen_big_memory_traces` exactly: chunks of `1 << log_max_big_size`,
//!    per-segment `column_length = values_chunk.len().next_power_of_two()` (min `N_LANES`), the id
//!    offset advancing by each segment's PADDED length, and `opt_n_components` padding segments
//!    running the same kernels with `n_values = 0` so their rc counts ((0, 0) pairs) and logup
//!    columns are produced like the host's.
//! 3. Interaction column order per big segment: 7 rc-pair columns (limb quads `4i..4i+3`,
//!    relation-id pairs by `i % 4`), then the final memory column (`MEMORY_ID_TO_BIG_RELATION_ID`,
//!    segment id offset, the `LARGE_MEMORY_VALUE_ID_BASE` tag, 28 limbs). Small: 2 pair columns,
//!    then the final memory column (id offset 0, tag 0, 8 limbs).
//! 4. The base-trace columns handed to the tree are device CLONES (D2D copies) of the
//!    limb/multiplicity buffers; the originals stay in the interaction state, so the commit
//!    pipeline never aliases the interaction inputs.
//!
//! Gates: `STWO_CUDA_WITNESS_VERIFY=1` runs the host writers on cloned inputs and
//! byte-compares every trace column, the rc count tables, and the interaction
//! columns + claimed sums (panicking on mismatch — a qualification harness), and
//! the Cairo e2e proof byte-equality test is the global gate. The device lane
//! falls back to the host path per component when the CUDA kernels are not built
//! or `STWO_CUDA_MEMORY_WITNESS=0`.

use cairo_air::components::memory_id_to_big::Claim as BigClaim;
use cairo_air::components::memory_id_to_small::Claim as SmallClaim;
use cairo_air::components::range_check_9_9::LOG_SIZE as RC99_LOG_SIZE;
use cairo_air::relations::{
    CommonLookupElements, MEMORY_ID_TO_BIG_RELATION_ID, RANGE_CHECK_9_9_B_RELATION_ID,
    RANGE_CHECK_9_9_C_RELATION_ID, RANGE_CHECK_9_9_D_RELATION_ID, RANGE_CHECK_9_9_E_RELATION_ID,
    RANGE_CHECK_9_9_F_RELATION_ID, RANGE_CHECK_9_9_G_RELATION_ID, RANGE_CHECK_9_9_H_RELATION_ID,
    RANGE_CHECK_9_9_RELATION_ID,
};
use num_traits::Zero;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{memory_witness as device, BaseFieldVec, CudaBackend};
use stwo_cairo_adapter::memory::u128_to_4_limbs;
use stwo_cairo_common::memory::{LARGE_MEMORY_VALUE_ID_BASE, N_M31_IN_SMALL_FELT252};
use stwo_cairo_common::prover_types::cpu::FELT252_N_WORDS;
use stwo_constraint_framework::LogupFinalizeBackend;

use crate::witness::components::{memory_id_to_big, range_check_9_9};

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// The number of little-endian u32 words per f252 value in the adapter's table.
const F252_N_WORDS: usize = 8;

/// The rc_9_9 relation-id pairs in pair-column order: column `i` of a segment
/// combines `(rel_ids[i % 4].0, limb_{4i}, limb_{4i+1})` and
/// `(rel_ids[i % 4].1, limb_{4i+2}, limb_{4i+3})` — exactly the host writer's
/// `match i % 4` (big) and `match i % 2` (small, which only reaches the first two).
const RC99_PAIR_RELATION_IDS: [(M31, M31); 4] = [
    (RANGE_CHECK_9_9_RELATION_ID, RANGE_CHECK_9_9_B_RELATION_ID),
    (RANGE_CHECK_9_9_C_RELATION_ID, RANGE_CHECK_9_9_D_RELATION_ID),
    (RANGE_CHECK_9_9_E_RELATION_ID, RANGE_CHECK_9_9_F_RELATION_ID),
    (RANGE_CHECK_9_9_G_RELATION_ID, RANGE_CHECK_9_9_H_RELATION_ID),
];

/// `(big segment traces, small trace, (big claim, small claim), interaction
/// state)` — the result of [`MemoryIdToBigWitness::write_trace`].
pub type MemoryTraceResult<B> = (
    Vec<Evals<B>>,
    Evals<B>,
    (BigClaim, SmallClaim),
    <B as MemoryIdToBigWitness>::InteractionGen,
);

/// Backend hook for the `memory_id_to_big` witness (base trace + rc_9_9 feed +
/// finalized logup interaction trace). See the module docs.
pub trait MemoryIdToBigWitness: FromSimdColumns + LogupFinalizeBackend {
    /// Backend-resident state carried from the base-trace write to the
    /// interaction write (replaces the component's `RawLogupTrace` flow).
    type InteractionGen: Send;

    /// Writes the big/small memory base traces on `Self`, feeds the rc_9_9
    /// multiplicities, and returns the claims plus the interaction state.
    /// Trace/claim bytes must be identical to the host writer's.
    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self>;

    /// Writes and FINALIZES the logup interaction trace on `Self`. Returns
    /// `(big segment traces in segment order, small trace, big claimed sums in
    /// segment order, small claimed sum)`.
    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Vec<Evals<Self>>, Evals<Self>, Vec<SecureField>, SecureField);
}

impl MemoryIdToBigWitness for SimdBackend {
    type InteractionGen = memory_id_to_big::InteractionClaimGenerator;

    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self> {
        // The pre-hook flow, verbatim: byte-identical by construction.
        gen.write_trace(rc99, log_max_big_size, opt_n_components)
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Vec<Evals<Self>>, Evals<Self>, Vec<SecureField>, SecureField) {
        let (big_raws, small_raw, _build_big_claim, _build_small_claim) =
            gen.write_interaction_trace(elements);
        let mut big_traces = Vec::with_capacity(big_raws.len());
        let mut big_claimed_sums = Vec::with_capacity(big_raws.len());
        for raw in big_raws {
            let (trace, claimed_sum) = raw.finalize_on_simd();
            big_traces.push(trace);
            big_claimed_sums.push(claimed_sum);
        }
        let (small_trace, small_claimed_sum) = small_raw.finalize_on_simd();
        (big_traces, small_trace, big_claimed_sums, small_claimed_sum)
    }
}

/// One device-resident memory table segment: the limb columns and the
/// multiplicity column (the ORIGINALS — the committed trace got D2D clones),
/// plus the segment geometry the interaction writer needs.
pub struct DeviceMemorySegment {
    limbs: Vec<BaseFieldVec>,
    mults: BaseFieldVec,
    log_size: u32,
    id_offset: u32,
}

/// Device-resident interaction state for `memory_id_to_big`.
pub struct DeviceMemoryWitness {
    big_segments: Vec<DeviceMemorySegment>,
    small: DeviceMemorySegment,
    /// Host replica of the lookup data, present only under
    /// `STWO_CUDA_WITNESS_VERIFY=1` for the interaction differential.
    verify_host: Option<memory_id_to_big::InteractionClaimGenerator>,
}

/// CudaBackend interaction state: device lane, or the host fallback (kernels
/// not built / `STWO_CUDA_MEMORY_WITNESS=0`).
pub enum CudaMemoryInteractionGen {
    Device(DeviceMemoryWitness),
    Host(memory_id_to_big::InteractionClaimGenerator),
}

impl MemoryIdToBigWitness for CudaBackend {
    type InteractionGen = CudaMemoryInteractionGen;

    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self> {
        if !device_lane_enabled() {
            // Host fallback: today's writer, bridged via `from_simd_evals`.
            let (big_traces, small_trace, claims, interaction_gen) =
                gen.write_trace(rc99, log_max_big_size, opt_n_components);
            return (
                big_traces.into_iter().map(Self::from_simd_evals).collect(),
                Self::from_simd_evals(small_trace),
                claims,
                CudaMemoryInteractionGen::Host(interaction_gen),
            );
        }

        let (big_values, big_mults, small_values, small_mults) = gen.into_parts();
        let host_inputs = witness_verify_enabled().then(|| {
            (
                big_values.clone(),
                big_mults.clone(),
                small_values.clone(),
                small_mults.clone(),
            )
        });

        let rc_table_size = 1usize << RC99_LOG_SIZE;
        let rc_lut = rc99.input_to_row_lut();
        let mut rc_counts = vec![0u32; 8 * rc_table_size];

        // --- Big segments: chunked exactly like `gen_big_memory_traces`.
        assert!(log_max_big_size >= LOG_N_LANES);
        let max_big_size = 1usize << log_max_big_size;
        assert_eq!(big_values.len() / N_LANES, big_mults.len());
        let mut big_segments = Vec::new();
        let mut id_offset = 0u32;
        for (values_chunk, mults_chunk) in big_values
            .chunks(max_big_size)
            .zip(big_mults.chunks(max_big_size / N_LANES))
        {
            let n_values = values_chunk.len();
            assert_eq!(n_values, mults_chunk.len() * N_LANES);
            let column_length = n_values.next_power_of_two().max(N_LANES);
            let words: Vec<BaseField> = values_chunk
                .iter()
                .flat_map(|value| value.iter().map(|&w| BaseField::from_u32_unchecked(w)))
                .collect();
            let values_dev = BaseFieldVec::from_vec(words);
            let limbs = device::limb_split_big(&values_dev, n_values, column_length);
            accumulate_counts(
                &mut rc_counts,
                &device::rc99_count(&limbs, column_length, &rc_lut, rc_table_size),
            );
            let mults = upload_mults(mults_chunk, column_length);
            big_segments.push(DeviceMemorySegment {
                limbs,
                mults,
                log_size: column_length.ilog2(),
                id_offset,
            });
            // The id offset advances by the PADDED segment length, as the host does
            // (`offset += big_multiplicities.len() * N_LANES` over the padded column).
            id_offset += column_length as u32;
        }
        if let Some(n_components) = opt_n_components {
            assert!(n_components >= big_segments.len());
            for _ in big_segments.len()..n_components {
                // Padding segments (all-zero, length N_LANES) run the same kernels
                // with n_values = 0, so their rc counts ((0, 0) pairs) and logup
                // columns are produced exactly like the host's padding traces.
                let values_dev = BaseFieldVec::new_zeroes(F252_N_WORDS);
                let limbs = device::limb_split_big(&values_dev, 0, N_LANES);
                accumulate_counts(
                    &mut rc_counts,
                    &device::rc99_count(&limbs, N_LANES, &rc_lut, rc_table_size),
                );
                big_segments.push(DeviceMemorySegment {
                    limbs,
                    mults: BaseFieldVec::new_zeroes(N_LANES),
                    log_size: LOG_N_LANES,
                    id_offset,
                });
                id_offset += N_LANES as u32;
            }
        }

        // --- Small table (single segment; id offset 0, no tag).
        assert_eq!(small_values.len(), small_mults.len() * N_LANES);
        let small_n = small_values.len();
        let small_column_length = small_n.next_power_of_two();
        let words: Vec<BaseField> = small_values
            .iter()
            .flat_map(|&value| u128_to_4_limbs(value).map(BaseField::from_u32_unchecked))
            .collect();
        let values_dev = BaseFieldVec::from_vec(words);
        let small_limbs = device::limb_split_small(&values_dev, small_n, small_column_length);
        accumulate_counts(
            &mut rc_counts,
            &device::rc99_count(&small_limbs, small_column_length, &rc_lut, rc_table_size),
        );
        let small = DeviceMemorySegment {
            limbs: small_limbs,
            mults: upload_mults(&small_mults, small_column_length),
            log_size: small_column_length.ilog2(),
            id_offset: 0,
        };

        // --- Differential gate (trace + rc-count legs); also builds the host
        // lookup-data replica for the deferred interaction leg.
        let verify_host = host_inputs.map(|(bv, bm, sv, sm)| {
            let host_big =
                memory_id_to_big::gen_big_memory_traces(bv, bm, log_max_big_size, opt_n_components);
            let host_small = memory_id_to_big::gen_small_memory_trace(sv, sm);
            verify_trace_columns(&big_segments, &small, &host_big, &host_small);
            verify_rc_counts(&rc_counts, &host_big, &host_small, &rc_lut, rc_table_size);
            host_interaction_gen(&host_big, &host_small)
        });

        // Merge the device counts into the rc generator's multiplicities. The
        // caller sequences this before range_check_9_9 writes its trace.
        rc99.add_count_tables(&rc_counts);

        // --- Assemble the committed trace (D2D clones; originals stay here) and
        // the claims, exactly as the host builds them.
        let big_traces: Vec<Evals<Self>> = big_segments.iter().map(segment_trace).collect();
        let small_trace = segment_trace(&small);
        let big_claim = BigClaim {
            big_log_sizes: big_segments.iter().map(|seg| seg.log_size).collect(),
        };
        let small_claim = SmallClaim {
            log_size: small.log_size,
        };

        (
            big_traces,
            small_trace,
            (big_claim, small_claim),
            CudaMemoryInteractionGen::Device(DeviceMemoryWitness {
                big_segments,
                small,
                verify_host,
            }),
        )
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        elements: &CommonLookupElements,
    ) -> (Vec<Evals<Self>>, Evals<Self>, Vec<SecureField>, SecureField) {
        match gen {
            CudaMemoryInteractionGen::Host(interaction_gen) => {
                // Host fallback: raw writes on the host, finalize on device (the
                // existing W1 lane).
                let (big_raws, small_raw, _build_big_claim, _build_small_claim) =
                    interaction_gen.write_interaction_trace(elements);
                let mut big_traces = Vec::with_capacity(big_raws.len());
                let mut big_claimed_sums = Vec::with_capacity(big_raws.len());
                for raw in big_raws {
                    let (trace, claimed_sum) =
                        <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
                    big_traces.push(trace);
                    big_claimed_sums.push(claimed_sum);
                }
                let (small_trace, small_claimed_sum) =
                    <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(small_raw);
                (big_traces, small_trace, big_claimed_sums, small_claimed_sum)
            }
            CudaMemoryInteractionGen::Device(state) => {
                let z = elements.z();
                let alphas = elements.alpha_powers();
                let mut big_traces = Vec::with_capacity(state.big_segments.len());
                let mut big_claimed_sums = Vec::with_capacity(state.big_segments.len());
                for seg in &state.big_segments {
                    let (trace, claimed_sum) =
                        segment_interaction(seg, alphas, z, LARGE_MEMORY_VALUE_ID_BASE);
                    big_traces.push(trace);
                    big_claimed_sums.push(claimed_sum);
                }
                let (small_trace, small_claimed_sum) =
                    segment_interaction(&state.small, alphas, z, 0);

                if let Some(host_gen) = state.verify_host {
                    verify_interaction(
                        host_gen,
                        elements,
                        &big_traces,
                        &big_claimed_sums,
                        &small_trace,
                        small_claimed_sum,
                    );
                }

                (big_traces, small_trace, big_claimed_sums, small_claimed_sum)
            }
        }
    }
}

fn device_lane_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_MEMORY_WITNESS").as_deref() != Ok("0")
}

fn witness_verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

/// Pads the multiplicities to `column_length / N_LANES` packed entries (like
/// `gen_single_big_memory_trace`) and uploads them.
fn upload_mults(mults: &[PackedM31], column_length: usize) -> BaseFieldVec {
    let mut padded = mults.to_vec();
    padded.resize(column_length / N_LANES, PackedM31::zero());
    let host: Vec<BaseField> = padded.iter().flat_map(|packed| packed.to_array()).collect();
    BaseFieldVec::from_vec(host)
}

fn accumulate_counts(total: &mut [u32], delta: &[u32]) {
    assert_eq!(total.len(), delta.len());
    for (t, d) in total.iter_mut().zip(delta) {
        *t = t.wrapping_add(*d);
    }
}

/// The committed base-trace columns of one segment: D2D clones of the limb
/// columns, then the multiplicity column — the `gen_single_big_memory_trace` /
/// `gen_small_memory_trace` column order.
fn segment_trace(seg: &DeviceMemorySegment) -> Evals<CudaBackend> {
    let domain = CanonicCoset::new(seg.log_size).circle_domain();
    seg.limbs
        .iter()
        .chain(std::iter::once(&seg.mults))
        .map(|col| CircleEvaluation::new(domain, col.clone()))
        .collect()
}

/// One segment's finalized logup trace: the rc-pair columns over limb quads,
/// then the memory-relation column, finalized on device.
fn segment_interaction(
    seg: &DeviceMemorySegment,
    alphas: &[SecureField],
    z: SecureField,
    id_tag: u32,
) -> (Evals<CudaBackend>, SecureField) {
    let column_length = 1usize << seg.log_size;
    let n_limbs = seg.limbs.len();
    let mut columns = Vec::with_capacity(n_limbs / 4 + 1);
    for (i, quad) in seg.limbs.chunks_exact(4).enumerate() {
        let (rel_id0, rel_id1) = RC99_PAIR_RELATION_IDS[i % 4];
        columns.push(device::memory_rc_pair_logup(
            [&quad[0], &quad[1], &quad[2], &quad[3]],
            rel_id0.0,
            rel_id1.0,
            column_length,
            &alphas[..3],
            z,
        ));
    }
    columns.push(device::memory_logup_inputs(
        &seg.limbs,
        &seg.mults,
        MEMORY_ID_TO_BIG_RELATION_ID.0,
        seg.id_offset,
        id_tag,
        column_length,
        &alphas[..n_limbs + 2],
        z,
    ));
    device::finalize_device_raw_logup(seg.log_size, columns)
}

/// Builds the host `InteractionClaimGenerator` from host traces — the same
/// lookup-data extraction as the host `write_trace`.
fn host_interaction_gen(
    host_big: &[Vec<BaseColumn>],
    host_small: &[BaseColumn],
) -> memory_id_to_big::InteractionClaimGenerator {
    memory_id_to_big::InteractionClaimGenerator {
        big_components_values: host_big
            .iter()
            .map(|trace| std::array::from_fn(|i| trace[i].data.clone()))
            .collect(),
        big_multiplicities: host_big
            .iter()
            .map(|trace| trace.last().unwrap().data.clone())
            .collect(),
        small_values: std::array::from_fn(|i| host_small[i].data.clone()),
        small_multiplicities: host_small.last().unwrap().data.clone(),
    }
}

// --- STWO_CUDA_WITNESS_VERIFY differential legs (qualification harness). ---

fn verify_trace_columns(
    big_segments: &[DeviceMemorySegment],
    small: &DeviceMemorySegment,
    host_big: &[Vec<BaseColumn>],
    host_small: &[BaseColumn],
) {
    let mut ok = true;
    assert_eq!(
        big_segments.len(),
        host_big.len(),
        "STWO_CUDA_WITNESS_VERIFY: big segment count mismatch"
    );
    for (seg_idx, (seg, host_trace)) in big_segments.iter().zip(host_big).enumerate() {
        ok &= compare_segment_trace(&format!("big[{seg_idx}]"), seg, host_trace);
    }
    ok &= compare_segment_trace("small", small, host_small);
    if !ok {
        panic!("STWO_CUDA_WITNESS_VERIFY: memory_id_to_big trace mismatch (report above)");
    }
}

fn compare_segment_trace(label: &str, seg: &DeviceMemorySegment, host: &[BaseColumn]) -> bool {
    let mut ok = true;
    assert_eq!(
        host.len(),
        seg.limbs.len() + 1,
        "STWO_CUDA_WITNESS_VERIFY: {label} column count mismatch"
    );
    for (col_idx, host_col) in host.iter().enumerate() {
        let device_col = if col_idx < seg.limbs.len() {
            &seg.limbs[col_idx]
        } else {
            &seg.mults
        };
        ok &= compare_values(
            &format!("{label} trace col {col_idx}"),
            &host_col.to_cpu(),
            &device_col.to_cpu(),
        );
    }
    ok
}

fn verify_rc_counts(
    device_counts: &[u32],
    host_big: &[Vec<BaseColumn>],
    host_small: &[BaseColumn],
    lut: &[u32],
    table_size: usize,
) {
    // The host-computed mults delta: what the host writer's rc_9_9 feed loops
    // would add, recomputed as order-independent counts through the same LUT.
    let mut host_counts = vec![0u32; 8 * table_size];
    let mut count_pairs = |cols: &[BaseColumn], n_relations: usize| {
        for (i, pair) in cols.chunks_exact(2).enumerate() {
            let relation = i % n_relations;
            let col0 = pair[0].to_cpu();
            let col1 = pair[1].to_cpu();
            for (v0, v1) in col0.into_iter().zip(col1) {
                let slot = &mut host_counts
                    [relation * table_size + lut[((v0.0 as usize) << 9) | v1.0 as usize] as usize];
                *slot = slot.wrapping_add(1);
            }
        }
    };
    for trace in host_big {
        count_pairs(&trace[..FELT252_N_WORDS], 8);
    }
    count_pairs(&host_small[..N_M31_IN_SMALL_FELT252], 4);

    let mut ok = true;
    for relation in 0..8 {
        let range = relation * table_size..(relation + 1) * table_size;
        let host = &host_counts[range.clone()];
        let dev = &device_counts[range];
        if host != dev {
            let n = host.iter().zip(dev).filter(|(h, d)| h != d).count();
            let first = host.iter().zip(dev).position(|(h, d)| h != d).unwrap();
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY mismatch: rc_9_9 counts relation {relation}: \
                 {n} rows differ; first at row {first}: host {} device {}",
                host[first], dev[first]
            );
            ok = false;
        }
    }
    if !ok {
        panic!("STWO_CUDA_WITNESS_VERIFY: memory_id_to_big rc_9_9 count mismatch (report above)");
    }
}

fn verify_interaction(
    host_gen: memory_id_to_big::InteractionClaimGenerator,
    elements: &CommonLookupElements,
    device_big: &[Evals<CudaBackend>],
    device_big_sums: &[SecureField],
    device_small: &Evals<CudaBackend>,
    device_small_sum: SecureField,
) {
    let (big_raws, small_raw, _build_big_claim, _build_small_claim) =
        host_gen.write_interaction_trace(elements);
    let mut ok = true;
    assert_eq!(
        big_raws.len(),
        device_big.len(),
        "STWO_CUDA_WITNESS_VERIFY: interaction segment count mismatch"
    );
    for (seg_idx, (raw, (dev_trace, dev_sum))) in big_raws
        .into_iter()
        .zip(device_big.iter().zip(device_big_sums))
        .enumerate()
    {
        let (host_trace, host_sum) = raw.finalize_on_simd();
        ok &= compare_interaction(
            &format!("big[{seg_idx}]"),
            &host_trace,
            host_sum,
            dev_trace,
            *dev_sum,
        );
    }
    let (host_small_trace, host_small_sum) = small_raw.finalize_on_simd();
    ok &= compare_interaction(
        "small",
        &host_small_trace,
        host_small_sum,
        device_small,
        device_small_sum,
    );
    if !ok {
        panic!("STWO_CUDA_WITNESS_VERIFY: memory_id_to_big interaction mismatch (report above)");
    }
}

fn compare_interaction(
    label: &str,
    host_trace: &Evals<SimdBackend>,
    host_sum: SecureField,
    device_trace: &Evals<CudaBackend>,
    device_sum: SecureField,
) -> bool {
    let mut ok = host_trace.len() == device_trace.len();
    if !ok {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY mismatch: {label}: interaction column count \
             host {} vs device {}",
            host_trace.len(),
            device_trace.len()
        );
    }
    for (col_idx, (host_col, device_col)) in host_trace.iter().zip(device_trace).enumerate() {
        ok &= compare_values(
            &format!("{label} interaction col {col_idx}"),
            &host_col.values.to_cpu(),
            &device_col.values.to_cpu(),
        );
    }
    if host_sum != device_sum {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY mismatch: {label}: claimed sum host {host_sum:?} \
             device {device_sum:?}"
        );
        ok = false;
    }
    ok
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
