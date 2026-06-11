//! Backend-specific witness generation for the `memory_id_to_big` component
//! (witness-on-GPU P1, the first vertical slice of W3).
//!
//! The trait mirrors the `GenPreprocessedTrace`/`LogupFinalizeBackend` seam style:
//! `SimdBackend` delegates verbatim to the existing host writers (today's bytes);
//! `CudaBackend` generates the component's base limb columns, rc_9_9 multiplicity
//! feed, and logup interaction columns entirely on device — no host columns, no
//! `lookup_data` — via the `stwo-backend-cuda` `memory_witness` lane.
//!
//! Every device formula is a port of the generated SIMD writer (see
//! `gpu_benchmarks/WITNESS_ON_GPU.md`). The authoritative gates:
//! - `STWO_CUDA_WITNESS_VERIFY=1` — run host and device writers, byte-compare all trace columns,
//!   the rc_9_9 count delta, and the interaction columns + sums; panic on any mismatch.
//! - the Cairo e2e proof byte-equality (CUDA vs SIMD).
//!
//! `STWO_CUDA_MEMORY_WITNESS=0` disables the device path per component (the
//! `PREPROCESSED_TRACE_GPU_GENERATE` pattern); the fallback is the host writer
//! bridged with `from_simd_evals`, byte-identical to the pre-P1 pipeline.

use cairo_air::components::memory_id_to_big::Claim as BigClaim;
use cairo_air::components::memory_id_to_small::Claim as SmallClaim;
use cairo_air::relations::{
    CommonLookupElements, MEMORY_ID_TO_BIG_RELATION_ID, RANGE_CHECK_9_9_B_RELATION_ID,
    RANGE_CHECK_9_9_C_RELATION_ID, RANGE_CHECK_9_9_D_RELATION_ID, RANGE_CHECK_9_9_E_RELATION_ID,
    RANGE_CHECK_9_9_F_RELATION_ID, RANGE_CHECK_9_9_G_RELATION_ID, RANGE_CHECK_9_9_H_RELATION_ID,
    RANGE_CHECK_9_9_RELATION_ID,
};
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{memory_witness as device_witness, BaseFieldVec, CudaBackend};
use stwo_cairo_adapter::memory::u128_to_4_limbs;
use stwo_cairo_common::memory::{LARGE_MEMORY_VALUE_ID_BASE, N_M31_IN_SMALL_FELT252};
use stwo_cairo_common::prover_types::cpu::FELT252_N_WORDS;
use stwo_constraint_framework::LogupFinalizeBackend;

use crate::witness::components::{memory_id_to_big, range_check_9_9};

pub type MemoryEvals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// The `write_trace` result: per-segment big traces, the small trace, the claims,
/// and the backend-specific interaction generator.
pub type MemoryTraceResult<B> = (
    Vec<MemoryEvals<B>>,
    MemoryEvals<B>,
    (BigClaim, SmallClaim),
    <B as MemoryIdToBigWitness>::InteractionGen,
);

/// Finalized interaction traces + claimed sums, in the eager extension order
/// (big segments in chunk order, then the small table).
pub struct MemoryInteractionResult<B: MemoryIdToBigWitness> {
    pub big: Vec<(MemoryEvals<B>, SecureField)>,
    pub small: (MemoryEvals<B>, SecureField),
}

/// Backend hook for the `memory_id_to_big` witness: base trace (+ rc_9_9 feed) and
/// the component's finalized interaction trace, both born on `Self`.
pub trait MemoryIdToBigWitness: FromSimdColumns {
    type InteractionGen: Send;

    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self>;

    fn write_interaction(
        gen: Self::InteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> MemoryInteractionResult<Self>;
}

impl MemoryIdToBigWitness for SimdBackend {
    type InteractionGen = memory_id_to_big::InteractionClaimGenerator;

    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self> {
        gen.write_trace(range_check_9_9, log_max_big_size, opt_n_components)
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> MemoryInteractionResult<Self> {
        let (big_raws, small_raw, _build_big_claim, _build_small_claim) =
            gen.write_interaction_trace(common_lookup_elements);
        MemoryInteractionResult {
            big: big_raws
                .into_iter()
                .map(|raw| raw.finalize_on_simd())
                .collect(),
            small: small_raw.finalize_on_simd(),
        }
    }
}

/// One device-resident memory segment: the limb columns and the multiplicity
/// column, all `column_length` long (the padded power-of-two length).
pub struct DeviceMemorySegment {
    limbs: Vec<BaseFieldVec>,
    mults: BaseFieldVec,
    column_length: usize,
}

/// Device-born witness state carried from `write_trace` to `write_interaction`.
pub struct DeviceMemoryWitness {
    big_segments: Vec<DeviceMemorySegment>,
    small_segment: DeviceMemorySegment,
    /// Host mirror, populated only under `STWO_CUDA_WITNESS_VERIFY=1`, used for
    /// the interaction-column differential.
    verify_host: Option<memory_id_to_big::InteractionClaimGenerator>,
}

pub enum CudaMemoryInteractionGen {
    Device(Box<DeviceMemoryWitness>),
    Host(Box<memory_id_to_big::InteractionClaimGenerator>),
}

fn device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_MEMORY_WITNESS").as_deref() != Ok("0")
}

fn verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

const RC_TABLE_SIZE: usize = 1 << cairo_air::components::range_check_9_9::LOG_SIZE;

/// The relation-id pairs of the pair-batched rc_9_9 logup columns, indexed by the
/// host writer's `i % 4` (big) / `i % 2` (small) match.
const PAIR_RELATION_IDS: [(M31, M31); 4] = [
    (RANGE_CHECK_9_9_RELATION_ID, RANGE_CHECK_9_9_B_RELATION_ID),
    (RANGE_CHECK_9_9_C_RELATION_ID, RANGE_CHECK_9_9_D_RELATION_ID),
    (RANGE_CHECK_9_9_E_RELATION_ID, RANGE_CHECK_9_9_F_RELATION_ID),
    (RANGE_CHECK_9_9_G_RELATION_ID, RANGE_CHECK_9_9_H_RELATION_ID),
];

impl MemoryIdToBigWitness for CudaBackend {
    type InteractionGen = CudaMemoryInteractionGen;

    fn write_trace(
        gen: memory_id_to_big::ClaimGenerator,
        range_check_9_9: &range_check_9_9::ClaimGenerator,
        log_max_big_size: u32,
        opt_n_components: Option<usize>,
    ) -> MemoryTraceResult<Self> {
        if !device_path_enabled() {
            let (big_traces, small_trace, claims, interaction_gen) =
                gen.write_trace(range_check_9_9, log_max_big_size, opt_n_components);
            return (
                big_traces.into_iter().map(Self::from_simd_evals).collect(),
                Self::from_simd_evals(small_trace),
                claims,
                CudaMemoryInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let (big_values, big_mults, small_values, small_mults) = gen.into_parts();
        let host_inputs = verify.then(|| {
            (
                big_values.clone(),
                big_mults.clone(),
                small_values.clone(),
                small_mults.clone(),
            )
        });

        // Segmentation mirror of `gen_big_memory_traces`: chunks of
        // `1 << log_max_big_size` values, each padded to its own power of two.
        assert!(log_max_big_size >= LOG_N_LANES);
        let max_big_size = 1usize << log_max_big_size;
        assert_eq!(big_values.len() / N_LANES, big_mults.len());

        let lut = range_check_9_9.input_to_row_lut();
        let mut rc_counts = vec![0u32; 8 * RC_TABLE_SIZE];
        let mut big_segments = Vec::new();
        let mut big_traces = Vec::new();
        let mut big_log_sizes = Vec::new();

        for (values_chunk, mults_chunk) in big_values
            .chunks(max_big_size)
            .zip(big_mults.chunks(max_big_size / N_LANES))
        {
            let column_length = values_chunk.len().next_power_of_two();
            let flat: Vec<BaseField> = values_chunk
                .iter()
                .flat_map(|words| words.iter().copied())
                .map(BaseField::from_u32_unchecked)
                .collect();
            let values_dev = BaseFieldVec::from_vec(flat);
            let limbs =
                device_witness::limb_split_big(&values_dev, values_chunk.len(), column_length);
            let mut mults_host: Vec<BaseField> = mults_chunk
                .iter()
                .flat_map(|packed| packed.to_array())
                .collect();
            mults_host.resize(column_length, BaseField::from_u32_unchecked(0));
            let mults_dev = BaseFieldVec::from_vec(mults_host);

            accumulate_counts(
                &mut rc_counts,
                &device_witness::rc99_count(&limbs, column_length, &lut, RC_TABLE_SIZE),
            );
            big_traces.push(segment_trace(&limbs, &mults_dev, column_length));
            big_log_sizes.push(column_length.ilog2());
            big_segments.push(DeviceMemorySegment {
                limbs,
                mults: mults_dev,
                column_length,
            });
        }

        // Padding segments (`opt_n_components`): all-zero columns of N_LANES rows.
        // They feed the rc counts and emit logup columns exactly like the host.
        if let Some(n_components) = opt_n_components {
            assert!(n_components >= big_segments.len());
            for _ in big_segments.len()..n_components {
                let column_length = N_LANES;
                // Placeholder allocation; the kernel reads no values when n_values=0.
                let values_dev = BaseFieldVec::new_zeroes(FELT252_N_WORDS);
                let limbs = device_witness::limb_split_big(&values_dev, 0, column_length);
                let mults_dev = BaseFieldVec::new_zeroes(column_length);
                accumulate_counts(
                    &mut rc_counts,
                    &device_witness::rc99_count(&limbs, column_length, &lut, RC_TABLE_SIZE),
                );
                big_traces.push(segment_trace(&limbs, &mults_dev, column_length));
                big_log_sizes.push(LOG_N_LANES);
                big_segments.push(DeviceMemorySegment {
                    limbs,
                    mults: mults_dev,
                    column_length,
                });
            }
        }

        // Small table.
        assert_eq!(small_values.len() / N_LANES, small_mults.len());
        let small_column_length = small_values.len().next_power_of_two();
        let small_flat: Vec<BaseField> = small_values
            .iter()
            .flat_map(|value| u128_to_4_limbs(*value))
            .map(BaseField::from_u32_unchecked)
            .collect();
        let small_values_dev = BaseFieldVec::from_vec(small_flat);
        let small_limbs = device_witness::limb_split_small(
            &small_values_dev,
            small_values.len(),
            small_column_length,
        );
        let mut small_mults_host: Vec<BaseField> = small_mults
            .iter()
            .flat_map(|packed| packed.to_array())
            .collect();
        small_mults_host.resize(small_column_length, BaseField::from_u32_unchecked(0));
        let small_mults_dev = BaseFieldVec::from_vec(small_mults_host);
        accumulate_counts(
            &mut rc_counts,
            &device_witness::rc99_count(&small_limbs, small_column_length, &lut, RC_TABLE_SIZE),
        );
        let small_trace = segment_trace(&small_limbs, &small_mults_dev, small_column_length);
        let small_log_size = small_column_length.ilog2();
        let small_segment = DeviceMemorySegment {
            limbs: small_limbs,
            mults: small_mults_dev,
            column_length: small_column_length,
        };

        // The differential, before any state is merged: host writers on the cloned
        // inputs, then byte-compare trace columns and the rc count delta.
        let verify_host = host_inputs.map(|(bv, bm, sv, sm)| {
            let host_big =
                memory_id_to_big::gen_big_memory_traces(bv, bm, log_max_big_size, opt_n_components);
            let host_small = memory_id_to_big::gen_small_memory_trace(sv, sm);
            verify_trace_columns(&big_segments, &small_segment, &host_big, &host_small);
            verify_rc_counts(&rc_counts, &lut, &host_big, &host_small);
            host_interaction_generator(&host_big, &host_small)
        });

        // Merge the device-computed rc_9_9 counts into the host atomics (adds
        // commute in u32 — byte-equal to the host's per-input feeding).
        range_check_9_9.add_count_tables(&rc_counts);

        (
            big_traces,
            small_trace,
            (
                BigClaim { big_log_sizes },
                SmallClaim {
                    log_size: small_log_size,
                },
            ),
            CudaMemoryInteractionGen::Device(Box::new(DeviceMemoryWitness {
                big_segments,
                small_segment,
                verify_host,
            })),
        )
    }

    fn write_interaction(
        gen: Self::InteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> MemoryInteractionResult<Self> {
        let witness = match gen {
            CudaMemoryInteractionGen::Host(host_gen) => {
                let (big_raws, small_raw, _build_big_claim, _build_small_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return MemoryInteractionResult {
                    big: big_raws
                        .into_iter()
                        .map(<CudaBackend as LogupFinalizeBackend>::finalize_raw_logup)
                        .collect(),
                    small: <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(small_raw),
                };
            }
            CudaMemoryInteractionGen::Device(witness) => witness,
        };
        let DeviceMemoryWitness {
            big_segments,
            small_segment,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();

        // Big segments: ids are `(offset + row) | LARGE_MEMORY_VALUE_ID_BASE`, the
        // offset advancing by each segment's PADDED length (the host advances by
        // `big_multiplicities.len() * N_LANES`, which is over the padded column).
        let mut offset = 0u32;
        let mut big = Vec::with_capacity(big_segments.len());
        for segment in &big_segments {
            big.push(segment_interaction(
                segment,
                offset,
                LARGE_MEMORY_VALUE_ID_BASE,
                alphas,
                z,
            ));
            offset += segment.column_length as u32;
        }
        // Small table: ids are plain row indices (offset 0, no tag).
        let small = segment_interaction(&small_segment, 0, 0, alphas, z);

        if let Some(host_gen) = verify_host {
            let (host_big_raws, host_small_raw, _build_big_claim, _build_small_claim) =
                host_gen.write_interaction_trace(common_lookup_elements);
            assert_eq!(host_big_raws.len(), big.len());
            let mut mismatches = 0usize;
            for (idx, (host_raw, (dev_trace, dev_sum))) in
                host_big_raws.into_iter().zip(big.iter()).enumerate()
            {
                let (host_trace, host_sum) = host_raw.finalize_on_simd();
                mismatches += compare_interaction(
                    &format!("memory_id_to_big big[{idx}]"),
                    dev_trace,
                    *dev_sum,
                    &host_trace,
                    host_sum,
                );
            }
            let (host_small_trace, host_small_sum) = host_small_raw.finalize_on_simd();
            mismatches += compare_interaction(
                "memory_id_to_small",
                &small.0,
                small.1,
                &host_small_trace,
                host_small_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: memory interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: memory_id_to_big interaction columns + sums OK");
        }

        MemoryInteractionResult { big, small }
    }
}

fn accumulate_counts(acc: &mut [u32], counts: &[u32]) {
    assert_eq!(acc.len(), counts.len());
    for (a, c) in acc.iter_mut().zip(counts) {
        *a += c;
    }
}

/// The committed base-trace columns of one segment: D2D clones of the limb columns
/// plus the multiplicity column (the originals stay live for the interaction write).
fn segment_trace(
    limbs: &[BaseFieldVec],
    mults: &BaseFieldVec,
    column_length: usize,
) -> MemoryEvals<CudaBackend> {
    let domain = CanonicCoset::new(column_length.ilog2()).circle_domain();
    limbs
        .iter()
        .map(|col| CircleEvaluation::new(domain, col.clone()))
        .chain([CircleEvaluation::new(domain, mults.clone())])
        .collect()
}

/// One segment's logup columns in the host writer's order: the pair-batched rc_9_9
/// columns (limb quads `4i..4i+3`, relation pair `i % 4`), then the final memory
/// column — chained and finalized on device.
fn segment_interaction(
    segment: &DeviceMemorySegment,
    id_offset: u32,
    id_tag: u32,
    alphas: &[SecureField],
    z: SecureField,
) -> (MemoryEvals<CudaBackend>, SecureField) {
    let n_limbs = segment.limbs.len();
    debug_assert!(n_limbs == FELT252_N_WORDS || n_limbs == N_M31_IN_SMALL_FELT252);
    let n_pair_columns = n_limbs / 4;
    let mut columns = Vec::with_capacity(n_pair_columns + 1);
    for i in 0..n_pair_columns {
        let (rel_id0, rel_id1) = PAIR_RELATION_IDS[i % 4];
        columns.push(device_witness::memory_rc_pair_logup(
            [
                &segment.limbs[4 * i],
                &segment.limbs[4 * i + 1],
                &segment.limbs[4 * i + 2],
                &segment.limbs[4 * i + 3],
            ],
            rel_id0.0,
            rel_id1.0,
            segment.column_length,
            alphas,
            z,
        ));
    }
    columns.push(device_witness::memory_logup_inputs(
        &segment.limbs,
        &segment.mults,
        MEMORY_ID_TO_BIG_RELATION_ID.0,
        id_offset,
        id_tag,
        segment.column_length,
        &alphas[..n_limbs + 2],
        z,
    ));
    device_witness::finalize_device_raw_logup(segment.column_length.ilog2(), columns)
}

/// Rebuilds the host `InteractionClaimGenerator` from host trace columns, exactly
/// like the host `write_trace` does (verify mode only).
fn host_interaction_generator(
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

fn base_column_values(column: &BaseColumn) -> Vec<BaseField> {
    column
        .data
        .iter()
        .flat_map(|packed| packed.to_array())
        .collect()
}

fn verify_trace_columns(
    big_segments: &[DeviceMemorySegment],
    small_segment: &DeviceMemorySegment,
    host_big: &[Vec<BaseColumn>],
    host_small: &[BaseColumn],
) {
    assert_eq!(
        big_segments.len(),
        host_big.len(),
        "STWO_CUDA_WITNESS_VERIFY: big segment count mismatch"
    );
    let mut mismatches = 0usize;
    let mut compare_segment = |label: &str, segment: &DeviceMemorySegment, host: &[BaseColumn]| {
        assert_eq!(segment.limbs.len() + 1, host.len());
        for (col_idx, (device_col, host_col)) in segment
            .limbs
            .iter()
            .chain([&segment.mults])
            .zip(host)
            .enumerate()
        {
            let device_values = device_col.to_vec();
            let host_values = base_column_values(host_col);
            if device_values != host_values {
                let first_diff = device_values
                    .iter()
                    .zip(&host_values)
                    .position(|(d, h)| d != h);
                eprintln!(
                    "STWO_CUDA_WITNESS_VERIFY: {label} col {col_idx} MISMATCH \
                     (first diff at row {first_diff:?})"
                );
                mismatches += 1;
            }
        }
    };
    for (idx, (segment, host)) in big_segments.iter().zip(host_big).enumerate() {
        compare_segment(&format!("memory_id_to_big big[{idx}]"), segment, host);
    }
    compare_segment("memory_id_to_small", small_segment, host_small);
    assert_eq!(
        mismatches, 0,
        "STWO_CUDA_WITNESS_VERIFY: memory trace differential failed"
    );
    eprintln!("STWO_CUDA_WITNESS_VERIFY: memory_id_to_big trace columns OK");
}

/// Recomputes the rc_9_9 count delta from the HOST trace columns (pairs `(2j, 2j+1)`
/// into relation `j % 8`, padding rows included — the host feeding loop) and
/// compares against the device-accumulated count tables.
fn verify_rc_counts(
    device_counts: &[u32],
    lut: &[u32],
    host_big: &[Vec<BaseColumn>],
    host_small: &[BaseColumn],
) {
    let mut expected = vec![0u32; 8 * RC_TABLE_SIZE];
    let mut feed = |limb_columns: &[BaseColumn]| {
        for (pair_idx, pair) in limb_columns.chunks(2).enumerate() {
            let col0 = base_column_values(&pair[0]);
            let col1 = base_column_values(&pair[1]);
            for (v0, v1) in col0.iter().zip(&col1) {
                let row = lut[((v0.0 << 9) | v1.0) as usize] as usize;
                expected[(pair_idx % 8) * RC_TABLE_SIZE + row] += 1;
            }
        }
    };
    for trace in host_big {
        feed(&trace[..FELT252_N_WORDS]);
    }
    feed(&host_small[..N_M31_IN_SMALL_FELT252]);

    let mut mismatches = 0usize;
    for relation in 0..8 {
        let device = &device_counts[relation * RC_TABLE_SIZE..(relation + 1) * RC_TABLE_SIZE];
        let host = &expected[relation * RC_TABLE_SIZE..(relation + 1) * RC_TABLE_SIZE];
        if device != host {
            let first_diff = device.iter().zip(host).position(|(d, h)| d != h);
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY: rc_9_9 count table {relation} MISMATCH \
                 (first diff at row {first_diff:?})"
            );
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "STWO_CUDA_WITNESS_VERIFY: rc_9_9 count differential failed"
    );
    eprintln!("STWO_CUDA_WITNESS_VERIFY: rc_9_9 count tables OK");
}

fn compare_interaction(
    label: &str,
    device_trace: &MemoryEvals<CudaBackend>,
    device_sum: SecureField,
    host_trace: &MemoryEvals<SimdBackend>,
    host_sum: SecureField,
) -> usize {
    let mut mismatches = 0usize;
    if device_trace.len() != host_trace.len() {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY: {label} interaction column count mismatch: \
             device {} vs host {}",
            device_trace.len(),
            host_trace.len()
        );
        return 1;
    }
    for (col_idx, (device_col, host_col)) in device_trace.iter().zip(host_trace).enumerate() {
        let device_values = device_col.values.to_vec();
        let host_values = host_col.values.to_cpu();
        if device_values != host_values {
            let first_diff = device_values
                .iter()
                .zip(&host_values)
                .position(|(d, h)| d != h);
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY: {label} interaction col {col_idx} MISMATCH \
                 (first diff at row {first_diff:?})"
            );
            mismatches += 1;
        }
    }
    if device_sum != host_sum {
        eprintln!(
            "STWO_CUDA_WITNESS_VERIFY: {label} claimed sum MISMATCH: \
             device {device_sum:?} vs host {host_sum:?}"
        );
        mismatches += 1;
    }
    mismatches
}
