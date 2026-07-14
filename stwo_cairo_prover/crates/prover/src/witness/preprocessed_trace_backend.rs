//! Backend-specific preprocessed-trace generation.
//!
//! The default for every backend generates columns on the SIMD (witness) backend and
//! transfers them via `FromSimdColumns`. `CudaBackend` overrides the column families
//! with GPU generators (Seq, RangeCheck, BitwiseXor — values identical to the CPU
//! constructors, so commitment roots stay byte-equal; the Cairo e2e gate enforces
//! this). Columns whose id doesn't parse fall back to the SIMD path per column, so an
//! id-format drift degrades performance, never correctness.
//!
//! Family caches are keyed by the family PARAMETERS (content), never by pointers.

use std::collections::HashMap;
use std::sync::Arc;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::FromSimdColumns;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{BaseFieldVec, CudaBackend};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedColumn, PreProcessedTrace,
};

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

/// Backend hook for preprocessed-trace generation.
pub trait GenPreprocessedTrace: FromSimdColumns {
    fn gen_preprocessed_trace(trace: Arc<PreProcessedTrace>) -> Evals<Self> {
        Self::from_simd_evals(super::preprocessed_trace::gen_trace(trace))
    }
}

impl GenPreprocessedTrace for SimdBackend {}

impl GenPreprocessedTrace for CudaBackend {
    fn gen_preprocessed_trace(trace: Arc<PreProcessedTrace>) -> Evals<Self> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
            || std::env::var("PREPROCESSED_TRACE_GPU_GENERATE").as_deref() == Ok("0")
        {
            return Self::from_simd_evals(super::preprocessed_trace::gen_trace(trace));
        }

        let mut range_check_cache: HashMap<Vec<u32>, Vec<BaseFieldVec>> = HashMap::new();
        let mut bitwise_xor_cache: HashMap<u32, Vec<BaseFieldVec>> = HashMap::new();

        trace
            .columns
            .iter()
            .map(|column| {
                let id = column.id().id;
                let log_size = column.log_size();
                let domain = CanonicCoset::new(log_size).circle_domain();

                if let Some(log) = id.strip_prefix("seq_").and_then(|s| s.parse::<u32>().ok()) {
                    let values = BaseFieldVec::new_uninitialized(1 << log);
                    unsafe {
                        stwo_backend_cuda_kernels::raw::gen_seq_column_on_gpu(
                            values.device_ptr.cast_mut(),
                            log,
                        );
                    }
                    return CircleEvaluation::new(domain, values);
                }

                if let Some((bits, idx)) = parse_range_check_id(&id) {
                    let columns = range_check_cache.entry(bits.clone()).or_insert_with(|| {
                        let total_bits: u32 = bits.iter().sum();
                        let outputs: Vec<BaseFieldVec> = (0..bits.len())
                            .map(|_| BaseFieldVec::new_uninitialized(1 << total_bits))
                            .collect();
                        let ptrs: Vec<*mut u32> =
                            outputs.iter().map(|o| o.device_ptr.cast_mut()).collect();
                        unsafe {
                            stwo_backend_cuda_kernels::raw::gen_range_check_columns_on_gpu(
                                ptrs.as_ptr(),
                                bits.len() as u32,
                                bits.as_ptr(),
                                bits.len() as u32,
                            );
                        }
                        outputs
                    });
                    if let Some(values) = columns.get(idx) {
                        return CircleEvaluation::new(domain, values.clone());
                    }
                }

                if let Some((n_bits, idx)) = parse_bitwise_xor_id(&id) {
                    let columns = bitwise_xor_cache.entry(n_bits).or_insert_with(|| {
                        let outputs: Vec<BaseFieldVec> = (0..3)
                            .map(|_| BaseFieldVec::new_uninitialized(1 << (2 * n_bits)))
                            .collect();
                        let ptrs: Vec<*mut u32> =
                            outputs.iter().map(|o| o.device_ptr.cast_mut()).collect();
                        unsafe {
                            stwo_backend_cuda_kernels::raw::gen_bitwise_xor_columns_on_gpu(
                                ptrs.as_ptr(),
                                n_bits,
                            );
                        }
                        outputs
                    });
                    if let Some(values) = columns.get(idx) {
                        return CircleEvaluation::new(domain, values.clone());
                    }
                }

                // Per-column SIMD fallback (tiny columns, unknown families).
                Self::from_simd_evals(vec![column.gen_column_simd()])
                    .pop()
                    .expect("single column conversion")
            })
            .collect()
    }
}

/// Bounded cold-source generator for the resident prover.
///
/// The legacy hook above returns every device evaluation at once. A resident
/// arena is already allocated when fixed columns are staged, so retaining that
/// vector creates a second multi-gigabyte live set. This generator instead
/// returns exactly one column (or one small multi-output family) at a time. The
/// caller must consume and drop the result before requesting the next column.
pub struct CudaPreprocessedColumnStreamer {
    gpu_generate: bool,
}

impl CudaPreprocessedColumnStreamer {
    /// Fixed resident source lane. Legacy whole-trace generation keeps its env
    /// switch above; immutable resident backends never consult it after sealing.
    pub fn gpu_preferred() -> Self {
        Self {
            gpu_generate: stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT,
        }
    }

    /// Maximum detached device storage created while generating this column.
    /// Multi-output range/XOR kernels allocate their complete family together;
    /// every other path owns only the selected column.
    pub fn detached_staging_bytes(column: &dyn PreProcessedColumn) -> Option<usize> {
        let id = column.id().id;
        let words = if let Some((bits, _)) = parse_range_check_id(&id) {
            1usize
                .checked_shl(bits.iter().copied().sum::<u32>())?
                .checked_mul(bits.len())?
        } else if let Some((n_bits, _)) = parse_bitwise_xor_id(&id) {
            1usize.checked_shl(n_bits.checked_mul(2)?)?.checked_mul(3)?
        } else {
            1usize.checked_shl(column.log_size())?
        };
        words.checked_mul(core::mem::size_of::<u32>())
    }

    pub fn generate(
        &mut self,
        column: &dyn PreProcessedColumn,
    ) -> CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> {
        if !self.gpu_generate {
            return CudaBackend::from_simd_evals(vec![column.gen_column_simd()])
                .pop()
                .expect("single preprocessed column conversion");
        }

        let id = column.id().id;
        let log_size = column.log_size();
        let domain = CanonicCoset::new(log_size).circle_domain();
        if let Some(log) = id.strip_prefix("seq_").and_then(|s| s.parse::<u32>().ok()) {
            let values = BaseFieldVec::new_uninitialized(1 << log);
            unsafe {
                stwo_backend_cuda_kernels::raw::gen_seq_column_on_gpu(
                    values.device_ptr.cast_mut(),
                    log,
                );
            }
            return CircleEvaluation::new(domain, values);
        }

        if let Some((bits, idx)) = parse_range_check_id(&id) {
            let total_bits: u32 = bits.iter().sum();
            let outputs: Vec<BaseFieldVec> = (0..bits.len())
                .map(|_| BaseFieldVec::new_uninitialized(1 << total_bits))
                .collect();
            let ptrs: Vec<*mut u32> = outputs
                .iter()
                .map(|output| output.device_ptr.cast_mut())
                .collect();
            unsafe {
                stwo_backend_cuda_kernels::raw::gen_range_check_columns_on_gpu(
                    ptrs.as_ptr(),
                    bits.len() as u32,
                    bits.as_ptr(),
                    bits.len() as u32,
                );
            }
            if let Some(values) = outputs.into_iter().nth(idx) {
                return CircleEvaluation::new(domain, values);
            }
        }

        if let Some((n_bits, idx)) = parse_bitwise_xor_id(&id) {
            let outputs: Vec<BaseFieldVec> = (0..3)
                .map(|_| BaseFieldVec::new_uninitialized(1 << (2 * n_bits)))
                .collect();
            let ptrs: Vec<*mut u32> = outputs
                .iter()
                .map(|output| output.device_ptr.cast_mut())
                .collect();
            unsafe {
                stwo_backend_cuda_kernels::raw::gen_bitwise_xor_columns_on_gpu(
                    ptrs.as_ptr(),
                    n_bits,
                );
            }
            if let Some(values) = outputs.into_iter().nth(idx) {
                return CircleEvaluation::new(domain, values);
            }
        }

        CudaBackend::from_simd_evals(vec![column.gen_column_simd()])
            .pop()
            .expect("single preprocessed column conversion")
    }
}

/// `range_check_{b1}_{b2}_..._{bN}_column_{idx}`
fn parse_range_check_id(id: &str) -> Option<(Vec<u32>, usize)> {
    let stripped = id.strip_prefix("range_check_")?;
    let column_pos = stripped.find("_column_")?;
    let bits: Vec<u32> = stripped[..column_pos]
        .split('_')
        .map(|part| part.parse::<u32>())
        .collect::<Result<_, _>>()
        .ok()?;
    let idx = stripped[column_pos + "_column_".len()..]
        .parse::<usize>()
        .ok()?;
    (idx < bits.len()).then_some((bits, idx))
}

/// `bitwise_xor_{n}_column_{idx}`
fn parse_bitwise_xor_id(id: &str) -> Option<(u32, usize)> {
    let stripped = id.strip_prefix("bitwise_xor_")?;
    let mut parts = stripped.split("_column_");
    let n_bits = parts.next()?.parse::<u32>().ok()?;
    let idx = parts.next()?.parse::<usize>().ok()?;
    (idx < 3).then_some((n_bits, idx))
}

#[cfg(test)]
mod tests {
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

    use super::*;

    #[test]
    fn canonical_streaming_source_peak_is_bounded_to_128_mib() {
        let maxima = PreProcessedTraceVariant::ALL_VARIANTS.map(|variant| {
            variant
                .to_preprocessed_trace()
                .columns
                .iter()
                .map(|column| {
                    CudaPreprocessedColumnStreamer::detached_staging_bytes(column.as_ref()).unwrap()
                })
                .max()
                .unwrap()
        });
        assert_eq!(maxima[0], 128 * 1024 * 1024);
        assert_eq!(maxima[1], 128 * 1024 * 1024);
        assert!(maxima[2] < 128 * 1024 * 1024);
    }
}
