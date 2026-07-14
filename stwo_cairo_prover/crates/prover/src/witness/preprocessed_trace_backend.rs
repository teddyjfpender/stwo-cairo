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
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_backend_cuda::{BaseFieldVec, CudaBackend, CudaRuntimeError};
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
pub struct CudaPreprocessedColumnStreamer;

fn checked_cuda(operation: &'static str, code: i32) -> Result<(), CudaRuntimeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaRuntimeError::Cuda { operation, code })
    }
}

fn generate_family<T, U>(
    count: usize,
    mut allocate: impl FnMut() -> Result<T, CudaRuntimeError>,
    launch: impl FnOnce(&[T]) -> Result<(), CudaRuntimeError>,
    select: impl FnOnce(Vec<T>) -> U,
    mut fence: impl FnMut() -> Result<(), CudaRuntimeError>,
) -> Result<U, CudaRuntimeError> {
    let mut outputs = Vec::with_capacity(count);
    for _ in 0..count {
        match allocate() {
            Ok(output) => outputs.push(output),
            Err(error) => {
                drop(outputs);
                // Preserve the allocation error even if cleanup fencing also
                // fails: it is the first failed operation and therefore the
                // actionable primary diagnosis.
                let _ = fence();
                return Err(error);
            }
        }
    }
    if let Err(error) = launch(&outputs) {
        drop(outputs);
        // As above, rollback is mandatory but cannot replace the primary
        // launch status when the CUDA context reports multiple failures.
        let _ = fence();
        return Err(error);
    }

    // Selection drops every unselected family output, enqueueing its legacy-
    // stream free after the generation kernel. Fence only after that rollback
    // so success cannot leave hidden detached allocations in flight.
    let selected = select(outputs);
    if let Err(error) = fence() {
        drop(selected);
        let _ = fence();
        return Err(error);
    }
    Ok(selected)
}

fn select_family_output<T>(outputs: Vec<T>, index: usize) -> Option<T> {
    outputs
        .into_iter()
        .enumerate()
        .find_map(|(candidate, output)| (candidate == index).then_some(output))
}

impl CudaPreprocessedColumnStreamer {
    /// Fixed resident source lane. Legacy whole-trace generation keeps its env
    /// switch above; immutable resident backends never consult it after sealing.
    pub fn gpu_preferred() -> Self {
        Self
    }

    /// Maximum detached device storage created while generating this column.
    /// Multi-output range/XOR kernels allocate their complete family together;
    /// every other path owns only the selected column.
    pub fn detached_staging_bytes(column: &dyn PreProcessedColumn) -> Option<usize> {
        let id = column.id().id;
        let words = if let Some(log) = id.strip_prefix("seq_").and_then(|s| s.parse::<u32>().ok()) {
            1u32.checked_shl(log)? as usize
        } else if let Some((bits, _)) = parse_range_check_id(&id) {
            let total_bits = bits
                .iter()
                .try_fold(0u32, |sum, bits| sum.checked_add(*bits))?;
            (1u32.checked_shl(total_bits)? as usize).checked_mul(bits.len())?
        } else if let Some((n_bits, _)) = parse_bitwise_xor_id(&id) {
            (1u32.checked_shl(n_bits.checked_mul(2)?)? as usize).checked_mul(3)?
        } else {
            1usize.checked_shl(column.log_size())?
        };
        words.checked_mul(core::mem::size_of::<u32>())
    }

    pub fn generate(
        &mut self,
        column: &dyn PreProcessedColumn,
    ) -> Result<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>, CudaRuntimeError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(CudaRuntimeError::Unavailable);
        }

        let id = column.id().id;
        let log_size = column.log_size();
        let domain = CanonicCoset::new(log_size).circle_domain();
        if let Some(log) = id.strip_prefix("seq_").and_then(|s| s.parse::<u32>().ok()) {
            let words = 1u32
                .checked_shl(log)
                .ok_or(CudaRuntimeError::SizeOverflow)? as usize;
            let values = generate_family(
                1,
                || Self::allocate(words),
                |outputs| {
                    let code = unsafe {
                        stwo_backend_cuda_kernels::raw::stwo_preprocessed_gen_seq_checked(
                            outputs[0].device_ptr.cast_mut(),
                            log,
                        )
                    };
                    checked_cuda("preprocessed_gen_seq", code)
                },
                |mut outputs| outputs.pop().expect("one allocated sequence output"),
                Self::sync,
            )?;
            return Ok(CircleEvaluation::new(domain, values));
        }

        if let Some((bits, idx)) = parse_range_check_id(&id) {
            let total_bits = bits
                .iter()
                .try_fold(0u32, |sum, bits| sum.checked_add(*bits))
                .ok_or(CudaRuntimeError::SizeOverflow)?;
            let words = 1u32
                .checked_shl(total_bits)
                .ok_or(CudaRuntimeError::SizeOverflow)? as usize;
            let count = u32::try_from(bits.len()).map_err(|_| CudaRuntimeError::SizeOverflow)?;
            let values = generate_family(
                bits.len(),
                || Self::allocate(words),
                |outputs| {
                    let pointers = outputs
                        .iter()
                        .map(|output| output.device_ptr.cast_mut())
                        .collect::<Vec<_>>();
                    let code = unsafe {
                        stwo_backend_cuda_kernels::raw::stwo_preprocessed_gen_range_checked(
                            pointers.as_ptr(),
                            count,
                            bits.as_ptr(),
                            count,
                        )
                    };
                    checked_cuda("preprocessed_gen_range", code)
                },
                |outputs| select_family_output(outputs, idx),
                Self::sync,
            )?;
            if let Some(values) = values {
                return Ok(CircleEvaluation::new(domain, values));
            }
        }

        if let Some((n_bits, idx)) = parse_bitwise_xor_id(&id) {
            let doubled_bits = n_bits
                .checked_mul(2)
                .ok_or(CudaRuntimeError::SizeOverflow)?;
            let words = 1u32
                .checked_shl(doubled_bits)
                .ok_or(CudaRuntimeError::SizeOverflow)? as usize;
            let values = generate_family(
                3,
                || Self::allocate(words),
                |outputs| {
                    let pointers = outputs
                        .iter()
                        .map(|output| output.device_ptr.cast_mut())
                        .collect::<Vec<_>>();
                    let code = unsafe {
                        stwo_backend_cuda_kernels::raw::stwo_preprocessed_gen_xor_checked(
                            pointers.as_ptr(),
                            n_bits,
                        )
                    };
                    checked_cuda("preprocessed_gen_xor", code)
                },
                |outputs| select_family_output(outputs, idx),
                Self::sync,
            )?;
            if let Some(values) = values {
                return Ok(CircleEvaluation::new(domain, values));
            }
        }

        let host_evaluation = column.gen_column_simd();
        let host_values = host_evaluation.values.to_cpu();
        let words = host_values.len();
        let values = generate_family(
            1,
            || Self::allocate(words),
            |outputs| {
                let code = unsafe {
                    stwo_backend_cuda_kernels::raw::stwo_preprocessed_copy_h2d_checked(
                        host_values.as_ptr().cast(),
                        outputs[0].device_ptr.cast_mut(),
                        words,
                    )
                };
                checked_cuda("preprocessed_copy_h2d", code)
            },
            |mut outputs| outputs.pop().expect("one allocated fallback output"),
            Self::sync,
        )?;
        Ok(CircleEvaluation::new(host_evaluation.domain, values))
    }

    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        Self::sync()
    }

    fn allocate(words: usize) -> Result<BaseFieldVec, CudaRuntimeError> {
        words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(CudaRuntimeError::SizeOverflow)?;
        let mut raw = core::ptr::null_mut();
        let code = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_preprocessed_alloc_u32_checked(words, &mut raw)
        };
        checked_cuda("preprocessed_alloc_u32", code)?;
        let pointer = core::ptr::NonNull::new(raw).ok_or(CudaRuntimeError::NullPointer {
            operation: "preprocessed_alloc_u32",
        })?;
        Ok(BaseFieldVec::new(pointer.as_ptr().cast_const(), words))
    }

    fn sync() -> Result<(), CudaRuntimeError> {
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return Err(CudaRuntimeError::Unavailable);
        }
        let code =
            unsafe { stwo_backend_cuda_kernels::raw::stwo_preprocessed_stream_sync_checked() };
        checked_cuda("preprocessed_stream_sync", code)
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
    // An id with an out-of-family index is not a GPU-family match. Route it to
    // the canonical per-column fallback instead of indexing a device family.
    (idx < bits.len()).then_some((bits, idx))
}

/// `bitwise_xor_{n}_column_{idx}`
fn parse_bitwise_xor_id(id: &str) -> Option<(u32, usize)> {
    let stripped = id.strip_prefix("bitwise_xor_")?;
    let mut parts = stripped.split("_column_");
    let n_bits = parts.next()?.parse::<u32>().ok()?;
    let idx = parts.next()?.parse::<usize>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    // See `parse_range_check_id`: malformed indices are canonical-fallback
    // inputs, never unchecked vector indices.
    (idx < 3).then_some((n_bits, idx))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

    use super::*;

    #[derive(Debug)]
    struct DropProbe(Rc<Cell<usize>>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn injected(operation: &'static str, code: i32) -> CudaRuntimeError {
        CudaRuntimeError::Cuda { operation, code }
    }

    #[test]
    fn family_allocation_failure_drops_prefix_and_preserves_primary_error() {
        let drops = Rc::new(Cell::new(0));
        let attempts = Cell::new(0);
        let fences = Cell::new(0);
        let primary = injected("injected_allocate", 2);
        let result: Result<DropProbe, _> = generate_family(
            3,
            || {
                let attempt = attempts.get();
                attempts.set(attempt + 1);
                if attempt == 2 {
                    Err(primary.clone())
                } else {
                    Ok(DropProbe(Rc::clone(&drops)))
                }
            },
            |_| unreachable!("launch must not run after allocation failure"),
            |mut outputs| outputs.pop().expect("test allocation prefix is non-empty"),
            || {
                fences.set(fences.get() + 1);
                Err(injected("injected_cleanup_fence", 700))
            },
        );
        assert_eq!(result.unwrap_err(), primary);
        assert_eq!(attempts.get(), 3);
        assert_eq!(drops.get(), 2);
        assert_eq!(fences.get(), 1);
    }

    #[test]
    fn family_launch_failure_drops_every_output_before_cleanup_fence() {
        let drops = Rc::new(Cell::new(0));
        let fences = Cell::new(0);
        let primary = injected("injected_launch", 719);
        let result: Result<DropProbe, _> = generate_family(
            3,
            || Ok(DropProbe(Rc::clone(&drops))),
            |_| Err(primary.clone()),
            |_| unreachable!("selection must not run after launch failure"),
            || {
                assert_eq!(drops.get(), 3);
                fences.set(fences.get() + 1);
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err(), primary);
        assert_eq!(drops.get(), 3);
        assert_eq!(fences.get(), 1);
    }

    #[test]
    fn family_fence_failure_drops_selection_then_refences_cleanup() {
        let drops = Rc::new(Cell::new(0));
        let fences = Cell::new(0);
        let primary = injected("injected_fence", 700);
        let result = generate_family(
            3,
            || Ok(DropProbe(Rc::clone(&drops))),
            |_| Ok(()),
            |outputs| select_family_output(outputs, 1).expect("valid test selection"),
            || {
                let call = fences.get();
                fences.set(call + 1);
                assert_eq!(drops.get(), if call == 0 { 2 } else { 3 });
                if call == 0 {
                    Err(primary.clone())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result.unwrap_err(), primary);
        assert_eq!(drops.get(), 3);
        assert_eq!(fences.get(), 2);
    }

    #[test]
    fn family_success_drops_unselected_outputs_before_fence() {
        let drops = Rc::new(Cell::new(0));
        let fences = Cell::new(0);
        let selected = generate_family(
            3,
            || Ok(DropProbe(Rc::clone(&drops))),
            |_| Ok(()),
            |outputs| select_family_output(outputs, 1).expect("valid test selection"),
            || {
                assert_eq!(drops.get(), 2);
                fences.set(fences.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(drops.get(), 2);
        assert_eq!(fences.get(), 1);
        drop(selected);
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn family_missing_selection_drops_all_outputs_without_panicking() {
        let drops = Rc::new(Cell::new(0));
        let selected = generate_family(
            3,
            || Ok(DropProbe(Rc::clone(&drops))),
            |_| Ok(()),
            |outputs| select_family_output(outputs, 3),
            || {
                assert_eq!(drops.get(), 3);
                Ok(())
            },
        )
        .unwrap();
        assert!(selected.is_none());
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn malformed_family_indices_route_to_canonical_fallback() {
        assert!(parse_range_check_id("range_check_4_3_column_2").is_none());
        assert!(parse_bitwise_xor_id("bitwise_xor_4_column_3").is_none());
        assert!(parse_bitwise_xor_id("bitwise_xor_4_column_1_column_0").is_none());
    }

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
