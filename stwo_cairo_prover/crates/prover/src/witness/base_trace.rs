//! Base-trace hand-off from `write_trace` to the commitment path — the seam for
//! Stage A″ (pipelined commit under witness, `STWO_CUDA_PIPELINED_COMMIT`).
//!
//! By default `write_trace` returns [`BaseTrace::Evals`]: the raw committed columns
//! in canonical order, interpolated by `TreeBuilder::extend_evals` at commit time
//! exactly as before — byte-identical to the pre-A″ flow.
//!
//! When pipelined commit is enabled, the opcode-prefix columns (produced by the
//! parallel witness scope, already device-resident) are interpolated on a committer
//! thread WHILE the serial host-heavy components (blake_round / partial_ec_mul /
//! pedersen_aggregator — the ~2.8s host block) generate. `write_trace` then returns
//! [`BaseTrace::Polys`]: the whole base trace already interpolated to
//! `CircleCoefficients`, in the SAME canonical column order `extend_evals` would have
//! produced.
//!
//! ## Byte-identity
//!
//! `PolyOps::interpolate_columns` is per-column independent and order-preserving, so
//! for a prefix/suffix partition of the columns
//! `interpolate(prefix) ++ interpolate(suffix) == interpolate(prefix ++ suffix)`
//! provided the SAME twiddle tree is used for both. `domain_line_twiddles_from_tree`
//! slices the tree relative to its own length, so a differently sized tree yields
//! different coefficients — hence the committer's tree identity is carried in
//! [`BaseTrace::Polys::tree_ptr`] and verified against the commitment tree at the
//! call site (fail-closed: a mismatch means the trace size changed mid-process and
//! the pipelined interpolation would be byte-wrong, so the prover panics rather than
//! emit a corrupt proof). The property is unit-tested on `SimdBackend` in
//! `tests::prefix_split_interpolation_is_byte_identical`.

use stwo::core::fields::m31::BaseField;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;

/// The base trace as returned by `write_trace`, in canonical committed-column order.
pub enum BaseTrace<B: PolyOps> {
    /// Raw evaluations; the caller interpolates at commit time (default path,
    /// byte-identical to the pre-A″ flow).
    Evals(Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>),
    /// Already-interpolated coefficients (pipelined commit). `tree_ptr` is the
    /// identity (`*const TwiddleTree<B> as usize`) of the twiddle tree used; the
    /// caller MUST verify it equals the commitment tree before committing.
    Polys {
        polys: Vec<CircleCoefficients<B>>,
        tree_ptr: usize,
    },
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;

    /// The load-bearing A″ invariant: interpolating a prefix and a suffix separately
    /// (as the committer thread + the main thread do) and concatenating the resulting
    /// coefficient columns is byte-for-byte identical to interpolating the whole set
    /// at once (what `extend_evals` does today) — provided the same twiddle tree is
    /// used. If this ever fails, pipelined commit would produce a different proof.
    #[test]
    fn prefix_split_interpolation_is_byte_identical() {
        // Columns of several sizes, mirroring the base trace's mixed log sizes.
        let log_sizes = [8u32, 5, 8, 6, 7, 5, 8];
        let max_log = *log_sizes.iter().max().unwrap();
        let twiddles =
            SimdBackend::precompute_twiddles(CanonicCoset::new(max_log).circle_domain().half_coset);

        let make_cols = || -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
            log_sizes
                .iter()
                .enumerate()
                .map(|(col, &log_size)| {
                    let domain: CircleDomain = CanonicCoset::new(log_size).circle_domain();
                    let values = (0..(1u32 << log_size))
                        .map(|row| {
                            BaseField::from(
                                (row.wrapping_mul(2_654_435_761)
                                    .wrapping_add(col as u32 * 97))
                                    % 1000,
                            )
                        })
                        .collect::<Vec<_>>();
                    CircleEvaluation::new(domain, values.into_iter().collect())
                })
                .collect()
        };

        // Reference: interpolate all columns at once (the extend_evals path).
        let whole = SimdBackend::interpolate_columns(make_cols(), &twiddles);

        // Split at an arbitrary prefix boundary and interpolate the two halves
        // independently, then concatenate — the pipelined-commit path.
        for split in [0usize, 1, 3, log_sizes.len()] {
            let mut cols = make_cols();
            let suffix = cols.split_off(split);
            let mut spliced = SimdBackend::interpolate_columns(cols, &twiddles);
            spliced.extend(SimdBackend::interpolate_columns(suffix, &twiddles));

            assert_eq!(whole.len(), spliced.len(), "column count (split={split})");
            for (i, (a, b)) in whole.iter().zip(&spliced).enumerate() {
                assert_eq!(
                    a.coeffs.to_cpu(),
                    b.coeffs.to_cpu(),
                    "coefficients differ at column {i} (split={split})"
                );
            }
        }
    }
}
