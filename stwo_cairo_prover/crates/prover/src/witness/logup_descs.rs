//! §6a descriptor resolution + host mirror (GPU_RESIDENT_PROVER_DESIGN.md §5.3).
//!
//! The transformer emits per-component `JIT_LOGUP_DESCS` FACTS — one entry per
//! logup column, parsed from the generated `write_interaction_trace` (pairing
//! order is arbitrary and sign/mult patterns vary per component, so these are
//! never derived by rule). This module resolves the facts against the
//! component's `JIT_LOOKUP_FIELDS` into the CUDA backend's [`LogupColDesc`]
//! (word offsets, mult sources, explicit signs), and provides a pure-host
//! MIRROR that builds a `RawLogupTrace` from the word-major lookup flats using
//! the same per-column math the device kernel runs.
//!
//! The mirror is the LOCAL gate's engine: `finalize_raw_logup(mirror(flats))`
//! must byte-match `finalize_raw_logup(writer(flats))` per component — validated
//! on fixtures without any CUDA. The device kernel then only re-implements the
//! (gate-proven) descriptor semantics; the pod differential covers the kernel
//! itself.

use cairo_air::relations::CommonLookupElements;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::simd::m31::{PackedM31, N_LANES};
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_backend_cuda::logup_pairs::{LogupColDesc, MultSrc};
use stwo_constraint_framework::{RawLogupTrace, RawLogupTraceGenerator};

use crate::witness::utils::Enabler;

/// One emitted fact: (a_field, a_mult, a_neg, b_field, b_mult, b_neg);
/// `b_field == ""` for a trailing solo column; mult encoding `"1"`, `"enabler"`,
/// or a scalar lookup-data field name.
pub type LogupDescFact = (
    &'static str,
    &'static str,
    bool,
    &'static str,
    &'static str,
    bool,
);

/// Resolve emitted facts against the component's field list into device
/// descriptors. Panics on any inconsistency (unknown field, non-scalar mult) —
/// the facts and fields are emitted together; a mismatch is codegen drift.
pub fn resolve_logup_descs(fields: &[(&str, usize)], facts: &[LogupDescFact]) -> Vec<LogupColDesc> {
    let mut offsets = std::collections::HashMap::new();
    let mut off = 0u32;
    for (name, w) in fields {
        offsets.insert(*name, (off, *w as u32));
        off += *w as u32;
    }
    let field = |name: &str| -> (u32, u32) {
        *offsets
            .get(name)
            .unwrap_or_else(|| panic!("JIT_LOGUP_DESCS references unknown field {name}"))
    };
    let mult = |m: &str| -> MultSrc {
        match m {
            "1" => MultSrc::One,
            "enabler" => MultSrc::Enabler,
            name => {
                let (o, w) = field(name);
                assert_eq!(w, 1, "mult field {name} must be scalar");
                MultSrc::Flat(o)
            }
        }
    };
    facts
        .iter()
        .map(|(af, am, an, bf, bm, bn)| {
            let (off_a, width_a) = field(af);
            if bf.is_empty() {
                LogupColDesc {
                    kind: 1,
                    off_a,
                    width_a,
                    mult_a: mult(am),
                    neg_a: *an,
                    off_b: 0,
                    width_b: 0,
                    mult_b: MultSrc::One,
                    neg_b: false,
                }
            } else {
                let (off_b, width_b) = field(bf);
                LogupColDesc {
                    kind: 0,
                    off_a,
                    width_a,
                    mult_a: mult(am),
                    neg_a: *an,
                    off_b,
                    width_b,
                    mult_b: mult(bm),
                    neg_b: *bn,
                }
            }
        })
        .collect()
}

/// The maximal tuple width across descriptors — the alpha-powers count the
/// combine needs (`alpha^0..alpha^(max_width-1)`).
pub fn max_tuple_width(descs: &[LogupColDesc]) -> usize {
    descs
        .iter()
        .map(|d| d.width_a.max(d.width_b) as usize)
        .max()
        .unwrap_or(0)
}

/// Pure-host mirror of `logup_pairs.cu` over word-major flats
/// (`words[w * n_rows + r]`): per column, per packed row —
/// combine both tuples (`Σ αⁱ·vᵢ − z`), fetch the mults per source with signs
/// folded in, `num = m_b′·d_a + m_a′·d_b`, `den = d_a·d_b` (solo: `num = m_a′`,
/// `den = d_a`) — written through the writers' own `RawLogupTraceGenerator`.
pub fn host_mirror_raw_logup(
    words: &[u32],
    n_rows: usize,
    n_real: usize,
    descs: &[LogupColDesc],
    elements: &CommonLookupElements,
) -> RawLogupTrace {
    assert!(n_rows.is_power_of_two() && n_rows >= N_LANES);
    let log_size = n_rows.ilog2();
    let n_vec = n_rows / N_LANES;
    let enabler = Enabler::new(n_real);
    let alphas = elements.alpha_powers();
    let z = elements.z();

    let packed_word = |w: usize, vi: usize| -> PackedM31 {
        PackedM31::from_array(std::array::from_fn(|l| {
            M31::from_u32_unchecked(words[w * n_rows + vi * N_LANES + l])
        }))
    };
    let combine = |off: u32, width: u32, vi: usize| -> PackedQM31 {
        let mut acc = -PackedQM31::broadcast(z);
        for i in 0..width as usize {
            acc += PackedQM31::broadcast(alphas[i]) * packed_word(off as usize + i, vi);
        }
        acc
    };
    let fetch_mult = |src: MultSrc, neg: bool, vi: usize| -> PackedM31 {
        let m = match src {
            MultSrc::One => PackedM31::broadcast(M31::from(1)),
            MultSrc::Enabler => enabler.packed_at(vi),
            MultSrc::Flat(off) => packed_word(off as usize, vi),
        };
        if neg {
            -m
        } else {
            m
        }
    };

    let mut logup_gen = unsafe { RawLogupTraceGenerator::uninitialized(log_size) };
    for d in descs {
        let mut col_gen = logup_gen.new_col();
        {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            col_gen.par_iter_mut().enumerate().for_each(|(vi, writer)| {
                if d.kind == 0 {
                    let da = combine(d.off_a, d.width_a, vi);
                    let db = combine(d.off_b, d.width_b, vi);
                    let ma = fetch_mult(d.mult_a, d.neg_a, vi);
                    let mb = fetch_mult(d.mult_b, d.neg_b, vi);
                    writer.write_frac(da * mb + db * ma, da * db);
                } else {
                    let da = combine(d.off_a, d.width_a, vi);
                    let ma = fetch_mult(d.mult_a, d.neg_a, vi);
                    writer.write_frac(ma.into(), da);
                }
            });
        }
        col_gen.finalize_col();
    }
    logup_gen.into_raw()
}
