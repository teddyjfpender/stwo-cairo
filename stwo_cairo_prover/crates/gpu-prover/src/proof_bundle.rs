//! One-copy host boundary for the resident prover.
//!
//! The captured tail gathers every proof-visible device value into this sealed
//! word layout. The host performs one D2H copy and one synchronization, then
//! this module mechanically decodes the buffer into the existing STWO proof
//! adapter inputs. No transcript, hashing, folding, or field arithmetic is
//! performed here.

use core::ops::Range;

use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_backend_cuda::{Blake2sProofAssemblyShape, DecommitAssembly, PreparedDecommitError};

const HASH_WORDS: usize = 8;
const NONCE_WORDS: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentProofBundleLayout {
    pub commitments: Range<usize>,
    pub interaction_claim: Range<usize>,
    pub interaction_pow: Range<usize>,
    pub sampled_values: Range<usize>,
    pub fri_commitments: Range<usize>,
    pub final_line_poly: Range<usize>,
    pub query_pow: Range<usize>,
    pub decommitment: Range<usize>,
    pub total_words: usize,
}

/// Stable device range that the decommit tail may target directly. The range
/// is deliberately pointer-free: the resident runtime binds it to its exact
/// arena allocation, while this module owns the canonical proof-byte offsets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentProofBundleDeviceRange {
    pub offset_words: usize,
    pub len_words: usize,
}

impl ResidentProofBundleLayout {
    pub fn new(
        interaction_claim_words: usize,
        sampled_value_words: usize,
        fri_tree_count: usize,
        final_line_poly_words: usize,
        decommitment_words: usize,
    ) -> Result<Self, ResidentProofBundleError> {
        if interaction_claim_words == 0 || interaction_claim_words % SECURE_EXTENSION_DEGREE != 0 {
            return Err(ResidentProofBundleError::InvalidSectionWidth(
                "interaction claim",
            ));
        }
        if sampled_value_words == 0 || sampled_value_words % SECURE_EXTENSION_DEGREE != 0 {
            return Err(ResidentProofBundleError::InvalidSectionWidth(
                "sampled values",
            ));
        }
        if fri_tree_count == 0 {
            return Err(ResidentProofBundleError::InvalidSectionWidth(
                "FRI commitments",
            ));
        }
        if final_line_poly_words == 0
            || final_line_poly_words % SECURE_EXTENSION_DEGREE != 0
            || decommitment_words == 0
        {
            return Err(ResidentProofBundleError::InvalidSectionWidth(
                "final proof tail",
            ));
        }

        let mut cursor = 0usize;
        let mut take = |words: usize| -> Result<Range<usize>, ResidentProofBundleError> {
            let start = cursor;
            cursor = cursor
                .checked_add(words)
                .ok_or(ResidentProofBundleError::SizeOverflow)?;
            Ok(start..cursor)
        };
        let commitments = take(4 * HASH_WORDS)?;
        let interaction_claim = take(interaction_claim_words)?;
        let interaction_pow = take(NONCE_WORDS)?;
        let sampled_values = take(sampled_value_words)?;
        let fri_commitments = take(
            fri_tree_count
                .checked_mul(HASH_WORDS)
                .ok_or(ResidentProofBundleError::SizeOverflow)?,
        )?;
        let final_line_poly = take(final_line_poly_words)?;
        let query_pow = take(NONCE_WORDS)?;
        let decommitment = take(decommitment_words)?;
        Ok(Self {
            commitments,
            interaction_claim,
            interaction_pow,
            sampled_values,
            fri_commitments,
            final_line_poly,
            query_pow,
            decommitment,
            total_words: cursor,
        })
    }

    pub fn direct_decommitment_destination(&self) -> ResidentProofBundleDeviceRange {
        ResidentProofBundleDeviceRange {
            offset_words: self.decommitment.start,
            len_words: self.decommitment.len(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentProofBundleError {
    InvalidSectionWidth(&'static str),
    SizeOverflow,
    WordCount { expected: usize, actual: usize },
    ProofShape(&'static str),
    NonCanonicalM31 { section: &'static str, word: u32 },
    Decommit(PreparedDecommitError),
}

impl core::fmt::Display for ResidentProofBundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid resident proof bundle: {self:?}")
    }
}

impl std::error::Error for ResidentProofBundleError {}

impl From<PreparedDecommitError> for ResidentProofBundleError {
    fn from(value: PreparedDecommitError) -> Self {
        Self::Decommit(value)
    }
}

pub struct ResidentProofBundle {
    pub commitments: Vec<Blake2sHash>,
    pub interaction_claim: Vec<SecureField>,
    pub interaction_pow: u64,
    pub sampled_values: TreeVec<Vec<Vec<SecureField>>>,
    pub fri_commitments: Vec<Blake2sHash>,
    pub final_line_poly_words: Vec<u32>,
    pub query_pow: u64,
    pub decommitment: DecommitAssembly,
}

impl ResidentProofBundle {
    pub fn decode(
        words: Vec<u32>,
        layout: &ResidentProofBundleLayout,
        shape: &Blake2sProofAssemblyShape,
    ) -> Result<Self, ResidentProofBundleError> {
        if words.len() != layout.total_words {
            return Err(ResidentProofBundleError::WordCount {
                expected: layout.total_words,
                actual: words.len(),
            });
        }
        if shape.trace_trees.len() != 4 {
            return Err(ResidentProofBundleError::ProofShape(
                "resident Starknet proof must contain four trace trees",
            ));
        }
        if shape.fri_trees.len() * HASH_WORDS != layout.fri_commitments.len() {
            return Err(ResidentProofBundleError::ProofShape(
                "FRI tree count disagrees with the sealed bundle layout",
            ));
        }
        let expected_sample_words = shape
            .trace_trees
            .iter()
            .flat_map(|tree| &tree.oods_samples_per_column)
            .try_fold(0usize, |total, &samples| {
                samples
                    .checked_mul(SECURE_EXTENSION_DEGREE)
                    .and_then(|words| total.checked_add(words))
            })
            .ok_or(ResidentProofBundleError::SizeOverflow)?;
        if expected_sample_words != layout.sampled_values.len() {
            return Err(ResidentProofBundleError::ProofShape(
                "OODS sample topology disagrees with the sealed bundle layout",
            ));
        }

        let commitments = decode_hashes(&words[layout.commitments.clone()]);
        let interaction_claim = decode_secure_fields(
            "interaction claim",
            &words[layout.interaction_claim.clone()],
        )?;
        let interaction_pow = decode_nonce(&words[layout.interaction_pow.clone()])?;
        let flat_samples =
            decode_secure_fields("sampled values", &words[layout.sampled_values.clone()])?;
        let mut flat_samples = flat_samples.into_iter();
        let sampled_values = TreeVec(
            shape
                .trace_trees
                .iter()
                .map(|tree| {
                    tree.oods_samples_per_column
                        .iter()
                        .map(|&count| flat_samples.by_ref().take(count).collect())
                        .collect()
                })
                .collect(),
        );
        if flat_samples.next().is_some() {
            return Err(ResidentProofBundleError::ProofShape(
                "unconsumed OODS samples",
            ));
        }
        let fri_commitments = decode_hashes(&words[layout.fri_commitments.clone()]);
        let final_line_poly_words = words[layout.final_line_poly.clone()].to_vec();
        // Final coefficients are field elements too; reject corrupt/non-canonical
        // words before handing them to the STWO adapter.
        decode_secure_fields("final line polynomial", &final_line_poly_words)?;
        let query_pow = decode_nonce(&words[layout.query_pow.clone()])?;
        let decommitment = DecommitAssembly::decode(words[layout.decommitment.clone()].to_vec())?;
        Ok(Self {
            commitments,
            interaction_claim,
            interaction_pow,
            sampled_values,
            fri_commitments,
            final_line_poly_words,
            query_pow,
            decommitment,
        })
    }
}

fn decode_hashes(words: &[u32]) -> Vec<Blake2sHash> {
    words
        .chunks_exact(HASH_WORDS)
        .map(|hash| {
            let mut bytes = [0u8; 32];
            for (&word, destination) in hash.iter().zip(bytes.chunks_exact_mut(4)) {
                destination.copy_from_slice(&word.to_le_bytes());
            }
            Blake2sHash(bytes)
        })
        .collect()
}

fn decode_nonce(words: &[u32]) -> Result<u64, ResidentProofBundleError> {
    let [low, high] = words else {
        return Err(ResidentProofBundleError::InvalidSectionWidth("PoW nonce"));
    };
    Ok(u64::from(*low) | (u64::from(*high) << 32))
}

fn decode_secure_fields(
    section: &'static str,
    words: &[u32],
) -> Result<Vec<SecureField>, ResidentProofBundleError> {
    if words.len() % SECURE_EXTENSION_DEGREE != 0 {
        return Err(ResidentProofBundleError::InvalidSectionWidth(section));
    }
    words
        .chunks_exact(SECURE_EXTENSION_DEGREE)
        .map(|coordinates| {
            let mut values = [M31::from_u32_unchecked(0); SECURE_EXTENSION_DEGREE];
            for (destination, &word) in values.iter_mut().zip(coordinates) {
                if word >= P {
                    return Err(ResidentProofBundleError::NonCanonicalM31 { section, word });
                }
                *destination = M31::from_u32_unchecked(word);
            }
            Ok(SecureField::from_m31_array(values))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::{Blake2sFriAssemblyShape, Blake2sTraceAssemblyShape, TraceTreeRole};

    use super::*;

    #[test]
    fn layout_is_contiguous_and_overflow_checked() {
        let layout = ResidentProofBundleLayout::new(12, 24, 3, 16, 200).unwrap();
        assert_eq!(layout.commitments, 0..32);
        assert_eq!(layout.interaction_claim, 32..44);
        assert_eq!(layout.interaction_pow, 44..46);
        assert_eq!(layout.decommitment.end, layout.total_words);
        assert_eq!(
            layout.direct_decommitment_destination(),
            ResidentProofBundleDeviceRange {
                offset_words: layout.decommitment.start,
                len_words: 200,
            }
        );
        assert!(matches!(
            ResidentProofBundleLayout::new(3, 24, 3, 16, 200),
            Err(ResidentProofBundleError::InvalidSectionWidth(
                "interaction claim"
            ))
        ));
    }

    #[test]
    fn direct_decommit_writes_are_byte_identical_to_the_legacy_tail_copy() {
        let layout = ResidentProofBundleLayout::new(12, 24, 3, 16, 200).unwrap();
        let mut copied = vec![0u32; layout.total_words];
        let mut direct = vec![0u32; layout.total_words];
        let sections = [
            layout.commitments.clone(),
            layout.interaction_claim.clone(),
            layout.interaction_pow.clone(),
            layout.sampled_values.clone(),
            layout.fri_commitments.clone(),
            layout.final_line_poly.clone(),
            layout.query_pow.clone(),
            layout.decommitment.clone(),
        ];
        for (section_index, section) in sections.iter().enumerate() {
            for (word_index, destination) in copied[section.clone()].iter_mut().enumerate() {
                *destination = ((section_index as u32 + 1) << 24) | word_index as u32;
            }
        }

        let decommit = layout.direct_decommitment_destination();
        direct[decommit.offset_words..decommit.offset_words + decommit.len_words]
            .copy_from_slice(&copied[layout.decommitment.clone()]);
        for section in &sections[..sections.len() - 1] {
            direct[section.clone()].copy_from_slice(&copied[section.clone()]);
        }
        assert_eq!(direct, copied);
    }

    #[test]
    fn decoder_rejects_shape_mismatch_before_proof_assembly() {
        let layout = ResidentProofBundleLayout::new(4, 16, 1, 4, 8).unwrap();
        let shape = Blake2sProofAssemblyShape {
            query_log_size: 10,
            n_queries: 2,
            trace_trees: vec![Blake2sTraceAssemblyShape {
                role: TraceTreeRole::Base,
                leaf_log_size: 10,
                query_log_size: 10,
                oods_samples_per_column: vec![1],
                commit_to_proof_column: vec![0],
            }],
            fri_trees: vec![Blake2sFriAssemblyShape {
                evaluation_log_size: 10,
                cumulative_fold: 0,
                outgoing_fold_step: 1,
                log_rows_per_leaf: 0,
            }],
        };
        assert!(matches!(
            ResidentProofBundle::decode(vec![0; layout.total_words], &layout, &shape),
            Err(ResidentProofBundleError::ProofShape(_))
        ));
    }
}
