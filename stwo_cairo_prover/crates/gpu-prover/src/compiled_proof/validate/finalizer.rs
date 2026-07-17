use stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE;
use stwo::core::vcs_lifted::verifier::LOG_PACKED_LEAF_SIZE;
use stwo_backend_cuda::{TraceTreeRole, TranscriptOperation};

use super::*;
use crate::transcript_plan::{CairoTranscriptInput, CairoTranscriptOutput};

const HASH_WORDS: usize = 8;

pub(super) fn validate(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    let finalizer = &input.host_finalizer;
    if finalizer.bundle_codec() != &input.output.codec
        || finalizer.proof_semantic_digest() != input.identity.proof_semantic_digest()
        || finalizer.execution_build_digest() != input.identity.execution_build_digest()
        || !valid_shape(
            finalizer.pcs(),
            finalizer.assembly_shape(),
            &input.output.layout,
        )?
        || !matches_transcript(finalizer, &input.output.layout, transcript)
    {
        return Err(CompiledProofError::InvalidHostFinalizer);
    }
    Ok(())
}

fn matches_transcript(
    finalizer: &HostFinalizerAuthority,
    layout: &crate::proof_bundle::ResidentProofBundleLayout,
    transcript: &CairoBlake2sTranscriptPlan,
) -> bool {
    let input_words = |semantic| {
        transcript
            .inputs()
            .iter()
            .find(|requirement| requirement.semantic == semantic)
            .map(|requirement| requirement.min_words)
    };
    let fri_roots = transcript
        .inputs()
        .iter()
        .filter(|requirement| matches!(requirement.semantic, CairoTranscriptInput::FriLayerRoot(_)))
        .count();
    let query_words = transcript
        .outputs()
        .iter()
        .find(|requirement| requirement.semantic == CairoTranscriptOutput::QueryPositions)
        .map(|requirement| requirement.min_words);
    let query_pow_bits = CairoTranscriptInput::QueryPowNonce
        .id()
        .ok()
        .and_then(|id| {
            transcript
                .schedule()
                .operations()
                .iter()
                .find_map(|operation| match operation {
                    TranscriptOperation::AbsorbPowNonce {
                        source, pow_bits, ..
                    } if *source == id => Some(*pow_bits),
                    _ => None,
                })
        });
    input_words(CairoTranscriptInput::InteractionClaim) == Some(layout.interaction_claim.len())
        && input_words(CairoTranscriptInput::OodsSampledValues) == Some(layout.sampled_values.len())
        && input_words(CairoTranscriptInput::FriLastLayerPolynomial)
            == Some(layout.final_line_poly.len())
        && fri_roots == finalizer.assembly_shape().fri_trees.len()
        && query_words == Some(finalizer.assembly_shape().n_queries)
        && query_pow_bits == Some(finalizer.pcs().pow_bits)
}

fn valid_shape(
    pcs: stwo::core::pcs::PcsConfig,
    shape: &stwo_backend_cuda::Blake2sProofAssemblyShape,
    layout: &crate::proof_bundle::ResidentProofBundleLayout,
) -> Result<bool, CompiledProofError> {
    const ROLES: [TraceTreeRole; 4] = [
        TraceTreeRole::Preprocessed,
        TraceTreeRole::Base,
        TraceTreeRole::Interaction,
        TraceTreeRole::Composition,
    ];
    if !(1..=16).contains(&pcs.fri_config.log_blowup_factor)
        || pcs.fri_config.log_last_layer_degree_bound > 10
        || pcs.fri_config.fold_step == 0
        || shape.query_log_size == 0
        || shape.query_log_size >= 31
        || shape.n_queries == 0
        || shape.n_queries != pcs.fri_config.n_queries
        || pcs
            .lifting_log_size
            .is_some_and(|lifting| lifting != shape.query_log_size)
        || shape.trace_trees.len() != ROLES.len()
        || shape.fri_trees.is_empty()
    {
        return Ok(false);
    }

    let mut sampled_words = 0usize;
    for (tree, expected_role) in shape.trace_trees.iter().zip(ROLES) {
        if tree.role != expected_role
            || tree.leaf_log_size >= 31
            || tree.query_log_size >= 31
            || tree.query_log_size != tree.leaf_log_size
            || (tree.role != TraceTreeRole::Preprocessed
                && tree.query_log_size != shape.query_log_size)
        {
            return Ok(false);
        }
        let mut permutation = tree.commit_to_proof_column.clone();
        permutation.sort_unstable();
        if permutation != (0..tree.oods_samples_per_column.len()).collect::<Vec<_>>() {
            return Ok(false);
        }
        sampled_words =
            tree.oods_samples_per_column
                .iter()
                .try_fold(sampled_words, |total, &samples| {
                    samples
                        .checked_mul(SECURE_EXTENSION_DEGREE)
                        .and_then(|words| total.checked_add(words))
                        .ok_or(CompiledProofError::SizeOverflow)
                })?;
    }

    let final_log_size = pcs
        .fri_config
        .log_last_layer_degree_bound
        .checked_add(pcs.fri_config.log_blowup_factor)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let mut cumulative = 0u32;
    for (index, tree) in shape.fri_trees.iter().enumerate() {
        let Some(evaluation) = shape.query_log_size.checked_sub(cumulative) else {
            return Ok(false);
        };
        let Some(remaining) = evaluation.checked_sub(final_log_size) else {
            return Ok(false);
        };
        let fold = pcs.fri_config.fold_step.min(remaining);
        let packing = if evaluation >= LOG_PACKED_LEAF_SIZE && fold > 1 {
            LOG_PACKED_LEAF_SIZE
        } else {
            0
        };
        if fold == 0
            || tree.cumulative_fold != cumulative
            || tree.evaluation_log_size != evaluation
            || tree.outgoing_fold_step != fold
            || tree.log_rows_per_leaf != packing
        {
            return Ok(false);
        }
        cumulative = cumulative
            .checked_add(fold)
            .ok_or(CompiledProofError::SizeOverflow)?;
        if (evaluation - fold == final_log_size) != (index + 1 == shape.fri_trees.len()) {
            return Ok(false);
        }
    }

    let final_line_felts = 1usize
        .checked_shl(pcs.fri_config.log_last_layer_degree_bound)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let final_line_words = final_line_felts
        .checked_mul(SECURE_EXTENSION_DEGREE)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let fri_commitment_words = shape
        .fri_trees
        .len()
        .checked_mul(HASH_WORDS)
        .ok_or(CompiledProofError::SizeOverflow)?;
    Ok(layout.sampled_values.len() == sampled_words
        && layout.fri_commitments.len() == fri_commitment_words
        && layout.final_line_poly.len() == final_line_words)
}
