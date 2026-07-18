//! Exact transactional projection of the generated Interaction commitment.
//!
//! CUDA operation semantics are intentionally role-neutral and come from the
//! upstream canonical commit compiler. This layer binds those operations to
//! Interaction-owned arena values, the InteractionCommit lifetime, and the
//! exact transcript segment which absorbs the resulting root.

use stwo_backend_cuda::{InteractionCommitProgramAuthority, TraceTreeRole, TranscriptOperation};

use super::base_commit_projection::{self, CommitInventoryKind, LoweredBaseCommitOperation};
use super::*;
use crate::arena_plan::{CommitmentTreeId, ProofArenaPlan};
use crate::compiled_proof::{StaticCudaWrapperAuthority, StaticCudaWrapperId};
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptBoundary, CairoTranscriptInput,
    CairoTranscriptOutput, CairoTranscriptSegment, TranscriptBoundaryPlan, TranscriptSegmentPlan,
};

const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-interaction-commit.v1\0";
const OPERATION_RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-interaction-commit.operations.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InteractionCommitTranscriptStage {
    schedule_key: u64,
    operation_start: u32,
    operation_end: u32,
    interaction_claim_felts: u32,
    interaction_claim: u32,
    interaction_root: u32,
    composition_random_coefficient: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionCommit {
    authority: InteractionCommitProgramAuthority,
    stage: InteractionCommitTranscriptStage,
    operations: Vec<LoweredBaseCommitOperation>,
    digest: [u8; 32],
}

impl LoweredInteractionCommit {
    pub(super) const fn authority(&self) -> &InteractionCommitProgramAuthority {
        &self.authority
    }

    pub(super) const fn stage(&self) -> InteractionCommitTranscriptStage {
        self.stage
    }

    pub(super) fn operations(&self) -> &[LoweredBaseCommitOperation] {
        &self.operations
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredInteractionCommit, InvocationShapeError> {
    let planned = arena
        .commitment(CommitmentTreeId::Interaction)
        .ok_or(InvocationShapeError::InvalidInteractionCommitAuthority)?;
    let commit = planned
        .commit_program
        .as_ref()
        .ok_or(InvocationShapeError::InvalidInteractionCommitAuthority)?;
    let direct = planned
        .direct_retained_b2n_program
        .as_ref()
        .ok_or(InvocationShapeError::InvalidInteractionCommitAuthority)?;
    let authority = InteractionCommitProgramAuthority::compile(commit, direct)
        .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?;
    if authority.role() != TraceTreeRole::Interaction {
        return Err(InvocationShapeError::InvalidInteractionCommitAuthority);
    }
    authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?;
    let stage = InteractionCommitTranscriptStage::compile(arena, transcript)?;

    let mut next_values = values.clone();
    let operations = base_commit_projection::lower_semantics(
        CommitInventoryKind::Interaction,
        arena,
        planned,
        authority.canonical(),
        &mut next_values,
    )
    .map_err(interaction_projection_error)?;
    let digest = receipt_digest(&authority, stage, &operations)?;
    let lowered = LoweredInteractionCommit {
        authority,
        stage,
        operations,
        digest,
    };
    validate_receipt(arena, transcript, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredInteractionCommit,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, transcript, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidInteractionCommitBinding)
    }
}

pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    lowered: &LoweredInteractionCommit,
    operation_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate_receipt(arena, transcript, lowered)?;
    let Some(linked) = lowered
        .authority
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?
    else {
        return Ok(None);
    };
    linked
        .validate(&lowered.authority)
        .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?;
    base_commit_projection::project_wrapper_parts(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        linked.identity(),
        lowered.authority.canonical(),
        &lowered.operations,
        operation_ordinal,
    )
    .map(Some)
    .map_err(interaction_projection_error)
}

fn validate_receipt(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    lowered: &LoweredInteractionCommit,
) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidInteractionCommitAuthority)?;
    let stage = InteractionCommitTranscriptStage::compile(arena, transcript)?;
    if lowered.authority.role() != TraceTreeRole::Interaction
        || lowered.stage != stage
        || lowered.operations.len() != lowered.authority.operations().len()
        || lowered
            .operations
            .iter()
            .zip(lowered.authority.operations())
            .enumerate()
            .any(|(ordinal, (local, exact))| {
                local.ordinal() as usize != ordinal || local.authority() != exact
            })
        || receipt_digest(&lowered.authority, lowered.stage, &lowered.operations)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidInteractionCommitBinding);
    }
    Ok(())
}

fn receipt_digest(
    authority: &InteractionCommitProgramAuthority,
    stage: InteractionCommitTranscriptStage,
    operations: &[LoweredBaseCommitOperation],
) -> Result<[u8; 32], InvocationShapeError> {
    let operation_receipt = base_commit_projection::receipt_digest_parts(
        OPERATION_RECEIPT_DOMAIN,
        authority.identity(),
        operations,
    )
    .map_err(interaction_projection_error)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hasher.update(&stage.schedule_key.to_le_bytes());
    for ordinal in [
        stage.operation_start,
        stage.operation_end,
        stage.interaction_claim_felts,
        stage.interaction_claim,
        stage.interaction_root,
        stage.composition_random_coefficient,
    ] {
        hasher.update(&ordinal.to_le_bytes());
    }
    hasher.update(&operation_receipt);
    Ok(*hasher.finalize().as_bytes())
}

impl InteractionCommitTranscriptStage {
    pub(super) fn compile(
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, InvocationShapeError> {
        if arena.transcript().schedule_key != transcript.schedule_key() {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        let mut matching = transcript
            .segments()
            .iter()
            .filter(|segment| segment.segment == CairoTranscriptSegment::InteractionAndComposition);
        let segment = matching
            .next()
            .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
        if matching.next().is_some() {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        let stage = Self::compile_segment(transcript, segment)?;
        stage.validate_input(arena, transcript)?;
        Ok(stage)
    }

    pub(super) const fn schedule_key(self) -> u64 {
        self.schedule_key
    }

    pub(super) const fn interaction_claim_felts(self) -> u32 {
        self.interaction_claim_felts
    }

    pub(super) const fn interaction_claim_operation(self) -> u32 {
        self.interaction_claim
    }

    fn compile_segment(
        transcript: &CairoBlake2sTranscriptPlan,
        segment: &TranscriptSegmentPlan,
    ) -> Result<Self, InvocationShapeError> {
        if segment.segment != CairoTranscriptSegment::InteractionAndComposition
            || segment.starts_after != Some(CairoTranscriptBoundary::CommonLookupElements)
            || segment.ends_at != CairoTranscriptBoundary::CompositionRandomCoefficient
            || segment.operation_range.len() != 3
        {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        let operations = transcript
            .schedule()
            .operations()
            .get(segment.operation_range.clone())
            .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
        let claim_boundary = CairoTranscriptBoundary::InteractionClaim
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let claim_input = CairoTranscriptInput::InteractionClaim
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let root_boundary = CairoTranscriptBoundary::InteractionRoot
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let root_input = CairoTranscriptInput::InteractionRoot
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let random_boundary = CairoTranscriptBoundary::CompositionRandomCoefficient
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let random_output = CairoTranscriptOutput::CompositionRandomCoefficient
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let [TranscriptOperation::MixFelts {
            boundary,
            source,
            n_felts,
        }, TranscriptOperation::AbsorbRoot {
            boundary: root,
            source: root_source,
        }, TranscriptOperation::DrawSecureFelt {
            boundary: random,
            output,
        }] = operations
        else {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        };
        if *boundary != claim_boundary
            || *source != claim_input
            || *n_felts == 0
            || *root != root_boundary
            || *root_source != root_input
            || *random != random_boundary
            || *output != random_output
        {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        let boundaries = exact_segment_boundaries(transcript, segment)?;
        let [claim, root, random] = boundaries.as_slice() else {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        };
        let start = segment.operation_range.start;
        if !boundary_is(
            claim,
            CairoTranscriptBoundary::InteractionClaim,
            start,
            segment.segment,
        ) || !boundary_is(
            root,
            CairoTranscriptBoundary::InteractionRoot,
            start + 1,
            segment.segment,
        ) || !boundary_is(
            random,
            CairoTranscriptBoundary::CompositionRandomCoefficient,
            start + 2,
            segment.segment,
        ) {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        Ok(Self {
            schedule_key: transcript.schedule_key(),
            operation_start: u32::try_from(start)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            operation_end: u32::try_from(segment.operation_range.end)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            interaction_claim_felts: *n_felts,
            interaction_claim: u32::try_from(claim.operation_index)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            interaction_root: u32::try_from(root.operation_index)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            composition_random_coefficient: u32::try_from(random.operation_index)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
        })
    }

    fn validate_input(
        self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), InvocationShapeError> {
        let input = CairoTranscriptInput::InteractionClaim;
        let input_id = input
            .id()
            .map_err(|_| InvocationShapeError::InvalidInteractionCommitBinding)?;
        let expected_words = usize::try_from(self.interaction_claim_felts)
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .checked_mul(4)
            .ok_or(InvocationShapeError::SizeOverflow)?;

        let mut transcript_requirements = transcript
            .inputs()
            .iter()
            .filter(|requirement| requirement.semantic == input);
        let transcript_requirement = transcript_requirements
            .next()
            .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
        let planned = arena.transcript();
        let mut arena_requirements = planned
            .requirements
            .inputs
            .iter()
            .filter(|requirement| requirement.id == input_id);
        let arena_requirement = arena_requirements
            .next()
            .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
        let mut arena_inputs = planned
            .inputs
            .iter()
            .filter(|(candidate, _)| *candidate == input_id);
        let (_, binding) = arena_inputs
            .next()
            .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)?;
        if transcript_requirements.next().is_some()
            || arena_requirements.next().is_some()
            || arena_inputs.next().is_some()
            || transcript_requirement.min_words != expected_words
            || arena_requirement.min_words != expected_words
            || binding.len_words != expected_words
            || arena.binding(binding.logical) != Some(*binding)
        {
            return Err(InvocationShapeError::InvalidInteractionCommitBinding);
        }
        Ok(())
    }
}

fn exact_segment_boundaries<'a>(
    transcript: &'a CairoBlake2sTranscriptPlan,
    segment: &TranscriptSegmentPlan,
) -> Result<Vec<&'a TranscriptBoundaryPlan>, InvocationShapeError> {
    let boundaries = transcript
        .boundaries()
        .iter()
        .filter(|boundary| boundary.segment == segment.segment)
        .collect::<Vec<_>>();
    (!boundaries.is_empty())
        .then_some(boundaries)
        .ok_or(InvocationShapeError::InvalidInteractionCommitBinding)
}

fn boundary_is(
    boundary: &TranscriptBoundaryPlan,
    semantic: CairoTranscriptBoundary,
    operation_index: usize,
    segment: CairoTranscriptSegment,
) -> bool {
    boundary.semantic == semantic
        && boundary.operation_index == operation_index
        && boundary.segment == segment
}

fn interaction_projection_error(error: InvocationShapeError) -> InvocationShapeError {
    match error {
        InvocationShapeError::InvalidBaseCommitAuthority => {
            InvocationShapeError::InvalidInteractionCommitAuthority
        }
        InvocationShapeError::SizeOverflow => InvocationShapeError::SizeOverflow,
        _ => InvocationShapeError::InvalidInteractionCommitBinding,
    }
}

#[cfg(test)]
#[path = "interaction_commit_projection_tests.rs"]
mod tests;
