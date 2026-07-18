//! Exact Interaction commitment-root handoff to the transcript.
//!
//! The commit remains the root producer. This stage only copies its final
//! eight-word output into the distinct transcript-owned input value.

use super::base_commit_projection::{BaseCommitInventory, CommitInventoryKind};
use super::interaction_commit_projection::{
    self, InteractionCommitTranscriptStage, LoweredInteractionCommit,
};
use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, CommitmentTreeId, ProofArenaPlan};
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange,
    ExecutionPrimitive, ProofStage, ValueRange, ValueVersion,
};
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptSegment,
};

const ROOT_WORDS: usize = 8;
const COPY_BYTES: usize = ROOT_WORDS * core::mem::size_of::<u32>();
const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.interaction-root-staging.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionRootSource {
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionRootDestination {
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionRootStaging {
    stage: InteractionCommitTranscriptStage,
    source: LoweredInteractionRootSource,
    destination: LoweredInteractionRootDestination,
    effect: EffectContract,
    digest: [u8; 32],
}

impl LoweredInteractionRootStaging {
    pub(super) const fn stage(&self) -> ProofStage {
        ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)
    }

    pub(super) const fn source(&self) -> LoweredInteractionRootSource {
        self.source
    }

    pub(super) const fn destination(&self) -> LoweredInteractionRootDestination {
        self.destination
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }

    pub(super) const fn primitive(&self) -> ExecutionPrimitive {
        ExecutionPrimitive::DeviceCopyD2D { bytes: COPY_BYTES }
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedRoot {
    source: LoweredInteractionRootSource,
    destination_catalog: ArenaCatalogValueId,
    destination_arena: ArenaBinding,
    stage: InteractionCommitTranscriptStage,
}

/// Lower the copy or publish no transcript-root semantic version.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    commit: &LoweredInteractionCommit,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredInteractionRootStaging, InvocationShapeError> {
    let planned = inventory(arena, transcript, commit)?;
    let mut next_values = values.clone();
    if next_values.version(planned.source.catalog)? != planned.source.version {
        return Err(InvocationShapeError::InvalidInteractionRootStaging);
    }
    let destination = LoweredInteractionRootDestination {
        catalog: planned.destination_catalog,
        arena: planned.destination_arena,
        version: next_values.allocate_output(planned.destination_catalog)?,
    };
    let effect = effect(planned.source, destination)?;
    let mut lowered = LoweredInteractionRootStaging {
        stage: planned.stage,
        source: planned.source,
        destination,
        effect,
        digest: [0; 32],
    };
    lowered.digest = receipt_digest(commit, &lowered)?;
    validate_receipt(arena, transcript, commit, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    commit: &LoweredInteractionCommit,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredInteractionRootStaging,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, transcript, commit, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidInteractionRootStaging)
    }
}

fn inventory(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    commit: &LoweredInteractionCommit,
) -> Result<PlannedRoot, InvocationShapeError> {
    interaction_commit_projection::validate_receipt(arena, transcript, commit)
        .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?;
    let planned_commit = arena
        .commitment(CommitmentTreeId::Interaction)
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    let inventory = BaseCommitInventory::compile_for(
        CommitInventoryKind::Interaction,
        arena,
        planned_commit,
        commit.authority().canonical(),
    )
    .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?;
    let root_role = commit.authority().root();
    let (source_catalog, source_arena) = inventory
        .role(root_role)
        .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?;
    let source_output = commit
        .final_root_output()
        .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?;
    let mut layouts = commit
        .authority()
        .layouts()
        .iter()
        .filter(|layout| layout.role == root_role);
    let root_layout = layouts
        .next()
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    if layouts.next().is_some()
        || root_layout.logical_words != ROOT_WORDS
        || source_output.arena != source_arena
        || source_arena != planned_commit.root
        || source_arena.len_words != ROOT_WORDS
    {
        return Err(InvocationShapeError::InvalidInteractionRootStaging);
    }

    let input = CairoTranscriptInput::InteractionRoot;
    let input_id = input
        .id()
        .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?;
    let mut transcript_requirements = transcript
        .inputs()
        .iter()
        .filter(|requirement| requirement.semantic == input);
    let transcript_requirement = transcript_requirements
        .next()
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    let planned_transcript = arena.transcript();
    let mut arena_requirements = planned_transcript
        .requirements
        .inputs
        .iter()
        .filter(|requirement| requirement.id == input_id);
    let arena_requirement = arena_requirements
        .next()
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    let mut arena_inputs = planned_transcript
        .inputs
        .iter()
        .filter_map(|&(id, binding)| (id == input_id).then_some(binding));
    let expected_arena = arena_inputs
        .next()
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    let (logical, destination_arena) = arena
        .find(None, None, BufferPurpose::TranscriptInput, input_id.0)
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    let catalog = BaseProducerCatalog::compile(arena)?;
    let destination = catalog.value(ArenaCatalogValueId(logical.id.0))?;
    if transcript_requirements.next().is_some()
        || arena_requirements.next().is_some()
        || arena_inputs.next().is_some()
        || transcript_requirement.min_words != ROOT_WORDS
        || arena_requirement.min_words != ROOT_WORDS
        || expected_arena != destination_arena
        || logical.len_words != ROOT_WORDS
        || destination_arena.len_words != ROOT_WORDS
        || destination.logical != logical.id
        || destination.physical != destination_arena.physical
        || destination.purpose != BufferPurpose::TranscriptInput
        || destination.ordinal != input_id.0
        || destination.words != ROOT_WORDS
        || destination.id == source_catalog
    {
        return Err(InvocationShapeError::InvalidInteractionRootStaging);
    }
    Ok(PlannedRoot {
        source: LoweredInteractionRootSource {
            catalog: source_catalog,
            arena: source_arena,
            version: source_output.version,
        },
        destination_catalog: destination.id,
        destination_arena,
        stage: InteractionCommitTranscriptStage::compile(arena, transcript)
            .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)?,
    })
}

fn effect(
    source: LoweredInteractionRootSource,
    destination: LoweredInteractionRootDestination,
) -> Result<EffectContract, InvocationShapeError> {
    let elements = ElementRange::new(0, ROOT_WORDS)
        .ok_or(InvocationShapeError::InvalidInteractionRootStaging)?;
    EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, source.version, elements),
            },
            EffectAccess::Write {
                destination: bound(1, destination.version, elements),
            },
        ],
        vec![],
    )
    .map_err(|_| InvocationShapeError::InvalidInteractionRootStaging)
}

fn validate_receipt(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    commit: &LoweredInteractionCommit,
    lowered: &LoweredInteractionRootStaging,
) -> Result<(), InvocationShapeError> {
    let planned = inventory(arena, transcript, commit)?;
    let destination = LoweredInteractionRootDestination {
        catalog: planned.destination_catalog,
        arena: planned.destination_arena,
        version: lowered.destination.version,
    };
    if planned.source != lowered.source
        || planned.destination_catalog != lowered.destination.catalog
        || planned.destination_arena != lowered.destination.arena
        || planned.stage != lowered.stage
        || effect(planned.source, destination)? != lowered.effect
        || receipt_digest(commit, lowered)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidInteractionRootStaging);
    }
    Ok(())
}

fn receipt_digest(
    commit: &LoweredInteractionCommit,
    lowered: &LoweredInteractionRootStaging,
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&commit.digest());
    hasher.update(&lowered.stage.schedule_key().to_le_bytes());
    hasher.update(&lowered.stage.interaction_root_operation().to_le_bytes());
    for value in [
        lowered.source.catalog.0,
        lowered.source.arena.logical.0,
        lowered.source.arena.physical.0,
        lowered.source.version.0,
        lowered.destination.catalog.0,
        lowered.destination.arena.logical.0,
        lowered.destination.arena.physical.0,
        lowered.destination.version.0,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    for words in [
        lowered.source.arena.len_words,
        lowered.destination.arena.len_words,
    ] {
        hasher.update(
            &u64::try_from(words)
                .map_err(|_| InvocationShapeError::SizeOverflow)?
                .to_le_bytes(),
        );
    }
    let encoding = lowered.effect.canonical_encoding();
    hasher.update(
        &u64::try_from(encoding.len())
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(encoding);
    Ok(*hasher.finalize().as_bytes())
}

const fn bound(binding: u32, version: ValueVersion, elements: ElementRange) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value: ValueRange { version, elements },
    }
}

#[cfg(test)]
#[path = "interaction_root_staging_tests.rs"]
mod tests;
