//! Exact Interaction claim staging from Relation outputs.
//!
//! The eager runtime submits one ordered 16-byte D2D copy per claimed sum.
//! This projection preserves that sequence as one `OrderedComposite`: each
//! child writes one disjoint range of the single transcript-owned output.

use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::interaction_commit_projection::InteractionCommitTranscriptStage;
use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange, ExecutableStep,
    ExecutionPrimitive, ProofStage, ValueRange, ValueVersion,
};
use crate::relation_execution::interaction_claim_order_key;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptSegment,
};

const CLAIM_WORDS: usize = 4;
const COPY_BYTES: usize = CLAIM_WORDS * core::mem::size_of::<u32>();
const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.interaction-claim-staging.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InteractionClaimSourceKey {
    pub(super) component: &'static str,
    pub(super) part: TracePartId,
    pub(super) instance: u32,
    pub(super) relation_ordinal: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionClaimSource {
    pub(super) key: InteractionClaimSourceKey,
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionClaimDestination {
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) version: ValueVersion,
    pub(super) words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredInteractionClaimStaging {
    stage: InteractionCommitTranscriptStage,
    sources: Vec<LoweredInteractionClaimSource>,
    destination: LoweredInteractionClaimDestination,
    children: Box<[ExecutableStep]>,
    child_effects: Vec<EffectContract>,
    boundary_effect: EffectContract,
    digest: [u8; 32],
}

impl LoweredInteractionClaimStaging {
    pub(super) const fn stage(&self) -> ProofStage {
        ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)
    }

    pub(super) fn sources(&self) -> &[LoweredInteractionClaimSource] {
        &self.sources
    }

    pub(super) const fn destination(&self) -> LoweredInteractionClaimDestination {
        self.destination
    }

    pub(super) fn children(&self) -> &[ExecutableStep] {
        &self.children
    }

    pub(super) fn child_effects(&self) -> &[EffectContract] {
        &self.child_effects
    }

    pub(super) const fn boundary_effect(&self) -> &EffectContract {
        &self.boundary_effect
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub(super) fn primitive(&self) -> ExecutionPrimitive {
        ExecutionPrimitive::OrderedComposite {
            children: self.children.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedSource {
    key: InteractionClaimSourceKey,
    order: (usize, u8, usize),
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedDestination {
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
    words: usize,
}

/// Lower all copies or publish no semantic version.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredInteractionClaimStaging, InvocationShapeError> {
    let (planned_sources, planned_destination, stage) = inventory(arena, transcript)?;
    let mut next_values = values.clone();
    let sources = planned_sources
        .iter()
        .map(|source| {
            Ok(LoweredInteractionClaimSource {
                key: source.key,
                catalog: source.catalog,
                arena: source.arena,
                version: next_values.version(source.catalog)?,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let destination = LoweredInteractionClaimDestination {
        catalog: planned_destination.catalog,
        arena: planned_destination.arena,
        version: next_values.allocate_output(planned_destination.catalog)?,
        words: planned_destination.words,
    };
    let (children, child_effects, boundary_effect) = execution(&sources, destination)?;
    let mut lowered = LoweredInteractionClaimStaging {
        stage,
        sources,
        destination,
        children,
        child_effects,
        boundary_effect,
        digest: [0; 32],
    };
    lowered.digest = receipt_digest(&lowered)?;
    validate_receipt(arena, transcript, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

/// Rebuild from the pre-stage allocator and require the exact post-state.
pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredInteractionClaimStaging,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, transcript, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidInteractionClaimStaging)
    }
}

fn inventory(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<
    (
        Vec<PlannedSource>,
        PlannedDestination,
        InteractionCommitTranscriptStage,
    ),
    InvocationShapeError,
> {
    let relation = arena.relation();
    if relation.source_plan.is_empty()
        || relation.source_plan.len() != relation.requirements.instances.len()
        || relation.source_plan.len() != relation.slots.instances.len()
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }
    let catalog = BaseProducerCatalog::compile(arena)?;
    let mut sources = relation
        .source_plan
        .iter()
        .enumerate()
        .map(|(ordinal, source)| {
            let requirement = relation
                .requirements
                .instances
                .get(ordinal)
                .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?;
            let expected_slot = relation
                .slots
                .instances
                .get(ordinal)
                .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?
                .claimed_sum;
            let relation_ordinal =
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?;
            let (logical, arena_binding) = arena
                .find(
                    None,
                    None,
                    BufferPurpose::RelationClaimedSum,
                    relation_ordinal,
                )
                .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?;
            let value = catalog.value(ArenaCatalogValueId(logical.id.0))?;
            let order = interaction_claim_order_key(source.batch, source.instance_index)
                .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?;
            if relation.execution.batches.get(requirement.batch_index) != Some(&source.batch)
                || requirement.instance_index != source.instance_index
                || requirement.claimed_sum_words != CLAIM_WORDS
                || logical.len_words != CLAIM_WORDS
                || arena_binding.len_words != CLAIM_WORDS
                || arena_binding.physical != expected_slot
                || value.logical != logical.id
                || value.physical != arena_binding.physical
                || value.purpose != BufferPurpose::RelationClaimedSum
                || value.ordinal != relation_ordinal
                || value.words != CLAIM_WORDS
            {
                return Err(InvocationShapeError::InvalidInteractionClaimStaging);
            }
            Ok(PlannedSource {
                key: InteractionClaimSourceKey {
                    component: source.batch.component,
                    part: source.part,
                    instance: u32::try_from(source.instance_index)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                    relation_ordinal,
                },
                order,
                catalog: value.id,
                arena: arena_binding,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    sources.sort_unstable_by_key(|source| source.order);
    if sources
        .windows(2)
        .any(|pair| pair[0].order >= pair[1].order)
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }

    let input_id = CairoTranscriptInput::InteractionClaim
        .id()
        .map_err(|_| InvocationShapeError::InvalidInteractionClaimStaging)?;
    let mut inputs = arena
        .transcript()
        .inputs
        .iter()
        .filter_map(|&(id, binding)| (id == input_id).then_some(binding));
    let expected_binding = inputs
        .next()
        .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?;
    let (logical, arena_binding) = arena
        .find(None, None, BufferPurpose::TranscriptInput, input_id.0)
        .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?;
    let value = catalog.value(ArenaCatalogValueId(logical.id.0))?;
    let words = sources
        .len()
        .checked_mul(CLAIM_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if inputs.next().is_some()
        || expected_binding != arena_binding
        || logical.len_words != words
        || arena_binding.len_words != words
        || value.logical != logical.id
        || value.physical != arena_binding.physical
        || value.purpose != BufferPurpose::TranscriptInput
        || value.ordinal != input_id.0
        || value.words != words
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }
    let stage = InteractionCommitTranscriptStage::compile(arena, transcript)?;
    if usize::try_from(stage.interaction_claim_felts())
        .map_err(|_| InvocationShapeError::SizeOverflow)?
        != sources.len()
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }
    Ok((
        sources,
        PlannedDestination {
            catalog: value.id,
            arena: arena_binding,
            words,
        },
        stage,
    ))
}

fn execution(
    sources: &[LoweredInteractionClaimSource],
    destination: LoweredInteractionClaimDestination,
) -> Result<(Box<[ExecutableStep]>, Vec<EffectContract>, EffectContract), InvocationShapeError> {
    if sources
        .len()
        .checked_mul(CLAIM_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?
        != destination.words
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }
    let mut children = Vec::with_capacity(sources.len());
    let mut child_effects = Vec::with_capacity(sources.len());
    let mut boundary = Vec::with_capacity(sources.len() * 2);
    for (index, source) in sources.iter().enumerate() {
        let start = index
            .checked_mul(CLAIM_WORDS)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let source_range = value_range(source.version, 0, CLAIM_WORDS)?;
        let destination_range = value_range(destination.version, start, start + CLAIM_WORDS)?;
        let child_effect = EffectContract::new(
            vec![
                EffectAccess::Read {
                    source: bound(0, source_range),
                },
                EffectAccess::Write {
                    destination: bound(1, destination_range),
                },
            ],
            vec![],
        )
        .map_err(|_| InvocationShapeError::InvalidInteractionClaimStaging)?;
        let read_binding = u32::try_from(
            index
                .checked_mul(2)
                .ok_or(InvocationShapeError::SizeOverflow)?,
        )
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
        boundary.extend([
            EffectAccess::Read {
                source: bound(read_binding, source_range),
            },
            EffectAccess::Write {
                destination: bound(
                    read_binding
                        .checked_add(1)
                        .ok_or(InvocationShapeError::SizeOverflow)?,
                    destination_range,
                ),
            },
        ]);
        children.push(ExecutableStep {
            primitive: ExecutionPrimitive::DeviceCopyD2D { bytes: COPY_BYTES },
            invocation: None,
            effect: child_effect.id(),
        });
        child_effects.push(child_effect);
    }
    let boundary_effect = EffectContract::new(boundary, vec![])
        .map_err(|_| InvocationShapeError::InvalidInteractionClaimStaging)?;
    Ok((children.into_boxed_slice(), child_effects, boundary_effect))
}

fn validate_receipt(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    lowered: &LoweredInteractionClaimStaging,
) -> Result<(), InvocationShapeError> {
    let (planned_sources, planned_destination, stage) = inventory(arena, transcript)?;
    let expected_sources = planned_sources
        .iter()
        .zip(&lowered.sources)
        .all(|(planned, actual)| {
            planned.key == actual.key
                && planned.catalog == actual.catalog
                && planned.arena == actual.arena
        });
    let (children, child_effects, boundary_effect) =
        execution(&lowered.sources, lowered.destination)?;
    if !expected_sources
        || planned_sources.len() != lowered.sources.len()
        || stage != lowered.stage
        || planned_destination.catalog != lowered.destination.catalog
        || planned_destination.arena != lowered.destination.arena
        || planned_destination.words != lowered.destination.words
        || children != lowered.children
        || child_effects != lowered.child_effects
        || boundary_effect != lowered.boundary_effect
        || receipt_digest(lowered)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidInteractionClaimStaging);
    }
    Ok(())
}

fn receipt_digest(
    lowered: &LoweredInteractionClaimStaging,
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&lowered.stage.schedule_key().to_le_bytes());
    hasher.update(&lowered.stage.interaction_claim_operation().to_le_bytes());
    hash_size(&mut hasher, lowered.sources.len())?;
    for source in &lowered.sources {
        hash_bytes(&mut hasher, source.key.component.as_bytes())?;
        hash_part(&mut hasher, source.key.part);
        for value in [
            source.key.instance,
            source.key.relation_ordinal,
            source.catalog.0,
            source.arena.logical.0,
            source.arena.physical.0,
            source.version.0,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        hash_size(&mut hasher, source.arena.len_words)?;
    }
    for value in [
        lowered.destination.catalog.0,
        lowered.destination.arena.logical.0,
        lowered.destination.arena.physical.0,
        lowered.destination.version.0,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    hash_size(&mut hasher, lowered.destination.arena.len_words)?;
    hash_size(&mut hasher, lowered.destination.words)?;
    for effect in &lowered.child_effects {
        hash_bytes(&mut hasher, effect.canonical_encoding())?;
    }
    hash_bytes(&mut hasher, lowered.boundary_effect.canonical_encoding())?;
    Ok(*hasher.finalize().as_bytes())
}

fn value_range(
    version: ValueVersion,
    start: usize,
    end: usize,
) -> Result<ValueRange, InvocationShapeError> {
    Ok(ValueRange {
        version,
        elements: ElementRange::new(start, end)
            .ok_or(InvocationShapeError::InvalidInteractionClaimStaging)?,
    })
}

const fn bound(binding: u32, value: ValueRange) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value,
    }
}

fn hash_part(hasher: &mut blake3::Hasher, part: TracePartId) {
    match part {
        TracePartId::Main => {
            hasher.update(&[0]);
        }
        TracePartId::MemoryBig(index) => {
            hasher.update(&[1]);
            hasher.update(&index.to_le_bytes());
        }
        TracePartId::MemorySmall => {
            hasher.update(&[2]);
        }
    }
}

fn hash_size(hasher: &mut blake3::Hasher, value: usize) -> Result<(), InvocationShapeError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<(), InvocationShapeError> {
    hasher.update(
        &u64::try_from(bytes.len())
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

#[cfg(test)]
#[path = "interaction_claim_staging_tests.rs"]
mod tests;
