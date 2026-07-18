//! Transactional publication of challenge expansion, fused Relation, and tail.

use super::*;
use crate::compiled_proof::{AotInvocation, ProofStage};
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

type RelationStaticResolver =
    fn(
        StaticCudaWrapperId,
        u32,
        &relation_projection::LoweredRelation,
        usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError>;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SealedRelation {
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: relation_projection::LoweredRelation,
    external_roots: BTreeSet<ValueVersion>,
    emitted: bool,
}

impl SealedRelation {
    pub(super) const fn lowered(&self) -> &relation_projection::LoweredRelation {
        &self.lowered
    }

    pub(super) const fn after(&self) -> &adapter::SemanticValueMap {
        &self.after
    }
}

impl CompiledBaseDagBuilder {
    /// Seal Relation semantics after the exact lookup transcript release.
    pub(super) fn append_relation_semantics(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.relation.is_some() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_transcript_chain(arena, transcript, true)?;
        let interaction = self
            .interaction_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if self.prefix.values != *interaction.after() {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let transcript_roots = self
            .interaction_transcript
            .as_ref()
            .map(transcript::SealedTranscriptSegment::external_roots)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if self.prefix.causal_external_roots.as_ref() != Some(transcript_roots) {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let before = self.prefix.values.clone();
        let mut after = before.clone();
        let lowered = relation_projection::lower_stage(arena, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        relation_projection::validate_from(arena, &before, &after, &lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let external_roots = self.relation_external_roots(&before, &lowered)?;

        self.prefix.values = after.clone();
        self.relation = Some(SealedRelation {
            before,
            after,
            lowered,
            external_roots,
            emitted: false,
        });
        Ok(())
    }

    /// Append exactly challenge expansion, fused body, and segmented tail.
    pub(super) fn emit_relation_operations(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        self.emit_relation_operations_using(arena, transcript, resolve_relation_static_wrapper)
    }

    pub(super) fn emit_relation_operations_using(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
        resolve: RelationStaticResolver,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .relation
            .as_ref()
            .filter(|sealed| !sealed.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if self.prefix.partitions != vec![PartitionAuthority::monolithic()] {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.validate_transcript_chain(arena, transcript, true)?;
        let interaction = self
            .interaction_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if sealed.before != *interaction.after()
            || self.prefix.values != sealed.after
            || self.prefix.causal_external_roots
                != self
                    .interaction_transcript
                    .as_ref()
                    .map(|transcript| transcript.external_roots().clone())
            || self.relation_external_roots(&sealed.before, &sealed.lowered)?
                != sealed.external_roots
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        relation_projection::validate_from(arena, &sealed.before, &sealed.after, &sealed.lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;

        let mut effects = exact_effect_map(&self.prefix.effects)?;
        let mut operations = self.prefix.operations.clone();
        let mut wrappers = Vec::with_capacity(3);
        let monolithic = PartitionAuthority::monolithic();
        for operation_ordinal in 0..3 {
            let (invocation, effect) = relation_operation(&sealed.lowered, operation_ordinal)?;
            let existing = self
                .prefix
                .static_wrappers
                .len()
                .checked_add(wrappers.len())
                .ok_or(CompiledBaseDagAppendError::Lowering)?;
            let id = wrapper_id(existing).map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            let Some(wrapper) = resolve(
                id,
                self.prefix.target_sm,
                &sealed.lowered,
                operation_ordinal,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            else {
                return Err(CompiledBaseDagAppendError::MissingRelationStaticWrapper {
                    operation_ordinal: u32::try_from(operation_ordinal)
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                });
            };
            if wrapper.id() != id
                || wrapper.consumer_target_sm() != self.prefix.target_sm
                || wrapper.accepted_invocation()
                    != invocation
                        .contract_id()
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?
                || wrapper.accepted_effect() != effect.id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            insert_effect(&mut effects, effect.clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            push_operation_at_stage(
                ExecutionPrimitive::StaticCudaWrapper { wrapper: id },
                Some(invocation.clone()),
                effect.id(),
                &monolithic,
                ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition),
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }

        let effects = effects.into_values().collect::<Vec<_>>();
        let mut static_wrappers = self.prefix.static_wrappers.clone();
        static_wrappers.extend(wrappers);
        let structural_sources = self.released_transcript_sources()?;
        let roots = validate_causal_value_closure_with_sources(
            &self.prefix.values,
            &effects,
            &operations,
            &structural_sources,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if roots != sealed.external_roots
            || effects
                .iter()
                .map(EffectContract::id)
                .collect::<BTreeSet<_>>()
                != operations
                    .iter()
                    .map(|operation| operation.effect)
                    .collect()
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        self.prefix.static_wrappers = static_wrappers;
        self.prefix.effects = effects;
        self.prefix.operations = operations;
        self.prefix.causal_external_roots = Some(roots);
        self.relation
            .as_mut()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?
            .emitted = true;
        Ok(())
    }

    pub(super) fn has_complete_relation_stage(&self) -> bool {
        self.relation.as_ref().is_some_and(|sealed| sealed.emitted)
    }

    fn relation_external_roots(
        &self,
        before: &adapter::SemanticValueMap,
        lowered: &relation_projection::LoweredRelation,
    ) -> Result<BTreeSet<ValueVersion>, CompiledBaseDagAppendError> {
        let interaction = self
            .interaction_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let output = interaction
            .lowered()
            .output(crate::transcript_plan::CairoTranscriptOutput::CommonLookupElements)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if before.version(output.catalog).ok() != Some(output.binding.value)
            || lowered.challenge().drawn_version() != output.binding.value
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let mut roots = self
            .interaction_transcript
            .as_ref()
            .map(|sealed| sealed.external_roots().clone())
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        roots.insert(output.binding.value);
        Ok(roots)
    }
}

fn relation_operation(
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<(&AotInvocation, &EffectContract), CompiledBaseDagAppendError> {
    match ordinal {
        0 => Ok((
            lowered.challenge().invocation(),
            lowered.challenge().effect(),
        )),
        1 | 2 => lowered
            .wrappers()
            .get(ordinal - 1)
            .map(|wrapper| (wrapper.invocation(), wrapper.effect()))
            .ok_or(CompiledBaseDagAppendError::Lowering),
        _ => Err(CompiledBaseDagAppendError::Lowering),
    }
}

fn resolve_relation_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &relation_projection::LoweredRelation,
    ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    match ordinal {
        0 => relation_projection::resolve_challenge_static_wrapper(id, target_sm, lowered),
        1 | 2 => relation_projection::resolve_wrapper_static_authority(
            id,
            target_sm,
            lowered,
            ordinal - 1,
        ),
        _ => Err(InvocationShapeError::InvalidRelationBinding),
    }
}
