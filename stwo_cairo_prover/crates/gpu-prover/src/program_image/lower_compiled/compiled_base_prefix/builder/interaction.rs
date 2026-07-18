//! Exact Relation-output handoff through the Interaction commitment.

use super::super::super::{
    interaction_claim_staging, interaction_commit_projection, interaction_root_staging,
};
use super::*;
use crate::compiled_proof::ProofStage;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

type InteractionCommitStaticResolver =
    fn(
        StaticCudaWrapperId,
        u32,
        &ProofArenaPlan,
        &CairoBlake2sTranscriptPlan,
        &interaction_commit_projection::LoweredInteractionCommit,
        usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError>;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SealedInteractionStage {
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    claim: interaction_claim_staging::LoweredInteractionClaimStaging,
    commit: interaction_commit_projection::LoweredInteractionCommit,
    root: interaction_root_staging::LoweredInteractionRootStaging,
    emitted: bool,
}

impl SealedInteractionStage {
    pub(super) const fn after(&self) -> &adapter::SemanticValueMap {
        &self.after
    }

    pub(super) const fn commit(&self) -> &interaction_commit_projection::LoweredInteractionCommit {
        &self.commit
    }

    pub(super) const fn claim(&self) -> &interaction_claim_staging::LoweredInteractionClaimStaging {
        &self.claim
    }
}

impl CompiledBaseDagBuilder {
    /// Seal claim staging, Interaction commitment, and root staging atomically.
    pub(super) fn append_interaction_semantics(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.interaction_stage.is_some() || !self.has_complete_relation_stage() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_arena_authority(arena)?;
        self.validate_transcript_chain(arena, transcript, true)?;
        if self
            .relation
            .as_ref()
            .is_none_or(|relation| relation.after() != &self.prefix.values)
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let before = self.prefix.values.clone();
        let mut after = before.clone();
        let claim = interaction_claim_staging::lower_stage(arena, transcript, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let commit = interaction_commit_projection::lower_stage(arena, transcript, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let root = interaction_root_staging::lower_stage(arena, transcript, &commit, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let sealed = SealedInteractionStage {
            before,
            after,
            claim,
            commit,
            root,
            emitted: false,
        };
        validate_interaction_receipt(arena, transcript, &sealed)?;

        self.prefix.values = sealed.after.clone();
        self.interaction_stage = Some(sealed);
        Ok(())
    }

    pub(super) fn emit_interaction_operations(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        self.emit_interaction_operations_using(
            arena,
            transcript,
            interaction_commit_projection::resolve_static_wrapper,
        )
    }

    pub(super) fn emit_interaction_operations_using(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
        resolve: InteractionCommitStaticResolver,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .interaction_stage
            .as_ref()
            .filter(|sealed| !sealed.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        validate_interaction_receipt(arena, transcript, sealed)?;
        if sealed.after != self.prefix.values
            || self.prefix.partitions != vec![PartitionAuthority::monolithic()]
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let monolithic = PartitionAuthority::monolithic();
        let stage = ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition);
        let mut effects = exact_effect_map(&self.prefix.effects)?;
        let mut operations = self.prefix.operations.clone();
        let mut wrappers = Vec::with_capacity(sealed.commit.operations().len());

        for effect in sealed.claim.child_effects() {
            insert_effect(&mut effects, effect.clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        }
        insert_effect(&mut effects, sealed.claim.boundary_effect().clone())
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        push_operation_at_stage(
            sealed.claim.primitive(),
            None,
            sealed.claim.boundary_effect().id(),
            &monolithic,
            stage,
            &mut operations,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;

        for (operation_ordinal, operation) in sealed.commit.operations().iter().enumerate() {
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
                arena,
                transcript,
                &sealed.commit,
                operation_ordinal,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            else {
                return Err(
                    CompiledBaseDagAppendError::MissingInteractionCommitStaticWrapper {
                        operation_ordinal: u32::try_from(operation_ordinal)
                            .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                    },
                );
            };
            if wrapper.id() != id
                || wrapper.consumer_target_sm() != self.prefix.target_sm
                || wrapper.accepted_invocation()
                    != operation
                        .invocation()
                        .contract_id()
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?
                || wrapper.accepted_effect() != operation.effect().id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            insert_effect(&mut effects, operation.effect().clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            push_operation_at_stage(
                ExecutionPrimitive::StaticCudaWrapper { wrapper: id },
                Some(operation.invocation().clone()),
                operation.effect().id(),
                &monolithic,
                stage,
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }

        insert_effect(&mut effects, sealed.root.effect().clone())
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        push_operation_at_stage(
            sealed.root.primitive(),
            None,
            sealed.root.effect().id(),
            &monolithic,
            stage,
            &mut operations,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;

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
        if self.prefix.causal_external_roots.as_ref() != Some(&roots) {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        self.prefix.effects = effects;
        self.prefix.operations = operations;
        self.prefix.static_wrappers = static_wrappers;
        self.interaction_stage
            .as_mut()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?
            .emitted = true;
        Ok(())
    }

    pub(super) fn has_complete_interaction_stage(&self) -> bool {
        self.interaction_stage
            .as_ref()
            .is_some_and(|sealed| sealed.emitted)
    }
}

fn validate_interaction_receipt(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    sealed: &SealedInteractionStage,
) -> Result<(), CompiledBaseDagAppendError> {
    let mut exact = sealed.before.clone();
    let claim = interaction_claim_staging::lower_stage(arena, transcript, &mut exact)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let commit = interaction_commit_projection::lower_stage(arena, transcript, &mut exact)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let root = interaction_root_staging::lower_stage(arena, transcript, &commit, &mut exact)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    if claim == sealed.claim
        && commit == sealed.commit
        && root == sealed.root
        && exact == sealed.after
    {
        Ok(())
    } else {
        Err(CompiledBaseDagAppendError::Lowering)
    }
}
