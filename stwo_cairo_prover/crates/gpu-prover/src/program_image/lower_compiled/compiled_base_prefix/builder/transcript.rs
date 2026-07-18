//! Exact Base-checkpoint handoff into the first transcript segment.

use super::super::super::{base_commit_projection, transcript_semantic_projection};
use super::validation::{seal_base_commit_checkpoint, seal_base_commit_external_roots};
use super::*;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SealedBootstrapTranscript {
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: transcript_semantic_projection::LoweredTranscriptSegment,
}

impl SealedBootstrapTranscript {
    pub(super) const fn lowered(
        &self,
    ) -> &transcript_semantic_projection::LoweredTranscriptSegment {
        &self.lowered
    }
}

impl CompiledBaseDagBuilder {
    /// Release the canonical first transcript segment or publish nothing.
    pub(super) fn append_bootstrap_transcript(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.bootstrap_transcript.is_some() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_base_checkpoint_for_transcript(arena)?;

        let before = self.prefix.values.clone();
        let mut after = before.clone();
        let lowered = transcript_semantic_projection::lower_segment(
            arena,
            transcript,
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &mut after,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        transcript_semantic_projection::validate_from(
            arena,
            transcript,
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &before,
            &after,
            &lowered,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if !lowered.outputs().is_empty() || before != after {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        self.prefix.values = after.clone();
        self.bootstrap_transcript = Some(SealedBootstrapTranscript {
            before,
            after,
            lowered,
        });
        Ok(())
    }

    pub(super) fn has_complete_bootstrap_transcript(&self) -> bool {
        self.bootstrap_transcript.as_ref().is_some_and(|sealed| {
            sealed.before == sealed.after
                && sealed.lowered.segment() == CairoTranscriptSegment::BootstrapThroughBase
        })
    }

    fn validate_base_checkpoint_for_transcript(
        &self,
        arena: &ProofArenaPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        self.validate_arena_authority(arena)?;
        let sealed = self
            .base_commit
            .as_ref()
            .filter(|sealed| sealed.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let expected_digest = sealed
            .checkpoint_digest
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        base_commit_projection::validate_from(
            arena,
            &sealed.before,
            &self.prefix.values,
            &sealed.lowered,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if seal_base_commit_external_roots(&sealed.before, &self.prefix.values, &sealed.lowered)?
            != sealed.external_roots
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let roots = validate_causal_value_closure(
            &self.prefix.values,
            &self.prefix.effects,
            &self.prefix.operations,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if self.prefix.causal_external_roots.as_ref() != Some(&roots)
            || seal_base_commit_checkpoint(
                &sealed.lowered,
                &self.prefix.operations,
                &self.prefix.static_wrappers,
                &roots,
            )? != expected_digest
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        Ok(())
    }
}
