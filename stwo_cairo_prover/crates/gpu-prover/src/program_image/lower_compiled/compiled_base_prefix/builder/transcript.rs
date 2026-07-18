//! Exact handoff from BaseCommit through the first two transcript segments.

use super::super::super::{base_commit_projection, transcript_semantic_projection};
use super::validation::{seal_base_commit_checkpoint, seal_base_commit_external_roots};
use super::*;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptSegment,
    CAIRO_STATIC_TRANSCRIPT_INPUTS,
};

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SealedTranscriptSegment {
    pre_seed: adapter::SemanticValueMap,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: transcript_semantic_projection::LoweredTranscriptSegment,
    external_roots: BTreeSet<ValueVersion>,
}

impl SealedTranscriptSegment {
    pub(super) const fn lowered(
        &self,
    ) -> &transcript_semantic_projection::LoweredTranscriptSegment {
        &self.lowered
    }

    pub(super) const fn after(&self) -> &adapter::SemanticValueMap {
        &self.after
    }

    pub(super) const fn external_roots(&self) -> &BTreeSet<ValueVersion> {
        &self.external_roots
    }

    pub(super) fn consumed_versions(&self) -> impl Iterator<Item = ValueVersion> + '_ {
        self.lowered
            .inputs()
            .iter()
            .map(|input| input.binding.value)
    }
}

impl CompiledBaseDagBuilder {
    /// Release the canonical first transcript segment or publish nothing.
    pub(super) fn append_bootstrap_transcript(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.bootstrap_transcript.is_some()
            || self.interaction_transcript.is_some()
            || self.relation.is_some()
        {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_bootstrap_root_stage(arena, transcript)?;
        let roots = self
            .bootstrap_roots
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if &self.prefix.values != roots.after()
            || self.prefix.causal_external_roots.as_ref() != Some(roots.causal_roots())
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let sealed = seal_transcript_segment(
            arena,
            transcript,
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &self.prefix.values,
            &CAIRO_STATIC_TRANSCRIPT_INPUTS,
            roots.causal_roots(),
        )?;
        if !sealed.lowered.outputs().is_empty() || sealed.before != sealed.after {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.validate_transcript_roots(&sealed.before, &[&sealed.lowered], &sealed.external_roots)?;

        self.prefix.values = sealed.after.clone();
        self.prefix.causal_external_roots = Some(sealed.external_roots.clone());
        self.bootstrap_transcript = Some(sealed);
        Ok(())
    }

    /// Release `InteractionPowAndLookup` after the sealed Base transcript.
    pub(super) fn append_interaction_transcript(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.interaction_transcript.is_some() || self.relation.is_some() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_transcript_chain(arena, transcript, false)?;
        let bootstrap = self
            .bootstrap_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if self.prefix.values != bootstrap.after
            || self.prefix.causal_external_roots.as_ref() != Some(&bootstrap.external_roots)
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let sealed = seal_transcript_segment(
            arena,
            transcript,
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap.lowered),
            &self.prefix.values,
            &[CairoTranscriptInput::InteractionPowNonce],
            &bootstrap.external_roots,
        )?;
        if sealed.lowered.outputs().len() != 1
            || sealed
                .lowered
                .output(crate::transcript_plan::CairoTranscriptOutput::CommonLookupElements)
                .is_none()
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.validate_transcript_roots(
            &sealed.before,
            &[&bootstrap.lowered, &sealed.lowered],
            &sealed.external_roots,
        )?;

        self.prefix.values = sealed.after.clone();
        self.prefix.causal_external_roots = Some(sealed.external_roots.clone());
        self.interaction_transcript = Some(sealed);
        Ok(())
    }

    pub(super) fn has_complete_bootstrap_transcript(&self) -> bool {
        self.bootstrap_transcript.as_ref().is_some_and(|sealed| {
            sealed.before == sealed.after
                && sealed.lowered.segment() == CairoTranscriptSegment::BootstrapThroughBase
        })
    }

    pub(super) fn has_complete_interaction_transcript(&self) -> bool {
        self.interaction_transcript.as_ref().is_some_and(|sealed| {
            sealed.lowered.segment() == CairoTranscriptSegment::InteractionPowAndLookup
                && sealed.lowered.outputs().len() == 1
        })
    }

    pub(super) fn validate_transcript_chain(
        &self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
        require_interaction: bool,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let bootstrap = self
            .bootstrap_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        self.validate_bootstrap_root_stage(arena, transcript)?;
        let roots = self
            .bootstrap_roots
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if &bootstrap.pre_seed != roots.after() {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let exact_bootstrap = seal_transcript_segment(
            arena,
            transcript,
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &bootstrap.pre_seed,
            &CAIRO_STATIC_TRANSCRIPT_INPUTS,
            roots.causal_roots(),
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if &exact_bootstrap != bootstrap
            || !bootstrap.lowered.outputs().is_empty()
            || bootstrap.before != bootstrap.after
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.validate_transcript_roots(
            &bootstrap.before,
            &[&bootstrap.lowered],
            &bootstrap.external_roots,
        )?;

        let Some(interaction) = self.interaction_transcript.as_ref() else {
            return if require_interaction {
                Err(CompiledBaseDagAppendError::InvalidStage)
            } else {
                Ok(())
            };
        };
        if interaction.pre_seed != bootstrap.after {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let exact_interaction = seal_transcript_segment(
            arena,
            transcript,
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap.lowered),
            &interaction.pre_seed,
            &[CairoTranscriptInput::InteractionPowNonce],
            &bootstrap.external_roots,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if &exact_interaction != interaction
            || interaction.lowered.outputs().len() != 1
            || interaction
                .lowered
                .output(crate::transcript_plan::CairoTranscriptOutput::CommonLookupElements)
                .is_none()
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.validate_transcript_roots(
            &interaction.before,
            &[&bootstrap.lowered, &interaction.lowered],
            &interaction.external_roots,
        )?;
        Ok(())
    }

    pub(super) fn released_transcript_sources(
        &self,
    ) -> Result<BTreeSet<ValueVersion>, CompiledBaseDagAppendError> {
        let bootstrap = self
            .bootstrap_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let interaction = self
            .interaction_transcript
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        Ok(bootstrap
            .consumed_versions()
            .chain(interaction.consumed_versions())
            .collect())
    }

    fn validate_transcript_roots(
        &self,
        values: &adapter::SemanticValueMap,
        segments: &[&transcript_semantic_projection::LoweredTranscriptSegment],
        expected: &BTreeSet<ValueVersion>,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .bootstrap_roots
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let operation_count = sealed.operation_count();
        let operations = self
            .prefix
            .operations
            .get(..operation_count)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let effect_ids = operations
            .iter()
            .map(|operation| operation.effect)
            .collect::<BTreeSet<_>>();
        let effects = self
            .prefix
            .effects
            .iter()
            .filter(|effect| effect_ids.contains(&effect.id()))
            .cloned()
            .collect::<Vec<_>>();
        let sources = segments
            .iter()
            .flat_map(|segment| segment.inputs())
            .map(|input| input.binding.value)
            .collect::<BTreeSet<_>>();
        let roots =
            validate_causal_value_closure_with_sources(values, &effects, operations, &sources)
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if roots == *expected {
            Ok(())
        } else {
            Err(CompiledBaseDagAppendError::Lowering)
        }
    }

    pub(super) fn validate_base_checkpoint_for_values(
        &self,
        arena: &ProofArenaPlan,
        values: &adapter::SemanticValueMap,
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
        let operation_count = sealed
            .operation_count
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let wrapper_count = sealed
            .wrapper_count
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let operations = self
            .prefix
            .operations
            .get(..operation_count)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let wrappers = self
            .prefix
            .static_wrappers
            .get(..wrapper_count)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let effect_ids = operations
            .iter()
            .map(|operation| operation.effect)
            .collect::<BTreeSet<_>>();
        let effects = self
            .prefix
            .effects
            .iter()
            .filter(|effect| effect_ids.contains(&effect.id()))
            .cloned()
            .collect::<Vec<_>>();
        if effects.len() != effect_ids.len() {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        base_commit_projection::validate_from(arena, &sealed.before, values, &sealed.lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if seal_base_commit_external_roots(&sealed.before, values, &sealed.lowered)?
            != sealed.external_roots
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let roots = validate_causal_value_closure(values, &effects, operations)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if sealed.causal_roots.as_ref() != Some(&roots)
            || seal_base_commit_checkpoint(&sealed.lowered, operations, wrappers, &roots)?
                != expected_digest
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        Ok(())
    }
}

fn seal_transcript_segment(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    expected: CairoTranscriptSegment,
    prior: Option<&transcript_semantic_projection::LoweredTranscriptSegment>,
    pre_seed: &adapter::SemanticValueMap,
    external_inputs: &[CairoTranscriptInput],
    prior_roots: &BTreeSet<ValueVersion>,
) -> Result<SealedTranscriptSegment, CompiledBaseDagAppendError> {
    let mut before = pre_seed.clone();
    let seeded = seed_external_inputs(arena, external_inputs, &mut before)?;
    let mut after = before.clone();
    let lowered = transcript_semantic_projection::lower_segment(
        arena, transcript, expected, prior, &mut after,
    )
    .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    transcript_semantic_projection::validate_from(
        arena, transcript, expected, prior, &before, &after, &lowered,
    )
    .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let mut external_roots = prior_roots.clone();
    external_roots.extend(seeded);
    Ok(SealedTranscriptSegment {
        pre_seed: pre_seed.clone(),
        before,
        after,
        lowered,
        external_roots,
    })
}

fn seed_external_inputs(
    arena: &ProofArenaPlan,
    semantics: &[CairoTranscriptInput],
    values: &mut adapter::SemanticValueMap,
) -> Result<BTreeSet<ValueVersion>, CompiledBaseDagAppendError> {
    let mut catalogs = Vec::with_capacity(semantics.len());
    for &semantic in semantics {
        let id = semantic
            .id()
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let mut matching = arena
            .transcript()
            .inputs
            .iter()
            .filter_map(|(candidate, binding)| (*candidate == id).then_some(binding.logical));
        let logical = matching
            .next()
            .filter(|_| matching.next().is_none())
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let catalog = crate::program_image::ArenaCatalogValueId(logical.0);
        if values.version(catalog).is_ok() || catalogs.contains(&catalog) {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        catalogs.push(catalog);
    }
    values
        .extend_ordered(catalogs.iter().copied())
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    catalogs
        .into_iter()
        .map(|catalog| {
            values
                .version(catalog)
                .map_err(|_| CompiledBaseDagAppendError::Lowering)
        })
        .collect()
}
