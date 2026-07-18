//! Exact commitment-root handoff into the bootstrap transcript.
//!
//! The prepared fixed commitment and generated Base commit remain the root
//! producers. This stage only installs the prepared root as an external source
//! and copies both eight-word roots into their transcript-owned destinations.

use stwo_backend_cuda::BaseCommitAccessKind;

use super::super::super::base_commit_projection::{BaseCommitInventory, CommitInventoryKind};
use super::super::super::{base_commit_projection, BaseProducerCatalog};
use super::*;
use crate::arena_plan::{
    ArenaBinding, BufferLifetime, BufferPurpose, CommitmentTreeId, ProofArenaPlan, ProofEpoch,
};
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange,
    ExecutionPrimitive, PartitionAuthority, ProofStage, ValueRange, ValueVersion,
};
use crate::program_image::ArenaCatalogValueId;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptSegment,
};

const ROOT_WORDS: usize = 8;
const COPY_BYTES: usize = ROOT_WORDS * core::mem::size_of::<u32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RootValue {
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
    version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RootCopy {
    input: CairoTranscriptInput,
    source: RootValue,
    destination: RootValue,
    effect: EffectContract,
}

impl RootCopy {
    const fn stage(&self) -> ProofStage {
        ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
    }

    const fn primitive(&self) -> ExecutionPrimitive {
        ExecutionPrimitive::DeviceCopyD2D { bytes: COPY_BYTES }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LoweredBootstrapRoots {
    preprocessed: RootCopy,
    base: RootCopy,
}

impl LoweredBootstrapRoots {
    fn copies(&self) -> [&RootCopy; 2] {
        [&self.preprocessed, &self.base]
    }

    const fn prepared_source(&self) -> ValueVersion {
        self.preprocessed.source.version
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SealedBootstrapRoots {
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredBootstrapRoots,
    causal_roots: BTreeSet<ValueVersion>,
    operation_count: usize,
}

impl SealedBootstrapRoots {
    pub(super) const fn after(&self) -> &adapter::SemanticValueMap {
        &self.after
    }

    pub(super) const fn causal_roots(&self) -> &BTreeSet<ValueVersion> {
        &self.causal_roots
    }

    pub(super) const fn operation_count(&self) -> usize {
        self.operation_count
    }
}

impl CompiledBaseDagBuilder {
    /// Publish both real root-copy operations or leave the Base checkpoint
    /// untouched. No transcript input is classified as an external seed.
    pub(super) fn append_bootstrap_root_stage(
        &mut self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.bootstrap_roots.is_some()
            || self.bootstrap_transcript.is_some()
            || self.interaction_transcript.is_some()
            || self.relation.is_some()
        {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_base_checkpoint_for_values(arena, &self.prefix.values)?;
        let base = self
            .base_commit
            .as_ref()
            .filter(|sealed| sealed.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let base_roots = base
            .causal_roots
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if self.prefix.causal_external_roots.as_ref() != Some(base_roots) {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let before = self.prefix.values.clone();
        let (after, lowered) = lower_stage(arena, transcript, &base.lowered, &before)?;
        validate_from(arena, transcript, &base.lowered, &before, &after, &lowered)?;

        let mut effects = exact_effect_map(&self.prefix.effects)?;
        let mut operations = self.prefix.operations.clone();
        let monolithic = PartitionAuthority::monolithic();
        for copy in lowered.copies() {
            insert_effect(&mut effects, copy.effect.clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            push_operation_at_stage(
                copy.primitive(),
                None,
                copy.effect.id(),
                &monolithic,
                copy.stage(),
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        }
        let effects = effects.into_values().collect::<Vec<_>>();
        let roots = validate_causal_value_closure(&after, &effects, &operations)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let mut expected_roots = base_roots.clone();
        if !expected_roots.insert(lowered.prepared_source()) || roots != expected_roots {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let sealed = SealedBootstrapRoots {
            before,
            after: after.clone(),
            lowered,
            causal_roots: roots.clone(),
            operation_count: operations.len(),
        };
        self.prefix.values = after;
        self.prefix.effects = effects;
        self.prefix.operations = operations;
        self.prefix.causal_external_roots = Some(roots);
        self.bootstrap_roots = Some(sealed);
        Ok(())
    }

    pub(super) fn has_complete_bootstrap_root_stage(&self) -> bool {
        match (
            self.bootstrap_roots.as_ref(),
            self.base_commit
                .as_ref()
                .and_then(|base| base.operation_count),
        ) {
            (Some(sealed), Some(base_count)) => {
                base_count.checked_add(2) == Some(sealed.operation_count)
            }
            _ => false,
        }
    }

    pub(super) fn validate_bootstrap_root_stage(
        &self,
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .bootstrap_roots
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        self.validate_base_checkpoint_for_values(arena, &sealed.before)?;
        let base = self
            .base_commit
            .as_ref()
            .filter(|base| base.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        validate_from(
            arena,
            transcript,
            &base.lowered,
            &sealed.before,
            &sealed.after,
            &sealed.lowered,
        )?;

        let base_count = base
            .operation_count
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if sealed.operation_count
            != base_count
                .checked_add(2)
                .ok_or(CompiledBaseDagAppendError::Lowering)?
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let mut exact_operations = self
            .prefix
            .operations
            .get(..base_count)
            .ok_or(CompiledBaseDagAppendError::Lowering)?
            .to_vec();
        let monolithic = PartitionAuthority::monolithic();
        for copy in sealed.lowered.copies() {
            push_operation_at_stage(
                copy.primitive(),
                None,
                copy.effect.id(),
                &monolithic,
                copy.stage(),
                &mut exact_operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        }
        if self.prefix.operations.get(..sealed.operation_count) != Some(exact_operations.as_slice())
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let effect_ids = exact_operations
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
        if effects.len() != effect_ids.len()
            || sealed
                .lowered
                .copies()
                .iter()
                .any(|copy| !effects.contains(&copy.effect))
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let roots = validate_causal_value_closure(&sealed.after, &effects, &exact_operations)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let mut expected_roots = base
            .causal_roots
            .clone()
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if !expected_roots.insert(sealed.lowered.prepared_source())
            || roots != expected_roots
            || roots != sealed.causal_roots
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        Ok(())
    }
}

fn lower_stage(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    base: &base_commit_projection::LoweredBaseCommit,
    before: &adapter::SemanticValueMap,
) -> Result<(adapter::SemanticValueMap, LoweredBootstrapRoots), CompiledBaseDagAppendError> {
    let (preprocessed_catalog, preprocessed_arena) = prepared_root(arena)?;
    let base_source = base_root(arena, base, before)?;
    let (preprocessed_destination_catalog, preprocessed_destination_arena) =
        transcript_root(arena, transcript, CairoTranscriptInput::PreprocessedRoot)?;
    let (base_destination_catalog, base_destination_arena) =
        transcript_root(arena, transcript, CairoTranscriptInput::BaseRoot)?;
    if preprocessed_catalog == base_source.catalog
        || preprocessed_catalog == preprocessed_destination_catalog
        || preprocessed_catalog == base_destination_catalog
        || base_source.catalog == preprocessed_destination_catalog
        || base_source.catalog == base_destination_catalog
        || preprocessed_destination_catalog == base_destination_catalog
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }

    let mut after = before.clone();
    if after.version(preprocessed_catalog).is_ok()
        || after.version(preprocessed_destination_catalog).is_ok()
        || after.version(base_destination_catalog).is_ok()
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    after
        .extend_ordered([preprocessed_catalog])
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let preprocessed_source = RootValue {
        catalog: preprocessed_catalog,
        arena: preprocessed_arena,
        version: after
            .version(preprocessed_catalog)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
    };
    let preprocessed_destination = allocate_destination(
        &mut after,
        preprocessed_destination_catalog,
        preprocessed_destination_arena,
    )?;
    let base_destination =
        allocate_destination(&mut after, base_destination_catalog, base_destination_arena)?;
    Ok((
        after,
        LoweredBootstrapRoots {
            preprocessed: root_copy(
                CairoTranscriptInput::PreprocessedRoot,
                preprocessed_source,
                preprocessed_destination,
            )?,
            base: root_copy(
                CairoTranscriptInput::BaseRoot,
                base_source,
                base_destination,
            )?,
        },
    ))
}

fn validate_from(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    base: &base_commit_projection::LoweredBaseCommit,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredBootstrapRoots,
) -> Result<(), CompiledBaseDagAppendError> {
    let (exact_after, exact) = lower_stage(arena, transcript, base, before)?;
    if &exact_after == after && &exact == supplied {
        Ok(())
    } else {
        Err(CompiledBaseDagAppendError::Lowering)
    }
}

fn prepared_root(
    arena: &ProofArenaPlan,
) -> Result<(ArenaCatalogValueId, ArenaBinding), CompiledBaseDagAppendError> {
    let planned = arena
        .commitment(CommitmentTreeId::Preprocessed)
        .filter(|planned| planned.id == CommitmentTreeId::Preprocessed)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    if planned.root.len_words != ROOT_WORDS
        || planned.retained_layers_bottom_up.last() != Some(&planned.root)
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let logical = arena
        .logical_buffers()
        .get(planned.root.logical.0 as usize)
        .filter(|logical| logical.id == planned.root.logical)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let catalog =
        BaseProducerCatalog::compile(arena).map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let id = ArenaCatalogValueId(planned.root.logical.0);
    let value = catalog
        .value(id)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    if arena.binding(logical.id) != Some(planned.root)
        || logical.purpose != BufferPurpose::RetainedMerkleLayers
        || logical.len_words != ROOT_WORDS
        || logical.lifetime
            != (BufferLifetime {
                first: ProofEpoch::Ingest,
                last: ProofEpoch::Assemble,
            })
        || value.logical != logical.id
        || value.physical != planned.root.physical
        || value.purpose != logical.purpose
        || value.words != ROOT_WORDS
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    Ok((id, planned.root))
}

fn base_root(
    arena: &ProofArenaPlan,
    base: &base_commit_projection::LoweredBaseCommit,
    values: &adapter::SemanticValueMap,
) -> Result<RootValue, CompiledBaseDagAppendError> {
    base_commit_projection::validate_receipt(base)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let planned = arena
        .commitment(CommitmentTreeId::Base)
        .filter(|planned| planned.id == CommitmentTreeId::Base)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let inventory = BaseCommitInventory::compile_for(
        CommitInventoryKind::Base,
        arena,
        planned,
        base.authority(),
    )
    .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let root_role = base.authority().root();
    let (catalog, binding) = inventory
        .role(root_role)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let (last, prefix) = base
        .operations()
        .split_last()
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    if prefix
        .iter()
        .flat_map(|operation| operation.accesses())
        .any(|access| access.role == root_role && access.kind == BaseCommitAccessKind::Write)
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let mut writes = last
        .accesses()
        .iter()
        .filter(|access| access.role == root_role && access.kind == BaseCommitAccessKind::Write);
    let output = writes.next().ok_or(CompiledBaseDagAppendError::Lowering)?;
    let authority = last
        .authority()
        .effect
        .accesses
        .get(
            usize::try_from(output.authority_index)
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
        )
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    if writes.next().is_some()
        || authority.kind != BaseCommitAccessKind::Write
        || authority.role != root_role
        || authority.first_word != 0
        || authority.word_len != ROOT_WORDS
        || output.arena != binding
        || binding != planned.root
        || binding.len_words != ROOT_WORDS
        || values.version(catalog).ok() != Some(output.version)
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    Ok(RootValue {
        catalog,
        arena: binding,
        version: output.version,
    })
}

fn transcript_root(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    input: CairoTranscriptInput,
) -> Result<(ArenaCatalogValueId, ArenaBinding), CompiledBaseDagAppendError> {
    let id = input
        .id()
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let mut transcript_requirements = transcript
        .inputs()
        .iter()
        .filter(|requirement| requirement.semantic == input);
    let transcript_requirement = transcript_requirements
        .next()
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let mut arena_requirements = arena
        .transcript()
        .requirements
        .inputs
        .iter()
        .filter(|requirement| requirement.id == id);
    let arena_requirement = arena_requirements
        .next()
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let mut inputs = arena
        .transcript()
        .inputs
        .iter()
        .filter_map(|&(candidate, binding)| (candidate == id).then_some(binding));
    let binding = inputs.next().ok_or(CompiledBaseDagAppendError::Lowering)?;
    let logical = arena
        .logical_buffers()
        .get(binding.logical.0 as usize)
        .filter(|logical| logical.id == binding.logical)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let catalog =
        BaseProducerCatalog::compile(arena).map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let value = catalog
        .value(ArenaCatalogValueId(logical.id.0))
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    if transcript_requirements.next().is_some()
        || arena_requirements.next().is_some()
        || inputs.next().is_some()
        || transcript_requirement.min_words != ROOT_WORDS
        || arena_requirement.min_words != ROOT_WORDS
        || logical.purpose != BufferPurpose::TranscriptInput
        || logical.ordinal != id.0
        || logical.len_words != ROOT_WORDS
        || binding.len_words != ROOT_WORDS
        || arena.binding(logical.id) != Some(binding)
        || value.logical != logical.id
        || value.physical != binding.physical
        || value.purpose != logical.purpose
        || value.ordinal != id.0
        || value.words != ROOT_WORDS
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    Ok((value.id, binding))
}

fn allocate_destination(
    values: &mut adapter::SemanticValueMap,
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
) -> Result<RootValue, CompiledBaseDagAppendError> {
    Ok(RootValue {
        catalog,
        arena,
        version: values
            .allocate_output(catalog)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
    })
}

fn root_copy(
    input: CairoTranscriptInput,
    source: RootValue,
    destination: RootValue,
) -> Result<RootCopy, CompiledBaseDagAppendError> {
    let elements = ElementRange::new(0, ROOT_WORDS).ok_or(CompiledBaseDagAppendError::Lowering)?;
    let effect = EffectContract::new(
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
    .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    Ok(RootCopy {
        input,
        source,
        destination,
        effect,
    })
}

const fn bound(binding: u32, version: ValueVersion, elements: ElementRange) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value: ValueRange { version, elements },
    }
}
