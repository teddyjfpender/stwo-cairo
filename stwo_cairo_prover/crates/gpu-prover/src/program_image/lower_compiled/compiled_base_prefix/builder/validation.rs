//! Structural publication gate for the post-fixed, pre-BaseCommit causal DAG.

use std::collections::{BTreeMap, BTreeSet};

use super::super::super::{adapter, base_commit_projection, fixed_table_materialization};
use super::super::{emission, validate_causal_value_closure, CompiledWitnessWriterPrefix};
use super::{
    CompiledBaseDagAppendError, SealedBaseCommit, SealedFixedTables, SealedMemoryBaseTrace,
};
use crate::arena_plan::{ArenaBinding, BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, EffectContractId, ElementRange,
    ExecutionPrimitive, OpNode, PartitionAuthority, ProofStage, StaticCudaWrapperAuthority,
    ValueVersion,
};
use crate::memory_ledger::MemoryPurposeClass;
use crate::memory_ledger::MemoryPurposeClass::FixedData;
use crate::program_image::ArenaCatalogValueId;
use crate::transcript_plan::CairoTranscriptSegment;

const FIXED_ROOT_DOMAIN: &[u8] = b"stwo-cairo.base.fixed-image-roots.v1\0";
const BASE_COMMIT_CHECKPOINT_DOMAIN: &[u8] = b"stwo-cairo.base-commit.semantic-checkpoint.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
struct FixedImageRootOccurrence {
    table_ordinal: u32,
    source_ordinal: u32,
    identity: String,
    arena: ArenaBinding,
    catalog: ArenaCatalogValueId,
    version: ValueVersion,
    elements: ElementRange,
    binding: EffectBindingId,
}

/// Exact ordered fixed-image reads plus their distinct first allocations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FixedImageRootReceipt {
    occurrences: Vec<FixedImageRootOccurrence>,
    distinct: Vec<FixedImageRootOccurrence>,
    digest: [u8; 32],
}

impl FixedImageRootReceipt {
    pub(super) fn distinct_versions(&self) -> impl Iterator<Item = ValueVersion> + '_ {
        self.distinct.iter().map(|root| root.version)
    }

    #[cfg(test)]
    pub(super) fn occurrence_count(&self) -> usize {
        self.occurrences.len()
    }

    #[cfg(test)]
    pub(super) fn distinct_count(&self) -> usize {
        self.distinct.len()
    }

    #[cfg(test)]
    pub(super) const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    fn validate_against(
        &self,
        lowered: &fixed_table_materialization::LoweredFixedTableStage,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let occurrences = fixed_root_occurrences(lowered)?;
        let mut seen = BTreeSet::new();
        let distinct = occurrences
            .iter()
            .filter(|root| seen.insert(root.catalog))
            .cloned()
            .collect::<Vec<_>>();
        let digest = *blake3::hash(&encode_fixed_roots(&occurrences, &distinct)?).as_bytes();
        if self.occurrences == occurrences && self.distinct == distinct && self.digest == digest {
            Ok(())
        } else {
            Err(CompiledBaseDagAppendError::Lowering)
        }
    }

    #[cfg(test)]
    pub(super) fn omit_last_distinct_for_test(&mut self) {
        self.distinct.pop();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PublishedBaseStage {
    Memory,
    FixedTables,
    BaseCommit,
}

/// Seal every arena-backed fixed source in table/source encounter order.
///
/// Registered Pedersen columns deliberately do not appear here: they are
/// process-owned registered reads and never acquire an SSA `ValueVersion`.
pub(super) fn seal_fixed_image_roots(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    lowered: &fixed_table_materialization::LoweredFixedTableStage,
) -> Result<FixedImageRootReceipt, CompiledBaseDagAppendError> {
    let occurrences = fixed_root_occurrences(lowered)?;
    if occurrences.is_empty() {
        return Err(CompiledBaseDagAppendError::Lowering);
    }

    for root in &occurrences {
        let logical = arena
            .logical_buffers()
            .get(root.arena.logical.0 as usize)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let full =
            ElementRange::new(0, logical.len_words).ok_or(CompiledBaseDagAppendError::Lowering)?;
        if logical.id != root.arena.logical
            || ArenaCatalogValueId(logical.id.0) != root.catalog
            || logical.purpose != BufferPurpose::PreprocessedEvaluations
            || MemoryPurposeClass::of(logical.purpose) != FixedData
            || logical.len_words != root.arena.len_words
            || root.elements != full
            || after.version(root.catalog).ok() != Some(root.version)
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
    }

    let mut seen = BTreeSet::new();
    let distinct = occurrences
        .iter()
        .filter(|root| seen.insert(root.catalog))
        .cloned()
        .collect::<Vec<_>>();
    for root in &distinct {
        if before.version(root.catalog).is_ok()
            || after.versions_for(root.catalog).collect::<Vec<_>>() != vec![root.version]
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
    }
    let canonical = encode_fixed_roots(&occurrences, &distinct)?;
    Ok(FixedImageRootReceipt {
        occurrences,
        distinct,
        digest: *blake3::hash(&canonical).as_bytes(),
    })
}

fn fixed_root_occurrences(
    lowered: &fixed_table_materialization::LoweredFixedTableStage,
) -> Result<Vec<FixedImageRootOccurrence>, CompiledBaseDagAppendError> {
    let mut occurrences = Vec::new();
    for (table_ordinal, table) in lowered.tables().iter().enumerate() {
        for (source_ordinal, source) in table.sources().iter().enumerate() {
            let fixed_table_materialization::LoweredFixedTableSource::Arena {
                identity,
                arena: binding,
                value,
                version,
                elements,
                binding: effect_binding,
            } = source
            else {
                continue;
            };
            let exact_source = table
                .effect()
                .accesses()
                .iter()
                .filter_map(|access| access.source())
                .find(|source| source.binding == *effect_binding)
                .filter(|source| {
                    source.value.version == *version && source.value.elements == *elements
                });
            if exact_source.is_none() {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            occurrences.push(FixedImageRootOccurrence {
                table_ordinal: u32::try_from(table_ordinal)
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                source_ordinal: u32::try_from(source_ordinal)
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                identity: identity.clone(),
                arena: *binding,
                catalog: *value,
                version: *version,
                elements: *elements,
                binding: *effect_binding,
            });
        }
    }
    Ok(occurrences)
}

fn encode_fixed_roots(
    occurrences: &[FixedImageRootOccurrence],
    distinct: &[FixedImageRootOccurrence],
) -> Result<Vec<u8>, CompiledBaseDagAppendError> {
    let mut out = Vec::new();
    out.extend_from_slice(FIXED_ROOT_DOMAIN);
    encode_root_list(&mut out, occurrences)?;
    encode_root_list(&mut out, distinct)?;
    Ok(out)
}

fn encode_root_list(
    out: &mut Vec<u8>,
    roots: &[FixedImageRootOccurrence],
) -> Result<(), CompiledBaseDagAppendError> {
    encode_usize(out, roots.len())?;
    for root in roots {
        out.extend_from_slice(&root.table_ordinal.to_le_bytes());
        out.extend_from_slice(&root.source_ordinal.to_le_bytes());
        encode_usize(out, root.identity.len())?;
        out.extend_from_slice(root.identity.as_bytes());
        out.extend_from_slice(&root.arena.logical.0.to_le_bytes());
        out.extend_from_slice(&root.arena.physical.0.to_le_bytes());
        encode_usize(out, root.arena.len_words)?;
        out.extend_from_slice(&root.catalog.0.to_le_bytes());
        out.extend_from_slice(&root.version.0.to_le_bytes());
        encode_usize(out, root.elements.start)?;
        encode_usize(out, root.elements.end)?;
        out.extend_from_slice(&root.binding.0.to_le_bytes());
    }
    Ok(())
}

fn encode_usize(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledBaseDagAppendError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            .to_le_bytes(),
    );
    Ok(())
}

/// Seal only newly allocated BaseCommit inputs that have no operation writer.
pub(super) fn seal_base_commit_external_roots(
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    lowered: &base_commit_projection::LoweredBaseCommit,
) -> Result<BTreeSet<ValueVersion>, CompiledBaseDagAppendError> {
    base_commit_projection::validate_receipt(lowered)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let before_allocated = before.allocated_versions().collect::<BTreeSet<_>>();
    let after_allocated = after.allocated_versions().collect::<BTreeSet<_>>();
    if !before_allocated.is_subset(&after_allocated) {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let new = after_allocated
        .difference(&before_allocated)
        .copied()
        .collect::<BTreeSet<_>>();
    let mut sources = BTreeSet::new();
    let mut destinations = BTreeSet::new();
    for operation in lowered.operations() {
        for access in operation.effect().accesses() {
            if let Some(source) = access.source() {
                sources.insert(source.value.version);
            }
            if let Some(destination) = access.destination() {
                destinations.insert(destination.value.version);
            }
        }
    }
    if !sources.is_subset(&after_allocated)
        || !destinations.is_subset(&new)
        || new
            .iter()
            .any(|version| !sources.contains(version) && !destinations.contains(version))
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let external = sources
        .difference(&destinations)
        .filter(|version| new.contains(version))
        .copied()
        .collect::<BTreeSet<_>>();
    let (catalog_first, transitions, fixed) = after.allocation_classes();
    if !external.is_subset(&catalog_first)
        || !external.is_disjoint(&transitions)
        || !external.is_disjoint(&fixed)
        || new
            .iter()
            .any(|version| !destinations.contains(version) && !external.contains(version))
    {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    Ok(external)
}

/// Target-independent identity of the exact published BaseCommit suffix.
pub(super) fn seal_base_commit_checkpoint(
    lowered: &base_commit_projection::LoweredBaseCommit,
    operations: &[OpNode],
    wrappers: &[StaticCudaWrapperAuthority],
    roots: &BTreeSet<ValueVersion>,
) -> Result<[u8; 32], CompiledBaseDagAppendError> {
    base_commit_projection::validate_receipt(lowered)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    let count = lowered.operations().len();
    let operation_start = operations
        .len()
        .checked_sub(count)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let wrapper_start = wrappers
        .len()
        .checked_sub(count)
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(BASE_COMMIT_CHECKPOINT_DOMAIN);
    hasher.update(&lowered.digest());
    hash_usize(&mut hasher, operation_start)?;
    hash_usize(&mut hasher, wrapper_start)?;
    hash_usize(&mut hasher, operations.len())?;
    hash_usize(&mut hasher, wrappers.len())?;
    hash_usize(&mut hasher, count)?;
    for (ordinal, local) in lowered.operations().iter().enumerate() {
        let operation = operations
            .get(operation_start + ordinal)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let wrapper = wrappers
            .get(wrapper_start + ordinal)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if local.ordinal() as usize != ordinal
            || operation.id.0 as usize != operation_start + ordinal
            || operation.id.0.checked_add(1) != Some(operation.semantic_id.0)
            || operation.primitive
                != (ExecutionPrimitive::StaticCudaWrapper {
                    wrapper: wrapper.id(),
                })
            || operation.invocation.as_ref() != Some(local.invocation())
            || operation.effect != local.effect().id()
            || operation.stage
                != ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        hasher.update(&local.ordinal().to_le_bytes());
        hasher.update(&operation.id.0.to_le_bytes());
        hasher.update(&operation.semantic_id.0.to_le_bytes());
        hasher.update(&wrapper.id().0.to_le_bytes());
        hasher.update(
            operation
                .invocation
                .as_ref()
                .ok_or(CompiledBaseDagAppendError::Lowering)?
                .contract_id()
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?
                .as_bytes(),
        );
        hasher.update(operation.effect.as_bytes());
        hasher.update(operation.partition.as_bytes());
    }
    hash_usize(&mut hasher, roots.len())?;
    for root in roots {
        hasher.update(&root.0.to_le_bytes());
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) -> Result<(), CompiledBaseDagAppendError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            .to_le_bytes(),
    );
    Ok(())
}

/// Revalidate the exact witness prefix and every published Base operation.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_published_base_dag(
    prefix: &CompiledWitnessWriterPrefix,
    memory: Option<&SealedMemoryBaseTrace>,
    fixed: Option<&SealedFixedTables>,
    base: Option<&SealedBaseCommit>,
    values: &adapter::SemanticValueMap,
    effects: &[EffectContract],
    operations: &[OpNode],
    wrappers: &[StaticCudaWrapperAuthority],
    stage: PublishedBaseStage,
) -> Result<BTreeSet<ValueVersion>, CompiledBaseDagAppendError> {
    let memory_steps = memory
        .and_then(|sealed| sealed.lowered.as_ref())
        .map_or(0, |lowered| lowered.steps().len());
    let fixed_tables = fixed.map_or(0, |sealed| sealed.lowered.tables().len());
    let base_operations = base.map_or(0, |sealed| sealed.lowered.operations().len());
    let valid_stage = match stage {
        PublishedBaseStage::Memory => {
            memory.is_some_and(|sealed| !sealed.emitted) && fixed.is_none() && base.is_none()
        }
        PublishedBaseStage::FixedTables => {
            memory.is_some_and(|sealed| sealed.emitted)
                && fixed.is_some_and(|sealed| !sealed.emitted)
                && base.is_none()
        }
        PublishedBaseStage::BaseCommit => {
            memory.is_some_and(|sealed| sealed.emitted)
                && fixed.is_some_and(|sealed| sealed.emitted)
                && base.is_some_and(|sealed| !sealed.emitted)
        }
    };
    if !valid_stage {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let witness_operations = operations
        .len()
        .checked_sub(memory_steps)
        .and_then(|count| count.checked_sub(fixed_tables))
        .and_then(|count| count.checked_sub(base_operations))
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let witness_wrappers = wrappers
        .len()
        .checked_sub(memory_steps)
        .and_then(|count| count.checked_sub(fixed_tables))
        .and_then(|count| count.checked_sub(base_operations))
        .ok_or(CompiledBaseDagAppendError::Lowering)?;
    let witness_effect_ids = operations[..witness_operations]
        .iter()
        .map(|operation| operation.effect)
        .collect::<BTreeSet<_>>();
    let witness_effects = effects
        .iter()
        .filter(|effect| witness_effect_ids.contains(&effect.id()))
        .cloned()
        .collect::<Vec<_>>();
    let monolithic = PartitionAuthority::monolithic();
    emission::validate_sealed_prefix(
        &prefix.base_authority,
        &prefix.causal_setup,
        prefix.next_producer,
        prefix.target_sm,
        &prefix.kernel_by_build_authority,
        &prefix.kernels,
        &wrappers[..witness_wrappers],
        &prefix.module_global_initializers,
        &witness_effects,
        &prefix.partitions,
        &monolithic,
        &operations[..witness_operations],
    )
    .map_err(|_| CompiledBaseDagAppendError::Lowering)?;

    let effect_by_id = effects
        .iter()
        .map(|effect| (effect.id(), effect))
        .collect::<BTreeMap<_, _>>();
    if effect_by_id.len() != effects.len() {
        return Err(CompiledBaseDagAppendError::Lowering);
    }
    let mut operation_cursor = witness_operations;
    let mut wrapper_cursor = witness_wrappers;
    if let Some(lowered) = memory.and_then(|sealed| sealed.lowered.as_ref()) {
        validate_static_segment(
            lowered
                .steps()
                .iter()
                .map(|step| (step.invocation(), step.effect())),
            &mut operation_cursor,
            &mut wrapper_cursor,
            operations,
            wrappers,
            &effect_by_id,
            prefix.target_sm,
        )?;
    }
    if let Some(sealed) = fixed {
        sealed.fixed_image_roots.validate_against(&sealed.lowered)?;
        validate_static_segment(
            sealed
                .lowered
                .tables()
                .iter()
                .map(|table| (table.invocation(), table.effect())),
            &mut operation_cursor,
            &mut wrapper_cursor,
            operations,
            wrappers,
            &effect_by_id,
            prefix.target_sm,
        )?;
    }
    if let Some(sealed) = base {
        base_commit_projection::validate_receipt(&sealed.lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if seal_base_commit_external_roots(&sealed.before, values, &sealed.lowered)?
            != sealed.external_roots
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        validate_static_segment(
            sealed
                .lowered
                .operations()
                .iter()
                .map(|operation| (operation.invocation(), operation.effect())),
            &mut operation_cursor,
            &mut wrapper_cursor,
            operations,
            wrappers,
            &effect_by_id,
            prefix.target_sm,
        )?;
    }
    if operation_cursor != operations.len()
        || wrapper_cursor != wrappers.len()
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
    validate_causal_value_closure(values, effects, operations)
        .map_err(|_| CompiledBaseDagAppendError::Lowering)
}

#[allow(clippy::too_many_arguments)]
fn validate_static_segment<'a>(
    expected: impl IntoIterator<Item = (&'a AotInvocation, &'a EffectContract)>,
    operation_cursor: &mut usize,
    wrapper_cursor: &mut usize,
    operations: &[OpNode],
    wrappers: &[StaticCudaWrapperAuthority],
    effect_by_id: &BTreeMap<EffectContractId, &EffectContract>,
    target_sm: u32,
) -> Result<(), CompiledBaseDagAppendError> {
    let monolithic = PartitionAuthority::monolithic();
    for (invocation, expected_effect) in expected {
        let operation = operations
            .get(*operation_cursor)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        let wrapper = wrappers
            .get(*wrapper_cursor)
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        if operation.id.0 as usize != *operation_cursor
            || operation.semantic_id.0 as usize != *operation_cursor + 1
            || operation.primitive
                != (ExecutionPrimitive::StaticCudaWrapper {
                    wrapper: wrapper.id(),
                })
            || operation.invocation.as_ref() != Some(invocation)
            || operation.effect != expected_effect.id()
            || operation.partition != monolithic.id()
            || operation.stage
                != ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
            || effect_by_id.get(&operation.effect).copied() != Some(expected_effect)
            || wrapper.id().0 as usize != *wrapper_cursor + 1
            || wrapper.consumer_target_sm() != target_sm
            || wrapper.accepted_invocation()
                != invocation
                    .contract_id()
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            || wrapper.accepted_effect() != expected_effect.id()
            || !wrapper
                .has_valid_identity()
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?
        {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        *operation_cursor += 1;
        *wrapper_cursor += 1;
    }
    Ok(())
}
