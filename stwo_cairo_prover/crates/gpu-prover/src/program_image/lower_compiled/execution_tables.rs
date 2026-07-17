//! Semantic lowering for canonical execution-table host ingress and limb splits.
//!
//! The three raw tables are proof inputs, not CUDA-kernel outputs. This module
//! records their exact encodings without inventing an `ExternalInputId`; the
//! proof builder must bind those origins. Big and small limb columns are the
//! explicit outputs of two ordered static-wrapper operations.

use stwo_backend_cuda::{
    ExecutionTablesContract, ExecutionTablesHostIngressEncoding, ExecutionTablesHostIngressField,
    ExecutionTablesHostIngressRole, ExecutionTablesLinkedContract, ExecutionTablesStage,
    ExecutionTablesStageContract,
};

use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueVersion,
};

mod projection;
mod semantic;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExecutionTablesRelocations {
    pub(super) table_pointers: ArenaBinding,
    pub(super) table_strides: ArenaBinding,
}

/// One eager host copy whose `version` remains origin-unbound here.
///
/// Empty logical tables have a one-word arena sentinel but no copied content,
/// hence no semantic value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExecutionTableHostIngress {
    pub(super) role: ExecutionTablesHostIngressRole,
    pub(super) encoding: ExecutionTablesHostIngressEncoding,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: Option<ElementRange>,
    pub(super) copied_words: usize,
    pub(super) arena_words: usize,
    pub(super) version: Option<ValueVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExecutionTableOutputBinding {
    pub(super) column_ordinal: u32,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredExecutionTableStage {
    pub(super) stage: ExecutionTablesStage,
    pub(super) source: Option<ValueVersion>,
    pub(super) source_binding: Option<EffectBindingId>,
    pub(super) outputs: Vec<ExecutionTableOutputBinding>,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredExecutionTables {
    pub(super) contract: ExecutionTablesContract,
    pub(super) relocations: ExecutionTablesRelocations,
    pub(super) host_ingress: [ExecutionTableHostIngress; 3],
    pub(super) stages: [LoweredExecutionTableStage; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedExecutionTableStage {
    pub(super) wrapper: StaticCudaWrapperAuthority,
}

/// Lower the ingest boundary and both ordered split operations transactionally.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredExecutionTables, InvocationShapeError> {
    let tables = arena
        .execution_tables()
        .ok_or(InvocationShapeError::MissingPreparedExecutionTables)?;
    let contract = ExecutionTablesContract::compile(&tables.requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    require_contract_identity(&contract)?;

    let catalog = BaseProducerCatalog::compile(arena)?;
    let exact_ingress = [
        ingress_catalog(
            arena,
            &catalog,
            tables.slots.raw_addr_to_id,
            BufferPurpose::ExecutionTableRawAddressToId,
            contract.host_ingress().fields[0],
        )?,
        ingress_catalog(
            arena,
            &catalog,
            tables.slots.raw_f252_words,
            BufferPurpose::ExecutionTableRawF252Words,
            contract.host_ingress().fields[1],
        )?,
        ingress_catalog(
            arena,
            &catalog,
            tables.slots.raw_small_words,
            BufferPurpose::ExecutionTableRawSmallWords,
            contract.host_ingress().fields[2],
        )?,
    ];
    let relocations = ExecutionTablesRelocations {
        table_pointers: exact_global_arena(
            arena,
            catalog_value(
                &catalog,
                tables.slots.table_pointers,
                BufferPurpose::ExecutionTablePointers,
            )?,
            0,
            tables.requirements.table_pointer_words,
        )?,
        table_strides: exact_global_arena(
            arena,
            catalog_value(
                &catalog,
                tables.slots.table_strides,
                BufferPurpose::ExecutionTableStrides,
            )?,
            0,
            tables.requirements.table_stride_words,
        )?,
    };
    let big = output_catalogs(
        arena,
        &catalog,
        &tables.slots.big_limbs,
        BufferPurpose::ExecutionTableBigLimb,
        tables.requirements.big_column_words,
    )?;
    let small = output_catalogs(
        arena,
        &catalog,
        &tables.slots.small_limbs,
        BufferPurpose::ExecutionTableSmallLimb,
        tables.requirements.small_column_words,
    )?;

    let mut next_values = values.clone();
    next_values.extend_ordered(
        exact_ingress
            .iter()
            .filter(|ingress| ingress.field.copied_words != 0)
            .map(|ingress| ingress.value.id)
            .chain(big.iter().chain(&small).map(|output| output.id)),
    )?;
    let host_ingress = exact_ingress.map(|ingress| bind_ingress(ingress, &next_values));
    let host_ingress = [
        host_ingress[0].clone()?,
        host_ingress[1].clone()?,
        host_ingress[2].clone()?,
    ];
    validate_value_classes(&next_values, &host_ingress, big.iter().chain(&small))?;

    let big_stage = semantic::lower(&contract.stages()[0], &host_ingress[1], big, &next_values)?;
    let small_stage =
        semantic::lower(&contract.stages()[1], &host_ingress[2], small, &next_values)?;
    if [big_stage.stage, small_stage.stage]
        != [ExecutionTablesStage::Big, ExecutionTablesStage::Small]
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }

    *values = next_values;
    Ok(LoweredExecutionTables {
        contract,
        relocations,
        host_ingress,
        stages: [big_stage, small_stage],
    })
}

pub(super) fn validate(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    supplied: &LoweredExecutionTables,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = values.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if exact == *supplied && &exact_values == values {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

pub(super) fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &ExecutionTablesLinkedContract,
    lowered: &LoweredExecutionTables,
    stage: ExecutionTablesStage,
) -> Result<LinkedExecutionTableStage, InvocationShapeError> {
    projection::linked(id, linked, lowered, stage)
}

#[derive(Clone, Copy)]
struct ExactIngress<'a> {
    value: &'a BaseCatalogValue,
    arena: ArenaBinding,
    field: ExecutionTablesHostIngressField,
}

fn ingress_catalog<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    field: ExecutionTablesHostIngressField,
) -> Result<ExactIngress<'a>, InvocationShapeError> {
    let value = catalog_value(catalog, slot, purpose)?;
    let arena = exact_global_arena(arena, value, 0, field.arena_words)?;
    Ok(ExactIngress {
        value,
        arena,
        field,
    })
}

fn bind_ingress(
    exact: ExactIngress<'_>,
    values: &adapter::SemanticValueMap,
) -> Result<ExecutionTableHostIngress, InvocationShapeError> {
    let (elements, version) = if exact.field.copied_words == 0 {
        (None, None)
    } else {
        (
            ElementRange::new(0, exact.field.copied_words),
            Some(values.version(exact.value.id)?),
        )
    };
    Ok(ExecutionTableHostIngress {
        role: exact.field.role,
        encoding: exact.field.encoding,
        arena: exact.arena,
        value: exact.value.id,
        elements,
        copied_words: exact.field.copied_words,
        arena_words: exact.field.arena_words,
        version,
    })
}

fn output_catalogs<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    slots: &[ArenaSlotId],
    purpose: BufferPurpose,
    expected_words: usize,
) -> Result<Vec<&'a BaseCatalogValue>, InvocationShapeError> {
    slots
        .iter()
        .enumerate()
        .map(|(ordinal, &slot)| {
            let value = catalog_value(catalog, slot, purpose)?;
            if value.ordinal as usize != ordinal {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            exact_global_arena(arena, value, ordinal as u32, expected_words)?;
            Ok(value)
        })
        .collect()
}

fn exact_global_arena(
    arena: &ProofArenaPlan,
    value: &BaseCatalogValue,
    ordinal: u32,
    words: usize,
) -> Result<ArenaBinding, InvocationShapeError> {
    if value.component.is_some()
        || value.part.is_some()
        || value.ordinal != ordinal
        || value.words != words
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let binding = ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    };
    if arena.binding(value.logical) == Some(binding) {
        Ok(binding)
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn validate_value_classes<'a>(
    values: &adapter::SemanticValueMap,
    ingress: &[ExecutionTableHostIngress; 3],
    outputs: impl Iterator<Item = &'a &'a BaseCatalogValue>,
) -> Result<(), InvocationShapeError> {
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    let versions = ingress.iter().filter_map(|ingress| ingress.version).chain(
        outputs
            .map(|output| values.version(output.id))
            .collect::<Result<Vec<_>, _>>()?,
    );
    if versions.into_iter().all(|version| {
        catalog_first.contains(&version)
            && !transitions.contains(&version)
            && !fixed.contains(&version)
    }) {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn require_contract_identity(
    contract: &ExecutionTablesContract,
) -> Result<(), InvocationShapeError> {
    if [
        contract.static_source_identity(),
        contract.wrapper_source_identity(),
        contract.source_identity(),
        contract.requirements_identity(),
        contract.host_ingress_identity(),
        contract.fixed_identity(),
        contract.abi_identity(),
        contract.effect_identity(),
        contract.launch_identity(),
        contract.identity(),
    ]
    .contains(&[0; 32])
    {
        Err(InvocationShapeError::InvalidStructuredAbi)
    } else {
        Ok(())
    }
}

fn stage_contract(
    contract: &ExecutionTablesContract,
    stage: ExecutionTablesStage,
) -> Result<&ExecutionTablesStageContract, InvocationShapeError> {
    contract
        .stages()
        .iter()
        .find(|candidate| candidate.stage() == stage)
        .ok_or(InvocationShapeError::InvalidStructuredAbi)
}
