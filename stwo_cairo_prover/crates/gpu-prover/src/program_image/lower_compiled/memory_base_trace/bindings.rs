//! Exact ProofArenaPlan-to-memory-authority binding.

use stwo_backend_cuda::{MemoryBaseTraceRequirements, MemoryBaseTraceValuePartRequirements};

use super::*;
use crate::arena_plan::{PlannedMemoryBaseTraceWorkspace, PlannedMemoryTracePartWorkspace};

pub(super) fn requirements(
    arena: &ProofArenaPlan,
    memory: &PlannedMemoryBaseTraceWorkspace,
) -> Result<MemoryBaseTraceRequirements, InvocationShapeError> {
    let execution = arena
        .execution_tables()
        .ok_or(InvocationShapeError::MissingPreparedExecutionTables)?;
    let big_parts = memory
        .big_parts
        .iter()
        .enumerate()
        .map(|(ordinal, part)| {
            let planned = memory
                .plan
                .big_parts
                .get(ordinal)
                .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
            if part.part
                != TracePartId::MemoryBig(
                    u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                )
                || planned.part != part.part
                || planned.source_offset != part.source_offset
                || planned.row_count != part.row_count
            {
                return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
            }
            Ok(MemoryBaseTraceValuePartRequirements {
                part_ordinal: u32::try_from(ordinal)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?,
                source_offset: part.source_offset,
                row_count: part.row_count,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if memory.big_parts.len() != memory.plan.big_parts.len()
        || memory.small_part.part != TracePartId::MemorySmall
        || memory.plan.small_part.part != memory.small_part.part
        || memory.plan.small_part.source_offset != memory.small_part.source_offset
        || memory.plan.small_part.row_count != memory.small_part.row_count
        || memory.plan.small_count_words != memory.small_part.row_count
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    Ok(MemoryBaseTraceRequirements {
        n_addrs: execution.requirements.n_addrs,
        raw_address_words: execution.requirements.raw_addr_to_id_words,
        address_rows: memory.plan.address_rows,
        address_count_words: memory.plan.address_count_words,
        big_source_words: execution.requirements.big_column_words,
        big_count_words: memory.plan.big_count_words,
        big_parts,
        small_source_words: execution.requirements.small_column_words,
        small_count_words: memory.plan.small_count_words,
        small_part: MemoryBaseTraceValuePartRequirements {
            part_ordinal: 0,
            source_offset: memory.small_part.source_offset,
            row_count: memory.small_part.row_count,
        },
        rc99_lut_words: memory.plan.rc99_lut_words,
        rc99_count_words: memory.plan.rc99_count_words,
    })
}

pub(super) fn exact_inventory(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    memory: &PlannedMemoryBaseTraceWorkspace,
) -> Result<ExactInventory, InvocationShapeError> {
    let execution = arena
        .execution_tables()
        .ok_or(InvocationShapeError::MissingPreparedExecutionTables)?;
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    let runtime = |name| {
        multiplicity
            .multiplicities
            .iter()
            .find_map(|&(candidate, binding)| (candidate == name).then_some(binding))
            .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    };
    let raw_address = exact_physical(
        arena,
        catalog,
        execution.slots.raw_addr_to_id,
        BufferPurpose::ExecutionTableRawAddressToId,
        None,
        None,
        0,
        execution.requirements.raw_addr_to_id_words,
    )?;
    let address_counts = exact_bound(
        arena,
        catalog,
        runtime("memory_address_to_id")?,
        BufferPurpose::RuntimeMultiplicity,
        Some("memory_address_to_id"),
        Some(TracePartId::Main),
        memory.plan.address_count_words,
    )?;
    let big_counts = exact_bound(
        arena,
        catalog,
        runtime("memory_id_to_big")?,
        BufferPurpose::RuntimeMultiplicity,
        Some("memory_id_to_big"),
        None,
        memory.plan.big_count_words,
    )?;
    let small_counts = exact_bound(
        arena,
        catalog,
        runtime("memory_id_to_big#small")?,
        BufferPurpose::RuntimeMultiplicity,
        Some("memory_id_to_big"),
        None,
        memory.plan.small_count_words,
    )?;
    let big_sources = execution
        .slots
        .big_limbs
        .iter()
        .enumerate()
        .map(|(ordinal, &slot)| {
            exact_physical(
                arena,
                catalog,
                slot,
                BufferPurpose::ExecutionTableBigLimb,
                None,
                None,
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                execution.requirements.big_column_words,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let small_sources = execution
        .slots
        .small_limbs
        .iter()
        .enumerate()
        .map(|(ordinal, &slot)| {
            exact_physical(
                arena,
                catalog,
                slot,
                BufferPurpose::ExecutionTableSmallLimb,
                None,
                None,
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                execution.requirements.small_column_words,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let big_parts = memory
        .big_parts
        .iter()
        .map(|part| {
            exact_part(
                arena,
                catalog,
                part,
                &big_sources,
                &big_counts,
                "memory_id_to_big",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let small_part = exact_part(
        arena,
        catalog,
        &memory.small_part,
        &small_sources,
        &small_counts,
        "memory_id_to_big",
    )?;
    Ok(ExactInventory {
        raw_address,
        address_counts,
        address_outputs: exact_outputs(
            arena,
            catalog,
            &memory.address_outputs,
            "memory_address_to_id",
            TracePartId::Main,
            memory.plan.address_rows,
        )?,
        big_parts,
        small_part,
        rc99_lut: exact_bound(
            arena,
            catalog,
            memory.rc99_lut,
            BufferPurpose::WitnessFeedLut,
            None,
            None,
            memory.plan.rc99_lut_words,
        )?,
        rc99_counts: exact_bound(
            arena,
            catalog,
            memory.rc99_counts,
            BufferPurpose::FixedMultiplicity,
            Some("range_check_9_9"),
            Some(TracePartId::Main),
            memory.plan.rc99_count_words,
        )?,
    })
}

fn exact_part(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    part: &PlannedMemoryTracePartWorkspace,
    sources: &[ExactValue],
    counts: &ExactValue,
    component: &'static str,
) -> Result<ExactPart, InvocationShapeError> {
    Ok(ExactPart {
        part: part.part,
        sources: sources.to_vec(),
        counts: counts.clone(),
        outputs: exact_outputs(
            arena,
            catalog,
            &part.outputs,
            component,
            part.part,
            part.row_count,
        )?,
    })
}

fn exact_outputs(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    bindings: &[ArenaBinding],
    component: &'static str,
    part: TracePartId,
    words: usize,
) -> Result<Vec<ExactValue>, InvocationShapeError> {
    bindings
        .iter()
        .enumerate()
        .map(|(ordinal, &binding)| {
            let exact = exact_bound(
                arena,
                catalog,
                binding,
                BufferPurpose::BaseTrace,
                Some(component),
                Some(part),
                words,
            )?;
            if exact.catalog.ordinal as usize != ordinal {
                return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
            }
            Ok(exact)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn exact_physical(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    physical: stwo_backend_cuda::ArenaSlotId,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    ordinal: u32,
    words: usize,
) -> Result<ExactValue, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.physical == physical
            && value.purpose == purpose
            && value.component == component
            && value.part == part
            && value.ordinal == ordinal
            && value.words == words
    });
    let value = matches
        .next()
        .cloned()
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    exact_bound(
        arena,
        catalog,
        ArenaBinding {
            logical: value.logical,
            physical: value.physical,
            len_words: value.words,
        },
        purpose,
        component,
        part,
        words,
    )
}

fn exact_bound(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    binding: ArenaBinding,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    words: usize,
) -> Result<ExactValue, InvocationShapeError> {
    let value = catalog
        .value(ArenaCatalogValueId(binding.logical.0))?
        .clone();
    if value.logical != binding.logical
        || value.physical != binding.physical
        || value.words != binding.len_words
        || value.words != words
        || value.purpose != purpose
        || value.component != component
        || value.part != part
        || arena.binding(value.logical) != Some(binding)
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    Ok(ExactValue {
        catalog: value,
        arena: binding,
    })
}
