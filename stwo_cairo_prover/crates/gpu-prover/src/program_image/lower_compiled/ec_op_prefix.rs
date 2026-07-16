//! Exact static contract at the native `ec_op_builtin` schedule boundary.
//!
//! The prepared wrapper is one composite submission of three ordered kernels,
//! not a recorded-witness AOT invocation. This module binds every semantic
//! range and atomic transition while leaving executable module authority as
//! an explicit later gate.

use stwo_backend_cuda::{
    EcOpCompositeAbi, EcOpCompositeContract, EcOpEffectAbi, EcOpExecutionTableShape,
    EcOpKernelStage, EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_POINTERS,
    EXECUTION_TABLE_SMALL_LIMBS,
};

use super::*;
use crate::arena_plan::{BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    AtomicOperation, BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange,
    InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
    ValueVersion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeEcOpRangeBinding {
    pub(super) value: ArenaCatalogRange,
    pub(super) binding: Option<EffectBindingId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeEcOpAtomicBinding {
    pub(super) value: ArenaCatalogRange,
    pub(super) binding: EffectBindingId,
    pub(super) source: ValueVersion,
    pub(super) destination: ValueVersion,
    pub(super) alias: InPlaceAliasAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StaticEcOpInvocation {
    /// Device-resident descriptor passed as raw argument zero. Pointer bytes
    /// are relocation data, not semantic proof values in the effect contract.
    pub(super) execution_table_pointers: ArenaCatalogRange,
    pub(super) execution_tables: Vec<NativeEcOpRangeBinding>,
    pub(super) segment_start: NativeEcOpRangeBinding,
    pub(super) trace_columns: Vec<NativeEcOpRangeBinding>,
    pub(super) lookup_words: NativeEcOpRangeBinding,
    pub(super) partial_input_columns: Vec<NativeEcOpRangeBinding>,
    pub(super) multiplicities: Vec<NativeEcOpAtomicBinding>,
    pub(super) row_count: u32,
    pub(super) partial_row_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredNativeEcOpContract {
    pub(super) authority: EcOpCompositeContract,
    pub(super) invocation: StaticEcOpInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug)]
pub(super) struct PendingNativeEcOpContract {
    authority: EcOpCompositeContract,
    execution_table_pointers: ArenaCatalogRange,
    execution_tables: Vec<ArenaCatalogRange>,
    segment_start: ArenaCatalogRange,
    trace_columns: Vec<ArenaCatalogRange>,
    lookup_words: ArenaCatalogRange,
    partial_input_columns: Vec<ArenaCatalogRange>,
    multiplicities: Vec<ArenaCatalogRange>,
}

impl PendingNativeEcOpContract {
    pub(super) fn catalog_order(&self) -> impl Iterator<Item = ArenaCatalogValueId> + '_ {
        self.execution_tables
            .iter()
            .chain(std::iter::once(&self.segment_start))
            .chain(&self.trace_columns)
            .chain(std::iter::once(&self.lookup_words))
            .chain(&self.partial_input_columns)
            .chain(&self.multiplicities)
            .filter(|range| !range.value_words.is_empty())
            .map(|range| range.value)
    }
}

pub(super) fn prepare(
    image: &ArenaProgramInventory,
    arena: &ProofArenaPlan,
) -> Result<PendingNativeEcOpContract, InvocationShapeError> {
    let ec_op = arena
        .ec_op()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    let tables = arena
        .execution_tables()
        .ok_or(InvocationShapeError::MissingPreparedExecutionTables)?;
    let table_shape = EcOpExecutionTableShape {
        n_addresses: tables.requirements.n_addrs,
        n_big: tables.requirements.n_big,
        n_small: tables.requirements.n_small,
    };
    let authority = EcOpCompositeContract::compile(&ec_op.requirements, table_shape)
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)?;
    if authority.abi() != EcOpCompositeAbi::ProjectiveChainNormalizePaddingV1
        || authority.effect() != EcOpEffectAbi::FullTraceLookupPartialAndAtomicMultiplicitiesV1
        || authority.abi().arguments().len() != 19
        || authority.identity() == [0; 32]
        || authority.source_identity() == [0; 32]
        || (*authority.launches()).map(|launch| launch.stage)
            != [
                EcOpKernelStage::ProjectiveChain,
                EcOpKernelStage::NormalizeRoundTiles,
                EcOpKernelStage::PartialInputPadding,
            ]
    {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }

    let execution_table_pointers = exact_range(
        image,
        arena,
        tables.slots.table_pointers,
        BufferPurpose::ExecutionTablePointers,
        0..tables.requirements.table_pointer_words,
    )?;
    if tables.requirements.table_pointer_words
        != EXECUTION_TABLE_POINTERS
            .checked_mul(POINTER_WORDS)
            .ok_or(InvocationShapeError::SizeOverflow)?
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }

    let mut execution_tables = Vec::with_capacity(EXECUTION_TABLE_POINTERS);
    execution_tables.push(exact_range(
        image,
        arena,
        tables.slots.raw_addr_to_id,
        BufferPurpose::ExecutionTableRawAddressToId,
        0..tables.requirements.n_addrs,
    )?);
    execution_tables.extend(
        tables
            .slots
            .big_limbs
            .iter()
            .copied()
            .map(|slot| {
                exact_range(
                    image,
                    arena,
                    slot,
                    BufferPurpose::ExecutionTableBigLimb,
                    0..tables.requirements.n_big,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    execution_tables.extend(
        tables
            .slots
            .small_limbs
            .iter()
            .copied()
            .map(|slot| {
                exact_range(
                    image,
                    arena,
                    slot,
                    BufferPurpose::ExecutionTableSmallLimb,
                    0..tables.requirements.n_small,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    if tables.slots.big_limbs.len() != EXECUTION_TABLE_BIG_LIMBS
        || tables.slots.small_limbs.len() != EXECUTION_TABLE_SMALL_LIMBS
        || execution_tables.len() != EXECUTION_TABLE_POINTERS
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }

    let segment_start = exact_role(
        image,
        arena,
        ec_op.slots.segment_start,
        BufferPurpose::EcOpSegmentStart,
        Some("ec_op_builtin"),
        Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
        0,
        0..1,
    )?;
    let trace_columns = ec_op
        .slots
        .trace_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, slot)| {
            exact_role(
                image,
                arena,
                slot,
                BufferPurpose::BaseTrace,
                Some("ec_op_builtin"),
                Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                0..ec_op.requirements.row_count,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lookup_words = exact_role(
        image,
        arena,
        ec_op.slots.lookup_words,
        BufferPurpose::LookupInputs,
        Some("ec_op_builtin"),
        Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
        0,
        0..ec_op.requirements.lookup_words,
    )?;
    let (_, consumer_input_slots) = ec_op
        .slots
        .partial_input_columns
        .split_last()
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    let partial_consumer = arena
        .witness()
        .components
        .iter()
        .find(|component| {
            component.component == "partial_ec_mul_generic"
                && component.part == stwo_cairo_prover::witness::proof_shape::TracePartId::Main
        })
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
    if partial_consumer.native_input_producer != Some("ec_op_builtin")
        || partial_consumer.slots.input_columns.as_slice() != consumer_input_slots
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let mut partial_input_columns = Vec::with_capacity(ec_op.slots.partial_input_columns.len());
    for (ordinal, &slot) in ec_op.slots.partial_input_columns.iter().enumerate() {
        let (purpose, component, expected_ordinal) =
            if ordinal + 1 == ec_op.slots.partial_input_columns.len() {
                (BufferPurpose::EcOpPartialIota, "ec_op_builtin", 0)
            } else {
                (
                    BufferPurpose::WitnessInput,
                    "partial_ec_mul_generic",
                    u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                )
            };
        partial_input_columns.push(exact_role(
            image,
            arena,
            slot,
            purpose,
            Some(component),
            Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
            expected_ordinal,
            0..ec_op.requirements.partial_row_count,
        )?);
    }

    let multiplicities = [
        (
            ec_op.slots.address_counts,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_address_to_id"),
            Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
            ec_op.requirements.address_count_words,
        ),
        (
            ec_op.slots.big_counts,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_id_to_big"),
            None,
            ec_op.requirements.big_count_words,
        ),
        (
            ec_op.slots.small_counts,
            BufferPurpose::RuntimeMultiplicity,
            Some("memory_id_to_big"),
            None,
            ec_op.requirements.small_count_words,
        ),
        (
            ec_op.slots.range_check_8_counts,
            BufferPurpose::FixedMultiplicity,
            Some("range_check_8"),
            Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main),
            ec_op.requirements.range_check_8_count_words,
        ),
    ]
    .into_iter()
    .map(|(slot, purpose, component, part, words)| {
        let range = exact_range(image, arena, slot, purpose, 0..words)?;
        let value = &image.values[range.value.0 as usize];
        if value.component != component || value.part != part {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        Ok(range)
    })
    .collect::<Result<Vec<_>, _>>()?;

    if trace_columns.len() != ec_op.requirements.trace_column_words.len()
        || partial_input_columns.len() != ec_op.requirements.partial_input_column_words.len()
        || multiplicities.len() != 4
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(PendingNativeEcOpContract {
        authority,
        execution_table_pointers,
        execution_tables,
        segment_start,
        trace_columns,
        lookup_words,
        partial_input_columns,
        multiplicities,
    })
}

pub(super) fn lower(
    pending: PendingNativeEcOpContract,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredNativeEcOpContract, InvocationShapeError> {
    values.extend_ordered(pending.catalog_order())?;
    let mut next_binding = 0u32;
    let mut accesses = Vec::new();
    let execution_tables = pending
        .execution_tables
        .into_iter()
        .map(|value| bind_range(value, values, &mut next_binding, &mut accesses, false))
        .collect::<Result<Vec<_>, _>>()?;
    let segment_start = bind_range(
        pending.segment_start,
        values,
        &mut next_binding,
        &mut accesses,
        false,
    )?;
    let trace_columns = pending
        .trace_columns
        .into_iter()
        .map(|value| bind_range(value, values, &mut next_binding, &mut accesses, true))
        .collect::<Result<Vec<_>, _>>()?;
    let lookup_words = bind_range(
        pending.lookup_words,
        values,
        &mut next_binding,
        &mut accesses,
        true,
    )?;
    let partial_input_columns = pending
        .partial_input_columns
        .into_iter()
        .map(|value| bind_range(value, values, &mut next_binding, &mut accesses, true))
        .collect::<Result<Vec<_>, _>>()?;
    let multiplicities = pending
        .multiplicities
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let binding = EffectBindingId(next_binding);
            next_binding = next_binding
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let (source, destination) = values.transition(value.value)?;
            let elements = element_range(&value)?;
            let alias = InPlaceAliasAuthority {
                id: InPlaceAliasId(
                    u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            };
            accesses.push(EffectAccess::Atomic {
                source: bound(binding, source, elements),
                destination: bound(binding, destination, elements),
                operation: AtomicOperation::AddU32,
                in_place: alias,
            });
            Ok(NativeEcOpAtomicBinding {
                value,
                binding,
                source,
                destination,
                alias,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let effect = EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpBinding)?;
    Ok(LoweredNativeEcOpContract {
        invocation: StaticEcOpInvocation {
            execution_table_pointers: pending.execution_table_pointers,
            execution_tables,
            segment_start,
            trace_columns,
            lookup_words,
            partial_input_columns,
            multiplicities,
            row_count: u32::try_from(pending.authority.requirements().row_count)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            partial_row_count: u32::try_from(pending.authority.requirements().partial_row_count)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
        },
        authority: pending.authority,
        effect,
    })
}

fn bind_range(
    value: ArenaCatalogRange,
    values: &adapter::SemanticValueMap,
    next_binding: &mut u32,
    accesses: &mut Vec<EffectAccess>,
    write: bool,
) -> Result<NativeEcOpRangeBinding, InvocationShapeError> {
    if value.value_words.is_empty() {
        return Ok(NativeEcOpRangeBinding {
            value,
            binding: None,
        });
    }
    let binding = EffectBindingId(*next_binding);
    *next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let range = bound(
        binding,
        values.version(value.value)?,
        element_range(&value)?,
    );
    accesses.push(if write {
        EffectAccess::Write { destination: range }
    } else {
        EffectAccess::Read { source: range }
    });
    Ok(NativeEcOpRangeBinding {
        value,
        binding: Some(binding),
    })
}

const fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}

fn element_range(value: &ArenaCatalogRange) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(value.value_words.start, value.value_words.end)
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)
}

fn exact_role(
    image: &ArenaProgramInventory,
    arena: &ProofArenaPlan,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<stwo_cairo_prover::witness::proof_shape::TracePartId>,
    ordinal: u32,
    elements: Range<usize>,
) -> Result<ArenaCatalogRange, InvocationShapeError> {
    let range = exact_range(image, arena, slot, purpose, elements)?;
    let value = &image.values[range.value.0 as usize];
    if value.component != component || value.part != part || value.ordinal != ordinal {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(range)
}

fn exact_range(
    image: &ArenaProgramInventory,
    arena: &ProofArenaPlan,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    elements: Range<usize>,
) -> Result<ArenaCatalogRange, InvocationShapeError> {
    let value = catalog_value(image, arena, slot, purpose)?;
    if elements.end
        > value
            .layout
            .element_count()
            .map_err(|_| InvocationShapeError::SizeOverflow)?
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(ArenaCatalogRange {
        value: value.id,
        value_words: elements,
    })
}
