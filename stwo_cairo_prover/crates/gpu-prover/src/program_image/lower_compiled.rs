//! Exact schedule-order authority for Base witness producers.
//!
//! Ordinary recorded witnesses lower through the source-emitter-owned typed
//! AOT ABI. Stateful deduces also carry an address-free resource/relocation
//! recipe. ReplacementV1 stores that pure authority in `ShapeExecutable`; a
//! loaded-module publication receipt remains a separate runtime requirement.
//! The LegacyResident mapper remains only a diagnostic migration frontier, and
//! neither path fabricates the later operations needed by a full `CompiledProof`.

use std::collections::BTreeSet;
use std::ops::Range;

use stwo_backend_cuda::aot::{self, AotKernelAbiAccess, AotKernelAbiKind, AotKernelAbiSchema};
use stwo_backend_cuda::jit_witness::isa::{WitnessOp, WitnessProgram};
use stwo_backend_cuda::{
    ArenaSlotId, EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_POINTERS, EXECUTION_TABLE_STRIDES,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::*;
use crate::arena_plan::{
    BlakeGWitnessContract, LogicalBufferId, PlannedWitnessComponent, ProofArenaPlan,
};
use crate::compiled_proof::LaunchGeometry;
use crate::resident_runtime::producer_schedule::WitnessProducer;
#[cfg(test)]
use crate::resident_runtime::producer_schedule::{BaseProducerSchedule, WitnessProducerKind};

mod adapter;
mod base_commit_projection;
mod blake_g_direct_execution_authority;
mod blake_g_direct_prefix;
#[cfg(test)]
mod blake_g_direct_tests;
mod compiled_base_prefix;
#[cfg(test)]
mod compiled_base_prefix_tests;
mod ec_op_execution_authority;
#[cfg(test)]
mod ec_op_pair_tests;
mod ec_op_prefix;
mod ec_op_setup_sources;
mod execution_tables;
mod fixed_table_materialization;
mod interaction_commit_projection;
mod loaded_authority;
mod loaded_base_binding;
mod loaded_writer_binding;
mod memory_base_trace;
mod multiplicity_clear;
mod multiplicity_coordinator;
mod multiplicity_feed;
#[cfg(test)]
mod post_base_authority_tests;
mod producer_prefix;
mod recorded_deduce_authority;
#[cfg(test)]
mod recorded_deduce_tests;
mod relation_projection;
mod resolved_recorded_build_authority;
mod schedule_prefix;
mod static_wrapper_invocation;
mod static_wrapper_projection;
#[cfg(test)]
mod static_wrapper_projection_tests;
mod witness_casm_input;
mod witness_input_gather;
#[cfg(test)]
mod witness_input_gather_tests;
mod witness_input_seed_compact;

pub(crate) use producer_prefix::{
    BaseProducerAuthority, LoadedBaseProducerAuthority, PreparedBaseProducerInventory,
    PreparedBlakeGDirectKernel, PreparedEcOpSegment, PreparedGenericMultiplicityFeed,
    PreparedPublicMemorySeed, PreparedRecordedKernel,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BaseProducerAuthorityError;

pub(crate) fn compile_replacement_base_authority(
    arena: &ProofArenaPlan,
    preprocessed_trace_variant: PreProcessedTraceVariant,
) -> Result<BaseProducerAuthority, BaseProducerAuthorityError> {
    producer_prefix::BaseProducerAuthority::compile_replacement(arena, preprocessed_trace_variant)
        .map_err(|_| BaseProducerAuthorityError)
}

pub(crate) fn bind_replacement_base_authority(
    authority: &BaseProducerAuthority,
    arena: &ProofArenaPlan,
    prepared: PreparedBaseProducerInventory<'_, '_>,
    device_ordinal: u32,
    sm_major: u32,
    sm_minor: u32,
) -> Result<LoadedBaseProducerAuthority, BaseProducerAuthorityError> {
    authority
        .bind_loaded(arena, prepared, device_ordinal, sm_major, sm_minor)
        .map_err(|_| BaseProducerAuthorityError)
}

const POINTER_WORDS: usize = core::mem::size_of::<*const u32>().div_ceil(WORD_BYTES);
const WORD_BYTES: usize = core::mem::size_of::<u32>();

/// Address-free Base-only relocation catalog. Unlike [`ArenaProgramInventory`],
/// this is production input: it contains only exact logical roles, extents and
/// arena slot identities needed to bind Base producers. It makes no claim about
/// the later transcript/proof DAG or semantic SSA origins.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BaseProducerCatalog {
    values: Vec<BaseCatalogValue>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BaseCatalogValue {
    id: ArenaCatalogValueId,
    logical: LogicalBufferId,
    physical: ArenaSlotId,
    component: Option<&'static str>,
    part: Option<stwo_cairo_prover::witness::proof_shape::TracePartId>,
    purpose: BufferPurpose,
    ordinal: u32,
    words: usize,
}

impl BaseProducerCatalog {
    fn compile(arena: &ProofArenaPlan) -> Result<Self, InvocationShapeError> {
        if arena.logical_buffers().is_empty()
            || arena.logical_buffers().len() != arena.bindings().len()
        {
            return Err(InvocationShapeError::InvalidBaseCatalog);
        }
        let values = arena
            .logical_buffers()
            .iter()
            .enumerate()
            .map(|(index, logical)| {
                let dense = LogicalBufferId(
                    u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                );
                let binding = arena
                    .binding(logical.id)
                    .filter(|binding| binding.logical == logical.id)
                    .ok_or(InvocationShapeError::InvalidBaseCatalog)?;
                let slot = arena
                    .layout()
                    .slot(binding.physical)
                    .ok_or(InvocationShapeError::MissingCatalogValue(binding.physical))?;
                if logical.id != dense
                    || binding.len_words != logical.len_words
                    || binding.len_words > slot.len_words
                    || logical.len_words == 0
                {
                    return Err(InvocationShapeError::InvalidBaseCatalog);
                }
                Ok(BaseCatalogValue {
                    id: ArenaCatalogValueId(logical.id.0),
                    logical: logical.id,
                    physical: binding.physical,
                    component: logical.component,
                    part: logical.part,
                    purpose: logical.purpose,
                    ordinal: logical.ordinal,
                    words: logical.len_words,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { values })
    }

    fn value(&self, id: ArenaCatalogValueId) -> Result<&BaseCatalogValue, InvocationShapeError> {
        self.values
            .get(id.0 as usize)
            .filter(|value| value.id == id)
            .ok_or(InvocationShapeError::InvalidCatalogRange(id))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvocationAccess {
    Inactive,
    Read,
    Write,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InvocationTarget {
    value: ArenaCatalogValueId,
    elements: Range<usize>,
    access: InvocationAccess,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PointerEntryBinding {
    entry: u32,
    descriptor_words: Range<usize>,
    target: InvocationTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScalarEntryBinding {
    index: u32,
    value: u32,
    access: InvocationAccess,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SourceArgument {
    PointerTable {
        ordinal: u8,
        descriptor: ArenaCatalogRange,
        descriptor_access: InvocationAccess,
        entries: Vec<PointerEntryBinding>,
    },
    ScalarArray {
        ordinal: u8,
        value: ArenaCatalogRange,
        entries: Vec<ScalarEntryBinding>,
    },
    DirectPointer {
        ordinal: u8,
        target: InvocationTarget,
    },
    U32 {
        ordinal: u8,
        value: u32,
    },
}

/// Source-emitter-owned ABI shape; binary/module authority remains separate.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordedWitnessInvocationShape {
    program_identity: [u8; 32],
    semantic_hash: u64,
    cache_key: u64,
    kernel_symbol: String,
    abi_schema_identity: [u8; 32],
    deduce: recorded_deduce_authority::RecordedDeduceAuthority,
    launch: LaunchGeometry,
    source_arguments: Vec<SourceArgument>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvocationShapeError {
    InvalidBaseCatalog,
    MissingRecordedWitness,
    MissingPreparedExecutionTables,
    LegacyExecutionTables,
    MultiplicityNeedsSemanticVersions,
    InvalidProgramRole,
    InvalidStructuredAbi,
    MissingCatalogValue(ArenaSlotId),
    AmbiguousCatalogValue(ArenaSlotId),
    InvalidCatalogRange(ArenaCatalogValueId),
    MissingSemanticValueMap(ArenaCatalogValueId),
    InvalidAdapterEffect,
    SizeOverflow,
    SourceEmitterRejected,
    MissingLoadedAotAuthority,
    MissingLoadedModuleStateAuthority,
    LoadedModuleStateAuthorityMismatch,
    LoadedAotAuthorityMismatch,
    InvocationMismatch,
    FrontierDidNotAdvance,
    ScheduledProducerInvalidProgram(WitnessProducer),
    ScheduledProducerInvalidEffect(WitnessProducer),
    InvalidScheduledProducerBinding,
    InvalidNativeEcOpAuthority,
    InvalidNativeEcOpBinding,
    InvalidNativeBlakeGDirectAuthority,
    InvalidNativeBlakeGDirectBinding,
    MissingNativeBlakeGDirectAuthority,
    InvalidBaseInterpolationAuthority,
    InvalidBaseInterpolationBinding,
    InvalidProductionBaseAuthority,
    InvalidMemoryBaseTraceAuthority,
    InvalidMemoryBaseTraceBinding,
    InvalidFixedTableAuthority,
    InvalidFixedTableBinding,
    InvalidBaseCommitAuthority,
    InvalidBaseCommitBinding,
    InvalidInteractionCommitAuthority,
    InvalidInteractionCommitBinding,
    InvalidRelationAuthority,
    InvalidRelationBinding,
}

fn validate_invocation(
    supplied: &RecordedWitnessInvocationShape,
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    producer: WitnessProducer,
) -> Result<(), InvocationShapeError> {
    let planned = planned_recorded_component(arena, producer)?;
    let expected = derive_invocation(catalog, arena, planned)?;
    if supplied == &expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvocationMismatch)
    }
}

fn planned_recorded_component<'a>(
    arena: &'a ProofArenaPlan,
    producer: WitnessProducer,
) -> Result<&'a PlannedWitnessComponent, InvocationShapeError> {
    arena
        .witness()
        .components
        .iter()
        .find(|planned| {
            producer.component == planned.component && producer.part == Some(planned.part)
        })
        .filter(|planned| planned.blake_g_contract == BlakeGWitnessContract::Recorded)
        .ok_or(InvocationShapeError::MissingRecordedWitness)
}

fn derive_invocation(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    planned: &PlannedWitnessComponent,
) -> Result<RecordedWitnessInvocationShape, InvocationShapeError> {
    validate_multiplicity_free(planned)?;
    let program_use = validate_program_roles(&planned.program)?;
    validate_recorded_witness_abi()?;

    let emitted = aot::witness_kernel_source(&planned.program)
        .ok_or(InvocationShapeError::SourceEmitterRejected)?;
    let schema = emitted
        .abi_schema
        .filter(|schema| *schema == AotKernelAbiSchema::RecordedWitnessV1)
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    let program_identity = emitted
        .program_identity
        .filter(|identity| identity == &planned.program.semantic_identity())
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    let deduce = recorded_deduce_authority::bind(&planned.program, &emitted.source)?;

    let tables = arena
        .execution_tables()
        .ok_or(InvocationShapeError::MissingPreparedExecutionTables)?;
    if planned.slots.execution_table_pointers != tables.slots.table_pointers
        || planned.slots.execution_table_strides != tables.slots.table_strides
    {
        return Err(InvocationShapeError::LegacyExecutionTables);
    }
    let input_columns = usize::try_from(planned.program.n_inputs)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    if planned.slots.input_columns.len() != input_columns
        || planned.requirements.input_column_words.len() != input_columns
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let table_use = table_use(&planned.program)?;
    let mut source_arguments = Vec::with_capacity(8);
    source_arguments.push(pointer_table(
        0,
        catalog_value(
            catalog,
            planned.slots.input_pointers,
            BufferPurpose::WitnessInputPointers,
        )?,
        planned
            .slots
            .input_columns
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, slot)| {
                target(
                    catalog,
                    slot,
                    BufferPurpose::WitnessInput,
                    0..planned.requirements.row_count,
                    if program_use.inputs.contains(
                        &u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    ) {
                        InvocationAccess::Read
                    } else {
                        InvocationAccess::Inactive
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    )?);
    source_arguments.push(table_pointer_argument(catalog, tables, &table_use)?);
    source_arguments.push(table_stride_argument(catalog, tables, &table_use)?);
    source_arguments.push(pointer_table(
        3,
        catalog_value(
            catalog,
            planned.slots.output_pointers,
            BufferPurpose::WitnessOutputPointers,
        )?,
        planned
            .slots
            .output_columns
            .iter()
            .copied()
            .map(|slot| {
                target(
                    catalog,
                    slot,
                    BufferPurpose::BaseTrace,
                    0..planned.requirements.row_count,
                    InvocationAccess::Write,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    )?);
    let multiplicity_dummy = planned
        .slots
        .multiplicity_dummy
        .ok_or(InvocationShapeError::MultiplicityNeedsSemanticVersions)?;
    source_arguments.push(pointer_table(
        4,
        catalog_value(
            catalog,
            planned.slots.multiplicity_pointers,
            BufferPurpose::WitnessMultiplicityPointers,
        )?,
        vec![target(
            catalog,
            multiplicity_dummy,
            BufferPurpose::WitnessMultiplicityDummy,
            0..1,
            InvocationAccess::Inactive,
        )?],
    )?);
    source_arguments.push(direct_output(
        5,
        catalog,
        planned.slots.lookup_words,
        if planned.program.n_lookup_words == 0 {
            BufferPurpose::WitnessLookupDummy
        } else {
            BufferPurpose::LookupInputs
        },
        words_per_row(
            planned.program.n_lookup_words,
            planned.requirements.row_count,
        )?,
    )?);
    source_arguments.push(direct_output(
        6,
        catalog,
        planned.slots.sub_words,
        if planned.program.n_sub_words == 0 {
            BufferPurpose::WitnessSubDummy
        } else {
            BufferPurpose::SubcomponentInputs
        },
        words_per_row(planned.program.n_sub_words, planned.requirements.row_count)?,
    )?);
    let row_count = u32::try_from(planned.requirements.row_count)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    source_arguments.push(SourceArgument::U32 {
        ordinal: 7,
        value: row_count,
    });
    Ok(RecordedWitnessInvocationShape {
        program_identity,
        semantic_hash: emitted.semantic_hash,
        cache_key: emitted.cache_key,
        kernel_symbol: emitted.kernel_name,
        abi_schema_identity: schema.identity(),
        deduce,
        launch: LaunchGeometry {
            grid: [row_count.div_ceil(256), 1, 1],
            block: [256, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        source_arguments,
    })
}

fn validate_multiplicity_free(
    planned: &PlannedWitnessComponent,
) -> Result<(), InvocationShapeError> {
    if planned.program.n_mult_tables != 0
        || !planned.requirements.multiplicity_column_words.is_empty()
        || !planned.slots.multiplicity_columns.is_empty()
        || planned
            .program
            .insts
            .iter()
            .any(|instruction| WitnessOp::from_raw(instruction.op) == Some(WitnessOp::MultPush))
    {
        Err(InvocationShapeError::MultiplicityNeedsSemanticVersions)
    } else {
        Ok(())
    }
}

#[derive(Default)]
struct TableUse {
    address: bool,
    limbs: BTreeSet<u32>,
}

fn table_use(program: &WitnessProgram) -> Result<TableUse, InvocationShapeError> {
    let mut use_set = TableUse::default();
    for instruction in &program.insts {
        if WitnessOp::from_raw(instruction.op) != Some(WitnessOp::TableLimb) {
            continue;
        }
        match instruction.b {
            0 if instruction.imm == 0 => use_set.address = true,
            1 if instruction.imm < EXECUTION_TABLE_BIG_LIMBS as u32 => {
                use_set.limbs.insert(instruction.imm);
            }
            _ => return Err(InvocationShapeError::InvalidProgramRole),
        }
    }
    Ok(use_set)
}

fn table_pointer_argument(
    catalog: &BaseProducerCatalog,
    tables: &crate::arena_plan::PlannedExecutionTablesWorkspace,
    used: &TableUse,
) -> Result<SourceArgument, InvocationShapeError> {
    let descriptor = catalog_value(
        catalog,
        tables.slots.table_pointers,
        BufferPurpose::ExecutionTablePointers,
    )?;
    let mut targets = Vec::with_capacity(EXECUTION_TABLE_POINTERS);
    targets.push(target(
        catalog,
        tables.slots.raw_addr_to_id,
        BufferPurpose::ExecutionTableRawAddressToId,
        0..tables.requirements.n_addrs,
        if used.address {
            InvocationAccess::Read
        } else {
            InvocationAccess::Inactive
        },
    )?);
    for (limb, &slot) in tables.slots.big_limbs.iter().enumerate() {
        targets.push(target(
            catalog,
            slot,
            BufferPurpose::ExecutionTableBigLimb,
            0..tables.requirements.n_big,
            if used.limbs.contains(&(limb as u32)) {
                InvocationAccess::Read
            } else {
                InvocationAccess::Inactive
            },
        )?);
    }
    for (limb, &slot) in tables.slots.small_limbs.iter().enumerate() {
        targets.push(target(
            catalog,
            slot,
            BufferPurpose::ExecutionTableSmallLimb,
            0..tables.requirements.n_small,
            if used.limbs.contains(&(limb as u32)) {
                InvocationAccess::Read
            } else {
                InvocationAccess::Inactive
            },
        )?);
    }
    pointer_table(1, descriptor, targets)
}

fn table_stride_argument(
    catalog: &BaseProducerCatalog,
    tables: &crate::arena_plan::PlannedExecutionTablesWorkspace,
    used: &TableUse,
) -> Result<SourceArgument, InvocationShapeError> {
    let value = whole_catalog_range(catalog_value(
        catalog,
        tables.slots.table_strides,
        BufferPurpose::ExecutionTableStrides,
    )?)?;
    let values = [
        tables.requirements.n_addrs,
        tables.requirements.n_big,
        tables.requirements.n_small,
    ];
    let entries = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Ok(ScalarEntryBinding {
                index: index as u32,
                value: u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)?,
                access: if (index == 0 && used.address) || (index > 0 && !used.limbs.is_empty()) {
                    InvocationAccess::Read
                } else {
                    InvocationAccess::Inactive
                },
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    if entries.len() != EXECUTION_TABLE_STRIDES {
        return Err(InvocationShapeError::InvalidProgramRole);
    }
    Ok(SourceArgument::ScalarArray {
        ordinal: 2,
        value,
        entries,
    })
}

fn pointer_table(
    ordinal: u8,
    descriptor: &BaseCatalogValue,
    targets: Vec<InvocationTarget>,
) -> Result<SourceArgument, InvocationShapeError> {
    let descriptor = whole_catalog_range(descriptor)?;
    let required_words = targets
        .len()
        .checked_mul(POINTER_WORDS)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if descriptor.value_words != (0..required_words) {
        return Err(InvocationShapeError::InvalidCatalogRange(descriptor.value));
    }
    let entries = targets
        .into_iter()
        .enumerate()
        .map(|(entry, target)| {
            let start = entry
                .checked_mul(POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let end = start
                .checked_add(POINTER_WORDS)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            if end > descriptor.value_words.end {
                return Err(InvocationShapeError::InvalidCatalogRange(descriptor.value));
            }
            Ok(PointerEntryBinding {
                entry: u32::try_from(entry).map_err(|_| InvocationShapeError::SizeOverflow)?,
                descriptor_words: start..end,
                target,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let descriptor_access = if entries
        .iter()
        .any(|entry| entry.target.access != InvocationAccess::Inactive)
    {
        InvocationAccess::Read
    } else {
        InvocationAccess::Inactive
    };
    Ok(SourceArgument::PointerTable {
        ordinal,
        descriptor,
        descriptor_access,
        entries,
    })
}

fn direct_output(
    ordinal: u8,
    catalog: &BaseProducerCatalog,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    words: usize,
) -> Result<SourceArgument, InvocationShapeError> {
    Ok(SourceArgument::DirectPointer {
        ordinal,
        target: target(
            catalog,
            slot,
            purpose,
            0..words.max(1),
            if words == 0 {
                InvocationAccess::Inactive
            } else {
                InvocationAccess::Write
            },
        )?,
    })
}

fn target(
    catalog: &BaseProducerCatalog,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    elements: Range<usize>,
    access: InvocationAccess,
) -> Result<InvocationTarget, InvocationShapeError> {
    let value = catalog_value(catalog, slot, purpose)?;
    if elements.end > value.words || (access != InvocationAccess::Inactive && elements.is_empty()) {
        return Err(InvocationShapeError::InvalidCatalogRange(value.id));
    }
    Ok(InvocationTarget {
        value: value.id,
        elements,
        access,
    })
}

fn catalog_value<'a>(
    catalog: &'a BaseProducerCatalog,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
) -> Result<&'a BaseCatalogValue, InvocationShapeError> {
    let mut matches = catalog
        .values
        .iter()
        .filter(|value| value.purpose == purpose && value.physical == slot);
    let value = matches
        .next()
        .ok_or(InvocationShapeError::MissingCatalogValue(slot))?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::AmbiguousCatalogValue(slot));
    }
    Ok(value)
}

fn whole_catalog_range(
    value: &BaseCatalogValue,
) -> Result<ArenaCatalogRange, InvocationShapeError> {
    Ok(ArenaCatalogRange {
        value: value.id,
        value_words: 0..value.words,
    })
}

fn words_per_row(count: u32, rows: usize) -> Result<usize, InvocationShapeError> {
    usize::try_from(count)
        .map_err(|_| InvocationShapeError::SizeOverflow)?
        .checked_mul(rows)
        .ok_or(InvocationShapeError::SizeOverflow)
}

fn produced_values(
    catalog: &BaseProducerCatalog,
    planned: &PlannedWitnessComponent,
) -> Result<Vec<ArenaCatalogValueId>, InvocationShapeError> {
    let mut produced = planned
        .slots
        .output_columns
        .iter()
        .copied()
        .map(|slot| catalog_value(catalog, slot, BufferPurpose::BaseTrace).map(|value| value.id))
        .collect::<Result<Vec<_>, _>>()?;
    if planned.program.n_lookup_words != 0 {
        produced.push(
            catalog_value(
                catalog,
                planned.slots.lookup_words,
                BufferPurpose::LookupInputs,
            )?
            .id,
        );
    }
    if planned.program.n_sub_words != 0 {
        produced.push(
            catalog_value(
                catalog,
                planned.slots.sub_words,
                BufferPurpose::SubcomponentInputs,
            )?
            .id,
        );
    }
    produced.sort_unstable();
    produced.dedup();
    Ok(produced)
}

struct ProgramUse {
    inputs: BTreeSet<u32>,
}

fn validate_program_roles(program: &WitnessProgram) -> Result<ProgramUse, InvocationShapeError> {
    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut lookups = BTreeSet::new();
    let mut subs = BTreeSet::new();
    for instruction in &program.insts {
        match WitnessOp::from_raw(instruction.op).ok_or(InvocationShapeError::InvalidProgramRole)? {
            WitnessOp::Input => {
                inputs.insert(instruction.a);
            }
            WitnessOp::ColWrite => {
                outputs.insert(instruction.imm);
            }
            WitnessOp::LookupWord => {
                lookups.insert(instruction.imm);
            }
            WitnessOp::SubWord => {
                subs.insert(instruction.imm);
            }
            WitnessOp::MultPush => {
                return Err(InvocationShapeError::MultiplicityNeedsSemanticVersions)
            }
            _ => {}
        }
    }
    if inputs.iter().any(|&ordinal| ordinal >= program.n_inputs)
        || !dense(&outputs, program.n_cols)
        || !dense(&lookups, program.n_lookup_words)
        || !dense(&subs, program.n_sub_words)
    {
        return Err(InvocationShapeError::InvalidProgramRole);
    }
    Ok(ProgramUse { inputs })
}

fn dense(values: &BTreeSet<u32>, count: u32) -> bool {
    values.len() == count as usize && values.iter().copied().eq(0..count)
}

fn validate_recorded_witness_abi() -> Result<(), InvocationShapeError> {
    use AotKernelAbiAccess::{LaunchRowCount, Read, ReadWrite, Write};
    use AotKernelAbiKind::{DevicePointerTableU32, DevicePointerU32, U32};
    let expected = [
        (0, "input_cols", DevicePointerTableU32, Read),
        (1, "table_bases", DevicePointerTableU32, Read),
        (2, "table_strides", DevicePointerU32, Read),
        (3, "out_cols", DevicePointerTableU32, Write),
        (4, "mult_counts", DevicePointerTableU32, ReadWrite),
        (5, "lookup_words", DevicePointerU32, Write),
        (6, "sub_words", DevicePointerU32, Write),
        (7, "row_count", U32, LaunchRowCount),
    ];
    let actual = AotKernelAbiSchema::RecordedWitnessV1.arguments();
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(actual, expected)| {
            (actual.ordinal, actual.name, actual.kind, actual.access) != expected
        })
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
