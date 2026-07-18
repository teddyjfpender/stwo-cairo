//! Exact post-memory lowering for fixed Base-trace and LookupInputs tables.
//!
//! One semantic operation corresponds to the real prepared CUDA graph launch.
//! Its source pointer table may mix arena values and the process-registered
//! Pedersen columns, but neither kind can substitute for the other.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    FixedTableContiguousWorkspaceSlots, FixedTableMaterializerContract,
    FixedTableMaterializerLinkedContract,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::{
    ArenaBinding, BufferPurpose, PlannedFixedTableMaterializer, PlannedFixedTableSource,
};
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, BoundValueRange, EffectAccess,
    EffectBindingId, EffectContract, ElementRange, FixedSourcePointerEntry,
    RegisteredFixedSourceRead, StaticCudaLaunchIdentity, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueRange, ValueVersion,
};
use crate::fixed_table_materializer::{
    pedersen_points_18_column_index, PEDERSEN_POINTS_18_ROW_COUNT,
};
use crate::program_image::lower_compiled::compiled_base_prefix::module_globals::registered_pedersen_source;
use crate::program_image::lower_compiled::recorded_deduce_authority::PedersenTableColumnsAndRowsV1;

mod runtime;
mod validation;

use validation::{source_argument, validate_internal, validate_table};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LoweredFixedTableSource {
    Arena {
        identity: String,
        arena: ArenaBinding,
        value: ArenaCatalogValueId,
        version: ValueVersion,
        elements: ElementRange,
        binding: EffectBindingId,
    },
    Registered {
        identity: String,
        read: RegisteredFixedSourceRead,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredFixedTableMaterialization {
    component: &'static str,
    contract: FixedTableMaterializerContract,
    sources: Vec<LoweredFixedTableSource>,
    multiplicity: ArenaBinding,
    workspace: FixedTableContiguousWorkspaceSlots,
    trace_outputs: Vec<ArenaBinding>,
    lookup_output: ArenaBinding,
    invocation: AotInvocation,
    effect: EffectContract,
}

impl LoweredFixedTableMaterialization {
    pub(super) const fn component(&self) -> &'static str {
        self.component
    }
    pub(super) const fn contract(&self) -> &FixedTableMaterializerContract {
        &self.contract
    }
    pub(super) fn sources(&self) -> &[LoweredFixedTableSource] {
        &self.sources
    }
    pub(super) const fn multiplicity(&self) -> ArenaBinding {
        self.multiplicity
    }
    pub(super) const fn workspace(&self) -> &FixedTableContiguousWorkspaceSlots {
        &self.workspace
    }
    pub(super) fn trace_outputs(&self) -> &[ArenaBinding] {
        &self.trace_outputs
    }
    pub(super) const fn lookup_output(&self) -> ArenaBinding {
        self.lookup_output
    }
    pub(super) const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }
    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredFixedTableStage {
    tables: Vec<LoweredFixedTableMaterialization>,
}

impl LoweredFixedTableStage {
    pub(super) fn tables(&self) -> &[LoweredFixedTableMaterialization] {
        &self.tables
    }
}

#[derive(Clone)]
struct ExactValue {
    arena: ArenaBinding,
    catalog: ArenaCatalogValueId,
}

/// Lower every planned table and publish the allocator only after all tables
/// have passed exact source, geometry, ABI and effect validation.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredFixedTableStage, InvocationShapeError> {
    let multiplicity = arena
        .multiplicity()
        .filter(|plan| {
            plan.coverage_complete() && plan.blockers.is_empty() && !plan.fixed_tables.is_empty()
        })
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    let catalog = BaseProducerCatalog::compile(arena)?;
    preflight_stage_outputs(arena, &catalog, &multiplicity.fixed_tables, values)?;
    let mut next_values = values.clone();
    let mut tables = Vec::with_capacity(multiplicity.fixed_tables.len());
    for table in &multiplicity.fixed_tables {
        tables.push(lower_table(arena, &catalog, table, &mut next_values)?);
    }
    let stage = LoweredFixedTableStage { tables };
    validate_internal(&stage)?;
    *values = next_values;
    Ok(stage)
}

/// Staged-cursor equality gate: re-lower from the exact pre-stage allocator and
/// require both the receipt and resulting allocator to match.
pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredFixedTableStage,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidFixedTableBinding)
    }
}

pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &LoweredFixedTableStage,
    table_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate_internal(lowered)?;
    let table = lowered
        .tables
        .get(table_ordinal)
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    let Some(linked) = table
        .contract
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?
    else {
        return Ok(None);
    };
    project_static_wrapper(id, &linked, table).map(Some)
}

fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &FixedTableMaterializerLinkedContract,
    table: &LoweredFixedTableMaterialization,
) -> Result<StaticCudaWrapperAuthority, InvocationShapeError> {
    linked
        .validate(&table.contract)
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    let launch = table.contract.launch();
    let launch = StaticCudaLaunchIdentity::new(
        table.contract.abi().kernel_symbol().as_bytes().to_vec(),
        crate::compiled_proof::LaunchGeometry {
            grid: launch.grid,
            block: launch.block,
            cluster: launch.cluster,
            dynamic_shared_bytes: launch.dynamic_shared_bytes,
            cooperative: launch.cooperative,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        table.contract.abi().entry_symbol().as_bytes().to_vec(),
        table.contract.abi_identity(),
        table.contract.effect_identity(),
        table.contract.identity(),
        linked.identity(),
        vec![launch],
        table
            .invocation()
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?,
        table.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)
}

fn lower_table(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedFixedTableMaterializer,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredFixedTableMaterialization, InvocationShapeError> {
    let materializer = &planned.plan.materializer;
    if materializer.component() != planned.plan.component
        || materializer.config().row_count != planned.plan.row_count
        || materializer.config().multiplicity_column_count != planned.plan.columns
        || planned.plan.slab_words
            != planned
                .plan
                .row_count
                .checked_mul(planned.plan.columns)
                .ok_or(InvocationShapeError::SizeOverflow)?
        || materializer.preprocessed_sources().len() != planned.sources.len()
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    let contract = FixedTableMaterializerContract::compile(materializer.config())
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    if contract.requirements() != materializer.requirements()
        || contract
            .requirements()
            .arena_slot_requirements_contiguous(&planned.slots)
            .is_err()
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }

    let rows = planned.plan.row_count;
    let full_rows = range(0, rows)?;
    let trace_outputs = (0..contract.requirements().trace_output_count)
        .map(|ordinal| {
            exact_role(
                arena,
                catalog,
                planned.plan.component,
                BufferPurpose::BaseTrace,
                ordinal,
                rows,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lookup_words = rows
        .checked_mul(contract.requirements().lookup_output_count)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let lookup_output = exact_role(
        arena,
        catalog,
        planned.plan.component,
        BufferPurpose::LookupInputs,
        0,
        lookup_words,
    )?;

    let mut accesses = Vec::new();
    let mut next_binding = 0u32;
    let mut source_arguments = Vec::with_capacity(planned.sources.len());
    let mut sources = Vec::with_capacity(planned.sources.len());
    let registered_authority = registered_pedersen_source(PedersenTableColumnsAndRowsV1::CANONICAL)
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
    let mut registered_reads = Vec::new();
    for (&identity, source) in materializer
        .preprocessed_sources()
        .iter()
        .zip(&planned.sources)
    {
        match source {
            PlannedFixedTableSource::Arena(binding) => {
                let preprocessed = arena
                    .preprocessed()
                    .columns
                    .iter()
                    .filter(|column| column.identity == identity)
                    .collect::<Vec<_>>();
                let exact = preprocessed
                    .first()
                    .filter(|_| preprocessed.len() == 1)
                    .and_then(|column| {
                        (column.evaluations == Some(*binding)
                            && (1usize.checked_shl(column.log_size) == Some(rows)))
                        .then_some(())
                    })
                    .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
                let _ = exact;
                let value = exact_binding(
                    arena,
                    catalog,
                    *binding,
                    None,
                    None,
                    BufferPurpose::PreprocessedEvaluations,
                    None,
                    rows,
                )?;
                // Retained preprocessed evaluations are catalog-first fixed
                // sources. No witness producer owns their semantic allocation;
                // publish it here, in the same all-or-none stage transaction.
                values.extend_ordered([value.catalog])?;
                let version = values.version(value.catalog)?;
                let binding_id =
                    push_access(&mut accesses, &mut next_binding, version, full_rows, false)?;
                source_arguments.push(FixedSourcePointerEntry::EffectBinding(binding_id));
                sources.push(LoweredFixedTableSource::Arena {
                    identity: identity.to_owned(),
                    arena: value.arena,
                    value: value.catalog,
                    version,
                    elements: full_rows,
                    binding: binding_id,
                });
            }
            PlannedFixedTableSource::RegisteredPedersen18 { column } => {
                if pedersen_points_18_column_index(identity) != Some(*column)
                    || rows != PEDERSEN_POINTS_18_ROW_COUNT
                {
                    return Err(InvocationShapeError::InvalidFixedTableBinding);
                }
                let read = RegisteredFixedSourceRead::new(
                    registered_authority.clone(),
                    *column,
                    full_rows,
                )
                .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
                registered_reads.push(read.clone());
                source_arguments.push(FixedSourcePointerEntry::Registered(read.clone()));
                sources.push(LoweredFixedTableSource::Registered {
                    identity: identity.to_owned(),
                    read,
                });
            }
        }
    }
    if registered_reads.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }

    let multiplicity = exact_binding(
        arena,
        catalog,
        planned.multiplicity,
        Some(planned.plan.component),
        Some(TracePartId::Main),
        BufferPurpose::FixedMultiplicity,
        Some(0),
        planned.plan.slab_words,
    )?;
    let multiplicity_version = values.version(multiplicity.catalog)?;
    let mut multiplicity_arguments = Vec::with_capacity(planned.plan.columns);
    for column in 0..planned.plan.columns {
        let start = column
            .checked_mul(rows)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let elements = range(
            start,
            start
                .checked_add(rows)
                .ok_or(InvocationShapeError::SizeOverflow)?,
        )?;
        multiplicity_arguments.push(Some(push_access(
            &mut accesses,
            &mut next_binding,
            multiplicity_version,
            elements,
            false,
        )?));
    }

    let trace_mapping = values.register_fixed_u32(
        contract
            .requirements()
            .trace_multiplicity_columns()
            .to_vec(),
    )?;
    let trace_mapping_binding = push_access(
        &mut accesses,
        &mut next_binding,
        trace_mapping,
        range(
            0,
            contract.requirements().trace_multiplicity_columns().len(),
        )?,
        false,
    )?;

    values.extend_ordered(trace_outputs.iter().map(|value| value.catalog))?;
    let mut trace_arguments = Vec::with_capacity(trace_outputs.len());
    for output in &trace_outputs {
        trace_arguments.push(Some(push_access(
            &mut accesses,
            &mut next_binding,
            values.version(output.catalog)?,
            full_rows,
            true,
        )?));
    }

    let lookup_descriptors =
        values.register_fixed_u32(contract.requirements().lookup_descriptors().to_vec())?;
    let lookup_descriptor_binding = push_access(
        &mut accesses,
        &mut next_binding,
        lookup_descriptors,
        range(0, contract.requirements().lookup_descriptor_words)?,
        false,
    )?;
    values.extend_ordered([lookup_output.catalog])?;
    let lookup_version = values.version(lookup_output.catalog)?;
    let mut lookup_arguments = Vec::with_capacity(contract.requirements().lookup_output_count);
    for output in 0..contract.requirements().lookup_output_count {
        let start = output
            .checked_mul(rows)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        lookup_arguments.push(Some(push_access(
            &mut accesses,
            &mut next_binding,
            lookup_version,
            range(
                start,
                start
                    .checked_add(rows)
                    .ok_or(InvocationShapeError::SizeOverflow)?,
            )?,
            true,
        )?));
    }

    let effect = EffectContract::new_with_registered_fixed_source_reads(
        accesses,
        Vec::new(),
        registered_reads,
    )
    .map_err(|_| InvocationShapeError::InvalidFixedTableBinding)?;
    let source_argument = source_argument(source_arguments)?;
    let invocation = AotInvocation {
        arguments: vec![
            argument(0, source_argument),
            argument(
                1,
                AotArgumentValue::DevicePointerTable(multiplicity_arguments),
            ),
            argument(
                2,
                AotArgumentValue::DeviceFixedU32 {
                    value: trace_mapping,
                    binding: trace_mapping_binding,
                },
            ),
            argument(3, AotArgumentValue::DevicePointerTable(trace_arguments)),
            argument(
                4,
                AotArgumentValue::U32(
                    u32::try_from(contract.requirements().trace_output_count)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
            ),
            argument(
                5,
                AotArgumentValue::DeviceFixedU32 {
                    value: lookup_descriptors,
                    binding: lookup_descriptor_binding,
                },
            ),
            argument(6, AotArgumentValue::DevicePointerTable(lookup_arguments)),
            argument(
                7,
                AotArgumentValue::U32(
                    u32::try_from(contract.requirements().lookup_output_count)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
            ),
            argument(
                8,
                AotArgumentValue::U32(
                    u32::try_from(rows).map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
            ),
        ],
    };
    let lowered = LoweredFixedTableMaterialization {
        component: planned.plan.component,
        contract,
        sources,
        multiplicity: multiplicity.arena,
        workspace: planned.slots.clone(),
        trace_outputs: trace_outputs.iter().map(|value| value.arena).collect(),
        lookup_output: lookup_output.arena,
        invocation,
        effect,
    };
    validate_table(&lowered)?;
    Ok(lowered)
}

fn preflight_stage_outputs(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned_tables: &[PlannedFixedTableMaterializer],
    values: &adapter::SemanticValueMap,
) -> Result<(), InvocationShapeError> {
    let mut outputs = BTreeSet::new();
    for planned in planned_tables {
        let requirements = planned.plan.materializer.requirements();
        let rows = planned.plan.row_count;
        for ordinal in 0..requirements.trace_output_count {
            let output = exact_role(
                arena,
                catalog,
                planned.plan.component,
                BufferPurpose::BaseTrace,
                ordinal,
                rows,
            )?;
            if !outputs.insert(output.catalog) || values.version(output.catalog).is_ok() {
                return Err(InvocationShapeError::InvalidFixedTableBinding);
            }
        }
        let lookup_words = rows
            .checked_mul(requirements.lookup_output_count)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let output = exact_role(
            arena,
            catalog,
            planned.plan.component,
            BufferPurpose::LookupInputs,
            0,
            lookup_words,
        )?;
        if !outputs.insert(output.catalog) || values.version(output.catalog).is_ok() {
            return Err(InvocationShapeError::InvalidFixedTableBinding);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn exact_binding(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    binding: ArenaBinding,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    purpose: BufferPurpose,
    ordinal: Option<u32>,
    words: usize,
) -> Result<ExactValue, InvocationShapeError> {
    let id = ArenaCatalogValueId(binding.logical.0);
    let value = catalog.value(id)?;
    let logical = arena
        .logical_buffers()
        .get(binding.logical.0 as usize)
        .filter(|logical| logical.id == binding.logical)
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    if value.logical != binding.logical
        || value.physical != binding.physical
        || value.words != binding.len_words
        || binding.len_words != words
        || logical.component != component
        || logical.part != part
        || logical.purpose != purpose
        || ordinal.is_some_and(|ordinal| logical.ordinal != ordinal)
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    Ok(ExactValue {
        arena: binding,
        catalog: id,
    })
}

fn exact_role(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &'static str,
    purpose: BufferPurpose,
    ordinal: usize,
    words: usize,
) -> Result<ExactValue, InvocationShapeError> {
    let ordinal = u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?;
    let matches = arena
        .logical_buffers()
        .iter()
        .filter(|logical| {
            logical.component == Some(component)
                && logical.part == Some(TracePartId::Main)
                && logical.purpose == purpose
                && logical.ordinal == ordinal
        })
        .collect::<Vec<_>>();
    let logical = matches
        .first()
        .filter(|_| matches.len() == 1)
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    let binding = arena
        .binding(logical.id)
        .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
    exact_binding(
        arena,
        catalog,
        binding,
        Some(component),
        Some(TracePartId::Main),
        purpose,
        Some(ordinal),
        words,
    )
}

fn push_access(
    accesses: &mut Vec<EffectAccess>,
    next_binding: &mut u32,
    version: ValueVersion,
    elements: ElementRange,
    write: bool,
) -> Result<EffectBindingId, InvocationShapeError> {
    let binding = EffectBindingId(*next_binding);
    *next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let bound = BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    };
    accesses.push(if write {
        EffectAccess::Write { destination: bound }
    } else {
        EffectAccess::Read { source: bound }
    });
    Ok(binding)
}

fn argument(ordinal: u8, value: AotArgumentValue) -> AotArgumentBinding {
    AotArgumentBinding { ordinal, value }
}

fn range(start: usize, end: usize) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidFixedTableBinding)
}

#[cfg(test)]
mod tests {
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

    use super::*;
    use crate::compiled_proof::RegisteredFixedSourceAuthority;
    use crate::program_image::lower_compiled::memory_base_trace;
    use crate::program_image::lower_compiled::producer_prefix::BaseProducerAuthority;

    fn registered_read() -> RegisteredFixedSourceRead {
        let source = RegisteredFixedSourceAuthority::new(
            [0x81; 32],
            8,
            8,
            core::mem::size_of::<u32>(),
            vec![b"column_0".to_vec()],
        )
        .unwrap();
        RegisteredFixedSourceRead::new(source, 0, ElementRange { start: 0, end: 8 }).unwrap()
    }

    #[test]
    fn mixed_source_argument_preserves_exact_physical_pointer_order() {
        let ordinary = FixedSourcePointerEntry::EffectBinding(EffectBindingId(7));
        let registered = FixedSourcePointerEntry::Registered(registered_read());
        let expected = vec![ordinary.clone(), registered.clone()];
        assert_eq!(
            source_argument(expected.clone()).unwrap(),
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(expected)
        );
        assert_ne!(
            source_argument(vec![registered, ordinary]).unwrap(),
            source_argument(vec![
                FixedSourcePointerEntry::EffectBinding(EffectBindingId(7)),
                FixedSourcePointerEntry::Registered(registered_read()),
            ])
            .unwrap()
        );
    }

    #[test]
    fn output_catalogs_are_exact_once_and_fail_transactionally() {
        let executable = super::super::tests::generated_sn2_replacement();
        let arena = executable.arena();
        let mut initial = adapter::SemanticValueMap::allocate_ordered(std::iter::empty()).unwrap();
        BaseProducerAuthority::compile_replacement_into(
            arena,
            PreProcessedTraceVariant::Canonical,
            &mut initial,
        )
        .unwrap();
        memory_base_trace::lower_stage(arena, &mut initial).unwrap();

        let mut repeated = initial.clone();
        lower_stage(arena, &mut repeated).unwrap();
        let snapshot = repeated.clone();
        assert_eq!(
            lower_stage(arena, &mut repeated),
            Err(InvocationShapeError::InvalidFixedTableBinding)
        );
        assert_eq!(repeated, snapshot);

        let component = arena.multiplicity().unwrap().fixed_tables[0].plan.component;
        let output = arena
            .logical_buffers()
            .iter()
            .find(|logical| {
                logical.component == Some(component)
                    && logical.part == Some(TracePartId::Main)
                    && logical.purpose == BufferPurpose::BaseTrace
                    && logical.ordinal == 0
            })
            .unwrap();
        let mut preseeded = initial;
        preseeded
            .extend_ordered([ArenaCatalogValueId(output.id.0)])
            .unwrap();
        let snapshot = preseeded.clone();
        assert_eq!(
            lower_stage(arena, &mut preseeded),
            Err(InvocationShapeError::InvalidFixedTableBinding)
        );
        assert_eq!(preseeded, snapshot);
    }
}
