//! Stage-local lowering for prepared witness-input gathers.
//!
//! This is a fragment of the existing `CompiledProof` IR, not another program
//! representation. Pointer tables remain arena relocation metadata. Only the
//! dereferenced producer/output ranges and the canonical descriptor constant
//! receive semantic bindings.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    WitnessInputGatherAbiAccess, WitnessInputGatherAbiArgument, WitnessInputGatherAbiArgumentKind,
    WitnessInputGatherContract,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::producer_prefix::ProducerSchedulePosition;
use super::*;
use crate::arena_plan::{ArenaBinding, PlannedWitnessComponent, ProofArenaPlan};
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, EffectAccess, EffectBindingId,
    EffectContract, ElementRange, ValueVersion,
};
use crate::resident_runtime::producer_schedule::BaseProducerSchedule;

mod bindings;
use bindings::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessInputGatherRelocations {
    pub(super) source_pointers: ArenaBinding,
    pub(super) descriptor_workspace: ArenaBinding,
    pub(super) output_pointers: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessInputGatherSourceBinding {
    pub(super) producer: &'static str,
    pub(super) arena: ArenaBinding,
    pub(super) pointer_words: ElementRange,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessInputGatherOutputBinding {
    pub(super) ordinal: u32,
    pub(super) arena: ArenaBinding,
    pub(super) pointer_words: ElementRange,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
}

/// One transcript-free gather at its witness-DAG schedule position.
///
/// Global `OpId` and `SemanticOpId` allocation belongs to the proof builder
/// that later consumes this fragment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredWitnessInputGather {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'static str,
    pub(super) part: TracePartId,
    pub(super) contract: WitnessInputGatherContract,
    pub(super) relocations: WitnessInputGatherRelocations,
    pub(super) sources: Vec<WitnessInputGatherSourceBinding>,
    pub(super) descriptor_value: ValueVersion,
    pub(super) descriptor_binding: EffectBindingId,
    pub(super) outputs: Vec<WitnessInputGatherOutputBinding>,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

/// Lower every gather in witness-DAG order without allocating any global
/// operation identity. The update is transactional: a rejected gather leaves
/// the caller's semantic map unchanged.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<Vec<LoweredWitnessInputGather>, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let levels = producer_levels(&schedule)?;
    let mut next_values = values.clone();
    let mut lowered = Vec::new();
    let mut ordinal = 0usize;
    for (level, lanes) in schedule.witness_levels().iter().enumerate() {
        for (lane, producer) in lanes.iter().enumerate() {
            let position = ProducerSchedulePosition {
                level: to_u32(level)?,
                lane: to_u32(lane)?,
                ordinal: to_u32(ordinal)?,
            };
            ordinal = ordinal
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            let Some(part) = producer.part else {
                continue;
            };
            let planned = exact_component(arena, producer.component, part)?;
            if planned.input_gather.is_some() {
                lowered.push(lower_component_inner(
                    arena,
                    &catalog,
                    planned,
                    position,
                    &levels,
                    &mut next_values,
                )?);
            }
        }
    }
    *values = next_values;
    Ok(lowered)
}

#[cfg(test)]
pub(super) fn lower_component(
    arena: &ProofArenaPlan,
    component: &PlannedWitnessComponent,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessInputGather, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let levels = producer_levels(&schedule)?;
    let position = position_for(&schedule, component.component, component.part)?;
    let mut next_values = values.clone();
    let lowered = lower_component_inner(
        arena,
        &catalog,
        component,
        position,
        &levels,
        &mut next_values,
    )?;
    *values = next_values;
    Ok(lowered)
}

/// Revalidate a retained fragment from its exact arena plan and the current
/// proof-wide semantic map.
pub(super) fn validate(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    supplied: &LoweredWitnessInputGather,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = values.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    let matches = exact
        .iter()
        .filter(|fragment| {
            fragment.position == supplied.position
                && fragment.component == supplied.component
                && fragment.part == supplied.part
        })
        .collect::<Vec<_>>();
    if matches.len() == 1 && matches[0] == supplied && &exact_values == values {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_component_inner(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    position: ProducerSchedulePosition,
    producer_levels: &BTreeMap<&'static str, u32>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessInputGather, InvocationShapeError> {
    let planned = component
        .input_gather
        .as_ref()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let contract = WitnessInputGatherContract::compile(&planned.requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    if contract.requirements() != &planned.requirements
        || [
            contract.static_source_identity(),
            contract.wrapper_source_identity(),
            contract.source_identity(),
            contract.requirements_identity(),
            contract.descriptor_identity(),
            contract.abi_identity(),
            contract.effect_identity(),
            contract.launch_identity(),
            contract.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }

    let relocations = exact_relocations(arena, catalog, component, planned, &contract)?;
    let sources = exact_sources(
        arena,
        catalog,
        component,
        planned,
        &contract,
        position,
        producer_levels,
        values,
    )?;
    if sources.last().map(|source| source.pointer_words.end)
        != Some(planned.requirements.source_pointer_words)
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let output_catalog = exact_output_catalog(arena, catalog, component, planned, &contract)?;
    values.extend_ordered(output_catalog.iter().map(|value| value.id))?;
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    let mut outputs = Vec::with_capacity(output_catalog.len());
    let mut next_binding = to_u32(sources.len())?;
    next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    for (ordinal, value) in output_catalog.into_iter().enumerate() {
        let version = values.version(value.id)?;
        if !catalog_first.contains(&version)
            || transitions.contains(&version)
            || fixed.contains(&version)
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        let write = contract
            .effect_geometry()
            .output_writes
            .get(ordinal)
            .filter(|write| write.output_ordinal as usize == ordinal)
            .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
        let elements = exact_range(write.write_start_words, write.write_len_words, value.words)?;
        if elements.start != 0 || elements.end != value.words {
            return Err(InvocationShapeError::InvalidStructuredAbi);
        }
        outputs.push(WitnessInputGatherOutputBinding {
            ordinal: to_u32(ordinal)?,
            arena: exact_arena_binding(arena, value)?,
            pointer_words: pointer_entry_range(ordinal, planned.requirements.output_pointer_words)?,
            value: value.id,
            elements,
            binding: EffectBindingId(next_binding),
        });
        next_binding = next_binding
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
    }
    if outputs
        .iter()
        .map(|output| output.value)
        .collect::<BTreeSet<_>>()
        .len()
        != outputs.len()
        || outputs.last().map(|output| output.pointer_words.end)
            != Some(planned.requirements.output_pointer_words)
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }

    let descriptor_value = values.register_fixed_u32(contract.descriptor_words().to_vec())?;
    let (_, transitions, fixed) = values.allocation_classes();
    if transitions.contains(&descriptor_value) || !fixed.contains(&descriptor_value) {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let descriptor_binding = EffectBindingId(to_u32(sources.len())?);
    let geometry = contract.effect_geometry();
    let descriptor_elements = exact_range(
        geometry.descriptor_read_start_words,
        geometry.descriptor_read_len_words,
        contract.descriptor_words().len(),
    )?;
    if descriptor_elements.start != 0
        || descriptor_elements.end != contract.descriptor_words().len()
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let mut accesses = Vec::with_capacity(sources.len() + outputs.len() + 1);
    for source in &sources {
        accesses.push(EffectAccess::Read {
            source: bound(
                source.binding,
                values.version(source.value)?,
                source.elements,
            ),
        });
    }
    accesses.push(EffectAccess::Read {
        source: bound(descriptor_binding, descriptor_value, descriptor_elements),
    });
    for output in &outputs {
        accesses.push(EffectAccess::Write {
            destination: bound(
                output.binding,
                values.version(output.value)?,
                output.elements,
            ),
        });
    }
    let effect = EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)?;
    let invocation = exact_invocation(
        &contract,
        contract.abi().arguments(),
        &sources,
        descriptor_value,
        descriptor_binding,
        &outputs,
    )?;
    validate_exact_binding_consumption(&invocation, &effect, descriptor_value, descriptor_binding)?;

    Ok(LoweredWitnessInputGather {
        position,
        component: component.component,
        part: component.part,
        contract,
        relocations,
        sources,
        descriptor_value,
        descriptor_binding,
        outputs,
        invocation,
        effect,
    })
}

fn exact_invocation(
    contract: &WitnessInputGatherContract,
    abi: &[WitnessInputGatherAbiArgument],
    sources: &[WitnessInputGatherSourceBinding],
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    outputs: &[WitnessInputGatherOutputBinding],
) -> Result<AotInvocation, InvocationShapeError> {
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let geometry = contract.effect_geometry();
    let mut arguments = Vec::with_capacity(abi.len().saturating_sub(1));
    for descriptor in abi {
        use WitnessInputGatherAbiAccess as Access;
        use WitnessInputGatherAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (
                0,
                "producer_subs_dev",
                Kind::DeviceConstPointerTableU32,
                Access::ReadPackedProducerColumns,
            ) => AotArgumentValue::DevicePointerTable(
                sources.iter().map(|source| Some(source.binding)).collect(),
            ),
            (
                1,
                "edge_descs_dev",
                Kind::DeviceConstPointerU32,
                Access::ReadCanonicalEdgeDescriptors,
            ) => AotArgumentValue::DeviceFixedU32 {
                value: descriptor_value,
                binding: descriptor_binding,
            },
            (2, "n_edges", Kind::U32, Access::EdgeCount) => {
                AotArgumentValue::U32(to_u32(sources.len())?)
            }
            (3, "input_width", Kind::U32, Access::InputWidth) => {
                AotArgumentValue::U32(geometry.input_columns)
            }
            (4, "total_real_rows", Kind::U32, Access::TotalRealRows) => {
                AotArgumentValue::U32(geometry.total_real_rows)
            }
            (5, "consumer_rows", Kind::U32, Access::ConsumerRows) => {
                AotArgumentValue::U32(geometry.consumer_rows)
            }
            (
                6,
                "consumer_cols_dev",
                Kind::DeviceMutPointerTableU32,
                Access::WriteConsumerColumns,
            ) => AotArgumentValue::DevicePointerTable(
                outputs.iter().map(|output| Some(output.binding)).collect(),
            ),
            (7, "include_enabler", Kind::U32, Access::IncludeEnabler) => {
                AotArgumentValue::U32(u32::from(geometry.include_enabler))
            }
            (8, "include_iota", Kind::U32, Access::IncludeIota) => {
                AotArgumentValue::U32(u32::from(geometry.include_iota))
            }
            (9, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    if arguments.len() + 1 != abi.len()
        || arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(AotInvocation { arguments })
}

fn validate_exact_binding_consumption(
    invocation: &AotInvocation,
    effect: &EffectContract,
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
) -> Result<(), InvocationShapeError> {
    if !effect.registered_fixed_source_reads().is_empty() {
        return Err(InvocationShapeError::InvalidAdapterEffect);
    }
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for argument in &invocation.arguments {
        let mut insert = |binding| {
            if actual.insert(binding) {
                Ok(())
            } else {
                Err(InvocationShapeError::InvalidAdapterEffect)
            }
        };
        match &argument.value {
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding)?,
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() || entries.iter().any(Option::is_none) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceFixedU32 { value, binding } => {
                if *value != descriptor_value || *binding != descriptor_binding {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                insert(*binding)?;
            }
        }
    }
    if actual == expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

#[cfg(test)]
pub(super) fn invocation_using_abi_for_test(
    fragment: &LoweredWitnessInputGather,
    abi: &[WitnessInputGatherAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    exact_invocation(
        &fragment.contract,
        abi,
        &fragment.sources,
        fragment.descriptor_value,
        fragment.descriptor_binding,
        &fragment.outputs,
    )
}
