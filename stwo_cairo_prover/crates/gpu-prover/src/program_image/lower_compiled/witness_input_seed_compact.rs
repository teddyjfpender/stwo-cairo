//! Exact preproducer lowering for device-seeded and compacted witness inputs.
//!
//! These operations run immediately before their recorded Base writer at the
//! same schedule position. Pointer tables remain relocation metadata. Every
//! dereferenced scalar, producer range, output column, descriptor constant and
//! compact scratch range receives one proof-wide semantic binding.

use stwo_backend_cuda::{WitnessInputCompactContract, WitnessInputSeedContract};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::producer_prefix::ProducerSchedulePosition;
use super::*;
use crate::arena_plan::{ArenaBinding, PlannedWitnessComponent, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueVersion,
};
use crate::resident_runtime::producer_schedule::{BaseProducerSchedule, WitnessProducerKind};

mod bindings;
mod projection;
mod semantic;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SemanticArenaBinding {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessInputSeedRelocations {
    pub(super) output_pointers: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessInputCompactRelocations {
    pub(super) source_pointers: ArenaBinding,
    pub(super) descriptor_workspace: ArenaBinding,
    pub(super) output_pointers: ArenaBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredWitnessInputSeed {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'static str,
    pub(super) part: TracePartId,
    pub(super) contract: WitnessInputSeedContract,
    pub(super) relocations: WitnessInputSeedRelocations,
    pub(super) scalar_source: SemanticArenaBinding,
    pub(super) outputs: Vec<SemanticArenaBinding>,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredWitnessInputCompact {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'static str,
    pub(super) part: TracePartId,
    pub(super) contract: WitnessInputCompactContract,
    pub(super) relocations: WitnessInputCompactRelocations,
    pub(super) sources: Vec<SemanticArenaBinding>,
    pub(super) descriptor_value: ValueVersion,
    pub(super) descriptor_binding: EffectBindingId,
    pub(super) outputs: Vec<SemanticArenaBinding>,
    pub(super) scratch: Vec<SemanticArenaBinding>,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LoweredWitnessInputSetup {
    Seed(LoweredWitnessInputSeed),
    Compact(LoweredWitnessInputCompact),
}

impl LoweredWitnessInputSetup {
    pub(super) const fn position(&self) -> ProducerSchedulePosition {
        match self {
            Self::Seed(lowered) => lowered.position,
            Self::Compact(lowered) => lowered.position,
        }
    }

    pub(super) const fn component(&self) -> &'static str {
        match self {
            Self::Seed(lowered) => lowered.component,
            Self::Compact(lowered) => lowered.component,
        }
    }

    pub(super) const fn part(&self) -> TracePartId {
        match self {
            Self::Seed(lowered) => lowered.part,
            Self::Compact(lowered) => lowered.part,
        }
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        match self {
            Self::Seed(lowered) => &lowered.effect,
            Self::Compact(lowered) => &lowered.effect,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedWitnessInputSeedExecution {
    pub(super) wrapper: StaticCudaWrapperAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedWitnessInputCompactExecution {
    pub(super) wrapper: StaticCudaWrapperAuthority,
    pub(super) invocation: AotInvocation,
    pub(super) exact_sort_temp_bytes: usize,
    pub(super) exact_scan_temp_bytes: usize,
}

/// Lower all generated seed/compact preproducers in witness-DAG order.
///
/// Publication is transactional: any invalid component, catalog role, source
/// edge, effect or ABI leaves `values` byte-for-byte unchanged.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<Vec<LoweredWitnessInputSetup>, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let levels = bindings::producer_levels(&schedule)?;
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
            let component = bindings::exact_component(arena, producer.component, part)?;
            let setup_count = usize::from(component.input_seed.is_some())
                + usize::from(component.input_compact.is_some());
            if setup_count == 0 {
                continue;
            }
            if setup_count != 1 || producer.kind != WitnessProducerKind::Recorded {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            lowered.push(if component.input_seed.is_some() {
                LoweredWitnessInputSetup::Seed(lower_seed_inner(
                    arena,
                    &catalog,
                    component,
                    position,
                    &mut next_values,
                )?)
            } else {
                LoweredWitnessInputSetup::Compact(lower_compact_inner(
                    arena,
                    &catalog,
                    component,
                    position,
                    &levels,
                    &mut next_values,
                )?)
            });
        }
    }
    *values = next_values;
    Ok(lowered)
}

/// Rebuild a retained stage from its arena authority and current semantic map.
pub(super) fn validate(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    supplied: &[LoweredWitnessInputSetup],
) -> Result<(), InvocationShapeError> {
    let mut exact_values = values.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if exact == supplied && &exact_values == values {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

#[cfg(test)]
pub(super) fn lower_component(
    arena: &ProofArenaPlan,
    component: &PlannedWitnessComponent,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessInputSetup, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let schedule = BaseProducerSchedule::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let levels = bindings::producer_levels(&schedule)?;
    let position = bindings::position_for(&schedule, component.component, component.part)?;
    let mut next_values = values.clone();
    let setup = match (
        component.input_seed.as_ref(),
        component.input_compact.as_ref(),
    ) {
        (Some(_), None) => LoweredWitnessInputSetup::Seed(lower_seed_inner(
            arena,
            &catalog,
            component,
            position,
            &mut next_values,
        )?),
        (None, Some(_)) => LoweredWitnessInputSetup::Compact(lower_compact_inner(
            arena,
            &catalog,
            component,
            position,
            &levels,
            &mut next_values,
        )?),
        _ => return Err(InvocationShapeError::InvalidScheduledProducerBinding),
    };
    *values = next_values;
    Ok(setup)
}

fn lower_seed_inner(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    position: ProducerSchedulePosition,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessInputSeed, InvocationShapeError> {
    let planned = component
        .input_seed
        .as_ref()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let contract = WitnessInputSeedContract::compile(&planned.requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    bindings::validate_seed_contract(&contract, planned)?;
    let exact = bindings::seed(arena, catalog, component, planned, &contract)?;

    values.extend_ordered(
        std::iter::once(exact.scalar.value).chain(exact.outputs.iter().map(|output| output.value)),
    )?;
    let scalar_source = semantic::bind_semantic(exact.scalar, values, EffectBindingId(0))?;
    let outputs = exact
        .outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            semantic::bind_semantic(output, values, EffectBindingId(to_u32(index + 1)?))
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?;
    let effect = semantic::seed_effect(&scalar_source, &outputs)?;
    let invocation = semantic::seed_invocation(&contract, &scalar_source, &outputs)?;
    semantic::validate_exact_bindings(&invocation, &effect, None)?;

    Ok(LoweredWitnessInputSeed {
        position,
        component: component.component,
        part: component.part,
        contract,
        relocations: exact.relocations,
        scalar_source,
        outputs,
        invocation,
        effect,
    })
}

fn lower_compact_inner(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    position: ProducerSchedulePosition,
    producer_levels: &std::collections::BTreeMap<&'static str, u32>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessInputCompact, InvocationShapeError> {
    let planned = component
        .input_compact
        .as_ref()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let contract = WitnessInputCompactContract::compile(&planned.requirements)
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?;
    bindings::validate_compact_contract(&contract, planned)?;
    let exact = bindings::compact(
        arena,
        catalog,
        component,
        planned,
        &contract,
        position,
        producer_levels,
    )?;

    for source in &exact.sources {
        values.version(source.value)?;
    }
    values.extend_ordered(
        exact
            .outputs
            .iter()
            .chain(&exact.scratch)
            .map(|binding| binding.value),
    )?;

    let mut next_binding = 0u32;
    let sources = semantic::bind_many(exact.sources, values, &mut next_binding)?;
    let descriptor_value = values.register_fixed_u32(contract.descriptor_words().to_vec())?;
    let descriptor_binding = semantic::take_binding(&mut next_binding)?;
    let outputs = semantic::bind_many(exact.outputs, values, &mut next_binding)?;
    let scratch = semantic::bind_many(exact.scratch, values, &mut next_binding)?;
    let effect = semantic::compact_effect(
        &sources,
        descriptor_value,
        descriptor_binding,
        contract.descriptor_words().len(),
        &outputs,
        &scratch,
    )?;

    Ok(LoweredWitnessInputCompact {
        position,
        component: component.component,
        part: component.part,
        contract,
        relocations: exact.relocations,
        sources,
        descriptor_value,
        descriptor_binding,
        outputs,
        scratch,
        effect,
    })
}

fn to_u32(value: usize) -> Result<u32, InvocationShapeError> {
    u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)
}

pub(super) fn project_seed_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &stwo_backend_cuda::WitnessInputSeedLinkedContract,
    lowered: &LoweredWitnessInputSeed,
) -> Result<LinkedWitnessInputSeedExecution, InvocationShapeError> {
    projection::seed(id, linked, lowered)
}

pub(super) fn project_compact_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &stwo_backend_cuda::WitnessInputCompactLinkedContract,
    lowered: &LoweredWitnessInputCompact,
) -> Result<LinkedWitnessInputCompactExecution, InvocationShapeError> {
    projection::compact(id, linked, lowered)
}

pub(super) fn compact_invocation_from_wrapper(
    lowered: &LoweredWitnessInputCompact,
    wrapper: &StaticCudaWrapperAuthority,
) -> Result<AotInvocation, InvocationShapeError> {
    projection::compact_invocation_from_execution_steps(lowered, wrapper.execution_steps())
}

#[cfg(test)]
pub(super) fn compact_test_execution(
    lowered: &LoweredWitnessInputCompact,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
) -> Result<
    (
        AotInvocation,
        Vec<crate::compiled_proof::StaticCudaExecutionStepIdentity>,
    ),
    InvocationShapeError,
> {
    Ok((
        projection::compact_invocation_for_test(
            lowered,
            exact_sort_temp_bytes,
            exact_scan_temp_bytes,
        )?,
        projection::compact_steps_for_test(
            &lowered.contract,
            exact_sort_temp_bytes,
            exact_scan_temp_bytes,
        )?,
    ))
}

#[cfg(test)]
pub(super) fn seed_invocation_using_abi_for_test(
    lowered: &LoweredWitnessInputSeed,
    abi: &[stwo_backend_cuda::WitnessInputSeedAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    semantic::seed_invocation_using_abi(
        &lowered.contract,
        &lowered.scalar_source,
        &lowered.outputs,
        abi,
    )
}
