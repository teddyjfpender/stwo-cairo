//! Semantic lowering for canonical row-major CASM witness ingress.
//!
//! Every scheduled lane consumes a distinct SSA version of the one reusable
//! staging allocation. Those versions deliberately have no origin here:
//! eager host ingress remains a required outer operation, not a fabricated
//! `ExternalInput` or runtime publication receipt.

use stwo_backend_cuda::{WitnessCasmInputContract, WitnessCasmInputLinkedContract};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::producer_prefix::ProducerSchedulePosition;
use super::*;
#[cfg(test)]
use crate::arena_plan::PlannedWitnessComponent;
use crate::arena_plan::{ArenaBinding, ProofArenaPlan};
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, StaticCudaWrapperAuthority,
    StaticCudaWrapperId, ValueVersion,
};

mod bindings;
mod projection;
mod semantic;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WitnessCasmWriterUse {
    Active,
    InactiveMechanical,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessCasmStagingBinding {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    /// Previous contents of the reused staging storage, if this is an
    /// overwrite rather than its first proof-local host ingress.
    pub(super) previous: Option<ValueVersion>,
    /// Unresolved output of the real eager host-ingress operation.
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WitnessCasmOutputBinding {
    pub(super) ordinal: u32,
    pub(super) value_kind: stwo_backend_cuda::WitnessCasmInputColumnValue,
    pub(super) writer_use: WitnessCasmWriterUse,
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredWitnessCasmInput {
    pub(super) position: ProducerSchedulePosition,
    pub(super) component: &'static str,
    pub(super) part: TracePartId,
    pub(super) contract: WitnessCasmInputContract,
    pub(super) staging: WitnessCasmStagingBinding,
    pub(super) outputs: Vec<WitnessCasmOutputBinding>,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

impl LoweredWitnessCasmInput {
    pub(super) fn active_output_count(&self) -> usize {
        self.outputs
            .iter()
            .filter(|output| output.writer_use == WitnessCasmWriterUse::Active)
            .count()
    }

    pub(super) fn inactive_mechanical_output_count(&self) -> usize {
        self.outputs.len() - self.active_output_count()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LinkedWitnessCasmInputExecution {
    pub(super) wrapper: StaticCudaWrapperAuthority,
}

/// Lower all row-major CASM materializers in witness-DAG order.
///
/// Publication is transactional. Repeating this operation over its completed
/// map reuses the exact staging lineage and returns byte-identical fragments.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<Vec<LoweredWitnessCasmInput>, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let scheduled = bindings::scheduled_lanes(arena)?;
    let exact = scheduled
        .into_iter()
        .map(|lane| bindings::bind_lane(arena, &catalog, lane))
        .collect::<Result<Vec<_>, _>>()?;
    if exact.is_empty() {
        return Ok(Vec::new());
    }

    let staging_value = exact[0].staging.value;
    if exact.iter().any(|lane| lane.staging.value != staging_value) {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    for lane in &exact {
        for output in &lane.outputs {
            values.version(output.value)?;
        }
    }

    let mut next_values = values.clone();
    let lineage = semantic::exact_staging_lineage(&mut next_values, staging_value, exact.len())?;
    let lowered = exact
        .into_iter()
        .zip(lineage)
        .map(|(lane, transition)| lower_bound_lane(lane, transition, &next_values))
        .collect::<Result<Vec<_>, _>>()?;
    *values = next_values;
    Ok(lowered)
}

/// Reconstruct retained fragments from the completed semantic map.
pub(super) fn validate(
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    supplied: &[LoweredWitnessCasmInput],
) -> Result<(), InvocationShapeError> {
    let mut exact_values = values.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if exact == supplied && &exact_values == values {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn lower_bound_lane(
    lane: bindings::ExactWitnessCasmLane<'_>,
    transition: semantic::StagingTransition,
    values: &adapter::SemanticValueMap,
) -> Result<LoweredWitnessCasmInput, InvocationShapeError> {
    let staging = semantic::staging(lane.staging, transition)?;
    let outputs = semantic::outputs(lane.outputs, values)?;
    let effect = semantic::effect(&staging, &outputs)?;
    let invocation = semantic::invocation(&lane.contract, &staging, &outputs)?;
    semantic::validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredWitnessCasmInput {
        position: lane.position,
        component: lane.component.component,
        part: lane.component.part,
        contract: lane.contract,
        staging,
        outputs,
        invocation,
        effect,
    })
}

pub(super) fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &WitnessCasmInputLinkedContract,
    lowered: &LoweredWitnessCasmInput,
) -> Result<LinkedWitnessCasmInputExecution, InvocationShapeError> {
    projection::linked(id, linked, lowered)
}

#[cfg(test)]
pub(super) fn invocation_using_abi_for_test(
    lowered: &LoweredWitnessCasmInput,
    abi: &[stwo_backend_cuda::WitnessCasmInputAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    semantic::invocation_using_abi(&lowered.contract, &lowered.staging, &lowered.outputs, abi)
}

#[cfg(test)]
pub(super) fn lower_component_for_test(
    arena: &ProofArenaPlan,
    component: &PlannedWitnessComponent,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredWitnessCasmInput, InvocationShapeError> {
    let catalog = BaseProducerCatalog::compile(arena)?;
    let position = bindings::position_for(arena, component)?;
    let exact = bindings::bind_lane(
        arena,
        &catalog,
        bindings::ScheduledWitnessCasmLane {
            position,
            component,
        },
    )?;
    for output in &exact.outputs {
        values.version(output.value)?;
    }
    let mut next_values = values.clone();
    let transition = semantic::exact_staging_lineage(&mut next_values, exact.staging.value, 1)?
        .pop()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    let lowered = lower_bound_lane(exact, transition, &next_values)?;
    *values = next_values;
    Ok(lowered)
}
