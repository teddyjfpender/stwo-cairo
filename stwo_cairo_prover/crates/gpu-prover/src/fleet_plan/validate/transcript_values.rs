use stwo_backend_cuda::{TranscriptInputId, TranscriptOperation, TranscriptOutputId};

use super::super::*;
use super::{operation_placement, owner_ready_at, replica_ready_at};
use crate::compiled_proof::ValueRange;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

pub(super) fn validate_transcript_values(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), FleetPlanError> {
    for binding in plan.compiled.transcript_inputs() {
        let release = release_for_input(plan, transcript, binding.id)?;
        validate_input_location(
            plan,
            ValueRange {
                version: binding.value,
                elements: binding.elements,
            },
            release,
        )?;
    }
    for binding in plan.compiled.transcript_outputs() {
        let range = ValueRange {
            version: binding.value,
            elements: binding.elements,
        };
        let release = release_for_output(plan, transcript, binding.id)?;
        validate_output_location(plan, range, release)?;
        validate_output_consumers(plan, range, release)?;
    }
    Ok(())
}

fn validate_input_location(
    plan: &FleetProofPlan,
    range: ValueRange,
    release: ScheduleStep,
) -> Result<(), FleetPlanError> {
    let coordinator = plan.placement.topology.coordinator;
    let owner = plan.placement.owners.iter().find(|owner| {
        owner.worker == coordinator
            && owner.value.version == range.version
            && owner.value.elements.contains(range.elements)
    });
    let replica = plan.placement.replicas.iter().find(|replica| {
        replica.worker == coordinator
            && replica.value.version == range.version
            && replica.value.elements.contains(range.elements)
            && plan
                .compiled
                .value(range.version)
                .is_some_and(|value| replica.layout == value.layout)
    });
    let available = match (owner, replica) {
        (Some(owner), None) => {
            owner_ready_at(plan, owner)? < release
                && owner.live.start <= release
                && owner.live.end >= release
        }
        (None, Some(replica)) => {
            replica_ready_at(plan, replica)? < release
                && replica.live.start <= release
                && replica.live.end >= release
        }
        _ => false,
    };
    if available {
        Ok(())
    } else {
        Err(FleetPlanError::TranscriptValueCausality(range.version))
    }
}

fn validate_output_location(
    plan: &FleetProofPlan,
    range: ValueRange,
    release: ScheduleStep,
) -> Result<(), FleetPlanError> {
    let mut owners = plan.placement.owners.iter().filter(|owner| {
        owner.worker == plan.placement.topology.coordinator && owner.value == range
    });
    let valid = owners.next().is_some_and(|owner| {
        owners.next().is_none()
            && owner.live.start == release
            && owner.live.end > release
            && owner_ready_at(plan, owner) == Ok(release)
    });
    if valid {
        Ok(())
    } else {
        Err(FleetPlanError::TranscriptValueCausality(range.version))
    }
}

fn validate_output_consumers(
    plan: &FleetProofPlan,
    range: ValueRange,
    release: ScheduleStep,
) -> Result<(), FleetPlanError> {
    for operation in plan.compiled.operations() {
        let effect = plan
            .compiled
            .effect_for(operation.id)
            .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
        let reads = effect.accesses().iter().any(|access| {
            access
                .source()
                .is_some_and(|source| ranges_overlap(source.value, range))
        });
        if reads && operation_placement(plan, operation.id)?.during.start < release {
            return Err(FleetPlanError::TranscriptValueCausality(range.version));
        }
    }
    if plan.placement.transitions.iter().any(|transition| {
        ranges_overlap(transition.value, range) && transition.during.start < release
    }) || plan
        .placement
        .replicas
        .iter()
        .any(|replica| ranges_overlap(replica.value, range) && replica.live.start < release)
    {
        return Err(FleetPlanError::TranscriptValueCausality(range.version));
    }
    for spill in &plan.placement.spills {
        for chunk in spill
            .chunks
            .iter()
            .filter(|chunk| ranges_overlap(chunk.value, range))
        {
            let [d2h, _, _, _] = spill.bounds(chunk.id)?;
            if d2h.during.start < release {
                return Err(FleetPlanError::TranscriptValueCausality(range.version));
            }
        }
    }
    Ok(())
}

pub(super) fn release_for_input(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptInputId,
) -> Result<ScheduleStep, FleetPlanError> {
    release_for_io(plan, transcript, |operation| {
        operation_input(operation) == Some(id)
    })
    .ok_or(FleetPlanError::TranscriptMismatch)
}

pub(super) fn release_for_output(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptOutputId,
) -> Result<ScheduleStep, FleetPlanError> {
    release_for_io(plan, transcript, |operation| {
        operation_output(operation) == Some(id)
    })
    .ok_or(FleetPlanError::TranscriptMismatch)
}

fn release_for_io(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    matches: impl Fn(&TranscriptOperation) -> bool,
) -> Option<ScheduleStep> {
    let boundary = transcript.boundaries().iter().find(|boundary| {
        transcript
            .schedule()
            .operations()
            .get(boundary.operation_index)
            .is_some_and(&matches)
    })?;
    plan.barriers
        .iter()
        .find(|barrier| barrier.segment == boundary.segment)
        .map(|barrier| barrier.release_step)
}

fn operation_input(operation: &TranscriptOperation) -> Option<TranscriptInputId> {
    match *operation {
        TranscriptOperation::MixFelts { source, .. }
        | TranscriptOperation::MixU32s { source, .. }
        | TranscriptOperation::MixU64 { source, .. }
        | TranscriptOperation::AbsorbRoot { source, .. }
        | TranscriptOperation::AbsorbPowNonce { source, .. } => Some(source),
        _ => None,
    }
}

fn operation_output(operation: &TranscriptOperation) -> Option<TranscriptOutputId> {
    match *operation {
        TranscriptOperation::DrawSecureFelt { output, .. }
        | TranscriptOperation::DrawSecureFelts { output, .. }
        | TranscriptOperation::DrawU32s { output, .. }
        | TranscriptOperation::DrawQueries { output, .. } => Some(output),
        _ => None,
    }
}

fn ranges_overlap(left: ValueRange, right: ValueRange) -> bool {
    left.version == right.version && left.elements.overlaps(right.elements)
}
