use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{TranscriptInputId, TranscriptOperation, TranscriptOutputId};

use super::super::*;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

pub(super) fn validate_transcript_values(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
) -> Result<(), FleetPlanError> {
    if plan.input.transcript_inputs.len() != transcript.inputs().len() {
        let id = transcript
            .inputs()
            .get(plan.input.transcript_inputs.len())
            .or_else(|| transcript.inputs().last())
            .ok_or(FleetPlanError::TranscriptMismatch)?
            .semantic
            .id()
            .map_err(|_| FleetPlanError::TranscriptMismatch)?;
        return Err(FleetPlanError::InvalidTranscriptInput(id));
    }
    for (binding, requirement) in plan.input.transcript_inputs.iter().zip(transcript.inputs()) {
        let expected = requirement
            .semantic
            .id()
            .map_err(|_| FleetPlanError::TranscriptMismatch)?;
        if binding.id != expected {
            return Err(FleetPlanError::InvalidTranscriptInput(expected));
        }
        validate_input_binding(plan, transcript, binding, requirement.min_words, values)?;
    }
    for (index, left) in plan.input.transcript_inputs.iter().enumerate() {
        if let Some(right) = plan.input.transcript_inputs[index + 1..]
            .iter()
            .find(|right| left.value == right.value && left.elements.overlaps(right.elements))
        {
            return Err(FleetPlanError::InvalidTranscriptInput(right.id));
        }
    }

    if plan.input.transcript_outputs.len() != transcript.outputs().len() {
        let id = transcript
            .outputs()
            .get(plan.input.transcript_outputs.len())
            .or_else(|| transcript.outputs().last())
            .ok_or(FleetPlanError::TranscriptMismatch)?
            .semantic
            .id()
            .map_err(|_| FleetPlanError::TranscriptMismatch)?;
        return Err(FleetPlanError::InvalidTranscriptOutput(id));
    }
    let mut output_values = BTreeSet::new();
    for (binding, requirement) in plan
        .input
        .transcript_outputs
        .iter()
        .zip(transcript.outputs())
    {
        let expected = requirement
            .semantic
            .id()
            .map_err(|_| FleetPlanError::TranscriptMismatch)?;
        if binding.id != expected {
            return Err(FleetPlanError::InvalidTranscriptOutput(expected));
        }
        if !output_values.insert(binding.value) {
            return Err(FleetPlanError::InvalidTranscriptOutput(binding.id));
        }
        validate_output_binding(
            plan,
            transcript,
            binding,
            requirement.min_words,
            values,
            operations,
        )?;
    }

    for value in values.values() {
        if let ValueOrigin::TranscriptOutput(id) = value.origin {
            let exact = plan
                .input
                .transcript_outputs
                .iter()
                .filter(|binding| binding.id == id && binding.value == value.id)
                .count();
            if exact != 1 {
                return Err(FleetPlanError::InvalidTranscriptOutput(id));
            }
        }
    }
    Ok(())
}

fn validate_input_binding(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    binding: &TranscriptInputValueBinding,
    required_words: usize,
    values: &BTreeMap<ValueId, &ValueDesc>,
) -> Result<(), FleetPlanError> {
    let value = values
        .get(&binding.value)
        .ok_or(FleetPlanError::UnknownValue(binding.value))?;
    validate_word_range(value, binding.elements, required_words)
        .map_err(|_| FleetPlanError::InvalidTranscriptInput(binding.id))?;
    let release = release_for_input(plan, transcript, binding.id)?;
    let owner = plan.input.owners.iter().find(|owner| {
        owner.value == binding.value
            && owner.worker == plan.input.topology.coordinator
            && owner.elements.contains(binding.elements)
    });
    let replica = plan.input.replicas.iter().find(|replica| {
        replica.value == binding.value
            && replica.worker == plan.input.topology.coordinator
            && replica.elements.contains(binding.elements)
            && replica.layout == value.layout
    });
    let (ready_at, live) = match (owner, replica) {
        (Some(owner), None) => (owner.ready_at, owner.live),
        (None, Some(replica)) => (replica.ready_at, replica.live),
        _ => return Err(FleetPlanError::TranscriptValueCausality(binding.value)),
    };
    if ready_at >= release || live.start > ready_at || live.end < release {
        return Err(FleetPlanError::TranscriptValueCausality(binding.value));
    }
    Ok(())
}

fn validate_output_binding(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    binding: &TranscriptOutputValueBinding,
    required_words: usize,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
) -> Result<(), FleetPlanError> {
    let value = values
        .get(&binding.value)
        .ok_or(FleetPlanError::UnknownValue(binding.value))?;
    validate_word_range(value, binding.elements, required_words)
        .map_err(|_| FleetPlanError::InvalidTranscriptOutput(binding.id))?;
    let total = value.layout.element_count()?;
    if value.origin != ValueOrigin::TranscriptOutput(binding.id)
        || binding.elements.start != 0
        || binding.elements.end != total
    {
        return Err(FleetPlanError::InvalidTranscriptOutput(binding.id));
    }
    let release = release_for_output(plan, transcript, binding.id)?;
    let mut owners = plan.input.owners.iter().filter(|owner| {
        owner.value == binding.value
            && owner.elements == binding.elements
            && owner.worker == plan.input.topology.coordinator
    });
    let owner = owners
        .next()
        .ok_or(FleetPlanError::TranscriptValueCausality(binding.value))?;
    if owners.next().is_some()
        || owner.producer.is_some()
        || owner.ready_at != release
        || owner.live.start != release
        || owner.live.end <= release
    {
        return Err(FleetPlanError::TranscriptValueCausality(binding.value));
    }
    validate_output_consumers(plan, binding, release, operations)
}

fn validate_word_range(
    value: &ValueDesc,
    elements: ElementRange,
    required_words: usize,
) -> Result<(), FleetPlanError> {
    if value.layout.element.bytes != size_of::<u32>()
        || value.alignment_bytes < align_of::<u32>()
        || elements.is_empty()
        || elements.len() != required_words
        || elements.end > value.layout.element_count()?
    {
        return Err(FleetPlanError::InvalidRange(value.id));
    }
    Ok(())
}

fn validate_output_consumers(
    plan: &FleetProofPlan,
    binding: &TranscriptOutputValueBinding,
    release: ScheduleStep,
    operations: &BTreeMap<OperationId, &OperationDesc>,
) -> Result<(), FleetPlanError> {
    for operation in operations.values() {
        let reads_output = operation
            .reads
            .iter()
            .any(|read| read.value == binding.value && read.elements.overlaps(binding.elements));
        if reads_output && operation.during.start < release {
            return Err(FleetPlanError::TranscriptValueCausality(binding.value));
        }
    }
    if plan.input.transitions.iter().any(|transition| {
        transition.value == binding.value
            && transition.elements.overlaps(binding.elements)
            && transition.during.start < release
    }) || plan.input.replicas.iter().any(|replica| {
        replica.value == binding.value
            && replica.elements.overlaps(binding.elements)
            && (replica.live.start < release || replica.ready_at < release)
    }) {
        return Err(FleetPlanError::TranscriptValueCausality(binding.value));
    }
    for spill in &plan.input.spills {
        for chunk in spill.chunks.iter().filter(|chunk| {
            chunk.value == binding.value && chunk.elements.overlaps(binding.elements)
        }) {
            let [d2h, _, _, _] = spill.chain(chunk.id)?;
            if d2h.during.start < release {
                return Err(FleetPlanError::TranscriptValueCausality(binding.value));
            }
        }
    }
    Ok(())
}

fn release_for_input(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptInputId,
) -> Result<ScheduleStep, FleetPlanError> {
    release_for_io(plan, transcript, |operation| {
        operation_input(operation) == Some(id)
    })
    .ok_or(FleetPlanError::InvalidTranscriptInput(id))
}

fn release_for_output(
    plan: &FleetProofPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    id: TranscriptOutputId,
) -> Result<ScheduleStep, FleetPlanError> {
    release_for_io(plan, transcript, |operation| {
        operation_output(operation) == Some(id)
    })
    .ok_or(FleetPlanError::InvalidTranscriptOutput(id))
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
