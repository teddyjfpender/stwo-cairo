use stwo_cairo_gpu_prover::transcript_plan::{CairoTranscriptOutput, CairoTranscriptSegment};

use super::*;

const TRANSCRIPT_VALUE_TAG: u32 = 0x5452_4e53;
const TRANSCRIPT_STORAGE_START: usize = 128;

pub(super) fn bind_transcript_values(input: &mut FleetPlanInput) {
    let worker = input.topology.coordinator;
    let storage = StorageId(0);
    let storage_index = input
        .storages
        .iter()
        .position(|desc| desc.id == storage && desc.worker == worker)
        .unwrap();
    let mut offset = align_up(
        input.storages[storage_index]
            .bytes
            .max(TRANSCRIPT_STORAGE_START),
        32,
    );

    for (ordinal, requirement) in transcript().inputs().iter().enumerate() {
        let id = requirement.semantic.id().unwrap();
        let value = append_value(
            input,
            ValueOrigin::ExternalInput(0x1000 + ordinal as u32),
            requirement.min_words,
            ScheduleStep(0),
            during(0, terminal_step().0),
            storage,
            offset,
            worker,
        );
        input.transcript_inputs.push(TranscriptInputValueBinding {
            id,
            value,
            elements: range(0, requirement.min_words),
        });
        offset = align_up(offset + requirement.min_words * 4, 32);
    }

    for requirement in transcript().outputs() {
        let id = requirement.semantic.id().unwrap();
        let release = output_release(input, requirement.semantic);
        let value = append_value(
            input,
            ValueOrigin::TranscriptOutput(id),
            requirement.min_words,
            release,
            during(release.0, terminal_step().0),
            storage,
            offset,
            worker,
        );
        input.transcript_outputs.push(TranscriptOutputValueBinding {
            id,
            value,
            elements: range(0, requirement.min_words),
        });
        offset = align_up(offset + requirement.min_words * 4, 32);
    }

    input.storages[storage_index].bytes = offset;
    let worker_spec = input
        .topology
        .workers
        .iter_mut()
        .find(|spec| spec.id == worker)
        .unwrap();
    let reserved = input
        .storages
        .iter()
        .filter(|desc| desc.worker == worker)
        .map(|desc| desc.bytes)
        .sum::<usize>();
    worker_spec.capacity_bytes = worker_spec
        .capacity_bytes
        .max(reserved + worker_spec.exchange_reserve_bytes + 1024);
}

fn append_value(
    input: &mut FleetPlanInput,
    origin: ValueOrigin,
    words: usize,
    ready_at: ScheduleStep,
    live: ScheduleRange,
    storage: StorageId,
    offset_bytes: usize,
    worker: WorkerId,
) -> ValueId {
    let value = ValueId(input.values.len() as u32);
    input.values.push(ValueDesc {
        id: value,
        layout: transcript_layout(words),
        alignment_bytes: 4,
        origin,
    });
    let elements = range(0, words);
    input.owners.push(OwnedValueRange {
        value,
        elements,
        worker,
        producer: None,
        ready_at,
        live,
    });
    input.storage_bindings.push(StorageBinding {
        storage,
        value,
        elements,
        worker,
        offset_bytes,
        bytes: words * 4,
    });
    value
}

fn transcript_layout(words: usize) -> ValueLayout {
    ValueLayout {
        element: ElementType {
            tag: TRANSCRIPT_VALUE_TAG,
            bytes: 4,
        },
        axes: vec![LayoutAxis {
            tag: 0,
            extent: words,
            stride_bytes: 4,
        }],
    }
}

fn output_release(input: &FleetPlanInput, output: CairoTranscriptOutput) -> ScheduleStep {
    let segment = match output {
        CairoTranscriptOutput::CommonLookupElements => {
            CairoTranscriptSegment::InteractionPowAndLookup
        }
        CairoTranscriptOutput::CompositionRandomCoefficient => {
            CairoTranscriptSegment::InteractionAndComposition
        }
        CairoTranscriptOutput::OodsPointParameter => CairoTranscriptSegment::CompositionAndOods,
        CairoTranscriptOutput::QuotientRandomCoefficient => CairoTranscriptSegment::OodsAndQuotient,
        CairoTranscriptOutput::FriFoldingChallenge(layer) => {
            CairoTranscriptSegment::FriLayer(layer)
        }
        CairoTranscriptOutput::QueryPositions => CairoTranscriptSegment::QueryPowAndPositions,
    };
    let ordinal = transcript()
        .segments()
        .iter()
        .position(|candidate| candidate.segment == segment)
        .unwrap();
    input.barrier_steps[ordinal]
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

fn equal_width_input_pair(input: &FleetPlanInput) -> (usize, usize) {
    for (left, binding) in input.transcript_inputs.iter().enumerate() {
        if let Some(right) = input.transcript_inputs[left + 1..]
            .iter()
            .position(|candidate| candidate.elements.len() == binding.elements.len())
        {
            return (left, left + 1 + right);
        }
    }
    panic!("fixture has two equal-width transcript inputs")
}

#[test]
fn transcript_input_order_and_completeness_fail_closed() {
    let expected = one_worker_input().transcript_inputs[0].id;
    let mut swapped = one_worker_input();
    swapped.transcript_inputs.swap(0, 1);
    assert_eq!(
        compile(swapped).unwrap_err(),
        FleetPlanError::InvalidTranscriptInput(expected)
    );

    let mut missing = one_worker_input();
    missing.transcript_inputs.remove(0);
    assert!(matches!(
        compile(missing).unwrap_err(),
        FleetPlanError::InvalidTranscriptInput(_)
    ));
}

#[test]
fn transcript_input_range_is_exact_u32_words() {
    let mut broken = one_worker_input();
    broken.transcript_inputs[0].elements.end -= 1;
    let id = broken.transcript_inputs[0].id;
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::InvalidTranscriptInput(id)
    );

    let mut wrong_layout = one_worker_input();
    let binding = wrong_layout.transcript_inputs[0];
    wrong_layout.values[binding.value.0 as usize]
        .layout
        .element
        .bytes = 8;
    wrong_layout.values[binding.value.0 as usize].layout.axes[0].stride_bytes = 8;
    assert_eq!(
        compile(wrong_layout).unwrap_err(),
        FleetPlanError::InvalidTranscriptInput(binding.id)
    );
}

#[test]
fn transcript_inputs_cannot_alias_and_valid_sources_are_identity_bound() {
    let baseline = compile(one_worker_input()).unwrap();
    let mut swapped_sources = one_worker_input();
    let (left, right) = equal_width_input_pair(&swapped_sources);
    let left_value = swapped_sources.transcript_inputs[left].value;
    swapped_sources.transcript_inputs[left].value = swapped_sources.transcript_inputs[right].value;
    swapped_sources.transcript_inputs[right].value = left_value;
    assert_ne!(
        baseline.identity(),
        compile(swapped_sources).unwrap().identity()
    );

    let mut overlapping = one_worker_input();
    let (left, right) = equal_width_input_pair(&overlapping);
    overlapping.transcript_inputs[right].value = overlapping.transcript_inputs[left].value;
    overlapping.transcript_inputs[right].elements = overlapping.transcript_inputs[left].elements;
    let id = overlapping.transcript_inputs[right].id;
    assert_eq!(
        compile(overlapping).unwrap_err(),
        FleetPlanError::InvalidTranscriptInput(id)
    );
}

#[test]
fn transcript_output_origin_and_id_fail_closed() {
    let mut missing = one_worker_input();
    missing.transcript_outputs.remove(0);
    assert!(matches!(
        compile(missing).unwrap_err(),
        FleetPlanError::InvalidTranscriptOutput(_)
    ));

    let mut fake_origin = one_worker_input();
    let binding = fake_origin.transcript_outputs[0];
    fake_origin.values[binding.value.0 as usize].origin = ValueOrigin::ExternalInput(99);
    assert_eq!(
        compile(fake_origin).unwrap_err(),
        FleetPlanError::InvalidTranscriptOutput(binding.id)
    );

    let mut fake_id = one_worker_input();
    let expected = fake_id.transcript_outputs[0].id;
    fake_id.transcript_outputs[0].id = fake_id.transcript_outputs[1].id;
    assert_eq!(
        compile(fake_id).unwrap_err(),
        FleetPlanError::InvalidTranscriptOutput(expected)
    );
}

#[test]
fn transcript_output_owner_is_ready_at_exact_release() {
    for delta in [-1i32, 1] {
        let mut broken = one_worker_input();
        let value = broken.transcript_outputs[0].value;
        let owner = broken
            .owners
            .iter_mut()
            .find(|owner| owner.value == value)
            .unwrap();
        owner.ready_at.0 = owner.ready_at.0.checked_add_signed(delta).unwrap();
        assert_eq!(
            compile(broken).unwrap_err(),
            FleetPlanError::TranscriptValueCausality(value)
        );
    }
}

#[test]
fn transcript_operation_input_cannot_be_produced_after_its_release() {
    let mut broken = one_worker_input();
    let binding = broken.transcript_inputs[0];
    let operation = OperationId(broken.operations.len() as u32);
    broken.values[binding.value.0 as usize].origin = ValueOrigin::Operation;
    let owner = broken
        .owners
        .iter_mut()
        .find(|owner| owner.value == binding.value)
        .unwrap();
    owner.producer = Some(operation);
    owner.ready_at = ScheduleStep(102);
    owner.live = during(101, terminal_step().0);
    broken.operations.push(OperationDesc {
        id: operation,
        semantic: b"late-transcript-input".to_vec(),
        effect_identity: storage_alias::effect(90),
        interval: ExecutionInterval::BeforeBarrier(1),
        during: during(101, 102),
        reads: vec![],
        writes: vec![ValueUse {
            value: binding.value,
            elements: binding.elements,
            layout: broken.values[binding.value.0 as usize].layout.clone(),
        }],
    });
    broken.assignments.push(OperationAssignment {
        operation,
        worker: WorkerId(0),
    });
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(binding.value)
    );
}

#[test]
fn challenge_consumer_cannot_start_before_release() {
    let mut broken = one_worker_input();
    let binding = broken.transcript_outputs[0];
    let operation = OperationId(broken.operations.len() as u32);
    broken.operations.push(OperationDesc {
        id: operation,
        semantic: b"early-challenge-consumer".to_vec(),
        effect_identity: storage_alias::effect(91),
        interval: ExecutionInterval::BeforeBarrier(1),
        during: during(101, 102),
        reads: vec![ValueUse {
            value: binding.value,
            elements: binding.elements,
            layout: broken.values[binding.value.0 as usize].layout.clone(),
        }],
        writes: vec![],
    });
    broken.assignments.push(OperationAssignment {
        operation,
        worker: WorkerId(0),
    });
    assert_eq!(
        compile(broken).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(binding.value)
    );
}

#[test]
fn challenge_consumer_can_start_in_the_released_segment() {
    let mut input = one_worker_input();
    let binding = input.transcript_outputs[0];
    let operation = OperationId(input.operations.len() as u32);
    input.operations.push(OperationDesc {
        id: operation,
        semantic: b"released-challenge-consumer".to_vec(),
        effect_identity: storage_alias::effect(92),
        interval: ExecutionInterval::BeforeBarrier(2),
        during: during(201, 202),
        reads: vec![ValueUse {
            value: binding.value,
            elements: binding.elements,
            layout: input.values[binding.value.0 as usize].layout.clone(),
        }],
        writes: vec![],
    });
    input.assignments.push(OperationAssignment {
        operation,
        worker: WorkerId(0),
    });
    compile(input).unwrap();
}

#[test]
fn challenge_transfer_and_spill_cannot_start_before_release() {
    let mut transfer = two_worker_input();
    let binding = transfer.transcript_outputs[0];
    let replica = ReplicaId(transfer.replicas.len() as u32);
    let transition = LayoutTransitionId(transfer.transitions.len() as u32);
    let layout = transfer.values[binding.value.0 as usize].layout.clone();
    transfer.replicas.push(DeclaredReplica {
        id: replica,
        value: binding.value,
        elements: binding.elements,
        canonical_worker: WorkerId(0),
        worker: WorkerId(1),
        layout: layout.clone(),
        origin: ReplicaOrigin::Transition(transition),
        ready_at: ScheduleStep(102),
        live: during(101, 210),
    });
    transfer.transitions.push(LayoutTransition {
        id: transition,
        value: binding.value,
        elements: binding.elements,
        source_worker: WorkerId(0),
        destination_replica: replica,
        source_layout: layout.clone(),
        destination_layout: layout,
        axes: vec![AxisMap {
            source: 0,
            destination: 0,
        }],
        interval: ExecutionInterval::BeforeBarrier(1),
        during: during(101, 102),
        bytes: binding.elements.len() * 4,
        scratch_bytes: 0,
        scratch_worker: WorkerId(0),
        route: FleetLinkId(0),
    });
    assert_eq!(
        compile(transfer).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(binding.value)
    );

    let mut spilled = one_worker_input();
    let binding = spilled.transcript_outputs[0];
    spilled.spills = vec![spill(WorkerId(0), binding.value, 110)];
    assert_eq!(
        compile(spilled).unwrap_err(),
        FleetPlanError::TranscriptValueCausality(binding.value)
    );
}
