use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{MemoryBaseTraceAbi, MemoryBaseTraceStepKind};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::*;
use crate::compiled_proof::{
    AotArgumentValue, AtomicOperation, EffectAccess, EffectBindingId, ElementRange,
    InPlaceAliasRequirement, InPlaceDiscipline,
};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredMemoryBaseTrace,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let mut before =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
                .unwrap();
        BaseProducerAuthority::compile_replacement_into(
            executable.arena(),
            PreProcessedTraceVariant::Canonical,
            &mut before,
        )
        .unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(executable.arena(), &mut after)
            .unwrap()
            .expect("SN2 replacement must contain memory Base traces");
        Fixture {
            executable,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn memory_trace_is_the_exact_post_witness_ordered_graph() {
    let fixture = fixture();
    let memory = fixture
        .executable
        .arena()
        .multiplicity()
        .unwrap()
        .memory_traces
        .as_ref()
        .unwrap();
    let steps = fixture.lowered.steps();
    assert_eq!(steps.len(), 3 + 2 * memory.big_parts.len());
    assert_eq!(steps[0].kind(), MemoryBaseTraceStepKind::Address);
    assert_eq!(steps[0].part(), TracePartId::Main);
    for (ordinal, pair) in steps[1..1 + 2 * memory.big_parts.len()]
        .chunks_exact(2)
        .enumerate()
    {
        assert_eq!(pair[0].kind(), MemoryBaseTraceStepKind::BigValue);
        assert_eq!(pair[1].kind(), MemoryBaseTraceStepKind::BigRc99);
        assert_eq!(pair[0].part(), TracePartId::MemoryBig(ordinal as u32));
        assert_eq!(pair[1].part(), pair[0].part());
    }
    let tail = &steps[steps.len() - 2..];
    assert_eq!(tail[0].kind(), MemoryBaseTraceStepKind::SmallValue);
    assert_eq!(tail[1].kind(), MemoryBaseTraceStepKind::SmallRc99);
    assert_eq!(tail[0].part(), TracePartId::MemorySmall);
    assert_eq!(tail[1].part(), TracePartId::MemorySmall);
    assert_eq!(
        fixture.lowered.contract().steps().len(),
        fixture.lowered.steps().len()
    );
    validate_from(
        fixture.executable.arena(),
        &fixture.before,
        &fixture.after,
        &Some(fixture.lowered.clone()),
    )
    .unwrap();
}

#[test]
fn every_real_wrapper_owns_exact_inputs_outputs_and_scalar_abi() {
    let fixture = fixture();
    for step in fixture.lowered.steps() {
        let arguments = &step.invocation().arguments;
        assert!(arguments
            .iter()
            .enumerate()
            .all(|(ordinal, argument)| argument.ordinal as usize == ordinal));
        let access_bindings = step
            .effect()
            .accesses()
            .iter()
            .flat_map(|access| [access.source(), access.destination()])
            .flatten()
            .map(|value| value.binding)
            .collect::<std::collections::BTreeSet<_>>();
        let invocation_bindings = arguments
            .iter()
            .flat_map(|argument| match &argument.value {
                AotArgumentValue::DevicePointer(Some(binding)) => vec![*binding],
                AotArgumentValue::DevicePointerTable(bindings) => {
                    bindings.iter().flatten().copied().collect()
                }
                _ => Vec::new(),
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(invocation_bindings, access_bindings);
        match step.kind() {
            MemoryBaseTraceStepKind::Address => {
                assert_eq!(arguments.len(), 6);
                assert_eq!(
                    fixture.lowered.contract().steps()[step.contract_ordinal as usize].abi(),
                    MemoryBaseTraceAbi::AddressSlicedV2
                );
                assert_eq!(step.reads.len(), 2);
                assert_eq!(step.writes.len(), 32);
                assert!(step.atomic.is_none());
            }
            MemoryBaseTraceStepKind::BigValue => {
                assert_eq!(arguments.len(), 7);
                let contract = &fixture.lowered.contract().steps()[step.contract_ordinal as usize];
                assert_eq!(contract.abi(), MemoryBaseTraceAbi::ValueSlicedV2);
                assert_eq!(
                    step.reads.len(),
                    if contract.source_words() == 0 { 1 } else { 29 }
                );
                assert_eq!(step.writes.len(), 29);
                assert!(step.atomic.is_none());
            }
            MemoryBaseTraceStepKind::SmallValue => {
                assert_eq!(arguments.len(), 7);
                let contract = &fixture.lowered.contract().steps()[step.contract_ordinal as usize];
                assert_eq!(contract.abi(), MemoryBaseTraceAbi::ValueSlicedV2);
                assert_eq!(
                    step.reads.len(),
                    if contract.source_words() == 0 { 1 } else { 9 }
                );
                assert_eq!(step.writes.len(), 9);
                assert!(step.atomic.is_none());
            }
            MemoryBaseTraceStepKind::BigRc99 | MemoryBaseTraceStepKind::SmallRc99 => {
                assert_eq!(arguments.len(), 6);
                assert!(step.writes.is_empty());
                let EffectAccess::Atomic {
                    operation,
                    in_place,
                    ..
                } = step.effect().accesses().last().unwrap()
                else {
                    panic!("rc9_9 step must end in one atomic transition")
                };
                assert_eq!(*operation, AtomicOperation::AddU32);
                assert_eq!(in_place.requirement, InPlaceAliasRequirement::Required);
                assert_eq!(
                    in_place.discipline,
                    InPlaceDiscipline::ElementWiseReadBeforeWrite
                );
            }
        }
    }
}

#[test]
fn sliced_v2_invocations_use_effect_start_pointers_without_absolute_offsets() {
    let fixture = fixture();
    let address = &fixture.lowered.steps()[0];
    let address_contract = &fixture.lowered.contract().steps()[address.contract_ordinal as usize];
    assert_eq!(address_contract.source_offset(), 1);
    assert_eq!(address.reads[0].elements.start, 1);
    assert_eq!(
        address.invocation.arguments[1].value,
        AotArgumentValue::U32(address.reads[0].elements.len() as u32)
    );

    for step in fixture.lowered.steps().iter().filter(|step| {
        matches!(
            step.kind(),
            MemoryBaseTraceStepKind::BigValue | MemoryBaseTraceStepKind::SmallValue
        )
    }) {
        let contract = &fixture.lowered.contract().steps()[step.contract_ordinal as usize];
        let source_entries = match &step.invocation.arguments[0].value {
            AotArgumentValue::DevicePointerTable(entries) => entries,
            other => panic!("unexpected source table argument: {other:?}"),
        };
        assert_eq!(source_entries.len(), contract.limb_or_pair_count() as usize);
        let source_words = match &step.invocation.arguments[2].value {
            AotArgumentValue::U32(words) => *words,
            other => panic!("unexpected source word argument: {other:?}"),
        };
        assert_eq!(source_words, contract.source_words());
        assert_eq!(
            source_entries.iter().all(Option::is_some),
            source_words != 0
        );
        let count_index = contract
            .reads()
            .iter()
            .position(|read| read.role == MemoryBaseTraceEffectRole::ValueMultiplicity)
            .unwrap();
        let counts = &step.reads[count_index];
        assert_eq!(
            step.invocation.arguments[3].value,
            AotArgumentValue::DevicePointer(Some(counts.binding))
        );
        assert_eq!(
            step.invocation.arguments[4].value,
            AotArgumentValue::U32(contract.row_count())
        );
        assert_eq!(
            step.invocation.arguments[5].value,
            AotArgumentValue::U32(contract.row_count())
        );
    }
}

#[test]
fn zero_read_value_tail_uses_an_all_null_source_table_and_keeps_counts_exact() {
    let fixture = fixture();
    let template = fixture
        .lowered
        .steps()
        .iter()
        .find(|step| step.kind() == MemoryBaseTraceStepKind::BigValue)
        .unwrap();
    let mut requirements = fixture.lowered.contract().requirements().clone();
    let source_offset = requirements
        .big_count_words
        .max(requirements.big_source_words)
        + 8;
    let tail_ordinal = {
        let tail = requirements.big_parts.last_mut().unwrap();
        tail.source_offset = source_offset;
        tail.row_count = 16;
        tail.part_ordinal
    };
    requirements.big_count_words = source_offset + 16;
    let contract = MemoryBaseTraceContract::compile(&requirements).unwrap();
    let (contract_ordinal, zero_tail) = contract
        .steps()
        .iter()
        .enumerate()
        .find(|(_, step)| {
            step.kind() == MemoryBaseTraceStepKind::BigValue
                && step.part_ordinal() == Some(tail_ordinal)
        })
        .unwrap();
    assert_eq!(zero_tail.source_words(), 0);
    assert_eq!(zero_tail.reads().len(), 1);

    let mut count = template.reads.last().unwrap().clone();
    count.binding = EffectBindingId(0);
    count.elements = range(zero_tail.reads()[0]);
    let writes = template
        .writes
        .iter()
        .cloned()
        .zip(zero_tail.writes())
        .enumerate()
        .map(|(ordinal, (mut binding, effect))| {
            binding.binding = EffectBindingId(ordinal as u32 + 1);
            binding.elements = range(*effect);
            binding
        })
        .collect();
    let lowered = semantic::compile(
        contract_ordinal,
        zero_tail,
        TracePartId::MemoryBig(tail_ordinal),
        vec![count.clone()],
        writes,
        None,
    )
    .unwrap();
    assert_eq!(
        lowered.invocation.arguments[0].value,
        AotArgumentValue::DevicePointerTable(vec![None; 28])
    );
    assert_eq!(
        lowered.invocation.arguments[2].value,
        AotArgumentValue::U32(0)
    );
    assert_eq!(
        lowered.invocation.arguments[3].value,
        AotArgumentValue::DevicePointer(Some(count.binding))
    );
    assert_eq!(
        lowered.invocation.arguments[4].value,
        AotArgumentValue::U32(16)
    );
}

fn range(access: stwo_backend_cuda::MemoryBaseTraceEffectAccess) -> ElementRange {
    ElementRange::new(access.start_words, access.start_words + access.len_words).unwrap()
}

#[test]
fn rc99_lineage_is_strictly_serial_and_small_touches_only_four_relations() {
    let fixture = fixture();
    let rc = fixture
        .lowered
        .steps()
        .iter()
        .filter_map(|step| step.atomic.as_ref())
        .collect::<Vec<_>>();
    assert!(!rc.is_empty());
    for pair in rc.windows(2) {
        assert_eq!(pair[0].destination, pair[1].source);
        assert_eq!(pair[0].value, pair[1].value);
        assert_eq!(pair[0].arena, pair[1].arena);
    }
    let final_transition = rc.last().unwrap();
    assert_eq!(
        fixture.after.version(final_transition.value).unwrap(),
        final_transition.destination
    );
    assert_eq!(final_transition.elements.start, 0);
    assert_eq!(
        final_transition.elements.len(),
        fixture.lowered.contract().requirements().rc99_lut_words * 4
    );
    for big in &rc[..rc.len() - 1] {
        assert_eq!(
            big.elements.len(),
            fixture.lowered.contract().requirements().rc99_count_words
        );
    }
}

#[test]
fn rc99_rebinds_value_outputs_into_its_local_abi_namespace() {
    let fixture = fixture();
    for pair in fixture.lowered.steps()[1..].chunks_exact(2) {
        let value = &pair[0];
        let rc99 = &pair[1];
        let contract = &fixture.lowered.contract().steps()[rc99.contract_ordinal as usize];
        let carried = contract.limb_or_pair_count() as usize * 2;
        assert_eq!(rc99.reads.len(), carried + 1);
        for (ordinal, (written, read)) in value.writes[..carried]
            .iter()
            .zip(&rc99.reads[..carried])
            .enumerate()
        {
            assert_eq!(read.binding, EffectBindingId(ordinal as u32));
            assert_eq!(
                (read.arena, read.value, read.elements, read.version),
                (
                    written.arena,
                    written.value,
                    written.elements,
                    written.version
                )
            );
        }
        assert_eq!(
            rc99.reads.last().unwrap().binding,
            EffectBindingId(carried as u32)
        );
    }
}

#[test]
fn duplicate_or_mutated_stage_fails_without_advancing_the_allocator() {
    let fixture = fixture();
    let mut repeated = fixture.after.clone();
    let snapshot = repeated.clone();
    assert_eq!(
        lower_stage(fixture.executable.arena(), &mut repeated),
        Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    );
    assert_eq!(repeated, snapshot);

    let mut changed = fixture.lowered.clone();
    changed.steps[0].writes.swap(0, 1);
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            &fixture.before,
            &fixture.after,
            &Some(changed)
        ),
        Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    );

    let mut changed = fixture.lowered.clone();
    changed
        .steps
        .last_mut()
        .unwrap()
        .atomic
        .as_mut()
        .unwrap()
        .source = ValueVersion(u32::MAX);
    assert!(resolve_static_wrapper(
        StaticCudaWrapperId(1),
        89,
        &changed,
        changed.steps.len() - 1
    )
    .is_err());

    let mut changed = fixture.lowered.clone();
    let rc99_ordinal = changed
        .steps
        .iter()
        .position(|step| step.kind() == MemoryBaseTraceStepKind::BigRc99)
        .unwrap();
    changed.steps[rc99_ordinal].reads[0].binding = EffectBindingId(1);
    assert!(resolve_static_wrapper(StaticCudaWrapperId(2), 89, &changed, rc99_ordinal).is_err());
}

#[test]
fn linked_projection_is_one_real_wrapper_and_never_a_synthetic_graph() {
    let fixture = fixture();
    for (ordinal, step) in fixture.lowered.steps().iter().enumerate() {
        let Some(wrapper) = resolve_static_wrapper(
            StaticCudaWrapperId(ordinal as u32 + 1),
            89,
            &fixture.lowered,
            ordinal,
        )
        .unwrap() else {
            return;
        };
        assert_eq!(wrapper.accepted_effect(), step.effect().id());
        assert_eq!(wrapper.kernel_launches().count(), 1);
        assert_eq!(
            wrapper.wrapper_symbol(),
            fixture.lowered.contract().steps()[ordinal]
                .abi()
                .entry_symbol()
                .as_bytes()
        );
    }
}
