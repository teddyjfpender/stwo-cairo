use super::*;

fn input_with_partial_transition(
    source_elements: ElementRange,
    destination_elements: ElementRange,
    atomic: bool,
    discipline: InPlaceDiscipline,
) -> CompiledProofInput {
    let mut input = valid_input();
    let destination = input.output.sections[0].value;
    let words = input.values[destination.0 as usize]
        .layout
        .element_count()
        .unwrap();
    let source = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        source,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(90_000)),
        Region::Input,
    ));
    let mut accesses = input.effects[0].accesses().to_vec();
    let EffectAccess::Write { destination: out } = accesses.pop().unwrap() else {
        panic!("fixture must end in the proof output write")
    };
    assert_eq!(out.value.version, destination);
    let binding = out.binding.0;
    let source = bound(
        binding,
        ValueRange {
            version: source,
            elements: source_elements,
        },
    );
    let destination = bound(
        binding,
        ValueRange {
            version: destination,
            elements: destination_elements,
        },
    );
    let alias = InPlaceAliasAuthority {
        id: InPlaceAliasId(0),
        requirement: InPlaceAliasRequirement::Required,
        discipline,
    };
    accesses.push(if atomic {
        EffectAccess::Atomic {
            source,
            destination,
            operation: AtomicOperation::AddU32,
            in_place: alias,
        }
    } else {
        EffectAccess::ReadWrite {
            source,
            destination: BoundValueRange {
                binding: EffectBindingId(binding + 1),
                ..destination
            },
            in_place: Some(alias),
        }
    });
    install_effect(&mut input, EffectContract::new(accesses, vec![]).unwrap());
    input
}

#[test]
fn exact_atomic_prefix_carries_the_unwritten_suffix() {
    let input = valid_input();
    let words = input.output.layout.total_words;
    let carried = input_with_partial_transition(
        range(0, words / 2),
        range(0, words / 2),
        true,
        InPlaceDiscipline::ElementWiseReadBeforeWrite,
    );
    CompiledProof::compile(carried, transcript()).unwrap();
}

#[test]
fn partial_read_write_and_non_prefix_atomics_remain_rejected() {
    let input = valid_input();
    let words = input.output.layout.total_words;
    let cases = [
        input_with_partial_transition(
            range(0, words / 2),
            range(0, words / 2),
            false,
            InPlaceDiscipline::ElementWiseReadBeforeWrite,
        ),
        input_with_partial_transition(
            range(1, words / 2 + 1),
            range(1, words / 2 + 1),
            true,
            InPlaceDiscipline::ElementWiseReadBeforeWrite,
        ),
        input_with_partial_transition(
            range(1, words / 2 + 1),
            range(0, words / 2),
            true,
            InPlaceDiscipline::ElementWiseReadBeforeWrite,
        ),
        input_with_partial_transition(
            range(0, words / 2),
            range(0, words / 2),
            true,
            InPlaceDiscipline::BlockBarrierPhases,
        ),
    ];
    for invalid in cases {
        assert!(matches!(
            CompiledProof::compile(invalid, transcript()),
            Err(CompiledProofError::IncompleteWrite { .. })
        ));
    }
}
