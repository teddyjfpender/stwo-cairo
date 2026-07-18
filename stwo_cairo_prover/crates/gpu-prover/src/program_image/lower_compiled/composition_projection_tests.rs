use super::*;
use crate::compiled_proof::PartitionAuthorityKind;

fn fixture() -> (
    std::sync::Arc<crate::shape_executable::ShapeExecutable>,
    adapter::SemanticValueMap,
    adapter::SemanticValueMap,
    LoweredCompositionWaves,
) {
    let executable = super::super::tests::generated_sn2_replacement();
    let mut values = adapter::SemanticValueMap::allocate_ordered(
        wave_input_catalogs(executable.arena()).unwrap(),
    )
    .unwrap();
    let before = values.clone();
    let lowered = lower_waves(executable.arena(), &mut values).unwrap();
    (executable, before, values, lowered)
}

#[test]
fn generated_sn2_projects_all_waves_into_exact_row_partitions() {
    let (executable, before, after, lowered) = fixture();
    assert_eq!(lowered.waves().len(), 14);
    assert_eq!(
        lowered
            .waves()
            .iter()
            .map(LoweredCompositionWave::operation_ordinal)
            .collect::<Vec<_>>(),
        (2..16).collect::<Vec<_>>()
    );
    assert_ne!(lowered.digest(), [0; 32]);
    assert!(validate_from(executable.arena(), &before, &after, &lowered).is_ok());

    for (index, wave) in lowered.waves().iter().enumerate() {
        let expected = &executable.arena().composition().requirements.waves[index];
        assert_eq!(wave.shard().wave_index(), index);
        assert_eq!(wave.shard().effect(), wave.effect().id());
        assert_eq!(wave.shard().full_rows(), expected.row_count);
        let PartitionAuthorityKind::Exact(partition) = wave.shard().partition().kind() else {
            panic!("composition wave must retain exact row authority");
        };
        assert_eq!(partition.domain().start, 0);
        assert_eq!(partition.domain().end, expected.row_count);
        assert_eq!(wave.invocation().arguments.len(), 9);
        assert_eq!(
            wave.invocation().arguments[7].value,
            AotArgumentValue::U32(0)
        );
        assert_eq!(
            wave.invocation().arguments[8].value,
            AotArgumentValue::U32(expected.row_count as u32)
        );
        assert_eq!(
            wave.effect()
                .accesses()
                .iter()
                .filter(|access| access.destination().is_some())
                .count(),
            4
        );
        assert!(wave
            .bindings()
            .iter()
            .filter(|binding| binding.kind == CompositionAccessKind::Write)
            .all(|binding| binding.elements
                == ElementRange {
                    start: 0,
                    end: expected.row_count,
                }));
        assert_eq!(
            wave.launch().grid[0],
            (expected.row_count as u32).div_ceil(128)
        );
    }
}

#[test]
fn pointer_graph_preserves_record_and_field_shape_without_duplicate_bindings() {
    let (executable, _, _, lowered) = fixture();
    for (wave, requirement) in lowered
        .waves()
        .iter()
        .zip(&executable.arena().composition().requirements.waves)
    {
        let AotArgumentValue::DeviceRecordPointerGraphValue { root, records } =
            &wave.invocation().arguments[0].value
        else {
            panic!("wave parts must be one exact record pointer graph");
        };
        assert_eq!(records.len(), requirement.parts.len());
        assert!(records.iter().all(|record| record.fields.len() == 5));
        let AotArgumentValue::DevicePointerRangeSetValue { ranges } =
            &wave.invocation().arguments[1].value
        else {
            panic!("random powers must preserve exact reached ranges");
        };
        assert!(!ranges.is_empty());

        let mut bindings = BTreeSet::from([*root]);
        for record in records {
            for field in &record.fields {
                for binding in field.entries.iter().flatten() {
                    assert!(bindings.insert(*binding));
                }
            }
        }
        for binding in ranges {
            assert!(bindings.insert(*binding));
        }
        for argument in &wave.invocation().arguments[2..6] {
            let AotArgumentValue::DevicePointer(Some(binding)) = argument.value else {
                panic!("coordinate must be a direct pointer");
            };
            assert!(bindings.insert(binding));
        }
        assert_eq!(bindings.len(), wave.effect().accesses().len());
    }
}

#[test]
fn projection_is_transactional_and_receipt_tamper_fails_closed() {
    let (executable, before, after, lowered) = fixture();
    let mut missing =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = missing.clone();
    assert!(lower_waves(executable.arena(), &mut missing).is_err());
    assert_eq!(missing, unchanged);

    let mut forged = lowered.clone();
    forged.digest[0] ^= 1;
    assert_eq!(
        validate_from(executable.arena(), &before, &after, &forged),
        Err(InvocationShapeError::InvalidCompositionBinding)
    );
}
