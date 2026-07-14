use super::*;

fn bf(value: u32) -> BaseField {
    BaseField::from_u32_unchecked(value)
}

fn probe(source: SegmentStartSource, original: u32, first: u32, second: u32) -> SegmentProbe {
    SegmentProbe {
        source,
        original: bf(original),
        first: bf(first),
        second: bf(second),
    }
}

#[test]
fn equal_real_starts_and_permuted_probes_keep_distinct_sources() {
    let probes = [
        probe(SegmentStartSource::Bitwise, 7, 101, 202),
        probe(SegmentStartSource::EcOp, 7, 102, 201),
    ];
    assert_eq!(
        classify_base_params(
            "test",
            0,
            &[bf(7), bf(7), bf(13)],
            &[bf(101), bf(102), bf(13)],
            &[bf(202), bf(201), bf(13)],
            &probes,
        )
        .unwrap(),
        vec![
            BaseParamSource::SegmentStart(SegmentStartSource::Bitwise),
            BaseParamSource::SegmentStart(SegmentStartSource::EcOp),
            BaseParamSource::Constant(bf(13)),
        ]
    );
}

#[test]
fn constant_equal_to_one_probe_value_cannot_alias_the_tagged_source() {
    let probes = [probe(SegmentStartSource::Bitwise, 7, 101, 202)];
    assert_eq!(
        classify_base_params(
            "test",
            0,
            &[bf(101), bf(7)],
            &[bf(101), bf(101)],
            &[bf(101), bf(202)],
            &probes,
        )
        .unwrap(),
        vec![
            BaseParamSource::Constant(bf(101)),
            BaseParamSource::SegmentStart(SegmentStartSource::Bitwise),
        ]
    );
}

#[test]
fn unknown_dynamic_or_affine_slot_fails_closed() {
    let probes = [probe(SegmentStartSource::Bitwise, 7, 101, 202)];
    assert!(matches!(
        classify_base_params("test", 3, &[bf(8)], &[bf(102)], &[bf(203)], &probes,),
        Err(CompositionPlanError::UnclassifiedBaseParam {
            component: "test",
            instance: 3,
            slot: 0,
            ..
        })
    ));
}

#[test]
fn truncated_or_extra_probe_words_fail_before_classification() {
    let probes = [probe(SegmentStartSource::Bitwise, 7, 101, 202)];
    for (first, second) in [
        (vec![bf(101)], vec![bf(202), bf(11)]),
        (vec![bf(101), bf(11), bf(13)], vec![bf(202), bf(11)]),
    ] {
        assert!(matches!(
            classify_base_params("test", 0, &[bf(7), bf(11)], &first, &second, &probes,),
            Err(CompositionPlanError::BaseParamProbeCountMismatch {
                component: "test",
                instance: 0,
                expected: 2,
                ..
            })
        ));
    }
}

#[test]
fn duplicate_probe_identity_is_rejected_even_when_a_slot_matches() {
    let probes = [
        probe(SegmentStartSource::Bitwise, 7, 101, 202),
        probe(SegmentStartSource::EcOp, 7, 101, 202),
    ];
    assert!(matches!(
        classify_base_params("test", 0, &[bf(7)], &[bf(101)], &[bf(202)], &probes,),
        Err(CompositionPlanError::AmbiguousBaseParamProbe {
            component: "test",
            instance: 0,
        })
    ));
}
