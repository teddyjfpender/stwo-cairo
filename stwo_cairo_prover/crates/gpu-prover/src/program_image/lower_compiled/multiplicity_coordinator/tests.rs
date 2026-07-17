use super::*;

#[test]
fn generated_sn2_aligns_23_schedule_nodes_with_22_generic_feeds() {
    let executable = super::super::tests::generated_sn2();
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    let scheduled = exact_schedule(executable.arena(), &schedule).unwrap();
    let (feeds, lut_words) = exact_feed_alignment(executable.arena(), &scheduled).unwrap();

    assert_eq!(scheduled.len(), 23);
    assert_eq!(feeds.len(), 22);
    assert!(!lut_words.is_empty());
    assert_eq!(
        scheduled[7],
        (
            ProducerSchedulePosition {
                level: 0,
                lane: 7,
                ordinal: 7,
            },
            WitnessProducer {
                component: "ec_op_builtin",
                part: None,
                kind: WitnessProducerKind::NativeEcOp,
            },
        )
    );
    for (ordinal, (_, producer)) in scheduled.iter().enumerate() {
        let feed = feeds.get(producer.component);
        if ordinal == 7 {
            assert!(feed.is_none());
        } else {
            assert_eq!(feed.map(|feed| feed.kind), Some(FeedPlanKind::Generic));
        }
    }
}

#[test]
fn generated_sn2_feed_alignment_is_schedule_order_not_arena_order() {
    let executable = super::super::tests::generated_sn2();
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    let scheduled = exact_schedule(executable.arena(), &schedule).unwrap();
    let (feeds, _) = exact_feed_alignment(executable.arena(), &scheduled).unwrap();

    let scheduled_names = scheduled
        .iter()
        .filter(|(_, producer)| producer.kind == WitnessProducerKind::Recorded)
        .map(|(_, producer)| producer.component)
        .collect::<Vec<_>>();
    let feed_names = scheduled_names
        .iter()
        .map(|name| {
            let reference = feeds.get(name).unwrap();
            let PlannedRecordedMultiplicityFeedGraph::Generic(feed) =
                &executable.arena().multiplicity().unwrap().feeds[reference.index]
            else {
                panic!("generated SN2 recorded writer must own a generic feed");
            };
            feed.plan.producer
        })
        .collect::<Vec<_>>();
    assert_eq!(feed_names, scheduled_names);
}

#[test]
fn generated_sn2_lowers_all_feeds_and_accepts_recorded_zero_lut_feeds() {
    let executable = super::super::tests::generated_sn2_replacement();
    let authority = executable.replacement_base_producers().unwrap();
    let feeds = authority
        .multiplicity
        .after_producer
        .iter()
        .flatten()
        .collect::<Vec<_>>();

    assert_eq!(authority.multiplicity.after_producer.len(), 23);
    assert_eq!(feeds.len(), 22);
    assert!(authority.multiplicity.after_producer[7].is_none());
    let SemanticBaseProducer::NativeEcOp { contract, .. } = &authority.producers[7] else {
        panic!("schedule ordinal 7 must be the native EC producer");
    };
    assert_eq!(ec_op_transitions(contract).unwrap().len(), 4);
    assert_eq!(
        feeds
            .iter()
            .map(|feed| feed.destinations.len())
            .sum::<usize>(),
        63
    );

    let add_ap = authority.multiplicity.after_producer[0].as_ref().unwrap();
    assert_eq!(
        add_ap.owner,
        multiplicity_feed::MultiplicityFeedOwner::Recorded {
            component: "add_ap_opcode",
            part: TracePartId::Main,
        }
    );
    assert!(add_ap.luts.is_empty());
    multiplicity_feed::validate_lowered(add_ap).unwrap();
}
