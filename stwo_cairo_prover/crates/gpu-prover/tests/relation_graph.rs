use stwo_cairo_gpu_prover::relation::{
    ChallengeEpoch, MultiplicitySource, RelationGraph, RelationPlanError, RelationTracePart,
    RelationUse, TupleSource, COMMON_LOOKUP_CHALLENGE_EPOCH,
};
use stwo_cairo_gpu_prover::relation_table::{CAIRO_RELATION_GRAPH, EXPECTED_HASH};
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;

#[test]
fn generated_relation_graph_covers_all_components() {
    let plan = CAIRO_RELATION_GRAPH.plan(&CAIRO_SCHEDULE).unwrap();
    assert_eq!(plan.components().len(), 67);
    assert!(CAIRO_RELATION_GRAPH.relations.len() > 32);

    let mut uses = 0;
    for component in plan.components() {
        for trace in component.traces {
            for (output_column, column) in trace.columns.iter().enumerate() {
                assert_eq!(column.output_column as usize, output_column);
                for relation_use in column.uses {
                    assert_eq!(relation_use.challenge_epoch, COMMON_LOOKUP_CHALLENGE_EPOCH);
                    uses += 1;
                }
            }
        }
    }
    assert_eq!(uses, 1566);
}

#[test]
fn computed_relation_components_are_explicit() {
    let plan = CAIRO_RELATION_GRAPH.plan(&CAIRO_SCHEDULE).unwrap();

    let memory_address = plan.component("memory_address_to_id").unwrap();
    assert_eq!(memory_address.traces.len(), 1);
    assert_eq!(memory_address.traces[0].output_columns, 8);
    assert!(matches!(
        memory_address.traces[0].columns[0].uses[0]
            .denominator
            .tuple
            .source,
        TupleSource::MemoryAddressChunk { chunk: 0 }
    ));
    assert!(matches!(
        memory_address.traces[0].columns[0].uses[0]
            .multiplicity
            .source,
        MultiplicitySource::MemoryAddressChunk { chunk: 0 }
    ));

    let memory_value = plan.component("memory_id_to_big").unwrap();
    assert_eq!(memory_value.traces.len(), 2);
    assert_eq!(
        memory_value.traces[0].part,
        RelationTracePart::EachMemoryBig
    );
    assert_eq!(memory_value.traces[0].output_columns, 8);
    assert_eq!(memory_value.traces[1].part, RelationTracePart::MemorySmall);
    assert_eq!(memory_value.traces[1].output_columns, 3);

    let xor = plan.component("verify_bitwise_xor_12").unwrap();
    assert_eq!(xor.traces[0].output_columns, 8);
    assert!(matches!(
        xor.traces[0].columns[7].uses[1].denominator.tuple.source,
        TupleSource::BitwiseXor12 {
            multiplicity_column: 15
        }
    ));
}

#[test]
fn relation_graph_hash_is_stable() {
    let plan = CAIRO_RELATION_GRAPH.plan(&CAIRO_SCHEDULE).unwrap();
    assert_eq!(EXPECTED_HASH, 0x7396_3831_c53d_f4a2);
    assert_eq!(plan.relation_graph_hash(), EXPECTED_HASH);
}

fn graph_with_first_use(mut edit: impl FnMut(&mut RelationUse)) -> RelationGraph {
    let mut components = CAIRO_RELATION_GRAPH.components.to_vec();
    let mut traces = components[0].traces.to_vec();
    let mut columns = traces[0].columns.to_vec();
    let mut uses = columns[0].uses.to_vec();
    edit(&mut uses[0]);
    columns[0].uses = Box::leak(uses.into_boxed_slice());
    traces[0].columns = Box::leak(columns.into_boxed_slice());
    components[0].traces = Box::leak(traces.into_boxed_slice());
    RelationGraph {
        relations: CAIRO_RELATION_GRAPH.relations,
        components: Box::leak(components.into_boxed_slice()),
        expected_hash: CAIRO_RELATION_GRAPH.expected_hash,
    }
}

#[test]
fn unknown_relation_and_wrong_epoch_fail_closed() {
    let graph = RelationGraph {
        relations: &[],
        components: CAIRO_RELATION_GRAPH.components,
        expected_hash: CAIRO_RELATION_GRAPH.expected_hash,
    };
    assert!(matches!(
        graph.validate(&CAIRO_SCHEDULE),
        Err(RelationPlanError::UnknownRelation(_))
    ));

    let graph = graph_with_first_use(|relation_use| {
        relation_use.challenge_epoch = ChallengeEpoch::BeforeBaseCommitment;
    });
    assert!(matches!(
        graph.validate(&CAIRO_SCHEDULE),
        Err(RelationPlanError::WrongChallengeEpoch(_))
    ));
}

#[test]
fn output_mapping_and_hash_fail_closed() {
    let mut components = CAIRO_RELATION_GRAPH.components.to_vec();
    let mut traces = components[0].traces.to_vec();
    let mut columns = traces[0].columns.to_vec();
    columns[0].output_column = 1;
    traces[0].columns = Box::leak(columns.into_boxed_slice());
    components[0].traces = Box::leak(traces.into_boxed_slice());
    let graph = RelationGraph {
        relations: CAIRO_RELATION_GRAPH.relations,
        components: Box::leak(components.into_boxed_slice()),
        expected_hash: CAIRO_RELATION_GRAPH.expected_hash,
    };
    assert!(matches!(
        graph.validate(&CAIRO_SCHEDULE),
        Err(RelationPlanError::OutputColumnsNotContiguous(_))
    ));

    let graph = RelationGraph {
        relations: CAIRO_RELATION_GRAPH.relations,
        components: CAIRO_RELATION_GRAPH.components,
        expected_hash: EXPECTED_HASH ^ 1,
    };
    assert!(matches!(
        graph.validate(&CAIRO_SCHEDULE),
        Err(RelationPlanError::HashMismatch { .. })
    ));
}
