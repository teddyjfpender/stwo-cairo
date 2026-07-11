//! Gates for the GENERATED schedule table (design §16.3): structural validity,
//! real topological depth, and pins of the device-resident producer edges —
//! if regeneration ever changes these, the transformer metadata changed and the
//! gather kernels' addressing must be re-verified.

use std::collections::BTreeSet;

use stwo_cairo_gpu_prover::fixed_table_materializer::compile_cairo_fixed_table_materializations;
use stwo_cairo_gpu_prover::schedule::{
    ComponentRowSource, InputEdge, KernelIdentitySource, ScheduleError, TraceColumnCount,
    WitnessWriterKind, WitnessWriterReadiness,
};
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;

#[test]
fn generated_schedule_validates() {
    CAIRO_SCHEDULE.validate().unwrap();
    let levels = CAIRO_SCHEDULE.levels().unwrap();
    // Real depth: opcodes/builtins → their feed targets (memory tables,
    // verify_instruction, w18, blake_g, …). At least two levels, everything placed.
    assert!(levels.len() >= 2, "expected a multi-level DAG: {levels:?}");
    let placed: usize = levels.iter().map(|l| l.len()).sum();
    assert_eq!(placed, CAIRO_SCHEDULE.nodes.len());
}

#[test]
fn native_ec_op_is_weighted_before_its_direct_consumer() {
    let levels = CAIRO_SCHEDULE.levels().unwrap();
    let level_of = |component| {
        levels
            .iter()
            .position(|level| level.contains(&component))
            .unwrap()
    };
    let ec_level = level_of("ec_op_builtin");
    assert!(ec_level < level_of("partial_ec_mul_generic"));
    assert!(
        levels[ec_level].iter().any(|component| {
            CAIRO_SCHEDULE
                .nodes
                .iter()
                .find(|node| node.id == *component)
                .is_some_and(|node| {
                    node.id != "ec_op_builtin"
                        && node.facts.witness_writer.kind == WitnessWriterKind::RecordedAot
                })
        }),
        "EC-op has no prepared recorded same-level work to overlap"
    );
    for component in ["ec_op_builtin", "memory_address_to_id", "memory_id_to_big"] {
        let writer = CAIRO_SCHEDULE
            .nodes
            .iter()
            .find(|node| node.id == component)
            .unwrap()
            .facts
            .witness_writer;
        assert_eq!(writer.kind, WitnessWriterKind::NativeCuda, "{component}");
        assert_eq!(
            writer.readiness,
            WitnessWriterReadiness::CaptureSafe,
            "{component}"
        );
    }
}

#[test]
fn generated_schedule_covers_the_complete_claim_generator() {
    let shape = stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator::default()
        .proof_shape(None)
        .unwrap();
    let schedule_ids: BTreeSet<_> = CAIRO_SCHEDULE.nodes.iter().map(|node| node.id).collect();
    let shape_ids: BTreeSet<_> = shape
        .components()
        .iter()
        .map(|component| component.id)
        .collect();
    assert_eq!(schedule_ids, shape_ids);
    assert_eq!(schedule_ids.len(), 67);
}

#[test]
fn every_device_edge_fits_the_generated_sub_buffer() {
    for producer in CAIRO_SCHEDULE.nodes {
        for output in producer.outputs {
            let sub_words = producer
                .facts
                .sub_words
                .unwrap_or_else(|| panic!("{} has an edge but no sub width", producer.id));
            let end = u64::from(output.word_base)
                + u64::from(output.words_per_instance) * u64::from(output.n_instances);
            assert!(
                end <= u64::from(sub_words),
                "{} -> {} ends at {end}, sub width is {sub_words}",
                producer.id,
                output.to,
            );
        }
    }
}

fn producer_edge(consumer: &str, of: &str) -> (u32, u32, u32) {
    let node = CAIRO_SCHEDULE
        .nodes
        .iter()
        .find(|n| n.id == consumer)
        .unwrap_or_else(|| panic!("{consumer} missing from schedule"));
    node.inputs
        .iter()
        .find_map(|e| match e {
            InputEdge::Producer {
                of: p,
                word_base,
                words_per_instance,
                n_instances,
            } if *p == of => Some((*word_base, *words_per_instance, *n_instances)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{consumer} has no producer edge from {of}"))
}

fn ordered_producer_edges(consumer: &str) -> Vec<(&'static str, u32, u32, u32)> {
    CAIRO_SCHEDULE
        .nodes
        .iter()
        .find(|node| node.id == consumer)
        .unwrap_or_else(|| panic!("{consumer} missing from schedule"))
        .inputs
        .iter()
        .filter_map(|edge| match edge {
            InputEdge::Producer {
                of,
                word_base,
                words_per_instance,
                n_instances,
            } => Some((*of, *word_base, *words_per_instance, *n_instances)),
            InputEdge::ExecTables | InputEdge::DeviceTable(_) => None,
        })
        .collect()
}

/// These consumers are `Mutex<Vec<_>>`, not multisets. Arm S appends producer
/// outputs in aggregator, partial-round, full-round order, and the prepared
/// gather must retain that byte-significant row order (including its first-row
/// padding source).
#[test]
fn poseidon_vec_consumer_edges_match_legacy_append_order() {
    assert_eq!(
        ordered_producer_edges("cube_252"),
        [
            ("poseidon_aggregator", 282, 10, 2),
            ("poseidon_3_partial_rounds_chain", 1, 10, 3),
            ("poseidon_full_round_chain", 0, 10, 3),
        ],
    );
    assert_eq!(
        ordered_producer_edges("range_check_252_width_27"),
        [
            ("poseidon_aggregator", 262, 10, 2),
            ("poseidon_3_partial_rounds_chain", 61, 10, 3),
        ],
    );
    for consumer in ["cube_252", "range_check_252_width_27"] {
        let node = CAIRO_SCHEDULE
            .nodes
            .iter()
            .find(|node| node.id == consumer)
            .unwrap();
        assert_eq!(
            node.capacity_inputs
                .iter()
                .map(|feed| (feed.from, feed.n_instances))
                .collect::<Vec<_>>(),
            ordered_producer_edges(consumer)
                .into_iter()
                .map(|(producer, _, _, n_instances)| (producer, n_instances))
                .collect::<Vec<_>>(),
            "{consumer} capacity order drifted from materialization order",
        );
    }
}

/// Device-edge addressing is part of the prepared gather/compact ABI.  Pin the
/// hardware-certified legacy edges plus the Poseidon Graph-A chain so schedule
/// regeneration cannot silently retarget a captured graph.
#[test]
fn certified_edges_pinned() {
    assert_eq!(
        producer_edge(
            "partial_ec_mul_window_bits_18",
            "pedersen_aggregator_window_bits_18"
        ),
        (7, 72, 28),
        "aggregator→w18 edge changed"
    );
    assert_eq!(
        producer_edge("blake_g", "blake_round"),
        (81, 6, 8),
        "blake_round→blake_g edge changed"
    );
    assert_eq!(
        producer_edge("poseidon_aggregator", "poseidon_builtin"),
        (6, 6, 1),
        "poseidon_builtin→aggregator compact edge changed"
    );
    assert_eq!(
        producer_edge("poseidon_full_round_chain", "poseidon_aggregator"),
        (6, 32, 8),
        "poseidon_aggregator→full-round edge changed"
    );
    assert_eq!(
        producer_edge("poseidon_3_partial_rounds_chain", "poseidon_aggregator"),
        (342, 42, 27),
        "poseidon_aggregator→partial-round edge changed"
    );
}

/// M4 completeness fence for the fixed-table lane: the CUDA fixed-table
/// materializer unconditionally writes the word-major flattened LookupInputs
/// buffer for EVERY fixed table, and the arena planner sizes that buffer from
/// `facts.lookup_words` — so a missing or wrong fact only surfaces as a strict
/// resident `MissingWitnessBuffer` once the component is PRESENT (the SN2
/// fixture has no blake, so `verify_bitwise_xor_12`'s `lookup_words: None`
/// survived every fixture gate and failed on the real SN PIE). Pin the whole
/// fixed-table set against the compiled CUDA ABI on any host, both directions.
#[test]
fn fixed_table_lookup_facts_match_the_cuda_materializer_abi() {
    let compiled = compile_cairo_fixed_table_materializations().unwrap();
    let fixed_nodes: Vec<_> = CAIRO_SCHEDULE
        .nodes
        .iter()
        .filter(|node| node.facts.witness_writer.kind == WitnessWriterKind::FixedTableCuda)
        .collect();
    assert_eq!(
        fixed_nodes.len(),
        compiled.len(),
        "fixed-table schedule set drifted from the materialization descriptors"
    );
    for node in fixed_nodes {
        let materializer = compiled
            .iter()
            .find(|candidate| candidate.component() == node.id)
            .unwrap_or_else(|| panic!("{}: no compiled fixed-table materializer", node.id));
        let requirements = materializer.requirements();
        assert!(
            node.facts.lookup_words.is_some_and(|words| words > 0),
            "{}: every fixed table writes lookup words; a None here starves the \
             resident fixed-table workspace of its LookupInputs buffer",
            node.id
        );
        assert_eq!(
            node.facts.lookup_words,
            u32::try_from(requirements.lookup_output_count).ok(),
            "{}: schedule lookup_words must size exactly the materializer's \
             flattened LookupInputs output",
            node.id
        );
        let ComponentRowSource::FixedLogSize(log_size) = node.facts.row_source else {
            panic!("{}: fixed table without a FixedLogSize row source", node.id);
        };
        assert_eq!(
            requirements.row_count,
            1usize << log_size,
            "{}: materializer row count disagrees with the schedule log size",
            node.id
        );
        assert!(
            node.facts.sub_words.is_none(),
            "{}: the fixed-table writer has no sub-word output; a Some here would \
             allocate a SubcomponentInputs buffer no writer fills",
            node.id
        );
    }
}

#[test]
fn certified_edges_reach_the_runtime_artifact_plan() {
    let plan = CAIRO_SCHEDULE.artifact_plan().unwrap();
    let blake = plan.edge("blake_round", "blake_g").unwrap();
    assert_eq!(
        (blake.word_base, blake.words_per_instance, blake.n_instances,),
        (81, 6, 8)
    );
    let pedersen = plan
        .edge(
            "pedersen_aggregator_window_bits_18",
            "partial_ec_mul_window_bits_18",
        )
        .unwrap();
    assert_eq!(
        (
            pedersen.word_base,
            pedersen.words_per_instance,
            pedersen.n_instances,
        ),
        (7, 72, 28)
    );
}

#[test]
fn error_types_are_exercised_by_validate() {
    // Compile-time reminder that ScheduleError variants stay matched to validate();
    // the unit tests in schedule.rs own the negative cases.
    let _ = ScheduleError::Cycle("x");
}

/// M3 completeness fence: every lane recording label is a schedule node — a lane
/// added without schedule metadata (or a schedule regeneration that loses a lane
/// component) fails here, so the AOT kernel set and the DAG stay in lockstep.
#[test]
fn lane_recordings_are_schedule_nodes() {
    let recordings = stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings();
    assert!(
        recordings.len() >= 19,
        "lane registry shrank: {}",
        recordings.len()
    );
    for (label, program) in &recordings {
        let node = CAIRO_SCHEDULE
            .nodes
            .iter()
            .find(|node| node.id == *label)
            .unwrap_or_else(|| panic!("lane {label} missing from the generated schedule table"));
        assert_eq!(
            node.facts.kernel_identity,
            KernelIdentitySource::RecordedWitness
        );
        assert_eq!(
            node.facts.witness_writer.kind,
            WitnessWriterKind::RecordedAot
        );
        assert_eq!(
            node.facts.witness_writer.readiness,
            WitnessWriterReadiness::CaptureSafe
        );
        let TraceColumnCount::Fixed(trace_columns) = node.facts.trace_columns else {
            panic!("recorded lane {label} has a split trace")
        };
        assert_eq!(program.n_cols, trace_columns, "{label}: trace width");
        assert_eq!(
            Some(program.n_lookup_words),
            node.facts.lookup_words,
            "{label}: lookup width"
        );
        assert_eq!(
            Some(program.n_sub_words),
            node.facts.sub_words,
            "{label}: sub width"
        );
        assert!(program.n_cols > 0, "{label}: empty recording");
    }
    let labels = recordings
        .iter()
        .map(|(label, _)| *label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"blake_g"));
    assert!(labels.contains(&"qm_31_add_mul_opcode"));
    assert!(labels.contains(&"poseidon_builtin"));
    assert!(labels.contains(&"poseidon_aggregator"));
    assert!(labels.contains(&"poseidon_full_round_chain"));
    assert!(labels.contains(&"poseidon_3_partial_rounds_chain"));
}
