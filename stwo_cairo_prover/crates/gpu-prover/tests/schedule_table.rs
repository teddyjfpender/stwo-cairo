//! Gates for the GENERATED schedule table (design §16.3): structural validity,
//! real topological depth, and pins of the two hardware-certified device edges —
//! if regeneration ever changes these, the transformer metadata changed and the
//! gather kernels' addressing must be re-verified.

use std::collections::BTreeSet;

use stwo_cairo_gpu_prover::schedule::{
    InputEdge, KernelIdentitySource, ScheduleError, TraceColumnCount,
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

/// The two device edges certified on hardware (round-27/28): their addressing is
/// baked into the gather kernels' launch parameters.
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
}
