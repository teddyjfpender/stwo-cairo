//! Gates for the GENERATED schedule table (design §16.3): structural validity,
//! real topological depth, and pins of the two hardware-certified device edges —
//! if regeneration ever changes these, the transformer metadata changed and the
//! gather kernels' addressing must be re-verified.

use stwo_cairo_gpu_prover::schedule::{InputEdge, ScheduleError};
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
fn error_types_are_exercised_by_validate() {
    // Compile-time reminder that ScheduleError variants stay matched to validate();
    // the unit tests in schedule.rs own the negative cases.
    let _ = ScheduleError::Cycle("x");
}
