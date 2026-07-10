//! Host-side fixture-profile gate for the strict resident whole-proof tests.
//!
//! Runs on any machine (no CUDA): proves the checked-in fixtures actually
//! present the component families the hardware gates claim to exercise, and
//! none of the writers the strict resident path cannot admit. This keeps the
//! hardware gate from silently narrowing when a fixture or the schedule
//! changes.

use cairo_vm::types::layout_name::LayoutName;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::phases;
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::schedule::WitnessWriterKind;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;

struct FixtureProfile {
    present: Vec<&'static str>,
    unsupported: Vec<&'static str>,
    recorded: Vec<&'static str>,
}

fn fixture_profile(fixture: &str, variant: PreProcessedTraceVariant) -> FixtureProfile {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path(fixture),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let ingest = phases::ingest::run(input, variant, None);
    let exact = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    let present: Vec<&'static str> = exact
        .components
        .iter()
        .filter(|component| component.runtime.is_present())
        .map(|component| component.node.id)
        .collect();
    let unsupported = exact
        .components
        .iter()
        .filter(|component| {
            component.runtime.is_present()
                && !component.node.facts.witness_writer.is_capture_safe()
        })
        .map(|component| component.node.id)
        .collect();
    let recorded = exact
        .components
        .iter()
        .filter(|component| {
            component.runtime.is_present()
                && component.node.facts.witness_writer.kind == WitnessWriterKind::RecordedAot
        })
        .map(|component| component.node.id)
        .collect();
    FixtureProfile {
        present,
        unsupported,
        recorded,
    }
}

/// The SN2-shape fixture must exercise every builtin family SN PIEs exercise
/// (poseidon, pedersen, bitwise, ec_op, range-check) with zero writers the
/// strict resident path cannot admit, under the same Canonical preprocessed
/// variant the SN PIE lane uses.
#[test]
fn sn2_profile_fixture_is_strict_resident_admissible_under_canonical() {
    let profile = fixture_profile(
        "test_prove_verify_sn2_profile",
        PreProcessedTraceVariant::Canonical,
    );
    assert!(
        profile.unsupported.is_empty(),
        "sn2-profile fixture contains unsupported writers: {:?}",
        profile.unsupported
    );
    for required in [
        "poseidon_builtin",
        "poseidon_aggregator",
        "poseidon_full_round_chain",
        "poseidon_3_partial_rounds_chain",
        "pedersen_builtin",
        "pedersen_aggregator_window_bits_18",
        "partial_ec_mul_window_bits_18",
        "partial_ec_mul_generic",
        "bitwise_builtin",
        "range_check_builtin",
    ] {
        assert!(
            profile.recorded.contains(&required),
            "sn2-profile fixture lost recorded component {required}: {:?}",
            profile.recorded
        );
    }
    for required in [
        "ec_op_builtin",
        "memory_address_to_id",
        "memory_id_to_big",
        "pedersen_points_table_window_bits_18",
    ] {
        assert!(
            profile.present.contains(&required),
            "sn2-profile fixture lost coverage for {required}: {:?}",
            profile.present
        );
    }
    for excluded in [
        "generic_opcode",
        "add_mod_builtin",
        "mul_mod_builtin",
        "range_check96_builtin",
        "partial_ec_mul_window_bits_9",
        "pedersen_aggregator_window_bits_9",
        "pedersen_builtin_narrow_windows",
    ] {
        assert!(
            !profile.present.contains(&excluded),
            "sn2-profile fixture must not present detached writer {excluded}"
        );
    }
}

/// The interim poseidon fixture stays admissible under its variant so the
/// currently counted hardware gate keeps its meaning while the SN2-profile
/// fixture lands.
#[test]
fn poseidon_fixture_is_strict_resident_admissible() {
    let profile = fixture_profile(
        "test_prove_verify_poseidon_builtin",
        PreProcessedTraceVariant::CanonicalWithoutPedersen,
    );
    assert!(
        profile.unsupported.is_empty(),
        "poseidon fixture contains unsupported writers: {:?}",
        profile.unsupported
    );
    for required in [
        "poseidon_builtin",
        "poseidon_aggregator",
        "poseidon_full_round_chain",
        "poseidon_3_partial_rounds_chain",
    ] {
        assert!(
            profile.recorded.contains(&required),
            "poseidon fixture lost recorded component {required}: {:?}",
            profile.recorded
        );
    }
}
