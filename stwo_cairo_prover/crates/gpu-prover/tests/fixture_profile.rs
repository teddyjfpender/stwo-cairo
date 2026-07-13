//! Host-side fixture-profile gate for the strict resident whole-proof tests.
//!
//! Runs on any machine (no CUDA): proves the checked-in fixtures actually
//! present the component families the hardware gates claim to exercise, and
//! none of the writers the strict resident path cannot admit. This keeps the
//! hardware gate from silently narrowing when a fixture or the schedule
//! changes.

use std::sync::Arc;

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::plan::ProofPlan;
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::resident_session::plan_resident_preflight;
use stwo_cairo_gpu_prover::schedule::WitnessWriterKind;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, WorkspaceKey};
use stwo_cairo_prover::witness::exec_context::WitnessExecContext;
use stwo_cairo_prover::witness::jit_prove_backend::recorded_input_compaction_geometry;
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};

#[path = "common/base_param_variant.rs"]
mod base_param_variant;
use base_param_variant::swap_bitwise_and_ec_op_segments;

/// The device-compacted consumer set (RLE multiset compaction on device; the
/// registry is `recorded_input_compaction_geometry`).
const COMPACTED_CONSUMERS: [&str; 3] = [
    "verify_instruction",
    "pedersen_aggregator_window_bits_18",
    "poseidon_aggregator",
];

struct FixtureProfile {
    present: Vec<&'static str>,
    unsupported: Vec<&'static str>,
    recorded: Vec<&'static str>,
    /// `(label, n_real_rows, padded_rows)` of every PRESENT device-compacted
    /// consumer in the strict exact plan — sealed pre-witness by the host
    /// derivation (the exactness oracle is the differential test below).
    compacted_rows: Vec<(&'static str, u64, u64)>,
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
        .expect(
            "strict_resident_exact must SUCCEED on the sealed ingest plan (the compacted \
             consumers' exact rows are host-derived pre-witness)",
        );
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
            component.runtime.is_present() && !component.node.facts.witness_writer.is_capture_safe()
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
    let compacted_rows = exact
        .components
        .iter()
        .filter(|component| {
            component.runtime.is_present()
                && recorded_input_compaction_geometry(component.node.id).is_some()
        })
        .map(|component| {
            let RowResolution::Resolved(parts) = &component.runtime.rows else {
                panic!(
                    "compacted consumer {} is not exact in the strict resident plan",
                    component.node.id
                );
            };
            assert_eq!(parts.len(), 1);
            assert_eq!(parts[0].part, TracePartId::Main);
            (
                component.node.id,
                parts[0].n_real_rows,
                parts[0].padded_rows,
            )
        })
        .collect();
    FixtureProfile {
        present,
        unsupported,
        recorded,
        compacted_rows,
    }
}

/// THE SUPERVISED CHANGE'S VERIFICATION ORACLE.
///
/// INVARIANT (soundness-critical): for every device-compacted consumer
/// (`verify_instruction`, `pedersen_aggregator_window_bits_18`,
/// `poseidon_aggregator`), the strict resident plan's row resolution — sealed
/// at ingest by the pre-witness host derivation — must equal the row count the
/// SIMD `CairoClaimGenerator` realizes for the same ProverInput: same dedup
/// semantics, same padding to the next power of two.
///
/// This runs the FULL SIMD witness (`write_trace::<SimdBackend>`) against a
/// capacity-bounded shape ledger to obtain the independently REALIZED exact
/// shape, and asserts the plan-time derivation equals it — per compacted
/// component, and then for the whole strict plan shape (every plain-feed
/// promotion downstream of the sealed rows must be exact too, or the planned
/// claim would diverge from the realized claim).
fn assert_plan_time_compacted_rows_match_simd_realization(
    fixture: &str,
    variant: PreProcessedTraceVariant,
) {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path(fixture),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let ingest = phases::ingest::run(input, variant, None);

    // Plan-time: the ingest-sealed shape resolves every present compacted
    // consumer exactly, and the strict resolution succeeds on it.
    let planned_rows: Vec<(&'static str, u64, u64)> = COMPACTED_CONSUMERS
        .iter()
        .filter_map(|label| {
            let component = ingest.proof_plan.proof_shape().component(label).unwrap();
            match &component.rows {
                RowResolution::Absent => None,
                RowResolution::Resolved(parts) => {
                    assert_eq!(parts.len(), 1, "{label}");
                    assert_eq!(parts[0].part, TracePartId::Main, "{label}");
                    Some((*label, parts[0].n_real_rows, parts[0].padded_rows))
                }
                other => panic!("{label} is not sealed exact at ingest: {other:?}"),
            }
        })
        .collect();
    assert!(
        !planned_rows.is_empty(),
        "fixture {fixture} exercises no compacted consumer"
    );
    let exact_plan = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();

    // Realization: run the full SIMD witness against an INDEPENDENT
    // capacity-bounded ledger (the pre-derivation contract), then seal the
    // realized exact shape from what the writers actually produced.
    let capacity_shape = ingest.generator.proof_shape(None).unwrap();
    let capacity_plan =
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &capacity_shape).unwrap();
    let exec_context = WitnessExecContext::planned_with_shape(
        Arc::new(CAIRO_SCHEDULE.artifact_plan().unwrap()),
        capacity_plan.proof_shape().clone(),
    );
    let (_trace, _claim, _interaction_generator) =
        ingest
            .generator
            .write_trace::<SimdBackend>(&exec_context, None, None);
    let realized = exec_context.seal_final_proof_shape().unwrap();

    // The oracle: plan-time derivation == SIMD realization, exactly.
    for (label, planned_real, planned_padded) in &planned_rows {
        let RowResolution::Resolved(parts) = &realized.component(label).unwrap().rows else {
            panic!("realized shape left {label} unresolved");
        };
        assert_eq!(
            (parts[0].n_real_rows, parts[0].padded_rows),
            (*planned_real, *planned_padded),
            "plan-time derived rows for {label} diverge from the SIMD-realized rows \
             (same-dedup/same-padding invariant violated)"
        );
    }
    // And the whole strict plan (compacted seals + plain-feed promotions)
    // must equal the realized shape component-for-component.
    assert_eq!(
        exact_plan.proof_shape(),
        &realized,
        "strict resident plan shape diverges from the SIMD-realized exact shape"
    );
}

#[test]
fn plan_time_compacted_rows_match_simd_realization_for_poseidon_fixture() {
    assert_plan_time_compacted_rows_match_simd_realization(
        "test_prove_verify_poseidon_builtin",
        PreProcessedTraceVariant::CanonicalWithoutPedersen,
    );
}

#[test]
fn plan_time_compacted_rows_match_simd_realization_for_sn2_profile_fixture() {
    assert_plan_time_compacted_rows_match_simd_realization(
        "test_prove_verify_sn2_profile",
        PreProcessedTraceVariant::Canonical,
    );
}

/// The actual SN2 fixture can change the hoisted bitwise/EC-op BASE values
/// without changing the exact proof shape or resident workspace cache key.
/// This is the host oracle for the hardware A/B workspace-reuse parity test.
#[test]
fn sn2_profile_base_param_variant_preserves_resident_workspace_key() {
    let first = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let first_bitwise = first.builtin_segments.bitwise_builtin.unwrap();
    let first_ec_op = first.builtin_segments.ec_op_builtin.unwrap();
    let second = swap_bitwise_and_ec_op_segments(first.clone());
    let second_bitwise = second.builtin_segments.bitwise_builtin.unwrap();
    let second_ec_op = second.builtin_segments.ec_op_builtin.unwrap();

    assert_ne!(first_bitwise.begin_addr, second_bitwise.begin_addr);
    assert_ne!(first_ec_op.begin_addr, second_ec_op.begin_addr);
    assert_eq!(
        first_bitwise.stop_ptr - first_bitwise.begin_addr,
        second_bitwise.stop_ptr - second_bitwise.begin_addr
    );
    assert_eq!(
        first_ec_op.stop_ptr - first_ec_op.begin_addr,
        second_ec_op.stop_ptr - second_ec_op.begin_addr
    );
    assert_eq!(
        (
            first.memory.address_to_id.len(),
            first.memory.f252_values.len(),
            first.memory.small_values.len(),
            first.public_memory_addresses.len(),
        ),
        (
            second.memory.address_to_id.len(),
            second.memory.f252_values.len(),
            second.memory.small_values.len(),
            second.public_memory_addresses.len(),
        )
    );

    let first = phases::ingest::run(first, PreProcessedTraceVariant::Canonical, None);
    let second = phases::ingest::run(second, PreProcessedTraceVariant::Canonical, None);
    assert_eq!(first.proof_plan.shape_key, second.proof_plan.shape_key);

    let first_preflight = plan_resident_preflight(
        &first.generator,
        &first.proof_plan,
        &first.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    let second_preflight = plan_resident_preflight(
        &second.generator,
        &second.proof_plan,
        &second.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    assert_eq!(
        WorkspaceKey::from_plan(&first_preflight.arena),
        WorkspaceKey::from_plan(&second_preflight.arena)
    );
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
    // strict_resident_exact SUCCEEDED (fixture_profile unwraps it) with all
    // three device-compacted consumers sealed to exact pre-witness rows —
    // previously this fixture failed closed with
    // StrictResidentCompactedRowsUnresolved. Exactness against the SIMD
    // realization is pinned by the differential oracle test above.
    let mut sealed: Vec<&str> = profile
        .compacted_rows
        .iter()
        .map(|(label, ..)| *label)
        .collect();
    sealed.sort_unstable();
    assert_eq!(
        sealed,
        vec![
            "pedersen_aggregator_window_bits_18",
            "poseidon_aggregator",
            "verify_instruction",
        ],
        "sn2-profile fixture lost a sealed compacted consumer"
    );
    for (label, n_real, padded) in &profile.compacted_rows {
        assert!(
            *n_real >= 1 && n_real <= padded && padded.is_power_of_two(),
            "sealed compacted rows for {label} are malformed: {n_real}/{padded}"
        );
    }
}

/// The runtime fails closed on fixed-multiplicity coverage gaps and feed
/// blockers; both are plan-level facts, so the SN2-shape fixture pins them on
/// any machine before hardware ever runs.
#[test]
fn sn2_profile_fixture_plans_complete_multiplicity_coverage() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let ingest = phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
    let exact = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    let multiplicities =
        stwo_cairo_gpu_prover::multiplicity_pipeline::plan_graph_a_multiplicities(&exact).unwrap();
    assert!(
        multiplicities.coverage_gaps.is_empty(),
        "fixed-multiplicity coverage gaps: {:?}",
        multiplicities.coverage_gaps
    );
    assert!(
        multiplicities.blockers.is_empty(),
        "multiplicity feed blockers: {:?}",
        multiplicities.blockers
    );
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
    let mut sealed: Vec<&str> = profile
        .compacted_rows
        .iter()
        .map(|(label, ..)| *label)
        .collect();
    sealed.sort_unstable();
    assert_eq!(
        sealed,
        vec!["poseidon_aggregator", "verify_instruction"],
        "poseidon fixture lost a sealed compacted consumer"
    );
}
