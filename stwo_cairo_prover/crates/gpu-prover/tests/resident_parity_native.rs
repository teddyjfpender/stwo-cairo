//! Counted whole-proof gate for the strict resident CUDA architecture.
//!
//! CPU-only builds compile zero tests. Hardware admission requires the exact
//! executed-test count from this target, so a missing CUDA archive cannot report
//! a vacuous green result.

#![cfg(stwo_cuda_link)]

use cairo_air::verifier::verify_cairo;
use cairo_air::CairoProof;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::schedule::WitnessWriterKind;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};

#[path = "common/reference_cache.rs"]
mod reference_cache;
use reference_cache::{cached_reference_felts, serialize_felts};

// The all-opcode fixture intentionally calls `generic()`, whose indirect JNZ is
// one real `generic_opcode` row. Keep that statement as the generic-writer oracle;
// strict resident parity uses this broad, capture-safe opcode + Poseidon statement.
const STRICT_RESIDENT_FIXTURE: &str = "test_prove_verify_sn2_profile";
const UNCHANGED_REFERENCE_TAG: &str = "shared";

fn resident_input() -> ProverInput {
    run_and_adapt(
        &get_compiled_cairo_program_path(STRICT_RESIDENT_FIXTURE),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap()
}

fn resident_params() -> ProverParameters {
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig::default(),
        preprocessed_trace: PreProcessedTraceVariant::Canonical,
        channel_salt: 0,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

fn verify_and_roots(
    proof: &CairoProof<Blake2sMerkleHasher>,
) -> Vec<stwo::core::vcs::blake2_hash::Blake2sHash> {
    verify_cairo::<Blake2sMerkleChannel>(proof.clone().into())
        .expect("resident proof verification");
    let roots = proof.extended_stark_proof.proof.0.commitments.0.clone();
    assert_eq!(
        roots.len(),
        4,
        "resident proof must carry four commitment roots"
    );
    roots
}

fn assert_capture_safe_fixture(input: &ProverInput, params: ProverParameters) {
    let ingest = phases::ingest::run(
        input.clone(),
        params.preprocessed_trace,
        params.opt_n_id_to_big_components,
    );
    let exact = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    let present = exact
        .components
        .iter()
        .filter(|component| component.runtime.is_present())
        .collect::<Vec<_>>();
    let unsupported = present
        .iter()
        .filter(|component| !component.node.facts.witness_writer.is_capture_safe())
        .map(|component| component.node.id)
        .collect::<Vec<_>>();
    assert!(
        unsupported.is_empty(),
        "strict resident fixture contains unsupported writers: {unsupported:?}"
    );

    let recorded = present
        .iter()
        .filter(|component| {
            component.node.facts.witness_writer.kind == WitnessWriterKind::RecordedAot
        })
        .map(|component| component.node.id)
        .collect::<Vec<_>>();
    assert!(
        recorded.len() >= 10,
        "strict resident fixture lost broad recorded-witness coverage: {recorded:?}"
    );
    for required in [
        "add_opcode_small",
        "assert_eq_opcode",
        "call_opcode_rel_imm",
        "jnz_opcode_non_taken",
        "jnz_opcode_taken",
        "ret_opcode",
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
            recorded.contains(&required),
            "strict resident fixture lost recorded component {required}: {recorded:?}"
        );
    }
    for required in [
        "memory_address_to_id",
        "memory_id_to_big",
        "ec_op_builtin",
        "pedersen_points_table_window_bits_18",
    ] {
        assert!(
            present
                .iter()
                .any(|component| component.node.id == required),
            "strict resident fixture lost changed-memory coverage for {required}"
        );
    }
}

/// Deterministic SIMD reference proofs are expensive (~20 minutes of pod CPU
/// per green round) and fixed for a given (fixture, params, input tag), so an
/// opt-in cache (STWO_PARITY_REF_CACHE=<dir>) stores the serialized reference
/// felts once and replays them byte-for-byte. Flag sweeps are byte-identical
/// by construction, so the cache stays valid across the whole measurement
/// campaign; delete the directory to force recomputation after any change
/// that legitimately moves the reference.

#[test]
fn strict_resident_cold_and_warm_proofs_match_simd_bytes() {
    let params = resident_params();
    let reference_input = resident_input();
    assert_capture_safe_fixture(&reference_input, params);

    // Resident session prep and proving run before the ~20-minute SIMD
    // reference so a session/prepare failure surfaces in about a minute.
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let cold = prover
        .prove_resident_blake2s(resident_input(), params)
        .unwrap();
    let warm = prover
        .prove_resident_blake2s(resident_input(), params)
        .unwrap();
    let cold_roots = verify_and_roots(&cold);
    let warm_roots = verify_and_roots(&warm);
    assert_eq!(cold_roots, warm_roots, "cold/warm commitment roots drifted");

    let expected = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        UNCHANGED_REFERENCE_TAG,
        reference_input,
        params,
    );
    assert_eq!(
        expected,
        serialize_felts(&cold),
        "cold resident proof drifted"
    );
    assert_eq!(
        expected,
        serialize_felts(&warm),
        "warm resident proof drifted"
    );

    let telemetry = prover.last_pcs_telemetry().unwrap();
    assert!(telemetry.is_complete());
    assert!(telemetry.exec.is_some());
}

/// Same workspace geometry with different compact memory content must rebuild
/// the session-owned execution tables. This catches the former process-global
/// cache keyed only by a reusable host pointer.
#[test]
fn strict_resident_same_shape_changed_memory_matches_second_simd_proof() {
    let params = resident_params();
    let first = resident_input();
    assert_capture_safe_fixture(&first, params);
    let mut second = first.clone();
    assert!(second.memory.small_values.len() >= 2);
    assert_ne!(second.memory.small_values[0], second.memory.small_values[1]);
    second.memory.small_values.swap(0, 1);
    // Preserve the value at every Cairo address while changing the compact
    // table contents at each stable index. Shape/claim/workspace identity stay
    // exact; a pointer-keyed stale table is therefore observable.
    for encoded in &mut second.memory.address_to_id {
        encoded.0 = match encoded.0 {
            0 => 1,
            1 => 0,
            raw => raw,
        };
    }

    // Resident session prep and proving run before the two SIMD reference
    // proofs so a session/prepare failure surfaces in about a minute.
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let resident_first = prover
        .prove_resident_blake2s(first.clone(), params)
        .unwrap();
    let resident_second = prover
        .prove_resident_blake2s(second.clone(), params)
        .unwrap();
    let first_roots = verify_and_roots(&resident_first);
    let second_roots = verify_and_roots(&resident_second);
    assert_ne!(
        first_roots, second_roots,
        "mutated statement did not change the four commitment roots"
    );

    let expected_first = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        UNCHANGED_REFERENCE_TAG,
        first,
        params,
    );
    let expected_second = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        "changed-memory-second",
        second,
        params,
    );
    assert_ne!(
        expected_first, expected_second,
        "changed compact memory did not affect the reference proof"
    );

    assert_eq!(expected_first, serialize_felts(&resident_first));
    assert_eq!(
        expected_second,
        serialize_felts(&resident_second),
        "same-shape replay reused stale execution-table content"
    );
    assert!(
        prover
            .last_resident_session_telemetry()
            .expect("resident setup telemetry")
            .cache_hit(),
        "changed-memory test did not reuse the exact workspace key"
    );
}

/// Graph-A admission for the active Starknet Poseidon chain.  This fixture makes
/// `poseidon_builtin` produce the real six-word tuples, requires the prepared
/// device sort/RLE compactor to feed `poseidon_aggregator`, and then exercises
/// the eight full-round plus twenty-seven partial-round producer edges.  The
/// fail-closed coverage check in `ResidentGraphRuntime::prepare` validates the
/// exact compact source and all nine aggregator input destinations before any
/// graph is exposed; whole-proof bytes are the final semantic oracle.
#[test]
fn strict_resident_poseidon_graph_a_matches_simd_bytes() {
    let params = resident_params();
    let reference_input = resident_input();
    assert_capture_safe_fixture(&reference_input, params);

    // Resident session prep and proving run before the ~20-minute SIMD
    // reference so a session/prepare failure surfaces in about a minute.
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let actual = prover
        .prove_resident_blake2s(resident_input(), params)
        .unwrap();

    let expected = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        UNCHANGED_REFERENCE_TAG,
        reference_input,
        params,
    );
    assert_eq!(
        expected,
        serialize_felts(&actual),
        "Poseidon Graph-A proof drifted"
    );
    prover
        .last_resident_session_telemetry()
        .expect("Poseidon resident setup telemetry")
        .require_strict_graph_a()
        .unwrap();
}

/// One-proof transcript diagnostic. This deliberately skips the SIMD oracle so
/// a fresh secure GPU can classify the device/host channel mirror immediately.
#[test]
fn strict_resident_transcript_mirror_diagnostic_once() {
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let mirrored = prover
        .prove_resident_blake2s_with_transcript_mirror(resident_input(), resident_params())
        .unwrap();
    let mirror = &mirrored.transcript_mirror;
    assert!(mirror.report.boundaries_verified > 0);
    assert!(mirror.report.output_words_verified > 0);
    assert!(!mirror.performance_admissible);
    assert!(!mirror.performance_claim_admissible());
}

/// U4 transcript-migration qualification: three consecutive mirrored resident
/// proofs must byte-match the SIMD reference while the host Blake2s channel
/// replays and verifies every device transcript boundary. The mirrored return
/// type is correctness-only by construction, so this gate can never leak into
/// an MHz claim.
#[test]
fn strict_resident_mirrored_transcript_matches_host_channel() {
    let params = resident_params();
    let reference_input = resident_input();
    assert_capture_safe_fixture(&reference_input, params);

    // The first mirrored resident proof (and with it all session prep) runs
    // before the ~20-minute SIMD reference so a session/prepare failure
    // surfaces in about a minute; the reference is still computed exactly once
    // before the three comparison rounds.
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let mut first_mirrored = Some(
        prover
            .prove_resident_blake2s_with_transcript_mirror(resident_input(), params)
            .unwrap(),
    );

    let expected = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        UNCHANGED_REFERENCE_TAG,
        reference_input,
        params,
    );

    for round in 0..3 {
        let mirrored = match first_mirrored.take() {
            Some(mirrored) => mirrored,
            None => prover
                .prove_resident_blake2s_with_transcript_mirror(resident_input(), params)
                .unwrap(),
        };
        assert_eq!(
            expected,
            serialize_felts(&mirrored.proof),
            "mirrored resident proof drifted on round {round}"
        );
        let mirror = &mirrored.transcript_mirror;
        assert!(
            mirror.report.boundaries_verified > 0,
            "round {round} verified no transcript boundaries"
        );
        assert!(
            mirror.report.output_words_verified > 0,
            "round {round} verified no transcript output words"
        );
        assert!(
            !mirror.performance_admissible && !mirror.performance_claim_admissible(),
            "mirrored runs must never be performance-admissible"
        );
    }
}
