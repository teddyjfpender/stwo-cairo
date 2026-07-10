//! Counted whole-proof gate for the strict resident CUDA architecture.
//!
//! CPU-only builds compile zero tests. Hardware admission requires the exact
//! executed-test count from this target, so a missing CUDA archive cannot report
//! a vacuous green result.

#![cfg(stwo_cuda_link)]

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::schedule::WitnessWriterKind;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

// The all-opcode fixture intentionally calls `generic()`, whose indirect JNZ is
// one real `generic_opcode` row. Keep that statement as the generic-writer oracle;
// strict resident parity uses this broad, capture-safe opcode + Poseidon statement.
const STRICT_RESIDENT_FIXTURE: &str = "test_prove_verify_poseidon_builtin";

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
        preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
        channel_salt: 0,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
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
    ] {
        assert!(
            recorded.contains(&required),
            "strict resident fixture lost recorded component {required}: {recorded:?}"
        );
    }
    for required in ["memory_address_to_id", "memory_id_to_big"] {
        assert!(
            present
                .iter()
                .any(|component| component.node.id == required),
            "strict resident fixture lost changed-memory coverage for {required}"
        );
    }
}

fn serialize_felts<H>(proof: &cairo_air::CairoProof<H>) -> Vec<starknet_ff::FieldElement>
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    H::Hash: CairoSerialize,
{
    let mut felts = Vec::new();
    CairoSerialize::serialize(proof, &mut felts);
    felts
}

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

    let expected = serialize_felts(
        &prove_cairo::<SimdBackend, Blake2sMerkleChannel>(reference_input, params).unwrap(),
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

    let expected_first =
        serialize_felts(&prove_cairo::<SimdBackend, Blake2sMerkleChannel>(first, params).unwrap());
    let expected_second =
        serialize_felts(&prove_cairo::<SimdBackend, Blake2sMerkleChannel>(second, params).unwrap());
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

    let expected = serialize_felts(
        &prove_cairo::<SimdBackend, Blake2sMerkleChannel>(reference_input, params).unwrap(),
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

    let expected = serialize_felts(
        &prove_cairo::<SimdBackend, Blake2sMerkleChannel>(reference_input, params).unwrap(),
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
