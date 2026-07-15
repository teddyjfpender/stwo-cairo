//! Counted whole-proof gate for the strict resident CUDA architecture.
//!
//! CPU-only builds compile zero tests. Hardware admission requires the exact
//! executed-test count from this target, so a missing CUDA archive cannot report
//! a vacuous green result.

#![cfg(stwo_cuda_link)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use cairo_air::verifier::verify_cairo;
use cairo_air::CairoProof;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::resident_runtime::ResidentRuntimeError;
use stwo_cairo_gpu_prover::schedule::WitnessWriterKind;
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::prover::GpuError;
use stwo_cairo_gpu_prover::{
    phases, GpuCairoProver, GpuProverConfig, PreparedRuntimeMaterialization,
    ResidentSessionError, ResidentSessionTelemetry, WorkspaceMaterialization,
};
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};

#[path = "common/base_param_variant.rs"]
mod base_param_variant;
#[path = "common/reference_cache.rs"]
mod reference_cache;
use base_param_variant::swap_bitwise_and_ec_op_segments;
use reference_cache::{cached_reference_felts, serialize_felts};

// The all-opcode fixture intentionally calls `generic()`, whose indirect JNZ is
// one real `generic_opcode` row. Keep that statement as the generic-writer oracle;
// strict resident parity uses this broad, capture-safe opcode + Poseidon statement.
const STRICT_RESIDENT_FIXTURE: &str = "test_prove_verify_sn2_profile";
const UNCHANGED_REFERENCE_TAG: &str = "shared";
// Performance ceiling for this fixed fixture under the complete qualification
// policy. Correctness uses the dynamically enumerated graph-node equality in
// `ResidentHotPathBudget`; a lower count here is a valid future optimization.
const FULL_POLICY_KERNEL_LAUNCH_CEILING: u64 = 2_473;
const FULL_POLICY_FLAGS: [&str; 4] = [
    "STWO_CUDA_B2N_STAGE_FUSED",
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
];

fn full_policy_enabled() -> bool {
    FULL_POLICY_FLAGS
        .iter()
        .all(|name| std::env::var(name).as_deref() == Ok("1"))
}

fn assert_cold_runtime_materialization(telemetry: &ResidentSessionTelemetry) {
    assert_eq!(
        telemetry.workspace_materialization,
        Some(WorkspaceMaterialization::Materialized),
        "cold proof did not materialize its workspace",
    );
    assert_eq!(
        telemetry.prepared_runtime_materialization,
        Some(PreparedRuntimeMaterialization::Materialized),
        "cold proof did not materialize its prepared runtime",
    );
    assert_eq!(
        telemetry.prepared_runtime_capture_ready_at_entry,
        Some(false),
        "cold proof unexpectedly reported a pre-existing complete capture",
    );
    assert_eq!(
        telemetry.prepared_runtime_capture_ready_at_exit,
        Some(true),
        "cold proof did not leave a complete captured topology",
    );
    assert!(
        telemetry.statement_refresh.is_none(),
        "cold proof refreshed a runtime that it had just materialized",
    );
}

fn assert_warm_runtime_reuse(telemetry: &ResidentSessionTelemetry) {
    assert_eq!(
        telemetry.workspace_materialization,
        Some(WorkspaceMaterialization::Reused),
        "warm proof did not reuse its workspace",
    );
    assert_eq!(
        telemetry.prepared_runtime_materialization,
        Some(PreparedRuntimeMaterialization::Reused),
        "warm proof did not reuse its prepared runtime",
    );
    assert_eq!(
        telemetry.prepared_runtime_capture_ready_at_entry,
        Some(true),
        "warm proof did not inherit the complete cold capture",
    );
    assert_eq!(
        telemetry.prepared_runtime_capture_ready_at_exit,
        Some(true),
        "warm proof did not preserve the complete captured topology",
    );
    assert!(
        telemetry.statement_refresh.is_some(),
        "warm proof did not refresh statement-dependent runtime inputs",
    );
}

fn assert_installed_runtime_seals_workspace(
    prover: &mut GpuCairoProver<Blake2sMerkleChannel>,
    telemetry: &ResidentSessionTelemetry,
) {
    let key = telemetry
        .workspace_key
        .expect("resident session did not report its workspace key");
    assert!(prover.graph_workspace().is_none());
    assert!(prover.graph_workspace_for(key).is_none());
    assert!(prover.graph_workspace_for_mut(key).is_none());
    assert!(prover.take_graph_workspace().is_none());
    assert!(prover.take_graph_workspace_for(key).is_none());
}

fn assert_persistent_runtime_poisoned(error: GpuError) {
    assert!(matches!(
        error,
        GpuError::ResidentSession(ResidentSessionError::Runtime(
            ResidentRuntimeError::PersistentRuntimePoisoned
        ))
    ));
}

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
    let invalid_channel_input = reference_input.clone();
    assert_capture_safe_fixture(&reference_input, params);

    // Resident session prep and proving run before the ~20-minute SIMD
    // reference so a session/prepare failure surfaces in about a minute.
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let cold = prover
        .prove_resident_blake2s(resident_input(), params)
        .unwrap();
    let cold_session = prover
        .last_resident_session_telemetry()
        .expect("cold resident setup telemetry")
        .clone();
    assert_installed_runtime_seals_workspace(&mut prover, &cold_session);
    let warm = prover
        .prove_resident_blake2s(resident_input(), params)
        .unwrap();
    let warm_session = prover
        .last_resident_session_telemetry()
        .expect("warm resident setup telemetry")
        .clone();
    assert_cold_runtime_materialization(&cold_session);
    assert_warm_runtime_reuse(&warm_session);
    assert_installed_runtime_seals_workspace(&mut prover, &warm_session);
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
    let exec = telemetry
        .exec
        .expect("strict whole-proof execution telemetry");
    assert_eq!(exec.graph_launches, 29);
    assert!(exec.kernel_launches >= exec.graph_launches);
    if full_policy_enabled() {
        assert!(
            exec.kernel_launches <= FULL_POLICY_KERNEL_LAUNCH_CEILING,
            "full-policy captured kernel-node count regressed: {} > {}",
            exec.kernel_launches,
            FULL_POLICY_KERNEL_LAUNCH_CEILING,
        );
    }

    let mut invalid_params = resident_params();
    invalid_params.channel_hash = ChannelHash::Blake2sM31;
    prover
        .prove_resident_blake2s(invalid_channel_input, invalid_params)
        .expect_err("unsupported resident channel unexpectedly proved");
    assert!(prover.last_pcs_telemetry().is_none());
    assert!(prover.last_resident_session_telemetry().is_none());
    assert!(prover.last_aot_stats().is_none());
}

#[test]
fn strict_resident_runtime_poisoned_after_incomplete_success_error_or_unwind() {
    let params = resident_params();

    let mut incomplete_config = GpuProverConfig::default();
    incomplete_config.strict = true;
    let mut incomplete_prover =
        GpuCairoProver::<Blake2sMerkleChannel>::new(incomplete_config).unwrap();
    let incomplete_error = incomplete_prover
        .with_strict_resident_session(resident_input(), params, |_, _| Ok(()))
        .expect_err("incomplete successful callback unexpectedly disarmed its runtime lease");
    assert!(matches!(
        incomplete_error,
        GpuError::ResidentSession(ResidentSessionError::Runtime(
            ResidentRuntimeError::CapturedGraphTopology { actual: 0, .. }
        ))
    ));
    let poisoned_incomplete = incomplete_prover
        .with_strict_resident_session(resident_input(), params, |_, _| Ok(()))
        .expect_err("incomplete successful callback did not poison the prepared runtime");
    assert_persistent_runtime_poisoned(poisoned_incomplete);

    let mut error_config = GpuProverConfig::default();
    error_config.strict = true;
    let mut error_prover =
        GpuCairoProver::<Blake2sMerkleChannel>::new(error_config).unwrap();
    let callback_error = error_prover
        .with_strict_resident_session(resident_input(), params, |_, _| {
            Err::<(), _>(ResidentRuntimeError::MissingPreparedRuntimeMaterialization)
        })
        .expect_err("intentional resident callback error unexpectedly succeeded");
    assert!(matches!(
        callback_error,
        GpuError::ResidentSession(ResidentSessionError::Runtime(
            ResidentRuntimeError::MissingPreparedRuntimeMaterialization
        ))
    ));
    let poisoned_error = error_prover
        .with_strict_resident_session(resident_input(), params, |_, _| Ok(()))
        .expect_err("callback error did not poison the prepared runtime");
    assert_persistent_runtime_poisoned(poisoned_error);

    let mut unwind_config = GpuProverConfig::default();
    unwind_config.strict = true;
    let mut unwind_prover =
        GpuCairoProver::<Blake2sMerkleChannel>::new(unwind_config).unwrap();
    let unwind = catch_unwind(AssertUnwindSafe(|| {
        unwind_prover.with_strict_resident_session(
            resident_input(),
            params,
            |_, _| -> Result<(), ResidentRuntimeError> {
                panic!("intentional resident callback unwind")
            },
        )
    }));
    assert!(unwind.is_err(), "resident callback did not unwind");
    let poisoned_unwind = unwind_prover
        .with_strict_resident_session(resident_input(), params, |_, _| Ok(()))
        .expect_err("callback unwind did not poison the prepared runtime");
    assert_persistent_runtime_poisoned(poisoned_unwind);
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
    let first_session = prover
        .last_resident_session_telemetry()
        .expect("first resident setup telemetry")
        .clone();
    let resident_second = prover
        .prove_resident_blake2s(second.clone(), params)
        .unwrap();
    let second_session = prover
        .last_resident_session_telemetry()
        .expect("second resident setup telemetry")
        .clone();
    assert_cold_runtime_materialization(&first_session);
    assert_warm_runtime_reuse(&second_session);
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
        second_session.cache_hit(),
        "changed-memory test did not reuse the exact workspace key"
    );
}

/// A workspace-cache hit must refresh statement-dependent BASE parameter
/// buffers. This catches a stale captured bitwise/EC-op segment start even when
/// the AOT kernel identity, proof shape, arena geometry, and graph topology are
/// all unchanged.
#[test]
fn strict_resident_same_workspace_changed_base_params_matches_second_simd_proof() {
    let params = resident_params();
    let first = resident_input();
    assert_capture_safe_fixture(&first, params);
    let second = swap_bitwise_and_ec_op_segments(first.clone());
    let first_bitwise_start = first
        .builtin_segments
        .bitwise_builtin
        .expect("first bitwise segment")
        .begin_addr;
    let second_bitwise_start = second
        .builtin_segments
        .bitwise_builtin
        .expect("second bitwise segment")
        .begin_addr;
    assert_ne!(
        first_bitwise_start, second_bitwise_start,
        "BASE-parameter variant did not move the bitwise segment"
    );
    assert_eq!(
        first.memory.address_to_id.len(),
        second.memory.address_to_id.len(),
        "BASE-parameter variant changed execution-table geometry"
    );

    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let resident_first = prover.prove_resident_blake2s(first, params).unwrap();
    let first_aot = prover
        .last_aot_stats()
        .expect("first strict AOT provenance telemetry");
    let first_session = prover
        .last_resident_session_telemetry()
        .expect("first resident setup telemetry")
        .clone();
    assert_cold_runtime_materialization(&first_session);
    let first_workspace = first_session
        .workspace_key
        .expect("first resident workspace key");
    let resident_second = prover
        .prove_resident_blake2s(second.clone(), params)
        .unwrap();
    let second_aot = prover
        .last_aot_stats()
        .expect("second strict AOT provenance telemetry");
    let second_session = prover
        .last_resident_session_telemetry()
        .expect("second resident setup telemetry")
        .clone();
    assert_warm_runtime_reuse(&second_session);
    assert_eq!(
        second_session.workspace_key,
        Some(first_workspace),
        "BASE-parameter variant changed the exact workspace key"
    );
    assert!(
        second_session.cache_hit(),
        "BASE-parameter variant did not reuse the resident workspace"
    );

    let first_roots = verify_and_roots(&resident_first);
    let second_roots = verify_and_roots(&resident_second);
    assert_ne!(
        first_roots, second_roots,
        "relocated statement did not change the four commitment roots"
    );
    let expected_second = cached_reference_felts(
        STRICT_RESIDENT_FIXTURE,
        "changed-base-params-second",
        second,
        params,
    );
    assert_eq!(
        expected_second,
        serialize_felts(&resident_second),
        "same-workspace replay reused stale BASE parameter values"
    );

    assert!(
        first_aot.aot_loads
            + first_aot.aot_cache_hits
            + second_aot.aot_loads
            + second_aot.aot_cache_hits
            > 0,
        "BASE-parameter regression exercised no AOT kernel lookup"
    );
    for (round, stats) in [("first", first_aot), ("second", second_aot)] {
        assert_eq!(stats.aot_misses, 0, "{round} strict proof missed AOT");
        assert_eq!(stats.runtime_loads, 0, "{round} strict proof loaded JIT");
        assert_eq!(
            stats.runtime_cache_hits, 0,
            "{round} strict proof reused JIT"
        );
        assert_eq!(
            stats.strict_rejections, 0,
            "{round} strict proof rejected AOT"
        );
    }
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
