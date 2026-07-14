use std::panic::{catch_unwind, AssertUnwindSafe};

use stwo::core::channel::{Blake2sChannel, Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;

use super::*;
use crate::fri_round6_capture::{CaptureShape, CaptureSource, Observed, PcsShape, VerifiedCapture};

#[test]
fn wrapper_is_cryptographically_identical_to_canonical_blake2s() {
    let mut canonical = Blake2sChannel::default();
    let mut recorded = RecordingBlake2sChannel::default();
    let felts = [SecureField::from(3_u32), SecureField::from(7_u32)];

    canonical.mix_u32s(&[1, 2, 3]);
    recorded.mix_u32s(&[1, 2, 3]);
    canonical.mix_felts(&felts);
    recorded.mix_felts(&felts);
    canonical.mix_u64(0x0123_4567_89ab_cdef);
    recorded.mix_u64(0x0123_4567_89ab_cdef);
    assert_eq!(
        canonical.verify_pow_nonce(3, 17),
        recorded.verify_pow_nonce(3, 17)
    );
    assert_eq!(
        canonical.draw_secure_felts(3),
        recorded.draw_secure_felts(3)
    );
    assert_eq!(canonical.draw_u32s(), recorded.draw_u32s());

    let root = Blake2sHash([0x5a; 32]);
    <Blake2sMerkleChannel as MerkleChannel>::mix_root(&mut canonical, root);
    <RecordingBlake2sMerkleChannel as MerkleChannel>::mix_root(&mut recorded, root);
    assert_eq!(canonical.draw_secure_felt(), recorded.draw_secure_felt());
    assert_eq!(state(&canonical), state(&recorded.inner));
}

#[test]
fn internal_extractor_matches_one_exact_round6_sequence() {
    let (events, capture) = recorded_round();
    let matched = bind_for_test(&events, &capture).unwrap();

    assert_eq!(matched.capture_sha256, capture.capture_sha256);
    assert_eq!(
        matched.adapted_prover_input_sha256,
        capture.source.prover_input_sha256
    );
    assert_eq!(matched.verifier_events, expected_events(&capture).unwrap());
    assert_eq!(matched.verifier_events_sha256.len(), 64);
}

#[test]
fn every_captured_round6_value_is_required() {
    let (events, capture) = recorded_round();
    for mutate in 0..8 {
        let mut hostile = clone_capture(&capture);
        match mutate {
            0 => hostile.cursor32[0] ^= 1,
            1 => hostile.observed.root6_words[0] ^= 1,
            2 => hostile.observed.alpha6_words[0] ^= 1,
            3 => hostile.observed.cursor34_state_words[0] ^= 1,
            4 => hostile.observed.root7_words[0] ^= 1,
            5 => hostile.observed.cursor35_state_words[0] ^= 1,
            6 => hostile.observed.alpha7_words[0] ^= 1,
            7 => hostile.observed.cursor36_state_words[0] ^= 1,
            _ => unreachable!(),
        }
        let error = bind_for_test(&events, &hostile).unwrap_err();
        assert!(error.contains("no exact round-6 capture match"), "{error}");
    }
}

#[test]
fn duplicate_exact_sequence_is_rejected_as_ambiguous() {
    let (mut events, capture) = recorded_round();
    events.extend(events.clone());
    let error = bind_for_test(&events, &capture).unwrap_err();
    assert!(
        error.contains("multiple round-6 capture matches"),
        "{error}"
    );
}

#[test]
fn disconnected_event_evidence_is_rejected() {
    let (mut events, capture) = recorded_round();
    let VerifierEvent::MixRoot { after, .. } = &mut events[0] else {
        unreachable!();
    };
    after.digest_words[0] ^= 1;
    let error = bind_for_test(&events, &capture).unwrap_err();
    assert!(error.contains("no exact round-6 capture match"), "{error}");
}

#[test]
fn failed_or_panicking_capture_cannot_poison_the_next_run() {
    let nested = record_verifier_events(|| record_verifier_events(|| ())).unwrap();
    assert!(nested.0.is_err());

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = record_verifier_events(|| panic!("hostile verifier panic"));
    }));
    assert!(panic.is_err());
    assert!(record_verifier_events(|| 7).is_ok());
}

#[test]
fn bounded_recorder_rejects_overflow_without_changing_channel_execution() {
    let mut channel = RecordingBlake2sChannel::default();
    let error = record_verifier_events(|| {
        for _ in 0..=MAX_RECORDED_EVENTS {
            channel.draw_secure_felt();
        }
    })
    .unwrap_err();
    assert!(error.contains("exceeded 4096 events"), "{error}");
    assert!(channel.inner.n_draws() >= (MAX_RECORDED_EVENTS + 1) as u32);
}

#[test]
#[ignore = "set STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY=1 on a cheap host"]
fn accepted_proof_records_but_post_transcript_failure_cannot_match() {
    use std::path::PathBuf;

    use cairo_air::verifier::verify_cairo;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::PcsConfig;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo_cairo_adapter::ProverInput;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
    use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};

    assert!(
        matches!(
            std::env::var("STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY").as_deref(),
            Ok("1")
        ),
        "real-proof recorder test requires its explicit cheap-host gate"
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../../stwo_cairo_prover/test_data/test_prove_verify_all_opcode_components/prover_input.json",
    );
    let input: ProverInput = serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
    )
    .expect("deserialize all-opcodes ProverInput");
    let parameters = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config: PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: None,
        },
        preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };
    let proof = prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, parameters)
        .expect("prove real all-opcodes Cairo fixture");
    let proof_bytes = bincode::serialize(&proof).expect("serialize accepted proof");

    let accepted =
        fri_round6_proof::decode_exact_bincode::<Blake2sCairoProof>(&proof_bytes, "accepted proof")
            .expect("decode accepted proof");
    derive_capture_shape(&accepted).expect("derive accepted proof capture shape");
    let (accepted_result, accepted_events) =
        record_verifier_events(|| verify_cairo::<RecordingBlake2sMerkleChannel>(accepted.into()))
            .expect("record accepted proof verification");
    accepted_result.expect("accepted proof verifies through recording wrapper");
    assert!(
        !accepted_events.is_empty(),
        "accepted proof emitted no events"
    );

    let mut post_transcript = fri_round6_proof::decode_exact_bincode::<Blake2sCairoProof>(
        &proof_bytes,
        "post-transcript mutation",
    )
    .expect("decode proof for post-transcript mutation");
    let queried_value = post_transcript
        .extended_stark_proof
        .proof
        .0
        .queried_values
        .iter_mut()
        .flat_map(|tree| tree.iter_mut())
        .flat_map(|column| column.iter_mut())
        .next()
        .expect("real proof has a queried value");
    *queried_value += stwo::core::fields::m31::BaseField::from(1_u32);
    let (rejected_result, rejected_events) = record_verifier_events(|| {
        verify_cairo::<RecordingBlake2sMerkleChannel>(post_transcript.into())
    })
    .expect("record rejected post-transcript proof verification");
    assert!(
        rejected_result.is_err(),
        "post-transcript queried-value mutation unexpectedly verified"
    );
    assert_eq!(
        accepted_events, rejected_events,
        "post-transcript verifier failure changed the captured transcript"
    );
}

fn recorded_round() -> (Vec<VerifierEvent>, VerifiedCapture) {
    let root6 = Blake2sHash([0x61; 32]);
    let root7 = Blake2sHash([0x72; 32]);
    let (_, events) = record_verifier_events(|| {
        let mut channel = RecordingBlake2sChannel::default();
        channel.mix_u32s(&[0x5354_574f, 6]);
        <RecordingBlake2sMerkleChannel as MerkleChannel>::mix_root(&mut channel, root6);
        channel.draw_secure_felt();
        <RecordingBlake2sMerkleChannel as MerkleChannel>::mix_root(&mut channel, root7);
        channel.draw_secure_felt();
    })
    .unwrap();
    assert_eq!(events.len(), 4);
    let extracted = extract_round6(&events).unwrap();
    let capture = capture_from_events(&extracted);
    (events, capture)
}

fn capture_from_events(events: &FriRound6VerifierEvents) -> VerifiedCapture {
    VerifiedCapture {
        capture_sha256: "11".repeat(32),
        source: CaptureSource {
            observer: "stwo-cairo.production-simd-fri-observer.v1".into(),
            prover_input_sha256: "22".repeat(32),
            prover_input_bytes: 42,
            observer_proof_shape_id: "77".repeat(32),
        },
        shape: CaptureShape {
            circle_log_size: 24,
            claim_enable_felts: 1,
            claim_log_size_felts: 1,
            claim_public_data_felts: 1,
            interaction_claim_felts: 1,
            oods_sampled_values_felts: 1,
            interaction_pow_bits: 24,
            pcs: PcsShape {
                pow_bits: 26,
                log_blowup_factor: 1,
                n_queries: 70,
                log_last_layer_degree_bound: 0,
                fold_step: 3,
                lifting_log_size: 24,
            },
            fri_tree_count: 8,
        },
        cursor32: capture_state_words(&events.pre_root6),
        entry_words: vec![0; 256],
        observed: Observed {
            root6_words: events.root6_words.to_vec(),
            alpha6_words: events.alpha6_words.to_vec(),
            cursor34_state_words: capture_state_words(&events.cursor34),
            root7_words: events.root7_words.to_vec(),
            alpha7_words: events.alpha7_words.to_vec(),
            cursor35_state_words: capture_state_words(&events.cursor35),
            cursor36_state_words: capture_state_words(&events.cursor36),
        },
        chains: [0; 5],
        device_protocol_key: 0,
        cairo_schedule_key: 0,
    }
}

fn capture_state_words(state: &ChannelState) -> Vec<u32> {
    let mut words = state.digest_words.to_vec();
    words.extend([state.n_draws, 0, 0, 0, 0, 0, 0, 0]);
    words
}

fn clone_capture(capture: &VerifiedCapture) -> VerifiedCapture {
    VerifiedCapture {
        capture_sha256: capture.capture_sha256.clone(),
        source: capture.source.clone(),
        shape: capture.shape.clone(),
        cursor32: capture.cursor32.clone(),
        entry_words: capture.entry_words.clone(),
        observed: capture.observed.clone(),
        chains: capture.chains,
        device_protocol_key: capture.device_protocol_key,
        cairo_schedule_key: capture.cairo_schedule_key,
    }
}

fn bind_for_test(
    events: &[VerifierEvent],
    capture: &VerifiedCapture,
) -> Result<VerifiedCaptureMatch, String> {
    bind_loaded_capture(
        events,
        capture,
        BoundIdentities {
            manifest_sha256: "33".repeat(32),
            adapted_prover_input_sha256: capture.source.prover_input_sha256.clone(),
            adapted_prover_input_bytes: capture.source.prover_input_bytes,
            proof_sha256: "44".repeat(32),
            canonical_transport_sha256: "55".repeat(32),
            verifier_source_closure_sha256: "66".repeat(32),
            proof_shape_sha256: capture.source.observer_proof_shape_id.clone(),
        },
    )
}
