use std::panic::{catch_unwind, AssertUnwindSafe};

use stwo::core::channel::{Blake2sChannel, Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::fri::FriCommitObserver;
use stwo::prover::line::LineEvaluation;

use super::*;
use crate::fri_round6_capture::{
    CaptureShape, CaptureSource, Observed, ObserverChannelState, ObserverRound6, PcsShape,
    VerifiedCapture,
};

const REAL_PRODUCER_ROLE: &str = "STWO_GPU_LAB_REAL_CAPTURE_PRODUCER";
const REAL_MUTATION_ROLE: &str = "STWO_GPU_LAB_REAL_CAPTURE_MUTATION";
const REAL_OUTPUT_ROOT: &str = "STWO_GPU_LAB_REAL_CAPTURE_ROOT";
const REAL_TEST_NAME: &str = "fri_round6_proof::verifier_capture_recorder::tests::real_observer_capture_matches_canonical_verifier";

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
fn real_observer_capture_matches_canonical_verifier() {
    assert!(
        matches!(
            std::env::var("STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY").as_deref(),
            Ok("1")
        ),
        "real-proof recorder test requires its explicit cheap-host gate"
    );
    if matches!(std::env::var(REAL_PRODUCER_ROLE).as_deref(), Ok("1")) {
        let root = std::env::var_os(REAL_OUTPUT_ROOT).expect("producer output root is required");
        produce_real_observer_boundary(std::path::Path::new(&root));
        return;
    }
    if matches!(std::env::var(REAL_MUTATION_ROLE).as_deref(), Ok("1")) {
        let root = std::env::var_os(REAL_OUTPUT_ROOT).expect("mutation output root is required");
        reject_mutated_observer_boundary(std::path::Path::new(&root));
        return;
    }

    let scratch = ScratchRoot::new("stwo-real-fri-round6-boundary");
    let output = run_boundary_child(REAL_PRODUCER_ROLE, scratch.path());
    assert!(
        output.status.success(),
        "real-proof capture producer failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest_path = scratch.path().join("bundle/fri_round6_provenance.v1.json");
    let capture_path = scratch.path().join("capture.json");
    let manifest_sha256 = crate::model::sha256_hex(
        &std::fs::read(&manifest_path).expect("read produced provenance manifest"),
    );
    let capture_bytes = std::fs::read(&capture_path).expect("read produced observer capture");
    let capture_sha256 = crate::model::sha256_hex(&capture_bytes);

    let mut mutated: serde_json::Value =
        serde_json::from_slice(&capture_bytes).expect("parse produced observer capture");
    let root6_word = &mut mutated["observed"]["root6_words"][0];
    let original = root6_word.as_u64().expect("root6 word is a u32");
    *root6_word = serde_json::json!(original ^ 1);
    let mutated_bytes =
        serde_json::to_vec_pretty(&mutated).expect("serialize mutated observer capture");
    let mutated_path = scratch.path().join("capture-mutated.json");
    std::fs::write(&mutated_path, &mutated_bytes).expect("write mutated observer capture");

    let matched = verify_and_match_fri_round6(
        &manifest_path,
        &manifest_sha256,
        &capture_path,
        &capture_sha256,
    )
    .expect("real proof verifies and has one exact observer capture match");
    assert_eq!(
        matched.proof_shape_sha256, matched.observer_proof_shape_id,
        "observer capture is not bound to the exact proof shape"
    );
    assert_eq!(
        matched.adapted_prover_input_sha256,
        crate::fri_round6_capture::load(&capture_path, &capture_sha256)
            .expect("reload matched capture")
            .source
            .prover_input_sha256,
        "observer capture is not bound to the exact serialized ProverInput"
    );

    let output = run_boundary_child(REAL_MUTATION_ROLE, scratch.path());
    assert!(
        output.status.success(),
        "mutated-capture matcher child failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_boundary_child(role: &str, output_root: &std::path::Path) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().expect("resolve test binary"))
        .arg(REAL_TEST_NAME)
        .arg("--exact")
        .arg("--ignored")
        .arg("--test-threads=1")
        .env_remove(REAL_PRODUCER_ROLE)
        .env_remove(REAL_MUTATION_ROLE)
        .env(role, "1")
        .env(REAL_OUTPUT_ROOT, output_root)
        .output()
        .expect("spawn real-proof boundary child")
}

fn reject_mutated_observer_boundary(root: &std::path::Path) {
    let manifest_path = root.join("bundle/fri_round6_provenance.v1.json");
    let capture_path = root.join("capture-mutated.json");
    let manifest_sha256 = crate::model::sha256_hex(
        &std::fs::read(&manifest_path).expect("read produced provenance manifest"),
    );
    let capture_sha256 = crate::model::sha256_hex(
        &std::fs::read(&capture_path).expect("read mutated observer capture"),
    );
    let error = verify_and_match_fri_round6(
        &manifest_path,
        &manifest_sha256,
        &capture_path,
        &capture_sha256,
    )
    .expect_err("one-word observer capture mutation unexpectedly matched");
    assert!(error.contains("no exact round-6 capture match"), "{error}");
}

#[derive(Clone)]
struct ProverObservedFold {
    log_size: u32,
    entry_words: Option<Vec<u32>>,
    alpha: SecureField,
    root: Blake2sHash,
    digest: Blake2sHash,
    n_draws: u32,
}

#[derive(Default)]
struct RealProofCaptureObserver {
    folds: Vec<ProverObservedFold>,
}

impl FriCommitObserver<SimdBackend, Blake2sMerkleChannel> for RealProofCaptureObserver {
    fn observe_inner_fold(
        &mut self,
        input: &LineEvaluation<SimdBackend>,
        alpha: SecureField,
        root: Blake2sHash,
        channel: &Blake2sChannel,
    ) {
        let log_size = input.domain().log_size();
        if !matches!(log_size, 9 | 6 | 3) {
            return;
        }
        let entry_words = (log_size == 6).then(|| {
            input
                .to_cpu()
                .values
                .columns
                .iter()
                .flat_map(|column| column.iter().map(|value| value.0))
                .collect()
        });
        self.folds.push(ProverObservedFold {
            log_size,
            entry_words,
            alpha,
            root,
            digest: channel.digest(),
            n_draws: channel.n_draws(),
        });
    }
}

impl RealProofCaptureObserver {
    fn finish(self) -> Result<(Vec<u32>, ObserverRound6), String> {
        let count = self.folds.len();
        let [previous, round6, round3]: [ProverObservedFold; 3] = self
            .folds
            .try_into()
            .map_err(|_| format!("observer captured {count} target folds; expected exactly 3"))?;
        if [previous.log_size, round6.log_size, round3.log_size] != [9, 6, 3] {
            return Err("observer did not capture the unique log9 -> log6 -> log3 window".into());
        }
        if Blake2sHasher::concat_and_hash(&previous.digest, &round6.root) != round6.digest {
            return Err("observer root6 transcript transition is disconnected".into());
        }
        let cursor35 = Blake2sHasher::concat_and_hash(&round6.digest, &round3.root);
        if cursor35 != round3.digest {
            return Err("observer root7 transcript transition is disconnected".into());
        }
        let entry_words = round6
            .entry_words
            .ok_or("observer did not retain the log6 FRI input")?;
        Ok((
            entry_words,
            ObserverRound6 {
                pre_root6: observer_state(previous.digest, previous.n_draws),
                root6_words: hash_words(round6.root),
                alpha6_words: secure_words(round6.alpha),
                cursor34: observer_state(round6.digest, round6.n_draws),
                root7_words: hash_words(round3.root),
                cursor35: observer_state(cursor35, 0),
                alpha7_words: secure_words(round3.alpha),
                cursor36: observer_state(round3.digest, round3.n_draws),
            },
        ))
    }
}

fn observer_state(digest: Blake2sHash, n_draws: u32) -> ObserverChannelState {
    ObserverChannelState {
        digest_words: hash_words(digest),
        n_draws,
    }
}

fn produce_real_observer_boundary(root: &std::path::Path) {
    use std::path::PathBuf;

    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::PcsConfig;
    use stwo_cairo_adapter::ProverInput;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
    use stwo_cairo_prover::prover::{prove_cairo_with_fri_observer, ChannelHash, ProverParameters};

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../../stwo_cairo_prover/test_data/test_prove_verify_all_opcode_components/prover_input.json",
    );
    let input: ProverInput = serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
    )
    .expect("deserialize all-opcodes ProverInput");
    let adapted_input_bytes =
        bincode::serialize(&input).expect("serialize all-opcodes ProverInput");
    let parameters = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config: PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: Some(24),
        },
        preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };
    let mut observer = RealProofCaptureObserver::default();
    let proof = prove_cairo_with_fri_observer::<SimdBackend, Blake2sMerkleChannel, _>(
        input,
        parameters,
        &mut observer,
    )
    .expect("prove observed all-opcodes Cairo fixture");
    let (entry_words, observed) = observer.finish().expect("seal observer round6 window");
    let sealed = crate::fri_round6_proof::tests::seal_proof(&proof, &adapted_input_bytes);
    let capture = crate::fri_round6_capture::encode_observer_seed(
        CaptureSource {
            observer: "stwo-cairo.production-simd-fri-observer.v1".into(),
            prover_input_sha256: crate::model::sha256_hex(&adapted_input_bytes),
            prover_input_bytes: adapted_input_bytes.len() as u64,
            observer_proof_shape_id: sealed.proof_shape.sha256.clone(),
        },
        derive_capture_shape(&proof).expect("derive observer capture shape"),
        entry_words,
        observed,
    )
    .expect("encode observer capture through production schema");

    std::fs::create_dir_all(root).expect("create real-proof output root");
    crate::fri_round6_proof::tests::write_boundary_bundle(
        &root.join("bundle"),
        &sealed,
        &adapted_input_bytes,
    );
    std::fs::write(root.join("capture.json"), capture).expect("write real observer capture");
}

struct ScratchRoot(std::path::PathBuf);

impl ScratchRoot {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time precedes Unix epoch")
            .as_nanos();
        Self(std::env::temp_dir().join(format!("{label}-{}-{nonce}", std::process::id())))
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for ScratchRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
