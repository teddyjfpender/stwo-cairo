use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

fn shape() -> CaptureShape {
    CaptureShape {
        circle_log_size: 24,
        claim_enable_felts: 2,
        claim_log_size_felts: 2,
        claim_public_data_felts: 3,
        interaction_claim_felts: 5,
        oods_sampled_values_felts: 7,
        interaction_pow_bits: 20,
        pcs: PcsShape {
            pow_bits: 24,
            log_blowup_factor: 1,
            n_queries: 13,
            log_last_layer_degree_bound: 3,
            fold_step: 3,
            lifting_log_size: 24,
        },
        fri_tree_count: 8,
    }
}

fn state(cursor: u32, chain: u64, n_draws: u32) -> Vec<u32> {
    let mut words = vec![cursor.wrapping_mul(0x10203); 8];
    words.extend([
        n_draws,
        cursor,
        0,
        0,
        chain as u32,
        (chain >> 32) as u32,
        0,
        0,
    ]);
    words
}

fn seed() -> CaptureSeed {
    let shape = shape();
    let operations = cairo_operations(&shape).unwrap();
    let schedule = Blake2sTranscriptSchedule::new(
        TranscriptStart::Default,
        operations.clone(),
        MAX_REJECTION_ROUNDS,
    )
    .unwrap();
    let prefix = independent_prefix_chains(&operations);
    let chains: [u64; 5] = prefix[32..=36].try_into().unwrap();
    let cursor32_state_words = state(32, chains[0], 0);
    let entry_pong_words = (0..256).map(|word| word as u32).collect::<Vec<_>>();
    CaptureSeed {
        schema_version: CAPTURE_SCHEMA.into(),
        source: CaptureSource {
            observer: "stwo-cairo.production-simd-fri-observer.v1".into(),
            prover_input_sha256: "22".repeat(32),
            prover_input_bytes: 1_024,
            observer_proof_shape_id: "captured-test-shape".into(),
        },
        shape: shape.clone(),
        schedule: ScheduleSeal {
            device_protocol_key: format!("{:016x}", schedule.protocol_key()),
            cairo_schedule_key: format!(
                "{:016x}",
                compute_cairo_schedule_key(schedule.protocol_key(), &shape)
            ),
            c32: format!("{:016x}", chains[0]),
            c33: format!("{:016x}", chains[1]),
            c34: format!("{:016x}", chains[2]),
            c35: format!("{:016x}", chains[3]),
            c36: format!("{:016x}", chains[4]),
        },
        cursor32_state_words_sha256: sha256_hex(&word_bytes(&cursor32_state_words)),
        cursor32_state_words,
        entry_pong_words_sha256: sha256_hex(&word_bytes(&entry_pong_words)),
        entry_pong_words,
        observed: Observed {
            root6_words: vec![1; 8],
            alpha6_words: vec![2; 4],
            cursor34_state_words: state(34, chains[2], 1),
            root7_words: vec![3; 8],
            alpha7_words: vec![4; 4],
            cursor35_state_words: state(35, chains[3], 0),
            cursor36_state_words: state(36, chains[4], 1),
        },
    }
}

#[test]
fn valid_seed_rebuilds_full_schedule_closure() {
    let verified = verify(seed(), "33".repeat(32)).unwrap();
    assert_eq!(verified.capture_sha256, "33".repeat(32));
    assert_eq!(
        verified.source.observer_proof_shape_id,
        "captured-test-shape"
    );
    assert_eq!(verified.chains.len(), 5);
}

#[test]
fn observer_encoder_roundtrips_through_the_capture_verifier() {
    let source = CaptureSource {
        observer: "stwo-cairo.production-simd-fri-observer.v1".into(),
        prover_input_sha256: "22".repeat(32),
        prover_input_bytes: 1_024,
        observer_proof_shape_id: "44".repeat(32),
    };
    let channel = |word, n_draws| ObserverChannelState {
        digest_words: [word; 8],
        n_draws,
    };
    let bytes = encode_observer_seed(
        source,
        shape(),
        (0..256).map(|word| word as u32).collect(),
        ObserverRound6 {
            pre_root6: channel(1, 2),
            root6_words: [2; 8],
            alpha6_words: [3; 4],
            cursor34: channel(4, 1),
            root7_words: [5; 8],
            cursor35: channel(6, 0),
            alpha7_words: [7; 4],
            cursor36: channel(6, 1),
        },
    )
    .unwrap();
    let seed: CaptureSeed = serde_json::from_slice(&bytes).unwrap();
    let verified = verify(seed, sha256_hex(&bytes)).unwrap();
    assert_eq!(verified.source.observer_proof_shape_id, "44".repeat(32));
    assert_eq!(verified.cursor32[8], 2);
    assert_eq!(verified.observed.root6_words, [2; 8]);
}

#[test]
fn tampered_schedule_prefix_is_rejected() {
    let mut capture = seed();
    capture.schedule.c34 = "0000000000000000".into();
    let error = verify(capture, "33".repeat(32)).err().unwrap();
    assert!(error.contains("C32-C36"));
}

#[test]
fn tampered_cursor_chain_is_rejected() {
    let mut capture = seed();
    capture.cursor32_state_words[12] ^= 1;
    capture.cursor32_state_words_sha256 = sha256_hex(&word_bytes(&capture.cursor32_state_words));
    let error = verify(capture, "33".repeat(32)).err().unwrap();
    assert!(error.contains("cursor32 control"));
}

#[test]
fn parser_rejects_unknown_capture_fields() {
    let mut document = serde_json::to_value(seed()).unwrap();
    document["unreviewed"] = serde_json::json!(true);
    let error = serde_json::from_value::<CaptureSeed>(document)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("unknown field"));
}

#[test]
fn load_requires_the_exact_capture_file_hash() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "stwo-fri-capture-test-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("capture.json");
    let bytes = serde_json::to_vec_pretty(&seed()).unwrap();
    fs::write(&path, &bytes).unwrap();
    let error = load(&path, &"00".repeat(32)).err().unwrap();
    assert!(error.contains("!= required"));
    load(&path, &sha256_hex(&bytes)).unwrap();
    let _ = fs::remove_dir_all(root);
}
