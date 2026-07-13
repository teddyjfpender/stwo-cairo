use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::artifact_io::M31_ROW_ENCODING;
use crate::legacy;
use crate::model;
use crate::streaming;

fn temporary_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "stwo-fixture-stream-test-{}-{nonce}",
        process::id()
    ))
}

fn store_words(root: &Path, words: &[u32]) -> (String, String, u64) {
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let digest = model::sha256_hex(&bytes);
    let relative = format!("chunks/sha256/{digest}.m31le");
    let path = root.join(&relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, &bytes).unwrap();
    (relative, digest, bytes.len() as u64)
}

fn tiny_fixture() -> legacy::Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../stwo/gpu-lab/cases/tiny/witness_pedersen_builtin.semantic.json");
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn fixture_index(root: &Path, corrupt_expected: bool) -> PathBuf {
    let fixture = tiny_fixture();
    let payload = &fixture.semantic_payload;
    let mut input_words = Vec::with_capacity(payload.row_count * 3);
    let mut expected_words = Vec::with_capacity(payload.row_count * 23);
    for row in 0..payload.row_count {
        input_words.extend([
            payload.inputs.segment_start[row],
            payload.inputs.enabler[row],
            payload.inputs.iota[row],
        ]);
        expected_words.extend((0..3).map(|word| payload.expected.output_columns[word][row]));
        expected_words.extend((0..14).map(|word| payload.expected.lookup_words[word][row]));
        expected_words.extend((0..6).map(|word| payload.expected.sub_words[word][row]));
    }
    if corrupt_expected {
        expected_words[0] = (expected_words[0] + 1) % model::M31_P;
    }
    let (table_path, table_sha, table_bytes) = store_words(root, &payload.tables.address_to_id);
    let (input_path, input_sha, input_bytes) = store_words(root, &input_words);
    let (expected_path, expected_sha, expected_bytes) = store_words(root, &expected_words);
    let row_count = payload.row_count as u64;
    let descriptor = |path: String, sha256: String, byte_len: u64, words_per_row| {
        json!({
            "encoding": M31_ROW_ENCODING,
            "path": path,
            "sha256": sha256,
            "row_start": 0,
            "row_count": row_count,
            "words_per_row": words_per_row,
            "byte_len": byte_len,
        })
    };
    let source_hash = "a".repeat(64);
    let fixture_id =
        format!("representative.witness_pedersen_builtin.prover-input-{source_hash}.v1");
    let index = json!({
        "schema_version": model::INDEX_SCHEMA,
        "fixture_id": fixture_id.clone(),
        "fixture_class": "representative",
        "proof_identity": {
            "kind": "real_prover_input_kernel_slice",
            "id": fixture_id,
            "full_proof": false,
            "source_prover_input_sha256": source_hash,
            "source_prover_input_bytes": 1024,
        },
        "semantic_identity": fixture.semantic_identity,
        "full_proof_semantic_hash": null,
        "boundary": fixture.boundary,
        "oracle": {
            "implementation": model::INDEX_ORACLE_IMPLEMENTATION,
            "version": model::INDEX_ORACLE_VERSION,
            "candidate_independent": true,
            "provenance": model::INDEX_ORACLE_PROVENANCE,
            "reference_closure_sha256": fixture.oracle.reference_closure_sha256,
            "reference_sources": fixture.oracle.reference_sources,
        },
        "exporter_executable_sha256": "b".repeat(64),
        "production_crosscheck": model::production_crosscheck().unwrap(),
        "semantic_payload": {
            "field": "M31",
            "row_count": row_count,
            "address_to_id": {
                "encoding": M31_ROW_ENCODING,
                "path": table_path,
                "sha256": table_sha,
                "element_count": payload.tables.address_to_id.len(),
                "byte_len": table_bytes,
            },
            "input_chunks": [descriptor(input_path, input_sha, input_bytes, 3)],
            "expected_chunks": [descriptor(expected_path, expected_sha, expected_bytes, 23)],
        },
    });
    let path = root.join(if corrupt_expected {
        "fixture-mutated.json"
    } else {
        "fixture.json"
    });
    fs::write(&path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    path
}

#[test]
fn streaming_index_is_deterministic_and_checked() {
    let root = temporary_root();
    fs::create_dir_all(&root).unwrap();
    let result = (|| {
        let fixture = fixture_index(&root, false);
        let first = root.join("oracle-a.json");
        let second = root.join("oracle-b.json");
        streaming::export(&fixture, &first, true, None).unwrap();
        streaming::export(&fixture, &second, true, None).unwrap();
        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
        let artifact: serde_json::Value =
            serde_json::from_slice(&fs::read(first).unwrap()).unwrap();
        assert_eq!(
            artifact["schema_version"],
            "stwo.gpu-lab.host-oracle-index.v2"
        );
        assert_eq!(artifact["validation"]["scalar_golden_rows"], 32);
        assert_eq!(
            artifact["validation"]["production_simd_crosschecked_rows"],
            0
        );
        assert_eq!(
            artifact["validation"]["words_compared_to_checked_fixture"],
            736
        );

        let mutated = fixture_index(&root, true);
        let error = streaming::export(&mutated, &root.join("must-not-seal.json"), true, None)
            .expect_err("validly hashed semantic mutation must fail the host gate");
        assert!(error.contains("differ from the canonical host export"));
    })();
    let _ = fs::remove_dir_all(&root);
    result
}
