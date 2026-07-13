use stwo::core::vcs::blake2_hash::Blake2sHash;

use crate::fri_round6_capture::VerifiedCapture;
use crate::fri_round6_index::Source;

pub fn source_index(capture: Option<&VerifiedCapture>, exporter_sha256: &str) -> Source {
    match capture {
        None => Source {
            kind: "synthetic-layout-self-test",
            exporter_executable_sha256: exporter_sha256.to_owned(),
            capture_seed_sha256: None,
            capture: None,
            capture_shape: None,
            device_protocol_key: None,
            cairo_schedule_key: None,
        },
        Some(capture) => Source {
            kind: "hash-pinned-production-simd-observer-claim-unsealed",
            exporter_executable_sha256: exporter_sha256.to_owned(),
            capture_seed_sha256: Some(capture.capture_sha256.clone()),
            capture: Some(capture.source.clone()),
            capture_shape: Some(capture.shape.clone()),
            device_protocol_key: Some(format!("{:016x}", capture.device_protocol_key)),
            cairo_schedule_key: Some(format!("{:016x}", capture.cairo_schedule_key)),
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_observed(
    capture: &VerifiedCapture,
    root6: &[u32],
    alpha6: &[u32],
    cursor34: &[u32],
    root7: &[u32],
    alpha7: &[u32],
    cursor35: &[u32],
    cursor36: &[u32],
) -> Result<(), String> {
    let observed = &capture.observed;
    for (actual, expected, label) in [
        (root6, observed.root6_words.as_slice(), "root6"),
        (alpha6, observed.alpha6_words.as_slice(), "alpha6"),
        (
            cursor34,
            observed.cursor34_state_words.as_slice(),
            "cursor34 state",
        ),
        (root7, observed.root7_words.as_slice(), "root7"),
        (alpha7, observed.alpha7_words.as_slice(), "alpha7"),
        (
            cursor35,
            observed.cursor35_state_words.as_slice(),
            "cursor35 state",
        ),
        (
            cursor36,
            observed.cursor36_state_words.as_slice(),
            "cursor36 state",
        ),
    ] {
        if actual != expected {
            return Err(format!(
                "captured {label} does not match the independent CPU replay"
            ));
        }
    }
    Ok(())
}

pub fn words_hash(words: &[u32]) -> Result<Blake2sHash, String> {
    if words.len() != 8 {
        return Err("Blake2s digest encoding must contain eight words".into());
    }
    let mut bytes = [0; 32];
    for (chunk, word) in bytes.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    Ok(Blake2sHash(bytes))
}
