use std::fs;
use std::path::Path;

use crate::artifact_io::{write_immutable_bytes, write_immutable_json};
use crate::fri_round6::{build, capture_context, synthetic_context, validate_pair, Artifact, Case};
use crate::fri_round6_capture::VerifiedCapture;
use crate::model::{current_executable_sha256, load_bounded, validate_sha256};

const MAX_INDEX_BYTES: u64 = 1024 * 1024;

pub fn export_synthetic(output_dir: &Path) -> Result<(), String> {
    let executable_sha256 = current_executable_sha256()?;
    let context = synthetic_context(executable_sha256.clone())?;
    let (primary, hostile) = build_pair(&context)?;
    write_case(output_dir, "synthetic-layout", Case::Primary, &primary)?;
    write_case(output_dir, "synthetic-layout", Case::Hostile, &hostile)?;
    ensure_executable_unchanged(&executable_sha256)
}

pub fn export_capture(
    output_dir: &Path,
    capture: &VerifiedCapture,
    expected_executable_sha256: &str,
) -> Result<(), String> {
    let executable_sha256 = authenticate_executable(expected_executable_sha256)?;
    let context = capture_context(capture, executable_sha256.clone())?;
    let (primary, hostile) = build_pair(&context)?;
    write_case(output_dir, "captured-unsealed", Case::Primary, &primary)?;
    write_case(output_dir, "captured-unsealed", Case::Hostile, &hostile)?;
    ensure_executable_unchanged(&executable_sha256)
}

/// Rebuild both production artifacts from the authenticated capture and compare
/// every payload and index byte. A matching digest declared by an arbitrary
/// index is insufficient: the canonical CPU oracle is the acceptance source.
pub fn validate_capture_dir(
    output_dir: &Path,
    capture: &VerifiedCapture,
    expected_executable_sha256: &str,
) -> Result<(), String> {
    reject_symlink_or_non_directory(output_dir)?;
    let executable_sha256 = authenticate_executable(expected_executable_sha256)?;
    let context = capture_context(capture, executable_sha256.clone())?;
    let (primary, hostile) = build_pair(&context)?;
    verify_case(output_dir, Case::Primary, &primary)?;
    verify_case(output_dir, Case::Hostile, &hostile)?;
    ensure_executable_unchanged(&executable_sha256)?;
    println!(
        "FRI_ROUND6_CAPTURE_VALIDATE=PASS production_admissible=false capture_sha256={} exporter_sha256={executable_sha256}",
        capture.capture_sha256
    );
    Ok(())
}

fn build_pair(
    context: &crate::fri_round6::BuildContext<'_>,
) -> Result<(Artifact, Artifact), String> {
    let primary = build(context, Case::Primary)?;
    let hostile = build(context, Case::Hostile)?;
    validate_pair(&primary, &hostile)?;
    Ok((primary, hostile))
}

fn write_case(
    output_dir: &Path,
    family: &str,
    case: Case,
    artifact: &Artifact,
) -> Result<(), String> {
    let stem = format!("fri-round6-{family}-{}", case.name());
    write_immutable_bytes(
        &output_dir.join(format!("{stem}.payload.bin")),
        &artifact.payload,
    )?;
    write_immutable_json(
        &output_dir.join(format!("{stem}.index.json")),
        &artifact.index,
    )
}

fn verify_case(output_dir: &Path, case: Case, artifact: &Artifact) -> Result<(), String> {
    let stem = format!("fri-round6-captured-unsealed-{}", case.name());
    let payload_path = output_dir.join(format!("{stem}.payload.bin"));
    let index_path = output_dir.join(format!("{stem}.index.json"));
    reject_symlink_or_non_file(&payload_path, "FRI payload")?;
    reject_symlink_or_non_file(&index_path, "FRI index")?;
    let (payload, _) = load_bounded(&payload_path, artifact.payload.len() as u64, "FRI payload")?;
    if payload != artifact.payload {
        return Err(format!(
            "{} is not the canonical payload rebuilt from the authenticated capture",
            payload_path.display()
        ));
    }
    let (index, _) = load_bounded(&index_path, MAX_INDEX_BYTES, "FRI index")?;
    let mut expected = serde_json::to_vec_pretty(&artifact.index)
        .map_err(|error| format!("serialize canonical FRI index: {error}"))?;
    expected.push(b'\n');
    if index != expected {
        return Err(format!(
            "{} is not the canonical index rebuilt from the authenticated capture",
            index_path.display()
        ));
    }
    Ok(())
}

fn authenticate_executable(expected: &str) -> Result<String, String> {
    validate_sha256(expected, "expected exporter executable sha256")?;
    let actual = current_executable_sha256()?;
    if actual != expected {
        return Err(format!(
            "exporter executable sha256 {actual} != required {expected}"
        ));
    }
    Ok(actual)
}

fn ensure_executable_unchanged(expected: &str) -> Result<(), String> {
    if current_executable_sha256()? != expected {
        return Err("exporter executable changed during FRI fixture operation".into());
    }
    Ok(())
}

fn reject_symlink_or_non_directory(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect FRI fixture directory {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "FRI fixture directory {} is a symlink or non-directory",
            path.display()
        ));
    }
    Ok(())
}

fn reject_symlink_or_non_file(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "{label} {} is a symlink or non-file",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn canonical_regeneration_rejects_altered_bytes() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("stwo-fri-gate-test-{}-{nonce}", std::process::id()));
        let context = synthetic_context("11".repeat(32)).unwrap();
        let artifact = build(&context, Case::Primary).unwrap();
        write_case(&root, "captured-unsealed", Case::Primary, &artifact).unwrap();
        verify_case(&root, Case::Primary, &artifact).unwrap();

        let payload = root.join("fri-round6-captured-unsealed-primary.payload.bin");
        let mut bytes = fs::read(&payload).unwrap();
        bytes[0] ^= 1;
        fs::write(&payload, bytes).unwrap();
        let error = verify_case(&root, Case::Primary, &artifact).err().unwrap();
        assert!(error.contains("canonical payload"));

        fs::write(&payload, &artifact.payload).unwrap();
        let index = root.join("fri-round6-captured-unsealed-primary.index.json");
        let mut bytes = fs::read(&index).unwrap();
        bytes[0] ^= 1;
        fs::write(&index, bytes).unwrap();
        let error = verify_case(&root, Case::Primary, &artifact).err().unwrap();
        assert!(error.contains("canonical index"));
        let _ = fs::remove_dir_all(root);
    }
}
