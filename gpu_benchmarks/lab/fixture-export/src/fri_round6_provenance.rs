//! Fail-closed provenance preflight for captured FRI replay bundles.

use std::fs::{self, File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{canonical_value_hash, load_bounded, validate_sha256};

pub const SCHEMA: &str = "stwo.gpu-lab.fri-round6-provenance.v1";
const ADAPTER_RUN_SCHEMA: &str = "stwo.gpu-lab.pie-adapter-run.v1";
const SOURCE_CLOSURE_SCHEMA: &str = "stwo.gpu-lab.source-closure.v1";
const PROOF_SHAPE_SCHEMA: &str = "stwo.gpu-lab.cairo-proof-shape.v1";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_DOCUMENT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PIE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PROVER_INPUT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PROOF_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SOURCE_FILES: usize = 4096;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSeal {
    pub kind: String,
    pub path: String,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterBinding {
    pub run_record: ArtifactSeal,
    pub invocation: ArtifactSeal,
    pub executable: ArtifactSeal,
    pub source_closure: ArtifactSeal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PcsShape {
    pub pow_bits: u32,
    pub log_blowup_factor: u32,
    pub log_last_layer_degree_bound: u32,
    pub n_queries: u64,
    pub fold_step: u32,
    pub lifting_log_size: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProofShape {
    pub schema_version: String,
    pub pcs: PcsShape,
    pub channel_salt: u32,
    pub preprocessed_trace_variant_sha256: String,
    pub component_slots: u64,
    pub component_enable_bits_sha256: String,
    pub component_log_sizes: Vec<u32>,
    pub trace_column_log_sizes: Vec<Vec<u32>>,
    pub public_data_word_counts: [u64; 3],
    pub interaction_claim_felts: u64,
    pub commitment_trees: u64,
    pub sampled_value_counts: Vec<Vec<u64>>,
    pub decommitment_trees: u64,
    pub queried_value_counts: Vec<Vec<u64>>,
    pub fri_inner_layers: u64,
    pub fri_witness_counts: Vec<u64>,
    pub fri_last_layer_coefficients: u64,
    pub unsorted_query_locations: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProofShapeSeal {
    pub sha256: String,
    pub shape: ProofShape,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceManifest {
    pub schema_version: String,
    pub status: String,
    pub production_admissible: bool,
    pub source_pie: ArtifactSeal,
    pub adapted_prover_input: ArtifactSeal,
    pub adapter: AdapterBinding,
    pub extended_cairo_proof_bincode: ArtifactSeal,
    pub canonical_cairo_transport: ArtifactSeal,
    pub verifier_source_closure: ArtifactSeal,
    pub proof_shape: ProofShapeSeal,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterRunRecord {
    schema_version: String,
    completed: bool,
    source_pie: ArtifactIdentity,
    adapted_prover_input: ArtifactIdentity,
    invocation: ArtifactIdentity,
    adapter_executable: ArtifactIdentity,
    adapter_source_closure: ArtifactIdentity,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceClosure {
    schema_version: String,
    component: String,
    git_commit: String,
    sources: Vec<SourceIdentity>,
    closure_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceIdentity {
    path: String,
    byte_length: u64,
    sha256: String,
}

pub struct VerifiedProvenanceInputs {
    pub manifest_sha256: String,
    pub proof_shape_sha256: String,
    pub bundle_root: PathBuf,
    pub extended_cairo_proof_bincode: ArtifactSeal,
    pub canonical_cairo_transport: ArtifactSeal,
    pub verifier_source_closure: ArtifactSeal,
    pub expected_proof_shape: ProofShape,
}

pub fn preflight(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
) -> Result<VerifiedProvenanceInputs, String> {
    validate_sha256(expected_manifest_sha256, "FRI provenance manifest sha256")?;
    reject_symlink(manifest_path, "FRI provenance manifest")?;
    let (bytes, manifest_sha256) =
        load_bounded(manifest_path, MAX_MANIFEST_BYTES, "FRI provenance manifest")?;
    if manifest_sha256 != expected_manifest_sha256 {
        return Err(format!(
            "FRI provenance manifest sha256 {manifest_sha256} != required {expected_manifest_sha256}"
        ));
    }
    let manifest: ProvenanceManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse FRI provenance manifest: {error}"))?;
    validate_header(&manifest)?;
    let root = manifest_path
        .parent()
        .ok_or("FRI provenance manifest has no parent directory")?
        .canonicalize()
        .map_err(|error| format!("resolve FRI provenance bundle root: {error}"))?;

    verify_artifact(
        &root,
        &manifest.source_pie,
        "cairo-pie-zip",
        MAX_PIE_BYTES,
        false,
    )?;
    verify_artifact(
        &root,
        &manifest.adapted_prover_input,
        "stwo-prover-input-bincode-v1",
        MAX_PROVER_INPUT_BYTES,
        false,
    )?;
    let run = verify_artifact(
        &root,
        &manifest.adapter.run_record,
        "adapter-run-json-v1",
        MAX_DOCUMENT_BYTES,
        true,
    )?;
    verify_artifact(
        &root,
        &manifest.adapter.invocation,
        "adapter-invocation-json-v1",
        MAX_DOCUMENT_BYTES,
        false,
    )?;
    verify_artifact(
        &root,
        &manifest.adapter.executable,
        "adapter-executable",
        MAX_EXECUTABLE_BYTES,
        false,
    )?;
    let adapter_sources = verify_artifact(
        &root,
        &manifest.adapter.source_closure,
        "adapter-source-closure-json-v1",
        MAX_DOCUMENT_BYTES,
        true,
    )?;
    verify_artifact(
        &root,
        &manifest.extended_cairo_proof_bincode,
        "extended-cairo-proof-bincode-v1",
        MAX_PROOF_BYTES,
        false,
    )?;
    verify_artifact(
        &root,
        &manifest.canonical_cairo_transport,
        "canonical-cairo-proof-felts-be32-v1",
        MAX_PROOF_BYTES,
        false,
    )?;
    let verifier_sources = verify_artifact(
        &root,
        &manifest.verifier_source_closure,
        "verifier-source-closure-json-v1",
        MAX_DOCUMENT_BYTES,
        true,
    )?;

    let run: AdapterRunRecord = serde_json::from_slice(&run)
        .map_err(|error| format!("parse adapter run record: {error}"))?;
    validate_adapter_run(&run, &manifest)?;
    validate_source_closure(&adapter_sources, "adapter")?;
    validate_source_closure(&verifier_sources, "verifier")?;
    validate_shape(&manifest.proof_shape)?;

    Ok(VerifiedProvenanceInputs {
        manifest_sha256,
        proof_shape_sha256: manifest.proof_shape.sha256,
        bundle_root: root,
        extended_cairo_proof_bincode: manifest.extended_cairo_proof_bincode,
        canonical_cairo_transport: manifest.canonical_cairo_transport,
        verifier_source_closure: manifest.verifier_source_closure,
        expected_proof_shape: manifest.proof_shape.shape,
    })
}

fn validate_header(manifest: &ProvenanceManifest) -> Result<(), String> {
    if manifest.schema_version != SCHEMA {
        return Err(format!(
            "unsupported FRI provenance schema: {}",
            manifest.schema_version
        ));
    }
    if manifest.status != "captured-unsealed" || manifest.production_admissible {
        return Err("FRI provenance must remain captured-unsealed and non-admissible".into());
    }
    Ok(())
}

fn validate_adapter_run(
    run: &AdapterRunRecord,
    manifest: &ProvenanceManifest,
) -> Result<(), String> {
    if run.schema_version != ADAPTER_RUN_SCHEMA || !run.completed {
        return Err("adapter run is incomplete or has the wrong schema".into());
    }
    for (label, actual, seal) in [
        ("source PIE", &run.source_pie, &manifest.source_pie),
        (
            "adapted ProverInput",
            &run.adapted_prover_input,
            &manifest.adapted_prover_input,
        ),
        (
            "adapter invocation",
            &run.invocation,
            &manifest.adapter.invocation,
        ),
        (
            "adapter executable",
            &run.adapter_executable,
            &manifest.adapter.executable,
        ),
        (
            "adapter source closure",
            &run.adapter_source_closure,
            &manifest.adapter.source_closure,
        ),
    ] {
        if actual.byte_length != seal.byte_length || actual.sha256 != seal.sha256 {
            return Err(format!("adapter run does not bind the sealed {label}"));
        }
    }
    Ok(())
}

fn validate_source_closure(bytes: &[u8], expected_component: &str) -> Result<(), String> {
    let closure: SourceClosure = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse {expected_component} source closure: {error}"))?;
    if closure.schema_version != SOURCE_CLOSURE_SCHEMA || closure.component != expected_component {
        return Err(format!(
            "wrong {expected_component} source closure identity"
        ));
    }
    if closure.git_commit.len() != 40
        || !closure
            .git_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{expected_component} git commit must be 40 lowercase hex characters"
        ));
    }
    if closure.sources.is_empty() || closure.sources.len() > MAX_SOURCE_FILES {
        return Err(format!(
            "{expected_component} source closure has invalid cardinality"
        ));
    }
    let mut previous = None;
    for source in &closure.sources {
        validate_relative_path(&source.path, "source closure path")?;
        validate_sha256(&source.sha256, "source closure sha256")?;
        if source.byte_length == 0
            || previous.is_some_and(|path: &str| path >= source.path.as_str())
        {
            return Err(format!(
                "{expected_component} sources must be nonempty and strictly sorted"
            ));
        }
        previous = Some(source.path.as_str());
    }
    validate_sha256(&closure.closure_sha256, "source closure digest")?;
    let value = serde_json::json!({
        "schema_version": closure.schema_version,
        "component": closure.component,
        "git_commit": closure.git_commit,
        "sources": closure.sources,
    });
    if canonical_value_hash(&value)? != closure.closure_sha256 {
        return Err(format!(
            "{expected_component} source closure digest mismatch"
        ));
    }
    Ok(())
}

fn validate_shape(seal: &ProofShapeSeal) -> Result<(), String> {
    validate_sha256(&seal.sha256, "proof shape sha256")?;
    if seal.shape.schema_version != PROOF_SHAPE_SCHEMA {
        return Err("wrong proof shape schema".into());
    }
    validate_sha256(
        &seal.shape.preprocessed_trace_variant_sha256,
        "preprocessed trace variant sha256",
    )?;
    validate_sha256(
        &seal.shape.component_enable_bits_sha256,
        "component enable bits sha256",
    )?;
    let value = serde_json::to_value(&seal.shape)
        .map_err(|error| format!("serialize proof shape: {error}"))?;
    if canonical_value_hash(&value)? != seal.sha256 {
        return Err("proof shape digest mismatch".into());
    }
    Ok(())
}

fn verify_artifact(
    root: &Path,
    seal: &ArtifactSeal,
    expected_kind: &str,
    maximum: u64,
    load: bool,
) -> Result<Vec<u8>, String> {
    if seal.kind != expected_kind {
        return Err(format!(
            "artifact {} has kind {}, expected {expected_kind}",
            seal.path, seal.kind
        ));
    }
    validate_sha256(&seal.sha256, &format!("{} sha256", seal.path))?;
    if seal.byte_length == 0 || seal.byte_length > maximum {
        return Err(format!(
            "artifact {} has invalid byte length {}",
            seal.path, seal.byte_length
        ));
    }
    let path = resolve_bundle_path(root, &seal.path)?;
    let mut file =
        File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let before = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?;
    let live = fs::metadata(&path).map_err(|error| format!("stat {}: {error}", path.display()))?;
    if !before.is_file()
        || fingerprint(&before) != fingerprint(&live)
        || before.len() != seal.byte_length
    {
        return Err(format!(
            "artifact {} is not the declared stable regular file",
            seal.path
        ));
    }
    let capacity = if load { seal.byte_length as usize } else { 0 };
    let mut bytes = Vec::with_capacity(capacity);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        if load {
            bytes.extend_from_slice(&buffer[..count]);
        }
    }
    let after = file
        .metadata()
        .map_err(|error| format!("restat {}: {error}", path.display()))?;
    if fingerprint(&before) != fingerprint(&after)
        || format!("{:x}", hasher.finalize()) != seal.sha256
    {
        return Err(format!(
            "artifact {} changed or failed its sha256 seal",
            seal.path
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind {}: {error}", path.display()))?;
    Ok(bytes)
}

fn resolve_bundle_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    validate_relative_path(relative, "artifact path")?;
    let joined = root.join(relative);
    reject_symlink(&joined, "provenance artifact")?;
    let resolved = joined
        .canonicalize()
        .map_err(|error| format!("resolve {}: {error}", joined.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!("artifact path escapes bundle root: {relative}"));
    }
    Ok(resolved)
}

fn validate_relative_path(value: &str, label: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(format!("{label} must be a normalized relative path"));
    }
    Ok(())
}

fn reject_symlink(path: &Path, label: &str) -> Result<(), String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("stat {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{label} cannot be a symlink: {}", path.display()));
    }
    Ok(())
}

fn fingerprint(metadata: &Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}
