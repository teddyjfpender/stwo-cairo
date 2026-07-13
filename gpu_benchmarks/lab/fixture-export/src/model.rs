use std::fs;
use std::path::Path;

pub use crate::semantic_support::{canonical_value_hash, sha256_hex, M31_P};
use serde::{Deserialize, Serialize};
pub const INLINE_SCHEMA: &str = "stwo.gpu-lab.semantic-fixture.v1";
pub const INDEX_SCHEMA: &str = "stwo.gpu-lab.semantic-fixture-index.v2";
pub const FIXTURE_ID: &str = "tiny.witness_pedersen_builtin.v1";
pub const FIXTURE_CLASS: &str = "tiny";
pub const OPERATION: &str = "cairo.witness.pedersen_builtin";
pub const ORACLE_IMPLEMENTATION: &str = "gpu-lab/tools/lab.py:pedersen_oracle";
pub const ORACLE_VERSION: &str = "candidate-free-scalar-v2";
pub const ORACLE_PROVENANCE: &str = "reviewed candidate-free PedersenBuiltin slab descriptor";
pub const INDEX_ORACLE_IMPLEMENTATION: &str = "stwo-cairo/gpu_benchmarks/lab/fixture-export";
pub const INDEX_ORACLE_VERSION: &str = "independent-scalar-plus-production-simd-v3";
pub const INDEX_ORACLE_PROVENANCE: &str =
    "candidate-free scalar golden cross-checked against production PedersenBuiltin SIMD bytes";
pub const REFERENCE_EVALUATOR_PATH: &str =
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/pedersen_builtin_semantics.rs";
pub const REFERENCE_DESCRIPTOR_PATH: &str =
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/semantics/pedersen_builtin.slab-semantics.v1.json";
pub const REFERENCE_SUPPORT_PATH: &str =
    "stwo-cairo/gpu_benchmarks/lab/fixture-export/src/semantic_support.rs";
const REFERENCE_DESCRIPTOR_BYTES: &[u8] =
    include_bytes!("../semantics/pedersen_builtin.slab-semantics.v1.json");
const REFERENCE_EVALUATOR_BYTES: &[u8] = include_bytes!("pedersen_builtin_semantics.rs");
const REFERENCE_SUPPORT_BYTES: &[u8] = include_bytes!("semantic_support.rs");

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProofIdentity {
    pub kind: String,
    pub id: String,
    pub full_proof: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_prover_input_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_prover_input_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticIdentity {
    pub kind: String,
    pub schema: String,
    pub scope: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub operation: String,
    pub entry: String,
    pub exit: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureOracle {
    pub implementation: String,
    pub version: String,
    pub candidate_independent: bool,
    pub provenance: String,
    pub reference_closure_sha256: String,
    pub reference_sources: Vec<ReferenceSource>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSource {
    pub role: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceClosure {
    pub closure_sha256: String,
    pub sources: Vec<SourceIdentity>,
}

#[derive(Debug, Serialize)]
pub struct OracleIdentity {
    pub engine: &'static str,
    pub golden_evaluator_path: &'static str,
    pub candidate_gpu_executed: bool,
    pub candidate_recording_used: bool,
}

pub struct CommonFixture<'a> {
    pub fixture_id: &'a str,
    pub fixture_class: &'a str,
    pub proof_identity: &'a ProofIdentity,
    pub semantic_identity: &'a SemanticIdentity,
    pub full_proof_semantic_hash: &'a Option<String>,
    pub boundary: &'a Boundary,
    pub oracle: &'a FixtureOracle,
}

pub fn load_bounded(path: &Path, maximum: u64, label: &str) -> Result<(Vec<u8>, String), String> {
    let size = fs::metadata(path)
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if size > maximum {
        return Err(format!(
            "{label} is {size} bytes; maximum is {maximum}; use the chunk-index format"
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let digest = sha256_hex(&bytes);
    Ok((bytes, digest))
}

pub fn current_executable_sha256() -> Result<String, String> {
    const MAX_EXECUTABLE_BYTES: u64 = 128 * 1024 * 1024;
    let path = std::env::current_exe()
        .map_err(|error| format!("resolve current exporter executable: {error}"))?;
    load_bounded(&path, MAX_EXECUTABLE_BYTES, "exporter executable").map(|(_, digest)| digest)
}

pub fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

pub fn reference_evaluator() -> Result<(Vec<ReferenceSource>, String), String> {
    let mut sources = Vec::new();
    for (role, relative, bytes) in [
        (
            "semantic_descriptor",
            REFERENCE_DESCRIPTOR_PATH,
            REFERENCE_DESCRIPTOR_BYTES,
        ),
        (
            "golden_evaluator",
            REFERENCE_EVALUATOR_PATH,
            REFERENCE_EVALUATOR_BYTES,
        ),
        (
            "golden_support",
            REFERENCE_SUPPORT_PATH,
            REFERENCE_SUPPORT_BYTES,
        ),
    ] {
        sources.push(ReferenceSource {
            role: role.into(),
            path: relative.into(),
            sha256: sha256_hex(bytes),
        });
    }
    let value = serde_json::to_value(&sources)
        .map_err(|error| format!("serialize reference evaluator closure: {error}"))?;
    let closure = canonical_value_hash(&value)?;
    Ok((sources, closure))
}

/// Compile-time closure for the code that executes and checks production SIMD.
/// A stale exporter therefore records its old sources and fails live validation.
pub fn production_crosscheck() -> Result<SourceClosure, String> {
    let mut sources = [
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.lock", include_bytes!("../Cargo.lock")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/Cargo.toml", include_bytes!("../Cargo.toml")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/rust-toolchain.toml", include_bytes!("../rust-toolchain.toml")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/semantics/pedersen_builtin.slab-semantics.v1.json", include_bytes!("../semantics/pedersen_builtin.slab-semantics.v1.json")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/artifact_io.rs", include_bytes!("artifact_io.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/main.rs", include_bytes!("main.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/model.rs", include_bytes!("model.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/oracle.rs", include_bytes!("oracle.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/pedersen_builtin_semantics.rs", include_bytes!("pedersen_builtin_semantics.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/producer.rs", include_bytes!("producer.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/semantic_support.rs", include_bytes!("semantic_support.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/source_snapshot.rs", include_bytes!("source_snapshot.rs")),
        source("stwo-cairo/gpu_benchmarks/lab/fixture-export/src/streaming.rs", include_bytes!("streaming.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/adapter/src/memory.rs", include_bytes!("../../../../stwo_cairo_prover/crates/adapter/src/memory.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/common/src/builtins.rs", include_bytes!("../../../../stwo_cairo_prover/crates/common/src/builtins.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/common/src/preprocessed_columns/preprocessed_trace.rs", include_bytes!("../../../../stwo_cairo_prover/crates/common/src/preprocessed_columns/preprocessed_trace.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/prover/Cargo.toml", include_bytes!("../../../../stwo_cairo_prover/crates/prover/Cargo.toml")),
        source("stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/memory_address_to_id.rs", include_bytes!("../../../../stwo_cairo_prover/crates/prover/src/witness/components/memory_address_to_id.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_aggregator_window_bits_18.rs", include_bytes!("../../../../stwo_cairo_prover/crates/prover/src/witness/components/pedersen_aggregator_window_bits_18.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs", include_bytes!("../../../../stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin.rs")),
        source("stwo-cairo/stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin/gpu_lab_oracle_bridge.rs", include_bytes!("../../../../stwo_cairo_prover/crates/prover/src/witness/components/pedersen_builtin/gpu_lab_oracle_bridge.rs")),
        source("stwo/crates/stwo/src/core/fields/m31.rs", include_bytes!("../../../../../stwo/crates/stwo/src/core/fields/m31.rs")),
    ];
    sources.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    let value = serde_json::to_value(&sources)
        .map_err(|error| format!("serialize production crosscheck closure: {error}"))?;
    Ok(SourceClosure {
        closure_sha256: canonical_value_hash(&value)?,
        sources: sources.into(),
    })
}

fn source(path: &str, bytes: &[u8]) -> SourceIdentity {
    SourceIdentity {
        path: path.into(),
        sha256: sha256_hex(bytes),
    }
}

pub fn validate_common(common: CommonFixture<'_>, inline: bool) -> Result<(), String> {
    if inline && (common.fixture_id != FIXTURE_ID || common.fixture_class != FIXTURE_CLASS) {
        return Err(format!(
            "wrong inline fixture identity/class: {}/{}",
            common.fixture_id, common.fixture_class
        ));
    }
    if !matches!(
        common.fixture_class,
        "tiny" | "representative" | "stress" | "unseen-same-class"
    ) {
        return Err(format!(
            "unsupported fixture class: {}",
            common.fixture_class
        ));
    }
    if common.fixture_id.is_empty()
        || common.proof_identity.kind.is_empty()
        || common.proof_identity.id != common.fixture_id
    {
        return Err("fixture/proof identity is empty or inconsistent".into());
    }
    if inline {
        if common.proof_identity.kind != "synthetic_kernel_slice"
            || common.proof_identity.full_proof
            || common.proof_identity.source_prover_input_sha256.is_some()
            || common.proof_identity.source_prover_input_bytes.is_some()
        {
            return Err("tiny fixture proof identity changed".into());
        }
    } else {
        match common.proof_identity.kind.as_str() {
            "synthetic_kernel_slice" => {
                if common.proof_identity.source_prover_input_sha256.is_some()
                    || common.proof_identity.source_prover_input_bytes.is_some()
                {
                    return Err("synthetic fixture cannot name a prover-input source".into());
                }
            }
            "real_prover_input_kernel_slice" => {
                let source = common
                    .proof_identity
                    .source_prover_input_sha256
                    .as_deref()
                    .ok_or("real fixture must bind its source ProverInput")?;
                validate_sha256(source, "source ProverInput sha256")?;
                if common.proof_identity.source_prover_input_bytes == Some(0) {
                    return Err("real fixture source ProverInput byte length is zero".into());
                }
                common
                    .proof_identity
                    .source_prover_input_bytes
                    .ok_or("real fixture must bind its source ProverInput byte length")?;
            }
            kind => return Err(format!("unsupported proof identity kind: {kind}")),
        }
        if common.proof_identity.full_proof {
            return Err("kernel-slice fixture cannot claim to be a full proof".into());
        }
    }
    if common.boundary.operation != OPERATION
        || common.boundary.entry != crate::pedersen_builtin_semantics::ENTRY_BOUNDARY
        || common.boundary.exit != crate::pedersen_builtin_semantics::EXIT_BOUNDARY
    {
        return Err("fixture transcript boundary changed".into());
    }
    let (oracle_implementation, oracle_version, oracle_provenance) = if inline {
        (ORACLE_IMPLEMENTATION, ORACLE_VERSION, ORACLE_PROVENANCE)
    } else {
        (
            INDEX_ORACLE_IMPLEMENTATION,
            INDEX_ORACLE_VERSION,
            INDEX_ORACLE_PROVENANCE,
        )
    };
    if common.oracle.implementation != oracle_implementation
        || common.oracle.version != oracle_version
        || !common.oracle.candidate_independent
        || common.oracle.provenance != oracle_provenance
    {
        return Err("fixture oracle identity/provenance changed".into());
    }
    if common.full_proof_semantic_hash.is_some() {
        return Err("kernel-slice fixture cannot claim a full-proof semantic hash".into());
    }
    let expected_semantics = crate::pedersen_builtin_semantics::semantic_identity()?;
    if common.semantic_identity != &expected_semantics {
        return Err("kernel-slice semantic identity differs from reviewed descriptor".into());
    }
    let (sources, closure) = reference_evaluator()?;
    validate_sha256(
        &common.oracle.reference_closure_sha256,
        "reference closure hash",
    )?;
    if common.oracle.reference_sources != sources
        || common.oracle.reference_closure_sha256 != closure
    {
        return Err("golden evaluator source closure differs from reviewed sources".into());
    }
    Ok(())
}

pub fn oracle_identity() -> OracleIdentity {
    OracleIdentity {
        engine: "candidate-free scalar slab semantics",
        golden_evaluator_path: REFERENCE_EVALUATOR_PATH,
        candidate_gpu_executed: false,
        candidate_recording_used: false,
    }
}
