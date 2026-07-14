//! Causal proof validation layered on the immutable FRI provenance v1 identity preflight.
// gpu-lab-cohesion-review: Keep the OS containment, canonical decoding,
// transport/shape comparison, verifier call, and direct boundary tests in one
// fail-closed causal trust boundary; splitting them risks divergent admission.

use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use bincode::Options;
use cairo_air::air::CairoProof;
use cairo_air::verifier::verify_cairo;
use serde::{de::DeserializeOwned, Serialize};
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo_cairo_serialize::CairoSerialize;

use crate::fri_round6_provenance::{
    self, PcsShape, ProofShape, ProofShapeSeal, PROOF_SHAPE_SCHEMA,
};
use crate::model::{canonical_value_hash, sha256_hex};

pub const PROOF_PREFLIGHT_STATUS: &str = "FRI_ROUND6_PROOF_PREFLIGHT=PASS production_admissible=false resource_limits=pass proof_bincode_roundtrip_match=pass proof_verification=pass canonical_transport_match=pass proof_shape_match=pass adapter_execution_attestation=pending verifier_closure_match=pending capture_verifier_match=pending";

const GIB: u64 = 1024 * 1024 * 1024;
const DATA_LIMIT_BYTES: u64 = 2 * GIB;
const ADDRESS_SPACE_LIMIT_BYTES: u64 = 4 * GIB;
const CPU_LIMIT_SECONDS: u64 = 60;

pub(crate) type Blake2sCairoProof = CairoProof<Blake2sMerkleHasher>;

pub struct VerifiedProofPreflight {
    pub manifest_sha256: String,
    pub proof_sha256: String,
    pub canonical_transport_sha256: String,
    pub proof_shape_sha256: String,
}

pub fn preflight(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
) -> Result<VerifiedProofPreflight, String> {
    with_proof_resource_limits(|| preflight_after_limits(manifest_path, expected_manifest_sha256))
}

pub(crate) fn with_proof_resource_limits<T>(
    load_and_validate: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    with_resource_limits(install_resource_limits, load_and_validate)
}

fn with_resource_limits<T>(
    install: impl FnOnce() -> Result<(), String>,
    load_and_validate: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    install()?;
    load_and_validate()
}

fn preflight_after_limits(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
) -> Result<VerifiedProofPreflight, String> {
    let sealed = fri_round6_provenance::preflight_with_proof_inputs(
        manifest_path,
        expected_manifest_sha256,
    )?;
    let manifest_sha256 = sealed.manifest_sha256.clone();
    let proof_sha256 = sealed.proof_sha256.clone();
    let canonical_transport_sha256 = sealed.canonical_transport_sha256.clone();
    let proof_shape_sha256 = validate_sealed_proof(sealed)?;
    Ok(VerifiedProofPreflight {
        manifest_sha256,
        proof_sha256,
        canonical_transport_sha256,
        proof_shape_sha256,
    })
}

fn validate_sealed_proof(
    sealed: fri_round6_provenance::SealedProofInputs,
) -> Result<String, String> {
    panic_safe(move || {
        let fri_round6_provenance::SealedProofInputs {
            proof_bytes,
            canonical_transport_bytes,
            proof_shape,
            ..
        } = sealed;
        let proof =
            decode_exact_bincode::<Blake2sCairoProof>(&proof_bytes, "extended Cairo proof")?;
        drop(proof_bytes);
        require_canonical_transport(&proof, &canonical_transport_bytes)?;
        drop(canonical_transport_bytes);
        let derived_shape = derive_shape(&proof)?;
        let proof_shape_sha256 = require_proof_shape(&derived_shape, &proof_shape)?;
        verify_cairo::<Blake2sMerkleChannel>(proof.into())
            .map_err(|error| format!("verify extended Cairo proof: {error}"))?;
        Ok(proof_shape_sha256)
    })
}

fn install_resource_limits() -> Result<(), String> {
    cap_resource(libc::RLIMIT_DATA as libc::c_int, DATA_LIMIT_BYTES, "data")?;
    cap_resource(
        libc::RLIMIT_AS as libc::c_int,
        ADDRESS_SPACE_LIMIT_BYTES,
        "address space",
    )?;
    cap_resource(
        libc::RLIMIT_CPU as libc::c_int,
        CPU_LIMIT_SECONDS,
        "CPU time",
    )
}

fn cap_resource(resource: libc::c_int, cap: u64, label: &str) -> Result<(), String> {
    let cap = libc::rlim_t::try_from(cap)
        .map_err(|_| format!("{label} resource cap does not fit rlim_t"))?;
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `current` is initialized, its pointer is valid for this call, and
    // `getrlimit` does not retain the pointer.
    let result = unsafe { libc::getrlimit(resource as _, &mut current) };
    require_rlimit_call(result, "read", label)?;
    let limited = libc::rlimit {
        rlim_cur: current.rlim_cur.min(cap),
        rlim_max: current.rlim_max.min(cap),
    };
    // SAFETY: `limited` lives for the call, the resource constants come from
    // libc, and `setrlimit` copies rather than retaining the pointed-to value.
    let result = unsafe { libc::setrlimit(resource as _, &limited) };
    require_rlimit_call(result, "install", label)
}

fn require_rlimit_call(result: libc::c_int, operation: &str, label: &str) -> Result<(), String> {
    if result != 0 {
        return Err(format!(
            "{operation} {label} resource limit: {}",
            io::Error::last_os_error()
        ));
    }
    Ok(())
}

pub(crate) fn decode_exact_bincode<T>(bytes: &[u8], label: &str) -> Result<T, String>
where
    T: DeserializeOwned + Serialize,
{
    // bincode 1.3 replaces configured limits with `Infinite` for slice
    // deserialization. The sealed artifact caps plus process rlimits, not a
    // misleading `.with_limit`, contain decoded allocation here.
    let value = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| format!("decode exact {label} bincode: {error}"))?;
    require_exact_bincode(&value, bytes, label)?;
    Ok(value)
}

fn require_exact_bincode<T: Serialize>(
    value: &T,
    expected: &[u8],
    label: &str,
) -> Result<(), String> {
    let mut writer = ExactBytesWriter::new(expected);
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(&mut writer, value)
        .map_err(|_| format!("canonical {label} bincode mismatch"))?;
    writer.finish(&format!("canonical {label} bincode"))
}

struct ExactBytesWriter<'a> {
    expected: &'a [u8],
    offset: usize,
}

impl<'a> ExactBytesWriter<'a> {
    const fn new(expected: &'a [u8]) -> Self {
        Self {
            expected,
            offset: 0,
        }
    }

    fn finish(self, label: &str) -> Result<(), String> {
        if self.offset != self.expected.len() {
            return Err(format!(
                "{label} mismatch: derived {} bytes, sealed {} bytes",
                self.offset,
                self.expected.len()
            ));
        }
        Ok(())
    }
}

impl Write for ExactBytesWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .offset
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "byte count overflow"))?;
        if self.expected.get(self.offset..end) != Some(bytes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "serialized bytes differ",
            ));
        }
        self.offset = end;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn require_canonical_transport(
    proof: &Blake2sCairoProof,
    expected: &[u8],
) -> Result<(), String> {
    let mut felts = Vec::new();
    CairoSerialize::serialize(proof, &mut felts);
    require_exact_chunks(
        felts.iter().map(|felt| felt.to_bytes_be()),
        expected,
        "canonical Cairo transport",
    )
}

fn require_exact_chunks<const N: usize>(
    chunks: impl ExactSizeIterator<Item = [u8; N]>,
    expected: &[u8],
    label: &str,
) -> Result<(), String> {
    let byte_length = chunks
        .len()
        .checked_mul(N)
        .ok_or_else(|| format!("{label} byte length overflow"))?;
    if byte_length != expected.len() {
        return Err(format!(
            "{label} mismatch: derived {byte_length} bytes, sealed {} bytes",
            expected.len()
        ));
    }
    for (index, chunk) in chunks.enumerate() {
        let start = index * N;
        if chunk.as_slice() != &expected[start..start + N] {
            return Err(format!("{label} mismatch at chunk {index}"));
        }
    }
    Ok(())
}

pub(crate) fn derive_shape(proof: &Blake2sCairoProof) -> Result<ProofShape, String> {
    let flat_claim = proof.claim.flatten_claim();
    let enable_bytes = flat_claim
        .component_enable_bits
        .iter()
        .map(|&enabled| u8::from(enabled))
        .collect::<Vec<_>>();
    let (public_claim, output_claim, program_claim) = proof.claim.public_data.pack_into_u32s();
    let mut interaction_claim_felts = Vec::new();
    CairoSerialize::serialize(&proof.interaction_claim, &mut interaction_claim_felts);
    let mut trace_column_log_sizes = proof.claim.log_sizes().0;
    trace_column_log_sizes.insert(
        0,
        proof
            .preprocessed_trace_variant
            .to_preprocessed_trace()
            .log_sizes(),
    );
    let commitment_scheme_proof = &proof.extended_stark_proof.proof.0;
    let config = commitment_scheme_proof.config;
    let fri_proof = &commitment_scheme_proof.fri_proof;
    let fri_witness_counts = std::iter::once(&fri_proof.first_layer)
        .chain(fri_proof.inner_layers.iter())
        .map(|layer| count(layer.fri_witness.len(), "FRI witness values"))
        .collect::<Result<Vec<_>, _>>()?;
    let variant_bytes = bincode::serialize(&proof.preprocessed_trace_variant)
        .map_err(|error| format!("serialize preprocessed trace variant: {error}"))?;
    Ok(ProofShape {
        schema_version: PROOF_SHAPE_SCHEMA.into(),
        pcs: PcsShape {
            pow_bits: config.pow_bits,
            log_blowup_factor: config.fri_config.log_blowup_factor,
            log_last_layer_degree_bound: config.fri_config.log_last_layer_degree_bound,
            n_queries: count(config.fri_config.n_queries, "FRI queries")?,
            fold_step: config.fri_config.fold_step,
            lifting_log_size: config.lifting_log_size,
        },
        channel_salt: proof.channel_salt,
        preprocessed_trace_variant_sha256: sha256_hex(&variant_bytes),
        component_slots: count(flat_claim.component_enable_bits.len(), "component slots")?,
        component_enable_bits_sha256: sha256_hex(&enable_bytes),
        component_log_sizes: flat_claim.component_log_sizes,
        trace_column_log_sizes,
        public_data_word_counts: [
            count(public_claim.len(), "public claim words")?,
            count(output_claim.len(), "public output words")?,
            count(program_claim.len(), "public program words")?,
        ],
        interaction_claim_felts: count(interaction_claim_felts.len(), "interaction claim felts")?,
        commitment_trees: count(
            commitment_scheme_proof.commitments.len(),
            "commitment trees",
        )?,
        sampled_value_counts: value_counts(commitment_scheme_proof.sampled_values.as_slice())?,
        decommitment_trees: count(
            commitment_scheme_proof.decommitments.len(),
            "decommitment trees",
        )?,
        queried_value_counts: value_counts(commitment_scheme_proof.queried_values.as_slice())?,
        fri_inner_layers: count(fri_proof.inner_layers.len(), "FRI inner layers")?,
        fri_witness_counts,
        fri_last_layer_coefficients: count(
            fri_proof.last_layer_poly.len(),
            "FRI last-layer coefficients",
        )?,
        unsorted_query_locations: count(
            proof
                .extended_stark_proof
                .aux
                .unsorted_query_locations
                .len(),
            "unsorted query locations",
        )?,
    })
}

fn value_counts<T>(trees: &[Vec<Vec<T>>]) -> Result<Vec<Vec<u64>>, String> {
    trees
        .iter()
        .map(|columns| {
            columns
                .iter()
                .map(|values| count(values.len(), "proof values"))
                .collect()
        })
        .collect()
}

fn count(value: usize, label: &str) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("{label} count does not fit in u64"))
}

pub(crate) fn require_proof_shape(
    derived: &ProofShape,
    sealed: &ProofShapeSeal,
) -> Result<String, String> {
    let value =
        serde_json::to_value(derived).map_err(|error| format!("serialize proof shape: {error}"))?;
    let derived_sha256 = canonical_value_hash(&value)?;
    if derived != &sealed.shape || derived_sha256 != sealed.sha256 {
        return Err("proof-derived shape does not match the sealed proof shape".into());
    }
    Ok(derived_sha256)
}

pub(crate) fn panic_safe<T>(validate: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(validate))
        .map_err(|_| "Cairo proof validation panicked; rejecting the proof".to_string())?
}

#[cfg(test)]
#[path = "fri_round6_proof_test_support.rs"]
mod test_support;

#[cfg(test)]
#[path = "verifier_capture_recorder.rs"]
mod verifier_capture_recorder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_trailing_bincode_even_when_the_prefix_decodes() {
        let mut bytes = bincode::serialize(&7_u64).unwrap();
        bytes.push(0);
        let error = decode_exact_bincode::<u64>(&bytes, "test value").unwrap_err();
        assert!(error.contains("decode exact test value bincode"), "{error}");
    }

    #[test]
    fn rejects_same_length_canonical_transport_substitution() {
        let expected = [0_u8; 32];
        let mut substituted = expected;
        substituted[31] = 1;
        let error = require_exact_chunks(
            std::iter::once(substituted),
            &expected,
            "canonical Cairo transport",
        )
        .unwrap_err();
        assert!(
            error.contains("canonical Cairo transport mismatch"),
            "{error}"
        );
    }

    #[test]
    fn rejects_self_consistent_but_proof_independent_shape() {
        let derived = super::test_support::sample_shape();
        let mut substituted = derived.clone();
        substituted.channel_salt = 1;
        let value = serde_json::to_value(&substituted).unwrap();
        let sealed = ProofShapeSeal {
            sha256: canonical_value_hash(&value).unwrap(),
            shape: substituted,
        };
        let error = require_proof_shape(&derived, &sealed).unwrap_err();
        assert!(error.contains("proof-derived shape"), "{error}");
    }

    #[test]
    fn rejects_panic_from_validation_boundary() {
        let error = panic_safe::<()>(|| panic!("deliberate malformed-proof panic")).unwrap_err();
        assert!(error.contains("validation panicked"), "{error}");
    }

    #[test]
    fn rejects_resource_limit_syscall_failure() {
        let error = require_rlimit_call(-1, "install", "test").unwrap_err();
        assert!(error.contains("install test resource limit"), "{error}");
    }

    #[test]
    fn resource_limits_precede_and_gate_artifact_loading() {
        use std::cell::{Cell, RefCell};

        let stages = RefCell::new(Vec::new());
        with_resource_limits(
            || {
                stages.borrow_mut().push("limits");
                Ok(())
            },
            || {
                stages.borrow_mut().push("artifact-load");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*stages.borrow(), ["limits", "artifact-load"]);
        let loaded = Cell::new(false);
        let error = with_resource_limits::<()>(
            || Err("mock resource failure".into()),
            || {
                loaded.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error, "mock resource failure");
        assert!(!loaded.get(), "artifact load ran after limit failure");
    }

    #[test]
    #[ignore = "set STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY=1 on a cheap host"]
    fn real_cairo_proof_crosses_the_full_validation_boundary() {
        assert!(
            matches!(
                std::env::var("STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY").as_deref(),
                Ok("1")
            ),
            "the real-proof boundary test requires its explicit cheap-host gate"
        );
        let proof = real_cairo_proof();
        let positive = seal_proof(&proof);
        let positive_shape_sha256 = positive.proof_shape.sha256.clone();
        let mut invalid_proof = proof;
        invalid_proof.channel_salt ^= 1;
        let proof_mutation = seal_proof(&invalid_proof);
        let mut transport_mutation = positive.clone();
        *transport_mutation
            .canonical_transport_bytes
            .last_mut()
            .expect("real Cairo transport is nonempty") ^= 1;
        transport_mutation.canonical_transport_sha256 =
            sha256_hex(&transport_mutation.canonical_transport_bytes);
        let mut shape_mutation = positive.clone();
        shape_mutation.proof_shape.shape.channel_salt ^= 1;
        let value = serde_json::to_value(&shape_mutation.proof_shape.shape).unwrap();
        shape_mutation.proof_shape.sha256 = canonical_value_hash(&value).unwrap();

        if let Some(root) = std::env::var_os("STWO_GPU_LAB_PROOF_BUNDLE_DIR") {
            write_boundary_bundles(
                Path::new(&root),
                [
                    ("positive", &positive),
                    ("proof_mutation", &proof_mutation),
                    ("transport_mutation", &transport_mutation),
                    ("shape_mutation", &shape_mutation),
                ],
            );
        }

        assert_eq!(
            validate_sealed_proof(positive).unwrap(),
            positive_shape_sha256
        );

        let error = validate_sealed_proof(proof_mutation).unwrap_err();
        assert!(
            error.contains("verify extended Cairo proof") || error.contains("validation panicked"),
            "{error}"
        );

        let error = validate_sealed_proof(transport_mutation).unwrap_err();
        assert!(
            error.contains("canonical Cairo transport mismatch"),
            "{error}"
        );

        let error = validate_sealed_proof(shape_mutation).unwrap_err();
        assert!(error.contains("proof-derived shape"), "{error}");
    }

    fn real_cairo_proof() -> Blake2sCairoProof {
        use std::path::PathBuf;

        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::prover::backend::simd::SimdBackend;
        use stwo_cairo_adapter::ProverInput;
        use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
        use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};

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
        prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, parameters)
            .expect("prove real all-opcodes Cairo fixture")
    }

    fn seal_proof(proof: &Blake2sCairoProof) -> fri_round6_provenance::SealedProofInputs {
        let proof_bytes = bincode::serialize(proof).expect("serialize real Cairo proof");
        let mut felts = Vec::new();
        CairoSerialize::serialize(proof, &mut felts);
        let canonical_transport_bytes = felts
            .into_iter()
            .flat_map(|felt| felt.to_bytes_be())
            .collect::<Vec<_>>();
        let shape = derive_shape(proof).expect("derive real Cairo proof shape");
        let value = serde_json::to_value(&shape).expect("serialize real Cairo proof shape");
        let proof_shape = ProofShapeSeal {
            sha256: canonical_value_hash(&value).expect("hash real Cairo proof shape"),
            shape,
        };
        fri_round6_provenance::SealedProofInputs {
            manifest_sha256: "00".repeat(32),
            adapted_prover_input_sha256: sha256_hex(b"ephemeral adapted input\n"),
            adapted_prover_input_bytes: b"ephemeral adapted input\n".len() as u64,
            proof_sha256: sha256_hex(&proof_bytes),
            proof_bytes,
            canonical_transport_sha256: sha256_hex(&canonical_transport_bytes),
            canonical_transport_bytes,
            verifier_source_closure_sha256: sha256_hex(&source_closure("verifier")),
            proof_shape,
        }
    }

    fn write_boundary_bundles(
        root: &Path,
        cases: [(&str, &fri_round6_provenance::SealedProofInputs); 4],
    ) {
        std::fs::create_dir(root).expect("create fresh public-boundary bundle root");
        for (name, sealed) in cases {
            write_boundary_bundle(&root.join(name), sealed);
        }
    }

    fn write_boundary_bundle(root: &Path, sealed: &fri_round6_provenance::SealedProofInputs) {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        use fri_round6_provenance::{AdapterBinding, ProvenanceManifest};

        fs::create_dir(root).expect("create public-boundary case");
        let source_pie = write_artifact(root, "source.pie", "cairo-pie-zip", b"ephemeral PIE\n");
        let adapted_input = write_artifact(
            root,
            "adapted-input.bin",
            "stwo-prover-input-bincode-v1",
            b"ephemeral adapted input\n",
        );
        let invocation = write_artifact(
            root,
            "adapter-invocation.json",
            "adapter-invocation-json-v1",
            b"{\"ephemeral\":true}\n",
        );
        let executable = write_artifact(
            root,
            "adapter.sh",
            "adapter-executable",
            b"#!/bin/sh\nexit 0\n",
        );
        fs::set_permissions(
            root.join(&executable.path),
            fs::Permissions::from_mode(0o700),
        )
        .expect("mark ephemeral adapter executable");
        let adapter_sources = write_artifact(
            root,
            "adapter-sources.json",
            "adapter-source-closure-json-v1",
            &source_closure("adapter"),
        );
        let verifier_sources = write_artifact(
            root,
            "verifier-sources.json",
            "verifier-source-closure-json-v1",
            &source_closure("verifier"),
        );
        let run_bytes = serde_json::to_vec(&serde_json::json!({
            "schema_version": "stwo.gpu-lab.pie-adapter-run.v1",
            "completed": true,
            "source_pie": artifact_identity(&source_pie),
            "adapted_prover_input": artifact_identity(&adapted_input),
            "invocation": artifact_identity(&invocation),
            "adapter_executable": artifact_identity(&executable),
            "adapter_source_closure": artifact_identity(&adapter_sources),
        }))
        .expect("serialize ephemeral adapter run");
        let run_record =
            write_artifact(root, "adapter-run.json", "adapter-run-json-v1", &run_bytes);
        let proof = write_artifact(
            root,
            "extended-proof.bin",
            "extended-cairo-proof-bincode-v1",
            &sealed.proof_bytes,
        );
        let transport = write_artifact(
            root,
            "canonical-transport.bin",
            "canonical-cairo-proof-felts-be32-v1",
            &sealed.canonical_transport_bytes,
        );
        assert_eq!(&proof.sha256, &sealed.proof_sha256);
        assert_eq!(&transport.sha256, &sealed.canonical_transport_sha256);

        let manifest = ProvenanceManifest {
            schema_version: fri_round6_provenance::SCHEMA.into(),
            status: "captured-unsealed".into(),
            production_admissible: false,
            source_pie,
            adapted_prover_input: adapted_input,
            adapter: AdapterBinding {
                run_record,
                invocation,
                executable,
                source_closure: adapter_sources,
            },
            extended_cairo_proof_bincode: proof,
            canonical_cairo_transport: transport,
            verifier_source_closure: verifier_sources,
            proof_shape: sealed.proof_shape.clone(),
        };
        fs::write(
            root.join("fri_round6_provenance.v1.json"),
            serde_json::to_vec_pretty(&manifest).expect("serialize ephemeral provenance manifest"),
        )
        .expect("write ephemeral provenance manifest");
        fs::write(
            root.join("proof_shape.sha256"),
            format!("{}\n", sealed.proof_shape.sha256),
        )
        .expect("write ephemeral proof-shape digest");
    }

    fn write_artifact(
        root: &Path,
        path: &str,
        kind: &str,
        bytes: &[u8],
    ) -> fri_round6_provenance::ArtifactSeal {
        std::fs::write(root.join(path), bytes).expect("write ephemeral provenance artifact");
        fri_round6_provenance::ArtifactSeal {
            kind: kind.into(),
            path: path.into(),
            byte_length: u64::try_from(bytes.len()).expect("artifact length fits u64"),
            sha256: sha256_hex(bytes),
        }
    }

    fn artifact_identity(seal: &fri_round6_provenance::ArtifactSeal) -> serde_json::Value {
        serde_json::json!({
            "byte_length": seal.byte_length,
            "sha256": &seal.sha256,
        })
    }

    fn source_closure(component: &str) -> Vec<u8> {
        let source = b"ephemeral source\n";
        let sources = serde_json::json!([{
            "path": "src/lib.rs",
            "byte_length": source.len(),
            "sha256": sha256_hex(source),
        }]);
        let core = serde_json::json!({
            "schema_version": "stwo.gpu-lab.source-closure.v1",
            "component": component,
            "git_commit": "0".repeat(40),
            "sources": &sources,
        });
        serde_json::to_vec(&serde_json::json!({
            "schema_version": "stwo.gpu-lab.source-closure.v1",
            "component": component,
            "git_commit": "0".repeat(40),
            "sources": sources,
            "closure_sha256": canonical_value_hash(&core).expect("hash ephemeral source closure"),
        }))
        .expect("serialize ephemeral source closure")
    }
}
