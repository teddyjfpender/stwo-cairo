//! Internal observation of the canonical Blake2s verifier channel.
//!
//! The wrapper delegates every cryptographic operation to Stwo's canonical
//! [`Blake2sChannel`] and [`Blake2sMerkleChannel`]. Recording is observational:
//! exhaustion or ambiguity rejects the resulting record, never changes a
//! channel result. This untrusted module cannot emit or admit a replay fixture.
// gpu-lab-cohesion-review: Keep canonical channel delegation, bounded event
// capture, proof/capture identity binding, and exact round extraction together;
// splitting this test-only trust boundary would make causal review harder.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::relations::CommonLookupElements;
use cairo_air::verifier::{verify_cairo, INTERACTION_POW_BITS};
use serde::Serialize;
use stwo::core::air::Components;
use stwo::core::channel::{Blake2sChannel, Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::COMPOSITION_LOG_SPLIT;

use crate::fri_round6_capture::{self, CaptureShape, PcsShape, VerifiedCapture};
use crate::fri_round6_proof::{self, Blake2sCairoProof};
use crate::fri_round6_provenance;
use crate::model::{canonical_value_hash, validate_sha256};

const MAX_RECORDED_EVENTS: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ChannelState {
    pub(crate) digest_words: [u32; 8],
    pub(crate) n_draws: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(crate) enum VerifierEvent {
    MixRoot {
        root_words: [u32; 8],
        before: ChannelState,
        after: ChannelState,
    },
    DrawSecureFelt {
        value_words: [u32; 4],
        before: ChannelState,
        after: ChannelState,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct FriRound6VerifierEvents {
    pub(crate) pre_root6: ChannelState,
    pub(crate) root6_words: [u32; 8],
    pub(crate) alpha6_words: [u32; 4],
    pub(crate) cursor34: ChannelState,
    pub(crate) root7_words: [u32; 8],
    pub(crate) cursor35: ChannelState,
    pub(crate) alpha7_words: [u32; 4],
    pub(crate) cursor36: ChannelState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VerifiedCaptureMatch {
    pub(crate) manifest_sha256: String,
    pub(crate) proof_sha256: String,
    pub(crate) canonical_transport_sha256: String,
    pub(crate) proof_shape_sha256: String,
    pub(crate) verifier_source_closure_sha256: String,
    pub(crate) capture_sha256: String,
    pub(crate) adapted_prover_input_sha256: String,
    pub(crate) adapted_prover_input_bytes: u64,
    pub(crate) observer_proof_shape_id: String,
    pub(crate) verifier_events_sha256: String,
    pub(crate) verifier_events: FriRound6VerifierEvents,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RecordingBlake2sChannel {
    inner: Blake2sChannel,
    // The sink is thread-local. Make moving a live channel to another thread a
    // compile-time error instead of silently losing part of the observation.
    _thread_bound: PhantomData<Rc<()>>,
}

#[derive(Default)]
pub(crate) struct RecordingBlake2sMerkleChannel;

impl Channel for RecordingBlake2sChannel {
    const BYTES_PER_HASH: usize = <Blake2sChannel as Channel>::BYTES_PER_HASH;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        self.inner.verify_pow_nonce(n_bits, nonce)
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        self.inner.mix_u32s(data);
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        self.inner.mix_felts(felts);
    }

    fn mix_u64(&mut self, value: u64) {
        self.inner.mix_u64(value);
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let before = state(&self.inner);
        let value = self.inner.draw_secure_felt();
        record(VerifierEvent::DrawSecureFelt {
            value_words: secure_words(value),
            before,
            after: state(&self.inner),
        });
        value
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        self.inner.draw_secure_felts(n_felts)
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        self.inner.draw_u32s()
    }
}

impl MerkleChannel for RecordingBlake2sMerkleChannel {
    type C = RecordingBlake2sChannel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2sHash) {
        let before = state(&channel.inner);
        <Blake2sMerkleChannel as MerkleChannel>::mix_root(&mut channel.inner, root);
        record(VerifierEvent::MixRoot {
            root_words: hash_words(root),
            before,
            after: state(&channel.inner),
        });
    }
}

/// Runs one canonical-verifier call site and returns its bounded observation.
/// Nested capture on the same thread is rejected before `run` executes.
pub(crate) fn record_verifier_events<T>(
    run: impl FnOnce() -> T,
) -> Result<(T, Vec<VerifierEvent>), String> {
    EVENT_SINK.with(|sink| -> Result<(), String> {
        let mut sink = sink.borrow_mut();
        if sink.is_some() {
            return Err("nested verifier event capture is not supported".into());
        }
        *sink = Some(EventSink::default());
        Ok(())
    })?;

    let guard = CaptureGuard { active: true };
    let result = run();
    Ok((result, guard.finish()?))
}

/// Loads every causal input from its hash-pinned artifact, runs the unchanged
/// canonical verifier through the recording channel, requires verifier
/// success, and only then extracts the round-6 match. This internal result is
/// not an artifact or admission: a valid proof does not prove that the adapter
/// generated it.
pub(crate) fn verify_and_match_fri_round6(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
    capture_path: &Path,
    expected_capture_sha256: &str,
) -> Result<VerifiedCaptureMatch, String> {
    fri_round6_proof::with_proof_resource_limits(|| {
        fri_round6_proof::panic_safe(|| {
            verify_and_bind_after_limits(
                manifest_path,
                expected_manifest_sha256,
                capture_path,
                expected_capture_sha256,
            )
        })
    })
}

fn verify_and_bind_after_limits(
    manifest_path: &Path,
    expected_manifest_sha256: &str,
    capture_path: &Path,
    expected_capture_sha256: &str,
) -> Result<VerifiedCaptureMatch, String> {
    let sealed = fri_round6_provenance::preflight_with_proof_inputs(
        manifest_path,
        expected_manifest_sha256,
    )?;
    let capture = fri_round6_capture::load(capture_path, expected_capture_sha256)?;
    let prepared = prepare_sealed_proof(sealed)?;
    require_identity_and_shape(&prepared, &capture)?;

    let PreparedSealedProof {
        proof,
        manifest_sha256,
        adapted_prover_input_sha256,
        adapted_prover_input_bytes,
        proof_sha256,
        canonical_transport_sha256,
        verifier_source_closure_sha256,
        proof_shape_sha256,
        ..
    } = prepared;
    let (verification, events) =
        record_verifier_events(|| verify_cairo::<RecordingBlake2sMerkleChannel>(proof.into()))?;
    verification.map_err(|error| format!("verify extended Cairo proof with recorder: {error}"))?;

    bind_loaded_capture(
        &events,
        &capture,
        BoundIdentities {
            manifest_sha256,
            adapted_prover_input_sha256,
            adapted_prover_input_bytes,
            proof_sha256,
            canonical_transport_sha256,
            verifier_source_closure_sha256,
            proof_shape_sha256,
        },
    )
}

fn bind_loaded_capture(
    events: &[VerifierEvent],
    capture: &VerifiedCapture,
    identities: BoundIdentities,
) -> Result<VerifiedCaptureMatch, String> {
    let expected = expected_events(capture)?;
    let matches = events
        .windows(4)
        .filter_map(extract_round6)
        .filter(|candidate| candidate == &expected)
        .collect::<Vec<_>>();
    let verifier_events = match matches.as_slice() {
        [only] => only.clone(),
        [] => {
            return Err(
                "verified canonical event stream contains no exact round-6 capture match".into(),
            )
        }
        _ => {
            return Err(
                "verified canonical event stream contains multiple round-6 capture matches".into(),
            )
        }
    };
    let value = serde_json::to_value(&verifier_events)
        .map_err(|error| format!("serialize verifier capture events: {error}"))?;

    Ok(VerifiedCaptureMatch {
        manifest_sha256: identities.manifest_sha256,
        proof_sha256: identities.proof_sha256,
        canonical_transport_sha256: identities.canonical_transport_sha256,
        proof_shape_sha256: identities.proof_shape_sha256,
        verifier_source_closure_sha256: identities.verifier_source_closure_sha256,
        capture_sha256: capture.capture_sha256.clone(),
        adapted_prover_input_sha256: identities.adapted_prover_input_sha256,
        adapted_prover_input_bytes: identities.adapted_prover_input_bytes,
        observer_proof_shape_id: capture.source.observer_proof_shape_id.clone(),
        verifier_events_sha256: canonical_value_hash(&value)?,
        verifier_events,
    })
}

struct BoundIdentities {
    manifest_sha256: String,
    adapted_prover_input_sha256: String,
    adapted_prover_input_bytes: u64,
    proof_sha256: String,
    canonical_transport_sha256: String,
    verifier_source_closure_sha256: String,
    proof_shape_sha256: String,
}

struct PreparedSealedProof {
    proof: Blake2sCairoProof,
    manifest_sha256: String,
    adapted_prover_input_sha256: String,
    adapted_prover_input_bytes: u64,
    proof_sha256: String,
    canonical_transport_sha256: String,
    verifier_source_closure_sha256: String,
    proof_shape_sha256: String,
}

fn prepare_sealed_proof(
    sealed: fri_round6_provenance::SealedProofInputs,
) -> Result<PreparedSealedProof, String> {
    let fri_round6_provenance::SealedProofInputs {
        manifest_sha256,
        adapted_prover_input_sha256,
        adapted_prover_input_bytes,
        proof_bytes,
        proof_sha256,
        canonical_transport_bytes,
        canonical_transport_sha256,
        verifier_source_closure_sha256,
        proof_shape,
    } = sealed;
    let proof = fri_round6_proof::decode_exact_bincode::<Blake2sCairoProof>(
        &proof_bytes,
        "extended Cairo proof",
    )?;
    drop(proof_bytes);
    fri_round6_proof::require_canonical_transport(&proof, &canonical_transport_bytes)?;
    drop(canonical_transport_bytes);
    let derived_shape = fri_round6_proof::derive_shape(&proof)?;
    let proof_shape_sha256 = fri_round6_proof::require_proof_shape(&derived_shape, &proof_shape)?;
    Ok(PreparedSealedProof {
        proof,
        manifest_sha256,
        adapted_prover_input_sha256,
        adapted_prover_input_bytes,
        proof_sha256,
        canonical_transport_sha256,
        verifier_source_closure_sha256,
        proof_shape_sha256,
    })
}

fn require_identity_and_shape(
    prepared: &PreparedSealedProof,
    capture: &VerifiedCapture,
) -> Result<(), String> {
    for (value, label) in [
        (&prepared.manifest_sha256, "provenance manifest sha256"),
        (&prepared.proof_sha256, "extended proof sha256"),
        (
            &prepared.canonical_transport_sha256,
            "canonical transport sha256",
        ),
        (
            &prepared.verifier_source_closure_sha256,
            "verifier source closure sha256",
        ),
        (&prepared.proof_shape_sha256, "proof shape sha256"),
        (&capture.capture_sha256, "FRI capture seed sha256"),
        (
            &capture.source.prover_input_sha256,
            "capture ProverInput sha256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    if prepared.adapted_prover_input_sha256 != capture.source.prover_input_sha256
        || prepared.adapted_prover_input_bytes != capture.source.prover_input_bytes
    {
        return Err("capture source does not match the manifest's adapted ProverInput".into());
    }
    let proof_capture_shape = derive_capture_shape(&prepared.proof)?;
    if proof_capture_shape != capture.shape {
        return Err("capture transcript shape does not match the decoded proof".into());
    }
    Ok(())
}

fn derive_capture_shape(proof: &Blake2sCairoProof) -> Result<CaptureShape, String> {
    let flat_claim = proof.claim.flatten_claim();
    let (public_claim, _, _) = proof.claim.public_data.pack_into_u32s();
    let interaction_claim_felts = proof.interaction_claim.flatten_interaction_claim().len();
    let commitment_proof = &proof.extended_stark_proof.proof.0;
    let config = commitment_proof.config;

    let preprocessed = proof.preprocessed_trace_variant.to_preprocessed_trace();
    let preprocessed_ids = preprocessed.ids();
    let lookup_elements = CommonLookupElements::dummy();
    let cairo_components = CairoComponents::new(
        &proof.claim,
        &lookup_elements,
        &proof.interaction_claim,
        &preprocessed_ids,
    );
    let components = Components {
        components: cairo_components.components(),
        n_preprocessed_columns: preprocessed_ids.len(),
    };
    let split_composition_log_degree = components
        .composition_log_degree_bound()
        .checked_sub(COMPOSITION_LOG_SPLIT)
        .ok_or("composition log degree is smaller than the canonical split")?;
    let split_composition_log_size = split_composition_log_degree
        .checked_add(config.fri_config.log_blowup_factor)
        .ok_or("split composition log size overflow")?;
    let lifting_log_size = config
        .lifting_log_size
        .unwrap_or(split_composition_log_size);
    if lifting_log_size < split_composition_log_size {
        return Err("proof lifting log size is smaller than its split composition tree".into());
    }

    let oods_sampled_values_felts = commitment_proof
        .sampled_values
        .iter()
        .flat_map(|tree| tree.iter())
        .try_fold(0usize, |total, column| total.checked_add(column.len()))
        .ok_or("OODS sampled-value count overflow")?;
    let fri_tree_count = commitment_proof
        .fri_proof
        .inner_layers
        .len()
        .checked_add(1)
        .ok_or("FRI tree count overflow")?;

    Ok(CaptureShape {
        circle_log_size: lifting_log_size,
        claim_enable_felts: packed_secure_felts(flat_claim.component_enable_bits.len())?,
        claim_log_size_felts: packed_secure_felts(flat_claim.component_log_sizes.len())?,
        claim_public_data_felts: packed_secure_felts(public_claim.len())?,
        interaction_claim_felts: u32_count(interaction_claim_felts, "interaction claim")?,
        oods_sampled_values_felts: u32_count(oods_sampled_values_felts, "OODS sampled values")?,
        interaction_pow_bits: INTERACTION_POW_BITS,
        pcs: PcsShape {
            pow_bits: config.pow_bits,
            log_blowup_factor: config.fri_config.log_blowup_factor,
            n_queries: u32_count(config.fri_config.n_queries, "FRI queries")?,
            log_last_layer_degree_bound: config.fri_config.log_last_layer_degree_bound,
            fold_step: config.fri_config.fold_step,
            lifting_log_size,
        },
        fri_tree_count: u32_count(fri_tree_count, "FRI trees")?,
    })
}

fn packed_secure_felts(words: usize) -> Result<u32, String> {
    u32_count(words.div_ceil(4), "packed claim felts")
}

fn u32_count(value: usize, label: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label} count does not fit in u32"))
}

#[derive(Default)]
struct EventSink {
    events: Vec<VerifierEvent>,
    overflowed: bool,
}

thread_local! {
    static EVENT_SINK: RefCell<Option<EventSink>> = const { RefCell::new(None) };
}

struct CaptureGuard {
    active: bool,
}

impl CaptureGuard {
    fn finish(mut self) -> Result<Vec<VerifierEvent>, String> {
        let sink = EVENT_SINK
            .with(|slot| slot.borrow_mut().take())
            .ok_or("verifier event capture disappeared before completion")?;
        self.active = false;
        if sink.overflowed {
            return Err(format!(
                "verifier event capture exceeded {MAX_RECORDED_EVENTS} events"
            ));
        }
        Ok(sink.events)
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            EVENT_SINK.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }
}

fn record(event: VerifierEvent) {
    EVENT_SINK.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(sink) = slot.as_mut() else {
            return;
        };
        if sink.events.len() == MAX_RECORDED_EVENTS {
            sink.overflowed = true;
        } else if !sink.overflowed {
            sink.events.push(event);
        }
    });
}

fn state(channel: &Blake2sChannel) -> ChannelState {
    ChannelState {
        digest_words: hash_words(channel.digest()),
        n_draws: channel.n_draws(),
    }
}

fn hash_words(hash: Blake2sHash) -> [u32; 8] {
    std::array::from_fn(|index| {
        let offset = index * 4;
        u32::from_le_bytes(hash.0[offset..offset + 4].try_into().unwrap())
    })
}

fn secure_words(value: SecureField) -> [u32; 4] {
    value.to_m31_array().map(|felt| felt.0)
}

fn expected_events(capture: &VerifiedCapture) -> Result<FriRound6VerifierEvents, String> {
    Ok(FriRound6VerifierEvents {
        pre_root6: capture_state(&capture.cursor32, "cursor32")?,
        root6_words: capture_words(&capture.observed.root6_words, "root6")?,
        alpha6_words: capture_words(&capture.observed.alpha6_words, "alpha6")?,
        cursor34: capture_state(&capture.observed.cursor34_state_words, "cursor34")?,
        root7_words: capture_words(&capture.observed.root7_words, "root7")?,
        cursor35: capture_state(&capture.observed.cursor35_state_words, "cursor35")?,
        alpha7_words: capture_words(&capture.observed.alpha7_words, "alpha7")?,
        cursor36: capture_state(&capture.observed.cursor36_state_words, "cursor36")?,
    })
}

fn capture_state(words: &[u32], label: &str) -> Result<ChannelState, String> {
    let digest_words = capture_words(
        words
            .get(..8)
            .ok_or_else(|| format!("captured {label} state is shorter than eight digest words"))?,
        label,
    )?;
    let n_draws = *words
        .get(8)
        .ok_or_else(|| format!("captured {label} state has no draw counter"))?;
    Ok(ChannelState {
        digest_words,
        n_draws,
    })
}

fn capture_words<const N: usize>(words: &[u32], label: &str) -> Result<[u32; N], String> {
    words
        .try_into()
        .map_err(|_| format!("captured {label} has {} words; expected {N}", words.len()))
}

fn extract_round6(events: &[VerifierEvent]) -> Option<FriRound6VerifierEvents> {
    let [VerifierEvent::MixRoot {
        root_words: root6_words,
        before: pre_root6,
        after: after_root6,
    }, VerifierEvent::DrawSecureFelt {
        value_words: alpha6_words,
        before: before_alpha6,
        after: cursor34,
    }, VerifierEvent::MixRoot {
        root_words: root7_words,
        before: before_root7,
        after: cursor35,
    }, VerifierEvent::DrawSecureFelt {
        value_words: alpha7_words,
        before: before_alpha7,
        after: cursor36,
    }] = events
    else {
        return None;
    };
    if after_root6 != before_alpha6
        || cursor34 != before_root7
        || cursor35 != before_alpha7
        || after_root6.n_draws != 0
        || cursor35.n_draws != 0
        || before_alpha6.digest_words != cursor34.digest_words
        || cursor34.n_draws <= before_alpha6.n_draws
        || before_alpha7.digest_words != cursor36.digest_words
        || cursor36.n_draws <= before_alpha7.n_draws
    {
        return None;
    }
    Some(FriRound6VerifierEvents {
        pre_root6: pre_root6.clone(),
        root6_words: *root6_words,
        alpha6_words: *alpha6_words,
        cursor34: cursor34.clone(),
        root7_words: *root7_words,
        cursor35: cursor35.clone(),
        alpha7_words: *alpha7_words,
        cursor36: cursor36.clone(),
    })
}

#[cfg(test)]
#[path = "verifier_capture_recorder_tests.rs"]
mod tests;
