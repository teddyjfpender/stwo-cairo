//! Diagnostic boundary-stepped smoke runner for the strict resident CUDA path.
//!
//! This is a DEBUGGING INSTRUMENT, not a qualification gate. The manifest
//! validator (`gpu_benchmarks/test_validate_architecture_record.py`)
//! auto-discovers `*_native.rs` test targets and requires exact counted-gate
//! registration for each; this file deliberately does NOT match that glob so it
//! never enters the soundness manifest. Whole-proof qualification remains owned
//! by `tests/resident_parity_native.rs`.
//!
//! What it does: one strict resident proof of the sn2 profile fixture in which
//! the transcript-bounded replay is stepped boundary-by-boundary. After each
//! segment replay the CUDA context is drained (diagnostic-mode sync) and a
//! checkpoint line is printed, so a device fault names the exact protocol
//! boundary in its panic instead of surfacing only at the final bundle read.
//! The completed proof is still byte-compared against the cached SIMD
//! reference, so a "green" smoke run is also a real correctness observation.

#![cfg(stwo_cuda_link)]

use std::time::{Duration, Instant};

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::prover::backend::simd::SimdBackend;
use stwo_backend_cuda::{assemble_blake2s_stark_proof, Blake2sProofAssemblyInput};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::arena_plan::CommitmentTreeId;
use stwo_cairo_gpu_prover::graphs::GraphSegment;
use stwo_cairo_gpu_prover::protocol_discovery::interaction_claim_from_flattened;
use stwo_cairo_gpu_prover::resident_runtime::{ResidentGraphRuntime, ResidentRuntimeError};
use stwo_cairo_gpu_prover::{GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

// Same fixture as the strict resident qualification gate: broad, capture-safe
// opcode + Poseidon statement.
const STRICT_RESIDENT_FIXTURE: &str = "test_prove_verify_sn2_profile";

/// Production replay generation: capture consumes generation 1, the first warm
/// replay is generation 2 (`prove_resident_blake2s_with_mode` passes the same
/// literal to `replay_all_prepared_subgraphs`).
const REPLAY_GENERATION: u64 = 2;

/// `GraphSegment::FriLayer` is keyed by `u8` and layer 0 is the first FRI tree,
/// so round probing can never exceed 255 rounds.
const MAX_FRI_ROUNDS: usize = 255;

fn resident_input() -> ProverInput {
    run_and_adapt(
        &get_compiled_cairo_program_path(STRICT_RESIDENT_FIXTURE),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap()
}

fn resident_params() -> ProverParameters {
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig::default(),
        preprocessed_trace: PreProcessedTraceVariant::Canonical,
        channel_salt: 0,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

/// Verbatim copy of the reference cache helper from
/// `tests/resident_parity_native.rs` — same `STWO_PARITY_REF_CACHE` directory,
/// same `"{fixture}-{tag}-v1-{fnv-of-params-debug}.ref"` key. Called below with
/// tag `"shared"` and identical params, so a smoke run POPULATES exactly the
/// cache entry the qualification gate (`strict_resident_poseidon_graph_a_*`,
/// `strict_resident_mirrored_transcript_*`) later reads, and vice versa.
///
/// Deterministic SIMD reference proofs are expensive (~20 minutes of pod CPU
/// per green round) and fixed for a given (fixture, params, input tag), so an
/// opt-in cache (STWO_PARITY_REF_CACHE=<dir>) stores the serialized reference
/// felts once and replays them byte-for-byte. Flag sweeps are byte-identical
/// by construction, so the cache stays valid across the whole measurement
/// campaign; delete the directory to force recomputation after any change
/// that legitimately moves the reference.
fn cached_reference_felts(
    tag: &str,
    input: ProverInput,
    params: ProverParameters,
) -> Vec<starknet_ff::FieldElement> {
    const REFERENCE_SCHEMA: u32 = 1;
    let cache_dir = std::env::var_os("STWO_PARITY_REF_CACHE").map(std::path::PathBuf::from);
    let key_path = cache_dir.as_ref().map(|dir| {
        dir.join(format!(
            "{STRICT_RESIDENT_FIXTURE}-{tag}-v{REFERENCE_SCHEMA}-{:x}.ref",
            {
                // Stable fingerprint of the parameters that shape the proof.
                let text = format!("{params:?}");
                let mut hash = 0xcbf29ce484222325u64;
                for byte in text.bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                hash
            }
        ))
    });
    if let Some(path) = &key_path {
        if let Ok(bytes) = std::fs::read(path) {
            if bytes.len() % 32 == 0 {
                return bytes
                    .chunks_exact(32)
                    .map(|chunk| {
                        starknet_ff::FieldElement::from_bytes_be(chunk.try_into().unwrap())
                            .expect("cached reference felt")
                    })
                    .collect();
            }
        }
    }
    let felts =
        serialize_felts(&prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, params).unwrap());
    if let Some(path) = &key_path {
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        let mut bytes = Vec::with_capacity(felts.len() * 32);
        for felt in &felts {
            bytes.extend_from_slice(&felt.to_bytes_be());
        }
        let staging = path.with_extension("ref.tmp");
        if std::fs::write(&staging, &bytes).is_ok() {
            let _ = std::fs::rename(&staging, path);
        }
    }
    felts
}

fn serialize_felts<H>(proof: &cairo_air::CairoProof<H>) -> Vec<starknet_ff::FieldElement>
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    H::Hash: CairoSerialize,
{
    let mut felts = Vec::new();
    CairoSerialize::serialize(proof, &mut felts);
    felts
}

/// Diagnostic-mode sync: drain the workspace's single proof stream through an
/// existing public seam. `read_commitment_root(Preprocessed)` enqueues one
/// 32-byte D2H of the fixed preprocessed root (valid at every boundary — it is
/// staged during cold workspace setup, before any runtime exists) and then
/// calls `context().sync()`, which waits for ALL previously enqueued work.
/// This deliberately performs host activity the production hot path forbids,
/// which is one of the reasons this runner does not assert the replay hot-path
/// budget.
fn diagnostic_sync(runtime: &ResidentGraphRuntime<'_>) -> Result<(), ResidentRuntimeError> {
    runtime
        .read_commitment_root(CommitmentTreeId::Preprocessed)
        .map(|_| ())
}

/// Replay one transcript-bounded segment, drain the device, and print a
/// checkpoint. A failure panics with the boundary's name, so a device fault is
/// attributed to the segment that produced it rather than to the final bundle
/// read.
fn stepped_boundary<'a>(
    runtime: &mut ResidentGraphRuntime<'a>,
    rows: &mut Vec<(GraphSegment, Duration)>,
    previous: &mut Instant,
    segment: GraphSegment,
    replay: impl FnOnce(&mut ResidentGraphRuntime<'a>) -> Result<(), ResidentRuntimeError>,
) {
    replay(runtime)
        .and_then(|()| diagnostic_sync(runtime))
        .unwrap_or_else(|e| panic!("boundary {segment:?} failed: {e:?}"));
    let elapsed = previous.elapsed();
    *previous = Instant::now();
    eprintln!(
        "smoke boundary {segment:?}: {:.3} ms since previous boundary",
        elapsed.as_secs_f64() * 1e3
    );
    rows.push((segment, elapsed));
}

/// One strict resident proof with the production replay decomposed at every
/// true Fiat-Shamir boundary, each followed by a diagnostic context sync and a
/// checkpoint line. Proof completion (single bundle read, host assembly) and
/// the SIMD byte-comparison are exactly the production/qualification flow.
///
/// NOT asserted here, by design: the resident hot-path budget
/// (`require_hot_path_budget`). The per-boundary syncs and 32-byte root reads
/// intentionally violate it; this runner measures and attributes, it does not
/// qualify.
#[test]
fn smoke_single_resident_proof_boundary_stepped() {
    let params = resident_params();
    let reference_input = resident_input();

    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();

    let total_start = Instant::now();
    let ((claim, bundle, shape, rows), session_telemetry) = prover
        .with_strict_resident_session(resident_input(), params, |runtime, artifacts| {
            runtime.require_prepared_witness_coverage()?;

            let capture_start = Instant::now();
            runtime.capture_all_prepared_subgraphs()?;
            // Derive the FRI round count through the public requirements seam:
            // `fri_round_output_tree` fails with InvalidRoundIndex exactly when
            // the round does not exist.
            let mut fri_rounds = 0usize;
            while fri_rounds < MAX_FRI_ROUNDS && runtime.fri_round_output_tree(fri_rounds).is_ok() {
                fri_rounds += 1;
            }
            eprintln!(
                "smoke capture: {} subgraphs, {} FRI rounds, proof bundle {:?} bytes, {:.3} ms",
                runtime.captured_graph_count(),
                fri_rounds,
                runtime.workspace_proof_bundle_bytes(),
                capture_start.elapsed().as_secs_f64() * 1e3
            );

            // Telemetry is reset for observability only; see the test doc
            // comment — this runner never asserts the hot-path budget.
            runtime.begin_hot_path_telemetry();
            runtime.begin_transcript_generation(REPLAY_GENERATION)?;

            // Boundary-stepped equivalent of `replay_all_prepared_subgraphs`:
            // identical calls in identical production order, with a diagnostic
            // sync + checkpoint inserted at every transcript boundary.
            let mut rows = Vec::new();
            let mut previous = Instant::now();
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::IngestWitnessBaseCommit,
                |r| r.replay_base_commit_only(),
            );
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::InteractionCommit,
                |r| r.replay_interaction_relation_and_commit(),
            );
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::CompositionQuotientCommit,
                |r| r.replay_composition_commit_only(),
            );
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::OodsEvaluation,
                |r| r.replay_oods_transcript_boundary(),
            );
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::FriLayer(0),
                |r| r.replay_fri_first_tree(),
            );
            for round in 0..fri_rounds {
                // `fri_round_segment` reserves layer 0 for the first tree, so
                // round i replays into segment FriLayer(i + 1).
                let layer = u8::try_from(round + 1).expect("FRI round count fits in u8");
                stepped_boundary(
                    runtime,
                    &mut rows,
                    &mut previous,
                    GraphSegment::FriLayer(layer),
                    |r| r.replay_fri_round(round),
                );
            }
            stepped_boundary(
                runtime,
                &mut rows,
                &mut previous,
                GraphSegment::OodsQueriesDecommitAssemble,
                |r| r.replay_final_transcript_boundary(),
            );

            // Production host boundary: one contiguous D2H copy + sync.
            let bundle_start = Instant::now();
            let bundle = runtime.read_proof_bundle_once()?;
            eprintln!(
                "smoke proof bundle read: {:.3} ms",
                bundle_start.elapsed().as_secs_f64() * 1e3
            );
            eprintln!(
                "smoke replay telemetry (diagnostic mode, budget NOT asserted): {:?}",
                runtime.hot_path_telemetry()
            );

            Ok((
                artifacts.claim.clone(),
                bundle,
                runtime.proof_assembly_shape().clone(),
                rows,
            ))
        })
        .expect("strict resident session failed");

    // Production enforces Graph-A admission telemetry before assembling.
    session_telemetry
        .require_strict_graph_a()
        .expect("strict Graph-A telemetry");

    // Complete the proof exactly as `prove_resident_blake2s_with_mode` does
    // after replay: decode the flat interaction claim, assemble the Blake2s
    // STARK proof from the single bundle, wrap into the verifier-facing type.
    let interaction_claim = interaction_claim_from_flattened(&claim, &bundle.interaction_claim)
        .expect("interaction claim decode");
    let stark_proof = assemble_blake2s_stark_proof(Blake2sProofAssemblyInput {
        config: params.pcs_config,
        shape,
        commitments: bundle.commitments,
        sampled_values: bundle.sampled_values,
        raw_queries: bundle.decommitment.raw_queries().to_vec(),
        proof_of_work: bundle.query_pow,
        final_line_poly_words: bundle.final_line_poly_words,
        fri_commitments: bundle.fri_commitments,
        decommitment: bundle.decommitment,
    })
    .expect("proof assembly");
    let proof: cairo_air::CairoProof<Blake2sMerkleHasher> = cairo_air::CairoProof {
        claim,
        interaction_pow: bundle.interaction_pow,
        interaction_claim,
        extended_stark_proof: stark_proof,
        channel_salt: params.channel_salt,
        preprocessed_trace_variant: params.preprocessed_trace,
    };
    let actual = serialize_felts(&proof);
    let total = total_start.elapsed();

    eprintln!("smoke boundary table ({} boundaries):", rows.len());
    for (segment, elapsed) in &rows {
        eprintln!(
            "  {:<32} {:>10.3} ms",
            format!("{segment:?}"),
            elapsed.as_secs_f64() * 1e3
        );
    }
    eprintln!(
        "smoke total wall time (session + stepped replay + assembly): {:.3} s",
        total.as_secs_f64()
    );

    // Byte-oracle last: the resident proof above already surfaced any device
    // fault at its boundary, so the ~20-minute SIMD reference (or its cached
    // felts — see `cached_reference_felts`) only runs on a completed proof.
    let expected = cached_reference_felts("shared", reference_input, params);
    assert_eq!(
        expected, actual,
        "boundary-stepped resident proof drifted from the SIMD reference"
    );
}
