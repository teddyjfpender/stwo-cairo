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
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

#[path = "common/reference_cache.rs"]
mod reference_cache;
use reference_cache::{cached_reference_felts, serialize_felts};

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

/// The shared reference-cache module uses the same hermetic key and validated
/// payload for smoke and parity, so tag `"shared"` populates exactly the entry
/// consumed by the qualification gate and vice versa.
///
/// Deterministic SIMD reference proofs are expensive (~20 minutes of pod CPU
/// per green round) and fixed for a given (fixture, params, input tag), so an
/// opt-in cache (STWO_PARITY_REF_CACHE=<dir>) stores the serialized reference
/// felts once and replays them byte-for-byte. Flag sweeps are byte-identical
/// by construction, so the cache stays valid across the whole measurement
/// campaign; delete the directory to force recomputation after any change
/// that legitimately moves the reference.

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
    let expected =
        cached_reference_felts(STRICT_RESIDENT_FIXTURE, "shared", reference_input, params);
    if expected != actual {
        report_section_offsets(&proof);
        report_divergence(&expected, &actual);
        panic!(
            "boundary-stepped resident proof drifted from the SIMD reference \
             (structured divergence report above; streams dumped if \
             STWO_SMOKE_DIVERGENCE_DIR is set)"
        );
    }
}

/// Map felt indices to proof sections by mirroring the manual
/// `CairoSerialize for CairoProof` impl (cairo-air/src/serde_utils.rs), which
/// emits: claim, interaction_pow, interaction_claim, then the commitment
/// scheme proof fields (config, commitments, sampled_values, decommitments,
/// sorted_queried_values, proof_of_work, fri_proof), then channel_salt.
/// Cumulative offsets printed here attribute `first_diff_index` from the
/// divergence report immediately. Both streams share the claim-derived shape,
/// so the resident proof's offsets apply to the SIMD stream as well unless
/// total lengths differ (also reported).
fn report_section_offsets(proof: &cairo_air::CairoProof<Blake2sMerkleHasher>) {
    use std::ops::Deref;
    fn len_of(serialize: impl FnOnce(&mut Vec<starknet_ff::FieldElement>)) -> usize {
        let mut felts = Vec::new();
        serialize(&mut felts);
        felts.len()
    }
    let scheme = &proof.extended_stark_proof.proof.0;
    let trace_log_sizes = proof.claim.log_sizes();
    let sorted_queried_values = cairo_air::utils::sort_and_transpose_queried_values(
        &scheme.queried_values,
        trace_log_sizes.iter().map(|c| c.as_slice()).collect(),
    );
    let sections: [(&str, usize); 11] = [
        (
            "claim",
            len_of(|out| CairoSerialize::serialize(&proof.claim, out)),
        ),
        (
            "interaction_pow",
            len_of(|out| CairoSerialize::serialize(&proof.interaction_pow, out)),
        ),
        (
            "interaction_claim",
            len_of(|out| CairoSerialize::serialize(&proof.interaction_claim, out)),
        ),
        (
            "pcs_config",
            len_of(|out| CairoSerialize::serialize(&scheme.config, out)),
        ),
        (
            "commitments",
            len_of(|out| CairoSerialize::serialize(scheme.commitments.deref(), out)),
        ),
        (
            "sampled_values",
            len_of(|out| CairoSerialize::serialize(scheme.sampled_values.deref(), out)),
        ),
        (
            "decommitments",
            len_of(|out| CairoSerialize::serialize(scheme.decommitments.deref(), out)),
        ),
        (
            "sorted_queried_values",
            len_of(|out| CairoSerialize::serialize(sorted_queried_values.deref(), out)),
        ),
        (
            "proof_of_work",
            len_of(|out| CairoSerialize::serialize(&scheme.proof_of_work, out)),
        ),
        (
            "fri_proof",
            len_of(|out| CairoSerialize::serialize(&scheme.fri_proof, out)),
        ),
        (
            "channel_salt",
            len_of(|out| CairoSerialize::serialize(&proof.channel_salt, out)),
        ),
    ];
    let mut offset = 0usize;
    for (name, len) in sections {
        eprintln!("smoke proof section {name}: offset={offset} len={len}");
        offset += len;
    }
    eprintln!("smoke proof section TOTAL: {offset}");
}

/// Structured divergence evidence instead of `assert_eq!`'s multi-megabyte
/// Debug dump of both felt vectors: first divergent index, a hex window
/// around it, per-stream lengths, and (when STWO_SMOKE_DIVERGENCE_DIR is set)
/// both streams dumped as raw 32-byte big-endian felts for offline diffing.
fn report_divergence(expected: &[starknet_ff::FieldElement], actual: &[starknet_ff::FieldElement]) {
    let hex = |felt: &starknet_ff::FieldElement| {
        felt.to_bytes_be()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let shared_len = expected.len().min(actual.len());
    let first_diff = (0..shared_len).find(|&i| expected[i] != actual[i]);
    eprintln!(
        "smoke divergence: expected_len={} actual_len={} first_diff_index={:?}",
        expected.len(),
        actual.len(),
        first_diff
    );
    let divergent = first_diff.unwrap_or(shared_len);
    let window_start = divergent.saturating_sub(4);
    let window_end = (divergent + 8).min(shared_len);
    for index in window_start..window_end {
        eprintln!(
            "  [{index}] expected={} actual={}{}",
            hex(&expected[index]),
            hex(&actual[index]),
            if expected[index] == actual[index] {
                ""
            } else {
                "   <-- DIVERGES"
            }
        );
    }
    let total_diffs = (0..shared_len)
        .filter(|&i| expected[i] != actual[i])
        .count();
    eprintln!(
        "smoke divergence: {total_diffs} of {shared_len} shared positions differ \
         (plus {} length-tail positions)",
        expected.len().abs_diff(actual.len())
    );
    if let Some(dir) = std::env::var_os("STWO_SMOKE_DIVERGENCE_DIR").map(std::path::PathBuf::from) {
        let _ = std::fs::create_dir_all(&dir);
        for (name, felts) in [("expected_simd", expected), ("actual_resident", actual)] {
            let mut bytes = Vec::with_capacity(felts.len() * 32);
            for felt in felts {
                bytes.extend_from_slice(&felt.to_bytes_be());
            }
            let path = dir.join(format!("{name}.felts.bin"));
            match std::fs::write(&path, &bytes) {
                Ok(()) => eprintln!("smoke divergence: wrote {}", path.display()),
                Err(error) => eprintln!(
                    "smoke divergence: FAILED to write {}: {error}",
                    path.display()
                ),
            }
        }
    }
}
