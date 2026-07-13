use std::path::Path;

use stwo::core::circle::Coset;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::line::LineDomain;
use stwo::core::utils::bit_reverse_index;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::prover::backend::cpu::fold_line_cpu;
use stwo::prover::backend::CpuBackend;
use stwo::prover::line::LineEvaluation;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::vcs_lifted::ops::{MerkleOpsLifted, PackLeavesOps};
use stwo_backend_cuda::BLAKE2S_TRANSCRIPT_PROTOCOL_TAG;

use crate::artifact_io::{write_immutable_bytes, write_immutable_json};
use crate::fri_round6_capture::VerifiedCapture;
use crate::fri_round6_index::{
    Chains, Chunk, Index, Oracle, Payload, PredecessorSeal, SemanticIds, Shape, Transcript,
};
use crate::fri_round6_transcript::{
    cairo_schedule, cursor32_state, hash_words, independently_checked_chains, replay, schedule,
    state_words, ALPHA6, ALPHA7, CAIRO_MAX_REJECTION_ROUNDS, MAX_REJECTION_ROUNDS, ROOT6, ROOT7,
    STATE32,
};
use crate::fri_round6_validation::{source_index, verify_observed, words_hash};
use crate::model::sha256_hex;

const SYNTHETIC_SCHEMA: &str = "stwo.gpu-lab.fri-round6-synthetic-layout-index.v1";
const CAPTURED_SCHEMA: &str = "stwo.gpu-lab.fri-round6-captured-index.v1";
const SOURCE_CIRCLE_LOG: u32 = 24;
const ENTRY_LOG: u32 = 6;
const EXIT_LOG: u32 = 3;
const PACKED_LEAF_LOG: u32 = 2;
const FULL_TWIDDLE_WORDS: u32 = 1 << (SOURCE_CIRCLE_LOG - 1);
const PAYLOAD_BYTES: usize = 1_936;

#[derive(Clone, Copy)]
enum Case {
    Primary,
    Hostile,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Hostile => "hostile",
        }
    }

    fn class(self) -> &'static str {
        match self {
            Self::Primary => "representative",
            Self::Hostile => "hostile-mutation",
        }
    }
}

struct Artifact {
    payload: Vec<u8>,
    index: Index,
    entry: Vec<u32>,
    root: Vec<u32>,
    challenge: Vec<u32>,
    chains: [u64; 5],
}

#[derive(Clone, Copy)]
enum Origin<'a> {
    Synthetic,
    Captured(&'a VerifiedCapture),
}

struct BuildContext<'a> {
    origin: Origin<'a>,
    primary: SecureColumnByCoords<CpuBackend>,
    cursor32: Vec<u32>,
    chains: [u64; 5],
}

pub fn export_synthetic(output_dir: &Path) -> Result<(), String> {
    let context = synthetic_context()?;
    export_context(output_dir, &context)
}

fn synthetic_context() -> Result<BuildContext<'static>, String> {
    let full_schedule = schedule(4)?;
    let chains = independently_checked_chains(&full_schedule)?;
    Ok(BuildContext {
        origin: Origin::Synthetic,
        primary: synthetic_entry_values(),
        cursor32: cursor32_state(chains[0], SOURCE_CIRCLE_LOG, ENTRY_LOG),
        chains,
    })
}

pub fn export_capture(output_dir: &Path, capture: &VerifiedCapture) -> Result<(), String> {
    let context = BuildContext {
        origin: Origin::Captured(capture),
        primary: decode_column_words(&capture.entry_words)?,
        cursor32: capture.cursor32.clone(),
        chains: capture.chains,
    };
    export_context(output_dir, &context)
}

fn export_context(output_dir: &Path, context: &BuildContext<'_>) -> Result<(), String> {
    let primary = build(context, Case::Primary)?;
    let hostile = build(context, Case::Hostile)?;
    validate_pair(&primary, &hostile)?;
    write_case(output_dir, context, Case::Primary, &primary)?;
    write_case(output_dir, context, Case::Hostile, &hostile)
}

fn write_case(
    output_dir: &Path,
    context: &BuildContext<'_>,
    case: Case,
    artifact: &Artifact,
) -> Result<(), String> {
    let family = match context.origin {
        Origin::Synthetic => "synthetic-layout",
        Origin::Captured(_) => "sn2",
    };
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

fn build(context: &BuildContext<'_>, case: Case) -> Result<Artifact, String> {
    let entry_values = case_values(&context.primary, case);
    let (root6_leaves, root6) = commit(&entry_values)?;
    let (prefix_schedule, full_schedule) = match context.origin {
        Origin::Synthetic => (schedule(2)?, schedule(4)?),
        Origin::Captured(_) => (cairo_schedule(2)?, cairo_schedule(4)?),
    };
    let predecessor = replay(&prefix_schedule, &context.cursor32, &[root6])?;
    let alpha6_words = predecessor.output_words.clone();
    let alpha6 = secure(&alpha6_words)?;
    let twiddles = inverse_twiddles();
    let final_values = fold_three(entry_values.clone(), alpha6);
    let (leaves, root7) = commit(&final_values)?;
    let full = replay(&full_schedule, &context.cursor32, &[root6, root7])?;

    if predecessor.boundaries != full.boundaries[..2]
        || predecessor.output_words != full.output_words[..4]
    {
        return Err("FRI predecessor replay changed under full tail schedule".into());
    }
    let challenge = full.output_words[4..8].to_vec();
    let entry_state = state_words(&full.boundaries[1], 34, context.chains[2]);
    let boundary_mix = state_words(&full.boundaries[2], 35, context.chains[3]);
    let boundary_draw = state_words(&full.boundaries[3], 36, context.chains[4]);
    let entry_words = column_words(&entry_values);
    let final_words = column_words(&final_values);
    let root_words = hash_words(root7);
    let leaf_words = leaves.into_iter().flat_map(hash_words).collect::<Vec<_>>();
    if let Origin::Captured(capture) = context.origin {
        if matches!(case, Case::Primary) {
            verify_observed(
                capture,
                &hash_words(root6),
                &alpha6_words,
                &entry_state,
                &root_words,
                &challenge,
                &boundary_mix,
                &boundary_draw,
            )?;
        }
    }
    let chunks = vec![
        ("entry_pong", entry_words.clone()),
        ("inverse_twiddles", twiddles),
        ("alpha6", alpha6_words.clone()),
        ("entry_state", entry_state),
        ("expected_final_ping", final_words.clone()),
        ("expected_root", root_words.clone()),
        ("expected_exit_state", boundary_draw.clone()),
        ("expected_challenge7", challenge.clone()),
        ("expected_retained", final_words),
        ("expected_leaves", leaf_words),
        ("expected_mix_input", root_words.clone()),
        ("expected_draw_output", challenge.clone()),
        ("expected_boundary_mix", boundary_mix),
        ("expected_boundary_draw", boundary_draw),
    ];
    let (payload, chunk_index) = encode_chunks(&chunks)?;
    let (schema, family, scope, max_rejections, semantic_ids) = match context.origin {
        Origin::Synthetic => (
            SYNTHETIC_SCHEMA,
            "synthetic-layout",
            "synthetic-layout-cursor32-through-cursor36",
            MAX_REJECTION_ROUNDS,
            SemanticIds {
                cursor32_state_input: STATE32.0,
                root6_input: ROOT6.0,
                alpha6_output: ALPHA6.0,
                round6_root_input: ROOT7.0,
                challenge7_output: ALPHA7.0,
                operation_boundaries: [32, 33, 34, 35],
            },
        ),
        Origin::Captured(_) => (
            CAPTURED_SCHEMA,
            "sn2",
            "full-cairo-plan-cursor32-through-cursor36",
            CAIRO_MAX_REJECTION_ROUNDS,
            SemanticIds {
                cursor32_state_input: 0,
                root6_input: 0x1_0018,
                alpha6_output: 0x1_0019,
                round6_root_input: 0x1_001c,
                challenge7_output: 0x1_001d,
                operation_boundaries: [0x1_001a, 0x1_001b, 0x1_001e, 0x1_001f],
            },
        ),
    };
    let payload_name = format!("fri-round6-{family}-{}.payload.bin", case.name());
    let index = Index {
        schema_version: schema,
        fixture_id: format!("{family}.fri.round6.{}.v1", case.name()),
        fixture_class: case.class(),
        production_admissible: matches!(context.origin, Origin::Captured(_)),
        source: source_index(match context.origin {
            Origin::Synthetic => None,
            Origin::Captured(capture) => Some(capture),
        }),
        payload: Payload {
            path: payload_name,
            byte_length: payload.len(),
            sha256: sha256_hex(&payload),
            encoding: "headerless-le-u32-v1",
        },
        shape: Shape {
            source_circle_log: SOURCE_CIRCLE_LOG,
            predecessor_tree_log: ENTRY_LOG,
            entry_line_log: ENTRY_LOG,
            exit_line_log: EXIT_LOG,
            fold_count: ENTRY_LOG - EXIT_LOG,
            packed_leaf_log: PACKED_LEAF_LOG,
            leaf_count: 1 << (EXIT_LOG - PACKED_LEAF_LOG),
            inverse_twiddle_words: 56,
            normalized_twiddle_offsets_words: [0, 32, 48],
            full_twiddle_offsets_words: [
                FULL_TWIDDLE_WORDS - 64,
                FULL_TWIDDLE_WORDS - 32,
                FULL_TWIDDLE_WORDS - 16,
            ],
            fold_input_words: [64, 32, 16],
        },
        transcript: Transcript {
            schedule_scope: scope,
            protocol_tag: BLAKE2S_TRANSCRIPT_PROTOCOL_TAG,
            max_rejection_rounds: max_rejections,
            semantic_ids,
            chains: chain_strings(context.chains),
        },
        predecessor_seal: PredecessorSeal {
            status: "PASS",
            cursor32_state_words_sha256: sha256_words(&context.cursor32),
            cursor32_digest_blake2s: words_hash(&context.cursor32[..8])?.to_string(),
            cursor32_n_draws: context.cursor32[8],
            root6_blake2s: root6.to_string(),
            root6_words_sha256: sha256_words(&hash_words(root6)),
            root6_leaf_count: root6_leaves.len(),
            alpha6_words_sha256: sha256_words(&alpha6_words),
            observed_primary_match: match (context.origin, case) {
                (Origin::Captured(_), Case::Primary) => Some("PASS"),
                _ => None,
            },
            derivation: [
                "commit entry_pong as a canonical packed-leaf log-6 FRI tree",
                "absorb root6 with Blake2s Merkle-channel semantics at cursor32",
                "draw alpha6 with bounded canonical rejection semantics at cursor33",
                "match the two-operation prefix against the complete cursor32-cursor36 replay",
            ],
        },
        oracle: Oracle {
            candidate_independent: true,
            fold: "stwo::prover::backend::cpu::fold_line_cpu",
            commitment: "CpuBackend PackLeavesOps + MerkleOpsLifted<Blake2sMerkleHasher>",
            transcript: "stwo_backend_cuda::replay_blake2s_reference plus independent FNV encoding",
        },
        chunks: chunk_index,
    };
    Ok(Artifact {
        payload,
        index,
        entry: entry_words,
        root: root_words,
        challenge,
        chains: context.chains,
    })
}

fn synthetic_entry_values() -> SecureColumnByCoords<CpuBackend> {
    (0..1 << ENTRY_LOG)
        .map(|row| {
            SecureField::from_m31_array(std::array::from_fn(|coord| {
                let x = ((row as u64 + 1) * [17, 257, 65_537, 1_000_003][coord]
                    + (coord as u64 + 3) * 97)
                    % u64::from(P);
                BaseField::from(x as u32)
            }))
        })
        .collect()
}

fn case_values(
    primary: &SecureColumnByCoords<CpuBackend>,
    case: Case,
) -> SecureColumnByCoords<CpuBackend> {
    let mut values = primary.to_vec();
    if matches!(case, Case::Hostile) {
        values.rotate_right(1);
        values[0] = secure_from([P - 1, 0, 1, P - 2]);
        values[31] = secure_from([0, P - 1, P - 2, 1]);
        values[63] = secure_from([1, 0, P - 1, P - 2]);
    }
    values.into_iter().collect()
}

fn decode_column_words(words: &[u32]) -> Result<SecureColumnByCoords<CpuBackend>, String> {
    if words.len() != 256 || words.iter().any(|word| *word >= P) {
        return Err("captured entry_pong is not 256 canonical M31 words".into());
    }
    Ok((0..64)
        .map(|row| {
            SecureField::from_m31_array(std::array::from_fn(|coord| {
                BaseField::from(words[coord * 64 + row])
            }))
        })
        .collect())
}

fn fold_three(
    values: SecureColumnByCoords<CpuBackend>,
    alpha: SecureField,
) -> SecureColumnByCoords<CpuBackend> {
    let domain = LineDomain::new(Coset::half_odds(SOURCE_CIRCLE_LOG - 1))
        .repeated_double(SOURCE_CIRCLE_LOG - 1 - ENTRY_LOG);
    let mut evaluation = LineEvaluation::new(domain, values);
    let alpha2 = alpha * alpha;
    for folding_alpha in [alpha, alpha2, alpha2 * alpha2] {
        evaluation = fold_line_cpu(&evaluation, folding_alpha);
    }
    evaluation.values
}

fn inverse_twiddles() -> Vec<u32> {
    let mut domain = LineDomain::new(Coset::half_odds(SOURCE_CIRCLE_LOG - 1))
        .repeated_double(SOURCE_CIRCLE_LOG - 1 - ENTRY_LOG);
    let mut words = Vec::with_capacity(56);
    while domain.log_size() > EXIT_LOG {
        for i in 0..domain.size() / 2 {
            words.push(
                domain
                    .at(bit_reverse_index(i << 1, domain.log_size()))
                    .inverse()
                    .0,
            );
        }
        domain = domain.double();
    }
    words
}

fn commit(
    values: &SecureColumnByCoords<CpuBackend>,
) -> Result<(Vec<Blake2sHash>, Blake2sHash), String> {
    if values.len() < 4 || !values.len().is_power_of_two() {
        return Err(
            "packed FRI commitment requires a power-of-two evaluation of at least 4".into(),
        );
    }
    let refs = [
        &values.columns[0],
        &values.columns[1],
        &values.columns[2],
        &values.columns[3],
    ];
    let packed = <CpuBackend as PackLeavesOps>::pack_leaves_input(&refs);
    let packed_refs = packed.iter().collect::<Vec<_>>();
    let mut layer = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
        &packed_refs,
        values.len().ilog2() - PACKED_LEAF_LOG,
    );
    let leaves = layer.clone();
    while layer.len() > 1 {
        layer = <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(&layer);
    }
    Ok((leaves, layer[0]))
}

fn column_words(values: &SecureColumnByCoords<CpuBackend>) -> Vec<u32> {
    values
        .columns
        .iter()
        .flat_map(|column| column.iter().map(|value| value.0))
        .collect()
}

fn secure(words: &[u32]) -> Result<SecureField, String> {
    let words: [u32; 4] = words
        .try_into()
        .map_err(|_| "secure field encoding must contain four words")?;
    if words.iter().any(|word| *word >= P) {
        return Err("secure field encoding contains a non-canonical M31 word".into());
    }
    Ok(secure_from(words))
}

fn secure_from(words: [u32; 4]) -> SecureField {
    SecureField::from_m31_array(words.map(BaseField::from))
}

fn encode_chunks(chunks: &[(&'static str, Vec<u32>)]) -> Result<(Vec<u8>, Vec<Chunk>), String> {
    let mut payload = Vec::with_capacity(PAYLOAD_BYTES);
    let mut index = Vec::with_capacity(chunks.len());
    for (id, words) in chunks {
        let offset = payload.len();
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        payload.extend(&bytes);
        index.push(Chunk {
            id,
            offset_bytes: offset,
            byte_length: bytes.len(),
            word_count: words.len(),
            sha256: sha256_hex(&bytes),
        });
    }
    if payload.len() != PAYLOAD_BYTES {
        return Err(format!(
            "FRI round-6 payload is {} bytes, expected {PAYLOAD_BYTES}",
            payload.len()
        ));
    }
    Ok((payload, index))
}

fn sha256_words(words: &[u32]) -> String {
    sha256_hex(
        &words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn chain_strings(chains: [u64; 5]) -> Chains {
    let mut encoded = chains.map(|chain| format!("{chain:016x}"));
    Chains {
        c32: std::mem::take(&mut encoded[0]),
        c33: std::mem::take(&mut encoded[1]),
        c34: std::mem::take(&mut encoded[2]),
        c35: std::mem::take(&mut encoded[3]),
        c36: std::mem::take(&mut encoded[4]),
    }
}

fn validate_pair(primary: &Artifact, hostile: &Artifact) -> Result<(), String> {
    if primary.chains != hostile.chains {
        return Err("hostile fixture changed the transcript graph chains".into());
    }
    if primary.entry == hostile.entry
        || primary.root == hostile.root
        || primary.challenge == hostile.challenge
    {
        return Err("hostile mutation did not propagate through root and challenge".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "fri_round6_tests.rs"]
mod tests;
