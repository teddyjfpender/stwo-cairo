use serde::Serialize;

use crate::fri_round6_capture::{CaptureShape, CaptureSource};

#[derive(Serialize)]
pub struct Index {
    pub schema_version: &'static str,
    pub fixture_id: String,
    pub fixture_class: &'static str,
    pub production_admissible: bool,
    pub source: Source,
    pub payload: Payload,
    pub shape: Shape,
    pub transcript: Transcript,
    pub predecessor_seal: PredecessorSeal,
    pub oracle: Oracle,
    pub chunks: Vec<Chunk>,
}

#[derive(Serialize)]
pub struct Source {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_seed_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<CaptureSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_shape: Option<CaptureShape>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_protocol_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cairo_schedule_key: Option<String>,
}

#[derive(Serialize)]
pub struct Payload {
    pub path: String,
    pub byte_length: usize,
    pub sha256: String,
    pub encoding: &'static str,
}

#[derive(Serialize)]
pub struct Shape {
    pub source_circle_log: u32,
    pub predecessor_tree_log: u32,
    pub entry_line_log: u32,
    pub exit_line_log: u32,
    pub fold_count: u32,
    pub packed_leaf_log: u32,
    pub leaf_count: u32,
    pub inverse_twiddle_words: u32,
    pub normalized_twiddle_offsets_words: [u32; 3],
    pub full_twiddle_offsets_words: [u32; 3],
    pub fold_input_words: [u32; 3],
}

#[derive(Serialize)]
pub struct Transcript {
    pub schedule_scope: &'static str,
    pub protocol_tag: &'static str,
    pub max_rejection_rounds: u32,
    pub semantic_ids: SemanticIds,
    pub chains: Chains,
}

#[derive(Serialize)]
pub struct SemanticIds {
    pub cursor32_state_input: u32,
    pub root6_input: u32,
    pub alpha6_output: u32,
    pub round6_root_input: u32,
    pub challenge7_output: u32,
    pub operation_boundaries: [u32; 4],
}

#[derive(Clone, Serialize)]
pub struct Chains {
    pub c32: String,
    pub c33: String,
    pub c34: String,
    pub c35: String,
    pub c36: String,
}

#[derive(Serialize)]
pub struct PredecessorSeal {
    pub status: &'static str,
    pub cursor32_state_words_sha256: String,
    pub cursor32_digest_blake2s: String,
    pub cursor32_n_draws: u32,
    pub root6_blake2s: String,
    pub root6_words_sha256: String,
    pub root6_leaf_count: usize,
    pub alpha6_words_sha256: String,
    pub observed_primary_match: Option<&'static str>,
    pub derivation: [&'static str; 4],
}

#[derive(Serialize)]
pub struct Oracle {
    pub candidate_independent: bool,
    pub fold: &'static str,
    pub commitment: &'static str,
    pub transcript: &'static str,
}

#[derive(Serialize)]
pub struct Chunk {
    pub id: &'static str,
    pub offset_bytes: usize,
    pub byte_length: usize,
    pub word_count: usize,
    pub sha256: String,
}
