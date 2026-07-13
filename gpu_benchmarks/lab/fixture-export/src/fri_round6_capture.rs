use std::path::Path;

use serde::{Deserialize, Serialize};
use stwo_backend_cuda::{
    Blake2sTranscriptSchedule, TranscriptBoundaryId, TranscriptInputId, TranscriptOperation,
    TranscriptOutputId, TranscriptStart, BLAKE2S_TRANSCRIPT_PROTOCOL_TAG,
};

use crate::model::{load_bounded, sha256_hex, validate_sha256, M31_P};

pub const CAPTURE_SCHEMA: &str = "stwo.gpu-lab.fri-round6-capture-seed.v1";
const MAX_CAPTURE_BYTES: u64 = 1024 * 1024;
const MAX_REJECTION_ROUNDS: u32 = 64;
const CAIRO_SCHEDULE_TAG: &str = "stwo-cairo.blake2s.transcript.schedule.v1";
const FRI_ID_BASE: u32 = 0x1_0000;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSource {
    pub observer: String,
    pub prover_input_sha256: String,
    pub prover_input_bytes: u64,
    pub proof_shape_id: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureShape {
    pub circle_log_size: u32,
    pub claim_enable_felts: u32,
    pub claim_log_size_felts: u32,
    pub claim_public_data_felts: u32,
    pub interaction_claim_felts: u32,
    pub oods_sampled_values_felts: u32,
    pub interaction_pow_bits: u32,
    pub pcs: PcsShape,
    pub fri_tree_count: u32,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PcsShape {
    pub pow_bits: u32,
    pub log_blowup_factor: u32,
    pub n_queries: u32,
    pub log_last_layer_degree_bound: u32,
    pub fold_step: u32,
    pub lifting_log_size: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSeed {
    schema_version: String,
    source: CaptureSource,
    shape: CaptureShape,
    schedule: ScheduleSeal,
    cursor32_state_words: Vec<u32>,
    cursor32_state_words_sha256: String,
    entry_pong_words: Vec<u32>,
    entry_pong_words_sha256: String,
    observed: Observed,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleSeal {
    device_protocol_key: String,
    cairo_schedule_key: String,
    c32: String,
    c33: String,
    c34: String,
    c35: String,
    c36: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observed {
    pub root6_words: Vec<u32>,
    pub alpha6_words: Vec<u32>,
    pub cursor34_state_words: Vec<u32>,
    pub root7_words: Vec<u32>,
    pub alpha7_words: Vec<u32>,
    pub cursor35_state_words: Vec<u32>,
    pub cursor36_state_words: Vec<u32>,
}

pub struct VerifiedCapture {
    pub capture_sha256: String,
    pub source: CaptureSource,
    pub shape: CaptureShape,
    pub cursor32: Vec<u32>,
    pub entry_words: Vec<u32>,
    pub observed: Observed,
    pub chains: [u64; 5],
    pub device_protocol_key: u64,
    pub cairo_schedule_key: u64,
}

pub fn load(path: &Path, expected_sha256: &str) -> Result<VerifiedCapture, String> {
    validate_sha256(expected_sha256, "FRI capture seed sha256")?;
    let (bytes, actual_sha256) = load_bounded(path, MAX_CAPTURE_BYTES, "FRI capture seed")?;
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "FRI capture seed sha256 {actual_sha256} != required {expected_sha256}"
        ));
    }
    let seed: CaptureSeed = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse FRI capture seed {}: {error}", path.display()))?;
    verify(seed, actual_sha256)
}

fn verify(seed: CaptureSeed, capture_sha256: String) -> Result<VerifiedCapture, String> {
    if seed.schema_version != CAPTURE_SCHEMA {
        return Err(format!(
            "unsupported FRI capture schema: {}",
            seed.schema_version
        ));
    }
    validate_source(&seed.source)?;
    validate_shape(&seed.shape)?;
    validate_words(&seed.cursor32_state_words, 16, false, "cursor32 state")?;
    validate_words(&seed.entry_pong_words, 256, true, "entry_pong")?;
    validate_word_hash(
        &seed.cursor32_state_words,
        &seed.cursor32_state_words_sha256,
        "cursor32 state",
    )?;
    validate_word_hash(
        &seed.entry_pong_words,
        &seed.entry_pong_words_sha256,
        "entry_pong",
    )?;
    validate_observed(&seed.observed)?;

    let operations = cairo_operations(&seed.shape)?;
    let schedule = Blake2sTranscriptSchedule::new(
        TranscriptStart::Default,
        operations.clone(),
        MAX_REJECTION_ROUNDS,
    )
    .map_err(|error| error.to_string())?;
    let chains = independent_prefix_chains(&operations);
    let supplied = [
        parse_u64(&seed.schedule.c32, "C32")?,
        parse_u64(&seed.schedule.c33, "C33")?,
        parse_u64(&seed.schedule.c34, "C34")?,
        parse_u64(&seed.schedule.c35, "C35")?,
        parse_u64(&seed.schedule.c36, "C36")?,
    ];
    if supplied != chains[32..=36] {
        return Err("captured C32-C36 do not match the independently rebuilt Cairo plan".into());
    }
    let device_protocol_key = parse_u64(&seed.schedule.device_protocol_key, "device protocol key")?;
    if device_protocol_key != schedule.protocol_key() {
        return Err("captured device protocol key does not match the rebuilt Cairo plan".into());
    }
    let cairo_schedule_key = parse_u64(&seed.schedule.cairo_schedule_key, "Cairo schedule key")?;
    if cairo_schedule_key != compute_cairo_schedule_key(device_protocol_key, &seed.shape) {
        return Err("captured Cairo schedule key does not match the rebuilt segment plan".into());
    }
    let control = &seed.cursor32_state_words;
    let state_chain = u64::from(control[12]) | (u64::from(control[13]) << 32);
    if control[9] != 32
        || control[10] != 0
        || control[11] != 0
        || control[14] != 0
        || control[15] != 0
        || state_chain != supplied[0]
    {
        return Err("captured cursor32 control state or C32 chain is invalid".into());
    }
    Ok(VerifiedCapture {
        capture_sha256,
        source: seed.source,
        shape: seed.shape,
        cursor32: seed.cursor32_state_words,
        entry_words: seed.entry_pong_words,
        observed: seed.observed,
        chains: supplied,
        device_protocol_key,
        cairo_schedule_key,
    })
}

fn validate_source(source: &CaptureSource) -> Result<(), String> {
    validate_sha256(&source.prover_input_sha256, "source ProverInput sha256")?;
    if source.observer != "stwo-cairo.production-simd-fri-observer.v1"
        || source.prover_input_bytes == 0
        || source.proof_shape_id.is_empty()
    {
        return Err(
            "capture source identity is incomplete or not the production SIMD observer".into(),
        );
    }
    Ok(())
}

fn validate_shape(shape: &CaptureShape) -> Result<(), String> {
    let counts = [
        shape.claim_enable_felts,
        shape.claim_log_size_felts,
        shape.claim_public_data_felts,
        shape.interaction_claim_felts,
        shape.oods_sampled_values_felts,
        shape.pcs.n_queries,
        shape.fri_tree_count,
    ];
    if counts.contains(&0)
        || shape.circle_log_size != 24
        || shape.pcs.lifting_log_size != 24
        || shape.pcs.fold_step != 3
        || shape.fri_tree_count < 8
        || shape.pcs.log_last_layer_degree_bound >= 32
    {
        return Err("capture is not an exact SN2-compatible fold-step-3 FRI shape".into());
    }
    Ok(())
}

fn validate_observed(observed: &Observed) -> Result<(), String> {
    for (words, len, m31, label) in [
        (&observed.root6_words, 8, false, "observed root6"),
        (&observed.alpha6_words, 4, true, "observed alpha6"),
        (
            &observed.cursor34_state_words,
            16,
            false,
            "observed cursor34",
        ),
        (&observed.root7_words, 8, false, "observed root7"),
        (&observed.alpha7_words, 4, true, "observed alpha7"),
        (
            &observed.cursor35_state_words,
            16,
            false,
            "observed cursor35",
        ),
        (
            &observed.cursor36_state_words,
            16,
            false,
            "observed cursor36",
        ),
    ] {
        validate_words(words, len, m31, label)?;
    }
    Ok(())
}

fn validate_words(words: &[u32], len: usize, m31: bool, label: &str) -> Result<(), String> {
    if words.len() != len || (m31 && words.iter().any(|word| *word >= M31_P)) {
        return Err(format!("{label} has an invalid length or field encoding"));
    }
    Ok(())
}

fn validate_word_hash(words: &[u32], expected: &str, label: &str) -> Result<(), String> {
    validate_sha256(expected, &format!("{label} sha256"))?;
    if sha256_hex(&word_bytes(words)) != expected {
        return Err(format!("{label} sha256 mismatch"));
    }
    Ok(())
}

fn cairo_operations(shape: &CaptureShape) -> Result<Vec<TranscriptOperation>, String> {
    let mut ops = vec![
        mix(1, 1, 1),
        mix(2, 2, 2),
        absorb(3, 3),
        mix(10, 10, 1),
        mix(11, 11, shape.claim_enable_felts),
        mix(12, 12, shape.claim_log_size_felts),
        mix(13, 13, 1),
        mix(14, 14, shape.claim_public_data_felts),
        absorb(15, 15),
        absorb(16, 16),
        absorb(20, 20),
        pow(21, 21, shape.interaction_pow_bits),
        draw_many(22, 1, 2),
        mix(30, 22, shape.interaction_claim_felts),
        absorb(31, 23),
        draw(32, 2),
        absorb(40, 24),
        draw(41, 3),
        mix(50, 25, shape.oods_sampled_values_felts),
        draw(51, 4),
    ];
    for layer in 0..shape.fri_tree_count {
        ops.push(absorb(fri_id(layer, 2)?, fri_id(layer, 0)?));
        ops.push(draw(fri_id(layer, 3)?, fri_id(layer, 1)?));
    }
    ops.push(mix(60, 30, 1u32 << shape.pcs.log_last_layer_degree_bound));
    ops.push(pow(61, 31, shape.pcs.pow_bits));
    ops.push(TranscriptOperation::DrawQueries {
        boundary: TranscriptBoundaryId(62),
        output: TranscriptOutputId(5),
        log_domain_size: shape.pcs.lifting_log_size,
        n_queries: shape.pcs.n_queries,
    });
    Ok(ops)
}

fn independent_prefix_chains(operations: &[TranscriptOperation]) -> Vec<u64> {
    let mut hash = StableHash::new();
    hash.bytes(BLAKE2S_TRANSCRIPT_PROTOCOL_TAG.as_bytes());
    hash.u32(MAX_REJECTION_ROUNDS);
    hash.u32(0);
    let mut chains = vec![hash.finish()];
    for operation in operations {
        hash_operation(&mut hash, operation);
        chains.push(hash.finish());
    }
    chains
}

fn hash_operation(hash: &mut StableHash, operation: &TranscriptOperation) {
    hash.u32(operation.boundary().0);
    match *operation {
        TranscriptOperation::MixFelts {
            source, n_felts, ..
        } => hash.words(&[1, source.0, n_felts]),
        TranscriptOperation::MixU32s {
            source, n_words, ..
        } => hash.words(&[2, source.0, n_words]),
        TranscriptOperation::MixU64 { source, .. } => hash.words(&[3, source.0]),
        TranscriptOperation::AbsorbRoot { source, .. } => hash.words(&[4, source.0]),
        TranscriptOperation::AbsorbPowNonce {
            source, pow_bits, ..
        } => hash.words(&[5, source.0, pow_bits]),
        TranscriptOperation::DrawSecureFelt { output, .. } => hash.words(&[6, output.0]),
        TranscriptOperation::DrawSecureFelts {
            output, n_felts, ..
        } => hash.words(&[7, output.0, n_felts]),
        TranscriptOperation::DrawU32s { output, .. } => hash.words(&[8, output.0]),
        TranscriptOperation::DrawQueries {
            output,
            log_domain_size,
            n_queries,
            ..
        } => hash.words(&[9, output.0, log_domain_size, n_queries]),
    }
}

fn compute_cairo_schedule_key(protocol_key: u64, shape: &CaptureShape) -> u64 {
    let mut hash = StableHash::new();
    hash.bytes(CAIRO_SCHEDULE_TAG.as_bytes());
    hash.bytes(&protocol_key.to_le_bytes());
    let mut segments = vec![
        (0, 0, 0, 11),
        (1, 0, 11, 13),
        (2, 0, 13, 16),
        (3, 0, 16, 18),
        (4, 0, 18, 20),
    ];
    for layer in 0..shape.fri_tree_count {
        let start = 20 + 2 * layer;
        segments.push((5, layer, start, start + 2));
    }
    let last = 20 + 2 * shape.fri_tree_count;
    segments.push((6, 0, last, last + 1));
    segments.push((7, 0, last + 1, last + 3));
    for (tag, index, start, end) in segments {
        hash.u32(tag);
        hash.u32(index);
        hash.bytes(&u64::from(start).to_le_bytes());
        hash.bytes(&u64::from(end).to_le_bytes());
    }
    hash.finish()
}

fn mix(boundary: u32, source: u32, n_felts: u32) -> TranscriptOperation {
    TranscriptOperation::MixFelts {
        boundary: TranscriptBoundaryId(boundary),
        source: TranscriptInputId(source),
        n_felts,
    }
}
fn absorb(boundary: u32, source: u32) -> TranscriptOperation {
    TranscriptOperation::AbsorbRoot {
        boundary: TranscriptBoundaryId(boundary),
        source: TranscriptInputId(source),
    }
}
fn pow(boundary: u32, source: u32, pow_bits: u32) -> TranscriptOperation {
    TranscriptOperation::AbsorbPowNonce {
        boundary: TranscriptBoundaryId(boundary),
        source: TranscriptInputId(source),
        pow_bits,
    }
}
fn draw(boundary: u32, output: u32) -> TranscriptOperation {
    TranscriptOperation::DrawSecureFelt {
        boundary: TranscriptBoundaryId(boundary),
        output: TranscriptOutputId(output),
    }
}
fn draw_many(boundary: u32, output: u32, n_felts: u32) -> TranscriptOperation {
    TranscriptOperation::DrawSecureFelts {
        boundary: TranscriptBoundaryId(boundary),
        output: TranscriptOutputId(output),
        n_felts,
    }
}
fn fri_id(layer: u32, offset: u32) -> Result<u32, String> {
    layer
        .checked_mul(4)
        .and_then(|value| FRI_ID_BASE.checked_add(value + offset))
        .ok_or_else(|| "FRI semantic ID overflow".into())
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be 16 lowercase hexadecimal characters"
        ));
    }
    u64::from_str_radix(value, 16).map_err(|error| format!("parse {label}: {error}"))
}

fn word_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[derive(Clone, Copy)]
struct StableHash(u64);

impl StableHash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }
    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }
    fn words(&mut self, words: &[u32]) {
        words.iter().for_each(|word| self.u32(*word));
    }
    fn finish(self) -> u64 {
        self.0
    }
}
