use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::artifact_io::{self, ChunkReader, ChunkWriter, M31_ROW_ENCODING};
use crate::model::{
    self, Boundary, CommonFixture, FixtureOracle, ProofIdentity, SemanticIdentity, INDEX_SCHEMA,
    M31_P, OPERATION,
};
use crate::oracle::SimdCrosscheckEvidence;
use crate::oracle::MAX_SIMD_ROWS_PER_SEGMENT;
use crate::pedersen_builtin_semantics::{self as semantics, InputRow, OUTPUT_WORDS};

pub const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CHUNKS: usize = 65_536;
const MAX_CHUNK_ROWS: u64 = 65_536;
const MAX_TABLE_WORDS: u64 = 1 << 28;
const MAX_TABLE_BYTES: u64 = MAX_TABLE_WORDS * 4;
const MAX_INPUT_CHUNK_BYTES: u64 = MAX_CHUNK_ROWS * 3 * 4;
const MAX_EXPECTED_CHUNK_BYTES: u64 = MAX_CHUNK_ROWS * OUTPUT_WORDS as u64 * 4;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureIndex {
    schema_version: String,
    fixture_id: String,
    fixture_class: String,
    proof_identity: ProofIdentity,
    semantic_identity: SemanticIdentity,
    full_proof_semantic_hash: Option<String>,
    boundary: Boundary,
    oracle: FixtureOracle,
    exporter_executable_sha256: String,
    production_crosscheck: model::SourceClosure,
    semantic_payload: PayloadIndex,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadIndex {
    field: String,
    row_count: u64,
    address_to_id: TableChunk,
    input_chunks: Vec<RowChunk>,
    expected_chunks: Vec<RowChunk>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TableChunk {
    encoding: String,
    path: String,
    sha256: String,
    element_count: u64,
    byte_len: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowChunk {
    encoding: String,
    path: String,
    sha256: String,
    row_start: u64,
    row_count: u64,
    words_per_row: u32,
    byte_len: u64,
}

#[derive(Debug, Serialize)]
struct ArtifactIndex {
    schema_version: &'static str,
    fixture_id: String,
    fixture_index_sha256: String,
    semantic_identity: SemanticIdentity,
    full_proof_semantic_hash: Option<String>,
    semantic_operation: &'static str,
    oracle: model::OracleIdentity,
    production_crosscheck: model::SourceClosure,
    validation: StreamingValidation,
    expected: ExpectedIndex,
}

#[derive(Debug, Serialize)]
struct StreamingValidation {
    scalar_golden_rows: u64,
    production_simd_crosschecked_rows: u64,
    words_compared_to_checked_fixture: u64,
    checked_fixture_match: bool,
    peak_chunk_rows: u64,
    peak_chunk_payload_bytes: u64,
    resident_table_words: u64,
    max_simd_rows_per_segment: usize,
    estimated_exporter_peak_bytes: u64,
    max_exporter_estimated_peak_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ExpectedIndex {
    encoding: &'static str,
    words_per_row: usize,
    row_count: u64,
    logical_sha256: String,
    chunks: Vec<OutputChunk>,
}

#[derive(Debug, Serialize)]
struct OutputChunk {
    encoding: &'static str,
    path: String,
    sha256: String,
    row_start: u64,
    row_count: u64,
    words_per_row: usize,
    byte_len: u64,
}

pub fn export(
    index_path: &Path,
    output: &Path,
    check: bool,
    simd_evidence: Option<&SimdCrosscheckEvidence>,
) -> Result<(), String> {
    let (bytes, fixture_index_sha256) =
        model::load_bounded(index_path, MAX_INDEX_BYTES, "fixture index JSON")?;
    let index: FixtureIndex = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", index_path.display()))?;
    validate_index(&index)?;
    let row_count = index.semantic_payload.row_count;
    let production_simd_crosschecked_rows = simd_evidence.map_or(0, SimdCrosscheckEvidence::rows);
    if production_simd_crosschecked_rows != 0 && production_simd_crosschecked_rows != row_count {
        return Err(format!(
            "production SIMD evidence covers {production_simd_crosschecked_rows} rows, expected {row_count}"
        ));
    }
    let source_bytes = index
        .proof_identity
        .source_prover_input_bytes
        .ok_or("production fixture lacks source ProverInput byte length")?;
    let table_words = usize::try_from(index.semantic_payload.address_to_id.element_count)
        .map_err(|_| "table words exceed host usize")?;
    let rows = usize::try_from(row_count).map_err(|_| "row count exceeds host usize")?;
    crate::producer::enforce_exporter_budget(source_bytes, table_words, rows)?;
    let peak_estimate =
        crate::producer::estimated_exporter_peak_bytes(source_bytes, table_words, rows)?;
    let table = read_table(index_path, &index.semantic_payload.address_to_id)?;
    semantics::validate_table(&table)?;
    let mut logical_hasher = Sha256::new();
    let mut output_chunks = Vec::with_capacity(index.semantic_payload.input_chunks.len());
    let mut fixture_match = true;
    let mut peak_chunk_rows = 0;
    let mut peak_chunk_payload_bytes = 0;

    for (serial, (input_chunk, expected_chunk)) in index
        .semantic_payload
        .input_chunks
        .iter()
        .zip(&index.semantic_payload.expected_chunks)
        .enumerate()
    {
        let inputs = read_inputs(index_path, input_chunk)?;
        let rows = inputs
            .into_iter()
            .map(|input| semantics::evaluate(&table, input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut expected = ChunkReader::open(
            index_path,
            &expected_chunk.path,
            &expected_chunk.sha256,
            expected_chunk.byte_len,
            MAX_EXPECTED_CHUNK_BYTES,
        )?;
        let mut output_chunk = ChunkWriter::create(output, serial)?;
        for (local_row, row) in rows.iter().enumerate() {
            for (word_index, word) in semantics::flatten(row).into_iter().enumerate() {
                let checked = expected.read_word()?;
                if checked >= M31_P {
                    return Err(format!(
                        "expected chunk row {} word {word_index}={checked} is not canonical M31",
                        expected_chunk.row_start + local_row as u64
                    ));
                }
                fixture_match &= word == checked;
                logical_hasher.update(word.to_le_bytes());
                output_chunk.write_word(word)?;
            }
        }
        expected.finish()?;
        let stored = output_chunk.finish(output)?;
        output_chunks.push(OutputChunk {
            encoding: M31_ROW_ENCODING,
            path: stored.relative_path,
            sha256: stored.sha256,
            row_start: input_chunk.row_start,
            row_count: input_chunk.row_count,
            words_per_row: OUTPUT_WORDS,
            byte_len: stored.byte_len,
        });
        peak_chunk_rows = peak_chunk_rows.max(input_chunk.row_count);
        peak_chunk_payload_bytes = peak_chunk_payload_bytes.max(
            input_chunk
                .byte_len
                .checked_add(expected_chunk.byte_len)
                .ok_or("peak chunk byte count overflow")?,
        );
    }
    if check && !fixture_match {
        return Err(
            "checked fixture chunks differ from the canonical host export; index not sealed".into(),
        );
    }
    let artifact = ArtifactIndex {
        schema_version: "stwo.gpu-lab.host-oracle-index.v2",
        fixture_id: index.fixture_id,
        fixture_index_sha256,
        semantic_identity: index.semantic_identity,
        full_proof_semantic_hash: index.full_proof_semantic_hash,
        semantic_operation: OPERATION,
        oracle: model::oracle_identity(),
        production_crosscheck: index.production_crosscheck,
        validation: StreamingValidation {
            scalar_golden_rows: row_count,
            production_simd_crosschecked_rows,
            words_compared_to_checked_fixture: if check {
                row_count
                    .checked_mul(OUTPUT_WORDS as u64)
                    .ok_or("checked word count overflow")?
            } else {
                0
            },
            checked_fixture_match: fixture_match,
            peak_chunk_rows,
            peak_chunk_payload_bytes,
            resident_table_words: index.semantic_payload.address_to_id.element_count,
            max_simd_rows_per_segment: MAX_SIMD_ROWS_PER_SEGMENT,
            estimated_exporter_peak_bytes: peak_estimate,
            max_exporter_estimated_peak_bytes: crate::producer::MAX_EXPORTER_ESTIMATED_PEAK_BYTES,
        },
        expected: ExpectedIndex {
            encoding: M31_ROW_ENCODING,
            words_per_row: OUTPUT_WORDS,
            row_count,
            logical_sha256: format!("{:x}", logical_hasher.finalize()),
            chunks: output_chunks,
        },
    };
    artifact_io::write_immutable_json(output, &artifact)
}

fn validate_index(index: &FixtureIndex) -> Result<(), String> {
    if index.schema_version != INDEX_SCHEMA {
        return Err(format!(
            "unsupported fixture index schema: {}",
            index.schema_version
        ));
    }
    model::validate_common(
        CommonFixture {
            fixture_id: &index.fixture_id,
            fixture_class: &index.fixture_class,
            proof_identity: &index.proof_identity,
            semantic_identity: &index.semantic_identity,
            full_proof_semantic_hash: &index.full_proof_semantic_hash,
            boundary: &index.boundary,
            oracle: &index.oracle,
        },
        false,
    )?;
    if index.production_crosscheck != model::production_crosscheck()? {
        return Err("production SIMD crosscheck source closure changed".into());
    }
    model::validate_sha256(
        &index.exporter_executable_sha256,
        "exporter executable sha256",
    )?;
    let payload = &index.semantic_payload;
    if payload.field != "M31" || payload.row_count == 0 {
        return Err("streaming payload field must be M31 and row_count non-zero".into());
    }
    validate_table(&payload.address_to_id)?;
    if payload.input_chunks.is_empty()
        || payload.input_chunks.len() > MAX_CHUNKS
        || payload.input_chunks.len() != payload.expected_chunks.len()
    {
        return Err(format!(
            "input/expected chunk count must match in 1..={MAX_CHUNKS}"
        ));
    }
    validate_row_chunks(&payload.input_chunks, payload.row_count, 3, "input")?;
    validate_row_chunks(
        &payload.expected_chunks,
        payload.row_count,
        OUTPUT_WORDS as u32,
        "expected",
    )?;
    for (input, expected) in payload.input_chunks.iter().zip(&payload.expected_chunks) {
        if (input.row_start, input.row_count) != (expected.row_start, expected.row_count) {
            return Err(format!(
                "input/expected chunk boundary mismatch at row {}",
                input.row_start
            ));
        }
    }
    Ok(())
}

fn validate_table(table: &TableChunk) -> Result<(), String> {
    if table.encoding != M31_ROW_ENCODING || table.element_count < 4 {
        return Err("address_to_id must be m31-le-u32-row-major-v1 with at least 4 words".into());
    }
    if table.element_count > MAX_TABLE_WORDS {
        return Err(format!(
            "address_to_id has {} words; bounded maximum is {MAX_TABLE_WORDS}",
            table.element_count
        ));
    }
    let expected = table
        .element_count
        .checked_mul(4)
        .ok_or("address_to_id byte count overflow")?;
    if table.byte_len != expected {
        return Err(format!(
            "address_to_id byte_len {} != expected {expected}",
            table.byte_len
        ));
    }
    Ok(())
}

fn validate_row_chunks(
    chunks: &[RowChunk],
    row_count: u64,
    words_per_row: u32,
    label: &str,
) -> Result<(), String> {
    let mut next_row = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.encoding != M31_ROW_ENCODING
            || chunk.row_start != next_row
            || chunk.row_count == 0
            || chunk.row_count > MAX_CHUNK_ROWS
            || chunk.words_per_row != words_per_row
        {
            return Err(format!("invalid {label} chunk {index} geometry/encoding"));
        }
        let expected_bytes = chunk
            .row_count
            .checked_mul(words_per_row as u64)
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| format!("{label} chunk {index} byte count overflow"))?;
        if chunk.byte_len != expected_bytes {
            return Err(format!(
                "{label} chunk {index} byte_len {} != {expected_bytes}",
                chunk.byte_len
            ));
        }
        next_row = next_row
            .checked_add(chunk.row_count)
            .ok_or_else(|| format!("{label} row coverage overflow"))?;
    }
    if next_row != row_count {
        return Err(format!(
            "{label} chunks cover {next_row} rows, expected {row_count}"
        ));
    }
    Ok(())
}

fn read_table(index_path: &Path, table: &TableChunk) -> Result<Vec<u32>, String> {
    let count = usize::try_from(table.element_count)
        .map_err(|_| "address_to_id does not fit host usize")?;
    let mut reader = ChunkReader::open(
        index_path,
        &table.path,
        &table.sha256,
        table.byte_len,
        MAX_TABLE_BYTES,
    )?;
    let mut words = Vec::with_capacity(count);
    for _ in 0..count {
        words.push(reader.read_word()?);
    }
    reader.finish()?;
    Ok(words)
}

fn read_inputs(index_path: &Path, chunk: &RowChunk) -> Result<Vec<InputRow>, String> {
    let count = usize::try_from(chunk.row_count).map_err(|_| "chunk rows do not fit host usize")?;
    let mut reader = ChunkReader::open(
        index_path,
        &chunk.path,
        &chunk.sha256,
        chunk.byte_len,
        MAX_INPUT_CHUNK_BYTES,
    )?;
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        rows.push(InputRow {
            segment_start: reader.read_word()?,
            enabler: reader.read_word()?,
            iota: reader.read_word()?,
        });
    }
    reader.finish()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_gap_in_streaming_chunks() {
        let chunks = vec![RowChunk {
            encoding: M31_ROW_ENCODING.into(),
            path: format!("chunks/sha256/{}.m31le", "a".repeat(64)),
            sha256: "a".repeat(64),
            row_start: 1,
            row_count: 4,
            words_per_row: 3,
            byte_len: 48,
        }];
        assert!(validate_row_chunks(&chunks, 4, 3, "input").is_err());
    }
}
