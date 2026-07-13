//! Deterministic real-ProverInput fixture production for the first replacement slab.

use std::path::Path;

use serde::Serialize;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::builtins::PEDERSEN_BUILTIN_MEMORY_CELLS;

use crate::artifact_io::{self, ChunkWriter, M31_ROW_ENCODING};
use crate::model::{
    self, Boundary, FixtureOracle, ProofIdentity, SemanticIdentity, INDEX_ORACLE_IMPLEMENTATION,
    INDEX_ORACLE_PROVENANCE, INDEX_ORACLE_VERSION, INDEX_SCHEMA, OPERATION,
};
use crate::oracle::{SimdCrosscheck, MAX_SIMD_ROWS_PER_SEGMENT};
use crate::pedersen_builtin_semantics::{self as semantics, InputRow, SemanticRow, OUTPUT_WORDS};

pub const MAX_EXPORTER_ESTIMATED_PEAK_BYTES: u64 = 1024 * 1024 * 1024;
const ROWS_PER_CHUNK: usize = 65_536;

#[derive(Serialize)]
struct FixtureIndex {
    schema_version: &'static str,
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

#[derive(Serialize)]
struct PayloadIndex {
    field: &'static str,
    row_count: u64,
    address_to_id: TableChunk,
    input_chunks: Vec<RowChunk>,
    expected_chunks: Vec<RowChunk>,
}

#[derive(Serialize)]
struct TableChunk {
    encoding: &'static str,
    path: String,
    sha256: String,
    element_count: u64,
    byte_len: u64,
}

#[derive(Serialize)]
struct RowChunk {
    encoding: &'static str,
    path: String,
    sha256: String,
    row_start: u64,
    row_count: u64,
    words_per_row: u32,
    byte_len: u64,
}

struct OpenChunk {
    row_start: usize,
    rows: usize,
    input: ChunkWriter,
    expected: ChunkWriter,
}

struct SegmentSink<'a> {
    index_path: &'a Path,
    total_rows: usize,
    next_row: usize,
    serial: usize,
    open: Option<OpenChunk>,
    inputs: Vec<RowChunk>,
    expected: Vec<RowChunk>,
}

impl<'a> SegmentSink<'a> {
    fn new(index_path: &'a Path, total_rows: usize) -> Self {
        Self {
            index_path,
            total_rows,
            next_row: 0,
            serial: 1,
            open: None,
            inputs: Vec::new(),
            expected: Vec::new(),
        }
    }

    fn push(&mut self, segment_start: u32, row: usize, output: &SemanticRow) -> Result<(), String> {
        if row != self.next_row {
            return Err(format!(
                "canonical row stream is non-contiguous: got {row}, expected {}",
                self.next_row
            ));
        }
        if self.open.is_none() {
            let input = ChunkWriter::create(self.index_path, self.serial)?;
            let expected = ChunkWriter::create(self.index_path, self.serial + 1)?;
            self.serial += 2;
            self.open = Some(OpenChunk {
                row_start: row,
                rows: 0,
                input,
                expected,
            });
        }
        let chunk = self.open.as_mut().expect("chunk opened above");
        for word in [
            segment_start,
            1,
            u32::try_from(row).map_err(|_| "row exceeds u32")?,
        ] {
            chunk.input.write_word(word)?;
        }
        for word in semantics::flatten(output) {
            chunk.expected.write_word(word)?;
        }
        chunk.rows += 1;
        self.next_row += 1;
        if chunk.rows == ROWS_PER_CHUNK || self.next_row == self.total_rows {
            self.finish_open()?;
        }
        Ok(())
    }

    fn finish_open(&mut self) -> Result<(), String> {
        let chunk = self.open.take().ok_or("no open row chunk")?;
        let input = chunk.input.finish(self.index_path)?;
        let expected = chunk.expected.finish(self.index_path)?;
        let row_start = u64::try_from(chunk.row_start).map_err(|_| "row start exceeds u64")?;
        let row_count = u64::try_from(chunk.rows).map_err(|_| "row count exceeds u64")?;
        self.inputs.push(RowChunk {
            encoding: M31_ROW_ENCODING,
            path: input.relative_path,
            sha256: input.sha256,
            row_start,
            row_count,
            words_per_row: 3,
            byte_len: input.byte_len,
        });
        self.expected.push(RowChunk {
            encoding: M31_ROW_ENCODING,
            path: expected.relative_path,
            sha256: expected.sha256,
            row_start,
            row_count,
            words_per_row: OUTPUT_WORDS as u32,
            byte_len: expected.byte_len,
        });
        Ok(())
    }

    fn finish(self) -> Result<(Vec<RowChunk>, Vec<RowChunk>), String> {
        if self.open.is_some() || self.next_row != self.total_rows {
            return Err(format!(
                "canonical row stream ended at {}, expected {}",
                self.next_row, self.total_rows
            ));
        }
        Ok((self.inputs, self.expected))
    }
}

pub fn generate(
    prover_input_path: &Path,
    expected_prover_input_sha256: &str,
    fixture_class: &str,
    index_path: &Path,
    oracle_output: &Path,
    exporter_executable_sha256: &str,
) -> Result<(), String> {
    model::validate_sha256(expected_prover_input_sha256, "expected ProverInput sha256")?;
    model::validate_sha256(exporter_executable_sha256, "exporter executable sha256")?;
    if !matches!(
        fixture_class,
        "representative" | "stress" | "unseen-same-class"
    ) {
        return Err(
            "real ProverInput fixture class must be representative, stress, or unseen-same-class"
                .into(),
        );
    }
    crate::source_snapshot::reject_aliases(prover_input_path, index_path, oracle_output)?;
    let (input, input_sha256, input_bytes) =
        crate::source_snapshot::read_pinned(prover_input_path, expected_prover_input_sha256)?;
    let ProverInput {
        memory,
        builtin_segments,
        ..
    } = input;
    let segment = builtin_segments
        .pedersen_builtin
        .ok_or("ProverInput has no Pedersen builtin segment")?;
    let segment_start =
        u32::try_from(segment.begin_addr).map_err(|_| "Pedersen segment start exceeds u32")?;
    let segment_words = segment
        .stop_ptr
        .checked_sub(segment.begin_addr)
        .ok_or("Pedersen segment stop precedes start")?;
    if !segment_words.is_multiple_of(PEDERSEN_BUILTIN_MEMORY_CELLS) {
        return Err("Pedersen segment length is not a whole number of instances".into());
    }
    let row_count = segment_words / PEDERSEN_BUILTIN_MEMORY_CELLS;
    if row_count < 16 || !row_count.is_power_of_two() || row_count > MAX_SIMD_ROWS_PER_SEGMENT {
        return Err(format!(
            "Pedersen instance count {row_count} is outside the supported canonical SIMD domain"
        ));
    }
    if segment.stop_ptr > memory.address_to_id.len() || segment.begin_addr == 0 {
        return Err("Pedersen segment is outside canonical address_to_id memory".into());
    }
    enforce_exporter_budget(input_bytes, memory.address_to_id.len(), row_count)?;
    let table = memory
        .address_to_id
        .into_iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let table_chunk = store_table(index_path, &table)?;
    semantics::validate_table(&table)?;
    let oracle = SimdCrosscheck::new(&table)?;
    let mut sink = SegmentSink::new(index_path, row_count);
    let simd_evidence = oracle.visit_segment(segment_start, row_count, |row, simd| {
        let iota = u32::try_from(row).map_err(|_| "row exceeds u32")?;
        let golden = semantics::evaluate(
            &table,
            InputRow {
                segment_start,
                enabler: 1,
                iota,
            },
        )?;
        if simd != golden {
            return Err(format!(
                "row {row}: production SIMD bytes differ from candidate-free scalar golden"
            ));
        }
        sink.push(segment_start, row, &golden)
    })?;
    let (input_chunks, expected_chunks) = sink.finish()?;

    let fixture_id =
        format!("{fixture_class}.witness_pedersen_builtin.prover-input-{input_sha256}.v1");
    let (reference_sources, reference_closure_sha256) = model::reference_evaluator()?;
    let proof_identity = ProofIdentity {
        kind: "real_prover_input_kernel_slice".into(),
        id: fixture_id.clone(),
        full_proof: false,
        source_prover_input_sha256: Some(input_sha256),
        source_prover_input_bytes: Some(input_bytes),
    };
    let semantic_identity = semantics::semantic_identity()?;
    let production_crosscheck = model::production_crosscheck()?;
    let fixture = FixtureIndex {
        schema_version: INDEX_SCHEMA,
        fixture_id,
        fixture_class: fixture_class.into(),
        proof_identity,
        semantic_identity,
        full_proof_semantic_hash: None,
        boundary: Boundary {
            operation: OPERATION.into(),
            entry: semantics::ENTRY_BOUNDARY.into(),
            exit: semantics::EXIT_BOUNDARY.into(),
        },
        oracle: FixtureOracle {
            implementation: INDEX_ORACLE_IMPLEMENTATION.into(),
            version: INDEX_ORACLE_VERSION.into(),
            candidate_independent: true,
            provenance: INDEX_ORACLE_PROVENANCE.into(),
            reference_closure_sha256,
            reference_sources,
        },
        exporter_executable_sha256: exporter_executable_sha256.into(),
        production_crosscheck,
        semantic_payload: PayloadIndex {
            field: "M31",
            row_count: row_count as u64,
            address_to_id: table_chunk,
            input_chunks,
            expected_chunks,
        },
    };
    artifact_io::write_immutable_json(index_path, &fixture)?;
    crate::streaming::export(index_path, oracle_output, true, Some(&simd_evidence))
}

fn store_table(index_path: &Path, table: &[u32]) -> Result<TableChunk, String> {
    let mut writer = ChunkWriter::create(index_path, 0)?;
    for &word in table {
        writer.write_word(word)?;
    }
    let stored = writer.finish(index_path)?;
    Ok(TableChunk {
        encoding: M31_ROW_ENCODING,
        path: stored.relative_path,
        sha256: stored.sha256,
        element_count: table.len() as u64,
        byte_len: stored.byte_len,
    })
}

pub fn estimated_exporter_peak_bytes(
    prover_input_bytes: u64,
    table_words: usize,
    rows: usize,
) -> Result<u64, String> {
    let table_words = u64::try_from(table_words).map_err(|_| "table words exceed u64")?;
    let rows = u64::try_from(rows).map_err(|_| "row count exceeds u64")?;
    prover_input_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(table_words.checked_mul(16)?))
        .and_then(|bytes| bytes.checked_add(rows.checked_mul(192)?))
        .and_then(|bytes| bytes.checked_add(128 * 1024 * 1024))
        .ok_or_else(|| "exporter peak estimate overflow".into())
}

pub(crate) fn enforce_exporter_budget(
    input_bytes: u64,
    table_words: usize,
    rows: usize,
) -> Result<(), String> {
    let estimate = estimated_exporter_peak_bytes(input_bytes, table_words, rows)?;
    if estimate > MAX_EXPORTER_ESTIMATED_PEAK_BYTES {
        return Err(format!(
            "estimated exporter peak {estimate} bytes exceeds bounded maximum {MAX_EXPORTER_ESTIMATED_PEAK_BYTES}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_class_gate_is_explicit() {
        assert!(!matches!(
            "tiny",
            "representative" | "stress" | "unseen-same-class"
        ));
    }

    #[test]
    fn chunk_geometry_is_pinned() {
        assert_eq!(ROWS_PER_CHUNK, 1 << 16);
        assert_eq!(OUTPUT_WORDS, 23);
    }

    #[test]
    fn aggregate_budget_admits_multiple_chunks_but_rejects_old_maximum() {
        let input = 155 * 1024 * 1024;
        let table = 9_000_000;
        assert!(
            estimated_exporter_peak_bytes(input, table, 1 << 17).unwrap()
                < MAX_EXPORTER_ESTIMATED_PEAK_BYTES
        );
        assert!(
            estimated_exporter_peak_bytes(input, table, 1 << 23).unwrap()
                > MAX_EXPORTER_ESTIMATED_PEAK_BYTES
        );
    }
}
