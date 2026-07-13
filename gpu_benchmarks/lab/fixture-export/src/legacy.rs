use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::artifact_io;
use crate::model::{
    self, Boundary, CommonFixture, FixtureOracle, ProofIdentity, SemanticIdentity, INLINE_SCHEMA,
    M31_P, OPERATION,
};
use crate::pedersen_builtin_semantics::{self as semantics, InputRow};

pub const MAX_INLINE_BYTES: u64 = 1024 * 1024;
const MAX_INLINE_ROWS: usize = 4096;
const MAX_INLINE_TABLE_WORDS: usize = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fixture {
    pub schema_version: String,
    pub fixture_id: String,
    pub fixture_class: String,
    pub proof_identity: ProofIdentity,
    pub semantic_identity: SemanticIdentity,
    pub full_proof_semantic_hash: Option<String>,
    pub boundary: Boundary,
    pub oracle: FixtureOracle,
    pub semantic_payload: Payload,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub field: String,
    pub row_count: usize,
    pub inputs: Inputs,
    pub tables: Tables,
    pub expected: Expected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Inputs {
    pub segment_start: Vec<u32>,
    pub enabler: Vec<u32>,
    pub iota: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Tables {
    pub address_to_id: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Expected {
    pub output_columns: Vec<Vec<u32>>,
    pub lookup_words: Vec<Vec<u32>>,
    pub sub_words: Vec<Vec<u32>>,
}

#[derive(Debug, Serialize)]
struct Artifact {
    schema_version: &'static str,
    fixture_id: String,
    fixture_sha256: String,
    semantic_identity: SemanticIdentity,
    full_proof_semantic_hash: Option<String>,
    semantic_operation: &'static str,
    oracle: model::OracleIdentity,
    validation: Validation,
    expected: Expected,
}

#[derive(Debug, Serialize)]
struct Validation {
    scalar_golden_rows: usize,
    production_simd_crosschecked_rows: usize,
    words_compared_to_checked_fixture: usize,
    checked_fixture_match: bool,
}

pub fn export(path: &Path, output: Option<&Path>, check: bool) -> Result<(), String> {
    let (bytes, fixture_sha256) =
        model::load_bounded(path, MAX_INLINE_BYTES, "inline fixture JSON")?;
    let fixture: Fixture = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    validate_fixture(&fixture)?;

    let Fixture {
        fixture_id,
        semantic_identity,
        full_proof_semantic_hash,
        semantic_payload,
        ..
    } = fixture;
    let inputs = (0..semantic_payload.row_count)
        .map(|row| InputRow {
            segment_start: semantic_payload.inputs.segment_start[row],
            enabler: semantic_payload.inputs.enabler[row],
            iota: semantic_payload.inputs.iota[row],
        })
        .collect::<Vec<_>>();
    semantics::validate_table(&semantic_payload.tables.address_to_id)?;
    let rows = inputs
        .into_iter()
        .map(|input| semantics::evaluate(&semantic_payload.tables.address_to_id, input))
        .collect::<Result<Vec<_>, _>>()?;
    let transpose = |width: usize, select: fn(&semantics::SemanticRow) -> &[u32]| {
        (0..width)
            .map(|word| rows.iter().map(|row| select(row)[word]).collect())
            .collect::<Vec<Vec<u32>>>()
    };
    let expected = Expected {
        output_columns: transpose(3, |row| row.columns.as_slice()),
        lookup_words: transpose(14, |row| row.lookup_words.as_slice()),
        sub_words: transpose(6, |row| row.sub_words.as_slice()),
    };
    let fixture_match = expected == semantic_payload.expected;
    if check && !fixture_match {
        return Err(
            "checked fixture expected buffers differ from the canonical host export".into(),
        );
    }
    let artifact = Artifact {
        schema_version: "stwo.gpu-lab.host-oracle.v1",
        fixture_id,
        fixture_sha256,
        semantic_identity,
        full_proof_semantic_hash,
        semantic_operation: OPERATION,
        oracle: model::oracle_identity(),
        validation: Validation {
            scalar_golden_rows: rows.len(),
            production_simd_crosschecked_rows: 0,
            words_compared_to_checked_fixture: if check { rows.len() * 23 } else { 0 },
            checked_fixture_match: fixture_match,
        },
        expected,
    };
    artifact_io::write_json(output, &artifact)
}

fn validate_fixture(fixture: &Fixture) -> Result<(), String> {
    if fixture.schema_version != INLINE_SCHEMA {
        return Err(format!(
            "unsupported inline fixture schema: {}",
            fixture.schema_version
        ));
    }
    model::validate_common(
        CommonFixture {
            fixture_id: &fixture.fixture_id,
            fixture_class: &fixture.fixture_class,
            proof_identity: &fixture.proof_identity,
            semantic_identity: &fixture.semantic_identity,
            full_proof_semantic_hash: &fixture.full_proof_semantic_hash,
            boundary: &fixture.boundary,
            oracle: &fixture.oracle,
        },
        true,
    )?;
    let payload = &fixture.semantic_payload;
    if payload.field != "M31" {
        return Err(format!("wrong payload field: {}", payload.field));
    }
    if payload.row_count == 0 || payload.row_count > MAX_INLINE_ROWS {
        return Err(format!(
            "inline row_count {} is outside 1..={MAX_INLINE_ROWS}; use the chunk-index schema",
            payload.row_count
        ));
    }
    for (name, values) in [
        ("segment_start", &payload.inputs.segment_start),
        ("enabler", &payload.inputs.enabler),
        ("iota", &payload.inputs.iota),
    ] {
        validate_column(name, values, payload.row_count)?;
    }
    if let Some((row, value)) = payload
        .inputs
        .enabler
        .iter()
        .enumerate()
        .find(|(_, value)| **value != 1)
    {
        return Err(format!(
            "enabler[{row}]={value} is not canonical constant one"
        ));
    }
    if payload.tables.address_to_id.len() > MAX_INLINE_TABLE_WORDS {
        return Err(format!(
            "inline address_to_id has {} words; maximum is {MAX_INLINE_TABLE_WORDS}; use the chunk-index schema",
            payload.tables.address_to_id.len()
        ));
    }
    validate_column(
        "address_to_id",
        &payload.tables.address_to_id,
        payload.tables.address_to_id.len(),
    )?;
    if payload.tables.address_to_id.len() < 4 {
        return Err("address_to_id must contain address zero plus at least 1..3".into());
    }
    for (name, columns, width) in [
        ("output_columns", &payload.expected.output_columns, 3),
        ("lookup_words", &payload.expected.lookup_words, 14),
        ("sub_words", &payload.expected.sub_words, 6),
    ] {
        if columns.len() != width {
            return Err(format!(
                "{name} has {} columns, expected {width}",
                columns.len()
            ));
        }
        for (column, values) in columns.iter().enumerate() {
            validate_column(&format!("{name}[{column}]"), values, payload.row_count)?;
        }
    }
    Ok(())
}

fn validate_column(name: &str, values: &[u32], expected: usize) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!(
            "{name} has {} rows, expected {expected}",
            values.len()
        ));
    }
    if let Some((row, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| **value >= M31_P)
    {
        return Err(format!("{name}[{row}]={value} is not canonical M31"));
    }
    Ok(())
}
