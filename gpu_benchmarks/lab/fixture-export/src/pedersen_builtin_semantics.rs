//! Candidate-free logical semantics for the first replacement slab.

use serde::{Deserialize, Serialize};

use crate::model::SemanticIdentity;
use crate::semantic_support::{canonical_value_hash, M31_P};

pub const DESCRIPTOR: &[u8] =
    include_bytes!("../semantics/pedersen_builtin.slab-semantics.v1.json");
pub const SEMANTIC_SCHEMA: &str = "stwo.gpu-lab.slab-semantics.v1";
pub const SEMANTIC_SCOPE: &str = "cairo.witness.pedersen_builtin.base-trace-and-facts";
pub const ENTRY_BOUNDARY: &str = "cairo.witness.pedersen_builtin.logical-inputs";
pub const EXIT_BOUNDARY: &str = "cairo.witness.pedersen_builtin.base-trace-and-facts";
pub const MEMORY_ADDRESS_TO_ID_RELATION: u32 = 1_444_891_767;
pub const PEDERSEN_AGGREGATOR_RELATION: u32 = 520_578_465;
pub const OUTPUT_WORDS: usize = 23;

#[derive(Clone, Copy, Debug)]
pub struct InputRow {
    pub segment_start: u32,
    pub enabler: u32,
    pub iota: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticRow {
    pub columns: [u32; 3],
    pub lookup_words: [u32; 14],
    pub sub_words: [u32; 6],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DescriptorHeader {
    schema: String,
    scope: String,
}

pub fn semantic_identity() -> Result<SemanticIdentity, String> {
    let value: serde_json::Value = serde_json::from_slice(DESCRIPTOR)
        .map_err(|error| format!("parse slab semantic descriptor: {error}"))?;
    let header: DescriptorHeader = serde_json::from_value(value.clone())
        .map_err(|error| format!("parse slab semantic descriptor header: {error}"))?;
    if header.schema != SEMANTIC_SCHEMA || header.scope != SEMANTIC_SCOPE {
        return Err("slab semantic descriptor header drifted".into());
    }
    Ok(SemanticIdentity {
        kind: "kernel_slice".into(),
        schema: header.schema,
        scope: header.scope,
        sha256: canonical_value_hash(&value)?,
    })
}

pub fn validate_table(table: &[u32]) -> Result<(), String> {
    if table.len() < 4 {
        return Err("address_to_id must contain address zero plus at least 1..3".into());
    }
    if let Some((index, value)) = table.iter().enumerate().find(|(_, value)| **value >= M31_P) {
        return Err(format!(
            "address_to_id[{index}]={value} is not canonical M31"
        ));
    }
    Ok(())
}

pub fn evaluate(table: &[u32], input: InputRow) -> Result<SemanticRow, String> {
    validate_input(input)?;
    let addresses = addresses(input.segment_start, input.iota);
    let mut columns = [0; 3];
    for (index, address) in addresses.into_iter().enumerate() {
        if address == 0 || address as usize >= table.len() {
            return Err(format!(
                "row {} reaches invalid address {address}; zero/OOB is outside slab semantics",
                input.iota
            ));
        }
        columns[index] = table[address as usize];
    }
    Ok(semantic_row(columns, addresses))
}

pub fn addresses(segment_start: u32, iota: u32) -> [u32; 3] {
    let base = ((segment_start as u64 + 3 * iota as u64) % M31_P as u64) as u32;
    [base, add_m31(base, 1), add_m31(base, 2)]
}

pub fn flatten(row: &SemanticRow) -> [u32; OUTPUT_WORDS] {
    let mut words = [0; OUTPUT_WORDS];
    words[..3].copy_from_slice(&row.columns);
    words[3..17].copy_from_slice(&row.lookup_words);
    words[17..].copy_from_slice(&row.sub_words);
    words
}

fn validate_input(input: InputRow) -> Result<(), String> {
    if input.segment_start >= M31_P {
        return Err(format!(
            "segment_start={} is not canonical M31",
            input.segment_start
        ));
    }
    if input.enabler != 1 {
        return Err(format!(
            "enabler={} is not the canonical constant one",
            input.enabler
        ));
    }
    if input.iota >= M31_P {
        return Err(format!("iota={} is not canonical M31", input.iota));
    }
    Ok(())
}

fn add_m31(value: u32, addend: u32) -> u32 {
    let sum = value + addend;
    if sum >= M31_P {
        sum - M31_P
    } else {
        sum
    }
}

fn semantic_row(columns: [u32; 3], addresses: [u32; 3]) -> SemanticRow {
    SemanticRow {
        columns,
        lookup_words: [
            MEMORY_ADDRESS_TO_ID_RELATION,
            addresses[0],
            columns[0],
            MEMORY_ADDRESS_TO_ID_RELATION,
            addresses[1],
            columns[1],
            MEMORY_ADDRESS_TO_ID_RELATION,
            addresses[2],
            columns[2],
            PEDERSEN_AGGREGATOR_RELATION,
            columns[0],
            columns[1],
            columns[2],
            1,
        ],
        sub_words: [
            addresses[0],
            addresses[1],
            addresses[2],
            columns[0],
            columns[1],
            columns[2],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_and_boundary_are_pinned() {
        let identity = semantic_identity().unwrap();
        assert_eq!(identity.kind, "kernel_slice");
        assert_eq!(identity.schema, SEMANTIC_SCHEMA);
        assert_eq!(identity.scope, SEMANTIC_SCOPE);
        assert_eq!(identity.sha256.len(), 64);
    }

    #[test]
    fn row_formula_wraps_and_orders_every_word() {
        let mut table = (0..64).map(|value| value * 17).collect::<Vec<_>>();
        table[0] = 0;
        let row = evaluate(
            &table,
            InputRow {
                segment_start: 61,
                enabler: 1,
                iota: 0,
            },
        )
        .unwrap();
        assert_eq!(row.columns, [1037, 1054, 1071]);
        assert_eq!(flatten(&row).len(), OUTPUT_WORDS);
        assert_eq!(&row.lookup_words[9..], &[520_578_465, 1037, 1054, 1071, 1]);
    }

    #[test]
    fn disabled_rows_are_not_proof_semantics() {
        let table = vec![0; 8];
        assert!(evaluate(
            &table,
            InputRow {
                segment_start: 1,
                enabler: 0,
                iota: 0
            }
        )
        .unwrap_err()
        .contains("constant one"));
    }
}
