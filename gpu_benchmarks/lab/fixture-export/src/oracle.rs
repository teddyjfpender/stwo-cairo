//! Production-SIMD cross-check for the candidate-free scalar golden.

use std::sync::Arc;

use stwo_cairo_adapter::memory::{EncodedMemoryValueId, Memory, MemoryConfig};
use stwo_cairo_prover::witness::components::{memory_address_to_id, pedersen_builtin};

use crate::pedersen_builtin_semantics::{self as semantics, SemanticRow};

const N_LANES: usize = 16;
pub const MAX_SIMD_ROWS_PER_SEGMENT: usize = 1 << 20;

pub struct SimdCrosscheck {
    memory_state: memory_address_to_id::ClaimGenerator,
    table_words: usize,
}

/// Non-serializable evidence returned only after the production writer visits
/// the entire segment successfully. JSON fixtures cannot manufacture it.
pub struct SimdCrosscheckEvidence {
    rows: u64,
}

impl SimdCrosscheckEvidence {
    pub fn rows(&self) -> u64 {
        self.rows
    }
}

impl SimdCrosscheck {
    pub fn new(table: &[u32]) -> Result<Self, String> {
        semantics::validate_table(table)?;
        let table_words = table.len();
        let memory = Arc::new(Memory {
            config: MemoryConfig::default(),
            address_to_id: table.iter().copied().map(EncodedMemoryValueId).collect(),
            f252_values: Vec::new(),
            small_values: Vec::new(),
        });
        Ok(Self {
            memory_state: memory_address_to_id::ClaimGenerator::new(memory),
            table_words,
        })
    }

    /// Execute the untouched production writer exactly once and expose every logical row.
    pub fn visit_segment<F>(
        &self,
        segment_start: u32,
        row_count: usize,
        mut visit: F,
    ) -> Result<SimdCrosscheckEvidence, String>
    where
        F: FnMut(usize, SemanticRow) -> Result<(), String>,
    {
        if row_count < N_LANES || !row_count.is_power_of_two() {
            return Err(format!(
                "production segment rows {row_count} must be a power of two >= {N_LANES}"
            ));
        }
        if row_count > MAX_SIMD_ROWS_PER_SEGMENT {
            return Err(format!(
                "production segment rows {row_count} exceed bounded SIMD maximum {MAX_SIMD_ROWS_PER_SEGMENT}"
            ));
        }
        for row in 0..row_count {
            let iota = u32::try_from(row).map_err(|_| "row index exceeds u32")?;
            for address in semantics::addresses(segment_start, iota) {
                if address == 0 || address as usize >= self.table_words {
                    return Err(format!(
                        "segment {segment_start}: row {row} reaches invalid address {address}"
                    ));
                }
            }
        }
        let mut visited = 0usize;
        pedersen_builtin::gpu_lab_visit_simd_rows(
            row_count.ilog2(),
            segment_start,
            &self.memory_state,
            |row, actual| {
                visit(
                    row,
                    SemanticRow {
                        columns: actual.columns,
                        lookup_words: actual.lookup_words,
                        sub_words: actual.sub_words,
                    },
                )?;
                visited = visited
                    .checked_add(1)
                    .ok_or("production SIMD visited-row count overflow")?;
                Ok(())
            },
        )?;
        if visited != row_count {
            return Err(format!(
                "production SIMD visited {visited} rows, expected {row_count}"
            ));
        }
        Ok(SimdCrosscheckEvidence {
            rows: row_count as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crosscheck_domain_is_aggregate_bounded() {
        assert!(MAX_SIMD_ROWS_PER_SEGMENT >= 1 << 16);
        assert!(MAX_SIMD_ROWS_PER_SEGMENT < 1 << 24);
    }
}
