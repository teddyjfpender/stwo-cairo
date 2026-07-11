//! Pure descriptors for arena-native fixed-table witness materialization.
//!
//! The generated table in [`crate::fixed_table_table`] is the complete bridge
//! between transformer-emitted witness writers and the eventual generic CUDA
//! materializer. Each output is word-major: output word/column `w` and scalar
//! row `r` is stored at `w * (1 << log_size) + r`.

use std::collections::BTreeSet;

use crate::schedule::{
    ComponentRowSource, Schedule, TraceColumnCount, WitnessWriterKind, WitnessWriterReadiness,
};

const M31_MODULUS: u32 = (1 << 31) - 1;

/// One base-trace column copied from an arena-owned multiplicity column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedTableTraceColumn {
    pub output_column: u32,
    pub multiplicity_column: u32,
}

/// Scalar source for one word-major flattened `LookupInputs` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedTableWordSource {
    Constant(u32),
    PreprocessedColumn(&'static str),
    MultiplicityColumn(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedTableLookupWord {
    pub output_word: u32,
    pub source: FixedTableWordSource,
}

/// Closed-form layout for the expanded 12-bit XOR table.
///
/// Flattening is deterministic and word-major: all four tuple words for
/// multiplicity column `0`, then all four for column `1`, and so on, followed
/// by one multiplicity word for every column. This preserves the relation-id
/// first-word invariant without materializing preprocessed XOR columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandedXorLayout {
    pub relation_id: u32,
    pub limb_bits: u32,
    pub expand_bits: u32,
}

impl ExpandedXorLayout {
    pub const TUPLE_WORDS: u32 = 4;

    pub fn multiplicity_columns(self) -> Option<u32> {
        1u32.checked_shl(self.expand_bits.checked_mul(2)?)
    }

    pub fn row_count(self) -> Option<u32> {
        1u32.checked_shl(self.limb_bits.checked_mul(2)?)
    }

    pub fn lookup_word_count(self) -> Option<u32> {
        self.multiplicity_columns()?
            .checked_mul(Self::TUPLE_WORDS.checked_add(1)?)
    }

    pub fn tuple_word_offset(self, multiplicity_column: u32, tuple_word: u32) -> Option<u32> {
        (multiplicity_column < self.multiplicity_columns()? && tuple_word < Self::TUPLE_WORDS)
            .then(|| {
                multiplicity_column
                    .checked_mul(Self::TUPLE_WORDS)?
                    .checked_add(tuple_word)
            })?
    }

    pub fn multiplicity_word_offset(self, multiplicity_column: u32) -> Option<u32> {
        (multiplicity_column < self.multiplicity_columns()?).then(|| {
            self.multiplicity_columns()?
                .checked_mul(Self::TUPLE_WORDS)?
                .checked_add(multiplicity_column)
        })?
    }

    /// Evaluate `[relation_id, a, b, a ^ b]` for one scalar table row.
    pub fn tuple_at(self, multiplicity_column: u32, row: u32) -> Option<[u32; 4]> {
        if multiplicity_column >= self.multiplicity_columns()? || row >= self.row_count()? {
            return None;
        }
        let expand_mask = 1u32.checked_shl(self.expand_bits)?.checked_sub(1)?;
        let limb_mask = 1u32.checked_shl(self.limb_bits)?.checked_sub(1)?;
        let a_high = multiplicity_column >> self.expand_bits;
        let b_high = multiplicity_column & expand_mask;
        let a_low = row >> self.limb_bits;
        let b_low = row & limb_mask;
        let a = a_high.checked_shl(self.limb_bits)? | a_low;
        let b = b_high.checked_shl(self.limb_bits)? | b_low;
        Some([self.relation_id, a, b, a ^ b])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedTableLookupLayout {
    /// Exact fields parsed from the generated `LookupData` assignments.
    Words(&'static [FixedTableLookupWord]),
    /// The one generated writer which synthesizes tuples from row/column bits.
    ExpandedXor(ExpandedXorLayout),
}

impl FixedTableLookupLayout {
    pub fn word_count(self) -> Option<u32> {
        match self {
            Self::Words(words) => u32::try_from(words.len()).ok(),
            Self::ExpandedXor(layout) => layout.lookup_word_count(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedTableMaterializationPlan {
    pub component: &'static str,
    pub log_size: u32,
    pub multiplicity_columns: u32,
    pub trace_columns: &'static [FixedTableTraceColumn],
    pub lookup: FixedTableLookupLayout,
}

#[derive(Clone, Copy, Debug)]
pub struct FixedTableMaterializationTable {
    pub plans: &'static [FixedTableMaterializationPlan],
    pub expected_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedTablePlanError {
    DuplicateComponent(&'static str),
    EmptyComponent,
    InvalidGeometry(&'static str),
    TraceColumnGap {
        component: &'static str,
        expected: u32,
        actual: u32,
    },
    TraceMultiplicityOutOfRange {
        component: &'static str,
        column: u32,
    },
    DuplicateTraceMultiplicity {
        component: &'static str,
        column: u32,
    },
    LookupWordGap {
        component: &'static str,
        expected: u32,
        actual: u32,
    },
    LookupMultiplicityOutOfRange {
        component: &'static str,
        column: u32,
    },
    DuplicateLookupMultiplicity {
        component: &'static str,
        column: u32,
    },
    MissingLookupMultiplicity {
        component: &'static str,
        column: u32,
    },
    InvalidM31Constant {
        component: &'static str,
        value: u32,
    },
    EmptyPreprocessedColumn(&'static str),
    ScheduleMismatch(&'static str),
    HashMismatch {
        expected: u64,
        actual: u64,
    },
}

impl std::fmt::Display for FixedTablePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FixedTablePlanError {}

impl FixedTableMaterializationPlan {
    fn validate(&self) -> Result<(), FixedTablePlanError> {
        if self.component.is_empty() {
            return Err(FixedTablePlanError::EmptyComponent);
        }
        if self.log_size >= usize::BITS || self.multiplicity_columns == 0 {
            return Err(FixedTablePlanError::InvalidGeometry(self.component));
        }
        if self.trace_columns.len() != self.multiplicity_columns as usize {
            return Err(FixedTablePlanError::InvalidGeometry(self.component));
        }

        let mut trace_multiplicities = vec![false; self.multiplicity_columns as usize];
        for (expected, trace) in self.trace_columns.iter().enumerate() {
            let expected = expected as u32;
            if trace.output_column != expected {
                return Err(FixedTablePlanError::TraceColumnGap {
                    component: self.component,
                    expected,
                    actual: trace.output_column,
                });
            }
            let Some(seen) = trace_multiplicities.get_mut(trace.multiplicity_column as usize)
            else {
                return Err(FixedTablePlanError::TraceMultiplicityOutOfRange {
                    component: self.component,
                    column: trace.multiplicity_column,
                });
            };
            if std::mem::replace(seen, true) {
                return Err(FixedTablePlanError::DuplicateTraceMultiplicity {
                    component: self.component,
                    column: trace.multiplicity_column,
                });
            }
        }

        match self.lookup {
            FixedTableLookupLayout::Words(words) => {
                let mut lookup_multiplicities = vec![false; self.multiplicity_columns as usize];
                for (expected, word) in words.iter().enumerate() {
                    let expected = expected as u32;
                    if word.output_word != expected {
                        return Err(FixedTablePlanError::LookupWordGap {
                            component: self.component,
                            expected,
                            actual: word.output_word,
                        });
                    }
                    match word.source {
                        FixedTableWordSource::Constant(value) if value >= M31_MODULUS => {
                            return Err(FixedTablePlanError::InvalidM31Constant {
                                component: self.component,
                                value,
                            });
                        }
                        FixedTableWordSource::PreprocessedColumn("") => {
                            return Err(FixedTablePlanError::EmptyPreprocessedColumn(
                                self.component,
                            ));
                        }
                        FixedTableWordSource::MultiplicityColumn(column) => {
                            let Some(seen) = lookup_multiplicities.get_mut(column as usize) else {
                                return Err(FixedTablePlanError::LookupMultiplicityOutOfRange {
                                    component: self.component,
                                    column,
                                });
                            };
                            if std::mem::replace(seen, true) {
                                return Err(FixedTablePlanError::DuplicateLookupMultiplicity {
                                    component: self.component,
                                    column,
                                });
                            }
                        }
                        FixedTableWordSource::Constant(_)
                        | FixedTableWordSource::PreprocessedColumn(_) => {}
                    }
                }
                if let Some(column) = lookup_multiplicities.iter().position(|seen| !seen) {
                    return Err(FixedTablePlanError::MissingLookupMultiplicity {
                        component: self.component,
                        column: column as u32,
                    });
                }
            }
            FixedTableLookupLayout::ExpandedXor(layout) => {
                if layout.relation_id >= M31_MODULUS
                    || layout.multiplicity_columns() != Some(self.multiplicity_columns)
                    || layout.row_count() != 1u32.checked_shl(self.log_size)
                    || layout.lookup_word_count().is_none()
                {
                    return Err(FixedTablePlanError::InvalidGeometry(self.component));
                }
            }
        }
        Ok(())
    }
}

impl FixedTableMaterializationTable {
    pub fn validate(&self) -> Result<(), FixedTablePlanError> {
        let mut components = BTreeSet::new();
        for plan in self.plans {
            if !components.insert(plan.component) {
                return Err(FixedTablePlanError::DuplicateComponent(plan.component));
            }
            plan.validate()?;
        }
        let actual = self.semantic_hash();
        if actual != self.expected_hash {
            return Err(FixedTablePlanError::HashMismatch {
                expected: self.expected_hash,
                actual,
            });
        }
        Ok(())
    }

    /// Prove the generated descriptor set is exactly the schedule's fixed-table
    /// set and is admitted only through the prepared capture-safe CUDA writer.
    pub fn validate_against_schedule(
        &self,
        schedule: &Schedule,
    ) -> Result<(), FixedTablePlanError> {
        self.validate()?;
        let fixed_nodes = schedule
            .nodes
            .iter()
            .filter(|node| matches!(node.facts.row_source, ComponentRowSource::FixedLogSize(_)))
            .collect::<Vec<_>>();
        if fixed_nodes.len() != self.plans.len() {
            return Err(FixedTablePlanError::ScheduleMismatch("fixed-table set"));
        }
        for plan in self.plans {
            let node = fixed_nodes
                .iter()
                .find(|node| node.id == plan.component)
                .ok_or(FixedTablePlanError::ScheduleMismatch(plan.component))?;
            let ComponentRowSource::FixedLogSize(schedule_log_size) = node.facts.row_source else {
                unreachable!("fixed_nodes contains only fixed tables")
            };
            if schedule_log_size != plan.log_size
                || node.facts.trace_columns
                    != TraceColumnCount::Fixed(plan.trace_columns.len() as u32)
                || node.facts.witness_writer.kind != WitnessWriterKind::FixedTableCuda
                || node.facts.witness_writer.readiness != WitnessWriterReadiness::CaptureSafe
            {
                return Err(FixedTablePlanError::ScheduleMismatch(plan.component));
            }
            // The CUDA fixed-table writer always materializes the flattened
            // LookupInputs buffer, so the schedule fact must carry the exact
            // per-row word count for BOTH layouts — the arena planner sizes
            // (and the resident workspace requires) that buffer from it.
            match plan.lookup {
                FixedTableLookupLayout::Words(words)
                    if node.facts.lookup_words == u32::try_from(words.len()).ok() => {}
                FixedTableLookupLayout::ExpandedXor(layout)
                    if node.facts.lookup_words.is_some()
                        && node.facts.lookup_words == layout.lookup_word_count() => {}
                _ => return Err(FixedTablePlanError::ScheduleMismatch(plan.component)),
            }
        }
        Ok(())
    }

    pub fn semantic_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for plan in self.plans {
            fnv_str(&mut hash, plan.component);
            fnv_u32(&mut hash, plan.log_size);
            fnv_u32(&mut hash, plan.multiplicity_columns);
            for trace in plan.trace_columns {
                fnv_u32(&mut hash, trace.output_column);
                fnv_u32(&mut hash, trace.multiplicity_column);
            }
            match plan.lookup {
                FixedTableLookupLayout::Words(words) => {
                    fnv_u32(&mut hash, 0);
                    fnv_u32(&mut hash, words.len() as u32);
                    for word in words {
                        fnv_u32(&mut hash, word.output_word);
                        match word.source {
                            FixedTableWordSource::Constant(value) => {
                                fnv_u32(&mut hash, 0);
                                fnv_u32(&mut hash, value);
                            }
                            FixedTableWordSource::PreprocessedColumn(id) => {
                                fnv_u32(&mut hash, 1);
                                fnv_str(&mut hash, id);
                            }
                            FixedTableWordSource::MultiplicityColumn(column) => {
                                fnv_u32(&mut hash, 2);
                                fnv_u32(&mut hash, column);
                            }
                        }
                    }
                }
                FixedTableLookupLayout::ExpandedXor(layout) => {
                    fnv_u32(&mut hash, 1);
                    fnv_u32(&mut hash, layout.relation_id);
                    fnv_u32(&mut hash, layout.limb_bits);
                    fnv_u32(&mut hash, layout.expand_bits);
                }
            }
        }
        hash
    }
}

fn fnv_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn fnv_str(hash: &mut u64, value: &str) {
    fnv_u32(hash, value.len() as u32);
    for byte in value.bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_table_table::CAIRO_FIXED_TABLE_MATERIALIZATION;
    use crate::schedule_table::CAIRO_SCHEDULE;

    #[test]
    fn generated_fixed_table_descriptors_are_exactly_the_schedule_set() {
        CAIRO_FIXED_TABLE_MATERIALIZATION
            .validate_against_schedule(&CAIRO_SCHEDULE)
            .unwrap();
        assert_eq!(CAIRO_FIXED_TABLE_MATERIALIZATION.plans.len(), 22);
    }

    #[test]
    fn expanded_xor_layout_has_exact_word_major_geometry() {
        let plan = CAIRO_FIXED_TABLE_MATERIALIZATION
            .plans
            .iter()
            .find(|plan| plan.component == "verify_bitwise_xor_12")
            .unwrap();
        let FixedTableLookupLayout::ExpandedXor(layout) = plan.lookup else {
            panic!("xor12 must retain its row-bit layout")
        };
        assert_eq!(layout.multiplicity_columns(), Some(16));
        assert_eq!(layout.row_count(), Some(1 << 20));
        assert_eq!(layout.lookup_word_count(), Some(80));

        let column = (3 << 2) | 2;
        let row = (5 << 10) | 7;
        let tuple = layout.tuple_at(column, row).unwrap();
        assert_eq!(tuple[1], (3 << 10) | 5);
        assert_eq!(tuple[2], (2 << 10) | 7);
        assert_eq!(tuple[3], tuple[1] ^ tuple[2]);
        assert_eq!(layout.tuple_word_offset(column, 3), Some(column * 4 + 3));
        assert_eq!(layout.multiplicity_word_offset(column), Some(64 + column));
    }
}
