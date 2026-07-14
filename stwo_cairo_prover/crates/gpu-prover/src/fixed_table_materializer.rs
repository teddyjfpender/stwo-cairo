//! Compile the generated Cairo fixed-table plans into the STWO CUDA materializer ABI.
//!
//! Preprocessed identities stay symbolic here. The resident arena supplies their
//! concrete bindings later, in this module's canonical first-use order.

use std::collections::BTreeMap;

use stwo_backend_cuda::{
    fixed_table_workspace_requirements, FixedTableLookupSource, FixedTableMaterializationConfig,
    FixedTableWorkspaceRequirements, PreparedFixedTableError,
};

use crate::fixed_table::{
    FixedTableLookupLayout, FixedTableMaterializationPlan, FixedTableMaterializationTable,
    FixedTablePlanError, FixedTableWordSource,
};
use crate::fixed_table_table::CAIRO_FIXED_TABLE_MATERIALIZATION;

pub const PEDERSEN_POINTS_18_COLUMN_COUNT: usize = 56;
pub const PEDERSEN_POINTS_18_LOG_SIZE: u32 = 23;
pub const PEDERSEN_POINTS_18_ROW_COUNT: usize = 1 << PEDERSEN_POINTS_18_LOG_SIZE;
pub const PEDERSEN_POINTS_18_EVALUATION_BYTES: usize = PEDERSEN_POINTS_18_COLUMN_COUNT
    * PEDERSEN_POINTS_18_ROW_COUNT
    * core::mem::size_of::<u32>();

/// Numeric column encoded by the exact canonical `pedersen_points_N` identity.
/// Similar prefixes, leading-zero aliases and out-of-range columns fail closed.
pub fn pedersen_points_18_column_index(identity: &str) -> Option<usize> {
    let suffix = identity.strip_prefix("pedersen_points_")?;
    if suffix.is_empty()
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
        || (suffix.len() > 1 && suffix.starts_with('0'))
    {
        return None;
    }
    let index = suffix.parse::<usize>().ok()?;
    (index < PEDERSEN_POINTS_18_COLUMN_COUNT).then_some(index)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledFixedTableMaterialization {
    component: &'static str,
    config: FixedTableMaterializationConfig,
    requirements: FixedTableWorkspaceRequirements,
    preprocessed_sources: Vec<&'static str>,
}

impl CompiledFixedTableMaterialization {
    pub fn component(&self) -> &'static str {
        self.component
    }

    pub fn config(&self) -> &FixedTableMaterializationConfig {
        &self.config
    }

    pub fn requirements(&self) -> &FixedTableWorkspaceRequirements {
        &self.requirements
    }

    /// Canonical first-use order used by `SourceColumn(index)` descriptors.
    pub fn preprocessed_sources(&self) -> &[&'static str] {
        &self.preprocessed_sources
    }

    /// Resolve symbolic preprocessed identities without coupling this compiler
    /// to an arena type. A resident caller normally chooses `T = ArenaSlice`.
    pub fn resolve_preprocessed<T>(
        &self,
        mut binding: impl FnMut(&'static str) -> Option<T>,
    ) -> Result<Vec<T>, FixedTableMaterializerError> {
        self.preprocessed_sources
            .iter()
            .map(|&identity| {
                binding(identity).ok_or(FixedTableMaterializerError::MissingPreprocessedBinding {
                    component: self.component,
                    identity,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixedTableMaterializerError {
    Plan(FixedTablePlanError),
    Prepared(PreparedFixedTableError),
    RowCountOverflow(&'static str),
    SourceCountOverflow(&'static str),
    MissingPreprocessedBinding {
        component: &'static str,
        identity: &'static str,
    },
}

impl core::fmt::Display for FixedTableMaterializerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fixed-table materializer rejected: {self:?}")
    }
}

impl std::error::Error for FixedTableMaterializerError {}

impl From<FixedTablePlanError> for FixedTableMaterializerError {
    fn from(value: FixedTablePlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<PreparedFixedTableError> for FixedTableMaterializerError {
    fn from(value: PreparedFixedTableError) -> Self {
        Self::Prepared(value)
    }
}

pub fn compile_cairo_fixed_table_materializations(
) -> Result<Vec<CompiledFixedTableMaterialization>, FixedTableMaterializerError> {
    compile_fixed_table_materializations(&CAIRO_FIXED_TABLE_MATERIALIZATION)
}

pub fn compile_fixed_table_materializations(
    table: &'static FixedTableMaterializationTable,
) -> Result<Vec<CompiledFixedTableMaterialization>, FixedTableMaterializerError> {
    table.validate()?;
    table.plans.iter().map(compile_plan).collect()
}

fn compile_plan(
    plan: &FixedTableMaterializationPlan,
) -> Result<CompiledFixedTableMaterialization, FixedTableMaterializerError> {
    let row_count =
        1usize
            .checked_shl(plan.log_size)
            .ok_or(FixedTableMaterializerError::RowCountOverflow(
                plan.component,
            ))?;
    let mut source_indices = BTreeMap::new();
    let mut preprocessed_sources = Vec::new();
    let mut lookup_sources = Vec::new();
    match plan.lookup {
        FixedTableLookupLayout::Words(words) => {
            lookup_sources.reserve(words.len());
            for word in words {
                lookup_sources.push(match word.source {
                    FixedTableWordSource::Constant(value) => {
                        FixedTableLookupSource::Constant(value)
                    }
                    FixedTableWordSource::MultiplicityColumn(column) => {
                        FixedTableLookupSource::MultiplicityColumn(column)
                    }
                    FixedTableWordSource::PreprocessedColumn(identity) => {
                        let index = if let Some(&index) = source_indices.get(identity) {
                            index
                        } else {
                            let index =
                                u32::try_from(preprocessed_sources.len()).map_err(|_| {
                                    FixedTableMaterializerError::SourceCountOverflow(plan.component)
                                })?;
                            preprocessed_sources.push(identity);
                            source_indices.insert(identity, index);
                            index
                        };
                        FixedTableLookupSource::SourceColumn(index)
                    }
                });
            }
        }
        FixedTableLookupLayout::ExpandedXor(layout) => {
            let columns = layout.multiplicity_columns().ok_or(
                FixedTableMaterializerError::RowCountOverflow(plan.component),
            )?;
            let words =
                layout
                    .lookup_word_count()
                    .ok_or(FixedTableMaterializerError::RowCountOverflow(
                        plan.component,
                    ))?;
            lookup_sources.reserve(words as usize);
            for multiplicity_column in 0..columns {
                lookup_sources.extend([
                    FixedTableLookupSource::Constant(layout.relation_id),
                    FixedTableLookupSource::ExpandedXorA {
                        multiplicity_column,
                        limb_bits: layout.limb_bits,
                        expand_bits: layout.expand_bits,
                    },
                    FixedTableLookupSource::ExpandedXorB {
                        multiplicity_column,
                        limb_bits: layout.limb_bits,
                        expand_bits: layout.expand_bits,
                    },
                    FixedTableLookupSource::ExpandedXor {
                        multiplicity_column,
                        limb_bits: layout.limb_bits,
                        expand_bits: layout.expand_bits,
                    },
                ]);
            }
            lookup_sources.extend((0..columns).map(FixedTableLookupSource::MultiplicityColumn));
        }
    }

    let config = FixedTableMaterializationConfig {
        row_count,
        source_column_count: preprocessed_sources.len(),
        trace_multiplicity_columns: plan
            .trace_columns
            .iter()
            .map(|column| column.multiplicity_column)
            .collect(),
        multiplicity_column_count: plan.multiplicity_columns as usize,
        lookup_sources,
    };
    let requirements = fixed_table_workspace_requirements(&config)?;
    Ok(CompiledFixedTableMaterialization {
        component: plan.component,
        config,
        requirements,
        preprocessed_sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_generated_plans_compile_to_valid_prepared_descriptors() {
        let compiled = compile_cairo_fixed_table_materializations().unwrap();
        assert_eq!(compiled.len(), 22);
        assert_eq!(
            compiled
                .iter()
                .map(|component| component.requirements.trace_output_count)
                .sum::<usize>(),
            53
        );
        assert_eq!(
            compiled
                .iter()
                .map(|component| component.requirements.lookup_output_count)
                .sum::<usize>(),
            381
        );
        assert_eq!(
            compiled
                .iter()
                .map(|component| component.preprocessed_sources.len())
                .sum::<usize>(),
            202
        );

        for (plan, component) in CAIRO_FIXED_TABLE_MATERIALIZATION
            .plans
            .iter()
            .zip(&compiled)
        {
            assert_eq!(component.component(), plan.component);
            assert_eq!(component.config.row_count, 1usize << plan.log_size);
            assert_eq!(
                component.config.multiplicity_column_count,
                plan.multiplicity_columns as usize
            );
            assert_eq!(
                component.config.trace_multiplicity_columns,
                plan.trace_columns
                    .iter()
                    .map(|column| column.multiplicity_column)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                component.requirements.lookup_descriptor_words,
                component.config.lookup_sources.len() * 4
            );
            let resolved = component
                .resolve_preprocessed(|identity| {
                    component
                        .preprocessed_sources
                        .iter()
                        .position(|candidate| *candidate == identity)
                })
                .unwrap();
            assert_eq!(resolved, (0..resolved.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn expanded_xor_is_lowered_in_exact_word_major_order() {
        let compiled = compile_cairo_fixed_table_materializations().unwrap();
        let xor = compiled
            .iter()
            .find(|component| component.component == "verify_bitwise_xor_12")
            .unwrap();
        assert!(xor.preprocessed_sources.is_empty());
        assert_eq!(xor.config.lookup_sources.len(), 80);
        assert_eq!(
            &xor.config.lookup_sources[..5],
            &[
                FixedTableLookupSource::Constant(648362599),
                FixedTableLookupSource::ExpandedXorA {
                    multiplicity_column: 0,
                    limb_bits: 10,
                    expand_bits: 2,
                },
                FixedTableLookupSource::ExpandedXorB {
                    multiplicity_column: 0,
                    limb_bits: 10,
                    expand_bits: 2,
                },
                FixedTableLookupSource::ExpandedXor {
                    multiplicity_column: 0,
                    limb_bits: 10,
                    expand_bits: 2,
                },
                FixedTableLookupSource::Constant(648362599),
            ]
        );
        assert_eq!(
            &xor.config.lookup_sources[64..],
            &(0..16)
                .map(FixedTableLookupSource::MultiplicityColumn)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_preprocessed_binding_fails_closed_with_identity() {
        let compiled = compile_cairo_fixed_table_materializations().unwrap();
        let component = compiled
            .iter()
            .find(|component| !component.preprocessed_sources.is_empty())
            .unwrap();
        assert_eq!(
            component.resolve_preprocessed::<()>(|_| None).unwrap_err(),
            FixedTableMaterializerError::MissingPreprocessedBinding {
                component: component.component,
                identity: component.preprocessed_sources[0],
            }
        );
    }

    #[test]
    fn process_owned_pedersen_identity_and_bytes_are_exact() {
        assert_eq!(PEDERSEN_POINTS_18_EVALUATION_BYTES, 1_879_048_192);
        assert_eq!(PEDERSEN_POINTS_18_EVALUATION_BYTES >> 20, 1_792);
        for index in 0..PEDERSEN_POINTS_18_COLUMN_COUNT {
            assert_eq!(
                pedersen_points_18_column_index(&format!("pedersen_points_{index}")),
                Some(index)
            );
        }
        for forged in [
            "pedersen_points_",
            "pedersen_points_00",
            "pedersen_points_56",
            "pedersen_points_small_0",
            "pedersen_points_+1",
            "seq_23",
        ] {
            assert_eq!(pedersen_points_18_column_index(forged), None, "{forged}");
        }

        let pedersen = compile_cairo_fixed_table_materializations()
            .unwrap()
            .into_iter()
            .find(|plan| plan.component() == "pedersen_points_table_window_bits_18")
            .unwrap();
        let external = pedersen
            .preprocessed_sources()
            .iter()
            .filter_map(|identity| pedersen_points_18_column_index(identity))
            .collect::<Vec<_>>();
        assert_eq!(external, (0..PEDERSEN_POINTS_18_COLUMN_COUNT).collect::<Vec<_>>());
        assert_eq!(pedersen.config().row_count, PEDERSEN_POINTS_18_ROW_COUNT);
    }
}
