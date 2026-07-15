//! Address-free schedule for one accumulator-owning composition wave per log domain.
//!
//! This module does not select or launch CUDA. It proves the only legal input to
//! a future wave kernel: every emitted constraint part remains in canonical
//! Cairo plan order, keeps its exact descending-random-coefficient range, and is
//! assigned once to the accumulator whose evaluation log it already targets.

use std::collections::BTreeMap;

use crate::composition_plan::CompositionPlan;

const SECURE_COORDINATES: u64 = 4;
const WORD_BYTES: u64 = 4;
const ACCUMULATOR_READ_WRITE_BYTES_PER_ROW: u64 = SECURE_COORDINATES * WORD_BYTES * 2;
const ACCUMULATOR_WRITE_BYTES_PER_ROW: u64 = SECURE_COORDINATES * WORD_BYTES;

/// One canonical AOT constraint part and its proof-global coefficient span.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWavePart {
    pub ordinal: usize,
    pub component_index: usize,
    pub kernel_index: usize,
    pub component: &'static str,
    pub instance: usize,
    pub evaluation_log_size: u32,
    pub row_count: u64,
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub coefficient_start: usize,
    pub coefficient_end: usize,
}

/// All constraint parts which contribute to one disjoint log-size accumulator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWave {
    pub evaluation_log_size: u32,
    pub row_count: u64,
    /// Indices into [`CompositionWaveProgram::parts`], in canonical plan order.
    pub part_ordinals: Vec<usize>,
}

/// Exact logical accumulator traffic implied by the address-free schedule.
///
/// These are source-level global-memory bytes, not measured HBM transactions or
/// a duration estimate. A native implementation receives no time credit from
/// this model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionWaveTraffic {
    pub legacy_launches: usize,
    pub wave_launches: usize,
    pub legacy_row_passes: u64,
    pub distinct_accumulator_rows: u64,
    pub legacy_clear_bytes: u64,
    pub legacy_read_modify_write_bytes: u64,
    pub legacy_total_bytes: u64,
    pub wave_final_write_bytes: u64,
    pub eliminated_bytes: u64,
}

/// Immutable, plan-keyed input to a future checked CUDA binder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWaveProgram {
    source_plan_key: u64,
    total_constraints: usize,
    parts: Vec<CompositionWavePart>,
    waves: Vec<CompositionWave>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionWaveError {
    EmptyPlan,
    InvalidPlanTotal {
        declared: usize,
        actual: usize,
    },
    InvalidPlanMaxLog {
        declared: u32,
        actual: u32,
    },
    ZeroConstraints {
        component: usize,
    },
    RandomCoefficientOrder {
        component: usize,
        expected: usize,
        actual: usize,
    },
    EmptyKernelProgram {
        component: usize,
    },
    FirstKernelOffset {
        component: usize,
        actual: u32,
    },
    KernelOffsetOrder {
        component: usize,
        kernel: usize,
        previous: u32,
        actual: u32,
    },
    KernelOffsetOutOfRange {
        component: usize,
        kernel: usize,
        offset: u32,
        constraints: usize,
    },
    InvalidKernelIdentity {
        component: usize,
        kernel: usize,
    },
    LogSizeOutOfRange {
        component: usize,
        log_size: u32,
    },
    ArithmeticOverflow,
    SourcePlanDrift {
        expected: u64,
        actual: u64,
    },
    ProgramDrift,
}

impl std::fmt::Display for CompositionWaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid composition wave program: {self:?}")
    }
}

impl std::error::Error for CompositionWaveError {}

impl CompositionWaveProgram {
    /// Derive the sole canonical wave schedule from an exact composition plan.
    pub fn from_plan(plan: &CompositionPlan) -> Result<Self, CompositionWaveError> {
        if plan.components.is_empty() || plan.total_constraints == 0 {
            return Err(CompositionWaveError::EmptyPlan);
        }
        let actual_total = plan.components.iter().try_fold(0usize, |total, component| {
            total.checked_add(component.n_constraints)
        });
        let actual_total = actual_total.ok_or(CompositionWaveError::ArithmeticOverflow)?;
        if actual_total != plan.total_constraints {
            return Err(CompositionWaveError::InvalidPlanTotal {
                declared: plan.total_constraints,
                actual: actual_total,
            });
        }
        let actual_max_log = plan
            .components
            .iter()
            .map(|component| component.evaluation_log_size)
            .max()
            .ok_or(CompositionWaveError::EmptyPlan)?;
        if actual_max_log != plan.max_evaluation_log_size {
            return Err(CompositionWaveError::InvalidPlanMaxLog {
                declared: plan.max_evaluation_log_size,
                actual: actual_max_log,
            });
        }

        let mut expected_coefficient = 0usize;
        let mut parts = Vec::new();
        for (component_index, component) in plan.components.iter().enumerate() {
            if component.n_constraints == 0 {
                return Err(CompositionWaveError::ZeroConstraints {
                    component: component_index,
                });
            }
            if component.random_coefficient_offset != expected_coefficient {
                return Err(CompositionWaveError::RandomCoefficientOrder {
                    component: component_index,
                    expected: expected_coefficient,
                    actual: component.random_coefficient_offset,
                });
            }
            let row_count = 1u64.checked_shl(component.evaluation_log_size).ok_or(
                CompositionWaveError::LogSizeOutOfRange {
                    component: component_index,
                    log_size: component.evaluation_log_size,
                },
            )?;
            if component.kernels.is_empty() {
                return Err(CompositionWaveError::EmptyKernelProgram {
                    component: component_index,
                });
            }
            if component.kernels[0].rc_base != 0 {
                return Err(CompositionWaveError::FirstKernelOffset {
                    component: component_index,
                    actual: component.kernels[0].rc_base,
                });
            }
            for (kernel_index, kernel) in component.kernels.iter().enumerate() {
                if kernel.cache_key == 0
                    || kernel.semantic_hash == 0
                    || kernel.kernel_name.is_empty()
                    || kernel.source.is_empty()
                {
                    return Err(CompositionWaveError::InvalidKernelIdentity {
                        component: component_index,
                        kernel: kernel_index,
                    });
                }
                let local_start = kernel.rc_base as usize;
                if local_start >= component.n_constraints {
                    return Err(CompositionWaveError::KernelOffsetOutOfRange {
                        component: component_index,
                        kernel: kernel_index,
                        offset: kernel.rc_base,
                        constraints: component.n_constraints,
                    });
                }
                let local_end = component
                    .kernels
                    .get(kernel_index + 1)
                    .map_or(component.n_constraints, |next| next.rc_base as usize);
                if local_end <= local_start {
                    return Err(CompositionWaveError::KernelOffsetOrder {
                        component: component_index,
                        kernel: kernel_index + 1,
                        previous: kernel.rc_base,
                        actual: component
                            .kernels
                            .get(kernel_index + 1)
                            .map_or(kernel.rc_base, |next| next.rc_base),
                    });
                }
                if local_end > component.n_constraints {
                    let next = &component.kernels[kernel_index + 1];
                    return Err(CompositionWaveError::KernelOffsetOutOfRange {
                        component: component_index,
                        kernel: kernel_index + 1,
                        offset: next.rc_base,
                        constraints: component.n_constraints,
                    });
                }
                let coefficient_start = component
                    .random_coefficient_offset
                    .checked_add(local_start)
                    .ok_or(CompositionWaveError::ArithmeticOverflow)?;
                let coefficient_end = component
                    .random_coefficient_offset
                    .checked_add(local_end)
                    .ok_or(CompositionWaveError::ArithmeticOverflow)?;
                parts.push(CompositionWavePart {
                    ordinal: parts.len(),
                    component_index,
                    kernel_index,
                    component: component.component,
                    instance: component.instance,
                    evaluation_log_size: component.evaluation_log_size,
                    row_count,
                    cache_key: kernel.cache_key,
                    semantic_hash: kernel.semantic_hash,
                    coefficient_start,
                    coefficient_end,
                });
            }
            expected_coefficient = expected_coefficient
                .checked_add(component.n_constraints)
                .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        }
        debug_assert_eq!(expected_coefficient, plan.total_constraints);

        // BTreeMap order is the existing accumulator/lift order. Within each
        // domain, insertion order is the canonical component/kernel order.
        let mut wave_parts = BTreeMap::<u32, Vec<usize>>::new();
        for part in &parts {
            wave_parts
                .entry(part.evaluation_log_size)
                .or_default()
                .push(part.ordinal);
        }
        let waves = wave_parts
            .into_iter()
            .map(|(evaluation_log_size, part_ordinals)| CompositionWave {
                evaluation_log_size,
                row_count: 1u64 << evaluation_log_size,
                part_ordinals,
            })
            .collect();
        Ok(Self {
            source_plan_key: plan.key(),
            total_constraints: plan.total_constraints,
            parts,
            waves,
        })
    }

    pub fn source_plan_key(&self) -> u64 {
        self.source_plan_key
    }

    pub fn total_constraints(&self) -> usize {
        self.total_constraints
    }

    pub fn parts(&self) -> &[CompositionWavePart] {
        &self.parts
    }

    pub fn waves(&self) -> &[CompositionWave] {
        &self.waves
    }

    /// Re-derive every field; a stale or mutated program is never bindable.
    pub fn validate_against(&self, plan: &CompositionPlan) -> Result<(), CompositionWaveError> {
        let actual_key = plan.key();
        if actual_key != self.source_plan_key {
            return Err(CompositionWaveError::SourcePlanDrift {
                expected: self.source_plan_key,
                actual: actual_key,
            });
        }
        let expected = Self::from_plan(plan)?;
        if &expected != self {
            return Err(CompositionWaveError::ProgramDrift);
        }
        Ok(())
    }

    pub fn traffic(&self) -> Result<CompositionWaveTraffic, CompositionWaveError> {
        let legacy_row_passes = self
            .parts
            .iter()
            .try_fold(0u64, |rows, part| rows.checked_add(part.row_count));
        let legacy_row_passes =
            legacy_row_passes.ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let distinct_accumulator_rows = self
            .waves
            .iter()
            .try_fold(0u64, |rows, wave| rows.checked_add(wave.row_count));
        let distinct_accumulator_rows =
            distinct_accumulator_rows.ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let legacy_clear_bytes = distinct_accumulator_rows
            .checked_mul(ACCUMULATOR_WRITE_BYTES_PER_ROW)
            .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let legacy_read_modify_write_bytes = legacy_row_passes
            .checked_mul(ACCUMULATOR_READ_WRITE_BYTES_PER_ROW)
            .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let legacy_total_bytes = legacy_clear_bytes
            .checked_add(legacy_read_modify_write_bytes)
            .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let wave_final_write_bytes = distinct_accumulator_rows
            .checked_mul(ACCUMULATOR_WRITE_BYTES_PER_ROW)
            .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        let eliminated_bytes = legacy_total_bytes
            .checked_sub(wave_final_write_bytes)
            .ok_or(CompositionWaveError::ArithmeticOverflow)?;
        Ok(CompositionWaveTraffic {
            legacy_launches: self.parts.len(),
            wave_launches: self.waves.len(),
            legacy_row_passes,
            distinct_accumulator_rows,
            legacy_clear_bytes,
            legacy_read_modify_write_bytes,
            legacy_total_bytes,
            wave_final_write_bytes,
            eliminated_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;

    use super::*;
    use crate::composition_plan::{CompositionComponentPlan, CompositionKernelPart};

    fn kernel(id: u64, rc_base: u32) -> CompositionKernelPart {
        CompositionKernelPart {
            kernel_name: format!("wave_test_{id}"),
            cache_key: id,
            semantic_hash: id + 100,
            source: format!("source_{id}"),
            rc_base,
        }
    }

    fn component(
        name: &'static str,
        evaluation_log_size: u32,
        n_constraints: usize,
        random_coefficient_offset: usize,
        kernels: &[(u64, u32)],
    ) -> CompositionComponentPlan {
        CompositionComponentPlan {
            component: name,
            instance: 0,
            trace_locations: Vec::new(),
            preprocessed_column_indices: Vec::new(),
            trace_log_size: evaluation_log_size - 1,
            evaluation_log_size,
            n_constraints,
            random_coefficient_offset,
            denominator_inverses: vec![BaseField::from(1)],
            base_param_values: Vec::new(),
            ext_param_values: Vec::new(),
            ext_param_sources: Vec::new(),
            kernels: kernels.iter().map(|&(id, rc)| kernel(id, rc)).collect(),
        }
    }

    fn plan() -> CompositionPlan {
        CompositionPlan {
            max_kernel_instrs: 192,
            total_constraints: 15,
            max_evaluation_log_size: 8,
            components: vec![
                component("a", 8, 5, 0, &[(1, 0), (2, 2)]),
                component("b", 6, 3, 5, &[(3, 0)]),
                component("c", 8, 7, 8, &[(4, 0), (5, 3), (6, 5)]),
            ],
            wave_kernels: Vec::new(),
        }
    }

    #[test]
    fn waves_are_one_per_log_and_keep_canonical_coefficient_ranges() {
        let plan = plan();
        let program = CompositionWaveProgram::from_plan(&plan).unwrap();
        assert_eq!(program.waves.len(), 2);
        assert_eq!(program.waves[0].evaluation_log_size, 6);
        assert_eq!(program.waves[0].part_ordinals, vec![2]);
        assert_eq!(program.waves[1].evaluation_log_size, 8);
        assert_eq!(program.waves[1].part_ordinals, vec![0, 1, 3, 4, 5]);
        assert_eq!(
            program
                .parts
                .iter()
                .map(|part| (part.coefficient_start, part.coefficient_end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (2, 5), (5, 8), (8, 11), (11, 13), (13, 15)]
        );
        program.validate_against(&plan).unwrap();
    }

    #[test]
    fn traffic_counts_only_logical_accumulator_passes() {
        let traffic = CompositionWaveProgram::from_plan(&plan())
            .unwrap()
            .traffic()
            .unwrap();
        assert_eq!(traffic.legacy_launches, 6);
        assert_eq!(traffic.wave_launches, 2);
        assert_eq!(traffic.legacy_row_passes, 1_344);
        assert_eq!(traffic.distinct_accumulator_rows, 320);
        assert_eq!(traffic.legacy_clear_bytes, 5_120);
        assert_eq!(traffic.legacy_read_modify_write_bytes, 43_008);
        assert_eq!(traffic.legacy_total_bytes, 48_128);
        assert_eq!(traffic.wave_final_write_bytes, 5_120);
        assert_eq!(traffic.eliminated_bytes, 43_008);
    }

    #[test]
    fn coefficient_and_kernel_order_drift_fail_closed() {
        let valid = plan();
        let program = CompositionWaveProgram::from_plan(&valid).unwrap();

        let mut offset_drift = valid.clone();
        offset_drift.components[1].random_coefficient_offset += 1;
        assert!(matches!(
            CompositionWaveProgram::from_plan(&offset_drift),
            Err(CompositionWaveError::RandomCoefficientOrder { .. })
        ));
        assert!(matches!(
            program.validate_against(&offset_drift),
            Err(CompositionWaveError::SourcePlanDrift { .. })
        ));

        let mut missing_zero = valid.clone();
        missing_zero.components[0].kernels[0].rc_base = 1;
        assert!(matches!(
            CompositionWaveProgram::from_plan(&missing_zero),
            Err(CompositionWaveError::FirstKernelOffset { .. })
        ));

        let mut reversed = valid;
        reversed.components[2].kernels[2].rc_base = 2;
        assert!(matches!(
            CompositionWaveProgram::from_plan(&reversed),
            Err(CompositionWaveError::KernelOffsetOrder { .. })
        ));
    }

    #[test]
    fn mutated_address_free_program_fails_rebinding() {
        let plan = plan();
        let mut program = CompositionWaveProgram::from_plan(&plan).unwrap();
        program.waves[1].part_ordinals.swap(0, 1);
        assert_eq!(
            program.validate_against(&plan),
            Err(CompositionWaveError::ProgramDrift)
        );
    }
}
