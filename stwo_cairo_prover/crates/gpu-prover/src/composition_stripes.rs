//! Candidate resource-bounded composition stripes without production selection.
//!
//! The baseline deliberately reuses one ordinary AOT composition part per
//! stripe. It preserves the canonical coefficient order, clears the contiguous
//! accumulator slab exactly once, then applies its stripes as ordered
//! read-modify-writes. A stripe is not admissible without measured final-cubin
//! resources for the target SM.
//!
//! This address-free model is deliberately insufficient for runtime admission:
//! production must additionally bind every stripe to the loaded AOT manifest,
//! installed function/current device, exact launch geometry, and complete CUDA
//! function-resource receipt.

use crate::composition_plan::CompositionPlan;
use crate::composition_wave::{CompositionWaveError, CompositionWavePart, CompositionWaveProgram};

pub const COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD: u32 = 128;

/// Resource facts measured from the final loadable cubin, never source or PTX
/// estimates. The kernel identity may be shared by several ordinary AOT parts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionFinalCubinResource {
    pub cache_key: u64,
    pub semantic_hash: u64,
    pub cubin_identity: [u8; 32],
    pub target_sm: u32,
    pub registers_per_thread: u32,
    pub stack_frame_bytes: u64,
    pub spill_store_bytes: u64,
    pub spill_load_bytes: u64,
}

/// One resource-qualified ordinary AOT part.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceBoundedCompositionStripe {
    pub part: CompositionWavePart,
    pub accumulator_index: usize,
    pub cubin: CompositionFinalCubinResource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceBoundedCompositionAccumulator {
    pub evaluation_log_size: u32,
    pub row_count: u64,
}

/// The only legal accumulator state transition in the baseline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceBoundedCompositionStep {
    ZeroAccumulators,
    ReadModifyWrite {
        stripe_index: usize,
        accumulator_index: usize,
    },
}

/// Pure, address-free input to future native/runtime integration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceBoundedCompositionSchedule {
    source_plan_key: u64,
    target_sm: u32,
    accumulators: Vec<ResourceBoundedCompositionAccumulator>,
    stripes: Vec<ResourceBoundedCompositionStripe>,
    steps: Vec<ResourceBoundedCompositionStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceBoundedCompositionError {
    Canonical(CompositionWaveError),
    InvalidTargetSm,
    FallbackCountLength {
        expected: usize,
        actual: usize,
    },
    FallbackSource {
        component: usize,
        fallback_count: usize,
    },
    MissingAccumulator {
        stripe: usize,
        evaluation_log_size: u32,
    },
    MissingFinalCubinResource {
        stripe: usize,
        cache_key: u64,
        semantic_hash: u64,
    },
    DuplicateFinalCubinResource {
        stripe: usize,
        cache_key: u64,
        semantic_hash: u64,
    },
    EmptyCubinIdentity {
        stripe: usize,
    },
    CubinTargetSm {
        stripe: usize,
        expected: u32,
        actual: u32,
    },
    RegistersPerThread {
        stripe: usize,
        limit: u32,
        actual: u32,
    },
    StackFrame {
        stripe: usize,
        actual_bytes: u64,
    },
    SpillStores {
        stripe: usize,
        actual_bytes: u64,
    },
    SpillLoads {
        stripe: usize,
        actual_bytes: u64,
    },
    ScheduleDrift,
}

impl core::fmt::Display for ResourceBoundedCompositionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "invalid resource-bounded composition schedule: {self:?}"
        )
    }
}

impl std::error::Error for ResourceBoundedCompositionError {}

impl From<CompositionWaveError> for ResourceBoundedCompositionError {
    fn from(value: CompositionWaveError) -> Self {
        Self::Canonical(value)
    }
}

impl ResourceBoundedCompositionSchedule {
    pub fn compile(
        plan: &CompositionPlan,
        fallback_counts: &[usize],
        target_sm: u32,
        cubins: &[CompositionFinalCubinResource],
    ) -> Result<Self, ResourceBoundedCompositionError> {
        if target_sm == 0 {
            return Err(ResourceBoundedCompositionError::InvalidTargetSm);
        }
        if fallback_counts.len() != plan.components.len() {
            return Err(ResourceBoundedCompositionError::FallbackCountLength {
                expected: plan.components.len(),
                actual: fallback_counts.len(),
            });
        }
        if let Some((component, &fallback_count)) = fallback_counts
            .iter()
            .enumerate()
            .find(|(_, fallback_count)| **fallback_count != 0)
        {
            return Err(ResourceBoundedCompositionError::FallbackSource {
                component,
                fallback_count,
            });
        }

        let canonical = CompositionWaveProgram::from_plan(plan)?;
        let accumulators = canonical
            .waves()
            .iter()
            .map(|wave| ResourceBoundedCompositionAccumulator {
                evaluation_log_size: wave.evaluation_log_size,
                row_count: wave.row_count,
            })
            .collect::<Vec<_>>();
        let mut stripes = Vec::with_capacity(canonical.parts().len());
        for part in canonical.parts() {
            let accumulator_index = canonical
                .waves()
                .binary_search_by_key(&part.evaluation_log_size, |wave| wave.evaluation_log_size)
                .map_err(|_| ResourceBoundedCompositionError::MissingAccumulator {
                    stripe: part.ordinal,
                    evaluation_log_size: part.evaluation_log_size,
                })?;
            let mut matching = cubins.iter().filter(|resource| {
                resource.cache_key == part.cache_key && resource.semantic_hash == part.semantic_hash
            });
            let cubin = *matching.next().ok_or(
                ResourceBoundedCompositionError::MissingFinalCubinResource {
                    stripe: part.ordinal,
                    cache_key: part.cache_key,
                    semantic_hash: part.semantic_hash,
                },
            )?;
            if matching.next().is_some() {
                return Err(
                    ResourceBoundedCompositionError::DuplicateFinalCubinResource {
                        stripe: part.ordinal,
                        cache_key: part.cache_key,
                        semantic_hash: part.semantic_hash,
                    },
                );
            }
            qualify_cubin(part.ordinal, target_sm, cubin)?;
            stripes.push(ResourceBoundedCompositionStripe {
                part: part.clone(),
                accumulator_index,
                cubin,
            });
        }

        let mut steps = Vec::with_capacity(1 + stripes.len());
        steps.push(ResourceBoundedCompositionStep::ZeroAccumulators);
        for stripe in &stripes {
            steps.push(ResourceBoundedCompositionStep::ReadModifyWrite {
                stripe_index: stripe.part.ordinal,
                accumulator_index: stripe.accumulator_index,
            });
        }

        Ok(Self {
            source_plan_key: canonical.source_plan_key(),
            target_sm,
            accumulators,
            stripes,
            steps,
        })
    }

    pub fn validate_against(
        &self,
        plan: &CompositionPlan,
        fallback_counts: &[usize],
        target_sm: u32,
        cubins: &[CompositionFinalCubinResource],
    ) -> Result<(), ResourceBoundedCompositionError> {
        let expected = Self::compile(plan, fallback_counts, target_sm, cubins)?;
        if &expected != self {
            return Err(ResourceBoundedCompositionError::ScheduleDrift);
        }
        Ok(())
    }

    pub const fn source_plan_key(&self) -> u64 {
        self.source_plan_key
    }

    pub const fn target_sm(&self) -> u32 {
        self.target_sm
    }

    pub fn accumulators(&self) -> &[ResourceBoundedCompositionAccumulator] {
        &self.accumulators
    }

    pub fn stripes(&self) -> &[ResourceBoundedCompositionStripe] {
        &self.stripes
    }

    pub fn steps(&self) -> &[ResourceBoundedCompositionStep] {
        &self.steps
    }
}

fn qualify_cubin(
    stripe: usize,
    target_sm: u32,
    cubin: CompositionFinalCubinResource,
) -> Result<(), ResourceBoundedCompositionError> {
    if cubin.cubin_identity == [0; 32] {
        return Err(ResourceBoundedCompositionError::EmptyCubinIdentity { stripe });
    }
    if cubin.target_sm != target_sm {
        return Err(ResourceBoundedCompositionError::CubinTargetSm {
            stripe,
            expected: target_sm,
            actual: cubin.target_sm,
        });
    }
    if cubin.registers_per_thread == 0
        || cubin.registers_per_thread > COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD
    {
        return Err(ResourceBoundedCompositionError::RegistersPerThread {
            stripe,
            limit: COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD,
            actual: cubin.registers_per_thread,
        });
    }
    if cubin.stack_frame_bytes != 0 {
        return Err(ResourceBoundedCompositionError::StackFrame {
            stripe,
            actual_bytes: cubin.stack_frame_bytes,
        });
    }
    if cubin.spill_store_bytes != 0 {
        return Err(ResourceBoundedCompositionError::SpillStores {
            stripe,
            actual_bytes: cubin.spill_store_bytes,
        });
    }
    if cubin.spill_load_bytes != 0 {
        return Err(ResourceBoundedCompositionError::SpillLoads {
            stripe,
            actual_bytes: cubin.spill_load_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;

    use super::*;
    use crate::composition_plan::{CompositionComponentPlan, CompositionKernelPart};

    const TARGET_SM: u32 = 90;

    fn component(
        component: &'static str,
        evaluation_log_size: u32,
        n_constraints: usize,
        random_coefficient_offset: usize,
        kernels: &[(u64, u32)],
    ) -> CompositionComponentPlan {
        CompositionComponentPlan {
            component,
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
            kernels: kernels
                .iter()
                .map(|&(id, rc_base)| CompositionKernelPart {
                    kernel_name: format!("stripe_test_{id}"),
                    cache_key: id,
                    semantic_hash: id + 100,
                    source: format!("source_{id}"),
                    rc_base,
                })
                .collect(),
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

    fn resources() -> Vec<CompositionFinalCubinResource> {
        (1..=6)
            .map(|id| CompositionFinalCubinResource {
                cache_key: id,
                semantic_hash: id + 100,
                cubin_identity: [id as u8; 32],
                target_sm: TARGET_SM,
                registers_per_thread: 128,
                stack_frame_bytes: 0,
                spill_store_bytes: 0,
                spill_load_bytes: 0,
            })
            .collect()
    }

    fn schedule() -> ResourceBoundedCompositionSchedule {
        ResourceBoundedCompositionSchedule::compile(&plan(), &[0, 0, 0], TARGET_SM, &resources())
            .unwrap()
    }

    #[test]
    fn baseline_is_one_ordinary_part_per_stripe_in_canonical_order() {
        let schedule = schedule();
        assert_eq!(schedule.target_sm(), TARGET_SM);
        assert_eq!(
            schedule
                .accumulators()
                .iter()
                .map(|value| (value.evaluation_log_size, value.row_count))
                .collect::<Vec<_>>(),
            vec![(6, 64), (8, 256)]
        );
        assert_eq!(
            schedule
                .stripes()
                .iter()
                .map(|stripe| {
                    (
                        stripe.part.ordinal,
                        stripe.part.component_index,
                        stripe.part.kernel_index,
                        stripe.accumulator_index,
                        stripe.part.coefficient_start,
                        stripe.part.coefficient_end,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 0, 1, 0, 2),
                (1, 0, 1, 1, 2, 5),
                (2, 1, 0, 0, 5, 8),
                (3, 2, 0, 1, 8, 11),
                (4, 2, 1, 1, 11, 13),
                (5, 2, 2, 1, 13, 15),
            ]
        );
        schedule
            .validate_against(&plan(), &[0, 0, 0], TARGET_SM, &resources())
            .unwrap();
    }

    #[test]
    fn accumulator_slab_is_zeroed_once_before_ordered_read_modify_writes() {
        use ResourceBoundedCompositionStep::{ReadModifyWrite, ZeroAccumulators};

        assert_eq!(
            schedule().steps(),
            &[
                ZeroAccumulators,
                ReadModifyWrite {
                    stripe_index: 0,
                    accumulator_index: 1,
                },
                ReadModifyWrite {
                    stripe_index: 1,
                    accumulator_index: 1,
                },
                ReadModifyWrite {
                    stripe_index: 2,
                    accumulator_index: 0,
                },
                ReadModifyWrite {
                    stripe_index: 3,
                    accumulator_index: 1,
                },
                ReadModifyWrite {
                    stripe_index: 4,
                    accumulator_index: 1,
                },
                ReadModifyWrite {
                    stripe_index: 5,
                    accumulator_index: 1,
                },
            ]
        );
    }

    #[test]
    fn any_fallback_or_missing_final_cubin_fails_closed() {
        let mut coefficient_drift = plan();
        coefficient_drift.components[1].random_coefficient_offset += 1;
        assert!(matches!(
            ResourceBoundedCompositionSchedule::compile(
                &coefficient_drift,
                &[0, 0, 0],
                TARGET_SM,
                &resources()
            ),
            Err(ResourceBoundedCompositionError::Canonical(
                CompositionWaveError::RandomCoefficientOrder { component: 1, .. }
            ))
        ));
        assert!(matches!(
            ResourceBoundedCompositionSchedule::compile(&plan(), &[0, 0], TARGET_SM, &resources()),
            Err(ResourceBoundedCompositionError::FallbackCountLength { .. })
        ));
        assert!(matches!(
            ResourceBoundedCompositionSchedule::compile(
                &plan(),
                &[0, 1, 0],
                TARGET_SM,
                &resources()
            ),
            Err(ResourceBoundedCompositionError::FallbackSource {
                component: 1,
                fallback_count: 1,
            })
        ));
        assert!(matches!(
            ResourceBoundedCompositionSchedule::compile(
                &plan(),
                &[0, 0, 0],
                TARGET_SM,
                &resources()[1..]
            ),
            Err(ResourceBoundedCompositionError::MissingFinalCubinResource { stripe: 0, .. })
        ));
    }

    fn rejected_resource(
        mutate: impl FnOnce(&mut CompositionFinalCubinResource),
    ) -> ResourceBoundedCompositionError {
        let mut resources = resources();
        mutate(&mut resources[0]);
        ResourceBoundedCompositionSchedule::compile(&plan(), &[0, 0, 0], TARGET_SM, &resources)
            .unwrap_err()
    }

    #[test]
    fn final_cubin_gate_rejects_identity_register_stack_and_spill_drift() {
        assert!(matches!(
            rejected_resource(|resource| resource.cubin_identity = [0; 32]),
            ResourceBoundedCompositionError::EmptyCubinIdentity { stripe: 0 }
        ));
        assert!(matches!(
            rejected_resource(|resource| resource.target_sm = 89),
            ResourceBoundedCompositionError::CubinTargetSm { stripe: 0, .. }
        ));
        for registers in [0, 129] {
            assert!(matches!(
                rejected_resource(|resource| resource.registers_per_thread = registers),
                ResourceBoundedCompositionError::RegistersPerThread {
                    stripe: 0,
                    actual,
                    ..
                } if actual == registers
            ));
        }
        assert!(matches!(
            rejected_resource(|resource| resource.stack_frame_bytes = 8),
            ResourceBoundedCompositionError::StackFrame {
                stripe: 0,
                actual_bytes: 8,
            }
        ));
        assert!(matches!(
            rejected_resource(|resource| resource.spill_store_bytes = 8),
            ResourceBoundedCompositionError::SpillStores {
                stripe: 0,
                actual_bytes: 8,
            }
        ));
        assert!(matches!(
            rejected_resource(|resource| resource.spill_load_bytes = 8),
            ResourceBoundedCompositionError::SpillLoads {
                stripe: 0,
                actual_bytes: 8,
            }
        ));
    }

    #[test]
    fn duplicate_resource_and_schedule_mutation_fail_closed() {
        let mut resources = resources();
        let duplicate = resources[0];
        resources.push(duplicate);
        assert!(matches!(
            ResourceBoundedCompositionSchedule::compile(&plan(), &[0, 0, 0], TARGET_SM, &resources),
            Err(ResourceBoundedCompositionError::DuplicateFinalCubinResource { stripe: 0, .. })
        ));

        let clean_resources = self::resources();
        let mut schedule = schedule();
        schedule.steps.swap(0, 1);
        assert_eq!(
            schedule.validate_against(&plan(), &[0, 0, 0], TARGET_SM, &clean_resources),
            Err(ResourceBoundedCompositionError::ScheduleDrift)
        );
    }
}
