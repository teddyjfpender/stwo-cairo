//! Exact resident bindings for the prepared Cairo composition graph.

use stwo_backend_cuda::{ArenaSlotId, PreparedRelationGraph};

use crate::arena_plan::ArenaBinding;
use crate::composition_plan::{
    CompositionExtParamSource, CompositionPlan, CompositionProofBindings,
};
use crate::direct_composition_retention::DirectCompositionRetentionPlan;
use crate::graphs::{bind_arena_binding, GraphWorkspace};
use crate::prepared_composition::{
    CompositionDeviceInputs, CompositionDirectEvaluationBinding, PreparedCompositionError,
    PreparedCompositionGraph,
};
use crate::relation::RelationTracePart;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentCompositionError {
    CurrentPlanTopologyDrift {
        cached_key: u64,
        current_key: u64,
    },
    MissingRelationBatch {
        component: &'static str,
        instance: usize,
        relation_component: &'static str,
        trace_part: RelationTracePart,
    },
    MissingRelationOutput {
        component: &'static str,
        instance: usize,
        batch: usize,
        relation_instance: usize,
    },
    DuplicateRelationOutput {
        component: &'static str,
        instance: usize,
        batch: usize,
        relation_instance: usize,
    },
    DirectRetentionIdentityDrift(&'static str),
    DuplicateDirectRetentionConsumer(usize),
    MissingDirectEvaluation(usize),
    ConflictingDirectEvaluation(usize),
    Prepared(PreparedCompositionError),
}

impl core::fmt::Display for ResidentCompositionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident composition binding failed: {self:?}")
    }
}

impl std::error::Error for ResidentCompositionError {}

impl From<PreparedCompositionError> for ResidentCompositionError {
    fn from(value: PreparedCompositionError) -> Self {
        Self::Prepared(value)
    }
}

/// Bind the exact arena-planned composition graph to relation challenge and
/// claimed-sum outputs. All mappings are structural; no proof value is copied
/// back to the host.
pub(crate) fn prepare_resident_composition<'a>(
    workspace: &'a GraphWorkspace,
    relation: &PreparedRelationGraph<'a>,
    current_plan: &CompositionPlan,
    proof_bindings: &CompositionProofBindings,
) -> Result<PreparedCompositionGraph<'a>, ResidentCompositionError> {
    let cached = workspace.plan().composition();
    let current_plan = require_current_composition_plan(&cached.plan, current_plan)?;
    let claimed_sums = current_plan
        .components
        .iter()
        .map(|component| {
            if !component
                .ext_param_sources
                .iter()
                .any(|source| matches!(source, CompositionExtParamSource::ClaimedSumScaled))
            {
                return Ok(None);
            }
            let (relation_component, trace_part, relation_instance) =
                relation_claimed_sum_key(component.component, component.instance);
            let batch = workspace
                .plan()
                .relation()
                .execution
                .batches
                .iter()
                .position(|candidate| {
                    candidate.component == relation_component && candidate.trace_part == trace_part
                })
                .ok_or(ResidentCompositionError::MissingRelationBatch {
                    component: component.component,
                    instance: component.instance,
                    relation_component,
                    trace_part,
                })?;
            let mut matching = relation.outputs().filter(|output| {
                output.batch_index == batch && output.instance_index == relation_instance
            });
            let output =
                matching
                    .next()
                    .ok_or(ResidentCompositionError::MissingRelationOutput {
                        component: component.component,
                        instance: component.instance,
                        batch,
                        relation_instance,
                    })?;
            if matching.next().is_some() {
                return Err(ResidentCompositionError::DuplicateRelationOutput {
                    component: component.component,
                    instance: component.instance,
                    batch,
                    relation_instance,
                });
            }
            Ok(Some(output.claimed_sum.id()))
        })
        .collect::<Result<Vec<Option<ArenaSlotId>>, ResidentCompositionError>>()?;
    let inputs = CompositionDeviceInputs {
        random_coefficient: cached.random_coefficient.physical,
        forward_twiddles: bind_arena_binding(workspace.arena(), cached.forward_twiddles)
            .map_err(PreparedCompositionError::Arena)?,
        inverse_twiddles: bind_arena_binding(workspace.arena(), cached.inverse_twiddles)
            .map_err(PreparedCompositionError::Arena)?,
        // Pass the relation graph's logically-truncated challenge slices, not
        // slot ids: the composition alpha-power count derives from the slice
        // length, which must be the logical challenge extent even when the
        // physical slot is pooled larger.
        relation_z: relation.z_source(),
        relation_alpha_powers: relation.alpha_powers_source(),
        claimed_sums,
        ext_params: cached.ext_param_bindings(),
    };
    let direct_evaluations =
        canonical_direct_evaluations(cached.direct_retention.as_ref(), &cached.direct_bindings)?
            .into_iter()
            .map(|(plan_column, binding)| {
                Ok(CompositionDirectEvaluationBinding {
                    plan_column,
                    evaluation: bind_arena_binding(workspace.arena(), binding)
                        .map_err(PreparedCompositionError::Arena)?,
                })
            })
            .collect::<Result<Vec<_>, ResidentCompositionError>>()?;
    Ok(PreparedCompositionGraph::prepare_with_proof_bindings(
        workspace.arena(),
        &cached.plan,
        proof_bindings,
        &cached.trace_topology(),
        &inputs,
        &cached.slots,
        cached.requirements.mode,
        cached.direct_retention.as_ref(),
        &direct_evaluations,
    )?)
}

/// Prove that the current statement has exactly the cached workspace topology,
/// allowing only base-parameter values to differ. The full structural equality
/// check is deliberate: the 64-bit plan key is an index, not a soundness proof.
fn require_current_composition_plan<'a>(
    cached: &CompositionPlan,
    current: &'a CompositionPlan,
) -> Result<&'a CompositionPlan, ResidentCompositionError> {
    let drift = || ResidentCompositionError::CurrentPlanTopologyDrift {
        cached_key: cached.key(),
        current_key: current.key(),
    };
    if cached.components.len() != current.components.len() {
        return Err(drift());
    }
    let mut normalized = current.clone();
    for (normalized_component, cached_component) in
        normalized.components.iter_mut().zip(&cached.components)
    {
        if normalized_component.base_param_values.len() != cached_component.base_param_values.len()
        {
            return Err(drift());
        }
        normalized_component
            .base_param_values
            .clone_from(&cached_component.base_param_values);
    }
    if normalized != *cached {
        return Err(drift());
    }
    Ok(current)
}

fn canonical_direct_evaluations(
    retention: Option<&DirectCompositionRetentionPlan>,
    planned_bindings: &[crate::arena_plan::PlannedDirectCompositionBinding],
) -> Result<Vec<(usize, ArenaBinding)>, ResidentCompositionError> {
    let Some(retention) = retention else {
        return if planned_bindings.is_empty() {
            Ok(Vec::new())
        } else {
            Err(ResidentCompositionError::DirectRetentionIdentityDrift(
                "planned bindings exist without a retention plan",
            ))
        };
    };
    if planned_bindings.len() != retention.bindings.len() {
        return Err(ResidentCompositionError::DirectRetentionIdentityDrift(
            "planned occurrence count differs",
        ));
    }

    let mut seen_consumers = vec![false; retention.bindings.len()];
    let mut canonical = vec![None; retention.columns.len()];
    for binding in planned_bindings {
        let seen = seen_consumers.get_mut(binding.consumer).ok_or(
            ResidentCompositionError::DirectRetentionIdentityDrift(
                "planned consumer is out of range",
            ),
        )?;
        if core::mem::replace(seen, true) {
            return Err(ResidentCompositionError::DuplicateDirectRetentionConsumer(
                binding.consumer,
            ));
        }
        let occurrence = &retention.bindings[binding.consumer];
        let column = retention.columns.get(occurrence.column).ok_or(
            ResidentCompositionError::DirectRetentionIdentityDrift(
                "protocol plan column is out of range",
            ),
        )?;
        if occurrence.consumer != binding.consumer
            || binding.plan_column != occurrence.column
            || binding.source != column.source
            || binding.tree != column.tree
            || binding.proof_column != column.proof_column
            || binding.group != column.group
            || binding.column_in_group != column.column_in_group
            || binding.canonical_column != column.canonical_column
            || binding.evaluation_log_size != column.evaluation_log_size
            || binding.evaluation.is_some() != occurrence.direct
        {
            return Err(ResidentCompositionError::DirectRetentionIdentityDrift(
                "planned occurrence differs from the protocol plan",
            ));
        }
        if !occurrence.direct {
            continue;
        }
        let evaluation =
            binding
                .evaluation
                .ok_or(ResidentCompositionError::MissingDirectEvaluation(
                    binding.plan_column,
                ))?;
        let destination = canonical.get_mut(binding.plan_column).ok_or(
            ResidentCompositionError::DirectRetentionIdentityDrift(
                "planned canonical column is out of range",
            ),
        )?;
        if let Some(existing) = destination {
            if *existing != evaluation {
                return Err(ResidentCompositionError::ConflictingDirectEvaluation(
                    binding.plan_column,
                ));
            }
        } else {
            *destination = Some(evaluation);
        }
    }
    if seen_consumers.iter().any(|seen| !seen) {
        return Err(ResidentCompositionError::DirectRetentionIdentityDrift(
            "planned occurrence is missing",
        ));
    }

    let expected = retention
        .bindings
        .iter()
        .filter(|binding| binding.direct)
        .map(|binding| binding.column)
        .collect::<std::collections::BTreeSet<_>>();
    if expected.len() != retention.direct_column_count {
        return Err(ResidentCompositionError::DirectRetentionIdentityDrift(
            "protocol direct column count differs",
        ));
    }
    expected
        .into_iter()
        .map(|plan_column| {
            canonical[plan_column]
                .map(|evaluation| (plan_column, evaluation))
                .ok_or(ResidentCompositionError::MissingDirectEvaluation(
                    plan_column,
                ))
        })
        .collect()
}

fn relation_claimed_sum_key(
    component: &'static str,
    instance: usize,
) -> (&'static str, RelationTracePart, usize) {
    match component {
        "memory_id_to_big" => (
            "memory_id_to_big",
            RelationTracePart::EachMemoryBig,
            instance,
        ),
        "memory_id_to_small" => ("memory_id_to_big", RelationTracePart::MemorySmall, 0),
        _ => (component, RelationTracePart::Component, 0),
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo_cairo_prover::witness::proof_shape::TracePartId;

    use super::*;
    use crate::arena_plan::{
        BufferLifetime, BufferPurpose, CommitmentTreeId, LogicalBufferId, OpenedColumnSource,
        PlannedDirectCompositionBinding, ProofEpoch,
    };
    use crate::direct_composition_retention::{
        DirectCompositionBinding, DirectCompositionColumn, DirectCompositionRetentionPlan,
    };

    fn composition_fixture(base_values: &[u32]) -> CompositionPlan {
        CompositionPlan {
            max_kernel_instrs: 2048,
            total_constraints: 1,
            max_evaluation_log_size: 5,
            components: vec![crate::composition_plan::CompositionComponentPlan {
                component: "fixture",
                instance: 0,
                trace_locations: Vec::new(),
                preprocessed_column_indices: Vec::new(),
                trace_log_size: 4,
                evaluation_log_size: 5,
                n_constraints: 1,
                random_coefficient_offset: 0,
                denominator_inverses: Vec::new(),
                base_param_values: base_values
                    .iter()
                    .copied()
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
                ext_param_values: Vec::new(),
                ext_param_sources: Vec::new(),
                kernels: Vec::new(),
            }],
            wave_kernels: Vec::new(),
        }
    }

    fn direct_fixture() -> (
        DirectCompositionRetentionPlan,
        Vec<PlannedDirectCompositionBinding>,
    ) {
        let source = OpenedColumnSource::Trace {
            component: "fixture",
            part: TracePartId::Main,
            purpose: BufferPurpose::BaseCoefficients,
            ordinal: 0,
        };
        let column = DirectCompositionColumn {
            source,
            tree: CommitmentTreeId::Base,
            proof_column: 3,
            group: 0,
            column_in_group: 0,
            canonical_column: 3,
            coefficient_log_size: 4,
            evaluation_log_size: 5,
            lifetime: BufferLifetime::new(ProofEpoch::BaseCommit, ProofEpoch::Composition).unwrap(),
        };
        let plan = DirectCompositionRetentionPlan {
            columns: vec![column],
            bindings: vec![
                DirectCompositionBinding {
                    consumer: 0,
                    column: 0,
                    consumer_evaluation_log_size: 5,
                    direct: true,
                },
                DirectCompositionBinding {
                    consumer: 1,
                    column: 0,
                    consumer_evaluation_log_size: 5,
                    direct: true,
                },
            ],
            direct_bitmap: vec![3],
            buckets: Vec::new(),
            direct_column_count: 1,
            direct_bytes: 1 << 7,
            cache_key: 7,
        };
        let evaluation = ArenaBinding {
            logical: LogicalBufferId(9),
            physical: ArenaSlotId(11),
            len_words: 1 << 5,
        };
        let bindings = (0..2)
            .map(|consumer| PlannedDirectCompositionBinding {
                consumer,
                plan_column: 0,
                source,
                tree: column.tree,
                proof_column: column.proof_column,
                group: column.group,
                column_in_group: column.column_in_group,
                canonical_column: column.canonical_column,
                evaluation_log_size: column.evaluation_log_size,
                evaluation: Some(evaluation),
            })
            .collect();
        (plan, bindings)
    }

    #[test]
    fn claimed_sums_follow_cairo_component_and_split_memory_order() {
        assert_eq!(
            relation_claimed_sum_key("add_opcode", 0),
            ("add_opcode", RelationTracePart::Component, 0)
        );
        assert_eq!(
            relation_claimed_sum_key("memory_id_to_big", 3),
            ("memory_id_to_big", RelationTracePart::EachMemoryBig, 3)
        );
        assert_eq!(
            relation_claimed_sum_key("memory_id_to_small", 0),
            ("memory_id_to_big", RelationTracePart::MemorySmall, 0)
        );
    }

    #[test]
    fn workspace_reuse_accepts_current_base_values_only_after_full_topology_check() {
        let cached = composition_fixture(&[7, 11]);
        let current = composition_fixture(&[17, 19]);
        assert_eq!(cached.key(), current.key());
        assert!(core::ptr::eq(
            require_current_composition_plan(&cached, &current).unwrap(),
            &current
        ));

        let wrong_slot_count = composition_fixture(&[17]);
        assert!(matches!(
            require_current_composition_plan(&cached, &wrong_slot_count),
            Err(ResidentCompositionError::CurrentPlanTopologyDrift { .. })
        ));

        let mut structural_drift = current.clone();
        structural_drift.components[0].trace_log_size += 1;
        assert!(matches!(
            require_current_composition_plan(&cached, &structural_drift),
            Err(ResidentCompositionError::CurrentPlanTopologyDrift { .. })
        ));
    }

    #[test]
    fn canonical_direct_bindings_deduplicate_reuse_and_reject_identity_drift() {
        let (plan, bindings) = direct_fixture();
        assert_eq!(
            canonical_direct_evaluations(Some(&plan), &bindings).unwrap(),
            vec![(0, bindings[0].evaluation.unwrap())]
        );

        let mut drifted = bindings.clone();
        drifted[1].plan_column = 1;
        assert!(matches!(
            canonical_direct_evaluations(Some(&plan), &drifted),
            Err(ResidentCompositionError::DirectRetentionIdentityDrift(_))
        ));

        let mut conflicting = bindings.clone();
        conflicting[1].evaluation.as_mut().unwrap().physical = ArenaSlotId(12);
        assert_eq!(
            canonical_direct_evaluations(Some(&plan), &conflicting),
            Err(ResidentCompositionError::ConflictingDirectEvaluation(0))
        );

        assert!(matches!(
            canonical_direct_evaluations(Some(&plan), &bindings[..1]),
            Err(ResidentCompositionError::DirectRetentionIdentityDrift(
                "planned occurrence count differs"
            ))
        ));
    }
}
