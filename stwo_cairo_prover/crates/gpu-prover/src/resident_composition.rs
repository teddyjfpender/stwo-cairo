//! Exact resident bindings for the prepared Cairo composition graph.

use stwo_backend_cuda::{ArenaSlotId, PreparedRelationGraph};

use crate::composition_plan::CompositionExtParamSource;
use crate::graphs::GraphWorkspace;
use crate::prepared_composition::{
    CompositionDeviceInputs, PreparedCompositionError, PreparedCompositionGraph,
};
use crate::relation::RelationTracePart;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentCompositionError {
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
) -> Result<PreparedCompositionGraph<'a>, ResidentCompositionError> {
    let planned = workspace.plan().composition();
    let claimed_sums = planned
        .plan
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
        random_coefficient: planned.random_coefficient.physical,
        forward_twiddles: planned.forward_twiddles.physical,
        inverse_twiddles: planned.inverse_twiddles.physical,
        relation_z: relation.z_source().id(),
        relation_alpha_powers: relation.alpha_powers_source().id(),
        claimed_sums,
        ext_params: planned.ext_param_bindings(),
    };
    Ok(PreparedCompositionGraph::prepare(
        workspace.arena(),
        &planned.plan,
        &planned.trace_topology(),
        &inputs,
        &planned.slots,
    )?)
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
    use super::*;

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
}
