use std::collections::BTreeMap;

use super::{
    operation, operation_placement, owner_ready_at, projected_range, range_covered, require_worker,
    validate_value_range,
};
use crate::compiled_proof::{exact_partial_atomic_carry_forward, ValueDesc, ValueOrigin};
use crate::fleet_plan::*;

pub(super) fn validate(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    for value in plan.compiled.values() {
        let total = value
            .layout
            .element_count()
            .map_err(|_| FleetPlanError::SizeOverflow)?;
        let mut owners = plan
            .placement
            .owners
            .iter()
            .filter(|owner| owner.value.version == value.version)
            .collect::<Vec<_>>();
        owners.sort_unstable_by_key(|owner| {
            (
                owner.value.elements.start,
                owner.value.elements.end,
                owner.worker,
            )
        });
        let mut cursor = 0usize;
        for owner in owners {
            require_worker(workers, owner.worker)?;
            validate_value_range(plan, owner.value)?;
            if owner.value.elements.start != cursor
                || !owner.live.is_valid()
                || owner.live.end > plan.placement.terminal_step
                || owner_ready_at(plan, owner)? >= owner.live.end
            {
                return Err(FleetPlanError::OwnershipCoverage(value.version));
            }
            validate_origin(plan, owner, value)?;
            cursor = owner.value.elements.end;
        }
        if cursor != total {
            return Err(FleetPlanError::OwnershipCoverage(value.version));
        }
    }
    if let Some(owner) = plan
        .placement
        .owners
        .iter()
        .find(|owner| plan.compiled.value(owner.value.version).is_none())
    {
        return Err(FleetPlanError::UnknownValue(owner.value.version));
    }
    Ok(())
}

fn validate_origin(
    plan: &FleetProofPlan,
    owner: &FleetOwnerPlacement,
    value: &ValueDesc,
) -> Result<(), FleetPlanError> {
    let valid = match value.origin {
        ValueOrigin::ExternalInput(_) => owner.live.start == ScheduleStep(0),
        ValueOrigin::Constant(_) => {
            owner.live.start == ScheduleStep(0) && owner.live.end == plan.placement.terminal_step
        }
        ValueOrigin::TranscriptOutput(_) => owner.worker == plan.placement.topology.coordinator,
        ValueOrigin::OpOutput(producer) => {
            let placement = operation_placement(plan, producer)?;
            let effect = plan
                .compiled
                .effect_for(producer)
                .ok_or(FleetPlanError::InvalidOperation(producer))?;
            let operation = operation(plan, producer)?;
            let mut destinations = Vec::new();
            for execution in placement
                .executions
                .iter()
                .filter(|execution| execution.worker == owner.worker)
            {
                for access in effect.accesses() {
                    let Some(destination) = access.destination() else {
                        continue;
                    };
                    let range =
                        projected_range(&plan.compiled, operation, *destination, execution)?;
                    let carried = (execution.domain == OperationDomain::Monolithic)
                        .then(|| exact_partial_atomic_carry_forward(plan.compiled.values(), access))
                        .flatten()
                        .filter(|carry| carry.full_destination.version == value.version)
                        .filter(|carry| {
                            plan.placement.owners.iter().any(|source_owner| {
                                source_owner.worker == owner.worker
                                    && source_owner.value == carry.full_source
                                    && source_owner.live.contains(placement.during)
                                    && owner_ready_at(plan, source_owner)
                                        .is_ok_and(|ready| ready <= placement.during.start)
                            })
                        });
                    if let Some(carry) = carried {
                        destinations.push(carry.full_destination);
                    } else if range.version == value.version {
                        destinations.push(range);
                    }
                }
            }
            !destinations.is_empty()
                && owner.live.start == placement.during.start
                && owner.live.contains(placement.during)
                && range_covered(owner.value.elements, &destinations)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(FleetPlanError::InvalidProducer(value.version))
    }
}
