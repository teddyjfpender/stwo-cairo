use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::compiled_proof::{
    BoundValueRange, ExactPartitionAuthority, ExecutionPrimitive, InPlaceAliasRequirement, OpNode,
    PartitionAuthorityKind, PartitionEffectProjection, ValueRange,
};

pub(super) fn validate_executions(
    plan: &FleetProofPlan,
    operation: &OpNode,
    placement: &FleetOperationPlacement,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let partition = operation_partition(plan, operation)?;
    let effect = plan
        .compiled
        .effect_for(operation.id)
        .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
    let requires_alias = effect.accesses().iter().any(|access| {
        access
            .in_place()
            .is_some_and(|alias| alias.requirement == InPlaceAliasRequirement::Required)
    });
    match partition {
        PartitionAuthorityKind::Monolithic => {
            let worker = placement
                .monolithic_worker()
                .ok_or(FleetPlanError::InvalidOperationDomain(operation.id))?;
            require_worker(workers, worker)?;
        }
        PartitionAuthorityKind::Exact(authority) => {
            if matches!(
                operation.primitive,
                ExecutionPrimitive::OrderedComposite { .. }
            ) || requires_alias
            {
                return Err(FleetPlanError::InvalidOperationDomain(operation.id));
            }
            validate_exact_executions(operation.id, placement, authority, workers)?;
        }
    }
    Ok(())
}

fn validate_exact_executions(
    operation: OpId,
    placement: &FleetOperationPlacement,
    authority: &ExactPartitionAuthority,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    if placement.executions.is_empty() {
        return Err(FleetPlanError::InvalidOperationDomain(operation));
    }
    let domain = authority.domain();
    let granularity = authority.granularity();
    let mut cursor = domain.start;
    let mut assigned = BTreeSet::new();
    for execution in &placement.executions {
        require_worker(workers, execution.worker)?;
        let OperationDomain::Exact(range) = execution.domain else {
            return Err(FleetPlanError::InvalidOperationDomain(operation));
        };
        let start = range.start.checked_sub(domain.start);
        let end = range.end.checked_sub(domain.start);
        if range.is_empty()
            || range.start != cursor
            || range.end > domain.end
            || start.is_none_or(|offset| offset % granularity != 0)
            || end.is_none_or(|offset| offset % granularity != 0)
            || !assigned.insert(execution.worker)
        {
            return Err(FleetPlanError::InvalidOperationDomain(operation));
        }
        cursor = range.end;
    }
    if cursor != domain.end {
        return Err(FleetPlanError::InvalidOperationDomain(operation));
    }
    Ok(())
}

pub(in crate::fleet_plan) fn projected_range(
    compiled: &CompiledProof,
    operation: &OpNode,
    bound: BoundValueRange,
    execution: &FleetOperationExecution,
) -> Result<ValueRange, FleetPlanError> {
    let partition = compiled
        .partitions()
        .iter()
        .find(|partition| partition.id() == operation.partition)
        .map(|partition| partition.kind())
        .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
    match (partition, execution.domain) {
        (PartitionAuthorityKind::Monolithic, OperationDomain::Monolithic) => Ok(bound.value),
        (PartitionAuthorityKind::Exact(authority), OperationDomain::Exact(execution_domain)) => {
            let projection = authority
                .projections()
                .iter()
                .find(|projection| projection.binding() == bound.binding)
                .ok_or(FleetPlanError::InvalidOperationDomain(operation.id))?;
            match projection {
                PartitionEffectProjection::ReplicatedRead { .. } => Ok(bound.value),
                PartitionEffectProjection::ContiguousAxisSlice { .. } => {
                    project_contiguous_range(operation.id, bound.value, authority, execution_domain)
                }
            }
        }
        _ => Err(FleetPlanError::InvalidOperationDomain(operation.id)),
    }
}

fn project_contiguous_range(
    operation: OpId,
    value: ValueRange,
    authority: &ExactPartitionAuthority,
    execution: ElementRange,
) -> Result<ValueRange, FleetPlanError> {
    let domain = authority.domain();
    let inner = value
        .elements
        .len()
        .checked_div(domain.len())
        .filter(|inner| *inner > 0 && value.elements.len() % domain.len() == 0)
        .ok_or(FleetPlanError::InvalidOperationDomain(operation))?;
    let relative_start = execution
        .start
        .checked_sub(domain.start)
        .ok_or(FleetPlanError::InvalidOperationDomain(operation))?;
    let relative_end = execution
        .end
        .checked_sub(domain.start)
        .ok_or(FleetPlanError::InvalidOperationDomain(operation))?;
    let start = relative_start
        .checked_mul(inner)
        .and_then(|offset| value.elements.start.checked_add(offset))
        .ok_or(FleetPlanError::SizeOverflow)?;
    let end = relative_end
        .checked_mul(inner)
        .and_then(|offset| value.elements.start.checked_add(offset))
        .ok_or(FleetPlanError::SizeOverflow)?;
    if execution.is_empty() || execution.end > domain.end || end > value.elements.end {
        return Err(FleetPlanError::InvalidOperationDomain(operation));
    }
    Ok(ValueRange {
        version: value.version,
        elements: ElementRange { start, end },
    })
}

fn operation_partition<'a>(
    plan: &'a FleetProofPlan,
    operation: &OpNode,
) -> Result<&'a PartitionAuthorityKind, FleetPlanError> {
    plan.compiled
        .partitions()
        .iter()
        .find(|partition| partition.id() == operation.partition)
        .map(|partition| partition.kind())
        .ok_or(FleetPlanError::InvalidOperation(operation.id))
}

pub(super) fn range_covered(target: ElementRange, ranges: &[ValueRange]) -> bool {
    let mut ranges = ranges
        .iter()
        .map(|range| range.elements)
        .filter(|range| target.overlaps(*range))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut cursor = target.start;
    for range in ranges {
        let start = range.start.max(target.start);
        let end = range.end.min(target.end);
        if start != cursor {
            return false;
        }
        cursor = end;
    }
    cursor == target.end
}

pub(super) fn ranges_overlap(left: ValueRange, right: ValueRange) -> bool {
    left.version == right.version && left.elements.overlaps(right.elements)
}

pub(super) fn index_unique<T, K: Ord + Copy>(
    values: &[T],
    key: impl Fn(&T) -> K,
) -> Option<BTreeMap<K, &T>> {
    let mut indexed = BTreeMap::new();
    for value in values {
        if indexed.insert(key(value), value).is_some() {
            return None;
        }
    }
    Some(indexed)
}
