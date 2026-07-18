use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::compiled_proof::{
    ExecutionPrimitive, InPlaceAliasRequirement, ValueDesc, ValueRange, ValueVersion,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StorageId(pub u32);

/// One stable worker-local virtual allocation. Physical backing remains mapped
/// for the installed graph unless an exact whole-storage VMM reclaim says when
/// it is safely unmapped and remapped at the same address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageDesc {
    pub id: StorageId,
    pub worker: WorkerId,
    pub bytes: usize,
    pub alignment_bytes: usize,
}

pub(super) fn validate(
    plan: &FleetProofPlan,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    let mut storages = BTreeMap::new();
    for (ordinal, storage) in plan.placement.storages.iter().enumerate() {
        if storage.id.0 as usize != ordinal
            || storages.insert(storage.id, storage).is_some()
            || !workers.contains_key(&storage.worker)
            || storage.bytes == 0
            || storage.alignment_bytes == 0
            || !storage.alignment_bytes.is_power_of_two()
        {
            return Err(FleetPlanError::InvalidStorage(storage.id));
        }
    }

    for binding in &plan.placement.storage_bindings {
        validate_binding(plan, binding, &storages)?;
    }
    validate_location_coverage(plan)?;
    validate_aliases(plan, &storages)?;
    validate_reuse(plan)?;
    validate_proof_output(plan, &storages)?;

    for storage in storages.keys() {
        if !plan
            .placement
            .storage_bindings
            .iter()
            .any(|binding| binding.storage == *storage)
        {
            return Err(FleetPlanError::InvalidStorage(*storage));
        }
    }
    Ok(())
}

fn validate_binding(
    plan: &FleetProofPlan,
    binding: &FleetStoragePlacement,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let storage = storages
        .get(&binding.storage)
        .ok_or(FleetPlanError::InvalidStorage(binding.storage))?;
    let value = value(plan, binding.value.version)?;
    let bytes = range_bytes(value, binding.value.elements)?;
    let end = binding
        .offset_bytes
        .checked_add(bytes)
        .ok_or(FleetPlanError::SizeOverflow)?;
    if value.alignment == 0
        || !value.alignment.is_power_of_two()
        || storage.alignment_bytes < value.alignment
        || binding.offset_bytes % value.alignment != 0
        || end > storage.bytes
        || matching_locations(plan, binding) != 1
    {
        return Err(invalid_binding(binding));
    }
    Ok(())
}

fn validate_location_coverage(plan: &FleetProofPlan) -> Result<(), FleetPlanError> {
    for owner in &plan.placement.owners {
        validate_one_location_coverage(plan, owner.value, owner.worker)?;
    }
    for replica in &plan.placement.replicas {
        validate_one_location_coverage(plan, replica.value, replica.worker)?;
    }
    for binding in &plan.placement.storage_bindings {
        if matching_locations(plan, binding) != 1 {
            return Err(invalid_binding(binding));
        }
    }
    Ok(())
}

fn validate_one_location_coverage(
    plan: &FleetProofPlan,
    location: ValueRange,
    worker: WorkerId,
) -> Result<(), FleetPlanError> {
    let mut ranges = plan
        .placement
        .storage_bindings
        .iter()
        .filter_map(|binding| {
            let storage = storage(plan, binding.storage)?;
            (storage.worker == worker
                && binding.value.version == location.version
                && location.elements.contains(binding.value.elements))
            .then_some(binding.value.elements)
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut cursor = location.elements.start;
    for range in ranges {
        if range.start != cursor || range.end > location.elements.end {
            return Err(FleetPlanError::StorageCoverage(location.version));
        }
        cursor = range.end;
    }
    if cursor != location.elements.end {
        return Err(FleetPlanError::StorageCoverage(location.version));
    }
    Ok(())
}

fn validate_aliases(
    plan: &FleetProofPlan,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let mut keys = BTreeSet::new();
    for alias in &plan.placement.in_place_aliases {
        if !keys.insert((alias.operation, alias.alias)) {
            return Err(invalid_alias(alias.operation, alias.alias));
        }
        validate_alias(plan, alias, storages)?;
    }

    for operation in plan.compiled.operations() {
        let effect = plan
            .compiled
            .effect_for(operation.id)
            .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
        for access in effect.accesses() {
            let Some(authority) = access.in_place() else {
                continue;
            };
            let count = plan
                .placement
                .in_place_aliases
                .iter()
                .filter(|placement| {
                    placement.operation == operation.id && placement.alias == authority.id
                })
                .count();
            if count > 1 || authority.requirement == InPlaceAliasRequirement::Required && count != 1
            {
                return Err(invalid_alias(operation.id, authority.id));
            }
        }
    }
    Ok(())
}

fn validate_alias(
    plan: &FleetProofPlan,
    alias: &InPlaceAliasPlacement,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let operation = plan
        .compiled
        .operations()
        .get(alias.operation.0 as usize)
        .filter(|operation| operation.id == alias.operation)
        .ok_or(FleetPlanError::UnknownOperation(alias.operation))?;
    // A composite is atomic at fleet-plan granularity. Without child-indexed
    // alias timing, the validator cannot prove that no later child reads the
    // overwritten source. Keep it out-of-place until that authority exists.
    if matches!(
        &operation.primitive,
        ExecutionPrimitive::OrderedComposite { .. }
    ) {
        return Err(invalid_alias(operation.id, alias.alias));
    }
    let operation_placement = operation_placement(plan, operation.id)?;
    let operation_worker = operation_placement
        .monolithic_worker()
        .ok_or_else(|| invalid_alias(operation.id, alias.alias))?;
    let effect = plan
        .compiled
        .effect_for(operation.id)
        .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
    let bound = effect
        .in_place_alias(alias.alias)
        .ok_or_else(|| invalid_alias(operation.id, alias.alias))?;
    let source = bound
        .source()
        .ok_or_else(|| invalid_alias(operation.id, alias.alias))?
        .value;
    let destination = bound
        .destination()
        .ok_or_else(|| invalid_alias(operation.id, alias.alias))?
        .value;
    let storage = storages
        .get(&alias.storage)
        .ok_or(FleetPlanError::InvalidStorage(alias.storage))?;
    let (source_binding, source_offset) = full_binding_offset(plan, alias.storage, source)?;
    let (destination_binding, destination_offset) =
        full_binding_offset(plan, alias.storage, destination)?;
    let source_bytes = range_bytes(value(plan, source.version)?, source.elements)?;
    let destination_bytes = range_bytes(value(plan, destination.version)?, destination.elements)?;
    let source_live = binding_live(plan, source_binding)?;
    let destination_live = binding_live(plan, destination_binding)?;
    if source.version == destination.version
        || source_bytes != destination_bytes
        || storage.worker != operation_worker
        || source_offset != alias.offset_bytes
        || destination_offset != alias.offset_bytes
        || source_live.end != operation_placement.during.end
        || destination_live.start != operation_placement.during.start
        || has_concurrent_source_consumer(plan, operation.id, source, operation_placement)?
    {
        return Err(invalid_alias(operation.id, alias.alias));
    }
    Ok(())
}

fn validate_reuse(plan: &FleetProofPlan) -> Result<(), FleetPlanError> {
    for (index, left) in plan.placement.storage_bindings.iter().enumerate() {
        let left_live = binding_live(plan, left)?;
        for right in &plan.placement.storage_bindings[index + 1..] {
            if left.storage != right.storage
                || !binding_bytes(plan, left)?.overlaps(binding_bytes(plan, right)?)
                || !left_live.overlaps(binding_live(plan, right)?)
            {
                continue;
            }
            if !plan
                .placement
                .in_place_aliases
                .iter()
                .any(|alias| alias_matches_pair(plan, alias, left, right))
            {
                return Err(FleetPlanError::IllegalStorageReuse(left.storage));
            }
        }
    }
    Ok(())
}

fn validate_proof_output(
    plan: &FleetProofPlan,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let id = plan.placement.output_storage;
    let storage = storages
        .get(&id)
        .ok_or(FleetPlanError::InvalidProofOutput(id))?;
    let output = plan.compiled.output();
    let expected_bytes = output
        .layout
        .total_words
        .checked_mul(size_of::<u32>())
        .ok_or(FleetPlanError::SizeOverflow)?;
    if storage.worker != plan.placement.topology.coordinator
        || storage.bytes != expected_bytes
        || storage.alignment_bytes < align_of::<u32>()
    {
        return Err(FleetPlanError::InvalidProofOutput(id));
    }
    let bindings = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == id)
        .collect::<Vec<_>>();
    let mut covered_bindings = 0usize;
    for fragment in &output.fragments {
        let offset = fragment
            .destination
            .start
            .checked_mul(size_of::<u32>())
            .ok_or(FleetPlanError::SizeOverflow)?;
        let bytes = fragment
            .destination
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or(FleetPlanError::SizeOverflow)?;
        let source = value(plan, fragment.source.version)?;
        let source_bytes = range_bytes(source, fragment.source.elements)?;
        if source_bytes != bytes {
            return Err(FleetPlanError::InvalidProofOutput(id));
        }

        let mut covered = bindings
            .iter()
            .copied()
            .filter(|binding| {
                binding.value.version == fragment.source.version
                    && fragment.source.elements.contains(binding.value.elements)
            })
            .collect::<Vec<_>>();
        covered.sort_unstable_by_key(|binding| {
            (
                binding.value.elements.start,
                binding.value.elements.end,
                binding.offset_bytes,
            )
        });
        let mut cursor = fragment.source.elements.start;
        for binding in covered {
            let relative_elements = binding
                .value
                .elements
                .start
                .checked_sub(fragment.source.elements.start)
                .ok_or(FleetPlanError::SizeOverflow)?;
            let expected_offset = relative_elements
                .checked_mul(source.layout.element.bytes)
                .and_then(|relative| offset.checked_add(relative))
                .ok_or(FleetPlanError::SizeOverflow)?;
            if binding.value.elements.start != cursor
                || binding.value.elements.end > fragment.source.elements.end
                || binding.offset_bytes != expected_offset
                || binding_live(plan, binding)?.end != plan.placement.terminal_step
            {
                return Err(FleetPlanError::InvalidProofOutput(id));
            }
            cursor = binding.value.elements.end;
            covered_bindings = covered_bindings
                .checked_add(1)
                .ok_or(FleetPlanError::SizeOverflow)?;
        }
        if cursor != fragment.source.elements.end {
            return Err(FleetPlanError::InvalidProofOutput(id));
        }
    }
    if covered_bindings == bindings.len() {
        Ok(())
    } else {
        Err(FleetPlanError::InvalidProofOutput(id))
    }
}

fn has_concurrent_source_consumer(
    plan: &FleetProofPlan,
    aliased_operation: OpId,
    source: ValueRange,
    aliased_placement: &FleetOperationPlacement,
) -> Result<bool, FleetPlanError> {
    let Some(aliased_worker) = aliased_placement.monolithic_worker() else {
        return Ok(true);
    };
    for placement in &plan.placement.operations {
        if placement.operation == aliased_operation
            || !placement.executes_on(aliased_worker)
            || !placement.during.overlaps(aliased_placement.during)
        {
            continue;
        }
        let operation = operation(plan, placement.operation)?;
        let effect = plan
            .compiled
            .effect_for(operation.id)
            .ok_or(FleetPlanError::InvalidOperation(operation.id))?;
        if effect.accesses().iter().any(|effect| {
            effect
                .source()
                .is_some_and(|read| ranges_overlap(read.value, source))
        }) {
            return Ok(true);
        }
    }
    if plan.placement.transitions.iter().any(|transition| {
        transition.source_worker == aliased_worker
            && ranges_overlap(transition.value, source)
            && transition.during.overlaps(aliased_placement.during)
    }) {
        return Ok(true);
    }
    for spill in plan
        .placement
        .spills
        .iter()
        .filter(|spill| spill.store.worker == aliased_worker)
    {
        for chunk in spill
            .chunks
            .iter()
            .filter(|chunk| ranges_overlap(chunk.value, source))
        {
            let [d2h, _, _, h2d] = spill.bounds(chunk.id)?;
            let cycle = ScheduleRange::new(d2h.during.start, h2d.during.end)
                .ok_or(FleetPlanError::InvalidSchedule)?;
            if cycle.overlaps(aliased_placement.during) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn alias_matches_pair(
    plan: &FleetProofPlan,
    alias: &InPlaceAliasPlacement,
    left: &FleetStoragePlacement,
    right: &FleetStoragePlacement,
) -> bool {
    let Some(operation) = plan.compiled.operations().get(alias.operation.0 as usize) else {
        return false;
    };
    let Some(effect) = plan.compiled.effect_for(operation.id) else {
        return false;
    };
    let Some(bound) = effect.in_place_alias(alias.alias) else {
        return false;
    };
    let (Some(source), Some(destination)) = (bound.source(), bound.destination()) else {
        return false;
    };
    let matches = |source_binding: &FleetStoragePlacement,
                   destination_binding: &FleetStoragePlacement| {
        alias.storage == source_binding.storage
            && source_binding.storage == destination_binding.storage
            && source_binding.value.version == source.value.version
            && destination_binding.value.version == destination.value.version
            && source_binding.offset_bytes == destination_binding.offset_bytes
            && is_full_binding(plan, source_binding)
            && is_full_binding(plan, destination_binding)
    };
    matches(left, right) || matches(right, left)
}

fn full_binding_offset(
    plan: &FleetProofPlan,
    storage: StorageId,
    target: ValueRange,
) -> Result<(&FleetStoragePlacement, usize), FleetPlanError> {
    let value = value(plan, target.version)?;
    let full = ElementRange::new(
        0,
        value
            .layout
            .element_count()
            .map_err(|_| FleetPlanError::SizeOverflow)?,
    )
    .ok_or(FleetPlanError::InvalidRange(target.version))?;
    if !full.contains(target.elements) {
        return Err(FleetPlanError::InvalidRange(target.version));
    }
    let mut matches = plan.placement.storage_bindings.iter().filter(|binding| {
        binding.storage == storage
            && binding.value.version == target.version
            && binding.value.elements == full
    });
    let binding = matches
        .next()
        .ok_or(FleetPlanError::InvalidStorageBinding {
            value: target.version,
            storage,
        })?;
    if matches.next().is_some() {
        return Err(invalid_binding(binding));
    }
    let offset = target
        .elements
        .start
        .checked_mul(value.layout.element.bytes)
        .and_then(|relative| binding.offset_bytes.checked_add(relative))
        .ok_or(FleetPlanError::SizeOverflow)?;
    Ok((binding, offset))
}

fn is_full_binding(plan: &FleetProofPlan, binding: &FleetStoragePlacement) -> bool {
    plan.compiled
        .value(binding.value.version)
        .is_some_and(|value| {
            value
                .layout
                .element_count()
                .ok()
                .and_then(|words| ElementRange::new(0, words))
                == Some(binding.value.elements)
        })
}

pub(super) fn binding_live(
    plan: &FleetProofPlan,
    binding: &FleetStoragePlacement,
) -> Result<ScheduleRange, FleetPlanError> {
    let worker = storage(plan, binding.storage)
        .ok_or(FleetPlanError::InvalidStorage(binding.storage))?
        .worker;
    let mut lives = plan
        .placement
        .owners
        .iter()
        .filter(|owner| {
            owner.worker == worker
                && owner.value.version == binding.value.version
                && owner.value.elements.contains(binding.value.elements)
        })
        .map(|owner| owner.live)
        .chain(
            plan.placement
                .replicas
                .iter()
                .filter(|replica| {
                    replica.worker == worker
                        && replica.value.version == binding.value.version
                        && replica.value.elements.contains(binding.value.elements)
                })
                .map(|replica| replica.live),
        );
    let live = lives.next().ok_or_else(|| invalid_binding(binding))?;
    if lives.next().is_some() {
        return Err(invalid_binding(binding));
    }
    Ok(live)
}

pub(super) fn storage(plan: &FleetProofPlan, id: StorageId) -> Option<&StorageDesc> {
    plan.placement
        .storages
        .get(id.0 as usize)
        .filter(|storage| storage.id == id)
}

pub(super) fn range_bytes(value: &ValueDesc, range: ElementRange) -> Result<usize, FleetPlanError> {
    let total = value
        .layout
        .element_count()
        .map_err(|_| FleetPlanError::SizeOverflow)?;
    if range.is_empty() || range.end > total {
        return Err(FleetPlanError::InvalidRange(value.version));
    }
    range
        .len()
        .checked_mul(value.layout.element.bytes)
        .ok_or(FleetPlanError::SizeOverflow)
}

fn value(plan: &FleetProofPlan, version: ValueVersion) -> Result<&ValueDesc, FleetPlanError> {
    plan.compiled
        .value(version)
        .ok_or(FleetPlanError::UnknownValue(version))
}

fn operation(
    plan: &FleetProofPlan,
    id: OpId,
) -> Result<&crate::compiled_proof::OpNode, FleetPlanError> {
    plan.compiled
        .operations()
        .get(id.0 as usize)
        .filter(|operation| operation.id == id)
        .ok_or(FleetPlanError::UnknownOperation(id))
}

fn operation_placement(
    plan: &FleetProofPlan,
    id: OpId,
) -> Result<&FleetOperationPlacement, FleetPlanError> {
    plan.placement
        .operations
        .get(id.0 as usize)
        .filter(|placement| placement.operation == id)
        .ok_or(FleetPlanError::MissingOperation(id))
}

fn matching_locations(plan: &FleetProofPlan, binding: &FleetStoragePlacement) -> usize {
    let Some(storage) = storage(plan, binding.storage) else {
        return 0;
    };
    plan.placement
        .owners
        .iter()
        .filter(|owner| {
            owner.worker == storage.worker
                && owner.value.version == binding.value.version
                && owner.value.elements.contains(binding.value.elements)
        })
        .count()
        + plan
            .placement
            .replicas
            .iter()
            .filter(|replica| {
                replica.worker == storage.worker
                    && replica.value.version == binding.value.version
                    && replica.value.elements.contains(binding.value.elements)
            })
            .count()
}

#[derive(Clone, Copy)]
pub(super) struct ByteWindow {
    pub start: usize,
    pub end: usize,
}

impl ByteWindow {
    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

pub(super) fn binding_bytes(
    plan: &FleetProofPlan,
    binding: &FleetStoragePlacement,
) -> Result<ByteWindow, FleetPlanError> {
    let bytes = range_bytes(value(plan, binding.value.version)?, binding.value.elements)?;
    Ok(ByteWindow {
        start: binding.offset_bytes,
        end: binding
            .offset_bytes
            .checked_add(bytes)
            .ok_or(FleetPlanError::SizeOverflow)?,
    })
}

fn ranges_overlap(left: ValueRange, right: ValueRange) -> bool {
    left.version == right.version && left.elements.overlaps(right.elements)
}

fn invalid_binding(binding: &FleetStoragePlacement) -> FleetPlanError {
    FleetPlanError::InvalidStorageBinding {
        value: binding.value.version,
        storage: binding.storage,
    }
}

fn invalid_alias(operation: OpId, alias: crate::compiled_proof::InPlaceAliasId) -> FleetPlanError {
    FleetPlanError::InvalidInPlaceAlias { operation, alias }
}
