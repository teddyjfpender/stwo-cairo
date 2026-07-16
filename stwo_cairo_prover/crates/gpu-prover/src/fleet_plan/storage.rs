use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::compiled_proof::{InPlaceAliasRequirement, ValueDesc, ValueRange, ValueVersion};

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
    let operation_placement = operation_placement(plan, operation.id)?;
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
    let source_binding = exact_binding(plan, alias.storage, source)?;
    let destination_binding = exact_binding(plan, alias.storage, destination)?;
    let source_bytes = range_bytes(value(plan, source.version)?, source.elements)?;
    let destination_bytes = range_bytes(value(plan, destination.version)?, destination.elements)?;
    let source_live = binding_live(plan, source_binding)?;
    let destination_live = binding_live(plan, destination_binding)?;
    if source.version == destination.version
        || source_bytes != destination_bytes
        || storage.worker != operation_placement.worker
        || source_binding.offset_bytes != alias.offset_bytes
        || destination_binding.offset_bytes != alias.offset_bytes
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
    let destinations = output_ranges(&output.layout);
    if output.sections.len() != destinations.len() {
        return Err(FleetPlanError::InvalidProofOutput(id));
    }
    let bindings = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == id)
        .collect::<Vec<_>>();
    if bindings.len() != output.sections.len() {
        return Err(FleetPlanError::InvalidProofOutput(id));
    }
    for (section, destination) in output.sections.iter().zip(destinations) {
        let offset = destination
            .start
            .checked_mul(size_of::<u32>())
            .ok_or(FleetPlanError::SizeOverflow)?;
        let bytes = destination
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or(FleetPlanError::SizeOverflow)?;
        let source_bytes = range_bytes(value(plan, section.value)?, section.elements)?;
        if source_bytes != bytes {
            return Err(FleetPlanError::InvalidProofOutput(id));
        }
        let mut exact = bindings.iter().filter(|binding| {
            binding.value.version == section.value
                && binding.value.elements == section.elements
                && binding.offset_bytes == offset
        });
        let Some(binding) = exact.next() else {
            return Err(FleetPlanError::InvalidProofOutput(id));
        };
        if exact.next().is_some()
            || binding_live(plan, binding)?.end != plan.placement.terminal_step
        {
            return Err(FleetPlanError::InvalidProofOutput(id));
        }
    }
    Ok(())
}

fn has_concurrent_source_consumer(
    plan: &FleetProofPlan,
    aliased_operation: OpId,
    source: ValueRange,
    aliased_placement: &FleetOperationPlacement,
) -> Result<bool, FleetPlanError> {
    for placement in &plan.placement.operations {
        if placement.operation == aliased_operation
            || placement.worker != aliased_placement.worker
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
        transition.source_worker == aliased_placement.worker
            && ranges_overlap(transition.value, source)
            && transition.during.overlaps(aliased_placement.during)
    }) {
        return Ok(true);
    }
    for spill in plan
        .placement
        .spills
        .iter()
        .filter(|spill| spill.store.worker == aliased_placement.worker)
    {
        for chunk in spill
            .chunks
            .iter()
            .filter(|chunk| ranges_overlap(chunk.value, source))
        {
            let [d2h, _, _, _] = spill.chain(chunk.id)?;
            if d2h.during.overlaps(aliased_placement.during) {
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
            && source_binding.value == source.value
            && destination_binding.value == destination.value
            && source_binding.offset_bytes == alias.offset_bytes
            && destination_binding.offset_bytes == alias.offset_bytes
    };
    matches(left, right) || matches(right, left)
}

fn exact_binding(
    plan: &FleetProofPlan,
    storage: StorageId,
    value: ValueRange,
) -> Result<&FleetStoragePlacement, FleetPlanError> {
    let mut matches = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == storage && binding.value == value);
    let binding = matches
        .next()
        .ok_or(FleetPlanError::InvalidStorageBinding {
            value: value.version,
            storage,
        })?;
    if matches.next().is_some() {
        return Err(invalid_binding(binding));
    }
    Ok(binding)
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

fn output_ranges(
    layout: &crate::proof_bundle::ResidentProofBundleLayout,
) -> [core::ops::Range<usize>; 8] {
    [
        layout.commitments.clone(),
        layout.interaction_claim.clone(),
        layout.interaction_pow.clone(),
        layout.sampled_values.clone(),
        layout.fri_commitments.clone(),
        layout.final_line_poly.clone(),
        layout.query_pow.clone(),
        layout.decommitment.clone(),
    ]
}
