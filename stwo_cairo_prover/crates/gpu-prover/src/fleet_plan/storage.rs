use std::collections::{BTreeMap, BTreeSet};

use super::{
    DeclaredReplica, ElementRange, FleetPlanError, FleetProofPlan, OperationAssignment,
    OperationDesc, OperationId, OwnedValueRange, ScheduleRange, ValueDesc, ValueId, WorkerId,
    WorkerSpec,
};
pub use crate::compiled_proof::EffectContractId;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StorageId(pub u32);

/// One worker-local physical allocation retained for the installed graph.
/// Logical values may share it only through disjoint byte ranges, disjoint
/// lifetimes, or a verified alias. No spill or logical death releases VRAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageDesc {
    pub id: StorageId,
    pub worker: WorkerId,
    pub bytes: usize,
    pub alignment_bytes: usize,
}

/// Exact placement of one canonical owner or declared replica in storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageBinding {
    pub storage: StorageId,
    pub value: ValueId,
    pub elements: ElementRange,
    pub worker: WorkerId,
    pub offset_bytes: usize,
    pub bytes: usize,
}

/// Explicit permission for one operation to consume an old ValueId and
/// produce a distinct ValueId in the exact same physical byte range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InPlaceAlias {
    pub operation: OperationId,
    pub effect: EffectContractId,
    pub source: ValueId,
    pub source_elements: ElementRange,
    pub destination: ValueId,
    pub destination_elements: ElementRange,
    pub storage: StorageId,
    pub offset_bytes: usize,
    pub bytes: usize,
}

pub(super) fn validate(
    plan: &FleetProofPlan,
    values: &BTreeMap<ValueId, &ValueDesc>,
    operations: &BTreeMap<OperationId, &OperationDesc>,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    workers: &BTreeMap<WorkerId, &WorkerSpec>,
) -> Result<(), FleetPlanError> {
    for operation in operations.values() {
        if operation.effect_identity.0 == [0; 32] {
            return Err(FleetPlanError::InvalidEffectContract(operation.id));
        }
    }

    let mut storages = BTreeMap::new();
    for (ordinal, storage) in plan.input.storages.iter().enumerate() {
        if storage.id.0 as usize != ordinal || storages.insert(storage.id, storage).is_some() {
            return Err(FleetPlanError::InvalidStorage(storage.id));
        }
        if !workers.contains_key(&storage.worker)
            || storage.bytes == 0
            || storage.alignment_bytes == 0
            || !storage.alignment_bytes.is_power_of_two()
        {
            return Err(FleetPlanError::InvalidStorage(storage.id));
        }
    }

    for binding in &plan.input.storage_bindings {
        validate_binding(plan, binding, values, &storages)?;
    }
    validate_exact_location_coverage(plan)?;

    let mut alias_keys = BTreeSet::new();
    for alias in &plan.input.in_place_aliases {
        let key = (
            alias.operation,
            alias.source,
            alias.source_elements.start,
            alias.source_elements.end,
            alias.destination,
            alias.destination_elements.start,
            alias.destination_elements.end,
        );
        if !alias_keys.insert(key) {
            return Err(FleetPlanError::InvalidInPlaceAlias(alias.operation));
        }
        validate_alias(plan, alias, operations, assignments, &storages)?;
    }

    for (index, left) in plan.input.storage_bindings.iter().enumerate() {
        let left_live = binding_live(plan, left)?;
        for right in &plan.input.storage_bindings[index + 1..] {
            if left.storage != right.storage
                || !byte_range(left)?.overlaps(byte_range(right)?)
                || !left_live.overlaps(binding_live(plan, right)?)
            {
                continue;
            }
            if !plan
                .input
                .in_place_aliases
                .iter()
                .any(|alias| alias_matches_pair(alias, left, right))
            {
                return Err(FleetPlanError::IllegalStorageReuse(left.storage));
            }
        }
    }

    for storage in storages.keys() {
        if !plan
            .input
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
    binding: &StorageBinding,
    values: &BTreeMap<ValueId, &ValueDesc>,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let storage = storages
        .get(&binding.storage)
        .ok_or(FleetPlanError::InvalidStorage(binding.storage))?;
    let value = values
        .get(&binding.value)
        .ok_or(FleetPlanError::UnknownValue(binding.value))?;
    let expected = binding
        .elements
        .len()
        .checked_mul(value.layout.element.bytes)
        .ok_or(FleetPlanError::SizeOverflow)?;
    let end = binding
        .offset_bytes
        .checked_add(binding.bytes)
        .ok_or(FleetPlanError::SizeOverflow)?;
    if binding.elements.is_empty()
        || binding.elements.end > value.layout.element_count()?
        || binding.worker != storage.worker
        || binding.bytes != expected
        || value.alignment_bytes == 0
        || !value.alignment_bytes.is_power_of_two()
        || storage.alignment_bytes < value.alignment_bytes
        || binding.offset_bytes % value.alignment_bytes != 0
        || end > storage.bytes
        || matching_locations(plan, binding) != 1
    {
        return Err(FleetPlanError::InvalidStorageBinding {
            value: binding.value,
            storage: binding.storage,
        });
    }
    Ok(())
}

fn validate_exact_location_coverage(plan: &FleetProofPlan) -> Result<(), FleetPlanError> {
    for owner in &plan.input.owners {
        let count = plan
            .input
            .storage_bindings
            .iter()
            .filter(|binding| binding_matches_owner(binding, owner))
            .count();
        if count != 1 {
            return Err(FleetPlanError::StorageCoverage(owner.value));
        }
    }
    for replica in &plan.input.replicas {
        let count = plan
            .input
            .storage_bindings
            .iter()
            .filter(|binding| binding_matches_replica(binding, replica))
            .count();
        if count != 1 {
            return Err(FleetPlanError::StorageCoverage(replica.value));
        }
    }
    Ok(())
}

fn validate_alias(
    plan: &FleetProofPlan,
    alias: &InPlaceAlias,
    operations: &BTreeMap<OperationId, &OperationDesc>,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
    storages: &BTreeMap<StorageId, &StorageDesc>,
) -> Result<(), FleetPlanError> {
    let operation = operations
        .get(&alias.operation)
        .ok_or(FleetPlanError::UnknownOperation(alias.operation))?;
    let assignment = assignments
        .get(&alias.operation)
        .ok_or(FleetPlanError::MissingAssignment(alias.operation))?;
    let storage = storages
        .get(&alias.storage)
        .ok_or(FleetPlanError::InvalidStorage(alias.storage))?;
    let source = unique_binding(
        plan,
        alias.storage,
        alias.source,
        alias.source_elements,
        alias.offset_bytes,
        alias.bytes,
    )?;
    let destination = unique_binding(
        plan,
        alias.storage,
        alias.destination,
        alias.destination_elements,
        alias.offset_bytes,
        alias.bytes,
    )?;
    let source_live = binding_live(plan, source)?;
    let destination_live = binding_live(plan, destination)?;
    let overlapping_reads = operation
        .reads
        .iter()
        .filter(|read| read.value == alias.source && read.elements.overlaps(alias.source_elements))
        .collect::<Vec<_>>();
    let overlapping_writes = operation
        .writes
        .iter()
        .filter(|write| {
            write.value == alias.destination && write.elements.overlaps(alias.destination_elements)
        })
        .collect::<Vec<_>>();
    let exact_read =
        overlapping_reads.len() == 1 && overlapping_reads[0].elements == alias.source_elements;
    let exact_write = overlapping_writes.len() == 1
        && overlapping_writes[0].elements == alias.destination_elements;
    if alias.source == alias.destination
        || alias.effect.0 == [0; 32]
        || alias.effect != operation.effect_identity
        || assignment.worker != storage.worker
        || source.worker != storage.worker
        || destination.worker != storage.worker
        || !exact_read
        || !exact_write
        || source_live.end != operation.during.end
        || destination_live.start != operation.during.start
        || has_concurrent_source_consumer(plan, alias, source, operation, assignments)?
    {
        return Err(FleetPlanError::InvalidInPlaceAlias(alias.operation));
    }
    Ok(())
}

fn has_concurrent_source_consumer(
    plan: &FleetProofPlan,
    alias: &InPlaceAlias,
    source: &StorageBinding,
    operation: &OperationDesc,
    assignments: &BTreeMap<OperationId, &OperationAssignment>,
) -> Result<bool, FleetPlanError> {
    let operation_consumer = plan.input.operations.iter().any(|other| {
        other.id != alias.operation
            && assignments
                .get(&other.id)
                .is_some_and(|assignment| assignment.worker == source.worker)
            && other.during.overlaps(operation.during)
            && other.reads.iter().any(|read| {
                read.value == alias.source && read.elements.overlaps(alias.source_elements)
            })
    });
    let transition_consumer = plan.input.transitions.iter().any(|transition| {
        transition.source_worker == source.worker
            && transition.value == alias.source
            && transition.elements.overlaps(alias.source_elements)
            && transition.during.overlaps(operation.during)
    });
    let mut spill_consumer = false;
    for spill in plan
        .input
        .spills
        .iter()
        .filter(|spill| spill.store.worker == source.worker)
    {
        for chunk in spill.chunks.iter().filter(|chunk| {
            chunk.value == alias.source && chunk.elements.overlaps(alias.source_elements)
        }) {
            let [d2h, _, _, _] = spill.chain(chunk.id)?;
            spill_consumer |= d2h.during.overlaps(operation.during);
        }
    }
    Ok(operation_consumer || transition_consumer || spill_consumer)
}

fn unique_binding(
    plan: &FleetProofPlan,
    storage: StorageId,
    value: ValueId,
    elements: ElementRange,
    offset_bytes: usize,
    bytes: usize,
) -> Result<&StorageBinding, FleetPlanError> {
    let mut matches = plan.input.storage_bindings.iter().filter(|binding| {
        binding.storage == storage
            && binding.value == value
            && binding.elements == elements
            && binding.offset_bytes == offset_bytes
            && binding.bytes == bytes
    });
    let first = matches
        .next()
        .ok_or(FleetPlanError::InvalidStorageBinding { value, storage })?;
    if matches.next().is_some() {
        return Err(FleetPlanError::InvalidStorageBinding { value, storage });
    }
    Ok(first)
}

fn alias_matches_pair(alias: &InPlaceAlias, left: &StorageBinding, right: &StorageBinding) -> bool {
    let matches = |source: &StorageBinding, destination: &StorageBinding| {
        alias.storage == source.storage
            && source.storage == destination.storage
            && alias.source == source.value
            && alias.source_elements == source.elements
            && alias.destination == destination.value
            && alias.destination_elements == destination.elements
            && alias.offset_bytes == source.offset_bytes
            && source.offset_bytes == destination.offset_bytes
            && alias.bytes == source.bytes
            && source.bytes == destination.bytes
    };
    matches(left, right) || matches(right, left)
}

#[derive(Clone, Copy)]
struct ByteRange {
    start: usize,
    end: usize,
}

impl ByteRange {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn byte_range(binding: &StorageBinding) -> Result<ByteRange, FleetPlanError> {
    Ok(ByteRange {
        start: binding.offset_bytes,
        end: binding
            .offset_bytes
            .checked_add(binding.bytes)
            .ok_or(FleetPlanError::SizeOverflow)?,
    })
}

fn matching_locations(plan: &FleetProofPlan, binding: &StorageBinding) -> usize {
    plan.input
        .owners
        .iter()
        .filter(|owner| binding_matches_owner(binding, owner))
        .count()
        + plan
            .input
            .replicas
            .iter()
            .filter(|replica| binding_matches_replica(binding, replica))
            .count()
}

fn binding_matches_owner(binding: &StorageBinding, owner: &OwnedValueRange) -> bool {
    binding.value == owner.value
        && binding.elements == owner.elements
        && binding.worker == owner.worker
}

fn binding_matches_replica(binding: &StorageBinding, replica: &DeclaredReplica) -> bool {
    binding.value == replica.value
        && binding.elements == replica.elements
        && binding.worker == replica.worker
}

fn binding_live(
    plan: &FleetProofPlan,
    binding: &StorageBinding,
) -> Result<ScheduleRange, FleetPlanError> {
    let owner = plan
        .input
        .owners
        .iter()
        .find(|owner| binding_matches_owner(binding, owner));
    let replica = plan
        .input
        .replicas
        .iter()
        .find(|replica| binding_matches_replica(binding, replica));
    match (owner, replica) {
        (Some(owner), None) => Ok(owner.live),
        (None, Some(replica)) => Ok(replica.live),
        _ => Err(FleetPlanError::InvalidStorageBinding {
            value: binding.value,
            storage: binding.storage,
        }),
    }
}

pub(super) fn reserved_bytes(
    plan: &FleetProofPlan,
    worker: WorkerId,
) -> Result<usize, FleetPlanError> {
    plan.input
        .storages
        .iter()
        .filter(|storage| storage.worker == worker)
        .try_fold(0usize, |total, storage| {
            total
                .checked_add(storage.bytes)
                .ok_or(FleetPlanError::SizeOverflow)
        })
}
