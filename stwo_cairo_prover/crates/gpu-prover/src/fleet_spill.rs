//! Address-free L1 host-spill contract for a cooperative proof plan.
//!
//! `HostSpillStore` is ordinary pageable/hugepage capacity. `DmaRing` is the
//! separately bounded pinned staging window. This module plans neither CUDA
//! allocations nor copies; it rejects schedules that a later runtime could not
//! execute without overwriting live staging data.

use std::collections::{BTreeMap, BTreeSet};

use crate::fleet_plan::{ElementRange, ExecutionInterval, ScheduleRange, ValueId, WorkerId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StoreExtentId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RingSlotId(pub u16);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SpillChunkId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SpillTransitionId(pub u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreExtent {
    pub id: StoreExtentId,
    pub offset_bytes: usize,
    pub len_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostSpillStore {
    pub worker: WorkerId,
    pub capacity_bytes: usize,
    pub alignment_bytes: usize,
    pub numa_node: u32,
    pub extents: Vec<StoreExtent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RingSlot {
    pub id: RingSlotId,
    pub offset_bytes: usize,
    pub len_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmaRing {
    pub worker: WorkerId,
    pub numa_node: u32,
    pub capacity_bytes: usize,
    pub memlock_limit_bytes: usize,
    pub alignment_bytes: usize,
    pub slots: Vec<RingSlot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpillChunk {
    pub id: SpillChunkId,
    pub value: ValueId,
    pub elements: ElementRange,
    pub worker: WorkerId,
    pub bytes: usize,
    pub store_extent: StoreExtentId,
    pub ring_slot: RingSlotId,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SpillTransitionKind {
    DeviceToRing,
    RingToStore,
    StoreToRing,
    RingToDevice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpillTransition {
    pub id: SpillTransitionId,
    pub chunk: SpillChunkId,
    pub kind: SpillTransitionKind,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
    pub bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpillPlan {
    pub store: HostSpillStore,
    pub ring: DmaRing,
    pub chunks: Vec<SpillChunk>,
    pub transitions: Vec<SpillTransition>,
}

impl SpillPlan {
    pub fn empty(worker: WorkerId) -> Self {
        Self {
            store: HostSpillStore {
                worker,
                capacity_bytes: 0,
                alignment_bytes: 1,
                numa_node: 0,
                extents: Vec::new(),
            },
            ring: DmaRing {
                worker,
                numa_node: 0,
                capacity_bytes: 0,
                memlock_limit_bytes: 0,
                alignment_bytes: 1,
                slots: Vec::new(),
            },
            chunks: Vec::new(),
            transitions: Vec::new(),
        }
    }

    pub(crate) fn canonicalize(&mut self) {
        self.store.extents.sort_unstable_by_key(|extent| extent.id);
        self.ring.slots.sort_unstable_by_key(|slot| slot.id);
        self.chunks.sort_unstable_by_key(|chunk| chunk.id);
        self.transitions
            .sort_unstable_by_key(|transition| (transition.chunk, transition.kind, transition.id));
    }

    pub fn validate(&self) -> Result<(), SpillPlanError> {
        if self.store.worker != self.ring.worker || self.store.numa_node != self.ring.numa_node {
            return Err(SpillPlanError::WrongOwner);
        }
        validate_store(&self.store)?;
        validate_ring(&self.ring)?;
        if self.chunks.is_empty() {
            return if self.transitions.is_empty()
                && self.store.extents.is_empty()
                && self.ring.slots.is_empty()
                && self.store.capacity_bytes == 0
                && self.ring.capacity_bytes == 0
                && self.ring.memlock_limit_bytes == 0
            {
                Ok(())
            } else {
                Err(SpillPlanError::OrphanedResource)
            };
        }

        let extents = unique_by(
            &self.store.extents,
            |extent| extent.id,
            SpillPlanError::DuplicateStoreExtent,
        )?;
        let slots = unique_by(
            &self.ring.slots,
            |slot| slot.id,
            SpillPlanError::DuplicateRingSlot,
        )?;
        let chunks = unique_by(
            &self.chunks,
            |chunk| chunk.id,
            SpillPlanError::DuplicateChunk,
        )?;
        let mut claimed_extents = BTreeSet::new();
        let mut claimed_slots = BTreeSet::new();
        for chunk in &self.chunks {
            if chunk.worker != self.store.worker {
                return Err(SpillPlanError::WrongOwner);
            }
            if chunk.bytes == 0 || chunk.elements.is_empty() {
                return Err(SpillPlanError::InvalidChunk(chunk.id));
            }
            let extent = extents
                .get(&chunk.store_extent)
                .ok_or(SpillPlanError::MissingStoreExtent(chunk.store_extent))?;
            let slot = slots
                .get(&chunk.ring_slot)
                .ok_or(SpillPlanError::MissingRingSlot(chunk.ring_slot))?;
            if extent.len_bytes < chunk.bytes || slot.len_bytes < chunk.bytes {
                return Err(SpillPlanError::ChunkCapacity(chunk.id));
            }
            claimed_extents.insert(chunk.store_extent);
            claimed_slots.insert(chunk.ring_slot);
        }
        if claimed_extents.len() != extents.len() || claimed_slots.len() != slots.len() {
            return Err(SpillPlanError::OrphanedResource);
        }

        let mut transition_ids = BTreeSet::new();
        let mut by_chunk = BTreeMap::<SpillChunkId, Vec<&SpillTransition>>::new();
        for transition in &self.transitions {
            if !transition_ids.insert(transition.id) {
                return Err(SpillPlanError::DuplicateTransition(transition.id));
            }
            if !chunks.contains_key(&transition.chunk) {
                return Err(SpillPlanError::MissingChunk(transition.chunk));
            }
            if !transition.during.is_valid() {
                return Err(SpillPlanError::InvalidSchedule);
            }
            by_chunk
                .entry(transition.chunk)
                .or_default()
                .push(transition);
        }

        let mut slot_windows = BTreeMap::<RingSlotId, Vec<ScheduleRange>>::new();
        let mut extent_windows = BTreeMap::<StoreExtentId, Vec<ScheduleRange>>::new();
        for chunk in &self.chunks {
            let stages = by_chunk
                .get(&chunk.id)
                .ok_or(SpillPlanError::IncompleteChain(chunk.id))?;
            let [d2h, retire, prefetch, h2d] = exact_stages(chunk, stages)?;
            if d2h.during.end > retire.during.start
                || retire.during.end > prefetch.during.start
                || prefetch.during.end > h2d.during.start
            {
                return Err(SpillPlanError::OutOfOrderChain(chunk.id));
            }
            slot_windows.entry(chunk.ring_slot).or_default().extend([
                ScheduleRange::new(d2h.during.start, retire.during.end)
                    .ok_or(SpillPlanError::InvalidSchedule)?,
                ScheduleRange::new(prefetch.during.start, h2d.during.end)
                    .ok_or(SpillPlanError::InvalidSchedule)?,
            ]);
            extent_windows.entry(chunk.store_extent).or_default().push(
                ScheduleRange::new(retire.during.start, prefetch.during.end)
                    .ok_or(SpillPlanError::OutOfOrderChain(chunk.id))?,
            );
        }
        if by_chunk.len() != self.chunks.len() {
            return Err(SpillPlanError::OrphanedResource);
        }
        for windows in slot_windows.values_mut() {
            windows.sort_unstable_by_key(|window| (window.start, window.end));
            if windows.windows(2).any(|pair| pair[0].overlaps(pair[1])) {
                return Err(SpillPlanError::RingSlotOverlap);
            }
        }
        for windows in extent_windows.values_mut() {
            windows.sort_unstable_by_key(|window| (window.start, window.end));
            if windows.windows(2).any(|pair| pair[0].overlaps(pair[1])) {
                return Err(SpillPlanError::StoreExtentOverlap);
            }
        }
        Ok(())
    }

    pub(crate) fn chain(
        &self,
        chunk: SpillChunkId,
    ) -> Result<[&SpillTransition; 4], SpillPlanError> {
        let stages = self
            .transitions
            .iter()
            .filter(|transition| transition.chunk == chunk)
            .collect::<Vec<_>>();
        let chunk = self
            .chunks
            .iter()
            .find(|candidate| candidate.id == chunk)
            .ok_or(SpillPlanError::MissingChunk(chunk))?;
        exact_stages(chunk, &stages)
    }
}

fn validate_store(store: &HostSpillStore) -> Result<(), SpillPlanError> {
    validate_alignment(store.alignment_bytes)?;
    validate_regions(
        store.capacity_bytes,
        store.alignment_bytes,
        store
            .extents
            .iter()
            .map(|extent| (extent.offset_bytes, extent.len_bytes)),
    )
}

fn validate_ring(ring: &DmaRing) -> Result<(), SpillPlanError> {
    validate_alignment(ring.alignment_bytes)?;
    if ring.capacity_bytes > ring.memlock_limit_bytes {
        return Err(SpillPlanError::MemlockExceeded);
    }
    validate_regions(
        ring.capacity_bytes,
        ring.alignment_bytes,
        ring.slots
            .iter()
            .map(|slot| (slot.offset_bytes, slot.len_bytes)),
    )
}

fn validate_alignment(alignment: usize) -> Result<(), SpillPlanError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(SpillPlanError::InvalidAlignment(alignment));
    }
    Ok(())
}

fn validate_regions(
    capacity: usize,
    alignment: usize,
    regions: impl Iterator<Item = (usize, usize)>,
) -> Result<(), SpillPlanError> {
    let mut regions = regions.collect::<Vec<_>>();
    regions.sort_unstable();
    let mut previous_end = 0usize;
    for (index, &(offset, len)) in regions.iter().enumerate() {
        if len == 0 || offset % alignment != 0 {
            return Err(SpillPlanError::InvalidRegion);
        }
        let end = offset
            .checked_add(len)
            .ok_or(SpillPlanError::SizeOverflow)?;
        if end > capacity || index > 0 && previous_end > offset {
            return Err(SpillPlanError::InvalidRegion);
        }
        previous_end = end;
    }
    Ok(())
}

fn unique_by<'a, T, K: Ord + Copy>(
    values: &'a [T],
    key: impl Fn(&T) -> K,
    duplicate: impl Fn(K) -> SpillPlanError,
) -> Result<BTreeMap<K, &'a T>, SpillPlanError> {
    let mut indexed = BTreeMap::new();
    for value in values {
        let key = key(value);
        if indexed.insert(key, value).is_some() {
            return Err(duplicate(key));
        }
    }
    Ok(indexed)
}

fn exact_stages<'a>(
    chunk: &SpillChunk,
    stages: &[&'a SpillTransition],
) -> Result<[&'a SpillTransition; 4], SpillPlanError> {
    if stages.len() != 4 {
        return Err(SpillPlanError::IncompleteChain(chunk.id));
    }
    let mut exact = [None, None, None, None];
    for stage in stages {
        if stage.bytes != chunk.bytes {
            return Err(SpillPlanError::TransitionBytes(stage.id));
        }
        let index = match stage.kind {
            SpillTransitionKind::DeviceToRing => 0,
            SpillTransitionKind::RingToStore => 1,
            SpillTransitionKind::StoreToRing => 2,
            SpillTransitionKind::RingToDevice => 3,
        };
        if exact[index].replace(*stage).is_some() {
            return Err(SpillPlanError::IncompleteChain(chunk.id));
        }
    }
    exact
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.try_into().ok())
        .ok_or(SpillPlanError::IncompleteChain(chunk.id))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpillPlanError {
    InvalidAlignment(usize),
    InvalidRegion,
    SizeOverflow,
    MemlockExceeded,
    DuplicateStoreExtent(StoreExtentId),
    DuplicateRingSlot(RingSlotId),
    DuplicateChunk(SpillChunkId),
    DuplicateTransition(SpillTransitionId),
    MissingStoreExtent(StoreExtentId),
    MissingRingSlot(RingSlotId),
    MissingChunk(SpillChunkId),
    InvalidChunk(SpillChunkId),
    ChunkCapacity(SpillChunkId),
    IncompleteChain(SpillChunkId),
    OutOfOrderChain(SpillChunkId),
    TransitionBytes(SpillTransitionId),
    RingSlotOverlap,
    StoreExtentOverlap,
    OrphanedResource,
    WrongOwner,
    InvalidSchedule,
}

impl core::fmt::Display for SpillPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet spill plan: {self:?}")
    }
}

impl std::error::Error for SpillPlanError {}
