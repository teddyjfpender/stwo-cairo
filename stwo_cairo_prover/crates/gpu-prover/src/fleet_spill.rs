//! Address-free L1 host-spill contract for a cooperative proof plan.
//!
//! `HostSpillStore` is ordinary pageable/hugepage capacity. `DmaRing` is the
//! separately bounded pinned staging window. This module plans neither CUDA
//! allocations nor copies. Optional VMM records describe only sound whole-
//! storage reclaim windows; a later runtime must attest and execute them.

use std::collections::{BTreeMap, BTreeSet};

use crate::compiled_proof::ValueRange;
use crate::fleet_plan::{ExecutionInterval, ScheduleRange, StorageId, WorkerId};

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
    /// Maximum bytes transferred per DMA tile. A spill chunk may be larger;
    /// its transition windows cover the contiguous tile sequence through this
    /// bounded pinned slot.
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
    pub value: ValueRange,
    pub worker: WorkerId,
    /// One exact contiguous worker-local device allocation binding. A chunk
    /// may not straddle or reorder several storages.
    pub storage: StorageId,
    pub store_extent: StoreExtentId,
    /// Exact logical bytes covered by `value`. Fleet admission re-derives and
    /// checks this from the compiled value type before runtime installation.
    pub len_bytes: usize,
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
    /// Dense tile ordinal within the chunk.
    pub tile_ordinal: u32,
    /// Offset in the chunk's exact device binding and host-store extent. The
    /// pinned destination/source starts at `RingSlot::offset_bytes`.
    pub chunk_offset_bytes: usize,
    pub len_bytes: usize,
    pub ring_slot: RingSlotId,
    pub kind: SpillTransitionKind,
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmmTransition {
    pub interval: ExecutionInterval,
    pub during: ScheduleRange,
}

/// Whole-allocation physical reclaim while the canonical bytes live in host
/// spill storage. The allocation starts mapped; stable virtual addressing is
/// retained across this one generation-1 remap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmmReclaim {
    pub chunk: SpillChunkId,
    pub storage: StorageId,
    pub allocation_granularity_bytes: usize,
    pub unmap: VmmTransition,
    pub remap: VmmTransition,
    pub remap_generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpillPlan {
    pub store: HostSpillStore,
    pub ring: DmaRing,
    pub chunks: Vec<SpillChunk>,
    pub transitions: Vec<SpillTransition>,
    pub vmm_reclaims: Vec<VmmReclaim>,
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
            vmm_reclaims: Vec::new(),
        }
    }

    pub(crate) fn canonicalize(&mut self) {
        self.store.extents.sort_unstable_by_key(|extent| extent.id);
        self.ring.slots.sort_unstable_by_key(|slot| slot.id);
        self.chunks.sort_unstable_by_key(|chunk| chunk.id);
        self.transitions.sort_unstable_by_key(|transition| {
            (
                transition.during.start,
                transition.during.end,
                transition.chunk,
                transition.tile_ordinal,
                transition.kind,
                transition.id,
            )
        });
        self.vmm_reclaims
            .sort_unstable_by_key(|reclaim| (reclaim.storage, reclaim.chunk));
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
                && self.vmm_reclaims.is_empty()
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
            if chunk.value.elements.is_empty() || chunk.len_bytes == 0 {
                return Err(SpillPlanError::InvalidChunk(chunk.id));
            }
            let extent = extents
                .get(&chunk.store_extent)
                .ok_or(SpillPlanError::MissingStoreExtent(chunk.store_extent))?;
            if extent.len_bytes < chunk.len_bytes {
                return Err(SpillPlanError::InvalidChunk(chunk.id));
            }
            claimed_extents.insert(chunk.store_extent);
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
            let chunk = chunks[&transition.chunk];
            let end = transition
                .chunk_offset_bytes
                .checked_add(transition.len_bytes)
                .ok_or(SpillPlanError::SizeOverflow)?;
            let slot = slots
                .get(&transition.ring_slot)
                .ok_or(SpillPlanError::MissingRingSlot(transition.ring_slot))?;
            if !transition.during.is_valid()
                || transition.len_bytes == 0
                || end > chunk.len_bytes
                || transition.len_bytes > slot.len_bytes
            {
                return Err(SpillPlanError::InvalidSchedule);
            }
            claimed_slots.insert(transition.ring_slot);
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
            let tiles = exact_tiles(chunk, stages)?;
            for [d2h, retire, prefetch, h2d] in &tiles {
                if d2h.during.end > retire.during.start || prefetch.during.end > h2d.during.start {
                    return Err(SpillPlanError::OutOfOrderChain(chunk.id));
                }
                slot_windows.entry(d2h.ring_slot).or_default().extend([
                    ScheduleRange::new(d2h.during.start, retire.during.end)
                        .ok_or(SpillPlanError::InvalidSchedule)?,
                    ScheduleRange::new(prefetch.during.start, h2d.during.end)
                        .ok_or(SpillPlanError::InvalidSchedule)?,
                ]);
            }
            let [_, last_retire, first_prefetch, _] = tile_bounds(&tiles);
            if last_retire.during.end > first_prefetch.during.start {
                return Err(SpillPlanError::OutOfOrderChain(chunk.id));
            }
            let first_retire = tiles[0][1];
            let last_prefetch = tiles[tiles.len() - 1][2];
            extent_windows.entry(chunk.store_extent).or_default().push(
                ScheduleRange::new(first_retire.during.start, last_prefetch.during.end)
                    .ok_or(SpillPlanError::OutOfOrderChain(chunk.id))?,
            );
        }
        if by_chunk.len() != self.chunks.len()
            || claimed_extents.len() != extents.len()
            || claimed_slots.len() != slots.len()
        {
            return Err(SpillPlanError::OrphanedResource);
        }
        validate_vmm_reclaims(&self.vmm_reclaims, &chunks)?;
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

    pub(crate) fn bounds(
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
        let tiles = exact_tiles(chunk, &stages)?;
        Ok(tile_bounds(&tiles))
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

fn exact_tiles<'a>(
    chunk: &SpillChunk,
    stages: &[&'a SpillTransition],
) -> Result<Vec<[&'a SpillTransition; 4]>, SpillPlanError> {
    let mut grouped = BTreeMap::<u32, Vec<&SpillTransition>>::new();
    for stage in stages {
        grouped.entry(stage.tile_ordinal).or_default().push(stage);
    }
    let mut cursor = 0usize;
    let mut previous = None::<[ScheduleRange; 4]>;
    let mut tiles = Vec::with_capacity(grouped.len());
    for (expected, (ordinal, tile_stages)) in grouped.into_iter().enumerate() {
        if ordinal as usize != expected {
            return Err(SpillPlanError::IncompleteChain(chunk.id));
        }
        let tile = exact_stages(chunk, &tile_stages)?;
        let first = tile[0];
        if first.chunk_offset_bytes != cursor
            || tile.iter().any(|stage| {
                stage.tile_ordinal != ordinal
                    || stage.chunk_offset_bytes != first.chunk_offset_bytes
                    || stage.len_bytes != first.len_bytes
                    || stage.ring_slot != first.ring_slot
            })
        {
            return Err(SpillPlanError::IncompleteChain(chunk.id));
        }
        let end = cursor
            .checked_add(first.len_bytes)
            .ok_or(SpillPlanError::SizeOverflow)?;
        if end > chunk.len_bytes {
            return Err(SpillPlanError::InvalidChunk(chunk.id));
        }
        let ranges = tile.map(|stage| stage.during);
        if previous.is_some_and(|previous| {
            previous
                .into_iter()
                .zip(ranges.iter().copied())
                .any(|(left, right)| left.start > right.start || left.end > right.end)
        }) {
            return Err(SpillPlanError::OutOfOrderChain(chunk.id));
        }
        previous = Some(ranges);
        cursor = end;
        tiles.push(tile);
    }
    if tiles.is_empty() || cursor != chunk.len_bytes {
        return Err(SpillPlanError::IncompleteChain(chunk.id));
    }
    Ok(tiles)
}

fn tile_bounds<'a>(tiles: &[[&'a SpillTransition; 4]]) -> [&'a SpillTransition; 4] {
    [
        tiles[0][0],
        tiles[tiles.len() - 1][1],
        tiles[0][2],
        tiles[tiles.len() - 1][3],
    ]
}

fn validate_vmm_reclaims(
    reclaims: &[VmmReclaim],
    chunks: &BTreeMap<SpillChunkId, &SpillChunk>,
) -> Result<(), SpillPlanError> {
    let mut reclaimed_chunks = BTreeSet::new();
    let mut reclaimed_storages = BTreeSet::new();
    for reclaim in reclaims {
        if !chunks.contains_key(&reclaim.chunk) {
            return Err(SpillPlanError::MissingChunk(reclaim.chunk));
        }
        if !reclaimed_chunks.insert(reclaim.chunk) || !reclaimed_storages.insert(reclaim.storage) {
            return Err(SpillPlanError::DuplicateVmmReclaim(reclaim.chunk));
        }
        if reclaim.allocation_granularity_bytes == 0
            || !reclaim.allocation_granularity_bytes.is_power_of_two()
            || reclaim.remap_generation != 1
            || !reclaim.unmap.during.is_valid()
            || !reclaim.remap.during.is_valid()
            || reclaim.unmap.during.end > reclaim.remap.during.start
        {
            return Err(SpillPlanError::InvalidVmmReclaim(reclaim.chunk));
        }
    }
    Ok(())
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
    IncompleteChain(SpillChunkId),
    OutOfOrderChain(SpillChunkId),
    DuplicateVmmReclaim(SpillChunkId),
    InvalidVmmReclaim(SpillChunkId),
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
