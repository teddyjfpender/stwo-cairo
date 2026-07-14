//! Deterministic, checked range packing for one stable proof arena.
//!
//! Unlike `arena_plan`'s whole-slot colorer, this planner binds every logical
//! value to an offset and permits spatial overlap only when the values' live
//! epoch masks are disjoint. It is planner-only: the CUDA arena gains offset
//! binding in the later runtime vertical slice.

use std::collections::{BTreeMap, BTreeSet};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RangeId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AliasGroupId(pub u32);

/// One logical word range. A bit set in `live_mask` means the range is live
/// for that inclusive proof epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeRequest {
    pub id: RangeId,
    pub len_words: usize,
    pub alignment_words: usize,
    pub live_mask: u16,
    /// Members share an exact offset. Their masks must be pairwise disjoint and
    /// their lengths and alignments identical.
    pub must_alias: Option<AliasGroupId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeBinding {
    pub id: RangeId,
    pub offset_words: usize,
    pub len_words: usize,
}

impl RangeBinding {
    pub fn offset_bytes(self) -> Result<usize, RangeAllocationError> {
        self.offset_words
            .checked_mul(WORD_BYTES)
            .ok_or(RangeAllocationError::SizeOverflow)
    }

    pub fn len_bytes(self) -> Result<usize, RangeAllocationError> {
        self.len_words
            .checked_mul(WORD_BYTES)
            .ok_or(RangeAllocationError::SizeOverflow)
    }
}

/// A stable slab layout. `raw_peak_words` is a hard arena-word lower bound.
/// `excess_over_raw_peak_words` is the allocation's exact distance above that
/// weak bound; it includes required alignment, placement constraints, and any
/// heuristic fragmentation, so it is neither an optimality gap nor a claim of
/// reclaimable memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RangeLayout {
    total_words: usize,
    raw_peak_words: usize,
    excess_over_raw_peak_words: usize,
    /// Canonical `RangeId` order, independent of input order.
    bindings: Vec<RangeBinding>,
}

impl RangeLayout {
    pub const fn total_words(&self) -> usize {
        self.total_words
    }

    pub const fn raw_peak_words(&self) -> usize {
        self.raw_peak_words
    }

    pub const fn excess_over_raw_peak_words(&self) -> usize {
        self.excess_over_raw_peak_words
    }

    pub fn bindings(&self) -> &[RangeBinding] {
        &self.bindings
    }

    pub fn binding(&self, id: RangeId) -> Option<RangeBinding> {
        self.bindings
            .binary_search_by_key(&id, |binding| binding.id)
            .ok()
            .map(|index| self.bindings[index])
    }

    pub fn total_bytes(&self) -> Result<usize, RangeAllocationError> {
        self.total_words
            .checked_mul(WORD_BYTES)
            .ok_or(RangeAllocationError::SizeOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RangeAllocationError {
    EmptyPlan,
    DuplicateRange(RangeId),
    EmptyRange(RangeId),
    EmptyLiveMask(RangeId),
    InvalidAlignment {
        id: Option<RangeId>,
        alignment: usize,
    },
    AliasShapeMismatch {
        group: AliasGroupId,
        id: RangeId,
    },
    AliasLifetimeOverlap {
        group: AliasGroupId,
        first: RangeId,
        second: RangeId,
    },
    MissingBinding(RangeId),
    UnexpectedBinding(RangeId),
    DuplicateBinding(RangeId),
    NonCanonicalBindingOrder,
    BindingLengthMismatch {
        id: RangeId,
        expected: usize,
        actual: usize,
    },
    MisalignedBinding(RangeId),
    OutOfBounds(RangeId),
    AliasOffsetMismatch {
        group: AliasGroupId,
        id: RangeId,
    },
    LiveRangeOverlap {
        first: RangeId,
        second: RangeId,
    },
    CapacityExceeded {
        required_words: usize,
        capacity_words: usize,
    },
    LayoutMetadataMismatch,
    SizeOverflow,
}

impl core::fmt::Display for RangeAllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid proof arena range allocation: {self:?}")
    }
}

impl std::error::Error for RangeAllocationError {}

#[derive(Debug)]
struct PlacementUnit {
    members: Vec<usize>,
    id: RangeId,
    len_words: usize,
    alignment_words: usize,
    live_mask: u16,
}

/// Pack requests at the lowest aligned offsets admitted by a deterministic
/// size-first order. `capacity_words` is an optional hard admission bound.
pub fn allocate_ranges(
    requests: &[RangeRequest],
    slab_alignment_words: usize,
    capacity_words: Option<usize>,
) -> Result<RangeLayout, RangeAllocationError> {
    validate_requests(requests, slab_alignment_words)?;
    let raw_peak_words = raw_peak_words(requests)?;
    let mut units = collapse_alias_groups(requests)?;
    units.sort_unstable_by_key(|unit| {
        (
            core::cmp::Reverse(unit.len_words),
            core::cmp::Reverse(unit.live_mask.count_ones()),
            unit.live_mask.trailing_zeros(),
            unit.id,
        )
    });

    let mut occupied: [BTreeMap<usize, (usize, RangeId)>; u16::BITS as usize] =
        std::array::from_fn(|_| BTreeMap::new());
    let mut bindings = Vec::with_capacity(requests.len());
    let mut max_end = 0usize;
    for unit in units {
        let offset_words = lowest_available_offset(&unit, &occupied)?;
        let end_words = offset_words
            .checked_add(unit.len_words)
            .ok_or(RangeAllocationError::SizeOverflow)?;
        max_end = max_end.max(end_words);
        for bit in live_bits(unit.live_mask) {
            let previous = occupied[bit].insert(offset_words, (end_words, unit.id));
            debug_assert!(previous.is_none(), "validated placement has a unique start");
        }
        bindings.extend(unit.members.into_iter().map(|index| RangeBinding {
            id: requests[index].id,
            offset_words,
            len_words: requests[index].len_words,
        }));
    }

    let total_words = align_up(max_end, slab_alignment_words)?;
    if let Some(capacity_words) = capacity_words {
        if total_words > capacity_words {
            return Err(RangeAllocationError::CapacityExceeded {
                required_words: total_words,
                capacity_words,
            });
        }
    }
    bindings.sort_unstable_by_key(|binding| binding.id);
    let layout = RangeLayout {
        total_words,
        raw_peak_words,
        excess_over_raw_peak_words: total_words
            .checked_sub(raw_peak_words)
            .ok_or(RangeAllocationError::SizeOverflow)?,
        bindings,
    };
    validate_range_layout(requests, slab_alignment_words, capacity_words, &layout)?;
    Ok(layout)
}

/// Independently prove a proposed layout: exact membership and lengths,
/// alignment/bounds, required aliases, and no spatial overlap at any live bit.
pub fn validate_range_layout(
    requests: &[RangeRequest],
    slab_alignment_words: usize,
    capacity_words: Option<usize>,
    layout: &RangeLayout,
) -> Result<(), RangeAllocationError> {
    validate_requests(requests, slab_alignment_words)?;
    collapse_alias_groups(requests)?;
    if let Some(capacity_words) = capacity_words {
        if layout.total_words > capacity_words {
            return Err(RangeAllocationError::CapacityExceeded {
                required_words: layout.total_words,
                capacity_words,
            });
        }
    }
    if layout.total_words % slab_alignment_words != 0 {
        return Err(RangeAllocationError::InvalidAlignment {
            id: None,
            alignment: slab_alignment_words,
        });
    }

    let requests_by_id = requests
        .iter()
        .map(|request| (request.id, request))
        .collect::<BTreeMap<_, _>>();
    let mut bindings_by_id = BTreeMap::new();
    let mut max_end = 0usize;
    for binding in &layout.bindings {
        if bindings_by_id.insert(binding.id, *binding).is_some() {
            return Err(RangeAllocationError::DuplicateBinding(binding.id));
        }
        let request = requests_by_id
            .get(&binding.id)
            .ok_or(RangeAllocationError::UnexpectedBinding(binding.id))?;
        if binding.len_words != request.len_words {
            return Err(RangeAllocationError::BindingLengthMismatch {
                id: binding.id,
                expected: request.len_words,
                actual: binding.len_words,
            });
        }
        if binding.offset_words % request.alignment_words != 0 {
            return Err(RangeAllocationError::MisalignedBinding(binding.id));
        }
        let end = binding
            .offset_words
            .checked_add(binding.len_words)
            .ok_or(RangeAllocationError::SizeOverflow)?;
        if end > layout.total_words {
            return Err(RangeAllocationError::OutOfBounds(binding.id));
        }
        max_end = max_end.max(end);
    }
    if layout
        .bindings
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
    {
        return Err(RangeAllocationError::NonCanonicalBindingOrder);
    }
    for request in requests {
        if !bindings_by_id.contains_key(&request.id) {
            return Err(RangeAllocationError::MissingBinding(request.id));
        }
    }
    if bindings_by_id.len() != requests.len()
        || align_up(max_end, slab_alignment_words)? != layout.total_words
    {
        return Err(RangeAllocationError::LayoutMetadataMismatch);
    }

    validate_alias_offsets(requests, &bindings_by_id)?;
    validate_live_ranges(requests, &bindings_by_id)?;
    let raw_peak = raw_peak_words(requests)?;
    if layout.raw_peak_words != raw_peak
        || layout.total_words < raw_peak
        || layout.excess_over_raw_peak_words != layout.total_words - raw_peak
    {
        return Err(RangeAllocationError::LayoutMetadataMismatch);
    }
    Ok(())
}

pub fn raw_peak_words(requests: &[RangeRequest]) -> Result<usize, RangeAllocationError> {
    let mut peak = 0usize;
    for bit in 0..u16::BITS {
        let live_bit = 1u16 << bit;
        let words = requests
            .iter()
            .filter(|request| request.live_mask & live_bit != 0)
            .try_fold(0usize, |total, request| {
                total
                    .checked_add(request.len_words)
                    .ok_or(RangeAllocationError::SizeOverflow)
            })?;
        peak = peak.max(words);
    }
    Ok(peak)
}

fn validate_requests(
    requests: &[RangeRequest],
    slab_alignment_words: usize,
) -> Result<(), RangeAllocationError> {
    if requests.is_empty() {
        return Err(RangeAllocationError::EmptyPlan);
    }
    if !slab_alignment_words.is_power_of_two() {
        return Err(RangeAllocationError::InvalidAlignment {
            id: None,
            alignment: slab_alignment_words,
        });
    }
    let mut ids = BTreeSet::new();
    for request in requests {
        if !ids.insert(request.id) {
            return Err(RangeAllocationError::DuplicateRange(request.id));
        }
        if request.len_words == 0 {
            return Err(RangeAllocationError::EmptyRange(request.id));
        }
        if request.live_mask == 0 {
            return Err(RangeAllocationError::EmptyLiveMask(request.id));
        }
        if !request.alignment_words.is_power_of_two() {
            return Err(RangeAllocationError::InvalidAlignment {
                id: Some(request.id),
                alignment: request.alignment_words,
            });
        }
    }
    Ok(())
}

fn collapse_alias_groups(
    requests: &[RangeRequest],
) -> Result<Vec<PlacementUnit>, RangeAllocationError> {
    let mut groups = BTreeMap::<AliasGroupId, Vec<usize>>::new();
    let mut units = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        match request.must_alias {
            Some(group) => groups.entry(group).or_default().push(index),
            None => units.push(PlacementUnit {
                members: vec![index],
                id: request.id,
                len_words: request.len_words,
                alignment_words: request.alignment_words,
                live_mask: request.live_mask,
            }),
        }
    }
    for (group, members) in groups {
        let first = requests[members[0]];
        let mut live_mask = 0u16;
        for &index in &members {
            let request = requests[index];
            if request.len_words != first.len_words
                || request.alignment_words != first.alignment_words
            {
                return Err(RangeAllocationError::AliasShapeMismatch {
                    group,
                    id: request.id,
                });
            }
            if live_mask & request.live_mask != 0 {
                let prior = members
                    .iter()
                    .copied()
                    .find(|&prior| {
                        requests[prior].id != request.id
                            && requests[prior].live_mask & request.live_mask != 0
                    })
                    .expect("overlapping alias mask has a prior owner");
                return Err(RangeAllocationError::AliasLifetimeOverlap {
                    group,
                    first: requests[prior].id,
                    second: request.id,
                });
            }
            live_mask |= request.live_mask;
        }
        units.push(PlacementUnit {
            id: members
                .iter()
                .map(|&index| requests[index].id)
                .min()
                .expect("alias group is non-empty"),
            members,
            len_words: first.len_words,
            alignment_words: first.alignment_words,
            live_mask,
        });
    }
    Ok(units)
}

fn lowest_available_offset(
    unit: &PlacementUnit,
    occupied: &[BTreeMap<usize, (usize, RangeId)>; u16::BITS as usize],
) -> Result<usize, RangeAllocationError> {
    let mut cursor = 0usize;
    loop {
        let candidate = align_up(cursor, unit.alignment_words)?;
        let candidate_end = candidate
            .checked_add(unit.len_words)
            .ok_or(RangeAllocationError::SizeOverflow)?;
        let conflict_end = live_bits(unit.live_mask)
            .filter_map(|bit| {
                occupied[bit]
                    .range(..candidate_end)
                    .next_back()
                    .and_then(|(_, &(end, _))| (end > candidate).then_some(end))
            })
            .max();
        match conflict_end {
            Some(end) => cursor = end,
            None => return Ok(candidate),
        }
    }
}

fn live_bits(mask: u16) -> impl Iterator<Item = usize> {
    (0..u16::BITS as usize).filter(move |&bit| mask & (1u16 << bit) != 0)
}

fn validate_alias_offsets(
    requests: &[RangeRequest],
    bindings: &BTreeMap<RangeId, RangeBinding>,
) -> Result<(), RangeAllocationError> {
    let mut offsets = BTreeMap::<AliasGroupId, (RangeId, usize)>::new();
    for request in requests {
        let Some(group) = request.must_alias else {
            continue;
        };
        let offset = bindings[&request.id].offset_words;
        match offsets.insert(group, (request.id, offset)) {
            Some((_, expected)) if expected != offset => {
                return Err(RangeAllocationError::AliasOffsetMismatch {
                    group,
                    id: request.id,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_live_ranges(
    requests: &[RangeRequest],
    bindings: &BTreeMap<RangeId, RangeBinding>,
) -> Result<(), RangeAllocationError> {
    for bit in 0..u16::BITS {
        let live_bit = 1u16 << bit;
        let mut ranges = requests
            .iter()
            .filter(|request| request.live_mask & live_bit != 0)
            .map(|request| {
                let binding = bindings[&request.id];
                (
                    binding.offset_words,
                    binding.offset_words + binding.len_words,
                    request.id,
                )
            })
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|&(start, end, id)| (start, end, id));
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(RangeAllocationError::LiveRangeOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
    }
    Ok(())
}

pub const fn live_masks_conflict(first: u16, second: u16) -> bool {
    first & second != 0
}

fn align_up(value: usize, alignment: usize) -> Result<usize, RangeAllocationError> {
    if !alignment.is_power_of_two() {
        return Err(RangeAllocationError::InvalidAlignment {
            id: None,
            alignment,
        });
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(RangeAllocationError::SizeOverflow)
}

#[cfg(test)]
#[path = "range_allocator_tests.rs"]
mod tests;
