//! Deterministic, address-free transfer view for an already validated fleet plan.
//!
//! This is not runtime admission. It resolves logical transition ranges into
//! exact worker-local storage byte windows and charges dedicated IPC exchange
//! allocations without installing a pointer, context, module or transport.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetStorageWindow {
    pub storage: StorageId,
    pub worker: WorkerId,
    pub offset_bytes: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetTransferSpan {
    pub edge_ordinal: u64,
    pub transition: LayoutTransitionId,
    pub span_ordinal: u32,
    pub route: FleetLinkId,
    pub owner: WorkerId,
    pub peer: WorkerId,
    pub elements: ElementRange,
    pub during: ScheduleRange,
    pub source: FleetStorageWindow,
    pub destination: FleetStorageWindow,
}

impl FleetTransferSpan {
    pub const fn logical_bytes(self) -> usize {
        self.source.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetExchangeReserve {
    pub worker: WorkerId,
    pub required_bytes: usize,
    pub declared_bytes: usize,
}

/// Canonical host-only projection of the plan's L2 byte transfers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetRuntimeView {
    plan_identity: [u8; 32],
    spans: Vec<FleetTransferSpan>,
    exchange_reserves: Vec<FleetExchangeReserve>,
}

impl FleetRuntimeView {
    pub const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub fn spans(&self) -> &[FleetTransferSpan] {
        &self.spans
    }

    pub fn exchange_reserves(&self) -> &[FleetExchangeReserve] {
        &self.exchange_reserves
    }
}

impl FleetProofPlan {
    /// Resolve exact transfer spans and validate the plan's rounded owner-side
    /// exchange reserve. The plan remains address-free and non-admitted.
    pub fn runtime_view(&self) -> Result<FleetRuntimeView, FleetRuntimeViewError> {
        let mut transitions = self.placement().transitions.iter().collect::<Vec<_>>();
        transitions.sort_unstable_by_key(|transition| transition.id);
        if let Some(pair) = transitions.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(FleetRuntimeViewError::DuplicateTransition(pair[0].id));
        }

        let mut spans = Vec::new();
        for transition in transitions {
            derive_transition_spans(self, transition, &mut spans)?;
        }
        for (ordinal, span) in spans.iter_mut().enumerate() {
            span.edge_ordinal =
                u64::try_from(ordinal).map_err(|_| FleetRuntimeViewError::SizeOverflow)?;
        }

        let mut required = BTreeMap::<WorkerId, usize>::new();
        for span in &spans {
            let allocation = rounded_exchange_bytes(span.logical_bytes())?;
            let total = required.entry(span.owner).or_default();
            *total = total
                .checked_add(allocation)
                .ok_or(FleetRuntimeViewError::SizeOverflow)?;
        }

        let mut workers = self.placement().topology.workers.iter().collect::<Vec<_>>();
        workers.sort_unstable_by_key(|worker| worker.id);
        let mut seen = BTreeSet::new();
        let mut exchange_reserves = Vec::with_capacity(workers.len());
        for worker in workers {
            if !seen.insert(worker.id) {
                return Err(FleetRuntimeViewError::DuplicateWorker(worker.id));
            }
            let required_bytes = required.remove(&worker.id).unwrap_or(0);
            if required_bytes > worker.exchange_reserve_bytes {
                return Err(FleetRuntimeViewError::ExchangeReserveExceeded {
                    worker: worker.id,
                    required: required_bytes,
                    declared: worker.exchange_reserve_bytes,
                });
            }
            exchange_reserves.push(FleetExchangeReserve {
                worker: worker.id,
                required_bytes,
                declared_bytes: worker.exchange_reserve_bytes,
            });
        }
        if let Some((&worker, _)) = required.first_key_value() {
            return Err(FleetRuntimeViewError::UnknownWorker(worker));
        }

        Ok(FleetRuntimeView {
            plan_identity: self.identity(),
            spans,
            exchange_reserves,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetRuntimeViewError {
    DuplicateTransition(LayoutTransitionId),
    InvalidTransition(LayoutTransitionId),
    InvalidReplica(ReplicaId),
    UnknownValue(ValueVersion),
    UnknownStorage(StorageId),
    UnknownWorker(WorkerId),
    DuplicateWorker(WorkerId),
    MissingStorageWindow {
        transition: LayoutTransitionId,
        worker: WorkerId,
        elements: ElementRange,
    },
    AmbiguousStorageWindow {
        transition: LayoutTransitionId,
        worker: WorkerId,
        elements: ElementRange,
    },
    InvalidStorageWindow {
        transition: LayoutTransitionId,
        storage: StorageId,
    },
    ExchangeReserveExceeded {
        worker: WorkerId,
        required: usize,
        declared: usize,
    },
    SizeOverflow,
}

impl core::fmt::Display for FleetRuntimeViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet runtime view: {self:?}")
    }
}

impl std::error::Error for FleetRuntimeViewError {}

#[derive(Clone, Copy)]
struct BoundWindow {
    storage: StorageId,
    worker: WorkerId,
    elements: ElementRange,
    offset_bytes: usize,
}

fn derive_transition_spans(
    plan: &FleetProofPlan,
    transition: &FleetTransitionPlacement,
    output: &mut Vec<FleetTransferSpan>,
) -> Result<(), FleetRuntimeViewError> {
    let value = plan.compiled().value(transition.value.version).ok_or(
        FleetRuntimeViewError::UnknownValue(transition.value.version),
    )?;
    let replica = plan
        .placement()
        .replicas
        .iter()
        .find(|replica| replica.id == transition.destination_replica)
        .ok_or(FleetRuntimeViewError::InvalidReplica(
            transition.destination_replica,
        ))?;
    let link = plan
        .placement()
        .topology
        .links
        .iter()
        .find(|link| link.id == transition.route)
        .ok_or(FleetRuntimeViewError::InvalidTransition(transition.id))?;
    if replica.value != transition.value
        || replica.canonical_worker != transition.source_worker
        || replica.origin != ReplicaOrigin::Transition(transition.id)
        || replica.layout != value.layout
        || link.source != transition.source_worker
        || link.destination != replica.worker
        || !identity_axes(&value.layout, &transition.axes)
    {
        return Err(FleetRuntimeViewError::InvalidTransition(transition.id));
    }

    let element_bytes = value.layout.element.bytes;
    let transition_bytes = transition
        .value
        .elements
        .len()
        .checked_mul(element_bytes)
        .ok_or(FleetRuntimeViewError::SizeOverflow)?;
    if transition_bytes == 0 || transition_bytes > link.max_transfer_bytes {
        return Err(FleetRuntimeViewError::InvalidTransition(transition.id));
    }

    let source = location_windows(plan, transition, transition.source_worker, element_bytes)?;
    let destination = location_windows(plan, transition, replica.worker, element_bytes)?;
    let mut boundaries = BTreeSet::from([
        transition.value.elements.start,
        transition.value.elements.end,
    ]);
    for window in source.iter().chain(&destination) {
        boundaries.extend([window.elements.start, window.elements.end]);
    }
    let boundaries = boundaries.into_iter().collect::<Vec<_>>();
    let span_base = output.len();
    for pair in boundaries.windows(2) {
        let elements = ElementRange::new(pair[0], pair[1])
            .ok_or(FleetRuntimeViewError::InvalidTransition(transition.id))?;
        let source = exact_window(&source, transition, transition.source_worker, elements)?;
        let destination = exact_window(&destination, transition, replica.worker, elements)?;
        let bytes = elements
            .len()
            .checked_mul(element_bytes)
            .ok_or(FleetRuntimeViewError::SizeOverflow)?;
        let span_ordinal = u32::try_from(output.len() - span_base)
            .map_err(|_| FleetRuntimeViewError::SizeOverflow)?;
        output.push(FleetTransferSpan {
            edge_ordinal: 0,
            transition: transition.id,
            span_ordinal,
            route: transition.route,
            owner: transition.source_worker,
            peer: replica.worker,
            elements,
            during: transition.during,
            source: physical_window(plan, transition, source, elements, bytes, element_bytes)?,
            destination: physical_window(
                plan,
                transition,
                destination,
                elements,
                bytes,
                element_bytes,
            )?,
        });
    }
    Ok(())
}

fn location_windows(
    plan: &FleetProofPlan,
    transition: &FleetTransitionPlacement,
    worker: WorkerId,
    element_bytes: usize,
) -> Result<Vec<BoundWindow>, FleetRuntimeViewError> {
    let target = transition.value.elements;
    let mut windows = Vec::new();
    for binding in &plan.placement().storage_bindings {
        if binding.value.version != transition.value.version
            || !binding.value.elements.overlaps(target)
        {
            continue;
        }
        let storage = storage(plan, binding.storage)?;
        if storage.worker != worker {
            continue;
        }
        let start = binding.value.elements.start.max(target.start);
        let end = binding.value.elements.end.min(target.end);
        let elements = ElementRange::new(start, end)
            .ok_or(FleetRuntimeViewError::InvalidTransition(transition.id))?;
        let skipped = start
            .checked_sub(binding.value.elements.start)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or(FleetRuntimeViewError::SizeOverflow)?;
        let offset_bytes = binding
            .offset_bytes
            .checked_add(skipped)
            .ok_or(FleetRuntimeViewError::SizeOverflow)?;
        windows.push(BoundWindow {
            storage: binding.storage,
            worker,
            elements,
            offset_bytes,
        });
    }
    windows.sort_unstable_by_key(|window| {
        (
            window.elements.start,
            window.elements.end,
            window.storage,
            window.offset_bytes,
        )
    });
    Ok(windows)
}

fn exact_window(
    windows: &[BoundWindow],
    transition: &FleetTransitionPlacement,
    worker: WorkerId,
    elements: ElementRange,
) -> Result<BoundWindow, FleetRuntimeViewError> {
    let mut matching = windows
        .iter()
        .copied()
        .filter(|window| window.elements.contains(elements));
    let Some(window) = matching.next() else {
        return Err(FleetRuntimeViewError::MissingStorageWindow {
            transition: transition.id,
            worker,
            elements,
        });
    };
    if matching.next().is_some() {
        return Err(FleetRuntimeViewError::AmbiguousStorageWindow {
            transition: transition.id,
            worker: window.worker,
            elements,
        });
    }
    Ok(window)
}

fn physical_window(
    plan: &FleetProofPlan,
    transition: &FleetTransitionPlacement,
    bound: BoundWindow,
    elements: ElementRange,
    bytes: usize,
    element_bytes: usize,
) -> Result<FleetStorageWindow, FleetRuntimeViewError> {
    let relative = elements
        .start
        .checked_sub(bound.elements.start)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(FleetRuntimeViewError::SizeOverflow)?;
    let offset_bytes = bound
        .offset_bytes
        .checked_add(relative)
        .ok_or(FleetRuntimeViewError::SizeOverflow)?;
    let end = offset_bytes
        .checked_add(bytes)
        .ok_or(FleetRuntimeViewError::SizeOverflow)?;
    let storage = storage(plan, bound.storage)?;
    if end > storage.bytes {
        return Err(FleetRuntimeViewError::InvalidStorageWindow {
            transition: transition.id,
            storage: bound.storage,
        });
    }
    Ok(FleetStorageWindow {
        storage: bound.storage,
        worker: bound.worker,
        offset_bytes,
        bytes,
    })
}

fn storage(plan: &FleetProofPlan, id: StorageId) -> Result<&StorageDesc, FleetRuntimeViewError> {
    plan.placement()
        .storages
        .iter()
        .find(|storage| storage.id == id)
        .ok_or(FleetRuntimeViewError::UnknownStorage(id))
}

fn identity_axes(layout: &ValueLayout, axes: &[AxisMap]) -> bool {
    axes.len() == layout.axes.len()
        && layout.axes.iter().all(|expected| {
            axes.iter()
                .any(|axis| axis.source == expected.tag && axis.destination == expected.tag)
        })
}

fn rounded_exchange_bytes(bytes: usize) -> Result<usize, FleetRuntimeViewError> {
    if bytes == 0 {
        return Err(FleetRuntimeViewError::SizeOverflow);
    }
    bytes
        .checked_add(IPC_EXCHANGE_ALLOCATION_ALIGNMENT - 1)
        .map(|bytes| bytes & !(IPC_EXCHANGE_ALLOCATION_ALIGNMENT - 1))
        .ok_or(FleetRuntimeViewError::SizeOverflow)
}
