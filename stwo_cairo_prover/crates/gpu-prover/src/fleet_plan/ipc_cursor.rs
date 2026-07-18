//! Coordinator cursor for one proof attempt's authenticated CUDA-IPC enqueues.
//!
//! Runtime admission requires an exact backend exchange key for every
//! plan-derived edge. Progress consumes the opaque receipt returned by the
//! successful backend enqueue; callers cannot synthesize it. CUDA IPC events,
//! rather than the receipt, carry device-completion ordering.

use std::collections::BTreeMap;

pub(crate) use stwo_backend_cuda::IpcExchangePhase as FleetIpcPhase;
use stwo_backend_cuda::{
    CudaDeviceIdentityReceipt, CudaDeviceUuid, IpcExchangeInstallDomain, IpcExchangeKey,
    IpcExchangePhaseReceipt,
};

use super::*;

const INSTALL_DOMAIN_TAG: &[u8] = b"stwo-cairo.fleet-ipc-install.v1\0";

const fn next_phase(phase: FleetIpcPhase) -> Option<FleetIpcPhase> {
    match phase {
        FleetIpcPhase::Published => Some(FleetIpcPhase::Consumed),
        FleetIpcPhase::Consumed => Some(FleetIpcPhase::Reclaimed),
        FleetIpcPhase::Reclaimed => Some(FleetIpcPhase::Armed),
        FleetIpcPhase::Armed => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FleetIpcAttemptState {
    Running { wave_step: ScheduleStep },
    Complete,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FleetIpcCursorProgress {
    WavePending {
        step: ScheduleStep,
    },
    WaveComplete {
        completed: ScheduleStep,
        next: ScheduleStep,
    },
    Complete {
        completed: ScheduleStep,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetIpcCursorError {
    GenerationOverflow,
    ZeroControllerInstallNonce,
    RuntimeRosterCount {
        expected: usize,
        actual: usize,
    },
    NonDenseRuntimeRoster {
        expected: WorkerId,
        actual: WorkerId,
    },
    ZeroRuntimeDevice(WorkerId),
    DuplicateRuntimeDevice {
        first: WorkerId,
        duplicate: WorkerId,
    },
    LocalWorkerMissing(WorkerId),
    LocalDeviceAnnouncementMismatch(WorkerId),
    RuntimeRosterPlanMismatch,
    InvalidInstallDomain,
    NonDenseEdge {
        expected: u64,
        actual: u64,
    },
    ExchangeKeyCount {
        expected: usize,
        actual: usize,
    },
    InstallDomainMismatch(u64),
    DeviceRosterMismatch(u64),
    ExchangeKeyMismatch(u64),
    UnknownEdge(u64),
    GenerationMismatch {
        expected: u64,
        actual: u64,
    },
    OutOfOrderPhase {
        edge: u64,
        expected: Option<FleetIpcPhase>,
        actual: FleetIpcPhase,
    },
    WrongWave {
        edge: u64,
        active_step: ScheduleStep,
    },
    AttemptComplete,
    AttemptPoisoned,
}

impl core::fmt::Display for FleetIpcCursorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet IPC runtime state: {self:?}")
    }
}

impl std::error::Error for FleetIpcCursorError {}

struct EdgeCursor {
    span: FleetTransferSpan,
    next: Option<FleetIpcPhase>,
}

struct IpcWave {
    step: ScheduleStep,
    edges: Vec<u64>,
}

/// Rank-local binding from a dense public roster to one live CUDA context.
///
/// Every process consumes only its own opaque backend receipt. The complete
/// roster is copyable control-plane data used to derive the common install
/// domain; it is not authority for another rank's CUDA context.
pub(crate) struct FleetIpcRuntimeRosterBinding {
    plan_identity: [u8; 32],
    controller_install_nonce: [u8; 32],
    local_worker: WorkerId,
    roster: Box<[(WorkerId, CudaDeviceUuid)]>,
    _local_identity: RuntimeDeviceIdentity,
}

struct RuntimeDeviceIdentity {
    uuid: CudaDeviceUuid,
    #[cfg(not(test))]
    _backend_receipt: CudaDeviceIdentityReceipt,
    #[cfg(test)]
    _backend_receipt: Option<CudaDeviceIdentityReceipt>,
}

impl FleetIpcRuntimeRosterBinding {
    /// Bind this rank to one exact plan install. The controller must supply a
    /// fresh globally unique nonce; local validation proves only nonzero.
    pub(crate) fn bind_local(
        view: &FleetRuntimeView,
        controller_install_nonce: [u8; 32],
        local_worker: WorkerId,
        local_identity: CudaDeviceIdentityReceipt,
        roster: &[(WorkerId, CudaDeviceUuid)],
    ) -> Result<Self, FleetIpcCursorError> {
        let local_identity = RuntimeDeviceIdentity {
            uuid: local_identity.device_uuid(),
            #[cfg(not(test))]
            _backend_receipt: local_identity,
            #[cfg(test)]
            _backend_receipt: Some(local_identity),
        };
        Self::bind_parts(
            view,
            controller_install_nonce,
            local_worker,
            local_identity,
            roster,
        )
    }

    #[cfg(test)]
    pub(super) fn bind_test_only(
        view: &FleetRuntimeView,
        controller_install_nonce: [u8; 32],
        local_worker: WorkerId,
        observed_local_uuid: CudaDeviceUuid,
        roster: &[(WorkerId, CudaDeviceUuid)],
    ) -> Result<Self, FleetIpcCursorError> {
        Self::bind_parts(
            view,
            controller_install_nonce,
            local_worker,
            RuntimeDeviceIdentity {
                uuid: observed_local_uuid,
                _backend_receipt: None,
            },
            roster,
        )
    }

    fn bind_parts(
        view: &FleetRuntimeView,
        controller_install_nonce: [u8; 32],
        local_worker: WorkerId,
        local_identity: RuntimeDeviceIdentity,
        roster: &[(WorkerId, CudaDeviceUuid)],
    ) -> Result<Self, FleetIpcCursorError> {
        if controller_install_nonce == [0; 32] {
            return Err(FleetIpcCursorError::ZeroControllerInstallNonce);
        }
        if roster.len() != view.exchange_reserves().len() {
            return Err(FleetIpcCursorError::RuntimeRosterCount {
                expected: view.exchange_reserves().len(),
                actual: roster.len(),
            });
        }

        let mut first_by_uuid = BTreeMap::<[u8; 16], WorkerId>::new();
        for (index, ((worker, uuid), reserve)) in
            roster.iter().zip(view.exchange_reserves()).enumerate()
        {
            let expected = WorkerId(u16::try_from(index).map_err(|_| {
                FleetIpcCursorError::RuntimeRosterCount {
                    expected: view.exchange_reserves().len(),
                    actual: roster.len(),
                }
            })?);
            if *worker != expected || reserve.worker != expected {
                return Err(FleetIpcCursorError::NonDenseRuntimeRoster {
                    expected,
                    actual: *worker,
                });
            }
            let bytes = *uuid.as_bytes();
            if bytes == [0; 16] {
                return Err(FleetIpcCursorError::ZeroRuntimeDevice(*worker));
            }
            if let Some(first) = first_by_uuid.insert(bytes, *worker) {
                return Err(FleetIpcCursorError::DuplicateRuntimeDevice {
                    first,
                    duplicate: *worker,
                });
            }
        }
        let local_uuid = roster
            .get(usize::from(local_worker.0))
            .and_then(|&(worker, uuid)| (worker == local_worker).then_some(uuid))
            .ok_or(FleetIpcCursorError::LocalWorkerMissing(local_worker))?;
        if local_identity.uuid != local_uuid {
            return Err(FleetIpcCursorError::LocalDeviceAnnouncementMismatch(
                local_worker,
            ));
        }
        Ok(Self {
            plan_identity: view.plan_identity(),
            controller_install_nonce,
            local_worker,
            roster: roster.to_vec().into_boxed_slice(),
            _local_identity: local_identity,
        })
    }

    pub(crate) fn install_domain(&self) -> Result<IpcExchangeInstallDomain, FleetIpcCursorError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(INSTALL_DOMAIN_TAG);
        hasher.update(&self.plan_identity);
        hasher.update(&self.controller_install_nonce);
        hasher.update(&(self.roster.len() as u64).to_le_bytes());
        for (worker, uuid) in &self.roster {
            hasher.update(&worker.0.to_le_bytes());
            hasher.update(uuid.as_bytes());
        }
        IpcExchangeInstallDomain::from_digest(*hasher.finalize().as_bytes())
            .map_err(|_| FleetIpcCursorError::InvalidInstallDomain)
    }

    pub(super) fn device(&self, worker: WorkerId) -> Option<CudaDeviceUuid> {
        self.roster
            .get(usize::from(worker.0))
            .and_then(|(rank, uuid)| (*rank == worker).then_some(*uuid))
    }

    pub(crate) const fn local_worker(&self) -> WorkerId {
        self.local_worker
    }
}

/// Copyable rank-process acknowledgement of one locally accepted CUDA phase.
///
/// The local rank can construct this only by consuming the opaque backend
/// receipt. The controller must still authenticate the process channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FleetIpcRankPhaseStatement {
    plan_identity: [u8; 32],
    worker: WorkerId,
    key: IpcExchangeKey,
    phase: FleetIpcPhase,
    generation: u64,
}

impl FleetIpcRankPhaseStatement {
    pub(super) fn from_backend(
        plan_identity: [u8; 32],
        worker: WorkerId,
        receipt: IpcExchangePhaseReceipt,
    ) -> Self {
        Self {
            plan_identity,
            worker,
            key: receipt.key(),
            phase: receipt.phase(),
            generation: receipt.generation(),
        }
    }

    pub(crate) const fn worker(&self) -> WorkerId {
        self.worker
    }

    pub(crate) const fn key(&self) -> IpcExchangeKey {
        self.key
    }

    pub(crate) const fn phase(&self) -> FleetIpcPhase {
        self.phase
    }

    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Pure ordering model used by both runtime admission and structural replay.
pub(super) struct IpcScheduleCursor {
    proof_generation: u64,
    state: FleetIpcAttemptState,
    edges: Vec<EdgeCursor>,
    waves: Vec<IpcWave>,
    active_wave: usize,
}

impl IpcScheduleCursor {
    pub(super) fn new(
        view: &FleetRuntimeView,
        proof_generation: u64,
    ) -> Result<Self, FleetIpcCursorError> {
        proof_generation
            .checked_add(1)
            .ok_or(FleetIpcCursorError::GenerationOverflow)?;
        let mut grouped = BTreeMap::<ScheduleStep, Vec<u64>>::new();
        let mut edges = Vec::with_capacity(view.spans().len());
        for (expected, &span) in view.spans().iter().enumerate() {
            let expected =
                u64::try_from(expected).map_err(|_| FleetIpcCursorError::NonDenseEdge {
                    expected: u64::MAX,
                    actual: span.edge_ordinal,
                })?;
            if span.edge_ordinal != expected {
                return Err(FleetIpcCursorError::NonDenseEdge {
                    expected,
                    actual: span.edge_ordinal,
                });
            }
            grouped
                .entry(span.during.start)
                .or_default()
                .push(span.edge_ordinal);
            edges.push(EdgeCursor {
                span,
                next: Some(FleetIpcPhase::Published),
            });
        }
        let waves = grouped
            .into_iter()
            .map(|(step, edges)| IpcWave { step, edges })
            .collect::<Vec<_>>();
        let state = waves
            .first()
            .map_or(FleetIpcAttemptState::Complete, |wave| {
                FleetIpcAttemptState::Running {
                    wave_step: wave.step,
                }
            });
        Ok(Self {
            proof_generation,
            state,
            edges,
            waves,
            active_wave: 0,
        })
    }

    pub(super) const fn state(&self) -> FleetIpcAttemptState {
        self.state
    }

    pub(super) fn active_wave_edges(&self) -> &[u64] {
        self.waves
            .get(self.active_wave)
            .map_or(&[], |wave| wave.edges.as_slice())
    }

    pub(super) fn accept_phase(
        &mut self,
        edge_ordinal: u64,
        phase: FleetIpcPhase,
        generation: u64,
    ) -> Result<FleetIpcCursorProgress, FleetIpcCursorError> {
        match self.state {
            FleetIpcAttemptState::Poisoned => return Err(FleetIpcCursorError::AttemptPoisoned),
            FleetIpcAttemptState::Complete => {
                return self.poison(FleetIpcCursorError::AttemptComplete)
            }
            FleetIpcAttemptState::Running { .. } => {}
        }
        let edge_index = match usize::try_from(edge_ordinal) {
            Ok(index) if index < self.edges.len() => index,
            _ => return self.poison(FleetIpcCursorError::UnknownEdge(edge_ordinal)),
        };
        let active = &self.waves[self.active_wave];
        if !active.edges.contains(&edge_ordinal) {
            return self.poison(FleetIpcCursorError::WrongWave {
                edge: edge_ordinal,
                active_step: active.step,
            });
        }

        let edge = &self.edges[edge_index];
        if edge.span.edge_ordinal != edge_ordinal {
            return self.poison(FleetIpcCursorError::UnknownEdge(edge_ordinal));
        }
        if edge.next != Some(phase) {
            return self.poison(FleetIpcCursorError::OutOfOrderPhase {
                edge: edge_ordinal,
                expected: edge.next,
                actual: phase,
            });
        }
        let expected = if phase == FleetIpcPhase::Armed {
            self.proof_generation
                .checked_add(1)
                .ok_or(FleetIpcCursorError::GenerationOverflow)?
        } else {
            self.proof_generation
        };
        if generation != expected {
            return self.poison(FleetIpcCursorError::GenerationMismatch {
                expected,
                actual: generation,
            });
        }

        self.edges[edge_index].next = next_phase(phase);
        if !active
            .edges
            .iter()
            .all(|&edge| self.edges[edge as usize].next.is_none())
        {
            return Ok(FleetIpcCursorProgress::WavePending { step: active.step });
        }

        let completed = active.step;
        self.active_wave += 1;
        if let Some(next) = self.waves.get(self.active_wave) {
            self.state = FleetIpcAttemptState::Running {
                wave_step: next.step,
            };
            Ok(FleetIpcCursorProgress::WaveComplete {
                completed,
                next: next.step,
            })
        } else {
            self.state = FleetIpcAttemptState::Complete;
            Ok(FleetIpcCursorProgress::Complete { completed })
        }
    }

    fn poison<T>(&mut self, error: FleetIpcCursorError) -> Result<T, FleetIpcCursorError> {
        self.state = FleetIpcAttemptState::Poisoned;
        Err(error)
    }
}

/// Fail-closed runtime state for one proof attempt.
pub(crate) struct FleetIpcCoordinatorCursor {
    plan_identity: [u8; 32],
    proof_generation: u64,
    schedule: IpcScheduleCursor,
    keys: Vec<IpcExchangeKey>,
}

impl FleetIpcCoordinatorCursor {
    /// Install the exact backend keys already created for every transfer edge.
    pub(crate) fn install(
        view: &FleetRuntimeView,
        proof_generation: u64,
        runtime: &FleetIpcRuntimeRosterBinding,
        keys: &[IpcExchangeKey],
    ) -> Result<Self, FleetIpcCursorError> {
        if runtime.plan_identity != view.plan_identity() {
            return Err(FleetIpcCursorError::RuntimeRosterPlanMismatch);
        }
        let schedule = IpcScheduleCursor::new(view, proof_generation)?;
        let install_domain = runtime.install_domain()?;
        if keys.len() != view.spans().len() {
            return Err(FleetIpcCursorError::ExchangeKeyCount {
                expected: view.spans().len(),
                actual: keys.len(),
            });
        }
        for (&key, span) in keys.iter().zip(view.spans()) {
            if key.install_domain() != install_domain {
                return Err(FleetIpcCursorError::InstallDomainMismatch(
                    span.edge_ordinal,
                ));
            }
            if runtime.device(span.owner) != Some(key.owner_device())
                || runtime.device(span.peer) != Some(key.peer_device())
            {
                return Err(FleetIpcCursorError::DeviceRosterMismatch(span.edge_ordinal));
            }
            if key.edge_id() != span.edge_ordinal
                || key.owner_rank() != u32::from(span.owner.0)
                || key.peer_rank() != u32::from(span.peer.0)
                || key.logical_bytes() != span.logical_bytes()
                || key.initial_generation() > proof_generation
            {
                return Err(FleetIpcCursorError::ExchangeKeyMismatch(span.edge_ordinal));
            }
        }
        Ok(Self {
            plan_identity: view.plan_identity(),
            proof_generation,
            schedule,
            keys: keys.to_vec(),
        })
    }

    pub(crate) const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub(crate) const fn proof_generation(&self) -> u64 {
        self.proof_generation
    }

    pub(crate) const fn state(&self) -> FleetIpcAttemptState {
        self.schedule.state()
    }

    pub(crate) fn active_wave_edges(&self) -> &[u64] {
        self.schedule.active_wave_edges()
    }

    /// Accept one statement received from the authenticated rank channel.
    ///
    /// CUDA authority remains rank-local: this advances only controller
    /// protocol state after validating the exact plan, worker, key and phase.
    pub(crate) fn accept_statement(
        &mut self,
        statement: FleetIpcRankPhaseStatement,
    ) -> Result<FleetIpcCursorProgress, FleetIpcCursorError> {
        match self.schedule.state() {
            FleetIpcAttemptState::Poisoned => return Err(FleetIpcCursorError::AttemptPoisoned),
            FleetIpcAttemptState::Complete => {
                return self.schedule.poison(FleetIpcCursorError::AttemptComplete)
            }
            FleetIpcAttemptState::Running { .. } => {}
        }
        if statement.plan_identity != self.plan_identity {
            return self
                .schedule
                .poison(FleetIpcCursorError::RuntimeRosterPlanMismatch);
        }
        let key = statement.key();
        let edge = key.edge_id();
        let expected = usize::try_from(edge)
            .ok()
            .and_then(|index| self.keys.get(index))
            .copied()
            .ok_or(FleetIpcCursorError::UnknownEdge(edge));
        let expected = match expected {
            Ok(expected) => expected,
            Err(error) => return self.schedule.poison(error),
        };
        if key != expected {
            return self
                .schedule
                .poison(FleetIpcCursorError::ExchangeKeyMismatch(edge));
        }
        let expected_worker = match statement.phase() {
            FleetIpcPhase::Published | FleetIpcPhase::Reclaimed => WorkerId(
                u16::try_from(key.owner_rank())
                    .map_err(|_| FleetIpcCursorError::ExchangeKeyMismatch(edge))?,
            ),
            FleetIpcPhase::Consumed | FleetIpcPhase::Armed => WorkerId(
                u16::try_from(key.peer_rank())
                    .map_err(|_| FleetIpcCursorError::ExchangeKeyMismatch(edge))?,
            ),
        };
        if statement.worker() != expected_worker {
            return self
                .schedule
                .poison(FleetIpcCursorError::ExchangeKeyMismatch(edge));
        }
        self.schedule
            .accept_phase(edge, statement.phase(), statement.generation())
    }
}

#[cfg(test)]
mod canonical_domain_tests {
    use super::INSTALL_DOMAIN_TAG;

    #[test]
    fn install_domain_tag_is_frozen_byte_for_byte() {
        assert_eq!(INSTALL_DOMAIN_TAG, b"stwo-cairo.fleet-ipc-install.v1\0");
    }
}
