//! Pure wait-all ordering cursor for proof-internal fleet transcript barriers.
//!
//! This does not authenticate workers or allocate globally unique proof
//! generations. A production controller must bind receipts to its queue-owned
//! attempt ID, transport identity and exact completion event before admission.

use crate::fleet_plan::{FleetProofPlan, ScheduleStep, WorkerId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BarrierReceipt {
    pub plan_identity: [u8; 32],
    pub proof_generation: u64,
    pub barrier_ordinal: u32,
    pub worker: WorkerId,
    pub ready_step: ScheduleStep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BarrierRelease {
    pub plan_identity: [u8; 32],
    pub proof_generation: u64,
    pub barrier_ordinal: u32,
    pub coordinator: WorkerId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrivalState {
    Waiting { remaining: u16 },
    Ready,
}

pub struct CoordinatorBarrierCursor {
    plan_identity: [u8; 32],
    proof_generation: u64,
    coordinator: WorkerId,
    barrier_count: u32,
    next_ordinal: u32,
    arrived: Vec<bool>,
    ready_steps: Vec<ScheduleStep>,
}

impl CoordinatorBarrierCursor {
    pub fn new(plan: &FleetProofPlan, proof_generation: u64) -> Result<Self, FleetBarrierError> {
        let barrier_count = plan
            .fence_count()
            .map_err(|_| FleetBarrierError::SizeOverflow)?;
        let worker_count = plan.placement().topology.workers.len();
        if barrier_count == 0 || worker_count == 0 {
            return Err(FleetBarrierError::InvalidPlan);
        }
        let receipt_count = usize::try_from(barrier_count)
            .map_err(|_| FleetBarrierError::SizeOverflow)?
            .checked_mul(worker_count)
            .ok_or(FleetBarrierError::SizeOverflow)?;
        if plan.placement().barrier_arrivals.len() != receipt_count {
            return Err(FleetBarrierError::InvalidPlan);
        }
        Ok(Self {
            plan_identity: plan.identity(),
            proof_generation,
            coordinator: plan.placement().topology.coordinator,
            barrier_count,
            next_ordinal: 0,
            arrived: vec![false; worker_count],
            ready_steps: plan
                .placement()
                .barrier_arrivals
                .iter()
                .map(|arrival| arrival.ready_step)
                .collect(),
        })
    }

    pub fn arrive(&mut self, receipt: BarrierReceipt) -> Result<ArrivalState, FleetBarrierError> {
        if self.next_ordinal >= self.barrier_count {
            return Err(FleetBarrierError::Complete);
        }
        if receipt.plan_identity != self.plan_identity
            || receipt.proof_generation != self.proof_generation
            || receipt.barrier_ordinal != self.next_ordinal
        {
            return Err(FleetBarrierError::StaleOrFutureReceipt);
        }
        let rank = usize::from(receipt.worker.0);
        if rank >= self.arrived.len() {
            return Err(FleetBarrierError::UnknownWorker(receipt.worker));
        }
        let ready_step_index = usize::try_from(self.next_ordinal)
            .map_err(|_| FleetBarrierError::SizeOverflow)?
            .checked_mul(self.arrived.len())
            .and_then(|base| base.checked_add(rank))
            .ok_or(FleetBarrierError::SizeOverflow)?;
        let expected_ready_step = self
            .ready_steps
            .get(ready_step_index)
            .copied()
            .ok_or(FleetBarrierError::UnknownWorker(receipt.worker))?;
        if receipt.ready_step != expected_ready_step {
            return Err(FleetBarrierError::UnexpectedReadyStep {
                worker: receipt.worker,
                expected: expected_ready_step,
                actual: receipt.ready_step,
            });
        }
        let arrived = self
            .arrived
            .get_mut(rank)
            .expect("worker rank was bounds checked");
        if core::mem::replace(arrived, true) {
            return Err(FleetBarrierError::DuplicateWorker(receipt.worker));
        }
        let remaining = self.arrived.iter().filter(|&&seen| !seen).count();
        if remaining == 0 {
            Ok(ArrivalState::Ready)
        } else {
            Ok(ArrivalState::Waiting {
                remaining: u16::try_from(remaining).map_err(|_| FleetBarrierError::SizeOverflow)?,
            })
        }
    }

    pub fn release(&mut self, coordinator: WorkerId) -> Result<BarrierRelease, FleetBarrierError> {
        if coordinator != self.coordinator {
            return Err(FleetBarrierError::WrongCoordinator(coordinator));
        }
        if self.next_ordinal >= self.barrier_count {
            return Err(FleetBarrierError::Complete);
        }
        if self.arrived.iter().any(|&seen| !seen) {
            return Err(FleetBarrierError::IncompleteBarrier);
        }
        let release = BarrierRelease {
            plan_identity: self.plan_identity,
            proof_generation: self.proof_generation,
            barrier_ordinal: self.next_ordinal,
            coordinator,
        };
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or(FleetBarrierError::SizeOverflow)?;
        self.arrived.fill(false);
        Ok(release)
    }

    pub const fn is_complete(&self) -> bool {
        self.next_ordinal >= self.barrier_count
    }
}

pub struct WorkerBarrierCursor {
    plan_identity: [u8; 32],
    proof_generation: u64,
    coordinator: WorkerId,
    barrier_count: u32,
    next_release: u32,
}

impl WorkerBarrierCursor {
    pub fn new(plan: &FleetProofPlan, proof_generation: u64) -> Result<Self, FleetBarrierError> {
        Ok(Self {
            plan_identity: plan.identity(),
            proof_generation,
            coordinator: plan.placement().topology.coordinator,
            barrier_count: plan
                .fence_count()
                .map_err(|_| FleetBarrierError::SizeOverflow)?,
            next_release: 0,
        })
    }

    pub fn accept(&mut self, release: BarrierRelease) -> Result<(), FleetBarrierError> {
        if release.plan_identity != self.plan_identity
            || release.proof_generation != self.proof_generation
            || release.barrier_ordinal != self.next_release
        {
            return Err(FleetBarrierError::StaleOrFutureRelease);
        }
        if release.coordinator != self.coordinator {
            return Err(FleetBarrierError::WrongCoordinator(release.coordinator));
        }
        if self.next_release >= self.barrier_count {
            return Err(FleetBarrierError::Complete);
        }
        self.next_release = self
            .next_release
            .checked_add(1)
            .ok_or(FleetBarrierError::SizeOverflow)?;
        Ok(())
    }

    /// Interval zero is initially runnable; interval `n+1` requires release
    /// `n`. The interval after the final transcript release is the proof tail;
    /// its release is the terminal wait-all fence.
    pub fn can_start_segment(&self, segment: u32) -> bool {
        segment < self.barrier_count && segment == self.next_release
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetBarrierError {
    InvalidPlan,
    UnknownWorker(WorkerId),
    DuplicateWorker(WorkerId),
    WrongCoordinator(WorkerId),
    StaleOrFutureReceipt,
    StaleOrFutureRelease,
    UnexpectedReadyStep {
        worker: WorkerId,
        expected: ScheduleStep,
        actual: ScheduleStep,
    },
    IncompleteBarrier,
    Complete,
    SizeOverflow,
}

impl core::fmt::Display for FleetBarrierError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet transcript barrier: {self:?}")
    }
}

impl std::error::Error for FleetBarrierError {}
