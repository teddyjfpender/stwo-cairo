//! Pure structural closure oracle for one two-rank proof attempt.
//!
//! This replays only the address-free schedule already sealed by
//! [`FleetProofPlan`]. It allocates no CUDA resource, executes no kernel or
//! transfer, advances no real transcript, proves no transcript or output bytes
//! and grants no runtime admission, qualification or benchmark credit. Its
//! synthesized transcript phases prove controller ordering only.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::compiled_proof::{CompiledProofError, OpId};
use crate::fleet_barrier::{
    ArrivalState, BarrierReceipt, CoordinatorBarrierCursor, FleetBarrierError, WorkerBarrierCursor,
};

pub(super) mod validation;
use validation::{
    inside_interval, operation_segment, require_two_rank_topology, transition_segment,
    validate_install_closure, validate_span_projection,
};

/// Sealed observational result of one successful structural replay.
///
/// Private fields prevent callers from constructing a receipt independently of
/// [`FleetProofPlan::simulate_two_rank_structural_closure`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetTwoRankStructuralClosureReceipt {
    plan_identity: [u8; 32],
    proof_generation: u64,
    worker_executions: u64,
    exact_shards: u64,
    transfer_spans: u64,
    ipc_phase_receipts: u64,
    fence_releases: u32,
    event_steps: u32,
    synthesized_transcript_phases: u32,
}

impl FleetTwoRankStructuralClosureReceipt {
    pub const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub const fn proof_generation(&self) -> u64 {
        self.proof_generation
    }

    pub const fn worker_executions(&self) -> u64 {
        self.worker_executions
    }

    pub const fn exact_shards(&self) -> u64 {
        self.exact_shards
    }

    pub const fn transfer_spans(&self) -> u64 {
        self.transfer_spans
    }

    pub const fn ipc_phase_receipts(&self) -> u64 {
        self.ipc_phase_receipts
    }

    pub const fn fence_releases(&self) -> u32 {
        self.fence_releases
    }

    pub const fn event_steps(&self) -> u32 {
        self.event_steps
    }

    /// Number of controller-only transcript phases synthesized at their
    /// declared release step. This is not hardware or transcript-byte evidence.
    pub const fn synthesized_transcript_phases(&self) -> u32 {
        self.synthesized_transcript_phases
    }
}

impl FleetProofPlan {
    /// Prove that the sealed two-rank metadata admits one complete structural
    /// execution/transfer/barrier replay.
    ///
    /// The returned receipt is deliberately non-admitting: hardware completion
    /// events, transcript values and proof bytes remain outside this oracle.
    pub fn simulate_two_rank_structural_closure(
        &self,
        proof_generation: u64,
    ) -> Result<FleetTwoRankStructuralClosureReceipt, FleetTwoRankStructuralClosureError> {
        require_two_rank_topology(self)?;
        if super::identity::compute(self)? != self.identity() {
            return Err(FleetTwoRankStructuralClosureError::PlanIdentityMismatch);
        }

        let view = self.runtime_view()?;
        let installs = [
            self.worker_install_plan(WorkerId(0))?,
            self.worker_install_plan(WorkerId(1))?,
        ];
        let transcript_ordinals = validate_install_closure(self, &view, &installs)?;
        let replay = ReplayIndex::new(self, &view, &installs, transcript_ordinals)?;

        let mut ipc = FleetIpcCoordinatorCursor::new(&view, proof_generation)?;
        let mut coordinator = CoordinatorBarrierCursor::new(self, proof_generation)?;
        let mut workers = [
            WorkerBarrierCursor::new(self, proof_generation)?,
            WorkerBarrierCursor::new(self, proof_generation)?,
        ];
        let mut started_executions = BTreeSet::new();
        let mut completed_executions = BTreeSet::new();
        let mut started_edges = BTreeSet::new();
        let mut completed_edges = BTreeSet::new();
        let mut completed_transcript = BTreeSet::new();
        let mut ready_ordinals = BTreeSet::new();

        for &step in &replay.event_steps {
            complete_step(
                step,
                &replay,
                &view,
                &started_executions,
                &started_edges,
                &mut completed_executions,
                &mut completed_edges,
            )?;

            if let Some(ordinal) = release_ordinal(self, step)? {
                if !ready_ordinals.remove(&ordinal) {
                    return Err(FleetBarrierError::IncompleteBarrier.into());
                }
                synthesize_transcript_phase(
                    self,
                    &replay,
                    ordinal,
                    step,
                    &mut completed_transcript,
                )?;
                let release = coordinator.release(WorkerId(0))?;
                if release.barrier_ordinal != ordinal {
                    return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
                }
                for worker in &mut workers {
                    worker.accept(release)?;
                }
            }

            submit_arrivals(
                self,
                proof_generation,
                step,
                &replay,
                &completed_executions,
                &completed_edges,
                &mut coordinator,
                &mut ready_ordinals,
            )?;

            start_transfer_wave(
                self,
                &view,
                proof_generation,
                step,
                &mut ipc,
                &workers,
                &mut started_edges,
            )?;
            start_executions(self, step, &replay, &workers, &mut started_executions)?;
        }

        if started_executions != replay.executions
            || completed_executions != replay.executions
            || started_edges != replay.edges
            || completed_edges != replay.edges
            || ipc.state() != FleetIpcAttemptState::Complete
            || !coordinator.is_complete()
            || workers.iter().any(|worker| !worker.is_complete())
            || completed_transcript.len() != self.barriers().len()
            || !ready_ordinals.is_empty()
        {
            return Err(FleetTwoRankStructuralClosureError::Incomplete);
        }

        let worker_executions = u64::try_from(replay.executions.len())
            .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?;
        let transfer_spans = u64::try_from(view.spans().len())
            .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?;
        let ipc_phase_receipts = transfer_spans
            .checked_mul(4)
            .ok_or(FleetTwoRankStructuralClosureError::SizeOverflow)?;
        Ok(FleetTwoRankStructuralClosureReceipt {
            plan_identity: self.identity(),
            proof_generation,
            worker_executions,
            exact_shards: replay.exact_shards,
            transfer_spans,
            ipc_phase_receipts,
            fence_releases: self.fence_count()?,
            event_steps: u32::try_from(replay.event_steps.len())
                .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?,
            synthesized_transcript_phases: u32::try_from(completed_transcript.len())
                .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExecutionKey {
    worker: WorkerId,
    operation: OpId,
    domain_tag: u8,
    domain_start: usize,
    domain_end: usize,
    start: ScheduleStep,
    end: ScheduleStep,
}

impl ExecutionKey {
    const fn new(
        worker: WorkerId,
        operation: OpId,
        domain: OperationDomain,
        during: ScheduleRange,
    ) -> Self {
        let (domain_tag, domain_start, domain_end) = match domain {
            OperationDomain::Monolithic => (0, 0, 0),
            OperationDomain::Exact(range) => (1, range.start, range.end),
        };
        Self {
            worker,
            operation,
            domain_tag,
            domain_start,
            domain_end,
            start: during.start,
            end: during.end,
        }
    }
}

#[derive(Default)]
struct SegmentWork {
    executions: BTreeSet<ExecutionKey>,
    edges: BTreeSet<u64>,
}

struct ReplayIndex {
    executions: BTreeSet<ExecutionKey>,
    exact_shards: u64,
    edges: BTreeSet<u64>,
    segment_work: BTreeMap<(u32, WorkerId), SegmentWork>,
    transcript_ordinals: BTreeSet<u32>,
    event_steps: BTreeSet<ScheduleStep>,
}

impl ReplayIndex {
    fn new(
        plan: &FleetProofPlan,
        view: &FleetRuntimeView,
        installs: &[FleetWorkerInstallPlan; 2],
        transcript_ordinals: BTreeSet<u32>,
    ) -> Result<Self, FleetTwoRankStructuralClosureError> {
        let expected = plan
            .placement()
            .operations
            .iter()
            .flat_map(|placement| {
                placement.executions.iter().map(|execution| {
                    ExecutionKey::new(
                        execution.worker,
                        placement.operation,
                        execution.domain,
                        placement.during,
                    )
                })
            })
            .collect::<BTreeSet<_>>();
        let mut executions = BTreeSet::new();
        let mut exact_shards = 0u64;
        let mut segment_work = BTreeMap::<(u32, WorkerId), SegmentWork>::new();

        for install in installs {
            let worker = install.target().worker;
            for execution in install.executions() {
                if execution.step != execution.during.start || execution.executables.is_empty() {
                    return Err(FleetTwoRankStructuralClosureError::ExecutionClosure);
                }
                let key = ExecutionKey::new(
                    worker,
                    execution.operation,
                    execution.domain,
                    execution.during,
                );
                if !executions.insert(key) {
                    return Err(FleetTwoRankStructuralClosureError::ExecutionClosure);
                }
                let segment = operation_segment(plan, execution.operation)?;
                if !inside_interval(plan, segment, execution.during) {
                    return Err(FleetTwoRankStructuralClosureError::ExecutionClosure);
                }
                segment_work
                    .entry((segment, worker))
                    .or_default()
                    .executions
                    .insert(key);
                if let OperationDomain::Exact(shard) = execution.domain {
                    plan.compiled()
                        .materialize_exact_shard(execution.operation, shard)?;
                    exact_shards = exact_shards
                        .checked_add(1)
                        .ok_or(FleetTwoRankStructuralClosureError::SizeOverflow)?;
                }
            }
        }
        if executions != expected {
            return Err(FleetTwoRankStructuralClosureError::ExecutionClosure);
        }

        let mut edges = BTreeSet::new();
        for &span in view.spans() {
            if span.during.end > plan.terminal_step() || !edges.insert(span.edge_ordinal) {
                return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                    edge: span.edge_ordinal,
                });
            }
            validate_span_projection(plan, span)?;
            let segment = transition_segment(plan, span.transition)?;
            if !inside_interval(plan, segment, span.during) {
                return Err(FleetTwoRankStructuralClosureError::TransferClosure {
                    edge: span.edge_ordinal,
                });
            }
            for worker in [span.owner, span.peer] {
                segment_work
                    .entry((segment, worker))
                    .or_default()
                    .edges
                    .insert(span.edge_ordinal);
            }
        }

        let event_steps = executions
            .iter()
            .flat_map(|execution| [execution.start, execution.end])
            .chain(
                view.spans()
                    .iter()
                    .flat_map(|span| [span.during.start, span.during.end]),
            )
            .chain(
                plan.placement()
                    .barrier_arrivals
                    .iter()
                    .map(|arrival| arrival.ready_step),
            )
            .chain(plan.barriers().iter().map(|barrier| barrier.release_step))
            .chain(core::iter::once(plan.terminal_step()))
            .collect::<BTreeSet<_>>();
        if event_steps.iter().any(|&step| step > plan.terminal_step()) {
            return Err(FleetTwoRankStructuralClosureError::MissedEvent {
                step: plan.terminal_step(),
            });
        }
        Ok(Self {
            executions,
            exact_shards,
            edges,
            segment_work,
            transcript_ordinals,
            event_steps,
        })
    }
}

fn complete_step(
    step: ScheduleStep,
    replay: &ReplayIndex,
    view: &FleetRuntimeView,
    started_executions: &BTreeSet<ExecutionKey>,
    started_edges: &BTreeSet<u64>,
    completed_executions: &mut BTreeSet<ExecutionKey>,
    completed_edges: &mut BTreeSet<u64>,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    for &execution in replay
        .executions
        .iter()
        .filter(|execution| execution.end == step)
    {
        if !started_executions.contains(&execution) || !completed_executions.insert(execution) {
            return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
        }
    }
    for edge in view
        .spans()
        .iter()
        .filter(|span| span.during.end == step)
        .map(|span| span.edge_ordinal)
    {
        if !started_edges.contains(&edge) || !completed_edges.insert(edge) {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure { edge });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn start_transfer_wave(
    plan: &FleetProofPlan,
    view: &FleetRuntimeView,
    proof_generation: u64,
    step: ScheduleStep,
    ipc: &mut FleetIpcCoordinatorCursor,
    workers: &[WorkerBarrierCursor; 2],
    started_edges: &mut BTreeSet<u64>,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let expected = view
        .spans()
        .iter()
        .filter(|span| span.during.start == step)
        .map(|span| span.edge_ordinal)
        .collect::<Vec<_>>();
    match ipc.state() {
        FleetIpcAttemptState::Running { wave_step } if wave_step < step => {
            return Err(FleetTwoRankStructuralClosureError::MissedEvent { step: wave_step })
        }
        FleetIpcAttemptState::Running { wave_step } if wave_step == step => {}
        FleetIpcAttemptState::Running { .. } | FleetIpcAttemptState::Complete
            if expected.is_empty() =>
        {
            return Ok(())
        }
        FleetIpcAttemptState::Poisoned => {
            return Err(FleetTwoRankStructuralClosureError::Incomplete)
        }
        _ => return Err(FleetTwoRankStructuralClosureError::MissedEvent { step }),
    }

    let active = ipc.active_wave_edges().to_vec();
    if active != expected {
        return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
    }
    for &edge in &active {
        let span = span(view, edge)?;
        let segment = transition_segment(plan, span.transition)?;
        require_unlocked(workers, span.owner, step, segment)?;
        require_unlocked(workers, span.peer, step, segment)?;
        if span.during.start != step {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure { edge });
        }
    }
    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
        FleetIpcPhase::Armed,
    ] {
        for &edge in &active {
            let receipt = FleetIpcPhaseReceipt::for_span(
                plan.identity(),
                proof_generation,
                span(view, edge)?,
                phase,
            )?;
            ipc.accept(receipt)?;
        }
    }
    for edge in active {
        if !started_edges.insert(edge) {
            return Err(FleetTwoRankStructuralClosureError::TransferClosure { edge });
        }
    }
    Ok(())
}

fn start_executions(
    plan: &FleetProofPlan,
    step: ScheduleStep,
    replay: &ReplayIndex,
    workers: &[WorkerBarrierCursor; 2],
    started: &mut BTreeSet<ExecutionKey>,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    for &execution in replay
        .executions
        .iter()
        .filter(|execution| execution.start == step)
    {
        let segment = operation_segment(plan, execution.operation)?;
        require_unlocked(workers, execution.worker, step, segment)?;
        if !started.insert(execution) {
            return Err(FleetTwoRankStructuralClosureError::ExecutionClosure);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn submit_arrivals(
    plan: &FleetProofPlan,
    proof_generation: u64,
    step: ScheduleStep,
    replay: &ReplayIndex,
    completed_executions: &BTreeSet<ExecutionKey>,
    completed_edges: &BTreeSet<u64>,
    coordinator: &mut CoordinatorBarrierCursor,
    ready_ordinals: &mut BTreeSet<u32>,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    for &arrival in plan
        .placement()
        .barrier_arrivals
        .iter()
        .filter(|arrival| arrival.ready_step == step)
    {
        let complete = replay
            .segment_work
            .get(&(arrival.barrier_ordinal, arrival.worker))
            .is_none_or(|work| {
                work.executions.is_subset(completed_executions)
                    && work.edges.is_subset(completed_edges)
            });
        if !complete {
            return Err(FleetTwoRankStructuralClosureError::PrematureArrival {
                worker: arrival.worker,
                barrier_ordinal: arrival.barrier_ordinal,
            });
        }
        let state = coordinator.arrive(BarrierReceipt {
            plan_identity: plan.identity(),
            proof_generation,
            barrier_ordinal: arrival.barrier_ordinal,
            worker: arrival.worker,
            ready_step: arrival.ready_step,
        })?;
        if state == ArrivalState::Ready && !ready_ordinals.insert(arrival.barrier_ordinal) {
            return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
        }
    }
    Ok(())
}

fn require_unlocked(
    workers: &[WorkerBarrierCursor; 2],
    worker: WorkerId,
    step: ScheduleStep,
    segment: u32,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let cursor = workers
        .get(usize::from(worker.0))
        .ok_or(FleetTwoRankStructuralClosureError::InstallClosure { worker })?;
    if !cursor.can_start_segment(segment) {
        return Err(FleetTwoRankStructuralClosureError::WorkBeforeRelease {
            worker,
            step,
            segment,
        });
    }
    Ok(())
}

/// Synthesize one controller-only transcript phase after exact installed
/// bindings and arrivals are structurally ready. No transcript bytes run here.
fn synthesize_transcript_phase(
    plan: &FleetProofPlan,
    replay: &ReplayIndex,
    ordinal: u32,
    step: ScheduleStep,
    completed: &mut BTreeSet<u32>,
) -> Result<(), FleetTwoRankStructuralClosureError> {
    let terminal = u32::try_from(plan.barriers().len())
        .map_err(|_| FleetTwoRankStructuralClosureError::SizeOverflow)?;
    if ordinal == terminal {
        return Ok(());
    }
    let barrier = plan
        .barriers()
        .get(ordinal as usize)
        .filter(|barrier| barrier.ordinal == ordinal && barrier.release_step == step)
        .ok_or(FleetTwoRankStructuralClosureError::MissedEvent { step })?;
    if !replay.transcript_ordinals.contains(&ordinal) || !completed.insert(barrier.ordinal) {
        return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
    }
    Ok(())
}

fn release_ordinal(
    plan: &FleetProofPlan,
    step: ScheduleStep,
) -> Result<Option<u32>, FleetTwoRankStructuralClosureError> {
    let mut ordinals = plan
        .barriers()
        .iter()
        .filter(|barrier| barrier.release_step == step)
        .map(|barrier| barrier.ordinal)
        .chain((step == plan.terminal_step()).then(|| plan.barriers().len() as u32));
    let ordinal = ordinals.next();
    if ordinals.next().is_some() {
        return Err(FleetTwoRankStructuralClosureError::MissedEvent { step });
    }
    Ok(ordinal)
}

fn span(
    view: &FleetRuntimeView,
    edge: u64,
) -> Result<FleetTransferSpan, FleetTwoRankStructuralClosureError> {
    usize::try_from(edge)
        .ok()
        .and_then(|index| view.spans().get(index))
        .copied()
        .filter(|span| span.edge_ordinal == edge)
        .ok_or(FleetTwoRankStructuralClosureError::TransferClosure { edge })
}

#[derive(Debug, Eq, PartialEq)]
pub enum FleetTwoRankStructuralClosureError {
    WorkerCount {
        actual: usize,
    },
    PlanIdentityMismatch,
    Install(FleetWorkerInstallError),
    Compiled(CompiledProofError),
    RuntimeView(FleetRuntimeViewError),
    Ipc(FleetIpcCursorError),
    Barrier(FleetBarrierError),
    Plan(FleetPlanError),
    InstallClosure {
        worker: WorkerId,
    },
    IdleWorker {
        worker: WorkerId,
    },
    ExecutionClosure,
    TranscriptClosure,
    TransferClosure {
        edge: u64,
    },
    WorkBeforeRelease {
        worker: WorkerId,
        step: ScheduleStep,
        segment: u32,
    },
    PrematureArrival {
        worker: WorkerId,
        barrier_ordinal: u32,
    },
    MissedEvent {
        step: ScheduleStep,
    },
    Incomplete,
    SizeOverflow,
}

impl core::fmt::Display for FleetTwoRankStructuralClosureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "two-rank structural closure failed: {self:?}")
    }
}

impl std::error::Error for FleetTwoRankStructuralClosureError {}

macro_rules! impl_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for FleetTwoRankStructuralClosureError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_from!(FleetWorkerInstallError, Install);
impl_from!(CompiledProofError, Compiled);
impl_from!(FleetRuntimeViewError, RuntimeView);
impl_from!(FleetIpcCursorError, Ipc);
impl_from!(FleetBarrierError, Barrier);
impl_from!(FleetPlanError, Plan);
