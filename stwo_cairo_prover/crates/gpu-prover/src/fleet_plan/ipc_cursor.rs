//! Pure coordinator cursor for one proof attempt's CUDA-IPC enqueue receipts.
//!
//! The cursor owns no device address or transport. It admits only the exact
//! plan-derived edge identity and preserves equal-start-step work as a wave.
//! Receipts are ordering metadata, not authenticated DMA-completion evidence.
//! The installed runtime must bind each phase to the exact device, IPC key and
//! completion event for its plan-derived byte operation before submitting it.

use std::collections::BTreeMap;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetIpcPhase {
    Published,
    Consumed,
    Reclaimed,
    Armed,
}

impl FleetIpcPhase {
    const fn next(self) -> Option<Self> {
        match self {
            Self::Published => Some(Self::Consumed),
            Self::Consumed => Some(Self::Reclaimed),
            Self::Reclaimed => Some(Self::Armed),
            Self::Armed => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetIpcPhaseBinding {
    pub plan_identity: [u8; 32],
    pub proof_generation: u64,
    pub transition: LayoutTransitionId,
    pub span_ordinal: u32,
    pub edge_ordinal: u64,
    pub owner: WorkerId,
    pub peer: WorkerId,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetIpcPhaseReceipt {
    pub binding: FleetIpcPhaseBinding,
    pub phase: FleetIpcPhase,
    pub worker: WorkerId,
}

impl FleetIpcPhaseReceipt {
    pub fn for_span(
        plan_identity: [u8; 32],
        proof_generation: u64,
        span: FleetTransferSpan,
        phase: FleetIpcPhase,
    ) -> Result<Self, FleetIpcCursorError> {
        let generation = if phase == FleetIpcPhase::Armed {
            proof_generation
                .checked_add(1)
                .ok_or(FleetIpcCursorError::GenerationOverflow)?
        } else {
            proof_generation
        };
        let worker = match phase {
            FleetIpcPhase::Published | FleetIpcPhase::Reclaimed => span.owner,
            FleetIpcPhase::Consumed | FleetIpcPhase::Armed => span.peer,
        };
        Ok(Self {
            binding: FleetIpcPhaseBinding {
                plan_identity,
                proof_generation,
                transition: span.transition,
                span_ordinal: span.span_ordinal,
                edge_ordinal: span.edge_ordinal,
                owner: span.owner,
                peer: span.peer,
                generation,
            },
            phase,
            worker,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetIpcAttemptState {
    Running { wave_step: ScheduleStep },
    Complete,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetIpcCursorProgress {
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
    NonDenseEdge {
        expected: u64,
        actual: u64,
    },
    WrongAttempt,
    UnknownEdge(u64),
    WrongEdgeIdentity(u64),
    WrongWorker {
        expected: WorkerId,
        actual: WorkerId,
    },
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
        write!(f, "invalid fleet IPC phase receipt: {self:?}")
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

/// Fail-closed coordinator state for one proof attempt.
pub struct FleetIpcCoordinatorCursor {
    plan_identity: [u8; 32],
    proof_generation: u64,
    state: FleetIpcAttemptState,
    edges: Vec<EdgeCursor>,
    waves: Vec<IpcWave>,
    active_wave: usize,
}

impl FleetIpcCoordinatorCursor {
    pub fn new(
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
            plan_identity: view.plan_identity(),
            proof_generation,
            state,
            edges,
            waves,
            active_wave: 0,
        })
    }

    pub const fn state(&self) -> FleetIpcAttemptState {
        self.state
    }

    pub fn active_wave_edges(&self) -> &[u64] {
        self.waves
            .get(self.active_wave)
            .map_or(&[], |wave| wave.edges.as_slice())
    }

    pub fn accept(
        &mut self,
        receipt: FleetIpcPhaseReceipt,
    ) -> Result<FleetIpcCursorProgress, FleetIpcCursorError> {
        match self.state {
            FleetIpcAttemptState::Poisoned => return Err(FleetIpcCursorError::AttemptPoisoned),
            FleetIpcAttemptState::Complete => {
                return self.poison(FleetIpcCursorError::AttemptComplete)
            }
            FleetIpcAttemptState::Running { .. } => {}
        }
        if receipt.binding.plan_identity != self.plan_identity
            || receipt.binding.proof_generation != self.proof_generation
        {
            return self.poison(FleetIpcCursorError::WrongAttempt);
        }
        let edge_index = match usize::try_from(receipt.binding.edge_ordinal) {
            Ok(index) if index < self.edges.len() => index,
            _ => {
                return self.poison(FleetIpcCursorError::UnknownEdge(
                    receipt.binding.edge_ordinal,
                ))
            }
        };
        let active = &self.waves[self.active_wave];
        if !active.edges.contains(&receipt.binding.edge_ordinal) {
            return self.poison(FleetIpcCursorError::WrongWave {
                edge: receipt.binding.edge_ordinal,
                active_step: active.step,
            });
        }

        let edge = &self.edges[edge_index];
        if receipt.binding.transition != edge.span.transition
            || receipt.binding.span_ordinal != edge.span.span_ordinal
            || receipt.binding.edge_ordinal != edge.span.edge_ordinal
            || receipt.binding.owner != edge.span.owner
            || receipt.binding.peer != edge.span.peer
        {
            return self.poison(FleetIpcCursorError::WrongEdgeIdentity(
                receipt.binding.edge_ordinal,
            ));
        }
        if edge.next != Some(receipt.phase) {
            return self.poison(FleetIpcCursorError::OutOfOrderPhase {
                edge: receipt.binding.edge_ordinal,
                expected: edge.next,
                actual: receipt.phase,
            });
        }
        let expected = FleetIpcPhaseReceipt::for_span(
            self.plan_identity,
            self.proof_generation,
            edge.span,
            receipt.phase,
        )?;
        if receipt.binding.generation != expected.binding.generation {
            return self.poison(FleetIpcCursorError::GenerationMismatch {
                expected: expected.binding.generation,
                actual: receipt.binding.generation,
            });
        }
        if receipt.worker != expected.worker {
            return self.poison(FleetIpcCursorError::WrongWorker {
                expected: expected.worker,
                actual: receipt.worker,
            });
        }

        self.edges[edge_index].next = receipt.phase.next();
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
