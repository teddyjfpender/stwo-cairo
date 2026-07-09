//! Transcript-bounded CUDA graph ownership.
//!
//! A graph is tied to one proof-shape key and one stable arena allocation. The
//! captured nodes therefore never need pointer rebinding: warm proofs of the same
//! shape refill the same identity slots and replay the same executable. A shape or
//! protocol change builds a different workspace/graph entry.

use std::collections::HashMap;
use std::sync::Arc;

use stwo_backend_cuda::{
    ArenaError, ArenaSlice, CudaExecContext, CudaGraphExec, CudaRuntimeError, DeviceArena,
};
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

use crate::arena_plan::{ArenaPlanError, LogicalBufferId, ProofArenaPlan};

/// True Fiat-Shamir boundaries in the Cairo proof protocol. `FriLayer` is keyed
/// per layer because each root determines the next folding challenge.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphSegment {
    IngestWitnessBaseCommit,
    InteractionCommit,
    CompositionQuotientCommit,
    OodsEvaluation,
    FriLayer(u8),
    OodsQueriesDecommitAssemble,
}

/// Cache identity. `protocol_key` is supplied by the proof planner and must cover
/// every graph-topology parameter not already present in the component shape
/// (PCS/FRI configuration, channel variant, and opening strategy).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphKey {
    pub shape: ProofShapeKey,
    pub protocol_key: u64,
    pub segment: GraphSegment,
}

#[derive(Debug)]
pub enum GraphError {
    Runtime(CudaRuntimeError),
    Plan(ArenaPlanError),
    Arena(ArenaError),
    Enqueue(Box<dyn std::error::Error + Send + Sync>),
    MissingSegment(GraphSegment),
    WorkspaceKeyMismatch {
        expected_shape: ProofShapeKey,
        expected_protocol: u64,
        actual_shape: ProofShapeKey,
        actual_protocol: u64,
    },
    /// Captured kernel arguments point into a different stable arena slab.
    ArenaIdentityMismatch {
        captured_base: usize,
        launch_base: usize,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Runtime(error) => write!(f, "CUDA graph runtime error: {error}"),
            Self::Plan(error) => write!(f, "CUDA graph arena-plan error: {error}"),
            Self::Arena(error) => write!(f, "CUDA graph arena error: {error}"),
            Self::Enqueue(error) => write!(f, "CUDA graph segment enqueue failed: {error}"),
            Self::MissingSegment(segment) => {
                write!(f, "CUDA graph segment {segment:?} is not captured")
            }
            Self::WorkspaceKeyMismatch {
                expected_shape,
                expected_protocol,
                actual_shape,
                actual_protocol,
            } => write!(
                f,
                "CUDA graph key does not belong to workspace: expected shape={expected_shape:?} \
                 protocol={expected_protocol:#x}, got shape={actual_shape:?} \
                 protocol={actual_protocol:#x}"
            ),
            Self::ArenaIdentityMismatch {
                captured_base,
                launch_base,
            } => write!(
                f,
                "CUDA graph arena changed: captured=0x{captured_base:x}, launch=0x{launch_base:x}"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<CudaRuntimeError> for GraphError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<ArenaPlanError> for GraphError {
    fn from(value: ArenaPlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<ArenaError> for GraphError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

/// One instantiated segment. This field must be dropped before the arena whose
/// addresses it captured; [`GraphWorkspace`] enforces that order structurally.
pub struct PhaseGraph {
    key: GraphKey,
    arena_base: usize,
    exec: CudaGraphExec,
}

impl PhaseGraph {
    /// Capture the exact same allocation-free enqueue sequence used by eager mode.
    /// The closure may only call explicit-stream, arena-backed launch wrappers.
    pub fn capture<E>(
        key: GraphKey,
        arena: &DeviceArena,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<Self, GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let capture = arena.context().capture()?;
        if let Err(error) = enqueue(arena) {
            // Prefer the launch error. Abort is still attempted so the stream
            // cannot remain in capture mode and poison later eager work.
            let _ = capture.abort();
            return Err(GraphError::Enqueue(Box::new(error)));
        }
        let exec = capture.finish()?;
        Ok(Self {
            key,
            arena_base: arena.base_ptr().as_ptr() as usize,
            exec,
        })
    }

    pub const fn key(&self) -> GraphKey {
        self.key
    }

    /// Enqueue one replay. Synchronization belongs to the real transcript edge,
    /// not graph launch, so this method never blocks the host.
    pub fn replay(&self, arena: &DeviceArena) -> Result<(), GraphError> {
        let launch_base = arena.base_ptr().as_ptr() as usize;
        if launch_base != self.arena_base {
            return Err(GraphError::ArenaIdentityMismatch {
                captured_base: self.arena_base,
                launch_base,
            });
        }
        self.exec.launch(arena.context())?;
        Ok(())
    }

    /// Debug/reference path: enqueue the segment directly on the same context and
    /// slots. Whole-proof parity compares this with [`Self::replay`].
    pub fn eager<E>(
        arena: &DeviceArena,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<(), GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        enqueue(arena).map_err(|error| GraphError::Enqueue(Box::new(error)))?;
        Ok(())
    }
}

/// Graphs precede `arena` in declaration order, so Rust destroys all graph execs
/// before freeing the slab or its stream/pool context.
pub struct GraphWorkspace {
    graphs: HashMap<GraphKey, PhaseGraph>,
    plan: Arc<ProofArenaPlan>,
    arena: DeviceArena,
}

impl GraphWorkspace {
    pub fn from_plan(
        context: CudaExecContext,
        plan: Arc<ProofArenaPlan>,
    ) -> Result<Self, GraphError> {
        let arena = plan.allocate(context)?;
        Ok(Self {
            graphs: HashMap::new(),
            plan,
            arena,
        })
    }

    pub fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    pub fn plan(&self) -> &ProofArenaPlan {
        &self.plan
    }

    pub fn key(&self, segment: GraphSegment) -> GraphKey {
        GraphKey {
            shape: self.plan.shape_key,
            protocol_key: self.plan.protocol_key,
            segment,
        }
    }

    /// Bind a logical identity to its stable physical slot. The returned length
    /// is the logical capacity; the underlying slice may be larger because a
    /// disjoint-lifetime buffer reuses the same physical range.
    pub fn bind(&self, logical: LogicalBufferId) -> Result<(ArenaSlice, usize), GraphError> {
        let binding = self
            .plan
            .binding(logical)
            .ok_or(ArenaPlanError::MissingBinding(logical))?;
        Ok((self.arena.bind(binding.physical)?, binding.len_words))
    }

    pub fn graph(&self, key: GraphKey) -> Option<&PhaseGraph> {
        if self.owns_key(key) {
            self.graphs.get(&key)
        } else {
            None
        }
    }

    pub fn graph_segment(&self, segment: GraphSegment) -> Option<&PhaseGraph> {
        self.graph(self.key(segment))
    }

    pub fn capture<E>(
        &mut self,
        key: GraphKey,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<&PhaseGraph, GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        if !self.owns_key(key) {
            return Err(GraphError::WorkspaceKeyMismatch {
                expected_shape: self.plan.shape_key,
                expected_protocol: self.plan.protocol_key,
                actual_shape: key.shape,
                actual_protocol: key.protocol_key,
            });
        }
        let graph = PhaseGraph::capture(key, &self.arena, enqueue)?;
        self.graphs.insert(key, graph);
        Ok(self.graphs.get(&key).expect("graph inserted"))
    }

    pub fn capture_segment<E>(
        &mut self,
        segment: GraphSegment,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<&PhaseGraph, GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.capture(self.key(segment), enqueue)
    }

    pub fn replay_segment(&self, segment: GraphSegment) -> Result<(), GraphError> {
        let key = self.key(segment);
        let graph = self
            .graphs
            .get(&key)
            .ok_or(GraphError::MissingSegment(segment))?;
        graph.replay(&self.arena)
    }

    fn owns_key(&self, key: GraphKey) -> bool {
        key.shape == self.plan.shape_key && key.protocol_key == self.plan.protocol_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_key_separates_shape_protocol_and_transcript_segment() {
        let base = GraphKey {
            shape: ProofShapeKey(7),
            protocol_key: 11,
            segment: GraphSegment::InteractionCommit,
        };
        let mut keys = std::collections::HashSet::from([base]);
        keys.insert(GraphKey {
            shape: ProofShapeKey(8),
            ..base
        });
        keys.insert(GraphKey {
            protocol_key: 12,
            ..base
        });
        keys.insert(GraphKey {
            segment: GraphSegment::FriLayer(0),
            ..base
        });
        assert_eq!(keys.len(), 4);
    }
}
