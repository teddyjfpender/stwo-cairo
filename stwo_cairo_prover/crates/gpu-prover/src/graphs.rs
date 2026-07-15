//! Transcript-bounded CUDA graph ownership.
//!
//! A graph is tied to one proof-shape key and one stable arena allocation. The
//! captured nodes therefore never need pointer rebinding: warm proofs of the same
//! shape refill the same identity slots and replay the same executable. A shape or
//! protocol change builds a different workspace/graph entry.

use std::cell::{Cell, Ref, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use stwo_backend_cuda::{
    ArenaError, ArenaSlice, CudaExecContext, CudaExecTelemetry, CudaGraphExec, CudaRuntimeError,
    DeviceArena,
};
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

use crate::arena_plan::{ArenaBinding, ArenaPlanError, LogicalBufferId, ProofArenaPlan};
use crate::shape_executable::WorkspaceAdmission;

/// Bind one exact stable range view, truncated defensively to its logical
/// length. Distinct epoch-disjoint identities may reuse the same address while
/// retaining distinct slot ids; every resident [`ArenaBinding`] still passes
/// through here (or [`GraphWorkspace::bind`]) so kernel extents, memsets and
/// END-relative indexing observe only the declared logical requirement.
pub(crate) fn bind_arena_binding(
    arena: &DeviceArena,
    binding: ArenaBinding,
) -> Result<ArenaSlice, ArenaError> {
    Ok(arena.bind(binding.physical)?.truncated(binding.len_words))
}

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
    /// Host/runtime activity that cannot be represented by a replayable device
    /// graph occurred while the enqueue closure was being captured.
    CaptureHostActivity(CaptureHostActivity),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CaptureHostActivity {
    pub sync_calls: u64,
    pub allocations: u64,
    pub frees: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

impl CaptureHostActivity {
    fn between(before: CudaExecTelemetry, after: CudaExecTelemetry) -> Self {
        Self {
            sync_calls: after.sync_calls.saturating_sub(before.sync_calls),
            allocations: after.allocations.saturating_sub(before.allocations),
            frees: after.frees.saturating_sub(before.frees),
            h2d_bytes: after.h2d_bytes.saturating_sub(before.h2d_bytes),
            d2h_bytes: after.d2h_bytes.saturating_sub(before.d2h_bytes),
        }
    }

    fn is_empty(self) -> bool {
        self == Self::default()
    }
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
            Self::CaptureHostActivity(activity) => {
                write!(
                    f,
                    "CUDA graph capture performed host activity: {activity:?}"
                )
            }
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
        let before = arena.context().telemetry();
        let capture = arena.context().capture()?;
        if let Err(error) = enqueue(arena) {
            // Prefer the launch error. Abort is still attempted so the stream
            // cannot remain in capture mode and poison later eager work.
            let _ = capture.abort();
            return Err(GraphError::Enqueue(Box::new(error)));
        }
        let activity = CaptureHostActivity::between(before, arena.context().telemetry());
        if !activity.is_empty() {
            let _ = capture.abort();
            return Err(GraphError::CaptureHostActivity(activity));
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

    pub fn kernel_nodes(&self) -> u64 {
        self.exec.kernel_nodes()
    }

    /// Enqueue one replay. Synchronization belongs to the real transcript edge,
    /// not graph launch, so this method never blocks the host.
    pub fn replay(&self, arena: &DeviceArena) -> Result<(), GraphError> {
        let launch_base = arena.base_ptr().as_ptr() as usize;
        require_arena_identity(self.arena_base, launch_base)?;
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

/// Whether a segment was instantiated in this call or was already resident in
/// the stable workspace. A reused capture never invokes its enqueue closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphCaptureStatus {
    Captured,
    Reused,
}

impl GraphCaptureStatus {
    pub const fn is_reused(self) -> bool {
        matches!(self, Self::Reused)
    }
}

struct PersistentGraphCache<T> {
    entries: RefCell<HashMap<GraphKey, T>>,
}

impl<T> Default for PersistentGraphCache<T> {
    fn default() -> Self {
        Self {
            entries: RefCell::new(HashMap::new()),
        }
    }
}

impl<T> PersistentGraphCache<T> {
    fn contains(&self, key: GraphKey) -> bool {
        self.entries.borrow().contains_key(&key)
    }

    fn get(&self, key: GraphKey) -> Option<Ref<'_, T>> {
        Ref::filter_map(self.entries.borrow(), |entries| entries.get(&key)).ok()
    }

    fn insert(&self, key: GraphKey, value: T) {
        let replaced = self.entries.borrow_mut().insert(key, value);
        debug_assert!(replaced.is_none(), "resident graph replaced after capture");
    }

    fn capture_or_reuse<E>(
        &self,
        key: GraphKey,
        capture: impl FnOnce() -> Result<T, E>,
    ) -> Result<GraphCaptureStatus, E> {
        if self.contains(key) {
            return Ok(GraphCaptureStatus::Reused);
        }
        self.insert(key, capture()?);
        Ok(GraphCaptureStatus::Captured)
    }

    fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    fn keys(&self) -> Vec<GraphKey> {
        self.entries.borrow().keys().copied().collect()
    }
}

/// Graphs precede `arena` in declaration order, so Rust destroys all graph execs
/// before freeing the slab or its stream/pool context.
pub struct GraphWorkspace {
    graphs: PersistentGraphCache<PhaseGraph>,
    admission: WorkspaceAdmission,
    plan: Arc<ProofArenaPlan>,
    arena: DeviceArena,
    /// Fixed preprocessed coefficients, Merkle tree and transcript root are
    /// initialized as one cold-only unit per stable arena. A failed setup leaves
    /// this false, so cache reuse never trusts a partial fixed oracle.
    preprocessed_commitment_ready: Cell<bool>,
    /// Forward, inverse and quotient-subdomain twiddles are immutable for the
    /// workspace protocol key and therefore staged only once.
    fixed_twiddles_ready: Cell<bool>,
}

impl GraphWorkspace {
    pub(crate) fn from_plan(
        context: CudaExecContext,
        plan: Arc<ProofArenaPlan>,
        admission: WorkspaceAdmission,
    ) -> Result<Self, GraphError> {
        let arena = plan.allocate(context)?;
        Ok(Self {
            graphs: PersistentGraphCache::default(),
            admission,
            plan,
            arena,
            preprocessed_commitment_ready: Cell::new(false),
            fixed_twiddles_ready: Cell::new(false),
        })
    }

    pub fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    pub fn plan(&self) -> &ProofArenaPlan {
        &self.plan
    }

    pub fn admission(&self) -> &WorkspaceAdmission {
        &self.admission
    }

    pub fn preprocessed_commitment_ready(&self) -> bool {
        self.preprocessed_commitment_ready.get()
    }

    /// Mark the fixed oracle reusable only after coefficient interpolation,
    /// commitment and the root handoff have all synchronized successfully.
    pub(crate) fn mark_preprocessed_commitment_ready(&self) {
        self.preprocessed_commitment_ready.set(true);
    }

    pub fn fixed_twiddles_ready(&self) -> bool {
        self.fixed_twiddles_ready.get()
    }

    pub(crate) fn mark_fixed_twiddles_ready(&self) {
        self.fixed_twiddles_ready.set(true);
    }

    pub fn key(&self, segment: GraphSegment) -> GraphKey {
        GraphKey {
            shape: self.plan.shape_key,
            protocol_key: self.plan.protocol_key,
            segment,
        }
    }

    /// Bind a logical identity to its exact stable range view. Epoch-disjoint
    /// identities may reuse an address without sharing slot identity; the
    /// defensive truncation keeps `slice.len_words()` equal to the declared
    /// logical length.
    pub fn bind(&self, logical: LogicalBufferId) -> Result<(ArenaSlice, usize), GraphError> {
        let binding = self
            .plan
            .binding(logical)
            .ok_or(ArenaPlanError::MissingBinding(logical))?;
        Ok((bind_arena_binding(&self.arena, binding)?, binding.len_words))
    }

    pub fn graph(&self, key: GraphKey) -> Option<Ref<'_, PhaseGraph>> {
        if self.owns_key(key) {
            self.graphs.get(key)
        } else {
            None
        }
    }

    pub fn graph_segment(&self, segment: GraphSegment) -> Option<Ref<'_, PhaseGraph>> {
        self.graph(self.key(segment))
    }

    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    pub(crate) fn captured_graph_keys(&self) -> Vec<GraphKey> {
        self.graphs.keys()
    }

    pub fn graph_kernel_node_count(&self) -> Option<u64> {
        self.graphs
            .entries
            .borrow()
            .values()
            .try_fold(0u64, |total, graph| total.checked_add(graph.kernel_nodes()))
    }

    pub(crate) fn capture<E>(
        &self,
        key: GraphKey,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<GraphCaptureStatus, GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        require_workspace_key(self.plan.shape_key, self.plan.protocol_key, key)?;
        self.graphs
            .capture_or_reuse(key, || PhaseGraph::capture(key, &self.arena, enqueue))
    }

    pub(crate) fn capture_segment<E>(
        &self,
        segment: GraphSegment,
        enqueue: impl FnOnce(&DeviceArena) -> Result<(), E>,
    ) -> Result<GraphCaptureStatus, GraphError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.capture(self.key(segment), enqueue)
    }

    pub fn replay_segment(&self, segment: GraphSegment) -> Result<(), GraphError> {
        let key = self.key(segment);
        let graph = self
            .graphs
            .get(key)
            .ok_or(GraphError::MissingSegment(segment))?;
        graph.replay(&self.arena)
    }

    fn owns_key(&self, key: GraphKey) -> bool {
        key.shape == self.plan.shape_key && key.protocol_key == self.plan.protocol_key
    }
}

fn require_workspace_key(
    expected_shape: ProofShapeKey,
    expected_protocol: u64,
    key: GraphKey,
) -> Result<(), GraphError> {
    if key.shape != expected_shape || key.protocol_key != expected_protocol {
        return Err(GraphError::WorkspaceKeyMismatch {
            expected_shape,
            expected_protocol,
            actual_shape: key.shape,
            actual_protocol: key.protocol_key,
        });
    }
    Ok(())
}

fn require_arena_identity(captured_base: usize, launch_base: usize) -> Result<(), GraphError> {
    if launch_base != captured_base {
        return Err(GraphError::ArenaIdentityMismatch {
            captured_base,
            launch_base,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_host_activity_rejects_only_non_replayable_runtime_work() {
        let before = CudaExecTelemetry {
            memset_bytes: 16,
            d2d_bytes: 32,
            ..CudaExecTelemetry::default()
        };
        let device_only = CudaExecTelemetry {
            memset_bytes: 64,
            d2d_bytes: 128,
            capture_begins: 1,
            kernel_launches: 3,
            ..before
        };
        assert!(CaptureHostActivity::between(before, device_only).is_empty());

        let host_activity = CudaExecTelemetry {
            sync_calls: 1,
            allocations: 2,
            frees: 1,
            h2d_bytes: 64,
            d2h_bytes: 128,
            ..device_only
        };
        assert_eq!(
            CaptureHostActivity::between(device_only, host_activity),
            CaptureHostActivity {
                sync_calls: 1,
                allocations: 2,
                frees: 1,
                h2d_bytes: 64,
                d2h_bytes: 128,
            }
        );
    }

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

    #[test]
    fn second_session_sees_the_first_sessions_resident_graph() {
        let key = GraphKey {
            shape: ProofShapeKey(7),
            protocol_key: 11,
            segment: GraphSegment::InteractionCommit,
        };
        let cache = PersistentGraphCache::default();
        let captures = std::cell::Cell::new(0);
        assert_eq!(
            cache
                .capture_or_reuse(key, || {
                    captures.set(captures.get() + 1);
                    Ok::<_, std::convert::Infallible>("first-session-executable")
                })
                .unwrap(),
            GraphCaptureStatus::Captured
        );
        assert_eq!(
            cache
                .capture_or_reuse(key, || {
                    captures.set(captures.get() + 1);
                    Ok::<_, std::convert::Infallible>("second-session-executable")
                })
                .unwrap(),
            GraphCaptureStatus::Reused
        );
        assert_eq!(captures.get(), 1, "warm session recaptured the graph");
        assert_eq!(*cache.get(key).unwrap(), "first-session-executable");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn protocol_and_arena_mismatches_fail_closed() {
        let key = GraphKey {
            shape: ProofShapeKey(7),
            protocol_key: 12,
            segment: GraphSegment::InteractionCommit,
        };
        assert!(matches!(
            require_workspace_key(ProofShapeKey(7), 11, key),
            Err(GraphError::WorkspaceKeyMismatch {
                expected_protocol: 11,
                actual_protocol: 12,
                ..
            })
        ));
        assert!(matches!(
            require_arena_identity(0x1000, 0x2000),
            Err(GraphError::ArenaIdentityMismatch {
                captured_base: 0x1000,
                launch_base: 0x2000,
            })
        ));
    }
}
