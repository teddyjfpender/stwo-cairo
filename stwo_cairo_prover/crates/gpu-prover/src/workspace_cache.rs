//! Bounded ownership for warm, shape/protocol-exact graph workspaces.
//!
//! Entries are never evicted implicitly: captured graphs retain pointers into
//! their arena, so dropping an entry is always an explicit caller decision.

use std::cell::Cell;
use std::pin::Pin;
use std::sync::Arc;

use stwo_backend_cuda::{CudaExecContext, CudaRuntimeError};
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

use crate::arena_plan::ProofArenaPlan;
use crate::graphs::{GraphError, GraphWorkspace};
use crate::resident_runtime::{
    ResidentGraphRuntime, ResidentRuntimeError, ResidentWorkspaceIdentity,
};
use crate::shape_executable::{ShapeExecutable, WorkspaceAdmission};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkspaceKey {
    pub shape: ProofShapeKey,
    pub protocol_key: u64,
}

impl WorkspaceKey {
    pub const fn new(shape: ProofShapeKey, protocol_key: u64) -> Self {
        Self {
            shape,
            protocol_key,
        }
    }

    pub fn from_plan(plan: &ProofArenaPlan) -> Self {
        Self::new(plan.shape_key, plan.protocol_key)
    }

    pub fn from_workspace(workspace: &GraphWorkspace) -> Self {
        workspace.admission().workspace_key()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceCacheTelemetry {
    pub hits: u64,
    pub misses: u64,
    pub materializations: u64,
    pub capacity_rejections: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceMaterialization {
    Reused,
    Materialized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedRuntimeMaterialization {
    Reused,
    Materialized,
}

#[derive(Debug)]
pub enum WorkspaceCacheError {
    ZeroCapacity,
    Occupied(WorkspaceKey),
    AdmissionPlanMismatch {
        admission: WorkspaceKey,
        plan: WorkspaceKey,
    },
    AtCapacity {
        capacity: usize,
        requested: WorkspaceKey,
    },
    PreparedRuntimeInstalled(WorkspaceKey),
    Runtime(CudaRuntimeError),
    Graph(GraphError),
}

impl std::fmt::Display for WorkspaceCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "workspace cache capacity must be non-zero"),
            Self::Occupied(key) => write!(f, "workspace cache key is already occupied: {key:?}"),
            Self::AdmissionPlanMismatch { admission, plan } => write!(
                f,
                "workspace admission does not match the requested arena plan: admission={admission:?} plan={plan:?}"
            ),
            Self::AtCapacity {
                capacity,
                requested,
            } => write!(
                f,
                "workspace cache capacity {capacity} reached; refusing to evict captured graphs \
                 for {requested:?}"
            ),
            Self::PreparedRuntimeInstalled(key) => write!(
                f,
                "workspace {key:?} has a prepared runtime and cannot be mutably detached"
            ),
            Self::Runtime(error) => write!(f, "workspace CUDA context error: {error}"),
            Self::Graph(error) => write!(f, "workspace materialization error: {error}"),
        }
    }
}

/// Stable self-referential owner for one prepared runtime.
///
/// `runtime` is declared before `workspace`, so Rust drops every borrowed
/// prepared graph before the pinned workspace and its arena. The workspace is
/// heap-pinned before its address is extended; no API exposes mutable access or
/// permits removal after a runtime is installed.
pub(crate) struct ResidentWorkspaceEntry {
    runtime: Option<ResidentGraphRuntime<'static>>,
    workspace: Pin<Box<GraphWorkspace>>,
    runtime_poisoned: bool,
}

/// Armed mutable access to one cached runtime. [`ResidentRuntimeLease::with_runtime`]
/// disarms internally only after a successful callback and identity check; any
/// ordinary error or unwind drops the armed lease and poisons the entry.
#[must_use = "dropping an armed runtime lease poisons the cached runtime"]
pub(crate) struct ResidentRuntimeLease<'entry> {
    entry: &'entry mut ResidentWorkspaceEntry,
    armed: bool,
}

impl ResidentWorkspaceEntry {
    fn new(workspace: GraphWorkspace) -> Self {
        Self {
            runtime: None,
            workspace: Box::pin(workspace),
            runtime_poisoned: false,
        }
    }

    pub(crate) fn workspace(&self) -> &GraphWorkspace {
        self.workspace.as_ref().get_ref()
    }

    fn workspace_mut(&mut self) -> Option<&mut GraphWorkspace> {
        (self.runtime.is_none() && !self.runtime_poisoned)
            .then(|| self.workspace.as_mut().get_mut())
    }

    fn observable_workspace(&self) -> Option<&GraphWorkspace> {
        self.can_detach().then(|| self.workspace())
    }

    fn observable_workspace_mut(&mut self) -> Option<&mut GraphWorkspace> {
        self.workspace_mut()
    }

    fn prepare_runtime<F>(
        &mut self,
        prepare: F,
    ) -> Result<PreparedRuntimeMaterialization, ResidentRuntimeError>
    where
        F: for<'workspace> FnOnce(
            &'workspace GraphWorkspace,
        )
            -> Result<ResidentGraphRuntime<'workspace>, ResidentRuntimeError>,
    {
        if self.runtime_poisoned {
            return Err(ResidentRuntimeError::PersistentRuntimePoisoned);
        }
        let materialization = if self.runtime.is_some() {
            self.validate_installed_runtime_identity()?;
            PreparedRuntimeMaterialization::Reused
        } else {
            // Borrow the pinned workspace field directly. Borrowing through
            // `self.workspace()` would conservatively borrow the whole entry
            // for the lifetime carried by the returned runtime, preventing
            // the disjoint runtime/poison fields from being updated.
            let workspace = self.workspace.as_ref().get_ref();
            let expected = ResidentWorkspaceIdentity::of(workspace);
            let runtime = prepare(workspace);
            self.runtime = match runtime {
                Ok(runtime) => {
                    if let Err(error) = require_runtime_identity(expected, runtime.identity()) {
                        self.runtime_poisoned = true;
                        return Err(error);
                    }
                    // SAFETY: the caller never observes an extended workspace
                    // reference: the HRTB above supplies only a borrow tied to
                    // its call. The workspace is heap-pinned, runtime is
                    // private and dropped first, and all detach/mutable APIs
                    // fail once installation begins.
                    Some(unsafe {
                        core::mem::transmute::<
                            ResidentGraphRuntime<'_>,
                            ResidentGraphRuntime<'static>,
                        >(runtime)
                    })
                }
                Err(error) => {
                    // Preparation may already have uploaded descriptors or
                    // captured arena addresses. Never retry over a partially
                    // initialized workspace.
                    self.runtime_poisoned = true;
                    return Err(error);
                }
            };
            PreparedRuntimeMaterialization::Materialized
        };
        Ok(materialization)
    }

    fn validate_installed_runtime_identity(&mut self) -> Result<(), ResidentRuntimeError> {
        let expected = ResidentWorkspaceIdentity::of(self.workspace.as_ref().get_ref());
        let actual = self
            .runtime
            .as_ref()
            .expect("runtime was materialized before identity validation")
            .identity();
        if let Err(error) = require_runtime_identity(expected, actual) {
            self.runtime_poisoned = true;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn poison_runtime(&mut self) {
        self.runtime_poisoned = true;
    }

    pub(crate) fn arm_runtime_lease(
        &mut self,
    ) -> Result<ResidentRuntimeLease<'_>, ResidentRuntimeError> {
        if self.runtime_poisoned {
            return Err(ResidentRuntimeError::PersistentRuntimePoisoned);
        }
        if self.runtime.is_some() {
            self.validate_installed_runtime_identity()?;
        }
        Ok(ResidentRuntimeLease {
            entry: self,
            armed: true,
        })
    }

    fn can_detach(&self) -> bool {
        self.runtime.is_none() && !self.runtime_poisoned
    }

    fn into_workspace(self) -> GraphWorkspace {
        assert!(
            self.can_detach(),
            "installed or poisoned resident runtime cannot release its workspace"
        );
        *Pin::into_inner(self.workspace)
    }
}

impl ResidentRuntimeLease<'_> {
    pub(crate) fn workspace(&self) -> &GraphWorkspace {
        self.entry.workspace()
    }

    pub(crate) fn with_runtime<P, F, R, E>(mut self, prepare: P, use_runtime: F) -> Result<R, E>
    where
        E: From<ResidentRuntimeError>,
        P: for<'workspace> FnOnce(
            &'workspace GraphWorkspace,
        )
            -> Result<ResidentGraphRuntime<'workspace>, ResidentRuntimeError>,
        F: for<'scope> FnOnce(
            &'scope mut ResidentGraphRuntime<'scope>,
            PreparedRuntimeMaterialization,
        ) -> Result<R, E>,
    {
        let materialization = self.entry.prepare_runtime(prepare).map_err(E::from)?;
        self.entry
            .validate_installed_runtime_identity()
            .map_err(E::from)?;
        let runtime = self
            .entry
            .runtime
            .as_mut()
            .expect("runtime was reused or materialized");
        // SAFETY: `ResidentWorkspaceEntry` pins and outlives the stored runtime,
        // which is dropped before the workspace. This cast only shortens the
        // private storage lifetime for one HRTB-scoped callback. `R` is outside
        // that binder, so neither the runtime nor a reference derived from it
        // can escape in safe code. Runtime mutators must never retain a
        // callback-scoped borrow in the stored runtime.
        let runtime = unsafe {
            &mut *(runtime as *mut ResidentGraphRuntime<'static> as *mut ResidentGraphRuntime<'_>)
        };
        let result = use_runtime(runtime, materialization);
        if let Err(error) = self.entry.validate_installed_runtime_identity() {
            // A callback must never be able to leave a runtime rooted in any
            // owner other than this pinned workspace. Drop a replaced runtime
            // now, while the callback's scope is still active, and leave the
            // entry permanently poisoned.
            self.entry.runtime_poisoned = true;
            drop(self.entry.runtime.take());
            return Err(E::from(error));
        }
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

fn require_runtime_identity(
    expected: ResidentWorkspaceIdentity,
    actual: ResidentWorkspaceIdentity,
) -> Result<(), ResidentRuntimeError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ResidentRuntimeError::WorkspaceIdentityMismatch { expected, actual })
    }
}

impl Drop for ResidentRuntimeLease<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.entry.poison_runtime();
        }
    }
}

impl std::error::Error for WorkspaceCacheError {}

impl From<CudaRuntimeError> for WorkspaceCacheError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<GraphError> for WorkspaceCacheError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

struct CacheEntry<K, V> {
    admission: K,
    key: WorkspaceKey,
    value: V,
}

/// Generic exact-equality core keeps admission behavior testable without CUDA.
/// Production stores boxed workspaces so vector growth never moves the
/// host-side workspace object.
struct BoundedCache<K, V> {
    capacity: usize,
    entries: Vec<CacheEntry<K, V>>,
    telemetry: Cell<WorkspaceCacheTelemetry>,
}

impl<K: Eq, V> BoundedCache<K, V> {
    fn new(capacity: usize) -> Result<Self, WorkspaceCacheError> {
        if capacity == 0 {
            return Err(WorkspaceCacheError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            telemetry: Cell::new(WorkspaceCacheTelemetry::default()),
        })
    }

    fn exact_index(&self, admission: &K) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| &entry.admission == admission)
    }

    fn unique_key_index(&self, key: WorkspaceKey) -> Option<usize> {
        let mut indices = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| (entry.key == key).then_some(index));
        let index = indices.next()?;
        indices.next().is_none().then_some(index)
    }

    fn get_by_key(&self, key: WorkspaceKey) -> Option<&V> {
        let index = self.unique_key_index(key);
        self.record_lookup(index.is_some());
        index.map(|index| &self.entries[index].value)
    }

    fn get_by_key_mut(&mut self, key: WorkspaceKey) -> Option<&mut V> {
        let index = self.unique_key_index(key);
        self.record_lookup(index.is_some());
        index.map(|index| &mut self.entries[index].value)
    }

    fn insert(
        &mut self,
        admission: K,
        key: WorkspaceKey,
        value: V,
    ) -> Result<(), WorkspaceCacheError> {
        self.admit(&admission, key)?;
        self.entries.push(CacheEntry {
            admission,
            key,
            value,
        });
        Ok(())
    }

    fn admit(&self, admission: &K, key: WorkspaceKey) -> Result<(), WorkspaceCacheError> {
        if self.exact_index(admission).is_some() {
            return Err(WorkspaceCacheError::Occupied(key));
        }
        if self.entries.len() >= self.capacity {
            self.update_telemetry(|telemetry| telemetry.capacity_rejections += 1);
            return Err(WorkspaceCacheError::AtCapacity {
                capacity: self.capacity,
                requested: key,
            });
        }
        Ok(())
    }

    #[cfg(test)]
    fn take_by_key(&mut self, key: WorkspaceKey) -> Option<V> {
        let index = self.unique_key_index(key)?;
        Some(self.entries.remove(index).value)
    }

    fn only(&self) -> Option<&V> {
        if self.entries.len() == 1 {
            Some(&self.entries[0].value)
        } else {
            None
        }
    }

    #[cfg(test)]
    fn take_only(&mut self) -> Option<V> {
        if self.entries.len() != 1 {
            return None;
        }
        Some(self.entries.remove(0).value)
    }

    fn record_lookup(&self, hit: bool) {
        self.update_telemetry(|telemetry| {
            if hit {
                telemetry.hits += 1;
            } else {
                telemetry.misses += 1;
            }
        });
    }

    fn record_materialization(&self) {
        self.update_telemetry(|telemetry| telemetry.materializations += 1);
    }

    fn update_telemetry(&self, update: impl FnOnce(&mut WorkspaceCacheTelemetry)) {
        let mut telemetry = self.telemetry.get();
        update(&mut telemetry);
        self.telemetry.set(telemetry);
    }
}

/// One cache per prover/device. Each miss creates a fresh CUDA execution context
/// and transfers it into exactly one stable graph workspace.
pub struct WorkspaceCache {
    inner: BoundedCache<WorkspaceAdmission, ResidentWorkspaceEntry>,
}

impl WorkspaceCache {
    pub fn new(capacity: usize) -> Result<Self, WorkspaceCacheError> {
        Ok(Self {
            inner: BoundedCache::new(capacity)?,
        })
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.entries.is_empty()
    }

    pub fn telemetry(&self) -> WorkspaceCacheTelemetry {
        self.inner.telemetry.get()
    }

    /// Diagnostic short-key lookup. Returns `None` when distinct exact
    /// admissions intentionally collide on the same short key or the entry has
    /// installed/poisoned its persistent runtime.
    pub fn get(&self, key: WorkspaceKey) -> Option<&GraphWorkspace> {
        self.inner
            .get_by_key(key)
            .and_then(ResidentWorkspaceEntry::observable_workspace)
    }

    /// Mutable diagnostic short-key lookup with the same ambiguity and runtime
    /// ownership rules as [`Self::get`]. Production session admission never
    /// uses this path.
    pub fn get_mut(&mut self, key: WorkspaceKey) -> Option<&mut GraphWorkspace> {
        self.inner
            .get_by_key_mut(key)
            .and_then(ResidentWorkspaceEntry::observable_workspace_mut)
    }

    pub fn materialize_or_reuse(
        &mut self,
        executable: &ShapeExecutable,
    ) -> Result<(&mut GraphWorkspace, WorkspaceMaterialization), WorkspaceCacheError> {
        let key = executable.workspace_admission().workspace_key();
        let (entry, materialization) = self.materialize_entry_or_reuse(executable)?;
        let workspace = entry
            .workspace_mut()
            .ok_or(WorkspaceCacheError::PreparedRuntimeInstalled(key))?;
        Ok((workspace, materialization))
    }

    pub(crate) fn materialize_entry_or_reuse(
        &mut self,
        executable: &ShapeExecutable,
    ) -> Result<(&mut ResidentWorkspaceEntry, WorkspaceMaterialization), WorkspaceCacheError> {
        let admission = executable.workspace_admission();
        let plan = Arc::clone(executable.arena());
        let key = WorkspaceKey::from_plan(&plan);
        if !admission.matches_plan(&plan) {
            return Err(WorkspaceCacheError::AdmissionPlanMismatch {
                admission: admission.workspace_key(),
                plan: key,
            });
        }
        let (index, materialization) = if let Some(index) = self.inner.exact_index(admission) {
            self.inner.record_lookup(true);
            (index, WorkspaceMaterialization::Reused)
        } else {
            self.inner.record_lookup(false);
            self.inner.admit(admission, key)?;
            let context = CudaExecContext::new()?;
            let workspace = GraphWorkspace::from_plan(context, plan, admission.clone())?;
            self.inner.insert(
                admission.clone(),
                key,
                ResidentWorkspaceEntry::new(workspace),
            )?;
            self.inner.record_materialization();
            (
                self.inner
                    .exact_index(admission)
                    .expect("materialized exact admission inserted"),
                WorkspaceMaterialization::Materialized,
            )
        };
        Ok((&mut self.inner.entries[index].value, materialization))
    }

    /// Installs a caller-owned workspace after a temporary take. Existing keys
    /// and full caches fail closed; no captured graph is displaced.
    pub fn install(
        &mut self,
        workspace: GraphWorkspace,
    ) -> Result<WorkspaceKey, WorkspaceCacheError> {
        let key = WorkspaceKey::from_workspace(&workspace);
        let admission = workspace.admission().clone();
        self.inner
            .insert(admission, key, ResidentWorkspaceEntry::new(workspace))?;
        Ok(key)
    }

    pub fn take(&mut self, key: WorkspaceKey) -> Option<GraphWorkspace> {
        let index = self.inner.unique_key_index(key)?;
        if !self.inner.entries[index].value.can_detach() {
            return None;
        }
        Some(self.inner.entries.remove(index).value.into_workspace())
    }

    pub(crate) fn only(&self) -> Option<&GraphWorkspace> {
        self.inner
            .only()
            .and_then(ResidentWorkspaceEntry::observable_workspace)
    }

    pub(crate) fn take_only(&mut self) -> Option<GraphWorkspace> {
        if self.inner.entries.len() != 1 || !self.inner.entries[0].value.can_detach() {
            return None;
        }
        Some(self.inner.entries.remove(0).value.into_workspace())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestAdmission {
        short: WorkspaceKey,
        topology: u64,
        layout: u64,
    }

    fn key(shape: u64, protocol: u64) -> WorkspaceKey {
        WorkspaceKey::new(ProofShapeKey(shape), protocol)
    }

    fn admission(shape: u64, protocol: u64, topology: u64, layout: u64) -> TestAdmission {
        TestAdmission {
            short: key(shape, protocol),
            topology,
            layout,
        }
    }

    #[test]
    fn key_separates_shape_and_protocol() {
        let base = key(7, 11);
        let keys = std::collections::HashSet::from([base, key(8, 11), key(7, 12)]);
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn bounded_cache_records_lookups_and_never_evicts() {
        let exact = admission(7, 11, 13, 17);
        let mut cache = BoundedCache::new(1).unwrap();
        cache.insert(exact.clone(), exact.short, 41u8).unwrap();

        assert_eq!(cache.get_by_key(key(7, 11)), Some(&41));
        assert_eq!(cache.get_by_key(key(8, 11)), None);
        assert!(matches!(
            cache.insert(exact.clone(), exact.short, 99),
            Err(WorkspaceCacheError::Occupied(_))
        ));
        let distinct = admission(8, 11, 19, 23);
        assert!(matches!(
            cache.insert(distinct.clone(), distinct.short, 42),
            Err(WorkspaceCacheError::AtCapacity { capacity: 1, .. })
        ));
        assert_eq!(cache.entries[0].value, 41);
        assert_eq!(
            cache.telemetry.get(),
            WorkspaceCacheTelemetry {
                hits: 1,
                misses: 1,
                materializations: 0,
                capacity_rejections: 1,
            }
        );
    }

    #[test]
    fn forced_short_key_collisions_require_exact_topology_and_layout() {
        let exact = admission(7, 11, 13, 17);
        let topology_collision = admission(7, 11, 19, 17);
        let layout_collision = admission(7, 11, 13, 23);
        let mut cache = BoundedCache::new(3).unwrap();
        cache.insert(exact.clone(), exact.short, 1u8).unwrap();

        assert_eq!(cache.exact_index(&exact), Some(0));
        assert_eq!(cache.exact_index(&topology_collision), None);
        assert_eq!(cache.exact_index(&layout_collision), None);

        cache
            .insert(topology_collision.clone(), topology_collision.short, 2)
            .unwrap();
        cache
            .insert(layout_collision.clone(), layout_collision.short, 3)
            .unwrap();
        assert_eq!(cache.exact_index(&topology_collision), Some(1));
        assert_eq!(cache.exact_index(&layout_collision), Some(2));
        assert_eq!(cache.get_by_key(exact.short), None);
        assert_eq!(cache.take_by_key(exact.short), None);
    }

    #[test]
    fn explicit_take_frees_capacity_and_get_mut_is_counted() {
        let first = admission(7, 11, 13, 17);
        let second = admission(8, 12, 19, 23);
        let mut cache = BoundedCache::new(1).unwrap();
        cache.insert(first.clone(), first.short, 41u8).unwrap();
        assert_eq!(cache.only(), Some(&41));
        assert_eq!(cache.take_only(), Some(41));
        cache.insert(second.clone(), second.short, 42).unwrap();
        *cache.get_by_key_mut(second.short).unwrap() = 43;
        cache.record_materialization();
        assert_eq!(cache.entries[0].value, 43);
        assert_eq!(cache.telemetry.get().hits, 1);
        assert_eq!(cache.telemetry.get().materializations, 1);
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            BoundedCache::<TestAdmission, u8>::new(0),
            Err(WorkspaceCacheError::ZeroCapacity)
        ));
    }

    #[test]
    fn runtime_identity_requires_exact_shape_protocol_and_arena_base() {
        let expected = ResidentWorkspaceIdentity {
            shape_key: ProofShapeKey(7),
            protocol_key: 11,
            arena_base: 13,
        };
        assert!(require_runtime_identity(expected, expected).is_ok());

        for actual in [
            ResidentWorkspaceIdentity {
                shape_key: ProofShapeKey(8),
                ..expected
            },
            ResidentWorkspaceIdentity {
                protocol_key: 12,
                ..expected
            },
            ResidentWorkspaceIdentity {
                arena_base: 14,
                ..expected
            },
        ] {
            assert!(matches!(
                require_runtime_identity(expected, actual),
                Err(ResidentRuntimeError::WorkspaceIdentityMismatch {
                    expected: rejected_expected,
                    actual: rejected_actual,
                }) if rejected_expected == expected && rejected_actual == actual
            ));
        }
    }
}
