//! Bounded ownership for warm, shape/protocol-exact graph workspaces.
//!
//! Entries are never evicted implicitly: captured graphs retain pointers into
//! their arena, so dropping an entry is always an explicit caller decision.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

use stwo_backend_cuda::{CudaExecContext, CudaRuntimeError};
use stwo_cairo_prover::witness::proof_shape::ProofShapeKey;

use crate::arena_plan::ProofArenaPlan;
use crate::graphs::{GraphError, GraphWorkspace};

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
        Self::from_plan(workspace.plan())
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

#[derive(Debug)]
pub enum WorkspaceCacheError {
    ZeroCapacity,
    Occupied(WorkspaceKey),
    AtCapacity {
        capacity: usize,
        requested: WorkspaceKey,
    },
    Runtime(CudaRuntimeError),
    Graph(GraphError),
}

impl std::fmt::Display for WorkspaceCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "workspace cache capacity must be non-zero"),
            Self::Occupied(key) => write!(f, "workspace cache key is already occupied: {key:?}"),
            Self::AtCapacity {
                capacity,
                requested,
            } => write!(
                f,
                "workspace cache capacity {capacity} reached; refusing to evict captured graphs \
                 for {requested:?}"
            ),
            Self::Runtime(error) => write!(f, "workspace CUDA context error: {error}"),
            Self::Graph(error) => write!(f, "workspace materialization error: {error}"),
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

/// Generic core keeps key/capacity behavior testable without constructing CUDA
/// resources. Production stores boxed workspaces so HashMap growth never moves
/// the host-side workspace object.
struct BoundedCache<V> {
    capacity: usize,
    entries: HashMap<WorkspaceKey, V>,
    telemetry: Cell<WorkspaceCacheTelemetry>,
}

impl<V> BoundedCache<V> {
    fn new(capacity: usize) -> Result<Self, WorkspaceCacheError> {
        if capacity == 0 {
            return Err(WorkspaceCacheError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            telemetry: Cell::new(WorkspaceCacheTelemetry::default()),
        })
    }

    fn get(&self, key: WorkspaceKey) -> Option<&V> {
        let result = self.entries.get(&key);
        self.record_lookup(result.is_some());
        result
    }

    fn get_mut(&mut self, key: WorkspaceKey) -> Option<&mut V> {
        let hit = self.entries.contains_key(&key);
        self.record_lookup(hit);
        self.entries.get_mut(&key)
    }

    fn insert(&mut self, key: WorkspaceKey, value: V) -> Result<(), WorkspaceCacheError> {
        self.admit(key)?;
        self.entries.insert(key, value);
        Ok(())
    }

    fn admit(&self, key: WorkspaceKey) -> Result<(), WorkspaceCacheError> {
        if self.entries.contains_key(&key) {
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

    fn take(&mut self, key: WorkspaceKey) -> Option<V> {
        self.entries.remove(&key)
    }

    fn only(&self) -> Option<&V> {
        if self.entries.len() == 1 {
            self.entries.values().next()
        } else {
            None
        }
    }

    fn take_only(&mut self) -> Option<V> {
        if self.entries.len() != 1 {
            return None;
        }
        let key = self.entries.keys().next().copied()?;
        self.take(key)
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
    inner: BoundedCache<Box<GraphWorkspace>>,
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

    pub fn get(&self, key: WorkspaceKey) -> Option<&GraphWorkspace> {
        self.inner.get(key).map(Box::as_ref)
    }

    pub fn get_mut(&mut self, key: WorkspaceKey) -> Option<&mut GraphWorkspace> {
        self.inner.get_mut(key).map(Box::as_mut)
    }

    pub fn materialize_or_reuse(
        &mut self,
        plan: Arc<ProofArenaPlan>,
    ) -> Result<(&mut GraphWorkspace, WorkspaceMaterialization), WorkspaceCacheError> {
        let key = WorkspaceKey::from_plan(&plan);
        if self.inner.entries.contains_key(&key) {
            self.inner.record_lookup(true);
            return Ok((
                self.inner
                    .entries
                    .get_mut(&key)
                    .expect("workspace key checked")
                    .as_mut(),
                WorkspaceMaterialization::Reused,
            ));
        }

        self.inner.record_lookup(false);
        self.inner.admit(key)?;

        let context = CudaExecContext::new()?;
        let workspace = Box::new(GraphWorkspace::from_plan(context, plan)?);
        self.inner.entries.insert(key, workspace);
        self.inner.record_materialization();
        Ok((
            self.inner
                .entries
                .get_mut(&key)
                .expect("materialized workspace inserted")
                .as_mut(),
            WorkspaceMaterialization::Materialized,
        ))
    }

    /// Installs a caller-owned workspace after a temporary take. Existing keys
    /// and full caches fail closed; no captured graph is displaced.
    pub fn install(
        &mut self,
        workspace: GraphWorkspace,
    ) -> Result<WorkspaceKey, WorkspaceCacheError> {
        let key = WorkspaceKey::from_workspace(&workspace);
        self.inner.insert(key, Box::new(workspace))?;
        Ok(key)
    }

    pub fn take(&mut self, key: WorkspaceKey) -> Option<GraphWorkspace> {
        self.inner.take(key).map(|workspace| *workspace)
    }

    pub(crate) fn only(&self) -> Option<&GraphWorkspace> {
        self.inner.only().map(Box::as_ref)
    }

    pub(crate) fn take_only(&mut self) -> Option<GraphWorkspace> {
        self.inner.take_only().map(|workspace| *workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(shape: u64, protocol: u64) -> WorkspaceKey {
        WorkspaceKey::new(ProofShapeKey(shape), protocol)
    }

    #[test]
    fn key_separates_shape_and_protocol() {
        let base = key(7, 11);
        let keys = std::collections::HashSet::from([base, key(8, 11), key(7, 12)]);
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn bounded_cache_records_lookups_and_never_evicts() {
        let mut cache = BoundedCache::new(1).unwrap();
        cache.insert(key(7, 11), 41u8).unwrap();

        assert_eq!(cache.get(key(7, 11)), Some(&41));
        assert_eq!(cache.get(key(8, 11)), None);
        assert!(matches!(
            cache.insert(key(7, 11), 99),
            Err(WorkspaceCacheError::Occupied(_))
        ));
        assert!(matches!(
            cache.insert(key(8, 11), 42),
            Err(WorkspaceCacheError::AtCapacity { capacity: 1, .. })
        ));
        assert_eq!(cache.entries.get(&key(7, 11)), Some(&41));
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
    fn explicit_take_frees_capacity_and_get_mut_is_counted() {
        let mut cache = BoundedCache::new(1).unwrap();
        cache.insert(key(7, 11), 41u8).unwrap();
        assert_eq!(cache.only(), Some(&41));
        assert_eq!(cache.take_only(), Some(41));
        cache.insert(key(8, 12), 42).unwrap();
        *cache.get_mut(key(8, 12)).unwrap() = 43;
        cache.record_materialization();
        assert_eq!(cache.entries.get(&key(8, 12)), Some(&43));
        assert_eq!(cache.telemetry.get().hits, 1);
        assert_eq!(cache.telemetry.get().materializations, 1);
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            BoundedCache::<u8>::new(0),
            Err(WorkspaceCacheError::ZeroCapacity)
        ));
    }
}
