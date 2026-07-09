//! Per-proof transport for device-resident witness artifacts.
//!
//! Witness generation produces lookup and component-edge buffers that are consumed
//! later in the same proof. Keeping them here makes that lifetime explicit: every
//! proof owns one context, concurrent proofs cannot see each other's buffers, and
//! unconsumed buffers are released when the proof context drops.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use stwo_backend_cuda::BaseFieldVec;

use super::proof_shape::{
    ComponentId, ProofShape, ProofShapeError, RowResolution, RuntimeComponentShape, TracePartId,
    TracePartShape,
};
use super::proof_shape_generated::FinalComponentShape;

pub(crate) struct DeviceLookup {
    pub buffer: BaseFieldVec,
    pub n_rows: usize,
    pub n_real: usize,
}

pub(crate) struct DeviceEdge {
    pub buffer: BaseFieldVec,
    pub host_flat: Vec<u32>,
    pub n_rows: usize,
    pub plan: PlannedDeviceEdge,
}

/// One generated producer-to-consumer witness edge. The GPU-native prover builds
/// these facts from `CAIRO_SCHEDULE`; the legacy prover derives the same fact from
/// the generated `SUB_FEED_LAYOUT` at the producer seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedDeviceEdge {
    pub producer: &'static str,
    pub consumer: &'static str,
    pub word_base: u32,
    pub words_per_instance: u32,
    pub n_instances: u32,
}

/// Schedule-neutral projection of the generated Cairo component schedule used by
/// the lower witness crate. It deliberately contains only the facts needed to
/// validate live device artifacts; the full schedule remains owned by gpu-prover.
#[derive(Debug)]
pub struct WitnessArtifactPlan {
    components: HashSet<&'static str>,
    edges: HashMap<(&'static str, &'static str), PlannedDeviceEdge>,
}

impl WitnessArtifactPlan {
    pub fn new(
        components: Vec<&'static str>,
        edges: Vec<PlannedDeviceEdge>,
    ) -> WitnessArtifactPlan {
        let n_components = components.len();
        let components: HashSet<_> = components.into_iter().collect();
        assert_eq!(
            components.len(),
            n_components,
            "duplicate component in witness artifact plan"
        );

        let mut edge_map = HashMap::with_capacity(edges.len());
        for edge in edges {
            assert!(
                components.contains(edge.producer),
                "witness artifact edge has unknown producer: {}",
                edge.producer
            );
            assert!(
                components.contains(edge.consumer),
                "witness artifact edge has unknown consumer: {}",
                edge.consumer
            );
            let previous = edge_map.insert((edge.producer, edge.consumer), edge);
            assert!(
                previous.is_none(),
                "duplicate witness artifact edge: {} -> {}",
                edge.producer,
                edge.consumer
            );
        }

        Self {
            components,
            edges: edge_map,
        }
    }

    pub fn contains_component(&self, component: &str) -> bool {
        self.components.contains(component)
    }

    pub fn edge(&self, producer: &str, consumer: &str) -> Option<PlannedDeviceEdge> {
        self.edges.get(&(producer, consumer)).copied()
    }
}

struct ProofScopedStash<K, T>(Mutex<HashMap<K, T>>);

impl<K, T> Default for ProofScopedStash<K, T> {
    fn default() -> Self {
        Self(Mutex::new(HashMap::new()))
    }
}

#[derive(Debug)]
struct FinalShapeLedgerState {
    observed: HashMap<ComponentId, RuntimeComponentShape>,
    sealed: bool,
}

#[derive(Debug)]
struct FinalShapeLedger {
    expected: ProofShape,
    state: Mutex<FinalShapeLedgerState>,
}

impl FinalShapeLedger {
    fn new(expected: ProofShape) -> Self {
        Self {
            expected,
            state: Mutex::new(FinalShapeLedgerState {
                observed: HashMap::new(),
                sealed: false,
            }),
        }
    }

    fn record(&self, actual: RuntimeComponentShape) -> Result<(), FinalShapeError> {
        let expected = self
            .expected
            .component(actual.id)
            .ok_or(FinalShapeError::UnknownComponent(actual.id))?;
        validate_final_shape(expected, &actual)?;

        let mut state = self
            .state
            .lock()
            .expect("final witness shape ledger mutex poisoned");
        if state.sealed {
            return Err(FinalShapeError::AlreadySealed);
        }
        if state.observed.insert(actual.id, actual).is_some() {
            return Err(FinalShapeError::DuplicateComponent(expected.id));
        }
        Ok(())
    }

    fn seal(&self) -> Result<ProofShape, FinalShapeError> {
        let mut state = self
            .state
            .lock()
            .expect("final witness shape ledger mutex poisoned");
        if state.sealed {
            return Err(FinalShapeError::AlreadySealed);
        }

        let mut components = Vec::with_capacity(self.expected.components().len());
        for expected in self.expected.components() {
            match expected.rows {
                RowResolution::Absent => {
                    if state.observed.contains_key(expected.id) {
                        return Err(FinalShapeError::UnexpectedPresent(expected.id));
                    }
                    components.push(expected.clone());
                }
                _ => components.push(
                    state
                        .observed
                        .get(expected.id)
                        .cloned()
                        .ok_or(FinalShapeError::MissingComponent(expected.id))?,
                ),
            }
        }
        let shape = ProofShape::new(components).map_err(FinalShapeError::InvalidShape)?;
        shape
            .require_capture_ready()
            .map_err(FinalShapeError::InvalidShape)?;
        state.sealed = true;
        Ok(shape)
    }

    fn exact_part(
        &self,
        component: ComponentId,
        part: TracePartId,
    ) -> Result<TracePartShape, FinalShapeError> {
        let state = self
            .state
            .lock()
            .expect("final witness shape ledger mutex poisoned");
        if !state.sealed {
            return Err(FinalShapeError::NotSealed);
        }
        let shape = state
            .observed
            .get(component)
            .ok_or(FinalShapeError::MissingComponent(component))?;
        let RowResolution::Resolved(parts) = &shape.rows else {
            return Err(FinalShapeError::ActualRowsNotResolved(component));
        };
        parts
            .iter()
            .find(|shape| shape.part == part)
            .copied()
            .ok_or(FinalShapeError::MissingTracePart { component, part })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalShapeError {
    NoPlannedShape,
    NotSealed,
    AlreadySealed,
    UnknownComponent(ComponentId),
    DuplicateComponent(ComponentId),
    MissingComponent(ComponentId),
    MissingTracePart {
        component: ComponentId,
        part: TracePartId,
    },
    UnexpectedPresent(ComponentId),
    ExpectedRowsStillPending(ComponentId),
    ActualRowsNotResolved(ComponentId),
    ResolvedGeometryChanged {
        component: ComponentId,
        expected: RuntimeComponentShape,
        actual: RuntimeComponentShape,
    },
    FinalRowsBelowObserved {
        component: ComponentId,
        observed_rows: u64,
        final_rows: u64,
    },
    FinalRowsExceedCapacity {
        component: ComponentId,
        final_rows: u64,
        max_rows: u64,
    },
    FinalPaddingExceedsCapacity {
        component: ComponentId,
        final_padding: u64,
        padded_capacity: u64,
    },
    InvalidShape(ProofShapeError),
}

impl std::fmt::Display for FinalShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FinalShapeError {}

fn validate_final_shape(
    expected: &RuntimeComponentShape,
    actual: &RuntimeComponentShape,
) -> Result<(), FinalShapeError> {
    let RowResolution::Resolved(actual_parts) = &actual.rows else {
        return Err(FinalShapeError::ActualRowsNotResolved(expected.id));
    };
    match &expected.rows {
        RowResolution::Absent => Err(FinalShapeError::UnexpectedPresent(expected.id)),
        RowResolution::Resolved(_) => {
            if expected != actual {
                return Err(FinalShapeError::ResolvedGeometryChanged {
                    component: expected.id,
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
            Ok(())
        }
        RowResolution::Pending { .. } => {
            Err(FinalShapeError::ExpectedRowsStillPending(expected.id))
        }
        RowResolution::Bounded { bound, .. } => {
            if actual_parts.len() != 1 || actual_parts[0].part != TracePartId::Main {
                return Err(FinalShapeError::ResolvedGeometryChanged {
                    component: expected.id,
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
            let part = actual_parts[0];
            if part.n_real_rows < bound.observed_rows {
                return Err(FinalShapeError::FinalRowsBelowObserved {
                    component: expected.id,
                    observed_rows: bound.observed_rows,
                    final_rows: part.n_real_rows,
                });
            }
            if part.n_real_rows > bound.max_rows {
                return Err(FinalShapeError::FinalRowsExceedCapacity {
                    component: expected.id,
                    final_rows: part.n_real_rows,
                    max_rows: bound.max_rows,
                });
            }
            if part.padded_rows > bound.padded_capacity {
                return Err(FinalShapeError::FinalPaddingExceedsCapacity {
                    component: expected.id,
                    final_padding: part.padded_rows,
                    padded_capacity: bound.padded_capacity,
                });
            }
            Ok(())
        }
    }
}

impl<K, T> ProofScopedStash<K, T>
where
    K: Copy + Debug + Eq + Hash + Ord,
{
    fn contains(&self, key: K) -> bool {
        self.0
            .lock()
            .expect("witness execution context mutex poisoned")
            .contains_key(&key)
    }

    fn insert(&self, key: K, value: T) {
        let previous = self
            .0
            .lock()
            .expect("witness execution context mutex poisoned")
            .insert(key, value);
        assert!(
            previous.is_none(),
            "duplicate per-proof witness artifact: {key:?}"
        );
    }

    fn take(&self, key: K) -> Option<T> {
        self.0
            .lock()
            .expect("witness execution context mutex poisoned")
            .remove(&key)
    }

    fn keys(&self) -> Vec<K> {
        let mut keys: Vec<_> = self
            .0
            .lock()
            .expect("witness execution context mutex poisoned")
            .keys()
            .copied()
            .collect();
        keys.sort_unstable();
        keys
    }
}

/// Device witness artifacts owned by one proof execution.
#[derive(Default)]
pub struct WitnessExecContext {
    plan: Option<Arc<WitnessArtifactPlan>>,
    final_shape: Option<FinalShapeLedger>,
    device_lookups: ProofScopedStash<&'static str, DeviceLookup>,
    edges: ProofScopedStash<(&'static str, &'static str), DeviceEdge>,
}

impl WitnessExecContext {
    pub fn planned(plan: Arc<WitnessArtifactPlan>) -> Self {
        Self {
            plan: Some(plan),
            ..Self::default()
        }
    }

    pub fn planned_with_shape(plan: Arc<WitnessArtifactPlan>, shape: ProofShape) -> Self {
        Self {
            plan: Some(plan),
            final_shape: Some(FinalShapeLedger::new(shape)),
            ..Self::default()
        }
    }

    /// Records the exact geometry at the last safe point: immediately before
    /// the component generator is consumed. Legacy contexts have no proof plan
    /// and deliberately skip this GPU graph-capture ledger.
    pub(crate) fn record_final_component<G: FinalComponentShape>(
        &self,
        generator: &G,
        opt_n_id_to_big_components: Option<usize>,
    ) {
        let Some(ledger) = &self.final_shape else {
            return;
        };
        let device_feed_rows = self.device_feed_rows(G::COMPONENT);
        assert!(
            device_feed_rows == 0 || G::ACCEPTS_DEVICE_FEED_ROWS,
            "component {} received a device edge but its generated row source does not accept one",
            G::COMPONENT
        );
        let shape = generator
            .final_component_shape(opt_n_id_to_big_components, device_feed_rows)
            .expect("component final row geometry is invalid");
        assert_eq!(
            shape.id,
            G::COMPONENT,
            "generated final-shape trait reported the wrong component"
        );
        ledger
            .record(shape)
            .expect("component final row geometry violated its generated capacity contract");
    }

    /// Seals all generated component rows to exact values. Missing or duplicate
    /// witness consumption fails closed before any CUDA graph can be prepared.
    pub fn seal_final_proof_shape(&self) -> Result<ProofShape, FinalShapeError> {
        self.final_shape
            .as_ref()
            .ok_or(FinalShapeError::NoPlannedShape)?
            .seal()
    }

    /// Returns an authoritative component part only after the witness ledger is
    /// sealed. Relation-source exports must never infer real rows from padding.
    pub fn exact_relation_part(
        &self,
        component: ComponentId,
        part: TracePartId,
    ) -> Result<TracePartShape, FinalShapeError> {
        self.final_shape
            .as_ref()
            .ok_or(FinalShapeError::NoPlannedShape)?
            .exact_part(component, part)
    }

    fn device_feed_rows(&self, consumer: ComponentId) -> u64 {
        self.edges
            .0
            .lock()
            .expect("witness execution context mutex poisoned")
            .values()
            .filter(|edge| edge.plan.consumer == consumer)
            .try_fold(0u64, |total, edge| {
                let rows = u64::from(edge.plan.n_instances).checked_mul(edge.n_rows as u64)?;
                total.checked_add(rows)
            })
            .unwrap_or_else(|| panic!("device feed row count overflow for {consumer}"))
    }

    fn require_component(&self, component: &'static str) {
        if let Some(plan) = &self.plan {
            assert!(
                plan.contains_component(component),
                "device witness artifact has no scheduled component: {component}"
            );
        }
    }

    fn require_edge(&self, edge: PlannedDeviceEdge) {
        if let Some(plan) = &self.plan {
            let scheduled = plan.edge(edge.producer, edge.consumer).unwrap_or_else(|| {
                panic!(
                    "device witness artifact has no scheduled edge: {} -> {}",
                    edge.producer, edge.consumer
                )
            });
            assert_eq!(
                scheduled, edge,
                "live device edge geometry disagrees with CAIRO_SCHEDULE"
            );
        }
    }

    pub(crate) fn has_device_lookup(&self, component: &'static str) -> bool {
        self.require_component(component);
        self.device_lookups.contains(component)
    }

    pub(crate) fn insert_device_lookup(
        &self,
        component: &'static str,
        buffer: BaseFieldVec,
        n_rows: usize,
        n_real: usize,
    ) {
        self.require_component(component);
        self.device_lookups.insert(
            component,
            DeviceLookup {
                buffer,
                n_rows,
                n_real,
            },
        );
    }

    pub(crate) fn take_device_lookup(&self, component: &'static str) -> Option<DeviceLookup> {
        self.require_component(component);
        self.device_lookups.take(component)
    }

    pub(crate) fn insert_edge(
        &self,
        plan: PlannedDeviceEdge,
        buffer: BaseFieldVec,
        host_flat: Vec<u32>,
        n_rows: usize,
    ) {
        self.require_edge(plan);
        self.edges.insert(
            (plan.producer, plan.consumer),
            DeviceEdge {
                buffer,
                host_flat,
                n_rows,
                plan,
            },
        );
    }

    pub(crate) fn take_edge(
        &self,
        producer: &'static str,
        consumer: &'static str,
    ) -> Option<DeviceEdge> {
        if let Some(plan) = &self.plan {
            assert!(
                plan.edge(producer, consumer).is_some(),
                "device witness artifact has no scheduled edge: {producer} -> {consumer}"
            );
        }
        self.edges.take((producer, consumer))
    }

    /// Producer edges must be consumed inside base witness generation. Lookup
    /// buffers intentionally remain live until Fiat-Shamir draws their challenges.
    pub fn assert_witness_drained(&self) {
        let edges = self.edges.keys();
        assert!(
            edges.is_empty(),
            "base witness phase left unconsumed device edges: {edges:?}"
        );
    }

    /// Interaction is the final consumer of every witness-side device artifact.
    pub fn assert_interaction_drained(&self) {
        let edges = self.edges.keys();
        let lookups = self.device_lookups.keys();
        assert!(
            edges.is_empty() && lookups.is_empty(),
            "interaction phase left device artifacts: edges={edges:?}, lookups={lookups:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::witness::components::blake_g;
    use crate::witness::proof_shape::{CapacityBound, PendingRowsReason, TracePartShape};

    fn borrowed_empty_buffer() -> BaseFieldVec {
        BaseFieldVec::from_borrowed_ptr(std::ptr::null(), 0)
    }

    fn test_edge() -> PlannedDeviceEdge {
        PlannedDeviceEdge {
            producer: "producer",
            consumer: "consumer",
            word_base: 7,
            words_per_instance: 72,
            n_instances: 28,
        }
    }

    fn test_plan() -> Arc<WitnessArtifactPlan> {
        Arc::new(WitnessArtifactPlan::new(
            vec!["component", "producer", "consumer"],
            vec![test_edge()],
        ))
    }

    #[test]
    fn proof_contexts_are_isolated() {
        let first = WitnessExecContext::planned(test_plan());
        let second = WitnessExecContext::planned(test_plan());

        first.insert_device_lookup("component", borrowed_empty_buffer(), 16, 8);
        first.insert_edge(test_edge(), borrowed_empty_buffer(), vec![1, 2], 16);

        assert!(first.has_device_lookup("component"));
        assert!(!second.has_device_lookup("component"));
        assert!(second.take_device_lookup("component").is_none());
        assert!(second.take_edge("producer", "consumer").is_none());

        let lookup = first.take_device_lookup("component").unwrap();
        assert_eq!((lookup.n_rows, lookup.n_real), (16, 8));
        let edge = first.take_edge("producer", "consumer").unwrap();
        assert_eq!(edge.plan, test_edge());
        assert_eq!((edge.host_flat, edge.n_rows), (vec![1, 2], 16));
        first.assert_witness_drained();
        first.assert_interaction_drained();
    }

    #[test]
    fn phase_drains_report_live_artifacts() {
        let context = WitnessExecContext::planned(test_plan());
        context.insert_edge(test_edge(), borrowed_empty_buffer(), Vec::new(), 16);
        assert!(std::panic::catch_unwind(|| context.assert_witness_drained()).is_err());
        context.take_edge("producer", "consumer").unwrap();
        context.assert_witness_drained();

        context.insert_device_lookup("component", borrowed_empty_buffer(), 16, 8);
        assert!(std::panic::catch_unwind(|| context.assert_interaction_drained()).is_err());
        context.take_device_lookup("component").unwrap();
        context.assert_interaction_drained();
    }

    #[test]
    fn plan_rejects_unknown_artifacts_and_geometry_drift() {
        let context = WitnessExecContext::planned(test_plan());
        assert!(std::panic::catch_unwind(|| {
            context.insert_device_lookup("unknown", borrowed_empty_buffer(), 16, 8)
        })
        .is_err());

        let mut drifted = test_edge();
        drifted.word_base += 1;
        assert!(std::panic::catch_unwind(|| {
            context.insert_edge(drifted, borrowed_empty_buffer(), Vec::new(), 16)
        })
        .is_err());
    }

    #[test]
    fn proof_scoped_stash_drops_unconsumed_values() {
        struct DropSpy(Arc<AtomicUsize>);
        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let stash = ProofScopedStash::default();
            stash.insert("device-buffer", DropSpy(Arc::clone(&drops)));
            assert_eq!(drops.load(Ordering::Relaxed), 0);
        }
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn final_shape_ledger_requires_every_present_component() {
        let expected = ProofShape::new(vec![
            RuntimeComponentShape::uniform("present", 7, 16).unwrap(),
            RuntimeComponentShape::absent("absent"),
        ])
        .unwrap();
        let ledger = FinalShapeLedger::new(expected);
        assert_eq!(
            ledger.seal(),
            Err(FinalShapeError::MissingComponent("present"))
        );
    }

    #[test]
    fn exact_relation_parts_are_available_only_after_sealing() {
        let component = RuntimeComponentShape::uniform("component", 9, 16).unwrap();
        let ledger = FinalShapeLedger::new(ProofShape::new(vec![component.clone()]).unwrap());
        ledger.record(component).unwrap();
        assert_eq!(
            ledger.exact_part("component", TracePartId::Main),
            Err(FinalShapeError::NotSealed)
        );
        ledger.seal().unwrap();
        assert_eq!(
            ledger.exact_part("component", TracePartId::Main).unwrap(),
            TracePartShape {
                part: TracePartId::Main,
                n_real_rows: 9,
                padded_rows: 16,
            }
        );
    }

    #[test]
    fn generated_final_shape_counts_resident_device_edge_rows() {
        let edge = PlannedDeviceEdge {
            producer: "producer",
            consumer: "blake_g",
            word_base: 0,
            words_per_instance: 6,
            n_instances: 10,
        };
        let artifacts = Arc::new(WitnessArtifactPlan::new(
            vec!["producer", "blake_g"],
            vec![edge],
        ));
        let expected = ProofShape::new(vec![RuntimeComponentShape::bounded(
            "blake_g",
            PendingRowsReason::WitnessRelationFeeds,
            CapacityBound {
                observed_rows: 0,
                max_rows: 160,
                padded_capacity: 256,
            },
        )])
        .unwrap();
        let context = WitnessExecContext::planned_with_shape(artifacts, expected);
        context.insert_edge(edge, borrowed_empty_buffer(), Vec::new(), 16);

        context.record_final_component(&blake_g::ClaimGenerator::new(), None);
        let sealed = context.seal_final_proof_shape().unwrap();
        assert_eq!(
            sealed.component("blake_g").unwrap().rows,
            RowResolution::Resolved(vec![TracePartShape {
                part: TracePartId::Main,
                n_real_rows: 160,
                padded_rows: 256,
            }])
        );
    }
}
