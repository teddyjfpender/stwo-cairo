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
}
