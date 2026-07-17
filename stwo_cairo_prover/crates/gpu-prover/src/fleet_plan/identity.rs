use super::*;
use crate::compiled_proof::{OpId, ValueLayout, ValueRange, ValueVersion};
use crate::fleet_spill::{SpillPlan, SpillTransitionKind};

const DOMAIN: &[u8] = b"stwo-cairo.track-a.fleet-proof-plan.structural-v5\0";

pub(super) fn compute(plan: &FleetProofPlan) -> Result<[u8; 32], FleetPlanError> {
    Ok(*blake3::hash(&encode(plan)?).as_bytes())
}

/// Encode the owning shape once, followed only by fleet placement and the
/// validator's derived physical peak. The shape already binds the exact
/// CompiledProof and transcript; copying either semantic authority here would
/// create a second representation that could drift.
pub(super) fn encode(plan: &FleetProofPlan) -> Result<Vec<u8>, FleetPlanError> {
    let placement = &plan.placement;
    let mut out = Encoder::new();
    out.bytes(&plan.shape_encoding)?;

    out.byte(match placement.topology.gpu_class {
        ConsumerGpuClass::Rtx3090Sm86 => 0,
        ConsumerGpuClass::Rtx4090Sm89 => 1,
        ConsumerGpuClass::Rtx5090Sm120 => 2,
    });
    out.raw(&placement.topology.module_pack_identity);
    out.raw(&placement.topology.fixed_image_identity);
    out.worker(placement.topology.coordinator);
    out.count(placement.topology.workers.len())?;
    for worker in &placement.topology.workers {
        out.worker(worker.id);
        out.size(worker.capacity_bytes)?;
        out.size(worker.exchange_reserve_bytes)?;
    }
    out.count(placement.topology.links.len())?;
    for link in &placement.topology.links {
        out.u16(link.id.0);
        out.worker(link.source);
        out.worker(link.destination);
        out.size(link.max_transfer_bytes)?;
    }
    out.count(placement.topology.host_numa.len())?;
    for capacity in &placement.topology.host_numa {
        out.u32(capacity.numa_node);
        out.size(capacity.store_capacity_bytes)?;
        out.size(capacity.memlock_limit_bytes)?;
    }

    for pow in [placement.pow.interaction, placement.pow.query] {
        out.u32(pow.workers_per_rank);
        out.u64(pow.indices_per_attempt);
    }
    out.count(placement.barrier_steps.len())?;
    for step in &placement.barrier_steps {
        out.u32(step.0);
    }
    out.u32(placement.terminal_step.0);
    out.count(placement.barrier_arrivals.len())?;
    for arrival in &placement.barrier_arrivals {
        out.u32(arrival.barrier_ordinal);
        out.worker(arrival.worker);
        out.u32(arrival.ready_step.0);
    }

    out.count(placement.operations.len())?;
    for operation in &placement.operations {
        out.operation(operation.operation);
        out.schedule(operation.during);
        out.count(operation.executions.len())?;
        for execution in &operation.executions {
            out.worker(execution.worker);
            match execution.domain {
                OperationDomain::Monolithic => out.byte(0),
                OperationDomain::Exact(range) => {
                    out.byte(1);
                    out.elements(range)?;
                }
            }
        }
    }
    out.count(placement.owners.len())?;
    for owner in &placement.owners {
        out.value_range(owner.value)?;
        out.worker(owner.worker);
        out.schedule(owner.live);
    }
    out.count(placement.replicas.len())?;
    for replica in &placement.replicas {
        out.u32(replica.id.0);
        out.value_range(replica.value)?;
        out.worker(replica.canonical_worker);
        out.worker(replica.worker);
        out.layout(&replica.layout)?;
        match replica.origin {
            ReplicaOrigin::InstalledFixed => out.byte(0),
            ReplicaOrigin::Transition(transition) => {
                out.byte(1);
                out.u32(transition.0);
            }
        }
        out.schedule(replica.live);
    }
    out.count(placement.transitions.len())?;
    for transition in &placement.transitions {
        out.u32(transition.id.0);
        out.value_range(transition.value)?;
        out.worker(transition.source_worker);
        out.u32(transition.destination_replica.0);
        out.count(transition.axes.len())?;
        for axis in &transition.axes {
            out.u16(axis.source);
            out.u16(axis.destination);
        }
        out.interval(transition.interval);
        out.schedule(transition.during);
        out.size(transition.scratch_bytes)?;
        out.worker(transition.scratch_worker);
        out.u16(transition.route.0);
    }
    out.count(placement.spills.len())?;
    for spill in &placement.spills {
        out.spill(spill)?;
    }
    out.count(placement.storages.len())?;
    for storage in &placement.storages {
        out.u32(storage.id.0);
        out.worker(storage.worker);
        out.size(storage.bytes)?;
        out.size(storage.alignment_bytes)?;
    }
    out.count(placement.storage_bindings.len())?;
    for binding in &placement.storage_bindings {
        out.u32(binding.storage.0);
        out.value_range(binding.value)?;
        out.size(binding.offset_bytes)?;
    }
    out.count(placement.in_place_aliases.len())?;
    for alias in &placement.in_place_aliases {
        out.operation(alias.operation);
        out.u32(alias.alias.0);
        out.u32(alias.storage.0);
        out.size(alias.offset_bytes)?;
    }
    out.u32(placement.output_storage.0);

    out.count(plan.workers.len())?;
    for worker in &plan.workers {
        out.worker(worker.worker);
        out.size(worker.peak_resident_bytes)?;
        out.size(worker.capacity_bytes)?;
    }
    Ok(out.finish())
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self {
            bytes: DOMAIN.to_vec(),
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn raw(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn count(&mut self, value: usize) -> Result<(), FleetPlanError> {
        self.u64(u64::try_from(value).map_err(|_| FleetPlanError::SizeOverflow)?);
        Ok(())
    }

    fn size(&mut self, value: usize) -> Result<(), FleetPlanError> {
        self.u64(u64::try_from(value).map_err(|_| FleetPlanError::SizeOverflow)?);
        Ok(())
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), FleetPlanError> {
        self.count(value.len())?;
        self.raw(value);
        Ok(())
    }

    fn worker(&mut self, value: WorkerId) {
        self.u16(value.0);
    }

    fn operation(&mut self, value: OpId) {
        self.u32(value.0);
    }

    fn value(&mut self, value: ValueVersion) {
        self.u32(value.0);
    }

    fn value_range(&mut self, value: ValueRange) -> Result<(), FleetPlanError> {
        self.value(value.version);
        self.elements(value.elements)
    }

    fn schedule(&mut self, value: ScheduleRange) {
        self.u32(value.start.0);
        self.u32(value.end.0);
    }

    fn interval(&mut self, value: ExecutionInterval) {
        match value {
            ExecutionInterval::BeforeBarrier(barrier) => {
                self.byte(0);
                self.u32(barrier);
            }
            ExecutionInterval::AfterFinalBarrier => self.byte(1),
        }
    }

    fn elements(&mut self, value: ElementRange) -> Result<(), FleetPlanError> {
        self.size(value.start)?;
        self.size(value.end)
    }

    fn layout(&mut self, layout: &ValueLayout) -> Result<(), FleetPlanError> {
        self.u32(layout.element.tag);
        self.size(layout.element.bytes)?;
        self.count(layout.axes.len())?;
        for axis in &layout.axes {
            self.u16(axis.tag);
            self.size(axis.extent)?;
            self.size(axis.stride_bytes)?;
        }
        Ok(())
    }

    fn spill(&mut self, spill: &SpillPlan) -> Result<(), FleetPlanError> {
        self.worker(spill.store.worker);
        self.size(spill.store.capacity_bytes)?;
        self.size(spill.store.alignment_bytes)?;
        self.u32(spill.store.numa_node);
        self.count(spill.store.extents.len())?;
        for extent in &spill.store.extents {
            self.u32(extent.id.0);
            self.size(extent.offset_bytes)?;
            self.size(extent.len_bytes)?;
        }

        self.worker(spill.ring.worker);
        self.u32(spill.ring.numa_node);
        self.size(spill.ring.capacity_bytes)?;
        self.size(spill.ring.memlock_limit_bytes)?;
        self.size(spill.ring.alignment_bytes)?;
        self.count(spill.ring.slots.len())?;
        for slot in &spill.ring.slots {
            self.u16(slot.id.0);
            self.size(slot.offset_bytes)?;
            self.size(slot.len_bytes)?;
        }

        self.count(spill.chunks.len())?;
        for chunk in &spill.chunks {
            self.u32(chunk.id.0);
            self.value_range(chunk.value)?;
            self.worker(chunk.worker);
            self.u32(chunk.storage.0);
            self.u32(chunk.store_extent.0);
            self.size(chunk.len_bytes)?;
        }
        self.count(spill.transitions.len())?;
        for transition in &spill.transitions {
            self.u32(transition.id.0);
            self.u32(transition.chunk.0);
            self.u32(transition.tile_ordinal);
            self.size(transition.chunk_offset_bytes)?;
            self.size(transition.len_bytes)?;
            self.u16(transition.ring_slot.0);
            self.byte(match transition.kind {
                SpillTransitionKind::DeviceToRing => 0,
                SpillTransitionKind::RingToStore => 1,
                SpillTransitionKind::StoreToRing => 2,
                SpillTransitionKind::RingToDevice => 3,
            });
            self.interval(transition.interval);
            self.schedule(transition.during);
        }
        self.count(spill.vmm_reclaims.len())?;
        for reclaim in &spill.vmm_reclaims {
            self.u32(reclaim.chunk.0);
            self.u32(reclaim.storage.0);
            self.size(reclaim.allocation_granularity_bytes)?;
            self.interval(reclaim.unmap.interval);
            self.schedule(reclaim.unmap.during);
            self.interval(reclaim.remap.interval);
            self.schedule(reclaim.remap.during);
            self.u32(reclaim.remap_generation);
        }
        Ok(())
    }
}
