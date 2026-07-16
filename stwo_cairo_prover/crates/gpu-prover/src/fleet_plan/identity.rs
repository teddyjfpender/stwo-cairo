use crate::fleet_spill::{SpillPlan, SpillTransitionKind};
use crate::transcript_plan::{CairoTranscriptBoundary, CairoTranscriptSegment};
use stwo_backend_cuda::{TranscriptOperation, TranscriptStart};

use super::*;

const DOMAIN: &[u8] = b"stwo-cairo.track-a.fleet-proof-plan.structural-v0\0";

pub(super) fn compute(plan: &FleetProofPlan) -> Result<[u8; 32], FleetPlanError> {
    Ok(*blake3::hash(&encode(plan)?).as_bytes())
}

pub(super) fn encode(plan: &FleetProofPlan) -> Result<Vec<u8>, FleetPlanError> {
    let mut out = Encoder::default();
    out.raw(DOMAIN);
    out.byte(match plan.input.topology.gpu_class {
        ConsumerGpuClass::Rtx3090Sm86 => 0,
        ConsumerGpuClass::Rtx4090Sm89 => 1,
        ConsumerGpuClass::Rtx5090Sm120 => 2,
    });
    out.raw(&plan.input.topology.module_pack_identity);
    out.raw(&plan.input.topology.fixed_image_identity);
    out.count(plan.input.topology.executable_identity.len())?;
    out.raw(&plan.input.topology.executable_identity);
    out.worker(plan.input.topology.coordinator);
    out.count(plan.input.topology.workers.len())?;
    for worker in &plan.input.topology.workers {
        out.worker(worker.id);
        out.size(worker.capacity_bytes)?;
        out.size(worker.exchange_reserve_bytes)?;
    }
    out.count(plan.input.topology.links.len())?;
    for link in &plan.input.topology.links {
        out.u16(link.id.0);
        out.worker(link.source);
        out.worker(link.destination);
        out.size(link.max_transfer_bytes)?;
    }
    out.count(plan.input.topology.host_numa.len())?;
    for capacity in &plan.input.topology.host_numa {
        out.u32(capacity.numa_node);
        out.size(capacity.store_capacity_bytes)?;
        out.size(capacity.memlock_limit_bytes)?;
    }
    for pow in [plan.input.pow.interaction, plan.input.pow.query] {
        out.u32(pow.workers_per_rank);
        out.u64(pow.indices_per_attempt);
    }
    out.count(plan.input.barrier_steps.len())?;
    for step in &plan.input.barrier_steps {
        out.u32(step.0);
    }
    out.u32(plan.input.terminal_step.0);
    out.count(plan.input.barrier_arrivals.len())?;
    for arrival in &plan.input.barrier_arrivals {
        out.u32(arrival.barrier_ordinal);
        out.worker(arrival.worker);
        out.u32(arrival.ready_step.0);
    }

    out.count(plan.input.values.len())?;
    for value in &plan.input.values {
        out.value(value.id);
        out.layout(&value.layout)?;
        match value.origin {
            ValueOrigin::ExternalInput(binding) => {
                out.byte(0);
                out.u32(binding);
            }
            ValueOrigin::FixedImage(constant) => {
                out.byte(1);
                out.u32(constant);
            }
            ValueOrigin::Operation => out.byte(2),
        }
    }
    out.count(plan.input.operations.len())?;
    for operation in &plan.input.operations {
        out.operation(operation.id);
        out.count(operation.semantic.len())?;
        out.raw(&operation.semantic);
        out.interval(operation.interval);
        out.schedule(operation.during);
        out.count(operation.reads.len())?;
        for read in &operation.reads {
            out.value(read.value);
            out.elements(read.elements)?;
            out.layout(&read.layout)?;
        }
        out.count(operation.writes.len())?;
        for write in &operation.writes {
            out.value(write.value);
            out.elements(write.elements)?;
            out.layout(&write.layout)?;
        }
    }
    out.count(plan.input.assignments.len())?;
    for assignment in &plan.input.assignments {
        out.operation(assignment.operation);
        out.worker(assignment.worker);
    }
    out.count(plan.input.owners.len())?;
    for owner in &plan.input.owners {
        out.value(owner.value);
        out.elements(owner.elements)?;
        out.worker(owner.worker);
        match owner.producer {
            Some(producer) => {
                out.byte(1);
                out.operation(producer);
            }
            None => out.byte(0),
        }
        out.u32(owner.ready_at.0);
        out.schedule(owner.live);
    }
    out.count(plan.input.replicas.len())?;
    for replica in &plan.input.replicas {
        out.u32(replica.id.0);
        out.value(replica.value);
        out.elements(replica.elements)?;
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
        out.u32(replica.ready_at.0);
        out.schedule(replica.live);
    }
    out.count(plan.input.transitions.len())?;
    for transition in &plan.input.transitions {
        out.u32(transition.id.0);
        out.value(transition.value);
        out.elements(transition.elements)?;
        out.worker(transition.source_worker);
        out.u32(transition.destination_replica.0);
        out.layout(&transition.source_layout)?;
        out.layout(&transition.destination_layout)?;
        out.count(transition.axes.len())?;
        for axis in &transition.axes {
            out.u16(axis.source);
            out.u16(axis.destination);
        }
        out.interval(transition.interval);
        out.schedule(transition.during);
        out.size(transition.bytes)?;
        out.size(transition.scratch_bytes)?;
        out.worker(transition.scratch_worker);
        out.u16(transition.route.0);
    }
    out.count(plan.input.spills.len())?;
    for spill in &plan.input.spills {
        out.spill(spill)?;
    }

    out.u64(plan.schedule_key);
    out.count(plan.transcript_encoding.len())?;
    out.raw(&plan.transcript_encoding);
    out.count(plan.barriers.len())?;
    for barrier in &plan.barriers {
        out.u32(barrier.ordinal);
        out.worker(barrier.coordinator);
        out.segment(barrier.segment);
        out.size(barrier.operation_range.start)?;
        out.size(barrier.operation_range.end)?;
        match barrier.starts_after {
            Some(boundary) => {
                out.byte(1);
                out.boundary(boundary)?;
            }
            None => out.byte(0),
        }
        out.boundary(barrier.ends_at)?;
        out.u32(barrier.release_step.0);
    }
    out.count(plan.workers.len())?;
    for worker in &plan.workers {
        out.worker(worker.worker);
        out.size(worker.peak_live_bytes)?;
        out.size(worker.capacity_bytes)?;
    }
    Ok(out.bytes)
}

pub(super) fn encode_transcript(
    plan: &crate::transcript_plan::CairoBlake2sTranscriptPlan,
) -> Result<Vec<u8>, FleetPlanError> {
    let schedule = plan.schedule();
    let mut out = Encoder::default();
    out.raw(b"stwo-cairo.track-a.transcript-operations-v0\0");
    out.count(crate::transcript_plan::CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG.len())?;
    out.raw(crate::transcript_plan::CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG.as_bytes());
    out.count(schedule.protocol_tag().len())?;
    out.raw(schedule.protocol_tag().as_bytes());
    out.u32(schedule.max_rejection_rounds());
    match schedule.start() {
        TranscriptStart::Default => out.byte(0),
        TranscriptStart::DeviceState(input) => {
            out.byte(1);
            out.u32(input.0);
        }
    }
    out.count(schedule.operations().len())?;
    for operation in schedule.operations() {
        match *operation {
            TranscriptOperation::MixFelts {
                boundary,
                source,
                n_felts,
            } => {
                out.byte(0);
                out.u32(boundary.0);
                out.u32(source.0);
                out.u32(n_felts);
            }
            TranscriptOperation::MixU32s {
                boundary,
                source,
                n_words,
            } => {
                out.byte(1);
                out.u32(boundary.0);
                out.u32(source.0);
                out.u32(n_words);
            }
            TranscriptOperation::MixU64 { boundary, source } => {
                out.byte(2);
                out.u32(boundary.0);
                out.u32(source.0);
            }
            TranscriptOperation::AbsorbRoot { boundary, source } => {
                out.byte(3);
                out.u32(boundary.0);
                out.u32(source.0);
            }
            TranscriptOperation::AbsorbPowNonce {
                boundary,
                source,
                pow_bits,
            } => {
                out.byte(4);
                out.u32(boundary.0);
                out.u32(source.0);
                out.u32(pow_bits);
            }
            TranscriptOperation::DrawSecureFelt { boundary, output } => {
                out.byte(5);
                out.u32(boundary.0);
                out.u32(output.0);
            }
            TranscriptOperation::DrawSecureFelts {
                boundary,
                output,
                n_felts,
            } => {
                out.byte(6);
                out.u32(boundary.0);
                out.u32(output.0);
                out.u32(n_felts);
            }
            TranscriptOperation::DrawU32s { boundary, output } => {
                out.byte(7);
                out.u32(boundary.0);
                out.u32(output.0);
            }
            TranscriptOperation::DrawQueries {
                boundary,
                output,
                log_domain_size,
                n_queries,
            } => {
                out.byte(8);
                out.u32(boundary.0);
                out.u32(output.0);
                out.u32(log_domain_size);
                out.u32(n_queries);
            }
        }
    }
    let requirements = schedule.requirements();
    out.size(requirements.state_words)?;
    out.size(requirements.boundary_snapshot_words)?;
    out.size(requirements.input_snapshot_words)?;
    out.size(requirements.input_snapshot_used_words)?;
    out.size(requirements.output_snapshot_words)?;
    out.size(requirements.output_snapshot_used_words)?;
    out.count(requirements.inputs.len())?;
    for input in &requirements.inputs {
        out.u32(input.id.0);
        out.size(input.min_words)?;
    }
    out.count(requirements.outputs.len())?;
    for output in &requirements.outputs {
        out.u32(output.id.0);
        out.size(output.min_words)?;
    }
    out.count(plan.inputs().len())?;
    for input in plan.inputs() {
        out.u32(
            input
                .semantic
                .id()
                .map_err(|_| FleetPlanError::TranscriptMismatch)?
                .0,
        );
        out.size(input.min_words)?;
    }
    out.count(plan.outputs().len())?;
    for output in plan.outputs() {
        out.u32(
            output
                .semantic
                .id()
                .map_err(|_| FleetPlanError::TranscriptMismatch)?
                .0,
        );
        out.size(output.min_words)?;
    }
    out.count(plan.boundaries().len())?;
    for boundary in plan.boundaries() {
        out.boundary(boundary.semantic)?;
        out.size(boundary.operation_index)?;
        out.segment(boundary.segment);
    }
    out.count(plan.segments().len())?;
    for segment in plan.segments() {
        out.segment(segment.segment);
        out.size(segment.operation_range.start)?;
        out.size(segment.operation_range.end)?;
        match segment.starts_after {
            Some(boundary) => {
                out.byte(1);
                out.boundary(boundary)?;
            }
            None => out.byte(0),
        }
        out.boundary(segment.ends_at)?;
    }
    Ok(out.bytes)
}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
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
        self.u32(u32::try_from(value).map_err(|_| FleetPlanError::SizeOverflow)?);
        Ok(())
    }

    fn size(&mut self, value: usize) -> Result<(), FleetPlanError> {
        self.u64(u64::try_from(value).map_err(|_| FleetPlanError::SizeOverflow)?);
        Ok(())
    }

    fn worker(&mut self, value: WorkerId) {
        self.u16(value.0);
    }

    fn operation(&mut self, value: OperationId) {
        self.u32(value.0);
    }

    fn value(&mut self, value: ValueId) {
        self.u32(value.0);
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

    fn segment(&mut self, segment: CairoTranscriptSegment) {
        match segment {
            CairoTranscriptSegment::BootstrapThroughBase => self.byte(0),
            CairoTranscriptSegment::InteractionPowAndLookup => self.byte(1),
            CairoTranscriptSegment::InteractionAndComposition => self.byte(2),
            CairoTranscriptSegment::CompositionAndOods => self.byte(3),
            CairoTranscriptSegment::OodsAndQuotient => self.byte(4),
            CairoTranscriptSegment::FriLayer(layer) => {
                self.byte(5);
                self.u32(layer);
            }
            CairoTranscriptSegment::FriLastLayer => self.byte(6),
            CairoTranscriptSegment::QueryPowAndPositions => self.byte(7),
        }
    }

    fn boundary(&mut self, boundary: CairoTranscriptBoundary) -> Result<(), FleetPlanError> {
        self.u32(
            boundary
                .id()
                .map_err(|_| FleetPlanError::TranscriptMismatch)?
                .0,
        );
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
            self.value(chunk.value);
            self.elements(chunk.elements)?;
            self.worker(chunk.worker);
            self.size(chunk.bytes)?;
            self.u32(chunk.store_extent.0);
            self.u16(chunk.ring_slot.0);
        }
        self.count(spill.transitions.len())?;
        for transition in &spill.transitions {
            self.u32(transition.id.0);
            self.u32(transition.chunk.0);
            self.byte(match transition.kind {
                SpillTransitionKind::DeviceToRing => 0,
                SpillTransitionKind::RingToStore => 1,
                SpillTransitionKind::StoreToRing => 2,
                SpillTransitionKind::RingToDevice => 3,
            });
            self.interval(transition.interval);
            self.schedule(transition.during);
            self.size(transition.bytes)?;
        }
        Ok(())
    }
}
