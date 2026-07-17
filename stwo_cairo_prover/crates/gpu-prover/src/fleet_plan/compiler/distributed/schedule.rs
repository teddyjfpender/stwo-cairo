//! Deterministic operation placement and transcript-barrier scheduling.

use super::*;
use crate::compiled_proof::{ExecutionPrimitive, PartitionAuthorityKind};

pub(super) struct DistributedSchedule {
    pub(super) barrier_steps: Vec<ScheduleStep>,
    pub(super) terminal_step: ScheduleStep,
    pub(super) barrier_arrivals: Vec<BarrierArrival>,
    pub(super) operations: Vec<FleetOperationPlacement>,
    pub(super) pre_operation: Vec<ScheduleRange>,
    pub(super) transcript_gathers: Vec<ScheduleRange>,
    pub(super) tail_gather: ScheduleRange,
}

impl DistributedSchedule {
    pub(super) fn compile(
        compiled: &CompiledProof,
        topology: &FleetPlacementTopology,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetCompileError> {
        let mut cursor = ScheduleStep(0);
        let mut operations = Vec::with_capacity(compiled.operations().len());
        let mut pre_operation = vec![None; compiled.operations().len()];
        let mut barrier_steps = Vec::with_capacity(transcript.segments().len());
        let mut transcript_gathers = Vec::with_capacity(transcript.segments().len());

        for segment in transcript.segments() {
            for operation in compiled.operations().iter().filter(|operation| {
                operation.stage == ProofStage::BeforeTranscript(segment.segment)
            }) {
                let pre = take_step(&mut cursor)?;
                pre_operation[operation.id.0 as usize] = Some(pre);
                operations.push(FleetOperationPlacement {
                    operation: operation.id,
                    during: take_step(&mut cursor)?,
                    executions: operation_executions(compiled, operation, topology)?,
                });
            }
            transcript_gathers.push(take_step(&mut cursor)?);
            cursor = increment(cursor)?;
            barrier_steps.push(cursor);
        }
        for operation in compiled
            .operations()
            .iter()
            .filter(|operation| operation.stage == ProofStage::AfterTranscript)
        {
            let pre = take_step(&mut cursor)?;
            pre_operation[operation.id.0 as usize] = Some(pre);
            operations.push(FleetOperationPlacement {
                operation: operation.id,
                during: take_step(&mut cursor)?,
                executions: operation_executions(compiled, operation, topology)?,
            });
        }
        let tail_gather = take_step(&mut cursor)?;
        let terminal_step = increment(cursor)?;
        operations.sort_unstable_by_key(|placement| placement.operation);
        if operations.len() != compiled.operations().len()
            || operations
                .iter()
                .zip(compiled.operations())
                .any(|(placement, operation)| placement.operation != operation.id)
        {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
        let pre_operation = pre_operation
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let barrier_arrivals = conservative_arrivals(topology, &barrier_steps, terminal_step)?;
        Ok(Self {
            barrier_steps,
            terminal_step,
            barrier_arrivals,
            operations,
            pre_operation,
            transcript_gathers,
            tail_gather,
        })
    }
}

fn operation_executions(
    compiled: &CompiledProof,
    operation: &OpNode,
    topology: &FleetPlacementTopology,
) -> Result<Vec<FleetOperationExecution>, FleetCompileError> {
    let partition = compiled
        .partitions()
        .iter()
        .find(|partition| partition.id() == operation.partition)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    match partition.kind() {
        PartitionAuthorityKind::Monolithic => Ok(vec![FleetOperationExecution {
            worker: topology.coordinator,
            domain: OperationDomain::Monolithic,
        }]),
        PartitionAuthorityKind::Exact(authority) => {
            if matches!(
                operation.primitive,
                ExecutionPrimitive::OrderedComposite { .. }
            ) {
                return Err(FleetCompileError::UnsupportedExactComposite {
                    operation: operation.id,
                });
            }
            let granules = authority.domain().len() / authority.granularity();
            let shard_count = granules.min(topology.workers.len());
            if granules == 0 || shard_count == 0 {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            let base = granules / shard_count;
            let extra = granules % shard_count;
            let mut cursor = authority.domain().start;
            let mut executions = Vec::with_capacity(shard_count);
            for (index, worker) in topology.workers.iter().take(shard_count).enumerate() {
                let units = base + usize::from(index < extra);
                let elements = units
                    .checked_mul(authority.granularity())
                    .ok_or(FleetCompileError::SizeOverflow)?;
                let end = cursor
                    .checked_add(elements)
                    .ok_or(FleetCompileError::SizeOverflow)?;
                executions.push(FleetOperationExecution {
                    worker: worker.id,
                    domain: OperationDomain::Exact(
                        ElementRange::new(cursor, end)
                            .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
                    ),
                });
                cursor = end;
            }
            if cursor != authority.domain().end {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            Ok(executions)
        }
    }
}

fn conservative_arrivals(
    topology: &FleetPlacementTopology,
    barriers: &[ScheduleStep],
    terminal: ScheduleStep,
) -> Result<Vec<BarrierArrival>, FleetCompileError> {
    let capacity = barriers
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(topology.workers.len()))
        .ok_or(FleetCompileError::SizeOverflow)?;
    let mut arrivals = Vec::with_capacity(capacity);
    for (ordinal, &release) in barriers.iter().enumerate() {
        let ready_step = ScheduleStep(
            release
                .0
                .checked_sub(1)
                .ok_or(FleetCompileError::SizeOverflow)?,
        );
        for worker in &topology.workers {
            arrivals.push(BarrierArrival {
                barrier_ordinal: u32::try_from(ordinal)
                    .map_err(|_| FleetCompileError::SizeOverflow)?,
                worker: worker.id,
                ready_step,
            });
        }
    }
    let terminal_ordinal =
        u32::try_from(barriers.len()).map_err(|_| FleetCompileError::SizeOverflow)?;
    let ready_step = ScheduleStep(
        terminal
            .0
            .checked_sub(1)
            .ok_or(FleetCompileError::SizeOverflow)?,
    );
    for worker in &topology.workers {
        arrivals.push(BarrierArrival {
            barrier_ordinal: terminal_ordinal,
            worker: worker.id,
            ready_step,
        });
    }
    Ok(arrivals)
}
