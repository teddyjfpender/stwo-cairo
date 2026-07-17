//! Deterministic Track-A physical lowering.
//!
//! This MVP compiler is intentionally non-distributed: it admits exactly one
//! worker and keeps every semantic operation and value there. It removes
//! caller-authored placement without pretending that an untyped operation can
//! be sharded, colored onto reused storage, or legally placed in-place.

use std::sync::Arc;

use super::*;
use crate::compiled_proof::{
    CompiledProof, InPlaceAliasRequirement, OpNode, PartitionAuthorityKind, ProofStage, Region,
    ValueOrigin, ValueRange,
};
use crate::fleet_pow::{FleetPowError, FleetPowSchedule};
use crate::shape_executable::ShapeExecutableIdentity;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

impl FleetProofPlan {
    /// Compile one validated semantic proof into a deterministic, fail-closed
    /// single-rank physical plan. Multi-worker placement, storage coloring and
    /// required in-place aliases remain fail-closed until their own physical
    /// authorities are implemented.
    pub fn compile_track_a_monolithic(
        compiled: Arc<CompiledProof>,
        shape: ShapeExecutableIdentity,
        topology: FleetPlacementTopology,
        pow: FleetPowSchedule,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, FleetCompileError> {
        if topology.workers.len() != 1 {
            return Err(FleetCompileError::MonolithicWorkerCount {
                actual: topology.workers.len(),
            });
        }
        pow.validate(1).map_err(FleetCompileError::Pow)?;
        reject_required_aliases(&compiled)?;
        let schedule = compile_schedule(&compiled, &topology, transcript)?;
        let (owners, storages, storage_bindings, output_storage) = compile_storage(
            &compiled,
            topology.coordinator,
            schedule.terminal_step,
            &schedule.barrier_steps,
            &schedule.operations,
        )?;
        let placement = FleetPlacementInput {
            topology,
            pow,
            barrier_steps: schedule.barrier_steps,
            terminal_step: schedule.terminal_step,
            barrier_arrivals: schedule.barrier_arrivals,
            operations: schedule.operations,
            owners,
            replicas: vec![],
            transitions: vec![],
            spills: vec![],
            storages,
            storage_bindings,
            in_place_aliases: vec![],
            output_storage,
        };
        Self::lower_compiled(compiled, shape, placement, transcript)
            .map_err(FleetCompileError::Lowering)
    }
}

struct CompiledSchedule {
    barrier_steps: Vec<ScheduleStep>,
    terminal_step: ScheduleStep,
    barrier_arrivals: Vec<BarrierArrival>,
    operations: Vec<FleetOperationPlacement>,
}

fn compile_schedule(
    compiled: &CompiledProof,
    topology: &FleetPlacementTopology,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<CompiledSchedule, FleetCompileError> {
    let coordinator = topology.coordinator;
    let mut cursor = ScheduleStep(0);
    let mut operations = Vec::with_capacity(compiled.operations().len());
    let mut barrier_steps = Vec::with_capacity(transcript.segments().len());

    for segment in transcript.segments() {
        for operation in compiled
            .operations()
            .iter()
            .filter(|operation| operation.stage == ProofStage::BeforeTranscript(segment.segment))
        {
            operations.push(FleetOperationPlacement {
                operation: operation.id,
                during: take_step(&mut cursor)?,
                executions: vec![FleetOperationExecution {
                    worker: coordinator,
                    domain: operation_domain(compiled, operation)?,
                }],
            });
        }
        cursor = increment(cursor)?;
        barrier_steps.push(cursor);
    }
    for operation in compiled
        .operations()
        .iter()
        .filter(|operation| operation.stage == ProofStage::AfterTranscript)
    {
        operations.push(FleetOperationPlacement {
            operation: operation.id,
            during: take_step(&mut cursor)?,
            executions: vec![FleetOperationExecution {
                worker: coordinator,
                domain: operation_domain(compiled, operation)?,
            }],
        });
    }
    operations.sort_unstable_by_key(|placement| placement.operation);
    if operations.len() != compiled.operations().len()
        || operations
            .iter()
            .zip(compiled.operations())
            .any(|(placement, operation)| placement.operation != operation.id)
    {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    let terminal_step = increment(cursor)?;
    let mut barrier_arrivals = Vec::new();
    for (ordinal, &release) in barrier_steps.iter().enumerate() {
        let ready_step = ScheduleStep(
            release
                .0
                .checked_sub(1)
                .ok_or(FleetCompileError::SizeOverflow)?,
        );
        for worker in &topology.workers {
            barrier_arrivals.push(BarrierArrival {
                barrier_ordinal: u32::try_from(ordinal)
                    .map_err(|_| FleetCompileError::SizeOverflow)?,
                worker: worker.id,
                ready_step,
            });
        }
    }
    let terminal_ordinal =
        u32::try_from(barrier_steps.len()).map_err(|_| FleetCompileError::SizeOverflow)?;
    let terminal_ready = ScheduleStep(
        terminal_step
            .0
            .checked_sub(1)
            .ok_or(FleetCompileError::SizeOverflow)?,
    );
    for worker in &topology.workers {
        barrier_arrivals.push(BarrierArrival {
            barrier_ordinal: terminal_ordinal,
            worker: worker.id,
            ready_step: terminal_ready,
        });
    }
    Ok(CompiledSchedule {
        barrier_steps,
        terminal_step,
        barrier_arrivals,
        operations,
    })
}

fn operation_domain(
    compiled: &CompiledProof,
    operation: &OpNode,
) -> Result<OperationDomain, FleetCompileError> {
    let authority = compiled
        .partitions()
        .iter()
        .find(|authority| authority.id() == operation.partition)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    Ok(match authority.kind() {
        PartitionAuthorityKind::Monolithic => OperationDomain::Monolithic,
        PartitionAuthorityKind::Exact(authority) => OperationDomain::Exact(authority.domain()),
    })
}

fn compile_storage(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    terminal_step: ScheduleStep,
    barrier_steps: &[ScheduleStep],
    operations: &[FleetOperationPlacement],
) -> Result<
    (
        Vec<FleetOwnerPlacement>,
        Vec<StorageDesc>,
        Vec<FleetStoragePlacement>,
        StorageId,
    ),
    FleetCompileError,
> {
    let owners = compile_owners(
        compiled,
        coordinator,
        terminal_step,
        barrier_steps,
        operations,
    )?;

    let mut storages = Vec::new();
    let mut bindings = Vec::new();
    for value in compiled
        .values()
        .iter()
        .filter(|value| value.region != Region::Output)
    {
        let id = next_storage_id(storages.len())?;
        storages.push(StorageDesc {
            id,
            worker: coordinator,
            bytes: value
                .layout
                .logical_bytes()
                .map_err(|_| FleetCompileError::SizeOverflow)?,
            alignment_bytes: value.alignment,
        });
        bindings.push(FleetStoragePlacement {
            storage: id,
            value: ValueRange {
                version: value.version,
                elements: full_range(value)?,
            },
            offset_bytes: 0,
        });
    }

    let output_storage = next_storage_id(storages.len())?;
    let output_alignment = compiled
        .output()
        .fragments
        .iter()
        .filter_map(|fragment| compiled.value(fragment.source.version))
        .map(|value| value.alignment)
        .max()
        .unwrap_or(core::mem::align_of::<u32>())
        .max(core::mem::align_of::<u32>());
    storages.push(StorageDesc {
        id: output_storage,
        worker: coordinator,
        bytes: compiled
            .output()
            .layout
            .total_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FleetCompileError::SizeOverflow)?,
        alignment_bytes: output_alignment,
    });
    for fragment in &compiled.output().fragments {
        bindings.push(FleetStoragePlacement {
            storage: output_storage,
            value: fragment.source,
            offset_bytes: fragment
                .destination
                .start
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(FleetCompileError::SizeOverflow)?,
        });
    }
    Ok((owners, storages, bindings, output_storage))
}

fn compile_owners(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    terminal_step: ScheduleStep,
    barrier_steps: &[ScheduleStep],
    operations: &[FleetOperationPlacement],
) -> Result<Vec<FleetOwnerPlacement>, FleetCompileError> {
    let mut lives = Vec::with_capacity(compiled.values().len());
    for value in compiled.values() {
        let start = match value.origin {
            ValueOrigin::ExternalInput(_) | ValueOrigin::Constant(_) => ScheduleStep(0),
            ValueOrigin::TranscriptOutput(_) => {
                transcript_output_release(compiled, barrier_steps, value.version)?
            }
            ValueOrigin::OpOutput(producer) => {
                operation_placement(operations, producer)?.during.start
            }
        };
        let end = match value.origin {
            ValueOrigin::Constant(_) => terminal_step,
            ValueOrigin::OpOutput(producer) => {
                increment(operation_placement(operations, producer)?.during.end)?
            }
            ValueOrigin::ExternalInput(_) | ValueOrigin::TranscriptOutput(_) => increment(start)?,
        };
        lives.push((start, end));
    }

    // Only the operation's validated outer effect is a fleet-level value
    // authority. Ordered-composite child effects are implementation detail.
    for placement in operations {
        let effect = compiled
            .effect_for(placement.operation)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        for range in effect
            .accesses()
            .iter()
            .flat_map(|access| [access.source(), access.destination()])
            .flatten()
        {
            extend_live_end(&mut lives, range.value.version, placement.during.end)?;
        }
    }
    for (segment, &release) in compiled.transcript_segments().iter().zip(barrier_steps) {
        for consumed in &segment.consumed {
            extend_live_end(&mut lives, consumed.version, release)?;
        }
    }
    for fragment in &compiled.output().fragments {
        extend_live_end(&mut lives, fragment.source.version, terminal_step)?;
    }

    compiled
        .values()
        .iter()
        .zip(lives)
        .map(|(value, (start, end))| {
            Ok(FleetOwnerPlacement {
                value: ValueRange {
                    version: value.version,
                    elements: full_range(value)?,
                },
                worker: coordinator,
                live: ScheduleRange::new(start, end)
                    .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
            })
        })
        .collect()
}

fn extend_live_end(
    lives: &mut [(ScheduleStep, ScheduleStep)],
    version: crate::compiled_proof::ValueVersion,
    end: ScheduleStep,
) -> Result<(), FleetCompileError> {
    let live = lives
        .get_mut(version.0 as usize)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    live.1 = live.1.max(end);
    Ok(())
}

fn reject_required_aliases(compiled: &CompiledProof) -> Result<(), FleetCompileError> {
    for operation in compiled.operations() {
        let effect = compiled
            .effect_for(operation.id)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        for access in effect.accesses() {
            if let Some(alias) = access
                .in_place()
                .filter(|alias| alias.requirement == InPlaceAliasRequirement::Required)
            {
                return Err(FleetCompileError::RequiredAlias {
                    operation: operation.id,
                    alias: alias.id,
                });
            }
        }
    }
    Ok(())
}

fn transcript_output_release(
    compiled: &CompiledProof,
    barrier_steps: &[ScheduleStep],
    version: crate::compiled_proof::ValueVersion,
) -> Result<ScheduleStep, FleetCompileError> {
    compiled
        .transcript_segments()
        .iter()
        .position(|segment| {
            segment
                .produced
                .iter()
                .any(|range| range.version == version)
        })
        .and_then(|ordinal| barrier_steps.get(ordinal).copied())
        .ok_or(FleetCompileError::InvalidSemanticSchedule)
}

fn operation_placement(
    operations: &[FleetOperationPlacement],
    operation: crate::compiled_proof::OpId,
) -> Result<&FleetOperationPlacement, FleetCompileError> {
    operations
        .get(operation.0 as usize)
        .filter(|placement| placement.operation == operation)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)
}

fn full_range(value: &crate::compiled_proof::ValueDesc) -> Result<ElementRange, FleetCompileError> {
    ElementRange::new(
        0,
        value
            .layout
            .element_count()
            .map_err(|_| FleetCompileError::SizeOverflow)?,
    )
    .ok_or(FleetCompileError::InvalidSemanticSchedule)
}

fn next_storage_id(index: usize) -> Result<StorageId, FleetCompileError> {
    Ok(StorageId(
        u32::try_from(index).map_err(|_| FleetCompileError::SizeOverflow)?,
    ))
}

fn take_step(cursor: &mut ScheduleStep) -> Result<ScheduleRange, FleetCompileError> {
    let end = increment(*cursor)?;
    let range = ScheduleRange::new(*cursor, end).ok_or(FleetCompileError::SizeOverflow)?;
    *cursor = end;
    Ok(range)
}

fn increment(step: ScheduleStep) -> Result<ScheduleStep, FleetCompileError> {
    step.0
        .checked_add(1)
        .map(ScheduleStep)
        .ok_or(FleetCompileError::SizeOverflow)
}

#[derive(Debug)]
pub enum FleetCompileError {
    MonolithicWorkerCount {
        actual: usize,
    },
    RequiredAlias {
        operation: crate::compiled_proof::OpId,
        alias: crate::compiled_proof::InPlaceAliasId,
    },
    InvalidSemanticSchedule,
    SizeOverflow,
    Pow(FleetPowError),
    Lowering(FleetLoweringError),
}

impl core::fmt::Display for FleetCompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "failed to compile Track-A fleet plan: {self:?}")
    }
}

impl std::error::Error for FleetCompileError {}
