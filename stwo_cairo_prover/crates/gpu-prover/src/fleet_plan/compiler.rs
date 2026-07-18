//! Deterministic Track-A physical lowering.
//!
//! The monolithic compiler is the exact one-worker reference. The partitioned
//! compiler derives conservative multi-worker exact shards and transfers from
//! the same semantic authority. The one-worker compiler realizes only exact,
//! whole-value required aliases whose validated effects and lifetimes prove
//! read-before-overwrite safety. All other storage reuse remains fail-closed.

use std::sync::Arc;

use super::*;
use crate::compiled_proof::{
    exact_partial_atomic_carry_forward, CompiledProof, ExecutionPrimitive, InPlaceAliasId,
    InPlaceAliasRequirement, InPlaceDiscipline, OpId, OpNode, PartitionAuthorityKind, ProofStage,
    Region, ValueOrigin, ValueRange, ValueVersion,
};
use crate::fleet_pow::{FleetPowError, FleetPowSchedule};
use crate::shape_executable::ShapeExecutableIdentity;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

mod distributed;
mod statement_host_ingress;

use statement_host_ingress::{
    compile_components as compile_ingress_components, statement_host_destination,
    statement_host_reuses,
};

impl FleetProofPlan {
    /// Compile one validated semantic proof into a deterministic, fail-closed
    /// single-rank physical plan. Use `compile_track_a_partitioned` for
    /// conservative multi-worker exact shards. Storage sharing is limited to
    /// exact required aliases and validated statement-ingress lineages.
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
        let schedule = compile_schedule(&compiled, &topology, transcript)?;
        let (owners, storages, storage_bindings, in_place_aliases, output_storage) =
            compile_storage(
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
            in_place_aliases,
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

#[derive(Clone, Copy)]
struct RequiredAlias {
    operation: OpId,
    alias: InPlaceAliasId,
    source: ValueRange,
    destination: ValueRange,
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
            reserve_statement_overwrite_epoch(operation, &mut cursor)?;
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
        reserve_statement_overwrite_epoch(operation, &mut cursor)?;
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

fn reserve_statement_overwrite_epoch(
    operation: &OpNode,
    cursor: &mut ScheduleStep,
) -> Result<(), FleetCompileError> {
    // A predecessor dies at this operation's start. Keep one logical epoch
    // between its producer-ready edge and the overwrite edge.
    if matches!(
        operation.primitive,
        ExecutionPrimitive::StatementHostIngress {
            predecessor: Some(_),
            ..
        }
    ) {
        *cursor = increment(*cursor)?;
    }
    Ok(())
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
        Vec<InPlaceAliasPlacement>,
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
    let required_aliases = compile_required_aliases(compiled, &owners, operations)?;
    let alias_components = compile_alias_components(compiled.values().len(), &required_aliases)?;
    let ingress_reuses = statement_host_reuses(compiled)?;
    let ingress_components = compile_ingress_components(compiled.values().len(), &ingress_reuses)?;

    let mut storages = Vec::<StorageDesc>::new();
    let mut bindings = Vec::new();
    let mut value_storages = vec![None::<StorageId>; compiled.values().len()];
    let mut component_storages = vec![None::<StorageId>; required_aliases.len()];
    let mut ingress_storages = vec![None::<StorageId>; ingress_reuses.len()];
    for value in compiled
        .values()
        .iter()
        .filter(|value| value.region != Region::Output)
    {
        let value_index = value.version.0 as usize;
        let component = alias_components[value_index];
        let ingress_component = ingress_components[value_index];
        if component.is_some() && ingress_component.is_some() {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
        let existing = component
            .and_then(|component| component_storages[component])
            .or_else(|| ingress_component.and_then(|component| ingress_storages[component]));
        let bytes = value
            .layout
            .logical_bytes()
            .map_err(|_| FleetCompileError::SizeOverflow)?;
        let id = match existing {
            Some(id) => {
                let storage = storages
                    .get_mut(id.0 as usize)
                    .filter(|storage| storage.id == id)
                    .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
                storage.bytes = storage.bytes.max(bytes);
                storage.alignment_bytes = storage.alignment_bytes.max(value.alignment);
                id
            }
            None => {
                let id = next_storage_id(storages.len())?;
                storages.push(StorageDesc {
                    id,
                    worker: coordinator,
                    bytes,
                    alignment_bytes: value.alignment,
                });
                if let Some(component) = component {
                    component_storages[component] = Some(id);
                }
                if let Some(component) = ingress_component {
                    ingress_storages[component] = Some(id);
                }
                id
            }
        };
        value_storages[value_index] = Some(id);
        bindings.push(FleetStoragePlacement {
            storage: id,
            value: ValueRange {
                version: value.version,
                elements: full_range(value)?,
            },
            offset_bytes: 0,
        });
    }
    let in_place_aliases = required_aliases
        .iter()
        .map(|alias| {
            Ok(InPlaceAliasPlacement {
                operation: alias.operation,
                alias: alias.alias,
                storage: value_storages
                    .get(alias.source.version.0 as usize)
                    .copied()
                    .flatten()
                    .ok_or(FleetCompileError::InvalidSemanticSchedule)?,
                offset_bytes: alias
                    .source
                    .elements
                    .start
                    .checked_mul(
                        compiled
                            .value(alias.source.version)
                            .ok_or(FleetCompileError::InvalidSemanticSchedule)?
                            .layout
                            .element
                            .bytes,
                    )
                    .ok_or(FleetCompileError::SizeOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, FleetCompileError>>()?;

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
    Ok((owners, storages, bindings, in_place_aliases, output_storage))
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
    for operation in compiled.operations().iter().filter(|operation| {
        matches!(
            operation.primitive,
            ExecutionPrimitive::StatementHostIngress { .. }
        )
    }) {
        let destination = statement_host_destination(compiled, operation.id)?;
        let live = lives
            .get_mut(destination.version.0 as usize)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        live.1 = terminal_step;
    }
    for reuse in statement_host_reuses(compiled)? {
        let overwrite = operation_placement(operations, reuse.operation)?
            .during
            .start;
        let live = lives
            .get_mut(reuse.predecessor.version.0 as usize)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        live.1 = overwrite;
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

fn compile_required_aliases(
    compiled: &CompiledProof,
    owners: &[FleetOwnerPlacement],
    operations: &[FleetOperationPlacement],
) -> Result<Vec<RequiredAlias>, FleetCompileError> {
    let mut aliases = Vec::new();
    for operation in compiled.operations() {
        let effect = compiled
            .effect_for(operation.id)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        for (access_index, access) in effect.accesses().iter().enumerate() {
            let Some(authority) = access
                .in_place()
                .filter(|alias| alias.requirement == InPlaceAliasRequirement::Required)
            else {
                continue;
            };
            let invalid = || FleetCompileError::RequiredAlias {
                operation: operation.id,
                alias: authority.id,
            };
            let placement = operation_placement(operations, operation.id)?;
            let source = access.source().ok_or_else(invalid)?.value;
            let destination = access.destination().ok_or_else(invalid)?.value;
            let source_value = compiled.value(source.version).ok_or_else(invalid)?;
            let destination_value = compiled.value(destination.version).ok_or_else(invalid)?;
            let whole_value = source.elements == full_range(source_value)?
                && destination.elements == full_range(destination_value)?;
            let exact_carried_prefix =
                exact_partial_atomic_carry_forward(compiled.values(), access).is_some();
            if matches!(
                &operation.primitive,
                ExecutionPrimitive::OrderedComposite { .. }
            ) || !matches!(
                placement.executions.as_slice(),
                [FleetOperationExecution {
                    domain: OperationDomain::Monolithic,
                    ..
                }]
            ) || !(whole_value || exact_carried_prefix)
                || !required_alias_layouts_match(
                    authority.discipline,
                    source_value,
                    destination_value,
                )
                || source_value.alignment != destination_value.alignment
                || source_value.region == Region::FixedData
                || destination_value.region == Region::FixedData
                || source_value.region == Region::Output
                || destination_value.region == Region::Output
                || matches!(source_value.origin, ValueOrigin::Constant(_))
                || matches!(destination_value.origin, ValueOrigin::Constant(_))
                || effect
                    .accesses()
                    .iter()
                    .enumerate()
                    .any(|(other_index, other)| {
                        other_index != access_index
                            && other
                                .source()
                                .is_some_and(|other| other.value.version == source.version)
                    })
            {
                return Err(invalid());
            }
            let source_owner = owner(owners, source.version).ok_or_else(invalid)?;
            let destination_owner = owner(owners, destination.version).ok_or_else(invalid)?;
            if source_owner.live.end != placement.during.end
                || destination_owner.live.start != placement.during.start
            {
                return Err(invalid());
            }
            aliases.push(RequiredAlias {
                operation: operation.id,
                alias: authority.id,
                source,
                destination,
            });
        }
    }
    Ok(aliases)
}

fn required_alias_layouts_match(
    discipline: InPlaceDiscipline,
    source: &crate::compiled_proof::ValueDesc,
    destination: &crate::compiled_proof::ValueDesc,
) -> bool {
    if source.layout.element != destination.layout.element {
        return false;
    }
    match discipline {
        InPlaceDiscipline::ExactLowerPrefixReadBeforeWrite => source
            .layout
            .element_count()
            .ok()
            .zip(destination.layout.element_count().ok())
            .is_some_and(|(source, destination)| source < destination),
        InPlaceDiscipline::OrderedCompositeInPlace => true,
        InPlaceDiscipline::ElementWiseReadBeforeWrite
        | InPlaceDiscipline::BlockBarrierPhases
        | InPlaceDiscipline::CooperativeGridPhases => source.layout == destination.layout,
    }
}

fn compile_alias_components(
    value_count: usize,
    aliases: &[RequiredAlias],
) -> Result<Vec<Option<usize>>, FleetCompileError> {
    let mut incoming = vec![None; value_count];
    let mut outgoing = vec![None; value_count];
    for (edge, alias) in aliases.iter().enumerate() {
        let source = alias.source.version.0 as usize;
        let destination = alias.destination.version.0 as usize;
        if source >= value_count
            || destination >= value_count
            || outgoing[source].replace(edge).is_some()
            || incoming[destination].replace(edge).is_some()
        {
            return Err(required_alias_error(alias));
        }
    }

    let mut components = vec![None; value_count];
    let mut visited = vec![false; aliases.len()];
    let mut component = 0usize;
    for root in 0..value_count {
        if incoming[root].is_some() || outgoing[root].is_none() {
            continue;
        }
        let mut version = root;
        loop {
            if components[version].replace(component).is_some() {
                let edge = outgoing[version]
                    .or(incoming[version])
                    .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
                return Err(required_alias_error(&aliases[edge]));
            }
            let Some(edge) = outgoing[version] else {
                break;
            };
            if visited[edge] {
                return Err(required_alias_error(&aliases[edge]));
            }
            visited[edge] = true;
            version = aliases[edge].destination.version.0 as usize;
            if incoming[version] != Some(edge) {
                return Err(required_alias_error(&aliases[edge]));
            }
        }
        component = component
            .checked_add(1)
            .ok_or(FleetCompileError::SizeOverflow)?;
    }
    if let Some((edge, _)) = visited.iter().enumerate().find(|(_, visited)| !**visited) {
        return Err(required_alias_error(&aliases[edge]));
    }
    Ok(components)
}

fn required_alias_error(alias: &RequiredAlias) -> FleetCompileError {
    FleetCompileError::RequiredAlias {
        operation: alias.operation,
        alias: alias.alias,
    }
}

fn owner(owners: &[FleetOwnerPlacement], version: ValueVersion) -> Option<&FleetOwnerPlacement> {
    owners.iter().find(|owner| owner.value.version == version)
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
    UnsupportedExactComposite {
        operation: crate::compiled_proof::OpId,
    },
    InvalidOwnership(crate::compiled_proof::ValueVersion),
    MissingRoute {
        source: WorkerId,
        destination: WorkerId,
    },
    TransferTooLarge {
        route: FleetLinkId,
        bytes: usize,
        limit: usize,
    },
    Projection(FleetPlanError),
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
