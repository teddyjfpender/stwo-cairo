//! Leaf-only executable projection of the fused body and segmented tail.
//!
//! Device pointer-table storage is runtime relocation metadata. Invocation
//! identity seals its nested shape and leaves; effects name only dereferenced
//! semantic values.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    RelationAbiArgumentKind, RelationExecutionAuthority, RelationExecutionStage,
    RelationInvocationValue, RelationPartitionAuthority, RelationPointerTableKind,
    RelationValueOwnership, RelationValueRole, RelationWrapperExecution,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, DevicePointerTableBinding, EffectAccess,
    EffectBindingId, ElementRange, InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement,
    InPlaceDiscipline, LaunchGeometry, StaticCudaExecutionStepIdentity, StaticCudaLaunchIdentity,
    ValueRange,
};

pub(super) fn lower(
    execution: &RelationExecutionAuthority,
    authority: &RelationWrapperExecution,
    accesses: Vec<LoweredRelationAccess>,
    roles: &[LoweredRelationRole],
) -> Result<LoweredRelationWrapper, InvocationShapeError> {
    if execution
        .wrappers()
        .iter()
        .filter(|candidate| *candidate == authority)
        .count()
        != 1
        || authority.partition != RelationPartitionAuthority::Monolithic
        || authority.accesses.len() != accesses.len()
        || authority
            .accesses
            .iter()
            .zip(&accesses)
            .enumerate()
            .any(|(index, (exact, local))| {
                local.authority_index as usize != index
                    || local.role != exact.role
                    || local.kind != exact.kind
                    || local.arena.words != exact.words
            })
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let mut builder = BindingBuilder::new(authority.stage, roles);
    let invocation = builder.invocation(execution, authority, &accesses)?;
    let effect = EffectContract::new(builder.accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidRelationBinding)?;
    validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredRelationWrapper {
        authority: authority.clone(),
        accesses,
        invocation,
        effect,
    })
}

pub(super) fn validate(
    lowered: &LoweredRelationWrapper,
    roles: &[LoweredRelationRole],
    execution: &RelationExecutionAuthority,
) -> Result<(), InvocationShapeError> {
    let exact = lower(
        execution,
        &lowered.authority,
        lowered.accesses.clone(),
        roles,
    )?;
    (&exact == lowered)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidRelationBinding)
}

pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    authority: &RelationExecutionAuthority,
    lowered: &LoweredRelationWrapper,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    authority
        .validate()
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    let exact = authority
        .wrappers()
        .iter()
        .find(|exact| exact.stage == lowered.authority.stage)
        .filter(|exact| *exact == &lowered.authority)
        .ok_or(InvocationShapeError::InvalidRelationAuthority)?;
    let Some(linked) = authority
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?
    else {
        return Ok(None);
    };
    linked
        .validate_for_target(authority, target_sm)
        .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?;
    if linked.contract_identity() != authority.identity()
        || [
            linked.module_build_identity(),
            linked.static_build_source_identity(),
            linked.static_build_identity(),
            linked.sm_identity(),
            linked.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidRelationAuthority);
    }
    StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        linked.module_build_identity(),
        target_sm,
        exact.abi.wrapper_symbol().as_bytes().to_vec(),
        authority.abi_identity(),
        authority.effect_identity(),
        authority.identity(),
        linked.identity(),
        exact
            .children
            .iter()
            .map(|child| {
                StaticCudaLaunchIdentity::new(
                    child.symbol.as_bytes().to_vec(),
                    LaunchGeometry {
                        grid: child.grid,
                        block: child.block,
                        cluster: child.cluster,
                        dynamic_shared_bytes: child.dynamic_shared_bytes,
                        cooperative: child.cooperative,
                    },
                )
                .map(StaticCudaExecutionStepIdentity::KernelLaunch)
                .map_err(|_| InvocationShapeError::InvalidRelationAuthority)
            })
            .collect::<Result<Vec<_>, _>>()?,
        lowered
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidRelationBinding)?,
        lowered.effect.id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)
}

struct BindingBuilder<'a> {
    stage: RelationExecutionStage,
    roles: &'a [LoweredRelationRole],
    accesses: Vec<EffectAccess>,
    next_binding: u32,
    next_alias: u32,
}

impl<'a> BindingBuilder<'a> {
    const fn new(stage: RelationExecutionStage, roles: &'a [LoweredRelationRole]) -> Self {
        Self {
            stage,
            roles,
            accesses: Vec::new(),
            next_binding: 0,
            next_alias: 0,
        }
    }

    fn invocation(
        &mut self,
        execution: &RelationExecutionAuthority,
        authority: &RelationWrapperExecution,
        raw: &[LoweredRelationAccess],
    ) -> Result<AotInvocation, InvocationShapeError> {
        if authority.invocation.abi != authority.abi
            || authority.invocation.arguments.len() != authority.abi.arguments().len()
        {
            return Err(InvocationShapeError::InvalidRelationAuthority);
        }
        let mut arguments = Vec::with_capacity(authority.invocation.arguments.len() - 1);
        for (descriptor, supplied) in authority
            .abi
            .arguments()
            .iter()
            .zip(&authority.invocation.arguments)
        {
            if descriptor.ordinal != supplied.ordinal
                || descriptor.name != supplied.name
                || supplied.ordinal as usize >= authority.invocation.arguments.len()
            {
                return Err(InvocationShapeError::InvalidRelationAuthority);
            }
            let value = match supplied.value {
                RelationInvocationValue::Role(role) => {
                    self.role_argument(descriptor.kind, role, execution, raw)?
                }
                RelationInvocationValue::U32(value)
                    if descriptor.kind == RelationAbiArgumentKind::U32 =>
                {
                    AotArgumentValue::U32(value)
                }
                RelationInvocationValue::HostMask(mask)
                    if descriptor.kind == RelationAbiArgumentKind::HostConstPointerU32 =>
                {
                    AotArgumentValue::HostFixedU32(mask.to_vec())
                }
                RelationInvocationValue::OrderedStream
                    if descriptor.kind == RelationAbiArgumentKind::CudaStream =>
                {
                    continue;
                }
                _ => return Err(InvocationShapeError::InvalidRelationAuthority),
            };
            let ordinal =
                u8::try_from(arguments.len()).map_err(|_| InvocationShapeError::SizeOverflow)?;
            arguments.push(AotArgumentBinding { ordinal, value });
        }
        if arguments.len() + 1 != authority.invocation.arguments.len() {
            return Err(InvocationShapeError::InvalidRelationAuthority);
        }
        Ok(AotInvocation { arguments })
    }

    fn role_argument(
        &mut self,
        kind: RelationAbiArgumentKind,
        role: RelationValueRole,
        execution: &RelationExecutionAuthority,
        raw: &[LoweredRelationAccess],
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        match (self.stage, role, kind) {
            (
                RelationExecutionStage::FusedBody,
                RelationValueRole::DispatchPointers(RelationPointerTableKind::Sources),
                RelationAbiArgumentKind::DeviceConstPointerTableU32,
            ) => self.source_graph(execution),
            (
                RelationExecutionStage::FusedBody,
                RelationValueRole::DispatchPointers(RelationPointerTableKind::Descriptors),
                RelationAbiArgumentKind::DeviceConstPointerTableU32,
            ) => self.descriptor_table(execution, raw),
            (
                RelationExecutionStage::FusedBody,
                RelationValueRole::DispatchPointers(RelationPointerTableKind::Outputs),
                RelationAbiArgumentKind::DeviceMutPointerTableU32,
            ) => self.output_graph(execution, false),
            (
                RelationExecutionStage::SegmentedTail,
                RelationValueRole::DispatchPointers(RelationPointerTableKind::Outputs),
                RelationAbiArgumentKind::DeviceMutPointerTableU32,
            ) => self.output_graph(execution, true),
            (
                RelationExecutionStage::SegmentedTail,
                RelationValueRole::DispatchPointers(RelationPointerTableKind::ClaimedSums),
                RelationAbiArgumentKind::DeviceMutPointerTableU32,
            ) => self.claimed_sum_table(execution),
            (_, RelationValueRole::Geometry, RelationAbiArgumentKind::DeviceConstPointerU32) => {
                self.fixed_pointer(role)
            }
            (
                RelationExecutionStage::FusedBody,
                RelationValueRole::AlphaPowers | RelationValueRole::ChallengeZ,
                RelationAbiArgumentKind::DeviceConstPointerU32,
            ) => self.direct_pointer(role),
            (
                RelationExecutionStage::SegmentedTail,
                RelationValueRole::ReductionPartials | RelationValueRole::ScanBlockSums,
                RelationAbiArgumentKind::DeviceMutPointerU32,
            ) => self.direct_pointer(role),
            _ => Err(InvocationShapeError::InvalidRelationAuthority),
        }
    }

    fn source_graph(
        &mut self,
        execution: &RelationExecutionAuthority,
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        let instances = exact_instances(execution, self.roles)?;
        let entries = instances
            .iter()
            .map(|&(batch, instance, sources, _)| {
                let leaves = (0..sources)
                    .map(|source| {
                        self.bind(
                            RelationValueRole::InstanceSource {
                                batch,
                                instance,
                                source,
                            },
                            None,
                        )
                        .map(Some)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DevicePointerTableBinding { entries: leaves })
            })
            .collect::<Result<Vec<_>, InvocationShapeError>>()?;
        Ok(AotArgumentValue::DeviceNestedPointerTableValue { entries })
    }

    fn descriptor_table(
        &mut self,
        execution: &RelationExecutionAuthority,
        raw: &[LoweredRelationAccess],
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        let instances = exact_instances(execution, self.roles)?;
        let ranges = raw
            .iter()
            .filter(|access| access.role == RelationValueRole::Descriptors)
            .map(|access| access.arena)
            .collect::<Vec<_>>();
        if ranges.len() != instances.len() {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        let entries = ranges
            .into_iter()
            .map(|range| {
                self.bind(RelationValueRole::Descriptors, Some(range))
                    .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AotArgumentValue::DevicePointerTableValue(
            DevicePointerTableBinding { entries },
        ))
    }

    fn output_graph(
        &mut self,
        execution: &RelationExecutionAuthority,
        tail_only: bool,
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        let instances = exact_instances(execution, self.roles)?;
        let entries = instances
            .iter()
            .map(|&(batch, instance, _, outputs)| {
                let first = if tail_only {
                    outputs
                        .checked_sub(4)
                        .ok_or(InvocationShapeError::InvalidRelationAuthority)?
                } else {
                    0
                };
                let leaves = (0..outputs)
                    .map(|coordinate| {
                        if coordinate < first {
                            Ok(None)
                        } else {
                            self.bind(
                                RelationValueRole::OutputCoordinate {
                                    batch,
                                    instance,
                                    coordinate,
                                },
                                None,
                            )
                            .map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DevicePointerTableBinding { entries: leaves })
            })
            .collect::<Result<Vec<_>, InvocationShapeError>>()?;
        Ok(AotArgumentValue::DeviceNestedPointerTableValue { entries })
    }

    fn claimed_sum_table(
        &mut self,
        execution: &RelationExecutionAuthority,
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        let entries = exact_instances(execution, self.roles)?
            .into_iter()
            .map(|(batch, instance, ..)| {
                self.bind(RelationValueRole::ClaimedSum { batch, instance }, None)
                    .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AotArgumentValue::DevicePointerTableValue(
            DevicePointerTableBinding { entries },
        ))
    }

    fn fixed_pointer(
        &mut self,
        role: RelationValueRole,
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        let local = exact_role(self.roles, role)?;
        let value = local
            .first_version
            .filter(|version| Some(*version) == local.final_version)
            .ok_or(InvocationShapeError::InvalidRelationBinding)?;
        let binding = self.bind(role, None)?;
        Ok(AotArgumentValue::DeviceFixedU32 { value, binding })
    }

    fn direct_pointer(
        &mut self,
        role: RelationValueRole,
    ) -> Result<AotArgumentValue, InvocationShapeError> {
        self.bind(role, None)
            .map(|binding| AotArgumentValue::DevicePointer(Some(binding)))
    }

    fn bind(
        &mut self,
        role: RelationValueRole,
        range: Option<RelationArenaRange>,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let local = exact_role(self.roles, role)?;
        let arena = range.unwrap_or(local.arena);
        if arena.catalog != local.arena.catalog
            || arena.arena != local.arena.arena
            || arena.start_word < local.arena.start_word
            || arena
                .start_word
                .checked_add(arena.words)
                .is_none_or(|end| end > local.arena.start_word.saturating_add(local.arena.words))
        {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        let binding = EffectBindingId(self.next_binding);
        self.next_binding = self
            .next_binding
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let bound = |version| bound(binding, version, arena);
        let access = match (self.stage, local.ownership, role) {
            (
                _,
                RelationValueOwnership::PreparedMetadata
                | RelationValueOwnership::ExternalSource
                | RelationValueOwnership::TranscriptChallenge,
                _,
            ) => EffectAccess::Read {
                source: bound(
                    local
                        .first_version
                        .filter(|version| Some(*version) == local.final_version)
                        .ok_or(InvocationShapeError::InvalidRelationBinding)?,
                )?,
            },
            (
                RelationExecutionStage::FusedBody,
                RelationValueOwnership::ExecutionOutput,
                RelationValueRole::OutputCoordinate { .. },
            ) => EffectAccess::Write {
                destination: bound(
                    local
                        .first_version
                        .ok_or(InvocationShapeError::InvalidRelationBinding)?,
                )?,
            },
            (
                RelationExecutionStage::SegmentedTail,
                RelationValueOwnership::ExecutionOutput,
                RelationValueRole::OutputCoordinate { .. },
            ) => {
                let alias = InPlaceAliasAuthority {
                    id: InPlaceAliasId(self.next_alias),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline: InPlaceDiscipline::OrderedCompositeInPlace,
                };
                self.next_alias = self
                    .next_alias
                    .checked_add(1)
                    .ok_or(InvocationShapeError::SizeOverflow)?;
                EffectAccess::ReadWrite {
                    source: bound(
                        local
                            .first_version
                            .ok_or(InvocationShapeError::InvalidRelationBinding)?,
                    )?,
                    destination: bound(
                        local
                            .final_version
                            .filter(|version| Some(*version) != local.first_version)
                            .ok_or(InvocationShapeError::InvalidRelationBinding)?,
                    )?,
                    in_place: Some(alias),
                }
            }
            (
                RelationExecutionStage::SegmentedTail,
                RelationValueOwnership::ExecutionOutput | RelationValueOwnership::ExecutionScratch,
                RelationValueRole::ClaimedSum { .. }
                | RelationValueRole::ReductionPartials
                | RelationValueRole::ScanBlockSums,
            ) => EffectAccess::Write {
                destination: bound(
                    local
                        .final_version
                        .filter(|version| Some(*version) == local.first_version)
                        .ok_or(InvocationShapeError::InvalidRelationBinding)?,
                )?,
            },
            _ => return Err(InvocationShapeError::InvalidRelationAuthority),
        };
        self.accesses.push(access);
        Ok(binding)
    }
}

fn exact_instances(
    authority: &RelationExecutionAuthority,
    roles: &[LoweredRelationRole],
) -> Result<Vec<(u32, u32, u32, u32)>, InvocationShapeError> {
    authority
        .instances()
        .iter()
        .map(|instance| {
            let batch = instance.batch_index;
            let ordinal = instance.instance_index;
            for role in [
                RelationValueRole::InstanceSourcePointers {
                    batch,
                    instance: ordinal,
                },
                RelationValueRole::InstanceOutputPointers {
                    batch,
                    instance: ordinal,
                },
            ] {
                exact_role(roles, role)?;
            }
            Ok((
                batch,
                ordinal,
                instance.source_pointer_count,
                instance.output_coordinate_count,
            ))
        })
        .collect()
}

fn exact_role(
    roles: &[LoweredRelationRole],
    role: RelationValueRole,
) -> Result<&LoweredRelationRole, InvocationShapeError> {
    let mut matches = roles.iter().filter(|candidate| candidate.role == role);
    let exact = matches
        .next()
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok(exact)
}

fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    arena: RelationArenaRange,
) -> Result<BoundValueRange, InvocationShapeError> {
    let end = arena
        .start_word
        .checked_add(arena.words)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    Ok(BoundValueRange {
        binding,
        value: ValueRange {
            version,
            elements: ElementRange::new(arena.start_word, end)
                .ok_or(InvocationShapeError::InvalidRelationBinding)?,
        },
    })
}

fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    if invocation.arguments.is_empty()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| usize::from(argument.ordinal) != ordinal)
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|bound| bound.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for argument in &invocation.arguments {
        collect_bindings(&argument.value, &mut actual)?;
    }
    (actual == expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidRelationBinding)
}

fn collect_bindings(
    value: &AotArgumentValue,
    bindings: &mut BTreeSet<EffectBindingId>,
) -> Result<(), InvocationShapeError> {
    let mut insert = |binding| {
        bindings
            .insert(binding)
            .then_some(())
            .ok_or(InvocationShapeError::InvalidRelationBinding)
    };
    match value {
        AotArgumentValue::U32(_) | AotArgumentValue::HostFixedU32(_) => Ok(()),
        AotArgumentValue::DevicePointer(Some(binding))
        | AotArgumentValue::DeviceFixedU32 { binding, value: _ } => insert(*binding),
        AotArgumentValue::DevicePointerTableValue(table) => {
            for &binding in table.entries.iter().flatten() {
                insert(binding)?;
            }
            Ok(())
        }
        AotArgumentValue::DeviceNestedPointerTableValue { entries } => {
            for entry in entries {
                for &binding in entry.entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            Ok(())
        }
        _ => Err(InvocationShapeError::InvalidRelationBinding),
    }
}
