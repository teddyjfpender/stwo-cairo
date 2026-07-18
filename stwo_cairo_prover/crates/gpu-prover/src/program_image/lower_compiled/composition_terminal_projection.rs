//! Proof-wide projection of Composition lifts and the terminal split.
//!
//! Ordinary CUDA wrappers are one compiled-proof primitive. Their producer
//! authority still seals every ordered child launch; this projection exposes
//! only the exact outer value transition visible at each wrapper boundary.

use std::collections::{BTreeMap, BTreeSet};

use super::composition_projection::LoweredCompositionWaves;
use super::{adapter, ArenaCatalogValueId, InvocationShapeError};
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, BoundValueRange, EffectAccess,
    EffectBindingId, EffectContract, ElementRange, InPlaceAliasAuthority, InPlaceAliasId,
    InPlaceAliasRequirement, InPlaceDiscipline, LaunchGeometry, StaticCudaExecutionStepIdentity,
    StaticCudaLaunchIdentity, StaticCudaWrapperAuthority, StaticCudaWrapperId, ValueRange,
    ValueVersion,
};
use crate::prepared_composition::{
    CompositionAbi, CompositionAccessKind, CompositionExecutionAuthority,
    CompositionInvocationValue, CompositionLayout, CompositionOperation, CompositionOperationKind,
    CompositionValueAccess, CompositionValueRole,
};

const COORDINATES: usize = 4;
const RETAINED: usize = 8;
const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-composition-terminal.v1\0";

mod binding;
mod validation;
use binding::TerminalBuilder;
use validation::{receipt_digest, validate_invocation_bindings, validate_receipt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionTerminalBinding {
    pub(super) binding: EffectBindingId,
    pub(super) source_role: Option<CompositionValueRole>,
    pub(super) destination_role: Option<CompositionValueRole>,
    pub(super) arena: CompositionLayout,
    pub(super) source_version: Option<ValueVersion>,
    pub(super) destination_version: Option<ValueVersion>,
    pub(super) role_elements: ElementRange,
    pub(super) elements: ElementRange,
    pub(super) kind: CompositionAccessKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionTerminalOperation {
    operation_ordinal: u32,
    operation: CompositionOperation,
    invocation: AotInvocation,
    effect: EffectContract,
    bindings: Vec<LoweredCompositionTerminalBinding>,
}

impl LoweredCompositionTerminalOperation {
    pub(super) const fn operation_ordinal(&self) -> u32 {
        self.operation_ordinal
    }

    pub(super) const fn operation(&self) -> &CompositionOperation {
        &self.operation
    }

    pub(super) const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }

    pub(super) fn bindings(&self) -> &[LoweredCompositionTerminalBinding] {
        &self.bindings
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionTerminalOutput {
    pub(super) role: CompositionValueRole,
    pub(super) arena: CompositionLayout,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionTerminal {
    authority: CompositionExecutionAuthority,
    operations: Vec<LoweredCompositionTerminalOperation>,
    outputs: [LoweredCompositionTerminalOutput; RETAINED],
    digest: [u8; 32],
}

impl LoweredCompositionTerminal {
    pub(super) const fn authority(&self) -> &CompositionExecutionAuthority {
        &self.authority
    }

    pub(super) fn operations(&self) -> &[LoweredCompositionTerminalOperation] {
        &self.operations
    }

    pub(super) const fn outputs(&self) -> &[LoweredCompositionTerminalOutput; RETAINED] {
        &self.outputs
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

pub(super) fn required_upstream_catalogs(
    arena: &ProofArenaPlan,
) -> Result<Vec<ArenaCatalogValueId>, InvocationShapeError> {
    validation::required_upstream_catalogs(arena)
}

/// Lower all post-wave Composition wrappers transactionally.
pub(super) fn lower_terminal(
    arena: &ProofArenaPlan,
    waves: &LoweredCompositionWaves,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionTerminal, InvocationShapeError> {
    let authority = CompositionExecutionAuthority::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    if waves.authority() != &authority {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let layouts = exact_layouts(&authority)?;
    let mut next_values = values.clone();
    let mut versions = wave_outputs(waves, &layouts, &next_values)?;
    let first = authority
        .operations()
        .len()
        .checked_sub(2)
        .and_then(|split| split.checked_sub(lift_count(&authority)))
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let expected_first = 2usize
        .checked_add(waves.waves().len())
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if first != expected_first {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }

    let mut operations = Vec::with_capacity(authority.operations().len() - first);
    for (ordinal, operation) in authority.operations().iter().enumerate().skip(first) {
        let lowered = match operation.kind {
            CompositionOperationKind::LiftAccumulate { .. } => lower_lift(
                ordinal,
                operation,
                &layouts,
                &mut versions,
                &mut next_values,
            )?,
            CompositionOperationKind::SplitInverseFusedFirstForward { .. } => lower_split_inverse(
                ordinal,
                operation,
                &layouts,
                &mut versions,
                &mut next_values,
            )?,
            CompositionOperationKind::SplitForwardAfterFirstInterval { .. } => lower_split_forward(
                ordinal,
                operation,
                &layouts,
                &mut versions,
                &mut next_values,
            )?,
            _ => return Err(InvocationShapeError::InvalidCompositionAuthority),
        };
        operations.push(lowered);
    }
    if operations.len() != lift_count(&authority) + 2 {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let outputs = authority.outputs().map(|role| {
        let arena = layouts
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionBinding)?;
        let version = versions
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionBinding)?;
        if next_values.version(catalog(arena))? != version {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        Ok(LoweredCompositionTerminalOutput {
            role,
            arena,
            version,
        })
    });
    let outputs = outputs
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?;
    let digest = receipt_digest(&authority, waves, &operations, &outputs)?;
    let lowered = LoweredCompositionTerminal {
        authority,
        operations,
        outputs,
        digest,
    };
    validate_receipt(arena, waves, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    waves: &LoweredCompositionWaves,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredCompositionTerminal,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_terminal(arena, waves, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidCompositionBinding)
    }
}

/// Bind one projected ordinary-CUDA wrapper to the exact installed module.
pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    arena: &ProofArenaPlan,
    waves: &LoweredCompositionWaves,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    terminal: &LoweredCompositionTerminal,
    operation_ordinal: u32,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate_from(arena, waves, before, after, terminal)?;
    terminal
        .authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let lowered = terminal
        .operations
        .iter()
        .find(|operation| operation.operation_ordinal == operation_ordinal)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let exact = terminal
        .authority
        .operations()
        .get(operation_ordinal as usize)
        .filter(|exact| *exact == &lowered.operation)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let Some(linked) = terminal
        .authority
        .bind_linked(arena, target_sm)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?
    else {
        return Ok(None);
    };
    linked
        .validate(&terminal.authority, arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        linked.static_module_build_identity(),
        target_sm,
        exact.abi.wrapper_symbol().as_bytes().to_vec(),
        exact.abi_identity,
        exact.effect_identity,
        exact.identity,
        linked.identity(),
        exact
            .children
            .iter()
            .map(|child| {
                StaticCudaLaunchIdentity::new(
                    child.symbol.as_bytes().to_vec(),
                    LaunchGeometry {
                        grid: child.launch.grid,
                        block: child.launch.block,
                        cluster: None,
                        dynamic_shared_bytes: child.launch.dynamic_shared_bytes,
                        cooperative: child.launch.cooperative,
                    },
                )
                .map(StaticCudaExecutionStepIdentity::KernelLaunch)
                .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)
            })
            .collect::<Result<Vec<_>, _>>()?,
        lowered
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?,
        lowered.effect.id(),
    )
    .map(Some)
    .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)
}

fn lower_lift(
    ordinal: usize,
    operation: &CompositionOperation,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
    versions: &mut BTreeMap<CompositionValueRole, ValueVersion>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionTerminalOperation, InvocationShapeError> {
    let CompositionOperationKind::LiftAccumulate {
        previous_log_size,
        current_log_size,
        ..
    } = operation.kind
    else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    if operation.abi != CompositionAbi::LiftAccumulateV1 {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let [child] = operation.children.as_slice() else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    let mut builder = TerminalBuilder::new(layouts, versions, values);
    let previous = (0..COORDINATES)
        .map(|coordinate| {
            let access = child_access(child, coordinate)?;
            let role = read_role(access)?;
            expect_accumulator(role, previous_log_size, coordinate)?;
            builder.read_ephemeral(role, access.elements)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let current = (0..COORDINATES)
        .map(|coordinate| {
            let access = child_access(child, COORDINATES + coordinate)?;
            let (source, destination) = transition_roles(access)?;
            let source_generation = expect_accumulator(source, current_log_size, coordinate)?;
            if expect_accumulator(destination, current_log_size, coordinate)?
                != source_generation + 1
            {
                return Err(InvocationShapeError::InvalidCompositionAuthority);
            }
            builder.transition_ephemeral(
                source,
                destination,
                access.elements,
                InPlaceDiscipline::ElementWiseReadBeforeWrite,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if child.effect.accesses.len() != COORDINATES * 2 {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let invocation = project_invocation(
        operation,
        BTreeMap::from([
            (
                0,
                AotArgumentValue::DevicePointerRangeSetValue { ranges: previous },
            ),
            (
                2,
                AotArgumentValue::DevicePointerRangeSetValue { ranges: current },
            ),
        ]),
    )?;
    builder.finish(ordinal, operation, invocation)
}

fn lower_split_inverse(
    ordinal: usize,
    operation: &CompositionOperation,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
    versions: &mut BTreeMap<CompositionValueRole, ValueVersion>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionTerminalOperation, InvocationShapeError> {
    let CompositionOperationKind::SplitInverseFusedFirstForward {
        evaluation_log_size,
    } = operation.kind
    else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    if operation.abi != CompositionAbi::SplitInverseFusedFirstForwardV1 {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let [first, second, boundary] = operation.children.as_slice() else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    let mut builder = TerminalBuilder::new(layouts, versions, values);
    let sources = (0..COORDINATES)
        .map(|coordinate| {
            let one = child_access(first, coordinate)?;
            let two = child_access(second, coordinate)?;
            let (source, middle) = transition_roles(one)?;
            let (same_middle, destination) = transition_roles(two)?;
            let first_generation = expect_accumulator(source, evaluation_log_size, coordinate)?;
            if middle != same_middle
                || expect_accumulator(middle, evaluation_log_size, coordinate)?
                    != first_generation + 1
                || expect_accumulator(destination, evaluation_log_size, coordinate)?
                    != first_generation + 2
                || read_role(child_access(boundary, coordinate)?)? != destination
            {
                return Err(InvocationShapeError::InvalidCompositionAuthority);
            }
            builder.transition_ephemeral(
                source,
                destination,
                one.elements,
                InPlaceDiscipline::OrderedCompositeInPlace,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if first.effect.accesses.len() != COORDINATES + 1
        || second.effect.accesses.len() != COORDINATES + 1
        || boundary.effect.accesses.len() != COORDINATES + RETAINED + 2
        || read_role(child_access(first, COORDINATES)?)? != CompositionValueRole::InverseTwiddles
        || read_role(child_access(second, COORDINATES)?)? != CompositionValueRole::InverseTwiddles
    {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let retained = (0..RETAINED)
        .map(|column| {
            let access = child_access(boundary, COORDINATES + column)?;
            let role = write_role(access)?;
            expect_retained(role, column, 1)?;
            builder.write_catalog(role, access.elements)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let inverse =
        builder.read_catalog(read_role(child_access(boundary, COORDINATES + RETAINED)?)?)?;
    let forward = builder.read_catalog(read_role(child_access(
        boundary,
        COORDINATES + RETAINED + 1,
    )?)?)?;
    let invocation = project_invocation(
        operation,
        BTreeMap::from([
            (
                0,
                AotArgumentValue::DevicePointerTable(sources.into_iter().map(Some).collect()),
            ),
            (
                1,
                AotArgumentValue::DevicePointerTable(retained.into_iter().map(Some).collect()),
            ),
            (3, AotArgumentValue::DevicePointer(Some(inverse))),
            (5, AotArgumentValue::DevicePointer(Some(forward))),
        ]),
    )?;
    builder.finish(ordinal, operation, invocation)
}

fn lower_split_forward(
    ordinal: usize,
    operation: &CompositionOperation,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
    versions: &mut BTreeMap<CompositionValueRole, ValueVersion>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionTerminalOperation, InvocationShapeError> {
    let CompositionOperationKind::SplitForwardAfterFirstInterval { .. } = operation.kind else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    if operation.abi != CompositionAbi::SplitForwardAfterFirstIntervalV1 {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let [first, second] = operation.children.as_slice() else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    let mut builder = TerminalBuilder::new(layouts, versions, values);
    let retained = (0..RETAINED)
        .map(|column| {
            let one = child_access(first, column)?;
            let two = child_access(second, column)?;
            let (source, middle) = transition_roles(one)?;
            let (same_middle, destination) = transition_roles(two)?;
            if middle != same_middle {
                return Err(InvocationShapeError::InvalidCompositionAuthority);
            }
            expect_retained(source, column, 1)?;
            expect_retained(middle, column, 2)?;
            expect_retained(destination, column, 3)?;
            builder.transition_catalog(
                source,
                destination,
                one.elements,
                InPlaceDiscipline::OrderedCompositeInPlace,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if first.effect.accesses.len() != RETAINED + 1
        || second.effect.accesses.len() != RETAINED + 1
        || read_role(child_access(first, RETAINED)?)? != CompositionValueRole::ForwardTwiddles
        || read_role(child_access(second, RETAINED)?)? != CompositionValueRole::ForwardTwiddles
    {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let forward = builder.read_catalog(CompositionValueRole::ForwardTwiddles)?;
    let invocation = project_invocation(
        operation,
        BTreeMap::from([
            (
                0,
                AotArgumentValue::DevicePointerTable(retained.into_iter().map(Some).collect()),
            ),
            (3, AotArgumentValue::DevicePointer(Some(forward))),
        ]),
    )?;
    builder.finish(ordinal, operation, invocation)
}

fn project_invocation(
    operation: &CompositionOperation,
    mut pointers: BTreeMap<u8, AotArgumentValue>,
) -> Result<AotInvocation, InvocationShapeError> {
    let mut arguments = Vec::with_capacity(operation.invocation.arguments.len() - 1);
    for source in &operation.invocation.arguments {
        if matches!(source.value, CompositionInvocationValue::ExecutionStream) {
            if source.ordinal as usize + 1 != operation.invocation.arguments.len() {
                return Err(InvocationShapeError::InvalidCompositionAuthority);
            }
            continue;
        }
        if source.ordinal as usize != arguments.len() {
            return Err(InvocationShapeError::InvalidCompositionAuthority);
        }
        let value = match pointers.remove(&source.ordinal) {
            Some(value)
                if matches!(
                    source.value,
                    CompositionInvocationValue::Access(_)
                        | CompositionInvocationValue::PointerTable { .. }
                ) =>
            {
                value
            }
            None => match source.value {
                CompositionInvocationValue::U32(value) => AotArgumentValue::U32(value),
                _ => return Err(InvocationShapeError::InvalidCompositionAuthority),
            },
            _ => return Err(InvocationShapeError::InvalidCompositionAuthority),
        };
        arguments.push(AotArgumentBinding {
            ordinal: source.ordinal,
            value,
        });
    }
    if !pointers.is_empty() {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    Ok(AotInvocation { arguments })
}

fn read_role(
    access: &CompositionValueAccess,
) -> Result<CompositionValueRole, InvocationShapeError> {
    if access.kind == CompositionAccessKind::Read && access.destination.is_none() {
        access
            .source
            .ok_or(InvocationShapeError::InvalidCompositionAuthority)
    } else {
        Err(InvocationShapeError::InvalidCompositionAuthority)
    }
}

fn write_role(
    access: &CompositionValueAccess,
) -> Result<CompositionValueRole, InvocationShapeError> {
    if access.kind == CompositionAccessKind::Write && access.source.is_none() {
        access
            .destination
            .ok_or(InvocationShapeError::InvalidCompositionAuthority)
    } else {
        Err(InvocationShapeError::InvalidCompositionAuthority)
    }
}

fn transition_roles(
    access: &CompositionValueAccess,
) -> Result<(CompositionValueRole, CompositionValueRole), InvocationShapeError> {
    if access.kind != CompositionAccessKind::ReadWriteRequired {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    access
        .source
        .zip(access.destination)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)
}

fn expect_accumulator(
    role: CompositionValueRole,
    log_size: u32,
    coordinate: usize,
) -> Result<u8, InvocationShapeError> {
    match role {
        CompositionValueRole::Accumulator {
            log_size: actual_log,
            coordinate: actual_coordinate,
            generation,
        } if actual_log == log_size && actual_coordinate as usize == coordinate => Ok(generation),
        _ => Err(InvocationShapeError::InvalidCompositionAuthority),
    }
}

fn expect_retained(
    role: CompositionValueRole,
    column: usize,
    generation: u8,
) -> Result<(), InvocationShapeError> {
    matches!(
        role,
        CompositionValueRole::SplitRetained {
            canonical_column,
            generation: actual_generation,
        } if canonical_column as usize == column && actual_generation == generation
    )
    .then_some(())
    .ok_or(InvocationShapeError::InvalidCompositionAuthority)
}

fn child_access(
    child: &crate::prepared_composition::CompositionChildLaunch,
    index: usize,
) -> Result<&CompositionValueAccess, InvocationShapeError> {
    child
        .effect
        .accesses
        .get(index)
        .filter(|access| access.binding.0 as usize == index)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)
}

fn wave_outputs(
    waves: &LoweredCompositionWaves,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
    values: &adapter::SemanticValueMap,
) -> Result<BTreeMap<CompositionValueRole, ValueVersion>, InvocationShapeError> {
    let allocated = values.allocated_versions().collect::<BTreeSet<_>>();
    let mut outputs = BTreeMap::new();
    for wave in waves.waves() {
        let mut count = 0;
        for binding in wave.bindings() {
            if binding.kind != CompositionAccessKind::Write {
                continue;
            }
            let CompositionValueRole::Accumulator { generation: 0, .. } = binding.role else {
                return Err(InvocationShapeError::InvalidCompositionBinding);
            };
            if layouts.get(&binding.role) != Some(&binding.arena)
                || !allocated.contains(&binding.version)
                || outputs.insert(binding.role, binding.version).is_some()
            {
                return Err(InvocationShapeError::InvalidCompositionBinding);
            }
            count += 1;
        }
        if count != COORDINATES {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
    }
    Ok(outputs)
}

fn exact_layouts(
    authority: &CompositionExecutionAuthority,
) -> Result<BTreeMap<CompositionValueRole, CompositionLayout>, InvocationShapeError> {
    let layouts = authority
        .layouts()
        .iter()
        .map(|layout| (layout.role, *layout))
        .collect::<BTreeMap<_, _>>();
    if layouts.len() != authority.layouts().len() {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    Ok(layouts)
}

fn lift_count(authority: &CompositionExecutionAuthority) -> usize {
    authority
        .operations()
        .iter()
        .filter(|operation| {
            matches!(
                operation.kind,
                CompositionOperationKind::LiftAccumulate { .. }
            )
        })
        .count()
}

fn full(layout: CompositionLayout) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(0, layout.word_len).ok_or(InvocationShapeError::InvalidCompositionBinding)
}

fn offset(
    layout: CompositionLayout,
    relative: ElementRange,
) -> Result<ElementRange, InvocationShapeError> {
    if relative.is_empty() || relative.end > layout.word_len {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    ElementRange::new(
        layout
            .first_word
            .checked_add(relative.start)
            .ok_or(InvocationShapeError::SizeOverflow)?,
        layout
            .first_word
            .checked_add(relative.end)
            .ok_or(InvocationShapeError::SizeOverflow)?,
    )
    .ok_or(InvocationShapeError::InvalidCompositionBinding)
}

const fn catalog(layout: CompositionLayout) -> ArenaCatalogValueId {
    ArenaCatalogValueId(layout.logical.0)
}

const fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}

#[cfg(test)]
#[path = "composition_terminal_projection_tests.rs"]
mod tests;
