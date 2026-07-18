//! Stream-free canonical invocation projection for BaseCommit wrappers.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    BaseCommitAbiArgumentKind, BaseCommitDependencyRange, BaseCommitDependencyRole,
    BaseCommitInvocationValue, BaseCommitOperation, BaseCommitPointerTarget,
};

use super::semantic::LocalPointer;
use super::*;
use crate::compiled_proof::{AotArgumentBinding, AotArgumentValue};

pub(super) fn compile(
    operation: &BaseCommitOperation,
    pointers: &[LocalPointer],
    inventory: &BaseCommitInventory,
) -> Result<AotInvocation, InvocationShapeError> {
    if pointers.len() != operation.effect.pointer_bindings.len()
        || operation.invocation.arguments.len() != operation.abi.arguments().len()
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let capacity = operation
        .invocation
        .arguments
        .len()
        .checked_sub(1)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    let mut arguments = Vec::with_capacity(capacity);
    for (descriptor, supplied) in operation
        .abi
        .arguments()
        .iter()
        .zip(&operation.invocation.arguments)
    {
        if descriptor.ordinal != supplied.ordinal
            || descriptor.name != supplied.name
            || descriptor.kind != supplied.kind
        {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        let value = match supplied.value {
            BaseCommitInvocationValue::PointerEffect {
                binding_index,
                installed_range,
            } => {
                let index = usize::try_from(binding_index)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?;
                let upstream = operation
                    .effect
                    .pointer_bindings
                    .get(index)
                    .filter(|binding| binding.argument_ordinal == supplied.ordinal)
                    .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
                if pointer_range(&upstream.target) != installed_range {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                match pointers
                    .get(index)
                    .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?
                {
                    LocalPointer::Single(binding) => {
                        AotArgumentValue::DevicePointer(Some(*binding))
                    }
                    LocalPointer::Table(bindings) => AotArgumentValue::DevicePointerTable(
                        bindings.iter().copied().map(Some).collect(),
                    ),
                    LocalPointer::Stream => {
                        return Err(InvocationShapeError::InvalidBaseCommitBinding)
                    }
                }
            }
            BaseCommitInvocationValue::U32(value) => {
                if descriptor.kind != BaseCommitAbiArgumentKind::U32 {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                AotArgumentValue::U32(value)
            }
            BaseCommitInvocationValue::Unsigned(value) => {
                if descriptor.kind != BaseCommitAbiArgumentKind::Unsigned {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                AotArgumentValue::U32(value)
            }
            BaseCommitInvocationValue::InstalledWordLength { dependency } => {
                if !matches!(
                    descriptor.kind,
                    BaseCommitAbiArgumentKind::U32 | BaseCommitAbiArgumentKind::Unsigned
                ) {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                AotArgumentValue::U32(installed_words(operation, inventory, dependency)?)
            }
            BaseCommitInvocationValue::ExecutionStream => {
                if descriptor.kind != BaseCommitAbiArgumentKind::CudaStream
                    || !matches!(
                        pointers
                            .iter()
                            .zip(&operation.effect.pointer_bindings)
                            .find(|(_, binding)| binding.argument_ordinal == supplied.ordinal)
                            .map(|(local, _)| local),
                        Some(LocalPointer::Stream)
                    )
                {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                continue;
            }
        };
        let ordinal =
            u8::try_from(arguments.len()).map_err(|_| InvocationShapeError::SizeOverflow)?;
        if supplied.ordinal != ordinal {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        arguments.push(AotArgumentBinding { ordinal, value });
    }
    if arguments.len() != capacity {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(AotInvocation { arguments })
}

fn installed_words(
    operation: &BaseCommitOperation,
    inventory: &BaseCommitInventory,
    dependency: BaseCommitDependencyRole,
) -> Result<u32, InvocationShapeError> {
    let mut matches = operation
        .effect
        .pointer_bindings
        .iter()
        .filter_map(|binding| match binding.target {
            BaseCommitPointerTarget::Installed { access } if access.role == dependency => {
                Some(access)
            }
            _ => None,
        });
    let access = matches
        .next()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let (_, _, elements) = inventory.installed_value(access.role, access.range)?;
    u32::try_from(elements.len()).map_err(|_| InvocationShapeError::SizeOverflow)
}

fn pointer_range(target: &BaseCommitPointerTarget) -> Option<BaseCommitDependencyRange> {
    match target {
        BaseCommitPointerTarget::PointerTable { table, .. } => Some(table.range),
        BaseCommitPointerTarget::Installed { access } => Some(access.range),
        BaseCommitPointerTarget::Values { .. } | BaseCommitPointerTarget::ExecutionStream => None,
    }
}

pub(super) fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    if invocation.arguments.is_empty()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for argument in &invocation.arguments {
        match &argument.value {
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::HostFixedU32(_) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => {
                if !actual.insert(*binding) {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
            }
            AotArgumentValue::DevicePointerTable(bindings) => {
                if bindings.is_empty() {
                    return Err(InvocationShapeError::InvalidBaseCommitBinding);
                }
                for binding in bindings.iter().flatten() {
                    if !actual.insert(*binding) {
                        return Err(InvocationShapeError::InvalidBaseCommitBinding);
                    }
                }
            }
            AotArgumentValue::DevicePointerTableValue(_)
            | AotArgumentValue::DeviceNestedPointerTableValue { .. } => {
                return Err(InvocationShapeError::InvalidBaseCommitBinding);
            }
            _ => return Err(InvocationShapeError::InvalidBaseCommitBinding),
        }
    }
    (actual == expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}
