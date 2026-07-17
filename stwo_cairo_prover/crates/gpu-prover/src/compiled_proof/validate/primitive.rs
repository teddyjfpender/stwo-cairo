use std::collections::BTreeSet;

use super::*;

pub(super) fn validate(
    input: &CompiledProofInput,
    operation: &OpNode,
    effect: &EffectContract,
) -> Result<(), CompiledProofError> {
    match operation.primitive {
        ExecutionPrimitive::AotKernel { kernel, launch } => {
            if !valid_launch(launch) {
                return Err(CompiledProofError::InvalidLaunchGeometry(operation.id));
            }
            let authority = input
                .kernels
                .iter()
                .find(|authority| authority.id() == kernel)
                .ok_or(CompiledProofError::UnknownKernel {
                    operation: operation.id,
                })?;
            if authority
                .accepted_effects()
                .binary_search(&operation.effect)
                .is_err()
            {
                return Err(CompiledProofError::KernelEffectNotAccepted {
                    operation: operation.id,
                });
            }
            for global in effect.module_globals() {
                let initializer = input
                    .module_global_initializers
                    .get(global.initializer.0 as usize)
                    .filter(|initializer| initializer.id() == global.initializer)
                    .ok_or(CompiledProofError::UnknownModuleGlobalInitializer(
                        global.initializer,
                    ))?;
                if initializer.module() != authority.module()
                    || global.bytes.end > initializer.bytes()
                {
                    return Err(CompiledProofError::ModuleGlobalAuthorityMismatch {
                        operation: operation.id,
                    });
                }
            }
            validate_invocation(input, operation, effect)?;
        }
        ExecutionPrimitive::DeviceCopyD2D { bytes } => {
            if operation.invocation.is_some()
                || !effect.module_globals().is_empty()
                || bytes == 0
                || effect.accesses().len() != 2
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            }
            let (EffectAccess::Read { source }, EffectAccess::Write { destination }) =
                (&effect.accesses()[0], &effect.accesses()[1])
            else {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            };
            if super::range_bytes(input, source.value)? != bytes
                || super::range_bytes(input, destination.value)? != bytes
                || super::value(input, source.value.version)?.layout
                    != super::value(input, destination.value.version)?.layout
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            }
        }
        ExecutionPrimitive::DeviceMemsetByte { bytes, .. } => {
            if operation.invocation.is_some()
                || !effect.module_globals().is_empty()
                || bytes == 0
                || effect.accesses().len() != 1
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            }
            let EffectAccess::Write { destination } = &effect.accesses()[0] else {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            };
            if super::range_bytes(input, destination.value)? != bytes {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation.id));
            }
        }
    }
    Ok(())
}

fn validate_invocation(
    input: &CompiledProofInput,
    operation: &OpNode,
    effect: &EffectContract,
) -> Result<(), CompiledProofError> {
    let invalid = || CompiledProofError::InvalidKernelInvocation(operation.id);
    let invocation = operation.invocation.as_ref().ok_or_else(invalid)?;
    if invocation.arguments.is_empty() {
        return Err(invalid());
    }

    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|bound| bound.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for (ordinal, argument) in invocation.arguments.iter().enumerate() {
        if usize::from(argument.ordinal) != ordinal {
            return Err(invalid());
        }
        let mut insert = |binding| {
            if actual.insert(binding) {
                Ok(())
            } else {
                Err(invalid())
            }
        };
        match &argument.value {
            AotArgumentValue::U32(_) | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding)?,
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() {
                    return Err(invalid());
                }
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DeviceFixedU32 {
                value: fixed_version,
                binding,
            } => {
                insert(*binding)?;
                let range = effect
                    .accesses()
                    .iter()
                    .flat_map(|access| [access.source(), access.destination()])
                    .flatten()
                    .find(|range| range.binding == *binding)
                    .ok_or_else(invalid)?;
                let fixed = input
                    .fixed_values
                    .iter()
                    .find(|fixed| fixed.value() == *fixed_version)
                    .ok_or_else(invalid)?;
                let fixed_value = super::value(input, *fixed_version).map_err(|_| invalid())?;
                let full = ElementRange::new(
                    0,
                    fixed_value.layout.element_count().map_err(|_| invalid())?,
                )
                .ok_or_else(invalid)?;
                if range.value.version != *fixed_version
                    || range.value.elements != full
                    || fixed_value.alignment < core::mem::align_of::<u32>()
                    || !matches!(fixed.initializer(), FixedValueInitializer::InlineU32(_))
                {
                    return Err(invalid());
                }
            }
        }
    }
    if actual != expected {
        return Err(invalid());
    }
    Ok(())
}

fn valid_launch(launch: LaunchGeometry) -> bool {
    let block_threads = launch
        .block
        .into_iter()
        .try_fold(1u64, |product, value| product.checked_mul(u64::from(value)));
    launch.grid[0] != 0
        && launch.grid[0] <= i32::MAX as u32
        && launch.grid[1] != 0
        && launch.grid[1] <= u16::MAX as u32
        && launch.grid[2] != 0
        && launch.grid[2] <= u16::MAX as u32
        && launch.block[0] != 0
        && launch.block[0] <= 1024
        && launch.block[1] != 0
        && launch.block[1] <= 1024
        && launch.block[2] != 0
        && launch.block[2] <= 64
        // Cluster limits depend on the installed target. This target-neutral
        // authority cannot truthfully admit one.
        && launch.cluster.is_none()
        && block_threads.is_some_and(|threads| threads <= 1024)
}
