use std::collections::{BTreeMap, BTreeSet};

use super::*;

pub(super) fn validate(
    input: &CompiledProofInput,
    operation: &OpNode,
    effect: &EffectContract,
) -> Result<(), CompiledProofError> {
    match &operation.primitive {
        ExecutionPrimitive::OrderedComposite { children } => {
            validate_composite(input, operation, effect, children)
        }
        primitive => validate_step(
            input,
            operation.id,
            primitive,
            operation.invocation.as_ref(),
            operation.effect,
            operation.partition,
            effect,
        ),
    }
}

fn validate_step(
    input: &CompiledProofInput,
    operation: OpId,
    primitive: &ExecutionPrimitive,
    invocation: Option<&AotInvocation>,
    effect_id: EffectContractId,
    partition: PartitionAuthorityId,
    effect: &EffectContract,
) -> Result<(), CompiledProofError> {
    match primitive {
        ExecutionPrimitive::AotKernel { kernel, launch } => {
            if !valid_launch(*launch) {
                return Err(CompiledProofError::InvalidLaunchGeometry(operation));
            }
            let authority = input
                .kernels
                .iter()
                .find(|authority| authority.id() == *kernel)
                .ok_or(CompiledProofError::UnknownKernel { operation })?;
            if authority
                .accepted_executions()
                .binary_search(&(effect_id, partition))
                .is_err()
            {
                return Err(CompiledProofError::KernelEffectNotAccepted { operation });
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
                    return Err(CompiledProofError::ModuleGlobalAuthorityMismatch { operation });
                }
            }
            validate_invocation(
                input,
                invocation,
                effect,
                CompiledProofError::InvalidKernelInvocation(operation),
            )?;
        }
        ExecutionPrimitive::StaticCudaWrapper { wrapper } => {
            let authority = input
                .static_wrappers
                .iter()
                .find(|authority| authority.id() == *wrapper)
                .ok_or(CompiledProofError::UnknownStaticWrapper { operation })?;
            if partition != PartitionAuthority::monolithic().id() {
                return Err(CompiledProofError::StaticWrapperRequiresMonolithic { operation });
            }
            if authority.accepted_effect() != effect_id {
                return Err(CompiledProofError::StaticWrapperEffectNotAccepted { operation });
            }
            if !effect.module_globals().is_empty() {
                return Err(CompiledProofError::ModuleGlobalAuthorityMismatch { operation });
            }
            validate_invocation(
                input,
                invocation,
                effect,
                CompiledProofError::InvalidStaticWrapperInvocation(operation),
            )?;
        }
        ExecutionPrimitive::DeviceCopyD2D { bytes } => {
            if invocation.is_some()
                || !effect.module_globals().is_empty()
                || *bytes == 0
                || effect.accesses().len() != 2
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            }
            let (EffectAccess::Read { source }, EffectAccess::Write { destination }) =
                (&effect.accesses()[0], &effect.accesses()[1])
            else {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            };
            if super::range_bytes(input, source.value)? != *bytes
                || super::range_bytes(input, destination.value)? != *bytes
                || super::value(input, source.value.version)?.layout
                    != super::value(input, destination.value.version)?.layout
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            }
        }
        ExecutionPrimitive::DeviceMemsetByte { bytes, .. } => {
            if invocation.is_some()
                || !effect.module_globals().is_empty()
                || *bytes == 0
                || effect.accesses().len() != 1
            {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            }
            let EffectAccess::Write { destination } = &effect.accesses()[0] else {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            };
            if super::range_bytes(input, destination.value)? != *bytes {
                return Err(CompiledProofError::PrimitiveEffectMismatch(operation));
            }
        }
        ExecutionPrimitive::OrderedComposite { .. } => {
            return Err(CompiledProofError::InvalidOrderedComposite {
                operation,
                child: None,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BoundaryAccess {
    Read(ValueRange),
    Write(ValueRange),
    ReadWrite {
        source: ValueRange,
        destination: ValueRange,
        alias: Option<(InPlaceAliasRequirement, InPlaceDiscipline)>,
    },
    Atomic {
        source: ValueRange,
        destination: ValueRange,
        operation: AtomicOperation,
        alias: (InPlaceAliasRequirement, InPlaceDiscipline),
    },
}

fn validate_composite(
    input: &CompiledProofInput,
    operation: &OpNode,
    effect: &EffectContract,
    children: &[ExecutableStep],
) -> Result<(), CompiledProofError> {
    if children.is_empty()
        || operation.invocation.is_some()
        || operation.partition != PartitionAuthority::monolithic().id()
    {
        return Err(CompiledProofError::InvalidOrderedComposite {
            operation: operation.id,
            child: None,
        });
    }

    let mut initialized = BTreeMap::<ValueVersion, Vec<ElementRange>>::new();
    let mut expected_boundary = BTreeSet::new();
    let mut expected_globals = Vec::new();
    for (child_index, child) in children.iter().enumerate() {
        if matches!(
            child.primitive,
            ExecutionPrimitive::OrderedComposite { .. }
                | ExecutionPrimitive::StaticCudaWrapper { .. }
        ) {
            return Err(CompiledProofError::InvalidOrderedComposite {
                operation: operation.id,
                child: Some(child_index),
            });
        }
        let child_effect =
            super::effect(input, child.effect).ok_or(CompiledProofError::UnknownEffect {
                operation: operation.id,
            })?;
        validate_step(
            input,
            operation.id,
            &child.primitive,
            child.invocation.as_ref(),
            child.effect,
            operation.partition,
            child_effect,
        )?;
        expected_globals.extend_from_slice(child_effect.module_globals());

        // Effect accesses are an unordered memory contract, not an execution
        // sequence. A write made by this child cannot initialize a separate
        // read in the same child; only an earlier child can do that.
        let mut child_destinations = Vec::new();
        for access in child_effect.accesses() {
            if let Some(source) = access.source() {
                super::validate_bound_range(input, operation.id, *source)?;
                let internal = super::value(input, source.value.version)?.origin
                    == ValueOrigin::OpOutput(operation.id);
                if internal
                    && !range_is_initialized(
                        initialized.get(&source.value.version),
                        source.value.elements,
                    )
                {
                    return Err(CompiledProofError::CompositeUninitializedRead {
                        operation: operation.id,
                        child: child_index,
                        value: source.value.version,
                    });
                }
            }
            expected_boundary.insert(boundary_access(access));

            if let Some(destination) = access.destination() {
                super::validate_bound_range(input, operation.id, *destination)?;
                let value = super::value(input, destination.value.version)?;
                if value.origin != ValueOrigin::OpOutput(operation.id) {
                    return Err(CompiledProofError::ProducerMismatch {
                        value: value.version,
                    });
                }
                child_destinations.push((value.version, destination.value.elements));
            }
        }
        for (version, elements) in child_destinations {
            let ranges = initialized.entry(version).or_default();
            if ranges.iter().any(|range| range.overlaps(elements)) {
                return Err(CompiledProofError::OverlappingWrite { value: version });
            }
            ranges.push(elements);
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
        }
    }
    canonicalize_globals(&mut expected_globals);

    let actual_boundary = effect
        .accesses()
        .iter()
        .map(boundary_access)
        .collect::<BTreeSet<_>>();
    if actual_boundary.len() != effect.accesses().len() || actual_boundary != expected_boundary {
        return Err(CompiledProofError::CompositeBoundaryEffectMismatch(
            operation.id,
        ));
    }
    if effect.module_globals() != expected_globals {
        return Err(CompiledProofError::CompositeBoundaryEffectMismatch(
            operation.id,
        ));
    }
    Ok(())
}

fn boundary_access(access: &EffectAccess) -> BoundaryAccess {
    match access {
        EffectAccess::Read { source } => BoundaryAccess::Read(source.value),
        EffectAccess::Write { destination } => BoundaryAccess::Write(destination.value),
        EffectAccess::ReadWrite {
            source,
            destination,
            in_place,
        } => BoundaryAccess::ReadWrite {
            source: source.value,
            destination: destination.value,
            alias: in_place.map(|alias| (alias.requirement, alias.discipline)),
        },
        EffectAccess::Atomic {
            source,
            destination,
            operation,
            in_place,
        } => BoundaryAccess::Atomic {
            source: source.value,
            destination: destination.value,
            operation: *operation,
            alias: (in_place.requirement, in_place.discipline),
        },
    }
}

fn canonicalize_globals(globals: &mut Vec<ModuleGlobalEffect>) {
    globals.sort();
    let mut canonical = Vec::<ModuleGlobalEffect>::with_capacity(globals.len());
    for global in globals.drain(..) {
        if let Some(previous) = canonical.last_mut() {
            if previous.initializer == global.initializer
                && previous.bytes.end >= global.bytes.start
            {
                previous.bytes.end = previous.bytes.end.max(global.bytes.end);
                continue;
            }
        }
        canonical.push(global);
    }
    *globals = canonical;
}

fn range_is_initialized(ranges: Option<&Vec<ElementRange>>, required: ElementRange) -> bool {
    let mut cursor = required.start;
    for range in ranges.into_iter().flatten() {
        if range.end <= cursor {
            continue;
        }
        if range.start > cursor {
            return false;
        }
        cursor = cursor.max(range.end);
        if cursor >= required.end {
            return true;
        }
    }
    false
}

fn validate_invocation(
    input: &CompiledProofInput,
    invocation: Option<&AotInvocation>,
    effect: &EffectContract,
    error: CompiledProofError,
) -> Result<(), CompiledProofError> {
    let invalid = || error.clone();
    let invocation = invocation.ok_or_else(invalid)?;
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
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(None) => {}
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
