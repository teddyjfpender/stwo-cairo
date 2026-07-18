//! Exact ordered child-manifest projection for one BaseCommit wrapper.

use stwo_backend_cuda::{
    BaseCommitDependencyRange, BaseCommitExecutionBuffer, BaseCommitExecutionStep,
    BaseCommitLinkedAuthority, BaseCommitOperation, BaseCommitPointerTarget,
};

use super::*;
use crate::compiled_proof::{
    LaunchGeometry, StaticCudaExecutionStepIdentity, StaticCudaLaunchIdentity,
    StaticCudaLibraryCallIdentity, StaticCudaMemcpyD2DV1,
};

pub(super) fn project_wrapper(
    id: StaticCudaWrapperId,
    linked: &BaseCommitLinkedAuthority,
    lowered: &LoweredBaseCommit,
    operation_ordinal: usize,
) -> Result<StaticCudaWrapperAuthority, InvocationShapeError> {
    linked
        .validate(&lowered.authority)
        .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
    let local = lowered
        .operations
        .get(operation_ordinal)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    let exact = lowered
        .authority
        .operations()
        .get(operation_ordinal)
        .filter(|exact| local.ordinal as usize == operation_ordinal && *exact == &local.authority)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        exact.abi.wrapper_symbol().as_bytes().to_vec(),
        exact.abi_identity,
        exact.effect.identity,
        exact.identity,
        linked.identity(),
        project_steps(exact)?,
        local
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidBaseCommitBinding)?,
        local.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)
}

pub(super) fn project_steps(
    operation: &BaseCommitOperation,
) -> Result<Vec<StaticCudaExecutionStepIdentity>, InvocationShapeError> {
    operation
        .execution
        .iter()
        .map(|step| match step {
            BaseCommitExecutionStep::KernelLaunch(launch) => {
                let launch = StaticCudaLaunchIdentity::new(
                    launch.symbol.as_bytes().to_vec(),
                    LaunchGeometry {
                        grid: launch.grid,
                        block: launch.block,
                        cluster: launch.cluster,
                        dynamic_shared_bytes: launch.dynamic_shared_bytes,
                        cooperative: launch.cooperative,
                    },
                )
                .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
                Ok(StaticCudaExecutionStepIdentity::KernelLaunch(launch))
            }
            BaseCommitExecutionStep::DeviceCopyD2D {
                source,
                destination,
                bytes,
            } => {
                let (source_argument, source_byte_offset) = wrapper_buffer(*source)?;
                let (destination_argument, destination_byte_offset) = wrapper_buffer(*destination)?;
                validate_copy_range(operation, source_argument, source_byte_offset, *bytes)?;
                validate_copy_range(
                    operation,
                    destination_argument,
                    destination_byte_offset,
                    *bytes,
                )?;
                let copy = StaticCudaMemcpyD2DV1::new(
                    source_argument,
                    source_byte_offset,
                    destination_argument,
                    destination_byte_offset,
                    *bytes,
                )
                .map_err(|_| InvocationShapeError::InvalidBaseCommitAuthority)?;
                Ok(StaticCudaExecutionStepIdentity::LibraryCall(
                    StaticCudaLibraryCallIdentity::MemcpyD2DV1(copy),
                ))
            }
        })
        .collect()
}

fn wrapper_buffer(buffer: BaseCommitExecutionBuffer) -> Result<(u8, u64), InvocationShapeError> {
    match buffer {
        BaseCommitExecutionBuffer::WrapperArgument {
            ordinal,
            byte_offset,
        } => Ok((ordinal, byte_offset)),
        BaseCommitExecutionBuffer::DependencySuffix { .. } => {
            Err(InvocationShapeError::InvalidBaseCommitAuthority)
        }
    }
}

fn validate_copy_range(
    operation: &BaseCommitOperation,
    ordinal: u8,
    byte_offset: u64,
    bytes: u64,
) -> Result<(), InvocationShapeError> {
    let capacity = wrapper_capacity_words(operation, ordinal)?
        .checked_mul(core::mem::size_of::<u32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(InvocationShapeError::SizeOverflow)?;
    match byte_offset.checked_add(bytes) {
        Some(end) if end <= capacity => Ok(()),
        _ => Err(InvocationShapeError::InvalidBaseCommitAuthority),
    }
}

fn wrapper_capacity_words(
    operation: &BaseCommitOperation,
    ordinal: u8,
) -> Result<usize, InvocationShapeError> {
    let target = operation
        .effect
        .pointer_bindings
        .iter()
        .find(|binding| binding.argument_ordinal == ordinal)
        .map(|binding| &binding.target)
        .ok_or(InvocationShapeError::InvalidBaseCommitAuthority)?;
    match target {
        BaseCommitPointerTarget::Values { access_indices } => access_indices
            .iter()
            .map(|&index| {
                let access = operation
                    .effect
                    .accesses
                    .get(index as usize)
                    .ok_or(InvocationShapeError::InvalidBaseCommitAuthority)?;
                access
                    .first_word
                    .checked_add(access.word_len)
                    .ok_or(InvocationShapeError::SizeOverflow)
            })
            .try_fold(0usize, |capacity, end| end.map(|end| capacity.max(end))),
        BaseCommitPointerTarget::PointerTable { table, .. } => range_words(table.range),
        BaseCommitPointerTarget::Installed { access } => range_words(access.range),
        BaseCommitPointerTarget::ExecutionStream => {
            Err(InvocationShapeError::InvalidBaseCommitAuthority)
        }
    }
}

fn range_words(range: BaseCommitDependencyRange) -> Result<usize, InvocationShapeError> {
    match range {
        BaseCommitDependencyRange::Whole { words }
        | BaseCommitDependencyRange::Suffix { words }
        | BaseCommitDependencyRange::Slice { words, .. }
            if words != 0 =>
        {
            Ok(words)
        }
        _ => Err(InvocationShapeError::InvalidBaseCommitAuthority),
    }
}
