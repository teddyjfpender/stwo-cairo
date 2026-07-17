//! Typed semantic invocations for linked static CUDA wrappers.
//!
//! CUDA streams are execution-context state and are intentionally absent from
//! `AotInvocation`. Every other host-wrapper ABI ordinal is retained exactly,
//! and every effect binding must appear once.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    BlakeGDirectAbiAccess, BlakeGDirectAbiArgument, BlakeGDirectAbiArgumentKind, EcOpAbiAccess,
    EcOpAbiArgument, EcOpAbiArgumentKind, EcOpCompositeAbi, EcOpEffectAbi, EcOpKernelStage,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, EffectBindingId, EffectContract,
};

pub(super) fn blake_g_direct(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
) -> Result<AotInvocation, InvocationShapeError> {
    blake_g_direct_using_abi(contract, contract.authority.abi().arguments())
}

fn blake_g_direct_using_abi(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    abi: &[BlakeGDirectAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    blake_g_direct_prefix::validate_lowered(
        &contract.authority,
        &contract.invocation,
        &contract.effect,
    )?;
    if abi.len() != 7 {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        if let Some(value) = blake_g_direct_value(contract, *descriptor)? {
            arguments.push(argument(descriptor.ordinal, value));
        }
    }
    let invocation = AotInvocation { arguments };
    validate_exact_bindings(&invocation, &contract.effect)
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    Ok(invocation)
}

fn blake_g_direct_value(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    descriptor: BlakeGDirectAbiArgument,
) -> Result<Option<AotArgumentValue>, InvocationShapeError> {
    use {BlakeGDirectAbiAccess as Access, BlakeGDirectAbiArgumentKind as Kind};

    let invocation = &contract.invocation;
    let value = match (
        descriptor.ordinal,
        descriptor.name,
        descriptor.kind,
        descriptor.access,
    ) {
        (
            0,
            "input_cols_host",
            Kind::HostConstDevicePointerTableU32,
            Access::ReadSixInputColumns,
        ) => AotArgumentValue::DevicePointerTable(
            invocation
                .inputs
                .iter()
                .map(|binding| Some(binding.binding))
                .collect(),
        ),
        (1, "n_rows", Kind::U32, Access::RealRowCount) => {
            AotArgumentValue::U32(invocation.n_real_rows)
        }
        (2, "column_length", Kind::U32, Access::PaddedRowCount) => {
            AotArgumentValue::U32(invocation.padded_rows)
        }
        (
            3,
            "trace_cols_host",
            Kind::HostMutDevicePointerTableU32,
            Access::WriteFiftyThreeTraceColumns,
        ) => AotArgumentValue::DevicePointerTable(
            invocation
                .traces
                .iter()
                .map(|binding| Some(binding.binding))
                .collect(),
        ),
        (4, "luts_host", Kind::HostConstDevicePointerTableU32, Access::ReadFourCanonicalLuts) => {
            AotArgumentValue::DevicePointerTable(
                invocation
                    .luts
                    .iter()
                    .map(|binding| Some(binding.binding))
                    .collect(),
            )
        }
        (
            5,
            "counts_host",
            Kind::HostMutDevicePointerTableU32,
            Access::AtomicAddFiveCanonicalCountDestinations,
        ) => AotArgumentValue::DevicePointerTable(
            invocation
                .counts
                .iter()
                .map(|binding| Some(binding.binding))
                .collect(),
        ),
        (6, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => return Ok(None),
        _ => return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority),
    };
    Ok(Some(value))
}

pub(super) fn ec_op(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
) -> Result<AotInvocation, InvocationShapeError> {
    ec_op_using_abi(contract, contract.authority.abi().arguments())
}

fn ec_op_using_abi(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
    abi: &[EcOpAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    if ec_op_prefix::exact_effect(&contract.invocation)? != contract.effect {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let requirements = contract.authority.requirements();
    let counts = &contract.invocation.multiplicities;
    if contract.authority.abi() != EcOpCompositeAbi::ProjectiveChainNormalizePaddingV1
        || contract.authority.effect()
            != EcOpEffectAbi::FullTraceLookupPartialAndAtomicMultiplicitiesV1
        || (*contract.authority.launches()).map(|launch| launch.stage)
            != [
                EcOpKernelStage::ProjectiveChain,
                EcOpKernelStage::NormalizeRoundTiles,
                EcOpKernelStage::PartialInputPadding,
            ]
        || [
            contract.authority.source_identity(),
            contract.authority.abi_identity(),
            contract.authority.effect_identity(),
            contract.authority.launch_identity(),
            contract.authority.identity(),
        ]
        .contains(&[0; 32])
        || abi.len() != 19
        || contract.invocation.execution_tables.len() != EXECUTION_TABLE_POINTERS
        || contract.invocation.trace_columns.len() != requirements.trace_column_words.len()
        || contract.invocation.partial_input_columns.len()
            != requirements.partial_input_column_words.len()
        || counts.len() != 4
        || contract.invocation.row_count as usize != requirements.row_count
        || contract.invocation.partial_row_count as usize != requirements.partial_row_count
    {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        if let Some(value) = ec_op_value(contract, *descriptor)? {
            arguments.push(argument(descriptor.ordinal, value));
        }
    }
    let invocation = AotInvocation { arguments };
    validate_exact_bindings(&invocation, &contract.effect)
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpBinding)?;
    Ok(invocation)
}

fn ec_op_value(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
    descriptor: EcOpAbiArgument,
) -> Result<Option<AotArgumentValue>, InvocationShapeError> {
    use {EcOpAbiAccess as Access, EcOpAbiArgumentKind as Kind};

    let invocation = &contract.invocation;
    let requirements = contract.authority.requirements();
    let tables = contract.authority.execution_tables();
    let counts = &invocation.multiplicities;
    let value = match (
        descriptor.ordinal,
        descriptor.name,
        descriptor.kind,
        descriptor.access,
    ) {
        (0, "execution_tables", Kind::DevicePointerTableU32, Access::Read) => {
            AotArgumentValue::DevicePointerTable(
                invocation
                    .execution_tables
                    .iter()
                    .map(|binding| binding.binding)
                    .collect(),
            )
        }
        (1, "n_addresses", Kind::U32, Access::ExecutionTableShape) => {
            AotArgumentValue::U32(to_u32(tables.n_addresses)?)
        }
        (2, "n_big", Kind::U32, Access::ExecutionTableShape) => {
            AotArgumentValue::U32(to_u32(tables.n_big)?)
        }
        (3, "n_small", Kind::U32, Access::ExecutionTableShape) => {
            AotArgumentValue::U32(to_u32(tables.n_small)?)
        }
        (4, "segment_start_source", Kind::DevicePointerU32, Access::Read) => {
            AotArgumentValue::DevicePointer(invocation.segment_start.binding)
        }
        (5, "row_count", Kind::U32, Access::RowCount) => {
            AotArgumentValue::U32(invocation.row_count)
        }
        (6, "trace_columns_host", Kind::HostPointerTableU32, Access::Write) => {
            AotArgumentValue::DevicePointerTable(
                invocation
                    .trace_columns
                    .iter()
                    .map(|binding| binding.binding)
                    .collect(),
            )
        }
        (7, "lookup_words", Kind::DevicePointerU32, Access::Write) => {
            AotArgumentValue::DevicePointer(invocation.lookup_words.binding)
        }
        (8, "partial_input_columns_host", Kind::HostPointerTableU32, Access::Write) => {
            AotArgumentValue::DevicePointerTable(
                invocation
                    .partial_input_columns
                    .iter()
                    .map(|binding| binding.binding)
                    .collect(),
            )
        }
        (9, "partial_row_count", Kind::U32, Access::PartialRowCount) => {
            AotArgumentValue::U32(invocation.partial_row_count)
        }
        (10, "address_counts", Kind::DevicePointerU32, Access::AtomicAddU32) => {
            AotArgumentValue::DevicePointer(Some(counts[0].binding))
        }
        (11, "address_count_words", Kind::U32, Access::DestinationWords) => {
            AotArgumentValue::U32(to_u32(requirements.address_count_words)?)
        }
        (12, "big_counts", Kind::DevicePointerU32, Access::AtomicAddU32) => {
            AotArgumentValue::DevicePointer(Some(counts[1].binding))
        }
        (13, "big_count_words", Kind::U32, Access::DestinationWords) => {
            AotArgumentValue::U32(to_u32(requirements.big_count_words)?)
        }
        (14, "small_counts", Kind::DevicePointerU32, Access::AtomicAddU32) => {
            AotArgumentValue::DevicePointer(Some(counts[2].binding))
        }
        (15, "small_count_words", Kind::U32, Access::DestinationWords) => {
            AotArgumentValue::U32(to_u32(requirements.small_count_words)?)
        }
        (16, "range_check_8_counts", Kind::DevicePointerU32, Access::AtomicAddU32) => {
            AotArgumentValue::DevicePointer(Some(counts[3].binding))
        }
        (17, "range_check_8_count_words", Kind::U32, Access::DestinationWords) => {
            AotArgumentValue::U32(to_u32(requirements.range_check_8_count_words)?)
        }
        (18, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => return Ok(None),
        _ => return Err(InvocationShapeError::InvalidNativeEcOpAuthority),
    };
    Ok(Some(value))
}

#[cfg(test)]
pub(super) fn blake_g_direct_using_abi_for_test(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    abi: &[BlakeGDirectAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    blake_g_direct_using_abi(contract, abi)
}

#[cfg(test)]
pub(super) fn ec_op_using_abi_for_test(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
    abi: &[EcOpAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    ec_op_using_abi(contract, abi)
}

#[cfg(test)]
pub(super) fn validate_blake_g_direct_invocation_for_test(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    invocation: &AotInvocation,
) -> Result<(), InvocationShapeError> {
    let expected = blake_g_direct(contract)?;
    (invocation == &expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

#[cfg(test)]
pub(super) fn validate_ec_op_invocation_for_test(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
    invocation: &AotInvocation,
) -> Result<(), InvocationShapeError> {
    let expected = ec_op(contract)?;
    (invocation == &expected)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)
}

fn argument(ordinal: u8, value: AotArgumentValue) -> AotArgumentBinding {
    AotArgumentBinding { ordinal, value }
}

fn to_u32(value: usize) -> Result<u32, InvocationShapeError> {
    u32::try_from(value).map_err(|_| InvocationShapeError::SizeOverflow)
}

fn validate_exact_bindings(invocation: &AotInvocation, effect: &EffectContract) -> Result<(), ()> {
    if invocation.arguments.is_empty()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(());
    }
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::<EffectBindingId>::new();
    for argument in &invocation.arguments {
        match &argument.value {
            AotArgumentValue::U32(_) | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(&mut actual, *binding)?,
            AotArgumentValue::DevicePointerTable(bindings) => {
                if bindings.is_empty() {
                    return Err(());
                }
                for &binding in bindings.iter().flatten() {
                    insert(&mut actual, binding)?;
                }
            }
            AotArgumentValue::DeviceFixedU32 { .. } => return Err(()),
        }
    }
    (actual == expected).then_some(()).ok_or(())
}

fn insert(bindings: &mut BTreeSet<EffectBindingId>, binding: EffectBindingId) -> Result<(), ()> {
    bindings.insert(binding).then_some(()).ok_or(())
}
