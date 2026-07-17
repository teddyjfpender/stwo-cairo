//! Typed semantic invocations for linked static CUDA wrappers.
//!
//! CUDA streams are execution-context state and are intentionally absent from
//! `AotInvocation`. Every other host-wrapper ABI ordinal is retained exactly,
//! and every effect binding must appear once.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    BlakeGDirectAbiAccess, EcOpAbiAccess, EcOpCompositeAbi, EcOpEffectAbi, EcOpKernelStage,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, EffectBindingId, EffectContract,
};

pub(super) fn blake_g_direct(
    contract: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
) -> Result<AotInvocation, InvocationShapeError> {
    blake_g_direct_prefix::validate_lowered(
        &contract.authority,
        &contract.invocation,
        &contract.effect,
    )?;
    let abi = contract.authority.abi().arguments();
    if abi.len() != 7
        || abi[..6]
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
        || abi[6].access != BlakeGDirectAbiAccess::OrderedExecutionStream
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    let invocation = AotInvocation {
        arguments: vec![
            argument(
                0,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .inputs
                        .iter()
                        .map(|binding| Some(binding.binding))
                        .collect(),
                ),
            ),
            argument(1, AotArgumentValue::U32(contract.invocation.n_real_rows)),
            argument(2, AotArgumentValue::U32(contract.invocation.padded_rows)),
            argument(
                3,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .traces
                        .iter()
                        .map(|binding| Some(binding.binding))
                        .collect(),
                ),
            ),
            argument(
                4,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .luts
                        .iter()
                        .map(|binding| Some(binding.binding))
                        .collect(),
                ),
            ),
            argument(
                5,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .counts
                        .iter()
                        .map(|binding| Some(binding.binding))
                        .collect(),
                ),
            ),
        ],
    };
    validate_exact_bindings(&invocation, &contract.effect)
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    Ok(invocation)
}

pub(super) fn ec_op(
    contract: &ec_op_prefix::LoweredNativeEcOpContract,
) -> Result<AotInvocation, InvocationShapeError> {
    if ec_op_prefix::exact_effect(&contract.invocation)? != contract.effect {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    let abi = contract.authority.abi().arguments();
    let requirements = contract.authority.requirements();
    let tables = contract.authority.execution_tables();
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
        || abi[..18]
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
        || abi[18].access != EcOpAbiAccess::OrderedExecutionStream
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
    let invocation = AotInvocation {
        arguments: vec![
            argument(
                0,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .execution_tables
                        .iter()
                        .map(|binding| binding.binding)
                        .collect(),
                ),
            ),
            argument(1, AotArgumentValue::U32(to_u32(tables.n_addresses)?)),
            argument(2, AotArgumentValue::U32(to_u32(tables.n_big)?)),
            argument(3, AotArgumentValue::U32(to_u32(tables.n_small)?)),
            argument(
                4,
                AotArgumentValue::DevicePointer(contract.invocation.segment_start.binding),
            ),
            argument(5, AotArgumentValue::U32(contract.invocation.row_count)),
            argument(
                6,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .trace_columns
                        .iter()
                        .map(|binding| binding.binding)
                        .collect(),
                ),
            ),
            argument(
                7,
                AotArgumentValue::DevicePointer(contract.invocation.lookup_words.binding),
            ),
            argument(
                8,
                AotArgumentValue::DevicePointerTable(
                    contract
                        .invocation
                        .partial_input_columns
                        .iter()
                        .map(|binding| binding.binding)
                        .collect(),
                ),
            ),
            argument(
                9,
                AotArgumentValue::U32(contract.invocation.partial_row_count),
            ),
            argument(10, AotArgumentValue::DevicePointer(Some(counts[0].binding))),
            argument(
                11,
                AotArgumentValue::U32(to_u32(requirements.address_count_words)?),
            ),
            argument(12, AotArgumentValue::DevicePointer(Some(counts[1].binding))),
            argument(
                13,
                AotArgumentValue::U32(to_u32(requirements.big_count_words)?),
            ),
            argument(14, AotArgumentValue::DevicePointer(Some(counts[2].binding))),
            argument(
                15,
                AotArgumentValue::U32(to_u32(requirements.small_count_words)?),
            ),
            argument(16, AotArgumentValue::DevicePointer(Some(counts[3].binding))),
            argument(
                17,
                AotArgumentValue::U32(to_u32(requirements.range_check_8_count_words)?),
            ),
        ],
    };
    validate_exact_bindings(&invocation, &contract.effect)
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpBinding)?;
    Ok(invocation)
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
