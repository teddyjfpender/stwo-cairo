//! Proof-wide semantic versions, effects and the seed wrapper invocation.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    WitnessInputSeedAbiAccess, WitnessInputSeedAbiArgument, WitnessInputSeedAbiArgumentKind,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, EffectAccess, ValueRange,
};

pub(super) fn bind_many(
    exact: Vec<bindings::ExactSemanticArenaBinding>,
    values: &adapter::SemanticValueMap,
    next_binding: &mut u32,
) -> Result<Vec<SemanticArenaBinding>, InvocationShapeError> {
    exact
        .into_iter()
        .map(|exact| {
            let binding = take_binding(next_binding)?;
            bind_semantic(exact, values, binding)
        })
        .collect()
}

pub(super) fn bind_semantic(
    exact: bindings::ExactSemanticArenaBinding,
    values: &adapter::SemanticValueMap,
    binding: EffectBindingId,
) -> Result<SemanticArenaBinding, InvocationShapeError> {
    Ok(SemanticArenaBinding {
        arena: exact.arena,
        value: exact.value,
        elements: exact.elements,
        binding,
        version: values.version(exact.value)?,
    })
}

pub(super) fn seed_effect(
    scalar: &SemanticArenaBinding,
    outputs: &[SemanticArenaBinding],
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses = Vec::with_capacity(outputs.len() + 1);
    accesses.push(EffectAccess::Read {
        source: bound(scalar),
    });
    accesses.extend(outputs.iter().map(|output| EffectAccess::Write {
        destination: bound(output),
    }));
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)
}

pub(super) fn compact_effect(
    sources: &[SemanticArenaBinding],
    descriptor_value: ValueVersion,
    descriptor_binding: EffectBindingId,
    descriptor_words: usize,
    outputs: &[SemanticArenaBinding],
    scratch: &[SemanticArenaBinding],
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses = Vec::with_capacity(sources.len() + outputs.len() + scratch.len() + 1);
    accesses.extend(sources.iter().map(|source| EffectAccess::Read {
        source: bound(source),
    }));
    accesses.push(EffectAccess::Read {
        source: BoundValueRange {
            binding: descriptor_binding,
            value: ValueRange {
                version: descriptor_value,
                elements: ElementRange::new(0, descriptor_words)
                    .ok_or(InvocationShapeError::InvalidStructuredAbi)?,
            },
        },
    });
    accesses.extend(
        outputs
            .iter()
            .chain(scratch)
            .map(|destination| EffectAccess::Write {
                destination: bound(destination),
            }),
    );
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)
}

fn bound(binding: &SemanticArenaBinding) -> BoundValueRange {
    BoundValueRange {
        binding: binding.binding,
        value: ValueRange {
            version: binding.version,
            elements: binding.elements,
        },
    }
}

pub(super) fn seed_invocation(
    contract: &WitnessInputSeedContract,
    scalar: &SemanticArenaBinding,
    outputs: &[SemanticArenaBinding],
) -> Result<AotInvocation, InvocationShapeError> {
    seed_invocation_using_abi(contract, scalar, outputs, contract.abi().arguments())
}

pub(super) fn seed_invocation_using_abi(
    contract: &WitnessInputSeedContract,
    scalar: &SemanticArenaBinding,
    outputs: &[SemanticArenaBinding],
    abi: &[WitnessInputSeedAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(index, argument)| argument.ordinal as usize != index)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let fixed = contract.fixed_words();
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        use {WitnessInputSeedAbiAccess as Access, WitnessInputSeedAbiArgumentKind as Kind};
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (0, "scalars_dev", Kind::DeviceConstPointerU32, Access::ReadScalarWords) => {
                AotArgumentValue::DevicePointer(Some(scalar.binding))
            }
            (1, "n_scalars", Kind::U32, Access::ScalarWordCount) => AotArgumentValue::U32(fixed[0]),
            (2, "n_real_rows", Kind::U32, Access::RealRowCount) => AotArgumentValue::U32(fixed[1]),
            (3, "consumer_rows", Kind::U32, Access::ConsumerRowCount) => {
                AotArgumentValue::U32(fixed[2])
            }
            (
                4,
                "consumer_cols_dev",
                Kind::DeviceMutPointerTableU32,
                Access::WriteConsumerColumns,
            ) => AotArgumentValue::DevicePointerTable(
                outputs.iter().map(|output| Some(output.binding)).collect(),
            ),
            (5, "include_enabler", Kind::U32, Access::IncludeEnabler) => {
                AotArgumentValue::U32(fixed[3])
            }
            (6, "include_iota", Kind::U32, Access::IncludeIota) => AotArgumentValue::U32(fixed[4]),
            (7, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    Ok(AotInvocation { arguments })
}

pub(super) fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
    fixed: Option<(ValueVersion, EffectBindingId)>,
) -> Result<(), InvocationShapeError> {
    if !effect.registered_fixed_source_reads().is_empty()
        || invocation.arguments.is_empty()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(index, argument)| argument.ordinal as usize != index)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
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
        let mut insert = |binding| {
            actual
                .insert(binding)
                .then_some(())
                .ok_or(InvocationShapeError::InvalidAdapterEffect)
        };
        match &argument.value {
            AotArgumentValue::U32(_)
            | AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding)?,
            AotArgumentValue::DevicePointerTable(entries) => {
                if entries.is_empty() || entries.iter().any(Option::is_none) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                for &binding in entries.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::DevicePointerTableValue(_)
            | AotArgumentValue::DeviceNestedPointerTableValue { .. }
            | AotArgumentValue::DeviceRecordPointerGraphValue { .. }
            | AotArgumentValue::DevicePointerRangeSetValue { .. }
            | AotArgumentValue::HostFixedU32(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(_) => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
            AotArgumentValue::DeviceFixedU32 { value, binding } => {
                if fixed != Some((*value, *binding)) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                insert(*binding)?;
            }
        }
    }
    if actual == expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

pub(super) fn take_binding(next: &mut u32) -> Result<EffectBindingId, InvocationShapeError> {
    let binding = EffectBindingId(*next);
    *next = next
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    Ok(binding)
}
