//! Reused-staging SSA, exact effect and ABI projection.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    WitnessCasmInputAbiAccess, WitnessCasmInputAbiArgument, WitnessCasmInputAbiArgumentKind,
    WitnessCasmInputColumnValue,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, EffectAccess, ValueRange,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StagingTransition {
    pub(super) previous: Option<ValueVersion>,
    pub(super) version: ValueVersion,
}

pub(super) fn exact_staging_lineage(
    values: &mut adapter::SemanticValueMap,
    catalog: ArenaCatalogValueId,
    lane_count: usize,
) -> Result<Vec<StagingTransition>, InvocationShapeError> {
    if lane_count == 0 {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let existing = values.versions_for(catalog).collect::<Vec<_>>();
    let lineage = if existing.is_empty() {
        values.extend_ordered([catalog])?;
        let first = values.version(catalog)?;
        let mut lineage = Vec::with_capacity(lane_count);
        lineage.push(StagingTransition {
            previous: None,
            version: first,
        });
        let mut previous = first;
        for _ in 1..lane_count {
            let (source, destination) = values.transition(catalog)?;
            if source != previous {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            lineage.push(StagingTransition {
                previous: Some(source),
                version: destination,
            });
            previous = destination;
        }
        lineage
    } else {
        if existing.len() != lane_count || values.version(catalog)? != *existing.last().unwrap() {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        existing
            .iter()
            .enumerate()
            .map(|(index, &version)| StagingTransition {
                previous: index.checked_sub(1).map(|previous| existing[previous]),
                version,
            })
            .collect()
    };
    validate_lineage(values, catalog, &lineage)?;
    Ok(lineage)
}

fn validate_lineage(
    values: &adapter::SemanticValueMap,
    catalog: ArenaCatalogValueId,
    lineage: &[StagingTransition],
) -> Result<(), InvocationShapeError> {
    let exact = values.versions_for(catalog).collect::<Vec<_>>();
    let versions = lineage
        .iter()
        .map(|transition| transition.version)
        .collect::<Vec<_>>();
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    if exact != versions
        || lineage
            .first()
            .and_then(|transition| transition.previous)
            .is_some()
        || !lineage
            .windows(2)
            .all(|pair| pair[1].previous == Some(pair[0].version))
        || !catalog_first.contains(&lineage[0].version)
        || lineage[1..]
            .iter()
            .any(|transition| !transitions.contains(&transition.version))
        || lineage
            .iter()
            .any(|transition| fixed.contains(&transition.version))
        || values.version(catalog)? != lineage.last().unwrap().version
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(())
}

pub(super) fn staging(
    exact: bindings::ExactArenaRange,
    transition: StagingTransition,
) -> Result<WitnessCasmStagingBinding, InvocationShapeError> {
    Ok(WitnessCasmStagingBinding {
        arena: exact.arena,
        value: exact.value,
        elements: exact.elements,
        binding: EffectBindingId(0),
        previous: transition.previous,
        version: transition.version,
    })
}

pub(super) fn outputs(
    exact: Vec<bindings::ExactOutput>,
    values: &adapter::SemanticValueMap,
) -> Result<Vec<WitnessCasmOutputBinding>, InvocationShapeError> {
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    exact
        .into_iter()
        .enumerate()
        .map(|(index, exact)| {
            let version = values.version(exact.value)?;
            if !catalog_first.contains(&version)
                || transitions.contains(&version)
                || fixed.contains(&version)
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            Ok(WitnessCasmOutputBinding {
                ordinal: exact.ordinal,
                value_kind: exact.value_kind,
                writer_use: exact.writer_use,
                arena: exact.arena,
                value: exact.value,
                elements: exact.elements,
                binding: EffectBindingId(
                    u32::try_from(index + 1).map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
                version,
            })
        })
        .collect()
}

pub(super) fn effect(
    staging: &WitnessCasmStagingBinding,
    outputs: &[WitnessCasmOutputBinding],
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses = Vec::with_capacity(outputs.len() + 1);
    accesses.push(EffectAccess::Read {
        source: bound(staging.binding, staging.version, staging.elements),
    });
    accesses.extend(outputs.iter().map(|output| EffectAccess::Write {
        destination: bound(output.binding, output.version, output.elements),
    }));
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)
}

pub(super) fn invocation(
    contract: &WitnessCasmInputContract,
    staging: &WitnessCasmStagingBinding,
    outputs: &[WitnessCasmOutputBinding],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(contract, staging, outputs, contract.abi().arguments())
}

pub(super) fn invocation_using_abi(
    contract: &WitnessCasmInputContract,
    staging: &WitnessCasmStagingBinding,
    outputs: &[WitnessCasmOutputBinding],
    abi: &[WitnessCasmInputAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(index, argument)| argument.ordinal as usize != index)
        || staging.elements.len() != contract.requirements().staging_words
        || outputs.len() != contract.effect_geometry().output_columns.len()
        || outputs
            .iter()
            .zip(&contract.effect_geometry().output_columns)
            .any(|(output, expected)| {
                output.ordinal != expected.column_ordinal
                    || output.value_kind != expected.value
                    || output.elements.len() != expected.written_words as usize
            })
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let fixed = contract.fixed_words();
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        use WitnessCasmInputAbiAccess as Access;
        use WitnessCasmInputAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (0, "rows_dev", Kind::DeviceConstPointerU32, Access::ReadRowMajorStates) => {
                AotArgumentValue::DevicePointer(Some(staging.binding))
            }
            (1, "n_real", Kind::U32, Access::RealRowCount) => AotArgumentValue::U32(fixed[1]),
            (2, "consumer_rows", Kind::U32, Access::ConsumerRowCount) => {
                AotArgumentValue::U32(fixed[2])
            }
            (3, "pc_dev", Kind::DeviceMutPointerU32, Access::WritePc) => {
                output_pointer(outputs, WitnessCasmInputColumnValue::StateWord(0))?
            }
            (4, "ap_dev", Kind::DeviceMutPointerU32, Access::WriteAp) => {
                output_pointer(outputs, WitnessCasmInputColumnValue::StateWord(1))?
            }
            (5, "fp_dev", Kind::DeviceMutPointerU32, Access::WriteFp) => {
                output_pointer(outputs, WitnessCasmInputColumnValue::StateWord(2))?
            }
            (6, "enabler_dev", Kind::DeviceMutPointerU32, Access::WriteEnabler) => {
                output_pointer(outputs, WitnessCasmInputColumnValue::Enabler)?
            }
            (7, "iota_dev", Kind::OptionalDeviceMutPointerU32, Access::WriteOptionalIota) => {
                optional_output_pointer(outputs, contract.requirements().include_iota)?
            }
            (8, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    if arguments.len() + 1 != abi.len()
        || arguments
            .iter()
            .enumerate()
            .any(|(index, argument)| argument.ordinal as usize != index)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(AotInvocation { arguments })
}

fn output_pointer(
    outputs: &[WitnessCasmOutputBinding],
    value_kind: WitnessCasmInputColumnValue,
) -> Result<AotArgumentValue, InvocationShapeError> {
    let mut matches = outputs
        .iter()
        .filter(|output| output.value_kind == value_kind);
    let output = matches
        .next()
        .ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(AotArgumentValue::DevicePointer(Some(output.binding)))
}

fn optional_output_pointer(
    outputs: &[WitnessCasmOutputBinding],
    include_iota: bool,
) -> Result<AotArgumentValue, InvocationShapeError> {
    let matches = outputs
        .iter()
        .filter(|output| output.value_kind == WitnessCasmInputColumnValue::Iota)
        .collect::<Vec<_>>();
    match (include_iota, matches.as_slice()) {
        (true, [output]) => Ok(AotArgumentValue::DevicePointer(Some(output.binding))),
        (false, []) => Ok(AotArgumentValue::DevicePointer(None)),
        _ => Err(InvocationShapeError::InvalidStructuredAbi),
    }
}

pub(super) fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
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
            AotArgumentValue::U32(_) | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => {
                if !actual.insert(*binding) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
            }
            AotArgumentValue::Usize(_)
            | AotArgumentValue::DevicePointerTable(_)
            | AotArgumentValue::DeviceFixedU32 { .. } => {
                return Err(InvocationShapeError::InvalidAdapterEffect)
            }
        }
    }
    if actual == expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidAdapterEffect)
    }
}

fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}
