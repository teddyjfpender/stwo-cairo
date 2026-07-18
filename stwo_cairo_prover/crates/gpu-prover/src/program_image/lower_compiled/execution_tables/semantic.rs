//! Exact effect and invocation projection for one execution-table split.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    ExecutionTablesAbiAccess, ExecutionTablesAbiArgument, ExecutionTablesAbiArgumentKind,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, BoundValueRange, EffectAccess, ValueRange,
};

pub(super) fn lower(
    stage: &ExecutionTablesStageContract,
    source: &ExecutionTableHostIngress,
    outputs: Vec<&BaseCatalogValue>,
    values: &adapter::SemanticValueMap,
) -> Result<LoweredExecutionTableStage, InvocationShapeError> {
    let geometry = stage.effect_geometry();
    if geometry.stage != stage.stage()
        || source.copied_words != geometry.source_read_words
        || source.elements.map(ElementRange::len) != source.version.map(|_| source.copied_words)
        || outputs.len() != geometry.output_writes.len()
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let source_binding = source.version.map(|_| EffectBindingId(0));
    let first_output = u32::from(source_binding.is_some());
    let outputs = outputs
        .into_iter()
        .zip(&geometry.output_writes)
        .enumerate()
        .map(|(index, (value, write))| {
            if write.column_ordinal as usize != index
                || write.write_start_word != 0
                || write.written_words as usize != value.words
            {
                return Err(InvocationShapeError::InvalidStructuredAbi);
            }
            Ok(ExecutionTableOutputBinding {
                column_ordinal: write.column_ordinal,
                arena: exact_arena_for_semantic(value)?,
                value: value.id,
                elements: ElementRange::new(0, value.words)
                    .ok_or(InvocationShapeError::InvalidStructuredAbi)?,
                binding: EffectBindingId(
                    first_output
                        .checked_add(
                            u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                        )
                        .ok_or(InvocationShapeError::SizeOverflow)?,
                ),
                version: values.version(value.id)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let effect = effect(source, source_binding, &outputs)?;
    let invocation = invocation(stage, source_binding, &outputs)?;
    validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredExecutionTableStage {
        stage: stage.stage(),
        source: source.version,
        source_binding,
        outputs,
        invocation,
        effect,
    })
}

fn exact_arena_for_semantic(
    value: &BaseCatalogValue,
) -> Result<ArenaBinding, InvocationShapeError> {
    Ok(ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    })
}

fn effect(
    source: &ExecutionTableHostIngress,
    source_binding: Option<EffectBindingId>,
    outputs: &[ExecutionTableOutputBinding],
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses = Vec::with_capacity(outputs.len() + usize::from(source_binding.is_some()));
    match (source.version, source.elements, source_binding) {
        (Some(version), Some(elements), Some(binding)) => accesses.push(EffectAccess::Read {
            source: bound(binding, version, elements),
        }),
        (None, None, None) => {}
        _ => return Err(InvocationShapeError::InvalidAdapterEffect),
    }
    accesses.extend(outputs.iter().map(|output| EffectAccess::Write {
        destination: bound(output.binding, output.version, output.elements),
    }));
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)
}

fn invocation(
    stage: &ExecutionTablesStageContract,
    source_binding: Option<EffectBindingId>,
    outputs: &[ExecutionTableOutputBinding],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(stage, source_binding, outputs, stage.abi().arguments())
}

fn invocation_using_abi(
    stage: &ExecutionTablesStageContract,
    source_binding: Option<EffectBindingId>,
    outputs: &[ExecutionTableOutputBinding],
    abi: &[ExecutionTablesAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    if abi != stage.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let geometry = stage.effect_geometry();
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        use ExecutionTablesAbiAccess as Access;
        use ExecutionTablesAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (0, "values", Kind::OptionalDeviceConstPointerU32, Access::ReadValuesWhenNonEmpty) => {
                AotArgumentValue::DevicePointer(source_binding)
            }
            (1, "n_values", Kind::U32, Access::RealRowCount) => {
                AotArgumentValue::U32(geometry.real_rows)
            }
            (2, "column_length", Kind::U32, Access::ColumnRowCount) => {
                AotArgumentValue::U32(geometry.column_rows)
            }
            (
                3,
                "limb_cols_host",
                Kind::HostConstPointerTableDeviceMutU32,
                Access::ReadHostPointersWriteDeviceColumns,
            ) => AotArgumentValue::DevicePointerTable(
                outputs.iter().map(|output| Some(output.binding)).collect(),
            ),
            (4, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    if arguments.len() + 1 != abi.len() {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    Ok(AotInvocation { arguments })
}

fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    if !effect.registered_fixed_source_reads().is_empty() {
        return Err(InvocationShapeError::InvalidAdapterEffect);
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
            if actual.insert(binding) {
                Ok(())
            } else {
                Err(InvocationShapeError::InvalidAdapterEffect)
            }
        };
        match &argument.value {
            AotArgumentValue::U32(_) | AotArgumentValue::DevicePointer(None) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => insert(*binding)?,
            AotArgumentValue::DevicePointerTable(bindings) => {
                if bindings.is_empty() || bindings.iter().any(Option::is_none) {
                    return Err(InvocationShapeError::InvalidAdapterEffect);
                }
                for &binding in bindings.iter().flatten() {
                    insert(binding)?;
                }
            }
            AotArgumentValue::Usize(_)
            | AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_)
            | AotArgumentValue::DeviceMixedFixedSourcePointerTable(_)
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

#[cfg(test)]
pub(super) fn invocation_for_test(
    stage: &ExecutionTablesStageContract,
    source_binding: Option<EffectBindingId>,
    outputs: &[ExecutionTableOutputBinding],
    abi: &[ExecutionTablesAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    invocation_using_abi(stage, source_binding, outputs, abi)
}
