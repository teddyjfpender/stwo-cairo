//! Effect and real-wrapper invocation projection for memory Base traces.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    MemoryBaseTraceAbi, MemoryBaseTraceAbiAccess, MemoryBaseTraceAbiArgumentKind,
    MemoryBaseTraceStepContract,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AtomicOperation, BoundValueRange, EffectAccess,
    InPlaceAliasId, InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
};

pub(super) fn atomic_alias() -> InPlaceAliasAuthority {
    InPlaceAliasAuthority {
        id: InPlaceAliasId(0),
        requirement: InPlaceAliasRequirement::Required,
        discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
    }
}

pub(super) fn compile(
    contract_ordinal: usize,
    contract: &MemoryBaseTraceStepContract,
    part: TracePartId,
    reads: Vec<MemoryBaseTraceValueBinding>,
    writes: Vec<MemoryBaseTraceValueBinding>,
    atomic: Option<MemoryBaseTraceTransition>,
) -> Result<LoweredMemoryBaseTraceStep, InvocationShapeError> {
    validate_geometry(contract, &reads, &writes, atomic.as_ref())?;
    let effect = effect(&reads, &writes, atomic.as_ref())?;
    let invocation = invocation(contract, &reads, &writes, atomic.as_ref())?;
    validate_exact_bindings(&invocation, &effect)?;
    Ok(LoweredMemoryBaseTraceStep {
        contract_ordinal: u32::try_from(contract_ordinal)
            .map_err(|_| InvocationShapeError::SizeOverflow)?,
        kind: contract.kind(),
        part,
        reads,
        writes,
        atomic,
        invocation,
        effect,
    })
}

pub(super) fn rebuild(
    supplied: &LoweredMemoryBaseTraceStep,
    contract: &MemoryBaseTraceStepContract,
) -> Result<(AotInvocation, EffectContract), InvocationShapeError> {
    validate_geometry(
        contract,
        &supplied.reads,
        &supplied.writes,
        supplied.atomic.as_ref(),
    )?;
    let effect = effect(&supplied.reads, &supplied.writes, supplied.atomic.as_ref())?;
    let invocation = invocation(
        contract,
        &supplied.reads,
        &supplied.writes,
        supplied.atomic.as_ref(),
    )?;
    validate_exact_bindings(&invocation, &effect)?;
    Ok((invocation, effect))
}

fn effect(
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses =
        Vec::with_capacity(reads.len() + writes.len() + usize::from(atomic.is_some()));
    accesses.extend(reads.iter().map(|read| EffectAccess::Read {
        source: bound(read.binding, read.version, read.elements),
    }));
    accesses.extend(writes.iter().map(|write| EffectAccess::Write {
        destination: bound(write.binding, write.version, write.elements),
    }));
    if let Some(atomic) = atomic {
        accesses.push(EffectAccess::Atomic {
            source: bound(atomic.binding, atomic.source, atomic.elements),
            destination: bound(atomic.binding, atomic.destination, atomic.elements),
            operation: AtomicOperation::AddU32,
            in_place: atomic.alias,
        });
    }
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceBinding)
}

fn invocation(
    contract: &MemoryBaseTraceStepContract,
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<AotInvocation, InvocationShapeError> {
    let arguments = match contract.abi() {
        MemoryBaseTraceAbi::AddressSlicedV2 => address_invocation(contract, reads, writes, atomic)?,
        MemoryBaseTraceAbi::ValueSlicedV2 => value_invocation(contract, reads, writes, atomic)?,
        MemoryBaseTraceAbi::Rc99V1 => rc99_invocation(contract, reads, writes, atomic)?,
        MemoryBaseTraceAbi::AddressV1 | MemoryBaseTraceAbi::ValueV1 => {
            return Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority)
        }
    };
    Ok(AotInvocation { arguments })
}

fn address_invocation(
    contract: &MemoryBaseTraceStepContract,
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<Vec<AotArgumentBinding>, InvocationShapeError> {
    if reads.len() != 2 || writes.is_empty() || atomic.is_some() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    invocation_using(
        contract,
        [
            AotArgumentValue::DevicePointer(Some(reads[0].binding)),
            AotArgumentValue::U32(contract.source_words()),
            AotArgumentValue::DevicePointer(Some(reads[1].binding)),
            AotArgumentValue::U32(contract.multiplicity_words()),
            AotArgumentValue::U32(contract.row_count()),
            AotArgumentValue::DevicePointerTable(
                writes.iter().map(|value| Some(value.binding)).collect(),
            ),
        ],
    )
}

fn value_invocation(
    contract: &MemoryBaseTraceStepContract,
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<Vec<AotArgumentBinding>, InvocationShapeError> {
    let limbs = contract.limb_or_pair_count() as usize;
    let counts = read_binding(
        contract,
        reads,
        MemoryBaseTraceEffectRole::ValueMultiplicity,
        0,
    )?
    .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    if writes.len() != limbs + 1 || atomic.is_some() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    let sources = (0..limbs)
        .map(|ordinal| {
            read_binding(
                contract,
                reads,
                MemoryBaseTraceEffectRole::ValueSource,
                ordinal as u32,
            )
            .map(|binding| binding.map(|binding| binding.binding))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exact_source_presence = if contract.source_words() == 0 {
        sources.iter().all(Option::is_none)
    } else {
        sources.iter().all(Option::is_some)
    };
    if !exact_source_presence {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    invocation_using(
        contract,
        [
            AotArgumentValue::DevicePointerTable(sources),
            AotArgumentValue::U32(contract.limb_or_pair_count()),
            AotArgumentValue::U32(contract.source_words()),
            AotArgumentValue::DevicePointer(Some(counts.binding)),
            AotArgumentValue::U32(contract.multiplicity_words()),
            AotArgumentValue::U32(contract.row_count()),
            AotArgumentValue::DevicePointerTable(
                writes.iter().map(|value| Some(value.binding)).collect(),
            ),
        ],
    )
}

fn read_binding<'a>(
    contract: &MemoryBaseTraceStepContract,
    reads: &'a [MemoryBaseTraceValueBinding],
    role: MemoryBaseTraceEffectRole,
    ordinal: u32,
) -> Result<Option<&'a MemoryBaseTraceValueBinding>, InvocationShapeError> {
    let mut matches = contract
        .reads()
        .iter()
        .zip(reads)
        .filter(|(effect, _)| effect.role == role && effect.ordinal == ordinal);
    let value = matches.next().map(|(_, value)| value);
    if matches.next().is_some() {
        Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority)
    } else {
        Ok(value)
    }
}

fn rc99_invocation(
    contract: &MemoryBaseTraceStepContract,
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<Vec<AotArgumentBinding>, InvocationShapeError> {
    let limbs = contract.limb_or_pair_count() as usize * 2;
    let atomic = atomic.ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    if reads.len() != limbs + 1 || !writes.is_empty() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    invocation_using(
        contract,
        [
            AotArgumentValue::DevicePointerTable(
                reads[..limbs]
                    .iter()
                    .map(|value| Some(value.binding))
                    .collect(),
            ),
            AotArgumentValue::U32(contract.limb_or_pair_count()),
            AotArgumentValue::U32(contract.row_count()),
            AotArgumentValue::DevicePointer(Some(reads[limbs].binding)),
            AotArgumentValue::U32(1 << 18),
            AotArgumentValue::DevicePointer(Some(atomic.binding)),
        ],
    )
}

fn invocation_using<const N: usize>(
    contract: &MemoryBaseTraceStepContract,
    values: [AotArgumentValue; N],
) -> Result<Vec<AotArgumentBinding>, InvocationShapeError> {
    let abi = contract.abi().arguments();
    if abi.len() != N + 1
        || abi
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| argument.ordinal as usize != ordinal)
        || !matches!(
            abi.last().map(|argument| (argument.kind, argument.access)),
            Some((
                MemoryBaseTraceAbiArgumentKind::CudaStream,
                MemoryBaseTraceAbiAccess::OrderedExecutionStream
            ))
        )
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority);
    }
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(ordinal, value)| AotArgumentBinding {
            ordinal: ordinal as u8,
            value,
        })
        .collect())
}

fn validate_geometry(
    contract: &MemoryBaseTraceStepContract,
    reads: &[MemoryBaseTraceValueBinding],
    writes: &[MemoryBaseTraceValueBinding],
    atomic: Option<&MemoryBaseTraceTransition>,
) -> Result<(), InvocationShapeError> {
    if reads.len() != contract.reads().len()
        || writes.len() != contract.writes().len()
        || atomic.is_some() != contract.atomic().is_some()
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    for (binding, geometry) in reads.iter().zip(contract.reads()) {
        validate_binding(binding.binding, binding.elements, geometry, None)?;
    }
    for (binding, geometry) in writes.iter().zip(contract.writes()) {
        validate_binding(binding.binding, binding.elements, geometry, None)?;
    }
    if let (Some(binding), Some(geometry)) = (atomic, contract.atomic()) {
        validate_binding(
            binding.binding,
            binding.elements,
            &geometry,
            Some(binding.alias),
        )?;
        if binding.source == binding.destination {
            return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
        }
    }
    let expected = reads
        .iter()
        .map(|value| value.binding)
        .chain(writes.iter().map(|value| value.binding))
        .chain(atomic.map(|value| value.binding))
        .enumerate()
        .all(|(ordinal, binding)| binding.0 as usize == ordinal);
    expected
        .then_some(())
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)
}

fn validate_binding(
    binding: EffectBindingId,
    elements: ElementRange,
    geometry: &stwo_backend_cuda::MemoryBaseTraceEffectAccess,
    alias: Option<InPlaceAliasAuthority>,
) -> Result<(), InvocationShapeError> {
    if elements.start != geometry.start_words
        || elements.len() != geometry.len_words
        || (geometry.role == MemoryBaseTraceEffectRole::Rc99Counts)
            != alias.is_some_and(|alias| alias == atomic_alias())
        || binding.0 == u32::MAX
    {
        Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    } else {
        Ok(())
    }
}

fn validate_exact_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    if !effect.registered_fixed_source_reads().is_empty() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
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
            AotArgumentValue::U32(_) | AotArgumentValue::Usize(_) => {}
            AotArgumentValue::DevicePointer(Some(binding)) => {
                actual.insert(*binding);
            }
            AotArgumentValue::DevicePointerTable(bindings) if !bindings.is_empty() => {
                actual.extend(bindings.iter().flatten().copied());
            }
            AotArgumentValue::DevicePointer(None)
            | AotArgumentValue::DevicePointerTable(_)
            | AotArgumentValue::DevicePointerTableValue(_)
            | AotArgumentValue::DeviceNestedPointerTableValue { .. }
            | AotArgumentValue::HostFixedU32(_)
            | AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(_)
            | AotArgumentValue::DeviceMixedFixedSourcePointerTable(_)
            | AotArgumentValue::DeviceFixedU32 { .. } => {
                return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
            }
        }
    }
    if actual == expected {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    }
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
