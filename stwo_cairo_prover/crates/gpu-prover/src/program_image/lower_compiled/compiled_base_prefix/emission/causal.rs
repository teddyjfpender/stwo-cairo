use stwo_cairo_prover::witness::proof_shape::TracePartId;

#[cfg(test)]
use super::sequence::causal_sequence;
use super::sequence::CausalSetupEvent;
use super::*;
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, ExecutionPrimitive, StatementHostEncoding, StatementHostPart,
    StatementHostSource, StatementHostSourceKind, ValueRange,
};

pub(super) enum ResolvedSetupEvent<'a> {
    Static(ResolvedEvent<'a>),
    Casm {
        lowered: &'a witness_casm_input::LoweredWitnessCasmInput,
        scatter: ResolvedEvent<'a>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) enum ResolveSetupError {
    Missing(MissingBaseAdapter),
    Invalid,
}

/// Resolve a setup wrapper without mutating any emitted authority. The caller
/// publishes this only after the same producer's writer and feed have also
/// resolved, so a missing middle authority cannot strand a partial group.
pub(super) fn resolve_setup_event<'a>(
    setup: CausalSetupEvent<'a>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
    resolve_static: StaticWrapperResolver,
) -> Result<ResolvedSetupEvent<'a>, ResolveSetupError> {
    let (request, casm) = match setup {
        CausalSetupEvent::Static(request) => (request, None),
        CausalSetupEvent::Casm(lowered) => (
            StaticWrapperRequest::WitnessCasmScatter(lowered),
            Some(lowered),
        ),
    };
    let execution = match resolve_static(request, id, target_sm, arena, values)
        .map_err(|_| ResolveSetupError::Invalid)?
    {
        Some(execution) => execution,
        None => {
            return Err(ResolveSetupError::Missing(
                request
                    .missing_adapter()
                    .ok_or(ResolveSetupError::Invalid)?,
            ));
        }
    };
    let resolved = ResolvedEvent::Static { request, execution };
    Ok(match casm {
        Some(lowered) => ResolvedSetupEvent::Casm {
            lowered,
            scatter: resolved,
        },
        None => ResolvedSetupEvent::Static(resolved),
    })
}

pub(super) fn statement_host_ingress(
    lowered: &witness_casm_input::LoweredWitnessCasmInput,
) -> Result<(ExecutionPrimitive, EffectContract), InvocationShapeError> {
    let requirements = lowered.contract.requirements();
    if lowered.staging.elements.start != 0
        || lowered.staging.elements.len() != requirements.staging_words
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let destination = BoundValueRange {
        binding: lowered.staging.binding,
        value: ValueRange {
            version: lowered.staging.version,
            elements: lowered.staging.elements,
        },
    };
    let effect = EffectContract::new(vec![EffectAccess::Write { destination }], Vec::new())
        .map_err(|_| InvocationShapeError::InvalidScheduledProducerBinding)?;
    let part = match lowered.part {
        TracePartId::Main => StatementHostPart::Main,
        TracePartId::MemoryBig(ordinal) => StatementHostPart::MemoryBig(ordinal),
        TracePartId::MemorySmall => StatementHostPart::MemorySmall,
    };
    let source = StatementHostSource {
        kind: StatementHostSourceKind::WitnessCasm,
        producer_ordinal: lowered.position.ordinal,
        component: Box::<str>::from(lowered.component),
        part,
        encoding: StatementHostEncoding::RowMajorU32,
        words: requirements.staging_words,
        real_rows: requirements.n_real_rows,
        consumer_rows: requirements.consumer_rows,
        include_iota: requirements.include_iota,
        casm_contract_identity: lowered.contract.identity(),
    };
    if !source.has_valid_shape() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let predecessor = lowered.staging.previous.map(|version| ValueRange {
        version,
        elements: lowered.staging.elements,
    });
    Ok((
        ExecutionPrimitive::StatementHostIngress {
            source,
            predecessor,
        },
        effect,
    ))
}

#[cfg(test)]
pub(in crate::program_image::lower_compiled) fn resolve_setup_counts_for_test(
    arena: &ProofArenaPlan,
    prefix: &CompiledWitnessWriterPrefix,
    resolve_static: StaticWrapperResolver,
) -> Result<(usize, usize), ResolveSetupError> {
    prefix
        .causal_setup
        .validate(arena, &prefix.values, &prefix.base_authority)
        .map_err(|_| ResolveSetupError::Invalid)?;
    let sequence = causal_sequence(&prefix.base_authority, &prefix.causal_setup)
        .map_err(|_| ResolveSetupError::Invalid)?;
    let mut next_wrapper = 0usize;
    let mut ordinary = 0usize;
    let mut casm = 0usize;
    for (ordinal, group) in sequence.witnesses.into_iter().enumerate() {
        let Some(setup) = group.setup else {
            continue;
        };
        let id = wrapper_id(next_wrapper).map_err(|_| ResolveSetupError::Invalid)?;
        match resolve_setup_event(
            setup,
            id,
            prefix.target_sm,
            arena,
            &prefix.values,
            resolve_static,
        )? {
            ResolvedSetupEvent::Static(ResolvedEvent::Static { .. }) => ordinary += 1,
            ResolvedSetupEvent::Casm {
                lowered,
                scatter: ResolvedEvent::Static { request, .. },
            } if request.kind() == StaticWrapperKind::WitnessCasmScatter
                && lowered.position.ordinal as usize == ordinal =>
            {
                casm += 1;
            }
            _ => return Err(ResolveSetupError::Invalid),
        }
        next_wrapper += 1;
    }
    Ok((ordinary, casm))
}

pub(super) fn append_setup_event(
    event: ResolvedSetupEvent<'_>,
    target_sm: u32,
    monolithic: &PartitionAuthority,
    buffers: &mut Buffers,
) -> Result<(), CompiledWitnessWriterPrefixError> {
    match event {
        ResolvedSetupEvent::Static(event) => append_event(event, target_sm, monolithic, buffers),
        ResolvedSetupEvent::Casm { lowered, scatter } => {
            let (primitive, effect) = statement_host_ingress(lowered)
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            insert_effect(&mut buffers.effects, effect.clone())
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            push_operation(
                primitive,
                None,
                effect.id(),
                monolithic,
                &mut buffers.operations,
            )?;
            append_event(scatter, target_sm, monolithic, buffers)
        }
    }
}
