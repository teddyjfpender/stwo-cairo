//! Canonical Base-prefix event view and executable emission.
//!
//! This is a borrowed view over `BaseProducerAuthority`, never a second
//! schedule. Prelude operations are admitted together. Each witness producer
//! and its optional generic feed form one transactional cursor unit.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::ExecutionTablesStage;

use super::super::{
    execution_tables, multiplicity_clear, multiplicity_feed, static_wrapper_invocation,
    static_wrapper_projection, witness_casm_input, witness_input_gather,
    witness_input_seed_compact,
};
use super::*;
use crate::compiled_proof::{
    AotInvocation, EffectContractId, ExecutionPrimitive, ModuleGlobalInitializer, OpNode,
    StaticCudaWrapperAuthority, StaticCudaWrapperId,
};

mod causal;
mod sequence;
mod static_resolver;
mod validation;

#[cfg(test)]
pub(in crate::program_image::lower_compiled) use causal::resolve_setup_counts_for_test;
pub(in crate::program_image::lower_compiled) use causal::ResolveSetupError;
use causal::{append_setup_event, resolve_setup_event};
pub(in super::super) use sequence::ordered_effects;
pub(super) use sequence::CausalWitnessSetup;
use sequence::{causal_sequence, BaseEvent};
pub(super) use static_resolver::resolve_static;
pub(super) use validation::validate_sealed_prefix;

#[cfg(test)]
pub(in crate::program_image::lower_compiled) fn statement_host_ingress_for_test(
    lowered: &witness_casm_input::LoweredWitnessCasmInput,
) -> Result<(ExecutionPrimitive, EffectContract), InvocationShapeError> {
    causal::statement_host_ingress(lowered)
}

pub(in crate::program_image::lower_compiled) type StaticWrapperResolver =
    for<'a> fn(
        StaticWrapperRequest<'a>,
        StaticCudaWrapperId,
        u32,
        &ProofArenaPlan,
        &adapter::SemanticValueMap,
    ) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError>;

/// Exact executable pair returned by a linked static build.
///
/// Most wrappers reuse an invocation fixed by semantic lowering. Composite
/// wrappers may instead learn execution-exact arguments from the linked build,
/// so the resolver must publish the invocation together with the wrapper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) struct ResolvedStaticExecution {
    pub(in crate::program_image::lower_compiled) wrapper: StaticCudaWrapperAuthority,
    pub(in crate::program_image::lower_compiled) invocation: AotInvocation,
}

#[derive(Clone, Copy)]
pub(in crate::program_image::lower_compiled) enum StaticWrapperRequest<'a> {
    ExecutionTable {
        lowered: &'a execution_tables::LoweredExecutionTables,
        stage: ExecutionTablesStage,
    },
    MultiplicityClear(&'a multiplicity_clear::LoweredMultiplicityClear),
    WitnessInputGather(&'a witness_input_gather::LoweredWitnessInputGather),
    WitnessInputSeed(&'a witness_input_seed_compact::LoweredWitnessInputSeed),
    WitnessInputCompact(&'a witness_input_seed_compact::LoweredWitnessInputCompact),
    WitnessCasmScatter(&'a witness_casm_input::LoweredWitnessCasmInput),
    MultiplicityFeed(&'a multiplicity_feed::LoweredMultiplicityFeed),
    NativeBlakeGDirect(&'a super::super::blake_g_direct_prefix::LoweredNativeBlakeGDirectContract),
    NativeEcOp(&'a super::super::ec_op_prefix::LoweredNativeEcOpContract),
}

impl<'a> StaticWrapperRequest<'a> {
    pub(in crate::program_image::lower_compiled) fn kind(self) -> StaticWrapperKind {
        match self {
            Self::ExecutionTable {
                stage: ExecutionTablesStage::Big,
                ..
            } => StaticWrapperKind::ExecutionTableBig,
            Self::ExecutionTable {
                stage: ExecutionTablesStage::Small,
                ..
            } => StaticWrapperKind::ExecutionTableSmall,
            Self::MultiplicityClear(_) => StaticWrapperKind::MultiplicityClear,
            Self::WitnessInputGather(_) => StaticWrapperKind::WitnessInputGather,
            Self::WitnessInputSeed(_) => StaticWrapperKind::WitnessInputSeed,
            Self::WitnessInputCompact(_) => StaticWrapperKind::WitnessInputCompact,
            Self::WitnessCasmScatter(_) => StaticWrapperKind::WitnessCasmScatter,
            Self::MultiplicityFeed(feed)
                if feed.owner == multiplicity_feed::MultiplicityFeedOwner::PublicMemorySeed =>
            {
                StaticWrapperKind::PublicMemorySeed
            }
            Self::MultiplicityFeed(_) => StaticWrapperKind::MultiplicityFeed,
            Self::NativeBlakeGDirect(_) => StaticWrapperKind::NativeBlakeGDirect,
            Self::NativeEcOp(_) => StaticWrapperKind::NativeEcOp,
        }
    }

    pub(in crate::program_image::lower_compiled) fn effect(
        self,
    ) -> Result<&'a EffectContract, InvocationShapeError> {
        match self {
            Self::ExecutionTable { lowered, stage } => lowered
                .stages
                .iter()
                .find(|candidate| candidate.stage == stage)
                .map(|stage| &stage.effect)
                .ok_or(InvocationShapeError::InvalidStructuredAbi),
            Self::MultiplicityClear(lowered) => Ok(&lowered.effect),
            Self::WitnessInputGather(lowered) => Ok(&lowered.effect),
            Self::WitnessInputSeed(lowered) => Ok(&lowered.effect),
            Self::WitnessInputCompact(lowered) => Ok(&lowered.effect),
            Self::WitnessCasmScatter(lowered) => Ok(&lowered.effect),
            Self::MultiplicityFeed(lowered) => Ok(&lowered.effect),
            Self::NativeBlakeGDirect(lowered) => Ok(&lowered.effect),
            Self::NativeEcOp(lowered) => Ok(&lowered.effect),
        }
    }

    pub(in crate::program_image::lower_compiled) fn invocation(
        self,
    ) -> Result<AotInvocation, InvocationShapeError> {
        match self {
            Self::ExecutionTable { lowered, stage } => lowered
                .stages
                .iter()
                .find(|candidate| candidate.stage == stage)
                .map(|stage| stage.invocation.clone())
                .ok_or(InvocationShapeError::InvalidStructuredAbi),
            Self::MultiplicityClear(lowered) => Ok(lowered.invocation.clone()),
            Self::WitnessInputGather(lowered) => Ok(lowered.invocation.clone()),
            Self::WitnessInputSeed(lowered) => Ok(lowered.invocation.clone()),
            // The compact invocation contains exact CUB temporary sizes learned
            // from its linked build and is therefore available only from the
            // resolved execution.
            Self::WitnessInputCompact(_) => {
                Err(InvocationShapeError::InvalidProductionBaseAuthority)
            }
            Self::WitnessCasmScatter(lowered) => Ok(lowered.invocation.clone()),
            Self::MultiplicityFeed(lowered) => Ok(lowered.invocation.clone()),
            Self::NativeBlakeGDirect(lowered) => static_wrapper_invocation::blake_g_direct(lowered),
            Self::NativeEcOp(lowered) => static_wrapper_invocation::ec_op(lowered),
        }
    }

    fn invocation_for_wrapper(
        self,
        wrapper: &StaticCudaWrapperAuthority,
    ) -> Result<AotInvocation, InvocationShapeError> {
        match self {
            Self::WitnessInputCompact(lowered) => {
                witness_input_seed_compact::compact_invocation_from_wrapper(lowered, wrapper)
            }
            _ => self.invocation(),
        }
    }

    fn missing_adapter(self) -> Option<MissingBaseAdapter> {
        match self.kind() {
            StaticWrapperKind::WitnessInputGather => {
                Some(MissingBaseAdapter::WitnessInputGatherStaticWrapper)
            }
            StaticWrapperKind::WitnessInputSeed => {
                Some(MissingBaseAdapter::WitnessInputSeedStaticWrapper)
            }
            StaticWrapperKind::WitnessInputCompact => {
                Some(MissingBaseAdapter::WitnessInputCompactStaticWrapper)
            }
            StaticWrapperKind::WitnessCasmScatter => {
                Some(MissingBaseAdapter::WitnessCasmScatterStaticWrapper)
            }
            StaticWrapperKind::MultiplicityFeed => {
                Some(MissingBaseAdapter::MultiplicityFeedStaticWrapper)
            }
            StaticWrapperKind::NativeBlakeGDirect => {
                Some(MissingBaseAdapter::NativeBlakeGDirectStaticWrapper)
            }
            StaticWrapperKind::NativeEcOp => Some(MissingBaseAdapter::NativeEcOpStaticWrapper),
            StaticWrapperKind::ExecutionTableBig
            | StaticWrapperKind::ExecutionTableSmall
            | StaticWrapperKind::MultiplicityClear
            | StaticWrapperKind::PublicMemorySeed => None,
        }
    }
}

#[derive(Clone, Default)]
struct Buffers {
    kernels: Vec<AotKernelAuthority>,
    static_wrappers: Vec<StaticCudaWrapperAuthority>,
    module_global_initializers: Vec<ModuleGlobalInitializer>,
    effects: BTreeMap<EffectContractId, EffectContract>,
    operations: Vec<OpNode>,
    kernel_by_build_authority: BTreeMap<[u8; 32], usize>,
}

enum ResolvedEvent<'a> {
    Recorded {
        recorded: &'a super::super::producer_prefix::LoweredRecordedWitnessProducer,
        fields: ResolvedRecordedBuildAuthority,
        effect: EffectContract,
        new_initializers: Vec<ModuleGlobalInitializer>,
    },
    Static {
        request: StaticWrapperRequest<'a>,
        execution: ResolvedStaticExecution,
    },
}

pub(super) fn emit_using(
    arena: &ProofArenaPlan,
    preprocessed_trace_variant: PreProcessedTraceVariant,
    manifest: [u8; 32],
    target_sm: u32,
    mut resolve_recorded: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<
        ResolvedRecordedBuildAuthority,
        ResolveRecordedAuthorityError,
    >,
    resolve_static: StaticWrapperResolver,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    split_sm(target_sm).map_err(|_| CompiledWitnessWriterPrefixError::InvalidTargetSm)?;
    if manifest == [0; 32] {
        return Err(
            CompiledWitnessWriterPrefixError::MissingRecordedAotAuthority(
                first_scheduled_producer(arena)?,
            ),
        );
    }
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let authority = BaseProducerAuthority::compile_replacement_into(
        arena,
        preprocessed_trace_variant,
        &mut values,
    )
    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let required_preproducer_versions = validate_witness_writer_transitions(&authority, &values)
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let causal_setup = CausalWitnessSetup::lower(arena, &mut values, &authority)
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let sequence = causal_sequence(&authority, &causal_setup)
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let monolithic = PartitionAuthority::monolithic();
    let mut buffers = Buffers::default();

    // Prelude publication is all-or-nothing, so `next_producer == 0` always
    // means the complete prelude is already present in a returned prefix.
    let mut resolved_prelude = Vec::with_capacity(sequence.prelude.len());
    for event in sequence.prelude.iter().copied() {
        let BaseEvent::Static(request) = event else {
            return Err(CompiledWitnessWriterPrefixError::Lowering);
        };
        let id = wrapper_id(
            buffers
                .static_wrappers
                .len()
                .checked_add(resolved_prelude.len())
                .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
        )?;
        let execution = resolve_static(request, id, target_sm, arena, &values)
            .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
            .ok_or(CompiledWitnessWriterPrefixError::Lowering)?;
        resolved_prelude.push(ResolvedEvent::Static { request, execution });
    }
    for event in resolved_prelude {
        append_event(event, target_sm, &monolithic, &mut buffers)?;
    }

    for (producer_index, group) in sequence.witnesses.iter().copied().enumerate() {
        let semantic = &authority.producers[producer_index];
        let position = semantic.position();
        let producer = semantic.producer();
        let resolved_setup = match group.setup {
            None => None,
            Some(setup) => {
                let id = wrapper_id(buffers.static_wrappers.len())?;
                match resolve_setup_event(setup, id, target_sm, arena, &values, resolve_static) {
                    Ok(event) => Some(event),
                    Err(ResolveSetupError::Missing(adapter)) => {
                        return missing_adapter(
                            adapter,
                            position,
                            producer,
                            manifest,
                            target_sm,
                            values,
                            causal_setup.clone(),
                            required_preproducer_versions.clone(),
                            buffers,
                            monolithic,
                            authority.clone(),
                            producer_index,
                        )
                    }
                    Err(ResolveSetupError::Invalid) => {
                        return Err(CompiledWitnessWriterPrefixError::Lowering)
                    }
                }
            }
        };
        let setup_static = usize::from(resolved_setup.is_some());
        let resolved_producer = match group.producer {
            BaseEvent::Recorded(recorded) => {
                let fields = resolve_recorded(&recorded.source).map_err(|error| match error {
                    ResolveRecordedAuthorityError::Missing => {
                        CompiledWitnessWriterPrefixError::MissingRecordedAotAuthority(producer)
                    }
                    ResolveRecordedAuthorityError::Invalid => {
                        CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
                    }
                })?;
                fields
                    .validate(&recorded.source, manifest, target_sm)
                    .map_err(|_| {
                        CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
                    })?;
                let (new_initializers, globals) = resolve_recorded_module_globals(
                    &recorded.source,
                    &fields,
                    &buffers.module_global_initializers,
                )
                .map_err(|_| {
                    CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
                })?;
                let effect = EffectContract::new(recorded.effect.accesses().to_vec(), globals)
                    .map_err(|_| {
                        CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
                    })?;
                ResolvedEvent::Recorded {
                    recorded,
                    fields,
                    effect,
                    new_initializers,
                }
            }
            BaseEvent::Static(request) => {
                let id = wrapper_id(
                    buffers
                        .static_wrappers
                        .len()
                        .checked_add(setup_static)
                        .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
                )?;
                let Some(execution) = resolve_static(request, id, target_sm, arena, &values)
                    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
                else {
                    return missing_adapter(
                        request
                            .missing_adapter()
                            .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
                        position,
                        producer,
                        manifest,
                        target_sm,
                        values,
                        causal_setup.clone(),
                        required_preproducer_versions.clone(),
                        buffers,
                        monolithic,
                        authority.clone(),
                        producer_index,
                    );
                };
                ResolvedEvent::Static { request, execution }
            }
        };
        let resolved_feed = match group.feed {
            None => None,
            Some(BaseEvent::Static(request)) => {
                let producer_static = if matches!(&resolved_producer, ResolvedEvent::Static { .. })
                {
                    1
                } else {
                    0
                };
                let id = wrapper_id(
                    buffers
                        .static_wrappers
                        .len()
                        .checked_add(setup_static)
                        .and_then(|count| count.checked_add(producer_static))
                        .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
                )?;
                let Some(execution) = resolve_static(request, id, target_sm, arena, &values)
                    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
                else {
                    return missing_adapter(
                        MissingBaseAdapter::MultiplicityFeedStaticWrapper,
                        position,
                        producer,
                        manifest,
                        target_sm,
                        values,
                        causal_setup.clone(),
                        required_preproducer_versions.clone(),
                        buffers,
                        monolithic,
                        authority.clone(),
                        producer_index,
                    );
                };
                Some(ResolvedEvent::Static { request, execution })
            }
            Some(BaseEvent::Recorded(_)) => return Err(CompiledWitnessWriterPrefixError::Lowering),
        };

        // Every authority in this schedule position is resolved before a
        // cloned buffer set is mutated. Host ingress, scatter, writer and feed
        // therefore publish as one cursor unit or not at all.
        let mut next_buffers = buffers.clone();
        if let Some(setup) = resolved_setup {
            append_setup_event(setup, target_sm, &monolithic, &mut next_buffers)?;
        }
        append_event(resolved_producer, target_sm, &monolithic, &mut next_buffers)?;
        if let Some(feed) = resolved_feed {
            append_event(feed, target_sm, &monolithic, &mut next_buffers)?;
        }
        buffers = next_buffers;
    }
    drop(sequence);
    let next_producer = authority.producers.len();
    finish_prefix(
        manifest,
        target_sm,
        values,
        causal_setup,
        required_preproducer_versions,
        buffers,
        monolithic,
        authority,
        next_producer,
    )
}

fn append_event(
    event: ResolvedEvent<'_>,
    target_sm: u32,
    monolithic: &PartitionAuthority,
    buffers: &mut Buffers,
) -> Result<(), CompiledWitnessWriterPrefixError> {
    match event {
        ResolvedEvent::Recorded {
            recorded,
            fields,
            effect,
            new_initializers,
        } => {
            let producer = recorded.producer;
            for initializer in &new_initializers {
                if initializer.id().0 as usize != buffers.module_global_initializers.len()
                    || !initializer.has_valid_identity().map_err(|_| {
                        CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
                    })?
                    || buffers.module_global_initializers.iter().any(|existing| {
                        existing.module() == initializer.module()
                            && existing.symbol() == initializer.symbol()
                    })
                {
                    return Err(
                        CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer),
                    );
                }
                buffers.module_global_initializers.push(initializer.clone());
            }
            insert_effect(&mut buffers.effects, effect.clone()).map_err(|_| {
                CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
            })?;
            let kernel = install_recorded_kernel(
                &mut buffers.kernels,
                &mut buffers.kernel_by_build_authority,
                &recorded.source,
                &fields,
                &recorded.invocation,
                effect.id(),
            )
            .map_err(|_| CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer))?;
            push_operation(
                ExecutionPrimitive::AotKernel {
                    kernel,
                    launch: recorded.source.launch,
                },
                Some(recorded.invocation.clone()),
                effect.id(),
                monolithic,
                &mut buffers.operations,
            )
        }
        ResolvedEvent::Static { request, execution } => {
            let effect = request
                .effect()
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            let ResolvedStaticExecution {
                wrapper,
                invocation,
            } = execution;
            let expected_id = wrapper_id(buffers.static_wrappers.len())?;
            if wrapper.id() != expected_id
                || wrapper.consumer_target_sm() != target_sm
                || wrapper.accepted_invocation()
                    != invocation
                        .contract_id()
                        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
                || wrapper.accepted_effect() != effect.id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
            {
                return Err(CompiledWitnessWriterPrefixError::Lowering);
            }
            insert_effect(&mut buffers.effects, effect.clone())
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            buffers.static_wrappers.push(wrapper);
            push_operation(
                ExecutionPrimitive::StaticCudaWrapper {
                    wrapper: expected_id,
                },
                Some(invocation),
                effect.id(),
                monolithic,
                &mut buffers.operations,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn missing_adapter(
    adapter: MissingBaseAdapter,
    position: super::super::producer_prefix::ProducerSchedulePosition,
    producer: WitnessProducer,
    manifest: [u8; 32],
    target_sm: u32,
    values: adapter::SemanticValueMap,
    causal_setup: CausalWitnessSetup,
    required_preproducer_versions: BTreeSet<ValueVersion>,
    buffers: Buffers,
    monolithic: PartitionAuthority,
    authority: BaseProducerAuthority,
    next_producer: usize,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    Err(CompiledWitnessWriterPrefixError::MissingTypedAdapter {
        missing: MissingBaseAdapterAt {
            adapter,
            producer,
            schedule_ordinal: position.ordinal,
        },
        prefix: Box::new(finish_prefix(
            manifest,
            target_sm,
            values,
            causal_setup,
            required_preproducer_versions,
            buffers,
            monolithic,
            authority,
            next_producer,
        )?),
    })
}

fn finish_prefix(
    execution_manifest_identity: [u8; 32],
    target_sm: u32,
    values: adapter::SemanticValueMap,
    causal_setup: CausalWitnessSetup,
    required_preproducer_versions: BTreeSet<ValueVersion>,
    buffers: Buffers,
    monolithic: PartitionAuthority,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    let Buffers {
        kernels,
        static_wrappers,
        module_global_initializers,
        effects,
        operations,
        kernel_by_build_authority,
    } = buffers;
    let effects = effects.into_values().collect::<Vec<_>>();
    let partitions = (!operations.is_empty())
        .then_some(monolithic.clone())
        .into_iter()
        .collect::<Vec<_>>();
    validate_sealed_prefix(
        &base_authority,
        &causal_setup,
        next_producer,
        target_sm,
        &kernel_by_build_authority,
        &kernels,
        &static_wrappers,
        &module_global_initializers,
        &effects,
        &partitions,
        &monolithic,
        &operations,
    )
    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    let causal_external_roots = (next_producer == base_authority.producers.len())
        .then(|| validate_causal_value_closure(&values, &effects, &operations))
        .transpose()
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    Ok(CompiledWitnessWriterPrefix {
        execution_manifest_identity,
        target_sm,
        values,
        base_authority,
        causal_setup,
        next_producer,
        causal_external_roots,
        required_preproducer_versions,
        kernel_by_build_authority,
        kernels,
        static_wrappers,
        module_global_initializers,
        effects,
        partitions,
        operations,
    })
}
