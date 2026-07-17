//! Canonical Base-prefix event view and executable emission.
//!
//! This is a borrowed view over `BaseProducerAuthority`, never a second
//! schedule. Prelude operations are admitted together. Each witness producer
//! and its optional generic feed form one transactional cursor unit.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::ExecutionTablesStage;

use super::super::{
    execution_tables, multiplicity_clear, multiplicity_feed, static_wrapper_invocation,
    static_wrapper_projection,
};
use super::*;
use crate::compiled_proof::{
    AotInvocation, EffectContractId, ExecutionPrimitive, ModuleGlobalInitializer, OpNode,
    StaticCudaWrapperAuthority, StaticCudaWrapperId,
};

mod sequence;

pub(in super::super) use sequence::ordered_effects;
use sequence::{sequence, BaseEvent};

pub(in crate::program_image::lower_compiled) type StaticWrapperResolver =
    for<'a> fn(
        StaticWrapperRequest<'a>,
        StaticCudaWrapperId,
        u32,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError>;

#[derive(Clone, Copy)]
pub(in crate::program_image::lower_compiled) enum StaticWrapperRequest<'a> {
    ExecutionTable {
        lowered: &'a execution_tables::LoweredExecutionTables,
        stage: ExecutionTablesStage,
    },
    MultiplicityClear(&'a multiplicity_clear::LoweredMultiplicityClear),
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
            Self::MultiplicityFeed(lowered) => Ok(&lowered.effect),
            Self::NativeBlakeGDirect(lowered) => Ok(&lowered.effect),
            Self::NativeEcOp(lowered) => Ok(&lowered.effect),
        }
    }

    fn invocation(self) -> Result<AotInvocation, InvocationShapeError> {
        match self {
            Self::ExecutionTable { lowered, stage } => lowered
                .stages
                .iter()
                .find(|candidate| candidate.stage == stage)
                .map(|stage| stage.invocation.clone())
                .ok_or(InvocationShapeError::InvalidStructuredAbi),
            Self::MultiplicityClear(lowered) => Ok(lowered.invocation.clone()),
            Self::MultiplicityFeed(lowered) => Ok(lowered.invocation.clone()),
            Self::NativeBlakeGDirect(lowered) => static_wrapper_invocation::blake_g_direct(lowered),
            Self::NativeEcOp(lowered) => static_wrapper_invocation::ec_op(lowered),
        }
    }

    fn missing_adapter(self) -> Option<MissingBaseAdapter> {
        match self.kind() {
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

pub(super) fn resolve_static(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    let wrapper = match request {
        StaticWrapperRequest::ExecutionTable { lowered, stage } => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            execution_tables::project_static_wrapper(id, &linked, lowered, stage)?.wrapper
        }
        StaticWrapperRequest::MultiplicityClear(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            multiplicity_clear::project_static_wrapper(id, &linked, lowered)?.wrapper
        }
        StaticWrapperRequest::MultiplicityFeed(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            multiplicity_feed::project_static_wrapper(id, &linked, lowered)?.wrapper
        }
        StaticWrapperRequest::NativeBlakeGDirect(lowered) => {
            let Some(linked) =
                super::super::blake_g_direct_execution_authority::
                    NativeBlakeGDirectLinkedModuleAuthority::bind_linked(
                        &lowered.authority,
                        target_sm,
                    )?
            else {
                return Ok(None);
            };
            static_wrapper_projection::blake_g_direct(id, &linked, lowered)?
        }
        StaticWrapperRequest::NativeEcOp(lowered) => {
            let Some(linked) = super::super::ec_op_execution_authority::
                NativeEcOpLinkedModuleAuthority::bind_linked(&lowered.authority)?
            else {
                return Ok(None);
            };
            linked.validate_active_sm(target_sm)?;
            static_wrapper_projection::ec_op(id, &linked, lowered)?
        }
    };
    Ok(Some(wrapper))
}

#[derive(Default)]
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
        wrapper: StaticCudaWrapperAuthority,
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
    let sequence = sequence(&authority).map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
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
        let wrapper = resolve_static(request, id, target_sm)
            .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
            .ok_or(CompiledWitnessWriterPrefixError::Lowering)?;
        resolved_prelude.push(ResolvedEvent::Static { request, wrapper });
    }
    for event in resolved_prelude {
        append_event(event, target_sm, &monolithic, &mut buffers)?;
    }

    for (producer_index, group) in sequence.witnesses.iter().copied().enumerate() {
        let semantic = &authority.producers[producer_index];
        let position = semantic.position();
        let producer = semantic.producer();
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
                let id = wrapper_id(buffers.static_wrappers.len())?;
                let Some(wrapper) = resolve_static(request, id, target_sm)
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
                        buffers,
                        monolithic,
                        authority.clone(),
                        producer_index,
                    );
                };
                ResolvedEvent::Static { request, wrapper }
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
                        .checked_add(producer_static)
                        .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
                )?;
                let Some(wrapper) = resolve_static(request, id, target_sm)
                    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
                else {
                    return missing_adapter(
                        MissingBaseAdapter::MultiplicityFeedStaticWrapper,
                        position,
                        producer,
                        manifest,
                        target_sm,
                        values,
                        buffers,
                        monolithic,
                        authority.clone(),
                        producer_index,
                    );
                };
                Some(ResolvedEvent::Static { request, wrapper })
            }
            Some(BaseEvent::Recorded(_)) => return Err(CompiledWitnessWriterPrefixError::Lowering),
        };

        // Both authorities are resolved before either event mutates the
        // returned prefix. A missing feed therefore cannot strand its writer.
        append_event(resolved_producer, target_sm, &monolithic, &mut buffers)?;
        if let Some(feed) = resolved_feed {
            append_event(feed, target_sm, &monolithic, &mut buffers)?;
        }
    }
    drop(sequence);
    let next_producer = authority.producers.len();
    finish_prefix(
        manifest,
        target_sm,
        values,
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
                effect.id(),
            )
            .map_err(|_| CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer))?;
            push_operation(
                ExecutionPrimitive::AotKernel {
                    kernel,
                    launch: recorded.source.launch,
                },
                recorded.invocation.clone(),
                effect.id(),
                monolithic,
                &mut buffers.operations,
            )
        }
        ResolvedEvent::Static { request, wrapper } => {
            let effect = request
                .effect()
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            let expected_id = wrapper_id(buffers.static_wrappers.len())?;
            if wrapper.id() != expected_id
                || wrapper.consumer_target_sm() != target_sm
                || wrapper.accepted_effect() != effect.id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
            {
                return Err(CompiledWitnessWriterPrefixError::Lowering);
            }
            let invocation = request
                .invocation()
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            insert_effect(&mut buffers.effects, effect.clone())
                .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
            buffers.static_wrappers.push(wrapper);
            push_operation(
                ExecutionPrimitive::StaticCudaWrapper {
                    wrapper: expected_id,
                },
                invocation,
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
    let required_preproducer_versions =
        validate_witness_writer_transitions(&base_authority, &values)
            .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    validate_sealed_prefix(
        &base_authority,
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
    Ok(CompiledWitnessWriterPrefix {
        execution_manifest_identity,
        target_sm,
        values,
        base_authority,
        next_producer,
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

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_sealed_prefix(
    authority: &BaseProducerAuthority,
    next_producer: usize,
    target_sm: u32,
    kernel_by_build_authority: &BTreeMap<[u8; 32], usize>,
    kernels: &[AotKernelAuthority],
    static_wrappers: &[StaticCudaWrapperAuthority],
    module_global_initializers: &[ModuleGlobalInitializer],
    effects: &[EffectContract],
    partitions: &[PartitionAuthority],
    monolithic: &PartitionAuthority,
    operations: &[OpNode],
) -> Result<(), ()> {
    let sequence = sequence(authority)?;
    if next_producer > sequence.witnesses.len() {
        return Err(());
    }
    let expected = sequence
        .prelude
        .iter()
        .copied()
        .chain(
            sequence.witnesses[..next_producer]
                .iter()
                .flat_map(|group| [Some(group.producer), group.feed].into_iter().flatten()),
        )
        .collect::<Vec<_>>();
    let expected_partitions = (!operations.is_empty())
        .then(|| monolithic.clone())
        .into_iter()
        .collect::<Vec<_>>();
    if operations.len() != expected.len()
        || partitions != expected_partitions
        || kernel_by_build_authority.len() != kernels.len()
        || kernel_by_build_authority
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            != (0..kernels.len()).collect()
        || kernels
            .iter()
            .enumerate()
            .any(|(index, kernel)| kernel.id().0 as usize != index + 1)
        || static_wrappers.iter().enumerate().any(|(index, wrapper)| {
            wrapper.id().0 as usize != index + 1
                || wrapper.consumer_target_sm() != target_sm
                || !wrapper.has_valid_identity().unwrap_or(false)
        })
        || effects.windows(2).any(|pair| pair[0].id() >= pair[1].id())
    {
        return Err(());
    }

    let mut used_kernels = BTreeMap::<AotKernelId, BTreeSet<_>>::new();
    let mut used_effects = BTreeSet::new();
    let mut used_initializers = BTreeSet::new();
    let mut next_wrapper = 1usize;
    for (index, (operation, event)) in operations.iter().zip(expected).enumerate() {
        if operation.id.0 as usize != index
            || operation.semantic_id.0 as usize != index + 1
            || operation.invocation.as_ref() != Some(&event.invocation().map_err(|_| ())?)
            || operation.partition != monolithic.id()
            || operation.stage
                != ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
        {
            return Err(());
        }
        let effect = effects
            .iter()
            .find(|effect| effect.id() == operation.effect)
            .ok_or(())?;
        used_effects.insert(operation.effect);
        match (event, &operation.primitive) {
            (BaseEvent::Recorded(recorded), ExecutionPrimitive::AotKernel { kernel, launch })
                if *launch == recorded.source.launch =>
            {
                let authority = kernels
                    .iter()
                    .find(|authority| authority.id() == *kernel)
                    .ok_or(())?;
                if effect.accesses() != recorded.effect.accesses()
                    || validate_recorded_module_global_effect(
                        &recorded.source,
                        authority.module(),
                        effect,
                        module_global_initializers,
                    )
                    .is_err()
                {
                    return Err(());
                }
                used_initializers.extend(
                    effect
                        .module_globals()
                        .iter()
                        .map(|global| global.initializer),
                );
                used_kernels
                    .entry(*kernel)
                    .or_default()
                    .insert((operation.effect, operation.partition));
            }
            (BaseEvent::Static(request), ExecutionPrimitive::StaticCudaWrapper { wrapper })
                if wrapper.0 as usize == next_wrapper =>
            {
                let authority = static_wrappers.get(next_wrapper - 1).ok_or(())?;
                if authority.id() != *wrapper
                    || authority.accepted_effect() != operation.effect
                    || effect != request.effect().map_err(|_| ())?
                {
                    return Err(());
                }
                next_wrapper += 1;
            }
            _ => return Err(()),
        }
    }
    let declared_effects = effects
        .iter()
        .map(EffectContract::id)
        .collect::<BTreeSet<_>>();
    let declared_initializers = module_global_initializers
        .iter()
        .enumerate()
        .map(|(index, initializer)| {
            (initializer.id().0 as usize == index
                && initializer.has_valid_identity().unwrap_or(false))
            .then_some(initializer.id())
            .ok_or(())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if next_wrapper != static_wrappers.len() + 1
        || declared_effects.len() != effects.len()
        || declared_effects != used_effects
        || declared_initializers.len() != module_global_initializers.len()
        || declared_initializers != used_initializers
        || used_kernels.len() != kernels.len()
    {
        return Err(());
    }
    for kernel in kernels {
        let accepted = kernel
            .accepted_executions()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if accepted.len() != kernel.accepted_executions().len()
            || used_kernels.remove(&kernel.id()) != Some(accepted)
        {
            return Err(());
        }
    }
    used_kernels.is_empty().then_some(()).ok_or(())
}
