//! Production `CompiledProof` prefix for ordinary recorded Base producers.
//!
//! This is not a second IR. It emits canonical [`OpNode`] values directly from
//! the producer-owned invocation/effect contract and retains the one semantic
//! value allocator that later stages must continue. Native linked wrappers are
//! a hard frontier until their `StaticCudaWrapper` authority is available.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::aot::{self, AotKernelModuleGlobals};

use super::loaded_authority::LoadedAuthorityFields;
use super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::compiled_proof::{
    AotKernelAuthority, AotKernelId, EffectContract, EffectContractId, ExecutionPrimitive,
    FixedValueDesc, ModuleIdentity, OpId, OpNode, PartitionAuthority, ProofStage, SemanticOpId,
    ValueDesc,
};
use crate::resident_runtime::producer_schedule::WitnessProducer;
use crate::transcript_plan::CairoTranscriptSegment;

const MODULE_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.module.v1\0";
const SEMANTIC_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.semantic.v1\0";
const BUILD_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.execution-build.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MissingBaseAdapter {
    NativeBlakeGDirectStaticWrapper,
    NativeEcOpStaticWrapper,
    RecordedModuleGlobals,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MissingBaseAdapterAt {
    pub(super) adapter: MissingBaseAdapter,
    pub(super) producer: WitnessProducer,
    pub(super) schedule_ordinal: u32,
}

/// A real `CompiledProof` prefix, not a promotable partial proof.
///
/// The fixed descriptors are derived from `values` on demand so there is no
/// duplicate constant-content authority. Catalog-backed `ValueDesc` values are
/// deliberately deferred until all pre-Base producers and origins are typed.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct CompiledBasePrefix {
    pub(super) execution_manifest_identity: [u8; 32],
    pub(super) target_sm: u32,
    /// The one proof-wide map used to lower `base_authority`. Until
    /// `next_producer == base_authority.producers.len()`, its current versions
    /// describe the terminal Base state and must travel with the unconsumed
    /// producer contracts rather than being treated as the emitted frontier.
    values: adapter::SemanticValueMap,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
    kernel_by_loaded_authority: BTreeMap<[u8; 32], usize>,
    pub(super) kernels: Vec<AotKernelAuthority>,
    pub(super) effects: Vec<EffectContract>,
    pub(super) partitions: Vec<PartitionAuthority>,
    pub(super) operations: Vec<OpNode>,
}

impl CompiledBasePrefix {
    pub(super) fn fixed_values(&self) -> Vec<FixedValueDesc> {
        self.values.fixed_values()
    }

    pub(super) fn fixed_value_versions(&self) -> Vec<ValueDesc> {
        self.values.fixed_value_versions()
    }

    pub(super) fn direct_retained_b2n(&self) -> &stwo_backend_cuda::DirectRetainedB2nProgram {
        &self.base_authority.direct_retained_b2n
    }

    pub(super) fn pending_producers(&self) -> &[SemanticBaseProducer] {
        &self.base_authority.producers[self.next_producer..]
    }

    #[cfg(test)]
    pub(super) const fn values(&self) -> &adapter::SemanticValueMap {
        &self.values
    }

    #[cfg(test)]
    pub(super) const fn base_authority(&self) -> &BaseProducerAuthority {
        &self.base_authority
    }

    #[cfg(test)]
    pub(super) const fn next_producer(&self) -> usize {
        self.next_producer
    }

    #[cfg(test)]
    pub(super) const fn recorded_kernel_authorities(&self) -> &BTreeMap<[u8; 32], usize> {
        &self.kernel_by_loaded_authority
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) enum CompiledBasePrefixError {
    Lowering,
    InvalidTargetSm,
    MissingRecordedAotAuthority(WitnessProducer),
    InvalidRecordedAotAuthority(WitnessProducer),
    MissingTypedAdapter {
        missing: MissingBaseAdapterAt,
        prefix: Box<CompiledBasePrefix>,
    },
}

/// Resolve exact embedded AOT authorities without initializing CUDA, emit every
/// preceding ordinary Base operation, then fail at the first native wrapper.
pub(super) fn emit_recorded_base_prefix(
    arena: &ProofArenaPlan,
    target_sm: u32,
) -> Result<CompiledBasePrefix, CompiledBasePrefixError> {
    let manifest = aot::loaded_manifest_identity();
    emit_recorded_base_prefix_using(arena, manifest, target_sm, |source| {
        let (major, minor) = split_sm(target_sm)?;
        let kernel = aot::loaded_kernel_authority(source.cache_key, major, minor)
            .ok_or(ResolveRecordedAuthorityError::Missing)?;
        Ok(LoadedAuthorityFields::from_loaded(manifest, kernel))
    })
}

fn emit_recorded_base_prefix_using(
    arena: &ProofArenaPlan,
    manifest: [u8; 32],
    target_sm: u32,
    mut resolve: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<LoadedAuthorityFields, ResolveRecordedAuthorityError>,
) -> Result<CompiledBasePrefix, CompiledBasePrefixError> {
    let (sm_major, sm_minor) =
        split_sm(target_sm).map_err(|_| CompiledBasePrefixError::InvalidTargetSm)?;
    if manifest == [0; 32] {
        return Err(CompiledBasePrefixError::MissingRecordedAotAuthority(
            first_scheduled_producer(arena)?,
        ));
    }

    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .map_err(|_| CompiledBasePrefixError::Lowering)?;
    let authority = BaseProducerAuthority::compile_replacement_into(arena, &mut values)
        .map_err(|_| CompiledBasePrefixError::Lowering)?;
    let mut kernels = Vec::new();
    let mut effects = BTreeMap::<EffectContractId, EffectContract>::new();
    let mut operations = Vec::new();
    let mut kernel_by_loaded_authority = BTreeMap::<[u8; 32], usize>::new();
    let monolithic = PartitionAuthority::monolithic();

    for producer_index in 0..authority.producers.len() {
        let semantic = authority.producers[producer_index].clone();
        let position = semantic.position();
        let producer = semantic.producer();
        let recorded = match semantic {
            SemanticBaseProducer::Recorded(recorded) => recorded,
            SemanticBaseProducer::NativeBlakeGDirect {
                position, producer, ..
            } => {
                return missing_adapter(
                    MissingBaseAdapter::NativeBlakeGDirectStaticWrapper,
                    position,
                    producer,
                    manifest,
                    target_sm,
                    values,
                    kernels,
                    kernel_by_loaded_authority,
                    effects,
                    operations,
                    monolithic,
                    authority,
                    producer_index,
                )
            }
            SemanticBaseProducer::NativeEcOp {
                position, producer, ..
            } => {
                return missing_adapter(
                    MissingBaseAdapter::NativeEcOpStaticWrapper,
                    position,
                    producer,
                    manifest,
                    target_sm,
                    values,
                    kernels,
                    kernel_by_loaded_authority,
                    effects,
                    operations,
                    monolithic,
                    authority,
                    producer_index,
                )
            }
        };
        if position.ordinal
            != u32::try_from(operations.len())
                .map_err(|_| CompiledBasePrefixError::InvalidRecordedAotAuthority(producer))?
        {
            return Err(CompiledBasePrefixError::InvalidRecordedAotAuthority(
                producer,
            ));
        }
        if recorded.source.deduce.module_state.is_some() {
            return missing_adapter(
                MissingBaseAdapter::RecordedModuleGlobals,
                position,
                producer,
                manifest,
                target_sm,
                values,
                kernels,
                kernel_by_loaded_authority,
                effects,
                operations,
                monolithic,
                authority,
                producer_index,
            );
        }

        let fields = resolve(&recorded.source).map_err(|error| match error {
            ResolveRecordedAuthorityError::Missing => {
                CompiledBasePrefixError::MissingRecordedAotAuthority(producer)
            }
            ResolveRecordedAuthorityError::Invalid => {
                CompiledBasePrefixError::InvalidRecordedAotAuthority(producer)
            }
        })?;
        if fields.manifest_identity != manifest
            || fields.module_globals != AotKernelModuleGlobals::None
            || super::loaded_authority::validate_fields(
                &recorded.source,
                sm_major,
                sm_minor,
                &fields,
            )
            .is_err()
        {
            return Err(CompiledBasePrefixError::InvalidRecordedAotAuthority(
                producer,
            ));
        }
        insert_effect(&mut effects, recorded.effect.clone())
            .map_err(|_| CompiledBasePrefixError::InvalidRecordedAotAuthority(producer))?;
        let kernel_id = install_recorded_kernel(
            &mut kernels,
            &mut kernel_by_loaded_authority,
            &recorded.source,
            &fields,
            recorded.effect.id(),
        )
        .map_err(|_| CompiledBasePrefixError::InvalidRecordedAotAuthority(producer))?;
        let id = OpId(
            u32::try_from(operations.len())
                .map_err(|_| CompiledBasePrefixError::InvalidRecordedAotAuthority(producer))?,
        );
        operations.push(OpNode {
            id,
            semantic_id: SemanticOpId(position.ordinal.checked_add(1).ok_or(
                CompiledBasePrefixError::InvalidRecordedAotAuthority(producer),
            )?),
            primitive: ExecutionPrimitive::AotKernel {
                kernel: kernel_id,
                launch: recorded.source.launch,
            },
            invocation: Some(recorded.invocation.clone()),
            effect: recorded.effect.id(),
            partition: monolithic.id(),
            stage: ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase),
        });
    }

    let next_producer = authority.producers.len();
    Ok(finish_prefix(
        manifest,
        target_sm,
        values,
        kernels,
        kernel_by_loaded_authority,
        effects,
        operations,
        monolithic,
        authority,
        next_producer,
    ))
}

#[allow(clippy::too_many_arguments)]
fn missing_adapter(
    adapter: MissingBaseAdapter,
    position: super::producer_prefix::ProducerSchedulePosition,
    producer: WitnessProducer,
    manifest: [u8; 32],
    target_sm: u32,
    values: adapter::SemanticValueMap,
    kernels: Vec<AotKernelAuthority>,
    kernel_by_loaded_authority: BTreeMap<[u8; 32], usize>,
    effects: BTreeMap<EffectContractId, EffectContract>,
    operations: Vec<OpNode>,
    monolithic: PartitionAuthority,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
) -> Result<CompiledBasePrefix, CompiledBasePrefixError> {
    Err(CompiledBasePrefixError::MissingTypedAdapter {
        missing: MissingBaseAdapterAt {
            adapter,
            producer,
            schedule_ordinal: position.ordinal,
        },
        prefix: Box::new(finish_prefix(
            manifest,
            target_sm,
            values,
            kernels,
            kernel_by_loaded_authority,
            effects,
            operations,
            monolithic,
            base_authority,
            next_producer,
        )),
    })
}

fn finish_prefix(
    execution_manifest_identity: [u8; 32],
    target_sm: u32,
    values: adapter::SemanticValueMap,
    kernels: Vec<AotKernelAuthority>,
    kernel_by_loaded_authority: BTreeMap<[u8; 32], usize>,
    effects: BTreeMap<EffectContractId, EffectContract>,
    operations: Vec<OpNode>,
    monolithic: PartitionAuthority,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
) -> CompiledBasePrefix {
    debug_assert!(next_producer <= base_authority.producers.len());
    CompiledBasePrefix {
        execution_manifest_identity,
        target_sm,
        values,
        base_authority,
        next_producer,
        kernel_by_loaded_authority,
        kernels,
        effects: effects.into_values().collect(),
        partitions: (!operations.is_empty())
            .then_some(monolithic)
            .into_iter()
            .collect(),
        operations,
    }
}

fn compiled_kernel(
    id: AotKernelId,
    source: &RecordedWitnessInvocationShape,
    fields: &LoadedAuthorityFields,
    effect: EffectContractId,
) -> Result<AotKernelAuthority, ()> {
    let module_encoding = encode_module(fields)?;
    let module = ModuleIdentity::new(module_encoding).map_err(|_| ())?;
    AotKernelAuthority::new(
        id,
        module,
        encode_semantic(source, fields)?,
        encode_execution_build(fields)?,
        vec![effect],
    )
    .map_err(|_| ())
}

fn install_recorded_kernel(
    kernels: &mut Vec<AotKernelAuthority>,
    kernel_by_loaded_authority: &mut BTreeMap<[u8; 32], usize>,
    source: &RecordedWitnessInvocationShape,
    fields: &LoadedAuthorityFields,
    effect: EffectContractId,
) -> Result<AotKernelId, ()> {
    let Some(&index) = kernel_by_loaded_authority.get(&fields.authority_identity) else {
        let id = AotKernelId(u32::try_from(kernels.len() + 1).map_err(|_| ())?);
        let kernel = compiled_kernel(id, source, fields, effect)?;
        kernel_by_loaded_authority.insert(fields.authority_identity, kernels.len());
        kernels.push(kernel);
        return Ok(id);
    };

    let existing = kernels.get(index).ok_or(())?;
    let id = existing.id();
    let candidate = compiled_kernel(id, source, fields, effect)?;
    if existing.module() != candidate.module()
        || existing.semantic_encoding() != candidate.semantic_encoding()
        || existing.execution_build_encoding() != candidate.execution_build_encoding()
    {
        return Err(());
    }
    let monolithic = PartitionAuthority::monolithic().id();
    let mut accepted = existing
        .accepted_executions()
        .iter()
        .map(|&(accepted_effect, partition)| {
            (partition == monolithic)
                .then_some(accepted_effect)
                .ok_or(())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    accepted.insert(effect);
    let module = existing.module().clone();
    let semantic_encoding = existing.semantic_encoding().to_vec();
    let execution_build_encoding = existing.execution_build_encoding().to_vec();
    kernels[index] = AotKernelAuthority::new(
        id,
        module,
        semantic_encoding,
        execution_build_encoding,
        accepted.into_iter().collect(),
    )
    .map_err(|_| ())?;
    Ok(id)
}

#[cfg(test)]
pub(super) fn install_recorded_kernel_twice_for_test(
    source: &RecordedWitnessInvocationShape,
    fields: &LoadedAuthorityFields,
    effect: EffectContractId,
) -> Result<(Vec<AotKernelAuthority>, AotKernelId, AotKernelId), ()> {
    let mut kernels = Vec::new();
    let mut by_authority = BTreeMap::new();
    let first = install_recorded_kernel(&mut kernels, &mut by_authority, source, fields, effect)?;
    let second = install_recorded_kernel(&mut kernels, &mut by_authority, source, fields, effect)?;
    Ok((kernels, first, second))
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), ()> {
    out.extend_from_slice(&u64::try_from(bytes.len()).map_err(|_| ())?.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_module(fields: &LoadedAuthorityFields) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(MODULE_DOMAIN);
    out.extend_from_slice(&fields.manifest_identity);
    out.extend_from_slice(&fields.cubin_identity);
    out.extend_from_slice(&fields.authority_identity);
    out.extend_from_slice(&fields.target_sm.to_le_bytes());
    push_bytes(&mut out, fields.kernel_symbol.as_bytes())?;
    Ok(out)
}

fn encode_semantic(
    source: &RecordedWitnessInvocationShape,
    fields: &LoadedAuthorityFields,
) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(SEMANTIC_DOMAIN);
    out.extend_from_slice(&source.program_identity);
    out.extend_from_slice(&source.deduce.source_identity);
    out.extend_from_slice(&source.abi_schema_identity);
    out.extend_from_slice(&source.semantic_hash.to_le_bytes());
    out.extend_from_slice(&source.cache_key.to_le_bytes());
    push_bytes(&mut out, source.kernel_symbol.as_bytes())?;
    Ok(out)
}

fn encode_execution_build(fields: &LoadedAuthorityFields) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(BUILD_DOMAIN);
    out.extend_from_slice(&fields.manifest_identity);
    out.extend_from_slice(&fields.source_identity);
    out.extend_from_slice(&fields.cubin_identity);
    out.extend_from_slice(&fields.authority_identity);
    out.extend_from_slice(&fields.target_sm.to_le_bytes());
    out.extend_from_slice(&fields.cache_key.to_le_bytes());
    out.push(AotKernelModuleGlobals::None as u8);
    push_bytes(&mut out, fields.kernel_symbol.as_bytes())?;
    Ok(out)
}

fn insert_effect(
    effects: &mut BTreeMap<EffectContractId, EffectContract>,
    effect: EffectContract,
) -> Result<(), ()> {
    match effects.insert(effect.id(), effect.clone()) {
        None => Ok(()),
        Some(existing) if existing == effect => Ok(()),
        Some(_) => Err(()),
    }
}

fn split_sm(target_sm: u32) -> Result<(u32, u32), ResolveRecordedAuthorityError> {
    let major = target_sm / 10;
    let minor = target_sm % 10;
    if major == 0 {
        Err(ResolveRecordedAuthorityError::Invalid)
    } else {
        Ok((major, minor))
    }
}

fn first_scheduled_producer(
    arena: &ProofArenaPlan,
) -> Result<WitnessProducer, CompiledBasePrefixError> {
    crate::resident_runtime::producer_schedule::BaseProducerSchedule::compile(arena)
        .map_err(|_| CompiledBasePrefixError::Lowering)?
        .witness_levels()
        .iter()
        .flatten()
        .copied()
        .next()
        .ok_or(CompiledBasePrefixError::Lowering)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResolveRecordedAuthorityError {
    Missing,
    Invalid,
}

#[cfg(test)]
pub(super) fn emit_recorded_base_prefix_for_test(
    arena: &ProofArenaPlan,
    manifest: [u8; 32],
    target_sm: u32,
    resolve: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<LoadedAuthorityFields, ResolveRecordedAuthorityError>,
) -> Result<CompiledBasePrefix, CompiledBasePrefixError> {
    emit_recorded_base_prefix_using(arena, manifest, target_sm, resolve)
}
