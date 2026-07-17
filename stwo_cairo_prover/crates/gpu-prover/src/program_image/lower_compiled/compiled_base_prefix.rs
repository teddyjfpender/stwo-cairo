//! `CompiledProof` witness-writer prefix for scheduled Base trace writers.
//!
//! This is not a second IR. It emits canonical [`OpNode`] values directly from
//! the producer-owned invocation/effect contract and retains the one semantic
//! value allocator that later stages must continue. Execution-table, input,
//! multiplicity, memory and fixed-table preproducers are deliberately retained
//! as unresolved version requirements; this type is not a complete Base stage
//! and has no conversion into [`crate::compiled_proof::CompiledProof`].

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::aot::{self, AotKernelModuleGlobals};

use super::loaded_authority::LoadedAuthorityFields;
use super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::compiled_proof::{
    AotKernelAuthority, AotKernelId, EffectContract, EffectContractId, ExecutionPrimitive,
    FixedValueDesc, ModuleIdentity, OpId, OpNode, PartitionAuthority, ProofStage, SemanticOpId,
    ValueDesc, ValueVersion,
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

/// An IR-shaped witness-writer prefix, not a promotable partial proof.
///
/// The fixed descriptors are derived from `values` on demand so there is no
/// duplicate constant-content authority. Catalog-backed `ValueDesc` values are
/// deliberately deferred until all pre-Base producers and origins are typed.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct CompiledWitnessWriterPrefix {
    pub(super) execution_manifest_identity: [u8; 32],
    pub(super) target_sm: u32,
    /// The one proof-wide map used to lower `base_authority`. Until
    /// `next_producer == base_authority.producers.len()`, its current versions
    /// describe the terminal Base state and must travel with the unconsumed
    /// producer contracts rather than being treated as the emitted frontier.
    values: adapter::SemanticValueMap,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
    required_preproducer_versions: BTreeSet<ValueVersion>,
    kernel_by_loaded_authority: BTreeMap<[u8; 32], usize>,
    pub(super) kernels: Vec<AotKernelAuthority>,
    pub(super) effects: Vec<EffectContract>,
    pub(super) partitions: Vec<PartitionAuthority>,
    pub(super) operations: Vec<OpNode>,
}

impl CompiledWitnessWriterPrefix {
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
    pub(super) const fn required_preproducer_versions(&self) -> &BTreeSet<ValueVersion> {
        &self.required_preproducer_versions
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
pub(super) enum CompiledWitnessWriterPrefixError {
    Lowering,
    InvalidTargetSm,
    MissingRecordedAotAuthority(WitnessProducer),
    InvalidRecordedAotAuthority(WitnessProducer),
    MissingTypedAdapter {
        missing: MissingBaseAdapterAt,
        prefix: Box<CompiledWitnessWriterPrefix>,
    },
}

/// Resolve exact embedded AOT authorities without initializing CUDA, emit each
/// preceding recorded witness writer, then fail at the first native wrapper.
pub(super) fn emit_recorded_witness_writer_prefix(
    arena: &ProofArenaPlan,
    target_sm: u32,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    let manifest = aot::loaded_manifest_identity();
    emit_recorded_witness_writer_prefix_using(arena, manifest, target_sm, |source| {
        let (major, minor) = split_sm(target_sm)?;
        let kernel = aot::loaded_kernel_authority(source.cache_key, major, minor)
            .ok_or(ResolveRecordedAuthorityError::Missing)?;
        Ok(LoadedAuthorityFields::from_loaded(manifest, kernel))
    })
}

fn emit_recorded_witness_writer_prefix_using(
    arena: &ProofArenaPlan,
    manifest: [u8; 32],
    target_sm: u32,
    mut resolve: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<LoadedAuthorityFields, ResolveRecordedAuthorityError>,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    let (sm_major, sm_minor) =
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
    let authority = BaseProducerAuthority::compile_replacement_into(arena, &mut values)
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
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
            != u32::try_from(operations.len()).map_err(|_| {
                CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
            })?
        {
            return Err(CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer));
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
                CompiledWitnessWriterPrefixError::MissingRecordedAotAuthority(producer)
            }
            ResolveRecordedAuthorityError::Invalid => {
                CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
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
            return Err(CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer));
        }
        insert_effect(&mut effects, recorded.effect.clone())
            .map_err(|_| CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer))?;
        let kernel_id = install_recorded_kernel(
            &mut kernels,
            &mut kernel_by_loaded_authority,
            &recorded.source,
            &fields,
            recorded.effect.id(),
        )
        .map_err(|_| CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer))?;
        let id = OpId(u32::try_from(operations.len()).map_err(|_| {
            CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer)
        })?);
        operations.push(OpNode {
            id,
            semantic_id: SemanticOpId(
                position.ordinal.checked_add(1).ok_or(
                    CompiledWitnessWriterPrefixError::InvalidRecordedAotAuthority(producer),
                )?,
            ),
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
    finish_prefix(
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
    )
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
            kernels,
            kernel_by_loaded_authority,
            effects,
            operations,
            monolithic,
            base_authority,
            next_producer,
        )?),
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
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
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
        &kernel_by_loaded_authority,
        &kernels,
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
        kernel_by_loaded_authority,
        kernels,
        effects,
        partitions,
        operations,
    })
}

fn validate_witness_writer_transitions(
    authority: &BaseProducerAuthority,
    values: &adapter::SemanticValueMap,
) -> Result<BTreeSet<ValueVersion>, ()> {
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    let allocated = catalog_first
        .iter()
        .chain(&transitions)
        .chain(&fixed)
        .copied()
        .collect::<BTreeSet<_>>();
    if allocated != values.allocated_versions().collect()
        || !catalog_first.is_disjoint(&transitions)
        || !catalog_first.is_disjoint(&fixed)
        || !transitions.is_disjoint(&fixed)
    {
        return Err(());
    }
    let mut destination_producer = BTreeMap::<ValueVersion, usize>::new();
    let mut sources = Vec::new();
    for (producer_index, producer) in authority.producers.iter().enumerate() {
        if producer.position().ordinal as usize != producer_index {
            return Err(());
        }
        for access in producer.effect().accesses() {
            if let Some(destination) = access.destination() {
                let version = destination.value.version;
                if !allocated.contains(&version) {
                    return Err(());
                }
                match destination_producer.insert(version, producer_index) {
                    Some(previous) if previous != producer_index => return Err(()),
                    _ => {}
                }
            }
            if let Some(source) = access.source() {
                if !allocated.contains(&source.value.version) {
                    return Err(());
                }
                sources.push((producer_index, source.value.version));
            }
        }
    }

    if transitions
        .iter()
        .any(|version| !destination_producer.contains_key(version))
    {
        return Err(());
    }

    // Sources without a writer destination have no ValueOrigin yet. Fixed
    // sources already have Constant origins; the whole-proof builder must
    // classify each remaining first catalog allocation as a typed ExternalInput
    // or the OpOutput of a real preproducer.
    let required_preproducer_versions = sources
        .iter()
        .filter_map(|&(_, source)| {
            (!destination_producer.contains_key(&source) && !fixed.contains(&source))
                .then_some(source)
        })
        .collect::<BTreeSet<_>>();
    if !required_preproducer_versions.is_subset(&catalog_first)
        || !required_preproducer_versions.is_disjoint(&transitions)
        || !required_preproducer_versions.is_disjoint(&fixed)
        || required_preproducer_versions
            .iter()
            .any(|version| destination_producer.contains_key(version))
    {
        return Err(());
    }

    for (producer_index, source) in sources {
        match destination_producer.get(&source) {
            Some(&source_producer) if source_producer < producer_index => {}
            None if required_preproducer_versions.contains(&source) || fixed.contains(&source) => {}
            _ => return Err(()),
        }
    }
    Ok(required_preproducer_versions)
}

fn validate_sealed_prefix(
    authority: &BaseProducerAuthority,
    next_producer: usize,
    kernel_by_loaded_authority: &BTreeMap<[u8; 32], usize>,
    kernels: &[AotKernelAuthority],
    effects: &[EffectContract],
    partitions: &[PartitionAuthority],
    monolithic: &PartitionAuthority,
    operations: &[OpNode],
) -> Result<(), ()> {
    let expected_partitions = (!operations.is_empty())
        .then(|| monolithic.clone())
        .into_iter()
        .collect::<Vec<_>>();
    if next_producer != operations.len()
        || next_producer > authority.producers.len()
        || partitions != expected_partitions
        || kernel_by_loaded_authority.len() != kernels.len()
        || kernel_by_loaded_authority
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            != (0..kernels.len()).collect()
    {
        return Err(());
    }

    let mut used = BTreeMap::<AotKernelId, BTreeSet<_>>::new();
    let mut used_effects = BTreeSet::new();
    for (index, operation) in operations.iter().enumerate() {
        let ExecutionPrimitive::AotKernel { kernel, .. } = &operation.primitive else {
            return Err(());
        };
        if operation.id.0 as usize != index
            || operation.semantic_id.0 as usize != index + 1
            || operation.effect != authority.producers[index].effect().id()
            || operation.partition != monolithic.id()
        {
            return Err(());
        }
        used_effects.insert(operation.effect);
        used.entry(*kernel)
            .or_default()
            .insert((operation.effect, operation.partition));
    }
    let declared_effects = effects
        .iter()
        .map(EffectContract::id)
        .collect::<BTreeSet<_>>();
    if declared_effects.len() != effects.len() || declared_effects != used_effects {
        return Err(());
    }
    if used.len() != kernels.len() {
        return Err(());
    }
    for kernel in kernels {
        let accepted = kernel
            .accepted_executions()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if accepted.len() != kernel.accepted_executions().len()
            || used.remove(&kernel.id()) != Some(accepted)
        {
            return Err(());
        }
    }
    used.is_empty().then_some(()).ok_or(())
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
pub(super) fn install_recorded_kernel_pair_for_test(
    first_source: &RecordedWitnessInvocationShape,
    first_fields: &LoadedAuthorityFields,
    first_effect: EffectContractId,
    second_source: &RecordedWitnessInvocationShape,
    second_fields: &LoadedAuthorityFields,
    second_effect: EffectContractId,
) -> Result<(Vec<AotKernelAuthority>, AotKernelId, AotKernelId), ()> {
    let mut kernels = Vec::new();
    let mut by_authority = BTreeMap::new();
    let first = install_recorded_kernel(
        &mut kernels,
        &mut by_authority,
        first_source,
        first_fields,
        first_effect,
    )?;
    let second = install_recorded_kernel(
        &mut kernels,
        &mut by_authority,
        second_source,
        second_fields,
        second_effect,
    )?;
    Ok((kernels, first, second))
}

#[cfg(test)]
pub(super) fn validate_witness_writer_transitions_for_test(
    authority: &BaseProducerAuthority,
    values: &adapter::SemanticValueMap,
) -> Result<BTreeSet<ValueVersion>, ()> {
    validate_witness_writer_transitions(authority, values)
}

#[cfg(test)]
pub(super) fn validate_sealed_prefix_for_test(
    prefix: &CompiledWitnessWriterPrefix,
    next_producer: usize,
    operations: &[OpNode],
) -> Result<(), ()> {
    validate_sealed_prefix(
        &prefix.base_authority,
        next_producer,
        &prefix.kernel_by_loaded_authority,
        &prefix.kernels,
        &prefix.effects,
        &prefix.partitions,
        &PartitionAuthority::monolithic(),
        operations,
    )
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
) -> Result<WitnessProducer, CompiledWitnessWriterPrefixError> {
    crate::resident_runtime::producer_schedule::BaseProducerSchedule::compile(arena)
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?
        .witness_levels()
        .iter()
        .flatten()
        .copied()
        .next()
        .ok_or(CompiledWitnessWriterPrefixError::Lowering)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResolveRecordedAuthorityError {
    Missing,
    Invalid,
}

#[cfg(test)]
pub(super) fn emit_recorded_witness_writer_prefix_for_test(
    arena: &ProofArenaPlan,
    manifest: [u8; 32],
    target_sm: u32,
    resolve: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<LoadedAuthorityFields, ResolveRecordedAuthorityError>,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    emit_recorded_witness_writer_prefix_using(arena, manifest, target_sm, resolve)
}
