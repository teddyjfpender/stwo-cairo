//! `CompiledProof` witness-writer prefix for scheduled Base trace writers.
//!
//! This is not a second IR. It emits canonical [`OpNode`] values directly from
//! the producer-owned invocation/effect contract and retains the one semantic
//! value allocator that later stages must continue. Execution-table splits and
//! multiplicity operations are explicit; host ingress, witness inputs, memory,
//! and fixed-table work still retain unresolved origins. This type is not a
//! complete Base stage and cannot become a [`crate::compiled_proof::CompiledProof`].

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::aot;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::resolved_recorded_build_authority::ResolvedRecordedBuildAuthority;
use super::*;
use crate::compiled_proof::{
    AotInvocation, AotKernelAuthority, AotKernelId, EffectContract, EffectContractId,
    ExecutionPrimitive, FixedValueDesc, ModuleGlobalInitializer, ModuleIdentity, OpId, OpNode,
    PartitionAuthority, ProofStage, SemanticOpId, StaticCudaWrapperAuthority, StaticCudaWrapperId,
    ValueDesc, ValueVersion,
};
use crate::resident_runtime::producer_schedule::WitnessProducer;
use crate::transcript_plan::CairoTranscriptSegment;

pub(super) mod emission;
#[cfg(test)]
mod module_global_tests;
pub(super) mod module_globals;
#[cfg(test)]
pub(super) mod test_support;

use module_globals::{
    resolve as resolve_recorded_module_globals,
    validate_effect as validate_recorded_module_global_effect,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StaticWrapperKind {
    ExecutionTableBig,
    ExecutionTableSmall,
    MultiplicityClear,
    PublicMemorySeed,
    MultiplicityFeed,
    NativeBlakeGDirect,
    NativeEcOp,
}

const MODULE_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.module.v1\0";
const SEMANTIC_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.semantic.v1\0";
const BUILD_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.execution-build.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MissingBaseAdapter {
    MultiplicityFeedStaticWrapper,
    NativeBlakeGDirectStaticWrapper,
    NativeEcOpStaticWrapper,
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
    /// The one proof-wide map used to lower `base_authority`. It describes the
    /// post-witness semantic state for the complete authority, including
    /// unconsumed producer contracts in a partial prefix. It is neither the
    /// emitted frontier nor the proof's final Base state: memory and fixed-table
    /// lowering still follow.
    values: adapter::SemanticValueMap,
    base_authority: BaseProducerAuthority,
    next_producer: usize,
    required_preproducer_versions: BTreeSet<ValueVersion>,
    kernel_by_build_authority: BTreeMap<[u8; 32], usize>,
    pub(super) kernels: Vec<AotKernelAuthority>,
    pub(super) static_wrappers: Vec<StaticCudaWrapperAuthority>,
    pub(super) module_global_initializers: Vec<ModuleGlobalInitializer>,
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
        &self.kernel_by_build_authority
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

/// Resolve exact embedded AOT and linked-static authorities without
/// initializing CUDA, emitting the longest fully authorized witness prefix.
pub(super) fn emit_recorded_witness_writer_prefix(
    arena: &ProofArenaPlan,
    preprocessed_trace_variant: PreProcessedTraceVariant,
    target_sm: u32,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    let manifest = aot::loaded_manifest_identity();
    emission::emit_using(
        arena,
        preprocessed_trace_variant,
        manifest,
        target_sm,
        |source| {
            let (major, minor) = split_sm(target_sm)?;
            let kernel = aot::loaded_kernel_authority(source.cache_key, major, minor)
                .ok_or(ResolveRecordedAuthorityError::Missing)?;
            Ok(ResolvedRecordedBuildAuthority::from_embedded(
                manifest, kernel,
            ))
        },
        emission::resolve_static,
    )
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
    let effects = emission::ordered_effects(authority)?;
    for (operation_index, effect) in effects.iter().enumerate() {
        for access in effect.accesses() {
            if let Some(destination) = access.destination() {
                let version = destination.value.version;
                if !allocated.contains(&version) {
                    return Err(());
                }
                match destination_producer.insert(version, operation_index) {
                    Some(previous) if previous != operation_index => return Err(()),
                    _ => {}
                }
            }
            if let Some(source) = access.source() {
                if !allocated.contains(&source.value.version) {
                    return Err(());
                }
                sources.push((operation_index, source.value.version));
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

fn compiled_kernel(
    id: AotKernelId,
    source: &RecordedWitnessInvocationShape,
    fields: &ResolvedRecordedBuildAuthority,
    effect: EffectContractId,
) -> Result<AotKernelAuthority, ()> {
    let module = compiled_module(fields)?;
    AotKernelAuthority::new(
        id,
        module,
        encode_semantic(source)?,
        encode_execution_build(fields)?,
        vec![effect],
    )
    .map_err(|_| ())
}

fn compiled_module(fields: &ResolvedRecordedBuildAuthority) -> Result<ModuleIdentity, ()> {
    ModuleIdentity::new(encode_module(fields)?).map_err(|_| ())
}

fn install_recorded_kernel(
    kernels: &mut Vec<AotKernelAuthority>,
    kernel_by_build_authority: &mut BTreeMap<[u8; 32], usize>,
    source: &RecordedWitnessInvocationShape,
    fields: &ResolvedRecordedBuildAuthority,
    effect: EffectContractId,
) -> Result<AotKernelId, ()> {
    let Some(&index) = kernel_by_build_authority.get(&fields.authority_identity) else {
        let id = AotKernelId(u32::try_from(kernels.len() + 1).map_err(|_| ())?);
        let kernel = compiled_kernel(id, source, fields, effect)?;
        kernel_by_build_authority.insert(fields.authority_identity, kernels.len());
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
    first_fields: &ResolvedRecordedBuildAuthority,
    first_effect: EffectContractId,
    second_source: &RecordedWitnessInvocationShape,
    second_fields: &ResolvedRecordedBuildAuthority,
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
    emission::validate_sealed_prefix(
        &prefix.base_authority,
        next_producer,
        prefix.target_sm,
        &prefix.kernel_by_build_authority,
        &prefix.kernels,
        &prefix.static_wrappers,
        &prefix.module_global_initializers,
        &prefix.effects,
        &prefix.partitions,
        &PartitionAuthority::monolithic(),
        operations,
    )
}

#[cfg(test)]
pub(super) fn validate_recorded_module_global_effect_for_test(
    source: &RecordedWitnessInvocationShape,
    module: &ModuleIdentity,
    effect: &EffectContract,
    initializers: &[ModuleGlobalInitializer],
) -> Result<(), ()> {
    validate_recorded_module_global_effect(source, module, effect, initializers)
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), ()> {
    out.extend_from_slice(&u64::try_from(bytes.len()).map_err(|_| ())?.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_module(fields: &ResolvedRecordedBuildAuthority) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(MODULE_DOMAIN);
    out.extend_from_slice(&fields.manifest_identity);
    out.extend_from_slice(&fields.cubin_identity);
    out.extend_from_slice(&fields.authority_identity);
    out.extend_from_slice(&fields.target_sm.to_le_bytes());
    push_bytes(&mut out, fields.kernel_symbol.as_bytes())?;
    Ok(out)
}

fn encode_semantic(source: &RecordedWitnessInvocationShape) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(SEMANTIC_DOMAIN);
    out.extend_from_slice(&source.program_identity);
    out.extend_from_slice(&source.deduce.source_identity);
    out.extend_from_slice(&source.abi_schema_identity);
    out.extend_from_slice(&source.semantic_hash.to_le_bytes());
    out.extend_from_slice(&source.cache_key.to_le_bytes());
    push_bytes(&mut out, source.kernel_symbol.as_bytes())?;
    Ok(out)
}

fn encode_execution_build(fields: &ResolvedRecordedBuildAuthority) -> Result<Vec<u8>, ()> {
    let mut out = Vec::from(BUILD_DOMAIN);
    out.extend_from_slice(&fields.manifest_identity);
    out.extend_from_slice(&fields.source_identity);
    out.extend_from_slice(&fields.cubin_identity);
    out.extend_from_slice(&fields.authority_identity);
    out.extend_from_slice(&fields.target_sm.to_le_bytes());
    out.extend_from_slice(&fields.cache_key.to_le_bytes());
    out.push(fields.module_globals as u8);
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

fn push_operation(
    primitive: ExecutionPrimitive,
    invocation: AotInvocation,
    effect: EffectContractId,
    monolithic: &PartitionAuthority,
    operations: &mut Vec<OpNode>,
) -> Result<(), CompiledWitnessWriterPrefixError> {
    let ordinal =
        u32::try_from(operations.len()).map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?;
    operations.push(OpNode {
        id: OpId(ordinal),
        semantic_id: SemanticOpId(
            ordinal
                .checked_add(1)
                .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
        ),
        primitive,
        invocation: Some(invocation),
        effect,
        partition: monolithic.id(),
        stage: ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase),
    });
    Ok(())
}

fn wrapper_id(existing: usize) -> Result<StaticCudaWrapperId, CompiledWitnessWriterPrefixError> {
    Ok(StaticCudaWrapperId(
        u32::try_from(
            existing
                .checked_add(1)
                .ok_or(CompiledWitnessWriterPrefixError::Lowering)?,
        )
        .map_err(|_| CompiledWitnessWriterPrefixError::Lowering)?,
    ))
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
    preprocessed_trace_variant: PreProcessedTraceVariant,
    manifest: [u8; 32],
    target_sm: u32,
    resolve: impl FnMut(
        &RecordedWitnessInvocationShape,
    ) -> Result<ResolvedRecordedBuildAuthority, ResolveRecordedAuthorityError>,
    resolve_static: emission::StaticWrapperResolver,
) -> Result<CompiledWitnessWriterPrefix, CompiledWitnessWriterPrefixError> {
    emission::emit_using(
        arena,
        preprocessed_trace_variant,
        manifest,
        target_sm,
        resolve,
        resolve_static,
    )
}
