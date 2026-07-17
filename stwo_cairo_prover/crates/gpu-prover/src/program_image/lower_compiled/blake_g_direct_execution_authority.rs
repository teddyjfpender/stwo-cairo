//! Linked-static and prepared-runtime authority for direct Blake-G.
//!
//! The backend source contract is address-free. This boundary separately
//! proves the linked ordinary-CUDA archive, active SM, exact direct writer,
//! canonical fused feed, and their arena bindings.

use std::collections::BTreeSet;
#[cfg(test)]
use std::sync::Arc;

use stwo_backend_cuda::{
    ArenaSlice, BlakeGDirectCompositeContract, BlakeGDirectLutContentIdentity, DeviceArena,
    PreparedBlakeGFusedFeed, PreparedWitnessGraph,
};
#[cfg(test)]
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
#[cfg(test)]
use stwo_cairo_prover::witness::device_feed::canonical_count_lut;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::blake_g_direct_prefix::{
    intervals_are_disjoint, validate_lowered, BlakeGDirectAtomicBinding, BlakeGDirectRangeBinding,
    LoweredNativeBlakeGDirectContract, StaticBlakeGDirectInvocation,
};
use super::{ArenaCatalogRange, InvocationShapeError};
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::EffectContractId;

const LINKED_DOMAIN: &[u8] = b"stwo-cairo-native-blake-g-direct-linked-v1\0";
const INVOCATION_DOMAIN: &[u8] = b"stwo-cairo-native-blake-g-direct-invocation-v1\0";
const EXECUTION_DOMAIN: &[u8] = b"stwo-cairo-native-blake-g-direct-execution-v1\0";
const ZERO_IDENTITY: [u8; 32] = [0; 32];
pub(super) const CANONICAL_BLAKE_G_DIRECT_LUT_CONTENT_ID_V1: [u8; 32] = [
    91, 162, 153, 247, 151, 228, 202, 214, 255, 222, 8, 137, 83, 98, 154, 124, 7, 89, 37, 212, 11,
    186, 164, 71, 121, 150, 254, 29, 48, 27, 199, 218,
];

pub(super) fn canonical_lut_content_is_exact(identity: &BlakeGDirectLutContentIdentity) -> bool {
    identity.identity() == CANONICAL_BLAKE_G_DIRECT_LUT_CONTENT_ID_V1
}

/// Borrowed runtime objects offered to the loaded-authority gate. Construction
/// alone grants nothing; `bind_prepared` revalidates every field.
#[derive(Clone, Copy)]
pub(crate) struct PreparedBlakeGDirectKernel<'prepared, 'arena> {
    pub(crate) component: &'static str,
    pub(crate) part: TracePartId,
    pub(crate) arena: &'arena DeviceArena,
    pub(crate) writer: &'prepared PreparedWitnessGraph<'arena>,
    pub(crate) feed: &'prepared PreparedBlakeGFusedFeed<'arena>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeBlakeGDirectLinkedModuleAuthority {
    pub(super) static_module_build_identity: [u8; 32],
    pub(super) expected_static_module_build_identity: [u8; 32],
    pub(super) consumer_target_sm: u32,
    pub(super) contract: BlakeGDirectCompositeContract,
    pub(super) identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeBlakeGDirectExecutionAuthority {
    pub(super) linked: NativeBlakeGDirectLinkedModuleAuthority,
    pub(super) invocation_identity: [u8; 32],
    pub(super) compiled_effect_identity: EffectContractId,
    pub(super) lut_content_identity: BlakeGDirectLutContentIdentity,
    pub(super) identity: [u8; 32],
}

#[cfg(test)]
pub(super) fn generated_canonical_lut_content_identity(
    variant: PreProcessedTraceVariant,
) -> Result<BlakeGDirectLutContentIdentity, InvocationShapeError> {
    const FAMILIES: [&str; 4] = [
        "verify_bitwise_xor_8_state",
        "verify_bitwise_xor_4_state",
        "verify_bitwise_xor_7_state",
        "verify_bitwise_xor_9_state",
    ];
    let trace = Arc::new(variant.to_preprocessed_trace());
    let luts = FAMILIES
        .map(|family| canonical_count_lut(family, Arc::clone(&trace)))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?;
    BlakeGDirectLutContentIdentity::from_host_words([&luts[0], &luts[1], &luts[2], &luts[3]])
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)
}

impl NativeBlakeGDirectLinkedModuleAuthority {
    pub(super) fn bind_linked(
        contract: &BlakeGDirectCompositeContract,
        active_sm: u32,
    ) -> Result<Option<Self>, InvocationShapeError> {
        let linked =
            match stwo_backend_cuda_kernels::static_cuda_module_build_identity() {
                Ok(identity) => identity,
                Err(
                    stwo_backend_cuda_kernels::StaticCudaModuleBuildIdentityError::Unavailable(_),
                ) if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT => return Ok(None),
                Err(_) => return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority),
            };
        let expected = stwo_backend_cuda_kernels::expected_static_cuda_module_build_identity();
        let [consumer_target_sm] = stwo_backend_cuda_kernels::static_cuda_module_target_sms()
        else {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
        };
        Self::derive_exact(contract, linked, expected, *consumer_target_sm, active_sm).map(Some)
    }

    #[cfg(test)]
    pub(super) fn bind_exact(
        contract: &BlakeGDirectCompositeContract,
        linked: [u8; 32],
        expected: [u8; 32],
        consumer_target_sm: u32,
        active_sm: u32,
    ) -> Result<Self, InvocationShapeError> {
        Self::derive_exact(contract, linked, expected, consumer_target_sm, active_sm)
    }

    fn derive_exact(
        contract: &BlakeGDirectCompositeContract,
        linked: [u8; 32],
        expected: [u8; 32],
        consumer_target_sm: u32,
        active_sm: u32,
    ) -> Result<Self, InvocationShapeError> {
        if linked == ZERO_IDENTITY
            || linked != expected
            || consumer_target_sm == 0
            || active_sm != consumer_target_sm
            || [
                contract.source_identity(),
                contract.abi_identity(),
                contract.effect_identity(),
                contract.launch_identity(),
                contract.identity(),
            ]
            .contains(&ZERO_IDENTITY)
        {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
        }
        let mut authority = Self {
            static_module_build_identity: linked,
            expected_static_module_build_identity: expected,
            consumer_target_sm,
            contract: *contract,
            identity: ZERO_IDENTITY,
        };
        authority.identity = linked_identity(&authority);
        Ok(authority)
    }

    pub(super) fn bind_prepared(
        self,
        contract: &BlakeGDirectCompositeContract,
        lowered: &LoweredNativeBlakeGDirectContract,
        plan: &ProofArenaPlan,
        prepared: PreparedBlakeGDirectKernel<'_, '_>,
    ) -> Result<NativeBlakeGDirectExecutionAuthority, InvocationShapeError> {
        if self.contract != *contract
            || lowered.authority != *contract
            || prepared.component != "blake_g"
            || prepared.part != TracePartId::Main
        {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
        }
        validate_lowered(contract, &lowered.invocation, &lowered.effect)?;
        validate_prepared(plan, contract, &lowered.invocation, prepared)?;
        let invocation_identity = invocation_identity(&lowered.invocation)?;
        let compiled_effect_identity = lowered.effect.id();
        let lut_content_identity = *prepared.feed.lut_content_identity();
        let identity = paired_identity(
            &self,
            invocation_identity,
            compiled_effect_identity,
            lut_content_identity,
        );
        Ok(NativeBlakeGDirectExecutionAuthority {
            linked: self,
            invocation_identity,
            compiled_effect_identity,
            lut_content_identity,
            identity,
        })
    }
}

fn validate_prepared(
    plan: &ProofArenaPlan,
    contract: &BlakeGDirectCompositeContract,
    invocation: &StaticBlakeGDirectInvocation,
    prepared: PreparedBlakeGDirectKernel<'_, '_>,
) -> Result<(), InvocationShapeError> {
    let writer = prepared.writer;
    let inputs = writer.input_columns();
    let traces = writer.output_columns();
    let luts = prepared.feed.luts();
    let counts = prepared.feed.counts();
    if !writer.is_blake_g_direct()
        || !writer.belongs_to(prepared.arena)
        || writer.blake_g_direct_contract() != Some(contract)
        || writer.row_count() != contract.padded_rows()
        || !writer.multiplicity_columns().is_empty()
        || !prepared.feed.belongs_to(prepared.arena)
        || !canonical_lut_content_is_exact(prepared.feed.lut_content_identity())
        || inputs.len() != invocation.inputs.len()
        || traces.len() != invocation.traces.len()
        || !slices_match(plan, prepared.arena, &invocation.inputs, inputs)
        || !slices_match(plan, prepared.arena, &invocation.traces, traces)
        || !slices_match(plan, prepared.arena, &invocation.luts, &luts)
        || !atomic_slices_match(plan, prepared.arena, &invocation.counts, &counts)
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    let mut physical = BTreeSet::new();
    let mut intervals = Vec::new();
    if inputs
        .iter()
        .chain(traces)
        .chain(&luts)
        .chain(&counts)
        .any(|slice| {
            let start = slice.as_u32_ptr() as usize;
            let Some(end) = start.checked_add(slice.len_bytes()) else {
                return true;
            };
            intervals.push((start, end));
            !physical.insert(slice.id())
        })
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    if !intervals_are_disjoint(&mut intervals) {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    Ok(())
}

fn slices_match(
    plan: &ProofArenaPlan,
    arena: &DeviceArena,
    bindings: &[BlakeGDirectRangeBinding],
    slices: &[ArenaSlice],
) -> bool {
    bindings.len() == slices.len()
        && bindings
            .iter()
            .zip(slices)
            .all(|(binding, &slice)| slice_matches(plan, arena, &binding.value, slice))
}

fn atomic_slices_match(
    plan: &ProofArenaPlan,
    arena: &DeviceArena,
    bindings: &[BlakeGDirectAtomicBinding],
    slices: &[ArenaSlice],
) -> bool {
    bindings.len() == slices.len()
        && bindings
            .iter()
            .zip(slices)
            .all(|(binding, &slice)| slice_matches(plan, arena, &binding.value, slice))
}

fn slice_matches(
    plan: &ProofArenaPlan,
    arena: &DeviceArena,
    range: &ArenaCatalogRange,
    slice: ArenaSlice,
) -> bool {
    let Some(logical) = plan.logical_buffers().get(range.value.0 as usize) else {
        return false;
    };
    let Some(binding) = plan.binding(logical.id) else {
        return false;
    };
    let Ok(expected) = arena.bind(binding.physical) else {
        return false;
    };
    logical.id.0 == range.value.0
        && range.value_words.start == 0
        && range.value_words.end == logical.len_words
        && binding.logical == logical.id
        && binding.len_words == logical.len_words
        && slice.id() == binding.physical
        && slice.as_u32_ptr() == expected.as_u32_ptr()
        && slice.len_words() == logical.len_words
        && slice.belongs_to(arena.context())
}

fn linked_identity(authority: &NativeBlakeGDirectLinkedModuleAuthority) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINKED_DOMAIN);
    hasher.update(&authority.static_module_build_identity);
    hasher.update(&authority.expected_static_module_build_identity);
    hasher.update(&authority.consumer_target_sm.to_le_bytes());
    hasher.update(&authority.contract.identity());
    *hasher.finalize().as_bytes()
}

fn invocation_identity(
    invocation: &StaticBlakeGDirectInvocation,
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INVOCATION_DOMAIN);
    for binding in invocation
        .inputs
        .iter()
        .chain(&invocation.traces)
        .chain(&invocation.luts)
    {
        hash_range(&mut hasher, &binding.value)?;
        hasher.update(&binding.binding.0.to_le_bytes());
        hasher.update(&binding.version.0.to_le_bytes());
    }
    hasher.update(&invocation.n_real_rows.to_le_bytes());
    hasher.update(&invocation.padded_rows.to_le_bytes());
    for binding in &invocation.counts {
        hash_range(&mut hasher, &binding.value)?;
        hasher.update(&binding.binding.0.to_le_bytes());
        hasher.update(&binding.source.0.to_le_bytes());
        hasher.update(&binding.destination.0.to_le_bytes());
        hasher.update(&binding.alias.id.0.to_le_bytes());
        hasher.update(&[
            binding.alias.requirement as u8,
            binding.alias.discipline as u8,
        ]);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_range(
    hasher: &mut blake3::Hasher,
    range: &ArenaCatalogRange,
) -> Result<(), InvocationShapeError> {
    hasher.update(&range.value.0.to_le_bytes());
    for extent in [range.value_words.start, range.value_words.end] {
        hasher.update(
            &u64::try_from(extent)
                .map_err(|_| InvocationShapeError::SizeOverflow)?
                .to_le_bytes(),
        );
    }
    Ok(())
}

pub(super) fn paired_identity(
    linked: &NativeBlakeGDirectLinkedModuleAuthority,
    invocation: [u8; 32],
    effect: EffectContractId,
    lut_content_identity: BlakeGDirectLutContentIdentity,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EXECUTION_DOMAIN);
    hasher.update(&linked.identity);
    hasher.update(&invocation);
    hasher.update(effect.as_bytes());
    hasher.update(&lut_content_identity.identity());
    *hasher.finalize().as_bytes()
}
