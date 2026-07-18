//! Exact address-free authority for the production Wave composition segment.
//!
//! This compiler follows [`super::PreparedCompositionGraph::launch`] in wrapper
//! order. It admits only the replacement path: direct retained evaluations,
//! no coefficient LDE fallback, one installed AOT kernel per wave, ascending
//! accumulator lifts, and the fused composition split. Runtime addresses and
//! streams remain outside every identity.

use stwo_backend_cuda::CompositionSplitLaunchMode;

use super::CompositionLaunchMode;
use crate::arena_plan::{LogicalBufferId, ProofArenaPlan};
use crate::compiled_proof::{EffectBindingId, EffectContract, ElementRange};
use crate::composition_wave::{
    CompositionWaveProgram, CompositionWaveShardAuthority, LoadedCompositionWaveShardAuthority,
};

mod compiler;
mod encoding;
mod semantic;

use compiler::Compiler;
use encoding::{linked_identity, program_identity, source_identity};

const ZERO_IDENTITY: [u8; 32] = [0; 32];

/// One immutable semantic version consumed or produced by composition.
///
/// Generations are explicit only where the real launch sequence mutates the
/// same storage in place. `Descriptor` roles name exact subranges of the
/// prepared descriptor image rather than granting the whole buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompositionValueRole {
    Descriptor {
        kind: CompositionDescriptorRole,
        index: u32,
    },
    RandomCoefficient,
    RandomCoefficientPowers,
    RelationZ,
    RelationAlphaPowers,
    ClaimedSum {
        component: u32,
    },
    ExtParam {
        component: u32,
        slot: u32,
    },
    DirectEvaluation {
        plan_column: u32,
    },
    Accumulator {
        log_size: u32,
        coordinate: u8,
        generation: u8,
    },
    SplitRetained {
        canonical_column: u8,
        generation: u8,
    },
    ForwardTwiddles,
    InverseTwiddles,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompositionDescriptorRole {
    DynamicSourceKinds,
    DynamicSourceIndices,
    DynamicScales,
    WaveParts,
    InteractionOffsets,
    DenominatorInverses,
    BaseParams,
}

/// Exact logical range occupied by one semantic role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionLayout {
    pub role: CompositionValueRole,
    pub logical: LogicalBufferId,
    pub first_word: usize,
    pub word_len: usize,
    pub alignment_words: usize,
}

/// Address-bearing storage installed during preparation.
///
/// These ranges are validated and identity-bound as physical relocation
/// metadata. They are deliberately not semantic values and therefore never
/// receive a [`ValueVersion`](crate::compiled_proof::ValueVersion) or appear
/// in an [`EffectContract`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompositionRelocationRole {
    DynamicDestinations,
    ClaimedSumPointers,
    EvaluationPointers { component: u32 },
    SplitSourcePointers,
    SplitRetainedPointers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionRelocationLayout {
    pub role: CompositionRelocationRole,
    pub logical: LogicalBufferId,
    pub first_word: usize,
    pub word_len: usize,
    pub alignment_words: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CompositionAccessKind {
    Read,
    Write,
    ReadWriteRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionValueAccess {
    pub binding: EffectBindingId,
    pub kind: CompositionAccessKind,
    pub source: Option<CompositionValueRole>,
    pub destination: Option<CompositionValueRole>,
    pub elements: ElementRange,
}

/// Role-bound effect plus the generic compiled-proof effect identity consumed
/// by the existing wave-shard authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionEffect {
    pub accesses: Vec<CompositionValueAccess>,
    pub contract: EffectContract,
    pub identity: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionAbi {
    MaterializeExtParamsV1,
    GenerateDescendingPowersV1,
    WaveV2,
    LiftAccumulateV1,
    SplitInverseFusedFirstForwardV1,
    SplitForwardAfterFirstIntervalV1,
}

impl CompositionAbi {
    pub const fn wrapper_symbol(self) -> &'static str {
        match self {
            Self::MaterializeExtParamsV1 => "stwo_composition_materialize_ext_params_on",
            Self::GenerateDescendingPowersV1 => "stwo_composition_generate_descending_powers_on",
            Self::WaveV2 => "stwo_cuda_jit_eval_composition_wave_on",
            Self::LiftAccumulateV1 => "stwo_composition_lift_accumulate_on",
            Self::SplitInverseFusedFirstForwardV1 => {
                "stwo_ntt_b2n_composition_fused_first_forward_on"
            }
            Self::SplitForwardAfterFirstIntervalV1 => {
                "stwo_ntt_n2b_columns_after_first_stage_two_interval_on"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionInvocationValue {
    Access(u32),
    PointerTable { pointee_accesses: Vec<Option<u32>> },
    U32(u32),
    ExecutionStream,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionInvocationArgument {
    pub ordinal: u8,
    pub name: &'static str,
    pub value: CompositionInvocationValue,
}

/// One address-free pointer graph reached through prepared descriptor storage.
///
/// The outer order, every null, and every leaf effect binding are semantic.
/// The address bytes and the allocation holding them are relocation metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionEmbeddedPointerTable {
    pub pointee_accesses: Vec<Option<u32>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionInvocation {
    pub arguments: Vec<CompositionInvocationArgument>,
    pub embedded_pointer_tables: Vec<CompositionEmbeddedPointerTable>,
    pub identity: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionLaunchGeometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shared_bytes: u32,
    pub cooperative: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionChildLaunch {
    pub symbol: Box<str>,
    pub launch: CompositionLaunchGeometry,
    pub parameters: Vec<(&'static str, u32)>,
    pub effect: CompositionEffect,
    pub identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionOperationKind {
    MaterializeExtParams {
        count: u32,
        alpha_power_count: u32,
        claimed_sum_count: u32,
    },
    GenerateDescendingPowers {
        count: u32,
    },
    Wave {
        wave_index: u32,
        part_count: u32,
        evaluation_log_size: u32,
        row_count: u32,
    },
    LiftAccumulate {
        lift_index: u32,
        previous_log_size: u32,
        current_log_size: u32,
    },
    SplitInverseFusedFirstForward {
        evaluation_log_size: u32,
    },
    SplitForwardAfterFirstInterval {
        evaluation_log_size: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionOperation {
    pub kind: CompositionOperationKind,
    pub abi: CompositionAbi,
    pub invocation: CompositionInvocation,
    pub children: Vec<CompositionChildLaunch>,
    pub source_identity: [u8; 32],
    pub abi_identity: [u8; 32],
    pub effect_identity: [u8; 32],
    pub execution_identity: [u8; 32],
    pub identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionExecutionAuthority {
    source_plan_key: u64,
    layouts: Vec<CompositionLayout>,
    relocations: Vec<CompositionRelocationLayout>,
    operations: Vec<CompositionOperation>,
    waves: Vec<CompositionWaveShardAuthority>,
    outputs: [CompositionValueRole; 8],
    source_identity: [u8; 32],
    identity: [u8; 32],
}

/// Target-specific binding of ordinary static CUDA and all generated wave
/// cubins. A no-CUDA local build returns `None`; it cannot fabricate a linked
/// authority from structural metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionLinkedAuthority {
    program_identity: [u8; 32],
    static_source_identity: [u8; 32],
    static_module_build_identity: [u8; 32],
    target_sm: u32,
    loaded_waves: Vec<LoadedCompositionWaveShardAuthority>,
    identity: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionAuthorityError {
    UnsupportedLaunchMode(CompositionLaunchMode),
    UnsupportedOutputMode,
    UnsupportedSplitMode(CompositionSplitLaunchMode),
    CoefficientFallback { component: usize, count: usize },
    ShapeDrift(&'static str),
    MissingLogicalRole(&'static str),
    AmbiguousLogicalRole(&'static str),
    InvalidEffect,
    InvalidInvocation,
    InvalidExecution,
    InvalidIdentity,
    InvalidTargetSm(u32),
    StaticBuildUnavailable,
    StaticBuildMismatch,
    UnsupportedTargetSm(u32),
    SizeOverflow,
    Wave(crate::composition_wave::CompositionWaveError),
    WaveAuthority(crate::composition_wave::CompositionWaveShardAuthorityError),
    Compiled(crate::compiled_proof::CompiledProofError),
}

impl core::fmt::Display for CompositionAuthorityError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "invalid composition execution authority: {self:?}"
        )
    }
}

impl std::error::Error for CompositionAuthorityError {}

impl From<crate::composition_wave::CompositionWaveError> for CompositionAuthorityError {
    fn from(value: crate::composition_wave::CompositionWaveError) -> Self {
        Self::Wave(value)
    }
}

impl From<crate::composition_wave::CompositionWaveShardAuthorityError>
    for CompositionAuthorityError
{
    fn from(value: crate::composition_wave::CompositionWaveShardAuthorityError) -> Self {
        Self::WaveAuthority(value)
    }
}

impl From<crate::compiled_proof::CompiledProofError> for CompositionAuthorityError {
    fn from(value: crate::compiled_proof::CompiledProofError) -> Self {
        Self::Compiled(value)
    }
}

impl CompositionExecutionAuthority {
    pub fn compile(plan: &ProofArenaPlan) -> Result<Self, CompositionAuthorityError> {
        let source_identity = source_identity(plan)?;
        let mut compiler = Compiler::new(plan, source_identity)?;
        compiler.compile()?;
        let (layouts, relocations, operations, waves, outputs) = compiler.finish()?;
        let identity = program_identity(
            plan.composition().plan.key(),
            source_identity,
            &layouts,
            &relocations,
            &operations,
            &waves,
            outputs,
        )?;
        Ok(Self {
            source_plan_key: plan.composition().plan.key(),
            layouts,
            relocations,
            operations,
            waves,
            outputs,
            source_identity,
            identity,
        })
    }

    pub fn validate_against(&self, plan: &ProofArenaPlan) -> Result<(), CompositionAuthorityError> {
        (Self::compile(plan)? == *self)
            .then_some(())
            .ok_or(CompositionAuthorityError::InvalidIdentity)
    }

    pub fn bind_linked(
        &self,
        plan: &ProofArenaPlan,
        target_sm: u32,
    ) -> Result<Option<CompositionLinkedAuthority>, CompositionAuthorityError> {
        self.validate_against(plan)?;
        if target_sm < 10 || target_sm % 10 > 9 {
            return Err(CompositionAuthorityError::InvalidTargetSm(target_sm));
        }
        let actual = match stwo_backend_cuda_kernels::static_cuda_module_build_identity() {
            Ok(identity) => identity,
            Err(stwo_backend_cuda_kernels::StaticCudaModuleBuildIdentityError::Unavailable(_))
                if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT =>
            {
                return Ok(None);
            }
            Err(_) => return Err(CompositionAuthorityError::StaticBuildMismatch),
        };
        let expected = stwo_backend_cuda_kernels::expected_static_cuda_module_build_identity();
        if actual == ZERO_IDENTITY || actual != expected {
            return Err(CompositionAuthorityError::StaticBuildMismatch);
        }
        if !stwo_backend_cuda_kernels::static_cuda_module_target_sms().contains(&target_sm) {
            return Err(CompositionAuthorityError::UnsupportedTargetSm(target_sm));
        }
        let loaded_waves = self
            .waves
            .iter()
            .cloned()
            .map(|wave| wave.bind_loaded(target_sm / 10, target_sm % 10))
            .collect::<Result<Vec<_>, _>>()?;
        let static_source_identity = stwo_backend_cuda_kernels::static_cuda_source_identity();
        let identity = linked_identity(
            self.identity,
            static_source_identity,
            actual,
            target_sm,
            &loaded_waves,
        )?;
        Ok(Some(CompositionLinkedAuthority {
            program_identity: self.identity,
            static_source_identity,
            static_module_build_identity: actual,
            target_sm,
            loaded_waves,
            identity,
        }))
    }

    pub const fn source_plan_key(&self) -> u64 {
        self.source_plan_key
    }

    pub fn layouts(&self) -> &[CompositionLayout] {
        &self.layouts
    }

    pub fn operations(&self) -> &[CompositionOperation] {
        &self.operations
    }

    pub fn relocations(&self) -> &[CompositionRelocationLayout] {
        &self.relocations
    }

    pub fn waves(&self) -> &[CompositionWaveShardAuthority] {
        &self.waves
    }

    pub const fn outputs(&self) -> &[CompositionValueRole; 8] {
        &self.outputs
    }

    pub const fn source_identity(&self) -> [u8; 32] {
        self.source_identity
    }

    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

impl CompositionLinkedAuthority {
    pub fn validate(
        &self,
        structural: &CompositionExecutionAuthority,
        plan: &ProofArenaPlan,
    ) -> Result<(), CompositionAuthorityError> {
        let exact = structural
            .bind_linked(plan, self.target_sm)?
            .ok_or(CompositionAuthorityError::StaticBuildUnavailable)?;
        (*self == exact)
            .then_some(())
            .ok_or(CompositionAuthorityError::StaticBuildMismatch)
    }

    pub const fn program_identity(&self) -> [u8; 32] {
        self.program_identity
    }

    pub const fn static_source_identity(&self) -> [u8; 32] {
        self.static_source_identity
    }

    pub const fn static_module_build_identity(&self) -> [u8; 32] {
        self.static_module_build_identity
    }

    pub const fn target_sm(&self) -> u32 {
        self.target_sm
    }

    pub fn loaded_waves(&self) -> &[LoadedCompositionWaveShardAuthority] {
        &self.loaded_waves
    }

    pub const fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

#[cfg(test)]
#[path = "execution_authority/tests.rs"]
mod tests;
