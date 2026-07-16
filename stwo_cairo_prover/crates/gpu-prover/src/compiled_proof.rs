//! Validated, address-free semantic proof program.
//!
//! A value is an immutable semantic version, never a physical allocation.
//! Every operation names one real execution primitive and one exact, body-
//! derived effect contract. Fleet placement and storage reuse are later
//! authorities and may not invent or weaken these ranges.

use crate::proof_bundle::ResidentProofBundleLayout;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

mod effect;
mod identity;
mod validate;

pub use effect::*;
pub use identity::{CompiledProofIdentity, ProofCodecIdentity, ProofIdentity};
pub use stwo_backend_cuda::{TranscriptInputId, TranscriptOutputId};

macro_rules! numeric_id {
    ($name:ident, $raw:ty) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub $raw);

        impl $name {
            pub const fn get(self) -> $raw {
                self.0
            }
        }
    };
}

numeric_id!(OpId, u32);
numeric_id!(SemanticOpId, u32);
numeric_id!(AotKernelId, u32);
numeric_id!(ValueVersion, u32);
numeric_id!(ExternalInputId, u32);
numeric_id!(ConstantId, u32);
numeric_id!(EffectBindingId, u32);
numeric_id!(InPlaceAliasId, u32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ElementRange {
    pub start: usize,
    pub end: usize,
}

impl ElementRange {
    pub const fn new(start: usize, end: usize) -> Option<Self> {
        if start < end {
            Some(Self { start, end })
        } else {
            None
        }
    }

    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub const fn contains(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ElementType {
    pub tag: u32,
    pub bytes: usize,
}

impl ElementType {
    /// Canonical resident ABI element for proof words and `AddU32` atomics.
    pub const U32: Self = Self { tag: 1, bytes: 4 };
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LayoutAxis {
    pub tag: u16,
    pub extent: usize,
    pub stride_bytes: usize,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueLayout {
    pub element: ElementType,
    pub axes: Vec<LayoutAxis>,
}

impl ValueLayout {
    pub fn element_count(&self) -> Result<usize, CompiledProofError> {
        self.axes.iter().try_fold(1usize, |count, axis| {
            count
                .checked_mul(axis.extent)
                .ok_or(CompiledProofError::SizeOverflow)
        })
    }

    pub fn logical_bytes(&self) -> Result<usize, CompiledProofError> {
        self.element_count()?
            .checked_mul(self.element.bytes)
            .ok_or(CompiledProofError::SizeOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Region {
    FixedData,
    Input,
    Dynamic,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofStage {
    /// Work completed before the named transcript segment executes.
    BeforeTranscript(CairoTranscriptSegment),
    AfterTranscript,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchGeometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: Option<[u32; 3]>,
    pub dynamic_shared_bytes: u32,
    pub cooperative: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPrimitive {
    AotKernel {
        kernel: AotKernelId,
        launch: LaunchGeometry,
    },
    /// Device-to-device contiguous byte copy. Host ingress/egress is outside
    /// the semantic proof program and cannot be disguised as this primitive.
    DeviceCopyD2D { bytes: usize },
    /// CUDA byte-pattern memset; `value` is one repeated byte, not a word.
    DeviceMemsetByte { bytes: usize, value: u8 },
}

/// Exact value carried by one ordinal in an AOT kernel invocation.
///
/// Pointer-table order is semantic: swapping two entries can execute a valid
/// cubin over the wrong columns while leaving its coarse read/write set
/// unchanged. `None` represents an explicitly unused/dummy pointer whose
/// absence of dereferences is proven by the generator-owned program authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AotArgumentValue {
    U32(u32),
    DevicePointer(Option<EffectBindingId>),
    DevicePointerTable(Vec<Option<EffectBindingId>>),
    /// Immutable device-side u32 data installed with the executable.
    DeviceU32Literals(Vec<u32>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AotArgumentBinding {
    pub ordinal: u8,
    pub value: AotArgumentValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AotInvocation {
    pub arguments: Vec<AotArgumentBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpNode {
    pub id: OpId,
    pub semantic_id: SemanticOpId,
    pub primitive: ExecutionPrimitive,
    /// Present exactly for [`ExecutionPrimitive::AotKernel`]. Every effect
    /// binding must occur once in this ABI map; non-kernel primitives have no
    /// invocation.
    pub invocation: Option<AotInvocation>,
    pub effect: EffectContractId,
    pub stage: ProofStage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueOrigin {
    ExternalInput(ExternalInputId),
    Constant(ConstantId),
    OpOutput(OpId),
    TranscriptOutput(TranscriptOutputId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueDesc {
    pub version: ValueVersion,
    pub layout: ValueLayout,
    pub alignment: usize,
    pub origin: ValueOrigin,
    pub region: Region,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptInputBinding {
    pub id: TranscriptInputId,
    pub value: ValueVersion,
    pub elements: ElementRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptOutputBinding {
    pub id: TranscriptOutputId,
    pub value: ValueVersion,
    pub elements: ElementRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofBundleSection {
    Commitments,
    InteractionClaim,
    InteractionPow,
    SampledValues,
    FriCommitments,
    FinalLinePolynomial,
    QueryPow,
    Decommitment,
}

impl ProofBundleSection {
    pub const CANONICAL: [Self; 8] = [
        Self::Commitments,
        Self::InteractionClaim,
        Self::InteractionPow,
        Self::SampledValues,
        Self::FriCommitments,
        Self::FinalLinePolynomial,
        Self::QueryPow,
        Self::Decommitment,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofOutputSection {
    pub section: ProofBundleSection,
    pub value: ValueVersion,
    pub elements: ElementRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofOutputLayout {
    pub codec: ProofCodecIdentity,
    pub layout: ResidentProofBundleLayout,
    pub sections: Vec<ProofOutputSection>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledProofInput {
    pub identity: ProofIdentity,
    pub kernels: Vec<AotKernelAuthority>,
    pub effects: Vec<EffectContract>,
    pub operations: Vec<OpNode>,
    pub values: Vec<ValueDesc>,
    pub transcript_inputs: Vec<TranscriptInputBinding>,
    pub transcript_outputs: Vec<TranscriptOutputBinding>,
    pub output: ProofOutputLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledProof {
    input: CompiledProofInput,
    identity: CompiledProofIdentity,
}

impl CompiledProof {
    pub fn compile(
        input: CompiledProofInput,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, CompiledProofError> {
        validate::validate(&input, transcript)?;
        let identity = identity::compiled_identity(&input, transcript)?;
        Ok(Self { input, identity })
    }

    pub const fn input(&self) -> &CompiledProofInput {
        &self.input
    }

    pub const fn identity(&self) -> &CompiledProofIdentity {
        &self.identity
    }

    pub fn transcript_encoding(&self) -> &[u8] {
        self.identity.transcript_encoding()
    }

    pub const fn program_identity(&self) -> &ProofIdentity {
        &self.input.identity
    }

    pub fn kernels(&self) -> &[AotKernelAuthority] {
        &self.input.kernels
    }

    pub fn effects(&self) -> &[EffectContract] {
        &self.input.effects
    }

    pub fn operations(&self) -> &[OpNode] {
        &self.input.operations
    }

    pub fn values(&self) -> &[ValueDesc] {
        &self.input.values
    }

    pub fn transcript_inputs(&self) -> &[TranscriptInputBinding] {
        &self.input.transcript_inputs
    }

    pub fn transcript_outputs(&self) -> &[TranscriptOutputBinding] {
        &self.input.transcript_outputs
    }

    pub const fn output(&self) -> &ProofOutputLayout {
        &self.input.output
    }

    pub fn kernel(&self, id: AotKernelId) -> Option<&AotKernelAuthority> {
        self.input.kernels.iter().find(|kernel| kernel.id() == id)
    }

    pub fn effect(&self, id: EffectContractId) -> Option<&EffectContract> {
        self.input.effects.iter().find(|effect| effect.id() == id)
    }

    pub fn operation(&self, id: OpId) -> Option<&OpNode> {
        self.input
            .operations
            .get(id.0 as usize)
            .filter(|operation| operation.id == id)
    }

    pub fn effect_for(&self, operation: OpId) -> Option<&EffectContract> {
        self.operation(operation)
            .and_then(|operation| self.effect(operation.effect))
    }

    pub fn in_place_alias(
        &self,
        effect: EffectContractId,
        alias: InPlaceAliasId,
    ) -> Option<&EffectAccess> {
        self.effect(effect)?.in_place_alias(alias)
    }

    pub fn value(&self, version: ValueVersion) -> Option<&ValueDesc> {
        self.input
            .values
            .get(version.0 as usize)
            .filter(|value| value.version == version)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityKind {
    ProofSemantics,
    ExecutionBuild,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingKind {
    TranscriptInput,
    TranscriptOutput,
    ProofOutput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledProofError {
    SizeOverflow,
    EmptyIdentity(IdentityKind),
    EmptyModuleIdentity,
    EmptyKernelIdentity(AotKernelId),
    EmptyKernelEffectAuthority(AotKernelId),
    NonCanonicalKernelEffects(AotKernelId),
    NonCanonicalEffectBindings,
    NonCanonicalEffectAliases,
    InvalidEffectRange,
    InvalidValueTransition,
    InvalidModuleGlobalEffect,
    NonCanonicalModuleGlobals,
    NonCanonicalKernelAuthority,
    NonCanonicalEffectAuthority,
    DuplicateSemanticOperation(SemanticOpId),
    NonDenseOperation {
        expected: OpId,
        actual: OpId,
    },
    NonDenseValue {
        expected: ValueVersion,
        actual: ValueVersion,
    },
    UnknownStage {
        operation: OpId,
    },
    StageOrder {
        previous: OpId,
        current: OpId,
    },
    UnknownValue {
        operation: OpId,
        value: ValueVersion,
    },
    InvalidValue {
        value: ValueVersion,
    },
    UnknownProducer {
        value: ValueVersion,
        producer: OpId,
    },
    ProducerMismatch {
        value: ValueVersion,
    },
    ProducerAfterConsumer {
        value: ValueVersion,
        consumer: OpId,
    },
    OverlappingWrite {
        value: ValueVersion,
    },
    IncompleteWrite {
        value: ValueVersion,
    },
    UnknownEffect {
        operation: OpId,
    },
    InvalidEffectContract(EffectContractId),
    UnknownKernel {
        operation: OpId,
    },
    KernelEffectNotAccepted {
        operation: OpId,
    },
    ModuleGlobalAuthorityMismatch {
        operation: OpId,
    },
    InvalidLaunchGeometry(OpId),
    PrimitiveEffectMismatch(OpId),
    InvalidKernelInvocation(OpId),
    TranscriptInputCount {
        expected: usize,
        actual: usize,
    },
    TranscriptOutputCount {
        expected: usize,
        actual: usize,
    },
    TranscriptBindingOrder {
        kind: BindingKind,
        index: usize,
    },
    TranscriptBindingOrigin {
        output: TranscriptOutputId,
    },
    OrphanTranscriptOutput {
        value: ValueVersion,
    },
    TranscriptCausality {
        kind: BindingKind,
        id: u32,
    },
    BindingRange {
        kind: BindingKind,
        id: u32,
    },
    BindingOverlap {
        kind: BindingKind,
        value: ValueVersion,
    },
    NonCanonicalProofLayout,
    ProofSectionCount {
        expected: usize,
        actual: usize,
    },
    ProofSectionOrder {
        index: usize,
    },
    ProofSectionOrigin {
        index: usize,
    },
    InvalidProofAssembly,
    TranscriptPlan(crate::transcript_plan::TranscriptPlanError),
}

impl core::fmt::Display for CompiledProofError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid compiled proof: {self:?}")
    }
}

impl std::error::Error for CompiledProofError {}

impl From<crate::transcript_plan::TranscriptPlanError> for CompiledProofError {
    fn from(value: crate::transcript_plan::TranscriptPlanError) -> Self {
        Self::TranscriptPlan(value)
    }
}
