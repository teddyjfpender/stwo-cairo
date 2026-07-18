//! Validated, address-free semantic proof program.
//!
//! A value is an immutable semantic version, never a physical allocation.
//! Every operation names one real execution primitive and one exact, body-
//! derived effect contract. Fleet placement and storage reuse are later
//! authorities and may not invent or weaken these ranges.

use crate::proof_bundle::ResidentProofBundleLayout;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

mod effect;
mod finalizer;
mod identity;
mod invocation_contract;
mod partition_authority;
mod registered_fixed_source;
mod static_wrapper;
mod structural_authority;
mod validate;

pub use effect::*;
pub use finalizer::*;
#[cfg(test)]
pub(crate) use identity::module_global_initializer_structure_identity_for_test;
pub use identity::{CompiledProofIdentity, ProofCodecIdentity, ProofIdentity};
pub(crate) use invocation_contract::encode_invocation_payload;
pub use invocation_contract::InvocationContractId;
pub use partition_authority::*;
pub use registered_fixed_source::*;
pub use static_wrapper::*;
pub use structural_authority::*;
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
numeric_id!(StaticCudaWrapperId, u32);
numeric_id!(ValueVersion, u32);
numeric_id!(ExternalInputId, u32);
numeric_id!(ConstantId, u32);
numeric_id!(ModuleGlobalInitializerId, u32);
numeric_id!(EffectBindingId, u32);
numeric_id!(InPlaceAliasId, u32);
numeric_id!(TranscriptStateVersion, u32);

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionPrimitive {
    AotKernel {
        kernel: AotKernelId,
        launch: LaunchGeometry,
    },
    /// One linked ordinary-CUDA host entry. The wrapper is one semantic
    /// primitive with one outer effect; its authority binds an ordered launch
    /// identity manifest without fabricating child SSA or partition claims.
    StaticCudaWrapper { wrapper: StaticCudaWrapperId },
    /// Device-to-device contiguous byte copy. Host ingress/egress is outside
    /// the semantic proof program and cannot be disguised as this primitive.
    DeviceCopyD2D { bytes: usize },
    /// CUDA byte-pattern memset; `value` is one repeated byte, not a word.
    DeviceMemsetByte { bytes: usize, value: u8 },
    /// One transcript-free, monolithic operation whose children execute in
    /// this exact order. Children reuse the same typed execution authority as
    /// ordinary operations but do not introduce stages or partitions.
    OrderedComposite { children: Box<[ExecutableStep]> },
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
    /// Canonical address-free encoding of a host `size_t` argument. The
    /// executable performs a checked conversion to its local `usize`; using a
    /// fixed u64 here keeps proof-program identity independent of the compiler
    /// host's pointer width.
    Usize(u64),
    DevicePointer(Option<EffectBindingId>),
    DevicePointerTable(Vec<Option<EffectBindingId>>),
    /// Ordered table of pointers resolved from checked process registrations.
    /// Every entry must have one exact immutable read in the operation effect;
    /// normal value bindings and module-global relocations cannot substitute
    /// for this authority.
    DeviceRegisteredFixedSourcePointerTable(Vec<RegisteredFixedSourceRead>),
    /// Ordered heterogeneous pointer table whose ordinary entries are exact
    /// effect bindings and whose process-owned entries are exact registered
    /// immutable reads.
    DeviceMixedFixedSourcePointerTable(Vec<FixedSourcePointerEntry>),
    /// Full immutable u32 value installed with the executable. Literal bytes
    /// live only in its [`FixedValueInitializer`]; the invocation carries no
    /// second allocation or content channel.
    DeviceFixedU32 {
        value: ValueVersion,
        binding: EffectBindingId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixedSourcePointerEntry {
    EffectBinding(EffectBindingId),
    Registered(RegisteredFixedSourceRead),
}

impl AotArgumentValue {
    /// Temporary fail-closed bridge for the non-promotable source inventory.
    /// It discards the old duplicate payload and produces an unbound reference
    /// that [`CompiledProof::compile`] necessarily rejects. A real emitter must
    /// first install a `FixedValueDesc` and construct `DeviceFixedU32`.
    #[doc(hidden)]
    pub fn legacy_unbound_fixed_u32(_words: Vec<u32>) -> Self {
        Self::DeviceFixedU32 {
            value: ValueVersion(u32::MAX),
            binding: EffectBindingId(u32::MAX),
        }
    }
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

/// Exact executable authority shared by an ordinary operation and every
/// child of an [`ExecutionPrimitive::OrderedComposite`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableStep {
    pub primitive: ExecutionPrimitive,
    pub invocation: Option<AotInvocation>,
    pub effect: EffectContractId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpNode {
    pub id: OpId,
    pub semantic_id: SemanticOpId,
    pub primitive: ExecutionPrimitive,
    /// Present exactly for [`ExecutionPrimitive::AotKernel`] and
    /// [`ExecutionPrimitive::StaticCudaWrapper`]. Every effect binding must
    /// occur once in this ABI map, as must every registered fixed-source read;
    /// other primitives have no invocation.
    pub invocation: Option<AotInvocation>,
    pub effect: EffectContractId,
    pub partition: PartitionAuthorityId,
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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
pub struct ProofOutputFragment {
    pub section: ProofBundleSection,
    pub ordinal: u32,
    pub source: ValueRange,
    /// Canonical destination words in the complete resident proof bundle.
    pub destination: ElementRange,
}

/// Compatibility projection consumed by the current fleet storage lowering.
/// It is validated as an exact, one-fragment-per-section view and is not an
/// independent proof-layout authority.
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
    pub fragments: Vec<ProofOutputFragment>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledProofInput {
    pub identity: ProofIdentity,
    pub fixed_values: Vec<FixedValueDesc>,
    pub module_global_initializers: Vec<ModuleGlobalInitializer>,
    pub kernels: Vec<AotKernelAuthority>,
    pub static_wrappers: Vec<StaticCudaWrapperAuthority>,
    pub effects: Vec<EffectContract>,
    pub partitions: Vec<PartitionAuthority>,
    pub operations: Vec<OpNode>,
    pub values: Vec<ValueDesc>,
    pub transcript_inputs: Vec<TranscriptInputBinding>,
    pub transcript_outputs: Vec<TranscriptOutputBinding>,
    pub transcript_segments: Vec<CompiledTranscriptSegment>,
    pub output: ProofOutputLayout,
    pub host_finalizer: HostFinalizerAuthority,
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

    pub fn static_wrappers(&self) -> &[StaticCudaWrapperAuthority] {
        &self.input.static_wrappers
    }

    pub fn fixed_values(&self) -> &[FixedValueDesc] {
        &self.input.fixed_values
    }

    pub fn module_global_initializers(&self) -> &[ModuleGlobalInitializer] {
        &self.input.module_global_initializers
    }

    pub fn effects(&self) -> &[EffectContract] {
        &self.input.effects
    }

    pub fn partitions(&self) -> &[PartitionAuthority] {
        &self.input.partitions
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

    pub fn transcript_segments(&self) -> &[CompiledTranscriptSegment] {
        &self.input.transcript_segments
    }

    pub const fn output(&self) -> &ProofOutputLayout {
        &self.input.output
    }

    pub const fn host_finalizer(&self) -> &HostFinalizerAuthority {
        &self.input.host_finalizer
    }

    pub fn kernel(&self, id: AotKernelId) -> Option<&AotKernelAuthority> {
        self.input.kernels.iter().find(|kernel| kernel.id() == id)
    }

    pub fn static_wrapper(&self, id: StaticCudaWrapperId) -> Option<&StaticCudaWrapperAuthority> {
        self.input
            .static_wrappers
            .iter()
            .find(|wrapper| wrapper.id() == id)
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
    InvalidStaticWrapperManifest,
    InvalidStaticWrapperAuthority(StaticCudaWrapperId),
    NonCanonicalStaticWrapperAuthority,
    NonCanonicalEffectBindings,
    NonCanonicalEffectAliases,
    InvalidEffectRange,
    InvalidValueTransition,
    InvalidModuleGlobalEffect,
    InvalidFixedValue,
    NonCanonicalFixedValues,
    InvalidModuleGlobalInitializer,
    InvalidRegisteredFixedSource,
    InvalidRegisteredFixedSourceRead,
    NonCanonicalRegisteredFixedSourceReads,
    NonCanonicalModuleGlobalInitializers,
    UnknownModuleGlobalInitializer(ModuleGlobalInitializerId),
    InvalidPartitionAuthority,
    NonCanonicalPartitionAuthority,
    UnknownPartitionAuthority {
        operation: OpId,
    },
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
    KernelInvocationNotAccepted {
        operation: OpId,
    },
    UnknownStaticWrapper {
        operation: OpId,
    },
    StaticWrapperEffectNotAccepted {
        operation: OpId,
    },
    StaticWrapperInvocationNotAccepted {
        operation: OpId,
    },
    StaticWrapperRequiresMonolithic {
        operation: OpId,
    },
    ModuleGlobalAuthorityMismatch {
        operation: OpId,
    },
    InvalidLaunchGeometry(OpId),
    PrimitiveEffectMismatch(OpId),
    InvalidKernelInvocation(OpId),
    InvalidStaticWrapperInvocation(OpId),
    InvalidOrderedComposite {
        operation: OpId,
        child: Option<usize>,
    },
    CompositeBoundaryEffectMismatch(OpId),
    CompositeUninitializedRead {
        operation: OpId,
        child: usize,
        value: ValueVersion,
    },
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
    TranscriptSegmentCount {
        expected: usize,
        actual: usize,
    },
    TranscriptSegmentBinding {
        index: usize,
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
    ProofFragmentOrdinal {
        index: usize,
    },
    ProofSectionOrigin {
        index: usize,
    },
    InvalidProofAssembly,
    InvalidHostFinalizer,
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
