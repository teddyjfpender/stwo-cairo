//! Validated, address-free semantic proof program.
//!
//! This is the authority consumed before fleet placement. It binds the exact
//! operation/value DAG, transcript bindings, AOT/effect authorities and
//! canonical proof output without assigning storage, addresses or workers.

use core::ops::Range;

use crate::proof_bundle::ResidentProofBundleLayout;
use crate::transcript_plan::{CairoBlake2sTranscriptPlan, CairoTranscriptSegment};

mod identity;
mod validate;

pub use identity::{CompiledProofIdentity, ProofCodecIdentity, ProofIdentity};
pub use stwo_backend_cuda::{TranscriptInputId, TranscriptOutputId};

pub use crate::fleet_plan::{ElementType, LayoutAxis, ValueLayout};

macro_rules! numeric_id {
    ($name:ident, $raw:ty) => {
        #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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
numeric_id!(ValueId, u32);
numeric_id!(ExternalInputId, u32);
numeric_id!(ConstantId, u32);

/// Opaque content identity of one externally audited kernel effect contract.
/// The contract body belongs to the sealed AOT authority; this digest is the
/// exact token shared with storage alias validation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct EffectContractId(pub [u8; 32]);

impl EffectContractId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact externally supplied semantic/AOT/effect authority used by this proof.
/// Lists are canonical sorted sets; the validator never derives an authority
/// identifier from caller-declared edges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticAuthority {
    pub operations: Vec<SemanticOpId>,
    pub kernels: Vec<AotKernelId>,
    pub effects: Vec<EffectContractId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofStage {
    /// Work whose completion happens-before the named segment's transcript
    /// operations. A root produced here may therefore be absorbed by that
    /// segment; a challenge drawn by it is available only to later stages.
    BeforeTranscript(CairoTranscriptSegment),
    AfterTranscript,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpNode {
    pub id: OpId,
    pub semantic_id: SemanticOpId,
    pub kernel_id: AotKernelId,
    pub effects: EffectContractId,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub stage: ProofStage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueOrigin {
    ExternalInput(ExternalInputId),
    Constant(ConstantId),
    OpOutput(OpId),
    /// Challenge/query words produced by the canonical transcript schedule,
    /// not by an AOT semantic operation.
    TranscriptOutput(TranscriptOutputId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueDesc {
    pub id: ValueId,
    pub layout: ValueLayout,
    pub alignment: usize,
    pub origin: ValueOrigin,
    /// Canonical ascending operation IDs. The validator checks this is the
    /// exact reciprocal set of operation input edges.
    pub consumers: Vec<OpId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptInputBinding {
    pub id: TranscriptInputId,
    pub value: ValueId,
    pub value_words: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptOutputBinding {
    pub id: TranscriptOutputId,
    pub value: ValueId,
    pub value_words: Range<usize>,
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

/// Source range for one exact canonical bundle section. Destination ranges are
/// supplied by `ResidentProofBundleLayout` and must cover `0..total_words`
/// exactly once in `ProofBundleSection::CANONICAL` order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofOutputSection {
    pub section: ProofBundleSection,
    pub value: ValueId,
    pub value_words: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofOutputLayout {
    pub codec: ProofCodecIdentity,
    pub layout: ResidentProofBundleLayout,
    pub sections: Vec<ProofOutputSection>,
}

/// Complete input to the all-or-nothing semantic validation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledProofInput {
    pub identity: ProofIdentity,
    pub authority: SemanticAuthority,
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

    pub const fn program_identity(&self) -> &ProofIdentity {
        &self.input.identity
    }

    pub fn operations(&self) -> &[OpNode] {
        &self.input.operations
    }

    pub fn values(&self) -> &[ValueDesc] {
        &self.input.values
    }

    pub const fn output(&self) -> &ProofOutputLayout {
        &self.input.output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityKind {
    ProofSemantics,
    ExecutionBuild,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorityKind {
    SemanticOperation,
    AotKernel,
    EffectContract,
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
    NonCanonicalAuthority(AuthorityKind),
    AuthorityMismatch(AuthorityKind),
    NonDenseOperation { expected: OpId, actual: OpId },
    NonDenseValue { expected: ValueId, actual: ValueId },
    UnknownStage { operation: OpId },
    StageOrder { previous: OpId, current: OpId },
    DuplicateOperationEdge { operation: OpId, value: ValueId },
    InputOutputAlias { operation: OpId, value: ValueId },
    InvalidValue { value: ValueId },
    UnknownValue { operation: OpId, value: ValueId },
    UnknownProducer { value: ValueId, producer: OpId },
    ProducerMismatch { value: ValueId },
    ProducerAfterConsumer { value: ValueId, consumer: OpId },
    ConsumerOrder { value: ValueId },
    ConsumerMismatch { value: ValueId },
    TranscriptInputCount { expected: usize, actual: usize },
    TranscriptOutputCount { expected: usize, actual: usize },
    TranscriptBindingOrder { kind: BindingKind, index: usize },
    TranscriptBindingOrigin { output: TranscriptOutputId },
    OrphanTranscriptOutput { value: ValueId },
    TranscriptCausality { kind: BindingKind, id: u32 },
    BindingRange { kind: BindingKind, id: u32 },
    BindingOverlap { kind: BindingKind, value: ValueId },
    NonCanonicalProofLayout,
    ProofSectionCount { expected: usize, actual: usize },
    ProofSectionOrder { index: usize },
    ProofSectionOrigin { index: usize },
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
