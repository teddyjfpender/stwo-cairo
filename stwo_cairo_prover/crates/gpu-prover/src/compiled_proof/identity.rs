use super::{CompiledProofError, IdentityKind, *};

const PROOF_IDENTITY_TAG: &[u8] = b"stwo-cairo.compiled-proof.identity.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
struct CanonicalIdentity {
    encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl CanonicalIdentity {
    fn new(kind: IdentityKind, tag: &[u8], bytes: Vec<u8>) -> Result<Self, CompiledProofError> {
        if bytes.is_empty() {
            return Err(CompiledProofError::EmptyIdentity(kind));
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(tag);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
        Ok(Self {
            encoding: bytes.into_boxed_slice(),
            digest: *hasher.finalize().as_bytes(),
        })
    }

    fn from_static(tag: &[u8], bytes: &'static [u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(tag);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        Self {
            encoding: bytes.into(),
            digest: *hasher.finalize().as_bytes(),
        }
    }
}

/// Separate semantic and execution-build identities. The program-image digest
/// is domain-separated and explicitly includes the semantic digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofIdentity {
    semantics: CanonicalIdentity,
    execution_build: CanonicalIdentity,
    program_image_digest: [u8; 32],
}

impl ProofIdentity {
    pub fn new(
        semantic_encoding: Vec<u8>,
        execution_build_encoding: Vec<u8>,
    ) -> Result<Self, CompiledProofError> {
        let semantics = CanonicalIdentity::new(
            IdentityKind::ProofSemantics,
            b"stwo-cairo.proof-semantics.v1",
            semantic_encoding,
        )?;
        let execution_build = CanonicalIdentity::new(
            IdentityKind::ExecutionBuild,
            b"stwo-cairo.execution-build.v1",
            execution_build_encoding,
        )?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROOF_IDENTITY_TAG);
        hasher.update(&semantics.digest);
        hasher.update(&execution_build.digest);
        Ok(Self {
            semantics,
            execution_build,
            program_image_digest: *hasher.finalize().as_bytes(),
        })
    }

    pub const fn proof_semantic_digest(&self) -> &[u8; 32] {
        &self.semantics.digest
    }

    pub const fn program_image_digest(&self) -> &[u8; 32] {
        &self.program_image_digest
    }

    pub fn semantic_encoding(&self) -> &[u8] {
        &self.semantics.encoding
    }

    pub fn execution_build_encoding(&self) -> &[u8] {
        &self.execution_build.encoding
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofCodecIdentity(CanonicalIdentity);

impl ProofCodecIdentity {
    /// The only Track-A codec admitted by this module. Protocol-shape and
    /// decommitment-size agreement are supplied by the real ShapeExecutable
    /// emitter; this identity fixes the section ABI and word encoding.
    pub fn track_a_resident_bundle() -> Self {
        Self(CanonicalIdentity::from_static(
            b"stwo-cairo.proof-codec.v1",
            b"stwo-cairo.resident-proof-bundle.u32-le.v1",
        ))
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.0.digest
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.0.encoding
    }
}

/// Full structural identity of one validated compiled proof. It has no public
/// constructor: only `CompiledProof::compile` can derive it from the complete
/// canonical DAG, transcript and proof-output authority.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CompiledProofIdentity {
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl CompiledProofIdentity {
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }
}

pub(super) fn compiled_identity(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<CompiledProofIdentity, CompiledProofError> {
    let mut out = Encoder::new(b"stwo-cairo.compiled-proof.structure.v1");
    out.bytes(input.identity.semantic_encoding())?;
    out.bytes(input.identity.execution_build_encoding())?;
    out.raw(input.identity.proof_semantic_digest());
    out.raw(input.identity.program_image_digest());

    out.count(input.authority.operations.len())?;
    for id in &input.authority.operations {
        out.u32(id.0);
    }
    out.count(input.authority.kernels.len())?;
    for id in &input.authority.kernels {
        out.u32(id.0);
    }
    out.count(input.authority.effects.len())?;
    for id in &input.authority.effects {
        out.raw(&id.0);
    }

    out.count(input.operations.len())?;
    for operation in &input.operations {
        out.u32(operation.id.0);
        out.u32(operation.semantic_id.0);
        out.u32(operation.kernel_id.0);
        out.raw(&operation.effects.0);
        out.stage(operation.stage);
        out.ids(&operation.inputs)?;
        out.ids(&operation.outputs)?;
    }

    out.count(input.values.len())?;
    for value in &input.values {
        out.u32(value.id.0);
        out.u32(value.layout.element.tag);
        out.usize(value.layout.element.bytes)?;
        out.count(value.layout.axes.len())?;
        for axis in &value.layout.axes {
            out.u32(u32::from(axis.tag));
            out.usize(axis.extent)?;
            out.usize(axis.stride_bytes)?;
        }
        out.usize(value.alignment)?;
        match value.origin {
            ValueOrigin::ExternalInput(id) => {
                out.byte(0);
                out.u32(id.0);
            }
            ValueOrigin::Constant(id) => {
                out.byte(1);
                out.u32(id.0);
            }
            ValueOrigin::OpOutput(id) => {
                out.byte(2);
                out.u32(id.0);
            }
            ValueOrigin::TranscriptOutput(id) => {
                out.byte(3);
                out.u32(id.0);
            }
        }
        out.count(value.consumers.len())?;
        for consumer in &value.consumers {
            out.u32(consumer.0);
        }
    }

    out.count(input.transcript_inputs.len())?;
    for binding in &input.transcript_inputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.range(&binding.value_words)?;
    }
    out.count(input.transcript_outputs.len())?;
    for binding in &input.transcript_outputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.range(&binding.value_words)?;
    }

    out.bytes(input.output.codec.canonical_encoding())?;
    out.raw(input.output.codec.digest());
    for range in output_ranges(&input.output.layout) {
        out.range(&range)?;
    }
    out.usize(input.output.layout.total_words)?;
    out.count(input.output.sections.len())?;
    for section in &input.output.sections {
        out.byte(section_tag(section.section));
        out.u32(section.value.0);
        out.range(&section.value_words)?;
    }

    encode_transcript(&mut out, transcript)?;
    let canonical_encoding = out.finish();
    Ok(CompiledProofIdentity {
        digest: *blake3::hash(&canonical_encoding).as_bytes(),
        canonical_encoding: canonical_encoding.into_boxed_slice(),
    })
}

fn encode_transcript(
    out: &mut Encoder,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    let canonical = transcript.canonical_encoding()?;
    out.count(canonical.len())?;
    out.raw(&canonical);
    Ok(())
}

struct Encoder(Vec<u8>);

impl Encoder {
    fn new(tag: &[u8]) -> Self {
        Self(tag.to_vec())
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<(), CompiledProofError> {
        let value = u64::try_from(value).map_err(|_| CompiledProofError::SizeOverflow)?;
        self.raw(&value.to_le_bytes());
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), CompiledProofError> {
        self.usize(value)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CompiledProofError> {
        self.count(bytes.len())?;
        self.raw(bytes);
        Ok(())
    }

    fn range(&mut self, range: &core::ops::Range<usize>) -> Result<(), CompiledProofError> {
        self.usize(range.start)?;
        self.usize(range.end)
    }

    fn ids(&mut self, ids: &[ValueId]) -> Result<(), CompiledProofError> {
        self.count(ids.len())?;
        for id in ids {
            self.u32(id.0);
        }
        Ok(())
    }

    fn stage(&mut self, stage: ProofStage) {
        match stage {
            ProofStage::BeforeTranscript(segment) => {
                self.byte(0);
                self.segment(segment);
            }
            ProofStage::AfterTranscript => self.byte(1),
        }
    }

    fn segment(&mut self, segment: CairoTranscriptSegment) {
        let (tag, index) = match segment {
            CairoTranscriptSegment::BootstrapThroughBase => (0, 0),
            CairoTranscriptSegment::InteractionPowAndLookup => (1, 0),
            CairoTranscriptSegment::InteractionAndComposition => (2, 0),
            CairoTranscriptSegment::CompositionAndOods => (3, 0),
            CairoTranscriptSegment::OodsAndQuotient => (4, 0),
            CairoTranscriptSegment::FriLayer(index) => (5, index),
            CairoTranscriptSegment::FriLastLayer => (6, 0),
            CairoTranscriptSegment::QueryPowAndPositions => (7, 0),
        };
        self.byte(tag);
        self.u32(index);
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }
}

fn section_tag(section: ProofBundleSection) -> u8 {
    match section {
        ProofBundleSection::Commitments => 0,
        ProofBundleSection::InteractionClaim => 1,
        ProofBundleSection::InteractionPow => 2,
        ProofBundleSection::SampledValues => 3,
        ProofBundleSection::FriCommitments => 4,
        ProofBundleSection::FinalLinePolynomial => 5,
        ProofBundleSection::QueryPow => 6,
        ProofBundleSection::Decommitment => 7,
    }
}

fn output_ranges(layout: &ResidentProofBundleLayout) -> [core::ops::Range<usize>; 8] {
    [
        layout.commitments.clone(),
        layout.interaction_claim.clone(),
        layout.interaction_pow.clone(),
        layout.sampled_values.clone(),
        layout.fri_commitments.clone(),
        layout.final_line_poly.clone(),
        layout.query_pow.clone(),
        layout.decommitment.clone(),
    ]
}
