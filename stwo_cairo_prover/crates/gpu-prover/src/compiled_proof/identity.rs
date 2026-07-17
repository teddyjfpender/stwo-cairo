use super::{CompiledProofError, IdentityKind, *};

const PROOF_IDENTITY_TAG: &[u8] = b"stwo-cairo.compiled-proof.identity.v2";
const STRUCTURE_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.structure.v4\0";

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
        let len = u64::try_from(bytes.len()).map_err(|_| CompiledProofError::SizeOverflow)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(tag);
        hasher.update(&len.to_le_bytes());
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
            b"stwo-cairo.proof-semantics.v2",
            semantic_encoding,
        )?;
        let execution_build = CanonicalIdentity::new(
            IdentityKind::ExecutionBuild,
            b"stwo-cairo.execution-build.v2",
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

    pub const fn execution_build_digest(&self) -> &[u8; 32] {
        &self.execution_build.digest
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
    pub fn track_a_resident_bundle() -> Self {
        Self(CanonicalIdentity::from_static(
            b"stwo-cairo.proof-codec.v2",
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CompiledProofIdentity {
    canonical_encoding: Box<[u8]>,
    transcript_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl CompiledProofIdentity {
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub fn transcript_encoding(&self) -> &[u8] {
        &self.transcript_encoding
    }
}

pub(super) fn compiled_identity(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<CompiledProofIdentity, CompiledProofError> {
    let mut out = Encoder::new(STRUCTURE_DOMAIN);
    out.bytes(input.identity.semantic_encoding())?;
    out.bytes(input.identity.execution_build_encoding())?;
    out.raw(input.identity.proof_semantic_digest());
    out.raw(input.identity.program_image_digest());

    out.count(input.fixed_values.len())?;
    for fixed in &input.fixed_values {
        out.u32(fixed.constant().0);
        out.u32(fixed.value().0);
        out.raw(fixed.content_digest());
        out.fixed_initializer(fixed.initializer())?;
    }
    out.count(input.module_global_initializers.len())?;
    for initializer in &input.module_global_initializers {
        out.u32(initializer.id().0);
        out.bytes(initializer.module().canonical_encoding())?;
        out.raw(initializer.module().digest());
        out.bytes(initializer.symbol())?;
        out.size(initializer.bytes())?;
        out.size(initializer.alignment())?;
        out.byte(u8::from(initializer.immutable()));
        out.count(initializer.atoms().len())?;
        for atom in initializer.atoms() {
            out.module_initializer_atom(atom)?;
        }
        out.raw(initializer.content_or_recipe_digest());
    }

    out.count(input.kernels.len())?;
    for kernel in &input.kernels {
        out.bytes(kernel.canonical_encoding())?;
        out.raw(kernel.digest());
    }
    out.count(input.effects.len())?;
    for effect in &input.effects {
        out.bytes(effect.canonical_encoding())?;
        out.raw(effect.id().as_bytes());
    }
    out.count(input.partitions.len())?;
    for partition in &input.partitions {
        out.bytes(partition.canonical_encoding())?;
        out.raw(partition.id().as_bytes());
    }

    out.count(input.operations.len())?;
    for operation in &input.operations {
        out.u32(operation.id.0);
        out.u32(operation.semantic_id.0);
        out.raw(operation.effect.as_bytes());
        out.raw(operation.partition.as_bytes());
        out.stage(operation.stage);
        out.primitive(operation.primitive)?;
        out.invocation(operation.invocation.as_ref())?;
    }

    out.count(input.values.len())?;
    for value in &input.values {
        out.u32(value.version.0);
        out.layout(&value.layout)?;
        out.size(value.alignment)?;
        out.origin(value.origin);
        out.region(value.region);
    }

    out.count(input.transcript_inputs.len())?;
    for binding in &input.transcript_inputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.elements(binding.elements)?;
    }
    out.count(input.transcript_outputs.len())?;
    for binding in &input.transcript_outputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.elements(binding.elements)?;
    }
    out.count(input.transcript_segments.len())?;
    for binding in &input.transcript_segments {
        out.segment(binding.segment);
        out.u32(binding.entry_state.0);
        out.u32(binding.exit_state.0);
        out.value_ranges(&binding.consumed)?;
        out.value_ranges(&binding.produced)?;
    }

    out.bytes(input.output.codec.canonical_encoding())?;
    out.raw(input.output.codec.digest());
    for range in output_ranges(&input.output.layout) {
        out.range(&range)?;
    }
    out.size(input.output.layout.total_words)?;
    out.count(input.output.fragments.len())?;
    for fragment in &input.output.fragments {
        out.byte(section_tag(fragment.section));
        out.u32(fragment.ordinal);
        out.value_range(fragment.source)?;
        out.elements(fragment.destination)?;
    }

    out.finalizer(&input.host_finalizer)?;

    let transcript_encoding = transcript.canonical_encoding()?;
    out.bytes(&transcript_encoding)?;
    let canonical_encoding = out.finish();
    Ok(CompiledProofIdentity {
        digest: *blake3::hash(&canonical_encoding).as_bytes(),
        canonical_encoding: canonical_encoding.into_boxed_slice(),
        transcript_encoding: transcript_encoding.into_boxed_slice(),
    })
}

struct Encoder(Vec<u8>);

impl Encoder {
    fn new(domain: &[u8]) -> Self {
        Self(domain.to_vec())
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn size(&mut self, value: usize) -> Result<(), CompiledProofError> {
        self.raw(
            &u64::try_from(value)
                .map_err(|_| CompiledProofError::SizeOverflow)?
                .to_le_bytes(),
        );
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), CompiledProofError> {
        self.size(value)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CompiledProofError> {
        self.count(bytes.len())?;
        self.raw(bytes);
        Ok(())
    }

    fn range(&mut self, range: &std::ops::Range<usize>) -> Result<(), CompiledProofError> {
        self.size(range.start)?;
        self.size(range.end)
    }

    fn elements(&mut self, range: ElementRange) -> Result<(), CompiledProofError> {
        self.size(range.start)?;
        self.size(range.end)
    }

    fn layout(&mut self, layout: &ValueLayout) -> Result<(), CompiledProofError> {
        self.u32(layout.element.tag);
        self.size(layout.element.bytes)?;
        self.count(layout.axes.len())?;
        for axis in &layout.axes {
            self.u32(u32::from(axis.tag));
            self.size(axis.extent)?;
            self.size(axis.stride_bytes)?;
        }
        Ok(())
    }

    fn origin(&mut self, origin: ValueOrigin) {
        match origin {
            ValueOrigin::ExternalInput(id) => {
                self.byte(0);
                self.u32(id.0);
            }
            ValueOrigin::Constant(id) => {
                self.byte(1);
                self.u32(id.0);
            }
            ValueOrigin::OpOutput(id) => {
                self.byte(2);
                self.u32(id.0);
            }
            ValueOrigin::TranscriptOutput(id) => {
                self.byte(3);
                self.u32(id.0);
            }
        }
    }

    fn region(&mut self, region: Region) {
        self.byte(match region {
            Region::FixedData => 0,
            Region::Input => 1,
            Region::Dynamic => 2,
            Region::Output => 3,
        });
    }

    fn primitive(&mut self, primitive: ExecutionPrimitive) -> Result<(), CompiledProofError> {
        match primitive {
            ExecutionPrimitive::AotKernel { kernel, launch } => {
                self.byte(0);
                self.u32(kernel.0);
                self.launch(launch);
            }
            ExecutionPrimitive::DeviceCopyD2D { bytes } => {
                self.byte(1);
                self.size(bytes)?;
            }
            ExecutionPrimitive::DeviceMemsetByte { bytes, value } => {
                self.byte(2);
                self.size(bytes)?;
                self.byte(value);
            }
        }
        Ok(())
    }

    fn invocation(&mut self, invocation: Option<&AotInvocation>) -> Result<(), CompiledProofError> {
        let Some(invocation) = invocation else {
            self.byte(0);
            return Ok(());
        };
        self.byte(1);
        self.count(invocation.arguments.len())?;
        for argument in &invocation.arguments {
            self.byte(argument.ordinal);
            match &argument.value {
                AotArgumentValue::U32(value) => {
                    self.byte(0);
                    self.u32(*value);
                }
                AotArgumentValue::DevicePointer(binding) => {
                    self.byte(1);
                    self.effect_binding(*binding);
                }
                AotArgumentValue::DevicePointerTable(entries) => {
                    self.byte(2);
                    self.count(entries.len())?;
                    for &entry in entries {
                        self.effect_binding(entry);
                    }
                }
                AotArgumentValue::DeviceFixedU32 { value, binding } => {
                    self.byte(3);
                    self.u32(value.0);
                    self.u32(binding.0);
                }
            }
        }
        Ok(())
    }

    fn effect_binding(&mut self, binding: Option<EffectBindingId>) {
        match binding {
            Some(binding) => {
                self.byte(1);
                self.u32(binding.0);
            }
            None => self.byte(0),
        }
    }

    fn launch(&mut self, launch: LaunchGeometry) {
        for value in launch.grid {
            self.u32(value);
        }
        for value in launch.block {
            self.u32(value);
        }
        match launch.cluster {
            Some(cluster) => {
                self.byte(1);
                for value in cluster {
                    self.u32(value);
                }
            }
            None => self.byte(0),
        }
        self.u32(launch.dynamic_shared_bytes);
        self.byte(u8::from(launch.cooperative));
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

    fn fixed_initializer(
        &mut self,
        initializer: &FixedValueInitializer,
    ) -> Result<(), CompiledProofError> {
        match initializer {
            FixedValueInitializer::InlineBytes(bytes) => {
                self.byte(0);
                self.bytes(bytes)?;
            }
            FixedValueInitializer::InlineU32(words) => {
                self.byte(1);
                self.count(words.len())?;
                for word in words {
                    self.u32(*word);
                }
            }
            FixedValueInitializer::DeterministicRecipe(recipe) => {
                self.byte(2);
                self.bytes(recipe)?;
            }
        }
        Ok(())
    }

    fn module_initializer_atom(
        &mut self,
        atom: &ModuleGlobalInitializerAtom,
    ) -> Result<(), CompiledProofError> {
        match atom {
            ModuleGlobalInitializerAtom::Literal { destination, bytes } => {
                self.byte(0);
                self.byte_range(*destination)?;
                self.bytes(bytes)?;
            }
            ModuleGlobalInitializerAtom::FixedValueAddress {
                destination,
                value,
                source_byte_offset,
            } => {
                self.byte(1);
                self.byte_range(*destination)?;
                self.u32(value.0);
                self.size(*source_byte_offset)?;
            }
        }
        Ok(())
    }

    fn byte_range(&mut self, range: ByteRange) -> Result<(), CompiledProofError> {
        self.size(range.start)?;
        self.size(range.end)
    }

    fn value_range(&mut self, range: ValueRange) -> Result<(), CompiledProofError> {
        self.u32(range.version.0);
        self.elements(range.elements)
    }

    fn value_ranges(&mut self, ranges: &[ValueRange]) -> Result<(), CompiledProofError> {
        self.count(ranges.len())?;
        for range in ranges {
            self.value_range(*range)?;
        }
        Ok(())
    }

    fn finalizer(&mut self, finalizer: &HostFinalizerAuthority) -> Result<(), CompiledProofError> {
        self.bytes(finalizer.bundle_codec().canonical_encoding())?;
        self.raw(finalizer.bundle_codec().digest());
        let pcs = finalizer.pcs();
        self.u32(pcs.pow_bits);
        self.u32(pcs.fri_config.log_blowup_factor);
        self.u32(pcs.fri_config.log_last_layer_degree_bound);
        self.size(pcs.fri_config.n_queries)?;
        self.u32(pcs.fri_config.fold_step);
        match pcs.lifting_log_size {
            Some(log_size) => {
                self.byte(1);
                self.u32(log_size);
            }
            None => self.byte(0),
        }
        let shape = finalizer.assembly_shape();
        self.u32(shape.query_log_size);
        self.size(shape.n_queries)?;
        self.count(shape.trace_trees.len())?;
        for tree in &shape.trace_trees {
            self.u32(tree.role as u32);
            self.u32(tree.leaf_log_size);
            self.u32(tree.query_log_size);
            self.count(tree.oods_samples_per_column.len())?;
            for count in &tree.oods_samples_per_column {
                self.size(*count)?;
            }
            self.count(tree.commit_to_proof_column.len())?;
            for column in &tree.commit_to_proof_column {
                self.size(*column)?;
            }
        }
        self.count(shape.fri_trees.len())?;
        for tree in &shape.fri_trees {
            self.u32(tree.evaluation_log_size);
            self.u32(tree.cumulative_fold);
            self.u32(tree.outgoing_fold_step);
            self.u32(tree.log_rows_per_leaf);
        }
        self.finalizer_identity(
            finalizer.claim_codec().canonical_encoding(),
            finalizer.claim_codec().digest(),
        )?;
        self.finalizer_identity(
            finalizer.interaction_claim_codec().canonical_encoding(),
            finalizer.interaction_claim_codec().digest(),
        )?;
        self.finalizer_identity(
            finalizer.channel_schema().canonical_encoding(),
            finalizer.channel_schema().digest(),
        )?;
        self.finalizer_identity(
            finalizer.preprocessed_schema().canonical_encoding(),
            finalizer.preprocessed_schema().digest(),
        )?;
        self.byte(match finalizer.decoder() {
            DirectProofDecoder::ResidentBlake2sV1 => 0,
        });
        self.byte(match finalizer.oods_recipe() {
            OodsConsistencyRecipe::CairoComponentsV1 => 0,
        });
        self.byte(match finalizer.envelope() {
            CairoProofEnvelope::CairoProofV1 => 0,
        });
        self.raw(finalizer.proof_semantic_digest());
        self.raw(finalizer.execution_build_digest());
        Ok(())
    }

    fn finalizer_identity(
        &mut self,
        encoding: &[u8],
        digest: &[u8; 32],
    ) -> Result<(), CompiledProofError> {
        self.bytes(encoding)?;
        self.raw(digest);
        Ok(())
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

fn output_ranges(
    layout: &crate::proof_bundle::ResidentProofBundleLayout,
) -> [std::ops::Range<usize>; 8] {
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
