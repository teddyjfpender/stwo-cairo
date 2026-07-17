use stwo_backend_cuda::TranscriptOperation;

use super::*;

const FIXED_CONTENT_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.fixed-content.v1\0";
const MODULE_INITIALIZER_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.module-initializer.v1\0";
const PARTITION_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.partition.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixedValueInitializer {
    InlineBytes(Box<[u8]>),
    InlineU32(Box<[u32]>),
    DeterministicRecipe(Box<[u8]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedValueDesc {
    constant: ConstantId,
    value: ValueVersion,
    content_digest: [u8; 32],
    initializer: FixedValueInitializer,
}

impl FixedValueDesc {
    pub fn inline_bytes(constant: ConstantId, value: ValueVersion, bytes: Vec<u8>) -> Self {
        Self {
            constant,
            value,
            content_digest: fixed_content_digest(&bytes),
            initializer: FixedValueInitializer::InlineBytes(bytes.into_boxed_slice()),
        }
    }

    pub fn inline_u32(constant: ConstantId, value: ValueVersion, words: Vec<u32>) -> Self {
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        Self {
            constant,
            value,
            content_digest: fixed_content_digest(&bytes),
            initializer: FixedValueInitializer::InlineU32(words.into_boxed_slice()),
        }
    }

    pub fn deterministic_recipe(
        constant: ConstantId,
        value: ValueVersion,
        content_digest: [u8; 32],
        recipe: Vec<u8>,
    ) -> Result<Self, CompiledProofError> {
        if recipe.is_empty() {
            return Err(CompiledProofError::InvalidFixedValue);
        }
        Ok(Self {
            constant,
            value,
            content_digest,
            initializer: FixedValueInitializer::DeterministicRecipe(recipe.into_boxed_slice()),
        })
    }

    pub const fn constant(&self) -> ConstantId {
        self.constant
    }

    pub const fn value(&self) -> ValueVersion {
        self.value
    }

    pub const fn content_digest(&self) -> &[u8; 32] {
        &self.content_digest
    }

    pub const fn initializer(&self) -> &FixedValueInitializer {
        &self.initializer
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModuleGlobalInitializerAtom {
    Literal {
        destination: ByteRange,
        bytes: Box<[u8]>,
    },
    /// Cold-install relocation of the current fixed-arena address. No process
    /// address is serialized into this authority.
    FixedValueAddress {
        destination: ByteRange,
        value: ValueVersion,
        source_byte_offset: usize,
    },
}

impl ModuleGlobalInitializerAtom {
    pub const fn destination(&self) -> ByteRange {
        match self {
            Self::Literal { destination, .. } | Self::FixedValueAddress { destination, .. } => {
                *destination
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleGlobalInitializer {
    id: ModuleGlobalInitializerId,
    module: ModuleIdentity,
    symbol: Box<[u8]>,
    bytes: usize,
    alignment: usize,
    immutable: bool,
    atoms: Box<[ModuleGlobalInitializerAtom]>,
    content_or_recipe_digest: [u8; 32],
}

impl ModuleGlobalInitializer {
    pub fn new(
        id: ModuleGlobalInitializerId,
        module: ModuleIdentity,
        symbol: Vec<u8>,
        bytes: usize,
        alignment: usize,
        immutable: bool,
        atoms: Vec<ModuleGlobalInitializerAtom>,
    ) -> Result<Self, CompiledProofError> {
        if symbol.is_empty()
            || symbol.contains(&0)
            || bytes == 0
            || alignment == 0
            || !alignment.is_power_of_two()
            || !immutable
        {
            return Err(CompiledProofError::InvalidModuleGlobalInitializer);
        }
        let mut cursor = 0;
        for atom in &atoms {
            let destination = atom.destination();
            if destination.start != cursor
                || destination.end > bytes
                || match atom {
                    ModuleGlobalInitializerAtom::Literal { destination, bytes } => {
                        destination.len() != bytes.len()
                    }
                    ModuleGlobalInitializerAtom::FixedValueAddress { destination, .. } => {
                        destination.len() != core::mem::size_of::<u64>()
                            || destination.start % core::mem::align_of::<u64>() != 0
                            || alignment < core::mem::align_of::<u64>()
                    }
                }
            {
                return Err(CompiledProofError::InvalidModuleGlobalInitializer);
            }
            cursor = destination.end;
        }
        if cursor != bytes {
            return Err(CompiledProofError::InvalidModuleGlobalInitializer);
        }
        let content_or_recipe_digest =
            initializer_digest(id, &module, &symbol, bytes, alignment, immutable, &atoms)?;
        Ok(Self {
            id,
            module,
            symbol: symbol.into_boxed_slice(),
            bytes,
            alignment,
            immutable,
            atoms: atoms.into_boxed_slice(),
            content_or_recipe_digest,
        })
    }

    pub const fn id(&self) -> ModuleGlobalInitializerId {
        self.id
    }

    pub const fn module(&self) -> &ModuleIdentity {
        &self.module
    }

    pub fn symbol(&self) -> &[u8] {
        &self.symbol
    }

    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn immutable(&self) -> bool {
        self.immutable
    }

    pub fn atoms(&self) -> &[ModuleGlobalInitializerAtom] {
        &self.atoms
    }

    pub const fn content_or_recipe_digest(&self) -> &[u8; 32] {
        &self.content_or_recipe_digest
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        Ok(self.content_or_recipe_digest
            == initializer_digest(
                self.id,
                &self.module,
                &self.symbol,
                self.bytes,
                self.alignment,
                self.immutable,
                &self.atoms,
            )?)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PartitionAuthorityId([u8; 32]);

impl PartitionAuthorityId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PartitionAuthorityKind {
    /// v4 is deliberately fail-closed. Typed, executable shard projections
    /// arrive with the v5 fleet compiler; opaque byte recipes are not proof.
    Monolithic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionAuthority {
    id: PartitionAuthorityId,
    kind: PartitionAuthorityKind,
    canonical_encoding: Box<[u8]>,
}

impl PartitionAuthority {
    pub fn monolithic() -> Self {
        let canonical_encoding = encode_partition(&PartitionAuthorityKind::Monolithic)
            .expect("monolithic partition encoding is infallible");
        Self {
            id: PartitionAuthorityId(partition_digest(&canonical_encoding)),
            kind: PartitionAuthorityKind::Monolithic,
            canonical_encoding: canonical_encoding.into_boxed_slice(),
        }
    }

    pub const fn id(&self) -> PartitionAuthorityId {
        self.id
    }

    pub const fn kind(&self) -> &PartitionAuthorityKind {
        &self.kind
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        let canonical = encode_partition(&self.kind)?;
        Ok(canonical == self.canonical_encoding.as_ref()
            && self.id.0 == partition_digest(&canonical))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledTranscriptSegment {
    pub segment: CairoTranscriptSegment,
    pub entry_state: TranscriptStateVersion,
    pub exit_state: TranscriptStateVersion,
    pub consumed: Vec<ValueRange>,
    pub produced: Vec<ValueRange>,
}

impl CompiledTranscriptSegment {
    pub fn bind_plan(
        transcript: &CairoBlake2sTranscriptPlan,
        inputs: &[TranscriptInputBinding],
        outputs: &[TranscriptOutputBinding],
    ) -> Result<Vec<Self>, CompiledProofError> {
        transcript
            .segments()
            .iter()
            .enumerate()
            .map(|(index, planned)| {
                let consumed = planned
                    .operation_range
                    .clone()
                    .filter_map(|operation| {
                        operation_input(transcript.schedule().operations()[operation])
                    })
                    .map(|id| {
                        inputs
                            .iter()
                            .find(|binding| binding.id == id)
                            .map(|binding| ValueRange {
                                version: binding.value,
                                elements: binding.elements,
                            })
                            .ok_or(CompiledProofError::TranscriptSegmentBinding { index })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let produced = planned
                    .operation_range
                    .clone()
                    .filter_map(|operation| {
                        operation_output(transcript.schedule().operations()[operation])
                    })
                    .map(|id| {
                        outputs
                            .iter()
                            .find(|binding| binding.id == id)
                            .map(|binding| ValueRange {
                                version: binding.value,
                                elements: binding.elements,
                            })
                            .ok_or(CompiledProofError::TranscriptSegmentBinding { index })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    segment: planned.segment,
                    entry_state: TranscriptStateVersion(
                        u32::try_from(index).map_err(|_| CompiledProofError::SizeOverflow)?,
                    ),
                    exit_state: TranscriptStateVersion(
                        u32::try_from(index + 1).map_err(|_| CompiledProofError::SizeOverflow)?,
                    ),
                    consumed,
                    produced,
                })
            })
            .collect()
    }
}

fn operation_input(operation: TranscriptOperation) -> Option<TranscriptInputId> {
    match operation {
        TranscriptOperation::MixFelts { source, .. }
        | TranscriptOperation::MixU32s { source, .. }
        | TranscriptOperation::MixU64 { source, .. }
        | TranscriptOperation::AbsorbRoot { source, .. }
        | TranscriptOperation::AbsorbPowNonce { source, .. } => Some(source),
        TranscriptOperation::DrawSecureFelt { .. }
        | TranscriptOperation::DrawSecureFelts { .. }
        | TranscriptOperation::DrawU32s { .. }
        | TranscriptOperation::DrawQueries { .. } => None,
    }
}

fn operation_output(operation: TranscriptOperation) -> Option<TranscriptOutputId> {
    match operation {
        TranscriptOperation::DrawSecureFelt { output, .. }
        | TranscriptOperation::DrawSecureFelts { output, .. }
        | TranscriptOperation::DrawU32s { output, .. }
        | TranscriptOperation::DrawQueries { output, .. } => Some(output),
        TranscriptOperation::MixFelts { .. }
        | TranscriptOperation::MixU32s { .. }
        | TranscriptOperation::MixU64 { .. }
        | TranscriptOperation::AbsorbRoot { .. }
        | TranscriptOperation::AbsorbPowNonce { .. } => None,
    }
}

pub(super) fn fixed_content_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FIXED_CONTENT_DOMAIN);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn initializer_digest(
    id: ModuleGlobalInitializerId,
    module: &ModuleIdentity,
    symbol: &[u8],
    bytes: usize,
    alignment: usize,
    immutable: bool,
    atoms: &[ModuleGlobalInitializerAtom],
) -> Result<[u8; 32], CompiledProofError> {
    let mut out = Vec::from(MODULE_INITIALIZER_DOMAIN);
    push_u32(&mut out, id.0);
    push_bytes(&mut out, module.canonical_encoding())?;
    out.extend_from_slice(module.digest());
    push_bytes(&mut out, symbol)?;
    push_size(&mut out, bytes)?;
    push_size(&mut out, alignment)?;
    out.push(u8::from(immutable));
    push_size(&mut out, atoms.len())?;
    for atom in atoms {
        let destination = atom.destination();
        push_size(&mut out, destination.start)?;
        push_size(&mut out, destination.end)?;
        match atom {
            ModuleGlobalInitializerAtom::Literal { bytes, .. } => {
                out.push(0);
                push_bytes(&mut out, bytes)?;
            }
            ModuleGlobalInitializerAtom::FixedValueAddress {
                value,
                source_byte_offset,
                ..
            } => {
                out.push(1);
                push_u32(&mut out, value.0);
                push_size(&mut out, *source_byte_offset)?;
            }
        }
    }
    Ok(*blake3::hash(&out).as_bytes())
}

fn encode_partition(kind: &PartitionAuthorityKind) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Vec::from(PARTITION_DOMAIN);
    match kind {
        PartitionAuthorityKind::Monolithic => out.push(0),
    }
    Ok(out)
}

fn partition_digest(canonical: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PARTITION_DOMAIN);
    hasher.update(&(canonical.len() as u64).to_le_bytes());
    hasher.update(canonical);
    *hasher.finalize().as_bytes()
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_size(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledProofError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CompiledProofError> {
    push_size(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}
