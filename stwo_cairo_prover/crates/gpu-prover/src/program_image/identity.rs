use super::*;
use crate::transcript_plan::CairoBlake2sTranscriptPlan;

const DOMAIN: &[u8] = b"stwo-cairo.real-arena-program-inventory.catalog.v1\0";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ArenaProgramInventoryIdentity {
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl ArenaProgramInventoryIdentity {
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

pub(super) fn build(
    topology: &[u8],
    transcript: &CairoBlake2sTranscriptPlan,
    values: &[ProgramValueDesc],
    inputs: &[ProgramTranscriptInputBinding],
    outputs: &[ProgramTranscriptOutputBinding],
    proof: &ProgramProofOutput,
    frontier: &OperationFrontier,
) -> Result<ArenaProgramInventoryIdentity, ArenaProgramInventoryError> {
    let mut out = Encoder::new();
    out.bytes(topology)?;
    out.bytes(&transcript.canonical_encoding()?)?;
    out.count(values.len())?;
    for value in values {
        out.u32(value.id.0);
        out.u32(value.logical.0);
        match value.component {
            Some(component) => {
                out.byte(1);
                out.bytes(component.as_bytes())?;
            }
            None => out.byte(0),
        }
        match value.part {
            Some(part) => {
                out.byte(1);
                encode_part(&mut out, part);
            }
            None => out.byte(0),
        }
        out.u32(value.purpose as u32);
        out.u32(value.ordinal);
        out.usize(value.layout.element.bytes)?;
        out.u32(value.layout.element.tag);
        out.count(value.layout.axes.len())?;
        for axis in &value.layout.axes {
            out.u16(axis.tag);
            out.usize(axis.extent)?;
            out.usize(axis.stride_bytes)?;
        }
        out.usize(value.alignment)?;
        out.byte(value.lifetime.first as u8);
        out.byte(value.lifetime.last as u8);
        encode_origin(&mut out, value.origin);
    }
    out.count(inputs.len())?;
    for binding in inputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.range(&binding.value_words)?;
    }
    out.count(outputs.len())?;
    for binding in outputs {
        out.u32(binding.id.0);
        out.u32(binding.value.0);
        out.range(&binding.value_words)?;
    }
    out.bytes(proof.codec.canonical_encoding())?;
    out.raw(proof.codec.digest());
    out.count(proof.sections.len())?;
    for section in &proof.sections {
        out.byte(section_tag(section.section));
        out.u32(section.value.0);
        out.range(&section.value_words)?;
    }
    out.usize(proof.layout.total_words)?;
    encode_frontier(&mut out, frontier)?;
    let canonical_encoding = out.finish();
    Ok(ArenaProgramInventoryIdentity {
        digest: *blake3::hash(&canonical_encoding).as_bytes(),
        canonical_encoding: canonical_encoding.into_boxed_slice(),
    })
}

fn encode_origin(out: &mut Encoder, origin: ProgramValueOrigin) {
    match origin {
        ProgramValueOrigin::ExternalInput(id) => {
            out.byte(0);
            out.u32(id.0);
        }
        ProgramValueOrigin::FixedImage(id) => {
            out.byte(1);
            out.u32(id.0);
        }
        ProgramValueOrigin::TranscriptOutput(id) => {
            out.byte(2);
            out.u32(id.0);
        }
        ProgramValueOrigin::PendingOperationContract => out.byte(3),
    }
}

fn encode_frontier(
    out: &mut Encoder,
    frontier: &OperationFrontier,
) -> Result<(), ArenaProgramInventoryError> {
    out.u32(frontier.value.0);
    out.u32(frontier.logical.0);
    out.u32(frontier.purpose as u32);
    out.u32(frontier.ordinal);
    out.byte(frontier.producer_epoch as u8);
    out.byte(segment_tag(frontier.earliest_transcript_stage));
    out.byte(frontier.primitive as u8);
    match &frontier.witness_kernel_candidate {
        Some(kernel) => {
            out.byte(1);
            out.bytes(kernel.label.as_bytes())?;
            out.bytes(kernel.kernel_name.as_bytes())?;
            out.u64(kernel.semantic_hash);
            out.u64(kernel.cache_key);
            out.u64(kernel.aot_manifest_hash);
        }
        None => out.byte(0),
    }
    match frontier.launch_candidate {
        Some(launch) => {
            out.byte(1);
            for value in launch.grid.into_iter().chain(launch.block) {
                out.u32(value);
            }
            out.u32(launch.dynamic_shared_bytes);
        }
        None => out.byte(0),
    }
    match &frontier.partition_candidate {
        Some(partition) => {
            out.byte(1);
            out.usize(partition.row_count)?;
            out.usize(partition.row_granularity)?;
            out.u32(partition.global_multiplicity_outputs);
            out.byte(match partition.multiplicity_rule {
                GlobalMultiplicityPartitionRule::CoordinatorOwnedOrCanonicalReduction => 0,
            });
        }
        None => out.byte(0),
    }
    encode_ranges(out, &frontier.known_reads)?;
    encode_ranges(out, &frontier.known_writes)?;
    out.count(frontier.missing.len())?;
    for field in &frontier.missing {
        out.byte(*field as u8);
    }
    Ok(())
}

fn encode_ranges(
    out: &mut Encoder,
    ranges: &[ArenaCatalogRange],
) -> Result<(), ArenaProgramInventoryError> {
    out.count(ranges.len())?;
    for range in ranges {
        out.u32(range.value.0);
        out.range(&range.value_words)?;
    }
    Ok(())
}

fn encode_part(out: &mut Encoder, part: TracePartId) {
    match part {
        TracePartId::Main => out.byte(0),
        TracePartId::MemoryBig(index) => {
            out.byte(1);
            out.u32(index);
        }
        TracePartId::MemorySmall => out.byte(2),
    }
}

fn segment_tag(segment: CairoTranscriptSegment) -> u8 {
    match segment {
        CairoTranscriptSegment::BootstrapThroughBase => 0,
        CairoTranscriptSegment::InteractionPowAndLookup => 1,
        CairoTranscriptSegment::InteractionAndComposition => 2,
        CairoTranscriptSegment::CompositionAndOods => 3,
        CairoTranscriptSegment::OodsAndQuotient => 4,
        CairoTranscriptSegment::FriLayer(_) => 5,
        CairoTranscriptSegment::FriLastLayer => 6,
        CairoTranscriptSegment::QueryPowAndPositions => 7,
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

struct Encoder(Vec<u8>);

impl Encoder {
    fn new() -> Self {
        Self(DOMAIN.to_vec())
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<(), ArenaProgramInventoryError> {
        self.u64(u64::try_from(value).map_err(|_| ArenaProgramInventoryError::SizeOverflow)?);
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), ArenaProgramInventoryError> {
        self.usize(value)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), ArenaProgramInventoryError> {
        self.count(bytes.len())?;
        self.raw(bytes);
        Ok(())
    }

    fn range(&mut self, range: &Range<usize>) -> Result<(), ArenaProgramInventoryError> {
        self.usize(range.start)?;
        self.usize(range.end)
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }
}
