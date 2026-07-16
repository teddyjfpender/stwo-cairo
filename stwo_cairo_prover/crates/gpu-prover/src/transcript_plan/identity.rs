use stwo_backend_cuda::{TranscriptOperation, TranscriptStart};

use super::*;

const ENCODING_TAG: &[u8] = b"stwo-cairo.blake2s.transcript.canonical.v1\0";

pub(super) fn encode(plan: &CairoBlake2sTranscriptPlan) -> Result<Vec<u8>, TranscriptPlanError> {
    let schedule = plan.schedule();
    let mut out = Encoder::default();
    out.raw(ENCODING_TAG);
    out.bytes(CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG.as_bytes())?;
    out.bytes(schedule.protocol_tag().as_bytes())?;
    out.u32(schedule.max_rejection_rounds());
    match schedule.start() {
        TranscriptStart::Default => out.byte(0),
        TranscriptStart::DeviceState(input) => {
            out.byte(1);
            out.u32(input.0);
        }
    }
    out.count(schedule.operations().len())?;
    for operation in schedule.operations() {
        encode_operation(&mut out, *operation);
    }

    let requirements = schedule.requirements();
    out.size(requirements.state_words)?;
    out.size(requirements.boundary_snapshot_words)?;
    out.size(requirements.input_snapshot_words)?;
    out.size(requirements.input_snapshot_used_words)?;
    out.size(requirements.output_snapshot_words)?;
    out.size(requirements.output_snapshot_used_words)?;
    out.count(requirements.inputs.len())?;
    for input in &requirements.inputs {
        out.u32(input.id.0);
        out.size(input.min_words)?;
    }
    out.count(requirements.outputs.len())?;
    for output in &requirements.outputs {
        out.u32(output.id.0);
        out.size(output.min_words)?;
    }

    out.count(plan.inputs().len())?;
    for input in plan.inputs() {
        out.u32(input.semantic.id()?.0);
        out.size(input.min_words)?;
    }
    out.count(plan.outputs().len())?;
    for output in plan.outputs() {
        out.u32(output.semantic.id()?.0);
        out.size(output.min_words)?;
    }
    out.count(plan.boundaries().len())?;
    for boundary in plan.boundaries() {
        out.u32(boundary.semantic.id()?.0);
        out.size(boundary.operation_index)?;
        out.segment(boundary.segment);
    }
    out.count(plan.segments().len())?;
    for segment in plan.segments() {
        out.segment(segment.segment);
        out.size(segment.operation_range.start)?;
        out.size(segment.operation_range.end)?;
        match segment.starts_after {
            Some(boundary) => {
                out.byte(1);
                out.u32(boundary.id()?.0);
            }
            None => out.byte(0),
        }
        out.u32(segment.ends_at.id()?.0);
    }
    Ok(out.bytes)
}

fn encode_operation(out: &mut Encoder, operation: TranscriptOperation) {
    match operation {
        TranscriptOperation::MixFelts {
            boundary,
            source,
            n_felts,
        } => {
            out.byte(0);
            out.u32(boundary.0);
            out.u32(source.0);
            out.u32(n_felts);
        }
        TranscriptOperation::MixU32s {
            boundary,
            source,
            n_words,
        } => {
            out.byte(1);
            out.u32(boundary.0);
            out.u32(source.0);
            out.u32(n_words);
        }
        TranscriptOperation::MixU64 { boundary, source } => {
            out.byte(2);
            out.u32(boundary.0);
            out.u32(source.0);
        }
        TranscriptOperation::AbsorbRoot { boundary, source } => {
            out.byte(3);
            out.u32(boundary.0);
            out.u32(source.0);
        }
        TranscriptOperation::AbsorbPowNonce {
            boundary,
            source,
            pow_bits,
        } => {
            out.byte(4);
            out.u32(boundary.0);
            out.u32(source.0);
            out.u32(pow_bits);
        }
        TranscriptOperation::DrawSecureFelt { boundary, output } => {
            out.byte(5);
            out.u32(boundary.0);
            out.u32(output.0);
        }
        TranscriptOperation::DrawSecureFelts {
            boundary,
            output,
            n_felts,
        } => {
            out.byte(6);
            out.u32(boundary.0);
            out.u32(output.0);
            out.u32(n_felts);
        }
        TranscriptOperation::DrawU32s { boundary, output } => {
            out.byte(7);
            out.u32(boundary.0);
            out.u32(output.0);
        }
        TranscriptOperation::DrawQueries {
            boundary,
            output,
            log_domain_size,
            n_queries,
        } => {
            out.byte(8);
            out.u32(boundary.0);
            out.u32(output.0);
            out.u32(log_domain_size);
            out.u32(n_queries);
        }
    }
}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn raw(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), TranscriptPlanError> {
        self.count(value.len())?;
        self.raw(value);
        Ok(())
    }

    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn size(&mut self, value: usize) -> Result<(), TranscriptPlanError> {
        let value = u64::try_from(value).map_err(|_| TranscriptPlanError::SizeOverflow)?;
        self.raw(&value.to_le_bytes());
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), TranscriptPlanError> {
        self.size(value)
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
}

#[cfg(test)]
mod tests {
    use stwo::core::fri::FriConfig;

    use super::*;

    fn plan(dynamic_felts: usize) -> CairoBlake2sTranscriptPlan {
        plan_with_interaction_pow_bits(
            ClaimMixShape {
                enable_felts: 2,
                log_size_felts: 2,
                public_data_felts: 3,
            },
            PcsConfig {
                pow_bits: 0,
                fri_config: FriConfig::new(2, 1, 13, 2),
                lifting_log_size: Some(10),
            },
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(dynamic_felts),
                oods_sampled_values_felts: Some(7),
            },
            0,
        )
        .unwrap()
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    #[test]
    fn canonical_encoding_binds_domains_requirements_and_boundaries() {
        let base = plan(5);
        let encoded = encode(&base).unwrap();
        assert!(contains(
            &encoded,
            CAIRO_BLAKE2S_TRANSCRIPT_SCHEDULE_TAG.as_bytes()
        ));
        assert!(contains(
            &encoded,
            base.schedule().protocol_tag().as_bytes()
        ));

        let changed_requirement = plan(6);
        assert_ne!(encoded, encode(&changed_requirement).unwrap());

        let mut changed_boundary = base.clone();
        changed_boundary.boundaries[0].operation_index += 1;
        assert_ne!(encoded, encode(&changed_boundary).unwrap());
    }
}
