use super::*;

pub(super) fn validate(
    input: &CompiledProofInput,
    transcript: &CairoBlake2sTranscriptPlan,
) -> Result<(), CompiledProofError> {
    if input.transcript_segments.len() != transcript.segments().len() {
        return Err(CompiledProofError::TranscriptSegmentCount {
            expected: transcript.segments().len(),
            actual: input.transcript_segments.len(),
        });
    }
    let expected = CompiledTranscriptSegment::bind_plan(
        transcript,
        &input.transcript_inputs,
        &input.transcript_outputs,
    )?;
    for (index, (binding, expected)) in input.transcript_segments.iter().zip(expected).enumerate() {
        if binding != &expected {
            return Err(CompiledProofError::TranscriptSegmentBinding { index });
        }
    }
    Ok(())
}
