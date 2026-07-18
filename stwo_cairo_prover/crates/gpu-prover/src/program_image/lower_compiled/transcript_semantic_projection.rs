//! Exact semantic releases from the canonical transcript schedule.
//!
//! Transcript work is structural authority, not a CUDA [`OpNode`]. This
//! projection only binds ready inputs and allocates transcript-owned outputs
//! at the release point of one exact schedule segment.

use std::collections::BTreeMap;

use stwo_backend_cuda::{TranscriptInputId, TranscriptOperation, TranscriptOutputId};

use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, ProofArenaPlan};
use crate::compiled_proof::{
    CompiledTranscriptSegment, ElementRange, ElementType, LayoutAxis, Region,
    TranscriptInputBinding, TranscriptOutputBinding, TranscriptStateVersion, ValueDesc,
    ValueLayout, ValueOrigin, ValueRange, ValueVersion,
};
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CairoTranscriptOutput,
    CairoTranscriptSegment, TranscriptSegmentPlan,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const WORD_AXIS_TAG: u16 = 0;
const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.transcript-semantic-segment.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredTranscriptInput {
    pub(super) semantic: CairoTranscriptInput,
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) binding: TranscriptInputBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredTranscriptOutput {
    pub(super) semantic: CairoTranscriptOutput,
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) binding: TranscriptOutputBinding,
    pub(super) value: ValueDesc,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredTranscriptSegment {
    plan_digest: [u8; 32],
    prior_digest: Option<[u8; 32]>,
    index: usize,
    plan: TranscriptSegmentPlan,
    inputs: Vec<LoweredTranscriptInput>,
    outputs: Vec<LoweredTranscriptOutput>,
    compiled: CompiledTranscriptSegment,
    digest: [u8; 32],
}

impl LoweredTranscriptSegment {
    pub(super) const fn segment(&self) -> CairoTranscriptSegment {
        self.plan.segment
    }

    pub(super) const fn index(&self) -> usize {
        self.index
    }

    pub(super) const fn operation_range(&self) -> &core::ops::Range<usize> {
        &self.plan.operation_range
    }

    pub(super) fn inputs(&self) -> &[LoweredTranscriptInput] {
        &self.inputs
    }

    pub(super) fn outputs(&self) -> &[LoweredTranscriptOutput] {
        &self.outputs
    }

    pub(super) const fn compiled(&self) -> &CompiledTranscriptSegment {
        &self.compiled
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub(super) fn output(
        &self,
        semantic: CairoTranscriptOutput,
    ) -> Option<&LoweredTranscriptOutput> {
        self.outputs
            .iter()
            .find(|output| output.semantic == semantic)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputInventory {
    semantic: CairoTranscriptInput,
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
    elements: ElementRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputInventory {
    semantic: CairoTranscriptOutput,
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
    elements: ElementRange,
    alignment: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TranscriptInventory {
    plan_digest: [u8; 32],
    inputs: Vec<InputInventory>,
    outputs: Vec<OutputInventory>,
}

/// Release one exact next segment or publish no semantic version.
pub(super) fn lower_segment(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    expected: CairoTranscriptSegment,
    prior: Option<&LoweredTranscriptSegment>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredTranscriptSegment, InvocationShapeError> {
    let inventory = TranscriptInventory::compile(arena, transcript)?;
    if let Some(prior) = prior {
        validate_receipt(arena, transcript, prior)?;
        for output in &prior.outputs {
            if values.version(output.catalog).map_err(invalid)? != output.binding.value {
                return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
            }
        }
    }
    let index = match prior {
        Some(prior) => prior
            .index
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?,
        None => 0,
    };
    let plan = transcript
        .segments()
        .get(index)
        .filter(|plan| plan.segment == expected)
        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
    let mut next_values = values.clone();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for operation in &transcript.schedule().operations()[plan.operation_range.clone()] {
        if let Some(id) = operation_input(*operation) {
            let input = exact_input(&inventory, id)?;
            inputs.push(LoweredTranscriptInput {
                semantic: input.semantic,
                catalog: input.catalog,
                arena: input.arena,
                binding: TranscriptInputBinding {
                    id,
                    value: next_values.version(input.catalog).map_err(invalid)?,
                    elements: input.elements,
                },
            });
        }
        if let Some(id) = operation_output(*operation) {
            let output = exact_output(&inventory, id)?;
            let version = next_values
                .allocate_output(output.catalog)
                .map_err(invalid)?;
            outputs.push(LoweredTranscriptOutput {
                semantic: output.semantic,
                catalog: output.catalog,
                arena: output.arena,
                binding: TranscriptOutputBinding {
                    id,
                    value: version,
                    elements: output.elements,
                },
                value: output_value(output, version, id),
            });
        }
    }
    let compiled = compiled_segment(index, plan.segment, &inputs, &outputs)?;
    let mut lowered = LoweredTranscriptSegment {
        plan_digest: inventory.plan_digest,
        prior_digest: prior.map(LoweredTranscriptSegment::digest),
        index,
        plan: plan.clone(),
        inputs,
        outputs,
        compiled,
        digest: [0; 32],
    };
    lowered.digest = receipt_digest(&lowered)?;
    validate_receipt(arena, transcript, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    expected: CairoTranscriptSegment,
    prior: Option<&LoweredTranscriptSegment>,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredTranscriptSegment,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_segment(arena, transcript, expected, prior, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
    }
}

impl TranscriptInventory {
    fn compile(
        arena: &ProofArenaPlan,
        transcript: &CairoBlake2sTranscriptPlan,
    ) -> Result<Self, InvocationShapeError> {
        validate_plan(transcript)?;
        let planned = arena.transcript();
        if planned.schedule_key != transcript.schedule_key()
            || planned.requirements != *transcript.schedule().requirements()
        {
            return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
        }
        let input_bindings = unique_bindings(&planned.inputs)?;
        let output_bindings = unique_bindings(&planned.outputs)?;
        let schedule_inputs = transcript
            .schedule()
            .requirements()
            .inputs
            .iter()
            .map(|requirement| (requirement.id, requirement.min_words))
            .collect::<BTreeMap<_, _>>();
        let schedule_outputs = transcript
            .schedule()
            .requirements()
            .outputs
            .iter()
            .map(|requirement| (requirement.id, requirement.min_words))
            .collect::<BTreeMap<_, _>>();
        if schedule_inputs.len() != transcript.inputs().len()
            || schedule_outputs.len() != transcript.outputs().len()
            || input_bindings.len() != schedule_inputs.len()
            || output_bindings.len() != schedule_outputs.len()
        {
            return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
        }
        let catalog = BaseProducerCatalog::compile(arena)?;
        let inputs = transcript
            .inputs()
            .iter()
            .map(|requirement| {
                let id = requirement.semantic.id().map_err(invalid)?;
                let words = schedule_inputs
                    .get(&id)
                    .copied()
                    .filter(|&words| words == requirement.min_words)
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                let binding = *input_bindings
                    .get(&id)
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                let catalog_id = validate_catalog_binding(
                    &catalog,
                    binding,
                    BufferPurpose::TranscriptInput,
                    id.0,
                    words,
                )?;
                Ok::<_, InvocationShapeError>(InputInventory {
                    semantic: requirement.semantic,
                    catalog: catalog_id,
                    arena: binding,
                    elements: ElementRange::new(0, words)
                        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = transcript
            .outputs()
            .iter()
            .map(|requirement| {
                let id = requirement.semantic.id().map_err(invalid)?;
                let words = schedule_outputs
                    .get(&id)
                    .copied()
                    .filter(|&words| words == requirement.min_words)
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                let binding = *output_bindings
                    .get(&id)
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                let catalog_id = validate_catalog_binding(
                    &catalog,
                    binding,
                    BufferPurpose::TranscriptOutput,
                    id.0,
                    words,
                )?;
                let slot = arena
                    .layout()
                    .slot(binding.physical)
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                let alignment = slot
                    .alignment_words
                    .checked_mul(WORD_BYTES)
                    .filter(|alignment| alignment.is_power_of_two())
                    .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
                Ok::<_, InvocationShapeError>(OutputInventory {
                    semantic: requirement.semantic,
                    catalog: catalog_id,
                    arena: binding,
                    elements: ElementRange::new(0, words)
                        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?,
                    alignment,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan_digest: plan_digest(transcript)?,
            inputs,
            outputs,
        })
    }
}

fn validate_plan(transcript: &CairoBlake2sTranscriptPlan) -> Result<(), InvocationShapeError> {
    let operations = transcript.schedule().operations();
    let boundaries = transcript.boundaries();
    let segments = transcript.segments();
    if operations.is_empty()
        || operations.len() != boundaries.len()
        || segments.is_empty()
        || transcript
            .inputs()
            .iter()
            .map(|requirement| requirement.semantic.id().map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?
            != operations
                .iter()
                .filter_map(|operation| operation_input(*operation))
                .collect::<Vec<_>>()
        || transcript
            .outputs()
            .iter()
            .map(|requirement| requirement.semantic.id().map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?
            != operations
                .iter()
                .filter_map(|operation| operation_output(*operation))
                .collect::<Vec<_>>()
    {
        return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
    }
    let mut cursor = 0;
    let mut previous_end = None;
    let mut seen = Vec::new();
    for segment in segments {
        if segment.operation_range.start != cursor
            || segment.operation_range.start >= segment.operation_range.end
            || segment.operation_range.end > operations.len()
            || segment.starts_after != previous_end
            || seen.contains(&segment.segment)
        {
            return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
        }
        seen.push(segment.segment);
        for index in segment.operation_range.clone() {
            let boundary = boundaries
                .get(index)
                .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
            if boundary.operation_index != index
                || boundary.segment != segment.segment
                || boundary.semantic.id().map_err(invalid)? != operations[index].boundary()
            {
                return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
            }
        }
        if boundaries[segment.operation_range.end - 1].semantic != segment.ends_at {
            return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
        }
        cursor = segment.operation_range.end;
        previous_end = Some(segment.ends_at);
    }
    (cursor == operations.len())
        .then_some(())
        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)
}

fn validate_receipt(
    arena: &ProofArenaPlan,
    transcript: &CairoBlake2sTranscriptPlan,
    lowered: &LoweredTranscriptSegment,
) -> Result<(), InvocationShapeError> {
    let inventory = TranscriptInventory::compile(arena, transcript)?;
    let plan = transcript
        .segments()
        .get(lowered.index)
        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
    let expected_inputs = plan
        .operation_range
        .clone()
        .filter_map(|index| operation_input(transcript.schedule().operations()[index]))
        .map(|id| {
            let input = exact_input(&inventory, id)?;
            let supplied = lowered
                .inputs
                .iter()
                .find(|candidate| candidate.binding.id == id)
                .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
            Ok::<_, InvocationShapeError>(LoweredTranscriptInput {
                semantic: input.semantic,
                catalog: input.catalog,
                arena: input.arena,
                binding: TranscriptInputBinding {
                    id,
                    value: supplied.binding.value,
                    elements: input.elements,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_outputs = plan
        .operation_range
        .clone()
        .filter_map(|index| operation_output(transcript.schedule().operations()[index]))
        .map(|id| {
            let output = exact_output(&inventory, id)?;
            let supplied = lowered
                .outputs
                .iter()
                .find(|candidate| candidate.binding.id == id)
                .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
            Ok::<_, InvocationShapeError>(LoweredTranscriptOutput {
                semantic: output.semantic,
                catalog: output.catalog,
                arena: output.arena,
                binding: TranscriptOutputBinding {
                    id,
                    value: supplied.binding.value,
                    elements: output.elements,
                },
                value: output_value(output, supplied.binding.value, id),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if lowered.plan_digest != inventory.plan_digest
        || &lowered.plan != plan
        || lowered.inputs != expected_inputs
        || lowered.outputs != expected_outputs
        || lowered.compiled
            != compiled_segment(
                lowered.index,
                plan.segment,
                &expected_inputs,
                &expected_outputs,
            )?
        || receipt_digest(lowered)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
    }
    Ok(())
}

fn compiled_segment(
    index: usize,
    segment: CairoTranscriptSegment,
    inputs: &[LoweredTranscriptInput],
    outputs: &[LoweredTranscriptOutput],
) -> Result<CompiledTranscriptSegment, InvocationShapeError> {
    let entry = u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?;
    let exit = entry
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    Ok(CompiledTranscriptSegment {
        segment,
        entry_state: TranscriptStateVersion(entry),
        exit_state: TranscriptStateVersion(exit),
        consumed: inputs
            .iter()
            .map(|input| ValueRange {
                version: input.binding.value,
                elements: input.binding.elements,
            })
            .collect(),
        produced: outputs
            .iter()
            .map(|output| ValueRange {
                version: output.binding.value,
                elements: output.binding.elements,
            })
            .collect(),
    })
}

fn output_value(
    output: &OutputInventory,
    version: ValueVersion,
    id: TranscriptOutputId,
) -> ValueDesc {
    ValueDesc {
        version,
        layout: ValueLayout {
            element: ElementType::U32,
            axes: vec![LayoutAxis {
                tag: WORD_AXIS_TAG,
                extent: output.arena.len_words,
                stride_bytes: WORD_BYTES,
            }],
        },
        alignment: output.alignment,
        origin: ValueOrigin::TranscriptOutput(id),
        region: Region::Dynamic,
    }
}

fn exact_input(
    inventory: &TranscriptInventory,
    id: TranscriptInputId,
) -> Result<&InputInventory, InvocationShapeError> {
    one(inventory
        .inputs
        .iter()
        .filter(|input| input.semantic.id().ok() == Some(id)))
}

fn exact_output(
    inventory: &TranscriptInventory,
    id: TranscriptOutputId,
) -> Result<&OutputInventory, InvocationShapeError> {
    one(inventory
        .outputs
        .iter()
        .filter(|output| output.semantic.id().ok() == Some(id)))
}

fn one<T>(mut matches: impl Iterator<Item = T>) -> Result<T, InvocationShapeError> {
    let exact = matches
        .next()
        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
    }
    Ok(exact)
}

fn unique_bindings<I: Copy + Ord>(
    bindings: &[(I, ArenaBinding)],
) -> Result<BTreeMap<I, ArenaBinding>, InvocationShapeError> {
    let exact = bindings.iter().copied().collect::<BTreeMap<_, _>>();
    (exact.len() == bindings.len())
        .then_some(exact)
        .ok_or(InvocationShapeError::InvalidTranscriptSemanticProjection)
}

fn validate_catalog_binding(
    catalog: &BaseProducerCatalog,
    arena: ArenaBinding,
    purpose: BufferPurpose,
    ordinal: u32,
    words: usize,
) -> Result<ArenaCatalogValueId, InvocationShapeError> {
    let id = ArenaCatalogValueId(arena.logical.0);
    let value = catalog.value(id)?;
    if value.logical != arena.logical
        || value.physical != arena.physical
        || value.purpose != purpose
        || value.ordinal != ordinal
        || value.words != words
        || arena.len_words != words
        || value.component.is_some()
        || value.part.is_some()
    {
        return Err(InvocationShapeError::InvalidTranscriptSemanticProjection);
    }
    Ok(id)
}

fn plan_digest(transcript: &CairoBlake2sTranscriptPlan) -> Result<[u8; 32], InvocationShapeError> {
    let encoding = transcript.canonical_encoding().map_err(invalid)?;
    Ok(*blake3::hash(&encoding).as_bytes())
}

fn receipt_digest(lowered: &LoweredTranscriptSegment) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&lowered.plan_digest);
    match lowered.prior_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hasher.update(&digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hash_size(&mut hasher, lowered.index)?;
    hash_size(&mut hasher, lowered.plan.operation_range.start)?;
    hash_size(&mut hasher, lowered.plan.operation_range.end)?;
    hasher.update(&lowered.plan.ends_at.id().map_err(invalid)?.0.to_le_bytes());
    hash_size(&mut hasher, lowered.inputs.len())?;
    for input in &lowered.inputs {
        hasher.update(&input.binding.id.0.to_le_bytes());
        hash_catalog_binding(
            &mut hasher,
            input.catalog,
            input.arena,
            input.binding.value,
            input.binding.elements,
        )?;
    }
    hash_size(&mut hasher, lowered.outputs.len())?;
    for output in &lowered.outputs {
        hasher.update(&output.binding.id.0.to_le_bytes());
        hash_catalog_binding(
            &mut hasher,
            output.catalog,
            output.arena,
            output.binding.value,
            output.binding.elements,
        )?;
        hash_size(&mut hasher, output.value.alignment)?;
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_catalog_binding(
    hasher: &mut blake3::Hasher,
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
    value: ValueVersion,
    elements: ElementRange,
) -> Result<(), InvocationShapeError> {
    hasher.update(&catalog.0.to_le_bytes());
    hasher.update(&arena.logical.0.to_le_bytes());
    hasher.update(&arena.physical.0.to_le_bytes());
    hash_size(hasher, arena.len_words)?;
    hasher.update(&value.0.to_le_bytes());
    hash_size(hasher, elements.start)?;
    hash_size(hasher, elements.end)
}

fn hash_size(hasher: &mut blake3::Hasher, value: usize) -> Result<(), InvocationShapeError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

const fn operation_input(operation: TranscriptOperation) -> Option<TranscriptInputId> {
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

const fn operation_output(operation: TranscriptOperation) -> Option<TranscriptOutputId> {
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

fn invalid<T>(_: T) -> InvocationShapeError {
    InvocationShapeError::InvalidTranscriptSemanticProjection
}

#[cfg(test)]
#[path = "transcript_semantic_projection_tests.rs"]
mod tests;
