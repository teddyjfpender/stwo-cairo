use std::collections::{BTreeMap, BTreeSet};

use stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE;

use super::*;
use crate::arena_plan::{ArenaBinding, BlakeGWitnessContract, ProofArenaPlan};
use crate::compiled_proof::{ElementType, LayoutAxis};
use crate::memory_ledger::{MemoryPurposeClass, MemoryPurposeClass::*};
use crate::shape_executable::TopologyKey;
use crate::transcript_plan::{
    CairoBlake2sTranscriptPlan, CairoTranscriptInput, CAIRO_STATIC_TRANSCRIPT_INPUTS,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();
const WORD_ELEMENT_TAG: u32 = 1;
const WORD_AXIS_TAG: u16 = 0;

pub(super) fn from_planned_parts(
    topology: &TopologyKey,
    transcript: &CairoBlake2sTranscriptPlan,
    arena: &ProofArenaPlan,
) -> Result<ArenaProgramInventory, ArenaProgramInventoryError> {
    validate_arena_cardinality(arena)?;
    let transcript_inputs = transcript_inputs(transcript, arena)?;
    let transcript_outputs = transcript_outputs(transcript, arena)?;
    let output = proof_output(arena)?;
    let values = catalog_values(arena, &transcript_inputs, &transcript_outputs, &output)?;
    let frontier = first_missing_producer_frontier(arena, &values)?;
    let identity = identity::build(
        topology.canonical_encoding(),
        transcript,
        &values,
        &transcript_inputs,
        &transcript_outputs,
        &output,
        &frontier,
    )?;
    Ok(ArenaProgramInventory {
        identity,
        values,
        transcript_inputs,
        transcript_outputs,
        output,
        frontier,
    })
}

fn validate_arena_cardinality(arena: &ProofArenaPlan) -> Result<(), ArenaProgramInventoryError> {
    if arena.logical_buffers().is_empty() || arena.logical_buffers().len() != arena.bindings().len()
    {
        return Err(ArenaProgramInventoryError::InvalidArena(
            "logical value and binding cardinality differs",
        ));
    }
    Ok(())
}

fn transcript_inputs(
    transcript: &CairoBlake2sTranscriptPlan,
    arena: &ProofArenaPlan,
) -> Result<Vec<ProgramTranscriptInputBinding>, ArenaProgramInventoryError> {
    let planned = &arena.transcript().inputs;
    let requirements = transcript.inputs();
    if planned.len() != requirements.len() {
        return Err(ArenaProgramInventoryError::Transcript(
            "input binding cardinality differs",
        ));
    }
    let planned = planned.iter().copied().collect::<BTreeMap<_, _>>();
    if planned.len() != requirements.len() {
        return Err(ArenaProgramInventoryError::Transcript(
            "duplicate input binding id",
        ));
    }
    requirements
        .iter()
        .map(|requirement| {
            let id = requirement.semantic.id()?;
            let binding = *planned
                .get(&id)
                .ok_or(ArenaProgramInventoryError::Transcript(
                    "missing input binding id",
                ))?;
            if binding.len_words != requirement.min_words {
                return Err(ArenaProgramInventoryError::Transcript(
                    "input binding width differs",
                ));
            }
            validate_binding(arena, binding)?;
            Ok(ProgramTranscriptInputBinding {
                id,
                value: catalog_value_id(binding.logical)?,
                value_words: 0..requirement.min_words,
            })
        })
        .collect()
}

fn transcript_outputs(
    transcript: &CairoBlake2sTranscriptPlan,
    arena: &ProofArenaPlan,
) -> Result<Vec<ProgramTranscriptOutputBinding>, ArenaProgramInventoryError> {
    let planned = &arena.transcript().outputs;
    let requirements = transcript.outputs();
    if planned.len() != requirements.len() {
        return Err(ArenaProgramInventoryError::Transcript(
            "output binding cardinality differs",
        ));
    }
    let planned = planned.iter().copied().collect::<BTreeMap<_, _>>();
    if planned.len() != requirements.len() {
        return Err(ArenaProgramInventoryError::Transcript(
            "duplicate output binding id",
        ));
    }
    requirements
        .iter()
        .map(|requirement| {
            let id = requirement.semantic.id()?;
            let binding = *planned
                .get(&id)
                .ok_or(ArenaProgramInventoryError::Transcript(
                    "missing output binding id",
                ))?;
            if binding.len_words != requirement.min_words {
                return Err(ArenaProgramInventoryError::Transcript(
                    "output binding width differs",
                ));
            }
            validate_binding(arena, binding)?;
            Ok(ProgramTranscriptOutputBinding {
                id,
                value: catalog_value_id(binding.logical)?,
                value_words: 0..requirement.min_words,
            })
        })
        .collect()
}

fn proof_output(arena: &ProofArenaPlan) -> Result<ProgramProofOutput, ArenaProgramInventoryError> {
    let decommit = arena.decommit();
    let layout = decommit.proof_bundle_layout.clone();
    layout.validate()?;
    validate_binding(arena, decommit.proof_bundle)?;
    if decommit.proof_bundle.len_words != layout.total_words
        || decommit.requirements.assembly_words != layout.decommitment.len()
        || decommit.proof_shape.trace_trees.len() != 4
        || decommit.proof_shape.fri_trees.len() * 8 != layout.fri_commitments.len()
    {
        return Err(ArenaProgramInventoryError::ProofBundle(
            "bundle extent disagrees with decommit geometry",
        ));
    }
    let sampled_words = decommit
        .proof_shape
        .trace_trees
        .iter()
        .flat_map(|tree| &tree.oods_samples_per_column)
        .try_fold(0usize, |total, &samples| {
            samples
                .checked_mul(SECURE_EXTENSION_DEGREE)
                .and_then(|words| total.checked_add(words))
                .ok_or(ArenaProgramInventoryError::SizeOverflow)
        })?;
    if sampled_words != layout.sampled_values.len() {
        return Err(ArenaProgramInventoryError::ProofBundle(
            "sample topology disagrees with bundle layout",
        ));
    }
    let bundle = catalog_value_id(decommit.proof_bundle.logical)?;
    let sections = ProofBundleSection::CANONICAL
        .into_iter()
        .zip(output_ranges(&layout))
        .map(|(section, value_words)| ProgramProofOutputSection {
            section,
            value: bundle,
            value_words,
        })
        .collect();
    Ok(ProgramProofOutput {
        codec: ProofCodecIdentity::track_a_resident_bundle(),
        layout,
        sections,
    })
}

fn catalog_values(
    arena: &ProofArenaPlan,
    inputs: &[ProgramTranscriptInputBinding],
    outputs: &[ProgramTranscriptOutputBinding],
    proof: &ProgramProofOutput,
) -> Result<Vec<ProgramValueDesc>, ArenaProgramInventoryError> {
    let static_ids = CAIRO_STATIC_TRANSCRIPT_INPUTS
        .into_iter()
        .map(CairoTranscriptInput::id)
        .collect::<Result<BTreeSet<_>, _>>()?;
    let input_by_value = inputs
        .iter()
        .map(|binding| (binding.value, binding.id))
        .collect::<BTreeMap<_, _>>();
    let output_by_value = outputs
        .iter()
        .map(|binding| (binding.value, binding.id))
        .collect::<BTreeMap<_, _>>();
    if input_by_value.len() != inputs.len() || output_by_value.len() != outputs.len() {
        return Err(ArenaProgramInventoryError::Transcript(
            "multiple transcript ids share one logical value",
        ));
    }
    let bundle = proof
        .sections
        .first()
        .ok_or(ArenaProgramInventoryError::ProofBundle(
            "empty proof sections",
        ))?
        .value;

    arena
        .logical_buffers()
        .iter()
        .enumerate()
        .map(|(index, logical)| {
            let expected = LogicalBufferId(
                u32::try_from(index).map_err(|_| ArenaProgramInventoryError::SizeOverflow)?,
            );
            if logical.id != expected {
                return Err(ArenaProgramInventoryError::NonDenseLogicalValue {
                    expected,
                    actual: logical.id,
                });
            }
            let id = catalog_value_id(logical.id)?;
            let binding = arena
                .binding(logical.id)
                .ok_or(ArenaProgramInventoryError::MissingArenaBinding(logical.id))?;
            validate_binding(arena, binding)?;
            let slot = arena
                .layout()
                .slot(binding.physical)
                .ok_or(ArenaProgramInventoryError::MissingArenaSlot(logical.id))?;
            let alignment = slot
                .alignment_words
                .checked_mul(WORD_BYTES)
                .ok_or(ArenaProgramInventoryError::SizeOverflow)?;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(ArenaProgramInventoryError::InvalidAlignment(logical.id));
            }
            let origin = match (
                output_by_value.get(&id).copied(),
                input_by_value.get(&id).copied(),
            ) {
                (Some(output), None) => ProgramValueOrigin::TranscriptOutput(output),
                (None, Some(input)) if static_ids.contains(&input) => {
                    ProgramValueOrigin::ExternalInput(logical.id)
                }
                (Some(_), Some(_)) => {
                    return Err(ArenaProgramInventoryError::Transcript(
                        "one value is both a transcript input and output",
                    ))
                }
                _ if id == bundle => ProgramValueOrigin::PendingOperationContract,
                _ => match MemoryPurposeClass::of(logical.purpose) {
                    FixedData => ProgramValueOrigin::FixedImage(logical.id),
                    Input => ProgramValueOrigin::ExternalInput(logical.id),
                    Dynamic | Output => ProgramValueOrigin::PendingOperationContract,
                },
            };
            Ok(ProgramValueDesc {
                id,
                logical: logical.id,
                component: logical.component,
                part: logical.part,
                purpose: logical.purpose,
                ordinal: logical.ordinal,
                layout: ValueLayout {
                    element: ElementType {
                        tag: WORD_ELEMENT_TAG,
                        bytes: WORD_BYTES,
                    },
                    axes: vec![LayoutAxis {
                        tag: WORD_AXIS_TAG,
                        extent: logical.len_words,
                        stride_bytes: WORD_BYTES,
                    }],
                },
                alignment,
                lifetime: logical.lifetime,
                origin,
            })
        })
        .collect()
}

fn first_missing_producer_frontier(
    arena: &ProofArenaPlan,
    values: &[ProgramValueDesc],
) -> Result<OperationFrontier, ArenaProgramInventoryError> {
    let value = values
        .iter()
        .find(|value| value.origin == ProgramValueOrigin::PendingOperationContract)
        .ok_or(ArenaProgramInventoryError::InvalidArena(
            "real shape unexpectedly has no pending operation value",
        ))?;
    let witness = value.component.and_then(|component| {
        arena
            .witness()
            .components
            .iter()
            .find(|planned| planned.component == component && Some(planned.part) == value.part)
    });
    let primitive = match value.purpose {
        BufferPurpose::ProofBytes => ExecutionPrimitiveClass::ProofAssembly,
        BufferPurpose::TranscriptState
        | BufferPurpose::TranscriptBoundarySnapshots
        | BufferPurpose::TranscriptInputSnapshots
        | BufferPurpose::TranscriptOutputSnapshots
        | BufferPurpose::TranscriptInput
        | BufferPurpose::TranscriptOutput => ExecutionPrimitiveClass::Transcript,
        _ => ExecutionPrimitiveClass::KernelLaunch,
    };
    let (
        witness_kernel_candidate,
        launch_candidate,
        partition_candidate,
        known_reads,
        known_writes,
        missing,
    ) = match witness {
        Some(planned) if planned.blake_g_contract == BlakeGWitnessContract::Recorded => {
            if planned.program.n_mult_tables != 0
                || !planned.requirements.multiplicity_column_words.is_empty()
                || !planned.slots.multiplicity_columns.is_empty()
            {
                return Err(ArenaProgramInventoryError::InvalidArena(
                    "recorded witness multiplicity-free contract drifted",
                ));
            }
            let semantic_hash = planned.program.semantic_hash();
            let row_count = planned.requirements.row_count;
            let row_count_u32 =
                u32::try_from(row_count).map_err(|_| ArenaProgramInventoryError::SizeOverflow)?;
            let known_reads = planned
                .slots
                .input_columns
                .iter()
                .map(|&physical| {
                    value_for_physical(
                        arena,
                        values,
                        planned.component,
                        planned.part,
                        BufferPurpose::WitnessInput,
                        physical,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut known_writes = (0..planned.program.n_cols)
                .map(|ordinal| {
                    value_for_role(
                        values,
                        planned.component,
                        planned.part,
                        BufferPurpose::BaseTrace,
                        ordinal,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            if planned.program.n_lookup_words != 0 {
                known_writes.push(value_for_role(
                    values,
                    planned.component,
                    planned.part,
                    BufferPurpose::LookupInputs,
                    0,
                )?);
            }
            if planned.program.n_sub_words != 0 {
                known_writes.push(value_for_role(
                    values,
                    planned.component,
                    planned.part,
                    BufferPurpose::SubcomponentInputs,
                    0,
                )?);
            }
            let manifest = arena.protocol_identity().kernel_manifest_hash;
            if manifest == 0 {
                return Err(ArenaProgramInventoryError::InvalidArena(
                    "recorded witness has an empty AOT manifest",
                ));
            }
            (
                Some(WitnessKernelCandidate {
                    label: planned.program.label.clone(),
                    kernel_name: stwo_backend_cuda::jit_witness::codegen::witness_kernel_name(
                        semantic_hash,
                    ),
                    semantic_hash,
                    cache_key: stwo_backend_cuda::jit_witness::codegen::witness_jit_cache_key(
                        semantic_hash,
                    ),
                    aot_manifest_hash: manifest,
                }),
                Some(KernelLaunchCandidate {
                    grid: [row_count_u32.div_ceil(256), 1, 1],
                    block: [256, 1, 1],
                    dynamic_shared_bytes: 0,
                }),
                Some(WitnessPartitionCandidate {
                    row_count,
                    row_granularity: 1,
                    global_multiplicity_outputs: planned.program.n_mult_tables,
                    multiplicity_rule:
                        GlobalMultiplicityPartitionRule::CoordinatorOwnedOrCanonicalReduction,
                }),
                known_reads,
                known_writes,
                vec![
                    MissingOperationField::PrimitiveAuthority,
                    MissingOperationField::LaunchGeometry,
                    MissingOperationField::PartitionAuthority,
                    MissingOperationField::ReadValueRanges,
                    MissingOperationField::EffectContract,
                ],
            )
        }
        _ => (
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
            vec![
                MissingOperationField::PrimitiveAuthority,
                MissingOperationField::LaunchGeometry,
                MissingOperationField::PartitionAuthority,
                MissingOperationField::ReadValueRanges,
                MissingOperationField::CompleteWriteSet,
                MissingOperationField::EffectContract,
            ],
        ),
    };
    Ok(OperationFrontier {
        value: value.id,
        logical: value.logical,
        component: value.component,
        part: value.part,
        purpose: value.purpose,
        ordinal: value.ordinal,
        producer_epoch: value.lifetime.first,
        earliest_transcript_stage: CairoTranscriptSegment::BootstrapThroughBase,
        primitive,
        witness_kernel_candidate,
        launch_candidate,
        partition_candidate,
        known_reads,
        known_writes,
        missing,
    })
}

fn value_for_physical(
    arena: &ProofArenaPlan,
    values: &[ProgramValueDesc],
    component: &'static str,
    part: stwo_cairo_prover::witness::proof_shape::TracePartId,
    purpose: BufferPurpose,
    physical: stwo_backend_cuda::ArenaSlotId,
) -> Result<ArenaCatalogRange, ArenaProgramInventoryError> {
    let mut matches = values.iter().filter(|value| {
        value.component == Some(component)
            && value.part == Some(part)
            && value.purpose == purpose
            && arena
                .binding(value.logical)
                .is_some_and(|binding| binding.physical == physical)
    });
    let value = matches
        .next()
        .ok_or(ArenaProgramInventoryError::InvalidArena(
            "witness physical input has no logical value",
        ))?;
    if matches.next().is_some() {
        return Err(ArenaProgramInventoryError::InvalidArena(
            "witness physical input has multiple live logical values",
        ));
    }
    whole_value_range(value)
}

fn value_for_role(
    values: &[ProgramValueDesc],
    component: &'static str,
    part: stwo_cairo_prover::witness::proof_shape::TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
) -> Result<ArenaCatalogRange, ArenaProgramInventoryError> {
    let value = values
        .iter()
        .find(|value| {
            value.component == Some(component)
                && value.part == Some(part)
                && value.purpose == purpose
                && value.ordinal == ordinal
        })
        .ok_or(ArenaProgramInventoryError::InvalidArena(
            "witness output role has no logical value",
        ))?;
    whole_value_range(value)
}

fn whole_value_range(
    value: &ProgramValueDesc,
) -> Result<ArenaCatalogRange, ArenaProgramInventoryError> {
    let words = value
        .layout
        .element_count()
        .map_err(|_| ArenaProgramInventoryError::SizeOverflow)?;
    Ok(ArenaCatalogRange {
        value: value.id,
        value_words: 0..words,
    })
}

fn validate_binding(
    arena: &ProofArenaPlan,
    binding: ArenaBinding,
) -> Result<(), ArenaProgramInventoryError> {
    let logical = arena
        .logical_buffers()
        .get(binding.logical.0 as usize)
        .filter(|logical| logical.id == binding.logical)
        .ok_or(ArenaProgramInventoryError::MissingArenaBinding(
            binding.logical,
        ))?;
    let slot = arena.layout().slot(binding.physical).ok_or(
        ArenaProgramInventoryError::MissingArenaSlot(binding.logical),
    )?;
    if binding.len_words != logical.len_words || binding.len_words > slot.len_words {
        return Err(ArenaProgramInventoryError::BindingLength(binding.logical));
    }
    Ok(())
}

fn catalog_value_id(
    logical: LogicalBufferId,
) -> Result<ArenaCatalogValueId, ArenaProgramInventoryError> {
    Ok(ArenaCatalogValueId(logical.0))
}

fn output_ranges(layout: &ResidentProofBundleLayout) -> [Range<usize>; 8] {
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
