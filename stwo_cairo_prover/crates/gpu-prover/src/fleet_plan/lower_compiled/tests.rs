use std::sync::OnceLock;

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;

use super::*;
use crate::compiled_proof::{
    AotKernelId, CompiledProofInput, EffectContractId, ExternalInputId, OpId, OpNode,
    ProofBundleSection, ProofCodecIdentity, ProofIdentity, ProofOutputLayout, ProofOutputSection,
    SemanticAuthority, SemanticOpId, TranscriptInputBinding, TranscriptOutputBinding,
    ValueDesc as CompiledValue, ValueId as CompiledValueId, ValueOrigin as CompiledOrigin,
};
use crate::fleet_pow::{FleetPowPlan, FleetPowSchedule};
use crate::proof_bundle::ResidentProofBundleLayout;
use crate::transcript_plan::{
    plan_cairo_blake2s_transcript, CairoTranscriptOutput, DynamicTranscriptShape,
};

fn transcript() -> &'static CairoBlake2sTranscriptPlan {
    static PLAN: OnceLock<CairoBlake2sTranscriptPlan> = OnceLock::new();
    PLAN.get_or_init(|| {
        let claim: CairoClaim = serde_json::from_value(serde_json::json!({
            "public_data": PublicData::default(),
            "add_opcode": { "log_size": 4 },
            "memory_id_to_big": { "big_log_sizes": [] }
        }))
        .unwrap();
        plan_cairo_blake2s_transcript(
            &claim,
            PcsConfig {
                pow_bits: 0,
                fri_config: FriConfig::new(2, 1, 13, 2),
                lifting_log_size: Some(10),
            },
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(5),
                oods_sampled_values_felts: Some(7),
            },
        )
        .unwrap()
    })
}

fn u32_value(
    id: CompiledValueId,
    words: usize,
    origin: CompiledOrigin,
    consumers: Vec<OpId>,
) -> CompiledValue {
    CompiledValue {
        id,
        layout: ValueLayout {
            element: ElementType { tag: 1, bytes: 4 },
            axes: vec![LayoutAxis {
                tag: 0,
                extent: words,
                stride_bytes: 4,
            }],
        },
        alignment: 4,
        origin,
        consumers,
    }
}

fn output_ranges(layout: &ResidentProofBundleLayout) -> [std::ops::Range<usize>; 8] {
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

fn compiled_proof() -> CompiledProof {
    let mut values = Vec::new();
    let mut transcript_inputs = Vec::new();
    for requirement in transcript().inputs() {
        let id = requirement.semantic.id().unwrap();
        let value = CompiledValueId(values.len() as u32);
        values.push(u32_value(
            value,
            requirement.min_words,
            CompiledOrigin::ExternalInput(ExternalInputId(id.0)),
            vec![],
        ));
        transcript_inputs.push(TranscriptInputBinding {
            id,
            value,
            value_words: 0..requirement.min_words,
        });
    }

    let mut transcript_outputs = Vec::new();
    let mut challenge_values = Vec::new();
    for requirement in transcript().outputs() {
        let id = requirement.semantic.id().unwrap();
        let value = CompiledValueId(values.len() as u32);
        values.push(u32_value(
            value,
            requirement.min_words,
            CompiledOrigin::TranscriptOutput(id),
            vec![],
        ));
        transcript_outputs.push(TranscriptOutputBinding {
            id,
            value,
            value_words: 0..requirement.min_words,
        });
        challenge_values.push(value);
    }

    let assembly = OpId(0);
    for &value in &challenge_values {
        values[value.0 as usize].consumers.push(assembly);
    }
    let layout = ResidentProofBundleLayout::new(4, 4, 1, 4, 1).unwrap();
    let bundle = CompiledValueId(values.len() as u32);
    values.push(u32_value(
        bundle,
        layout.total_words,
        CompiledOrigin::OpOutput(assembly),
        vec![],
    ));
    let operations = vec![OpNode {
        id: assembly,
        semantic_id: SemanticOpId(11),
        kernel_id: AotKernelId(22),
        effects: EffectContractId([33; 32]),
        inputs: challenge_values,
        outputs: vec![bundle],
        stage: ProofStage::AfterTranscript,
    }];
    let sections = ProofBundleSection::CANONICAL
        .into_iter()
        .zip(output_ranges(&layout))
        .map(|(section, value_words)| ProofOutputSection {
            section,
            value: bundle,
            value_words,
        })
        .collect();
    CompiledProof::compile(
        CompiledProofInput {
            identity: ProofIdentity::new(b"semantics-v1".to_vec(), b"aot-v1".to_vec()).unwrap(),
            authority: SemanticAuthority {
                operations: vec![SemanticOpId(11)],
                kernels: vec![AotKernelId(22)],
                effects: vec![EffectContractId([33; 32])],
            },
            operations,
            values,
            transcript_inputs,
            transcript_outputs,
            output: ProofOutputLayout {
                codec: ProofCodecIdentity::track_a_resident_bundle(),
                layout,
                sections,
            },
        },
        transcript(),
    )
    .unwrap()
}

fn schedule_range(start: u32, end: u32) -> ScheduleRange {
    ScheduleRange::new(ScheduleStep(start), ScheduleStep(end)).unwrap()
}

fn barrier_steps() -> Vec<ScheduleStep> {
    (1..=transcript().segments().len())
        .map(|ordinal| ScheduleStep((ordinal * 100) as u32))
        .collect()
}

fn output_release(id: TranscriptOutputId, steps: &[ScheduleStep]) -> ScheduleStep {
    let output = transcript()
        .outputs()
        .iter()
        .find(|requirement| requirement.semantic.id() == Ok(id))
        .unwrap()
        .semantic;
    let segment = match output {
        CairoTranscriptOutput::CommonLookupElements => {
            CairoTranscriptSegment::InteractionPowAndLookup
        }
        CairoTranscriptOutput::CompositionRandomCoefficient => {
            CairoTranscriptSegment::InteractionAndComposition
        }
        CairoTranscriptOutput::OodsPointParameter => CairoTranscriptSegment::CompositionAndOods,
        CairoTranscriptOutput::QuotientRandomCoefficient => CairoTranscriptSegment::OodsAndQuotient,
        CairoTranscriptOutput::FriFoldingChallenge(layer) => {
            CairoTranscriptSegment::FriLayer(layer)
        }
        CairoTranscriptOutput::QueryPositions => CairoTranscriptSegment::QueryPowAndPositions,
    };
    let ordinal = transcript()
        .segments()
        .iter()
        .position(|candidate| candidate.segment == segment)
        .unwrap();
    steps[ordinal]
}

fn placement(compiled: &CompiledProof) -> FleetPlacementInput {
    let steps = barrier_steps();
    let final_release = steps.last().unwrap().0;
    let operation_window = schedule_range(final_release + 1, final_release + 2);
    let terminal_step = ScheduleStep(final_release + 100);
    let mut owners = Vec::new();
    let mut storages = Vec::new();
    let mut storage_bindings = Vec::new();
    let mut reserved = 0usize;
    for value in compiled.values() {
        let (ready_at, live) = match value.origin {
            CompiledOrigin::ExternalInput(_) | CompiledOrigin::Constant(_) => {
                (ScheduleStep(0), schedule_range(0, terminal_step.0))
            }
            CompiledOrigin::TranscriptOutput(id) => {
                let release = output_release(id, &steps);
                (release, schedule_range(release.0, terminal_step.0))
            }
            CompiledOrigin::OpOutput(_) => (
                operation_window.end,
                schedule_range(operation_window.start.0, terminal_step.0),
            ),
        };
        owners.push(FleetOwnerPlacement {
            value: value.id,
            worker: WorkerId(0),
            ready_at,
            live,
        });
        let bytes = value.layout.logical_bytes().unwrap();
        reserved += bytes;
        storages.push(StorageDesc {
            id: StorageId(value.id.0),
            worker: WorkerId(0),
            bytes,
            alignment_bytes: value.alignment,
        });
        storage_bindings.push(FleetStoragePlacement {
            storage: StorageId(value.id.0),
            value: value.id,
            worker: WorkerId(0),
            offset_bytes: 0,
        });
    }
    let mut barrier_arrivals = steps
        .iter()
        .enumerate()
        .map(|(ordinal, release)| BarrierArrival {
            barrier_ordinal: ordinal as u32,
            worker: WorkerId(0),
            ready_step: ScheduleStep(release.0 - 1),
        })
        .collect::<Vec<_>>();
    barrier_arrivals.push(BarrierArrival {
        barrier_ordinal: steps.len() as u32,
        worker: WorkerId(0),
        ready_step: operation_window.end,
    });
    FleetPlacementInput {
        topology: FleetPlacementTopology {
            gpu_class: ConsumerGpuClass::Rtx4090Sm89,
            module_pack_identity: [7; 32],
            fixed_image_identity: [8; 32],
            coordinator: WorkerId(0),
            workers: vec![WorkerSpec {
                id: WorkerId(0),
                capacity_bytes: reserved + 1024,
                exchange_reserve_bytes: 0,
            }],
            links: vec![],
            host_numa: vec![],
        },
        pow: FleetPowSchedule {
            interaction: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
            query: FleetPowPlan {
                workers_per_rank: 2,
                indices_per_attempt: 8,
            },
        },
        barrier_steps: steps,
        terminal_step,
        barrier_arrivals,
        operations: vec![FleetOperationPlacement {
            operation: OpId(0),
            worker: WorkerId(0),
            during: operation_window,
        }],
        owners,
        replicas: vec![],
        transitions: vec![],
        spills: vec![],
        storages,
        storage_bindings,
        in_place_aliases: vec![],
    }
}

#[test]
fn semantic_projection_is_derived_exactly_from_compiled_proof() {
    let compiled = compiled_proof();
    let placement = placement(&compiled);
    let operation_placements = operation_placements(&compiled, &placement.operations).unwrap();
    let operations = lower_operations(&compiled, transcript(), &operation_placements).unwrap();
    let operation = &operations[0];
    let source = &compiled.operations()[0];
    assert_eq!(operation.effect_identity, source.effects);
    assert_eq!(operation.semantic, operation_reference(source));
    assert_eq!(operation.interval, ExecutionInterval::AfterFinalBarrier);
    assert_eq!(operation.reads.len(), source.inputs.len());
    for (read, id) in operation.reads.iter().zip(&source.inputs) {
        let value = &compiled.values()[id.0 as usize];
        assert_eq!(read.value, ValueId(id.0));
        assert_eq!(read.elements, whole_range(value).unwrap());
        assert_eq!(read.layout, value.layout);
    }
    assert_eq!(
        lower_transcript_inputs(&compiled),
        compiled
            .input()
            .transcript_inputs
            .iter()
            .map(|binding| TranscriptInputValueBinding {
                id: binding.id,
                value: ValueId(binding.value.0),
                elements: ElementRange {
                    start: binding.value_words.start,
                    end: binding.value_words.end,
                },
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn unknown_duplicate_and_missing_placement_ids_fail_closed() {
    let compiled = compiled_proof();
    let valid = placement(&compiled);
    let mut unknown = valid.operations.clone();
    unknown[0].operation = OpId(99);
    assert!(matches!(
        operation_placements(&compiled, &unknown),
        Err(FleetLoweringError::UnknownOperation(OpId(99)))
    ));

    let mut duplicate = valid.operations.clone();
    duplicate.push(duplicate[0]);
    assert!(matches!(
        operation_placements(&compiled, &duplicate),
        Err(FleetLoweringError::DuplicateOperationPlacement(OpId(0)))
    ));
    assert!(matches!(
        operation_placements(&compiled, &[]),
        Err(FleetLoweringError::MissingOperationPlacement(OpId(0)))
    ));

    let missing_value = valid.owners[0].value;
    let owners = valid.owners[1..].to_vec();
    assert!(matches!(
        owner_placements(&compiled, &owners),
        Err(FleetLoweringError::MissingOwnerPlacement(id)) if id == missing_value
    ));
    assert!(matches!(
        lower_storage(
            &compiled,
            vec![FleetStoragePlacement {
                storage: StorageId(0),
                value: CompiledValueId(999),
                worker: WorkerId(0),
                offset_bytes: 0,
            }]
        ),
        Err(FleetLoweringError::UnknownValue(CompiledValueId(999)))
    ));
}

#[test]
fn physical_ranges_layouts_bytes_and_effects_are_not_caller_fields() {
    let compiled = compiled_proof();
    let value = &compiled.values()[0];
    let storage = lower_storage(
        &compiled,
        vec![FleetStoragePlacement {
            storage: StorageId(0),
            value: value.id,
            worker: WorkerId(0),
            offset_bytes: 64,
        }],
    )
    .unwrap();
    assert_eq!(storage[0].elements, whole_range(value).unwrap());
    assert_eq!(storage[0].bytes, value.layout.logical_bytes().unwrap());

    let transition = lower_transitions(
        &compiled,
        vec![FleetTransitionPlacement {
            id: LayoutTransitionId(0),
            value: value.id,
            source_worker: WorkerId(0),
            destination_replica: ReplicaId(0),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: schedule_range(1, 2),
            scratch_bytes: 0,
            scratch_worker: WorkerId(0),
            route: FleetLinkId(0),
        }],
    )
    .unwrap();
    assert_eq!(transition[0].source_layout, value.layout);
    assert_eq!(transition[0].destination_layout, value.layout);
    assert_eq!(transition[0].bytes, value.layout.logical_bytes().unwrap());
    assert_eq!(transition[0].axes[0].source, value.layout.axes[0].tag);

    let output = compiled.operations()[0].outputs[0];
    assert!(matches!(
        lower_aliases(
            &compiled,
            vec![InPlaceAliasPlacement {
                operation: OpId(0),
                source: value.id,
                destination: output,
                storage: StorageId(0),
                offset_bytes: 0,
            }]
        ),
        Err(FleetLoweringError::AliasSizeMismatch { .. })
    ));
}

fn shape_identity(compiled: &CompiledProof, transcript_encoding: &[u8]) -> ShapeExecutableIdentity {
    crate::shape_executable::shape_identity_for_test(
        b"synthetic-topology",
        b"synthetic-workspace",
        transcript_encoding,
        compiled.identity().canonical_encoding(),
    )
    .unwrap()
}

#[test]
fn complete_lowering_binds_full_shape_bytes_and_stays_runtime_fenced() {
    let compiled = compiled_proof();
    let transcript_encoding = transcript().canonical_encoding().unwrap();
    let shape = shape_identity(&compiled, &transcript_encoding);
    let plan =
        FleetProofPlan::lower_compiled(&compiled, &shape, placement(&compiled), transcript())
            .unwrap();

    assert_eq!(
        plan.input().topology.executable_identity,
        shape.canonical_encoding()
    );
    assert_eq!(
        plan.input().values,
        lower_values(&compiled),
        "the caller never supplies semantic values or layouts"
    );
    assert_eq!(
        plan.input().transcript_inputs,
        lower_transcript_inputs(&compiled)
    );
    assert_eq!(
        plan.input().transcript_outputs,
        lower_transcript_outputs(&compiled)
    );
    assert_eq!(
        plan.input().operations[0].effect_identity,
        EffectContractId([33; 32])
    );
    assert_eq!(
        plan.require_real_sn_runtime(),
        Err(FleetRuntimeAdmissionError::MissingTypedExecutionPrimitives)
    );
    plan.validate(transcript()).unwrap();
}

#[test]
fn semantic_value_layout_effect_and_transcript_binding_drift_rejects_exact_shape() {
    let baseline = compiled_proof();
    let transcript_encoding = transcript().canonical_encoding().unwrap();
    let shape = shape_identity(&baseline, &transcript_encoding);
    let rejects = |input: CompiledProofInput| {
        let changed = CompiledProof::compile(input, transcript()).unwrap();
        assert!(matches!(
            FleetProofPlan::lower_compiled(&changed, &shape, placement(&changed), transcript()),
            Err(FleetLoweringError::ShapeCompiledProofMismatch)
        ));
    };

    let mut semantic = baseline.input().clone();
    semantic.operations[0].semantic_id = SemanticOpId(12);
    semantic.authority.operations = vec![SemanticOpId(12)];
    rejects(semantic);

    let mut effect = baseline.input().clone();
    effect.operations[0].effects = EffectContractId([34; 32]);
    effect.authority.effects = vec![EffectContractId([34; 32])];
    rejects(effect);

    let mut layout = baseline.input().clone();
    layout.values[0].layout.element.tag += 1;
    rejects(layout);

    let mut bindings = baseline.input().clone();
    let pair = bindings
        .transcript_inputs
        .iter()
        .enumerate()
        .find_map(|(left, binding)| {
            bindings.transcript_inputs[left + 1..]
                .iter()
                .position(|candidate| candidate.value_words.len() == binding.value_words.len())
                .map(|right| (left, left + right + 1))
        })
        .unwrap();
    let left = bindings.transcript_inputs[pair.0].value;
    bindings.transcript_inputs[pair.0].value = bindings.transcript_inputs[pair.1].value;
    bindings.transcript_inputs[pair.1].value = left;
    rejects(bindings);
}

#[test]
fn shape_transcript_bytes_must_match_even_when_compiled_bytes_match() {
    let compiled = compiled_proof();
    let mut wrong_transcript = transcript().canonical_encoding().unwrap();
    wrong_transcript.push(0xff);
    let shape = shape_identity(&compiled, &wrong_transcript);
    assert!(matches!(
        FleetProofPlan::lower_compiled(&compiled, &shape, placement(&compiled), transcript()),
        Err(FleetLoweringError::ShapeTranscriptMismatch)
    ));
}
