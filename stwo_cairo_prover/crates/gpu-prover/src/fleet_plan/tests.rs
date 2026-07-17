use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    Blake2sFriAssemblyShape, Blake2sProofAssemblyShape, Blake2sTraceAssemblyShape, TraceTreeRole,
    TranscriptOperation,
};

use super::*;
use crate::compiled_proof::*;
use crate::fleet_pow::{FleetPowPlan, FleetPowSchedule};
use crate::proof_bundle::ResidentProofBundleLayout;
use crate::shape_executable::{shape_identity_for_test, ShapeExecutableIdentity};
use crate::transcript_plan::{
    plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan, DynamicTranscriptShape,
};

mod adversarial;
mod barrier;
mod compiler;
mod composite_internal_read;
mod ipc_cursor;
mod pow;
mod runtime_view;
mod spill;
mod storage_contract;

const OP_ASSEMBLE: OpId = OpId(0);

#[derive(Clone)]
struct Fixture {
    compiled: Arc<CompiledProof>,
    shape: ShapeExecutableIdentity,
    placement: FleetPlacementInput,
    spill_value: ValueVersion,
    output_value: ValueVersion,
}

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
            pcs_config(),
            10,
            DynamicTranscriptShape {
                interaction_claim_felts: Some(5),
                oods_sampled_values_felts: Some(7),
            },
        )
        .unwrap()
    })
}

fn pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 0,
        fri_config: FriConfig::new(2, 1, 13, 2),
        lifting_log_size: Some(10),
    }
}

fn proof_assembly_shape() -> Blake2sProofAssemblyShape {
    let trace = |role, log_size, samples| Blake2sTraceAssemblyShape {
        role,
        leaf_log_size: log_size,
        query_log_size: log_size,
        oods_samples_per_column: vec![samples],
        commit_to_proof_column: vec![0],
    };
    Blake2sProofAssemblyShape {
        query_log_size: 10,
        n_queries: 13,
        trace_trees: vec![
            trace(TraceTreeRole::Preprocessed, 8, 2),
            trace(TraceTreeRole::Base, 10, 2),
            trace(TraceTreeRole::Interaction, 10, 2),
            trace(TraceTreeRole::Composition, 10, 1),
        ],
        fri_trees: vec![
            Blake2sFriAssemblyShape {
                evaluation_log_size: 10,
                cumulative_fold: 0,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 8,
                cumulative_fold: 2,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 6,
                cumulative_fold: 4,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 4,
                cumulative_fold: 6,
                outgoing_fold_step: 1,
                log_rows_per_leaf: 0,
            },
        ],
    }
}

fn host_finalizer(identity: &ProofIdentity, codec: ProofCodecIdentity) -> HostFinalizerAuthority {
    HostFinalizerAuthority::new(HostFinalizerAuthorityInput {
        bundle_codec: codec,
        assembly_shape: proof_assembly_shape(),
        pcs: pcs_config(),
        claim_codec: ClaimCodecIdentity::new(b"fleet-claim-codec-v1".to_vec()).unwrap(),
        interaction_claim_codec: InteractionClaimCodecIdentity::new(
            b"fleet-interaction-claim-codec-v1".to_vec(),
        )
        .unwrap(),
        channel_schema: ChannelSchemaIdentity::new(b"fleet-blake2s-channel-v1".to_vec()).unwrap(),
        preprocessed_schema: PreprocessedSchemaIdentity::new(b"fleet-preprocessed-v1".to_vec())
            .unwrap(),
        decoder: DirectProofDecoder::ResidentBlake2sV1,
        oods_recipe: OodsConsistencyRecipe::CairoComponentsV1,
        envelope: CairoProofEnvelope::CairoProofV1,
        proof_semantic_digest: *identity.proof_semantic_digest(),
        execution_build_digest: *identity.execution_build_digest(),
    })
}

fn range(start: usize, end: usize) -> ElementRange {
    ElementRange::new(start, end).unwrap()
}

fn value_range(version: ValueVersion, words: usize) -> ValueRange {
    ValueRange {
        version,
        elements: range(0, words),
    }
}

fn during(start: u32, end: u32) -> ScheduleRange {
    ScheduleRange::new(ScheduleStep(start), ScheduleStep(end)).unwrap()
}

fn u32_value(
    version: ValueVersion,
    words: usize,
    alignment: usize,
    origin: ValueOrigin,
    region: Region,
) -> ValueDesc {
    ValueDesc {
        version,
        layout: ValueLayout {
            element: ElementType::U32,
            axes: vec![LayoutAxis {
                tag: 0,
                extent: words,
                stride_bytes: 4,
            }],
        },
        alignment,
        origin,
        region,
    }
}

fn bound(binding: u32, value: ValueRange) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value,
    }
}

fn invocation(effect: &EffectContract) -> Option<AotInvocation> {
    let bindings = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|bound| bound.binding)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Some)
        .collect();
    Some(AotInvocation {
        arguments: vec![AotArgumentBinding {
            ordinal: 0,
            value: AotArgumentValue::DevicePointerTable(bindings),
        }],
    })
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

fn compiled_proof() -> (CompiledProof, ValueVersion, ValueVersion) {
    let mut values = Vec::new();
    let mut transcript_inputs = Vec::new();
    for requirement in transcript().inputs() {
        let id = requirement.semantic.id().unwrap();
        let version = ValueVersion(values.len() as u32);
        values.push(u32_value(
            version,
            requirement.min_words,
            4,
            ValueOrigin::ExternalInput(ExternalInputId(id.0)),
            Region::Input,
        ));
        transcript_inputs.push(TranscriptInputBinding {
            id,
            value: version,
            elements: range(0, requirement.min_words),
        });
    }

    let mut transcript_outputs = Vec::new();
    let mut challenge_values = Vec::new();
    for requirement in transcript().outputs() {
        let id = requirement.semantic.id().unwrap();
        let version = ValueVersion(values.len() as u32);
        values.push(u32_value(
            version,
            requirement.min_words,
            4,
            ValueOrigin::TranscriptOutput(id),
            Region::Dynamic,
        ));
        transcript_outputs.push(TranscriptOutputBinding {
            id,
            value: version,
            elements: range(0, requirement.min_words),
        });
        challenge_values.push((version, requirement.min_words));
    }

    let spill_value = ValueVersion(values.len() as u32);
    values.push(u32_value(
        spill_value,
        16,
        64,
        ValueOrigin::ExternalInput(ExternalInputId(1_000_000)),
        Region::Input,
    ));

    let output_layout = ResidentProofBundleLayout::new(20, 28, 4, 16, 1).unwrap();
    let output_value = ValueVersion(values.len() as u32);
    values.push(u32_value(
        output_value,
        output_layout.total_words,
        4,
        ValueOrigin::OpOutput(OP_ASSEMBLE),
        Region::Output,
    ));

    let mut accesses = challenge_values
        .iter()
        .enumerate()
        .map(|(binding, &(version, words))| EffectAccess::Read {
            source: bound(binding as u32, value_range(version, words)),
        })
        .collect::<Vec<_>>();
    accesses.push(EffectAccess::Write {
        destination: bound(
            accesses.len() as u32,
            value_range(output_value, output_layout.total_words),
        ),
    });
    let effect = EffectContract::new(accesses, vec![]).unwrap();
    let effect_id = effect.id();
    let invocation = invocation(&effect);
    let module = ModuleIdentity::new(b"fleet-test-sm89-cubin-v2".to_vec()).unwrap();
    let kernel = AotKernelAuthority::new(
        AotKernelId(1),
        module,
        b"fleet-test-proof-assembly-v2".to_vec(),
        b"fleet-test-build-v2".to_vec(),
        vec![effect_id],
    )
    .unwrap();
    let sections = ProofBundleSection::CANONICAL
        .into_iter()
        .zip(output_ranges(&output_layout))
        .map(|(section, words)| ProofOutputSection {
            section,
            value: output_value,
            elements: range(words.start, words.end),
        })
        .collect::<Vec<_>>();
    let fragments = sections
        .iter()
        .zip(output_ranges(&output_layout))
        .enumerate()
        .map(|(ordinal, (section, destination))| ProofOutputFragment {
            section: section.section,
            ordinal: ordinal as u32,
            source: ValueRange {
                version: section.value,
                elements: section.elements,
            },
            destination: range(destination.start, destination.end),
        })
        .collect();
    let identity = ProofIdentity::new(
        b"fleet-test-semantics-v2".to_vec(),
        b"fleet-test-program-v2".to_vec(),
    )
    .unwrap();
    let codec = ProofCodecIdentity::track_a_resident_bundle();
    let transcript_segments =
        CompiledTranscriptSegment::bind_plan(transcript(), &transcript_inputs, &transcript_outputs)
            .unwrap();
    let partition = PartitionAuthority::monolithic();
    let compiled = CompiledProof::compile(
        CompiledProofInput {
            host_finalizer: host_finalizer(&identity, codec.clone()),
            identity,
            fixed_values: vec![],
            module_global_initializers: vec![],
            kernels: vec![kernel],
            effects: vec![effect],
            partitions: vec![partition.clone()],
            operations: vec![OpNode {
                id: OP_ASSEMBLE,
                semantic_id: SemanticOpId(1),
                primitive: ExecutionPrimitive::AotKernel {
                    kernel: AotKernelId(1),
                    launch: LaunchGeometry {
                        grid: [1, 1, 1],
                        block: [128, 1, 1],
                        cluster: None,
                        dynamic_shared_bytes: 0,
                        cooperative: false,
                    },
                },
                invocation,
                effect: effect_id,
                partition: partition.id(),
                stage: ProofStage::AfterTranscript,
            }],
            values,
            transcript_inputs,
            transcript_outputs,
            transcript_segments,
            output: ProofOutputLayout {
                codec,
                layout: output_layout,
                sections,
                fragments,
            },
        },
        transcript(),
    )
    .unwrap();
    (compiled, spill_value, output_value)
}

fn barrier_steps() -> Vec<ScheduleStep> {
    (1..=transcript().segments().len())
        .map(|ordinal| ScheduleStep(u32::try_from(ordinal * 100).unwrap()))
        .collect()
}

fn output_release(id: TranscriptOutputId, steps: &[ScheduleStep]) -> ScheduleStep {
    let operation_index = transcript()
        .schedule()
        .operations()
        .iter()
        .position(|operation| operation_output(*operation) == Some(id))
        .unwrap();
    let segment = transcript()
        .boundaries()
        .iter()
        .find(|boundary| boundary.operation_index == operation_index)
        .unwrap()
        .segment;
    let ordinal = transcript()
        .segments()
        .iter()
        .position(|candidate| candidate.segment == segment)
        .unwrap();
    steps[ordinal]
}

fn operation_output(operation: TranscriptOperation) -> Option<TranscriptOutputId> {
    match operation {
        TranscriptOperation::DrawSecureFelt { output, .. }
        | TranscriptOperation::DrawSecureFelts { output, .. }
        | TranscriptOperation::DrawU32s { output, .. }
        | TranscriptOperation::DrawQueries { output, .. } => Some(output),
        _ => None,
    }
}

fn fixture() -> Fixture {
    let (compiled, spill_value, output_value) = compiled_proof();
    let compiled = Arc::new(compiled);
    let steps = barrier_steps();
    let final_release = steps.last().unwrap().0;
    let operation_window = during(final_release + 10, final_release + 20);
    let terminal_step = ScheduleStep(final_release + 100);
    let mut owners = Vec::new();
    let mut storages = Vec::new();
    let mut storage_bindings = Vec::new();
    let mut capacity_bytes = 0usize;
    for value in compiled.values() {
        let live = match value.origin {
            ValueOrigin::ExternalInput(_) => during(0, terminal_step.0),
            ValueOrigin::Constant(_) => during(0, terminal_step.0),
            ValueOrigin::TranscriptOutput(id) => {
                during(output_release(id, &steps).0, terminal_step.0)
            }
            ValueOrigin::OpOutput(_) => during(operation_window.start.0, terminal_step.0),
        };
        owners.push(FleetOwnerPlacement {
            value: value_range(value.version, value.layout.element_count().unwrap()),
            worker: WorkerId(0),
            live,
        });
        let bytes = value.layout.logical_bytes().unwrap();
        capacity_bytes += bytes;
        let storage = StorageId(value.version.0);
        storages.push(StorageDesc {
            id: storage,
            worker: WorkerId(0),
            bytes,
            alignment_bytes: value.alignment,
        });
        if value.version == output_value {
            storage_bindings.extend(compiled.output().sections.iter().map(|section| {
                FleetStoragePlacement {
                    storage,
                    value: ValueRange {
                        version: section.value,
                        elements: section.elements,
                    },
                    offset_bytes: section.elements.start * size_of::<u32>(),
                }
            }));
        } else {
            storage_bindings.push(FleetStoragePlacement {
                storage,
                value: value_range(value.version, value.layout.element_count().unwrap()),
                offset_bytes: 0,
            });
        }
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
    let placement = FleetPlacementInput {
        topology: FleetPlacementTopology {
            gpu_class: ConsumerGpuClass::Rtx4090Sm89,
            module_pack_identity: [7; 32],
            fixed_image_identity: [8; 32],
            coordinator: WorkerId(0),
            workers: vec![WorkerSpec {
                id: WorkerId(0),
                capacity_bytes: capacity_bytes + 1024,
                exchange_reserve_bytes: 0,
            }],
            links: vec![],
            host_numa: vec![],
        },
        pow: FleetPowSchedule {
            interaction: FleetPowPlan {
                workers_per_rank: 1,
                indices_per_attempt: 8,
            },
            query: FleetPowPlan {
                workers_per_rank: 1,
                indices_per_attempt: 8,
            },
        },
        barrier_steps: steps,
        terminal_step,
        barrier_arrivals,
        operations: vec![FleetOperationPlacement {
            operation: OP_ASSEMBLE,
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
        output_storage: StorageId(output_value.0),
    };
    let shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    Fixture {
        compiled,
        shape,
        placement,
        spill_value,
        output_value,
    }
}

fn compile(fixture: Fixture) -> Result<FleetProofPlan, FleetPlanError> {
    FleetProofPlan::lower_compiled(
        fixture.compiled,
        fixture.shape,
        fixture.placement,
        transcript(),
    )
    .map_err(|error| match error {
        FleetLoweringError::Plan(error) => error,
        other => panic!("test fixture violated identity authority: {other}"),
    })
}

#[test]
fn compiled_proof_is_the_only_semantic_authority() {
    let fixture = fixture();
    let expected = Arc::clone(&fixture.compiled);
    let plan = compile(fixture).unwrap();
    assert_eq!(plan.compiled(), expected.as_ref());
    assert_eq!(plan.placement().operations.len(), 1);
    assert_eq!(plan.workers().len(), 1);
    assert_eq!(plan.barriers().len(), transcript().segments().len());
    assert!(plan.require_real_sn_runtime().is_err());
}

#[test]
fn canonical_identity_ignores_placement_input_order() {
    let baseline = compile(fixture()).unwrap();
    let mut reordered = fixture();
    reordered.placement.owners.reverse();
    reordered.placement.storages.reverse();
    reordered.placement.storage_bindings.reverse();
    reordered.placement.barrier_arrivals.reverse();
    let reordered = compile(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );
}

#[test]
fn physical_coverage_and_output_storage_fail_closed() {
    let mut missing_owner = fixture();
    missing_owner.placement.owners[0].value.elements.end -= 1;
    assert!(matches!(
        compile(missing_owner),
        Err(FleetPlanError::OwnershipCoverage(_))
    ));

    let mut wrong_output = fixture();
    wrong_output.placement.output_storage = StorageId(wrong_output.spill_value.0);
    assert_eq!(
        compile(wrong_output).unwrap_err(),
        FleetPlanError::InvalidProofOutput(StorageId(fixture().spill_value.0))
    );
}
