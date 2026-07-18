use std::collections::BTreeSet;
use std::sync::OnceLock;

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo_cairo_gpu_prover::compiled_proof::*;
use stwo_cairo_gpu_prover::proof_bundle::ResidentProofBundleLayout;
use stwo_cairo_gpu_prover::transcript_plan::{
    plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan, CairoTranscriptSegment,
    DynamicTranscriptShape,
};

#[path = "compiled_proof_host/fixture.rs"]
mod fixture;
#[path = "compiled_proof_host/ordered_composite.rs"]
mod ordered_composite;
#[path = "compiled_proof_host/partial_atomic.rs"]
mod partial_atomic;
#[path = "compiled_proof_host/partition_authority.rs"]
mod partition_authority;
#[path = "compiled_proof_host/primitive.rs"]
mod primitive;
#[path = "compiled_proof_host/registered_fixed_source_read.rs"]
mod registered_fixed_source_read;
#[path = "compiled_proof_host/static_wrapper.rs"]
mod static_wrapper;
#[path = "compiled_proof_host/structural_authority.rs"]
mod structural_authority;

use fixture::{host_finalizer, pcs_config};

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

fn range(start: usize, end: usize) -> ElementRange {
    ElementRange::new(start, end).unwrap()
}

fn u32_value(
    version: ValueVersion,
    words: usize,
    origin: ValueOrigin,
    region: Region,
) -> ValueDesc {
    ValueDesc {
        version,
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
        region,
    }
}

fn bound(binding: u32, value: ValueRange) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value,
    }
}

fn value_range(version: ValueVersion, words: usize) -> ValueRange {
    ValueRange {
        version,
        elements: range(0, words),
    }
}

fn module() -> ModuleIdentity {
    ModuleIdentity::new(b"sm_86:resident-proof-test-cubin-v1".to_vec()).unwrap()
}

fn module_initializer(
    module: ModuleIdentity,
    symbol: &[u8],
    bytes: usize,
) -> ModuleGlobalInitializer {
    ModuleGlobalInitializer::new(
        ModuleGlobalInitializerId(0),
        module,
        symbol.to_vec(),
        bytes,
        8,
        true,
        vec![ModuleGlobalInitializerAtom::Literal {
            destination: ByteRange::new(0, bytes).unwrap(),
            bytes: vec![0; bytes].into_boxed_slice(),
        }],
    )
    .unwrap()
}

fn kernel(
    module: ModuleIdentity,
    accepted_executions: Vec<(EffectContractId, AotInvocation)>,
    build: &[u8],
) -> AotKernelAuthority {
    AotKernelAuthority::new(
        AotKernelId(1),
        module,
        b"terminal-proof-assembly-semantics-v1".to_vec(),
        build.to_vec(),
        accepted_executions
            .into_iter()
            .map(|(effect, invocation)| (effect, invocation.contract_id().unwrap()))
            .collect(),
    )
    .unwrap()
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

fn layout_ranges(layout: &ResidentProofBundleLayout) -> [std::ops::Range<usize>; 8] {
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

fn install_effect(input: &mut CompiledProofInput, contract: EffectContract) {
    let effect_id = contract.id();
    let invocation = invocation(&contract).unwrap();
    input.operations[0].invocation = Some(invocation.clone());
    input.effects = vec![contract];
    input.operations[0].effect = effect_id;
    input.kernels = vec![kernel(
        module(),
        vec![(effect_id, invocation)],
        b"assembly-build-v1",
    )];
}

fn refresh_kernel_invocation_authorities(input: &mut CompiledProofInput) {
    let kernels = input.kernels.clone();
    input.kernels = kernels
        .into_iter()
        .map(|kernel| {
            let mut accepted = Vec::new();
            for operation in &input.operations {
                let steps: Vec<_> = match &operation.primitive {
                    ExecutionPrimitive::OrderedComposite { children } => children
                        .iter()
                        .map(|child| (&child.primitive, child.invocation.as_ref(), child.effect))
                        .collect(),
                    primitive => vec![(primitive, operation.invocation.as_ref(), operation.effect)],
                };
                for (primitive, invocation, effect) in steps {
                    if matches!(
                        primitive,
                        ExecutionPrimitive::AotKernel { kernel: id, .. } if *id == kernel.id()
                    ) {
                        accepted.push((
                            effect,
                            operation.partition,
                            invocation.unwrap().contract_id().unwrap(),
                        ));
                    }
                }
            }
            accepted.sort_unstable();
            accepted.dedup();
            AotKernelAuthority::new_with_accepted_executions(
                kernel.id(),
                kernel.module().clone(),
                kernel.semantic_encoding().to_vec(),
                kernel.execution_build_encoding().to_vec(),
                accepted,
            )
            .unwrap()
        })
        .collect();
}

fn valid_input() -> CompiledProofInput {
    let transcript = transcript();
    let mut values = Vec::new();
    let mut transcript_inputs = Vec::new();
    for requirement in transcript.inputs() {
        let id = requirement.semantic.id().unwrap();
        let version = ValueVersion(values.len() as u32);
        values.push(u32_value(
            version,
            requirement.min_words,
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
    for requirement in transcript.outputs() {
        let version = ValueVersion(values.len() as u32);
        let id = requirement.semantic.id().unwrap();
        values.push(u32_value(
            version,
            requirement.min_words,
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

    let layout = ResidentProofBundleLayout::new(20, 28, 4, 16, 1).unwrap();
    let assembly = OpId(0);
    let bundle = ValueVersion(values.len() as u32);
    values.push(u32_value(
        bundle,
        layout.total_words,
        ValueOrigin::OpOutput(assembly),
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
            value_range(bundle, layout.total_words),
        ),
    });
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    let effect_id = contract.id();
    let invocation = invocation(&contract).unwrap();

    let sections = ProofBundleSection::CANONICAL
        .into_iter()
        .zip(layout_ranges(&layout))
        .map(|(section, words)| ProofOutputSection {
            section,
            value: bundle,
            elements: range(words.start, words.end),
        })
        .collect::<Vec<_>>();
    let fragments = sections
        .iter()
        .zip(layout_ranges(&layout))
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
    let identity =
        ProofIdentity::new(b"semantic-v2".to_vec(), b"program-image-v2".to_vec()).unwrap();
    let codec = ProofCodecIdentity::track_a_resident_bundle();
    let transcript_segments =
        CompiledTranscriptSegment::bind_plan(transcript, &transcript_inputs, &transcript_outputs)
            .unwrap();
    let partition = PartitionAuthority::monolithic();

    CompiledProofInput {
        host_finalizer: host_finalizer(&identity, codec.clone()),
        identity,
        fixed_values: vec![],
        module_global_initializers: vec![],
        kernels: vec![kernel(
            module(),
            vec![(effect_id, invocation.clone())],
            b"assembly-build-v1",
        )],
        static_wrappers: vec![],
        effects: vec![contract],
        partitions: vec![partition.clone()],
        operations: vec![OpNode {
            id: assembly,
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
            invocation: Some(invocation),
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
            layout,
            sections,
            fragments,
        },
    }
}

#[test]
fn complete_authority_compiles_and_retains_exact_bytes() {
    let input = valid_input();
    let compiled = CompiledProof::compile(input.clone(), transcript()).unwrap();
    let repeated = CompiledProof::compile(input.clone(), transcript()).unwrap();
    assert_eq!(compiled.input(), &input);
    assert_eq!(compiled.identity(), repeated.identity());
    assert_eq!(
        compiled.effect_for(OpId(0)).unwrap().id(),
        input.effects[0].id()
    );
    assert_eq!(compiled.kernel(AotKernelId(1)).unwrap(), &input.kernels[0]);
    assert_eq!(
        compiled
            .value(ValueVersion((input.values.len() - 1) as u32))
            .unwrap()
            .region,
        Region::Output
    );
    assert_eq!(
        compiled.transcript_encoding(),
        transcript().canonical_encoding().unwrap()
    );
    assert!(!compiled.identity().canonical_encoding().is_empty());

    let mut different_build = valid_input();
    let effect_id = different_build.effects[0].id();
    let invocation = different_build.operations[0].invocation.clone().unwrap();
    different_build.kernels = vec![kernel(
        module(),
        vec![(effect_id, invocation)],
        b"assembly-build-v2",
    )];
    let different = CompiledProof::compile(different_build, transcript()).unwrap();
    assert_ne!(compiled.identity(), different.identity());
}

#[test]
fn aot_invocation_binds_every_effect_range_to_one_exact_abi_ordinal() {
    let mut permuted = valid_input();
    let AotArgumentValue::DevicePointerTable(entries) = &mut permuted.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments[0]
        .value
    else {
        panic!("fixture must use one pointer table")
    };
    entries.swap(0, 1);
    assert_eq!(
        CompiledProof::compile(permuted, transcript()).unwrap_err(),
        CompiledProofError::KernelInvocationNotAccepted { operation: OpId(0) }
    );

    let mut missing = valid_input();
    let AotArgumentValue::DevicePointerTable(entries) =
        &mut missing.operations[0].invocation.as_mut().unwrap().arguments[0].value
    else {
        panic!("fixture must use one pointer table")
    };
    entries.pop();
    assert!(matches!(
        CompiledProof::compile(missing, transcript()),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));

    let mut duplicate = valid_input();
    let AotArgumentValue::DevicePointerTable(entries) = &mut duplicate.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments[0]
        .value
    else {
        panic!("fixture must use one pointer table")
    };
    entries[1] = entries[0];
    assert!(matches!(
        CompiledProof::compile(duplicate, transcript()),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));

    let mut wrong_ordinal = valid_input();
    wrong_ordinal.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments[0]
        .ordinal = 1;
    assert!(matches!(
        CompiledProof::compile(wrong_ordinal, transcript()),
        Err(CompiledProofError::InvalidKernelInvocation(OpId(0)))
    ));
}

#[test]
fn aot_invocation_authority_rejects_shape_valid_scalar_drift() {
    let mut input = valid_input();
    input.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments
        .push(AotArgumentBinding {
            ordinal: 1,
            value: AotArgumentValue::U32(7),
        });
    let effect = input.operations[0].effect;
    let accepted = input.operations[0].invocation.clone().unwrap();
    input.kernels = vec![kernel(
        module(),
        vec![(effect, accepted)],
        b"assembly-build-v1",
    )];
    input.operations[0].invocation.as_mut().unwrap().arguments[1].value = AotArgumentValue::U32(8);
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::KernelInvocationNotAccepted { operation: OpId(0) }
    );
}

#[test]
fn kernel_execution_inventory_rejects_surplus_invocation_authority() {
    let mut input = valid_input();
    let kernel = input.kernels[0].clone();
    let mut surplus = input.operations[0].invocation.clone().unwrap();
    surplus.arguments.push(AotArgumentBinding {
        ordinal: u8::try_from(surplus.arguments.len()).unwrap(),
        value: AotArgumentValue::U32(7),
    });
    let mut accepted = kernel.accepted_executions().to_vec();
    accepted.push((
        input.operations[0].effect,
        input.operations[0].partition,
        surplus.contract_id().unwrap(),
    ));
    accepted.sort_unstable();
    input.kernels = vec![
        AotKernelAuthority::new_with_accepted_executions(
            kernel.id(),
            kernel.module().clone(),
            kernel.semantic_encoding().to_vec(),
            kernel.execution_build_encoding().to_vec(),
            accepted,
        )
        .unwrap(),
    ];

    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::NonCanonicalKernelEffects(AotKernelId(1))
    );
}

#[test]
fn host_usize_argument_is_canonical_u64_identity() {
    let with_size = |value| {
        let mut input = valid_input();
        input.operations[0]
            .invocation
            .as_mut()
            .unwrap()
            .arguments
            .push(AotArgumentBinding {
                ordinal: 1,
                value: AotArgumentValue::Usize(value),
            });
        let effect = input.operations[0].effect;
        let invocation = input.operations[0].invocation.clone().unwrap();
        input.kernels = vec![kernel(
            module(),
            vec![(effect, invocation)],
            b"assembly-build-v1",
        )];
        CompiledProof::compile(input, transcript()).unwrap()
    };

    let first = with_size(u64::from(u32::MAX) + 1);
    let repeated = with_size(u64::from(u32::MAX) + 1);
    let different = with_size(u64::from(u32::MAX) + 2);
    assert_eq!(first.identity(), repeated.identity());
    assert_ne!(first.identity(), different.identity());
}

#[test]
fn effect_and_kernel_authorities_are_body_derived_and_canonical() {
    assert!(matches!(
        ModuleIdentity::new(vec![]),
        Err(CompiledProofError::EmptyModuleIdentity)
    ));
    let source = ValueRange {
        version: ValueVersion(0),
        elements: range(0, 4),
    };
    let canonical = EffectContract::new(
        vec![EffectAccess::Read {
            source: bound(0, source),
        }],
        vec![],
    )
    .unwrap();
    let repeated = EffectContract::new(
        vec![EffectAccess::Read {
            source: bound(0, source),
        }],
        vec![],
    )
    .unwrap();
    assert_eq!(canonical.id(), repeated.id());
    assert_eq!(
        canonical.canonical_encoding(),
        repeated.canonical_encoding()
    );

    assert!(matches!(
        EffectContract::new(
            vec![EffectAccess::Read {
                source: bound(1, source),
            }],
            vec![],
        ),
        Err(CompiledProofError::NonCanonicalEffectBindings)
    ));
    assert!(matches!(
        EffectContract::new(
            vec![
                EffectAccess::Read {
                    source: bound(0, source),
                },
                EffectAccess::Write {
                    destination: bound(
                        0,
                        ValueRange {
                            version: ValueVersion(1),
                            elements: source.elements,
                        },
                    ),
                },
            ],
            vec![],
        ),
        Err(CompiledProofError::NonCanonicalEffectBindings)
    ));
    assert!(matches!(
        AotKernelAuthority::new(
            AotKernelId(1),
            module(),
            b"semantics".to_vec(),
            b"build".to_vec(),
            vec![],
        ),
        Err(CompiledProofError::EmptyKernelEffectAuthority(AotKernelId(
            1
        )))
    ));
}

#[test]
fn paired_read_write_and_atomic_accesses_require_distinct_versions() {
    for atomic in [false, true] {
        let mut input = valid_input();
        let bundle = input.output.sections[0].value;
        let words = input.output.layout.total_words;
        let source = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            source,
            words,
            ValueOrigin::ExternalInput(ExternalInputId(10_000)),
            Region::Input,
        ));
        let mut accesses =
            input.effects[0].accesses()[..input.effects[0].accesses().len() - 1].to_vec();
        let binding = accesses.len() as u32;
        let access = if atomic {
            EffectAccess::Atomic {
                source: bound(binding, value_range(source, words)),
                destination: bound(binding, value_range(bundle, words)),
                operation: AtomicOperation::AddU32,
                in_place: InPlaceAliasAuthority {
                    id: InPlaceAliasId(0),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
                },
            }
        } else {
            EffectAccess::ReadWrite {
                source: bound(binding, value_range(source, words)),
                destination: bound(binding + 1, value_range(bundle, words)),
                in_place: Some(InPlaceAliasAuthority {
                    id: InPlaceAliasId(0),
                    requirement: InPlaceAliasRequirement::Permitted,
                    discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
                }),
            }
        };
        accesses.push(access);
        let contract = EffectContract::new(accesses, vec![]).unwrap();
        install_effect(&mut input, contract);
        let compiled = CompiledProof::compile(input, transcript()).unwrap();
        assert!(compiled.effects()[0]
            .in_place_alias(InPlaceAliasId(0))
            .is_some());
    }

    let same = value_range(ValueVersion(0), 1);
    assert!(matches!(
        EffectContract::new(
            vec![EffectAccess::ReadWrite {
                source: bound(0, same),
                destination: bound(1, same),
                in_place: None,
            }],
            vec![],
        ),
        Err(CompiledProofError::InvalidValueTransition)
    ));

    let mut wide = valid_input();
    let bundle = wide.output.sections[0].value;
    let words = wide.output.layout.total_words;
    let source = ValueVersion(wide.values.len() as u32);
    wide.values.push(u32_value(
        source,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(10_001)),
        Region::Input,
    ));
    for version in [source, bundle] {
        wide.values[version.0 as usize].layout.element = ElementType { tag: 2, bytes: 8 };
        wide.values[version.0 as usize].layout.axes[0].stride_bytes = 8;
    }
    let mut accesses = wide.effects[0].accesses()[..wide.effects[0].accesses().len() - 1].to_vec();
    let binding = accesses.len() as u32;
    accesses.push(EffectAccess::Atomic {
        source: bound(binding, value_range(source, words)),
        destination: bound(binding, value_range(bundle, words)),
        operation: AtomicOperation::AddU32,
        in_place: InPlaceAliasAuthority {
            id: InPlaceAliasId(0),
            requirement: InPlaceAliasRequirement::Required,
            discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
        },
    });
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    install_effect(&mut wide, contract);
    assert!(matches!(
        CompiledProof::compile(wide, transcript()),
        Err(CompiledProofError::InvalidValueTransition)
    ));
}

#[test]
fn rejects_unknown_reads_incomplete_writes_and_unaccepted_effects() {
    let mut unknown = valid_input();
    let mut accesses = unknown.effects[0].accesses().to_vec();
    accesses[0] = EffectAccess::Read {
        source: bound(0, value_range(ValueVersion(u32::MAX), 1)),
    };
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    install_effect(&mut unknown, contract);
    assert!(matches!(
        CompiledProof::compile(unknown, transcript()),
        Err(CompiledProofError::UnknownValue { .. })
    ));

    let mut incomplete = valid_input();
    let bundle = incomplete.output.sections[0].value;
    let words = incomplete.output.layout.total_words;
    let mut accesses = incomplete.effects[0].accesses().to_vec();
    *accesses.last_mut().unwrap() = EffectAccess::Write {
        destination: bound(
            accesses.len() as u32 - 1,
            ValueRange {
                version: bundle,
                elements: range(0, words - 1),
            },
        ),
    };
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    install_effect(&mut incomplete, contract);
    assert!(matches!(
        CompiledProof::compile(incomplete, transcript()),
        Err(CompiledProofError::IncompleteWrite { value }) if value == bundle
    ));

    let mut unaccepted = valid_input();
    unaccepted.module_global_initializers = vec![module_initializer(module(), b"X", 4)];
    let other = EffectContract::new(
        unaccepted.effects[0].accesses().to_vec(),
        vec![ModuleGlobalEffect {
            initializer: ModuleGlobalInitializerId(0),
            bytes: ByteRange::new(0, 4).unwrap(),
        }],
    )
    .unwrap();
    unaccepted.operations[0].effect = other.id();
    unaccepted.effects = vec![other];
    assert!(matches!(
        CompiledProof::compile(unaccepted, transcript()),
        Err(CompiledProofError::KernelEffectNotAccepted { .. })
    ));
}

#[test]
fn duplicate_external_origin_identity_is_rejected() {
    let mut input = valid_input();
    let mut duplicate = input.values[0].clone();
    duplicate.version = ValueVersion(input.values.len() as u32);
    let duplicate_version = duplicate.version;
    input.values.push(duplicate);
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidValue { value }) if value == duplicate_version
    ));
}

#[test]
fn value_origin_and_region_must_be_compatible() {
    let mut input = valid_input();
    input.values[0].region = Region::Dynamic;
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidValue {
            value: ValueVersion(0)
        })
    ));
}

#[test]
fn rejects_transcript_omissions_ranges_and_early_consumers() {
    let mut omitted = valid_input();
    omitted.transcript_inputs.pop();
    assert!(matches!(
        CompiledProof::compile(omitted, transcript()),
        Err(CompiledProofError::TranscriptInputCount { .. })
    ));

    let mut reordered = valid_input();
    reordered.transcript_inputs.swap(0, 1);
    assert!(matches!(
        CompiledProof::compile(reordered, transcript()),
        Err(CompiledProofError::TranscriptBindingOrder {
            kind: BindingKind::TranscriptInput,
            index: 0,
        })
    ));

    let mut truncated = valid_input();
    truncated.transcript_inputs[0].elements.end -= 1;
    assert!(matches!(
        CompiledProof::compile(truncated, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptInput,
            ..
        })
    ));

    let mut wrong_word_tag = valid_input();
    wrong_word_tag.values[0].layout.element.tag = 7;
    assert!(matches!(
        CompiledProof::compile(wrong_word_tag, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptInput,
            ..
        })
    ));

    let mut early = valid_input();
    early.operations[0].stage =
        ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase);
    assert!(matches!(
        CompiledProof::compile(early, transcript()),
        Err(CompiledProofError::TranscriptCausality { .. })
    ));
}

#[test]
fn rejects_noncanonical_or_overlapping_proof_output() {
    let mut reordered = valid_input();
    reordered.output.sections.swap(0, 1);
    assert!(matches!(
        CompiledProof::compile(reordered, transcript()),
        Err(CompiledProofError::ProofSectionOrder { index: 0 })
    ));

    let mut gap = valid_input();
    gap.output.sections[1].elements.start += 1;
    gap.output.sections[1].elements.end += 1;
    gap.output.fragments[1].source.elements = gap.output.sections[1].elements;
    assert!(matches!(
        CompiledProof::compile(gap, transcript()),
        Err(CompiledProofError::InvalidProofAssembly)
    ));

    let mut overlap = valid_input();
    overlap.output.sections[4].elements = overlap.output.sections[0].elements;
    overlap.output.fragments[4].source.elements = overlap.output.sections[4].elements;
    let result = CompiledProof::compile(overlap, transcript());
    assert!(
        matches!(result, Err(CompiledProofError::InvalidProofAssembly)),
        "{result:?}"
    );
}
