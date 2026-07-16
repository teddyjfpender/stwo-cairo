use std::sync::OnceLock;

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_gpu_prover::compiled_proof::*;
use stwo_cairo_gpu_prover::proof_bundle::ResidentProofBundleLayout;
use stwo_cairo_gpu_prover::transcript_plan::{
    plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan, CairoTranscriptInput,
    CairoTranscriptOutput, CairoTranscriptSegment, DynamicTranscriptShape,
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

fn u32_value(id: ValueId, words: usize, origin: ValueOrigin, consumers: Vec<OpId>) -> ValueDesc {
    ValueDesc {
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

fn valid_input() -> CompiledProofInput {
    let transcript = transcript();
    let mut values = Vec::new();
    let mut transcript_inputs = Vec::new();
    for requirement in transcript.inputs() {
        let id = requirement.semantic.id().unwrap();
        let value = ValueId(values.len() as u32);
        values.push(u32_value(
            value,
            requirement.min_words,
            ValueOrigin::ExternalInput(ExternalInputId(id.0)),
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
    for requirement in transcript.outputs() {
        let value = ValueId(values.len() as u32);
        let id = requirement.semantic.id().unwrap();
        values.push(u32_value(
            value,
            requirement.min_words,
            ValueOrigin::TranscriptOutput(id),
            vec![],
        ));
        transcript_outputs.push(TranscriptOutputBinding {
            id,
            value,
            value_words: 0..requirement.min_words,
        });
        challenge_values.push(value);
    }

    let layout = ResidentProofBundleLayout::new(4, 4, 1, 4, 1).unwrap();
    let assembly_op = OpId(0);
    for &value in &challenge_values {
        values[value.0 as usize].consumers.push(assembly_op);
    }
    let bundle = ValueId(values.len() as u32);
    values.push(u32_value(
        bundle,
        layout.total_words,
        ValueOrigin::OpOutput(assembly_op),
        vec![],
    ));
    let operations = vec![OpNode {
        id: assembly_op,
        semantic_id: SemanticOpId(1),
        kernel_id: AotKernelId(2),
        effects: EffectContractId([2; 32]),
        inputs: challenge_values,
        outputs: vec![bundle],
        stage: ProofStage::AfterTranscript,
    }];

    let sections = ProofBundleSection::CANONICAL
        .into_iter()
        .zip(layout_ranges(&layout))
        .map(|(section, value_words)| ProofOutputSection {
            section,
            value: bundle,
            value_words,
        })
        .collect();
    CompiledProofInput {
        identity: ProofIdentity::new(b"semantic-v1".to_vec(), b"aot-build-v1".to_vec()).unwrap(),
        authority: SemanticAuthority {
            operations: operations
                .iter()
                .map(|operation| operation.semantic_id)
                .collect(),
            kernels: vec![AotKernelId(2)],
            effects: vec![EffectContractId([2; 32])],
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
    }
}

#[test]
fn complete_authority_compiles_and_identities_are_domain_separated() {
    let input = valid_input();
    assert_ne!(
        input.identity.proof_semantic_digest(),
        input.identity.program_image_digest()
    );
    assert_ne!(
        input.identity.program_image_digest(),
        input.output.codec.digest()
    );
    let compiled = CompiledProof::compile(input.clone(), transcript()).unwrap();
    let repeated = CompiledProof::compile(input.clone(), transcript()).unwrap();
    assert_eq!(compiled.input(), &input);
    assert_eq!(compiled.identity(), repeated.identity());
    assert!(!compiled.identity().canonical_encoding().is_empty());
    assert_eq!(
        compiled.transcript_encoding(),
        transcript().canonical_encoding().unwrap()
    );
    assert_eq!(compiled.output().layout.total_words, 57);

    let mut different_kernel = valid_input();
    different_kernel.operations.last_mut().unwrap().kernel_id = AotKernelId(3);
    different_kernel.authority.kernels = vec![AotKernelId(3)];
    let different = CompiledProof::compile(different_kernel, transcript()).unwrap();
    assert_ne!(compiled.identity(), different.identity());
}

#[test]
fn rejects_authority_and_graph_omissions_or_duplicates() {
    let mut missing_authority = valid_input();
    missing_authority.authority.effects.pop();
    assert!(matches!(
        CompiledProof::compile(missing_authority, transcript()),
        Err(CompiledProofError::NonCanonicalAuthority(
            AuthorityKind::EffectContract
        ))
    ));

    let mut duplicate_edge = valid_input();
    let assembly = duplicate_edge.operations.last_mut().unwrap();
    assembly.inputs.push(assembly.inputs[0]);
    assert!(matches!(
        CompiledProof::compile(duplicate_edge, transcript()),
        Err(CompiledProofError::DuplicateOperationEdge { .. })
    ));

    let mut missing_consumer = valid_input();
    let challenge = missing_consumer.transcript_outputs[0].value;
    missing_consumer.values[challenge.0 as usize]
        .consumers
        .clear();
    assert!(matches!(
        CompiledProof::compile(missing_consumer, transcript()),
        Err(CompiledProofError::ConsumerMismatch { .. })
    ));

    let mut wrong_producer = valid_input();
    let produced = wrong_producer.operations[0].outputs[0];
    wrong_producer.values[produced.0 as usize].origin =
        ValueOrigin::ExternalInput(ExternalInputId(99));
    assert!(matches!(
        CompiledProof::compile(wrong_producer, transcript()),
        Err(CompiledProofError::ProducerMismatch { .. })
    ));

    let mut zero_effect = valid_input();
    zero_effect.operations[0].effects = EffectContractId([0; 32]);
    zero_effect.authority.effects = vec![EffectContractId([0; 32])];
    assert!(matches!(
        CompiledProof::compile(zero_effect, transcript()),
        Err(CompiledProofError::NonCanonicalAuthority(
            AuthorityKind::EffectContract
        ))
    ));
}

#[test]
fn rejects_transcript_omissions_order_ranges_and_early_consumers() {
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
            ..
        })
    ));

    let mut truncated = valid_input();
    truncated.transcript_outputs[0].value_words.end -= 1;
    assert!(matches!(
        CompiledProof::compile(truncated, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptOutput,
            ..
        })
    ));

    let mut byte_elements = valid_input();
    let input_value = byte_elements.transcript_inputs[0].value;
    let value = &mut byte_elements.values[input_value.0 as usize];
    value.layout.element.bytes = 1;
    value.layout.axes[0].stride_bytes = 1;
    value.layout.axes[0].extent *= 4;
    assert!(matches!(
        CompiledProof::compile(byte_elements, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptInput,
            ..
        })
    ));

    let mut under_aligned = valid_input();
    let input_value = under_aligned.transcript_inputs[0].value;
    under_aligned.values[input_value.0 as usize].alignment = 1;
    assert!(matches!(
        CompiledProof::compile(under_aligned, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptInput,
            ..
        })
    ));

    let mut early = valid_input();
    let assembly = early.operations.last_mut().unwrap();
    assembly.stage = ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase);
    assert!(matches!(
        CompiledProof::compile(early, transcript()),
        Err(CompiledProofError::TranscriptCausality {
            kind: BindingKind::TranscriptOutput,
            ..
        })
    ));

    let mut same_segment = valid_input();
    let pow_input = CairoTranscriptInput::InteractionPowNonce.id().unwrap();
    let lookup_output = CairoTranscriptOutput::CommonLookupElements.id().unwrap();
    let challenge = same_segment
        .transcript_outputs
        .iter()
        .find(|binding| binding.id == lookup_output)
        .unwrap()
        .value;
    let binding = same_segment
        .transcript_inputs
        .iter_mut()
        .find(|binding| binding.id == pow_input)
        .unwrap();
    binding.value = challenge;
    binding.value_words = 0..2;
    assert!(matches!(
        CompiledProof::compile(same_segment, transcript()),
        Err(CompiledProofError::TranscriptCausality {
            kind: BindingKind::TranscriptInput,
            ..
        })
    ));

    let mut unwritten_tail = valid_input();
    let oods_output = CairoTranscriptOutput::OodsPointParameter.id().unwrap();
    let output = unwritten_tail
        .transcript_outputs
        .iter()
        .find(|binding| binding.id == oods_output)
        .unwrap()
        .value;
    unwritten_tail.values[output.0 as usize].layout.axes[0].extent += 8;
    assert!(matches!(
        CompiledProof::compile(unwritten_tail, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::TranscriptOutput,
            ..
        })
    ));

    let mut orphan = valid_input();
    let value = ValueId(orphan.values.len() as u32);
    orphan.values.push(u32_value(
        value,
        4,
        ValueOrigin::TranscriptOutput(oods_output),
        vec![],
    ));
    assert!(matches!(
        CompiledProof::compile(orphan, transcript()),
        Err(CompiledProofError::OrphanTranscriptOutput { value: actual }) if actual == value
    ));
}

#[test]
fn rejects_noncanonical_or_overlapping_proof_output() {
    let mut reordered = valid_input();
    reordered.output.sections.swap(0, 1);
    assert!(matches!(
        CompiledProof::compile(reordered, transcript()),
        Err(CompiledProofError::ProofSectionOrder { .. })
    ));

    let mut gap = valid_input();
    gap.output.layout.sampled_values.start += 1;
    assert!(matches!(
        CompiledProof::compile(gap, transcript()),
        Err(CompiledProofError::NonCanonicalProofLayout)
    ));

    let mut wrong_width = valid_input();
    wrong_width.output.layout.commitments = 0..31;
    wrong_width.output.layout.interaction_claim = 31..36;
    wrong_width.output.sections[0].value_words = 0..31;
    wrong_width.output.sections[1].value_words = 31..36;
    assert!(matches!(
        CompiledProof::compile(wrong_width, transcript()),
        Err(CompiledProofError::NonCanonicalProofLayout)
    ));

    let mut overlap = valid_input();
    let interaction = overlap.output.sections[1].value_words.clone();
    overlap.output.sections[3].value_words = interaction;
    assert!(matches!(
        CompiledProof::compile(overlap, transcript()),
        Err(CompiledProofError::BindingRange {
            kind: BindingKind::ProofOutput,
            ..
        })
    ));

    let mut external = valid_input();
    let external_value = external.transcript_inputs[0].value;
    external.values[external_value.0 as usize].layout.axes[0].extent =
        external.output.layout.total_words;
    for section in &mut external.output.sections {
        section.value = external_value;
    }
    assert!(matches!(
        CompiledProof::compile(external, transcript()),
        Err(CompiledProofError::ProofSectionOrigin { index: 0 })
    ));

    let mut extra_assembly_output = valid_input();
    let extra = ValueId(extra_assembly_output.values.len() as u32);
    extra_assembly_output
        .values
        .push(u32_value(extra, 1, ValueOrigin::OpOutput(OpId(0)), vec![]));
    extra_assembly_output.operations[0].outputs.push(extra);
    assert!(matches!(
        CompiledProof::compile(extra_assembly_output, transcript()),
        Err(CompiledProofError::InvalidProofAssembly)
    ));
}

#[test]
fn empty_identity_components_fail_before_compilation() {
    assert!(matches!(
        ProofIdentity::new(vec![], b"build".to_vec()),
        Err(CompiledProofError::EmptyIdentity(
            IdentityKind::ProofSemantics
        ))
    ));
}
