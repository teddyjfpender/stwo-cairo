use super::*;

#[test]
fn rejects_primitive_geometry_and_effect_mismatches() {
    let mut zero_grid = valid_input();
    zero_grid.operations[0].primitive = ExecutionPrimitive::AotKernel {
        kernel: AotKernelId(1),
        launch: LaunchGeometry {
            grid: [0, 1, 1],
            block: [128, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    };
    assert!(matches!(
        CompiledProof::compile(zero_grid, transcript()),
        Err(CompiledProofError::InvalidLaunchGeometry(OpId(0)))
    ));

    for launch in [
        LaunchGeometry {
            grid: [u32::MAX, 1, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, u32::MAX, 1],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, u32::MAX],
            block: [1, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1025, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1025, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [1, 1, 65],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [33, 32, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [128, 1, 1],
            cluster: Some([1, 1, 1]),
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    ] {
        let mut impossible = valid_input();
        impossible.operations[0].primitive = ExecutionPrimitive::AotKernel {
            kernel: AotKernelId(1),
            launch,
        };
        assert!(matches!(
            CompiledProof::compile(impossible, transcript()),
            Err(CompiledProofError::InvalidLaunchGeometry(OpId(0)))
        ));
    }

    let mut copy = valid_input();
    copy.operations[0].primitive = ExecutionPrimitive::DeviceCopyD2D { bytes: 4 };
    copy.kernels.clear();
    assert!(matches!(
        CompiledProof::compile(copy, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));

    let mut memset = valid_input();
    memset.operations[0].primitive = ExecutionPrimitive::DeviceMemsetByte { bytes: 4, value: 0 };
    memset.kernels.clear();
    assert!(matches!(
        CompiledProof::compile(memset, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));
}

#[test]
fn raw_d2d_copy_preserves_one_exact_layout_and_memset_is_byte_typed() {
    let mut copy = valid_input();
    let bundle = copy.output.sections[0].value;
    let words = copy.output.layout.total_words;
    let source = ValueVersion(copy.values.len() as u32);
    copy.values.push(u32_value(
        source,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(10_001)),
        Region::Input,
    ));
    let contract = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, value_range(source, words)),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(bundle, words)),
            },
        ],
        vec![],
    )
    .unwrap();
    copy.operations[0].effect = contract.id();
    copy.operations[0].primitive = ExecutionPrimitive::DeviceCopyD2D {
        bytes: words * core::mem::size_of::<u32>(),
    };
    copy.effects = vec![contract];
    copy.kernels.clear();
    CompiledProof::compile(copy.clone(), transcript()).unwrap();

    copy.values[source.0 as usize].layout.axes[0].tag = 9;
    assert!(matches!(
        CompiledProof::compile(copy, transcript()),
        Err(CompiledProofError::PrimitiveEffectMismatch(OpId(0)))
    ));

    let mut memset = valid_input();
    let bundle = memset.output.sections[0].value;
    let words = memset.output.layout.total_words;
    let contract = EffectContract::new(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(bundle, words)),
        }],
        vec![],
    )
    .unwrap();
    memset.operations[0].effect = contract.id();
    memset.operations[0].primitive = ExecutionPrimitive::DeviceMemsetByte {
        bytes: words * core::mem::size_of::<u32>(),
        value: 0xa5,
    };
    memset.effects = vec![contract];
    memset.kernels.clear();
    CompiledProof::compile(memset, transcript()).unwrap();
}

#[test]
fn retained_output_may_be_produced_before_a_transcript_barrier() {
    let mut input = valid_input();
    let bundle = input.output.sections[0].value;
    let words = input.output.layout.total_words;
    let contract = EffectContract::new(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(bundle, words)),
        }],
        vec![],
    )
    .unwrap();
    install_effect(&mut input, contract);
    input.operations[0].stage =
        ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase);
    CompiledProof::compile(input, transcript()).unwrap();
}

#[test]
fn canonical_proof_sections_may_own_distinct_output_versions() {
    let mut input = valid_input();
    input.values.pop().unwrap();
    let mut accesses =
        input.effects[0].accesses()[..input.effects[0].accesses().len() - 1].to_vec();
    let destinations = layout_ranges(&input.output.layout);
    for (index, (section, destination)) in input
        .output
        .sections
        .iter_mut()
        .zip(destinations)
        .enumerate()
    {
        let words = destination.len();
        let version = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            version,
            words,
            ValueOrigin::OpOutput(OpId(0)),
            Region::Output,
        ));
        section.value = version;
        section.elements = range(0, words);
        accesses.push(EffectAccess::Write {
            destination: bound(
                u32::try_from(accesses.len()).unwrap(),
                value_range(version, words),
            ),
        });
        assert_eq!(section.section, ProofBundleSection::CANONICAL[index]);
    }
    let contract = EffectContract::new(accesses, vec![]).unwrap();
    install_effect(&mut input, contract);
    CompiledProof::compile(input, transcript()).unwrap();
}
