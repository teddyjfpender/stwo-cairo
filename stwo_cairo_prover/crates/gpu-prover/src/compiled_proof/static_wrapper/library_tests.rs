use super::*;

fn launch(symbol: &[u8]) -> StaticCudaLaunchIdentity {
    StaticCudaLaunchIdentity::new(
        symbol.to_vec(),
        LaunchGeometry {
            grid: [4, 1, 1],
            block: [256, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .unwrap()
}

fn effect() -> EffectContractId {
    EffectContract::new(
        vec![EffectAccess::Read {
            source: BoundValueRange {
                binding: EffectBindingId(0),
                value: ValueRange {
                    version: ValueVersion(0),
                    elements: ElementRange::new(0, 1).unwrap(),
                },
            },
        }],
        vec![],
    )
    .unwrap()
    .id()
}

fn invocation() -> InvocationContractId {
    AotInvocation {
        arguments: vec![AotArgumentBinding {
            ordinal: 0,
            value: AotArgumentValue::DevicePointer(Some(EffectBindingId(0))),
        }],
    }
    .contract_id()
    .unwrap()
}

fn sort(
    word: u32,
    indices_from: StaticCudaCubBuffer,
    indices_to: StaticCudaCubBuffer,
) -> StaticCudaLibraryCallIdentity {
    StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            word,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            indices_from,
            indices_to,
            0,
            32,
            1024,
            8192,
        )
        .unwrap(),
    )
}

fn scan() -> StaticCudaLibraryCallIdentity {
    StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(
        StaticCudaCubInclusiveSumU32V1::new(1024, 4096).unwrap(),
    )
}

fn compact_steps() -> Vec<StaticCudaExecutionStepIdentity> {
    vec![
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(
            b"witness_input_compact_gather_kernel",
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(b"witness_input_compact_key_kernel")),
        StaticCudaExecutionStepIdentity::LibraryCall(sort(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(b"witness_input_compact_key_kernel")),
        StaticCudaExecutionStepIdentity::LibraryCall(sort(
            1,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(b"witness_input_compact_key_kernel")),
        StaticCudaExecutionStepIdentity::LibraryCall(sort(
            0,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(
            b"witness_input_compact_heads_kernel",
        )),
        StaticCudaExecutionStepIdentity::LibraryCall(scan()),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(
            b"witness_input_compact_clear_output_kernel",
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(
            b"witness_input_compact_scatter_kernel",
        )),
        StaticCudaExecutionStepIdentity::KernelLaunch(launch(
            b"witness_input_compact_finalize_kernel",
        )),
    ]
}

fn authority() -> StaticCudaWrapperAuthority {
    StaticCudaWrapperAuthority::new_with_execution_steps(
        StaticCudaWrapperId(1),
        [7; 32],
        89,
        b"stwo_witness_input_compact_on".to_vec(),
        [9; 32],
        [11; 32],
        [13; 32],
        [15; 32],
        compact_steps(),
        invocation(),
        effect(),
    )
    .unwrap()
}

#[test]
fn typed_calls_are_complete_fixed_width_and_address_free() {
    let sort = sort(2, StaticCudaCubBuffer::A, StaticCudaCubBuffer::B);
    assert_eq!(sort.api(), b"cub::DeviceRadixSort::SortPairs");
    assert!(sort.library_managed_launch_geometry());
    assert!(sort.ordered_on_wrapper_stream());
    let StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(call) = &sort else {
        panic!("expected sort");
    };
    assert_eq!(call.word(), 2);
    assert_eq!(call.keys_from(), StaticCudaCubBuffer::A);
    assert_eq!(call.keys_to(), StaticCudaCubBuffer::B);
    assert_eq!(call.indices_from(), StaticCudaCubBuffer::A);
    assert_eq!(call.indices_to(), StaticCudaCubBuffer::B);
    assert_eq!(call.begin_bit(), 0);
    assert_eq!(call.end_bit(), 32);
    assert_eq!(call.rows(), 1024);
    assert_eq!(call.exact_temp_bytes(), 8192);

    let mut encoded = Vec::new();
    sort.encode_into(&mut encoded);
    let mut expected = vec![1];
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&[
        StaticCudaCubBuffer::A as u8,
        StaticCudaCubBuffer::B as u8,
        StaticCudaCubBuffer::A as u8,
        StaticCudaCubBuffer::B as u8,
        0,
        32,
    ]);
    expected.extend_from_slice(&1024u32.to_le_bytes());
    expected.extend_from_slice(&8192u64.to_le_bytes());
    assert_eq!(encoded, expected);
    assert_eq!(encoded.len(), 23);
    let sort_encoding = encoded;

    let scan = scan();
    assert_eq!(scan.api(), b"cub::DeviceScan::InclusiveSum");
    let StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(call) = &scan else {
        panic!("expected scan");
    };
    assert_eq!(call.rows(), 1024);
    assert_eq!(call.exact_temp_bytes(), 4096);
    let mut encoded = Vec::new();
    scan.encode_into(&mut encoded);
    let mut expected = vec![2];
    expected.extend_from_slice(&1024u32.to_le_bytes());
    expected.extend_from_slice(&4096u64.to_le_bytes());
    assert_eq!(encoded, expected);
    assert_eq!(encoded.len(), 13);
    let scan_encoding = encoded;

    let execution = encode_execution_steps(&[
        StaticCudaExecutionStepIdentity::LibraryCall(sort),
        StaticCudaExecutionStepIdentity::LibraryCall(scan),
    ])
    .unwrap();
    let mut expected = Vec::from(AGGREGATE_DOMAIN);
    expected.extend_from_slice(&2u64.to_le_bytes());
    expected.push(2);
    expected.extend_from_slice(&sort_encoding);
    expected.push(2);
    expected.extend_from_slice(&scan_encoding);
    assert_eq!(execution, expected);
}

#[test]
fn invalid_sort_and_scan_fields_fail_closed() {
    let sort = |keys_from, keys_to, indices_from, indices_to, begin_bit, end_bit, rows, temp| {
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            0,
            keys_from,
            keys_to,
            indices_from,
            indices_to,
            begin_bit,
            end_bit,
            rows,
            temp,
        )
    };
    for invalid in [
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            1,
            1,
        ),
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::A,
            0,
            32,
            1,
            1,
        ),
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            32,
            32,
            1,
            1,
        ),
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            33,
            1,
            1,
        ),
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            0,
            1,
        ),
        sort(
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            1,
            0,
        ),
    ] {
        assert_eq!(
            invalid,
            Err(CompiledProofError::InvalidStaticWrapperManifest)
        );
    }
    for invalid in [
        StaticCudaCubInclusiveSumU32V1::new(0, 1),
        StaticCudaCubInclusiveSumU32V1::new(1, 0),
    ] {
        assert_eq!(
            invalid,
            Err(CompiledProofError::InvalidStaticWrapperManifest)
        );
    }
}

#[test]
fn complete_compact_library_sequence_and_every_field_are_sealed() {
    let baseline = authority();
    assert!(baseline.has_valid_identity().unwrap());
    assert_eq!(baseline.execution_steps().len(), 12);
    assert_eq!(baseline.kernel_launches().count(), 8);
    for (index, word, from, to) in [
        (2, 2, StaticCudaCubBuffer::A, StaticCudaCubBuffer::B),
        (4, 1, StaticCudaCubBuffer::B, StaticCudaCubBuffer::A),
        (6, 0, StaticCudaCubBuffer::A, StaticCudaCubBuffer::B),
    ] {
        let StaticCudaExecutionStepIdentity::LibraryCall(
            StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(call),
        ) = &baseline.execution_steps[index]
        else {
            panic!("expected sort at {index}");
        };
        assert_eq!(call.word(), word);
        assert_eq!(call.indices_from(), from);
        assert_eq!(call.indices_to(), to);
    }
    assert!(matches!(
        &baseline.execution_steps[8],
        StaticCudaExecutionStepIdentity::LibraryCall(
            StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(_)
        )
    ));

    let valid_sort_alternates = [
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            1,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            1024,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            1024,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            0,
            32,
            1024,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            1,
            32,
            1024,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            31,
            1024,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            2048,
            8192,
        )
        .unwrap(),
        StaticCudaCubStableAscendingSortPairsU32V1::new(
            2,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            StaticCudaCubBuffer::A,
            StaticCudaCubBuffer::B,
            0,
            32,
            1024,
            8193,
        )
        .unwrap(),
    ];
    for alternate in valid_sort_alternates {
        let mut changed = baseline.clone();
        changed.execution_steps[2] = StaticCudaExecutionStepIdentity::LibraryCall(
            StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(alternate),
        );
        assert!(!changed.has_valid_identity().unwrap());
    }
    for alternate in [
        StaticCudaCubInclusiveSumU32V1::new(2048, 4096).unwrap(),
        StaticCudaCubInclusiveSumU32V1::new(1024, 4097).unwrap(),
    ] {
        let mut changed = baseline.clone();
        changed.execution_steps[8] = StaticCudaExecutionStepIdentity::LibraryCall(
            StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(alternate),
        );
        assert!(!changed.has_valid_identity().unwrap());
    }
    let mut reordered = baseline;
    reordered.execution_steps.swap(2, 8);
    assert!(!reordered.has_valid_identity().unwrap());
}
