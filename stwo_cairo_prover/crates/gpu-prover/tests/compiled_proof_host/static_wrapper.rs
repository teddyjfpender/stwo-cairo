use super::*;

const WRAPPER_ID: StaticCudaWrapperId = StaticCudaWrapperId(1);

fn launch(symbol: &[u8], grid_x: u32) -> StaticCudaLaunchIdentity {
    StaticCudaLaunchIdentity::new(
        symbol.to_vec(),
        LaunchGeometry {
            grid: [grid_x, 1, 1],
            block: [128, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .unwrap()
}

fn wrapper(
    id: StaticCudaWrapperId,
    effect: EffectContractId,
    launches: Vec<StaticCudaLaunchIdentity>,
) -> StaticCudaWrapperAuthority {
    StaticCudaWrapperAuthority::new(
        id,
        [0x51; 32],
        89,
        format!("stwo_static_wrapper_{}", id.0).into_bytes(),
        [0xa7; 32],
        [0xb3; 32],
        [0xc5; 32],
        [0xd9; 32],
        launches,
        effect,
    )
    .unwrap()
}

fn wrapper_input(launches: Vec<StaticCudaLaunchIdentity>) -> CompiledProofInput {
    let mut input = valid_input();
    let effect = input.operations[0].effect;
    input.kernels.clear();
    input.static_wrappers = vec![wrapper(WRAPPER_ID, effect, launches)];
    input.operations[0].primitive = ExecutionPrimitive::StaticCudaWrapper {
        wrapper: WRAPPER_ID,
    };
    input
}

fn three_launches() -> Vec<StaticCudaLaunchIdentity> {
    vec![
        launch(b"prepare_kernel", 8),
        launch(b"execute_kernel", 4),
        launch(b"finalize_kernel", 1),
    ]
}

fn exact_projections(effect: &EffectContract) -> Vec<PartitionEffectProjection> {
    effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|binding| {
            if effect.accesses().iter().any(|access| {
                access
                    .destination()
                    .is_some_and(|range| range.binding == binding)
            }) {
                PartitionEffectProjection::ContiguousAxisSlice { binding }
            } else {
                PartitionEffectProjection::ReplicatedRead { binding }
            }
        })
        .collect()
}

#[test]
fn one_and_three_launch_wrappers_compile_with_exact_identity() {
    let one_input = wrapper_input(vec![launch(b"execute_kernel", 4)]);
    let one = CompiledProof::compile(one_input.clone(), transcript()).unwrap();
    let repeated = CompiledProof::compile(one_input, transcript()).unwrap();
    assert_eq!(one.identity(), repeated.identity());
    assert_eq!(one.static_wrappers().len(), 1);
    assert_eq!(
        one.static_wrapper(WRAPPER_ID).unwrap().launches()[0].symbol(),
        b"execute_kernel"
    );

    let three_input = wrapper_input(three_launches());
    let three = CompiledProof::compile(three_input, transcript()).unwrap();
    let manifest = three.static_wrapper(WRAPPER_ID).unwrap();
    assert_eq!(manifest.launches().len(), 3);
    assert_ne!(one.identity(), three.identity());
    assert_ne!(
        one.static_wrapper(WRAPPER_ID)
            .unwrap()
            .aggregate_launch_identity(),
        manifest.aggregate_launch_identity()
    );

    let mut reordered = three_launches();
    reordered.swap(0, 2);
    let reordered = CompiledProof::compile(wrapper_input(reordered), transcript()).unwrap();
    assert_ne!(three.identity(), reordered.identity());
}

#[test]
fn wrapper_ids_are_nonzero_canonical_and_exactly_used() {
    assert_eq!(
        StaticCudaWrapperAuthority::new(
            StaticCudaWrapperId(0),
            [0x51; 32],
            89,
            b"zero_wrapper".to_vec(),
            [0xa7; 32],
            [0xb3; 32],
            [0xc5; 32],
            [0xd9; 32],
            vec![launch(b"execute_kernel", 1)],
            valid_input().effects[0].id(),
        ),
        Err(CompiledProofError::InvalidStaticWrapperAuthority(
            StaticCudaWrapperId(0)
        ))
    );

    let mut duplicate = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    duplicate
        .static_wrappers
        .push(duplicate.static_wrappers[0].clone());
    assert_eq!(
        CompiledProof::compile(duplicate, transcript()).unwrap_err(),
        CompiledProofError::NonCanonicalStaticWrapperAuthority
    );

    let mut reversed = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    let effect = reversed.effects[0].id();
    reversed.static_wrappers = vec![
        wrapper(
            StaticCudaWrapperId(2),
            effect,
            vec![launch(b"second_kernel", 1)],
        ),
        reversed.static_wrappers.remove(0),
    ];
    assert_eq!(
        CompiledProof::compile(reversed, transcript()).unwrap_err(),
        CompiledProofError::NonCanonicalStaticWrapperAuthority
    );

    let mut unused = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    unused.static_wrappers.push(wrapper(
        StaticCudaWrapperId(2),
        effect,
        vec![launch(b"unused_kernel", 1)],
    ));
    assert_eq!(
        CompiledProof::compile(unused, transcript()).unwrap_err(),
        CompiledProofError::NonCanonicalStaticWrapperAuthority
    );
}

#[test]
fn unknown_wrapper_id_fails_closed() {
    let mut input = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    input.operations[0].primitive = ExecutionPrimitive::StaticCudaWrapper {
        wrapper: StaticCudaWrapperId(99),
    };
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::UnknownStaticWrapper { operation: OpId(0) }
    );
}

#[test]
fn wrapper_effect_mismatch_fails_closed() {
    let mut input = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    let source = input.effects[0].accesses()[0].source().unwrap().value;
    let other_effect = EffectContract::new(
        vec![EffectAccess::Read {
            source: bound(
                0,
                ValueRange {
                    version: source.version,
                    elements: range(source.elements.start, source.elements.start + 1),
                },
            ),
        }],
        vec![],
    )
    .unwrap();
    input.static_wrappers = vec![wrapper(
        WRAPPER_ID,
        other_effect.id(),
        vec![launch(b"execute_kernel", 1)],
    )];
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::StaticWrapperEffectNotAccepted { operation: OpId(0) }
    );
}

#[test]
fn malformed_wrapper_invocation_never_compiles() {
    let mut input = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    input.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments
        .clear();
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::InvalidStaticWrapperInvocation(OpId(0))
    );
}

#[test]
fn static_wrapper_rejects_exact_partition_claims() {
    let mut input = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    let effect = &input.effects[0];
    let output_words = input.output.layout.total_words;
    let exact = ExactPartitionAuthority::new(
        0,
        range(0, output_words),
        1,
        4,
        exact_projections(effect),
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 1).unwrap(),
    )
    .unwrap();
    let partition = PartitionAuthority::exact(exact).unwrap();
    input.operations[0].partition = partition.id();
    input.partitions = vec![partition];
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::StaticWrapperRequiresMonolithic { operation: OpId(0) }
    );
}

#[test]
fn static_wrapper_cannot_fabricate_nested_child_authority() {
    let mut input = wrapper_input(vec![launch(b"execute_kernel", 1)]);
    let child = ExecutableStep {
        primitive: input.operations[0].primitive.clone(),
        invocation: input.operations[0].invocation.clone(),
        effect: input.operations[0].effect,
    };
    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![child].into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::InvalidOrderedComposite {
            operation: OpId(0),
            child: Some(0),
        }
    );
}
