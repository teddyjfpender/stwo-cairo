use std::collections::BTreeSet;

use super::*;

fn output_version(input: &CompiledProofInput) -> ValueVersion {
    input.output.fragments[0].source.version
}

fn output_words(input: &CompiledProofInput) -> usize {
    input.values[output_version(input).0 as usize]
        .layout
        .element_count()
        .unwrap()
}

fn projections(input: &CompiledProofInput) -> Vec<PartitionEffectProjection> {
    let effect = &input.effects[0];
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

fn install_exact(
    input: &mut CompiledProofInput,
    axis: u16,
    domain: ElementRange,
    granularity: usize,
    alignment_bytes: usize,
    projections: Vec<PartitionEffectProjection>,
) -> Result<(), CompiledProofError> {
    let invocation = input.operations[0].invocation.as_mut().unwrap();
    let start_argument = u8::try_from(invocation.arguments.len()).unwrap();
    invocation.arguments.push(AotArgumentBinding {
        ordinal: start_argument,
        value: AotArgumentValue::U32(u32::try_from(domain.start).unwrap()),
    });
    let length_argument = u8::try_from(invocation.arguments.len()).unwrap();
    invocation.arguments.push(AotArgumentBinding {
        ordinal: length_argument,
        value: AotArgumentValue::U32(u32::try_from(domain.len()).unwrap()),
    });

    let ExecutionPrimitive::AotKernel { launch, .. } = &mut input.operations[0].primitive else {
        panic!("fixture must use an AOT kernel")
    };
    launch.grid[0] = u32::try_from(domain.len() / granularity).unwrap();
    let launch = PartitionLaunchDerivation::new(
        start_argument,
        length_argument,
        PartitionGridAxis::X,
        granularity,
    )?;
    let exact = ExactPartitionAuthority::new(
        axis,
        domain,
        granularity,
        alignment_bytes,
        projections,
        launch,
    )?;
    let partition = PartitionAuthority::exact(exact)?;
    input.operations[0].partition = partition.id();
    let kernel = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        vec![(input.operations[0].effect, partition.id())],
    )?;
    input.partitions = vec![partition];
    Ok(())
}

fn exact_input() -> CompiledProofInput {
    let mut input = valid_input();
    let words = output_words(&input);
    let exact_projections = projections(&input);
    install_exact(&mut input, 0, range(0, words), 1, 4, exact_projections).unwrap();
    input
}

#[test]
fn exact_partition_projection_is_typed_complete_and_identity_bound() {
    let first = CompiledProof::compile(exact_input(), transcript()).unwrap();
    let repeated = CompiledProof::compile(exact_input(), transcript()).unwrap();
    assert_eq!(first.identity(), repeated.identity());

    let mut changed = valid_input();
    let words = output_words(&changed);
    let changed_projections = projections(&changed);
    install_exact(
        &mut changed,
        0,
        range(0, words),
        words,
        4,
        changed_projections,
    )
    .unwrap();
    let changed = CompiledProof::compile(changed, transcript()).unwrap();
    assert_ne!(first.identity(), changed.identity());
}

#[test]
fn exact_partition_rejects_missing_or_duplicate_binding_projection() {
    let mut missing = valid_input();
    let words = output_words(&missing);
    let mut missing_projections = projections(&missing);
    missing_projections.remove(0);
    install_exact(&mut missing, 0, range(0, words), 1, 4, missing_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(missing, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    let input = valid_input();
    let words = output_words(&input);
    let mut duplicate = projections(&input);
    duplicate.insert(1, duplicate[0]);
    assert!(matches!(
        ExactPartitionAuthority::new(
            0,
            range(0, words),
            1,
            4,
            duplicate,
            PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 1).unwrap(),
        ),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_replicated_write() {
    let mut input = valid_input();
    let words = output_words(&input);
    let mut invalid = projections(&input);
    let output = invalid.last_mut().unwrap();
    *output = PartitionEffectProjection::ReplicatedRead {
        binding: output.binding(),
    };
    install_exact(&mut input, 0, range(0, words), 1, 4, invalid).unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_noncontiguous_wrong_axis_or_partial_coverage() {
    let mut wrong_axis = valid_input();
    let words = output_words(&wrong_axis);
    let wrong_axis_projections = projections(&wrong_axis);
    install_exact(
        &mut wrong_axis,
        9,
        range(0, words),
        1,
        4,
        wrong_axis_projections,
    )
    .unwrap();
    assert!(matches!(
        CompiledProof::compile(wrong_axis, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    let mut partial = valid_input();
    let partial_projections = projections(&partial);
    install_exact(
        &mut partial,
        0,
        range(0, words - 1),
        1,
        4,
        partial_projections,
    )
    .unwrap();
    assert!(matches!(
        CompiledProof::compile(partial, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    let mut noncontiguous = valid_input();
    let noncontiguous_output = output_version(&noncontiguous);
    noncontiguous.values[noncontiguous_output.0 as usize]
        .layout
        .axes = vec![
        LayoutAxis {
            tag: 0,
            extent: 1,
            stride_bytes: 4,
        },
        LayoutAxis {
            tag: 1,
            extent: words,
            stride_bytes: 4,
        },
    ];
    let noncontiguous_projections = projections(&noncontiguous);
    install_exact(
        &mut noncontiguous,
        0,
        range(0, 1),
        1,
        4,
        noncontiguous_projections,
    )
    .unwrap();
    assert!(matches!(
        CompiledProof::compile(noncontiguous, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_misaligned_slice() {
    let mut input = valid_input();
    let output = output_version(&input);
    input.values[output.0 as usize].alignment = 2;
    let words = output_words(&input);
    let exact_projections = projections(&input);
    install_exact(&mut input, 0, range(0, words), 1, 4, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_requires_explicit_reduction_operation() {
    let mut input = valid_input();
    let destination = output_version(&input);
    let words = output_words(&input);
    let source = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        source,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(999)),
        Region::Input,
    ));
    let mut accesses = input.effects[0].accesses().to_vec();
    let EffectAccess::Write { destination: out } = accesses.pop().unwrap() else {
        panic!("fixture output must be one terminal write")
    };
    assert_eq!(out.value.version, destination);
    accesses.push(EffectAccess::Atomic {
        source: bound(out.binding.0, value_range(source, words)),
        destination: out,
        operation: AtomicOperation::AddU32,
        in_place: InPlaceAliasAuthority {
            id: InPlaceAliasId(0),
            requirement: InPlaceAliasRequirement::Required,
            discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
        },
    });
    install_effect(&mut input, EffectContract::new(accesses, vec![]).unwrap());
    let exact_projections = projections(&input);
    install_exact(&mut input, 0, range(0, words), 1, 4, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_binds_start_length_abi_and_grid_derivation() {
    let mut wrong_start = exact_input();
    wrong_start.operations[0]
        .invocation
        .as_mut()
        .unwrap()
        .arguments[1]
        .value = AotArgumentValue::U32(1);
    assert!(matches!(
        CompiledProof::compile(wrong_start, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    let mut wrong_grid = exact_input();
    let ExecutionPrimitive::AotKernel { launch, .. } = &mut wrong_grid.operations[0].primitive
    else {
        panic!("fixture must use an AOT kernel")
    };
    launch.grid[0] += 1;
    assert!(matches!(
        CompiledProof::compile(wrong_grid, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    assert!(matches!(
        PartitionLaunchDerivation::new(1, 1, PartitionGridAxis::X, 1),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_requires_kernel_execution_pair_admission() {
    let mut input = exact_input();
    let kernel = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        vec![input.operations[0].effect],
    )
    .unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::KernelEffectNotAccepted { .. })
    ));
}

#[test]
fn kernel_execution_pairs_do_not_admit_an_effect_partition_cross_product() {
    let input = exact_input();
    let exact = input.operations[0].partition;
    let monolithic = PartitionAuthority::monolithic().id();
    let first = input.operations[0].effect;
    let source = input.effects[0].accesses()[0].source().unwrap().value;
    let second = EffectContract::new(
        vec![EffectAccess::Read {
            source: bound(0, source),
        }],
        vec![],
    )
    .unwrap()
    .id();
    let mut accepted = vec![(first, exact), (second, monolithic)];
    accepted.sort_unstable();
    let kernel = AotKernelAuthority::new_with_accepted_executions(
        AotKernelId(99),
        module(),
        b"paired-semantics".to_vec(),
        b"paired-build".to_vec(),
        accepted,
    )
    .unwrap();
    assert!(kernel
        .accepted_executions()
        .binary_search(&(second, exact))
        .is_err());
}

#[test]
fn exact_partition_rejects_cooperative_or_cluster_launches() {
    let mutations: [fn(&mut LaunchGeometry); 2] = [
        |launch: &mut LaunchGeometry| launch.cooperative = true,
        |launch: &mut LaunchGeometry| launch.cluster = Some([1, 1, 1]),
    ];
    for mutate in mutations {
        let mut input = exact_input();
        let ExecutionPrimitive::AotKernel { launch, .. } = &mut input.operations[0].primitive
        else {
            panic!("fixture must use an AOT kernel")
        };
        mutate(launch);
        assert!(matches!(
            CompiledProof::compile(input, transcript()),
            Err(CompiledProofError::InvalidPartitionAuthority)
        ));
    }
}

#[test]
fn exact_partition_rejects_non_elementwise_alias_discipline() {
    let mut input = valid_input();
    let destination = input.effects[0]
        .accesses()
        .last()
        .unwrap()
        .destination()
        .unwrap();
    let source_version = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        source_version,
        destination.value.elements.len(),
        ValueOrigin::ExternalInput(ExternalInputId(999)),
        Region::Input,
    ));
    let source = bound(
        0,
        ValueRange {
            version: source_version,
            elements: destination.value.elements,
        },
    );
    let effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source,
            destination: bound(1, destination.value),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::BlockBarrierPhases,
            }),
        }],
        vec![],
    )
    .unwrap();
    install_effect(&mut input, effect);
    let words = output_words(&input);
    let exact_projections = projections(&input);
    install_exact(&mut input, 0, range(0, words), 1, 4, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_replicated_source_for_elementwise_alias() {
    let mut input = valid_input();
    let words = output_words(&input);
    let source_version = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        source_version,
        words,
        ValueOrigin::ExternalInput(ExternalInputId(u32::MAX)),
        Region::Input,
    ));
    let destination = input.effects[0]
        .accesses()
        .last()
        .unwrap()
        .destination()
        .unwrap()
        .value;
    let effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(source_version, words)),
            destination: bound(1, destination),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            }),
        }],
        vec![],
    )
    .unwrap();
    install_effect(&mut input, effect);
    let exact_projections = projections(&input);
    assert!(matches!(
        exact_projections.as_slice(),
        [
            PartitionEffectProjection::ReplicatedRead { .. },
            PartitionEffectProjection::ContiguousAxisSlice { .. }
        ]
    ));
    install_exact(&mut input, 0, range(0, words), 1, 4, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(input, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_pure_read_and_subnatural_alignment() {
    let mut pure_read = valid_input();
    let source = *pure_read.effects[0].accesses()[0].source().unwrap();
    install_effect(
        &mut pure_read,
        EffectContract::new(vec![EffectAccess::Read { source }], vec![]).unwrap(),
    );
    let words = output_words(&pure_read);
    let exact_projections = projections(&pure_read);
    install_exact(&mut pure_read, 0, range(0, words), 1, 4, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(pure_read, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));

    let mut unaligned = valid_input();
    for value in &mut unaligned.values {
        value.alignment = 1;
    }
    let words = output_words(&unaligned);
    let exact_projections = projections(&unaligned);
    install_exact(&mut unaligned, 0, range(0, words), 1, 1, exact_projections).unwrap();
    assert!(matches!(
        CompiledProof::compile(unaligned, transcript()),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}

#[test]
fn exact_partition_rejects_u32_overflow_and_uncertified_tail() {
    let launch = PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 1).unwrap();
    assert!(matches!(
        ExactPartitionAuthority::new(
            0,
            range(0, u32::MAX as usize + 1),
            1,
            4,
            vec![PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(0),
            }],
            launch,
        ),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
    assert!(matches!(
        ExactPartitionAuthority::new(
            0,
            range(0, 8),
            3,
            4,
            vec![PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(0),
            }],
            launch,
        ),
        Err(CompiledProofError::InvalidPartitionAuthority)
    ));
}
