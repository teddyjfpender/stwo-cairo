use super::compiler::{
    alias_ranges, compile_monolithic, recompile, with_late_source_read, with_required_alias,
};
use super::*;

fn compile_partitioned(fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    FleetProofPlan::compile_track_a_partitioned(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

fn add_idle_worker(fixture: &mut Fixture) {
    let capacity_bytes = fixture.placement.topology.workers[0].capacity_bytes;
    fixture.placement.topology.workers.push(WorkerSpec {
        id: WorkerId(1),
        capacity_bytes,
        exchange_reserve_bytes: 0,
    });
}

fn wrapper_launch(symbol: &[u8]) -> StaticCudaLaunchIdentity {
    StaticCudaLaunchIdentity::new(
        symbol.to_vec(),
        LaunchGeometry {
            grid: [1, 1, 1],
            block: [128, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
    )
    .unwrap()
}

fn with_base_commit_required_alias(fixture: Fixture, destination_words: usize) -> Fixture {
    let fixture = with_required_alias(fixture);
    let mut input = fixture.compiled.input().clone();
    let (source, destination) = alias_ranges(&fixture);
    input.values[destination.version.0 as usize].layout.axes[0].extent = destination_words;
    let effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, source),
            destination: bound(0, value_range(destination.version, destination_words)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::OrderedCompositeInPlace,
            }),
        }],
        vec![],
    )
    .unwrap();
    let invocation = invocation(&effect).unwrap();
    let old_effect = input.operations[0].effect;
    input.operations[0].primitive = ExecutionPrimitive::StaticCudaWrapper {
        wrapper: StaticCudaWrapperId(1),
    };
    input.operations[0].invocation = Some(invocation.clone());
    input.operations[0].effect = effect.id();
    input.static_wrappers = vec![StaticCudaWrapperAuthority::new(
        StaticCudaWrapperId(1),
        [0x21; 32],
        89,
        b"base_commit_alias_wrapper".to_vec(),
        [0x22; 32],
        [0x23; 32],
        [0x24; 32],
        [0x25; 32],
        vec![wrapper_launch(b"base_copy"), wrapper_launch(b"base_finish")],
        invocation.contract_id().unwrap(),
        effect.id(),
    )
    .unwrap()];
    input
        .effects
        .retain(|candidate| candidate.id() != old_effect);
    input.effects.push(effect);
    input.effects.sort_unstable_by_key(EffectContract::id);

    let kernel = input.kernels[0].clone();
    let accepted = input
        .operations
        .iter()
        .filter_map(|operation| {
            matches!(
                operation.primitive,
                ExecutionPrimitive::AotKernel { kernel: id, .. } if id == kernel.id()
            )
            .then(|| {
                (
                    operation.effect,
                    operation.partition,
                    operation
                        .invocation
                        .as_ref()
                        .unwrap()
                        .contract_id()
                        .unwrap(),
                )
            })
        })
        .collect();
    input.kernels = vec![AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap()];
    recompile(fixture, input)
}

fn with_base_commit_alias_chain(fixture: Fixture) -> (Fixture, [ValueVersion; 3]) {
    let fixture = with_base_commit_required_alias(fixture, 32);
    let mut input = fixture.compiled.input().clone();
    let (source, wide) = alias_ranges(&fixture);
    let narrow = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        narrow,
        8,
        input.values[wide.version.0 as usize].alignment,
        ValueOrigin::OpOutput(OpId(1)),
        Region::Dynamic,
    ));
    let effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, wide),
            destination: bound(0, value_range(narrow, 8)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::OrderedCompositeInPlace,
            }),
        }],
        vec![],
    )
    .unwrap();
    let invocation = invocation(&effect).unwrap();
    input.static_wrappers.push(
        StaticCudaWrapperAuthority::new(
            StaticCudaWrapperId(2),
            [0x31; 32],
            89,
            b"base_commit_alias_narrow_wrapper".to_vec(),
            [0x32; 32],
            [0x33; 32],
            [0x34; 32],
            [0x35; 32],
            vec![
                wrapper_launch(b"base_narrow_copy"),
                wrapper_launch(b"base_narrow_finish"),
            ],
            invocation.contract_id().unwrap(),
            effect.id(),
        )
        .unwrap(),
    );
    let mut assembly = input.operations[1].clone();
    assembly.id = OpId(2);
    input.values[fixture.output_value.0 as usize].origin = ValueOrigin::OpOutput(assembly.id);
    input.operations = vec![
        input.operations[0].clone(),
        OpNode {
            id: OpId(1),
            semantic_id: SemanticOpId(3),
            primitive: ExecutionPrimitive::StaticCudaWrapper {
                wrapper: StaticCudaWrapperId(2),
            },
            invocation: Some(invocation),
            effect: effect.id(),
            partition: input.operations[0].partition,
            stage: ProofStage::AfterTranscript,
        },
        assembly,
    ];
    input.effects.push(effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
    (
        recompile(fixture, input),
        [source.version, wide.version, narrow],
    )
}

fn with_remote_source_read(fixture: Fixture, before_alias: bool) -> Fixture {
    let mut input = fixture.compiled.input().clone();
    let (source, destination) = alias_ranges(&fixture);
    let mut alias = input.operations[0].clone();
    let mut assembly = input.operations[1].clone();
    let read_id = if before_alias { OpId(0) } else { OpId(1) };
    let read_output = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        read_output,
        16,
        size_of::<u32>(),
        ValueOrigin::OpOutput(read_id),
        Region::Dynamic,
    ));
    let read_effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(
                    0,
                    ValueRange {
                        version: source.version,
                        elements: range(0, 16),
                    },
                ),
            },
            EffectAccess::Write {
                destination: bound(1, value_range(read_output, 16)),
            },
        ],
        vec![],
    )
    .unwrap();
    let read_invocation = AotInvocation {
        arguments: vec![
            AotArgumentBinding {
                ordinal: 0,
                value: AotArgumentValue::DevicePointerTable(vec![
                    Some(EffectBindingId(0)),
                    Some(EffectBindingId(1)),
                ]),
            },
            AotArgumentBinding {
                ordinal: 1,
                value: AotArgumentValue::U32(0),
            },
            AotArgumentBinding {
                ordinal: 2,
                value: AotArgumentValue::U32(16),
            },
        ],
    };
    let read_authority = ExactPartitionAuthority::new(
        0,
        range(0, 16),
        4,
        size_of::<u32>(),
        vec![
            PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(0),
            },
            PartitionEffectProjection::ContiguousAxisSlice {
                binding: EffectBindingId(1),
            },
        ],
        PartitionLaunchDerivation::new(1, 2, PartitionGridAxis::X, 2).unwrap(),
    )
    .unwrap();
    let read_partition = PartitionAuthority::exact(read_authority).unwrap();
    let kernel = input.kernels[0].clone();
    let read = OpNode {
        id: read_id,
        semantic_id: SemanticOpId(9_999),
        primitive: ExecutionPrimitive::AotKernel {
            kernel: kernel.id(),
            launch: LaunchGeometry {
                grid: [8, 1, 1],
                block: [128, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
        },
        invocation: Some(read_invocation.clone()),
        effect: read_effect.id(),
        partition: read_partition.id(),
        stage: ProofStage::AfterTranscript,
    };
    assembly.id = OpId(2);
    input.values[fixture.output_value.0 as usize].origin = ValueOrigin::OpOutput(assembly.id);
    if before_alias {
        alias.id = OpId(1);
        input.values[destination.version.0 as usize].origin = ValueOrigin::OpOutput(alias.id);
        input.operations = vec![read, alias, assembly];
    } else {
        input.operations = vec![alias, read, assembly];
    }
    let mut accepted = kernel.accepted_executions().to_vec();
    accepted.push((
        read_effect.id(),
        read_partition.id(),
        read_invocation.contract_id().unwrap(),
    ));
    accepted.sort_unstable();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap();
    input.effects.push(read_effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
    input.partitions.push(read_partition);
    input
        .partitions
        .sort_unstable_by_key(PartitionAuthority::id);
    recompile(fixture, input)
}

fn add_source_route(fixture: &mut Fixture) {
    add_idle_worker(fixture);
    fixture.placement.topology.links.push(FleetLink {
        id: FleetLinkId(0),
        source: WorkerId(0),
        destination: WorkerId(1),
        max_transfer_bytes: 8 * size_of::<u32>(),
    });
    let reserve = 8 * stwo_backend_cuda::IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].exchange_reserve_bytes = reserve;
    fixture.placement.topology.workers[0].capacity_bytes += reserve;
}

#[test]
fn compile_track_a_monolithic_places_widening_and_narrowing_base_aliases() {
    for destination_words in [8, 32] {
        let fixture = with_base_commit_required_alias(fixture(), destination_words);
        let source_words = fixture
            .compiled
            .value(fixture.spill_value)
            .unwrap()
            .layout
            .element_count()
            .unwrap();
        let plan = compile_monolithic(fixture).unwrap();
        let alias = plan.placement().in_place_aliases.as_slice();
        assert_eq!(alias.len(), 1);
        let storage = plan
            .placement()
            .storages
            .iter()
            .find(|storage| storage.id == alias[0].storage)
            .unwrap();
        assert_eq!(
            storage.bytes,
            source_words.max(destination_words) * size_of::<u32>()
        );
        let install = plan
            .worker_install_plan(plan.placement().topology.coordinator)
            .unwrap();
        let binding = install.executions()[0].executables[0]
            .effects
            .iter()
            .find(|binding| binding.binding == EffectBindingId(0))
            .unwrap();
        assert!(binding.source.is_some());
        assert!(binding.destination.is_some());
        assert_eq!(
            binding.window.bytes,
            source_words.max(destination_words) * size_of::<u32>()
        );
        plan.validate(transcript()).unwrap();

        let mut undersized = plan.clone();
        undersized.placement.storages[storage.id.0 as usize].bytes -= size_of::<u32>();
        let error = undersized.validate(transcript()).unwrap_err();
        assert!(
            matches!(
                &error,
                FleetPlanError::UndeclaredRead {
                    operation: OpId(0),
                    ..
                } | FleetPlanError::InvalidProducer(_)
            ),
            "{error:?}"
        );
    }
}

#[test]
fn partitioned_compiler_places_coordinator_base_required_aliases_exactly() {
    for destination_words in [8, 32] {
        let mut fixture = with_base_commit_required_alias(fixture(), destination_words);
        let (source, destination) = alias_ranges(&fixture);
        let source_words = fixture
            .compiled
            .value(source.version)
            .unwrap()
            .layout
            .element_count()
            .unwrap();
        add_idle_worker(&mut fixture);

        let plan = compile_partitioned(fixture).unwrap();
        let [alias] = plan.placement().in_place_aliases.as_slice() else {
            panic!("expected one required alias")
        };
        let storage = plan
            .placement()
            .storages
            .iter()
            .find(|storage| storage.id == alias.storage)
            .unwrap();
        assert_eq!(storage.worker, plan.placement().topology.coordinator);
        assert_eq!(
            storage.bytes,
            source_words.max(destination_words) * size_of::<u32>()
        );
        for range in [source, destination] {
            let binding = plan
                .placement()
                .storage_bindings
                .iter()
                .find(|binding| binding.value.version == range.version)
                .unwrap();
            assert_eq!(binding.storage, alias.storage);
            assert_eq!(binding.offset_bytes, 0);
        }
        assert!(
            plan.placement()
                .storage_bindings
                .iter()
                .filter(|binding| [source.version, destination.version]
                    .contains(&binding.value.version))
                .all(|binding| {
                    plan.placement().storages[binding.storage.0 as usize].worker == WorkerId(0)
                })
        );
        plan.validate(transcript()).unwrap();
    }
}

#[test]
fn partitioned_compiler_shares_one_max_storage_for_widen_then_shrink_wrappers() {
    let (mut fixture, versions) = with_base_commit_alias_chain(fixture());
    add_idle_worker(&mut fixture);
    let plan = compile_partitioned(fixture).unwrap();
    let [widen, shrink] = plan.placement().in_place_aliases.as_slice() else {
        panic!("expected the widening and shrinking aliases")
    };
    assert_eq!(widen.storage, shrink.storage);
    let storage = &plan.placement().storages[widen.storage.0 as usize];
    assert_eq!(storage.worker, WorkerId(0));
    assert_eq!(storage.bytes, 32 * size_of::<u32>());
    for version in versions {
        let bindings = plan
            .placement()
            .storage_bindings
            .iter()
            .filter(|binding| binding.value.version == version)
            .collect::<Vec<_>>();
        let [binding] = bindings.as_slice() else {
            panic!("expected one full alias binding")
        };
        assert_eq!(binding.storage, widen.storage);
        assert_eq!(binding.offset_bytes, 0);
    }
    plan.validate(transcript()).unwrap();
}

#[test]
fn partitioned_compiler_rejects_a_late_coordinator_read_after_required_alias() {
    let mut fixture = with_late_source_read(fixture());
    let (source, _) = alias_ranges(&fixture);
    add_idle_worker(&mut fixture);
    assert!(matches!(
        compile_partitioned(fixture),
        Err(FleetCompileError::Lowering(FleetLoweringError::Plan(
            FleetPlanError::UndeclaredRead {
                operation: OpId(1),
                value,
            }
        ))) if value == source.version
    ));
}

#[test]
fn partitioned_required_alias_allows_only_pre_alias_remote_snapshots() {
    let mut before = with_remote_source_read(with_base_commit_required_alias(fixture(), 32), true);
    let source = before.spill_value;
    add_source_route(&mut before);
    let plan = compile_partitioned(before).unwrap();
    let [alias] = plan.placement().in_place_aliases.as_slice() else {
        panic!("expected one required alias")
    };
    let remote = plan
        .placement()
        .replicas
        .iter()
        .find(|replica| replica.worker == WorkerId(1) && replica.value.version == source)
        .unwrap();
    let remote_storage = plan
        .placement()
        .storage_bindings
        .iter()
        .find(|binding| {
            binding.value == remote.value
                && plan.placement().storages[binding.storage.0 as usize].worker == WorkerId(1)
        })
        .unwrap()
        .storage;
    assert_ne!(remote_storage, alias.storage);
    assert!(remote.live.end <= plan.placement().operations[1].during.start);
    plan.validate(transcript()).unwrap();

    let mut after = with_remote_source_read(with_base_commit_required_alias(fixture(), 32), false);
    add_source_route(&mut after);
    assert!(matches!(
        compile_partitioned(after),
        Err(FleetCompileError::Lowering(FleetLoweringError::Plan(
            FleetPlanError::InvalidTransition(_)
        )))
    ));
}

#[test]
fn partitioned_required_alias_is_deterministic_and_capacity_exact() {
    let build = || {
        let mut fixture = with_base_commit_required_alias(fixture(), 32);
        add_idle_worker(&mut fixture);
        fixture
    };
    let baseline = compile_partitioned(build()).unwrap();
    let required = baseline
        .workers()
        .iter()
        .find(|worker| worker.worker == WorkerId(0))
        .unwrap()
        .peak_resident_bytes;

    let mut reordered = build();
    reordered.placement.topology.workers.reverse();
    let reordered = compile_partitioned(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );

    let mut exact = build();
    exact.placement.topology.workers[0].capacity_bytes = required;
    compile_partitioned(exact).unwrap();

    let mut short = build();
    short.placement.topology.workers[0].capacity_bytes = required - 1;
    assert!(matches!(
        compile_partitioned(short),
        Err(FleetCompileError::Lowering(FleetLoweringError::Plan(
            FleetPlanError::CapacityExceeded {
                worker: WorkerId(0),
                required: actual,
                capacity,
            }
        ))) if actual == required && capacity + 1 == required
    ));
}
