use super::*;

fn compile_monolithic(fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    FleetProofPlan::compile_track_a_monolithic(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

fn recompile(mut fixture: Fixture, input: CompiledProofInput) -> Fixture {
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    fixture
}

fn with_early_operation(fixture: Fixture) -> Fixture {
    let mut input = fixture.compiled.input().clone();
    let version = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        version,
        1,
        4,
        ValueOrigin::OpOutput(OpId(0)),
        Region::Dynamic,
    ));
    input
        .values
        .iter_mut()
        .find(|value| value.region == Region::Output)
        .unwrap()
        .origin = ValueOrigin::OpOutput(OpId(1));

    let effect = EffectContract::new(
        vec![EffectAccess::Write {
            destination: bound(0, value_range(version, 1)),
        }],
        vec![],
    )
    .unwrap();
    let mut late = input.operations[0].clone();
    late.id = OpId(1);
    let previous_effect = input
        .effects
        .iter()
        .find(|candidate| candidate.id() == late.effect)
        .unwrap();
    let mut late_accesses = previous_effect.accesses().to_vec();
    late_accesses.push(EffectAccess::Read {
        source: bound(late_accesses.len() as u32, value_range(version, 1)),
    });
    let late_effect =
        EffectContract::new(late_accesses, previous_effect.module_globals().to_vec()).unwrap();
    late.effect = late_effect.id();
    late.invocation = invocation(&late_effect);
    let previous_kernel = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new(
        previous_kernel.id(),
        previous_kernel.module().clone(),
        previous_kernel.semantic_encoding().to_vec(),
        previous_kernel.execution_build_encoding().to_vec(),
        vec![late_effect.id()],
    )
    .unwrap();
    input.operations = vec![
        OpNode {
            id: OpId(0),
            semantic_id: SemanticOpId(2),
            primitive: ExecutionPrimitive::DeviceMemsetByte { bytes: 4, value: 0 },
            invocation: None,
            effect: effect.id(),
            partition: late.partition,
            stage: ProofStage::BeforeTranscript(transcript().segments()[0].segment),
        },
        late,
    ];
    input.effects = vec![effect, late_effect];
    input.effects.sort_unstable_by_key(EffectContract::id);
    recompile(fixture, input)
}

fn with_constant(fixture: Fixture) -> (Fixture, ValueVersion) {
    let mut input = fixture.compiled.input().clone();
    let version = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        version,
        1,
        4,
        ValueOrigin::Constant(ConstantId(0)),
        Region::FixedData,
    ));
    input.fixed_values = vec![FixedValueDesc::inline_u32(ConstantId(0), version, vec![7])];
    (recompile(fixture, input), version)
}

fn with_required_alias(fixture: Fixture) -> Fixture {
    let mut input = fixture.compiled.input().clone();
    let source = fixture.spill_value;
    let words = input.values[source.0 as usize]
        .layout
        .element_count()
        .unwrap();
    let destination = ValueVersion(input.values.len() as u32);
    input.values.push(u32_value(
        destination,
        words,
        input.values[source.0 as usize].alignment,
        ValueOrigin::OpOutput(OpId(0)),
        Region::Dynamic,
    ));
    input
        .values
        .iter_mut()
        .find(|value| value.region == Region::Output)
        .unwrap()
        .origin = ValueOrigin::OpOutput(OpId(1));

    let effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(source, words)),
            destination: bound(1, value_range(destination, words)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            }),
        }],
        vec![],
    )
    .unwrap();
    let mut late = input.operations[0].clone();
    late.id = OpId(1);
    let previous_kernel = input.kernels[0].clone();
    let mut accepted = previous_kernel.accepted_executions().to_vec();
    accepted.push((effect.id(), late.partition));
    accepted.sort_unstable();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        previous_kernel.id(),
        previous_kernel.module().clone(),
        previous_kernel.semantic_encoding().to_vec(),
        previous_kernel.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap();
    input.operations = vec![
        OpNode {
            id: OpId(0),
            semantic_id: SemanticOpId(2),
            primitive: late.primitive.clone(),
            invocation: invocation(&effect),
            effect: effect.id(),
            partition: late.partition,
            stage: ProofStage::AfterTranscript,
        },
        late,
    ];
    input.effects.push(effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
    recompile(fixture, input)
}

fn owner(plan: &FleetProofPlan, version: ValueVersion) -> &FleetOwnerPlacement {
    plan.placement()
        .owners
        .iter()
        .find(|owner| owner.value.version == version)
        .unwrap()
}

#[test]
fn compile_track_a_monolithic_is_deterministic_and_caller_placement_free() {
    let mut reordered = fixture();
    reordered.placement.topology.host_numa = vec![
        HostNumaCapacity {
            numa_node: 1,
            store_capacity_bytes: 1,
            memlock_limit_bytes: 1,
        },
        HostNumaCapacity {
            numa_node: 0,
            store_capacity_bytes: 1,
            memlock_limit_bytes: 1,
        },
    ];
    reordered.placement.topology.host_numa.reverse();
    let baseline = compile_monolithic(reordered.clone()).unwrap();
    reordered.placement.topology.workers.reverse();
    reordered.placement.topology.host_numa.reverse();
    let reordered = compile_monolithic(reordered).unwrap();
    assert_eq!(baseline.identity(), reordered.identity());
    assert_eq!(
        baseline.canonical_bytes().unwrap(),
        reordered.canonical_bytes().unwrap()
    );
    assert_eq!(
        baseline.placement().operations.len(),
        baseline.compiled().operations().len()
    );
    assert!(baseline.placement().replicas.is_empty());
    assert!(baseline.placement().transitions.is_empty());
    assert!(baseline.placement().spills.is_empty());
}

#[test]
fn compile_track_a_monolithic_schedules_each_stage_inside_its_barrier() {
    let plan = compile_monolithic(with_early_operation(fixture())).unwrap();
    let early = plan.placement().operations[0];
    let late = plan.placement().operations[1];
    assert_eq!(early.during, during(0, 1));
    assert_eq!(plan.barriers()[0].release_step, ScheduleStep(2));
    assert!(early.during.end < plan.barriers()[0].release_step);
    assert_eq!(
        late.during.start,
        plan.barriers().last().unwrap().release_step
    );
    assert_eq!(plan.terminal_step(), ScheduleStep(late.during.end.0 + 1));
    assert_eq!(
        plan.placement().barrier_arrivals[0].ready_step,
        early.during.end
    );
    assert_eq!(
        plan.placement().barrier_arrivals.last().unwrap().ready_step,
        late.during.end
    );
}

#[test]
fn compile_track_a_monolithic_rejects_unmodeled_remote_workers() {
    let mut input = fixture();
    input.placement.topology.workers.push(WorkerSpec {
        id: WorkerId(1),
        capacity_bytes: 1 << 30,
        exchange_reserve_bytes: 0,
    });
    assert!(matches!(
        compile_monolithic(input),
        Err(FleetCompileError::MonolithicWorkerCount { actual: 2 })
    ));
}

#[test]
fn compile_track_a_monolithic_preserves_reserve_and_capacity_is_exact() {
    let baseline = compile_monolithic(fixture()).unwrap();
    let mut reserved = fixture();
    reserved.placement.topology.workers[0].exchange_reserve_bytes = 512;
    let reserved = compile_monolithic(reserved).unwrap();
    assert_eq!(
        reserved.placement().topology.workers[0].exchange_reserve_bytes,
        512
    );
    assert_eq!(
        reserved.workers()[0].peak_resident_bytes,
        baseline.workers()[0].peak_resident_bytes + 512
    );
    let required = reserved.workers()[0].peak_resident_bytes;

    let mut exact = fixture();
    exact.placement.topology.workers[0].exchange_reserve_bytes = 512;
    exact.placement.topology.workers[0].capacity_bytes = required;
    compile_monolithic(exact).unwrap();

    let mut short = fixture();
    short.placement.topology.workers[0].exchange_reserve_bytes = 512;
    short.placement.topology.workers[0].capacity_bytes = required - 1;
    assert!(matches!(
        compile_monolithic(short),
        Err(FleetCompileError::Lowering(FleetLoweringError::Plan(
            FleetPlanError::CapacityExceeded {
                worker: WorkerId(0),
                required: actual,
                capacity,
            }
        ))) if actual == required && capacity + 1 == required
    ));
}

#[test]
fn compile_track_a_monolithic_validates_and_binds_pow_schedule() {
    let baseline = compile_monolithic(fixture()).unwrap();
    let mut changed = fixture();
    changed.placement.pow.interaction.workers_per_rank = 2;
    changed.placement.pow.interaction.indices_per_attempt = 16;
    let changed = compile_monolithic(changed).unwrap();
    assert_eq!(changed.placement().pow.interaction.workers_per_rank, 2);
    assert_eq!(changed.placement().pow.interaction.indices_per_attempt, 16);
    assert_ne!(baseline.identity(), changed.identity());

    let mut invalid = fixture();
    invalid.placement.pow.query.workers_per_rank = 0;
    assert!(matches!(
        compile_monolithic(invalid),
        Err(FleetCompileError::Pow(FleetPowError::InvalidGeometry))
    ));
}

#[test]
fn compile_track_a_monolithic_derives_minimal_authorized_liveness() {
    let (fixture, constant) = with_constant(fixture());
    let unused = fixture.spill_value;
    let plan = compile_monolithic(fixture).unwrap();
    let operation = plan.placement().operations[0];

    assert_eq!(owner(&plan, unused).live, during(0, 1));
    assert_eq!(owner(&plan, constant).live.end, plan.terminal_step());
    for (segment, barrier) in plan
        .compiled()
        .transcript_segments()
        .iter()
        .zip(plan.barriers())
    {
        for consumed in &segment.consumed {
            assert_eq!(
                owner(&plan, consumed.version).live.end,
                barrier.release_step
            );
        }
        for produced in &segment.produced {
            assert_eq!(
                owner(&plan, produced.version).live.start,
                barrier.release_step
            );
            assert_eq!(
                owner(&plan, produced.version).live.end,
                operation.during.end
            );
        }
    }
    for fragment in &plan.compiled().output().fragments {
        let live = owner(&plan, fragment.source.version).live;
        assert_eq!(live.start, operation.during.start);
        assert_eq!(live.end, plan.terminal_step());
    }
}

#[test]
fn compile_track_a_monolithic_uses_canonical_output_fragments() {
    let plan = compile_monolithic(fixture()).unwrap();
    let bindings = plan
        .placement()
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == plan.placement().output_storage)
        .collect::<Vec<_>>();
    assert_eq!(bindings.len(), plan.compiled().output().fragments.len());
    for (binding, fragment) in bindings
        .into_iter()
        .zip(&plan.compiled().output().fragments)
    {
        assert_eq!(binding.value, fragment.source);
        assert_eq!(
            binding.offset_bytes,
            fragment.destination.start * size_of::<u32>()
        );
    }
}

#[test]
fn compile_track_a_monolithic_rejects_required_alias_until_placement_exists() {
    assert!(matches!(
        compile_monolithic(with_required_alias(fixture())),
        Err(FleetCompileError::RequiredAlias {
            operation: OpId(0),
            alias: InPlaceAliasId(0),
        })
    ));
}
