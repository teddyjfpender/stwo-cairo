use super::*;

// Exact generated SN2 memory-base geometry: eight RC99 relations, while the
// final SmallRc99 wrapper updates only the first four.
const SN2_RC99_WORDS: usize = 8 << 18;
const SN2_SMALL_RC99_WORDS: usize = 4 << 18;

#[derive(Clone, Copy)]
struct AliasStep {
    source: usize,
    written_words: usize,
}

fn rc99_fixture(steps: &[AliasStep]) -> (Fixture, Vec<ValueVersion>) {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let source = fixture.spill_value;
    input.values[source.0 as usize].layout.axes[0].extent = SN2_RC99_WORDS;
    let mut versions = vec![source];
    let primitive = input.operations[0].primitive.clone();
    let partition = input.operations[0].partition;
    let stage = input.operations[0].stage;
    let mut effects = Vec::with_capacity(steps.len() + 1);
    let mut operations = Vec::with_capacity(steps.len() + 1);

    for (ordinal, step) in steps.iter().enumerate() {
        let source = versions[step.source];
        let destination = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            destination,
            SN2_RC99_WORDS,
            input.values[source.0 as usize].alignment,
            ValueOrigin::OpOutput(OpId(ordinal as u32)),
            Region::Dynamic,
        ));
        versions.push(destination);
        let elements = range(0, step.written_words);
        let effect = EffectContract::new(
            vec![EffectAccess::Atomic {
                source: bound(
                    0,
                    ValueRange {
                        version: source,
                        elements,
                    },
                ),
                destination: bound(
                    0,
                    ValueRange {
                        version: destination,
                        elements,
                    },
                ),
                operation: AtomicOperation::AddU32,
                in_place: InPlaceAliasAuthority {
                    id: InPlaceAliasId(0),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
                },
            }],
            vec![],
        )
        .unwrap();
        let invocation = invocation(&effect);
        operations.push(OpNode {
            id: OpId(ordinal as u32),
            semantic_id: SemanticOpId(10_000 + ordinal as u32),
            primitive: primitive.clone(),
            invocation,
            effect: effect.id(),
            partition,
            stage,
        });
        effects.push(effect);
    }

    let mut assembly = input.operations[0].clone();
    assembly.id = OpId(steps.len() as u32);
    input.values[fixture.output_value.0 as usize].origin = ValueOrigin::OpOutput(assembly.id);
    operations.push(assembly);
    effects.push(input.effects[0].clone());
    effects.sort_unstable_by_key(EffectContract::id);
    input.operations = operations;
    input.effects = effects;

    let authority = input.kernels[0].clone();
    let mut accepted = input
        .operations
        .iter()
        .map(|operation| {
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
        .collect::<Vec<_>>();
    accepted.sort_unstable();
    input.kernels = vec![AotKernelAuthority::new_with_accepted_executions(
        authority.id(),
        authority.module().clone(),
        authority.semantic_encoding().to_vec(),
        authority.execution_build_encoding().to_vec(),
        accepted,
    )
    .unwrap()];

    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-rc99-topology-v1",
        b"fleet-rc99-workspace-v1",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    fixture.placement.topology.workers[0].capacity_bytes = 1 << 30;
    (fixture, versions)
}

fn compile_rc99(fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    FleetProofPlan::compile_track_a_monolithic(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

fn generated_sn2_chain() -> (Fixture, Vec<ValueVersion>) {
    rc99_fixture(&[
        AliasStep {
            source: 0,
            written_words: SN2_RC99_WORDS,
        },
        AliasStep {
            source: 1,
            written_words: SN2_SMALL_RC99_WORDS,
        },
    ])
}

#[test]
fn generated_sn2_big_then_small_rc99_is_one_linear_whole_storage_chain() {
    let (fixture, versions) = generated_sn2_chain();
    let plan = compile_rc99(fixture).unwrap();
    let aliases = &plan.placement().in_place_aliases;
    assert_eq!(aliases.len(), 2);
    assert_eq!(aliases[0].storage, aliases[1].storage);
    assert_eq!(aliases[0].offset_bytes, 0);
    assert_eq!(aliases[1].offset_bytes, 0);

    for version in versions {
        let bindings = plan
            .placement()
            .storage_bindings
            .iter()
            .filter(|binding| binding.value.version == version)
            .collect::<Vec<_>>();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].storage, aliases[0].storage);
        assert_eq!(bindings[0].value.elements, range(0, SN2_RC99_WORDS));
    }
    plan.validate(transcript()).unwrap();

    let install = plan.worker_install_plan(WorkerId(0)).unwrap();
    let big = install
        .executions()
        .iter()
        .find(|execution| execution.operation == OpId(0))
        .unwrap()
        .executables[0]
        .effects[0];
    let small = install
        .executions()
        .iter()
        .find(|execution| execution.operation == OpId(1))
        .unwrap()
        .executables[0]
        .effects[0];
    assert_eq!(big.window.bytes, SN2_RC99_WORDS * size_of::<u32>());
    assert_eq!(small.window.bytes, SN2_SMALL_RC99_WORDS * size_of::<u32>());
    assert_eq!(
        small.source.unwrap().elements,
        range(0, SN2_SMALL_RC99_WORDS)
    );
    assert_eq!(
        small.destination.unwrap().elements,
        range(0, SN2_SMALL_RC99_WORDS)
    );
}

#[test]
fn required_alias_fork_is_rejected_instead_of_reusing_one_storage() {
    let (fixture, _) = rc99_fixture(&[
        AliasStep {
            source: 0,
            written_words: SN2_RC99_WORDS,
        },
        AliasStep {
            source: 1,
            written_words: SN2_SMALL_RC99_WORDS,
        },
        AliasStep {
            source: 1,
            written_words: SN2_SMALL_RC99_WORDS,
        },
    ]);
    assert!(matches!(
        compile_rc99(fixture),
        Err(FleetCompileError::RequiredAlias {
            operation: OpId(1),
            alias: InPlaceAliasId(0),
        })
    ));
}

#[test]
fn full_bindings_resolve_a_shared_nonzero_partial_window() {
    let (fixture, _) = generated_sn2_chain();
    let plan = compile_rc99(fixture).unwrap();
    let compiled = Arc::new(plan.compiled().clone());
    let shape = shape_identity_for_test(
        b"fleet-rc99-topology-v1",
        b"fleet-rc99-workspace-v1",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    let mut placement = plan.placement().clone();
    let storage = placement.in_place_aliases[0].storage;
    let base = 64;
    placement.storages[storage.0 as usize].bytes += base;
    placement.topology.workers[0].capacity_bytes += base;
    for binding in placement
        .storage_bindings
        .iter_mut()
        .filter(|binding| binding.storage == storage)
    {
        binding.offset_bytes += base;
    }
    for alias in &mut placement.in_place_aliases {
        alias.offset_bytes += base;
    }
    let shifted = FleetProofPlan::lower_compiled(compiled, shape, placement, transcript()).unwrap();
    let install = shifted.worker_install_plan(WorkerId(0)).unwrap();
    let small = install
        .executions()
        .iter()
        .find(|execution| execution.operation == OpId(1))
        .unwrap()
        .executables[0]
        .effects[0];
    assert_eq!(small.window.offset_bytes, base);

    let compiled = Arc::new(shifted.compiled().clone());
    let shape = shape_identity_for_test(
        b"fleet-rc99-topology-v1",
        b"fleet-rc99-workspace-v1",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    let mut invalid = shifted.placement().clone();
    invalid.in_place_aliases[1].offset_bytes += size_of::<u32>();
    assert!(matches!(
        FleetProofPlan::lower_compiled(compiled, shape, invalid, transcript()),
        Err(FleetLoweringError::Plan(
            FleetPlanError::InvalidInPlaceAlias {
                operation: OpId(1),
                alias: InPlaceAliasId(0),
            }
        ))
    ));
}
