use super::*;

fn refresh_shape(fixture: &mut Fixture) {
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        fixture.compiled.transcript_encoding(),
        fixture.compiled.identity().canonical_encoding(),
    )
    .unwrap();
}

fn kernel(
    id: AotKernelId,
    module: ModuleIdentity,
    semantic: &[u8],
    build: &[u8],
    effect: EffectContractId,
) -> AotKernelAuthority {
    AotKernelAuthority::new(id, module, semantic.to_vec(), build.to_vec(), vec![effect]).unwrap()
}

fn alias_fixture(concurrent_consumer: bool) -> Fixture {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let output = fixture.output_value;
    input.values[output.0 as usize].origin = ValueOrigin::OpOutput(OpId(1));

    let old = ValueVersion(input.values.len() as u32);
    let new = ValueVersion(old.0 + 1);
    input.values.push(u32_value(
        old,
        8,
        4,
        ValueOrigin::ExternalInput(ExternalInputId(2_000_000)),
        Region::Input,
    ));
    input.values.push(u32_value(
        new,
        8,
        4,
        ValueOrigin::OpOutput(OpId(0)),
        Region::Dynamic,
    ));
    let alias_effect = EffectContract::new(
        vec![EffectAccess::ReadWrite {
            source: bound(0, value_range(old, 8)),
            destination: bound(0, value_range(new, 8)),
            in_place: Some(InPlaceAliasAuthority {
                id: InPlaceAliasId(0),
                requirement: InPlaceAliasRequirement::Required,
                discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
            }),
        }],
        vec![],
    )
    .unwrap();

    let original = input.effects[0].clone();
    let assembly_effect = if concurrent_consumer {
        let mut accesses = original.accesses().to_vec();
        accesses.push(EffectAccess::Read {
            source: bound(accesses.len() as u32, value_range(old, 8)),
        });
        EffectContract::new(accesses, vec![]).unwrap()
    } else {
        original
    };
    let module = input.kernels[0].module().clone();
    input.kernels = vec![
        kernel(
            AotKernelId(1),
            module.clone(),
            b"fleet-test-proof-assembly-v2",
            b"fleet-test-build-v2",
            assembly_effect.id(),
        ),
        kernel(
            AotKernelId(2),
            module,
            b"fleet-test-required-alias-v2",
            b"fleet-test-alias-build-v2",
            alias_effect.id(),
        ),
    ];
    input.effects = vec![alias_effect, assembly_effect];
    input.effects.sort_by_key(EffectContract::id);
    let alias_effect = input
        .effects
        .iter()
        .find(|effect| effect.in_place_alias(InPlaceAliasId(0)).is_some())
        .unwrap();
    let alias_effect_id = alias_effect.id();
    let alias_invocation = invocation(alias_effect);
    let assembly_effect = input
        .effects
        .iter()
        .find(|effect| effect.in_place_alias(InPlaceAliasId(0)).is_none())
        .unwrap();
    let assembly_effect_id = assembly_effect.id();
    let assembly_invocation = invocation(assembly_effect);
    let assembly = input.operations[0].clone();
    input.operations = vec![
        OpNode {
            id: OpId(0),
            semantic_id: SemanticOpId(2),
            primitive: ExecutionPrimitive::AotKernel {
                kernel: AotKernelId(2),
                launch: LaunchGeometry {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    cluster: None,
                    dynamic_shared_bytes: 0,
                    cooperative: false,
                },
            },
            invocation: alias_invocation,
            effect: alias_effect_id,
            stage: ProofStage::AfterTranscript,
        },
        OpNode {
            id: OpId(1),
            semantic_id: SemanticOpId(3),
            invocation: assembly_invocation,
            effect: assembly_effect_id,
            ..assembly
        },
    ];
    input.identity = ProofIdentity::new(
        b"fleet-alias-semantics-v2".to_vec(),
        b"fleet-alias-aot-v2".to_vec(),
    )
    .unwrap();
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());

    let final_release = fixture.placement.barrier_steps.last().unwrap().0;
    let alias_window = during(final_release + 5, final_release + 10);
    let assembly_window = during(final_release + 20, final_release + 30);
    fixture.placement.operations = vec![
        FleetOperationPlacement {
            operation: OpId(0),
            worker: WorkerId(0),
            during: alias_window,
        },
        FleetOperationPlacement {
            operation: OpId(1),
            worker: WorkerId(0),
            during: assembly_window,
        },
    ];
    fixture
        .placement
        .owners
        .iter_mut()
        .find(|owner| owner.value.version == output)
        .unwrap()
        .live = during(assembly_window.start.0, fixture.placement.terminal_step.0);
    fixture.placement.owners.extend([
        FleetOwnerPlacement {
            value: value_range(old, 8),
            worker: WorkerId(0),
            live: during(0, alias_window.end.0),
        },
        FleetOwnerPlacement {
            value: value_range(new, 8),
            worker: WorkerId(0),
            live: during(alias_window.start.0, fixture.placement.terminal_step.0),
        },
    ]);
    let storage = StorageId(fixture.placement.storages.len() as u32);
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(0),
        bytes: 32,
        alignment_bytes: 4,
    });
    fixture.placement.storage_bindings.extend([
        FleetStoragePlacement {
            storage,
            value: value_range(old, 8),
            offset_bytes: 0,
        },
        FleetStoragePlacement {
            storage,
            value: value_range(new, 8),
            offset_bytes: 0,
        },
    ]);
    fixture.placement.in_place_aliases = vec![InPlaceAliasPlacement {
        operation: OpId(0),
        alias: InPlaceAliasId(0),
        storage,
        offset_bytes: 0,
    }];
    fixture.placement.topology.workers[0].capacity_bytes += 32;
    let terminal = fixture.placement.barrier_steps.len() as u32;
    fixture
        .placement
        .barrier_arrivals
        .iter_mut()
        .find(|arrival| arrival.barrier_ordinal == terminal)
        .unwrap()
        .ready_step = assembly_window.end;
    fixture.compiled = compiled;
    refresh_shape(&mut fixture);
    fixture
}

fn distinct_output_fixture() -> Fixture {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let output_storage = fixture.placement.output_storage;
    let first = fixture.output_value;
    let widths = output_ranges(&input.output.layout).map(|range| range.len());
    input.values[first.0 as usize].layout.axes[0].extent = widths[0];
    let mut output_versions = vec![first];
    for words in widths.into_iter().skip(1) {
        let version = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            version,
            words,
            4,
            ValueOrigin::OpOutput(OP_ASSEMBLE),
            Region::Output,
        ));
        output_versions.push(version);
    }
    let old_effect = input.effects[0].clone();
    let mut accesses = old_effect.accesses()[..old_effect.accesses().len() - 1].to_vec();
    for (&version, &words) in output_versions.iter().zip(&widths) {
        accesses.push(EffectAccess::Write {
            destination: bound(accesses.len() as u32, value_range(version, words)),
        });
    }
    let effect = EffectContract::new(accesses, vec![]).unwrap();
    input.effects = vec![effect.clone()];
    input.operations[0].invocation = invocation(&effect);
    input.operations[0].effect = effect.id();
    input.kernels = vec![kernel(
        AotKernelId(1),
        input.kernels[0].module().clone(),
        b"fleet-test-distinct-output-v2",
        b"fleet-test-output-build-v2",
        effect.id(),
    )];
    input.identity = ProofIdentity::new(
        b"fleet-distinct-output-semantics-v2".to_vec(),
        b"fleet-distinct-output-aot-v2".to_vec(),
    )
    .unwrap();
    for ((section, &version), &words) in input
        .output
        .sections
        .iter_mut()
        .zip(&output_versions)
        .zip(&widths)
    {
        section.value = version;
        section.elements = range(0, words);
    }
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());

    let output_live = fixture
        .placement
        .owners
        .iter_mut()
        .find(|owner| owner.value.version == first)
        .unwrap();
    output_live.value = value_range(first, widths[0]);
    let live = output_live.live;
    fixture
        .placement
        .owners
        .extend(
            output_versions
                .iter()
                .zip(&widths)
                .skip(1)
                .map(|(&version, &words)| FleetOwnerPlacement {
                    value: value_range(version, words),
                    worker: WorkerId(0),
                    live,
                }),
        );
    fixture
        .placement
        .storage_bindings
        .retain(|binding| binding.storage != output_storage);
    fixture.placement.storage_bindings.extend(
        compiled
            .output()
            .sections
            .iter()
            .zip(output_ranges(&compiled.output().layout))
            .map(|(section, destination)| FleetStoragePlacement {
                storage: output_storage,
                value: ValueRange {
                    version: section.value,
                    elements: section.elements,
                },
                offset_bytes: destination.start * size_of::<u32>(),
            }),
    );
    fixture.compiled = compiled;
    refresh_shape(&mut fixture);
    fixture
}

#[test]
fn required_in_place_alias_is_exact_and_physically_audited() {
    compile(alias_fixture(false)).unwrap();

    let mut missing = alias_fixture(false);
    missing.placement.in_place_aliases.clear();
    assert!(matches!(
        compile(missing),
        Err(FleetPlanError::InvalidInPlaceAlias {
            operation: OpId(0),
            alias: InPlaceAliasId(0),
        })
    ));

    let mut wrong_offset = alias_fixture(false);
    wrong_offset.placement.in_place_aliases[0].offset_bytes = 4;
    assert!(matches!(
        compile(wrong_offset),
        Err(FleetPlanError::InvalidInPlaceAlias { .. })
    ));

    let mut concurrent = alias_fixture(true);
    let alias_window = concurrent.placement.operations[0].during;
    concurrent.placement.operations[1].during =
        during(alias_window.start.0, alias_window.end.0 - 1);
    let output = concurrent.output_value;
    concurrent
        .placement
        .owners
        .iter_mut()
        .find(|owner| owner.value.version == output)
        .unwrap()
        .live = during(alias_window.start.0, concurrent.placement.terminal_step.0);
    assert!(matches!(
        compile(concurrent),
        Err(FleetPlanError::InvalidInPlaceAlias { .. })
    ));
}

#[test]
fn distinct_output_versions_pack_at_destination_offsets() {
    let fixture = distinct_output_fixture();
    let output_storage = fixture.placement.output_storage;
    let expected = output_ranges(&fixture.compiled.output().layout).map(|range| range.start * 4);
    let actual = fixture
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == output_storage)
        .map(|binding| binding.offset_bytes)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    compile(fixture).unwrap();
}

#[test]
fn storage_alignment_coverage_and_reuse_fail_closed() {
    let mut misaligned = fixture();
    let storage = StorageId(misaligned.spill_value.0);
    misaligned
        .placement
        .storage_bindings
        .iter_mut()
        .find(|binding| binding.storage == storage)
        .unwrap()
        .offset_bytes = 4;
    assert!(matches!(
        compile(misaligned),
        Err(FleetPlanError::InvalidStorageBinding { .. })
    ));

    let mut uncovered = fixture();
    uncovered
        .placement
        .storage_bindings
        .retain(|binding| binding.storage != StorageId(uncovered.spill_value.0));
    assert!(matches!(
        compile(uncovered),
        Err(FleetPlanError::StorageCoverage(_))
    ));

    let mut reused = fixture();
    let spill_storage = StorageId(reused.spill_value.0);
    let reused_binding = reused
        .placement
        .storage_bindings
        .iter_mut()
        .find(|binding| {
            binding.storage != spill_storage
                && binding.storage != reused.placement.output_storage
                && binding.value.elements.len() <= 16
        })
        .expect("fixture must contain a small concurrently live value");
    reused_binding.storage = spill_storage;
    reused_binding.offset_bytes = 0;
    assert_eq!(
        compile(reused).unwrap_err(),
        FleetPlanError::IllegalStorageReuse(spill_storage)
    );
}
