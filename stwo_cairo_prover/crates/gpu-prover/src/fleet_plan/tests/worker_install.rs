use super::*;
use crate::fleet_spill::SpillPlan;

#[test]
fn coordinator_install_is_deterministic_and_resolves_every_effect() {
    let baseline = compile(fixture()).unwrap();
    let install = baseline.worker_install_plan(WorkerId(0)).unwrap();
    assert_eq!(install.plan_identity(), baseline.identity());
    assert_eq!(install.executions().len(), 1);
    assert_eq!(
        install.executions()[0].executables[0].effects.len(),
        baseline
            .compiled()
            .effect_for(OP_ASSEMBLE)
            .unwrap()
            .accesses()
            .iter()
            .flat_map(|access| [access.source(), access.destination()])
            .flatten()
            .map(|bound| bound.binding)
            .collect::<BTreeSet<_>>()
            .len()
    );
    for storage in install.storages() {
        assert_eq!(storage.slab_offset_bytes % storage.alignment_bytes, 0);
    }
    assert!(install.capacity().packed_slab_and_exchange_bytes <= install.capacity().capacity_bytes);
    assert_eq!(
        install.barrier_arrivals().len(),
        baseline.fence_count().unwrap() as usize
    );
    let coordinator = install.coordinator().unwrap();
    assert_eq!(coordinator.transcript_barriers, baseline.barriers());
    assert_eq!(
        coordinator.transcript.len(),
        baseline.compiled().transcript_inputs().len()
            + baseline.compiled().transcript_outputs().len()
    );
    assert!(coordinator
        .transcript
        .iter()
        .all(|binding| binding.window.bytes == binding.value.elements.len() * size_of::<u32>()));
    for binding in &coordinator.transcript {
        let segment = baseline
            .compiled()
            .transcript_segments()
            .iter()
            .find(|segment| match binding.binding {
                FleetTranscriptBinding::Input(_) => segment.consumed.contains(&binding.value),
                FleetTranscriptBinding::Output(_) => segment.produced.contains(&binding.value),
            })
            .unwrap();
        let barrier = baseline
            .barriers()
            .iter()
            .find(|barrier| barrier.segment == segment.segment)
            .unwrap();
        assert_eq!(
            (binding.barrier_ordinal, binding.release_step),
            (barrier.ordinal, barrier.release_step)
        );
    }
    assert_eq!(
        coordinator.output.storage,
        baseline.placement().output_storage
    );

    let mut reordered = fixture();
    reordered.placement.storages.reverse();
    reordered.placement.storage_bindings.reverse();
    reordered.placement.operations.reverse();
    reordered.placement.barrier_arrivals.reverse();
    let reordered = compile(reordered)
        .unwrap()
        .worker_install_plan(WorkerId(0))
        .unwrap();
    assert_eq!(install, reordered);
}

#[test]
fn transcript_and_slab_geometry_are_exact_when_storage_is_fragmented() {
    let mut fixture = fixture();
    let input = fixture
        .compiled
        .transcript_inputs()
        .iter()
        .find(|binding| binding.elements.len() > 1)
        .copied()
        .unwrap();
    let binding_index = fixture
        .placement
        .storage_bindings
        .iter()
        .position(|binding| {
            binding.value.version == input.value && binding.value.elements == input.elements
        })
        .unwrap();
    let binding = fixture.placement.storage_bindings.remove(binding_index);
    let middle = input.elements.start + input.elements.len() / 2;
    fixture.placement.storage_bindings.extend([
        FleetStoragePlacement {
            value: ValueRange {
                version: input.value,
                elements: range(input.elements.start, middle),
            },
            ..binding
        },
        FleetStoragePlacement {
            value: ValueRange {
                version: input.value,
                elements: range(middle, input.elements.end),
            },
            offset_bytes: binding.offset_bytes + (middle - input.elements.start) * size_of::<u32>(),
            ..binding
        },
    ]);
    let storage = fixture.placement.storages.last_mut().unwrap();
    storage.alignment_bytes = 4096;
    fixture.placement.topology.workers[0].capacity_bytes += 4096;

    let plan = compile(fixture).unwrap();
    let install = plan.worker_install_plan(WorkerId(0)).unwrap();
    let transcript = install
        .coordinator()
        .unwrap()
        .transcript
        .iter()
        .find(|binding| binding.binding == FleetTranscriptBinding::Input(input.id))
        .unwrap();
    assert_eq!(transcript.value.elements, input.elements);
    assert_eq!(
        transcript.window.bytes,
        input.elements.len() * size_of::<u32>()
    );
    assert_eq!(install.capacity().slab_alignment_bytes, 4096);
    assert!(install
        .storages()
        .iter()
        .all(|storage| storage.slab_offset_bytes % storage.alignment_bytes == 0));
}

#[test]
fn exact_workers_receive_only_their_ordered_shard_and_roles() {
    let plan = compile(super::operation_execution::exact_fixture()).unwrap();
    let coordinator = plan.worker_install_plan(WorkerId(0)).unwrap();
    let peer = plan.worker_install_plan(WorkerId(1)).unwrap();

    assert_eq!(peer.executions().len(), 1);
    assert_eq!(peer.executions()[0].operation, OpId(0));
    assert_eq!(
        peer.executions()[0].domain,
        OperationDomain::Exact(range(4, 8))
    );
    assert_eq!(peer.executions()[0].executables[0].effects.len(), 1);
    assert!(peer.coordinator().is_none());
    assert!(coordinator.coordinator().is_some());
    assert_eq!(
        coordinator.executions()[0].domain,
        OperationDomain::Exact(range(0, 4))
    );
    assert_eq!(
        coordinator.executions()[1].domain,
        OperationDomain::Monolithic
    );
}

#[test]
fn ordered_composite_preserves_child_effect_order() {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let operation = input.operations[0].clone();
    let outer = input
        .effects
        .iter()
        .find(|effect| effect.id() == operation.effect)
        .unwrap();
    let child_effect = EffectContract::new(
        outer
            .accesses()
            .iter()
            .rev()
            .enumerate()
            .map(|(binding, access)| match access {
                EffectAccess::Read { source } => EffectAccess::Read {
                    source: bound(binding as u32, source.value),
                },
                EffectAccess::Write { destination } => EffectAccess::Write {
                    destination: bound(binding as u32, destination.value),
                },
                _ => unreachable!("the base fixture has only read and write accesses"),
            })
            .collect(),
        vec![],
    )
    .unwrap();
    let authority = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        authority.id(),
        authority.module().clone(),
        authority.semantic_encoding().to_vec(),
        authority.execution_build_encoding().to_vec(),
        vec![(child_effect.id(), operation.partition)],
    )
    .unwrap();
    input.operations[0].primitive = ExecutionPrimitive::OrderedComposite {
        children: vec![ExecutableStep {
            primitive: operation.primitive,
            invocation: invocation(&child_effect),
            effect: child_effect.id(),
        }]
        .into_boxed_slice(),
    };
    input.operations[0].invocation = None;
    input.effects.push(child_effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    let output = fixture.output_value;

    let install = compile(fixture)
        .unwrap()
        .worker_install_plan(WorkerId(0))
        .unwrap();
    assert_eq!(install.executions()[0].executables.len(), 1);
    assert_eq!(
        install.executions()[0].executables[0].child_ordinal,
        Some(0)
    );
    let first = install.executions()[0].executables[0].effects[0];
    assert_eq!(first.binding, EffectBindingId(0));
    assert_eq!(first.destination.unwrap().version, output);
}

#[test]
fn transfer_endpoints_are_projected_once_per_worker() {
    let mut fixture = super::runtime_view::transfer_fixture();
    fixture.placement.transitions[0].scratch_bytes = 0;
    let plan = compile(fixture).unwrap();
    let owner = plan.worker_install_plan(WorkerId(0)).unwrap();
    let peer = plan.worker_install_plan(WorkerId(1)).unwrap();
    let spans = plan.runtime_view().unwrap().spans().to_vec();

    assert_eq!(owner.outbound(), spans);
    assert!(owner.inbound().is_empty());
    assert_eq!(peer.inbound(), spans);
    assert!(peer.outbound().is_empty());
}

#[test]
fn install_projection_rejects_uninstalled_transition_scratch() {
    let plan = compile(super::runtime_view::transfer_fixture()).unwrap();
    let transition = plan.placement.transitions[0].id;
    assert_eq!(
        plan.worker_install_plan(WorkerId(0)).unwrap_err(),
        FleetWorkerInstallError::UnsupportedScratch(transition)
    );
}

#[test]
fn install_projection_rejects_host_spill_until_it_is_installed() {
    let mut fixture = fixture();
    fixture.placement.spills = vec![SpillPlan::empty(WorkerId(0))];
    let plan = compile(fixture).unwrap();
    assert_eq!(
        plan.worker_install_plan(WorkerId(0)).unwrap_err(),
        FleetWorkerInstallError::UnsupportedSpill
    );
}

#[test]
fn install_projection_rejects_unknown_worker_and_packed_capacity_shortfall() {
    let plan = compile(fixture()).unwrap();
    assert_eq!(
        plan.worker_install_plan(WorkerId(99)).unwrap_err(),
        FleetWorkerInstallError::UnknownWorker(WorkerId(99))
    );

    let mut short = plan;
    let worker = &mut short.placement.topology.workers[0];
    worker.capacity_bytes = 1;
    assert!(matches!(
        short.worker_install_plan(WorkerId(0)),
        Err(FleetWorkerInstallError::CapacityExceeded {
            worker: WorkerId(0),
            ..
        })
    ));
}

#[test]
fn install_projection_rejects_non_word_geometry() {
    let mut fixture = fixture();
    let version = fixture.spill_value;
    let mut input = fixture.compiled.input().clone();
    let value = &mut input.values[version.0 as usize];
    value.layout.element = ElementType { tag: 9, bytes: 8 };
    let mut stride = value.layout.element.bytes;
    for axis in &mut value.layout.axes {
        axis.stride_bytes = stride;
        stride *= axis.extent;
    }
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    let storage = &mut fixture.placement.storages[version.0 as usize];
    let added = storage.bytes;
    storage.bytes *= 2;
    fixture.placement.topology.workers[0].capacity_bytes += added;

    let plan = compile(fixture).unwrap();
    assert_eq!(
        plan.worker_install_plan(WorkerId(0)).unwrap_err(),
        FleetWorkerInstallError::UnsupportedElement(version)
    );
}

#[test]
fn install_projection_rejects_ambiguous_effect_storage() {
    let mut plan = compile(fixture()).unwrap();
    let output = plan.placement.output_storage;
    let duplicate = StorageId(plan.placement.storages.len() as u32);
    let mut desc = plan.placement.storages[output.0 as usize];
    desc.id = duplicate;
    plan.placement.storages.push(desc);
    let duplicate_bindings = plan
        .placement
        .storage_bindings
        .iter()
        .filter(|binding| binding.storage == output)
        .map(|binding| FleetStoragePlacement {
            storage: duplicate,
            ..*binding
        })
        .collect::<Vec<_>>();
    plan.placement.storage_bindings.extend(duplicate_bindings);

    assert!(matches!(
        plan.worker_install_plan(WorkerId(0)),
        Err(FleetWorkerInstallError::AmbiguousEffectWindow {
            worker: WorkerId(0),
            ..
        })
    ));
}
