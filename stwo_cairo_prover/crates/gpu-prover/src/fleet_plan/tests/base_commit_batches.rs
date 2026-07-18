use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    BaseCommitAccess, BaseCommitAccessKind, BaseCommitAliasDiscipline, BaseCommitAliasRequirement,
    BaseCommitOperation, BaseCommitOperationKind, BaseCommitProgramAuthority, BaseCommitValueRole,
};

use super::compiler::recompile;
use super::*;
use crate::arena_plan::CommitmentTreeId;
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement,
    InPlaceDiscipline, StaticCudaWrapperId,
};
use crate::transcript_plan::CairoTranscriptSegment;

const CAPACITY_BYTES: usize = 29 << 30;
const EXCHANGE_RESERVE_BYTES: usize = 4 << 30;

struct BaseFixture {
    fixture: Fixture,
    authority: BaseCommitProgramAuthority,
    roles: BTreeMap<BaseCommitValueRole, ValueVersion>,
}

fn real_sn2_base_authority() -> BaseCommitProgramAuthority {
    let executable = crate::program_image::generated_sn2_replacement();
    let planned = executable
        .arena()
        .commitment(CommitmentTreeId::Base)
        .unwrap();
    BaseCommitProgramAuthority::compile(
        planned.commit_program.as_ref().unwrap(),
        planned.direct_retained_b2n_program.as_ref().unwrap(),
    )
    .unwrap()
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

fn base_fixture() -> BaseFixture {
    let authority = real_sn2_base_authority();
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let mut producer = BTreeMap::<BaseCommitValueRole, OpId>::new();
    for (ordinal, operation) in authority.operations().iter().enumerate() {
        for access in &operation.effect.accesses {
            if access.kind == BaseCommitAccessKind::Write {
                assert_eq!(producer.insert(access.role, OpId(ordinal as u32)), None);
            }
        }
    }

    let mut roles = BTreeMap::new();
    for layout in authority.layouts() {
        let version = ValueVersion(input.values.len() as u32);
        let origin = match layout.role {
            BaseCommitValueRole::SourceEvaluation { canonical_column } => {
                ValueOrigin::ExternalInput(ExternalInputId(3_000_000 + canonical_column))
            }
            _ => ValueOrigin::OpOutput(producer[&layout.role]),
        };
        input.values.push(u32_value(
            version,
            layout.logical_words,
            layout.alignment_words * size_of::<u32>(),
            origin,
            if matches!(layout.role, BaseCommitValueRole::SourceEvaluation { .. }) {
                Region::Input
            } else {
                Region::Dynamic
            },
        ));
        assert_eq!(roles.insert(layout.role, version), None);
    }

    let monolithic = PartitionAuthority::monolithic();
    let mut effects = Vec::with_capacity(authority.operations().len() + 1);
    let mut wrappers = Vec::with_capacity(authority.operations().len());
    let mut operations = Vec::with_capacity(authority.operations().len() + 1);
    for (ordinal, exact) in authority.operations().iter().enumerate() {
        let effect = local_effect(exact, &roles);
        let invocation = invocation(&effect).unwrap();
        let wrapper_id = StaticCudaWrapperId(ordinal as u32 + 1);
        wrappers.push(
            StaticCudaWrapperAuthority::new(
                wrapper_id,
                [0xa1; 32],
                89,
                exact.abi.wrapper_symbol().as_bytes().to_vec(),
                exact.abi_identity,
                exact.effect.identity,
                exact.identity,
                [0xa2; 32],
                vec![
                    wrapper_launch(b"base_commit_batch_test_begin"),
                    wrapper_launch(b"base_commit_batch_test_end"),
                ],
                invocation.contract_id().unwrap(),
                effect.id(),
            )
            .unwrap(),
        );
        operations.push(OpNode {
            id: OpId(ordinal as u32),
            semantic_id: SemanticOpId(ordinal as u32 + 2),
            primitive: ExecutionPrimitive::StaticCudaWrapper {
                wrapper: wrapper_id,
            },
            invocation: Some(invocation),
            effect: effect.id(),
            partition: monolithic.id(),
            stage: ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase),
        });
        effects.push(effect);
    }

    let old_assembly = input.operations[0].clone();
    let old_effect = input
        .effects
        .iter()
        .find(|effect| effect.id() == old_assembly.effect)
        .unwrap();
    let mut assembly_accesses = old_effect.accesses().to_vec();
    assembly_accesses.push(EffectAccess::Read {
        source: bound(
            assembly_accesses.len() as u32,
            value_range(roles[&authority.root()], 8),
        ),
    });
    let assembly_effect = EffectContract::new(assembly_accesses, vec![]).unwrap();
    let assembly_invocation = invocation(&assembly_effect).unwrap();
    let mut assembly = old_assembly;
    assembly.id = OpId(authority.operations().len() as u32);
    assembly.semantic_id = SemanticOpId(authority.operations().len() as u32 + 2);
    assembly.effect = assembly_effect.id();
    assembly.invocation = Some(assembly_invocation.clone());
    input.values[fixture.output_value.0 as usize].origin = ValueOrigin::OpOutput(assembly.id);
    operations.push(assembly.clone());
    effects.push(assembly_effect);
    effects.sort_unstable_by_key(EffectContract::id);

    let kernel = input.kernels[0].clone();
    input.kernels = vec![AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        vec![(
            assembly.effect,
            assembly.partition,
            assembly_invocation.contract_id().unwrap(),
        )],
    )
    .unwrap()];
    input.static_wrappers = wrappers;
    input.effects = effects;
    input.partitions = vec![monolithic];
    input.operations = operations;
    fixture = recompile(fixture, input);
    BaseFixture {
        fixture,
        authority,
        roles,
    }
}

fn local_effect(
    operation: &BaseCommitOperation,
    roles: &BTreeMap<BaseCommitValueRole, ValueVersion>,
) -> EffectContract {
    let aliases = operation
        .effect
        .aliases
        .iter()
        .enumerate()
        .map(|(index, alias)| (alias.source_access as usize, (index, alias)))
        .collect::<BTreeMap<_, _>>();
    let destinations = operation
        .effect
        .aliases
        .iter()
        .map(|alias| alias.destination_access as usize)
        .collect::<BTreeSet<_>>();
    let mut next_binding = 0u32;
    let mut accesses = Vec::new();
    for (index, access) in operation.effect.accesses.iter().enumerate() {
        if destinations.contains(&index) {
            continue;
        }
        if let Some(&(alias_index, alias)) = aliases.get(&index) {
            let destination = operation.effect.accesses[alias.destination_access as usize];
            let source_binding = EffectBindingId(next_binding);
            next_binding += 1;
            let destination_binding = if alias.requirement == BaseCommitAliasRequirement::Required {
                source_binding
            } else {
                let binding = EffectBindingId(next_binding);
                next_binding += 1;
                binding
            };
            accesses.push(EffectAccess::ReadWrite {
                source: local_bound(source_binding, *access, roles),
                destination: local_bound(destination_binding, destination, roles),
                in_place: Some(InPlaceAliasAuthority {
                    id: InPlaceAliasId(alias_index as u32),
                    requirement: match alias.requirement {
                        BaseCommitAliasRequirement::Required => InPlaceAliasRequirement::Required,
                        BaseCommitAliasRequirement::Optional => InPlaceAliasRequirement::Permitted,
                    },
                    discipline: match alias.discipline {
                        BaseCommitAliasDiscipline::ExactLowerPrefixReadBeforeWrite => {
                            InPlaceDiscipline::ExactLowerPrefixReadBeforeWrite
                        }
                        BaseCommitAliasDiscipline::ElementWiseReadBeforeWrite => {
                            InPlaceDiscipline::ElementWiseReadBeforeWrite
                        }
                        BaseCommitAliasDiscipline::OrderedCompositeInPlace => {
                            InPlaceDiscipline::OrderedCompositeInPlace
                        }
                    },
                }),
            });
            continue;
        }
        let binding = EffectBindingId(next_binding);
        next_binding += 1;
        let bound = local_bound(binding, *access, roles);
        accesses.push(match access.kind {
            BaseCommitAccessKind::Read => EffectAccess::Read { source: bound },
            BaseCommitAccessKind::Write => EffectAccess::Write { destination: bound },
            BaseCommitAccessKind::ReadWrite => panic!("upstream Base access is not canonical"),
        });
    }
    EffectContract::new(accesses, vec![]).unwrap()
}

fn local_bound(
    binding: EffectBindingId,
    access: BaseCommitAccess,
    roles: &BTreeMap<BaseCommitValueRole, ValueVersion>,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange {
            version: roles[&access.role],
            elements: range(
                access.first_word,
                access.first_word.checked_add(access.word_len).unwrap(),
            ),
        },
    }
}

fn topology(ranks: usize) -> FleetPlacementTopology {
    let workers = (0..ranks)
        .map(|rank| WorkerSpec {
            id: WorkerId(rank as u16),
            capacity_bytes: CAPACITY_BYTES,
            exchange_reserve_bytes: if ranks == 1 {
                0
            } else {
                EXCHANGE_RESERVE_BYTES
            },
        })
        .collect::<Vec<_>>();
    let mut links = Vec::new();
    for source in 0..ranks {
        for destination in 0..ranks {
            if source == destination {
                continue;
            }
            links.push(FleetLink {
                id: FleetLinkId(links.len() as u16),
                source: WorkerId(source as u16),
                destination: WorkerId(destination as u16),
                max_transfer_bytes: CAPACITY_BYTES,
            });
        }
    }
    FleetPlacementTopology {
        gpu_class: ConsumerGpuClass::Rtx5090Sm120,
        module_pack_identity: [0xb1; 32],
        fixed_image_identity: [0xb2; 32],
        coordinator: WorkerId(0),
        workers,
        links,
        host_numa: vec![],
    }
}

fn compile_ranks(base: &BaseFixture, ranks: usize) -> FleetProofPlan {
    FleetProofPlan::compile_track_a_partitioned_with_base_commit_batches(
        Arc::clone(&base.fixture.compiled),
        base.fixture.shape.clone(),
        topology(ranks),
        base.fixture.placement.pow,
        OpId(0),
        &base.authority,
        transcript(),
    )
    .unwrap()
}

#[test]
fn real_sn2_base_batches_cover_once_and_remain_contiguous_on_one_two_four_ranks() {
    let base = base_fixture();
    for ranks in [1, 2, 4] {
        let plan = compile_ranks(&base, ranks);
        assert_batch_cover(&plan, &base.authority, ranks);
        assert_local_commit_sources(&plan, &base.authority, &base.roles);
        assert_state_handoffs(&plan, &base.authority, &base.roles, ranks);
    }
}

fn assert_batch_cover(plan: &FleetProofPlan, authority: &BaseCommitProgramAuthority, ranks: usize) {
    let mut by_batch = BTreeMap::<u32, (WorkerId, Vec<u32>)>::new();
    for (ordinal, operation) in authority.operations().iter().enumerate() {
        let BaseCommitOperationKind::DirectB2n {
            batch_index,
            canonical_columns,
            ..
        } = &operation.kind
        else {
            continue;
        };
        let worker = operation_worker(plan, OpId(ordinal as u32));
        let entry = by_batch
            .entry(*batch_index)
            .or_insert_with(|| (worker, Vec::new()));
        assert_eq!(entry.0, worker);
        entry.1.extend(canonical_columns);
        assert_eq!(operation_worker(plan, OpId(ordinal as u32 + 1)), worker);
        assert_eq!(operation_worker(plan, OpId(ordinal as u32 + 2)), worker);
    }
    assert_eq!(
        by_batch.keys().copied().collect::<Vec<_>>(),
        (0..by_batch.len() as u32).collect::<Vec<_>>()
    );
    let columns = by_batch
        .values()
        .flat_map(|(_, columns)| columns.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(
        columns,
        authority
            .retained_evaluations()
            .iter()
            .map(|retained| retained.canonical_column)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        columns.iter().copied().collect::<BTreeSet<_>>().len(),
        columns.len()
    );
    let batch_workers = by_batch
        .values()
        .map(|(worker, _)| *worker)
        .collect::<Vec<_>>();
    assert!(batch_workers.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(
        batch_workers.iter().copied().collect::<BTreeSet<_>>().len(),
        ranks
    );
    assert_eq!(
        operation_worker(plan, OpId(authority.operations().len() as u32 - 1)),
        *batch_workers.last().unwrap()
    );
}

fn assert_state_handoffs(
    plan: &FleetProofPlan,
    authority: &BaseCommitProgramAuthority,
    roles: &BTreeMap<BaseCommitValueRole, ValueVersion>,
    ranks: usize,
) {
    let mut expected = 0usize;
    for (ordinal, operation) in authority.operations().iter().enumerate() {
        let consumer = operation_worker(plan, OpId(ordinal as u32));
        for access in &operation.effect.accesses {
            if access.kind != BaseCommitAccessKind::Read
                || !matches!(access.role, BaseCommitValueRole::State { .. })
            {
                continue;
            }
            let version = roles[&access.role];
            let owner = plan
                .placement()
                .owners
                .iter()
                .find(|owner| owner.value.version == version)
                .unwrap();
            if owner.worker == consumer {
                continue;
            }
            expected += 1;
            let transition = plan
                .placement()
                .transitions
                .iter()
                .find(|transition| {
                    transition.value.version == version
                        && transition.source_worker == owner.worker
                        && plan
                            .placement()
                            .replicas
                            .get(transition.destination_replica.0 as usize)
                            .is_some_and(|replica| replica.worker == consumer)
                })
                .unwrap();
            let operation = &plan.placement().operations[ordinal];
            assert!(transition.during.end <= operation.during.start);
            assert_eq!(transition.value.elements.start, 0);
            assert_eq!(
                transition.value.elements.end,
                plan.compiled()
                    .value(version)
                    .unwrap()
                    .layout
                    .element_count()
                    .unwrap()
            );
            let layout = authority
                .layouts()
                .iter()
                .find(|layout| layout.role == access.role)
                .unwrap();
            assert_eq!(layout.words_per_row, 24);
        }
    }
    assert_eq!(expected, ranks.saturating_sub(1));
}

fn assert_local_commit_sources(
    plan: &FleetProofPlan,
    authority: &BaseCommitProgramAuthority,
    roles: &BTreeMap<BaseCommitValueRole, ValueVersion>,
) {
    for (ordinal, operation) in authority.operations().iter().enumerate() {
        let worker = operation_worker(plan, OpId(ordinal as u32));
        for access in &operation.effect.accesses {
            let local_role = match (&operation.kind, access.kind, access.role) {
                (
                    BaseCommitOperationKind::DirectB2n { .. },
                    BaseCommitAccessKind::Read,
                    BaseCommitValueRole::SourceEvaluation { .. },
                )
                | (
                    BaseCommitOperationKind::DirectN2b { .. },
                    BaseCommitAccessKind::Read,
                    BaseCommitValueRole::RetainedStageTwo { .. },
                )
                | (
                    BaseCommitOperationKind::StateAbsorb { .. },
                    BaseCommitAccessKind::Read,
                    BaseCommitValueRole::RetainedEvaluation { .. },
                ) => true,
                _ => false,
            };
            if !local_role {
                continue;
            }
            let value = ValueRange {
                version: roles[&access.role],
                elements: range(access.first_word, access.first_word + access.word_len),
            };
            assert!(local_at_operation(plan, worker, value, ordinal));
        }
    }
}

fn local_at_operation(
    plan: &FleetProofPlan,
    worker: WorkerId,
    value: ValueRange,
    operation: usize,
) -> bool {
    let start = plan.placement().operations[operation].during.start;
    plan.placement().owners.iter().any(|owner| {
        owner.worker == worker
            && owner.value.version == value.version
            && owner.value.elements.start <= value.elements.start
            && owner.value.elements.end >= value.elements.end
    }) || plan.placement().replicas.iter().any(|replica| {
        replica.worker == worker
            && replica.value.version == value.version
            && replica.value.elements.start <= value.elements.start
            && replica.value.elements.end >= value.elements.end
            && match replica.origin {
                ReplicaOrigin::InstalledFixed => true,
                ReplicaOrigin::Transition(id) => {
                    plan.placement().transitions[id.0 as usize].during.end <= start
                }
            }
    })
}

fn operation_worker(plan: &FleetProofPlan, operation: OpId) -> WorkerId {
    let placement = &plan.placement().operations[operation.0 as usize];
    let [execution] = placement.executions.as_slice() else {
        panic!("Base operation must remain monolithic");
    };
    assert_eq!(execution.domain, OperationDomain::Monolithic);
    execution.worker
}
