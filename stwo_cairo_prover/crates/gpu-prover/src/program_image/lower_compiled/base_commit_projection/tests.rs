use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{
    BaseCommitAliasDiscipline, BaseCommitAliasRequirement, BaseCommitDependencyRole,
    BaseCommitExecutionStep, BaseCommitOperationKind, BaseCommitPointerTarget,
    BaseCommitProgramAuthority, BaseCommitValueRole,
};

use super::*;
use crate::arena_plan::BufferPurpose;
use crate::compiled_proof::{
    AotArgumentValue, EffectAccess, InPlaceAliasRequirement, InPlaceDiscipline,
    StaticCudaExecutionStepIdentity, StaticCudaLibraryCallIdentity, StaticCudaWrapperId,
};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    inventory: BaseCommitInventory,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredBaseCommit,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let planned = executable
            .arena()
            .commitment(CommitmentTreeId::Base)
            .unwrap();
        let authority = BaseCommitProgramAuthority::compile(
            planned.commit_program.as_ref().unwrap(),
            planned.direct_retained_b2n_program.as_ref().unwrap(),
        )
        .unwrap();
        let inventory =
            BaseCommitInventory::compile(executable.arena(), planned, &authority).unwrap();
        let before =
            adapter::SemanticValueMap::allocate_ordered(inventory.source_catalogs()).unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(executable.arena(), &mut after).unwrap();
        Fixture {
            executable,
            inventory,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn arena_inventory_preserves_sources_outputs_state_and_installed_dependencies() {
    let fixture = fixture();
    let arena = fixture.executable.arena();
    let planned = arena.commitment(CommitmentTreeId::Base).unwrap();
    let catalog = BaseProducerCatalog::compile(arena).unwrap();

    let mut canonical = 0u32;
    for outputs in &planned.evaluation_output_groups {
        for &output in outputs.as_ref().unwrap() {
            let source_role = BaseCommitValueRole::SourceEvaluation {
                canonical_column: canonical,
            };
            let (source_catalog, source) = fixture.inventory.role(source_role).unwrap();
            assert_eq!(
                fixture.before.version(source_catalog).unwrap(),
                fixture.after.version(source_catalog).unwrap()
            );
            let source_value = catalog.value(source_catalog).unwrap();
            assert_eq!(source_value.purpose, BufferPurpose::BaseTrace);
            assert_eq!(source_value.logical, source.logical);
            assert_eq!(source_value.physical, source.physical);

            let stage_two = fixture
                .inventory
                .role(BaseCommitValueRole::RetainedStageTwo {
                    canonical_column: canonical,
                })
                .unwrap();
            let evaluation = fixture
                .inventory
                .role(BaseCommitValueRole::RetainedEvaluation {
                    canonical_column: canonical,
                })
                .unwrap();
            assert_eq!(stage_two, evaluation);
            assert_eq!(stage_two.1, output);
            canonical += 1;
        }
    }
    assert_eq!(
        canonical as usize,
        fixture.lowered.authority().retained_evaluations().len()
    );

    let state_role = fixture
        .lowered
        .authority()
        .layouts()
        .iter()
        .find_map(|layout| {
            matches!(layout.role, BaseCommitValueRole::State { .. }).then_some(layout.role)
        })
        .unwrap();
    let state = fixture.inventory.role(state_role).unwrap();
    for layout in fixture.lowered.authority().layouts() {
        match layout.role {
            BaseCommitValueRole::State { .. } => {
                assert_eq!(fixture.inventory.role(layout.role).unwrap(), state);
            }
            BaseCommitValueRole::HashLayer { .. }
                if !fixture
                    .lowered
                    .authority()
                    .retained_layers_bottom_up()
                    .iter()
                    .any(|retained| retained.role == layout.role) =>
            {
                assert_eq!(fixture.inventory.role(layout.role).unwrap(), state);
            }
            _ => {}
        }
    }
    assert_eq!(
        fixture
            .inventory
            .role(BaseCommitValueRole::HashLayer { log_size: u32::MAX }),
        Err(InvocationShapeError::InvalidBaseCommitBinding)
    );
    assert_eq!(
        fixture.inventory.role(BaseCommitValueRole::State {
            version: u32::MAX,
            log_size: u32::MAX,
        }),
        Err(InvocationShapeError::InvalidBaseCommitBinding)
    );

    let mut installed_reads = 0usize;
    let mut pointer_tables = 0usize;
    for operation in fixture.lowered.operations() {
        for binding in &operation.authority().effect.pointer_bindings {
            match &binding.target {
                BaseCommitPointerTarget::Installed { access }
                    if matches!(
                        access.role,
                        BaseCommitDependencyRole::InverseTwiddles
                            | BaseCommitDependencyRole::ForwardTwiddles
                    ) =>
                {
                    let (_, exact, _) = fixture
                        .inventory
                        .installed_value(access.role, access.range)
                        .unwrap();
                    if access.role == BaseCommitDependencyRole::ForwardTwiddles {
                        assert_eq!(exact, planned.twiddles);
                    }
                    installed_reads += 1;
                }
                BaseCommitPointerTarget::PointerTable { .. } => {
                    assert!(matches!(
                        operation.invocation().arguments[binding.argument_ordinal as usize].value,
                        AotArgumentValue::DevicePointerTable(_)
                    ));
                    pointer_tables += 1;
                }
                _ => {}
            }
        }
    }
    assert!(installed_reads > 0);
    assert!(pointer_tables > 0);
}

#[test]
fn generated_sn2_authority_and_projection_are_exact_and_ordered() {
    let fixture = fixture();
    let authority = fixture.lowered.authority();
    assert_eq!(authority.operations().len(), 198);
    assert_eq!(
        authority.identity(),
        decode32("9120acc6df05fdffcd863da289a80943efc678cc08f7da16126ccfd91d3b6592")
    );

    let mut kinds = [0usize; 8];
    for operation in authority.operations() {
        let index = match operation.kind {
            BaseCommitOperationKind::DirectB2n { .. } => 0,
            BaseCommitOperationKind::DirectN2b { .. } => 1,
            BaseCommitOperationKind::StateInit { .. } => 2,
            BaseCommitOperationKind::StateExpandInPlace { .. } => 3,
            BaseCommitOperationKind::StateAbsorb { .. } => 4,
            BaseCommitOperationKind::StateFinalizeInPlace { .. } => 5,
            BaseCommitOperationKind::MerkleLayerInPlace { .. } => 6,
            BaseCommitOperationKind::MerkleLayer { .. } => 7,
        };
        kinds[index] += 1;
    }
    assert_eq!(kinds, [53, 53, 1, 13, 53, 1, 3, 21]);

    for (ordinal, (local, exact)) in fixture
        .lowered
        .operations()
        .iter()
        .zip(authority.operations())
        .enumerate()
    {
        assert_eq!(local.ordinal(), ordinal as u32);
        assert_eq!(local.authority(), exact);
        assert_eq!(local.accesses().len(), exact.effect.accesses.len());
        for (index, (receipt, access)) in local
            .accesses()
            .iter()
            .zip(&exact.effect.accesses)
            .enumerate()
        {
            assert_eq!(receipt.authority_index, index as u32);
            assert_eq!(receipt.kind, access.kind);
            assert_eq!(receipt.role, access.role);
        }
        assert_eq!(
            local.invocation().arguments.len() + 1,
            exact.invocation.arguments.len()
        );
        assert!(local
            .invocation()
            .arguments
            .iter()
            .enumerate()
            .all(|(index, argument)| argument.ordinal as usize == index));
    }
}

#[test]
fn aliases_preserve_required_and_permitted_physical_meaning() {
    let fixture = fixture();
    let mut required = 0usize;
    let mut permitted = 0usize;
    for operation in fixture.lowered.operations() {
        let aliases = operation
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.in_place())
            .collect::<Vec<_>>();
        assert_eq!(aliases.len(), operation.authority().effect.aliases.len());
        for (local, exact) in aliases
            .into_iter()
            .zip(&operation.authority().effect.aliases)
        {
            assert_eq!(
                local.requirement,
                match exact.requirement {
                    BaseCommitAliasRequirement::Required => {
                        required += 1;
                        InPlaceAliasRequirement::Required
                    }
                    BaseCommitAliasRequirement::Optional => {
                        permitted += 1;
                        InPlaceAliasRequirement::Permitted
                    }
                }
            );
            assert_eq!(
                local.discipline,
                match exact.discipline {
                    BaseCommitAliasDiscipline::ExactLowerPrefixReadBeforeWrite => {
                        InPlaceDiscipline::ExactLowerPrefixReadBeforeWrite
                    }
                    BaseCommitAliasDiscipline::ElementWiseReadBeforeWrite => {
                        InPlaceDiscipline::ElementWiseReadBeforeWrite
                    }
                    BaseCommitAliasDiscipline::OrderedCompositeInPlace => {
                        InPlaceDiscipline::OrderedCompositeInPlace
                    }
                }
            );
            let source = &operation.accesses()[exact.source_access as usize];
            let destination = &operation.accesses()[exact.destination_access as usize];
            match exact.requirement {
                BaseCommitAliasRequirement::Required => {
                    assert_eq!(source.binding, destination.binding);
                    assert_eq!(source.arena, destination.arena);
                    assert_ne!(source.version, destination.version);
                }
                BaseCommitAliasRequirement::Optional => {
                    assert_ne!(source.binding, destination.binding);
                    assert_ne!(source.arena.physical, destination.arena.physical);
                }
            }
        }
    }
    assert!(required > 0);
    assert!(permitted > 0);
}

#[test]
fn scratch_is_an_initialized_write_only_ephemeral_value() {
    let fixture = fixture();
    let ephemeral = fixture
        .after
        .ephemeral_versions()
        .collect::<Vec<ValueVersion>>();
    let scratches = fixture
        .lowered
        .operations()
        .iter()
        .filter_map(LoweredBaseCommitOperation::scratch)
        .collect::<Vec<_>>();
    assert!(!scratches.is_empty());
    assert_eq!(scratches.len(), ephemeral.len());

    for (operation, scratch) in fixture
        .lowered
        .operations()
        .iter()
        .filter_map(|operation| operation.scratch().map(|scratch| (operation, scratch)))
    {
        assert!(ephemeral.contains(&scratch.version));
        let exact = operation
            .effect()
            .accesses()
            .iter()
            .filter(|access| {
                access
                    .destination()
                    .is_some_and(|value| value.binding == scratch.binding)
            })
            .collect::<Vec<_>>();
        assert_eq!(exact.len(), 1);
        assert!(matches!(exact[0], EffectAccess::Write { destination }
            if destination.value.version == scratch.version));
        assert!(fixture
            .lowered
            .operations()
            .iter()
            .all(|candidate| candidate
                .effect()
                .accesses()
                .iter()
                .filter_map(EffectAccess::source)
                .all(|source| source.value.version != scratch.version)));

        let installed = operation
            .authority()
            .effect
            .pointer_bindings
            .iter()
            .find_map(|binding| match binding.target {
                BaseCommitPointerTarget::Installed { access }
                    if access.role == BaseCommitDependencyRole::InPlaceScratch =>
                {
                    Some(binding.argument_ordinal)
                }
                _ => None,
            })
            .unwrap();
        assert!(matches!(
            operation.authority().execution.first(),
            Some(BaseCommitExecutionStep::DeviceCopyD2D {
                destination: stwo_backend_cuda::BaseCommitExecutionBuffer::WrapperArgument {
                    ordinal,
                    byte_offset: 0
                },
                ..
            }) if *ordinal == installed
        ));
    }
}

#[test]
fn projection_is_transactional_and_receipt_validation_is_adversarial() {
    let fixture = fixture();
    assert!(validate_from(
        fixture.executable.arena(),
        &fixture.before,
        &fixture.after,
        &fixture.lowered,
    )
    .is_ok());

    let mut empty =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = empty.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut empty).is_err());
    assert_eq!(empty, unchanged);

    let mut reordered = fixture.lowered.clone();
    reordered.operations.swap(0, 1);
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            &fixture.before,
            &fixture.after,
            &reordered,
        ),
        Err(InvocationShapeError::InvalidBaseCommitBinding)
    );

    let mut forged = fixture.lowered.clone();
    forged.digest[0] ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            &fixture.before,
            &fixture.after,
            &forged,
        ),
        Err(InvocationShapeError::InvalidBaseCommitBinding)
    );
}

#[test]
fn ordered_execution_projection_preserves_every_kernel_and_copy() {
    let fixture = fixture();
    let mut copy_count = 0usize;
    for operation in fixture.lowered.operations() {
        let projected = static_execution::project_steps(operation.authority()).unwrap();
        assert_eq!(projected.len(), operation.authority().execution.len());
        for (local, exact) in projected.iter().zip(&operation.authority().execution) {
            match (local, exact) {
                (
                    StaticCudaExecutionStepIdentity::KernelLaunch(local),
                    BaseCommitExecutionStep::KernelLaunch(exact),
                ) => {
                    assert_eq!(local.symbol(), exact.symbol.as_bytes());
                    assert_eq!(local.launch().grid, exact.grid);
                    assert_eq!(local.launch().block, exact.block);
                    assert_eq!(local.launch().cluster, exact.cluster);
                    assert_eq!(
                        local.launch().dynamic_shared_bytes,
                        exact.dynamic_shared_bytes
                    );
                    assert_eq!(local.launch().cooperative, exact.cooperative);
                }
                (
                    StaticCudaExecutionStepIdentity::LibraryCall(
                        StaticCudaLibraryCallIdentity::MemcpyD2DV1(local),
                    ),
                    BaseCommitExecutionStep::DeviceCopyD2D {
                        source:
                            stwo_backend_cuda::BaseCommitExecutionBuffer::WrapperArgument {
                                ordinal: source_ordinal,
                                byte_offset: source_byte_offset,
                            },
                        destination:
                            stwo_backend_cuda::BaseCommitExecutionBuffer::WrapperArgument {
                                ordinal: destination_ordinal,
                                byte_offset: destination_byte_offset,
                            },
                        bytes,
                    },
                ) => {
                    copy_count += 1;
                    assert_eq!(local.source_argument(), *source_ordinal);
                    assert_eq!(local.source_byte_offset(), *source_byte_offset);
                    assert_eq!(local.destination_argument(), *destination_ordinal);
                    assert_eq!(local.destination_byte_offset(), *destination_byte_offset);
                    assert_eq!(local.bytes(), *bytes);
                }
                _ => panic!("execution kind or order changed"),
            }
        }
    }
    assert!(copy_count > 0);

    for target_sm in [80, 86, 89, 90] {
        let resolved =
            resolve_static_wrapper(StaticCudaWrapperId(1), target_sm, &fixture.lowered, 0);
        if let Ok(Some(wrapper)) = resolved {
            assert_eq!(
                wrapper.execution_steps().len(),
                fixture.lowered.operations()[0].authority().execution.len()
            );
            assert_eq!(
                wrapper.accepted_effect(),
                fixture.lowered.operations()[0].effect().id()
            );
        }
    }
}

#[test]
fn generated_sn2_projection_receipt_is_stable() {
    let actual = fixture().lowered.digest();
    assert_eq!(
        actual,
        decode32("b86ab261cbda8e9dbacccd82e4a69a2ecb853c5bf9814ee6a7f4bd007251c9b6")
    );
}

fn decode32(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    let mut bytes = [0u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    bytes
}
