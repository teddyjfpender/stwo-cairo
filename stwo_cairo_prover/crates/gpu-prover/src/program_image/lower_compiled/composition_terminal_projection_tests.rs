use super::*;
use crate::transcript_plan::CairoTranscriptSegment;

struct Fixture {
    executable: std::sync::Arc<crate::shape_executable::ShapeExecutable>,
    before_waves: adapter::SemanticValueMap,
    before_terminal: adapter::SemanticValueMap,
    after_terminal: adapter::SemanticValueMap,
    waves: LoweredCompositionWaves,
    terminal: LoweredCompositionTerminal,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let mut values = adapter::SemanticValueMap::allocate_ordered(
            executable
                .arena()
                .transcript()
                .inputs
                .iter()
                .map(|(_, binding)| ArenaCatalogValueId(binding.logical.0)),
        )
        .unwrap();
        let bootstrap = super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &mut values,
        )
        .unwrap();
        let lookup = super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap),
            &mut values,
        )
        .unwrap();
        values
            .extend_ordered(
                super::super::composition_prelude_projection::required_upstream_catalogs(
                    executable.arena(),
                )
                .unwrap()
                .into_iter()
                .chain(
                    super::super::composition_projection::wave_external_catalogs(
                        executable.arena(),
                    )
                    .unwrap(),
                )
                .chain(required_upstream_catalogs(executable.arena()).unwrap()),
            )
            .unwrap();
        super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::InteractionAndComposition,
            Some(&lookup),
            &mut values,
        )
        .unwrap();
        let prelude = super::super::composition_prelude_projection::lower_stage(
            executable.arena(),
            &mut values,
        )
        .unwrap();
        let before_waves = values.clone();
        let waves = super::super::composition_projection::lower_waves(
            executable.arena(),
            &prelude,
            &mut values,
        )
        .unwrap();
        let before_terminal = values.clone();
        let terminal = lower_terminal(executable.arena(), &waves, &mut values).unwrap();
        Fixture {
            executable,
            before_waves,
            before_terminal,
            after_terminal: values,
            waves,
            terminal,
        }
    })
}

#[test]
fn sn2_projects_thirteen_lifts_and_the_exact_five_launch_split() {
    let fixture = fixture();
    let operations = fixture.terminal.operations();
    assert_eq!(operations.len(), 15);
    assert_eq!(
        operations
            .iter()
            .map(LoweredCompositionTerminalOperation::operation_ordinal)
            .collect::<Vec<_>>(),
        (16..31).collect::<Vec<_>>()
    );
    assert_ne!(fixture.terminal.digest(), [0; 32]);

    for operation in &operations[..13] {
        assert!(matches!(
            operation.operation().kind,
            CompositionOperationKind::LiftAccumulate { .. }
        ));
        assert_eq!(operation.operation().children.len(), 1);
        assert_eq!(operation.effect().accesses().len(), 8);
        assert_eq!(operation.invocation().arguments.len(), 4);
        assert_eq!(
            operation
                .bindings()
                .iter()
                .filter(|binding| binding.kind == CompositionAccessKind::ReadWriteRequired)
                .count(),
            4
        );
        assert!(operation.effect().accesses()[4..].iter().all(|access| {
            access.in_place().is_some_and(|alias| {
                alias.requirement == InPlaceAliasRequirement::Required
                    && alias.discipline == InPlaceDiscipline::ElementWiseReadBeforeWrite
            })
        }));
    }

    let inverse = &operations[13];
    assert_eq!(inverse.operation().children.len(), 3);
    assert_eq!(inverse.effect().accesses().len(), 14);
    assert_eq!(inverse.invocation().arguments.len(), 8);
    assert_eq!(
        inverse
            .operation()
            .children
            .iter()
            .map(|child| child.symbol.as_ref())
            .collect::<Vec<_>>(),
        vec![
            "b2n_init_block_warp_batch<2>",
            "b2n_noinit_block_batch<4,false>",
            "composition_split_boundary_batch<3,true>",
        ]
    );
    let forward = &operations[14];
    assert_eq!(forward.operation().children.len(), 2);
    assert_eq!(forward.effect().accesses().len(), 9);
    assert_eq!(forward.invocation().arguments.len(), 6);
    assert_eq!(
        forward
            .operation()
            .children
            .iter()
            .map(|child| child.symbol.as_ref())
            .collect::<Vec<_>>(),
        vec![
            "n2b_nofinal_block_batch<4,4>",
            "n2b_final_block_warp_batch<2,true>",
        ]
    );
}

#[test]
fn retained_outputs_are_distinct_generation_three_catalog_values() {
    let fixture = fixture();
    let mut catalogs = BTreeSet::new();
    let mut versions = BTreeSet::new();
    for (column, output) in fixture.terminal.outputs().iter().enumerate() {
        assert_eq!(
            output.role,
            CompositionValueRole::SplitRetained {
                canonical_column: column as u8,
                generation: 3,
            }
        );
        assert!(catalogs.insert(ArenaCatalogValueId(output.arena.logical.0)));
        assert!(versions.insert(output.version));
        assert_eq!(
            fixture
                .after_terminal
                .version(ArenaCatalogValueId(output.arena.logical.0))
                .unwrap(),
            output.version
        );
    }
    assert!(validate_from(
        fixture.executable.arena(),
        &fixture.waves,
        &fixture.before_terminal,
        &fixture.after_terminal,
        &fixture.terminal,
    )
    .is_ok());
}

#[test]
fn receipt_and_wave_version_ownership_fail_closed() {
    let fixture = fixture();
    let mut missing_wave_versions = fixture.before_waves.clone();
    let unchanged = missing_wave_versions.clone();
    assert!(lower_terminal(
        fixture.executable.arena(),
        &fixture.waves,
        &mut missing_wave_versions,
    )
    .is_err());
    assert_eq!(missing_wave_versions, unchanged);

    let mut forged = fixture.terminal.clone();
    forged.digest[0] ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            &fixture.waves,
            &fixture.before_terminal,
            &fixture.after_terminal,
            &forged,
        ),
        Err(InvocationShapeError::InvalidCompositionBinding)
    );

    let mut binding_tamper = fixture.terminal.clone();
    binding_tamper.operations[0].bindings[0].source_role = None;
    assert!(
        validate_receipt(fixture.executable.arena(), &fixture.waves, &binding_tamper,).is_err()
    );

    let mut truncated = fixture.terminal.clone();
    truncated.operations.remove(0);
    truncated.digest = receipt_digest(
        truncated.authority(),
        &fixture.waves,
        truncated.operations(),
        truncated.outputs(),
    )
    .unwrap();
    assert!(validate_receipt(fixture.executable.arena(), &fixture.waves, &truncated,).is_err());

    assert!(resolve_static_wrapper(
        StaticCudaWrapperId(1),
        90,
        fixture.executable.arena(),
        &fixture.waves,
        &fixture.before_terminal,
        &fixture.after_terminal,
        &forged,
        29,
    )
    .is_err());
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        assert!(resolve_static_wrapper(
            StaticCudaWrapperId(1),
            90,
            fixture.executable.arena(),
            &fixture.waves,
            &fixture.before_terminal,
            &fixture.after_terminal,
            &fixture.terminal,
            29,
        )
        .unwrap()
        .is_none());
    }
}
