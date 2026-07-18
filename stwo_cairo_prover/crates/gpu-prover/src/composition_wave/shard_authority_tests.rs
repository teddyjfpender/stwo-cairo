use stwo::core::fields::m31::BaseField;
use stwo_backend_cuda::aot;

use super::*;
use crate::compiled_proof::{
    BoundValueRange, CompiledProofError, EffectAccess, EffectBindingId, EffectContract,
    ElementRange, PartitionAuthorityKind, PartitionEffectProjection, PartitionGridAxis, ValueRange,
    ValueVersion,
};
use crate::composition_plan::{
    CompositionComponentPlan, CompositionKernelPart, CompositionPlan, CompositionWaveKernelPlan,
};
use crate::prepared_composition::{
    CompositionAccumulatorRequirements, CompositionWavePartRequirement, CompositionWaveRequirements,
};

fn kernel(id: u64, rc_base: u32) -> CompositionKernelPart {
    CompositionKernelPart {
        kernel_name: format!("wave_part_{id}"),
        cache_key: id,
        semantic_hash: id + 100,
        source: format!("part_source_{id}"),
        rc_base,
    }
}

fn component(
    name: &'static str,
    evaluation_log_size: u32,
    n_constraints: usize,
    random_coefficient_offset: usize,
    kernels: &[(u64, u32)],
) -> CompositionComponentPlan {
    CompositionComponentPlan {
        component: name,
        instance: 0,
        trace_locations: Vec::new(),
        preprocessed_column_indices: Vec::new(),
        trace_log_size: evaluation_log_size - 1,
        evaluation_log_size,
        n_constraints,
        random_coefficient_offset,
        denominator_inverses: vec![BaseField::from(1)],
        base_param_values: Vec::new(),
        ext_param_values: Vec::new(),
        ext_param_sources: Vec::new(),
        kernels: kernels.iter().map(|&(id, rc)| kernel(id, rc)).collect(),
    }
}

fn fixture() -> (
    CompositionPlan,
    CompositionWaveProgram,
    Vec<CompositionWaveRequirements>,
    Vec<CompositionAccumulatorRequirements>,
) {
    let mut plan = CompositionPlan {
        max_kernel_instrs: 192,
        total_constraints: 15,
        max_evaluation_log_size: 8,
        components: vec![
            component("a", 8, 5, 0, &[(1, 0), (2, 2)]),
            component("b", 6, 3, 5, &[(3, 0)]),
            component("c", 8, 7, 8, &[(4, 0), (5, 3), (6, 5)]),
        ],
        wave_kernels: Vec::new(),
    };
    let initial = CompositionWaveProgram::from_plan(&plan).unwrap();
    plan.wave_kernels = initial
        .waves()
        .iter()
        .map(|wave| {
            let parts = wave
                .part_ordinals
                .iter()
                .map(|&ordinal| {
                    let part = &initial.parts()[ordinal];
                    aot::CompositionWaveKernelPartIdentity {
                        semantic_hash: part.semantic_hash,
                        coefficient_start: part.coefficient_start as u32,
                        coefficient_end: part.coefficient_end as u32,
                    }
                })
                .collect::<Vec<_>>();
            let identity =
                aot::composition_wave_kernel_identity(wave.evaluation_log_size, &parts).unwrap();
            CompositionWaveKernelPlan {
                evaluation_log_size: wave.evaluation_log_size,
                parts,
                kernel_name: identity.kernel_name,
                cache_key: identity.cache_key,
                semantic_hash: identity.semantic_hash,
                program_identity: [wave.evaluation_log_size as u8; 32],
                source: format!("canonical_source_{}", wave.evaluation_log_size),
            }
        })
        .collect();
    let program = CompositionWaveProgram::from_plan(&plan).unwrap();
    let mut offset = 0usize;
    let requirements = program
        .waves()
        .iter()
        .enumerate()
        .map(|(wave_index, wave)| {
            let row_count = usize::try_from(wave.row_count).unwrap();
            let requirement = CompositionWaveRequirements {
                evaluation_log_size: wave.evaluation_log_size,
                row_count,
                accumulator_offset_words: offset,
                descriptor_offset_words: 100 + wave_index * 20,
                parts: wave
                    .part_ordinals
                    .iter()
                    .map(|&ordinal| {
                        let part = &program.parts()[ordinal];
                        CompositionWavePartRequirement {
                            component: part.component_index,
                            kernel: part.kernel_index,
                            identity: aot::CompositionWaveKernelPartIdentity {
                                semantic_hash: part.semantic_hash,
                                coefficient_start: part.coefficient_start as u32,
                                coefficient_end: part.coefficient_end as u32,
                            },
                        }
                    })
                    .collect(),
            };
            offset += row_count * 4;
            requirement
        })
        .collect::<Vec<_>>();
    let accumulators = requirements
        .iter()
        .map(|wave| CompositionAccumulatorRequirements {
            log_size: wave.evaluation_log_size,
            offset_words: wave.accumulator_offset_words,
            len_words: wave.row_count * 4,
        })
        .collect();
    (plan, program, requirements, accumulators)
}

fn accumulator_bindings() -> [EffectBindingId; 4] {
    [
        EffectBindingId(3),
        EffectBindingId(4),
        EffectBindingId(5),
        EffectBindingId(6),
    ]
}

fn bound(binding: u32, version: u32, rows: usize) -> BoundValueRange {
    BoundValueRange {
        binding: EffectBindingId(binding),
        value: ValueRange {
            version: ValueVersion(version),
            elements: ElementRange::new(0, rows).unwrap(),
        },
    }
}

fn effect_with_first_read_version(rows: usize, first_read_version: u32) -> EffectContract {
    EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, first_read_version, rows),
            },
            EffectAccess::Read {
                source: bound(1, 1, rows),
            },
            EffectAccess::Read {
                source: bound(2, 2, rows),
            },
            EffectAccess::Write {
                destination: bound(3, 10, rows),
            },
            EffectAccess::Write {
                destination: bound(4, 11, rows),
            },
            EffectAccess::Write {
                destination: bound(5, 12, rows),
            },
            EffectAccess::Write {
                destination: bound(6, 13, rows),
            },
        ],
        Vec::new(),
    )
    .unwrap()
}

fn effect(rows: usize) -> EffectContract {
    effect_with_first_read_version(rows, 0)
}

#[test]
fn fixture_effect_binding_namespace_is_dense_and_canonical() {
    let effect = effect(64);
    assert_eq!(
        effect
            .accesses()
            .iter()
            .flat_map(|access| [access.source(), access.destination()])
            .flatten()
            .map(|range| range.binding)
            .collect::<Vec<_>>(),
        (0..7).map(EffectBindingId).collect::<Vec<_>>()
    );
    assert_eq!(
        EffectContract::new(
            vec![
                EffectAccess::Read {
                    source: bound(0, 0, 64),
                },
                EffectAccess::Write {
                    destination: bound(2, 1, 64),
                },
            ],
            Vec::new(),
        ),
        Err(CompiledProofError::NonCanonicalEffectBindings)
    );
}

fn derive(
    plan: &CompositionPlan,
    program: &CompositionWaveProgram,
    wave_index: usize,
    requirement: &CompositionWaveRequirements,
    accumulator: &CompositionAccumulatorRequirements,
) -> Result<CompositionWaveShardAuthority, CompositionWaveShardAuthorityError> {
    CompositionWaveShardAuthority::derive(
        plan,
        program,
        wave_index,
        requirement,
        accumulator,
        &effect(requirement.row_count),
        accumulator_bindings(),
    )
}

#[test]
fn authority_seals_full_rows_kernel_abi_and_four_sliced_coordinates() {
    let (plan, program, requirements, accumulators) = fixture();
    let authority = derive(&plan, &program, 1, &requirements[1], &accumulators[1]).unwrap();
    let repeated = derive(&plan, &program, 1, &requirements[1], &accumulators[1]).unwrap();

    assert_eq!(authority, repeated);
    assert_eq!(authority.evaluation_log_size(), 8);
    assert_eq!(authority.full_rows(), 256);
    assert_eq!(authority.row_axis_tag(), 0);
    assert_eq!(authority.row_alignment_bytes(), 4);
    assert_eq!(authority.projections().len(), 7);
    assert_eq!(
        authority
            .projections()
            .iter()
            .filter(|projection| matches!(
                projection,
                PartitionEffectProjection::ContiguousAxisSlice { .. }
            ))
            .count(),
        4
    );
    assert_eq!(
        authority.required_range_abi(),
        CompositionWaveRangeAbiRequirement {
            codegen_version: 2,
            argument_count: 9,
            full_domain_rows_argument: 6,
            shard_start_argument: 7,
            shard_rows_argument: 8,
            threads_per_block: 128,
        }
    );
    assert_eq!(authority.require_structural_range_abi(), Ok(()));
    assert_ne!(authority.kernel_source_identity(), &[0; 32]);
    assert_ne!(authority.kernel_program_identity(), &[0; 32]);
    assert_eq!(
        authority.kernel_abi_schema_identity(),
        &aot::AotKernelAbiSchema::CompositionWaveV2.identity()
    );
    let PartitionAuthorityKind::Exact(partition) = authority.partition().kind() else {
        panic!("composition wave must carry exact partition authority");
    };
    assert_eq!(partition.domain(), ElementRange { start: 0, end: 256 });
    assert_eq!(partition.granularity(), 128);
    assert_eq!(partition.projections(), authority.projections());
    assert_eq!(partition.launch().start_argument(), 7);
    assert_eq!(partition.launch().length_argument(), 8);
    assert_eq!(partition.launch().grid_axis(), PartitionGridAxis::X);
    assert_eq!(partition.launch().elements_per_grid_unit(), 128);
}

#[test]
fn requirement_part_and_accumulator_drift_fail_closed() {
    let (plan, program, mut requirements, accumulators) = fixture();
    requirements[1].row_count -= 1;
    assert_eq!(
        derive(&plan, &program, 1, &requirements[1], &accumulators[1],),
        Err(CompositionWaveShardAuthorityError::RequirementDrift(
            "wave domain"
        ))
    );

    let (plan, program, mut requirements, accumulators) = fixture();
    requirements[1].parts.swap(0, 1);
    assert_eq!(
        derive(&plan, &program, 1, &requirements[1], &accumulators[1],),
        Err(CompositionWaveShardAuthorityError::RequirementDrift(
            "wave part identity/order"
        ))
    );

    let (plan, program, requirements, mut accumulators) = fixture();
    accumulators[1].len_words -= 1;
    assert_eq!(
        derive(&plan, &program, 1, &requirements[1], &accumulators[1],),
        Err(CompositionWaveShardAuthorityError::AccumulatorDrift(
            "four coordinate extents"
        ))
    );
}

#[test]
fn program_and_kernel_identity_drift_fail_closed() {
    let (mut plan, _, requirements, accumulators) = fixture();
    plan.wave_kernels[1].cache_key ^= 1;
    let program = CompositionWaveProgram::from_plan(&plan).unwrap();
    assert_eq!(
        derive(&plan, &program, 1, &requirements[1], &accumulators[1]),
        Err(CompositionWaveShardAuthorityError::KernelIdentityDrift)
    );

    let (plan, mut program, requirements, accumulators) = fixture();
    program.waves[1].part_ordinals.swap(0, 1);
    assert_eq!(
        derive(&plan, &program, 1, &requirements[1], &accumulators[1]),
        Err(CompositionWaveShardAuthorityError::Program(
            CompositionWaveError::ProgramDrift
        ))
    );
}

#[test]
fn program_and_source_identities_are_structural_but_not_loaded_authority() {
    let (plan, program, requirements, accumulators) = fixture();
    let original = derive(&plan, &program, 1, &requirements[1], &accumulators[1]).unwrap();

    let mut changed_program_plan = plan.clone();
    changed_program_plan.wave_kernels[1].program_identity[0] ^= 1;
    assert!(matches!(
        program.validate_against(&changed_program_plan),
        Err(CompositionWaveError::SourcePlanDrift { .. })
    ));
    let changed_program = CompositionWaveProgram::from_plan(&changed_program_plan).unwrap();
    let changed = derive(
        &changed_program_plan,
        &changed_program,
        1,
        &requirements[1],
        &accumulators[1],
    )
    .unwrap();
    assert_ne!(original.digest(), changed.digest());

    let mut changed_source_plan = plan.clone();
    changed_source_plan.wave_kernels[1]
        .source
        .push_str("_drift");
    let changed_source_program = CompositionWaveProgram::from_plan(&changed_source_plan).unwrap();
    let changed_source = derive(
        &changed_source_plan,
        &changed_source_program,
        1,
        &requirements[1],
        &accumulators[1],
    )
    .unwrap();
    assert_ne!(
        original.kernel_source_identity(),
        changed_source.kernel_source_identity()
    );
    assert_ne!(original.digest(), changed_source.digest());

    let mut zero_program_plan = plan;
    zero_program_plan.wave_kernels[1].program_identity = [0; 32];
    let zero_program = CompositionWaveProgram::from_plan(&zero_program_plan).unwrap();
    assert_eq!(
        derive(
            &zero_program_plan,
            &zero_program,
            1,
            &requirements[1],
            &accumulators[1],
        ),
        Err(CompositionWaveShardAuthorityError::KernelIdentityDrift)
    );

    assert_eq!(
        original.clone().bind_loaded(8, 10),
        Err(CompositionWaveShardAuthorityError::InvalidTargetSm)
    );
    assert!(matches!(
        original.bind_loaded(99, 0),
        Err(CompositionWaveShardAuthorityError::MissingLoadedManifest)
            | Err(CompositionWaveShardAuthorityError::MissingLoadedKernel { target_sm: 990, .. })
    ));
}

#[test]
fn incomplete_or_overlapping_effect_bindings_are_rejected() {
    let (plan, program, requirements, accumulators) = fixture();
    let rows = requirements[1].row_count;
    assert_eq!(
        CompositionWaveShardAuthority::derive(
            &plan,
            &program,
            1,
            &requirements[1],
            &accumulators[1],
            &effect(rows),
            [
                EffectBindingId(3),
                EffectBindingId(3),
                EffectBindingId(5),
                EffectBindingId(6),
            ],
        ),
        Err(CompositionWaveShardAuthorityError::InvalidEffect(
            "module globals or coordinate bindings"
        ))
    );
    assert_eq!(
        CompositionWaveShardAuthority::derive(
            &plan,
            &program,
            1,
            &requirements[1],
            &accumulators[1],
            &effect(rows),
            [
                EffectBindingId(1),
                EffectBindingId(4),
                EffectBindingId(5),
                EffectBindingId(6),
            ],
        ),
        Err(CompositionWaveShardAuthorityError::InvalidEffect(
            "replicated reads or four full-row destinations"
        ))
    );

    let incomplete = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(0, 0, rows),
            },
            EffectAccess::Read {
                source: bound(1, 1, rows),
            },
            EffectAccess::Write {
                destination: bound(2, 10, rows),
            },
            EffectAccess::Write {
                destination: bound(3, 11, rows),
            },
            EffectAccess::Write {
                destination: bound(4, 12, rows),
            },
        ],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        CompositionWaveShardAuthority::derive(
            &plan,
            &program,
            1,
            &requirements[1],
            &accumulators[1],
            &incomplete,
            accumulator_bindings(),
        ),
        Err(CompositionWaveShardAuthorityError::InvalidEffect(
            "replicated reads or four full-row destinations"
        ))
    );
}

#[test]
fn coordinate_order_effect_and_shape_are_identity_bound() {
    let (plan, program, requirements, accumulators) = fixture();
    let original = derive(&plan, &program, 1, &requirements[1], &accumulators[1]).unwrap();
    let reordered = CompositionWaveShardAuthority::derive(
        &plan,
        &program,
        1,
        &requirements[1],
        &accumulators[1],
        &effect(requirements[1].row_count),
        [
            EffectBindingId(4),
            EffectBindingId(3),
            EffectBindingId(5),
            EffectBindingId(6),
        ],
    )
    .unwrap();
    let changed_effect = effect_with_first_read_version(requirements[1].row_count, 99);
    let effect_changed = CompositionWaveShardAuthority::derive(
        &plan,
        &program,
        1,
        &requirements[1],
        &accumulators[1],
        &changed_effect,
        accumulator_bindings(),
    )
    .unwrap();
    let other_wave = derive(&plan, &program, 0, &requirements[0], &accumulators[0]).unwrap();

    assert_ne!(original.digest(), reordered.digest());
    assert_ne!(original.digest(), effect_changed.digest());
    assert_ne!(original.digest(), other_wave.digest());
    let PartitionAuthorityKind::Exact(other_partition) = other_wave.partition().kind() else {
        panic!("small wave must remain exactly representable");
    };
    assert_eq!(other_partition.domain(), ElementRange { start: 0, end: 64 });
    assert_eq!(other_partition.granularity(), 64);
    assert_ne!(original.program_digest(), &[0; 32]);
    assert_ne!(original.current_abi_digest(), &[0; 32]);
}
