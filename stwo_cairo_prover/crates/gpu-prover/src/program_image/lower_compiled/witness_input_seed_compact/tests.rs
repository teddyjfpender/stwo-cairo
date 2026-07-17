use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{
    WitnessInputCompactCubStage, WitnessInputCompactExecution, WitnessInputCompactKernelStage,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::super::compiled_base_prefix;
use super::super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::arena_plan::BufferPurpose;
use crate::compiled_proof::{
    AotArgumentValue, FixedValueInitializer, StaticCudaExecutionStepIdentity,
    StaticCudaLibraryCallIdentity,
};
use crate::shape_executable::ShapeExecutable;

mod adversarial;

pub(super) struct Fixture {
    pub(super) executable: Arc<ShapeExecutable>,
    pub(super) authority: BaseProducerAuthority,
    pub(super) initial_values: adapter::SemanticValueMap,
    pub(super) values: adapter::SemanticValueMap,
    pub(super) required_preproducers: BTreeSet<ValueVersion>,
    pub(super) lowered: Vec<LoweredWitnessInputSetup>,
}

fn executable() -> Arc<ShapeExecutable> {
    static EXECUTABLE: OnceLock<Arc<ShapeExecutable>> = OnceLock::new();
    Arc::clone(EXECUTABLE.get_or_init(super::super::tests::generated_sn2_replacement))
}

pub(super) fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = executable();
        let mut initial_values =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
                .unwrap();
        let authority = BaseProducerAuthority::compile_replacement_into(
            executable.arena(),
            PreProcessedTraceVariant::Canonical,
            &mut initial_values,
        )
        .unwrap();
        let required_preproducers =
            compiled_base_prefix::validate_witness_writer_transitions_for_test(
                &authority,
                &initial_values,
            )
            .unwrap();
        let mut values = initial_values.clone();
        let lowered = lower_stage(executable.arena(), &mut values).unwrap();
        Fixture {
            executable,
            authority,
            initial_values,
            values,
            required_preproducers,
            lowered,
        }
    })
}

fn writer<'a>(
    authority: &'a BaseProducerAuthority,
    component: &'static str,
    part: TracePartId,
) -> &'a SemanticBaseProducer {
    authority
        .producers
        .iter()
        .find(|producer| {
            producer.producer().component == component && producer.producer().part == Some(part)
        })
        .expect("generated setup endpoint must have one recorded writer")
}

#[test]
fn generated_sn2_has_four_seed_and_three_compact_preproducers() {
    let fixture = fixture();
    let planned = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .filter(|component| component.input_seed.is_some() || component.input_compact.is_some())
        .map(|component| (component.component, component.part))
        .collect::<Vec<_>>();
    let lowered = fixture
        .lowered
        .iter()
        .map(|setup| (setup.component(), setup.part()))
        .collect::<Vec<_>>();
    assert_eq!(planned, lowered);
    assert_eq!(fixture.lowered.len(), 7);
    assert_eq!(
        fixture
            .lowered
            .iter()
            .filter(|setup| matches!(setup, LoweredWitnessInputSetup::Seed(_)))
            .count(),
        4
    );
    assert_eq!(
        fixture
            .lowered
            .iter()
            .filter(|setup| matches!(setup, LoweredWitnessInputSetup::Compact(_)))
            .count(),
        3
    );
    assert!(fixture
        .lowered
        .windows(2)
        .all(|pair| pair[0].position().ordinal < pair[1].position().ordinal));

    let seed_outputs = fixture
        .lowered
        .iter()
        .filter_map(|setup| match setup {
            LoweredWitnessInputSetup::Seed(seed) => Some(seed.outputs.len()),
            LoweredWitnessInputSetup::Compact(_) => None,
        })
        .sum::<usize>();
    let compact_outputs = fixture
        .lowered
        .iter()
        .filter_map(|setup| match setup {
            LoweredWitnessInputSetup::Compact(compact) => Some(compact.outputs.len()),
            LoweredWitnessInputSetup::Seed(_) => None,
        })
        .sum::<usize>();
    assert_eq!((seed_outputs, compact_outputs), (12, 25));

    let output_versions = fixture
        .lowered
        .iter()
        .flat_map(|setup| match setup {
            LoweredWitnessInputSetup::Seed(seed) => seed.outputs.as_slice(),
            LoweredWitnessInputSetup::Compact(compact) => compact.outputs.as_slice(),
        })
        .map(|output| output.version)
        .collect::<BTreeSet<_>>();
    assert_eq!(output_versions.len(), 37);
    let inactive = fixture
        .lowered
        .iter()
        .flat_map(|setup| {
            let outputs = match setup {
                LoweredWitnessInputSetup::Seed(seed) => seed.outputs.as_slice(),
                LoweredWitnessInputSetup::Compact(compact) => compact.outputs.as_slice(),
            };
            outputs
                .iter()
                .enumerate()
                .filter(|(_, output)| !fixture.required_preproducers.contains(&output.version))
                .map(|(ordinal, _)| (setup.component(), setup.part(), ordinal))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        inactive,
        [
            ("bitwise_builtin", TracePartId::Main, 1),
            ("pedersen_builtin", TracePartId::Main, 1),
            ("poseidon_builtin", TracePartId::Main, 1),
            ("range_check_builtin", TracePartId::Main, 1),
            ("pedersen_aggregator_window_bits_18", TracePartId::Main, 3),
            ("poseidon_aggregator", TracePartId::Main, 6),
            ("verify_instruction", TracePartId::Main, 7),
            ("verify_instruction", TracePartId::Main, 8),
        ]
    );
    assert_eq!(
        output_versions
            .intersection(&fixture.required_preproducers)
            .count(),
        29
    );
    assert_eq!(fixture.required_preproducers.len(), 278);

    let mut consumed_versions = BTreeSet::new();
    for setup in &fixture.lowered {
        let consumer = writer(&fixture.authority, setup.component(), setup.part());
        assert_eq!(consumer.position(), setup.position());
        let outputs = match setup {
            LoweredWitnessInputSetup::Seed(seed) => &seed.outputs,
            LoweredWitnessInputSetup::Compact(compact) => &compact.outputs,
        };
        for output in outputs {
            let consumer_reads = consumer
                .effect()
                .accesses()
                .iter()
                .filter_map(|access| access.source())
                .filter(|source| {
                    source.value.version == output.version
                        && source.value.elements == output.elements
                })
                .count();
            assert_eq!(
                consumer_reads,
                usize::from(fixture.required_preproducers.contains(&output.version)),
                "{} input",
                setup.component()
            );
            if consumer_reads == 1 {
                assert!(consumed_versions.insert(output.version));
            }
            assert_eq!(
                setup
                    .effect()
                    .accesses()
                    .iter()
                    .filter_map(|access| access.destination())
                    .filter(|destination| {
                        destination.binding == output.binding
                            && destination.value.version == output.version
                            && destination.value.elements == output.elements
                    })
                    .count(),
                1
            );
        }
    }
    assert_eq!(
        consumed_versions,
        output_versions
            .intersection(&fixture.required_preproducers)
            .copied()
            .collect()
    );
    validate(
        fixture.executable.arena(),
        &fixture.values,
        &fixture.lowered,
    )
    .unwrap();
}

#[test]
fn roots_scratch_and_relocations_keep_their_exact_roles() {
    let fixture = fixture();
    let catalog = BaseProducerCatalog::compile(fixture.executable.arena()).unwrap();
    let mut seed_roots = BTreeSet::new();
    let mut compact_scratch = BTreeSet::new();
    let mut descriptors = BTreeSet::new();
    for setup in &fixture.lowered {
        match setup {
            LoweredWitnessInputSetup::Seed(seed) => {
                let value = catalog.value(seed.scalar_source.value).unwrap();
                assert_eq!(value.purpose, BufferPurpose::WitnessInputSeedScalars);
                assert_eq!(value.component, Some(seed.component));
                assert_eq!(value.part, Some(seed.part));
                assert!(seed_roots.insert(seed.scalar_source.version));
                assert!(!fixture
                    .required_preproducers
                    .contains(&seed.scalar_source.version));
                assert!(fixture
                    .values
                    .version(ArenaCatalogValueId(
                        seed.relocations.output_pointers.logical.0
                    ))
                    .is_err());
            }
            LoweredWitnessInputSetup::Compact(compact) => {
                descriptors.insert(compact.descriptor_value);
                for scratch in &compact.scratch {
                    let value = catalog.value(scratch.value).unwrap();
                    assert!(matches!(
                        value.purpose,
                        BufferPurpose::WitnessInputCompactTupleScratch
                            | BufferPurpose::WitnessInputCompactSortKey
                            | BufferPurpose::WitnessInputCompactSortIndex
                            | BufferPurpose::WitnessInputCompactRunHeads
                            | BufferPurpose::WitnessInputCompactRunPositions
                            | BufferPurpose::WitnessInputCompactUniqueCount
                            | BufferPurpose::WitnessInputCompactSortTemp
                            | BufferPurpose::WitnessInputCompactScanTemp
                    ));
                    assert!(compact_scratch.insert(scratch.version));
                    assert!(!fixture.required_preproducers.contains(&scratch.version));
                }
                for relocation in [
                    compact.relocations.source_pointers,
                    compact.relocations.descriptor_workspace,
                    compact.relocations.output_pointers,
                ] {
                    assert!(fixture
                        .values
                        .version(ArenaCatalogValueId(relocation.logical.0))
                        .is_err());
                }
            }
        }
    }
    assert_eq!(seed_roots.len(), 4);
    assert_eq!(compact_scratch.len(), 30);
    assert_eq!(descriptors.len(), 3);
    let fixed = fixture.values.fixed_values();
    for setup in &fixture.lowered {
        let LoweredWitnessInputSetup::Compact(compact) = setup else {
            continue;
        };
        let descriptor = fixed
            .iter()
            .find(|fixed| fixed.value() == compact.descriptor_value)
            .unwrap();
        let FixedValueInitializer::InlineU32(words) = descriptor.initializer() else {
            panic!("compact descriptor must be inline u32")
        };
        assert_eq!(words.as_ref(), compact.contract.descriptor_words());
    }
}

#[test]
fn compact_sources_are_prior_level_producer_outputs() {
    let fixture = fixture();
    for setup in &fixture.lowered {
        let LoweredWitnessInputSetup::Compact(compact) = setup else {
            continue;
        };
        for source in &compact.sources {
            let catalog = BaseProducerCatalog::compile(fixture.executable.arena()).unwrap();
            let value = catalog.value(source.value).unwrap();
            let producer = writer(
                &fixture.authority,
                value.component.unwrap(),
                value.part.unwrap(),
            );
            assert!(producer.position().level < compact.position.level);
            assert_eq!(
                producer
                    .effect()
                    .accesses()
                    .iter()
                    .filter_map(|access| access.destination())
                    .filter(|destination| {
                        destination.value.version == source.version
                            && destination.value.elements.contains(source.elements)
                    })
                    .count(),
                1
            );
        }
    }
}

#[test]
fn seed_abi_and_compact_cub_steps_are_exact_and_stable() {
    let fixture = fixture();
    for setup in &fixture.lowered {
        match setup {
            LoweredWitnessInputSetup::Seed(seed) => {
                assert_eq!(seed.invocation.arguments.len(), 7);
                assert!(seed
                    .invocation
                    .arguments
                    .iter()
                    .enumerate()
                    .all(|(index, argument)| argument.ordinal as usize == index));
                assert_eq!(
                    seed.invocation.arguments[0].value,
                    AotArgumentValue::DevicePointer(Some(seed.scalar_source.binding))
                );
                assert_eq!(
                    seed.invocation.arguments[4].value,
                    AotArgumentValue::DevicePointerTable(
                        seed.outputs
                            .iter()
                            .map(|output| Some(output.binding))
                            .collect()
                    )
                );
            }
            LoweredWitnessInputSetup::Compact(compact) => {
                let first = projection::compact_steps_for_test(&compact.contract, 4, 8).unwrap();
                let repeated = projection::compact_steps_for_test(&compact.contract, 4, 8).unwrap();
                let changed = projection::compact_steps_for_test(&compact.contract, 8, 4).unwrap();
                assert_eq!(first, repeated);
                assert_ne!(first, changed);
                assert_eq!(first.len(), compact.contract.stages().len());
                for (expected, actual) in compact.contract.stages().iter().zip(&first) {
                    match (expected.execution, actual) {
                        (
                            WitnessInputCompactExecution::Kernel {
                                stage: expected_stage,
                                launch,
                            },
                            StaticCudaExecutionStepIdentity::KernelLaunch(actual),
                        ) => {
                            assert_eq!(actual.symbol(), expected_stage.symbol().as_bytes());
                            assert_eq!(actual.launch().grid, launch.grid);
                            assert_eq!(actual.launch().block, launch.block);
                        }
                        (
                            WitnessInputCompactExecution::Cub {
                                stage:
                                    WitnessInputCompactCubStage::StableRadixSortPairs { word, .. },
                                ..
                            },
                            StaticCudaExecutionStepIdentity::LibraryCall(
                                StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(
                                    actual,
                                ),
                            ),
                        ) => {
                            assert_eq!(actual.word(), word);
                            assert_eq!(actual.rows(), compact.contract.effect_geometry().sort_rows);
                            assert_eq!(actual.exact_temp_bytes(), 4);
                        }
                        (
                            WitnessInputCompactExecution::Cub {
                                stage: WitnessInputCompactCubStage::InclusiveSum,
                                ..
                            },
                            StaticCudaExecutionStepIdentity::LibraryCall(
                                StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(actual),
                            ),
                        ) => {
                            assert_eq!(actual.rows(), compact.contract.effect_geometry().sort_rows);
                            assert_eq!(actual.exact_temp_bytes(), 8);
                        }
                        _ => panic!("CUB/kernel stage kind drifted"),
                    }
                }
                assert!(matches!(
                    compact.contract.stages().first().unwrap().execution,
                    WitnessInputCompactExecution::Kernel {
                        stage: WitnessInputCompactKernelStage::Gather,
                        ..
                    }
                ));

                let invocation = projection::compact_invocation_for_test(compact, 4, 8).unwrap();
                assert_eq!(invocation.arguments.len(), 25);
                assert_eq!(invocation.arguments[22].value, AotArgumentValue::Usize(4));
                assert_eq!(invocation.arguments[24].value, AotArgumentValue::Usize(8));
                super::semantic::validate_exact_bindings(
                    &invocation,
                    &compact.effect,
                    Some((compact.descriptor_value, compact.descriptor_binding)),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn repeated_lowering_is_idempotent() {
    let fixture = fixture();
    let mut first_values = fixture.initial_values.clone();
    let first = lower_stage(fixture.executable.arena(), &mut first_values).unwrap();
    let mut second_values = fixture.initial_values.clone();
    let second = lower_stage(fixture.executable.arena(), &mut second_values).unwrap();
    assert_eq!(first, second);
    assert_eq!(first_values, second_values);
    assert_eq!(first, fixture.lowered);
    assert_eq!(first_values, fixture.values);

    let retained = first_values.clone();
    let repeated = lower_stage(fixture.executable.arena(), &mut first_values).unwrap();
    assert_eq!(repeated, first);
    assert_eq!(first_values, retained);
}
