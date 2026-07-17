use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{
    WitnessInputGatherAbiAccess, WitnessInputGatherAbiArgumentKind, WitnessInputGatherContract,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::arena_plan::BufferPurpose;
use crate::compiled_proof::{
    AotArgumentValue, EffectContract, FixedValueInitializer, ValueVersion,
};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    authority: BaseProducerAuthority,
    initial_values: adapter::SemanticValueMap,
    values: adapter::SemanticValueMap,
    required_preproducers: BTreeSet<ValueVersion>,
    fragments: Vec<witness_input_gather::LoweredWitnessInputGather>,
}

fn executable() -> Arc<ShapeExecutable> {
    static EXECUTABLE: OnceLock<Arc<ShapeExecutable>> = OnceLock::new();
    Arc::clone(EXECUTABLE.get_or_init(tests::generated_sn2_replacement))
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture)
}

fn build_fixture() -> Fixture {
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
    let required_preproducers = compiled_base_prefix::validate_witness_writer_transitions_for_test(
        &authority,
        &initial_values,
    )
    .unwrap();
    let mut values = initial_values.clone();
    let fragments = witness_input_gather::lower_stage(executable.arena(), &mut values).unwrap();
    Fixture {
        executable,
        authority,
        initial_values,
        values,
        required_preproducers,
        fragments,
    }
}

fn writer<'a>(
    authority: &'a BaseProducerAuthority,
    component: &'static str,
    part: Option<TracePartId>,
) -> &'a SemanticBaseProducer {
    authority
        .producers
        .iter()
        .find(|producer| {
            producer.producer().component == component && producer.producer().part == part
        })
        .expect("generated gather endpoint must have one scheduled writer")
}

#[test]
fn generated_sn2_five_gathers_and_native_ec_account_for_303_typed_values() {
    let fixture = fixture();
    let planned = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .filter(|component| component.input_gather.is_some())
        .map(|component| (component.component, component.part))
        .collect::<Vec<_>>();
    let lowered = fixture
        .fragments
        .iter()
        .map(|fragment| (fragment.component, fragment.part))
        .collect::<Vec<_>>();
    assert_eq!(planned.len(), 5);
    assert_eq!(lowered, planned);
    assert!(fixture
        .fragments
        .windows(2)
        .all(|pair| pair[0].position.ordinal < pair[1].position.ordinal));

    let outputs = fixture
        .fragments
        .iter()
        .flat_map(|fragment| &fragment.outputs)
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 171);
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.value)
            .collect::<BTreeSet<_>>()
            .len(),
        171
    );
    let gather_versions = outputs
        .iter()
        .map(|output| fixture.values.version(output.value).unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(gather_versions.len(), 171);
    let descriptor_versions = fixture
        .fragments
        .iter()
        .map(|fragment| fragment.descriptor_value)
        .collect::<BTreeSet<_>>();
    assert_eq!(descriptor_versions.len(), 5);
    let (_, transitions, fixed) = fixture.values.allocation_classes();
    assert!(descriptor_versions.is_subset(&fixed));
    assert!(descriptor_versions.is_disjoint(&transitions));
    assert!(descriptor_versions.is_disjoint(&gather_versions));

    for fragment in &fixture.fragments {
        fragment.contract.validate().unwrap();
        witness_input_gather::validate(fixture.executable.arena(), &fixture.values, fragment)
            .unwrap();
        let writer = writer(&fixture.authority, fragment.component, Some(fragment.part));
        assert_eq!(writer.position(), fragment.position);
        for output in &fragment.outputs {
            let version = fixture.values.version(output.value).unwrap();
            assert!(fixture.required_preproducers.contains(&version));
            let consumers = writer
                .effect()
                .accesses()
                .iter()
                .filter_map(|access| access.source())
                .filter(|source| {
                    source.value.version == version && source.value.elements == output.elements
                })
                .count();
            assert_eq!(
                consumers, 1,
                "{} input {}",
                fragment.component, output.ordinal
            );
            assert_eq!(output.elements.start, 0);
            assert_eq!(
                output.elements.end,
                fragment.contract.effect_geometry().consumer_rows as usize
            );
        }
    }

    let native = fixture
        .authority
        .producers
        .iter()
        .filter_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp {
                producer, contract, ..
            } => Some((producer, contract)),
            SemanticBaseProducer::Recorded(_) | SemanticBaseProducer::NativeBlakeGDirect { .. } => {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(native.len(), 1);
    let (producer, contract) = native[0];
    assert_eq!(producer.component, "ec_op_builtin");
    assert_eq!(producer.part, None);
    assert_eq!(contract.invocation.partial_input_columns.len(), 127);
    let (partial_iota, partial_inputs) = contract
        .invocation
        .partial_input_columns
        .split_last()
        .unwrap();
    assert_eq!(partial_inputs.len(), 126);
    let mut native_versions = BTreeSet::new();
    let partial_writer = writer(
        &fixture.authority,
        "partial_ec_mul_generic",
        Some(TracePartId::Main),
    );
    for partial in partial_inputs {
        let version = partial
            .version
            .expect("native EC partial output must have one semantic version");
        let binding = partial
            .binding
            .expect("native EC partial output must have one effect binding");
        assert_eq!(
            fixture.values.version(partial.value.value).unwrap(),
            version
        );
        assert!(!fixture.required_preproducers.contains(&version));
        assert!(native_versions.insert(version));
        assert_eq!(
            contract
                .effect
                .accesses()
                .iter()
                .filter_map(|access| access.destination())
                .filter(|destination| {
                    destination.binding == binding
                        && destination.value.version == version
                        && destination.value.elements.start == partial.value.value_words.start
                        && destination.value.elements.end == partial.value.value_words.end
                })
                .count(),
            1
        );
        assert_eq!(
            partial_writer
                .effect()
                .accesses()
                .iter()
                .filter_map(|access| access.source())
                .filter(|source| {
                    source.value.version == version
                        && source.value.elements.start == partial.value.value_words.start
                        && source.value.elements.end == partial.value.value_words.end
                })
                .count(),
            1
        );
    }
    let iota_version = partial_iota
        .version
        .expect("native EC partial iota must have one semantic version");
    let iota_binding = partial_iota
        .binding
        .expect("native EC partial iota must have one effect binding");
    assert_eq!(
        fixture.values.version(partial_iota.value.value).unwrap(),
        iota_version
    );
    assert!(!fixture.required_preproducers.contains(&iota_version));
    assert!(native_versions.insert(iota_version));
    assert_eq!(
        contract
            .effect
            .accesses()
            .iter()
            .filter_map(|access| access.destination())
            .filter(|destination| {
                destination.binding == iota_binding
                    && destination.value.version == iota_version
                    && destination.value.elements.start == partial_iota.value.value_words.start
                    && destination.value.elements.end == partial_iota.value.value_words.end
            })
            .count(),
        1
    );
    assert_eq!(
        partial_writer
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.source())
            .filter(|source| source.value.version == iota_version)
            .count(),
        0
    );
    assert_eq!(native_versions.len(), 127);
    assert!(gather_versions.is_disjoint(&native_versions));
    assert!(descriptor_versions.is_disjoint(&native_versions));

    // The old "six gathers / 303 output columns" shorthand mixed categories.
    // Generated SN2 has five gather operations plus one native-EC operation:
    // 171 gather output columns + 5 descriptor constants + 127 native outputs
    // = 303 typed semantic values, of which exactly 298 are columns.
    let stage_values = gather_versions
        .iter()
        .chain(&descriptor_versions)
        .chain(&native_versions)
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(stage_values.len(), 303);
    assert_eq!(gather_versions.len() + native_versions.len(), 298);

    let catalog = BaseProducerCatalog::compile(fixture.executable.arena()).unwrap();
    let purpose_by_version = fixture
        .initial_values
        .entries()
        .map(|(catalog_id, version)| (version, catalog.value(catalog_id).unwrap().purpose))
        .collect::<BTreeMap<_, _>>();
    let mut required_by_purpose = BTreeMap::new();
    for version in &fixture.required_preproducers {
        let purpose = *purpose_by_version
            .get(version)
            .expect("every required preproducer must retain its typed catalog purpose");
        *required_by_purpose.entry(purpose).or_insert(0_usize) += 1;
    }
    assert_eq!(
        required_by_purpose,
        BTreeMap::from([
            (BufferPurpose::WitnessInput, 236),
            (BufferPurpose::ExecutionTableBigLimb, 28),
            (BufferPurpose::ExecutionTableSmallLimb, 8),
            (BufferPurpose::ExecutionTableRawAddressToId, 1),
            (BufferPurpose::EcOpSegmentStart, 1),
            (BufferPurpose::RuntimeMultiplicity, 3),
            (BufferPurpose::FixedMultiplicity, 1),
        ])
    );
    assert_eq!(fixture.required_preproducers.len(), 278);
    assert!(gather_versions.is_subset(&fixture.required_preproducers));
    assert_eq!(
        fixture
            .required_preproducers
            .difference(&gather_versions)
            .count(),
        107
    );
}

#[test]
fn every_source_is_a_prior_level_subcomponent_output() {
    let fixture = fixture();
    for fragment in &fixture.fragments {
        for source in &fragment.sources {
            let producer = writer(&fixture.authority, source.producer, Some(TracePartId::Main));
            assert!(producer.position().level < fragment.position.level);
            let version = fixture.values.version(source.value).unwrap();
            let producers = producer
                .effect()
                .accesses()
                .iter()
                .filter_map(|access| access.destination())
                .filter(|destination| {
                    destination.value.version == version
                        && destination.value.elements.contains(source.elements)
                })
                .count();
            assert_eq!(producers, 1, "source {}", source.producer);
            assert_eq!(source.arena.logical.0, source.value.0);
        }
    }
}

#[test]
fn descriptors_are_single_constants_and_pointer_tables_stay_relocations() {
    let fixture = fixture();
    let fixed = fixture.values.fixed_values();
    for fragment in &fixture.fragments {
        let descriptor = fixed
            .iter()
            .find(|fixed| fixed.value() == fragment.descriptor_value)
            .expect("descriptor must have one fixed-value authority");
        assert_eq!(
            fixed
                .iter()
                .filter(|fixed| fixed.value() == fragment.descriptor_value)
                .count(),
            1
        );
        let FixedValueInitializer::InlineU32(words) = descriptor.initializer() else {
            panic!("gather descriptor must be inline u32")
        };
        assert_eq!(words.as_ref(), fragment.contract.descriptor_words());
        assert_eq!(
            fixture.values.version(ArenaCatalogValueId(
                fragment.relocations.descriptor_workspace.logical.0
            )),
            Err(InvocationShapeError::MissingSemanticValueMap(
                ArenaCatalogValueId(fragment.relocations.descriptor_workspace.logical.0)
            ))
        );
        for relocation in [
            fragment.relocations.source_pointers,
            fragment.relocations.output_pointers,
        ] {
            assert!(fixture
                .values
                .version(ArenaCatalogValueId(relocation.logical.0))
                .is_err());
        }

        let arguments = &fragment.invocation.arguments;
        assert_eq!(arguments.len(), 9);
        assert!(arguments
            .iter()
            .enumerate()
            .all(|(ordinal, argument)| argument.ordinal as usize == ordinal));
        assert_eq!(fragment.contract.abi().arguments().len(), 10);
        assert_eq!(
            fragment.contract.abi().arguments()[9].kind,
            WitnessInputGatherAbiArgumentKind::CudaStream
        );
        assert_eq!(
            arguments[0].value,
            AotArgumentValue::DevicePointerTable(
                fragment
                    .sources
                    .iter()
                    .map(|source| Some(source.binding))
                    .collect()
            )
        );
        assert_eq!(
            arguments[1].value,
            AotArgumentValue::DeviceFixedU32 {
                value: fragment.descriptor_value,
                binding: fragment.descriptor_binding,
            }
        );
        assert_eq!(
            arguments[6].value,
            AotArgumentValue::DevicePointerTable(
                fragment
                    .outputs
                    .iter()
                    .map(|output| Some(output.binding))
                    .collect()
            )
        );
        assert_eq!(
            fragment.effect.accesses().len(),
            fragment.sources.len() + 1 + fragment.outputs.len()
        );
        let descriptor_read = fragment.effect.accesses()[fragment.sources.len()]
            .source()
            .unwrap();
        assert_eq!(descriptor_read.binding, fragment.descriptor_binding);
        assert_eq!(descriptor_read.value.version, fragment.descriptor_value);
        assert_eq!(descriptor_read.value.elements.start, 0);
        assert_eq!(
            descriptor_read.value.elements.end,
            fragment.contract.descriptor_words().len()
        );
        assert_eq!(fragment.sources[0].pointer_words.start, 0);
        assert_eq!(
            fragment.sources.last().unwrap().pointer_words.end,
            fragment.relocations.source_pointers.len_words
        );
        assert_eq!(fragment.outputs[0].pointer_words.start, 0);
        assert_eq!(
            fragment.outputs.last().unwrap().pointer_words.end,
            fragment.relocations.output_pointers.len_words
        );
    }
}

#[test]
fn repeated_lowering_is_byte_stable_and_idempotent() {
    let fixture = fixture();
    let mut first_values = fixture.initial_values.clone();
    let first =
        witness_input_gather::lower_stage(fixture.executable.arena(), &mut first_values).unwrap();
    let mut second_values = fixture.initial_values.clone();
    let second =
        witness_input_gather::lower_stage(fixture.executable.arena(), &mut second_values).unwrap();
    assert_eq!(first, second);
    assert_eq!(first_values, second_values);
    assert_eq!(first, fixture.fragments);
    assert_eq!(first_values, fixture.values);

    let retained = first_values.clone();
    let repeated =
        witness_input_gather::lower_stage(fixture.executable.arena(), &mut first_values).unwrap();
    assert_eq!(repeated, first);
    assert_eq!(first_values, retained);
}

#[test]
fn missing_source_versions_fail_without_mutating_the_map() {
    let executable = executable();
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = values.clone();
    assert!(witness_input_gather::lower_stage(executable.arena(), &mut values).is_err());
    assert_eq!(values, unchanged);
}

#[test]
fn planned_source_order_output_and_descriptor_drift_fail_closed() {
    let fixture = fixture();
    let fragment = fixture
        .fragments
        .iter()
        .find(|fragment| fragment.sources.len() > 1 && fragment.outputs.len() > 1)
        .unwrap();
    let component = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| {
            component.component == fragment.component && component.part == fragment.part
        })
        .unwrap()
        .clone();

    let mut mutations: Vec<Box<dyn Fn(&mut crate::arena_plan::PlannedWitnessComponent)>> =
        Vec::new();
    mutations.push(Box::new(|component| {
        component.input_gather.as_mut().unwrap().sources[0].len_words -= 1;
    }));
    mutations.push(Box::new(|component| {
        component.input_gather.as_mut().unwrap().sources.swap(0, 1);
    }));
    mutations.push(Box::new(|component| {
        component
            .input_gather
            .as_mut()
            .unwrap()
            .producers
            .swap(0, 1);
    }));
    mutations.push(Box::new(|component| {
        component
            .input_gather
            .as_mut()
            .unwrap()
            .slots
            .consumer_input_columns
            .swap(0, 1);
    }));
    mutations.push(Box::new(|component| {
        component
            .input_gather
            .as_mut()
            .unwrap()
            .requirements
            .descriptor_words += 1;
    }));
    mutations.push(Box::new(|component| {
        component.input_gather.as_mut().unwrap().requirements.edges[0]
            .edge
            .word_base += 1;
    }));
    for mutate in mutations {
        let mut changed = component.clone();
        mutate(&mut changed);
        let mut values = fixture.initial_values.clone();
        assert!(witness_input_gather::lower_component(
            fixture.executable.arena(),
            &changed,
            &mut values,
        )
        .is_err());
        assert_eq!(values, fixture.initial_values);
    }
}

#[test]
fn abi_and_retained_fragment_mutations_fail_closed() {
    let fixture = fixture();
    let fragment = &fixture.fragments[0];
    let authoritative = fragment.contract.abi().arguments();
    assert!(witness_input_gather::invocation_using_abi_for_test(
        fragment,
        &authoritative[..authoritative.len() - 1],
    )
    .is_err());
    let mut rotated = authoritative.to_vec();
    rotated.rotate_right(1);
    assert!(witness_input_gather::invocation_using_abi_for_test(fragment, &rotated).is_err());
    for index in 0..authoritative.len() {
        let mut changed = authoritative.to_vec();
        changed[index].ordinal ^= 0x80;
        assert!(witness_input_gather::invocation_using_abi_for_test(fragment, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].name = "wrong_role";
        assert!(witness_input_gather::invocation_using_abi_for_test(fragment, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].kind =
            if changed[index].kind == WitnessInputGatherAbiArgumentKind::CudaStream {
                WitnessInputGatherAbiArgumentKind::U32
            } else {
                WitnessInputGatherAbiArgumentKind::CudaStream
            };
        assert!(witness_input_gather::invocation_using_abi_for_test(fragment, &changed).is_err());
        let mut changed = authoritative.to_vec();
        changed[index].access =
            if changed[index].access == WitnessInputGatherAbiAccess::OrderedExecutionStream {
                WitnessInputGatherAbiAccess::EdgeCount
            } else {
                WitnessInputGatherAbiAccess::OrderedExecutionStream
            };
        assert!(witness_input_gather::invocation_using_abi_for_test(fragment, &changed).is_err());
    }

    let mut mutations = Vec::new();
    let mut changed = fragment.clone();
    changed.sources[0].arena.len_words -= 1;
    mutations.push(changed);
    let mut changed = fragment.clone();
    changed.outputs.swap(0, 1);
    mutations.push(changed);
    let mut changed = fragment.clone();
    changed.descriptor_value = fixture.values.version(changed.outputs[0].value).unwrap();
    mutations.push(changed);
    let mut changed = fragment.clone();
    changed.position.level += 1;
    mutations.push(changed);
    let mut changed = fragment.clone();
    changed.invocation.arguments.swap(0, 1);
    mutations.push(changed);
    let mut changed = fragment.clone();
    let AotArgumentValue::DevicePointerTable(entries) = &mut changed.invocation.arguments[0].value
    else {
        unreachable!()
    };
    entries.pop();
    mutations.push(changed);
    let mut changed = fragment.clone();
    let mut accesses = changed.effect.accesses().to_vec();
    accesses[0].source_mut().unwrap().value.elements.end -= 1;
    changed.effect = EffectContract::new(accesses, Vec::new()).unwrap();
    mutations.push(changed);
    let mut changed = fragment.clone();
    changed.contract = fixture
        .fragments
        .iter()
        .find(|candidate| candidate.contract.identity() != changed.contract.identity())
        .unwrap()
        .contract
        .clone();
    mutations.push(changed);

    for changed in mutations {
        assert!(witness_input_gather::validate(
            fixture.executable.arena(),
            &fixture.values,
            &changed,
        )
        .is_err());
    }
}

#[test]
fn canonical_contract_cannot_be_recompiled_from_descriptor_drift() {
    let fixture = fixture();
    for fragment in &fixture.fragments {
        let mut requirements = fragment.contract.requirements().clone();
        requirements.descriptor_words += 1;
        assert!(WitnessInputGatherContract::compile(&requirements).is_err());
    }
}
