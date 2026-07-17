use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{WitnessCasmInputColumnValue, WitnessCasmInputRowDomain};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::super::compiled_base_prefix;
use super::super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::*;
use crate::arena_plan::BufferPurpose;
use crate::compiled_proof::{AotArgumentValue, EffectBindingId, StaticCudaWrapperId};
use crate::shape_executable::ShapeExecutable;

mod adversarial;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    authority: BaseProducerAuthority,
    initial_values: adapter::SemanticValueMap,
    values: adapter::SemanticValueMap,
    required_preproducers: BTreeSet<ValueVersion>,
    lowered: Vec<LoweredWitnessCasmInput>,
}

fn executable() -> Arc<ShapeExecutable> {
    static EXECUTABLE: OnceLock<Arc<ShapeExecutable>> = OnceLock::new();
    Arc::clone(EXECUTABLE.get_or_init(super::super::tests::generated_sn2_replacement))
}

fn fixture() -> &'static Fixture {
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
        .expect("generated CASM endpoint must have one recorded writer")
}

#[test]
fn generated_sn2_closes_exactly_36_active_casm_writer_sources() {
    let fixture = fixture();
    let planned = fixture
        .executable
        .arena()
        .witness()
        .components
        .iter()
        .filter(|component| component.input_casm.is_some())
        .map(|component| (component.component, component.part))
        .collect::<Vec<_>>();
    let lowered = fixture
        .lowered
        .iter()
        .map(|lane| (lane.component, lane.part))
        .collect::<Vec<_>>();
    assert_eq!(planned, lowered);
    assert_eq!(lowered.len(), 9);
    assert!(fixture
        .lowered
        .windows(2)
        .all(|pair| pair[0].position.ordinal < pair[1].position.ordinal));

    let outputs = fixture
        .lowered
        .iter()
        .flat_map(|lane| &lane.outputs)
        .collect::<Vec<_>>();
    let output_versions = outputs
        .iter()
        .map(|output| output.version)
        .collect::<BTreeSet<_>>();
    assert_eq!(outputs.len(), 36);
    assert_eq!(output_versions.len(), 36);
    assert_eq!(
        fixture
            .lowered
            .iter()
            .map(LoweredWitnessCasmInput::active_output_count)
            .sum::<usize>(),
        36
    );
    assert_eq!(
        fixture
            .lowered
            .iter()
            .map(LoweredWitnessCasmInput::inactive_mechanical_output_count)
            .sum::<usize>(),
        0
    );
    assert!(output_versions.is_subset(&fixture.required_preproducers));

    for lane in &fixture.lowered {
        assert_eq!(lane.outputs.len(), 4);
        let consumer = writer(&fixture.authority, lane.component, lane.part);
        assert_eq!(consumer.position(), lane.position);
        for output in &lane.outputs {
            assert_eq!(output.writer_use, WitnessCasmWriterUse::Active);
            assert_eq!(
                consumer
                    .effect()
                    .accesses()
                    .iter()
                    .filter_map(|access| access.source())
                    .filter(|source| {
                        source.value.version == output.version
                            && source.value.elements == output.elements
                    })
                    .count(),
                1,
                "{} input {}",
                lane.component,
                output.ordinal
            );
            assert_eq!(
                lane.effect
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
}

#[test]
fn one_reused_staging_slot_has_nine_explicit_host_ingress_versions() {
    let fixture = fixture();
    let first = &fixture.lowered[0].staging;
    assert!(fixture.initial_values.version(first.value).is_err());
    assert_eq!(first.previous, None);
    let mut versions = BTreeSet::new();
    for (index, lane) in fixture.lowered.iter().enumerate() {
        assert_eq!(lane.staging.arena, first.arena);
        assert_eq!(lane.staging.value, first.value);
        assert!(versions.insert(lane.staging.version));
        if index > 0 {
            assert_eq!(
                lane.staging.previous,
                Some(fixture.lowered[index - 1].staging.version)
            );
        }
        let read = lane.effect.accesses()[0].source().unwrap();
        assert_eq!(read.binding, lane.staging.binding);
        assert_eq!(read.value.version, lane.staging.version);
        assert_eq!(read.value.elements, lane.staging.elements);
        assert_eq!(
            lane.staging.elements.len(),
            lane.contract.requirements().staging_words
        );
    }
    assert_eq!(versions.len(), 9);
    assert_eq!(
        fixture.values.version(first.value).unwrap(),
        fixture.lowered.last().unwrap().staging.version
    );
    assert_eq!(
        fixture.values.versions_for(first.value).collect::<Vec<_>>(),
        fixture
            .lowered
            .iter()
            .map(|lane| lane.staging.version)
            .collect::<Vec<_>>()
    );
    validate(
        fixture.executable.arena(),
        &fixture.values,
        &fixture.lowered,
    )
    .unwrap();
}

#[test]
fn arena_roles_extents_row_zero_domain_and_optional_iota_abi_are_exact() {
    let fixture = fixture();
    let catalog = BaseProducerCatalog::compile(fixture.executable.arena()).unwrap();
    for lane in &fixture.lowered {
        let staging = catalog.value(lane.staging.value).unwrap();
        assert_eq!(staging.component, None);
        assert_eq!(staging.part, None);
        assert_eq!(staging.purpose, BufferPurpose::WitnessInput);
        assert!(staging.words >= lane.contract.requirements().staging_words);
        assert_eq!(
            lane.contract.row_domain(),
            WitnessCasmInputRowDomain::RealPrefixWithRowZeroPaddingV1
        );
        assert_eq!(lane.invocation.arguments.len(), 8);
        assert_eq!(
            lane.invocation.arguments[0].value,
            AotArgumentValue::DevicePointer(Some(lane.staging.binding))
        );
        assert_eq!(
            lane.invocation.arguments[1].value,
            AotArgumentValue::U32(lane.contract.fixed_words()[1])
        );
        assert_eq!(
            lane.invocation.arguments[2].value,
            AotArgumentValue::U32(lane.contract.fixed_words()[2])
        );
        assert_eq!(
            lane.invocation.arguments[7].value,
            AotArgumentValue::DevicePointer(None)
        );
        assert_eq!(lane.effect.accesses().len(), 5);
        for output in &lane.outputs {
            let value = catalog.value(output.value).unwrap();
            assert_eq!(value.component, Some(lane.component));
            assert_eq!(value.part, Some(lane.part));
            assert_eq!(value.purpose, BufferPurpose::WitnessInput);
            assert_eq!(value.ordinal, output.ordinal);
            assert_eq!(
                output.elements.len(),
                lane.contract.requirements().consumer_rows
            );
        }
    }
}

#[test]
fn real_blake_casm_lane_owns_a_distinct_iota_output_and_linkable_effect() {
    let executable = super::super::tests::generated_casm_blake_replacement();
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    BaseProducerAuthority::compile_replacement_into(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        &mut values,
    )
    .unwrap();
    let lowered = lower_stage(executable.arena(), &mut values).unwrap();
    let blake = lowered
        .iter()
        .find(|lane| lane.component == "blake_compress_opcode")
        .unwrap();
    assert!(blake.contract.requirements().include_iota);
    assert_eq!(blake.outputs.len(), 5);
    assert_eq!(
        blake
            .outputs
            .iter()
            .map(|output| output.value_kind)
            .collect::<Vec<_>>(),
        [
            WitnessCasmInputColumnValue::StateWord(0),
            WitnessCasmInputColumnValue::StateWord(1),
            WitnessCasmInputColumnValue::StateWord(2),
            WitnessCasmInputColumnValue::Enabler,
            WitnessCasmInputColumnValue::Iota,
        ]
    );
    let iota = &blake.outputs[4];
    assert_eq!(
        blake.invocation.arguments[7].value,
        AotArgumentValue::DevicePointer(Some(iota.binding))
    );
    assert!(blake
        .outputs
        .iter()
        .enumerate()
        .all(|(index, output)| output.binding == EffectBindingId((index + 1) as u32)));
    assert_eq!(
        blake
            .outputs
            .iter()
            .map(|output| output.arena.logical)
            .collect::<BTreeSet<_>>()
            .len(),
        5
    );
    projection::validate_lowered(blake).unwrap();

    if stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        let target = stwo_backend_cuda_kernels::static_cuda_module_target_sms()[0];
        let linked = blake.contract.bind_static_build(target).unwrap().unwrap();
        let projected = project_static_wrapper(StaticCudaWrapperId(1), &linked, blake).unwrap();
        assert_eq!(projected.wrapper.accepted_effect(), blake.effect.id());
    } else {
        assert_eq!(blake.contract.bind_static_build(89).unwrap(), None);
    }
}

#[test]
fn lowering_is_transactional_byte_stable_and_idempotent() {
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
