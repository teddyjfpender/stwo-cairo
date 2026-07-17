use std::sync::{Arc, OnceLock};

use super::*;
use crate::compiled_proof::{AotArgumentValue, EffectAccess, StaticCudaWrapperId};
use crate::shape_executable::ShapeExecutable;

mod adversarial;

const SN2_DESTINATIONS: [(&str, usize); 22] = [
    ("pedersen_points_table_window_bits_18", 8_388_608),
    ("poseidon_round_keys", 64),
    ("range_check_11", 2_048),
    ("range_check_12", 4_096),
    ("range_check_18", 524_288),
    ("range_check_20", 8_388_608),
    ("range_check_3_3_3_3_3", 32_768),
    ("range_check_3_6_6_3", 262_144),
    ("range_check_4_3", 128),
    ("range_check_4_4", 256),
    ("range_check_4_4_4_4", 65_536),
    ("range_check_6", 64),
    ("range_check_7_2_5", 16_384),
    ("range_check_8", 256),
    ("range_check_9_9", 2_097_152),
    ("verify_bitwise_xor_4", 256),
    ("verify_bitwise_xor_7", 16_384),
    ("verify_bitwise_xor_8", 131_072),
    ("verify_bitwise_xor_9", 262_144),
    ("memory_address_to_id", 8_192),
    ("memory_id_to_big", 64),
    ("memory_id_to_big#small", 1_024),
];

struct Fixture {
    executable: Arc<ShapeExecutable>,
    initial: adapter::SemanticValueMap,
    values: adapter::SemanticValueMap,
    lowered: LoweredMultiplicityClear,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let initial =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
                .unwrap();
        let mut values = initial.clone();
        let lowered = lower_stage(executable.arena(), &mut values).unwrap();
        Fixture {
            executable,
            initial,
            values,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_clears_exactly_22_destinations_in_canonical_order() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    assert_eq!(
        lowered
            .destinations
            .iter()
            .map(|destination| (destination.name, destination.elements.len()))
            .collect::<Vec<_>>(),
        SN2_DESTINATIONS
    );
    assert_eq!(
        lowered
            .contract
            .effect_geometry()
            .destination_lengths
            .iter()
            .map(|&words| words as usize)
            .collect::<Vec<_>>(),
        SN2_DESTINATIONS
            .iter()
            .map(|(_, words)| *words)
            .collect::<Vec<_>>()
    );
    assert_eq!(lowered.contract.launch().grid, [32_768, 22, 1]);
    assert_eq!(lowered.contract.launch().block, [256, 1, 1]);
    assert_eq!(lowered.effect.accesses().len(), 23);
    assert!(lowered.effect.accesses()[..22]
        .iter()
        .all(|access| matches!(access, EffectAccess::Write { .. })));
    assert_eq!(
        lowered
            .effect
            .accesses()
            .iter()
            .filter(|access| access.source().is_some())
            .count(),
        1,
        "only the immutable length vector is read"
    );
    assert!(!lowered.effect.accesses().iter().any(|access| matches!(
        access,
        EffectAccess::ReadWrite { .. } | EffectAccess::Atomic { .. }
    )));
    assert!(matches!(
        lowered.effect.accesses()[22],
        EffectAccess::Read { .. }
    ));
    validate(
        fixture.executable.arena(),
        &fixture.values,
        &fixture.lowered,
    )
    .unwrap();
}

#[test]
fn runtime_destination_catalog_roles_are_exact() {
    let fixture = fixture();
    let catalog = BaseProducerCatalog::compile(fixture.executable.arena()).unwrap();
    let expected = [
        (
            "memory_address_to_id",
            Some("memory_address_to_id"),
            Some(TracePartId::Main),
            19,
        ),
        ("memory_id_to_big", Some("memory_id_to_big"), None, 20),
        ("memory_id_to_big#small", Some("memory_id_to_big"), None, 21),
    ];
    for (destination, (name, component, part, ordinal)) in
        fixture.lowered.destinations[19..].iter().zip(expected)
    {
        let value = catalog.value(destination.value).unwrap();
        assert_eq!(destination.name, name);
        assert_eq!(value.purpose, BufferPurpose::RuntimeMultiplicity);
        assert_eq!(value.component, component);
        assert_eq!(value.part, part);
        assert_eq!(value.ordinal, ordinal);
        assert_eq!(value.logical, destination.arena.logical);
        assert_eq!(value.physical, destination.arena.physical);
    }
}

#[test]
fn clear_outputs_are_catalog_first_with_no_fabricated_predecessors() {
    let fixture = fixture();
    let (catalog_first, transitions, fixed) = fixture.values.allocation_classes();
    assert_eq!(fixture.lowered.destinations.len(), 22);
    assert_eq!(fixed, [fixture.lowered.lengths_value].into_iter().collect());
    for destination in &fixture.lowered.destinations {
        assert!(catalog_first.contains(&destination.version));
        assert!(!transitions.contains(&destination.version));
        assert_eq!(
            fixture
                .values
                .versions_for(destination.value)
                .collect::<Vec<_>>(),
            [destination.version]
        );
        assert_eq!(
            fixture.values.version(destination.value).unwrap(),
            destination.version
        );
    }
    assert_eq!(
        fixture.values.allocated_versions().count(),
        fixture.initial.allocated_versions().count() + 23
    );
    for relocation in [
        fixture.lowered.relocations.destination_pointers,
        fixture.lowered.relocations.destination_lengths,
    ] {
        assert!(fixture
            .values
            .version(ArenaCatalogValueId(relocation.logical.0))
            .is_err());
    }
}

#[test]
fn invocation_owns_every_destination_and_the_exact_length_constant_once() {
    let fixture = fixture();
    let arguments = &fixture.lowered.invocation.arguments;
    assert_eq!(arguments.len(), 4);
    assert_eq!(
        arguments[0].value,
        AotArgumentValue::DevicePointerTable(
            fixture
                .lowered
                .destinations
                .iter()
                .map(|destination| Some(destination.binding))
                .collect()
        )
    );
    assert_eq!(
        arguments[1].value,
        AotArgumentValue::DeviceFixedU32 {
            value: fixture.lowered.lengths_value,
            binding: fixture.lowered.lengths_binding,
        }
    );
    assert_eq!(arguments[2].value, AotArgumentValue::U32(22));
    assert_eq!(arguments[3].value, AotArgumentValue::U32(8_388_608));
    assert!(fixture.lowered.effect.module_globals().is_empty());
}

#[test]
fn linked_projection_is_exact_when_a_local_static_build_receipt_exists() {
    let fixture = fixture();
    let linked = fixture.lowered.contract.bind_static_build(89).unwrap();
    if let Some(linked) = linked {
        let projected =
            project_static_wrapper(StaticCudaWrapperId(1), &linked, &fixture.lowered).unwrap();
        assert_eq!(
            projected.wrapper.accepted_effect(),
            fixture.lowered.effect.id()
        );
    }
}
