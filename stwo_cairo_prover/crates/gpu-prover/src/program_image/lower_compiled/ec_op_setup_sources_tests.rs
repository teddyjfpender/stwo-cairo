use std::collections::BTreeSet;

use super::*;
use crate::compiled_proof::ValueVersion;

fn generated() -> (
    std::sync::Arc<crate::shape_executable::ShapeExecutable>,
    SemanticValueMap,
    super::super::producer_prefix::BaseProducerAuthority,
) {
    let executable = super::super::tests::generated_sn2_replacement();
    let mut values =
        SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>()).unwrap();
    let authority = super::super::producer_prefix::BaseProducerAuthority::compile_replacement_into(
        executable.arena(),
        &mut values,
    )
    .unwrap();
    (executable, values, authority)
}

#[test]
fn generated_sn2_binds_all_five_ec_setup_sources_by_role() {
    let (executable, values, base) = generated();
    let setup = bind(executable.arena(), &base.producers, &values).unwrap();
    assert_eq!(setup.source_count(), EC_OP_SETUP_SOURCE_COUNT);
    assert_eq!(setup.unresolved_origin_count(), EC_OP_SETUP_SOURCE_COUNT);
    assert_eq!(
        setup.missing_origin_authorities(),
        [
            MissingEcOpSetupOriginAuthority::SegmentStartExternalInput,
            MissingEcOpSetupOriginAuthority::BatchedMultiplicityClear,
            MissingEcOpSetupOriginAuthority::GenericWitnessFeeds,
        ]
    );
    assert_eq!(setup.segment_start.value.value_words, 0..1);

    assert_eq!(
        setup.multiplicities.each_ref().map(|source| source.role),
        [
            EcOpMultiplicityRole::AddressCounts,
            EcOpMultiplicityRole::BigCounts,
            EcOpMultiplicityRole::SmallCounts,
            EcOpMultiplicityRole::RangeCheck8Counts,
        ]
    );
    assert_eq!(
        setup
            .multiplicities
            .each_ref()
            .map(|source| source.value.value_words.len()),
        [8192, 64, 1024, 256]
    );
    assert_eq!(
        setup
            .multiplicities
            .each_ref()
            .map(|source| source.clear_destination_index)
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    for source in &setup.multiplicities {
        assert_eq!(
            source.additive_owners.last(),
            Some(&EcOpMultiplicityAdditiveOwner::NativeEcOp)
        );
        assert_ne!(source.source, source.native_destination);
    }
    for role in [
        EcOpMultiplicityRole::AddressCounts,
        EcOpMultiplicityRole::BigCounts,
        EcOpMultiplicityRole::SmallCounts,
    ] {
        let source = setup
            .multiplicities
            .iter()
            .find(|source| source.role == role)
            .unwrap();
        assert!(!source
            .additive_owners
            .contains(&EcOpMultiplicityAdditiveOwner::PublicMemorySeed));
    }
    assert!(executable
        .arena()
        .multiplicity()
        .unwrap()
        .public_memory_seed
        .is_none());
    let range_check = setup
        .multiplicities
        .iter()
        .find(|source| source.role == EcOpMultiplicityRole::RangeCheck8Counts)
        .unwrap();
    assert!(!range_check
        .additive_owners
        .contains(&EcOpMultiplicityAdditiveOwner::PublicMemorySeed));

    let setup_versions = std::iter::once(setup.segment_start.source)
        .chain(setup.multiplicities.iter().map(|source| source.source))
        .collect::<BTreeSet<_>>();
    let required =
        super::super::compiled_base_prefix::validate_witness_writer_transitions_for_test(
            &base, &values,
        )
        .unwrap();
    assert_eq!(setup_versions.len(), EC_OP_SETUP_SOURCE_COUNT);
    assert!(setup_versions.is_subset(&required));

    let catalog = BaseProducerCatalog::compile(executable.arena()).unwrap();
    let setup_required = required
        .iter()
        .copied()
        .filter(|version| {
            values.entries().any(|(id, candidate)| {
                candidate == *version
                    && matches!(
                        catalog.value(id).unwrap().purpose,
                        BufferPurpose::EcOpSegmentStart
                            | BufferPurpose::RuntimeMultiplicity
                            | BufferPurpose::FixedMultiplicity
                    )
            })
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(setup_required, setup_versions);
}

#[test]
fn setup_binding_rejects_native_value_and_effect_drift() {
    let (executable, values, base) = generated();

    let mut changed = base.producers.clone();
    let contract = changed
        .iter_mut()
        .find_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp { contract, .. } => Some(contract),
            _ => None,
        })
        .unwrap();
    contract.invocation.segment_start.version = Some(ValueVersion(u32::MAX));
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let mut changed = base.producers.clone();
    let contract = changed
        .iter_mut()
        .find_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp { contract, .. } => Some(contract),
            _ => None,
        })
        .unwrap();
    contract.invocation.multiplicities.swap(0, 1);
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let setup = bind(executable.arena(), &base.producers, &values).unwrap();
    let mut transitioned = values.clone();
    transitioned
        .transition(setup.segment_start.value.value)
        .unwrap();
    assert!(bind(executable.arena(), &base.producers, &transitioned).is_err());
}
