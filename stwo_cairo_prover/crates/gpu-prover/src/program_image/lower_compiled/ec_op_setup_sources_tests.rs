use std::collections::BTreeSet;

use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

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
        PreProcessedTraceVariant::Canonical,
        &mut values,
    )
    .unwrap();
    (executable, values, authority)
}

#[test]
fn generated_sn2_binds_all_five_ec_setup_sources_by_role() {
    let (executable, values, base) = generated();
    let setup = bind(executable.arena(), &base, &values).unwrap();
    assert_eq!(setup.source_count(), EC_OP_SETUP_SOURCE_COUNT);
    assert_eq!(setup.unresolved_origin_count(), 1);
    assert_eq!(
        setup.missing_origin_authorities(),
        [MissingEcOpSetupOriginAuthority::SegmentStartExternalInput]
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
        let native_positions = source
            .additive_owners
            .iter()
            .enumerate()
            .filter_map(|(position, owner)| {
                (*owner == EcOpMultiplicityAdditiveOwner::NativeEcOp).then_some(position)
            })
            .collect::<Vec<_>>();
        assert_eq!(native_positions.len(), 1);
        let native_position = native_positions[0];
        let lineage = values.versions_for(source.value.value).collect::<Vec<_>>();
        assert_eq!(lineage.len(), source.additive_owners.len() + 1);
        assert_eq!(lineage[native_position], source.source);
        assert_eq!(lineage[native_position + 1], source.native_destination);
        assert_ne!(source.source, source.native_destination);
    }
    assert!(setup.multiplicities.iter().any(|source| {
        let native_position = source
            .additive_owners
            .iter()
            .position(|owner| *owner == EcOpMultiplicityAdditiveOwner::NativeEcOp)
            .unwrap();
        native_position + 1 < source.additive_owners.len()
    }));
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
    assert!(required.contains(&setup.segment_start.source));
    assert!(setup
        .multiplicities
        .iter()
        .all(|source| !required.contains(&source.source)));

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
    assert_eq!(setup_required, BTreeSet::from([setup.segment_start.source]));
}

#[test]
fn setup_binding_rejects_native_value_and_effect_drift() {
    let (executable, values, base) = generated();

    let mut changed = base.clone();
    let contract = changed
        .producers
        .iter_mut()
        .find_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp { contract, .. } => Some(contract),
            _ => None,
        })
        .unwrap();
    contract.invocation.segment_start.version = Some(ValueVersion(u32::MAX));
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let mut changed = base.clone();
    let contract = changed
        .producers
        .iter_mut()
        .find_map(|producer| match producer {
            SemanticBaseProducer::NativeEcOp { contract, .. } => Some(contract),
            _ => None,
        })
        .unwrap();
    contract.invocation.multiplicities.swap(0, 1);
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let setup = bind(executable.arena(), &base, &values).unwrap();
    let mut transitioned = values.clone();
    transitioned
        .transition(setup.segment_start.value.value)
        .unwrap();
    assert!(bind(executable.arena(), &base, &transitioned).is_err());
}

#[test]
fn setup_binding_rejects_clear_feed_and_post_witness_lineage_drift() {
    let (executable, values, base) = generated();
    let setup = bind(executable.arena(), &base, &values).unwrap();
    let target = &setup.multiplicities[0];

    let mut changed = base.clone();
    changed.multiplicity.clear.destinations[target.clear_destination_index].version =
        ValueVersion(u32::MAX);
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let feed_indices = base
        .multiplicity
        .after_producer
        .iter()
        .enumerate()
        .filter_map(|(index, feed)| feed.as_ref().map(|_| index))
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(feed_indices.len(), 2);
    let mut changed = base.clone();
    changed
        .multiplicity
        .after_producer
        .swap(feed_indices[0], feed_indices[1]);
    assert!(bind(executable.arena(), &changed, &values).is_err());

    let mut changed = base.clone();
    let post_index = changed
        .multiplicity
        .post_witness_current
        .iter()
        .position(|entry| entry.value == target.value.value)
        .unwrap();
    changed.multiplicity.post_witness_current[post_index].current = ValueVersion(u32::MAX);
    assert!(bind(executable.arena(), &changed, &values).is_err());
}
