use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::compiled_proof::{EffectAccess, ExecutionPrimitive};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredInteractionClaimStaging,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let (sources, ..) = inventory(executable.arena(), executable.transcript()).unwrap();
        let before = adapter::SemanticValueMap::allocate_ordered(
            sources.iter().map(|source| source.catalog),
        )
        .unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(executable.arena(), executable.transcript(), &mut after).unwrap();
        Fixture {
            executable,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_is_one_exact_45_copy_ordered_composite() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    assert_eq!(
        lowered.stage(),
        ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)
    );
    assert_eq!(lowered.sources().len(), 45);
    assert_eq!(lowered.children().len(), 45);
    assert_eq!(lowered.child_effects().len(), 45);
    assert_eq!(lowered.destination().words, 180);
    assert_eq!(lowered.destination().arena.len_words, 180);
    assert_eq!(lowered.boundary_effect().accesses().len(), 90);
    assert_ne!(lowered.digest(), [0; 32]);
    assert_eq!(
        lowered.stage.interaction_claim_felts(),
        u32::try_from(lowered.sources().len()).unwrap()
    );

    let ExecutionPrimitive::OrderedComposite { children } = lowered.primitive() else {
        panic!("claim staging must remain one ordered composite");
    };
    assert_eq!(children.as_ref(), lowered.children());
    assert!(lowered
        .children()
        .iter()
        .all(|child| child.invocation.is_none()
            && matches!(
                child.primitive,
                ExecutionPrimitive::DeviceCopyD2D { bytes: COPY_BYTES }
            )));

    let ordinals = lowered
        .sources()
        .iter()
        .map(|source| source.key.relation_ordinal)
        .collect::<BTreeSet<_>>();
    assert_eq!(ordinals, (0..45).collect());
    let source_versions = lowered
        .sources()
        .iter()
        .map(|source| source.version)
        .collect::<BTreeSet<_>>();
    assert_eq!(source_versions.len(), 45);
    assert!(!source_versions.contains(&lowered.destination().version));
    assert_eq!(
        fixture.after.entries().count(),
        fixture.before.entries().count() + 1
    );
}

#[test]
fn every_child_and_boundary_range_is_typed_contiguous_and_disjoint() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    for (index, ((source, child), effect)) in lowered
        .sources()
        .iter()
        .zip(lowered.children())
        .zip(lowered.child_effects())
        .enumerate()
    {
        assert_eq!(child.effect, effect.id());
        assert_eq!(effect.accesses().len(), 2);
        let [EffectAccess::Read { source: read }, EffectAccess::Write { destination: write }] =
            effect.accesses()
        else {
            panic!("copy child must be one read followed by one write");
        };
        assert_eq!(read.binding, EffectBindingId(0));
        assert_eq!(read.value.version, source.version);
        assert_eq!(
            read.value.elements,
            ElementRange::new(0, CLAIM_WORDS).unwrap()
        );
        assert_eq!(write.binding, EffectBindingId(1));
        assert_eq!(write.value.version, lowered.destination().version);
        assert_eq!(
            write.value.elements,
            ElementRange::new(index * CLAIM_WORDS, (index + 1) * CLAIM_WORDS).unwrap()
        );

        let boundary = &lowered.boundary_effect().accesses()[index * 2..index * 2 + 2];
        let [EffectAccess::Read {
            source: boundary_read,
        }, EffectAccess::Write {
            destination: boundary_write,
        }] = boundary
        else {
            panic!("boundary must preserve child read/write order");
        };
        assert_eq!(boundary_read.value, read.value);
        assert_eq!(boundary_write.value, write.value);
        assert_eq!(
            boundary_read.binding,
            EffectBindingId(u32::try_from(index * 2).unwrap())
        );
        assert_eq!(
            boundary_write.binding,
            EffectBindingId(u32::try_from(index * 2 + 1).unwrap())
        );
    }
}

#[test]
fn source_order_is_the_shared_cairo_claim_order() {
    let fixture = fixture();
    let relation = fixture.executable.arena().relation();
    let ordered = fixture
        .lowered
        .sources()
        .iter()
        .map(|source| {
            let source_plan = &relation.source_plan[source.key.relation_ordinal as usize];
            assert_eq!(source.key.component, source_plan.batch.component);
            assert_eq!(source.key.part, source_plan.part);
            assert_eq!(
                usize::try_from(source.key.instance).unwrap(),
                source_plan.instance_index
            );
            interaction_claim_order_key(source_plan.batch, source_plan.instance_index).unwrap()
        })
        .collect::<Vec<_>>();
    assert!(ordered.windows(2).all(|pair| pair[0] < pair[1]));

    let memory_parts = fixture
        .lowered
        .sources()
        .iter()
        .filter(|source| source.key.component == "memory_id_to_big")
        .map(|source| source.key.part)
        .collect::<Vec<_>>();
    assert!(!memory_parts.is_empty());
    assert!(matches!(
        memory_parts.last(),
        Some(TracePartId::MemorySmall)
    ));
    assert!(memory_parts[..memory_parts.len() - 1]
        .iter()
        .all(|part| matches!(part, TracePartId::MemoryBig(_))));
}

#[test]
fn projection_is_transactional_and_receipt_tamper_fails_closed() {
    let fixture = fixture();
    assert!(validate_from(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &fixture.before,
        &fixture.after,
        &fixture.lowered,
    )
    .is_ok());

    let mut missing =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let unchanged = missing.clone();
    assert!(lower_stage(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &mut missing
    )
    .is_err());
    assert_eq!(missing, unchanged);

    let (_, destination, _) =
        inventory(fixture.executable.arena(), fixture.executable.transcript()).unwrap();
    let mut occupied = fixture.before.clone();
    occupied.extend_ordered([destination.catalog]).unwrap();
    let unchanged = occupied.clone();
    assert!(lower_stage(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &mut occupied
    )
    .is_err());
    assert_eq!(occupied, unchanged);

    let mut reordered = fixture.lowered.clone();
    reordered.children.swap(0, 1);
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &reordered,
        ),
        Err(InvocationShapeError::InvalidInteractionClaimStaging)
    );

    let mut forged_destination = fixture.lowered.clone();
    forged_destination.destination.words -= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &forged_destination,
        ),
        Err(InvocationShapeError::InvalidInteractionClaimStaging)
    );

    let mut forged_digest = fixture.lowered.clone();
    forged_digest.digest[0] ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            &fixture.before,
            &fixture.after,
            &forged_digest,
        ),
        Err(InvocationShapeError::InvalidInteractionClaimStaging)
    );
}
