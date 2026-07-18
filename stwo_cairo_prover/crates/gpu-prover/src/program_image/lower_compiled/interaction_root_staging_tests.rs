use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::InteractionCommitProgramAuthority;

use super::*;
use crate::compiled_proof::EffectAccess;
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    commit: LoweredInteractionCommit,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredInteractionRootStaging,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let planned = executable
            .arena()
            .commitment(CommitmentTreeId::Interaction)
            .unwrap();
        let authority = InteractionCommitProgramAuthority::compile(
            planned.commit_program.as_ref().unwrap(),
            planned.direct_retained_b2n_program.as_ref().unwrap(),
        )
        .unwrap();
        let inventory = BaseCommitInventory::compile_for(
            CommitInventoryKind::Interaction,
            executable.arena(),
            planned,
            authority.canonical(),
        )
        .unwrap();
        let mut before =
            adapter::SemanticValueMap::allocate_ordered(inventory.source_catalogs()).unwrap();
        let commit = interaction_commit_projection::lower_stage(
            executable.arena(),
            executable.transcript(),
            &mut before,
        )
        .unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(
            executable.arena(),
            executable.transcript(),
            &commit,
            &mut after,
        )
        .unwrap();
        Fixture {
            executable,
            commit,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_is_one_exact_interaction_root_copy() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    assert_eq!(
        lowered.stage(),
        ProofStage::BeforeTranscript(CairoTranscriptSegment::InteractionAndComposition)
    );
    assert_eq!(
        lowered.primitive(),
        ExecutionPrimitive::DeviceCopyD2D { bytes: 32 }
    );
    assert_eq!(lowered.source().arena.len_words, ROOT_WORDS);
    assert_eq!(lowered.destination().arena.len_words, ROOT_WORDS);
    assert_ne!(lowered.source().catalog, lowered.destination().catalog);
    assert_ne!(lowered.source().version, lowered.destination().version);
    assert_ne!(lowered.digest(), [0; 32]);
    assert_eq!(
        fixture.after.entries().count(),
        fixture.before.entries().count() + 1
    );

    let root = fixture.commit.final_root_output().unwrap();
    assert_eq!(lowered.source().arena, root.arena);
    assert_eq!(lowered.source().version, root.version);
    assert_eq!(
        fixture.before.version(lowered.source().catalog).unwrap(),
        root.version
    );
    assert_eq!(
        fixture
            .after
            .version(lowered.destination().catalog)
            .unwrap(),
        lowered.destination().version
    );
    let stage = InteractionCommitTranscriptStage::compile(
        fixture.executable.arena(),
        fixture.executable.transcript(),
    )
    .unwrap();
    assert_eq!(
        lowered.stage.interaction_root_operation(),
        stage.interaction_claim_operation() + 1
    );
}

#[test]
fn effect_reads_final_root_and_writes_the_transcript_output() {
    let lowered = &fixture().lowered;
    assert_eq!(lowered.effect().accesses().len(), 2);
    let [EffectAccess::Read { source }, EffectAccess::Write { destination }] =
        lowered.effect().accesses()
    else {
        panic!("root handoff must be one read followed by one write");
    };
    assert_eq!(source.binding, EffectBindingId(0));
    assert_eq!(source.value.version, lowered.source().version);
    assert_eq!(
        source.value.elements,
        ElementRange::new(0, ROOT_WORDS).unwrap()
    );
    assert_eq!(destination.binding, EffectBindingId(1));
    assert_eq!(destination.value.version, lowered.destination().version);
    assert_eq!(
        destination.value.elements,
        ElementRange::new(0, ROOT_WORDS).unwrap()
    );
    assert!(lowered.effect().module_globals().is_empty());
    assert!(lowered.effect().registered_fixed_source_reads().is_empty());
}

#[test]
fn projection_is_transactional_and_commit_tamper_fails_closed() {
    let fixture = fixture();
    assert!(validate_from(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &fixture.commit,
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
        &fixture.commit,
        &mut missing,
    )
    .is_err());
    assert_eq!(missing, unchanged);

    let planned = inventory(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &fixture.commit,
    )
    .unwrap();
    let mut occupied = fixture.before.clone();
    occupied
        .extend_ordered([planned.destination_catalog])
        .unwrap();
    let unchanged = occupied.clone();
    assert!(lower_stage(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &fixture.commit,
        &mut occupied,
    )
    .is_err());
    assert_eq!(occupied, unchanged);

    let mut forged_commit = fixture.commit.clone();
    interaction_commit_projection::tamper_receipt_digest_for_test(&mut forged_commit);
    let mut unchanged = fixture.before.clone();
    assert!(lower_stage(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        &forged_commit,
        &mut unchanged,
    )
    .is_err());
    assert_eq!(unchanged, fixture.before);
}

#[test]
fn root_receipt_rejects_value_effect_and_digest_drift() {
    let fixture = fixture();
    let assert_invalid = |forged: &LoweredInteractionRootStaging| {
        assert_eq!(
            validate_from(
                fixture.executable.arena(),
                fixture.executable.transcript(),
                &fixture.commit,
                &fixture.before,
                &fixture.after,
                forged,
            ),
            Err(InvocationShapeError::InvalidInteractionRootStaging)
        );
    };

    let mut source = fixture.lowered.clone();
    source.source.version.0 ^= 1;
    assert_invalid(&source);

    let mut destination = fixture.lowered.clone();
    destination.destination.version.0 ^= 1;
    assert_invalid(&destination);

    let mut effect_drift = fixture.lowered.clone();
    effect_drift.effect = EffectContract::new(
        vec![
            EffectAccess::Read {
                source: bound(
                    0,
                    effect_drift.source.version,
                    ElementRange::new(0, ROOT_WORDS - 1).unwrap(),
                ),
            },
            EffectAccess::Write {
                destination: bound(
                    1,
                    effect_drift.destination.version,
                    ElementRange::new(0, ROOT_WORDS).unwrap(),
                ),
            },
        ],
        vec![],
    )
    .unwrap();
    assert_invalid(&effect_drift);

    let mut digest = fixture.lowered.clone();
    digest.digest[0] ^= 1;
    assert_invalid(&digest);
}
