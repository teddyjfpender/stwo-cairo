use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use stwo_backend_cuda::{
    RelationChallengeAccessKind, RelationChallengeValueRole, RelationExecutionStage,
    RelationValueOwnership, RelationValueRole,
};

use super::*;
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    before: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredRelation,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let planned = executable.arena().relation();
        let authority = RelationExecutionAuthority::compile(
            planned.execution.kernel_program(),
            &planned.requirements,
            planned.launch_mode,
            executable.arena().protocol_identity().relation_tail_mode,
        )
        .unwrap();
        let challenge =
            RelationChallengeExpansionAuthority::compile(authority.program().max_alpha_powers)
                .unwrap();
        let inventory =
            RelationInventory::compile(executable.arena(), &authority, &challenge).unwrap();
        let mut roots = authority
            .values()
            .iter()
            .filter(|layout| layout.ownership == RelationValueOwnership::ExternalSource)
            .map(|layout| inventory.role(layout.role).unwrap().catalog)
            .collect::<BTreeSet<_>>();
        roots.insert(inventory.drawn().catalog);
        let before = adapter::SemanticValueMap::allocate_ordered(roots).unwrap();
        let mut after = before.clone();
        let lowered = lower_stage(executable.arena(), &mut after).unwrap();
        Fixture {
            executable,
            before,
            after,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_transaction_is_expansion_body_tail_exact() {
    let fixture = fixture();
    let authority = fixture.lowered.authority();
    assert_eq!(authority.program().batches.len(), 68);
    assert_eq!(authority.instances().len(), 45);
    assert_eq!(
        authority
            .program()
            .batches
            .iter()
            .map(|batch| batch.columns.len())
            .sum::<usize>(),
        807
    );
    assert_eq!(authority.program().template_use_count, 1_566);

    let expansion = fixture.lowered.challenge().authority();
    assert_eq!(expansion.drawn_words(), 8);
    assert_eq!(
        expansion.accesses()[0].role,
        RelationChallengeValueRole::DrawnZAlpha
    );
    assert_eq!(
        expansion.accesses()[0].kind,
        RelationChallengeAccessKind::Read
    );
    assert_eq!(expansion.child().grid, [1, 1, 1]);
    assert_eq!(expansion.child().block, [1, 1, 1]);

    let [body, tail] = fixture.lowered.wrappers();
    assert_eq!(body.authority().stage, RelationExecutionStage::FusedBody);
    assert_eq!(
        tail.authority().stage,
        RelationExecutionStage::SegmentedTail
    );
    assert_eq!(body.authority().children.len(), 1);
    assert_eq!(tail.authority().children.len(), 5);
    assert_eq!(body.accesses().len(), body.authority().accesses.len());
    assert_eq!(tail.accesses().len(), tail.authority().accesses.len());
    assert_ne!(fixture.lowered.digest(), [0; 32]);
}

#[test]
fn only_real_sources_and_the_transcript_draw_preexist() {
    let fixture = fixture();
    let challenge = fixture.lowered.challenge();
    assert_eq!(
        fixture.before.version(challenge.drawn.catalog).unwrap(),
        challenge.drawn_version()
    );
    assert!(fixture.before.version(challenge.alpha.catalog).is_err());
    assert!(fixture.before.version(challenge.z.catalog).is_err());
    assert_eq!(
        fixture.after.version(challenge.alpha.catalog).unwrap(),
        challenge.alpha_version()
    );
    assert_eq!(
        fixture.after.version(challenge.z.catalog).unwrap(),
        challenge.z_version()
    );

    for role in fixture.lowered.roles() {
        match role.ownership {
            RelationValueOwnership::ExternalSource => {
                assert_eq!(
                    fixture.before.version(role.arena.catalog).unwrap(),
                    role.first_version.unwrap()
                );
            }
            RelationValueOwnership::TranscriptChallenge => {
                assert!(fixture.before.version(role.arena.catalog).is_err());
                assert!(role.first_version.is_some());
            }
            RelationValueOwnership::PreparedMetadata | RelationValueOwnership::ReservedUnused => {
                assert_eq!((role.first_version, role.final_version), (None, None));
                assert!(fixture.before.version(role.arena.catalog).is_err());
            }
            RelationValueOwnership::ExecutionOutput | RelationValueOwnership::ExecutionScratch => {
                assert!(role.first_version.is_some());
                assert!(role.final_version.is_some());
            }
        }
    }
}

#[test]
fn reserved_storage_and_host_mask_never_become_device_dependencies() {
    let fixture = fixture();
    let reserved = fixture
        .lowered
        .roles()
        .iter()
        .filter(|role| role.ownership == RelationValueOwnership::ReservedUnused)
        .map(|role| role.role)
        .collect::<Vec<_>>();
    assert!(!reserved.is_empty());
    for wrapper in fixture.lowered.wrappers() {
        assert!(wrapper
            .accesses()
            .iter()
            .all(|access| !reserved.contains(&access.role)));
    }
    let body = fixture.lowered.wrappers()[0].authority();
    assert!(matches!(
        body.invocation.arguments[9].value,
        stwo_backend_cuda::RelationInvocationValue::HostMask(_)
    ));
    assert!(fixture
        .lowered
        .roles()
        .iter()
        .all(|role| role.role != RelationValueRole::InverseScratchUnused
            || role.ownership == RelationValueOwnership::ReservedUnused));
}

#[test]
fn segmented_tail_publishes_one_final_transition_per_instance_coordinate() {
    let fixture = fixture();
    let transitioned = fixture
        .lowered
        .roles()
        .iter()
        .filter(|role| {
            matches!(role.role, RelationValueRole::OutputCoordinate { .. })
                && role.first_version != role.final_version
        })
        .count();
    assert_eq!(transitioned, 45 * 4);

    let claimed = fixture
        .lowered
        .roles()
        .iter()
        .filter(|role| matches!(role.role, RelationValueRole::ClaimedSum { .. }))
        .collect::<Vec<_>>();
    assert_eq!(claimed.len(), 45);
    assert!(claimed
        .iter()
        .all(|role| { role.first_version.is_some() && role.first_version == role.final_version }));
}

#[test]
fn failure_and_validation_are_whole_transaction_atomic() {
    let fixture = fixture();
    let mut missing = adapter::SemanticValueMap::allocate_ordered(std::iter::empty()).unwrap();
    let untouched = missing.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut missing).is_err());
    assert_eq!(missing, untouched);

    validate_from(
        fixture.executable.arena(),
        &fixture.before,
        &fixture.after,
        &fixture.lowered,
    )
    .unwrap();
    let mut changed = fixture.lowered.clone();
    changed.digest[0] ^= 1;
    assert!(validate_from(
        fixture.executable.arena(),
        &fixture.before,
        &fixture.after,
        &changed,
    )
    .is_err());
}
