use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::shape_executable::ShapeExecutable;
use crate::transcript_plan::{CairoTranscriptOutput, CairoTranscriptSegment};

struct Fixture {
    executable: Arc<ShapeExecutable>,
    before: adapter::SemanticValueMap,
    after_lookup: adapter::SemanticValueMap,
    after: adapter::SemanticValueMap,
    lowered: LoweredCompositionPrelude,
    random: ValueVersion,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let input_catalogs = executable
            .arena()
            .transcript()
            .inputs
            .iter()
            .map(|(_, binding)| ArenaCatalogValueId(binding.logical.0))
            .collect::<BTreeSet<_>>();
        let mut values = adapter::SemanticValueMap::allocate_ordered(input_catalogs).unwrap();
        let bootstrap = super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &mut values,
        )
        .unwrap();
        let lookup = super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap),
            &mut values,
        )
        .unwrap();
        values
            .extend_ordered(required_upstream_catalogs(executable.arena()).unwrap())
            .unwrap();
        let after_lookup = values.clone();
        let composition = super::super::transcript_semantic_projection::lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::InteractionAndComposition,
            Some(&lookup),
            &mut values,
        )
        .unwrap();
        let random = composition
            .output(CairoTranscriptOutput::CompositionRandomCoefficient)
            .unwrap()
            .binding
            .value;
        let before = values.clone();
        let lowered = lower_stage(executable.arena(), &mut values).unwrap();
        Fixture {
            executable,
            before,
            after_lookup,
            after: values,
            lowered,
            random,
        }
    })
}

#[test]
fn generated_sn2_projects_exact_prelude_order_counts_and_transcript_value() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    assert_eq!(lowered.operations().len(), 2);
    assert_eq!(
        lowered
            .operations()
            .iter()
            .map(LoweredCompositionPreludeOperation::operation_ordinal)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    let materialize = &lowered.operations()[0];
    let powers = &lowered.operations()[1];
    assert_eq!(
        materialize.operation().abi,
        CompositionAbi::MaterializeExtParamsV1
    );
    assert_eq!(
        powers.operation().abi,
        CompositionAbi::GenerateDescendingPowersV1
    );
    let CompositionOperationKind::MaterializeExtParams { count, .. } = materialize.operation().kind
    else {
        panic!("first operation must materialize extension parameters");
    };
    let CompositionOperationKind::GenerateDescendingPowers {
        count: powers_count,
    } = powers.operation().kind
    else {
        panic!("second operation must generate descending powers");
    };
    assert_eq!(
        count as usize,
        fixture
            .executable
            .arena()
            .composition()
            .requirements
            .dynamic_ext_param_count
    );
    assert_eq!(count, 4_782);
    assert_eq!(powers_count, 1_053);
    assert_eq!(materialize.invocation().arguments.len(), 10);
    assert_eq!(powers.invocation().arguments.len(), 3);
    assert_eq!(
        materialize.bindings().len(),
        materialize.operation().children[0].effect.accesses.len()
    );
    assert_eq!(
        powers.bindings().len(),
        powers.operation().children[0].effect.accesses.len()
    );
    assert!(!materialize.launch().grid.contains(&0));
    assert!(!powers.launch().grid.contains(&0));

    let random = lowered
        .values()
        .get(&CompositionValueRole::RandomCoefficient)
        .unwrap();
    assert_eq!(random.version, fixture.random);
    let dynamic = lowered
        .values()
        .values()
        .filter(|value| {
            matches!(
                value.kind,
                LoweredCompositionPreludeValueKind::DynamicOutput
            )
        })
        .count();
    assert_eq!(dynamic, 4_782);
    assert!(validate_from(
        fixture.executable.arena(),
        &fixture.before,
        &fixture.after,
        lowered
    )
    .is_ok());
    assert_ne!(lowered.digest(), [0; 32]);
}

#[test]
fn missing_release_occupied_output_and_receipt_tamper_fail_transactionally() {
    let fixture = fixture();
    let mut missing_release = fixture.after_lookup.clone();
    let unchanged = missing_release.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut missing_release).is_err());
    assert_eq!(missing_release, unchanged);

    let powers = fixture
        .lowered
        .values()
        .get(&CompositionValueRole::RandomCoefficientPowers)
        .unwrap();
    let LoweredCompositionPreludeValueKind::CatalogOutput(powers_catalog) = powers.kind else {
        panic!("random powers must be the canonical arena output");
    };
    let mut occupied = fixture.before.clone();
    occupied.extend_ordered([powers_catalog]).unwrap();
    let unchanged = occupied.clone();
    assert!(lower_stage(fixture.executable.arena(), &mut occupied).is_err());
    assert_eq!(occupied, unchanged);

    let mut forged = fixture.lowered.clone();
    forged.digest[0] ^= 1;
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            &fixture.before,
            &fixture.after,
            &forged
        ),
        Err(InvocationShapeError::InvalidCompositionPreludeBinding)
    );
}
