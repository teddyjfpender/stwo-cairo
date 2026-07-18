use std::sync::{Arc, OnceLock};

use super::*;
use crate::compiled_proof::{ExternalInputId, ValueOrigin};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    initial: adapter::SemanticValueMap,
    after_bootstrap: adapter::SemanticValueMap,
    after_lookup: adapter::SemanticValueMap,
    bootstrap: LoweredTranscriptSegment,
    lookup: LoweredTranscriptSegment,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let inventory =
            TranscriptInventory::compile(executable.arena(), executable.transcript()).unwrap();
        let catalogs = executable
            .transcript()
            .segments()
            .iter()
            .take(2)
            .flat_map(|segment| segment.operation_range.clone())
            .filter_map(|index| {
                operation_input(executable.transcript().schedule().operations()[index])
            })
            .map(|id| exact_input(&inventory, id).unwrap().catalog)
            .collect::<Vec<_>>();
        let initial = adapter::SemanticValueMap::allocate_ordered(catalogs).unwrap();
        let mut after_bootstrap = initial.clone();
        let bootstrap = lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::BootstrapThroughBase,
            None,
            &mut after_bootstrap,
        )
        .unwrap();
        let mut after_lookup = after_bootstrap.clone();
        let lookup = lower_segment(
            executable.arena(),
            executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap),
            &mut after_lookup,
        )
        .unwrap();
        Fixture {
            executable,
            initial,
            after_bootstrap,
            after_lookup,
            bootstrap,
            lookup,
        }
    })
}

#[test]
fn generated_sn2_releases_lookup_at_the_exact_second_segment() {
    let fixture = fixture();
    assert_eq!(fixture.bootstrap.index(), 0);
    assert_eq!(
        fixture.bootstrap.segment(),
        CairoTranscriptSegment::BootstrapThroughBase
    );
    assert_eq!(fixture.bootstrap.operation_range(), &(0..11));
    assert_eq!(fixture.bootstrap.inputs().len(), 11);
    assert!(fixture.bootstrap.outputs().is_empty());
    assert_eq!(fixture.after_bootstrap, fixture.initial);
    assert_eq!(
        fixture.bootstrap.compiled().entry_state,
        TranscriptStateVersion(0)
    );
    assert_eq!(
        fixture.bootstrap.compiled().exit_state,
        TranscriptStateVersion(1)
    );

    assert_eq!(fixture.lookup.index(), 1);
    assert_eq!(
        fixture.lookup.segment(),
        CairoTranscriptSegment::InteractionPowAndLookup
    );
    assert_eq!(fixture.lookup.operation_range(), &(11..13));
    assert_eq!(fixture.lookup.inputs().len(), 1);
    assert_eq!(fixture.lookup.outputs().len(), 1);
    assert_eq!(
        fixture.lookup.compiled().entry_state,
        TranscriptStateVersion(1)
    );
    assert_eq!(
        fixture.lookup.compiled().exit_state,
        TranscriptStateVersion(2)
    );
    assert_eq!(
        fixture.lookup.compiled().consumed.len(),
        fixture.lookup.inputs().len()
    );
    assert_eq!(
        fixture.lookup.compiled().produced.len(),
        fixture.lookup.outputs().len()
    );

    let output = fixture
        .lookup
        .output(CairoTranscriptOutput::CommonLookupElements)
        .unwrap();
    assert_eq!(
        output.binding.id,
        CairoTranscriptOutput::CommonLookupElements.id().unwrap()
    );
    assert_eq!(output.binding.elements, ElementRange::new(0, 8).unwrap());
    assert_eq!(output.arena.len_words, 8);
    assert_eq!(
        output.value.origin,
        ValueOrigin::TranscriptOutput(output.binding.id)
    );
    assert_eq!(output.value.version, output.binding.value);
    assert_eq!(output.value.region, Region::Dynamic);
    assert_eq!(output.value.layout.element, ElementType::U32);
    assert_eq!(output.value.layout.element_count().unwrap(), 8);
    assert_eq!(
        fixture.after_lookup.version(output.catalog).unwrap(),
        output.binding.value
    );
    assert_eq!(
        fixture.after_lookup.entries().count(),
        fixture.initial.entries().count() + 1
    );
    assert_ne!(fixture.bootstrap.digest(), [0; 32]);
    assert_ne!(fixture.lookup.digest(), [0; 32]);
}

#[test]
fn segment_order_input_readiness_and_output_ownership_fail_closed() {
    let fixture = fixture();
    let mut wrong_order = fixture.initial.clone();
    assert_eq!(
        lower_segment(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            None,
            &mut wrong_order,
        ),
        Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
    );
    assert_eq!(wrong_order, fixture.initial);

    let nonce = fixture.lookup.inputs()[0].catalog;
    let catalogs = fixture
        .initial
        .entries()
        .filter_map(|(catalog, _)| (catalog != nonce).then_some(catalog));
    let mut missing_nonce = adapter::SemanticValueMap::allocate_ordered(catalogs).unwrap();
    let bootstrap = lower_segment(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        CairoTranscriptSegment::BootstrapThroughBase,
        None,
        &mut missing_nonce,
    )
    .unwrap();
    let unchanged = missing_nonce.clone();
    assert_eq!(
        lower_segment(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&bootstrap),
            &mut missing_nonce,
        ),
        Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
    );
    assert_eq!(missing_nonce, unchanged);

    let output = fixture.lookup.outputs()[0].catalog;
    let mut occupied = fixture.after_bootstrap.clone();
    occupied.extend_ordered([output]).unwrap();
    let unchanged = occupied.clone();
    assert_eq!(
        lower_segment(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&fixture.bootstrap),
            &mut occupied,
        ),
        Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
    );
    assert_eq!(occupied, unchanged);
}

#[test]
fn exact_replay_accepts_only_the_transactional_allocator_result() {
    let fixture = fixture();
    assert!(validate_from(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        CairoTranscriptSegment::BootstrapThroughBase,
        None,
        &fixture.initial,
        &fixture.after_bootstrap,
        &fixture.bootstrap,
    )
    .is_ok());
    assert!(validate_from(
        fixture.executable.arena(),
        fixture.executable.transcript(),
        CairoTranscriptSegment::InteractionPowAndLookup,
        Some(&fixture.bootstrap),
        &fixture.after_bootstrap,
        &fixture.after_lookup,
        &fixture.lookup,
    )
    .is_ok());

    let mut wrong_after = fixture.after_lookup.clone();
    wrong_after
        .extend_ordered([ArenaCatalogValueId(u32::MAX)])
        .unwrap();
    assert_eq!(
        validate_from(
            fixture.executable.arena(),
            fixture.executable.transcript(),
            CairoTranscriptSegment::InteractionPowAndLookup,
            Some(&fixture.bootstrap),
            &fixture.after_bootstrap,
            &wrong_after,
            &fixture.lookup,
        ),
        Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
    );
}

#[test]
fn receipt_rejects_range_binding_origin_chain_and_digest_tamper() {
    let fixture = fixture();
    let assert_invalid = |forged: &LoweredTranscriptSegment| {
        assert_eq!(
            validate_from(
                fixture.executable.arena(),
                fixture.executable.transcript(),
                CairoTranscriptSegment::InteractionPowAndLookup,
                Some(&fixture.bootstrap),
                &fixture.after_bootstrap,
                &fixture.after_lookup,
                forged,
            ),
            Err(InvocationShapeError::InvalidTranscriptSemanticProjection)
        );
    };

    let mut range = fixture.lookup.clone();
    range.plan.operation_range.start -= 1;
    assert_invalid(&range);

    let mut input = fixture.lookup.clone();
    input.inputs[0].binding.value.0 ^= 1;
    assert_invalid(&input);

    let mut origin = fixture.lookup.clone();
    origin.outputs[0].value.origin = ValueOrigin::ExternalInput(ExternalInputId(0));
    assert_invalid(&origin);

    let mut chain = fixture.lookup.clone();
    chain.prior_digest = None;
    assert_invalid(&chain);

    let mut digest = fixture.lookup.clone();
    digest.digest[0] ^= 1;
    assert_invalid(&digest);
}
