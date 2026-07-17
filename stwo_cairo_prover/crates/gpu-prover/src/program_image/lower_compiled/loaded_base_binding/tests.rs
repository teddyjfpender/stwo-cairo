use std::collections::BTreeMap;

use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::statement_state::{
    SourceAttemptCounter, StatementSourceKind, StatementSourceState, StatementSourceStateError,
};
use super::{exact_key, insert_unique, producer_key};
use crate::program_image::lower_compiled::InvocationShapeError;

#[test]
fn closed_inventory_rejects_duplicate_keys() {
    let mut indexed = BTreeMap::new();
    let key = exact_key("pedersen", TracePartId::Main);
    insert_unique(&mut indexed, key, 1).unwrap();
    assert_eq!(
        insert_unique(&mut indexed, key, 2),
        Err(InvocationShapeError::InvalidProductionBaseAuthority)
    );
    assert_eq!(indexed.len(), 1);
    assert_eq!(indexed[&key], 1);
}

#[test]
fn producer_key_rejects_an_untyped_trace_part() {
    assert_eq!(
        producer_key("pedersen", None),
        Err(InvocationShapeError::InvalidProductionBaseAuthority)
    );
    assert_eq!(
        producer_key("pedersen", Some(TracePartId::Main)),
        Ok(exact_key("pedersen", TracePartId::Main))
    );
}

fn admit_all(state: &mut StatementSourceState, generation: u64) {
    for kind in [
        StatementSourceKind::ExecutionTables,
        StatementSourceKind::PublicMemorySeed,
        StatementSourceKind::EcOpSegmentStart,
        StatementSourceKind::WitnessInputs,
        StatementSourceKind::StaticTranscriptInputs,
    ] {
        state.admit(kind, generation).unwrap();
    }
}

#[test]
fn publication_accepts_only_a_complete_source_set() {
    let mut state = StatementSourceState::new([true; 5]);
    state
        .admit(StatementSourceKind::ExecutionTables, 1)
        .unwrap();
    assert_eq!(
        state.publish(),
        Err(StatementSourceStateError),
        "one source cannot make the statement launch-ready"
    );
    assert_eq!(
        state.validate_ready([Some(1), None, None, None, None]),
        Err(StatementSourceStateError),
        "failed publication invalidates the partial transaction"
    );
    admit_all(&mut state, 2);
    state.publish().unwrap();
    let generations = [Some(2); 5];
    state.validate_ready(generations).unwrap();
    state.validate_ready(generations).unwrap();
}

#[test]
fn begin_and_partial_failure_leave_the_transaction_closed() {
    let mut state = StatementSourceState::new([true; 5]);
    admit_all(&mut state, 1);
    state.publish().unwrap();
    state.begin();
    assert_eq!(
        state.validate_ready([Some(1); 5]),
        Err(StatementSourceStateError)
    );
    state
        .admit(StatementSourceKind::ExecutionTables, 2)
        .unwrap();
    assert_eq!(state.publish(), Err(StatementSourceStateError));
    assert_eq!(
        state.admit(StatementSourceKind::ExecutionTables, 2),
        Err(StatementSourceStateError),
        "duplicate candidate invalidates the partial transaction"
    );
    assert_eq!(state.publish(), Err(StatementSourceStateError));
}

#[test]
fn consume_is_single_use_and_same_generation_cannot_be_replayed() {
    let mut state = StatementSourceState::new([true; 5]);
    admit_all(&mut state, 1);
    state.publish().unwrap();
    let first = [Some(1); 5];
    state.validate_ready(first).unwrap();
    state.consume(first).unwrap();
    assert_eq!(state.consume(first), Err(StatementSourceStateError));

    state.begin();
    assert_eq!(
        state.admit(StatementSourceKind::ExecutionTables, 1),
        Err(StatementSourceStateError)
    );
    assert_eq!(state.publish(), Err(StatementSourceStateError));
}

#[test]
fn second_statement_requires_strictly_new_generations() {
    let mut state = StatementSourceState::new([true; 5]);
    admit_all(&mut state, 1);
    state.publish().unwrap();
    state.consume([Some(1); 5]).unwrap();

    state.begin();
    admit_all(&mut state, 2);
    state.publish().unwrap();
    let second = [Some(2); 5];
    state.validate_ready(second).unwrap();
    state.consume(second).unwrap();
}

#[test]
fn absent_sources_are_exact_and_ec_cannot_be_silently_omitted() {
    let mut no_sources = StatementSourceState::new([false; 5]);
    no_sources.publish().unwrap();
    no_sources.validate_ready([None; 5]).unwrap();

    let mut ec_only = StatementSourceState::new([false, false, true, false, false]);
    assert_eq!(ec_only.publish(), Err(StatementSourceStateError));
    assert_eq!(
        ec_only.admit(StatementSourceKind::ExecutionTables, 1),
        Err(StatementSourceStateError)
    );
    ec_only
        .admit(StatementSourceKind::EcOpSegmentStart, 1)
        .unwrap();
    ec_only.publish().unwrap();
    ec_only
        .validate_ready([None, None, Some(1), None, None])
        .unwrap();
}

#[test]
fn witness_and_static_transcript_are_both_required() {
    let mut missing_witness = StatementSourceState::new([false, false, false, true, true]);
    assert_eq!(
        missing_witness.reserve_final(StatementSourceKind::StaticTranscriptInputs),
        Err(StatementSourceStateError),
        "transcript DMA cannot start before exact witness admission"
    );

    let mut missing_transcript = StatementSourceState::new([false, false, false, true, true]);
    missing_transcript
        .admit(StatementSourceKind::WitnessInputs, 1)
        .unwrap();
    assert_eq!(missing_transcript.publish(), Err(StatementSourceStateError));
}

#[test]
fn failed_upload_attempt_burns_its_generation_before_retry() {
    let mut attempts = SourceAttemptCounter::default();
    let mut state = StatementSourceState::new([false, false, false, true, false]);

    let failed = attempts.begin().unwrap();
    state.reserve(StatementSourceKind::WitnessInputs).unwrap();
    assert!(std::panic::catch_unwind(|| panic!("simulated upload unwind")).is_err());
    state
        .validate_before_final(StatementSourceKind::WitnessInputs, [None; 5])
        .unwrap();

    let retry = attempts.begin().unwrap();
    assert_eq!((failed, retry), (1, 2));
    state.reserve(StatementSourceKind::WitnessInputs).unwrap();
    state
        .admit(StatementSourceKind::WitnessInputs, retry)
        .unwrap();
    state.publish().unwrap();
    state
        .validate_ready([None, None, None, Some(2), None])
        .unwrap();
}

#[test]
fn transcript_unwind_keeps_the_final_source_absent_and_retryable() {
    let mut attempts = SourceAttemptCounter::default();
    let mut state = StatementSourceState::new([false, false, false, true, true]);
    state.admit(StatementSourceKind::WitnessInputs, 1).unwrap();

    let failed = attempts.begin().unwrap();
    state
        .reserve_final(StatementSourceKind::StaticTranscriptInputs)
        .unwrap();
    assert!(std::panic::catch_unwind(|| panic!("simulated transcript unwind")).is_err());
    let pending = [None, None, None, Some(1), None];
    state
        .validate_before_final(StatementSourceKind::StaticTranscriptInputs, pending)
        .unwrap();

    let retry = attempts.begin().unwrap();
    assert_eq!((failed, retry), (1, 2));
    state
        .reserve_final(StatementSourceKind::StaticTranscriptInputs)
        .unwrap();
    state
        .admit(StatementSourceKind::StaticTranscriptInputs, retry)
        .unwrap();
    state.publish().unwrap();
    state
        .validate_ready([None, None, None, Some(1), Some(2)])
        .unwrap();
}

#[test]
fn exact_source_set_accepts_optional_absence_and_rejects_extras() {
    let mut exact = StatementSourceState::new([true, false, true, true, true]);
    for kind in [
        StatementSourceKind::ExecutionTables,
        StatementSourceKind::EcOpSegmentStart,
        StatementSourceKind::WitnessInputs,
        StatementSourceKind::StaticTranscriptInputs,
    ] {
        exact.admit(kind, 1).unwrap();
    }
    exact.publish().unwrap();
    exact
        .validate_ready([Some(1), None, Some(1), Some(1), Some(1)])
        .unwrap();

    let mut extra = StatementSourceState::new([true, false, true, true, true]);
    assert_eq!(
        extra.admit(StatementSourceKind::PublicMemorySeed, 1),
        Err(StatementSourceStateError)
    );
    assert_eq!(extra.publish(), Err(StatementSourceStateError));
}
