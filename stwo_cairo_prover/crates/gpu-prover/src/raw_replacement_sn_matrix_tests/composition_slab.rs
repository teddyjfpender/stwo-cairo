//! Exact sealed-SN physical receipt for direct Composition output ownership.

use std::path::Path;

use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

use super::{admit_fixture_identities, SealedSnFixture, SEALED_SN_FIXTURES};
use crate::arena_plan::{CompositionSlabArenaCounterfactual, ProofEpoch};
use crate::phases;
use crate::resident_session::plan_raw_resident_preflight;

fn secure_pcs() -> PcsConfig {
    PcsConfig {
        pow_bits: 26,
        fri_config: FriConfig::new(0, 1, 70, 3),
        lifting_log_size: None,
    }
}

fn bytes(words: usize) -> usize {
    words.checked_mul(size_of::<u32>()).expect("byte overflow")
}

fn signed_delta(fallback: usize, direct: usize) -> i64 {
    i64::try_from(fallback).expect("fallback words fit i64")
        - i64::try_from(direct).expect("direct words fit i64")
}

fn epoch_json(receipt: &CompositionSlabArenaCounterfactual) -> Vec<serde_json::Value> {
    ProofEpoch::ALL
        .into_iter()
        .map(|epoch| {
            let direct = receipt
                .direct
                .epoch_live_words
                .iter()
                .find_map(|&(candidate, words)| (candidate == epoch).then_some(words))
                .expect("direct epoch receipt");
            let fallback = receipt
                .forced_coefficient_fallback
                .epoch_live_words
                .iter()
                .find_map(|&(candidate, words)| (candidate == epoch).then_some(words))
                .expect("fallback epoch receipt");
            let expected = if matches!(
                epoch,
                ProofEpoch::Composition | ProofEpoch::CompositionCommit
            ) {
                receipt.fallback_coefficient_words
            } else {
                0
            };
            assert_eq!(
                fallback,
                direct.checked_add(expected).expect("epoch words overflow"),
                "only the exact Composition coefficient lifetime may change"
            );
            serde_json::json!({
                "epoch": format!("{epoch:?}"),
                "direct_live_words": direct,
                "fallback_live_words": fallback,
                "fallback_minus_direct_words": signed_delta(fallback, direct),
            })
        })
        .collect()
}

fn profile_receipt(directory: &Path, fixture: &SealedSnFixture) -> serde_json::Value {
    let path = directory.join(fixture.file);
    let encoded = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read sealed {}: {error}", path.display()));
    let input: ProverInput = bincode::deserialize(&encoded)
        .unwrap_or_else(|error| panic!("decode sealed {}: {error}", path.display()));
    drop(encoded);
    let ingest = phases::ingest::run_replacement(input, PreProcessedTraceVariant::Canonical, None)
        .unwrap_or_else(|error| panic!("{} replacement ingest: {error}", fixture.profile));
    let report = plan_raw_resident_preflight(
        &ingest.input,
        &ingest.proof_plan,
        &ingest.preprocessed_trace,
        secure_pcs(),
        false,
    )
    .unwrap_or_else(|error| panic!("{} replacement preflight: {error:?}", fixture.profile));
    let evaluation_log = report
        .arena
        .composition()
        .requirements
        .max_evaluation_log_size;
    assert!(matches!(evaluation_log, 24 | 25));
    let receipt = report
        .arena
        .composition_slab_arena_counterfactual()
        .unwrap_or_else(|error| panic!("{} Composition slab receipt: {error:?}", fixture.profile));
    let expected_coefficient_words = 8usize
        .checked_mul(
            1usize
                .checked_shl(evaluation_log - 1)
                .expect("Composition coefficient log fits usize"),
        )
        .expect("Composition coefficient words fit usize");
    assert_eq!(receipt.fallback_coefficient_buffers, 8);
    assert_eq!(
        receipt.fallback_coefficient_words,
        expected_coefficient_words
    );
    let total_delta = signed_delta(
        receipt.forced_coefficient_fallback.total_words,
        receipt.direct.total_words,
    );
    let granted_words = usize::try_from(total_delta).unwrap_or(0);

    serde_json::json!({
        "profile": fixture.profile,
        "fixture_bytes": fixture.bytes,
        "fixture_sha256": fixture.sha256,
        "fixture_blake3": fixture.blake3,
        "composition_evaluation_log": evaluation_log,
        "non_output_protocol_geometry_identical": true,
        "commitment_geometry_identical": true,
        "alias_inputs_reconstructed_and_revalidated": true,
        "timing_credit": false,
        "fallback_coefficient_buffers": receipt.fallback_coefficient_buffers,
        "fallback_coefficient_words": receipt.fallback_coefficient_words,
        "fallback_coefficient_bytes": bytes(receipt.fallback_coefficient_words),
        "identical_logical_prefix_buffers": receipt.identical_logical_prefix_buffers,
        "transition_alias_pairs": receipt.transition_alias_pairs,
        "released_commitment_alias_pairs": receipt.released_commitment_alias_pairs,
        "direct": {
            "total_words": receipt.direct.total_words,
            "total_bytes": bytes(receipt.direct.total_words),
            "whole_slot_total_words": receipt.direct.whole_slot_total_words,
            "raw_peak_words": receipt.direct.raw_peak_words,
            "excess_over_raw_peak_words": receipt.direct.excess_over_raw_peak_words,
            "range_view_count": receipt.direct.range_view_count,
            "range_view_words": receipt.direct.range_view_words,
        },
        "forced_coefficient_fallback": {
            "total_words": receipt.forced_coefficient_fallback.total_words,
            "total_bytes": bytes(receipt.forced_coefficient_fallback.total_words),
            "whole_slot_total_words": receipt.forced_coefficient_fallback.whole_slot_total_words,
            "raw_peak_words": receipt.forced_coefficient_fallback.raw_peak_words,
            "excess_over_raw_peak_words": receipt.forced_coefficient_fallback.excess_over_raw_peak_words,
            "range_view_count": receipt.forced_coefficient_fallback.range_view_count,
            "range_view_words": receipt.forced_coefficient_fallback.range_view_words,
        },
        "forced_fallback_minus_direct": {
            "total_words": total_delta,
            "whole_slot_total_words": signed_delta(
                receipt.forced_coefficient_fallback.whole_slot_total_words,
                receipt.direct.whole_slot_total_words,
            ),
            "raw_peak_words": signed_delta(
                receipt.forced_coefficient_fallback.raw_peak_words,
                receipt.direct.raw_peak_words,
            ),
        },
        "physical_arena_reduction_granted_bytes": bytes(granted_words),
        "physical_arena_reduction_grant_status": if total_delta < 0 {
            "withheld-because-forced-fallback-colored-smaller"
        } else {
            "exact-forced-fallback-minus-direct"
        },
        "epoch_live_words": epoch_json(&receipt),
    })
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_1..4.adapted.bin"]
fn direct_composition_slab_physical_receipt_on_sn1_through_sn4() {
    let directory = std::env::var("STWO_SN_ADAPTED_DIR")
        .expect("set STWO_SN_ADAPTED_DIR to the sealed adapted-input directory");
    let directory = Path::new(&directory);
    admit_fixture_identities(directory);
    let receipts = SEALED_SN_FIXTURES
        .iter()
        .map(|fixture| profile_receipt(directory, fixture))
        .collect::<Vec<_>>();
    eprintln!(
        "STWO_COMPOSITION_SLAB_PHYSICAL_RECEIPT_JSON={}",
        serde_json::to_string(&serde_json::json!({
            "schema": "stwo.composition-slab-physical-counterfactual.v1",
            "profiles": receipts,
        }))
        .unwrap()
    );
}
