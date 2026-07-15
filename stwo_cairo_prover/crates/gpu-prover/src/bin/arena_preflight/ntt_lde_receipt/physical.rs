//! Exact physical-arena formatting for the Composition output counterfactual.

use serde_json::{json, Value};
use stwo_cairo_gpu_prover::arena_plan::{
    CompositionSlabArenaFootprint, ProofArenaPlan, ProofEpoch,
};
use stwo_cairo_gpu_prover::prepared_composition::CompositionOutputMode;

pub(super) fn json(arena: &ProofArenaPlan) -> Result<(Value, usize), String> {
    if arena.composition().output_plan.mode() != CompositionOutputMode::DirectRetainedEvaluations {
        return Ok((
            json!({
                "schema": "stwo.composition-slab-physical-counterfactual.v1",
                "status": "not-applicable",
                "reason": "current Composition output already uses CoefficientSplit",
                "physical_arena_reduction_granted_bytes": 0,
                "timing_credit": false,
            }),
            0,
        ));
    }
    let receipt = arena
        .composition_slab_arena_counterfactual()
        .map_err(|error| format!("Composition slab physical counterfactual: {error:?}"))?;
    let total_delta = signed_delta(
        receipt.forced_coefficient_fallback.total_words,
        receipt.direct.total_words,
    )?;
    let granted_words = usize::try_from(total_delta).unwrap_or(0);
    let granted_bytes = bytes(granted_words)?;

    Ok((
        json!({
            "schema": "stwo.composition-slab-physical-counterfactual.v1",
            "comparison": "current Direct output versus forced CoefficientSplit output",
            "non_output_protocol_geometry_identical": true,
            "commitment_geometry_identical": true,
            "logical_prefix_identical": true,
            "alias_inputs_reconstructed_and_revalidated": true,
            "timing_credit": false,
            "fallback_coefficient_buffers": receipt.fallback_coefficient_buffers,
            "fallback_coefficient_words": receipt.fallback_coefficient_words,
            "fallback_coefficient_bytes": bytes(receipt.fallback_coefficient_words)?,
            "identical_logical_prefix_buffers": receipt.identical_logical_prefix_buffers,
            "transition_alias_pairs": receipt.transition_alias_pairs,
            "released_commitment_alias_pairs": receipt.released_commitment_alias_pairs,
            "direct": footprint_json(&receipt.direct)?,
            "forced_coefficient_fallback": footprint_json(
                &receipt.forced_coefficient_fallback,
            )?,
            "forced_fallback_minus_direct": {
                "total_words": total_delta,
                "total_bytes": signed_bytes(total_delta)?,
                "whole_slot_total_words": signed_delta(
                    receipt.forced_coefficient_fallback.whole_slot_total_words,
                    receipt.direct.whole_slot_total_words,
                )?,
                "raw_peak_words": signed_delta(
                    receipt.forced_coefficient_fallback.raw_peak_words,
                    receipt.direct.raw_peak_words,
                )?,
                "range_view_count": signed_delta(
                    receipt.forced_coefficient_fallback.range_view_count,
                    receipt.direct.range_view_count,
                )?,
                "range_view_words": signed_delta(
                    receipt.forced_coefficient_fallback.range_view_words,
                    receipt.direct.range_view_words,
                )?,
            },
            "physical_arena_reduction_granted_bytes": granted_bytes,
            "physical_arena_reduction_grant_status": if total_delta < 0 {
                "withheld-because-forced-fallback-colored-smaller"
            } else {
                "exact-forced-fallback-minus-direct"
            },
            "physical_reduction_masked_by_range_reuse":
                total_delta == 0 && receipt.fallback_coefficient_words != 0,
            "physical_rule": "total_words is the exact range-packed allocation; raw_peak is only a lower bound",
            "epoch_live_words": epoch_json(&receipt.direct, &receipt.forced_coefficient_fallback)?,
        }),
        granted_bytes,
    ))
}

fn footprint_json(footprint: &CompositionSlabArenaFootprint) -> Result<Value, String> {
    Ok(json!({
        "total_words": footprint.total_words,
        "total_bytes": bytes(footprint.total_words)?,
        "whole_slot_total_words": footprint.whole_slot_total_words,
        "whole_slot_total_bytes": bytes(footprint.whole_slot_total_words)?,
        "raw_peak_words": footprint.raw_peak_words,
        "raw_peak_bytes": bytes(footprint.raw_peak_words)?,
        "excess_over_raw_peak_words": footprint.excess_over_raw_peak_words,
        "excess_over_raw_peak_bytes": bytes(footprint.excess_over_raw_peak_words)?,
        "range_view_count": footprint.range_view_count,
        "range_view_words": footprint.range_view_words,
        "range_view_bytes": bytes(footprint.range_view_words)?,
        "maximum_epoch_live_words": footprint
            .epoch_live_words
            .iter()
            .map(|&(_, words)| words)
            .max()
            .unwrap_or(0),
    }))
}

fn epoch_json(
    direct: &CompositionSlabArenaFootprint,
    fallback: &CompositionSlabArenaFootprint,
) -> Result<Vec<Value>, String> {
    ProofEpoch::ALL
        .into_iter()
        .map(|epoch| {
            let direct_words = epoch_words(direct, epoch)?;
            let fallback_words = epoch_words(fallback, epoch)?;
            let delta = signed_delta(fallback_words, direct_words)?;
            Ok(json!({
                "epoch": epoch_name(epoch),
                "direct_live_words": direct_words,
                "direct_live_bytes": bytes(direct_words)?,
                "fallback_live_words": fallback_words,
                "fallback_live_bytes": bytes(fallback_words)?,
                "fallback_minus_direct_words": delta,
                "fallback_minus_direct_bytes": signed_bytes(delta)?,
            }))
        })
        .collect()
}

fn epoch_words(
    footprint: &CompositionSlabArenaFootprint,
    epoch: ProofEpoch,
) -> Result<usize, String> {
    footprint
        .epoch_live_words
        .iter()
        .find_map(|&(candidate, words)| (candidate == epoch).then_some(words))
        .ok_or_else(|| format!("missing {} epoch live-word receipt", epoch_name(epoch)))
}

fn bytes(words: usize) -> Result<usize, String> {
    words
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| "Composition slab byte count overflow".to_owned())
}

fn signed_delta(fallback: usize, direct: usize) -> Result<i64, String> {
    let fallback = i64::try_from(fallback)
        .map_err(|_| "fallback Composition slab count does not fit i64".to_owned())?;
    let direct = i64::try_from(direct)
        .map_err(|_| "direct Composition slab count does not fit i64".to_owned())?;
    fallback
        .checked_sub(direct)
        .ok_or_else(|| "Composition slab signed delta overflow".to_owned())
}

fn signed_bytes(words: i64) -> Result<i64, String> {
    words
        .checked_mul(i64::try_from(size_of::<u32>()).expect("u32 byte width fits i64"))
        .ok_or_else(|| "Composition slab signed byte delta overflow".to_owned())
}

const fn epoch_name(epoch: ProofEpoch) -> &'static str {
    match epoch {
        ProofEpoch::Ingest => "ingest",
        ProofEpoch::Witness => "witness",
        ProofEpoch::BaseCommit => "base_commit",
        ProofEpoch::Interaction => "interaction",
        ProofEpoch::InteractionCommit => "interaction_commit",
        ProofEpoch::Composition => "composition",
        ProofEpoch::CompositionCommit => "composition_commit",
        ProofEpoch::Oods => "oods",
        ProofEpoch::Quotient => "quotient",
        ProofEpoch::Fri => "fri",
        ProofEpoch::Decommit => "decommit",
        ProofEpoch::Assemble => "assemble",
    }
}
