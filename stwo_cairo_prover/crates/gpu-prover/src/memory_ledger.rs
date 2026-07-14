//! Physical-memory ledger for the one resident proof-arena allocation.

use std::collections::{BTreeMap, BTreeSet};

use crate::arena_plan::{
    BufferPurpose, LogicalBuffer, OpenedColumnSource, ProofArenaPlan, ProofEpoch,
};
use crate::fixed_table_materializer::PEDERSEN_POINTS_18_EVALUATION_BYTES;
use crate::resident_sources::MAX_PREPROCESSED_DETACHED_STAGING_BYTES;

mod physical_rows;
pub use physical_rows::{
    AllocatorPoolCheckpoint, PhysicalAllocationId, PhysicalAllocationOwnerId, PhysicalMemoryInputs,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

/// Coarse accounting class inferred from a buffer purpose.
///
/// This is not an allocation-owner identity: shared commitment purposes may
/// contain either fixed or dynamic data. Range-live byte totals remain exact;
/// the single allocation is not partitioned by this diagnostic classification.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemoryPurposeClass {
    FixedData,
    Input,
    Dynamic,
    Output,
}

impl MemoryPurposeClass {
    pub const ALL: [Self; 4] = [Self::FixedData, Self::Input, Self::Dynamic, Self::Output];

    pub fn of(purpose: BufferPurpose) -> Self {
        match purpose {
            BufferPurpose::PreprocessedCoefficients
            | BufferPurpose::PreprocessedEvaluations
            | BufferPurpose::PreprocessedInterpolationPointers
            | BufferPurpose::PreprocessedInverseTwiddles
            | BufferPurpose::ForwardTwiddles
            | BufferPurpose::InverseTwiddles
            | BufferPurpose::QuotientInverseTwiddles => Self::FixedData,
            BufferPurpose::WitnessInput
            | BufferPurpose::WitnessInputSeedScalars
            | BufferPurpose::ExecutionTableRawAddressToId
            | BufferPurpose::ExecutionTableRawF252Words
            | BufferPurpose::ExecutionTableRawSmallWords
            | BufferPurpose::EcOpSegmentStart
            | BufferPurpose::PublicMemoryMultiplicitySeed => Self::Input,
            BufferPurpose::ProofBytes => Self::Output,
            _ => Self::Dynamic,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpochMemoryLedger {
    pub epoch: ProofEpoch,
    pub logical_bytes_by_purpose_class: BTreeMap<MemoryPurposeClass, usize>,
    pub range_live_bytes_by_purpose_class: BTreeMap<MemoryPurposeClass, usize>,
    pub logical_live_bytes: usize,
    pub range_live_bytes: usize,
    pub arena_idle_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalMemoryLedger {
    pub arena_allocation_bytes: usize,
    pub whole_slot_comparator_bytes: usize,
    pub raw_peak_bytes: usize,
    pub excess_over_raw_peak_bytes: usize,
    pub range_view_count: usize,
    pub aggregate_range_view_bytes: usize,
    pub epochs: Vec<EpochMemoryLedger>,
}

#[derive(Clone, Copy, Debug)]
struct EpochRange {
    id: u32,
    class: MemoryPurposeClass,
    logical_words: usize,
    range_words: usize,
    offset_words: usize,
}

pub const ARENA_IDLE_DEFINITION: &str = concat!(
    "arena allocation bytes minus the exact disjoint live-address union at that epoch; ",
    "includes temporarily dead/reused bytes and alignment gaps; diagnostic only, ",
    "not reclaimable or additive memory"
);

impl PhysicalMemoryLedger {
    pub fn from_plan(plan: &ProofArenaPlan) -> Result<Self, &'static str> {
        if plan.bindings().len() != plan.logical_buffers().len() {
            return Err("ledger binding cardinality does not match logical values");
        }
        let arena_words = plan.total_words();
        let mut range_views = BTreeMap::new();
        for binding in plan.bindings() {
            let range = plan
                .layout()
                .slot(binding.physical)
                .ok_or("ledger range view is missing")?;
            range_views.insert(binding.physical.0, range.len_words);
        }
        let aggregate_range_view_words = range_views.values().try_fold(0usize, |sum, &words| {
            sum.checked_add(words)
                .ok_or("ledger aggregate range-view size overflow")
        })?;
        if range_views.len() != plan.range_view_count()
            || aggregate_range_view_words != plan.range_view_words()
        {
            return Err("ledger range-view metrics do not reconcile");
        }
        let epochs = ProofEpoch::ALL
            .into_iter()
            .map(|epoch| {
                let ranges = plan
                    .logical_buffers()
                    .iter()
                    .filter(|buffer| buffer.lifetime.contains(epoch))
                    .map(|buffer| {
                        let binding = plan.binding(buffer.id).ok_or("ledger binding is missing")?;
                        if binding.len_words != buffer.len_words {
                            return Err("ledger binding length does not match logical value");
                        }
                        let range = plan
                            .layout()
                            .slot(binding.physical)
                            .ok_or("ledger range view is missing")?;
                        Ok(EpochRange {
                            id: binding.physical.0,
                            class: MemoryPurposeClass::of(buffer.purpose),
                            logical_words: buffer.len_words,
                            range_words: range.len_words,
                            offset_words: range.offset_words,
                        })
                    })
                    .collect::<Result<Vec<_>, &'static str>>()?;
                reconcile_epoch(epoch, arena_words, plan.high_water_words(epoch), ranges)
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        let raw_peak_words = ProofEpoch::ALL
            .into_iter()
            .map(|epoch| plan.high_water_words(epoch))
            .max()
            .unwrap_or(0);
        if raw_peak_words != plan.raw_peak_words()
            || arena_words.checked_sub(raw_peak_words) != Some(plan.excess_over_raw_peak_words())
        {
            return Err("ledger range-allocation metrics do not reconcile");
        }
        Ok(Self {
            arena_allocation_bytes: words_to_bytes(arena_words)?,
            whole_slot_comparator_bytes: words_to_bytes(plan.whole_slot_total_words())?,
            raw_peak_bytes: words_to_bytes(raw_peak_words)?,
            excess_over_raw_peak_bytes: words_to_bytes(plan.excess_over_raw_peak_words())?,
            range_view_count: range_views.len(),
            aggregate_range_view_bytes: words_to_bytes(aggregate_range_view_words)?,
            epochs,
        })
    }

    pub fn json(
        plan: &ProofArenaPlan,
        operational_ceiling_bytes: usize,
    ) -> Result<serde_json::Value, &'static str> {
        Self::json_with_inputs(
            plan,
            operational_ceiling_bytes,
            &PhysicalMemoryInputs::default(),
        )
    }

    pub fn json_with_inputs(
        plan: &ProofArenaPlan,
        operational_ceiling_bytes: usize,
        physical_inputs: &PhysicalMemoryInputs,
    ) -> Result<serde_json::Value, &'static str> {
        let ledger = Self::from_plan(plan)?;
        let epochs = ledger
            .epochs
            .iter()
            .map(|epoch| {
                serde_json::json!({
                    "epoch": format!("{:?}", epoch.epoch),
                    "logical_by_purpose_class_bytes": purpose_class_json(
                        &epoch.logical_bytes_by_purpose_class
                    ),
                    "range_live_by_purpose_class_bytes": purpose_class_json(
                        &epoch.range_live_bytes_by_purpose_class
                    ),
                    "logical_live_bytes": epoch.logical_live_bytes,
                    "range_live_bytes": epoch.range_live_bytes,
                    "range_live_reconciles_logical": epoch.range_live_bytes == epoch.logical_live_bytes,
                    "arena_idle_bytes": epoch.arena_idle_bytes,
                })
            })
            .collect::<Vec<_>>();
        let mut coefficient_bytes = BTreeMap::<ProofEpoch, usize>::new();
        for ownership in plan.late_coefficient_ownership().entries() {
            let buffer = coefficient_buffer(plan.logical_buffers(), ownership.source)
                .ok_or("late ownership source has no logical coefficient buffer")?;
            checked_add(
                &mut coefficient_bytes,
                ownership.final_consumer,
                buffer.len_words,
            )?;
        }
        let coefficient_bytes = coefficient_bytes
            .into_iter()
            .map(|(epoch, words)| {
                Ok((
                    format!("{epoch:?}"),
                    words
                        .checked_mul(WORD_BYTES)
                        .ok_or("coefficient byte size overflow")?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, &'static str>>()?;
        let arena_allocation_fit = ledger.arena_allocation_bytes <= operational_ceiling_bytes;
        let process_owned_pedersen_table_bytes = plan
            .requires_registered_pedersen_table()
            .then_some(PEDERSEN_POINTS_18_EVALUATION_BYTES)
            .unwrap_or(0);
        let omitted_pedersen_evaluation_bytes = plan.process_owned_pedersen_evaluation_bytes();
        if omitted_pedersen_evaluation_bytes != 0
            && omitted_pedersen_evaluation_bytes != PEDERSEN_POINTS_18_EVALUATION_BYTES
        {
            return Err("process-owned Pedersen evaluation omission is not the complete table");
        }
        let known_allocation_subtotal_bytes = ledger
            .arena_allocation_bytes
            .checked_add(process_owned_pedersen_table_bytes)
            .ok_or("known physical allocation subtotal overflow")?;
        let known_allocation_subtotal_fit =
            known_allocation_subtotal_bytes <= operational_ceiling_bytes;
        let twiddle_detached_staging_payload_bytes = [
            BufferPurpose::ForwardTwiddles,
            BufferPurpose::PreprocessedInverseTwiddles,
            BufferPurpose::InverseTwiddles,
            BufferPurpose::QuotientInverseTwiddles,
        ]
        .into_iter()
        .try_fold(0usize, |bytes, purpose| {
            let (buffer, _) = plan
                .find(None, None, purpose, 0)
                .ok_or("fixed twiddle staging source is missing")?;
            bytes
                .checked_add(words_to_bytes(buffer.len_words)?)
                .ok_or("fixed twiddle staging byte size overflow")
        })?;
        let known_cold_source_payload_peak_bytes =
            twiddle_detached_staging_payload_bytes.max(MAX_PREPROCESSED_DETACHED_STAGING_BYTES);
        let known_cold_payload_subtotal_bytes = known_allocation_subtotal_bytes
            .checked_add(known_cold_source_payload_peak_bytes)
            .ok_or("known cold payload subtotal overflow")?;
        let known_cold_payload_subtotal_fit =
            known_cold_payload_subtotal_bytes <= operational_ceiling_bytes;
        let physical = physical_inputs
            .admission(known_cold_payload_subtotal_bytes, operational_ceiling_bytes)?;
        let record = serde_json::json!({
            "allocation_model": "one stable shape-arena allocation with lifetime-reused stable range views",
            "purpose_class_caveat": "purpose classes diagnose live logical/range views; the one arena allocation cannot be partitioned by purpose because addresses are reused over time, and shared commitment purposes do not encode exact allocation ownership",
            "arena_idle_definition": ARENA_IDLE_DEFINITION,
            "arena_allocation_id": "shape_arena",
            "arena_allocation_owner_id": "resident_shape_workspace",
            "arena_allocation_count": 1,
            "arena_allocation_bytes": ledger.arena_allocation_bytes,
            "process_owned_pedersen_table_allocation_id": "pedersen_points_18_evaluations",
            "process_owned_pedersen_table_owner_id": "pedersen_points_18_registry",
            "process_owned_pedersen_table_bytes": process_owned_pedersen_table_bytes,
            "omitted_duplicate_pedersen_evaluation_bytes": omitted_pedersen_evaluation_bytes,
            "known_allocation_subtotal_bytes": known_allocation_subtotal_bytes,
            "known_allocation_subtotal_fit": known_allocation_subtotal_fit,
            "known_allocation_subtotal_headroom_bytes": operational_ceiling_bytes
                .saturating_sub(known_allocation_subtotal_bytes),
            "twiddle_detached_staging_payload_bytes": twiddle_detached_staging_payload_bytes,
            "preprocessed_detached_staging_payload_bound_bytes": MAX_PREPROCESSED_DETACHED_STAGING_BYTES,
            "known_cold_source_payload_peak_bytes": known_cold_source_payload_peak_bytes,
            "known_cold_source_payload_peak_owner_id": "resident_cold_setup",
            "known_cold_payload_subtotal_bytes": known_cold_payload_subtotal_bytes,
            "known_cold_payload_subtotal_fit": known_cold_payload_subtotal_fit,
            "known_cold_payload_subtotal_headroom_bytes": operational_ceiling_bytes
                .saturating_sub(known_cold_payload_subtotal_bytes),
            "whole_slot_comparator_bytes": ledger.whole_slot_comparator_bytes,
            "raw_peak_bytes": ledger.raw_peak_bytes,
            "excess_over_raw_peak_bytes": ledger.excess_over_raw_peak_bytes,
            "range_view_count": ledger.range_view_count,
            "aggregate_range_view_bytes": ledger.aggregate_range_view_bytes,
            "epochs": epochs,
            "late_coefficient_bytes_by_final_consumer": coefficient_bytes,
            "operational_ceiling_bytes": operational_ceiling_bytes,
            "arena_allocation_fit": arena_allocation_fit,
            "arena_only_fit": arena_allocation_fit,
            "deprecated_compatibility_aliases": {
                "arena_only_fit": "arena_allocation_fit",
                "reason": "preserve the public preflight JSON contract during range-arena migration",
                "removal_condition": "remove after all external consumers read arena_allocation_fit",
            },
            "physical_allocation_rows": physical.rows,
            "measured_non_arena_subtotal_bytes": physical.measured_bytes,
            "physical_peak_bytes": physical.peak_bytes,
            "admission_complete": physical.complete,
            "admission_pass": physical.pass,
            "missing_allocation_ids": physical.missing_ids,
            "missing_allocation_rows": physical.missing_rows,
        });
        if !fit_alias_matches(&record) {
            return Err("deprecated arena fit alias drifted from allocation field");
        }
        Ok(record)
    }
}

fn fit_alias_matches(record: &serde_json::Value) -> bool {
    record.get("arena_only_fit").is_some()
        && record.get("arena_only_fit") == record.get("arena_allocation_fit")
}

fn reconcile_epoch(
    epoch: ProofEpoch,
    arena_words: usize,
    reported_live_words: usize,
    ranges: Vec<EpochRange>,
) -> Result<EpochMemoryLedger, &'static str> {
    let mut logical_words = BTreeMap::<MemoryPurposeClass, usize>::new();
    let mut range_words = BTreeMap::<MemoryPurposeClass, usize>::new();
    let mut identities = BTreeSet::new();
    let mut occupied = Vec::with_capacity(ranges.len());
    for range in ranges {
        if range.logical_words == 0 || range.range_words == 0 {
            return Err("ledger found an empty live range view");
        }
        if range.logical_words != range.range_words {
            return Err("ledger range-view length does not match logical value");
        }
        if !identities.insert(range.id) {
            return Err("ledger found two live owners for one range view");
        }
        let end = range
            .offset_words
            .checked_add(range.range_words)
            .ok_or("ledger range end overflow")?;
        if end > arena_words {
            return Err("ledger live range exceeds the arena allocation");
        }
        checked_add(&mut logical_words, range.class, range.logical_words)?;
        checked_add(&mut range_words, range.class, range.range_words)?;
        occupied.push((range.offset_words, end));
    }
    occupied.sort_unstable();
    if occupied.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("ledger found overlapping live range views");
    }

    let logical_total = sum_words(&logical_words, "ledger logical sum overflow")?;
    let range_total = sum_words(&range_words, "ledger range-live sum overflow")?;
    if logical_total != range_total || range_total != reported_live_words {
        return Err("ledger range-live bytes do not reconcile with logical live bytes");
    }
    Ok(EpochMemoryLedger {
        epoch,
        logical_bytes_by_purpose_class: into_bytes(logical_words)?,
        range_live_bytes_by_purpose_class: into_bytes(range_words)?,
        logical_live_bytes: words_to_bytes(logical_total)?,
        range_live_bytes: words_to_bytes(range_total)?,
        arena_idle_bytes: words_to_bytes(
            arena_words
                .checked_sub(range_total)
                .ok_or("ledger range-live bytes exceed arena allocation")?,
        )?,
    })
}

fn coefficient_buffer(
    logical: &[LogicalBuffer],
    source: OpenedColumnSource,
) -> Option<&LogicalBuffer> {
    logical.iter().find(|buffer| match source {
        OpenedColumnSource::Trace {
            component,
            part,
            purpose,
            ordinal,
        } => {
            buffer.component == Some(component)
                && buffer.part == Some(part)
                && buffer.purpose == purpose
                && buffer.ordinal == ordinal
        }
        OpenedColumnSource::Composition { ordinal } => {
            buffer.component.is_none()
                && buffer.part.is_none()
                && buffer.purpose == BufferPurpose::CompositionCoefficients
                && buffer.ordinal == ordinal
        }
        OpenedColumnSource::Preprocessed { .. } => false,
    })
}

fn checked_add<K: Ord + Copy>(
    totals: &mut BTreeMap<K, usize>,
    key: K,
    value: usize,
) -> Result<(), &'static str> {
    let total = totals.entry(key).or_default();
    *total = total.checked_add(value).ok_or("memory ledger overflow")?;
    Ok(())
}

fn sum_words<K: Ord>(
    totals: &BTreeMap<K, usize>,
    overflow: &'static str,
) -> Result<usize, &'static str> {
    totals
        .values()
        .try_fold(0usize, |sum, &words| sum.checked_add(words).ok_or(overflow))
}

fn words_to_bytes(words: usize) -> Result<usize, &'static str> {
    words.checked_mul(WORD_BYTES).ok_or("ledger byte overflow")
}

fn into_bytes<K: Ord>(words: BTreeMap<K, usize>) -> Result<BTreeMap<K, usize>, &'static str> {
    words
        .into_iter()
        .map(|(key, words)| {
            Ok((
                key,
                words
                    .checked_mul(WORD_BYTES)
                    .ok_or("ledger byte overflow")?,
            ))
        })
        .collect()
}

fn purpose_class_json(bytes: &BTreeMap<MemoryPurposeClass, usize>) -> BTreeMap<String, usize> {
    MemoryPurposeClass::ALL
        .into_iter()
        .map(|class| {
            (
                format!("{class:?}"),
                bytes.get(&class).copied().unwrap_or(0),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purpose_class_contract_separates_fixed_input_dynamic_and_output() {
        assert_eq!(
            MemoryPurposeClass::of(BufferPurpose::ForwardTwiddles),
            MemoryPurposeClass::FixedData
        );
        assert_eq!(
            MemoryPurposeClass::of(BufferPurpose::WitnessInput),
            MemoryPurposeClass::Input
        );
        assert_eq!(
            MemoryPurposeClass::of(BufferPurpose::CommitLdeTile),
            MemoryPurposeClass::Dynamic
        );
        assert_eq!(
            MemoryPurposeClass::of(BufferPurpose::ProofBytes),
            MemoryPurposeClass::Output
        );
    }

    fn range(id: u32, class: MemoryPurposeClass, words: usize, offset_words: usize) -> EpochRange {
        EpochRange {
            id,
            class,
            logical_words: words,
            range_words: words,
            offset_words,
        }
    }

    #[test]
    fn epoch_range_live_reconciles_exactly_and_idle_is_all_unused_arena_bytes() {
        let ledger = reconcile_epoch(
            ProofEpoch::Witness,
            128,
            96,
            vec![
                range(1, MemoryPurposeClass::Input, 32, 0),
                range(2, MemoryPurposeClass::Dynamic, 64, 64),
            ],
        )
        .unwrap();

        assert_eq!(ledger.logical_live_bytes, 96 * WORD_BYTES);
        assert_eq!(ledger.range_live_bytes, ledger.logical_live_bytes);
        assert_eq!(ledger.arena_idle_bytes, 32 * WORD_BYTES);
        assert_eq!(
            ledger.logical_bytes_by_purpose_class,
            ledger.range_live_bytes_by_purpose_class
        );
    }

    #[test]
    fn epoch_ledger_rejects_overlapping_live_addresses() {
        assert_eq!(
            reconcile_epoch(
                ProofEpoch::Witness,
                128,
                96,
                vec![
                    range(1, MemoryPurposeClass::Input, 64, 0),
                    range(2, MemoryPurposeClass::Dynamic, 32, 48),
                ],
            )
            .unwrap_err(),
            "ledger found overlapping live range views"
        );
    }

    #[test]
    fn epoch_ledger_rejects_duplicate_live_view_identity_even_at_distinct_addresses() {
        assert_eq!(
            reconcile_epoch(
                ProofEpoch::Witness,
                128,
                64,
                vec![
                    range(7, MemoryPurposeClass::Input, 32, 0),
                    range(7, MemoryPurposeClass::Dynamic, 32, 64),
                ],
            )
            .unwrap_err(),
            "ledger found two live owners for one range view"
        );
    }

    #[test]
    fn epoch_ledger_rejects_logical_range_length_or_high_water_drift() {
        let mut mismatched = range(1, MemoryPurposeClass::Dynamic, 32, 0);
        mismatched.range_words = 64;
        assert_eq!(
            reconcile_epoch(ProofEpoch::Witness, 128, 32, vec![mismatched]).unwrap_err(),
            "ledger range-view length does not match logical value"
        );
        assert_eq!(
            reconcile_epoch(
                ProofEpoch::Witness,
                128,
                31,
                vec![range(1, MemoryPurposeClass::Dynamic, 32, 0)],
            )
            .unwrap_err(),
            "ledger range-live bytes do not reconcile with logical live bytes"
        );
    }

    #[test]
    fn epoch_ledger_rejects_ranges_outside_the_single_allocation() {
        assert_eq!(
            reconcile_epoch(
                ProofEpoch::Witness,
                128,
                32,
                vec![range(1, MemoryPurposeClass::Dynamic, 32, 100)],
            )
            .unwrap_err(),
            "ledger live range exceeds the arena allocation"
        );
    }

    #[test]
    fn deprecated_arena_only_fit_alias_must_equal_allocation_fit() {
        let mut record = serde_json::json!({
            "arena_allocation_fit": true,
            "arena_only_fit": true,
        });
        assert!(fit_alias_matches(&record));
        record["arena_only_fit"] = serde_json::json!(false);
        assert!(!fit_alias_matches(&record));
    }
}
