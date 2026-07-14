//! Allocation-deduplicated host ledger for the resident proof arena.

use std::collections::{BTreeMap, BTreeSet};

use crate::arena_plan::{
    BufferPurpose, LogicalBuffer, OpenedColumnSource, ProofArenaPlan, ProofEpoch,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

/// Coarse accounting class inferred from a buffer purpose.
///
/// This is not an allocation-owner identity: shared commitment purposes may
/// contain either fixed or dynamic data. Physical byte totals remain exact;
/// this split is diagnostic until ownership is carried by each logical value.
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
    pub physical_bytes_by_purpose_class: BTreeMap<MemoryPurposeClass, usize>,
    pub slot_slack_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalMemoryLedger {
    pub arena_allocation_bytes: usize,
    pub physical_slots: usize,
    pub epochs: Vec<EpochMemoryLedger>,
}

impl PhysicalMemoryLedger {
    pub fn from_plan(plan: &ProofArenaPlan) -> Result<Self, &'static str> {
        let mut capacities = BTreeMap::new();
        for binding in plan.bindings() {
            capacities
                .entry(binding.physical)
                .and_modify(|words: &mut usize| *words = (*words).max(binding.len_words))
                .or_insert(binding.len_words);
        }
        let epochs = ProofEpoch::ALL
            .into_iter()
            .map(|epoch| {
                let mut logical_words = BTreeMap::<MemoryPurposeClass, usize>::new();
                let mut physical_words = BTreeMap::<MemoryPurposeClass, usize>::new();
                let mut seen = BTreeSet::new();
                for buffer in plan
                    .logical_buffers()
                    .iter()
                    .filter(|buffer| buffer.lifetime.contains(epoch))
                {
                    let class = MemoryPurposeClass::of(buffer.purpose);
                    checked_add(&mut logical_words, class, buffer.len_words)?;
                    let binding = plan.binding(buffer.id).ok_or("ledger binding is missing")?;
                    if !seen.insert(binding.physical) {
                        return Err("ledger found two live owners for one physical slot");
                    }
                    checked_add(
                        &mut physical_words,
                        class,
                        *capacities
                            .get(&binding.physical)
                            .ok_or("ledger slot capacity is missing")?,
                    )?;
                }
                let logical_total = logical_words.values().try_fold(0usize, |sum, &words| {
                    sum.checked_add(words).ok_or("ledger logical sum overflow")
                })?;
                let physical_total = physical_words.values().try_fold(0usize, |sum, &words| {
                    sum.checked_add(words).ok_or("ledger physical sum overflow")
                })?;
                if physical_total != plan.high_water_words(epoch) || logical_total > physical_total
                {
                    return Err("ledger does not reconcile with arena high-water");
                }
                Ok(EpochMemoryLedger {
                    epoch,
                    logical_bytes_by_purpose_class: into_bytes(logical_words)?,
                    physical_bytes_by_purpose_class: into_bytes(physical_words)?,
                    slot_slack_bytes: physical_total
                        .checked_sub(logical_total)
                        .and_then(|words| words.checked_mul(WORD_BYTES))
                        .ok_or("ledger slack overflow")?,
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        Ok(Self {
            arena_allocation_bytes: plan
                .total_words()
                .checked_mul(WORD_BYTES)
                .ok_or("arena byte size overflow")?,
            physical_slots: capacities.len(),
            epochs,
        })
    }

    pub fn json(
        plan: &ProofArenaPlan,
        operational_ceiling_bytes: usize,
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
                    "physical_by_purpose_class_bytes": purpose_class_json(
                        &epoch.physical_bytes_by_purpose_class
                    ),
                    "slot_slack_bytes": epoch.slot_slack_bytes,
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
        Ok(serde_json::json!({
            "allocation_model": "one stable shape arena; subslots deduplicated by ArenaSlotId",
            "purpose_class_caveat": "purpose classes are diagnostic; shared commitment purposes do not encode exact allocation ownership",
            "arena_allocation_id": "shape_arena",
            "arena_allocation_bytes": ledger.arena_allocation_bytes,
            "physical_slots": ledger.physical_slots,
            "epochs": epochs,
            "late_coefficient_bytes_by_final_consumer": coefficient_bytes,
            "operational_ceiling_bytes": operational_ceiling_bytes,
            "arena_only_fit": ledger.arena_allocation_bytes <= operational_ceiling_bytes,
            "admission_complete": false,
            "admission_pass": false,
            "missing_allocation_rows": [
                "primary context and driver baseline",
                "module code and globals",
                "graph and event metadata",
                "allocator pool slack",
                "profiling overhead",
                "operational safety reserve"
            ],
        }))
    }
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
}
