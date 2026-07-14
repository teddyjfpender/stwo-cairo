//! Exact late consumers for dynamic commitment coefficients.
//!
//! This is deliberately narrower than the eventual unified commitment compiler:
//! the current OODS implementation still reads coefficients. The plan records
//! only the late stages whose source mode is already sealed by
//! [`ProtocolGeometry`], so the arena may release a coefficient after its real
//! final reader without assuming a future evaluation-source OODS path.

use std::collections::HashMap;

use crate::arena_plan::{
    BufferPurpose, CommitmentTreeId, OpenedColumnSource, ProofEpoch, ProtocolGeometry,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LateCoefficientOwnership {
    pub source: OpenedColumnSource,
    pub oods_reads_coefficients: bool,
    pub quotient_reads_coefficients: bool,
    pub decommit_reads_coefficients: bool,
    pub final_consumer: ProofEpoch,
}

#[derive(Clone, Debug)]
pub struct LateCoefficientOwnershipPlan {
    entries: Vec<LateCoefficientOwnership>,
    by_source: HashMap<OpenedColumnSource, usize>,
}

impl LateCoefficientOwnershipPlan {
    pub fn compile(protocol: &ProtocolGeometry) -> Result<Self, &'static str> {
        let mut entries = Vec::new();
        let mut by_source = HashMap::new();
        for commitment in &protocol.commitments {
            if commitment.id == CommitmentTreeId::Preprocessed {
                continue;
            }
            let group_count = commitment.grouped_column_sources.len();
            if commitment.grouped_column_log_sizes.len() != group_count
                || commitment.retained_evaluation_groups.len() != group_count
                || commitment.numerator_evaluation_groups.len() != group_count
            {
                return Err("late coefficient ownership group shape drifted");
            }
            for group in 0..group_count {
                let sources = &commitment.grouped_column_sources[group];
                if sources.len() != commitment.grouped_column_log_sizes[group].len() {
                    return Err("late coefficient ownership source/log shape drifted");
                }
                for &source in sources {
                    let source = OpenedColumnSource::from(source);
                    if !matches!(
                        source,
                        OpenedColumnSource::Trace {
                            purpose: BufferPurpose::BaseCoefficients
                                | BufferPurpose::InteractionCoefficients,
                            ..
                        } | OpenedColumnSource::Composition { .. }
                    ) {
                        return Err("late coefficient ownership contains a non-dynamic source");
                    }
                    let mut matching_oods = protocol
                        .oods
                        .columns
                        .iter()
                        .filter(|column| column.source == source);
                    let oods = matching_oods
                        .next()
                        .ok_or("dynamic commitment source has no OODS column")?;
                    if matching_oods.next().is_some() {
                        return Err("dynamic commitment source has duplicate OODS columns");
                    }
                    let quotient_reads_coefficients = !oods.shape_points.is_empty()
                        && !commitment.numerator_evaluation_groups[group];
                    let decommit_reads_coefficients = !commitment.retained_evaluation_groups[group];
                    let final_consumer =
                        final_consumer(quotient_reads_coefficients, decommit_reads_coefficients);
                    let index = entries.len();
                    if by_source.insert(source, index).is_some() {
                        return Err("dynamic commitment source appears in multiple groups");
                    }
                    entries.push(LateCoefficientOwnership {
                        source,
                        oods_reads_coefficients: true,
                        quotient_reads_coefficients,
                        decommit_reads_coefficients,
                        final_consumer,
                    });
                }
            }
        }
        Ok(Self { entries, by_source })
    }

    pub fn entries(&self) -> &[LateCoefficientOwnership] {
        &self.entries
    }

    pub fn get(&self, source: OpenedColumnSource) -> Option<&LateCoefficientOwnership> {
        self.by_source
            .get(&source)
            .map(|&index| &self.entries[index])
    }

    pub fn final_consumer(&self, source: OpenedColumnSource) -> Result<ProofEpoch, &'static str> {
        self.get(source)
            .map(|ownership| ownership.final_consumer)
            .ok_or("dynamic coefficient source lacks late ownership")
    }
}

const fn final_consumer(
    quotient_reads_coefficients: bool,
    decommit_reads_coefficients: bool,
) -> ProofEpoch {
    if decommit_reads_coefficients {
        ProofEpoch::Decommit
    } else if quotient_reads_coefficients {
        ProofEpoch::Quotient
    } else {
        // OODS remains coefficient-sourced in the current backend.
        ProofEpoch::Oods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_consumer_is_the_latest_real_coefficient_reader() {
        assert_eq!(final_consumer(false, false), ProofEpoch::Oods);
        assert_eq!(final_consumer(true, false), ProofEpoch::Quotient);
        assert_eq!(final_consumer(false, true), ProofEpoch::Decommit);
        assert_eq!(final_consumer(true, true), ProofEpoch::Decommit);
    }
}
