//! Exact consumers for commitment coefficients.
//!
//! Every downstream source choice is sealed by [`ProtocolGeometry`]. A
//! RetainEvaluation group therefore stops extending coefficient lifetime once
//! composition, OODS, numerator, and decommit have each selected the retained
//! evaluation representation.

use std::collections::HashMap;

use stwo_backend_cuda::{OodsSourceKind, QuotientNumeratorSourceKind};

use crate::arena_plan::{BufferPurpose, OpenedColumnSource, ProofEpoch, ProtocolGeometry};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LateCoefficientOwnership {
    pub source: OpenedColumnSource,
    pub composition_reads_coefficients: bool,
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
        let oods_source_kinds = protocol
            .oods_source_kinds()
            .map_err(|_| "late coefficient ownership OODS source policy is invalid")?;
        let numerator_source_kinds = protocol
            .quotient_numerator_source_kinds()
            .map_err(|_| "late coefficient ownership numerator source policy is invalid")?;
        let mut entries = Vec::new();
        let mut by_source = HashMap::new();
        for commitment in &protocol.commitments {
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
                        OpenedColumnSource::Preprocessed { .. }
                            | OpenedColumnSource::Trace {
                                purpose: BufferPurpose::BaseCoefficients
                                    | BufferPurpose::InteractionCoefficients,
                                ..
                            }
                            | OpenedColumnSource::Composition { .. }
                    ) {
                        return Err("late coefficient ownership contains an unsupported source");
                    }
                    let mut matching_oods = protocol
                        .oods
                        .columns
                        .iter()
                        .zip(&oods_source_kinds)
                        .zip(&numerator_source_kinds)
                        .filter(|((column, _), _)| column.source == source);
                    let ((oods, &oods_source_kind), &numerator_source_kind) = matching_oods
                        .next()
                        .ok_or("commitment source has no OODS column")?;
                    if matching_oods.next().is_some() {
                        return Err("commitment source has duplicate OODS columns");
                    }
                    let quotient_reads_coefficients = !oods.shape_points.is_empty()
                        && numerator_source_kind == QuotientNumeratorSourceKind::Coefficients;
                    let decommit_reads_coefficients = !commitment.retained_evaluation_groups[group];
                    let composition_reads_coefficients = match source {
                        OpenedColumnSource::Preprocessed { .. }
                        | OpenedColumnSource::Trace {
                            purpose:
                                BufferPurpose::BaseCoefficients | BufferPurpose::InteractionCoefficients,
                            ..
                        } => match &protocol.direct_composition_retention {
                            None => true,
                            Some(plan) => {
                                let mut reads_coefficients = false;
                                for binding in &plan.bindings {
                                    let column = plan
                                        .columns
                                        .get(binding.column)
                                        .ok_or("direct composition ownership column is missing")?;
                                    if column.source == source && !binding.direct {
                                        reads_coefficients = true;
                                    }
                                }
                                reads_coefficients
                            }
                        },
                        OpenedColumnSource::Composition { .. } => false,
                        OpenedColumnSource::Trace { .. } => {
                            return Err("late coefficient ownership contains an unsupported source")
                        }
                    };
                    let oods_reads_coefficients = !oods.shape_points.is_empty()
                        && oods_source_kind == OodsSourceKind::Coefficients;
                    let final_consumer = final_consumer(
                        commitment.created,
                        composition_reads_coefficients,
                        oods_reads_coefficients,
                        quotient_reads_coefficients,
                        decommit_reads_coefficients,
                    );
                    let index = entries.len();
                    if by_source.insert(source, index).is_some() {
                        return Err("commitment source appears in multiple groups");
                    }
                    entries.push(LateCoefficientOwnership {
                        source,
                        composition_reads_coefficients,
                        oods_reads_coefficients,
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
            .ok_or("coefficient source lacks late ownership")
    }
}

const fn final_consumer(
    commitment: ProofEpoch,
    composition_reads_coefficients: bool,
    oods_reads_coefficients: bool,
    quotient_reads_coefficients: bool,
    decommit_reads_coefficients: bool,
) -> ProofEpoch {
    if decommit_reads_coefficients {
        ProofEpoch::Decommit
    } else if quotient_reads_coefficients {
        ProofEpoch::Quotient
    } else if oods_reads_coefficients {
        ProofEpoch::Oods
    } else if composition_reads_coefficients {
        ProofEpoch::Composition
    } else {
        commitment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_consumer_is_the_latest_real_coefficient_reader() {
        let committed = ProofEpoch::BaseCommit;
        assert_eq!(
            final_consumer(committed, false, false, false, false),
            committed
        );
        assert_eq!(
            final_consumer(committed, true, false, false, false),
            ProofEpoch::Composition
        );
        assert_eq!(
            final_consumer(committed, true, true, false, false),
            ProofEpoch::Oods
        );
        assert_eq!(
            final_consumer(committed, true, true, true, false),
            ProofEpoch::Quotient
        );
        assert_eq!(
            final_consumer(committed, true, true, true, true),
            ProofEpoch::Decommit
        );
    }
}
