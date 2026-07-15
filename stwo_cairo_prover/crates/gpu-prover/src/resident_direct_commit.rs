//! Fail-closed Base/Interaction evaluation ownership for direct compact commits.

use stwo_backend_cuda::{ArenaSlice, DirectRetainedB2nColumn};

use crate::arena_plan::{
    ArenaBinding, BufferLifetime, BufferPurpose, CommitmentColumnSource, CommitmentTreeId,
    PlannedCommitment, ProofEpoch,
};
use crate::graphs::GraphWorkspace;
use crate::resident_sources::ResidentSourceStageError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TraceCommitInputMode {
    Interpolate,
    DirectEvaluations,
}

pub(crate) fn trace_commit_input_mode(
    planned_direct: bool,
    prepared_direct: bool,
) -> Result<TraceCommitInputMode, ()> {
    match (planned_direct, prepared_direct) {
        (true, true) => Ok(TraceCommitInputMode::DirectEvaluations),
        (false, false) => Ok(TraceCommitInputMode::Interpolate),
        _ => Err(()),
    }
}

pub(crate) struct DirectCommitmentInputs {
    pub columns: Vec<DirectRetainedB2nColumn>,
    pub inverse_twiddles: ArenaSlice,
    pub forward_twiddles: ArenaSlice,
}

#[derive(Clone, Copy)]
struct BoundTracePair {
    evaluations: ArenaSlice,
    coefficients: ArenaSlice,
    evaluation_binding: ArenaBinding,
    coefficient_binding: ArenaBinding,
    evaluation_lifetime: BufferLifetime,
    coefficient_lifetime: BufferLifetime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectSourceBindingViolation {
    LifetimeHandoff,
    SourceOutputAlias,
}

fn validate_direct_source_binding(
    lifetime_handoff: bool,
    source_output_disjoint: bool,
) -> Result<(), DirectSourceBindingViolation> {
    if !lifetime_handoff {
        return Err(DirectSourceBindingViolation::LifetimeHandoff);
    }
    if !source_output_disjoint {
        return Err(DirectSourceBindingViolation::SourceOutputAlias);
    }
    Ok(())
}

/// Bind the direct Base/Interaction inputs in the commitment program's exact
/// canonical order. The only admitted zero-copy ownership handoff is the
/// evaluation Value into its dormant, reader-free coefficient alias. Retained
/// outputs remain disjoint because the direct kernel has no proved in-place
/// source/output ordering contract.
pub(crate) fn direct_commitment_inputs(
    workspace: &GraphWorkspace,
    planned: &PlannedCommitment,
) -> Result<DirectCommitmentInputs, ResidentSourceStageError> {
    let commit_epoch = match planned.id {
        CommitmentTreeId::Base => ProofEpoch::BaseCommit,
        CommitmentTreeId::Interaction => ProofEpoch::InteractionCommit,
        _ => {
            return Err(ResidentSourceStageError::DirectCommitmentShapeMismatch(
                planned.id,
            ))
        }
    };
    if planned.direct_retained_b2n_program.is_none()
        || !planned.interpolation_batches.is_empty()
        || planned.grouped_column_sources.len() != planned.grouped_column_log_sizes.len()
        || planned.grouped_column_sources.len() != planned.evaluation_output_groups.len()
    {
        return Err(ResidentSourceStageError::DirectCommitmentShapeMismatch(
            planned.id,
        ));
    }

    let mut columns = Vec::new();
    for ((sources, logs), outputs) in planned
        .grouped_column_sources
        .iter()
        .zip(&planned.grouped_column_log_sizes)
        .zip(&planned.evaluation_output_groups)
    {
        let outputs =
            outputs
                .as_ref()
                .ok_or(ResidentSourceStageError::DirectCommitmentShapeMismatch(
                    planned.id,
                ))?;
        if sources.len() != logs.len() || sources.len() != outputs.len() {
            return Err(ResidentSourceStageError::DirectCommitmentShapeMismatch(
                planned.id,
            ));
        }
        for ((&source, &log_size), &output) in sources.iter().zip(logs).zip(outputs) {
            let pair = bind_trace_pair(workspace, planned.id, source)?;
            let ownership = workspace
                .plan()
                .late_coefficient_ownership()
                .get(source.into());
            let source_words = checked_words(log_size)?;
            let retained_words = checked_words(
                log_size
                    .checked_add(planned.config.log_blowup_factor)
                    .ok_or(ResidentSourceStageError::SizeOverflow)?,
            )?;
            if pair.evaluations.len_words() != source_words {
                return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
                    source,
                    expected_words: source_words,
                    actual_words: pair.evaluations.len_words(),
                });
            }
            let retained_output = workspace.bind(output.logical)?.0;
            if retained_output.len_words() != retained_words {
                return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
                    source,
                    expected_words: retained_words,
                    actual_words: retained_output.len_words(),
                });
            }
            let lifetime_handoff = ownership.is_some_and(|ownership| {
                !ownership.composition_reads_coefficients
                    && !ownership.oods_reads_coefficients
                    && !ownership.quotient_reads_coefficients
                    && !ownership.decommit_reads_coefficients
                    && ownership.final_consumer == commit_epoch
            }) && direct_source_lifetime_is_proved(pair, commit_epoch)
                && direct_source_has_single_commit_owner(workspace, pair, commit_epoch);
            let source_output_disjoint = !ranges_overlap(pair.evaluations, retained_output)?;
            match validate_direct_source_binding(lifetime_handoff, source_output_disjoint) {
                Ok(()) => {}
                Err(DirectSourceBindingViolation::LifetimeHandoff) => {
                    return Err(ResidentSourceStageError::DirectCommitmentSourceLifetime {
                        tree: planned.id,
                        source,
                        evaluations: pair.evaluation_lifetime,
                        coefficients: pair.coefficient_lifetime,
                    })
                }
                Err(DirectSourceBindingViolation::SourceOutputAlias) => {
                    return Err(
                        ResidentSourceStageError::DirectCommitmentSourceOutputAlias {
                            tree: planned.id,
                            source,
                        },
                    )
                }
            }
            columns.push(DirectRetainedB2nColumn {
                source_evaluations: pair.evaluations,
                retained_output,
            });
        }
    }

    let (_, forward_binding) = workspace
        .plan()
        .find(None, None, BufferPurpose::ForwardTwiddles, 0)
        .ok_or(ResidentSourceStageError::MissingGlobalArenaSource(
            BufferPurpose::ForwardTwiddles,
        ))?;
    if forward_binding != planned.twiddles {
        return Err(ResidentSourceStageError::DirectTwiddlePurposeMismatch {
            tree: planned.id,
            purpose: BufferPurpose::ForwardTwiddles,
        });
    }
    let forward_twiddles = workspace.bind(forward_binding.logical)?.0;
    let (inverse, _) = workspace
        .plan()
        .find(None, None, BufferPurpose::InverseTwiddles, 0)
        .ok_or(ResidentSourceStageError::MissingGlobalArenaSource(
            BufferPurpose::InverseTwiddles,
        ))?;
    let inverse_twiddles = workspace.bind(inverse.id)?.0;
    if ranges_overlap(inverse_twiddles, forward_twiddles)? {
        return Err(ResidentSourceStageError::DirectTwiddleAlias(planned.id));
    }
    Ok(DirectCommitmentInputs {
        columns,
        inverse_twiddles,
        forward_twiddles,
    })
}

fn bind_trace_pair(
    workspace: &GraphWorkspace,
    tree: CommitmentTreeId,
    source: CommitmentColumnSource,
) -> Result<BoundTracePair, ResidentSourceStageError> {
    let CommitmentColumnSource::Trace {
        component,
        part,
        purpose,
        ordinal,
    } = source
    else {
        return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source });
    };
    let (evaluation_purpose, coefficient_purpose) = match tree {
        CommitmentTreeId::Base => (BufferPurpose::BaseTrace, BufferPurpose::BaseCoefficients),
        CommitmentTreeId::Interaction => (
            BufferPurpose::InteractionTrace,
            BufferPurpose::InteractionCoefficients,
        ),
        _ => return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source }),
    };
    if purpose != coefficient_purpose {
        return Err(ResidentSourceStageError::InvalidInterpolationSource { tree, source });
    }
    let (evaluation, evaluation_binding) = workspace
        .plan()
        .find(Some(component), Some(part), evaluation_purpose, ordinal)
        .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
    let (coefficient, coefficient_binding) = workspace
        .plan()
        .find(Some(component), Some(part), coefficient_purpose, ordinal)
        .ok_or(ResidentSourceStageError::MissingArenaSource(source))?;
    let evaluation_lifetime = evaluation.lifetime;
    let coefficient_lifetime = coefficient.lifetime;
    let (evaluations, evaluation_words) = workspace.bind(evaluation.id)?;
    let (coefficients, coefficient_words) = workspace.bind(coefficient.id)?;
    if evaluation_words != coefficient_words {
        return Err(ResidentSourceStageError::ArenaSourceSizeMismatch {
            source,
            expected_words: evaluation_words,
            actual_words: coefficient_words,
        });
    }
    Ok(BoundTracePair {
        evaluations,
        coefficients,
        evaluation_binding,
        coefficient_binding,
        evaluation_lifetime,
        coefficient_lifetime,
    })
}

fn direct_source_lifetime_is_proved(pair: BoundTracePair, commit_epoch: ProofEpoch) -> bool {
    if pair.evaluation_lifetime.first >= commit_epoch
        || pair.coefficient_lifetime != BufferLifetime::at(commit_epoch)
    {
        return false;
    }
    if pair.evaluation_lifetime.contains(commit_epoch) {
        return true;
    }
    pair.evaluation_lifetime.last as u8 + 1 == commit_epoch as u8
        && pair.coefficient_lifetime.first == commit_epoch
        && pair.evaluation_binding.physical == pair.coefficient_binding.physical
        && pair.evaluation_binding.len_words == pair.coefficient_binding.len_words
        && pair.evaluations.id() == pair.coefficients.id()
        && pair.evaluations.as_u32_ptr() == pair.coefficients.as_u32_ptr()
        && pair.evaluations.len_words() == pair.coefficients.len_words()
}

fn direct_source_has_single_commit_owner(
    workspace: &GraphWorkspace,
    pair: BoundTracePair,
    commit_epoch: ProofEpoch,
) -> bool {
    let expected_owner = if pair.evaluation_lifetime.contains(commit_epoch) {
        pair.evaluation_binding.logical
    } else {
        pair.coefficient_binding.logical
    };
    let mut owners = workspace
        .plan()
        .logical_buffers()
        .iter()
        .filter(|buffer| buffer.lifetime.contains(commit_epoch))
        .filter(|buffer| {
            workspace
                .plan()
                .binding(buffer.id)
                .is_some_and(|binding| binding.physical == pair.evaluation_binding.physical)
        });
    owners
        .next()
        .is_some_and(|owner| owner.id == expected_owner)
        && owners.next().is_none()
}

fn ranges_overlap(left: ArenaSlice, right: ArenaSlice) -> Result<bool, ResidentSourceStageError> {
    let range = |slice: ArenaSlice| {
        let start = slice.as_u32_ptr() as usize;
        let bytes = slice
            .len_words()
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(ResidentSourceStageError::SizeOverflow)?;
        Ok::<_, ResidentSourceStageError>((
            start,
            start
                .checked_add(bytes)
                .ok_or(ResidentSourceStageError::SizeOverflow)?,
        ))
    };
    let left = range(left)?;
    let right = range(right)?;
    Ok(left.0 < right.1 && right.0 < left.1)
}

fn checked_words(log_size: u32) -> Result<usize, ResidentSourceStageError> {
    1usize
        .checked_shl(log_size)
        .ok_or(ResidentSourceStageError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifetime_and_output_alias_adversaries_are_independent() {
        assert_eq!(
            validate_direct_source_binding(false, true),
            Err(DirectSourceBindingViolation::LifetimeHandoff)
        );
        assert_eq!(
            validate_direct_source_binding(true, false),
            Err(DirectSourceBindingViolation::SourceOutputAlias)
        );
        assert_eq!(
            validate_direct_source_binding(false, false),
            Err(DirectSourceBindingViolation::LifetimeHandoff)
        );
        assert_eq!(validate_direct_source_binding(true, true), Ok(()));
    }

    #[test]
    fn direct_word_geometry_is_checked() {
        assert_eq!(checked_words(3).unwrap(), 8);
        assert!(checked_words(usize::BITS).is_err());
    }

    #[test]
    fn capture_rejects_planned_direct_with_legacy_prepared_graph() {
        assert_eq!(trace_commit_input_mode(true, false), Err(()));
    }

    #[test]
    fn eager_rejects_legacy_plan_with_direct_prepared_graph() {
        assert_eq!(trace_commit_input_mode(false, true), Err(()));
    }

    #[test]
    fn direct_and_interpolation_modes_are_explicit() {
        assert_eq!(
            trace_commit_input_mode(true, true),
            Ok(TraceCommitInputMode::DirectEvaluations)
        );
        assert_eq!(
            trace_commit_input_mode(false, false),
            Ok(TraceCommitInputMode::Interpolate)
        );
    }
}
