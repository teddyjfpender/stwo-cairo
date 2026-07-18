//! Exact Composition accumulator-to-retained-evaluation ownership.

use stwo_backend_cuda::{
    ArenaSlice, CompositionSplitColumns, CompositionSplitError, CompositionSplitLaunchMode,
    CompositionSplitPointerSlots, CompositionSplitProgram, DeviceArena,
    PreparedCompositionSplitGraph, COMPOSITION_RETAINED_COLUMNS, COMPOSITION_SOURCE_COORDINATES,
};

use super::{CompositionWorkspaceRequirements, PreparedCompositionError, SECURE_COORDINATES};

// Both production shapes use the authority-sealed, spill-free 256-thread
// fused boundary. Hardware qualification is fail-closed before the complete
// proof checkpoint.
const fn production_split_mode(evaluation_log_size: u32) -> CompositionSplitLaunchMode {
    match evaluation_log_size {
        24 | 25 => CompositionSplitLaunchMode::FusedFirstForward,
        _ => CompositionSplitLaunchMode::TerminalFallback,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionOutputMode {
    CoefficientSplit,
    DirectRetainedEvaluations,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionOutputPlan {
    CoefficientSplit,
    DirectRetainedEvaluations(CompositionSplitProgram),
}

impl CompositionOutputPlan {
    pub const fn mode(self) -> CompositionOutputMode {
        match self {
            Self::CoefficientSplit => CompositionOutputMode::CoefficientSplit,
            Self::DirectRetainedEvaluations(_) => CompositionOutputMode::DirectRetainedEvaluations,
        }
    }

    pub const fn direct_program(self) -> Option<CompositionSplitProgram> {
        match self {
            Self::CoefficientSplit => None,
            Self::DirectRetainedEvaluations(program) => Some(program),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompositionDirectSplitBinding {
    pub program: CompositionSplitProgram,
    pub pointer_slots: CompositionSplitPointerSlots,
    pub retained_evaluations: [ArenaSlice; COMPOSITION_RETAINED_COLUMNS],
}

pub(super) fn prepare_direct_split<'a>(
    arena: &'a DeviceArena,
    requirements: &CompositionWorkspaceRequirements,
    accumulators: ArenaSlice,
    inverse_twiddles: ArenaSlice,
    forward_twiddles: ArenaSlice,
    binding: CompositionDirectSplitBinding,
) -> Result<PreparedCompositionSplitGraph<'a>, PreparedCompositionError> {
    let max = requirements
        .accumulators
        .last()
        .ok_or(PreparedCompositionError::EmptyPlan)?;
    if max.log_size != requirements.max_evaluation_log_size
        || binding.program.schedule().evaluation_log_size != max.log_size
    {
        return Err(PreparedCompositionError::DirectSplit(
            CompositionSplitError::InvalidSchedule,
        ));
    }
    let rows = 1usize
        .checked_shl(max.log_size)
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    let coordinate_words = rows
        .checked_mul(SECURE_COORDINATES)
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    if max.len_words != coordinate_words {
        return Err(PreparedCompositionError::DirectSplit(
            CompositionSplitError::InvalidSchedule,
        ));
    }
    let max_accumulator = accumulators.checked_subslice(max.offset_words, max.len_words)?;
    let source_evaluations = std::array::from_fn(|coordinate| {
        max_accumulator
            .checked_subslice(coordinate * rows, rows)
            .expect("validated exact four-coordinate accumulator extent")
    });
    let graph = PreparedCompositionSplitGraph::prepare(
        arena,
        binding.program,
        production_split_mode(binding.program.schedule().evaluation_log_size),
        binding.pointer_slots,
        CompositionSplitColumns {
            source_evaluations,
            retained_evaluations: binding.retained_evaluations,
        },
        inverse_twiddles,
        forward_twiddles,
    )?;
    let retained_identity = graph
        .retained_evaluations()
        .iter()
        .zip(&binding.retained_evaluations)
        .all(|(&actual, &expected)| {
            actual.id() == expected.id()
                && actual.as_u32_ptr() == expected.as_u32_ptr()
                && actual.len_words() == expected.len_words()
        });
    if !retained_identity || COMPOSITION_SOURCE_COORDINATES != SECURE_COORDINATES {
        return Err(PreparedCompositionError::DirectSplit(
            CompositionSplitError::InvalidSchedule,
        ));
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_split_mode_respects_the_sm90_launch_budget() {
        assert_eq!(
            production_split_mode(24),
            CompositionSplitLaunchMode::FusedFirstForward
        );
        assert_eq!(
            production_split_mode(25),
            CompositionSplitLaunchMode::FusedFirstForward
        );
    }
}
