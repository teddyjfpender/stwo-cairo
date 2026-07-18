//! Prepare-time admission for the complete eager Composition execution.
//!
//! CUDA execution remains owned by `PreparedCompositionGraph`: it already
//! carries the installed wave functions and exact resident-arena pointers.
//! This module closes the missing proof that that prepared graph is the one
//! described by the complete structural and linked Composition authorities.

use stwo_backend_cuda::{CompositionSplitLaunchMode, COMPOSITION_RETAINED_COLUMNS};

use crate::arena_plan::ProofArenaPlan;
use crate::prepared_composition::{
    CompositionAbi, CompositionExecutionAuthority, CompositionOperationKind, CompositionOutputMode,
    CompositionValueRole, CompositionWorkspaceRequirements, PreparedCompositionError,
    PreparedCompositionGraph,
};

const PRELUDE_OPERATIONS: usize = 2;
const TERMINAL_OPERATIONS: usize = 2;
const TERMINAL_CHILDREN: usize = 5;

/// Immutable evidence that one prepared Composition graph was admitted against
/// the exact structural program and target-specific module pack.
#[derive(Clone, Copy)]
pub(super) struct AdmittedCompositionExecution {
    _program_identity: [u8; 32],
    _linked_identity: [u8; 32],
    _static_module_build_identity: [u8; 32],
    _target_sm: u32,
}

/// Fail-closed lifecycle for the opt-in eager vertical path. A partial run
/// poisons the runtime; rebuilding it is the only sound retry.
#[derive(Clone, Copy)]
pub(super) enum CompiledCompositionAdmission {
    Unadmitted,
    Ready(AdmittedCompositionExecution),
    Poisoned,
}

impl Default for CompiledCompositionAdmission {
    fn default() -> Self {
        Self::Unadmitted
    }
}

impl CompiledCompositionAdmission {
    pub(super) fn needs_publish(&self) -> Result<bool, &'static str> {
        match self {
            Self::Unadmitted => Ok(true),
            Self::Ready(_) => Ok(false),
            Self::Poisoned => Err("compiled eager runtime is poisoned"),
        }
    }

    pub(super) fn publish(
        &mut self,
        admitted: AdmittedCompositionExecution,
    ) -> Result<(), &'static str> {
        if matches!(self, Self::Poisoned) {
            return Err("compiled eager runtime is poisoned");
        }
        *self = Self::Ready(admitted);
        Ok(())
    }

    pub(super) fn begin_run(&mut self) -> Result<AdmittedCompositionExecution, &'static str> {
        match *self {
            Self::Ready(admitted) => {
                *self = Self::Poisoned;
                Ok(admitted)
            }
            Self::Unadmitted => Err("compiled Composition was not admitted before replay"),
            Self::Poisoned => Err("compiled eager runtime is poisoned"),
        }
    }

    pub(super) fn complete_run(
        &mut self,
        admitted: AdmittedCompositionExecution,
    ) -> Result<(), &'static str> {
        if !matches!(self, Self::Poisoned) {
            return Err("compiled eager runtime did not own the active run");
        }
        *self = Self::Ready(admitted);
        Ok(())
    }
}

impl AdmittedCompositionExecution {
    /// Admit once before entering the replay window. No authority recompilation,
    /// module lookup, or receipt scan occurs in timed replay.
    pub(super) fn admit(
        plan: &ProofArenaPlan,
        prepared: &PreparedCompositionGraph<'_>,
        target_sm: u32,
    ) -> Result<Self, &'static str> {
        let authority = CompositionExecutionAuthority::compile(plan)
            .map_err(|_| "structural Composition authority")?;
        let requirements = &plan.composition().requirements;
        validate_exact_program(&authority, requirements)?;
        validate_prepared_graph(plan, prepared)?;
        let receipt = requirements
            .execution_receipt
            .ok_or("planned Composition execution receipt")?;

        let linked = authority
            .bind_linked(plan, target_sm)
            .map_err(|_| "linked Composition authority")?
            .ok_or("static CUDA Composition module is unavailable")?;
        if linked.program_identity() != authority.identity()
            || linked.target_sm() != target_sm
            || linked.static_source_identity() == [0; 32]
            || linked.static_module_build_identity() == [0; 32]
            || linked.identity() == [0; 32]
            || linked.loaded_waves().len() != receipt.wave_count
        {
            return Err("linked Composition identity");
        }
        for (expected, loaded) in authority.waves().iter().zip(linked.loaded_waves()) {
            if loaded.structural() != expected
                || loaded.target_sm() != target_sm
                || loaded.manifest_identity() == &[0; 32]
                || loaded.cubin_identity() == &[0; 32]
                || loaded.kernel_authority_identity() == &[0; 32]
            {
                return Err("linked Composition wave identity");
            }
        }

        Ok(Self {
            _program_identity: authority.identity(),
            _linked_identity: linked.identity(),
            _static_module_build_identity: linked.static_module_build_identity(),
            _target_sm: target_sm,
        })
    }

    /// Enqueue the already-admitted complete Composition graph.
    pub(super) fn launch_eager(
        &self,
        prepared: &PreparedCompositionGraph<'_>,
    ) -> Result<(), PreparedCompositionError> {
        prepared.launch()
    }
}

fn validate_exact_program(
    authority: &CompositionExecutionAuthority,
    requirements: &CompositionWorkspaceRequirements,
) -> Result<(), &'static str> {
    let operations = authority.operations();
    let receipt = requirements
        .execution_receipt
        .ok_or("planned Composition execution receipt")?;
    let lift_count = requirements
        .accumulators
        .len()
        .checked_sub(1)
        .ok_or("Composition accumulator topology")?;
    let operation_count = PRELUDE_OPERATIONS
        .checked_add(receipt.wave_count)
        .and_then(|count| count.checked_add(lift_count))
        .and_then(|count| count.checked_add(TERMINAL_OPERATIONS))
        .ok_or("Composition operation count overflow")?;
    let child_count = PRELUDE_OPERATIONS
        .checked_add(receipt.wave_count)
        .and_then(|count| count.checked_add(lift_count))
        .and_then(|count| count.checked_add(TERMINAL_CHILDREN))
        .ok_or("Composition child count overflow")?;
    if operations.len() != operation_count
        || operations
            .iter()
            .map(|operation| operation.children.len())
            .sum::<usize>()
            != child_count
        || authority.waves().len() != receipt.wave_count
    {
        return Err("Composition operation topology");
    }

    let mut cursor = operations.iter();
    let materialize = cursor.next().ok_or("Composition materialize operation")?;
    let powers = cursor.next().ok_or("Composition powers operation")?;
    if materialize.abi != CompositionAbi::MaterializeExtParamsV1
        || !matches!(
            materialize.kind,
            CompositionOperationKind::MaterializeExtParams { .. }
        )
        || powers.abi != CompositionAbi::GenerateDescendingPowersV1
        || !matches!(
            powers.kind,
            CompositionOperationKind::GenerateDescendingPowers { .. }
        )
    {
        return Err("Composition prelude order");
    }

    let mut parts = 0usize;
    for expected_wave in 0..receipt.wave_count {
        let operation = cursor.next().ok_or("Composition wave operation")?;
        let CompositionOperationKind::Wave {
            wave_index,
            part_count,
            ..
        } = operation.kind
        else {
            return Err("Composition wave order");
        };
        if operation.abi != CompositionAbi::WaveV2
            || wave_index as usize != expected_wave
            || operation.children.len() != 1
        {
            return Err("Composition wave order");
        }
        parts = parts
            .checked_add(part_count as usize)
            .ok_or("Composition part count overflow")?;
    }
    if parts != receipt.part_count {
        return Err("Composition part count");
    }

    for expected_lift in 0..lift_count {
        let operation = cursor.next().ok_or("Composition lift operation")?;
        let CompositionOperationKind::LiftAccumulate { lift_index, .. } = operation.kind else {
            return Err("Composition lift order");
        };
        if operation.abi != CompositionAbi::LiftAccumulateV1
            || lift_index as usize != expected_lift
            || operation.children.len() != 1
        {
            return Err("Composition lift order");
        }
    }

    let inverse = cursor.next().ok_or("Composition split inverse")?;
    let forward = cursor.next().ok_or("Composition split forward")?;
    if inverse.abi != CompositionAbi::SplitInverseFusedFirstForwardV1
        || !matches!(
            inverse.kind,
            CompositionOperationKind::SplitInverseFusedFirstForward { .. }
        )
        || inverse.children.len() != 3
        || forward.abi != CompositionAbi::SplitForwardAfterFirstIntervalV1
        || !matches!(
            forward.kind,
            CompositionOperationKind::SplitForwardAfterFirstInterval { .. }
        )
        || forward.children.len() != 2
        || cursor.next().is_some()
    {
        return Err("Composition terminal order");
    }

    if authority.outputs().len() != COMPOSITION_RETAINED_COLUMNS {
        return Err("Composition terminal output count");
    }
    for (column, output) in authority.outputs().iter().enumerate() {
        if *output
            != (CompositionValueRole::SplitRetained {
                canonical_column: column as u8,
                generation: 3,
            })
        {
            return Err("Composition terminal outputs");
        }
    }
    Ok(())
}

fn validate_prepared_graph(
    plan: &ProofArenaPlan,
    prepared: &PreparedCompositionGraph<'_>,
) -> Result<(), &'static str> {
    let planned = &plan.composition().requirements;
    let actual = prepared.requirements();
    let receipt = actual
        .execution_receipt
        .ok_or("prepared Composition execution receipt")?;
    if !actual.same_address_free_structure(planned)
        || actual.mode != crate::prepared_composition::CompositionLaunchMode::Wave
        || receipt.wave_count == 0
        || receipt.part_count < receipt.wave_count
        || actual.waves.len() != receipt.wave_count
        || actual
            .waves
            .iter()
            .map(|wave| wave.parts.len())
            .sum::<usize>()
            != receipt.part_count
        || prepared.output_mode() != CompositionOutputMode::DirectRetainedEvaluations
        || prepared.direct_split_launch_mode()
            != Some(CompositionSplitLaunchMode::FusedFirstForward)
        || plan.composition().output_plan.mode() != CompositionOutputMode::DirectRetainedEvaluations
    {
        return Err("prepared Composition graph receipt");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_sn2_seals_exact_eager_orchestration() {
        const OPERATION_COUNT: usize = 31;
        const CHILD_COUNT: usize = 34;
        const WAVE_COUNT: usize = 14;
        const PART_COUNT: usize = 123;
        const LIFT_COUNT: usize = 13;

        let executable = crate::program_image::generated_sn2_replacement();
        let plan = executable.arena();
        let authority = CompositionExecutionAuthority::compile(plan).unwrap();

        validate_exact_program(&authority, &plan.composition().requirements).unwrap();
        assert_eq!(authority.operations().len(), OPERATION_COUNT);
        assert_eq!(
            authority
                .operations()
                .iter()
                .map(|operation| operation.children.len())
                .sum::<usize>(),
            CHILD_COUNT
        );
        assert_eq!(authority.outputs().len(), COMPOSITION_RETAINED_COLUMNS);
        assert_eq!(
            plan.composition().requirements.accumulators.len() - 1,
            LIFT_COUNT
        );

        let requirements = &plan.composition().requirements;
        assert_eq!(
            requirements.execution_receipt,
            Some(crate::prepared_composition::CompositionExecutionReceipt {
                part_count: PART_COUNT,
                wave_count: WAVE_COUNT,
            })
        );
        assert_eq!(
            plan.composition().output_plan.mode(),
            CompositionOutputMode::DirectRetainedEvaluations
        );
    }

    #[test]
    fn partial_eager_run_poison_is_fail_closed() {
        let admitted = AdmittedCompositionExecution {
            _program_identity: [1; 32],
            _linked_identity: [2; 32],
            _static_module_build_identity: [3; 32],
            _target_sm: 89,
        };
        let mut state = CompiledCompositionAdmission::default();
        assert_eq!(state.needs_publish(), Ok(true));
        assert!(state.begin_run().is_err());
        state.publish(admitted).unwrap();
        assert_eq!(state.needs_publish(), Ok(false));
        state.publish(admitted).unwrap();
        let active = state.begin_run().unwrap();
        assert!(state.begin_run().is_err());
        assert!(state.needs_publish().is_err());
        assert!(state.publish(admitted).is_err());
        state.complete_run(active).unwrap();
        assert!(state.begin_run().is_ok());
        assert!(state.publish(admitted).is_err());
    }
}
