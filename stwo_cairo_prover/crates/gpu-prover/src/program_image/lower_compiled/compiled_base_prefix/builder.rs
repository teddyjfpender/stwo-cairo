//! Append-only cursor from the complete witness prefix into later Base work.
//!
//! This cursor owns the only semantic value map. A stage receipt is sealed
//! before executable operations are appended, and missing linked authority
//! leaves that receipt retryable without publishing a partial operation list.

use std::collections::BTreeMap;

use stwo_backend_cuda::MemoryBaseTraceStepKind;

use super::super::producer_prefix::BaseProducerAuthority;
use super::super::{fixed_table_materialization, memory_base_trace, InvocationShapeError};
use super::{
    emission, insert_effect, push_operation, validate_causal_value_closure, wrapper_id,
    CompiledWitnessWriterPrefix,
};
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{
    EffectContract, EffectContractId, ExecutionPrimitive, PartitionAuthority,
    StaticCudaWrapperAuthority, StaticCudaWrapperId,
};

mod validation;

use validation::{
    seal_fixed_image_roots, validate_published_base_dag, FixedImageRootReceipt, PublishedBaseStage,
};

type MemoryStaticResolver = fn(
    StaticCudaWrapperId,
    u32,
    &memory_base_trace::LoweredMemoryBaseTrace,
    usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError>;

type FixedTableStaticResolver =
    fn(
        StaticCudaWrapperId,
        u32,
        &fixed_table_materialization::LoweredFixedTableStage,
        usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError>;

/// The sole continuation of a source-complete witness prefix.
///
/// Later Base producers belong here so they cannot allocate from a detached
/// semantic map or overtake the memory stage.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum CompiledBaseDagStartError {
    IncompleteWitnessPrefix(Box<CompiledWitnessWriterPrefix>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompiledBaseDagAppendError {
    InvalidStage,
    Lowering,
    MissingMemoryStaticWrapper {
        step_ordinal: u32,
        kind: MemoryBaseTraceStepKind,
    },
    MissingFixedTableStaticWrapper {
        table_ordinal: u32,
        component: &'static str,
    },
}

#[derive(Debug, Eq, PartialEq)]
struct SealedMemoryBaseTrace {
    lowered: Option<memory_base_trace::LoweredMemoryBaseTrace>,
    emitted: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct SealedFixedTables {
    lowered: fixed_table_materialization::LoweredFixedTableStage,
    fixed_image_roots: FixedImageRootReceipt,
    emitted: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct CompiledBaseDagBuilder {
    prefix: CompiledWitnessWriterPrefix,
    memory: Option<SealedMemoryBaseTrace>,
    fixed_tables: Option<SealedFixedTables>,
}

impl CompiledBaseDagBuilder {
    pub(super) fn from_complete_witness(
        prefix: CompiledWitnessWriterPrefix,
    ) -> Result<Self, CompiledBaseDagStartError> {
        let monolithic = PartitionAuthority::monolithic();
        let complete = prefix.next_producer == prefix.base_authority.producers.len()
            && validate_causal_value_closure(&prefix.values, &prefix.effects, &prefix.operations)
                .map(|roots| prefix.causal_external_roots.as_ref() == Some(&roots))
                .unwrap_or(false)
            && emission::validate_sealed_prefix(
                &prefix.base_authority,
                &prefix.causal_setup,
                prefix.next_producer,
                prefix.target_sm,
                &prefix.kernel_by_build_authority,
                &prefix.kernels,
                &prefix.static_wrappers,
                &prefix.module_global_initializers,
                &prefix.effects,
                &prefix.partitions,
                &monolithic,
                &prefix.operations,
            )
            .is_ok();
        if !complete {
            return Err(CompiledBaseDagStartError::IncompleteWitnessPrefix(
                Box::new(prefix),
            ));
        }
        Ok(Self {
            prefix,
            memory: None,
            fixed_tables: None,
        })
    }

    /// Seal the exact post-memory allocator and semantic receipt atomically.
    pub(super) fn append_memory_semantics(
        &mut self,
        arena: &ProofArenaPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.memory.is_some() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_arena_authority(arena)?;
        let before = self.prefix.values.clone();
        let mut after = before.clone();
        let lowered = memory_base_trace::lower_stage(arena, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        memory_base_trace::validate_from(arena, &before, &after, &lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        self.prefix.values = after;
        self.memory = Some(SealedMemoryBaseTrace {
            lowered,
            emitted: false,
        });
        Ok(())
    }

    /// Append every memory operation or none.
    pub(super) fn emit_memory_operations(&mut self) -> Result<(), CompiledBaseDagAppendError> {
        self.emit_memory_operations_using(memory_base_trace::resolve_static_wrapper)
    }

    fn emit_memory_operations_using(
        &mut self,
        resolve: MemoryStaticResolver,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .memory
            .as_ref()
            .filter(|memory| !memory.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let Some(lowered) = sealed.lowered.as_ref() else {
            let roots = validate_published_base_dag(
                &self.prefix,
                self.memory.as_ref(),
                None,
                &self.prefix.values,
                &self.prefix.effects,
                &self.prefix.operations,
                &self.prefix.static_wrappers,
                PublishedBaseStage::Memory,
            )?;
            if self.prefix.causal_external_roots.as_ref() != Some(&roots) {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            self.memory
                .as_mut()
                .ok_or(CompiledBaseDagAppendError::InvalidStage)?
                .emitted = true;
            return Ok(());
        };
        if self.prefix.partitions != vec![PartitionAuthority::monolithic()] {
            return Err(CompiledBaseDagAppendError::Lowering);
        }

        let mut effects = exact_effect_map(&self.prefix.effects)?;
        let mut operations = self.prefix.operations.clone();
        let mut wrappers = Vec::with_capacity(lowered.steps().len());
        let monolithic = PartitionAuthority::monolithic();
        for (step_ordinal, step) in lowered.steps().iter().enumerate() {
            let existing = self
                .prefix
                .static_wrappers
                .len()
                .checked_add(wrappers.len())
                .ok_or(CompiledBaseDagAppendError::Lowering)?;
            let id = wrapper_id(existing).map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            let Some(wrapper) = resolve(id, self.prefix.target_sm, lowered, step_ordinal)
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            else {
                return Err(CompiledBaseDagAppendError::MissingMemoryStaticWrapper {
                    step_ordinal: u32::try_from(step_ordinal)
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                    kind: step.kind(),
                });
            };
            if wrapper.id() != id
                || wrapper.consumer_target_sm() != self.prefix.target_sm
                || wrapper.accepted_invocation()
                    != step
                        .invocation()
                        .contract_id()
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?
                || wrapper.accepted_effect() != step.effect().id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            insert_effect(&mut effects, step.effect().clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            push_operation(
                ExecutionPrimitive::StaticCudaWrapper { wrapper: id },
                Some(step.invocation().clone()),
                step.effect().id(),
                &monolithic,
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }

        let effects = effects.into_values().collect::<Vec<_>>();
        let mut static_wrappers = self.prefix.static_wrappers.clone();
        static_wrappers.extend(wrappers);
        let roots = validate_published_base_dag(
            &self.prefix,
            self.memory.as_ref(),
            None,
            &self.prefix.values,
            &effects,
            &operations,
            &static_wrappers,
            PublishedBaseStage::Memory,
        )?;
        if self.prefix.causal_external_roots.as_ref() != Some(&roots) {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.prefix.static_wrappers = static_wrappers;
        self.prefix.effects = effects;
        self.prefix.operations = operations;
        self.memory
            .as_mut()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?
            .emitted = true;
        Ok(())
    }

    pub(super) fn has_complete_memory_stage(&self) -> bool {
        self.memory.as_ref().is_some_and(|memory| memory.emitted)
    }

    /// Seal the fixed-table stage only after memory publication.
    pub(super) fn append_fixed_table_semantics(
        &mut self,
        arena: &ProofArenaPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        if self.fixed_tables.is_some() || !self.has_complete_memory_stage() {
            return Err(CompiledBaseDagAppendError::InvalidStage);
        }
        self.validate_arena_authority(arena)?;
        let before = self.prefix.values.clone();
        let mut after = before.clone();
        let lowered = fixed_table_materialization::lower_stage(arena, &mut after)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        fixed_table_materialization::validate_from(arena, &before, &after, &lowered)
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        let fixed_image_roots = seal_fixed_image_roots(arena, &before, &after, &lowered)?;
        self.prefix.values = after;
        self.fixed_tables = Some(SealedFixedTables {
            lowered,
            fixed_image_roots,
            emitted: false,
        });
        Ok(())
    }

    /// Append every fixed-table operation or none.
    pub(super) fn emit_fixed_table_operations(&mut self) -> Result<(), CompiledBaseDagAppendError> {
        self.emit_fixed_table_operations_using(fixed_table_materialization::resolve_static_wrapper)
    }

    fn emit_fixed_table_operations_using(
        &mut self,
        resolve: FixedTableStaticResolver,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let sealed = self
            .fixed_tables
            .as_ref()
            .filter(|fixed| !fixed.emitted)
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        if self.prefix.partitions != vec![PartitionAuthority::monolithic()] {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        let mut effects = exact_effect_map(&self.prefix.effects)?;
        let mut operations = self.prefix.operations.clone();
        let mut wrappers = Vec::with_capacity(sealed.lowered.tables().len());
        let monolithic = PartitionAuthority::monolithic();
        for (table_ordinal, table) in sealed.lowered.tables().iter().enumerate() {
            let existing = self
                .prefix
                .static_wrappers
                .len()
                .checked_add(wrappers.len())
                .ok_or(CompiledBaseDagAppendError::Lowering)?;
            let id = wrapper_id(existing).map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            let Some(wrapper) = resolve(id, self.prefix.target_sm, &sealed.lowered, table_ordinal)
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            else {
                return Err(CompiledBaseDagAppendError::MissingFixedTableStaticWrapper {
                    table_ordinal: u32::try_from(table_ordinal)
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?,
                    component: table.component(),
                });
            };
            if wrapper.id() != id
                || wrapper.consumer_target_sm() != self.prefix.target_sm
                || wrapper.accepted_invocation()
                    != table
                        .invocation()
                        .contract_id()
                        .map_err(|_| CompiledBaseDagAppendError::Lowering)?
                || wrapper.accepted_effect() != table.effect().id()
                || !wrapper
                    .has_valid_identity()
                    .map_err(|_| CompiledBaseDagAppendError::Lowering)?
            {
                return Err(CompiledBaseDagAppendError::Lowering);
            }
            insert_effect(&mut effects, table.effect().clone())
                .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            push_operation(
                ExecutionPrimitive::StaticCudaWrapper { wrapper: id },
                Some(table.invocation().clone()),
                table.effect().id(),
                &monolithic,
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }
        let effects = effects.into_values().collect::<Vec<_>>();
        let mut static_wrappers = self.prefix.static_wrappers.clone();
        static_wrappers.extend(wrappers);
        let roots = validate_published_base_dag(
            &self.prefix,
            self.memory.as_ref(),
            self.fixed_tables.as_ref(),
            &self.prefix.values,
            &effects,
            &operations,
            &static_wrappers,
            PublishedBaseStage::FixedTables,
        )?;
        let fixed = self
            .fixed_tables
            .as_ref()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?;
        let mut expected_roots = self
            .prefix
            .causal_external_roots
            .clone()
            .ok_or(CompiledBaseDagAppendError::Lowering)?;
        expected_roots.extend(fixed.fixed_image_roots.distinct_versions());
        if roots != expected_roots {
            return Err(CompiledBaseDagAppendError::Lowering);
        }
        self.prefix.static_wrappers = static_wrappers;
        self.prefix.effects = effects;
        self.prefix.operations = operations;
        self.prefix.causal_external_roots = Some(roots);
        self.fixed_tables
            .as_mut()
            .ok_or(CompiledBaseDagAppendError::InvalidStage)?
            .emitted = true;
        Ok(())
    }

    pub(super) fn has_complete_fixed_table_stage(&self) -> bool {
        self.fixed_tables
            .as_ref()
            .is_some_and(|fixed| fixed.emitted)
    }

    fn validate_arena_authority(
        &self,
        arena: &ProofArenaPlan,
    ) -> Result<(), CompiledBaseDagAppendError> {
        let exact = BaseProducerAuthority::compile_replacement(
            arena,
            self.prefix.base_authority.preprocessed_trace_variant,
        )
        .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
        if exact == self.prefix.base_authority {
            Ok(())
        } else {
            Err(CompiledBaseDagAppendError::Lowering)
        }
    }
}

fn exact_effect_map(
    effects: &[EffectContract],
) -> Result<BTreeMap<EffectContractId, EffectContract>, CompiledBaseDagAppendError> {
    let mut exact = BTreeMap::new();
    for effect in effects {
        insert_effect(&mut exact, effect.clone())
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
    }
    (exact.len() == effects.len())
        .then_some(exact)
        .ok_or(CompiledBaseDagAppendError::Lowering)
}

#[cfg(test)]
#[path = "builder/tests.rs"]
mod tests;
