//! Append-only cursor from the complete witness prefix into later Base work.
//!
//! This cursor owns the only semantic value map. A memory receipt is sealed
//! before executable operations may be appended, and missing linked authority
//! leaves that receipt retryable without publishing a partial operation list.

use std::collections::BTreeMap;

use stwo_backend_cuda::MemoryBaseTraceStepKind;

use super::super::producer_prefix::BaseProducerAuthority;
use super::super::{fixed_table_materialization, memory_base_trace, InvocationShapeError};
use super::{emission, insert_effect, push_operation, wrapper_id, CompiledWitnessWriterPrefix};
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{
    EffectContract, EffectContractId, ExecutionPrimitive, PartitionAuthority,
    StaticCudaWrapperAuthority, StaticCudaWrapperId,
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
    emitted: bool,
}

/// The sole continuation of a source-complete witness prefix.
///
/// Later Base producers belong here so they cannot allocate from a detached
/// `SemanticValueMap` or overtake the memory stage.
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
            && emission::validate_sealed_prefix(
                &prefix.base_authority,
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

    /// Append all memory operations or none. A missing loaded static build is
    /// a retryable frontier and cannot discard the semantic receipt.
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
                step.invocation().clone(),
                step.effect().id(),
                &monolithic,
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }

        self.prefix.static_wrappers.extend(wrappers);
        self.prefix.effects = effects.into_values().collect();
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

    /// Seal every fixed table and the post-fixed allocator atomically. This may
    /// run only after the memory operation sequence has been published.
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
        self.prefix.values = after;
        self.fixed_tables = Some(SealedFixedTables {
            lowered,
            emitted: false,
        });
        Ok(())
    }

    /// Append all fixed-table effects, wrappers and operations or none.
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
                table.invocation().clone(),
                table.effect().id(),
                &monolithic,
                &mut operations,
            )
            .map_err(|_| CompiledBaseDagAppendError::Lowering)?;
            wrappers.push(wrapper);
        }
        self.prefix.static_wrappers.extend(wrappers);
        self.prefix.effects = effects.into_values().collect();
        self.prefix.operations = operations;
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
mod tests {
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

    use super::*;
    use crate::compiled_proof::{
        LaunchGeometry, ProofStage, StaticCudaLaunchIdentity, StaticCudaWrapperAuthority,
    };
    use crate::program_image::lower_compiled::compiled_base_prefix::{
        emit_recorded_witness_writer_prefix_for_test, CompiledWitnessWriterPrefix,
    };
    use crate::program_image::lower_compiled::compiled_base_prefix_tests::{
        exact_fields, resolve_all_static_for_prefix, MANIFEST, TARGET_SM,
    };
    use crate::transcript_plan::CairoTranscriptSegment;

    fn complete_prefix() -> CompiledWitnessWriterPrefix {
        let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
        emit_recorded_witness_writer_prefix_for_test(
            executable.arena(),
            PreProcessedTraceVariant::Canonical,
            MANIFEST,
            TARGET_SM,
            |source| Ok(exact_fields(source)),
            resolve_all_static_for_prefix,
        )
        .unwrap()
    }

    fn fake_memory_wrapper(
        id: StaticCudaWrapperId,
        target_sm: u32,
        lowered: &memory_base_trace::LoweredMemoryBaseTrace,
        step_ordinal: usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
        let step = lowered
            .steps()
            .get(step_ordinal)
            .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
        let launch = StaticCudaLaunchIdentity::new(
            b"test_memory_base_kernel".to_vec(),
            LaunchGeometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
        )
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
        StaticCudaWrapperAuthority::new(
            id,
            [0x51; 32],
            target_sm,
            b"test_memory_base_wrapper".to_vec(),
            [0x52; 32],
            [0x53; 32],
            [0x54; 32],
            [0x55; 32],
            vec![launch],
            step.invocation()
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?,
            step.effect().id(),
        )
        .map(Some)
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)
    }

    fn miss_second_memory_wrapper(
        id: StaticCudaWrapperId,
        target_sm: u32,
        lowered: &memory_base_trace::LoweredMemoryBaseTrace,
        step_ordinal: usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
        if step_ordinal == 1 {
            Ok(None)
        } else {
            fake_memory_wrapper(id, target_sm, lowered, step_ordinal)
        }
    }

    fn fake_fixed_table_wrapper(
        id: StaticCudaWrapperId,
        target_sm: u32,
        lowered: &fixed_table_materialization::LoweredFixedTableStage,
        table_ordinal: usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
        let table = lowered
            .tables()
            .get(table_ordinal)
            .ok_or(InvocationShapeError::InvalidFixedTableBinding)?;
        let launch = StaticCudaLaunchIdentity::new(
            b"test_fixed_table_kernel".to_vec(),
            LaunchGeometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
        )
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
        StaticCudaWrapperAuthority::new(
            id,
            [0x61; 32],
            target_sm,
            b"test_fixed_table_wrapper".to_vec(),
            [0x62; 32],
            [0x63; 32],
            [0x64; 32],
            [0x65; 32],
            vec![launch],
            table
                .invocation()
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?,
            table.effect().id(),
        )
        .map(Some)
        .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)
    }

    fn miss_second_fixed_table_wrapper(
        id: StaticCudaWrapperId,
        target_sm: u32,
        lowered: &fixed_table_materialization::LoweredFixedTableStage,
        table_ordinal: usize,
    ) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
        if table_ordinal == 1 {
            Ok(None)
        } else {
            fake_fixed_table_wrapper(id, target_sm, lowered, table_ordinal)
        }
    }

    #[test]
    fn cursor_rejects_an_incomplete_witness_prefix_without_losing_it() {
        let mut prefix = complete_prefix();
        prefix.next_producer -= 1;
        let operation_count = prefix.operations.len();
        let CompiledBaseDagStartError::IncompleteWitnessPrefix(returned) =
            CompiledBaseDagBuilder::from_complete_witness(prefix).unwrap_err();
        assert_eq!(returned.operations.len(), operation_count);
        assert_eq!(
            returned.next_producer + 1,
            returned.base_authority.producers.len()
        );
    }

    #[test]
    fn semantic_appends_reject_arena_authority_drift_transactionally() {
        let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();

        let mut memory = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
        memory.prefix.base_authority.producers.pop().unwrap();
        let values = memory.prefix.values.clone();
        assert_eq!(
            memory.append_memory_semantics(executable.arena()),
            Err(CompiledBaseDagAppendError::Lowering)
        );
        assert_eq!(memory.prefix.values, values);
        assert!(memory.memory.is_none());

        let mut fixed = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
        fixed.append_memory_semantics(executable.arena()).unwrap();
        fixed
            .emit_memory_operations_using(fake_memory_wrapper)
            .unwrap();
        fixed.prefix.base_authority.producers.pop().unwrap();
        let values = fixed.prefix.values.clone();
        assert_eq!(
            fixed.append_fixed_table_semantics(executable.arena()),
            Err(CompiledBaseDagAppendError::Lowering)
        );
        assert_eq!(fixed.prefix.values, values);
        assert!(fixed.fixed_tables.is_none());
    }

    #[test]
    fn memory_receipt_and_operations_are_append_only_and_transactional() {
        let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
        let prefix = complete_prefix();
        let old_operations = prefix.operations.clone();
        let old_wrappers = prefix.static_wrappers.clone();
        let old_effects = prefix.effects.clone();
        let old_direct_b2n = prefix.base_authority.direct_retained_b2n.clone();
        let before = prefix.values.clone();
        let mut builder = CompiledBaseDagBuilder::from_complete_witness(prefix).unwrap();

        assert_eq!(
            builder.emit_memory_operations(),
            Err(CompiledBaseDagAppendError::InvalidStage)
        );
        builder.append_memory_semantics(executable.arena()).unwrap();
        assert_eq!(builder.prefix.operations, old_operations);
        assert_eq!(builder.prefix.static_wrappers, old_wrappers);
        assert_eq!(builder.prefix.effects, old_effects);
        assert_eq!(
            builder.prefix.base_authority.direct_retained_b2n,
            old_direct_b2n
        );
        let missing_kind = {
            let sealed = builder.memory.as_ref().unwrap();
            let lowered = sealed.lowered.as_ref().unwrap();
            assert!(lowered.steps().len() > 1);
            assert!(!sealed.emitted);
            memory_base_trace::validate_from(
                executable.arena(),
                &before,
                &builder.prefix.values,
                &sealed.lowered,
            )
            .unwrap();
            lowered.steps()[1].kind()
        };

        let post_memory_values = builder.prefix.values.clone();
        assert_eq!(
            builder.append_memory_semantics(executable.arena()),
            Err(CompiledBaseDagAppendError::InvalidStage)
        );
        assert_eq!(builder.prefix.values, post_memory_values);

        assert_eq!(
            builder.emit_memory_operations_using(miss_second_memory_wrapper),
            Err(CompiledBaseDagAppendError::MissingMemoryStaticWrapper {
                step_ordinal: 1,
                kind: missing_kind,
            })
        );
        assert_eq!(builder.prefix.operations, old_operations);
        assert_eq!(builder.prefix.static_wrappers, old_wrappers);
        assert_eq!(builder.prefix.effects, old_effects);
        assert_eq!(builder.prefix.values, post_memory_values);
        assert!(!builder.memory.as_ref().unwrap().emitted);

        let old_operation_count = builder.prefix.operations.len();
        let old_wrapper_count = builder.prefix.static_wrappers.len();
        builder
            .emit_memory_operations_using(fake_memory_wrapper)
            .unwrap();
        let lowered = builder.memory.as_ref().unwrap().lowered.as_ref().unwrap();
        assert_eq!(
            builder.prefix.operations.len(),
            old_operation_count + lowered.steps().len()
        );
        assert_eq!(
            builder.prefix.static_wrappers.len(),
            old_wrapper_count + lowered.steps().len()
        );
        for (ordinal, step) in lowered.steps().iter().enumerate() {
            let operation = &builder.prefix.operations[old_operation_count + ordinal];
            let wrapper = &builder.prefix.static_wrappers[old_wrapper_count + ordinal];
            assert_eq!(
                operation.primitive,
                ExecutionPrimitive::StaticCudaWrapper {
                    wrapper: wrapper.id()
                }
            );
            assert_eq!(operation.invocation.as_ref(), Some(step.invocation()));
            assert_eq!(operation.effect, step.effect().id());
            assert_eq!(
                operation.stage,
                ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
            );
            assert_eq!(wrapper.accepted_effect(), step.effect().id());
            assert_eq!(wrapper.consumer_target_sm(), TARGET_SM);
            assert!(builder
                .prefix
                .effects
                .iter()
                .any(|effect| effect == step.effect()));
        }
        assert!(builder.has_complete_memory_stage());
        assert_eq!(builder.prefix.values, post_memory_values);
        assert_eq!(
            builder.prefix.base_authority.direct_retained_b2n,
            old_direct_b2n
        );
        assert_eq!(
            builder.emit_memory_operations(),
            Err(CompiledBaseDagAppendError::InvalidStage)
        );
    }

    #[test]
    fn fixed_tables_publish_transactionally_after_memory_and_preserve_mixed_sources() {
        let executable = crate::program_image::lower_compiled::tests::generated_sn2_replacement();
        let mut builder = CompiledBaseDagBuilder::from_complete_witness(complete_prefix()).unwrap();
        assert_eq!(
            builder.append_fixed_table_semantics(executable.arena()),
            Err(CompiledBaseDagAppendError::InvalidStage)
        );
        builder.append_memory_semantics(executable.arena()).unwrap();
        builder
            .emit_memory_operations_using(fake_memory_wrapper)
            .unwrap();
        let before = builder.prefix.values.clone();
        builder
            .append_fixed_table_semantics(executable.arena())
            .unwrap();
        let sealed = builder.fixed_tables.as_ref().unwrap();
        fixed_table_materialization::validate_from(
            executable.arena(),
            &before,
            &builder.prefix.values,
            &sealed.lowered,
        )
        .unwrap();
        let mixed = sealed
            .lowered
            .tables()
            .iter()
            .find(|table| {
                table.sources().iter().any(|source| {
                    matches!(
                        source,
                        fixed_table_materialization::LoweredFixedTableSource::Arena { .. }
                    )
                }) && table.sources().iter().any(|source| {
                    matches!(
                        source,
                        fixed_table_materialization::LoweredFixedTableSource::Registered { .. }
                    )
                })
            })
            .expect("Pedersen-18 must retain seq_23 followed by registered columns");
        assert!(matches!(
            mixed.invocation().arguments[0].value,
            crate::compiled_proof::AotArgumentValue::DeviceMixedFixedSourcePointerTable(_)
        ));
        let (ordinary_catalog, ordinary_version) = mixed
            .sources()
            .iter()
            .find_map(|source| match source {
                fixed_table_materialization::LoweredFixedTableSource::Arena {
                    value,
                    version,
                    ..
                } => Some((*value, *version)),
                fixed_table_materialization::LoweredFixedTableSource::Registered { .. } => None,
            })
            .expect("mixed table must retain its ordinary preprocessed source");
        assert!(before.version(ordinary_catalog).is_err());
        assert_eq!(
            builder.prefix.values.version(ordinary_catalog),
            Ok(ordinary_version)
        );

        let old_operations = builder.prefix.operations.clone();
        let old_wrappers = builder.prefix.static_wrappers.clone();
        let old_effects = builder.prefix.effects.clone();
        assert!(matches!(
            builder.emit_fixed_table_operations_using(miss_second_fixed_table_wrapper),
            Err(CompiledBaseDagAppendError::MissingFixedTableStaticWrapper {
                table_ordinal: 1,
                ..
            })
        ));
        assert_eq!(builder.prefix.operations, old_operations);
        assert_eq!(builder.prefix.static_wrappers, old_wrappers);
        assert_eq!(builder.prefix.effects, old_effects);
        assert!(!builder.fixed_tables.as_ref().unwrap().emitted);

        let tables = builder
            .fixed_tables
            .as_ref()
            .unwrap()
            .lowered
            .tables()
            .len();
        builder
            .emit_fixed_table_operations_using(fake_fixed_table_wrapper)
            .unwrap();
        assert_eq!(
            builder.prefix.operations.len(),
            old_operations.len() + tables
        );
        assert_eq!(
            builder.prefix.static_wrappers.len(),
            old_wrappers.len() + tables
        );
        assert!(builder.has_complete_fixed_table_stage());
        assert_eq!(
            builder.emit_fixed_table_operations(),
            Err(CompiledBaseDagAppendError::InvalidStage)
        );
    }
}
