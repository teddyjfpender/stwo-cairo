//! Legacy migration mapping and production Base witness-producer authority.
//!
//! A producer enters `bound` only after its typed source ABI and ordinary value
//! effect compile. A stateful recorded producer additionally carries its exact
//! address-free resource/relocation recipe, but its value effect deliberately
//! omits module globals until a loaded-module publication receipt can bind real
//! addresses and canonical content. `BaseProducerBindingFrontier` is the
//! LegacyResident diagnostic frontier. `BaseProducerAuthority` is the
//! ReplacementV1 production Base slice, but is not permission to promote the
//! arena inventory into a full `CompiledProof`.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    DeviceArena, EcOpCompositeContract, EcOpSegmentStartReceipt, PreparedEcOpGraph,
    PreparedExecutionTablesGraph, PreparedWitnessFeedClearGraph, PreparedWitnessFeedGraph,
    PreparedWitnessGraph, WitnessFeedSourceUploadReceipt,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;

pub(crate) use super::blake_g_direct_execution_authority::PreparedBlakeGDirectKernel;
use super::ec_op_execution_authority::{
    NativeEcOpCompositeExecutionAuthority, NativeEcOpLinkedModuleAuthority,
};
use super::schedule_prefix::{
    append_interpolation_catalog_order, invocation_catalog_order, lower_base_interpolation,
    LoweredBaseInterpolationBatch,
};
use super::*;
use crate::arena_plan::{
    BufferPurpose, CommitmentColumnSource, LogicalBufferId, PlannedCommitment, ProofArenaPlan,
};
use crate::compiled_proof::{AotInvocation, EffectContract};
use crate::resident_runtime::producer_schedule::{
    BaseProducerSchedule, WitnessProducer, WitnessProducerKind,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ProducerSchedulePosition {
    pub(super) level: u32,
    pub(super) lane: u32,
    pub(super) ordinal: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MissingProducerAuthorityKind {
    BlakeGFusedComposite,
    BlakeGDirectComposite,
    NativeEcOpStaticModuleBuildIdentity,
    MultiplicityTransition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MissingProducerAuthority {
    pub(super) position: ProducerSchedulePosition,
    pub(super) producer: WitnessProducer,
    pub(super) missing: MissingProducerAuthorityKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredRecordedWitnessProducer {
    pub(super) position: ProducerSchedulePosition,
    pub(super) producer: WitnessProducer,
    pub(super) produced: Vec<ArenaCatalogValueId>,
    pub(super) source: RecordedWitnessInvocationShape,
    pub(super) invocation: AotInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredNativeEcOpProducer {
    pub(super) position: ProducerSchedulePosition,
    pub(super) producer: WitnessProducer,
    pub(super) execution: NativeEcOpCompositeExecutionAuthority,
    pub(super) contract: ec_op_prefix::LoweredNativeEcOpContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LoweredBaseProducer {
    Recorded(LoweredRecordedWitnessProducer),
    NativeEcOp(LoweredNativeEcOpProducer),
}

impl LoweredBaseProducer {
    pub(super) const fn position(&self) -> ProducerSchedulePosition {
        match self {
            Self::Recorded(producer) => producer.position,
            Self::NativeEcOp(producer) => producer.position,
        }
    }

    pub(super) const fn producer(&self) -> WitnessProducer {
        match self {
            Self::Recorded(producer) => producer.producer,
            Self::NativeEcOp(producer) => producer.producer,
        }
    }

    #[cfg(test)]
    pub(super) const fn recorded(&self) -> Option<&LoweredRecordedWitnessProducer> {
        match self {
            Self::Recorded(producer) => Some(producer),
            Self::NativeEcOp(_) => None,
        }
    }

    #[cfg(test)]
    pub(super) const fn native_ec_op(&self) -> Option<&LoweredNativeEcOpProducer> {
        match self {
            Self::Recorded(_) => None,
            Self::NativeEcOp(producer) => Some(producer),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BaseProducerBindingFrontier {
    pub(super) scheduled_producers: usize,
    pub(super) bound: Vec<LoweredBaseProducer>,
    pub(super) missing: Option<MissingProducerAuthority>,
    pub(super) semantic_values: adapter::SemanticValueMap,
    /// Independently sealed downstream primitive receipts. These are not part
    /// of the contiguous execution prefix while `missing` is present.
    pub(super) base_interpolation: Vec<LoweredBaseInterpolationBatch>,
}

/// Production authority for the currently closed ReplacementV1 Base slice.
/// It deliberately stops at Base: later transcript/proof operations still need
/// the real `CompiledProof` emitter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaseProducerAuthority {
    pub(super) execution_tables: Option<execution_tables::LoweredExecutionTables>,
    pub(super) multiplicity: multiplicity_coordinator::LoweredBaseMultiplicity,
    pub(super) producers: Vec<SemanticBaseProducer>,
    pub(super) direct_retained_b2n: stwo_backend_cuda::DirectRetainedB2nProgram,
    pub(super) preprocessed_trace_variant: PreProcessedTraceVariant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SemanticBaseProducer {
    Recorded(LoweredRecordedWitnessProducer),
    NativeBlakeGDirect {
        position: ProducerSchedulePosition,
        producer: WitnessProducer,
        contract: super::blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    },
    NativeEcOp {
        position: ProducerSchedulePosition,
        producer: WitnessProducer,
        contract: ec_op_prefix::LoweredNativeEcOpContract,
    },
}

impl SemanticBaseProducer {
    pub(super) const fn position(&self) -> ProducerSchedulePosition {
        match self {
            Self::Recorded(producer) => producer.position,
            Self::NativeBlakeGDirect { position, .. } | Self::NativeEcOp { position, .. } => {
                *position
            }
        }
    }

    pub(super) const fn producer(&self) -> WitnessProducer {
        match self {
            Self::Recorded(producer) => producer.producer,
            Self::NativeBlakeGDirect { producer, .. } | Self::NativeEcOp { producer, .. } => {
                *producer
            }
        }
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        match self {
            Self::Recorded(producer) => &producer.effect,
            Self::NativeBlakeGDirect { contract, .. } => &contract.effect,
            Self::NativeEcOp { contract, .. } => &contract.effect,
        }
    }
}

pub(crate) use super::loaded_base_binding::{
    LoadedBaseProducerAuthority, PreparedRegisteredFixedSource,
};

#[derive(Clone, Copy)]
pub(crate) struct PreparedRecordedKernel<'prepared, 'arena> {
    pub(crate) component: &'static str,
    pub(crate) part: stwo_cairo_prover::witness::proof_shape::TracePartId,
    pub(crate) arena: &'arena DeviceArena,
    pub(crate) writer: &'prepared PreparedWitnessGraph<'arena>,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedGenericMultiplicityFeed<'prepared, 'arena> {
    pub(crate) component: &'static str,
    pub(crate) part: stwo_cairo_prover::witness::proof_shape::TracePartId,
    pub(crate) graph: &'prepared PreparedWitnessFeedGraph<'arena>,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedPublicMemorySeed<'prepared, 'arena> {
    pub(crate) graph: &'prepared PreparedWitnessFeedGraph<'arena>,
    pub(crate) receipt: WitnessFeedSourceUploadReceipt,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedEcOpSegment<'prepared, 'arena> {
    pub(crate) graph: &'prepared PreparedEcOpGraph<'arena>,
    pub(crate) receipt: EcOpSegmentStartReceipt,
}

/// Complete raw runtime inventory for one exact Base witness prefix.
///
/// The binder consumes this as a closed set: duplicates, omissions, extras,
/// or a graph paired with the wrong schedule node are rejected.
#[derive(Clone, Copy)]
pub(crate) struct PreparedBaseProducerInventory<'prepared, 'arena> {
    pub(crate) arena: &'arena DeviceArena,
    pub(crate) registered_fixed_sources: &'prepared [PreparedRegisteredFixedSource],
    pub(crate) execution_tables: Option<&'prepared PreparedExecutionTablesGraph<'arena>>,
    pub(crate) recorded: &'prepared [PreparedRecordedKernel<'prepared, 'arena>],
    pub(crate) clear: &'prepared PreparedWitnessFeedClearGraph<'arena>,
    pub(crate) public_memory_seed: Option<PreparedPublicMemorySeed<'prepared, 'arena>>,
    pub(crate) generic_feeds: &'prepared [PreparedGenericMultiplicityFeed<'prepared, 'arena>],
    pub(crate) blake_g_direct: Option<PreparedBlakeGDirectKernel<'prepared, 'arena>>,
    pub(crate) ec_op_segment: Option<PreparedEcOpSegment<'prepared, 'arena>>,
}

impl BaseProducerAuthority {
    pub(crate) fn prepare_registered_fixed_sources(
        &self,
    ) -> Result<Vec<PreparedRegisteredFixedSource>, BaseProducerAuthorityError> {
        super::loaded_base_binding::prepare_registered_fixed_sources(self)
            .map_err(|_| BaseProducerAuthorityError)
    }

    pub(super) fn compile_replacement(
        arena: &ProofArenaPlan,
        preprocessed_trace_variant: PreProcessedTraceVariant,
    ) -> Result<Self, InvocationShapeError> {
        let mut values =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())?;
        Self::compile_replacement_into(arena, preprocessed_trace_variant, &mut values)
    }

    /// Lower Base into the caller-owned proof-wide semantic-version stream.
    /// The caller must retain this exact map for every later proof stage. The
    /// map is replaced only after the complete lowering succeeds.
    pub(super) fn compile_replacement_into(
        arena: &ProofArenaPlan,
        preprocessed_trace_variant: PreProcessedTraceVariant,
        values: &mut adapter::SemanticValueMap,
    ) -> Result<Self, InvocationShapeError> {
        let mut next_values = values.clone();
        let authority = Self::compile_replacement_uncommitted(
            arena,
            preprocessed_trace_variant,
            &mut next_values,
        )?;
        *values = next_values;
        Ok(authority)
    }

    fn compile_replacement_uncommitted(
        arena: &ProofArenaPlan,
        preprocessed_trace_variant: PreProcessedTraceVariant,
        values: &mut adapter::SemanticValueMap,
    ) -> Result<Self, InvocationShapeError> {
        let catalog = BaseProducerCatalog::compile(arena)?;
        let schedule = BaseProducerSchedule::compile(arena)
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
        if schedule.interpolation().is_some() {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        let base = arena
            .commitment(crate::arena_plan::CommitmentTreeId::Base)
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        let direct_retained_b2n = base
            .direct_retained_b2n_program
            .clone()
            .filter(|_| base.interpolation_batches.is_empty())
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;

        let scheduled = scheduled_producers(&schedule)?;
        let direct_outputs = direct_base_evaluations(&catalog, base)?;
        let mut pending = Vec::with_capacity(scheduled.len());
        for &(position, producer) in &scheduled {
            match producer.kind {
                WitnessProducerKind::Recorded => {
                    let planned = planned_recorded_component(arena, producer)?;
                    let source =
                        derive_invocation(&catalog, arena, planned).map_err(
                            |error| match error {
                                InvocationShapeError::MultiplicityNeedsSemanticVersions => {
                                    InvocationShapeError::InvalidProductionBaseAuthority
                                }
                                error => error,
                            },
                        )?;
                    let produced = produced_values(&catalog, planned)?;
                    validate_direct_recorded_outputs(
                        &catalog,
                        producer,
                        usize::try_from(planned.program.n_cols)
                            .map_err(|_| InvocationShapeError::SizeOverflow)?,
                        &produced,
                        &direct_outputs,
                    )?;
                    pending.push(PendingSemanticProducer::Recorded(PendingRecordedWitness {
                        position,
                        producer,
                        produced,
                        expected_trace_outputs: usize::try_from(planned.program.n_cols)
                            .map_err(|_| InvocationShapeError::SizeOverflow)?,
                        source,
                    }));
                }
                WitnessProducerKind::NativeEcOp => {
                    let contract = ec_op_prefix::prepare(&catalog, arena)?;
                    pending.push(PendingSemanticProducer::NativeEcOp {
                        position,
                        producer,
                        contract,
                    });
                }
                WitnessProducerKind::BlakeGDirect => {
                    let contract =
                        super::blake_g_direct_prefix::prepare(&catalog, arena, producer)?;
                    pending.push(PendingSemanticProducer::NativeBlakeGDirect {
                        position,
                        producer,
                        contract,
                    });
                }
                WitnessProducerKind::BlakeGFused => {
                    return Err(InvocationShapeError::InvalidProductionBaseAuthority)
                }
            }
        }

        let execution_tables = arena
            .execution_tables()
            .map(|_| execution_tables::lower_stage(arena, values))
            .transpose()?;
        let mut multiplicity = multiplicity_coordinator::BaseMultiplicityCoordinator::begin(
            arena,
            &schedule,
            preprocessed_trace_variant,
            values,
        )?;
        let mut producers = Vec::with_capacity(pending.len());
        for pending in pending {
            let semantic = match pending {
                PendingSemanticProducer::Recorded(recorded) => {
                    values.extend_ordered(invocation_catalog_order(
                        &recorded.source.source_arguments,
                    ))?;
                    let (invocation, effect) =
                        adapter::compile(&recorded.source.source_arguments, values)?;
                    SemanticBaseProducer::Recorded(LoweredRecordedWitnessProducer {
                        position: recorded.position,
                        producer: recorded.producer,
                        produced: recorded.produced,
                        source: recorded.source,
                        invocation,
                        effect,
                    })
                }
                PendingSemanticProducer::NativeEcOp {
                    position,
                    producer,
                    contract,
                } => {
                    let contract = ec_op_prefix::lower(contract, values)?;
                    validate_direct_native_outputs(&catalog, &contract, &direct_outputs)?;
                    SemanticBaseProducer::NativeEcOp {
                        position,
                        producer,
                        contract,
                    }
                }
                PendingSemanticProducer::NativeBlakeGDirect {
                    position,
                    producer,
                    contract,
                } => {
                    let contract = super::blake_g_direct_prefix::lower(contract, values)?;
                    validate_direct_blake_g_outputs(&catalog, &contract, &direct_outputs)?;
                    SemanticBaseProducer::NativeBlakeGDirect {
                        position,
                        producer,
                        contract,
                    }
                }
            };
            multiplicity.after_producer(&semantic, values)?;
            producers.push(semantic);
        }
        if producers.len() != scheduled.len() {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        let multiplicity = multiplicity.finish_witness(values)?;
        Ok(Self {
            execution_tables,
            multiplicity,
            producers,
            direct_retained_b2n,
            preprocessed_trace_variant,
        })
    }

    #[cfg(test)]
    pub(crate) fn producer_count(&self) -> usize {
        self.producers.len()
    }

    #[cfg(test)]
    pub(crate) fn direct_retained_b2n(&self) -> &stwo_backend_cuda::DirectRetainedB2nProgram {
        &self.direct_retained_b2n
    }

    pub(super) fn bind_loaded(
        &self,
        arena: &ProofArenaPlan,
        prepared: PreparedBaseProducerInventory<'_, '_>,
        device_ordinal: u32,
        sm_major: u32,
        sm_minor: u32,
    ) -> Result<LoadedBaseProducerAuthority, InvocationShapeError> {
        if self != &Self::compile_replacement(arena, self.preprocessed_trace_variant)? {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        if sm_minor >= 10 {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        let active_sm = sm_major
            .checked_mul(10)
            .and_then(|major| major.checked_add(sm_minor))
            .ok_or(InvocationShapeError::SizeOverflow)?;
        super::loaded_base_binding::bind(
            self,
            arena,
            prepared,
            device_ordinal,
            sm_major,
            sm_minor,
            active_sm,
        )
    }
}

#[derive(Clone, Debug)]
enum PendingSemanticProducer {
    Recorded(PendingRecordedWitness),
    NativeBlakeGDirect {
        position: ProducerSchedulePosition,
        producer: WitnessProducer,
        contract: super::blake_g_direct_prefix::PendingNativeBlakeGDirectContract,
    },
    NativeEcOp {
        position: ProducerSchedulePosition,
        producer: WitnessProducer,
        contract: ec_op_prefix::PendingNativeEcOpContract,
    },
}

fn validate_direct_blake_g_outputs(
    catalog: &BaseProducerCatalog,
    native: &super::blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
    direct_outputs: &BTreeSet<LogicalBufferId>,
) -> Result<(), InvocationShapeError> {
    if native.invocation.traces.len() != native.authority.trace_column_words().len()
        || native.invocation.traces.iter().any(|binding| {
            let Ok(value) = catalog.value(binding.value.value) else {
                return true;
            };
            value.component != Some("blake_g")
                || value.part != Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main)
                || !direct_outputs.contains(&value.logical)
        })
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn direct_base_evaluations(
    catalog: &BaseProducerCatalog,
    base: &PlannedCommitment,
) -> Result<BTreeSet<LogicalBufferId>, InvocationShapeError> {
    let mut values = BTreeSet::new();
    for &source in base.grouped_column_sources.iter().flatten() {
        let CommitmentColumnSource::Trace {
            component,
            part,
            purpose: BufferPurpose::BaseCoefficients,
            ordinal,
        } = source
        else {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        };
        let mut matches = catalog.values.iter().filter(|value| {
            value.component == Some(component)
                && value.part == Some(part)
                && value.purpose == BufferPurpose::BaseTrace
                && value.ordinal == ordinal
        });
        let value = matches
            .next()
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        if matches.next().is_some() || !values.insert(value.logical) {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
    }
    Ok(values)
}

fn validate_direct_recorded_outputs(
    catalog: &BaseProducerCatalog,
    producer: WitnessProducer,
    expected_trace_outputs: usize,
    produced: &[ArenaCatalogValueId],
    direct_outputs: &BTreeSet<LogicalBufferId>,
) -> Result<(), InvocationShapeError> {
    let mut trace_outputs = 0usize;
    for &id in produced {
        let value = catalog.value(id)?;
        if value.purpose != BufferPurpose::BaseTrace {
            continue;
        }
        trace_outputs = trace_outputs
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        if value.component != Some(producer.component)
            || value.part != producer.part
            || !direct_outputs.contains(&value.logical)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
    }
    if trace_outputs != expected_trace_outputs {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn validate_direct_native_outputs(
    catalog: &BaseProducerCatalog,
    native: &ec_op_prefix::LoweredNativeEcOpContract,
    direct_outputs: &BTreeSet<LogicalBufferId>,
) -> Result<(), InvocationShapeError> {
    if native.invocation.trace_columns.len()
        != native.authority.requirements().trace_column_words.len()
        || native.invocation.trace_columns.iter().any(|binding| {
            let Ok(value) = catalog.value(binding.value.value) else {
                return true;
            };
            !direct_outputs.contains(&value.logical)
        })
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct PendingRecordedWitness {
    position: ProducerSchedulePosition,
    producer: WitnessProducer,
    produced: Vec<ArenaCatalogValueId>,
    expected_trace_outputs: usize,
    source: RecordedWitnessInvocationShape,
}

#[derive(Clone, Debug)]
enum PendingBaseProducer {
    Recorded(PendingRecordedWitness),
    NativeEcOp {
        position: ProducerSchedulePosition,
        producer: WitnessProducer,
        linked: NativeEcOpLinkedModuleAuthority,
        contract: ec_op_prefix::PendingNativeEcOpContract,
    },
}

pub(super) fn map_scheduled_base_producers(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
) -> Result<BaseProducerBindingFrontier, InvocationShapeError> {
    map_scheduled_base_producers_using(
        catalog,
        arena,
        schedule,
        NativeEcOpLinkedModuleAuthority::bind_linked,
    )
}

#[cfg(test)]
pub(super) fn map_scheduled_base_producers_with_native_authority(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
    bind_native: impl FnMut(
        &EcOpCompositeContract,
    )
        -> Result<Option<NativeEcOpLinkedModuleAuthority>, InvocationShapeError>,
) -> Result<BaseProducerBindingFrontier, InvocationShapeError> {
    map_scheduled_base_producers_using(catalog, arena, schedule, bind_native)
}

fn map_scheduled_base_producers_using(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
    mut bind_native: impl FnMut(
        &EcOpCompositeContract,
    )
        -> Result<Option<NativeEcOpLinkedModuleAuthority>, InvocationShapeError>,
) -> Result<BaseProducerBindingFrontier, InvocationShapeError> {
    let scheduled = scheduled_producers(schedule)?;
    let mut pending = Vec::new();
    let mut native_ec_op_seen = false;
    let mut missing = None;
    for &(position, producer) in &scheduled {
        let missing_kind = match producer.kind {
            WitnessProducerKind::Recorded => {
                let planned = planned_recorded_component(arena, producer)?;
                match derive_invocation(catalog, arena, planned) {
                    Ok(source) => {
                        pending.push(PendingBaseProducer::Recorded(PendingRecordedWitness {
                            position,
                            producer,
                            produced: produced_values(catalog, planned)?,
                            expected_trace_outputs: usize::try_from(planned.program.n_cols)
                                .map_err(|_| InvocationShapeError::SizeOverflow)?,
                            source,
                        }));
                        None
                    }
                    Err(InvocationShapeError::MultiplicityNeedsSemanticVersions) => {
                        Some(MissingProducerAuthorityKind::MultiplicityTransition)
                    }
                    Err(InvocationShapeError::InvalidProgramRole) => {
                        return Err(InvocationShapeError::ScheduledProducerInvalidProgram(
                            producer,
                        ))
                    }
                    Err(error) => return Err(error),
                }
            }
            WitnessProducerKind::BlakeGFused => {
                Some(MissingProducerAuthorityKind::BlakeGFusedComposite)
            }
            WitnessProducerKind::BlakeGDirect => {
                Some(MissingProducerAuthorityKind::BlakeGDirectComposite)
            }
            WitnessProducerKind::NativeEcOp => {
                if native_ec_op_seen {
                    return Err(InvocationShapeError::InvalidNativeEcOpBinding);
                }
                native_ec_op_seen = true;
                let contract = ec_op_prefix::prepare(catalog, arena)?;
                match bind_native(contract.authority())? {
                    Some(linked) => {
                        pending.push(PendingBaseProducer::NativeEcOp {
                            position,
                            producer,
                            linked,
                            contract,
                        });
                        None
                    }
                    None => Some(MissingProducerAuthorityKind::NativeEcOpStaticModuleBuildIdentity),
                }
            }
        };
        if let Some(missing_kind) = missing_kind {
            missing = Some(MissingProducerAuthority {
                position,
                producer,
                missing: missing_kind,
            });
            break;
        }
    }

    validate_bound_base_outputs(catalog, schedule, &pending)?;
    let mut semantic_values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())?;
    let mut bound = Vec::with_capacity(pending.len());
    for pending in pending {
        match pending {
            PendingBaseProducer::Recorded(producer) => {
                semantic_values
                    .extend_ordered(invocation_catalog_order(&producer.source.source_arguments))?;
                let (invocation, effect) =
                    adapter::compile(&producer.source.source_arguments, &mut semantic_values)
                        .map_err(|error| match error {
                            InvocationShapeError::InvalidProgramRole
                            | InvocationShapeError::InvalidAdapterEffect => {
                                InvocationShapeError::ScheduledProducerInvalidEffect(
                                    producer.producer,
                                )
                            }
                            error => error,
                        })?;
                bound.push(LoweredBaseProducer::Recorded(
                    LoweredRecordedWitnessProducer {
                        position: producer.position,
                        producer: producer.producer,
                        produced: producer.produced,
                        source: producer.source,
                        invocation,
                        effect,
                    },
                ));
            }
            PendingBaseProducer::NativeEcOp {
                position,
                producer,
                linked,
                contract,
            } => {
                let contract = ec_op_prefix::lower(contract, &mut semantic_values)?;
                validate_native_ec_op_base_outputs(catalog, schedule, &contract)?;
                let execution = linked.bind_lowered(
                    &contract.authority,
                    &contract.invocation,
                    &contract.effect,
                )?;
                bound.push(LoweredBaseProducer::NativeEcOp(LoweredNativeEcOpProducer {
                    position,
                    producer,
                    execution,
                    contract,
                }));
            }
        }
    }
    let mut interpolation_values = Vec::new();
    append_interpolation_catalog_order(catalog, schedule, &mut interpolation_values)?;
    semantic_values.extend_ordered(interpolation_values)?;
    let base_interpolation = lower_base_interpolation(catalog, schedule, &semantic_values)?;
    Ok(BaseProducerBindingFrontier {
        scheduled_producers: scheduled.len(),
        bound,
        missing,
        semantic_values,
        base_interpolation,
    })
}

fn validate_native_ec_op_base_outputs(
    catalog: &BaseProducerCatalog,
    schedule: &BaseProducerSchedule,
    native: &ec_op_prefix::LoweredNativeEcOpContract,
) -> Result<(), InvocationShapeError> {
    let scheduled = schedule
        .interpolation()
        .ok_or(InvocationShapeError::FrontierDidNotAdvance)?
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .map(|column| column.evaluations.logical)
        .collect::<BTreeSet<_>>();
    if native.invocation.trace_columns.len()
        != native.authority.requirements().trace_column_words.len()
        || native.invocation.trace_columns.iter().any(|binding| {
            let Ok(value) = catalog.value(binding.value.value) else {
                return true;
            };
            value.id != binding.value.value || !scheduled.contains(&value.logical)
        })
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(())
}

fn scheduled_producers(
    schedule: &BaseProducerSchedule,
) -> Result<Vec<(ProducerSchedulePosition, WitnessProducer)>, InvocationShapeError> {
    let mut producers = Vec::new();
    for (level, lanes) in schedule.witness_levels().iter().enumerate() {
        for (lane, &producer) in lanes.iter().enumerate() {
            producers.push((
                ProducerSchedulePosition {
                    level: u32::try_from(level).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    lane: u32::try_from(lane).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    ordinal: u32::try_from(producers.len())
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                },
                producer,
            ));
        }
    }
    Ok(producers)
}

fn validate_bound_base_outputs(
    catalog: &BaseProducerCatalog,
    schedule: &BaseProducerSchedule,
    bound: &[PendingBaseProducer],
) -> Result<(), InvocationShapeError> {
    let interpolation = schedule
        .interpolation()
        .ok_or(InvocationShapeError::FrontierDidNotAdvance)?;
    let scheduled_evaluations = interpolation
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .map(|column| column.evaluations.logical)
        .collect::<BTreeSet<_>>();
    for producer in bound {
        let PendingBaseProducer::Recorded(producer) = producer else {
            continue;
        };
        let mut trace_outputs = 0usize;
        for &produced in &producer.produced {
            let value = catalog.value(produced)?;
            if value.purpose != BufferPurpose::BaseTrace {
                continue;
            }
            trace_outputs = trace_outputs
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            if !scheduled_evaluations.contains(&value.logical)
                || value.component != Some(producer.producer.component)
                || value.part != producer.producer.part
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
        }
        if producer.producer.kind != WitnessProducerKind::Recorded
            || producer.source.program_identity == [0; 32]
            || producer.source.abi_schema_identity == [0; 32]
            || trace_outputs != producer.expected_trace_outputs
        {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    Ok(())
}
