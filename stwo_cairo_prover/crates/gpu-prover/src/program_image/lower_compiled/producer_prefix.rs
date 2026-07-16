//! Contiguous schedule-order authority for Base witness producers.
//!
//! A producer enters `bound` only after its typed source ABI and ordinary value
//! effect compile. A stateful recorded producer additionally carries its exact
//! address-free resource/relocation recipe, but its value effect deliberately
//! omits module globals until a loaded-module publication receipt can bind real
//! addresses and canonical content. This is a source/relocation migration
//! frontier, not executable authority or permission to promote the arena
//! inventory into a production `CompiledProof`.

use std::collections::BTreeSet;

use stwo_backend_cuda::EcOpCompositeContract;

use super::ec_op_execution_authority::{
    NativeEcOpCompositeExecutionAuthority, NativeEcOpLinkedModuleAuthority,
};
use super::schedule_prefix::{
    append_interpolation_catalog_order, invocation_catalog_order, lower_base_interpolation,
    LoweredBaseInterpolationBatch,
};
use super::*;
use crate::arena_plan::{BufferPurpose, ProofArenaPlan};
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
    image: &ArenaProgramInventory,
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
) -> Result<BaseProducerBindingFrontier, InvocationShapeError> {
    map_scheduled_base_producers_using(
        image,
        arena,
        schedule,
        NativeEcOpLinkedModuleAuthority::bind_linked,
    )
}

#[cfg(test)]
pub(super) fn map_scheduled_base_producers_with_native_authority(
    image: &ArenaProgramInventory,
    arena: &ProofArenaPlan,
    schedule: &BaseProducerSchedule,
    bind_native: impl FnMut(
        &EcOpCompositeContract,
    )
        -> Result<Option<NativeEcOpLinkedModuleAuthority>, InvocationShapeError>,
) -> Result<BaseProducerBindingFrontier, InvocationShapeError> {
    map_scheduled_base_producers_using(image, arena, schedule, bind_native)
}

fn map_scheduled_base_producers_using(
    image: &ArenaProgramInventory,
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
                match derive_invocation(image, arena, planned) {
                    Ok(source) => {
                        pending.push(PendingBaseProducer::Recorded(PendingRecordedWitness {
                            position,
                            producer,
                            produced: produced_values(image, arena, planned)?,
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
                let contract = ec_op_prefix::prepare(image, arena)?;
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

    validate_bound_base_outputs(image, schedule, &pending)?;
    let mut semantic_values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())?;
    let mut bound = Vec::with_capacity(pending.len());
    for pending in pending {
        match pending {
            PendingBaseProducer::Recorded(producer) => {
                semantic_values
                    .extend_ordered(invocation_catalog_order(&producer.source.source_arguments))?;
                let (invocation, effect) = adapter::compile(
                    &producer.source.source_arguments,
                    &semantic_values,
                )
                .map_err(|error| match error {
                    InvocationShapeError::InvalidProgramRole
                    | InvocationShapeError::InvalidAdapterEffect => {
                        InvocationShapeError::ScheduledProducerInvalidEffect(producer.producer)
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
                validate_native_ec_op_base_outputs(image, schedule, &contract)?;
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
    append_interpolation_catalog_order(image, schedule, &mut interpolation_values)?;
    semantic_values.extend_ordered(interpolation_values)?;
    let base_interpolation = lower_base_interpolation(image, schedule, &semantic_values)?;
    Ok(BaseProducerBindingFrontier {
        scheduled_producers: scheduled.len(),
        bound,
        missing,
        semantic_values,
        base_interpolation,
    })
}

fn validate_native_ec_op_base_outputs(
    image: &ArenaProgramInventory,
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
            let value = &image.values[binding.value.value.0 as usize];
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
    image: &ArenaProgramInventory,
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
            let value = image
                .values
                .get(produced.0 as usize)
                .filter(|value| value.id == produced)
                .ok_or(InvocationShapeError::InvalidCatalogRange(produced))?;
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
