//! Typed authority for the producer prefix before the Base commitment.
//!
//! Runtime lane packing and migration lowering consume this one schedule. The
//! arena catalog and buffer lifetimes never define execution order.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    ArenaSlotId, InterpolationAuthorityError, InterpolationBatchAuthority, InterpolationLaunchMode,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use crate::arena_plan::{
    BlakeGWitnessContract, BufferPurpose, CommitmentColumnSource, CommitmentTreeId,
    LogicalBufferId, ProofArenaPlan,
};
use crate::schedule_table::CAIRO_SCHEDULE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WitnessProducerKind {
    Recorded,
    BlakeGFused,
    BlakeGDirect,
    NativeEcOp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WitnessProducer {
    pub(crate) component: &'static str,
    pub(crate) part: Option<TracePartId>,
    pub(crate) kind: WitnessProducerKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ScheduledValue {
    pub(crate) logical: LogicalBufferId,
    pub(crate) physical: ArenaSlotId,
    pub(crate) component: &'static str,
    pub(crate) part: TracePartId,
    pub(crate) purpose: BufferPurpose,
    pub(crate) ordinal: u32,
    pub(crate) words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ScheduledGlobalValue {
    pub(crate) logical: LogicalBufferId,
    pub(crate) physical: ArenaSlotId,
    pub(crate) purpose: BufferPurpose,
    pub(crate) ordinal: u32,
    pub(crate) words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaseInterpolationColumn {
    pub(crate) evaluations: ScheduledValue,
    pub(crate) coefficients: ScheduledValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaseInterpolationBatch {
    pub(crate) log_size: u32,
    pub(crate) authority: InterpolationBatchAuthority,
    pub(crate) input_pointers: ScheduledGlobalValue,
    pub(crate) output_pointers: ScheduledGlobalValue,
    pub(crate) columns: Vec<BaseInterpolationColumn>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaseInterpolationSchedule {
    pub(crate) mode: InterpolationLaunchMode,
    pub(crate) inverse_twiddles: ScheduledGlobalValue,
    pub(crate) batches: Vec<BaseInterpolationBatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BaseProducerStep {
    ExecutionTables,
    MultiplicityClear,
    PublicMemorySeed,
    WitnessDag { levels: u32, tasks: u32 },
    MemoryBaseTrace,
    FixedTables { count: u32 },
    BaseInterpolation { batches: u32, columns: u32 },
}

impl BaseProducerStep {
    pub(crate) fn witness(levels: usize, tasks: usize) -> Result<Self, ProducerScheduleError> {
        Ok(Self::WitnessDag {
            levels: as_u32(levels)?,
            tasks: as_u32(tasks)?,
        })
    }

    pub(crate) fn fixed_tables(count: usize) -> Result<Self, ProducerScheduleError> {
        Ok(Self::FixedTables {
            count: as_u32(count)?,
        })
    }

    pub(crate) fn interpolation(
        batches: usize,
        columns: usize,
    ) -> Result<Self, ProducerScheduleError> {
        Ok(Self::BaseInterpolation {
            batches: as_u32(batches)?,
            columns: as_u32(columns)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BaseProducerSchedule {
    steps: Vec<BaseProducerStep>,
    witness_levels: Vec<Vec<WitnessProducer>>,
    interpolation: Option<BaseInterpolationSchedule>,
}

impl BaseProducerSchedule {
    pub(crate) fn compile(arena: &ProofArenaPlan) -> Result<Self, ProducerScheduleError> {
        let witness_levels = witness_levels(arena)?;
        let interpolation = base_interpolation(arena)?;
        let mut steps = Vec::new();
        if arena.execution_tables().is_some() {
            steps.push(BaseProducerStep::ExecutionTables);
        }
        if let Some(multiplicity) = arena.multiplicity() {
            steps.push(BaseProducerStep::MultiplicityClear);
            if multiplicity.public_memory_seed.is_some() {
                steps.push(BaseProducerStep::PublicMemorySeed);
            }
        }
        if !witness_levels.is_empty() {
            steps.push(BaseProducerStep::witness(
                witness_levels.len(),
                witness_levels.iter().map(Vec::len).sum(),
            )?);
        }
        if let Some(multiplicity) = arena.multiplicity() {
            if multiplicity.memory_traces.is_some() {
                steps.push(BaseProducerStep::MemoryBaseTrace);
            }
            if !multiplicity.fixed_tables.is_empty() {
                steps.push(BaseProducerStep::fixed_tables(
                    multiplicity.fixed_tables.len(),
                )?);
            }
        }
        if let Some(interpolation) = &interpolation {
            steps.push(BaseProducerStep::interpolation(
                interpolation.batches.len(),
                interpolation
                    .batches
                    .iter()
                    .map(|batch| batch.columns.len())
                    .sum(),
            )?);
        }
        Ok(Self {
            steps,
            witness_levels,
            interpolation,
        })
    }

    pub(crate) fn steps(&self) -> &[BaseProducerStep] {
        &self.steps
    }

    pub(crate) fn witness_levels(&self) -> &[Vec<WitnessProducer>] {
        &self.witness_levels
    }

    pub(crate) fn interpolation(&self) -> Option<&BaseInterpolationSchedule> {
        self.interpolation.as_ref()
    }

    pub(crate) fn validate_runtime_steps(
        &self,
        actual: &[BaseProducerStep],
    ) -> Result<(), ProducerScheduleError> {
        if actual == self.steps {
            Ok(())
        } else {
            Err(ProducerScheduleError::RuntimeOrderMismatch)
        }
    }

    pub(crate) fn cursor(&self) -> BaseProducerCursor<'_> {
        BaseProducerCursor {
            expected: &self.steps,
            next: 0,
        }
    }
}

pub(crate) struct BaseProducerCursor<'a> {
    expected: &'a [BaseProducerStep],
    next: usize,
}

impl BaseProducerCursor<'_> {
    pub(crate) fn admit(&mut self, actual: BaseProducerStep) -> Result<(), ProducerScheduleError> {
        if self.expected.get(self.next) != Some(&actual) {
            return Err(ProducerScheduleError::RuntimeOrderMismatch);
        }
        self.next += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<(), ProducerScheduleError> {
        if self.next == self.expected.len() {
            Ok(())
        } else {
            Err(ProducerScheduleError::RuntimeOrderMismatch)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProducerScheduleError {
    InvalidWitnessDag,
    DuplicateWitness(&'static str),
    MissingWitness(&'static str),
    MissingBaseCommitment,
    DirectBaseHasDetachedInterpolation,
    MissingBaseInterpolation,
    InvalidBaseInterpolationSource(CommitmentColumnSource),
    MissingBaseValue {
        component: &'static str,
        part: TracePartId,
        purpose: BufferPurpose,
        ordinal: u32,
    },
    BaseValueShapeMismatch(CommitmentColumnSource),
    DuplicateBaseInterpolationValue(LogicalBufferId),
    MissingGlobalValueByOrdinal {
        purpose: BufferPurpose,
        ordinal: u32,
    },
    MissingGlobalValue {
        physical: ArenaSlotId,
        purpose: BufferPurpose,
    },
    AmbiguousGlobalValue {
        physical: ArenaSlotId,
        purpose: BufferPurpose,
    },
    GlobalValueShapeMismatch(LogicalBufferId),
    InverseTwiddlesTooSmall {
        required_words: usize,
        actual_words: usize,
    },
    InterpolationAuthority(InterpolationAuthorityError),
    RuntimeOrderMismatch,
    SizeOverflow,
}

impl core::fmt::Display for ProducerScheduleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid Base producer schedule: {self:?}")
    }
}

impl std::error::Error for ProducerScheduleError {}

fn witness_levels(
    arena: &ProofArenaPlan,
) -> Result<Vec<Vec<WitnessProducer>>, ProducerScheduleError> {
    let mut planned = BTreeMap::new();
    for witness in &arena.witness().components {
        let kind = match &witness.blake_g_contract {
            BlakeGWitnessContract::Recorded => WitnessProducerKind::Recorded,
            BlakeGWitnessContract::Fused => WitnessProducerKind::BlakeGFused,
            BlakeGWitnessContract::Direct(_) => WitnessProducerKind::BlakeGDirect,
        };
        if planned
            .insert(
                witness.component,
                WitnessProducer {
                    component: witness.component,
                    part: Some(witness.part),
                    kind,
                },
            )
            .is_some()
        {
            return Err(ProducerScheduleError::DuplicateWitness(witness.component));
        }
    }
    if arena.ec_op().is_some()
        && planned
            .insert(
                "ec_op_builtin",
                WitnessProducer {
                    component: "ec_op_builtin",
                    part: None,
                    kind: WitnessProducerKind::NativeEcOp,
                },
            )
            .is_some()
    {
        return Err(ProducerScheduleError::DuplicateWitness("ec_op_builtin"));
    }

    let mut levels = Vec::new();
    for level in CAIRO_SCHEDULE
        .levels()
        .map_err(|_| ProducerScheduleError::InvalidWitnessDag)?
    {
        let level = level
            .into_iter()
            .filter_map(|component| planned.remove(component))
            .collect::<Vec<_>>();
        if !level.is_empty() {
            levels.push(level);
        }
    }
    if let Some((&component, _)) = planned.first_key_value() {
        return Err(ProducerScheduleError::MissingWitness(component));
    }
    Ok(levels)
}

fn base_interpolation(
    arena: &ProofArenaPlan,
) -> Result<Option<BaseInterpolationSchedule>, ProducerScheduleError> {
    let base = arena
        .commitment(CommitmentTreeId::Base)
        .ok_or(ProducerScheduleError::MissingBaseCommitment)?;
    if base.direct_retained_b2n_program.is_some() {
        if !base.interpolation_batches.is_empty() {
            return Err(ProducerScheduleError::DirectBaseHasDetachedInterpolation);
        }
        return Ok(None);
    }
    if base.interpolation_batches.is_empty() {
        return Err(ProducerScheduleError::MissingBaseInterpolation);
    }

    let inverse_twiddles = scheduled_global(arena, BufferPurpose::InverseTwiddles, 0, None)?;
    let mut seen_evaluations = BTreeSet::new();
    let mut seen_coefficients = BTreeSet::new();
    let batches = base
        .interpolation_batches
        .iter()
        .map(|batch| {
            let words = 1usize
                .checked_shl(batch.log_size)
                .ok_or(ProducerScheduleError::SizeOverflow)?;
            let columns = batch
                .sources
                .iter()
                .map(|&source| {
                    let CommitmentColumnSource::Trace {
                        component,
                        part,
                        purpose: BufferPurpose::BaseCoefficients,
                        ordinal,
                    } = source
                    else {
                        return Err(ProducerScheduleError::InvalidBaseInterpolationSource(
                            source,
                        ));
                    };
                    let evaluations = scheduled_value(
                        arena,
                        component,
                        part,
                        BufferPurpose::BaseTrace,
                        ordinal,
                        words,
                    )?;
                    let coefficients = scheduled_value(
                        arena,
                        component,
                        part,
                        BufferPurpose::BaseCoefficients,
                        ordinal,
                        words,
                    )?;
                    if !seen_evaluations.insert(evaluations.logical) {
                        return Err(ProducerScheduleError::DuplicateBaseInterpolationValue(
                            evaluations.logical,
                        ));
                    }
                    if !seen_coefficients.insert(coefficients.logical) {
                        return Err(ProducerScheduleError::DuplicateBaseInterpolationValue(
                            coefficients.logical,
                        ));
                    }
                    Ok(BaseInterpolationColumn {
                        evaluations,
                        coefficients,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let pointer_words = columns
                .len()
                .checked_mul(
                    core::mem::size_of::<*const u32>().div_ceil(core::mem::size_of::<u32>()),
                )
                .ok_or(ProducerScheduleError::SizeOverflow)?;
            let authority = InterpolationBatchAuthority::compile(
                base.interpolation_mode,
                batch.log_size,
                columns.len(),
            )
            .map_err(ProducerScheduleError::InterpolationAuthority)?;
            let required_twiddle_words = usize::try_from(authority.evaluation_domain_size())
                .map_err(|_| ProducerScheduleError::SizeOverflow)?;
            if inverse_twiddles.words < required_twiddle_words {
                return Err(ProducerScheduleError::InverseTwiddlesTooSmall {
                    required_words: required_twiddle_words,
                    actual_words: inverse_twiddles.words,
                });
            }
            Ok(BaseInterpolationBatch {
                log_size: batch.log_size,
                authority,
                input_pointers: scheduled_global_for_physical(
                    arena,
                    batch.input_pointers,
                    BufferPurpose::InterpolationInputPointers,
                    pointer_words,
                )?,
                output_pointers: scheduled_global_for_physical(
                    arena,
                    batch.output_pointers,
                    BufferPurpose::InterpolationOutputPointers,
                    pointer_words,
                )?,
                columns,
            })
        })
        .collect::<Result<Vec<_>, ProducerScheduleError>>()?;
    Ok(Some(BaseInterpolationSchedule {
        mode: base.interpolation_mode,
        inverse_twiddles,
        batches,
    }))
}

fn scheduled_value(
    arena: &ProofArenaPlan,
    component: &'static str,
    part: TracePartId,
    purpose: BufferPurpose,
    ordinal: u32,
    words: usize,
) -> Result<ScheduledValue, ProducerScheduleError> {
    let (logical, binding) = arena
        .find(Some(component), Some(part), purpose, ordinal)
        .ok_or(ProducerScheduleError::MissingBaseValue {
            component,
            part,
            purpose,
            ordinal,
        })?;
    if logical.len_words != words || binding.len_words != words {
        return Err(ProducerScheduleError::BaseValueShapeMismatch(
            CommitmentColumnSource::Trace {
                component,
                part,
                purpose,
                ordinal,
            },
        ));
    }
    Ok(ScheduledValue {
        logical: logical.id,
        physical: binding.physical,
        component,
        part,
        purpose,
        ordinal,
        words,
    })
}

fn scheduled_global(
    arena: &ProofArenaPlan,
    purpose: BufferPurpose,
    ordinal: u32,
    expected_words: Option<usize>,
) -> Result<ScheduledGlobalValue, ProducerScheduleError> {
    let (logical, binding) = arena.find(None, None, purpose, ordinal).ok_or(
        ProducerScheduleError::MissingGlobalValueByOrdinal { purpose, ordinal },
    )?;
    scheduled_global_from_binding(logical, binding, purpose, expected_words)
}

fn scheduled_global_for_physical(
    arena: &ProofArenaPlan,
    physical: ArenaSlotId,
    purpose: BufferPurpose,
    expected_words: usize,
) -> Result<ScheduledGlobalValue, ProducerScheduleError> {
    let mut matches = arena.logical_buffers().iter().filter_map(|logical| {
        if logical.component.is_some() || logical.part.is_some() || logical.purpose != purpose {
            return None;
        }
        arena
            .binding(logical.id)
            .filter(|binding| binding.physical == physical)
            .map(|binding| (logical, binding))
    });
    let (logical, binding) = matches
        .next()
        .ok_or(ProducerScheduleError::MissingGlobalValue { physical, purpose })?;
    if matches.next().is_some() {
        return Err(ProducerScheduleError::AmbiguousGlobalValue { physical, purpose });
    }
    scheduled_global_from_binding(logical, binding, purpose, Some(expected_words))
}

fn scheduled_global_from_binding(
    logical: &crate::arena_plan::LogicalBuffer,
    binding: crate::arena_plan::ArenaBinding,
    purpose: BufferPurpose,
    expected_words: Option<usize>,
) -> Result<ScheduledGlobalValue, ProducerScheduleError> {
    if logical.purpose != purpose
        || binding.logical != logical.id
        || binding.len_words != logical.len_words
        || expected_words.is_some_and(|words| words != logical.len_words)
    {
        return Err(ProducerScheduleError::GlobalValueShapeMismatch(logical.id));
    }
    Ok(ScheduledGlobalValue {
        logical: logical.id,
        physical: binding.physical,
        purpose,
        ordinal: logical.ordinal,
        words: logical.len_words,
    })
}

fn as_u32(value: usize) -> Result<u32, ProducerScheduleError> {
    u32::try_from(value).map_err(|_| ProducerScheduleError::SizeOverflow)
}
