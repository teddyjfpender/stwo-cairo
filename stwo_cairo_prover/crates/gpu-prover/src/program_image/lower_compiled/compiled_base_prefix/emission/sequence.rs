use std::collections::BTreeSet;

use stwo_backend_cuda::ExecutionTablesStage;

use super::super::super::{
    adapter, witness_casm_input, witness_input_gather, witness_input_seed_compact,
    InvocationShapeError,
};
use super::super::{BaseProducerAuthority, SemanticBaseProducer};
use super::StaticWrapperRequest;
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::EffectContract;
use crate::resident_runtime::producer_schedule::WitnessProducerKind;

#[derive(Clone, Copy)]
pub(super) enum BaseEvent<'a> {
    Recorded(&'a super::super::super::producer_prefix::LoweredRecordedWitnessProducer),
    Static(StaticWrapperRequest<'a>),
}

impl<'a> BaseEvent<'a> {
    pub(super) fn effect(self) -> Result<&'a EffectContract, InvocationShapeError> {
        match self {
            Self::Recorded(recorded) => Ok(&recorded.effect),
            Self::Static(request) => request.effect(),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct WitnessGroup<'a> {
    pub(super) producer: BaseEvent<'a>,
    pub(super) feed: Option<BaseEvent<'a>>,
}

pub(super) struct EventSequence<'a> {
    pub(super) prelude: Vec<BaseEvent<'a>>,
    pub(super) witnesses: Vec<WitnessGroup<'a>>,
}

/// Retained semantic receipts for every generated witness preproducer.
///
/// These vectors are derived from the arena and canonical producer schedule;
/// they are not a second scheduling authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) struct CausalWitnessSetup {
    pub(in crate::program_image::lower_compiled) gathers:
        Vec<witness_input_gather::LoweredWitnessInputGather>,
    pub(in crate::program_image::lower_compiled) seed_compact:
        Vec<witness_input_seed_compact::LoweredWitnessInputSetup>,
    pub(in crate::program_image::lower_compiled) casm:
        Vec<witness_casm_input::LoweredWitnessCasmInput>,
    expected_ordinals: BTreeSet<u32>,
}

#[derive(Clone, Copy)]
pub(super) enum CausalSetupEvent<'a> {
    Static(StaticWrapperRequest<'a>),
    Casm(&'a witness_casm_input::LoweredWitnessCasmInput),
}

#[derive(Clone, Copy)]
pub(super) enum CausalBaseEvent<'a> {
    Recorded(&'a super::super::super::producer_prefix::LoweredRecordedWitnessProducer),
    Static(StaticWrapperRequest<'a>),
    HostIngress(&'a witness_casm_input::LoweredWitnessCasmInput),
}

#[derive(Clone, Copy)]
pub(super) struct CausalWitnessGroup<'a> {
    pub(super) setup: Option<CausalSetupEvent<'a>>,
    pub(super) producer: BaseEvent<'a>,
    pub(super) feed: Option<BaseEvent<'a>>,
}

pub(super) struct CausalEventSequence<'a> {
    pub(super) prelude: Vec<BaseEvent<'a>>,
    pub(super) witnesses: Vec<CausalWitnessGroup<'a>>,
}

impl CausalWitnessSetup {
    /// Lower all setup families transactionally against the one proof-wide
    /// semantic value map.
    pub(super) fn lower(
        arena: &ProofArenaPlan,
        values: &mut adapter::SemanticValueMap,
        authority: &BaseProducerAuthority,
    ) -> Result<Self, InvocationShapeError> {
        let mut next_values = values.clone();
        let setup = Self {
            gathers: witness_input_gather::lower_stage(arena, &mut next_values)?,
            seed_compact: witness_input_seed_compact::lower_stage(arena, &mut next_values)?,
            casm: witness_casm_input::lower_stage(arena, &mut next_values)?,
            expected_ordinals: planned_setup_ordinals(arena, authority)?,
        };
        setup.validate_positions(authority)?;
        *values = next_values;
        Ok(setup)
    }

    /// Re-lower from the completed allocator and require byte-exact retained
    /// receipts and idempotent value allocation.
    pub(super) fn validate(
        &self,
        arena: &ProofArenaPlan,
        values: &adapter::SemanticValueMap,
        authority: &BaseProducerAuthority,
    ) -> Result<(), InvocationShapeError> {
        let mut exact_values = values.clone();
        let exact = Self::lower(arena, &mut exact_values, authority)?;
        if exact == *self && exact_values == *values {
            Ok(())
        } else {
            Err(InvocationShapeError::InvalidScheduledProducerBinding)
        }
    }

    fn at(&self, ordinal: u32) -> Result<Option<CausalSetupEvent<'_>>, ()> {
        let mut matches = self
            .gathers
            .iter()
            .filter(|setup| setup.position.ordinal == ordinal)
            .map(|setup| CausalSetupEvent::Static(StaticWrapperRequest::WitnessInputGather(setup)))
            .chain(
                self.seed_compact
                    .iter()
                    .filter(|setup| setup.position().ordinal == ordinal)
                    .map(|setup| match setup {
                        witness_input_seed_compact::LoweredWitnessInputSetup::Seed(seed) => {
                            CausalSetupEvent::Static(StaticWrapperRequest::WitnessInputSeed(seed))
                        }
                        witness_input_seed_compact::LoweredWitnessInputSetup::Compact(compact) => {
                            CausalSetupEvent::Static(StaticWrapperRequest::WitnessInputCompact(
                                compact,
                            ))
                        }
                    }),
            )
            .chain(
                self.casm
                    .iter()
                    .filter(|setup| setup.position.ordinal == ordinal)
                    .map(CausalSetupEvent::Casm),
            );
        let first = matches.next();
        if matches.next().is_some() {
            Err(())
        } else {
            Ok(first)
        }
    }

    fn validate_positions(
        &self,
        authority: &BaseProducerAuthority,
    ) -> Result<(), InvocationShapeError> {
        let mut seen = BTreeSet::new();
        for (position, component, part) in self
            .gathers
            .iter()
            .map(|setup| (setup.position, setup.component, setup.part))
            .chain(
                self.seed_compact
                    .iter()
                    .map(|setup| (setup.position(), setup.component(), setup.part())),
            )
            .chain(
                self.casm
                    .iter()
                    .map(|setup| (setup.position, setup.component, setup.part)),
            )
        {
            if !seen.insert(position.ordinal)
                || !matches!(
                    authority.producers.get(position.ordinal as usize),
                    Some(producer)
                        if producer.position() == position
                            && producer.producer().component == component
                            && producer.producer().part == Some(part)
                            && producer.producer().kind == WitnessProducerKind::Recorded
                )
            {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
        }
        if seen.len() != self.gathers.len() + self.seed_compact.len() + self.casm.len() {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        if seen != self.expected_ordinals {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
        Ok(())
    }
}

fn planned_setup_ordinals(
    arena: &ProofArenaPlan,
    authority: &BaseProducerAuthority,
) -> Result<BTreeSet<u32>, InvocationShapeError> {
    let mut expected = BTreeSet::new();
    for component in arena.witness().components.iter().filter(|component| {
        component.input_gather.is_some()
            || component.input_seed.is_some()
            || component.input_compact.is_some()
            || component.input_casm.is_some()
    }) {
        let mut matches = authority.producers.iter().filter(|producer| {
            producer.producer().component == component.component
                && producer.producer().part == Some(component.part)
        });
        let producer = matches
            .next()
            .filter(|producer| producer.producer().kind == WitnessProducerKind::Recorded)
            .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
        if matches.next().is_some() || !expected.insert(producer.position().ordinal) {
            return Err(InvocationShapeError::InvalidScheduledProducerBinding);
        }
    }
    Ok(expected)
}

pub(super) fn sequence(authority: &BaseProducerAuthority) -> Result<EventSequence<'_>, ()> {
    if authority.multiplicity.after_producer.len() != authority.producers.len() {
        return Err(());
    }
    let mut prelude = Vec::with_capacity(4);
    if let Some(lowered) = &authority.execution_tables {
        if lowered.stages[0].stage != ExecutionTablesStage::Big
            || lowered.stages[1].stage != ExecutionTablesStage::Small
        {
            return Err(());
        }
        for stage in lowered.stages.iter().map(|stage| stage.stage) {
            prelude.push(BaseEvent::Static(StaticWrapperRequest::ExecutionTable {
                lowered,
                stage,
            }));
        }
    }
    prelude.push(BaseEvent::Static(StaticWrapperRequest::MultiplicityClear(
        &authority.multiplicity.clear,
    )));
    if let Some(seed) = &authority.multiplicity.public_memory_seed {
        prelude.push(BaseEvent::Static(StaticWrapperRequest::MultiplicityFeed(
            seed,
        )));
    }

    let witnesses = authority
        .producers
        .iter()
        .zip(&authority.multiplicity.after_producer)
        .enumerate()
        .map(|(index, (producer, feed))| {
            if producer.position().ordinal as usize != index {
                return Err(());
            }
            let event = match producer {
                SemanticBaseProducer::Recorded(recorded) if feed.is_some() => {
                    BaseEvent::Recorded(recorded)
                }
                SemanticBaseProducer::NativeBlakeGDirect { contract, .. } if feed.is_none() => {
                    BaseEvent::Static(StaticWrapperRequest::NativeBlakeGDirect(contract))
                }
                SemanticBaseProducer::NativeEcOp { contract, .. } if feed.is_none() => {
                    BaseEvent::Static(StaticWrapperRequest::NativeEcOp(contract))
                }
                _ => return Err(()),
            };
            Ok(WitnessGroup {
                producer: event,
                feed: feed
                    .as_ref()
                    .map(|feed| BaseEvent::Static(StaticWrapperRequest::MultiplicityFeed(feed))),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EventSequence { prelude, witnesses })
}

pub(super) fn causal_sequence<'a>(
    authority: &'a BaseProducerAuthority,
    setup: &'a CausalWitnessSetup,
) -> Result<CausalEventSequence<'a>, ()> {
    setup.validate_positions(authority).map_err(|_| ())?;
    let writer = sequence(authority)?;
    let witnesses = writer
        .witnesses
        .into_iter()
        .enumerate()
        .map(|(ordinal, group)| {
            Ok(CausalWitnessGroup {
                setup: setup.at(u32::try_from(ordinal).map_err(|_| ())?)?,
                producer: group.producer,
                feed: group.feed,
            })
        })
        .collect::<Result<Vec<_>, ()>>()?;
    Ok(CausalEventSequence {
        prelude: writer.prelude,
        witnesses,
    })
}

pub(super) fn causal_events<'a>(
    authority: &'a BaseProducerAuthority,
    setup: &'a CausalWitnessSetup,
    next_producer: usize,
) -> Result<Vec<CausalBaseEvent<'a>>, ()> {
    let sequence = causal_sequence(authority, setup)?;
    if next_producer > sequence.witnesses.len() {
        return Err(());
    }
    let mut events = Vec::new();
    events.extend(sequence.prelude.into_iter().map(causal_base_event));
    for group in &sequence.witnesses[..next_producer] {
        match group.setup {
            Some(CausalSetupEvent::Static(request)) => {
                events.push(CausalBaseEvent::Static(request));
            }
            Some(CausalSetupEvent::Casm(lowered)) => {
                events.push(CausalBaseEvent::HostIngress(lowered));
                events.push(CausalBaseEvent::Static(
                    StaticWrapperRequest::WitnessCasmScatter(lowered),
                ));
            }
            None => {}
        }
        events.push(causal_base_event(group.producer));
        if let Some(feed) = group.feed {
            events.push(causal_base_event(feed));
        }
    }
    Ok(events)
}

const fn causal_base_event(event: BaseEvent<'_>) -> CausalBaseEvent<'_> {
    match event {
        BaseEvent::Recorded(recorded) => CausalBaseEvent::Recorded(recorded),
        BaseEvent::Static(request) => CausalBaseEvent::Static(request),
    }
}

pub(in super::super::super) fn ordered_effects(
    authority: &BaseProducerAuthority,
) -> Result<Vec<&EffectContract>, ()> {
    let sequence = sequence(authority)?;
    let mut effects = Vec::new();
    for event in sequence.prelude {
        effects.push(event.effect().map_err(|_| ())?);
    }
    for group in sequence.witnesses {
        effects.push(group.producer.effect().map_err(|_| ())?);
        if let Some(feed) = group.feed {
            effects.push(feed.effect().map_err(|_| ())?);
        }
    }
    Ok(effects)
}
