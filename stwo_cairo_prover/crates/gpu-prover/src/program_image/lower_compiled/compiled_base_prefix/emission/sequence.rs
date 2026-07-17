use stwo_backend_cuda::ExecutionTablesStage;

use super::super::super::InvocationShapeError;
use super::super::{BaseProducerAuthority, SemanticBaseProducer};
use super::StaticWrapperRequest;
use crate::compiled_proof::{AotInvocation, EffectContract};

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

    pub(super) fn invocation(self) -> Result<AotInvocation, InvocationShapeError> {
        match self {
            Self::Recorded(recorded) => Ok(recorded.invocation.clone()),
            Self::Static(request) => request.invocation(),
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
