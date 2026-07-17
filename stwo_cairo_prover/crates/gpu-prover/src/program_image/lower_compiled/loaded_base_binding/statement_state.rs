//! Pure statement-source publication and single-use admission state.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StatementSourceKind {
    ExecutionTables,
    PublicMemorySeed,
    EcOpSegmentStart,
    WitnessInputs,
    StaticTranscriptInputs,
}

impl StatementSourceKind {
    const ALL: [Self; 5] = [
        Self::ExecutionTables,
        Self::PublicMemorySeed,
        Self::EcOpSegmentStart,
        Self::WitnessInputs,
        Self::StaticTranscriptInputs,
    ];

    const fn index(self) -> usize {
        match self {
            Self::ExecutionTables => 0,
            Self::PublicMemorySeed => 1,
            Self::EcOpSegmentStart => 2,
            Self::WitnessInputs => 3,
            Self::StaticTranscriptInputs => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StatementPhase {
    Collecting,
    Ready,
    Consumed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceSlot {
    expected: bool,
    highest_admitted_generation: u64,
    pending: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StatementSourceState {
    slots: [SourceSlot; 5],
    phase: StatementPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StatementSourceStateError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SourceAttemptCounter(u64);

impl SourceAttemptCounter {
    pub(super) fn begin(&mut self) -> Result<u64, StatementSourceStateError> {
        self.0 = self.0.checked_add(1).ok_or(StatementSourceStateError)?;
        Ok(self.0)
    }
}

impl StatementSourceState {
    pub(super) fn new(expected: [bool; 5]) -> Self {
        Self {
            slots: expected.map(|expected| SourceSlot {
                expected,
                highest_admitted_generation: 0,
                pending: None,
            }),
            phase: StatementPhase::Collecting,
        }
    }

    pub(super) fn begin(&mut self) {
        self.clear_pending();
        self.phase = StatementPhase::Collecting;
    }

    pub(super) fn invalidate(&mut self) {
        self.begin();
    }

    pub(super) fn admit(
        &mut self,
        kind: StatementSourceKind,
        generation: u64,
    ) -> Result<(), StatementSourceStateError> {
        let slot = &mut self.slots[kind.index()];
        if self.phase != StatementPhase::Collecting
            || !slot.expected
            || generation == 0
            || generation <= slot.highest_admitted_generation
            || slot.pending.is_some()
        {
            self.invalidate();
            return Err(StatementSourceStateError);
        }
        // Burn every admitted generation, even if another source later fails.
        // Retrying a partial transaction must upload a wholly fresh source set.
        slot.highest_admitted_generation = generation;
        slot.pending = Some(generation);
        Ok(())
    }

    pub(super) fn reserve(
        &mut self,
        kind: StatementSourceKind,
    ) -> Result<(), StatementSourceStateError> {
        let slot = self.slots[kind.index()];
        if self.phase != StatementPhase::Collecting || !slot.expected || slot.pending.is_some() {
            self.invalidate();
            return Err(StatementSourceStateError);
        }
        Ok(())
    }

    pub(super) fn reserve_final(
        &mut self,
        kind: StatementSourceKind,
    ) -> Result<(), StatementSourceStateError> {
        self.reserve(kind)?;
        if StatementSourceKind::ALL.into_iter().any(|candidate| {
            candidate != kind
                && self.slots[candidate.index()].expected
                    != self.slots[candidate.index()].pending.is_some()
        }) {
            self.invalidate();
            return Err(StatementSourceStateError);
        }
        Ok(())
    }

    pub(super) fn validate_before_final(
        &self,
        final_kind: StatementSourceKind,
        generations: [Option<u64>; 5],
    ) -> Result<(), StatementSourceStateError> {
        if self.phase != StatementPhase::Collecting
            || !self.slots[final_kind.index()].expected
            || self.slots[final_kind.index()].pending.is_some()
            || generations[final_kind.index()].is_some()
            || StatementSourceKind::ALL
                .into_iter()
                .filter(|kind| *kind != final_kind)
                .any(|kind| {
                    let slot = self.slots[kind.index()];
                    slot.expected != slot.pending.is_some()
                        || slot.pending != generations[kind.index()]
                })
        {
            return Err(StatementSourceStateError);
        }
        Ok(())
    }

    pub(super) fn publish(&mut self) -> Result<(), StatementSourceStateError> {
        if self.phase != StatementPhase::Collecting || !self.complete() {
            self.invalidate();
            return Err(StatementSourceStateError);
        }
        self.phase = StatementPhase::Ready;
        Ok(())
    }

    pub(super) fn validate_ready(
        &self,
        generations: [Option<u64>; 5],
    ) -> Result<(), StatementSourceStateError> {
        if self.phase != StatementPhase::Ready
            || !self.complete()
            || StatementSourceKind::ALL
                .into_iter()
                .any(|kind| self.slots[kind.index()].pending != generations[kind.index()])
        {
            return Err(StatementSourceStateError);
        }
        Ok(())
    }

    pub(super) fn consume(
        &mut self,
        generations: [Option<u64>; 5],
    ) -> Result<(), StatementSourceStateError> {
        if self.validate_ready(generations).is_err() {
            self.invalidate();
            return Err(StatementSourceStateError);
        }
        self.clear_pending();
        self.phase = StatementPhase::Consumed;
        Ok(())
    }

    fn complete(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.expected == slot.pending.is_some())
    }

    fn clear_pending(&mut self) {
        for slot in &mut self.slots {
            slot.pending = None;
        }
    }
}
