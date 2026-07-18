//! Exact stage order for the eager compiled-Composition proof vertical.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EagerVerticalStep {
    BeginTranscript,
    Base,
    Interaction,
    Composition,
    OodsAndQuotient,
    FriFirstTree,
    FriRound(usize),
    FinalPowAndBundle,
}

pub(super) struct EagerVerticalOrder {
    next: usize,
    fri_rounds: usize,
}

impl EagerVerticalOrder {
    pub(super) fn new(fri_rounds: usize) -> Result<Self, &'static str> {
        if fri_rounds == 0 {
            return Err("eager vertical has no FRI rounds");
        }
        7usize
            .checked_add(fri_rounds)
            .ok_or("eager vertical stage count overflow")?;
        Ok(Self {
            next: 0,
            fri_rounds,
        })
    }

    pub(super) fn admit(&mut self, actual: EagerVerticalStep) -> Result<(), &'static str> {
        if self.expected() != Some(actual) {
            return Err("eager vertical stage order");
        }
        self.next += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<(), &'static str> {
        self.expected()
            .is_none()
            .then_some(())
            .ok_or("eager vertical is incomplete")
    }

    fn expected(&self) -> Option<EagerVerticalStep> {
        match self.next {
            0 => Some(EagerVerticalStep::BeginTranscript),
            1 => Some(EagerVerticalStep::Base),
            2 => Some(EagerVerticalStep::Interaction),
            3 => Some(EagerVerticalStep::Composition),
            4 => Some(EagerVerticalStep::OodsAndQuotient),
            5 => Some(EagerVerticalStep::FriFirstTree),
            next if next < 6 + self.fri_rounds => Some(EagerVerticalStep::FriRound(next - 6)),
            next if next == 6 + self.fri_rounds => Some(EagerVerticalStep::FinalPowAndBundle),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_vertical_order_rejects_skip_duplicate_and_truncation() {
        let mut order = EagerVerticalOrder::new(3).unwrap();
        assert!(order.admit(EagerVerticalStep::Base).is_err());
        for step in [
            EagerVerticalStep::BeginTranscript,
            EagerVerticalStep::Base,
            EagerVerticalStep::Interaction,
            EagerVerticalStep::Composition,
            EagerVerticalStep::OodsAndQuotient,
            EagerVerticalStep::FriFirstTree,
            EagerVerticalStep::FriRound(0),
        ] {
            order.admit(step).unwrap();
        }
        assert!(order.admit(EagerVerticalStep::FriRound(0)).is_err());
        assert!(order.admit(EagerVerticalStep::FriRound(2)).is_err());
        order.admit(EagerVerticalStep::FriRound(1)).unwrap();
        order.admit(EagerVerticalStep::FriRound(2)).unwrap();
        assert!(order.finish().is_err());

        let mut complete = EagerVerticalOrder::new(1).unwrap();
        for step in [
            EagerVerticalStep::BeginTranscript,
            EagerVerticalStep::Base,
            EagerVerticalStep::Interaction,
            EagerVerticalStep::Composition,
            EagerVerticalStep::OodsAndQuotient,
            EagerVerticalStep::FriFirstTree,
            EagerVerticalStep::FriRound(0),
            EagerVerticalStep::FinalPowAndBundle,
        ] {
            complete.admit(step).unwrap();
        }
        complete.finish().unwrap();
    }
}
