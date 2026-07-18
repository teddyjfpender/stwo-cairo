use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetWorkerInstallError {
    UnknownWorker(WorkerId),
    UnsupportedSpill,
    UnsupportedScratch(LayoutTransitionId),
    UnsupportedElement(ValueVersion),
    InvalidStorage(StorageId),
    InvalidOperation(OpId),
    InvalidStatementHostIngress(OpId),
    MissingEffectWindow {
        worker: WorkerId,
        value: ValueRange,
    },
    AmbiguousEffectWindow {
        worker: WorkerId,
        value: ValueRange,
    },
    AmbiguousEffect {
        operation: OpId,
        binding: EffectBindingId,
    },
    MisalignedEffect {
        operation: OpId,
        binding: EffectBindingId,
    },
    InvalidCoordinator,
    CapacityExceeded {
        worker: WorkerId,
        required: usize,
        capacity: usize,
    },
    RuntimeView(FleetRuntimeViewError),
    Plan(FleetPlanError),
    SizeOverflow,
}

impl core::fmt::Display for FleetWorkerInstallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet worker install plan: {self:?}")
    }
}

impl std::error::Error for FleetWorkerInstallError {}

impl From<FleetRuntimeViewError> for FleetWorkerInstallError {
    fn from(value: FleetRuntimeViewError) -> Self {
        Self::RuntimeView(value)
    }
}
