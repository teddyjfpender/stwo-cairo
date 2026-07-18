//! Live rank-local slab installation for one validated worker plan.

use stwo_backend_cuda::{
    ArenaError, ArenaLayout, ArenaSlice, ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena,
};

use super::{FleetInstallWindow, FleetWorkerInstallPlan, StorageId, WorkerId};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FleetWorkerDeviceInstallError {
    EmptySlab,
    NonWordGeometry,
    UnknownStorage(StorageId),
    InvalidWindow(StorageId),
    Arena(ArenaError),
}

impl core::fmt::Display for FleetWorkerDeviceInstallError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid fleet worker device install: {self:?}")
    }
}

impl std::error::Error for FleetWorkerDeviceInstallError {}

impl From<ArenaError> for FleetWorkerDeviceInstallError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

/// One context-owned stable slab installed from the immutable worker plan.
pub(crate) struct FleetWorkerDeviceInstall {
    plan_identity: [u8; 32],
    worker: WorkerId,
    arena: DeviceArena,
}

impl FleetWorkerDeviceInstall {
    pub(crate) fn install(
        context: CudaExecContext,
        plan: &FleetWorkerInstallPlan,
    ) -> Result<Self, FleetWorkerDeviceInstallError> {
        let layout = arena_layout(plan)?;
        let arena = DeviceArena::new(context, layout)?;
        Ok(Self {
            plan_identity: plan.plan_identity(),
            worker: plan.target().worker,
            arena,
        })
    }

    pub(crate) const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub(crate) const fn worker(&self) -> WorkerId {
        self.worker
    }

    pub(crate) const fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    /// Bind one already-validated execution/transcript/output window.
    pub(crate) fn window(
        &self,
        plan: &FleetWorkerInstallPlan,
        window: FleetInstallWindow,
    ) -> Result<ArenaSlice, FleetWorkerDeviceInstallError> {
        if plan.plan_identity() != self.plan_identity || plan.target().worker != self.worker {
            return Err(FleetWorkerDeviceInstallError::InvalidWindow(window.storage));
        }
        let (slot, offset_words, len_words) = validate_window(plan, window)?;
        self.arena
            .bind(slot)?
            .checked_subslice(offset_words, len_words)
            .map_err(Into::into)
    }
}

fn validate_window(
    plan: &FleetWorkerInstallPlan,
    window: FleetInstallWindow,
) -> Result<(ArenaSlotId, usize, usize), FleetWorkerDeviceInstallError> {
    let storage = plan
        .storages()
        .iter()
        .find(|storage| storage.storage == window.storage)
        .ok_or(FleetWorkerDeviceInstallError::UnknownStorage(
            window.storage,
        ))?;
    let expected_slab_offset = storage
        .slab_offset_bytes
        .checked_add(window.offset_bytes)
        .ok_or(FleetWorkerDeviceInstallError::InvalidWindow(window.storage))?;
    let end = window
        .offset_bytes
        .checked_add(window.bytes)
        .ok_or(FleetWorkerDeviceInstallError::InvalidWindow(window.storage))?;
    if window.bytes == 0
        || end > storage.bytes
        || window.slab_offset_bytes != expected_slab_offset
        || window.offset_bytes % WORD_BYTES != 0
        || window.bytes % WORD_BYTES != 0
    {
        return Err(FleetWorkerDeviceInstallError::InvalidWindow(window.storage));
    }
    Ok((
        ArenaSlotId(window.storage.0),
        window.offset_bytes / WORD_BYTES,
        window.bytes / WORD_BYTES,
    ))
}

fn arena_layout(
    plan: &FleetWorkerInstallPlan,
) -> Result<ArenaLayout, FleetWorkerDeviceInstallError> {
    let slab_bytes = plan.capacity().slab_bytes;
    if slab_bytes == 0 {
        return Err(FleetWorkerDeviceInstallError::EmptySlab);
    }
    if slab_bytes % WORD_BYTES != 0
        || plan.storages().iter().any(|storage| {
            storage.slab_offset_bytes % WORD_BYTES != 0
                || storage.bytes % WORD_BYTES != 0
                || storage.alignment_bytes % WORD_BYTES != 0
        })
    {
        return Err(FleetWorkerDeviceInstallError::NonWordGeometry);
    }
    let specs = plan
        .storages()
        .iter()
        .map(|storage| ArenaSlotSpec {
            id: ArenaSlotId(storage.storage.0),
            offset_words: storage.slab_offset_bytes / WORD_BYTES,
            len_words: storage.bytes / WORD_BYTES,
            alignment_words: storage.alignment_bytes / WORD_BYTES,
        })
        .collect::<Vec<_>>();
    ArenaLayout::new(slab_bytes / WORD_BYTES, &specs).map_err(Into::into)
}

#[cfg(test)]
pub(super) fn arena_layout_for_test(
    plan: &FleetWorkerInstallPlan,
) -> Result<ArenaLayout, FleetWorkerDeviceInstallError> {
    arena_layout(plan)
}

#[cfg(test)]
pub(super) fn validate_window_for_test(
    plan: &FleetWorkerInstallPlan,
    window: FleetInstallWindow,
) -> Result<(ArenaSlotId, usize, usize), FleetWorkerDeviceInstallError> {
    validate_window(plan, window)
}
