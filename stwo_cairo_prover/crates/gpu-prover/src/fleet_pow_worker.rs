//! Persistent one-device worker for cooperative Blake2s proof of work.

use stwo_backend_cuda::{
    blake2s_pow_workspace_requirements, cuda_device_snapshot, ArenaError, ArenaLayout, ArenaSlotId,
    ArenaSlotSpec, Blake2sPowWorkspaceSlots, CudaDeviceSnapshot, CudaExecContext, CudaRuntimeError,
    DeviceArena, PreparedBlake2sPowError, PreparedBlake2sPowGraph, POW_U64_ALIGNMENT_WORDS,
};

use crate::fleet_plan::WorkerId;
use crate::fleet_pow_runtime::{FleetPowRankRequest, FleetPowRankResponse, FleetPowRuntimeError};

const STATE: ArenaSlotId = ArenaSlotId(1);
const NONCE: ArenaSlotId = ArenaSlotId(2);
const BEST_NONCE: ArenaSlotId = ArenaSlotId(3);
const COMPLETED_BLOCKS: ArenaSlotId = ArenaSlotId(4);
const PREFIX_DIGEST: ArenaSlotId = ArenaSlotId(5);

const SCRATCH: Blake2sPowWorkspaceSlots = Blake2sPowWorkspaceSlots {
    best_nonce: BEST_NONCE,
    completed_blocks: COMPLETED_BLOCKS,
    prefix_digest: PREFIX_DIGEST,
};

/// One logical fleet rank, permanently bound to visible CUDA ordinal zero.
pub struct FleetPowWorker {
    arena: DeviceArena,
    logical_rank: WorkerId,
}

impl FleetPowWorker {
    pub fn new(logical_rank: WorkerId) -> Result<Self, FleetPowWorkerError> {
        admit_device(cuda_device_snapshot()?)?;

        let layout = worker_layout()?;
        Ok(Self {
            arena: DeviceArena::new(CudaExecContext::new()?, layout)?,
            logical_rank,
        })
    }

    pub const fn logical_rank(&self) -> WorkerId {
        self.logical_rank
    }

    /// Execute one fixed request on the persistent stream and return only the
    /// rank-local result. The coordinator remains the sole transcript publisher.
    pub fn execute(
        &mut self,
        request: &FleetPowRankRequest,
    ) -> Result<FleetPowRankResponse, FleetPowWorkerError> {
        let graph = self.graph(request.pow_bits())?;
        if request.rank() != self.logical_rank {
            return Err(FleetPowWorkerError::RankMismatch {
                worker: self.logical_rank,
                request: request.rank(),
            });
        }
        let tile = request.rank_tile()?;

        let candidate = graph.execute_rank_tile(request.transcript_state(), tile)?;
        Ok(FleetPowRankResponse::completed(
            request,
            (candidate != u64::MAX).then_some(candidate),
        ))
    }

    fn graph(&self, pow_bits: u32) -> Result<PreparedBlake2sPowGraph<'_>, PreparedBlake2sPowError> {
        PreparedBlake2sPowGraph::prepare(
            &self.arena,
            self.arena.bind(STATE)?,
            pow_bits,
            self.arena.bind(NONCE)?,
            SCRATCH,
        )
    }
}

fn admit_device(device: CudaDeviceSnapshot) -> Result<(), FleetPowWorkerError> {
    if device.count != 1 {
        return Err(FleetPowWorkerError::VisibleDeviceCount(device.count));
    }
    if device.current != 0 {
        return Err(FleetPowWorkerError::CurrentDevice(device.current));
    }
    Ok(())
}

#[derive(Debug)]
pub enum FleetPowWorkerError {
    VisibleDeviceCount(u32),
    CurrentDevice(u32),
    RankMismatch { worker: WorkerId, request: WorkerId },
    Runtime(FleetPowRuntimeError),
    Pow(PreparedBlake2sPowError),
    Arena(ArenaError),
    Cuda(CudaRuntimeError),
}

impl core::fmt::Display for FleetPowWorkerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet PoW worker failed: {self:?}")
    }
}

impl std::error::Error for FleetPowWorkerError {}

impl From<FleetPowRuntimeError> for FleetPowWorkerError {
    fn from(value: FleetPowRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<PreparedBlake2sPowError> for FleetPowWorkerError {
    fn from(value: PreparedBlake2sPowError) -> Self {
        Self::Pow(value)
    }
}

impl From<ArenaError> for FleetPowWorkerError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for FleetPowWorkerError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

fn worker_layout() -> Result<ArenaLayout, ArenaError> {
    let requirements = blake2s_pow_workspace_requirements();
    let state_offset = 0;
    let nonce_offset = requirements
        .state_words
        .next_multiple_of(POW_U64_ALIGNMENT_WORDS);
    let best_nonce_offset =
        (nonce_offset + requirements.nonce_words).next_multiple_of(POW_U64_ALIGNMENT_WORDS);
    let completed_blocks_offset = best_nonce_offset + requirements.best_nonce_words;
    let prefix_digest_offset = completed_blocks_offset + requirements.completed_blocks_words;
    let total_words = prefix_digest_offset + requirements.prefix_digest_words;

    ArenaLayout::new(
        total_words,
        &[
            slot(STATE, state_offset, requirements.state_words, 1),
            slot(
                NONCE,
                nonce_offset,
                requirements.nonce_words,
                POW_U64_ALIGNMENT_WORDS,
            ),
            slot(
                BEST_NONCE,
                best_nonce_offset,
                requirements.best_nonce_words,
                POW_U64_ALIGNMENT_WORDS,
            ),
            slot(
                COMPLETED_BLOCKS,
                completed_blocks_offset,
                requirements.completed_blocks_words,
                1,
            ),
            slot(
                PREFIX_DIGEST,
                prefix_digest_offset,
                requirements.prefix_digest_words,
                1,
            ),
        ],
    )
}

const fn slot(
    id: ArenaSlotId,
    offset_words: usize,
    len_words: usize,
    alignment_words: usize,
) -> ArenaSlotSpec {
    ArenaSlotSpec {
        id,
        offset_words,
        len_words,
        alignment_words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_arena_is_the_exact_twenty_nine_word_pow_layout() {
        let layout = worker_layout().unwrap();
        assert_eq!(layout.total_words(), 29);
        assert_eq!(layout.slot(STATE).unwrap().offset_words, 0);
        assert_eq!(layout.slot(NONCE).unwrap().offset_words, 16);
        assert_eq!(layout.slot(BEST_NONCE).unwrap().offset_words, 18);
        assert_eq!(layout.slot(COMPLETED_BLOCKS).unwrap().offset_words, 20);
        assert_eq!(layout.slot(PREFIX_DIGEST).unwrap().offset_words, 21);
    }

    #[test]
    fn worker_admits_only_visible_ordinal_zero() {
        assert!(admit_device(CudaDeviceSnapshot {
            count: 1,
            current: 0,
            sm_major: 8,
            sm_minor: 9,
        })
        .is_ok());
        assert!(matches!(
            admit_device(CudaDeviceSnapshot {
                count: 2,
                ..Default::default()
            }),
            Err(FleetPowWorkerError::VisibleDeviceCount(2))
        ));
        assert!(matches!(
            admit_device(CudaDeviceSnapshot {
                count: 1,
                current: 1,
                ..Default::default()
            }),
            Err(FleetPowWorkerError::CurrentDevice(1))
        ));
    }
}
