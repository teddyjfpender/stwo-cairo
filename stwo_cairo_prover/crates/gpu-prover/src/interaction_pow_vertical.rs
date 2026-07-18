//! Narrow hardware check for the generated-SN2 Interaction PoW boundary.
//!
//! This is deliberately a direct prepared-graph runner, not proof-program IR.
//! It accepts the real transcript-state allocation and bytes, uses the arena's
//! existing PoW scratch, and requires the device nonce to equal SIMD's minimum.

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::prover::backend::simd::SimdBackend;
use stwo_backend_cuda::{
    ArenaSlice, Blake2sPowWorkspaceSlots, CudaRuntimeError, DeviceArena, PreparedBlake2sPowError,
    PreparedBlake2sPowGraph, BLAKE2S_TRANSCRIPT_STATE_WORDS, POW_NONCE_WORDS,
};

/// Cairo's protocol-fixed Interaction PoW strength used by generated SN2.
pub const SN2_INTERACTION_POW_BITS: u32 = cairo_air::verifier::INTERACTION_POW_BITS;
const SN2_BOOTSTRAP_CURSOR: u32 = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sn2InteractionPowReceipt {
    pub nonce_words: [u32; POW_NONCE_WORDS],
    pub nonce: u64,
}

#[derive(Debug)]
pub enum Sn2InteractionPowError {
    StateExtent {
        expected: usize,
        actual_slice: usize,
        actual_words: usize,
    },
    NonceExtent {
        expected: usize,
        actual: usize,
    },
    NotBootstrapThroughBase {
        n_draws: u32,
        cursor: u32,
        status: u32,
    },
    Pow(PreparedBlake2sPowError),
    Cuda(CudaRuntimeError),
    NonceMismatch {
        expected: u64,
        actual: u64,
    },
}

impl core::fmt::Display for Sn2InteractionPowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid generated-SN2 Interaction PoW run: {self:?}")
    }
}

impl std::error::Error for Sn2InteractionPowError {}

impl From<PreparedBlake2sPowError> for Sn2InteractionPowError {
    fn from(value: PreparedBlake2sPowError) -> Self {
        Self::Pow(value)
    }
}

impl From<CudaRuntimeError> for Sn2InteractionPowError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// Execute and validate only the generated-SN2 Interaction PoW edge.
///
/// `transcript_state_words` must be the exact bytes represented by
/// `transcript_state` at the end of `BootstrapThroughBase`. Passing both makes
/// the device pointer and the reference oracle explicit at this boundary.
pub fn run_sn2_interaction_pow(
    arena: &DeviceArena,
    transcript_state: ArenaSlice,
    transcript_state_words: &[u32],
    transcript_nonce: ArenaSlice,
    scratch: Blake2sPowWorkspaceSlots,
) -> Result<Sn2InteractionPowReceipt, Sn2InteractionPowError> {
    if transcript_state.len_words() != BLAKE2S_TRANSCRIPT_STATE_WORDS
        || transcript_state_words.len() != BLAKE2S_TRANSCRIPT_STATE_WORDS
    {
        return Err(Sn2InteractionPowError::StateExtent {
            expected: BLAKE2S_TRANSCRIPT_STATE_WORDS,
            actual_slice: transcript_state.len_words(),
            actual_words: transcript_state_words.len(),
        });
    }
    if transcript_nonce.len_words() != POW_NONCE_WORDS {
        return Err(Sn2InteractionPowError::NonceExtent {
            expected: POW_NONCE_WORDS,
            actual: transcript_nonce.len_words(),
        });
    }
    if transcript_state_words[8] != 0
        || transcript_state_words[9] != SN2_BOOTSTRAP_CURSOR
        || transcript_state_words[10] != 0
    {
        return Err(Sn2InteractionPowError::NotBootstrapThroughBase {
            n_draws: transcript_state_words[8],
            cursor: transcript_state_words[9],
            status: transcript_state_words[10],
        });
    }

    let graph = PreparedBlake2sPowGraph::prepare(
        arena,
        transcript_state,
        SN2_INTERACTION_POW_BITS,
        transcript_nonce,
        scratch,
    )?;
    unsafe {
        arena.context().memcpy_h2d_async(
            transcript_state.as_void_ptr(),
            transcript_state_words.as_ptr().cast(),
            core::mem::size_of_val(transcript_state_words),
        )?;
    }
    graph.launch()?;

    let mut nonce_words = [0u32; POW_NONCE_WORDS];
    unsafe {
        arena.context().memcpy_d2h_async(
            nonce_words.as_mut_ptr().cast(),
            graph.nonce_destination().as_void_ptr().cast_const(),
            core::mem::size_of_val(&nonce_words),
        )?;
    }
    arena.context().sync()?;

    let nonce = u64::from(nonce_words[0]) | (u64::from(nonce_words[1]) << 32);
    let expected = reference_nonce(transcript_state_words);
    if nonce != expected {
        return Err(Sn2InteractionPowError::NonceMismatch {
            expected,
            actual: nonce,
        });
    }
    Ok(Sn2InteractionPowReceipt { nonce_words, nonce })
}

fn reference_nonce(state: &[u32]) -> u64 {
    let mut digest = [0u8; 32];
    for (destination, word) in digest.chunks_exact_mut(4).zip(&state[..8]) {
        destination.copy_from_slice(&word.to_le_bytes());
    }
    let mut channel = Blake2sChannel::default();
    channel.update_digest(Blake2sHash(digest));
    let nonce = SimdBackend::grind(&channel, SN2_INTERACTION_POW_BITS);
    debug_assert!(channel.verify_pow_nonce(SN2_INTERACTION_POW_BITS, nonce));
    nonce
}
