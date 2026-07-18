//! Direct generated-SN2 Interaction PoW differential.

#![cfg(stwo_cuda_link)]

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo_backend_cuda::{
    ArenaLayout, ArenaSlotId, ArenaSlotSpec, Blake2sPowWorkspaceSlots, CudaExecContext, DeviceArena,
};
use stwo_cairo_gpu_prover::interaction_pow_vertical::run_sn2_interaction_pow;

const STATE: ArenaSlotId = ArenaSlotId(1);
const NONCE: ArenaSlotId = ArenaSlotId(2);
const BEST_NONCE: ArenaSlotId = ArenaSlotId(3);
const COMPLETED_BLOCKS: ArenaSlotId = ArenaSlotId(4);
const PREFIX_DIGEST: ArenaSlotId = ArenaSlotId(5);

#[test]
fn generated_sn2_interaction_pow_matches_the_simd_minimum() {
    let layout = ArenaLayout::new(
        29,
        &[
            slot(STATE, 0, 16, 8),
            slot(NONCE, 16, 2, 2),
            slot(BEST_NONCE, 18, 2, 2),
            slot(COMPLETED_BLOCKS, 20, 1, 1),
            slot(PREFIX_DIGEST, 21, 8, 1),
        ],
    )
    .unwrap();
    let arena = DeviceArena::new(CudaExecContext::new().unwrap(), layout).unwrap();
    let state_words = transcript_state_words();
    let receipt = run_sn2_interaction_pow(
        &arena,
        arena.bind(STATE).unwrap(),
        &state_words,
        arena.bind(NONCE).unwrap(),
        Blake2sPowWorkspaceSlots {
            best_nonce: BEST_NONCE,
            completed_blocks: COMPLETED_BLOCKS,
            prefix_digest: PREFIX_DIGEST,
        },
    )
    .unwrap();
    assert_eq!(
        receipt.nonce,
        u64::from(receipt.nonce_words[0]) | (u64::from(receipt.nonce_words[1]) << 32)
    );
}

fn transcript_state_words() -> [u32; 16] {
    let mut channel = Blake2sChannel::default();
    channel.mix_u32s(&[2, 0x1122_3344, 0xaabb_ccdd, 24]);
    let mut state = [0u32; 16];
    for (word, bytes) in state[..8]
        .iter_mut()
        .zip(channel.digest().0.chunks_exact(4))
    {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    state[8] = channel.n_draws();
    state[9] = 11; // End of BootstrapThroughBase.
    state[12] = 0x89ab_cdef; // Real state ABI fields; PoW depends only on digest.
    state[13] = 0x0123_4567;
    state
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
