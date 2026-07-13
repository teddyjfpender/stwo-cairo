use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_backend_cuda::{
    replay_blake2s_reference, Blake2sTranscriptSchedule, TranscriptBoundaryId,
    TranscriptBoundaryState, TranscriptInputId, TranscriptOperation, TranscriptOutputId,
    TranscriptReferenceTrace, TranscriptStart, BLAKE2S_TRANSCRIPT_PROTOCOL_TAG,
};

pub const MAX_REJECTION_ROUNDS: u32 = 32;
pub const CAIRO_MAX_REJECTION_ROUNDS: u32 = 64;
pub const STATE32: TranscriptInputId = TranscriptInputId(3_200);
pub const ROOT6: TranscriptInputId = TranscriptInputId(6_006);
pub const ALPHA6: TranscriptOutputId = TranscriptOutputId(7_006);
pub const ROOT7: TranscriptInputId = TranscriptInputId(6_007);
pub const ALPHA7: TranscriptOutputId = TranscriptOutputId(7_007);

pub fn cairo_schedule(operation_count: usize) -> Result<Blake2sTranscriptSchedule, String> {
    let operations = [
        TranscriptOperation::AbsorbRoot {
            boundary: TranscriptBoundaryId(0x1_001a),
            source: TranscriptInputId(0x1_0018),
        },
        TranscriptOperation::DrawSecureFelt {
            boundary: TranscriptBoundaryId(0x1_001b),
            output: TranscriptOutputId(0x1_0019),
        },
        TranscriptOperation::AbsorbRoot {
            boundary: TranscriptBoundaryId(0x1_001e),
            source: TranscriptInputId(0x1_001c),
        },
        TranscriptOperation::DrawSecureFelt {
            boundary: TranscriptBoundaryId(0x1_001f),
            output: TranscriptOutputId(0x1_001d),
        },
    ];
    Blake2sTranscriptSchedule::new(
        TranscriptStart::DeviceState(TranscriptInputId(0)),
        operations[..operation_count].to_vec(),
        CAIRO_MAX_REJECTION_ROUNDS,
    )
    .map_err(|error| error.to_string())
}

pub fn schedule(operation_count: usize) -> Result<Blake2sTranscriptSchedule, String> {
    let operations = [
        TranscriptOperation::AbsorbRoot {
            boundary: TranscriptBoundaryId(32),
            source: ROOT6,
        },
        TranscriptOperation::DrawSecureFelt {
            boundary: TranscriptBoundaryId(33),
            output: ALPHA6,
        },
        TranscriptOperation::AbsorbRoot {
            boundary: TranscriptBoundaryId(34),
            source: ROOT7,
        },
        TranscriptOperation::DrawSecureFelt {
            boundary: TranscriptBoundaryId(35),
            output: ALPHA7,
        },
    ];
    Blake2sTranscriptSchedule::new(
        TranscriptStart::DeviceState(STATE32),
        operations[..operation_count].to_vec(),
        MAX_REJECTION_ROUNDS,
    )
    .map_err(|error| error.to_string())
}

pub fn replay(
    schedule: &Blake2sTranscriptSchedule,
    cursor32: &[u32],
    roots: &[Blake2sHash],
) -> Result<TranscriptReferenceTrace, String> {
    let mut inputs = cursor32[..9].to_vec();
    inputs.extend(roots.iter().copied().flat_map(hash_words));
    replay_blake2s_reference(schedule, &inputs).map_err(|error| error.to_string())
}

pub fn independently_checked_chains(
    schedule: &Blake2sTranscriptSchedule,
) -> Result<[u64; 5], String> {
    let mut hash = StableHash::new();
    hash.bytes(BLAKE2S_TRANSCRIPT_PROTOCOL_TAG.as_bytes());
    hash.u32(MAX_REJECTION_ROUNDS);
    hash.u32(1);
    hash.u32(STATE32.0);
    let mut chains = [0; 5];
    chains[0] = hash.finish();
    for (index, encoded) in [
        [32, 4, ROOT6.0],
        [33, 6, ALPHA6.0],
        [34, 4, ROOT7.0],
        [35, 6, ALPHA7.0],
    ]
    .into_iter()
    .enumerate()
    {
        encoded.into_iter().for_each(|word| hash.u32(word));
        chains[index + 1] = hash.finish();
    }
    if schedule.initial_chain() != chains[0] || schedule.protocol_key() != chains[4] {
        return Err("independent FNV operation encoding disagrees with production schedule".into());
    }
    Ok(chains)
}

pub fn cursor32_state(chain: u64, source_circle_log: u32, entry_log: u32) -> Vec<u32> {
    let mut words = hash_words(seeded_digest(source_circle_log, entry_log));
    words.extend([0, 32, 0, 0, chain as u32, (chain >> 32) as u32, 0, 0]);
    words
}

pub fn seeded_digest(source_circle_log: u32, entry_log: u32) -> Blake2sHash {
    let mut channel = Blake2sChannel::default();
    channel.mix_u32s(&[0x5354_574f, 0x4652_4936, source_circle_log, entry_log]);
    channel.digest()
}

pub fn state_words(state: &TranscriptBoundaryState, cursor: u32, chain: u64) -> Vec<u32> {
    let mut words = hash_words(state.digest);
    words.extend([
        state.n_draws,
        cursor,
        0,
        0,
        chain as u32,
        (chain >> 32) as u32,
        0,
        0,
    ]);
    words
}

pub fn hash_words(hash: Blake2sHash) -> Vec<u32> {
    hash.0
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[derive(Clone, Copy)]
struct StableHash(u64);

impl StableHash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_chain_encoder_matches_all_production_prefixes() {
        let full = schedule(4).unwrap();
        let chains = independently_checked_chains(&full).unwrap();
        for count in 1..=4 {
            let prefix = schedule(count).unwrap();
            assert_eq!(prefix.initial_chain(), chains[0]);
            assert_eq!(prefix.protocol_key(), chains[count]);
        }
    }
}
