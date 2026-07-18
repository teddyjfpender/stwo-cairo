//! Exact resident-graph replay with two-rank ownership of both PoW searches.

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_backend_cuda::{Blake2sPowRankTile, BLAKE2S_TRANSCRIPT_STATE_WORDS};

use crate::fleet_pow::FleetPowSite;
use crate::fleet_pow_runtime::{
    FleetPowRankResponse, FleetPowResolution, FleetPowRuntimeError, FleetPowTransport,
    TwoRankFleetPowCoordinator,
};
use crate::graphs::ResidentGraphTopology;
use crate::resident_runtime::{ResidentGraphRuntime, ResidentRuntimeError};

const TRANSCRIPT_STATUS_WORD: usize = 10;

/// Replay one already-captured `FleetPowSplit` proof DAG.
pub fn replay_fleet_pow_split<T: FleetPowTransport>(
    runtime: &mut ResidentGraphRuntime<'_>,
    coordinator: &mut TwoRankFleetPowCoordinator<T>,
) -> Result<(), FleetPowReplayError> {
    runtime.require_complete_captured_topology_for(ResidentGraphTopology::FleetPowSplit)?;
    runtime.begin_next_transcript_generation()?;

    runtime.replay_base_prefix()?;
    let interaction_state = runtime.interaction_pow_state()?;
    let interaction = resolve(
        coordinator,
        FleetPowSite::Interaction,
        runtime.interaction_pow_bits(),
        interaction_state,
        |tile| runtime.launch_interaction_pow_rank_tile(tile),
    )?;
    runtime.upload_interaction_pow_nonce(interaction.nonce)?;
    runtime.replay_base_resume()?;

    runtime.replay_interaction_relation_and_commit()?;
    runtime.replay_composition_commit_only()?;
    runtime.replay_oods_transcript_boundary()?;
    runtime.replay_fri_first_tree()?;
    for round in 0..runtime.fri_round_count() {
        runtime.replay_fri_round(round)?;
    }

    runtime.replay_final_prefix()?;
    let query_state = runtime.query_pow_state()?;
    let query = resolve(
        coordinator,
        FleetPowSite::Query,
        runtime.query_pow_bits(),
        query_state,
        |tile| runtime.launch_query_pow_rank_tile(tile),
    )?;
    runtime.upload_query_pow_nonce(query.nonce)?;
    runtime.replay_final_resume()?;
    Ok(())
}

fn resolve<T: FleetPowTransport>(
    coordinator: &mut TwoRankFleetPowCoordinator<T>,
    site: FleetPowSite,
    pow_bits: u32,
    state: [u32; BLAKE2S_TRANSCRIPT_STATE_WORDS],
    mut launch: impl FnMut(Blake2sPowRankTile) -> Result<u64, ResidentRuntimeError>,
) -> Result<FleetPowResolution, FleetPowReplayError> {
    let channel = channel_from_state(site, &state)?;
    let mut resident_error = None;
    let resolution = coordinator.resolve(
        site,
        pow_bits,
        state,
        |request| {
            let tile = request.rank_tile()?;
            match launch(tile) {
                Ok(candidate) => Ok(FleetPowRankResponse::completed(
                    request,
                    (candidate != u64::MAX).then_some(candidate),
                )),
                Err(error) => {
                    resident_error = Some(error);
                    Err(FleetPowRuntimeError::Transport(
                        "rank-zero CUDA launch failed".into(),
                    ))
                }
            }
        },
        |nonce| channel.verify_pow_nonce(pow_bits, nonce),
    );
    if let Some(error) = resident_error {
        return Err(error.into());
    }
    Ok(resolution?)
}

fn channel_from_state(
    site: FleetPowSite,
    state: &[u32; BLAKE2S_TRANSCRIPT_STATE_WORDS],
) -> Result<Blake2sChannel, FleetPowReplayError> {
    let status = state[TRANSCRIPT_STATUS_WORD];
    if status != 0 {
        return Err(FleetPowReplayError::InvalidTranscriptState { site, status });
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(&state[..8]) {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    let mut channel = Blake2sChannel::default();
    channel.update_digest(Blake2sHash(digest));
    Ok(channel)
}

#[derive(Debug)]
pub enum FleetPowReplayError {
    Resident(ResidentRuntimeError),
    Fleet(FleetPowRuntimeError),
    InvalidTranscriptState { site: FleetPowSite, status: u32 },
}

impl core::fmt::Display for FleetPowReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet resident replay failed: {self:?}")
    }
}

impl std::error::Error for FleetPowReplayError {}

impl From<ResidentRuntimeError> for FleetPowReplayError {
    fn from(value: ResidentRuntimeError) -> Self {
        Self::Resident(value)
    }
}

impl From<FleetPowRuntimeError> for FleetPowReplayError {
    fn from(value: FleetPowRuntimeError) -> Self {
        Self::Fleet(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_rebuilds_the_exact_little_endian_channel_digest() {
        let mut state = [0u32; BLAKE2S_TRANSCRIPT_STATE_WORDS];
        for (index, word) in state[..8].iter_mut().enumerate() {
            *word = 0x0102_0304 + index as u32;
        }
        let channel = channel_from_state(FleetPowSite::Interaction, &state).unwrap();
        let expected = core::array::from_fn(|index| state[index / 4].to_le_bytes()[index % 4]);
        assert_eq!(channel.digest(), Blake2sHash(expected));
    }

    #[test]
    fn state_rejects_device_transcript_failure_before_pow() {
        let mut state = [0u32; BLAKE2S_TRANSCRIPT_STATE_WORDS];
        state[TRANSCRIPT_STATUS_WORD] = 7;
        assert!(matches!(
            channel_from_state(FleetPowSite::Query, &state),
            Err(FleetPowReplayError::InvalidTranscriptState {
                site: FleetPowSite::Query,
                status: 7
            })
        ));
    }
}
