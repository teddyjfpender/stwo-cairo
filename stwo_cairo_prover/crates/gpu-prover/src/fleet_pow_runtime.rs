//! Minimal two-rank control plane for resident Blake2s proof of work.

use stwo_backend_cuda::{
    Blake2sPowFleetAttempt, Blake2sPowRankTile, BLAKE2S_TRANSCRIPT_STATE_WORDS,
    POW_THREADS_PER_BLOCK,
};

use crate::fleet_plan::WorkerId;
use crate::fleet_pow::{FleetPowError, FleetPowSchedule, FleetPowSite, PowRankReceipt};

const MAGIC: [u8; 8] = *b"STW2POW1";
const VERSION: u16 = 1;
const REQUEST_KIND: u8 = 1;
const RESPONSE_KIND: u8 = 2;
const RANK_COUNT: usize = 2;
const TRANSCRIPT_STATUS_WORD: usize = 10;

pub const FLEET_POW_REQUEST_BYTES: usize = 160;
pub const FLEET_POW_RESPONSE_BYTES: usize = 136;

/// One rank's immutable share of a coordinator-owned attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetPowRankRequest {
    plan_identity: [u8; 32],
    proof_generation: u64,
    site: FleetPowSite,
    attempt_ordinal: u64,
    rank: WorkerId,
    pow_bits: u32,
    workers_per_rank: u32,
    start_index: u64,
    end_index: u64,
    grid_blocks: u32,
    transcript_state: [u32; BLAKE2S_TRANSCRIPT_STATE_WORDS],
}

impl FleetPowRankRequest {
    fn new(
        plan_identity: [u8; 32],
        proof_generation: u64,
        site: FleetPowSite,
        attempt_ordinal: u64,
        rank: WorkerId,
        pow_bits: u32,
        workers_per_rank: u32,
        attempt: Blake2sPowFleetAttempt,
        transcript_state: [u32; BLAKE2S_TRANSCRIPT_STATE_WORDS],
    ) -> Result<Self, FleetPowRuntimeError> {
        let request = Self {
            plan_identity,
            proof_generation,
            site,
            attempt_ordinal,
            rank,
            pow_bits,
            workers_per_rank,
            start_index: attempt.start_index(),
            end_index: attempt.end_index(),
            grid_blocks: attempt.grid_blocks(),
            transcript_state,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub const fn proof_generation(&self) -> u64 {
        self.proof_generation
    }

    pub const fn site(&self) -> FleetPowSite {
        self.site
    }

    pub const fn attempt_ordinal(&self) -> u64 {
        self.attempt_ordinal
    }

    pub const fn rank(&self) -> WorkerId {
        self.rank
    }

    pub const fn pow_bits(&self) -> u32 {
        self.pow_bits
    }

    pub const fn transcript_state(&self) -> &[u32; BLAKE2S_TRANSCRIPT_STATE_WORDS] {
        &self.transcript_state
    }

    pub fn rank_tile(&self) -> Result<Blake2sPowRankTile, FleetPowRuntimeError> {
        self.attempt()?
            .rank_tile(u32::from(self.rank.0))
            .map_err(|_| FleetPowRuntimeError::InvalidFrame("request rank"))
    }

    pub fn to_bytes(&self) -> [u8; FLEET_POW_REQUEST_BYTES] {
        let mut frame = [0u8; FLEET_POW_REQUEST_BYTES];
        frame[..8].copy_from_slice(&MAGIC);
        put_u16(&mut frame, 8, VERSION);
        frame[10] = REQUEST_KIND;
        frame[11] = site_to_wire(self.site);
        put_u16(&mut frame, 12, self.rank.0);
        put_u16(&mut frame, 14, RANK_COUNT as u16);
        put_u32(&mut frame, 16, self.pow_bits);
        put_u32(&mut frame, 20, self.workers_per_rank);
        put_u32(&mut frame, 24, self.grid_blocks);
        put_u64(&mut frame, 32, self.proof_generation);
        put_u64(&mut frame, 40, self.attempt_ordinal);
        put_u64(&mut frame, 48, self.start_index);
        put_u64(&mut frame, 56, self.end_index);
        frame[64..96].copy_from_slice(&self.plan_identity);
        for (index, word) in self.transcript_state.iter().enumerate() {
            put_u32(&mut frame, 96 + index * 4, *word);
        }
        frame
    }

    pub fn from_bytes(frame: &[u8; FLEET_POW_REQUEST_BYTES]) -> Result<Self, FleetPowRuntimeError> {
        validate_header(frame, REQUEST_KIND)?;
        if frame[28..32] != [0; 4] || get_u16(frame, 14) as usize != RANK_COUNT {
            return Err(FleetPowRuntimeError::InvalidFrame("request header"));
        }
        let mut plan_identity = [0u8; 32];
        plan_identity.copy_from_slice(&frame[64..96]);
        let mut transcript_state = [0u32; BLAKE2S_TRANSCRIPT_STATE_WORDS];
        for (index, word) in transcript_state.iter_mut().enumerate() {
            *word = get_u32(frame, 96 + index * 4);
        }
        let request = Self {
            plan_identity,
            proof_generation: get_u64(frame, 32),
            site: site_from_wire(frame[11])?,
            attempt_ordinal: get_u64(frame, 40),
            rank: WorkerId(get_u16(frame, 12)),
            pow_bits: get_u32(frame, 16),
            workers_per_rank: get_u32(frame, 20),
            start_index: get_u64(frame, 48),
            end_index: get_u64(frame, 56),
            grid_blocks: get_u32(frame, 24),
            transcript_state,
        };
        request.validate()?;
        Ok(request)
    }

    fn attempt(&self) -> Result<Blake2sPowFleetAttempt, FleetPowRuntimeError> {
        Blake2sPowFleetAttempt::new(
            RANK_COUNT as u32,
            self.start_index,
            self.end_index,
            self.grid_blocks,
        )
        .map_err(|_| FleetPowRuntimeError::InvalidFrame("request tile"))
    }

    fn validate(&self) -> Result<(), FleetPowRuntimeError> {
        let expected_workers = self
            .grid_blocks
            .checked_mul(POW_THREADS_PER_BLOCK)
            .ok_or(FleetPowRuntimeError::InvalidFrame("request workers"))?;
        if self.pow_bits > 32
            || self.rank.0 as usize >= RANK_COUNT
            || self.workers_per_rank != expected_workers
            || self.transcript_state[TRANSCRIPT_STATUS_WORD] != 0
        {
            return Err(FleetPowRuntimeError::InvalidFrame("request geometry"));
        }
        self.rank_tile()?;
        Ok(())
    }

    fn identity(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }
}

/// A completed rank result, bound to every byte of its request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetPowRankResponse {
    plan_identity: [u8; 32],
    request_identity: [u8; 32],
    proof_generation: u64,
    site: FleetPowSite,
    attempt_ordinal: u64,
    rank: WorkerId,
    pow_bits: u32,
    workers_per_rank: u32,
    start_index: u64,
    end_index: u64,
    grid_blocks: u32,
    candidate_nonce: Option<u64>,
}

impl FleetPowRankResponse {
    pub fn completed(request: &FleetPowRankRequest, candidate_nonce: Option<u64>) -> Self {
        Self {
            plan_identity: request.plan_identity,
            request_identity: request.identity(),
            proof_generation: request.proof_generation,
            site: request.site,
            attempt_ordinal: request.attempt_ordinal,
            rank: request.rank,
            pow_bits: request.pow_bits,
            workers_per_rank: request.workers_per_rank,
            start_index: request.start_index,
            end_index: request.end_index,
            grid_blocks: request.grid_blocks,
            candidate_nonce,
        }
    }

    pub const fn candidate_nonce(&self) -> Option<u64> {
        self.candidate_nonce
    }

    pub fn to_bytes(&self) -> [u8; FLEET_POW_RESPONSE_BYTES] {
        let mut frame = [0u8; FLEET_POW_RESPONSE_BYTES];
        frame[..8].copy_from_slice(&MAGIC);
        put_u16(&mut frame, 8, VERSION);
        frame[10] = RESPONSE_KIND;
        frame[11] = site_to_wire(self.site);
        put_u16(&mut frame, 12, self.rank.0);
        put_u16(&mut frame, 14, RANK_COUNT as u16);
        put_u32(&mut frame, 16, self.pow_bits);
        put_u32(&mut frame, 20, self.workers_per_rank);
        put_u32(&mut frame, 24, self.grid_blocks);
        put_u32(
            &mut frame,
            28,
            1 | u32::from(self.candidate_nonce.is_some()) << 1,
        );
        put_u64(&mut frame, 32, self.proof_generation);
        put_u64(&mut frame, 40, self.attempt_ordinal);
        put_u64(&mut frame, 48, self.start_index);
        put_u64(&mut frame, 56, self.end_index);
        put_u64(&mut frame, 64, self.candidate_nonce.unwrap_or(0));
        frame[72..104].copy_from_slice(&self.plan_identity);
        frame[104..136].copy_from_slice(&self.request_identity);
        frame
    }

    pub fn from_bytes(
        frame: &[u8; FLEET_POW_RESPONSE_BYTES],
    ) -> Result<Self, FleetPowRuntimeError> {
        validate_header(frame, RESPONSE_KIND)?;
        let flags = get_u32(frame, 28);
        if get_u16(frame, 14) as usize != RANK_COUNT || !matches!(flags, 1 | 3) {
            return Err(FleetPowRuntimeError::InvalidFrame("response header"));
        }
        let candidate = get_u64(frame, 64);
        if flags == 1 && candidate != 0 {
            return Err(FleetPowRuntimeError::InvalidFrame("response candidate"));
        }
        let mut plan_identity = [0u8; 32];
        plan_identity.copy_from_slice(&frame[72..104]);
        let mut request_identity = [0u8; 32];
        request_identity.copy_from_slice(&frame[104..136]);
        Ok(Self {
            plan_identity,
            request_identity,
            proof_generation: get_u64(frame, 32),
            site: site_from_wire(frame[11])?,
            attempt_ordinal: get_u64(frame, 40),
            rank: WorkerId(get_u16(frame, 12)),
            pow_bits: get_u32(frame, 16),
            workers_per_rank: get_u32(frame, 20),
            start_index: get_u64(frame, 48),
            end_index: get_u64(frame, 56),
            grid_blocks: get_u32(frame, 24),
            candidate_nonce: (flags == 3).then_some(candidate),
        })
    }

    fn validate_for(&self, request: &FleetPowRankRequest) -> Result<(), FleetPowRuntimeError> {
        let mismatch = if self.plan_identity != request.plan_identity {
            "plan identity"
        } else if self.proof_generation != request.proof_generation {
            "generation"
        } else if self.site != request.site {
            "site"
        } else if self.attempt_ordinal != request.attempt_ordinal {
            "attempt"
        } else if self.rank != request.rank {
            "rank"
        } else if self.pow_bits != request.pow_bits {
            "pow bits"
        } else if self.workers_per_rank != request.workers_per_rank
            || self.start_index != request.start_index
            || self.end_index != request.end_index
            || self.grid_blocks != request.grid_blocks
        {
            "tile"
        } else if self.request_identity != request.identity() {
            "request identity"
        } else {
            return Ok(());
        };
        Err(FleetPowRuntimeError::MismatchedResponse(mismatch))
    }

    fn receipt(&self) -> PowRankReceipt {
        PowRankReceipt {
            site: self.site,
            plan_identity: self.plan_identity,
            rank: self.rank,
            proof_generation: self.proof_generation,
            attempt_ordinal: self.attempt_ordinal,
            candidate_nonce: self.candidate_nonce,
        }
    }
}

/// Sends rank 1 before local rank 0 starts, then receives exactly one result.
pub trait FleetPowTransport {
    fn send(&mut self, request: &FleetPowRankRequest) -> Result<(), FleetPowRuntimeError>;
    fn receive(&mut self) -> Result<FleetPowRankResponse, FleetPowRuntimeError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetPowResolution {
    pub nonce: u64,
    pub attempt_ordinal: u64,
}

pub struct TwoRankFleetPowCoordinator<T> {
    schedule: FleetPowSchedule,
    plan_identity: [u8; 32],
    proof_generation: u64,
    next_site: Option<FleetPowSite>,
    transport: T,
}

impl<T: FleetPowTransport> TwoRankFleetPowCoordinator<T> {
    pub fn new(
        schedule: FleetPowSchedule,
        plan_identity: [u8; 32],
        proof_generation: u64,
        transport: T,
    ) -> Result<Self, FleetPowRuntimeError> {
        schedule.validate(RANK_COUNT)?;
        schedule.interaction.attempt(RANK_COUNT, 0)?;
        schedule.query.attempt(RANK_COUNT, 0)?;
        Ok(Self {
            schedule,
            plan_identity,
            proof_generation,
            next_site: Some(FleetPowSite::Interaction),
            transport,
        })
    }

    pub fn resolve(
        &mut self,
        site: FleetPowSite,
        pow_bits: u32,
        transcript_state: [u32; BLAKE2S_TRANSCRIPT_STATE_WORDS],
        mut execute_local: impl FnMut(
            &FleetPowRankRequest,
        ) -> Result<FleetPowRankResponse, FleetPowRuntimeError>,
        is_valid_nonce: impl Fn(u64) -> bool,
    ) -> Result<FleetPowResolution, FleetPowRuntimeError> {
        if self.next_site != Some(site) {
            return Err(FleetPowRuntimeError::UnexpectedSite {
                expected: self.next_site,
                actual: site,
            });
        }
        let plan = self.schedule.plan(site);
        for attempt_ordinal in 0.. {
            let attempt = plan.attempt(RANK_COUNT, attempt_ordinal)?;
            let local_request = FleetPowRankRequest::new(
                self.plan_identity,
                self.proof_generation,
                site,
                attempt_ordinal,
                WorkerId(0),
                pow_bits,
                plan.workers_per_rank,
                attempt,
                transcript_state,
            )?;
            let remote_request = FleetPowRankRequest::new(
                self.plan_identity,
                self.proof_generation,
                site,
                attempt_ordinal,
                WorkerId(1),
                pow_bits,
                plan.workers_per_rank,
                attempt,
                transcript_state,
            )?;

            self.transport.send(&remote_request)?;
            let local = execute_local(&local_request)?;
            let remote = self.transport.receive()?;
            local.validate_for(&local_request)?;
            remote.validate_for(&remote_request)?;
            let receipts = [local.receipt(), remote.receipt()];
            if let Some(nonce) = plan.reduce_attempt(
                site,
                RANK_COUNT,
                self.plan_identity,
                self.proof_generation,
                attempt_ordinal,
                &receipts,
                &is_valid_nonce,
            )? {
                self.next_site = match site {
                    FleetPowSite::Interaction => Some(FleetPowSite::Query),
                    FleetPowSite::Query => None,
                };
                return Ok(FleetPowResolution {
                    nonce,
                    attempt_ordinal,
                });
            }
        }
        unreachable!("attempt ordinal covers u64");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetPowRuntimeError {
    Pow(FleetPowError),
    InvalidFrame(&'static str),
    MismatchedResponse(&'static str),
    UnexpectedSite {
        expected: Option<FleetPowSite>,
        actual: FleetPowSite,
    },
    Transport(String),
}

impl core::fmt::Display for FleetPowRuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet PoW runtime failed: {self:?}")
    }
}

impl std::error::Error for FleetPowRuntimeError {}

impl From<FleetPowError> for FleetPowRuntimeError {
    fn from(value: FleetPowError) -> Self {
        Self::Pow(value)
    }
}

fn validate_header(frame: &[u8], kind: u8) -> Result<(), FleetPowRuntimeError> {
    if frame[..8] != MAGIC || get_u16(frame, 8) != VERSION || frame[10] != kind {
        return Err(FleetPowRuntimeError::InvalidFrame("wire header"));
    }
    Ok(())
}

const fn site_to_wire(site: FleetPowSite) -> u8 {
    match site {
        FleetPowSite::Interaction => 1,
        FleetPowSite::Query => 2,
    }
}

fn site_from_wire(value: u8) -> Result<FleetPowSite, FleetPowRuntimeError> {
    match value {
        1 => Ok(FleetPowSite::Interaction),
        2 => Ok(FleetPowSite::Query),
        _ => Err(FleetPowRuntimeError::InvalidFrame("site")),
    }
}

fn put_u16(frame: &mut [u8], offset: usize, value: u16) {
    frame[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(frame: &mut [u8], offset: usize, value: u32) {
    frame[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(frame: &mut [u8], offset: usize, value: u64) {
    frame[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(frame: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(frame[offset..offset + 2].try_into().expect("fixed frame"))
}

fn get_u32(frame: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(frame[offset..offset + 4].try_into().expect("fixed frame"))
}

fn get_u64(frame: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(frame[offset..offset + 8].try_into().expect("fixed frame"))
}

#[cfg(test)]
mod tests;
