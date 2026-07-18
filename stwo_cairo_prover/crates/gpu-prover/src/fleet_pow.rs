//! Exact cooperative partition of the existing SIMD proof-of-work lattice.

use std::collections::BTreeMap;

use stwo_backend_cuda::{
    pow_index_to_nonce, Blake2sPowFleetAttempt, POW_GRIND_LOW_BITS, POW_INDEX_LIMIT,
    POW_THREADS_PER_BLOCK,
};

use crate::fleet_plan::WorkerId;

// Mirrors resident_pow.cu: SIMD requires `hi < 2^31 - 1`, so this is the
// exclusive end of the `(hi << POW_GRIND_LOW_BITS) | low` index lattice.
const MAX_INDEX_EXCLUSIVE: u64 = POW_INDEX_LIMIT;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FleetPowSite {
    Interaction,
    Query,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetPowSchedule {
    pub interaction: FleetPowPlan,
    pub query: FleetPowPlan,
}

impl FleetPowSchedule {
    pub fn validate(self, rank_count: usize) -> Result<(), FleetPowError> {
        self.interaction.validate(rank_count)?;
        self.query.validate(rank_count)
    }

    pub const fn plan(self, site: FleetPowSite) -> FleetPowPlan {
        match site {
            FleetPowSite::Interaction => self.interaction,
            FleetPowSite::Query => self.query,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetPowPlan {
    /// Kernel-local workers participating on each rank (blocks × threads).
    pub workers_per_rank: u32,
    /// Contiguous global lattice indices proven complete before retrying.
    pub indices_per_attempt: u64,
}

impl FleetPowPlan {
    pub fn validate(self, rank_count: usize) -> Result<(), FleetPowError> {
        if self.workers_per_rank == 0
            || self.indices_per_attempt == 0
            || self.indices_per_attempt > MAX_INDEX_EXCLUSIVE
            || rank_count == 0
            || rank_count > u16::MAX as usize + 1
        {
            return Err(FleetPowError::InvalidGeometry);
        }
        u64::from(self.workers_per_rank)
            .checked_mul(u64::try_from(rank_count).map_err(|_| FleetPowError::SizeOverflow)?)
            .ok_or(FleetPowError::SizeOverflow)?;
        Ok(())
    }

    /// Global index for the persistent rank/worker stripe. These residue
    /// classes are disjoint and cover the SIMD lattice index space.
    pub fn search_index(
        self,
        rank_count: usize,
        rank: WorkerId,
        local_worker: u32,
        iteration: u64,
    ) -> Result<u64, FleetPowError> {
        self.validate(rank_count)?;
        if rank.0 as usize >= rank_count || local_worker >= self.workers_per_rank {
            return Err(FleetPowError::InvalidWorker);
        }
        let width = u64::from(self.workers_per_rank);
        let stride = width
            .checked_mul(u64::try_from(rank_count).map_err(|_| FleetPowError::SizeOverflow)?)
            .ok_or(FleetPowError::SizeOverflow)?;
        let index = u64::from(rank.0)
            .checked_mul(width)
            .and_then(|base| base.checked_add(u64::from(local_worker)))
            .and_then(|base| {
                iteration
                    .checked_mul(stride)
                    .and_then(|step| base.checked_add(step))
            })
            .ok_or(FleetPowError::SizeOverflow)?;
        (index < MAX_INDEX_EXCLUSIVE)
            .then_some(index)
            .ok_or(FleetPowError::SearchExhausted)
    }

    pub fn nonce(
        self,
        rank_count: usize,
        rank: WorkerId,
        local_worker: u32,
        iteration: u64,
    ) -> Result<u64, FleetPowError> {
        Ok(pow_index_to_nonce(self.search_index(
            rank_count,
            rank,
            local_worker,
            iteration,
        )?))
    }

    /// Returns the attempt's half-open tile. The final tile is truncated at
    /// the resident-kernel limit; an ordinal starting beyond it is exhausted.
    pub fn attempt_bounds(self, attempt_ordinal: u64) -> Result<(u64, u64), FleetPowError> {
        if self.indices_per_attempt == 0 || self.indices_per_attempt > MAX_INDEX_EXCLUSIVE {
            return Err(FleetPowError::InvalidGeometry);
        }
        let start = attempt_ordinal
            .checked_mul(self.indices_per_attempt)
            .ok_or(FleetPowError::SizeOverflow)?;
        if start >= MAX_INDEX_EXCLUSIVE {
            return Err(FleetPowError::SearchExhausted);
        }
        let end = start
            .checked_add(self.indices_per_attempt)
            .ok_or(FleetPowError::SizeOverflow)?
            .min(MAX_INDEX_EXCLUSIVE);
        Ok((start, end))
    }

    /// Bind one shared attempt directly to the production CUDA rank kernel.
    /// Every worker tile must be derived from the returned value.
    pub fn attempt(
        self,
        rank_count: usize,
        attempt_ordinal: u64,
    ) -> Result<Blake2sPowFleetAttempt, FleetPowError> {
        self.validate(rank_count)?;
        if self.workers_per_rank % POW_THREADS_PER_BLOCK != 0 {
            return Err(FleetPowError::InvalidGeometry);
        }
        let (start, end) = self.attempt_bounds(attempt_ordinal)?;
        Blake2sPowFleetAttempt::new(
            u32::try_from(rank_count).map_err(|_| FleetPowError::SizeOverflow)?,
            start,
            end,
            self.workers_per_rank / POW_THREADS_PER_BLOCK,
        )
        .map_err(|_| FleetPowError::InvalidGeometry)
    }

    /// Wait for exactly one completed result from every rank, validate its
    /// generation and ownership, and return the canonical global minimum.
    /// `None` means this complete attempt tile contained no winner.
    pub fn reduce_attempt(
        self,
        site: FleetPowSite,
        rank_count: usize,
        plan_identity: [u8; 32],
        proof_generation: u64,
        attempt_ordinal: u64,
        receipts: &[PowRankReceipt],
        is_valid_nonce: impl Fn(u64) -> bool,
    ) -> Result<Option<u64>, FleetPowError> {
        self.validate(rank_count)?;
        let (start, end) = self.attempt_bounds(attempt_ordinal)?;
        if receipts.len() != rank_count {
            return Err(FleetPowError::IncompleteReceipts);
        }
        let mut seen = vec![false; rank_count];
        let mut winner = None;
        for receipt in receipts {
            let rank = receipt.rank.0 as usize;
            if receipt.site != site
                || receipt.plan_identity != plan_identity
                || receipt.proof_generation != proof_generation
                || receipt.attempt_ordinal != attempt_ordinal
                || rank >= rank_count
                || std::mem::replace(&mut seen[rank], true)
            {
                return Err(FleetPowError::InvalidReceipt(receipt.rank));
            }
            if let Some(nonce) = receipt.candidate_nonce {
                let index = nonce_to_index(nonce)?;
                if index < start
                    || index >= end
                    || !self.index_belongs_to_rank(rank_count, receipt.rank, index)?
                    || !is_valid_nonce(nonce)
                {
                    return Err(FleetPowError::InvalidReceipt(receipt.rank));
                }
                winner = Some(winner.map_or(nonce, |current: u64| current.min(nonce)));
            }
        }
        if seen.iter().any(|&present| !present) {
            return Err(FleetPowError::IncompleteReceipts);
        }
        Ok(winner)
    }

    /// Reduce dense, wait-all attempt tiles. A receipt attests that its rank
    /// completed every owned index in the tile and reports its local minimum,
    /// if any. The first tile containing a candidate is therefore globally
    /// canonical; later tiles are rejected rather than ignored.
    /// Exhaustive host oracle for correctness gates, not the hot fleet path.
    /// A qualified runtime will use the same result after its kernel proves
    /// local-minimum/exhaustion parity without rescanning the tile on the CPU.
    pub fn verify_winner(
        self,
        site: FleetPowSite,
        rank_count: usize,
        plan_identity: [u8; 32],
        proof_generation: u64,
        receipts: &[PowRankReceipt],
        is_valid_nonce: impl Fn(u64) -> bool,
    ) -> Result<u64, FleetPowError> {
        self.validate(rank_count)?;
        let mut indexed = BTreeMap::new();
        for receipt in receipts {
            let key = (receipt.attempt_ordinal, receipt.rank);
            if receipt.site != site
                || receipt.plan_identity != plan_identity
                || receipt.proof_generation != proof_generation
                || receipt.rank.0 as usize >= rank_count
                || indexed.insert(key, receipt).is_some()
            {
                return Err(FleetPowError::InvalidReceipt(receipt.rank));
            }
        }
        let max_attempt = indexed
            .keys()
            .map(|(attempt, _)| *attempt)
            .max()
            .ok_or(FleetPowError::IncompleteReceipts)?;
        for attempt in 0..=max_attempt {
            let (start, end) = self.attempt_bounds(attempt)?;
            let mut local_minima = vec![None; rank_count];
            for index in start..end {
                let rank = self.rank_for_index(rank_count, index)?;
                if local_minima[rank].is_none() {
                    let nonce = pow_index_to_nonce(index);
                    if is_valid_nonce(nonce) {
                        local_minima[rank] = Some(nonce);
                    }
                }
            }
            let mut winner = None;
            for rank in 0..rank_count {
                let rank =
                    WorkerId(u16::try_from(rank).map_err(|_| FleetPowError::InvalidGeometry)?);
                let receipt = indexed
                    .get(&(attempt, rank))
                    .ok_or(FleetPowError::IncompleteReceipts)?;
                let candidate = receipt
                    .candidate_nonce
                    .map(|nonce| nonce_to_index(nonce).map(|index| (nonce, index)))
                    .transpose()?;
                if receipt.candidate_nonce != local_minima[rank.0 as usize] {
                    return Err(FleetPowError::InvalidReceipt(rank));
                }
                if let Some((nonce, index)) = candidate {
                    if index < start
                        || index >= end
                        || !self.index_belongs_to_rank(rank_count, rank, index)?
                    {
                        return Err(FleetPowError::InvalidReceipt(rank));
                    }
                    winner = Some(winner.map_or(nonce, |current: u64| current.min(nonce)));
                }
            }
            if let Some(winner) = winner {
                if attempt != max_attempt {
                    return Err(FleetPowError::TrailingAttempts);
                }
                return Ok(winner);
            }
        }
        Err(FleetPowError::NoWinner)
    }

    fn index_belongs_to_rank(
        self,
        rank_count: usize,
        rank: WorkerId,
        index: u64,
    ) -> Result<bool, FleetPowError> {
        let width = u64::from(self.workers_per_rank);
        let stride = width
            .checked_mul(u64::try_from(rank_count).map_err(|_| FleetPowError::SizeOverflow)?)
            .ok_or(FleetPowError::SizeOverflow)?;
        Ok((index % stride) / width == u64::from(rank.0))
    }

    fn rank_for_index(self, rank_count: usize, index: u64) -> Result<usize, FleetPowError> {
        let width = u64::from(self.workers_per_rank);
        let stride = width
            .checked_mul(u64::try_from(rank_count).map_err(|_| FleetPowError::SizeOverflow)?)
            .ok_or(FleetPowError::SizeOverflow)?;
        usize::try_from((index % stride) / width).map_err(|_| FleetPowError::SizeOverflow)
    }
}

fn nonce_to_index(nonce: u64) -> Result<u64, FleetPowError> {
    let low = nonce & u64::from(u32::MAX);
    if low >= 1u64 << POW_GRIND_LOW_BITS {
        return Err(FleetPowError::OutsideSimdLattice);
    }
    (nonce >> 32)
        .checked_shl(POW_GRIND_LOW_BITS)
        .and_then(|high| high.checked_add(low))
        .filter(|&index| index < MAX_INDEX_EXCLUSIVE)
        .ok_or(FleetPowError::OutsideSimdLattice)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowRankReceipt {
    pub site: FleetPowSite,
    pub plan_identity: [u8; 32],
    pub rank: WorkerId,
    pub proof_generation: u64,
    pub attempt_ordinal: u64,
    pub candidate_nonce: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FleetPowError {
    InvalidGeometry,
    InvalidWorker,
    InvalidReceipt(WorkerId),
    IncompleteReceipts,
    TrailingAttempts,
    NoWinner,
    OutsideSimdLattice,
    SearchExhausted,
    SizeOverflow,
}

impl core::fmt::Display for FleetPowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid fleet PoW partition: {self:?}")
    }
}

impl std::error::Error for FleetPowError {}
