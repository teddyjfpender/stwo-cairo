//! Exact PCS/FRI geometry derived from one Cairo claim.
//!
//! This is the protocol half of the proof-plan IR.  It contains no CUDA calls:
//! the claim, fixed-table binding, PCS security parameters, commitment order and
//! FRI folding schedule are validated before a device arena is allocated.

use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::CommitWorkspaceConfig;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};

use crate::arena_plan::{
    CommitmentGeometry, CommitmentTreeId, DecommitStrategy, ProofEpoch, ProtocolGeometry,
    ProtocolIdentity,
};
use crate::plan::ProofPlan;

/// Stable identity for the ordinary Blake2s channel/proof format used by the
/// Starknet block benchmark.  It is intentionally not a Rust `TypeId` or hash.
pub const BLAKE2S_MERKLE_CHANNEL_TAG: u64 = 0x424c_414b_4532_5331;

/// Runtime choices that affect graph topology or opening residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolPlanPolicy {
    pub channel_tag: u64,
    /// Hash of the AOT manifest actually loaded by the runtime.  Zero is never
    /// accepted: graph reuse without a bound kernel library is unsafe.
    pub kernel_manifest_hash: u64,
    pub decommit_strategy: DecommitStrategy,
    pub unretained_bottom_layers: u32,
    pub max_fused_tail_levels: u32,
}

impl ProtocolPlanPolicy {
    pub const fn starknet_blake2s(kernel_manifest_hash: u64) -> Self {
        Self {
            channel_tag: BLAKE2S_MERKLE_CHANNEL_TAG,
            kernel_manifest_hash,
            decommit_strategy: DecommitStrategy::RecomputeQueriedLde,
            unretained_bottom_layers: 4,
            max_fused_tail_levels: 12,
        }
    }

    /// Bind the plan to the AOT pack embedded in the running binary. Stub builds
    /// and binaries with no generated pack are rejected before CUDA allocation.
    pub fn loaded_starknet_blake2s() -> Result<Self, ProtocolPlanError> {
        let hash = stwo_backend_cuda::aot::loaded_manifest_hash();
        if hash == 0 {
            return Err(ProtocolPlanError::UnboundKernelManifest);
        }
        Ok(Self::starknet_blake2s(hash))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtocolPlanError {
    UnboundChannel,
    UnboundKernelManifest,
    InvalidClaimTreeCount(usize),
    EmptyClaimTree(usize),
    EmptyPreprocessedTrace,
    InvalidLiftingLogSize {
        lifting: u32,
        required: u32,
    },
    InvalidFriGeometry {
        lifting: u32,
        first_fold: u32,
        last_domain: u32,
    },
    SizeOverflow,
}

impl core::fmt::Display for ProtocolPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid CUDA protocol plan: {self:?}")
    }
}

impl std::error::Error for ProtocolPlanError {}

/// Build the exact protocol geometry consumed by [`crate::arena_plan::ProofArenaPlan`].
///
/// Claim tree order is protocol data: base then interaction.  Composition is
/// eight M31 coordinate polynomials (two secure halves), and the persistent
/// preprocessed tree participates in openings without acquiring a per-proof
/// commit workspace.
pub fn plan_protocol_geometry(
    proof_plan: &ProofPlan,
    claim: &CairoClaim,
    preprocessed_trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
    policy: ProtocolPlanPolicy,
) -> Result<ProtocolGeometry, ProtocolPlanError> {
    let claim_log_sizes = claim.log_sizes();
    plan_protocol_from_logs(
        proof_plan,
        &claim_log_sizes,
        preprocessed_trace,
        pcs,
        include_all_preprocessed_columns,
        policy,
    )
}

fn plan_protocol_from_logs(
    proof_plan: &ProofPlan,
    claim_log_sizes: &[Vec<u32>],
    preprocessed_trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
    policy: ProtocolPlanPolicy,
) -> Result<ProtocolGeometry, ProtocolPlanError> {
    if policy.channel_tag == 0 {
        return Err(ProtocolPlanError::UnboundChannel);
    }
    if policy.kernel_manifest_hash == 0 {
        return Err(ProtocolPlanError::UnboundKernelManifest);
    }
    if claim_log_sizes.len() != 2 {
        return Err(ProtocolPlanError::InvalidClaimTreeCount(
            claim_log_sizes.len(),
        ));
    }
    for (tree, logs) in claim_log_sizes.iter().enumerate() {
        if logs.is_empty() {
            return Err(ProtocolPlanError::EmptyClaimTree(tree));
        }
    }
    let preprocessed_logs = preprocessed_trace.log_sizes();
    if preprocessed_logs.is_empty() {
        return Err(ProtocolPlanError::EmptyPreprocessedTrace);
    }

    let blowup = pcs.fri_config.log_blowup_factor;
    let max_trace_log = claim_log_sizes
        .iter()
        .flatten()
        .copied()
        .chain(preprocessed_logs.iter().copied())
        .max()
        .ok_or(ProtocolPlanError::EmptyPreprocessedTrace)?;
    let required_lifting = max_trace_log
        .checked_add(blowup.max(1))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let lifting = pcs.lifting_log_size.unwrap_or(required_lifting);
    if lifting < required_lifting {
        return Err(ProtocolPlanError::InvalidLiftingLogSize {
            lifting,
            required: required_lifting,
        });
    }

    let tree_lifting = |logs: &[u32]| -> Result<u32, ProtocolPlanError> {
        let max = logs
            .iter()
            .copied()
            .max()
            .ok_or(ProtocolPlanError::EmptyPreprocessedTrace)?;
        let required = max
            .checked_add(blowup)
            .ok_or(ProtocolPlanError::SizeOverflow)?;
        let actual = pcs.lifting_log_size.unwrap_or(required);
        if actual < required {
            return Err(ProtocolPlanError::InvalidLiftingLogSize {
                lifting: actual,
                required,
            });
        }
        Ok(actual)
    };

    let preprocessed_lifting = tree_lifting(&preprocessed_logs)?;
    let base_lifting = tree_lifting(&claim_log_sizes[0])?;
    let interaction_lifting = tree_lifting(&claim_log_sizes[1])?;
    let composition_coefficient_log = lifting
        .checked_sub(blowup)
        .ok_or(ProtocolPlanError::SizeOverflow)?;

    let commit_config = |tree_lifting| CommitWorkspaceConfig {
        log_blowup_factor: blowup,
        lifting_log_size: tree_lifting,
        unretained_bottom_layers: policy.unretained_bottom_layers,
        max_fused_tail_levels: policy.max_fused_tail_levels,
    };
    let commitments = vec![
        CommitmentGeometry {
            id: CommitmentTreeId::Base,
            created: ProofEpoch::BaseCommit,
            config: commit_config(base_lifting),
            grouped_column_log_sizes: canonical_commit_groups(&claim_log_sizes[0]),
        },
        CommitmentGeometry {
            id: CommitmentTreeId::Interaction,
            created: ProofEpoch::InteractionCommit,
            config: commit_config(interaction_lifting),
            grouped_column_log_sizes: canonical_commit_groups(&claim_log_sizes[1]),
        },
        CommitmentGeometry {
            id: CommitmentTreeId::Composition,
            created: ProofEpoch::CompositionCommit,
            config: commit_config(lifting),
            grouped_column_log_sizes: vec![vec![composition_coefficient_log; 8]],
        },
    ];

    let fri_layer_log_sizes = fri_merkle_log_sizes(lifting, pcs.fri_config)?;
    let opened_tree_log_sizes = vec![
        preprocessed_lifting,
        base_lifting,
        interaction_lifting,
        lifting,
    ];
    let total_opened_columns = preprocessed_logs
        .len()
        .checked_add(claim_log_sizes[0].len())
        .and_then(|total| total.checked_add(claim_log_sizes[1].len()))
        .and_then(|total| total.checked_add(8))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let proof_capacity_words = proof_capacity_words(
        pcs,
        total_opened_columns,
        &opened_tree_log_sizes,
        &fri_layer_log_sizes,
    )?;
    let preprocessed_binding_hash =
        preprocessed_binding_hash(preprocessed_trace, pcs, include_all_preprocessed_columns);

    Ok(ProtocolGeometry {
        identity: ProtocolIdentity::from_pcs(
            pcs,
            policy.channel_tag,
            proof_plan.relation_graph_hash,
            preprocessed_binding_hash,
            policy.kernel_manifest_hash,
            policy.decommit_strategy,
        ),
        max_domain_log_size: lifting,
        lifting_log_size: lifting,
        n_queries: pcs.fri_config.n_queries,
        total_opened_columns,
        proof_capacity_words,
        commitments,
        opened_tree_log_sizes,
        fri_layer_log_sizes,
    })
}

/// Stable ascending leaf order, split into full 16-column Blake2s blocks and
/// one final block.  Sorting is stable so equal-log columns retain claim order.
fn canonical_commit_groups(log_sizes: &[u32]) -> Vec<Vec<u32>> {
    let mut sorted = log_sizes.to_vec();
    sorted.sort_by_key(|log| *log);
    sorted.chunks(16).map(<[u32]>::to_vec).collect()
}

/// Merkle leaf heights for the first and inner FRI commitments, in transcript
/// order.  Packed leaves remove two bits.  A final one-bit fold can therefore
/// produce the same Merkle height as its predecessor; order, not strict height,
/// identifies the graph segment.
pub fn fri_merkle_log_sizes(
    lifting_log_size: u32,
    config: FriConfig,
) -> Result<Vec<u32>, ProtocolPlanError> {
    let last_domain_log = config
        .log_last_layer_degree_bound
        .checked_add(config.log_blowup_factor)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    if config.fold_step == 0
        || lifting_log_size <= config.fold_step
        || lifting_log_size - config.fold_step < last_domain_log
    {
        return Err(ProtocolPlanError::InvalidFriGeometry {
            lifting: lifting_log_size,
            first_fold: config.fold_step,
            last_domain: last_domain_log,
        });
    }
    let packed_log = |domain_log: u32, fold_step: u32| {
        if fold_step > 1 && domain_log >= 2 {
            domain_log - 2
        } else {
            domain_log
        }
    };

    let mut output = vec![packed_log(lifting_log_size, config.fold_step)];
    let mut line_log = lifting_log_size - config.fold_step;
    while line_log > last_domain_log {
        let step = config.fold_step.min(line_log - last_domain_log);
        output.push(packed_log(line_log, step));
        line_log -= step;
    }
    Ok(output)
}

fn proof_capacity_words(
    pcs: &PcsConfig,
    total_opened_columns: usize,
    opened_tree_log_sizes: &[u32],
    fri_layer_log_sizes: &[u32],
) -> Result<usize, ProtocolPlanError> {
    let queries = pcs.fri_config.n_queries;
    let queried_values = queries
        .checked_mul(total_opened_columns)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let path_height = opened_tree_log_sizes
        .iter()
        .chain(fri_layer_log_sizes)
        .try_fold(0usize, |sum, &log| {
            sum.checked_add(log as usize)
                .ok_or(ProtocolPlanError::SizeOverflow)
        })?;
    let auth_paths = queries
        .checked_mul(path_height)
        .and_then(|words| words.checked_mul(8))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let fold_width = 1usize
        .checked_shl(pcs.fri_config.fold_step)
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let fri_witness = queries
        .checked_mul(fri_layer_log_sizes.len())
        .and_then(|words| words.checked_mul(fold_width))
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    // Sixteen secure mask points per column is a deliberately conservative
    // serialization bound; actual Cairo masks are smaller.
    let sampled_values = total_opened_columns
        .checked_mul(16)
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    let last_layer = 1usize
        .checked_shl(pcs.fri_config.log_last_layer_degree_bound)
        .and_then(|words| words.checked_mul(4))
        .ok_or(ProtocolPlanError::SizeOverflow)?;
    queried_values
        .checked_add(auth_paths)
        .and_then(|words| words.checked_add(fri_witness))
        .and_then(|words| words.checked_add(sampled_values))
        .and_then(|words| words.checked_add(last_layer))
        .and_then(|words| words.checked_add(4096))
        .ok_or(ProtocolPlanError::SizeOverflow)
}

pub fn preprocessed_binding_hash(
    trace: &PreProcessedTrace,
    pcs: &PcsConfig,
    include_all_preprocessed_columns: bool,
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(b"stwo-cairo-preprocessed-binding-v1\0");
    feed(&[match trace.variant {
        PreProcessedTraceVariant::Canonical => 0,
        PreProcessedTraceVariant::CanonicalWithoutPedersen => 1,
        PreProcessedTraceVariant::CanonicalSmall => 2,
    }]);
    feed(&[u8::from(include_all_preprocessed_columns)]);
    feed(&pcs.fri_config.log_blowup_factor.to_le_bytes());
    feed(&pcs.lifting_log_size.unwrap_or(0).to_le_bytes());
    for (id, log_size) in trace.ids().into_iter().zip(trace.log_sizes()) {
        feed(&(id.id.len() as u64).to_le_bytes());
        feed(id.id.as_bytes());
        feed(&log_size.to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::PcsConfig;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;

    use super::*;
    use crate::plan::ProofPlan;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    fn pcs(fold_step: u32) -> PcsConfig {
        PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, fold_step),
            lifting_log_size: None,
        }
    }

    #[test]
    fn canonical_groups_are_full_blocks_plus_one_tail() {
        let logs: Vec<_> = (0..35).rev().map(|index| 4 + index % 7).collect();
        let groups = canonical_commit_groups(&logs);
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), [16, 16, 3]);
        let flattened: Vec<_> = groups.into_iter().flatten().collect();
        assert!(flattened.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn fri_fold_three_matches_starknet_tree_schedule() {
        assert_eq!(
            fri_merkle_log_sizes(26, FriConfig::new(0, 1, 70, 3)).unwrap(),
            vec![24, 21, 18, 15, 12, 9, 6, 3, 2]
        );
    }

    #[test]
    fn packed_to_scalar_transition_can_repeat_a_merkle_height() {
        assert_eq!(
            fri_merkle_log_sizes(8, FriConfig::new(0, 1, 70, 2)).unwrap(),
            vec![6, 4, 2, 2]
        );
    }

    #[test]
    fn exact_claim_logs_produce_all_four_opening_trees() {
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let preprocessed = PreProcessedTrace::canonical();
        let geometry = plan_protocol_from_logs(
            &plan,
            &[vec![19; 33], vec![19; 17]],
            &preprocessed,
            &pcs(3),
            false,
            ProtocolPlanPolicy::starknet_blake2s(0x1234),
        )
        .unwrap();
        assert_eq!(geometry.commitments.len(), 3);
        assert_eq!(geometry.opened_tree_log_sizes.len(), 4);
        assert_eq!(
            geometry.total_opened_columns,
            preprocessed.log_sizes().len() + 58
        );
        assert_eq!(
            geometry.identity.relation_graph_hash,
            plan.relation_graph_hash
        );
        assert_ne!(geometry.identity.preprocessed_binding_hash, 0);
    }

    #[test]
    fn fixed_table_policy_and_manifest_are_part_of_identity() {
        let trace = PreProcessedTrace::canonical_small();
        let config = pcs(3);
        assert_ne!(
            preprocessed_binding_hash(&trace, &config, false),
            preprocessed_binding_hash(&trace, &config, true)
        );
        let shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let plan =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        assert_eq!(
            plan_protocol_from_logs(
                &plan,
                &[vec![18], vec![18]],
                &trace,
                &config,
                false,
                ProtocolPlanPolicy::starknet_blake2s(0),
            ),
            Err(ProtocolPlanError::UnboundKernelManifest)
        );
    }
}
