//! Exact admission and receipts for planned commitment programs.

use stwo_backend_cuda::{
    CommitProgram, CommitProgramTraffic, CompactDomainOperation, CompactDomainProgram,
    DomainCooperativeOperation, DomainCooperativeProgram, ProgressiveCommitStorageMode,
};
use stwo_cairo_gpu_prover::arena_plan::{CommitmentTreeId, PlannedCommitment, ResidentBackend};

fn tree_name(tree: CommitmentTreeId) -> String {
    match tree {
        CommitmentTreeId::Preprocessed => "preprocessed".to_owned(),
        CommitmentTreeId::Base => "base".to_owned(),
        CommitmentTreeId::Interaction => "interaction".to_owned(),
        CommitmentTreeId::Composition => "composition".to_owned(),
        CommitmentTreeId::Fri(layer) => format!("fri-{layer}"),
    }
}

fn retained_column_closure(commitment: &PlannedCommitment) -> (usize, usize, bool) {
    let total = commitment
        .grouped_column_log_sizes
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let retained = commitment
        .evaluation_output_groups
        .iter()
        .filter_map(Option::as_ref)
        .map(Vec::len)
        .sum::<usize>();
    (total, retained, total == retained)
}

pub(crate) fn validate_dynamic_commitment_leaf_programs(
    selected_backend: ResidentBackend,
    commitments: &[PlannedCommitment],
) -> Result<(), String> {
    let expected = [
        CommitmentTreeId::Preprocessed,
        CommitmentTreeId::Base,
        CommitmentTreeId::Interaction,
        CommitmentTreeId::Composition,
    ];
    if commitments.len() != expected.len()
        || commitments
            .iter()
            .map(|commitment| commitment.id)
            .ne(expected)
    {
        return Err("commitment program receipt is not the canonical four-tree prefix".to_owned());
    }

    for commitment in commitments {
        let dynamic = matches!(
            commitment.id,
            CommitmentTreeId::Base | CommitmentTreeId::Interaction | CommitmentTreeId::Composition
        );
        validate_program_tuple(
            selected_backend,
            commitment.id,
            commitment.commit_program.as_ref(),
            commitment.domain_cooperative_program.as_ref(),
            commitment.compact_domain_program.as_ref(),
        )?;
        match selected_backend {
            ResidentBackend::LegacyResident => validate_legacy_commitment(commitment)?,
            ResidentBackend::ReplacementV1 => validate_replacement_commitment(commitment, dynamic)?,
        }
    }
    Ok(())
}

fn validate_legacy_commitment(commitment: &PlannedCommitment) -> Result<(), String> {
    if commitment.storage_mode == ProgressiveCommitStorageMode::Separate {
        return Ok(());
    }
    Err(format!(
        "{} legacy commitment is not separate",
        tree_name(commitment.id)
    ))
}

fn validate_program_tuple(
    selected_backend: ResidentBackend,
    tree: CommitmentTreeId,
    base: Option<&CommitProgram>,
    domain: Option<&DomainCooperativeProgram>,
    compact: Option<&CompactDomainProgram>,
) -> Result<(), String> {
    let name = tree_name(tree);
    let dynamic = matches!(
        tree,
        CommitmentTreeId::Base | CommitmentTreeId::Interaction | CommitmentTreeId::Composition
    );
    match selected_backend {
        ResidentBackend::LegacyResident => {
            if base.is_some() || domain.is_some() || compact.is_some() {
                return Err(format!(
                    "{name} legacy commitment owns a replacement leaf program"
                ));
            }
            Ok(())
        }
        ResidentBackend::ReplacementV1 => {
            let base = base.ok_or_else(|| {
                format!("{name} replacement commitment is missing its base program")
            })?;
            if !dynamic {
                if domain.is_some() || compact.is_some() {
                    return Err(format!(
                        "{name} fixed commitment owns a dynamic leaf program"
                    ));
                }
                return Ok(());
            }
            let domain = domain.ok_or_else(|| {
                format!("{name} compact commitment is missing its domain program")
            })?;
            domain.validate_against(base).map_err(|error| {
                format!("{name} domain program/base identity mismatch: {error:?}")
            })?;
            let compact = compact.ok_or_else(|| {
                format!("{name} replacement commitment is missing its compact program")
            })?;
            compact.validate_against(base, domain).map_err(|error| {
                format!("{name} compact program/base/domain identity mismatch: {error:?}")
            })
        }
    }
}

fn validate_replacement_commitment(
    commitment: &PlannedCommitment,
    dynamic: bool,
) -> Result<(), String> {
    if commitment.storage_mode != ProgressiveCommitStorageMode::InPlaceSlab {
        return Err(format!(
            "{} replacement commitment is not in-place",
            tree_name(commitment.id)
        ));
    }
    if !dynamic {
        return Ok(());
    }

    let (total, retained, all_retained) = retained_column_closure(commitment);
    if !all_retained {
        return Err(format!(
            "{} Mode-A commitment retained {retained}/{total} columns",
            tree_name(commitment.id)
        ));
    }
    Ok(())
}

pub(crate) fn dynamic_commitment_leaf_program_receipts(
    commitments: &[PlannedCommitment],
) -> Vec<serde_json::Value> {
    commitments.iter().map(commitment_receipt).collect()
}

fn commitment_receipt(commitment: &PlannedCommitment) -> serde_json::Value {
    let (total_columns, retained_columns, all_columns_retained) =
        retained_column_closure(commitment);
    let base = commitment.commit_program.as_ref();
    let domain = commitment.domain_cooperative_program.as_ref();
    let compact = commitment.compact_domain_program.as_ref();
    let exact_base_validation = domain
        .zip(base)
        .map(|(program, base)| program.validate_against(base).is_ok());
    serde_json::json!({
        "tree": tree_name(commitment.id),
        "schedule_identity": schedule_identity(commitment.id, domain, compact),
        "storage_mode": storage_mode_name(commitment.storage_mode),
        "total_columns": total_columns,
        "retained_columns": retained_columns,
        "all_columns_retained": all_columns_retained,
        "base_program_present": base.is_some(),
        "base_program_cache_key": base.map(|base| format!("{:016x}", base.identity().cache_key)),
        "base_program_operations": base.map(|base| base.steps().len()),
        "enabled": domain.is_some() || compact.is_some(),
        "domain_cooperative_program_present": domain.is_some(),
        "compact_domain_program_present": compact.is_some(),
        "domain_cooperative_program": domain.map(|program| {
            let resource = program.resource_model();
            let comparison = program.comparison();
            serde_json::json!({
                "identity": "retained-domain-cooperative-mode-a-v1",
                "cache_key": format!("{:016x}", program.cache_key()),
                "slab_words": program.slab_words(),
                "operations": operation_receipt(program),
                "resource_model": {
                    "threads_per_row": resource.threads_per_row,
                    "rows_per_block": resource.rows_per_block,
                    "threads_per_block": resource.threads_per_block,
                    "launch_bounds_min_blocks_per_sm": resource.launch_bounds_min_blocks_per_sm,
                    "persistent_state_words_per_thread": resource.persistent_state_words_per_thread,
                    "register_ceiling_per_thread": resource.register_ceiling_per_thread,
                },
                "exact_base_validation": exact_base_validation,
                "comparison": {
                    "current_leaf_traffic": traffic_receipt(comparison.current_leaf_traffic),
                    "replacement_leaf_traffic": traffic_receipt(comparison.replacement_leaf_traffic),
                    "retained_evaluation_reread_bytes": comparison.retained_evaluation_reread_bytes,
                    "current_retained_evaluation_reread_bytes": comparison.current_retained_evaluation_reread_bytes,
                    "incremental_retained_evaluation_reread_bytes": comparison.incremental_retained_evaluation_reread_bytes,
                    "current_state_api_calls": comparison.current_state_api_calls,
                    "replacement_state_api_calls": comparison.replacement_state_api_calls,
                    "current_leaf_compressions": comparison.current_leaf_compressions,
                    "replacement_leaf_compressions": comparison.replacement_leaf_compressions,
                },
            })
        }),
        "compact_domain_program": compact.map(|program| {
            compact_program_receipt(program, base, domain)
        }),
    })
}

fn schedule_identity(
    tree: CommitmentTreeId,
    domain: Option<&DomainCooperativeProgram>,
    compact: Option<&CompactDomainProgram>,
) -> &'static str {
    match (domain.is_some(), compact.is_some()) {
        (false, false) if tree == CommitmentTreeId::Preprocessed => {
            "fixed-preprocessed-qualified-base"
        }
        (false, false) => "legacy-per-batch",
        (true, false) => "retained-domain-cooperative",
        (true, true) => "retained-domain-compact-h8",
        (false, true) => "invalid-compact-without-domain",
    }
}

fn compact_program_receipt(
    program: &CompactDomainProgram,
    base: Option<&CommitProgram>,
    domain: Option<&DomainCooperativeProgram>,
) -> serde_json::Value {
    let comparison = program.comparison();
    let exact_base_domain_validation = base
        .zip(domain)
        .is_some_and(|(base, domain)| program.validate_against(base, domain).is_ok());
    serde_json::json!({
        "identity": "retained-domain-compact-h8-v1",
        "cache_key": format!("{:016x}", program.cache_key()),
        "slab_words": program.slab_words(),
        "operations": compact_operation_receipt(program),
        "exact_base_domain_validation": exact_base_domain_validation,
        "comparison": {
            "current_leaf_traffic": traffic_receipt(comparison.current_leaf_traffic),
            "replacement_leaf_traffic": traffic_receipt(comparison.replacement_leaf_traffic),
            "tail_reconstruction_read_bytes": comparison.tail_reconstruction_read_bytes,
            "current_state_slab_words": comparison.current_state_slab_words,
            "replacement_state_slab_words": comparison.replacement_state_slab_words,
            "state_slab_words_saved": comparison.state_slab_words_saved,
            "current_state_api_calls": comparison.current_state_api_calls,
            "replacement_state_api_calls": comparison.replacement_state_api_calls,
            "current_leaf_compressions": comparison.current_leaf_compressions,
            "replacement_leaf_compressions": comparison.replacement_leaf_compressions,
        },
    })
}

fn operation_receipt(program: &stwo_backend_cuda::DomainCooperativeProgram) -> serde_json::Value {
    let mut lde_batches = 0usize;
    let mut absorb_domain_batches = 0usize;
    let mut state_expands = 0usize;
    let mut finalizations = 0usize;
    for step in program.steps() {
        match step.operation {
            DomainCooperativeOperation::LdeBatch { .. } => lde_batches += 1,
            DomainCooperativeOperation::AbsorbDomainBatch { .. } => absorb_domain_batches += 1,
            DomainCooperativeOperation::StateExpandInPlace { .. } => state_expands += 1,
            DomainCooperativeOperation::FinalizeInPlace { .. } => finalizations += 1,
        }
    }
    serde_json::json!({
        "total": program.steps().len(),
        "lde_batches": lde_batches,
        "absorb_domain_batches": absorb_domain_batches,
        "state_expands": state_expands,
        "finalizations": finalizations,
        "merkle_suffix": program.merkle_suffix().len(),
    })
}

fn compact_operation_receipt(program: &CompactDomainProgram) -> serde_json::Value {
    let mut lde_batches = 0usize;
    let mut absorb_domain_batches = 0usize;
    let mut state_expands = 0usize;
    let mut finalizations = 0usize;
    for step in program.steps() {
        match step.operation {
            CompactDomainOperation::LdeBatch { .. } => lde_batches += 1,
            CompactDomainOperation::AbsorbDomainBatch { .. } => absorb_domain_batches += 1,
            CompactDomainOperation::StateExpandInPlace { .. } => state_expands += 1,
            CompactDomainOperation::FinalizeInPlace { .. } => finalizations += 1,
        }
    }
    serde_json::json!({
        "total": program.steps().len(),
        "lde_batches": lde_batches,
        "absorb_domain_batches": absorb_domain_batches,
        "state_expands": state_expands,
        "finalizations": finalizations,
        "merkle_suffix": program.merkle_suffix().len(),
    })
}

fn traffic_receipt(traffic: CommitProgramTraffic) -> serde_json::Value {
    serde_json::json!({
        "owned_read_bytes": traffic.owned_read_bytes,
        "owned_write_bytes": traffic.owned_write_bytes,
        "kernel_launches": traffic.kernel_launches,
        "device_copies": traffic.device_copies,
    })
}

fn storage_mode_name(mode: ProgressiveCommitStorageMode) -> &'static str {
    match mode {
        ProgressiveCommitStorageMode::Separate => "separate",
        ProgressiveCommitStorageMode::InPlaceSlab => "in-place-slab",
    }
}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::{
        CommitWorkspaceConfig, ProgressiveCommitGeometry, ProgressiveCommitGroupGeometry,
        ProgressiveNttLeafFusionMode,
    };

    use super::*;

    fn programs(
        columns: usize,
    ) -> (
        CommitProgram,
        DomainCooperativeProgram,
        CompactDomainProgram,
    ) {
        let base = CommitProgram::compile(
            CommitWorkspaceConfig {
                log_blowup_factor: 1,
                lifting_log_size: 7,
                unretained_bottom_layers: 4,
                max_fused_tail_levels: 2,
            },
            ProgressiveCommitGeometry {
                lifting_log_size: 7,
                log_blowup_factor: 1,
                groups: vec![ProgressiveCommitGroupGeometry {
                    coefficient_log_sizes: vec![3; columns],
                    retain_evaluations: true,
                }],
            },
            ProgressiveNttLeafFusionMode::Fused16,
            true,
        )
        .unwrap();
        let domain = DomainCooperativeProgram::compile_mode_a(&base).unwrap();
        let compact = CompactDomainProgram::compile(&base, &domain).unwrap();
        (base, domain, compact)
    }

    #[test]
    fn compact_policy_requires_the_exact_dynamic_tuple_and_fixed_exclusion() {
        let (base, domain, compact) = programs(17);
        assert!(validate_program_tuple(
            ResidentBackend::ReplacementV1,
            CommitmentTreeId::Base,
            Some(&base),
            Some(&domain),
            Some(&compact),
        )
        .is_ok());
        assert!(validate_program_tuple(
            ResidentBackend::ReplacementV1,
            CommitmentTreeId::Preprocessed,
            Some(&base),
            None,
            None,
        )
        .is_ok());
        assert!(validate_program_tuple(
            ResidentBackend::ReplacementV1,
            CommitmentTreeId::Base,
            Some(&base),
            Some(&domain),
            None,
        )
        .is_err());
        assert!(validate_program_tuple(
            ResidentBackend::ReplacementV1,
            CommitmentTreeId::Preprocessed,
            Some(&base),
            Some(&domain),
            Some(&compact),
        )
        .is_err());
        assert!(validate_program_tuple(
            ResidentBackend::LegacyResident,
            CommitmentTreeId::Base,
            Some(&base),
            None,
            None,
        )
        .is_err());

        let (other_base, other_domain, _) = programs(18);
        assert!(validate_program_tuple(
            ResidentBackend::ReplacementV1,
            CommitmentTreeId::Base,
            Some(&other_base),
            Some(&other_domain),
            Some(&compact),
        )
        .is_err());
    }

    #[test]
    fn compact_receipt_exposes_identity_capacity_and_exact_validation() {
        let (base, domain, compact) = programs(17);
        let receipt = compact_program_receipt(&compact, Some(&base), Some(&domain));
        assert_eq!(receipt["identity"], "retained-domain-compact-h8-v1");
        assert_eq!(
            receipt["cache_key"],
            format!("{:016x}", compact.cache_key())
        );
        assert_eq!(
            receipt["slab_words"].as_u64(),
            Some(compact.slab_words() as u64)
        );
        assert_eq!(receipt["exact_base_domain_validation"], true);
        assert_eq!(
            receipt["comparison"]["replacement_state_slab_words"].as_u64(),
            Some(compact.comparison().replacement_state_slab_words as u64)
        );
        assert!(
            receipt["comparison"]["state_slab_words_saved"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(receipt["operations"]["finalizations"].as_u64(), Some(1));
        assert_eq!(
            schedule_identity(CommitmentTreeId::Base, Some(&domain), Some(&compact)),
            "retained-domain-compact-h8"
        );
        assert_eq!(
            schedule_identity(CommitmentTreeId::Preprocessed, None, None),
            "fixed-preprocessed-qualified-base"
        );
    }
}
