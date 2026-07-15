use serde_json::{json, Value};
use stwo_backend_cuda::{OodsSourceKind, QuotientNumeratorSourceKind};
use stwo_cairo_gpu_prover::arena_plan::{
    CommitmentColumnSource, CommitmentTreeId, OpenedColumnSource, ProofArenaPlan, ProofEpoch,
};

const WORD_BYTES: usize = core::mem::size_of::<u32>();

pub fn json(arena: &ProofArenaPlan) -> Result<Value, &'static str> {
    let fixed = arena
        .commitment(CommitmentTreeId::Preprocessed)
        .ok_or("preprocessed commitment is missing")?;
    let mut full_bytes = 0usize;
    let mut retained_bytes = 0usize;
    let mut direct_bytes = 0usize;
    let mut incremental_bytes = 0usize;
    let mut numerator_materialization_bytes = 0usize;
    let mut decommit_materialization_bytes = 0usize;
    let mut retained_columns = 0usize;
    let mut groups = Vec::with_capacity(fixed.grouped_column_log_sizes.len());

    for group in 0..fixed.grouped_column_log_sizes.len() {
        let logs = &fixed.grouped_column_log_sizes[group];
        let sources = &fixed.grouped_column_sources[group];
        if logs.len() != sources.len() {
            return Err("preprocessed source/log group width drifted");
        }
        let bytes = logs.iter().try_fold(0usize, |total, &log| {
            let evaluation_log = log
                .checked_add(fixed.config.log_blowup_factor)
                .ok_or("preprocessed evaluation log overflow")?;
            total
                .checked_add(
                    1usize
                        .checked_shl(evaluation_log)
                        .and_then(|words| words.checked_mul(WORD_BYTES))
                        .ok_or("preprocessed evaluation byte overflow")?,
                )
                .ok_or("preprocessed evaluation byte total overflow")
        })?;
        full_bytes = full_bytes
            .checked_add(bytes)
            .ok_or("preprocessed full image byte overflow")?;
        let direct = fixed.direct_composition_evaluation_groups[group].is_some();
        let numerator = fixed.numerator_evaluation_groups[group].is_some();
        let decommit = fixed.retained_evaluation_groups[group].is_some();
        let retained = direct || numerator || decommit;
        if retained {
            retained_bytes = retained_bytes
                .checked_add(bytes)
                .ok_or("preprocessed retained byte overflow")?;
            retained_columns += sources.len();
        }
        if direct {
            direct_bytes = direct_bytes
                .checked_add(bytes)
                .ok_or("preprocessed direct byte overflow")?;
        }
        if retained && !direct {
            incremental_bytes = incremental_bytes
                .checked_add(bytes)
                .ok_or("preprocessed incremental byte overflow")?;
        }
        if numerator {
            numerator_materialization_bytes = numerator_materialization_bytes
                .checked_add(bytes)
                .ok_or("preprocessed numerator byte overflow")?;
        }
        if decommit {
            decommit_materialization_bytes = decommit_materialization_bytes
                .checked_add(bytes)
                .ok_or("preprocessed decommit byte overflow")?;
        }
        let ordinals = sources
            .iter()
            .map(|source| match source {
                CommitmentColumnSource::Preprocessed { ordinal } => Ok(*ordinal),
                _ => Err("preprocessed group contains a foreign source"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        groups.push(json!({
            "group": group,
            "column_count": sources.len(),
            "column_ordinals": ordinals,
            "coefficient_log_sizes": logs,
            "evaluation_bytes": bytes,
            "direct_composition": direct,
            "numerator": numerator,
            "decommit": decommit,
            "retained": retained,
            "incremental": retained && !direct,
        }));
    }

    let identity_incremental = arena
        .protocol_identity()
        .fixed_image_incremental_evaluation_bytes;
    if incremental_bytes != identity_incremental {
        return Err("fixed-image receipt disagrees with protocol identity");
    }

    let fixed_ownership = arena
        .late_coefficient_ownership()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.source, OpenedColumnSource::Preprocessed { .. }))
        .collect::<Vec<_>>();
    let mut coefficient_bytes = 0usize;
    let mut warm_coefficient_bytes = 0usize;
    let mut warm_coefficient_columns = 0usize;
    let mut reader_columns = [0usize; 4];
    for entry in &fixed_ownership {
        let OpenedColumnSource::Preprocessed { ordinal } = entry.source else {
            unreachable!();
        };
        let column = arena
            .preprocessed_coefficients()
            .iter()
            .find(|column| column.ordinal == ordinal)
            .ok_or("preprocessed ownership source is absent from the arena")?;
        let bytes = column
            .coefficients
            .len_words
            .checked_mul(WORD_BYTES)
            .ok_or("preprocessed coefficient byte overflow")?;
        coefficient_bytes = coefficient_bytes
            .checked_add(bytes)
            .ok_or("preprocessed coefficient byte total overflow")?;
        if entry.final_consumer != ProofEpoch::Ingest {
            warm_coefficient_columns += 1;
            warm_coefficient_bytes = warm_coefficient_bytes
                .checked_add(bytes)
                .ok_or("warm preprocessed coefficient byte overflow")?;
        }
        reader_columns[0] += usize::from(entry.composition_reads_coefficients);
        reader_columns[1] += usize::from(entry.oods_reads_coefficients);
        reader_columns[2] += usize::from(entry.quotient_reads_coefficients);
        reader_columns[3] += usize::from(entry.decommit_reads_coefficients);
    }
    let retired_warm_coefficient_bytes = coefficient_bytes
        .checked_sub(warm_coefficient_bytes)
        .ok_or("warm coefficient ownership exceeds the full image")?;
    let net_persistent_bytes = i128::try_from(incremental_bytes)
        .map_err(|_| "incremental byte conversion overflow")?
        - i128::try_from(retired_warm_coefficient_bytes)
            .map_err(|_| "retired byte conversion overflow")?;

    let oods_evaluation_columns = arena
        .oods()
        .columns
        .iter()
        .filter(|column| {
            matches!(column.source, OpenedColumnSource::Preprocessed { .. })
                && column.source_kind == OodsSourceKind::Evaluations
        })
        .count();
    let numerator_evaluation_columns = arena
        .quotient_numerator()
        .columns
        .iter()
        .filter(|column| {
            matches!(column.source, OpenedColumnSource::Preprocessed { .. })
                && column.topology.source_kind == QuotientNumeratorSourceKind::Evaluation
        })
        .count();

    Ok(json!({
        "schema": "stwo.fixed-image-retention.v1",
        "total_preprocessed_columns": fixed_ownership.len(),
        "full_preprocessed_evaluation_bytes": full_bytes,
        "total_preprocessed_retained_evaluation_bytes": retained_bytes,
        "existing_direct_preprocessed_evaluation_bytes": direct_bytes,
        "incremental_fixed_image_evaluation_bytes": incremental_bytes,
        "retained_preprocessed_columns": retained_columns,
        "full_fixed_image": retained_columns == fixed_ownership.len(),
        "groups": groups,
        "warm_consumers": {
            "oods_evaluation_columns": oods_evaluation_columns,
            "numerator_evaluation_columns": numerator_evaluation_columns,
            "numerator_whole_lde_materialization_bytes_retired": numerator_materialization_bytes,
            "decommit_whole_lde_materialization_bytes_retired": decommit_materialization_bytes,
        },
        "coefficient_ownership": {
            "cold_commit_input_bytes": coefficient_bytes,
            "warm_persistent_bytes": warm_coefficient_bytes,
            "warm_persistent_bytes_retired": retired_warm_coefficient_bytes,
            "remaining_coefficient_backed_columns": warm_coefficient_columns,
            "reader_columns": {
                "composition": reader_columns[0],
                "oods": reader_columns[1],
                "quotient_numerator": reader_columns[2],
                "decommit": reader_columns[3],
            },
            "incremental_lde_minus_retired_warm_coefficients_bytes": net_persistent_bytes,
            "scope": "exact logical ownership bytes; arena packing and the complete physical ledger remain the admission authority",
        },
    }))
}
