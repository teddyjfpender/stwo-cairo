//! Address-free exact frontier for retiring dynamic coefficient slabs at commit.
//! Timing and physical credit stay zero until native differential and replay.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use stwo_backend_cuda::{
    b2n_stage_intervals, DecommitSourceMode, DecommitTreeGeometry, InterpolationLaunchMode,
    OodsSourceKind, QuotientNumeratorSourceKind, TraceTreeRole,
};
use stwo_cairo_gpu_prover::arena_plan::{
    BufferPurpose, CommitmentColumnSource, CommitmentTreeId, OpenedColumnSource, PlannedCommitment,
    ProofArenaPlan, ProofEpoch,
};
use stwo_cairo_gpu_prover::prepared_composition::CompositionOutputMode;

#[path = "ntt_lde_intervals.rs"]
mod intervals;
use intervals::{
    bytes, bytes_per_second, chunks, current_n2b_intervals, duplicate_first_intervals, pow2,
    tree_name,
};

const REQUIRED_H100_CUT_NS: u64 = 113_742_773;

#[derive(Default)]
struct Totals {
    columns: u64,
    coefficient_words: u64,
    resident_coefficient_words: u64,
    evaluation_words: u64,
    current_words: u64,
    direct_tail_words: u64,
    duplicate_first_words: u64,
    current_launches: u64,
    direct_tail_launches: u64,
    duplicate_first_launches: u64,
    eliminated_n2b_passes: u64,
}

impl Totals {
    fn add(&mut self, other: &Self) -> Result<(), String> {
        macro_rules! add {
            ($field:ident) => {
                self.$field = self
                    .$field
                    .checked_add(other.$field)
                    .ok_or_else(|| concat!(stringify!($field), " overflow").to_owned())?;
            };
        }
        add!(columns);
        add!(coefficient_words);
        add!(resident_coefficient_words);
        add!(evaluation_words);
        add!(current_words);
        add!(direct_tail_words);
        add!(duplicate_first_words);
        add!(current_launches);
        add!(direct_tail_launches);
        add!(duplicate_first_launches);
        add!(eliminated_n2b_passes);
        Ok(())
    }
}

pub(crate) fn json(arena: &ProofArenaPlan) -> Result<Value, String> {
    validate_late_consumers(arena)?;
    let mut totals = Totals::default();
    let mut trees = Vec::new();
    for tree in [
        CommitmentTreeId::Base,
        CommitmentTreeId::Interaction,
        CommitmentTreeId::Composition,
    ] {
        let commitment = arena
            .commitment(tree)
            .ok_or_else(|| format!("missing {} commitment", tree_name(tree)))?;
        let (receipt, tree_totals) = tree_receipt(arena, commitment)?;
        totals.add(&tree_totals)?;
        trees.push(receipt);
    }

    let current_bytes = bytes(totals.current_words)?;
    let tail_bytes = bytes(totals.direct_tail_words)?;
    let duplicate_bytes = bytes(totals.duplicate_first_words)?;
    let tail_saved = current_bytes
        .checked_sub(tail_bytes)
        .ok_or_else(|| "direct-tail traffic exceeds current traffic".to_owned())?;
    let duplicate_saved = current_bytes
        .checked_sub(duplicate_bytes)
        .ok_or_else(|| "duplicate-first traffic exceeds current traffic".to_owned())?;

    Ok(json!({
        "schema": "stwo-ntt-lde-direct-slab-frontier-v1",
        "status": "dynamic-retirement-implemented-awaiting-native-qualification",
        "native_implementation_present": true,
        "remaining_native_work": [
            "five-stage non-final N2B kernel for evaluation logs 13 and 14",
            "phantom Base/Interaction coefficient Value deletion after byte qualification",
        ],
        "unqualified_gates": [
            "Composition L24/L25 native output, leaf, retained-layer, and root byte identity",
            "sealed adapted-SN direct-vs-forced-fallback ProofArenaPlan total_words receipt",
            "counter-enabled replay timing after byte qualification",
        ],
        "h100_timing_credit_ns": 0,
        "required_h100_cut_ns": REQUIRED_H100_CUT_NS,
        "operator": "N2B_e(ZeroExtend_e(B2N_s(source))) where e=s+1",
        "canonical_output": "existing bit-reversed CanonicCoset(e) M31 byte order",
        "domain_rule": "source and target canonical cosets are not assumed nested",
        "first_stage_identity": "layer e-1 pairs i with i+N; butterfly(c,0,t)=(c,c)",
        "ownership": {
            "dynamic_coefficients_die_at_own_commit": true,
            "dynamic_late_consumers_use_retained_evaluations": true,
            "preprocessed_coefficients_remain_owned": true,
            "physical_arena_reduction_granted_bytes": 0,
            "physical_rule": "logical coefficient retirement requires allocator recoloring before any VRAM credit",
        },
        "trees": trees,
        "totals": {
            "columns": totals.columns,
            "coefficient_words": totals.coefficient_words,
            "coefficient_bytes": bytes(totals.coefficient_words)?,
            "resident_coefficient_words": totals.resident_coefficient_words,
            "resident_coefficient_bytes": bytes(totals.resident_coefficient_words)?,
            "evaluation_words": totals.evaluation_words,
            "evaluation_bytes": bytes(totals.evaluation_words)?,
            "current_pipeline_logical_bytes": current_bytes,
            "current_pipeline_kernel_launches": totals.current_launches,
            "direct_output_tail_zero_logical_bytes": tail_bytes,
            "direct_output_tail_zero_saved_bytes": tail_saved,
            "direct_output_tail_zero_kernel_launches": totals.direct_tail_launches,
            "duplicate_first_stage_logical_bytes": duplicate_bytes,
            "duplicate_first_stage_saved_bytes": duplicate_saved,
            "duplicate_first_stage_kernel_launches": totals.duplicate_first_launches,
            "duplicate_first_stage_eliminated_n2b_passes": totals.eliminated_n2b_passes,
            "minimum_saved_byte_rate_for_required_cut_bytes_per_second":
                bytes_per_second(duplicate_saved, REQUIRED_H100_CUT_NS)?,
        },
        "measurement_gate": {
            "cpu_transform_oracle_cases": 80,
            "cpu_transform_oracle_passed": true,
            "cpu_transform_oracle_scope": "host current-operator equality; not native CUDA",
            "cuda_byte_differential_required": true,
            "counter_enabled_replay_required": true,
            "timing_claim_admitted": false,
        },
    }))
}

fn tree_receipt(
    arena: &ProofArenaPlan,
    commitment: &PlannedCommitment,
) -> Result<(Value, Totals), String> {
    if commitment.config.log_blowup_factor != 1 {
        return Err(format!(
            "{} blowup log is {}, expected 1",
            tree_name(commitment.id),
            commitment.config.log_blowup_factor
        ));
    }
    if commitment.interpolation_mode != InterpolationLaunchMode::StageFusedOutOfPlace {
        return Err(format!(
            "{} is not using stage-fused B2N",
            tree_name(commitment.id)
        ));
    }
    let flattened = canonical_columns(commitment)?;
    validate_retained_outputs(commitment, &flattened)?;
    validate_commit_program(commitment, &flattened)?;
    validate_interpolation(commitment, &flattened)?;
    let resident_coefficient_words =
        validate_coefficient_logical_words(arena, commitment.id, &flattened)?;

    let program = commitment
        .commit_program
        .as_ref()
        .ok_or_else(|| format!("{} has no commit program", tree_name(commitment.id)))?;
    let mut totals = Totals::default();
    let mut batches = Vec::new();
    for batch in &program.requirements().leaves.plan.lde_batches {
        let columns = u64::try_from(batch.columns.len())
            .map_err(|_| "LDE column count does not fit u64".to_owned())?;
        let evaluation_log = batch.evaluation_log_size;
        let coefficient_log = evaluation_log
            .checked_sub(1)
            .ok_or_else(|| "LDE evaluation log underflow".to_owned())?;
        let coefficient_words_per_column = pow2(coefficient_log)?;
        let evaluation_words_per_column = pow2(evaluation_log)?;
        let current_intervals = current_n2b_intervals(evaluation_log)?;
        let duplicate_intervals = duplicate_first_intervals(evaluation_log)?;
        let current_passes = u64::try_from(current_intervals.len())
            .map_err(|_| "current N2B pass count does not fit u64".to_owned())?;
        let duplicate_passes = u64::try_from(duplicate_intervals.len())
            .map_err(|_| "replacement N2B pass count does not fit u64".to_owned())?;
        let eliminated_passes = current_passes
            .checked_sub(duplicate_passes)
            .ok_or_else(|| "replacement N2B adds a global pass".to_owned())?;
        if duplicate_intervals.iter().sum::<u32>() != evaluation_log - 1 {
            return Err(format!(
                "replacement interval partition for log {evaluation_log} is not exact"
            ));
        }
        let chunks = chunks(columns)?;
        let b2n = interpolation_traffic(commitment, coefficient_log, &batch.columns)?;
        let coefficient_words = coefficient_words_per_column
            .checked_mul(columns)
            .ok_or_else(|| "coefficient words overflow".to_owned())?;
        let evaluation_words = evaluation_words_per_column
            .checked_mul(columns)
            .ok_or_else(|| "evaluation words overflow".to_owned())?;

        let current_lde = coefficient_words
            .checked_mul(3 + 4 * current_passes)
            .ok_or_else(|| "current LDE traffic overflow".to_owned())?;
        let tail_lde = coefficient_words
            .checked_mul(1 + 4 * current_passes)
            .ok_or_else(|| "direct-tail traffic overflow".to_owned())?;
        let duplicate_lde = coefficient_words
            .checked_mul(1 + 4 * duplicate_passes)
            .ok_or_else(|| "duplicate-first traffic overflow".to_owned())?;
        let current_words = b2n
            .words
            .checked_add(current_lde)
            .ok_or_else(|| "current B2N plus LDE traffic overflow".to_owned())?;
        let tail_words = b2n
            .words
            .checked_add(tail_lde)
            .ok_or_else(|| "direct-tail pipeline traffic overflow".to_owned())?;
        let duplicate_words = b2n
            .words
            .checked_add(duplicate_lde)
            .ok_or_else(|| "duplicate-first pipeline traffic overflow".to_owned())?;
        let current_lde_launches = chunks
            .checked_mul(1 + current_passes)
            .ok_or_else(|| "current LDE launch count overflow".to_owned())?;
        let duplicate_lde_launches = chunks
            .checked_mul(duplicate_passes)
            .ok_or_else(|| "replacement LDE launch count overflow".to_owned())?;
        let eliminated_n2b_passes = chunks
            .checked_mul(eliminated_passes)
            .ok_or_else(|| "eliminated N2B pass count overflow".to_owned())?;
        let current_launches = b2n
            .launches
            .checked_add(current_lde_launches)
            .ok_or_else(|| "current launch count overflow".to_owned())?;
        let duplicate_launches = b2n
            .launches
            .checked_add(duplicate_lde_launches)
            .ok_or_else(|| "replacement launch count overflow".to_owned())?;
        let tail_saved_words = current_words
            .checked_sub(tail_words)
            .ok_or_else(|| "direct-tail traffic exceeds current traffic".to_owned())?;
        let duplicate_saved_words = current_words
            .checked_sub(duplicate_words)
            .ok_or_else(|| "duplicate-first traffic exceeds current traffic".to_owned())?;

        let batch_totals = Totals {
            columns,
            coefficient_words,
            resident_coefficient_words: 0,
            evaluation_words,
            current_words,
            direct_tail_words: tail_words,
            duplicate_first_words: duplicate_words,
            current_launches,
            direct_tail_launches: current_launches,
            duplicate_first_launches: duplicate_launches,
            eliminated_n2b_passes,
        };
        totals.add(&batch_totals)?;
        batches.push(json!({
            "evaluation_log": evaluation_log,
            "coefficient_log": coefficient_log,
            "columns": columns,
            "canonical_columns": batch.columns,
            "chunks": chunks,
            "b2n_intervals": b2n.intervals,
            "current_n2b_intervals": current_intervals,
            "duplicate_first_n2b_intervals": duplicate_intervals,
            "coefficient_bytes": bytes(coefficient_words)?,
            "evaluation_bytes": bytes(evaluation_words)?,
            "current_pipeline_logical_bytes": bytes(current_words)?,
            "direct_output_tail_zero_saved_bytes": bytes(tail_saved_words)?,
            "duplicate_first_stage_saved_bytes": bytes(duplicate_saved_words)?,
            "current_pipeline_kernel_launches": current_launches,
            "duplicate_first_stage_kernel_launches": duplicate_launches,
        }));
    }
    totals.resident_coefficient_words = resident_coefficient_words;

    Ok((
        json!({
            "tree": tree_name(commitment.id),
            "canonical_columns": flattened.len(),
            "interpolation_batches": commitment.interpolation_batches.len(),
            "direct_retained_b2n_batches": commitment
                .direct_retained_b2n_program
                .as_ref()
                .map_or(0, |program| program.batches().len()),
            "transform_coefficient_image_bytes": bytes(totals.coefficient_words)?,
            "logical_coefficient_bytes": bytes(totals.resident_coefficient_words)?,
            "retained_evaluation_bytes": bytes(totals.evaluation_words)?,
            "ownership_representation": if commitment.id == CommitmentTreeId::Composition
                && arena.composition().output_plan.mode()
                    == CompositionOutputMode::DirectRetainedEvaluations
            {
                "direct-retained-evaluations"
            } else {
                "coefficients-plus-retained-evaluations"
            },
            "batches": batches,
        }),
        totals,
    ))
}

struct InterpolationTraffic {
    words: u64,
    launches: u64,
    intervals: Vec<u32>,
}

fn interpolation_traffic(
    commitment: &PlannedCommitment,
    coefficient_log: u32,
    canonical_columns: &[usize],
) -> Result<InterpolationTraffic, String> {
    if commitment.id == CommitmentTreeId::Composition {
        return Ok(InterpolationTraffic {
            words: 0,
            launches: 0,
            intervals: Vec::new(),
        });
    }
    if let Some(program) = &commitment.direct_retained_b2n_program {
        let matching = program
            .batches()
            .iter()
            .filter(|batch| {
                batch.source_log_size == coefficient_log
                    && batch.canonical_columns == canonical_columns
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!(
                "{} direct B2N has {} exact matches for canonical batch {:?}",
                tree_name(commitment.id),
                matching.len(),
                canonical_columns
            ));
        }
        let columns = u64::try_from(canonical_columns.len())
            .map_err(|_| "direct B2N column count does not fit u64".to_owned())?;
        return b2n_traffic(coefficient_log, columns);
    }
    let wanted = canonical_columns
        .iter()
        .map(|&canonical| canonical_source(commitment, canonical))
        .collect::<Result<Vec<_>, _>>()?;
    let mut count = 0u64;
    let mut launch_chunks = 0u64;
    for batch in commitment
        .interpolation_batches
        .iter()
        .filter(|batch| batch.log_size == coefficient_log)
    {
        let matched = batch
            .sources
            .iter()
            .filter(|source| wanted.contains(source))
            .count();
        if matched == 0 {
            continue;
        }
        if matched != batch.sources.len() {
            return Err(format!(
                "{} interpolation batch crosses a same-log LDE boundary",
                tree_name(commitment.id)
            ));
        }
        let matched = u64::try_from(matched)
            .map_err(|_| "interpolation batch size does not fit u64".to_owned())?;
        count = count
            .checked_add(matched)
            .ok_or_else(|| "interpolation column count overflow".to_owned())?;
        launch_chunks = launch_chunks
            .checked_add(chunks(matched)?)
            .ok_or_else(|| "interpolation chunk count overflow".to_owned())?;
    }
    if count != wanted.len() as u64 {
        return Err(format!(
            "{} same-log interpolation covers {count}/{} columns",
            tree_name(commitment.id),
            wanted.len()
        ));
    }
    let mut traffic = b2n_traffic(coefficient_log, count)?;
    let passes = u64::try_from(traffic.intervals.len())
        .map_err(|_| "B2N pass count does not fit u64".to_owned())?;
    traffic.launches = launch_chunks
        .checked_mul(passes)
        .ok_or_else(|| "B2N launch count overflow".to_owned())?;
    Ok(traffic)
}

fn b2n_traffic(coefficient_log: u32, columns: u64) -> Result<InterpolationTraffic, String> {
    let intervals = b2n_stage_intervals(coefficient_log)
        .ok_or_else(|| format!("unsupported B2N log {coefficient_log}"))?;
    let passes =
        u64::try_from(intervals.len()).map_err(|_| "B2N pass count does not fit u64".to_owned())?;
    let words_per_column = pow2(coefficient_log)?;
    Ok(InterpolationTraffic {
        words: words_per_column
            .checked_mul(columns)
            .and_then(|words| words.checked_mul(2 * passes))
            .ok_or_else(|| "B2N traffic overflow".to_owned())?,
        launches: chunks(columns)?
            .checked_mul(passes)
            .ok_or_else(|| "B2N launch count overflow".to_owned())?,
        intervals,
    })
}

fn canonical_columns(
    commitment: &PlannedCommitment,
) -> Result<Vec<(u32, CommitmentColumnSource)>, String> {
    if commitment.grouped_column_log_sizes.len() != commitment.grouped_column_sources.len() {
        return Err(format!("{} group shape drifted", tree_name(commitment.id)));
    }
    let mut columns = Vec::new();
    for (logs, sources) in commitment
        .grouped_column_log_sizes
        .iter()
        .zip(&commitment.grouped_column_sources)
    {
        if logs.len() != sources.len() {
            return Err(format!(
                "{} source/log shape drifted",
                tree_name(commitment.id)
            ));
        }
        columns.extend(logs.iter().copied().zip(sources.iter().copied()));
    }
    if columns.is_empty() {
        return Err(format!(
            "{} has no canonical columns",
            tree_name(commitment.id)
        ));
    }
    Ok(columns)
}

fn canonical_source(
    commitment: &PlannedCommitment,
    canonical: usize,
) -> Result<CommitmentColumnSource, String> {
    canonical_columns(commitment)?
        .get(canonical)
        .map(|(_, source)| *source)
        .ok_or_else(|| format!("canonical column {canonical} is out of range"))
}

fn validate_retained_outputs(
    commitment: &PlannedCommitment,
    flattened: &[(u32, CommitmentColumnSource)],
) -> Result<(), String> {
    let mut seen = 0usize;
    for ((logs, outputs), retained) in commitment
        .grouped_column_log_sizes
        .iter()
        .zip(&commitment.evaluation_output_groups)
        .zip(&commitment.retained_evaluation_groups)
    {
        let outputs = outputs.as_ref().ok_or_else(|| {
            format!(
                "{} has an unmaterialized output group",
                tree_name(commitment.id)
            )
        })?;
        let retained = retained.as_ref().ok_or_else(|| {
            format!(
                "{} has an unretained output group",
                tree_name(commitment.id)
            )
        })?;
        if outputs.len() != logs.len() || retained.len() != logs.len() {
            return Err(format!(
                "{} retained output shape drifted",
                tree_name(commitment.id)
            ));
        }
        for ((&log, output), retained) in logs.iter().zip(outputs).zip(retained) {
            let expected = usize::try_from(pow2(log + 1)?)
                .map_err(|_| "evaluation length does not fit usize".to_owned())?;
            if output.len_words != expected || retained.len_words != expected {
                return Err(format!(
                    "{} retained output length drifted",
                    tree_name(commitment.id)
                ));
            }
            seen += 1;
        }
    }
    if seen != flattened.len() {
        return Err(format!(
            "{} retained output count drifted",
            tree_name(commitment.id)
        ));
    }
    Ok(())
}

fn validate_commit_program(
    commitment: &PlannedCommitment,
    flattened: &[(u32, CommitmentColumnSource)],
) -> Result<(), String> {
    let program = commitment
        .commit_program
        .as_ref()
        .ok_or_else(|| format!("{} has no commit program", tree_name(commitment.id)))?;
    let columns = &program.requirements().leaves.plan.columns;
    if columns.len() != flattened.len() {
        return Err(format!(
            "{} program column count drifted",
            tree_name(commitment.id)
        ));
    }
    for (canonical, (planned, &(log, _))) in columns.iter().zip(flattened).enumerate() {
        if planned.canonical_index != canonical
            || planned.coefficient_log_size != log
            || planned.evaluation_log_size != log + 1
            || !planned.retained_evaluation
        {
            return Err(format!(
                "{} canonical program column {canonical} drifted",
                tree_name(commitment.id)
            ));
        }
    }
    let mut coverage = vec![0u8; flattened.len()];
    for batch in &program.requirements().leaves.plan.lde_batches {
        for &canonical in &batch.columns {
            let count = coverage
                .get_mut(canonical)
                .ok_or_else(|| "LDE batch canonical index is out of range".to_owned())?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| "LDE batch coverage overflow".to_owned())?;
            if flattened[canonical].0 + 1 != batch.evaluation_log_size {
                return Err("LDE batch log disagrees with canonical column".to_owned());
            }
        }
    }
    if coverage.iter().any(|&count| count != 1) {
        return Err(format!(
            "{} LDE batches are not a partition",
            tree_name(commitment.id)
        ));
    }
    Ok(())
}

fn validate_interpolation(
    commitment: &PlannedCommitment,
    flattened: &[(u32, CommitmentColumnSource)],
) -> Result<(), String> {
    if commitment.direct_retained_b2n_program.is_some() {
        return validate_direct_interpolation(commitment, flattened);
    }
    let expected = if commitment.id == CommitmentTreeId::Composition {
        0
    } else {
        flattened.len()
    };
    let actual = commitment
        .interpolation_batches
        .iter()
        .map(|batch| batch.sources.len())
        .sum::<usize>();
    if actual != expected {
        return Err(format!(
            "{} interpolation covers {actual}/{expected} columns",
            tree_name(commitment.id)
        ));
    }
    let mut remaining = flattened.to_vec();
    for batch in &commitment.interpolation_batches {
        for source in &batch.sources {
            let position = remaining
                .iter()
                .position(|&(log, candidate)| log == batch.log_size && candidate == *source)
                .ok_or_else(|| {
                    format!("{} interpolation source drifted", tree_name(commitment.id))
                })?;
            remaining.swap_remove(position);
        }
    }
    if commitment.id != CommitmentTreeId::Composition && !remaining.is_empty() {
        return Err(format!(
            "{} interpolation is incomplete",
            tree_name(commitment.id)
        ));
    }
    Ok(())
}

fn validate_direct_interpolation(
    commitment: &PlannedCommitment,
    flattened: &[(u32, CommitmentColumnSource)],
) -> Result<(), String> {
    let direct = commitment
        .direct_retained_b2n_program
        .as_ref()
        .ok_or_else(|| "missing direct B2N program".to_owned())?;
    let expected_role = match commitment.id {
        CommitmentTreeId::Base => TraceTreeRole::Base,
        CommitmentTreeId::Interaction => TraceTreeRole::Interaction,
        _ => {
            return Err(format!(
                "{} cannot own a direct B2N program",
                tree_name(commitment.id)
            ))
        }
    };
    if direct.role() != expected_role || !commitment.interpolation_batches.is_empty() {
        return Err(format!(
            "{} direct B2N mode drifted",
            tree_name(commitment.id)
        ));
    }
    let commit = commitment
        .commit_program
        .as_ref()
        .ok_or_else(|| format!("{} has no commit program", tree_name(commitment.id)))?;
    if direct.commit_cache_key() != commit.identity().cache_key {
        return Err(format!(
            "{} direct B2N commit identity drifted",
            tree_name(commitment.id)
        ));
    }
    let lde_batches = &commit.requirements().leaves.plan.lde_batches;
    if direct.batches().len() != lde_batches.len() {
        return Err(format!(
            "{} direct B2N batch count drifted",
            tree_name(commitment.id)
        ));
    }
    validate_exact_canonical_coverage(
        flattened.len(),
        direct
            .batches()
            .iter()
            .map(|batch| batch.canonical_columns.as_slice()),
    )?;
    for (index, (batch, lde)) in direct.batches().iter().zip(lde_batches).enumerate() {
        let expected_index = u32::try_from(index)
            .map_err(|_| "direct B2N batch index does not fit u32".to_owned())?;
        let expected_retained_log = batch
            .source_log_size
            .checked_add(1)
            .ok_or_else(|| "direct B2N retained log overflow".to_owned())?;
        if batch.batch_index != expected_index
            || batch.canonical_columns != lde.columns
            || batch.retained_log_size != lde.evaluation_log_size
            || batch.retained_log_size != expected_retained_log
            || batch.canonical_columns.iter().any(|&canonical| {
                !matches!(
                    flattened.get(canonical),
                    Some(&(log, _)) if log == batch.source_log_size
                )
            })
        {
            return Err(format!(
                "{} direct B2N batch {index} drifted",
                tree_name(commitment.id)
            ));
        }
    }
    Ok(())
}

fn validate_exact_canonical_coverage<'a>(
    columns: usize,
    batches: impl IntoIterator<Item = &'a [usize]>,
) -> Result<(), String> {
    let mut seen = vec![false; columns];
    let mut next = 0usize;
    for batch in batches {
        if batch.is_empty() {
            return Err("direct B2N contains an empty batch".to_owned());
        }
        for &canonical in batch {
            let covered = seen
                .get_mut(canonical)
                .ok_or_else(|| "direct B2N canonical index is out of range".to_owned())?;
            if *covered {
                return Err(format!(
                    "direct B2N canonical column {canonical} is duplicated"
                ));
            }
            if canonical != next {
                return Err(format!(
                    "direct B2N canonical order drifted at {next}: found {canonical}"
                ));
            }
            *covered = true;
            next = next
                .checked_add(1)
                .ok_or_else(|| "direct B2N coverage overflow".to_owned())?;
        }
    }
    if next != columns || seen.iter().any(|covered| !covered) {
        return Err(format!("direct B2N covers {next}/{columns} columns"));
    }
    Ok(())
}

fn validate_coefficient_logical_words(
    arena: &ProofArenaPlan,
    tree: CommitmentTreeId,
    flattened: &[(u32, CommitmentColumnSource)],
) -> Result<u64, String> {
    let purpose = match tree {
        CommitmentTreeId::Base => BufferPurpose::BaseCoefficients,
        CommitmentTreeId::Interaction => BufferPurpose::InteractionCoefficients,
        CommitmentTreeId::Composition => BufferPurpose::CompositionCoefficients,
        _ => return Err("coefficient validation requested for a non-dynamic tree".to_owned()),
    };
    let planned = arena
        .logical_buffers()
        .iter()
        .filter(|buffer| buffer.purpose == purpose)
        .try_fold(0u64, |total, buffer| {
            total
                .checked_add(buffer.len_words as u64)
                .ok_or_else(|| "logical coefficient words overflow".to_owned())
        })?;
    let expected = flattened.iter().try_fold(0u64, |total, (log, _)| {
        total
            .checked_add(pow2(*log)?)
            .ok_or_else(|| "expected coefficient words overflow".to_owned())
    })?;
    let direct_composition = tree == CommitmentTreeId::Composition
        && arena.composition().output_plan.mode()
            == CompositionOutputMode::DirectRetainedEvaluations;
    if direct_composition {
        let program = arena
            .composition()
            .output_plan
            .direct_program()
            .ok_or_else(|| "direct Composition output has no split program".to_owned())?;
        if planned != 0
            || flattened.len() != 8
            || flattened
                .iter()
                .any(|&(log, _)| log.checked_add(1) != Some(program.schedule().evaluation_log_size))
        {
            return Err(format!(
                "Composition direct ownership has {planned} coefficient words or drifted geometry"
            ));
        }
        return Ok(0);
    }
    if planned != expected {
        return Err(format!(
            "{} logical coefficient words {planned} != canonical {expected}",
            tree_name(tree)
        ));
    }
    Ok(planned)
}

fn validate_late_consumers(arena: &ProofArenaPlan) -> Result<(), String> {
    let ownership = arena.late_coefficient_ownership().entries();
    let dynamic_columns = arena
        .commitments()
        .iter()
        .filter(|commitment| {
            matches!(
                commitment.id,
                CommitmentTreeId::Base
                    | CommitmentTreeId::Interaction
                    | CommitmentTreeId::Composition
            )
        })
        .map(|commitment| {
            commitment
                .grouped_column_log_sizes
                .iter()
                .map(Vec::len)
                .sum::<usize>()
        })
        .sum::<usize>();
    if ownership.len() != dynamic_columns {
        return Err("late coefficient ownership does not cover every dynamic column".to_owned());
    }
    for entry in ownership {
        let expected_epoch = source_commit_epoch(entry.source)?;
        if entry.final_consumer != expected_epoch
            || entry.composition_reads_coefficients
            || entry.oods_reads_coefficients
            || entry.quotient_reads_coefficients
            || entry.decommit_reads_coefficients
        {
            return Err(format!(
                "dynamic coefficient {:?} survives its commit",
                entry.source
            ));
        }
        if matches!(entry.source, OpenedColumnSource::Trace { .. })
            && !arena
                .composition()
                .direct_bindings
                .iter()
                .any(|binding| binding.source == entry.source && binding.evaluation.is_some())
        {
            return Err(format!(
                "dynamic trace {:?} has no direct composition evaluation binding",
                entry.source
            ));
        }
    }

    validate_oods_and_numerator_sources(arena, ownership)?;
    validate_decommit_sources(arena)?;
    Ok(())
}

fn validate_oods_and_numerator_sources(
    arena: &ProofArenaPlan,
    ownership: &[stwo_cairo_gpu_prover::source_ownership::LateCoefficientOwnership],
) -> Result<(), String> {
    let dynamic = ownership
        .iter()
        .map(|entry| entry.source)
        .collect::<Vec<_>>();
    let oods = arena
        .oods()
        .columns
        .iter()
        .filter(|column| dynamic.contains(&column.source))
        .collect::<Vec<_>>();
    if oods.len() != dynamic.len()
        || oods
            .iter()
            .any(|column| column.source_kind != OodsSourceKind::Evaluations)
    {
        return Err("dynamic OODS sources are not exactly retained evaluations".to_owned());
    }
    let numerator = arena
        .quotient_numerator()
        .columns
        .iter()
        .filter(|column| dynamic.contains(&column.source))
        .collect::<Vec<_>>();
    if numerator.len() != dynamic.len()
        || numerator
            .iter()
            .any(|column| column.topology.source_kind != QuotientNumeratorSourceKind::Evaluation)
    {
        return Err("dynamic numerator sources are not exactly retained evaluations".to_owned());
    }
    let direct_composition =
        arena.composition().output_plan.mode() == CompositionOutputMode::DirectRetainedEvaluations;
    if numerator.iter().any(|column| {
        let direct =
            direct_composition && matches!(column.source, OpenedColumnSource::Composition { .. });
        direct != column.coefficients.is_none()
    }) {
        return Err(
            "dynamic numerator coefficient presence disagrees with direct Composition ownership"
                .to_owned(),
        );
    }

    let preprocessed_oods = arena
        .oods()
        .columns
        .iter()
        .filter(|column| matches!(column.source, OpenedColumnSource::Preprocessed { .. }))
        .collect::<Vec<_>>();
    let preprocessed_numerator = arena
        .quotient_numerator()
        .columns
        .iter()
        .filter(|column| matches!(column.source, OpenedColumnSource::Preprocessed { .. }))
        .collect::<Vec<_>>();
    if preprocessed_oods.is_empty()
        || preprocessed_numerator.is_empty()
        || preprocessed_oods
            .iter()
            .any(|column| column.source_kind != OodsSourceKind::Coefficients)
        || preprocessed_numerator.iter().any(|column| {
            column.topology.source_kind != QuotientNumeratorSourceKind::Coefficients
                || column.coefficients.is_none()
        })
    {
        return Err("preprocessed OODS/numerator ownership is not coefficient-backed".to_owned());
    }
    Ok(())
}

fn validate_decommit_sources(arena: &ProofArenaPlan) -> Result<(), String> {
    let dynamic_roles = [
        TraceTreeRole::Base,
        TraceTreeRole::Interaction,
        TraceTreeRole::Composition,
    ];
    let mut seen = BTreeMap::new();
    for tree in &arena.decommit().config.trees {
        let DecommitTreeGeometry::Trace(trace) = tree else {
            continue;
        };
        if dynamic_roles.contains(&trace.role) {
            if trace
                .groups
                .iter()
                .any(|group| group.mode != DecommitSourceMode::ResidentEvaluations)
            {
                return Err(format!(
                    "{:?} decommit recomputes a dynamic LDE",
                    trace.role
                ));
            }
            seen.insert(trace.role as u32, true);
        }
    }
    if seen.len() != dynamic_roles.len() {
        return Err("dynamic decommit tree coverage is incomplete".to_owned());
    }
    Ok(())
}

fn source_commit_epoch(source: OpenedColumnSource) -> Result<ProofEpoch, String> {
    match source {
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::BaseCoefficients,
            ..
        } => Ok(ProofEpoch::BaseCommit),
        OpenedColumnSource::Trace {
            purpose: BufferPurpose::InteractionCoefficients,
            ..
        } => Ok(ProofEpoch::InteractionCommit),
        OpenedColumnSource::Composition { .. } => Ok(ProofEpoch::CompositionCommit),
        _ => Err(format!("non-dynamic source in ownership plan: {source:?}")),
    }
}

#[cfg(test)]
#[path = "ntt_lde_receipt_tests.rs"]
mod tests;
