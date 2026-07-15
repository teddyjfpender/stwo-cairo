//! Audited forward N2B partitions and checked byte/launch arithmetic.

use stwo_backend_cuda::INTERPOLATION_MAX_BATCH_COLUMNS;
use stwo_cairo_gpu_prover::arena_plan::CommitmentTreeId;

pub(super) fn current_n2b_intervals(log: u32) -> Result<Vec<u32>, String> {
    let intervals = match log {
        1..=12 => vec![1; log as usize],
        13 => vec![6, 7],
        14 => vec![6, 8],
        15 => vec![8, 7],
        16 => vec![8, 8],
        17 => vec![6, 11],
        18 => vec![8, 10],
        19 => vec![8, 11],
        20 => vec![6, 6, 8],
        21 => vec![6, 8, 7],
        22 => vec![6, 8, 8],
        23 => vec![8, 8, 7],
        24 => vec![8, 8, 8],
        25 => vec![6, 8, 11],
        26 => vec![8, 8, 10],
        27 => vec![8, 8, 11],
        28 => vec![6, 6, 6, 10],
        29 => vec![6, 6, 6, 11],
        30 => vec![6, 6, 8, 10],
        _ => return Err(format!("unsupported N2B log {log}")),
    };
    if intervals.iter().sum::<u32>() != log {
        return Err(format!("current N2B partition for log {log} drifted"));
    }
    Ok(intervals)
}

pub(super) fn duplicate_first_intervals(log: u32) -> Result<Vec<u32>, String> {
    let intervals = match log {
        2..=12 => vec![1; (log - 1) as usize],
        13 => vec![5, 7],
        14 => vec![5, 8],
        15 => vec![6, 8],
        16 => vec![8, 7],
        17 => vec![8, 8],
        18 => vec![6, 11],
        19 => vec![8, 10],
        20 => vec![8, 11],
        21 => vec![6, 6, 8],
        22 => vec![6, 8, 7],
        23 => vec![6, 8, 8],
        24 => vec![8, 8, 7],
        25 => vec![8, 8, 8],
        26 => vec![6, 8, 11],
        27 => vec![8, 8, 10],
        28 => vec![8, 8, 11],
        29 => vec![6, 6, 8, 8],
        30 => vec![6, 6, 6, 11],
        _ => return Err(format!("unsupported duplicate-first N2B log {log}")),
    };
    Ok(intervals)
}

pub(super) fn chunks(columns: u64) -> Result<u64, String> {
    if columns == 0 {
        return Err("empty same-log batch".to_owned());
    }
    columns
        .checked_add(INTERPOLATION_MAX_BATCH_COLUMNS as u64 - 1)
        .map(|rounded| rounded / INTERPOLATION_MAX_BATCH_COLUMNS as u64)
        .ok_or_else(|| "column chunk count overflow".to_owned())
}

pub(super) fn pow2(log: u32) -> Result<u64, String> {
    1u64.checked_shl(log)
        .ok_or_else(|| format!("2^{log} does not fit u64"))
}

pub(super) fn bytes(words: u64) -> Result<u64, String> {
    words
        .checked_mul(core::mem::size_of::<u32>() as u64)
        .ok_or_else(|| "byte count overflow".to_owned())
}

pub(super) fn bytes_per_second(saved_bytes: u64, nanoseconds: u64) -> Result<u64, String> {
    u128::from(saved_bytes)
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_div(u128::from(nanoseconds)))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "saved byte rate overflow".to_owned())
}

pub(super) fn tree_name(tree: CommitmentTreeId) -> &'static str {
    match tree {
        CommitmentTreeId::Preprocessed => "preprocessed",
        CommitmentTreeId::Base => "base",
        CommitmentTreeId::Interaction => "interaction",
        CommitmentTreeId::Composition => "composition",
        CommitmentTreeId::Fri(_) => "fri",
    }
}
