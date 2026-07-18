//! Canonical whole-batch placement for the direct-retained Base commitment.

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::{
    BaseCommitAccessKind, BaseCommitOperation, BaseCommitOperationKind, BaseCommitProgramAuthority,
    BaseCommitValueRole,
};

use super::*;
use crate::compiled_proof::{ExecutionPrimitive, OpId, PartitionAuthorityKind};
use crate::transcript_plan::CairoTranscriptSegment;

const PROGRESSIVE_STATE_WORDS_PER_ROW: usize = 24;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Batch {
    index: u32,
    columns: Vec<u32>,
    operation_ordinals: Vec<usize>,
}

#[derive(Debug)]
struct Placement {
    operation_workers: Vec<WorkerId>,
}

pub(super) fn base_commit_static_wrapper_workers(
    compiled: &CompiledProof,
    topology: &FleetPlacementTopology,
    first_operation: OpId,
    authority: &BaseCommitProgramAuthority,
) -> Result<BTreeMap<OpId, WorkerId>, FleetCompileError> {
    authority
        .validate()
        .map_err(|_| FleetCompileError::InvalidSemanticSchedule)?;
    let mut workers = topology
        .workers
        .iter()
        .map(|worker| worker.id)
        .collect::<Vec<_>>();
    workers.sort_unstable();
    if workers.is_empty()
        || workers.windows(2).any(|pair| pair[0] == pair[1])
        || !workers.contains(&topology.coordinator)
    {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    let placement = compile(authority, &workers)?;
    validate_compiled_range(
        compiled,
        first_operation,
        authority.operations(),
        &placement.operation_workers,
    )?;
    placement
        .operation_workers
        .into_iter()
        .enumerate()
        .map(|(ordinal, worker)| {
            let ordinal = u32::try_from(ordinal).map_err(|_| FleetCompileError::SizeOverflow)?;
            first_operation
                .0
                .checked_add(ordinal)
                .map(|operation| (OpId(operation), worker))
                .ok_or(FleetCompileError::SizeOverflow)
        })
        .collect()
}

fn compile(
    authority: &BaseCommitProgramAuthority,
    workers: &[WorkerId],
) -> Result<Placement, FleetCompileError> {
    let batches = parse_batches(authority)?;
    let batch_workers = partition_batches(&batches, workers)?;
    let mut operation_workers = vec![None; authority.operations().len()];
    for (batch, &worker) in batches.iter().zip(&batch_workers) {
        for &ordinal in &batch.operation_ordinals {
            if operation_workers
                .get_mut(ordinal)
                .ok_or(FleetCompileError::InvalidSemanticSchedule)?
                .replace(worker)
                .is_some()
            {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
        }
    }
    assign_state_and_merkle_operations(
        authority.operations(),
        &batch_workers,
        &mut operation_workers,
    )?;
    let operation_workers = operation_workers
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    validate_state_handoffs(authority)?;
    Ok(Placement { operation_workers })
}

fn parse_batches(authority: &BaseCommitProgramAuthority) -> Result<Vec<Batch>, FleetCompileError> {
    let operations = authority.operations();
    let mut segments = Vec::new();
    let mut ordinal = 0usize;
    while ordinal < operations.len() {
        let BaseCommitOperationKind::DirectB2n {
            batch_index,
            segment_offset,
            retained_log_size,
            canonical_columns,
            ..
        } = &operations[ordinal].kind
        else {
            if matches!(
                operations[ordinal].kind,
                BaseCommitOperationKind::DirectN2b { .. }
                    | BaseCommitOperationKind::StateAbsorb { .. }
            ) {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            ordinal += 1;
            continue;
        };
        let n2b = operations
            .get(ordinal + 1)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let absorb = operations
            .get(ordinal + 2)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let BaseCommitOperationKind::DirectN2b {
            batch_index: n2b_batch,
            segment_offset: n2b_offset,
            retained_log_size: n2b_log_size,
            canonical_columns: n2b_columns,
            ..
        } = &n2b.kind
        else {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        };
        let BaseCommitOperationKind::StateAbsorb {
            batch_index: absorb_batch,
            segment_offset: absorb_offset,
            log_size: absorb_log_size,
            canonical_columns: absorb_columns,
            ..
        } = &absorb.kind
        else {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        };
        if canonical_columns.is_empty()
            || n2b_batch != batch_index
            || absorb_batch != batch_index
            || n2b_offset != segment_offset
            || absorb_offset != segment_offset
            || n2b_log_size != retained_log_size
            || absorb_log_size != retained_log_size
            || n2b_columns != canonical_columns
            || absorb_columns != canonical_columns
        {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
        segments.push((
            *batch_index,
            *segment_offset,
            canonical_columns.clone(),
            [ordinal, ordinal + 1, ordinal + 2],
        ));
        ordinal += 3;
    }
    let mut batches = Vec::<Batch>::new();
    for (batch_index, segment_offset, columns, ordinals) in segments {
        if batches
            .last()
            .is_none_or(|batch| batch.index != batch_index)
        {
            if batch_index as usize != batches.len() {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            batches.push(Batch {
                index: batch_index,
                columns: Vec::new(),
                operation_ordinals: Vec::new(),
            });
        }
        let batch = batches
            .last_mut()
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        if batch.index != batch_index || segment_offset as usize != batch.columns.len() {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
        batch.columns.extend(columns);
        batch.operation_ordinals.extend(ordinals);
    }
    if batches.is_empty() {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    let actual = batches
        .iter()
        .flat_map(|batch| batch.columns.iter().copied())
        .collect::<Vec<_>>();
    let expected = authority
        .retained_evaluations()
        .iter()
        .map(|retained| retained.canonical_column)
        .collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected || unique.len() != actual.len() {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    Ok(batches)
}

/// Optimal contiguous minimax partition under the whole-batch constraint.
fn partition_batches(
    batches: &[Batch],
    workers: &[WorkerId],
) -> Result<Vec<WorkerId>, FleetCompileError> {
    if batches.is_empty() || workers.is_empty() {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    let groups = workers.len().min(batches.len());
    let mut prefix = Vec::with_capacity(batches.len() + 1);
    prefix.push(0usize);
    for batch in batches {
        let next = prefix
            .last()
            .copied()
            .and_then(|sum| sum.checked_add(batch.columns.len()))
            .ok_or(FleetCompileError::SizeOverflow)?;
        prefix.push(next);
    }
    let mut cost = vec![vec![usize::MAX; batches.len() + 1]; groups + 1];
    let mut split = vec![vec![0usize; batches.len() + 1]; groups + 1];
    cost[0][0] = 0;
    for group in 1..=groups {
        for end in group..=batches.len() {
            for start in group - 1..end {
                if cost[group - 1][start] == usize::MAX {
                    continue;
                }
                let load = prefix[end]
                    .checked_sub(prefix[start])
                    .ok_or(FleetCompileError::SizeOverflow)?;
                let candidate = cost[group - 1][start].max(load);
                if candidate < cost[group][end] {
                    cost[group][end] = candidate;
                    split[group][end] = start;
                }
            }
        }
    }
    if cost[groups][batches.len()] == usize::MAX {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    let mut boundaries = vec![0usize; groups + 1];
    boundaries[groups] = batches.len();
    for group in (1..=groups).rev() {
        boundaries[group - 1] = split[group][boundaries[group]];
    }
    let mut assignment = vec![workers[0]; batches.len()];
    for group in 0..groups {
        for worker in &mut assignment[boundaries[group]..boundaries[group + 1]] {
            *worker = workers[group];
        }
    }
    Ok(assignment)
}

fn assign_state_and_merkle_operations(
    operations: &[BaseCommitOperation],
    batch_workers: &[WorkerId],
    assignment: &mut [Option<WorkerId>],
) -> Result<(), FleetCompileError> {
    let first = *batch_workers
        .first()
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    let last = *batch_workers
        .last()
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    for (ordinal, operation) in operations.iter().enumerate() {
        if assignment[ordinal].is_some() {
            continue;
        }
        let worker = match operation.kind {
            BaseCommitOperationKind::StateInit { .. }
            | BaseCommitOperationKind::StateExpandInPlace { .. } => assignment[ordinal + 1..]
                .iter()
                .flatten()
                .next()
                .copied()
                .unwrap_or(last),
            BaseCommitOperationKind::StateFinalizeInPlace { .. }
            | BaseCommitOperationKind::MerkleLayerInPlace { .. }
            | BaseCommitOperationKind::MerkleLayer { .. } => last,
            BaseCommitOperationKind::DirectB2n { .. }
            | BaseCommitOperationKind::DirectN2b { .. }
            | BaseCommitOperationKind::StateAbsorb { .. } => {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
        };
        assignment[ordinal] = Some(worker);
    }
    if assignment.first().copied().flatten() != Some(first)
        || assignment.last().copied().flatten() != Some(last)
    {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    Ok(())
}

fn validate_state_handoffs(
    authority: &BaseCommitProgramAuthority,
) -> Result<(), FleetCompileError> {
    let layouts = authority
        .layouts()
        .iter()
        .map(|layout| (layout.role, layout))
        .collect::<BTreeMap<_, _>>();
    let mut producers = BTreeSet::<BaseCommitValueRole>::new();
    for operation in authority.operations() {
        for access in &operation.effect.accesses {
            let BaseCommitValueRole::State { .. } = access.role else {
                continue;
            };
            let layout = layouts
                .get(&access.role)
                .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
            if layout.words_per_row != PROGRESSIVE_STATE_WORDS_PER_ROW
                || access.first_word != 0
                || access.word_len != layout.logical_words
            {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            match access.kind {
                BaseCommitAccessKind::Read => {
                    if !producers.contains(&access.role) {
                        return Err(FleetCompileError::InvalidSemanticSchedule);
                    }
                }
                BaseCommitAccessKind::Write => {
                    if !producers.insert(access.role) {
                        return Err(FleetCompileError::InvalidSemanticSchedule);
                    }
                }
                BaseCommitAccessKind::ReadWrite => {
                    return Err(FleetCompileError::InvalidSemanticSchedule);
                }
            }
        }
    }
    Ok(())
}

fn validate_compiled_range(
    compiled: &CompiledProof,
    first: OpId,
    authority: &[BaseCommitOperation],
    workers: &[WorkerId],
) -> Result<(), FleetCompileError> {
    if authority.len() != workers.len() {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    for (ordinal, exact) in authority.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).map_err(|_| FleetCompileError::SizeOverflow)?;
        let id = OpId(
            first
                .0
                .checked_add(ordinal)
                .ok_or(FleetCompileError::SizeOverflow)?,
        );
        let operation = compiled
            .operation(id)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let ExecutionPrimitive::StaticCudaWrapper { wrapper } = operation.primitive else {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        };
        let wrapper = compiled
            .static_wrapper(wrapper)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let partition = compiled
            .partitions()
            .iter()
            .find(|partition| partition.id() == operation.partition)
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        if operation.stage
            != ProofStage::BeforeTranscript(CairoTranscriptSegment::BootstrapThroughBase)
            || !matches!(partition.kind(), PartitionAuthorityKind::Monolithic)
            || wrapper.wrapper_symbol() != exact.abi.wrapper_symbol().as_bytes()
            || wrapper.semantic_abi_identity() != &exact.abi_identity
            || wrapper.semantic_effect_identity() != &exact.effect.identity
            || wrapper.aggregate_contract_identity() != &exact.identity
        {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
    }
    Ok(())
}
