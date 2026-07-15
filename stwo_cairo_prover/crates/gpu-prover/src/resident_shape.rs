//! Generator-free projection of adapter input into the exact resident proof plan.

use std::collections::BTreeSet;
use std::sync::Arc;

use stwo::prover::backend::simd::m31::N_LANES;
use stwo_cairo_adapter::builtins::MemorySegmentAddresses;
use stwo_cairo_common::builtins::{
    ADD_MOD_BUILTIN_MEMORY_CELLS, BITWISE_BUILTIN_MEMORY_CELLS, EC_OP_BUILTIN_MEMORY_CELLS,
    MUL_MOD_BUILTIN_MEMORY_CELLS, PEDERSEN_BUILTIN_MEMORY_CELLS, POSEIDON_BUILTIN_MEMORY_CELLS,
    RANGE_CHECK_96_BUILTIN_MEMORY_CELLS, RANGE_CHECK_BUILTIN_MEMORY_CELLS,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, MAX_SEQUENCE_LOG_SIZE,
};
use stwo_cairo_prover::witness::builtins::get_builtins;
use stwo_cairo_prover::witness::cairo_claim_generator::get_sub_components;
use stwo_cairo_prover::witness::jit_prove_backend::{
    planned_compacted_consumer_shape_from_raw, recorded_input_compaction_geometry,
    PlannedCompactedRowsError,
};
use stwo_cairo_prover::witness::proof_shape::{
    padded_rows, PendingRowsReason, ProofShape, ProofShapeError, RowResolution,
    RuntimeComponentShape, TracePartId, TracePartShape,
};
use stwo_cairo_prover::witness::range_checks::get_range_checks;

use crate::plan::{ProofPlan, ProofPlanError};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::resident_input::ResidentProverInputOwner;
use crate::schedule::{ComponentId, ComponentRowSource};
use crate::schedule_table::CAIRO_SCHEDULE;

#[derive(Debug)]
pub enum RawResidentShapeError {
    GenericOpcodeUnsupported {
        rows: usize,
    },
    MissingDirectRows(ComponentId),
    MissingBuiltinSegment(ComponentId),
    InvalidBuiltinSegment(ComponentId),
    CompactedRowsExceedCapacity {
        component: ComponentId,
        n_real_rows: u64,
        padded_rows: u64,
        max_rows: u64,
        padded_capacity: u64,
    },
    MemoryComponentCount {
        requested: usize,
        required: usize,
    },
    SizeOverflow(ComponentId),
    Shape(ProofShapeError),
    Plan(ProofPlanError),
    Compacted(PlannedCompactedRowsError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCompactedRows {
    pub n_real_rows: u64,
    pub padded_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCompactedGeometry {
    pub component: ComponentId,
    pub rows: Option<RawCompactedRows>,
}

impl core::fmt::Display for RawResidentShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "raw resident shape rejected: {self:?}")
    }
}

impl std::error::Error for RawResidentShapeError {}

impl From<ProofShapeError> for RawResidentShapeError {
    fn from(value: ProofShapeError) -> Self {
        Self::Shape(value)
    }
}

impl From<ProofPlanError> for RawResidentShapeError {
    fn from(value: ProofPlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<PlannedCompactedRowsError> for RawResidentShapeError {
    fn from(value: PlannedCompactedRowsError) -> Self {
        Self::Compacted(value)
    }
}

/// Build the same sealed pre-witness capacity plan as legacy ingest without a
/// `CairoClaimGenerator`. Plain relation-feed consumers remain capacity-bounded;
/// the caller promotes them with `strict_resident_exact` immediately before
/// graph preparation, exactly as the existing resident session does.
pub fn raw_replacement_proof_plan(
    owner: &ResidentProverInputOwner,
    preprocessed_trace: Arc<PreProcessedTrace>,
    opt_n_id_to_big_components: Option<usize>,
) -> Result<ProofPlan, RawResidentShapeError> {
    if let Some(rows) = owner
        .direct_input_rows("generic_opcode")
        .filter(|&rows| rows != 0)
    {
        return Err(RawResidentShapeError::GenericOpcodeUnsupported { rows });
    }
    let present = present_components(owner, preprocessed_trace);

    let observed = ProofShape::new(
        CAIRO_SCHEDULE
            .nodes
            .iter()
            .map(|node| {
                project_component(
                    owner,
                    &present,
                    node.id,
                    node.facts.row_source,
                    opt_n_id_to_big_components,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let capacity = ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &observed)?;
    let sealed = seal_compacted(owner, &observed, &capacity)?;
    ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &sealed).map_err(Into::into)
}

/// Canonical small identity for the only data-dependent pre-witness row
/// projections. Different memory contents with the same derived geometry are
/// deliberately equivalent cache keys.
pub fn raw_replacement_compacted_geometry(
    owner: &ResidentProverInputOwner,
    preprocessed_trace: Arc<PreProcessedTrace>,
) -> Result<[RawCompactedGeometry; 3], RawResidentShapeError> {
    let present = present_components(owner, preprocessed_trace);
    Ok([
        compacted_geometry(owner, &present, "verify_instruction")?,
        compacted_geometry(owner, &present, "pedersen_aggregator_window_bits_18")?,
        compacted_geometry(owner, &present, "poseidon_aggregator")?,
    ])
}

fn compacted_geometry(
    owner: &ResidentProverInputOwner,
    present: &BTreeSet<ComponentId>,
    component: ComponentId,
) -> Result<RawCompactedGeometry, RawResidentShapeError> {
    let shape = planned_compacted_consumer_shape_from_raw(
        component,
        present.contains(component),
        owner.pc_count(),
        owner.execution_memory(),
        owner.builtin_segments(),
    )?;
    let rows = shape
        .map(|shape| {
            let RowResolution::Resolved(parts) = shape.rows else {
                return Err(RawResidentShapeError::MissingDirectRows(component));
            };
            let [part] = parts.as_slice() else {
                return Err(RawResidentShapeError::MissingDirectRows(component));
            };
            if part.part != TracePartId::Main {
                return Err(RawResidentShapeError::MissingDirectRows(component));
            }
            Ok(RawCompactedRows {
                n_real_rows: part.n_real_rows,
                padded_rows: part.padded_rows,
            })
        })
        .transpose()?;
    Ok(RawCompactedGeometry { component, rows })
}

fn present_components(
    owner: &ResidentProverInputOwner,
    preprocessed_trace: Arc<PreProcessedTrace>,
) -> BTreeSet<ComponentId> {
    let mut present = BTreeSet::new();
    let sources = owner
        .opcode_labels()
        .into_iter()
        .chain(get_builtins(owner.builtin_segments(), preprocessed_trace));
    for source in sources {
        present.extend(get_sub_components(source));
    }
    present.extend(get_range_checks());
    present.extend([
        "verify_bitwise_xor_4",
        "verify_bitwise_xor_7",
        "verify_bitwise_xor_8",
        "verify_bitwise_xor_9",
    ]);
    present
}

fn project_component(
    owner: &ResidentProverInputOwner,
    present: &BTreeSet<ComponentId>,
    component: ComponentId,
    source: ComponentRowSource,
    opt_n_id_to_big_components: Option<usize>,
) -> Result<RuntimeComponentShape, RawResidentShapeError> {
    if !present.contains(component) {
        return Ok(RuntimeComponentShape::absent(component));
    }
    match source {
        ComponentRowSource::DirectInputs => {
            let rows = owner
                .direct_input_rows(component)
                .ok_or(RawResidentShapeError::MissingDirectRows(component))?;
            uniform_padded(component, rows)
        }
        ComponentRowSource::StoredLogSize => {
            let rows = builtin_rows(owner, component)?;
            RuntimeComponentShape::uniform(component, rows as u64, rows as u64).map_err(Into::into)
        }
        ComponentRowSource::FixedLogSize(log_size) => {
            let rows = 1u64
                .checked_shl(log_size)
                .ok_or(RawResidentShapeError::SizeOverflow(component))?;
            RuntimeComponentShape::uniform(component, rows, rows).map_err(Into::into)
        }
        ComponentRowSource::WitnessRelationFeeds => Ok(RuntimeComponentShape::pending(
            component,
            PendingRowsReason::WitnessRelationFeeds,
            0,
        )),
        ComponentRowSource::MemoryAddress => memory_address_shape(owner, component),
        ComponentRowSource::MemoryIdToBig => {
            memory_id_shape(owner, component, opt_n_id_to_big_components)
        }
    }
}

fn uniform_padded(
    component: ComponentId,
    rows: usize,
) -> Result<RuntimeComponentShape, RawResidentShapeError> {
    let rows = u64::try_from(rows).map_err(|_| RawResidentShapeError::SizeOverflow(component))?;
    let padded = padded_rows(component, rows, N_LANES as u64)?;
    RuntimeComponentShape::uniform(component, rows, padded).map_err(Into::into)
}

fn builtin_rows(
    owner: &ResidentProverInputOwner,
    component: ComponentId,
) -> Result<usize, RawResidentShapeError> {
    let cells_per_instance = match component {
        "add_mod_builtin" => ADD_MOD_BUILTIN_MEMORY_CELLS,
        "bitwise_builtin" => BITWISE_BUILTIN_MEMORY_CELLS,
        "mul_mod_builtin" => MUL_MOD_BUILTIN_MEMORY_CELLS,
        "pedersen_builtin" | "pedersen_builtin_narrow_windows" => PEDERSEN_BUILTIN_MEMORY_CELLS,
        "poseidon_builtin" => POSEIDON_BUILTIN_MEMORY_CELLS,
        "range_check96_builtin" => RANGE_CHECK_96_BUILTIN_MEMORY_CELLS,
        "range_check_builtin" => RANGE_CHECK_BUILTIN_MEMORY_CELLS,
        "ec_op_builtin" => EC_OP_BUILTIN_MEMORY_CELLS,
        _ => return Err(RawResidentShapeError::MissingBuiltinSegment(component)),
    };
    let segment = owner
        .builtin_segments()
        .get_segment_by_name(component)
        .ok_or(RawResidentShapeError::MissingBuiltinSegment(component))?;
    checked_segment_rows(component, segment, cells_per_instance)
}

fn checked_segment_rows(
    component: ComponentId,
    segment: MemorySegmentAddresses,
    cells_per_instance: usize,
) -> Result<usize, RawResidentShapeError> {
    let cells = segment
        .stop_ptr
        .checked_sub(segment.begin_addr)
        .ok_or(RawResidentShapeError::InvalidBuiltinSegment(component))?;
    if cells == 0 || !cells.is_multiple_of(cells_per_instance) {
        return Err(RawResidentShapeError::InvalidBuiltinSegment(component));
    }
    let rows = cells / cells_per_instance;
    if !rows.is_power_of_two() {
        return Err(RawResidentShapeError::InvalidBuiltinSegment(component));
    }
    Ok(rows)
}

fn memory_address_shape(
    owner: &ResidentProverInputOwner,
    component: ComponentId,
) -> Result<RuntimeComponentShape, RawResidentShapeError> {
    let split = cairo_air::components::memory_address_to_id::MEMORY_ADDRESS_TO_ID_SPLIT;
    let table_rows = owner
        .execution_memory()
        .address_to_id
        .len()
        .saturating_sub(1)
        .div_ceil(N_LANES)
        * N_LANES;
    uniform_padded(component, table_rows / split)
}

fn memory_id_shape(
    owner: &ResidentProverInputOwner,
    component: ComponentId,
    requested: Option<usize>,
) -> Result<RuntimeComponentShape, RawResidentShapeError> {
    let memory = owner.execution_memory();
    let big_rows = memory.f252_values.len().next_multiple_of(N_LANES);
    let small_rows = memory.small_values.len().next_multiple_of(N_LANES);
    let max_big_rows = 1usize
        .checked_shl(MAX_SEQUENCE_LOG_SIZE)
        .ok_or(RawResidentShapeError::SizeOverflow(component))?;
    let required = big_rows.div_ceil(max_big_rows);
    let count = requested.unwrap_or(required);
    if count < required {
        return Err(RawResidentShapeError::MemoryComponentCount {
            requested: count,
            required,
        });
    }
    let mut parts = Vec::with_capacity(count + 1);
    for index in 0..count {
        let rows = if index < required {
            big_rows
                .saturating_sub(index * max_big_rows)
                .min(max_big_rows)
        } else {
            N_LANES
        };
        let n_real_rows =
            u64::try_from(rows).map_err(|_| RawResidentShapeError::SizeOverflow(component))?;
        parts.push(TracePartShape {
            part: TracePartId::MemoryBig(
                u32::try_from(index).map_err(|_| RawResidentShapeError::SizeOverflow(component))?,
            ),
            n_real_rows,
            padded_rows: padded_rows(component, n_real_rows, N_LANES as u64)?,
        });
    }
    let small_rows =
        u64::try_from(small_rows).map_err(|_| RawResidentShapeError::SizeOverflow(component))?;
    parts.push(TracePartShape {
        part: TracePartId::MemorySmall,
        n_real_rows: small_rows,
        padded_rows: padded_rows(component, small_rows, N_LANES as u64)?,
    });
    RuntimeComponentShape::parts(component, parts).map_err(Into::into)
}

fn seal_compacted(
    owner: &ResidentProverInputOwner,
    observed: &ProofShape,
    capacity: &ProofPlan,
) -> Result<ProofShape, RawResidentShapeError> {
    let mut components = Vec::with_capacity(observed.components().len());
    for component in observed.components() {
        if recorded_input_compaction_geometry(component.id).is_none()
            || matches!(component.rows, RowResolution::Absent)
        {
            components.push(component.clone());
            continue;
        }
        let derived = planned_compacted_consumer_shape_from_raw(
            component.id,
            true,
            owner.pc_count(),
            owner.execution_memory(),
            owner.builtin_segments(),
        )?
        .ok_or(RawResidentShapeError::MissingDirectRows(component.id))?;
        let RowResolution::Resolved(parts) = &derived.rows else {
            return Err(RawResidentShapeError::MissingDirectRows(component.id));
        };
        let RowResolution::Bounded { bound, .. } = &capacity
            .proof_shape()
            .component(component.id)
            .ok_or(RawResidentShapeError::MissingDirectRows(component.id))?
            .rows
        else {
            return Err(RawResidentShapeError::MissingDirectRows(component.id));
        };
        if parts.len() != 1 {
            return Err(RawResidentShapeError::MissingDirectRows(component.id));
        }
        let part = &parts[0];
        if part.n_real_rows > bound.max_rows || part.padded_rows > bound.padded_capacity {
            return Err(RawResidentShapeError::CompactedRowsExceedCapacity {
                component: component.id,
                n_real_rows: part.n_real_rows,
                padded_rows: part.padded_rows,
                max_rows: bound.max_rows,
                padded_capacity: bound.padded_capacity,
            });
        }
        components.push(derived);
    }
    ProofShape::new(components).map_err(Into::into)
}
