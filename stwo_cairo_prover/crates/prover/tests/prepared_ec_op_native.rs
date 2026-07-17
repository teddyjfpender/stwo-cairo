//! Native H100 gate for the resident `ec_op_builtin` writer.
//!
//! The generated SIMD writer is the oracle for all 273 base columns, all 488
//! word-major lookup words, all 127 padded `partial_ec_mul_generic` input
//! columns, and the four direct multiplicity destinations. Eager launch,
//! captured replay, same-geometry memory mutation, and a changed arena-seeded
//! segment start must be byte-identical.

#![cfg(stwo_cuda_link)]

use std::path::PathBuf;
use std::sync::Arc;

use stwo_backend_cuda::{
    ec_op_workspace_requirements, execution_tables_workspace_requirements, ArenaLayout, ArenaSlice,
    ArenaSlotId, ArenaSlotSpec, CudaExecContext, DeviceArena, EcOpArenaSlotRequirement,
    EcOpMultiplicityGeometry, EcOpWorkspaceRequirements, EcOpWorkspaceSlots,
    ExecutionTablesArenaSlotRequirement, ExecutionTablesHostData,
    ExecutionTablesWorkspaceRequirements, ExecutionTablesWorkspaceSlots, PreparedEcOpError,
    PreparedEcOpGraph, PreparedExecutionTablesGraph, EC_OP_LOOKUP_WORDS_PER_ROW,
    EC_OP_PARTIAL_INPUT_COLUMNS, EC_OP_TRACE_COLUMNS, EXECUTION_TABLE_BIG_LIMBS,
    EXECUTION_TABLE_SMALL_LIMBS,
};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
use stwo_cairo_prover::witness::jit_prove_backend::{BuiltinLaneSpec, PartialEcMulGenericLane};

#[derive(Clone)]
struct HostReference {
    segment_start: usize,
    rows: usize,
    trace: Vec<Vec<u32>>,
    lookup: Vec<u32>,
    partial_inputs: Vec<Vec<u32>>,
    addr_to_id: Vec<u32>,
    f252_values: Vec<[u32; 8]>,
    small_values: Vec<u128>,
}

impl HostReference {
    fn tables(&self) -> ExecutionTablesHostData<'_> {
        ExecutionTablesHostData {
            addr_to_id: &self.addr_to_id,
            f252_values: &self.f252_values,
            small_values: &self.small_values,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Snapshot {
    trace: Vec<Vec<u32>>,
    lookup: Vec<u32>,
    partial_inputs: Vec<Vec<u32>>,
    multiplicities: [Vec<u32>; 4],
}

fn fixture() -> ProverInput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/test_prove_verify_all_builtins/prover_input.json");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&json).expect("deserialize all-builtins prover input")
}

fn host_reference(input: ProverInput, segment_override: Option<usize>) -> HostReference {
    let addr_to_id = input.memory.address_to_id.iter().map(|id| id.0).collect();
    let f252_values = input.memory.f252_values.clone();
    let small_values = input.memory.small_values.clone();
    let preprocessed_trace = Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace());
    let mut generator = create_cairo_claim_generator(input, preprocessed_trace);

    let mut ec_op = generator
        .ec_op_builtin
        .take()
        .expect("fixture contains ec_op_builtin");
    if let Some(segment_start) = segment_override {
        ec_op.ec_op_builtin_segment_start = u32::try_from(segment_start).unwrap();
    }
    let rows = 1usize << ec_op.log_size;
    let segment_start = ec_op.ec_op_builtin_segment_start as usize;
    let (trace, claim, interaction) = ec_op.write_trace(
        generator
            .memory_address_to_id
            .as_ref()
            .expect("memory_address_to_id"),
        generator
            .memory_id_to_big
            .as_ref()
            .expect("memory_id_to_big"),
        generator.range_check_8.as_ref().expect("range_check_8"),
        generator
            .partial_ec_mul_generic
            .as_ref()
            .expect("partial_ec_mul_generic"),
    );
    assert_eq!(claim.log_size, rows.ilog2());

    let mut trace_columns = vec![vec![0u32; rows]; EC_OP_TRACE_COLUMNS];
    for row in 0..rows {
        for (column, value) in trace.row_at(row).into_iter().enumerate() {
            trace_columns[column][row] = value.0;
        }
    }
    let lookup = interaction.into_flat_lookup_words();
    let partial_inputs = <PartialEcMulGenericLane as BuiltinLaneSpec>::input_columns(
        generator
            .partial_ec_mul_generic
            .as_ref()
            .expect("partial_ec_mul_generic"),
    )
    .expect("EC-op emits packed generic inputs")
    .columns;

    assert_eq!(
        trace_columns.len(),
        cairo_air::components::ec_op_builtin::N_TRACE_COLUMNS
    );
    assert_eq!(trace_columns.len(), EC_OP_TRACE_COLUMNS);
    assert_eq!(lookup.len(), rows * EC_OP_LOOKUP_WORDS_PER_ROW);
    assert_eq!(partial_inputs.len(), EC_OP_PARTIAL_INPUT_COLUMNS);
    assert!(partial_inputs
        .iter()
        .all(|column| column.len() == rows * 256));

    HostReference {
        segment_start,
        rows,
        trace: trace_columns,
        lookup,
        partial_inputs,
        addr_to_id,
        f252_values,
        small_values,
    }
}

fn relocate_and_swap_instances(
    input: &mut ProverInput,
    segment_start: usize,
    rows: usize,
) -> usize {
    let relocated_start = 1usize;
    assert_ne!(relocated_start, segment_start);
    let words = 7 * rows;
    assert!(relocated_start + words <= input.memory.address_to_id.len());
    let original = input.memory.address_to_id[segment_start..segment_start + words].to_vec();
    input.memory.address_to_id[relocated_start..relocated_start + words].copy_from_slice(&original);
    for offset in 0..7 {
        input
            .memory
            .address_to_id
            .swap(relocated_start + offset, relocated_start + 7 + offset);
    }
    relocated_start
}

fn next_id(next: &mut u32) -> ArenaSlotId {
    let id = ArenaSlotId(*next);
    *next += 1;
    id
}

fn execution_slots(next: &mut u32) -> ExecutionTablesWorkspaceSlots {
    ExecutionTablesWorkspaceSlots {
        raw_addr_to_id: next_id(next),
        raw_f252_words: next_id(next),
        raw_small_words: next_id(next),
        big_limbs: (0..EXECUTION_TABLE_BIG_LIMBS)
            .map(|_| next_id(next))
            .collect(),
        small_limbs: (0..EXECUTION_TABLE_SMALL_LIMBS)
            .map(|_| next_id(next))
            .collect(),
        table_pointers: next_id(next),
        table_strides: next_id(next),
    }
}

fn ec_op_slots(next: &mut u32) -> EcOpWorkspaceSlots {
    EcOpWorkspaceSlots {
        trace_columns: (0..EC_OP_TRACE_COLUMNS).map(|_| next_id(next)).collect(),
        lookup_words: next_id(next),
        partial_input_columns: (0..EC_OP_PARTIAL_INPUT_COLUMNS)
            .map(|_| next_id(next))
            .collect(),
        segment_start: next_id(next),
        address_counts: next_id(next),
        big_counts: next_id(next),
        small_counts: next_id(next),
        range_check_8_counts: next_id(next),
    }
}

fn arena(
    execution: &ExecutionTablesWorkspaceRequirements,
    execution_slots: &ExecutionTablesWorkspaceSlots,
    ec_op: &EcOpWorkspaceRequirements,
    ec_op_slots: &EcOpWorkspaceSlots,
) -> DeviceArena {
    let mut offset = 0usize;
    let mut specs = Vec::new();
    for ExecutionTablesArenaSlotRequirement {
        id,
        len_words,
        alignment_words,
    } in execution.arena_slot_requirements(execution_slots).unwrap()
    {
        offset = offset.next_multiple_of(alignment_words);
        specs.push(ArenaSlotSpec {
            id,
            offset_words: offset,
            len_words,
            alignment_words,
        });
        offset += len_words;
    }
    for EcOpArenaSlotRequirement {
        id,
        len_words,
        alignment_words,
    } in ec_op.arena_slot_requirements(ec_op_slots).unwrap()
    {
        offset = offset.next_multiple_of(alignment_words);
        specs.push(ArenaSlotSpec {
            id,
            offset_words: offset,
            len_words,
            alignment_words,
        });
        offset += len_words;
    }
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap()
}

fn enqueue_read(arena: &DeviceArena, source: ArenaSlice, destination: &mut [u32]) {
    assert_eq!(source.len_words(), destination.len());
    unsafe {
        arena
            .context()
            .memcpy_d2h_async(
                destination.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                source.len_bytes(),
            )
            .unwrap();
    }
}

fn snapshot(arena: &DeviceArena, graph: &PreparedEcOpGraph<'_>) -> Snapshot {
    let mut trace = graph
        .trace_columns()
        .iter()
        .map(|column| vec![0u32; column.len_words()])
        .collect::<Vec<_>>();
    let mut lookup = vec![0u32; graph.lookup_words().len_words()];
    let mut partial_inputs = graph
        .partial_input_columns()
        .iter()
        .map(|column| vec![0u32; column.len_words()])
        .collect::<Vec<_>>();
    let mut multiplicities = graph
        .multiplicity_destinations()
        .map(|source| vec![0u32; source.len_words()]);
    for (&source, destination) in graph.trace_columns().iter().zip(&mut trace) {
        enqueue_read(arena, source, destination);
    }
    enqueue_read(arena, graph.lookup_words(), &mut lookup);
    for (&source, destination) in graph
        .partial_input_columns()
        .iter()
        .zip(&mut partial_inputs)
    {
        enqueue_read(arena, source, destination);
    }
    for (source, destination) in graph
        .multiplicity_destinations()
        .into_iter()
        .zip(&mut multiplicities)
    {
        enqueue_read(arena, source, destination);
    }
    arena.context().sync().unwrap();
    Snapshot {
        trace,
        lookup,
        partial_inputs,
        multiplicities,
    }
}

fn expected(reference: &HostReference, requirements: &EcOpWorkspaceRequirements) -> Snapshot {
    let mut multiplicities = [
        vec![0u32; requirements.address_count_words],
        vec![0u32; requirements.big_count_words],
        vec![0u32; requirements.small_count_words],
        vec![0u32; requirements.range_check_8_count_words],
    ];
    let id_columns = [0, 29, 58, 87, 116, 271, 272];
    for row in 0..reference.rows {
        for (offset, column) in id_columns.into_iter().enumerate() {
            multiplicities[0][reference.segment_start + 7 * row + offset - 1] += 1;
            let id = reference.trace[column][row];
            let tag = id >> 30;
            let index = (id & 0x3fff_ffff) as usize;
            match tag {
                1 => multiplicities[1][index] += 1,
                0 => multiplicities[2][index] += 1,
                _ => panic!("invalid memory id tag in host oracle"),
            }
        }
        multiplicities[3][reference.lookup[166 * reference.rows + row] as usize] += 1;
        multiplicities[3][reference.lookup[168 * reference.rows + row] as usize] += 1;
    }
    Snapshot {
        trace: reference.trace.clone(),
        lookup: reference.lookup.clone(),
        partial_inputs: reference.partial_inputs.clone(),
        multiplicities,
    }
}

fn clear_multiplicities(arena: &DeviceArena, graph: &PreparedEcOpGraph<'_>) {
    for destination in graph.multiplicity_destinations() {
        unsafe {
            arena
                .context()
                .memset_async(destination.as_void_ptr(), 0, destination.len_bytes())
                .unwrap();
        }
    }
    arena.context().sync().unwrap();
}

#[test]
fn prepared_ec_op_eager_capture_and_mutated_replay_match_generated_simd() {
    let input = fixture();
    let first = host_reference(input.clone(), None);
    assert!(first.rows >= 16);
    let mut mutated_input = input;
    let relocated_start =
        relocate_and_swap_instances(&mut mutated_input, first.segment_start, first.rows);
    let second = host_reference(mutated_input, Some(relocated_start));
    assert_eq!(second.rows, first.rows);
    assert_ne!(second.segment_start, first.segment_start);
    assert_eq!(second.f252_values.len(), first.f252_values.len());
    assert_eq!(second.small_values.len(), first.small_values.len());

    let execution_requirements = execution_tables_workspace_requirements(
        first.addr_to_id.len(),
        first.f252_values.len(),
        first.small_values.len(),
    )
    .unwrap();
    let ec_op_requirements = ec_op_workspace_requirements(
        first.rows,
        EcOpMultiplicityGeometry {
            address_count_words: first.addr_to_id.len(),
            big_count_words: first.f252_values.len(),
            small_count_words: first.small_values.len(),
            range_check_8_count_words: 256,
        },
    )
    .unwrap();
    let mut next = 1u32;
    let execution_slot_ids = execution_slots(&mut next);
    let ec_op_slot_ids = ec_op_slots(&mut next);
    let arena = arena(
        &execution_requirements,
        &execution_slot_ids,
        &ec_op_requirements,
        &ec_op_slot_ids,
    );
    let execution =
        PreparedExecutionTablesGraph::prepare(&arena, &execution_requirements, &execution_slot_ids)
            .unwrap();
    execution.ingest(first.tables()).unwrap();
    execution.launch().unwrap();
    let ec_op = PreparedEcOpGraph::prepare(
        &arena,
        execution.view().unwrap(),
        &ec_op_requirements,
        &ec_op_slot_ids,
    )
    .unwrap();
    assert_eq!(ec_op.segment_start_receipt(), None);
    assert!(matches!(
        ec_op.launch(),
        Err(PreparedEcOpError::SegmentStartNotIngested)
    ));
    let ingest = ec_op.ingest_segment_start(first.segment_start).unwrap();
    let first_receipt = ec_op.segment_start_receipt().unwrap();
    assert!(ec_op.segment_start_is_current(&first_receipt));
    assert_eq!(ingest.h2d_bytes, 0);
    assert_eq!(ingest.h2d_copies, 0);
    assert_eq!(ingest.fill_calls, 1);
    assert_eq!(ingest.sync_calls, 0);

    clear_multiplicities(&arena, &ec_op);
    arena.context().reset_telemetry();
    let launch = ec_op.launch().unwrap();
    assert_eq!(launch.kernel_launches, 3);
    assert_eq!(launch.allocations, 0);
    assert_eq!(launch.h2d_bytes, 0);
    assert_eq!(launch.d2h_bytes, 0);
    assert_eq!(launch.sync_calls, 0);
    assert_eq!(
        snapshot(&arena, &ec_op),
        expected(&first, &ec_op_requirements)
    );

    clear_multiplicities(&arena, &ec_op);
    let capture = arena.context().capture().unwrap();
    ec_op.launch().unwrap();
    let captured = capture.finish().unwrap();
    assert_eq!(captured.kernel_nodes(), 3);
    clear_multiplicities(&arena, &ec_op);
    captured.launch(arena.context()).unwrap();
    assert_eq!(
        snapshot(&arena, &ec_op),
        expected(&first, &ec_op_requirements)
    );

    execution.ingest(second.tables()).unwrap();
    execution.launch().unwrap();
    ec_op.ingest_segment_start(second.segment_start).unwrap();
    let second_receipt = ec_op.segment_start_receipt().unwrap();
    assert!(!ec_op.segment_start_is_current(&first_receipt));
    assert!(ec_op.segment_start_is_current(&second_receipt));
    assert!(second_receipt.generation() > first_receipt.generation());
    clear_multiplicities(&arena, &ec_op);
    captured.launch(arena.context()).unwrap();
    assert_eq!(
        snapshot(&arena, &ec_op),
        expected(&second, &ec_op_requirements)
    );
}
