use std::path::Path;
use std::sync::Arc;

use cairo_vm::types::layout_name::LayoutName;
use stwo_cairo_adapter::opcodes::RECORDED_CASM_DESCRIPTORS;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_prover::witness::jit_prove_backend::{
    lane_recording_metadata_initialization_count, ExecutionMemoryIdentity,
};

use crate::arena_plan::ResidentBackend;
use crate::phases;
use crate::prover::{prepare_resident_ingest, GpuError, PreparedResidentIngest};
use crate::recorded_witness_inputs::{
    recorded_witness_inputs_for_raw_replacement_plan, recorded_witness_inputs_for_replacement_plan,
};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::resident_input::ResidentProverInputOwner;
use crate::resident_session::ResidentPreWitnessInput;
use crate::resident_shape::{raw_replacement_proof_plan, RawResidentShapeError};
use crate::resident_witness::{planned_cairo_claim, planned_cairo_claim_from_public_data};
use crate::schedule_table::CAIRO_SCHEDULE;

fn assert_raw_replacement_matches_generator(input: ProverInput, case: &str) {
    let owner = ResidentProverInputOwner::encode(input.clone());
    let ingest = phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
    let raw_capacity = raw_replacement_proof_plan(&owner, ingest.preprocessed_trace.clone(), None)
        .unwrap_or_else(|error| panic!("{case}: raw capacity plan: {error}"));
    assert_eq!(
        raw_capacity.proof_shape(),
        ingest.proof_plan.proof_shape(),
        "{case}: complete capacity shape",
    );
    assert_eq!(
        raw_capacity.relation_graph_hash, ingest.proof_plan.relation_graph_hash,
        "{case}: relation graph",
    );
    let oracle_exact = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap_or_else(|error| panic!("{case}: exact plan: {error}"));
    let raw_exact = raw_capacity
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap_or_else(|error| panic!("{case}: raw exact plan: {error}"));
    assert_eq!(
        raw_exact.proof_shape(),
        oracle_exact.proof_shape(),
        "{case}: complete exact shape",
    );
    let oracle = recorded_witness_inputs_for_replacement_plan(&ingest.generator, &oracle_exact)
        .unwrap_or_else(|error| panic!("{case}: generator planner: {error}"));
    let raw = recorded_witness_inputs_for_raw_replacement_plan(&owner, &raw_exact)
        .unwrap_or_else(|error| panic!("{case}: raw planner: {error}"));
    oracle
        .require_resolved()
        .unwrap_or_else(|error| panic!("{case}: generator unresolved: {error}"));
    raw.require_resolved()
        .unwrap_or_else(|error| panic!("{case}: raw unresolved: {error}"));

    let oracle_claim = planned_cairo_claim(&ingest.generator, &oracle_exact).unwrap();
    let raw_claim = planned_cairo_claim_from_public_data(owner.public_data(), &raw_exact).unwrap();
    assert_eq!(
        serde_json::to_value(raw_claim).unwrap(),
        serde_json::to_value(oracle_claim).unwrap(),
        "{case}: planned claim",
    );

    assert!(Arc::ptr_eq(&raw.execution_memory, owner.execution_memory()));
    assert_eq!(
        raw.execution_memory_identity,
        ExecutionMemoryIdentity::of(owner.execution_memory())
    );
    let generator_memory = ingest.generator.jit_memory.as_ref().unwrap();
    assert!(Arc::ptr_eq(&oracle.execution_memory, generator_memory));
    assert_eq!(
        oracle.execution_memory_identity,
        ExecutionMemoryIdentity::of(generator_memory)
    );
    assert_eq!(
        raw.execution_memory.address_to_id, oracle.execution_memory.address_to_id,
        "{case}: memory address table",
    );
    assert_eq!(
        raw.execution_memory.f252_values, oracle.execution_memory.f252_values,
        "{case}: memory field table",
    );
    assert_eq!(
        raw.execution_memory.small_values, oracle.execution_memory.small_values,
        "{case}: memory small table",
    );

    assert_eq!(raw.lanes.len(), oracle.lanes.len(), "{case}: lane count");
    for (raw_lane, oracle_lane) in raw.lanes.iter().zip(&oracle.lanes) {
        assert_eq!(raw_lane.component, oracle_lane.component, "{case}: order");
        assert_eq!(
            raw_lane.program.semantic_hash(),
            oracle_lane.program.semantic_hash(),
            "{case}/{}: program hash",
            raw_lane.component,
        );
        assert_eq!(
            raw_lane.program, oracle_lane.program,
            "{case}/{}: program",
            raw_lane.component,
        );
        assert_eq!(
            (raw_lane.n_real, raw_lane.row_count),
            (oracle_lane.n_real, oracle_lane.row_count),
            "{case}/{}: row geometry",
            raw_lane.component,
        );
        assert_eq!(
            raw_lane.columns, oracle_lane.columns,
            "{case}/{}: provenance and seed scalars",
            raw_lane.component,
        );
        assert_eq!(
            raw_lane.tables.host_pedersen_points_18, oracle_lane.tables.host_pedersen_points_18,
            "{case}/{}: static table binding",
            raw_lane.component,
        );
        assert_eq!(
            raw_lane.tables.execution_memory, raw.execution_memory_identity,
            "{case}/{}: raw memory identity",
            raw_lane.component,
        );
        assert_eq!(
            oracle_lane.tables.execution_memory, oracle.execution_memory_identity,
            "{case}/{}: oracle memory identity",
            raw_lane.component,
        );
    }

    let raw_ec_op_start = owner
        .builtin_segments()
        .ec_op_builtin
        .map(|segment| segment.begin_addr);
    let oracle_ec_op_start = ingest
        .generator
        .ec_op_builtin
        .as_ref()
        .map(|ec_op| ec_op.ec_op_builtin_segment_start as usize);
    assert_eq!(raw_ec_op_start, oracle_ec_op_start, "{case}: ec_op start");
}

#[test]
fn raw_replacement_planner_matches_generator_on_two_changed_statements() {
    for fixture in [
        "test_prove_verify_sn2_profile",
        "test_prove_verify_poseidon_builtin",
    ] {
        let input = run_and_adapt(
            &get_compiled_cairo_program_path(fixture),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        assert_raw_replacement_matches_generator(input, fixture);
    }
    assert_eq!(lane_recording_metadata_initialization_count(), 1);
}

#[test]
fn raw_replacement_fails_closed_on_nonempty_generic_opcode() {
    let mut input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    input
        .state_transitions
        .casm_states_by_opcode
        .generic_opcode
        .clear();
    input
        .state_transitions
        .casm_states_by_opcode
        .generic_opcode
        .push(input.state_transitions.initial_state);
    assert!(matches!(
        prepare_resident_ingest(
            ResidentBackend::ReplacementV1,
            input,
            PreProcessedTraceVariant::Canonical,
            None,
        ),
        Err(GpuError::RawResidentShape(
            RawResidentShapeError::GenericOpcodeUnsupported { rows: 1 }
        ))
    ));
}

#[test]
fn raw_replacement_production_ingest_is_move_only_and_generator_free() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let address_to_id = input.memory.address_to_id.as_ptr();
    let f252_values = input.memory.f252_values.as_ptr();
    let small_values = input.memory.small_values.as_ptr();
    let public_memory_addresses = input.public_memory_addresses.as_ptr();
    let casm_states = RECORDED_CASM_DESCRIPTORS
        .iter()
        .filter_map(|descriptor| {
            let states = descriptor.states(&input.state_transitions.casm_states_by_opcode);
            (!states.is_empty()).then_some((descriptor.label, states.as_ptr(), states.len()))
        })
        .collect::<Vec<_>>();
    assert!(
        !casm_states.is_empty(),
        "fixture must contain recorded Casm lanes"
    );
    let constructions = stwo_cairo_prover::witness::cairo::claim_generator_constructions();

    let PreparedResidentIngest {
        input,
        audit,
        preprocessed_trace: _,
    } = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(audit.claim_generator_constructions, 0);
    assert_eq!(
        stwo_cairo_prover::witness::cairo::claim_generator_constructions(),
        constructions
    );
    assert!(audit.ingest_ns > 0);
    let ResidentPreWitnessInput::ReplacementV1 { input, .. } = input else {
        panic!("replacement dispatch returned a legacy generator")
    };
    assert_eq!(
        input.execution_memory().address_to_id.as_ptr(),
        address_to_id
    );
    assert_eq!(input.execution_memory().f252_values.as_ptr(), f252_values);
    assert_eq!(input.execution_memory().small_values.as_ptr(), small_values);
    assert_eq!(
        input.public_memory_addresses().as_ptr(),
        public_memory_addresses
    );
    for (label, pointer, rows) in casm_states {
        let moved = input.casm_input(label).unwrap().states;
        assert_eq!(moved.len(), rows, "{label}");
        assert_eq!(moved.as_ptr(), pointer, "{label}");
    }
}

#[test]
fn raw_replacement_dispatch_preserves_legacy_generator_path() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let PreparedResidentIngest { input, audit, .. } = prepare_resident_ingest(
        ResidentBackend::LegacyResident,
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(audit.claim_generator_constructions, 1);
    assert!(matches!(
        input,
        ResidentPreWitnessInput::LegacyResident { .. }
    ));
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing SN_PIE_1..4.adapted.bin"]
fn raw_replacement_planner_matches_generator_on_sn1_through_sn4() {
    let directory = std::env::var("STWO_SN_ADAPTED_DIR")
        .expect("set STWO_SN_ADAPTED_DIR to the sealed adapted-input directory");
    for index in 1..=4 {
        let path = Path::new(&directory).join(format!("SN_PIE_{index}.adapted.bin"));
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let input = bincode::deserialize(&bytes)
            .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()));
        assert_raw_replacement_matches_generator(input, &format!("SN{index}"));
    }
    assert_eq!(lane_recording_metadata_initialization_count(), 1);
}
