use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::memory::DEFAULT_ID;
use stwo_cairo_adapter::opcodes::RECORDED_CASM_DESCRIPTORS;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_prover::witness::jit_prove_backend::{
    lane_recording_metadata_initialization_count, ExecutionMemoryIdentity,
};

use crate::arena_plan::{ExecutionTableGeometry, ResidentBackend};
use crate::phases;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::{prepare_resident_ingest, GpuError, PreparedResidentIngest};
use crate::recorded_witness_inputs::{
    recorded_witness_inputs_for_raw_replacement_plan, recorded_witness_inputs_for_replacement_plan,
    PlannedRecordedWitnessInputs, RecordedInputColumnProvenance,
};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::replacement_host_cache::{
    ReplacementHostCache, ReplacementHostCacheError, ReplacementHostMaterialization,
};
use crate::resident_input::ResidentProverInputOwner;
use crate::resident_session::ResidentPreWitnessInput;
use crate::resident_shape::{
    raw_replacement_compacted_geometry, raw_replacement_proof_plan, RawResidentShapeError,
};
use crate::resident_witness::{planned_cairo_claim, planned_cairo_claim_from_public_data};
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::shape_executable::{ShapeExecutableCache, ShapeExecutableMaterialization};

pub(super) fn assert_cached_recorded_matches_fresh(
    cached: &PlannedRecordedWitnessInputs,
    fresh: &PlannedRecordedWitnessInputs,
) {
    assert_eq!(
        cached.execution_memory_identity,
        fresh.execution_memory_identity
    );
    assert_eq!(cached.lanes.len(), fresh.lanes.len());
    for (cached, fresh) in cached.lanes.iter().zip(&fresh.lanes) {
        assert_eq!(cached.component, fresh.component);
        assert_eq!(cached.program, fresh.program);
        assert_eq!(
            (cached.n_real, cached.row_count),
            (fresh.n_real, fresh.row_count)
        );
        assert_eq!(cached.tables, fresh.tables);
        assert_eq!(cached.host_build_error, fresh.host_build_error);
        assert_eq!(cached.columns.len(), fresh.columns.len());
        for (cached, fresh) in cached.columns.iter().zip(&fresh.columns) {
            match (cached, fresh) {
                (
                    RecordedInputColumnProvenance::StructuralEnabler(cached),
                    RecordedInputColumnProvenance::Host(fresh),
                ) => assert_eq!(cached.as_ref(), fresh.as_slice()),
                _ => assert_eq!(cached, fresh),
            }
        }
    }
}

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
    let mut cache = ReplacementHostCache::new(1).unwrap();
    assert!(matches!(
        prepare_resident_ingest(
            ResidentBackend::ReplacementV1,
            Some(&mut cache),
            input,
            PreProcessedTraceVariant::Canonical,
            None,
        ),
        Err(GpuError::ReplacementHostCache(
            ReplacementHostCacheError::RawShape(RawResidentShapeError::GenericOpcodeUnsupported {
                rows: 1
            })
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
    let mut cache = ReplacementHostCache::new(2).unwrap();

    let PreparedResidentIngest {
        input,
        audit,
        preprocessed_trace: _,
    } = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut cache),
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(audit.claim_generator_constructions, 0);
    assert_eq!(
        audit.replacement_host_cache.unwrap().materialization,
        ReplacementHostMaterialization::Compiled
    );
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
        None,
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(audit.claim_generator_constructions, 1);
    assert!(audit.replacement_host_cache.is_none());
    assert!(matches!(
        input,
        ResidentPreWitnessInput::LegacyResident { .. }
    ));
}

#[test]
fn raw_replacement_cached_claim_rebinds_every_public_data_field() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let owner = ResidentProverInputOwner::encode(input);
    let mut cache = ReplacementHostCache::new(1).unwrap();
    let template = cache
        .compile_or_bind(&owner, PreProcessedTraceVariant::Canonical, None)
        .unwrap()
        .template;
    let base = owner.public_data().clone();
    let base_claim = serde_json::to_value(template.bind_claim(&base)).unwrap();
    let assert_case = |label: &str, public_data: &cairo_air::air::PublicData| {
        let cached = template.bind_claim(public_data);
        let fresh = planned_cairo_claim_from_public_data(public_data, template.exact_plan())
            .unwrap_or_else(|error| panic!("{label}: fresh claim failed: {error}"));
        let cached = serde_json::to_value(cached).unwrap();
        let fresh = serde_json::to_value(fresh).unwrap();
        assert_ne!(cached, base_claim, "{label}: mutation was not observable");
        assert_eq!(cached, fresh, "{label}: cached claim binding drifted");
    };

    macro_rules! scalar_case {
        ($label:expr, $field:expr) => {{
            let mut data = base.clone();
            $field(&mut data);
            assert_case($label, &data);
        }};
    }
    scalar_case!("initial.pc", |data: &mut cairo_air::air::PublicData| {
        data.initial_state.pc.0 ^= 1
    });
    scalar_case!("initial.ap", |data: &mut cairo_air::air::PublicData| {
        data.initial_state.ap.0 ^= 1
    });
    scalar_case!("initial.fp", |data: &mut cairo_air::air::PublicData| {
        data.initial_state.fp.0 ^= 1
    });
    scalar_case!("final.pc", |data: &mut cairo_air::air::PublicData| data
        .final_state
        .pc
        .0 ^=
        1);
    scalar_case!("final.ap", |data: &mut cairo_air::air::PublicData| data
        .final_state
        .ap
        .0 ^=
        1);
    scalar_case!("final.fp", |data: &mut cairo_air::air::PublicData| data
        .final_state
        .fp
        .0 ^=
        1);
    scalar_case!("program.id", |data: &mut cairo_air::air::PublicData| {
        data.public_memory.program[0].0 ^= 1
    });
    scalar_case!("program.value", |data: &mut cairo_air::air::PublicData| {
        data.public_memory.program[0].1[0] ^= 1
    });
    scalar_case!(
        "safe_call_id[0]",
        |data: &mut cairo_air::air::PublicData| data.public_memory.safe_call_ids[0] ^= 1
    );
    scalar_case!(
        "safe_call_id[1]",
        |data: &mut cairo_air::air::PublicData| data.public_memory.safe_call_ids[1] ^= 1
    );
    if !base.public_memory.output.is_empty() {
        scalar_case!("output.id", |data: &mut cairo_air::air::PublicData| data
            .public_memory
            .output[0]
            .0 ^=
            1);
        scalar_case!("output.value", |data: &mut cairo_air::air::PublicData| {
            data.public_memory.output[0].1[0] ^= 1
        });
    }

    macro_rules! segment_cases {
        ($field:ident) => {{
            if base.public_memory.public_segments.$field.is_some() {
                scalar_case!(
                    concat!(stringify!($field), ".start"),
                    |data: &mut cairo_air::air::PublicData| data
                        .public_memory
                        .public_segments
                        .$field
                        .as_mut()
                        .unwrap()
                        .start_ptr
                        .value ^= 1
                );
                scalar_case!(
                    concat!(stringify!($field), ".stop"),
                    |data: &mut cairo_air::air::PublicData| data
                        .public_memory
                        .public_segments
                        .$field
                        .as_mut()
                        .unwrap()
                        .stop_ptr
                        .value ^= 1
                );
            }
        }};
    }
    scalar_case!(
        "output_segment.start",
        |data: &mut cairo_air::air::PublicData| data
            .public_memory
            .public_segments
            .output
            .start_ptr
            .value ^= 1
    );
    scalar_case!(
        "output_segment.stop",
        |data: &mut cairo_air::air::PublicData| data
            .public_memory
            .public_segments
            .output
            .stop_ptr
            .value ^= 1
    );
    segment_cases!(pedersen);
    segment_cases!(range_check_128);
    segment_cases!(ecdsa);
    segment_cases!(bitwise);
    segment_cases!(ec_op);
    segment_cases!(keccak);
    segment_cases!(poseidon);
    segment_cases!(range_check_96);
    segment_cases!(add_mod);
    segment_cases!(mul_mod);
}

#[test]
fn raw_replacement_host_cache_reuses_topology_and_rebinds_statement_memory_and_seeds() {
    let cold_input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let mut warm_input = cold_input.clone();
    warm_input.state_transitions.final_state.fp.0 ^= 1;
    if let Some(segment) = warm_input.builtin_segments.bitwise_builtin.as_mut() {
        assert!(segment.stop_ptr < warm_input.memory.address_to_id.len());
        segment.begin_addr += 1;
        segment.stop_ptr += 1;
    }

    let mut cache = ReplacementHostCache::new(2).unwrap();
    let cold_started = Instant::now();
    let cold = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut cache),
        cold_input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let cold_wall_ns = cold_started.elapsed().as_nanos();
    let cold_preprocessed = Arc::clone(&cold.preprocessed_trace);
    let ResidentPreWitnessInput::ReplacementV1 {
        input: cold_owner,
        template: cold_template,
    } = cold.input
    else {
        panic!("cold replacement dispatch returned legacy input")
    };
    assert_eq!(
        cold.audit.replacement_host_cache.unwrap().materialization,
        ReplacementHostMaterialization::Compiled
    );
    let cold_claim = cold_template.bind_claim(cold_owner.public_data());
    let cold_recorded = cold_template.bind_recorded(&cold_owner).unwrap();
    let cold_memory_ptr = cold_owner.execution_memory().address_to_id.as_ptr();

    let warm_started = Instant::now();
    let warm = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut cache),
        warm_input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let warm_wall_ns = warm_started.elapsed().as_nanos();
    assert!(Arc::ptr_eq(&cold_preprocessed, &warm.preprocessed_trace));
    let ResidentPreWitnessInput::ReplacementV1 {
        input: warm_owner,
        template: warm_template,
    } = warm.input
    else {
        panic!("warm replacement dispatch returned legacy input")
    };
    let warm_audit = warm.audit.replacement_host_cache.unwrap();
    assert_eq!(
        warm_audit.materialization,
        ReplacementHostMaterialization::Reused
    );
    assert_eq!(
        (warm_audit.telemetry.hits, warm_audit.telemetry.misses),
        (1, 1)
    );
    assert_eq!(warm_audit.telemetry.compilations, 1);
    assert!(Arc::ptr_eq(&cold_template, &warm_template));
    assert!(Arc::ptr_eq(
        cold_template.capacity_plan(),
        warm_template.capacity_plan()
    ));
    assert!(Arc::ptr_eq(
        cold_template.exact_plan(),
        warm_template.exact_plan()
    ));
    assert_ne!(
        warm_owner.execution_memory().address_to_id.as_ptr(),
        cold_memory_ptr,
        "the mutation oracle requires a distinct current memory allocation"
    );

    let warm_claim = warm_template.bind_claim(warm_owner.public_data());
    let fresh_claim =
        planned_cairo_claim_from_public_data(warm_owner.public_data(), warm_template.exact_plan())
            .unwrap();
    assert_ne!(
        serde_json::to_value(&cold_claim).unwrap(),
        serde_json::to_value(&warm_claim).unwrap(),
        "proof-varying PublicData must not come from the cached skeleton"
    );
    assert_eq!(
        serde_json::to_value(&warm_claim).unwrap(),
        serde_json::to_value(&fresh_claim).unwrap()
    );

    let rebound = warm_template.bind_recorded(&warm_owner).unwrap();
    let fresh =
        recorded_witness_inputs_for_raw_replacement_plan(&warm_owner, warm_template.exact_plan())
            .unwrap();
    assert!(Arc::ptr_eq(
        &rebound.execution_memory,
        warm_owner.execution_memory()
    ));
    assert_eq!(
        rebound.execution_memory_identity,
        fresh.execution_memory_identity
    );
    assert_cached_recorded_matches_fresh(&rebound, &fresh);
    let cold_enabler = cold_recorded
        .lanes
        .iter()
        .flat_map(|lane| &lane.columns)
        .find_map(|column| match column {
            RecordedInputColumnProvenance::StructuralEnabler(words) => Some(words),
            _ => None,
        })
        .expect("fixture must exercise a cached structural enabler");
    let warm_enabler = rebound
        .lanes
        .iter()
        .flat_map(|lane| &lane.columns)
        .find_map(|column| match column {
            RecordedInputColumnProvenance::StructuralEnabler(words) => Some(words),
            _ => None,
        })
        .expect("warm binding must retain the structural enabler");
    assert!(Arc::ptr_eq(cold_enabler, warm_enabler));

    let fresh_capacity = raw_replacement_proof_plan(
        &warm_owner,
        Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace()),
        None,
    )
    .unwrap();
    let fresh_exact = fresh_capacity
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    assert_eq!(
        warm_template.exact_plan().proof_shape(),
        fresh_exact.proof_shape()
    );

    let execution_geometry = |owner: &ResidentProverInputOwner,
                              claim: &cairo_air::claims::CairoClaim| {
        let public_memory_entries = claim
            .public_data
            .public_memory
            .get_entries(
                claim.public_data.initial_state.pc.0,
                claim.public_data.initial_state.ap.0,
                claim.public_data.final_state.ap.0,
            )
            .count();
        ExecutionTableGeometry::new(
            owner.execution_memory().address_to_id.len(),
            owner.execution_memory().f252_values.len(),
            owner.execution_memory().small_values.len(),
        )
        .with_public_memory_entries(public_memory_entries)
    };
    let cold_geometry = execution_geometry(&cold_owner, &cold_claim);
    let warm_geometry = execution_geometry(&warm_owner, &warm_claim);
    assert_eq!(cold_geometry, warm_geometry);
    let pcs = PcsConfig::default();
    let policy = ProtocolPlanPolicy::replacement_v1(0x1234, 2048);
    let mut executable_cache = ShapeExecutableCache::new(2).unwrap();
    let cold_shape = cold_template
        .select_shape_executable(
            &mut executable_cache,
            &cold_claim,
            &cold_preprocessed,
            pcs,
            false,
            Some(cold_geometry),
            policy,
        )
        .unwrap();
    let warm_shape = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &warm_claim,
            &cold_preprocessed,
            pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        cold_shape.materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert_eq!(
        warm_shape.materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert!(Arc::ptr_eq(&cold_shape.executable, &warm_shape.executable));
    assert_eq!(
        executable_cache.telemetry().topology_key_constructions,
        1,
        "cold plus warm replacement selection must build one full topology key"
    );
    assert_eq!(executable_cache.telemetry().hits, 1);
    assert_eq!(executable_cache.telemetry().compilations, 1);
    assert_eq!(
        executable_cache.telemetry().replacement_handle_lock_ops,
        3,
        "cold selection locks twice to read/install; warm selection reads once"
    );

    let mut changed_statement = warm_claim.clone();
    changed_statement
        .public_data
        .public_memory
        .public_segments
        .bitwise
        .as_mut()
        .expect("SN2 fixture must expose the bitwise public segment")
        .start_ptr
        .value ^= 1;
    let changed_statement_shape = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &changed_statement,
            &cold_preprocessed,
            pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        changed_statement_shape.materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert_ne!(
        warm_shape.bindings, changed_statement_shape.bindings,
        "a current public segment start must rebind statement values"
    );
    assert_eq!(
        executable_cache.telemetry().topology_key_constructions,
        1,
        "statement values do not change executable topology"
    );

    let mut alternate_pcs = pcs;
    alternate_pcs.pow_bits += 1;
    let alternate_shape = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &warm_claim,
            &cold_preprocessed,
            alternate_pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        alternate_shape.materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert_eq!(executable_cache.telemetry().topology_key_constructions, 2);
    let alternate_warm = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &warm_claim,
            &cold_preprocessed,
            alternate_pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        alternate_warm.materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert!(Arc::ptr_eq(
        &alternate_shape.executable,
        &alternate_warm.executable
    ));
    assert_eq!(
        executable_cache.telemetry().topology_key_constructions,
        2,
        "a refreshed B handle must make the next B proof key-free"
    );

    executable_cache.clear_entries_for_test();
    let rebound_after_eviction = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &warm_claim,
            &cold_preprocessed,
            alternate_pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        rebound_after_eviction.materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert!(!Arc::ptr_eq(
        &alternate_shape.executable,
        &rebound_after_eviction.executable
    ));
    assert_eq!(rebound_after_eviction.bindings, alternate_warm.bindings);
    assert_eq!(
        executable_cache.telemetry().topology_key_constructions,
        3,
        "an evicted handle must fall back to full topology admission"
    );
    let warm_after_eviction = warm_template
        .select_shape_executable(
            &mut executable_cache,
            &warm_claim,
            &cold_preprocessed,
            alternate_pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    assert_eq!(
        warm_after_eviction.materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert!(Arc::ptr_eq(
        &rebound_after_eviction.executable,
        &warm_after_eviction.executable
    ));
    assert_eq!(
        executable_cache.telemetry().topology_key_constructions,
        3,
        "the stale fallback must refresh the handle for its next warm proof"
    );
    assert_eq!(executable_cache.telemetry().replacement_handle_lock_ops, 10);
    eprintln!(
        "replacement_host_cache cold_wall_ms={:.3} warm_wall_ms={:.3} warm_identity_ms={:.3} warm_select_ms={:.3}",
        cold_wall_ns as f64 / 1e6,
        warm_wall_ns as f64 / 1e6,
        warm_audit.identity_ns as f64 / 1e6,
        warm_audit.select_ns as f64 / 1e6,
    );
}

#[test]
fn raw_replacement_host_cache_rejects_digest_collision() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let mut changed = input.clone();
    changed.pc_count += 1;
    let cold_owner = ResidentProverInputOwner::encode(input);
    let changed_owner = ResidentProverInputOwner::encode(changed);
    let forced_digest = [0x5a; 32];
    let mut cache = ReplacementHostCache::new(2).unwrap();
    cache
        .compile_or_bind_with_digest_for_test(
            &cold_owner,
            PreProcessedTraceVariant::Canonical,
            None,
            forced_digest,
        )
        .unwrap();
    assert!(matches!(
        cache.compile_or_bind_with_digest_for_test(
            &changed_owner,
            PreProcessedTraceVariant::Canonical,
            None,
            forced_digest,
        ),
        Err(ReplacementHostCacheError::DigestCollision { digest }) if digest == forced_digest
    ));
    assert_eq!(cache.telemetry().collisions, 1);
    assert_eq!(cache.telemetry().compilations, 1);
}

#[test]
fn raw_replacement_host_cache_keys_compacted_geometry_not_raw_ids() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let mut equal_geometry = input.clone();
    let mut changed_geometry = input.clone();
    let segment = input
        .builtin_segments
        .poseidon_builtin
        .expect("cache oracle requires the poseidon compacted builtin segment");
    let source = &input.memory.address_to_id[segment.begin_addr..segment.stop_ptr];
    assert!(
        source.len() >= 12,
        "cache oracle requires at least two poseidon rows"
    );
    let first = source[0];
    let second = source
        .iter()
        .copied()
        .find(|value| *value != first)
        .expect("cache oracle requires two distinct raw ids");
    for value in &mut equal_geometry.memory.address_to_id[segment.begin_addr..segment.stop_ptr] {
        if *value == first {
            *value = second;
        } else if *value == second {
            *value = first;
        }
    }
    let trace = || Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace());
    let base_geometry_owner = ResidentProverInputOwner::encode(input.clone());
    let base_geometry = raw_replacement_compacted_geometry(&base_geometry_owner, trace()).unwrap();
    let base_poseidon_rows = base_geometry[2]
        .rows
        .expect("poseidon fixture must contain compacted geometry")
        .n_real_rows;
    if base_poseidon_rows == 1 {
        let novel = input
            .memory
            .address_to_id
            .iter()
            .copied()
            .find(|value| value.0 != DEFAULT_ID && !source.contains(value))
            .expect("cache oracle requires a valid raw id outside the poseidon segment");
        // One novel cell makes row zero distinct while row one remains the
        // original tuple, changing the exact compacted unique-row count.
        changed_geometry.memory.address_to_id[segment.begin_addr] = novel;
    } else {
        // Collapse every tuple to one canonical key. This preserves segment
        // extent while forcing geometry away from any base count above one.
        for value in
            &mut changed_geometry.memory.address_to_id[segment.begin_addr..segment.stop_ptr]
        {
            *value = first;
        }
    }
    assert_ne!(
        equal_geometry.memory.address_to_id,
        input.memory.address_to_id
    );

    let base_owner = ResidentProverInputOwner::encode(input);
    let equal_owner = ResidentProverInputOwner::encode(equal_geometry);
    let changed_owner = ResidentProverInputOwner::encode(changed_geometry);
    assert_eq!(
        raw_replacement_compacted_geometry(&base_owner, trace()).unwrap(),
        raw_replacement_compacted_geometry(&equal_owner, trace()).unwrap(),
        "a bijective raw-id relabeling must preserve canonical compacted geometry"
    );
    assert_ne!(
        raw_replacement_compacted_geometry(&base_owner, trace()).unwrap(),
        raw_replacement_compacted_geometry(&changed_owner, trace()).unwrap(),
        "the deterministic raw-id mutation must change poseidon compacted geometry"
    );
    let mut cache = ReplacementHostCache::new(1).unwrap();
    let cold = cache
        .compile_or_bind(&base_owner, PreProcessedTraceVariant::Canonical, None)
        .unwrap();
    let equal = cache
        .compile_or_bind(&equal_owner, PreProcessedTraceVariant::Canonical, None)
        .unwrap();
    assert_eq!(
        equal.audit.materialization,
        ReplacementHostMaterialization::Reused,
        "a content mutation preserving canonical compacted geometry must hit"
    );
    assert!(Arc::ptr_eq(&cold.template, &equal.template));

    let fresh_capacity = raw_replacement_proof_plan(&equal_owner, trace(), None).unwrap();
    let fresh_exact = fresh_capacity
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    assert_eq!(
        equal.template.exact_plan().proof_shape(),
        fresh_exact.proof_shape()
    );
    let cached_claim = equal.template.bind_claim(equal_owner.public_data());
    let fresh_claim =
        planned_cairo_claim_from_public_data(equal_owner.public_data(), &fresh_exact).unwrap();
    assert_eq!(
        serde_json::to_value(cached_claim).unwrap(),
        serde_json::to_value(fresh_claim).unwrap()
    );
    let cached_lanes = equal.template.bind_recorded(&equal_owner).unwrap();
    let fresh_lanes =
        recorded_witness_inputs_for_raw_replacement_plan(&equal_owner, &fresh_exact).unwrap();
    assert_cached_recorded_matches_fresh(&cached_lanes, &fresh_lanes);

    let changed = cache
        .compile_or_bind(&changed_owner, PreProcessedTraceVariant::Canonical, None)
        .unwrap();
    assert_eq!(
        changed.audit.materialization,
        ReplacementHostMaterialization::Compiled,
        "a changed canonical compacted geometry must miss"
    );
    assert_eq!(changed.audit.telemetry.misses, 2);
    assert_eq!(changed.audit.telemetry.compilations, 2);
    assert_eq!(changed.audit.telemetry.evictions, 1);
    assert_ne!(
        cold.template.exact_plan().proof_shape(),
        changed.template.exact_plan().proof_shape()
    );
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
