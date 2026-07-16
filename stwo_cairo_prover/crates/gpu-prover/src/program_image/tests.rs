use super::*;

use std::path::Path;
use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use crate::arena_plan::{ExecutionTableGeometry, ResidentBackend};
use crate::phases;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::replacement_host_cache::ReplacementHostCache;
use crate::resident_session::ResidentPreWitnessInput;
use crate::resident_witness::planned_cairo_claim;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::shape_executable::{ShapeCompileRequest, ShapeExecutable, ShapeExecutableCache};

#[test]
fn non_kernel_primitives_are_distinct_from_kernel_launches() {
    assert_ne!(
        ExecutionPrimitiveClass::KernelLaunch,
        ExecutionPrimitiveClass::DeviceCopy
    );
    assert_ne!(
        ExecutionPrimitiveClass::KernelLaunch,
        ExecutionPrimitiveClass::Transcript
    );
    assert_ne!(
        ExecutionPrimitiveClass::KernelLaunch,
        ExecutionPrimitiveClass::ProofAssembly
    );
}

fn generated_sn2_executable() -> Arc<ShapeExecutable> {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let ingest = phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
    let proof_plan = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .unwrap();
    let claim = planned_cairo_claim(&ingest.generator, &proof_plan).unwrap();
    let mut cache = ShapeExecutableCache::new(1).unwrap();
    cache
        .compile_or_bind(ShapeCompileRequest {
            claim: &claim,
            proof_plan: &proof_plan,
            preprocessed_trace: &ingest.preprocessed_trace,
            pcs: PcsConfig::default(),
            include_all_preprocessed_columns: false,
            execution_tables: None,
            policy: ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
        })
        .unwrap()
        .executable
}

fn assert_recorded_witness_frontier(
    image: &ArenaProgramInventory,
    arena: &crate::arena_plan::ProofArenaPlan,
) {
    let frontier = image.frontier();
    let planned = arena
        .witness()
        .components
        .iter()
        .find(|planned| {
            frontier.component == Some(planned.component) && frontier.part == Some(planned.part)
        })
        .expect("frontier must name one planned witness component");
    assert_eq!(planned.program.n_mult_tables, 0);
    assert!(planned.requirements.multiplicity_column_words.is_empty());
    assert!(planned.slots.multiplicity_columns.is_empty());
    assert_eq!(
        frontier.missing,
        vec![
            MissingOperationField::PrimitiveAuthority,
            MissingOperationField::LaunchGeometry,
            MissingOperationField::PartitionAuthority,
            MissingOperationField::ReadValueRanges,
            MissingOperationField::EffectContract,
        ]
    );
    assert!(!frontier
        .missing
        .contains(&MissingOperationField::CompleteWriteSet));
    assert!(frontier.launch_candidate.is_some());
    assert_eq!(
        frontier.partition_candidate,
        Some(WitnessPartitionCandidate {
            row_count: planned.requirements.row_count,
            row_granularity: 1,
            global_multiplicity_outputs: 0,
            multiplicity_rule:
                GlobalMultiplicityPartitionRule::CoordinatorOwnedOrCanonicalReduction,
        })
    );
}

#[test]
fn generated_sn2_arena_inventory_is_complete_deterministic_and_fail_closed() {
    let executable = generated_sn2_executable();
    let emit = || {
        ArenaProgramInventory::from_planned_parts(
            executable.topology(),
            executable.transcript(),
            executable.arena(),
        )
        .unwrap()
    };
    let image = emit();
    let repeated = emit();
    let receipt = image.receipt().unwrap();
    assert_eq!(image.identity(), repeated.identity());
    assert_eq!(
        receipt.catalog_values,
        executable.arena().logical_buffers().len()
    );
    assert_eq!(image.output().sections.len(), 8);
    assert_eq!(
        image.output().layout.total_words,
        executable.arena().decommit().proof_bundle.len_words
    );
    assert_eq!(
        receipt.transcript_inputs,
        executable.transcript().inputs().len()
    );
    assert_eq!(
        receipt.transcript_outputs,
        executable.transcript().outputs().len()
    );
    assert!(receipt.pending_operation_values > 0);
    assert_recorded_witness_frontier(&image, executable.arena());
    assert_eq!(receipt.frontier, *image.frontier());
    assert!(matches!(
        image.try_promote_to_compiled_proof(),
        Err(ArenaProgramInventoryError::MissingTypedProducerFrontier(frontier))
            if frontier == receipt.frontier
    ));
}

fn execution_geometry(
    owner: &crate::resident_input::ResidentProverInputOwner,
    claim: &CairoClaim,
) -> ExecutionTableGeometry {
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
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_2.adapted.bin"]
fn sealed_sn2_arena_program_inventory_receipt() {
    const FILE: &str = "SN_PIE_2.adapted.bin";
    const BYTES: usize = 162_102_412;
    const BLAKE3: &str = "5375bd23b012fad243678af013db10498e137642c2cb273e1a1314306aa44b0d";

    let directory = std::env::var("STWO_SN_ADAPTED_DIR").expect("set STWO_SN_ADAPTED_DIR");
    let path = Path::new(&directory).join(FILE);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), BYTES);
    assert_eq!(blake3::hash(&bytes).to_hex().as_str(), BLAKE3);
    let input: ProverInput = bincode::deserialize(&bytes).unwrap();
    drop(bytes);

    let mut host_cache = ReplacementHostCache::new(1).unwrap();
    let ingest = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut host_cache),
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let preprocessed = Arc::clone(&ingest.preprocessed_trace);
    let ResidentPreWitnessInput::ReplacementV1 {
        input: owner,
        template,
    } = ingest.input
    else {
        panic!("replacement ingest returned a legacy owner")
    };
    let claim = template.bind_claim(owner.public_data());
    let geometry = execution_geometry(&owner, &claim);
    let policy = ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048);
    let mut shape_cache = ShapeExecutableCache::new(1).unwrap();
    let executable = template
        .select_shape_executable(
            &mut shape_cache,
            &claim,
            &preprocessed,
            PcsConfig::default(),
            false,
            Some(geometry),
            policy,
        )
        .unwrap()
        .executable;
    let image = ArenaProgramInventory::from_planned_parts(
        executable.topology(),
        executable.transcript(),
        executable.arena(),
    )
    .unwrap();
    let receipt = image.receipt().unwrap();
    assert_eq!(
        receipt.catalog_values,
        executable.arena().logical_buffers().len()
    );
    assert_eq!(receipt.lowered_operations, 0);
    assert!(receipt.pending_operation_values > 0);
    assert_recorded_witness_frontier(&image, executable.arena());
    assert_eq!(receipt.frontier, *image.frontier());
    assert!(matches!(
        image.try_promote_to_compiled_proof(),
        Err(ArenaProgramInventoryError::MissingTypedProducerFrontier(_))
    ));
    eprintln!(
        "SN_ARENA_PROGRAM_INVENTORY_RECEIPT {}",
        serde_json::json!({
            "schema": "stwo.real-sn-arena-program-inventory-frontier.v1",
            "profile": "SN2",
            "fixture": FILE,
            "fixture_blake3": BLAKE3,
            "arena_program_inventory_digest": image
                .identity()
                .digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            "catalog_values": receipt.catalog_values,
            "catalog_words": receipt.catalog_words,
            "external_values": receipt.external_values,
            "fixed_values": receipt.fixed_values,
            "transcript_output_values": receipt.transcript_output_values,
            "pending_operation_values": receipt.pending_operation_values,
            "transcript_inputs": receipt.transcript_inputs,
            "transcript_outputs": receipt.transcript_outputs,
            "proof_bundle_words": receipt.proof_bundle_words,
            "frontier": format!("{:?}", receipt.frontier),
        })
    );
}
