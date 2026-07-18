#![cfg(stwo_cuda_link)]

use std::path::Path;
use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{cuda_device_snapshot, CudaExecContext, DeviceArena};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use super::execution::{
    execute_first_execution_table_big, first_execution_table_arena_layout,
    FirstExecutionTableBigReceipt,
};
use super::{
    emit_recorded_witness_writer_prefix, CompiledWitnessWriterPrefix,
    CompiledWitnessWriterPrefixError,
};
use crate::arena_plan::{ExecutionTableGeometry, ResidentBackend};
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::replacement_host_cache::ReplacementHostCache;
use crate::resident_input::ResidentProverInputOwner;
use crate::resident_session::ResidentPreWitnessInput;
use crate::shape_executable::{ShapeExecutable, ShapeExecutableCache};

#[test]
fn generated_sn2_profile_executes_first_real_compiled_cuda_operation() {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let (owner, executable) = replacement_shape(input);
    let first = execute(&owner, &executable);
    let repeated = execute(&owner, &executable);

    assert_eq!(first, repeated);
    assert_eq!(
        first.wrapper_symbols.each_ref().map(AsRef::as_ref),
        [
            b"memory_limb_split_big_columns_on".as_slice(),
            b"memory_limb_split_small_columns_on".as_slice(),
        ]
    );
    assert_eq!(
        first.rows,
        [
            owner.execution_memory().address_to_id.len(),
            owner.execution_memory().f252_values.len(),
            owner.execution_memory().small_values.len(),
        ]
    );
    assert_eq!(first.next_unsupported_operation.0, 2);
    assert_eq!(
        first.next_unsupported_wrapper_symbol.as_ref(),
        b"stwo_witness_feed_clear_on"
    );
    assert_ne!(first.input_digest, [0; 32]);
    assert_ne!(first.output_digest, [0; 32]);
    report("generated_sn2_profile", &first);
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_2.adapted.bin"]
fn sealed_sn2_executes_first_real_compiled_cuda_operation() {
    const FILE: &str = "SN_PIE_2.adapted.bin";
    const BYTES: usize = 162_102_412;
    const BLAKE3: &str = "5375bd23b012fad243678af013db10498e137642c2cb273e1a1314306aa44b0d";

    let directory = std::env::var("STWO_SN_ADAPTED_DIR").expect("set STWO_SN_ADAPTED_DIR");
    let bytes = std::fs::read(Path::new(&directory).join(FILE)).unwrap();
    assert_eq!(bytes.len(), BYTES);
    assert_eq!(blake3::hash(&bytes).to_hex().as_str(), BLAKE3);
    let input: ProverInput = bincode::deserialize(&bytes).unwrap();
    drop(bytes);

    let (owner, executable) = replacement_shape(input);
    let receipt = execute(&owner, &executable);
    report("sealed_sn2", &receipt);
}

fn report(label: &str, receipt: &FirstExecutionTableBigReceipt) {
    eprintln!(
        "{label}_cuda_ops={:?} rows={:?} column_rows={:?} root_h2d_bytes={} \
         metadata_h2d_bytes={} d2h_bytes={} input_blake3={} output_blake3={} \
         next_unsupported_op={} next_unsupported_wrapper={}",
        receipt.operations.map(|operation| operation.0),
        receipt.rows,
        receipt.column_rows,
        receipt.root_h2d_bytes,
        receipt.metadata_h2d_bytes,
        receipt.validation_d2h_bytes,
        hex::encode(receipt.input_digest),
        hex::encode(receipt.output_digest),
        receipt.next_unsupported_operation.0,
        String::from_utf8_lossy(&receipt.next_unsupported_wrapper_symbol),
    );
}

fn replacement_shape(input: ProverInput) -> (ResidentProverInputOwner, Arc<ShapeExecutable>) {
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
    let mut shape_cache = ShapeExecutableCache::new(1).unwrap();
    let executable = template
        .select_shape_executable(
            &mut shape_cache,
            &claim,
            &preprocessed,
            PcsConfig::default(),
            false,
            Some(geometry),
            ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048),
        )
        .unwrap()
        .executable;
    (owner, executable)
}

fn execution_geometry(
    owner: &ResidentProverInputOwner,
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

fn execute(
    owner: &ResidentProverInputOwner,
    executable: &ShapeExecutable,
) -> FirstExecutionTableBigReceipt {
    let device = cuda_device_snapshot().unwrap();
    let target_sm = device.sm_major * 10 + device.sm_minor;
    let prefix = emitted_prefix(executable, target_sm);
    let layout = first_execution_table_arena_layout(executable.arena()).unwrap();
    let arena = DeviceArena::new(CudaExecContext::new().unwrap(), layout).unwrap();
    execute_first_execution_table_big(&prefix, executable.arena(), &arena, owner).unwrap()
}

fn emitted_prefix(executable: &ShapeExecutable, target_sm: u32) -> CompiledWitnessWriterPrefix {
    match emit_recorded_witness_writer_prefix(
        executable.arena(),
        PreProcessedTraceVariant::Canonical,
        target_sm,
    ) {
        Ok(prefix) => prefix,
        Err(CompiledWitnessWriterPrefixError::MissingTypedAdapter { prefix, .. }) => *prefix,
        Err(error) => panic!("failed to emit executable witness prefix: {error:?}"),
    }
}
