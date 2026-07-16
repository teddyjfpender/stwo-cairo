use std::path::Path;
use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::aot::AotKernelSchemaScope;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use super::*;
use crate::arena_plan::{ExecutionTableGeometry, ResidentBackend};
use crate::compiled_proof::ValueVersion;
use crate::phases;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::replacement_host_cache::ReplacementHostCache;
use crate::resident_runtime::producer_schedule::ProducerScheduleError;
use crate::resident_session::ResidentPreWitnessInput;
use crate::resident_witness::planned_cairo_claim;
use crate::shape_executable::{ShapeCompileRequest, ShapeExecutable, ShapeExecutableCache};

fn generated_sn2() -> Arc<ShapeExecutable> {
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
        .strict_resident_exact(
            &crate::schedule_table::CAIRO_SCHEDULE,
            &crate::relation_table::CAIRO_RELATION_GRAPH,
        )
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
            execution_tables: Some(ExecutionTableGeometry::new(19, 17, 5)),
            policy: ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
        })
        .unwrap()
        .executable
}

fn assert_exact_invocation_frontier(executable: &ShapeExecutable) {
    let image = ArenaProgramInventory::from_planned_parts(
        executable.topology(),
        executable.transcript(),
        executable.arena(),
    )
    .unwrap();
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    schedule.validate_runtime_steps(schedule.steps()).unwrap();
    let mut cursor = schedule.cursor();
    for &step in schedule.steps() {
        cursor.admit(step).unwrap();
    }
    cursor.finish().unwrap();
    let mut incomplete = schedule.steps().to_vec();
    incomplete.pop();
    assert_eq!(
        schedule.validate_runtime_steps(&incomplete),
        Err(ProducerScheduleError::RuntimeOrderMismatch)
    );

    let mapped = map_first_recorded_witness(&image, executable.arena(), &schedule).unwrap();
    let first_interpolation = mapped.base_interpolation.first().unwrap();
    let coefficients = &image.values[first_interpolation.coefficients.0 as usize];
    eprintln!(
        "RECORDED_WITNESS_BINDING_FRONTIER component={} part={:?} base_outputs={} first_coefficient_id={:?} first_coefficient_logical={:?} runtime_steps={}",
        mapped.producer.component,
        mapped.producer.part,
        mapped.base_interpolation.len(),
        coefficients.id,
        coefficients.logical,
        schedule.steps().len(),
    );
    assert_eq!(mapped.producer.kind, WitnessProducerKind::Recorded);
    assert!(mapped
        .base_interpolation
        .iter()
        .all(|frontier| mapped.produced.contains(&frontier.evaluations)));
    assert!(mapped.base_interpolation.iter().all(|frontier| {
        frontier.evaluation_version != frontier.coefficient_version
            && frontier.missing
                == [
                    MissingOperationField::PrimitiveAuthority,
                    MissingOperationField::EffectContract,
                ]
    }));
    assert_eq!(mapped.invocation.source_arguments.len(), 8);
    assert!(matches!(
        adapter::compile(
            &mapped.invocation.source_arguments,
            &adapter::SemanticValueMap::new(
                std::iter::empty::<(ArenaCatalogValueId, ValueVersion,)>()
            )
            .unwrap(),
        ),
        Err(InvocationShapeError::MissingSemanticValueMap(_))
    ));
    let (invocation, effect) =
        adapter::compile(&mapped.invocation.source_arguments, &mapped.semantic_values).unwrap();
    assert_eq!(invocation.arguments.len(), 8);
    assert!(!effect.accesses().is_empty());
    let mut versions = mapped
        .semantic_values
        .entries()
        .map(|(_, version)| version.0)
        .collect::<Vec<_>>();
    versions.sort_unstable();
    assert_eq!(
        versions,
        (0..u32::try_from(versions.len()).unwrap()).collect::<Vec<_>>()
    );
    let coefficient_start = mapped
        .base_interpolation
        .iter()
        .map(|frontier| frontier.coefficient_version.0)
        .min()
        .unwrap();
    assert!(mapped
        .base_interpolation
        .iter()
        .all(|frontier| frontier.evaluation_version.0 < coefficient_start));
    assert_eq!(
        mapped
            .base_interpolation
            .iter()
            .map(|frontier| frontier.coefficient_version.0)
            .collect::<Vec<_>>(),
        (coefficient_start
            ..coefficient_start + u32::try_from(mapped.base_interpolation.len()).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(mapped
        .base_interpolation
        .windows(2)
        .all(|pair| (pair[0].batch, pair[0].column) < (pair[1].batch, pair[1].column)));
    assert_eq!(
        schedule_prefix::try_lower_base_interpolation(&mapped.base_interpolation),
        Err(InvocationShapeError::MissingBaseInterpolationAuthority)
    );
    match loaded_authority::require(&mapped.invocation, 8, 6) {
        Ok(loaded) => {
            assert_ne!(loaded.manifest_identity, [0; 32]);
            assert_eq!(
                loaded.kernel.program_identity(),
                mapped.invocation.program_identity
            );
        }
        Err(InvocationShapeError::MissingLoadedAotAuthority) => {
            assert!(aot::loaded_kernel_authority(mapped.invocation.cache_key, 8, 6).is_none());
        }
        Err(error) => panic!("loaded recorded-witness authority drifted: {error:?}"),
    }
    validate_invocation(&mapped.invocation, &image, executable.arena(), &schedule).unwrap();
    assert!(image.try_promote_to_compiled_proof().is_err());

    let mut mutated = mapped.invocation.clone();
    mutated.program_identity[0] ^= 1;
    assert_eq!(
        validate_invocation(&mutated, &image, executable.arena(), &schedule),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = mapped.invocation.clone();
    mutated.abi_schema_identity[0] ^= 1;
    assert_eq!(
        validate_invocation(&mutated, &image, executable.arena(), &schedule),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = mapped.invocation.clone();
    let SourceArgument::PointerTable { entries, .. } = &mut mutated.source_arguments[0] else {
        panic!("argument zero must be a pointer table")
    };
    entries[0].target.elements.end -= 1;
    assert_eq!(
        validate_invocation(&mutated, &image, executable.arena(), &schedule),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = mapped.invocation;
    mutated.launch.block[0] = 128;
    assert_eq!(
        validate_invocation(&mutated, &image, executable.arena(), &schedule),
        Err(InvocationShapeError::InvocationMismatch)
    );
}

#[test]
fn generated_sn2_recorded_witness_invocation_is_exact_but_not_promoted() {
    assert_exact_invocation_frontier(&generated_sn2());
}

#[test]
fn loaded_authority_field_view_rejects_every_mutation() {
    let invocation = RecordedWitnessInvocationShape {
        program_identity: [1; 32],
        semantic_hash: 2,
        cache_key: 3,
        kernel_symbol: "recorded_witness".into(),
        abi_schema_identity: AotKernelAbiSchema::RecordedWitnessV1.identity(),
        launch: LaunchGeometry {
            grid: [1, 1, 1],
            block: [256, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        source_arguments: Vec::new(),
    };
    let fields = loaded_authority::LoadedAuthorityFields {
        manifest_identity: [4; 32],
        program_identity: invocation.program_identity,
        abi_schema: Some(AotKernelAbiSchema::RecordedWitnessV1),
        abi_schema_identity: invocation.abi_schema_identity,
        schema_scope: AotKernelSchemaScope::StructuredAbi,
        kernel_symbol: invocation.kernel_symbol.clone(),
        semantic_hash: invocation.semantic_hash,
        cache_key: invocation.cache_key,
        target_sm: 86,
        source_identity: [5; 32],
        cubin_identity: [6; 32],
        authority_identity: [7; 32],
    };
    loaded_authority::validate_fields(&invocation, 8, 6, &fields).unwrap();

    let mut mutations = Vec::new();
    let mut changed = fields.clone();
    changed.manifest_identity = [0; 32];
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.program_identity[0] ^= 1;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.abi_schema = None;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.abi_schema_identity[0] ^= 1;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.schema_scope = AotKernelSchemaScope::ExportedSymbolOnly;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.kernel_symbol.push('x');
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.semantic_hash ^= 1;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.cache_key ^= 1;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.target_sm = 89;
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.source_identity = [0; 32];
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.cubin_identity = [0; 32];
    mutations.push(changed);
    let mut changed = fields.clone();
    changed.authority_identity = [0; 32];
    mutations.push(changed);

    for changed in mutations {
        assert_eq!(
            loaded_authority::validate_fields(&invocation, 8, 6, &changed),
            Err(InvocationShapeError::LoadedAotAuthorityMismatch)
        );
    }
    assert_eq!(
        loaded_authority::validate_fields(&invocation, 8, 10, &fields),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
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
fn sealed_sn2_recorded_witness_invocation_is_exact_but_not_promoted() {
    const FILE: &str = "SN_PIE_2.adapted.bin";
    const BYTES: usize = 162_102_412;
    const BLAKE3: &str = "5375bd23b012fad243678af013db10498e137642c2cb273e1a1314306aa44b0d";

    let directory = std::env::var("STWO_SN_ADAPTED_DIR").expect("set STWO_SN_ADAPTED_DIR");
    let bytes = std::fs::read(Path::new(&directory).join(FILE)).unwrap();
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
    assert_exact_invocation_frontier(&executable);
}
