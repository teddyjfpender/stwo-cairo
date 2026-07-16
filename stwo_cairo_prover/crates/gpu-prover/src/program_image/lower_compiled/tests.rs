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
use crate::compiled_proof::{
    AotArgumentValue, AtomicOperation, EffectAccess, InPlaceAliasRequirement, ValueVersion,
};
use crate::phases;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::replacement_host_cache::ReplacementHostCache;
use crate::resident_runtime::producer_schedule::ProducerScheduleError;
use crate::resident_session::ResidentPreWitnessInput;
use crate::resident_witness::planned_cairo_claim;
use crate::shape_executable::{ShapeCompileRequest, ShapeExecutable, ShapeExecutableCache};

pub(super) fn generated_sn2() -> Arc<ShapeExecutable> {
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

    // The Mac build links the explicit no-CUDA stub. Inject one exact nonzero
    // module receipt here to exercise the post-link schedule frontier without
    // weakening the production binder's linked-receipt requirement.
    let module_identity = [9; 32];
    let mapped = producer_prefix::map_scheduled_base_producers_with_native_authority(
        &image,
        executable.arena(),
        &schedule,
        |contract| {
            ec_op_execution_authority::NativeEcOpLinkedModuleAuthority::bind_exact(
                contract,
                module_identity,
                module_identity,
                89,
            )
            .map(Some)
        },
    )
    .unwrap();
    let first_producer = mapped.bound.first().unwrap().recorded().unwrap();
    let missing = mapped.missing.unwrap();
    let native_producer = mapped.bound[7].native_ec_op().unwrap();
    let native = &native_producer.contract;
    let native_execution = &native_producer.execution;
    let linked_execution = &native_execution.linked;
    let linked =
        producer_prefix::map_scheduled_base_producers(&image, executable.arena(), &schedule)
            .unwrap();
    if stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        assert_eq!(linked.bound.len(), 9);
        assert_eq!(linked.missing, mapped.missing);
        let linked_native = linked.bound[7].native_ec_op().unwrap();
        assert_eq!(
            linked_native.execution.linked.static_module_build_identity,
            stwo_backend_cuda_kernels::expected_static_cuda_module_build_identity()
        );
        assert_eq!(
            stwo_backend_cuda_kernels::static_cuda_module_target_sms(),
            &[linked_native.execution.linked.consumer_target_sm]
        );
    } else {
        assert_eq!(linked.bound.len(), 7);
        assert_eq!(
            linked.missing.unwrap().missing,
            producer_prefix::MissingProducerAuthorityKind::NativeEcOpStaticModuleBuildIdentity
        );
    }
    let first_interpolation = mapped.base_interpolation.first().unwrap();
    let first_column = first_interpolation.invocation.columns.first().unwrap();
    let coefficients = &image.values[first_column.coefficients.0 as usize];
    let interpolation_columns = mapped
        .base_interpolation
        .iter()
        .map(|batch| batch.invocation.columns.len())
        .sum::<usize>();
    eprintln!(
        "BASE_PRODUCER_BINDING_FRONTIER bound={} scheduled={} first={} missing={} missing_kind={:?} missing_position={:?} ec_op_accesses={} ec_op_launches={} base_outputs={} first_coefficient_id={:?} first_coefficient_logical={:?} runtime_steps={}",
        mapped.bound.len(),
        mapped.scheduled_producers,
        first_producer.producer.component,
        missing.producer.component,
        missing.missing,
        missing.position,
        native.effect.accesses().len(),
        native.authority.launches().len(),
        interpolation_columns,
        coefficients.id,
        coefficients.logical,
        schedule.steps().len(),
    );
    assert_eq!(mapped.bound.len(), 9);
    assert_eq!(mapped.scheduled_producers, 23);
    assert_eq!(missing.position.ordinal as usize, mapped.bound.len());
    assert_eq!(missing.producer.component, "partial_ec_mul_window_bits_18");
    assert_eq!(
        missing.producer.part,
        Some(stwo_cairo_prover::witness::proof_shape::TracePartId::Main)
    );
    assert_eq!(missing.producer.kind, WitnessProducerKind::Recorded);
    assert_eq!(
        missing.missing,
        producer_prefix::MissingProducerAuthorityKind::ModuleGlobalEffects(
            DeduceKind::PartialEcMulW18
        )
    );
    assert_eq!(missing.position.ordinal, 9);
    let scheduled = schedule
        .witness_levels()
        .iter()
        .enumerate()
        .flat_map(|(level, producers)| {
            producers
                .iter()
                .copied()
                .enumerate()
                .map(move |(lane, producer)| (level, lane, producer))
        })
        .collect::<Vec<_>>();
    assert_eq!(mapped.scheduled_producers, scheduled.len());
    for (lowered, &(level, lane, expected)) in mapped.bound.iter().zip(&scheduled) {
        assert_eq!(lowered.producer(), expected);
        assert_eq!(lowered.position().level as usize, level);
        assert_eq!(lowered.position().lane as usize, lane);
    }
    assert_eq!(missing.producer, scheduled[mapped.bound.len()].2);
    assert_eq!(
        missing.position.level as usize,
        scheduled[mapped.bound.len()].0
    );
    assert_eq!(
        missing.position.lane as usize,
        scheduled[mapped.bound.len()].1
    );
    assert!(mapped.bound[..7]
        .iter()
        .enumerate()
        .all(|(ordinal, producer)| {
            producer.recorded().is_some_and(|producer| {
                producer.producer.kind == WitnessProducerKind::Recorded
                    && producer.position.ordinal as usize == ordinal
                    && producer.source.program_identity != [0; 32]
                    && producer.source.abi_schema_identity
                        == AotKernelAbiSchema::RecordedWitnessV1.identity()
                    && producer.source.source_arguments.len() == 8
                    && producer.invocation.arguments.len() == 8
                    && !producer.effect.accesses().is_empty()
            })
        }));
    assert_eq!(native_producer.position.ordinal, 7);
    assert_eq!(native_producer.producer.component, "ec_op_builtin");
    assert_eq!(
        native_producer.producer.kind,
        WitnessProducerKind::NativeEcOp
    );
    recorded_deduce_tests::assert_generated_partial_authority(executable, &image, &mapped);
    let bitwise = mapped
        .bound
        .iter()
        .filter_map(producer_prefix::LoweredBaseProducer::recorded)
        .find(|producer| producer.producer.component == "bitwise_builtin")
        .unwrap();
    let SourceArgument::PointerTable { entries, .. } = &bitwise.source.source_arguments[0] else {
        panic!("bitwise argument zero must be its input pointer table")
    };
    assert_eq!(entries.len(), 3);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.target.access)
            .collect::<Vec<_>>(),
        [
            InvocationAccess::Read,
            InvocationAccess::Inactive,
            InvocationAccess::Read,
        ]
    );
    let AotArgumentValue::DevicePointerTable(entries) = &bitwise.invocation.arguments[0].value
    else {
        panic!("bitwise argument zero must lower to its input pointer table")
    };
    assert_eq!(entries.len(), 3);
    assert!(entries[0].is_some());
    assert_eq!(entries[1], None);
    assert!(entries[2].is_some());
    assert_eq!(
        native.authority.abi(),
        stwo_backend_cuda::EcOpCompositeAbi::ProjectiveChainNormalizePaddingV1
    );
    assert_eq!(native.authority.abi().arguments().len(), 19);
    assert_eq!(
        native.authority.abi().entry_symbol(),
        "ec_op_builtin_witness_on"
    );
    assert_eq!(native.authority.launches().len(), 3);
    assert_eq!(
        native.authority.effect(),
        stwo_backend_cuda::EcOpEffectAbi::FullTraceLookupPartialAndAtomicMultiplicitiesV1
    );
    for identity in [
        native.authority.source_identity(),
        native.authority.abi_identity(),
        native.authority.effect_identity(),
        native.authority.launch_identity(),
        native.authority.identity(),
    ] {
        assert_ne!(identity, [0; 32]);
    }
    assert_eq!(
        linked_execution.static_module_build_identity,
        module_identity
    );
    assert_eq!(
        linked_execution.expected_static_module_build_identity,
        module_identity
    );
    assert_eq!(linked_execution.consumer_target_sm, 89);
    assert_eq!(linked_execution.abi, native.authority.abi());
    assert_eq!(linked_execution.effect, native.authority.effect());
    assert_eq!(
        linked_execution.source_identity,
        native.authority.source_identity()
    );
    assert_eq!(
        linked_execution.abi_identity,
        native.authority.abi_identity()
    );
    assert_eq!(
        linked_execution.effect_identity,
        native.authority.effect_identity()
    );
    assert_eq!(
        linked_execution.launch_identity,
        native.authority.launch_identity()
    );
    assert_eq!(
        linked_execution.contract_identity,
        native.authority.identity()
    );
    assert_eq!(linked_execution.entry_symbol, "ec_op_builtin_witness_on");
    assert_eq!(
        linked_execution.launches.map(|launch| launch.entry_symbol),
        [
            "ec_op_projective_chain_kernel",
            "ec_op_normalize_round_tiles_kernel",
            "partial_input_padding_kernel",
        ]
    );
    assert_eq!(
        linked_execution.launches.map(|launch| launch.launch),
        *native.authority.launches()
    );
    assert_ne!(linked_execution.identity, [0; 32]);
    assert_ne!(native_execution.invocation_identity, [0; 32]);
    assert_eq!(
        native_execution.compiled_effect_identity,
        native.effect.id()
    );
    assert_ne!(native_execution.identity, linked_execution.identity);
    assert_eq!(
        (*native.authority.launches()).map(|launch| launch.stage),
        [
            stwo_backend_cuda::EcOpKernelStage::ProjectiveChain,
            stwo_backend_cuda::EcOpKernelStage::NormalizeRoundTiles,
            stwo_backend_cuda::EcOpKernelStage::PartialInputPadding,
        ]
    );
    assert_eq!(native.invocation.execution_tables.len(), 37);
    assert_eq!(native.invocation.trace_columns.len(), 273);
    assert_eq!(native.invocation.partial_input_columns.len(), 127);
    assert_eq!(native.invocation.multiplicities.len(), 4);
    let row_count = native.invocation.row_count;
    assert_eq!(
        native.authority.launches()[0].grid,
        [row_count.div_ceil(16), 1, 1]
    );
    assert_eq!(native.authority.launches()[0].block, [16, 1, 1]);
    assert_eq!(
        native.authority.launches()[1].grid,
        [row_count.div_ceil(64), 63, 1]
    );
    assert_eq!(native.authority.launches()[1].block, [64, 1, 1]);
    assert_eq!(
        native.authority.launches()[2].grid,
        [(row_count * 4).div_ceil(64), 1, 1]
    );
    assert_eq!(native.authority.launches()[2].block, [64, 1, 1]);
    assert_eq!(
        native.invocation.execution_table_pointers.value_words.len(),
        37 * POINTER_WORDS
    );
    assert_eq!(
        native.invocation.lookup_words.value.value_words.len(),
        usize::try_from(native.invocation.row_count).unwrap() * 488
    );
    assert_eq!(
        native.invocation.partial_row_count,
        native.invocation.row_count * 256
    );
    assert!(native
        .invocation
        .execution_tables
        .iter()
        .all(|binding| { binding.value.value_words.is_empty() == binding.binding.is_none() }));
    assert!(native
        .invocation
        .trace_columns
        .iter()
        .all(|binding| binding.binding.is_some()
            && binding.value.value_words.len()
                == usize::try_from(native.invocation.row_count).unwrap()));
    for (index, binding) in native.invocation.partial_input_columns.iter().enumerate() {
        let value = &image.values[binding.value.value.0 as usize];
        assert!(binding.binding.is_some());
        assert_eq!(
            binding.value.value_words.len(),
            usize::try_from(native.invocation.partial_row_count).unwrap()
        );
        if index + 1 == native.invocation.partial_input_columns.len() {
            assert_eq!(value.component, Some("ec_op_builtin"));
            assert_eq!(value.purpose, BufferPurpose::EcOpPartialIota);
        } else {
            assert_eq!(value.component, Some("partial_ec_mul_generic"));
            assert_eq!(value.purpose, BufferPurpose::WitnessInput);
            assert_eq!(value.ordinal as usize, index);
        }
    }
    assert_eq!(native.effect.accesses().len(), 37 + 1 + 273 + 1 + 127 + 4);
    let atomic = native
        .effect
        .accesses()
        .iter()
        .filter_map(|access| match access {
            EffectAccess::Atomic {
                source,
                destination,
                operation,
                in_place,
            } => Some((source, destination, operation, in_place)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(atomic.len(), 4);
    for (index, ((source, destination, operation, in_place), receipt)) in atomic
        .into_iter()
        .zip(&native.invocation.multiplicities)
        .enumerate()
    {
        assert_eq!(*operation, AtomicOperation::AddU32);
        assert_eq!(source.binding, destination.binding);
        assert_eq!(source.binding, receipt.binding);
        assert_eq!(source.value.version, receipt.source);
        assert_eq!(destination.value.version, receipt.destination);
        assert_ne!(receipt.source, receipt.destination);
        assert_eq!(in_place, &receipt.alias);
        assert_eq!(in_place.id.0 as usize, index);
        assert_eq!(in_place.requirement, InPlaceAliasRequirement::Required);
    }
    let interpolation = schedule.interpolation().unwrap();
    assert_eq!(mapped.base_interpolation.len(), interpolation.batches.len());
    assert_eq!(
        interpolation_columns,
        interpolation
            .batches
            .iter()
            .map(|batch| batch.columns.len())
            .sum::<usize>()
    );
    for (batch_index, (lowered, scheduled)) in mapped
        .base_interpolation
        .iter()
        .zip(&interpolation.batches)
        .enumerate()
    {
        assert_eq!(lowered.batch as usize, batch_index);
        assert_eq!(lowered.authority, scheduled.authority);
        assert_ne!(lowered.authority.identity(), [0; 32]);
        assert_eq!(lowered.invocation.log_size, scheduled.log_size);
        assert_eq!(
            lowered.invocation.column_count as usize,
            scheduled.columns.len()
        );
        assert_eq!(
            lowered.invocation.evaluation_domain_size,
            scheduled.authority.evaluation_domain_size()
        );
        assert_eq!(
            lowered.effect.accesses().len(),
            lowered.invocation.columns.len() + 1
        );
        for (column, access) in lowered
            .invocation
            .columns
            .iter()
            .zip(lowered.effect.accesses())
        {
            let EffectAccess::ReadWrite {
                source,
                destination,
                in_place,
            } = access
            else {
                panic!("interpolation column must be one exact transition")
            };
            assert_eq!(source.binding, column.source);
            assert_eq!(destination.binding, column.destination);
            assert_eq!(
                in_place.is_some_and(|authority| {
                    authority.requirement == InPlaceAliasRequirement::Required
                }),
                column.exact_in_place
            );
            let evaluations = &image.values[column.evaluations.0 as usize];
            if evaluations.component == Some(first_producer.producer.component)
                && evaluations.part == first_producer.producer.part
            {
                assert!(first_producer.produced.contains(&column.evaluations));
            }
        }
        let EffectAccess::Read { source } = lowered.effect.accesses().last().unwrap() else {
            panic!("interpolation must read the exact inverse-twiddle suffix")
        };
        assert_eq!(source.binding, lowered.invocation.inverse_twiddles);
    }
    assert_eq!(first_producer.source.source_arguments.len(), 8);
    assert!(matches!(
        adapter::compile(
            &first_producer.source.source_arguments,
            &adapter::SemanticValueMap::new(
                std::iter::empty::<(ArenaCatalogValueId, ValueVersion,)>()
            )
            .unwrap(),
        ),
        Err(InvocationShapeError::MissingSemanticValueMap(_))
    ));
    let (invocation, effect) = adapter::compile(
        &first_producer.source.source_arguments,
        &mapped.semantic_values,
    )
    .unwrap();
    assert_eq!(invocation.arguments.len(), 8);
    assert!(!effect.accesses().is_empty());
    assert_eq!(invocation, first_producer.invocation);
    assert_eq!(effect, first_producer.effect);
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
    assert!(mapped
        .base_interpolation
        .windows(2)
        .all(|pair| pair[0].batch < pair[1].batch));
    match loaded_authority::require(&first_producer.source, 8, 6) {
        Ok(loaded) => {
            assert_ne!(loaded.manifest_identity, [0; 32]);
            assert_eq!(
                loaded.kernel.program_identity(),
                first_producer.source.program_identity
            );
        }
        Err(InvocationShapeError::MissingLoadedAotAuthority) => {
            assert!(aot::loaded_kernel_authority(first_producer.source.cache_key, 8, 6).is_none());
        }
        Err(error) => panic!("loaded recorded-witness authority drifted: {error:?}"),
    }
    validate_invocation(
        &first_producer.source,
        &image,
        executable.arena(),
        first_producer.producer,
    )
    .unwrap();
    // This advances only the typed semantic migration frontier. The arena
    // inventory still cannot claim or emit a real `CompiledProof`.
    assert!(image.try_promote_to_compiled_proof().is_err());

    let mut mutated = first_producer.source.clone();
    mutated.program_identity[0] ^= 1;
    assert_eq!(
        validate_invocation(
            &mutated,
            &image,
            executable.arena(),
            first_producer.producer
        ),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = first_producer.source.clone();
    mutated.abi_schema_identity[0] ^= 1;
    assert_eq!(
        validate_invocation(
            &mutated,
            &image,
            executable.arena(),
            first_producer.producer
        ),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = first_producer.source.clone();
    let SourceArgument::PointerTable { entries, .. } = &mut mutated.source_arguments[0] else {
        panic!("argument zero must be a pointer table")
    };
    entries[0].target.elements.end -= 1;
    assert_eq!(
        validate_invocation(
            &mutated,
            &image,
            executable.arena(),
            first_producer.producer
        ),
        Err(InvocationShapeError::InvocationMismatch)
    );

    let mut mutated = first_producer.source.clone();
    mutated.launch.block[0] = 128;
    assert_eq!(
        validate_invocation(
            &mutated,
            &image,
            executable.arena(),
            first_producer.producer
        ),
        Err(InvocationShapeError::InvocationMismatch)
    );
}

#[test]
fn generated_sn2_schedule_prefix_is_exact_and_stops_at_missing_authority() {
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
        deduce: recorded_deduce_authority::empty_for_test(),
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
        source_identity: invocation.deduce.source_identity,
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
    changed.source_identity[0] ^= 1;
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
