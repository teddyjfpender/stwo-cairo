use std::path::Path;
use std::sync::Arc;

use cairo_air::claims::CairoClaim;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use super::*;
use crate::arena_plan::{CommitmentTreeId, ExecutionTableGeometry, ResidentBackend};
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
    generated_sn2_with_policy(ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048))
}

pub(super) fn generated_sn2_replacement() -> Arc<ShapeExecutable> {
    generated_sn2_with_policy(ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048))
}

pub(super) fn generated_sn2_legacy_with_execution_tables(
    execution_tables: ExecutionTableGeometry,
) -> Arc<ShapeExecutable> {
    generated_sn2_with_policy_and_execution_tables(
        ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
        execution_tables,
    )
}

pub(super) fn generated_casm_blake_replacement() -> Arc<ShapeExecutable> {
    super::blake_g_direct_tests::direct_executable()
}

fn generated_sn2_with_policy(policy: ProtocolPlanPolicy) -> Arc<ShapeExecutable> {
    generated_sn2_with_policy_and_execution_tables(policy, ExecutionTableGeometry::new(19, 17, 5))
}

fn generated_sn2_with_policy_and_execution_tables(
    policy: ProtocolPlanPolicy,
    execution_tables: ExecutionTableGeometry,
) -> Arc<ShapeExecutable> {
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
            execution_tables: Some(execution_tables),
            policy,
        })
        .unwrap()
        .executable
}

fn assert_replacement_base_authority(executable: &ShapeExecutable) {
    let authority = executable
        .replacement_base_producers()
        .expect("ReplacementV1 must install production Base authority");
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    assert!(schedule.interpolation().is_none());
    assert_eq!(
        authority.producer_count(),
        schedule
            .witness_levels()
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    );
    let base = executable
        .arena()
        .commitment(CommitmentTreeId::Base)
        .unwrap();
    assert!(base.interpolation_batches.is_empty());
    assert_eq!(
        Some(authority.direct_retained_b2n()),
        base.direct_retained_b2n_program.as_ref()
    );

    // Base producer authority is deliberately not permission to invent the
    // rest of the proof DAG or promote the diagnostic inventory.
    let image = ArenaProgramInventory::from_planned_parts(
        executable.topology(),
        executable.transcript(),
        executable.arena(),
    )
    .unwrap();
    assert!(image.try_promote_to_compiled_proof().is_err());
}

fn assert_exact_invocation_frontier(executable: &ShapeExecutable) {
    let image = ArenaProgramInventory::from_planned_parts(
        executable.topology(),
        executable.transcript(),
        executable.arena(),
    )
    .unwrap();
    let catalog = BaseProducerCatalog::compile(executable.arena()).unwrap();
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
        &catalog,
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
    let native_producer = mapped.bound[7].native_ec_op().unwrap();
    let native = &native_producer.contract;
    let native_execution = &native_producer.execution;
    let linked_execution = &native_execution.linked;
    let linked =
        producer_prefix::map_scheduled_base_producers(&catalog, executable.arena(), &schedule)
            .unwrap();
    if stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        assert_eq!(linked.bound.len(), 23);
        assert_eq!(linked.missing, None);
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
        "BASE_PRODUCER_SOURCE_RELOCATION_FRONTIER bound={} scheduled={} first={} module_state_ordinals=15,18 ec_op_accesses={} ec_op_launches={} base_outputs={} first_coefficient_id={:?} first_coefficient_logical={:?} runtime_steps={}",
        mapped.bound.len(),
        mapped.scheduled_producers,
        first_producer.producer.component,
        native.effect.accesses().len(),
        native.authority.launches().len(),
        interpolation_columns,
        coefficients.id,
        coefficients.logical,
        schedule.steps().len(),
    );
    assert_eq!(mapped.bound.len(), 23);
    assert_eq!(mapped.scheduled_producers, 23);
    assert_eq!(mapped.missing, None);
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
    let expected = [
        (0, 0, "add_ap_opcode", WitnessProducerKind::Recorded),
        (0, 1, "add_opcode_small", WitnessProducerKind::Recorded),
        (0, 2, "assert_eq_opcode", WitnessProducerKind::Recorded),
        (
            0,
            3,
            "assert_eq_opcode_double_deref",
            WitnessProducerKind::Recorded,
        ),
        (0, 4, "assert_eq_opcode_imm", WitnessProducerKind::Recorded),
        (0, 5, "bitwise_builtin", WitnessProducerKind::Recorded),
        (0, 6, "call_opcode_rel_imm", WitnessProducerKind::Recorded),
        (0, 7, "ec_op_builtin", WitnessProducerKind::NativeEcOp),
        (0, 8, "jnz_opcode_non_taken", WitnessProducerKind::Recorded),
        (0, 9, "jnz_opcode_taken", WitnessProducerKind::Recorded),
        (0, 10, "pedersen_builtin", WitnessProducerKind::Recorded),
        (0, 11, "poseidon_builtin", WitnessProducerKind::Recorded),
        (0, 12, "range_check_builtin", WitnessProducerKind::Recorded),
        (0, 13, "ret_opcode", WitnessProducerKind::Recorded),
        (
            1,
            0,
            "partial_ec_mul_generic",
            WitnessProducerKind::Recorded,
        ),
        (
            1,
            1,
            "pedersen_aggregator_window_bits_18",
            WitnessProducerKind::Recorded,
        ),
        (1, 2, "poseidon_aggregator", WitnessProducerKind::Recorded),
        (1, 3, "verify_instruction", WitnessProducerKind::Recorded),
        (
            2,
            0,
            "partial_ec_mul_window_bits_18",
            WitnessProducerKind::Recorded,
        ),
        (
            2,
            1,
            "poseidon_3_partial_rounds_chain",
            WitnessProducerKind::Recorded,
        ),
        (
            2,
            2,
            "poseidon_full_round_chain",
            WitnessProducerKind::Recorded,
        ),
        (3, 0, "cube_252", WitnessProducerKind::Recorded),
        (
            3,
            1,
            "range_check_252_width_27",
            WitnessProducerKind::Recorded,
        ),
    ];
    assert_eq!(
        scheduled
            .iter()
            .map(|(level, lane, producer)| (*level, *lane, producer.component, producer.kind))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(mapped.scheduled_producers, scheduled.len());
    for (lowered, &(level, lane, expected)) in mapped.bound.iter().zip(&scheduled) {
        assert_eq!(lowered.producer(), expected);
        assert_eq!(lowered.position().level as usize, level);
        assert_eq!(lowered.position().lane as usize, lane);
    }
    assert!(mapped.bound.iter().enumerate().all(|(ordinal, producer)| {
        if ordinal == 7 {
            return producer.native_ec_op().is_some_and(|producer| {
                producer.producer.kind == WitnessProducerKind::NativeEcOp
                    && producer.position.ordinal as usize == ordinal
            });
        }
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
    recorded_deduce_tests::assert_generated_partial_authority(executable, &catalog, &mapped);
    recorded_deduce_tests::assert_generated_pedersen_state_authority(executable, &catalog, &mapped);
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
    let mut empty_values =
        adapter::SemanticValueMap::new(std::iter::empty::<(ArenaCatalogValueId, ValueVersion)>())
            .unwrap();
    assert!(matches!(
        adapter::compile(&first_producer.source.source_arguments, &mut empty_values,),
        Err(InvocationShapeError::MissingSemanticValueMap(_))
    ));
    let mut replay_values = mapped.semantic_values.clone();
    let (invocation, effect) =
        adapter::compile(&first_producer.source.source_arguments, &mut replay_values).unwrap();
    assert_eq!(invocation.arguments.len(), 8);
    assert!(!effect.accesses().is_empty());
    assert_eq!(invocation, first_producer.invocation);
    assert_eq!(effect, first_producer.effect);
    let mut versions = mapped
        .semantic_values
        .allocated_versions()
        .map(|version| version.0)
        .collect::<Vec<_>>();
    versions.sort_unstable();
    assert_eq!(
        versions,
        (0..u32::try_from(versions.len()).unwrap()).collect::<Vec<_>>()
    );
    assert!(!mapped.semantic_values.fixed_values().is_empty());
    assert_eq!(
        mapped.semantic_values.fixed_values().len(),
        mapped.semantic_values.fixed_value_versions().len()
    );
    assert_eq!(
        replay_values.fixed_values(),
        mapped.semantic_values.fixed_values()
    );
    assert!(mapped
        .base_interpolation
        .windows(2)
        .all(|pair| pair[0].batch < pair[1].batch));
    validate_invocation(
        &first_producer.source,
        &catalog,
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
            &catalog,
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
            &catalog,
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
            &catalog,
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
            &catalog,
            executable.arena(),
            first_producer.producer
        ),
        Err(InvocationShapeError::InvocationMismatch)
    );
}

#[test]
fn generated_sn2_source_relocation_frontier_is_exact_and_promotion_stays_closed() {
    let executable = generated_sn2();
    assert!(executable.replacement_base_producers().is_none());
    assert_exact_invocation_frontier(&executable);
}

#[test]
fn generated_sn2_replacement_base_is_direct_and_retains_exact_b2n() {
    let executable = generated_sn2_replacement();
    assert_replacement_base_authority(&executable);
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
    assert_replacement_base_authority(&executable);
}
