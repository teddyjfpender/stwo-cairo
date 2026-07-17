use std::sync::{Arc, OnceLock};

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use super::*;
use crate::arena_plan::{ExecutionTableGeometry, ResidentBackend};
use crate::compiled_proof::{AotArgumentValue, InPlaceAliasRequirement};
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::replacement_host_cache::ReplacementHostCache;
use crate::resident_runtime::producer_schedule::BaseProducerSchedule;
use crate::resident_session::ResidentPreWitnessInput;
use crate::shape_executable::{ShapeExecutable, ShapeExecutableCache};

fn direct_executable() -> Arc<ShapeExecutable> {
    static EXECUTABLE: OnceLock<Arc<ShapeExecutable>> = OnceLock::new();
    Arc::clone(EXECUTABLE.get_or_init(|| {
        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_blake_opcode"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let mut host_cache = ReplacementHostCache::new(1).unwrap();
        let ingest = prepare_resident_ingest(
            ResidentBackend::ReplacementV1,
            Some(&mut host_cache),
            input,
            PreProcessedTraceVariant::CanonicalWithoutPedersen,
            None,
        )
        .unwrap();
        let ResidentPreWitnessInput::ReplacementV1 { input, template } = &ingest.input else {
            panic!("replacement ingest returned legacy input")
        };
        let claim = template.bind_claim(input.public_data());
        let public_memory_entries = claim
            .public_data
            .public_memory
            .get_entries(
                claim.public_data.initial_state.pc.0,
                claim.public_data.initial_state.ap.0,
                claim.public_data.final_state.ap.0,
            )
            .count();
        let geometry = ExecutionTableGeometry::new(
            input.execution_memory().address_to_id.len(),
            input.execution_memory().f252_values.len(),
            input.execution_memory().small_values.len(),
        )
        .with_public_memory_entries(public_memory_entries);
        let mut cache = ShapeExecutableCache::new(1).unwrap();
        template
            .select_shape_executable(
                &mut cache,
                &claim,
                &ingest.preprocessed_trace,
                PcsConfig::default(),
                false,
                Some(geometry),
                ProtocolPlanPolicy::replacement_v1(0x424c_414b_4547, 2048),
            )
            .unwrap()
            .executable
    }))
}

pub(super) fn lowered_direct() -> (
    Arc<ShapeExecutable>,
    blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
) {
    let executable = direct_executable();
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    let producer = schedule
        .witness_levels()
        .iter()
        .flatten()
        .copied()
        .find(|producer| {
            producer.kind
                == crate::resident_runtime::producer_schedule::WitnessProducerKind::BlakeGDirect
        })
        .expect("tiny fixture must schedule direct Blake-G");
    let catalog = BaseProducerCatalog::compile(executable.arena()).unwrap();
    let pending = blake_g_direct_prefix::prepare(&catalog, executable.arena(), producer).unwrap();
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let lowered = blake_g_direct_prefix::lower(pending, &mut values).unwrap();
    (executable, lowered)
}

#[test]
fn real_recorded_program_compiles_into_production_direct_base_authority() {
    let (executable, lowered) = lowered_direct();
    let invocation = static_wrapper_invocation::blake_g_direct(&lowered).unwrap();
    assert_eq!(invocation.arguments.len(), 6);
    assert!(invocation
        .arguments
        .iter()
        .enumerate()
        .all(|(ordinal, argument)| argument.ordinal as usize == ordinal));
    static_wrapper_invocation::validate_blake_g_direct_invocation_for_test(&lowered, &invocation)
        .unwrap();
    for ordinal in [1, 2] {
        let mut changed = invocation.clone();
        let AotArgumentValue::U32(value) = &changed.arguments[ordinal].value else {
            panic!("Blake-G row geometry must be u32")
        };
        changed.arguments[ordinal].value = AotArgumentValue::U32(*value ^ 1);
        assert!(
            static_wrapper_invocation::validate_blake_g_direct_invocation_for_test(
                &lowered, &changed,
            )
            .is_err()
        );
    }
    let authoritative = lowered.authority.abi().arguments();
    assert!(
        static_wrapper_invocation::blake_g_direct_using_abi_for_test(
            &lowered,
            &authoritative[..authoritative.len() - 1],
        )
        .is_err()
    );
    let mut rotated = authoritative.to_vec();
    rotated.rotate_right(1);
    assert!(
        static_wrapper_invocation::blake_g_direct_using_abi_for_test(&lowered, &rotated).is_err()
    );
    for index in 0..authoritative.len() {
        let mut changed = authoritative.to_vec();
        changed[index].ordinal ^= 0x80;
        assert!(
            static_wrapper_invocation::blake_g_direct_using_abi_for_test(&lowered, &changed)
                .is_err()
        );
        let mut changed = authoritative.to_vec();
        changed[index].name = "wrong_role";
        assert!(
            static_wrapper_invocation::blake_g_direct_using_abi_for_test(&lowered, &changed)
                .is_err()
        );
        let mut changed = authoritative.to_vec();
        changed[index].kind =
            if changed[index].kind == stwo_backend_cuda::BlakeGDirectAbiArgumentKind::CudaStream {
                stwo_backend_cuda::BlakeGDirectAbiArgumentKind::U32
            } else {
                stwo_backend_cuda::BlakeGDirectAbiArgumentKind::CudaStream
            };
        assert!(
            static_wrapper_invocation::blake_g_direct_using_abi_for_test(&lowered, &changed)
                .is_err()
        );
        let mut changed = authoritative.to_vec();
        changed[index].access = if changed[index].access
            == stwo_backend_cuda::BlakeGDirectAbiAccess::OrderedExecutionStream
        {
            stwo_backend_cuda::BlakeGDirectAbiAccess::RealRowCount
        } else {
            stwo_backend_cuda::BlakeGDirectAbiAccess::OrderedExecutionStream
        };
        assert!(
            static_wrapper_invocation::blake_g_direct_using_abi_for_test(&lowered, &changed)
                .is_err()
        );
    }
    let component = executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| component.component == "blake_g")
        .unwrap();
    let public_contract = stwo_backend_cuda::BlakeGDirectCompositeContract::compile(
        &component.program,
        component.n_real_rows,
        component.requirements.row_count,
    )
    .unwrap();
    assert_eq!(public_contract, lowered.authority);
    assert_eq!(lowered.invocation.inputs.len(), 6);
    assert_eq!(lowered.invocation.traces.len(), 53);
    assert_eq!(lowered.invocation.luts.len(), 4);
    assert_eq!(lowered.invocation.counts.len(), 5);
    assert_eq!(lowered.effect.accesses().len(), 6 + 53 + 4 + 5);

    let authority = executable
        .replacement_base_producers()
        .expect("ReplacementV1 must retain Base authority in production");
    let schedule = BaseProducerSchedule::compile(executable.arena()).unwrap();
    assert_eq!(
        authority.producer_count(),
        schedule
            .witness_levels()
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    );

    let module = [9; 32];
    let linked =
        blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority::bind_exact(
            &public_contract,
            module,
            module,
            89,
            89,
        )
        .unwrap();
    assert_ne!(linked.identity, [0; 32]);
    assert!(
        blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority::bind_exact(
            &public_contract,
            module,
            module,
            89,
            90,
        )
        .is_err()
    );
}

#[test]
fn direct_lowering_rejects_every_semantic_mutation() {
    let (_, lowered) = lowered_direct();
    blake_g_direct_prefix::validate_lowered(
        &lowered.authority,
        &lowered.invocation,
        &lowered.effect,
    )
    .unwrap();

    let mut mutations = Vec::new();
    let mut changed = lowered.invocation.clone();
    changed.inputs[0].value.value_words.end -= 1;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.traces[0].value.value_words.start = 1;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.luts.swap(0, 1);
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.counts.swap(0, 1);
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.counts[0].destination = changed.counts[0].source;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.counts[0].alias.requirement = InPlaceAliasRequirement::Permitted;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.traces[0].binding.0 += 1;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.n_real_rows -= 1;
    mutations.push(changed);
    let mut changed = lowered.invocation.clone();
    changed.padded_rows *= 2;
    mutations.push(changed);

    for changed in mutations {
        assert!(blake_g_direct_prefix::validate_lowered(
            &lowered.authority,
            &changed,
            &lowered.effect,
        )
        .is_err());
    }
}

#[test]
fn direct_geometry_and_nonalias_checks_reject_short_extra_and_overlap() {
    let executable = direct_executable();
    let component = executable
        .arena()
        .witness()
        .components
        .iter()
        .find(|component| component.component == "blake_g")
        .unwrap();
    let authority = stwo_backend_cuda::BlakeGDirectCompositeContract::compile(
        &component.program,
        component.n_real_rows,
        component.requirements.row_count,
    )
    .unwrap();
    assert!(blake_g_direct_prefix::component_geometry_is_exact(
        component, &authority
    ));

    let mut short = component.clone();
    short
        .input_gather
        .as_mut()
        .unwrap()
        .requirements
        .consumer_input_column_words
        .pop();
    assert!(!blake_g_direct_prefix::component_geometry_is_exact(
        &short, &authority
    ));
    let mut extra = component.clone();
    extra
        .input_gather
        .as_mut()
        .unwrap()
        .requirements
        .consumer_input_column_words
        .push(authority.padded_rows());
    assert!(!blake_g_direct_prefix::component_geometry_is_exact(
        &extra, &authority
    ));
    let mut enabler_extent = component.clone();
    enabler_extent.requirements.input_column_words[6] -= 1;
    assert!(!blake_g_direct_prefix::component_geometry_is_exact(
        &enabler_extent,
        &authority
    ));

    assert!(blake_g_direct_prefix::intervals_are_disjoint(&mut [
        (0, 4),
        (4, 8),
        (12, 16),
    ]));
    assert!(!blake_g_direct_prefix::intervals_are_disjoint(&mut [
        (0, 8),
        (4, 12)
    ]));
}

#[test]
fn direct_receipt_rejects_same_shape_in_range_lut_substitution() {
    use stwo_backend_cuda::BlakeGDirectLutContentIdentity;
    use stwo_cairo_prover::witness::device_feed::canonical_count_lut;

    const FAMILIES: [&str; 4] = [
        "verify_bitwise_xor_8_state",
        "verify_bitwise_xor_4_state",
        "verify_bitwise_xor_7_state",
        "verify_bitwise_xor_9_state",
    ];
    let variant = PreProcessedTraceVariant::CanonicalWithoutPedersen;
    for variant in PreProcessedTraceVariant::ALL_VARIANTS {
        let generated =
            blake_g_direct_execution_authority::generated_canonical_lut_content_identity(variant)
                .unwrap();
        assert_eq!(
            generated.identity(),
            blake_g_direct_execution_authority::CANONICAL_BLAKE_G_DIRECT_LUT_CONTENT_ID_V1
        );
    }
    let expected =
        blake_g_direct_execution_authority::generated_canonical_lut_content_identity(variant)
            .unwrap();
    let trace = Arc::new(variant.to_preprocessed_trace());
    let mut luts = FAMILIES.map(|family| {
        canonical_count_lut(family, Arc::clone(&trace)).expect("canonical direct LUT")
    });
    luts[0].swap(0, 1);
    assert!(luts
        .iter()
        .all(|lut| lut.iter().all(|&row| (row as usize) < lut.len())));
    let substituted =
        BlakeGDirectLutContentIdentity::from_host_words([&luts[0], &luts[1], &luts[2], &luts[3]])
            .unwrap();
    assert_ne!(expected, substituted);
    assert!(blake_g_direct_execution_authority::canonical_lut_content_is_exact(&expected));
    assert!(!blake_g_direct_execution_authority::canonical_lut_content_is_exact(&substituted));

    let (_, lowered) = lowered_direct();
    let module = [9; 32];
    let linked =
        blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority::bind_exact(
            &lowered.authority,
            module,
            module,
            89,
            89,
        )
        .unwrap();
    let canonical_receipt = blake_g_direct_execution_authority::paired_identity(
        &linked,
        [7; 32],
        lowered.effect.id(),
        expected,
    );
    let substituted_receipt = blake_g_direct_execution_authority::paired_identity(
        &linked,
        [7; 32],
        lowered.effect.id(),
        substituted,
    );
    assert_ne!(canonical_receipt, substituted_receipt);
}
