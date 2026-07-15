use std::sync::OnceLock;

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use super::*;
use crate::phases;
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::resident_witness::planned_cairo_claim;
use crate::schedule_table::CAIRO_SCHEDULE;

struct Sn2Fixture {
    claim: CairoClaim,
    proof_plan: Arc<ProofPlan>,
    preprocessed_trace: Arc<PreProcessedTrace>,
}

fn sn2_fixture() -> &'static Sn2Fixture {
    static FIXTURE: OnceLock<Sn2Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let ingest = phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
        let proof_plan = Arc::new(
            ingest
                .proof_plan
                .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
                .unwrap(),
        );
        let claim = planned_cairo_claim(&ingest.generator, &proof_plan).unwrap();
        Sn2Fixture {
            claim,
            proof_plan,
            preprocessed_trace: ingest.preprocessed_trace,
        }
    })
}

fn sn2_request<'a>(claim: &'a CairoClaim, fixture: &'a Sn2Fixture) -> ShapeCompileRequest<'a> {
    ShapeCompileRequest {
        claim,
        proof_plan: &fixture.proof_plan,
        preprocessed_trace: &fixture.preprocessed_trace,
        pcs: PcsConfig::default(),
        include_all_preprocessed_columns: false,
        execution_tables: None,
        policy: ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
    }
}

fn sn2_executable() -> Arc<ShapeExecutable> {
    let fixture = sn2_fixture();
    let mut cache = ShapeExecutableCache::new(1).unwrap();
    let cold = cache
        .compile_or_bind(sn2_request(&fixture.claim, fixture))
        .unwrap();
    let warm = cache
        .compile_or_bind(sn2_request(&fixture.claim, fixture))
        .unwrap();
    assert_eq!(warm.materialization, ShapeExecutableMaterialization::Reused);
    assert!(Arc::ptr_eq(&cold.executable, &warm.executable));
    assert_eq!(cache.telemetry().compilations, 1);
    assert_eq!(cache.telemetry().hits, 1);
    assert_eq!(cache.telemetry().topology_key_constructions, 2);
    cold.executable
}

#[test]
fn noncanonical_preprocessed_geometry_is_rejected_before_admission() {
    let fixture = sn2_fixture();
    let mut cache = ShapeExecutableCache::new(1).unwrap();

    // No two canonical variants have identical ordered geometry. Forge that
    // exact adversarial case by relabeling the canonical-without-Pedersen
    // columns as Canonical; admission must reject the declaration.
    let mut forged_variant =
        PreProcessedTraceVariant::CanonicalWithoutPedersen.to_preprocessed_trace();
    forged_variant.variant = PreProcessedTraceVariant::Canonical;
    let error = cache
        .compile_or_bind(ShapeCompileRequest {
            preprocessed_trace: &forged_variant,
            ..sn2_request(&fixture.claim, fixture)
        })
        .err()
        .expect("forged preprocessed variant must fail closed");
    assert!(matches!(
        error,
        ShapeExecutableError::NonCanonicalPreprocessedGeometry {
            variant: PreProcessedTraceVariant::Canonical,
            supplied_columns: 105,
            expected_columns: 161,
            first_mismatch: _,
        }
    ));

    let mut reordered = PreProcessedTraceVariant::Canonical.to_preprocessed_trace();
    reordered.columns.swap(0, 1);
    let error = cache
        .compile_or_bind(ShapeCompileRequest {
            preprocessed_trace: &reordered,
            ..sn2_request(&fixture.claim, fixture)
        })
        .err()
        .expect("reordered preprocessed columns must fail closed");
    assert!(matches!(
        error,
        ShapeExecutableError::NonCanonicalPreprocessedGeometry {
            variant: PreProcessedTraceVariant::Canonical,
            supplied_columns: 161,
            expected_columns: 161,
            first_mismatch: 0,
        }
    ));
    assert_eq!(cache.telemetry(), ShapeExecutableCacheTelemetry::default());
}

#[test]
fn public_data_felt_count_separates_same_log_after_witness_topology() {
    let fixture = sn2_fixture();
    let mut changed_claim = fixture.claim.clone();
    let before_felts = claim_public_data_felt_count(&fixture.claim).unwrap();
    for offset in 0..4 {
        changed_claim
            .public_data
            .public_memory
            .program
            .push((u32::MAX - offset, [0; 8]));
    }
    let after_felts = claim_public_data_felt_count(&changed_claim).unwrap();
    assert_eq!(changed_claim.log_sizes().0, fixture.claim.log_sizes().0);
    assert_eq!(after_felts, before_felts + 1);

    let original = TopologyKey::new(
        &fixture.claim,
        &fixture.proof_plan,
        &fixture.preprocessed_trace,
        PcsConfig::default(),
        false,
        None,
        ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
    )
    .unwrap();
    let changed = TopologyKey::new(
        &changed_claim,
        &fixture.proof_plan,
        &fixture.preprocessed_trace,
        PcsConfig::default(),
        false,
        None,
        ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
    )
    .unwrap();
    assert_ne!(
        original.claim_public_data_felts,
        changed.claim_public_data_felts
    );
    assert_ne!(original.digest(), changed.digest());
    assert_ne!(original, changed);

    // Capacity one turns a would-be incorrect hit into an observable,
    // fail-closed rejection. This is the after-witness route: no prepared
    // execution-table geometry participates in either key.
    let mut cache = ShapeExecutableCache::new(1).unwrap();
    cache
        .compile_or_bind(sn2_request(&fixture.claim, fixture))
        .unwrap();
    let error = cache
        .compile_or_bind(sn2_request(&changed_claim, fixture))
        .err()
        .expect("changed public-data felt count must not hit the executable");
    assert!(matches!(error, ShapeExecutableError::AtCapacity { .. }));
    assert_eq!(cache.telemetry().hits, 0);
    assert_eq!(cache.telemetry().misses, 2);
    assert_eq!(cache.telemetry().capacity_rejections, 1);
    assert_eq!(cache.telemetry().topology_key_constructions, 2);
}

#[test]
fn component_identity_separates_equal_flattened_tree_geometry() {
    let fixture = sn2_fixture();
    let mut base = fixture.claim.clone();
    base.range_check_8 = None;
    base.range_check_11 = None;
    base.range_check_12 = None;
    base.range_check_18 = None;
    base.range_check_20 = None;
    base.range_check_4_3 = None;
    base.range_check_4_4 = None;
    base.range_check_9_9 = None;
    base.range_check_7_2_5 = None;
    base.range_check_3_6_6_3 = None;
    base.range_check_4_4_4_4 = None;
    base.range_check_3_3_3_3_3 = None;
    base.verify_bitwise_xor_4 = None;

    let mut range_check_8 = base.clone();
    range_check_8.range_check_8 = Some(cairo_air::components::range_check_8::Claim {});
    let mut range_check_4_4 = base.clone();
    range_check_4_4.range_check_4_4 = Some(cairo_air::components::range_check_4_4::Claim {});
    let mut xor_4 = base;
    xor_4.verify_bitwise_xor_4 = Some(cairo_air::components::verify_bitwise_xor_4::Claim {});

    // These three components have identical tree geometry. The old admission
    // identity could not distinguish them when paired with a stale ProofPlan.
    let flattened_geometry = range_check_8.log_sizes().0;
    assert_eq!(range_check_4_4.log_sizes().0, flattened_geometry);
    assert_eq!(xor_4.log_sizes().0, flattened_geometry);

    let (range_8_bits, range_8_logs) = range_check_8.component_topology();
    let (range_4_4_bits, range_4_4_logs) = range_check_4_4.component_topology();
    let (xor_4_bits, xor_4_logs) = xor_4.component_topology();
    assert_eq!(range_8_logs, range_4_4_logs);
    assert_eq!(range_8_logs, xor_4_logs);
    assert_ne!(range_8_bits, range_4_4_bits);
    assert_ne!(range_8_bits, xor_4_bits);
    assert_ne!(range_4_4_bits, xor_4_bits);

    let key = |claim| {
        TopologyKey::new(
            claim,
            &fixture.proof_plan,
            &fixture.preprocessed_trace,
            PcsConfig::default(),
            false,
            None,
            ProtocolPlanPolicy::starknet_blake2s(0x1234, 2048),
        )
        .unwrap()
    };
    let range_8_key = key(&range_check_8);
    let range_4_4_key = key(&range_check_4_4);
    let xor_4_key = key(&xor_4);
    assert_ne!(range_8_key, range_4_4_key);
    assert_ne!(range_8_key, xor_4_key);
    assert_ne!(range_4_4_key, xor_4_key);
    assert_ne!(range_8_key.digest(), range_4_4_key.digest());
    assert_ne!(range_8_key.digest(), xor_4_key.digest());
}

#[test]
fn admission_rejects_forced_digest_and_layout_collisions() {
    let executable = sn2_executable();
    let exact = executable.workspace_admission().clone();
    assert!(exact.matches_plan(executable.arena()));
    assert_eq!(exact, exact.clone());

    let mut topology = (*exact.topology).clone();
    topology.include_all_preprocessed_columns = true;
    topology.digest = exact.topology.digest;
    let topology_collision = WorkspaceAdmission {
        key: exact.key,
        topology: Arc::new(topology),
        layout: Arc::clone(&exact.layout),
    };
    assert_eq!(topology_collision.workspace_key(), exact.workspace_key());
    assert_eq!(
        topology_collision.topology.digest(),
        exact.topology.digest()
    );
    assert_ne!(topology_collision, exact);

    let mut variant_only = (*exact.topology).clone();
    variant_only.preprocessed_trace_variant = PreProcessedTraceVariant::CanonicalWithoutPedersen;
    variant_only.digest = variant_only.compute_digest();
    assert_ne!(variant_only.digest(), exact.topology.digest());
    assert_ne!(&variant_only, exact.topology.as_ref());

    let mut layout = (*exact.layout).clone();
    layout.total_words += 1;
    let layout_collision = WorkspaceAdmission {
        key: exact.key,
        topology: Arc::clone(&exact.topology),
        layout: Arc::new(layout),
    };
    assert_ne!(layout_collision, exact);
    assert!(!layout_collision.matches_plan(executable.arena()));
}

#[test]
fn composition_plan_has_one_arena_owned_allocation() {
    let executable = sn2_executable();
    assert!(core::ptr::eq(
        executable.composition(),
        &executable.arena().composition().plan,
    ));
}

#[test]
fn binding_recipe_rejects_identity_span_word_and_extension_drift() {
    let fixture = sn2_fixture();
    let executable = sn2_executable();
    let interaction = schema_zero_interaction_claim_for_composition(&fixture.claim).unwrap();
    let compile = |plan: &CompositionPlan| {
        compile_cairo_composition_binding_plan(
            &fixture.claim,
            &CommonLookupElements::dummy(),
            &interaction,
            &fixture.preprocessed_trace.ids(),
            plan,
        )
    };

    let mut identity = executable.composition().clone();
    identity.components[0].component = "forged_component";
    assert!(matches!(
        compile(&identity),
        Err(CompositionPlanError::BindingTopologyDrift { .. })
    ));

    let mut span = executable.composition().clone();
    let component_with_span = span
        .components
        .iter()
        .position(|component| !component.trace_locations.is_empty())
        .unwrap();
    span.components[component_with_span].trace_locations[0].col_end += 1;
    assert!(matches!(
        compile(&span),
        Err(CompositionPlanError::BindingTopologyDrift { .. })
    ));

    let component_with_base = executable
        .composition()
        .components
        .iter()
        .position(|component| !component.base_param_values.is_empty())
        .unwrap();
    for extra in [false, true] {
        let mut words = executable.composition().clone();
        if extra {
            words.components[component_with_base]
                .base_param_values
                .push(BaseField::from_u32_unchecked(17));
        } else {
            words.components[component_with_base]
                .base_param_values
                .pop();
        }
        assert!(matches!(
            compile(&words),
            Err(CompositionPlanError::BindingTopologyDrift { .. })
        ));
    }

    let component_with_ext = executable
        .composition()
        .components
        .iter()
        .position(|component| !component.ext_param_values.is_empty())
        .unwrap();
    let mut extension = executable.composition().clone();
    extension.components[component_with_ext].ext_param_values[0] +=
        SecureField::from_u32_unchecked(1, 0, 0, 0);
    assert!(matches!(
        compile(&extension),
        Err(CompositionPlanError::BindingTopologyDrift { .. })
    ));
}
