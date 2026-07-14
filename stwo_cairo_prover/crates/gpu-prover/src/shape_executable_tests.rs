use std::sync::OnceLock;

use cairo_vm::types::layout_name::LayoutName;
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
