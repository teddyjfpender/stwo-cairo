use std::collections::BTreeMap;

use super::{
    admission_verdict, arena_compatibility_aliases_match, budget_bytes_of, missing_aot_kernels,
    parse_resident_backend, parse_vram_budget_gb, preflight_fit_alias_matches, runtime_policy_json,
    verdict, AotKernelOccurrence, AotManifestKernel, GIB, WORD_BYTES,
};

#[test]
fn budget_bytes_is_gib_scaled() {
    assert_eq!(budget_bytes_of(1.0), 1024 * 1024 * 1024);
    assert_eq!(budget_bytes_of(79.0), 79 * 1024 * 1024 * 1024);
    assert_eq!(budget_bytes_of(0.5), 512 * 1024 * 1024);
}

#[test]
fn vram_budget_requires_a_positive_finite_number() {
    assert_eq!(parse_vram_budget_gb(None), Ok(79.0));
    assert_eq!(parse_vram_budget_gb(Some("76")), Ok(76.0));
    assert_eq!(budget_bytes_of(76.0), 76 * 1024 * 1024 * 1024);
    for invalid in ["0", "-1", "NaN", "inf"] {
        assert!(
            parse_vram_budget_gb(Some(invalid)).is_err(),
            "accepted invalid budget {invalid}"
        );
    }
}

#[test]
fn resident_backend_selector_is_explicit_and_fail_closed() {
    use stwo_cairo_gpu_prover::arena_plan::ResidentBackend;

    assert_eq!(
        parse_resident_backend(["arena_preflight"]),
        Ok(ResidentBackend::LegacyResident)
    );
    assert_eq!(
        parse_resident_backend(["arena_preflight", "--resident-backend", "replacement-v1"]),
        Ok(ResidentBackend::ReplacementV1)
    );
    assert_eq!(
        parse_resident_backend(["--resident-backend", "legacy-resident"]),
        Ok(ResidentBackend::LegacyResident)
    );
    for invalid in [
        vec!["--resident-backend"],
        vec!["--resident-backend", "--fixture", "test"],
        vec!["--resident-backend", "unknown"],
        vec!["--resident-backend=replacement-v1"],
        vec![
            "--resident-backend",
            "legacy-resident",
            "--resident-backend",
            "replacement-v1",
        ],
    ] {
        assert!(
            parse_resident_backend(invalid).is_err(),
            "accepted invalid resident backend form"
        );
    }
}

#[test]
fn replacement_policy_json_reports_the_exact_planned_tuple() {
    use stwo_backend_cuda::RelationLaunchMode;
    use stwo_cairo_gpu_prover::protocol_plan::ProtocolPlanPolicy;

    let value = runtime_policy_json(
        ProtocolPlanPolicy::replacement_v1(0x1234, 2048),
        RelationLaunchMode::Fused,
    );
    assert_eq!(value["resident_backend"], "replacement-v1");
    assert_eq!(value["quotient_numerator_schedule"], "hybrid-single-write");
    assert_eq!(value["kernel_manifest_hash"], "0000000000001234");
    assert_eq!(value["composition_max_kernel_instrs"], 2048);
    assert_eq!(
        value["retained_lde_budget_bytes"],
        64 * 1024 * 1024 * 1024_u64
    );
    assert_eq!(value["commit_mode"], "domain-progressive");
    assert_eq!(value["direct_composition_retention_mode"], "exact-native");
    assert_eq!(
        value["quotient_numerator_source_policy"],
        "reuse-retained-evaluations"
    );
    assert_eq!(value["interpolation_mode"], "stage-fused-out-of-place");
    assert_eq!(value["relation_launch_mode"], "fused");
}

#[test]
fn word_bytes_and_gib_are_the_arena_units() {
    assert_eq!(WORD_BYTES, 4);
    assert_eq!(GIB, (1u64 << 30) as f64);
}

#[test]
fn verdict_requires_every_gate() {
    assert!(verdict(true, 0, 0, 100, 100, true));
    assert!(!verdict(false, 0, 0, 100, 100, true));
    assert!(!verdict(true, 1, 0, 100, 100, true));
    assert!(!verdict(true, 0, 1, 100, 100, true));
    assert!(!verdict(true, 0, 0, 101, 100, true));
    assert!(!verdict(true, 0, 0, 100, 100, false));
}

#[test]
fn admission_requires_a_complete_passing_physical_ledger() {
    assert!(admission_verdict(true, true, true));
    assert!(!admission_verdict(false, true, true));
    assert!(!admission_verdict(true, false, true));
    assert!(!admission_verdict(true, true, false));
}

#[test]
fn deprecated_arena_aliases_must_equal_current_fields() {
    let mut arena = serde_json::json!({
        "allocation_words": 7,
        "allocation_bytes": 28,
        "allocation_gib": 28.0 / GIB,
        "total_words": 7,
        "total_bytes": 28,
        "total_gib": 28.0 / GIB,
        "logical_buffer_count": 3,
        "logical_buffers": 3,
    });
    assert!(arena_compatibility_aliases_match(&arena));
    arena["total_bytes"] = serde_json::json!(27);
    assert!(!arena_compatibility_aliases_match(&arena));
    arena["total_bytes"] = serde_json::json!(28);
    arena["logical_buffers"] = serde_json::json!(2);
    assert!(!arena_compatibility_aliases_match(&arena));
}

#[test]
fn deprecated_arena_vram_fit_alias_must_equal_allocation_fit() {
    let mut record = serde_json::json!({
        "arena_allocation_vram_fit": true,
        "arena_vram_fit": true,
    });
    assert!(preflight_fit_alias_matches(&record));
    record["arena_vram_fit"] = serde_json::json!(false);
    assert!(!preflight_fit_alias_matches(&record));
}

#[test]
fn aot_coverage_requires_exact_launch_identity() {
    let kernel = AotKernelOccurrence {
        kind: "constraint",
        component: "component".to_owned(),
        instance: 0,
        kernel: 0,
        kernel_name: "kernel".to_owned(),
        semantic_hash: 7,
        cache_key: 11,
    };
    let mut manifest = BTreeMap::from([(
        11,
        AotManifestKernel {
            kind: "constraint".to_owned(),
            kernel_name: "kernel".to_owned(),
            semantic_hash: 7,
        },
    )]);
    assert!(missing_aot_kernels(&[kernel.clone()], &manifest).is_empty());
    manifest.get_mut(&11).unwrap().semantic_hash ^= 1;
    assert_eq!(
        missing_aot_kernels(&[kernel.clone()], &manifest)[0].1,
        "identity_mismatch"
    );
    manifest.clear();
    assert_eq!(
        missing_aot_kernels(&[kernel], &manifest)[0].1,
        "missing_key"
    );
}
