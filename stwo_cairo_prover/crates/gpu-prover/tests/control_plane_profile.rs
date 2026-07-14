use std::time::Instant;

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::phases;
use stwo_cairo_gpu_prover::resident_session::{
    plan_resident_preflight, plan_resident_preflight_with_cache, ResidentPreflightError,
    ResidentSessionError,
};
use stwo_cairo_gpu_prover::shape_executable::{
    ShapeExecutableCache, ShapeExecutableError, ShapeExecutableMaterialization,
};

#[path = "common/base_param_variant.rs"]
mod base_param_variant;
use base_param_variant::swap_bitwise_and_ec_op_segments;

fn sn2_inputs() -> (ProverInput, ProverInput) {
    let first = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let second = swap_bitwise_and_ec_op_segments(first.clone());
    (first, second)
}

#[test]
fn same_shape_changed_statement_reuses_one_source_free_executable() {
    let (first, second) = sn2_inputs();
    let first = phases::ingest::run(first, PreProcessedTraceVariant::Canonical, None);
    let second = phases::ingest::run(second, PreProcessedTraceVariant::Canonical, None);
    assert_eq!(first.proof_plan.shape_key, second.proof_plan.shape_key);

    let mut cache = ShapeExecutableCache::new(1).unwrap();
    let cold = plan_resident_preflight_with_cache(
        &mut cache,
        &first.generator,
        &first.proof_plan,
        &first.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    assert_eq!(
        cold.shape_executable_materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert_eq!(cold.shape_executable_cache.misses, 1);
    assert_eq!(cold.shape_executable_cache.compilations, 1);
    assert_eq!(cold.shape_executable_cache.source_generation_passes, 1);
    assert_eq!(cold.shape_executable_cache.binding_recipe_compilations, 1);

    let warm = plan_resident_preflight_with_cache(
        &mut cache,
        &second.generator,
        &second.proof_plan,
        &second.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    assert_eq!(
        warm.shape_executable_materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert_eq!(
        warm.shape_executable_topology_digest,
        cold.shape_executable_topology_digest
    );
    assert_ne!(warm.composition_bindings, cold.composition_bindings);
    assert_eq!(
        warm.composition_bindings.component_count(),
        cold.composition_bindings.component_count()
    );
    assert_eq!(
        warm.composition_bindings.base_param_word_count(),
        cold.composition_bindings.base_param_word_count()
    );
    assert_eq!(warm.shape_executable_cache.hits, 1);
    assert_eq!(warm.shape_executable_cache.compilations, 1);
    assert_eq!(warm.shape_executable_cache.source_generation_passes, 1);
    assert_eq!(warm.shape_executable_cache.binding_recipe_compilations, 1);
    assert_eq!(
        warm.shape_executable_cache.source_generation_passes
            - cold.shape_executable_cache.source_generation_passes,
        0,
        "warm binding reran the source-generation planning pass"
    );

    let mut independent_cache = ShapeExecutableCache::new(1).unwrap();
    let independent_second = plan_resident_preflight_with_cache(
        &mut independent_cache,
        &second.generator,
        &second.proof_plan,
        &second.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    assert_eq!(
        independent_second.shape_executable_materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert_eq!(
        warm.composition_bindings,
        independent_second.composition_bindings
    );

    let drift = match plan_resident_preflight_with_cache(
        &mut cache,
        &second.generator,
        &second.proof_plan,
        &second.preprocessed_trace,
        PcsConfig::default(),
        true,
    ) {
        Err(error) => error,
        Ok(_) => panic!("topology drift reused a full one-entry cache"),
    };
    assert!(matches!(
        drift,
        ResidentPreflightError::Session(ResidentSessionError::ShapeExecutable(
            ShapeExecutableError::AtCapacity { capacity: 1, .. }
        ))
    ));
    assert_eq!(cache.telemetry().capacity_rejections, 1);
}

#[test]
#[ignore = "quantitative host control-plane profile"]
fn sn2_resident_planner_profile() {
    let (input, _) = sn2_inputs();
    let ingest = phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);

    let mut uncached_samples = Vec::new();
    let mut cold_compile_samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        let report = plan_resident_preflight(
            &ingest.generator,
            &ingest.proof_plan,
            &ingest.preprocessed_trace,
            PcsConfig::default(),
            false,
        )
        .unwrap();
        std::hint::black_box(report.arena.total_words());
        cold_compile_samples.push(report.shape_executable_control_plane_ns as f64 / 1_000_000.0);
        uncached_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let mut cache = ShapeExecutableCache::new(1).unwrap();
    let cold = plan_resident_preflight_with_cache(
        &mut cache,
        &ingest.generator,
        &ingest.proof_plan,
        &ingest.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap();
    assert_eq!(cold.shape_executable_cache.source_generation_passes, 1);
    assert_eq!(cold.shape_executable_cache.binding_recipe_compilations, 1);

    let mut warm_total_samples = Vec::new();
    let mut warm_select_and_bind_samples = Vec::new();
    for _ in 0..7 {
        let warm_start = Instant::now();
        let report = plan_resident_preflight_with_cache(
            &mut cache,
            &ingest.generator,
            &ingest.proof_plan,
            &ingest.preprocessed_trace,
            PcsConfig::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            report.shape_executable_materialization,
            ShapeExecutableMaterialization::Reused
        );
        assert_eq!(report.shape_executable_cache.source_generation_passes, 1);
        assert_eq!(report.shape_executable_cache.binding_recipe_compilations, 1);
        warm_select_and_bind_samples
            .push(report.shape_executable_control_plane_ns as f64 / 1_000_000.0);
        warm_total_samples.push(warm_start.elapsed().as_secs_f64() * 1_000.0);
    }
    let mut uncached_ordered = uncached_samples.clone();
    uncached_ordered.sort_by(f64::total_cmp);
    let mut cold_compile_ordered = cold_compile_samples.clone();
    cold_compile_ordered.sort_by(f64::total_cmp);
    let mut warm_total_ordered = warm_total_samples.clone();
    warm_total_ordered.sort_by(f64::total_cmp);
    let mut warm_select_and_bind_ordered = warm_select_and_bind_samples.clone();
    warm_select_and_bind_ordered.sort_by(f64::total_cmp);
    eprintln!("uncached_planner_ms={uncached_samples:?}");
    eprintln!(
        "uncached_planner_p50_ms={}",
        uncached_ordered[uncached_ordered.len() / 2]
    );
    eprintln!("cold_compile_ms={cold_compile_samples:?}");
    eprintln!(
        "cold_compile_p50_ms={}",
        cold_compile_ordered[cold_compile_ordered.len() / 2]
    );
    eprintln!("warm_total_ms={warm_total_samples:?}");
    eprintln!(
        "warm_total_p50_ms={}",
        warm_total_ordered[warm_total_ordered.len() / 2]
    );
    eprintln!("warm_cache_select_and_bind_ms={warm_select_and_bind_samples:?}");
    eprintln!(
        "warm_cache_select_and_bind_p50_ms={}",
        warm_select_and_bind_ordered[warm_select_and_bind_ordered.len() / 2]
    );
    eprintln!("shape_cache_telemetry={:?}", cache.telemetry());
}
