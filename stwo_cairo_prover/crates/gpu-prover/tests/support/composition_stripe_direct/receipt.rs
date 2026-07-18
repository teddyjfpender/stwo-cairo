use std::fs::OpenOptions;
use std::io::Write;

use serde_json::{json, Value};

use super::*;

pub(crate) fn optional_captured_abba<WE, SE, WC, SC>(
    ready: &Ready,
    mut wave_eager: WE,
    mut stripes_eager: SE,
    mut wave_captured: WC,
    mut stripes_captured: SC,
) -> Option<Value>
where
    WE: FnMut(),
    SE: FnMut(),
    WC: FnMut(),
    SC: FnMut(),
{
    if std::env::var_os("STWO_COMPOSITION_STRIPE_PERF").is_none() {
        return None;
    }
    let eager = timed_abba(&ready.arena, &mut wave_eager, &mut stripes_eager);
    let captured = timed_abba(&ready.arena, &mut wave_captured, &mut stripes_captured);
    Some(json!({
        "order": "ABBA",
        "same_exec_context": true,
        "eager": eager,
        "captured": captured,
        "promotion_credit": false,
    }))
}

fn timed_abba<W: FnMut(), S: FnMut()>(arena: &DeviceArena, wave: &mut W, stripes: &mut S) -> Value {
    for _ in 0..2 {
        wave();
        stripes();
    }
    arena.context().sync().unwrap();
    let capacity = arena.context().begin_timing().unwrap();
    let cycles = (capacity / 4).min(3);
    assert_ne!(cycles, 0, "CUDA timing capacity must admit one ABBA cycle");
    for _ in 0..cycles {
        wave();
        arena.context().mark_timing().unwrap();
        stripes();
        arena.context().mark_timing().unwrap();
        stripes();
        arena.context().mark_timing().unwrap();
        wave();
        arena.context().mark_timing().unwrap();
    }
    arena.context().sync().unwrap();
    let elapsed = arena.context().elapsed_timing_ms(4 * cycles).unwrap();
    let mut wave_ms = Vec::with_capacity(2 * cycles);
    let mut stripe_ms = Vec::with_capacity(2 * cycles);
    for samples in elapsed.chunks_exact(4) {
        wave_ms.extend([samples[0], samples[3]]);
        stripe_ms.extend([samples[1], samples[2]]);
    }
    wave_ms.sort_by(f32::total_cmp);
    stripe_ms.sort_by(f32::total_cmp);
    let wave_median = wave_ms[wave_ms.len() / 2];
    let stripe_median = stripe_ms[stripe_ms.len() / 2];
    assert!(wave_median.is_finite() && wave_median > 0.0);
    assert!(stripe_median.is_finite() && stripe_median > 0.0);
    json!({
        "samples_per_arm": wave_ms.len(),
        "wave_source_jit_median_ms": wave_median,
        "installed_stripes_median_ms": stripe_median,
        "observed_candidate_over_wave": f64::from(stripe_median) / f64::from(wave_median),
    })
}

pub(crate) fn publish_receipt(
    fixture: &Fixture,
    stripes: &PreparedCompositionGraph<'_>,
    eager_digest: [u8; 32],
    replay_digest: [u8; 32],
    timing: Option<Value>,
) {
    let boundary = stripes
        .resource_bounded_stripe_boundary_receipt_for_test()
        .unwrap();
    assert_ne!(eager_digest, replay_digest);
    let installed = stripes.resource_bounded_stripe_receipts_for_test();
    assert_eq!(installed.len(), boundary.stripe_count);
    assert!(installed.iter().all(|receipt| {
        receipt.source_identity != [0; 32]
            && receipt.cubin_identity != [0; 32]
            && receipt.target_sm != 0
            && receipt
                .launch
                .grid()
                .iter()
                .all(|&dimension| dimension != 0)
            && receipt
                .launch
                .block()
                .iter()
                .all(|&dimension| dimension != 0)
            && receipt.resources.max_threads_per_block != 0
            && receipt.resources.registers_per_thread != 0
            && receipt.resources.binary_version == receipt.target_sm
    }));
    let runtime_stats = aot::runtime_stats();
    assert_eq!(
        runtime_stats.strict_rejections, 0,
        "diagnostic eager/captured ordering must never enter strict-miss"
    );
    let resources = installed
        .into_iter()
        .map(|receipt| {
            json!({
                "component": receipt.component,
                "kernel": receipt.kernel,
                "cache_key": format!("{:#018x}", receipt.cache_key),
                "semantic_hash": format!("{:#018x}", receipt.semantic_hash),
                "target_sm": receipt.target_sm,
                "source_identity": hex::encode(receipt.source_identity),
                "cubin_identity": hex::encode(receipt.cubin_identity),
                "grid": receipt.launch.grid(),
                "block": receipt.launch.block(),
                "dynamic_shared_bytes": receipt.launch.dynamic_shared_bytes(),
                "registers_per_thread": receipt.resources.registers_per_thread,
                "max_threads_per_block": receipt.resources.max_threads_per_block,
                "binary_version": receipt.resources.binary_version,
                "ptx_version": receipt.resources.ptx_version,
                "local_bytes": receipt.resources.local_bytes,
                "static_shared_bytes": receipt.resources.static_shared_bytes,
            })
        })
        .collect::<Vec<_>>();
    let installed_resources_present = !resources.is_empty();
    let receipt = json!({
        "schema": "stwo.composition.stripe-direct-diagnostic.v1",
        "benchmark_class": "representative_direct_stripe_proxy",
        "passed": true,
        "correctness_checks": {
            "multiple_evaluation_domains": EVALUATION_LOGS,
            "direct_split_output": true,
            "shared_source_inputs": true,
            "single_process_device_arena_context_main_stream": true,
            "runtime_strict_rejections": runtime_stats.strict_rejections,
            "all_retained_bytes_equal_eager": true,
            "all_retained_bytes_equal_mutated_capture_replay": true,
            "mutated_replay_digest_changed": true,
            "installed_identity_and_resources_present": installed_resources_present,
        },
        "diagnostic_only": true,
        "promotion_credit": false,
        "real_sn2_coverage": false,
        "hardware_integration_gate":
            "real SN2/153 exact plan and installed same-pack Wave baseline",
        "target_useful_mhz": 5.0,
        "evaluation_logs": EVALUATION_LOGS,
        "component_count": fixture.plan.components.len(),
        "wave_count": fixture.plan.wave_kernels.len(),
        "stripe_count": boundary.stripe_count,
        "serial_component_count": boundary.serial_component_count,
        "launch_mode": format!("{:?}", boundary.launch_mode),
        "output_mode": format!("{:?}", boundary.output_mode),
        "direct_retention_plan_key":
            format!("{:#018x}", boundary.direct_retention_plan_key.unwrap()),
        "direct_column_count": fixture.retention.direct_column_count,
        "direct_bytes": fixture.retention.direct_bytes,
        "retained_columns": COMPOSITION_RETAINED_COLUMNS,
        "retained_words_checked_per_run":
            COMPOSITION_RETAINED_COLUMNS * (1usize << MAX_EVALUATION_LOG),
        "eager_retained_blake3": hex::encode(eager_digest),
        "mutated_replay_retained_blake3": hex::encode(replay_digest),
        "installed_functions": resources,
        "timing": timing,
    });
    let line = serde_json::to_string(&receipt).unwrap();
    println!("STWO_COMPOSITION_STRIPE_DIRECT_RECEIPT_JSON={line}");
    if let Some(path) = std::env::var_os("STWO_COMPOSITION_STRIPE_RECEIPT") {
        let path = std::path::PathBuf::from(path);
        let mut temporary = path.as_os_str().to_os_string();
        temporary.push(format!(".tmp-{}", std::process::id()));
        let temporary = std::path::PathBuf::from(temporary);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .unwrap();
        writeln!(file, "{line}").unwrap();
        file.sync_all().unwrap();
        std::fs::hard_link(&temporary, &path).unwrap();
        std::fs::remove_file(temporary).unwrap();
    }
}
