//! Source/JIT native parity for a bounded exact 18-wave Composition plan.
//!
//! This exercises the production descriptor and launch topology without closing
//! strict AOT admission. It grants neither timing nor current-pack promotion
//! credit; the source-free SN pack remains a separate hardware gate.

#![cfg(stwo_cuda_link)]

#[path = "support/prepared_composition_wave_fixture.rs"]
mod fixture_support;

use fixture_support::*;
use stwo::core::fields::qm31::SecureField;
use stwo_cairo_gpu_prover::prepared_composition::composition_workspace_requirements_with_retention_for_test;
use stwo_cairo_gpu_prover::{CompositionLaunchMode, PreparedCompositionGraph};

#[test]
fn exact_18_wave_source_jit_matches_serial_cpu_capture_and_canaries() {
    let fixture = fixture();
    let retention = retention(&fixture.logs);
    let serial_requirements = composition_workspace_requirements_with_retention_for_test(
        &fixture.plan,
        &fixture.trace,
        CompositionLaunchMode::Serial,
        Some(&retention),
    )
    .unwrap();
    let wave_requirements = composition_workspace_requirements_with_retention_for_test(
        &fixture.plan,
        &fixture.trace,
        CompositionLaunchMode::Wave,
        Some(&retention),
    )
    .unwrap();
    assert_eq!(wave_requirements.waves.len(), WAVE_COUNT);
    assert!(wave_requirements
        .waves
        .iter()
        .any(|wave| wave.parts.len() > 1));
    for (bound, manifest) in wave_requirements
        .waves
        .iter()
        .zip(&fixture.plan.wave_kernels)
    {
        assert_eq!(bound.evaluation_log_size, manifest.evaluation_log_size);
        assert_eq!(
            bound
                .parts
                .iter()
                .map(|part| part.identity)
                .collect::<Vec<_>>(),
            manifest.parts
        );
    }
    assert!(wave_requirements
        .components
        .iter()
        .all(|component| component.fallback_count == 0));
    assert_eq!(
        serial_requirements.accumulator_words,
        wave_requirements.accumulator_words
    );

    let slots = workspace_slots();
    let twiddles = twiddles();
    let eager_random = SecureField::from_u32_unchecked(2, 3, 5, 7);
    let serial_ready = ready_arena(
        &serial_requirements,
        &slots,
        &fixture.logs,
        &twiddles,
        11,
        eager_random,
        EAGER_Z,
        EAGER_ALPHA,
    );
    let wave_ready = ready_arena(
        &wave_requirements,
        &slots,
        &fixture.logs,
        &twiddles,
        11,
        eager_random,
        EAGER_Z,
        EAGER_ALPHA,
    );
    install_jit_wave_sources(&fixture.plan);
    let serial_inputs = inputs(
        &serial_ready.arena,
        &serial_requirements,
        fixture.plan.components.len(),
    );
    let wave_inputs = inputs(
        &wave_ready.arena,
        &wave_requirements,
        fixture.plan.components.len(),
    );
    let serial = PreparedCompositionGraph::prepare_with_mode_and_retention_jit_for_test(
        &serial_ready.arena,
        &fixture.plan,
        &fixture.trace,
        &serial_inputs,
        &slots,
        CompositionLaunchMode::Serial,
        Some(&retention),
        &serial_ready.direct,
    )
    .unwrap();
    let wave = PreparedCompositionGraph::prepare_with_mode_and_retention_jit_for_test(
        &wave_ready.arena,
        &fixture.plan,
        &fixture.trace,
        &wave_inputs,
        &slots,
        CompositionLaunchMode::Wave,
        Some(&retention),
        &wave_ready.direct,
    )
    .unwrap();

    let expected_eager = expected(
        &fixture,
        &serial_ready.data,
        eager_random,
        EAGER_Z,
        EAGER_ALPHA,
    );
    serial.launch().unwrap();
    let serial_eager = outputs(&serial_ready.arena, &serial);
    wave.launch().unwrap();
    let wave_eager = outputs(&wave_ready.arena, &wave);
    assert_eq!(serial_eager, expected_eager);
    assert_eq!(wave_eager, serial_eager);
    assert_guards(
        &serial_ready.arena,
        &serial_requirements,
        &slots,
        &serial_ready.data,
        &fixture.logs,
        eager_random,
        EAGER_Z,
        EAGER_ALPHA,
    );
    assert_guards(
        &wave_ready.arena,
        &wave_requirements,
        &slots,
        &wave_ready.data,
        &fixture.logs,
        eager_random,
        EAGER_Z,
        EAGER_ALPHA,
    );

    let serial_capture = serial_ready.arena.context().capture().unwrap();
    serial.launch().unwrap();
    let serial_graph = serial_capture.finish().unwrap();
    let wave_capture = wave_ready.arena.context().capture().unwrap();
    wave.launch().unwrap();
    let wave_graph = wave_capture.finish().unwrap();
    let replay_random = SecureField::from_u32_unchecked(11, 13, 17, 19);
    let serial_replay_data = refresh(
        &serial_ready.arena,
        &fixture.logs,
        97,
        replay_random,
        REPLAY_Z,
        REPLAY_ALPHA,
    );
    upload_data(&serial_ready.arena, &serial_replay_data, &fixture.logs);
    let wave_replay_data = refresh(
        &wave_ready.arena,
        &fixture.logs,
        97,
        replay_random,
        REPLAY_Z,
        REPLAY_ALPHA,
    );
    upload_data(&wave_ready.arena, &wave_replay_data, &fixture.logs);
    let expected_replay = expected(
        &fixture,
        &serial_replay_data,
        replay_random,
        REPLAY_Z,
        REPLAY_ALPHA,
    );
    serial_graph.launch(serial_ready.arena.context()).unwrap();
    let serial_replay = outputs(&serial_ready.arena, &serial);
    wave_graph.launch(wave_ready.arena.context()).unwrap();
    let wave_replay = outputs(&wave_ready.arena, &wave);
    assert_eq!(serial_replay, expected_replay);
    assert_eq!(wave_replay, serial_replay);
    assert_ne!(wave_replay, wave_eager, "captured graph must reread inputs");
    assert_guards(
        &serial_ready.arena,
        &serial_requirements,
        &slots,
        &serial_replay_data,
        &fixture.logs,
        replay_random,
        REPLAY_Z,
        REPLAY_ALPHA,
    );
    assert_guards(
        &wave_ready.arena,
        &wave_requirements,
        &slots,
        &wave_replay_data,
        &fixture.logs,
        replay_random,
        REPLAY_Z,
        REPLAY_ALPHA,
    );
}
