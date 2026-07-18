//! Diagnostic wave-versus-stripe parity at the production direct-split boundary.
//!
//! This is a representative three-domain log-24 fixture, not an SN proof shape.
//! The wave arm is source-JIT and therefore earns no AOT-pack or performance
//! promotion credit. The stripe arm must use exact installed ordinary functions.

#![cfg(stwo_cuda_link)]

#[path = "support/composition_stripe_direct.rs"]
mod support;

use stwo_cairo_gpu_prover::composition_plan::CompositionProofBindings;
use stwo_cairo_gpu_prover::PreparedCompositionGraph;
use support::*;

#[test]
fn multidomain_direct_split_wave_and_installed_stripes_match_eager_and_replay() {
    let fixture = fixture();
    let twiddles = twiddles();
    let ready = ready(&fixture, &twiddles, 11);
    let proof_bindings = CompositionProofBindings::from_plan(&fixture.plan);

    // Resolve the runtime-origin Wave functions first; the diagnostic stripe
    // arm installs exact AOT without closing process-wide strict admission.
    install_wave_sources(&fixture.plan);
    let wave = PreparedCompositionGraph::prepare_wave_direct_retained_jit_for_test(
        &ready.arena,
        &fixture.plan,
        &proof_bindings,
        &fixture.trace,
        &ready.inputs,
        &fixture.wave_slots,
        &fixture.retention,
        &ready.direct,
        ready.wave_split,
    )
    .unwrap();
    wave.launch().unwrap();
    let wave_capture = ready.arena.context().capture().unwrap();
    wave.launch().unwrap();
    let wave_graph = wave_capture.finish().unwrap();

    // This diagnostic constructor retains every exact installed ordinary
    // function receipt without invalidating the runtime-origin Wave baseline.
    let stripes = PreparedCompositionGraph::prepare_resource_bounded_stripes_for_test(
        &ready.arena,
        &fixture.plan,
        &proof_bindings,
        &fixture.trace,
        &ready.inputs,
        &fixture.stripe_slots,
        &fixture.retention,
        &ready.direct,
        ready.stripe_split,
    )
    .unwrap();
    stripes
        .launch_resource_bounded_stripes_eager_for_test()
        .unwrap();
    let stripe_capture = ready.arena.context().capture().unwrap();
    stripes
        .launch_resource_bounded_stripes_capture_for_test()
        .unwrap();
    let stripe_graph = stripe_capture.finish().unwrap();

    let eager_digest = assert_retained_equal(&ready);

    refresh(&ready, 97);
    wave_graph.launch(ready.arena.context()).unwrap();
    stripe_graph.launch(ready.arena.context()).unwrap();
    let replay_digest = assert_retained_equal(&ready);
    assert_ne!(
        eager_digest, replay_digest,
        "captured graphs must observe mutated direct inputs"
    );

    publish_receipt(
        &fixture,
        &stripes,
        eager_digest,
        replay_digest,
        optional_captured_abba(
            &ready,
            || wave.launch().unwrap(),
            || {
                stripes
                    .launch_resource_bounded_stripes_eager_for_test()
                    .unwrap()
            },
            || wave_graph.launch(ready.arena.context()).unwrap(),
            || stripe_graph.launch(ready.arena.context()).unwrap(),
        ),
    );
}
