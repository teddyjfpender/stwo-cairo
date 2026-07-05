//! Generic opt-in hook for the witness-JIT lane (witness-on-GPU W3 — the compiler that
//! aims to collapse the 57-component base-write tail into one recorded-and-codegen'd
//! per-row CUDA kernel per component).
//!
//! The heavy machinery lives in the stwo fork
//! (`stwo-backend-cuda::backend::jit_witness`): the witness bytecode ISA, the recording
//! builder, the reference interpreter, the CUDA codegen (reusing the constraint-JIT
//! NVRTC lane + its on-disk PTX cache), and the recorded programs for the three
//! generality-proving components. This module is the stwo-cairo-side seam: the gate a
//! component checks to opt in, and the kernel-warming entry the prover calls once per
//! process.
//!
//! # One-line registration
//!
//! A component opts into the lane by (a) having its decode recorded in
//! `jit_witness::programs` (done for `add_opcode`, `assert_eq_opcode`,
//! `jnz_opcode_taken`) and (b) branching in its `write_trace` on
//! [`jit_witness_lane_enabled`]. Because the generated writers under
//! `witness/components/*.rs` are marked `// This file was created by the AIR team.` and
//! must not be hand-edited, the branch is added the same way the memory / blake witness
//! lanes did it — via a backend wrapper the claim generator calls, NOT by editing the
//! generated file (see `memory_witness_backend.rs` / `blake_g_witness_backend.rs`). The
//! wrapper for a JIT component is a thin call into
//! [`stwo_backend_cuda::jit_witness`]; adding one is the "one-line registration" plus a
//! per-component backend shim.
//!
//! # Gates (default OFF, pod-gated — matches the round-8/9 witness-lane discipline)
//!
//! - Master kill switch: **`STWO_CUDA_WITNESS_JIT=1`** (anything else = host writer).
//! - Per-component bisect toggle: **`STWO_CUDA_WITNESS_JIT_<COMPONENT>=0`**.
//! - Differential: **`STWO_CUDA_WITNESS_VERIFY=1`** runs the host writer on cloned inputs and
//!   byte-compares every committed column + the finalized interaction columns and claimed sum (the
//!   same instrument the memory/blake lanes use). This is the promotion gate: 0 mismatches on a
//!   pod, then the Cairo e2e proof byte-equality, before the master default can flip ON.
//!
//! The lane is default OFF because — unlike the raw-logup finalize move (byte-identical
//! by construction) — a JIT-generated witness kernel is *new* arithmetic on the device;
//! it earns ON only after the pod differential + e2e byte-equality pass.

/// The components with a recorded witness-JIT program today (three distinct decode
/// shapes proving the framework generalizes). Kept in sync with
/// `stwo_backend_cuda::jit_witness` registry.
pub const WITNESS_JIT_COMPONENTS: &[&str] = &["add_opcode", "assert_eq_opcode", "jnz_opcode_taken"];

/// Whether `component`'s witness-JIT lane should run for this prove. Delegates to the
/// device-side gate (master switch + per-component toggle) and additionally requires the
/// CUDA kernels to be built — on a stub build (no nvcc) this is always `false`, so the
/// host writer runs unconditionally.
pub fn jit_witness_lane_enabled(component: &str) -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && stwo_backend_cuda::jit_witness::witness_jit_lane_enabled(component)
}

/// Whether the CUDA-vs-host differential should run alongside the device lane.
pub fn witness_verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

/// Warm the NVRTC/PTX cache for every enabled witness-JIT component once per process
/// (call from `prove_cairo` setup, alongside `prewarm_pedersen_tables`). Reuses the
/// constraint lane's `stwo_cuda_jit_precompile` — so a fresh process pays the compile
/// once and every subsequent prove hits the on-disk PTX cache. Returns the number of
/// kernels successfully warmed (0 on a stub build or with the lane OFF).
pub fn warm_witness_jit_kernels() -> usize {
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return 0;
    }
    WITNESS_JIT_COMPONENTS
        .iter()
        .filter(|component| stwo_backend_cuda::jit_witness::precompile_witness_kernel(component))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_is_off_on_stub_build() {
        // Without the CUDA kernels compiled (the local/CI default), every component's
        // lane is off and no kernels warm — the host writer path is untouched.
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            for component in WITNESS_JIT_COMPONENTS {
                assert!(!jit_witness_lane_enabled(component));
            }
            assert_eq!(warm_witness_jit_kernels(), 0);
        }
    }

    #[test]
    fn component_list_is_the_three_recorded_shapes() {
        assert_eq!(WITNESS_JIT_COMPONENTS.len(), 3);
        assert!(WITNESS_JIT_COMPONENTS.contains(&"add_opcode"));
        assert!(WITNESS_JIT_COMPONENTS.contains(&"assert_eq_opcode"));
        assert!(WITNESS_JIT_COMPONENTS.contains(&"jnz_opcode_taken"));
    }
}

/// The transformer-emitted full-width writer recordings, for dynamic registration in
/// the stwo witness-JIT registry (the automated-witness bridge). Kept here so the
/// `record_*` functions stay `pub(crate)` in their (tool-owned) component files.
pub fn emitted_recordings() -> Vec<(
    &'static str,
    crate::witness::witness_eval::recording::RecordingOutput,
)> {
    use crate::witness::components::{
        add_opcode, add_opcode_small, assert_eq_opcode, assert_eq_opcode_double_deref,
        assert_eq_opcode_imm, call_opcode_abs, call_opcode_rel_imm, jnz_opcode_non_taken,
        jnz_opcode_taken, jump_opcode_abs, jump_opcode_double_deref, jump_opcode_rel,
        jump_opcode_rel_imm, ret_opcode,
    };
    vec![
        ("add_opcode", add_opcode::record_add_opcode()),
        (
            "assert_eq_opcode",
            assert_eq_opcode::record_assert_eq_opcode(),
        ),
        (
            "jnz_opcode_taken",
            jnz_opcode_taken::record_jnz_opcode_taken(),
        ),
        (
            "add_opcode_small",
            add_opcode_small::record_add_opcode_small(),
        ),
        (
            "assert_eq_opcode_imm",
            assert_eq_opcode_imm::record_assert_eq_opcode_imm(),
        ),
        (
            "assert_eq_opcode_double_deref",
            assert_eq_opcode_double_deref::record_assert_eq_opcode_double_deref(),
        ),
        ("call_opcode_abs", call_opcode_abs::record_call_opcode_abs()),
        (
            "call_opcode_rel_imm",
            call_opcode_rel_imm::record_call_opcode_rel_imm(),
        ),
        (
            "jnz_opcode_non_taken",
            jnz_opcode_non_taken::record_jnz_opcode_non_taken(),
        ),
        ("jump_opcode_abs", jump_opcode_abs::record_jump_opcode_abs()),
        (
            "jump_opcode_double_deref",
            jump_opcode_double_deref::record_jump_opcode_double_deref(),
        ),
        ("jump_opcode_rel", jump_opcode_rel::record_jump_opcode_rel()),
        (
            "jump_opcode_rel_imm",
            jump_opcode_rel_imm::record_jump_opcode_rel_imm(),
        ),
        ("ret_opcode", ret_opcode::record_ret_opcode()),
    ]
}

/// §6a DEVICE-INTERACTION DIFFERENTIAL (pod gate, `STWO_DEVICE_INTERACTION_SELFTEST=1`):
/// for each covered component, run the witness kernel to get the DEVICE-resident
/// lookup words, then build the interaction trace BOTH ways from the same words —
/// host (`interaction_gen_from_flat_lookup_words` → `write_interaction_trace` →
/// SIMD finalize) and device (`descriptors_for_fields` → `logup_pairs.cu` →
/// device finalize) — and byte-compare every column and the claimed sum. Green
/// means the device-interaction prove branch is a pure wiring change.
pub fn run_device_interaction_selftest(input: &stwo_cairo_adapter::ProverInput) -> bool {
    use stwo::prover::backend::simd::m31::N_LANES;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo_backend_cuda::logup_pairs::{descriptors_for_fields, device_interaction_from_flats};
    use stwo_constraint_framework::LogupFinalizeBackend;

    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("DEVICE_INTERACTION_SELFTEST: CUDA kernels not built.");
        return false;
    }
    let mem = &input.memory;
    let addr_ids: Vec<u32> = mem.address_to_id.iter().map(|e| e.0).collect();
    let tables = stwo_backend_cuda::exec_tables::exec_tables_cached(
        mem.address_to_id.as_ptr() as usize,
        || {
            stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(
                &addr_ids,
                &mem.f252_values,
                &mem.small_values,
            )
        },
    );
    let mut channel = stwo::core::channel::Blake2sChannel::default();
    let elements = cairo_air::relations::CommonLookupElements::draw(&mut channel);
    let st = &input.state_transitions.casm_states_by_opcode;

    let mut all_ok = true;
    macro_rules! leg {
        ($module:ident, $label:literal, $record:path, $states:expr) => {{
            use crate::witness::components::$module;
            let states: &[stwo_cairo_common::prover_types::cpu::CasmState] = $states;
            if states.is_empty() {
                eprintln!("DEVICE_INTERACTION {}: SKIPPED (no states)", $label);
            } else {
                let t0 = std::time::Instant::now();
                let recording = $record();
                stwo_backend_cuda::jit_witness::register_recorded_program(
                    $label,
                    recording.program,
                );
                let n_real = states.len();
                let n_padded = std::cmp::max(n_real.next_power_of_two(), N_LANES);
                let first = states[0];
                let mut samples: Vec<(u32, u32, u32)> =
                    states.iter().map(|s| (s.pc.0, s.ap.0, s.fp.0)).collect();
                samples.resize(n_padded, (first.pc.0, first.ap.0, first.fp.0));
                let log_size = n_padded.ilog2();

                match stwo_backend_cuda::exec_tables::launch_recorded_witness_for_prove(
                    $label, &samples, n_real, tables, true,
                ) {
                    None => {
                        eprintln!("DEVICE_INTERACTION {}: FAIL (launch unavailable)", $label);
                        all_ok = false;
                    }
                    Some((_cols, lookup_dev, lookup_flat, _sub_dev, _sub)) => {
                        // HOST reference: accessors -> write_interaction_trace -> SIMD finalize.
                        let igen = $module::interaction_gen_from_flat_lookup_words(
                            log_size,
                            &lookup_flat,
                            n_padded,
                        );
                        let (raw, _claim_fn) = igen.write_interaction_trace(&elements);
                        let (host_cols, host_sum) =
                            <SimdBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);

                        // DEVICE: descriptors + pair kernel + device finalize.
                        let fields = $module::JIT_LOOKUP_FIELDS;
                        let n_fields = fields.len();
                        let tuple_fields: Vec<(&str, usize)> =
                            fields[..n_fields - 2].to_vec();
                        let m0 = (fields[..n_fields - 2].iter().map(|f| f.1).sum::<usize>()) as u32;
                        let m1 = m0 + 1;
                        let descs = descriptors_for_fields(&tuple_fields, m0, m1);
                        let max_w = tuple_fields.iter().map(|f| f.1).max().unwrap();
                        let alphas = &elements.alpha_powers()[..max_w];
                        let dev = device_interaction_from_flats(
                            lookup_dev.device_ptr,
                            n_padded,
                            n_padded, // opcode descs carry no ENABLER mult source
                            &descs,
                            alphas,
                            elements.z(),
                        );
                        match dev {
                            None => {
                                eprintln!("DEVICE_INTERACTION {}: FAIL (device path)", $label);
                                all_ok = false;
                            }
                            Some((dev_cols, dev_sum)) => {
                                let mut mismatch = 0usize;
                                if dev_sum != host_sum {
                                    eprintln!(
                                        "DEVICE_INTERACTION {}: claimed_sum {:?} vs {:?}",
                                        $label, dev_sum, host_sum
                                    );
                                    mismatch += 1;
                                }
                                if dev_cols.len() != host_cols.len() {
                                    eprintln!(
                                        "DEVICE_INTERACTION {}: col count {} vs {}",
                                        $label,
                                        dev_cols.len(),
                                        host_cols.len()
                                    );
                                    mismatch += 1;
                                } else {
                                    for (ci, (d, h)) in
                                        dev_cols.iter().zip(host_cols.iter()).enumerate()
                                    {
                                        let dv = d.values.to_cpu();
                                        let hv = h.values.to_cpu();
                                        for r in 0..n_padded {
                                            if dv[r] != hv[r] {
                                                if mismatch == 0 {
                                                    eprintln!(
                                                        "DEVICE_INTERACTION {}: col {ci} row                                                          {r}: device {} host {}",
                                                        $label, dv[r].0, hv[r].0
                                                    );
                                                }
                                                mismatch += 1;
                                            }
                                        }
                                    }
                                }
                                let checked = host_cols.len() * n_padded + 1;
                                eprintln!(
                                    "LEG di:{} : checked={checked} mismatch={mismatch}                                      [{:.3}s] {} rows x {} logup cols",
                                    $label,
                                    t0.elapsed().as_secs_f64(),
                                    n_padded,
                                    host_cols.len() / 4,
                                );
                                all_ok &= mismatch == 0;
                            }
                        }
                    }
                }
            }
        }};
    }

    eprintln!("=== DEVICE_INTERACTION_SELFTEST ===");
    leg!(
        add_opcode,
        "add_opcode",
        crate::witness::components::add_opcode::record_add_opcode,
        &st.add_opcode
    );
    leg!(
        assert_eq_opcode,
        "assert_eq_opcode",
        crate::witness::components::assert_eq_opcode::record_assert_eq_opcode,
        &st.assert_eq_opcode
    );
    leg!(
        ret_opcode,
        "ret_opcode",
        crate::witness::components::ret_opcode::record_ret_opcode,
        &st.ret_opcode
    );
    eprintln!(
        "=== DEVICE_INTERACTION {} ===",
        if all_ok { "PASS" } else { "FAIL" }
    );
    all_ok
}
