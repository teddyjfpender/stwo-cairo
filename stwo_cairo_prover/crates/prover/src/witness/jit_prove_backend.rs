//! Prove-path hook for JIT-witness components (the automated-witness lane's
//! production integration): a component whose transformer-emitted recording is
//! registered generates its base trace ON DEVICE via the witness-JIT kernel
//! (record → codegen → NVRTC → launch against the §2 device execution tables),
//! returning device-resident committed columns plus word-major host copies of the
//! flat lookup/sub-input words that rebuild the exact host `LookupData` /
//! `SubComponentInputs` flows.
//!
//! Structure: ONE backend trait ([`OpcodeJitBackend`]) generic over a
//! per-component spec ([`OpcodeLaneSpec`]). A new opcode component joins the lane
//! by (a) having its writer transformed + recording fenced by the differential
//! battery, (b) two accessor fns in its module (flat lookup words → its private
//! `LookupData`; flat sub words → its downstream feeds), (c) a ~25-line spec impl
//! here, (d) its spawn site routed through `lane_write_trace`. No new CUDA.
//!
//! The lane is DEFAULT OFF (`STWO_CUDA_WITNESS_JIT_PROVE=1` opts in, and
//! `STWO_CUDA_WITNESS_JIT_PROVE_<COMPONENT>=0` disables one component for A/B)
//! and falls back to the host writer on any unavailability — every fallback is
//! logged with its reason so a silently-idle GPU is visible in the bench logs.
//! Correctness gates: the local parity battery (interpreter flats → these
//! accessors vs the host writer, padding rows included), the truth-oracle
//! hardware selftest (columns + lookup + sub words vs real memory semantics), the
//! in-process shadow diff (`STWO_JIT_PROVE_SHADOW=1`), then prove-level
//! prefix-hash identity + verify + repeated-prove, lane ON vs OFF.
//!
//! Value-identity argument: the kernel executes the transformer-emitted recording
//! whose generic writer is byte-identical to the AIR-generated writer on real
//! fixtures; lookup/sub-input words are the same values the host writer computes,
//! re-packed into the identical structures; sub-input feeding uses the same
//! `add_inputs` entry points, in the same per-relation order, over the same FULL
//! padded extent (padding rows feed too — `mults_0 = 1` on every row).

use std::sync::Arc;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::m31::N_LANES;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::FromSimdColumns;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_common::prover_types::cpu::CasmState;

use crate::witness::components::{
    add_opcode, add_opcode_small, assert_eq_opcode, assert_eq_opcode_double_deref,
    assert_eq_opcode_imm, call_opcode_abs, call_opcode_rel_imm, jnz_opcode_non_taken,
    jnz_opcode_taken, jump_opcode_abs, jump_opcode_double_deref, jump_opcode_rel,
    jump_opcode_rel_imm, memory_address_to_id, memory_id_to_big, ret_opcode, verify_instruction,
};
use crate::witness::witness_eval::recording::RecordingOutput;

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;
type SimdEvals = Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>;

/// §6a master switch: at witness time the lane STASHES its device-resident lookup
/// buffer (skipping the flats D2H + host repack); at interaction time — after the
/// lookup elements are drawn, which is why it cannot happen earlier — the pair
/// kernel + device finalize replace the host `write_interaction_trace` wholesale.
pub(crate) fn device_interaction_enabled() -> bool {
    std::env::var("STWO_CUDA_DEVICE_INTERACTION").as_deref() == Ok("1")
}

type DeviceLookupStash = std::sync::Mutex<
    std::collections::HashMap<&'static str, (stwo_backend_cuda::BaseFieldVec, usize, usize)>,
>;

fn device_lookup_stash() -> &'static DeviceLookupStash {
    static STASH: std::sync::OnceLock<DeviceLookupStash> = std::sync::OnceLock::new();
    STASH.get_or_init(Default::default)
}

fn lane_enabled(label: &str) -> bool {
    if std::env::var("STWO_CUDA_WITNESS_JIT_PROVE").as_deref() != Ok("1") {
        return false;
    }
    let per = format!("STWO_CUDA_WITNESS_JIT_PROVE_{}", label.to_uppercase());
    std::env::var(per).as_deref() != Ok("0")
}

/// Per-component spec: everything the generic lane needs to run one opcode
/// component. All three downstream states are the CasmState-opcode family's
/// (memory_address_to_id, memory_id_to_big, verify_instruction).
pub trait OpcodeLaneSpec {
    const LABEL: &'static str;
    /// Committed base-trace columns; the recording must match EXACTLY.
    const N_TRACE: usize;
    const N_LOOKUP_WORDS: usize;
    const N_SUB_WORDS: usize;
    type Gen;
    type Claim;
    type IGen;

    fn inputs(gen: &Self::Gen) -> &[CasmState];
    /// `(name, width)` per `LookupData` field, declaration order (trailing two are
    /// `mults_0`/`mults_1`) — the §6a descriptor builder's input.
    fn lookup_fields() -> &'static [(&'static str, usize)];
    /// The emitted per-column descriptor FACTS (`JIT_LOGUP_DESCS`) — parsed from
    /// this component's generated `write_interaction_trace`; resolved via
    /// `logup_descs::resolve_logup_descs` and gate-proven by the host mirror.
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact];
    fn record() -> RecordingOutput;
    fn host_write(
        gen: Self::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
    ) -> (SimdEvals, Self::Claim, Self::IGen);
    fn claim(log_size: u32) -> Self::Claim;
    fn igen_from_flats(log_size: u32, words: &[u32], n_rows: usize) -> Self::IGen;
    fn feed_from_flats(
        words: &[u32],
        n_rows: usize,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
    );
    /// Debug instrument (`STWO_JIT_PROVE_SHADOW=1`): byte-diff lane outputs
    /// against the host writer with exact coordinates.
    #[allow(clippy::too_many_arguments)]
    fn shadow_compare(
        inputs: &[CasmState],
        device_cols: &[Vec<u32>],
        lookup_flat: &[u32],
        sub_flat: &[u32],
        n_padded: usize,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
    );
}

/// Everything the component-specific accessors need from a lane launch.
struct LaneOutput {
    log_size: u32,
    cols: Vec<
        <stwo_backend_cuda::CudaBackend as stwo::prover::backend::ColumnOps<BaseField>>::Column,
    >,
    /// Word-major flats: `words[w * column_length + row]`.
    lookup_flat: Vec<u32>,
    sub_flat: Vec<u32>,
    /// The SAME lookup words, still device-resident — the §6a device-interaction
    /// lane consumes them in place (no D2H) once its differential is green.
    #[allow(dead_code)]
    lookup_dev: stwo_backend_cuda::BaseFieldVec,
    column_length: usize,
}

/// Component-agnostic core of the CUDA JIT witness lane: validates the recording's
/// shape against the spec, pads the samples by the host writer's rule (replicate
/// the first input up to `max(n.next_power_of_two(), N_LANES)`), resolves the §2
/// device tables for this memory (one upload per memory, process-cached), launches,
/// and returns device columns + word-major host flats. `None` = fall back to the
/// host writer (reason logged).
fn cuda_jit_lane<C: OpcodeLaneSpec>(inputs: &[CasmState], mem: &Memory) -> Option<LaneOutput> {
    let t0 = std::time::Instant::now();
    let n_real = inputs.len();
    if n_real == 0 {
        return None;
    }

    let recording = C::record();
    let program = &recording.program;
    if program.n_cols as usize != C::N_TRACE
        || program.n_lookup_words as usize != C::N_LOOKUP_WORDS
        || program.n_sub_words as usize != C::N_SUB_WORDS
        || !recording.poisoned_cols.is_empty()
        || !recording.poisoned_lookup_words.is_empty()
        || !recording.poisoned_sub_words.is_empty()
    {
        eprintln!(
            "jit_prove[{}]: recording shape mismatch (cols {} vs {}, lookup {} vs {}, sub {} \
             vs {}, poison {}/{}/{}) — falling back",
            C::LABEL,
            program.n_cols,
            C::N_TRACE,
            program.n_lookup_words,
            C::N_LOOKUP_WORDS,
            program.n_sub_words,
            C::N_SUB_WORDS,
            recording.poisoned_cols.len(),
            recording.poisoned_lookup_words.len(),
            recording.poisoned_sub_words.len(),
        );
        return None;
    }
    stwo_backend_cuda::jit_witness::register_recorded_program(C::LABEL, recording.program);

    // Samples padded exactly like the host writer (`inputs.resize(size, first)`).
    let column_length = std::cmp::max(n_real.next_power_of_two(), N_LANES);
    let log_size = column_length.ilog2();
    let first = inputs[0];
    let mut samples: Vec<(u32, u32, u32)> =
        inputs.iter().map(|s| (s.pc.0, s.ap.0, s.fp.0)).collect();
    samples.resize(column_length, (first.pc.0, first.ap.0, first.fp.0));

    // The §2 device execution tables for THIS memory — the kernel deduces every
    // memory operand (pc decode AND dst/op0/op1 value chains) through them.
    let tables = stwo_backend_cuda::exec_tables::exec_tables_cached(
        mem.address_to_id.as_ptr() as usize,
        || {
            let addr_ids: Vec<u32> = mem.address_to_id.iter().map(|e| e.0).collect();
            stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(
                &addr_ids,
                &mem.f252_values,
                &mem.small_values,
            )
        },
    );

    let launched = stwo_backend_cuda::exec_tables::launch_recorded_witness_for_prove(
        C::LABEL,
        &samples,
        n_real,
        tables,
        !device_interaction_enabled(),
    );
    let Some((cols, lookup_dev, lookup_flat, _sub_dev, sub_flat)) = launched else {
        // launch_recorded_witness_for_prove logged the specific reason.
        return None;
    };
    if cols.len() != C::N_TRACE {
        eprintln!(
            "jit_prove[{}]: kernel returned {} columns, expected {} — falling back",
            C::LABEL,
            cols.len(),
            C::N_TRACE
        );
        return None;
    }
    eprintln!(
        "jit_prove[{}]: device witness {} real rows (log_size {}) in {:.1} ms",
        C::LABEL,
        n_real,
        log_size,
        t0.elapsed().as_secs_f64() * 1e3,
    );
    Some(LaneOutput {
        log_size,
        cols,
        lookup_flat,
        sub_flat,
        lookup_dev,
        column_length,
    })
}

/// Backend hook: opcode-component base-trace generation. Simd = the host writer
/// verbatim; Cuda = the JIT witness kernel when the lane is on (host fallback
/// otherwise).
pub trait OpcodeJitBackend: FromSimdColumns + Sized {
    /// True when this component's device lookup buffer is stashed (the host
    /// interaction write can be skipped entirely — `device_interaction` WILL
    /// produce the trace or panic; there is deliberately no silent fallback).
    fn device_interaction_pending<C: OpcodeLaneSpec>() -> bool {
        false
    }

    /// §6a interaction-time hook: when this component's device lookup buffer was
    /// stashed at witness time, build its interaction trace ON DEVICE (pair kernel
    /// + proven finalize) and return it with the claimed sum. `None` = use the
    /// host `write_interaction_trace` path.
    fn device_interaction<C: OpcodeLaneSpec>(
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let _ = elements;
        None
    }

    /// Builtin-lane variants of the §6a hooks (same stash, `BuiltinLaneSpec`-keyed).
    fn builtin_device_interaction_pending<C: BuiltinLaneSpec>() -> bool {
        false
    }
    fn builtin_device_interaction<C: BuiltinLaneSpec>(
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let _ = elements;
        None
    }

    fn lane_write_trace<C: OpcodeLaneSpec>(
        gen: C::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (Evals<Self>, C::Claim, C::IGen);
}

impl OpcodeJitBackend for SimdBackend {
    fn lane_write_trace<C: OpcodeLaneSpec>(
        gen: C::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
        _jit_memory: Option<&Arc<Memory>>,
    ) -> (Evals<Self>, C::Claim, C::IGen) {
        C::host_write(gen, addr_state, id_state, vi_state)
    }
}

impl OpcodeJitBackend for stwo_backend_cuda::CudaBackend {
    fn device_interaction_pending<C: OpcodeLaneSpec>() -> bool {
        device_lookup_stash().lock().unwrap().contains_key(C::LABEL)
    }

    fn device_interaction<C: OpcodeLaneSpec>(
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let (lookup_dev, n_rows, n_real) =
            device_lookup_stash().lock().unwrap().remove(C::LABEL)?;
        let t0 = std::time::Instant::now();
        // Resolved from the EMITTED per-column facts (gate-proven by the host
        // mirror against the generated writer) — no derivation rules.
        let descs =
            crate::witness::logup_descs::resolve_logup_descs(C::lookup_fields(), C::logup_descs());
        let max_w = crate::witness::logup_descs::max_tuple_width(&descs);
        let out = stwo_backend_cuda::logup_pairs::device_interaction_from_flats(
            lookup_dev.device_ptr,
            n_rows,
            n_real,
            &descs,
            &elements.alpha_powers()[..max_w],
            elements.z(),
        );
        if let Some((evals, sum)) = out {
            eprintln!(
                "jit_interaction[{}]: device logup {} rows x {} cols in {:.1} ms",
                C::LABEL,
                n_rows,
                descs.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
            Some((evals, sum))
        } else {
            // The host raw pairs were never computed (spawn skipped) and the flats
            // were never copied: no fallback exists by design. Fail loudly.
            panic!(
                "jit_interaction[{}]: device interaction kernel failed with no host \
                 fallback (STWO_CUDA_DEVICE_INTERACTION=1 skipped the flats copy)",
                C::LABEL
            );
        }
    }

    fn builtin_device_interaction_pending<C: BuiltinLaneSpec>() -> bool {
        device_lookup_stash().lock().unwrap().contains_key(C::LABEL)
    }

    fn builtin_device_interaction<C: BuiltinLaneSpec>(
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let (lookup_dev, n_rows, n_real) =
            device_lookup_stash().lock().unwrap().remove(C::LABEL)?;
        let t0 = std::time::Instant::now();
        let descs =
            crate::witness::logup_descs::resolve_logup_descs(C::lookup_fields(), C::logup_descs());
        let max_w = crate::witness::logup_descs::max_tuple_width(&descs);
        let out = stwo_backend_cuda::logup_pairs::device_interaction_from_flats(
            lookup_dev.device_ptr,
            n_rows,
            n_real,
            &descs,
            &elements.alpha_powers()[..max_w],
            elements.z(),
        );
        if let Some((evals, sum)) = out {
            eprintln!(
                "jit_interaction[{}]: device logup {} rows x {} cols in {:.1} ms",
                C::LABEL,
                n_rows,
                descs.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
            Some((evals, sum))
        } else {
            // The host flats were never copied (the stash skipped the D2H): no
            // fallback exists by design. Fail loudly.
            panic!(
                "jit_interaction[{}]: device interaction kernel failed with no host \
                 fallback (STWO_CUDA_DEVICE_INTERACTION=1 skipped the flats copy)",
                C::LABEL
            );
        }
    }

    fn lane_write_trace<C: OpcodeLaneSpec>(
        gen: C::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (Evals<Self>, C::Claim, C::IGen) {
        let host_fallback = |gen: C::Gen| {
            let (trace, claim, interaction_gen) =
                C::host_write(gen, addr_state, id_state, vi_state);
            (Self::from_simd_evals(trace), claim, interaction_gen)
        };

        if !lane_enabled(C::LABEL) || !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return host_fallback(gen);
        }
        let Some(mem) = jit_memory else {
            eprintln!(
                "jit_prove[{}]: lane on but no jit_memory — falling back",
                C::LABEL
            );
            return host_fallback(gen);
        };

        let Some(out) = cuda_jit_lane::<C>(C::inputs(&gen), mem) else {
            return host_fallback(gen);
        };

        // STWO_JIT_PROVE_SHADOW=1: byte-diff every lane output against the host
        // writer in-process (pure read; no double-feeding) with exact coordinates.
        if std::env::var("STWO_JIT_PROVE_SHADOW").as_deref() == Ok("1") {
            use stwo::prover::backend::Column;
            let device_cols: Vec<Vec<u32>> = out
                .cols
                .iter()
                .map(|c| c.to_cpu().into_iter().map(|v| v.0).collect())
                .collect();
            C::shadow_compare(
                C::inputs(&gen),
                &device_cols,
                &out.lookup_flat,
                &out.sub_flat,
                out.column_length,
                addr_state,
                id_state,
                vi_state,
            );
        }

        // Committed base trace: the kernel's columns, device-resident, consumed
        // without a device-to-device copy.
        let domain = CanonicCoset::new(out.log_size).circle_domain();
        let trace: Evals<Self> = out
            .cols
            .into_iter()
            .map(|c| CircleEvaluation::new(domain, c))
            .collect();

        let claim = C::claim(out.log_size);
        let interaction_gen = if device_interaction_enabled() {
            // §6a: park the device lookup buffer for interaction time; the host
            // igen below is an EMPTY placeholder that the device path replaces
            // (if the device path were to fail, the empty interaction trace fails
            // composition loudly — never silently wrong).
            device_lookup_stash().lock().unwrap().insert(
                C::LABEL,
                // No ENABLER mult source in opcode descriptors; n_real unused.
                (out.lookup_dev, out.column_length, out.column_length),
            );
            C::igen_from_flats(out.log_size, &[], 0)
        } else {
            C::igen_from_flats(out.log_size, &out.lookup_flat, out.column_length)
        };
        C::feed_from_flats(
            &out.sub_flat,
            out.column_length,
            addr_state,
            id_state,
            vi_state,
        );

        (trace, claim, interaction_gen)
    }
}

// ------------------------------- component specs ------------------------------------

/// Implements [`OpcodeLaneSpec`] for a CasmState opcode component whose module
/// provides the standard entry points (`record_*`, `interaction_gen_from_flat_
/// lookup_words`, `feed_sub_inputs_from_flat`, `shadow_compare_against_host`).
macro_rules! opcode_lane_spec {
    (
        $spec:ident, $module:ident, $label:literal,
        n_trace = $n_trace:expr, n_lookup = $n_lookup:expr, n_sub = $n_sub:expr,
        record = $record:path
    ) => {
        pub struct $spec;

        impl OpcodeLaneSpec for $spec {
            const LABEL: &'static str = $label;
            const N_TRACE: usize = $n_trace;
            const N_LOOKUP_WORDS: usize = $n_lookup;
            const N_SUB_WORDS: usize = $n_sub;
            type Gen = $module::ClaimGenerator;
            type Claim = cairo_air::components::$module::Claim;
            type IGen = $module::InteractionClaimGenerator;

            fn inputs(gen: &Self::Gen) -> &[CasmState] {
                &gen.inputs
            }
            fn lookup_fields() -> &'static [(&'static str, usize)] {
                $module::JIT_LOOKUP_FIELDS
            }
            fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
                $module::JIT_LOGUP_DESCS
            }
            fn record() -> RecordingOutput {
                $record()
            }
            fn host_write(
                gen: Self::Gen,
                addr_state: &memory_address_to_id::ClaimGenerator,
                id_state: &memory_id_to_big::ClaimGenerator,
                vi_state: &verify_instruction::ClaimGenerator,
            ) -> (SimdEvals, Self::Claim, Self::IGen) {
                let (trace, claim, igen) = gen.write_trace(addr_state, id_state, vi_state);
                (trace.to_evals(), claim, igen)
            }
            fn claim(log_size: u32) -> Self::Claim {
                Self::Claim { log_size }
            }
            fn igen_from_flats(log_size: u32, words: &[u32], n_rows: usize) -> Self::IGen {
                $module::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
            }
            fn feed_from_flats(
                words: &[u32],
                n_rows: usize,
                addr_state: &memory_address_to_id::ClaimGenerator,
                id_state: &memory_id_to_big::ClaimGenerator,
                vi_state: &verify_instruction::ClaimGenerator,
            ) {
                $module::feed_sub_inputs_from_flat(words, n_rows, addr_state, id_state, vi_state);
            }
            fn shadow_compare(
                inputs: &[CasmState],
                device_cols: &[Vec<u32>],
                lookup_flat: &[u32],
                sub_flat: &[u32],
                n_padded: usize,
                addr_state: &memory_address_to_id::ClaimGenerator,
                id_state: &memory_id_to_big::ClaimGenerator,
                vi_state: &verify_instruction::ClaimGenerator,
            ) {
                $module::shadow_compare_against_host(
                    inputs,
                    device_cols,
                    lookup_flat,
                    sub_flat,
                    n_padded,
                    addr_state,
                    id_state,
                    vi_state,
                );
            }
        }
    };
}

opcode_lane_spec!(
    AddOpcodeLane,
    add_opcode,
    "add_opcode",
    n_trace = 103,
    n_lookup = 117,
    n_sub = 13,
    record = add_opcode::record_add_opcode
);
opcode_lane_spec!(
    AssertEqOpcodeLane,
    assert_eq_opcode,
    "assert_eq_opcode",
    n_trace = 12,
    n_lookup = 24,
    n_sub = 9,
    record = assert_eq_opcode::record_assert_eq_opcode
);
opcode_lane_spec!(
    JnzOpcodeTakenLane,
    jnz_opcode_taken,
    "jnz_opcode_taken",
    n_trace = 47,
    n_lookup = 84,
    n_sub = 11,
    record = jnz_opcode_taken::record_jnz_opcode_taken
);
opcode_lane_spec!(
    AddOpcodeSmallLane,
    add_opcode_small,
    "add_opcode_small",
    n_trace = 39,
    n_lookup = 117,
    n_sub = 13,
    record = add_opcode_small::record_add_opcode_small
);
opcode_lane_spec!(
    AssertEqOpcodeImmLane,
    assert_eq_opcode_imm,
    "assert_eq_opcode_imm",
    n_trace = 9,
    n_lookup = 24,
    n_sub = 9,
    record = assert_eq_opcode_imm::record_assert_eq_opcode_imm
);
opcode_lane_spec!(
    AssertEqOpcodeDoubleDerefLane,
    assert_eq_opcode_double_deref,
    "assert_eq_opcode_double_deref",
    n_trace = 19,
    n_lookup = 57,
    n_sub = 11,
    record = assert_eq_opcode_double_deref::record_assert_eq_opcode_double_deref
);
opcode_lane_spec!(
    CallOpcodeAbsLane,
    call_opcode_abs,
    "call_opcode_abs",
    n_trace = 25,
    n_lookup = 117,
    n_sub = 13,
    record = call_opcode_abs::record_call_opcode_abs
);
opcode_lane_spec!(
    CallOpcodeRelImmLane,
    call_opcode_rel_imm,
    "call_opcode_rel_imm",
    n_trace = 24,
    n_lookup = 117,
    n_sub = 13,
    record = call_opcode_rel_imm::record_call_opcode_rel_imm
);
opcode_lane_spec!(
    JnzOpcodeNonTakenLane,
    jnz_opcode_non_taken,
    "jnz_opcode_non_taken",
    n_trace = 9,
    n_lookup = 51,
    n_sub = 9,
    record = jnz_opcode_non_taken::record_jnz_opcode_non_taken
);
opcode_lane_spec!(
    JumpOpcodeAbsLane,
    jump_opcode_abs,
    "jump_opcode_abs",
    n_trace = 14,
    n_lookup = 51,
    n_sub = 9,
    record = jump_opcode_abs::record_jump_opcode_abs
);
opcode_lane_spec!(
    JumpOpcodeDoubleDerefLane,
    jump_opcode_double_deref,
    "jump_opcode_double_deref",
    n_trace = 21,
    n_lookup = 84,
    n_sub = 11,
    record = jump_opcode_double_deref::record_jump_opcode_double_deref
);
opcode_lane_spec!(
    JumpOpcodeRelLane,
    jump_opcode_rel,
    "jump_opcode_rel",
    n_trace = 16,
    n_lookup = 51,
    n_sub = 9,
    record = jump_opcode_rel::record_jump_opcode_rel
);
opcode_lane_spec!(
    JumpOpcodeRelImmLane,
    jump_opcode_rel_imm,
    "jump_opcode_rel_imm",
    n_trace = 13,
    n_lookup = 51,
    n_sub = 9,
    record = jump_opcode_rel_imm::record_jump_opcode_rel_imm
);
opcode_lane_spec!(
    RetOpcodeLane,
    ret_opcode,
    "ret_opcode",
    n_trace = 16,
    n_lookup = 84,
    n_sub = 11,
    record = ret_opcode::record_ret_opcode
);

// ---------------- Builtin lane (D′: slot-layout inputs, computed EC/blake deduces) ---

/// Per-component spec for a BUILTIN JIT component (the pedersen_aggregator /
/// blake_round class): inputs arrive as caller-built slot columns — `[flat
/// words | enabler | iota | mults…]`, raw u32 — rather than CasmStates, and the
/// downstream sub feeds are component-specific (a closure at the seam, since the
/// downstream state tuples differ per component).
pub trait BuiltinLaneSpec {
    const LABEL: &'static str;
    const N_TRACE: usize;
    const N_LOOKUP_WORDS: usize;
    const N_SUB_WORDS: usize;
    /// Whether this component's kernel reads the pedersen points table
    /// (computed EC deduces). When true, the lane registers the HOST-BUILT
    /// table on device before launching — the GPU-generated table was
    /// falsified by the deduce-gate oracle and is quarantined.
    const NEEDS_PEDERSEN_TABLE: bool;
    type Claim;
    type IGen;
    fn record() -> RecordingOutput;
    fn claim(log_size: u32) -> Self::Claim;
    /// `(name, width)` per `LookupData` field, declaration order — the §6a
    /// device-interaction descriptor builder's input.
    fn lookup_fields() -> &'static [(&'static str, usize)];
    /// The emitted per-column descriptor FACTS (`JIT_LOGUP_DESCS`).
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact];
    /// `n_real` is the pre-padding row count (blake_round's IGen stores it as the
    /// enabler bound; the aggregator's IGen has no such field and ignores it).
    fn igen_from_flats(log_size: u32, n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen;
}

/// Every lane recording, for the AOT `kernel_emit` tool (design §4/§17, M3):
/// `(label, recorded witness program)`. Colocated with the spec definitions;
/// gpu-prover's schedule-table pin asserts each label is a schedule node, so a
/// lane added without a registry entry (or vice versa) fails a gate, not
/// silently ships NVRTC-only.
pub fn all_lane_recordings() -> Vec<(
    &'static str,
    stwo_backend_cuda::jit_witness::isa::WitnessProgram,
)> {
    fn op<C: OpcodeLaneSpec>() -> (
        &'static str,
        stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    ) {
        (C::LABEL, C::record().program)
    }
    fn bi<C: BuiltinLaneSpec>() -> (
        &'static str,
        stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    ) {
        (C::LABEL, C::record().program)
    }
    vec![
        op::<AddOpcodeLane>(),
        op::<AssertEqOpcodeLane>(),
        op::<JnzOpcodeTakenLane>(),
        op::<AddOpcodeSmallLane>(),
        op::<AssertEqOpcodeImmLane>(),
        op::<AssertEqOpcodeDoubleDerefLane>(),
        op::<CallOpcodeAbsLane>(),
        op::<CallOpcodeRelImmLane>(),
        op::<JnzOpcodeNonTakenLane>(),
        op::<JumpOpcodeAbsLane>(),
        op::<JumpOpcodeDoubleDerefLane>(),
        op::<JumpOpcodeRelLane>(),
        op::<JumpOpcodeRelImmLane>(),
        op::<RetOpcodeLane>(),
        bi::<PedersenAggregatorW18Lane>(),
        bi::<BlakeRoundLane>(),
        bi::<PartialEcMulW18Lane>(),
        bi::<PartialEcMulGenericLane>(),
        bi::<Cube252Lane>(),
    ]
}

/// The device-DAG count-feed plan for one component (B2): the emitted
/// `SUB_FEED_LAYOUT`, a LUT provider per relation family, and a per-family
/// merge into the downstream states' `add_count_tables`. Relations covered by
/// [`crate::witness::device_feed::COUNT_RELATIONS`] feed ON DEVICE from the
/// launch's resident sub buffer; the caller's host `feed` closure receives the
/// device-fed state set and MUST skip those relations (double-feeding corrupts
/// multiplicities). Fail-closed: any unavailability feeds everything on host.
/// B3 device edges master switch: when on, producer lanes SKIP their
/// input-list host feed and stash the sub buffer for the consumer.
pub(crate) fn edges_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_EDGES").as_deref() == Ok("1")
}

/// The B3 edge stash: producer label-keyed (BY CONSUMER edge key) device sub
/// buffer + its HOST flat mirror + the producer's padded row count. The host
/// flat makes every consumer-side failure recoverable on CPU (rebuild the
/// inputs with the edge-gate math) — exactly-once feeds with no pair rerun.
type EdgeStash = std::sync::Mutex<
    std::collections::HashMap<&'static str, (stwo_backend_cuda::BaseFieldVec, Vec<u32>, usize)>,
>;
fn edge_stash() -> &'static EdgeStash {
    static STASH: std::sync::OnceLock<EdgeStash> = std::sync::OnceLock::new();
    STASH.get_or_init(Default::default)
}

/// Pop a stashed edge for `consumer_key` (one-shot per prove).
pub(crate) fn take_edge(
    consumer_key: &'static str,
) -> Option<(stwo_backend_cuda::BaseFieldVec, Vec<u32>, usize)> {
    edge_stash().lock().unwrap().remove(consumer_key)
}

/// The consumer-side input source for a builtin launch.
pub(crate) enum BuiltinInputs<'a> {
    /// Host-built slot columns (uploaded by the launch).
    HostCols(&'a [Vec<u32>]),
    /// Device edge: gathered producer columns + host tail (enabler/iota).
    Edge {
        device_cols: Vec<stwo_backend_cuda::BaseFieldVec>,
        host_tail: Vec<Vec<u32>>,
    },
}

pub(crate) struct DeviceFeedPlan<'a> {
    pub layout: &'static [(&'static str, usize, &'static str, u32, usize, usize)],
    pub lut_for: &'a dyn Fn(&'static str) -> Vec<u32>,
    pub merge: &'a dyn Fn(&'static str, &[u32]),
    /// When EVERY relation the component feeds is count-style, the seam sets
    /// this and provides NO host feed: a device-feed failure then fails the
    /// whole lane (`None` → the host WRITER reruns — still exactly-once feeds,
    /// never a silent miss).
    pub require: bool,
}

/// Generic builtin device write: validate the recording against the spec, launch
/// it on the caller-built slot columns, and rebuild (trace, claim, igen); the
/// caller then applies its component-specific sub feeds via `feed(sub_flat,
/// n_padded, device_fed_states)`. `None` = fall back to the host writer (reason
/// logged) — the caller's `gen` is untouched (this only ever READS the inputs),
/// so the host path stays valid.
///
/// §6a note: builtins always take the host-flats interaction path for now
/// (`want_host_lookup = true`); extending the device-interaction stash to the
/// builtin specs is a separate, separately-gated step.
pub(crate) fn builtin_cuda_write_trace<C: BuiltinLaneSpec>(
    input_cols: &[Vec<u32>],
    n_real: usize,
    mem: &Arc<Memory>,
    device_feed: Option<DeviceFeedPlan<'_>>,
    feed: impl FnOnce(&[u32], usize, &[&'static str]),
) -> Option<(Evals<stwo_backend_cuda::CudaBackend>, C::Claim, C::IGen)> {
    builtin_cuda_write_trace_from::<C>(
        BuiltinInputs::HostCols(input_cols),
        n_real,
        mem,
        device_feed,
        None,
        feed,
    )
}

/// Full-generality builtin device write (host cols OR device edge inputs;
/// optional producer-side edge stash under `stash_edge`).
pub(crate) fn builtin_cuda_write_trace_from<C: BuiltinLaneSpec>(
    inputs: BuiltinInputs<'_>,
    n_real: usize,
    mem: &Arc<Memory>,
    device_feed: Option<DeviceFeedPlan<'_>>,
    stash_edge: Option<&'static str>,
    feed: impl FnOnce(&[u32], usize, &[&'static str]),
) -> Option<(Evals<stwo_backend_cuda::CudaBackend>, C::Claim, C::IGen)> {
    if !lane_enabled(C::LABEL) || !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return None;
    }
    let t0 = std::time::Instant::now();
    let recording = C::record();
    let program = &recording.program;
    if program.n_cols as usize != C::N_TRACE
        || program.n_lookup_words as usize != C::N_LOOKUP_WORDS
        || program.n_sub_words as usize != C::N_SUB_WORDS
        || !recording.poisoned_cols.is_empty()
        || !recording.poisoned_lookup_words.is_empty()
        || !recording.poisoned_sub_words.is_empty()
    {
        eprintln!(
            "jit_prove[{}]: recording shape mismatch (cols {} vs {}, lookup {} vs {}, sub {} \
             vs {}, poison {}/{}/{}) — falling back",
            C::LABEL,
            program.n_cols,
            C::N_TRACE,
            program.n_lookup_words,
            C::N_LOOKUP_WORDS,
            program.n_sub_words,
            C::N_SUB_WORDS,
            recording.poisoned_cols.len(),
            recording.poisoned_lookup_words.len(),
            recording.poisoned_sub_words.len(),
        );
        return None;
    }
    if C::NEEDS_PEDERSEN_TABLE && !ensure_device_pedersen_table() {
        eprintln!(
            "jit_prove[{}]: host pedersen table registration failed — falling back",
            C::LABEL
        );
        return None;
    }
    // Callers build the FULL canonical slot layout; a body that never reads its
    // trailing slots (blake_round's iota) records fewer inputs — trim to the
    // program's read extent. Slots are positional, so trailing-only truncation
    // is sound; too FEW columns is still a hard mismatch.
    let n_inputs = program.n_inputs as usize;
    let provided = match &inputs {
        BuiltinInputs::HostCols(c) => c.len(),
        BuiltinInputs::Edge {
            device_cols,
            host_tail,
        } => device_cols.len() + host_tail.len(),
    };
    if provided < n_inputs {
        eprintln!(
            "jit_prove[{}]: {provided} input columns built, program reads {n_inputs} —              falling back",
            C::LABEL,
        );
        return None;
    }
    let inputs = match inputs {
        BuiltinInputs::HostCols(c) => BuiltinInputs::HostCols(&c[..n_inputs.min(c.len())]),
        other => other,
    };
    stwo_backend_cuda::jit_witness::register_recorded_program(C::LABEL, recording.program);

    let tables = stwo_backend_cuda::exec_tables::exec_tables_cached(
        mem.address_to_id.as_ptr() as usize,
        || {
            let addr_ids: Vec<u32> = mem.address_to_id.iter().map(|e| e.0).collect();
            stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(
                &addr_ids,
                &mem.f252_values,
                &mem.small_values,
            )
        },
    );

    let want_host_lookup = !device_interaction_enabled();
    let (cols, lookup_dev, lookup_flat, sub_dev, sub_flat) = match &inputs {
        BuiltinInputs::HostCols(input_cols) => {
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
                C::LABEL,
                input_cols,
                tables,
                want_host_lookup,
            )?
        }
        BuiltinInputs::Edge {
            device_cols,
            host_tail,
        } => {
            let ptrs: Vec<*const u32> = device_cols.iter().map(|c| c.device_ptr).collect();
            let n = host_tail[0].len();
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_mixed(
                C::LABEL,
                &ptrs,
                host_tail,
                n,
                tables,
                want_host_lookup,
            )?
        }
    };
    let column_length = match &inputs {
        BuiltinInputs::HostCols(input_cols) => input_cols[0].len(),
        BuiltinInputs::Edge { host_tail, .. } => host_tail[0].len(),
    };
    let log_size = column_length.ilog2();
    if cols.len() != C::N_TRACE {
        eprintln!(
            "jit_prove[{}]: kernel returned {} columns, expected {} — falling back",
            C::LABEL,
            cols.len(),
            C::N_TRACE
        );
        return None;
    }
    eprintln!(
        "jit_prove[{}]: device witness {} real rows (log_size {}) in {:.1} ms",
        C::LABEL,
        n_real,
        log_size,
        t0.elapsed().as_secs_f64() * 1e3,
    );

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace: Evals<stwo_backend_cuda::CudaBackend> = cols
        .into_iter()
        .map(|c| CircleEvaluation::new(domain, c))
        .collect();
    let claim = C::claim(log_size);
    let igen = if !want_host_lookup {
        // §6a: park the device lookup buffer (with the ENABLER bound n_real) for
        // interaction time; the empty placeholder igen is replaced by the device
        // path — if that path were to fail, the empty interaction trace fails
        // composition loudly, never silently wrong.
        device_lookup_stash()
            .lock()
            .unwrap()
            .insert(C::LABEL, (lookup_dev, column_length, n_real));
        C::igen_from_flats(log_size, n_real, &[], 0)
    } else {
        C::igen_from_flats(log_size, n_real, &lookup_flat, column_length)
    };

    // Device-DAG count feed: count-style relations feed on device from the
    // resident sub buffer; the host closure then skips them. Any failure feeds
    // everything on host (empty skip set) — never a double feed, never a miss.
    let mut device_fed: Vec<&'static str> = Vec::new();
    if let Some(plan) = device_feed {
        let (descs, lut_slots, counts_slots) = crate::witness::device_feed::build_feed_descriptors(
            plan.layout,
            crate::witness::device_feed::COUNT_RELATIONS,
        );
        if !descs.is_empty() {
            let luts: Vec<Vec<u32>> = lut_slots.iter().map(|s| (plan.lut_for)(s)).collect();
            let sizes: Vec<usize> = counts_slots
                .iter()
                .map(|s| {
                    let rel = crate::witness::device_feed::COUNT_RELATIONS
                        .iter()
                        .find(|r| r.state_param == *s)
                        .expect("counts slot always registry-backed");
                    rel.n_relations * rel.table_size
                })
                .collect();
            match stwo_backend_cuda::exec_tables::run_witness_feed_counts(
                &sub_dev,
                column_length,
                &descs,
                &luts,
                &sizes,
            ) {
                Some(counts) => {
                    for (slot, c) in counts_slots.iter().zip(&counts) {
                        (plan.merge)(slot, c);
                    }
                    device_fed = counts_slots;
                    eprintln!(
                        "jit_prove[{}]: device count feed merged {} relation families",
                        C::LABEL,
                        device_fed.len()
                    );
                }
                None => {
                    if plan.require {
                        eprintln!(
                            "jit_prove[{}]: REQUIRED device count feed unavailable —                              falling back to the host writer",
                            C::LABEL
                        );
                        return None;
                    }
                    eprintln!(
                        "jit_prove[{}]: device count feed unavailable — host feeds all",
                        C::LABEL
                    );
                }
            }
        }
    }
    // B3 producer role: stash the sub buffer for the consumer edge and add the
    // consumer's state key to the skip set — its input-list feed happens as the
    // consumer's device inputs (or the consumer rebuilds from the stashed HOST
    // flat on any failure; exactly-once either way).
    if let Some(consumer_key) = stash_edge {
        if edges_enabled() {
            edge_stash()
                .lock()
                .unwrap()
                .insert(consumer_key, (sub_dev, sub_flat.clone(), column_length));
            device_fed.push(consumer_key);
            eprintln!(
                "jit_prove[{}]: stashed device edge for {consumer_key}",
                C::LABEL
            );
        }
    }
    feed(&sub_flat, column_length, &device_fed);
    Some((trace, claim, igen))
}

use crate::witness::components::{blake_round, pedersen_aggregator_window_bits_18};

/// Register the HOST-BUILT `PEDERSEN_TABLE_18` on device (borrowed mode) — the
/// deduce lane's only permitted table source: the oracle falsified the
/// GPU-generated table (144/256 rows, run 20260705T113615Z). Idempotent per
/// process; `false` (stub build / upload failure) → callers fall back to host.
pub(crate) fn ensure_device_pedersen_table() -> bool {
    use stwo_cairo_common::preprocessed_columns::pedersen::PedersenPoints;
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        return false;
    }
    let n_rows = PedersenPoints::<18>::new(0).get_data().len();
    stwo_backend_cuda::pedersen_table::register_borrowed_pedersen_table(n_rows, |c, buf| {
        buf.extend(PedersenPoints::<18>::new(c).get_data().iter().map(|m| m.0));
    })
}

pub struct PedersenAggregatorW18Lane;
impl BuiltinLaneSpec for PedersenAggregatorW18Lane {
    const LABEL: &'static str = "pedersen_aggregator_window_bits_18";
    const N_TRACE: usize = 206;
    const N_LOOKUP_WORDS: usize = pedersen_aggregator_window_bits_18::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = pedersen_aggregator_window_bits_18::N_SUB_INPUT_WORDS;
    type Claim = cairo_air::components::pedersen_aggregator_window_bits_18::Claim;
    type IGen = pedersen_aggregator_window_bits_18::InteractionClaimGenerator;

    fn record() -> RecordingOutput {
        pedersen_aggregator_window_bits_18::record_pedersen_aggregator_window_bits_18()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        pedersen_aggregator_window_bits_18::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        pedersen_aggregator_window_bits_18::JIT_LOGUP_DESCS
    }
    const NEEDS_PEDERSEN_TABLE: bool = true;
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        pedersen_aggregator_window_bits_18::interaction_gen_from_flat_lookup_words(
            log_size, words, n_rows,
        )
    }
}

pub struct BlakeRoundLane;
impl BuiltinLaneSpec for BlakeRoundLane {
    const LABEL: &'static str = "blake_round";
    const N_TRACE: usize = 212;
    const N_LOOKUP_WORDS: usize = blake_round::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = blake_round::N_SUB_INPUT_WORDS;
    type Claim = cairo_air::components::blake_round::Claim;
    type IGen = blake_round::InteractionClaimGenerator;

    fn record() -> RecordingOutput {
        blake_round::record_blake_round()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        blake_round::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        blake_round::JIT_LOGUP_DESCS
    }
    const NEEDS_PEDERSEN_TABLE: bool = false;
    fn igen_from_flats(log_size: u32, n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        blake_round::interaction_gen_from_flat_lookup_words(log_size, n_real, words, n_rows)
    }
}

use crate::witness::components::{cube_252, partial_ec_mul_generic, partial_ec_mul_window_bits_18};

pub struct PartialEcMulW18Lane;
impl BuiltinLaneSpec for PartialEcMulW18Lane {
    const LABEL: &'static str = "partial_ec_mul_window_bits_18";
    const N_TRACE: usize = 297;
    const N_LOOKUP_WORDS: usize = partial_ec_mul_window_bits_18::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = partial_ec_mul_window_bits_18::N_SUB_INPUT_WORDS;
    // The recorded body's points-table + EC-round deduces read the device table.
    const NEEDS_PEDERSEN_TABLE: bool = true;
    type Claim = cairo_air::components::partial_ec_mul_window_bits_18::Claim;
    type IGen = partial_ec_mul_window_bits_18::InteractionClaimGenerator;

    fn record() -> RecordingOutput {
        partial_ec_mul_window_bits_18::record_partial_ec_mul_window_bits_18()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        partial_ec_mul_window_bits_18::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        partial_ec_mul_window_bits_18::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        partial_ec_mul_window_bits_18::interaction_gen_from_flat_lookup_words(
            log_size, n_real, words, n_rows,
        )
    }
}

pub struct PartialEcMulGenericLane;
impl BuiltinLaneSpec for PartialEcMulGenericLane {
    const LABEL: &'static str = "partial_ec_mul_generic";
    const N_TRACE: usize = 624;
    const N_LOOKUP_WORDS: usize = partial_ec_mul_generic::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = partial_ec_mul_generic::N_SUB_INPUT_WORDS;
    // TRUE although the body never READS the table: the felt deduces embed the
    // fp256 chain, so this kernel's CUmodule declares the table globals and the
    // fail-closed module-load fill rejects it when no host table is registered
    // (ROUND-28 pod finding: generic runs BEFORE the aggregator, so it cannot
    // ride anyone else's registration). Registration is process-cached — the
    // extra ensure is idempotent.
    const NEEDS_PEDERSEN_TABLE: bool = true;
    type Claim = cairo_air::components::partial_ec_mul_generic::Claim;
    type IGen = partial_ec_mul_generic::InteractionClaimGenerator;

    fn record() -> RecordingOutput {
        partial_ec_mul_generic::record_partial_ec_mul_generic()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        partial_ec_mul_generic::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        partial_ec_mul_generic::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        partial_ec_mul_generic::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct Cube252Lane;
impl BuiltinLaneSpec for Cube252Lane {
    const LABEL: &'static str = "cube_252";
    const N_TRACE: usize = 141;
    const N_LOOKUP_WORDS: usize = cube_252::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = cube_252::N_SUB_INPUT_WORDS;
    // TRUE for the same reason as the generic lane: fp256 embed => the module
    // declares the table globals => registration must precede module load.
    const NEEDS_PEDERSEN_TABLE: bool = true;
    type Claim = cairo_air::components::cube_252::Claim;
    type IGen = cube_252::InteractionClaimGenerator;

    fn record() -> RecordingOutput {
        cube_252::record_cube_252()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        cube_252::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        cube_252::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        cube_252::interaction_gen_from_flat_lookup_words(log_size, n_real, words, n_rows)
    }
}

/// Device write for the ALL-COUNT builtins (w18 / generic / cube_252): every
/// downstream relation is count-style, so the device feed is REQUIRED and no
/// host feed exists — any unavailability falls back to the host writer. The
/// caller supplies the packed inputs already padded by the host preamble rule
/// (first-packed-row replication) flattened to slot columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn all_count_builtin_write_trace<C: BuiltinLaneSpec>(
    cols: &[Vec<u32>],
    n_real: usize,
    mem: &Arc<Memory>,
    layout: &'static [(&'static str, usize, &'static str, u32, usize, usize)],
    lut_for: &dyn Fn(&'static str) -> Vec<u32>,
    merge: &dyn Fn(&'static str, &[u32]),
) -> Option<(Evals<stwo_backend_cuda::CudaBackend>, C::Claim, C::IGen)> {
    let plan = DeviceFeedPlan {
        layout,
        lut_for,
        merge,
        require: true,
    };
    builtin_cuda_write_trace::<C>(cols, n_real, mem, Some(plan), |_sub, _n, fed| {
        debug_assert!(
            !fed.is_empty(),
            "require-mode feed reached with nothing fed"
        );
    })
}

use crate::witness::components::{range_check_20, range_check_9_9};

/// Backend seam for the `cube_252` base-trace write (poseidon-family fp256):
/// Simd = the generated writer verbatim; Cuda = the witness-JIT lane (12 slot
/// columns from the W27 input; ALL relations count-style — device feed
/// REQUIRED, host-writer fallback on any unavailability).
pub trait Cube252Witness: FromSimdColumns {
    fn write_trace(
        gen: cube_252::ClaimGenerator,
        range_check_9_9_state: &range_check_9_9::ClaimGenerator,
        range_check_20_state: &range_check_20::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::cube_252::Claim,
        cube_252::InteractionClaimGenerator,
    );
}

impl Cube252Witness for SimdBackend {
    fn write_trace(
        gen: cube_252::ClaimGenerator,
        range_check_9_9_state: &range_check_9_9::ClaimGenerator,
        range_check_20_state: &range_check_20::ClaimGenerator,
        _jit_memory: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::cube_252::Claim,
        cube_252::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(range_check_9_9_state, range_check_20_state);
        (trace.to_evals(), claim, igen)
    }
}

impl Cube252Witness for stwo_backend_cuda::CudaBackend {
    fn write_trace(
        gen: cube_252::ClaimGenerator,
        range_check_9_9_state: &range_check_9_9::ClaimGenerator,
        range_check_20_state: &range_check_20::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::cube_252::Claim,
        cube_252::InteractionClaimGenerator,
    ) {
        if let Some(mem) = jit_memory {
            let packed: Vec<cube_252::PackedInputType> = gen.packed_inputs.lock().unwrap().clone();
            let remainder_empty = gen.remainder_inputs.lock().unwrap().is_empty();
            if !packed.is_empty() && remainder_empty {
                let n_vec_rows = packed.len();
                let n_real = n_vec_rows * N_LANES;
                let packed_size = n_vec_rows.next_power_of_two();
                let size = packed_size * N_LANES;
                let mut padded = packed;
                padded.resize(packed_size, *padded.first().unwrap());
                // Slot columns: W27 words 0..10 | enabler 10 | iota 11.
                let mut cols: Vec<Vec<u32>> = vec![Vec::with_capacity(size); 12];
                for p in &padded {
                    for l in 0..N_LANES {
                        for i in 0..10 {
                            cols[i].push(p.get_m31(i).to_array()[l].0);
                        }
                    }
                }
                cols[10] = (0..size).map(|r| u32::from(r < n_real)).collect();
                cols[11] = (0..size).map(|r| r as u32).collect();
                let lut_for = |family: &'static str| -> Vec<u32> {
                    match family {
                        "range_check_9_9_state" => range_check_9_9_state.input_to_row_lut(),
                        other => panic!("unexpected LUT family {other}"),
                    }
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_9_9_state" => range_check_9_9_state.add_count_tables(counts),
                    "range_check_20_state" => range_check_20_state.add_count_tables(counts),
                    other => panic!("unexpected count family {other}"),
                };
                let launched = all_count_builtin_write_trace::<Cube252Lane>(
                    &cols,
                    n_real,
                    mem,
                    cube_252::SUB_FEED_LAYOUT,
                    &lut_for,
                    &merge,
                );
                if let Some(out) = launched {
                    return out;
                }
            }
        }
        let (trace, claim, igen) = gen.write_trace(range_check_9_9_state, range_check_20_state);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }
}
