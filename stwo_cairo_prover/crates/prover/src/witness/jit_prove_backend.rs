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
    std::collections::HashMap<&'static str, (stwo_backend_cuda::BaseFieldVec, usize)>,
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
    let Some((cols, lookup_dev, lookup_flat, sub_flat)) = launched else {
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
        let (lookup_dev, n_rows) = device_lookup_stash().lock().unwrap().remove(C::LABEL)?;
        let t0 = std::time::Instant::now();
        let fields = C::lookup_fields();
        let n = fields.len();
        let tuple_fields: Vec<(&str, usize)> = fields[..n - 2].to_vec();
        let m0 = fields[..n - 2].iter().map(|f| f.1).sum::<usize>() as u32;
        let descs =
            stwo_backend_cuda::logup_pairs::descriptors_for_fields(&tuple_fields, m0, m0 + 1);
        let max_w = tuple_fields.iter().map(|f| f.1).max()?;
        let out = stwo_backend_cuda::logup_pairs::device_interaction_from_flats(
            lookup_dev.device_ptr,
            n_rows,
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
            device_lookup_stash()
                .lock()
                .unwrap()
                .insert(C::LABEL, (out.lookup_dev, out.column_length));
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
