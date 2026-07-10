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
    add_ap_opcode, add_opcode, add_opcode_small, assert_eq_opcode, assert_eq_opcode_double_deref,
    assert_eq_opcode_imm, bitwise_builtin, blake_compress_opcode, blake_g, call_opcode_abs,
    call_opcode_rel_imm, jnz_opcode_non_taken, jnz_opcode_taken, jump_opcode_abs,
    jump_opcode_double_deref, jump_opcode_rel, jump_opcode_rel_imm, memory_address_to_id,
    memory_id_to_big, mul_opcode, mul_opcode_small, pedersen_builtin,
    poseidon_3_partial_rounds_chain, poseidon_aggregator, poseidon_builtin,
    poseidon_full_round_chain, qm_31_add_mul_opcode, range_check_252_width_27, range_check_builtin,
    ret_opcode, triple_xor_32, verify_instruction,
};
use crate::witness::exec_context::{PlannedDeviceEdge, WitnessExecContext};
use crate::witness::proof_shape::TracePartId;
use crate::witness::witness_eval::recording::RecordingOutput;

type Evals<B> = Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>;
type SimdEvals = Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>;

/// Canonical column-major inputs for one recorded witness program. `columns[i]`
/// is recorder input slot `i`, already padded by the generated writer's exact
/// first-row replication rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedInputColumns {
    pub n_real: usize,
    pub columns: Vec<Vec<u32>>,
}

impl RecordedInputColumns {
    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordedInputBuildError {
    EmptyInputs(&'static str),
    ScalarRemainder(&'static str),
    TooFewColumns {
        label: &'static str,
        expected: usize,
        actual: usize,
    },
    UnequalColumnLengths(&'static str),
    InvalidRealRowCount(&'static str),
    DeviceSeedOnly(&'static str),
}

impl core::fmt::Display for RecordedInputBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "recorded witness input build rejected: {self:?}")
    }
}

impl std::error::Error for RecordedInputBuildError {}

fn normalize_recorded_inputs(
    label: &'static str,
    program: &stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    mut inputs: RecordedInputColumns,
) -> Result<RecordedInputColumns, RecordedInputBuildError> {
    let expected = program.n_inputs as usize;
    if inputs.columns.len() < expected {
        return Err(RecordedInputBuildError::TooFewColumns {
            label,
            expected,
            actual: inputs.columns.len(),
        });
    }
    // Generated preambles sometimes expose a trailing iota slot that the
    // straight-line recording never reads. PreparedWitnessGraph binds only the
    // recorder ABI, so discard trailing-only columns at this single seam.
    inputs.columns.truncate(expected);
    let row_count = inputs.row_count();
    if row_count == 0 || inputs.n_real == 0 || inputs.n_real > row_count {
        return Err(RecordedInputBuildError::InvalidRealRowCount(label));
    }
    if inputs
        .columns
        .iter()
        .any(|column| column.len() != row_count)
    {
        return Err(RecordedInputBuildError::UnequalColumnLengths(label));
    }
    Ok(inputs)
}

/// §6a master switch: at witness time the lane STASHES its device-resident lookup
/// buffer (skipping the flats D2H + host repack); at interaction time — after the
/// lookup elements are drawn, which is why it cannot happen earlier — the pair
/// kernel + device finalize replace the host `write_interaction_trace` wholesale.
pub(crate) fn device_interaction_enabled() -> bool {
    std::env::var("STWO_CUDA_DEVICE_INTERACTION").as_deref() == Ok("1")
}

/// B2 v2 master switch: memory-table count families feed on device from the
/// lanes' resident sub buffers (gpu-native composed default; explicit =0 wins).
fn mem_count_feeds_enabled() -> bool {
    std::env::var("STWO_CUDA_MEM_COUNT_FEEDS").as_deref() == Ok("1")
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
    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_slot_columns(Self::LABEL, Self::inputs(gen))
    }
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
        device_fed: &[&'static str],
    );
    /// The emitted `SUB_FEED_LAYOUT` — the device count feed's descriptor input.
    fn sub_feed_layout() -> &'static [(&'static str, usize, &'static str, u32, usize, usize)];
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
    /// The device-resident SUB words — the B2 v2 memory count feed consumes
    /// them in place.
    sub_dev: stwo_backend_cuda::BaseFieldVec,
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
fn cuda_jit_lane<C: OpcodeLaneSpec>(
    exec_context: &WitnessExecContext,
    inputs: &[CasmState],
    mem: &Memory,
) -> Option<LaneOutput> {
    let t0 = std::time::Instant::now();
    let canonical = casm_slot_columns(C::LABEL, inputs).ok()?;
    let n_real = canonical.n_real;

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

    // The legacy row-major launcher and PreparedWitnessGraph now share this one
    // canonical column builder; transpose only the three state slots here.
    let column_length = canonical.row_count();
    let log_size = column_length.ilog2();
    let samples = (0..column_length)
        .map(|row| {
            (
                canonical.columns[0][row],
                canonical.columns[1][row],
                canonical.columns[2][row],
            )
        })
        .collect::<Vec<_>>();

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

    let destinations = exec_context.take_resident_witness_destination(
        C::LABEL,
        TracePartId::Main,
        C::N_TRACE,
        column_length,
        C::N_LOOKUP_WORDS * column_length,
        C::N_SUB_WORDS * column_length,
    );
    let resident_launch = destinations.is_some();
    let launched = match destinations {
        Some(destinations) => {
            stwo_backend_cuda::exec_tables::launch_recorded_witness_for_prove_into(
                C::LABEL,
                &samples,
                n_real,
                tables,
                !device_interaction_enabled(),
                destinations,
            )
        }
        None => stwo_backend_cuda::exec_tables::launch_recorded_witness_for_prove(
            C::LABEL,
            &samples,
            n_real,
            tables,
            !device_interaction_enabled(),
        ),
    };
    let Some((cols, lookup_dev, lookup_flat, sub_dev, sub_flat)) = launched else {
        // launch_recorded_witness_for_prove logged the specific reason.
        return None;
    };
    if resident_launch {
        exec_context.record_resident_witness_launch();
    }
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
        sub_dev,
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
    fn device_interaction_pending<C: OpcodeLaneSpec>(exec_context: &WitnessExecContext) -> bool {
        let _ = exec_context;
        false
    }

    /// §6a interaction-time hook: when this component's device lookup buffer was
    /// stashed at witness time, build its interaction trace ON DEVICE (pair kernel
    /// + proven finalize) and return it with the claimed sum. `None` = use the
    /// host `write_interaction_trace` path.
    fn device_interaction<C: OpcodeLaneSpec>(
        exec_context: &WitnessExecContext,
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let _ = exec_context;
        let _ = elements;
        None
    }

    /// Builtin-lane variants of the §6a hooks (same stash, `BuiltinLaneSpec`-keyed).
    fn builtin_device_interaction_pending<C: BuiltinLaneSpec>(
        exec_context: &WitnessExecContext,
    ) -> bool {
        let _ = exec_context;
        false
    }
    fn builtin_device_interaction<C: BuiltinLaneSpec>(
        exec_context: &WitnessExecContext,
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let _ = exec_context;
        let _ = elements;
        None
    }

    fn lane_write_trace<C: OpcodeLaneSpec>(
        exec_context: &WitnessExecContext,
        gen: C::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (Evals<Self>, C::Claim, C::IGen);
}

impl OpcodeJitBackend for SimdBackend {
    fn lane_write_trace<C: OpcodeLaneSpec>(
        _exec_context: &WitnessExecContext,
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
    fn device_interaction_pending<C: OpcodeLaneSpec>(exec_context: &WitnessExecContext) -> bool {
        exec_context.has_device_lookup(C::LABEL)
    }

    fn device_interaction<C: OpcodeLaneSpec>(
        exec_context: &WitnessExecContext,
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let lookup = exec_context.take_device_lookup(C::LABEL)?;
        let t0 = std::time::Instant::now();
        // Resolved from the EMITTED per-column facts (gate-proven by the host
        // mirror against the generated writer) — no derivation rules.
        let descs =
            crate::witness::logup_descs::resolve_logup_descs(C::lookup_fields(), C::logup_descs());
        let max_w = crate::witness::logup_descs::max_tuple_width(&descs);
        let out = stwo_backend_cuda::logup_pairs::device_interaction_from_flats(
            lookup.buffer.device_ptr,
            lookup.n_rows,
            lookup.n_real,
            &descs,
            &elements.alpha_powers()[..max_w],
            elements.z(),
        );
        if let Some((evals, sum)) = out {
            eprintln!(
                "jit_interaction[{}]: device logup {} rows x {} cols in {:.1} ms",
                C::LABEL,
                lookup.n_rows,
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

    fn builtin_device_interaction_pending<C: BuiltinLaneSpec>(
        exec_context: &WitnessExecContext,
    ) -> bool {
        exec_context.has_device_lookup(C::LABEL)
    }

    fn builtin_device_interaction<C: BuiltinLaneSpec>(
        exec_context: &WitnessExecContext,
        elements: &cairo_air::relations::CommonLookupElements,
    ) -> Option<(Evals<Self>, stwo::core::fields::qm31::SecureField)> {
        let lookup = exec_context.take_device_lookup(C::LABEL)?;
        let t0 = std::time::Instant::now();
        let descs =
            crate::witness::logup_descs::resolve_logup_descs(C::lookup_fields(), C::logup_descs());
        let max_w = crate::witness::logup_descs::max_tuple_width(&descs);
        let out = stwo_backend_cuda::logup_pairs::device_interaction_from_flats(
            lookup.buffer.device_ptr,
            lookup.n_rows,
            lookup.n_real,
            &descs,
            &elements.alpha_powers()[..max_w],
            elements.z(),
        );
        if let Some((evals, sum)) = out {
            eprintln!(
                "jit_interaction[{}]: device logup {} rows x {} cols in {:.1} ms",
                C::LABEL,
                lookup.n_rows,
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
        exec_context: &WitnessExecContext,
        gen: C::Gen,
        addr_state: &memory_address_to_id::ClaimGenerator,
        id_state: &memory_id_to_big::ClaimGenerator,
        vi_state: &verify_instruction::ClaimGenerator,
        jit_memory: Option<&Arc<Memory>>,
    ) -> (Evals<Self>, C::Claim, C::IGen) {
        let host_fallback = |gen: C::Gen, reason: &'static str| {
            exec_context.host_witness_fallback(C::LABEL, reason);
            let (trace, claim, interaction_gen) =
                C::host_write(gen, addr_state, id_state, vi_state);
            (Self::from_simd_evals(trace), claim, interaction_gen)
        };

        if !lane_enabled(C::LABEL) || !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            return host_fallback(gen, "recorded witness lane disabled or unavailable");
        }
        let Some(mem) = jit_memory else {
            eprintln!(
                "jit_prove[{}]: lane on but no jit_memory — falling back",
                C::LABEL
            );
            return host_fallback(gen, "missing device execution memory");
        };

        let Some(out) = cuda_jit_lane::<C>(exec_context, C::inputs(&gen), mem) else {
            return host_fallback(gen, "recorded witness lane rejected the launch");
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
            exec_context.insert_device_lookup(
                C::LABEL,
                // No ENABLER mult source in opcode descriptors; n_real unused.
                out.lookup_dev,
                out.column_length,
                out.column_length,
            );
            C::igen_from_flats(out.log_size, &[], 0)
        } else {
            C::igen_from_flats(out.log_size, &out.lookup_flat, out.column_length)
        };
        // B2 v2: the memory count families (every opcode's addr/id words) feed
        // ON DEVICE from the resident sub buffer; the host feeder skips them.
        // Any failure feeds everything on host — exactly-once either way.
        let mut device_fed: Vec<&'static str> = Vec::new();
        if mem_count_feeds_enabled() {
            let sizes = |st: &'static str| match st {
                "memory_address_to_id_state" => Some((addr_state.table_size(), 0)),
                "memory_id_to_big_state" => {
                    Some((id_state.big_table_size(), id_state.small_table_size()))
                }
                _ => None,
            };
            let (descs, lut_slots, counts_slots, slot_sizes) =
                crate::witness::device_feed::build_feed_descriptors_sized(
                    C::sub_feed_layout(),
                    crate::witness::device_feed::COUNT_RELATIONS,
                    &sizes,
                );
            if !descs.is_empty() {
                // The 13 lane opcodes feed no LUT families; a layout growing one
                // must extend this seam — fail loudly, never feed wrong.
                let luts: Vec<Vec<u32>> = lut_slots
                    .iter()
                    .map(|f| panic!("unexpected LUT count family in opcode lane: {f}"))
                    .collect();
                match stwo_backend_cuda::exec_tables::run_witness_feed_counts(
                    &out.sub_dev,
                    out.column_length,
                    &descs,
                    &luts,
                    &slot_sizes,
                ) {
                    Some(counts) => {
                        for (slot, c) in counts_slots.iter().zip(&counts) {
                            match *slot {
                                "memory_address_to_id_state" => addr_state.add_count_tables(c),
                                "memory_id_to_big_state" => id_state.add_big_count_tables(c),
                                "memory_id_to_big_state#small" => {
                                    id_state.add_small_count_tables(c)
                                }
                                other => panic!("unrouted count family {other}"),
                            }
                            if !slot.ends_with("#small") {
                                device_fed.push(slot);
                            }
                        }
                        eprintln!(
                            "jit_prove[{}]: device mem count feed merged {} families",
                            C::LABEL,
                            device_fed.len()
                        );
                    }
                    None => device_fed.clear(),
                }
            }
        }
        C::feed_from_flats(
            &out.sub_flat,
            out.column_length,
            addr_state,
            id_state,
            vi_state,
            &device_fed,
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
                device_fed: &[&'static str],
            ) {
                $module::feed_sub_inputs_from_flat(
                    words, n_rows, addr_state, id_state, vi_state, device_fed,
                );
            }
            fn sub_feed_layout(
            ) -> &'static [(&'static str, usize, &'static str, u32, usize, usize)] {
                $module::SUB_FEED_LAYOUT
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
    type Gen;
    type Claim;
    type IGen;
    fn input_columns(_gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        Err(RecordedInputBuildError::DeviceSeedOnly(Self::LABEL))
    }
    fn device_input_seed(_gen: &Self::Gen) -> Option<RecordedDeviceInputSeed> {
        None
    }
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
        bi::<AddApOpcodeLane>(),
        bi::<MulOpcodeLane>(),
        bi::<MulOpcodeSmallLane>(),
        bi::<RangeCheck252Width27Lane>(),
        bi::<TripleXor32Lane>(),
        bi::<VerifyInstructionLane>(),
        bi::<BlakeCompressOpcodeLane>(),
        bi::<PedersenAggregatorW18Lane>(),
        bi::<BlakeRoundLane>(),
        bi::<PartialEcMulW18Lane>(),
        bi::<PartialEcMulGenericLane>(),
        bi::<Cube252Lane>(),
        bi::<BlakeGRecordedLane>(),
        bi::<Qm31AddMulOpcodeLane>(),
        bi::<BitwiseBuiltinLane>(),
        bi::<RangeCheckBuiltinLane>(),
        bi::<PedersenBuiltinLane>(),
        bi::<PoseidonBuiltinLane>(),
        bi::<PoseidonAggregatorLane>(),
        bi::<PoseidonFullRoundChainLane>(),
        bi::<Poseidon3PartialRoundsChainLane>(),
    ]
}

pub type RecordedSubFeedLayoutEntry = (&'static str, usize, &'static str, u32, usize, usize);

#[derive(Clone, Copy, Debug)]
pub struct RecordedSubFeedLayout {
    pub component: &'static str,
    pub entries: &'static [RecordedSubFeedLayoutEntry],
}

/// Public projection of the same transformer-emitted `SUB_FEED_LAYOUT` facts
/// used by every live recorded lane. The arena planner consumes this registry;
/// no relation offsets are duplicated there.
pub fn all_lane_sub_feed_layouts() -> Vec<RecordedSubFeedLayout> {
    fn op<C: OpcodeLaneSpec>() -> RecordedSubFeedLayout {
        RecordedSubFeedLayout {
            component: C::LABEL,
            entries: C::sub_feed_layout(),
        }
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
        RecordedSubFeedLayout {
            component: AddApOpcodeLane::LABEL,
            entries: add_ap_opcode::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: MulOpcodeLane::LABEL,
            entries: mul_opcode::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: MulOpcodeSmallLane::LABEL,
            entries: mul_opcode_small::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: RangeCheck252Width27Lane::LABEL,
            entries: range_check_252_width_27::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: TripleXor32Lane::LABEL,
            entries: triple_xor_32::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: VerifyInstructionLane::LABEL,
            entries: verify_instruction::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: BlakeCompressOpcodeLane::LABEL,
            entries: blake_compress_opcode::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PedersenAggregatorW18Lane::LABEL,
            entries: pedersen_aggregator_window_bits_18::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: BlakeRoundLane::LABEL,
            entries: blake_round::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PartialEcMulW18Lane::LABEL,
            entries: partial_ec_mul_window_bits_18::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PartialEcMulGenericLane::LABEL,
            entries: partial_ec_mul_generic::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: Cube252Lane::LABEL,
            entries: cube_252::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: BlakeGRecordedLane::LABEL,
            entries: blake_g::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: Qm31AddMulOpcodeLane::LABEL,
            entries: qm_31_add_mul_opcode::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: BitwiseBuiltinLane::LABEL,
            entries: bitwise_builtin::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: RangeCheckBuiltinLane::LABEL,
            entries: range_check_builtin::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PedersenBuiltinLane::LABEL,
            entries: pedersen_builtin::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PoseidonBuiltinLane::LABEL,
            entries: poseidon_builtin::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PoseidonAggregatorLane::LABEL,
            entries: poseidon_aggregator::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: PoseidonFullRoundChainLane::LABEL,
            entries: poseidon_full_round_chain::SUB_FEED_LAYOUT,
        },
        RecordedSubFeedLayout {
            component: Poseidon3PartialRoundsChainLane::LABEL,
            entries: poseidon_3_partial_rounds_chain::SUB_FEED_LAYOUT,
        },
    ]
}

/// In-process identity of the immutable Cairo memory used to build the one
/// DeviceExecutionTables instance shared by every prepared recorded lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionMemoryIdentity {
    pub address_to_id_ptr: usize,
    pub address_to_id_len: usize,
    pub f252_values_ptr: usize,
    pub f252_values_len: usize,
    pub small_values_ptr: usize,
    pub small_values_len: usize,
}

impl ExecutionMemoryIdentity {
    pub fn of(memory: &Memory) -> Self {
        Self {
            address_to_id_ptr: memory.address_to_id.as_ptr() as usize,
            address_to_id_len: memory.address_to_id.len(),
            f252_values_ptr: memory.f252_values.as_ptr() as usize,
            f252_values_len: memory.f252_values.len(),
            small_values_ptr: memory.small_values.as_ptr() as usize,
            small_values_len: memory.small_values.len(),
        }
    }
}

/// Static tables a prepared lane must have registered before capture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecordedTableIdentity {
    pub execution_memory: ExecutionMemoryIdentity,
    /// The exact host-built `PedersenPoints<18>` table; never the quarantined
    /// GPU-generated variant.
    pub host_pedersen_points_18: bool,
}

/// One complete launch input for STWO's `PreparedWitnessGraph`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedWitnessInputPlan {
    pub label: &'static str,
    pub program: stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    pub row_count: usize,
    pub n_real: usize,
    pub input_columns: Vec<Vec<u32>>,
    pub tables: RecordedTableIdentity,
}

/// Pre-witness, non-consuming resident launch plan. Holding the Arc makes the
/// execution-table pointer identity stable through graph preparation/replay.
#[derive(Clone, Debug)]
pub struct RecordedWitnessInputs {
    pub execution_memory: Arc<Memory>,
    pub execution_memory_identity: ExecutionMemoryIdentity,
    pub lanes: Vec<RecordedWitnessInputPlan>,
}

#[derive(Debug)]
pub enum RecordedWitnessInputsError {
    UnsupportedLabels(Vec<&'static str>),
    MissingExecutionMemory(Vec<&'static str>),
    InputBuild(RecordedInputBuildError),
    RecordingShape(&'static str),
}

impl core::fmt::Display for RecordedWitnessInputsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "recorded resident witness inputs rejected: {self:?}")
    }
}

impl std::error::Error for RecordedWitnessInputsError {}

impl From<RecordedInputBuildError> for RecordedWitnessInputsError {
    fn from(value: RecordedInputBuildError) -> Self {
        Self::InputBuild(value)
    }
}

#[derive(Debug)]
pub struct RecordedWitnessInputAttempt {
    pub label: &'static str,
    pub program: stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    pub input_source: RecordedWitnessInputSource,
    pub host_pedersen_points_18: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedDeviceInputSeed {
    pub n_real: usize,
    pub row_count: usize,
    pub scalar_values: Vec<u32>,
}

#[derive(Debug)]
pub enum RecordedWitnessInputSource {
    Host(Result<RecordedInputColumns, RecordedInputBuildError>),
    DeviceSeed(RecordedDeviceInputSeed),
}

fn opcode_input_attempt<C: OpcodeLaneSpec>(
    gen: &C::Gen,
) -> Result<RecordedWitnessInputAttempt, RecordedWitnessInputsError> {
    let recording = C::record();
    validate_recording(
        C::LABEL,
        &recording,
        C::N_TRACE,
        C::N_LOOKUP_WORDS,
        C::N_SUB_WORDS,
    )?;
    let inputs = C::input_columns(gen)
        .and_then(|inputs| normalize_recorded_inputs(C::LABEL, &recording.program, inputs));
    Ok(RecordedWitnessInputAttempt {
        label: C::LABEL,
        program: recording.program,
        input_source: RecordedWitnessInputSource::Host(inputs),
        host_pedersen_points_18: false,
    })
}

fn builtin_input_attempt<C: BuiltinLaneSpec>(
    gen: &C::Gen,
) -> Result<RecordedWitnessInputAttempt, RecordedWitnessInputsError> {
    let recording = C::record();
    validate_recording(
        C::LABEL,
        &recording,
        C::N_TRACE,
        C::N_LOOKUP_WORDS,
        C::N_SUB_WORDS,
    )?;
    let input_source = match C::device_input_seed(gen) {
        Some(seed) => RecordedWitnessInputSource::DeviceSeed(seed),
        None => RecordedWitnessInputSource::Host(
            C::input_columns(gen)
                .and_then(|inputs| normalize_recorded_inputs(C::LABEL, &recording.program, inputs)),
        ),
    };
    Ok(RecordedWitnessInputAttempt {
        label: C::LABEL,
        program: recording.program,
        input_source,
        host_pedersen_points_18: C::NEEDS_PEDERSEN_TABLE,
    })
}

fn validate_recording(
    label: &'static str,
    recording: &RecordingOutput,
    n_trace: usize,
    n_lookup: usize,
    n_sub: usize,
) -> Result<(), RecordedWitnessInputsError> {
    let program = &recording.program;
    if program.label != label
        || program.n_cols as usize != n_trace
        || program.n_lookup_words as usize != n_lookup
        || program.n_sub_words as usize != n_sub
        || !recording.poisoned_cols.is_empty()
        || !recording.poisoned_lookup_words.is_empty()
        || !recording.poisoned_sub_words.is_empty()
    {
        return Err(RecordedWitnessInputsError::RecordingShape(label));
    }
    Ok(())
}

/// Returns whether the generator-to-input dispatcher has an explicit arm for a
/// registry label. The registry parity test below turns future lane additions
/// into a compile-time-local failing gate instead of a runtime host fallback.
pub const RECORDED_INPUT_LABELS: &[&str] = &[
    "add_opcode",
    "assert_eq_opcode",
    "jnz_opcode_taken",
    "add_opcode_small",
    "assert_eq_opcode_imm",
    "assert_eq_opcode_double_deref",
    "call_opcode_abs",
    "call_opcode_rel_imm",
    "jnz_opcode_non_taken",
    "jump_opcode_abs",
    "jump_opcode_double_deref",
    "jump_opcode_rel",
    "jump_opcode_rel_imm",
    "ret_opcode",
    "add_ap_opcode",
    "mul_opcode",
    "mul_opcode_small",
    "range_check_252_width_27",
    "triple_xor_32",
    "verify_instruction",
    "blake_compress_opcode",
    "pedersen_aggregator_window_bits_18",
    "blake_round",
    "partial_ec_mul_window_bits_18",
    "partial_ec_mul_generic",
    "cube_252",
    "blake_g",
    "qm_31_add_mul_opcode",
    "bitwise_builtin",
    "range_check_builtin",
    "pedersen_builtin",
    "poseidon_builtin",
    "poseidon_aggregator",
    "poseidon_full_round_chain",
    "poseidon_3_partial_rounds_chain",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecordedInputGeometry {
    pub enabler_slot: Option<usize>,
    pub iota_slot: Option<usize>,
}

/// Canonical device multiset-compaction contract for generated writers whose
/// host claim generator is a DashMap of tuple -> multiplicity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecordedInputCompactionGeometry {
    pub tuple_words: usize,
    pub key_words: usize,
    pub multiplicity_slot: usize,
}

pub fn recorded_input_compaction_geometry(label: &str) -> Option<RecordedInputCompactionGeometry> {
    Some(match label {
        "verify_instruction" => RecordedInputCompactionGeometry {
            tuple_words: 7,
            key_words: 1,
            multiplicity_slot: 9,
        },
        "pedersen_aggregator_window_bits_18" => RecordedInputCompactionGeometry {
            tuple_words: 3,
            key_words: 2,
            multiplicity_slot: 5,
        },
        "poseidon_aggregator" => RecordedInputCompactionGeometry {
            tuple_words: 6,
            key_words: 3,
            multiplicity_slot: 8,
        },
        _ => return None,
    })
}

pub fn recorded_input_geometry(label: &str) -> Option<RecordedInputGeometry> {
    let casm = RecordedInputGeometry {
        enabler_slot: Some(3),
        iota_slot: None,
    };
    Some(match label {
        "add_opcode"
        | "assert_eq_opcode"
        | "jnz_opcode_taken"
        | "add_opcode_small"
        | "assert_eq_opcode_imm"
        | "assert_eq_opcode_double_deref"
        | "call_opcode_abs"
        | "call_opcode_rel_imm"
        | "jnz_opcode_non_taken"
        | "jump_opcode_abs"
        | "jump_opcode_double_deref"
        | "jump_opcode_rel"
        | "jump_opcode_rel_imm"
        | "ret_opcode"
        | "add_ap_opcode"
        | "mul_opcode"
        | "mul_opcode_small"
        | "qm_31_add_mul_opcode" => casm,
        "blake_compress_opcode" => RecordedInputGeometry {
            enabler_slot: Some(3),
            iota_slot: Some(4),
        },
        "range_check_252_width_27" | "cube_252" => RecordedInputGeometry {
            enabler_slot: Some(10),
            iota_slot: Some(11),
        },
        "triple_xor_32" => RecordedInputGeometry {
            enabler_slot: Some(3),
            iota_slot: Some(4),
        },
        "verify_instruction" => RecordedInputGeometry {
            enabler_slot: Some(7),
            iota_slot: Some(8),
        },
        "pedersen_aggregator_window_bits_18" => RecordedInputGeometry {
            enabler_slot: Some(3),
            iota_slot: Some(4),
        },
        "blake_round" => RecordedInputGeometry {
            enabler_slot: Some(19),
            iota_slot: Some(20),
        },
        "partial_ec_mul_window_bits_18" => RecordedInputGeometry {
            enabler_slot: Some(72),
            iota_slot: Some(73),
        },
        "partial_ec_mul_generic" => RecordedInputGeometry {
            enabler_slot: Some(125),
            iota_slot: Some(126),
        },
        "blake_g" => RecordedInputGeometry {
            enabler_slot: Some(6),
            iota_slot: Some(7),
        },
        "bitwise_builtin" | "range_check_builtin" | "pedersen_builtin" | "poseidon_builtin" => {
            RecordedInputGeometry {
                enabler_slot: Some(1),
                iota_slot: Some(2),
            }
        }
        "poseidon_aggregator" => RecordedInputGeometry {
            enabler_slot: Some(6),
            iota_slot: Some(7),
        },
        "poseidon_full_round_chain" => RecordedInputGeometry {
            enabler_slot: Some(32),
            iota_slot: Some(33),
        },
        "poseidon_3_partial_rounds_chain" => RecordedInputGeometry {
            enabler_slot: Some(42),
            iota_slot: Some(43),
        },
        _ => return None,
    })
}

/// Compact device-seed scalar width for StoredLogSize recorded writers. This
/// is a capture-planning contract, not a readiness label: every listed lane
/// must also return a matching [`RecordedDeviceInputSeed`] at ingest.
pub fn recorded_device_seed_scalar_count(label: &str) -> Option<usize> {
    match label {
        "bitwise_builtin" | "range_check_builtin" | "pedersen_builtin" | "poseidon_builtin" => {
            Some(1)
        }
        _ => None,
    }
}

pub fn is_supported_recorded_input_label(label: &str) -> bool {
    RECORDED_INPUT_LABELS.contains(&label)
}

/// Try to build one lane's canonical host columns without consuming its claim
/// generator. An empty/scalar-remainder builder result is retained in the
/// attempt so gpu-prover can assign exact DeviceEdge/Unresolved provenance.
pub fn recorded_witness_input_attempt(
    generator: &crate::witness::cairo_claim_generator::CairoClaimGenerator,
    label: &'static str,
) -> Result<Option<RecordedWitnessInputAttempt>, RecordedWitnessInputsError> {
    macro_rules! op {
        ($field:ident, $lane:ty) => {
            generator
                .$field
                .as_ref()
                .map(opcode_input_attempt::<$lane>)
                .transpose()?
        };
    }
    macro_rules! bi {
        ($field:ident, $lane:ty) => {
            generator
                .$field
                .as_ref()
                .map(builtin_input_attempt::<$lane>)
                .transpose()?
        };
    }
    Ok(match label {
        "add_opcode" => op!(add_opcode, AddOpcodeLane),
        "assert_eq_opcode" => op!(assert_eq_opcode, AssertEqOpcodeLane),
        "jnz_opcode_taken" => op!(jnz_opcode_taken, JnzOpcodeTakenLane),
        "add_opcode_small" => op!(add_opcode_small, AddOpcodeSmallLane),
        "assert_eq_opcode_imm" => op!(assert_eq_opcode_imm, AssertEqOpcodeImmLane),
        "assert_eq_opcode_double_deref" => {
            op!(assert_eq_opcode_double_deref, AssertEqOpcodeDoubleDerefLane)
        }
        "call_opcode_abs" => op!(call_opcode_abs, CallOpcodeAbsLane),
        "call_opcode_rel_imm" => op!(call_opcode_rel_imm, CallOpcodeRelImmLane),
        "jnz_opcode_non_taken" => op!(jnz_opcode_non_taken, JnzOpcodeNonTakenLane),
        "jump_opcode_abs" => op!(jump_opcode_abs, JumpOpcodeAbsLane),
        "jump_opcode_double_deref" => op!(jump_opcode_double_deref, JumpOpcodeDoubleDerefLane),
        "jump_opcode_rel" => op!(jump_opcode_rel, JumpOpcodeRelLane),
        "jump_opcode_rel_imm" => op!(jump_opcode_rel_imm, JumpOpcodeRelImmLane),
        "ret_opcode" => op!(ret_opcode, RetOpcodeLane),
        "add_ap_opcode" => bi!(add_ap_opcode, AddApOpcodeLane),
        "mul_opcode" => bi!(mul_opcode, MulOpcodeLane),
        "mul_opcode_small" => bi!(mul_opcode_small, MulOpcodeSmallLane),
        "range_check_252_width_27" => {
            bi!(range_check_252_width_27, RangeCheck252Width27Lane)
        }
        "triple_xor_32" => bi!(triple_xor_32, TripleXor32Lane),
        "verify_instruction" => bi!(verify_instruction, VerifyInstructionLane),
        "blake_compress_opcode" => {
            bi!(blake_compress_opcode, BlakeCompressOpcodeLane)
        }
        "pedersen_aggregator_window_bits_18" => bi!(
            pedersen_aggregator_window_bits_18,
            PedersenAggregatorW18Lane
        ),
        "blake_round" => bi!(blake_round, BlakeRoundLane),
        "partial_ec_mul_window_bits_18" => {
            bi!(partial_ec_mul_window_bits_18, PartialEcMulW18Lane)
        }
        "partial_ec_mul_generic" => bi!(partial_ec_mul_generic, PartialEcMulGenericLane),
        "cube_252" => bi!(cube_252, Cube252Lane),
        "blake_g" => bi!(blake_g, BlakeGRecordedLane),
        "qm_31_add_mul_opcode" => bi!(qm_31_add_mul_opcode, Qm31AddMulOpcodeLane),
        "bitwise_builtin" => bi!(bitwise_builtin, BitwiseBuiltinLane),
        "range_check_builtin" => bi!(range_check_builtin, RangeCheckBuiltinLane),
        "pedersen_builtin" => bi!(pedersen_builtin, PedersenBuiltinLane),
        "poseidon_builtin" => bi!(poseidon_builtin, PoseidonBuiltinLane),
        "poseidon_aggregator" => bi!(poseidon_aggregator, PoseidonAggregatorLane),
        "poseidon_full_round_chain" => {
            bi!(poseidon_full_round_chain, PoseidonFullRoundChainLane)
        }
        "poseidon_3_partial_rounds_chain" => bi!(
            poseidon_3_partial_rounds_chain,
            Poseidon3PartialRoundsChainLane
        ),
        _ => None,
    })
}

/// Extract exactly the requested recorded lanes without executing any writer.
/// This host-only convenience fails on unresolved generator-fed columns; the
/// gpu-prover ProofPlan bridge preserves them with explicit provenance instead.
pub fn recorded_witness_inputs(
    generator: &crate::witness::cairo_claim_generator::CairoClaimGenerator,
    required_labels: &[&'static str],
) -> Result<RecordedWitnessInputs, RecordedWitnessInputsError> {
    let mut attempts = Vec::with_capacity(required_labels.len());
    let mut unsupported = Vec::new();
    for &label in required_labels {
        match recorded_witness_input_attempt(generator, label)? {
            Some(attempt) => attempts.push(attempt),
            None => unsupported.push(label),
        }
    }
    unsupported.sort_unstable();
    unsupported.dedup();
    if !unsupported.is_empty() {
        return Err(RecordedWitnessInputsError::UnsupportedLabels(unsupported));
    }

    let memory = generator.jit_memory.clone().ok_or_else(|| {
        RecordedWitnessInputsError::MissingExecutionMemory(required_labels.to_vec())
    })?;
    let execution_memory_identity = ExecutionMemoryIdentity::of(&memory);
    let lanes = attempts
        .into_iter()
        .map(|attempt| {
            let inputs = match attempt.input_source {
                RecordedWitnessInputSource::Host(inputs) => {
                    inputs.map_err(RecordedWitnessInputsError::InputBuild)?
                }
                RecordedWitnessInputSource::DeviceSeed(_) => {
                    return Err(RecordedWitnessInputsError::InputBuild(
                        RecordedInputBuildError::DeviceSeedOnly(attempt.label),
                    ));
                }
            };
            let row_count = inputs.row_count();
            Ok(RecordedWitnessInputPlan {
                label: attempt.label,
                program: attempt.program,
                row_count,
                n_real: inputs.n_real,
                input_columns: inputs.columns,
                tables: RecordedTableIdentity {
                    execution_memory: execution_memory_identity,
                    host_pedersen_points_18: attempt.host_pedersen_points_18,
                },
            })
        })
        .collect::<Result<Vec<_>, RecordedWitnessInputsError>>()?;
    Ok(RecordedWitnessInputs {
        execution_memory: memory,
        execution_memory_identity,
        lanes,
    })
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

fn hostless_certified_edges_enabled() -> bool {
    std::env::var("STWO_CUDA_HOSTLESS_CERTIFIED_EDGES").as_deref() == Ok("1")
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
    /// Runtime table sizes for the memory families (`(rows, small_rows)`);
    /// `None` leaves a runtime-sized family on the host feed path.
    pub sizes: &'a dyn Fn(&'static str) -> Option<(usize, usize)>,
    /// When EVERY relation the component feeds is count-style, the seam sets
    /// this and provides NO host feed: a device-feed failure then fails the
    /// whole lane (`None` → the host WRITER reruns — still exactly-once feeds,
    /// never a silent miss).
    pub require: bool,
}

/// One producer edge selected for device transport. `component` is the canonical
/// schedule node id; `feed_state` is the generated claim-generator parameter whose
/// host feed must be skipped while the edge is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceEdgeRecovery {
    /// Keep the producer's full sub-word D2H so the consumer can rebuild on CPU.
    HostMirror,
    /// The edge is conformance-certified; experimental opt-in omits the mirror.
    FailClosed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceEdgeTarget {
    pub component: &'static str,
    pub feed_state: &'static str,
    recovery: DeviceEdgeRecovery,
}

impl DeviceEdgeTarget {
    pub(crate) const fn recoverable(component: &'static str, feed_state: &'static str) -> Self {
        Self {
            component,
            feed_state,
            recovery: DeviceEdgeRecovery::HostMirror,
        }
    }

    pub(crate) const fn fail_closed(component: &'static str, feed_state: &'static str) -> Self {
        Self {
            component,
            feed_state,
            recovery: DeviceEdgeRecovery::FailClosed,
        }
    }
}

fn active_device_edge(
    target: Option<DeviceEdgeTarget>,
    edges_enabled: bool,
    hostless_certified_edges: bool,
) -> Option<DeviceEdgeTarget> {
    let mut target = edges_enabled.then_some(target).flatten()?;
    if target.recovery == DeviceEdgeRecovery::FailClosed && !hostless_certified_edges {
        target.recovery = DeviceEdgeRecovery::HostMirror;
    }
    Some(target)
}

fn want_host_sub_copy(
    all_count_feed_required: bool,
    configured_edge: bool,
    active_recovery: Option<DeviceEdgeRecovery>,
    shadow: bool,
) -> bool {
    if configured_edge && active_recovery.is_none() {
        return true;
    }
    match active_recovery {
        Some(DeviceEdgeRecovery::FailClosed) => false,
        Some(DeviceEdgeRecovery::HostMirror) => true,
        None => !all_count_feed_required || shadow,
    }
}

fn dispatch_host_sub_feed(
    skip_host_feed: bool,
    sub_flat: &[u32],
    column_length: usize,
    device_fed: &[&'static str],
    feed: impl FnOnce(&[u32], usize, &[&'static str]),
) {
    if !skip_host_feed {
        feed(sub_flat, column_length, device_fed);
    }
}

fn planned_device_edge<C: BuiltinLaneSpec>(
    target: DeviceEdgeTarget,
    layout: &'static [(&'static str, usize, &'static str, u32, usize, usize)],
) -> PlannedDeviceEdge {
    let mut word_base = None;
    let mut words_per_instance = None;
    let mut n_instances = 0usize;
    for &(component, instance, state, _relation, base, words) in layout {
        if state != target.feed_state {
            continue;
        }
        assert_eq!(
            component, target.component,
            "device edge target component disagrees with SUB_FEED_LAYOUT"
        );
        assert_eq!(
            instance, n_instances,
            "device edge instances are not contiguous in SUB_FEED_LAYOUT"
        );
        match (word_base, words_per_instance) {
            (None, None) => {
                word_base = Some(base);
                words_per_instance = Some(words);
            }
            (Some(first), Some(width)) => {
                assert_eq!(
                    words, width,
                    "device edge width varies across SUB_FEED_LAYOUT instances"
                );
                assert_eq!(
                    base,
                    first + n_instances * width,
                    "device edge words are not contiguous in SUB_FEED_LAYOUT"
                );
            }
            _ => unreachable!(),
        }
        n_instances += 1;
    }
    PlannedDeviceEdge {
        producer: C::LABEL,
        consumer: target.component,
        word_base: u32::try_from(word_base.expect("device edge missing from SUB_FEED_LAYOUT"))
            .expect("device edge word base exceeds u32"),
        words_per_instance: u32::try_from(words_per_instance.unwrap())
            .expect("device edge width exceeds u32"),
        n_instances: u32::try_from(n_instances).expect("device edge instance count exceeds u32"),
    }
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
    exec_context: &WitnessExecContext,
    input_cols: &[Vec<u32>],
    n_real: usize,
    mem: &Arc<Memory>,
    device_feed: Option<DeviceFeedPlan<'_>>,
    feed: impl FnOnce(&[u32], usize, &[&'static str]),
) -> Option<(Evals<stwo_backend_cuda::CudaBackend>, C::Claim, C::IGen)> {
    builtin_cuda_write_trace_from::<C>(
        exec_context,
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
    exec_context: &WitnessExecContext,
    inputs: BuiltinInputs<'_>,
    n_real: usize,
    mem: &Arc<Memory>,
    device_feed: Option<DeviceFeedPlan<'_>>,
    stash_edge: Option<DeviceEdgeTarget>,
    feed: impl FnOnce(&[u32], usize, &[&'static str]),
) -> Option<(Evals<stwo_backend_cuda::CudaBackend>, C::Claim, C::IGen)> {
    if !lane_enabled(C::LABEL) || !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        exec_context.host_witness_fallback(
            C::LABEL,
            "recorded builtin witness lane disabled or unavailable",
        );
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
        exec_context.host_witness_fallback(C::LABEL, "recorded builtin shape mismatch");
        return None;
    }
    if C::NEEDS_PEDERSEN_TABLE && !ensure_device_pedersen_table() {
        eprintln!(
            "jit_prove[{}]: host pedersen table registration failed — falling back",
            C::LABEL
        );
        exec_context.host_witness_fallback(C::LABEL, "resident pedersen table unavailable");
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
        exec_context.host_witness_fallback(C::LABEL, "recorded builtin input shape mismatch");
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
    let feed_layout = device_feed.as_ref().map(|plan| plan.layout);
    let active_edge = active_device_edge(
        stash_edge,
        edges_enabled(),
        hostless_certified_edges_enabled(),
    );
    let fail_closed_edge =
        active_edge.is_some_and(|target| target.recovery == DeviceEdgeRecovery::FailClosed);
    // Skip the largest witness D2H when every host sub-word consumer is already
    // closed on device: either an all-count builtin, or one of the certified
    // fail-closed producer edges. Recoverable edges and the env-off rollback keep
    // the old host transport exactly as before.
    let want_host_sub = want_host_sub_copy(
        device_feed.as_ref().is_some_and(|plan| plan.require),
        stash_edge.is_some(),
        active_edge.map(|target| target.recovery),
        std::env::var("STWO_JIT_PROVE_SHADOW").as_deref() == Ok("1"),
    );
    let column_length = match &inputs {
        BuiltinInputs::HostCols(input_cols) => input_cols[0].len(),
        BuiltinInputs::Edge { host_tail, .. } => host_tail[0].len(),
    };
    let destinations = exec_context.take_resident_witness_destination(
        C::LABEL,
        TracePartId::Main,
        C::N_TRACE,
        column_length,
        C::N_LOOKUP_WORDS * column_length,
        C::N_SUB_WORDS * column_length,
    );
    let resident_launch = destinations.is_some();
    let launched = match (&inputs, destinations) {
        (BuiltinInputs::HostCols(input_cols), Some(destinations)) => {
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove_into(
                C::LABEL,
                input_cols,
                tables,
                want_host_lookup,
                want_host_sub,
                destinations,
            )
        }
        (BuiltinInputs::HostCols(input_cols), None) => {
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
                C::LABEL,
                input_cols,
                tables,
                want_host_lookup,
                want_host_sub,
            )
        }
        (
            BuiltinInputs::Edge {
                device_cols,
                host_tail,
            },
            Some(destinations),
        ) => {
            let ptrs: Vec<*const u32> = device_cols.iter().map(|c| c.device_ptr).collect();
            let n = host_tail[0].len();
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_mixed_into(
                C::LABEL,
                &ptrs,
                host_tail,
                n,
                tables,
                want_host_lookup,
                want_host_sub,
                destinations,
            )
        }
        (
            BuiltinInputs::Edge {
                device_cols,
                host_tail,
            },
            None,
        ) => {
            let ptrs: Vec<*const u32> = device_cols.iter().map(|c| c.device_ptr).collect();
            let n = host_tail[0].len();
            stwo_backend_cuda::exec_tables::launch_recorded_builtin_mixed(
                C::LABEL,
                &ptrs,
                host_tail,
                n,
                tables,
                want_host_lookup,
                want_host_sub,
            )
        }
    };
    let Some((cols, lookup_dev, lookup_flat, sub_dev, sub_flat)) = launched else {
        exec_context.host_witness_fallback(C::LABEL, "recorded builtin launch failed");
        return None;
    };
    if resident_launch {
        exec_context.record_resident_witness_launch();
    }
    let log_size = column_length.ilog2();
    if cols.len() != C::N_TRACE {
        eprintln!(
            "jit_prove[{}]: kernel returned {} columns, expected {} — falling back",
            C::LABEL,
            cols.len(),
            C::N_TRACE
        );
        exec_context.host_witness_fallback(C::LABEL, "recorded builtin output shape mismatch");
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
        exec_context.insert_device_lookup(C::LABEL, lookup_dev, column_length, n_real);
        C::igen_from_flats(log_size, n_real, &[], 0)
    } else {
        C::igen_from_flats(log_size, n_real, &lookup_flat, column_length)
    };

    // Device-DAG count feed: count-style relations feed on device from the
    // resident sub buffer; the host closure then skips them. Any failure feeds
    // everything on host (empty skip set) — never a double feed, never a miss.
    let mut device_fed: Vec<&'static str> = Vec::new();
    if let Some(plan) = device_feed {
        let (descs, lut_slots, counts_slots, sizes) =
            crate::witness::device_feed::build_feed_descriptors_sized(
                plan.layout,
                crate::witness::device_feed::COUNT_RELATIONS,
                plan.sizes,
            );
        if !descs.is_empty() {
            let luts: Vec<Vec<u32>> = lut_slots.iter().map(|s| (plan.lut_for)(s)).collect();
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
                    if fail_closed_edge {
                        panic!(
                            "jit_prove[{}]: certified device edge requires producer-side device \
                             count feeds; host recovery is disabled",
                            C::LABEL
                        );
                    }
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
    // B3 producer role: stash the device sub buffer and skip the consumer's host
    // input feed. Only the still-migrating edge keeps a recovery mirror; certified
    // edges assert that every sub-word state is device-routed and fail closed.
    if let Some(target) = active_edge {
        let layout = feed_layout.expect("device edge requires a generated feed layout");
        let edge = planned_device_edge::<C>(target, layout);
        let host_flat = match target.recovery {
            DeviceEdgeRecovery::HostMirror => Some(sub_flat.clone()),
            DeviceEdgeRecovery::FailClosed => None,
        };
        exec_context.insert_edge(edge, sub_dev, host_flat, column_length);
        device_fed.push(target.feed_state);
        if target.recovery == DeviceEdgeRecovery::FailClosed {
            assert!(
                sub_flat.is_empty(),
                "certified device edge unexpectedly materialized a host mirror"
            );
            for &(_, _, state, ..) in layout {
                assert!(
                    device_fed.contains(&state),
                    "certified device edge left {state} on the host sub-feed path"
                );
            }
        }
        eprintln!(
            "jit_prove[{}]: stashed device edge for {} ({})",
            C::LABEL,
            target.component,
            match target.recovery {
                DeviceEdgeRecovery::HostMirror => "host recovery mirror retained",
                DeviceEdgeRecovery::FailClosed => "host recovery mirror skipped",
            }
        );
    }
    dispatch_host_sub_feed(
        fail_closed_edge,
        &sub_flat,
        column_length,
        &device_fed,
        feed,
    );
    Some((trace, claim, igen))
}

use crate::witness::components::{blake_round, pedersen_aggregator_window_bits_18};

/// Register the HOST-BUILT `PEDERSEN_TABLE_18` on device (borrowed mode) — the
/// deduce lane's only permitted table source: the oracle falsified the
/// GPU-generated table (144/256 rows, run 20260705T113615Z). Idempotent per
/// process; `false` (stub build / upload failure) → callers fall back to host.
pub fn ensure_device_pedersen_table() -> bool {
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
    type Gen = pedersen_aggregator_window_bits_18::ClaimGenerator;
    type Claim = cairo_air::components::pedersen_aggregator_window_bits_18::Claim;
    type IGen = pedersen_aggregator_window_bits_18::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        use std::sync::atomic::Ordering;
        let mut rows = gen
            .mults
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect::<Vec<_>>();
        rows.sort_by_key(|(input, _)| input.0);
        let n_real = rows.len();
        if n_real == 0 {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let size = std::cmp::max(n_real.next_power_of_two(), N_LANES);
        let first = rows[0].0;
        rows.resize(size, (first, 0));
        let columns = vec![
            rows.iter().map(|(input, _)| input.0[0].0).collect(),
            rows.iter().map(|(input, _)| input.0[1].0).collect(),
            rows.iter().map(|(input, _)| input.1 .0).collect(),
            (0..size).map(|row| u32::from(row < n_real)).collect(),
            (0..size).map(|row| row as u32).collect(),
            rows.iter().map(|(_, mult)| *mult).collect(),
        ];
        Ok(RecordedInputColumns { n_real, columns })
    }

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
    type Gen = blake_round::ClaimGenerator;
    type Claim = cairo_air::components::blake_round::Claim;
    type IGen = blake_round::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed: Vec<blake_round::PackedInputType> = gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 21];
        for input in &padded {
            let chain = input.0.to_array();
            let round = input.1.to_array();
            let message_pointer = input.2 .1.to_array();
            for lane in 0..N_LANES {
                columns[0].push(chain[lane].0);
                columns[1].push(round[lane].0);
                for (word, value) in input.2 .0.iter().enumerate() {
                    columns[2 + word].push(value.simd.as_array()[lane]);
                }
                columns[18].push(message_pointer[lane].0);
            }
        }
        columns[19] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[20] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

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

/// Recorded Graph-A counterpart of the former hand-written native writer.
pub struct BlakeGRecordedLane;
impl BuiltinLaneSpec for BlakeGRecordedLane {
    const LABEL: &'static str = "blake_g";
    const N_TRACE: usize = cairo_air::components::blake_g::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = blake_g::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = blake_g::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = blake_g::ClaimGenerator;
    type Claim = cairo_air::components::blake_g::Claim;
    type IGen = blake_g::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed = gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 8];
        for input in &padded {
            for lane in 0..N_LANES {
                for (word, value) in input.iter().enumerate() {
                    columns[word].push(value.simd.as_array()[lane]);
                }
            }
        }
        columns[6] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[7] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

    fn record() -> RecordingOutput {
        blake_g::record_blake_g()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        blake_g::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        blake_g::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        blake_g::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

/// Fully recorded opcode that is absent from the legacy opcode trait only
/// because its host writer has an additional range-check feed parameter.
pub struct Qm31AddMulOpcodeLane;
impl BuiltinLaneSpec for Qm31AddMulOpcodeLane {
    const LABEL: &'static str = "qm_31_add_mul_opcode";
    const N_TRACE: usize = cairo_air::components::qm_31_add_mul_opcode::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = qm_31_add_mul_opcode::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = qm_31_add_mul_opcode::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = qm_31_add_mul_opcode::ClaimGenerator;
    type Claim = cairo_air::components::qm_31_add_mul_opcode::Claim;
    type IGen = qm_31_add_mul_opcode::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_slot_columns(Self::LABEL, &gen.inputs)
    }
    fn record() -> RecordingOutput {
        qm_31_add_mul_opcode::record_qm_31_add_mul_opcode()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        qm_31_add_mul_opcode::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        qm_31_add_mul_opcode::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        qm_31_add_mul_opcode::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

fn stored_builtin_input_seed(log_size: u32, segment_start: u32) -> Option<RecordedDeviceInputSeed> {
    let row_count = 1usize
        .checked_shl(log_size)
        .filter(|&rows| rows >= N_LANES)?;
    Some(RecordedDeviceInputSeed {
        n_real: row_count,
        row_count,
        scalar_values: vec![segment_start],
    })
}

macro_rules! stored_builtin_lane {
    ($lane:ident, $module:ident, $segment:ident, $record:ident) => {
        pub struct $lane;
        impl BuiltinLaneSpec for $lane {
            const LABEL: &'static str = stringify!($module);
            const N_TRACE: usize = cairo_air::components::$module::N_TRACE_COLUMNS;
            const N_LOOKUP_WORDS: usize = $module::N_LOOKUP_WORDS;
            const N_SUB_WORDS: usize = $module::N_SUB_INPUT_WORDS;
            const NEEDS_PEDERSEN_TABLE: bool = false;
            type Gen = $module::ClaimGenerator;
            type Claim = cairo_air::components::$module::Claim;
            type IGen = $module::InteractionClaimGenerator;

            fn device_input_seed(gen: &Self::Gen) -> Option<RecordedDeviceInputSeed> {
                stored_builtin_input_seed(gen.log_size, gen.$segment)
            }
            fn record() -> RecordingOutput {
                $module::$record()
            }
            fn claim(log_size: u32) -> Self::Claim {
                Self::Claim { log_size }
            }
            fn lookup_fields() -> &'static [(&'static str, usize)] {
                $module::JIT_LOOKUP_FIELDS
            }
            fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
                $module::JIT_LOGUP_DESCS
            }
            fn igen_from_flats(
                log_size: u32,
                _n_real: usize,
                words: &[u32],
                n_rows: usize,
            ) -> Self::IGen {
                $module::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
            }
        }
    };
}

stored_builtin_lane!(
    BitwiseBuiltinLane,
    bitwise_builtin,
    bitwise_builtin_segment_start,
    record_bitwise_builtin
);
stored_builtin_lane!(
    RangeCheckBuiltinLane,
    range_check_builtin,
    range_check_builtin_segment_start,
    record_range_check_builtin
);
stored_builtin_lane!(
    PedersenBuiltinLane,
    pedersen_builtin,
    pedersen_builtin_segment_start,
    record_pedersen_builtin
);
stored_builtin_lane!(
    PoseidonBuiltinLane,
    poseidon_builtin,
    poseidon_builtin_segment_start,
    record_poseidon_builtin
);

macro_rules! poseidon_edge_lane {
    ($lane:ident, $module:ident, $record:ident) => {
        pub struct $lane;
        impl BuiltinLaneSpec for $lane {
            const LABEL: &'static str = stringify!($module);
            const N_TRACE: usize = cairo_air::components::$module::N_TRACE_COLUMNS;
            const N_LOOKUP_WORDS: usize = $module::N_LOOKUP_WORDS;
            const N_SUB_WORDS: usize = $module::N_SUB_INPUT_WORDS;
            const NEEDS_PEDERSEN_TABLE: bool = false;
            type Gen = $module::ClaimGenerator;
            type Claim = cairo_air::components::$module::Claim;
            type IGen = $module::InteractionClaimGenerator;

            fn record() -> RecordingOutput {
                $module::$record()
            }
            fn claim(log_size: u32) -> Self::Claim {
                Self::Claim { log_size }
            }
            fn lookup_fields() -> &'static [(&'static str, usize)] {
                $module::JIT_LOOKUP_FIELDS
            }
            fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
                $module::JIT_LOGUP_DESCS
            }
            fn igen_from_flats(
                log_size: u32,
                _n_real: usize,
                words: &[u32],
                n_rows: usize,
            ) -> Self::IGen {
                $module::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
            }
        }
    };
}

poseidon_edge_lane!(
    PoseidonAggregatorLane,
    poseidon_aggregator,
    record_poseidon_aggregator
);
poseidon_edge_lane!(
    PoseidonFullRoundChainLane,
    poseidon_full_round_chain,
    record_poseidon_full_round_chain
);
poseidon_edge_lane!(
    Poseidon3PartialRoundsChainLane,
    poseidon_3_partial_rounds_chain,
    record_poseidon_3_partial_rounds_chain
);

use crate::witness::components::{cube_252, partial_ec_mul_generic, partial_ec_mul_window_bits_18};

pub struct PartialEcMulW18Lane;
impl BuiltinLaneSpec for PartialEcMulW18Lane {
    const LABEL: &'static str = "partial_ec_mul_window_bits_18";
    const N_TRACE: usize = 297;
    const N_LOOKUP_WORDS: usize = partial_ec_mul_window_bits_18::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = partial_ec_mul_window_bits_18::N_SUB_INPUT_WORDS;
    // The recorded body's points-table + EC-round deduces read the device table.
    const NEEDS_PEDERSEN_TABLE: bool = true;
    type Gen = partial_ec_mul_window_bits_18::ClaimGenerator;
    type Claim = cairo_air::components::partial_ec_mul_window_bits_18::Claim;
    type IGen = partial_ec_mul_window_bits_18::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed: Vec<partial_ec_mul_window_bits_18::PackedInputType> =
            gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 74];
        for input in &padded {
            let a0 = input.0.to_array();
            let a1 = input.1.to_array();
            for lane in 0..N_LANES {
                columns[0].push(a0[lane].0);
                columns[1].push(a1[lane].0);
                for (window, value) in input.2 .0.iter().enumerate() {
                    columns[2 + window].push(value.to_array()[lane].0);
                }
                for (felt, value) in input.2 .1.iter().enumerate() {
                    for limb in 0..28 {
                        columns[16 + felt * 28 + limb].push(value.get_m31(limb).to_array()[lane].0);
                    }
                }
            }
        }
        columns[72] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[73] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

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
    // Felt-only fp256 deduces no longer declare the Pedersen globals in their
    // CUmodule; only kinds 2/3 carry that table ABI.
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = partial_ec_mul_generic::ClaimGenerator;
    type Claim = cairo_air::components::partial_ec_mul_generic::Claim;
    type IGen = partial_ec_mul_generic::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed: Vec<partial_ec_mul_generic::PackedInputType> =
            gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 127];
        for input in &padded {
            let a0 = input.0.to_array();
            let a1 = input.1.to_array();
            let tail = input.2 .3.to_array();
            for lane in 0..N_LANES {
                columns[0].push(a0[lane].0);
                columns[1].push(a1[lane].0);
                for limb in 0..10 {
                    columns[2 + limb].push(input.2 .0.get_m31(limb).to_array()[lane].0);
                }
                for (felt, value) in input.2 .1.iter().enumerate() {
                    for limb in 0..28 {
                        columns[12 + felt * 28 + limb].push(value.get_m31(limb).to_array()[lane].0);
                    }
                }
                for (felt, value) in input.2 .2.iter().enumerate() {
                    for limb in 0..28 {
                        columns[68 + felt * 28 + limb].push(value.get_m31(limb).to_array()[lane].0);
                    }
                }
                columns[124].push(tail[lane].0);
            }
        }
        columns[125] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[126] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

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
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = cube_252::ClaimGenerator;
    type Claim = cairo_air::components::cube_252::Claim;
    type IGen = cube_252::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed: Vec<cube_252::PackedInputType> = gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 12];
        for input in &padded {
            for lane in 0..N_LANES {
                for limb in 0..10 {
                    columns[limb].push(input.get_m31(limb).to_array()[lane].0);
                }
            }
        }
        columns[10] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[11] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

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

// These emitted writers have flat inputs/downstream states outside the original
// Casm-only `OpcodeLaneSpec` contract. Reuse the destination-backed builtin lane
// instead of growing a second launcher.
pub struct BlakeCompressOpcodeLane;
impl BuiltinLaneSpec for BlakeCompressOpcodeLane {
    const LABEL: &'static str = "blake_compress_opcode";
    const N_TRACE: usize = cairo_air::components::blake_compress_opcode::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = blake_compress_opcode::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = blake_compress_opcode::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = blake_compress_opcode::ClaimGenerator;
    type Claim = cairo_air::components::blake_compress_opcode::Claim;
    type IGen = blake_compress_opcode::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_iota_slot_columns(Self::LABEL, &gen.inputs)
    }

    fn record() -> RecordingOutput {
        blake_compress_opcode::record_blake_compress_opcode()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        blake_compress_opcode::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        blake_compress_opcode::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        blake_compress_opcode::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct AddApOpcodeLane;
impl BuiltinLaneSpec for AddApOpcodeLane {
    const LABEL: &'static str = "add_ap_opcode";
    const N_TRACE: usize = cairo_air::components::add_ap_opcode::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = add_ap_opcode::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = add_ap_opcode::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = add_ap_opcode::ClaimGenerator;
    type Claim = cairo_air::components::add_ap_opcode::Claim;
    type IGen = add_ap_opcode::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_slot_columns(Self::LABEL, &gen.inputs)
    }

    fn record() -> RecordingOutput {
        add_ap_opcode::record_add_ap_opcode()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        add_ap_opcode::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        add_ap_opcode::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        add_ap_opcode::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct MulOpcodeLane;
impl BuiltinLaneSpec for MulOpcodeLane {
    const LABEL: &'static str = "mul_opcode";
    const N_TRACE: usize = cairo_air::components::mul_opcode::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = mul_opcode::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = mul_opcode::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = mul_opcode::ClaimGenerator;
    type Claim = cairo_air::components::mul_opcode::Claim;
    type IGen = mul_opcode::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_slot_columns(Self::LABEL, &gen.inputs)
    }

    fn record() -> RecordingOutput {
        mul_opcode::record_mul_opcode()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        mul_opcode::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        mul_opcode::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        mul_opcode::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct MulOpcodeSmallLane;
impl BuiltinLaneSpec for MulOpcodeSmallLane {
    const LABEL: &'static str = "mul_opcode_small";
    const N_TRACE: usize = cairo_air::components::mul_opcode_small::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = mul_opcode_small::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = mul_opcode_small::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = mul_opcode_small::ClaimGenerator;
    type Claim = cairo_air::components::mul_opcode_small::Claim;
    type IGen = mul_opcode_small::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        casm_slot_columns(Self::LABEL, &gen.inputs)
    }

    fn record() -> RecordingOutput {
        mul_opcode_small::record_mul_opcode_small()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        mul_opcode_small::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        mul_opcode_small::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        mul_opcode_small::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct RangeCheck252Width27Lane;
impl BuiltinLaneSpec for RangeCheck252Width27Lane {
    const LABEL: &'static str = "range_check_252_width_27";
    const N_TRACE: usize = cairo_air::components::range_check_252_width_27::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = range_check_252_width_27::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = range_check_252_width_27::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = range_check_252_width_27::ClaimGenerator;
    type Claim = cairo_air::components::range_check_252_width_27::Claim;
    type IGen = range_check_252_width_27::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed = gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 12];
        for input in &padded {
            for lane in 0..N_LANES {
                for word in 0..10 {
                    columns[word].push(input.get_m31(word).to_array()[lane].0);
                }
            }
        }
        columns[10] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[11] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

    fn record() -> RecordingOutput {
        range_check_252_width_27::record_range_check_252_width_27()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        range_check_252_width_27::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        range_check_252_width_27::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        range_check_252_width_27::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct TripleXor32Lane;
impl BuiltinLaneSpec for TripleXor32Lane {
    const LABEL: &'static str = "triple_xor_32";
    const N_TRACE: usize = cairo_air::components::triple_xor_32::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = triple_xor_32::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = triple_xor_32::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = triple_xor_32::ClaimGenerator;
    type Claim = cairo_air::components::triple_xor_32::Claim;
    type IGen = triple_xor_32::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        let packed = gen.packed_inputs.lock().unwrap().clone();
        if !gen.remainder_inputs.lock().unwrap().is_empty() {
            return Err(RecordedInputBuildError::ScalarRemainder(Self::LABEL));
        }
        if packed.is_empty() {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let n_real = packed.len() * N_LANES;
        let packed_size = packed.len().next_power_of_two();
        let size = packed_size * N_LANES;
        let mut padded = packed;
        padded.resize(packed_size, padded[0]);
        let mut columns = vec![Vec::with_capacity(size); 5];
        for input in &padded {
            for lane in 0..N_LANES {
                for word in 0..3 {
                    columns[word].push(input[word].simd.as_array()[lane]);
                }
            }
        }
        columns[3] = (0..size).map(|row| u32::from(row < n_real)).collect();
        columns[4] = (0..size).map(|row| row as u32).collect();
        Ok(RecordedInputColumns { n_real, columns })
    }

    fn record() -> RecordingOutput {
        triple_xor_32::record_triple_xor_32()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        triple_xor_32::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        triple_xor_32::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        triple_xor_32::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

pub struct VerifyInstructionLane;
impl BuiltinLaneSpec for VerifyInstructionLane {
    const LABEL: &'static str = "verify_instruction";
    const N_TRACE: usize = cairo_air::components::verify_instruction::N_TRACE_COLUMNS;
    const N_LOOKUP_WORDS: usize = verify_instruction::N_LOOKUP_WORDS;
    const N_SUB_WORDS: usize = verify_instruction::N_SUB_INPUT_WORDS;
    const NEEDS_PEDERSEN_TABLE: bool = false;
    type Gen = verify_instruction::ClaimGenerator;
    type Claim = cairo_air::components::verify_instruction::Claim;
    type IGen = verify_instruction::InteractionClaimGenerator;

    fn input_columns(gen: &Self::Gen) -> Result<RecordedInputColumns, RecordedInputBuildError> {
        use std::sync::atomic::Ordering;
        let mut rows = gen
            .mults
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect::<Vec<_>>();
        rows.sort_by_key(|(input, _)| input.0);
        let n_real = rows.len();
        if n_real == 0 {
            return Err(RecordedInputBuildError::EmptyInputs(Self::LABEL));
        }
        let size = std::cmp::max(n_real.next_power_of_two(), N_LANES);
        let first = rows[0].0;
        rows.resize(size, (first, 0));
        let mut columns = vec![Vec::with_capacity(size); 10];
        for (input, mult) in rows {
            columns[0].push(input.0 .0);
            columns[1].push(input.1[0].0);
            columns[2].push(input.1[1].0);
            columns[3].push(input.1[2].0);
            columns[4].push(input.2[0].0);
            columns[5].push(input.2[1].0);
            columns[6].push(input.3 .0);
            columns[7].push(0);
            columns[8].push(0);
            columns[9].push(mult);
        }
        Ok(RecordedInputColumns { n_real, columns })
    }

    fn record() -> RecordingOutput {
        verify_instruction::record_verify_instruction()
    }
    fn claim(log_size: u32) -> Self::Claim {
        Self::Claim { log_size }
    }
    fn lookup_fields() -> &'static [(&'static str, usize)] {
        verify_instruction::JIT_LOOKUP_FIELDS
    }
    fn logup_descs() -> &'static [crate::witness::logup_descs::LogupDescFact] {
        verify_instruction::JIT_LOGUP_DESCS
    }
    fn igen_from_flats(log_size: u32, _n_real: usize, words: &[u32], n_rows: usize) -> Self::IGen {
        verify_instruction::interaction_gen_from_flat_lookup_words(log_size, words, n_rows)
    }
}

/// Device write for the ALL-COUNT builtins (w18 / generic / cube_252): every
/// downstream relation is count-style, so the device feed is REQUIRED and no
/// host feed exists — any unavailability falls back to the host writer. The
/// caller supplies the packed inputs already padded by the host preamble rule
/// (first-packed-row replication) flattened to slot columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn all_count_builtin_write_trace<C: BuiltinLaneSpec>(
    exec_context: &WitnessExecContext,
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
        // Memory families stay host-fed at this seam until sized.
        sizes: &|_| None,
        require: true,
    };
    builtin_cuda_write_trace::<C>(
        exec_context,
        cols,
        n_real,
        mem,
        Some(plan),
        |_sub, _n, fed| {
            debug_assert!(
                !fed.is_empty(),
                "require-mode feed reached with nothing fed"
            );
        },
    )
}

use crate::witness::components::{
    range_check_11, range_check_18, range_check_20, range_check_4_3, range_check_7_2_5,
    range_check_9_9, verify_bitwise_xor_8,
};

/// Backend seam for the `cube_252` base-trace write (poseidon-family fp256):
/// Simd = the generated writer verbatim; Cuda = the witness-JIT lane (12 slot
/// columns from the W27 input; ALL relations count-style — device feed
/// REQUIRED, host-writer fallback on any unavailability).
pub trait Cube252Witness: FromSimdColumns {
    fn write_trace(
        exec_context: &WitnessExecContext,
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
        _exec_context: &WitnessExecContext,
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
        exec_context: &WitnessExecContext,
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
            if let Ok(inputs) = Cube252Lane::input_columns(&gen) {
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
                    exec_context,
                    &inputs.columns,
                    inputs.n_real,
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

/// Backend seam for emitted flat-input writers that do not fit the original
/// three-state opcode lane. SIMD remains the generated writer; CUDA reuses the
/// destination-backed `BuiltinLaneSpec` launcher.
pub trait RecordedFlatWitness: FromSimdColumns {
    #[allow(clippy::too_many_arguments)]
    fn write_blake_compress_opcode(
        exec_context: &WitnessExecContext,
        gen: blake_compress_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        round: &blake_round::ClaimGenerator,
        triple_xor: &triple_xor_32::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::blake_compress_opcode::Claim,
        blake_compress_opcode::InteractionClaimGenerator,
    );

    #[allow(clippy::too_many_arguments)]
    fn write_add_ap_opcode(
        exec_context: &WitnessExecContext,
        gen: add_ap_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::add_ap_opcode::Claim,
        add_ap_opcode::InteractionClaimGenerator,
    );

    #[allow(clippy::too_many_arguments)]
    fn write_mul_opcode(
        exec_context: &WitnessExecContext,
        gen: mul_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc20: &range_check_20::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode::Claim,
        mul_opcode::InteractionClaimGenerator,
    );

    #[allow(clippy::too_many_arguments)]
    fn write_mul_opcode_small(
        exec_context: &WitnessExecContext,
        gen: mul_opcode_small::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode_small::Claim,
        mul_opcode_small::InteractionClaimGenerator,
    );

    fn write_range_check_252_width_27(
        exec_context: &WitnessExecContext,
        gen: range_check_252_width_27::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::range_check_252_width_27::Claim,
        range_check_252_width_27::InteractionClaimGenerator,
    );

    fn write_triple_xor_32(
        exec_context: &WitnessExecContext,
        gen: triple_xor_32::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::triple_xor_32::Claim,
        triple_xor_32::InteractionClaimGenerator,
    );

    #[allow(clippy::too_many_arguments)]
    fn write_verify_instruction(
        exec_context: &WitnessExecContext,
        gen: verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        rc43: &range_check_4_3::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::verify_instruction::Claim,
        verify_instruction::InteractionClaimGenerator,
    );
}

impl RecordedFlatWitness for SimdBackend {
    fn write_blake_compress_opcode(
        _exec_context: &WitnessExecContext,
        gen: blake_compress_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        round: &blake_round::ClaimGenerator,
        triple_xor: &triple_xor_32::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::blake_compress_opcode::Claim,
        blake_compress_opcode::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc725, xor8, round, triple_xor);
        (trace.to_evals(), claim, igen)
    }

    fn write_add_ap_opcode(
        _exec_context: &WitnessExecContext,
        gen: add_ap_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::add_ap_opcode::Claim,
        add_ap_opcode::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc18, rc11);
        (trace.to_evals(), claim, igen)
    }

    fn write_mul_opcode(
        _exec_context: &WitnessExecContext,
        gen: mul_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc20: &range_check_20::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode::Claim,
        mul_opcode::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc20);
        (trace.to_evals(), claim, igen)
    }

    fn write_mul_opcode_small(
        _exec_context: &WitnessExecContext,
        gen: mul_opcode_small::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode_small::Claim,
        mul_opcode_small::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc11);
        (trace.to_evals(), claim, igen)
    }

    fn write_range_check_252_width_27(
        _exec_context: &WitnessExecContext,
        gen: range_check_252_width_27::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::range_check_252_width_27::Claim,
        range_check_252_width_27::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(rc99, rc18);
        (trace.to_evals(), claim, igen)
    }

    fn write_triple_xor_32(
        _exec_context: &WitnessExecContext,
        gen: triple_xor_32::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::triple_xor_32::Claim,
        triple_xor_32::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(xor8);
        (trace.to_evals(), claim, igen)
    }

    fn write_verify_instruction(
        _exec_context: &WitnessExecContext,
        gen: verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        rc43: &range_check_4_3::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        _mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::verify_instruction::Claim,
        verify_instruction::InteractionClaimGenerator,
    ) {
        let (trace, claim, igen) = gen.write_trace(rc725, rc43, addr, ids);
        (trace.to_evals(), claim, igen)
    }
}

fn casm_slot_columns(
    label: &'static str,
    inputs: &[CasmState],
) -> Result<RecordedInputColumns, RecordedInputBuildError> {
    let n_real = inputs.len();
    if n_real == 0 {
        return Err(RecordedInputBuildError::EmptyInputs(label));
    }
    let size = std::cmp::max(n_real.next_power_of_two(), N_LANES);
    let mut padded = inputs.to_vec();
    padded.resize(size, padded[0]);
    let mut columns = vec![Vec::with_capacity(size); 4];
    for row in padded {
        columns[0].push(row.pc.0);
        columns[1].push(row.ap.0);
        columns[2].push(row.fp.0);
    }
    columns[3] = (0..size).map(|row| u32::from(row < n_real)).collect();
    Ok(RecordedInputColumns { n_real, columns })
}

fn casm_iota_slot_columns(
    label: &'static str,
    inputs: &[CasmState],
) -> Result<RecordedInputColumns, RecordedInputBuildError> {
    let mut columns = casm_slot_columns(label, inputs)?;
    let rows = columns.row_count();
    columns
        .columns
        .push((0..rows).map(|row| row as u32).collect());
    Ok(columns)
}

#[inline]
fn flat_m31(words: &[u32], rows: usize, word: usize, row: usize) -> BaseField {
    BaseField::from_u32_unchecked(words[word * rows + row])
}

fn feed_flat_tuple<const N: usize, S>(
    words: &[u32],
    rows: usize,
    base: usize,
    relation: usize,
    state: &S,
) where
    S: crate::witness::utils::AddInputs<InputType = [BaseField; N]>,
{
    for row in 0..rows {
        let input = std::array::from_fn(|i| flat_m31(words, rows, base + i, row));
        crate::witness::utils::AddInputs::add_input(state, &input, relation);
    }
}

fn feed_flat_scalar<S>(words: &[u32], rows: usize, base: usize, relation: usize, state: &S)
where
    S: crate::witness::utils::AddInputs<InputType = BaseField>,
{
    for row in 0..rows {
        crate::witness::utils::AddInputs::add_input(
            state,
            &flat_m31(words, rows, base, row),
            relation,
        );
    }
}

fn feed_flat_verify_instruction(
    words: &[u32],
    rows: usize,
    base: usize,
    state: &verify_instruction::ClaimGenerator,
) {
    for row in 0..rows {
        let w = |i| flat_m31(words, rows, base + i, row);
        let input = (w(0), [w(1), w(2), w(3)], [w(4), w(5)], w(6));
        crate::witness::utils::AddInputs::add_input(state, &input, 0);
    }
}

fn feed_flat_blake_round(
    words: &[u32],
    rows: usize,
    base: usize,
    relation: usize,
    state: &blake_round::ClaimGenerator,
) {
    use stwo::prover::backend::simd::m31::PackedM31;
    use stwo_cairo_common::prover_types::simd::PackedUInt32;

    assert_eq!(rows % N_LANES, 0, "blake_round feed must stay packed");
    let n_vec = rows / N_LANES;
    let m31 = |word: usize, vi: usize| {
        PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(words[(base + word) * rows + vi * N_LANES + lane])
        }))
    };
    let raw = |word: usize, vi: usize| PackedUInt32 {
        simd: std::simd::Simd::from_array(std::array::from_fn(|lane| {
            words[(base + word) * rows + vi * N_LANES + lane]
        })),
    };
    let inputs: Vec<blake_round::PackedInputType> = (0..n_vec)
        .map(|vi| {
            (
                m31(0, vi),
                m31(1, vi),
                (std::array::from_fn(|i| raw(2 + i, vi)), m31(18, vi)),
            )
        })
        .collect();
    crate::witness::utils::AddInputs::add_packed_inputs(state, &inputs, relation);
}

/// Fail-closed recovery for the `blake_compress_opcode -> blake_round` device
/// edge: rebuild the exact ten input instances per producer row from the host
/// mirror retained beside the device buffer. Geometry is read from the emitted
/// `SUB_FEED_LAYOUT`, never duplicated as hand-maintained offsets.
pub(crate) fn feed_blake_round_inputs_from_blake_compress_flat(
    words: &[u32],
    rows: usize,
    state: &blake_round::ClaimGenerator,
) {
    assert_eq!(
        words.len(),
        blake_compress_opcode::N_SUB_INPUT_WORDS * rows,
        "blake_compress_opcode recovery extent drift"
    );
    let mut instances = 0usize;
    for &(field, instance, state_name, relation, base, width) in
        blake_compress_opcode::SUB_FEED_LAYOUT
    {
        if state_name != "blake_round_state" {
            continue;
        }
        assert_eq!(field, "blake_round");
        assert_eq!(instance, instances, "blake_round edge instance gap");
        assert_eq!(width, 19, "blake_round edge word width drift");
        feed_flat_blake_round(words, rows, base, relation as usize, state);
        instances += 1;
    }
    assert_eq!(instances, 10, "blake_round edge instance-count drift");
}

fn feed_flat_triple_xor_32(
    words: &[u32],
    rows: usize,
    base: usize,
    relation: usize,
    state: &triple_xor_32::ClaimGenerator,
) {
    use stwo_cairo_common::prover_types::simd::PackedUInt32;

    assert_eq!(rows % N_LANES, 0, "triple_xor_32 feed must stay packed");
    let inputs: Vec<triple_xor_32::PackedInputType> = (0..rows / N_LANES)
        .map(|vi| {
            std::array::from_fn(|i| PackedUInt32 {
                simd: std::simd::Simd::from_array(std::array::from_fn(|lane| {
                    words[(base + i) * rows + vi * N_LANES + lane]
                })),
            })
        })
        .collect();
    crate::witness::utils::AddInputs::add_packed_inputs(state, &inputs, relation);
}

#[allow(clippy::too_many_arguments)]
fn feed_blake_compress_sub(
    words: &[u32],
    rows: usize,
    fed: &[&'static str],
    addr: &memory_address_to_id::ClaimGenerator,
    ids: &memory_id_to_big::ClaimGenerator,
    vi: &verify_instruction::ClaimGenerator,
    rc725: &range_check_7_2_5::ClaimGenerator,
    xor8: &verify_bitwise_xor_8::ClaimGenerator,
    round: &blake_round::ClaimGenerator,
    triple_xor: &triple_xor_32::ClaimGenerator,
) {
    assert_eq!(
        words.len(),
        blake_compress_opcode::N_SUB_INPUT_WORDS * rows,
        "blake_compress_opcode sub-word extent drift"
    );
    for &(field, _instance, state, relation, base, width) in blake_compress_opcode::SUB_FEED_LAYOUT
    {
        if fed.contains(&state) {
            continue;
        }
        match state {
            "verify_instruction_state" => {
                assert_eq!((field, width), ("verify_instruction", 7));
                feed_flat_verify_instruction(words, rows, base, vi);
            }
            "memory_address_to_id_state" => {
                assert_eq!((field, width), ("memory_address_to_id", 1));
                feed_flat_scalar(words, rows, base, relation as usize, addr);
            }
            "memory_id_to_big_state" => {
                assert_eq!((field, width), ("memory_id_to_big", 1));
                feed_flat_scalar(words, rows, base, relation as usize, ids);
            }
            "range_check_7_2_5_state" => {
                assert_eq!((field, width), ("range_check_7_2_5", 3));
                feed_flat_tuple::<3, _>(words, rows, base, relation as usize, rc725);
            }
            "verify_bitwise_xor_8_state" => {
                assert_eq!((field, width), ("verify_bitwise_xor_8", 3));
                feed_flat_tuple::<3, _>(words, rows, base, relation as usize, xor8);
            }
            "blake_round_state" => {
                assert_eq!((field, width), ("blake_round", 19));
                feed_flat_blake_round(words, rows, base, relation as usize, round);
            }
            "triple_xor_32_state" => {
                assert_eq!((field, width), ("triple_xor_32", 3));
                feed_flat_triple_xor_32(words, rows, base, relation as usize, triple_xor);
            }
            other => panic!("unrouted blake_compress_opcode sub feed {other}"),
        }
    }
}

fn memory_sizes<'a>(
    addr: &'a memory_address_to_id::ClaimGenerator,
    ids: &'a memory_id_to_big::ClaimGenerator,
) -> impl Fn(&'static str) -> Option<(usize, usize)> + 'a {
    move |family| match family {
        "memory_address_to_id_state" => Some((addr.table_size(), 0)),
        "memory_id_to_big_state" => Some((ids.big_table_size(), ids.small_table_size())),
        _ => None,
    }
}

fn feed_add_ap_sub(
    words: &[u32],
    rows: usize,
    fed: &[&'static str],
    addr: &memory_address_to_id::ClaimGenerator,
    ids: &memory_id_to_big::ClaimGenerator,
    vi: &verify_instruction::ClaimGenerator,
    rc18: &range_check_18::ClaimGenerator,
    rc11: &range_check_11::ClaimGenerator,
) {
    for &(_, _, state, relation, base, width) in add_ap_opcode::SUB_FEED_LAYOUT {
        if fed.contains(&state) {
            continue;
        }
        match state {
            "verify_instruction_state" => feed_flat_verify_instruction(words, rows, base, vi),
            "memory_address_to_id_state" => {
                feed_flat_scalar(words, rows, base, relation as usize, addr)
            }
            "memory_id_to_big_state" => feed_flat_scalar(words, rows, base, relation as usize, ids),
            "range_check_18_state" => {
                feed_flat_tuple::<1, _>(words, rows, base, relation as usize, rc18)
            }
            "range_check_11_state" => {
                feed_flat_tuple::<1, _>(words, rows, base, relation as usize, rc11)
            }
            other => panic!("unrouted add_ap_opcode sub feed {other} width {width}"),
        }
    }
}

fn feed_mul_sub(
    words: &[u32],
    rows: usize,
    fed: &[&'static str],
    addr: &memory_address_to_id::ClaimGenerator,
    ids: &memory_id_to_big::ClaimGenerator,
    vi: &verify_instruction::ClaimGenerator,
    rc20: &range_check_20::ClaimGenerator,
) {
    for &(_, _, state, relation, base, width) in mul_opcode::SUB_FEED_LAYOUT {
        if fed.contains(&state) {
            continue;
        }
        match state {
            "verify_instruction_state" => feed_flat_verify_instruction(words, rows, base, vi),
            "memory_address_to_id_state" => {
                feed_flat_scalar(words, rows, base, relation as usize, addr)
            }
            "memory_id_to_big_state" => feed_flat_scalar(words, rows, base, relation as usize, ids),
            "range_check_20_state" => {
                feed_flat_tuple::<1, _>(words, rows, base, relation as usize, rc20)
            }
            other => panic!("unrouted mul_opcode sub feed {other} width {width}"),
        }
    }
}

fn feed_mul_small_sub(
    words: &[u32],
    rows: usize,
    fed: &[&'static str],
    addr: &memory_address_to_id::ClaimGenerator,
    ids: &memory_id_to_big::ClaimGenerator,
    vi: &verify_instruction::ClaimGenerator,
    rc11: &range_check_11::ClaimGenerator,
) {
    for &(_, _, state, relation, base, width) in mul_opcode_small::SUB_FEED_LAYOUT {
        if fed.contains(&state) {
            continue;
        }
        match state {
            "verify_instruction_state" => feed_flat_verify_instruction(words, rows, base, vi),
            "memory_address_to_id_state" => {
                feed_flat_scalar(words, rows, base, relation as usize, addr)
            }
            "memory_id_to_big_state" => feed_flat_scalar(words, rows, base, relation as usize, ids),
            "range_check_11_state" => {
                feed_flat_tuple::<1, _>(words, rows, base, relation as usize, rc11)
            }
            other => panic!("unrouted mul_opcode_small sub feed {other} width {width}"),
        }
    }
}

impl RecordedFlatWitness for stwo_backend_cuda::CudaBackend {
    fn write_blake_compress_opcode(
        exec_context: &WitnessExecContext,
        gen: blake_compress_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        round: &blake_round::ClaimGenerator,
        triple_xor: &triple_xor_32::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::blake_compress_opcode::Claim,
        blake_compress_opcode::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            let inputs = BlakeCompressOpcodeLane::input_columns(&gen)
                .expect("present blake_compress_opcode has non-empty canonical inputs");
            let lut = |family: &'static str| match family {
                "range_check_7_2_5_state" => rc725.input_to_row_lut(),
                "verify_bitwise_xor_8_state" => xor8.input_to_row_lut(),
                other => panic!("unexpected blake_compress_opcode LUT family {other}"),
            };
            let merge = |family: &'static str, counts: &[u32]| match family {
                "memory_address_to_id_state" => addr.add_count_tables(counts),
                "memory_id_to_big_state" => ids.add_big_count_tables(counts),
                "memory_id_to_big_state#small" => ids.add_small_count_tables(counts),
                "range_check_7_2_5_state" => rc725.add_count_tables(counts),
                "verify_bitwise_xor_8_state" => xor8.add_count_tables(counts),
                other => panic!("unexpected blake_compress_opcode count family {other}"),
            };
            let sizes = memory_sizes(addr, ids);
            let plan = DeviceFeedPlan {
                layout: blake_compress_opcode::SUB_FEED_LAYOUT,
                lut_for: &lut,
                merge: &merge,
                sizes: &sizes,
                require: false,
            };
            if let Some(out) = builtin_cuda_write_trace_from::<BlakeCompressOpcodeLane>(
                exec_context,
                BuiltinInputs::HostCols(&inputs.columns),
                inputs.n_real,
                mem,
                Some(plan),
                Some(DeviceEdgeTarget::recoverable(
                    "blake_round",
                    "blake_round_state",
                )),
                |sub, rows, fed| {
                    feed_blake_compress_sub(
                        sub, rows, fed, addr, ids, vi, rc725, xor8, round, triple_xor,
                    )
                },
            ) {
                return out;
            }
        } else {
            exec_context.host_witness_fallback(
                BlakeCompressOpcodeLane::LABEL,
                "missing device execution memory",
            );
        }
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc725, xor8, round, triple_xor);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_add_ap_opcode(
        exec_context: &WitnessExecContext,
        gen: add_ap_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::add_ap_opcode::Claim,
        add_ap_opcode::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            let inputs = AddApOpcodeLane::input_columns(&gen)
                .expect("present add_ap_opcode has non-empty canonical inputs");
            let lut = |family| panic!("unexpected add_ap_opcode LUT family {family}");
            let merge = |family: &'static str, counts: &[u32]| match family {
                "memory_address_to_id_state" => addr.add_count_tables(counts),
                "memory_id_to_big_state" => ids.add_big_count_tables(counts),
                "memory_id_to_big_state#small" => ids.add_small_count_tables(counts),
                "range_check_18_state" => rc18.add_count_tables(counts),
                "range_check_11_state" => rc11.add_count_tables(counts),
                other => panic!("unexpected add_ap_opcode count family {other}"),
            };
            let sizes = memory_sizes(addr, ids);
            let plan = DeviceFeedPlan {
                layout: add_ap_opcode::SUB_FEED_LAYOUT,
                lut_for: &lut,
                merge: &merge,
                sizes: &sizes,
                require: false,
            };
            if let Some(out) = builtin_cuda_write_trace::<AddApOpcodeLane>(
                exec_context,
                &inputs.columns,
                inputs.n_real,
                mem,
                Some(plan),
                |sub, rows, fed| feed_add_ap_sub(sub, rows, fed, addr, ids, vi, rc18, rc11),
            ) {
                return out;
            }
        } else {
            exec_context
                .host_witness_fallback(AddApOpcodeLane::LABEL, "missing device execution memory");
        }
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc18, rc11);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_mul_opcode(
        exec_context: &WitnessExecContext,
        gen: mul_opcode::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc20: &range_check_20::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode::Claim,
        mul_opcode::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            let inputs = MulOpcodeLane::input_columns(&gen)
                .expect("present mul_opcode has non-empty canonical inputs");
            let lut = |family| panic!("unexpected mul_opcode LUT family {family}");
            let merge = |family: &'static str, counts: &[u32]| match family {
                "memory_address_to_id_state" => addr.add_count_tables(counts),
                "memory_id_to_big_state" => ids.add_big_count_tables(counts),
                "memory_id_to_big_state#small" => ids.add_small_count_tables(counts),
                "range_check_20_state" => rc20.add_count_tables(counts),
                other => panic!("unexpected mul_opcode count family {other}"),
            };
            let sizes = memory_sizes(addr, ids);
            let plan = DeviceFeedPlan {
                layout: mul_opcode::SUB_FEED_LAYOUT,
                lut_for: &lut,
                merge: &merge,
                sizes: &sizes,
                require: false,
            };
            if let Some(out) = builtin_cuda_write_trace::<MulOpcodeLane>(
                exec_context,
                &inputs.columns,
                inputs.n_real,
                mem,
                Some(plan),
                |sub, rows, fed| feed_mul_sub(sub, rows, fed, addr, ids, vi, rc20),
            ) {
                return out;
            }
        } else {
            exec_context
                .host_witness_fallback(MulOpcodeLane::LABEL, "missing device execution memory");
        }
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc20);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_mul_opcode_small(
        exec_context: &WitnessExecContext,
        gen: mul_opcode_small::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        vi: &verify_instruction::ClaimGenerator,
        rc11: &range_check_11::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::mul_opcode_small::Claim,
        mul_opcode_small::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            let inputs = MulOpcodeSmallLane::input_columns(&gen)
                .expect("present mul_opcode_small has non-empty canonical inputs");
            let lut = |family| panic!("unexpected mul_opcode_small LUT family {family}");
            let merge = |family: &'static str, counts: &[u32]| match family {
                "memory_address_to_id_state" => addr.add_count_tables(counts),
                "memory_id_to_big_state" => ids.add_big_count_tables(counts),
                "memory_id_to_big_state#small" => ids.add_small_count_tables(counts),
                "range_check_11_state" => rc11.add_count_tables(counts),
                other => panic!("unexpected mul_opcode_small count family {other}"),
            };
            let sizes = memory_sizes(addr, ids);
            let plan = DeviceFeedPlan {
                layout: mul_opcode_small::SUB_FEED_LAYOUT,
                lut_for: &lut,
                merge: &merge,
                sizes: &sizes,
                require: false,
            };
            if let Some(out) = builtin_cuda_write_trace::<MulOpcodeSmallLane>(
                exec_context,
                &inputs.columns,
                inputs.n_real,
                mem,
                Some(plan),
                |sub, rows, fed| feed_mul_small_sub(sub, rows, fed, addr, ids, vi, rc11),
            ) {
                return out;
            }
        } else {
            exec_context.host_witness_fallback(
                MulOpcodeSmallLane::LABEL,
                "missing device execution memory",
            );
        }
        let (trace, claim, igen) = gen.write_trace(addr, ids, vi, rc11);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_range_check_252_width_27(
        exec_context: &WitnessExecContext,
        gen: range_check_252_width_27::ClaimGenerator,
        rc99: &range_check_9_9::ClaimGenerator,
        rc18: &range_check_18::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::range_check_252_width_27::Claim,
        range_check_252_width_27::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            if let Ok(inputs) = RangeCheck252Width27Lane::input_columns(&gen) {
                let lut = |family: &'static str| match family {
                    "range_check_9_9_state" => rc99.input_to_row_lut(),
                    other => panic!("unexpected rc252 LUT family {other}"),
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_9_9_state" => rc99.add_count_tables(counts),
                    "range_check_18_state" => rc18.add_count_tables(counts),
                    other => panic!("unexpected rc252 count family {other}"),
                };
                let sizes = |_| None;
                let plan = DeviceFeedPlan {
                    layout: range_check_252_width_27::SUB_FEED_LAYOUT,
                    lut_for: &lut,
                    merge: &merge,
                    sizes: &sizes,
                    require: false,
                };
                if let Some(out) = builtin_cuda_write_trace::<RangeCheck252Width27Lane>(
                    exec_context,
                    &inputs.columns,
                    inputs.n_real,
                    mem,
                    Some(plan),
                    |sub, rows, fed| {
                        for &(_, _, state, relation, base, width) in
                            range_check_252_width_27::SUB_FEED_LAYOUT
                        {
                            if fed.contains(&state) {
                                continue;
                            }
                            match state {
                                "range_check_9_9_state" => feed_flat_tuple::<2, _>(
                                    sub,
                                    rows,
                                    base,
                                    relation as usize,
                                    rc99,
                                ),
                                "range_check_18_state" => feed_flat_tuple::<1, _>(
                                    sub,
                                    rows,
                                    base,
                                    relation as usize,
                                    rc18,
                                ),
                                other => panic!("unrouted rc252 sub feed {other} width {width}"),
                            }
                        }
                    },
                ) {
                    return out;
                }
            } else {
                exec_context.host_witness_fallback(
                    RangeCheck252Width27Lane::LABEL,
                    "recorded rc252 input shape unavailable",
                );
            }
        } else {
            exec_context.host_witness_fallback(
                RangeCheck252Width27Lane::LABEL,
                "missing device execution memory",
            );
        }
        let (trace, claim, igen) = gen.write_trace(rc99, rc18);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_triple_xor_32(
        exec_context: &WitnessExecContext,
        gen: triple_xor_32::ClaimGenerator,
        xor8: &verify_bitwise_xor_8::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::triple_xor_32::Claim,
        triple_xor_32::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            if let Ok(inputs) = TripleXor32Lane::input_columns(&gen) {
                if let Some(out) = builtin_cuda_write_trace::<TripleXor32Lane>(
                    exec_context,
                    &inputs.columns,
                    inputs.n_real,
                    mem,
                    None,
                    |sub, rows, _| {
                        for &(_, _, state, relation, base, width) in triple_xor_32::SUB_FEED_LAYOUT
                        {
                            match state {
                                "verify_bitwise_xor_8_state" => feed_flat_tuple::<3, _>(
                                    sub,
                                    rows,
                                    base,
                                    relation as usize,
                                    xor8,
                                ),
                                other => {
                                    panic!("unrouted triple_xor_32 sub feed {other} width {width}")
                                }
                            }
                        }
                    },
                ) {
                    return out;
                }
            } else {
                exec_context.host_witness_fallback(
                    TripleXor32Lane::LABEL,
                    "recorded triple-xor input shape unavailable",
                );
            }
        } else {
            exec_context
                .host_witness_fallback(TripleXor32Lane::LABEL, "missing device execution memory");
        }
        let (trace, claim, igen) = gen.write_trace(xor8);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }

    fn write_verify_instruction(
        exec_context: &WitnessExecContext,
        gen: verify_instruction::ClaimGenerator,
        rc725: &range_check_7_2_5::ClaimGenerator,
        rc43: &range_check_4_3::ClaimGenerator,
        addr: &memory_address_to_id::ClaimGenerator,
        ids: &memory_id_to_big::ClaimGenerator,
        mem: Option<&Arc<Memory>>,
    ) -> (
        Evals<Self>,
        cairo_air::components::verify_instruction::Claim,
        verify_instruction::InteractionClaimGenerator,
    ) {
        if let Some(mem) = mem {
            if let Ok(inputs) = VerifyInstructionLane::input_columns(&gen) {
                let lut = |family: &'static str| match family {
                    "range_check_7_2_5_state" => rc725.input_to_row_lut(),
                    other => panic!("unexpected verify_instruction LUT family {other}"),
                };
                let merge = |family: &'static str, counts: &[u32]| match family {
                    "range_check_7_2_5_state" => rc725.add_count_tables(counts),
                    "memory_address_to_id_state" => addr.add_count_tables(counts),
                    "memory_id_to_big_state" => ids.add_big_count_tables(counts),
                    "memory_id_to_big_state#small" => ids.add_small_count_tables(counts),
                    other => panic!("unexpected verify_instruction count family {other}"),
                };
                let sizes = memory_sizes(addr, ids);
                let plan = DeviceFeedPlan {
                    layout: verify_instruction::SUB_FEED_LAYOUT,
                    lut_for: &lut,
                    merge: &merge,
                    sizes: &sizes,
                    require: false,
                };
                if let Some(out) = builtin_cuda_write_trace::<VerifyInstructionLane>(
                    exec_context,
                    &inputs.columns,
                    inputs.n_real,
                    mem,
                    Some(plan),
                    |sub, rows, fed| {
                        for &(_, _, state, relation, base, width) in
                            verify_instruction::SUB_FEED_LAYOUT
                        {
                            if fed.contains(&state) {
                                continue;
                            }
                            match state {
                                "range_check_7_2_5_state" => feed_flat_tuple::<3, _>(
                                    sub,
                                    rows,
                                    base,
                                    relation as usize,
                                    rc725,
                                ),
                                "range_check_4_3_state" => feed_flat_tuple::<2, _>(
                                    sub,
                                    rows,
                                    base,
                                    relation as usize,
                                    rc43,
                                ),
                                "memory_address_to_id_state" => {
                                    feed_flat_scalar(sub, rows, base, relation as usize, addr)
                                }
                                "memory_id_to_big_state" => {
                                    feed_flat_scalar(sub, rows, base, relation as usize, ids)
                                }
                                other => panic!(
                                    "unrouted verify_instruction sub feed {other} width {width}"
                                ),
                            }
                        }
                    },
                ) {
                    return out;
                }
            } else {
                exec_context.host_witness_fallback(
                    VerifyInstructionLane::LABEL,
                    "recorded verify-instruction input shape unavailable",
                );
            }
        } else {
            exec_context.host_witness_fallback(
                VerifyInstructionLane::LABEL,
                "missing device execution memory",
            );
        }
        let (trace, claim, igen) = gen.write_trace(rc725, rc43, addr, ids);
        (Self::from_simd_evals(trace.to_evals()), claim, igen)
    }
}

#[cfg(test)]
mod emitted_lane_tests {
    use super::*;

    #[test]
    fn certified_edge_defaults_to_mirror_and_hostless_requires_opt_in() {
        let certified = Some(DeviceEdgeTarget::fail_closed("consumer", "consumer_state"));
        let default = active_device_edge(certified, true, false).unwrap();
        assert_eq!(default.recovery, DeviceEdgeRecovery::HostMirror);
        assert!(want_host_sub_copy(
            false,
            true,
            Some(default.recovery),
            false
        ));

        let experimental = active_device_edge(certified, true, true).unwrap();
        assert_eq!(experimental.recovery, DeviceEdgeRecovery::FailClosed);
        assert!(!want_host_sub_copy(
            false,
            true,
            Some(experimental.recovery),
            false
        ));
        assert!(!want_host_sub_copy(
            false,
            true,
            Some(experimental.recovery),
            true
        ));

        let rollback = active_device_edge(certified, false, true);
        assert!(rollback.is_none());
        assert!(want_host_sub_copy(false, true, None, false));
        assert!(want_host_sub_copy(true, true, None, false));

        let recoverable = active_device_edge(
            Some(DeviceEdgeTarget::recoverable("consumer", "consumer_state")),
            true,
            true,
        )
        .unwrap();
        assert!(want_host_sub_copy(
            false,
            true,
            Some(recoverable.recovery),
            false
        ));
    }

    #[test]
    fn certified_edge_layouts_leave_no_other_host_sub_feed() {
        let assert_closed = |layout: &[(&str, usize, &str, u32, usize, usize)], edge_state| {
            for &(_, _, state, ..) in layout {
                assert!(
                    state == edge_state
                        || crate::witness::device_feed::COUNT_RELATIONS
                            .iter()
                            .any(|relation| relation.state_param == state),
                    "certified edge still needs host feed state {state}"
                );
            }
        };
        assert_closed(blake_round::SUB_FEED_LAYOUT, "blake_g_state");
        assert_closed(
            pedersen_aggregator_window_bits_18::SUB_FEED_LAYOUT,
            "partial_ec_mul_window_bits_18_state",
        );
    }

    #[test]
    fn hostless_edge_bypasses_generated_length_guard_but_rollback_does_not() {
        use std::cell::Cell;

        let host_calls = Cell::new(0);
        let generated_feed = |sub_flat: &[u32], _rows, _fed: &[&'static str]| {
            host_calls.set(host_calls.get() + 1);
            assert_eq!(sub_flat.len(), 4, "generated sub layout drift");
        };
        dispatch_host_sub_feed(true, &[], 16, &["consumer_state"], generated_feed);
        assert_eq!(host_calls.get(), 0);

        dispatch_host_sub_feed(false, &[1, 2, 3, 4], 16, &[], generated_feed);
        dispatch_host_sub_feed(
            false,
            &[1, 2, 3, 4],
            16,
            &["consumer_state"],
            generated_feed,
        );
        assert_eq!(host_calls.get(), 2);
    }

    fn assert_complete<C: BuiltinLaneSpec>() {
        let recording = C::record();
        assert!(
            recording.poisoned_cols.is_empty(),
            "{}: poisoned trace",
            C::LABEL
        );
        assert!(
            recording.poisoned_lookup_words.is_empty(),
            "{}: poisoned lookup",
            C::LABEL
        );
        assert!(
            recording.poisoned_sub_words.is_empty(),
            "{}: poisoned sub inputs",
            C::LABEL
        );
        assert_eq!(
            recording.program.n_cols as usize,
            C::N_TRACE,
            "{}: trace",
            C::LABEL
        );
        assert_eq!(
            recording.program.n_lookup_words as usize,
            C::N_LOOKUP_WORDS,
            "{}: lookup",
            C::LABEL
        );
        assert_eq!(
            recording.program.n_sub_words as usize,
            C::N_SUB_WORDS,
            "{}: sub inputs",
            C::LABEL
        );
        assert_eq!(
            C::lookup_fields()
                .iter()
                .map(|(_, width)| width)
                .sum::<usize>(),
            C::N_LOOKUP_WORDS,
            "{}: lookup accessor",
            C::LABEL
        );
    }

    #[test]
    fn newly_wired_recordings_are_complete_and_registered() {
        assert_complete::<AddApOpcodeLane>();
        assert_complete::<MulOpcodeLane>();
        assert_complete::<MulOpcodeSmallLane>();
        assert_complete::<RangeCheck252Width27Lane>();
        assert_complete::<TripleXor32Lane>();
        assert_complete::<VerifyInstructionLane>();
        assert_complete::<BlakeCompressOpcodeLane>();
        assert_complete::<BlakeGRecordedLane>();
        assert_complete::<Qm31AddMulOpcodeLane>();
        let labels = [
            AddApOpcodeLane::LABEL,
            MulOpcodeLane::LABEL,
            MulOpcodeSmallLane::LABEL,
            RangeCheck252Width27Lane::LABEL,
            TripleXor32Lane::LABEL,
            VerifyInstructionLane::LABEL,
            BlakeCompressOpcodeLane::LABEL,
            BlakeGRecordedLane::LABEL,
            Qm31AddMulOpcodeLane::LABEL,
        ];
        let registry = all_lane_recordings();
        for label in labels {
            let (_, program) = registry
                .iter()
                .find(|(registered, _)| *registered == label)
                .unwrap_or_else(|| panic!("missing recorded lane {label}"));
            assert!(program.n_cols > 0, "{label}: empty trace");
            assert!(
                program.n_lookup_words > 0,
                "{label}: empty lookup recording"
            );
            assert!(program.n_sub_words > 0, "{label}: empty sub recording");
        }
    }

    #[test]
    fn blake_compress_recording_and_round_edge_are_complete_generated_facts() {
        let recording = BlakeCompressOpcodeLane::record();
        assert!(
            recording.poison_ops.is_empty(),
            "blake_compress_opcode left unsupported recorder operations: {:?}",
            recording.poison_ops
        );
        assert_eq!(recording.program.n_inputs, 5);
        assert_eq!(recording.program.n_cols, 174);
        assert_eq!(recording.program.n_lookup_words, 906);
        assert_eq!(recording.program.n_sub_words, 324);
        assert_eq!(recording.program.n_mult_tables, 0);
        assert!(
            stwo_backend_cuda::jit_witness::codegen::compile_witness_to_cuda_source(
                &recording.program
            )
            .is_some(),
            "blake_compress_opcode AOT source generation rejected the recording"
        );

        let edge = planned_device_edge::<BlakeCompressOpcodeLane>(
            DeviceEdgeTarget::recoverable("blake_round", "blake_round_state"),
            blake_compress_opcode::SUB_FEED_LAYOUT,
        );
        assert_eq!(edge.producer, "blake_compress_opcode");
        assert_eq!(edge.consumer, "blake_round");
        assert_eq!(edge.word_base, 110);
        assert_eq!(edge.words_per_instance, 19);
        assert_eq!(edge.n_instances, 10);
    }

    #[test]
    fn resident_input_dispatch_exactly_covers_the_recording_registry() {
        let registry = all_lane_recordings()
            .into_iter()
            .map(|(label, _)| label)
            .collect::<std::collections::BTreeSet<_>>();
        let dispatch = RECORDED_INPUT_LABELS
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(dispatch, registry);
        assert!(dispatch
            .iter()
            .all(|label| is_supported_recorded_input_label(label)));
        assert!(dispatch
            .iter()
            .all(|label| recorded_input_geometry(label).is_some()));
        assert!(!is_supported_recorded_input_label("not_a_recorded_lane"));
    }

    #[test]
    fn opcode_input_builder_pins_recorder_slots_padding_and_enabler() {
        use stwo_cairo_common::prover_types::cpu::M31;

        let states = vec![
            CasmState {
                pc: M31::from_u32_unchecked(1),
                ap: M31::from_u32_unchecked(2),
                fp: M31::from_u32_unchecked(3),
            },
            CasmState {
                pc: M31::from_u32_unchecked(4),
                ap: M31::from_u32_unchecked(5),
                fp: M31::from_u32_unchecked(6),
            },
            CasmState {
                pc: M31::from_u32_unchecked(7),
                ap: M31::from_u32_unchecked(8),
                fp: M31::from_u32_unchecked(9),
            },
        ];
        let generator = add_opcode::ClaimGenerator::new(states);
        let recording = AddOpcodeLane::record();
        let inputs = normalize_recorded_inputs(
            AddOpcodeLane::LABEL,
            &recording.program,
            AddOpcodeLane::input_columns(&generator).unwrap(),
        )
        .unwrap();
        assert_eq!(inputs.n_real, 3);
        assert_eq!(inputs.columns.len(), recording.program.n_inputs as usize);
        assert_eq!(inputs.row_count(), N_LANES);
        assert_eq!(&inputs.columns[0][..3], &[1, 4, 7]);
        assert!(inputs.columns[0][3..].iter().all(|&value| value == 1));
        assert_eq!(&inputs.columns[3][..3], &[1, 1, 1]);
        assert!(inputs.columns[3][3..].iter().all(|&value| value == 0));
    }

    #[test]
    fn resident_extraction_reports_exact_unsupported_labels_before_memory() {
        let generator = crate::witness::cairo_claim_generator::CairoClaimGenerator::default();
        assert!(matches!(
            recorded_witness_inputs(&generator, &["not_a_recorded_lane", "add_opcode"]),
            Err(RecordedWitnessInputsError::UnsupportedLabels(labels))
                if labels == vec!["add_opcode", "not_a_recorded_lane"]
        ));
    }
}
