//! The pilot gates: permanent byte-equality regression fences for the three
//! witness-genericize pilot components (`add_opcode`, `assert_eq_opcode`,
//! `jnz_opcode_taken`). All three marked blocks are TOOL-emitted
//! (`tools/witness_genericize`, re-runnable + idempotent); these tests are the gate that
//! makes that transformation safe, and must stay green across every upstream
//! regeneration + transformer re-run.
//!
//! Gate (a) — SIMD byte-equality: the generic `WitnessEval`-routed writer
//! (`write_trace_generic_simd`, instantiated on [`SimdWitnessEval`](super::simd)) must
//! reproduce the original monomorphic `write_trace_simd` **byte-for-byte** on a REAL
//! fixture: every committed trace column, every `LookupData` array, and every
//! `SubComponentInputs` scalar — zero tolerance.
//!
//! Gate (b) — recording/interpreter equality: the recording produced by running the SAME
//! generic row body on [`RecordingWitnessEval`](super::recording) is executed by
//! workstream A's reference interpreter (`stwo_backend_cuda::jit_witness::interp`) with
//! a `TableOracle` backed by the real host memory states; every non-poisoned committed
//! column and non-poisoned lookup word must equal the host writer's output exactly.
//!
//! Fixture recipe (reusable): deserialize
//! `test_data/test_prove_verify_all_opcode_components/prover_input.json` into a
//! [`ProverInput`] with `serde_json`, then populate a [`CairoClaimGenerator`] via
//! `fill_components` restricted to the pilot + the states it reads
//! (`memory_address_to_id`, `memory_id_to_big`, `verify_instruction`). The SIMD writers
//! are PURE reads of those states (no `add_inputs` draining), so a single shared state
//! set feeds original + generic + interpreter runs with no cross-contamination.

use std::path::PathBuf;
use std::sync::Arc;

use indexmap::IndexSet;
use stwo_backend_cuda::jit_witness::interp::interpret_row;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_adapter::ProverInput;

use super::{TABLE_ADDR_TO_ID, TABLE_ID_TO_BIG};
use crate::witness::cairo_claim_generator::CairoClaimGenerator;
use crate::witness::components::{
    add_opcode, assert_eq_opcode, blake_round_sigma, jnz_opcode_taken, memory_address_to_id,
    memory_id_to_big,
};
use crate::witness::prelude::*;

/// Deserialize the fixture and populate `components` on a fresh `CairoClaimGenerator`.
fn fill_fixture(components: &[&str]) -> CairoClaimGenerator {
    fill_fixture_with_memory(components).0
}

/// [`fill_fixture`], additionally returning the raw memory tables (addr→id ids,
/// f252 values, small values) that the DEVICE execution tables upload from — the
/// claim generator consumes the `Memory` itself.
#[allow(clippy::type_complexity)]
fn fill_fixture_with_memory(
    components: &[&str],
) -> (
    CairoClaimGenerator,
    Vec<u32>,
    Vec<[u32; 8]>,
    Vec<u128>,
    Arc<Memory>,
) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/test_prove_verify_all_opcode_components/prover_input.json");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    let input: ProverInput = serde_json::from_str(&json).expect("deserialize ProverInput");

    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let addr_ids: Vec<u32> = memory.address_to_id.iter().map(|e| e.0).collect();
    let f252_values = memory.f252_values.clone();
    let small_values = memory.small_values.clone();

    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in components {
        set.insert(c);
    }

    // None of the components used here consume the preprocessed trace; any variant is fine.
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    let memory = Arc::new(memory);
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        memory.clone(),
        preprocessed_trace,
    );
    (cg, addr_ids, f252_values, small_values, memory)
}

/// Byte-compare two `[Vec<PackedM31>]` field bundles lane-for-lane (via `to_array`), with
/// a descriptive failure locator.
/// Byte-compare two raw-lane field bundles (`Simd<u32, 16>` words — the sub-input
/// flat transport, which carries both canonical M31 words and full-32-bit u32 words).
fn assert_raw_field_bundles_eq(
    a: &[Vec<Simd<u32, N_LANES>>],
    b: &[Vec<Simd<u32, N_LANES>>],
    what: &str,
) {
    assert_eq!(a.len(), b.len(), "{what}: field count differs");
    for (fi, (fa, fb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(fa.len(), fb.len(), "{what}: field {fi} length differs");
        for (i, (x, y)) in fa.iter().zip(fb.iter()).enumerate() {
            assert_eq!(x, y, "{what}: field {fi} word {i} differs");
        }
    }
}

fn assert_packed_field_bundles_eq(a: &[Vec<PackedM31>], b: &[Vec<PackedM31>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: field count differs");
    for (fi, (fa, fb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(fa.len(), fb.len(), "{what}: field {fi} length differs");
        for (i, (x, y)) in fa.iter().zip(fb.iter()).enumerate() {
            assert_eq!(
                x.to_array(),
                y.to_array(),
                "{what}: field {fi} element {i} differs"
            );
        }
    }
}

/// Pack a pilot component's `CasmState` inputs exactly as its `write_trace` does,
/// returning `(unpacked_padded_inputs, packed_inputs, n_rows)`.
fn pack_pilot_inputs(mut inputs: Vec<CasmState>) -> (Vec<CasmState>, Vec<PackedCasmState>, usize) {
    let n_rows = inputs.len();
    assert_ne!(
        n_rows, 0,
        "fixture has no rows for this pilot; pick a fixture that exercises it"
    );
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    inputs.resize(size, *inputs.first().unwrap());
    let packed = pack_values(&inputs);
    (inputs, packed, n_rows)
}

/// Zero-tolerance byte-compare of a `GenericSimdDiff` bundle (works for any pilot's
/// bundle — same field names): trace columns, lookup arrays, sub-inputs, and (bonus) the
/// finalized interaction trace + claimed sum.
macro_rules! assert_generic_diff_byte_identical {
    ($diff:expr) => {{
        let diff = $diff;
        // (i) Every committed trace column, every row.
        assert_eq!(
            diff.orig_rows.len(),
            diff.gen_rows.len(),
            "trace row count differs (log_size {})",
            diff.log_size
        );
        for (r, (a, b)) in diff.orig_rows.iter().zip(diff.gen_rows.iter()).enumerate() {
            assert_eq!(a, b, "trace row {r} differs");
        }
        // (ii) Every LookupData array.
        assert_packed_field_bundles_eq(&diff.orig_lookup, &diff.gen_lookup, "lookup_data");
        // (iii) Every SubComponentInputs scalar (pre-drain).
        assert_raw_field_bundles_eq(&diff.orig_sub, &diff.gen_sub, "sub_component_inputs");
        // Bonus: finalized interaction trace + claimed sum through the real logup path.
        assert_eq!(
            diff.orig_claimed_sum, diff.gen_claimed_sum,
            "interaction claimed sum differs"
        );
        assert_eq!(
            diff.orig_interaction_cols.len(),
            diff.gen_interaction_cols.len(),
            "interaction column count differs"
        );
        for (i, (a, b)) in diff
            .orig_interaction_cols
            .iter()
            .zip(diff.gen_interaction_cols.iter())
            .enumerate()
        {
            assert_eq!(a, b, "interaction column {i} differs");
        }
    }};
}

/// Gate (b) core: interpret `$out`'s recording for every row and byte-compare every
/// non-poisoned committed column + non-poisoned lookup word against the host writer's
/// output (`$diff.orig_*`). The oracle wraps the real host memory states (broadcast +
/// lane-0 — host `deduce_output` semantics by construction, incl. the encoded-id tag
/// dispatch inside `memory_id_to_big`).
macro_rules! assert_recording_interpreter_matches_host {
    ($diff:expr, $out:expr, $inputs:expr, $n_rows:expr, $mem_addr:expr, $mem_big:expr) => {{
        let diff = &$diff;
        let out = &$out;
        let oracle = |table: u32, key: u32, limb: u32| -> u32 {
            match table {
                TABLE_ADDR_TO_ID => {
                    $mem_addr
                        .deduce_output(PackedM31::broadcast(M31::from(key)))
                        .to_array()[0]
                        .0
                }
                TABLE_ID_TO_BIG => {
                    $mem_big
                        .deduce_output(PackedM31::broadcast(M31::from(key)))
                        .get_m31(limb as usize)
                        .to_array()[0]
                        .0
                }
                t => panic!("unexpected table id {t}"),
            }
        };

        let n_packed_rows = 1usize << (diff.log_size - LOG_N_LANES);
        for (r, cs) in $inputs.iter().enumerate() {
            let row_inputs = [cs.pc.0, cs.ap.0, cs.fp.0, u32::from(r < $n_rows)];
            let ro = interpret_row(&out.program, &row_inputs, &oracle);

            // (i) Every non-poisoned committed column.
            for (c, hv) in diff.orig_rows[r].iter().enumerate() {
                if !out.poisoned_cols.contains(&c) {
                    assert_eq!(
                        ro.columns[c], hv.0,
                        "interpreter vs host: row {r} column {c} differs"
                    );
                }
            }

            // (ii) Every non-poisoned lookup word (flat word order = LookupData
            // declaration order; per-field widths derived from the flats, never
            // hardcoded).
            let (pr, lane) = (r / N_LANES, r % N_LANES);
            let mut w = 0usize;
            for (fi, field) in diff.orig_lookup.iter().enumerate() {
                let width = field.len() / n_packed_rows;
                for k in 0..width {
                    if !out.poisoned_lookup_words.contains(&w) {
                        let hv = field[pr * width + k].to_array()[lane].0;
                        assert_eq!(
                            ro.lookup_words[w], hv,
                            "interpreter vs host: row {r} lookup word {w} (field {fi}, offset {k})"
                        );
                    }
                    w += 1;
                }
            }
            assert_eq!(
                w as u32, out.program.n_lookup_words,
                "lookup word layout mismatch between flats and program"
            );

            // (iii) Every non-poisoned SUB-INPUT word (flat word order =
            // SubComponentInputs declaration order — what the prove path feeds
            // downstream states from).
            let mut w = 0usize;
            for (fi, field) in diff.orig_sub.iter().enumerate() {
                let width = field.len() / n_packed_rows;
                for k in 0..width {
                    if !out.poisoned_sub_words.contains(&w) {
                        let hv = field[pr * width + k].as_array()[lane];
                        assert_eq!(
                            ro.sub_words[w], hv,
                            "interpreter vs host: row {r} sub word {w} (field {fi}, offset {k})"
                        );
                    }
                    w += 1;
                }
            }
            assert_eq!(
                w as u32, out.program.n_sub_words,
                "sub word layout mismatch between flats and program"
            );
        }
    }};
}

// --------------------------------- add_opcode ---------------------------------------

/// THE permanent regression fence for the golden-template pilot `add_opcode`
/// (103 columns — the decode-idiom superset).
#[test]
fn add_opcode_generic_simd_byte_identical() {
    let cg = fill_fixture(&[
        "add_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.add_opcode.expect("add_opcode populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(add_opcode::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &verify,
    ));
}

/// Gate (b) for `add_opcode`: fully recordable (zero poison), interpreter output equals
/// host columns + lookup words everywhere.
#[test]
fn add_opcode_recording_interpreter_matches_host() {
    let cg = fill_fixture(&[
        "add_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.add_opcode.expect("add_opcode populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    let diff = add_opcode::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);

    let out = add_opcode::record_add_opcode();
    assert!(
        out.poisoned_cols.is_empty() && out.poison_ops.is_empty(),
        "add_opcode must record fully (poisoned: {:?}, ops: {:?})",
        out.poisoned_cols,
        out.poison_ops
    );
    assert_recording_interpreter_matches_host!(diff, out, inputs, n_rows, mem_addr, mem_big);
}

/// THE prove-accessor parity gate — the local stand-in for a GPU run of the
/// `STWO_CUDA_WITNESS_JIT_PROVE=1` lane. Reproduces the exact prove-path data flow
/// with the reference interpreter in place of the kernel (which the hardware
/// selftest separately proves bit-identical to the interpreter):
///
///   interpret every PADDED row (enabler 1/0)  →  WORD-MAJOR flats (the launch's
///   D2H format)  →  `interaction_gen_from_flat_lookup_words` +
///   `sub_inputs_from_flat`  →  byte-compare against the host writer's
///   `LookupData` and `SubComponentInputs`.
///
/// `write_interaction_trace` is a pure function of (log_size, lookup_data,
/// elements), so byte-equal `LookupData` ⇒ byte-equal interaction trace + claimed
/// sum. The sub-input comparison spans ALL padded rows — the host feeds padding
/// rows too (`mults_0 = 1` everywhere); a lane that truncates at the real row
/// count fails this test (and would produce an unverifiable proof).

/// §6a LOCAL GATE engine: the emitted `JIT_LOGUP_DESCS` facts, resolved against
/// `JIT_LOOKUP_FIELDS` and run through the HOST MIRROR of the device pair kernel
/// (`logup_descs::host_mirror_raw_logup`), must finalize to the SAME interaction
/// trace and claimed sum as the module's generated `write_interaction_trace` —
/// over the same word-major flats. Proves the descriptor semantics without any
/// CUDA; the pod differential then covers only the kernel's re-implementation.
#[allow(clippy::too_many_arguments)]
fn assert_logup_descs_match_writer(
    label: &str,
    fields: &[(&str, usize)],
    facts: &[crate::witness::logup_descs::LogupDescFact],
    raw_ref: stwo_constraint_framework::RawLogupTrace,
    lookup_flat: &[u32],
    n_padded: usize,
    n_real: usize,
    elements: &cairo_air::relations::CommonLookupElements,
) {
    use stwo::prover::backend::Column as _;
    use stwo_constraint_framework::LogupFinalizeBackend;

    use crate::witness::logup_descs::{host_mirror_raw_logup, resolve_logup_descs};

    let descs = resolve_logup_descs(fields, facts);
    let raw_mirror = host_mirror_raw_logup(lookup_flat, n_padded, n_real, &descs, elements);
    let (evals_ref, sum_ref) = <SimdBackend as LogupFinalizeBackend>::finalize_raw_logup(raw_ref);
    let (evals_mir, sum_mir) =
        <SimdBackend as LogupFinalizeBackend>::finalize_raw_logup(raw_mirror);
    assert_eq!(sum_ref, sum_mir, "{label}: claimed sum differs");
    assert_eq!(
        evals_ref.len(),
        evals_mir.len(),
        "{label}: column count differs"
    );
    for (c, (a, b)) in evals_ref.iter().zip(&evals_mir).enumerate() {
        assert_eq!(
            a.values.to_cpu(),
            b.values.to_cpu(),
            "{label}: interaction column {c} differs"
        );
    }
}

/// Deterministic lookup elements for the §6a legs (any elements work — the gate
/// compares two computations of the same function of them).
fn test_lookup_elements() -> cairo_air::relations::CommonLookupElements {
    use stwo::core::channel::Blake2sChannel;
    cairo_air::relations::CommonLookupElements::draw(&mut Blake2sChannel::default())
}

#[test]
fn add_opcode_prove_accessors_match_host() {
    let cg = fill_fixture(&[
        "add_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.add_opcode.expect("add_opcode populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    // The gate's whole point includes padding-row semantics (the host feeds padding
    // rows downstream; `mults_0 = 1` everywhere). A fixture with a power-of-two row
    // count would silently stop exercising that — fail loudly instead.
    assert!(
        n_rows < inputs.len(),
        "fixture must leave add_opcode padding rows (n_real {} == padded {})",
        n_rows,
        inputs.len()
    );
    let diff = add_opcode::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);
    let out = add_opcode::record_add_opcode();
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        match table {
            TABLE_ADDR_TO_ID => {
                mem_addr
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .to_array()[0]
                    .0
            }
            TABLE_ID_TO_BIG => {
                mem_big
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .get_m31(limb as usize)
                    .to_array()[0]
                    .0
            }
            t => panic!("unexpected table id {t}"),
        }
    };

    // Interpret ALL padded rows and assemble the word-major flats exactly as
    // `launch_recorded_witness_for_prove` returns them.
    let n_padded = inputs.len();
    let n_lookup = out.program.n_lookup_words as usize;
    let n_sub = out.program.n_sub_words as usize;
    let mut lookup_flat = vec![0u32; n_lookup * n_padded];
    let mut sub_flat = vec![0u32; n_sub * n_padded];
    for (r, cs) in inputs.iter().enumerate() {
        let row_inputs = [cs.pc.0, cs.ap.0, cs.fp.0, u32::from(r < n_rows)];
        let ro = interpret_row(&out.program, &row_inputs, &oracle);
        for (w, &v) in ro.lookup_words.iter().enumerate() {
            lookup_flat[w * n_padded + r] = v;
        }
        for (w, &v) in ro.sub_words.iter().enumerate() {
            sub_flat[w * n_padded + r] = v;
        }
    }

    // (i) The reconstructed interaction generator's LookupData, byte-for-byte.
    let igen =
        add_opcode::interaction_gen_from_flat_lookup_words(diff.log_size, &lookup_flat, n_padded);
    assert_packed_field_bundles_eq(
        &add_opcode::test_lookup_data_flat(&igen),
        &diff.orig_lookup,
        "prove-accessor lookup_data",
    );

    // §6a leg: emitted JIT_LOGUP_DESCS → host mirror ≡ the generated writer
    // (the opcode shape: mults_0/mults_1 flats columns + trailing solo negated).
    {
        let elements = test_lookup_elements();
        let (raw_ref, _claim) = igen.write_interaction_trace(&elements);
        assert_logup_descs_match_writer(
            "add_opcode",
            add_opcode::JIT_LOOKUP_FIELDS,
            add_opcode::JIT_LOGUP_DESCS,
            raw_ref,
            &lookup_flat,
            n_padded,
            n_rows,
            &elements,
        );
    }

    // (ii) The decoded sub-inputs, all padded rows, against the host writer's
    // pre-drain SubComponentInputs (fields: vi tuple ×7 words, 3 addrs, 3 ids).
    let sub = add_opcode::sub_inputs_from_flat(&sub_flat, n_padded);
    assert_eq!(sub.verify_instruction.len(), n_padded);
    let n_packed_rows = n_padded / N_LANES;
    for r in 0..n_padded {
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let vi = &sub.verify_instruction[r];
        let host_vi: Vec<u32> = (0..7)
            .map(|k| diff.orig_sub[0][pr * 7 + k].as_array()[lane])
            .collect();
        let got_vi = [
            vi.0 .0, vi.1[0].0, vi.1[1].0, vi.1[2].0, vi.2[0].0, vi.2[1].0, vi.3 .0,
        ];
        assert_eq!(got_vi.as_slice(), host_vi.as_slice(), "vi tuple row {r}");
        for (j, addrs) in sub.memory_address_to_id.iter().enumerate() {
            assert_eq!(
                addrs[r].0,
                diff.orig_sub[1 + j][pr].as_array()[lane],
                "addr feed {j} row {r}"
            );
        }
        for (j, ids) in sub.memory_id_to_big.iter().enumerate() {
            assert_eq!(
                ids[r].0,
                diff.orig_sub[4 + j][pr].as_array()[lane],
                "id feed {j} row {r}"
            );
        }
    }
    let _ = n_packed_rows;
}

/// Smoke test for the recording lane: `add_opcode` records fully (no EXTENDED op ⇒
/// nothing poisoned), committing all 103 columns and emitting 117 lookup words.
#[test]
fn add_opcode_records_without_poison() {
    let out = add_opcode::record_add_opcode();
    assert!(
        out.poisoned_cols.is_empty(),
        "unexpected poisoned columns: {:?}",
        out.poisoned_cols
    );
    assert!(
        out.poisoned_lookup_words.is_empty(),
        "unexpected poisoned lookup words: {:?}",
        out.poisoned_lookup_words
    );
    assert!(
        out.poison_ops.is_empty(),
        "unexpected poison ops: {:?}",
        out.poison_ops
    );
    assert_eq!(out.program.n_cols, 103, "expected 103 committed columns");
    assert_eq!(
        out.program.n_lookup_words, 117,
        "expected 117 emitted lookup words"
    );
    assert_eq!(out.program.n_sub_words, 13, "expected 13 sub-input words");
    // The prove launch is governed fail-closed at 2048 instrs (the NVRTC/ptxas
    // cliff guard) — the pilot must stay comfortably under it or the lane
    // silently falls back to the host writer on the pod.
    let n_instrs = out.program.n_instrs();
    assert!(
        n_instrs <= 2048,
        "add_opcode recording ({n_instrs} instrs) exceeds the prove-lane governor cap"
    );
}

/// The new source lane's permanent soundness fence: the mechanically genericized
/// Blake-compress writer must reproduce the original AIR-generated SIMD writer
/// byte-for-byte across trace, lookup and all 324 sub-input words on the real
/// opcode fixture. This also validates the recorder-side expanded Blake-round
/// deduction against `ClaimGenerator::deduce_output` on real memory.
#[test]
fn blake_compress_opcode_generic_simd_is_byte_identical() {
    use crate::witness::components::blake_compress_opcode;

    let cg = fill_fixture(&[
        "blake_compress_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
        "range_check_7_2_5",
        "verify_bitwise_xor_8",
        "blake_round",
        "triple_xor_32",
    ]);
    let mut inputs = cg
        .blake_compress_opcode
        .as_ref()
        .expect("blake compress fixture")
        .inputs
        .clone();
    let n_rows = inputs.len();
    assert!(n_rows > 0, "fixture must exercise blake_compress_opcode");
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    inputs.resize(size, inputs[0]);
    let packed = pack_values(&inputs);
    assert_generic_diff_byte_identical!(blake_compress_opcode::generic_simd_diff(
        packed,
        n_rows,
        cg.memory_address_to_id.as_ref().unwrap(),
        cg.memory_id_to_big.as_ref().unwrap(),
        cg.verify_instruction.as_ref().unwrap(),
        cg.range_check_7_2_5.as_ref().unwrap(),
        cg.verify_bitwise_xor_8.as_ref().unwrap(),
        cg.blake_round.as_ref().unwrap(),
        cg.triple_xor_32.as_ref().unwrap(),
    ));
}

/// Parameterized prove-accessor parity gate (see `add_opcode_prove_accessors_match_
/// host` for the full rationale): interpret every PADDED row with real enabler
/// semantics -> word-major flats (the launch's D2H format) -> the module's
/// macro-generated accessors -> byte-compare `LookupData` + every sub feed against
/// the host writer. `require_padding` pins fixtures that exercise padding rows.
macro_rules! prove_accessor_parity_gate {
    ($test:ident, $module:ident, $mstr:literal, record = $record:path,
     n_addr = $na:expr, n_id = $ni:expr, require_padding = $pad:expr) => {
        #[test]
        fn $test() {
            use crate::witness::components::$module;
            let (cg, addr_ids, f252_values, small_values, _mem_arc) = fill_fixture_with_memory(&[
                $mstr,
                "memory_address_to_id",
                "memory_id_to_big",
                "verify_instruction",
            ]);
            let gen = cg.$module.expect("component populated");
            let mem_addr = cg.memory_address_to_id.expect("state");
            let mem_big = cg.memory_id_to_big.expect("state");
            let verify = cg.verify_instruction.expect("state");

            let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
            if $pad {
                assert!(
                    n_rows < inputs.len(),
                    "fixture must leave padding rows (n_real {} == padded {})",
                    n_rows,
                    inputs.len()
                );
            }
            let diff = $module::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);
            let out = $record();
            assert!(
                out.poisoned_cols.is_empty()
                    && out.poisoned_lookup_words.is_empty()
                    && out.poisoned_sub_words.is_empty(),
                "recording must be poison-free for the prove lane"
            );
            let oracle = |table: u32, key: u32, limb: u32| -> u32 {
                match table {
                    TABLE_ADDR_TO_ID => {
                        mem_addr
                            .deduce_output(PackedM31::broadcast(M31::from(key)))
                            .to_array()[0]
                            .0
                    }
                    TABLE_ID_TO_BIG => {
                        mem_big
                            .deduce_output(PackedM31::broadcast(M31::from(key)))
                            .get_m31(limb as usize)
                            .to_array()[0]
                            .0
                    }
                    t => panic!("unexpected table id {t}"),
                }
            };

            let n_padded = inputs.len();
            let n_lookup = out.program.n_lookup_words as usize;
            let n_sub = out.program.n_sub_words as usize;
            let mut lookup_flat = vec![0u32; n_lookup * n_padded];
            let mut sub_flat = vec![0u32; n_sub * n_padded];
            let rows: Vec<Vec<u32>> = inputs
                .iter()
                .enumerate()
                .map(|(r, cs)| vec![cs.pc.0, cs.ap.0, cs.fp.0, u32::from(r < n_rows)])
                .collect();
            for (r, row_inputs) in rows.iter().enumerate() {
                let ro = interpret_row(&out.program, row_inputs, &oracle);
                for (w, &v) in ro.lookup_words.iter().enumerate() {
                    lookup_flat[w * n_padded + r] = v;
                }
                for (w, &v) in ro.sub_words.iter().enumerate() {
                    sub_flat[w * n_padded + r] = v;
                }
            }

            let igen = $module::interaction_gen_from_flat_lookup_words(
                diff.log_size,
                &lookup_flat,
                n_padded,
            );
            assert_packed_field_bundles_eq(
                &$module::test_lookup_data_flat(&igen),
                &diff.orig_lookup,
                "prove-accessor lookup_data",
            );

            let (vi, addrs, ids) = $module::sub_inputs_from_flat(&sub_flat, n_padded);
            assert_eq!(vi.len(), n_padded);
            assert_eq!(addrs.len(), $na);
            assert_eq!(ids.len(), $ni);
            for r in 0..n_padded {
                let (pr, lane) = (r / N_LANES, r % N_LANES);
                let t = &vi[r];
                let got = [
                    t.0 .0, t.1[0].0, t.1[1].0, t.1[2].0, t.2[0].0, t.2[1].0, t.3 .0,
                ];
                let host: Vec<u32> = (0..7)
                    .map(|k| diff.orig_sub[0][pr * 7 + k].as_array()[lane])
                    .collect();
                assert_eq!(got.as_slice(), host.as_slice(), "vi tuple row {r}");
                for (j, col) in addrs.iter().enumerate() {
                    assert_eq!(
                        col[r].0,
                        diff.orig_sub[1 + j][pr].as_array()[lane],
                        "addr feed {j} row {r}"
                    );
                }
                for (j, col) in ids.iter().enumerate() {
                    assert_eq!(
                        col[r].0,
                        diff.orig_sub[1 + $na + j][pr].as_array()[lane],
                        "id feed {j} row {r}"
                    );
                }
            }

            let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
            assert_device_builtin_leg_matches_host(
                $mstr,
                out.program,
                false,
                &rows,
                &addr_ids,
                &f252_values,
                &small_values,
                &host_rows,
                &diff.orig_lookup,
                &diff.orig_sub,
            );
        }
    };
}

prove_accessor_parity_gate!(
    assert_eq_opcode_prove_accessors_match_host,
    assert_eq_opcode,
    "assert_eq_opcode",
    record = assert_eq_opcode::record_assert_eq_opcode,
    n_addr = 2,
    n_id = 0,
    // The all-opcode fixture has exactly 64 assert_eq states (a power of two) —
    // no padding rows here; padding semantics are pinned by the add_opcode gate.
    require_padding = false
);

// ------------------------------- assert_eq_opcode -----------------------------------

/// Gate (a) for tool-emitted pilot #2 `assert_eq_opcode` (12 columns).
#[test]
fn assert_eq_opcode_generic_simd_byte_identical() {
    let cg = fill_fixture(&[
        "assert_eq_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.assert_eq_opcode.expect("assert_eq_opcode populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(assert_eq_opcode::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &verify,
    ));
}

/// Gate (b) for `assert_eq_opcode`: fully recordable (zero poison; 12 cols, 24 lookup
/// words), interpreter output equals host everywhere.
#[test]
fn assert_eq_opcode_recording_interpreter_matches_host() {
    let cg = fill_fixture(&[
        "assert_eq_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.assert_eq_opcode.expect("assert_eq_opcode populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    let diff = assert_eq_opcode::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);

    let out = assert_eq_opcode::record_assert_eq_opcode();
    assert!(
        out.poisoned_cols.is_empty() && out.poison_ops.is_empty(),
        "assert_eq_opcode must record fully (poisoned: {:?}, ops: {:?})",
        out.poisoned_cols,
        out.poison_ops
    );
    assert_eq!(out.program.n_cols, 12, "expected 12 committed columns");
    assert_eq!(
        out.program.n_lookup_words, 24,
        "expected 24 emitted lookup words"
    );
    assert_recording_interpreter_matches_host!(diff, out, inputs, n_rows, mem_addr, mem_big);
}

prove_accessor_parity_gate!(
    jnz_opcode_taken_prove_accessors_match_host,
    jnz_opcode_taken,
    "jnz_opcode_taken",
    record = jnz_opcode_taken::record_jnz_opcode_taken,
    n_addr = 2,
    n_id = 2,
    require_padding = true
);

// ------------------------------- jnz_opcode_taken -----------------------------------

/// Gate (a) for tool-emitted pilot #3 `jnz_opcode_taken` (47 columns — the
/// EXTENDED-op pilot: `.inverse()` + `.eq()`/mask small-sign decode).
#[test]
fn jnz_opcode_taken_generic_simd_byte_identical() {
    let cg = fill_fixture(&[
        "jnz_opcode_taken",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.jnz_opcode_taken.expect("jnz_opcode_taken populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(jnz_opcode_taken::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &verify,
    ));
}

/// Gate (b) for `jnz_opcode_taken` — previously the PARTIAL-recording pilot
/// (inverse/eq poisoned); FULLY recordable since ISA-V2 (`M31Inverse`/`M31Eq` +
/// 0/1-register mask lowerings). The interpreter must match the host on EVERY
/// column and lookup word.
#[test]
fn jnz_opcode_taken_recording_interpreter_matches_host_outside_poison() {
    let cg = fill_fixture(&[
        "jnz_opcode_taken",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
    ]);
    let gen = cg.jnz_opcode_taken.expect("jnz_opcode_taken populated");
    let mem_addr = cg
        .memory_address_to_id
        .expect("memory_address_to_id populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let verify = cg.verify_instruction.expect("verify_instruction populated");

    let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    let diff = jnz_opcode_taken::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);

    let out = jnz_opcode_taken::record_jnz_opcode_taken();
    assert!(
        out.poisoned_cols.is_empty()
            && out.poisoned_lookup_words.is_empty()
            && out.poisoned_sub_words.is_empty()
            && out.poison_ops.is_empty(),
        "jnz must record FULLY under ISA-V2 (poisoned cols {:?}, ops {:?})",
        out.poisoned_cols,
        out.poison_ops
    );
    assert_eq!(out.program.n_cols, 47, "expected 47 committed columns");
    assert_eq!(out.program.n_lookup_words, 84);
    assert_eq!(out.program.n_sub_words, 11);
    assert_recording_interpreter_matches_host!(diff, out, inputs, n_rows, mem_addr, mem_big);
}

// --------------------------- cohort components (round-11) ---------------------------

/// The full three-gate fence for a cohort component: (a) generic-vs-original SIMD
/// byte-identity, (b) recording/interpreter equality vs the host writer (poison-free
/// under ISA-V2), (c) the prove-accessor parity gate.
macro_rules! full_component_gates {
    ($module:ident, $mstr:literal, gate_a = $ga:ident, gate_b = $gb:ident,
     parity = $par:ident, record = $record:path, n_addr = $na:expr, n_id = $ni:expr,
     require_padding = $pad:expr) => {
        #[test]
        fn $ga() {
            use crate::witness::components::$module;
            let cg = fill_fixture(&[
                $mstr,
                "memory_address_to_id",
                "memory_id_to_big",
                "verify_instruction",
            ]);
            let gen = cg.$module.expect("component populated");
            let mem_addr = cg.memory_address_to_id.expect("state");
            let mem_big = cg.memory_id_to_big.expect("state");
            let verify = cg.verify_instruction.expect("state");
            let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
            assert_generic_diff_byte_identical!($module::generic_simd_diff(
                packed, n_rows, &mem_addr, &mem_big, &verify,
            ));
        }

        #[test]
        fn $gb() {
            use crate::witness::components::$module;
            let cg = fill_fixture(&[
                $mstr,
                "memory_address_to_id",
                "memory_id_to_big",
                "verify_instruction",
            ]);
            let gen = cg.$module.expect("component populated");
            let mem_addr = cg.memory_address_to_id.expect("state");
            let mem_big = cg.memory_id_to_big.expect("state");
            let verify = cg.verify_instruction.expect("state");
            let (inputs, packed, n_rows) = pack_pilot_inputs(gen.inputs);
            let diff = $module::generic_simd_diff(packed, n_rows, &mem_addr, &mem_big, &verify);
            let out = $record();
            assert!(
                out.poisoned_cols.is_empty()
                    && out.poisoned_lookup_words.is_empty()
                    && out.poisoned_sub_words.is_empty()
                    && out.poison_ops.is_empty(),
                "must record FULLY (poisoned cols {:?}, ops {:?})",
                out.poisoned_cols,
                out.poison_ops
            );
            assert_recording_interpreter_matches_host!(
                diff, out, inputs, n_rows, mem_addr, mem_big
            );
        }

        prove_accessor_parity_gate!(
            $par,
            $module,
            $mstr,
            record = $record,
            n_addr = $na,
            n_id = $ni,
            require_padding = $pad
        );
    };
}

full_component_gates!(
    add_opcode_small,
    "add_opcode_small",
    gate_a = add_opcode_small_generic_simd_byte_identical,
    gate_b = add_opcode_small_recording_interpreter_matches_host,
    parity = add_opcode_small_prove_accessors_match_host,
    record = add_opcode_small::record_add_opcode_small,
    n_addr = 3,
    n_id = 3,
    require_padding = true
);

full_component_gates!(
    assert_eq_opcode_imm,
    "assert_eq_opcode_imm",
    gate_a = assert_eq_opcode_imm_generic_simd_byte_identical,
    gate_b = assert_eq_opcode_imm_recording_interpreter_matches_host,
    parity = assert_eq_opcode_imm_prove_accessors_match_host,
    record = assert_eq_opcode_imm::record_assert_eq_opcode_imm,
    n_addr = 2,
    n_id = 0,
    require_padding = true
);

full_component_gates!(
    assert_eq_opcode_double_deref,
    "assert_eq_opcode_double_deref",
    gate_a = assert_eq_opcode_double_deref_generic_simd_byte_identical,
    gate_b = assert_eq_opcode_double_deref_recording_interpreter_matches_host,
    parity = assert_eq_opcode_double_deref_prove_accessors_match_host,
    record = assert_eq_opcode_double_deref::record_assert_eq_opcode_double_deref,
    n_addr = 3,
    n_id = 1,
    require_padding = true
);

full_component_gates!(
    call_opcode_abs,
    "call_opcode_abs",
    gate_a = call_opcode_abs_generic_simd_byte_identical,
    gate_b = call_opcode_abs_recording_interpreter_matches_host,
    parity = call_opcode_abs_prove_accessors_match_host,
    record = call_opcode_abs::record_call_opcode_abs,
    n_addr = 3,
    n_id = 3,
    require_padding = true
);

full_component_gates!(
    call_opcode_rel_imm,
    "call_opcode_rel_imm",
    gate_a = call_opcode_rel_imm_generic_simd_byte_identical,
    gate_b = call_opcode_rel_imm_recording_interpreter_matches_host,
    parity = call_opcode_rel_imm_prove_accessors_match_host,
    record = call_opcode_rel_imm::record_call_opcode_rel_imm,
    n_addr = 3,
    n_id = 3,
    require_padding = true
);

full_component_gates!(
    jnz_opcode_non_taken,
    "jnz_opcode_non_taken",
    gate_a = jnz_opcode_non_taken_generic_simd_byte_identical,
    gate_b = jnz_opcode_non_taken_recording_interpreter_matches_host,
    parity = jnz_opcode_non_taken_prove_accessors_match_host,
    record = jnz_opcode_non_taken::record_jnz_opcode_non_taken,
    n_addr = 1,
    n_id = 1,
    require_padding = true
);

full_component_gates!(
    jump_opcode_abs,
    "jump_opcode_abs",
    gate_a = jump_opcode_abs_generic_simd_byte_identical,
    gate_b = jump_opcode_abs_recording_interpreter_matches_host,
    parity = jump_opcode_abs_prove_accessors_match_host,
    record = jump_opcode_abs::record_jump_opcode_abs,
    n_addr = 1,
    n_id = 1,
    require_padding = true
);

full_component_gates!(
    jump_opcode_double_deref,
    "jump_opcode_double_deref",
    gate_a = jump_opcode_double_deref_generic_simd_byte_identical,
    gate_b = jump_opcode_double_deref_recording_interpreter_matches_host,
    parity = jump_opcode_double_deref_prove_accessors_match_host,
    record = jump_opcode_double_deref::record_jump_opcode_double_deref,
    n_addr = 2,
    n_id = 2,
    require_padding = true
);

full_component_gates!(
    jump_opcode_rel,
    "jump_opcode_rel",
    gate_a = jump_opcode_rel_generic_simd_byte_identical,
    gate_b = jump_opcode_rel_recording_interpreter_matches_host,
    parity = jump_opcode_rel_prove_accessors_match_host,
    record = jump_opcode_rel::record_jump_opcode_rel,
    n_addr = 1,
    n_id = 1,
    require_padding = true
);

full_component_gates!(
    jump_opcode_rel_imm,
    "jump_opcode_rel_imm",
    gate_a = jump_opcode_rel_imm_generic_simd_byte_identical,
    gate_b = jump_opcode_rel_imm_recording_interpreter_matches_host,
    parity = jump_opcode_rel_imm_prove_accessors_match_host,
    record = jump_opcode_rel_imm::record_jump_opcode_rel_imm,
    n_addr = 1,
    n_id = 1,
    require_padding = true
);

full_component_gates!(
    ret_opcode,
    "ret_opcode",
    gate_a = ret_opcode_generic_simd_byte_identical,
    gate_b = ret_opcode_recording_interpreter_matches_host,
    parity = ret_opcode_prove_accessors_match_host,
    record = ret_opcode::record_ret_opcode,
    n_addr = 2,
    n_id = 2,
    require_padding = true
);

// ------------------------ pedersen_aggregator_window_bits_18 ------------------------

/// The BUILTIN-lane recording manifest (ISA-V3): the aggregator's generic body now
/// records COMPLETELY — zero poisons; its 28 EC deduces are real `DeduceCall(kind=2)`
/// instructions the kernel lowers to the fp256 device function. Any poison
/// reappearing is a regression; a change in the deduce count is unreviewed drift.
#[test]
fn pedersen_aggregator_recording_poison_manifest() {
    use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};

    use crate::witness::components::pedersen_aggregator_window_bits_18 as agg;
    let rec = agg::record_pedersen_aggregator_window_bits_18();
    assert!(rec.poison_ops.is_empty(), "poisons: {:?}", rec.poison_ops);
    assert!(rec.poisoned_cols.is_empty() && rec.poisoned_lookup_words.is_empty());
    assert!(rec.poisoned_sub_words.is_empty());
    let deduces = rec
        .program
        .insts
        .iter()
        .filter(|i| {
            i.op == WitnessOp::DeduceCall as u8 && i.imm == DeduceKind::PartialEcMulW18 as u32
        })
        .count();
    assert_eq!(deduces, 28, "EC deduce count drifted");
}

/// Gate (a) for the BUILTIN pilot `pedersen_aggregator_window_bits_18`: the generic
/// body on `SimdWitnessEval` (flat input words + iota + mults columns + felt
/// sub-words + the REAL `deduce_partial_ec_mul_w18` hook) is byte-identical to the
/// original writer over the full fixture — every committed column, every lookup
/// word, every sub-component input word.
#[test]
fn pedersen_aggregator_generic_simd_byte_identical() {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    use crate::witness::components::pedersen_aggregator_window_bits_18 as agg;

    // The opcode fixture has no pedersen rows; run the 15-instance pedersen builtin
    // program through the VM (the same pipeline prover.rs's aggregator test uses).
    let compiled = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run_and_adapt pedersen fixture");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "pedersen_aggregator_window_bits_18",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_8",
        "partial_ec_mul_window_bits_18",
        "pedersen_builtin",
    ] {
        set.insert(c);
    }
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    // The aggregator's mults are FED by the pedersen builtin's write_trace — run it
    // first, exactly as the production write_trace ordering does.
    {
        let pb = cg
            .pedersen_builtin
            .take()
            .expect("pedersen_builtin populated");
        let mem_addr = cg
            .memory_address_to_id
            .as_ref()
            .expect("memory_address_to_id state");
        let agg_state = cg
            .pedersen_aggregator_window_bits_18
            .as_ref()
            .expect("aggregator state");
        let _ = pb.write_trace(mem_addr, agg_state);
    }
    let gen = cg
        .pedersen_aggregator_window_bits_18
        .expect("pedersen_aggregator_window_bits_18 populated");
    let mem_big = cg.memory_id_to_big.expect("memory_id_to_big populated");
    let rc8 = cg.range_check_8.expect("range_check_8 populated");
    let w18 = cg
        .partial_ec_mul_window_bits_18
        .expect("partial_ec_mul_window_bits_18 populated");

    // Replicate the write_trace preamble exactly (sort by key, unzip, pad, pack).
    let mut inputs_mults = gen
        .mults
        .iter()
        .map(|entry| (*entry.key(), M31(entry.value().load(Ordering::Relaxed))))
        .collect::<Vec<_>>();
    inputs_mults.sort_by_key(|(input, _)| input.0);
    let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let n_rows = inputs.len();
    assert_ne!(n_rows, 0, "fixture has no pedersen_aggregator rows");
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    inputs.resize(size, *inputs.first().unwrap());
    mults.resize(size, M31::zero());
    let packed_inputs = pack_values(&inputs);
    let packed_mults = pack_values(&mults);

    assert_generic_diff_byte_identical!(agg::generic_simd_diff(
        packed_inputs,
        vec![packed_mults],
        &mem_big,
        &rc8,
        &w18,
    ));
}

// ------------------- newly-emitted cohort (opcode fixture) -------------------------

/// Gate (a) for `mul_opcode_small` (u32-family opcode, newly emitted).
#[test]
fn mul_opcode_small_generic_simd_byte_identical() {
    use crate::witness::components::mul_opcode_small as m;
    let cg = fill_fixture(&[
        "mul_opcode_small",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
        "range_check_11",
    ]);
    let gen = cg.mul_opcode_small.expect("mul_opcode_small populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let vi = cg.verify_instruction.expect("verify_instruction");
    let rc11 = cg.range_check_11.expect("range_check_11");
    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &vi, &rc11,
    ));
}

/// Gate (a) for `qm_31_add_mul_opcode` (newly emitted).
#[test]
fn qm_31_add_mul_opcode_generic_simd_byte_identical() {
    use crate::witness::components::qm_31_add_mul_opcode as m;
    let cg = fill_fixture(&[
        "qm_31_add_mul_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
        "range_check_4_4_4_4",
    ]);
    let gen = cg
        .qm_31_add_mul_opcode
        .expect("qm_31_add_mul_opcode populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let vi = cg.verify_instruction.expect("verify_instruction");
    let rc4444 = cg.range_check_4_4_4_4.expect("range_check_4_4_4_4");
    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &vi, &rc4444,
    ));
}

/// Gate (a) for `verify_instruction` (mults-shaped, newly emitted): the SERIAL host
/// component every opcode feeds — its generic body includes the mults input column.
#[test]
fn verify_instruction_generic_simd_byte_identical() {
    use crate::witness::components::verify_instruction as m;
    let cg = fill_fixture(&[
        "verify_instruction",
        "range_check_7_2_5",
        "range_check_4_3",
        "memory_address_to_id",
        "memory_id_to_big",
        // Feed verify_instruction's mults the way production does: run an opcode
        // writer that pushes into it.
        "add_opcode",
    ]);
    let vi = cg.verify_instruction.expect("verify_instruction populated");
    let rc725 = cg.range_check_7_2_5.expect("range_check_7_2_5");
    let rc43 = cg.range_check_4_3.expect("range_check_4_3");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    {
        let add = cg.add_opcode.expect("add_opcode populated");
        let _ = add.write_trace(&mem_addr, &mem_big, &vi);
    }
    let mut inputs_mults = vi
        .mults
        .iter()
        .map(|entry| (*entry.key(), M31(entry.value().load(Ordering::Relaxed))))
        .collect::<Vec<_>>();
    inputs_mults.sort_by_key(|(input, _)| input.0);
    let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let n_rows = inputs.len();
    assert_ne!(
        n_rows, 0,
        "no verify_instruction rows after feeding add_opcode"
    );
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    inputs.resize(size, *inputs.first().unwrap());
    mults.resize(size, M31::zero());
    let packed_inputs = pack_values(&inputs);
    let packed_mults = pack_values(&mults);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed_inputs,
        vec![packed_mults],
        &rc725,
        &rc43,
        &mem_addr,
        &mem_big,
    ));
}

// --------------------------------- blake_round -------------------------------------

/// Gate (a) for `blake_round` (u32 family, newly emitted): the generic body — u32
/// input words (`input_u32`), `u32_low/high/from_limbs`, u32 sub-words, and the REAL
/// `deduce_blake_g` / `deduce_blake_round_sigma` hooks — is byte-identical to the
/// original writer. No test fixture exercises blake, so the inputs are SYNTHETIC but
/// memory-valid: the message pointer targets low program addresses present in the
/// opcode fixture's memory (both writers see identical inputs, so parity is exact
/// regardless of semantic meaning).
#[test]
fn blake_round_generic_simd_byte_identical() {
    use stwo_cairo_common::prover_types::cpu::UInt32;

    use crate::witness::components::blake_round as m;
    let cg = fill_fixture(&[
        "blake_round",
        "blake_round_sigma",
        "blake_g",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_7_2_5",
    ]);
    let sigma = cg.blake_round_sigma.expect("blake_round_sigma populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let rc725 = cg.range_check_7_2_5.expect("range_check_7_2_5");
    let blake_g = cg.blake_g.expect("blake_g populated");

    // Synthetic rows: chain id, round < 10, 16 message words, message pointer at a
    // low program address (the fixture's program segment starts at 1 and is far
    // longer than ptr+16, so every derived address deduces successfully).
    let inputs: Vec<m::InputType> = (0..24u32)
        .map(|i| {
            let words: [UInt32; 16] =
                std::array::from_fn(|j| UInt32::from(0x9E37_79B9u32.wrapping_mul(j as u32 + i)));
            (M31(i + 1), M31(i % 10), (words, M31(1 + (i % 4))))
        })
        .collect();
    let n_rows = inputs.len();
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    let mut padded = inputs;
    padded.resize(size, *padded.first().unwrap());
    let packed = pack_values(&padded);

    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &sigma, &mem_addr, &mem_big, &rc725, &blake_g,
    ));
}

/// The blake_round recording manifest (ISA-V3): fully recorded — zero poisons; 8
/// `DeduceCall(BlakeG)` + 1 `DeduceCall(BlakeRoundSigma)` instructions.
#[test]
fn blake_round_recording_poison_manifest() {
    use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};

    use crate::witness::components::blake_round as m;
    let rec = m::record_blake_round();
    assert!(rec.poison_ops.is_empty(), "poisons: {:?}", rec.poison_ops);
    let count = |k: DeduceKind| {
        rec.program
            .insts
            .iter()
            .filter(|i| i.op == WitnessOp::DeduceCall as u8 && i.imm == k as u32)
            .count()
    };
    assert_eq!(count(DeduceKind::BlakeG), 8);
    assert_eq!(count(DeduceKind::BlakeRoundSigma), 1);
}

// ---------------- ISA-V3: builtin recordings through the interpreter ----------------

/// The shared `DeduceHost`: kinds delegate to the REAL host `fast_deduction`
/// functions (broadcast one scalar row through the packed API, read lane 0) — the
/// reference is the host's own implementation, never a duplicate.
struct FastDeductionHost;
impl stwo_backend_cuda::jit_witness::interp::DeduceHost for FastDeductionHost {
    fn deduce(&mut self, kind: u32, args: &[u32]) -> Vec<u32> {
        use stwo_cairo_common::prover_types::cpu::UInt32;
        use stwo_cairo_common::prover_types::simd::{
            PackedFelt252, PackedFelt252Width27, PackedUInt32,
        };

        use crate::witness::fast_deduction::blake::{PackedBlakeG, PackedBlakeRoundSigma};
        use crate::witness::fast_deduction::pedersen::{
            PackedPartialEcMulWindowBits18, PackedPedersenPointsTableWindowBits18,
        };
        use crate::witness::fast_deduction::poseidon::{
            PackedCube252, PackedPoseidon3PartialRoundsChain, PackedPoseidonFullRoundChain,
            PackedPoseidonRoundKeys,
        };
        let m31 = |v: u32| PackedM31::broadcast(M31(v));
        let felt =
            |limbs: &[u32]| PackedFelt252::from_limbs(std::array::from_fn(|i| m31(limbs[i])));
        match kind {
            0 => {
                let words: [PackedUInt32; 6] =
                    std::array::from_fn(|i| PackedUInt32::broadcast(UInt32::from(args[i])));
                PackedBlakeG::deduce_output(words)
                    .iter()
                    .map(|w| w.simd.as_array()[0])
                    .collect()
            }
            1 => PackedBlakeRoundSigma::deduce_output(m31(args[0]))
                .iter()
                .map(|v| v.to_array()[0].0)
                .collect(),
            2 => {
                let windows: [PackedM31; 14] = std::array::from_fn(|i| m31(args[2 + i]));
                let acc = [felt(&args[16..44]), felt(&args[44..72])];
                let (chain, round, (wins, accs)) = PackedPartialEcMulWindowBits18::deduce_output((
                    m31(args[0]),
                    m31(args[1]),
                    (windows, acc),
                ));
                let mut out = vec![chain.to_array()[0].0, round.to_array()[0].0];
                out.extend(wins.iter().map(|w| w.to_array()[0].0));
                for f in &accs {
                    out.extend((0..28).map(|i| f.get_m31(i).to_array()[0].0));
                }
                out
            }
            3 => {
                let points = PackedPedersenPointsTableWindowBits18::deduce_output([m31(args[0])]);
                let mut out = Vec::with_capacity(56);
                for f in &points {
                    out.extend((0..28).map(|i| f.get_m31(i).to_array()[0].0));
                }
                out
            }
            // fp256 body ops: the host reference is Felt252's own operators
            // (canonical-value semantics — the device functions mirror them).
            4..=7 => {
                use stwo_cairo_common::prover_types::cpu::Felt252;
                let scalar = |limbs: &[u32]| {
                    let m31s: Vec<M31> = limbs.iter().map(|&v| M31(v)).collect();
                    Felt252::from_limbs(&m31s)
                };
                let a = scalar(&args[..28]);
                let b = scalar(&args[28..56]);
                let r = match kind {
                    4 => a + b,
                    5 => a - b,
                    6 => a * b,
                    _ => a / b,
                };
                (0..28).map(|i| r.get_m31(i).0).collect()
            }
            8 => PackedPoseidonRoundKeys::deduce_output([m31(args[0])])
                .iter()
                .flat_map(|felt| (0..10).map(|word| felt.get_m31(word).to_array()[0].0))
                .collect(),
            9 => {
                let input =
                    PackedFelt252Width27::from_limbs(std::array::from_fn(|word| m31(args[word])));
                let output = PackedCube252::deduce_output(input);
                (0..10)
                    .map(|word| output.get_m31(word).to_array()[0].0)
                    .collect()
            }
            10 => {
                let state = std::array::from_fn(|felt| {
                    PackedFelt252Width27::from_limbs(std::array::from_fn(|word| {
                        m31(args[2 + felt * 10 + word])
                    }))
                });
                let (chain, round, state) = PackedPoseidonFullRoundChain::deduce_output((
                    m31(args[0]),
                    m31(args[1]),
                    state,
                ));
                let mut output = vec![chain.to_array()[0].0, round.to_array()[0].0];
                for felt in &state {
                    output.extend((0..10).map(|word| felt.get_m31(word).to_array()[0].0));
                }
                output
            }
            11 => {
                let state = std::array::from_fn(|felt| {
                    PackedFelt252Width27::from_limbs(std::array::from_fn(|word| {
                        m31(args[2 + felt * 10 + word])
                    }))
                });
                let (chain, round, state) = PackedPoseidon3PartialRoundsChain::deduce_output((
                    m31(args[0]),
                    m31(args[1]),
                    state,
                ));
                let mut output = vec![chain.to_array()[0].0, round.to_array()[0].0];
                for felt in &state {
                    output.extend((0..10).map(|word| felt.get_m31(word).to_array()[0].0));
                }
                output
            }
            k => panic!("unexpected deduce kind {k}"),
        }
    }
}

fn assert_poseidon_recording_interpreter_matches_host<const N_TRACE_COLUMNS: usize>(
    program: &stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    rows: &[Vec<u32>],
    orig_rows: &[[M31; N_TRACE_COLUMNS]],
    orig_lookup: &[Vec<PackedM31>],
    orig_sub: &[Vec<Simd<u32, N_LANES>>],
) {
    use stwo_backend_cuda::jit_witness::interp::interpret_row_with;

    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };
    let n_packed_rows = rows.len() / N_LANES;
    for (r, row_inputs) in rows.iter().enumerate() {
        let ro = interpret_row_with(program, row_inputs, &oracle, &mut FastDeductionHost);
        for (c, hv) in orig_rows[r].iter().enumerate() {
            assert_eq!(ro.columns[c], hv.0, "row {r} column {c}");
        }
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let mut w = 0;
        for field in orig_lookup {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].to_array()[lane].0;
                assert_eq!(ro.lookup_words[w], hv, "row {r} lookup word {w} (+{k})");
                w += 1;
            }
        }
        let mut w = 0;
        for field in orig_sub {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].as_array()[lane];
                assert_eq!(ro.sub_words[w], hv, "row {r} sub word {w} (+{k})");
                w += 1;
            }
        }
    }
}

fn poseidon_chain_rows<const N_STATE: usize>(
    inputs: &[(PackedM31, PackedM31, [PackedFelt252Width27; N_STATE])],
    n_rows: usize,
) -> Vec<Vec<u32>> {
    (0..inputs.len() * N_LANES)
        .map(|r| {
            let (pr, lane) = (r / N_LANES, r % N_LANES);
            let input = &inputs[pr];
            let mut row = vec![input.0.to_array()[lane].0, input.1.to_array()[lane].0];
            for felt in &input.2 {
                row.extend((0..10).map(|word| felt.get_m31(word).to_array()[lane].0));
            }
            row.push(u32::from(r < n_rows));
            row.push(r as u32);
            row
        })
        .collect()
}

/// GATE (c), pod only: the same recorded program LAUNCHED AS A CUDA KERNEL on the
/// slot-layout input columns, byte-compared against the host writer everywhere the
/// interpreter gate compares (committed columns, lookup words, sub words — all padded
/// rows). No-op on stub builds (macOS/CI); on a pod build it runs as part of the
/// normal suite, so `cargo nextest` on hardware IS the device gate.
#[allow(clippy::too_many_arguments)]
fn assert_device_builtin_leg_matches_host(
    label: &'static str,
    program: stwo_backend_cuda::jit_witness::isa::WitnessProgram,
    needs_pedersen_table: bool,
    rows: &[Vec<u32>],
    addr_ids: &[u32],
    f252_values: &[[u32; 8]],
    small_values: &[u128],
    orig_rows: &[Vec<M31>],
    orig_lookup: &[Vec<PackedM31>],
    orig_sub: &[Vec<Simd<u32, N_LANES>>],
) {
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("device leg [{label}]: SKIPPED (stub build)");
        return;
    }
    if needs_pedersen_table {
        assert!(
            crate::witness::jit_prove_backend::ensure_device_pedersen_table(),
            "[{label}] host pedersen table registration failed on a CUDA build"
        );
    }
    let n = rows.len();
    // Transpose the interpreter gate's per-row slot vectors into raw input
    // columns, trimmed to the program's read extent (a body that never reads its
    // trailing slots — blake_round's iota — records fewer inputs than the
    // canonical layout provides; the launch requires an exact count).
    let n_slots = (program.n_inputs as usize).min(rows[0].len());
    let cols: Vec<Vec<u32>> = (0..n_slots)
        .map(|s| rows.iter().map(|r| r[s]).collect())
        .collect();
    stwo_backend_cuda::jit_witness::register_recorded_program(label, program);
    let tables = stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(
        addr_ids,
        f252_values,
        small_values,
    );
    let (dev_cols, _lookup_dev, lookup_flat, _sub_dev, sub_flat) =
        stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
            label, &cols, &tables, true, // want_host_sub
            true,
        )
        .unwrap_or_else(|| panic!("device leg [{label}]: launch unavailable"));

    assert_eq!(dev_cols.len(), orig_rows[0].len(), "[{label}] column count");
    for (c, col) in dev_cols.iter().enumerate() {
        let dv = col.to_vec();
        for r in 0..n {
            assert_eq!(
                dv[r].0, orig_rows[r][c].0,
                "[{label}] device col {c} row {r}"
            );
        }
    }
    let n_packed_rows = n / N_LANES;
    // Word-major flats: `flat[w * n + r]`, `w` enumerating fields in declaration
    // order then words within the field — the same enumeration the recorder used.
    let mut w = 0usize;
    for field in orig_lookup {
        let width = field.len() / n_packed_rows;
        for k in 0..width {
            for r in 0..n {
                let hv = field[(r / N_LANES) * width + k].to_array()[r % N_LANES].0;
                assert_eq!(
                    lookup_flat[w * n + r],
                    hv,
                    "[{label}] lookup word {w} row {r}"
                );
            }
            w += 1;
        }
    }
    let mut w = 0usize;
    for field in orig_sub {
        let width = field.len() / n_packed_rows;
        for k in 0..width {
            for r in 0..n {
                let hv = field[(r / N_LANES) * width + k].as_array()[r % N_LANES];
                assert_eq!(sub_flat[w * n + r], hv, "[{label}] sub word {w} row {r}");
            }
            w += 1;
        }
    }
    eprintln!(
        "device leg [{label}]: PASS ({n} rows, {} cols, {} lookup words, {} sub words)",
        dev_cols.len(),
        lookup_flat.len() / n,
        sub_flat.len() / n,
    );
}

/// GATE (b) for `blake_round` (ISA-V3): the RECORDED PROGRAM — the exact bytecode the
/// CUDA kernel will replay — interpreted with the fast_deduction reference host is
/// byte-identical to the host writer on every committed column, lookup word and sub
/// word, over all padded rows. This is the strongest pre-hardware validation the
/// witness lane has.
#[test]
fn blake_round_recording_interpreter_matches_host() {
    use stwo_backend_cuda::jit_witness::interp::interpret_row_with;
    use stwo_cairo_common::prover_types::cpu::UInt32;

    use crate::witness::components::blake_round as m;
    let (cg, addr_ids, f252_values, small_values, _mem_arc) = fill_fixture_with_memory(&[
        "blake_round",
        "blake_round_sigma",
        "blake_g",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_7_2_5",
    ]);
    let sigma = cg.blake_round_sigma.expect("sigma");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let rc725 = cg.range_check_7_2_5.expect("rc725");
    let blake_g = cg.blake_g.expect("blake_g");

    let inputs: Vec<m::InputType> = (0..24u32)
        .map(|i| {
            let words: [UInt32; 16] =
                std::array::from_fn(|j| UInt32::from(0x9E37_79B9u32.wrapping_mul(j as u32 + i)));
            (M31(i + 1), M31(i % 10), (words, M31(1 + (i % 4))))
        })
        .collect();
    let n_rows = inputs.len();
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    let mut padded = inputs;
    padded.resize(size, *padded.first().unwrap());
    let packed = pack_values(&padded);
    let diff = m::generic_simd_diff(
        packed, n_rows, &sigma, &mem_addr, &mem_big, &rc725, &blake_g,
    );

    let out = m::record_blake_round();
    assert!(out.poison_ops.is_empty(), "poisons: {:?}", out.poison_ops);

    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        match table {
            TABLE_ADDR_TO_ID => {
                mem_addr
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .to_array()[0]
                    .0
            }
            TABLE_ID_TO_BIG => {
                mem_big
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .get_m31(limb as usize)
                    .to_array()[0]
                    .0
            }
            t => panic!("unexpected table id {t}"),
        }
    };

    // Slot layout: flat input words 0..19 (m31, m31, 16 raw u32 words, m31),
    // enabler 19, iota 20.
    let rows: Vec<Vec<u32>> = (0..(1usize << diff.log_size))
        .map(|r| {
            let src = &padded[r];
            let mut row_inputs: Vec<u32> = vec![src.0 .0, src.1 .0];
            row_inputs.extend(src.2 .0.iter().map(|w| w.value));
            row_inputs.push(src.2 .1 .0);
            row_inputs.push(u32::from(r < n_rows)); // enabler
            row_inputs.push(r as u32); // iota (unused by this body)
            row_inputs
        })
        .collect();

    let n_packed_rows = 1usize << (diff.log_size - LOG_N_LANES);
    for (r, row_inputs) in rows.iter().enumerate() {
        let ro = interpret_row_with(&out.program, row_inputs, &oracle, &mut FastDeductionHost);

        for (c, hv) in diff.orig_rows[r].iter().enumerate() {
            assert_eq!(ro.columns[c], hv.0, "row {r} column {c}");
        }
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let mut w = 0usize;
        for field in diff.orig_lookup.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].to_array()[lane].0;
                assert_eq!(ro.lookup_words[w], hv, "row {r} lookup word {w} (+{k})");
                w += 1;
            }
        }
        let mut w = 0usize;
        for field in diff.orig_sub.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].as_array()[lane];
                assert_eq!(ro.sub_words[w], hv, "row {r} sub word {w} (+{k})");
                w += 1;
            }
        }
    }

    // Prove-accessor parity: the macro-generated igen rebuild from word-major
    // flats must reproduce the host `LookupData` byte-for-byte — fences the
    // 850-word field-list data entry against `LookupData` drift.
    let n_padded = 1usize << diff.log_size;
    let flat_from = |fields: &[Vec<PackedM31>]| -> Vec<u32> {
        let n_words: usize = fields.iter().map(|f| f.len() / n_packed_rows).sum();
        let mut flat = vec![0u32; n_words * n_padded];
        let mut w = 0usize;
        for field in fields {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                for r in 0..n_padded {
                    flat[w * n_padded + r] =
                        field[(r / N_LANES) * width + k].to_array()[r % N_LANES].0;
                }
                w += 1;
            }
        }
        flat
    };
    let lookup_flat = flat_from(&diff.orig_lookup);
    let igen =
        m::interaction_gen_from_flat_lookup_words(diff.log_size, n_rows, &lookup_flat, n_padded);
    assert_packed_field_bundles_eq(
        &m::test_lookup_data_flat(&igen),
        &diff.orig_lookup,
        "blake_round prove-accessor lookup_data",
    );

    // §6a leg: emitted JIT_LOGUP_DESCS → host mirror ≡ the generated writer.
    // Blake exercises what opcodes cannot: NON-ADJACENT pairing (sigma ⟷
    // rc_7_2_5), constant-one mults, and the real-row ENABLER (pair + solo).
    {
        let elements = test_lookup_elements();
        let (raw_ref, _claim) = igen.write_interaction_trace(&elements);
        assert_logup_descs_match_writer(
            "blake_round",
            m::JIT_LOOKUP_FIELDS,
            m::JIT_LOGUP_DESCS,
            raw_ref,
            &lookup_flat,
            n_padded,
            n_rows,
            &elements,
        );
    }

    // GATE (c), pod builds only: the same program as an actual CUDA kernel.
    let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
    assert_device_builtin_leg_matches_host(
        "blake_round",
        out.program,
        false,
        &rows,
        &addr_ids,
        &f252_values,
        &small_values,
        &host_rows,
        &diff.orig_lookup,
        &diff.orig_sub,
    );
}

/// GATE (b) for `pedersen_aggregator_window_bits_18` (ISA-V3): the recorded program —
/// 28 real EC-round `DeduceCall`s included — interpreted with the fast_deduction
/// reference host is byte-identical to the host writer everywhere.
#[test]
fn pedersen_aggregator_recording_interpreter_matches_host() {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_backend_cuda::jit_witness::interp::interpret_row_with;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    use crate::witness::components::pedersen_aggregator_window_bits_18 as agg;

    let compiled = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run_and_adapt pedersen fixture");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let addr_ids: Vec<u32> = memory.address_to_id.iter().map(|e| e.0).collect();
    let f252_values = memory.f252_values.clone();
    let small_values = memory.small_values.clone();
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "pedersen_aggregator_window_bits_18",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_8",
        "partial_ec_mul_window_bits_18",
        "pedersen_builtin",
    ] {
        set.insert(c);
    }
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    {
        let pb = cg
            .pedersen_builtin
            .take()
            .expect("pedersen_builtin populated");
        let mem_addr = cg.memory_address_to_id.as_ref().expect("mem addr");
        let agg_state = cg
            .pedersen_aggregator_window_bits_18
            .as_ref()
            .expect("aggregator state");
        let _ = pb.write_trace(mem_addr, agg_state);
    }
    let gen = cg
        .pedersen_aggregator_window_bits_18
        .expect("aggregator populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let rc8 = cg.range_check_8.expect("range_check_8");
    let w18 = cg.partial_ec_mul_window_bits_18.expect("w18");

    let mut inputs_mults = gen
        .mults
        .iter()
        .map(|entry| (*entry.key(), M31(entry.value().load(Ordering::Relaxed))))
        .collect::<Vec<_>>();
    inputs_mults.sort_by_key(|(input, _)| input.0);
    let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let n_rows = inputs.len();
    assert_ne!(n_rows, 0);
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    inputs.resize(size, *inputs.first().unwrap());
    mults.resize(size, M31::zero());
    let packed_inputs = pack_values(&inputs);
    let packed_mults = pack_values(&mults);
    let diff = agg::generic_simd_diff(packed_inputs, vec![packed_mults], &mem_big, &rc8, &w18);

    let out = agg::record_pedersen_aggregator_window_bits_18();
    assert!(out.poison_ops.is_empty(), "poisons: {:?}", out.poison_ops);

    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        match table {
            TABLE_ADDR_TO_ID => {
                mem_addr
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .to_array()[0]
                    .0
            }
            TABLE_ID_TO_BIG => {
                mem_big
                    .deduce_output(PackedM31::broadcast(M31::from(key)))
                    .get_m31(limb as usize)
                    .to_array()[0]
                    .0
            }
            t => panic!("unexpected table id {t}"),
        }
    };

    // Slot layout: inputs 0..3 (in.0[0], in.0[1], in.1), enabler 3, iota 4,
    // mults[0] 5.
    let rows: Vec<Vec<u32>> = (0..(1usize << diff.log_size))
        .map(|r| {
            let src = &inputs[r];
            vec![
                src.0[0].0,
                src.0[1].0,
                src.1 .0,
                u32::from(r < n_rows),
                r as u32,
                mults[r].0,
            ]
        })
        .collect();

    let n_packed_rows = 1usize << (diff.log_size - LOG_N_LANES);
    for (r, row_inputs) in rows.iter().enumerate() {
        let ro = interpret_row_with(&out.program, row_inputs, &oracle, &mut FastDeductionHost);

        for (c, hv) in diff.orig_rows[r].iter().enumerate() {
            assert_eq!(ro.columns[c], hv.0, "row {r} column {c}");
        }
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let mut w = 0usize;
        for field in diff.orig_lookup.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].to_array()[lane].0;
                assert_eq!(ro.lookup_words[w], hv, "row {r} lookup word {w} (+{k})");
                w += 1;
            }
        }
        let mut w = 0usize;
        for field in diff.orig_sub.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].as_array()[lane];
                assert_eq!(ro.sub_words[w], hv, "row {r} sub word {w} (+{k})");
                w += 1;
            }
        }
    }

    // Prove-accessor parity: the macro-generated igen rebuild from word-major
    // flats must reproduce the host `LookupData` byte-for-byte — fences the
    // 396-word field-list data entry against `LookupData` drift.
    let n_padded = 1usize << diff.log_size;
    let flat_from = |fields: &[Vec<PackedM31>]| -> Vec<u32> {
        let n_words: usize = fields.iter().map(|f| f.len() / n_packed_rows).sum();
        let mut flat = vec![0u32; n_words * n_padded];
        let mut w = 0usize;
        for field in fields {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                for r in 0..n_padded {
                    flat[w * n_padded + r] =
                        field[(r / N_LANES) * width + k].to_array()[r % N_LANES].0;
                }
                w += 1;
            }
        }
        flat
    };
    let lookup_flat = flat_from(&diff.orig_lookup);
    let igen = agg::interaction_gen_from_flat_lookup_words(diff.log_size, &lookup_flat, n_padded);
    assert_packed_field_bundles_eq(
        &agg::test_lookup_data_flat(&igen),
        &diff.orig_lookup,
        "pedersen_aggregator prove-accessor lookup_data",
    );

    // §6a leg: emitted JIT_LOGUP_DESCS → host mirror ≡ the generated writer.
    // The aggregator exercises SIGN-VARYING fractions (w18 yields negated
    // mid-stream; own-relation yield negated against mults_1).
    {
        let elements = test_lookup_elements();
        let (raw_ref, _claim) = igen.write_interaction_trace(&elements);
        assert_logup_descs_match_writer(
            "pedersen_aggregator_window_bits_18",
            agg::JIT_LOOKUP_FIELDS,
            agg::JIT_LOGUP_DESCS,
            raw_ref,
            &lookup_flat,
            n_padded,
            // No ENABLER mult source in the aggregator's descriptors (its mults
            // are flats columns); n_real is unused.
            n_padded,
            &elements,
        );
    }

    // GATE (c), pod builds only: the same program — 28 chained EC-round DeduceCalls
    // included — as an actual CUDA kernel against the fp256 device functions.
    let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
    assert_device_builtin_leg_matches_host(
        "pedersen_aggregator_window_bits_18",
        out.program,
        true,
        &rows,
        &addr_ids,
        &f252_values,
        &small_values,
        &host_rows,
        &diff.orig_lookup,
        &diff.orig_sub,
    );
}

/// Pins the builtin recordings' shapes to the `BuiltinLaneSpec` constants in
/// `jit_prove_backend.rs` — a mismatch there would otherwise only surface as a
/// silent runtime fallback. Also prints instruction counts (the prove-lane
/// governor input: raise `STWO_CUDA_WITNESS_JIT_MAX_INSTRS` past these on pod).
#[test]
fn builtin_lane_recording_shapes_match_specs() {
    use crate::witness::components::{
        blake_round as br, pedersen_aggregator_window_bits_18 as agg,
    };

    let a = agg::record_pedersen_aggregator_window_bits_18();
    eprintln!(
        "aggregator recording: {} instrs, {} cols, {} lookup, {} sub, {} inputs",
        a.program.n_instrs(),
        a.program.n_cols,
        a.program.n_lookup_words,
        a.program.n_sub_words,
        a.program.n_inputs,
    );
    assert_eq!(a.program.n_cols, 206);
    assert_eq!(a.program.n_lookup_words, 396);
    assert_eq!(a.program.n_sub_words, 3 + 4 + 28 * 72);
    assert_eq!(a.program.n_inputs, 6);
    let agg_field_words: usize = agg::JIT_LOOKUP_FIELDS.iter().map(|f| f.1).sum();
    assert_eq!(agg_field_words as u32, a.program.n_lookup_words);

    let b = br::record_blake_round();
    eprintln!(
        "blake_round recording: {} instrs, {} cols, {} lookup, {} sub, {} inputs",
        b.program.n_instrs(),
        b.program.n_cols,
        b.program.n_lookup_words,
        b.program.n_sub_words,
        b.program.n_inputs,
    );
    assert_eq!(b.program.n_cols, 212);
    assert_eq!(b.program.n_lookup_words, 850);
    assert_eq!(b.program.n_sub_words, 1 + 16 * 3 + 16 + 16 + 8 * 6);
    // 20, not 21: the blake body never reads its iota slot (20) — the seam
    // builds the canonical 21 and the launcher trims to the read extent.
    assert_eq!(b.program.n_inputs, 20);
    let blake_field_words: usize = br::JIT_LOOKUP_FIELDS.iter().map(|f| f.1).sum();
    assert_eq!(blake_field_words as u32, b.program.n_lookup_words);

    // PROVE-LANE LAUNCHABILITY for every fp256/Poseidon builtin lane: the launch path declines
    // (silent `None` → "launch unavailable" on pod) any program with mult tables
    // (counts flow through the SEPARATE feed kernel, never through MultPush) or
    // any program codegen can't lower. Both are properties of the RECORDING —
    // hardware-independent, so pin them here where a plain `cargo test` on a
    // laptop catches them before a pod ever spins up.
    use crate::witness::components::{
        cube_252 as cb, partial_ec_mul_generic as pg, partial_ec_mul_window_bits_18 as pw,
        poseidon_3_partial_rounds_chain as p3, poseidon_aggregator as pa, poseidon_builtin as pb,
        poseidon_full_round_chain as pf,
    };
    use crate::witness::jit_prove_backend::{
        BlakeRoundLane, BuiltinLaneSpec, Cube252Lane, PartialEcMulGenericLane, PartialEcMulW18Lane,
        PedersenAggregatorW18Lane, Poseidon3PartialRoundsChainLane, PoseidonAggregatorLane,
        PoseidonBuiltinLane, PoseidonFullRoundChainLane,
    };
    for (label, prog, needs_table) in [
        (
            "pedersen_aggregator_window_bits_18",
            &a.program,
            PedersenAggregatorW18Lane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "blake_round",
            &b.program,
            BlakeRoundLane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "partial_ec_mul_window_bits_18",
            &pw::record_partial_ec_mul_window_bits_18().program,
            PartialEcMulW18Lane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "partial_ec_mul_generic",
            &pg::record_partial_ec_mul_generic().program,
            PartialEcMulGenericLane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "cube_252",
            &cb::record_cube_252().program,
            Cube252Lane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "poseidon_builtin",
            &pb::record_poseidon_builtin().program,
            PoseidonBuiltinLane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "poseidon_aggregator",
            &pa::record_poseidon_aggregator().program,
            PoseidonAggregatorLane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "poseidon_full_round_chain",
            &pf::record_poseidon_full_round_chain().program,
            PoseidonFullRoundChainLane::NEEDS_PEDERSEN_TABLE,
        ),
        (
            "poseidon_3_partial_rounds_chain",
            &p3::record_poseidon_3_partial_rounds_chain().program,
            Poseidon3PartialRoundsChainLane::NEEDS_PEDERSEN_TABLE,
        ),
    ] {
        assert_eq!(
            prog.n_mult_tables, 0,
            "[{label}] records {} mult tables — the prove launch declines these",
            prog.n_mult_tables
        );
        assert!(
            stwo_backend_cuda::jit_witness::codegen::compile_witness_to_cuda_source(prog).is_some(),
            "[{label}] codegen returned None — the prove launch would silently fall back"
        );
        // Only deduce kinds 2/3 declare and read the module-local Pedersen table.
        // Felt/Poseidon fp256 helpers are self-contained and must not inherit a
        // multi-gigabyte table registration dependency merely by sharing math.
        use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};
        let reads_pedersen = prog.insts.iter().any(|inst| {
            WitnessOp::from_raw(inst.op) == Some(WitnessOp::DeduceCall)
                && DeduceKind::from_raw(inst.imm).is_some_and(|kind| {
                    matches!(
                        kind,
                        DeduceKind::PartialEcMulW18 | DeduceKind::PedersenPointsTableW18
                    )
                })
        });
        assert_eq!(
            needs_table, reads_pedersen,
            "[{label}] NEEDS_PEDERSEN_TABLE ({needs_table}) must equal table reads ({reads_pedersen})"
        );
    }
}

/// POD ORACLE LEGS (deduce kinds 2-11): the precompiled `stwo_wit_deduce_*` device
/// functions — the exact code the JIT kernels embed — vs the host `fast_deduction`
/// reference ([`FastDeductionHost`]). Kind 3 doubles as the device TABLE spot check:
/// the GPU-generated pedersen table vs the host `PEDERSEN_TABLE_18`, across every
/// section boundary. Kind 2 exercises the full W18 round (limb packing, table read,
/// `ec_add_affine` incl. `felt_inverse`, limb unpacking) on real curve points, with
/// chained rounds so each output feeds the next round's accumulator. No-op on stub
/// builds; runs as part of the normal suite on a pod build.
#[test]
fn stwo_wit_deduce_oracle_matches_fast_deduction() {
    use stwo_backend_cuda::jit_witness::interp::DeduceHost;

    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("deduce oracle: SKIPPED (stub build)");
        return;
    }
    // The oracle reads the device table for BOTH kinds; only the host-built
    // table is permitted (the GPU-generated one is what this leg falsified).
    assert!(
        crate::witness::jit_prove_backend::ensure_device_pedersen_table(),
        "host pedersen table registration failed on a CUDA build"
    );
    let mut host = FastDeductionHost;
    let run_oracle = |kind: u32, items: &[Vec<u32>], out_words: usize| -> Vec<Vec<u32>> {
        let in_words = items[0].len();
        let flat_in: Vec<u32> = items.iter().flatten().copied().collect();
        let mut flat_out = vec![0u32; items.len() * out_words];
        let rc = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_wit_deduce_oracle_run(
                kind,
                flat_in.as_ptr(),
                flat_out.as_mut_ptr(),
                items.len() as u32,
            )
        };
        assert_eq!(
            rc, 0,
            "oracle launch failed (kind {kind}, in_words {in_words})"
        );
        flat_out.chunks(out_words).map(<[u32]>::to_vec).collect()
    };

    // Deterministic LCG (no external randomness; reproducible failures).
    let mut state = 0x1234_5678u64;
    let mut next = |bound: u32| -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % bound
    };

    // ---- Kind 3: points-table reads, section boundaries + spread. --------------
    // Host layout [low0 | high0(16) | low2 | high2(16)]: 0, 3670016, 3670032, 7340048;
    // unpadded total 7340064.
    let mut table_rows: Vec<u32> = vec![
        0, 1, 262143, 262144, 3670015, 3670016, 3670031, 3670032, 3670033, 7340031, 7340047,
        7340048, 7340063,
    ];
    for _ in 0..243 {
        table_rows.push(next(7_340_064));
    }
    let items3: Vec<Vec<u32>> = table_rows.iter().map(|&r| vec![r]).collect();
    let dev3 = run_oracle(3, &items3, 56);
    let mut mismatches = 0usize;
    for (item, dev) in items3.iter().zip(&dev3) {
        let host_out = host.deduce(3, item);
        if dev != &host_out {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!(
                    "kind3 row {}: device {:?} host {:?}",
                    item[0], dev, host_out
                );
            }
        }
    }
    assert_eq!(mismatches, 0, "points-table oracle mismatches");
    eprintln!(
        "deduce oracle kind 3: PASS ({} rows incl. section boundaries)",
        items3.len()
    );

    // ---- Kind 2: W18 EC rounds on real curve points, chained. ------------------
    // Accumulators start from table points (valid curve points by construction);
    // each case then chains CHAIN_LEN rounds, device output feeding both device and
    // host next-round inputs (divergence localizes to the exact round).
    const CHAIN_LEN: usize = 4;
    let mut cases: Vec<Vec<u32>> = Vec::new();
    for case in 0..64 {
        let round = if case < 28 {
            case as u32
        } else {
            next(28 - CHAIN_LEN as u32)
        };
        let round = round.min(28 - CHAIN_LEN as u32);
        let acc_point = host.deduce(3, &[next(7_340_064)]);
        let mut args = vec![case as u32, round];
        args.extend((0..14).map(|_| next(1 << 18)));
        args.extend(&acc_point); // 56 limb words = both coordinates
        cases.push(args);
    }
    let mut current = cases;
    for step in 0..CHAIN_LEN {
        let dev2 = run_oracle(2, &current, 72);
        let mut mismatches = 0usize;
        for (item, dev) in current.iter().zip(&dev2) {
            let host_out = host.deduce(2, item);
            if dev != &host_out {
                mismatches += 1;
                if mismatches <= 3 {
                    eprintln!(
                        "kind2 step {step} (chain {}, round {}): device {:?} host {:?}",
                        item[0], item[1], dev, host_out
                    );
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "W18 round oracle mismatches at chain step {step}"
        );
        // Feed the outputs forward as the next round's inputs.
        current = dev2;
    }
    eprintln!("deduce oracle kind 2: PASS (64 cases x {CHAIN_LEN} chained rounds)");

    let compare =
        |kind: u32, items: &[Vec<u32>], out_words: usize, host: &mut FastDeductionHost| {
            let device = run_oracle(kind, items, out_words);
            for (row, (input, actual)) in items.iter().zip(&device).enumerate() {
                assert_eq!(
                    actual,
                    &host.deduce(kind, input),
                    "deduce oracle kind {kind} row {row}"
                );
            }
        };

    // Run the captured component input before the broader fp256 fuzz below as
    // a compact kind-11 control. The compact operation does not expose internal
    // writer intermediates such as `combination_37` (trace column 114).
    let captured_kind11 = std::env::var_os("STWO_POSEIDON_KIND11_REPRO_PATH").map(|path| {
        let words = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {path:?}: {error}"));
        let input = words
            .split_whitespace()
            .map(|word| {
                word.parse::<u32>()
                    .unwrap_or_else(|error| panic!("invalid u32 {word:?} in {path:?}: {error}"))
            })
            .collect::<Vec<_>>();
        assert_eq!(input.len(), 42, "kind-11 repro input in {path:?}");
        let actual = run_oracle(11, std::slice::from_ref(&input), 42).remove(0);
        let expected = host.deduce(11, &input);
        eprintln!(
            "deduce oracle kind 11 captured control from {path:?}: output[12] device={} host={}; compact output does not map to internal combination_37 (trace column 114)",
            actual[12], expected[12]
        );
        assert_eq!(actual, expected, "captured kind-11 row from {path:?}");
        (path, input)
    });

    // ---- Kinds 4-7: fp256 arithmetic, carry-heavy boundaries + fuzz. ----------
    use stwo_cairo_common::prover_types::cpu::{Felt252, P_FELTS};

    let zero = vec![0; 28];
    let mut one = zero.clone();
    one[0] = 1;
    let mut prime_minus_one = P_FELTS.to_vec();
    prime_minus_one[0] -= 1;
    let mut low_carry = zero.clone();
    low_carry[..21].fill(511);
    let alternating = (0..28)
        .map(|word| if word < 27 && word % 2 == 0 { 511 } else { 0 })
        .collect::<Vec<_>>();
    let boundaries = [
        zero.clone(),
        one.clone(),
        prime_minus_one,
        low_carry,
        alternating,
    ];
    let normalize = |limbs: Vec<u32>| {
        let limbs = limbs.into_iter().map(M31).collect::<Vec<_>>();
        let value = Felt252::from_limbs(&limbs) + Felt252::default();
        (0..28)
            .map(|word| value.get_m31(word).0)
            .collect::<Vec<_>>()
    };
    let is_canonical = |limbs: &[u32]| limbs.iter().rev().cmp(P_FELTS.iter().rev()).is_lt();
    let mut felt_pairs = Vec::new();
    for a in &boundaries {
        for b in &boundaries {
            let mut input = a.clone();
            input.extend(b);
            felt_pairs.push(input);
        }
    }
    for _ in 0..231 {
        let mut a = (0..28).map(|_| next(512)).collect::<Vec<_>>();
        let mut b = (0..28).map(|_| next(512)).collect::<Vec<_>>();
        a[27] = next(256);
        b[27] = next(256);
        // Force frequent low-limb carry/borrow ripples in half the cases.
        if next(2) == 0 {
            a[..8].fill(511);
            b[0] = 1;
        }
        let mut input = normalize(a);
        input.extend(normalize(b));
        felt_pairs.push(input);
    }
    assert!(felt_pairs.iter().all(|input| {
        input.len() == 56 && is_canonical(&input[..28]) && is_canonical(&input[28..])
    }));
    for kind in 4..=6 {
        compare(kind, &felt_pairs, 28, &mut host);
    }
    let division_pairs = felt_pairs
        .iter()
        .filter(|input| input[28..] != zero[..])
        .cloned()
        .collect::<Vec<_>>();
    assert!(division_pairs
        .iter()
        .all(|input| is_canonical(&input[28..]) && input[28..] != zero[..]));
    compare(7, &division_pairs, 28, &mut host);
    eprintln!(
        "deduce oracle kinds 4-7: PASS ({} carry-heavy/fuzz pairs)",
        felt_pairs.len()
    );

    // ---- Kinds 8-11: Cairo Poseidon W27 primitives. ---------------------------
    // Round-key outputs seed every arithmetic case, so the device constant table,
    // Width27 regrouping, cube, and both chain transitions are checked together.
    let rounds = (0..35).map(|round| vec![round]).collect::<Vec<_>>();
    compare(8, &rounds, 30, &mut host);

    let keys = rounds
        .iter()
        .map(|round| host.deduce(8, round))
        .collect::<Vec<_>>();
    let cubes = keys
        .iter()
        .flat_map(|row| (0..3).map(move |felt| row[felt * 10..felt * 10 + 10].to_vec()))
        .collect::<Vec<_>>();
    compare(9, &cubes, 10, &mut host);

    let full = keys
        .iter()
        .enumerate()
        .map(|(round, keys)| {
            let mut input = vec![round as u32, round as u32];
            input.extend(keys);
            input
        })
        .collect::<Vec<_>>();
    compare(10, &full, 32, &mut host);

    let mut partial = keys
        .iter()
        .enumerate()
        .map(|(round, keys)| {
            let mut input = vec![round as u32, round as u32];
            input.extend(keys);
            input.extend(&host.deduce(9, &keys[..10]));
            input
        })
        .collect::<Vec<_>>();
    let captured_case = captured_kind11.map(|(path, input)| {
        let case = partial.len();
        partial.push(input);
        eprintln!(
            "deduce oracle kind 11: appended captured compact-control case {case} from {path:?}; it does not exercise internal combination_37 (trace column 114)"
        );
        case
    });
    let device_partial = run_oracle(11, &partial, 42);
    for (row, (input, actual)) in partial.iter().zip(&device_partial).enumerate() {
        let expected = host.deduce(11, input);
        if captured_case == Some(row) {
            eprintln!(
                "deduce oracle kind 11 captured control: output[12] device={} host={}; not internal combination_37 (trace column 114)",
                actual[12], expected[12]
            );
        }
        assert_eq!(actual, &expected, "deduce oracle kind 11 row {row}");
    }
    eprintln!(
        "deduce oracle kinds 8-11: PASS (35 generated Poseidon rounds{})",
        if captured_case.is_some() {
            " + captured row-0 case"
        } else {
            ""
        }
    );
}

/// H100 controls for the exact source operands used by the generated
/// `combination_37` schedule. Each primitive runs independently through the
/// low-pressure generic oracle; the resident trace audit remains the definitive
/// failing schedule-context gate.
#[test]
fn poseidon_combination_37_exact_source_primitives_match_host() {
    use stwo_backend_cuda::jit_witness::interp::DeduceHost;

    const ROW0: [u32; 42] = [
        0, 4, 50414066, 128588089, 120633038, 63732151, 97038777, 32313651, 132029487, 122547581,
        103664913, 254, 70246675, 35346168, 94916093, 40649707, 36525582, 74717629, 46705327,
        50424067, 58946647, 39, 87784937, 111781535, 84088807, 86541649, 127250820, 6346412,
        29906354, 123707764, 53944726, 252, 22813865, 35298563, 79701982, 108932941, 43138495,
        66822320, 50165977, 5364451, 38958708, 247,
    ];

    struct CapturingDeduceHost {
        calls: Vec<(u32, Vec<u32>, Vec<u32>)>,
    }
    impl DeduceHost for CapturingDeduceHost {
        fn deduce(&mut self, kind: u32, args: &[u32]) -> Vec<u32> {
            let output = FastDeductionHost.deduce(kind, args);
            self.calls.push((kind, args.to_vec(), output.clone()));
            output
        }
    }

    let recording = crate::witness::components::poseidon_3_partial_rounds_chain::record_poseidon_3_partial_rounds_chain();
    let mut inputs = ROW0.to_vec();
    inputs.push(1); // enabler; this recording does not read iota
    assert_eq!(inputs.len(), recording.program.n_inputs as usize);
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };
    let mut capture = CapturingDeduceHost { calls: Vec::new() };
    let row = stwo_backend_cuda::jit_witness::interp::interpret_row_with(
        &recording.program,
        &inputs,
        &oracle,
        &mut capture,
    );
    assert_eq!(row.columns[114], 17_375_170, "captured SIMD column 114");

    const KINDS: [u32; 12] = [6, 4, 6, 4, 6, 4, 6, 4, 6, 5, 6, 4];
    let chain = capture
        .calls
        .get(17..29)
        .expect("recording must contain combination_37 calls 17..28");
    assert_eq!(chain.len(), KINDS.len());
    for (stage, ((kind, input, _), expected_kind)) in chain.iter().zip(KINDS).enumerate() {
        assert_eq!(*kind, expected_kind, "combination_37 call ordinal {stage}");
        assert_eq!(input.len(), 56, "combination_37 stage {stage} args");
    }
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("combination_37 device primitives: SKIPPED (stub build)");
        return;
    }
    for (stage, ((kind, input, expected), expected_kind)) in chain.iter().zip(KINDS).enumerate() {
        debug_assert_eq!(*kind, expected_kind);
        let mut output = vec![0; 28];
        let rc = unsafe {
            stwo_backend_cuda_kernels::raw::stwo_wit_deduce_oracle_run(
                *kind,
                input.as_ptr(),
                output.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rc, 0, "combination_37 primitive stage {stage} launch");
        assert_eq!(&output, expected, "combination_37 primitive stage {stage}");
    }
    eprintln!("combination_37 exact-source primitive controls: PASS (12/12)");
}

/// Opt-in H100 reproducer: launch the complete captured row through the strict
/// embedded-AOT recorded-program path. Pre-fix, column 114 is 17375169 on the
/// device and 17375170 in the host recording interpreter.
#[test]
#[ignore = "requires an H100 CUDA build with the embedded AOT witness pack"]
fn poseidon_combination_37_strict_aot_captured_row() {
    const ROW0: [u32; 42] = [
        0, 4, 50414066, 128588089, 120633038, 63732151, 97038777, 32313651, 132029487, 122547581,
        103664913, 254, 70246675, 35346168, 94916093, 40649707, 36525582, 74717629, 46705327,
        50424067, 58946647, 39, 87784937, 111781535, 84088807, 86541649, 127250820, 6346412,
        29906354, 123707764, 53944726, 252, 22813865, 35298563, 79701982, 108932941, 43138495,
        66822320, 50165977, 5364451, 38958708, 247,
    ];
    const LABEL: &str = "poseidon_3_partial_rounds_chain";

    let recording = crate::witness::components::poseidon_3_partial_rounds_chain::record_poseidon_3_partial_rounds_chain();
    let mut row_inputs = ROW0.to_vec();
    row_inputs.push(1); // enabler; this recording does not read iota
    assert_eq!(row_inputs.len(), recording.program.n_inputs as usize);
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };
    let host = stwo_backend_cuda::jit_witness::interp::interpret_row_with(
        &recording.program,
        &row_inputs,
        &oracle,
        &mut FastDeductionHost,
    );
    assert_eq!(host.columns[114], 17_375_170, "captured host column 114");

    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("strict-AOT captured row: SKIPPED (stub build)");
        return;
    }

    let rows = vec![row_inputs; N_LANES];
    let input_cols = (0..recording.program.n_inputs as usize)
        .map(|slot| rows.iter().map(|row| row[slot]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert!(
        crate::witness::jit_prove_backend::ensure_device_pedersen_table(),
        "strict-AOT captured row: device Pedersen table registration failed"
    );
    stwo_backend_cuda::aot::require_loaded_kernels();
    stwo_backend_cuda::aot::reset_runtime_stats();
    stwo_backend_cuda::jit_witness::register_recorded_program(LABEL, recording.program);
    let tables =
        stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(&[0], &[[0; 8]], &[0]);
    let (device_columns, _lookup_device, lookup_flat, _sub_device, sub_flat) =
        stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
            LABEL,
            &input_cols,
            &tables,
            true,
            true,
        )
        .expect("strict-AOT captured-row launch unavailable");

    let stats = stwo_backend_cuda::aot::runtime_stats();
    assert_eq!(stats.aot_misses, 0, "strict AOT miss");
    assert_eq!(stats.runtime_loads, 0, "runtime compilation");
    assert_eq!(stats.runtime_cache_hits, 0, "runtime cache hit");
    assert_eq!(stats.strict_rejections, 0, "strict AOT rejection");
    assert_eq!(
        stats.aot_loads + stats.aot_cache_hits,
        1,
        "launch did not use exactly one embedded AOT kernel"
    );
    assert_eq!(device_columns.len(), host.columns.len(), "column count");
    for (column, device_column) in device_columns.iter().enumerate() {
        for (row, value) in device_column.to_vec().iter().enumerate() {
            assert_eq!(
                value.0, host.columns[column],
                "strict-AOT poseidon device col {column} row {row}"
            );
        }
    }
    assert_eq!(
        lookup_flat.len(),
        host.lookup_words.len() * N_LANES,
        "lookup flat length"
    );
    for (word, expected) in host.lookup_words.iter().enumerate() {
        for row in 0..N_LANES {
            assert_eq!(
                lookup_flat[word * N_LANES + row],
                *expected,
                "strict-AOT poseidon lookup word {word} row {row}"
            );
        }
    }
    assert_eq!(
        sub_flat.len(),
        host.sub_words.len() * N_LANES,
        "sub flat length"
    );
    for (word, expected) in host.sub_words.iter().enumerate() {
        for row in 0..N_LANES {
            assert_eq!(
                sub_flat[word * N_LANES + row],
                *expected,
                "strict-AOT poseidon sub word {word} row {row}"
            );
        }
    }
}

/// NVRTC-path discriminator for the captured row. A dead constant changes only
/// the semantic hash/AOT key, forcing runtime compilation without changing outputs.
#[test]
#[ignore = "requires an H100 CUDA build with NVRTC"]
fn poseidon_combination_37_nvrtc_captured_row() {
    use stwo_backend_cuda::jit_witness::isa::{WitnessInst, WitnessOp};

    const ROW0: [u32; 42] = [
        0, 4, 50414066, 128588089, 120633038, 63732151, 97038777, 32313651, 132029487, 122547581,
        103664913, 254, 70246675, 35346168, 94916093, 40649707, 36525582, 74717629, 46705327,
        50424067, 58946647, 39, 87784937, 111781535, 84088807, 86541649, 127250820, 6346412,
        29906354, 123707764, 53944726, 252, 22813865, 35298563, 79701982, 108932941, 43138495,
        66822320, 50165977, 5364451, 38958708, 247,
    ];
    const LABEL: &str = "poseidon_3_partial_rounds_chain_nvrtc_discriminator";

    let mut program = crate::witness::components::poseidon_3_partial_rounds_chain::record_poseidon_3_partial_rounds_chain().program;
    let dead_reg = u16::try_from(program.n_regs).expect("witness register index exceeds u16");
    program.insts.push(WitnessInst::new(
        WitnessOp::Const,
        dead_reg,
        0,
        0,
        0xC011_A37,
    ));
    program.n_regs += 1;
    let mut row_inputs = ROW0.to_vec();
    row_inputs.push(1);
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };
    let host = stwo_backend_cuda::jit_witness::interp::interpret_row_with(
        &program,
        &row_inputs,
        &oracle,
        &mut FastDeductionHost,
    );
    assert_eq!(host.columns[114], 17_375_170, "captured host column 114");
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("NVRTC captured row: SKIPPED (stub build)");
        return;
    }

    let rows = vec![row_inputs; N_LANES];
    let input_cols = (0..program.n_inputs as usize)
        .map(|slot| rows.iter().map(|row| row[slot]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert!(
        crate::witness::jit_prove_backend::ensure_device_pedersen_table(),
        "NVRTC captured row: device Pedersen table registration failed"
    );
    unsafe {
        std::env::set_var("STWO_CUDA_WITNESS_JIT_MAX_INSTRS", "8192");
        stwo_backend_cuda_kernels::raw::stwo_cuda_jit_set_require_aot(false);
    }
    stwo_backend_cuda::aot::reset_runtime_stats();
    stwo_backend_cuda::jit_witness::register_recorded_program(LABEL, program);
    let tables =
        stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(&[0], &[[0; 8]], &[0]);
    let (device_columns, ..) = stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
        LABEL,
        &input_cols,
        &tables,
        false,
        false,
    )
    .expect("NVRTC captured-row launch unavailable");
    let stats = stwo_backend_cuda::aot::runtime_stats();
    assert_eq!(stats.aot_loads, 0, "unexpected AOT load");
    assert_eq!(stats.aot_cache_hits, 0, "unexpected AOT cache hit");
    assert_eq!(stats.aot_misses, 1, "perturbed key did not miss AOT");
    assert_eq!(
        stats.runtime_loads, 1,
        "perturbed key did not load via NVRTC"
    );
    assert_eq!(stats.runtime_cache_hits, 0, "unexpected runtime cache hit");
    assert_eq!(stats.strict_rejections, 0, "unexpected strict rejection");
    for (row, value) in device_columns[114].to_vec().iter().enumerate() {
        assert_eq!(
            value.0, host.columns[114],
            "NVRTC poseidon device col 114 row {row}"
        );
    }
}

/// Test-only schedule-context mirror. Inserts a low-W27-word mirror immediately
/// after one selected felt deduce while preserving every original instruction.
#[test]
#[ignore = "requires an H100 CUDA build with NVRTC"]
fn poseidon_combination_37_nvrtc_stage_mirror() {
    use stwo_backend_cuda::jit_witness::isa::{WitnessInst, WitnessOp};

    const ROW0: [u32; 42] = [
        0, 4, 50414066, 128588089, 120633038, 63732151, 97038777, 32313651, 132029487, 122547581,
        103664913, 254, 70246675, 35346168, 94916093, 40649707, 36525582, 74717629, 46705327,
        50424067, 58946647, 39, 87784937, 111781535, 84088807, 86541649, 127250820, 6346412,
        29906354, 123707764, 53944726, 252, 22813865, 35298563, 79701982, 108932941, 43138495,
        66822320, 50165977, 5364451, 38958708, 247,
    ];
    let selected = std::env::var("STWO_POSEIDON_MIRROR_CALL_ORDINAL")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("mirror ordinal must be usize")
        })
        .unwrap_or(28);
    assert!((17..=28).contains(&selected), "mirror ordinal {selected}");

    let mut program = crate::witness::components::poseidon_3_partial_rounds_chain::record_poseidon_3_partial_rounds_chain().program;
    let mut ordinal = 0usize;
    let call_index = program
        .insts
        .iter()
        .position(|inst| {
            if WitnessOp::from_raw(inst.op) != Some(WitnessOp::DeduceCall) {
                return false;
            }
            let matches = ordinal == selected;
            ordinal += 1;
            matches
        })
        .expect("selected deduce call missing");
    let call = program.insts[call_index];
    assert_eq!(call.b, 28, "selected call must produce one W9 felt");
    let reg = u16::try_from(program.n_regs).expect("witness register index exceeds u16");
    let mirror_col = program.n_cols;
    let extra = vec![
        WitnessInst::new(WitnessOp::Const, reg, 0, 0, 512),
        WitnessInst::new(WitnessOp::Const, reg + 1, 0, 0, 262_144),
        WitnessInst::new(
            WitnessOp::M31Mul,
            reg + 2,
            call.dst as u32 + 1,
            reg as u32,
            0,
        ),
        WitnessInst::new(
            WitnessOp::M31Add,
            reg + 3,
            call.dst as u32,
            reg as u32 + 2,
            0,
        ),
        WitnessInst::new(
            WitnessOp::M31Mul,
            reg + 4,
            call.dst as u32 + 2,
            reg as u32 + 1,
            0,
        ),
        WitnessInst::new(
            WitnessOp::M31Add,
            reg + 5,
            reg as u32 + 3,
            reg as u32 + 4,
            0,
        ),
        WitnessInst::new(WitnessOp::ColWrite, 0, reg as u32 + 5, 0, mirror_col),
    ];
    program.insts.splice(call_index + 1..call_index + 1, extra);
    program.n_regs += 6;
    program.n_cols += 1;

    let mut row_inputs = ROW0.to_vec();
    row_inputs.push(1);
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };
    let host = stwo_backend_cuda::jit_witness::interp::interpret_row_with(
        &program,
        &row_inputs,
        &oracle,
        &mut FastDeductionHost,
    );
    let expected = host.columns[mirror_col as usize];
    if selected == 28 {
        assert_eq!(expected, 17_375_170, "final combination_37 mirror");
    }
    if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
        eprintln!("NVRTC stage mirror ordinal {selected}: host={expected}, device SKIPPED");
        return;
    }

    let rows = vec![row_inputs; N_LANES];
    let input_cols = (0..program.n_inputs as usize)
        .map(|slot| rows.iter().map(|row| row[slot]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert!(
        crate::witness::jit_prove_backend::ensure_device_pedersen_table(),
        "NVRTC stage mirror: device Pedersen table registration failed"
    );
    unsafe {
        std::env::set_var("STWO_CUDA_WITNESS_JIT_MAX_INSTRS", "8192");
        stwo_backend_cuda_kernels::raw::stwo_cuda_jit_set_require_aot(false);
    }
    stwo_backend_cuda::aot::reset_runtime_stats();
    let label = format!("poseidon_3_partial_rounds_chain_nvrtc_mirror_{selected}");
    stwo_backend_cuda::jit_witness::register_recorded_program(&label, program);
    let tables =
        stwo_backend_cuda::exec_tables::DeviceExecutionTables::upload(&[0], &[[0; 8]], &[0]);
    let (device_columns, ..) = stwo_backend_cuda::exec_tables::launch_recorded_builtin_for_prove(
        &label,
        &input_cols,
        &tables,
        false,
        false,
    )
    .expect("NVRTC stage-mirror launch unavailable");
    let stats = stwo_backend_cuda::aot::runtime_stats();
    assert_eq!(
        stats.aot_loads + stats.aot_cache_hits,
        0,
        "unexpected AOT provenance"
    );
    assert_eq!(stats.aot_misses, 1, "mirrored key did not miss AOT");
    assert_eq!(
        stats.runtime_loads + stats.runtime_cache_hits,
        1,
        "mirrored key did not use runtime compiler/cache"
    );
    assert_eq!(stats.strict_rejections, 0, "unexpected strict rejection");
    for (row, value) in device_columns[mirror_col as usize]
        .to_vec()
        .iter()
        .enumerate()
    {
        assert_eq!(
            value.0, expected,
            "NVRTC schedule-context mirror call {selected} row {row}"
        );
    }
}

// ---------------- fp256/EC flagship: partial_ec_mul_window_bits_18 ------------------

/// Shared fixture prep for the w18 gates: run the pedersen fixture, feed the
/// aggregator (pedersen_builtin -> aggregator write_trace -> w18 inputs), and
/// return (w18 gen, points/rc states, padded packed inputs, n_rows, memory).
#[allow(clippy::type_complexity)]
fn w18_fixture() -> (
    Vec<crate::witness::components::partial_ec_mul_window_bits_18::PackedInputType>,
    usize,
    crate::witness::components::pedersen_points_table_window_bits_18::ClaimGenerator,
    crate::witness::components::range_check_9_9::ClaimGenerator,
    crate::witness::components::range_check_20::ClaimGenerator,
    (Vec<u32>, Vec<[u32; 8]>, Vec<u128>),
) {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    let compiled = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run_and_adapt pedersen fixture");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let addr_ids: Vec<u32> = memory.address_to_id.iter().map(|e| e.0).collect();
    let f252_values = memory.f252_values.clone();
    let small_values = memory.small_values.clone();
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "partial_ec_mul_window_bits_18",
        "pedersen_points_table_window_bits_18",
        "range_check_9_9",
        "range_check_20",
        "pedersen_aggregator_window_bits_18",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_8",
        "pedersen_builtin",
    ] {
        set.insert(c);
    }
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    // Production feed chain: pedersen_builtin -> aggregator mults; aggregator
    // write_trace -> w18 inputs (28 chained EC rounds per aggregator row).
    {
        let pb = cg.pedersen_builtin.take().expect("pedersen_builtin");
        let mem_addr = cg.memory_address_to_id.as_ref().expect("mem addr");
        let agg_state = cg
            .pedersen_aggregator_window_bits_18
            .as_ref()
            .expect("aggregator state");
        let _ = pb.write_trace(mem_addr, agg_state);
    }
    {
        let agg = cg
            .pedersen_aggregator_window_bits_18
            .take()
            .expect("aggregator populated");
        let mem_big = cg.memory_id_to_big.as_ref().expect("mem big");
        let rc8 = cg.range_check_8.as_ref().expect("range_check_8");
        let w18_state = cg
            .partial_ec_mul_window_bits_18
            .as_ref()
            .expect("w18 state");
        let _ = agg.write_trace(mem_big, rc8, w18_state);
    }
    let gen = cg.partial_ec_mul_window_bits_18.expect("w18 populated");
    let pts = cg
        .pedersen_points_table_window_bits_18
        .expect("points table state");
    let rc99 = cg.range_check_9_9.expect("range_check_9_9");
    let rc20 = cg.range_check_20.expect("range_check_20");

    let mut packed = gen.packed_inputs.into_inner().unwrap();
    assert!(!packed.is_empty(), "fixture fed no w18 inputs");
    assert!(gen.remainder_inputs.lock().unwrap().is_empty());
    let n_vec_rows = packed.len();
    let n_rows = n_vec_rows * N_LANES;
    let packed_size = n_vec_rows.next_power_of_two();
    packed.resize(packed_size, *packed.first().unwrap());
    (
        packed,
        n_rows,
        pts,
        rc99,
        rc20,
        (addr_ids, f252_values, small_values),
    )
}

/// Gate (a) for `partial_ec_mul_window_bits_18` — the fp256/EC flagship: felt
/// DeduceKinds 4-7 (inline slope arithmetic incl. division), the points-table
/// deduce, and the u32 lane, in one 297-column writer.
#[test]
fn partial_ec_mul_w18_generic_simd_byte_identical() {
    use crate::witness::components::partial_ec_mul_window_bits_18 as m;
    let (packed, n_rows, pts, rc99, rc20, _mem) = w18_fixture();
    assert_generic_diff_byte_identical!(m::generic_simd_diff(packed, n_rows, &pts, &rc99, &rc20));
}

/// The w18 recording manifest: FULLY recorded — zero poisons; the EC round's
/// felt arithmetic and table read all lower to real DeduceCalls.
#[test]
fn partial_ec_mul_w18_recording_poison_manifest() {
    use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};

    use crate::witness::components::partial_ec_mul_window_bits_18 as m;
    let rec = m::record_partial_ec_mul_window_bits_18();
    assert!(rec.poison_ops.is_empty(), "poisons: {:?}", rec.poison_ops);
    assert!(rec.poisoned_cols.is_empty());
    assert!(rec.poisoned_lookup_words.is_empty());
    assert!(rec.poisoned_sub_words.is_empty());
    let count = |k: DeduceKind| {
        rec.program
            .insts
            .iter()
            .filter(|i| i.op == WitnessOp::DeduceCall as u8 && i.imm == k as u32)
            .count()
    };
    eprintln!(
        "w18 deduces: points={} add={} sub={} mul={} div={} instrs={}",
        count(DeduceKind::PedersenPointsTableW18),
        count(DeduceKind::FeltAdd),
        count(DeduceKind::FeltSub),
        count(DeduceKind::FeltMul),
        count(DeduceKind::FeltDiv),
        rec.program.n_instrs(),
    );
    assert_eq!(count(DeduceKind::PedersenPointsTableW18), 1);
    // The EC-add round body: slope = (y2-y1)/(x2-x1) (1 div, subs), then
    // x3/y3 via 2 muls and more subs — pinned from the first green recording.
    assert_eq!(count(DeduceKind::FeltAdd), 0);
    assert_eq!(count(DeduceKind::FeltSub), 6);
    assert_eq!(count(DeduceKind::FeltMul), 2);
    assert_eq!(count(DeduceKind::FeltDiv), 1);
}

/// GATE (b)+(c) for w18: the recorded program interpreted with the
/// fast_deduction/Felt252 reference host is byte-identical to the host writer
/// everywhere; on a pod build the same program then runs as a CUDA kernel.
#[test]
fn partial_ec_mul_w18_recording_interpreter_matches_host() {
    use stwo_backend_cuda::jit_witness::interp::interpret_row_with;

    use crate::witness::components::partial_ec_mul_window_bits_18 as m;
    let (packed, n_rows, pts, rc99, rc20, (addr_ids, f252_values, small_values)) = w18_fixture();
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &pts, &rc99, &rc20);

    let out = m::record_partial_ec_mul_window_bits_18();
    assert!(out.poison_ops.is_empty(), "poisons: {:?}", out.poison_ops);

    // The w18 body reads NO memory tables (its felts arrive as input limbs);
    // any table read reaching the oracle is a recording bug.
    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };

    // Slot layout: flat input words 0..72 (in.0, in.1, 14 windows, acc0 limbs,
    // acc1 limbs), enabler 72, iota 73.
    let n_padded = packed.len() * N_LANES;
    let rows: Vec<Vec<u32>> = (0..n_padded)
        .map(|r| {
            let (pr, lane) = (r / N_LANES, r % N_LANES);
            let p = &packed[pr];
            let mut row = vec![p.0.to_array()[lane].0, p.1.to_array()[lane].0];
            row.extend(p.2 .0.iter().map(|w| w.to_array()[lane].0));
            for f in &p.2 .1 {
                row.extend((0..28).map(|i| f.get_m31(i).to_array()[lane].0));
            }
            row.push(u32::from(r < n_rows)); // enabler
            row.push(r as u32); // iota
            row
        })
        .collect();

    let n_packed_rows = n_padded / N_LANES;
    for (r, row_inputs) in rows.iter().enumerate() {
        let ro = interpret_row_with(&out.program, row_inputs, &oracle, &mut FastDeductionHost);
        for (c, hv) in diff.orig_rows[r].iter().enumerate() {
            assert_eq!(ro.columns[c], hv.0, "row {r} column {c}");
        }
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let mut w = 0usize;
        for field in diff.orig_lookup.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].to_array()[lane].0;
                assert_eq!(ro.lookup_words[w], hv, "row {r} lookup word {w} (+{k})");
                w += 1;
            }
        }
        let mut w = 0usize;
        for field in diff.orig_sub.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].as_array()[lane];
                assert_eq!(ro.sub_words[w], hv, "row {r} sub word {w} (+{k})");
                w += 1;
            }
        }
    }

    // GATE (c), pod builds only: the same program as an actual CUDA kernel.
    let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
    assert_device_builtin_leg_matches_host(
        "partial_ec_mul_window_bits_18",
        out.program,
        true,
        &rows,
        &addr_ids,
        &f252_values,
        &small_values,
        &host_rows,
        &diff.orig_lookup,
        &diff.orig_sub,
    );
}

// ---------------- u32-cohort unlocks (mul_opcode, add_ap_opcode, blake_g) -----------

/// Gate (a) for `mul_opcode` (u32-heavy opcode, unlocked by the u32 trait lane).
#[test]
fn mul_opcode_generic_simd_byte_identical() {
    use crate::witness::components::mul_opcode as m;
    let cg = fill_fixture(&[
        "mul_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
        "range_check_20",
    ]);
    let gen = cg.mul_opcode.expect("mul_opcode populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let vi = cg.verify_instruction.expect("verify_instruction");
    let rc20 = cg.range_check_20.expect("range_check_20");
    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &vi, &rc20,
    ));
}

/// Gate (a) for `add_ap_opcode` (u32-family opcode, unlocked by the u32 trait lane).
#[test]
fn add_ap_opcode_generic_simd_byte_identical() {
    use crate::witness::components::add_ap_opcode as m;
    let cg = fill_fixture(&[
        "add_ap_opcode",
        "memory_address_to_id",
        "memory_id_to_big",
        "verify_instruction",
        "range_check_18",
        "range_check_11",
    ]);
    let gen = cg.add_ap_opcode.expect("add_ap_opcode populated");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let vi = cg.verify_instruction.expect("verify_instruction");
    let rc18 = cg.range_check_18.expect("range_check_18");
    let rc11 = cg.range_check_11.expect("range_check_11");
    let (_, packed, n_rows) = pack_pilot_inputs(gen.inputs);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &mem_addr, &mem_big, &vi, &rc18, &rc11,
    ));
}

/// Gate (a) for `blake_g` (the g-function component itself — full u32 body).
/// Synthetic inputs: the two host lanes must be byte-identical on ANY words.
#[test]
fn blake_g_generic_simd_byte_identical() {
    use stwo_cairo_common::prover_types::cpu::UInt32;

    use crate::witness::components::blake_g as m;
    let cg = fill_fixture(&[
        "blake_g",
        "verify_bitwise_xor_8",
        "verify_bitwise_xor_12",
        "verify_bitwise_xor_4",
        "verify_bitwise_xor_7",
        "verify_bitwise_xor_9",
    ]);
    let xor8 = cg.verify_bitwise_xor_8.expect("xor8");
    let xor12 = cg.verify_bitwise_xor_12.expect("xor12");
    let xor4 = cg.verify_bitwise_xor_4.expect("xor4");
    let xor7 = cg.verify_bitwise_xor_7.expect("xor7");
    let xor9 = cg.verify_bitwise_xor_9.expect("xor9");

    let inputs: Vec<m::InputType> = (0..48u32)
        .map(|i| std::array::from_fn(|j| UInt32::from(0x9E37_79B9u32.wrapping_mul(i + j as u32))))
        .collect();
    let n_rows = inputs.len();
    let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    let mut padded = inputs;
    padded.resize(size, *padded.first().unwrap());
    let packed = pack_values(&padded);
    assert_generic_diff_byte_identical!(m::generic_simd_diff(
        packed, n_rows, &xor8, &xor12, &xor4, &xor7, &xor9,
    ));
}

// ---------------- fp256/EC final boss: partial_ec_mul_generic -----------------------

/// Shared fixture prep for the generic gates: the all-builtins fixture, fed
/// through the production chain (ec_op_builtin write_trace -> generic inputs).
#[allow(clippy::type_complexity)]
fn partial_ec_mul_generic_fixture() -> (
    Vec<crate::witness::components::partial_ec_mul_generic::PackedInputType>,
    usize,
    crate::witness::components::range_check_8::ClaimGenerator,
    crate::witness::components::range_check_9_9::ClaimGenerator,
    crate::witness::components::range_check_20::ClaimGenerator,
    (Vec<u32>, Vec<[u32; 8]>, Vec<u128>),
) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/test_prove_verify_all_builtins/prover_input.json");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    let input: ProverInput = serde_json::from_str(&json).expect("deserialize ProverInput");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let addr_ids: Vec<u32> = memory.address_to_id.iter().map(|e| e.0).collect();
    let f252_values = memory.f252_values.clone();
    let small_values = memory.small_values.clone();
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "partial_ec_mul_generic",
        "range_check_8",
        "range_check_9_9",
        "range_check_20",
        "ec_op_builtin",
        "memory_address_to_id",
        "memory_id_to_big",
    ] {
        set.insert(c);
    }
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    // Production feed: ec_op_builtin write_trace -> partial_ec_mul_generic inputs.
    {
        let ec_op = cg.ec_op_builtin.take().expect("ec_op_builtin populated");
        let mem_addr = cg.memory_address_to_id.as_ref().expect("mem addr");
        let mem_big = cg.memory_id_to_big.as_ref().expect("mem big");
        let rc8 = cg.range_check_8.as_ref().expect("range_check_8");
        let generic_state = cg.partial_ec_mul_generic.as_ref().expect("generic state");
        let _ = ec_op.write_trace(mem_addr, mem_big, rc8, generic_state);
    }
    let gen = cg.partial_ec_mul_generic.expect("generic populated");
    let rc8 = cg.range_check_8.expect("range_check_8");
    let rc99 = cg.range_check_9_9.expect("range_check_9_9");
    let rc20 = cg.range_check_20.expect("range_check_20");

    let mut packed = gen.packed_inputs.into_inner().unwrap();
    assert!(!packed.is_empty(), "fixture fed no generic inputs");
    assert!(gen.remainder_inputs.lock().unwrap().is_empty());
    let n_vec_rows = packed.len();
    let n_rows = n_vec_rows * N_LANES;
    let packed_size = n_vec_rows.next_power_of_two();
    packed.resize(packed_size, *packed.first().unwrap());
    (
        packed,
        n_rows,
        rc8,
        rc99,
        rc20,
        (addr_ids, f252_values, small_values),
    )
}

/// Gate (a) for `partial_ec_mul_generic` — the largest fp256 writer (624 cols):
/// felt arithmetic + the W27 input lane + the u32 family in one body.
#[test]
fn partial_ec_mul_generic_generic_simd_byte_identical() {
    use crate::witness::components::partial_ec_mul_generic as m;
    let (packed, n_rows, rc8, rc99, rc20, _mem) = partial_ec_mul_generic_fixture();
    assert_generic_diff_byte_identical!(m::generic_simd_diff(packed, n_rows, &rc8, &rc99, &rc20));
}

/// The generic recording manifest: fully recorded — zero poisons.
#[test]
fn partial_ec_mul_generic_recording_poison_manifest() {
    use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};

    use crate::witness::components::partial_ec_mul_generic as m;
    let rec = m::record_partial_ec_mul_generic();
    assert!(rec.poison_ops.is_empty(), "poisons: {:?}", rec.poison_ops);
    assert!(rec.poisoned_cols.is_empty());
    assert!(rec.poisoned_lookup_words.is_empty());
    assert!(rec.poisoned_sub_words.is_empty());
    let count = |k: DeduceKind| {
        rec.program
            .insts
            .iter()
            .filter(|i| i.op == WitnessOp::DeduceCall as u8 && i.imm == k as u32)
            .count()
    };
    eprintln!(
        "generic deduces: add={} sub={} mul={} div={} instrs={}",
        count(DeduceKind::FeltAdd),
        count(DeduceKind::FeltSub),
        count(DeduceKind::FeltMul),
        count(DeduceKind::FeltDiv),
        rec.program.n_instrs(),
    );
    // Two EC adds per round body (P + Q and the doubling chain) — pinned from
    // the first green recording.
    assert_eq!(count(DeduceKind::FeltAdd), 2);
    assert_eq!(count(DeduceKind::FeltSub), 10);
    assert_eq!(count(DeduceKind::FeltMul), 6);
    assert_eq!(count(DeduceKind::FeltDiv), 2);
}

/// GATE (b)+(c) for generic: recorded program vs host writer via the
/// interpreter (Felt252 reference host); pod builds run the CUDA kernel too.
#[test]
fn partial_ec_mul_generic_recording_interpreter_matches_host() {
    use stwo_backend_cuda::jit_witness::interp::interpret_row_with;

    use crate::witness::components::partial_ec_mul_generic as m;
    let (packed, n_rows, rc8, rc99, rc20, (addr_ids, f252_values, small_values)) =
        partial_ec_mul_generic_fixture();
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &rc8, &rc99, &rc20);

    let out = m::record_partial_ec_mul_generic();
    assert!(out.poison_ops.is_empty(), "poisons: {:?}", out.poison_ops);

    let oracle = |table: u32, key: u32, limb: u32| -> u32 {
        panic!("unexpected table read: table {table} key {key} limb {limb}")
    };

    // Slot layout: flat input words 0..125 (in.0, in.1, W27 10 words, 2 felts x28,
    // 2 felts x28, in.2.3), enabler 125, iota 126.
    let n_padded = packed.len() * N_LANES;
    let rows: Vec<Vec<u32>> = (0..n_padded)
        .map(|r| {
            let (pr, lane) = (r / N_LANES, r % N_LANES);
            let p = &packed[pr];
            let mut row = vec![p.0.to_array()[lane].0, p.1.to_array()[lane].0];
            row.extend((0..10).map(|i| p.2 .0.get_m31(i).to_array()[lane].0));
            for f in &p.2 .1 {
                row.extend((0..28).map(|i| f.get_m31(i).to_array()[lane].0));
            }
            for f in &p.2 .2 {
                row.extend((0..28).map(|i| f.get_m31(i).to_array()[lane].0));
            }
            row.push(p.2 .3.to_array()[lane].0);
            row.push(u32::from(r < n_rows)); // enabler
            row.push(r as u32); // iota
            row
        })
        .collect();

    let n_packed_rows = n_padded / N_LANES;
    for (r, row_inputs) in rows.iter().enumerate() {
        let ro = interpret_row_with(&out.program, row_inputs, &oracle, &mut FastDeductionHost);
        for (c, hv) in diff.orig_rows[r].iter().enumerate() {
            assert_eq!(ro.columns[c], hv.0, "row {r} column {c}");
        }
        let (pr, lane) = (r / N_LANES, r % N_LANES);
        let mut w = 0usize;
        for field in diff.orig_lookup.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].to_array()[lane].0;
                assert_eq!(ro.lookup_words[w], hv, "row {r} lookup word {w} (+{k})");
                w += 1;
            }
        }
        let mut w = 0usize;
        for field in diff.orig_sub.iter() {
            let width = field.len() / n_packed_rows;
            for k in 0..width {
                let hv = field[pr * width + k].as_array()[lane];
                assert_eq!(ro.sub_words[w], hv, "row {r} sub word {w} (+{k})");
                w += 1;
            }
        }
    }

    // GATE (c), pod builds only: the same program as an actual CUDA kernel.
    // Table required although the body never reads it: the felt deduces embed
    // the fp256 chain, so the CUmodule declares the table globals and the
    // fail-closed load fill needs the host table registered (ROUND-28).
    let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
    assert_device_builtin_leg_matches_host(
        "partial_ec_mul_generic",
        out.program,
        true,
        &rows,
        &addr_ids,
        &f252_values,
        &small_values,
        &host_rows,
        &diff.orig_lookup,
        &diff.orig_sub,
    );
}

// ---------------- poseidon-family fp256: cube_252 + range_check_252_width_27 --------

fn fresh_poseidon_claim_generator() -> crate::witness::cairo_claim_generator::CairoClaimGenerator {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    let compiled = get_compiled_cairo_program_path("test_prove_verify_poseidon_builtin");
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run_and_adapt poseidon fixture");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "cube_252",
        "range_check_252_width_27",
        "range_check_9_9",
        "range_check_20",
        "range_check_18",
        "range_check_3_3_3_3_3",
        "range_check_4_4_4_4",
        "range_check_4_4",
        "poseidon_builtin",
        "poseidon_aggregator",
        "poseidon_full_round_chain",
        "poseidon_3_partial_rounds_chain",
        "poseidon_round_keys",
        "memory_address_to_id",
        "memory_id_to_big",
    ] {
        set.insert(c);
    }
    let preprocessed_trace = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    cg
}

/// Shared fixture prep: the poseidon fixture fed through the production chain
/// (poseidon_builtin -> aggregator -> full/partial round chains), which fills
/// cube_252 and range_check_252_width_27 inputs.
#[allow(clippy::type_complexity)]
fn poseidon_family_fixture() -> (crate::witness::cairo_claim_generator::CairoClaimGenerator,) {
    let mut cg = fresh_poseidon_claim_generator();
    // Production feed chain, in spawn order.
    {
        let pb = cg.poseidon_builtin.take().expect("poseidon_builtin");
        let mem_addr = cg.memory_address_to_id.as_ref().expect("mem addr");
        let agg = cg.poseidon_aggregator.as_ref().expect("agg state");
        let _ = pb.write_trace(mem_addr, agg);
    }
    {
        let agg = cg.poseidon_aggregator.take().expect("aggregator populated");
        let _ = agg.write_trace(
            cg.memory_id_to_big.as_ref().expect("mem big"),
            cg.poseidon_full_round_chain.as_ref().expect("full chain"),
            cg.range_check_252_width_27.as_ref().expect("rc252"),
            cg.cube_252.as_ref().expect("cube state"),
            cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
            cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
            cg.range_check_4_4.as_ref().expect("rc44"),
            cg.poseidon_3_partial_rounds_chain
                .as_ref()
                .expect("partial chain"),
        );
    }
    {
        let full = cg
            .poseidon_full_round_chain
            .take()
            .expect("full chain populated");
        let _ = full.write_trace(
            cg.cube_252.as_ref().expect("cube state"),
            cg.poseidon_round_keys.as_ref().expect("round keys"),
            cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
        );
    }
    {
        let partial = cg
            .poseidon_3_partial_rounds_chain
            .take()
            .expect("partial chain populated");
        let _ = partial.write_trace(
            cg.poseidon_round_keys.as_ref().expect("round keys"),
            cg.cube_252.as_ref().expect("cube state"),
            cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
            cg.range_check_4_4.as_ref().expect("rc44"),
            cg.range_check_252_width_27.as_ref().expect("rc252"),
        );
    }
    (cg,)
}

/// Permanent source-of-truth gate for the four newly recorded Poseidon writers.
/// It follows the real producer order so every relation-fed input is exercised,
/// then compares every trace, lookup, and sub-feed word with zero tolerance.
#[test]
fn poseidon_recorded_source_writers_are_byte_identical() {
    // The mechanically generated aggregator writer has thousands of scalar SSA
    // locals.  Debug test builds retain enough of them to exceed libtest's small
    // default worker stack even though the production writer runs inside its
    // explicitly sized Rayon pool.  Keep this permanent conformance gate usable
    // under plain `cargo test` instead of requiring an ambient RUST_MIN_STACK.
    std::thread::Builder::new()
        .name("poseidon-recorded-parity".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(poseidon_recorded_source_writers_are_byte_identical_inner)
        .expect("spawn Poseidon parity thread")
        .join()
        .expect("Poseidon parity thread panicked");
}

fn poseidon_recorded_source_writers_are_byte_identical_inner() {
    use std::sync::atomic::Ordering;

    use crate::witness::components::{
        poseidon_3_partial_rounds_chain as partial, poseidon_aggregator as aggregator,
        poseidon_builtin as builtin, poseidon_full_round_chain as full,
    };

    let mut cg = fresh_poseidon_claim_generator();

    let builtin_gen = cg.poseidon_builtin.take().expect("poseidon builtin");
    assert_generic_diff_byte_identical!(builtin::generic_simd_diff(
        builtin_gen.log_size,
        builtin_gen.poseidon_builtin_segment_start,
        cg.memory_address_to_id.as_ref().expect("mem addr"),
        cg.poseidon_aggregator.as_ref().expect("aggregator state"),
    ));
    let _ = builtin_gen.write_trace(
        cg.memory_address_to_id.as_ref().expect("mem addr"),
        cg.poseidon_aggregator.as_ref().expect("aggregator state"),
    );

    let aggregator_gen = cg.poseidon_aggregator.take().expect("populated aggregator");
    let mut input_mults = aggregator_gen
        .mults
        .iter()
        .map(|entry| (*entry.key(), M31(entry.value().load(Ordering::Relaxed))))
        .collect::<Vec<_>>();
    input_mults.sort_by_key(|(input, _)| input.0);
    let (mut inputs, mut mults): (Vec<_>, Vec<_>) = input_mults.into_iter().unzip();
    let n_real = inputs.len();
    assert!(n_real > 0, "fixture fed no aggregator rows");
    let size = n_real.next_power_of_two().max(N_LANES);
    inputs.resize(size, inputs[0]);
    mults.resize(size, M31::zero());
    assert_generic_diff_byte_identical!(aggregator::generic_simd_diff(
        pack_values(&inputs),
        vec![pack_values(&mults)],
        cg.memory_id_to_big.as_ref().expect("mem big"),
        cg.poseidon_full_round_chain.as_ref().expect("full state"),
        cg.range_check_252_width_27.as_ref().expect("rc252"),
        cg.cube_252.as_ref().expect("cube"),
        cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
        cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
        cg.range_check_4_4.as_ref().expect("rc44"),
        cg.poseidon_3_partial_rounds_chain
            .as_ref()
            .expect("partial state"),
    ));
    let _ = aggregator_gen.write_trace(
        cg.memory_id_to_big.as_ref().expect("mem big"),
        cg.poseidon_full_round_chain.as_ref().expect("full state"),
        cg.range_check_252_width_27.as_ref().expect("rc252"),
        cg.cube_252.as_ref().expect("cube"),
        cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
        cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
        cg.range_check_4_4.as_ref().expect("rc44"),
        cg.poseidon_3_partial_rounds_chain
            .as_ref()
            .expect("partial state"),
    );

    let full_gen = cg
        .poseidon_full_round_chain
        .take()
        .expect("populated full chain");
    let mut full_inputs = full_gen.packed_inputs.lock().unwrap().clone();
    let full_n_rows = full_inputs.len() * N_LANES;
    let full_size = full_inputs.len().next_power_of_two();
    full_inputs.resize(full_size, full_inputs[0]);
    assert_generic_diff_byte_identical!(full::generic_simd_diff(
        full_inputs,
        full_n_rows,
        cg.cube_252.as_ref().expect("cube"),
        cg.poseidon_round_keys.as_ref().expect("round keys"),
        cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
    ));

    let partial_gen = cg
        .poseidon_3_partial_rounds_chain
        .take()
        .expect("populated partial chain");
    let mut partial_inputs = partial_gen.packed_inputs.lock().unwrap().clone();
    let partial_n_rows = partial_inputs.len() * N_LANES;
    let partial_size = partial_inputs.len().next_power_of_two();
    partial_inputs.resize(partial_size, partial_inputs[0]);
    assert_generic_diff_byte_identical!(partial::generic_simd_diff(
        partial_inputs,
        partial_n_rows,
        cg.poseidon_round_keys.as_ref().expect("round keys"),
        cg.cube_252.as_ref().expect("cube"),
        cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
        cg.range_check_4_4.as_ref().expect("rc44"),
        cg.range_check_252_width_27.as_ref().expect("rc252"),
    ));
}

/// Gate (b) for both Poseidon chain writers: replay the recorded bytecode through
/// the host deduce oracle and byte-compare every trace, lookup, and sub-feed word.
#[test]
fn poseidon_chain_recording_interpreters_match_host() {
    std::thread::Builder::new()
        .name("poseidon-chain-interpreter-parity".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(poseidon_chain_recording_interpreters_match_host_inner)
        .expect("spawn Poseidon chain interpreter parity thread")
        .join()
        .expect("Poseidon chain interpreter parity thread panicked");
}

fn poseidon_chain_recording_interpreters_match_host_inner() {
    use crate::witness::components::{
        poseidon_3_partial_rounds_chain as partial, poseidon_full_round_chain as full,
    };

    let mut cg = fresh_poseidon_claim_generator();
    let builtin = cg.poseidon_builtin.take().expect("poseidon builtin");
    let _ = builtin.write_trace(
        cg.memory_address_to_id.as_ref().expect("mem addr"),
        cg.poseidon_aggregator.as_ref().expect("aggregator state"),
    );
    let aggregator = cg.poseidon_aggregator.take().expect("aggregator populated");
    let _ = aggregator.write_trace(
        cg.memory_id_to_big.as_ref().expect("mem big"),
        cg.poseidon_full_round_chain.as_ref().expect("full state"),
        cg.range_check_252_width_27.as_ref().expect("rc252"),
        cg.cube_252.as_ref().expect("cube"),
        cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
        cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
        cg.range_check_4_4.as_ref().expect("rc44"),
        cg.poseidon_3_partial_rounds_chain
            .as_ref()
            .expect("partial state"),
    );

    let full_gen = cg
        .poseidon_full_round_chain
        .take()
        .expect("full chain populated");
    assert!(full_gen.remainder_inputs.lock().unwrap().is_empty());
    let mut full_inputs = full_gen.packed_inputs.into_inner().unwrap();
    let full_n_rows = full_inputs.len() * N_LANES;
    full_inputs.resize(full_inputs.len().next_power_of_two(), full_inputs[0]);
    let full_diff = full::generic_simd_diff(
        full_inputs.clone(),
        full_n_rows,
        cg.cube_252.as_ref().expect("cube"),
        cg.poseidon_round_keys.as_ref().expect("round keys"),
        cg.range_check_3_3_3_3_3.as_ref().expect("rc33333"),
    );
    let full_recording = full::record_poseidon_full_round_chain();
    assert!(
        full_recording.poison_ops.is_empty(),
        "full chain poisons: {:?}",
        full_recording.poison_ops
    );
    let full_rows = poseidon_chain_rows(&full_inputs, full_n_rows);
    assert_poseidon_recording_interpreter_matches_host(
        &full_recording.program,
        &full_rows,
        &full_diff.orig_rows,
        &full_diff.orig_lookup,
        &full_diff.orig_sub,
    );

    let partial_gen = cg
        .poseidon_3_partial_rounds_chain
        .take()
        .expect("partial chain populated");
    assert!(partial_gen.remainder_inputs.lock().unwrap().is_empty());
    let mut partial_inputs = partial_gen.packed_inputs.into_inner().unwrap();
    let partial_n_rows = partial_inputs.len() * N_LANES;
    partial_inputs.resize(partial_inputs.len().next_power_of_two(), partial_inputs[0]);
    let partial_diff = partial::generic_simd_diff(
        partial_inputs.clone(),
        partial_n_rows,
        cg.poseidon_round_keys.as_ref().expect("round keys"),
        cg.cube_252.as_ref().expect("cube"),
        cg.range_check_4_4_4_4.as_ref().expect("rc4444"),
        cg.range_check_4_4.as_ref().expect("rc44"),
        cg.range_check_252_width_27.as_ref().expect("rc252"),
    );
    let partial_recording = partial::record_poseidon_3_partial_rounds_chain();
    assert!(
        partial_recording.poison_ops.is_empty(),
        "partial chain poisons: {:?}",
        partial_recording.poison_ops
    );
    let partial_rows = poseidon_chain_rows(&partial_inputs, partial_n_rows);
    assert_poseidon_recording_interpreter_matches_host(
        &partial_recording.program,
        &partial_rows,
        &partial_diff.orig_rows,
        &partial_diff.orig_lookup,
        &partial_diff.orig_sub,
    );
}

/// Gate (a) for `cube_252` (poseidon-family fp256: x^3 mod p via W27 felts).
#[test]
fn cube_252_generic_simd_byte_identical() {
    use crate::witness::components::cube_252 as m;
    let (cg,) = poseidon_family_fixture();
    let gen = cg.cube_252.expect("cube_252 populated");
    let rc99 = cg.range_check_9_9.expect("rc99");
    let rc20 = cg.range_check_20.expect("rc20");
    let mut packed = gen.packed_inputs.into_inner().unwrap();
    assert!(!packed.is_empty(), "fixture fed no cube_252 inputs");
    let n_rows = packed.len() * N_LANES;
    let packed_size = packed.len().next_power_of_two();
    packed.resize(packed_size, *packed.first().unwrap());
    assert_generic_diff_byte_identical!(m::generic_simd_diff(packed, n_rows, &rc99, &rc20));
}

/// Gate (c) for `cube_252`: launch the recorded program as a real CUDA
/// kernel and compare every committed, lookup, and sub-feed word with the
/// production SIMD writer over the poseidon-family fixture.
#[test]
fn cube_252_recording_device_matches_host() {
    use crate::witness::components::cube_252 as m;

    let (cg,) = poseidon_family_fixture();
    let gen = cg.cube_252.expect("cube_252 populated");
    assert!(
        gen.remainder_inputs.lock().unwrap().is_empty(),
        "fixture left scalar cube_252 inputs"
    );
    let rc99 = cg.range_check_9_9.expect("rc99");
    let rc20 = cg.range_check_20.expect("rc20");
    let mut packed = gen.packed_inputs.into_inner().unwrap();
    assert!(!packed.is_empty(), "fixture fed no cube_252 inputs");
    let n_rows = packed.len() * N_LANES;
    packed.resize(packed.len().next_power_of_two(), packed[0]);
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &rc99, &rc20);

    // Slot layout: W27 words 0..10, enabler 10, iota 11.
    let n_padded = packed.len() * N_LANES;
    let rows = (0..n_padded)
        .map(|r| {
            let (packed_row, lane) = (r / N_LANES, r % N_LANES);
            let input = &packed[packed_row];
            let mut row = (0..10)
                .map(|word| input.get_m31(word).to_array()[lane].0)
                .collect::<Vec<_>>();
            row.push(u32::from(r < n_rows));
            row.push(r as u32);
            row
        })
        .collect::<Vec<_>>();
    let recording = m::record_cube_252();
    assert!(
        recording.poison_ops.is_empty(),
        "cube poisons: {:?}",
        recording.poison_ops
    );
    let host_rows = diff
        .orig_rows
        .iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    assert_device_builtin_leg_matches_host(
        "cube_252",
        recording.program,
        true,
        &rows,
        &[0],
        &[[0; 8]],
        &[0],
        &host_rows,
        &diff.orig_lookup,
        &diff.orig_sub,
    );
}

/// Gate (a) for `range_check_252_width_27` (W27 range-check component).
#[test]
fn range_check_252_width_27_generic_simd_byte_identical() {
    use crate::witness::components::range_check_252_width_27 as m;
    let (cg,) = poseidon_family_fixture();
    let gen = cg
        .range_check_252_width_27
        .expect("range_check_252_width_27 populated");
    let rc99 = cg.range_check_9_9.expect("rc99");
    let rc18 = cg.range_check_18.expect("rc18");
    let mut packed = gen.packed_inputs.into_inner().unwrap();
    assert!(!packed.is_empty(), "fixture fed no rc252w27 inputs");
    let n_rows = packed.len() * N_LANES;
    let packed_size = packed.len().next_power_of_two();
    packed.resize(packed_size, *packed.first().unwrap());
    assert_generic_diff_byte_identical!(m::generic_simd_diff(packed, n_rows, &rc99, &rc18));
}

/// Recording manifests for the poseidon-family pair: fully recorded.
#[test]
fn poseidon_family_recording_poison_manifests() {
    use stwo_backend_cuda::jit_witness::isa::{DeduceKind, WitnessOp};

    use crate::witness::components::{
        cube_252, poseidon_3_partial_rounds_chain, poseidon_aggregator, poseidon_builtin,
        poseidon_full_round_chain, range_check_252_width_27,
    };

    let c = cube_252::record_cube_252();
    assert!(c.poison_ops.is_empty(), "cube poisons: {:?}", c.poison_ops);
    let r = range_check_252_width_27::record_range_check_252_width_27();
    assert!(r.poison_ops.is_empty(), "rc252 poisons: {:?}", r.poison_ops);
    let recordings = [
        poseidon_builtin::record_poseidon_builtin(),
        poseidon_aggregator::record_poseidon_aggregator(),
        poseidon_full_round_chain::record_poseidon_full_round_chain(),
        poseidon_3_partial_rounds_chain::record_poseidon_3_partial_rounds_chain(),
    ];
    for recording in &recordings {
        assert!(
            recording.poison_ops.is_empty(),
            "{} poisons: {:?}",
            recording.program.label,
            recording.poison_ops
        );
        assert!(recording.poisoned_cols.is_empty());
        assert!(recording.poisoned_lookup_words.is_empty());
        assert!(recording.poisoned_sub_words.is_empty());
    }
    let kinds = recordings
        .iter()
        .flat_map(|recording| &recording.program.insts)
        .filter(|inst| inst.op == WitnessOp::DeduceCall as u8)
        .map(|inst| DeduceKind::from_raw(inst.imm).expect("known deduce"))
        .collect::<Vec<_>>();
    for expected in [
        DeduceKind::PoseidonRoundKeys,
        DeduceKind::Cube252,
        DeduceKind::PoseidonFullRoundChain,
        DeduceKind::Poseidon3PartialRoundsChain,
    ] {
        assert!(kinds.contains(&expected), "missing {expected:?}");
    }
    eprintln!(
        "cube_252 instrs={} rc252w27 instrs={} poseidon source instrs={:?}",
        c.program.n_instrs(),
        r.program.n_instrs(),
        recordings
            .iter()
            .map(|recording| recording.program.n_instrs())
            .collect::<Vec<_>>()
    );
}

// ---------------- B2 count-feed gate: descriptors + keying vs the consumers ---------

/// THE COUNT GATE: the device-feed path (SUB_FEED_LAYOUT -> descriptors ->
/// fold/LUT -> count tables -> add_count_tables merge) must produce EXACTLY the
/// multiplicities the consumers' own `add_input` feeds produce — on the real
/// w18 fixture, over every padded row, for every count relation the component
/// touches (points table, rc_9_9 x36 instances, rc_20 x24). Runs the pure-Rust
/// kernel mirror, so a keying/layout bug is caught with no hardware.
#[test]
fn partial_ec_mul_w18_count_feed_matches_consumer_feeds() {
    use crate::witness::components::{
        partial_ec_mul_window_bits_18 as m, pedersen_points_table_window_bits_18, range_check_20,
        range_check_9_9,
    };
    use crate::witness::device_feed::{build_feed_descriptors, host_feed_counts, COUNT_RELATIONS};

    let (packed, n_rows, pts, rc99, rc20, _mem) = w18_fixture();
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &pts, &rc99, &rc20);
    let n_padded = packed.len() * N_LANES;
    let n_packed_rows = n_padded / N_LANES;

    // Word-major sub flats from the host writer's own SubComponentInputs.
    let n_sub: usize = diff.orig_sub.iter().map(|f| f.len() / n_packed_rows).sum();
    let mut sub_flat = vec![0u32; n_sub * n_padded];
    let mut w = 0usize;
    for field in diff.orig_sub.iter() {
        let width = field.len() / n_packed_rows;
        for k in 0..width {
            for r in 0..n_padded {
                sub_flat[w * n_padded + r] =
                    field[(r / N_LANES) * width + k].as_array()[r % N_LANES];
            }
            w += 1;
        }
    }

    // Device path (host mirror): descriptors from the emitted layout + registry.
    let (descs, lut_slots, counts_slots) =
        build_feed_descriptors(m::SUB_FEED_LAYOUT, COUNT_RELATIONS);
    assert!(
        !descs.is_empty(),
        "w18 must have count-relation feed descriptors"
    );
    let luts: Vec<Vec<u32>> = lut_slots
        .iter()
        .map(|s| match *s {
            "range_check_9_9_state" => rc99.input_to_row_lut(),
            other => panic!("unexpected LUT slot {other}"),
        })
        .collect();
    let mut counts: Vec<Vec<u32>> = counts_slots
        .iter()
        .map(|s| {
            let rel = COUNT_RELATIONS
                .iter()
                .find(|r| r.state_param == *s)
                .unwrap();
            vec![0u32; rel.n_relations * rel.table_size]
        })
        .collect();
    host_feed_counts(&sub_flat, n_padded, &descs, &luts, &mut counts);

    // Merge into FRESH consumer states...
    let preproc = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    let device_pts = pedersen_points_table_window_bits_18::ClaimGenerator::new(preproc.clone());
    let device_rc99 = range_check_9_9::ClaimGenerator::new(preproc.clone());
    let device_rc20 = range_check_20::ClaimGenerator::new(preproc.clone());
    for (slot, c) in counts_slots.iter().zip(&counts) {
        match *slot {
            "pedersen_points_table_window_bits_18_state" => device_pts.add_count_tables(c),
            "range_check_9_9_state" => device_rc99.add_count_tables(c),
            "range_check_20_state" => device_rc20.add_count_tables(c),
            other => panic!("unexpected counts slot {other}"),
        }
    }

    // ...and compare against states fed by the consumers' OWN add_input calls
    // over the same sub tuples (the host writer's exact feed semantics).
    let host_pts = pedersen_points_table_window_bits_18::ClaimGenerator::new(preproc.clone());
    let host_rc99 = range_check_9_9::ClaimGenerator::new(preproc.clone());
    let host_rc20 = range_check_20::ClaimGenerator::new(preproc);
    for &(_f, _i, state, rel, base, words) in m::SUB_FEED_LAYOUT {
        for r in 0..n_padded {
            let word = |k: usize| sub_flat[(base + k) * n_padded + r];
            match state {
                "pedersen_points_table_window_bits_18_state" => {
                    use crate::witness::utils::AddInputs;
                    host_pts.add_input(&[M31(word(0))], rel as usize);
                }
                "range_check_9_9_state" => {
                    use crate::witness::utils::AddInputs;
                    host_rc99.add_input(&[M31(word(0)), M31(word(1))], rel as usize);
                }
                "range_check_20_state" => {
                    use crate::witness::utils::AddInputs;
                    host_rc20.add_input(&[M31(word(0))], rel as usize);
                }
                other => panic!("unexpected state {other}"),
            }
            let _ = words;
        }
    }

    // Byte-compare every multiplicity column.
    let cmp = |name: &str, a: Vec<_>, b: Vec<_>| {
        let (a, b): (Vec<Vec<PackedM31>>, Vec<Vec<PackedM31>>) = (a, b);
        assert_eq!(a.len(), b.len(), "{name} relation count");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            for (v, (p, q)) in x.iter().zip(y).enumerate() {
                assert_eq!(
                    p.to_array(),
                    q.to_array(),
                    "{name} relation {i} packed row {v}"
                );
            }
        }
    };
    cmp(
        "points_table",
        device_pts
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
        host_pts
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
    );
    cmp(
        "rc_9_9",
        device_rc99
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
        host_rc99
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
    );
    cmp(
        "rc_20",
        device_rc20
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
        host_rc20
            .mults
            .into_iter()
            .map(|m| m.into_simd_vec())
            .collect(),
    );
    eprintln!(
        "count gate [partial_ec_mul_window_bits_18]: PASS ({} descriptors, {n_padded} rows)",
        descs.len() / crate::witness::device_feed::WFC_DESC_STRIDE
    );
}

/// Shared engine for the count gates: run the device-feed path (descriptors +
/// pure-Rust kernel mirror) AND replay the same tuples through `host_feed`
/// (the consumers' own `add_input` semantics), so the caller can byte-compare
/// its two fresh state sets. Returns the descriptor count (0 = the component
/// touches no count relations — the caller should assert its expectation).
fn run_count_feed_paths(
    layout: &'static [(&'static str, usize, &'static str, u32, usize, usize)],
    sub_flat: &[u32],
    n_padded: usize,
    lut_for: impl Fn(&'static str) -> Vec<u32>,
    merge: impl Fn(&'static str, &[u32]),
    host_feed: impl FnMut(&'static str, u32, &[u32]),
) -> usize {
    run_count_feed_paths_sized(
        layout,
        sub_flat,
        n_padded,
        &|_| None,
        lut_for,
        merge,
        host_feed,
    )
}

/// The v2 engine: `sizes` opts runtime-sized families (the memory tables) into
/// the device path; the host reference replays add_input semantics per kind
/// (fold + offset; mem-id decode skips like the kernel).
#[allow(clippy::too_many_arguments)]
fn run_count_feed_paths_sized(
    layout: &'static [(&'static str, usize, &'static str, u32, usize, usize)],
    sub_flat: &[u32],
    n_padded: usize,
    sizes: &dyn Fn(&'static str) -> Option<(usize, usize)>,
    lut_for: impl Fn(&'static str) -> Vec<u32>,
    merge: impl Fn(&'static str, &[u32]),
    mut host_feed: impl FnMut(&'static str, u32, &[u32]),
) -> usize {
    use crate::witness::device_feed::{
        build_feed_descriptors_sized, host_feed_counts, COUNT_RELATIONS, WFC_DESC_STRIDE,
    };
    let (descs, lut_slots, counts_slots, slot_sizes) =
        build_feed_descriptors_sized(layout, COUNT_RELATIONS, sizes);
    if descs.is_empty() {
        return 0;
    }
    let luts: Vec<Vec<u32>> = lut_slots.iter().map(|s| lut_for(s)).collect();
    let mut counts: Vec<Vec<u32>> = slot_sizes.iter().map(|&n| vec![0u32; n]).collect();
    host_feed_counts(sub_flat, n_padded, &descs, &luts, &mut counts);
    for (slot, c) in counts_slots.iter().zip(&counts) {
        merge(slot, c);
    }
    // Host reference: the consumers' own add_input over the same tuples —
    // ONLY for count relations (the same set the descriptors cover). Tuples
    // whose fold key exceeds the domain are skipped, mirroring the kernel's
    // memory-safety guard (only synthetic gate inputs can produce them; a real
    // trace's host feed would panic on such a tuple).
    for &(_f, _i, state, rel, base, words) in layout {
        let Some(relation) = COUNT_RELATIONS.iter().find(|r| r.state_param == state) else {
            continue;
        };
        let (table_size, _small) = if relation.table_size == 0 {
            match sizes(relation.state_param) {
                Some(sz) => sz,
                None => continue,
            }
        } else {
            (relation.table_size, 0)
        };
        let mut tuple = vec![0u32; words];
        for r in 0..n_padded {
            let mut key: u64 = 0;
            for (k, t) in tuple.iter_mut().enumerate() {
                *t = sub_flat[(base + k) * n_padded + r];
                key = (key << relation.word_bits[k]) | u64::from(*t);
            }
            if relation.kind == 1 {
                // Mem-id decode: the consumer's add_input decodes; skip only the
                // defensive DEFAULT_ID like the kernel (valid traces never feed it).
                if tuple[0] == (1u32 << 30) - 1 {
                    continue;
                }
                host_feed(state, rel, &tuple);
                continue;
            }
            let keyed = key as i64 + relation.key_offset;
            if keyed < 0 || keyed as usize >= table_size {
                continue;
            }
            host_feed(state, rel, &tuple);
        }
    }
    descs.len() / WFC_DESC_STRIDE
}

/// Byte-compare two multiplicity column sets.
fn assert_mults_eq(name: &str, a: Vec<Vec<PackedM31>>, b: Vec<Vec<PackedM31>>) {
    assert_eq!(a.len(), b.len(), "{name} relation count");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        for (v, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(
                p.to_array(),
                q.to_array(),
                "{name} relation {i} packed row {v}"
            );
        }
    }
}

/// Word-major sub flats from a GenericSimdDiff's host SubComponentInputs.
fn sub_flat_from_diff(
    orig_sub: &[Vec<std::simd::Simd<u32, N_LANES>>],
    n_padded: usize,
) -> Vec<u32> {
    let n_packed_rows = n_padded / N_LANES;
    let n_sub: usize = orig_sub.iter().map(|f| f.len() / n_packed_rows).sum();
    let mut flat = vec![0u32; n_sub * n_padded];
    let mut w = 0usize;
    for field in orig_sub {
        let width = field.len() / n_packed_rows;
        for k in 0..width {
            for r in 0..n_padded {
                flat[w * n_padded + r] = field[(r / N_LANES) * width + k].as_array()[r % N_LANES];
            }
            w += 1;
        }
    }
    flat
}

/// COUNT GATE for `partial_ec_mul_generic` (rc_8 / rc_9_9 / rc_20).
#[test]
fn partial_ec_mul_generic_count_feed_matches_consumer_feeds() {
    use crate::witness::components::{
        partial_ec_mul_generic as m, range_check_20, range_check_8, range_check_9_9,
    };
    use crate::witness::utils::AddInputs;

    let (packed, n_rows, rc8, rc99, rc20, _mem) = partial_ec_mul_generic_fixture();
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &rc8, &rc99, &rc20);
    let n_padded = packed.len() * N_LANES;
    let sub_flat = sub_flat_from_diff(&diff.orig_sub, n_padded);

    let preproc = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    let (d8, d99, d20) = (
        range_check_8::ClaimGenerator::new(preproc.clone()),
        range_check_9_9::ClaimGenerator::new(preproc.clone()),
        range_check_20::ClaimGenerator::new(preproc.clone()),
    );
    let (h8, h99, h20) = (
        range_check_8::ClaimGenerator::new(preproc.clone()),
        range_check_9_9::ClaimGenerator::new(preproc.clone()),
        range_check_20::ClaimGenerator::new(preproc),
    );
    let n = run_count_feed_paths(
        m::SUB_FEED_LAYOUT,
        &sub_flat,
        n_padded,
        |f| match f {
            "range_check_9_9_state" => rc99.input_to_row_lut(),
            other => panic!("unexpected LUT family {other}"),
        },
        |f, c| match f {
            "range_check_8_state" => d8.add_count_tables(c),
            "range_check_9_9_state" => d99.add_count_tables(c),
            "range_check_20_state" => d20.add_count_tables(c),
            other => panic!("unexpected count family {other}"),
        },
        |f, rel, t| match f {
            "range_check_8_state" => h8.add_input(&[M31(t[0])], rel as usize),
            "range_check_9_9_state" => h99.add_input(&[M31(t[0]), M31(t[1])], rel as usize),
            "range_check_20_state" => h20.add_input(&[M31(t[0])], rel as usize),
            other => panic!("unexpected state {other}"),
        },
    );
    assert!(n > 0, "generic must have count descriptors");
    let flat = |g: [crate::witness::utils::AtomicMultiplicityColumn; 1]| {
        g.into_iter().map(|m| m.into_simd_vec()).collect::<Vec<_>>()
    };
    let _ = flat;
    assert_mults_eq(
        "generic rc_8",
        d8.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        h8.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
    );
    assert_mults_eq(
        "generic rc_9_9",
        d99.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        h99.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
    );
    assert_mults_eq(
        "generic rc_20",
        d20.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        h20.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
    );
    eprintln!("count gate [partial_ec_mul_generic]: PASS ({n} descriptors)");
}

/// COUNT GATE for `cube_252` (rc_9_9 / rc_20).
#[test]
fn cube_252_count_feed_matches_consumer_feeds() {
    use crate::witness::components::{cube_252 as m, range_check_20, range_check_9_9};
    use crate::witness::utils::AddInputs;

    let (cg,) = poseidon_family_fixture();
    let gen = cg.cube_252.expect("cube_252 populated");
    let rc99 = cg.range_check_9_9.expect("rc99");
    let rc20 = cg.range_check_20.expect("rc20");
    let mut packed = gen.packed_inputs.into_inner().unwrap();
    let n_rows = packed.len() * N_LANES;
    let packed_size = packed.len().next_power_of_two();
    packed.resize(packed_size, *packed.first().unwrap());
    let diff = m::generic_simd_diff(packed.clone(), n_rows, &rc99, &rc20);
    let n_padded = packed.len() * N_LANES;
    let sub_flat = sub_flat_from_diff(&diff.orig_sub, n_padded);

    let preproc = Arc::new(PreProcessedTrace::canonical_without_pedersen());
    let (d99, d20) = (
        range_check_9_9::ClaimGenerator::new(preproc.clone()),
        range_check_20::ClaimGenerator::new(preproc.clone()),
    );
    let (h99, h20) = (
        range_check_9_9::ClaimGenerator::new(preproc.clone()),
        range_check_20::ClaimGenerator::new(preproc),
    );
    let n = run_count_feed_paths(
        m::SUB_FEED_LAYOUT,
        &sub_flat,
        n_padded,
        |f| match f {
            "range_check_9_9_state" => rc99.input_to_row_lut(),
            other => panic!("unexpected LUT family {other}"),
        },
        |f, c| match f {
            "range_check_9_9_state" => d99.add_count_tables(c),
            "range_check_20_state" => d20.add_count_tables(c),
            other => panic!("unexpected count family {other}"),
        },
        |f, rel, t| match f {
            "range_check_9_9_state" => h99.add_input(&[M31(t[0]), M31(t[1])], rel as usize),
            "range_check_20_state" => h20.add_input(&[M31(t[0])], rel as usize),
            other => panic!("unexpected state {other}"),
        },
    );
    assert!(n > 0, "cube_252 must have count descriptors");
    assert_mults_eq(
        "cube rc_9_9",
        d99.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        h99.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
    );
    assert_mults_eq(
        "cube rc_20",
        d20.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        h20.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
    );
    eprintln!("count gate [cube_252]: PASS ({n} descriptors)");
}

/// COUNT GATES for the two PROVE-WIRED components (aggregator: rc_8;
/// blake_round: rc_7_2_5 through its LUT) — the exact split the seams run.
#[test]
fn aggregator_and_blake_count_feeds_match_consumer_feeds() {
    use stwo_cairo_common::prover_types::cpu::UInt32;

    use crate::witness::components::{
        blake_round, pedersen_aggregator_window_bits_18 as agg, range_check_7_2_5, range_check_8,
    };
    use crate::witness::utils::AddInputs;

    let preproc = Arc::new(PreProcessedTrace::canonical_without_pedersen());

    // Aggregator on the pedersen fixture.
    {
        let (packed, n_rows, pts, rc99v, rc20v, _mem) = w18_fixture();
        let _ = (packed, n_rows, pts, rc99v, rc20v); // fixture warms the chain
    }
    {
        use cairo_vm::types::layout_name::LayoutName;
        use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
        use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
        let compiled = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
        let input = run_and_adapt(
            &compiled,
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .expect("pedersen fixture");
        let ProverInput {
            state_transitions,
            memory,
            builtin_segments,
            ..
        } = input;
        let mut cg = CairoClaimGenerator::default();
        let mut set: IndexSet<&str> = IndexSet::new();
        for c in [
            "pedersen_aggregator_window_bits_18",
            "memory_address_to_id",
            "memory_id_to_big",
            "range_check_8",
            "partial_ec_mul_window_bits_18",
            "pedersen_builtin",
        ] {
            set.insert(c);
        }
        cg.fill_components(
            &set,
            state_transitions.casm_states_by_opcode,
            &builtin_segments,
            Arc::new(memory),
            Arc::new(PreProcessedTrace::canonical_without_pedersen()),
        );
        {
            let pb = cg.pedersen_builtin.take().unwrap();
            let _ = pb.write_trace(
                cg.memory_address_to_id.as_ref().unwrap(),
                cg.pedersen_aggregator_window_bits_18.as_ref().unwrap(),
            );
        }
        let gen = cg.pedersen_aggregator_window_bits_18.unwrap();
        let mem_big = cg.memory_id_to_big.unwrap();
        let rc8 = cg.range_check_8.unwrap();
        let w18s = cg.partial_ec_mul_window_bits_18.unwrap();
        let mut inputs_mults = gen
            .mults
            .iter()
            .map(|e| (*e.key(), M31(e.value().load(Ordering::Relaxed))))
            .collect::<Vec<_>>();
        inputs_mults.sort_by_key(|(i, _)| i.0);
        let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let n_rows = inputs.len();
        let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
        inputs.resize(size, *inputs.first().unwrap());
        mults.resize(size, M31::zero());
        let packed_inputs = pack_values(&inputs);
        let packed_mults = pack_values(&mults);
        let diff = agg::generic_simd_diff(packed_inputs, vec![packed_mults], &mem_big, &rc8, &w18s);
        let sub_flat = sub_flat_from_diff(&diff.orig_sub, size);

        let d8 = range_check_8::ClaimGenerator::new(preproc.clone());
        let h8 = range_check_8::ClaimGenerator::new(preproc.clone());
        let n = run_count_feed_paths(
            agg::SUB_FEED_LAYOUT,
            &sub_flat,
            size,
            |f| panic!("aggregator needs no LUT, got {f}"),
            |f, c| match f {
                "range_check_8_state" => d8.add_count_tables(c),
                other => panic!("unexpected count family {other}"),
            },
            |f, rel, t| match f {
                "range_check_8_state" => h8.add_input(&[M31(t[0])], rel as usize),
                other => panic!("unexpected state {other}"),
            },
        );
        assert!(n > 0);
        assert_mults_eq(
            "aggregator rc_8",
            d8.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
            h8.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        );
        eprintln!("count gate [pedersen_aggregator]: PASS ({n} descriptors)");
    }

    // blake_round on synthetic inputs (the interp-gate recipe).
    {
        let (cg, _a, _f, _s, _mem_arc) = fill_fixture_with_memory(&[
            "blake_round",
            "blake_round_sigma",
            "blake_g",
            "memory_address_to_id",
            "memory_id_to_big",
            "range_check_7_2_5",
        ]);
        let sigma = cg.blake_round_sigma.expect("sigma");
        let mem_addr = cg.memory_address_to_id.expect("mem addr");
        let mem_big = cg.memory_id_to_big.expect("mem big");
        let rc725 = cg.range_check_7_2_5.expect("rc725");
        let blake_g = cg.blake_g.expect("blake_g");
        let inputs: Vec<blake_round::InputType> = (0..24u32)
            .map(|i| {
                let words: [UInt32; 16] = std::array::from_fn(|j| {
                    UInt32::from(0x9E37_79B9u32.wrapping_mul(j as u32 + i))
                });
                (M31(i + 1), M31(i % 10), (words, M31(1 + (i % 4))))
            })
            .collect();
        let n_rows = inputs.len();
        let size = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
        let mut padded = inputs;
        padded.resize(size, *padded.first().unwrap());
        let packed = pack_values(&padded);
        let diff = blake_round::generic_simd_diff(
            packed, n_rows, &sigma, &mem_addr, &mem_big, &rc725, &blake_g,
        );
        let sub_flat = sub_flat_from_diff(&diff.orig_sub, size);

        let d725 = range_check_7_2_5::ClaimGenerator::new(preproc.clone());
        let h725 = range_check_7_2_5::ClaimGenerator::new(preproc.clone());
        // B2 v2 families: independent device/host state pairs over the SAME
        // memory — the device path merges kernel-mirror counts, the host path
        // replays the consumers' own add_input; mult columns must match.
        let d_sigma = blake_round_sigma::ClaimGenerator::new(preproc.clone());
        let h_sigma = blake_round_sigma::ClaimGenerator::new(preproc.clone());
        let d_addr = memory_address_to_id::ClaimGenerator::new(_mem_arc.clone());
        let h_addr = memory_address_to_id::ClaimGenerator::new(_mem_arc.clone());
        let d_big = memory_id_to_big::ClaimGenerator::new(_mem_arc.clone());
        let h_big = memory_id_to_big::ClaimGenerator::new(_mem_arc.clone());
        let n = run_count_feed_paths_sized(
            blake_round::SUB_FEED_LAYOUT,
            &sub_flat,
            size,
            &|f| match f {
                "memory_address_to_id_state" => Some((d_addr.table_size(), 0)),
                "memory_id_to_big_state" => {
                    Some((d_big.big_table_size(), d_big.small_table_size()))
                }
                _ => None,
            },
            |f| match f {
                "range_check_7_2_5_state" => rc725.input_to_row_lut(),
                "blake_round_sigma_state" => d_sigma.input_to_row_lut(),
                other => panic!("unexpected LUT family {other}"),
            },
            |f, c| match f {
                "range_check_7_2_5_state" => d725.add_count_tables(c),
                "blake_round_sigma_state" => d_sigma.add_count_tables(c),
                "memory_address_to_id_state" => d_addr.add_count_tables(c),
                "memory_id_to_big_state" => d_big.add_big_count_tables(c),
                "memory_id_to_big_state#small" => d_big.add_small_count_tables(c),
                other => panic!("unexpected count family {other}"),
            },
            |f, rel, t| match f {
                "range_check_7_2_5_state" => {
                    h725.add_input(&[M31(t[0]), M31(t[1]), M31(t[2])], rel as usize)
                }
                "blake_round_sigma_state" => h_sigma.add_input(&[M31(t[0])], rel as usize),
                "memory_address_to_id_state" => {
                    crate::witness::utils::AddInputs::add_input(&h_addr, &M31(t[0]), rel as usize)
                }
                "memory_id_to_big_state" => {
                    crate::witness::utils::AddInputs::add_input(&h_big, &M31(t[0]), rel as usize)
                }
                other => panic!("unexpected state {other}"),
            },
        );
        assert!(n > 0);
        assert_mults_eq(
            "blake rc_7_2_5",
            d725.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
            h725.mults.into_iter().map(|m| m.into_simd_vec()).collect(),
        );
        assert_mults_eq(
            "blake sigma",
            d_sigma.mults_snapshot(),
            h_sigma.mults_snapshot(),
        );
        assert_mults_eq(
            "blake memory_address_to_id",
            d_addr.mults_snapshot(),
            h_addr.mults_snapshot(),
        );
        assert_mults_eq(
            "blake memory_id_to_big (big)",
            d_big.big_mults_snapshot(),
            h_big.big_mults_snapshot(),
        );
        assert_mults_eq(
            "blake memory_id_to_big (small)",
            d_big.small_mults_snapshot(),
            h_big.small_mults_snapshot(),
        );
        eprintln!(
            "count gate [blake_round]: PASS ({n} descriptors incl. mem-table + sigma v2 families)"
        );
    }
}

// ---------------- B3 edge gate: producer sub buffer == consumer inputs --------------

/// Pure-Rust mirror of `witness_edge_gather_kernel` (same indexing + padding
/// rule): consumer col k at row = producer sub word (base + j*words + k) at
/// producer row r, with j/r from instance-major stacking and padding rows
/// replicating the first packed row's lanes.
fn host_edge_gather(
    producer_sub: &[u32],
    producer_rows: usize,
    word_base: usize,
    words_per_instance: usize,
    n_instances: usize,
    consumer_rows: usize,
) -> Vec<Vec<u32>> {
    let real_rows = n_instances * producer_rows;
    (0..words_per_instance)
        .map(|k| {
            (0..consumer_rows)
                .map(|row| {
                    let src = if row < real_rows { row } else { row & 15 };
                    let (j, r) = (src / producer_rows, src % producer_rows);
                    producer_sub[(word_base + j * words_per_instance + k) * producer_rows + r]
                })
                .collect()
        })
        .collect()
}

/// THE EDGE GATE (aggregator→w18): the device-edge gather of the aggregator's
/// sub buffer must be byte-identical to the w18 inputs the HOST FEED produced —
/// the full 72-word instances, instance-major stacking, padding rule included.
/// Validates the B3 layout contract with zero hardware.
#[test]
fn aggregator_to_w18_edge_gather_matches_host_feed() {
    use crate::witness::components::pedersen_aggregator_window_bits_18 as agg;

    // The fixture chain feeds w18 through the aggregator's REAL host feed.
    let (w18_packed, _n_rows, _pts, _rc99, _rc20, _mem) = w18_fixture();

    // Reconstruct the aggregator side: rerun the fixture up to the aggregator
    // diff (pure) to get its sub buffer.
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
    let compiled = get_compiled_cairo_program_path("test_prove_verify_pedersen_builtin");
    let input = run_and_adapt(
        &compiled,
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("pedersen fixture");
    let ProverInput {
        state_transitions,
        memory,
        builtin_segments,
        ..
    } = input;
    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in [
        "pedersen_aggregator_window_bits_18",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_8",
        "partial_ec_mul_window_bits_18",
        "pedersen_builtin",
    ] {
        set.insert(c);
    }
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        Arc::new(PreProcessedTrace::canonical_without_pedersen()),
    );
    {
        let pb = cg.pedersen_builtin.take().unwrap();
        let _ = pb.write_trace(
            cg.memory_address_to_id.as_ref().unwrap(),
            cg.pedersen_aggregator_window_bits_18.as_ref().unwrap(),
        );
    }
    let gen = cg.pedersen_aggregator_window_bits_18.unwrap();
    let mem_big = cg.memory_id_to_big.unwrap();
    let rc8 = cg.range_check_8.unwrap();
    let w18s = cg.partial_ec_mul_window_bits_18.unwrap();
    let mut inputs_mults = gen
        .mults
        .iter()
        .map(|e| (*e.key(), M31(e.value().load(Ordering::Relaxed))))
        .collect::<Vec<_>>();
    inputs_mults.sort_by_key(|(i, _)| i.0);
    let (mut inputs, mut mults) = inputs_mults.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let agg_real = inputs.len();
    let agg_padded = std::cmp::max(agg_real.next_power_of_two(), N_LANES);
    inputs.resize(agg_padded, *inputs.first().unwrap());
    mults.resize(agg_padded, M31::zero());
    let diff = agg::generic_simd_diff(
        pack_values(&inputs),
        vec![pack_values(&mults)],
        &mem_big,
        &rc8,
        &w18s,
    );
    let agg_sub = sub_flat_from_diff(&diff.orig_sub, agg_padded);

    // The device edge: aggregator sub base 7, 28 instances x 72 words.
    let w18_real = 28 * agg_padded;
    let w18_padded = w18_real.next_power_of_two().max(N_LANES);
    let gathered = host_edge_gather(&agg_sub, agg_padded, 7, 72, 28, w18_padded);

    // Host ground truth: w18's packed inputs (host-fed, padded by its writer's
    // rule) flattened to the same 72 slot columns.
    assert_eq!(
        w18_packed.len() * N_LANES,
        w18_padded,
        "w18 padded extent disagrees with the edge stacking"
    );
    for (row, (pr, lane)) in (0..w18_padded).map(|r| (r, (r / N_LANES, r % N_LANES))) {
        let p = &w18_packed[pr];
        assert_eq!(gathered[0][row], p.0.to_array()[lane].0, "chain row {row}");
        assert_eq!(gathered[1][row], p.1.to_array()[lane].0, "round row {row}");
        for (wi, w) in p.2 .0.iter().enumerate() {
            assert_eq!(
                gathered[2 + wi][row],
                w.to_array()[lane].0,
                "window {wi} row {row}"
            );
        }
        for (fi, f) in p.2 .1.iter().enumerate() {
            for i in 0..28 {
                assert_eq!(
                    gathered[16 + fi * 28 + i][row],
                    f.get_m31(i).to_array()[lane].0,
                    "felt {fi} limb {i} row {row}"
                );
            }
        }
    }
    eprintln!(
        "edge gate [aggregator->w18]: PASS ({agg_padded} producer rows x 28 instances -> \
         {w18_padded} consumer rows)"
    );
}

/// THE EDGE GATE (blake_round→blake_g): the row-major interleave of blake's
/// sub buffer must be byte-identical to the inputs blake_g's HOST FEED
/// produces (raw u32 words, instance-major, padding rule) — the exact layout
/// the certified hand kernel uploads.
#[test]
fn blake_to_blake_g_edge_interleave_matches_host_feed() {
    use stwo_cairo_common::prover_types::cpu::UInt32;

    use crate::witness::components::{blake_g, blake_round};

    let (cg, _a, _f, _s, _mem_arc) = fill_fixture_with_memory(&[
        "blake_round",
        "blake_round_sigma",
        "blake_g",
        "memory_address_to_id",
        "memory_id_to_big",
        "range_check_7_2_5",
    ]);
    let sigma = cg.blake_round_sigma.expect("sigma");
    let mem_addr = cg.memory_address_to_id.expect("mem addr");
    let mem_big = cg.memory_id_to_big.expect("mem big");
    let rc725 = cg.range_check_7_2_5.expect("rc725");
    let blake_g_state = cg.blake_g.expect("blake_g");
    let inputs: Vec<blake_round::InputType> = (0..24u32)
        .map(|i| {
            let words: [UInt32; 16] =
                std::array::from_fn(|j| UInt32::from(0x9E37_79B9u32.wrapping_mul(j as u32 + i)));
            (M31(i + 1), M31(i % 10), (words, M31(1 + (i % 4))))
        })
        .collect();
    let n_rows = inputs.len();
    let prod_padded = std::cmp::max(n_rows.next_power_of_two(), N_LANES);
    let mut padded = inputs;
    padded.resize(prod_padded, *padded.first().unwrap());
    let packed = pack_values(&padded);
    let diff = blake_round::generic_simd_diff(
        packed,
        n_rows,
        &sigma,
        &mem_addr,
        &mem_big,
        &rc725,
        &blake_g_state,
    );
    let sub_flat = sub_flat_from_diff(&diff.orig_sub, prod_padded);

    // Host feed ground truth: feed a FRESH blake_g state from the flat (the
    // producer's exact host semantics), then read its packed inputs.
    let fresh = blake_g::ClaimGenerator {
        packed_inputs: std::sync::Mutex::new(vec![]),
        remainder_inputs: std::sync::Mutex::new(vec![]),
    };
    blake_round::feed_blake_g_inputs_from_flat(&sub_flat, prod_padded, &fresh);
    let mut fed = fresh.packed_inputs.into_inner().unwrap();
    assert!(!fed.is_empty());
    let n_real = 8 * prod_padded;
    assert_eq!(fed.len() * N_LANES, n_real);
    let cons_padded = std::cmp::max(n_real.next_power_of_two(), N_LANES);
    fed.resize(cons_padded / N_LANES, *fed.first().unwrap());

    // Device-edge mirror: row-major interleave with the kernel's exact rule.
    for row in 0..cons_padded {
        let src = if row < n_real { row } else { row & 15 };
        let (j, r) = (src / prod_padded, src % prod_padded);
        let (pr, lane) = (row / N_LANES, row % N_LANES);
        for w in 0..6 {
            let mirrored = sub_flat[(81 + j * 6 + w) * prod_padded + r];
            let host = fed[pr][w].simd.as_array()[lane];
            assert_eq!(mirrored, host, "row {row} word {w}");
        }
    }
    eprintln!(
        "edge gate [blake_round->blake_g]: PASS ({prod_padded} producer rows x 8 -> \
         {cons_padded} consumer rows)"
    );
}

/// SN2 omission diagnostic: pin the two host-origin count classes which are
/// not produced by recorded component sub-feeds. These are the exact public
/// memory seed loop and memory-big rc_9_9 row semantics used by the SIMD path.
#[test]
fn sn2_host_origin_memory_count_shapes_are_nonzero() {
    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
        PreProcessedTrace, MAX_SEQUENCE_LOG_SIZE,
    };
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

    use crate::witness::components::range_check_9_9;

    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .expect("run SN2 fixture");
    let ProverInput {
        memory,
        public_memory_addresses,
        ..
    } = input;
    let memory = Arc::new(memory);
    let address_seeds = memory_address_to_id::ClaimGenerator::new(memory.clone());
    let value_seeds = memory_id_to_big::ClaimGenerator::new(memory.clone());

    // Keep this byte-for-byte semantic twin of create_cairo_claim_generator's
    // public-memory loop: resolve id, seed address, then seed encoded value id.
    for addr in public_memory_addresses
        .iter()
        .copied()
        .map(M31::from_u32_unchecked)
    {
        let id = address_seeds.get_id(addr);
        address_seeds.add_input(&addr, 0);
        value_seeds.add_input(&id, 0);
    }
    let scalar_counts = |columns: Vec<Vec<PackedM31>>| {
        columns
            .into_iter()
            .flat_map(|column| column.into_iter().flat_map(|packed| packed.to_array()))
            .map(|value| value.0)
            .collect::<Vec<_>>()
    };
    let address_counts = scalar_counts(address_seeds.mults_snapshot());
    let big_seed_counts = scalar_counts(value_seeds.big_mults_snapshot());
    let small_seed_counts = scalar_counts(value_seeds.small_mults_snapshot());
    let stats = |counts: &[u32]| {
        (
            counts.iter().map(|&count| u64::from(count)).sum::<u64>(),
            counts.iter().filter(|&&count| count != 0).count(),
        )
    };
    let (address_total, address_nonzero) = stats(&address_counts);
    let (big_seed_total, big_seed_nonzero) = stats(&big_seed_counts);
    let (small_seed_total, small_seed_nonzero) = stats(&small_seed_counts);
    assert_eq!(address_total, public_memory_addresses.len() as u64);
    assert_eq!(big_seed_total + small_seed_total, address_total);
    assert!(address_nonzero > 0 && big_seed_nonzero > 0 && small_seed_nonzero > 0);

    // Run the real memory table writer into a fresh rc_9_9 state. Its totals
    // include big rows (14 pairs, relation shape 2,2,2,2,2,2,1,1) plus small
    // rows (4 pairs in relations 0..3). Subtract the latter to expose the
    // otherwise omitted big-row contribution exactly.
    let rc99 = range_check_9_9::ClaimGenerator::new(Arc::new(PreProcessedTrace::canonical()));
    let memory_values = memory_id_to_big::ClaimGenerator::new(memory);
    let (_, _, _, interaction) = memory_values.write_trace(&rc99, MAX_SEQUENCE_LOG_SIZE, None);
    let big_rows = interaction
        .big_components_values
        .iter()
        .map(|component| component[0].len() * N_LANES)
        .sum::<usize>();
    let small_rows = interaction.small_values[0].len() * N_LANES;
    let rc_totals = rc99.mults.each_ref().map(|column| {
        column
            .snapshot_simd_vec()
            .into_iter()
            .flat_map(|packed| packed.to_array())
            .map(|value| u64::from(value.0))
            .sum::<u64>()
    });
    let big_pair_totals = std::array::from_fn::<_, 8, _>(|relation| {
        rc_totals[relation] - if relation < 4 { small_rows as u64 } else { 0 }
    });
    let expected_per_row = [2, 2, 2, 2, 2, 2, 1, 1];
    assert_eq!(
        big_pair_totals,
        expected_per_row.map(|pairs| pairs * big_rows as u64)
    );
    assert!(big_pair_totals.iter().all(|&count| count > 0));
    assert_eq!(big_pair_totals.iter().sum::<u64>(), 14 * big_rows as u64);
    eprintln!(
        "SN2 host-origin counts: public={} address_nonzero={} big_seeds={}/{} \
         small_seeds={}/{}; memory rows big={} small={} big_rc9_9={:?}",
        address_total,
        address_nonzero,
        big_seed_total,
        big_seed_nonzero,
        small_seed_total,
        small_seed_nonzero,
        big_rows,
        small_rows,
        big_pair_totals,
    );
}
