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
use stwo_cairo_adapter::ProverInput;

use super::{TABLE_ADDR_TO_ID, TABLE_ID_TO_BIG};
use crate::witness::cairo_claim_generator::CairoClaimGenerator;
use crate::witness::components::{add_opcode, assert_eq_opcode, jnz_opcode_taken};
use crate::witness::prelude::*;

/// Deserialize the fixture and populate `components` on a fresh `CairoClaimGenerator`.
fn fill_fixture(components: &[&str]) -> CairoClaimGenerator {
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

    let mut cg = CairoClaimGenerator::default();
    let mut set: IndexSet<&str> = IndexSet::new();
    for c in components {
        set.insert(c);
    }

    // None of the components used here consume the preprocessed trace; any variant is fine.
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
        use stwo_cairo_common::prover_types::simd::{PackedFelt252, PackedUInt32};

        use crate::witness::fast_deduction::blake::{PackedBlakeG, PackedBlakeRoundSigma};
        use crate::witness::fast_deduction::pedersen::PackedPartialEcMulWindowBits18;
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
            k => panic!("unexpected deduce kind {k}"),
        }
    }
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
    let cg = fill_fixture(&[
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

    let n_packed_rows = 1usize << (diff.log_size - LOG_N_LANES);
    for r in 0..(1usize << diff.log_size) {
        // Slot layout: flat input words 0..19 (m31, m31, 16 raw u32 words, m31),
        // enabler 19, iota 20.
        let src = &padded[r];
        let mut row_inputs: Vec<u32> = vec![src.0 .0, src.1 .0];
        row_inputs.extend(src.2 .0.iter().map(|w| w.value));
        row_inputs.push(src.2 .1 .0);
        row_inputs.push(u32::from(r < n_rows)); // enabler
        row_inputs.push(r as u32); // iota (unused by this body)
        let ro = interpret_row_with(&out.program, &row_inputs, &oracle, &mut FastDeductionHost);

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

    let n_packed_rows = 1usize << (diff.log_size - LOG_N_LANES);
    for r in 0..(1usize << diff.log_size) {
        // Slot layout: inputs 0..3 (in.0[0], in.0[1], in.1), enabler 3, iota 4,
        // mults[0] 5.
        let src = &inputs[r];
        let row_inputs: Vec<u32> = vec![
            src.0[0].0,
            src.0[1].0,
            src.1 .0,
            u32::from(r < n_rows),
            r as u32,
            mults[r].0,
        ];
        let ro = interpret_row_with(&out.program, &row_inputs, &oracle, &mut FastDeductionHost);

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
}
