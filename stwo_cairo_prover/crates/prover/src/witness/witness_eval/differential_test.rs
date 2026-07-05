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
    fill_fixture_with_memory(components).0
}

/// [`fill_fixture`], additionally returning the raw memory tables (addr→id ids,
/// f252 values, small values) that the DEVICE execution tables upload from — the
/// claim generator consumes the `Memory` itself.
fn fill_fixture_with_memory(
    components: &[&str],
) -> (CairoClaimGenerator, Vec<u32>, Vec<[u32; 8]>, Vec<u128>) {
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
    cg.fill_components(
        &set,
        state_transitions.casm_states_by_opcode,
        &builtin_segments,
        Arc::new(memory),
        preprocessed_trace,
    );
    (cg, addr_ids, f252_values, small_values)
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
        use crate::witness::fast_deduction::pedersen::{
            PackedPartialEcMulWindowBits18, PackedPedersenPointsTableWindowBits18,
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
            k => panic!("unexpected deduce kind {k}"),
        }
    }
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
            label, &cols, &tables, true,
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
    let (cg, addr_ids, f252_values, small_values) = fill_fixture_with_memory(&[
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
}

/// POD ORACLE LEGS (deduce kinds 2/3): the precompiled `stwo_wit_deduce_*` device
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
    // (No pedersen table: generic's felts arrive as inputs; its EC math is
    // inline felt arithmetic, not table-driven.)
    let host_rows: Vec<Vec<M31>> = diff.orig_rows.iter().map(|r| r.to_vec()).collect();
    assert_device_builtin_leg_matches_host(
        "partial_ec_mul_generic",
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

// ---------------- poseidon-family fp256: cube_252 + range_check_252_width_27 --------

/// Shared fixture prep: the poseidon fixture fed through the production chain
/// (poseidon_builtin -> aggregator -> full/partial round chains), which fills
/// cube_252 and range_check_252_width_27 inputs.
#[allow(clippy::type_complexity)]
fn poseidon_family_fixture() -> (crate::witness::cairo_claim_generator::CairoClaimGenerator,) {
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
    use crate::witness::components::{cube_252, range_check_252_width_27};
    let c = cube_252::record_cube_252();
    assert!(c.poison_ops.is_empty(), "cube poisons: {:?}", c.poison_ops);
    let r = range_check_252_width_27::record_range_check_252_width_27();
    assert!(r.poison_ops.is_empty(), "rc252 poisons: {:?}", r.poison_ops);
    eprintln!(
        "cube_252 instrs={} rc252w27 instrs={}",
        c.program.n_instrs(),
        r.program.n_instrs()
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
    mut host_feed: impl FnMut(&'static str, u32, &[u32]),
) -> usize {
    use crate::witness::device_feed::{
        build_feed_descriptors, host_feed_counts, COUNT_RELATIONS, WFC_DESC_STRIDE,
    };
    let (descs, lut_slots, counts_slots) = build_feed_descriptors(layout, COUNT_RELATIONS);
    if descs.is_empty() {
        return 0;
    }
    let luts: Vec<Vec<u32>> = lut_slots.iter().map(|s| lut_for(s)).collect();
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
        let mut tuple = vec![0u32; words];
        for r in 0..n_padded {
            let mut key: u64 = 0;
            for (k, t) in tuple.iter_mut().enumerate() {
                *t = sub_flat[(base + k) * n_padded + r];
                key = (key << relation.word_bits[k]) | u64::from(*t);
            }
            if key as usize >= relation.table_size {
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
        let (cg, _a, _f, _s) = fill_fixture_with_memory(&[
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
        let n = run_count_feed_paths(
            blake_round::SUB_FEED_LAYOUT,
            &sub_flat,
            size,
            |f| match f {
                "range_check_7_2_5_state" => rc725.input_to_row_lut(),
                other => panic!("unexpected LUT family {other}"),
            },
            |f, c| match f {
                "range_check_7_2_5_state" => d725.add_count_tables(c),
                other => panic!("unexpected count family {other}"),
            },
            |f, rel, t| match f {
                "range_check_7_2_5_state" => {
                    h725.add_input(&[M31(t[0]), M31(t[1]), M31(t[2])], rel as usize)
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
        eprintln!("count gate [blake_round]: PASS ({n} descriptors)");
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
