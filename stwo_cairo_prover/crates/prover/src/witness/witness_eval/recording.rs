//! [`RecordingWitnessEval`] — the JIT-recording [`WitnessEval`] impl.
//!
//! Wraps a [`WitnessRecorder`] (from the stwo fork's `jit_witness` lane). Instantiating a
//! generic per-row body on this evaluator emits the per-row scalar-SSA
//! [`WitnessProgram`] bytecode "for free" — the same record→codegen recipe the constraint
//! JIT lane uses, but for witness generation.
//!
//! # Value handles
//!
//! Values are opaque SSA-register handles wrapped in [`RecVal`]. The 32-bit witness ISA
//! has no inverse / compare / mask / select ops, so the EXTENDED trait methods
//! (`m31_inverse`, `m31_eq`, `mask_*`, `select`) return [`RecVal::Poison`] and emit NO
//! instruction. Any op consuming a `Poison` input also returns `Poison` (poison
//! propagation), so the recordable *prefix* of every writer is captured losslessly — and
//! every poison is an honest marker of an ISA-V2 gap, censused in
//! [`RecordingOutput::poison_ops`]. (For the `assert_eq_opcode` pilot the body uses no
//! EXTENDED op, so nothing is poisoned — the whole decode records.)

use std::collections::{BTreeMap, BTreeSet};

use stwo_backend_cuda::jit_witness::isa::WitnessProgram;
use stwo_backend_cuda::jit_witness::recording::{Val, WitnessRecorder};

use crate::witness::witness_eval::{
    WitnessEval, FELT_N_LIMBS, SLOT_ENABLER, TABLE_ADDR_TO_ID, TABLE_ID_TO_BIG,
};

/// An SSA-register handle, or a poison marker for an op the 32-bit ISA cannot express.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RecVal {
    Ok(Val),
    Poison,
}

/// A recorded felt handle: either a keyed table read (`memory_id_to_big`) whose limbs are
/// materialized lazily, or an explicit bundle of limb handles (`felt_from_limbs`).
#[derive(Clone, Debug)]
pub enum RecFelt {
    /// `mem_id_to_value` result — `felt_get_m31(_, i)` reads `table_limb(TABLE_ID_TO_BIG,
    /// key, i)`.
    Deduced { key: Val },
    /// `felt_from_limbs` bookkeeping — `felt_get_m31(_, i)` is `v[i]` (no emit).
    Limbs(Vec<RecVal>),
}

/// The output of recording one per-row body: the [`WitnessProgram`] plus the honest
/// census of what could NOT be recorded (the ISA-V2 backlog, not a failure).
#[derive(Clone, Debug)]
pub struct RecordingOutput {
    /// The recorded per-row bytecode.
    pub program: WitnessProgram,
    /// Columns whose value was poisoned (not committed to the program).
    pub poisoned_cols: BTreeSet<usize>,
    /// Lookup words whose value was poisoned (not emitted).
    pub poisoned_lookup_words: BTreeSet<usize>,
    /// Sub-input words whose value was poisoned (not emitted).
    pub poisoned_sub_words: BTreeSet<usize>,
    /// Census of EXTENDED / poison-producing ops encountered: op name → count.
    pub poison_ops: BTreeMap<&'static str, usize>,
}

/// Records a generic per-row witness body into [`WitnessProgram`] bytecode.
pub struct RecordingWitnessEval {
    recorder: WitnessRecorder,
    poisoned_cols: BTreeSet<usize>,
    poisoned_lookup_words: BTreeSet<usize>,
    poisoned_sub_words: BTreeSet<usize>,
    poison_ops: BTreeMap<&'static str, usize>,
}

impl RecordingWitnessEval {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            recorder: WitnessRecorder::new(label),
            poisoned_cols: BTreeSet::new(),
            poisoned_lookup_words: BTreeSet::new(),
            poisoned_sub_words: BTreeSet::new(),
            poison_ops: BTreeMap::new(),
        }
    }

    /// Finish recording and package the program + poison census.
    pub fn finish(self) -> RecordingOutput {
        RecordingOutput {
            program: self.recorder.finish(),
            poisoned_cols: self.poisoned_cols,
            poisoned_lookup_words: self.poisoned_lookup_words,
            poisoned_sub_words: self.poisoned_sub_words,
            poison_ops: self.poison_ops,
        }
    }

    /// Census a poison-producing EXTENDED op and return the poison marker (emits nothing).
    #[inline]
    fn poison(&mut self, op: &'static str) -> RecVal {
        *self.poison_ops.entry(op).or_insert(0) += 1;
        RecVal::Poison
    }

    /// Poison-propagating unary ISA-core op.
    #[inline]
    fn un(&mut self, a: RecVal, f: impl FnOnce(&mut WitnessRecorder, Val) -> Val) -> RecVal {
        match a {
            RecVal::Ok(x) => RecVal::Ok(f(&mut self.recorder, x)),
            RecVal::Poison => RecVal::Poison,
        }
    }

    /// Poison-propagating binary ISA-core op.
    #[inline]
    fn bin(
        &mut self,
        a: RecVal,
        b: RecVal,
        f: impl FnOnce(&mut WitnessRecorder, Val, Val) -> Val,
    ) -> RecVal {
        match (a, b) {
            (RecVal::Ok(x), RecVal::Ok(y)) => RecVal::Ok(f(&mut self.recorder, x, y)),
            _ => RecVal::Poison,
        }
    }
}

impl WitnessEval for RecordingWitnessEval {
    type M31 = RecVal;
    type U16 = RecVal;
    type Mask = RecVal;
    type Felt = RecFelt;

    // ---- Leaves ----------------------------------------------------------------

    #[inline]
    fn input(&mut self, slot: u32) -> RecVal {
        RecVal::Ok(self.recorder.input(slot))
    }
    #[inline]
    fn m31_const(&mut self, value: u32) -> RecVal {
        RecVal::Ok(self.recorder.constant(value))
    }
    #[inline]
    fn enabler(&mut self) -> RecVal {
        RecVal::Ok(self.recorder.input(SLOT_ENABLER))
    }

    // ---- M31 field ops (ISA-core) ----------------------------------------------

    #[inline]
    fn m31_add(&mut self, a: RecVal, b: RecVal) -> RecVal {
        self.bin(a, b, |r, x, y| r.m31_add(x, y))
    }
    #[inline]
    fn m31_sub(&mut self, a: RecVal, b: RecVal) -> RecVal {
        self.bin(a, b, |r, x, y| r.m31_sub(x, y))
    }
    #[inline]
    fn m31_mul(&mut self, a: RecVal, b: RecVal) -> RecVal {
        self.bin(a, b, |r, x, y| r.m31_mul(x, y))
    }

    // ---- EXTENDED (ISA-V2: masks are 0/1 registers; `m31_eq` produces one,
    // ---- the mask combinators lower to plain field arithmetic on 0/1 values,
    // ---- and the conversions are identities — byte-identical to the SIMD
    // ---- PackedBool semantics on every path the writers exercise) --------------

    #[inline]
    fn m31_inverse(&mut self, a: RecVal) -> RecVal {
        self.un(a, |r, x| r.m31_inverse(x))
    }
    #[inline]
    fn m31_eq(&mut self, a: RecVal, b: RecVal) -> RecVal {
        self.bin(a, b, |r, x, y| r.m31_eq(x, y))
    }
    #[inline]
    fn mask_and(&mut self, a: RecVal, b: RecVal) -> RecVal {
        // AND of 0/1 masks == product.
        self.bin(a, b, |r, x, y| r.m31_mul(x, y))
    }
    #[inline]
    fn mask_as_m31(&mut self, a: RecVal) -> RecVal {
        a
    }
    #[inline]
    fn mask_from_m31(&mut self, a: RecVal) -> RecVal {
        // The writers only convert 0/1 factors (`PackedBool::from_m31`'s own
        // contract), so the register already IS the mask representation.
        a
    }
    #[inline]
    fn select(&mut self, m: RecVal, a: RecVal, b: RecVal) -> RecVal {
        // Lane-safe conditional over a 0/1 mask: m*a + (1-m)*b.
        let one = self.m31_const(1);
        let not_m = self.bin(one, m, |r, x, y| r.m31_sub(x, y));
        let ma = self.bin(m, a, |r, x, y| r.m31_mul(x, y));
        let nb = self.bin(not_m, b, |r, x, y| r.m31_mul(x, y));
        self.bin(ma, nb, |r, x, y| r.m31_add(x, y))
    }

    // ---- u16 integer / bit ops (ISA-core) --------------------------------------

    #[inline]
    fn u16_from_m31(&mut self, a: RecVal) -> RecVal {
        self.un(a, |r, x| r.from_m31(x))
    }
    #[inline]
    fn u16_as_m31(&mut self, a: RecVal) -> RecVal {
        self.un(a, |r, x| r.as_m31(x))
    }
    #[inline]
    fn u16_add(&mut self, a: RecVal, b: RecVal) -> RecVal {
        self.bin(a, b, |r, x, y| r.u16_add(x, y))
    }
    #[inline]
    fn u16_shl(&mut self, a: RecVal, imm: u32) -> RecVal {
        self.un(a, |r, x| r.u16_shl(x, imm))
    }
    #[inline]
    fn u16_shr(&mut self, a: RecVal, imm: u32) -> RecVal {
        self.un(a, |r, x| r.u16_shr(x, imm))
    }
    #[inline]
    fn u16_and(&mut self, a: RecVal, mask: u32) -> RecVal {
        self.un(a, |r, x| r.u16_and(x, mask))
    }
    #[inline]
    fn u16_xor(&mut self, a: RecVal, b: RecVal) -> RecVal {
        // Bit-identical to a dedicated U16Xor because both operands are `< 2^16`.
        self.bin(a, b, |r, x, y| r.u32_xor(x, y))
    }

    // ---- Felt (bookkeeping) ----------------------------------------------------

    #[inline]
    fn felt_from_limbs(&mut self, limbs: [RecVal; FELT_N_LIMBS]) -> RecFelt {
        RecFelt::Limbs(limbs.to_vec())
    }
    #[inline]
    fn felt_get_m31(&mut self, felt: &RecFelt, i: usize) -> RecVal {
        match felt {
            RecFelt::Deduced { key } => {
                RecVal::Ok(self.recorder.table_limb(TABLE_ID_TO_BIG, *key, i as u32))
            }
            RecFelt::Limbs(v) => v[i],
        }
    }

    // ---- Memory ops (`mem_read` uses the trait default) ------------------------

    #[inline]
    fn mem_addr_to_id(&mut self, addr: RecVal) -> RecVal {
        match addr {
            RecVal::Ok(a) => RecVal::Ok(self.recorder.table_limb(TABLE_ADDR_TO_ID, a, 0)),
            RecVal::Poison => RecVal::Poison,
        }
    }
    #[inline]
    fn mem_id_to_value(&mut self, id: RecVal) -> RecFelt {
        match id {
            RecVal::Ok(key) => RecFelt::Deduced { key },
            // No Val to key the table read on: degrade to an all-poison limb bundle.
            RecVal::Poison => RecFelt::Limbs(vec![RecVal::Poison; FELT_N_LIMBS]),
        }
    }

    // ---- Effects ---------------------------------------------------------------

    #[inline]
    fn set_col(&mut self, col: usize, value: RecVal) {
        match value {
            RecVal::Ok(val) => self.recorder.col_write(col as u32, val),
            RecVal::Poison => {
                self.poisoned_cols.insert(col);
            }
        }
    }
    #[inline]
    fn set_lookup_word(&mut self, word: usize, value: RecVal) {
        match value {
            RecVal::Ok(val) => self.recorder.lookup_word(word as u32, val),
            RecVal::Poison => {
                self.poisoned_lookup_words.insert(word);
            }
        }
    }
    #[inline]
    fn set_sub_input_word(&mut self, word: usize, value: RecVal) {
        // Sub-component inputs are first-class ISA effects (`WitnessOp::SubWord`): the
        // kernel stores them into a flat per-row buffer that the prove-path hook D2H\'s
        // and feeds to the sibling generators exactly as the host writer\'s
        // `SubComponentInputs` drain does. NOT derivable from lookup words in general
        // (add_opcode\'s verify_instruction sub-tuple uses different intermediates than
        // its lookup tuple), hence the dedicated effect. Poison keeps the word out of
        // the recording, matching the lookup-word rule.
        match value {
            RecVal::Ok(val) => self.recorder.sub_word(word as u32, val),
            RecVal::Poison => {
                self.poisoned_sub_words.insert(word);
            }
        }
    }
}
