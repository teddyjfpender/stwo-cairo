//! [`SimdWitnessEval`] — the zero-cost passthrough [`WitnessEval`] impl.
//!
//! Every method is `#[inline(always)]` and lowers directly to the exact `PackedM31` /
//! `PackedUInt16` / `PackedBool` / `PackedFelt252` operation the original monomorphic
//! `write_trace_simd` body used. Instantiating a generic per-row body on this evaluator
//! therefore reproduces the original host SIMD writer **byte-for-byte** (the safety gate
//! — see [`super`] and `witness_eval::differential_test`).
//!
//! # Effects
//!
//! `set_col(i, v)` writes `*self.row[i] = v` straight into the committed
//! [`ComponentTrace`](stwo_air_utils::trace::component_trace::ComponentTrace) column.
//! The lookup-tuple words and sub-component-input words, however, are addressed by *flat
//! index* through the trait (the driver reconstructs the concrete typed `LookupData` /
//! `SubComponentInputs` only after the row body finishes), so they land in flat scratch
//! [`Vec`]s that the driver reads back and reshapes. See
//! [`SimdWitnessEval::lookup_scratch`] / [`SimdWitnessEval::sub_scratch`].

use crate::witness::components::{memory_address_to_id, memory_id_to_big};
use crate::witness::prelude::*;
use crate::witness::witness_eval::{WitnessEval, FELT_N_LIMBS, SLOT_AP, SLOT_FP, SLOT_PC};

/// Passthrough [`WitnessEval`] holding the mutable handles for one packed row.
///
/// `'trace` is the lifetime of the committed trace column borrows, `'a` the lifetime of
/// the borrowed sibling-component deduce states, and `N` the trace column count.
pub struct SimdWitnessEval<'a, 'trace, const N: usize> {
    /// Mutable handle to the current packed row's `N` trace columns (`set_col` target).
    row: Box<[&'trace mut PackedM31; N]>,
    /// `memory_address_to_id.deduce_output` device/host table.
    mem_addr_state: &'a memory_address_to_id::ClaimGenerator,
    /// `memory_id_to_big.deduce_output` device/host table.
    mem_big_state: &'a memory_id_to_big::ClaimGenerator,
    /// This packed row's `(pc, ap, fp)` input (the `input()` leaves).
    input: PackedCasmState,
    /// This packed row's index (for the enabler column).
    row_index: usize,
    /// The enabler column (1 for real rows, 0 for padding).
    enabler: &'a Enabler,
    /// Flat lookup-tuple words written by `set_lookup_word`, reshaped by the driver.
    lookup_scratch: Vec<PackedM31>,
    /// Flat sub-component-input words written by `set_sub_input_word`, reshaped by the
    /// driver.
    sub_scratch: Vec<PackedM31>,
}

impl<'a, 'trace, const N: usize> SimdWitnessEval<'a, 'trace, N> {
    /// Construct the evaluator for one packed row. `n_lookup_words` / `n_sub_words` size
    /// the flat scratch to the component's `LookupData` / `SubComponentInputs` word count
    /// (the driver knows these).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row: Box<[&'trace mut PackedM31; N]>,
        mem_addr_state: &'a memory_address_to_id::ClaimGenerator,
        mem_big_state: &'a memory_id_to_big::ClaimGenerator,
        input: PackedCasmState,
        row_index: usize,
        enabler: &'a Enabler,
        n_lookup_words: usize,
        n_sub_words: usize,
    ) -> Self {
        Self {
            row,
            mem_addr_state,
            mem_big_state,
            input,
            row_index,
            enabler,
            lookup_scratch: vec![PackedM31::zero(); n_lookup_words],
            sub_scratch: vec![PackedM31::zero(); n_sub_words],
        }
    }

    /// Flat lookup-tuple words (declaration order across `LookupData`).
    #[inline(always)]
    pub fn lookup_scratch(&self) -> &[PackedM31] {
        &self.lookup_scratch
    }

    /// Flat sub-component-input words (declaration order across `SubComponentInputs`).
    #[inline(always)]
    pub fn sub_scratch(&self) -> &[PackedM31] {
        &self.sub_scratch
    }
}

impl<const N: usize> WitnessEval for SimdWitnessEval<'_, '_, N> {
    type M31 = PackedM31;
    type U16 = PackedUInt16;
    type Mask = PackedBool;
    type Felt = PackedFelt252;

    // ---- Leaves ----------------------------------------------------------------

    #[inline(always)]
    fn input(&mut self, slot: u32) -> PackedM31 {
        match slot {
            SLOT_PC => self.input.pc,
            SLOT_AP => self.input.ap,
            SLOT_FP => self.input.fp,
            _ => panic!("SimdWitnessEval::input: unexpected slot {slot}"),
        }
    }

    #[inline(always)]
    fn m31_const(&mut self, value: u32) -> PackedM31 {
        PackedM31::broadcast(M31::from(value))
    }

    #[inline(always)]
    fn enabler(&mut self) -> PackedM31 {
        self.enabler.packed_at(self.row_index)
    }

    // ---- M31 field ops (ISA-core) ----------------------------------------------

    #[inline(always)]
    fn m31_add(&mut self, a: PackedM31, b: PackedM31) -> PackedM31 {
        a + b
    }
    #[inline(always)]
    fn m31_sub(&mut self, a: PackedM31, b: PackedM31) -> PackedM31 {
        a - b
    }
    #[inline(always)]
    fn m31_mul(&mut self, a: PackedM31, b: PackedM31) -> PackedM31 {
        a * b
    }

    // ---- M31 field ops (EXTENDED) ----------------------------------------------

    #[inline(always)]
    fn m31_inverse(&mut self, a: PackedM31) -> PackedM31 {
        a.inverse()
    }
    #[inline(always)]
    fn m31_eq(&mut self, a: PackedM31, b: PackedM31) -> PackedBool {
        EqExtend::eq(&a, b)
    }

    // ---- Masks + lane-wise select (EXTENDED) -----------------------------------

    #[inline(always)]
    fn mask_and(&mut self, a: PackedBool, b: PackedBool) -> PackedBool {
        a & b
    }
    #[inline(always)]
    fn mask_as_m31(&mut self, a: PackedBool) -> PackedM31 {
        a.as_m31()
    }
    #[inline(always)]
    fn mask_from_m31(&mut self, a: PackedM31) -> PackedBool {
        PackedBool::from_m31(a)
    }
    #[inline(always)]
    fn select(&mut self, m: PackedBool, a: PackedM31, b: PackedM31) -> PackedM31 {
        // Lane-wise `m ? a : b`: `f*a + (1 - f)*b` where `f` is the 0/1 mask factor.
        let f = m.as_m31();
        let one = PackedM31::broadcast(M31::from(1));
        f * a + (one - f) * b
    }

    // ---- u16 integer / bit ops (ISA-core) --------------------------------------

    #[inline(always)]
    fn u16_from_m31(&mut self, a: PackedM31) -> PackedUInt16 {
        PackedUInt16::from_m31(a)
    }
    #[inline(always)]
    fn u16_as_m31(&mut self, a: PackedUInt16) -> PackedM31 {
        a.as_m31()
    }
    #[inline(always)]
    fn u16_add(&mut self, a: PackedUInt16, b: PackedUInt16) -> PackedUInt16 {
        a + b
    }
    #[inline(always)]
    fn u16_shl(&mut self, a: PackedUInt16, imm: u32) -> PackedUInt16 {
        a << PackedUInt16::broadcast(UInt16::from(imm as u16))
    }
    #[inline(always)]
    fn u16_shr(&mut self, a: PackedUInt16, imm: u32) -> PackedUInt16 {
        a >> PackedUInt16::broadcast(UInt16::from(imm as u16))
    }
    #[inline(always)]
    fn u16_and(&mut self, a: PackedUInt16, mask: u32) -> PackedUInt16 {
        a & PackedUInt16::broadcast(UInt16::from(mask as u16))
    }
    #[inline(always)]
    fn u16_xor(&mut self, a: PackedUInt16, b: PackedUInt16) -> PackedUInt16 {
        a ^ b
    }

    // ---- Felt (bookkeeping) ----------------------------------------------------

    #[inline(always)]
    fn felt_from_limbs(&mut self, limbs: [PackedM31; FELT_N_LIMBS]) -> PackedFelt252 {
        PackedFelt252::from_limbs(limbs)
    }
    #[inline(always)]
    fn felt_get_m31(&mut self, felt: &PackedFelt252, i: usize) -> PackedM31 {
        felt.get_m31(i)
    }

    // ---- Memory ops (keystone binding; `mem_read` uses the trait default) ------

    #[inline(always)]
    fn mem_addr_to_id(&mut self, addr: PackedM31) -> PackedM31 {
        self.mem_addr_state.deduce_output(addr)
    }
    #[inline(always)]
    fn mem_id_to_value(&mut self, id: PackedM31) -> PackedFelt252 {
        self.mem_big_state.deduce_output(id)
    }

    // ---- Effects ---------------------------------------------------------------

    #[inline(always)]
    fn set_col(&mut self, col: usize, value: PackedM31) {
        *self.row[col] = value;
    }
    #[inline(always)]
    fn set_lookup_word(&mut self, word: usize, value: PackedM31) {
        self.lookup_scratch[word] = value;
    }
    #[inline(always)]
    fn set_sub_input_word(&mut self, word: usize, value: PackedM31) {
        self.sub_scratch[word] = value;
    }
}
