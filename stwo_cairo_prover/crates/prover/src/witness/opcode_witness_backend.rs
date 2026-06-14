//! Backend-specific witness generation for the opcode cohort (witness-on-GPU
//! W3, round 12): per-opcode base-trace kernels on the generic lane, with the
//! memory deduce lookups fused as device-table gathers.
//!
//! Layout per opcode (the verify_instruction recipe, generalized):
//! - `into_parts()` on the component's `ClaimGenerator` yields the padded raw inputs + the live row
//!   count (padding repeats `inputs[0]`; `Enabler` multiplicities handle the padding semantics).
//! - One base-trace kernel in `opcodes.cu` writes the trace columns plus the staged tuple columns
//!   (lookup expressions that are not plain trace columns). Memory reads gather from
//!   [`DeviceMemTables`] — the prove-wide raw tables uploaded once by
//!   [`OpcodeWitness::build_mem_tables`].
//! - Sub-component feeds run host-side from the same raw inputs and host tables (cheap u32 loops,
//!   value-identical to the writer's `sub_component_inputs` feeding; the vi feed's dedup dashmap is
//!   data-dependent, so device feeds are a later round).
//! - The interaction columns come from the lane's `tuple_pair_logup_slots` /
//!   `tuple_single_logup_slots` (constants fold into the combine base host-side; trailing
//!   constant-zero tuple terms are truncated — exact field identity).
//!
//! Gates: `STWO_CUDA_WITNESS_VERIFY=1` byte-compares every trace column, the
//! feed deltas and the interaction columns + sums against the host writer;
//! the Cairo e2e proof byte-equality (CUDA vs SIMD) is decisive.
//! `STWO_CUDA_RET_WITNESS=0` disables the device path per component.

use cairo_air::components::add_ap_opcode::{Claim as AddApClaim, N_TRACE_COLUMNS as ADD_AP_N_COLS};
use cairo_air::components::add_opcode::{Claim as AddClaim, N_TRACE_COLUMNS as ADD_N_COLS};
use cairo_air::components::add_opcode_small::{
    Claim as AddSmallClaim, N_TRACE_COLUMNS as ADD_SMALL_N_COLS,
};
use cairo_air::components::assert_eq_opcode::{
    Claim as AssertEqClaim, N_TRACE_COLUMNS as ASSERT_EQ_N_COLS,
};
use cairo_air::components::assert_eq_opcode_double_deref::{
    Claim as AssertEqDDerefClaim, N_TRACE_COLUMNS as ASSERT_EQ_DDREF_N_COLS,
};
use cairo_air::components::assert_eq_opcode_imm::{
    Claim as AssertEqImmClaim, N_TRACE_COLUMNS as ASSERT_EQ_IMM_N_COLS,
};
use cairo_air::components::call_opcode_rel_imm::{
    Claim as CallRelImmClaim, N_TRACE_COLUMNS as CALL_REL_IMM_N_COLS,
};
use cairo_air::components::jnz_opcode_taken::{
    Claim as JnzTakenClaim, N_TRACE_COLUMNS as JNZ_TAKEN_N_COLS,
};
use cairo_air::components::ret_opcode::{Claim as RetClaim, N_TRACE_COLUMNS as RET_N_COLS};
use cairo_air::relations::{
    CommonLookupElements, MEMORY_ADDRESS_TO_ID_RELATION_ID, MEMORY_ID_TO_BIG_RELATION_ID,
};
use rayon::prelude::*;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_backend_cuda::{memory_witness as device_witness, BaseFieldVec, CudaBackend};
use stwo_cairo_adapter::memory::u128_to_4_limbs;
use stwo_constraint_framework::LogupFinalizeBackend;

use crate::witness::components::{
    add_ap_opcode, add_opcode, add_opcode_small, assert_eq_opcode, assert_eq_opcode_double_deref,
    assert_eq_opcode_imm, call_opcode_rel_imm, jnz_opcode_taken, memory_address_to_id,
    memory_id_to_big, range_check_11, range_check_18, ret_opcode, verify_instruction,
};
use crate::witness::memory_witness_backend::{compare_interaction, MemoryEvals};
use crate::witness::utils::{pack_values, AddInputs};

/// Relation ids that exist only as literals in the generated writers (the
/// differential pins them against the host writer's lookup tuples).
const OPCODES_RELATION_ID: u32 = 428564188;
const VERIFY_INSTRUCTION_RELATION_ID: u32 = 1719106205;
/// add_ap is the first ported opcode that feeds the range checks. These ids are
/// the literals the generated `add_ap_opcode` writer emits in its
/// `range_check_18` / `range_check_11` lookup tuples (matching the relation ids
/// the `range_check_18` / `range_check_11` components use: see
/// `cairo-air/src/components/range_check_18.rs` / `range_check_11.rs`).
const RANGE_CHECK_18_RELATION_ID: u32 = 1109051422;
const RANGE_CHECK_11_RELATION_ID: u32 = 991608089;

/// The host-side decode of one add_opcode_small row (the sub-component feed
/// values; mirrors `write_trace_simd`'s row math exactly, in M31 arithmetic).
struct AddSmallDecoded {
    offset0: M31,
    offset1: M31,
    offset2: M31,
    vi_felt5: M31,
    vi_felt6: M31,
    dst_addr: M31,
    op0_addr: M31,
    op1_addr: M31,
    dst_id: M31,
    op0_id: M31,
    op1_id: M31,
}

/// Decodes an id's first 28 9-bit limbs from the prove-wide raw tables, exactly
/// like the kernel's `mem_id_to_limbs` / the host `deduce_output`: tag = id>>30,
/// val = id & 0x3FFFFFFF; tag 1 = big (8 words), else small (u128 -> 4 words).
fn id_to_limbs(id: M31, big_values: &[[u32; 8]], small_values: &[u128]) -> [M31; 28] {
    let raw = id.0;
    let tag = raw >> 30;
    let val = (raw & 0x3FFF_FFFF) as usize;
    let mut words = [0u32; 8];
    if tag == 1 {
        words = big_values[val];
    } else {
        let limbs = u128_to_4_limbs(small_values[val]);
        words[..4].copy_from_slice(&limbs);
    }
    stwo_cairo_common::prover_types::felt::split_f252(words)
}

fn decode_add_small_row(
    state: &crate::witness::prelude::CasmState,
    memory_address_to_id: &memory_address_to_id::ClaimGenerator,
    big_values: &[[u32; 8]],
    small_values: &[u128],
) -> AddSmallDecoded {
    let m31 = M31::from;
    let pc = state.pc;
    let ap = state.ap;
    let fp = state.fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    let instr_id = memory_address_to_id.get_id(pc);
    let il = id_to_limbs(instr_id, big_values, small_values);
    let l = |j: usize| il[j].0;
    let offset0_u = l(0) + ((l(1) & 127) << 9);
    let offset1_u = (l(1) >> 7) + (l(2) << 2) + ((l(3) & 31) << 11);
    let offset2_u = (l(3) >> 5) + (l(4) << 4) + ((l(5) & 7) << 13);
    let flags = (l(5) >> 3) + (l(6) << 6);
    let dst_base_fp = (flags >> 0) & 1;
    let op0_base_fp = (flags >> 1) & 1;
    let op1_imm = (flags >> 2) & 1;
    let op1_base_fp = (flags >> 3) & 1;
    let ap_update_add_1 = (flags >> 11) & 1;

    let offset0 = m31(offset0_u);
    let offset1 = m31(offset1_u);
    let offset2 = m31(offset2_u);
    let dst_base_fp = m31(dst_base_fp);
    let op0_base_fp = m31(op0_base_fp);
    let op1_imm = m31(op1_imm);
    let op1_base_fp = m31(op1_base_fp);
    let ap_update_add_1 = m31(ap_update_add_1);
    let op1_base_ap = m31(1) - op1_imm - op1_base_fp;

    let vi_felt5 = dst_base_fp * m31(8)
        + op0_base_fp * m31(16)
        + op1_imm * m31(32)
        + op1_base_fp * m31(64)
        + op1_base_ap * m31(128)
        + m31(256);
    let vi_felt6 = ap_update_add_1 * m31(32) + m31(256);

    let mem_dst_base = dst_base_fp * fp + (m31(1) - dst_base_fp) * ap;
    let mem0_base = op0_base_fp * fp + (m31(1) - op0_base_fp) * ap;
    let mem1_base = op1_imm * pc + op1_base_fp * fp + op1_base_ap * ap;

    let dst_addr = mem_dst_base + (offset0 - m31(32768));
    let op0_addr = mem0_base + (offset1 - m31(32768));
    let op1_addr = mem1_base + (offset2 - m31(32768));
    let dst_id = memory_address_to_id.get_id(dst_addr);
    let op0_id = memory_address_to_id.get_id(op0_addr);
    let op1_id = memory_address_to_id.get_id(op1_addr);

    AddSmallDecoded {
        offset0,
        offset1,
        offset2,
        vi_felt5,
        vi_felt6,
        dst_addr,
        op0_addr,
        op1_addr,
        dst_id,
        op0_id,
        op1_id,
    }
}

/// The host-side decode of one assert_eq_opcode row (the sub-component feed
/// values; mirrors `write_trace_simd`'s row math exactly, in M31 arithmetic).
struct AssertEqDecoded {
    offset0: M31,
    offset2: M31,
    vi_felt5: M31,
    vi_felt6: M31,
    dst_addr: M31,
    op1_addr: M31,
}

fn decode_assert_eq_row(
    state: &crate::witness::prelude::CasmState,
    memory_address_to_id: &memory_address_to_id::ClaimGenerator,
    big_values: &[[u32; 8]],
    small_values: &[u128],
) -> AssertEqDecoded {
    let m31 = M31::from;
    let pc = state.pc;
    let ap = state.ap;
    let fp = state.fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    let instr_id = memory_address_to_id.get_id(pc);
    let il = id_to_limbs(instr_id, big_values, small_values);
    let l = |j: usize| il[j].0;
    let offset0_u = l(0) + ((l(1) & 127) << 9);
    let offset2_u = (l(3) >> 5) + (l(4) << 4) + ((l(5) & 7) << 13);
    let flags = (l(5) >> 3) + (l(6) << 6);
    let dst_base_fp = (flags >> 0) & 1;
    let op1_base_fp = (flags >> 3) & 1;
    let ap_update_add_1 = (flags >> 11) & 1;

    let offset0 = m31(offset0_u);
    let offset2 = m31(offset2_u);
    let dst_base_fp = m31(dst_base_fp);
    let op1_base_fp = m31(op1_base_fp);
    let ap_update_add_1 = m31(ap_update_add_1);

    let vi_felt5 = ((dst_base_fp * m31(8) + m31(16)) + op1_base_fp * m31(64))
        + (m31(1) - op1_base_fp) * m31(128);
    let vi_felt6 = ap_update_add_1 * m31(32) + m31(256);

    let mem_dst_base = dst_base_fp * fp + (m31(1) - dst_base_fp) * ap;
    let mem1_base = op1_base_fp * fp + (m31(1) - op1_base_fp) * ap;

    let dst_addr = mem_dst_base + (offset0 - m31(32768));
    let op1_addr = mem1_base + (offset2 - m31(32768));

    AssertEqDecoded {
        offset0,
        offset2,
        vi_felt5,
        vi_felt6,
        dst_addr,
        op1_addr,
    }
}

/// The host-side decode of one assert_eq_opcode_imm row (the sub-component feed
/// values; mirrors `write_trace_simd`'s row math exactly, in M31 arithmetic).
/// The immediate sibling reads the second operand from pc+1 (the immediate), so
/// there is no op1 base / offset2 decode — the verify_instruction tuple uses the
/// constants 32767 / 32769 for offset1 / offset2.
struct AssertEqImmDecoded {
    offset0: M31,
    vi_felt5: M31,
    vi_felt6: M31,
    dst_addr: M31,
    imm_addr: M31,
}

fn decode_assert_eq_imm_row(
    state: &crate::witness::prelude::CasmState,
    memory_address_to_id: &memory_address_to_id::ClaimGenerator,
    big_values: &[[u32; 8]],
    small_values: &[u128],
) -> AssertEqImmDecoded {
    let m31 = M31::from;
    let pc = state.pc;
    let ap = state.ap;
    let fp = state.fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    let instr_id = memory_address_to_id.get_id(pc);
    let il = id_to_limbs(instr_id, big_values, small_values);
    let l = |j: usize| il[j].0;
    let offset0_u = l(0) + ((l(1) & 127) << 9);
    let flags = (l(5) >> 3) + (l(6) << 6);
    let dst_base_fp = (flags >> 0) & 1;
    let ap_update_add_1 = (flags >> 11) & 1;

    let offset0 = m31(offset0_u);
    let dst_base_fp = m31(dst_base_fp);
    let ap_update_add_1 = m31(ap_update_add_1);

    let vi_felt5 = (dst_base_fp * m31(8) + m31(16)) + m31(32);
    let vi_felt6 = ap_update_add_1 * m31(32) + m31(256);

    let mem_dst_base = dst_base_fp * fp + (m31(1) - dst_base_fp) * ap;

    let dst_addr = mem_dst_base + (offset0 - m31(32768));
    let imm_addr = pc + m31(1);

    AssertEqImmDecoded {
        offset0,
        vi_felt5,
        vi_felt6,
        dst_addr,
        imm_addr,
    }
}

/// The host-side decode of one assert_eq_opcode_double_deref row (the
/// sub-component feed values; mirrors `write_trace_simd`'s row math exactly, in
/// M31 arithmetic). The double-dereference reads the op0 operand as a pointer
/// (mem1_base_id at mem0_base + (offset1-32768)), recombines its 4 limbs into a
/// memory address, then reads the asserted-equal cell (the SAME id as dst) at
/// that pointer + (offset2-32768). It feeds verify_instruction once,
/// memory_address_to_id three times (mem1_base / dst / ddref reads) and
/// memory_id_to_big once (the pointer id read as a 29-bit value).
struct AssertEqDDerefDecoded {
    offset0: M31,
    offset1: M31,
    offset2: M31,
    vi_felt5: M31,
    vi_felt6: M31,
    mem1_base_addr: M31,
    dst_addr: M31,
    ddref_addr: M31,
    mem1_base_id: M31,
}

fn decode_assert_eq_ddref_row(
    state: &crate::witness::prelude::CasmState,
    memory_address_to_id: &memory_address_to_id::ClaimGenerator,
    big_values: &[[u32; 8]],
    small_values: &[u128],
) -> AssertEqDDerefDecoded {
    let m31 = M31::from;
    let pc = state.pc;
    let ap = state.ap;
    let fp = state.fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    let instr_id = memory_address_to_id.get_id(pc);
    let il = id_to_limbs(instr_id, big_values, small_values);
    let l = |j: usize| il[j].0;
    let offset0_u = l(0) + ((l(1) & 127) << 9);
    let offset1_u = (l(1) >> 7) + (l(2) << 2) + ((l(3) & 31) << 11);
    let offset2_u = (l(3) >> 5) + (l(4) << 4) + ((l(5) & 7) << 13);
    let flags = (l(5) >> 3) + (l(6) << 6);
    let dst_base_fp = (flags >> 0) & 1;
    let op0_base_fp = (flags >> 1) & 1;
    let ap_update_add_1 = (flags >> 11) & 1;

    let offset0 = m31(offset0_u);
    let offset1 = m31(offset1_u);
    let offset2 = m31(offset2_u);
    let dst_base_fp = m31(dst_base_fp);
    let op0_base_fp = m31(op0_base_fp);
    let ap_update_add_1 = m31(ap_update_add_1);

    let vi_felt5 = dst_base_fp * m31(8) + op0_base_fp * m31(16);
    let vi_felt6 = ap_update_add_1 * m31(32) + m31(256);

    let mem_dst_base = dst_base_fp * fp + (m31(1) - dst_base_fp) * ap;
    let mem0_base = op0_base_fp * fp + (m31(1) - op0_base_fp) * ap;

    // First deref: read the pointer's id at mem0_base + (offset1 - 32768).
    let mem1_base_addr = mem0_base + (offset1 - m31(32768));
    let mem1_base_id = memory_address_to_id.get_id(mem1_base_addr);

    // dst read at mem_dst_base + (offset0 - 32768).
    let dst_addr = mem_dst_base + (offset0 - m31(32768));

    // Second deref: recombine the pointer's first 4 limbs into a memory address
    // (limb0 + limb1*512 + limb2*262144 + limb3*134217728), then add the signed
    // offset2. memory_address_to_id_4 reads dst_id at this address.
    let pl = id_to_limbs(mem1_base_id, big_values, small_values);
    let ptr =
        pl[0] + pl[1] * m31(512) + pl[2] * m31(262144) + pl[3] * m31(134217728);
    let ddref_addr = ptr + (offset2 - m31(32768));

    AssertEqDDerefDecoded {
        offset0,
        offset1,
        offset2,
        vi_felt5,
        vi_felt6,
        mem1_base_addr,
        dst_addr,
        ddref_addr,
        mem1_base_id,
    }
}

/// The host-side decode of one add_ap_opcode row (the sub-component feed values;
/// mirrors `write_trace_simd`'s row math exactly, in M31 arithmetic). add_ap
/// reads a single small-signed operand (op1, which may be the immediate at pc,
/// or an fp/ap-based cell), forms next_ap = ap + read_small, and range-checks
/// next_ap (RangeCheck29: the bottom 11 bits feed range_check_11, the remaining
/// high bits scaled by 2^20 feed range_check_18). It feeds verify_instruction
/// once, memory_address_to_id once (the op1 read), memory_id_to_big once (the
/// op1 id as a 29-bit value), range_check_18 once and range_check_11 once.
struct AddApDecoded {
    offset2: M31,
    vi_felt: M31,
    op1_addr: M31,
    op1_id: M31,
    rc29_bot11bits: M31,
    rc18_val: M31,
}

fn decode_add_ap_row(
    state: &crate::witness::prelude::CasmState,
    memory_address_to_id: &memory_address_to_id::ClaimGenerator,
    big_values: &[[u32; 8]],
    small_values: &[u128],
) -> AddApDecoded {
    let m31 = M31::from;
    let pc = state.pc;
    let ap = state.ap;
    let fp = state.fp;

    // Decode Instruction: pc -> id -> first 7 limbs (u16 bit-extraction math).
    let instr_id = memory_address_to_id.get_id(pc);
    let il = id_to_limbs(instr_id, big_values, small_values);
    let l = |j: usize| il[j].0;
    let offset2_u = (l(3) >> 5) + (l(4) << 4) + ((l(5) & 7) << 13);
    let flags = (l(5) >> 3) + (l(6) << 6);
    let op1_imm = (flags >> 2) & 1;
    let op1_base_fp = (flags >> 3) & 1;

    let offset2 = m31(offset2_u);
    let op1_imm = m31(op1_imm);
    let op1_base_fp = m31(op1_base_fp);
    let op1_base_ap = (m31(1) - op1_imm) - op1_base_fp;

    let vi_felt = ((m31(24) + op1_imm * m31(32)) + op1_base_fp * m31(64)) + op1_base_ap * m31(128);

    // mem1_base = op1_imm*pc + op1_base_fp*fp + op1_base_ap*ap.
    let mem1_base = op1_imm * pc + op1_base_fp * fp + op1_base_ap * ap;
    // Read Small: op1 id at mem1_base + (offset2 - 32768).
    let op1_addr = mem1_base + (offset2 - m31(32768));
    let op1_id = memory_address_to_id.get_id(op1_addr);

    // Decode Small Sign + recombine the read value (read_small_output.0).
    let ol = id_to_limbs(op1_id, big_values, small_values);
    let msb = if ol[27] == m31(256) { m31(1) } else { m31(0) };
    let mid_limbs_set = if ol[20] == m31(511) && msb == m31(1) {
        m31(1)
    } else {
        m31(0)
    };
    let remainder_bits = m31(ol[3].0 & 3);
    let read_small = ((((ol[0] + ol[1] * m31(512)) + ol[2] * m31(262144))
        + remainder_bits * m31(134217728))
        - msb)
        - m31(536870912) * mid_limbs_set;

    // next_ap = ap + read_small; RangeCheck29 split of next_ap.
    let next_ap = ap + read_small;
    let rc29_bot11bits = m31(next_ap.0 & 2047);
    let rc18_val = (next_ap - rc29_bot11bits) * m31(1048576);

    AddApDecoded {
        offset2,
        vi_felt,
        op1_addr,
        op1_id,
        rc29_bot11bits,
        rc18_val,
    }
}

/// Backend hook for the opcode-cohort witness writers. One trait for the whole
/// cohort (one bound in `prover.rs`); each ported opcode adds a method pair +
/// an interaction-generator associated type.
pub trait OpcodeWitness: FromSimdColumns + LogupFinalizeBackend {
    /// Prove-wide device copies of the memory tables the opcode kernels
    /// gather from. Built once, before the opcode write scope; unit for SIMD.
    type MemTables: Send + Sync;
    type RetInteractionGen: Send;
    type AddSmallInteractionGen: Send;
    type AddOpcodeInteractionGen: Send;
    type AddApInteractionGen: Send;
    type JnzTakenInteractionGen: Send;
    type AssertEqInteractionGen: Send;
    type AssertEqImmInteractionGen: Send;
    type AssertEqDDerefInteractionGen: Send;
    type CallRelImmInteractionGen: Send;

    fn build_mem_tables(
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables;

    fn write_add_opcode_trace(
        gen: add_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddClaim, Self::AddOpcodeInteractionGen);

    fn write_add_opcode_interaction(
        gen: Self::AddOpcodeInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_add_opcode_small_trace(
        gen: add_opcode_small::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AddSmallClaim,
        Self::AddSmallInteractionGen,
    );

    fn write_add_opcode_small_interaction(
        gen: Self::AddSmallInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    #[allow(clippy::too_many_arguments)]
    fn write_add_ap_trace(
        gen: add_ap_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
        range_check_18: &range_check_18::ClaimGenerator,
        range_check_11: &range_check_11::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddApClaim, Self::AddApInteractionGen);

    fn write_add_ap_interaction(
        gen: Self::AddApInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_ret_trace(
        gen: ret_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, RetClaim, Self::RetInteractionGen);

    fn write_ret_interaction(
        gen: Self::RetInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_jnz_taken_trace(
        gen: jnz_opcode_taken::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        JnzTakenClaim,
        Self::JnzTakenInteractionGen,
    );

    fn write_jnz_taken_interaction(
        gen: Self::JnzTakenInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_assert_eq_trace(
        gen: assert_eq_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqClaim,
        Self::AssertEqInteractionGen,
    );

    fn write_assert_eq_interaction(
        gen: Self::AssertEqInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_assert_eq_imm_trace(
        gen: assert_eq_opcode_imm::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqImmClaim,
        Self::AssertEqImmInteractionGen,
    );

    fn write_assert_eq_imm_interaction(
        gen: Self::AssertEqImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_assert_eq_ddref_trace(
        gen: assert_eq_opcode_double_deref::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqDDerefClaim,
        Self::AssertEqDDerefInteractionGen,
    );

    fn write_assert_eq_ddref_interaction(
        gen: Self::AssertEqDDerefInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);

    fn write_call_rel_imm_trace(
        gen: call_opcode_rel_imm::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        CallRelImmClaim,
        Self::CallRelImmInteractionGen,
    );

    fn write_call_rel_imm_interaction(
        gen: Self::CallRelImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField);
}

impl OpcodeWitness for SimdBackend {
    type MemTables = ();
    type RetInteractionGen = ret_opcode::InteractionClaimGenerator;
    type AddSmallInteractionGen = add_opcode_small::InteractionClaimGenerator;
    type AddOpcodeInteractionGen = add_opcode::InteractionClaimGenerator;
    type AddApInteractionGen = add_ap_opcode::InteractionClaimGenerator;
    type JnzTakenInteractionGen = jnz_opcode_taken::InteractionClaimGenerator;
    type AssertEqInteractionGen = assert_eq_opcode::InteractionClaimGenerator;
    type AssertEqImmInteractionGen = assert_eq_opcode_imm::InteractionClaimGenerator;
    type AssertEqDDerefInteractionGen = assert_eq_opcode_double_deref::InteractionClaimGenerator;
    type CallRelImmInteractionGen = call_opcode_rel_imm::InteractionClaimGenerator;

    fn build_mem_tables(
        _memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        _memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables {
    }

    fn write_add_opcode_trace(
        gen: add_opcode::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddClaim, Self::AddOpcodeInteractionGen) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_add_opcode_interaction(
        gen: Self::AddOpcodeInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_add_opcode_small_trace(
        gen: add_opcode_small::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AddSmallClaim,
        Self::AddSmallInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_add_opcode_small_interaction(
        gen: Self::AddSmallInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_add_ap_trace(
        gen: add_ap_opcode::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
        range_check_18: &range_check_18::ClaimGenerator,
        range_check_11: &range_check_11::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddApClaim, Self::AddApInteractionGen) {
        let (trace, claim, interaction_gen) = gen.write_trace(
            memory_address_to_id,
            memory_id_to_big,
            verify_instruction,
            range_check_18,
            range_check_11,
        );
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_add_ap_interaction(
        gen: Self::AddApInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_ret_trace(
        gen: ret_opcode::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, RetClaim, Self::RetInteractionGen) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_ret_interaction(
        gen: Self::RetInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_jnz_taken_trace(
        gen: jnz_opcode_taken::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        JnzTakenClaim,
        Self::JnzTakenInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_jnz_taken_interaction(
        gen: Self::JnzTakenInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_assert_eq_trace(
        gen: assert_eq_opcode::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqClaim,
        Self::AssertEqInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_assert_eq_interaction(
        gen: Self::AssertEqInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_assert_eq_imm_trace(
        gen: assert_eq_opcode_imm::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqImmClaim,
        Self::AssertEqImmInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_assert_eq_imm_interaction(
        gen: Self::AssertEqImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_assert_eq_ddref_trace(
        gen: assert_eq_opcode_double_deref::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqDDerefClaim,
        Self::AssertEqDDerefInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_assert_eq_ddref_interaction(
        gen: Self::AssertEqDDerefInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }

    fn write_call_rel_imm_trace(
        gen: call_opcode_rel_imm::ClaimGenerator,
        _mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        CallRelImmClaim,
        Self::CallRelImmInteractionGen,
    ) {
        let (trace, claim, interaction_gen) =
            gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
        (trace.to_evals(), claim, interaction_gen)
    }

    fn write_call_rel_imm_interaction(
        gen: Self::CallRelImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let (raw, _build_claim) = gen.write_interaction_trace(common_lookup_elements);
        raw.finalize_on_simd()
    }
}

/// Device-born ret_opcode state: the 16 trace columns plus the 4 staged
/// columns its interaction tuples reference (fp-1, fp-2, next_pc, next_fp).
pub struct DeviceRetWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 4],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<ret_opcode::InteractionClaimGenerator>,
}

pub enum CudaRetInteractionGen {
    Device(Box<DeviceRetWitness>),
    Host(Box<ret_opcode::InteractionClaimGenerator>),
}

/// Device-born add_opcode_small state: the 39 trace columns plus the 19 staged
/// columns its interaction tuples reference.
pub struct DeviceAddSmallWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 19],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<add_opcode_small::InteractionClaimGenerator>,
}

pub enum CudaAddSmallInteractionGen {
    Device(Box<DeviceAddSmallWitness>),
    Host(Box<add_opcode_small::InteractionClaimGenerator>),
}

/// Device-born add_opcode state: the 103 trace columns plus the 7 staged
/// columns its interaction tuples reference (the two verify_instruction felts,
/// the three memory_address_to_id read addresses, and the opcodes-out next_pc /
/// next_ap).
pub struct DeviceAddWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 7],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<add_opcode::InteractionClaimGenerator>,
}

pub enum CudaAddInteractionGen {
    Device(Box<DeviceAddWitness>),
    Host(Box<add_opcode::InteractionClaimGenerator>),
}

/// Device-born add_ap_opcode state: the 17 trace columns plus the 9 staged
/// columns its interaction tuples reference (the verify_instruction expr, the
/// op1 read address, the four memory_id_to_big_2 small-sign slots, the
/// range_check_18 tuple value, and the opcodes-out next_pc / next_ap).
pub struct DeviceAddApWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 9],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<add_ap_opcode::InteractionClaimGenerator>,
}

pub enum CudaAddApInteractionGen {
    Device(Box<DeviceAddApWitness>),
    Host(Box<add_ap_opcode::InteractionClaimGenerator>),
}

/// Device-born jnz_opcode_taken state: the 47 trace columns plus the 10 staged
/// columns its interaction tuples reference (vi off1/off2, dst read addr, pc+1,
/// the four memory_id_to_big_4 slots, and the opcodes-out next_pc / next_ap).
pub struct DeviceJnzTakenWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 10],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<jnz_opcode_taken::InteractionClaimGenerator>,
}

pub enum CudaJnzTakenInteractionGen {
    Device(Box<DeviceJnzTakenWitness>),
    Host(Box<jnz_opcode_taken::InteractionClaimGenerator>),
}

/// Device-born assert_eq_opcode state: the 12 trace columns plus the 6 staged
/// columns its interaction tuples reference (vi_felt5/vi_felt6, the two read
/// addresses dst_addr / op1_addr, and the opcodes-out next_pc / next_ap).
pub struct DeviceAssertEqWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 6],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<assert_eq_opcode::InteractionClaimGenerator>,
}

pub enum CudaAssertEqInteractionGen {
    Device(Box<DeviceAssertEqWitness>),
    Host(Box<assert_eq_opcode::InteractionClaimGenerator>),
}

/// Device-born assert_eq_opcode_imm state: the 9 trace columns plus the 6 staged
/// columns its interaction tuples reference (vi_felt5/vi_felt6, the two read
/// addresses dst_addr / imm_addr [pc+1], and the opcodes-out next_pc [pc+2] /
/// next_ap).
pub struct DeviceAssertEqImmWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 6],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<assert_eq_opcode_imm::InteractionClaimGenerator>,
}

pub enum CudaAssertEqImmInteractionGen {
    Device(Box<DeviceAssertEqImmWitness>),
    Host(Box<assert_eq_opcode_imm::InteractionClaimGenerator>),
}

/// Device-born assert_eq_opcode_double_deref state: the 19 trace columns plus
/// the 7 staged columns its interaction tuples reference (vi_felt5/vi_felt6, the
/// three read addresses mem1_base_addr [mem_address_to_id_1, the pointer read] /
/// dst_addr [mem_address_to_id_3] / ddref_addr [mem_address_to_id_4, the second
/// deref], and the opcodes-out next_pc [pc+1] / next_ap).
pub struct DeviceAssertEqDDerefWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 7],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<assert_eq_opcode_double_deref::InteractionClaimGenerator>,
}

pub enum CudaAssertEqDDerefInteractionGen {
    Device(Box<DeviceAssertEqDDerefWitness>),
    Host(Box<assert_eq_opcode_double_deref::InteractionClaimGenerator>),
}

/// Device-born call_opcode_rel_imm state: the 24 trace columns plus the 8 staged
/// columns its interaction tuples reference (ret_pc read addr ap+1, next_pc read
/// addr pc+1, the four memory_id_to_big_6 sign slots, and the opcodes-out
/// next_pc / next_ap yield exprs).
pub struct DeviceCallRelImmWitness {
    cols: Vec<BaseFieldVec>,
    staged: [BaseFieldVec; 8],
    column_length: usize,
    n_rows: usize,
    verify_host: Option<call_opcode_rel_imm::InteractionClaimGenerator>,
}

pub enum CudaCallRelImmInteractionGen {
    Device(Box<DeviceCallRelImmWitness>),
    Host(Box<call_opcode_rel_imm::InteractionClaimGenerator>),
}

fn ret_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_RET_WITNESS").as_deref() != Ok("0")
}

fn add_small_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ADD_SMALL_WITNESS").as_deref() != Ok("0")
}

fn add_opcode_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ADD_OPCODE_WITNESS").as_deref() != Ok("0")
}

/// Default-on (`!= Ok("0")`), like the other ported opcodes: validated on a 3090
/// (sm_86) via `test_prove_verify_all_opcode_components_cuda` +
/// `STWO_CUDA_WITNESS_VERIFY=1` — add_ap (the first port feeding the range checks)
/// had trace + interaction columns + sums byte-identical to the host reference and
/// the Cairo e2e proof verified. `STWO_CUDA_ADD_AP_WITNESS=0` disables it.
fn add_ap_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ADD_AP_WITNESS").as_deref() != Ok("0")
}

fn jnz_taken_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_JNZ_TAKEN_WITNESS").as_deref() != Ok("0")
}

fn assert_eq_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ASSERT_EQ_WITNESS").as_deref() != Ok("0")
}

/// Default-on (`!= Ok("0")`), like the other ported opcodes: validated on a
/// 3090 (sm_86) via `test_prove_verify_all_opcode_components_cuda` with
/// `STWO_CUDA_WITNESS_VERIFY=1` — device trace + interaction columns + sums were
/// byte-identical to the host reference, and the Cairo e2e proof was byte-equal
/// to SIMD. `STWO_CUDA_ASSERT_EQ_IMM_WITNESS=0` disables it (regression valve).
fn assert_eq_imm_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ASSERT_EQ_IMM_WITNESS").as_deref() != Ok("0")
}

/// Default-on (`!= Ok("0")`), like the other ported opcodes: validated on H100
/// (sm_90) via `test_prove_verify_all_opcode_components_cuda` +
/// `STWO_CUDA_WITNESS_VERIFY=1` — device trace + interaction columns + sums were
/// byte-identical to the host reference, and the Cairo e2e proof verified. The
/// 2-whale (imm+ddref) warm prove measured +9.6% vs both off on SN_PIE_2.
/// `STWO_CUDA_ASSERT_EQ_DDEREF_WITNESS=0` disables it (regression valve).
fn assert_eq_ddref_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_ASSERT_EQ_DDEREF_WITNESS").as_deref() != Ok("0")
}

fn call_rel_imm_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_CALL_REL_IMM_WITNESS").as_deref() != Ok("0")
}

/// The prove-wide device memory tables are real (worth uploading the tens of MB)
/// iff ANY ported opcode's device path is enabled.
fn any_opcode_device_path_enabled() -> bool {
    add_opcode_device_path_enabled()
        || add_ap_device_path_enabled()
        || add_small_device_path_enabled()
        || assert_eq_device_path_enabled()
        || assert_eq_imm_device_path_enabled()
        || assert_eq_ddref_device_path_enabled()
        || call_rel_imm_device_path_enabled()
        || jnz_taken_device_path_enabled()
        || ret_device_path_enabled()
}

fn verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

impl OpcodeWitness for CudaBackend {
    type MemTables = device_witness::DeviceMemTables;
    type RetInteractionGen = CudaRetInteractionGen;
    type AddSmallInteractionGen = CudaAddSmallInteractionGen;
    type AddOpcodeInteractionGen = CudaAddInteractionGen;
    type AddApInteractionGen = CudaAddApInteractionGen;
    type JnzTakenInteractionGen = CudaJnzTakenInteractionGen;
    type AssertEqInteractionGen = CudaAssertEqInteractionGen;
    type AssertEqImmInteractionGen = CudaAssertEqImmInteractionGen;
    type AssertEqDDerefInteractionGen = CudaAssertEqDDerefInteractionGen;
    type CallRelImmInteractionGen = CudaCallRelImmInteractionGen;

    fn build_mem_tables(
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables {
        if !any_opcode_device_path_enabled() {
            // Kill-switched: every opcode write takes the host fallback, so
            // upload placeholder tables instead of the real ~tens-of-MB ones.
            return device_witness::DeviceMemTables::upload(vec![0], vec![0; 8], vec![0; 4]);
        }
        let addr_table = memory_address_to_id.raw_id_table().to_vec();
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let big_words: Vec<u32> = big_values.iter().flatten().copied().collect();
        let small_words: Vec<u32> = small_values
            .iter()
            .flat_map(|value| u128_to_4_limbs(*value))
            .collect();
        device_witness::DeviceMemTables::upload(addr_table, big_words, small_words)
    }

    fn write_add_opcode_trace(
        gen: add_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddClaim, Self::AddOpcodeInteractionGen) {
        if !add_opcode_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAddInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) =
            device_witness::add_opcode_trace([&pc, &ap, &fp], mem_tables, n_rows, column_length);

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. The
        // instruction decode and the three read addresses/ids are identical to
        // add_opcode_small (only the read width differs, not the fed values),
        // so we reuse `decode_add_small_row`.
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded =
                decode_add_small_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [decoded.offset0, decoded.offset1, decoded.offset2],
                    [decoded.vi_felt5, decoded.vi_felt6],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.dst_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.op0_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.op1_addr, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.dst_id, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.op0_id, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.op1_id, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = add_opcode::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_dst_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr_op0_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let addr_op1_feed = unpack(&host_feeds.memory_address_to_id[2]);
                let id_dst_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let id_op0_feed = unpack(&host_feeds.memory_id_to_big[1]);
                let id_op1_feed = unpack(&host_feeds.memory_id_to_big[2]);
                for (i, state) in padded.iter().enumerate() {
                    let decoded =
                        decode_add_small_row(state, memory_address_to_id, big_values, small_values);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_dst_feed[i], decoded.dst_addr),
                        (addr_op0_feed[i], decoded.op0_addr),
                        (addr_op1_feed[i], decoded.op1_addr),
                        (id_dst_feed[i], decoded.dst_id),
                        (id_op0_feed[i], decoded.op0_id),
                        (id_op1_feed[i], decoded.op1_id),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!("STWO_CUDA_WITNESS_VERIFY: add_opcode feed MISMATCH at row {i}");
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: add_opcode col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_opcode trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_opcode trace columns OK");
            add_opcode::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ADD_N_COLS);

        (
            trace_evals,
            AddClaim { log_size },
            CudaAddInteractionGen::Device(Box::new(DeviceAddWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_add_opcode_interaction(
        gen: Self::AddOpcodeInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAddInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAddInteractionGen::Device(witness) => witness,
        };
        let DeviceAddWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_felt5, [1]=vi_felt6, [2]=dst_addr, [3]=op0_addr,
        // [4]=op1_addr, [5]=next_pc, [6]=next_ap.
        // The three memory_id_to_big tuples are FULL 252-bit reads: id + all 28
        // limbs are trace columns (no staged sign slots, no interior zeros).
        let id_to_big_full = |id_col: usize, limb0: usize| {
            let mut v = vec![Col(&cols[id_col])];
            for c in &cols[limb0..limb0 + 28] {
                v.push(Col(c));
            }
            v
        };

        // Column order = the host writer's five col_gen blocks.
        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1[dst]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]),   // input_pc
                    Col(&cols[3]),   // offset0
                    Col(&cols[4]),   // offset1
                    Col(&cols[5]),   // offset2
                    Col(&staged[0]), // vi_felt5
                    Col(&staged[1]), // vi_felt6
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[2]), Col(&cols[14])], // dst_addr, dst_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_id_to_big_2[dst], memory_address_to_id_3[op0]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_full(14, 15),
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[3]), Col(&cols[43])], // op0_addr, op0_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. (memory_id_to_big_4[op0], memory_address_to_id_5[op1]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_full(43, 44),
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[4]), Col(&cols[72])], // op1_addr, op1_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 4. (memory_id_to_big_6[op1], opcodes-in) — (1, enabler).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_full(72, 73),
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 5. opcodes-out yield: -enabler / (next_pc, next_ap, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[5]), Col(&staged[6]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches =
                compare_interaction("add_opcode", &trace, claimed_sum, &host_trace, host_sum);
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_opcode interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_opcode interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_add_opcode_small_trace(
        gen: add_opcode_small::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AddSmallClaim,
        Self::AddSmallInteractionGen,
    ) {
        if !add_small_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAddSmallInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::add_opcode_small_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. We
        // replicate the writer's row math: decode the instruction at pc, then
        // the three small reads at the derived addresses.
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded =
                decode_add_small_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [decoded.offset0, decoded.offset1, decoded.offset2],
                    [decoded.vi_felt5, decoded.vi_felt6],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.dst_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.op0_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.op1_addr, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.dst_id, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.op0_id, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.op1_id, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = add_opcode_small::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_dst_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr_op0_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let addr_op1_feed = unpack(&host_feeds.memory_address_to_id[2]);
                let id_dst_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let id_op0_feed = unpack(&host_feeds.memory_id_to_big[1]);
                let id_op1_feed = unpack(&host_feeds.memory_id_to_big[2]);
                for (i, state) in padded.iter().enumerate() {
                    let decoded =
                        decode_add_small_row(state, memory_address_to_id, big_values, small_values);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_dst_feed[i], decoded.dst_addr),
                        (addr_op0_feed[i], decoded.op0_addr),
                        (addr_op1_feed[i], decoded.op1_addr),
                        (id_dst_feed[i], decoded.dst_id),
                        (id_op0_feed[i], decoded.op0_id),
                        (id_op1_feed[i], decoded.op1_id),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: add_opcode_small feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: add_opcode_small col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_opcode_small trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_opcode_small trace columns OK");
            add_opcode_small::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ADD_SMALL_N_COLS);

        (
            trace_evals,
            AddSmallClaim { log_size },
            CudaAddSmallInteractionGen::Device(Box::new(DeviceAddSmallWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_add_opcode_small_interaction(
        gen: Self::AddSmallInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAddSmallInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAddSmallInteractionGen::Device(witness) => witness,
        };
        let DeviceAddSmallWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // Staged layout (see the kernel): [0]=vi_felt5, [1]=vi_felt6,
        // [2]=next_pc, [3]=next_ap, then per read r in {dst, op0, op1} at base
        // 4 + 5*r: [+0]=addr, [+1]=s5, [+2]=s6 (=mid_limbs_set*511, repeated 17x),
        // [+3]=s23, [+4]=s29.
        // The id_to_big tuples' 5 interior zeros (idx 24..28) stay Const(0) to
        // keep slot positions aligned (the trailing s29 is at slot 28).
        let id_to_big_slots = |id, limb0, limb1, limb2, base: usize| {
            let s5 = base;
            let s6 = base + 1;
            let s23 = base + 2;
            let s29 = base + 3;
            // Host tuple: [rel_id, id, limb0..2, s5, s6 x17, s23, 0 x5, s29].
            let mut v = vec![
                Col(&cols[id]),
                Col(&cols[limb0]),
                Col(&cols[limb1]),
                Col(&cols[limb2]),
                Col(&staged[s5]),
            ];
            for _ in 0..17 {
                v.push(Col(&staged[s6]));
            }
            v.push(Col(&staged[s23]));
            for _ in 0..5 {
                v.push(Const(0));
            }
            v.push(Col(&staged[s29]));
            v
        };

        // Column order = the host writer's five col_gen blocks.
        let columns = vec![
            // 1. (verify_instruction, memory_address_to_id[dst]) — mults (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]),
                    Col(&cols[3]),
                    Col(&cols[4]),
                    Col(&cols[5]),
                    Col(&staged[0]),
                    Col(&staged[1]),
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[4]), Col(&cols[14])],
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_id_to_big[dst], memory_address_to_id[op0]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_slots(14, 17, 18, 19, 5),
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[9]), Col(&cols[22])],
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. (memory_id_to_big[op0], memory_address_to_id[op1]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_slots(22, 25, 26, 27, 10),
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[14]), Col(&cols[30])],
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 4. (memory_id_to_big[op1], opcodes-in) — (1, enabler).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_slots(30, 33, 34, 35, 15),
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])],
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 5. opcodes-out yield: -enabler / (next_pc, next_ap, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[2]), Col(&staged[3]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "add_opcode_small",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_opcode_small interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_opcode_small interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_add_ap_trace(
        gen: add_ap_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
        range_check_18: &range_check_18::ClaimGenerator,
        range_check_11: &range_check_11::ClaimGenerator,
    ) -> (MemoryEvals<Self>, AddApClaim, Self::AddApInteractionGen) {
        if !add_ap_device_path_enabled() {
            let (trace, claim, interaction_gen) = gen.write_trace(
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
                range_check_18,
                range_check_11,
            );
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAddApInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) =
            device_witness::add_ap_opcode_trace([&pc, &ap, &fp], mem_tables, n_rows, column_length);

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. add_ap
        // feeds verify_instruction once, memory_address_to_id once (the op1
        // read), memory_id_to_big once (the op1 id), range_check_18 once and
        // range_check_11 once. The verify_instruction offsets are
        // [32767, 32767, offset2] and the exprs are [vi_felt, 16].
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded = decode_add_ap_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [m31(32767), m31(32767), decoded.offset2],
                    [decoded.vi_felt, m31(16)],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.op1_addr, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.op1_id, 0);
            // The two range checks of next_ap's RangeCheck29 split.
            AddInputs::add_input(range_check_18, &[decoded.rc18_val], 0);
            AddInputs::add_input(range_check_11, &[decoded.rc29_bot11bits], 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = add_ap_opcode::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
                range_check_18,
                range_check_11,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element,
            // including the two range_check feeds (the new piece vs the assert_eq
            // / add siblings).
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_op1_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let id_op1_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let rc18_feed: Vec<M31> = host_feeds.range_check_18[0]
                    .iter()
                    .flat_map(|p| p[0].to_array())
                    .collect();
                let rc11_feed: Vec<M31> = host_feeds.range_check_11[0]
                    .iter()
                    .flat_map(|p| p[0].to_array())
                    .collect();
                for (i, state) in padded.iter().enumerate() {
                    let decoded =
                        decode_add_ap_row(state, memory_address_to_id, big_values, small_values);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_op1_feed[i], decoded.op1_addr),
                        (id_op1_feed[i], decoded.op1_id),
                        (rc18_feed[i], decoded.rc18_val),
                        (rc11_feed[i], decoded.rc29_bot11bits),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: add_ap_opcode feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: add_ap_opcode col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_ap_opcode trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_ap_opcode trace columns OK");
            add_ap_opcode::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ADD_AP_N_COLS);

        (
            trace_evals,
            AddApClaim { log_size },
            CudaAddApInteractionGen::Device(Box::new(DeviceAddApWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_add_ap_interaction(
        gen: Self::AddApInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAddApInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAddApInteractionGen::Device(witness) => witness,
        };
        let DeviceAddApWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_felt, [1]=op1_addr, [2]=m_s5, [3]=m_s6, [4]=m_s22,
        // [5]=m_s28, [6]=rc18_val, [7]=next_pc (pc+1+op1_imm), [8]=next_ap
        // (ap+read_small). Trace cols used in tuples: [0]=pc, [1]=ap, [2]=fp,
        // [3]=offset2, [7]=op1_id, [10..13)=op1_limb_0/1/2, [15]=rc29_bot11bits.
        //
        // Column order = the host writer's four col_gen blocks (logup in pairs):
        //   1. (verify_instruction_0, memory_address_to_id_1) — mults (1, 1)
        //   2. (memory_id_to_big_2, range_check_18_3)         — mults (1, 1)
        //   3. (range_check_11_4, opcodes_5[in])              — mults (1, enabler)
        //   4. opcodes_6[out] single yield                    — -enabler
        //
        // The memory_id_to_big_2 tuple is a 29-bit small read: id + 3 limbs, then
        // s5, s6 x17, s22, 5 interior zeros (kept as Const — the trailing s28
        // follows them, only trailing zeros may be truncated), s28. The two
        // range_check tuples are SINGLE-value lookups (one slot each); they sit
        // as the second / first tuple of a pair (range_check_11's value is the
        // trace column rc29_bot11bits, not a staged column).
        let id_to_big_2: Vec<device_witness::TupleSlot> = {
            let mut v = vec![
                Col(&cols[7]),   // op1_id
                Col(&cols[10]),  // op1_limb_0
                Col(&cols[11]),  // op1_limb_1
                Col(&cols[12]),  // op1_limb_2
                Col(&staged[2]), // m_s5 = remainder_bits + mid_limbs_set*508
            ];
            for _ in 0..17 {
                v.push(Col(&staged[3])); // m_s6 = mid_limbs_set*511
            }
            v.push(Col(&staged[4])); // m_s22 = msb*136 - mid_limbs_set
            for _ in 0..5 {
                v.push(Const(0));
            }
            v.push(Col(&staged[5])); // m_s28 = msb*256
            v
        };

        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1[op1]) — (1, 1).
            // verify_instruction offsets are [32767, 32767, offset2]; exprs are
            // [vi_felt, 16].
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]), // input_pc
                    Const(32767),
                    Const(32767),
                    Col(&cols[3]),   // offset2
                    Col(&staged[0]), // vi_felt
                    Const(16),
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[1]), Col(&cols[7])], // op1_addr, op1_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_id_to_big_2[op1], range_check_18_3) — mults (1, 1).
            // range_check_18 is a single-value tuple: [rel_id, rc18_val].
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &id_to_big_2,
                RANGE_CHECK_18_RELATION_ID,
                &[Col(&staged[6])], // (next_ap - rc29_bot11bits) * 2^20
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. (range_check_11_4, opcodes-in) — mults (1, enabler).
            // range_check_11 is a single-value tuple: [rel_id, rc29_bot11bits].
            device_witness::tuple_pair_logup_slots(
                RANGE_CHECK_11_RELATION_ID,
                &[Col(&cols[15])], // rc29_bot11bits (trace column, not staged)
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 4. opcodes-out yield: -enabler / (pc+1+op1_imm, next_ap, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[7]), Col(&staged[8]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches =
                compare_interaction("add_ap_opcode", &trace, claimed_sum, &host_trace, host_sum);
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: add_ap_opcode interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: add_ap_opcode interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_ret_trace(
        gen: ret_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (MemoryEvals<Self>, RetClaim, Self::RetInteractionGen) {
        if !ret_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaRetInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) =
            device_witness::ret_opcode_trace([&pc, &ap, &fp], mem_tables, n_rows, column_length);

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops.
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [m31(32766), m31(32767), m31(32767)],
                    [m31(88), m31(130)],
                    m31(0),
                ),
                0,
            );
            let addr0 = state.fp - m31(1);
            let addr1 = state.fp - m31(2);
            let id0 = memory_address_to_id.get_id(addr0);
            let id1 = memory_address_to_id.get_id(addr1);
            AddInputs::add_input(memory_address_to_id, &addr0, 0);
            AddInputs::add_input(memory_address_to_id, &addr1, 0);
            AddInputs::add_input(memory_id_to_big, &id0, 0);
            AddInputs::add_input(memory_id_to_big, &id1, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = ret_opcode::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| {
                        let pcs = p.0.to_array();
                        pcs.into_iter()
                    })
                    .collect();
                let addr0_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr1_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let id0_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let id1_feed = unpack(&host_feeds.memory_id_to_big[1]);
                for (i, state) in padded.iter().enumerate() {
                    let expected = [
                        (vi_feed[i], state.pc),
                        (addr0_feed[i], state.fp - M31::from(1)),
                        (addr1_feed[i], state.fp - M31::from(2)),
                        (
                            id0_feed[i],
                            memory_address_to_id.get_id(state.fp - M31::from(1)),
                        ),
                        (
                            id1_feed[i],
                            memory_address_to_id.get_id(state.fp - M31::from(2)),
                        ),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!("STWO_CUDA_WITNESS_VERIFY: ret_opcode feed MISMATCH at row {i}");
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: ret_opcode col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: ret_opcode trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: ret_opcode trace columns OK");
            ret_opcode::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), RET_N_COLS);

        (
            trace_evals,
            RetClaim { log_size },
            CudaRetInteractionGen::Device(Box::new(DeviceRetWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_ret_interaction(
        gen: Self::RetInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaRetInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaRetInteractionGen::Device(witness) => witness,
        };
        let DeviceRetWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // Column order = the host writer's. The id_to_big tuples' 24 trailing
        // zeros contribute exactly zero to the combine sum, so the tuples are
        // passed truncated — identical field value.
        let columns = vec![
            // (verify_instruction, memory_address_to_id[fp-1]) — mults (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]),
                    Const(32766),
                    Const(32767),
                    Const(32767),
                    Const(88),
                    Const(130),
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[0]), Col(&cols[3])],
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // (memory_id_to_big[next_pc], memory_address_to_id[fp-2]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &[
                    Col(&cols[3]),
                    Col(&cols[4]),
                    Col(&cols[5]),
                    Col(&cols[6]),
                    Col(&cols[7]),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[1]), Col(&cols[9])],
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // (memory_id_to_big[next_fp], opcodes-in) — (1, enabler).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &[
                    Col(&cols[9]),
                    Col(&cols[10]),
                    Col(&cols[11]),
                    Col(&cols[12]),
                    Col(&cols[13]),
                ],
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])],
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // opcodes-out yield: -enabler / (next_pc, ap, next_fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[2]), Col(&cols[1]), Col(&staged[3])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches =
                compare_interaction("ret_opcode", &trace, claimed_sum, &host_trace, host_sum);
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: ret_opcode interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: ret_opcode interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_jnz_taken_trace(
        gen: jnz_opcode_taken::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        JnzTakenClaim,
        Self::JnzTakenInteractionGen,
    ) {
        if !jnz_taken_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaJnzTakenInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::jnz_opcode_taken_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Per-row host decode of the instruction / dst / next_pc felts via the
        // scalar raw-table reader (value-identical to deduce_output, without
        // the 16-lane packed broadcast).
        let m31 = M31::from;
        let (big_values, small_values) = memory_id_to_big.value_tables();

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops.
        inputs.par_iter().for_each(|state| {
            // Decode Instruction at pc.
            let instr_id = memory_address_to_id.get_id(state.pc);
            let il = id_to_limbs(instr_id, big_values, small_values);
            let offset0 = m31(il[0].0 + ((il[1].0 & 127) << 9));
            let flags = (il[5].0 >> 3) + (il[6].0 << 6);
            let dst_base_fp = m31((flags >> 0) & 1);
            let ap_update_add_1 = m31((flags >> 11) & 1);
            let vi_off1 = ((dst_base_fp * m31(8)) + m31(16)) + m31(32);
            let vi_off2 = m31(8) + (ap_update_add_1 * m31(32));
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [offset0, m31(32767), m31(32769)],
                    [vi_off1, vi_off2],
                    m31(0),
                ),
                0,
            );

            // Read dst at mem_dst_base + (offset0 - 32768).
            let mem_dst_base = (dst_base_fp * state.fp) + ((m31(1) - dst_base_fp) * state.ap);
            let dst_addr = mem_dst_base + (offset0 - m31(32768));
            let dst_id = memory_address_to_id.get_id(dst_addr);
            AddInputs::add_input(memory_address_to_id, &dst_addr, 0);
            AddInputs::add_input(memory_id_to_big, &dst_id, 0);

            // Read small next_pc at pc + 1.
            let next_pc_addr = state.pc + m31(1);
            let next_pc_id = memory_address_to_id.get_id(next_pc_addr);
            AddInputs::add_input(memory_address_to_id, &next_pc_addr, 0);
            AddInputs::add_input(memory_id_to_big, &next_pc_id, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = jnz_opcode_taken::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let dst_addr_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let npc_addr_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let dst_id_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let npc_id_feed = unpack(&host_feeds.memory_id_to_big[1]);
                for (i, state) in padded.iter().enumerate() {
                    let instr_id = memory_address_to_id.get_id(state.pc);
                    let il = id_to_limbs(instr_id, big_values, small_values);
                    let offset0 = m31(il[0].0 + ((il[1].0 & 127) << 9));
                    let flags = (il[5].0 >> 3) + (il[6].0 << 6);
                    let dst_base_fp = m31((flags >> 0) & 1);
                    let mem_dst_base =
                        (dst_base_fp * state.fp) + ((m31(1) - dst_base_fp) * state.ap);
                    let dst_addr = mem_dst_base + (offset0 - m31(32768));
                    let next_pc_addr = state.pc + m31(1);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (dst_addr_feed[i], dst_addr),
                        (npc_addr_feed[i], next_pc_addr),
                        (dst_id_feed[i], memory_address_to_id.get_id(dst_addr)),
                        (npc_id_feed[i], memory_address_to_id.get_id(next_pc_addr)),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken trace columns OK");
            jnz_opcode_taken::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), JNZ_TAKEN_N_COLS);

        (
            trace_evals,
            JnzTakenClaim { log_size },
            CudaJnzTakenInteractionGen::Device(Box::new(DeviceJnzTakenWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_jnz_taken_interaction(
        gen: Self::JnzTakenInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaJnzTakenInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaJnzTakenInteractionGen::Device(witness) => witness,
        };
        let DeviceJnzTakenWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_off1, [1]=vi_off2, [2]=dst_addr, [3]=next_pc_addr,
        // [4]=m4_s4, [5]=m4_s5, [6]=m4_s22, [7]=m4_s28,
        // [8]=next_pc_out, [9]=next_ap_out.
        // Column order = the host writer's col_gen sequence.
        let columns = vec![
            // (verify_instruction_0, memory_address_to_id_1) — mults (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]), // input_pc
                    Col(&cols[3]), // offset0
                    Const(32767),
                    Const(32769),
                    Col(&staged[0]), // vi_off1
                    Col(&staged[1]), // vi_off2
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[2]), Col(&cols[7])], // dst_addr, dst_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // (memory_id_to_big_2[dst], memory_address_to_id_3[pc+1]) — (1, 1).
            // dst limbs are trace cols 8..35 (28 limbs).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &{
                    let mut s = vec![Col(&cols[7])]; // dst_id
                    for c in &cols[8..36] {
                        s.push(Col(c));
                    }
                    s
                },
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[3]), Col(&cols[38])], // pc+1, next_pc_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // (memory_id_to_big_4[next_pc small], opcodes-in) — (1, enabler).
            // Slots: next_pc_id, npc_limb0/1/2, m4_s4, m4_s5 (x17), m4_s22,
            // five zeros (truncated), m4_s28.
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &{
                    let mut s = vec![
                        Col(&cols[38]),  // next_pc_id
                        Col(&cols[41]),  // next_pc_limb_0
                        Col(&cols[42]),  // next_pc_limb_1
                        Col(&cols[43]),  // next_pc_limb_2
                        Col(&staged[4]), // remainder_bits + dss[2]
                    ];
                    for _ in 0..17 {
                        s.push(Col(&staged[5])); // dss[3]
                    }
                    s.push(Col(&staged[6])); // dss[4]
                                             // Five interior zeros — kept as Const because dss[5] follows
                                             // them (only trailing zeros may be truncated).
                    for _ in 0..5 {
                        s.push(Const(0));
                    }
                    s.push(Col(&staged[7])); // dss[5]
                    s
                },
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // opcodes-out yield: -enabler / (pc+read_small, ap+ap_update, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[8]), Col(&staged[9]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "jnz_opcode_taken",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: jnz_opcode_taken interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_assert_eq_trace(
        gen: assert_eq_opcode::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqClaim,
        Self::AssertEqInteractionGen,
    ) {
        if !assert_eq_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAssertEqInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::assert_eq_opcode_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. The
        // assert_eq writer feeds verify_instruction once and memory_address_to_id
        // twice (dst / op1 reads); it does NOT feed memory_id_to_big.
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded =
                decode_assert_eq_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [decoded.offset0, m31(32767), decoded.offset2],
                    [decoded.vi_felt5, decoded.vi_felt6],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.dst_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.op1_addr, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = assert_eq_opcode::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_dst_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr_op1_feed = unpack(&host_feeds.memory_address_to_id[1]);
                for (i, state) in padded.iter().enumerate() {
                    let decoded =
                        decode_assert_eq_row(state, memory_address_to_id, big_values, small_values);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_dst_feed[i], decoded.dst_addr),
                        (addr_op1_feed[i], decoded.op1_addr),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode trace columns OK");
            assert_eq_opcode::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ASSERT_EQ_N_COLS);

        (
            trace_evals,
            AssertEqClaim { log_size },
            CudaAssertEqInteractionGen::Device(Box::new(DeviceAssertEqWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_assert_eq_interaction(
        gen: Self::AssertEqInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAssertEqInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAssertEqInteractionGen::Device(witness) => witness,
        };
        let DeviceAssertEqWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_felt5, [1]=vi_felt6, [2]=dst_addr, [3]=op1_addr,
        // [4]=next_pc, [5]=next_ap. Column order = the host writer's col_gen
        // sequence (mults_0 = constant 1, mults_1 = enabler).
        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1) — mults (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]), // input_pc
                    Col(&cols[3]), // offset0
                    Const(32767),
                    Col(&cols[4]),   // offset2
                    Col(&staged[0]), // vi_felt5
                    Col(&staged[1]), // vi_felt6
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[2]), Col(&cols[10])], // dst_addr, dst_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_address_to_id_2, opcodes-in) — mults (1, enabler).
            // memory_address_to_id_2 reads the same dst_id at op1_addr.
            device_witness::tuple_pair_logup_slots(
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[3]), Col(&cols[10])], // op1_addr, dst_id
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. opcodes-out yield: -enabler / (pc+1, ap+ap_update, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[4]), Col(&staged[5]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "assert_eq_opcode",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode interaction differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode interaction columns + sums OK");
        }

        (trace, claimed_sum)
    }

    fn write_assert_eq_imm_trace(
        gen: assert_eq_opcode_imm::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqImmClaim,
        Self::AssertEqImmInteractionGen,
    ) {
        if !assert_eq_imm_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAssertEqImmInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::assert_eq_opcode_imm_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. The
        // assert_eq_imm writer feeds verify_instruction once and
        // memory_address_to_id twice (dst read / immediate read at pc+1); it does
        // NOT feed memory_id_to_big. The verify_instruction offsets are
        // [offset0, 32767, 32769] (offset1/offset2 are fixed constants for imm).
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded =
                decode_assert_eq_imm_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [decoded.offset0, m31(32767), m31(32769)],
                    [decoded.vi_felt5, decoded.vi_felt6],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.dst_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.imm_addr, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = assert_eq_opcode_imm::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_dst_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr_imm_feed = unpack(&host_feeds.memory_address_to_id[1]);
                for (i, state) in padded.iter().enumerate() {
                    let decoded = decode_assert_eq_imm_row(
                        state,
                        memory_address_to_id,
                        big_values,
                        small_values,
                    );
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_dst_feed[i], decoded.dst_addr),
                        (addr_imm_feed[i], decoded.imm_addr),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm trace columns OK");
            assert_eq_opcode_imm::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ASSERT_EQ_IMM_N_COLS);

        (
            trace_evals,
            AssertEqImmClaim { log_size },
            CudaAssertEqImmInteractionGen::Device(Box::new(DeviceAssertEqImmWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_assert_eq_imm_interaction(
        gen: Self::AssertEqImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAssertEqImmInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAssertEqImmInteractionGen::Device(witness) => witness,
        };
        let DeviceAssertEqImmWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_felt5, [1]=vi_felt6, [2]=dst_addr, [3]=imm_addr (pc+1),
        // [4]=next_pc (pc+2), [5]=next_ap. Column order = the host writer's
        // col_gen sequence (mults_0 = constant 1, mults_1 = enabler).
        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1) — mults (1, 1).
            // The verify_instruction tuple uses 32767 / 32769 for offset1 /
            // offset2 (fixed for the immediate operand).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]), // input_pc
                    Col(&cols[3]), // offset0
                    Const(32767),
                    Const(32769),
                    Col(&staged[0]), // vi_felt5
                    Col(&staged[1]), // vi_felt6
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[2]), Col(&cols[7])], // dst_addr, dst_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_address_to_id_2, opcodes-in) — mults (1, enabler).
            // memory_address_to_id_2 reads the same dst_id at imm_addr (pc+1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[3]), Col(&cols[7])], // imm_addr, dst_id
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. opcodes-out yield: -enabler / (pc+2, ap+ap_update, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[4]), Col(&staged[5]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "assert_eq_opcode_imm",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm interaction differential failed"
            );
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_imm interaction columns + sums OK"
            );
        }

        (trace, claimed_sum)
    }

    fn write_assert_eq_ddref_trace(
        gen: assert_eq_opcode_double_deref::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        AssertEqDDerefClaim,
        Self::AssertEqDDerefInteractionGen,
    ) {
        if !assert_eq_ddref_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaAssertEqDDerefInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::assert_eq_opcode_double_deref_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. The
        // double_deref writer feeds verify_instruction once, memory_address_to_id
        // three times (the pointer read mem1_base / dst / the doubly-deref'd
        // read) and memory_id_to_big once (the pointer id read as a 29-bit
        // value). The verify_instruction offsets are [offset0, offset1, offset2].
        let (big_values, small_values) = memory_id_to_big.value_tables();
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            let decoded =
                decode_assert_eq_ddref_row(state, memory_address_to_id, big_values, small_values);
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [decoded.offset0, decoded.offset1, decoded.offset2],
                    [decoded.vi_felt5, decoded.vi_felt6],
                    m31(0),
                ),
                0,
            );
            AddInputs::add_input(memory_address_to_id, &decoded.mem1_base_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.dst_addr, 0);
            AddInputs::add_input(memory_address_to_id, &decoded.ddref_addr, 0);
            AddInputs::add_input(memory_id_to_big, &decoded.mem1_base_id, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) =
                assert_eq_opcode_double_deref::write_trace_simd(
                    packed_inputs,
                    n_rows,
                    memory_address_to_id,
                    memory_id_to_big,
                    verify_instruction,
                );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let addr_mem1_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let addr_dst_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let addr_ddref_feed = unpack(&host_feeds.memory_address_to_id[2]);
                let id_mem1_feed = unpack(&host_feeds.memory_id_to_big[0]);
                for (i, state) in padded.iter().enumerate() {
                    let decoded = decode_assert_eq_ddref_row(
                        state,
                        memory_address_to_id,
                        big_values,
                        small_values,
                    );
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (addr_mem1_feed[i], decoded.mem1_base_addr),
                        (addr_dst_feed[i], decoded.dst_addr),
                        (addr_ddref_feed[i], decoded.ddref_addr),
                        (id_mem1_feed[i], decoded.mem1_base_id),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref feed \
                             MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref col {col_idx} \
                         MISMATCH (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref trace columns OK");
            assert_eq_opcode_double_deref::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), ASSERT_EQ_DDREF_N_COLS);

        (
            trace_evals,
            AssertEqDDerefClaim { log_size },
            CudaAssertEqDDerefInteractionGen::Device(Box::new(DeviceAssertEqDDerefWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_assert_eq_ddref_interaction(
        gen: Self::AssertEqDDerefInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaAssertEqDDerefInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaAssertEqDDerefInteractionGen::Device(witness) => witness,
        };
        let DeviceAssertEqDDerefWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=vi_felt5, [1]=vi_felt6, [2]=mem1_base_addr (mem_addr_1),
        // [3]=dst_addr (mem_addr_3), [4]=ddref_addr (mem_addr_4), [5]=next_pc
        // (pc+1), [6]=next_ap. Column order = the host writer's four col_gen
        // blocks (mults_0 = constant 1, mults_1 = enabler).
        //
        // Trace cols used directly in tuples: [11]=mem1_base_id, [12..16)=the 4
        // pointer limbs, [17]=dst_id.
        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1[mem1_base]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]),   // input_pc
                    Col(&cols[3]),   // offset0
                    Col(&cols[4]),   // offset1
                    Col(&cols[5]),   // offset2
                    Col(&staged[0]), // vi_felt5
                    Col(&staged[1]), // vi_felt6
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[2]), Col(&cols[11])], // mem1_base_addr, mem1_base_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_id_to_big_2[pointer], memory_address_to_id_3[dst]) — (1, 1).
            // The pointer is a 29-bit read: id + 4 limbs, then 24 trailing zeros
            // (truncated — they contribute exactly zero to the combine sum).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &[
                    Col(&cols[11]), // mem1_base_id
                    Col(&cols[12]), // mem1_base_limb_0
                    Col(&cols[13]), // mem1_base_limb_1
                    Col(&cols[14]), // mem1_base_limb_2
                    Col(&cols[15]), // mem1_base_limb_3
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[3]), Col(&cols[17])], // dst_addr, dst_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. (memory_address_to_id_4[ddref], opcodes-in) — (1, enabler).
            // The second deref reads the SAME id (dst_id) at the pointer address.
            device_witness::tuple_pair_logup_slots(
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[4]), Col(&cols[17])], // ddref_addr, dst_id
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 4. opcodes-out yield: -enabler / (pc+1, ap+ap_update, fp).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[5]), Col(&staged[6]), Col(&cols[2])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "assert_eq_opcode_double_deref",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref interaction \
                 differential failed"
            );
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY: assert_eq_opcode_double_deref interaction \
                 columns + sums OK"
            );
        }

        (trace, claimed_sum)
    }

    fn write_call_rel_imm_trace(
        gen: call_opcode_rel_imm::ClaimGenerator,
        mem_tables: &Self::MemTables,
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
        verify_instruction: &verify_instruction::ClaimGenerator,
    ) -> (
        MemoryEvals<Self>,
        CallRelImmClaim,
        Self::CallRelImmInteractionGen,
    ) {
        if !call_rel_imm_device_path_enabled() {
            let (trace, claim, interaction_gen) =
                gen.write_trace(memory_address_to_id, memory_id_to_big, verify_instruction);
            return (
                Self::from_simd_evals(trace.to_evals()),
                claim,
                CudaCallRelImmInteractionGen::Host(Box::new(interaction_gen)),
            );
        }

        let verify = verify_enabled();
        let verify_inputs = verify.then(|| gen.inputs.clone());
        let (inputs, n_rows) = gen.into_parts();
        let column_length = inputs.len();
        let log_size = column_length.ilog2();

        // SoA upload of the padded CasmState inputs.
        let pc = BaseFieldVec::from_vec(inputs.iter().map(|s| s.pc).collect());
        let ap = BaseFieldVec::from_vec(inputs.iter().map(|s| s.ap).collect());
        let fp = BaseFieldVec::from_vec(inputs.iter().map(|s| s.fp).collect());

        let (cols, staged) = device_witness::call_opcode_rel_imm_trace(
            [&pc, &ap, &fp],
            mem_tables,
            n_rows,
            column_length,
        );

        // Sub-component feeds, host-side over the padded inputs — identical
        // values and counts to the writer's sub_component_inputs loops. The call
        // instruction shape is fixed: the verify_instruction tuple is all
        // constants (except pc), and the three reads are at ap, ap+1, pc+1.
        let m31 = M31::from;
        inputs.par_iter().for_each(|state| {
            AddInputs::add_input(
                verify_instruction,
                &(
                    state.pc,
                    [m31(32768), m31(32769), m31(32769)],
                    [m31(32), m31(68)],
                    m31(0),
                ),
                0,
            );
            let fp_addr = state.ap;
            let ret_pc_addr = state.ap + m31(1);
            let next_pc_addr = state.pc + m31(1);
            let fp_id = memory_address_to_id.get_id(fp_addr);
            let ret_pc_id = memory_address_to_id.get_id(ret_pc_addr);
            let dist_id = memory_address_to_id.get_id(next_pc_addr);
            AddInputs::add_input(memory_address_to_id, &fp_addr, 0);
            AddInputs::add_input(memory_address_to_id, &ret_pc_addr, 0);
            AddInputs::add_input(memory_address_to_id, &next_pc_addr, 0);
            AddInputs::add_input(memory_id_to_big, &fp_id, 0);
            AddInputs::add_input(memory_id_to_big, &ret_pc_id, 0);
            AddInputs::add_input(memory_id_to_big, &dist_id, 0);
        });

        let verify_host = verify_inputs.map(|raw_inputs| {
            let mut padded = raw_inputs;
            padded.resize(column_length, *padded.first().unwrap());
            let packed_inputs = pack_values(&padded);
            let (host_trace, host_lookup_data, host_feeds) = call_opcode_rel_imm::write_trace_simd(
                packed_inputs,
                n_rows,
                memory_address_to_id,
                memory_id_to_big,
                verify_instruction,
            );
            let host_evals = host_trace.to_evals();
            let mut mismatches = 0usize;
            // Feed differential: the values this path fed host-side must match
            // the writer's sub_component_inputs buffers element-for-element.
            {
                let unpack = |cols: &[PackedM31]| -> Vec<M31> {
                    cols.iter().flat_map(|p| p.to_array()).collect()
                };
                let vi_pc_feed: Vec<_> = host_feeds.verify_instruction[0]
                    .iter()
                    .flat_map(|p| p.0.to_array())
                    .collect();
                let fp_addr_feed = unpack(&host_feeds.memory_address_to_id[0]);
                let ret_addr_feed = unpack(&host_feeds.memory_address_to_id[1]);
                let npc_addr_feed = unpack(&host_feeds.memory_address_to_id[2]);
                let fp_id_feed = unpack(&host_feeds.memory_id_to_big[0]);
                let ret_id_feed = unpack(&host_feeds.memory_id_to_big[1]);
                let dist_id_feed = unpack(&host_feeds.memory_id_to_big[2]);
                for (i, state) in padded.iter().enumerate() {
                    let fp_addr = state.ap;
                    let ret_pc_addr = state.ap + M31::from(1);
                    let next_pc_addr = state.pc + M31::from(1);
                    let expected = [
                        (vi_pc_feed[i], state.pc),
                        (fp_addr_feed[i], fp_addr),
                        (ret_addr_feed[i], ret_pc_addr),
                        (npc_addr_feed[i], next_pc_addr),
                        (fp_id_feed[i], memory_address_to_id.get_id(fp_addr)),
                        (ret_id_feed[i], memory_address_to_id.get_id(ret_pc_addr)),
                        (dist_id_feed[i], memory_address_to_id.get_id(next_pc_addr)),
                    ];
                    if expected.iter().any(|(host, device)| host != device) {
                        eprintln!(
                            "STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm feed MISMATCH at row {i}"
                        );
                        mismatches += 1;
                        break;
                    }
                }
            }
            for (col_idx, (device_col, host_col)) in cols.iter().zip(&host_evals).enumerate() {
                let device_values = device_col.to_vec();
                let host_values = host_col.values.to_cpu();
                if device_values != host_values {
                    let first_diff = device_values
                        .iter()
                        .zip(&host_values)
                        .position(|(d, h)| d != h);
                    eprintln!(
                        "STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm col {col_idx} MISMATCH \
                         (first diff at row {first_diff:?})"
                    );
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm trace differential failed"
            );
            eprintln!("STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm trace columns OK");
            call_opcode_rel_imm::InteractionClaimGenerator {
                log_size,
                lookup_data: host_lookup_data,
            }
        });

        let domain = CanonicCoset::new(log_size).circle_domain();
        let trace_evals: MemoryEvals<CudaBackend> = cols
            .iter()
            .map(|col| CircleEvaluation::new(domain, col.clone()))
            .collect();
        assert_eq!(trace_evals.len(), CALL_REL_IMM_N_COLS);

        (
            trace_evals,
            CallRelImmClaim { log_size },
            CudaCallRelImmInteractionGen::Device(Box::new(DeviceCallRelImmWitness {
                cols,
                staged,
                column_length,
                n_rows,
                verify_host,
            })),
        )
    }

    fn write_call_rel_imm_interaction(
        gen: Self::CallRelImmInteractionGen,
        common_lookup_elements: &CommonLookupElements,
    ) -> (MemoryEvals<Self>, SecureField) {
        let witness = match gen {
            CudaCallRelImmInteractionGen::Host(host_gen) => {
                let (raw, _build_claim) =
                    (*host_gen).write_interaction_trace(common_lookup_elements);
                return <CudaBackend as LogupFinalizeBackend>::finalize_raw_logup(raw);
            }
            CudaCallRelImmInteractionGen::Device(witness) => witness,
        };
        let DeviceCallRelImmWitness {
            cols,
            staged,
            column_length,
            n_rows,
            verify_host,
        } = *witness;

        let z = common_lookup_elements.z();
        let alphas = common_lookup_elements.alpha_powers();
        use device_witness::TupleSlot::{Col, Const};
        let enabler = || device_witness::Mult::Enabler(n_rows as u32);

        // staged: [0]=ret_pc_addr (ap+1), [1]=next_pc_addr (pc+1),
        // [2]=m6_s5, [3]=m6_s6, [4]=m6_s22, [5]=m6_s28,
        // [6]=next_pc_out (pc+read_small), [7]=next_ap_out (ap+2).
        // The id_to_big_2 / id_to_big_4 tuples' 24 trailing zeros contribute
        // exactly zero to the combine sum, so they are passed truncated. The
        // id_to_big_6 tuple keeps its 5 interior zeros as Const(0) (the trailing
        // m6_s28 follows them). Column order = the host writer's col_gen blocks.
        let columns = vec![
            // 1. (verify_instruction_0, memory_address_to_id_1) — mults (1, 1).
            device_witness::tuple_pair_logup_slots(
                VERIFY_INSTRUCTION_RELATION_ID,
                &[
                    Col(&cols[0]), // input_pc
                    Const(32768),
                    Const(32769),
                    Const(32769),
                    Const(32),
                    Const(68),
                    Const(0),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&cols[1]), Col(&cols[3])], // ap, stored_fp_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 2. (memory_id_to_big_2[stored_fp], memory_address_to_id_3[ap+1]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &[
                    Col(&cols[3]), // stored_fp_id
                    Col(&cols[4]),
                    Col(&cols[5]),
                    Col(&cols[6]),
                    Col(&cols[7]),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[0]), Col(&cols[9])], // ret_pc_addr, stored_ret_pc_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 3. (memory_id_to_big_4[stored_ret_pc], memory_address_to_id_5[pc+1]) — (1, 1).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &[
                    Col(&cols[9]), // stored_ret_pc_id
                    Col(&cols[10]),
                    Col(&cols[11]),
                    Col(&cols[12]),
                    Col(&cols[13]),
                ],
                MEMORY_ADDRESS_TO_ID_RELATION_ID.0,
                &[Col(&staged[1]), Col(&cols[15])], // next_pc_addr, dist_id
                device_witness::Mult::One,
                device_witness::Mult::One,
                false,
                column_length,
                alphas,
                z,
            ),
            // 4. (memory_id_to_big_6[dist small], opcodes-in) — (1, enabler).
            device_witness::tuple_pair_logup_slots(
                MEMORY_ID_TO_BIG_RELATION_ID.0,
                &{
                    let mut s = vec![
                        Col(&cols[15]),  // dist_id
                        Col(&cols[18]),  // dist_limb_0
                        Col(&cols[19]),  // dist_limb_1
                        Col(&cols[20]),  // dist_limb_2
                        Col(&staged[2]), // m6_s5 = remainder_bits + mid_limbs_set*508
                    ];
                    for _ in 0..17 {
                        s.push(Col(&staged[3])); // m6_s6 = mid_limbs_set*511
                    }
                    s.push(Col(&staged[4])); // m6_s22 = msb*136 - mid_limbs_set
                                             // Five interior zeros — kept as Const because m6_s28 follows
                                             // them (only trailing zeros may be truncated).
                    for _ in 0..5 {
                        s.push(Const(0));
                    }
                    s.push(Col(&staged[5])); // m6_s28 = msb*256
                    s
                },
                OPCODES_RELATION_ID,
                &[Col(&cols[0]), Col(&cols[1]), Col(&cols[2])], // pc, ap, fp
                device_witness::Mult::One,
                enabler(),
                false,
                column_length,
                alphas,
                z,
            ),
            // 5. opcodes-out yield: -enabler / (pc+read_small, ap+2, ap+2).
            device_witness::tuple_single_logup_slots(
                OPCODES_RELATION_ID,
                &[Col(&staged[6]), Col(&staged[7]), Col(&staged[7])],
                enabler(),
                true,
                column_length,
                alphas,
                z,
            ),
        ];
        let (trace, claimed_sum) =
            device_witness::finalize_device_raw_logup(column_length.ilog2(), columns);

        if let Some(host_gen) = verify_host {
            let (host_raw, _build_claim) = host_gen.write_interaction_trace(common_lookup_elements);
            let (host_trace, host_sum) = host_raw.finalize_on_simd();
            let mismatches = compare_interaction(
                "call_opcode_rel_imm",
                &trace,
                claimed_sum,
                &host_trace,
                host_sum,
            );
            assert_eq!(
                mismatches, 0,
                "STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm interaction differential failed"
            );
            eprintln!(
                "STWO_CUDA_WITNESS_VERIFY: call_opcode_rel_imm interaction columns + sums OK"
            );
        }

        (trace, claimed_sum)
    }
}
