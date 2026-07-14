//! Exhaustive compile-time whitelist for AIR inputs behind direct BASE binding.
//!
//! Every current component claim is empty, contains only `log_size`, or is the
//! memory-big `big_log_sizes` vector. Those values are already part of the full
//! `TopologyKey`. Public statement values are exhaustively named below; only
//! the eight builtin starts classified by `bindings` may vary on a warm hit.
//! Adding any field to these schemas makes this module fail to compile until
//! the topology key and binding recipe are deliberately reviewed together.

use cairo_air::air::{
    MemorySmallValue, PublicData, PublicMemory, PublicSegmentRanges, SegmentRange,
};
use cairo_air::claims::CairoClaim;
use cairo_air::components;
use stwo::core::fields::m31::BaseField;
use stwo_cairo_common::prover_types::cpu::CasmState;

macro_rules! log_size_claims {
    ($( $field:ident: $module:ident ),+ $(,)?) => {
        $(
            if let Some(components::$module::Claim { log_size: _ }) = $field.as_ref() {}
        )+
    };
}

macro_rules! empty_claims {
    ($( $field:ident: $module:ident ),+ $(,)?) => {
        $(
            if let Some(components::$module::Claim {}) = $field.as_ref() {}
        )+
    };
}

// Keep the evaluator ABI closed as well as the claim/public-data schemas. A
// new evaluator field is a new possible BASE-parameter source, so these
// exhaustive patterns must fail compilation until the binding recipe changes.
macro_rules! invariant_eval_schemas {
    ($( $module:ident ),+ $(,)?) => {
        $(
            const _: fn(components::$module::Eval) =
                |components::$module::Eval { claim: _, common_lookup_elements: _ }| {};
        )+
    };
}

macro_rules! segment_eval_schemas {
    ($( $module:ident => $segment_start:ident ),+ $(,)?) => {
        $(
            const _: fn(components::$module::Eval) = |components::$module::Eval {
                claim: _,
                common_lookup_elements: _,
                $segment_start: _,
            }| {};
        )+
    };
}

invariant_eval_schemas!(
    add_opcode,
    add_opcode_small,
    add_ap_opcode,
    assert_eq_opcode,
    assert_eq_opcode_imm,
    assert_eq_opcode_double_deref,
    blake_compress_opcode,
    call_opcode_abs,
    call_opcode_rel_imm,
    generic_opcode,
    jnz_opcode_non_taken,
    jnz_opcode_taken,
    jump_opcode_abs,
    jump_opcode_double_deref,
    jump_opcode_rel,
    jump_opcode_rel_imm,
    mul_opcode,
    mul_opcode_small,
    qm_31_add_mul_opcode,
    ret_opcode,
    verify_instruction,
    blake_round,
    blake_g,
    blake_round_sigma,
    triple_xor_32,
    verify_bitwise_xor_12,
    partial_ec_mul_generic,
    pedersen_aggregator_window_bits_18,
    partial_ec_mul_window_bits_18,
    pedersen_points_table_window_bits_18,
    pedersen_aggregator_window_bits_9,
    partial_ec_mul_window_bits_9,
    pedersen_points_table_window_bits_9,
    poseidon_aggregator,
    poseidon_3_partial_rounds_chain,
    poseidon_full_round_chain,
    cube_252,
    poseidon_round_keys,
    range_check_252_width_27,
    memory_address_to_id,
    memory_id_to_small,
    range_check_6,
    range_check_8,
    range_check_11,
    range_check_12,
    range_check_18,
    range_check_20,
    range_check_4_3,
    range_check_4_4,
    range_check_9_9,
    range_check_7_2_5,
    range_check_3_6_6_3,
    range_check_4_4_4_4,
    range_check_3_3_3_3_3,
    verify_bitwise_xor_4,
    verify_bitwise_xor_7,
    verify_bitwise_xor_8,
    verify_bitwise_xor_9,
);
segment_eval_schemas!(
    add_mod_builtin => add_mod_builtin_segment_start,
    bitwise_builtin => bitwise_builtin_segment_start,
    mul_mod_builtin => mul_mod_builtin_segment_start,
    pedersen_builtin => pedersen_builtin_segment_start,
    pedersen_builtin_narrow_windows => pedersen_builtin_segment_start,
    poseidon_builtin => poseidon_builtin_segment_start,
    range_check96_builtin => range_check96_builtin_segment_start,
    range_check_builtin => range_check_builtin_segment_start,
    ec_op_builtin => ec_op_builtin_segment_start,
);
const _: fn(components::memory_id_to_big::BigEval) =
    |components::memory_id_to_big::BigEval {
         log_n_rows: _,
         offset: _,
         common_lookup_elements: _,
     }| {};

pub(super) fn assert_binding_schema_is_whitelisted(claim: &CairoClaim) {
    let CairoClaim {
        public_data,
        add_opcode,
        add_opcode_small,
        add_ap_opcode,
        assert_eq_opcode,
        assert_eq_opcode_imm,
        assert_eq_opcode_double_deref,
        blake_compress_opcode,
        call_opcode_abs,
        call_opcode_rel_imm,
        generic_opcode,
        jnz_opcode_non_taken,
        jnz_opcode_taken,
        jump_opcode_abs,
        jump_opcode_double_deref,
        jump_opcode_rel,
        jump_opcode_rel_imm,
        mul_opcode,
        mul_opcode_small,
        qm_31_add_mul_opcode,
        ret_opcode,
        verify_instruction,
        blake_round,
        blake_g,
        blake_round_sigma,
        triple_xor_32,
        verify_bitwise_xor_12,
        add_mod_builtin,
        bitwise_builtin,
        mul_mod_builtin,
        pedersen_builtin,
        pedersen_builtin_narrow_windows,
        poseidon_builtin,
        range_check96_builtin,
        range_check_builtin,
        ec_op_builtin,
        partial_ec_mul_generic,
        pedersen_aggregator_window_bits_18,
        partial_ec_mul_window_bits_18,
        pedersen_points_table_window_bits_18,
        pedersen_aggregator_window_bits_9,
        partial_ec_mul_window_bits_9,
        pedersen_points_table_window_bits_9,
        poseidon_aggregator,
        poseidon_3_partial_rounds_chain,
        poseidon_full_round_chain,
        cube_252,
        poseidon_round_keys,
        range_check_252_width_27,
        memory_address_to_id,
        memory_id_to_big,
        memory_id_to_small,
        range_check_6,
        range_check_8,
        range_check_11,
        range_check_12,
        range_check_18,
        range_check_20,
        range_check_4_3,
        range_check_4_4,
        range_check_9_9,
        range_check_7_2_5,
        range_check_3_6_6_3,
        range_check_4_4_4_4,
        range_check_3_3_3_3_3,
        verify_bitwise_xor_4,
        verify_bitwise_xor_7,
        verify_bitwise_xor_8,
        verify_bitwise_xor_9,
    } = claim;

    log_size_claims!(
        add_opcode: add_opcode,
        add_opcode_small: add_opcode_small,
        add_ap_opcode: add_ap_opcode,
        assert_eq_opcode: assert_eq_opcode,
        assert_eq_opcode_imm: assert_eq_opcode_imm,
        assert_eq_opcode_double_deref: assert_eq_opcode_double_deref,
        blake_compress_opcode: blake_compress_opcode,
        call_opcode_abs: call_opcode_abs,
        call_opcode_rel_imm: call_opcode_rel_imm,
        generic_opcode: generic_opcode,
        jnz_opcode_non_taken: jnz_opcode_non_taken,
        jnz_opcode_taken: jnz_opcode_taken,
        jump_opcode_abs: jump_opcode_abs,
        jump_opcode_double_deref: jump_opcode_double_deref,
        jump_opcode_rel: jump_opcode_rel,
        jump_opcode_rel_imm: jump_opcode_rel_imm,
        mul_opcode: mul_opcode,
        mul_opcode_small: mul_opcode_small,
        qm_31_add_mul_opcode: qm_31_add_mul_opcode,
        ret_opcode: ret_opcode,
        verify_instruction: verify_instruction,
        blake_round: blake_round,
        blake_g: blake_g,
        triple_xor_32: triple_xor_32,
        add_mod_builtin: add_mod_builtin,
        bitwise_builtin: bitwise_builtin,
        mul_mod_builtin: mul_mod_builtin,
        pedersen_builtin: pedersen_builtin,
        pedersen_builtin_narrow_windows: pedersen_builtin_narrow_windows,
        poseidon_builtin: poseidon_builtin,
        range_check96_builtin: range_check96_builtin,
        range_check_builtin: range_check_builtin,
        ec_op_builtin: ec_op_builtin,
        partial_ec_mul_generic: partial_ec_mul_generic,
        pedersen_aggregator_window_bits_18: pedersen_aggregator_window_bits_18,
        partial_ec_mul_window_bits_18: partial_ec_mul_window_bits_18,
        pedersen_aggregator_window_bits_9: pedersen_aggregator_window_bits_9,
        partial_ec_mul_window_bits_9: partial_ec_mul_window_bits_9,
        poseidon_aggregator: poseidon_aggregator,
        poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain,
        poseidon_full_round_chain: poseidon_full_round_chain,
        cube_252: cube_252,
        range_check_252_width_27: range_check_252_width_27,
        memory_address_to_id: memory_address_to_id,
        memory_id_to_small: memory_id_to_small,
    );
    empty_claims!(
        blake_round_sigma: blake_round_sigma,
        verify_bitwise_xor_12: verify_bitwise_xor_12,
        pedersen_points_table_window_bits_18: pedersen_points_table_window_bits_18,
        pedersen_points_table_window_bits_9: pedersen_points_table_window_bits_9,
        poseidon_round_keys: poseidon_round_keys,
        range_check_6: range_check_6,
        range_check_8: range_check_8,
        range_check_11: range_check_11,
        range_check_12: range_check_12,
        range_check_18: range_check_18,
        range_check_20: range_check_20,
        range_check_4_3: range_check_4_3,
        range_check_4_4: range_check_4_4,
        range_check_9_9: range_check_9_9,
        range_check_7_2_5: range_check_7_2_5,
        range_check_3_6_6_3: range_check_3_6_6_3,
        range_check_4_4_4_4: range_check_4_4_4_4,
        range_check_3_3_3_3_3: range_check_3_3_3_3_3,
        verify_bitwise_xor_4: verify_bitwise_xor_4,
        verify_bitwise_xor_7: verify_bitwise_xor_7,
        verify_bitwise_xor_8: verify_bitwise_xor_8,
        verify_bitwise_xor_9: verify_bitwise_xor_9,
    );
    if let Some(components::memory_id_to_big::Claim { big_log_sizes: _ }) =
        memory_id_to_big.as_ref()
    {}
    assert_public_data_schema(public_data);
}

/// Same-shape claim with every existing public-data value changed except the
/// eight explicitly admitted builtin starts. Cold lowering must remain exactly
/// unchanged, proving no other public value feeds a BASE slot.
pub(super) fn public_data_negative_control(claim: &CairoClaim) -> CairoClaim {
    let mut control = claim.clone();
    let PublicData {
        public_memory,
        initial_state,
        final_state,
    } = &mut control.public_data;
    mutate_casm_state(initial_state);
    mutate_casm_state(final_state);
    let PublicMemory {
        program,
        public_segments,
        output,
        safe_call_ids,
    } = public_memory;
    for (id, limbs) in program.iter_mut().chain(output.iter_mut()) {
        perturb_u32(id);
        limbs.iter_mut().for_each(perturb_u32);
    }
    safe_call_ids.iter_mut().for_each(perturb_u32);

    let PublicSegmentRanges {
        output,
        pedersen,
        range_check_128,
        ecdsa,
        bitwise,
        ec_op,
        keccak,
        poseidon,
        range_check_96,
        add_mod,
        mul_mod,
    } = public_segments;
    mutate_segment(output, false);
    for segment in [ecdsa, keccak].into_iter().filter_map(Option::as_mut) {
        mutate_segment(segment, false);
    }
    for segment in [
        pedersen,
        range_check_128,
        bitwise,
        ec_op,
        poseidon,
        range_check_96,
        add_mod,
        mul_mod,
    ]
    .into_iter()
    .filter_map(Option::as_mut)
    {
        mutate_segment(segment, true);
    }
    control
}

fn assert_public_data_schema(public_data: &PublicData) {
    let PublicData {
        public_memory,
        initial_state,
        final_state,
    } = public_data;
    let CasmState {
        pc: _,
        ap: _,
        fp: _,
    } = initial_state;
    let CasmState {
        pc: _,
        ap: _,
        fp: _,
    } = final_state;
    let PublicMemory {
        program: _,
        public_segments,
        output: _,
        safe_call_ids: _,
    } = public_memory;
    let PublicSegmentRanges {
        output,
        pedersen,
        range_check_128,
        ecdsa,
        bitwise,
        ec_op,
        keccak,
        poseidon,
        range_check_96,
        add_mod,
        mul_mod,
    } = public_segments;
    assert_segment_schema(output);
    for segment in [
        pedersen,
        range_check_128,
        ecdsa,
        bitwise,
        ec_op,
        keccak,
        poseidon,
        range_check_96,
        add_mod,
        mul_mod,
    ]
    .into_iter()
    .flatten()
    {
        assert_segment_schema(segment);
    }
}

fn assert_segment_schema(segment: &SegmentRange) {
    let SegmentRange {
        start_ptr,
        stop_ptr,
    } = segment;
    let MemorySmallValue { id: _, value: _ } = start_ptr;
    let MemorySmallValue { id: _, value: _ } = stop_ptr;
}

fn mutate_casm_state(state: &mut CasmState) {
    let CasmState { pc, ap, fp } = state;
    for value in [pc, ap, fp] {
        *value = if value.0 == 1 {
            BaseField::from(2)
        } else {
            BaseField::from(1)
        };
    }
}

fn mutate_segment(segment: &mut SegmentRange, preserve_start_value: bool) {
    let SegmentRange {
        start_ptr,
        stop_ptr,
    } = segment;
    let MemorySmallValue { id, value } = start_ptr;
    perturb_u32(id);
    if !preserve_start_value {
        perturb_u32(value);
    }
    let MemorySmallValue { id, value } = stop_ptr;
    perturb_u32(id);
    perturb_u32(value);
}

fn perturb_u32(value: &mut u32) {
    *value = if BaseField::from(*value).0 == 1 { 2 } else { 1 };
}
