// This file was created by the AIR team.

use std::sync::Arc;

use cairo_air::air::PublicData;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::components::blake_g::InteractionClaim as BlakeGInteractionClaim;
use cairo_air::components::memory_id_to_big::InteractionClaim as MemoryBigInteractionClaim;
use cairo_air::components::memory_id_to_small::InteractionClaim as MemorySmallInteractionClaim;
use cairo_air::relations::CommonLookupElements;
use indexmap::IndexSet;
use rayon::scope;
use stwo::core::fields::qm31::SecureField;
pub use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo_cairo_adapter::builtins::BuiltinSegments;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_adapter::opcodes::CasmStatesByOpcode;
use stwo_cairo_common::builtins::*;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, MAX_SEQUENCE_LOG_SIZE,
};
use stwo_cairo_common::preprocessed_columns::simd_prelude::{BaseField, CircleEvaluation};

use crate::witness::base_trace::BaseTrace;
use crate::witness::blake_g_witness_backend::BlakeGWitness;
use crate::witness::blake_round_witness_backend::BlakeRoundWitness;
use crate::witness::components::*;
use crate::witness::exec_context::WitnessExecContext; // witness_exec_context_codegen
use crate::witness::jit_prove_backend::{
    AddApOpcodeLane, AddOpcodeLane, AddOpcodeSmallLane, AssertEqOpcodeDoubleDerefLane,
    AssertEqOpcodeImmLane, AssertEqOpcodeLane, BlakeCompressOpcodeLane, BlakeRoundLane,
    CallOpcodeAbsLane, CallOpcodeRelImmLane, Cube252Lane, Cube252Witness, JnzOpcodeNonTakenLane,
    JnzOpcodeTakenLane, JumpOpcodeAbsLane, JumpOpcodeDoubleDerefLane, JumpOpcodeRelImmLane,
    JumpOpcodeRelLane, MulOpcodeLane, MulOpcodeSmallLane, OpcodeJitBackend,
    PartialEcMulGenericLane, PartialEcMulW18Lane, PedersenAggregatorW18Lane,
    RangeCheck252Width27Lane, RecordedFlatWitness, RetOpcodeLane, TripleXor32Lane,
    VerifyInstructionLane,
};
use crate::witness::memory_witness_backend::MemoryIdToBigWitness;
use crate::witness::pedersen_witness_backend::{
    PartialEcMulGenericWitness, PartialEcMulWindowBits18Witness,
    PedersenAggregatorWindowBits18Witness,
};

#[derive(Default)]
pub struct CairoClaimGenerator {
    pub public_data: PublicData,
    pub add_opcode: Option<add_opcode::ClaimGenerator>,
    /// Adapter memory retained for prepared resident witness planning. This Arc
    /// is also the legacy JIT lane's execution-table source when that lane is on.
    pub jit_memory: Option<Arc<Memory>>,
    pub add_opcode_small: Option<add_opcode_small::ClaimGenerator>,
    pub add_ap_opcode: Option<add_ap_opcode::ClaimGenerator>,
    pub assert_eq_opcode: Option<assert_eq_opcode::ClaimGenerator>,
    pub assert_eq_opcode_imm: Option<assert_eq_opcode_imm::ClaimGenerator>,
    pub assert_eq_opcode_double_deref: Option<assert_eq_opcode_double_deref::ClaimGenerator>,
    pub blake_compress_opcode: Option<blake_compress_opcode::ClaimGenerator>,
    pub call_opcode_abs: Option<call_opcode_abs::ClaimGenerator>,
    pub call_opcode_rel_imm: Option<call_opcode_rel_imm::ClaimGenerator>,
    pub generic_opcode: Option<generic_opcode::ClaimGenerator>,
    pub jnz_opcode_non_taken: Option<jnz_opcode_non_taken::ClaimGenerator>,
    pub jnz_opcode_taken: Option<jnz_opcode_taken::ClaimGenerator>,
    pub jump_opcode_abs: Option<jump_opcode_abs::ClaimGenerator>,
    pub jump_opcode_double_deref: Option<jump_opcode_double_deref::ClaimGenerator>,
    pub jump_opcode_rel: Option<jump_opcode_rel::ClaimGenerator>,
    pub jump_opcode_rel_imm: Option<jump_opcode_rel_imm::ClaimGenerator>,
    pub mul_opcode: Option<mul_opcode::ClaimGenerator>,
    pub mul_opcode_small: Option<mul_opcode_small::ClaimGenerator>,
    pub qm_31_add_mul_opcode: Option<qm_31_add_mul_opcode::ClaimGenerator>,
    pub ret_opcode: Option<ret_opcode::ClaimGenerator>,
    pub verify_instruction: Option<verify_instruction::ClaimGenerator>,
    pub blake_round: Option<blake_round::ClaimGenerator>,
    pub blake_g: Option<blake_g::ClaimGenerator>,
    pub blake_round_sigma: Option<blake_round_sigma::ClaimGenerator>,
    pub triple_xor_32: Option<triple_xor_32::ClaimGenerator>,
    pub verify_bitwise_xor_12: Option<verify_bitwise_xor_12::ClaimGenerator>,
    pub add_mod_builtin: Option<add_mod_builtin::ClaimGenerator>,
    pub bitwise_builtin: Option<bitwise_builtin::ClaimGenerator>,
    pub mul_mod_builtin: Option<mul_mod_builtin::ClaimGenerator>,
    pub pedersen_builtin: Option<pedersen_builtin::ClaimGenerator>,
    pub pedersen_builtin_narrow_windows: Option<pedersen_builtin_narrow_windows::ClaimGenerator>,
    pub poseidon_builtin: Option<poseidon_builtin::ClaimGenerator>,
    pub range_check96_builtin: Option<range_check96_builtin::ClaimGenerator>,
    pub range_check_builtin: Option<range_check_builtin::ClaimGenerator>,
    pub ec_op_builtin: Option<ec_op_builtin::ClaimGenerator>,
    pub partial_ec_mul_generic: Option<partial_ec_mul_generic::ClaimGenerator>,
    pub pedersen_aggregator_window_bits_18:
        Option<pedersen_aggregator_window_bits_18::ClaimGenerator>,
    pub partial_ec_mul_window_bits_18: Option<partial_ec_mul_window_bits_18::ClaimGenerator>,
    pub pedersen_points_table_window_bits_18:
        Option<pedersen_points_table_window_bits_18::ClaimGenerator>,
    pub pedersen_aggregator_window_bits_9:
        Option<pedersen_aggregator_window_bits_9::ClaimGenerator>,
    pub partial_ec_mul_window_bits_9: Option<partial_ec_mul_window_bits_9::ClaimGenerator>,
    pub pedersen_points_table_window_bits_9:
        Option<pedersen_points_table_window_bits_9::ClaimGenerator>,
    pub poseidon_aggregator: Option<poseidon_aggregator::ClaimGenerator>,
    pub poseidon_3_partial_rounds_chain: Option<poseidon_3_partial_rounds_chain::ClaimGenerator>,
    pub poseidon_full_round_chain: Option<poseidon_full_round_chain::ClaimGenerator>,
    pub cube_252: Option<cube_252::ClaimGenerator>,
    pub poseidon_round_keys: Option<poseidon_round_keys::ClaimGenerator>,
    pub range_check_252_width_27: Option<range_check_252_width_27::ClaimGenerator>,
    pub memory_address_to_id: Option<memory_address_to_id::ClaimGenerator>,
    pub memory_id_to_big: Option<memory_id_to_big::ClaimGenerator>,
    pub range_check_6: Option<range_check_6::ClaimGenerator>,
    pub range_check_8: Option<range_check_8::ClaimGenerator>,
    pub range_check_11: Option<range_check_11::ClaimGenerator>,
    pub range_check_12: Option<range_check_12::ClaimGenerator>,
    pub range_check_18: Option<range_check_18::ClaimGenerator>,
    pub range_check_20: Option<range_check_20::ClaimGenerator>,
    pub range_check_4_3: Option<range_check_4_3::ClaimGenerator>,
    pub range_check_4_4: Option<range_check_4_4::ClaimGenerator>,
    pub range_check_9_9: Option<range_check_9_9::ClaimGenerator>,
    pub range_check_7_2_5: Option<range_check_7_2_5::ClaimGenerator>,
    pub range_check_3_6_6_3: Option<range_check_3_6_6_3::ClaimGenerator>,
    pub range_check_4_4_4_4: Option<range_check_4_4_4_4::ClaimGenerator>,
    pub range_check_3_3_3_3_3: Option<range_check_3_3_3_3_3::ClaimGenerator>,
    pub verify_bitwise_xor_4: Option<verify_bitwise_xor_4::ClaimGenerator>,
    pub verify_bitwise_xor_7: Option<verify_bitwise_xor_7::ClaimGenerator>,
    pub verify_bitwise_xor_8: Option<verify_bitwise_xor_8::ClaimGenerator>,
    pub verify_bitwise_xor_9: Option<verify_bitwise_xor_9::ClaimGenerator>,
}

impl CairoClaimGenerator {
    #[allow(clippy::redundant_closure)]
    pub fn fill_components(
        &mut self,
        components: &IndexSet<&str>,
        casm_states_by_opcode: CasmStatesByOpcode,
        builtin_segments: &BuiltinSegments,
        memory: Arc<Memory>,
        preprocessed_trace: Arc<PreProcessedTrace>,
    ) {
        // Prepared resident planning happens before any legacy writer or env-gated
        // lane executes, so retain the immutable adapter memory unconditionally.
        // This is one Arc; SIMD execution is otherwise unchanged.
        self.jit_memory = Some(memory.clone());
        let Self {
            jit_memory: _,
            add_opcode: add_opcode_ref,
            add_opcode_small: add_opcode_small_ref,
            add_ap_opcode: add_ap_opcode_ref,
            assert_eq_opcode: assert_eq_opcode_ref,
            assert_eq_opcode_imm: assert_eq_opcode_imm_ref,
            assert_eq_opcode_double_deref: assert_eq_opcode_double_deref_ref,
            blake_compress_opcode: blake_compress_opcode_ref,
            call_opcode_abs: call_opcode_abs_ref,
            call_opcode_rel_imm: call_opcode_rel_imm_ref,
            generic_opcode: generic_opcode_ref,
            jnz_opcode_non_taken: jnz_opcode_non_taken_ref,
            jnz_opcode_taken: jnz_opcode_taken_ref,
            jump_opcode_abs: jump_opcode_abs_ref,
            jump_opcode_double_deref: jump_opcode_double_deref_ref,
            jump_opcode_rel: jump_opcode_rel_ref,
            jump_opcode_rel_imm: jump_opcode_rel_imm_ref,
            mul_opcode: mul_opcode_ref,
            mul_opcode_small: mul_opcode_small_ref,
            qm_31_add_mul_opcode: qm_31_add_mul_opcode_ref,
            ret_opcode: ret_opcode_ref,
            verify_instruction: verify_instruction_ref,
            blake_round: blake_round_ref,
            blake_g: blake_g_ref,
            blake_round_sigma: blake_round_sigma_ref,
            triple_xor_32: triple_xor_32_ref,
            verify_bitwise_xor_12: verify_bitwise_xor_12_ref,
            add_mod_builtin: add_mod_builtin_ref,
            bitwise_builtin: bitwise_builtin_ref,
            mul_mod_builtin: mul_mod_builtin_ref,
            pedersen_builtin: pedersen_builtin_ref,
            pedersen_builtin_narrow_windows: pedersen_builtin_narrow_windows_ref,
            poseidon_builtin: poseidon_builtin_ref,
            range_check96_builtin: range_check96_builtin_ref,
            range_check_builtin: range_check_builtin_ref,
            ec_op_builtin: ec_op_builtin_ref,
            partial_ec_mul_generic: partial_ec_mul_generic_ref,
            pedersen_aggregator_window_bits_18: pedersen_aggregator_window_bits_18_ref,
            partial_ec_mul_window_bits_18: partial_ec_mul_window_bits_18_ref,
            pedersen_points_table_window_bits_18: pedersen_points_table_window_bits_18_ref,
            pedersen_aggregator_window_bits_9: pedersen_aggregator_window_bits_9_ref,
            partial_ec_mul_window_bits_9: partial_ec_mul_window_bits_9_ref,
            pedersen_points_table_window_bits_9: pedersen_points_table_window_bits_9_ref,
            poseidon_aggregator: poseidon_aggregator_ref,
            poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_ref,
            poseidon_full_round_chain: poseidon_full_round_chain_ref,
            cube_252: cube_252_ref,
            poseidon_round_keys: poseidon_round_keys_ref,
            range_check_252_width_27: range_check_252_width_27_ref,
            memory_address_to_id: memory_address_to_id_ref,
            memory_id_to_big: memory_id_to_big_ref,
            range_check_6: range_check_6_ref,
            range_check_8: range_check_8_ref,
            range_check_11: range_check_11_ref,
            range_check_12: range_check_12_ref,
            range_check_18: range_check_18_ref,
            range_check_20: range_check_20_ref,
            range_check_4_3: range_check_4_3_ref,
            range_check_4_4: range_check_4_4_ref,
            range_check_9_9: range_check_9_9_ref,
            range_check_7_2_5: range_check_7_2_5_ref,
            range_check_3_6_6_3: range_check_3_6_6_3_ref,
            range_check_4_4_4_4: range_check_4_4_4_4_ref,
            range_check_3_3_3_3_3: range_check_3_3_3_3_3_ref,
            verify_bitwise_xor_4: verify_bitwise_xor_4_ref,
            verify_bitwise_xor_7: verify_bitwise_xor_7_ref,
            verify_bitwise_xor_8: verify_bitwise_xor_8_ref,
            verify_bitwise_xor_9: verify_bitwise_xor_9_ref,
            public_data: _,
        } = self;

        scope(|s| {
            if components.contains(&"add_opcode") {
                s.spawn(|_| {
                    *add_opcode_ref = Some(add_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.add_opcode,
                    ));
                });
            }
            if components.contains(&"add_opcode_small") {
                s.spawn(|_| {
                    *add_opcode_small_ref = Some(add_opcode_small::ClaimGenerator::new(
                        casm_states_by_opcode.add_opcode_small,
                    ));
                });
            }
            if components.contains(&"add_ap_opcode") {
                s.spawn(|_| {
                    *add_ap_opcode_ref = Some(add_ap_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.add_ap_opcode,
                    ));
                });
            }
            if components.contains(&"assert_eq_opcode") {
                s.spawn(|_| {
                    *assert_eq_opcode_ref = Some(assert_eq_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.assert_eq_opcode,
                    ));
                });
            }
            if components.contains(&"assert_eq_opcode_imm") {
                s.spawn(|_| {
                    *assert_eq_opcode_imm_ref = Some(assert_eq_opcode_imm::ClaimGenerator::new(
                        casm_states_by_opcode.assert_eq_opcode_imm,
                    ));
                });
            }
            if components.contains(&"assert_eq_opcode_double_deref") {
                s.spawn(|_| {
                    *assert_eq_opcode_double_deref_ref =
                        Some(assert_eq_opcode_double_deref::ClaimGenerator::new(
                            casm_states_by_opcode.assert_eq_opcode_double_deref,
                        ));
                });
            }
            if components.contains(&"blake_compress_opcode") {
                s.spawn(|_| {
                    *blake_compress_opcode_ref = Some(blake_compress_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.blake_compress_opcode,
                    ));
                });
            }
            if components.contains(&"call_opcode_abs") {
                s.spawn(|_| {
                    *call_opcode_abs_ref = Some(call_opcode_abs::ClaimGenerator::new(
                        casm_states_by_opcode.call_opcode_abs,
                    ));
                });
            }
            if components.contains(&"call_opcode_rel_imm") {
                s.spawn(|_| {
                    *call_opcode_rel_imm_ref = Some(call_opcode_rel_imm::ClaimGenerator::new(
                        casm_states_by_opcode.call_opcode_rel_imm,
                    ));
                });
            }
            if components.contains(&"generic_opcode") {
                s.spawn(|_| {
                    *generic_opcode_ref = Some(generic_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.generic_opcode,
                    ));
                });
            }
            if components.contains(&"jnz_opcode_non_taken") {
                s.spawn(|_| {
                    *jnz_opcode_non_taken_ref = Some(jnz_opcode_non_taken::ClaimGenerator::new(
                        casm_states_by_opcode.jnz_opcode_non_taken,
                    ));
                });
            }
            if components.contains(&"jnz_opcode_taken") {
                s.spawn(|_| {
                    *jnz_opcode_taken_ref = Some(jnz_opcode_taken::ClaimGenerator::new(
                        casm_states_by_opcode.jnz_opcode_taken,
                    ));
                });
            }
            if components.contains(&"jump_opcode_abs") {
                s.spawn(|_| {
                    *jump_opcode_abs_ref = Some(jump_opcode_abs::ClaimGenerator::new(
                        casm_states_by_opcode.jump_opcode_abs,
                    ));
                });
            }
            if components.contains(&"jump_opcode_double_deref") {
                s.spawn(|_| {
                    *jump_opcode_double_deref_ref =
                        Some(jump_opcode_double_deref::ClaimGenerator::new(
                            casm_states_by_opcode.jump_opcode_double_deref,
                        ));
                });
            }
            if components.contains(&"jump_opcode_rel") {
                s.spawn(|_| {
                    *jump_opcode_rel_ref = Some(jump_opcode_rel::ClaimGenerator::new(
                        casm_states_by_opcode.jump_opcode_rel,
                    ));
                });
            }
            if components.contains(&"jump_opcode_rel_imm") {
                s.spawn(|_| {
                    *jump_opcode_rel_imm_ref = Some(jump_opcode_rel_imm::ClaimGenerator::new(
                        casm_states_by_opcode.jump_opcode_rel_imm,
                    ));
                });
            }
            if components.contains(&"mul_opcode") {
                s.spawn(|_| {
                    *mul_opcode_ref = Some(mul_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.mul_opcode,
                    ));
                });
            }
            if components.contains(&"mul_opcode_small") {
                s.spawn(|_| {
                    *mul_opcode_small_ref = Some(mul_opcode_small::ClaimGenerator::new(
                        casm_states_by_opcode.mul_opcode_small,
                    ));
                });
            }
            if components.contains(&"qm_31_add_mul_opcode") {
                s.spawn(|_| {
                    *qm_31_add_mul_opcode_ref = Some(qm_31_add_mul_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.qm_31_add_mul_opcode,
                    ));
                });
            }
            if components.contains(&"ret_opcode") {
                s.spawn(|_| {
                    *ret_opcode_ref = Some(ret_opcode::ClaimGenerator::new(
                        casm_states_by_opcode.ret_opcode,
                    ));
                });
            }
            if components.contains(&"verify_instruction") {
                s.spawn(|_| {
                    *verify_instruction_ref = Some(verify_instruction::ClaimGenerator::new());
                });
            }
            if components.contains(&"blake_round") {
                s.spawn(|_| {
                    *blake_round_ref = Some(blake_round::ClaimGenerator::new(memory.clone()));
                });
            }
            if components.contains(&"blake_g") {
                s.spawn(|_| {
                    *blake_g_ref = Some(blake_g::ClaimGenerator::new());
                });
            }
            if components.contains(&"blake_round_sigma") {
                s.spawn(|_| {
                    *blake_round_sigma_ref = Some(blake_round_sigma::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"triple_xor_32") {
                s.spawn(|_| {
                    *triple_xor_32_ref = Some(triple_xor_32::ClaimGenerator::new());
                });
            }
            if components.contains(&"verify_bitwise_xor_12") {
                s.spawn(|_| {
                    *verify_bitwise_xor_12_ref = Some(verify_bitwise_xor_12::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"add_mod_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("add_mod_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(ADD_MOD_BUILTIN_MEMORY_CELLS),
                        "add_mod_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / ADD_MOD_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "add_mod_builtin instances number is not a power of two"
                    );
                    *add_mod_builtin_ref = Some(add_mod_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"bitwise_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("bitwise_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(BITWISE_BUILTIN_MEMORY_CELLS),
                        "bitwise_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / BITWISE_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "bitwise_builtin instances number is not a power of two"
                    );
                    *bitwise_builtin_ref = Some(bitwise_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"mul_mod_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("mul_mod_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(MUL_MOD_BUILTIN_MEMORY_CELLS),
                        "mul_mod_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / MUL_MOD_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "mul_mod_builtin instances number is not a power of two"
                    );
                    *mul_mod_builtin_ref = Some(mul_mod_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"pedersen_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("pedersen_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(PEDERSEN_BUILTIN_MEMORY_CELLS),
                        "pedersen_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / PEDERSEN_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "pedersen_builtin instances number is not a power of two"
                    );
                    *pedersen_builtin_ref = Some(pedersen_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"pedersen_builtin_narrow_windows") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("pedersen_builtin_narrow_windows").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(PEDERSEN_BUILTIN_NARROW_WINDOWS_MEMORY_CELLS),
                        "pedersen_builtin_narrow_windows segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / PEDERSEN_BUILTIN_NARROW_WINDOWS_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "pedersen_builtin_narrow_windows instances number is not a power of two"
                    );
                    *pedersen_builtin_narrow_windows_ref = Some(pedersen_builtin_narrow_windows::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"poseidon_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("poseidon_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(POSEIDON_BUILTIN_MEMORY_CELLS),
                        "poseidon_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / POSEIDON_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "poseidon_builtin instances number is not a power of two"
                    );
                    *poseidon_builtin_ref = Some(poseidon_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"range_check96_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("range_check96_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(RANGE_CHECK_96_BUILTIN_MEMORY_CELLS),
                        "range_check96_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / RANGE_CHECK_96_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "range_check96_builtin instances number is not a power of two"
                    );
                    *range_check96_builtin_ref = Some(range_check96_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"range_check_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments.get_segment_by_name("range_check_builtin").unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(RANGE_CHECK_BUILTIN_MEMORY_CELLS),
                        "range_check_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / RANGE_CHECK_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "range_check_builtin instances number is not a power of two"
                    );
                    *range_check_builtin_ref = Some(range_check_builtin::ClaimGenerator::new(n_instances.ilog2(), segment.begin_addr as u32));
                });
            }
            if components.contains(&"ec_op_builtin") {
                s.spawn(|_| {
                    let segment = builtin_segments
                        .get_segment_by_name("ec_op_builtin")
                        .unwrap();
                    let segment_length = segment.stop_ptr - segment.begin_addr;
                    assert!(
                        segment_length.is_multiple_of(EC_OP_BUILTIN_MEMORY_CELLS),
                        "ec_op_builtin segment length is not a multiple of it's cells_per_instance"
                    );
                    let n_instances = segment_length / EC_OP_BUILTIN_MEMORY_CELLS;
                    assert!(
                        n_instances.is_power_of_two(),
                        "ec_op_builtin instances number is not a power of two"
                    );
                    *ec_op_builtin_ref = Some(ec_op_builtin::ClaimGenerator::new(
                        n_instances.ilog2(),
                        segment.begin_addr as u32,
                    ));
                });
            }
            if components.contains(&"partial_ec_mul_generic") {
                s.spawn(|_| {
                    *partial_ec_mul_generic_ref =
                        Some(partial_ec_mul_generic::ClaimGenerator::new());
                });
            }
            if components.contains(&"pedersen_aggregator_window_bits_18") {
                s.spawn(|_| {
                    *pedersen_aggregator_window_bits_18_ref =
                        Some(pedersen_aggregator_window_bits_18::ClaimGenerator::new());
                });
            }
            if components.contains(&"partial_ec_mul_window_bits_18") {
                s.spawn(|_| {
                    *partial_ec_mul_window_bits_18_ref =
                        Some(partial_ec_mul_window_bits_18::ClaimGenerator::new());
                });
            }
            if components.contains(&"pedersen_points_table_window_bits_18") {
                s.spawn(|_| {
                    *pedersen_points_table_window_bits_18_ref =
                        Some(pedersen_points_table_window_bits_18::ClaimGenerator::new(
                            preprocessed_trace.clone(),
                        ));
                });
            }
            if components.contains(&"pedersen_aggregator_window_bits_9") {
                s.spawn(|_| {
                    *pedersen_aggregator_window_bits_9_ref =
                        Some(pedersen_aggregator_window_bits_9::ClaimGenerator::new());
                });
            }
            if components.contains(&"partial_ec_mul_window_bits_9") {
                s.spawn(|_| {
                    *partial_ec_mul_window_bits_9_ref =
                        Some(partial_ec_mul_window_bits_9::ClaimGenerator::new());
                });
            }
            if components.contains(&"pedersen_points_table_window_bits_9") {
                s.spawn(|_| {
                    *pedersen_points_table_window_bits_9_ref =
                        Some(pedersen_points_table_window_bits_9::ClaimGenerator::new(
                            preprocessed_trace.clone(),
                        ));
                });
            }
            if components.contains(&"poseidon_aggregator") {
                s.spawn(|_| {
                    *poseidon_aggregator_ref = Some(poseidon_aggregator::ClaimGenerator::new());
                });
            }
            if components.contains(&"poseidon_3_partial_rounds_chain") {
                s.spawn(|_| {
                    *poseidon_3_partial_rounds_chain_ref =
                        Some(poseidon_3_partial_rounds_chain::ClaimGenerator::new());
                });
            }
            if components.contains(&"poseidon_full_round_chain") {
                s.spawn(|_| {
                    *poseidon_full_round_chain_ref =
                        Some(poseidon_full_round_chain::ClaimGenerator::new());
                });
            }
            if components.contains(&"cube_252") {
                s.spawn(|_| {
                    *cube_252_ref = Some(cube_252::ClaimGenerator::new());
                });
            }
            if components.contains(&"poseidon_round_keys") {
                s.spawn(|_| {
                    *poseidon_round_keys_ref = Some(poseidon_round_keys::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_252_width_27") {
                s.spawn(|_| {
                    *range_check_252_width_27_ref =
                        Some(range_check_252_width_27::ClaimGenerator::new());
                });
            }
            if components.contains(&"memory_address_to_id") {
                s.spawn(|_| {
                    *memory_address_to_id_ref =
                        Some(memory_address_to_id::ClaimGenerator::new(memory.clone()));
                });
            }
            if components.contains(&"memory_id_to_big") {
                s.spawn(|_| {
                    *memory_id_to_big_ref =
                        Some(memory_id_to_big::ClaimGenerator::new(memory.clone()));
                });
            }
            if components.contains(&"range_check_6") {
                s.spawn(|_| {
                    *range_check_6_ref = Some(range_check_6::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_8") {
                s.spawn(|_| {
                    *range_check_8_ref = Some(range_check_8::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_11") {
                s.spawn(|_| {
                    *range_check_11_ref = Some(range_check_11::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_12") {
                s.spawn(|_| {
                    *range_check_12_ref = Some(range_check_12::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_18") {
                s.spawn(|_| {
                    *range_check_18_ref = Some(range_check_18::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_20") {
                s.spawn(|_| {
                    *range_check_20_ref = Some(range_check_20::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_4_3") {
                s.spawn(|_| {
                    *range_check_4_3_ref = Some(range_check_4_3::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_4_4") {
                s.spawn(|_| {
                    *range_check_4_4_ref = Some(range_check_4_4::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_9_9") {
                s.spawn(|_| {
                    *range_check_9_9_ref = Some(range_check_9_9::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_7_2_5") {
                s.spawn(|_| {
                    *range_check_7_2_5_ref = Some(range_check_7_2_5::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_3_6_6_3") {
                s.spawn(|_| {
                    *range_check_3_6_6_3_ref = Some(range_check_3_6_6_3::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_4_4_4_4") {
                s.spawn(|_| {
                    *range_check_4_4_4_4_ref = Some(range_check_4_4_4_4::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"range_check_3_3_3_3_3") {
                s.spawn(|_| {
                    *range_check_3_3_3_3_3_ref = Some(range_check_3_3_3_3_3::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"verify_bitwise_xor_4") {
                s.spawn(|_| {
                    *verify_bitwise_xor_4_ref = Some(verify_bitwise_xor_4::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"verify_bitwise_xor_7") {
                s.spawn(|_| {
                    *verify_bitwise_xor_7_ref = Some(verify_bitwise_xor_7::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"verify_bitwise_xor_8") {
                s.spawn(|_| {
                    *verify_bitwise_xor_8_ref = Some(verify_bitwise_xor_8::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
            if components.contains(&"verify_bitwise_xor_9") {
                s.spawn(|_| {
                    *verify_bitwise_xor_9_ref = Some(verify_bitwise_xor_9::ClaimGenerator::new(
                        preprocessed_trace.clone(),
                    ));
                });
            }
        });
    }

    /// Writes the base trace. Each component's columns convert to `B` inside its
    /// spawned task (`from_simd_evals` — the device upload, for GPU backends), so
    /// transfers overlap the generation of later components; collection order is
    /// unchanged, so the committed column order is identical.
    /// `memory_id_to_big` goes through the [`MemoryIdToBigWitness`] backend hook:
    /// on GPU backends its columns are born on device (witness-on-GPU P1).
    pub fn write_trace<
        B: MemoryIdToBigWitness
            + BlakeGWitness
            + OpcodeJitBackend
            + RecordedFlatWitness
            + BlakeRoundWitness
            + crate::witness::jit_prove_backend::Cube252Witness
            + PartialEcMulGenericWitness
            + PartialEcMulWindowBits18Witness
            + PedersenAggregatorWindowBits18Witness
            + PolyOps,
    >(
        mut self,
        exec_context: &WitnessExecContext,
        opt_n_id_to_big_components: Option<usize>,
        // Stage A″ (pipelined commit): when `Some`, the opcode-prefix columns are
        // interpolated on a committer thread with this twiddle tree WHILE the serial
        // host-heavy components below generate, and the whole base trace is returned
        // already interpolated (`BaseTrace::Polys`). `None` = default byte-identical
        // path (`BaseTrace::Evals`, interpolated at commit time). The tree must be
        // `'static` (it comes from the process-wide leaked twiddle cache) so the
        // committer thread can borrow it; its identity is verified against the
        // commitment tree at the call site (fail-closed on a trace-size change).
        pipeline_twiddles: Option<&'static TwiddleTree<B>>,
    ) -> (BaseTrace<B>, CairoClaim, CairoInteractionClaimGenerator<B>) {
        let mut evals = Vec::new();
        let mut add_opcode_result = None;
        let mut add_opcode_small_result = None;
        let mut add_ap_opcode_result = None;
        let mut assert_eq_opcode_result = None;
        let mut assert_eq_opcode_imm_result = None;
        let mut assert_eq_opcode_double_deref_result = None;
        let mut blake_compress_opcode_result = None;
        let mut call_opcode_abs_result = None;
        let mut call_opcode_rel_imm_result = None;
        let mut generic_opcode_result = None;
        let mut jnz_opcode_non_taken_result = None;
        let mut jnz_opcode_taken_result = None;
        let mut jump_opcode_abs_result = None;
        let mut jump_opcode_double_deref_result = None;
        let mut jump_opcode_rel_result = None;
        let mut jump_opcode_rel_imm_result = None;
        let mut mul_opcode_result = None;
        let mut mul_opcode_small_result = None;
        let mut qm_31_add_mul_opcode_result = None;
        let mut ret_opcode_result = None;

        scope(|s| {
            if let Some(gen) = self.add_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut add_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:add_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<AddOpcodeLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.add_opcode_small {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut add_opcode_small_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:add_opcode_small").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<AddOpcodeSmallLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.add_ap_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let rc18_state = self.range_check_18.as_ref().unwrap();
                let rc11_state = self.range_check_11.as_ref().unwrap();
                let result_slot = &mut add_ap_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:add_ap_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as RecordedFlatWitness>::write_add_ap_opcode(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            rc18_state,
                            rc11_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.assert_eq_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut assert_eq_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:assert_eq_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<AssertEqOpcodeLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.assert_eq_opcode_imm {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut assert_eq_opcode_imm_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:assert_eq_opcode_imm").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<AssertEqOpcodeImmLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.assert_eq_opcode_double_deref {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut assert_eq_opcode_double_deref_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:assert_eq_opcode_double_deref").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<AssertEqOpcodeDoubleDerefLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.blake_compress_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let rc725_state = self.range_check_7_2_5.as_ref().unwrap();
                let xor8_state = self.verify_bitwise_xor_8.as_ref().unwrap();
                let round_state = self.blake_round.as_ref().unwrap();
                let triple_xor_state = self.triple_xor_32.as_ref().unwrap();
                let result_slot = &mut blake_compress_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:blake_compress_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as RecordedFlatWitness>::write_blake_compress_opcode(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            rc725_state,
                            xor8_state,
                            round_state,
                            triple_xor_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.call_opcode_abs {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut call_opcode_abs_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:call_opcode_abs").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<CallOpcodeAbsLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.call_opcode_rel_imm {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut call_opcode_rel_imm_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:call_opcode_rel_imm").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<CallOpcodeRelImmLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.generic_opcode {
                s.spawn(|_| {
                    generic_opcode_result = Some({
                        let _wt = tracing::info_span!("wt:generic_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            self.memory_address_to_id.as_ref().unwrap(),
                            self.memory_id_to_big.as_ref().unwrap(),
                            self.verify_instruction.as_ref().unwrap(),
                            self.range_check_9_9.as_ref().unwrap(),
                            self.range_check_20.as_ref().unwrap(),
                            self.range_check_18.as_ref().unwrap(),
                            self.range_check_11.as_ref().unwrap(),
                        );
                        (B::from_simd_evals(trace.to_evals()), claim, interaction_gen)
                    });
                });
            }
            if let Some(gen) = self.jnz_opcode_non_taken {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jnz_opcode_non_taken_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jnz_opcode_non_taken").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JnzOpcodeNonTakenLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.jnz_opcode_taken {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jnz_opcode_taken_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jnz_opcode_taken").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JnzOpcodeTakenLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.jump_opcode_abs {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jump_opcode_abs_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jump_opcode_abs").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JumpOpcodeAbsLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.jump_opcode_double_deref {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jump_opcode_double_deref_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jump_opcode_double_deref").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JumpOpcodeDoubleDerefLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.jump_opcode_rel {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jump_opcode_rel_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jump_opcode_rel").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JumpOpcodeRelLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.jump_opcode_rel_imm {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut jump_opcode_rel_imm_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:jump_opcode_rel_imm").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<JumpOpcodeRelImmLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.mul_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let rc20_state = self.range_check_20.as_ref().unwrap();
                let result_slot = &mut mul_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:mul_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as RecordedFlatWitness>::write_mul_opcode(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            rc20_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.mul_opcode_small {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let rc11_state = self.range_check_11.as_ref().unwrap();
                let result_slot = &mut mul_opcode_small_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:mul_opcode_small").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as RecordedFlatWitness>::write_mul_opcode_small(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            rc11_state,
                            jit_memory,
                        )
                    });
                });
            }
            if let Some(gen) = self.qm_31_add_mul_opcode {
                s.spawn(|_| {
                    qm_31_add_mul_opcode_result = Some({
                        let _wt = tracing::info_span!("wt:qm_31_add_mul_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            self.memory_address_to_id.as_ref().unwrap(),
                            self.memory_id_to_big.as_ref().unwrap(),
                            self.verify_instruction.as_ref().unwrap(),
                            self.range_check_4_4_4_4.as_ref().unwrap(),
                        );
                        (B::from_simd_evals(trace.to_evals()), claim, interaction_gen)
                    });
                });
            }
            if let Some(gen) = self.ret_opcode {
                let jit_memory = self.jit_memory.as_ref();
                let addr_state = self.memory_address_to_id.as_ref().unwrap();
                let id_state = self.memory_id_to_big.as_ref().unwrap();
                let vi_state = self.verify_instruction.as_ref().unwrap();
                let result_slot = &mut ret_opcode_result;
                s.spawn(move |_| {
                    *result_slot = Some({
                        let _wt = tracing::info_span!("wt:ret_opcode").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        <B as OpcodeJitBackend>::lane_write_trace::<RetOpcodeLane>(
                            exec_context,
                            gen,
                            addr_state,
                            id_state,
                            vi_state,
                            jit_memory,
                        )
                    });
                });
            }
        });

        let (add_opcode_claim, add_opcode_interaction_gen) = add_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (add_opcode_small_claim, add_opcode_small_interaction_gen) = add_opcode_small_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (add_ap_opcode_claim, add_ap_opcode_interaction_gen) = add_ap_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (assert_eq_opcode_claim, assert_eq_opcode_interaction_gen) = assert_eq_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (assert_eq_opcode_imm_claim, assert_eq_opcode_imm_interaction_gen) =
            assert_eq_opcode_imm_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (assert_eq_opcode_double_deref_claim, assert_eq_opcode_double_deref_interaction_gen) =
            assert_eq_opcode_double_deref_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (blake_compress_opcode_claim, blake_compress_opcode_interaction_gen) =
            blake_compress_opcode_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (call_opcode_abs_claim, call_opcode_abs_interaction_gen) = call_opcode_abs_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (call_opcode_rel_imm_claim, call_opcode_rel_imm_interaction_gen) =
            call_opcode_rel_imm_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (generic_opcode_claim, generic_opcode_interaction_gen) = generic_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (jnz_opcode_non_taken_claim, jnz_opcode_non_taken_interaction_gen) =
            jnz_opcode_non_taken_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (jnz_opcode_taken_claim, jnz_opcode_taken_interaction_gen) = jnz_opcode_taken_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (jump_opcode_abs_claim, jump_opcode_abs_interaction_gen) = jump_opcode_abs_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (jump_opcode_double_deref_claim, jump_opcode_double_deref_interaction_gen) =
            jump_opcode_double_deref_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (jump_opcode_rel_claim, jump_opcode_rel_interaction_gen) = jump_opcode_rel_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (jump_opcode_rel_imm_claim, jump_opcode_rel_imm_interaction_gen) =
            jump_opcode_rel_imm_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (mul_opcode_claim, mul_opcode_interaction_gen) = mul_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (mul_opcode_small_claim, mul_opcode_small_interaction_gen) = mul_opcode_small_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (qm_31_add_mul_opcode_claim, qm_31_add_mul_opcode_interaction_gen) =
            qm_31_add_mul_opcode_result
                .map(|(trace, claim, interaction_gen)| {
                    evals.extend(trace);
                    (claim, interaction_gen)
                })
                .unzip();
        let (ret_opcode_claim, ret_opcode_interaction_gen) = ret_opcode_result
            .map(|(trace, claim, interaction_gen)| {
                evals.extend(trace);
                (claim, interaction_gen)
            })
            .unzip();

        // Stage A″: the 20 opcode components above ran in the parallel scope and are
        // done + device-resident. If pipelined commit is on, hand their columns (the
        // canonical prefix of `evals`) to a committer thread that interpolates them
        // with the caller's twiddle tree WHILE the serial host-heavy components below
        // (verify_instruction, blake_round, partial_ec_mul, pedersen_aggregator, ...)
        // generate — hiding the opcode iFFTs under the ~2.8s host block. `evals` is
        // emptied here so the rest of the assembly refills it with the suffix columns.
        // M5b: the committer is a CHANNEL — batch 0 is the opcode prefix (as in
        // A2), and every builtin lane sends its columns the moment it finishes,
        // so their iFFTs run on the committer thread WHILE later arms and the
        // sequential tail still generate. Batches carry their canonical index
        // (the drain order below) and are re-sorted at the join, so the final
        // polynomial order is exactly the sequential path's `evals` order.
        #[allow(clippy::type_complexity)]
        let committer: Option<(
            std::sync::mpsc::Sender<(usize, Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>)>,
            std::thread::JoinHandle<Vec<(usize, Vec<CircleCoefficients<B>>)>>,
            usize,
        )> = pipeline_twiddles.map(|tree| {
            let opcode_evals = std::mem::take(&mut evals);
            let tree_ptr = tree as *const TwiddleTree<B> as usize;
            // One-time engage marker so the gate can confirm the ON path actually ran
            // (rather than silently falling back to Evals on a cold twiddle cache).
            {
                use std::sync::Once;
                static ENGAGED: Once = Once::new();
                ENGAGED.call_once(|| {
                    // ASCII-only marker ("A2", not "A\u{2033}"): a grep pattern like
                    // `A. engaged` matches a single BYTE for `.`, which cannot span the
                    // 3-byte UTF-8 prime, so a unicode marker reads as never-engaged.
                    eprintln!(
                        "STWO_CUDA_PIPELINED_COMMIT: A2 engaged - {} opcode-prefix \
                         columns + per-lane builtin batches interpolating on a committer \
                         thread under the witness arms",
                        opcode_evals.len()
                    );
                });
            }
            let (tx, rx) = std::sync::mpsc::channel();
            tx.send((0usize, opcode_evals))
                .expect("committer receiver alive at construction");
            let handle = std::thread::spawn(move || {
                let mut out = Vec::new();
                while let Ok((idx, batch)) = rx.recv() {
                    out.push((idx, B::interpolate_columns(batch, tree)));
                }
                out
            });
            (tx, handle, tree_ptr)
        });

        /// Hand a finished lane's columns to the committer (returning an empty
        /// vec for its slot) or keep them for the sequential interpolate path.
        fn send_or_keep<T>(
            idx: usize,
            ev: Vec<T>,
            tx: &Option<std::sync::mpsc::Sender<(usize, Vec<T>)>>,
        ) -> Vec<T> {
            match tx {
                Some(tx) => {
                    tx.send((idx, ev)).expect("lane committer thread alive");
                    Vec::new()
                }
                None => ev,
            }
        }

        // Canonical batch indices = the drain order below (batch 0 = opcode prefix;
        // the sequential tail's columns interpolate at the join, after all batches).
        const L_VI: usize = 1;
        const L_BLAKE_ROUND: usize = 2;
        const L_BLAKE_G: usize = 3;
        const L_SIGMA: usize = 4;
        const L_TRIPLE_XOR: usize = 5;
        const L_XOR12: usize = 6;
        const L_ADD_MOD: usize = 7;
        const L_BITWISE: usize = 8;
        const L_MUL_MOD: usize = 9;
        const L_PEDERSEN_B: usize = 10;
        const L_NARROW: usize = 11;
        const L_POSEIDON_B: usize = 12;
        const L_RC96: usize = 13;
        const L_RC_B: usize = 14;
        const L_EC_OP: usize = 15;
        const L_EC_GEN: usize = 16;
        const L_AGG18: usize = 17;
        const L_ECM18: usize = 18;
        const L_PTS18: usize = 19;
        const L_AGG9: usize = 20;
        const L_ECM9: usize = 21;
        const L_PTS9: usize = 22;
        const L_POS_AGG: usize = 23;
        const L_POS3: usize = 24;
        const L_POS_FULL: usize = 25;
        const L_CUBE: usize = 26;
        const L_KEYS: usize = 27;
        const L_RC252: usize = 28;
        let lane_tx = committer.as_ref().map(|(tx, ..)| tx.clone());

        // ---------------------------------------------------------------------
        // Builtin lanes: dependency arms (design M5a). The certified schedule +
        // the per-writer state arguments define the edges; each arm below runs
        // its chain in the original file order, arms run CONCURRENTLY (rayon).
        // Cross-arm shared states (memory/range-check/xor tables) take atomic,
        // commutative count adds — the same concurrency contract the opcode
        // scope above already exercises. Consumers of cross-arm-fed states
        // (memory, rc and xor lanes) stay in the sequential tail after the
        // join. `evals` order is restored canonically from per-lane slots.
        // ---------------------------------------------------------------------
        let vi_gen = self.verify_instruction.take();
        let blake_round_gen = self.blake_round.take();
        let blake_g_gen = self.blake_g.take();
        let blake_round_sigma_gen = self.blake_round_sigma.take();
        let triple_xor_32_gen = self.triple_xor_32.take();
        let verify_bitwise_xor_12_gen = self.verify_bitwise_xor_12.take();
        let add_mod_builtin_gen = self.add_mod_builtin.take();
        let bitwise_builtin_gen = self.bitwise_builtin.take();
        let mul_mod_builtin_gen = self.mul_mod_builtin.take();
        let pedersen_builtin_gen = self.pedersen_builtin.take();
        let pedersen_narrow_gen = self.pedersen_builtin_narrow_windows.take();
        let poseidon_builtin_gen = self.poseidon_builtin.take();
        let range_check96_builtin_gen = self.range_check96_builtin.take();
        let range_check_builtin_gen = self.range_check_builtin.take();
        let ec_op_builtin_gen = self.ec_op_builtin.take();
        let partial_ec_mul_generic_gen = self.partial_ec_mul_generic.take();
        let pedersen_aggregator_window_bits_18_gen = self.pedersen_aggregator_window_bits_18.take();
        let partial_ec_mul_window_bits_18_gen = self.partial_ec_mul_window_bits_18.take();
        let pedersen_points_table_window_bits_18_gen =
            self.pedersen_points_table_window_bits_18.take();
        let pedersen_aggregator_window_bits_9_gen = self.pedersen_aggregator_window_bits_9.take();
        let partial_ec_mul_window_bits_9_gen = self.partial_ec_mul_window_bits_9.take();
        let pedersen_points_table_window_bits_9_gen =
            self.pedersen_points_table_window_bits_9.take();
        let poseidon_aggregator_gen = self.poseidon_aggregator.take();
        let poseidon_3_partial_rounds_chain_gen = self.poseidon_3_partial_rounds_chain.take();
        let poseidon_full_round_chain_gen = self.poseidon_full_round_chain.take();
        let cube_252_gen = self.cube_252.take();
        let poseidon_round_keys_gen = self.poseidon_round_keys.take();
        let range_check_252_width_27_gen = self.range_check_252_width_27.take();

        let mut vi_out = None;
        let mut blake_round_out = None;
        let mut blake_g_out = None;
        let mut blake_round_sigma_out = None;
        let mut triple_xor_32_out = None;
        let mut verify_bitwise_xor_12_out = None;
        let mut add_mod_builtin_out = None;
        let mut bitwise_builtin_out = None;
        let mut mul_mod_builtin_out = None;
        let mut pedersen_builtin_out = None;
        let mut pedersen_narrow_out = None;
        let mut poseidon_builtin_out = None;
        let mut range_check96_builtin_out = None;
        let mut range_check_builtin_out = None;
        let mut ec_op_builtin_out = None;
        let mut partial_ec_mul_generic_out = None;
        let mut pedersen_aggregator_window_bits_18_out = None;
        let mut partial_ec_mul_window_bits_18_out = None;
        let mut pedersen_points_table_window_bits_18_out = None;
        let mut pedersen_aggregator_window_bits_9_out = None;
        let mut partial_ec_mul_window_bits_9_out = None;
        let mut pedersen_points_table_window_bits_9_out = None;
        let mut poseidon_aggregator_out = None;
        let mut poseidon_3_partial_rounds_chain_out = None;
        let mut poseidon_full_round_chain_out = None;
        let mut cube_252_out = None;
        let mut poseidon_round_keys_out = None;
        let mut range_check_252_width_27_out = None;

        scope(|s| {
            let jit_memory = self.jit_memory.as_ref();
            let addr_state = self.memory_address_to_id.as_ref();
            let id_state = self.memory_id_to_big.as_ref();
            let rc_7_2_5 = self.range_check_7_2_5.as_ref();
            let rc_4_3 = self.range_check_4_3.as_ref();
            let rc_6 = self.range_check_6.as_ref();
            let rc_8 = self.range_check_8.as_ref();
            let rc_9_9 = self.range_check_9_9.as_ref();
            let rc_12 = self.range_check_12.as_ref();
            let rc_18 = self.range_check_18.as_ref();
            let rc_20 = self.range_check_20.as_ref();
            let rc_3_6_6_3 = self.range_check_3_6_6_3.as_ref();
            let rc_3_3_3_3_3 = self.range_check_3_3_3_3_3.as_ref();
            let rc_4_4_4_4 = self.range_check_4_4_4_4.as_ref();
            let rc_4_4 = self.range_check_4_4.as_ref();
            let vbx_4 = self.verify_bitwise_xor_4.as_ref();
            let vbx_7 = self.verify_bitwise_xor_7.as_ref();
            let vbx_8 = self.verify_bitwise_xor_8.as_ref();
            let vbx_9 = self.verify_bitwise_xor_9.as_ref();

            // Arm V: verify_instruction (all opcode feeders joined above).
            if let Some(gen) = vi_gen {
                let slot = &mut vi_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let _wt = tracing::info_span!("wt:verify_instruction").entered();
                    exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                    let (trace, claim, interaction_gen) =
                        <B as RecordedFlatWitness>::write_verify_instruction(
                            exec_context,
                            gen,
                            rc_7_2_5.unwrap(),
                            rc_4_3.unwrap(),
                            addr_state.unwrap(),
                            id_state.unwrap(),
                            jit_memory,
                        );
                    *slot = Some((send_or_keep(L_VI, trace, &tx), claim, interaction_gen));
                });
            }

            // Arm B: blake_round -> blake_g -> sigma -> triple_xor_32 -> xor_12.
            {
                let blake_round_slot = &mut blake_round_out;
                let blake_g_slot = &mut blake_g_out;
                let sigma_slot = &mut blake_round_sigma_out;
                let txor_slot = &mut triple_xor_32_out;
                let vbx12_slot = &mut verify_bitwise_xor_12_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let sigma_gen = blake_round_sigma_gen;
                    let blake_g_state = blake_g_gen;
                    if let Some(gen) = blake_round_gen {
                        let _wt = tracing::info_span!("wt:blake_round").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                                                                                               // blake_round goes through the [`BlakeRoundWitness`] backend hook:
                                                                                               // SimdBackend runs the host writer; CudaBackend's device lane
                                                                                               // (pod-gated) is born on device and feeds blake_g device-to-device.
                        let (trace, claim, interaction_gen) = <B as BlakeRoundWitness>::write_trace(
                            exec_context,
                            gen,
                            sigma_gen.as_ref().unwrap(),
                            addr_state.unwrap(),
                            id_state.unwrap(),
                            rc_7_2_5.unwrap(),
                            blake_g_state.as_ref().unwrap(),
                            jit_memory,
                        );
                        *blake_round_slot = Some((
                            send_or_keep(L_BLAKE_ROUND, trace, &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = blake_g_state {
                        let _wt = tracing::info_span!("wt:blake_g").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = <B as BlakeGWitness>::write_trace(
                            exec_context,
                            gen,
                            vbx_8.unwrap(),
                            verify_bitwise_xor_12_gen.as_ref().unwrap(),
                            vbx_4.unwrap(),
                            vbx_7.unwrap(),
                            vbx_9.unwrap(),
                        );
                        *blake_g_slot =
                            Some((send_or_keep(L_BLAKE_G, trace, &tx), claim, interaction_gen));
                    }
                    if let Some(gen) = sigma_gen {
                        let _wt = tracing::info_span!("wt:blake_round_sigma").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace();
                        *sigma_slot = Some((
                            send_or_keep(L_SIGMA, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = triple_xor_32_gen {
                        let _wt = tracing::info_span!("wt:triple_xor_32").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            <B as RecordedFlatWitness>::write_triple_xor_32(
                                exec_context,
                                gen,
                                vbx_8.unwrap(),
                                jit_memory,
                            );
                        *txor_slot = Some((
                            send_or_keep(L_TRIPLE_XOR, trace, &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = verify_bitwise_xor_12_gen {
                        let _wt = tracing::info_span!("wt:verify_bitwise_xor_12").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace();
                        *vbx12_slot = Some((
                            send_or_keep(L_XOR12, B::from_simd_evals(trace), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                });
            }

            // Arm M: mod/bitwise/rc builtins (feed only atomic count states).
            {
                let add_mod_slot = &mut add_mod_builtin_out;
                let bitwise_slot = &mut bitwise_builtin_out;
                let mul_mod_slot = &mut mul_mod_builtin_out;
                let rc96_slot = &mut range_check96_builtin_out;
                let rcb_slot = &mut range_check_builtin_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    if let Some(gen) = add_mod_builtin_gen {
                        let _wt = tracing::info_span!("wt:add_mod_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), id_state.unwrap());
                        *add_mod_slot = Some((
                            send_or_keep(L_ADD_MOD, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = bitwise_builtin_gen {
                        let _wt = tracing::info_span!("wt:bitwise_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            addr_state.unwrap(),
                            id_state.unwrap(),
                            vbx_9.unwrap(),
                            vbx_8.unwrap(),
                        );
                        *bitwise_slot = Some((
                            send_or_keep(L_BITWISE, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = mul_mod_builtin_gen {
                        let _wt = tracing::info_span!("wt:mul_mod_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            addr_state.unwrap(),
                            id_state.unwrap(),
                            rc_12.unwrap(),
                            rc_3_6_6_3.unwrap(),
                            rc_18.unwrap(),
                        );
                        *mul_mod_slot = Some((
                            send_or_keep(L_MUL_MOD, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = range_check96_builtin_gen {
                        let _wt = tracing::info_span!("wt:range_check96_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), id_state.unwrap(), rc_6.unwrap());
                        *rc96_slot = Some((
                            send_or_keep(L_RC96, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = range_check_builtin_gen {
                        let _wt = tracing::info_span!("wt:range_check_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), id_state.unwrap());
                        *rcb_slot = Some((
                            send_or_keep(L_RC_B, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                });
            }

            // Arm P18: pedersen_builtin -> aggregator_w18 -> partial_ec_mul_w18 ->
            // points_table_w18.
            {
                let ped_slot = &mut pedersen_builtin_out;
                let agg18_slot = &mut pedersen_aggregator_window_bits_18_out;
                let ecm18_slot = &mut partial_ec_mul_window_bits_18_out;
                let pts18_slot = &mut pedersen_points_table_window_bits_18_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let agg18_gen = pedersen_aggregator_window_bits_18_gen;
                    let ecm18_gen = partial_ec_mul_window_bits_18_gen;
                    let pts18_gen = pedersen_points_table_window_bits_18_gen;
                    if let Some(gen) = pedersen_builtin_gen {
                        let _wt = tracing::info_span!("wt:pedersen_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), agg18_gen.as_ref().unwrap());
                        *ped_slot = Some((
                            send_or_keep(L_PEDERSEN_B, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = agg18_gen {
                        let _wt =
                            tracing::info_span!("wt:pedersen_aggregator_window_bits_18").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            <B as PedersenAggregatorWindowBits18Witness>::write_trace(
                                exec_context,
                                gen,
                                id_state.unwrap(),
                                rc_8.unwrap(),
                                ecm18_gen.as_ref().unwrap(),
                                jit_memory,
                            );
                        *agg18_slot =
                            Some((send_or_keep(L_AGG18, trace, &tx), claim, interaction_gen));
                    }
                    if let Some(gen) = ecm18_gen {
                        let _wt = tracing::info_span!("wt:partial_ec_mul_window_bits_18").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            <B as PartialEcMulWindowBits18Witness>::write_trace(
                                exec_context,
                                gen,
                                pts18_gen.as_ref().unwrap(),
                                rc_9_9.unwrap(),
                                rc_20.unwrap(),
                                jit_memory,
                            );
                        *ecm18_slot =
                            Some((send_or_keep(L_ECM18, trace, &tx), claim, interaction_gen));
                    }
                    if let Some(gen) = pts18_gen {
                        let _wt = tracing::info_span!("wt:pedersen_points_table_window_bits_18")
                            .entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace();
                        *pts18_slot = Some((
                            send_or_keep(L_PTS18, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                });
            }

            // Arm P9: narrow_windows -> aggregator_w9 -> partial_ec_mul_w9 -> points_table_w9.
            {
                let narrow_slot = &mut pedersen_narrow_out;
                let agg9_slot = &mut pedersen_aggregator_window_bits_9_out;
                let ecm9_slot = &mut partial_ec_mul_window_bits_9_out;
                let pts9_slot = &mut pedersen_points_table_window_bits_9_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let agg9_gen = pedersen_aggregator_window_bits_9_gen;
                    let ecm9_gen = partial_ec_mul_window_bits_9_gen;
                    let pts9_gen = pedersen_points_table_window_bits_9_gen;
                    if let Some(gen) = pedersen_narrow_gen {
                        let _wt =
                            tracing::info_span!("wt:pedersen_builtin_narrow_windows").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), agg9_gen.as_ref().unwrap());
                        *narrow_slot = Some((
                            send_or_keep(L_NARROW, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = agg9_gen {
                        let _wt =
                            tracing::info_span!("wt:pedersen_aggregator_window_bits_9").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            id_state.unwrap(),
                            rc_8.unwrap(),
                            ecm9_gen.as_ref().unwrap(),
                        );
                        *agg9_slot = Some((
                            send_or_keep(L_AGG9, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = ecm9_gen {
                        let _wt = tracing::info_span!("wt:partial_ec_mul_window_bits_9").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            pts9_gen.as_ref().unwrap(),
                            rc_9_9.unwrap(),
                            rc_20.unwrap(),
                        );
                        *ecm9_slot = Some((
                            send_or_keep(L_ECM9, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = pts9_gen {
                        let _wt =
                            tracing::info_span!("wt:pedersen_points_table_window_bits_9").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace();
                        *pts9_slot = Some((
                            send_or_keep(L_PTS9, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                });
            }

            // Arm E: ec_op_builtin -> partial_ec_mul_generic.
            {
                let ec_op_slot = &mut ec_op_builtin_out;
                let ec_gen_slot = &mut partial_ec_mul_generic_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let ec_generic_gen = partial_ec_mul_generic_gen;
                    if let Some(gen) = ec_op_builtin_gen {
                        let _wt = tracing::info_span!("wt:ec_op_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            addr_state.unwrap(),
                            id_state.unwrap(),
                            rc_8.unwrap(),
                            ec_generic_gen.as_ref().unwrap(),
                        );
                        *ec_op_slot = Some((
                            send_or_keep(L_EC_OP, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = ec_generic_gen {
                        let _wt = tracing::info_span!("wt:partial_ec_mul_generic").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            <B as PartialEcMulGenericWitness>::write_trace(
                                exec_context,
                                gen,
                                rc_8.unwrap(),
                                rc_9_9.unwrap(),
                                rc_20.unwrap(),
                                jit_memory,
                            );
                        *ec_gen_slot =
                            Some((send_or_keep(L_EC_GEN, trace, &tx), claim, interaction_gen));
                    }
                });
            }

            // Arm S: poseidon_builtin -> aggregator -> partial/full chains -> cube -> keys ->
            // rc252.
            {
                let pos_b_slot = &mut poseidon_builtin_out;
                let pos_agg_slot = &mut poseidon_aggregator_out;
                let pos3_slot = &mut poseidon_3_partial_rounds_chain_out;
                let pos_full_slot = &mut poseidon_full_round_chain_out;
                let cube_slot = &mut cube_252_out;
                let keys_slot = &mut poseidon_round_keys_out;
                let rc252_slot = &mut range_check_252_width_27_out;
                let tx = lane_tx.clone();
                s.spawn(move |_| {
                    let pos_agg_gen = poseidon_aggregator_gen;
                    let pos3_gen = poseidon_3_partial_rounds_chain_gen;
                    let pos_full_gen = poseidon_full_round_chain_gen;
                    let cube_gen = cube_252_gen;
                    let keys_gen = poseidon_round_keys_gen;
                    let rc252_gen = range_check_252_width_27_gen;
                    if let Some(gen) = poseidon_builtin_gen {
                        let _wt = tracing::info_span!("wt:poseidon_builtin").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            gen.write_trace(addr_state.unwrap(), pos_agg_gen.as_ref().unwrap());
                        *pos_b_slot = Some((
                            send_or_keep(L_POSEIDON_B, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = pos_agg_gen {
                        let _wt = tracing::info_span!("wt:poseidon_aggregator").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            id_state.unwrap(),
                            pos_full_gen.as_ref().unwrap(),
                            rc252_gen.as_ref().unwrap(),
                            cube_gen.as_ref().unwrap(),
                            rc_3_3_3_3_3.unwrap(),
                            rc_4_4_4_4.unwrap(),
                            rc_4_4.unwrap(),
                            pos3_gen.as_ref().unwrap(),
                        );
                        *pos_agg_slot = Some((
                            send_or_keep(L_POS_AGG, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = pos3_gen {
                        let _wt =
                            tracing::info_span!("wt:poseidon_3_partial_rounds_chain").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            keys_gen.as_ref().unwrap(),
                            cube_gen.as_ref().unwrap(),
                            rc_4_4_4_4.unwrap(),
                            rc_4_4.unwrap(),
                            rc252_gen.as_ref().unwrap(),
                        );
                        *pos3_slot = Some((
                            send_or_keep(L_POS3, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = pos_full_gen {
                        let _wt = tracing::info_span!("wt:poseidon_full_round_chain").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace(
                            cube_gen.as_ref().unwrap(),
                            keys_gen.as_ref().unwrap(),
                            rc_3_3_3_3_3.unwrap(),
                        );
                        *pos_full_slot = Some((
                            send_or_keep(L_POS_FULL, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = cube_gen {
                        let _wt = tracing::info_span!("wt:cube_252").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = <B as Cube252Witness>::write_trace(
                            exec_context,
                            gen,
                            rc_9_9.unwrap(),
                            rc_20.unwrap(),
                            jit_memory,
                        );
                        *cube_slot =
                            Some((send_or_keep(L_CUBE, trace, &tx), claim, interaction_gen));
                    }
                    if let Some(gen) = keys_gen {
                        let _wt = tracing::info_span!("wt:poseidon_round_keys").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) = gen.write_trace();
                        *keys_slot = Some((
                            send_or_keep(L_KEYS, B::from_simd_evals(trace.to_evals()), &tx),
                            claim,
                            interaction_gen,
                        ));
                    }
                    if let Some(gen) = rc252_gen {
                        let _wt = tracing::info_span!("wt:range_check_252_width_27").entered();
                        exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                        let (trace, claim, interaction_gen) =
                            <B as RecordedFlatWitness>::write_range_check_252_width_27(
                                exec_context,
                                gen,
                                rc_9_9.unwrap(),
                                rc_18.unwrap(),
                                jit_memory,
                            );
                        *rc252_slot =
                            Some((send_or_keep(L_RC252, trace, &tx), claim, interaction_gen));
                    }
                });
            }
        });

        // Canonical eval order (the exact pre-arm file order) + claim variables.
        macro_rules! drain {
            ($out:expr) => {
                $out.map(|(ev, claim, igen)| {
                    evals.extend(ev);
                    (claim, igen)
                })
                .unzip()
            };
        }
        let (verify_instruction_claim, verify_instruction_interaction_gen) = drain!(vi_out);
        let (blake_round_claim, blake_round_interaction_gen) = drain!(blake_round_out);
        let (blake_g_claim, blake_g_interaction_gen) = drain!(blake_g_out);
        let (blake_round_sigma_claim, blake_round_sigma_interaction_gen) =
            drain!(blake_round_sigma_out);
        let (triple_xor_32_claim, triple_xor_32_interaction_gen) = drain!(triple_xor_32_out);
        let (verify_bitwise_xor_12_claim, verify_bitwise_xor_12_interaction_gen) =
            drain!(verify_bitwise_xor_12_out);
        let (add_mod_builtin_claim, add_mod_builtin_interaction_gen) = drain!(add_mod_builtin_out);
        let (bitwise_builtin_claim, bitwise_builtin_interaction_gen) = drain!(bitwise_builtin_out);
        let (mul_mod_builtin_claim, mul_mod_builtin_interaction_gen) = drain!(mul_mod_builtin_out);
        let (pedersen_builtin_claim, pedersen_builtin_interaction_gen) =
            drain!(pedersen_builtin_out);
        let (
            pedersen_builtin_narrow_windows_claim,
            pedersen_builtin_narrow_windows_interaction_gen,
        ) = drain!(pedersen_narrow_out);
        let (poseidon_builtin_claim, poseidon_builtin_interaction_gen) =
            drain!(poseidon_builtin_out);
        let (range_check96_builtin_claim, range_check96_builtin_interaction_gen) =
            drain!(range_check96_builtin_out);
        let (range_check_builtin_claim, range_check_builtin_interaction_gen) =
            drain!(range_check_builtin_out);
        let (ec_op_builtin_claim, ec_op_builtin_interaction_gen) = drain!(ec_op_builtin_out);
        let (partial_ec_mul_generic_claim, partial_ec_mul_generic_interaction_gen) =
            drain!(partial_ec_mul_generic_out);
        let (
            pedersen_aggregator_window_bits_18_claim,
            pedersen_aggregator_window_bits_18_interaction_gen,
        ) = drain!(pedersen_aggregator_window_bits_18_out);
        let (partial_ec_mul_window_bits_18_claim, partial_ec_mul_window_bits_18_interaction_gen) =
            drain!(partial_ec_mul_window_bits_18_out);
        let (
            pedersen_points_table_window_bits_18_claim,
            pedersen_points_table_window_bits_18_interaction_gen,
        ) = drain!(pedersen_points_table_window_bits_18_out);
        let (
            pedersen_aggregator_window_bits_9_claim,
            pedersen_aggregator_window_bits_9_interaction_gen,
        ) = drain!(pedersen_aggregator_window_bits_9_out);
        let (partial_ec_mul_window_bits_9_claim, partial_ec_mul_window_bits_9_interaction_gen) =
            drain!(partial_ec_mul_window_bits_9_out);
        let (
            pedersen_points_table_window_bits_9_claim,
            pedersen_points_table_window_bits_9_interaction_gen,
        ) = drain!(pedersen_points_table_window_bits_9_out);
        let (poseidon_aggregator_claim, poseidon_aggregator_interaction_gen) =
            drain!(poseidon_aggregator_out);
        let (
            poseidon_3_partial_rounds_chain_claim,
            poseidon_3_partial_rounds_chain_interaction_gen,
        ) = drain!(poseidon_3_partial_rounds_chain_out);
        let (poseidon_full_round_chain_claim, poseidon_full_round_chain_interaction_gen) =
            drain!(poseidon_full_round_chain_out);
        let (cube_252_claim, cube_252_interaction_gen) = drain!(cube_252_out);
        let (poseidon_round_keys_claim, poseidon_round_keys_interaction_gen) =
            drain!(poseidon_round_keys_out);
        let (range_check_252_width_27_claim, range_check_252_width_27_interaction_gen) =
            drain!(range_check_252_width_27_out);
        let (memory_address_to_id_claim, memory_address_to_id_interaction_gen) = self
            .memory_address_to_id
            .map(|gen| {
                let _wt = tracing::info_span!("wt:memory_address_to_id").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace));
                (claim, interaction_gen)
            })
            .unzip();
        let (memory_id_to_big_claim, memory_id_to_big_interaction_gen) = self
            .memory_id_to_big
            .map(|gen| {
                let _wt = tracing::info_span!("wt:memory_id_to_big").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                const LOG_MAX_BIG_SIZE: u32 = MAX_SEQUENCE_LOG_SIZE;
                // The backend hook: SimdBackend runs the existing host writer;
                // CudaBackend generates the columns on device and merges the
                // rc_9_9 counts BEFORE range_check_9_9 writes its trace below.
                let (big_traces, small_trace, claim, interaction_gen) =
                    <B as MemoryIdToBigWitness>::write_trace(
                        exec_context,
                        gen,
                        self.range_check_9_9.as_ref().unwrap(),
                        LOG_MAX_BIG_SIZE,
                        opt_n_id_to_big_components,
                    );
                for big_trace in big_traces {
                    evals.extend(big_trace);
                }
                evals.extend(small_trace);
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_6_claim, range_check_6_interaction_gen) = self
            .range_check_6
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_6").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_8_claim, range_check_8_interaction_gen) = self
            .range_check_8
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_8").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_11_claim, range_check_11_interaction_gen) = self
            .range_check_11
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_11").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_12_claim, range_check_12_interaction_gen) = self
            .range_check_12
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_12").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_18_claim, range_check_18_interaction_gen) = self
            .range_check_18
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_18").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_20_claim, range_check_20_interaction_gen) = self
            .range_check_20
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_20").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_4_3_claim, range_check_4_3_interaction_gen) = self
            .range_check_4_3
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_4_3").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_4_4_claim, range_check_4_4_interaction_gen) = self
            .range_check_4_4
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_4_4").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_9_9_claim, range_check_9_9_interaction_gen) = self
            .range_check_9_9
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_9_9").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_7_2_5_claim, range_check_7_2_5_interaction_gen) = self
            .range_check_7_2_5
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_7_2_5").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_3_6_6_3_claim, range_check_3_6_6_3_interaction_gen) = self
            .range_check_3_6_6_3
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_3_6_6_3").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_4_4_4_4_claim, range_check_4_4_4_4_interaction_gen) = self
            .range_check_4_4_4_4
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_4_4_4_4").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (range_check_3_3_3_3_3_claim, range_check_3_3_3_3_3_interaction_gen) = self
            .range_check_3_3_3_3_3
            .map(|gen| {
                let _wt = tracing::info_span!("wt:range_check_3_3_3_3_3").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (verify_bitwise_xor_4_claim, verify_bitwise_xor_4_interaction_gen) = self
            .verify_bitwise_xor_4
            .map(|gen| {
                let _wt = tracing::info_span!("wt:verify_bitwise_xor_4").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (verify_bitwise_xor_7_claim, verify_bitwise_xor_7_interaction_gen) = self
            .verify_bitwise_xor_7
            .map(|gen| {
                let _wt = tracing::info_span!("wt:verify_bitwise_xor_7").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (verify_bitwise_xor_8_claim, verify_bitwise_xor_8_interaction_gen) = self
            .verify_bitwise_xor_8
            .map(|gen| {
                let _wt = tracing::info_span!("wt:verify_bitwise_xor_8").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();
        let (verify_bitwise_xor_9_claim, verify_bitwise_xor_9_interaction_gen) = self
            .verify_bitwise_xor_9
            .map(|gen| {
                let _wt = tracing::info_span!("wt:verify_bitwise_xor_9").entered();
                exec_context.record_final_component(&gen, opt_n_id_to_big_components); // final_shape_ledger_codegen
                let (trace, claim, interaction_gen) = gen.write_trace();
                evals.extend(B::from_simd_evals(trace.to_evals()));
                (claim, interaction_gen)
            })
            .unzip();

        let (memory_id_to_big_claim, memory_id_to_small_claim) = memory_id_to_big_claim.unzip();

        // Stage A″: assemble the base trace in canonical column order. With a
        // committer, `evals` now holds only the suffix (non-opcode) columns; join the
        // opcode polys and interpolate the suffix with the SAME tree, so the whole
        // trace is coefficients in canonical order — byte-identical to interpolating
        // the full `evals` at once (`interpolate_columns` is per-column independent).
        // Without a committer, `evals` is the full trace and the caller interpolates
        // it at commit time exactly as before (byte-identical to the pre-A″ flow).
        let base_trace = match committer {
            Some((tx, handle, tree_ptr)) => {
                // Dropping the sender ends the committer's recv loop after the
                // last in-flight batch.
                drop(lane_tx);
                drop(tx);
                let tree = pipeline_twiddles.expect("committer implies pipeline_twiddles");
                let mut batches = handle.join().expect("lane committer thread panicked");
                batches.sort_by_key(|(idx, _)| *idx);
                let mut polys: Vec<CircleCoefficients<B>> =
                    batches.into_iter().flat_map(|(_, b)| b).collect();
                // The sequential tail's columns (memory/rc/xor lanes) are the
                // final canonical suffix — interpolated here, appended last.
                polys.extend(B::interpolate_columns(evals, tree));
                BaseTrace::Polys { polys, tree_ptr }
            }
            None => BaseTrace::Evals(evals),
        };
        (
            base_trace,
            CairoClaim {
                public_data: self.public_data,
                add_opcode: add_opcode_claim,
                add_opcode_small: add_opcode_small_claim,
                add_ap_opcode: add_ap_opcode_claim,
                assert_eq_opcode: assert_eq_opcode_claim,
                assert_eq_opcode_imm: assert_eq_opcode_imm_claim,
                assert_eq_opcode_double_deref: assert_eq_opcode_double_deref_claim,
                blake_compress_opcode: blake_compress_opcode_claim,
                call_opcode_abs: call_opcode_abs_claim,
                call_opcode_rel_imm: call_opcode_rel_imm_claim,
                generic_opcode: generic_opcode_claim,
                jnz_opcode_non_taken: jnz_opcode_non_taken_claim,
                jnz_opcode_taken: jnz_opcode_taken_claim,
                jump_opcode_abs: jump_opcode_abs_claim,
                jump_opcode_double_deref: jump_opcode_double_deref_claim,
                jump_opcode_rel: jump_opcode_rel_claim,
                jump_opcode_rel_imm: jump_opcode_rel_imm_claim,
                mul_opcode: mul_opcode_claim,
                mul_opcode_small: mul_opcode_small_claim,
                qm_31_add_mul_opcode: qm_31_add_mul_opcode_claim,
                ret_opcode: ret_opcode_claim,
                verify_instruction: verify_instruction_claim,
                blake_round: blake_round_claim,
                blake_g: blake_g_claim,
                blake_round_sigma: blake_round_sigma_claim,
                triple_xor_32: triple_xor_32_claim,
                verify_bitwise_xor_12: verify_bitwise_xor_12_claim,
                add_mod_builtin: add_mod_builtin_claim,
                bitwise_builtin: bitwise_builtin_claim,
                mul_mod_builtin: mul_mod_builtin_claim,
                pedersen_builtin: pedersen_builtin_claim,
                pedersen_builtin_narrow_windows: pedersen_builtin_narrow_windows_claim,
                poseidon_builtin: poseidon_builtin_claim,
                range_check96_builtin: range_check96_builtin_claim,
                range_check_builtin: range_check_builtin_claim,
                ec_op_builtin: ec_op_builtin_claim,
                partial_ec_mul_generic: partial_ec_mul_generic_claim,
                pedersen_aggregator_window_bits_18: pedersen_aggregator_window_bits_18_claim,
                partial_ec_mul_window_bits_18: partial_ec_mul_window_bits_18_claim,
                pedersen_points_table_window_bits_18: pedersen_points_table_window_bits_18_claim,
                pedersen_aggregator_window_bits_9: pedersen_aggregator_window_bits_9_claim,
                partial_ec_mul_window_bits_9: partial_ec_mul_window_bits_9_claim,
                pedersen_points_table_window_bits_9: pedersen_points_table_window_bits_9_claim,
                poseidon_aggregator: poseidon_aggregator_claim,
                poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_claim,
                poseidon_full_round_chain: poseidon_full_round_chain_claim,
                cube_252: cube_252_claim,
                poseidon_round_keys: poseidon_round_keys_claim,
                range_check_252_width_27: range_check_252_width_27_claim,
                memory_address_to_id: memory_address_to_id_claim,
                memory_id_to_big: memory_id_to_big_claim,
                memory_id_to_small: memory_id_to_small_claim,
                range_check_6: range_check_6_claim,
                range_check_8: range_check_8_claim,
                range_check_11: range_check_11_claim,
                range_check_12: range_check_12_claim,
                range_check_18: range_check_18_claim,
                range_check_20: range_check_20_claim,
                range_check_4_3: range_check_4_3_claim,
                range_check_4_4: range_check_4_4_claim,
                range_check_9_9: range_check_9_9_claim,
                range_check_7_2_5: range_check_7_2_5_claim,
                range_check_3_6_6_3: range_check_3_6_6_3_claim,
                range_check_4_4_4_4: range_check_4_4_4_4_claim,
                range_check_3_3_3_3_3: range_check_3_3_3_3_3_claim,
                verify_bitwise_xor_4: verify_bitwise_xor_4_claim,
                verify_bitwise_xor_7: verify_bitwise_xor_7_claim,
                verify_bitwise_xor_8: verify_bitwise_xor_8_claim,
                verify_bitwise_xor_9: verify_bitwise_xor_9_claim,
            },
            CairoInteractionClaimGenerator {
                add_opcode: add_opcode_interaction_gen,
                add_opcode_small: add_opcode_small_interaction_gen,
                add_ap_opcode: add_ap_opcode_interaction_gen,
                assert_eq_opcode: assert_eq_opcode_interaction_gen,
                assert_eq_opcode_imm: assert_eq_opcode_imm_interaction_gen,
                assert_eq_opcode_double_deref: assert_eq_opcode_double_deref_interaction_gen,
                blake_compress_opcode: blake_compress_opcode_interaction_gen,
                call_opcode_abs: call_opcode_abs_interaction_gen,
                call_opcode_rel_imm: call_opcode_rel_imm_interaction_gen,
                generic_opcode: generic_opcode_interaction_gen,
                jnz_opcode_non_taken: jnz_opcode_non_taken_interaction_gen,
                jnz_opcode_taken: jnz_opcode_taken_interaction_gen,
                jump_opcode_abs: jump_opcode_abs_interaction_gen,
                jump_opcode_double_deref: jump_opcode_double_deref_interaction_gen,
                jump_opcode_rel: jump_opcode_rel_interaction_gen,
                jump_opcode_rel_imm: jump_opcode_rel_imm_interaction_gen,
                mul_opcode: mul_opcode_interaction_gen,
                mul_opcode_small: mul_opcode_small_interaction_gen,
                qm_31_add_mul_opcode: qm_31_add_mul_opcode_interaction_gen,
                ret_opcode: ret_opcode_interaction_gen,
                verify_instruction: verify_instruction_interaction_gen,
                blake_round: blake_round_interaction_gen,
                blake_g: blake_g_interaction_gen,
                blake_round_sigma: blake_round_sigma_interaction_gen,
                triple_xor_32: triple_xor_32_interaction_gen,
                verify_bitwise_xor_12: verify_bitwise_xor_12_interaction_gen,
                add_mod_builtin: add_mod_builtin_interaction_gen,
                bitwise_builtin: bitwise_builtin_interaction_gen,
                mul_mod_builtin: mul_mod_builtin_interaction_gen,
                pedersen_builtin: pedersen_builtin_interaction_gen,
                pedersen_builtin_narrow_windows: pedersen_builtin_narrow_windows_interaction_gen,
                poseidon_builtin: poseidon_builtin_interaction_gen,
                range_check96_builtin: range_check96_builtin_interaction_gen,
                range_check_builtin: range_check_builtin_interaction_gen,
                ec_op_builtin: ec_op_builtin_interaction_gen,
                partial_ec_mul_generic: partial_ec_mul_generic_interaction_gen,
                pedersen_aggregator_window_bits_18:
                    pedersen_aggregator_window_bits_18_interaction_gen,
                partial_ec_mul_window_bits_18: partial_ec_mul_window_bits_18_interaction_gen,
                pedersen_points_table_window_bits_18:
                    pedersen_points_table_window_bits_18_interaction_gen,
                pedersen_aggregator_window_bits_9:
                    pedersen_aggregator_window_bits_9_interaction_gen,
                partial_ec_mul_window_bits_9: partial_ec_mul_window_bits_9_interaction_gen,
                pedersen_points_table_window_bits_9:
                    pedersen_points_table_window_bits_9_interaction_gen,
                poseidon_aggregator: poseidon_aggregator_interaction_gen,
                poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_interaction_gen,
                poseidon_full_round_chain: poseidon_full_round_chain_interaction_gen,
                cube_252: cube_252_interaction_gen,
                poseidon_round_keys: poseidon_round_keys_interaction_gen,
                range_check_252_width_27: range_check_252_width_27_interaction_gen,
                memory_address_to_id: memory_address_to_id_interaction_gen,
                memory_id_to_big: memory_id_to_big_interaction_gen,
                range_check_6: range_check_6_interaction_gen,
                range_check_8: range_check_8_interaction_gen,
                range_check_11: range_check_11_interaction_gen,
                range_check_12: range_check_12_interaction_gen,
                range_check_18: range_check_18_interaction_gen,
                range_check_20: range_check_20_interaction_gen,
                range_check_4_3: range_check_4_3_interaction_gen,
                range_check_4_4: range_check_4_4_interaction_gen,
                range_check_9_9: range_check_9_9_interaction_gen,
                range_check_7_2_5: range_check_7_2_5_interaction_gen,
                range_check_3_6_6_3: range_check_3_6_6_3_interaction_gen,
                range_check_4_4_4_4: range_check_4_4_4_4_interaction_gen,
                range_check_3_3_3_3_3: range_check_3_3_3_3_3_interaction_gen,
                verify_bitwise_xor_4: verify_bitwise_xor_4_interaction_gen,
                verify_bitwise_xor_7: verify_bitwise_xor_7_interaction_gen,
                verify_bitwise_xor_8: verify_bitwise_xor_8_interaction_gen,
                verify_bitwise_xor_9: verify_bitwise_xor_9_interaction_gen,
            },
        )
    }
}

pub struct CairoInteractionClaimGenerator<B: MemoryIdToBigWitness + BlakeGWitness> {
    pub add_opcode: Option<add_opcode::InteractionClaimGenerator>,
    pub add_opcode_small: Option<add_opcode_small::InteractionClaimGenerator>,
    pub add_ap_opcode: Option<add_ap_opcode::InteractionClaimGenerator>,
    pub assert_eq_opcode: Option<assert_eq_opcode::InteractionClaimGenerator>,
    pub assert_eq_opcode_imm: Option<assert_eq_opcode_imm::InteractionClaimGenerator>,
    pub assert_eq_opcode_double_deref:
        Option<assert_eq_opcode_double_deref::InteractionClaimGenerator>,
    pub blake_compress_opcode: Option<blake_compress_opcode::InteractionClaimGenerator>,
    pub call_opcode_abs: Option<call_opcode_abs::InteractionClaimGenerator>,
    pub call_opcode_rel_imm: Option<call_opcode_rel_imm::InteractionClaimGenerator>,
    pub generic_opcode: Option<generic_opcode::InteractionClaimGenerator>,
    pub jnz_opcode_non_taken: Option<jnz_opcode_non_taken::InteractionClaimGenerator>,
    pub jnz_opcode_taken: Option<jnz_opcode_taken::InteractionClaimGenerator>,
    pub jump_opcode_abs: Option<jump_opcode_abs::InteractionClaimGenerator>,
    pub jump_opcode_double_deref: Option<jump_opcode_double_deref::InteractionClaimGenerator>,
    pub jump_opcode_rel: Option<jump_opcode_rel::InteractionClaimGenerator>,
    pub jump_opcode_rel_imm: Option<jump_opcode_rel_imm::InteractionClaimGenerator>,
    pub mul_opcode: Option<mul_opcode::InteractionClaimGenerator>,
    pub mul_opcode_small: Option<mul_opcode_small::InteractionClaimGenerator>,
    pub qm_31_add_mul_opcode: Option<qm_31_add_mul_opcode::InteractionClaimGenerator>,
    pub ret_opcode: Option<ret_opcode::InteractionClaimGenerator>,
    pub verify_instruction: Option<verify_instruction::InteractionClaimGenerator>,
    pub blake_round: Option<blake_round::InteractionClaimGenerator>,
    pub blake_g: Option<<B as BlakeGWitness>::InteractionGen>,
    pub blake_round_sigma: Option<blake_round_sigma::InteractionClaimGenerator>,
    pub triple_xor_32: Option<triple_xor_32::InteractionClaimGenerator>,
    pub verify_bitwise_xor_12: Option<verify_bitwise_xor_12::InteractionClaimGenerator>,
    pub add_mod_builtin: Option<add_mod_builtin::InteractionClaimGenerator>,
    pub bitwise_builtin: Option<bitwise_builtin::InteractionClaimGenerator>,
    pub mul_mod_builtin: Option<mul_mod_builtin::InteractionClaimGenerator>,
    pub pedersen_builtin: Option<pedersen_builtin::InteractionClaimGenerator>,
    pub pedersen_builtin_narrow_windows:
        Option<pedersen_builtin_narrow_windows::InteractionClaimGenerator>,
    pub poseidon_builtin: Option<poseidon_builtin::InteractionClaimGenerator>,
    pub range_check96_builtin: Option<range_check96_builtin::InteractionClaimGenerator>,
    pub range_check_builtin: Option<range_check_builtin::InteractionClaimGenerator>,
    pub ec_op_builtin: Option<ec_op_builtin::InteractionClaimGenerator>,
    pub partial_ec_mul_generic: Option<partial_ec_mul_generic::InteractionClaimGenerator>,
    pub pedersen_aggregator_window_bits_18:
        Option<pedersen_aggregator_window_bits_18::InteractionClaimGenerator>,
    pub partial_ec_mul_window_bits_18:
        Option<partial_ec_mul_window_bits_18::InteractionClaimGenerator>,
    pub pedersen_points_table_window_bits_18:
        Option<pedersen_points_table_window_bits_18::InteractionClaimGenerator>,
    pub pedersen_aggregator_window_bits_9:
        Option<pedersen_aggregator_window_bits_9::InteractionClaimGenerator>,
    pub partial_ec_mul_window_bits_9:
        Option<partial_ec_mul_window_bits_9::InteractionClaimGenerator>,
    pub pedersen_points_table_window_bits_9:
        Option<pedersen_points_table_window_bits_9::InteractionClaimGenerator>,
    pub poseidon_aggregator: Option<poseidon_aggregator::InteractionClaimGenerator>,
    pub poseidon_3_partial_rounds_chain:
        Option<poseidon_3_partial_rounds_chain::InteractionClaimGenerator>,
    pub poseidon_full_round_chain: Option<poseidon_full_round_chain::InteractionClaimGenerator>,
    pub cube_252: Option<cube_252::InteractionClaimGenerator>,
    pub poseidon_round_keys: Option<poseidon_round_keys::InteractionClaimGenerator>,
    pub range_check_252_width_27: Option<range_check_252_width_27::InteractionClaimGenerator>,
    pub memory_address_to_id: Option<memory_address_to_id::InteractionClaimGenerator>,
    pub memory_id_to_big: Option<<B as MemoryIdToBigWitness>::InteractionGen>,
    pub range_check_6: Option<range_check_6::InteractionClaimGenerator>,
    pub range_check_8: Option<range_check_8::InteractionClaimGenerator>,
    pub range_check_11: Option<range_check_11::InteractionClaimGenerator>,
    pub range_check_12: Option<range_check_12::InteractionClaimGenerator>,
    pub range_check_18: Option<range_check_18::InteractionClaimGenerator>,
    pub range_check_20: Option<range_check_20::InteractionClaimGenerator>,
    pub range_check_4_3: Option<range_check_4_3::InteractionClaimGenerator>,
    pub range_check_4_4: Option<range_check_4_4::InteractionClaimGenerator>,
    pub range_check_9_9: Option<range_check_9_9::InteractionClaimGenerator>,
    pub range_check_7_2_5: Option<range_check_7_2_5::InteractionClaimGenerator>,
    pub range_check_3_6_6_3: Option<range_check_3_6_6_3::InteractionClaimGenerator>,
    pub range_check_4_4_4_4: Option<range_check_4_4_4_4::InteractionClaimGenerator>,
    pub range_check_3_3_3_3_3: Option<range_check_3_3_3_3_3::InteractionClaimGenerator>,
    pub verify_bitwise_xor_4: Option<verify_bitwise_xor_4::InteractionClaimGenerator>,
    pub verify_bitwise_xor_7: Option<verify_bitwise_xor_7::InteractionClaimGenerator>,
    pub verify_bitwise_xor_8: Option<verify_bitwise_xor_8::InteractionClaimGenerator>,
    pub verify_bitwise_xor_9: Option<verify_bitwise_xor_9::InteractionClaimGenerator>,
}

// === BEGIN relation_lookup_source_codegen ===
impl<B: MemoryIdToBigWitness + BlakeGWitness> CairoInteractionClaimGenerator<B> {
    /// Consumes every active interaction state into typed relation sources.
    /// This is generated from the aggregate fields; component additions cannot
    /// silently bypass the GPU-native relation layer.
    pub fn into_relation_lookup_sources(
        self,
        exec_context: &WitnessExecContext,
    ) -> Result<
        crate::witness::relation_sources::CairoRelationSourceSet,
        crate::witness::relation_sources::RelationSourceError,
    > {
        use crate::witness::relation_sources::RelationLookupSourceExport;

        let mut sources = Vec::new();
        if let Some(gen) = self.add_opcode {
            sources.extend(gen.export_relation_lookup_sources("add_opcode", exec_context)?);
        }
        if let Some(gen) = self.add_opcode_small {
            sources.extend(gen.export_relation_lookup_sources("add_opcode_small", exec_context)?);
        }
        if let Some(gen) = self.add_ap_opcode {
            sources.extend(gen.export_relation_lookup_sources("add_ap_opcode", exec_context)?);
        }
        if let Some(gen) = self.assert_eq_opcode {
            sources.extend(gen.export_relation_lookup_sources("assert_eq_opcode", exec_context)?);
        }
        if let Some(gen) = self.assert_eq_opcode_imm {
            sources
                .extend(gen.export_relation_lookup_sources("assert_eq_opcode_imm", exec_context)?);
        }
        if let Some(gen) = self.assert_eq_opcode_double_deref {
            sources.extend(
                gen.export_relation_lookup_sources("assert_eq_opcode_double_deref", exec_context)?,
            );
        }
        if let Some(gen) = self.blake_compress_opcode {
            sources
                .extend(gen.export_relation_lookup_sources("blake_compress_opcode", exec_context)?);
        }
        if let Some(gen) = self.call_opcode_abs {
            sources.extend(gen.export_relation_lookup_sources("call_opcode_abs", exec_context)?);
        }
        if let Some(gen) = self.call_opcode_rel_imm {
            sources
                .extend(gen.export_relation_lookup_sources("call_opcode_rel_imm", exec_context)?);
        }
        if let Some(gen) = self.generic_opcode {
            sources.extend(gen.export_relation_lookup_sources("generic_opcode", exec_context)?);
        }
        if let Some(gen) = self.jnz_opcode_non_taken {
            sources
                .extend(gen.export_relation_lookup_sources("jnz_opcode_non_taken", exec_context)?);
        }
        if let Some(gen) = self.jnz_opcode_taken {
            sources.extend(gen.export_relation_lookup_sources("jnz_opcode_taken", exec_context)?);
        }
        if let Some(gen) = self.jump_opcode_abs {
            sources.extend(gen.export_relation_lookup_sources("jump_opcode_abs", exec_context)?);
        }
        if let Some(gen) = self.jump_opcode_double_deref {
            sources.extend(
                gen.export_relation_lookup_sources("jump_opcode_double_deref", exec_context)?,
            );
        }
        if let Some(gen) = self.jump_opcode_rel {
            sources.extend(gen.export_relation_lookup_sources("jump_opcode_rel", exec_context)?);
        }
        if let Some(gen) = self.jump_opcode_rel_imm {
            sources
                .extend(gen.export_relation_lookup_sources("jump_opcode_rel_imm", exec_context)?);
        }
        if let Some(gen) = self.mul_opcode {
            sources.extend(gen.export_relation_lookup_sources("mul_opcode", exec_context)?);
        }
        if let Some(gen) = self.mul_opcode_small {
            sources.extend(gen.export_relation_lookup_sources("mul_opcode_small", exec_context)?);
        }
        if let Some(gen) = self.qm_31_add_mul_opcode {
            sources
                .extend(gen.export_relation_lookup_sources("qm_31_add_mul_opcode", exec_context)?);
        }
        if let Some(gen) = self.ret_opcode {
            sources.extend(gen.export_relation_lookup_sources("ret_opcode", exec_context)?);
        }
        if let Some(gen) = self.verify_instruction {
            sources.extend(gen.export_relation_lookup_sources("verify_instruction", exec_context)?);
        }
        if let Some(gen) = self.blake_round {
            sources.extend(gen.export_relation_lookup_sources("blake_round", exec_context)?);
        }
        if let Some(gen) = self.blake_g {
            sources.extend(gen.export_relation_lookup_sources("blake_g", exec_context)?);
        }
        if let Some(gen) = self.blake_round_sigma {
            sources.extend(gen.export_relation_lookup_sources("blake_round_sigma", exec_context)?);
        }
        if let Some(gen) = self.triple_xor_32 {
            sources.extend(gen.export_relation_lookup_sources("triple_xor_32", exec_context)?);
        }
        if let Some(gen) = self.verify_bitwise_xor_12 {
            sources
                .extend(gen.export_relation_lookup_sources("verify_bitwise_xor_12", exec_context)?);
        }
        if let Some(gen) = self.add_mod_builtin {
            sources.extend(gen.export_relation_lookup_sources("add_mod_builtin", exec_context)?);
        }
        if let Some(gen) = self.bitwise_builtin {
            sources.extend(gen.export_relation_lookup_sources("bitwise_builtin", exec_context)?);
        }
        if let Some(gen) = self.mul_mod_builtin {
            sources.extend(gen.export_relation_lookup_sources("mul_mod_builtin", exec_context)?);
        }
        if let Some(gen) = self.pedersen_builtin {
            sources.extend(gen.export_relation_lookup_sources("pedersen_builtin", exec_context)?);
        }
        if let Some(gen) = self.pedersen_builtin_narrow_windows {
            sources.extend(
                gen.export_relation_lookup_sources(
                    "pedersen_builtin_narrow_windows",
                    exec_context,
                )?,
            );
        }
        if let Some(gen) = self.poseidon_builtin {
            sources.extend(gen.export_relation_lookup_sources("poseidon_builtin", exec_context)?);
        }
        if let Some(gen) = self.range_check96_builtin {
            sources
                .extend(gen.export_relation_lookup_sources("range_check96_builtin", exec_context)?);
        }
        if let Some(gen) = self.range_check_builtin {
            sources
                .extend(gen.export_relation_lookup_sources("range_check_builtin", exec_context)?);
        }
        if let Some(gen) = self.ec_op_builtin {
            sources.extend(gen.export_relation_lookup_sources("ec_op_builtin", exec_context)?);
        }
        if let Some(gen) = self.partial_ec_mul_generic {
            sources.extend(
                gen.export_relation_lookup_sources("partial_ec_mul_generic", exec_context)?,
            );
        }
        if let Some(gen) = self.pedersen_aggregator_window_bits_18 {
            sources.extend(gen.export_relation_lookup_sources(
                "pedersen_aggregator_window_bits_18",
                exec_context,
            )?);
        }
        if let Some(gen) = self.partial_ec_mul_window_bits_18 {
            sources.extend(
                gen.export_relation_lookup_sources("partial_ec_mul_window_bits_18", exec_context)?,
            );
        }
        if let Some(gen) = self.pedersen_points_table_window_bits_18 {
            sources.extend(gen.export_relation_lookup_sources(
                "pedersen_points_table_window_bits_18",
                exec_context,
            )?);
        }
        if let Some(gen) = self.pedersen_aggregator_window_bits_9 {
            sources.extend(gen.export_relation_lookup_sources(
                "pedersen_aggregator_window_bits_9",
                exec_context,
            )?);
        }
        if let Some(gen) = self.partial_ec_mul_window_bits_9 {
            sources.extend(
                gen.export_relation_lookup_sources("partial_ec_mul_window_bits_9", exec_context)?,
            );
        }
        if let Some(gen) = self.pedersen_points_table_window_bits_9 {
            sources.extend(gen.export_relation_lookup_sources(
                "pedersen_points_table_window_bits_9",
                exec_context,
            )?);
        }
        if let Some(gen) = self.poseidon_aggregator {
            sources
                .extend(gen.export_relation_lookup_sources("poseidon_aggregator", exec_context)?);
        }
        if let Some(gen) = self.poseidon_3_partial_rounds_chain {
            sources.extend(
                gen.export_relation_lookup_sources(
                    "poseidon_3_partial_rounds_chain",
                    exec_context,
                )?,
            );
        }
        if let Some(gen) = self.poseidon_full_round_chain {
            sources.extend(
                gen.export_relation_lookup_sources("poseidon_full_round_chain", exec_context)?,
            );
        }
        if let Some(gen) = self.cube_252 {
            sources.extend(gen.export_relation_lookup_sources("cube_252", exec_context)?);
        }
        if let Some(gen) = self.poseidon_round_keys {
            sources
                .extend(gen.export_relation_lookup_sources("poseidon_round_keys", exec_context)?);
        }
        if let Some(gen) = self.range_check_252_width_27 {
            sources.extend(
                gen.export_relation_lookup_sources("range_check_252_width_27", exec_context)?,
            );
        }
        if let Some(gen) = self.memory_address_to_id {
            sources
                .extend(gen.export_relation_lookup_sources("memory_address_to_id", exec_context)?);
        }
        if let Some(gen) = self.memory_id_to_big {
            sources.extend(gen.export_relation_lookup_sources("memory_id_to_big", exec_context)?);
        }
        if let Some(gen) = self.range_check_6 {
            sources.extend(gen.export_relation_lookup_sources("range_check_6", exec_context)?);
        }
        if let Some(gen) = self.range_check_8 {
            sources.extend(gen.export_relation_lookup_sources("range_check_8", exec_context)?);
        }
        if let Some(gen) = self.range_check_11 {
            sources.extend(gen.export_relation_lookup_sources("range_check_11", exec_context)?);
        }
        if let Some(gen) = self.range_check_12 {
            sources.extend(gen.export_relation_lookup_sources("range_check_12", exec_context)?);
        }
        if let Some(gen) = self.range_check_18 {
            sources.extend(gen.export_relation_lookup_sources("range_check_18", exec_context)?);
        }
        if let Some(gen) = self.range_check_20 {
            sources.extend(gen.export_relation_lookup_sources("range_check_20", exec_context)?);
        }
        if let Some(gen) = self.range_check_4_3 {
            sources.extend(gen.export_relation_lookup_sources("range_check_4_3", exec_context)?);
        }
        if let Some(gen) = self.range_check_4_4 {
            sources.extend(gen.export_relation_lookup_sources("range_check_4_4", exec_context)?);
        }
        if let Some(gen) = self.range_check_9_9 {
            sources.extend(gen.export_relation_lookup_sources("range_check_9_9", exec_context)?);
        }
        if let Some(gen) = self.range_check_7_2_5 {
            sources.extend(gen.export_relation_lookup_sources("range_check_7_2_5", exec_context)?);
        }
        if let Some(gen) = self.range_check_3_6_6_3 {
            sources
                .extend(gen.export_relation_lookup_sources("range_check_3_6_6_3", exec_context)?);
        }
        if let Some(gen) = self.range_check_4_4_4_4 {
            sources
                .extend(gen.export_relation_lookup_sources("range_check_4_4_4_4", exec_context)?);
        }
        if let Some(gen) = self.range_check_3_3_3_3_3 {
            sources
                .extend(gen.export_relation_lookup_sources("range_check_3_3_3_3_3", exec_context)?);
        }
        if let Some(gen) = self.verify_bitwise_xor_4 {
            sources
                .extend(gen.export_relation_lookup_sources("verify_bitwise_xor_4", exec_context)?);
        }
        if let Some(gen) = self.verify_bitwise_xor_7 {
            sources
                .extend(gen.export_relation_lookup_sources("verify_bitwise_xor_7", exec_context)?);
        }
        if let Some(gen) = self.verify_bitwise_xor_8 {
            sources
                .extend(gen.export_relation_lookup_sources("verify_bitwise_xor_8", exec_context)?);
        }
        if let Some(gen) = self.verify_bitwise_xor_9 {
            sources
                .extend(gen.export_relation_lookup_sources("verify_bitwise_xor_9", exec_context)?);
        }
        exec_context.assert_interaction_drained();
        crate::witness::relation_sources::CairoRelationSourceSet::new(sources)
    }
}
// === END relation_lookup_source_codegen ===

impl<B: MemoryIdToBigWitness + BlakeGWitness + OpcodeJitBackend> CairoInteractionClaimGenerator<B> {
    /// Writes the raw interaction fractions on the host (parallel across
    /// components), then finalizes each component's logup trace on `B` — the
    /// device, for GPU backends — in the same fixed component order as before.
    /// Claims are constructed from the finalized sums, so the Fiat-Shamir
    /// transcript is unchanged. `memory_id_to_big` goes through the
    /// [`MemoryIdToBigWitness`] hook: on the device path its denominators are
    /// computed from the device-resident limb columns (witness-on-GPU P1).
    pub fn write_interaction_trace(
        self,
        exec_context: &WitnessExecContext,
        common_lookup_elements: &CommonLookupElements,
    ) -> (
        Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        CairoInteractionClaim,
    ) {
        let mut evals = Vec::new();
        let mut add_opcode_result = None;
        let mut add_opcode_small_result = None;
        let mut add_ap_opcode_result = None;
        let mut assert_eq_opcode_result = None;
        let mut assert_eq_opcode_imm_result = None;
        let mut assert_eq_opcode_double_deref_result = None;
        let mut blake_compress_opcode_result = None;
        let mut call_opcode_abs_result = None;
        let mut call_opcode_rel_imm_result = None;
        let mut generic_opcode_result = None;
        let mut jnz_opcode_non_taken_result = None;
        let mut jnz_opcode_taken_result = None;
        let mut jump_opcode_abs_result = None;
        let mut jump_opcode_double_deref_result = None;
        let mut jump_opcode_rel_result = None;
        let mut jump_opcode_rel_imm_result = None;
        let mut mul_opcode_result = None;
        let mut mul_opcode_small_result = None;
        let mut qm_31_add_mul_opcode_result = None;
        let mut ret_opcode_result = None;
        let mut verify_instruction_result = None;
        let mut blake_round_result = None;
        let mut blake_g_result = None;
        let mut blake_round_sigma_result = None;
        let mut triple_xor_32_result = None;
        let mut verify_bitwise_xor_12_result = None;
        let mut add_mod_builtin_result = None;
        let mut bitwise_builtin_result = None;
        let mut mul_mod_builtin_result = None;
        let mut pedersen_builtin_result = None;
        let mut pedersen_builtin_narrow_windows_result = None;
        let mut poseidon_builtin_result = None;
        let mut range_check96_builtin_result = None;
        let mut range_check_builtin_result = None;
        let mut ec_op_builtin_result = None;
        let mut partial_ec_mul_generic_result = None;
        let mut pedersen_aggregator_window_bits_18_result = None;
        let mut partial_ec_mul_window_bits_18_result = None;
        let mut pedersen_points_table_window_bits_18_result = None;
        let mut pedersen_aggregator_window_bits_9_result = None;
        let mut partial_ec_mul_window_bits_9_result = None;
        let mut pedersen_points_table_window_bits_9_result = None;
        let mut poseidon_aggregator_result = None;
        let mut poseidon_3_partial_rounds_chain_result = None;
        let mut poseidon_full_round_chain_result = None;
        let mut cube_252_result = None;
        let mut poseidon_round_keys_result = None;
        let mut range_check_252_width_27_result = None;
        let mut memory_address_to_id_result = None;
        let mut memory_id_to_big_result = None;
        let mut range_check_6_result = None;
        let mut range_check_8_result = None;
        let mut range_check_11_result = None;
        let mut range_check_12_result = None;
        let mut range_check_18_result = None;
        let mut range_check_20_result = None;
        let mut range_check_4_3_result = None;
        let mut range_check_4_4_result = None;
        let mut range_check_9_9_result = None;
        let mut range_check_7_2_5_result = None;
        let mut range_check_3_6_6_3_result = None;
        let mut range_check_4_4_4_4_result = None;
        let mut range_check_3_3_3_3_3_result = None;
        let mut verify_bitwise_xor_4_result = None;
        let mut verify_bitwise_xor_7_result = None;
        let mut verify_bitwise_xor_8_result = None;
        let mut verify_bitwise_xor_9_result = None;

        scope(|s| {
            if let Some(gen) = self.add_opcode {
                if <B as OpcodeJitBackend>::device_interaction_pending::<AddOpcodeLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        add_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.add_opcode_small {
                if <B as OpcodeJitBackend>::device_interaction_pending::<AddOpcodeSmallLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        add_opcode_small_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.add_ap_opcode {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<AddApOpcodeLane>(
                    exec_context,
                ) {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        add_ap_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.assert_eq_opcode {
                if <B as OpcodeJitBackend>::device_interaction_pending::<AssertEqOpcodeLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        assert_eq_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.assert_eq_opcode_imm {
                if <B as OpcodeJitBackend>::device_interaction_pending::<AssertEqOpcodeImmLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        assert_eq_opcode_imm_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.assert_eq_opcode_double_deref {
                if <B as OpcodeJitBackend>::device_interaction_pending::<
                    AssertEqOpcodeDoubleDerefLane,
                >(exec_context)
                {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        assert_eq_opcode_double_deref_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.blake_compress_opcode {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<
                    BlakeCompressOpcodeLane,
                >(exec_context)
                {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        blake_compress_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.call_opcode_abs {
                if <B as OpcodeJitBackend>::device_interaction_pending::<CallOpcodeAbsLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        call_opcode_abs_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.call_opcode_rel_imm {
                if <B as OpcodeJitBackend>::device_interaction_pending::<CallOpcodeRelImmLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        call_opcode_rel_imm_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.generic_opcode {
                s.spawn(|_| {
                    generic_opcode_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.jnz_opcode_non_taken {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JnzOpcodeNonTakenLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jnz_opcode_non_taken_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.jnz_opcode_taken {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JnzOpcodeTakenLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jnz_opcode_taken_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.jump_opcode_abs {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JumpOpcodeAbsLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jump_opcode_abs_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.jump_opcode_double_deref {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JumpOpcodeDoubleDerefLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jump_opcode_double_deref_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.jump_opcode_rel {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JumpOpcodeRelLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jump_opcode_rel_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.jump_opcode_rel_imm {
                if <B as OpcodeJitBackend>::device_interaction_pending::<JumpOpcodeRelImmLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        jump_opcode_rel_imm_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.mul_opcode {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<MulOpcodeLane>(
                    exec_context,
                ) {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        mul_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.mul_opcode_small {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<MulOpcodeSmallLane>(
                    exec_context,
                ) {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        mul_opcode_small_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.qm_31_add_mul_opcode {
                s.spawn(|_| {
                    qm_31_add_mul_opcode_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.ret_opcode {
                if <B as OpcodeJitBackend>::device_interaction_pending::<RetOpcodeLane>(
                    exec_context,
                ) {
                    drop(gen); // device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        ret_opcode_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.verify_instruction {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<
                    VerifyInstructionLane,
                >(exec_context)
                {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        verify_instruction_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.blake_round {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<BlakeRoundLane>(
                    exec_context,
                ) {
                    drop(gen); // §6a: the device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        blake_round_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.blake_g {
                s.spawn(|_| {
                    blake_g_result = Some(<B as BlakeGWitness>::write_interaction(
                        gen,
                        common_lookup_elements,
                    ));
                });
            }
            if let Some(gen) = self.blake_round_sigma {
                s.spawn(|_| {
                    blake_round_sigma_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.triple_xor_32 {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<TripleXor32Lane>(
                    exec_context,
                ) {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        triple_xor_32_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.verify_bitwise_xor_12 {
                s.spawn(|_| {
                    verify_bitwise_xor_12_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.add_mod_builtin {
                s.spawn(|_| {
                    add_mod_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.bitwise_builtin {
                s.spawn(|_| {
                    bitwise_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.mul_mod_builtin {
                s.spawn(|_| {
                    mul_mod_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.pedersen_builtin {
                s.spawn(|_| {
                    pedersen_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.pedersen_builtin_narrow_windows {
                s.spawn(|_| {
                    pedersen_builtin_narrow_windows_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.poseidon_builtin {
                s.spawn(|_| {
                    poseidon_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check96_builtin {
                s.spawn(|_| {
                    range_check96_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_builtin {
                s.spawn(|_| {
                    range_check_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.ec_op_builtin {
                s.spawn(|_| {
                    ec_op_builtin_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.partial_ec_mul_generic {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<
                    PartialEcMulGenericLane,
                >(exec_context)
                {
                    drop(gen); // §6a: the device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        partial_ec_mul_generic_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.pedersen_aggregator_window_bits_18 {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<
                    PedersenAggregatorW18Lane,
                >(exec_context)
                {
                    drop(gen); // §6a: the device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        pedersen_aggregator_window_bits_18_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.partial_ec_mul_window_bits_18 {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<PartialEcMulW18Lane>(
                    exec_context,
                ) {
                    drop(gen); // §6a: the device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        partial_ec_mul_window_bits_18_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.pedersen_points_table_window_bits_18 {
                s.spawn(|_| {
                    pedersen_points_table_window_bits_18_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.pedersen_aggregator_window_bits_9 {
                s.spawn(|_| {
                    pedersen_aggregator_window_bits_9_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.partial_ec_mul_window_bits_9 {
                s.spawn(|_| {
                    partial_ec_mul_window_bits_9_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.pedersen_points_table_window_bits_9 {
                s.spawn(|_| {
                    pedersen_points_table_window_bits_9_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.poseidon_aggregator {
                s.spawn(|_| {
                    poseidon_aggregator_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.poseidon_3_partial_rounds_chain {
                s.spawn(|_| {
                    poseidon_3_partial_rounds_chain_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.poseidon_full_round_chain {
                s.spawn(|_| {
                    poseidon_full_round_chain_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.cube_252 {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<Cube252Lane>(
                    exec_context,
                ) {
                    drop(gen); // §6a: the device path owns this component's interaction
                } else {
                    s.spawn(|_| {
                        cube_252_result = Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.poseidon_round_keys {
                s.spawn(|_| {
                    poseidon_round_keys_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_252_width_27 {
                if <B as OpcodeJitBackend>::builtin_device_interaction_pending::<
                    RangeCheck252Width27Lane,
                >(exec_context)
                {
                    drop(gen);
                } else {
                    s.spawn(|_| {
                        range_check_252_width_27_result =
                            Some(gen.write_interaction_trace(common_lookup_elements));
                    });
                }
            }
            if let Some(gen) = self.memory_address_to_id {
                s.spawn(|_| {
                    memory_address_to_id_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.memory_id_to_big {
                s.spawn(|_| {
                    memory_id_to_big_result = Some(<B as MemoryIdToBigWitness>::write_interaction(
                        gen,
                        common_lookup_elements,
                    ));
                });
            }
            if let Some(gen) = self.range_check_6 {
                s.spawn(|_| {
                    range_check_6_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_8 {
                s.spawn(|_| {
                    range_check_8_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_11 {
                s.spawn(|_| {
                    range_check_11_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_12 {
                s.spawn(|_| {
                    range_check_12_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_18 {
                s.spawn(|_| {
                    range_check_18_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_20 {
                s.spawn(|_| {
                    range_check_20_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_4_3 {
                s.spawn(|_| {
                    range_check_4_3_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_4_4 {
                s.spawn(|_| {
                    range_check_4_4_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_9_9 {
                s.spawn(|_| {
                    range_check_9_9_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_7_2_5 {
                s.spawn(|_| {
                    range_check_7_2_5_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_3_6_6_3 {
                s.spawn(|_| {
                    range_check_3_6_6_3_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_4_4_4_4 {
                s.spawn(|_| {
                    range_check_4_4_4_4_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.range_check_3_3_3_3_3 {
                s.spawn(|_| {
                    range_check_3_3_3_3_3_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.verify_bitwise_xor_4 {
                s.spawn(|_| {
                    verify_bitwise_xor_4_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.verify_bitwise_xor_7 {
                s.spawn(|_| {
                    verify_bitwise_xor_7_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.verify_bitwise_xor_8 {
                s.spawn(|_| {
                    verify_bitwise_xor_8_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
            if let Some(gen) = self.verify_bitwise_xor_9 {
                s.spawn(|_| {
                    verify_bitwise_xor_9_result =
                        Some(gen.write_interaction_trace(common_lookup_elements));
                });
            }
        });

        let add_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<AddOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::add_opcode::InteractionClaim { claimed_sum })
        } else {
            add_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let add_opcode_small_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<AddOpcodeSmallLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::add_opcode_small::InteractionClaim { claimed_sum })
        } else {
            add_opcode_small_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let add_ap_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<AddApOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::add_ap_opcode::InteractionClaim { claimed_sum })
        } else {
            add_ap_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let assert_eq_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<AssertEqOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::assert_eq_opcode::InteractionClaim { claimed_sum })
        } else {
            assert_eq_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let assert_eq_opcode_imm_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<AssertEqOpcodeImmLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::assert_eq_opcode_imm::InteractionClaim { claimed_sum })
        } else {
            assert_eq_opcode_imm_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let assert_eq_opcode_double_deref_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<AssertEqOpcodeDoubleDerefLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(
                cairo_air::components::assert_eq_opcode_double_deref::InteractionClaim {
                    claimed_sum,
                },
            )
        } else {
            assert_eq_opcode_double_deref_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let blake_compress_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<BlakeCompressOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::blake_compress_opcode::InteractionClaim { claimed_sum })
        } else {
            blake_compress_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let call_opcode_abs_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<CallOpcodeAbsLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::call_opcode_abs::InteractionClaim { claimed_sum })
        } else {
            call_opcode_abs_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let call_opcode_rel_imm_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<CallOpcodeRelImmLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::call_opcode_rel_imm::InteractionClaim { claimed_sum })
        } else {
            call_opcode_rel_imm_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let generic_opcode_interaction_claim = generic_opcode_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let jnz_opcode_non_taken_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JnzOpcodeNonTakenLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jnz_opcode_non_taken::InteractionClaim { claimed_sum })
        } else {
            jnz_opcode_non_taken_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let jnz_opcode_taken_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JnzOpcodeTakenLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jnz_opcode_taken::InteractionClaim { claimed_sum })
        } else {
            jnz_opcode_taken_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let jump_opcode_abs_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JumpOpcodeAbsLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jump_opcode_abs::InteractionClaim { claimed_sum })
        } else {
            jump_opcode_abs_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let jump_opcode_double_deref_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JumpOpcodeDoubleDerefLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jump_opcode_double_deref::InteractionClaim { claimed_sum })
        } else {
            jump_opcode_double_deref_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let jump_opcode_rel_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JumpOpcodeRelLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jump_opcode_rel::InteractionClaim { claimed_sum })
        } else {
            jump_opcode_rel_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let jump_opcode_rel_imm_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<JumpOpcodeRelImmLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::jump_opcode_rel_imm::InteractionClaim { claimed_sum })
        } else {
            jump_opcode_rel_imm_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let mul_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<MulOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::mul_opcode::InteractionClaim { claimed_sum })
        } else {
            mul_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let mul_opcode_small_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<MulOpcodeSmallLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::mul_opcode_small::InteractionClaim { claimed_sum })
        } else {
            mul_opcode_small_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let qm_31_add_mul_opcode_interaction_claim =
            qm_31_add_mul_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let ret_opcode_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::device_interaction::<RetOpcodeLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::ret_opcode::InteractionClaim { claimed_sum })
        } else {
            ret_opcode_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let verify_instruction_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<VerifyInstructionLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::verify_instruction::InteractionClaim { claimed_sum })
        } else {
            verify_instruction_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let blake_round_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<BlakeRoundLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::blake_round::InteractionClaim { claimed_sum })
        } else {
            blake_round_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let blake_g_interaction_claim = blake_g_result.map(|(trace, claimed_sum)| {
            evals.extend(trace);
            BlakeGInteractionClaim { claimed_sum }
        });
        let blake_round_sigma_interaction_claim =
            blake_round_sigma_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let triple_xor_32_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<TripleXor32Lane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::triple_xor_32::InteractionClaim { claimed_sum })
        } else {
            triple_xor_32_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let verify_bitwise_xor_12_interaction_claim =
            verify_bitwise_xor_12_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let add_mod_builtin_interaction_claim = add_mod_builtin_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let bitwise_builtin_interaction_claim = bitwise_builtin_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let mul_mod_builtin_interaction_claim = mul_mod_builtin_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let pedersen_builtin_interaction_claim =
            pedersen_builtin_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let pedersen_builtin_narrow_windows_interaction_claim =
            pedersen_builtin_narrow_windows_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let poseidon_builtin_interaction_claim =
            poseidon_builtin_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check96_builtin_interaction_claim =
            range_check96_builtin_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check_builtin_interaction_claim =
            range_check_builtin_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let ec_op_builtin_interaction_claim = ec_op_builtin_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let partial_ec_mul_generic_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<PartialEcMulGenericLane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::partial_ec_mul_generic::InteractionClaim { claimed_sum })
        } else {
            partial_ec_mul_generic_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let pedersen_aggregator_window_bits_18_interaction_claim = if let Some((
            trace,
            claimed_sum,
        )) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<PedersenAggregatorW18Lane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(
                cairo_air::components::pedersen_aggregator_window_bits_18::InteractionClaim {
                    claimed_sum,
                },
            )
        } else {
            pedersen_aggregator_window_bits_18_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let partial_ec_mul_window_bits_18_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<PartialEcMulW18Lane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(
                cairo_air::components::partial_ec_mul_window_bits_18::InteractionClaim {
                    claimed_sum,
                },
            )
        } else {
            partial_ec_mul_window_bits_18_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let pedersen_points_table_window_bits_18_interaction_claim =
            pedersen_points_table_window_bits_18_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let pedersen_aggregator_window_bits_9_interaction_claim =
            pedersen_aggregator_window_bits_9_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let partial_ec_mul_window_bits_9_interaction_claim = partial_ec_mul_window_bits_9_result
            .map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let pedersen_points_table_window_bits_9_interaction_claim =
            pedersen_points_table_window_bits_9_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let poseidon_aggregator_interaction_claim =
            poseidon_aggregator_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let poseidon_3_partial_rounds_chain_interaction_claim =
            poseidon_3_partial_rounds_chain_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let poseidon_full_round_chain_interaction_claim =
            poseidon_full_round_chain_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let cube_252_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<Cube252Lane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::cube_252::InteractionClaim { claimed_sum })
        } else {
            cube_252_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let poseidon_round_keys_interaction_claim =
            poseidon_round_keys_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check_252_width_27_interaction_claim = if let Some((trace, claimed_sum)) =
            <B as OpcodeJitBackend>::builtin_device_interaction::<RangeCheck252Width27Lane>(
                exec_context,
                common_lookup_elements,
            ) {
            evals.extend(trace);
            Some(cairo_air::components::range_check_252_width_27::InteractionClaim { claimed_sum })
        } else {
            range_check_252_width_27_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            })
        };
        let memory_address_to_id_interaction_claim =
            memory_address_to_id_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let (memory_id_to_big_interaction_claim, memory_id_to_small_interaction_claim) =
            memory_id_to_big_result
                .map(
                    |(big_traces, small_trace, big_claimed_sums, small_claimed_sum)| {
                        // Extend each big segment's finalized trace then the small
                        // table's, in the same order the eager path extended them.
                        for trace in big_traces {
                            evals.extend(trace);
                        }
                        evals.extend(small_trace);
                        // The big claim's total is the field sum of the per-segment
                        // sums — associative/commutative, so identical to the eager
                        // computation.
                        let claimed_sum = big_claimed_sums.iter().sum::<SecureField>();
                        (
                            MemoryBigInteractionClaim {
                                big_claimed_sums,
                                claimed_sum,
                            },
                            MemorySmallInteractionClaim {
                                claimed_sum: small_claimed_sum,
                            },
                        )
                    },
                )
                .unzip();
        let range_check_6_interaction_claim = range_check_6_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_8_interaction_claim = range_check_8_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_11_interaction_claim = range_check_11_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_12_interaction_claim = range_check_12_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_18_interaction_claim = range_check_18_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_20_interaction_claim = range_check_20_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_4_3_interaction_claim = range_check_4_3_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_4_4_interaction_claim = range_check_4_4_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_9_9_interaction_claim = range_check_9_9_result.map(|(raw, build_claim)| {
            let (trace, claimed_sum) = B::finalize_raw_logup(raw);
            evals.extend(trace);
            build_claim(claimed_sum)
        });
        let range_check_7_2_5_interaction_claim =
            range_check_7_2_5_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check_3_6_6_3_interaction_claim =
            range_check_3_6_6_3_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check_4_4_4_4_interaction_claim =
            range_check_4_4_4_4_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let range_check_3_3_3_3_3_interaction_claim =
            range_check_3_3_3_3_3_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let verify_bitwise_xor_4_interaction_claim =
            verify_bitwise_xor_4_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let verify_bitwise_xor_7_interaction_claim =
            verify_bitwise_xor_7_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let verify_bitwise_xor_8_interaction_claim =
            verify_bitwise_xor_8_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });
        let verify_bitwise_xor_9_interaction_claim =
            verify_bitwise_xor_9_result.map(|(raw, build_claim)| {
                let (trace, claimed_sum) = B::finalize_raw_logup(raw);
                evals.extend(trace);
                build_claim(claimed_sum)
            });

        (
            evals,
            CairoInteractionClaim {
                add_opcode: add_opcode_interaction_claim,
                add_opcode_small: add_opcode_small_interaction_claim,
                add_ap_opcode: add_ap_opcode_interaction_claim,
                assert_eq_opcode: assert_eq_opcode_interaction_claim,
                assert_eq_opcode_imm: assert_eq_opcode_imm_interaction_claim,
                assert_eq_opcode_double_deref: assert_eq_opcode_double_deref_interaction_claim,
                blake_compress_opcode: blake_compress_opcode_interaction_claim,
                call_opcode_abs: call_opcode_abs_interaction_claim,
                call_opcode_rel_imm: call_opcode_rel_imm_interaction_claim,
                generic_opcode: generic_opcode_interaction_claim,
                jnz_opcode_non_taken: jnz_opcode_non_taken_interaction_claim,
                jnz_opcode_taken: jnz_opcode_taken_interaction_claim,
                jump_opcode_abs: jump_opcode_abs_interaction_claim,
                jump_opcode_double_deref: jump_opcode_double_deref_interaction_claim,
                jump_opcode_rel: jump_opcode_rel_interaction_claim,
                jump_opcode_rel_imm: jump_opcode_rel_imm_interaction_claim,
                mul_opcode: mul_opcode_interaction_claim,
                mul_opcode_small: mul_opcode_small_interaction_claim,
                qm_31_add_mul_opcode: qm_31_add_mul_opcode_interaction_claim,
                ret_opcode: ret_opcode_interaction_claim,
                verify_instruction: verify_instruction_interaction_claim,
                blake_round: blake_round_interaction_claim,
                blake_g: blake_g_interaction_claim,
                blake_round_sigma: blake_round_sigma_interaction_claim,
                triple_xor_32: triple_xor_32_interaction_claim,
                verify_bitwise_xor_12: verify_bitwise_xor_12_interaction_claim,
                add_mod_builtin: add_mod_builtin_interaction_claim,
                bitwise_builtin: bitwise_builtin_interaction_claim,
                mul_mod_builtin: mul_mod_builtin_interaction_claim,
                pedersen_builtin: pedersen_builtin_interaction_claim,
                pedersen_builtin_narrow_windows: pedersen_builtin_narrow_windows_interaction_claim,
                poseidon_builtin: poseidon_builtin_interaction_claim,
                range_check96_builtin: range_check96_builtin_interaction_claim,
                range_check_builtin: range_check_builtin_interaction_claim,
                ec_op_builtin: ec_op_builtin_interaction_claim,
                partial_ec_mul_generic: partial_ec_mul_generic_interaction_claim,
                pedersen_aggregator_window_bits_18:
                    pedersen_aggregator_window_bits_18_interaction_claim,
                partial_ec_mul_window_bits_18: partial_ec_mul_window_bits_18_interaction_claim,
                pedersen_points_table_window_bits_18:
                    pedersen_points_table_window_bits_18_interaction_claim,
                pedersen_aggregator_window_bits_9:
                    pedersen_aggregator_window_bits_9_interaction_claim,
                partial_ec_mul_window_bits_9: partial_ec_mul_window_bits_9_interaction_claim,
                pedersen_points_table_window_bits_9:
                    pedersen_points_table_window_bits_9_interaction_claim,
                poseidon_aggregator: poseidon_aggregator_interaction_claim,
                poseidon_3_partial_rounds_chain: poseidon_3_partial_rounds_chain_interaction_claim,
                poseidon_full_round_chain: poseidon_full_round_chain_interaction_claim,
                cube_252: cube_252_interaction_claim,
                poseidon_round_keys: poseidon_round_keys_interaction_claim,
                range_check_252_width_27: range_check_252_width_27_interaction_claim,
                memory_address_to_id: memory_address_to_id_interaction_claim,
                memory_id_to_big: memory_id_to_big_interaction_claim,
                memory_id_to_small: memory_id_to_small_interaction_claim,
                range_check_6: range_check_6_interaction_claim,
                range_check_8: range_check_8_interaction_claim,
                range_check_11: range_check_11_interaction_claim,
                range_check_12: range_check_12_interaction_claim,
                range_check_18: range_check_18_interaction_claim,
                range_check_20: range_check_20_interaction_claim,
                range_check_4_3: range_check_4_3_interaction_claim,
                range_check_4_4: range_check_4_4_interaction_claim,
                range_check_9_9: range_check_9_9_interaction_claim,
                range_check_7_2_5: range_check_7_2_5_interaction_claim,
                range_check_3_6_6_3: range_check_3_6_6_3_interaction_claim,
                range_check_4_4_4_4: range_check_4_4_4_4_interaction_claim,
                range_check_3_3_3_3_3: range_check_3_3_3_3_3_interaction_claim,
                verify_bitwise_xor_4: verify_bitwise_xor_4_interaction_claim,
                verify_bitwise_xor_7: verify_bitwise_xor_7_interaction_claim,
                verify_bitwise_xor_8: verify_bitwise_xor_8_interaction_claim,
                verify_bitwise_xor_9: verify_bitwise_xor_9_interaction_claim,
            },
        )
    }
}
pub fn get_sub_components(component_name: &str) -> Vec<&'static str> {
    match component_name {
        "generic_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "range_check_20",
                "range_check_18",
                "range_check_11",
                "generic_opcode",
            ]
        }
        "add_ap_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "range_check_18",
                "range_check_11",
                "add_ap_opcode",
            ]
        }
        "add_opcode_small" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "add_opcode_small",
            ]
        }
        "add_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "add_opcode",
            ]
        }
        "assert_eq_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "assert_eq_opcode",
            ]
        }
        "assert_eq_opcode_double_deref" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "assert_eq_opcode_double_deref",
            ]
        }
        "assert_eq_opcode_imm" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "assert_eq_opcode_imm",
            ]
        }
        "call_opcode_abs" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "call_opcode_abs",
            ]
        }
        "call_opcode_rel_imm" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "call_opcode_rel_imm",
            ]
        }
        "jnz_opcode_taken" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jnz_opcode_taken",
            ]
        }
        "jnz_opcode_non_taken" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jnz_opcode_non_taken",
            ]
        }
        "jump_opcode_rel_imm" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jump_opcode_rel_imm",
            ]
        }
        "jump_opcode_rel" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jump_opcode_rel",
            ]
        }
        "jump_opcode_double_deref" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jump_opcode_double_deref",
            ]
        }
        "jump_opcode_abs" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "jump_opcode_abs",
            ]
        }
        "mul_opcode_small" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "range_check_11",
                "mul_opcode_small",
            ]
        }
        "mul_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "range_check_20",
                "mul_opcode",
            ]
        }
        "ret_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "ret_opcode",
            ]
        }
        "qm_31_add_mul_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "range_check_4_4_4_4",
                "qm_31_add_mul_opcode",
            ]
        }
        "blake_compress_opcode" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_7_2_5",
                "range_check_4_3",
                "verify_instruction",
                "verify_bitwise_xor_8",
                "blake_round_sigma",
                "verify_bitwise_xor_12",
                "verify_bitwise_xor_4",
                "verify_bitwise_xor_7",
                "verify_bitwise_xor_9",
                "blake_g",
                "blake_round",
                "triple_xor_32",
                "blake_compress_opcode",
            ]
        }
        "bitwise_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "verify_bitwise_xor_9",
                "verify_bitwise_xor_8",
                "bitwise_builtin",
            ]
        }
        "range_check_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_builtin",
            ]
        }
        "range_check96_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_6",
                "range_check96_builtin",
            ]
        }
        "add_mod_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "add_mod_builtin",
            ]
        }
        "mul_mod_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_12",
                "range_check_3_6_6_3",
                "range_check_18",
                "mul_mod_builtin",
            ]
        }
        "poseidon_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_20",
                "cube_252",
                "poseidon_round_keys",
                "range_check_3_3_3_3_3",
                "poseidon_full_round_chain",
                "range_check_18",
                "range_check_252_width_27",
                "range_check_4_4_4_4",
                "range_check_4_4",
                "poseidon_3_partial_rounds_chain",
                "poseidon_aggregator",
                "poseidon_builtin",
            ]
        }
        "pedersen_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_8",
                "pedersen_points_table_window_bits_18",
                "range_check_20",
                "partial_ec_mul_window_bits_18",
                "pedersen_aggregator_window_bits_18",
                "pedersen_builtin",
            ]
        }
        "pedersen_builtin_narrow_windows" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_8",
                "pedersen_points_table_window_bits_9",
                "range_check_20",
                "partial_ec_mul_window_bits_9",
                "pedersen_aggregator_window_bits_9",
                "pedersen_builtin_narrow_windows",
            ]
        }
        "ec_op_builtin" => {
            vec![
                "memory_address_to_id",
                "range_check_9_9",
                "memory_id_to_big",
                "range_check_8",
                "range_check_20",
                "partial_ec_mul_generic",
                "ec_op_builtin",
            ]
        }
        _ => panic!("Unknown component: {component_name}"),
    }
}
