//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT.
//! Complete pre-consumption row-source projection of CairoClaimGenerator.

use stwo::prover::backend::simd::m31::N_LANES;

use super::cairo_claim_generator::CairoClaimGenerator;
use super::proof_shape::{
    padded_rows, rows_from_log_size, ComponentId, PendingRowsReason, ProofShape, ProofShapeError,
    RuntimeComponentShape, TracePartId, TracePartShape,
};

impl CairoClaimGenerator {
    pub fn proof_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
    ) -> Result<ProofShape, ProofShapeError> {
        let mut components = Vec::new();
        components.push(match &self.add_ap_opcode {
            None => RuntimeComponentShape::absent("add_ap_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("add_ap_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("add_ap_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.add_mod_builtin {
            None => RuntimeComponentShape::absent("add_mod_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("add_mod_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("add_mod_builtin", rows, rows)?
            }
        });
        components.push(match &self.add_opcode {
            None => RuntimeComponentShape::absent("add_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("add_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("add_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.add_opcode_small {
            None => RuntimeComponentShape::absent("add_opcode_small"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("add_opcode_small", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("add_opcode_small", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.assert_eq_opcode {
            None => RuntimeComponentShape::absent("assert_eq_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("assert_eq_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("assert_eq_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.assert_eq_opcode_double_deref {
            None => RuntimeComponentShape::absent("assert_eq_opcode_double_deref"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows =
                    padded_rows("assert_eq_opcode_double_deref", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform(
                    "assert_eq_opcode_double_deref",
                    n_real_rows,
                    padded_rows,
                )?
            }
        });
        components.push(match &self.assert_eq_opcode_imm {
            None => RuntimeComponentShape::absent("assert_eq_opcode_imm"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("assert_eq_opcode_imm", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("assert_eq_opcode_imm", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.bitwise_builtin {
            None => RuntimeComponentShape::absent("bitwise_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("bitwise_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("bitwise_builtin", rows, rows)?
            }
        });
        components.push(match &self.blake_compress_opcode {
            None => RuntimeComponentShape::absent("blake_compress_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows =
                    padded_rows("blake_compress_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("blake_compress_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.blake_g {
            None => RuntimeComponentShape::absent("blake_g"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("blake_g packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("blake_g remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow("blake_g"))?;
                RuntimeComponentShape::pending(
                    "blake_g",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.blake_round {
            None => RuntimeComponentShape::absent("blake_round"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("blake_round packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("blake_round remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow("blake_round"))?;
                RuntimeComponentShape::pending(
                    "blake_round",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.blake_round_sigma {
            None => RuntimeComponentShape::absent("blake_round_sigma"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "blake_round_sigma",
                    cairo_air::components::blake_round_sigma::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("blake_round_sigma", rows, rows)?
            }
        });
        components.push(match &self.call_opcode_abs {
            None => RuntimeComponentShape::absent("call_opcode_abs"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("call_opcode_abs", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("call_opcode_abs", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.call_opcode_rel_imm {
            None => RuntimeComponentShape::absent("call_opcode_rel_imm"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("call_opcode_rel_imm", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("call_opcode_rel_imm", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.cube_252 {
            None => RuntimeComponentShape::absent("cube_252"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("cube_252 packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("cube_252 remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow("cube_252"))?;
                RuntimeComponentShape::pending(
                    "cube_252",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.ec_op_builtin {
            None => RuntimeComponentShape::absent("ec_op_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("ec_op_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("ec_op_builtin", rows, rows)?
            }
        });
        components.push(match &self.generic_opcode {
            None => RuntimeComponentShape::absent("generic_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("generic_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("generic_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.jnz_opcode_non_taken {
            None => RuntimeComponentShape::absent("jnz_opcode_non_taken"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("jnz_opcode_non_taken", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("jnz_opcode_non_taken", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.jnz_opcode_taken {
            None => RuntimeComponentShape::absent("jnz_opcode_taken"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("jnz_opcode_taken", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("jnz_opcode_taken", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.jump_opcode_abs {
            None => RuntimeComponentShape::absent("jump_opcode_abs"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("jump_opcode_abs", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("jump_opcode_abs", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.jump_opcode_double_deref {
            None => RuntimeComponentShape::absent("jump_opcode_double_deref"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows =
                    padded_rows("jump_opcode_double_deref", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform(
                    "jump_opcode_double_deref",
                    n_real_rows,
                    padded_rows,
                )?
            }
        });
        components.push(match &self.jump_opcode_rel {
            None => RuntimeComponentShape::absent("jump_opcode_rel"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("jump_opcode_rel", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("jump_opcode_rel", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.jump_opcode_rel_imm {
            None => RuntimeComponentShape::absent("jump_opcode_rel_imm"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("jump_opcode_rel_imm", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("jump_opcode_rel_imm", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.memory_address_to_id {
            None => RuntimeComponentShape::absent("memory_address_to_id"),
            Some(gen) => {
                let split =
                    cairo_air::components::memory_address_to_id::MEMORY_ADDRESS_TO_ID_SPLIT as u64;
                let n_real_rows = (gen.table_size() as u64) / split;
                let padded_rows = padded_rows("memory_address_to_id", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("memory_address_to_id", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.memory_id_to_big {
            None => RuntimeComponentShape::absent("memory_id_to_big"),
            Some(gen) => {
                let max_big_rows = rows_from_log_size(
                    "memory_id_to_big",
                    stwo_cairo_common::preprocessed_columns::preprocessed_trace::MAX_SEQUENCE_LOG_SIZE,
                )? as usize;
                let big_rows = gen.big_table_size();
                let required_big_components = big_rows.div_ceil(max_big_rows);
                let n_big_components = opt_n_id_to_big_components
                    .unwrap_or(required_big_components);
                if n_big_components < required_big_components {
                    return Err(ProofShapeError::InvalidMemoryComponentCount {
                        requested: n_big_components,
                        required: required_big_components,
                    });
                }
                let mut parts = Vec::with_capacity(n_big_components + 1);
                for index in 0..n_big_components {
                    let n_real_rows = if index < required_big_components {
                        big_rows.saturating_sub(index * max_big_rows).min(max_big_rows)
                    } else {
                        N_LANES
                    } as u64;
                    let padded_rows = padded_rows(
                        "memory_id_to_big",
                        n_real_rows,
                        N_LANES as u64,
                    )?;
                    parts.push(TracePartShape {
                        part: TracePartId::MemoryBig(index as u32),
                        n_real_rows,
                        padded_rows,
                    });
                }
                let small_real_rows = gen.small_table_size() as u64;
                let small_padded_rows = padded_rows(
                    "memory_id_to_big",
                    small_real_rows,
                    N_LANES as u64,
                )?;
                parts.push(TracePartShape {
                    part: TracePartId::MemorySmall,
                    n_real_rows: small_real_rows,
                    padded_rows: small_padded_rows,
                });
                RuntimeComponentShape::parts("memory_id_to_big", parts)?
            },
        });
        components.push(match &self.mul_mod_builtin {
            None => RuntimeComponentShape::absent("mul_mod_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("mul_mod_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("mul_mod_builtin", rows, rows)?
            }
        });
        components.push(match &self.mul_opcode {
            None => RuntimeComponentShape::absent("mul_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("mul_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("mul_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.mul_opcode_small {
            None => RuntimeComponentShape::absent("mul_opcode_small"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("mul_opcode_small", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("mul_opcode_small", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.partial_ec_mul_generic {
            None => RuntimeComponentShape::absent("partial_ec_mul_generic"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("partial_ec_mul_generic packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("partial_ec_mul_generic remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow("partial_ec_mul_generic"))?;
                RuntimeComponentShape::pending(
                    "partial_ec_mul_generic",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.partial_ec_mul_window_bits_18 {
            None => RuntimeComponentShape::absent("partial_ec_mul_window_bits_18"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("partial_ec_mul_window_bits_18 packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("partial_ec_mul_window_bits_18 remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow(
                        "partial_ec_mul_window_bits_18",
                    ))?;
                RuntimeComponentShape::pending(
                    "partial_ec_mul_window_bits_18",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.partial_ec_mul_window_bits_9 {
            None => RuntimeComponentShape::absent("partial_ec_mul_window_bits_9"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("partial_ec_mul_window_bits_9 packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("partial_ec_mul_window_bits_9 remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow(
                        "partial_ec_mul_window_bits_9",
                    ))?;
                RuntimeComponentShape::pending(
                    "partial_ec_mul_window_bits_9",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.pedersen_aggregator_window_bits_18 {
            None => RuntimeComponentShape::absent("pedersen_aggregator_window_bits_18"),
            Some(gen) => RuntimeComponentShape::pending(
                "pedersen_aggregator_window_bits_18",
                PendingRowsReason::WitnessRelationFeeds,
                gen.mults.len() as u64,
            ),
        });
        components.push(match &self.pedersen_aggregator_window_bits_9 {
            None => RuntimeComponentShape::absent("pedersen_aggregator_window_bits_9"),
            Some(gen) => RuntimeComponentShape::pending(
                "pedersen_aggregator_window_bits_9",
                PendingRowsReason::WitnessRelationFeeds,
                gen.mults.len() as u64,
            ),
        });
        components.push(match &self.pedersen_builtin {
            None => RuntimeComponentShape::absent("pedersen_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("pedersen_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("pedersen_builtin", rows, rows)?
            }
        });
        components.push(match &self.pedersen_builtin_narrow_windows {
            None => RuntimeComponentShape::absent("pedersen_builtin_narrow_windows"),
            Some(gen) => {
                let rows = rows_from_log_size("pedersen_builtin_narrow_windows", gen.log_size)?;
                RuntimeComponentShape::uniform("pedersen_builtin_narrow_windows", rows, rows)?
            }
        });
        components.push(match &self.pedersen_points_table_window_bits_18 {
            None => RuntimeComponentShape::absent("pedersen_points_table_window_bits_18"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "pedersen_points_table_window_bits_18",
                    cairo_air::components::pedersen_points_table_window_bits_18::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("pedersen_points_table_window_bits_18", rows, rows)?
            }
        });
        components.push(match &self.pedersen_points_table_window_bits_9 {
            None => RuntimeComponentShape::absent("pedersen_points_table_window_bits_9"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "pedersen_points_table_window_bits_9",
                    cairo_air::components::pedersen_points_table_window_bits_9::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("pedersen_points_table_window_bits_9", rows, rows)?
            }
        });
        components.push(match &self.poseidon_3_partial_rounds_chain {
            None => RuntimeComponentShape::absent("poseidon_3_partial_rounds_chain"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("poseidon_3_partial_rounds_chain packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("poseidon_3_partial_rounds_chain remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow(
                        "poseidon_3_partial_rounds_chain",
                    ))?;
                RuntimeComponentShape::pending(
                    "poseidon_3_partial_rounds_chain",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.poseidon_aggregator {
            None => RuntimeComponentShape::absent("poseidon_aggregator"),
            Some(gen) => RuntimeComponentShape::pending(
                "poseidon_aggregator",
                PendingRowsReason::WitnessRelationFeeds,
                gen.mults.len() as u64,
            ),
        });
        components.push(match &self.poseidon_builtin {
            None => RuntimeComponentShape::absent("poseidon_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("poseidon_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("poseidon_builtin", rows, rows)?
            }
        });
        components.push(match &self.poseidon_full_round_chain {
            None => RuntimeComponentShape::absent("poseidon_full_round_chain"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("poseidon_full_round_chain packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("poseidon_full_round_chain remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow(
                        "poseidon_full_round_chain",
                    ))?;
                RuntimeComponentShape::pending(
                    "poseidon_full_round_chain",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.poseidon_round_keys {
            None => RuntimeComponentShape::absent("poseidon_round_keys"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "poseidon_round_keys",
                    cairo_air::components::poseidon_round_keys::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("poseidon_round_keys", rows, rows)?
            }
        });
        components.push(match &self.qm_31_add_mul_opcode {
            None => RuntimeComponentShape::absent("qm_31_add_mul_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("qm_31_add_mul_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("qm_31_add_mul_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.range_check96_builtin {
            None => RuntimeComponentShape::absent("range_check96_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("range_check96_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("range_check96_builtin", rows, rows)?
            }
        });
        components.push(match &self.range_check_11 {
            None => RuntimeComponentShape::absent("range_check_11"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_11",
                    cairo_air::components::range_check_11::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_11", rows, rows)?
            }
        });
        components.push(match &self.range_check_12 {
            None => RuntimeComponentShape::absent("range_check_12"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_12",
                    cairo_air::components::range_check_12::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_12", rows, rows)?
            }
        });
        components.push(match &self.range_check_18 {
            None => RuntimeComponentShape::absent("range_check_18"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_18",
                    cairo_air::components::range_check_18::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_18", rows, rows)?
            }
        });
        components.push(match &self.range_check_20 {
            None => RuntimeComponentShape::absent("range_check_20"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_20",
                    cairo_air::components::range_check_20::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_20", rows, rows)?
            }
        });
        components.push(match &self.range_check_252_width_27 {
            None => RuntimeComponentShape::absent("range_check_252_width_27"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("range_check_252_width_27 packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("range_check_252_width_27 remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow(
                        "range_check_252_width_27",
                    ))?;
                RuntimeComponentShape::pending(
                    "range_check_252_width_27",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.range_check_3_3_3_3_3 {
            None => RuntimeComponentShape::absent("range_check_3_3_3_3_3"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_3_3_3_3_3",
                    cairo_air::components::range_check_3_3_3_3_3::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_3_3_3_3_3", rows, rows)?
            }
        });
        components.push(match &self.range_check_3_6_6_3 {
            None => RuntimeComponentShape::absent("range_check_3_6_6_3"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_3_6_6_3",
                    cairo_air::components::range_check_3_6_6_3::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_3_6_6_3", rows, rows)?
            }
        });
        components.push(match &self.range_check_4_3 {
            None => RuntimeComponentShape::absent("range_check_4_3"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_4_3",
                    cairo_air::components::range_check_4_3::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_4_3", rows, rows)?
            }
        });
        components.push(match &self.range_check_4_4 {
            None => RuntimeComponentShape::absent("range_check_4_4"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_4_4",
                    cairo_air::components::range_check_4_4::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_4_4", rows, rows)?
            }
        });
        components.push(match &self.range_check_4_4_4_4 {
            None => RuntimeComponentShape::absent("range_check_4_4_4_4"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_4_4_4_4",
                    cairo_air::components::range_check_4_4_4_4::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_4_4_4_4", rows, rows)?
            }
        });
        components.push(match &self.range_check_6 {
            None => RuntimeComponentShape::absent("range_check_6"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_6",
                    cairo_air::components::range_check_6::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_6", rows, rows)?
            }
        });
        components.push(match &self.range_check_7_2_5 {
            None => RuntimeComponentShape::absent("range_check_7_2_5"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_7_2_5",
                    cairo_air::components::range_check_7_2_5::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_7_2_5", rows, rows)?
            }
        });
        components.push(match &self.range_check_8 {
            None => RuntimeComponentShape::absent("range_check_8"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_8",
                    cairo_air::components::range_check_8::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_8", rows, rows)?
            }
        });
        components.push(match &self.range_check_9_9 {
            None => RuntimeComponentShape::absent("range_check_9_9"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "range_check_9_9",
                    cairo_air::components::range_check_9_9::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("range_check_9_9", rows, rows)?
            }
        });
        components.push(match &self.range_check_builtin {
            None => RuntimeComponentShape::absent("range_check_builtin"),
            Some(gen) => {
                let rows = rows_from_log_size("range_check_builtin", gen.log_size)?;
                RuntimeComponentShape::uniform("range_check_builtin", rows, rows)?
            }
        });
        components.push(match &self.ret_opcode {
            None => RuntimeComponentShape::absent("ret_opcode"),
            Some(gen) => {
                let n_real_rows = gen.inputs.len() as u64;
                let padded_rows = padded_rows("ret_opcode", n_real_rows, N_LANES as u64)?;
                RuntimeComponentShape::uniform("ret_opcode", n_real_rows, padded_rows)?
            }
        });
        components.push(match &self.triple_xor_32 {
            None => RuntimeComponentShape::absent("triple_xor_32"),
            Some(gen) => {
                let packed_rows = gen
                    .packed_inputs
                    .lock()
                    .expect("triple_xor_32 packed-input mutex poisoned")
                    .len() as u64;
                let remainder_rows = gen
                    .remainder_inputs
                    .lock()
                    .expect("triple_xor_32 remainder-input mutex poisoned")
                    .len() as u64;
                let observed_n_real_rows = packed_rows
                    .checked_mul(N_LANES as u64)
                    .and_then(|rows| rows.checked_add(remainder_rows))
                    .ok_or(ProofShapeError::RowCountOverflow("triple_xor_32"))?;
                RuntimeComponentShape::pending(
                    "triple_xor_32",
                    PendingRowsReason::WitnessRelationFeeds,
                    observed_n_real_rows,
                )
            }
        });
        components.push(match &self.verify_bitwise_xor_12 {
            None => RuntimeComponentShape::absent("verify_bitwise_xor_12"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "verify_bitwise_xor_12",
                    cairo_air::components::verify_bitwise_xor_12::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("verify_bitwise_xor_12", rows, rows)?
            }
        });
        components.push(match &self.verify_bitwise_xor_4 {
            None => RuntimeComponentShape::absent("verify_bitwise_xor_4"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "verify_bitwise_xor_4",
                    cairo_air::components::verify_bitwise_xor_4::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("verify_bitwise_xor_4", rows, rows)?
            }
        });
        components.push(match &self.verify_bitwise_xor_7 {
            None => RuntimeComponentShape::absent("verify_bitwise_xor_7"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "verify_bitwise_xor_7",
                    cairo_air::components::verify_bitwise_xor_7::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("verify_bitwise_xor_7", rows, rows)?
            }
        });
        components.push(match &self.verify_bitwise_xor_8 {
            None => RuntimeComponentShape::absent("verify_bitwise_xor_8"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "verify_bitwise_xor_8",
                    cairo_air::components::verify_bitwise_xor_8::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("verify_bitwise_xor_8", rows, rows)?
            }
        });
        components.push(match &self.verify_bitwise_xor_9 {
            None => RuntimeComponentShape::absent("verify_bitwise_xor_9"),
            Some(_gen) => {
                let rows = rows_from_log_size(
                    "verify_bitwise_xor_9",
                    cairo_air::components::verify_bitwise_xor_9::LOG_SIZE,
                )?;
                RuntimeComponentShape::uniform("verify_bitwise_xor_9", rows, rows)?
            }
        });
        components.push(match &self.verify_instruction {
            None => RuntimeComponentShape::absent("verify_instruction"),
            Some(gen) => RuntimeComponentShape::pending(
                "verify_instruction",
                PendingRowsReason::WitnessRelationFeeds,
                gen.mults.len() as u64,
            ),
        });
        ProofShape::new(components)
    }
}

/// Generated exact row projection evaluated immediately before a component
/// generator is consumed. The aggregate witness post-processor inserts one
/// call per CairoClaimGenerator field; schedule drift is therefore checked.
pub(crate) trait FinalComponentShape {
    const COMPONENT: ComponentId;
    const ACCEPTS_DEVICE_FEED_ROWS: bool;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError>;
}

impl FinalComponentShape for super::components::add_ap_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "add_ap_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("add_ap_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("add_ap_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::add_mod_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "add_mod_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("add_mod_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("add_mod_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::add_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "add_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("add_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("add_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::add_opcode_small::ClaimGenerator {
    const COMPONENT: ComponentId = "add_opcode_small";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("add_opcode_small", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("add_opcode_small", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::assert_eq_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "assert_eq_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("assert_eq_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("assert_eq_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::assert_eq_opcode_double_deref::ClaimGenerator {
    const COMPONENT: ComponentId = "assert_eq_opcode_double_deref";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows =
                padded_rows("assert_eq_opcode_double_deref", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform(
                "assert_eq_opcode_double_deref",
                n_real_rows,
                padded_rows,
            )
        }
    }
}

impl FinalComponentShape for super::components::assert_eq_opcode_imm::ClaimGenerator {
    const COMPONENT: ComponentId = "assert_eq_opcode_imm";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("assert_eq_opcode_imm", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("assert_eq_opcode_imm", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::bitwise_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "bitwise_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("bitwise_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("bitwise_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::blake_compress_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "blake_compress_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("blake_compress_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("blake_compress_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::blake_g::ClaimGenerator {
    const COMPONENT: ComponentId = "blake_g";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("blake_g packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("blake_g remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow("blake_g"))?;
            let padded_rows = padded_rows("blake_g", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("blake_g", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::blake_round::ClaimGenerator {
    const COMPONENT: ComponentId = "blake_round";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("blake_round packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("blake_round remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow("blake_round"))?;
            let padded_rows = padded_rows("blake_round", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("blake_round", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::blake_round_sigma::ClaimGenerator {
    const COMPONENT: ComponentId = "blake_round_sigma";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "blake_round_sigma",
                cairo_air::components::blake_round_sigma::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("blake_round_sigma", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::call_opcode_abs::ClaimGenerator {
    const COMPONENT: ComponentId = "call_opcode_abs";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("call_opcode_abs", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("call_opcode_abs", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::call_opcode_rel_imm::ClaimGenerator {
    const COMPONENT: ComponentId = "call_opcode_rel_imm";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("call_opcode_rel_imm", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("call_opcode_rel_imm", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::cube_252::ClaimGenerator {
    const COMPONENT: ComponentId = "cube_252";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("cube_252 packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("cube_252 remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow("cube_252"))?;
            let padded_rows = padded_rows("cube_252", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("cube_252", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::ec_op_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "ec_op_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("ec_op_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("ec_op_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::generic_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "generic_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("generic_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("generic_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jnz_opcode_non_taken::ClaimGenerator {
    const COMPONENT: ComponentId = "jnz_opcode_non_taken";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jnz_opcode_non_taken", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jnz_opcode_non_taken", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jnz_opcode_taken::ClaimGenerator {
    const COMPONENT: ComponentId = "jnz_opcode_taken";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jnz_opcode_taken", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jnz_opcode_taken", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jump_opcode_abs::ClaimGenerator {
    const COMPONENT: ComponentId = "jump_opcode_abs";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jump_opcode_abs", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jump_opcode_abs", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jump_opcode_double_deref::ClaimGenerator {
    const COMPONENT: ComponentId = "jump_opcode_double_deref";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jump_opcode_double_deref", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jump_opcode_double_deref", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jump_opcode_rel::ClaimGenerator {
    const COMPONENT: ComponentId = "jump_opcode_rel";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jump_opcode_rel", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jump_opcode_rel", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::jump_opcode_rel_imm::ClaimGenerator {
    const COMPONENT: ComponentId = "jump_opcode_rel_imm";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("jump_opcode_rel_imm", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("jump_opcode_rel_imm", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::memory_address_to_id::ClaimGenerator {
    const COMPONENT: ComponentId = "memory_address_to_id";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let split =
                cairo_air::components::memory_address_to_id::MEMORY_ADDRESS_TO_ID_SPLIT as u64;
            let n_real_rows = (gen.table_size() as u64) / split;
            let padded_rows = padded_rows("memory_address_to_id", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("memory_address_to_id", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::memory_id_to_big::ClaimGenerator {
    const COMPONENT: ComponentId = "memory_id_to_big";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let max_big_rows = rows_from_log_size(
                "memory_id_to_big",
                stwo_cairo_common::preprocessed_columns::preprocessed_trace::MAX_SEQUENCE_LOG_SIZE,
            )? as usize;
            let big_rows = gen.big_table_size();
            let required_big_components = big_rows.div_ceil(max_big_rows);
            let n_big_components = opt_n_id_to_big_components.unwrap_or(required_big_components);
            if n_big_components < required_big_components {
                return Err(ProofShapeError::InvalidMemoryComponentCount {
                    requested: n_big_components,
                    required: required_big_components,
                });
            }
            let mut parts = Vec::with_capacity(n_big_components + 1);
            for index in 0..n_big_components {
                let n_real_rows = if index < required_big_components {
                    big_rows
                        .saturating_sub(index * max_big_rows)
                        .min(max_big_rows)
                } else {
                    N_LANES
                } as u64;
                let padded_rows = padded_rows("memory_id_to_big", n_real_rows, N_LANES as u64)?;
                parts.push(TracePartShape {
                    part: TracePartId::MemoryBig(index as u32),
                    n_real_rows,
                    padded_rows,
                });
            }
            let small_real_rows = gen.small_table_size() as u64;
            let small_padded_rows =
                padded_rows("memory_id_to_big", small_real_rows, N_LANES as u64)?;
            parts.push(TracePartShape {
                part: TracePartId::MemorySmall,
                n_real_rows: small_real_rows,
                padded_rows: small_padded_rows,
            });
            RuntimeComponentShape::parts("memory_id_to_big", parts)
        }
    }
}

impl FinalComponentShape for super::components::mul_mod_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "mul_mod_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("mul_mod_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("mul_mod_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::mul_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "mul_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("mul_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("mul_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::mul_opcode_small::ClaimGenerator {
    const COMPONENT: ComponentId = "mul_opcode_small";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("mul_opcode_small", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("mul_opcode_small", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::partial_ec_mul_generic::ClaimGenerator {
    const COMPONENT: ComponentId = "partial_ec_mul_generic";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("partial_ec_mul_generic packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("partial_ec_mul_generic remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow("partial_ec_mul_generic"))?;
            let padded_rows = padded_rows("partial_ec_mul_generic", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("partial_ec_mul_generic", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::partial_ec_mul_window_bits_18::ClaimGenerator {
    const COMPONENT: ComponentId = "partial_ec_mul_window_bits_18";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("partial_ec_mul_window_bits_18 packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("partial_ec_mul_window_bits_18 remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow(
                    "partial_ec_mul_window_bits_18",
                ))?;
            let padded_rows =
                padded_rows("partial_ec_mul_window_bits_18", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform(
                "partial_ec_mul_window_bits_18",
                n_real_rows,
                padded_rows,
            )
        }
    }
}

impl FinalComponentShape for super::components::partial_ec_mul_window_bits_9::ClaimGenerator {
    const COMPONENT: ComponentId = "partial_ec_mul_window_bits_9";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("partial_ec_mul_window_bits_9 packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("partial_ec_mul_window_bits_9 remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow(
                    "partial_ec_mul_window_bits_9",
                ))?;
            let padded_rows =
                padded_rows("partial_ec_mul_window_bits_9", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("partial_ec_mul_window_bits_9", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::pedersen_aggregator_window_bits_18::ClaimGenerator {
    const COMPONENT: ComponentId = "pedersen_aggregator_window_bits_18";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.mults.len() as u64;
            let padded_rows = padded_rows(
                "pedersen_aggregator_window_bits_18",
                n_real_rows,
                N_LANES as u64,
            )?;
            RuntimeComponentShape::uniform(
                "pedersen_aggregator_window_bits_18",
                n_real_rows,
                padded_rows,
            )
        }
    }
}

impl FinalComponentShape for super::components::pedersen_aggregator_window_bits_9::ClaimGenerator {
    const COMPONENT: ComponentId = "pedersen_aggregator_window_bits_9";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.mults.len() as u64;
            let padded_rows = padded_rows(
                "pedersen_aggregator_window_bits_9",
                n_real_rows,
                N_LANES as u64,
            )?;
            RuntimeComponentShape::uniform(
                "pedersen_aggregator_window_bits_9",
                n_real_rows,
                padded_rows,
            )
        }
    }
}

impl FinalComponentShape for super::components::pedersen_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "pedersen_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("pedersen_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("pedersen_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::pedersen_builtin_narrow_windows::ClaimGenerator {
    const COMPONENT: ComponentId = "pedersen_builtin_narrow_windows";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("pedersen_builtin_narrow_windows", gen.log_size)?;
            RuntimeComponentShape::uniform("pedersen_builtin_narrow_windows", rows, rows)
        }
    }
}

impl FinalComponentShape
    for super::components::pedersen_points_table_window_bits_18::ClaimGenerator
{
    const COMPONENT: ComponentId = "pedersen_points_table_window_bits_18";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "pedersen_points_table_window_bits_18",
                cairo_air::components::pedersen_points_table_window_bits_18::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("pedersen_points_table_window_bits_18", rows, rows)
        }
    }
}

impl FinalComponentShape
    for super::components::pedersen_points_table_window_bits_9::ClaimGenerator
{
    const COMPONENT: ComponentId = "pedersen_points_table_window_bits_9";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "pedersen_points_table_window_bits_9",
                cairo_air::components::pedersen_points_table_window_bits_9::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("pedersen_points_table_window_bits_9", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::poseidon_3_partial_rounds_chain::ClaimGenerator {
    const COMPONENT: ComponentId = "poseidon_3_partial_rounds_chain";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("poseidon_3_partial_rounds_chain packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("poseidon_3_partial_rounds_chain remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow(
                    "poseidon_3_partial_rounds_chain",
                ))?;
            let padded_rows = padded_rows(
                "poseidon_3_partial_rounds_chain",
                n_real_rows,
                N_LANES as u64,
            )?;
            RuntimeComponentShape::uniform(
                "poseidon_3_partial_rounds_chain",
                n_real_rows,
                padded_rows,
            )
        }
    }
}

impl FinalComponentShape for super::components::poseidon_aggregator::ClaimGenerator {
    const COMPONENT: ComponentId = "poseidon_aggregator";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.mults.len() as u64;
            let padded_rows = padded_rows("poseidon_aggregator", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("poseidon_aggregator", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::poseidon_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "poseidon_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("poseidon_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("poseidon_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::poseidon_full_round_chain::ClaimGenerator {
    const COMPONENT: ComponentId = "poseidon_full_round_chain";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("poseidon_full_round_chain packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("poseidon_full_round_chain remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow(
                    "poseidon_full_round_chain",
                ))?;
            let padded_rows =
                padded_rows("poseidon_full_round_chain", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("poseidon_full_round_chain", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::poseidon_round_keys::ClaimGenerator {
    const COMPONENT: ComponentId = "poseidon_round_keys";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "poseidon_round_keys",
                cairo_air::components::poseidon_round_keys::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("poseidon_round_keys", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::qm_31_add_mul_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "qm_31_add_mul_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("qm_31_add_mul_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("qm_31_add_mul_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check96_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check96_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("range_check96_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("range_check96_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_11::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_11";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_11",
                cairo_air::components::range_check_11::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_11", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_12::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_12";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_12",
                cairo_air::components::range_check_12::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_12", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_18::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_18";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_18",
                cairo_air::components::range_check_18::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_18", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_20::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_20";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_20",
                cairo_air::components::range_check_20::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_20", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_252_width_27::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_252_width_27";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("range_check_252_width_27 packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("range_check_252_width_27 remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow(
                    "range_check_252_width_27",
                ))?;
            let padded_rows = padded_rows("range_check_252_width_27", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("range_check_252_width_27", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_3_3_3_3_3::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_3_3_3_3_3";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_3_3_3_3_3",
                cairo_air::components::range_check_3_3_3_3_3::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_3_3_3_3_3", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_3_6_6_3::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_3_6_6_3";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_3_6_6_3",
                cairo_air::components::range_check_3_6_6_3::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_3_6_6_3", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_4_3::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_4_3";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_4_3",
                cairo_air::components::range_check_4_3::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_4_3", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_4_4::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_4_4";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_4_4",
                cairo_air::components::range_check_4_4::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_4_4", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_4_4_4_4::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_4_4_4_4";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_4_4_4_4",
                cairo_air::components::range_check_4_4_4_4::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_4_4_4_4", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_6::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_6";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_6",
                cairo_air::components::range_check_6::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_6", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_7_2_5::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_7_2_5";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_7_2_5",
                cairo_air::components::range_check_7_2_5::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_7_2_5", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_8::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_8";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_8",
                cairo_air::components::range_check_8::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_8", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_9_9::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_9_9";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "range_check_9_9",
                cairo_air::components::range_check_9_9::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("range_check_9_9", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::range_check_builtin::ClaimGenerator {
    const COMPONENT: ComponentId = "range_check_builtin";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size("range_check_builtin", gen.log_size)?;
            RuntimeComponentShape::uniform("range_check_builtin", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::ret_opcode::ClaimGenerator {
    const COMPONENT: ComponentId = "ret_opcode";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.inputs.len() as u64;
            let padded_rows = padded_rows("ret_opcode", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("ret_opcode", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::triple_xor_32::ClaimGenerator {
    const COMPONENT: ComponentId = "triple_xor_32";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = true;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let packed_rows = gen
                .packed_inputs
                .lock()
                .expect("triple_xor_32 packed-input mutex poisoned")
                .len() as u64;
            let remainder_rows = gen
                .remainder_inputs
                .lock()
                .expect("triple_xor_32 remainder-input mutex poisoned")
                .len() as u64;
            let n_real_rows = packed_rows
                .checked_mul(N_LANES as u64)
                .and_then(|rows| rows.checked_add(remainder_rows))
                .and_then(|rows| rows.checked_add(device_feed_rows))
                .ok_or(ProofShapeError::RowCountOverflow("triple_xor_32"))?;
            let padded_rows = padded_rows("triple_xor_32", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("triple_xor_32", n_real_rows, padded_rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_bitwise_xor_12::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_bitwise_xor_12";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "verify_bitwise_xor_12",
                cairo_air::components::verify_bitwise_xor_12::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("verify_bitwise_xor_12", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_bitwise_xor_4::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_bitwise_xor_4";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "verify_bitwise_xor_4",
                cairo_air::components::verify_bitwise_xor_4::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("verify_bitwise_xor_4", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_bitwise_xor_7::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_bitwise_xor_7";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "verify_bitwise_xor_7",
                cairo_air::components::verify_bitwise_xor_7::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("verify_bitwise_xor_7", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_bitwise_xor_8::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_bitwise_xor_8";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "verify_bitwise_xor_8",
                cairo_air::components::verify_bitwise_xor_8::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("verify_bitwise_xor_8", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_bitwise_xor_9::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_bitwise_xor_9";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let rows = rows_from_log_size(
                "verify_bitwise_xor_9",
                cairo_air::components::verify_bitwise_xor_9::LOG_SIZE,
            )?;
            RuntimeComponentShape::uniform("verify_bitwise_xor_9", rows, rows)
        }
    }
}

impl FinalComponentShape for super::components::verify_instruction::ClaimGenerator {
    const COMPONENT: ComponentId = "verify_instruction";
    const ACCEPTS_DEVICE_FEED_ROWS: bool = false;
    fn final_component_shape(
        &self,
        opt_n_id_to_big_components: Option<usize>,
        device_feed_rows: u64,
    ) -> Result<RuntimeComponentShape, ProofShapeError> {
        let _ = opt_n_id_to_big_components;
        let _ = device_feed_rows;
        let gen = self;
        let _ = gen;
        {
            let n_real_rows = gen.mults.len() as u64;
            let padded_rows = padded_rows("verify_instruction", n_real_rows, N_LANES as u64)?;
            RuntimeComponentShape::uniform("verify_instruction", n_real_rows, padded_rows)
        }
    }
}
