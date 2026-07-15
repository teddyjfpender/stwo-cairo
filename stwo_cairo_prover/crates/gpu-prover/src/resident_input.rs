//! Generator-free ownership of the adapter output used by ReplacementV1.
//!
//! This layer only moves raw input into stable owners and exposes typed Casm
//! views. Shape derivation and production dispatch land after the differential
//! gate proves these views equal the legacy generator oracle.

use std::sync::Arc;

use cairo_air::air::PublicData;
use stwo_cairo_adapter::builtins::BuiltinSegments;
use stwo_cairo_adapter::memory::Memory;
use stwo_cairo_adapter::opcodes::{
    recorded_casm_descriptor, CasmStatesByOpcode, RecordedCasmDescriptor, RECORDED_CASM_DESCRIPTORS,
};
use stwo_cairo_adapter::{ProverInput, PublicSegmentContext};
use stwo_cairo_common::prover_types::cpu::CasmState;
use stwo_cairo_prover::witness::cairo::public_data_from_prover_input;
use stwo_cairo_prover::witness::opcodes::get_opcodes;

#[derive(Clone, Copy, Debug)]
pub struct ResidentCasmInput<'a> {
    pub descriptor: &'static RecordedCasmDescriptor,
    pub states: &'a [CasmState],
}

impl ResidentCasmInput<'_> {
    /// Canonical tightly packed `[pc, ap, fp]` row words.
    pub fn words(&self) -> &[u32] {
        bytemuck::cast_slice(self.states)
    }
}

/// Generator-free owner for resident proving inputs. Large memory and Casm
/// vectors move exactly once; initial/final state is retained through PublicData.
/// Diagnostic-only relocated fields from `extract-mem-trace` are intentionally
/// outside the resident proof contract.
pub struct ResidentProverInputOwner {
    public_data: PublicData,
    execution_memory: Arc<Memory>,
    pc_count: usize,
    public_memory_addresses: Vec<u32>,
    builtin_segments: BuiltinSegments,
    public_segment_context: PublicSegmentContext,
    casm_states: CasmStatesByOpcode,
}

impl ResidentProverInputOwner {
    pub fn encode(input: ProverInput) -> Self {
        let public_data = public_data_from_prover_input(&input);
        let ProverInput {
            state_transitions,
            memory,
            pc_count,
            public_memory_addresses,
            builtin_segments,
            public_segment_context,
            ..
        } = input;

        Self {
            public_data,
            execution_memory: Arc::new(memory),
            pc_count,
            public_memory_addresses,
            builtin_segments,
            public_segment_context,
            casm_states: state_transitions.casm_states_by_opcode,
        }
    }

    pub fn public_data(&self) -> &PublicData {
        &self.public_data
    }

    pub fn execution_memory(&self) -> &Arc<Memory> {
        &self.execution_memory
    }

    pub fn pc_count(&self) -> usize {
        self.pc_count
    }

    pub fn public_memory_addresses(&self) -> &[u32] {
        &self.public_memory_addresses
    }

    pub fn builtin_segments(&self) -> &BuiltinSegments {
        &self.builtin_segments
    }

    pub fn public_segment_context(&self) -> &PublicSegmentContext {
        &self.public_segment_context
    }

    pub fn opcode_labels(&self) -> Vec<&'static str> {
        get_opcodes(&self.casm_states)
    }

    /// Row count only; raw Casm values stay behind descriptor-typed access.
    /// `generic_opcode` is intentionally not a recorded descriptor and must be
    /// rejected by ReplacementV1 until it has a resident writer.
    pub fn direct_input_rows(&self, label: &str) -> Option<usize> {
        if label == "generic_opcode" {
            Some(self.casm_states.generic_opcode.len())
        } else {
            self.casm_input(label).map(|input| input.states.len())
        }
    }

    pub fn casm_input(&self, label: &str) -> Option<ResidentCasmInput<'_>> {
        let descriptor = recorded_casm_descriptor(label)?;
        Some(ResidentCasmInput {
            descriptor,
            states: descriptor.states(&self.casm_states),
        })
    }

    pub fn casm_inputs(&self) -> impl ExactSizeIterator<Item = ResidentCasmInput<'_>> {
        RECORDED_CASM_DESCRIPTORS
            .iter()
            .map(|descriptor| ResidentCasmInput {
                descriptor,
                states: descriptor.states(&self.casm_states),
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use cairo_vm::types::layout_name::LayoutName;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
    use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
    use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
    use stwo_cairo_prover::witness::cairo::create_cairo_claim_generator;
    use stwo_cairo_prover::witness::jit_prove_backend::recorded_casm_input_attempt;

    use super::ResidentProverInputOwner;

    #[test]
    fn raw_owner_matches_public_memory_and_every_present_generator_casm_oracle() {
        let input = run_and_adapt(
            &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap();
        let expected_builtins = bincode::serialize(&input.builtin_segments).unwrap();
        let expected_addresses = input.public_memory_addresses.clone();
        let expected_pc_count = input.pc_count;
        let generator = create_cairo_claim_generator(
            input.clone(),
            Arc::new(PreProcessedTraceVariant::Canonical.to_preprocessed_trace()),
        );
        let encode_started = Instant::now();
        let owner = ResidentProverInputOwner::encode(input);
        eprintln!(
            "resident_raw_input_encode_ms={:.3}",
            encode_started.elapsed().as_secs_f64() * 1e3
        );

        assert_eq!(
            bincode::serialize(owner.public_data()).unwrap(),
            bincode::serialize(&generator.public_data).unwrap()
        );
        let oracle_memory = generator.jit_memory.as_deref().unwrap();
        assert_eq!(
            owner.execution_memory().address_to_id,
            oracle_memory.address_to_id
        );
        assert_eq!(
            owner.execution_memory().f252_values,
            oracle_memory.f252_values
        );
        assert_eq!(
            owner.execution_memory().small_values,
            oracle_memory.small_values
        );
        assert_eq!(
            owner.execution_memory().config.small_max,
            oracle_memory.config.small_max
        );
        assert_eq!(
            owner.execution_memory().config.log_small_value_capacity,
            oracle_memory.config.log_small_value_capacity
        );
        assert_eq!(
            bincode::serialize(owner.builtin_segments()).unwrap(),
            expected_builtins
        );
        assert_eq!(owner.public_memory_addresses(), expected_addresses);
        assert_eq!(owner.pc_count(), expected_pc_count);

        for source in owner.casm_inputs() {
            let oracle = recorded_casm_input_attempt(&generator, source.descriptor.label).unwrap();
            if source.states.is_empty() {
                assert!(oracle.is_none(), "{}", source.descriptor.label);
            } else {
                let oracle = oracle.unwrap_or_else(|| {
                    panic!("missing generator oracle for {}", source.descriptor.label)
                });
                assert_eq!(source.states, oracle.inputs, "{}", source.descriptor.label);
                let oracle_words: &[u32] = bytemuck::cast_slice(oracle.inputs);
                assert_eq!(source.words(), oracle_words);
                assert_eq!(source.descriptor.include_iota, oracle.include_iota);
            }
        }
    }
}
