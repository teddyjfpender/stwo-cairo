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

use cairo_air::components::ret_opcode::{Claim as RetClaim, N_TRACE_COLUMNS as RET_N_COLS};
use cairo_air::relations::{
    CommonLookupElements, MEMORY_ADDRESS_TO_ID_RELATION_ID, MEMORY_ID_TO_BIG_RELATION_ID,
};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Column, FromSimdColumns};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_backend_cuda::{memory_witness as device_witness, BaseFieldVec, CudaBackend};
use stwo_cairo_adapter::memory::u128_to_4_limbs;
use stwo_constraint_framework::LogupFinalizeBackend;

use crate::witness::components::{
    memory_address_to_id, memory_id_to_big, ret_opcode, verify_instruction,
};
use crate::witness::memory_witness_backend::{compare_interaction, MemoryEvals};
use crate::witness::utils::{pack_values, AddInputs};

/// Relation ids that exist only as literals in the generated writers (the
/// differential pins them against the host writer's lookup tuples).
const OPCODES_RELATION_ID: u32 = 428564188;
const VERIFY_INSTRUCTION_RELATION_ID: u32 = 1719106205;

/// Backend hook for the opcode-cohort witness writers. One trait for the whole
/// cohort (one bound in `prover.rs`); each ported opcode adds a method pair +
/// an interaction-generator associated type.
pub trait OpcodeWitness: FromSimdColumns + LogupFinalizeBackend {
    /// Prove-wide device copies of the memory tables the opcode kernels
    /// gather from. Built once, before the opcode write scope; unit for SIMD.
    type MemTables: Send + Sync;
    type RetInteractionGen: Send;

    fn build_mem_tables(
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables;

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
}

impl OpcodeWitness for SimdBackend {
    type MemTables = ();
    type RetInteractionGen = ret_opcode::InteractionClaimGenerator;

    fn build_mem_tables(
        _memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        _memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables {
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

fn ret_device_path_enabled() -> bool {
    stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT
        && std::env::var("STWO_CUDA_RET_WITNESS").as_deref() != Ok("0")
}

fn verify_enabled() -> bool {
    std::env::var("STWO_CUDA_WITNESS_VERIFY").as_deref() == Ok("1")
}

impl OpcodeWitness for CudaBackend {
    type MemTables = device_witness::DeviceMemTables;
    type RetInteractionGen = CudaRetInteractionGen;

    fn build_mem_tables(
        memory_address_to_id: &memory_address_to_id::ClaimGenerator,
        memory_id_to_big: &memory_id_to_big::ClaimGenerator,
    ) -> Self::MemTables {
        if !ret_device_path_enabled() {
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
        for state in &inputs {
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
        }

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
                use stwo::prover::backend::simd::m31::PackedM31;
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
}
