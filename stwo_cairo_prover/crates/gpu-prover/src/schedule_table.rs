//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT (design R8/§17).
//! Regenerate: `cargo run --manifest-path tools/schedule_emit/Cargo.toml`;
//! CI drift gate: same command with `--check`.
//!
//! The Cairo component feed DAG, derived from the transformer-emitted
//! SUB_FEED_LAYOUT consts + the COUNT_RELATIONS registry. Node order is
//! alphabetical (stable emission); EXECUTION order comes from
//! `Schedule::levels()`; trace COLLECTION order stays the claim
//! generator's (Fiat-Shamir-fixed) and is not this table's concern.

use crate::schedule::{
    CapacityFeed, ComponentNode, ComponentRowSource, ComponentStaticFacts, CountFeed, InputEdge,
    KernelIdentitySource, LogSizeSource, OutputEdge, Schedule, TraceColumnCount,
};

pub static CAIRO_SCHEDULE: Schedule = Schedule { nodes: NODES };

/// Canonical base/interaction commitment order from CairoClaimGenerator fields.
pub const CAIRO_COMMITMENT_COMPONENT_ORDER: &[&str] = &[
    "add_opcode",
    "add_opcode_small",
    "add_ap_opcode",
    "assert_eq_opcode",
    "assert_eq_opcode_imm",
    "assert_eq_opcode_double_deref",
    "blake_compress_opcode",
    "call_opcode_abs",
    "call_opcode_rel_imm",
    "generic_opcode",
    "jnz_opcode_non_taken",
    "jnz_opcode_taken",
    "jump_opcode_abs",
    "jump_opcode_double_deref",
    "jump_opcode_rel",
    "jump_opcode_rel_imm",
    "mul_opcode",
    "mul_opcode_small",
    "qm_31_add_mul_opcode",
    "ret_opcode",
    "verify_instruction",
    "blake_round",
    "blake_g",
    "blake_round_sigma",
    "triple_xor_32",
    "verify_bitwise_xor_12",
    "add_mod_builtin",
    "bitwise_builtin",
    "mul_mod_builtin",
    "pedersen_builtin",
    "pedersen_builtin_narrow_windows",
    "poseidon_builtin",
    "range_check96_builtin",
    "range_check_builtin",
    "ec_op_builtin",
    "partial_ec_mul_generic",
    "pedersen_aggregator_window_bits_18",
    "partial_ec_mul_window_bits_18",
    "pedersen_points_table_window_bits_18",
    "pedersen_aggregator_window_bits_9",
    "partial_ec_mul_window_bits_9",
    "pedersen_points_table_window_bits_9",
    "poseidon_aggregator",
    "poseidon_3_partial_rounds_chain",
    "poseidon_full_round_chain",
    "cube_252",
    "poseidon_round_keys",
    "range_check_252_width_27",
    "memory_address_to_id",
    "memory_id_to_big",
    "range_check_6",
    "range_check_8",
    "range_check_11",
    "range_check_12",
    "range_check_18",
    "range_check_20",
    "range_check_4_3",
    "range_check_4_4",
    "range_check_9_9",
    "range_check_7_2_5",
    "range_check_3_6_6_3",
    "range_check_4_4_4_4",
    "range_check_3_3_3_3_3",
    "verify_bitwise_xor_4",
    "verify_bitwise_xor_7",
    "verify_bitwise_xor_8",
    "verify_bitwise_xor_9",
];

static NODES: &[ComponentNode] = &[
    ComponentNode {
        id: "add_ap_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::add_ap_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(55),
            sub_words: Some(11),
            logup_columns: Some(4),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_11_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_18_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "add_mod_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::add_mod_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(808),
            sub_words: None,
            logup_columns: Some(27),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "add_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::add_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(117),
            sub_words: Some(13),
            logup_columns: Some(5),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "add_opcode_small",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::add_opcode_small::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(117),
            sub_words: Some(13),
            logup_columns: Some(5),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "assert_eq_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::assert_eq_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(24),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[CountFeed {
            family: "memory_address_to_id_state",
            n_relations: 1,
        }],
        slots: None,
    },
    ComponentNode {
        id: "assert_eq_opcode_double_deref",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::assert_eq_opcode_double_deref::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(57),
            sub_words: Some(11),
            logup_columns: Some(4),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "assert_eq_opcode_imm",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::assert_eq_opcode_imm::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(24),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[CountFeed {
            family: "memory_address_to_id_state",
            n_relations: 1,
        }],
        slots: None,
    },
    ComponentNode {
        id: "bitwise_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::bitwise_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(278),
            sub_words: None,
            logup_columns: Some(19),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "blake_compress_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::blake_compress_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(906),
            sub_words: None,
            logup_columns: Some(37),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "blake_g",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::blake_g::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(87),
            sub_words: Some(48),
            logup_columns: Some(9),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[InputEdge::Producer {
            of: "blake_round",
            word_base: 81,
            words_per_instance: 6,
            n_instances: 8,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "blake_round",
            n_instances: 8,
        }],
        outputs: &[
            OutputEdge {
                to: "verify_bitwise_xor_12",
                word_base: 24,
                words_per_instance: 3,
                n_instances: 2,
            },
            OutputEdge {
                to: "verify_bitwise_xor_4",
                word_base: 30,
                words_per_instance: 3,
                n_instances: 2,
            },
            OutputEdge {
                to: "verify_bitwise_xor_7",
                word_base: 36,
                words_per_instance: 3,
                n_instances: 2,
            },
            OutputEdge {
                to: "verify_bitwise_xor_8",
                word_base: 0,
                words_per_instance: 3,
                n_instances: 8,
            },
            OutputEdge {
                to: "verify_bitwise_xor_9",
                word_base: 42,
                words_per_instance: 3,
                n_instances: 2,
            },
        ],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "blake_round",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::blake_round::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(850),
            sub_words: Some(129),
            logup_columns: Some(30),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "blake_compress_opcode",
            n_instances: 10,
        }],
        outputs: &[OutputEdge {
            to: "blake_g",
            word_base: 81,
            words_per_instance: 6,
            n_instances: 8,
        }],
        counts: &[
            CountFeed {
                family: "blake_round_sigma_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_7_2_5_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "blake_round_sigma",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::blake_round_sigma::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(19),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(4),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::blake_round_sigma::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "blake_round",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "call_opcode_abs",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::call_opcode_abs::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(117),
            sub_words: Some(13),
            logup_columns: Some(5),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "call_opcode_rel_imm",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::call_opcode_rel_imm::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(117),
            sub_words: Some(13),
            logup_columns: Some(5),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "cube_252",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::cube_252::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(259),
            sub_words: Some(140),
            logup_columns: Some(50),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_3_partial_rounds_chain",
                n_instances: 3,
            },
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 2,
            },
            CapacityFeed {
                from: "poseidon_full_round_chain",
                n_instances: 3,
            },
        ],
        outputs: &[],
        counts: &[
            CountFeed {
                family: "range_check_20_state",
                n_relations: 8,
            },
            CountFeed {
                family: "range_check_9_9_state",
                n_relations: 8,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "ec_op_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::ec_op_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(488),
            sub_words: None,
            logup_columns: Some(9),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "generic_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::generic_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(261),
            sub_words: None,
            logup_columns: Some(34),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "jnz_opcode_non_taken",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jnz_opcode_non_taken::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(51),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "jnz_opcode_taken",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jnz_opcode_taken::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(84),
            sub_words: Some(11),
            logup_columns: Some(4),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "jump_opcode_abs",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jump_opcode_abs::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(51),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "jump_opcode_double_deref",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jump_opcode_double_deref::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(84),
            sub_words: Some(11),
            logup_columns: Some(4),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "jump_opcode_rel",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jump_opcode_rel::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(51),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "jump_opcode_rel_imm",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::jump_opcode_rel_imm::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(51),
            sub_words: Some(9),
            logup_columns: Some(3),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "memory_address_to_id",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::memory_address_to_id::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: None,
            sub_words: None,
            logup_columns: Some(8),
            row_source: ComponentRowSource::MemoryAddress,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "add_ap_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "add_mod_builtin",
                n_instances: 29,
            },
            CapacityFeed {
                from: "add_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "add_opcode_small",
                n_instances: 3,
            },
            CapacityFeed {
                from: "assert_eq_opcode",
                n_instances: 2,
            },
            CapacityFeed {
                from: "assert_eq_opcode_double_deref",
                n_instances: 3,
            },
            CapacityFeed {
                from: "assert_eq_opcode_imm",
                n_instances: 2,
            },
            CapacityFeed {
                from: "bitwise_builtin",
                n_instances: 5,
            },
            CapacityFeed {
                from: "blake_compress_opcode",
                n_instances: 20,
            },
            CapacityFeed {
                from: "blake_round",
                n_instances: 16,
            },
            CapacityFeed {
                from: "call_opcode_abs",
                n_instances: 3,
            },
            CapacityFeed {
                from: "call_opcode_rel_imm",
                n_instances: 3,
            },
            CapacityFeed {
                from: "ec_op_builtin",
                n_instances: 7,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "jnz_opcode_non_taken",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jnz_opcode_taken",
                n_instances: 2,
            },
            CapacityFeed {
                from: "jump_opcode_abs",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_double_deref",
                n_instances: 2,
            },
            CapacityFeed {
                from: "jump_opcode_rel",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_rel_imm",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_mod_builtin",
                n_instances: 29,
            },
            CapacityFeed {
                from: "mul_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "mul_opcode_small",
                n_instances: 3,
            },
            CapacityFeed {
                from: "pedersen_builtin",
                n_instances: 3,
            },
            CapacityFeed {
                from: "pedersen_builtin_narrow_windows",
                n_instances: 3,
            },
            CapacityFeed {
                from: "poseidon_builtin",
                n_instances: 6,
            },
            CapacityFeed {
                from: "qm_31_add_mul_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "range_check96_builtin",
                n_instances: 1,
            },
            CapacityFeed {
                from: "range_check_builtin",
                n_instances: 1,
            },
            CapacityFeed {
                from: "ret_opcode",
                n_instances: 2,
            },
            CapacityFeed {
                from: "verify_instruction",
                n_instances: 1,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "memory_id_to_big",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::SplitMemory {
                big: cairo_air::components::memory_id_to_big::BIG_N_COLUMNS as u32,
                small: cairo_air::components::memory_id_to_small::N_TRACE_COLUMNS as u32,
            },
            lookup_words: None,
            sub_words: None,
            logup_columns: None,
            row_source: ComponentRowSource::MemoryIdToBig,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "add_ap_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "add_mod_builtin",
                n_instances: 24,
            },
            CapacityFeed {
                from: "add_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "add_opcode_small",
                n_instances: 3,
            },
            CapacityFeed {
                from: "assert_eq_opcode_double_deref",
                n_instances: 1,
            },
            CapacityFeed {
                from: "bitwise_builtin",
                n_instances: 5,
            },
            CapacityFeed {
                from: "blake_compress_opcode",
                n_instances: 20,
            },
            CapacityFeed {
                from: "blake_round",
                n_instances: 16,
            },
            CapacityFeed {
                from: "call_opcode_abs",
                n_instances: 3,
            },
            CapacityFeed {
                from: "call_opcode_rel_imm",
                n_instances: 3,
            },
            CapacityFeed {
                from: "ec_op_builtin",
                n_instances: 7,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "jnz_opcode_non_taken",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jnz_opcode_taken",
                n_instances: 2,
            },
            CapacityFeed {
                from: "jump_opcode_abs",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_double_deref",
                n_instances: 2,
            },
            CapacityFeed {
                from: "jump_opcode_rel",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_rel_imm",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_mod_builtin",
                n_instances: 24,
            },
            CapacityFeed {
                from: "mul_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "mul_opcode_small",
                n_instances: 3,
            },
            CapacityFeed {
                from: "pedersen_aggregator_window_bits_18",
                n_instances: 3,
            },
            CapacityFeed {
                from: "pedersen_aggregator_window_bits_9",
                n_instances: 3,
            },
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 6,
            },
            CapacityFeed {
                from: "qm_31_add_mul_opcode",
                n_instances: 3,
            },
            CapacityFeed {
                from: "range_check96_builtin",
                n_instances: 1,
            },
            CapacityFeed {
                from: "range_check_builtin",
                n_instances: 1,
            },
            CapacityFeed {
                from: "ret_opcode",
                n_instances: 2,
            },
            CapacityFeed {
                from: "verify_instruction",
                n_instances: 1,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "mul_mod_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::mul_mod_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(1196),
            sub_words: None,
            logup_columns: Some(94),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "mul_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::mul_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(173),
            sub_words: Some(41),
            logup_columns: Some(19),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_20_state",
                n_relations: 8,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "mul_opcode_small",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::mul_opcode_small::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(123),
            sub_words: Some(16),
            logup_columns: Some(6),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_11_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "partial_ec_mul_generic",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::partial_ec_mul_generic::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(990),
            sub_words: Some(424),
            logup_columns: Some(157),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "ec_op_builtin",
            n_instances: 252,
        }],
        outputs: &[],
        counts: &[
            CountFeed {
                family: "range_check_20_state",
                n_relations: 8,
            },
            CountFeed {
                family: "range_check_8_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_9_9_state",
                n_relations: 8,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "partial_ec_mul_window_bits_18",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::partial_ec_mul_window_bits_18::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(498),
            sub_words: Some(169),
            logup_columns: Some(65),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[InputEdge::Producer {
            of: "pedersen_aggregator_window_bits_18",
            word_base: 7,
            words_per_instance: 72,
            n_instances: 28,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "pedersen_aggregator_window_bits_18",
            n_instances: 28,
        }],
        outputs: &[],
        counts: &[
            CountFeed {
                family: "pedersen_points_table_window_bits_18_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_20_state",
                n_relations: 8,
            },
            CountFeed {
                family: "range_check_9_9_state",
                n_relations: 8,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "partial_ec_mul_window_bits_9",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::partial_ec_mul_window_bits_9::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(526),
            sub_words: None,
            logup_columns: Some(65),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "pedersen_aggregator_window_bits_9",
            n_instances: 56,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_aggregator_window_bits_18",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_aggregator_window_bits_18::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(396),
            sub_words: Some(2023),
            logup_columns: Some(6),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "pedersen_builtin",
            n_instances: 1,
        }],
        outputs: &[OutputEdge {
            to: "partial_ec_mul_window_bits_18",
            word_base: 7,
            words_per_instance: 72,
            n_instances: 28,
        }],
        counts: &[
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_8_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_aggregator_window_bits_9",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_aggregator_window_bits_9::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(452),
            sub_words: None,
            logup_columns: Some(6),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "pedersen_builtin_narrow_windows",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(14),
            sub_words: None,
            logup_columns: Some(2),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_builtin_narrow_windows",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_builtin_narrow_windows::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(14),
            sub_words: None,
            logup_columns: Some(2),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_points_table_window_bits_18",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_points_table_window_bits_18::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(59),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(23),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(
            cairo_air::components::pedersen_points_table_window_bits_18::LOG_SIZE,
        ),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "partial_ec_mul_window_bits_18",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "pedersen_points_table_window_bits_9",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::pedersen_points_table_window_bits_9::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(59),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(15),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(
            cairo_air::components::pedersen_points_table_window_bits_9::LOG_SIZE,
        ),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "partial_ec_mul_window_bits_9",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "poseidon_3_partial_rounds_chain",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::poseidon_3_partial_rounds_chain::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(255),
            sub_words: None,
            logup_columns: Some(9),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "poseidon_aggregator",
            n_instances: 27,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "poseidon_aggregator",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::poseidon_aggregator::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(522),
            sub_words: None,
            logup_columns: Some(14),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "poseidon_builtin",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "poseidon_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::poseidon_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(26),
            sub_words: None,
            logup_columns: Some(4),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "poseidon_full_round_chain",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::poseidon_full_round_chain::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(199),
            sub_words: None,
            logup_columns: Some(6),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "poseidon_aggregator",
            n_instances: 8,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "poseidon_round_keys",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::poseidon_round_keys::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(33),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(6),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::poseidon_round_keys::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_3_partial_rounds_chain",
                n_instances: 1,
            },
            CapacityFeed {
                from: "poseidon_full_round_chain",
                n_instances: 1,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "qm_31_add_mul_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::qm_31_add_mul_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(132),
            sub_words: Some(25),
            logup_columns: Some(6),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_4_4_4_4_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "range_check96_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check96_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(36),
            sub_words: None,
            logup_columns: Some(2),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_11",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_11::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(3),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(11),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_11::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "add_ap_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_opcode_small",
                n_instances: 3,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_12",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_12::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(3),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(12),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_12::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "mul_mod_builtin",
            n_instances: 32,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_18",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_18::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(6),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(18),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_18::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "add_ap_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_mod_builtin",
                n_instances: 62,
            },
            CapacityFeed {
                from: "range_check_252_width_27",
                n_instances: 9,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_20",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_20::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(24),
            sub_words: None,
            logup_columns: Some(4),
            row_source: ComponentRowSource::FixedLogSize(20),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_20::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "cube_252",
                n_instances: 56,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 28,
            },
            CapacityFeed {
                from: "mul_opcode",
                n_instances: 28,
            },
            CapacityFeed {
                from: "partial_ec_mul_generic",
                n_instances: 196,
            },
            CapacityFeed {
                from: "partial_ec_mul_window_bits_18",
                n_instances: 84,
            },
            CapacityFeed {
                from: "partial_ec_mul_window_bits_9",
                n_instances: 84,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_252_width_27",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_252_width_27::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(46),
            sub_words: Some(19),
            logup_columns: Some(8),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_3_partial_rounds_chain",
                n_instances: 3,
            },
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 2,
            },
        ],
        outputs: &[],
        counts: &[
            CountFeed {
                family: "range_check_18_state",
                n_relations: 2,
            },
            CountFeed {
                family: "range_check_9_9_state",
                n_relations: 5,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "range_check_3_3_3_3_3",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_3_3_3_3_3::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(7),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(15),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_3_3_3_3_3::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 2,
            },
            CapacityFeed {
                from: "poseidon_full_round_chain",
                n_instances: 6,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_3_6_6_3",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_3_6_6_3::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(6),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(18),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_3_6_6_3::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "mul_mod_builtin",
            n_instances: 40,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_4_3",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_4_3::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(4),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(7),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_4_3::LOG_SIZE),
        inputs: &[InputEdge::Producer {
            of: "verify_instruction",
            word_base: 3,
            words_per_instance: 2,
            n_instances: 1,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "verify_instruction",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_4_4",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_4_4::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(4),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(8),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_4_4::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_3_partial_rounds_chain",
                n_instances: 3,
            },
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 3,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_4_4_4_4",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_4_4_4_4::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(6),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(16),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_4_4_4_4::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "poseidon_3_partial_rounds_chain",
                n_instances: 6,
            },
            CapacityFeed {
                from: "poseidon_aggregator",
                n_instances: 6,
            },
            CapacityFeed {
                from: "qm_31_add_mul_opcode",
                n_instances: 3,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_6",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_6::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(3),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(6),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_6::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "range_check96_builtin",
            n_instances: 1,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_7_2_5",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_7_2_5::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(5),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(14),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_7_2_5::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "blake_compress_opcode",
                n_instances: 17,
            },
            CapacityFeed {
                from: "blake_round",
                n_instances: 16,
            },
            CapacityFeed {
                from: "verify_instruction",
                n_instances: 1,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_8",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_8::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(3),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(8),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_8::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "ec_op_builtin",
                n_instances: 2,
            },
            CapacityFeed {
                from: "partial_ec_mul_generic",
                n_instances: 4,
            },
            CapacityFeed {
                from: "pedersen_aggregator_window_bits_18",
                n_instances: 4,
            },
            CapacityFeed {
                from: "pedersen_aggregator_window_bits_9",
                n_instances: 4,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_9_9",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_9_9::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(32),
            sub_words: None,
            logup_columns: Some(4),
            row_source: ComponentRowSource::FixedLogSize(18),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::range_check_9_9::LOG_SIZE),
        inputs: &[],
        capacity_inputs: &[
            CapacityFeed {
                from: "cube_252",
                n_instances: 42,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 28,
            },
            CapacityFeed {
                from: "partial_ec_mul_generic",
                n_instances: 112,
            },
            CapacityFeed {
                from: "partial_ec_mul_window_bits_18",
                n_instances: 42,
            },
            CapacityFeed {
                from: "partial_ec_mul_window_bits_9",
                n_instances: 42,
            },
            CapacityFeed {
                from: "range_check_252_width_27",
                n_instances: 5,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "range_check_builtin",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::range_check_builtin::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(34),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::StoredLogSize,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "ret_opcode",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::ret_opcode::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(84),
            sub_words: Some(11),
            logup_columns: Some(4),
            row_source: ComponentRowSource::DirectInputs,
            kernel_identity: KernelIdentitySource::RecordedWitness,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[],
        outputs: &[OutputEdge {
            to: "verify_instruction",
            word_base: 0,
            words_per_instance: 7,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
    ComponentNode {
        id: "triple_xor_32",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::triple_xor_32::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(43),
            sub_words: Some(24),
            logup_columns: Some(5),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[],
        capacity_inputs: &[CapacityFeed {
            from: "blake_compress_opcode",
            n_instances: 8,
        }],
        outputs: &[OutputEdge {
            to: "verify_bitwise_xor_8",
            word_base: 0,
            words_per_instance: 3,
            n_instances: 8,
        }],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_bitwise_xor_12",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_bitwise_xor_12::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: None,
            sub_words: None,
            logup_columns: Some(8),
            row_source: ComponentRowSource::FixedLogSize(20),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::verify_bitwise_xor_12::LOG_SIZE),
        inputs: &[InputEdge::Producer {
            of: "blake_g",
            word_base: 24,
            words_per_instance: 3,
            n_instances: 2,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "blake_g",
            n_instances: 2,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_bitwise_xor_4",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_bitwise_xor_4::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(5),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(8),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::verify_bitwise_xor_4::LOG_SIZE),
        inputs: &[InputEdge::Producer {
            of: "blake_g",
            word_base: 30,
            words_per_instance: 3,
            n_instances: 2,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "blake_g",
            n_instances: 2,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_bitwise_xor_7",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_bitwise_xor_7::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(5),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(14),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::verify_bitwise_xor_7::LOG_SIZE),
        inputs: &[InputEdge::Producer {
            of: "blake_g",
            word_base: 36,
            words_per_instance: 3,
            n_instances: 2,
        }],
        capacity_inputs: &[CapacityFeed {
            from: "blake_g",
            n_instances: 2,
        }],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_bitwise_xor_8",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_bitwise_xor_8::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(10),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(16),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::verify_bitwise_xor_8::LOG_SIZE),
        inputs: &[
            InputEdge::Producer {
                of: "blake_g",
                word_base: 0,
                words_per_instance: 3,
                n_instances: 8,
            },
            InputEdge::Producer {
                of: "triple_xor_32",
                word_base: 0,
                words_per_instance: 3,
                n_instances: 8,
            },
        ],
        capacity_inputs: &[
            CapacityFeed {
                from: "bitwise_builtin",
                n_instances: 1,
            },
            CapacityFeed {
                from: "blake_compress_opcode",
                n_instances: 4,
            },
            CapacityFeed {
                from: "blake_g",
                n_instances: 8,
            },
            CapacityFeed {
                from: "triple_xor_32",
                n_instances: 8,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_bitwise_xor_9",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_bitwise_xor_9::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(5),
            sub_words: None,
            logup_columns: Some(1),
            row_source: ComponentRowSource::FixedLogSize(18),
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::Fixed(cairo_air::components::verify_bitwise_xor_9::LOG_SIZE),
        inputs: &[InputEdge::Producer {
            of: "blake_g",
            word_base: 42,
            words_per_instance: 3,
            n_instances: 2,
        }],
        capacity_inputs: &[
            CapacityFeed {
                from: "bitwise_builtin",
                n_instances: 27,
            },
            CapacityFeed {
                from: "blake_g",
                n_instances: 2,
            },
        ],
        outputs: &[],
        counts: &[],
        slots: None,
    },
    ComponentNode {
        id: "verify_instruction",
        facts: ComponentStaticFacts {
            trace_columns: TraceColumnCount::Fixed(
                cairo_air::components::verify_instruction::N_TRACE_COLUMNS as u32,
            ),
            lookup_words: Some(50),
            sub_words: Some(7),
            logup_columns: Some(3),
            row_source: ComponentRowSource::WitnessRelationFeeds,
            kernel_identity: KernelIdentitySource::None,
        },
        kernel: None,
        log_size: LogSizeSource::FromStates,
        inputs: &[
            InputEdge::Producer {
                of: "add_ap_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "add_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "add_opcode_small",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "assert_eq_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "assert_eq_opcode_double_deref",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "assert_eq_opcode_imm",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "call_opcode_abs",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "call_opcode_rel_imm",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jnz_opcode_non_taken",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jnz_opcode_taken",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jump_opcode_abs",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jump_opcode_double_deref",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jump_opcode_rel",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "jump_opcode_rel_imm",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "mul_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "mul_opcode_small",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "qm_31_add_mul_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
            InputEdge::Producer {
                of: "ret_opcode",
                word_base: 0,
                words_per_instance: 7,
                n_instances: 1,
            },
        ],
        capacity_inputs: &[
            CapacityFeed {
                from: "add_ap_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "add_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "add_opcode_small",
                n_instances: 1,
            },
            CapacityFeed {
                from: "assert_eq_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "assert_eq_opcode_double_deref",
                n_instances: 1,
            },
            CapacityFeed {
                from: "assert_eq_opcode_imm",
                n_instances: 1,
            },
            CapacityFeed {
                from: "blake_compress_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "call_opcode_abs",
                n_instances: 1,
            },
            CapacityFeed {
                from: "call_opcode_rel_imm",
                n_instances: 1,
            },
            CapacityFeed {
                from: "generic_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jnz_opcode_non_taken",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jnz_opcode_taken",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_abs",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_double_deref",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_rel",
                n_instances: 1,
            },
            CapacityFeed {
                from: "jump_opcode_rel_imm",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "mul_opcode_small",
                n_instances: 1,
            },
            CapacityFeed {
                from: "qm_31_add_mul_opcode",
                n_instances: 1,
            },
            CapacityFeed {
                from: "ret_opcode",
                n_instances: 1,
            },
        ],
        outputs: &[OutputEdge {
            to: "range_check_4_3",
            word_base: 3,
            words_per_instance: 2,
            n_instances: 1,
        }],
        counts: &[
            CountFeed {
                family: "memory_address_to_id_state",
                n_relations: 1,
            },
            CountFeed {
                family: "memory_id_to_big_state",
                n_relations: 1,
            },
            CountFeed {
                family: "range_check_7_2_5_state",
                n_relations: 1,
            },
        ],
        slots: None,
    },
];
