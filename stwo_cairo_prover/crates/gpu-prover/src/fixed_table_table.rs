//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT.
//! Exact BaseTrace and flattened LookupInputs sources parsed from fixed-table writers.
//! Regenerate with `cargo run --manifest-path tools/schedule_emit/Cargo.toml`.

use crate::fixed_table::{
    ExpandedXorLayout, FixedTableLookupLayout, FixedTableLookupWord, FixedTableMaterializationPlan,
    FixedTableMaterializationTable, FixedTableTraceColumn, FixedTableWordSource,
};

pub static CAIRO_FIXED_TABLE_MATERIALIZATION: FixedTableMaterializationTable =
    FixedTableMaterializationTable {
        plans: PLANS,
        expected_hash: EXPECTED_HASH,
    };

pub const EXPECTED_HASH: u64 = 0x7383de8a8df6398b;

static PLANS: &[FixedTableMaterializationPlan] = &[
    FixedTableMaterializationPlan {
        component: "blake_round_sigma",
        log_size: 4,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1805967942),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_4"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_0"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_1"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_2"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_3"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_4"),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_5"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_6"),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_7"),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_8"),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_9"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_10"),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_11"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_12"),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_13"),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_14"),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::PreprocessedColumn("blake_sigma_15"),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "pedersen_points_table_window_bits_18",
        log_size: 23,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1444721856),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_23"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_0"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_1"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_2"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_3"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_4"),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_5"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_6"),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_7"),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_8"),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_9"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_10"),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_11"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_12"),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_13"),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_14"),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_15"),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_16"),
            },
            FixedTableLookupWord {
                output_word: 19,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_17"),
            },
            FixedTableLookupWord {
                output_word: 20,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_18"),
            },
            FixedTableLookupWord {
                output_word: 21,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_19"),
            },
            FixedTableLookupWord {
                output_word: 22,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_20"),
            },
            FixedTableLookupWord {
                output_word: 23,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_21"),
            },
            FixedTableLookupWord {
                output_word: 24,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_22"),
            },
            FixedTableLookupWord {
                output_word: 25,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_23"),
            },
            FixedTableLookupWord {
                output_word: 26,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_24"),
            },
            FixedTableLookupWord {
                output_word: 27,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_25"),
            },
            FixedTableLookupWord {
                output_word: 28,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_26"),
            },
            FixedTableLookupWord {
                output_word: 29,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_27"),
            },
            FixedTableLookupWord {
                output_word: 30,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_28"),
            },
            FixedTableLookupWord {
                output_word: 31,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_29"),
            },
            FixedTableLookupWord {
                output_word: 32,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_30"),
            },
            FixedTableLookupWord {
                output_word: 33,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_31"),
            },
            FixedTableLookupWord {
                output_word: 34,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_32"),
            },
            FixedTableLookupWord {
                output_word: 35,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_33"),
            },
            FixedTableLookupWord {
                output_word: 36,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_34"),
            },
            FixedTableLookupWord {
                output_word: 37,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_35"),
            },
            FixedTableLookupWord {
                output_word: 38,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_36"),
            },
            FixedTableLookupWord {
                output_word: 39,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_37"),
            },
            FixedTableLookupWord {
                output_word: 40,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_38"),
            },
            FixedTableLookupWord {
                output_word: 41,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_39"),
            },
            FixedTableLookupWord {
                output_word: 42,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_40"),
            },
            FixedTableLookupWord {
                output_word: 43,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_41"),
            },
            FixedTableLookupWord {
                output_word: 44,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_42"),
            },
            FixedTableLookupWord {
                output_word: 45,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_43"),
            },
            FixedTableLookupWord {
                output_word: 46,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_44"),
            },
            FixedTableLookupWord {
                output_word: 47,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_45"),
            },
            FixedTableLookupWord {
                output_word: 48,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_46"),
            },
            FixedTableLookupWord {
                output_word: 49,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_47"),
            },
            FixedTableLookupWord {
                output_word: 50,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_48"),
            },
            FixedTableLookupWord {
                output_word: 51,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_49"),
            },
            FixedTableLookupWord {
                output_word: 52,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_50"),
            },
            FixedTableLookupWord {
                output_word: 53,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_51"),
            },
            FixedTableLookupWord {
                output_word: 54,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_52"),
            },
            FixedTableLookupWord {
                output_word: 55,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_53"),
            },
            FixedTableLookupWord {
                output_word: 56,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_54"),
            },
            FixedTableLookupWord {
                output_word: 57,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_55"),
            },
            FixedTableLookupWord {
                output_word: 58,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "pedersen_points_table_window_bits_9",
        log_size: 15,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1791500038),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_15"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_0"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_1"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_2"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_3"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_4"),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_5"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_6"),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_7"),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_8"),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_9"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_10"),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_11"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_12"),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_13"),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_14"),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_15"),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_16"),
            },
            FixedTableLookupWord {
                output_word: 19,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_17"),
            },
            FixedTableLookupWord {
                output_word: 20,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_18"),
            },
            FixedTableLookupWord {
                output_word: 21,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_19"),
            },
            FixedTableLookupWord {
                output_word: 22,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_20"),
            },
            FixedTableLookupWord {
                output_word: 23,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_21"),
            },
            FixedTableLookupWord {
                output_word: 24,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_22"),
            },
            FixedTableLookupWord {
                output_word: 25,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_23"),
            },
            FixedTableLookupWord {
                output_word: 26,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_24"),
            },
            FixedTableLookupWord {
                output_word: 27,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_25"),
            },
            FixedTableLookupWord {
                output_word: 28,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_26"),
            },
            FixedTableLookupWord {
                output_word: 29,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_27"),
            },
            FixedTableLookupWord {
                output_word: 30,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_28"),
            },
            FixedTableLookupWord {
                output_word: 31,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_29"),
            },
            FixedTableLookupWord {
                output_word: 32,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_30"),
            },
            FixedTableLookupWord {
                output_word: 33,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_31"),
            },
            FixedTableLookupWord {
                output_word: 34,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_32"),
            },
            FixedTableLookupWord {
                output_word: 35,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_33"),
            },
            FixedTableLookupWord {
                output_word: 36,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_34"),
            },
            FixedTableLookupWord {
                output_word: 37,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_35"),
            },
            FixedTableLookupWord {
                output_word: 38,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_36"),
            },
            FixedTableLookupWord {
                output_word: 39,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_37"),
            },
            FixedTableLookupWord {
                output_word: 40,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_38"),
            },
            FixedTableLookupWord {
                output_word: 41,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_39"),
            },
            FixedTableLookupWord {
                output_word: 42,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_40"),
            },
            FixedTableLookupWord {
                output_word: 43,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_41"),
            },
            FixedTableLookupWord {
                output_word: 44,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_42"),
            },
            FixedTableLookupWord {
                output_word: 45,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_43"),
            },
            FixedTableLookupWord {
                output_word: 46,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_44"),
            },
            FixedTableLookupWord {
                output_word: 47,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_45"),
            },
            FixedTableLookupWord {
                output_word: 48,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_46"),
            },
            FixedTableLookupWord {
                output_word: 49,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_47"),
            },
            FixedTableLookupWord {
                output_word: 50,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_48"),
            },
            FixedTableLookupWord {
                output_word: 51,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_49"),
            },
            FixedTableLookupWord {
                output_word: 52,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_50"),
            },
            FixedTableLookupWord {
                output_word: 53,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_51"),
            },
            FixedTableLookupWord {
                output_word: 54,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_52"),
            },
            FixedTableLookupWord {
                output_word: 55,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_53"),
            },
            FixedTableLookupWord {
                output_word: 56,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_54"),
            },
            FixedTableLookupWord {
                output_word: 57,
                source: FixedTableWordSource::PreprocessedColumn("pedersen_points_small_55"),
            },
            FixedTableLookupWord {
                output_word: 58,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "poseidon_round_keys",
        log_size: 6,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1024310512),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_6"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_0"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_1"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_2"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_3"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_4"),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_5"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_6"),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_7"),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_8"),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_9"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_10"),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_11"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_12"),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_13"),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_14"),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_15"),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_16"),
            },
            FixedTableLookupWord {
                output_word: 19,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_17"),
            },
            FixedTableLookupWord {
                output_word: 20,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_18"),
            },
            FixedTableLookupWord {
                output_word: 21,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_19"),
            },
            FixedTableLookupWord {
                output_word: 22,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_20"),
            },
            FixedTableLookupWord {
                output_word: 23,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_21"),
            },
            FixedTableLookupWord {
                output_word: 24,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_22"),
            },
            FixedTableLookupWord {
                output_word: 25,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_23"),
            },
            FixedTableLookupWord {
                output_word: 26,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_24"),
            },
            FixedTableLookupWord {
                output_word: 27,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_25"),
            },
            FixedTableLookupWord {
                output_word: 28,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_26"),
            },
            FixedTableLookupWord {
                output_word: 29,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_27"),
            },
            FixedTableLookupWord {
                output_word: 30,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_28"),
            },
            FixedTableLookupWord {
                output_word: 31,
                source: FixedTableWordSource::PreprocessedColumn("poseidon_round_keys_29"),
            },
            FixedTableLookupWord {
                output_word: 32,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_11",
        log_size: 11,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(991608089),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_11"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_12",
        log_size: 12,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(941275232),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_12"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_18",
        log_size: 18,
        multiplicity_columns: 2,
        trace_columns: &[
            FixedTableTraceColumn {
                output_column: 0,
                multiplicity_column: 0,
            },
            FixedTableTraceColumn {
                output_column: 1,
                multiplicity_column: 1,
            },
        ],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1109051422),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_18"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::Constant(1424798916),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("seq_18"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::MultiplicityColumn(1),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_20",
        log_size: 20,
        multiplicity_columns: 8,
        trace_columns: &[
            FixedTableTraceColumn {
                output_column: 0,
                multiplicity_column: 0,
            },
            FixedTableTraceColumn {
                output_column: 1,
                multiplicity_column: 1,
            },
            FixedTableTraceColumn {
                output_column: 2,
                multiplicity_column: 2,
            },
            FixedTableTraceColumn {
                output_column: 3,
                multiplicity_column: 3,
            },
            FixedTableTraceColumn {
                output_column: 4,
                multiplicity_column: 4,
            },
            FixedTableTraceColumn {
                output_column: 5,
                multiplicity_column: 5,
            },
            FixedTableTraceColumn {
                output_column: 6,
                multiplicity_column: 6,
            },
            FixedTableTraceColumn {
                output_column: 7,
                multiplicity_column: 7,
            },
        ],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1410849886),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::Constant(514232941),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::Constant(531010560),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::Constant(480677703),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::Constant(497455322),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::Constant(447122465),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::Constant(463900084),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::Constant(682009131),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::PreprocessedColumn("seq_20"),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::MultiplicityColumn(1),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::MultiplicityColumn(2),
            },
            FixedTableLookupWord {
                output_word: 19,
                source: FixedTableWordSource::MultiplicityColumn(3),
            },
            FixedTableLookupWord {
                output_word: 20,
                source: FixedTableWordSource::MultiplicityColumn(4),
            },
            FixedTableLookupWord {
                output_word: 21,
                source: FixedTableWordSource::MultiplicityColumn(5),
            },
            FixedTableLookupWord {
                output_word: 22,
                source: FixedTableWordSource::MultiplicityColumn(6),
            },
            FixedTableLookupWord {
                output_word: 23,
                source: FixedTableWordSource::MultiplicityColumn(7),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_3_3_3_3_3",
        log_size: 15,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(502259093),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_3_3_3_3_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_3_3_3_3_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_3_3_3_3_column_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_3_3_3_3_column_3"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_3_3_3_3_column_4"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_3_6_6_3",
        log_size: 18,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1005786011),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_6_6_3_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_6_6_3_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_6_6_3_column_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("range_check_3_6_6_3_column_3"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_4_3",
        log_size: 7,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1567323731),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_3_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_3_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_4_4",
        log_size: 8,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1651211826),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_4_4_4_4",
        log_size: 16,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1027333874),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_4_4_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_4_4_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_4_4_column_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("range_check_4_4_4_4_column_3"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_6",
        log_size: 6,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1185356339),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_6"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_7_2_5",
        log_size: 14,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(371240602),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_7_2_5_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_7_2_5_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("range_check_7_2_5_column_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_8",
        log_size: 8,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(1420243005),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("seq_8"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "range_check_9_9",
        log_size: 18,
        multiplicity_columns: 8,
        trace_columns: &[
            FixedTableTraceColumn {
                output_column: 0,
                multiplicity_column: 0,
            },
            FixedTableTraceColumn {
                output_column: 1,
                multiplicity_column: 1,
            },
            FixedTableTraceColumn {
                output_column: 2,
                multiplicity_column: 2,
            },
            FixedTableTraceColumn {
                output_column: 3,
                multiplicity_column: 3,
            },
            FixedTableTraceColumn {
                output_column: 4,
                multiplicity_column: 4,
            },
            FixedTableTraceColumn {
                output_column: 5,
                multiplicity_column: 5,
            },
            FixedTableTraceColumn {
                output_column: 6,
                multiplicity_column: 6,
            },
            FixedTableTraceColumn {
                output_column: 7,
                multiplicity_column: 7,
            },
        ],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(517791011),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::Constant(1897792095),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::Constant(1881014476),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::Constant(1864236857),
            },
            FixedTableLookupWord {
                output_word: 10,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 11,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 12,
                source: FixedTableWordSource::Constant(1847459238),
            },
            FixedTableLookupWord {
                output_word: 13,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 14,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 15,
                source: FixedTableWordSource::Constant(1830681619),
            },
            FixedTableLookupWord {
                output_word: 16,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 17,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 18,
                source: FixedTableWordSource::Constant(1813904000),
            },
            FixedTableLookupWord {
                output_word: 19,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 20,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 21,
                source: FixedTableWordSource::Constant(2065568285),
            },
            FixedTableLookupWord {
                output_word: 22,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_0"),
            },
            FixedTableLookupWord {
                output_word: 23,
                source: FixedTableWordSource::PreprocessedColumn("range_check_9_9_column_1"),
            },
            FixedTableLookupWord {
                output_word: 24,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
            FixedTableLookupWord {
                output_word: 25,
                source: FixedTableWordSource::MultiplicityColumn(1),
            },
            FixedTableLookupWord {
                output_word: 26,
                source: FixedTableWordSource::MultiplicityColumn(2),
            },
            FixedTableLookupWord {
                output_word: 27,
                source: FixedTableWordSource::MultiplicityColumn(3),
            },
            FixedTableLookupWord {
                output_word: 28,
                source: FixedTableWordSource::MultiplicityColumn(4),
            },
            FixedTableLookupWord {
                output_word: 29,
                source: FixedTableWordSource::MultiplicityColumn(5),
            },
            FixedTableLookupWord {
                output_word: 30,
                source: FixedTableWordSource::MultiplicityColumn(6),
            },
            FixedTableLookupWord {
                output_word: 31,
                source: FixedTableWordSource::MultiplicityColumn(7),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "verify_bitwise_xor_12",
        log_size: 20,
        multiplicity_columns: 16,
        trace_columns: &[
            FixedTableTraceColumn {
                output_column: 0,
                multiplicity_column: 0,
            },
            FixedTableTraceColumn {
                output_column: 1,
                multiplicity_column: 1,
            },
            FixedTableTraceColumn {
                output_column: 2,
                multiplicity_column: 2,
            },
            FixedTableTraceColumn {
                output_column: 3,
                multiplicity_column: 3,
            },
            FixedTableTraceColumn {
                output_column: 4,
                multiplicity_column: 4,
            },
            FixedTableTraceColumn {
                output_column: 5,
                multiplicity_column: 5,
            },
            FixedTableTraceColumn {
                output_column: 6,
                multiplicity_column: 6,
            },
            FixedTableTraceColumn {
                output_column: 7,
                multiplicity_column: 7,
            },
            FixedTableTraceColumn {
                output_column: 8,
                multiplicity_column: 8,
            },
            FixedTableTraceColumn {
                output_column: 9,
                multiplicity_column: 9,
            },
            FixedTableTraceColumn {
                output_column: 10,
                multiplicity_column: 10,
            },
            FixedTableTraceColumn {
                output_column: 11,
                multiplicity_column: 11,
            },
            FixedTableTraceColumn {
                output_column: 12,
                multiplicity_column: 12,
            },
            FixedTableTraceColumn {
                output_column: 13,
                multiplicity_column: 13,
            },
            FixedTableTraceColumn {
                output_column: 14,
                multiplicity_column: 14,
            },
            FixedTableTraceColumn {
                output_column: 15,
                multiplicity_column: 15,
            },
        ],
        lookup: FixedTableLookupLayout::ExpandedXor(ExpandedXorLayout {
            relation_id: 648362599,
            limb_bits: 10,
            expand_bits: 2,
        }),
    },
    FixedTableMaterializationPlan {
        component: "verify_bitwise_xor_4",
        log_size: 8,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(45448144),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_4_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_4_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_4_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "verify_bitwise_xor_7",
        log_size: 14,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(62225763),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_7_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_7_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_7_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "verify_bitwise_xor_8",
        log_size: 16,
        multiplicity_columns: 2,
        trace_columns: &[
            FixedTableTraceColumn {
                output_column: 0,
                multiplicity_column: 0,
            },
            FixedTableTraceColumn {
                output_column: 1,
                multiplicity_column: 1,
            },
        ],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(112558620),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::Constant(521092554),
            },
            FixedTableLookupWord {
                output_word: 5,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_0"),
            },
            FixedTableLookupWord {
                output_word: 6,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_1"),
            },
            FixedTableLookupWord {
                output_word: 7,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_8_2"),
            },
            FixedTableLookupWord {
                output_word: 8,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
            FixedTableLookupWord {
                output_word: 9,
                source: FixedTableWordSource::MultiplicityColumn(1),
            },
        ]),
    },
    FixedTableMaterializationPlan {
        component: "verify_bitwise_xor_9",
        log_size: 18,
        multiplicity_columns: 1,
        trace_columns: &[FixedTableTraceColumn {
            output_column: 0,
            multiplicity_column: 0,
        }],
        lookup: FixedTableLookupLayout::Words(&[
            FixedTableLookupWord {
                output_word: 0,
                source: FixedTableWordSource::Constant(95781001),
            },
            FixedTableLookupWord {
                output_word: 1,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_9_0"),
            },
            FixedTableLookupWord {
                output_word: 2,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_9_1"),
            },
            FixedTableLookupWord {
                output_word: 3,
                source: FixedTableWordSource::PreprocessedColumn("bitwise_xor_9_2"),
            },
            FixedTableLookupWord {
                output_word: 4,
                source: FixedTableWordSource::MultiplicityColumn(0),
            },
        ]),
    },
];
