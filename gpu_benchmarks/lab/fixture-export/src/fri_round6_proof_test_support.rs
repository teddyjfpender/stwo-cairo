use crate::fri_round6_provenance::{PcsShape, ProofShape, PROOF_SHAPE_SCHEMA};

pub(crate) fn sample_shape() -> ProofShape {
    ProofShape {
        schema_version: PROOF_SHAPE_SCHEMA.into(),
        pcs: PcsShape {
            pow_bits: 1,
            log_blowup_factor: 2,
            log_last_layer_degree_bound: 3,
            n_queries: 4,
            fold_step: 1,
            lifting_log_size: Some(8),
        },
        channel_salt: 0,
        preprocessed_trace_variant_sha256: "11".repeat(32),
        component_slots: 1,
        component_enable_bits_sha256: "22".repeat(32),
        component_log_sizes: vec![8],
        trace_column_log_sizes: vec![vec![8]],
        public_data_word_counts: [1, 2, 3],
        interaction_claim_felts: 4,
        commitment_trees: 4,
        sampled_value_counts: vec![vec![1]],
        decommitment_trees: 4,
        queried_value_counts: vec![vec![1]],
        fri_inner_layers: 2,
        fri_witness_counts: vec![3, 2, 1],
        fri_last_layer_coefficients: 8,
        unsorted_query_locations: 4,
    }
}
