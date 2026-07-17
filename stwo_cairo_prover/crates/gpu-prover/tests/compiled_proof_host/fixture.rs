use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    Blake2sFriAssemblyShape, Blake2sProofAssemblyShape, Blake2sTraceAssemblyShape, TraceTreeRole,
};

use super::*;

pub(super) fn pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 0,
        fri_config: FriConfig::new(2, 1, 13, 2),
        lifting_log_size: Some(10),
    }
}

pub(super) fn proof_assembly_shape() -> Blake2sProofAssemblyShape {
    let trace = |role, log_size, samples| Blake2sTraceAssemblyShape {
        role,
        leaf_log_size: log_size,
        query_log_size: log_size,
        oods_samples_per_column: vec![samples],
        commit_to_proof_column: vec![0],
    };
    Blake2sProofAssemblyShape {
        query_log_size: 10,
        n_queries: 13,
        trace_trees: vec![
            trace(TraceTreeRole::Preprocessed, 8, 2),
            trace(TraceTreeRole::Base, 10, 2),
            trace(TraceTreeRole::Interaction, 10, 2),
            trace(TraceTreeRole::Composition, 10, 1),
        ],
        fri_trees: vec![
            Blake2sFriAssemblyShape {
                evaluation_log_size: 10,
                cumulative_fold: 0,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 8,
                cumulative_fold: 2,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 6,
                cumulative_fold: 4,
                outgoing_fold_step: 2,
                log_rows_per_leaf: 2,
            },
            Blake2sFriAssemblyShape {
                evaluation_log_size: 4,
                cumulative_fold: 6,
                outgoing_fold_step: 1,
                log_rows_per_leaf: 0,
            },
        ],
    }
}

pub(super) fn host_finalizer(
    identity: &ProofIdentity,
    codec: ProofCodecIdentity,
) -> HostFinalizerAuthority {
    host_finalizer_with_shape(identity, codec, proof_assembly_shape())
}

pub(super) fn host_finalizer_with_shape(
    identity: &ProofIdentity,
    codec: ProofCodecIdentity,
    shape: Blake2sProofAssemblyShape,
) -> HostFinalizerAuthority {
    host_finalizer_with_pcs(identity, codec, shape, pcs_config())
}

pub(super) fn host_finalizer_with_pcs(
    identity: &ProofIdentity,
    codec: ProofCodecIdentity,
    shape: Blake2sProofAssemblyShape,
    pcs: PcsConfig,
) -> HostFinalizerAuthority {
    HostFinalizerAuthority::new(HostFinalizerAuthorityInput {
        bundle_codec: codec,
        assembly_shape: shape,
        pcs,
        claim_codec: ClaimCodecIdentity::new(b"cairo-claim-codec-v1".to_vec()).unwrap(),
        interaction_claim_codec: InteractionClaimCodecIdentity::new(
            b"cairo-interaction-claim-codec-v1".to_vec(),
        )
        .unwrap(),
        channel_schema: ChannelSchemaIdentity::new(b"blake2s-channel-schema-v1".to_vec()).unwrap(),
        preprocessed_schema: PreprocessedSchemaIdentity::new(
            b"cairo-preprocessed-schema-v1".to_vec(),
        )
        .unwrap(),
        decoder: DirectProofDecoder::ResidentBlake2sV1,
        oods_recipe: OodsConsistencyRecipe::CairoComponentsV1,
        envelope: CairoProofEnvelope::CairoProofV1,
        proof_semantic_digest: *identity.proof_semantic_digest(),
        execution_build_digest: *identity.execution_build_digest(),
    })
}
