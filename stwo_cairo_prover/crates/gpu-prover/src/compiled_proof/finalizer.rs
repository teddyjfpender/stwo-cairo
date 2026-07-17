use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::Blake2sProofAssemblyShape;

use super::*;

macro_rules! finalizer_identity {
    ($name:ident, $domain:literal) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name {
            canonical_encoding: Box<[u8]>,
            digest: [u8; 32],
        }

        impl $name {
            pub fn new(canonical_encoding: Vec<u8>) -> Result<Self, CompiledProofError> {
                if canonical_encoding.is_empty() {
                    return Err(CompiledProofError::InvalidHostFinalizer);
                }
                let mut hasher = blake3::Hasher::new();
                hasher.update($domain);
                hasher.update(&(canonical_encoding.len() as u64).to_le_bytes());
                hasher.update(&canonical_encoding);
                Ok(Self {
                    canonical_encoding: canonical_encoding.into_boxed_slice(),
                    digest: *hasher.finalize().as_bytes(),
                })
            }

            pub fn canonical_encoding(&self) -> &[u8] {
                &self.canonical_encoding
            }

            pub const fn digest(&self) -> &[u8; 32] {
                &self.digest
            }
        }
    };
}

finalizer_identity!(
    ClaimCodecIdentity,
    b"stwo-cairo.compiled-proof.claim-codec.v1\0"
);
finalizer_identity!(
    InteractionClaimCodecIdentity,
    b"stwo-cairo.compiled-proof.interaction-claim-codec.v1\0"
);
finalizer_identity!(
    ChannelSchemaIdentity,
    b"stwo-cairo.compiled-proof.channel-schema.v1\0"
);
finalizer_identity!(
    PreprocessedSchemaIdentity,
    b"stwo-cairo.compiled-proof.preprocessed-schema.v1\0"
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectProofDecoder {
    ResidentBlake2sV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OodsConsistencyRecipe {
    CairoComponentsV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CairoProofEnvelope {
    CairoProofV1,
}

/// Exact authority for the only admitted host tail. It can decode, check OODS
/// consistency and wrap the canonical bundle; it cannot run protocol work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostFinalizerAuthority {
    bundle_codec: ProofCodecIdentity,
    assembly_shape: Blake2sProofAssemblyShape,
    pcs: PcsConfig,
    claim_codec: ClaimCodecIdentity,
    interaction_claim_codec: InteractionClaimCodecIdentity,
    channel_schema: ChannelSchemaIdentity,
    preprocessed_schema: PreprocessedSchemaIdentity,
    decoder: DirectProofDecoder,
    oods_recipe: OodsConsistencyRecipe,
    envelope: CairoProofEnvelope,
    proof_semantic_digest: [u8; 32],
    execution_build_digest: [u8; 32],
}

pub struct HostFinalizerAuthorityInput {
    pub bundle_codec: ProofCodecIdentity,
    pub assembly_shape: Blake2sProofAssemblyShape,
    pub pcs: PcsConfig,
    pub claim_codec: ClaimCodecIdentity,
    pub interaction_claim_codec: InteractionClaimCodecIdentity,
    pub channel_schema: ChannelSchemaIdentity,
    pub preprocessed_schema: PreprocessedSchemaIdentity,
    pub decoder: DirectProofDecoder,
    pub oods_recipe: OodsConsistencyRecipe,
    pub envelope: CairoProofEnvelope,
    pub proof_semantic_digest: [u8; 32],
    pub execution_build_digest: [u8; 32],
}

impl HostFinalizerAuthority {
    pub fn new(input: HostFinalizerAuthorityInput) -> Self {
        let HostFinalizerAuthorityInput {
            bundle_codec,
            assembly_shape,
            pcs,
            claim_codec,
            interaction_claim_codec,
            channel_schema,
            preprocessed_schema,
            decoder,
            oods_recipe,
            envelope,
            proof_semantic_digest,
            execution_build_digest,
        } = input;
        Self {
            bundle_codec,
            assembly_shape,
            pcs,
            claim_codec,
            interaction_claim_codec,
            channel_schema,
            preprocessed_schema,
            decoder,
            oods_recipe,
            envelope,
            proof_semantic_digest,
            execution_build_digest,
        }
    }

    pub const fn bundle_codec(&self) -> &ProofCodecIdentity {
        &self.bundle_codec
    }

    pub const fn assembly_shape(&self) -> &Blake2sProofAssemblyShape {
        &self.assembly_shape
    }

    pub const fn pcs(&self) -> PcsConfig {
        self.pcs
    }

    pub const fn claim_codec(&self) -> &ClaimCodecIdentity {
        &self.claim_codec
    }

    pub const fn interaction_claim_codec(&self) -> &InteractionClaimCodecIdentity {
        &self.interaction_claim_codec
    }

    pub const fn channel_schema(&self) -> &ChannelSchemaIdentity {
        &self.channel_schema
    }

    pub const fn preprocessed_schema(&self) -> &PreprocessedSchemaIdentity {
        &self.preprocessed_schema
    }

    pub const fn decoder(&self) -> DirectProofDecoder {
        self.decoder
    }

    pub const fn oods_recipe(&self) -> OodsConsistencyRecipe {
        self.oods_recipe
    }

    pub const fn envelope(&self) -> CairoProofEnvelope {
        self.envelope
    }

    pub const fn proof_semantic_digest(&self) -> &[u8; 32] {
        &self.proof_semantic_digest
    }

    pub const fn execution_build_digest(&self) -> &[u8; 32] {
        &self.execution_build_digest
    }
}
