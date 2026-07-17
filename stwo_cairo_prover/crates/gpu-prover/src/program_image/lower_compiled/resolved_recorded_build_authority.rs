//! Offline embedded-build authority for one recorded witness kernel.
//!
//! This value projects collision-resistant pack facts into `CompiledProof`
//! build identities. It carries no installed function, CUDA context, stream,
//! module handle, or process-local token and cannot satisfy runtime admission.

use stwo_backend_cuda::aot::{
    AotKernelAbiSchema, AotKernelAuthority, AotKernelModuleGlobals, AotKernelSchemaScope,
};

use super::RecordedWitnessInvocationShape;

const ZERO_IDENTITY: [u8; 32] = [0; 32];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedRecordedBuildAuthority {
    pub(super) manifest_identity: [u8; 32],
    pub(super) program_identity: [u8; 32],
    pub(super) abi_schema: Option<AotKernelAbiSchema>,
    pub(super) abi_schema_identity: [u8; 32],
    pub(super) schema_scope: AotKernelSchemaScope,
    pub(super) kernel_symbol: String,
    pub(super) semantic_hash: u64,
    pub(super) cache_key: u64,
    pub(super) target_sm: u32,
    pub(super) source_identity: [u8; 32],
    pub(super) cubin_identity: [u8; 32],
    pub(super) authority_identity: [u8; 32],
    pub(super) module_globals: AotKernelModuleGlobals,
}

impl ResolvedRecordedBuildAuthority {
    pub(super) fn from_embedded(manifest_identity: [u8; 32], kernel: AotKernelAuthority) -> Self {
        Self {
            manifest_identity,
            program_identity: kernel.program_identity(),
            abi_schema: kernel.abi_schema(),
            abi_schema_identity: kernel.abi_schema_identity(),
            schema_scope: kernel.schema_scope(),
            kernel_symbol: kernel.kernel_symbol().into(),
            semantic_hash: kernel.semantic_hash(),
            cache_key: kernel.cache_key(),
            target_sm: kernel.target_sm(),
            source_identity: kernel.source_identity(),
            cubin_identity: kernel.cubin_identity(),
            authority_identity: kernel.identity(),
            module_globals: kernel.module_globals(),
        }
    }

    pub(super) fn validate(
        &self,
        source: &RecordedWitnessInvocationShape,
        manifest_identity: [u8; 32],
        target_sm: u32,
    ) -> Result<(), ()> {
        if manifest_identity == ZERO_IDENTITY
            || self.manifest_identity != manifest_identity
            || self.program_identity != source.program_identity
            || self.abi_schema != Some(AotKernelAbiSchema::RecordedWitnessV1)
            || self.abi_schema_identity != source.abi_schema_identity
            || self.schema_scope != AotKernelSchemaScope::StructuredAbi
            || self.kernel_symbol != source.kernel_symbol
            || self.semantic_hash != source.semantic_hash
            || self.cache_key != source.cache_key
            || target_sm < 10
            || self.target_sm != target_sm
            || self.source_identity != source.deduce.source_identity
            || self.source_identity == ZERO_IDENTITY
            || self.cubin_identity == ZERO_IDENTITY
            || self.authority_identity == ZERO_IDENTITY
            || self.module_globals
                != if source.deduce.module_state.is_some() {
                    AotKernelModuleGlobals::WitnessPedersenV1
                } else {
                    AotKernelModuleGlobals::None
                }
        {
            return Err(());
        }
        Ok(())
    }
}
