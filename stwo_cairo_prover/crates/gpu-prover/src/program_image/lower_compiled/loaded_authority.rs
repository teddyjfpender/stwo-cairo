//! Loaded-pack cross-check kept separate from source-level invocation mapping.

use stwo_backend_cuda::aot::{self, AotKernelAbiSchema, AotKernelAuthority, AotKernelSchemaScope};

use super::{InvocationShapeError, RecordedWitnessInvocationShape};

fn reject_unpublished_module_state(
    invocation: &RecordedWitnessInvocationShape,
) -> Result<(), InvocationShapeError> {
    if invocation.deduce.module_state.is_some() {
        Err(InvocationShapeError::MissingLoadedModuleStateAuthority)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoadedRecordedWitnessAuthority {
    pub(super) manifest_identity: [u8; 32],
    pub(super) kernel: AotKernelAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedAuthorityFields {
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
}

impl LoadedAuthorityFields {
    fn from_loaded(manifest_identity: [u8; 32], kernel: AotKernelAuthority) -> Self {
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
        }
    }
}

pub(super) fn require(
    invocation: &RecordedWitnessInvocationShape,
    sm_major: u32,
    sm_minor: u32,
) -> Result<LoadedRecordedWitnessAuthority, InvocationShapeError> {
    let manifest_identity = aot::loaded_manifest_identity();
    let kernel = aot::loaded_kernel_authority(invocation.cache_key, sm_major, sm_minor)
        .ok_or(InvocationShapeError::MissingLoadedAotAuthority)?;
    validate_fields(
        invocation,
        sm_major,
        sm_minor,
        &LoadedAuthorityFields::from_loaded(manifest_identity, kernel),
    )?;
    Ok(LoadedRecordedWitnessAuthority {
        manifest_identity,
        kernel,
    })
}

pub(super) fn validate_fields(
    invocation: &RecordedWitnessInvocationShape,
    sm_major: u32,
    sm_minor: u32,
    fields: &LoadedAuthorityFields,
) -> Result<(), InvocationShapeError> {
    // A sealed pack entry proves binary/source identity, not that this specific
    // CUmodule received its process-local Pedersen pointer relocations. Keep the
    // loaded authority unavailable until that publication has its own receipt.
    reject_unpublished_module_state(invocation)?;
    if sm_minor >= 10 {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let target_sm = sm_major
        .checked_mul(10)
        .and_then(|major| major.checked_add(sm_minor))
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if fields.manifest_identity == [0; 32]
        || fields.program_identity != invocation.program_identity
        || fields.abi_schema != Some(AotKernelAbiSchema::RecordedWitnessV1)
        || fields.abi_schema_identity != invocation.abi_schema_identity
        || fields.schema_scope != AotKernelSchemaScope::StructuredAbi
        || fields.kernel_symbol != invocation.kernel_symbol
        || fields.semantic_hash != invocation.semantic_hash
        || fields.cache_key != invocation.cache_key
        || fields.target_sm != target_sm
        || fields.source_identity != invocation.deduce.source_identity
        || fields.cubin_identity == [0; 32]
        || fields.authority_identity == [0; 32]
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_proof::LaunchGeometry;
    use crate::program_image::lower_compiled::recorded_deduce_authority::{
        empty_for_test, PedersenTableColumnsAndRowsV1,
    };

    #[test]
    fn stateful_pack_authority_requires_a_separate_module_publication_receipt() {
        let mut deduce = empty_for_test();
        deduce.module_state = Some(PedersenTableColumnsAndRowsV1::CANONICAL);
        let invocation = RecordedWitnessInvocationShape {
            program_identity: [1; 32],
            semantic_hash: 1,
            cache_key: 1,
            kernel_symbol: "stateful".into(),
            abi_schema_identity: [1; 32],
            deduce,
            launch: LaunchGeometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
            source_arguments: Vec::new(),
        };
        assert_eq!(
            reject_unpublished_module_state(&invocation),
            Err(InvocationShapeError::MissingLoadedModuleStateAuthority)
        );
    }
}
