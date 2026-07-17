//! Loaded-pack and per-CUmodule publication checks, kept out of semantic IR.

use std::sync::OnceLock;

use stwo_backend_cuda::aot::{self, AotKernelAbiSchema, AotKernelAuthority, AotKernelSchemaScope};
use stwo_backend_cuda::pedersen_module_publication::{
    loaded_aot_pedersen_module_publication, PedersenModulePublicationError,
    PedersenModulePublicationReceipt,
};
use stwo_backend_cuda::pedersen_table::{
    RegisteredPedersenTable, PEDERSEN_TABLE_REGISTRATION_GENERATION,
};

use super::{InvocationShapeError, RecordedWitnessInvocationShape};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedRecordedWitnessAuthority {
    pub(super) manifest_identity: [u8; 32],
    pub(super) kernel: AotKernelAuthority,
    _pedersen_receipt: Option<PedersenModulePublicationReceipt>,
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
    device_ordinal: u32,
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
    let pedersen_receipt = invocation
        .deduce
        .module_state
        .map(|state| {
            require_pedersen_publication(
                invocation,
                state,
                device_ordinal,
                sm_major,
                sm_minor,
                kernel,
            )
        })
        .transpose()?;
    Ok(LoadedRecordedWitnessAuthority {
        manifest_identity,
        kernel,
        _pedersen_receipt: pedersen_receipt,
    })
}

pub(super) fn validate_fields(
    invocation: &RecordedWitnessInvocationShape,
    sm_major: u32,
    sm_minor: u32,
    fields: &LoadedAuthorityFields,
) -> Result<(), InvocationShapeError> {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessPedersenBinding {
    device_ordinal: u32,
    context_token: u64,
}

// ReplacementV1's current process-global Cairo table admits exactly one GPU
// and one live CUDA context per process. A fleet worker must use one process
// per GPU until an owning per-device table registry replaces this OnceLock.
static PROCESS_PEDERSEN_BINDING: OnceLock<ProcessPedersenBinding> = OnceLock::new();

fn require_pedersen_publication(
    invocation: &RecordedWitnessInvocationShape,
    state: super::recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
    device_ordinal: u32,
    sm_major: u32,
    sm_minor: u32,
    kernel: AotKernelAuthority,
) -> Result<PedersenModulePublicationReceipt, InvocationShapeError> {
    state.validate_exact()?;
    let table = stwo_cairo_prover::witness::jit_prove_backend::try_ensure_device_pedersen_table()
        .map_err(|_| InvocationShapeError::MissingLoadedModuleStateAuthority)?;
    validate_canonical_table(state, table)?;
    let publication =
        loaded_aot_pedersen_module_publication(invocation.cache_key, sm_major, sm_minor)
            .map_err(map_publication_error)?;
    validate_publication(
        invocation,
        state,
        table,
        kernel,
        device_ordinal,
        &publication,
    )?;
    let binding = ProcessPedersenBinding {
        device_ordinal: publication.device_ordinal(),
        context_token: publication.context_token(),
    };
    if PROCESS_PEDERSEN_BINDING.get_or_init(|| binding) != &binding {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    Ok(publication)
}

fn map_publication_error(error: PedersenModulePublicationError) -> InvocationShapeError {
    match error {
        PedersenModulePublicationError::CudaUnavailable
        | PedersenModulePublicationError::MissingManifestAuthority
        | PedersenModulePublicationError::MissingKernelAuthority
        | PedersenModulePublicationError::RegisteredTableUnavailable
        | PedersenModulePublicationError::NativePublicationUnavailable => {
            InvocationShapeError::MissingLoadedModuleStateAuthority
        }
        PedersenModulePublicationError::InvalidArchitecture
        | PedersenModulePublicationError::InvalidKernelAuthority
        | PedersenModulePublicationError::InvalidKernelSymbol
        | PedersenModulePublicationError::RegisteredTable(_)
        | PedersenModulePublicationError::AddressOverflow
        | PedersenModulePublicationError::NativePublicationMismatch => {
            InvocationShapeError::LoadedModuleStateAuthorityMismatch
        }
    }
}

fn validate_canonical_table(
    state: super::recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
    table: RegisteredPedersenTable,
) -> Result<(), InvocationShapeError> {
    let source_rows = usize::try_from(state.resource.registered_source_rows)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let padded_rows = usize::try_from(state.resource.registered_padded_rows)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let columns =
        usize::try_from(state.resource.columns).map_err(|_| InvocationShapeError::SizeOverflow)?;
    table
        .validate_exact_registration_geometry(table.content_digest(), source_rows, padded_rows)
        .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
    if table.columns().len() != columns
        || table.registration_generation() != PEDERSEN_TABLE_REGISTRATION_GENERATION
    {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    Ok(())
}

fn validate_publication(
    invocation: &RecordedWitnessInvocationShape,
    state: super::recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
    table: RegisteredPedersenTable,
    kernel: AotKernelAuthority,
    device_ordinal: u32,
    receipt: &PedersenModulePublicationReceipt,
) -> Result<(), InvocationShapeError> {
    let published_pointers = receipt.column_pointers();
    let source_rows = usize::try_from(state.resource.registered_source_rows)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let padded_rows = usize::try_from(state.resource.registered_padded_rows)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    if receipt.manifest_identity() != aot::loaded_manifest_identity()
        || receipt.source_identity() != invocation.deduce.source_identity
        || receipt.source_identity() != kernel.source_identity()
        || receipt.cubin_identity() != kernel.cubin_identity()
        || receipt.program_identity() != invocation.program_identity
        || receipt.program_identity() != kernel.program_identity()
        || receipt.abi_schema_identity() != invocation.abi_schema_identity
        || receipt.abi_schema_identity() != kernel.abi_schema_identity()
        || receipt.kernel_authority_identity() != kernel.identity()
        || receipt.kernel_symbol() != invocation.kernel_symbol
        || receipt.semantic_hash() != invocation.semantic_hash
        || receipt.cache_key() != invocation.cache_key
        || receipt.target_sm() != kernel.target_sm()
        || receipt.device_ordinal() != device_ordinal
        || receipt.table_content_digest() != table.content_digest()
        || receipt.table_source_rows() != table.source_n_rows()
        || receipt.table_padded_rows() != table.n_rows()
        || receipt.table_registration_generation() != table.registration_generation()
        || receipt.table_source_rows() != source_rows
        || receipt.table_padded_rows() != padded_rows
        || receipt.columns_symbol_bytes() != state.column_pointers.symbol_bytes
        || receipt.rows_symbol_bytes() != state.row_count.symbol_bytes
        || receipt.module_token() == 0
        || receipt.context_token() == 0
        || receipt.completion_event_token() == 0
        || table
            .columns()
            .iter()
            .zip(published_pointers)
            .any(|(column, published)| {
                u64::try_from(column.as_u32_ptr() as usize).ok() != Some(published)
            })
    {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
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
    fn publication_absence_and_drift_remain_distinct_fail_closed_states() {
        assert_eq!(
            map_publication_error(PedersenModulePublicationError::NativePublicationUnavailable),
            InvocationShapeError::MissingLoadedModuleStateAuthority
        );
        assert_eq!(
            map_publication_error(PedersenModulePublicationError::NativePublicationMismatch),
            InvocationShapeError::LoadedModuleStateAuthorityMismatch
        );
    }

    #[test]
    fn stateful_semantics_do_not_fabricate_a_loaded_module_receipt() {
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
        let fields = LoadedAuthorityFields {
            manifest_identity: [2; 32],
            program_identity: invocation.program_identity,
            abi_schema: Some(AotKernelAbiSchema::RecordedWitnessV1),
            abi_schema_identity: invocation.abi_schema_identity,
            schema_scope: AotKernelSchemaScope::StructuredAbi,
            kernel_symbol: invocation.kernel_symbol.clone(),
            semantic_hash: invocation.semantic_hash,
            cache_key: invocation.cache_key,
            target_sm: 89,
            source_identity: invocation.deduce.source_identity,
            cubin_identity: [3; 32],
            authority_identity: [4; 32],
        };
        validate_fields(&invocation, 8, 9, &fields).unwrap();
        if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT {
            assert_eq!(
                require(&invocation, 0, 8, 9),
                Err(InvocationShapeError::MissingLoadedAotAuthority)
            );
        }
    }
}
