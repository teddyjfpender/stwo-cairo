//! Runtime admission for the exact installed recorded-witness function.
//!
//! Semantic lowering carries no process token. Runtime admission starts from
//! the prepared writer that owns the live `InstalledAotFunction`, consumes its
//! immutable receipt, and retains a clone only after every semantic, launch,
//! device, stream and nested-publication fact matches.

use stwo_backend_cuda::aot::{
    AotKernelAbiSchema, AotKernelModuleGlobals, InstalledAotFunctionOwnership,
    InstalledAotFunctionReceipt,
};
use stwo_backend_cuda::pedersen_table::{
    registered_borrowed_pedersen_table, PedersenTableContentDigest, RegisteredPedersenTable,
    PEDERSEN_TABLE_REGISTRATION_GENERATION,
};
use stwo_backend_cuda::{
    DeviceArena, PreparedWitnessGraph, PreparedWitnessMode, WitnessKernelIdentity,
};

use super::loaded_writer_binding::{
    validate_writer_binding, WriterBindingFields, WriterBindingPlan,
};
use super::{
    BaseProducerCatalog, InvocationShapeError, RecordedWitnessInvocationShape, SourceArgument,
};
use crate::arena_plan::PlannedWitnessComponent;

const ZERO_IDENTITY: [u8; 32] = [0; 32];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedRecordedWitnessAuthority {
    _installed_receipt: InstalledAotFunctionReceipt,
}

pub(super) fn require_prepared(
    invocation: &RecordedWitnessInvocationShape,
    catalog: &BaseProducerCatalog,
    planned: &PlannedWitnessComponent,
    arena: &DeviceArena,
    writer: &PreparedWitnessGraph<'_>,
    device_ordinal: u32,
    sm_major: u32,
    sm_minor: u32,
) -> Result<LoadedRecordedWitnessAuthority, InvocationShapeError> {
    let row_count = recorded_row_count(invocation)?;
    if planned.program.label != planned.component {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let admission = WriterAdmissionFields::from_writer(arena, writer);
    validate_writer_admission(invocation, planned.component, row_count, &admission)?;
    let planned_binding = WriterBindingPlan::from_planned(planned)?;
    if planned_binding != WriterBindingPlan::from_invocation(invocation, catalog)? {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    let expected_binding = WriterBindingFields::from_plan(arena, &planned_binding)?;
    let actual_binding = WriterBindingFields::from_writer(arena, writer)?;
    validate_writer_binding(&expected_binding, &actual_binding)?;
    let receipt = writer
        .installed_aot_receipt()
        .ok_or(InvocationShapeError::MissingLoadedAotAuthority)?;
    let exec_context_token = arena.exec_context_token();
    let stream_token = u64::try_from(arena.context().stream_raw().as_ptr() as usize)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let canonical_pedersen = invocation
        .deduce
        .module_state
        .map(canonical_pedersen_fields)
        .transpose()?;
    validate_installed_receipt(
        invocation,
        admission.identity.aot_manifest_identity,
        device_ordinal,
        sm_major,
        sm_minor,
        exec_context_token,
        stream_token,
        &InstalledReceiptFields::from(receipt),
        canonical_pedersen.as_ref(),
    )?;
    Ok(LoadedRecordedWitnessAuthority {
        _installed_receipt: receipt.clone(),
    })
}

fn target_sm(sm_major: u32, sm_minor: u32) -> Result<u32, InvocationShapeError> {
    if sm_major == 0 || sm_minor >= 10 {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    sm_major
        .checked_mul(10)
        .and_then(|major| major.checked_add(sm_minor))
        .ok_or(InvocationShapeError::SizeOverflow)
}

fn recorded_row_count(
    invocation: &RecordedWitnessInvocationShape,
) -> Result<usize, InvocationShapeError> {
    if invocation.source_arguments.len() != 8 {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    match invocation.source_arguments.get(7) {
        Some(SourceArgument::U32 { ordinal: 7, value }) if *value != 0 => {
            usize::try_from(*value).map_err(|_| InvocationShapeError::SizeOverflow)
        }
        _ => Err(InvocationShapeError::LoadedAotAuthorityMismatch),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedWriterKind {
    Recorded,
    BlakeGFused,
    BlakeGDirect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WriterIdentityFields {
    label: String,
    kernel_name: String,
    semantic_hash: u64,
    cache_key: u64,
    aot_manifest_identity: [u8; 32],
    mode: PreparedWitnessMode,
}

impl From<&WitnessKernelIdentity> for WriterIdentityFields {
    fn from(identity: &WitnessKernelIdentity) -> Self {
        Self {
            label: identity.label.clone(),
            kernel_name: identity.kernel_name.clone(),
            semantic_hash: identity.semantic_hash,
            cache_key: identity.cache_key,
            aot_manifest_identity: identity.aot_manifest_identity,
            mode: identity.mode,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WriterAdmissionFields {
    belongs_to_arena: bool,
    kind: PreparedWriterKind,
    row_count: usize,
    identity: WriterIdentityFields,
    has_installed_receipt: bool,
}

impl WriterAdmissionFields {
    fn from_writer(arena: &DeviceArena, writer: &PreparedWitnessGraph<'_>) -> Self {
        let kind = if writer.is_blake_g_direct() {
            PreparedWriterKind::BlakeGDirect
        } else if writer.is_blake_g_fused() {
            PreparedWriterKind::BlakeGFused
        } else {
            PreparedWriterKind::Recorded
        };
        Self {
            belongs_to_arena: writer.belongs_to(arena),
            kind,
            row_count: writer.row_count(),
            identity: writer.kernel_identity().into(),
            has_installed_receipt: writer.installed_aot_receipt().is_some(),
        }
    }
}

fn validate_writer_admission(
    invocation: &RecordedWitnessInvocationShape,
    expected_label: &str,
    row_count: usize,
    writer: &WriterAdmissionFields,
) -> Result<(), InvocationShapeError> {
    if !writer.belongs_to_arena
        || writer.kind != PreparedWriterKind::Recorded
        || writer.row_count != row_count
        || writer.identity.mode != PreparedWitnessMode::RequireEmbeddedAot
        || writer.identity.aot_manifest_identity == ZERO_IDENTITY
        || writer.identity.label != expected_label
        || writer.identity.kernel_name != invocation.kernel_symbol
        || writer.identity.semantic_hash != invocation.semantic_hash
        || writer.identity.cache_key != invocation.cache_key
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    if !writer.has_installed_receipt {
        return Err(InvocationShapeError::MissingLoadedAotAuthority);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FunctionPublicationFields {
    manifest_identity: [u8; 32],
    source_identity: [u8; 32],
    cubin_identity: [u8; 32],
    program_identity: [u8; 32],
    abi_schema_identity: [u8; 32],
    authority_identity: [u8; 32],
    kernel_symbol: String,
    semantic_hash: u64,
    cache_key: u64,
    target_sm: u32,
    device_ordinal: u32,
    driver_context_token: u64,
    module_token: u64,
    function_token: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PedersenPublicationFields {
    manifest_identity: [u8; 32],
    source_identity: [u8; 32],
    cubin_identity: [u8; 32],
    program_identity: [u8; 32],
    abi_schema_identity: [u8; 32],
    authority_identity: [u8; 32],
    kernel_symbol: String,
    semantic_hash: u64,
    cache_key: u64,
    target_sm: u32,
    table_content_digest: PedersenTableContentDigest,
    table_source_rows: usize,
    table_padded_rows: usize,
    table_registration_generation: u64,
    device_ordinal: u32,
    module_token: u64,
    function_token: u64,
    context_token: u64,
    columns_symbol_token: u64,
    columns_symbol_bytes: u32,
    rows_symbol_token: u64,
    rows_symbol_bytes: u32,
    completion_event_token: u64,
    column_pointers: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InstalledReceiptFields {
    manifest_identity: [u8; 32],
    source_identity: [u8; 32],
    cubin_identity: [u8; 32],
    program_identity: [u8; 32],
    abi_schema_identity: [u8; 32],
    authority_identity: [u8; 32],
    kernel_symbol: String,
    semantic_hash: u64,
    cache_key: u64,
    target_sm: u32,
    abi_schema: AotKernelAbiSchema,
    module_globals: AotKernelModuleGlobals,
    ownership: InstalledAotFunctionOwnership,
    launch_grid: [u32; 3],
    launch_block: [u32; 3],
    dynamic_shared_bytes: u32,
    device_ordinal: u32,
    exec_context_token: u64,
    driver_context_token: u64,
    module_token: u64,
    function_token: u64,
    stream_token: u64,
    function_publication: FunctionPublicationFields,
    pedersen_publication: Option<PedersenPublicationFields>,
}

impl From<&InstalledAotFunctionReceipt> for InstalledReceiptFields {
    fn from(receipt: &InstalledAotFunctionReceipt) -> Self {
        let function = receipt.function_publication();
        let launch = receipt.launch();
        Self {
            manifest_identity: receipt.manifest_identity(),
            source_identity: receipt.source_identity(),
            cubin_identity: receipt.cubin_identity(),
            program_identity: receipt.program_identity(),
            abi_schema_identity: receipt.abi_schema_identity(),
            authority_identity: receipt.kernel_authority_identity(),
            kernel_symbol: receipt.kernel_symbol().into(),
            semantic_hash: receipt.semantic_hash(),
            cache_key: receipt.cache_key(),
            target_sm: receipt.target_sm(),
            abi_schema: receipt.abi_schema(),
            module_globals: receipt.module_globals(),
            ownership: receipt.ownership(),
            launch_grid: launch.grid(),
            launch_block: launch.block(),
            dynamic_shared_bytes: launch.dynamic_shared_bytes(),
            device_ordinal: receipt.device_ordinal(),
            exec_context_token: receipt.exec_context_token(),
            driver_context_token: receipt.driver_context_token(),
            module_token: receipt.module_token(),
            function_token: receipt.function_token(),
            stream_token: receipt.stream_token(),
            function_publication: FunctionPublicationFields {
                manifest_identity: function.manifest_identity(),
                source_identity: function.source_identity(),
                cubin_identity: function.cubin_identity(),
                program_identity: function.program_identity(),
                abi_schema_identity: function.abi_schema_identity(),
                authority_identity: function.kernel_authority_identity(),
                kernel_symbol: function.kernel_symbol().into(),
                semantic_hash: function.semantic_hash(),
                cache_key: function.cache_key(),
                target_sm: function.target_sm(),
                device_ordinal: function.device_ordinal(),
                driver_context_token: function.driver_context_token(),
                module_token: function.module_token(),
                function_token: function.function_token(),
            },
            pedersen_publication: receipt
                .pedersen_publication()
                .map(PedersenPublicationFields::from),
        }
    }
}

impl From<&stwo_backend_cuda::pedersen_module_publication::PedersenModulePublicationReceipt>
    for PedersenPublicationFields
{
    fn from(
        receipt: &stwo_backend_cuda::pedersen_module_publication::PedersenModulePublicationReceipt,
    ) -> Self {
        Self {
            manifest_identity: receipt.manifest_identity(),
            source_identity: receipt.source_identity(),
            cubin_identity: receipt.cubin_identity(),
            program_identity: receipt.program_identity(),
            abi_schema_identity: receipt.abi_schema_identity(),
            authority_identity: receipt.kernel_authority_identity(),
            kernel_symbol: receipt.kernel_symbol().into(),
            semantic_hash: receipt.semantic_hash(),
            cache_key: receipt.cache_key(),
            target_sm: receipt.target_sm(),
            table_content_digest: receipt.table_content_digest(),
            table_source_rows: receipt.table_source_rows(),
            table_padded_rows: receipt.table_padded_rows(),
            table_registration_generation: receipt.table_registration_generation(),
            device_ordinal: receipt.device_ordinal(),
            module_token: receipt.module_token(),
            function_token: receipt.function_token(),
            context_token: receipt.context_token(),
            columns_symbol_token: receipt.columns_symbol_token(),
            columns_symbol_bytes: receipt.columns_symbol_bytes(),
            rows_symbol_token: receipt.rows_symbol_token(),
            rows_symbol_bytes: receipt.rows_symbol_bytes(),
            completion_event_token: receipt.completion_event_token(),
            column_pointers: receipt.column_pointers().to_vec(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CanonicalPedersenFields {
    content_digest: PedersenTableContentDigest,
    source_rows: usize,
    padded_rows: usize,
    registration_generation: u64,
    column_pointers: Vec<u64>,
}

fn canonical_pedersen_fields(
    state: super::recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
) -> Result<CanonicalPedersenFields, InvocationShapeError> {
    state.validate_exact()?;
    let registered = registered_borrowed_pedersen_table()
        .ok_or(InvocationShapeError::MissingLoadedModuleStateAuthority)?;
    let table = stwo_cairo_prover::witness::jit_prove_backend::try_ensure_device_pedersen_table()
        .map_err(|_| InvocationShapeError::MissingLoadedModuleStateAuthority)?;
    if table != registered {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    validate_canonical_table(state, table)?;
    let column_pointers = table
        .columns()
        .iter()
        .map(|column| {
            u64::try_from(column.as_u32_ptr() as usize)
                .map_err(|_| InvocationShapeError::SizeOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CanonicalPedersenFields {
        content_digest: table.content_digest(),
        source_rows: table.source_n_rows(),
        padded_rows: table.n_rows(),
        registration_generation: table.registration_generation(),
        column_pointers,
    })
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

#[allow(clippy::too_many_arguments)]
fn validate_installed_receipt(
    invocation: &RecordedWitnessInvocationShape,
    writer_manifest_identity: [u8; 32],
    device_ordinal: u32,
    sm_major: u32,
    sm_minor: u32,
    exec_context_token: u64,
    stream_token: u64,
    receipt: &InstalledReceiptFields,
    canonical_pedersen: Option<&CanonicalPedersenFields>,
) -> Result<(), InvocationShapeError> {
    let target_sm = target_sm(sm_major, sm_minor)?;
    if invocation.launch.cluster.is_some()
        || invocation.launch.cooperative
        || invocation.launch.grid.contains(&0)
        || invocation.launch.block.contains(&0)
        || writer_manifest_identity == ZERO_IDENTITY
        || receipt.manifest_identity != writer_manifest_identity
        || receipt.source_identity != invocation.deduce.source_identity
        || receipt.source_identity == ZERO_IDENTITY
        || receipt.cubin_identity == ZERO_IDENTITY
        || receipt.program_identity != invocation.program_identity
        || receipt.abi_schema_identity != invocation.abi_schema_identity
        || receipt.authority_identity == ZERO_IDENTITY
        || receipt.kernel_symbol != invocation.kernel_symbol
        || receipt.semantic_hash != invocation.semantic_hash
        || receipt.cache_key != invocation.cache_key
        || receipt.target_sm != target_sm
        || receipt.abi_schema != AotKernelAbiSchema::RecordedWitnessV1
        || receipt.module_globals != expected_module_globals(invocation)
        || receipt.ownership != InstalledAotFunctionOwnership::BorrowedPublished
        || receipt.launch_grid != invocation.launch.grid
        || receipt.launch_block != invocation.launch.block
        || receipt.dynamic_shared_bytes != invocation.launch.dynamic_shared_bytes
        || receipt.device_ordinal != device_ordinal
        || receipt.exec_context_token != exec_context_token
        || exec_context_token == 0
        || receipt.driver_context_token == 0
        || receipt.module_token == 0
        || receipt.function_token == 0
        || receipt.stream_token != stream_token
        || stream_token == 0
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    validate_function_publication(receipt)?;
    validate_pedersen_publication(invocation, receipt, canonical_pedersen)
}

fn validate_function_publication(
    receipt: &InstalledReceiptFields,
) -> Result<(), InvocationShapeError> {
    let function = &receipt.function_publication;
    if function.manifest_identity != receipt.manifest_identity
        || function.source_identity != receipt.source_identity
        || function.cubin_identity != receipt.cubin_identity
        || function.program_identity != receipt.program_identity
        || function.abi_schema_identity != receipt.abi_schema_identity
        || function.authority_identity != receipt.authority_identity
        || function.kernel_symbol != receipt.kernel_symbol
        || function.semantic_hash != receipt.semantic_hash
        || function.cache_key != receipt.cache_key
        || function.target_sm != receipt.target_sm
        || function.device_ordinal != receipt.device_ordinal
        || function.driver_context_token != receipt.driver_context_token
        || function.module_token != receipt.module_token
        || function.function_token != receipt.function_token
    {
        return Err(InvocationShapeError::LoadedAotAuthorityMismatch);
    }
    Ok(())
}

fn validate_pedersen_publication(
    invocation: &RecordedWitnessInvocationShape,
    receipt: &InstalledReceiptFields,
    canonical: Option<&CanonicalPedersenFields>,
) -> Result<(), InvocationShapeError> {
    let Some(state) = invocation.deduce.module_state else {
        return if receipt.pedersen_publication.is_none() && canonical.is_none() {
            Ok(())
        } else {
            Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
        };
    };
    state.validate_exact()?;
    let publication = receipt
        .pedersen_publication
        .as_ref()
        .ok_or(InvocationShapeError::MissingLoadedModuleStateAuthority)?;
    let canonical = canonical.ok_or(InvocationShapeError::MissingLoadedModuleStateAuthority)?;
    if publication.manifest_identity != receipt.manifest_identity
        || publication.source_identity != receipt.source_identity
        || publication.cubin_identity != receipt.cubin_identity
        || publication.program_identity != receipt.program_identity
        || publication.abi_schema_identity != receipt.abi_schema_identity
        || publication.authority_identity != receipt.authority_identity
        || publication.kernel_symbol != receipt.kernel_symbol
        || publication.semantic_hash != receipt.semantic_hash
        || publication.cache_key != receipt.cache_key
        || publication.target_sm != receipt.target_sm
        || publication.device_ordinal != receipt.device_ordinal
        || publication.module_token != receipt.module_token
        || publication.function_token != receipt.function_token
        || publication.context_token != receipt.driver_context_token
        || publication.table_content_digest != canonical.content_digest
        || publication.table_source_rows != canonical.source_rows
        || publication.table_padded_rows != canonical.padded_rows
        || publication.table_registration_generation != canonical.registration_generation
        || publication.table_source_rows
            != usize::try_from(state.resource.registered_source_rows)
                .map_err(|_| InvocationShapeError::SizeOverflow)?
        || publication.table_padded_rows
            != usize::try_from(state.resource.registered_padded_rows)
                .map_err(|_| InvocationShapeError::SizeOverflow)?
        || publication.columns_symbol_bytes != state.column_pointers.symbol_bytes
        || publication.rows_symbol_bytes != state.row_count.symbol_bytes
        || publication.columns_symbol_token == 0
        || publication.rows_symbol_token == 0
        || publication.columns_symbol_token == publication.rows_symbol_token
        || publication.completion_event_token == 0
        || publication.column_pointers != canonical.column_pointers
    {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    Ok(())
}

fn expected_module_globals(invocation: &RecordedWitnessInvocationShape) -> AotKernelModuleGlobals {
    if invocation.deduce.module_state.is_some() {
        AotKernelModuleGlobals::WitnessPedersenV1
    } else {
        AotKernelModuleGlobals::None
    }
}

#[cfg(test)]
#[path = "loaded_authority_tests.rs"]
mod tests;
