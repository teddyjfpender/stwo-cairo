//! Installed authority and dispatch for ordinary composition stripes.
//!
//! The candidate keeps the generated kernel ABI unchanged. Eager launches use
//! the exact installed `CUfunction`; capture uses the legacy wrapper only after
//! checking the same retained receipt, with a null source under strict AOT.

use super::*;
#[cfg(feature = "direct-retention-test-api")]
use crate::composition_stripes::{
    COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD, COMPOSITION_STRIPE_THREADS_PER_BLOCK,
};

pub(super) struct PreparedStripeLaunch {
    pub trace_values: *const u32,
    pub interaction_offsets: *const u32,
    pub base_params: *const u32,
    pub ext_params: *const u32,
    pub random_coefficient_powers: *const u32,
    pub denominator_inverses: *const u32,
    pub coordinates: [*mut u32; SECURE_COORDINATES],
    pub row_count: u32,
    pub log_n_rows: u32,
    pub rc_base: u32,
}

pub(super) fn prepare_aot_kernel<'a>(
    arena: &'a DeviceArena,
    component: usize,
    kernel_index: usize,
    row_count: usize,
    kernel: &CompositionKernelPart,
    admission: CompositionStripeAdmission,
) -> Result<PreparedKernel<'a>, PreparedCompositionError> {
    #[cfg(not(feature = "direct-retention-test-api"))]
    let _ = (arena, row_count);
    let source = CString::new(kernel.source.as_bytes()).map_err(|_| {
        PreparedCompositionError::KernelSourceContainsNul {
            component,
            kernel: kernel_index,
        }
    })?;
    let source_identity = aot::emitted_source_identity(&kernel.source);
    let name = CString::new(kernel.kernel_name.as_bytes()).map_err(|_| {
        PreparedCompositionError::KernelNameContainsNul {
            component,
            kernel: kernel_index,
        }
    })?;
    let found = unsafe {
        raw::stwo_cuda_jit_precompile(source.as_ptr(), name.as_ptr(), kernel.cache_key, false)
    };
    if !found {
        return Err(PreparedCompositionError::AotKernelMiss {
            component,
            kernel: kernel_index,
            cache_key: kernel.cache_key,
        });
    }
    let installed = match admission {
        CompositionStripeAdmission::Wrapper => None,
        #[cfg(feature = "direct-retention-test-api")]
        CompositionStripeAdmission::InstalledResourceBounded => Some(install_resource_bounded(
            arena,
            component,
            kernel_index,
            u32::try_from(row_count).map_err(|_| PreparedCompositionError::SizeOverflow)?,
            kernel,
            source_identity,
        )?),
    };
    Ok(PreparedKernel {
        source,
        source_identity,
        name,
        cache_key: kernel.cache_key,
        semantic_hash: kernel.semantic_hash,
        rc_base: kernel.rc_base,
        installed,
    })
}

pub(super) fn enqueue_prepared_stripe(
    arena: &DeviceArena,
    component: usize,
    kernel_index: usize,
    kernel: &PreparedKernel<'_>,
    stream: *mut c_void,
    dispatch: CompositionStripeDispatch,
    launch: PreparedStripeLaunch,
) -> Result<(), PreparedCompositionError> {
    match dispatch {
        CompositionStripeDispatch::Wrapper => launch_wrapper(
            component,
            kernel_index,
            kernel,
            stream,
            launch,
            kernel.source.as_ptr(),
        ),
        #[cfg(feature = "direct-retention-test-api")]
        CompositionStripeDispatch::EagerInstalled => {
            require_resource_bounded(arena, component, kernel_index, kernel, stream, &launch)?;
            launch_installed(arena, component, kernel_index, kernel, launch)
        }
        #[cfg(feature = "direct-retention-test-api")]
        CompositionStripeDispatch::CaptureSafe => {
            require_resource_bounded(arena, component, kernel_index, kernel, stream, &launch)?;
            launch_wrapper(
                component,
                kernel_index,
                kernel,
                stream,
                launch,
                core::ptr::null(),
            )
        }
    }
}

fn launch_wrapper(
    component: usize,
    kernel_index: usize,
    kernel: &PreparedKernel<'_>,
    stream: *mut c_void,
    launch: PreparedStripeLaunch,
    source: *const core::ffi::c_char,
) -> Result<(), PreparedCompositionError> {
    let [coord_0, coord_1, coord_2, coord_3] = launch.coordinates;
    let launched = unsafe {
        raw::stwo_cuda_jit_eval_fused_on(
            source,
            kernel.name.as_ptr(),
            kernel.cache_key,
            launch.trace_values,
            launch.interaction_offsets,
            launch.base_params,
            launch.ext_params,
            launch.random_coefficient_powers,
            launch.denominator_inverses,
            coord_0,
            coord_1,
            coord_2,
            coord_3,
            launch.row_count,
            launch.log_n_rows,
            launch.rc_base,
            false,
            stream,
        )
    };
    if !launched {
        return Err(PreparedCompositionError::KernelLaunchMiss {
            component,
            kernel: kernel_index,
            cache_key: kernel.cache_key,
        });
    }
    Ok(())
}

#[cfg(feature = "direct-retention-test-api")]
fn install_resource_bounded<'a>(
    arena: &'a DeviceArena,
    component: usize,
    kernel_index: usize,
    row_count: u32,
    kernel: &CompositionKernelPart,
    source_identity: [u8; 32],
) -> Result<aot::InstalledAotFunction<'a>, PreparedCompositionError> {
    let snapshot = cuda_device_snapshot()?;
    let target_sm = current_target_sm(snapshot).ok_or(
        PreparedCompositionError::CompositionStripeAotAuthorityDrift {
            component,
            kernel: kernel_index,
            field: "current device",
        },
    )?;
    let authority = aot::loaded_kernel_authority(kernel.cache_key, target_sm / 10, target_sm % 10)
        .ok_or(
            PreparedCompositionError::CompositionStripeAotAuthorityMissing {
                component,
                kernel: kernel_index,
                cache_key: kernel.cache_key,
                target_sm,
            },
        )?;
    validate_authority(
        component,
        kernel_index,
        kernel,
        source_identity,
        target_sm,
        authority,
    )?;
    let launch = stripe_launch_facts(row_count)?;
    let installed = aot::InstalledAotFunction::install(arena.context(), authority, launch)
        .map_err(
            |error| PreparedCompositionError::CompositionStripeAotInstall {
                component,
                kernel: kernel_index,
                error,
            },
        )?;
    validate_receipt(
        component,
        kernel_index,
        kernel,
        arena,
        snapshot,
        target_sm,
        launch,
        authority,
        &installed,
    )?;
    Ok(installed)
}

#[cfg(feature = "direct-retention-test-api")]
fn validate_authority(
    component: usize,
    kernel_index: usize,
    kernel: &CompositionKernelPart,
    source_identity: [u8; 32],
    target_sm: u32,
    authority: aot::AotKernelAuthority,
) -> Result<(), PreparedCompositionError> {
    let schema = aot::AotKernelAbiSchema::OrdinaryConstraintV1;
    for (matches, field) in [
        (
            aot::loaded_manifest_identity() != [0; 32],
            "manifest identity",
        ),
        (
            authority.source_identity() == source_identity,
            "source identity",
        ),
        (
            authority.kernel_symbol() == kernel.kernel_name.as_str(),
            "kernel symbol",
        ),
        (
            authority.semantic_hash() == kernel.semantic_hash,
            "semantic hash",
        ),
        (authority.cache_key() == kernel.cache_key, "cache key"),
        (authority.target_sm() == target_sm, "target SM"),
        (authority.cubin_identity() != [0; 32], "cubin identity"),
        (
            aot::loaded_cubin_identity(kernel.cache_key, target_sm / 10, target_sm % 10)
                == authority.cubin_identity(),
            "loaded cubin identity",
        ),
        (authority.program_identity() != [0; 32], "program identity"),
        (authority.abi_schema() == Some(schema), "ABI schema"),
        (
            authority.abi_schema_identity() == schema.identity(),
            "ABI schema identity",
        ),
        (
            authority.schema_scope() == aot::AotKernelSchemaScope::StructuredAbi,
            "schema scope",
        ),
        (
            authority.module_globals() == aot::AotKernelModuleGlobals::None,
            "module globals",
        ),
        (authority.identity() != [0; 32], "authority identity"),
    ] {
        if !matches {
            return Err(
                PreparedCompositionError::CompositionStripeAotAuthorityDrift {
                    component,
                    kernel: kernel_index,
                    field,
                },
            );
        }
    }
    Ok(())
}

#[cfg(feature = "direct-retention-test-api")]
#[allow(clippy::too_many_arguments)]
fn validate_receipt(
    component: usize,
    kernel_index: usize,
    kernel: &CompositionKernelPart,
    arena: &DeviceArena,
    snapshot: CudaDeviceSnapshot,
    target_sm: u32,
    launch: aot::InstalledAotLaunchFacts,
    authority: aot::AotKernelAuthority,
    installed: &aot::InstalledAotFunction<'_>,
) -> Result<(), PreparedCompositionError> {
    let receipt = installed.receipt();
    let publication = receipt.function_publication();
    let resources = receipt.resources();
    let manifest = aot::loaded_manifest_identity();
    let schema = aot::AotKernelAbiSchema::OrdinaryConstraintV1;
    for (matches, field) in [
        (installed.belongs_to(arena.context()), "execution context"),
        (
            receipt.manifest_identity() == manifest && manifest != [0; 32],
            "manifest identity",
        ),
        (
            receipt.source_identity() == authority.source_identity(),
            "source identity",
        ),
        (
            receipt.cubin_identity() == authority.cubin_identity(),
            "cubin identity",
        ),
        (
            receipt.program_identity() == authority.program_identity(),
            "program identity",
        ),
        (
            receipt.abi_schema_identity() == schema.identity(),
            "ABI schema identity",
        ),
        (
            receipt.kernel_authority_identity() == authority.identity(),
            "authority identity",
        ),
        (
            receipt.kernel_symbol() == kernel.kernel_name.as_str(),
            "kernel symbol",
        ),
        (
            receipt.semantic_hash() == kernel.semantic_hash,
            "semantic hash",
        ),
        (receipt.cache_key() == kernel.cache_key, "cache key"),
        (receipt.target_sm() == target_sm, "target SM"),
        (receipt.abi_schema() == schema, "ABI schema"),
        (
            receipt.module_globals() == aot::AotKernelModuleGlobals::None,
            "module globals",
        ),
        (
            receipt.ownership() == aot::InstalledAotFunctionOwnership::BorrowedPublished,
            "ownership",
        ),
        (receipt.launch() == launch, "launch geometry"),
        (
            receipt.launch().dynamic_shared_bytes() == 0,
            "dynamic shared bytes",
        ),
        (receipt.device_ordinal() == snapshot.current, "device"),
        (
            receipt.exec_context_token() != 0
                && receipt.driver_context_token() != 0
                && receipt.module_token() != 0
                && receipt.function_token() != 0,
            "native tokens",
        ),
        (
            receipt.stream_token() == arena.context().stream_raw().as_ptr() as usize as u64,
            "stream",
        ),
        (
            receipt.pedersen_publication().is_none(),
            "unexpected globals",
        ),
        (
            resources.max_threads_per_block >= COMPOSITION_STRIPE_THREADS_PER_BLOCK,
            "max threads per block",
        ),
        (
            resources.registers_per_thread != 0
                && resources.registers_per_thread <= COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD,
            "registers per thread",
        ),
        (resources.binary_version == target_sm, "binary version"),
        (resources.local_bytes == 0, "local bytes"),
        (resources.static_shared_bytes == 0, "static shared bytes"),
        (
            publication.manifest_identity() == manifest
                && publication.source_identity() == authority.source_identity()
                && publication.cubin_identity() == authority.cubin_identity()
                && publication.program_identity() == authority.program_identity()
                && publication.abi_schema_identity() == schema.identity()
                && publication.kernel_authority_identity() == authority.identity()
                && publication.kernel_symbol() == kernel.kernel_name.as_str()
                && publication.semantic_hash() == kernel.semantic_hash
                && publication.cache_key() == kernel.cache_key
                && publication.target_sm() == target_sm
                && publication.device_ordinal() == snapshot.current
                && publication.driver_context_token() == receipt.driver_context_token()
                && publication.module_token() == receipt.module_token()
                && publication.function_token() == receipt.function_token(),
            "function publication",
        ),
    ] {
        if !matches {
            return Err(PreparedCompositionError::CompositionStripeAotReceiptDrift {
                component,
                kernel: kernel_index,
                field,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "direct-retention-test-api")]
fn require_resource_bounded(
    arena: &DeviceArena,
    component: usize,
    kernel_index: usize,
    kernel: &PreparedKernel<'_>,
    stream: *mut c_void,
    launch: &PreparedStripeLaunch,
) -> Result<(), PreparedCompositionError> {
    let installed = kernel.installed.as_ref().ok_or(
        PreparedCompositionError::CompositionStripeAotReceiptDrift {
            component,
            kernel: kernel_index,
            field: "installed ownership",
        },
    )?;
    let receipt = installed.receipt();
    let publication = receipt.function_publication();
    let resources = receipt.resources();
    let expected_launch = stripe_launch_facts(launch.row_count)?;
    for (matches, field) in [
        (installed.belongs_to(arena.context()), "execution context"),
        (
            receipt.stream_token() == stream as usize as u64 && !stream.is_null(),
            "stream",
        ),
        (
            receipt.abi_schema() == aot::AotKernelAbiSchema::OrdinaryConstraintV1,
            "ABI schema",
        ),
        (
            receipt.kernel_symbol().as_bytes() == kernel.name.as_bytes(),
            "kernel symbol",
        ),
        (receipt.cache_key() == kernel.cache_key, "cache key"),
        (
            receipt.semantic_hash() == kernel.semantic_hash,
            "semantic hash",
        ),
        (
            receipt.source_identity() == kernel.source_identity,
            "source identity",
        ),
        (receipt.cubin_identity() != [0; 32], "cubin identity"),
        (receipt.launch() == expected_launch, "launch geometry"),
        (
            receipt.launch().dynamic_shared_bytes() == 0,
            "dynamic shared bytes",
        ),
        (
            resources.max_threads_per_block >= COMPOSITION_STRIPE_THREADS_PER_BLOCK,
            "max threads per block",
        ),
        (
            resources.registers_per_thread != 0
                && resources.registers_per_thread <= COMPOSITION_STRIPE_MAX_REGISTERS_PER_THREAD,
            "registers per thread",
        ),
        (
            resources.binary_version == receipt.target_sm(),
            "binary version",
        ),
        (resources.local_bytes == 0, "local bytes"),
        (resources.static_shared_bytes == 0, "static shared bytes"),
        (
            publication.driver_context_token() == receipt.driver_context_token()
                && publication.module_token() == receipt.module_token()
                && publication.function_token() == receipt.function_token(),
            "function publication",
        ),
    ] {
        if !matches {
            return Err(PreparedCompositionError::CompositionStripeAotReceiptDrift {
                component,
                kernel: kernel_index,
                field,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "direct-retention-test-api")]
fn launch_installed(
    arena: &DeviceArena,
    component: usize,
    kernel_index: usize,
    kernel: &PreparedKernel<'_>,
    launch: PreparedStripeLaunch,
) -> Result<(), PreparedCompositionError> {
    let installed = kernel.installed.as_ref().ok_or(
        PreparedCompositionError::CompositionStripeAotReceiptDrift {
            component,
            kernel: kernel_index,
            field: "installed ownership",
        },
    )?;
    let [mut coord_0, mut coord_1, mut coord_2, mut coord_3] = launch.coordinates;
    let mut trace_cols = launch.trace_values.cast::<*const u32>();
    let mut interaction_offsets = launch.interaction_offsets;
    let mut base_params = launch.base_params;
    let mut ext_params = launch.ext_params;
    let mut random_coefficient_powers = launch.random_coefficient_powers;
    let mut denominator_inverses = launch.denominator_inverses;
    let mut row_count = launch.row_count;
    let mut log_n_rows = launch.log_n_rows;
    let mut rc_base = launch.rc_base;
    let mut arguments = [
        (&mut trace_cols as *mut *const *const u32).cast(),
        (&mut interaction_offsets as *mut *const u32).cast(),
        (&mut base_params as *mut *const u32).cast(),
        (&mut ext_params as *mut *const u32).cast(),
        (&mut random_coefficient_powers as *mut *const u32).cast(),
        (&mut denominator_inverses as *mut *const u32).cast(),
        (&mut coord_0 as *mut *mut u32).cast(),
        (&mut coord_1 as *mut *mut u32).cast(),
        (&mut coord_2 as *mut *mut u32).cast(),
        (&mut coord_3 as *mut *mut u32).cast(),
        (&mut row_count as *mut u32).cast(),
        (&mut log_n_rows as *mut u32).cast(),
        (&mut rc_base as *mut u32).cast(),
    ];
    let checked = installed
        .check_arguments(
            aot::AotKernelAbiSchema::OrdinaryConstraintV1,
            &mut arguments,
        )
        .map_err(
            |error| PreparedCompositionError::CompositionStripeAotLaunch {
                component,
                kernel: kernel_index,
                error,
            },
        )?;
    unsafe { installed.launch_raw(arena.context(), checked) }.map_err(|error| {
        PreparedCompositionError::CompositionStripeAotLaunch {
            component,
            kernel: kernel_index,
            error,
        }
    })
}

#[cfg(feature = "direct-retention-test-api")]
fn stripe_launch_facts(
    row_count: u32,
) -> Result<aot::InstalledAotLaunchFacts, PreparedCompositionError> {
    if row_count == 0 {
        return Err(PreparedCompositionError::SizeOverflow);
    }
    aot::InstalledAotLaunchFacts::new(
        [
            row_count.div_ceil(COMPOSITION_STRIPE_THREADS_PER_BLOCK),
            1,
            1,
        ],
        [COMPOSITION_STRIPE_THREADS_PER_BLOCK, 1, 1],
        0,
    )
    .map_err(|_| PreparedCompositionError::SizeOverflow)
}
