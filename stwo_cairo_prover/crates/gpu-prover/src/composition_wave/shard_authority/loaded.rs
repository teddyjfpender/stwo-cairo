//! Loaded-cubin and current-context binding for a composition-wave shard.

use stwo_backend_cuda::aot::{
    self, AotKernelAbiSchema, AotKernelAuthority, AotKernelModuleGlobals, AotKernelSchemaScope,
    InstalledAotFunction, InstalledAotFunctionOwnership, InstalledAotFunctionReceipt,
    InstalledAotLaunchFacts,
};
use stwo_backend_cuda::{cuda_device_snapshot, CudaDeviceSnapshot, CudaExecContext};
use stwo_backend_cuda_kernels::raw::CudaCompositionWavePart;

use super::{CompositionWaveShardAuthority, CompositionWaveShardAuthorityError, ZERO_IDENTITY};

const AUTHORITY_DOMAIN: &[u8] = b"stwo-cairo.loaded-composition-wave-shard-authority.v1\0";

/// A structural shard authority bound to one exact embedded `(kernel, SM)`
/// cubin.
///
/// This is still not execution authority: the proof's current CUDA context
/// must separately publish and install the exact function, then bind that
/// nonzero installed-function receipt before any launch is admitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedCompositionWaveShardAuthority {
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
    structural: CompositionWaveShardAuthority,
    target_sm: u32,
    manifest_identity: [u8; 32],
    cubin_identity: [u8; 32],
    kernel_authority_identity: [u8; 32],
}

/// Eager execution authority for one exact shard on one proof-owned context.
///
/// Process-local CUDA tokens remain in the retained backend receipt and never
/// enter the structural or loaded authority identities.
pub struct InstalledCompositionWaveShardAuthority<'context> {
    loaded: LoadedCompositionWaveShardAuthority,
    installed: InstalledAotFunction<'context>,
    shard_start: u32,
    shard_rows: u32,
}

impl CompositionWaveShardAuthority {
    /// Consume this structural authority and bind it to the exact embedded
    /// composition-wave cubin for one target architecture. The returned value
    /// is a pack binding, not a current-context installed-function receipt.
    pub fn bind_loaded(
        self,
        sm_major: u32,
        sm_minor: u32,
    ) -> Result<LoadedCompositionWaveShardAuthority, CompositionWaveShardAuthorityError> {
        let target_sm = encode_sm(sm_major, sm_minor)
            .ok_or(CompositionWaveShardAuthorityError::InvalidTargetSm)?;
        let manifest_identity = aot::loaded_manifest_identity();
        if manifest_identity == ZERO_IDENTITY {
            return Err(CompositionWaveShardAuthorityError::MissingLoadedManifest);
        }
        let kernel = aot::loaded_kernel_authority(self.kernel_cache_key, sm_major, sm_minor)
            .ok_or(CompositionWaveShardAuthorityError::MissingLoadedKernel {
                cache_key: self.kernel_cache_key,
                target_sm,
            })?;
        if kernel.target_sm() != target_sm
            || kernel.kernel_symbol() != self.kernel_name()
            || kernel.cache_key() != self.kernel_cache_key
            || kernel.semantic_hash() != self.kernel_semantic_hash
        {
            return Err(CompositionWaveShardAuthorityError::LoadedKernelDrift(
                "target, symbol, or lookup identity",
            ));
        }
        if kernel.source_identity() != self.kernel_source_identity
            || kernel.program_identity() != self.kernel_program_identity
            || kernel.abi_schema() != Some(AotKernelAbiSchema::CompositionWaveV2)
            || kernel.abi_schema_identity() != self.kernel_abi_schema_identity
            || kernel.schema_scope() != AotKernelSchemaScope::StructuredAbi
            || kernel.module_globals() != AotKernelModuleGlobals::None
        {
            return Err(CompositionWaveShardAuthorityError::LoadedKernelDrift(
                "source, program, or structured ABI",
            ));
        }
        let cubin_identity = kernel.cubin_identity();
        let kernel_authority_identity = kernel.identity();
        if self.kernel_source_identity == ZERO_IDENTITY
            || self.kernel_program_identity == ZERO_IDENTITY
            || self.kernel_abi_schema_identity == ZERO_IDENTITY
            || cubin_identity == ZERO_IDENTITY
            || kernel_authority_identity == ZERO_IDENTITY
            || aot::loaded_cubin_identity(self.kernel_cache_key, sm_major, sm_minor)
                != cubin_identity
        {
            return Err(CompositionWaveShardAuthorityError::LoadedKernelDrift(
                "nonzero loaded identities",
            ));
        }
        let canonical_encoding = encode(
            &self,
            target_sm,
            manifest_identity,
            cubin_identity,
            kernel_authority_identity,
        );
        let digest = *blake3::hash(&canonical_encoding).as_bytes();
        Ok(LoadedCompositionWaveShardAuthority {
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            digest,
            structural: self,
            target_sm,
            manifest_identity,
            cubin_identity,
            kernel_authority_identity,
        })
    }
}

impl LoadedCompositionWaveShardAuthority {
    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub const fn structural(&self) -> &CompositionWaveShardAuthority {
        &self.structural
    }

    pub const fn target_sm(&self) -> u32 {
        self.target_sm
    }

    pub const fn manifest_identity(&self) -> &[u8; 32] {
        &self.manifest_identity
    }

    pub const fn cubin_identity(&self) -> &[u8; 32] {
        &self.cubin_identity
    }

    pub const fn kernel_authority_identity(&self) -> &[u8; 32] {
        &self.kernel_authority_identity
    }

    /// Install the exact loaded function for one nonempty row shard.
    ///
    /// Installation neither compiles nor repairs a missing cache entry. The
    /// embedded AOT function must already be published on `context`.
    pub fn install<'context>(
        self,
        context: &'context CudaExecContext,
        shard_start: usize,
        shard_rows: usize,
    ) -> Result<InstalledCompositionWaveShardAuthority<'context>, CompositionWaveShardAuthorityError>
    {
        let full_rows = self.structural.full_rows();
        u32::try_from(full_rows).map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?;
        let (shard_start, shard_rows) = checked_shard_range(full_rows, shard_start, shard_rows)?;
        let snapshot = cuda_device_snapshot()?;
        let actual_sm = current_target_sm(snapshot).unwrap_or_default();
        if actual_sm != self.target_sm {
            return Err(CompositionWaveShardAuthorityError::CurrentDeviceDrift {
                expected_sm: self.target_sm,
                actual_sm,
            });
        }
        let sm_major = self.target_sm / 10;
        let sm_minor = self.target_sm % 10;
        let authority =
            aot::loaded_kernel_authority(self.structural.kernel_cache_key(), sm_major, sm_minor)
                .ok_or(CompositionWaveShardAuthorityError::MissingLoadedKernel {
                    cache_key: self.structural.kernel_cache_key(),
                    target_sm: self.target_sm,
                })?;
        validate_loaded_authority(&self, authority)?;
        let launch = shard_launch_facts(shard_rows)?;
        let installed = InstalledAotFunction::install(context, authority, launch)?;
        if !installed.belongs_to(context) {
            return Err(CompositionWaveShardAuthorityError::InstalledReceiptDrift(
                "execution context",
            ));
        }
        validate_installed_receipt(&self, context, snapshot, launch, installed.receipt())?;
        Ok(InstalledCompositionWaveShardAuthority {
            loaded: self,
            installed,
            shard_start,
            shard_rows,
        })
    }
}

impl InstalledCompositionWaveShardAuthority<'_> {
    pub const fn loaded(&self) -> &LoadedCompositionWaveShardAuthority {
        &self.loaded
    }

    pub const fn shard_start(&self) -> u32 {
        self.shard_start
    }

    pub const fn shard_rows(&self) -> u32 {
        self.shard_rows
    }

    pub fn receipt(&self) -> &InstalledAotFunctionReceipt {
        self.installed.receipt()
    }

    /// Enqueue the exact installed shard through the backend's checked typed
    /// function seam.
    ///
    /// This is eager-only. CUDA graph capture continues to require the
    /// capture-safe wrapper, fenced by a retained installed receipt.
    ///
    /// # Safety
    ///
    /// Every device pointer must belong to this authority's proof context,
    /// satisfy the structural effect extents, and remain live through the
    /// context stream's launch.
    pub unsafe fn launch(
        &self,
        context: &CudaExecContext,
        parts: *const CudaCompositionWavePart,
        random_coefficient_powers: *const u32,
        coordinates: [*mut u32; 4],
    ) -> Result<(), CompositionWaveShardAuthorityError> {
        let device_arguments = [
            parts.cast::<core::ffi::c_void>(),
            random_coefficient_powers.cast::<core::ffi::c_void>(),
            coordinates[0].cast::<core::ffi::c_void>(),
            coordinates[1].cast::<core::ffi::c_void>(),
            coordinates[2].cast::<core::ffi::c_void>(),
            coordinates[3].cast::<core::ffi::c_void>(),
        ];
        if let Some(ordinal) = device_arguments
            .iter()
            .position(|argument| argument.is_null())
        {
            return Err(CompositionWaveShardAuthorityError::NullInstalledArgument {
                ordinal: ordinal as u8,
            });
        }
        if !self.installed.belongs_to(context) {
            return Err(CompositionWaveShardAuthorityError::InstalledReceiptDrift(
                "execution context",
            ));
        }

        let mut parts = parts;
        let mut random_coefficient_powers = random_coefficient_powers;
        let mut coord_0 = coordinates[0];
        let mut coord_1 = coordinates[1];
        let mut coord_2 = coordinates[2];
        let mut coord_3 = coordinates[3];
        let mut full_rows = u32::try_from(self.loaded.structural.full_rows())
            .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?;
        let mut shard_start = self.shard_start;
        let mut shard_rows = self.shard_rows;
        let mut arguments = [
            (&mut parts as *mut *const CudaCompositionWavePart).cast(),
            (&mut random_coefficient_powers as *mut *const u32).cast(),
            (&mut coord_0 as *mut *mut u32).cast(),
            (&mut coord_1 as *mut *mut u32).cast(),
            (&mut coord_2 as *mut *mut u32).cast(),
            (&mut coord_3 as *mut *mut u32).cast(),
            (&mut full_rows as *mut u32).cast(),
            (&mut shard_start as *mut u32).cast(),
            (&mut shard_rows as *mut u32).cast(),
        ];
        let checked = self
            .installed
            .check_arguments(AotKernelAbiSchema::CompositionWaveV2, &mut arguments)?;
        unsafe { self.installed.launch_raw(context, checked)? };
        Ok(())
    }
}

fn checked_shard_range(
    full_rows: usize,
    shard_start: usize,
    shard_rows: usize,
) -> Result<(u32, u32), CompositionWaveShardAuthorityError> {
    if full_rows == 0
        || shard_rows == 0
        || shard_start >= full_rows
        || shard_rows > full_rows - shard_start
    {
        return Err(CompositionWaveShardAuthorityError::InvalidShardRange {
            full_rows,
            shard_start,
            shard_rows,
        });
    }
    Ok((
        u32::try_from(shard_start).map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?,
        u32::try_from(shard_rows).map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?,
    ))
}

fn shard_launch_facts(
    shard_rows: u32,
) -> Result<InstalledAotLaunchFacts, CompositionWaveShardAuthorityError> {
    let threads = u32::try_from(aot::COMPOSITION_WAVE_THREADS_PER_BLOCK)
        .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?;
    InstalledAotLaunchFacts::new([shard_rows.div_ceil(threads), 1, 1], [threads, 1, 1], 0)
        .map_err(Into::into)
}

fn current_target_sm(snapshot: CudaDeviceSnapshot) -> Option<u32> {
    (snapshot.count != 0
        && snapshot.current < snapshot.count
        && snapshot.sm_major != 0
        && snapshot.sm_minor <= 9)
        .then_some(())
        .and_then(|()| snapshot.sm_major.checked_mul(10))
        .and_then(|major| major.checked_add(snapshot.sm_minor))
}

fn validate_loaded_authority(
    loaded: &LoadedCompositionWaveShardAuthority,
    authority: AotKernelAuthority,
) -> Result<(), CompositionWaveShardAuthorityError> {
    let structural = loaded.structural();
    for (matches, field) in [
        (
            aot::loaded_manifest_identity() == loaded.manifest_identity,
            "manifest identity",
        ),
        (
            authority.source_identity() == *structural.kernel_source_identity(),
            "source identity",
        ),
        (
            authority.kernel_symbol() == structural.kernel_name(),
            "kernel symbol",
        ),
        (
            authority.semantic_hash() == structural.kernel_semantic_hash(),
            "semantic hash",
        ),
        (
            authority.cache_key() == structural.kernel_cache_key(),
            "cache key",
        ),
        (authority.target_sm() == loaded.target_sm, "target SM"),
        (
            authority.cubin_identity() == loaded.cubin_identity,
            "cubin identity",
        ),
        (
            authority.program_identity() == *structural.kernel_program_identity(),
            "program identity",
        ),
        (
            authority.abi_schema() == Some(AotKernelAbiSchema::CompositionWaveV2),
            "ABI schema",
        ),
        (
            authority.abi_schema_identity() == *structural.kernel_abi_schema_identity(),
            "ABI schema identity",
        ),
        (
            authority.identity() == loaded.kernel_authority_identity,
            "kernel authority identity",
        ),
        (
            authority.schema_scope() == AotKernelSchemaScope::StructuredAbi,
            "schema scope",
        ),
        (
            authority.module_globals() == AotKernelModuleGlobals::None,
            "module globals",
        ),
    ] {
        if !matches {
            return Err(CompositionWaveShardAuthorityError::LoadedKernelDrift(field));
        }
    }
    Ok(())
}

fn validate_installed_receipt(
    loaded: &LoadedCompositionWaveShardAuthority,
    context: &CudaExecContext,
    snapshot: CudaDeviceSnapshot,
    launch: InstalledAotLaunchFacts,
    receipt: &InstalledAotFunctionReceipt,
) -> Result<(), CompositionWaveShardAuthorityError> {
    let structural = loaded.structural();
    let publication = receipt.function_publication();
    for (matches, field) in [
        (
            receipt.manifest_identity() == loaded.manifest_identity,
            "manifest",
        ),
        (
            receipt.source_identity() == *structural.kernel_source_identity(),
            "source",
        ),
        (receipt.cubin_identity() == loaded.cubin_identity, "cubin"),
        (
            receipt.program_identity() == *structural.kernel_program_identity(),
            "program",
        ),
        (
            receipt.abi_schema_identity() == *structural.kernel_abi_schema_identity(),
            "ABI identity",
        ),
        (
            receipt.kernel_authority_identity() == loaded.kernel_authority_identity,
            "authority",
        ),
        (
            receipt.kernel_symbol() == structural.kernel_name(),
            "symbol",
        ),
        (
            receipt.semantic_hash() == structural.kernel_semantic_hash(),
            "semantic hash",
        ),
        (
            receipt.cache_key() == structural.kernel_cache_key(),
            "cache key",
        ),
        (receipt.target_sm() == loaded.target_sm, "target SM"),
        (
            receipt.abi_schema() == AotKernelAbiSchema::CompositionWaveV2,
            "ABI schema",
        ),
        (
            receipt.module_globals() == AotKernelModuleGlobals::None,
            "module globals",
        ),
        (
            receipt.ownership() == InstalledAotFunctionOwnership::BorrowedPublished,
            "ownership",
        ),
        (receipt.launch() == launch, "launch geometry"),
        (receipt.device_ordinal() == snapshot.current, "device"),
        (
            receipt.exec_context_token() != 0 && receipt.stream_token() != 0,
            "context tokens",
        ),
        (
            receipt.stream_token() == context.stream_raw().as_ptr() as usize as u64,
            "stream",
        ),
        (
            receipt.driver_context_token() == publication.driver_context_token(),
            "driver context",
        ),
        (
            receipt.module_token() == publication.module_token(),
            "module token",
        ),
        (
            receipt.function_token() == publication.function_token(),
            "function token",
        ),
        (
            receipt.pedersen_publication().is_none(),
            "unexpected Pedersen publication",
        ),
        (
            publication.manifest_identity() == loaded.manifest_identity
                && publication.source_identity() == *structural.kernel_source_identity()
                && publication.cubin_identity() == loaded.cubin_identity
                && publication.program_identity() == *structural.kernel_program_identity()
                && publication.abi_schema_identity() == *structural.kernel_abi_schema_identity()
                && publication.kernel_authority_identity() == loaded.kernel_authority_identity
                && publication.kernel_symbol() == structural.kernel_name()
                && publication.semantic_hash() == structural.kernel_semantic_hash()
                && publication.cache_key() == structural.kernel_cache_key()
                && publication.target_sm() == loaded.target_sm
                && publication.device_ordinal() == snapshot.current
                && publication.driver_context_token() != 0
                && publication.module_token() != 0
                && publication.function_token() != 0,
            "function publication",
        ),
    ] {
        if !matches {
            return Err(CompositionWaveShardAuthorityError::InstalledReceiptDrift(
                field,
            ));
        }
    }
    Ok(())
}

fn encode(
    structural: &CompositionWaveShardAuthority,
    target_sm: u32,
    manifest_identity: [u8; 32],
    cubin_identity: [u8; 32],
    kernel_authority_identity: [u8; 32],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(AUTHORITY_DOMAIN);
    out.extend_from_slice(structural.digest());
    out.extend_from_slice(&target_sm.to_le_bytes());
    out.extend_from_slice(&manifest_identity);
    out.extend_from_slice(structural.kernel_source_identity());
    out.extend_from_slice(structural.kernel_program_identity());
    out.extend_from_slice(structural.kernel_abi_schema_identity());
    out.extend_from_slice(&cubin_identity);
    out.extend_from_slice(&kernel_authority_identity);
    out
}

fn encode_sm(sm_major: u32, sm_minor: u32) -> Option<u32> {
    if sm_major == 0 || sm_minor > 9 {
        return None;
    }
    sm_major.checked_mul(10)?.checked_add(sm_minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_range_and_launch_geometry_fail_closed() {
        assert_eq!(checked_shard_range(1_024, 256, 384), Ok((256, 384)));
        for (full_rows, start, rows) in [
            (0, 0, 1),
            (1_024, 0, 0),
            (1_024, 1_024, 1),
            (1_024, 900, 125),
            (usize::MAX, usize::MAX - 1, 2),
        ] {
            assert!(matches!(
                checked_shard_range(full_rows, start, rows),
                Err(CompositionWaveShardAuthorityError::InvalidShardRange { .. })
            ));
        }
        assert_eq!(shard_launch_facts(1).unwrap().grid(), [1, 1, 1]);
        assert_eq!(shard_launch_facts(128).unwrap().grid(), [1, 1, 1]);
        assert_eq!(shard_launch_facts(129).unwrap().grid(), [2, 1, 1]);
        assert_eq!(shard_launch_facts(129).unwrap().block(), [128, 1, 1]);
    }

    #[test]
    fn current_device_and_sm_encoding_reject_malformed_snapshots() {
        let valid = CudaDeviceSnapshot {
            count: 2,
            current: 1,
            sm_major: 8,
            sm_minor: 9,
        };
        assert_eq!(current_target_sm(valid), Some(89));
        for changed in [
            CudaDeviceSnapshot { count: 0, ..valid },
            CudaDeviceSnapshot {
                current: 2,
                ..valid
            },
            CudaDeviceSnapshot {
                sm_major: 0,
                ..valid
            },
            CudaDeviceSnapshot {
                sm_minor: 10,
                ..valid
            },
        ] {
            assert_eq!(current_target_sm(changed), None);
        }
        assert_eq!(encode_sm(8, 9), Some(89));
        assert_eq!(encode_sm(0, 9), None);
        assert_eq!(encode_sm(8, 10), None);
    }
}
