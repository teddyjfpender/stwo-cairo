//! Exact row-shard authority for one canonical composition wave.
//!
//! The v2 CUDA wave ABI independently binds the immutable full-domain trace
//! stride, shard start and shard length. This module seals those roles together
//! with the canonical program, kernel identity and exact read/write projections
//! before constructing a compiled-proof [`crate::compiled_proof::PartitionAuthority`].

use stwo_backend_cuda::aot::{
    self, AotKernelAbiSchema, COMPOSITION_WAVE_CODEGEN_VERSION,
    COMPOSITION_WAVE_FULL_DOMAIN_ROWS_ARGUMENT, COMPOSITION_WAVE_KERNEL_ARGUMENT_COUNT,
    COMPOSITION_WAVE_SHARD_ROWS_ARGUMENT, COMPOSITION_WAVE_SHARD_START_ARGUMENT,
    COMPOSITION_WAVE_THREADS_PER_BLOCK,
};

use super::{CompositionWaveError, CompositionWaveProgram};
use crate::compiled_proof::{
    CompiledProofError, EffectAccess, EffectBindingId, EffectContract, EffectContractId,
    ElementRange, ExactPartitionAuthority, PartitionAuthority, PartitionEffectProjection,
    PartitionGridAxis, PartitionLaunchDerivation,
};
use crate::composition_plan::CompositionPlan;
use crate::prepared_composition::{
    CompositionAccumulatorRequirements, CompositionWaveRequirements,
};

const AUTHORITY_DOMAIN: &[u8] = b"stwo-cairo.composition-wave-shard-authority.v2\0";
const PROGRAM_DOMAIN: &[u8] = b"stwo-cairo.composition-wave-program.v1\0";
const CURRENT_ABI_DOMAIN: &[u8] = b"stwo-cairo.composition-wave-kernel-abi.v2\0";
const ZERO_IDENTITY: [u8; 32] = [0; 32];
const REQUIRED_CODEGEN_VERSION: u64 = 2;
const ROW_AXIS_TAG: u16 = 0;
const ROW_ALIGNMENT_BYTES: usize = core::mem::size_of::<u32>();
const ACCUMULATOR_COORDINATES: usize = 4;

mod loaded;
pub use loaded::{InstalledCompositionWaveShardAuthority, LoadedCompositionWaveShardAuthority};

/// Exact semantic bindings used by one row-partitioned wave launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWaveShardBindings {
    replicated_reads: Box<[EffectBindingId]>,
    accumulator_coordinates: [EffectBindingId; ACCUMULATOR_COORDINATES],
}

impl CompositionWaveShardBindings {
    fn from_effect(
        effect: &EffectContract,
        accumulator_coordinates: [EffectBindingId; ACCUMULATOR_COORDINATES],
        full_rows: usize,
    ) -> Result<Self, CompositionWaveShardAuthorityError> {
        if !effect.module_globals().is_empty()
            || accumulator_coordinates
                .iter()
                .enumerate()
                .any(|(index, binding)| accumulator_coordinates[..index].contains(binding))
        {
            return Err(CompositionWaveShardAuthorityError::InvalidEffect(
                "module globals or coordinate bindings",
            ));
        }
        let mut replicated_reads = Vec::new();
        let mut destinations = Vec::new();
        for access in effect.accesses() {
            match access {
                EffectAccess::Read { source } => replicated_reads.push(source.binding),
                EffectAccess::Write { destination } => destinations.push(*destination),
                EffectAccess::ReadWrite { .. } | EffectAccess::Atomic { .. } => {
                    return Err(CompositionWaveShardAuthorityError::InvalidEffect(
                        "read-write or atomic access",
                    ));
                }
            }
        }
        replicated_reads.sort_unstable();
        if replicated_reads.len() < 2
            || replicated_reads.windows(2).any(|pair| pair[0] == pair[1])
            || destinations.len() != ACCUMULATOR_COORDINATES
            || destinations.iter().any(|destination| {
                destination.value.elements
                    != (ElementRange {
                        start: 0,
                        end: full_rows,
                    })
            })
            || destinations.iter().enumerate().any(|(index, destination)| {
                destinations[..index]
                    .iter()
                    .any(|previous| previous.value.version == destination.value.version)
            })
            || replicated_reads
                .iter()
                .any(|binding| accumulator_coordinates.contains(binding))
            || destinations
                .iter()
                .map(|destination| destination.binding)
                .collect::<std::collections::BTreeSet<_>>()
                != accumulator_coordinates
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>()
        {
            return Err(CompositionWaveShardAuthorityError::InvalidEffect(
                "replicated reads or four full-row destinations",
            ));
        }
        Ok(Self {
            replicated_reads: replicated_reads.into_boxed_slice(),
            accumulator_coordinates,
        })
    }

    pub fn replicated_reads(&self) -> &[EffectBindingId] {
        &self.replicated_reads
    }

    pub const fn accumulator_coordinates(&self) -> &[EffectBindingId; ACCUMULATOR_COORDINATES] {
        &self.accumulator_coordinates
    }
}

/// Exact range ABI exposed by the v2 wave kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionWaveRangeAbiRequirement {
    pub codegen_version: u64,
    pub argument_count: u8,
    pub full_domain_rows_argument: u8,
    pub shard_start_argument: u8,
    pub shard_rows_argument: u8,
    pub threads_per_block: usize,
}

/// Shape- and kernel-bound proof that one wave has four exact sliced outputs.
///
/// Promotion constructs the real `ExactPartitionAuthority`; downstream
/// compiled-proof validation must still bind its projections to concrete value
/// layouts before execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionWaveShardAuthority {
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
    program_digest: [u8; 32],
    current_abi_digest: [u8; 32],
    wave_index: usize,
    evaluation_log_size: u32,
    full_rows: usize,
    accumulator_offset_words: usize,
    kernel_cache_key: u64,
    kernel_semantic_hash: u64,
    kernel_name: Box<str>,
    kernel_source_identity: [u8; 32],
    kernel_program_identity: [u8; 32],
    kernel_abi_schema_identity: [u8; 32],
    effect: EffectContractId,
    bindings: CompositionWaveShardBindings,
    projections: Box<[PartitionEffectProjection]>,
    partition: PartitionAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompositionWaveShardAuthorityError {
    Program(CompositionWaveError),
    MissingWave(usize),
    RequirementDrift(&'static str),
    KernelIdentityDrift,
    AccumulatorDrift(&'static str),
    InvalidEffect(&'static str),
    Partition(CompiledProofError),
    SizeOverflow,
    InvalidTargetSm,
    MissingLoadedManifest,
    MissingLoadedKernel {
        cache_key: u64,
        target_sm: u32,
    },
    LoadedKernelDrift(&'static str),
    InvalidShardRange {
        full_rows: usize,
        shard_start: usize,
        shard_rows: usize,
    },
    CurrentDeviceDrift {
        expected_sm: u32,
        actual_sm: u32,
    },
    InstalledReceiptDrift(&'static str),
    NullInstalledArgument {
        ordinal: u8,
    },
    Cuda(stwo_backend_cuda::CudaRuntimeError),
    InstalledAot(stwo_backend_cuda::aot::InstalledAotFunctionError),
    CurrentAbiLacksExplicitRange {
        codegen_version: u64,
        argument_count: u8,
    },
}

impl std::fmt::Display for CompositionWaveShardAuthorityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid composition wave shard authority: {self:?}")
    }
}

impl std::error::Error for CompositionWaveShardAuthorityError {}

impl From<CompositionWaveError> for CompositionWaveShardAuthorityError {
    fn from(value: CompositionWaveError) -> Self {
        Self::Program(value)
    }
}

impl From<CompiledProofError> for CompositionWaveShardAuthorityError {
    fn from(value: CompiledProofError) -> Self {
        Self::Partition(value)
    }
}

impl From<stwo_backend_cuda::CudaRuntimeError> for CompositionWaveShardAuthorityError {
    fn from(value: stwo_backend_cuda::CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

impl From<stwo_backend_cuda::aot::InstalledAotFunctionError>
    for CompositionWaveShardAuthorityError
{
    fn from(value: stwo_backend_cuda::aot::InstalledAotFunctionError) -> Self {
        Self::InstalledAot(value)
    }
}

impl CompositionWaveShardAuthority {
    /// Seal one exact wave against the canonical program and prepared shape.
    pub fn derive(
        plan: &CompositionPlan,
        program: &CompositionWaveProgram,
        wave_index: usize,
        requirement: &CompositionWaveRequirements,
        accumulator: &CompositionAccumulatorRequirements,
        effect: &EffectContract,
        accumulator_coordinates: [EffectBindingId; ACCUMULATOR_COORDINATES],
    ) -> Result<Self, CompositionWaveShardAuthorityError> {
        program.validate_against(plan)?;
        let wave = program
            .waves()
            .get(wave_index)
            .ok_or(CompositionWaveShardAuthorityError::MissingWave(wave_index))?;
        let kernel = plan
            .wave_kernels
            .get(wave_index)
            .ok_or(CompositionWaveShardAuthorityError::MissingWave(wave_index))?;
        if plan.wave_kernels.len() != program.waves().len() {
            return Err(CompositionWaveShardAuthorityError::RequirementDrift(
                "wave kernel count",
            ));
        }

        let full_rows = usize::try_from(wave.row_count)
            .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?;
        if requirement.evaluation_log_size != wave.evaluation_log_size
            || requirement.row_count != full_rows
        {
            return Err(CompositionWaveShardAuthorityError::RequirementDrift(
                "wave domain",
            ));
        }

        let expected_parts = wave
            .part_ordinals
            .iter()
            .map(|&ordinal| {
                let part = program.parts().get(ordinal).ok_or(
                    CompositionWaveShardAuthorityError::RequirementDrift("canonical part ordinal"),
                )?;
                Ok((
                    part.component_index,
                    part.kernel_index,
                    aot::CompositionWaveKernelPartIdentity {
                        semantic_hash: part.semantic_hash,
                        coefficient_start: u32::try_from(part.coefficient_start)
                            .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?,
                        coefficient_end: u32::try_from(part.coefficient_end)
                            .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?,
                    },
                ))
            })
            .collect::<Result<Vec<_>, CompositionWaveShardAuthorityError>>()?;
        if requirement.parts.len() != expected_parts.len()
            || requirement
                .parts
                .iter()
                .zip(&expected_parts)
                .any(|(actual, expected)| {
                    (actual.component, actual.kernel, actual.identity) != *expected
                })
        {
            return Err(CompositionWaveShardAuthorityError::RequirementDrift(
                "wave part identity/order",
            ));
        }
        let identities = expected_parts
            .iter()
            .map(|(_, _, identity)| *identity)
            .collect::<Vec<_>>();
        let expected_kernel =
            aot::composition_wave_kernel_identity(wave.evaluation_log_size, &identities)
                .ok_or(CompositionWaveShardAuthorityError::KernelIdentityDrift)?;
        let kernel_source_identity = aot::emitted_source_identity(&kernel.source);
        let kernel_abi_schema_identity = AotKernelAbiSchema::CompositionWaveV2.identity();
        if kernel.evaluation_log_size != wave.evaluation_log_size
            || kernel.parts != identities
            || kernel.kernel_name != expected_kernel.kernel_name
            || kernel.cache_key != expected_kernel.cache_key
            || kernel.semantic_hash != expected_kernel.semantic_hash
            || kernel.source.is_empty()
            || kernel_source_identity == ZERO_IDENTITY
            || kernel.program_identity == ZERO_IDENTITY
            || kernel_abi_schema_identity == ZERO_IDENTITY
        {
            return Err(CompositionWaveShardAuthorityError::KernelIdentityDrift);
        }

        if accumulator.log_size != wave.evaluation_log_size
            || accumulator.offset_words != requirement.accumulator_offset_words
        {
            return Err(CompositionWaveShardAuthorityError::AccumulatorDrift(
                "owner/log",
            ));
        }
        let expected_accumulator_words = full_rows
            .checked_mul(ACCUMULATOR_COORDINATES)
            .ok_or(CompositionWaveShardAuthorityError::SizeOverflow)?;
        if accumulator.len_words != expected_accumulator_words {
            return Err(CompositionWaveShardAuthorityError::AccumulatorDrift(
                "four coordinate extents",
            ));
        }

        let bindings =
            CompositionWaveShardBindings::from_effect(effect, accumulator_coordinates, full_rows)?;
        require_current_range_abi()?;
        let program_digest = program_digest(program)?;
        let current_abi_digest = current_abi_digest();
        let projections = projections(&bindings);
        let elements_per_grid_unit = full_rows.min(COMPOSITION_WAVE_THREADS_PER_BLOCK);
        let partition = PartitionAuthority::exact(ExactPartitionAuthority::new(
            ROW_AXIS_TAG,
            ElementRange {
                start: 0,
                end: full_rows,
            },
            elements_per_grid_unit,
            ROW_ALIGNMENT_BYTES,
            projections.clone(),
            PartitionLaunchDerivation::new(
                COMPOSITION_WAVE_SHARD_START_ARGUMENT,
                COMPOSITION_WAVE_SHARD_ROWS_ARGUMENT,
                PartitionGridAxis::X,
                elements_per_grid_unit,
            )?,
        )?)?;
        let canonical_encoding = encode_authority(
            program_digest,
            current_abi_digest,
            wave_index,
            wave.evaluation_log_size,
            full_rows,
            requirement,
            kernel,
            effect,
            &bindings,
            &projections,
            &partition,
        )?;
        let digest = *blake3::hash(&canonical_encoding).as_bytes();
        Ok(Self {
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            digest,
            program_digest,
            current_abi_digest,
            wave_index,
            evaluation_log_size: wave.evaluation_log_size,
            full_rows,
            accumulator_offset_words: requirement.accumulator_offset_words,
            kernel_cache_key: kernel.cache_key,
            kernel_semantic_hash: kernel.semantic_hash,
            kernel_name: kernel.kernel_name.clone().into_boxed_str(),
            kernel_source_identity,
            kernel_program_identity: kernel.program_identity,
            kernel_abi_schema_identity,
            effect: effect.id(),
            bindings,
            projections: projections.into_boxed_slice(),
            partition,
        })
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub const fn program_digest(&self) -> &[u8; 32] {
        &self.program_digest
    }

    pub const fn current_abi_digest(&self) -> &[u8; 32] {
        &self.current_abi_digest
    }

    pub const fn wave_index(&self) -> usize {
        self.wave_index
    }

    pub const fn evaluation_log_size(&self) -> u32 {
        self.evaluation_log_size
    }

    pub const fn full_rows(&self) -> usize {
        self.full_rows
    }

    pub const fn accumulator_offset_words(&self) -> usize {
        self.accumulator_offset_words
    }

    pub const fn kernel_cache_key(&self) -> u64 {
        self.kernel_cache_key
    }

    pub const fn kernel_semantic_hash(&self) -> u64 {
        self.kernel_semantic_hash
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub const fn kernel_source_identity(&self) -> &[u8; 32] {
        &self.kernel_source_identity
    }

    pub const fn kernel_program_identity(&self) -> &[u8; 32] {
        &self.kernel_program_identity
    }

    pub const fn kernel_abi_schema_identity(&self) -> &[u8; 32] {
        &self.kernel_abi_schema_identity
    }

    pub const fn effect(&self) -> EffectContractId {
        self.effect
    }

    pub const fn bindings(&self) -> &CompositionWaveShardBindings {
        &self.bindings
    }

    pub fn projections(&self) -> &[PartitionEffectProjection] {
        &self.projections
    }

    pub const fn partition(&self) -> &PartitionAuthority {
        &self.partition
    }

    pub const fn row_axis_tag(&self) -> u16 {
        ROW_AXIS_TAG
    }

    pub const fn row_alignment_bytes(&self) -> usize {
        ROW_ALIGNMENT_BYTES
    }

    pub const fn required_range_abi(&self) -> CompositionWaveRangeAbiRequirement {
        CompositionWaveRangeAbiRequirement {
            codegen_version: COMPOSITION_WAVE_CODEGEN_VERSION,
            argument_count: COMPOSITION_WAVE_KERNEL_ARGUMENT_COUNT,
            full_domain_rows_argument: COMPOSITION_WAVE_FULL_DOMAIN_ROWS_ARGUMENT,
            shard_start_argument: COMPOSITION_WAVE_SHARD_START_ARGUMENT,
            shard_rows_argument: COMPOSITION_WAVE_SHARD_ROWS_ARGUMENT,
            threads_per_block: COMPOSITION_WAVE_THREADS_PER_BLOCK,
        }
    }

    /// Recheck the structural range ABI. This does not bind a loaded cubin.
    pub const fn require_structural_range_abi(
        &self,
    ) -> Result<(), CompositionWaveShardAuthorityError> {
        require_current_range_abi()
    }
}

const fn require_current_range_abi() -> Result<(), CompositionWaveShardAuthorityError> {
    if COMPOSITION_WAVE_CODEGEN_VERSION == REQUIRED_CODEGEN_VERSION
        && COMPOSITION_WAVE_KERNEL_ARGUMENT_COUNT == 9
        && COMPOSITION_WAVE_FULL_DOMAIN_ROWS_ARGUMENT == 6
        && COMPOSITION_WAVE_SHARD_START_ARGUMENT == 7
        && COMPOSITION_WAVE_SHARD_ROWS_ARGUMENT == 8
    {
        Ok(())
    } else {
        Err(
            CompositionWaveShardAuthorityError::CurrentAbiLacksExplicitRange {
                codegen_version: COMPOSITION_WAVE_CODEGEN_VERSION,
                argument_count: COMPOSITION_WAVE_KERNEL_ARGUMENT_COUNT,
            },
        )
    }
}

fn projections(bindings: &CompositionWaveShardBindings) -> Vec<PartitionEffectProjection> {
    let mut projections = bindings
        .replicated_reads()
        .iter()
        .copied()
        .map(|binding| PartitionEffectProjection::ReplicatedRead { binding })
        .chain(
            bindings
                .accumulator_coordinates()
                .iter()
                .copied()
                .map(|binding| PartitionEffectProjection::ContiguousAxisSlice { binding }),
        )
        .collect::<Vec<_>>();
    projections.sort_unstable_by_key(|projection| projection.binding());
    projections
}

fn program_digest(
    program: &CompositionWaveProgram,
) -> Result<[u8; 32], CompositionWaveShardAuthorityError> {
    let mut out = Vec::new();
    out.extend_from_slice(PROGRAM_DOMAIN);
    out.extend_from_slice(&program.source_plan_key().to_le_bytes());
    push_usize(&mut out, program.total_constraints())?;
    push_usize(&mut out, program.parts().len())?;
    for part in program.parts() {
        push_usize(&mut out, part.ordinal)?;
        push_usize(&mut out, part.component_index)?;
        push_usize(&mut out, part.kernel_index)?;
        push_bytes(&mut out, part.component.as_bytes())?;
        push_usize(&mut out, part.instance)?;
        out.extend_from_slice(&part.evaluation_log_size.to_le_bytes());
        out.extend_from_slice(&part.row_count.to_le_bytes());
        out.extend_from_slice(&part.cache_key.to_le_bytes());
        out.extend_from_slice(&part.semantic_hash.to_le_bytes());
        push_usize(&mut out, part.coefficient_start)?;
        push_usize(&mut out, part.coefficient_end)?;
    }
    push_usize(&mut out, program.waves().len())?;
    for wave in program.waves() {
        out.extend_from_slice(&wave.evaluation_log_size.to_le_bytes());
        out.extend_from_slice(&wave.row_count.to_le_bytes());
        push_usize(&mut out, wave.part_ordinals.len())?;
        for &ordinal in &wave.part_ordinals {
            push_usize(&mut out, ordinal)?;
        }
    }
    Ok(*blake3::hash(&out).as_bytes())
}

fn current_abi_digest() -> [u8; 32] {
    let mut out = Vec::new();
    out.extend_from_slice(CURRENT_ABI_DOMAIN);
    out.extend_from_slice(&COMPOSITION_WAVE_CODEGEN_VERSION.to_le_bytes());
    out.extend_from_slice(&AotKernelAbiSchema::CompositionWaveV2.identity());
    *blake3::hash(&out).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn encode_authority(
    program_digest: [u8; 32],
    current_abi_digest: [u8; 32],
    wave_index: usize,
    evaluation_log_size: u32,
    full_rows: usize,
    requirement: &CompositionWaveRequirements,
    kernel: &crate::composition_plan::CompositionWaveKernelPlan,
    effect: &EffectContract,
    bindings: &CompositionWaveShardBindings,
    projections: &[PartitionEffectProjection],
    partition: &PartitionAuthority,
) -> Result<Vec<u8>, CompositionWaveShardAuthorityError> {
    let mut out = Vec::new();
    out.extend_from_slice(AUTHORITY_DOMAIN);
    out.extend_from_slice(&program_digest);
    out.extend_from_slice(&current_abi_digest);
    push_usize(&mut out, wave_index)?;
    out.extend_from_slice(&evaluation_log_size.to_le_bytes());
    push_usize(&mut out, full_rows)?;
    push_usize(&mut out, requirement.accumulator_offset_words)?;
    push_usize(&mut out, requirement.descriptor_offset_words)?;
    out.extend_from_slice(&kernel.cache_key.to_le_bytes());
    out.extend_from_slice(&kernel.semantic_hash.to_le_bytes());
    push_bytes(&mut out, kernel.kernel_name.as_bytes())?;
    out.extend_from_slice(&aot::emitted_source_identity(&kernel.source));
    out.extend_from_slice(&kernel.program_identity);
    out.extend_from_slice(&AotKernelAbiSchema::CompositionWaveV2.identity());
    out.extend_from_slice(effect.id().as_bytes());
    push_bytes(&mut out, effect.canonical_encoding())?;
    out.extend_from_slice(&COMPOSITION_WAVE_CODEGEN_VERSION.to_le_bytes());
    out.push(COMPOSITION_WAVE_FULL_DOMAIN_ROWS_ARGUMENT);
    out.push(COMPOSITION_WAVE_SHARD_START_ARGUMENT);
    out.push(COMPOSITION_WAVE_SHARD_ROWS_ARGUMENT);
    push_usize(&mut out, COMPOSITION_WAVE_THREADS_PER_BLOCK)?;
    out.extend_from_slice(&ROW_AXIS_TAG.to_le_bytes());
    push_usize(&mut out, ROW_ALIGNMENT_BYTES)?;
    push_usize(&mut out, bindings.replicated_reads().len())?;
    for binding in bindings.replicated_reads() {
        out.extend_from_slice(&binding.0.to_le_bytes());
    }
    for binding in bindings.accumulator_coordinates() {
        out.extend_from_slice(&binding.0.to_le_bytes());
    }
    push_usize(&mut out, projections.len())?;
    for projection in projections {
        match projection {
            PartitionEffectProjection::ReplicatedRead { .. } => out.push(0),
            PartitionEffectProjection::ContiguousAxisSlice { .. } => out.push(1),
        }
        out.extend_from_slice(&projection.binding().0.to_le_bytes());
    }
    out.extend_from_slice(partition.id().as_bytes());
    push_bytes(&mut out, partition.canonical_encoding())?;
    Ok(out)
}

fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), CompositionWaveShardAuthorityError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompositionWaveShardAuthorityError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CompositionWaveShardAuthorityError> {
    push_usize(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}
