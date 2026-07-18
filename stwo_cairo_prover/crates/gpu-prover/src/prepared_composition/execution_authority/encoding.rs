use stwo_backend_cuda::aot;

use super::*;

const SOURCE_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-source.v1\0";
const EFFECT_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-effect.v2\0";
const INVOCATION_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-invocation.v2\0";
const CHILD_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-child.v1\0";
const ABI_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-abi.v1\0";
const OPERATION_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-operation.v1\0";
const PROGRAM_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-program.v2\0";
const LINKED_DOMAIN: &[u8] = b"stwo-cairo.composition-execution-linked.v1\0";

const AUTHORITY_SOURCE: &[u8] = include_bytes!("../execution_authority.rs");
const COMPILER_SOURCE: &[u8] = include_bytes!("compiler.rs");
const LAYOUT_COMPILER_SOURCE: &[u8] = include_bytes!("compiler/layout.rs");
const PRELUDE_COMPILER_SOURCE: &[u8] = include_bytes!("compiler/prelude.rs");
const SPLIT_COMPILER_SOURCE: &[u8] = include_bytes!("compiler/split.rs");
const ENCODING_SOURCE: &[u8] = include_bytes!("encoding.rs");
const SEMANTIC_SOURCE: &[u8] = include_bytes!("semantic.rs");
const PREPARED_SOURCE: &[u8] = include_bytes!("../../prepared_composition.rs");
const SPLIT_SOURCE: &[u8] = include_bytes!("../direct_split.rs");
const WAVE_SOURCE: &[u8] = include_bytes!("../../composition_wave.rs");
const WAVE_AUTHORITY_SOURCE: &[u8] = include_bytes!("../../composition_wave/shard_authority.rs");
const LOADED_WAVE_SOURCE: &[u8] =
    include_bytes!("../../composition_wave/shard_authority/loaded.rs");
const RESIDENT_BINDER_SOURCE: &[u8] = include_bytes!("../../resident_composition.rs");

pub(super) fn source_identity(
    plan: &ProofArenaPlan,
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SOURCE_DOMAIN);
    for source in [
        AUTHORITY_SOURCE,
        COMPILER_SOURCE,
        LAYOUT_COMPILER_SOURCE,
        PRELUDE_COMPILER_SOURCE,
        SPLIT_COMPILER_SOURCE,
        ENCODING_SOURCE,
        SEMANTIC_SOURCE,
        PREPARED_SOURCE,
        SPLIT_SOURCE,
        WAVE_SOURCE,
        WAVE_AUTHORITY_SOURCE,
        LOADED_WAVE_SOURCE,
        RESIDENT_BINDER_SOURCE,
    ] {
        hash_bytes(&mut hasher, source)?;
    }
    hasher.update(&stwo_backend_cuda_kernels::static_cuda_source_identity());
    let composition = plan.composition();
    hasher.update(&composition.plan.key().to_le_bytes());
    hash_size(&mut hasher, composition.plan.wave_kernels.len())?;
    for kernel in &composition.plan.wave_kernels {
        hasher.update(&kernel.evaluation_log_size.to_le_bytes());
        hasher.update(&kernel.cache_key.to_le_bytes());
        hasher.update(&kernel.semantic_hash.to_le_bytes());
        hasher.update(&kernel.program_identity);
        hasher.update(&aot::emitted_source_identity(&kernel.source));
        hash_bytes(&mut hasher, kernel.kernel_name.as_bytes())?;
    }
    let identity = *hasher.finalize().as_bytes();
    if identity == ZERO_IDENTITY {
        return Err(CompositionAuthorityError::InvalidIdentity);
    }
    Ok(identity)
}

pub(super) fn effect_identity(
    accesses: &[CompositionValueAccess],
    contract: &EffectContract,
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EFFECT_DOMAIN);
    hasher.update(contract.id().as_bytes());
    hash_size(&mut hasher, accesses.len())?;
    for access in accesses {
        hasher.update(&access.binding.0.to_le_bytes());
        hasher.update(&[access_kind_tag(access.kind)]);
        hash_optional_role(&mut hasher, access.source);
        hash_optional_role(&mut hasher, access.destination);
        hash_size(&mut hasher, access.elements.start)?;
        hash_size(&mut hasher, access.elements.end)?;
    }
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn invocation_identity(
    abi: CompositionAbi,
    arguments: &[CompositionInvocationArgument],
    embedded_pointer_tables: &[CompositionEmbeddedPointerTable],
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INVOCATION_DOMAIN);
    hasher.update(&[abi_tag(abi)]);
    hash_size(&mut hasher, arguments.len())?;
    for argument in arguments {
        hasher.update(&[argument.ordinal]);
        hash_bytes(&mut hasher, argument.name.as_bytes())?;
        match &argument.value {
            CompositionInvocationValue::Access(access) => {
                hasher.update(&[1]);
                hasher.update(&access.to_le_bytes());
            }
            CompositionInvocationValue::PointerTable { pointee_accesses } => {
                hasher.update(&[2]);
                hash_pointer_table(&mut hasher, pointee_accesses)?;
            }
            CompositionInvocationValue::U32(value) => {
                hasher.update(&[3]);
                hasher.update(&value.to_le_bytes());
            }
            CompositionInvocationValue::ExecutionStream => {
                hasher.update(&[4]);
            }
        }
    }
    hash_size(&mut hasher, embedded_pointer_tables.len())?;
    for table in embedded_pointer_tables {
        hash_pointer_table(&mut hasher, &table.pointee_accesses)?;
    }
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn abi_identity(
    abi: CompositionAbi,
    invocation: &CompositionInvocation,
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ABI_DOMAIN);
    hasher.update(&[abi_tag(abi)]);
    hash_bytes(&mut hasher, abi.wrapper_symbol().as_bytes())?;
    hasher.update(&invocation.identity);
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn child_identity(
    symbol: &str,
    launch: CompositionLaunchGeometry,
    parameters: &[(&'static str, u32)],
    effect: &CompositionEffect,
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHILD_DOMAIN);
    hash_bytes(&mut hasher, symbol.as_bytes())?;
    for value in launch.grid.into_iter().chain(launch.block) {
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&launch.dynamic_shared_bytes.to_le_bytes());
    hasher.update(&[u8::from(launch.cooperative)]);
    hash_size(&mut hasher, parameters.len())?;
    for (name, value) in parameters {
        hash_bytes(&mut hasher, name.as_bytes())?;
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&effect.identity);
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn operation_identities(
    kind: &CompositionOperationKind,
    abi: CompositionAbi,
    invocation: &CompositionInvocation,
    children: &[CompositionChildLaunch],
    source_identity: [u8; 32],
) -> Result<([u8; 32], [u8; 32], [u8; 32], [u8; 32]), CompositionAuthorityError> {
    if source_identity == ZERO_IDENTITY || children.is_empty() {
        return Err(CompositionAuthorityError::InvalidExecution);
    }
    let abi_identity = abi_identity(abi, invocation)?;
    let mut effect_hasher = blake3::Hasher::new();
    effect_hasher.update(EFFECT_DOMAIN);
    hash_size(&mut effect_hasher, children.len())?;
    let mut execution_hasher = blake3::Hasher::new();
    execution_hasher.update(CHILD_DOMAIN);
    hash_size(&mut execution_hasher, children.len())?;
    for child in children {
        effect_hasher.update(&child.effect.identity);
        execution_hasher.update(&child.identity);
    }
    let effect_identity = *effect_hasher.finalize().as_bytes();
    let execution_identity = *execution_hasher.finalize().as_bytes();

    let mut hasher = blake3::Hasher::new();
    hasher.update(OPERATION_DOMAIN);
    hash_operation_kind(&mut hasher, kind);
    hasher.update(&[abi_tag(abi)]);
    hasher.update(&source_identity);
    hasher.update(&abi_identity);
    hasher.update(&effect_identity);
    hasher.update(&execution_identity);
    hasher.update(&invocation.identity);
    let identity = *hasher.finalize().as_bytes();
    Ok((abi_identity, effect_identity, execution_identity, identity))
}

pub(super) fn static_wrapper_source_identity(
    source_identity: [u8; 32],
    abi: CompositionAbi,
    children: &[CompositionChildLaunch],
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SOURCE_DOMAIN);
    hasher.update(&source_identity);
    hasher.update(&stwo_backend_cuda_kernels::static_cuda_source_identity());
    hash_bytes(&mut hasher, abi.wrapper_symbol().as_bytes())?;
    for child in children {
        hash_bytes(&mut hasher, child.symbol.as_bytes())?;
    }
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn program_identity(
    source_plan_key: u64,
    source_identity: [u8; 32],
    layouts: &[CompositionLayout],
    relocations: &[CompositionRelocationLayout],
    operations: &[CompositionOperation],
    waves: &[CompositionWaveShardAuthority],
    outputs: [CompositionValueRole; 8],
) -> Result<[u8; 32], CompositionAuthorityError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROGRAM_DOMAIN);
    hasher.update(&source_plan_key.to_le_bytes());
    hasher.update(&source_identity);
    hash_size(&mut hasher, layouts.len())?;
    for layout in layouts {
        hash_role(&mut hasher, layout.role);
        hasher.update(&layout.logical.0.to_le_bytes());
        hash_size(&mut hasher, layout.first_word)?;
        hash_size(&mut hasher, layout.word_len)?;
        hash_size(&mut hasher, layout.alignment_words)?;
    }
    hash_size(&mut hasher, relocations.len())?;
    for relocation in relocations {
        hash_relocation_role(&mut hasher, relocation.role);
        hasher.update(&relocation.logical.0.to_le_bytes());
        hash_size(&mut hasher, relocation.first_word)?;
        hash_size(&mut hasher, relocation.word_len)?;
        hash_size(&mut hasher, relocation.alignment_words)?;
    }
    hash_size(&mut hasher, operations.len())?;
    for operation in operations {
        hasher.update(&operation.identity);
    }
    hash_size(&mut hasher, waves.len())?;
    for wave in waves {
        hasher.update(wave.digest());
    }
    for output in outputs {
        hash_role(&mut hasher, output);
    }
    Ok(*hasher.finalize().as_bytes())
}

pub(super) fn linked_identity(
    program_identity: [u8; 32],
    static_source_identity: [u8; 32],
    static_module_build_identity: [u8; 32],
    target_sm: u32,
    loaded_waves: &[LoadedCompositionWaveShardAuthority],
) -> Result<[u8; 32], CompositionAuthorityError> {
    if [
        program_identity,
        static_source_identity,
        static_module_build_identity,
    ]
    .contains(&ZERO_IDENTITY)
        || loaded_waves.is_empty()
    {
        return Err(CompositionAuthorityError::InvalidIdentity);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINKED_DOMAIN);
    hasher.update(&program_identity);
    hasher.update(&static_source_identity);
    hasher.update(&static_module_build_identity);
    hasher.update(&target_sm.to_le_bytes());
    hash_size(&mut hasher, loaded_waves.len())?;
    for wave in loaded_waves {
        if wave.target_sm() != target_sm {
            return Err(CompositionAuthorityError::InvalidTargetSm(wave.target_sm()));
        }
        hasher.update(wave.digest());
        hasher.update(wave.manifest_identity());
        hasher.update(wave.cubin_identity());
        hasher.update(wave.kernel_authority_identity());
    }
    Ok(*hasher.finalize().as_bytes())
}

pub(super) const fn abi_tag(abi: CompositionAbi) -> u8 {
    match abi {
        CompositionAbi::MaterializeExtParamsV1 => 1,
        CompositionAbi::GenerateDescendingPowersV1 => 2,
        CompositionAbi::WaveV2 => 3,
        CompositionAbi::LiftAccumulateV1 => 4,
        CompositionAbi::SplitInverseFusedFirstForwardV1 => 5,
        CompositionAbi::SplitForwardAfterFirstIntervalV1 => 6,
    }
}

fn access_kind_tag(kind: CompositionAccessKind) -> u8 {
    match kind {
        CompositionAccessKind::Read => 1,
        CompositionAccessKind::Write => 2,
        CompositionAccessKind::ReadWriteRequired => 3,
    }
}

fn hash_optional_role(hasher: &mut blake3::Hasher, role: Option<CompositionValueRole>) {
    match role {
        Some(role) => {
            hasher.update(&[1]);
            hash_role(hasher, role);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

pub(super) fn hash_role(hasher: &mut blake3::Hasher, role: CompositionValueRole) {
    use CompositionValueRole as Role;
    match role {
        Role::Descriptor { kind, index } => {
            hasher.update(&[1, descriptor_tag(kind)]);
            hasher.update(&index.to_le_bytes());
        }
        Role::RandomCoefficient => {
            hasher.update(&[2]);
        }
        Role::RandomCoefficientPowers => {
            hasher.update(&[3]);
        }
        Role::RelationZ => {
            hasher.update(&[4]);
        }
        Role::RelationAlphaPowers => {
            hasher.update(&[5]);
        }
        Role::ClaimedSum { component } => {
            hasher.update(&[6]);
            hasher.update(&component.to_le_bytes());
        }
        Role::ExtParam { component, slot } => {
            hasher.update(&[7]);
            hasher.update(&component.to_le_bytes());
            hasher.update(&slot.to_le_bytes());
        }
        Role::DirectEvaluation { plan_column } => {
            hasher.update(&[8]);
            hasher.update(&plan_column.to_le_bytes());
        }
        Role::Accumulator {
            log_size,
            coordinate,
            generation,
        } => {
            hasher.update(&[9]);
            hasher.update(&log_size.to_le_bytes());
            hasher.update(&[coordinate, generation]);
        }
        Role::SplitRetained {
            canonical_column,
            generation,
        } => {
            hasher.update(&[10, canonical_column, generation]);
        }
        Role::ForwardTwiddles => {
            hasher.update(&[11]);
        }
        Role::InverseTwiddles => {
            hasher.update(&[12]);
        }
    }
}

fn descriptor_tag(role: CompositionDescriptorRole) -> u8 {
    match role {
        CompositionDescriptorRole::DynamicSourceKinds => 1,
        CompositionDescriptorRole::DynamicSourceIndices => 2,
        CompositionDescriptorRole::DynamicScales => 3,
        CompositionDescriptorRole::WaveParts => 4,
        CompositionDescriptorRole::InteractionOffsets => 5,
        CompositionDescriptorRole::DenominatorInverses => 6,
        CompositionDescriptorRole::BaseParams => 7,
    }
}

fn hash_relocation_role(hasher: &mut blake3::Hasher, role: CompositionRelocationRole) {
    match role {
        CompositionRelocationRole::DynamicDestinations => {
            hasher.update(&[1]);
        }
        CompositionRelocationRole::ClaimedSumPointers => {
            hasher.update(&[2]);
        }
        CompositionRelocationRole::EvaluationPointers { component } => {
            hasher.update(&[3]);
            hasher.update(&component.to_le_bytes());
        }
        CompositionRelocationRole::SplitSourcePointers => {
            hasher.update(&[4]);
        }
        CompositionRelocationRole::SplitRetainedPointers => {
            hasher.update(&[5]);
        }
    }
}

fn hash_pointer_table(
    hasher: &mut blake3::Hasher,
    entries: &[Option<u32>],
) -> Result<(), CompositionAuthorityError> {
    hash_size(hasher, entries.len())?;
    for entry in entries {
        match entry {
            Some(access) => {
                hasher.update(&[1]);
                hasher.update(&access.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    Ok(())
}

fn hash_operation_kind(hasher: &mut blake3::Hasher, kind: &CompositionOperationKind) {
    match kind {
        CompositionOperationKind::MaterializeExtParams {
            count,
            alpha_power_count,
            claimed_sum_count,
        } => {
            hasher.update(&[1]);
            hasher.update(&count.to_le_bytes());
            hasher.update(&alpha_power_count.to_le_bytes());
            hasher.update(&claimed_sum_count.to_le_bytes());
        }
        CompositionOperationKind::GenerateDescendingPowers { count } => {
            hasher.update(&[2]);
            hasher.update(&count.to_le_bytes());
        }
        CompositionOperationKind::Wave {
            wave_index,
            part_count,
            evaluation_log_size,
            row_count,
        } => {
            hasher.update(&[3]);
            for value in [wave_index, part_count, evaluation_log_size, row_count] {
                hasher.update(&value.to_le_bytes());
            }
        }
        CompositionOperationKind::LiftAccumulate {
            lift_index,
            previous_log_size,
            current_log_size,
        } => {
            hasher.update(&[4]);
            for value in [lift_index, previous_log_size, current_log_size] {
                hasher.update(&value.to_le_bytes());
            }
        }
        CompositionOperationKind::SplitInverseFusedFirstForward {
            evaluation_log_size,
        } => {
            hasher.update(&[5]);
            hasher.update(&evaluation_log_size.to_le_bytes());
        }
        CompositionOperationKind::SplitForwardAfterFirstInterval {
            evaluation_log_size,
        } => {
            hasher.update(&[6]);
            hasher.update(&evaluation_log_size.to_le_bytes());
        }
    }
}

fn hash_size(hasher: &mut blake3::Hasher, value: usize) -> Result<(), CompositionAuthorityError> {
    hasher.update(
        &u64::try_from(value)
            .map_err(|_| CompositionAuthorityError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<(), CompositionAuthorityError> {
    hash_size(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}
