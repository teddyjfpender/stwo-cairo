//! Address-free authority for one linked ordinary-CUDA host wrapper.
//!
//! A wrapper is one semantic primitive with one outer effect. Its ordered
//! execution manifest is identity only: it carries no child effects, SSA
//! values, partitions, streams, function pointers, or other process-local
//! addresses. A step is either one caller-controlled kernel launch or one
//! library call whose internal launch geometry is deliberately library-managed.
//! Version 2 intentionally replaces the old launch-only getters with ordered
//! execution-step access: presenting a filtered kernel list as the complete
//! wrapper execution would be unsound once library calls are present.

use super::*;

mod library;
pub use library::*;

const WRAPPER_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-wrapper.v3\0";
const AGGREGATE_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-execution.v2\0";
const ZERO_IDENTITY: [u8; 32] = [0; 32];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCudaLaunchIdentity {
    symbol: Box<[u8]>,
    launch: LaunchGeometry,
}

impl StaticCudaLaunchIdentity {
    pub fn new(symbol: Vec<u8>, launch: LaunchGeometry) -> Result<Self, CompiledProofError> {
        if !valid_execution_name(&symbol) || !valid_launch(launch) {
            return Err(CompiledProofError::InvalidStaticWrapperManifest);
        }
        Ok(Self {
            symbol: symbol.into_boxed_slice(),
            launch,
        })
    }

    pub fn symbol(&self) -> &[u8] {
        &self.symbol
    }

    pub const fn launch(&self) -> LaunchGeometry {
        self.launch
    }

    fn is_valid(&self) -> bool {
        valid_execution_name(&self.symbol) && valid_launch(self.launch)
    }
}

/// One address-free operation in exact wrapper-stream order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticCudaExecutionStepIdentity {
    KernelLaunch(StaticCudaLaunchIdentity),
    LibraryCall(StaticCudaLibraryCallIdentity),
}

impl StaticCudaExecutionStepIdentity {
    fn is_valid(&self) -> bool {
        match self {
            Self::KernelLaunch(launch) => launch.is_valid(),
            Self::LibraryCall(call) => call.is_valid(),
        }
    }
}

/// Canonical compiled-proof projection of an already-validated producer-side
/// linked-wrapper authority. The producer must copy the upstream receipts and
/// compare every local symbol and launch geometry before constructing this
/// object; this type seals that exact result but cannot recreate type-specific
/// CUDA contract validation from opaque identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCudaWrapperAuthority {
    id: StaticCudaWrapperId,
    static_module_build_identity: [u8; 32],
    consumer_target_sm: u32,
    wrapper_symbol: Box<[u8]>,
    semantic_abi_identity: [u8; 32],
    semantic_effect_identity: [u8; 32],
    aggregate_contract_identity: [u8; 32],
    linked_module_identity: [u8; 32],
    execution_steps: Box<[StaticCudaExecutionStepIdentity]>,
    aggregate_execution_identity: [u8; 32],
    accepted_invocation: InvocationContractId,
    accepted_effect: EffectContractId,
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl StaticCudaWrapperAuthority {
    pub fn new(
        id: StaticCudaWrapperId,
        static_module_build_identity: [u8; 32],
        consumer_target_sm: u32,
        wrapper_symbol: Vec<u8>,
        semantic_abi_identity: [u8; 32],
        semantic_effect_identity: [u8; 32],
        aggregate_contract_identity: [u8; 32],
        linked_module_identity: [u8; 32],
        launches: Vec<StaticCudaLaunchIdentity>,
        accepted_invocation: InvocationContractId,
        accepted_effect: EffectContractId,
    ) -> Result<Self, CompiledProofError> {
        Self::new_with_execution_steps(
            id,
            static_module_build_identity,
            consumer_target_sm,
            wrapper_symbol,
            semantic_abi_identity,
            semantic_effect_identity,
            aggregate_contract_identity,
            linked_module_identity,
            launches
                .into_iter()
                .map(StaticCudaExecutionStepIdentity::KernelLaunch)
                .collect(),
            accepted_invocation,
            accepted_effect,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_execution_steps(
        id: StaticCudaWrapperId,
        static_module_build_identity: [u8; 32],
        consumer_target_sm: u32,
        wrapper_symbol: Vec<u8>,
        semantic_abi_identity: [u8; 32],
        semantic_effect_identity: [u8; 32],
        aggregate_contract_identity: [u8; 32],
        linked_module_identity: [u8; 32],
        execution_steps: Vec<StaticCudaExecutionStepIdentity>,
        accepted_invocation: InvocationContractId,
        accepted_effect: EffectContractId,
    ) -> Result<Self, CompiledProofError> {
        if id.0 == 0
            || static_module_build_identity == ZERO_IDENTITY
            || consumer_target_sm < 10
            || semantic_abi_identity == ZERO_IDENTITY
            || semantic_effect_identity == ZERO_IDENTITY
            || aggregate_contract_identity == ZERO_IDENTITY
            || linked_module_identity == ZERO_IDENTITY
            || !valid_symbol(&wrapper_symbol)
            || execution_steps.is_empty()
            || execution_steps.iter().any(|step| !step.is_valid())
        {
            return Err(CompiledProofError::InvalidStaticWrapperAuthority(id));
        }
        let execution_encoding = encode_execution_steps(&execution_steps)?;
        let aggregate_execution_identity = digest(AGGREGATE_DOMAIN, &execution_encoding)?;
        let canonical_encoding = encode_wrapper(
            id,
            static_module_build_identity,
            consumer_target_sm,
            &wrapper_symbol,
            semantic_abi_identity,
            semantic_effect_identity,
            aggregate_contract_identity,
            linked_module_identity,
            &execution_encoding,
            aggregate_execution_identity,
            accepted_invocation,
            accepted_effect,
        )?;
        let digest = digest(WRAPPER_DOMAIN, &canonical_encoding)?;
        Ok(Self {
            id,
            static_module_build_identity,
            consumer_target_sm,
            wrapper_symbol: wrapper_symbol.into_boxed_slice(),
            semantic_abi_identity,
            semantic_effect_identity,
            aggregate_contract_identity,
            linked_module_identity,
            execution_steps: execution_steps.into_boxed_slice(),
            aggregate_execution_identity,
            accepted_invocation,
            accepted_effect,
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            digest,
        })
    }

    pub const fn id(&self) -> StaticCudaWrapperId {
        self.id
    }

    pub const fn static_module_build_identity(&self) -> &[u8; 32] {
        &self.static_module_build_identity
    }

    pub const fn consumer_target_sm(&self) -> u32 {
        self.consumer_target_sm
    }

    pub fn wrapper_symbol(&self) -> &[u8] {
        &self.wrapper_symbol
    }

    pub const fn semantic_abi_identity(&self) -> &[u8; 32] {
        &self.semantic_abi_identity
    }

    pub const fn semantic_effect_identity(&self) -> &[u8; 32] {
        &self.semantic_effect_identity
    }

    /// Identity of the upstream aggregate/composite contract. It seals the
    /// semantic source, ABI, effect, and ordered execution contract; the local
    /// manifest remains an independently inspectable exact execution list.
    pub const fn aggregate_contract_identity(&self) -> &[u8; 32] {
        &self.aggregate_contract_identity
    }

    /// Exact upstream linked-module receipt. This is distinct from the raw
    /// build identity because it also seals the expected archive, target SM,
    /// and aggregate contract admitted by the producer-side authority.
    pub const fn linked_module_identity(&self) -> &[u8; 32] {
        &self.linked_module_identity
    }

    pub fn execution_steps(&self) -> &[StaticCudaExecutionStepIdentity] {
        &self.execution_steps
    }

    pub fn kernel_launches(&self) -> impl Iterator<Item = &StaticCudaLaunchIdentity> {
        self.execution_steps.iter().filter_map(|step| match step {
            StaticCudaExecutionStepIdentity::KernelLaunch(launch) => Some(launch),
            StaticCudaExecutionStepIdentity::LibraryCall(_) => None,
        })
    }

    pub const fn aggregate_execution_identity(&self) -> &[u8; 32] {
        &self.aggregate_execution_identity
    }

    pub const fn accepted_invocation(&self) -> InvocationContractId {
        self.accepted_invocation
    }

    pub const fn accepted_effect(&self) -> EffectContractId {
        self.accepted_effect
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        if self.id.0 == 0
            || self.static_module_build_identity == ZERO_IDENTITY
            || self.consumer_target_sm < 10
            || self.semantic_abi_identity == ZERO_IDENTITY
            || self.semantic_effect_identity == ZERO_IDENTITY
            || self.aggregate_contract_identity == ZERO_IDENTITY
            || self.linked_module_identity == ZERO_IDENTITY
            || !valid_symbol(&self.wrapper_symbol)
            || self.execution_steps.is_empty()
            || self.execution_steps.iter().any(|step| !step.is_valid())
        {
            return Ok(false);
        }
        let execution_encoding = encode_execution_steps(&self.execution_steps)?;
        let aggregate_execution_identity = digest(AGGREGATE_DOMAIN, &execution_encoding)?;
        let canonical_encoding = encode_wrapper(
            self.id,
            self.static_module_build_identity,
            self.consumer_target_sm,
            &self.wrapper_symbol,
            self.semantic_abi_identity,
            self.semantic_effect_identity,
            self.aggregate_contract_identity,
            self.linked_module_identity,
            &execution_encoding,
            aggregate_execution_identity,
            self.accepted_invocation,
            self.accepted_effect,
        )?;
        Ok(
            self.aggregate_execution_identity == aggregate_execution_identity
                && self.canonical_encoding.as_ref() == canonical_encoding.as_slice()
                && self.digest == digest(WRAPPER_DOMAIN, &canonical_encoding)?,
        )
    }
}

fn encode_execution_steps(
    steps: &[StaticCudaExecutionStepIdentity],
) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Vec::from(AGGREGATE_DOMAIN);
    push_size(&mut out, steps.len())?;
    for step in steps {
        match step {
            StaticCudaExecutionStepIdentity::KernelLaunch(launch) => {
                out.push(1);
                push_bytes(&mut out, launch.symbol())?;
                encode_launch(&mut out, launch.launch());
            }
            StaticCudaExecutionStepIdentity::LibraryCall(call) => {
                out.push(2);
                call.encode_into(&mut out);
            }
        }
    }
    Ok(out)
}

fn encode_wrapper(
    id: StaticCudaWrapperId,
    build: [u8; 32],
    target_sm: u32,
    symbol: &[u8],
    abi: [u8; 32],
    semantic_effect: [u8; 32],
    aggregate_contract: [u8; 32],
    linked_module: [u8; 32],
    execution_steps: &[u8],
    aggregate: [u8; 32],
    invocation: InvocationContractId,
    effect: EffectContractId,
) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Vec::from(WRAPPER_DOMAIN);
    out.extend_from_slice(&id.0.to_le_bytes());
    out.extend_from_slice(&build);
    out.extend_from_slice(&target_sm.to_le_bytes());
    push_bytes(&mut out, symbol)?;
    out.extend_from_slice(&abi);
    out.extend_from_slice(&semantic_effect);
    out.extend_from_slice(&aggregate_contract);
    out.extend_from_slice(&linked_module);
    push_bytes(&mut out, execution_steps)?;
    out.extend_from_slice(&aggregate);
    out.extend_from_slice(invocation.as_bytes());
    out.extend_from_slice(effect.as_bytes());
    Ok(out)
}

fn encode_launch(out: &mut Vec<u8>, launch: LaunchGeometry) {
    for value in launch.grid.into_iter().chain(launch.block) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&launch.dynamic_shared_bytes.to_le_bytes());
    out.push(u8::from(launch.cooperative));
}

fn valid_symbol(symbol: &[u8]) -> bool {
    symbol
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        && symbol[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
}

/// Canonical identity spelling for one internally launched CUDA kernel.
///
/// Unlike a wrapper entry symbol, this is manifest metadata rather than a
/// Driver-API lookup name. It therefore admits the exact C++ template
/// instantiations emitted by the audited wrapper, while rejecting alternate
/// spellings of the same integer argument.
fn valid_execution_name(name: &[u8]) -> bool {
    if valid_symbol(name) {
        return true;
    }
    let Some(open) = name.iter().position(|byte| *byte == b'<') else {
        return false;
    };
    if !name.ends_with(b">")
        || !valid_symbol(&name[..open])
        || name[open + 1..name.len() - 1]
            .iter()
            .any(|byte| matches!(byte, b'<' | b'>'))
    {
        return false;
    }
    let arguments = &name[open + 1..name.len() - 1];
    !arguments.is_empty()
        && arguments
            .split(|byte| *byte == b',')
            .all(valid_template_argument)
}

fn valid_template_argument(argument: &[u8]) -> bool {
    if matches!(argument, b"true" | b"false" | b"0") {
        return true;
    }
    matches!(
        argument,
        [b'1'..=b'9', rest @ ..] if rest.iter().all(u8::is_ascii_digit)
    )
}

fn valid_launch(launch: LaunchGeometry) -> bool {
    let block_threads = launch
        .block
        .into_iter()
        .try_fold(1u64, |product, value| product.checked_mul(u64::from(value)));
    launch.grid[0] != 0
        && launch.grid[0] <= i32::MAX as u32
        && launch.grid[1] != 0
        && launch.grid[1] <= u16::MAX as u32
        && launch.grid[2] != 0
        && launch.grid[2] <= u16::MAX as u32
        && launch.block[0] != 0
        && launch.block[0] <= 1024
        && launch.block[1] != 0
        && launch.block[1] <= 1024
        && launch.block[2] != 0
        && launch.block[2] <= 64
        && launch.cluster.is_none()
        && block_threads.is_some_and(|threads| threads <= 1024)
}

fn digest(domain: &[u8], canonical: &[u8]) -> Result<[u8; 32], CompiledProofError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(
        &u64::try_from(canonical.len())
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(canonical);
    Ok(*hasher.finalize().as_bytes())
}

fn push_size(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledProofError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CompiledProofError> {
    push_size(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_WRAPPER_DOMAIN_V3: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-wrapper.v3\0";
    const GOLDEN_AGGREGATE_DOMAIN_V2: &[u8] =
        b"stwo-cairo.compiled-proof.static-cuda-execution.v2\0";
    const LEGACY_WRAPPER_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-wrapper.v1\0";
    const LEGACY_AGGREGATE_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-launches.v1\0";

    fn launch(symbol: &[u8], grid_x: u32) -> StaticCudaLaunchIdentity {
        StaticCudaLaunchIdentity::new(
            symbol.to_vec(),
            LaunchGeometry {
                grid: [grid_x, 1, 1],
                block: [128, 1, 1],
                cluster: None,
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
        )
        .unwrap()
    }

    fn effect() -> EffectContract {
        EffectContract::new(
            vec![EffectAccess::Read {
                source: BoundValueRange {
                    binding: EffectBindingId(0),
                    value: ValueRange {
                        version: ValueVersion(0),
                        elements: ElementRange::new(0, 1).unwrap(),
                    },
                },
            }],
            vec![],
        )
        .unwrap()
    }

    fn invocation() -> InvocationContractId {
        AotInvocation {
            arguments: vec![AotArgumentBinding {
                ordinal: 0,
                value: AotArgumentValue::DevicePointer(Some(EffectBindingId(0))),
            }],
        }
        .contract_id()
        .unwrap()
    }

    fn authority_with_receipts(
        id: StaticCudaWrapperId,
        target_sm: u32,
        abi: [u8; 32],
        semantic_effect: [u8; 32],
        aggregate_contract: [u8; 32],
        linked_module: [u8; 32],
        launches: Vec<StaticCudaLaunchIdentity>,
    ) -> Result<StaticCudaWrapperAuthority, CompiledProofError> {
        StaticCudaWrapperAuthority::new(
            id,
            [7; 32],
            target_sm,
            b"stwo_static_wrapper".to_vec(),
            abi,
            semantic_effect,
            aggregate_contract,
            linked_module,
            launches,
            invocation(),
            effect().id(),
        )
    }

    fn authority(launches: Vec<StaticCudaLaunchIdentity>) -> StaticCudaWrapperAuthority {
        authority_with_receipts(
            StaticCudaWrapperId(1),
            89,
            [9; 32],
            [11; 32],
            [13; 32],
            [15; 32],
            launches,
        )
        .unwrap()
    }

    fn kernel_mut(
        authority: &mut StaticCudaWrapperAuthority,
        index: usize,
    ) -> &mut StaticCudaLaunchIdentity {
        match &mut authority.execution_steps[index] {
            StaticCudaExecutionStepIdentity::KernelLaunch(launch) => launch,
            StaticCudaExecutionStepIdentity::LibraryCall(_) => panic!("expected kernel launch"),
        }
    }

    fn expected_kernel_only_canonical(
        wrapper_domain: &[u8],
        aggregate_domain: &[u8],
        tagged_kernel: bool,
    ) -> Vec<u8> {
        let mut execution = Vec::from(aggregate_domain);
        execution.extend_from_slice(&1u64.to_le_bytes());
        if tagged_kernel {
            execution.push(1);
        }
        execution.extend_from_slice(&(b"kernel_a".len() as u64).to_le_bytes());
        execution.extend_from_slice(b"kernel_a");
        for value in [4u32, 1, 1, 128, 1, 1, 0] {
            execution.extend_from_slice(&value.to_le_bytes());
        }
        execution.push(0);
        let aggregate = digest(aggregate_domain, &execution).unwrap();

        let mut canonical = Vec::from(wrapper_domain);
        canonical.extend_from_slice(&1u32.to_le_bytes());
        canonical.extend_from_slice(&[7; 32]);
        canonical.extend_from_slice(&89u32.to_le_bytes());
        canonical.extend_from_slice(&(b"stwo_static_wrapper".len() as u64).to_le_bytes());
        canonical.extend_from_slice(b"stwo_static_wrapper");
        for identity in [[9; 32], [11; 32], [13; 32], [15; 32]] {
            canonical.extend_from_slice(&identity);
        }
        canonical.extend_from_slice(&(execution.len() as u64).to_le_bytes());
        canonical.extend_from_slice(&execution);
        canonical.extend_from_slice(&aggregate);
        canonical.extend_from_slice(invocation().as_bytes());
        canonical.extend_from_slice(effect().id().as_bytes());
        canonical
    }

    #[test]
    fn one_launch_authority_is_exact_and_address_free() {
        let authority = authority(vec![launch(b"kernel_a", 4)]);
        assert!(authority.has_valid_identity().unwrap());
        assert_eq!(authority.wrapper_symbol(), b"stwo_static_wrapper");
        assert_eq!(authority.execution_steps().len(), 1);
        assert_eq!(authority.kernel_launches().count(), 1);
        assert_ne!(authority.digest(), &[0; 32]);
        let launch_encoding_bytes = AGGREGATE_DOMAIN.len()
            + core::mem::size_of::<u64>()
            + 1
            + core::mem::size_of::<u64>()
            + b"kernel_a".len()
            + 7 * core::mem::size_of::<u32>()
            + core::mem::size_of::<u8>();
        let canonical_bytes = WRAPPER_DOMAIN.len()
            + 2 * core::mem::size_of::<u32>()
            + 8 * 32
            + core::mem::size_of::<u64>()
            + b"stwo_static_wrapper".len()
            + core::mem::size_of::<u64>()
            + launch_encoding_bytes;
        assert_eq!(authority.canonical_encoding().len(), canonical_bytes);
    }

    #[test]
    fn child_template_names_are_exact_and_canonical() {
        for name in [
            b"b2n_init_warp_batch<3>".as_slice(),
            b"b2n_noinit_block_batch<4,true>",
            b"n2b_nofinal_block_batch<3,2>",
            b"ntt_b2n_stage_batch<false>",
        ] {
            assert!(StaticCudaLaunchIdentity::new(
                name.to_vec(),
                LaunchGeometry {
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                    cluster: None,
                    dynamic_shared_bytes: 0,
                    cooperative: false,
                },
            )
            .is_ok());
        }
        for name in [
            b"kernel<>".as_slice(),
            b"kernel<02>",
            b"kernel<True>",
            b"kernel<1,>",
            b"kernel<,1>",
            b"kernel<1><2>",
            b"kernel<-1>",
            b"kernel<1u>",
            b"1kernel<1>",
        ] {
            assert_eq!(
                StaticCudaLaunchIdentity::new(
                    name.to_vec(),
                    LaunchGeometry {
                        grid: [1, 1, 1],
                        block: [1, 1, 1],
                        cluster: None,
                        dynamic_shared_bytes: 0,
                        cooperative: false,
                    },
                ),
                Err(CompiledProofError::InvalidStaticWrapperManifest)
            );
        }
    }

    #[test]
    fn kernel_only_v3_identity_migration_is_explicit_and_deterministic() {
        let baseline = authority(vec![launch(b"kernel_a", 4)]);
        assert_eq!(WRAPPER_DOMAIN, GOLDEN_WRAPPER_DOMAIN_V3);
        assert_eq!(AGGREGATE_DOMAIN, GOLDEN_AGGREGATE_DOMAIN_V2);
        let expected_v3 = expected_kernel_only_canonical(
            GOLDEN_WRAPPER_DOMAIN_V3,
            GOLDEN_AGGREGATE_DOMAIN_V2,
            true,
        );
        let legacy_v1 =
            expected_kernel_only_canonical(LEGACY_WRAPPER_DOMAIN, LEGACY_AGGREGATE_DOMAIN, false);

        assert_eq!(baseline.canonical_encoding(), expected_v3);
        assert_eq!(
            baseline.digest(),
            &digest(WRAPPER_DOMAIN, &expected_v3).unwrap()
        );
        assert_ne!(baseline.canonical_encoding(), legacy_v1);
        assert_ne!(
            baseline.digest(),
            &digest(LEGACY_WRAPPER_DOMAIN, &legacy_v1).unwrap()
        );

        let repeated = authority(vec![launch(b"kernel_a", 4)]);
        assert_eq!(repeated.canonical_encoding(), baseline.canonical_encoding());
        assert_eq!(repeated.digest(), baseline.digest());
    }

    #[test]
    fn zero_wrapper_id_and_upstream_receipts_are_rejected() {
        assert_eq!(
            authority_with_receipts(
                StaticCudaWrapperId(0),
                89,
                [9; 32],
                [11; 32],
                [13; 32],
                [15; 32],
                vec![launch(b"kernel_a", 4)],
            ),
            Err(CompiledProofError::InvalidStaticWrapperAuthority(
                StaticCudaWrapperId(0)
            ))
        );
        for (target_sm, abi, semantic_effect, aggregate_contract, linked_module) in [
            (0, [9; 32], [11; 32], [13; 32], [15; 32]),
            (1, [9; 32], [11; 32], [13; 32], [15; 32]),
            (9, [9; 32], [11; 32], [13; 32], [15; 32]),
            (89, [0; 32], [11; 32], [13; 32], [15; 32]),
            (89, [9; 32], [0; 32], [13; 32], [15; 32]),
            (89, [9; 32], [11; 32], [0; 32], [15; 32]),
            (89, [9; 32], [11; 32], [13; 32], [0; 32]),
        ] {
            assert_eq!(
                authority_with_receipts(
                    StaticCudaWrapperId(1),
                    target_sm,
                    abi,
                    semantic_effect,
                    aggregate_contract,
                    linked_module,
                    vec![launch(b"kernel_a", 4)],
                ),
                Err(CompiledProofError::InvalidStaticWrapperAuthority(
                    StaticCudaWrapperId(1)
                ))
            );
        }
    }

    #[test]
    fn every_sealed_wrapper_field_invalidates_canonical_authority() {
        let baseline = authority(vec![
            launch(b"kernel_a", 4),
            launch(b"kernel_b", 2),
            launch(b"kernel_c", 1),
        ]);
        let mut mutations = Vec::new();
        let authority_mutations: [fn(&mut StaticCudaWrapperAuthority); 9] = [
            |changed: &mut StaticCudaWrapperAuthority| changed.static_module_build_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.consumer_target_sm += 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.wrapper_symbol[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.semantic_abi_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.semantic_effect_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.aggregate_contract_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.linked_module_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.aggregate_execution_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| {
                changed.accepted_invocation = AotInvocation {
                    arguments: vec![AotArgumentBinding {
                        ordinal: 0,
                        value: AotArgumentValue::U32(1),
                    }],
                }
                .contract_id()
                .unwrap()
            },
        ];
        for mutate in authority_mutations {
            let mut changed = baseline.clone();
            mutate(&mut changed);
            mutations.push(changed);
        }

        let mut order = baseline.clone();
        order.execution_steps.swap(0, 2);
        mutations.push(order);
        for launch_index in 0..baseline.execution_steps.len() {
            let mut symbol = baseline.clone();
            kernel_mut(&mut symbol, launch_index).symbol[0] ^= 1;
            mutations.push(symbol);
            for axis in 0..3 {
                let mut grid = baseline.clone();
                kernel_mut(&mut grid, launch_index).launch.grid[axis] += 1;
                mutations.push(grid);

                let mut block = baseline.clone();
                kernel_mut(&mut block, launch_index).launch.block[axis] += 1;
                mutations.push(block);
            }
            let mut shared = baseline.clone();
            kernel_mut(&mut shared, launch_index)
                .launch
                .dynamic_shared_bytes += 4;
            mutations.push(shared);

            let mut cooperative = baseline.clone();
            kernel_mut(&mut cooperative, launch_index)
                .launch
                .cooperative = true;
            mutations.push(cooperative);
        }

        let mut accepted_effect = baseline;
        accepted_effect.accepted_effect = EffectContract::new(
            vec![EffectAccess::Read {
                source: BoundValueRange {
                    binding: EffectBindingId(0),
                    value: ValueRange {
                        version: ValueVersion(1),
                        elements: ElementRange::new(0, 1).unwrap(),
                    },
                },
            }],
            vec![],
        )
        .unwrap()
        .id();
        mutations.push(accepted_effect);

        for changed in mutations {
            assert!(!changed.has_valid_identity().unwrap());
        }
    }
}

#[cfg(test)]
#[path = "static_wrapper/library_tests.rs"]
mod library_tests;
