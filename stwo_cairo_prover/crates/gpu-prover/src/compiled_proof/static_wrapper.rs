//! Address-free authority for one linked ordinary-CUDA host wrapper.
//!
//! A wrapper is one semantic primitive with one outer effect. Its ordered
//! launch manifest is identity only: it carries no child effects, SSA values,
//! partitions, streams, function pointers, or other process-local addresses.

use super::*;

const WRAPPER_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-wrapper.v1\0";
const AGGREGATE_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.static-cuda-launches.v1\0";
const ZERO_IDENTITY: [u8; 32] = [0; 32];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCudaLaunchIdentity {
    symbol: Box<[u8]>,
    launch: LaunchGeometry,
}

impl StaticCudaLaunchIdentity {
    pub fn new(symbol: Vec<u8>, launch: LaunchGeometry) -> Result<Self, CompiledProofError> {
        if !valid_symbol(&symbol) || !valid_launch(launch) {
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
        valid_symbol(&self.symbol) && valid_launch(self.launch)
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
    launches: Box<[StaticCudaLaunchIdentity]>,
    aggregate_launch_identity: [u8; 32],
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
            || launches.is_empty()
            || launches.iter().any(|launch| !launch.is_valid())
        {
            return Err(CompiledProofError::InvalidStaticWrapperAuthority(id));
        }
        let launch_encoding = encode_launches(&launches)?;
        let aggregate_launch_identity = digest(AGGREGATE_DOMAIN, &launch_encoding)?;
        let canonical_encoding = encode_wrapper(
            id,
            static_module_build_identity,
            consumer_target_sm,
            &wrapper_symbol,
            semantic_abi_identity,
            semantic_effect_identity,
            aggregate_contract_identity,
            linked_module_identity,
            &launch_encoding,
            aggregate_launch_identity,
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
            launches: launches.into_boxed_slice(),
            aggregate_launch_identity,
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
    /// semantic source, ABI, effect, and ordered launch contract; the local
    /// manifest remains an independently inspectable exact launch list.
    pub const fn aggregate_contract_identity(&self) -> &[u8; 32] {
        &self.aggregate_contract_identity
    }

    /// Exact upstream linked-module receipt. This is distinct from the raw
    /// build identity because it also seals the expected archive, target SM,
    /// and aggregate contract admitted by the producer-side authority.
    pub const fn linked_module_identity(&self) -> &[u8; 32] {
        &self.linked_module_identity
    }

    pub fn launches(&self) -> &[StaticCudaLaunchIdentity] {
        &self.launches
    }

    pub const fn aggregate_launch_identity(&self) -> &[u8; 32] {
        &self.aggregate_launch_identity
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
            || self.launches.is_empty()
            || self.launches.iter().any(|launch| !launch.is_valid())
        {
            return Ok(false);
        }
        let launch_encoding = encode_launches(&self.launches)?;
        let aggregate_launch_identity = digest(AGGREGATE_DOMAIN, &launch_encoding)?;
        let canonical_encoding = encode_wrapper(
            self.id,
            self.static_module_build_identity,
            self.consumer_target_sm,
            &self.wrapper_symbol,
            self.semantic_abi_identity,
            self.semantic_effect_identity,
            self.aggregate_contract_identity,
            self.linked_module_identity,
            &launch_encoding,
            aggregate_launch_identity,
            self.accepted_effect,
        )?;
        Ok(self.aggregate_launch_identity == aggregate_launch_identity
            && self.canonical_encoding.as_ref() == canonical_encoding.as_slice()
            && self.digest == digest(WRAPPER_DOMAIN, &canonical_encoding)?)
    }
}

fn encode_launches(launches: &[StaticCudaLaunchIdentity]) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Vec::from(AGGREGATE_DOMAIN);
    push_size(&mut out, launches.len())?;
    for launch in launches {
        push_bytes(&mut out, launch.symbol())?;
        encode_launch(&mut out, launch.launch());
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
    launches: &[u8],
    aggregate: [u8; 32],
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
    push_bytes(&mut out, launches)?;
    out.extend_from_slice(&aggregate);
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

    #[test]
    fn one_launch_authority_is_exact_and_address_free() {
        let authority = authority(vec![launch(b"kernel_a", 4)]);
        assert!(authority.has_valid_identity().unwrap());
        assert_eq!(authority.wrapper_symbol(), b"stwo_static_wrapper");
        assert_eq!(authority.launches().len(), 1);
        assert_ne!(authority.digest(), &[0; 32]);
        let launch_encoding_bytes = AGGREGATE_DOMAIN.len()
            + core::mem::size_of::<u64>()
            + core::mem::size_of::<u64>()
            + b"kernel_a".len()
            + 7 * core::mem::size_of::<u32>()
            + core::mem::size_of::<u8>();
        let canonical_bytes = WRAPPER_DOMAIN.len()
            + 2 * core::mem::size_of::<u32>()
            + 7 * 32
            + core::mem::size_of::<u64>()
            + b"stwo_static_wrapper".len()
            + core::mem::size_of::<u64>()
            + launch_encoding_bytes;
        assert_eq!(authority.canonical_encoding().len(), canonical_bytes);
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
        let authority_mutations: [fn(&mut StaticCudaWrapperAuthority); 8] = [
            |changed: &mut StaticCudaWrapperAuthority| changed.static_module_build_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.consumer_target_sm += 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.wrapper_symbol[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.semantic_abi_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.semantic_effect_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.aggregate_contract_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.linked_module_identity[0] ^= 1,
            |changed: &mut StaticCudaWrapperAuthority| changed.aggregate_launch_identity[0] ^= 1,
        ];
        for mutate in authority_mutations {
            let mut changed = baseline.clone();
            mutate(&mut changed);
            mutations.push(changed);
        }

        let mut order = baseline.clone();
        order.launches.swap(0, 2);
        mutations.push(order);
        for launch_index in 0..baseline.launches.len() {
            let mut symbol = baseline.clone();
            symbol.launches[launch_index].symbol[0] ^= 1;
            mutations.push(symbol);
            for axis in 0..3 {
                let mut grid = baseline.clone();
                grid.launches[launch_index].launch.grid[axis] += 1;
                mutations.push(grid);

                let mut block = baseline.clone();
                block.launches[launch_index].launch.block[axis] += 1;
                mutations.push(block);
            }
            let mut shared = baseline.clone();
            shared.launches[launch_index].launch.dynamic_shared_bytes += 4;
            mutations.push(shared);

            let mut cooperative = baseline.clone();
            cooperative.launches[launch_index].launch.cooperative = true;
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
