//! Linked static-module authority for the native EC-op composite.

use stwo_backend_cuda::{
    EcOpCompositeAbi, EcOpCompositeContract, EcOpEffectAbi, EcOpKernelLaunch,
    EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_POINTERS, EXECUTION_TABLE_SMALL_LIMBS,
};

use super::ec_op_prefix::{exact_effect, NativeEcOpAtomicBinding, NativeEcOpRangeBinding};
use super::{ArenaCatalogRange, InvocationShapeError};
use crate::compiled_proof::{EffectContract, EffectContractId};

const LINKED_IDENTITY_DOMAIN: &[u8] = b"stwo-cairo-native-ec-op-linked-module-v1\0";
const INVOCATION_IDENTITY_DOMAIN: &[u8] = b"stwo-cairo-native-ec-op-invocation-v1\0";
const EXECUTION_IDENTITY_DOMAIN: &[u8] = b"stwo-cairo-native-ec-op-execution-authority-v2\0";
const ZERO_IDENTITY: [u8; 32] = [0; 32];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeEcOpStaticLaunchAuthority {
    pub(super) entry_symbol: &'static str,
    pub(super) launch: EcOpKernelLaunch,
}

/// Exact linked-module authority for the native EC-op composite.
///
/// `EcOpCompositeContract` seals source, ABI, effects and the ordered three-
/// launch contract. This receipt additionally seals the linked ordinary CUDA
/// archive and the one SM for which the consumer build is admissible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeEcOpLinkedModuleAuthority {
    pub(super) static_module_build_identity: [u8; 32],
    pub(super) expected_static_module_build_identity: [u8; 32],
    pub(super) consumer_target_sm: u32,
    pub(super) abi: EcOpCompositeAbi,
    pub(super) effect: EcOpEffectAbi,
    pub(super) source_identity: [u8; 32],
    pub(super) abi_identity: [u8; 32],
    pub(super) effect_identity: [u8; 32],
    pub(super) launch_identity: [u8; 32],
    pub(super) contract_identity: [u8; 32],
    pub(super) entry_symbol: &'static str,
    pub(super) launches: [NativeEcOpStaticLaunchAuthority; 3],
    pub(super) identity: [u8; 32],
}

/// One sealed pairing of linked code and exact compiled-proof invocation/effect.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeEcOpCompositeExecutionAuthority {
    pub(super) linked: NativeEcOpLinkedModuleAuthority,
    pub(super) invocation_identity: [u8; 32],
    pub(super) compiled_effect_identity: EffectContractId,
    pub(super) identity: [u8; 32],
}

impl NativeEcOpLinkedModuleAuthority {
    /// Bind the receipt exported by the exact static archive linked into this
    /// process. A no-CUDA stub is an honest missing authority; any claimed
    /// CUDA build whose receipt or target set is malformed is invalid.
    pub(super) fn bind_linked(
        contract: &EcOpCompositeContract,
    ) -> Result<Option<Self>, InvocationShapeError> {
        let linked =
            match stwo_backend_cuda_kernels::static_cuda_module_build_identity() {
                Ok(identity) => identity,
                Err(
                    stwo_backend_cuda_kernels::StaticCudaModuleBuildIdentityError::Unavailable(_),
                ) if !stwo_backend_cuda_kernels::CUDA_KERNELS_BUILT => return Ok(None),
                Err(_) => return Err(InvocationShapeError::InvalidNativeEcOpAuthority),
            };
        let expected = stwo_backend_cuda_kernels::expected_static_cuda_module_build_identity();
        let [consumer_target_sm] = stwo_backend_cuda_kernels::static_cuda_module_target_sms()
        else {
            return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
        };
        Self::derive_exact(contract, linked, expected, *consumer_target_sm).map(Some)
    }

    #[cfg(test)]
    pub(super) fn bind_exact(
        contract: &EcOpCompositeContract,
        linked: [u8; 32],
        expected: [u8; 32],
        consumer_target_sm: u32,
    ) -> Result<Self, InvocationShapeError> {
        Self::derive_exact(contract, linked, expected, consumer_target_sm)
    }

    #[cfg(test)]
    pub(super) fn validate(
        &self,
        contract: &EcOpCompositeContract,
        linked: [u8; 32],
        expected: [u8; 32],
        consumer_target_sm: u32,
    ) -> Result<(), InvocationShapeError> {
        let exact = Self::derive_exact(contract, linked, expected, consumer_target_sm)?;
        if self == &exact {
            Ok(())
        } else {
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        }
    }

    fn derive_exact(
        contract: &EcOpCompositeContract,
        linked: [u8; 32],
        expected: [u8; 32],
        consumer_target_sm: u32,
    ) -> Result<Self, InvocationShapeError> {
        if linked == ZERO_IDENTITY
            || linked != expected
            || consumer_target_sm == 0
            || contract.source_identity() == ZERO_IDENTITY
            || contract.abi_identity() == ZERO_IDENTITY
            || contract.effect_identity() == ZERO_IDENTITY
            || contract.launch_identity() == ZERO_IDENTITY
            || contract.identity() == ZERO_IDENTITY
        {
            return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
        }
        let launches = contract
            .launches()
            .map(|launch| NativeEcOpStaticLaunchAuthority {
                entry_symbol: launch.stage.symbol(),
                launch,
            });
        let mut authority = Self {
            static_module_build_identity: linked,
            expected_static_module_build_identity: expected,
            consumer_target_sm,
            abi: contract.abi(),
            effect: contract.effect(),
            source_identity: contract.source_identity(),
            abi_identity: contract.abi_identity(),
            effect_identity: contract.effect_identity(),
            launch_identity: contract.launch_identity(),
            contract_identity: contract.identity(),
            entry_symbol: contract.abi().entry_symbol(),
            launches,
            identity: ZERO_IDENTITY,
        };
        authority.identity = execution_identity(&authority);
        Ok(authority)
    }

    pub(super) fn bind_lowered(
        self,
        contract: &EcOpCompositeContract,
        invocation: &super::ec_op_prefix::StaticEcOpInvocation,
        effect: &EffectContract,
    ) -> Result<NativeEcOpCompositeExecutionAuthority, InvocationShapeError> {
        self.validate_contract(contract)?;
        validate_invocation_contract(contract, invocation)?;
        if exact_effect(invocation)? != *effect
            || !effect
                .has_valid_identity()
                .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)?
        {
            return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
        }
        let invocation_identity = invocation_identity(invocation)?;
        let compiled_effect_identity = effect.id();
        let identity = paired_identity(&self, invocation_identity, compiled_effect_identity);
        Ok(NativeEcOpCompositeExecutionAuthority {
            linked: self,
            invocation_identity,
            compiled_effect_identity,
            identity,
        })
    }

    pub(super) fn validate_active_sm(&self, active_sm: u32) -> Result<(), InvocationShapeError> {
        if active_sm == self.consumer_target_sm {
            Ok(())
        } else {
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        }
    }

    fn validate_contract(
        &self,
        contract: &EcOpCompositeContract,
    ) -> Result<(), InvocationShapeError> {
        let exact = Self::derive_exact(
            contract,
            self.static_module_build_identity,
            self.expected_static_module_build_identity,
            self.consumer_target_sm,
        )?;
        if self == &exact {
            Ok(())
        } else {
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        }
    }
}

impl NativeEcOpCompositeExecutionAuthority {
    pub(super) fn validate(
        &self,
        contract: &EcOpCompositeContract,
        invocation: &super::ec_op_prefix::StaticEcOpInvocation,
        effect: &EffectContract,
    ) -> Result<(), InvocationShapeError> {
        let trusted_linked = NativeEcOpLinkedModuleAuthority::bind_linked(contract)?
            .ok_or(InvocationShapeError::InvalidNativeEcOpAuthority)?;
        self.validate_against(&trusted_linked, contract, invocation, effect)
    }

    #[cfg(test)]
    pub(super) fn validate_exact(
        &self,
        trusted_linked: &NativeEcOpLinkedModuleAuthority,
        contract: &EcOpCompositeContract,
        invocation: &super::ec_op_prefix::StaticEcOpInvocation,
        effect: &EffectContract,
    ) -> Result<(), InvocationShapeError> {
        self.validate_against(trusted_linked, contract, invocation, effect)
    }

    fn validate_against(
        &self,
        trusted_linked: &NativeEcOpLinkedModuleAuthority,
        contract: &EcOpCompositeContract,
        invocation: &super::ec_op_prefix::StaticEcOpInvocation,
        effect: &EffectContract,
    ) -> Result<(), InvocationShapeError> {
        let exact = trusted_linked
            .clone()
            .bind_lowered(contract, invocation, effect)?;
        if self == &exact {
            Ok(())
        } else {
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        }
    }
}

fn execution_identity(authority: &NativeEcOpLinkedModuleAuthority) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINKED_IDENTITY_DOMAIN);
    hasher.update(&authority.static_module_build_identity);
    hasher.update(&authority.expected_static_module_build_identity);
    hasher.update(&authority.consumer_target_sm.to_le_bytes());
    hasher.update(&[authority.abi as u8, authority.effect as u8]);
    for identity in [
        authority.source_identity,
        authority.abi_identity,
        authority.effect_identity,
        authority.launch_identity,
        authority.contract_identity,
    ] {
        hasher.update(&identity);
    }
    hash_bytes(&mut hasher, authority.entry_symbol.as_bytes());
    for launch in authority.launches {
        hash_bytes(&mut hasher, launch.entry_symbol.as_bytes());
        hasher.update(&[launch.launch.stage as u8]);
        for value in launch.launch.grid.into_iter().chain(launch.launch.block) {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&launch.launch.dynamic_shared_bytes.to_le_bytes());
        hasher.update(&[u8::from(launch.launch.cooperative)]);
    }
    *hasher.finalize().as_bytes()
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&u64::try_from(bytes.len()).unwrap().to_le_bytes());
    hasher.update(bytes);
}

fn validate_invocation_contract(
    contract: &EcOpCompositeContract,
    invocation: &super::ec_op_prefix::StaticEcOpInvocation,
) -> Result<(), InvocationShapeError> {
    let requirements = contract.requirements();
    let tables = contract.execution_tables();
    let row_count =
        u32::try_from(requirements.row_count).map_err(|_| InvocationShapeError::SizeOverflow)?;
    let partial_row_count = u32::try_from(requirements.partial_row_count)
        .map_err(|_| InvocationShapeError::SizeOverflow)?;
    let pointer_words = core::mem::size_of::<*const u32>().div_ceil(core::mem::size_of::<u32>());
    if invocation.row_count != row_count
        || invocation.partial_row_count != partial_row_count
        || !exact_range(
            &invocation.execution_table_pointers,
            EXECUTION_TABLE_POINTERS
                .checked_mul(pointer_words)
                .ok_or(InvocationShapeError::SizeOverflow)?,
        )
        || invocation.execution_tables.len() != EXECUTION_TABLE_POINTERS
        || invocation.trace_columns.len() != requirements.trace_column_words.len()
        || invocation.partial_input_columns.len() != requirements.partial_input_column_words.len()
        || invocation.multiplicities.len() != 4
        || !exact_range(
            &invocation.segment_start.value,
            requirements.segment_start_words,
        )
        || !exact_range(&invocation.lookup_words.value, requirements.lookup_words)
    {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }

    let mut expected_table_words = Vec::with_capacity(EXECUTION_TABLE_POINTERS);
    expected_table_words.push(tables.n_addresses);
    expected_table_words.extend(std::iter::repeat_n(tables.n_big, EXECUTION_TABLE_BIG_LIMBS));
    expected_table_words.extend(std::iter::repeat_n(
        tables.n_small,
        EXECUTION_TABLE_SMALL_LIMBS,
    ));
    if !ranges_are_exact(&invocation.execution_tables, &expected_table_words)
        || !ranges_are_exact(&invocation.trace_columns, &requirements.trace_column_words)
        || !ranges_are_exact(
            &invocation.partial_input_columns,
            &requirements.partial_input_column_words,
        )
    {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }
    let multiplicity_words = [
        requirements.address_count_words,
        requirements.big_count_words,
        requirements.small_count_words,
        requirements.range_check_8_count_words,
    ];
    if invocation
        .multiplicities
        .iter()
        .zip(multiplicity_words)
        .any(|(binding, words)| !exact_range(&binding.value, words))
    {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }
    Ok(())
}

fn ranges_are_exact(bindings: &[NativeEcOpRangeBinding], words: &[usize]) -> bool {
    bindings.len() == words.len()
        && bindings
            .iter()
            .zip(words)
            .all(|(binding, &words)| exact_range(&binding.value, words))
}

fn exact_range(range: &ArenaCatalogRange, words: usize) -> bool {
    range.value_words.start == 0 && range.value_words.end == words
}

fn invocation_identity(
    invocation: &super::ec_op_prefix::StaticEcOpInvocation,
) -> Result<[u8; 32], InvocationShapeError> {
    let mut out = InvocationEncoder::new();
    out.range(&invocation.execution_table_pointers)?;
    out.range_bindings(&invocation.execution_tables)?;
    out.range_binding(&invocation.segment_start)?;
    out.range_bindings(&invocation.trace_columns)?;
    out.range_binding(&invocation.lookup_words)?;
    out.range_bindings(&invocation.partial_input_columns)?;
    out.atomics(&invocation.multiplicities)?;
    out.u32(invocation.row_count);
    out.u32(invocation.partial_row_count);
    Ok(*blake3::hash(&out.0).as_bytes())
}

fn paired_identity(
    linked: &NativeEcOpLinkedModuleAuthority,
    invocation: [u8; 32],
    effect: EffectContractId,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EXECUTION_IDENTITY_DOMAIN);
    hasher.update(&linked.identity);
    hasher.update(&invocation);
    hasher.update(effect.as_bytes());
    *hasher.finalize().as_bytes()
}

struct InvocationEncoder(Vec<u8>);

impl InvocationEncoder {
    fn new() -> Self {
        Self(INVOCATION_IDENTITY_DOMAIN.to_vec())
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn size(&mut self, value: usize) -> Result<(), InvocationShapeError> {
        self.0.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| InvocationShapeError::SizeOverflow)?
                .to_le_bytes(),
        );
        Ok(())
    }

    fn range(&mut self, range: &ArenaCatalogRange) -> Result<(), InvocationShapeError> {
        self.u32(range.value.0);
        self.size(range.value_words.start)?;
        self.size(range.value_words.end)
    }

    fn range_binding(
        &mut self,
        binding: &NativeEcOpRangeBinding,
    ) -> Result<(), InvocationShapeError> {
        self.range(&binding.value)?;
        match (binding.binding, binding.version) {
            (None, None) => self.byte(0),
            (Some(binding), Some(version)) => {
                self.byte(1);
                self.u32(binding.0);
                self.u32(version.0);
            }
            _ => return Err(InvocationShapeError::InvalidNativeEcOpBinding),
        }
        Ok(())
    }

    fn range_bindings(
        &mut self,
        bindings: &[NativeEcOpRangeBinding],
    ) -> Result<(), InvocationShapeError> {
        self.size(bindings.len())?;
        for binding in bindings {
            self.range_binding(binding)?;
        }
        Ok(())
    }

    fn atomics(
        &mut self,
        bindings: &[NativeEcOpAtomicBinding],
    ) -> Result<(), InvocationShapeError> {
        use crate::compiled_proof::{InPlaceAliasRequirement, InPlaceDiscipline};

        self.size(bindings.len())?;
        for binding in bindings {
            self.range(&binding.value)?;
            self.u32(binding.binding.0);
            self.u32(binding.source.0);
            self.u32(binding.destination.0);
            self.u32(binding.alias.id.0);
            self.byte(match binding.alias.requirement {
                InPlaceAliasRequirement::Permitted => 0,
                InPlaceAliasRequirement::Required => 1,
            });
            self.byte(match binding.alias.discipline {
                InPlaceDiscipline::ElementWiseReadBeforeWrite => 0,
                InPlaceDiscipline::BlockBarrierPhases => 1,
                InPlaceDiscipline::CooperativeGridPhases => 2,
                InPlaceDiscipline::ExactLowerPrefixReadBeforeWrite => 3,
                InPlaceDiscipline::OrderedCompositeInPlace => 4,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::{
        ec_op_workspace_requirements, EcOpExecutionTableShape, EcOpKernelStage,
        EcOpMultiplicityGeometry,
    };

    use super::*;

    fn contract_for_rows(row_count: usize) -> EcOpCompositeContract {
        EcOpCompositeContract::compile(
            &ec_op_workspace_requirements(
                row_count,
                EcOpMultiplicityGeometry {
                    address_count_words: 511,
                    big_count_words: 128,
                    small_count_words: 64,
                    range_check_8_count_words: 256,
                },
            )
            .unwrap(),
            EcOpExecutionTableShape {
                n_addresses: 512,
                n_big: 128,
                n_small: 64,
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_execution_authority_rejects_every_mutation() {
        let contract = contract_for_rows(32);
        let module = [9; 32];
        let authority =
            NativeEcOpLinkedModuleAuthority::bind_exact(&contract, module, module, 89).unwrap();
        authority.validate_active_sm(89).unwrap();
        assert_eq!(
            authority.validate_active_sm(90),
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        );
        assert_ne!(authority.identity, ZERO_IDENTITY);
        assert_eq!(authority.entry_symbol, "ec_op_builtin_witness_on");
        assert_eq!(
            authority.launches.map(|launch| launch.entry_symbol),
            [
                "ec_op_projective_chain_kernel",
                "ec_op_normalize_round_tiles_kernel",
                "partial_input_padding_kernel",
            ]
        );

        let mut mutations = Vec::new();
        for field in 0..8 {
            let mut changed = authority.clone();
            match field {
                0 => changed.static_module_build_identity[0] ^= 1,
                1 => changed.expected_static_module_build_identity[0] ^= 1,
                2 => changed.source_identity[0] ^= 1,
                3 => changed.abi_identity[0] ^= 1,
                4 => changed.effect_identity[0] ^= 1,
                5 => changed.launch_identity[0] ^= 1,
                6 => changed.contract_identity[0] ^= 1,
                7 => changed.identity[0] ^= 1,
                _ => unreachable!(),
            }
            mutations.push(changed);
        }
        let mut changed = authority.clone();
        changed.consumer_target_sm = 90;
        mutations.push(changed);
        let mut changed = authority.clone();
        changed.entry_symbol = "wrong_ec_op_entry";
        mutations.push(changed);
        for index in 0..3 {
            let mut changed = authority.clone();
            changed.launches[index].entry_symbol = "wrong_stage_entry";
            mutations.push(changed);
            let mut changed = authority.clone();
            changed.launches[index].launch.stage = match index {
                0 => EcOpKernelStage::NormalizeRoundTiles,
                1 | 2 => EcOpKernelStage::ProjectiveChain,
                _ => unreachable!(),
            };
            mutations.push(changed);
            for axis in 0..3 {
                let mut changed = authority.clone();
                changed.launches[index].launch.grid[axis] ^= 1;
                mutations.push(changed);
                let mut changed = authority.clone();
                changed.launches[index].launch.block[axis] ^= 1;
                mutations.push(changed);
            }
            let mut changed = authority.clone();
            changed.launches[index].launch.dynamic_shared_bytes = 4;
            mutations.push(changed);
            let mut changed = authority.clone();
            changed.launches[index].launch.cooperative = true;
            mutations.push(changed);
        }
        for changed in mutations {
            assert_eq!(
                changed.validate(&contract, module, module, 89),
                Err(InvocationShapeError::InvalidNativeEcOpAuthority)
            );
        }
        let mut recomputed_sm_mutation = authority.clone();
        recomputed_sm_mutation.consumer_target_sm = 90;
        recomputed_sm_mutation.identity = execution_identity(&recomputed_sm_mutation);
        assert_eq!(
            recomputed_sm_mutation.validate(&contract, module, module, 89),
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        );
        assert_ne!(
            authority.identity,
            NativeEcOpLinkedModuleAuthority::bind_exact(&contract, [8; 32], [8; 32], 89)
                .unwrap()
                .identity
        );
        assert_ne!(
            authority.identity,
            NativeEcOpLinkedModuleAuthority::bind_exact(&contract, module, module, 90)
                .unwrap()
                .identity
        );
        // ABI and effect are deliberately single-variant typed enums. Their
        // exact semantics are sealed by their identities: changing the legal
        // row geometry preserves the enum discriminants and ABI identity, but
        // changes the effect, launch and composite contract identities.
        let changed_contract = contract_for_rows(64);
        assert_eq!(changed_contract.abi(), contract.abi());
        assert_eq!(changed_contract.effect(), contract.effect());
        assert_eq!(changed_contract.abi_identity(), contract.abi_identity());
        assert_ne!(
            changed_contract.effect_identity(),
            contract.effect_identity()
        );
        assert_ne!(
            changed_contract.launch_identity(),
            contract.launch_identity()
        );
        assert_ne!(changed_contract.identity(), contract.identity());
        assert_eq!(
            authority.validate(&changed_contract, module, module, 89),
            Err(InvocationShapeError::InvalidNativeEcOpAuthority)
        );
        assert!(
            NativeEcOpLinkedModuleAuthority::bind_exact(&contract, [0; 32], [0; 32], 89).is_err()
        );
        assert!(
            NativeEcOpLinkedModuleAuthority::bind_exact(&contract, module, [8; 32], 89).is_err()
        );
        assert!(NativeEcOpLinkedModuleAuthority::bind_exact(&contract, module, module, 0).is_err());
    }
}
