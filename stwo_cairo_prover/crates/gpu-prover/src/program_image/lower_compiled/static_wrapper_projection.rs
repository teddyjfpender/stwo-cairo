//! Exact projection of linked native Base wrappers into `CompiledProof`.
//!
//! The upstream linked authorities remain the trust boundary. This module only
//! copies their address-free receipts into the generic wrapper form after
//! revalidating the exact lowered semantic effect and ordered internal launches.

use super::blake_g_direct_execution_authority::NativeBlakeGDirectLinkedModuleAuthority;
use super::ec_op_execution_authority::NativeEcOpLinkedModuleAuthority;
use super::{blake_g_direct_prefix, ec_op_prefix, static_wrapper_invocation, InvocationShapeError};
use crate::compiled_proof::{
    LaunchGeometry, StaticCudaLaunchIdentity, StaticCudaWrapperAuthority, StaticCudaWrapperId,
};

pub(super) fn ec_op(
    id: StaticCudaWrapperId,
    linked: &NativeEcOpLinkedModuleAuthority,
    lowered: &ec_op_prefix::LoweredNativeEcOpContract,
) -> Result<StaticCudaWrapperAuthority, InvocationShapeError> {
    let execution =
        linked
            .clone()
            .bind_lowered(&lowered.authority, &lowered.invocation, &lowered.effect)?;
    if execution.linked != *linked || execution.compiled_effect_identity != lowered.effect.id() {
        return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
    }
    let invocation = static_wrapper_invocation::ec_op(lowered)
        .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)?;
    let launches = linked
        .launches
        .iter()
        .map(|launch| {
            static_launch(
                launch.entry_symbol,
                launch.launch.grid,
                launch.launch.block,
                launch.launch.dynamic_shared_bytes,
                launch.launch.cooperative,
            )
            .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)
        })
        .collect::<Result<Vec<_>, _>>()?;
    StaticCudaWrapperAuthority::new(
        id,
        linked.static_module_build_identity,
        linked.consumer_target_sm,
        linked.entry_symbol.as_bytes().to_vec(),
        linked.abi_identity,
        linked.effect_identity,
        linked.contract_identity,
        linked.identity,
        launches,
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)?,
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidNativeEcOpAuthority)
}

pub(super) fn blake_g_direct(
    id: StaticCudaWrapperId,
    linked: &NativeBlakeGDirectLinkedModuleAuthority,
    lowered: &blake_g_direct_prefix::LoweredNativeBlakeGDirectContract,
) -> Result<StaticCudaWrapperAuthority, InvocationShapeError> {
    linked.validate_contract(&lowered.authority)?;
    blake_g_direct_prefix::validate_lowered(
        &lowered.authority,
        &lowered.invocation,
        &lowered.effect,
    )?;
    let invocation = static_wrapper_invocation::blake_g_direct(lowered)
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?;
    let contract = &linked.contract;
    let launch = contract.wrapper_launch();
    let launches = vec![static_launch(
        launch.audited_internal_kernel_symbol(),
        launch.grid,
        launch.block,
        launch.dynamic_shared_bytes,
        launch.cooperative,
    )
    .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?];
    StaticCudaWrapperAuthority::new(
        id,
        linked.static_module_build_identity,
        linked.consumer_target_sm,
        contract.abi().entry_symbol().as_bytes().to_vec(),
        contract.abi_identity(),
        contract.effect_identity(),
        contract.identity(),
        linked.identity,
        launches,
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?,
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)
}

fn static_launch(
    symbol: &str,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
    cooperative: bool,
) -> Result<StaticCudaLaunchIdentity, crate::compiled_proof::CompiledProofError> {
    StaticCudaLaunchIdentity::new(
        symbol.as_bytes().to_vec(),
        LaunchGeometry {
            grid,
            block,
            cluster: None,
            dynamic_shared_bytes,
            cooperative,
        },
    )
}
