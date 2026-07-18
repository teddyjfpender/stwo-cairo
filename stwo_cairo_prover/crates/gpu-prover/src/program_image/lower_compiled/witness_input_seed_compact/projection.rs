//! Exact linked-build, CUDA-library and host-wrapper projection.

use stwo_backend_cuda::{
    WitnessInputCompactAbiAccess, WitnessInputCompactAbiArgument,
    WitnessInputCompactAbiArgumentKind, WitnessInputCompactCubStage, WitnessInputCompactExecution,
    WitnessInputCompactIndexBuffer, WitnessInputCompactKernelLaunch, WitnessInputCompactKeyBuffer,
    WitnessInputCompactLinkedContract, WitnessInputSeedLinkedContract,
};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, LaunchGeometry, StaticCudaCubBuffer,
    StaticCudaCubInclusiveSumU32V1, StaticCudaCubStableAscendingSortPairsU32V1,
    StaticCudaExecutionStepIdentity, StaticCudaLaunchIdentity, StaticCudaLibraryCallIdentity,
};

pub(super) fn seed(
    id: StaticCudaWrapperId,
    linked: &WitnessInputSeedLinkedContract,
    lowered: &LoweredWitnessInputSeed,
) -> Result<LinkedWitnessInputSeedExecution, InvocationShapeError> {
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_seed_receipt(linked, &lowered.contract)?;
    let launch = lowered.contract.launch();
    let wrapper = StaticCudaWrapperAuthority::new(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        lowered.contract.abi().entry_symbol().as_bytes().to_vec(),
        lowered.contract.abi_identity(),
        lowered.contract.effect_identity(),
        lowered.contract.identity(),
        linked.identity(),
        vec![seed_launch(launch.symbol(), launch)?],
        lowered
            .invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?,
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LinkedWitnessInputSeedExecution { wrapper })
}

pub(super) fn compact(
    id: StaticCudaWrapperId,
    linked: &WitnessInputCompactLinkedContract,
    lowered: &LoweredWitnessInputCompact,
) -> Result<LinkedWitnessInputCompactExecution, InvocationShapeError> {
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    require_compact_receipt(linked, lowered)?;
    let steps = compact_steps(
        &lowered.contract,
        linked.sort_temp_bytes(),
        linked.scan_temp_bytes(),
    )?;
    let invocation = compact_invocation(
        lowered,
        linked.sort_temp_bytes(),
        linked.scan_temp_bytes(),
        lowered.contract.abi().arguments(),
    )?;
    super::semantic::validate_exact_bindings(
        &invocation,
        &lowered.effect,
        Some((lowered.descriptor_value, lowered.descriptor_binding)),
    )?;
    let wrapper = StaticCudaWrapperAuthority::new_with_execution_steps(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        lowered.contract.abi().entry_symbol().as_bytes().to_vec(),
        lowered.contract.abi_identity(),
        lowered.contract.effect_identity(),
        lowered.contract.identity(),
        linked.identity(),
        steps,
        invocation
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?,
        lowered.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LinkedWitnessInputCompactExecution {
        wrapper,
        invocation,
        exact_sort_temp_bytes: linked.sort_temp_bytes(),
        exact_scan_temp_bytes: linked.scan_temp_bytes(),
    })
}

fn require_seed_receipt(
    linked: &WitnessInputSeedLinkedContract,
    contract: &WitnessInputSeedContract,
) -> Result<(), InvocationShapeError> {
    if linked.contract_identity() != contract.identity()
        || linked.target_sm() < 10
        || [
            linked.module_build_identity(),
            linked.static_build_source_identity(),
            linked.static_build_identity(),
            linked.sm_identity(),
            linked.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn require_compact_receipt(
    linked: &WitnessInputCompactLinkedContract,
    lowered: &LoweredWitnessInputCompact,
) -> Result<(), InvocationShapeError> {
    let scratch = lowered.contract.effect_geometry().scratch;
    let sort_capacity = scratch
        .sort_temp_capacity_words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let scan_capacity = scratch
        .scan_temp_capacity_words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if linked.contract_identity() != lowered.contract.identity()
        || linked.target_sm() < 10
        || linked.sort_temp_bytes() == 0
        || linked.sort_temp_bytes() > sort_capacity
        || linked.scan_temp_bytes() == 0
        || linked.scan_temp_bytes() > scan_capacity
        || [
            linked.module_build_identity(),
            linked.static_build_source_identity(),
            linked.static_build_identity(),
            linked.sm_identity(),
            linked.runtime_scratch_identity(),
            linked.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn compact_steps(
    contract: &WitnessInputCompactContract,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
) -> Result<Vec<StaticCudaExecutionStepIdentity>, InvocationShapeError> {
    let rows = contract.effect_geometry().sort_rows;
    let sort_temp =
        u64::try_from(exact_sort_temp_bytes).map_err(|_| InvocationShapeError::SizeOverflow)?;
    let scan_temp =
        u64::try_from(exact_scan_temp_bytes).map_err(|_| InvocationShapeError::SizeOverflow)?;
    contract
        .stages()
        .iter()
        .enumerate()
        .map(|(ordinal, stage)| {
            if stage.ordinal as usize != ordinal {
                return Err(InvocationShapeError::InvalidStructuredAbi);
            }
            match stage.execution {
                WitnessInputCompactExecution::Kernel { stage, launch } => {
                    Ok(StaticCudaExecutionStepIdentity::KernelLaunch(
                        compact_launch(stage.symbol(), launch)?,
                    ))
                }
                WitnessInputCompactExecution::Cub {
                    stage,
                    library_managed_launch_geometry,
                    ordered_on_wrapper_stream,
                } => {
                    if !library_managed_launch_geometry || !ordered_on_wrapper_stream {
                        return Err(InvocationShapeError::InvalidStructuredAbi);
                    }
                    let call = match stage {
                        WitnessInputCompactCubStage::StableRadixSortPairs {
                            word,
                            keys_from,
                            keys_to,
                            indices_from,
                            indices_to,
                            begin_bit,
                            end_bit,
                        } => StaticCudaLibraryCallIdentity::CubStableAscendingSortPairsU32V1(
                            StaticCudaCubStableAscendingSortPairsU32V1::new(
                                word,
                                key_buffer(keys_from),
                                key_buffer(keys_to),
                                index_buffer(indices_from),
                                index_buffer(indices_to),
                                begin_bit,
                                end_bit,
                                rows,
                                sort_temp,
                            )
                            .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?,
                        ),
                        WitnessInputCompactCubStage::InclusiveSum => {
                            StaticCudaLibraryCallIdentity::CubInclusiveSumU32V1(
                                StaticCudaCubInclusiveSumU32V1::new(rows, scan_temp)
                                    .map_err(|_| InvocationShapeError::InvalidStructuredAbi)?,
                            )
                        }
                    };
                    Ok(StaticCudaExecutionStepIdentity::LibraryCall(call))
                }
            }
        })
        .collect()
}

fn compact_invocation(
    lowered: &LoweredWitnessInputCompact,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
    abi: &[WitnessInputCompactAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    let contract = &lowered.contract;
    if abi != contract.abi().arguments()
        || abi
            .iter()
            .enumerate()
            .any(|(index, argument)| argument.ordinal as usize != index)
        || lowered.scratch.len() != 10
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let fixed = contract.fixed_words();
    let mut arguments = Vec::with_capacity(abi.len() - 1);
    for descriptor in abi {
        use WitnessInputCompactAbiAccess as Access;
        use WitnessInputCompactAbiArgumentKind as Kind;
        let value = match (
            descriptor.ordinal,
            descriptor.name,
            descriptor.kind,
            descriptor.access,
        ) {
            (
                0,
                "producer_subs_dev",
                Kind::DeviceConstPointerTableU32,
                Access::ReadPackedProducerColumns,
            ) => AotArgumentValue::DevicePointerTable(
                lowered
                    .sources
                    .iter()
                    .map(|source| Some(source.binding))
                    .collect(),
            ),
            (
                1,
                "edge_descs_dev",
                Kind::DeviceConstPointerU32,
                Access::ReadCanonicalEdgeDescriptors,
            ) => AotArgumentValue::DeviceFixedU32 {
                value: lowered.descriptor_value,
                binding: lowered.descriptor_binding,
            },
            (2, "n_edges", Kind::U32, Access::EdgeCount) => AotArgumentValue::U32(fixed[0]),
            (3, "tuple_words", Kind::U32, Access::TupleWords) => AotArgumentValue::U32(fixed[1]),
            (4, "key_words", Kind::U32, Access::KeyWords) => AotArgumentValue::U32(fixed[2]),
            (5, "total_rows", Kind::U32, Access::TotalRows) => AotArgumentValue::U32(fixed[3]),
            (6, "sort_rows", Kind::U32, Access::SortRows) => AotArgumentValue::U32(fixed[4]),
            (7, "consumer_rows", Kind::U32, Access::ConsumerRows) => {
                AotArgumentValue::U32(fixed[5])
            }
            (8, "n_inputs", Kind::U32, Access::ConsumerInputCount) => {
                AotArgumentValue::U32(fixed[6])
            }
            (
                9,
                "consumer_cols_dev",
                Kind::DeviceMutPointerTableU32,
                Access::WriteConsumerColumns,
            ) => AotArgumentValue::DevicePointerTable(
                lowered
                    .outputs
                    .iter()
                    .map(|output| Some(output.binding))
                    .collect(),
            ),
            (10, "enabler_slot", Kind::U32, Access::EnablerSlot) => AotArgumentValue::U32(fixed[7]),
            (11, "iota_slot", Kind::U32, Access::IotaSlot) => AotArgumentValue::U32(fixed[8]),
            (12, "multiplicity_slot", Kind::U32, Access::MultiplicitySlot) => {
                AotArgumentValue::U32(fixed[9])
            }
            (13, "tuples_dev", Kind::DeviceMutPointerU32, Access::TupleScratch) => {
                pointer(&lowered.scratch[0])
            }
            (14, "keys_a_dev", Kind::DeviceMutPointerU32, Access::SortKeysA) => {
                pointer(&lowered.scratch[1])
            }
            (15, "keys_b_dev", Kind::DeviceMutPointerU32, Access::SortKeysB) => {
                pointer(&lowered.scratch[2])
            }
            (16, "indices_a_dev", Kind::DeviceMutPointerU32, Access::SortIndicesA) => {
                pointer(&lowered.scratch[3])
            }
            (17, "indices_b_dev", Kind::DeviceMutPointerU32, Access::SortIndicesB) => {
                pointer(&lowered.scratch[4])
            }
            (18, "heads_dev", Kind::DeviceMutPointerU32, Access::RunHeads) => {
                pointer(&lowered.scratch[5])
            }
            (19, "positions_dev", Kind::DeviceMutPointerU32, Access::RunPositions) => {
                pointer(&lowered.scratch[6])
            }
            (20, "n_unique_dev", Kind::DeviceMutPointerU32, Access::UniqueCount) => {
                pointer(&lowered.scratch[7])
            }
            (21, "sort_temp_dev", Kind::DeviceMutPointerBytes, Access::SortScratch) => {
                pointer(&lowered.scratch[8])
            }
            (22, "sort_temp_bytes", Kind::Usize, Access::SortScratchBytes) => {
                AotArgumentValue::Usize(
                    u64::try_from(exact_sort_temp_bytes)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                )
            }
            (23, "scan_temp_dev", Kind::DeviceMutPointerBytes, Access::ScanScratch) => {
                pointer(&lowered.scratch[9])
            }
            (24, "scan_temp_bytes", Kind::Usize, Access::ScanScratchBytes) => {
                AotArgumentValue::Usize(
                    u64::try_from(exact_scan_temp_bytes)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                )
            }
            (25, "stream", Kind::CudaStream, Access::OrderedExecutionStream) => continue,
            _ => return Err(InvocationShapeError::InvalidStructuredAbi),
        };
        arguments.push(AotArgumentBinding {
            ordinal: descriptor.ordinal,
            value,
        });
    }
    Ok(AotInvocation { arguments })
}

fn pointer(binding: &SemanticArenaBinding) -> AotArgumentValue {
    AotArgumentValue::DevicePointer(Some(binding.binding))
}

fn seed_launch(
    symbol: &str,
    launch: stwo_backend_cuda::WitnessInputSeedKernelLaunch,
) -> Result<StaticCudaLaunchIdentity, InvocationShapeError> {
    static_launch(
        symbol,
        launch.grid,
        launch.block,
        launch.cluster,
        launch.dynamic_shared_bytes,
        launch.cooperative,
    )
}

fn compact_launch(
    symbol: &str,
    launch: WitnessInputCompactKernelLaunch,
) -> Result<StaticCudaLaunchIdentity, InvocationShapeError> {
    static_launch(
        symbol,
        launch.grid,
        launch.block,
        launch.cluster,
        launch.dynamic_shared_bytes,
        launch.cooperative,
    )
}

fn static_launch(
    symbol: &str,
    grid: [u32; 3],
    block: [u32; 3],
    cluster: Option<[u32; 3]>,
    dynamic_shared_bytes: u32,
    cooperative: bool,
) -> Result<StaticCudaLaunchIdentity, InvocationShapeError> {
    StaticCudaLaunchIdentity::new(
        symbol.as_bytes().to_vec(),
        LaunchGeometry {
            grid,
            block,
            cluster,
            dynamic_shared_bytes,
            cooperative,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidStructuredAbi)
}

const fn key_buffer(buffer: WitnessInputCompactKeyBuffer) -> StaticCudaCubBuffer {
    match buffer {
        WitnessInputCompactKeyBuffer::A => StaticCudaCubBuffer::A,
        WitnessInputCompactKeyBuffer::B => StaticCudaCubBuffer::B,
    }
}

const fn index_buffer(buffer: WitnessInputCompactIndexBuffer) -> StaticCudaCubBuffer {
    match buffer {
        WitnessInputCompactIndexBuffer::A => StaticCudaCubBuffer::A,
        WitnessInputCompactIndexBuffer::B => StaticCudaCubBuffer::B,
    }
}

#[cfg(test)]
pub(super) fn compact_steps_for_test(
    contract: &WitnessInputCompactContract,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
) -> Result<Vec<StaticCudaExecutionStepIdentity>, InvocationShapeError> {
    compact_steps(contract, exact_sort_temp_bytes, exact_scan_temp_bytes)
}

#[cfg(test)]
pub(super) fn compact_invocation_for_test(
    lowered: &LoweredWitnessInputCompact,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
) -> Result<AotInvocation, InvocationShapeError> {
    compact_invocation(
        lowered,
        exact_sort_temp_bytes,
        exact_scan_temp_bytes,
        lowered.contract.abi().arguments(),
    )
}

#[cfg(test)]
pub(super) fn compact_invocation_using_abi_for_test(
    lowered: &LoweredWitnessInputCompact,
    exact_sort_temp_bytes: usize,
    exact_scan_temp_bytes: usize,
    abi: &[WitnessInputCompactAbiArgument],
) -> Result<AotInvocation, InvocationShapeError> {
    compact_invocation(lowered, exact_sort_temp_bytes, exact_scan_temp_bytes, abi)
}
