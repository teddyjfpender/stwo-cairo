//! Narrow executable bridge for the first compiled Base operation.
//!
//! This is deliberately not a second scheduler. It admits the exact first
//! operation already emitted by [`CompiledWitnessWriterPrefix`], installs its
//! one external F252 root, calls the linked static wrapper, and checks every
//! output word against the scalar split.

use stwo_backend_cuda::{
    cuda_device_snapshot, ArenaError, ArenaLayout, ArenaSlice, ArenaSlotSpec, CudaRuntimeError,
    DeviceArena, ExecutionTablesHostData, ExecutionTablesStage, PreparedExecutionTablesError,
    PreparedExecutionTablesGraph, EXECUTION_TABLE_BIG_LIMBS, EXECUTION_TABLE_SMALL_LIMBS,
};

use super::super::{adapter, execution_tables};
use super::CompiledWitnessWriterPrefix;
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{ExecutionPrimitive, LaunchGeometry, OpId};
use crate::resident_input::ResidentProverInputOwner;

const OUTPUT_DIGEST_DOMAIN: &[u8] = b"stwo-cairo.compiled-prefix.execution-tables.output.v1\0";
const INPUT_DIGEST_DOMAIN: &[u8] = b"stwo-cairo.compiled-prefix.execution-tables.input.v1\0";
const WRAPPER_SYMBOLS: [&[u8]; 2] = [
    b"memory_limb_split_big_columns_on",
    b"memory_limb_split_small_columns_on",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) struct FirstExecutionTableBigReceipt {
    pub(super) operations: [OpId; 2],
    pub(super) wrapper_symbols: [Box<[u8]>; 2],
    pub(super) rows: [usize; 3],
    pub(super) column_rows: [usize; 2],
    pub(super) root_h2d_bytes: usize,
    pub(super) metadata_h2d_bytes: usize,
    pub(super) validation_d2h_bytes: usize,
    pub(super) input_digest: [u8; 32],
    pub(super) output_digest: [u8; 32],
    pub(super) next_unsupported_operation: OpId,
    pub(super) next_unsupported_wrapper_symbol: Box<[u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) enum FirstExecutionTableBigError {
    InvalidPrefixAuthority,
    DeviceTargetMismatch {
        compiled_sm: u32,
        active_sm: u32,
    },
    SizeOverflow,
    Arena(ArenaError),
    Prepared(PreparedExecutionTablesError),
    Cuda(CudaRuntimeError),
    OutputMismatch {
        role: &'static str,
        column: usize,
        row: usize,
        expected: u32,
        actual: u32,
    },
}

impl core::fmt::Display for FirstExecutionTableBigError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "first compiled execution-table Big operation rejected: {self:?}"
        )
    }
}

impl std::error::Error for FirstExecutionTableBigError {}

impl From<PreparedExecutionTablesError> for FirstExecutionTableBigError {
    fn from(value: PreparedExecutionTablesError) -> Self {
        Self::Prepared(value)
    }
}

impl From<ArenaError> for FirstExecutionTableBigError {
    fn from(value: ArenaError) -> Self {
        Self::Arena(value)
    }
}

impl From<CudaRuntimeError> for FirstExecutionTableBigError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Cuda(value)
    }
}

/// Compact allocation containing only the prepared execution-table workspace.
///
/// This preserves every semantic slot identity while avoiding allocation of
/// unrelated proof slabs during the first-operation development gate.
pub(in crate::program_image::lower_compiled) fn first_execution_table_arena_layout(
    plan: &ProofArenaPlan,
) -> Result<ArenaLayout, FirstExecutionTableBigError> {
    let planned = plan
        .execution_tables()
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let requirements = planned
        .requirements
        .arena_slot_requirements(&planned.slots)?;
    let mut offset_words = 0usize;
    let mut specs = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let remainder = offset_words % requirement.alignment_words;
        if remainder != 0 {
            offset_words = offset_words
                .checked_add(requirement.alignment_words - remainder)
                .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
        }
        specs.push(ArenaSlotSpec {
            id: requirement.id,
            offset_words,
            len_words: requirement.len_words,
            alignment_words: requirement.alignment_words,
        });
        offset_words = offset_words
            .checked_add(requirement.len_words)
            .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    }
    ArenaLayout::new(offset_words, &specs).map_err(Into::into)
}

/// Execute operation zero of an admitted witness-writer prefix.
///
/// The caller owns `arena` and therefore the stream lifetime. This call fences
/// the borrowed host input before returning and fences each diagnostic D2H
/// column before reading it. Those validation fences are outside the eventual
/// hot fleet path.
pub(in crate::program_image::lower_compiled) fn execute_first_execution_table_big(
    prefix: &CompiledWitnessWriterPrefix,
    plan: &ProofArenaPlan,
    arena: &DeviceArena,
    owner: &ResidentProverInputOwner,
) -> Result<FirstExecutionTableBigReceipt, FirstExecutionTableBigError> {
    let planned = plan
        .execution_tables()
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let (lowered, operations, wrappers, next_operation, next_wrapper) =
        exact_execution_table_frontier(prefix, plan)?;
    let active = cuda_device_snapshot()?;
    let active_sm = active
        .sm_major
        .checked_mul(10)
        .and_then(|major| major.checked_add(active.sm_minor))
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    if active_sm != prefix.target_sm {
        return Err(FirstExecutionTableBigError::DeviceTargetMismatch {
            compiled_sm: prefix.target_sm,
            active_sm,
        });
    }

    let prepared =
        PreparedExecutionTablesGraph::prepare(arena, &planned.requirements, &planned.slots)?;
    validate_loaded_ranges(&lowered, &prepared)?;
    let host = owner.execution_tables_host_data();
    let ingest = prepared.ingest(host)?;
    let launch = prepared.launch()?;
    if launch.kernel_launches != 2
        || launch.allocations != 0
        || launch.h2d_bytes != 0
        || launch.d2h_bytes != 0
        || launch.d2d_bytes != 0
        || launch.sync_calls != 0
    {
        return Err(FirstExecutionTableBigError::InvalidPrefixAuthority);
    }
    let (output_digest, validation_d2h_bytes) = validate_outputs(arena, &prepared, host)?;
    let root_h2d_bytes = usize::try_from(ingest.compact_h2d_bytes)
        .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?;
    let metadata_h2d_bytes = usize::try_from(ingest.descriptor_h2d_bytes)
        .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?;
    Ok(FirstExecutionTableBigReceipt {
        operations: operations.map(|operation| operation.id),
        wrapper_symbols: wrappers.map(|wrapper| wrapper.wrapper_symbol().into()),
        rows: [
            planned.requirements.n_addrs,
            planned.requirements.n_big,
            planned.requirements.n_small,
        ],
        column_rows: [
            planned.requirements.big_column_words,
            planned.requirements.small_column_words,
        ],
        root_h2d_bytes,
        metadata_h2d_bytes,
        validation_d2h_bytes,
        input_digest: input_digest(host)?,
        output_digest,
        next_unsupported_operation: next_operation.id,
        next_unsupported_wrapper_symbol: next_wrapper.wrapper_symbol().into(),
    })
}

fn exact_execution_table_frontier<'a>(
    prefix: &'a CompiledWitnessWriterPrefix,
    plan: &ProofArenaPlan,
) -> Result<
    (
        execution_tables::LoweredExecutionTables,
        [&'a crate::compiled_proof::OpNode; 2],
        [&'a crate::compiled_proof::StaticCudaWrapperAuthority; 2],
        &'a crate::compiled_proof::OpNode,
        &'a crate::compiled_proof::StaticCudaWrapperAuthority,
    ),
    FirstExecutionTableBigError,
> {
    let mut values = adapter::SemanticValueMap::allocate_ordered(core::iter::empty())
        .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let lowered = execution_tables::lower_stage(plan, &mut values)
        .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let operations = [
        exact_static_operation(prefix, &lowered, 0)?,
        exact_static_operation(prefix, &lowered, 1)?,
    ];
    let next_operation = prefix
        .operations
        .get(2)
        .filter(|operation| operation.id == OpId(2))
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let ExecutionPrimitive::StaticCudaWrapper {
        wrapper: next_wrapper_id,
    } = next_operation.primitive
    else {
        return Err(FirstExecutionTableBigError::InvalidPrefixAuthority);
    };
    let next_wrapper = prefix
        .static_wrappers
        .iter()
        .find(|wrapper| wrapper.id() == next_wrapper_id)
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    Ok((
        lowered,
        [operations[0].0, operations[1].0],
        [operations[0].1, operations[1].1],
        next_operation,
        next_wrapper,
    ))
}

fn exact_static_operation<'a>(
    prefix: &'a CompiledWitnessWriterPrefix,
    lowered: &execution_tables::LoweredExecutionTables,
    index: usize,
) -> Result<
    (
        &'a crate::compiled_proof::OpNode,
        &'a crate::compiled_proof::StaticCudaWrapperAuthority,
    ),
    FirstExecutionTableBigError,
> {
    let stage = lowered
        .stages
        .get(index)
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let operation = prefix
        .operations
        .get(index)
        .filter(|operation| operation.id == OpId(index as u32))
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let ExecutionPrimitive::StaticCudaWrapper {
        wrapper: wrapper_id,
    } = operation.primitive
    else {
        return Err(FirstExecutionTableBigError::InvalidPrefixAuthority);
    };
    let wrapper = prefix
        .static_wrappers
        .iter()
        .find(|wrapper| wrapper.id() == wrapper_id)
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let effect = prefix
        .effects
        .iter()
        .find(|effect| effect.id() == operation.effect)
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let contract_stage = lowered
        .contract
        .stages()
        .get(index)
        .filter(|contract| contract.stage() == stage.stage)
        .ok_or(FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let launch = contract_stage.launch();
    let expected_launch = LaunchGeometry {
        grid: launch.grid,
        block: launch.block,
        cluster: launch.cluster,
        dynamic_shared_bytes: launch.dynamic_shared_bytes,
        cooperative: launch.cooperative,
    };
    let launches = wrapper.kernel_launches().collect::<Vec<_>>();
    let exact = stage.stage == [ExecutionTablesStage::Big, ExecutionTablesStage::Small][index]
        && operation.invocation.as_ref() == Some(&stage.invocation)
        && effect == &stage.effect
        && wrapper.wrapper_symbol() == WRAPPER_SYMBOLS[index]
        && wrapper.consumer_target_sm() == prefix.target_sm
        && wrapper.semantic_abi_identity() == &lowered.contract.abi_identity()
        && wrapper.semantic_effect_identity() == &lowered.contract.effect_identity()
        && wrapper.aggregate_contract_identity() == &lowered.contract.identity()
        && wrapper.accepted_invocation()
            == stage
                .invocation
                .contract_id()
                .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?
        && wrapper.accepted_effect() == stage.effect.id()
        && wrapper.execution_steps().len() == 1
        && launches.len() == 1
        && launches[0].symbol() == launch.symbol().as_bytes()
        && launches[0].launch() == expected_launch;
    if exact {
        Ok((operation, wrapper))
    } else {
        Err(FirstExecutionTableBigError::InvalidPrefixAuthority)
    }
}

fn validate_loaded_ranges(
    lowered: &execution_tables::LoweredExecutionTables,
    prepared: &PreparedExecutionTablesGraph<'_>,
) -> Result<(), FirstExecutionTableBigError> {
    let raw = [
        prepared.raw_addr_to_id(),
        prepared.raw_f252_words(),
        prepared.raw_small_words(),
    ];
    let exact = raw
        .iter()
        .zip(&lowered.host_ingress)
        .all(|(actual, expected)| {
            actual.id() == expected.arena.physical && actual.len_words() == expected.arena.len_words
        })
        && [
            (
                prepared.table_pointers(),
                lowered.relocations.table_pointers,
            ),
            (prepared.table_strides(), lowered.relocations.table_strides),
        ]
        .iter()
        .all(|(actual, expected)| {
            actual.id() == expected.physical && actual.len_words() == expected.len_words
        })
        && [prepared.big_limbs(), prepared.small_limbs()]
            .iter()
            .zip(&lowered.stages)
            .all(|(actual, stage)| {
                actual.len() == stage.outputs.len()
                    && actual.iter().zip(&stage.outputs).all(|(actual, expected)| {
                        actual.id() == expected.arena.physical
                            && actual.len_words() == expected.arena.len_words
                    })
            });
    if exact {
        Ok(())
    } else {
        Err(FirstExecutionTableBigError::InvalidPrefixAuthority)
    }
}

fn validate_outputs(
    arena: &DeviceArena,
    prepared: &PreparedExecutionTablesGraph<'_>,
    host: ExecutionTablesHostData<'_>,
) -> Result<([u8; 32], usize), FirstExecutionTableBigError> {
    let requirements = prepared.requirements();
    let mut hasher = blake3::Hasher::new();
    hasher.update(OUTPUT_DIGEST_DOMAIN);
    for size in [
        requirements.n_addrs,
        requirements.n_big,
        requirements.n_small,
        requirements.big_column_words,
        requirements.small_column_words,
    ] {
        hash_size(&mut hasher, size)?;
    }

    let address_bytes = host
        .addr_to_id
        .len()
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    if address_bytes != 0 {
        let mut actual = vec![0u32; host.addr_to_id.len()];
        let source = prepared
            .raw_addr_to_id()
            .checked_subslice(0, actual.len())
            .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?;
        unsafe {
            arena.context().memcpy_d2h_async(
                actual.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                address_bytes,
            )?;
        }
        arena.context().sync()?;
        for (row, (&actual, &expected)) in actual.iter().zip(host.addr_to_id).enumerate() {
            if actual != expected {
                return Err(FirstExecutionTableBigError::OutputMismatch {
                    role: "addr_to_id",
                    column: 0,
                    row,
                    expected,
                    actual,
                });
            }
        }
        hasher.update(bytemuck::cast_slice(&actual));
    }

    let big_bytes = validate_columns(
        arena,
        prepared.big_limbs(),
        requirements.big_column_words,
        "f252_limbs",
        |row, limb| expected_big_limb(host.f252_values, row, limb),
        &mut hasher,
    )?;
    let small_bytes = validate_columns(
        arena,
        prepared.small_limbs(),
        requirements.small_column_words,
        "small_limbs",
        |row, limb| expected_small_limb(host.small_values, row, limb),
        &mut hasher,
    )?;
    let validation_d2h_bytes = address_bytes
        .checked_add(big_bytes)
        .and_then(|bytes| bytes.checked_add(small_bytes))
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    Ok((*hasher.finalize().as_bytes(), validation_d2h_bytes))
}

fn validate_columns(
    arena: &DeviceArena,
    sources: &[ArenaSlice],
    column_rows: usize,
    role: &'static str,
    expected: impl Fn(usize, usize) -> u32,
    hasher: &mut blake3::Hasher,
) -> Result<usize, FirstExecutionTableBigError> {
    let column_bytes = column_rows
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    let total_bytes = column_bytes
        .checked_mul(sources.len())
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    let mut values = vec![0u32; column_rows];
    for (column, source) in sources.iter().copied().enumerate() {
        unsafe {
            arena.context().memcpy_d2h_async(
                values.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                column_bytes,
            )?;
        }
        arena.context().sync()?;
        for (row, &actual) in values.iter().enumerate() {
            let expected = expected(row, column);
            if actual != expected {
                return Err(FirstExecutionTableBigError::OutputMismatch {
                    role,
                    column,
                    row,
                    expected,
                    actual,
                });
            }
        }
        hasher.update(bytemuck::cast_slice(&values));
    }
    Ok(total_bytes)
}

fn expected_big_limb(values: &[[u32; 8]], row: usize, limb: usize) -> u32 {
    let Some(words) = values.get(row) else {
        return 0;
    };
    expected_limb(words, limb)
}

fn expected_small_limb(values: &[u128], row: usize, limb: usize) -> u32 {
    let Some(&value) = values.get(row) else {
        return 0;
    };
    expected_limb(
        &[
            value as u32,
            (value >> 32) as u32,
            (value >> 64) as u32,
            (value >> 96) as u32,
        ],
        limb,
    )
}

fn expected_limb(words: &[u32], limb: usize) -> u32 {
    let bit = limb * 9;
    let word = bit / 32;
    let shift = bit % 32;
    let low = words[word] >> shift;
    let high = if shift > 23 {
        words.get(word + 1).copied().unwrap_or(0) << (32 - shift)
    } else {
        0
    };
    (low | high) & 0x1ff
}

fn input_digest(
    host: ExecutionTablesHostData<'_>,
) -> Result<[u8; 32], FirstExecutionTableBigError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INPUT_DIGEST_DOMAIN);
    for size in [
        host.addr_to_id.len(),
        host.f252_values.len(),
        host.small_values.len(),
    ] {
        hash_size(&mut hasher, size)?;
    }
    hasher.update(bytemuck::cast_slice(host.addr_to_id));
    hasher.update(bytemuck::cast_slice(host.f252_values));
    for value in host.small_values {
        hasher.update(&value.to_le_bytes());
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_size(hasher: &mut blake3::Hasher, size: usize) -> Result<(), FirstExecutionTableBigError> {
    hasher.update(
        &u64::try_from(size)
            .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_bit_projection_matches_the_generic_nine_bit_split() {
        let values = [[
            0x0123_4567,
            0x89ab_cdef,
            0xfedc_ba98,
            0x7654_3210,
            0x0f0f_f0f0,
            0x55aa_aa55,
            0xffff_0000,
            0x000f_ffff,
        ]];
        let expected = split_nine_bit(&values[0]);
        for (limb, &value) in expected.iter().enumerate() {
            assert_eq!(expected_big_limb(&values, 0, limb), value);
        }
        assert_eq!(expected_big_limb(&values, 1, 0), 0);

        let small = [0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128];
        let words = [
            small[0] as u32,
            (small[0] >> 32) as u32,
            (small[0] >> 64) as u32,
            (small[0] >> 96) as u32,
        ];
        for (limb, &value) in split_nine_bit(&words)[..EXECUTION_TABLE_SMALL_LIMBS]
            .iter()
            .enumerate()
        {
            assert_eq!(expected_small_limb(&small, 0, limb), value);
        }
    }

    #[test]
    fn generated_sn2_compact_layout_contains_every_execution_table_slot_exactly() {
        let executable = super::super::super::tests::generated_sn2_replacement();
        let planned = executable.arena().execution_tables().unwrap();
        let layout = first_execution_table_arena_layout(executable.arena()).unwrap();
        let requirements = planned
            .requirements
            .arena_slot_requirements(&planned.slots)
            .unwrap();
        for requirement in requirements {
            let slot = layout.slot(requirement.id).unwrap();
            assert_eq!(slot.len_words, requirement.len_words);
            assert_eq!(slot.alignment_words, requirement.alignment_words);
        }
        assert_ne!(layout.total_words(), 0);
    }

    fn split_nine_bit(words: &[u32]) -> [u32; EXECUTION_TABLE_BIG_LIMBS] {
        let mut result = [0; EXECUTION_TABLE_BIG_LIMBS];
        let mut bits_left = 32u32;
        let mut word_index = 0usize;
        let mut word = words[0];
        for limb in &mut result {
            if bits_left > 9 {
                *limb = word & 0x1ff;
                word >>= 9;
                bits_left -= 9;
                continue;
            }
            *limb = word;
            word_index += 1;
            word = words.get(word_index).copied().unwrap_or(0);
            if bits_left < 9 {
                *limb |= (word << bits_left) & 0x1ff;
                word >>= 9 - bits_left;
            }
            bits_left += 32 - 9;
        }
        result
    }
}
