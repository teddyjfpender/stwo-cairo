//! Narrow executable bridge for the first compiled Base operation.
//!
//! This is deliberately not a second scheduler. It admits the exact first
//! operation already emitted by [`CompiledWitnessWriterPrefix`], installs its
//! one external F252 root, calls the linked static wrapper, and checks every
//! output word against the scalar split.

use core::ffi::c_void;

use stwo_backend_cuda::{
    cuda_device_snapshot, ArenaError, ArenaLayout, ArenaSlotSpec, CudaRuntimeError, DeviceArena,
    ExecutionTablesStage, PreparedExecutionTablesError, PreparedExecutionTablesGraph,
    EXECUTION_TABLE_BIG_LIMBS,
};

use super::super::{adapter, execution_tables};
use super::CompiledWitnessWriterPrefix;
use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{ExecutionPrimitive, LaunchGeometry, OpId};
use crate::resident_input::ResidentProverInputOwner;

const OUTPUT_DIGEST_DOMAIN: &[u8] = b"stwo-cairo.compiled-prefix.execution-table-big.output.v1\0";
const INPUT_DIGEST_DOMAIN: &[u8] = b"stwo-cairo.compiled-prefix.execution-table-big.input.v1\0";
const WRAPPER_SYMBOL: &[u8] = b"memory_limb_split_big_columns_on";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) struct FirstExecutionTableBigReceipt {
    pub(super) operation: OpId,
    pub(super) wrapper_symbol: Box<[u8]>,
    pub(super) real_rows: usize,
    pub(super) column_rows: usize,
    pub(super) input_h2d_bytes: usize,
    pub(super) validation_d2h_bytes: usize,
    pub(super) input_digest: [u8; 32],
    pub(super) output_digest: [u8; 32],
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
        limb: usize,
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
    let (lowered, operation, wrapper) = exact_first_operation(prefix, plan)?;
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
    if host.f252_values.len() != planned.requirements.n_big {
        return Err(FirstExecutionTableBigError::InvalidPrefixAuthority);
    }
    let input_words = host
        .f252_values
        .len()
        .checked_mul(8)
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    let input_h2d_bytes = input_words
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    if input_h2d_bytes != 0 {
        unsafe {
            arena.context().memcpy_h2d_async(
                prepared.raw_f252_words().as_void_ptr(),
                host.f252_values.as_ptr().cast::<c_void>(),
                input_h2d_bytes,
            )?;
        }
        arena.context().sync()?;
    }

    let output_pointers = prepared
        .big_limbs()
        .iter()
        .map(|column| column.as_u32_ptr())
        .collect::<Vec<_>>();
    let code = unsafe {
        stwo_backend_cuda_kernels::raw::memory_limb_split_big_columns_on(
            prepared.raw_f252_words().as_u32_ptr(),
            u32::try_from(planned.requirements.n_big)
                .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?,
            u32::try_from(planned.requirements.big_column_words)
                .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?,
            output_pointers.as_ptr(),
            arena.context().launch_context().stream_raw().as_ptr(),
        )
    };
    if code != 0 {
        return Err(CudaRuntimeError::Cuda {
            operation: "memory_limb_split_big_columns_on",
            code,
        }
        .into());
    }

    let (output_digest, validation_d2h_bytes) =
        validate_outputs(arena, &prepared, host.f252_values)?;
    Ok(FirstExecutionTableBigReceipt {
        operation: operation.id,
        wrapper_symbol: wrapper.wrapper_symbol().into(),
        real_rows: planned.requirements.n_big,
        column_rows: planned.requirements.big_column_words,
        input_h2d_bytes,
        validation_d2h_bytes,
        input_digest: input_digest(host.f252_values),
        output_digest,
    })
}

fn exact_first_operation<'a>(
    prefix: &'a CompiledWitnessWriterPrefix,
    plan: &ProofArenaPlan,
) -> Result<
    (
        execution_tables::LoweredExecutionTables,
        &'a crate::compiled_proof::OpNode,
        &'a crate::compiled_proof::StaticCudaWrapperAuthority,
    ),
    FirstExecutionTableBigError,
> {
    let mut values = adapter::SemanticValueMap::allocate_ordered(core::iter::empty())
        .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let lowered = execution_tables::lower_stage(plan, &mut values)
        .map_err(|_| FirstExecutionTableBigError::InvalidPrefixAuthority)?;
    let stage = &lowered.stages[0];
    let operation = prefix
        .operations
        .first()
        .filter(|operation| operation.id == OpId(0))
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
        .first()
        .filter(|stage| stage.stage() == ExecutionTablesStage::Big)
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
    let exact = stage.stage == ExecutionTablesStage::Big
        && operation.invocation.as_ref() == Some(&stage.invocation)
        && effect == &stage.effect
        && wrapper.wrapper_symbol() == WRAPPER_SYMBOL
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
    if !exact {
        return Err(FirstExecutionTableBigError::InvalidPrefixAuthority);
    }
    Ok((lowered, operation, wrapper))
}

fn validate_loaded_ranges(
    lowered: &execution_tables::LoweredExecutionTables,
    prepared: &PreparedExecutionTablesGraph<'_>,
) -> Result<(), FirstExecutionTableBigError> {
    let stage = &lowered.stages[0];
    let source = prepared.raw_f252_words();
    let outputs = prepared.big_limbs();
    let exact = source.id() == lowered.host_ingress[1].arena.physical
        && source.len_words() == lowered.host_ingress[1].arena.len_words
        && outputs.len() == stage.outputs.len()
        && outputs
            .iter()
            .zip(&stage.outputs)
            .all(|(actual, expected)| {
                actual.id() == expected.arena.physical
                    && actual.len_words() == expected.arena.len_words
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
    values: &[[u32; 8]],
) -> Result<([u8; 32], usize), FirstExecutionTableBigError> {
    let column_rows = prepared.requirements().big_column_words;
    let column_bytes = column_rows
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    let validation_d2h_bytes = column_bytes
        .checked_mul(EXECUTION_TABLE_BIG_LIMBS)
        .ok_or(FirstExecutionTableBigError::SizeOverflow)?;
    let mut column = vec![0u32; column_rows];
    let mut hasher = blake3::Hasher::new();
    hasher.update(OUTPUT_DIGEST_DOMAIN);
    hasher.update(
        &u64::try_from(values.len())
            .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(
        &u64::try_from(column_rows)
            .map_err(|_| FirstExecutionTableBigError::SizeOverflow)?
            .to_le_bytes(),
    );
    for (limb, source) in prepared.big_limbs().iter().copied().enumerate() {
        unsafe {
            arena.context().memcpy_d2h_async(
                column.as_mut_ptr().cast(),
                source.as_void_ptr().cast_const(),
                column_bytes,
            )?;
        }
        arena.context().sync()?;
        for (row, &actual) in column.iter().enumerate() {
            let expected = expected_limb(values, row, limb);
            if actual != expected {
                return Err(FirstExecutionTableBigError::OutputMismatch {
                    limb,
                    row,
                    expected,
                    actual,
                });
            }
        }
        hasher.update(bytemuck::cast_slice(&column));
    }
    Ok((*hasher.finalize().as_bytes(), validation_d2h_bytes))
}

fn expected_limb(values: &[[u32; 8]], row: usize, limb: usize) -> u32 {
    let Some(words) = values.get(row) else {
        return 0;
    };
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

fn input_digest(values: &[[u32; 8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INPUT_DIGEST_DOMAIN);
    hasher.update(bytemuck::cast_slice(values));
    *hasher.finalize().as_bytes()
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
            assert_eq!(expected_limb(&values, 0, limb), value);
        }
        assert_eq!(expected_limb(&values, 1, 0), 0);
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

    fn split_nine_bit(words: &[u32; 8]) -> [u32; EXECUTION_TABLE_BIG_LIMBS] {
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
