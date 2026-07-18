//! Diagnostic receipt for the exact resident quotient-numerator output.
//!
//! This is deliberately outside the production replay path: it copies the
//! numerator destinations to the host, synchronizes once, and hashes them so a
//! vertical hardware run can prove which exact output the prepared CUDA graph
//! produced before quotient evaluation consumes it.

use core::ffi::c_void;

use stwo_backend_cuda::{DeviceArena, PreparedNumeratorSchedule, PreparedQuotientNumeratorGraph};

use super::ResidentOodsError;

const COORDINATE_COUNT: usize = 4;
const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SHAPE_DOMAIN: &[u8] = b"stwo.cairo.resident.quotient-numerator.shape.v1";
const OUTPUT_DOMAIN: &[u8] = b"stwo.cairo.resident.quotient-numerator.output.v1";

/// Compact evidence from one real resident quotient-numerator launch.
///
/// `output_digest` covers every destination word in coordinate-major, then
/// canonical group order. The receipt never includes arena addresses or slot
/// ids, so equal logical work has equal evidence across workspace instances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentQuotientNumeratorReceipt {
    pub schedule: PreparedNumeratorSchedule,
    pub group_count: usize,
    pub batch_count: usize,
    pub term_count: usize,
    pub output_rows: usize,
    pub output_words: usize,
    pub validation_d2h_bytes: usize,
    pub shape_digest: [u8; 32],
    pub output_digest: [u8; 32],
}

pub(super) fn read_numerator_receipt(
    arena: &DeviceArena,
    numerator: &PreparedQuotientNumeratorGraph<'_>,
) -> Result<ResidentQuotientNumeratorReceipt, ResidentOodsError> {
    let requirements = numerator.requirements();
    let destinations = numerator.destinations();
    if destinations.len() != requirements.groups.len() {
        return Err(ResidentOodsError::NumeratorDestinationCount {
            expected: requirements.groups.len(),
            actual: destinations.len(),
        });
    }

    let mut output_rows = 0usize;
    for (group_index, (group, destination)) in
        requirements.groups.iter().zip(destinations).enumerate()
    {
        if destination.log_size != group.log_size {
            return Err(ResidentOodsError::NumeratorDestinationLogSize {
                group: group_index,
                expected: group.log_size,
                actual: destination.log_size,
            });
        }
        for (coordinate, destination) in destination.coordinates.iter().enumerate() {
            if destination.len_words() != group.value_words {
                return Err(ResidentOodsError::NumeratorDestinationWords {
                    group: group_index,
                    coordinate,
                    expected: group.value_words,
                    actual: destination.len_words(),
                });
            }
        }
        output_rows = output_rows
            .checked_add(group.value_words)
            .ok_or(ResidentOodsError::SizeOverflow)?;
    }

    let output_words = output_rows
        .checked_mul(COORDINATE_COUNT)
        .ok_or(ResidentOodsError::SizeOverflow)?;
    let validation_d2h_bytes = output_words
        .checked_mul(WORD_BYTES)
        .ok_or(ResidentOodsError::SizeOverflow)?;
    let mut host_words = vec![0u32; output_words];

    let mut host_offset = 0usize;
    for coordinate in 0..COORDINATE_COUNT {
        for destination in destinations {
            let source = destination.coordinates[coordinate];
            let next_offset = host_offset
                .checked_add(source.len_words())
                .ok_or(ResidentOodsError::SizeOverflow)?;
            unsafe {
                arena.context().memcpy_d2h_async(
                    host_words.as_mut_ptr().add(host_offset).cast::<c_void>(),
                    source.as_void_ptr().cast_const(),
                    source.len_bytes(),
                )?;
            }
            host_offset = next_offset;
        }
    }
    debug_assert_eq!(host_offset, output_words);
    arena.context().sync()?;

    let schedule = numerator.schedule();
    let shape_digest = digest_shape(schedule, requirements)?;
    let mut output_hasher = blake3::Hasher::new();
    output_hasher.update(OUTPUT_DOMAIN);
    output_hasher.update(&shape_digest);
    output_hasher.update(bytemuck::cast_slice(&host_words));

    Ok(ResidentQuotientNumeratorReceipt {
        schedule,
        group_count: requirements.groups.len(),
        batch_count: requirements.batches.len(),
        term_count: requirements.term_count,
        output_rows,
        output_words,
        validation_d2h_bytes,
        shape_digest,
        output_digest: *output_hasher.finalize().as_bytes(),
    })
}

fn digest_shape(
    schedule: PreparedNumeratorSchedule,
    requirements: &stwo_backend_cuda::QuotientNumeratorWorkspaceRequirements,
) -> Result<[u8; 32], ResidentOodsError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SHAPE_DOMAIN);
    match schedule {
        PreparedNumeratorSchedule::LegacyBatches => {
            hasher.update(&[0]);
        }
        PreparedNumeratorSchedule::SingleWriteCandidate => {
            hasher.update(&[1]);
        }
        PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows } => {
            hasher.update(&[2]);
            hasher.update(&packed_output_rows.to_le_bytes());
        }
        PreparedNumeratorSchedule::HybridCandidate {
            eligible_groups,
            legacy_groups,
        } => {
            hasher.update(&[3]);
            update_usize(&mut hasher, eligible_groups)?;
            update_usize(&mut hasher, legacy_groups)?;
        }
        PreparedNumeratorSchedule::StagedPrepackedSingleWrite { packed_output_rows } => {
            // Receipt tags are append-only: changing 0..=3 would invalidate
            // otherwise identical historical resident-output evidence.
            hasher.update(&[4]);
            hasher.update(&packed_output_rows.to_le_bytes());
        }
    }
    update_usize(&mut hasher, requirements.groups.len())?;
    update_usize(&mut hasher, requirements.batches.len())?;
    update_usize(&mut hasher, requirements.input_sample_count)?;
    update_usize(&mut hasher, requirements.term_count)?;
    for group in &requirements.groups {
        hasher.update(&group.log_size.to_le_bytes());
        update_usize(&mut hasher, group.value_words)?;
        update_usize(&mut hasher, group.coefficient_source_count)?;
    }
    for batch in &requirements.batches {
        hasher.update(&batch.evaluation_log_size.to_le_bytes());
        update_usize(&mut hasher, batch.source_count)?;
        update_usize(&mut hasher, batch.coefficient_count)?;
        update_usize(&mut hasher, batch.term_count)?;
        update_usize(&mut hasher, batch.lde_words)?;
    }
    Ok(*hasher.finalize().as_bytes())
}

fn update_usize(hasher: &mut blake3::Hasher, value: usize) -> Result<(), ResidentOodsError> {
    let value = u64::try_from(value).map_err(|_| ResidentOodsError::SizeOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}
