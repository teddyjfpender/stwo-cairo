//! Diagnostic receipt for the exact resident quotient-numerator output.
//!
//! This is deliberately outside the production replay path: it copies the
//! numerator destinations to the host, synchronizes once, and hashes them so a
//! vertical hardware run can prove which exact output the prepared CUDA graph
//! produced before quotient evaluation consumes it.

use core::ffi::c_void;

use stwo_backend_cuda::{DeviceArena, PreparedNumeratorSchedule, PreparedQuotientNumeratorGraph};

use super::ResidentOodsError;
use crate::arena_plan::QuotientNumeratorSchedule;
use crate::resident_session::ResidentNumeratorRunSumTelemetry;

const COORDINATE_COUNT: usize = 4;
const WORD_BYTES: usize = core::mem::size_of::<u32>();
const SHAPE_DOMAIN: &[u8] = b"stwo.cairo.resident.quotient-numerator.shape.v1";
const ADAPTIVE_SHAPE_DOMAIN: &[u8] = b"stwo.cairo.resident.quotient-numerator.shape.adaptive.v2";
const OUTPUT_DOMAIN: &[u8] = b"stwo.cairo.resident.quotient-numerator.output.v1";

/// Compact evidence from one real resident quotient-numerator launch.
///
/// `output_digest` covers every destination word in coordinate-major, then
/// canonical group order. The receipt never includes arena addresses or slot
/// ids, so equal logical work has equal evidence across workspace instances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentQuotientNumeratorReceipt {
    pub planned_schedule: QuotientNumeratorSchedule,
    pub schedule: PreparedNumeratorSchedule,
    pub run_sum_identity: Option<[u8; 32]>,
    pub run_sum_target_group: Option<usize>,
    pub run_sum_victim_group: Option<usize>,
    pub run_sum_run_count: Option<u32>,
    pub run_sum_scratch_words_per_coordinate: Option<usize>,
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
    planned_schedule: QuotientNumeratorSchedule,
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
    let run_sum = super::numerator_run_sum_telemetry(numerator);
    let shape_digest = digest_shape(planned_schedule, schedule, run_sum, requirements)?;
    let mut output_hasher = blake3::Hasher::new();
    output_hasher.update(OUTPUT_DOMAIN);
    output_hasher.update(&shape_digest);
    output_hasher.update(bytemuck::cast_slice(&host_words));

    Ok(ResidentQuotientNumeratorReceipt {
        planned_schedule,
        schedule,
        run_sum_identity: run_sum.map(|receipt| receipt.identity),
        run_sum_target_group: run_sum.map(|receipt| receipt.target_group),
        run_sum_victim_group: run_sum.map(|receipt| receipt.victim_group),
        run_sum_run_count: run_sum.map(|receipt| receipt.run_count),
        run_sum_scratch_words_per_coordinate: run_sum
            .map(|receipt| receipt.scratch_words_per_coordinate),
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
    planned_schedule: QuotientNumeratorSchedule,
    schedule: PreparedNumeratorSchedule,
    run_sum: Option<ResidentNumeratorRunSumTelemetry>,
    requirements: &stwo_backend_cuda::QuotientNumeratorWorkspaceRequirements,
) -> Result<[u8; 32], ResidentOodsError> {
    let mut hasher = blake3::Hasher::new();
    if planned_schedule == QuotientNumeratorSchedule::StagedRunSumOrPacked {
        hasher.update(ADAPTIVE_SHAPE_DOMAIN);
        hasher.update(&[QuotientNumeratorSchedule::StagedRunSumOrPacked as u8]);
        match (schedule, run_sum) {
            (PreparedNumeratorSchedule::StagedGroupDirect { output_rows }, Some(receipt))
                if receipt.is_complete() =>
            {
                hasher.update(&[5]);
                hasher.update(&output_rows.to_le_bytes());
                hasher.update(&[1]);
                hasher.update(&receipt.identity);
            }
            (PreparedNumeratorSchedule::StagedPackedSingleWrite { packed_output_rows }, None) => {
                hasher.update(&[2]);
                hasher.update(&packed_output_rows.to_le_bytes());
                hasher.update(&[0]);
            }
            _ => {
                return Err(ResidentOodsError::StagedNumeratorBinding(
                    "adaptive receipt schedule and run-sum identity disagree",
                ))
            }
        }
    } else {
        hasher.update(SHAPE_DOMAIN);
        if planned_schedule == QuotientNumeratorSchedule::StagedGroupDirect {
            if !matches!(
                schedule,
                PreparedNumeratorSchedule::StagedGroupDirect { .. }
            ) || run_sum.is_some_and(|receipt| !receipt.is_complete())
            {
                return Err(ResidentOodsError::StagedNumeratorBinding(
                    "explicit group-direct receipt is malformed",
                ));
            }
        } else if run_sum.is_some() {
            return Err(ResidentOodsError::StagedNumeratorBinding(
                "non-adaptive receipt unexpectedly owns a run-sum identity",
            ));
        }
        update_v1_schedule(&mut hasher, schedule)?;
    }
    update_requirements(&mut hasher, requirements)?;
    Ok(*hasher.finalize().as_bytes())
}

fn update_v1_schedule(
    hasher: &mut blake3::Hasher,
    schedule: PreparedNumeratorSchedule,
) -> Result<(), ResidentOodsError> {
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
            update_usize(hasher, eligible_groups)?;
            update_usize(hasher, legacy_groups)?;
        }
        PreparedNumeratorSchedule::StagedPrepackedSingleWrite { packed_output_rows } => {
            // Receipt tags are append-only: changing 0..=4 would invalidate
            // otherwise identical historical resident-output evidence.
            hasher.update(&[4]);
            hasher.update(&packed_output_rows.to_le_bytes());
        }
        PreparedNumeratorSchedule::StagedGroupDirect { output_rows } => {
            hasher.update(&[5]);
            hasher.update(&output_rows.to_le_bytes());
        }
    }
    Ok(())
}

fn update_requirements(
    hasher: &mut blake3::Hasher,
    requirements: &stwo_backend_cuda::QuotientNumeratorWorkspaceRequirements,
) -> Result<(), ResidentOodsError> {
    update_usize(hasher, requirements.groups.len())?;
    update_usize(hasher, requirements.batches.len())?;
    update_usize(hasher, requirements.input_sample_count)?;
    update_usize(hasher, requirements.term_count)?;
    for group in &requirements.groups {
        hasher.update(&group.log_size.to_le_bytes());
        update_usize(hasher, group.value_words)?;
        update_usize(hasher, group.coefficient_source_count)?;
    }
    for batch in &requirements.batches {
        hasher.update(&batch.evaluation_log_size.to_le_bytes());
        update_usize(hasher, batch.source_count)?;
        update_usize(hasher, batch.coefficient_count)?;
        update_usize(hasher, batch.term_count)?;
        update_usize(hasher, batch.lde_words)?;
    }
    Ok(())
}

fn update_usize(hasher: &mut blake3::Hasher, value: usize) -> Result<(), ResidentOodsError> {
    let value = u64::try_from(value).map_err(|_| ResidentOodsError::SizeOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use stwo::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use stwo_backend_cuda::{
        PreparedNumeratorSchedule, QuotientNumeratorBatchRequirements,
        QuotientNumeratorGroupRequirements, QuotientNumeratorWorkspaceConfig,
        QuotientNumeratorWorkspaceRequirements,
    };

    use super::*;

    fn deterministic_requirements() -> QuotientNumeratorWorkspaceRequirements {
        QuotientNumeratorWorkspaceRequirements {
            config: QuotientNumeratorWorkspaceConfig {
                lifting_log_size: 6,
                log_blowup_factor: 2,
                max_lde_tile_words: 256,
            },
            input_sample_count: 5,
            term_count: 7,
            groups: vec![QuotientNumeratorGroupRequirements {
                shape_point: SECURE_FIELD_CIRCLE_GEN,
                log_size: 17,
                value_words: 0x0102,
                coefficient_source_count: 0x0304,
            }],
            batches: vec![QuotientNumeratorBatchRequirements {
                evaluation_log_size: 19,
                source_count: 0x0506,
                coefficient_count: 0x0708,
                term_count: 0x090a,
                lde_words: 0x0b0c,
            }],
            runtime_term_words: 0,
            group_term_index_words: 0,
            group_offset_words: 0,
            line_coefficient_words: 0,
            term_point_words: 0,
            batch_term_words: 0,
            batch_group_offset_words: 0,
            batch_source_pointer_words: 0,
            coefficient_pointer_words: 0,
            coefficient_size_words: 0,
            coefficient_output_pointer_words: 0,
            output_pointer_words: 0,
            output_log_size_words: 0,
            lde_tile_words: 0,
            forward_twiddle_words: 0,
            max_output_size: 0,
        }
    }

    fn frozen_v1_digest(
        schedule: PreparedNumeratorSchedule,
        requirements: &QuotientNumeratorWorkspaceRequirements,
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"stwo.cairo.resident.quotient-numerator.shape.v1");
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
                frozen_usize(&mut hasher, eligible_groups);
                frozen_usize(&mut hasher, legacy_groups);
            }
            PreparedNumeratorSchedule::StagedPrepackedSingleWrite { packed_output_rows } => {
                hasher.update(&[4]);
                hasher.update(&packed_output_rows.to_le_bytes());
            }
            PreparedNumeratorSchedule::StagedGroupDirect { output_rows } => {
                hasher.update(&[5]);
                hasher.update(&output_rows.to_le_bytes());
            }
        }
        frozen_usize(&mut hasher, requirements.groups.len());
        frozen_usize(&mut hasher, requirements.batches.len());
        frozen_usize(&mut hasher, requirements.input_sample_count);
        frozen_usize(&mut hasher, requirements.term_count);
        for group in &requirements.groups {
            hasher.update(&group.log_size.to_le_bytes());
            frozen_usize(&mut hasher, group.value_words);
            frozen_usize(&mut hasher, group.coefficient_source_count);
        }
        for batch in &requirements.batches {
            hasher.update(&batch.evaluation_log_size.to_le_bytes());
            frozen_usize(&mut hasher, batch.source_count);
            frozen_usize(&mut hasher, batch.coefficient_count);
            frozen_usize(&mut hasher, batch.term_count);
            frozen_usize(&mut hasher, batch.lde_words);
        }
        *hasher.finalize().as_bytes()
    }

    fn frozen_usize(hasher: &mut blake3::Hasher, value: usize) {
        hasher.update(&u64::try_from(value).unwrap().to_le_bytes());
    }

    #[test]
    fn legacy_v1_digest_matches_the_frozen_encoding() {
        let requirements = deterministic_requirements();
        for schedule in [
            PreparedNumeratorSchedule::LegacyBatches,
            PreparedNumeratorSchedule::SingleWriteCandidate,
            PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 0x0102_0304_0506_0708,
            },
            PreparedNumeratorSchedule::HybridCandidate {
                eligible_groups: 0x0102,
                legacy_groups: 0x0304,
            },
            PreparedNumeratorSchedule::StagedPrepackedSingleWrite {
                packed_output_rows: 0x1112_1314_1516_1718,
            },
            PreparedNumeratorSchedule::StagedGroupDirect {
                output_rows: 0x2122_2324_2526_2728,
            },
        ] {
            assert_eq!(
                digest_shape(
                    QuotientNumeratorSchedule::LegacyBatches,
                    schedule,
                    None,
                    &requirements,
                )
                .unwrap(),
                frozen_v1_digest(schedule, &requirements),
                "{schedule:?}"
            );
        }
    }

    #[test]
    fn adaptive_digest_binds_actual_schedule_and_complete_identity() {
        let requirements = deterministic_requirements();
        let receipt = ResidentNumeratorRunSumTelemetry {
            identity: [0x5a; 32],
            target_group: 0,
            victim_group: 12,
            run_count: 17,
            scratch_words_per_coordinate: 8_388_048,
        };
        let direct = digest_shape(
            QuotientNumeratorSchedule::StagedRunSumOrPacked,
            PreparedNumeratorSchedule::StagedGroupDirect { output_rows: 64 },
            Some(receipt),
            &requirements,
        )
        .unwrap();
        let packed = digest_shape(
            QuotientNumeratorSchedule::StagedRunSumOrPacked,
            PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 64,
            },
            None,
            &requirements,
        )
        .unwrap();
        let legacy_packed = digest_shape(
            QuotientNumeratorSchedule::StagedPackedSingleWrite,
            PreparedNumeratorSchedule::StagedPackedSingleWrite {
                packed_output_rows: 64,
            },
            None,
            &requirements,
        )
        .unwrap();
        let changed_identity = digest_shape(
            QuotientNumeratorSchedule::StagedRunSumOrPacked,
            PreparedNumeratorSchedule::StagedGroupDirect { output_rows: 64 },
            Some(ResidentNumeratorRunSumTelemetry {
                identity: [0xa5; 32],
                ..receipt
            }),
            &requirements,
        )
        .unwrap();
        assert_ne!(direct, packed);
        assert_ne!(packed, legacy_packed);
        assert_ne!(direct, changed_identity);

        assert!(digest_shape(
            QuotientNumeratorSchedule::StagedRunSumOrPacked,
            PreparedNumeratorSchedule::StagedGroupDirect { output_rows: 64 },
            Some(ResidentNumeratorRunSumTelemetry {
                target_group: 12,
                victim_group: 0,
                ..receipt
            }),
            &requirements,
        )
        .is_err());
        assert!(digest_shape(
            QuotientNumeratorSchedule::StagedRunSumOrPacked,
            PreparedNumeratorSchedule::StagedGroupDirect { output_rows: 64 },
            None,
            &requirements,
        )
        .is_err());
    }

    #[test]
    fn explicit_group_direct_keeps_its_v1_digest() {
        let requirements = deterministic_requirements();
        let schedule = PreparedNumeratorSchedule::StagedGroupDirect { output_rows: 64 };
        let expected = frozen_v1_digest(schedule, &requirements);
        assert_eq!(
            digest_shape(
                QuotientNumeratorSchedule::StagedGroupDirect,
                schedule,
                None,
                &requirements,
            )
            .unwrap(),
            expected,
        );
        assert_eq!(
            digest_shape(
                QuotientNumeratorSchedule::StagedGroupDirect,
                schedule,
                Some(ResidentNumeratorRunSumTelemetry {
                    identity: [0x5a; 32],
                    target_group: 0,
                    victim_group: 12,
                    run_count: 17,
                    scratch_words_per_coordinate: 8_388_048,
                }),
                &requirements,
            )
            .unwrap(),
            expected,
        );
    }
}
