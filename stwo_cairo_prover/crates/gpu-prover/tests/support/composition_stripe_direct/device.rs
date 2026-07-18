use core::mem::size_of_val;
use std::collections::BTreeSet;

use stwo_backend_cuda::{ArenaLayout, ArenaSlotSpec, CudaExecContext};
use stwo_cairo_gpu_prover::prepared_composition::composition_workspace_requirements_with_retention_for_test;
use stwo_cairo_gpu_prover::CompositionExtParamBinding;

use super::*;

fn add_slot(
    specs: &mut Vec<ArenaSlotSpec>,
    ids: &mut BTreeSet<ArenaSlotId>,
    offset: &mut usize,
    id: ArenaSlotId,
    len_words: usize,
    alignment_words: usize,
) {
    assert!(ids.insert(id), "duplicate arena slot {id:?}");
    *offset = offset.next_multiple_of(alignment_words);
    specs.push(ArenaSlotSpec {
        id,
        offset_words: *offset,
        len_words: len_words.max(1),
        alignment_words,
    });
    *offset += len_words.max(1);
}

fn arena(
    fixture: &Fixture,
    wave_requirements: &CompositionWorkspaceRequirements,
    stripe_requirements: &CompositionWorkspaceRequirements,
) -> DeviceArena {
    let mut specs = Vec::new();
    let mut ids = BTreeSet::new();
    let mut offset = 0usize;
    for (requirements, slots) in [
        (wave_requirements, &fixture.wave_slots),
        (stripe_requirements, &fixture.stripe_slots),
    ] {
        for requirement in requirements.arena_slot_requirements(slots).unwrap() {
            add_slot(
                &mut specs,
                &mut ids,
                &mut offset,
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            );
        }
    }
    for pointer_slots in [fixture.wave_pointer_slots, fixture.stripe_pointer_slots] {
        for requirement in fixture
            .split_program
            .arena_slot_requirements(pointer_slots)
            .unwrap()
        {
            add_slot(
                &mut specs,
                &mut ids,
                &mut offset,
                requirement.id,
                requirement.len_words,
                requirement.alignment_words,
            );
        }
    }
    for tree in &fixture.trace.trees {
        for source in tree {
            add_slot(
                &mut specs,
                &mut ids,
                &mut offset,
                source.slot,
                1usize << source.log_size,
                1,
            );
        }
    }
    for (index, column) in fixture.retention.columns.iter().enumerate() {
        add_slot(
            &mut specs,
            &mut ids,
            &mut offset,
            slot(DIRECT_BASE, index),
            1usize << column.evaluation_log_size,
            1,
        );
    }
    let rows = 1usize << MAX_EVALUATION_LOG;
    for base in [WAVE_RETAINED_BASE, STRIPE_RETAINED_BASE] {
        for index in 0..COMPOSITION_RETAINED_COLUMNS {
            add_slot(
                &mut specs,
                &mut ids,
                &mut offset,
                slot(base, index),
                rows,
                2,
            );
        }
    }
    for (index, component) in fixture.plan.components.iter().enumerate() {
        add_slot(
            &mut specs,
            &mut ids,
            &mut offset,
            slot(EXT_BASE, index),
            component.ext_param_values.len() * 4,
            4,
        );
    }
    for (id, len, alignment) in [
        (RANDOM, 4, 4),
        (FORWARD, wave_requirements.forward_twiddle_words, 1),
        (INVERSE, wave_requirements.inverse_twiddle_words, 1),
        (RELATION_Z, 4, 4),
        (RELATION_ALPHA, 4, 4),
    ] {
        add_slot(&mut specs, &mut ids, &mut offset, id, len, alignment);
    }
    DeviceArena::new(
        CudaExecContext::new().unwrap(),
        ArenaLayout::new(offset, &specs).unwrap(),
    )
    .unwrap()
}

fn upload(arena: &DeviceArena, slice: ArenaSlice, words: &[u32]) {
    assert_eq!(slice.len_words(), words.len());
    unsafe {
        arena
            .context()
            .memcpy_h2d_async(
                slice.as_void_ptr(),
                words.as_ptr().cast(),
                size_of_val(words),
            )
            .unwrap();
    }
    arena.context().sync().unwrap();
}

fn fill(arena: &DeviceArena, slice: ArenaSlice, word: u32) {
    unsafe {
        arena
            .context()
            .fill_u32_async(slice.as_u32_ptr(), word, slice.len_words())
            .unwrap();
    }
}

pub(crate) fn ready(fixture: &Fixture, twiddles: &Twiddles, seed: u32) -> Ready {
    let wave_requirements = composition_workspace_requirements_with_retention_for_test(
        &fixture.plan,
        &fixture.trace,
        CompositionLaunchMode::Wave,
        Some(&fixture.retention),
    )
    .unwrap();
    let stripe_requirements = composition_workspace_requirements_with_retention_for_test(
        &fixture.plan,
        &fixture.trace,
        CompositionLaunchMode::Serial,
        Some(&fixture.retention),
    )
    .unwrap();
    assert_eq!(
        wave_requirements.forward_twiddle_words,
        stripe_requirements.forward_twiddle_words
    );
    assert_eq!(
        wave_requirements
            .components
            .iter()
            .map(|component| component.evaluation_log_size)
            .collect::<Vec<_>>(),
        EVALUATION_LOGS
    );
    assert!([&wave_requirements, &stripe_requirements]
        .into_iter()
        .flat_map(|requirements| &requirements.components)
        .all(|component| component.fallback_count == 0));
    let arena = arena(fixture, &wave_requirements, &stripe_requirements);
    upload(
        &arena,
        arena
            .bind(FORWARD)
            .unwrap()
            .truncated(wave_requirements.forward_twiddle_words),
        &twiddles.forward,
    );
    upload(
        &arena,
        arena
            .bind(INVERSE)
            .unwrap()
            .truncated(wave_requirements.inverse_twiddle_words),
        &twiddles.inverse,
    );
    upload(
        &arena,
        arena.bind(RELATION_Z).unwrap().truncated(4),
        &[0; 4],
    );
    upload(
        &arena,
        arena.bind(RELATION_ALPHA).unwrap().truncated(4),
        &[0; 4],
    );
    let direct = fixture
        .retention
        .columns
        .iter()
        .enumerate()
        .map(|(plan_column, column)| CompositionDirectEvaluationBinding {
            plan_column,
            evaluation: arena
                .bind(slot(DIRECT_BASE, plan_column))
                .unwrap()
                .truncated(1usize << column.evaluation_log_size),
        })
        .collect::<Vec<_>>();
    let wave_retained = std::array::from_fn(|index| {
        arena
            .bind(slot(WAVE_RETAINED_BASE, index))
            .unwrap()
            .truncated(1usize << MAX_EVALUATION_LOG)
    });
    let stripe_retained = std::array::from_fn(|index| {
        arena
            .bind(slot(STRIPE_RETAINED_BASE, index))
            .unwrap()
            .truncated(1usize << MAX_EVALUATION_LOG)
    });
    let inputs = CompositionDeviceInputs {
        random_coefficient: RANDOM,
        forward_twiddles: arena
            .bind(FORWARD)
            .unwrap()
            .truncated(wave_requirements.forward_twiddle_words),
        inverse_twiddles: arena
            .bind(INVERSE)
            .unwrap()
            .truncated(wave_requirements.inverse_twiddle_words),
        relation_z: arena.bind(RELATION_Z).unwrap().truncated(4),
        relation_alpha_powers: arena.bind(RELATION_ALPHA).unwrap().truncated(4),
        claimed_sums: vec![None; fixture.plan.components.len()],
        ext_params: (0..fixture.plan.components.len())
            .map(|index| {
                Some(CompositionExtParamBinding {
                    slot: slot(EXT_BASE, index),
                    offset_words: 0,
                })
            })
            .collect(),
    };
    let ready = Ready {
        wave_split: CompositionDirectSplitBinding {
            program: fixture.split_program,
            pointer_slots: fixture.wave_pointer_slots,
            retained_evaluations: wave_retained,
        },
        stripe_split: CompositionDirectSplitBinding {
            program: fixture.split_program,
            pointer_slots: fixture.stripe_pointer_slots,
            retained_evaluations: stripe_retained,
        },
        arena,
        inputs,
        direct,
        wave_retained,
        stripe_retained,
    };
    refresh(&ready, seed);
    ready
}

fn secure(seed: u32) -> SecureField {
    SecureField::from_u32_unchecked(seed + 2, seed + 3, seed + 5, seed + 7)
}

pub(crate) fn refresh(ready: &Ready, seed: u32) {
    upload(
        &ready.arena,
        ready.arena.bind(RANDOM).unwrap().truncated(4),
        &secure(seed).to_m31_array().map(|value| value.0),
    );
    for binding in &ready.direct {
        let word = seed
            .wrapping_mul(1_009)
            .wrapping_add((binding.plan_column as u32 + 1).wrapping_mul(65_537))
            % MODULUS;
        fill(&ready.arena, binding.evaluation, word);
    }
    ready.arena.context().sync().unwrap();
}
