use super::*;

pub(crate) fn ready_arena(
    requirements: &CompositionWorkspaceRequirements,
    slots: &CompositionWorkspaceSlots,
    logs: &[u32],
    twiddles: &Twiddles,
    seed: u32,
    random: SecureField,
    z: SecureField,
    alpha_power: SecureField,
) -> ReadyArena {
    let arena = arena(requirements, slots, logs);
    for id in [
        slots.descriptors,
        slots.lde_tile,
        slots.accumulators,
        slots.random_coefficient_powers,
        RANDOM,
        FORWARD,
        INVERSE,
        Z,
        ALPHA,
    ] {
        fill(&arena, arena.bind(id).unwrap(), GUARD);
    }
    if let CompositionOutputSlots::CoefficientSplit(outputs) = slots.output {
        for id in outputs {
            fill(&arena, arena.bind(id).unwrap(), GUARD);
        }
    }
    for index in 0..logs.len() {
        fill(&arena, arena.bind(slot(EXT_BASE, index)).unwrap(), GUARD);
    }
    upload(&arena, arena.bind(FORWARD).unwrap(), &twiddles.forward);
    upload(&arena, arena.bind(INVERSE).unwrap(), &twiddles.inverse);
    let data = refresh(&arena, logs, seed, random, z, alpha_power);
    let direct = upload_data(&arena, &data, logs);
    ReadyArena {
        arena,
        data,
        direct,
    }
}

pub(crate) fn inputs(
    arena: &DeviceArena,
    requirements: &CompositionWorkspaceRequirements,
    component_count: usize,
) -> CompositionDeviceInputs {
    CompositionDeviceInputs {
        random_coefficient: RANDOM,
        forward_twiddles: arena
            .bind(FORWARD)
            .unwrap()
            .truncated(requirements.forward_twiddle_words),
        inverse_twiddles: arena
            .bind(INVERSE)
            .unwrap()
            .truncated(requirements.inverse_twiddle_words),
        relation_z: arena.bind(Z).unwrap().truncated(4),
        relation_alpha_powers: arena.bind(ALPHA).unwrap().truncated(4),
        claimed_sums: vec![None; component_count],
        ext_params: (0..component_count)
            .map(|index| {
                Some(CompositionExtParamBinding {
                    slot: slot(EXT_BASE, index),
                    offset_words: 0,
                })
            })
            .collect(),
    }
}

pub(crate) fn refresh(
    arena: &DeviceArena,
    logs: &[u32],
    seed: u32,
    random: SecureField,
    z: SecureField,
    alpha_power: SecureField,
) -> Data {
    upload(
        arena,
        arena.bind(RANDOM).unwrap(),
        &random.to_m31_array().map(|value| value.0),
    );
    upload(
        arena,
        arena.bind(Z).unwrap(),
        &z.to_m31_array().map(|value| value.0),
    );
    upload(
        arena,
        arena.bind(ALPHA).unwrap(),
        &alpha_power.to_m31_array().map(|value| value.0),
    );
    data(logs, seed)
}

pub(crate) fn assert_guards(
    arena: &DeviceArena,
    requirements: &CompositionWorkspaceRequirements,
    slots: &CompositionWorkspaceSlots,
    data: &Data,
    logs: &[u32],
    random: SecureField,
    z: SecureField,
    alpha_power: SecureField,
) {
    let mut logical = vec![
        (slots.descriptors, requirements.descriptor_words),
        (slots.lde_tile, requirements.lde_tile_words),
        (slots.accumulators, requirements.accumulator_words),
        (
            slots.random_coefficient_powers,
            requirements.random_power_words,
        ),
    ];
    if let CompositionOutputSlots::CoefficientSplit(outputs) = slots.output {
        logical.extend(outputs.map(|id| (id, requirements.output_coefficient_words)));
    }
    logical.extend([
        (RANDOM, 4),
        (FORWARD, requirements.forward_twiddle_words),
        (INVERSE, requirements.inverse_twiddle_words),
        (Z, 4),
        (ALPHA, 4),
    ]);
    logical.extend((0..logs.len()).map(|index| (slot(EXT_BASE, index), 8)));
    for (id, words) in logical {
        assert!(
            read(
                arena,
                arena
                    .bind(id)
                    .unwrap()
                    .checked_subslice(words, GUARD_WORDS)
                    .unwrap()
            )
            .iter()
            .all(|&word| word == GUARD),
            "tail canary {id:?}"
        );
    }
    assert_eq!(
        read(arena, arena.bind(RANDOM).unwrap().truncated(4)),
        random.to_m31_array().map(|value| value.0).to_vec()
    );
    assert_eq!(
        read(arena, arena.bind(Z).unwrap().truncated(4)),
        z.to_m31_array().map(|value| value.0).to_vec()
    );
    assert_eq!(
        read(arena, arena.bind(ALPHA).unwrap().truncated(4)),
        alpha_power.to_m31_array().map(|value| value.0).to_vec()
    );
    let ext_words = z
        .to_m31_array()
        .into_iter()
        .chain(alpha_power.to_m31_array())
        .map(|value| value.0)
        .collect::<Vec<_>>();
    for index in 0..logs.len() {
        assert_eq!(
            read(
                arena,
                arena.bind(slot(EXT_BASE, index)).unwrap().truncated(8)
            ),
            ext_words
        );
    }
    for (index, &log_size) in logs.iter().enumerate() {
        let coefficient_words = 1usize << log_size;
        let evaluation_words = 1usize << (log_size + 1);
        for (tree, (trace_base, direct_base)) in [
            (TRACE_BASE, DIRECT_BASE),
            (TRACE_INTERACTION, DIRECT_INTERACTION),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                read(
                    arena,
                    arena
                        .bind(slot(trace_base, index))
                        .unwrap()
                        .truncated(coefficient_words)
                ),
                data.coefficients[tree][index]
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                read(
                    arena,
                    arena
                        .bind(slot(direct_base, index))
                        .unwrap()
                        .truncated(evaluation_words)
                ),
                data.evaluations[tree][index]
                    .values
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>()
            );
            for (id, used) in [
                (slot(trace_base, index), coefficient_words),
                (slot(direct_base, index), evaluation_words),
            ] {
                assert!(read(
                    arena,
                    arena
                        .bind(id)
                        .unwrap()
                        .checked_subslice(used, GUARD_WORDS)
                        .unwrap()
                )
                .iter()
                .all(|&word| word == GUARD));
            }
        }
    }
}
