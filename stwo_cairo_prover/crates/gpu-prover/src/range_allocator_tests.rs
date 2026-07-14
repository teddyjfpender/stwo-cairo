use super::*;

fn request(id: u32, len_words: usize, live_mask: u16) -> RangeRequest {
    RangeRequest {
        id: RangeId(id),
        len_words,
        alignment_words: 1,
        live_mask,
        must_alias: None,
    }
}

#[test]
fn disjoint_small_ranges_share_one_historical_large_range() {
    let requests = [
        request(0, 16, 0b01),
        request(1, 8, 0b10),
        request(2, 8, 0b10),
    ];
    let layout = allocate_ranges(&requests, 1, None).unwrap();
    assert_eq!(layout.total_words, 16);
    assert_eq!(layout.raw_peak_words, 16);
    assert_eq!(layout.packing_overhead_words, 0);
    assert_eq!(layout.binding(RangeId(0)).unwrap().offset_words, 0);
    assert_eq!(layout.binding(RangeId(1)).unwrap().offset_words, 0);
    assert_eq!(layout.binding(RangeId(2)).unwrap().offset_words, 8);
}

#[test]
fn required_transition_aliases_share_exact_offsets() {
    let group = Some(AliasGroupId(7));
    let mut evaluation = request(0, 32, 0b0001);
    evaluation.must_alias = group;
    let mut coefficient = request(1, 32, 0b1110);
    coefficient.must_alias = group;
    let blocker = request(2, 16, 0b0010);
    let layout = allocate_ranges(&[evaluation, coefficient, blocker], 8, None).unwrap();
    assert_eq!(layout.binding(RangeId(0)).unwrap().offset_words, 0);
    assert_eq!(layout.binding(RangeId(1)).unwrap().offset_words, 0);
    assert_eq!(layout.binding(RangeId(2)).unwrap().offset_words, 32);
    assert_eq!(layout.total_words, 48);
}

#[test]
fn invalid_inputs_and_capacity_fail_closed() {
    assert_eq!(
        allocate_ranges(&[], 1, None).unwrap_err(),
        RangeAllocationError::EmptyPlan
    );
    assert!(matches!(
        allocate_ranges(&[request(0, 1, 1)], 0, None),
        Err(RangeAllocationError::InvalidAlignment {
            id: None,
            alignment: 0
        })
    ));
    let mut bad = request(0, 1, 1);
    bad.alignment_words = 3;
    assert!(matches!(
        allocate_ranges(&[bad], 1, None),
        Err(RangeAllocationError::InvalidAlignment {
            id: Some(RangeId(0)),
            alignment: 3
        })
    ));
    assert_eq!(
        allocate_ranges(&[request(0, 8, 1)], 1, Some(7)).unwrap_err(),
        RangeAllocationError::CapacityExceeded {
            required_words: 8,
            capacity_words: 7
        }
    );
    assert!(matches!(
        allocate_ranges(&[request(0, usize::MAX, 1)], 2, None),
        Err(RangeAllocationError::SizeOverflow)
    ));
}

#[test]
fn alias_contract_rejects_shape_or_lifetime_drift() {
    let group = Some(AliasGroupId(3));
    let mut first = request(0, 8, 1);
    first.must_alias = group;
    let mut second = request(1, 7, 2);
    second.must_alias = group;
    assert!(matches!(
        allocate_ranges(&[first, second], 1, None),
        Err(RangeAllocationError::AliasShapeMismatch {
            group: AliasGroupId(3),
            id: RangeId(1)
        })
    ));
    second.len_words = 8;
    second.live_mask = 1;
    assert!(matches!(
        allocate_ranges(&[first, second], 1, None),
        Err(RangeAllocationError::AliasLifetimeOverlap { .. })
    ));
}

#[test]
fn validator_rejects_a_live_overlap() {
    let requests = [request(0, 8, 0b11), request(1, 8, 0b10)];
    let bad = RangeLayout {
        total_words: 8,
        raw_peak_words: 16,
        packing_overhead_words: 0,
        bindings: vec![
            RangeBinding {
                id: RangeId(0),
                offset_words: 0,
                len_words: 8,
            },
            RangeBinding {
                id: RangeId(1),
                offset_words: 0,
                len_words: 8,
            },
        ],
    };
    assert!(matches!(
        validate_range_layout(&requests, 1, None, &bad),
        Err(RangeAllocationError::LiveRangeOverlap { .. })
    ));
}

#[test]
fn layout_is_stable_across_input_order() {
    let original = [
        request(0, 7, 0b0011),
        request(1, 11, 0b0110),
        request(2, 5, 0b1000),
        request(3, 3, 0b0100),
    ];
    let expected = allocate_ranges(&original, 4, None).unwrap();
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
        let permuted = order.map(|index| original[index]);
        assert_eq!(allocate_ranges(&permuted, 4, None).unwrap(), expected);
    }
}

#[test]
fn exhaustive_single_epoch_cases_hit_the_hard_lower_bound() {
    for case in 0..6561u32 {
        let mut digits = case;
        let mut requests = Vec::new();
        for id in 0..4 {
            let choice = digits % 9;
            digits /= 9;
            requests.push(request(id, (choice / 3 + 1) as usize, 1 << (choice % 3)));
        }
        let layout = allocate_ranges(&requests, 1, None).unwrap();
        assert_eq!(layout.total_words, layout.raw_peak_words, "case {case}");
    }
}

#[test]
fn exact_small_oracle_bounds_deterministic_variable_cases() {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut strict_gaps = 0;
    for case in 0..256 {
        let mut requests = Vec::new();
        for id in 0..4 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let first = (state >> 32) as u32 % 3;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let last = first + (state >> 32) as u32 % (3 - first);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut value = request(
                id,
                (state as usize % 3) + 1,
                ((1u16 << (last + 1)) - 1) ^ ((1u16 << first) - 1),
            );
            value.alignment_words = 1 << ((state >> 16) & 1);
            requests.push(value);
        }
        let layout = allocate_ranges(&requests, 2, None).unwrap();
        let exact = exact_minimum_words(&requests, 2, layout.total_words);
        assert!(layout.raw_peak_words <= exact, "case {case}");
        assert!(exact <= layout.total_words, "case {case}");
        strict_gaps += usize::from(exact < layout.total_words);
        validate_range_layout(&requests, 2, None, &layout).unwrap();
    }
    assert!(
        strict_gaps > 0,
        "the exact oracle must detect heuristic overhead"
    );
}

fn exact_minimum_words(
    requests: &[RangeRequest],
    slab_alignment_words: usize,
    upper_bound: usize,
) -> usize {
    fn search(
        request_index: usize,
        requests: &[RangeRequest],
        offsets: &mut Vec<usize>,
        slab_alignment_words: usize,
        lower_bound: usize,
        best: &mut usize,
    ) {
        if request_index == requests.len() {
            let end = requests
                .iter()
                .zip(offsets.iter())
                .map(|(request, offset)| offset + request.len_words)
                .max()
                .unwrap();
            *best = (*best).min(align_up(end, slab_alignment_words).unwrap());
            return;
        }
        let request = requests[request_index];
        for offset in
            (0..=*best - request.len_words).filter(|offset| offset % request.alignment_words == 0)
        {
            let end = offset + request.len_words;
            if align_up(end, slab_alignment_words).unwrap() > *best {
                continue;
            }
            let legal = requests[..request_index].iter().zip(offsets.iter()).all(
                |(other, other_offset)| {
                    !live_masks_conflict(request.live_mask, other.live_mask)
                        || end <= *other_offset
                        || *other_offset + other.len_words <= offset
                },
            );
            if legal {
                offsets.push(offset);
                search(
                    request_index + 1,
                    requests,
                    offsets,
                    slab_alignment_words,
                    lower_bound,
                    best,
                );
                offsets.pop();
                if *best == lower_bound {
                    return;
                }
            }
        }
    }

    let lower_bound = align_up(raw_peak_words(requests).unwrap(), slab_alignment_words).unwrap();
    let mut best = upper_bound;
    search(
        0,
        requests,
        &mut Vec::new(),
        slab_alignment_words,
        lower_bound,
        &mut best,
    );
    best
}
