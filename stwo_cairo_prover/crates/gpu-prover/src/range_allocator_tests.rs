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
    assert_eq!(layout.total_words(), 16);
    assert_eq!(layout.raw_peak_words(), 16);
    assert_eq!(layout.excess_over_raw_peak_words(), 0);
    assert_eq!(layout.bindings().len(), requests.len());
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
    assert_eq!(layout.total_words(), 48);
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
        excess_over_raw_peak_words: 0,
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
fn validator_rejects_every_mutable_layout_contract() {
    let group = Some(AliasGroupId(9));
    let mut first = request(0, 8, 0b01);
    first.alignment_words = 4;
    first.must_alias = group;
    let mut second = request(1, 8, 0b10);
    second.alignment_words = 4;
    second.must_alias = group;
    let mut blocker = request(2, 8, 0b11);
    blocker.alignment_words = 4;
    let requests = [first, second, blocker];
    let valid = allocate_ranges(&requests, 4, None).unwrap();

    let mut bad = valid.clone();
    bad.bindings.reverse();
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::NonCanonicalBindingOrder)
    );

    let mut bad = valid.clone();
    bad.bindings[1].offset_words = 8;
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::AliasOffsetMismatch {
            group: AliasGroupId(9),
            id: RangeId(1),
        })
    );

    let mut bad = valid.clone();
    bad.bindings.pop();
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::MissingBinding(RangeId(2)))
    );

    let mut bad = valid.clone();
    bad.bindings.push(RangeBinding {
        id: RangeId(99),
        offset_words: 0,
        len_words: 8,
    });
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::UnexpectedBinding(RangeId(99)))
    );

    let mut bad = valid.clone();
    bad.bindings.push(bad.bindings[2]);
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::DuplicateBinding(RangeId(2)))
    );

    let mut bad = valid.clone();
    bad.bindings[0].len_words = 7;
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::BindingLengthMismatch {
            id: RangeId(0),
            expected: 8,
            actual: 7,
        })
    );

    let mut bad = valid.clone();
    bad.bindings[2].offset_words += 1;
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::MisalignedBinding(RangeId(2)))
    );

    let mut bad = valid.clone();
    bad.bindings[2].offset_words = valid.total_words();
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::OutOfBounds(RangeId(2)))
    );

    let mut bad = valid.clone();
    bad.raw_peak_words += 1;
    assert_eq!(
        validate_range_layout(&requests, 4, None, &bad),
        Err(RangeAllocationError::LayoutMetadataMismatch)
    );

    assert_eq!(
        validate_range_layout(&requests, 4, Some(valid.total_words() - 1), &valid),
        Err(RangeAllocationError::CapacityExceeded {
            required_words: valid.total_words(),
            capacity_words: valid.total_words() - 1,
        })
    );
}

#[test]
fn excess_over_raw_peak_includes_required_alignment() {
    let layout = allocate_ranges(&[request(0, 1, 1)], 32, None).unwrap();
    assert_eq!(layout.raw_peak_words(), 1);
    assert_eq!(layout.total_words(), 32);
    assert_eq!(layout.excess_over_raw_peak_words(), 31);
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
fn monotonic_search_is_byte_identical_to_uncached_first_fit() {
    let masks = [
        0b0000_0000_0001,
        0b0000_0000_0011,
        0b0000_0000_0110,
        0b0000_0000_1110,
        0b0000_0001_1100,
        0b0000_0011_1000,
        0b0000_0111_0000,
        0b0000_1110_0000,
    ];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut requests = Vec::new();
    for id in 0..768u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut value = request(
            id,
            [32, 64, 96, 128][(state as usize >> 8) & 3],
            masks[(state as usize >> 16) & 7],
        );
        value.alignment_words = [1, 2, 4, 32][(state as usize >> 24) & 3];
        requests.push(value);
    }
    for group in 1..=12u32 {
        let mut evaluation = request(768 + (group - 1) * 2, 96, 0b0000_0000_0011);
        evaluation.alignment_words = 32;
        evaluation.must_alias = Some(AliasGroupId(group));
        let mut coefficients = request(769 + (group - 1) * 2, 96, 0b0000_0000_1100);
        coefficients.alignment_words = 32;
        coefficients.must_alias = Some(AliasGroupId(group));
        requests.extend([evaluation, coefficients]);
    }

    let expected = allocate_ranges_uncached_reference(&requests, 32, None).unwrap();
    let actual = allocate_ranges(&requests, 32, None).unwrap();
    assert_eq!(actual, expected);

    requests.reverse();
    assert_eq!(allocate_ranges(&requests, 32, None).unwrap(), expected);
}

#[test]
fn repeated_shape_20k_scale_regression() {
    let mut requests = (0..20_500u32)
        .map(|id| {
            let mut value = request(id, 128, 0b0000_0000_1111);
            value.alignment_words = 32;
            value
        })
        .collect::<Vec<_>>();
    // Interleave a second repeated shape so the cursor proof is exercised
    // across intervening placements, not only one contiguous sort run.
    for request in requests.iter_mut().step_by(11) {
        request.live_mask = 0b0000_1111_0000;
    }

    let started = std::time::Instant::now();
    let layout = allocate_ranges(&requests, 32, None).unwrap();
    eprintln!(
        "range_allocator_scale ranges={} elapsed_ms={}",
        requests.len(),
        started.elapsed().as_millis()
    );
    assert_eq!(layout.bindings().len(), requests.len());
    validate_range_layout(&requests, 32, None, &layout).unwrap();
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
        assert_eq!(layout.total_words(), layout.raw_peak_words(), "case {case}");
    }
}

#[test]
fn independent_exact_oracle_bounds_deterministic_variable_cases() {
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
        let exact = exact_minimum_words(&requests, 2);
        assert!(
            independent_raw_peak_words(&requests) <= exact,
            "case {case}"
        );
        assert!(exact <= layout.total_words(), "case {case}");
        strict_gaps += usize::from(exact < layout.total_words());
        validate_range_layout(&requests, 2, None, &layout).unwrap();
    }
    assert!(
        strict_gaps > 0,
        "the exact oracle must detect heuristic overhead"
    );
}

#[test]
fn independent_exact_oracle_enforces_alias_groups() {
    let group = Some(AliasGroupId(17));
    let mut first = request(0, 1, 0b001);
    first.must_alias = group;
    let left_blocker = request(1, 1, 0b101);
    let mut second = request(2, 1, 0b010);
    second.must_alias = group;
    let right_blocker = request(3, 1, 0b110);
    let constrained = [first, left_blocker, second, right_blocker];
    assert_eq!(exact_minimum_words(&constrained, 1), 3);
    let relaxed = constrained.map(|mut request| {
        request.must_alias = None;
        request
    });
    assert_eq!(exact_minimum_words(&relaxed, 1), 2);

    for case in 0..64u32 {
        let shared_len = (case as usize % 3) + 1;
        let shared_alignment = 1usize << ((case >> 2) & 1);
        let mut first = request(0, shared_len, 0b0001);
        first.alignment_words = shared_alignment;
        first.must_alias = group;
        let left_blocker = request(1, ((case >> 1) as usize % 3) + 1, 0b0011);
        let mut second = request(2, shared_len, 0b0100);
        second.alignment_words = shared_alignment;
        second.must_alias = group;
        let right_blocker = request(3, ((case >> 3) as usize % 3) + 1, 0b0110);
        let requests = [first, left_blocker, second, right_blocker];

        let layout = allocate_ranges(&requests, 2, None).unwrap();
        let exact = exact_minimum_words(&requests, 2);
        assert!(
            independent_raw_peak_words(&requests) <= exact,
            "case {case}"
        );
        assert!(exact <= layout.total_words(), "case {case}");
        assert_eq!(
            layout.binding(RangeId(0)).unwrap().offset_words,
            layout.binding(RangeId(2)).unwrap().offset_words,
            "case {case}"
        );
    }
}

fn exact_minimum_words(requests: &[RangeRequest], slab_alignment_words: usize) -> usize {
    fn place(
        request_index: usize,
        offset: usize,
        requests: &[RangeRequest],
        offsets: &mut Vec<usize>,
        slab_alignment_words: usize,
        lower_bound: usize,
        best: &mut usize,
    ) {
        let request = requests[request_index];
        if offset % request.alignment_words != 0 {
            return;
        }
        let end = offset + request.len_words;
        if independent_align_up(end, slab_alignment_words) > *best {
            return;
        }
        let legal =
            requests[..request_index]
                .iter()
                .zip(offsets.iter())
                .all(|(other, other_offset)| {
                    request.live_mask & other.live_mask == 0
                        || end <= *other_offset
                        || *other_offset + other.len_words <= offset
                });
        if !legal {
            return;
        }
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
    }

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
            *best = (*best).min(independent_align_up(end, slab_alignment_words));
            return;
        }
        let request = requests[request_index];
        if request.len_words > *best {
            return;
        }
        let alias_offset = request.must_alias.and_then(|group| {
            requests[..request_index]
                .iter()
                .position(|other| other.must_alias == Some(group))
                .map(|index| offsets[index])
        });
        if let Some(offset) = alias_offset {
            place(
                request_index,
                offset,
                requests,
                offsets,
                slab_alignment_words,
                lower_bound,
                best,
            );
            return;
        }
        for offset in
            (0..=*best - request.len_words).filter(|offset| offset % request.alignment_words == 0)
        {
            place(
                request_index,
                offset,
                requests,
                offsets,
                slab_alignment_words,
                lower_bound,
                best,
            );
            if *best == lower_bound {
                return;
            }
        }
    }

    let lower_bound =
        independent_align_up(independent_raw_peak_words(requests), slab_alignment_words);
    let mut best = independent_serial_upper_bound(requests, slab_alignment_words);
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

fn independent_serial_upper_bound(requests: &[RangeRequest], slab_alignment_words: usize) -> usize {
    let mut aliases = Vec::<AliasGroupId>::new();
    let mut cursor = 0usize;
    for request in requests {
        if request
            .must_alias
            .is_some_and(|group| aliases.contains(&group))
        {
            continue;
        }
        cursor = independent_align_up(cursor, request.alignment_words);
        if let Some(group) = request.must_alias {
            aliases.push(group);
        }
        cursor += request.len_words;
    }
    independent_align_up(cursor, slab_alignment_words)
}

fn independent_raw_peak_words(requests: &[RangeRequest]) -> usize {
    (0..u16::BITS)
        .map(|bit| {
            let live_bit = 1u16 << bit;
            requests
                .iter()
                .filter(|request| request.live_mask & live_bit != 0)
                .map(|request| request.len_words)
                .sum()
        })
        .max()
        .unwrap_or(0)
}

fn independent_align_up(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + alignment - remainder
    }
}
