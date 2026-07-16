use super::*;

const INDEX_LIMIT: u64 = ((1u64 << 31) - 1) << stwo_backend_cuda::POW_GRIND_LOW_BITS;

#[test]
fn search_closes_at_the_resident_kernel_index_limit() {
    let pow = FleetPowPlan {
        workers_per_rank: 1,
        indices_per_attempt: 1,
    };
    let last = INDEX_LIMIT - 1;

    assert_eq!(pow.search_index(1, WorkerId(0), 0, last).unwrap(), last);
    assert_eq!(
        pow.nonce(1, WorkerId(0), 0, last).unwrap(),
        (0x7fff_fffeu64 << 32) | ((1u64 << stwo_backend_cuda::POW_GRIND_LOW_BITS) - 1)
    );
    assert_eq!(
        pow.search_index(1, WorkerId(0), 0, INDEX_LIMIT)
            .unwrap_err(),
        FleetPowError::SearchExhausted
    );
}

#[test]
fn receipt_outside_the_simd_lattice_fails_closed() {
    let pow = FleetPowPlan {
        workers_per_rank: 1,
        indices_per_attempt: 1,
    };
    for candidate_nonce in [
        1u64 << stwo_backend_cuda::POW_GRIND_LOW_BITS,
        stwo_backend_cuda::pow_index_to_nonce(INDEX_LIMIT),
    ] {
        let receipt = PowRankReceipt {
            site: FleetPowSite::Interaction,
            plan_identity: [5; 32],
            rank: WorkerId(0),
            proof_generation: 9,
            attempt_ordinal: 0,
            candidate_nonce: Some(candidate_nonce),
        };
        assert_eq!(
            pow.verify_winner(FleetPowSite::Interaction, 1, [5; 32], 9, &[receipt], |_| {
                false
            },)
                .unwrap_err(),
            FleetPowError::OutsideSimdLattice
        );
    }
}

#[test]
fn final_non_divisor_attempt_is_partial_then_exhausted() {
    let pow = FleetPowPlan {
        workers_per_rank: 1,
        indices_per_attempt: INDEX_LIMIT - 1,
    };

    assert_eq!(pow.attempt_bounds(0).unwrap(), (0, INDEX_LIMIT - 1));
    assert_eq!(
        pow.attempt_bounds(1).unwrap(),
        (INDEX_LIMIT - 1, INDEX_LIMIT)
    );
    assert_eq!(
        pow.attempt_bounds(2).unwrap_err(),
        FleetPowError::SearchExhausted
    );
}

#[test]
fn partition_is_exact_and_site_bound_winner_waits_for_every_rank() {
    let pow = FleetPowPlan {
        workers_per_rank: 2,
        indices_per_attempt: 8,
    };
    let mut indices = Vec::new();
    for iteration in 0..2 {
        for rank in [WorkerId(0), WorkerId(1)] {
            for local_worker in 0..2 {
                let index = pow.search_index(2, rank, local_worker, iteration).unwrap();
                assert_eq!(
                    pow.nonce(2, rank, local_worker, iteration).unwrap(),
                    stwo_backend_cuda::pow_index_to_nonce(index)
                );
                indices.push(index);
            }
        }
    }
    indices.sort_unstable();
    assert_eq!(indices, (0..8).collect::<Vec<_>>());

    let receipt = |rank, candidate_nonce| PowRankReceipt {
        site: FleetPowSite::Interaction,
        plan_identity: [9; 32],
        rank,
        proof_generation: 7,
        attempt_ordinal: 0,
        candidate_nonce,
    };
    let receipts = [
        receipt(WorkerId(1), Some(pow.nonce(2, WorkerId(1), 0, 0).unwrap())),
        receipt(WorkerId(0), Some(pow.nonce(2, WorkerId(0), 0, 0).unwrap())),
    ];
    let expected = receipts
        .iter()
        .filter_map(|receipt| receipt.candidate_nonce)
        .min()
        .unwrap();
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, [9; 32], 7, &receipts, |_| {
            true
        },)
            .unwrap(),
        expected
    );
    assert_eq!(
        pow.verify_winner(
            FleetPowSite::Interaction,
            2,
            [9; 32],
            7,
            &receipts[..1],
            |_| true,
        )
        .unwrap_err(),
        FleetPowError::IncompleteReceipts
    );

    let mut forged = receipts;
    forged[0].candidate_nonce = Some(pow.nonce(2, WorkerId(0), 0, 0).unwrap());
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, [9; 32], 7, &forged, |_| true,)
            .unwrap_err(),
        FleetPowError::InvalidReceipt(WorkerId(1))
    );
    let mut wrong_site = receipts;
    wrong_site[0].site = FleetPowSite::Query;
    assert_eq!(
        pow.verify_winner(
            FleetPowSite::Interaction,
            2,
            [9; 32],
            7,
            &wrong_site,
            |_| true,
        )
        .unwrap_err(),
        FleetPowError::InvalidReceipt(WorkerId(1))
    );
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, [9; 32], 7, &receipts, |_| {
            false
        },)
            .unwrap_err(),
        FleetPowError::InvalidReceipt(WorkerId(0))
    );

    let mut trailing = receipts.to_vec();
    trailing.extend([WorkerId(0), WorkerId(1)].map(|rank| PowRankReceipt {
        site: FleetPowSite::Interaction,
        plan_identity: [9; 32],
        rank,
        proof_generation: 7,
        attempt_ordinal: 1,
        candidate_nonce: None,
    }));
    assert_eq!(
        pow.verify_winner(FleetPowSite::Interaction, 2, [9; 32], 7, &trailing, |_| {
            true
        },)
            .unwrap_err(),
        FleetPowError::TrailingAttempts
    );
}
