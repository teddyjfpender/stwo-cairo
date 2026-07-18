use std::collections::VecDeque;

use stwo_backend_cuda::pow_index_to_nonce;

use super::*;
use crate::fleet_pow::FleetPowPlan;

const IDENTITY: [u8; 32] = [7; 32];

struct FakeTransport {
    sent: Option<FleetPowRankRequest>,
    responses: VecDeque<FleetPowRankResponse>,
}

impl FleetPowTransport for FakeTransport {
    fn send(&mut self, request: &FleetPowRankRequest) -> Result<(), FleetPowRuntimeError> {
        self.sent = Some(request.clone());
        Ok(())
    }

    fn receive(&mut self) -> Result<FleetPowRankResponse, FleetPowRuntimeError> {
        self.responses
            .pop_front()
            .or_else(|| {
                self.sent
                    .as_ref()
                    .map(|request| FleetPowRankResponse::completed(request, None))
            })
            .ok_or_else(|| FleetPowRuntimeError::Transport("no response".into()))
    }
}

fn schedule() -> FleetPowSchedule {
    let plan = FleetPowPlan {
        workers_per_rank: POW_THREADS_PER_BLOCK,
        indices_per_attempt: 1024,
    };
    FleetPowSchedule {
        interaction: plan,
        query: plan,
    }
}

fn coordinator(responses: Vec<FleetPowRankResponse>) -> TwoRankFleetPowCoordinator<FakeTransport> {
    TwoRankFleetPowCoordinator::new(
        schedule(),
        IDENTITY,
        9,
        FakeTransport {
            sent: None,
            responses: responses.into(),
        },
    )
    .unwrap()
}

fn request(rank: u16, attempt_ordinal: u64) -> FleetPowRankRequest {
    let plan = schedule().interaction;
    FleetPowRankRequest::new(
        IDENTITY,
        9,
        FleetPowSite::Interaction,
        attempt_ordinal,
        WorkerId(rank),
        26,
        plan.workers_per_rank,
        plan.attempt(RANK_COUNT, attempt_ordinal).unwrap(),
        state(),
    )
    .unwrap()
}

fn state() -> [u32; BLAKE2S_TRANSCRIPT_STATE_WORDS] {
    let mut state = [3; BLAKE2S_TRANSCRIPT_STATE_WORDS];
    state[TRANSCRIPT_STATUS_WORD] = 0;
    state
}

#[test]
fn fixed_frames_round_trip_and_bind_the_state() {
    let request = request(1, 2);
    assert_eq!(
        FleetPowRankRequest::from_bytes(&request.to_bytes()).unwrap(),
        request
    );
    let response = FleetPowRankResponse::completed(&request, Some(17));
    assert_eq!(
        FleetPowRankResponse::from_bytes(&response.to_bytes()).unwrap(),
        response
    );
    let mut other_state = request.clone();
    other_state.transcript_state[0] ^= 1;
    assert_eq!(
        response.validate_for(&other_state),
        Err(FleetPowRuntimeError::MismatchedResponse("request identity"))
    );
}

#[test]
fn wait_all_retries_then_returns_the_shared_global_minimum() {
    let remote_attempt_zero = FleetPowRankResponse::completed(&request(1, 0), None);
    let remote_nonce = pow_index_to_nonce(1280);
    let remote_attempt_one = FleetPowRankResponse::completed(&request(1, 1), Some(remote_nonce));
    let mut coordinator = coordinator(vec![remote_attempt_zero, remote_attempt_one]);
    let result = coordinator
        .resolve(
            FleetPowSite::Interaction,
            26,
            state(),
            |request| Ok(FleetPowRankResponse::completed(request, None)),
            |nonce| nonce == remote_nonce,
        )
        .unwrap();
    assert_eq!(
        result,
        FleetPowResolution {
            nonce: remote_nonce,
            attempt_ordinal: 1
        }
    );
}

#[test]
fn local_failure_still_drains_the_owned_remote_response() {
    let remote = FleetPowRankResponse::completed(&request(1, 0), None);
    let mut coordinator = coordinator(vec![remote]);
    assert_eq!(
        coordinator.resolve(
            FleetPowSite::Interaction,
            26,
            state(),
            |_| Err(FleetPowRuntimeError::Transport("local failure".into())),
            |_| false,
        ),
        Err(FleetPowRuntimeError::Transport("local failure".into()))
    );
    assert!(coordinator.transport.responses.is_empty());
}

#[test]
fn rejects_stale_duplicate_and_wrong_identity_fields() {
    let base = FleetPowRankResponse::completed(&request(1, 0), None);
    let forged = [
        {
            let mut value = base.clone();
            value.proof_generation += 1;
            (value, "generation")
        },
        {
            let mut value = base.clone();
            value.site = FleetPowSite::Query;
            (value, "site")
        },
        {
            let mut value = base.clone();
            value.rank = WorkerId(0);
            (value, "rank")
        },
        {
            let mut value = base.clone();
            value.start_index += 1;
            (value, "tile")
        },
        {
            let mut value = base;
            value.request_identity = [0; 32];
            (value, "request identity")
        },
    ];
    for (response, expected) in forged {
        let error = coordinator(vec![response])
            .resolve(
                FleetPowSite::Interaction,
                26,
                state(),
                |request| Ok(FleetPowRankResponse::completed(request, None)),
                |_| false,
            )
            .unwrap_err();
        assert_eq!(error, FleetPowRuntimeError::MismatchedResponse(expected));
    }
}

#[test]
fn rejects_duplicate_site_and_malformed_wire_geometry() {
    let nonce = pow_index_to_nonce(256);
    let response = FleetPowRankResponse::completed(&request(1, 0), Some(nonce));
    let mut coordinator = coordinator(vec![response]);
    coordinator
        .resolve(
            FleetPowSite::Interaction,
            26,
            state(),
            |request| Ok(FleetPowRankResponse::completed(request, None)),
            |candidate| candidate == nonce,
        )
        .unwrap();
    assert!(matches!(
        coordinator.resolve(
            FleetPowSite::Interaction,
            26,
            state(),
            |_| unreachable!(),
            |_| false,
        ),
        Err(FleetPowRuntimeError::UnexpectedSite { .. })
    ));

    let mut bytes = request(1, 0).to_bytes();
    put_u32(&mut bytes, 20, POW_THREADS_PER_BLOCK + 1);
    assert_eq!(
        FleetPowRankRequest::from_bytes(&bytes),
        Err(FleetPowRuntimeError::InvalidFrame("request geometry"))
    );

    let mut bytes = request(1, 0).to_bytes();
    put_u32(&mut bytes, 96 + TRANSCRIPT_STATUS_WORD * 4, 1);
    assert_eq!(
        FleetPowRankRequest::from_bytes(&bytes),
        Err(FleetPowRuntimeError::InvalidFrame("request geometry"))
    );
}
