use stwo_backend_cuda::{IpcExchangeInstallDomain, IPC_EXCHANGE_DESCRIPTOR_BYTES};

use super::super::ipc_control_frame::test_only_remac;
use super::runtime_view::transfer_fixture;
use super::*;

const GENERATION: u64 = 23;
const NONCE: [u8; 32] = [0x51; 32];
const CHANNEL_SECRET: [u8; 32] = [0xa7; 32];

fn view() -> FleetRuntimeView {
    compile(transfer_fixture()).unwrap().runtime_view().unwrap()
}

fn changed_view() -> FleetRuntimeView {
    let mut fixture = transfer_fixture();
    fixture.placement.topology.workers[1].capacity_bytes += 1;
    compile(fixture).unwrap().runtime_view().unwrap()
}

fn domain(byte: u8) -> IpcExchangeInstallDomain {
    IpcExchangeInstallDomain::from_digest([byte; 32]).unwrap()
}

fn codec(
    view: &FleetRuntimeView,
    generation: u64,
    install_domain: IpcExchangeInstallDomain,
    nonce: [u8; 32],
    rank: WorkerId,
    role: FleetIpcControlEndpointRole,
    key: [u8; 32],
) -> FleetIpcControlCodec {
    FleetIpcControlCodec::new(view, generation, install_domain, nonce, rank, role, key).unwrap()
}

#[test]
fn descriptor_payload_round_trips_byte_exactly_with_bound_header_fields() {
    let view = view();
    let install_domain = domain(0x61);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );
    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let payload = [0x3c; IPC_EXCHANGE_DESCRIPTOR_BYTES];
    let frame = sender
        .seal(
            FleetIpcControlMessageKind::RankDescriptorStatement,
            &payload,
        )
        .unwrap();
    let message = receiver.open(&frame).unwrap();

    assert_eq!(
        message.kind(),
        FleetIpcControlMessageKind::RankDescriptorStatement
    );
    assert_eq!(message.sequence(), 0);
    assert_eq!(message.payload(), payload);
}

#[test]
fn exact_install_handshake_shares_one_sequence_per_direction() {
    let view = view();
    let install_domain = domain(0x69);
    let mut controller = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let mut rank = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );

    let statement = rank
        .seal(
            FleetIpcControlMessageKind::RankDescriptorStatement,
            b"statement",
        )
        .unwrap();
    assert_eq!(controller.open(&statement).unwrap().sequence(), 0);

    let bundle = controller
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"bundle")
        .unwrap();
    assert_eq!(rank.open(&bundle).unwrap().sequence(), 0);

    let acknowledgement = rank
        .seal(
            FleetIpcControlMessageKind::InstallAcknowledgement,
            b"acknowledgement",
        )
        .unwrap();
    assert_eq!(controller.open(&acknowledgement).unwrap().sequence(), 1);
}

#[test]
fn direction_subkeys_reject_reflection_and_accept_the_opposite_role() {
    let view = view();
    let install_domain = domain(0x68);
    let mut controller = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let mut rank = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );

    let to_rank = controller
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"to-rank")
        .unwrap();
    assert_eq!(
        controller.open(&to_rank),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );
    assert_eq!(rank.open(&to_rank).unwrap().payload(), b"to-rank");

    let to_controller = rank
        .seal(
            FleetIpcControlMessageKind::RankDescriptorStatement,
            b"to-controller",
        )
        .unwrap();
    assert_eq!(
        rank.open(&to_controller),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );
    assert_eq!(
        controller.open(&to_controller).unwrap().payload(),
        b"to-controller"
    );
}

#[test]
fn wrong_direction_kinds_reject_without_advancing_either_sequence() {
    let view = view();
    let install_domain = domain(0x6a);

    let mut controller = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let mut rank = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );
    assert_eq!(
        controller.seal(
            FleetIpcControlMessageKind::RankDescriptorStatement,
            b"wrong"
        ),
        Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
            FleetIpcControlMessageKind::RankDescriptorStatement
        ))
    );
    let bundle = controller
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"bundle")
        .unwrap();
    let mut forged_ack = bundle.clone();
    forged_ack[10..12].copy_from_slice(
        &(FleetIpcControlMessageKind::InstallAcknowledgement as u16).to_le_bytes(),
    );
    test_only_remac(&mut forged_ack, &controller);
    assert_eq!(
        rank.open(&forged_ack),
        Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
            FleetIpcControlMessageKind::InstallAcknowledgement
        ))
    );
    assert_eq!(rank.open(&bundle).unwrap().sequence(), 0);

    let mut controller = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let mut rank = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );
    assert_eq!(
        rank.seal(FleetIpcControlMessageKind::DescriptorBundle, b"wrong"),
        Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
            FleetIpcControlMessageKind::DescriptorBundle
        ))
    );
    let statement = rank
        .seal(
            FleetIpcControlMessageKind::RankDescriptorStatement,
            b"statement",
        )
        .unwrap();
    let mut forged_bundle = statement.clone();
    forged_bundle[10..12]
        .copy_from_slice(&(FleetIpcControlMessageKind::DescriptorBundle as u16).to_le_bytes());
    test_only_remac(&mut forged_bundle, &rank);
    assert_eq!(
        controller.open(&forged_bundle),
        Err(FleetIpcControlFrameError::MessageKindForWrongDirection(
            FleetIpcControlMessageKind::DescriptorBundle
        ))
    );
    assert_eq!(controller.open(&statement).unwrap().sequence(), 0);
}

#[test]
fn receive_sequence_rejects_replay_and_reorder_without_advancing() {
    let view = view();
    let install_domain = domain(0x62);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );
    let first = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"first")
        .unwrap();
    let second = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"second")
        .unwrap();

    assert!(matches!(
        receiver.open(&second),
        Err(FleetIpcControlFrameError::SequenceMismatch {
            expected: 0,
            actual: 1
        })
    ));
    assert_eq!(receiver.open(&first).unwrap().payload(), b"first");
    assert!(matches!(
        receiver.open(&first),
        Err(FleetIpcControlFrameError::SequenceMismatch {
            expected: 1,
            actual: 0
        })
    ));
    assert_eq!(receiver.open(&second).unwrap().payload(), b"second");
}

#[test]
fn unauthenticated_changes_fail_before_semantics_and_do_not_advance_sequence() {
    let view = view();
    let install_domain = domain(0x63);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let frame = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"payload")
        .unwrap();

    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );

    let mut unknown_version = frame.clone();
    unknown_version[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert_eq!(
        receiver.open(&unknown_version),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );

    let mut unknown_kind = frame.clone();
    unknown_kind[10..12].copy_from_slice(&99u16.to_le_bytes());
    assert_eq!(
        receiver.open(&unknown_kind),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );

    let mut bad_mac = frame.clone();
    *bad_mac.last_mut().unwrap() ^= 1;
    assert_eq!(
        receiver.open(&bad_mac),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );

    let mut bad_payload = frame.clone();
    let payload_offset = bad_payload.len() - 32 - b"payload".len();
    bad_payload[payload_offset] ^= 1;
    assert_eq!(
        receiver.open(&bad_payload),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );
    assert_eq!(receiver.open(&frame).unwrap().payload(), b"payload");
}

#[test]
fn a_different_channel_secret_cannot_authenticate_the_frame() {
    let view = view();
    let install_domain = domain(0x6b);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let frame = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"payload")
        .unwrap();
    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        [0xa8; 32],
    );

    assert_eq!(
        receiver.open(&frame),
        Err(FleetIpcControlFrameError::AuthenticationFailed)
    );
}

#[test]
fn authenticated_malformed_frames_reject_version_kind_header_length_and_truncation() {
    let view = view();
    let install_domain = domain(0x67);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let frame = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"payload")
        .unwrap();
    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );

    let mut unknown_version = frame.clone();
    unknown_version[8..10].copy_from_slice(&2u16.to_le_bytes());
    test_only_remac(&mut unknown_version, &sender);
    assert_eq!(
        receiver.open(&unknown_version),
        Err(FleetIpcControlFrameError::UnknownProtocolVersion(2))
    );

    let mut unknown_kind = frame.clone();
    unknown_kind[10..12].copy_from_slice(&99u16.to_le_bytes());
    test_only_remac(&mut unknown_kind, &sender);
    assert_eq!(
        receiver.open(&unknown_kind),
        Err(FleetIpcControlFrameError::UnknownMessageKind(99))
    );

    let mut wrong_header_length = frame.clone();
    wrong_header_length[14..16].copy_from_slice(&135u16.to_le_bytes());
    test_only_remac(&mut wrong_header_length, &sender);
    assert_eq!(
        receiver.open(&wrong_header_length),
        Err(FleetIpcControlFrameError::InvalidHeaderLength(135))
    );

    let mut encoded_truncation = frame.clone();
    encoded_truncation[32..40].copy_from_slice(&8u64.to_le_bytes());
    test_only_remac(&mut encoded_truncation, &sender);
    assert!(matches!(
        receiver.open(&encoded_truncation),
        Err(FleetIpcControlFrameError::Truncated { .. })
    ));
    assert!(matches!(
        receiver.open(&frame[..167]),
        Err(FleetIpcControlFrameError::Truncated { .. })
    ));

    let mut trailing = frame.clone();
    let mac_offset = trailing.len() - 32;
    trailing.insert(mac_offset, 0x91);
    test_only_remac(&mut trailing, &sender);
    assert!(matches!(
        receiver.open(&trailing),
        Err(FleetIpcControlFrameError::TrailingBytes { .. })
    ));
    assert_eq!(receiver.open(&frame).unwrap().payload(), b"payload");
}

#[test]
fn context_bound_keys_reject_plan_generation_domain_nonce_and_rank_drift() {
    let view = view();
    let other_view = changed_view();
    assert_ne!(view.plan_identity(), other_view.plan_identity());
    let install_domain = domain(0x64);

    let cases = [
        codec(
            &other_view,
            GENERATION,
            install_domain,
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        codec(
            &view,
            GENERATION + 1,
            install_domain,
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        codec(
            &view,
            GENERATION,
            domain(0x65),
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        codec(
            &view,
            GENERATION,
            install_domain,
            [0x52; 32],
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        codec(
            &view,
            GENERATION,
            install_domain,
            NONCE,
            WorkerId(1),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
    ];

    for mut sender in cases {
        let frame = sender
            .seal(FleetIpcControlMessageKind::DescriptorBundle, b"bound")
            .unwrap();
        let mut receiver = codec(
            &view,
            GENERATION,
            install_domain,
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Rank,
            CHANNEL_SECRET,
        );
        assert_eq!(
            receiver.open(&frame),
            Err(FleetIpcControlFrameError::AuthenticationFailed)
        );
    }
}

#[test]
fn authenticated_context_header_drift_is_rejected_without_advancing() {
    let view = view();
    let install_domain = domain(0x64);
    let mut sender = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Controller,
        CHANNEL_SECRET,
    );
    let frame = sender
        .seal(FleetIpcControlMessageKind::DescriptorBundle, b"bound")
        .unwrap();

    let mut wrong_plan = frame.clone();
    wrong_plan[40..72].fill(0x55);
    test_only_remac(&mut wrong_plan, &sender);
    let mut wrong_generation = frame.clone();
    wrong_generation[16..24].copy_from_slice(&(GENERATION + 1).to_le_bytes());
    test_only_remac(&mut wrong_generation, &sender);
    let mut wrong_domain = frame.clone();
    wrong_domain[72..104].fill(0x65);
    test_only_remac(&mut wrong_domain, &sender);
    let mut wrong_nonce = frame.clone();
    wrong_nonce[104..136].fill(0x52);
    test_only_remac(&mut wrong_nonce, &sender);
    let mut wrong_rank = frame.clone();
    wrong_rank[12..14].copy_from_slice(&1u16.to_le_bytes());
    test_only_remac(&mut wrong_rank, &sender);

    let cases = [
        (wrong_plan, FleetIpcControlFrameError::PlanIdentityMismatch),
        (
            wrong_generation,
            FleetIpcControlFrameError::ProofGenerationMismatch,
        ),
        (
            wrong_domain,
            FleetIpcControlFrameError::InstallDomainMismatch,
        ),
        (wrong_nonce, FleetIpcControlFrameError::InstallNonceMismatch),
        (
            wrong_rank,
            FleetIpcControlFrameError::RankMismatch {
                expected: WorkerId(0),
                actual: WorkerId(1),
            },
        ),
    ];
    let mut receiver = codec(
        &view,
        GENERATION,
        install_domain,
        NONCE,
        WorkerId(0),
        FleetIpcControlEndpointRole::Rank,
        CHANNEL_SECRET,
    );
    for (malformed, expected) in cases {
        assert_eq!(receiver.open(&malformed), Err(expected));
    }
    assert_eq!(receiver.open(&frame).unwrap().sequence(), 0);
}

#[test]
fn construction_rejects_zero_secrets_and_unknown_rank() {
    let view = view();
    let install_domain = domain(0x66);
    assert!(matches!(
        FleetIpcControlCodec::new(
            &view,
            GENERATION,
            install_domain,
            [0; 32],
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        Err(FleetIpcControlFrameError::ZeroInstallNonce)
    ));
    assert!(matches!(
        FleetIpcControlCodec::new(
            &view,
            GENERATION,
            install_domain,
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            [0; 32],
        ),
        Err(FleetIpcControlFrameError::ZeroMacKey)
    ));
    assert!(matches!(
        FleetIpcControlCodec::new(
            &view,
            u64::MAX,
            install_domain,
            NONCE,
            WorkerId(0),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        Err(FleetIpcControlFrameError::ProofGenerationOverflow)
    ));
    assert!(matches!(
        FleetIpcControlCodec::new(
            &view,
            GENERATION,
            install_domain,
            NONCE,
            WorkerId(2),
            FleetIpcControlEndpointRole::Controller,
            CHANNEL_SECRET,
        ),
        Err(FleetIpcControlFrameError::UnknownRank(WorkerId(2)))
    ));
}
