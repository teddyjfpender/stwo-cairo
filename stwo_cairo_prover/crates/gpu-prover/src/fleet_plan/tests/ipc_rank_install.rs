use stwo_backend_cuda::{
    IpcExchangeDescriptor, IpcExchangeInstallDomain, IPC_EXCHANGE_DESCRIPTOR_BYTES,
};

use super::runtime_view::transfer_fixture;
use super::*;

const GENERATION: u64 = 17;

fn view() -> FleetRuntimeView {
    compile(transfer_fixture()).unwrap().runtime_view().unwrap()
}

fn domain(byte: u8) -> IpcExchangeInstallDomain {
    IpcExchangeInstallDomain::from_digest([byte; 32]).unwrap()
}

fn descriptor(
    view: &FleetRuntimeView,
    install_domain: IpcExchangeInstallDomain,
    generation: u64,
    edge: u64,
    owner: u32,
    peer: u32,
    logical_bytes: usize,
    handle_seed: u8,
) -> IpcExchangeDescriptor {
    let allocation_bytes = logical_bytes.next_multiple_of(2 * 1024 * 1024) as u64;
    let mut wire = [0u8; IPC_EXCHANGE_DESCRIPTOR_BYTES];
    wire[0..8].copy_from_slice(b"STWOIPCX");
    wire[8..12].copy_from_slice(&2u32.to_le_bytes());
    wire[12..16].copy_from_slice(&1u32.to_le_bytes());
    wire[16..24].copy_from_slice(&edge.to_le_bytes());
    wire[24..28].copy_from_slice(&owner.to_le_bytes());
    wire[28..32].copy_from_slice(&peer.to_le_bytes());
    wire[32..40].copy_from_slice(&generation.to_le_bytes());
    wire[40..48].copy_from_slice(&(logical_bytes as u64).to_le_bytes());
    wire[48..56].copy_from_slice(&allocation_bytes.to_le_bytes());
    wire[56..72].fill(1);
    wire[72..88].fill(2);
    wire[88..120].copy_from_slice(install_domain.as_bytes());
    wire[120..184].fill(handle_seed);
    wire[184..248].fill(handle_seed.wrapping_add(1));
    wire[248..312].fill(handle_seed.wrapping_add(2));
    let decoded = IpcExchangeDescriptor::decode(&wire).unwrap();
    assert_eq!(
        decoded.key().logical_bytes(),
        view.spans()[0].logical_bytes()
    );
    decoded
}

fn statements(
    view: &FleetRuntimeView,
    descriptor: IpcExchangeDescriptor,
    install_domain: IpcExchangeInstallDomain,
) -> [FleetIpcRankDescriptorStatement; 2] {
    [
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(0),
            GENERATION,
            install_domain,
            vec![descriptor],
        ),
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(1),
            GENERATION,
            install_domain,
            vec![],
        ),
    ]
}

fn valid_descriptor(
    view: &FleetRuntimeView,
    install_domain: IpcExchangeInstallDomain,
) -> IpcExchangeDescriptor {
    let span = view.spans()[0];
    descriptor(
        view,
        install_domain,
        GENERATION,
        span.edge_ordinal,
        u32::from(span.owner.0),
        u32::from(span.peer.0),
        span.logical_bytes(),
        0x31,
    )
}

#[test]
fn two_rank_bundle_is_dense_plan_bound_and_digest_stable() {
    let view = view();
    let install_domain = domain(0x71);
    let statements = statements(
        &view,
        valid_descriptor(&view, install_domain),
        install_domain,
    );
    let bundle =
        FleetIpcDescriptorBundle::join_two_rank(&view, [&statements[0], &statements[1]]).unwrap();
    let repeat =
        FleetIpcDescriptorBundle::join_two_rank(&view, [&statements[0], &statements[1]]).unwrap();

    assert_eq!(statements[0].worker(), WorkerId(0));
    assert_eq!(statements[1].worker(), WorkerId(1));
    assert_eq!(statements[0].descriptors().len(), 1);
    assert!(statements[1].descriptors().is_empty());
    assert_eq!(bundle.descriptors().len(), view.spans().len());
    assert_eq!(bundle.descriptor_digest(), repeat.descriptor_digest());
}

#[test]
fn bundle_rejects_rank_plan_generation_domain_and_owner_statement_drift() {
    let view = view();
    let install_domain = domain(0x72);
    let descriptor = valid_descriptor(&view, install_domain);
    let valid = statements(&view, descriptor.clone(), install_domain);
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&valid[1], &valid[0]]),
        Err(FleetIpcRankInstallError::DescriptorStatementOrder { .. })
    ));

    let wrong_plan = FleetIpcRankDescriptorStatement::test_only(
        [0x55; 32],
        WorkerId(1),
        GENERATION,
        install_domain,
        vec![],
    );
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&valid[0], &wrong_plan]),
        Err(FleetIpcRankInstallError::DescriptorStatementMismatch(
            WorkerId(1)
        ))
    ));

    let wrong_generation = FleetIpcRankDescriptorStatement::test_only(
        view.plan_identity(),
        WorkerId(1),
        GENERATION + 1,
        install_domain,
        vec![],
    );
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&valid[0], &wrong_generation]),
        Err(FleetIpcRankInstallError::DescriptorStatementMismatch(
            WorkerId(1)
        ))
    ));

    let wrong_domain = FleetIpcRankDescriptorStatement::test_only(
        view.plan_identity(),
        WorkerId(1),
        GENERATION,
        domain(0x73),
        vec![],
    );
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&valid[0], &wrong_domain]),
        Err(FleetIpcRankInstallError::DescriptorStatementMismatch(
            WorkerId(1)
        ))
    ));

    let false_owner = [
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(0),
            GENERATION,
            install_domain,
            vec![],
        ),
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(1),
            GENERATION,
            install_domain,
            vec![descriptor],
        ),
    ];
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&false_owner[0], &false_owner[1]]),
        Err(FleetIpcRankInstallError::DescriptorKeyMismatch(0))
    ));
}

#[test]
fn bundle_rejects_missing_duplicate_and_exact_edge_geometry_drift() {
    let view = view();
    let install_domain = domain(0x74);
    let valid_descriptor = valid_descriptor(&view, install_domain);
    let missing = [
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(0),
            GENERATION,
            install_domain,
            vec![],
        ),
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(1),
            GENERATION,
            install_domain,
            vec![],
        ),
    ];
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&missing[0], &missing[1]]),
        Err(FleetIpcRankInstallError::DescriptorCount {
            expected: 1,
            actual: 0
        })
    ));

    let duplicate = [
        FleetIpcRankDescriptorStatement::test_only(
            view.plan_identity(),
            WorkerId(0),
            GENERATION,
            install_domain,
            vec![valid_descriptor.clone(), valid_descriptor.clone()],
        ),
        missing[1].clone(),
    ];
    assert!(matches!(
        FleetIpcDescriptorBundle::join_two_rank(&view, [&duplicate[0], &duplicate[1]]),
        Err(FleetIpcRankInstallError::DescriptorCount {
            expected: 1,
            actual: 2
        })
    ));

    let span = view.spans()[0];
    for malformed in [
        descriptor(
            &view,
            install_domain,
            GENERATION,
            9,
            0,
            1,
            span.logical_bytes(),
            0x41,
        ),
        descriptor(
            &view,
            install_domain,
            GENERATION,
            0,
            1,
            0,
            span.logical_bytes(),
            0x42,
        ),
        descriptor(
            &view,
            install_domain,
            GENERATION + 1,
            span.edge_ordinal,
            0,
            1,
            span.logical_bytes(),
            0x43,
        ),
        descriptor(
            &view,
            domain(0x75),
            GENERATION,
            span.edge_ordinal,
            0,
            1,
            span.logical_bytes(),
            0x44,
        ),
    ] {
        let malformed = statements(&view, malformed, install_domain);
        assert!(
            FleetIpcDescriptorBundle::join_two_rank(&view, [&malformed[0], &malformed[1]]).is_err()
        );
    }
}

#[test]
fn handle_bytes_are_bound_into_the_bundle_digest() {
    let view = view();
    let install_domain = domain(0x76);
    let first = statements(
        &view,
        valid_descriptor(&view, install_domain),
        install_domain,
    );
    let second = statements(
        &view,
        descriptor(
            &view,
            install_domain,
            GENERATION,
            0,
            0,
            1,
            view.spans()[0].logical_bytes(),
            0x7f,
        ),
        install_domain,
    );
    let first_bundle =
        FleetIpcDescriptorBundle::join_two_rank(&view, [&first[0], &first[1]]).unwrap();
    let second_bundle =
        FleetIpcDescriptorBundle::join_two_rank(&view, [&second[0], &second[1]]).unwrap();
    assert_ne!(
        first_bundle.descriptor_digest(),
        second_bundle.descriptor_digest()
    );
}
