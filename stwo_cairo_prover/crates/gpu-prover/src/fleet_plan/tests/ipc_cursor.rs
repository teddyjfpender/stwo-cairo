use stwo_backend_cuda::{
    CudaDeviceIdentityReceipt, CudaDeviceUuid, IpcExchangeDescriptor, IpcExchangeInstallDomain,
    IpcExchangeKey, IPC_EXCHANGE_ALLOCATION_ALIGNMENT, IPC_EXCHANGE_DESCRIPTOR_BYTES,
};

use super::runtime_view::transfer_fixture;
use super::*;

const PROOF_GENERATION: u64 = 41;
const INSTALL_NONCE: [u8; 32] = [0x51; 32];

fn view() -> FleetRuntimeView {
    compile(transfer_fixture()).unwrap().runtime_view().unwrap()
}

fn two_wave_view() -> FleetRuntimeView {
    let mut fixture = transfer_fixture();
    let second = fixture
        .placement
        .owners
        .iter()
        .find(|owner| {
            owner.live.start == ScheduleStep(0) && owner.value.version != fixture.spill_value
        })
        .unwrap()
        .value;
    let value = fixture.compiled.value(second.version).unwrap();
    let layout = value.layout.clone();
    let alignment = value.alignment;
    let bytes = layout.logical_bytes().unwrap();
    let storage = StorageId(fixture.placement.storages.len() as u32);

    fixture.placement.topology.links[0].max_transfer_bytes = fixture.placement.topology.links[0]
        .max_transfer_bytes
        .max(bytes);
    fixture.placement.topology.workers[0].exchange_reserve_bytes +=
        IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[0].capacity_bytes += IPC_EXCHANGE_ALLOCATION_ALIGNMENT;
    fixture.placement.topology.workers[1].capacity_bytes += bytes;
    fixture.placement.replicas.push(FleetReplicaPlacement {
        id: ReplicaId(1),
        value: second,
        canonical_worker: WorkerId(0),
        worker: WorkerId(1),
        layout: layout.clone(),
        origin: ReplicaOrigin::Transition(LayoutTransitionId(1)),
        live: during(3, 50),
    });
    fixture
        .placement
        .transitions
        .push(FleetTransitionPlacement {
            id: LayoutTransitionId(1),
            value: second,
            source_worker: WorkerId(0),
            destination_replica: ReplicaId(1),
            axes: layout
                .axes
                .iter()
                .map(|axis| AxisMap {
                    source: axis.tag,
                    destination: axis.tag,
                })
                .collect(),
            interval: ExecutionInterval::BeforeBarrier(0),
            during: during(3, 4),
            scratch_bytes: 8,
            scratch_worker: WorkerId(1),
            route: FleetLinkId(0),
        });
    fixture.placement.storages.push(StorageDesc {
        id: storage,
        worker: WorkerId(1),
        bytes,
        alignment_bytes: alignment,
    });
    fixture
        .placement
        .storage_bindings
        .push(FleetStoragePlacement {
            storage,
            value: second,
            offset_bytes: 0,
        });
    compile(fixture).unwrap().runtime_view().unwrap()
}

fn device(worker: WorkerId) -> CudaDeviceUuid {
    synthetic_key(
        0,
        0,
        1,
        [worker.0 as u8 + 1; 16],
        [worker.0 as u8 + 101; 16],
        IpcExchangeInstallDomain::from_digest([0x77; 32]).unwrap(),
        64,
        0,
    )
    .owner_device()
}

#[allow(clippy::too_many_arguments)]
fn synthetic_key(
    edge: u64,
    owner_rank: u32,
    peer_rank: u32,
    owner_device: [u8; 16],
    peer_device: [u8; 16],
    install_domain: IpcExchangeInstallDomain,
    logical_bytes: usize,
    initial_generation: u64,
) -> IpcExchangeKey {
    let allocation_bytes = logical_bytes.next_multiple_of(IPC_EXCHANGE_ALLOCATION_ALIGNMENT) as u64;
    let mut wire = [0u8; IPC_EXCHANGE_DESCRIPTOR_BYTES];
    wire[0..8].copy_from_slice(b"STWOIPCX");
    wire[8..12].copy_from_slice(&2u32.to_le_bytes());
    wire[12..16].copy_from_slice(&1u32.to_le_bytes());
    wire[16..24].copy_from_slice(&edge.to_le_bytes());
    wire[24..28].copy_from_slice(&owner_rank.to_le_bytes());
    wire[28..32].copy_from_slice(&peer_rank.to_le_bytes());
    wire[32..40].copy_from_slice(&initial_generation.to_le_bytes());
    wire[40..48].copy_from_slice(&(logical_bytes as u64).to_le_bytes());
    wire[48..56].copy_from_slice(&allocation_bytes.to_le_bytes());
    wire[56..72].copy_from_slice(&owner_device);
    wire[72..88].copy_from_slice(&peer_device);
    wire[88..120].copy_from_slice(install_domain.as_bytes());
    IpcExchangeDescriptor::decode(&wire).unwrap().key()
}

fn runtime(view: &FleetRuntimeView, nonce: [u8; 32]) -> FleetIpcRuntimeRosterBinding {
    FleetIpcRuntimeRosterBinding::bind_test_only(
        view,
        nonce,
        WorkerId(0),
        device(WorkerId(0)),
        &[
            (WorkerId(0), device(WorkerId(0))),
            (WorkerId(1), device(WorkerId(1))),
        ],
    )
    .unwrap()
}

fn key(
    span: FleetTransferSpan,
    initial_generation: u64,
    install_domain: IpcExchangeInstallDomain,
) -> IpcExchangeKey {
    IpcExchangeKey::new(
        span.edge_ordinal,
        u32::from(span.owner.0),
        u32::from(span.peer.0),
        device(span.owner),
        device(span.peer),
        install_domain,
        span.logical_bytes(),
        initial_generation,
    )
    .unwrap()
}

fn keys(
    view: &FleetRuntimeView,
    initial_generation: u64,
    install_domain: IpcExchangeInstallDomain,
) -> Vec<IpcExchangeKey> {
    view.spans()
        .iter()
        .copied()
        .map(|span| key(span, initial_generation, install_domain))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn rekey(
    edge: u64,
    owner_rank: u32,
    peer_rank: u32,
    owner_device: CudaDeviceUuid,
    peer_device: CudaDeviceUuid,
    install_domain: IpcExchangeInstallDomain,
    logical_bytes: usize,
    initial_generation: u64,
) -> IpcExchangeKey {
    IpcExchangeKey::new(
        edge,
        owner_rank,
        peer_rank,
        owner_device,
        peer_device,
        install_domain,
        logical_bytes,
        initial_generation,
    )
    .unwrap()
}

fn unwrap_cursor_error<T>(result: Result<T, FleetIpcCursorError>) -> FleetIpcCursorError {
    match result {
        Ok(_) => panic!("fleet IPC operation unexpectedly succeeded"),
        Err(error) => error,
    }
}

fn generation(phase: FleetIpcPhase) -> u64 {
    if phase == FleetIpcPhase::Armed {
        PROOF_GENERATION + 1
    } else {
        PROOF_GENERATION
    }
}

fn accept(
    cursor: &mut IpcScheduleCursor,
    edge: u64,
    phase: FleetIpcPhase,
) -> Result<FleetIpcCursorProgress, FleetIpcCursorError> {
    cursor.accept_phase(edge, phase, generation(phase))
}

#[test]
fn installed_cursor_binds_every_exact_backend_key() {
    let view = view();
    let runtime_binding = runtime(&view, INSTALL_NONCE);
    let exchange_keys = keys(&view, 0, runtime_binding.install_domain().unwrap());
    let cursor = FleetIpcCoordinatorCursor::install(
        &view,
        PROOF_GENERATION,
        &runtime_binding,
        &exchange_keys,
    )
    .unwrap();
    assert_eq!(cursor.plan_identity(), view.plan_identity());
    assert_eq!(cursor.proof_generation(), PROOF_GENERATION);
    assert_eq!(
        cursor.state(),
        FleetIpcAttemptState::Running {
            wave_step: ScheduleStep(1)
        }
    );
    assert_eq!(cursor.active_wave_edges(), [0]);

    assert!(matches!(
        FleetIpcCoordinatorCursor::install(&view, PROOF_GENERATION, &runtime_binding, &[]),
        Err(FleetIpcCursorError::ExchangeKeyCount {
            expected: 1,
            actual: 0
        })
    ));

    let mut wrong = exchange_keys;
    wrong[0] = IpcExchangeKey::new(
        9,
        wrong[0].owner_rank(),
        wrong[0].peer_rank(),
        wrong[0].owner_device(),
        wrong[0].peer_device(),
        wrong[0].install_domain(),
        wrong[0].logical_bytes(),
        wrong[0].initial_generation(),
    )
    .unwrap();
    assert!(matches!(
        FleetIpcCoordinatorCursor::install(&view, PROOF_GENERATION, &runtime_binding, &wrong),
        Err(FleetIpcCursorError::ExchangeKeyMismatch(0))
    ));

    let future = keys(
        &view,
        PROOF_GENERATION + 1,
        runtime_binding.install_domain().unwrap(),
    );
    assert!(matches!(
        FleetIpcCoordinatorCursor::install(&view, PROOF_GENERATION, &runtime_binding, &future),
        Err(FleetIpcCursorError::ExchangeKeyMismatch(0))
    ));
}

#[test]
fn install_domain_binds_plan_nonce_roster_and_exact_edge_geometry() {
    let view = view();
    let runtime_binding = runtime(&view, INSTALL_NONCE);
    let domain = runtime_binding.install_domain().unwrap();
    let valid = keys(&view, 0, domain);
    let original = valid[0];
    let reject = |keys: &[IpcExchangeKey], error| {
        assert_eq!(
            unwrap_cursor_error(FleetIpcCoordinatorCursor::install(
                &view,
                PROOF_GENERATION,
                &runtime_binding,
                keys,
            )),
            error
        );
    };

    let mut mutated = valid.clone();
    mutated[0] = rekey(
        9,
        original.owner_rank(),
        original.peer_rank(),
        original.owner_device(),
        original.peer_device(),
        domain,
        original.logical_bytes(),
        original.initial_generation(),
    );
    reject(&mutated, FleetIpcCursorError::ExchangeKeyMismatch(0));

    mutated[0] = rekey(
        original.edge_id(),
        2,
        original.peer_rank(),
        original.owner_device(),
        original.peer_device(),
        domain,
        original.logical_bytes(),
        original.initial_generation(),
    );
    reject(&mutated, FleetIpcCursorError::ExchangeKeyMismatch(0));

    mutated[0] = rekey(
        original.edge_id(),
        original.owner_rank(),
        original.peer_rank(),
        original.owner_device(),
        original.peer_device(),
        domain,
        original.logical_bytes() + 1,
        original.initial_generation(),
    );
    reject(&mutated, FleetIpcCursorError::ExchangeKeyMismatch(0));

    mutated[0] = rekey(
        original.edge_id(),
        original.owner_rank(),
        original.peer_rank(),
        original.owner_device(),
        original.peer_device(),
        domain,
        original.logical_bytes(),
        PROOF_GENERATION + 1,
    );
    reject(&mutated, FleetIpcCursorError::ExchangeKeyMismatch(0));

    mutated[0] = rekey(
        original.edge_id(),
        original.owner_rank(),
        original.peer_rank(),
        original.owner_device(),
        original.peer_device(),
        IpcExchangeInstallDomain::from_digest([0x91; 32]).unwrap(),
        original.logical_bytes(),
        original.initial_generation(),
    );
    reject(&mutated, FleetIpcCursorError::InstallDomainMismatch(0));

    mutated[0] = rekey(
        original.edge_id(),
        original.owner_rank(),
        original.peer_rank(),
        device(WorkerId(2)),
        original.peer_device(),
        domain,
        original.logical_bytes(),
        original.initial_generation(),
    );
    reject(&mutated, FleetIpcCursorError::DeviceRosterMismatch(0));

    let other_install = runtime(&view, [0x52; 32]);
    assert_eq!(
        unwrap_cursor_error(FleetIpcCoordinatorCursor::install(
            &view,
            PROOF_GENERATION,
            &other_install,
            &valid,
        )),
        FleetIpcCursorError::InstallDomainMismatch(0)
    );

    let swapped_roster = FleetIpcRuntimeRosterBinding::bind_test_only(
        &view,
        INSTALL_NONCE,
        WorkerId(0),
        device(WorkerId(1)),
        &[
            (WorkerId(0), device(WorkerId(1))),
            (WorkerId(1), device(WorkerId(0))),
        ],
    )
    .unwrap();
    assert_ne!(
        runtime_binding.install_domain().unwrap(),
        swapped_roster.install_domain().unwrap()
    );
    assert_eq!(
        unwrap_cursor_error(FleetIpcCoordinatorCursor::install(
            &view,
            PROOF_GENERATION,
            &swapped_roster,
            &valid,
        )),
        FleetIpcCursorError::InstallDomainMismatch(0)
    );

    let mut foreign_fixture = transfer_fixture();
    foreign_fixture.placement.topology.workers[1].capacity_bytes += 1;
    let foreign_view = compile(foreign_fixture).unwrap().runtime_view().unwrap();
    assert_ne!(view.plan_identity(), foreign_view.plan_identity());
    assert_eq!(
        unwrap_cursor_error(FleetIpcCoordinatorCursor::install(
            &foreign_view,
            PROOF_GENERATION,
            &runtime_binding,
            &valid,
        )),
        FleetIpcCursorError::RuntimeRosterPlanMismatch
    );
    let foreign_runtime = runtime(&foreign_view, INSTALL_NONCE);
    assert_eq!(
        unwrap_cursor_error(FleetIpcCoordinatorCursor::install(
            &foreign_view,
            PROOF_GENERATION,
            &foreign_runtime,
            &valid,
        )),
        FleetIpcCursorError::InstallDomainMismatch(0)
    );
}

#[test]
fn runtime_roster_is_dense_complete_nonzero_nonce_and_uuid_unique() {
    let view = view();
    let roster = [
        (WorkerId(0), device(WorkerId(0))),
        (WorkerId(1), device(WorkerId(1))),
    ];
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            [0; 32],
            WorkerId(0),
            device(WorkerId(0)),
            &roster,
        )),
        FleetIpcCursorError::ZeroControllerInstallNonce
    );
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            INSTALL_NONCE,
            WorkerId(0),
            device(WorkerId(0)),
            &roster[..1],
        )),
        FleetIpcCursorError::RuntimeRosterCount {
            expected: 2,
            actual: 1,
        }
    );

    let reversed = [roster[1], roster[0]];
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            INSTALL_NONCE,
            WorkerId(0),
            device(WorkerId(0)),
            &reversed,
        )),
        FleetIpcCursorError::NonDenseRuntimeRoster {
            expected: WorkerId(0),
            actual: WorkerId(1),
        }
    );

    let duplicate = [roster[0], (WorkerId(1), roster[0].1)];
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            INSTALL_NONCE,
            WorkerId(0),
            device(WorkerId(0)),
            &duplicate,
        )),
        FleetIpcCursorError::DuplicateRuntimeDevice {
            first: WorkerId(0),
            duplicate: WorkerId(1),
        }
    );
}

#[test]
fn each_process_binds_only_its_local_context_to_the_same_public_roster() {
    let view = view();
    let roster = [
        (WorkerId(0), device(WorkerId(0))),
        (WorkerId(1), device(WorkerId(1))),
    ];
    let rank_0 = FleetIpcRuntimeRosterBinding::bind_test_only(
        &view,
        INSTALL_NONCE,
        WorkerId(0),
        roster[0].1,
        &roster,
    )
    .unwrap();
    let rank_1 = FleetIpcRuntimeRosterBinding::bind_test_only(
        &view,
        INSTALL_NONCE,
        WorkerId(1),
        roster[1].1,
        &roster,
    )
    .unwrap();

    assert_eq!(rank_0.local_worker(), WorkerId(0));
    assert_eq!(rank_1.local_worker(), WorkerId(1));
    assert_eq!(
        rank_0.install_domain().unwrap(),
        rank_1.install_domain().unwrap()
    );
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            INSTALL_NONCE,
            WorkerId(1),
            roster[0].1,
            &roster,
        )),
        FleetIpcCursorError::LocalDeviceAnnouncementMismatch(WorkerId(1))
    );
    assert_eq!(
        unwrap_cursor_error(FleetIpcRuntimeRosterBinding::bind_test_only(
            &view,
            INSTALL_NONCE,
            WorkerId(2),
            roster[0].1,
            &roster,
        )),
        FleetIpcCursorError::LocalWorkerMissing(WorkerId(2))
    );
}

#[test]
fn production_roster_constructor_requires_backend_identity_receipts() {
    let _constructor: fn(
        &FleetRuntimeView,
        [u8; 32],
        WorkerId,
        CudaDeviceIdentityReceipt,
        &[(WorkerId, CudaDeviceUuid)],
    ) -> Result<FleetIpcRuntimeRosterBinding, FleetIpcCursorError> =
        FleetIpcRuntimeRosterBinding::bind_local;
}

#[test]
fn exact_four_phase_sequence_completes_and_armed_uses_next_generation() {
    let view = view();
    let mut cursor = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(cursor.active_wave_edges(), [0]);
    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
    ] {
        assert_eq!(
            accept(&mut cursor, 0, phase).unwrap(),
            FleetIpcCursorProgress::WavePending {
                step: ScheduleStep(1)
            }
        );
    }
    assert_eq!(
        accept(&mut cursor, 0, FleetIpcPhase::Armed).unwrap(),
        FleetIpcCursorProgress::Complete {
            completed: ScheduleStep(1)
        }
    );
    assert_eq!(cursor.state(), FleetIpcAttemptState::Complete);
}

#[test]
fn same_step_spans_are_one_wave_without_edge_ordering() {
    let plan = compile(super::runtime_view::split_transfer_fixture()).unwrap();
    let view = plan.runtime_view().unwrap();
    let mut cursor = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(cursor.active_wave_edges(), [0, 1, 2, 3]);

    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
        FleetIpcPhase::Armed,
    ] {
        for edge in [3, 1, 0, 2] {
            let progress = accept(&mut cursor, edge, phase).unwrap();
            if phase != FleetIpcPhase::Armed || edge != 2 {
                assert_eq!(
                    progress,
                    FleetIpcCursorProgress::WavePending {
                        step: ScheduleStep(1)
                    }
                );
            }
        }
    }
    assert_eq!(cursor.state(), FleetIpcAttemptState::Complete);
}

#[test]
fn future_wave_poisoned_and_next_wave_opens_only_after_current_wave_arms() {
    let view = two_wave_view();
    assert_eq!(view.spans().len(), 2);

    let mut future = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(
        accept(&mut future, 1, FleetIpcPhase::Published).unwrap_err(),
        FleetIpcCursorError::WrongWave {
            edge: 1,
            active_step: ScheduleStep(1),
        }
    );
    assert_eq!(future.state(), FleetIpcAttemptState::Poisoned);

    let mut cursor = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    for phase in [
        FleetIpcPhase::Published,
        FleetIpcPhase::Consumed,
        FleetIpcPhase::Reclaimed,
    ] {
        accept(&mut cursor, 0, phase).unwrap();
    }
    assert_eq!(
        accept(&mut cursor, 0, FleetIpcPhase::Armed).unwrap(),
        FleetIpcCursorProgress::WaveComplete {
            completed: ScheduleStep(1),
            next: ScheduleStep(3),
        }
    );
    assert_eq!(cursor.active_wave_edges(), [1]);
}

#[test]
fn stale_duplicate_unknown_and_overflow_fail_closed() {
    let view = view();

    let mut future = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert!(matches!(
        future.accept_phase(0, FleetIpcPhase::Consumed, PROOF_GENERATION + 1),
        Err(FleetIpcCursorError::OutOfOrderPhase {
            expected: Some(FleetIpcPhase::Published),
            actual: FleetIpcPhase::Consumed,
            ..
        })
    ));
    assert_eq!(future.state(), FleetIpcAttemptState::Poisoned);

    let mut duplicate = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    accept(&mut duplicate, 0, FleetIpcPhase::Published).unwrap();
    assert!(matches!(
        accept(&mut duplicate, 0, FleetIpcPhase::Published),
        Err(FleetIpcCursorError::OutOfOrderPhase {
            expected: Some(FleetIpcPhase::Consumed),
            actual: FleetIpcPhase::Published,
            ..
        })
    ));
    assert_eq!(duplicate.state(), FleetIpcAttemptState::Poisoned);

    let mut stale = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert!(matches!(
        stale.accept_phase(0, FleetIpcPhase::Published, PROOF_GENERATION - 1),
        Err(FleetIpcCursorError::GenerationMismatch { .. })
    ));
    assert_eq!(stale.state(), FleetIpcAttemptState::Poisoned);
    assert_eq!(
        stale
            .accept_phase(0, FleetIpcPhase::Published, PROOF_GENERATION)
            .unwrap_err(),
        FleetIpcCursorError::AttemptPoisoned
    );

    let mut unknown = IpcScheduleCursor::new(&view, PROOF_GENERATION).unwrap();
    assert_eq!(
        accept(&mut unknown, 9, FleetIpcPhase::Published).unwrap_err(),
        FleetIpcCursorError::UnknownEdge(9)
    );
    assert_eq!(unknown.state(), FleetIpcAttemptState::Poisoned);

    assert!(matches!(
        IpcScheduleCursor::new(&view, u64::MAX),
        Err(FleetIpcCursorError::GenerationOverflow)
    ));
}
