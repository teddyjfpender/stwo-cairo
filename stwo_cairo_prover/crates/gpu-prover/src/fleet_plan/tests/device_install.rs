use stwo_backend_cuda::ArenaSlotId;

use super::*;

#[test]
fn worker_storage_geometry_maps_exactly_to_one_device_arena() {
    let plan = compile(fixture()).unwrap();
    let install = plan.worker_install_plan(WorkerId(0)).unwrap();
    let layout = super::super::device_install::arena_layout_for_test(&install).unwrap();

    assert_eq!(
        layout.total_words(),
        install.capacity().slab_bytes / core::mem::size_of::<u32>()
    );
    for storage in install.storages() {
        let slot = layout.slot(ArenaSlotId(storage.storage.0)).unwrap();
        assert_eq!(
            (slot.offset_words, slot.len_words, slot.alignment_words),
            (
                storage.slab_offset_bytes / core::mem::size_of::<u32>(),
                storage.bytes / core::mem::size_of::<u32>(),
                storage.alignment_bytes / core::mem::size_of::<u32>(),
            )
        );
    }
}

#[test]
fn execution_windows_fail_closed_on_storage_or_byte_geometry_drift() {
    let plan = compile(fixture()).unwrap();
    let install = plan.worker_install_plan(WorkerId(0)).unwrap();
    let window = install.executions()[0].executables[0].effects[0].window;
    let (_, offset_words, len_words) =
        super::super::device_install::validate_window_for_test(&install, window).unwrap();
    assert_eq!(
        offset_words * core::mem::size_of::<u32>(),
        window.offset_bytes
    );
    assert_eq!(len_words * core::mem::size_of::<u32>(), window.bytes);

    let mut shifted = window;
    shifted.slab_offset_bytes += core::mem::size_of::<u32>();
    assert_eq!(
        super::super::device_install::validate_window_for_test(&install, shifted),
        Err(FleetWorkerDeviceInstallError::InvalidWindow(window.storage))
    );

    let mut empty = window;
    empty.bytes = 0;
    assert_eq!(
        super::super::device_install::validate_window_for_test(&install, empty),
        Err(FleetWorkerDeviceInstallError::InvalidWindow(window.storage))
    );

    let mut unknown = window;
    unknown.storage = StorageId(u32::MAX);
    assert_eq!(
        super::super::device_install::validate_window_for_test(&install, unknown),
        Err(FleetWorkerDeviceInstallError::UnknownStorage(StorageId(
            u32::MAX
        )))
    );
}
