"""Provider-free regression checks for lease reconciliation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from . import common as c
from . import lifecycle, runtime


def fake_pod(args, offer, lease_name: str) -> c.api.PodInfo:
    return c.api.PodInfo(
        id="pod-test",
        name=lease_name,
        status="RUNNING",
        cost_per_hr=0.5,
        gpu=offer["display_name"],
        dc=args.volume_dc,
        vcpu=args.min_vcpu,
        mem_gb=args.min_mem_gb,
        ssh_host="127.0.0.1",
        ssh_port=22,
        raw={"gpuCount": 1},
    )


def failed_open_cleanup_self_test() -> None:
    old_state = c.STATE
    saved = (c.api.list_pods, runtime._cancel_watchdog)
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            reservation = {
                "lease_name": "stwo-test-lease",
                "phase": "creating",
                "watchdog_pid": 12345,
            }
            c._write_state(reservation)
            canceled = []
            runtime._cancel_watchdog = lambda state: canceled.append(state["watchdog_pid"])
            c.api.list_pods = lambda: []
            lifecycle._cleanup_failed_open(reservation, reservation["lease_name"], None)
            assert canceled == [12345] and not c.STATE.exists()

            c._write_state(reservation)

            def ambiguous_reconciliation():
                raise RuntimeError("provider observation unavailable")

            c.api.list_pods = ambiguous_reconciliation
            lifecycle._cleanup_failed_open(reservation, reservation["lease_name"], None)
            state = c._read_state()
            assert state["phase"] == "creating"
            assert state["reconciliation_error"] == "provider observation unavailable"
            assert canceled == [12345]
    finally:
        c.api.list_pods, runtime._cancel_watchdog = saved
        c.STATE = old_state
