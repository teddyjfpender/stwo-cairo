"""Provider-free regression checks for lease reconciliation."""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from pathlib import Path

from . import common as c
from . import lifecycle, provider, runtime


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


def ssh_preflight_self_test() -> None:
    class MissingIdentity:
        def __iter__(self):
            raise ValueError("no exact SSH private key")

    args = argparse.Namespace(
        confirm=None,
        gpu="4090",
        image="registry.example/stwo:lab@sha256:" + "a" * 64,
        idle_min=30,
        max_total_usd=2.0,
        max_usd_hr=1.0,
        min_mem_gb=32,
        min_vcpu=8,
        name="stwo-test",
        ready_timeout=600.0,
        ttl_hours=4.0,
        volume_dc="EU-1",
        volume_id="volume-test",
    )
    offer = {
        "display_name": "NVIDIA GeForce RTX 4090",
        "gpu_type_id": c.GPU_IDS["4090"],
        "usd_hr": 0.5,
    }
    volume = {
        "data_center_id": "EU-1",
        "id": "volume-test",
        "name": "test-volume",
        "size_gb": 100,
        "source": "test",
    }
    plan = lifecycle._launch_plan(args, offer, volume)
    args.confirm = c._token("OPEN", plan)
    old_state, old_opts = c.STATE, c.SSH_OPTS
    saved = provider._secure_offer, provider._network_volume_attestation
    create = provider._create_pod_once
    calls = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            c.SSH_OPTS = MissingIdentity()
            provider._secure_offer = lambda _gpu: offer
            provider._network_volume_attestation = lambda *_args: volume
            provider._create_pod_once = lambda **_kwargs: calls.append("create")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lifecycle.cmd_open(args)
            except ValueError as error:
                assert "SSH private key" in str(error)
            else:
                raise AssertionError("confirmed open accepted no exact SSH identity")
            assert calls == [] and not c.STATE.exists()
    finally:
        c.STATE, c.SSH_OPTS = old_state, old_opts
        provider._secure_offer, provider._network_volume_attestation = saved
        provider._create_pod_once = create
