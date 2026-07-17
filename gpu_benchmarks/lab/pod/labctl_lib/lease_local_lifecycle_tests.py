"""Lease-local lifecycle and source-routing regression checks."""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import bootstrap_profile as profile
from . import common as c
from . import lease_local_root
from . import lifecycle
from . import persistence
from . import provider
from . import runtime
from . import sync


StateFactory = Callable[[argparse.Namespace], dict]


def _guard_checks(args: argparse.Namespace, state_factory: StateFactory) -> None:
    guard = runtime._guard_command(state_factory(args), 60, 300)
    subprocess.run(["bash", "-n"], input=guard, text=True, check=True)
    assert lease_local_root.PROVIDER_MOUNT in guard
    assert lease_local_root.QUARANTINED_ROOT in guard
    assert lease_local_root.MARKER in guard
    assert 'test "$WORKSPACE_MOUNT" = "$LOCAL_MOUNT"' in guard
    assert 'test "$WORKSPACE_DEVICE" = "$LOCAL_DEVICE"' in guard
    assert "mount --bind" not in guard
    assert "LABCTL_PERSIST_SHA256" not in guard
    assert "SEAL.sha256" not in guard
    assert "lease-local outputs discarded by remote TTL guard" in guard
    assert "lease-local outputs discarded by remote idle guard" in guard
    assert 'install -m 0600 -o root -g root /dev/null "$HB"' in guard
    assert "chmod 0600 /var/run/stwo-lab-ttl.pid" in guard


def _runtime_checks(args: argparse.Namespace, state_factory: StateFactory) -> None:
    created_at = time.time()
    state = {
        **state_factory(args),
        "created_at": created_at,
        "expires_at": created_at + 3600,
        "idle_minutes": 30,
        "max_total_usd": 4.8,
        "max_usd_hr": 0.8,
        "phase": "open",
        "remote_guard_installed_at": 2.0,
        "usd_hr": 0.69,
    }
    pod = c.api.PodInfo(
        id="pod-test", name="bootstrap", status="RUNNING", cost_per_hr=0.69,
        gpu="NVIDIA GeForce RTX 4090", dc=profile.VOLUME_DC, vcpu=8,
        mem_gb=32, ssh_host="host", ssh_port=22, raw={"gpuCount": 1},
    )
    old_state = c.STATE
    saved = (
        c.api.get_pod,
        lease_local_root.attest,
        provider._terminate_pod_once,
        c.ledger.append,
        c.ssh_capture,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            c._write_state(state)
            c.api.get_pod = lambda _pod_id: pod
            lease_local_root.attest = lambda *_a: (_ for _ in ()).throw(
                RuntimeError("guard process missing")
            )
            terminated = []
            provider._terminate_pod_once = terminated.append
            c.ledger.append = lambda *_a, **_kw: None
            try:
                runtime._active()
            except RuntimeError as error:
                assert "attestation failed; terminated" in str(error)
            else:
                raise AssertionError("accepted restarted lease-local controller")
            assert terminated == [pod.id]
            closed = c._read_state()
            assert closed["phase"] == "terminated"
            assert closed["termination_reason"] == (
                "lease-local-runtime-attestation-failed"
            )
            assert closed["discarded_lease_local_outputs"] is True

            calls = []
            c.ssh_capture = lambda _ep, command, **_kw: (
                calls.append(command)
                or (0, "LABCTL_LEASE_LOCAL_ATTESTED=pod-test")
            )
            sync._verify_remote_volume(c.Endpoint("host", 22), state)
            assert sync._seed_specs(state)[0][2] == (
                f"{lease_local_root.PROVIDER_MOUNT}/src/stwo"
            )
            assert lease_local_root.QUARANTINED_ROOT in calls[0]
            assert "/workspace/gpu-lab/NETWORK_VOLUME_ID" not in calls[0]

            calls.clear()
            formal = {"pod_id": "pod-test", "volume_id": "volume-test"}
            c.ssh_capture = lambda _ep, command, **_kw: (
                calls.append(command) or (0, "/dev/nfs nfs /workspace")
            )
            sync._verify_remote_volume(c.Endpoint("host", 22), formal)
            assert sync._seed_specs(formal)[0][2] == "/workspace/src/stwo"
            assert "/workspace/gpu-lab/NETWORK_VOLUME_ID" in calls[0]
    finally:
        (
            c.api.get_pod,
            lease_local_root.attest,
            provider._terminate_pod_once,
            c.ledger.append,
            c.ssh_capture,
        ) = saved
        c.STATE = old_state


def _close_checks(args: argparse.Namespace, state_factory: StateFactory) -> None:
    old_state = c.STATE
    saved = (
        c.api.get_pod,
        c.api.list_pods,
        provider._terminate_pod_once,
        persistence.persist_remote,
        c.ledger.append,
    )
    pod = c.api.PodInfo(
        id="pod-test", name="bootstrap", status="RUNNING", cost_per_hr=0.69,
        gpu="NVIDIA GeForce RTX 4090", dc=profile.VOLUME_DC, vcpu=8,
        mem_gb=32, ssh_host=None, ssh_port=None, raw={"gpuCount": 1},
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            base = {
                **state_factory(args),
                "expires_at": 9e12,
                "lease_name": pod.name,
                "phase": "open",
                "remote_guard_installed_at": 1.0,
            }
            token = c._token(
                "CLOSE",
                {
                    "lease_name": pod.name,
                    "pod_id": pod.id,
                    "volume_id": profile.VOLUME_ID,
                },
            )
            persistence.persist_remote = lambda *_a, **_kw: (_ for _ in ()).throw(
                AssertionError("lease-local close attempted formal persistence")
            )
            c.ledger.append = lambda *_a, **_kw: None

            c._write_state(base)
            c.api.get_pod = lambda _pod_id: None
            c.api.list_pods = lambda: []
            with contextlib.redirect_stdout(io.StringIO()):
                assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 0
            assert not c.STATE.exists()

            c._write_state(base)
            c.api.get_pod = lambda _pod_id: pod
            c.api.list_pods = lambda: [pod]
            terminated = []
            provider._terminate_pod_once = terminated.append
            with contextlib.redirect_stdout(io.StringIO()):
                assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 0
            assert terminated == [pod.id] and not c.STATE.exists()

            exited = c.api.PodInfo(**{**vars(pod), "status": "EXITED"})
            c._write_state(base)
            c.api.get_pod = lambda _pod_id: exited
            c.api.list_pods = lambda: [exited]
            terminated.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 0
            assert terminated == [pod.id] and not c.STATE.exists()

            other = c.api.PodInfo(**{**vars(pod), "id": "pod-other"})
            c._write_state(base)
            c.api.get_pod = lambda _pod_id: pod
            c.api.list_pods = lambda: [pod, other]
            try:
                lifecycle.cmd_close(argparse.Namespace(confirm=token))
            except RuntimeError:
                pass
            else:
                raise AssertionError("accepted ambiguous lease-local close")
            assert c.STATE.exists()
    finally:
        (
            c.api.get_pod,
            c.api.list_pods,
            provider._terminate_pod_once,
            persistence.persist_remote,
            c.ledger.append,
        ) = saved
        c.STATE.unlink(missing_ok=True)
        c.STATE = old_state


def lease_local_lifecycle_self_test(
    args: argparse.Namespace, state_factory: StateFactory
) -> None:
    _guard_checks(args, state_factory)
    _runtime_checks(args, state_factory)
    _close_checks(args, state_factory)
