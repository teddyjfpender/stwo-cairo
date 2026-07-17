"""Offline regression checks for every fail-closed control boundary."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import acceptance_tests, bootstrap_profile_tests, generation_tests, lifecycle
from . import lifecycle_tests, persistence_tests
from . import provider, runtime, sync
from . import common as c
def _rejected(call, label: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError, BlockingIOError):
        return
    raise AssertionError(f"accepted invalid {label}")
def _open_args(image: str) -> argparse.Namespace:
    return argparse.Namespace(
        confirm=None,
        gpu="4090",
        image=image,
        idle_min=30,
        max_total_usd=2.0,
        max_usd_hr=1.0,
        min_mem_gb=32,
        min_vcpu=8,
        name="stwo-test",
        ready_timeout=60.0,
        ttl_hours=1.0,
        volume_dc="EU-1",
        volume_id="volume-test",
    )
def _check_inputs_and_tokens(args, offer) -> tuple[dict, str]:
    c._validate_open_args(args)
    _rejected(
        lambda: c._validate_open_args(
            argparse.Namespace(**{**vars(args), "image": "repo@sha256:not-a-digest"})
        ),
        "image digest",
    )
    _rejected(
        lambda: c._validate_open_args(
            argparse.Namespace(
                **{
                    **vars(args),
                    "image": "repo@sha256:" + "b" * 64 + "@sha256:" + "a" * 64,
                }
            )
        ),
        "multiple image digests",
    )
    volume = {
        "data_center_id": args.volume_dc,
        "id": args.volume_id,
        "name": "test-volume",
        "size_gb": 100,
        "source": "RunPod REST /v1/networkvolumes/{id}",
    }
    plan = lifecycle._launch_plan(args, offer, volume)
    canonical = c._token("OPEN", plan)
    assert canonical == c._token("OPEN", dict(reversed(list(plan.items()))))
    for key, value in plan.items():
        changed = dict(plan)
        if isinstance(value, bool):
            changed[key] = not value
        elif isinstance(value, (int, float)):
            changed[key] = value + 1
        else:
            changed[key] = str(value) + "-changed"
        assert c._token("OPEN", changed) != canonical, key
    return plan, canonical
def _check_budgets() -> None:
    c._check_budget(0.5, 2, 1, 1)
    for bad in (
        (1.01, 1, 1, 2),
        (0.5, 3, 1, 1),
        (0.5, 7, 1, 9),
        (math.nan, 1, 1, 2),
        (0.5, math.nan, 1, 2),
        (0.5, 1, math.nan, 2),
        (0.5, 1, 1, math.nan),
        (math.inf, 1, 1, 2),
        (0.5, 1, math.inf, 2),
    ):
        _rejected(lambda bad=bad: c._check_budget(*bad), f"budget {bad}")
def _check_provider(offer) -> None:
    old_gql = c.api.gql
    try:
        c.api.gql = lambda *_args, **_kw: {
            "gpuTypes": [
                {
                    "id": c.GPU_IDS["4090"],
                    "displayName": offer["display_name"],
                    "secureCloud": True,
                    "securePrice": "0.5",
                }
            ]
        }
        assert provider._secure_offer(c.GPU_IDS["4090"])["usd_hr"] == 0.5
        c.api.gql = lambda *_args, **_kw: {
            "gpuTypes": [
                {
                    "id": c.GPU_IDS["4090"],
                    "secureCloud": False,
                    "securePrice": "0.1",
                }
            ]
        }
        _rejected(lambda: provider._secure_offer(c.GPU_IDS["4090"]), "community-only offer")
    finally:
        c.api.gql = old_gql
    old_rest = provider._rest_get
    try:
        provider._rest_get = lambda resource: {
            "dataCenterId": "EU-1",
            "id": "volume-test",
            "name": "test-volume",
            "size": 100,
        }
        assert provider._network_volume_attestation("volume-test", "EU-1")["size_gb"] == 100
        _rejected(
            lambda: provider._network_volume_attestation("volume-test", "US-1"),
            "network-volume data center",
        )
        def flat_volume_rest(resource):
            if resource == "pods/pod-test":
                return {"id": "pod-test", "networkVolumeId": "volume-test"}
            return {
                "dataCenterId": "EU-1",
                "id": "volume-test",
                "name": "test-volume",
                "size": 100,
            }

        provider._rest_get = flat_volume_rest
        assert provider._attest_pod_volume("pod-test", "volume-test", "EU-1")[
            "pod_id"
        ] == "pod-test"
        _rejected(
            lambda: provider._attest_pod_volume("pod-test", "wrong-volume", "EU-1"),
            "created-pod volume",
        )
        provider._rest_get = lambda resource: {
            "id": "pod-test",
            "networkVolume": {"id": "volume-test", "dataCenterId": "EU-1"},
        }
        assert provider._attest_pod_volume("pod-test", "volume-test", "EU-1")[
            "pod_id"
        ] == "pod-test"
    finally:
        provider._rest_get = old_rest
    old_gql, old_get_pod, old_list_pods = c.api.gql, c.api.get_pod, c.api.list_pods
    try:
        c.api.gql = lambda *_args, **_kw: {"podTerminate": None}
        c.api.get_pod = lambda _pod_id: None
        c.api.list_pods = lambda: []
        provider._terminate_pod_once("pod-gone")

        c.api.get_pod = lambda _pod_id: object()
        _rejected(
            lambda: provider._terminate_pod_once("pod-live"),
            "unconfirmed termination",
        )

        def lost_mutation(*_args, **_kwargs):
            raise c.api.ApiError("lost response")

        c.api.gql = lost_mutation
        observations = iter((None, None))
        c.api.get_pod = lambda _pod_id: next(observations)
        c.api.list_pods = lambda: []
        provider._terminate_pod_once("pod-lost-response")
    finally:
        c.api.gql = old_gql
        c.api.get_pod = old_get_pod
        c.api.list_pods = old_list_pods
def _check_generated_commands(valid_image: str) -> None:
    guard = runtime._guard_command("pod-test", "volume-test", 60, 300)
    assert "SEAL.sha256" in guard and "sleep 60" in guard
    assert "RUNPOD_API_KEY" not in guard and "mountpoint -q /workspace" in guard
    assert "NETWORK_VOLUME_ID" in guard and "kill -TERM 1" in guard
    assert "stwo-lab-ttl" in guard and "stwo-lab-idle" in guard
    assert "IDLE_S=300" in guard and c.HEARTBEAT_PATH in guard
    assert "LOCAL_ROOT.json" in guard and "GPU_LAB_LOCAL_ROOT" in guard
    assert "DEV_LAYOUT.json" in guard
    assert 'install -d -m 0710 -o root -g dev "$LOCAL_ROOT"' in guard
    assert 'install -d -m 0700 -o dev -g dev "$LOCAL_ROOT/build"' in guard
    assert '"build_uid": 1000' in guard and '"fixtures_uid": 1000' in guard
    assert 'marker: (0, 0, 0o444, 1)' in guard
    assert "refusing unsealed TTL exit" in guard and "LABCTL_PERSIST_SHA256" in guard
    assert runtime._elapsed_spend({"created_at": 0, "usd_hr": 2}, 7200) == 4
    subprocess.run(["bash", "-n"], input=guard, text=True, check=True)

    old_capture, old_run = c.ssh_capture, c.ssh_run
    try:
        c.ssh_capture = lambda *_a, **_kw: (0, "LABCTL_IDLE age=361 busy=0")
        assert runtime._remote_is_idle(c.Endpoint("host", 22), 300)
        c.ssh_capture = lambda *_a, **_kw: (0, "LABCTL_IDLE age=999 busy=1")
        assert not runtime._remote_is_idle(c.Endpoint("host", 22), 300)
        c.ssh_run = lambda *_a, **_kw: 0
        runtime._touch_heartbeat(c.Endpoint("host", 22))
    finally:
        c.ssh_capture, c.ssh_run = old_capture, old_run
    acceptance_tests.acceptance_self_test(valid_image)
def _check_tree_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory) / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "labctl@test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.name", "labctl"], check=True
        )
        tracked = repo / "tracked.txt"
        tracked.write_text("base\n")
        subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
        assert sync._local_tree_identity(repo)["entries"] == []
        tracked.write_text("changed\n")
        (repo / "untracked.txt").write_text("new\n")
        dirty = sync._local_tree_identity(repo)
        assert {item["path"] for item in dirty["entries"]} == {
            "tracked.txt",
            "untracked.txt",
        }
        tampered = dict(dirty)
        tampered["identity_sha256"] = "0" * 64
        _rejected(
            lambda: sync._parse_tree_identity(json.dumps(tampered), label="tampered"),
            "tree identity digest",
        )


def _check_state_machine(args, offer, plan, canonical) -> None:
    old_state = c.STATE
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            with c._lease_lock():
                _rejected(
                    lambda: c._lease_lock(blocking=False).__enter__(),
                    "concurrent lease lock",
                )
                c._write_state({"phase": "creating", "volume_id": "volume-test"})
                assert c._read_state()["phase"] == "creating"
            c.STATE.unlink()
            with c._operation_lock():
                with c._lease_lock(blocking=False):
                    pass

            lease_name = f"{plan['name_prefix']}-{canonical[-8:].lower()}"
            fake_pod = lifecycle_tests.fake_pod(args, offer, lease_name)
            saved = (
                provider._attest_pod_volume,
                provider._create_pod_once,
                provider._network_volume_attestation,
                provider._secure_offer,
                runtime._spawn_watchdog,
                c.ssh_run,
                c.wait_ready,
                c.api.list_pods,
                c.ledger.append,
            )
            try:
                provider._secure_offer = lambda _gpu: offer
                provider._network_volume_attestation = lambda *_a: plan["volume_rest_attestation"]
                provider._attest_pod_volume = lambda *_a: {
                    "data_center_id": args.volume_dc,
                    "id": args.volume_id,
                    "pod_id": fake_pod.id,
                    "source": "RunPod REST /v1/pods/{id}.networkVolume",
                }
                provider._create_pod_once = lambda **_kw: fake_pod
                runtime._spawn_watchdog = lambda _state: 12345
                c.wait_ready = lambda *_a, **_kw: fake_pod
                c.ssh_run = lambda *_a, **_kw: 0
                list_calls = iter(([], [fake_pod]))
                c.api.list_pods = lambda: next(list_calls)
                c.ledger.append = lambda *_a, **_kw: None
                fake_args = argparse.Namespace(**{**vars(args), "confirm": canonical})
                with contextlib.redirect_stdout(io.StringIO()):
                    assert lifecycle.cmd_open(fake_args) == 0
                state = c._read_state()
                assert state["phase"] == "open" and state["pod_id"] == "pod-test"
                assert state["watchdog_pid"] == 12345
                assert state["remote_guard_installed_at"] > state["created_at"]
            finally:
                (
                    provider._attest_pod_volume,
                    provider._create_pod_once,
                    provider._network_volume_attestation,
                    provider._secure_offer,
                    runtime._spawn_watchdog,
                    c.ssh_run,
                    c.wait_ready,
                    c.api.list_pods,
                    c.ledger.append,
                ) = saved
                c.STATE.unlink(missing_ok=True)

            _check_watch_and_close(fake_pod, lease_name, plan)
    finally:
        c.STATE = old_state


def _check_watch_and_close(fake_pod, lease_name: str, plan: dict) -> None:
    c._write_state(
        {
            "created_at": 1.0,
            "expires_at": 2.0,
            "lease_name": lease_name,
            "phase": "creating",
            "plan": plan,
            "usd_hr": 0.5,
            "volume_id": "volume-test",
        }
    )
    saved = (c.api.get_pod, c.api.list_pods, provider._terminate_pod_once, c.ledger.append)
    terminated = []
    try:
        c.api.get_pod = lambda _pod_id: fake_pod
        c.api.list_pods = lambda: [fake_pod]
        provider._terminate_pod_once = terminated.append
        c.ledger.append = lambda *_a, **_kw: None
        assert lifecycle.cmd_watch(
            argparse.Namespace(created_at=1.0, expires_at=2.0, lease_name=lease_name)
        ) == 0
        assert terminated == ["pod-test"]
        assert c._read_state()["phase"] == "terminated"
    finally:
        c.api.get_pod, c.api.list_pods, provider._terminate_pod_once, c.ledger.append = saved
        c.STATE.unlink(missing_ok=True)

    c._write_state(
        {
            "created_at": 1.0,
            "expires_at": time.time() + 3600,
            "idle_minutes": 30,
            "lease_name": lease_name,
            "phase": "open",
            "pod_id": "pod-test",
            "remote_guard_installed_at": time.time(),
            "usd_hr": 0.5,
            "volume_id": "volume-test",
        }
    )
    exited = c.api.PodInfo(**{**vars(fake_pod), "status": "EXITED", "ssh_host": None})
    saved = (c.api.get_pod, c.api.list_pods, provider._terminate_pod_once,
             c.ledger.append, runtime._remote_is_idle)
    terminated = []
    try:
        c.api.get_pod = lambda _pod_id: fake_pod
        runtime._remote_is_idle = lambda *_a: True
        assert lifecycle._watch_stop_reason(c._read_state()).endswith("-idle")
        c.api.get_pod = lambda _pod_id: None
        c.api.list_pods = lambda: [exited]
        provider._terminate_pod_once = terminated.append
        c.ledger.append = lambda *_a, **_kw: None
        token = c._token(
            "CLOSE",
            {"lease_name": lease_name, "pod_id": "pod-test", "volume_id": "volume-test"},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 1
        assert terminated == ["pod-test"] and not c.STATE.exists()
    finally:
        (c.api.get_pod, c.api.list_pods, provider._terminate_pod_once,
         c.ledger.append, runtime._remote_is_idle) = saved
        c.STATE.unlink(missing_ok=True)
def cmd_self_test(_args) -> int:
    def forbidden_remote(*_args, **_kwargs):
        raise AssertionError("self-test attempted provider/SSH transport")

    args = _open_args("registry.example/stwo:lab@sha256:" + "a" * 64)
    offer = {
        "display_name": "NVIDIA GeForce RTX 4090",
        "gpu_type_id": c.GPU_IDS["4090"],
        "usd_hr": 0.5,
    }
    remote_functions = (
        c.api.gql,
        c.api.get_pod,
        c.api.list_pods,
        c.ssh_capture,
        c.ssh_run,
        c.rsync,
        c.wait_ready,
        provider._rest_get,
    )
    original_ssh_opts = c.SSH_OPTS
    c.SSH_OPTS = [
        "-i", "/offline/labctl-private-key",
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
    ]
    try:
        c.api.gql = forbidden_remote
        c.api.get_pod = forbidden_remote
        c.api.list_pods = forbidden_remote
        c.ssh_capture = forbidden_remote
        c.ssh_run = forbidden_remote
        c.rsync = forbidden_remote
        c.wait_ready = forbidden_remote
        provider._rest_get = forbidden_remote
        plan, canonical = _check_inputs_and_tokens(args, offer)
        _check_budgets()
        _check_provider(offer)
        _check_generated_commands(args.image)
        bootstrap_profile_tests.bootstrap_profile_self_test()
        persistence_tests.persistence_self_test()
        _check_tree_identity()
        generation_tests.generation_self_test()
        lifecycle_tests.failed_open_cleanup_self_test()
        lifecycle_tests.ssh_preflight_self_test()
        _check_state_machine(args, offer, plan, canonical)
    finally:
        c.SSH_OPTS = original_ssh_opts
        (
            c.api.gql,
            c.api.get_pod,
            c.api.list_pods,
            c.ssh_capture,
            c.ssh_run,
            c.rsync,
            c.wait_ready,
            provider._rest_get,
        ) = remote_functions
    print("labctl self-test: PASS (no network/API/SSH calls)")
    return 0
