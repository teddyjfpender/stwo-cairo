"""Provider-free hostile checks for the temporary bootstrap profile."""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import acceptance
from . import bootstrap_profile as profile
from . import common as c
from . import lifecycle
from . import provider
from . import runtime


def _rejected(call, label: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"bootstrap profile accepted invalid {label}")


def _args(**changes) -> argparse.Namespace:
    args = argparse.Namespace(bootstrap_profile=profile.NAME, confirm=None)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def _configuration_checks() -> argparse.Namespace:
    args = _args()
    profile.configure(args, {})
    assert args.gpu == "4090"
    assert args.image == profile.IMAGE
    assert args.volume_id == "2kpphx92fr" and args.volume_dc == "EU-RO-1"
    assert (args.ttl_hours, args.idle_min) == (6.0, 30)
    assert (args.max_usd_hr, args.max_total_usd) == (0.8, 4.8)
    assert (args.min_vcpu, args.min_mem_gb) == (8, 32)
    expected = {
        "formal": False,
        "image_digest": profile.IMAGE_DIGEST,
        "image_digest_authority": "requested-reference-not-runtime-attested",
        "lane": "consumer-development",
        "profile": profile.NAME,
        "qualification_eligible": False,
    }
    assert profile.metadata(args) == expected

    exact_mutations = {
        "gpu": "5090",
        "image": profile.IMAGE + "-changed",
        "volume_id": "other",
        "volume_dc": "US-1",
        "name": "other",
    }
    for field, value in exact_mutations.items():
        _rejected(
            lambda field=field, value=value: profile.configure(
                _args(**{field: value}), {}
            ),
            field,
        )
    for field, value in {
        "ttl_hours": 6.01,
        "idle_min": 31,
        "max_usd_hr": 0.81,
        "max_total_usd": 4.81,
        "min_vcpu": 7,
        "min_mem_gb": 31,
    }.items():
        _rejected(
            lambda field=field, value=value: profile.configure(
                _args(**{field: value}), {}
            ),
            field,
        )
    _rejected(
        lambda: profile.configure(
            _args(bootstrap_profile="consumer-3090-bootstrap"), {}
        ),
        "profile name",
    )

    ordinary = argparse.Namespace(bootstrap_profile=None, confirm=None)
    environment = {
        "LABCTL_IMAGE": "registry/formal@sha256:" + "a" * 64,
        "LABCTL_VOLUME_ID": "formal-volume",
        "LABCTL_VOLUME_DC": "EU-FORMAL-1",
    }
    profile.configure(ordinary, environment)
    assert ordinary.gpu == "4090" and ordinary.ttl_hours == 4.0
    assert ordinary.max_usd_hr == 1.0 and ordinary.max_total_usd == 4.0
    assert ordinary.image == environment["LABCTL_IMAGE"]
    assert profile.metadata(ordinary) == {}
    return args


def _command_checks(args: argparse.Namespace) -> None:
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
    command = profile._bootstrap_command(public_key)
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "getent group 1000" in command and "getent passwd 1000" in command
    assert "1000:1000:600:1" in command and "passwd -S dev" in command
    assert "99-stwo-consumer-bootstrap.conf" in command
    assert "sshd -t" in command and "authenticationmethods" in command
    assert "0:0:755" in command
    assert "/root/.ssh/authorized_keys" not in command

    state = {
        **profile.metadata(args),
        "bootstrap_key_sha256": "a" * 64,
        "image": profile.IMAGE,
        "pod_id": "pod-test",
        "volume_dc": profile.VOLUME_DC,
        "volume_id": profile.VOLUME_ID,
    }
    record, path, digest = profile._record_command(state)
    subprocess.run(["bash", "-n"], input=record, text=True, check=True)
    assert path.endswith(f"/{profile.NAME}.json")
    assert len(digest) == 64 and "0:0:400:1" in record
    payload = profile._record_payload(state)
    assert payload["formal"] is False and payload["lane"] == profile.LANE
    assert payload["qualification_eligible"] is False
    assert payload["image_digest_authority"].startswith("requested-reference")

    guard = runtime._guard_command("pod-test", "volume-test", 60, 300)
    subprocess.run(["bash", "-n"], input=guard, text=True, check=True)
    assert "test ! -L /workspace" in guard
    assert "/workspace/gpu-lab/NETWORK_VOLUME_ID" in guard
    assert "identity != (0, 0, 0o600, 1)" in guard
    assert "os.O_NOFOLLOW" in guard and "stat.S_ISREG" in guard
    assert "tempfile.mkstemp" in guard and ".ACTIVE_ROOT.tmp" not in guard

    _rejected(
        lambda: runtime._require_open_state({"phase": "open", "pod_id": "pod-test"}),
        "open state without remote guard",
    )
    runtime._require_open_state(
        {"phase": "open", "pod_id": "pod-test", "remote_guard_installed_at": 1}
    )


def _selected_key_checks() -> None:
    old_options, old_run = c.SSH_OPTS, profile.subprocess.run
    try:
        c.SSH_OPTS = ["-i", "/selected/key", "-o", "IdentitiesOnly=yes"]
        profile.subprocess.run = lambda argv, **_kw: SimpleNamespace(
            returncode=0,
            stdout="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest ignored-comment\n",
        )
        assert profile._public_key() == "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
        c.SSH_OPTS = ["-i", "/one", "-i", "/two"]
        _rejected(profile._public_key, "ambiguous selected key")
        c.SSH_OPTS = ["-i", "/selected/key"]
        profile.subprocess.run = lambda argv, **_kw: SimpleNamespace(
            returncode=0, stdout="not-a-key\n"
        )
        _rejected(profile._public_key, "invalid derived public key")
    finally:
        c.SSH_OPTS, profile.subprocess.run = old_options, old_run


def _ordering_checks(args: argparse.Namespace) -> None:
    old_state = c.STATE
    saved = (
        profile.bootstrap_dev,
        profile.verify_ssh,
        profile.persist_record,
        c.ssh_run,
    )
    calls = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            profile.bootstrap_dev = lambda _ep: calls.append("bootstrap") or "a" * 64
            profile.verify_ssh = lambda _ep, _key: calls.append("fresh-root+dev")
            c.ssh_run = lambda *_a, **_kw: calls.append("guard") or 0
            profile.persist_record = (
                lambda *_a: calls.append("record")
                or ("/workspace/profile.json", "b" * 64)
            )
            state = {
                **profile.metadata(args),
                "phase": "bootstrapping",
                "pod_id": "pod-test",
                "volume_id": profile.VOLUME_ID,
            }
            lifecycle._install_remote_controls(
                state, args, c.Endpoint("host", 22), 3600
            )
            assert calls == ["bootstrap", "fresh-root+dev", "guard", "record"]
            assert c._read_state()["phase"] == "bootstrapping"

            calls.clear()
            c.ssh_run = lambda *_a, **_kw: calls.append("guard-failed") or 1
            _rejected(
                lambda: lifecycle._install_remote_controls(
                    state, args, c.Endpoint("host", 22), 3600
                ),
                "failed guard",
            )
            assert calls == ["bootstrap", "fresh-root+dev", "guard-failed"]
    finally:
        (
            profile.bootstrap_dev,
            profile.verify_ssh,
            profile.persist_record,
            c.ssh_run,
        ) = saved
        c.STATE = old_state


def _nonformal_gate_check() -> None:
    saved = runtime._active
    try:
        runtime._active = lambda: (
            {"formal": False},
            object(),
            c.Endpoint("host", 22),
        )
        _rejected(
            lambda: acceptance.cmd_accept(argparse.Namespace(profile=True)),
            "formal acceptance",
        )
    finally:
        runtime._active = saved


def _bootstrapping_close_check() -> None:
    old_state = c.STATE
    saved = (
        c.api.get_pod,
        c.api.list_pods,
        provider._terminate_pod_once,
        runtime._persist_for_termination,
        c.ledger.append,
    )
    pod = c.api.PodInfo(
        id="pod-test", name="bootstrap", status="RUNNING", cost_per_hr=0.69,
        gpu="NVIDIA GeForce RTX 4090", dc=profile.VOLUME_DC, vcpu=8,
        mem_gb=32, ssh_host="host", ssh_port=22, raw={"gpuCount": 1},
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            state = {
                "expires_at": 9e12,
                "lease_name": pod.name,
                "phase": "bootstrapping",
                "pod_id": pod.id,
                "volume_id": profile.VOLUME_ID,
            }
            c._write_state(state)
            c.api.get_pod = lambda _pod_id: pod
            c.api.list_pods = lambda: [pod]
            terminated = []
            provider._terminate_pod_once = terminated.append
            runtime._persist_for_termination = lambda *_a, **_kw: (_ for _ in ()).throw(
                AssertionError("bootstrapping close attempted formal persistence")
            )
            c.ledger.append = lambda *_a, **_kw: None
            token = c._token(
                "CLOSE",
                {"lease_name": pod.name, "pod_id": pod.id,
                 "volume_id": profile.VOLUME_ID},
            )
            with contextlib.redirect_stdout(io.StringIO()):
                assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 0
            assert terminated == [pod.id] and not c.STATE.exists()
    finally:
        (
            c.api.get_pod,
            c.api.list_pods,
            provider._terminate_pod_once,
            runtime._persist_for_termination,
            c.ledger.append,
        ) = saved
        c.STATE = old_state


def bootstrap_profile_self_test() -> None:
    args = _configuration_checks()
    _selected_key_checks()
    _command_checks(args)
    _ordering_checks(args)
    _nonformal_gate_check()
    _bootstrapping_close_check()
