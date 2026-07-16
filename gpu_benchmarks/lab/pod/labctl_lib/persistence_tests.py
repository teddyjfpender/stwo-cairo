"""Hostile CPU-only checks for local-output persistence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import common as c
from . import persistence
from . import provider
from . import runtime


def _run(local: Path, workspace: Path, *, freeze: bool = False,
         reason: str = "test") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", persistence.PERSIST_WORKER, "pod-test", "volume-test",
         str(local), str(workspace), "1" if freeze else "0", reason],
        capture_output=True, text=True, check=False,
    )


def _marker(local: Path, *, volume_id: str = "volume-test") -> None:
    document = persistence.marker_document(
        "pod-test", volume_id, "overlay overlay /", "nfs nfs /workspace"
    )
    document["local_root"] = str(local.resolve())
    (local / "LOCAL_ROOT.json").write_text(persistence.canonical_marker(document))


def persistence_self_test() -> None:
    guard = runtime._guard_command("pod-test", "volume-test", 60, 300)
    marker_worker = guard.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(marker_worker, "labctl-local-root-marker", "exec")
    for name in ("ttl", "idle"):
        script = guard.split(f"cat > /usr/local/bin/stwo-lab-{name} <<'LAB{name.upper()}'\n", 1)[1]
        script = script.split(f"\nLAB{name.upper()}\n", 1)[0]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        local, workspace = root / "local", root / "workspace"
        local.mkdir()
        workspace.mkdir()
        _marker(local)
        records = local / "records"
        run = local / "build" / "gpu-lab-sm86" / "runs" / "case"
        profiles = run / "profiles"
        records.mkdir()
        profiles.mkdir(parents=True)
        (records / "accept.txt").write_text("accepted\n")
        (run / "loop.json").write_text("loop\n")
        (run / "result.json").write_text("result\n")
        (profiles / "timeline.nsys-rep").write_text("profile\n")

        first = _run(local, workspace)
        assert first.returncode == 0, first.stderr
        parsed = persistence.parse(first.stdout, sealed=False)
        assert parsed["entries"] == 5
        manifest = Path(parsed["manifest"])
        assert manifest.is_file() and manifest.is_relative_to(workspace.resolve())
        document = json.loads(manifest.read_text())
        assert {item["category"] for item in document["entries"]} == {
            "records", "results", "profiles",
        }
        for item in document["entries"]:
            durable = workspace / item["persistent_path"]
            assert durable.is_file() and durable.name == item["sha256"]

        closed = _run(local, workspace, freeze=True, reason="explicit-close")
        assert closed.returncode == 0, closed.stderr
        sealed = persistence.parse(closed.stdout, sealed=True)
        assert sealed["entries"] == 6
        lease = workspace / "gpu-lab" / "leases" / "pod-test"
        assert (lease / "SEAL.sha256").is_file()
        assert (local / "SEALED").read_text().startswith("sealed ")
        assert _run(local, workspace).returncode != 0

        try:
            persistence.parse("LABCTL_PERSIST_ENTRIES=1\n", sealed=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted incomplete persistence response")

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        local, workspace = root / "local", root / "workspace"
        local.mkdir()
        workspace.mkdir()
        _marker(local, volume_id="wrong")
        (local / "records").mkdir()
        (local / "records" / "record").write_text("evidence")
        assert _run(local, workspace).returncode != 0
        _marker(local)
        (local / "records" / "link").symlink_to(local / "records" / "record")
        assert _run(local, workspace).returncode != 0
        (local / "records" / "link").unlink()
        good = _run(local, workspace)
        assert good.returncode == 0
        item = json.loads(Path(persistence.parse(good.stdout, sealed=False)["manifest"])
                          .read_text())["entries"][0]
        durable = workspace / item["persistent_path"]
        os.chmod(durable, 0o600)
        durable.write_text("mutated")
        assert _run(local, workspace).returncode != 0

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        local, workspace = root / "local", root / "workspace"
        local.mkdir()
        workspace.mkdir()
        _marker(local)
        assert _run(local, workspace).returncode != 0
    _termination_gate_self_test()


def _termination_gate_self_test() -> None:
    pod = c.api.PodInfo(
        id="pod-test", name="lease", status="RUNNING", cost_per_hr=0.5,
        gpu="GPU", dc="EU-1", vcpu=8, mem_gb=32, ssh_host="127.0.0.1",
        ssh_port=22, raw={"gpuCount": 1},
    )
    state = {
        "phase": "open", "pod_id": pod.id, "remote_guard_installed_at": 1.0,
        "volume_id": "volume-test",
    }
    saved = c.STATE, persistence.persist_remote, provider._terminate_pod_once, c.ledger.append
    try:
        with tempfile.TemporaryDirectory() as name:
            c.STATE = Path(name) / "state.json"
            terminated = []
            provider._terminate_pod_once = terminated.append
            c.ledger.append = lambda *_args, **_kwargs: None

            def fail(*_args, **_kwargs):
                raise RuntimeError("hostile incomplete persist")

            persistence.persist_remote = fail
            try:
                runtime._terminate_state_pod(state, pod, reason="test-failure")
            except RuntimeError:
                pass
            else:
                raise AssertionError("termination accepted failed persistence")
            assert terminated == [] and c._read_state()["persistence_error"]
            no_ssh = c.api.PodInfo(**{**vars(pod), "ssh_host": None})
            try:
                runtime._terminate_state_pod(state, no_ssh, reason="test-no-ssh")
            except RuntimeError:
                pass
            else:
                raise AssertionError("termination accepted a running unobservable pod")
            assert terminated == []

            persistence.persist_remote = lambda *_args, **_kwargs: (
                {"manifest": "/workspace/manifest", "manifest_sha256": "a" * 64,
                 "entries": 1, "sealed": True}, "",
            )
            runtime._terminate_state_pod(state, pod, reason="test-success")
            assert terminated == [pod.id]
            assert c._read_state()["phase"] == "terminated"
    finally:
        c.STATE, persistence.persist_remote, provider._terminate_pod_once, c.ledger.append = saved
