"""Provider-free checks for the nonformal lease-local controller root."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from . import common as c
from . import lease_local_root as root


BOOT_ID = "11111111-1111-1111-1111-111111111111"


def _legacy_fixture(base: Path) -> tuple[Path, Path]:
    target = base / "gpu-lab"
    target.mkdir()
    target.chmod(0o777)
    marker = target / root.LEGACY_MARKER
    marker.write_text("volume-test\n")
    marker.chmod(0o666)
    return target, marker


def _preflight(
    target: Path, volume_id: str = "volume-test"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            "-c",
            root.LEGACY_VOLUME_PREFLIGHT,
            str(target),
            volume_id,
            str(os.getuid()),
            str(os.getgid()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _reject(target: Path, label: str, volume_id: str = "volume-test") -> None:
    assert _preflight(target, volume_id).returncode != 0, f"accepted hostile {label}"


def _evidence(pod_id: str = "pod-test") -> str:
    values = (
        ("LABCTL_LEASE_LOCAL_ROOT", root.SCHEMA),
        ("LABCTL_LEASE_LOCAL_TARGET", root.TARGET),
        ("LABCTL_LEASE_LOCAL_QUARANTINE", root.QUARANTINED_ROOT),
        ("LABCTL_LEASE_LOCAL_MARKER", f"{root.TARGET}/{root.MARKER}"),
        ("LABCTL_LEASE_LOCAL_BOOT_ID", BOOT_ID),
        ("LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID", "7"),
        ("LABCTL_LEASE_LOCAL_CONTAINER_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID", "7"),
        ("LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_QUARANTINE_MOUNT_ID", "41"),
        ("LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE", "0:41"),
        ("LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID", "7"),
        ("LABCTL_LEASE_LOCAL_TARGET_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE", root.PERSISTENCE_SCOPE),
        ("LABCTL_LEASE_LOCAL_QUALIFICATION", "0"),
    )
    return "\n".join(f"{key}={value}" for key, value in values)


def lease_local_root_self_test() -> None:
    compile(root.LEGACY_VOLUME_PREFLIGHT, "quarantined-volume-preflight", "exec")
    compile(root.LOCAL_MARKER_WORKER, "lease-local-marker", "exec")
    generated = root.command("pod-test", "volume-test")
    attestation = root.attestation_command(
        "pod-test", "volume-test", BOOT_ID, guards=True
    )
    for script in (generated, attestation):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert root.PROVIDER_MOUNT in generated
    assert root.QUARANTINED_ROOT in generated
    assert "findmnt -nr -o ID --target /" in generated
    assert "findmnt -nr -o MAJ:MIN --target /" in generated
    assert "findmnt -n -o" not in generated
    assert "findmnt -n -o" not in attestation
    assert "mount --bind" not in generated
    assert f"install -d -m 0755 -o root -g root {root.TARGET}" in generated
    assert 'test "$TARGET_MOUNT_ID:$TARGET_DEVICE" = ' in generated
    assert 'test "$QUARANTINE_DEVICE" != "$CONTAINER_DEVICE"' in generated
    assert 'test "$LEGACY_MOUNT_ID:$LEGACY_DEVICE" = ' in generated
    assert 'test "$WORKSPACE_BEFORE" = "$CONTAINER_BEFORE"' in generated
    assert "LEASE_LOCAL_ROOT.json" in generated
    assert "/proc/sys/kernel/random/boot_id" in generated
    assert "0:0:600:1" in attestation
    assert "/proc/$PID/cmdline" in attestation

    payload = json.loads(root.marker_payload("pod-test", "volume-test"))
    assert payload == {
        "persistence_scope": root.PERSISTENCE_SCOPE,
        "pod_id": "pod-test",
        "provider_mount": root.PROVIDER_MOUNT,
        "qualification": False,
        "quarantined_root": root.QUARANTINED_ROOT,
        "schema_version": root.SCHEMA,
        "target": root.TARGET,
        "volume_id": "volume-test",
    }

    with tempfile.TemporaryDirectory() as directory:
        target, marker = _legacy_fixture(Path(directory))
        assert _preflight(target).returncode == 0
        marker.write_text("other-volume\n")
        _reject(target, "wrong marker content")
        marker.write_text("volume-test\n")
        marker.chmod(0o600)
        _reject(target, "secured legacy marker")
        marker.chmod(0o666)
        target.chmod(0o755)
        _reject(target, "repaired legacy root")

    with tempfile.TemporaryDirectory() as directory:
        target, marker = _legacy_fixture(Path(directory))
        marker.unlink()
        marker.symlink_to(Path(directory) / "elsewhere")
        _reject(target, "marker symlink")

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        target, marker = _legacy_fixture(base)
        os.link(marker, target / "marker-hardlink")
        _reject(target, "hardlinked marker")
        target_link = base / "target-link"
        target_link.symlink_to(target, target_is_directory=True)
        _reject(target_link, "target symlink")

    for value in ("", "../pod", "/pod", "pod id", "pod/child"):
        for call in (
            lambda value=value: root.command(value, "volume-test"),
            lambda value=value: root.command("pod-test", value),
        ):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted unsafe lease identity: {value!r}")
    try:
        root.attestation_command("pod-test", "volume-test", "bad", guards=True)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unsafe boot id")

    result = root._parse(_evidence(), "pod-test")
    assert result["persistence_scope"] == root.PERSISTENCE_SCOPE
    assert result["qualification"] is False
    assert result["boot_id"] == BOOT_ID
    assert result["container_device"] == result["target_device"]
    assert result["container_device"] != result["quarantine_device"]
    padded = _evidence().replace(
        "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=0:7",
        "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE= 0:7",
    )
    try:
        root._parse(padded, "pod-test")
    except RuntimeError as error:
        assert "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=' 0:7'" in str(error)
    else:
        raise AssertionError("accepted padded device evidence")
    for changed in (
        _evidence().replace("LABCTL_LEASE_LOCAL_QUALIFICATION=0",
                            "LABCTL_LEASE_LOCAL_QUALIFICATION=1"),
        _evidence().replace("LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=7",
                            "LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=41"),
        _evidence().replace("LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE=0:41",
                            "LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE=0:7"),
        _evidence().replace("LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID=7",
                            "LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID= 7"),
        _evidence().replace("LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=0:7",
                            "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=0:7 "),
        _evidence().replace(BOOT_ID, "invalid"),
        _evidence() + "\nunexpected warning",
        _evidence() + "\nLABCTL_LEASE_LOCAL_EXTRA=1",
    ):
        try:
            root._parse(changed, "pod-test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted hostile lease-local evidence")

    saved = c.ssh_capture
    try:
        calls = []
        c.ssh_capture = lambda ep, command, *, timeout: (
            calls.append((ep, command, timeout)) or (0, _evidence())
        )
        installed = root.install(
            c.Endpoint("host", 22), "pod-test", "volume-test"
        )
        assert installed["volume_id"] == "volume-test"
        assert calls[0][2] == 30 and "mount --bind" not in calls[0][1]
        c.ssh_capture = lambda *_a, **_kw: (
            0, "LABCTL_LEASE_LOCAL_ATTESTED=pod-test"
        )
        root.attest(
            c.Endpoint("host", 22), "pod-test", "volume-test", BOOT_ID
        )
        c.ssh_capture = lambda *_a, **_kw: (1, "guard missing")
        try:
            root.attest(
                c.Endpoint("host", 22), "pod-test", "volume-test", BOOT_ID
            )
        except RuntimeError as error:
            assert "guard missing" in str(error)
        else:
            raise AssertionError("accepted missing lease-local guard")
    finally:
        c.ssh_capture = saved
