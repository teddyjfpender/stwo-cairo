"""Provider-free checks for the nonformal lease-local controller root."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from . import common as c
from . import lease_local_root as root


def _legacy_fixture(base: Path) -> tuple[Path, Path]:
    target = base / "gpu-lab"
    target.mkdir()
    target.chmod(0o777)
    marker = target / root.MARKER
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
            root.LEGACY_TARGET_PREFLIGHT,
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
    result = _preflight(target, volume_id)
    assert result.returncode != 0, f"accepted hostile {label}"


def _evidence(pod_id: str = "pod-test") -> str:
    values = (
        ("LABCTL_LEASE_LOCAL_ROOT", root.SCHEMA),
        ("LABCTL_LEASE_LOCAL_TARGET", root.TARGET),
        ("LABCTL_LEASE_LOCAL_BACKING", root.backing_root(pod_id)),
        ("LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID", "7"),
        ("LABCTL_LEASE_LOCAL_CONTAINER_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID", "41"),
        ("LABCTL_LEASE_LOCAL_SOURCE_MOUNT_ID", "7"),
        ("LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID", "52"),
        ("LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE", "0:41"),
        ("LABCTL_LEASE_LOCAL_SOURCE_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_TARGET_DEVICE", "0:7"),
        ("LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE", root.PERSISTENCE_SCOPE),
        ("LABCTL_LEASE_LOCAL_QUALIFICATION", "0"),
    )
    return "\n".join(f"{key}={value}" for key, value in values)


def lease_local_root_self_test() -> None:
    compile(root.LEGACY_TARGET_PREFLIGHT, "lease-local-root-preflight", "exec")
    generated = root.command("pod-test", "volume-test")
    subprocess.run(["bash", "-n"], input=generated, text=True, check=True)
    assert "mount --bind \"$BACKING\" \"$TARGET\"" in generated
    assert generated.index("python3 - ") < generated.index("mount --bind")
    assert "test \"$(stat -c '%u:%g:%a' \"$TARGET\")\" = 0:0:755" in generated
    assert "test \"$TARGET_DEVICE\" = \"$SOURCE_DEVICE\"" in generated
    assert "test \"$SOURCE_DEVICE\" = \"$CONTAINER_DEVICE\"" in generated
    assert "test \"$SOURCE_DEVICE\" != \"$WORKSPACE_DEVICE\"" in generated
    assert "test \"$TARGET_MOUNT_ID\" != \"$WORKSPACE_MOUNT_ID\"" in generated
    assert "test ! -e \"$TARGET/NETWORK_VOLUME_ID\"" in generated
    assert "chmod" not in generated and "fchmod" not in generated

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

    with tempfile.TemporaryDirectory() as directory:
        target, marker = _legacy_fixture(Path(directory))
        marker.unlink()
        _reject(target, "absent marker")

    for value in ("", "../pod", "/pod", "pod id", "pod/child"):
        try:
            root.backing_root(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe pod id: {value!r}")
        try:
            root.command("pod-test", value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe volume id: {value!r}")

    result = root._parse(_evidence(), "pod-test")
    assert result["persistence_scope"] == root.PERSISTENCE_SCOPE
    assert result["qualification"] is False
    assert result["workspace_device"] != result["source_device"]
    assert result["source_device"] == result["target_device"]
    for changed in (
        _evidence().replace("LABCTL_LEASE_LOCAL_QUALIFICATION=0",
                            "LABCTL_LEASE_LOCAL_QUALIFICATION=1"),
        _evidence().replace("LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=52",
                            "LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=41"),
        _evidence().replace("LABCTL_LEASE_LOCAL_TARGET_DEVICE=0:7",
                            "LABCTL_LEASE_LOCAL_TARGET_DEVICE=0:8"),
        _evidence().replace("LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=0:7",
                            "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=0:8"),
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
        assert calls[0][2] == 30 and "mount --bind" in calls[0][1]
        c.ssh_capture = lambda *_a, **_kw: (1, "mount denied")
        try:
            root.install(c.Endpoint("host", 22), "pod-test", "volume-test")
        except RuntimeError as error:
            assert "mount denied" in str(error)
        else:
            raise AssertionError("accepted failed lease-local mount")
    finally:
        c.ssh_capture = saved
