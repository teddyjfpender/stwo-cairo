"""One-shot authority-preserving migration for the bootstrap volume root."""

from __future__ import annotations

import os
import shlex

from . import common as c


SCHEMA = "stwo.gpu-lab.legacy-root-migration.v1"
ROOT = "/workspace/gpu-lab"
MARKER = "NETWORK_VOLUME_ID"
CHANGED = "root-0777-marker-0666-to-root-0755-marker-0600"
RESUMED_MARKER = "root-0700-marker-0666-to-root-0755-marker-0600"
RESUMED_ROOT = "root-0700-marker-0600-to-root-0755-marker-0600"
UNCHANGED = "root-0755-marker-0600-unchanged"

WORKER = r"""
import os
import stat
import sys

root, volume_id = sys.argv[1:3]
expected_uid, expected_gid = map(int, sys.argv[3:5])
expected_marker = (volume_id + "\n").encode()

def identity(info):
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid,
        stat.S_IMODE(info.st_mode), info.st_nlink,
    )

def require_root(info, modes):
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != expected_uid
        or info.st_gid != expected_gid
        or stat.S_IMODE(info.st_mode) not in modes
    ):
        raise RuntimeError(f"legacy gpu-lab root mismatch: {identity(info)}")

root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
marker_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
root_fd = os.open(root, root_flags)
try:
    before = os.fstat(root_fd)
    require_root(before, {0o700, 0o755, 0o777})
    path_before = os.stat(root, follow_symlinks=False)
    if identity(path_before) != identity(before):
        raise RuntimeError("legacy gpu-lab root raced during open")

    marker_fd = os.open("NETWORK_VOLUME_ID", marker_flags, dir_fd=root_fd)
    try:
        marker_before = os.fstat(marker_fd)
        marker_identity = identity(marker_before)
        if (
            not stat.S_ISREG(marker_before.st_mode)
            or marker_before.st_uid != expected_uid
            or marker_before.st_gid != expected_gid
            or stat.S_IMODE(marker_before.st_mode) not in {0o600, 0o666}
            or marker_before.st_nlink != 1
            or marker_before.st_size != len(expected_marker)
        ):
            raise RuntimeError(
                f"legacy network-volume marker mismatch: {marker_identity}"
            )
        marker_path = os.stat(
            "NETWORK_VOLUME_ID", dir_fd=root_fd, follow_symlinks=False
        )
        if identity(marker_path) != marker_identity:
            raise RuntimeError("legacy network-volume marker raced during open")

        initial = (
            stat.S_IMODE(before.st_mode),
            stat.S_IMODE(marker_before.st_mode),
        )
        admitted = {
            (0o755, 0o600),
            (0o777, 0o666),
            (0o700, 0o666),
            (0o700, 0o600),
        }
        if initial not in admitted:
            raise RuntimeError(f"legacy root/marker mode pair mismatch: {initial}")

        changed = initial != (0o755, 0o600)
        if changed and initial[0] == 0o777:
            os.fchmod(root_fd, 0o700)
            os.fsync(root_fd)
        if changed and initial[1] == 0o666:
            os.fchmod(marker_fd, 0o600)
            os.fsync(marker_fd)

        locked_root = os.fstat(root_fd)
        require_root(locked_root, {0o700} if changed else {0o755})
        locked_marker = os.fstat(marker_fd)
        if (
            not stat.S_ISREG(locked_marker.st_mode)
            or locked_marker.st_uid != expected_uid
            or locked_marker.st_gid != expected_gid
            or stat.S_IMODE(locked_marker.st_mode) != 0o600
            or locked_marker.st_nlink != 1
            or (locked_marker.st_dev, locked_marker.st_ino)
            != (marker_before.st_dev, marker_before.st_ino)
        ):
            raise RuntimeError(
                f"locked network-volume marker mismatch: {identity(locked_marker)}"
            )
        locked_marker_path = os.stat(
            "NETWORK_VOLUME_ID", dir_fd=root_fd, follow_symlinks=False
        )
        if identity(locked_marker_path) != identity(locked_marker):
            raise RuntimeError("legacy network-volume marker changed while locking")
        os.lseek(marker_fd, 0, os.SEEK_SET)
        if os.read(marker_fd, len(expected_marker) + 1) != expected_marker:
            raise RuntimeError("legacy network-volume marker content mismatch")

        if changed:
            os.fchmod(root_fd, 0o755)
            os.fsync(root_fd)

        after = os.fstat(root_fd)
        require_root(after, {0o755})
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("legacy gpu-lab root descriptor changed")
        path_after = os.stat(root, follow_symlinks=False)
        if identity(path_after) != identity(after):
            raise RuntimeError("legacy gpu-lab root path changed")
        marker_after = os.stat(
            "NETWORK_VOLUME_ID", dir_fd=root_fd, follow_symlinks=False
        )
        if identity(marker_after) != identity(locked_marker):
            raise RuntimeError("legacy network-volume marker changed")
        os.lseek(marker_fd, 0, os.SEEK_SET)
        if os.read(marker_fd, len(expected_marker) + 1) != expected_marker:
            raise RuntimeError("legacy network-volume marker content changed")
    finally:
        os.close(marker_fd)
finally:
    os.close(root_fd)

print(
    "LABCTL_LEGACY_ROOT_MIGRATION="
    + {
        (0o777, 0o666):
            "root-0777-marker-0666-to-root-0755-marker-0600",
        (0o700, 0o666):
            "root-0700-marker-0666-to-root-0755-marker-0600",
        (0o700, 0o600):
            "root-0700-marker-0600-to-root-0755-marker-0600",
        (0o755, 0o600): "root-0755-marker-0600-unchanged",
    }[initial]
)
"""


def command(
    volume_id: str,
    *,
    root: str = ROOT,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> str:
    """Return the descriptor-anchored migration command."""
    arguments = " ".join(
        shlex.quote(str(value))
        for value in (root, volume_id, expected_uid, expected_gid)
    )
    mount_gate = ""
    if root == ROOT:
        mount_gate = "mountpoint -q /workspace\ntest ! -L /workspace\n"
    return f"set -eu\n{mount_gate}python3 - {arguments} <<'PY'\n{WORKER}\nPY\n"


def migrate(ep: c.Endpoint, volume_id: str) -> dict[str, object]:
    """Run the bootstrap-only migration and return its evidence record."""
    rc, output = c.ssh_capture(ep, command(volume_id), timeout=30)
    marker = output.strip()
    prefix = "LABCTL_LEGACY_ROOT_MIGRATION="
    results = {
        CHANGED: ("0777", "0666", False),
        RESUMED_MARKER: ("0700", "0666", True),
        RESUMED_ROOT: ("0700", "0600", True),
        UNCHANGED: ("0755", "0600", False),
    }
    if (
        rc
        or not marker.startswith(prefix)
        or marker.removeprefix(prefix) not in results
    ):
        raise RuntimeError(f"legacy gpu-lab root migration failed: {output[-300:]}")
    result = marker.removeprefix(prefix)
    from_mode, marker_from_mode, resumed = results[result]
    return {
        "changed": result != UNCHANGED,
        "from_mode": from_mode,
        "marker_from_mode": marker_from_mode,
        "marker_to_mode": "0600",
        "owner": "0:0",
        "resumed": resumed,
        "schema_version": SCHEMA,
        "to_mode": "0755",
        "volume_id": volume_id,
    }
