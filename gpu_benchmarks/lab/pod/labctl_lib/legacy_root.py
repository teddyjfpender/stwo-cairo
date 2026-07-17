"""One-shot authority-preserving migration for the bootstrap volume root."""

from __future__ import annotations

import os
import shlex

from . import common as c


SCHEMA = "stwo.gpu-lab.legacy-root-migration.v1"
ROOT = "/workspace/gpu-lab"
MARKER = "NETWORK_VOLUME_ID"
CHANGED = "root-0777-to-0755"
UNCHANGED = "root-0755-unchanged"

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
    require_root(before, {0o755, 0o777})
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
            or stat.S_IMODE(marker_before.st_mode) != 0o600
            or marker_before.st_nlink != 1
            or marker_before.st_size != len(expected_marker)
        ):
            raise RuntimeError(
                f"legacy network-volume marker mismatch: {marker_identity}"
            )
        if os.read(marker_fd, len(expected_marker) + 1) != expected_marker:
            raise RuntimeError("legacy network-volume marker content mismatch")
        marker_path = os.stat(
            "NETWORK_VOLUME_ID", dir_fd=root_fd, follow_symlinks=False
        )
        if identity(marker_path) != marker_identity:
            raise RuntimeError("legacy network-volume marker raced during open")

        changed = stat.S_IMODE(before.st_mode) == 0o777
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
        if identity(marker_after) != marker_identity:
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
    + ("root-0777-to-0755" if changed else "root-0755-unchanged")
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
    if rc or marker not in {prefix + CHANGED, prefix + UNCHANGED}:
        raise RuntimeError(f"legacy gpu-lab root migration failed: {output[-300:]}")
    result = marker.removeprefix(prefix)
    return {
        "changed": result == CHANGED,
        "from_mode": "0777" if result == CHANGED else "0755",
        "owner": "0:0",
        "schema_version": SCHEMA,
        "to_mode": "0755",
        "volume_id": volume_id,
    }
