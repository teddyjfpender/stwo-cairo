"""Lease-local controller root for the nonformal consumer bootstrap lane."""

from __future__ import annotations

import re
import shlex

from . import common as c


SCHEMA = "stwo.gpu-lab.lease-local-root.v1"
PERSISTENCE_SCOPE = "lease-local-container-disk"
TARGET = "/workspace/gpu-lab"
BACKING_PARENT = "/var/lib/stwo-gpu-lab-lease-roots"
MARKER = "NETWORK_VOLUME_ID"

LEGACY_TARGET_PREFLIGHT = r"""
import os
import stat
import sys

target, volume_id = sys.argv[1:3]
expected_uid, expected_gid = map(int, sys.argv[3:5])
expected_marker = (volume_id + "\n").encode()

def identity(info):
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid,
        stat.S_IMODE(info.st_mode), info.st_nlink,
    )

root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    root_info = os.fstat(root_fd)
    root_path = os.stat(target, follow_symlinks=False)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != expected_uid
        or root_info.st_gid != expected_gid
        or stat.S_IMODE(root_info.st_mode) != 0o777
        or identity(root_path) != identity(root_info)
    ):
        raise RuntimeError(f"legacy target mismatch: {identity(root_info)}")

    marker_fd = os.open(
        "NETWORK_VOLUME_ID",
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )
    try:
        marker_info = os.fstat(marker_fd)
        marker_path = os.stat(
            "NETWORK_VOLUME_ID", dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_uid != expected_uid
            or marker_info.st_gid != expected_gid
            or stat.S_IMODE(marker_info.st_mode) != 0o666
            or marker_info.st_nlink != 1
            or marker_info.st_size != len(expected_marker)
            or identity(marker_path) != identity(marker_info)
        ):
            raise RuntimeError(f"legacy marker mismatch: {identity(marker_info)}")
        if os.read(marker_fd, len(expected_marker) + 1) != expected_marker:
            raise RuntimeError("legacy marker content mismatch")
    finally:
        os.close(marker_fd)
finally:
    os.close(root_fd)
"""


def backing_root(pod_id: str) -> str:
    if not c.ID_RE.fullmatch(pod_id):
        raise ValueError(f"unsafe pod id for lease-local root: {pod_id!r}")
    return f"{BACKING_PARENT}/{pod_id}"


def command(pod_id: str, volume_id: str) -> str:
    """Create and attest the bootstrap-only bind mount without repairing the volume."""
    if not c.ID_RE.fullmatch(volume_id):
        raise ValueError(f"unsafe volume id for lease-local root: {volume_id!r}")
    backing = backing_root(pod_id)
    arguments = " ".join(
        shlex.quote(value) for value in (TARGET, volume_id, "0", "0")
    )
    return f"""
set -euE
trap 'rc=$?; trap - ERR; printf "LABCTL_LEASE_LOCAL_ROOT_ERROR_LINE=%s RC=%s\\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
test "$(id -u):$(id -un)" = 0:root
command -v findmnt >/dev/null
command -v mount >/dev/null
command -v mountpoint >/dev/null
TARGET={shlex.quote(TARGET)}
BACKING_PARENT={shlex.quote(BACKING_PARENT)}
BACKING={shlex.quote(backing)}
test ! -L /workspace
test -d /workspace
mountpoint -q /workspace
test ! -L "$TARGET"
test -d "$TARGET"
if mountpoint -q "$TARGET"; then
  printf 'LABCTL_LEASE_LOCAL_ROOT_CONFLICT=target-already-mounted\\n' >&2
  exit 1
fi
python3 - {arguments} <<'PY'
{LEGACY_TARGET_PREFLIGHT}
PY

# The parent lives on the container filesystem and is never a cache authority.
for path in /var /var/lib; do
  test ! -L "$path"
  test -d "$path"
  test "$(stat -c '%u:%g:%a' "$path")" = 0:0:755
done
test ! -L "$BACKING_PARENT"
if test -e "$BACKING_PARENT"; then
  test -d "$BACKING_PARENT"
  test "$(stat -c '%u:%g:%a' "$BACKING_PARENT")" = 0:0:700
else
  mkdir -m 0700 "$BACKING_PARENT"
fi
test ! -L "$BACKING"
test ! -e "$BACKING"
mkdir -m 0755 "$BACKING"
test ! -L "$BACKING"
test -d "$BACKING"
test "$(stat -c '%u:%g:%a' "$BACKING")" = 0:0:755
test -z "$(find "$BACKING" -mindepth 1 -maxdepth 1 -print -quit)"

CONTAINER_MOUNT_ID=$(findmnt -n -o ID --target /)
CONTAINER_DEVICE=$(findmnt -n -o MAJ:MIN --target /)
WORKSPACE_MOUNT_ID=$(findmnt -n -o ID --target /workspace)
WORKSPACE_DEVICE=$(findmnt -n -o MAJ:MIN --target /workspace)
TARGET_BEFORE_MOUNT_ID=$(findmnt -n -o ID --target "$TARGET")
SOURCE_MOUNT_ID=$(findmnt -n -o ID --target "$BACKING")
SOURCE_DEVICE=$(findmnt -n -o MAJ:MIN --target "$BACKING")
SOURCE_IDENTITY=$(stat -c '%d:%i' "$BACKING")
TARGET_BEFORE_IDENTITY=$(stat -c '%d:%i' "$TARGET")
test -n "$WORKSPACE_MOUNT_ID"
test -n "$WORKSPACE_DEVICE"
test "$TARGET_BEFORE_MOUNT_ID" = "$WORKSPACE_MOUNT_ID"
test "$SOURCE_MOUNT_ID" = "$CONTAINER_MOUNT_ID"
test "$SOURCE_DEVICE" = "$CONTAINER_DEVICE"
test "$SOURCE_MOUNT_ID" != "$WORKSPACE_MOUNT_ID"
test "$SOURCE_DEVICE" != "$WORKSPACE_DEVICE"
test "$SOURCE_IDENTITY" != "$TARGET_BEFORE_IDENTITY"

mount --bind "$BACKING" "$TARGET"
mountpoint -q "$TARGET"
test ! -L "$TARGET"
test ! -L "$BACKING"
test "$(stat -c '%u:%g:%a' "$TARGET")" = 0:0:755
test "$(stat -c '%u:%g:%a' "$BACKING")" = 0:0:755
test "$(stat -c '%d:%i' "$TARGET")" = "$SOURCE_IDENTITY"
test ! -e "$TARGET/{MARKER}"
test -z "$(find "$TARGET" -mindepth 1 -maxdepth 1 -print -quit)"

TARGET_MOUNT_ID=$(findmnt -n -o ID --target "$TARGET")
TARGET_DEVICE=$(findmnt -n -o MAJ:MIN --target "$TARGET")
test "$TARGET_MOUNT_ID" != "$WORKSPACE_MOUNT_ID"
test "$TARGET_MOUNT_ID" != "$SOURCE_MOUNT_ID"
test "$TARGET_DEVICE" = "$SOURCE_DEVICE"
test "$(findmnt -n -o ID --target /workspace)" = "$WORKSPACE_MOUNT_ID"
test "$(findmnt -n -o MAJ:MIN --target /workspace)" = "$WORKSPACE_DEVICE"
test "$(findmnt -n -o ID --target "$BACKING")" = "$SOURCE_MOUNT_ID"
test "$(findmnt -n -o MAJ:MIN --target "$BACKING")" = "$SOURCE_DEVICE"

printf 'LABCTL_LEASE_LOCAL_ROOT={SCHEMA}\\n'
printf 'LABCTL_LEASE_LOCAL_TARGET=%s\\n' "$TARGET"
printf 'LABCTL_LEASE_LOCAL_BACKING=%s\\n' "$BACKING"
printf 'LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID=%s\\n' "$CONTAINER_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=%s\\n' "$CONTAINER_DEVICE"
printf 'LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID=%s\\n' "$WORKSPACE_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_SOURCE_MOUNT_ID=%s\\n' "$SOURCE_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=%s\\n' "$TARGET_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE=%s\\n' "$WORKSPACE_DEVICE"
printf 'LABCTL_LEASE_LOCAL_SOURCE_DEVICE=%s\\n' "$SOURCE_DEVICE"
printf 'LABCTL_LEASE_LOCAL_TARGET_DEVICE=%s\\n' "$TARGET_DEVICE"
printf 'LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE={PERSISTENCE_SCOPE}\\n'
printf 'LABCTL_LEASE_LOCAL_QUALIFICATION=0\\n'
"""


def _parse(output: str, pod_id: str) -> dict[str, object]:
    backing = backing_root(pod_id)
    expected_keys = {
        "LABCTL_LEASE_LOCAL_ROOT",
        "LABCTL_LEASE_LOCAL_TARGET",
        "LABCTL_LEASE_LOCAL_BACKING",
        "LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE",
        "LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_SOURCE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE",
        "LABCTL_LEASE_LOCAL_SOURCE_DEVICE",
        "LABCTL_LEASE_LOCAL_TARGET_DEVICE",
        "LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE",
        "LABCTL_LEASE_LOCAL_QUALIFICATION",
    }
    lines = output.splitlines()
    if any("=" not in line for line in lines):
        raise RuntimeError("lease-local root emitted malformed evidence")
    pairs = [line.split("=", 1) for line in lines]
    if len(pairs) != len(expected_keys) or {key for key, _ in pairs} != expected_keys:
        raise RuntimeError("lease-local root emitted incomplete evidence")
    values = dict(pairs)
    exact = {
        "LABCTL_LEASE_LOCAL_ROOT": SCHEMA,
        "LABCTL_LEASE_LOCAL_TARGET": TARGET,
        "LABCTL_LEASE_LOCAL_BACKING": backing,
        "LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE": PERSISTENCE_SCOPE,
        "LABCTL_LEASE_LOCAL_QUALIFICATION": "0",
    }
    if any(values[key] != value for key, value in exact.items()):
        raise RuntimeError("lease-local root evidence did not match the requested lease")
    mount_keys = (
        "LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_SOURCE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID",
    )
    if any(not re.fullmatch(r"[1-9][0-9]*", values[key]) for key in mount_keys):
        raise RuntimeError("lease-local root emitted an invalid mount id")
    container_mount, workspace_mount, source_mount, target_mount = (
        int(values[key]) for key in mount_keys
    )
    if (
        container_mount != source_mount
        or len({workspace_mount, source_mount, target_mount}) != 3
    ):
        raise RuntimeError("lease-local root mount identities are not distinct")
    device_keys = (
        "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE",
        "LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE",
        "LABCTL_LEASE_LOCAL_SOURCE_DEVICE",
        "LABCTL_LEASE_LOCAL_TARGET_DEVICE",
    )
    if any(not re.fullmatch(r"[0-9]+:[0-9]+", values[key]) for key in device_keys):
        raise RuntimeError("lease-local root emitted an invalid device identity")
    container_device, workspace_device, source_device, target_device = (
        values[key] for key in device_keys
    )
    if (
        container_device != source_device
        or workspace_device == source_device
        or source_device != target_device
    ):
        raise RuntimeError("lease-local root device authority is inconsistent")
    return {
        "backing": backing,
        "container_device": container_device,
        "container_mount_id": container_mount,
        "persistence_scope": PERSISTENCE_SCOPE,
        "pod_id": pod_id,
        "qualification": False,
        "schema_version": SCHEMA,
        "source_device": source_device,
        "source_mode": "0755",
        "source_mount_id": source_mount,
        "source_owner": "0:0",
        "target": TARGET,
        "target_device": target_device,
        "target_mount_id": target_mount,
        "workspace_device": workspace_device,
        "workspace_mount_id": workspace_mount,
    }


def install(ep: c.Endpoint, pod_id: str, volume_id: str) -> dict[str, object]:
    """Install and return the authenticated lease-local root evidence."""
    rc, output = c.ssh_capture(ep, command(pod_id, volume_id), timeout=30)
    if rc:
        raise RuntimeError(f"lease-local gpu-lab root failed: {output[-500:]}")
    result = _parse(output, pod_id)
    result["volume_id"] = volume_id
    return result
