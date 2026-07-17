"""Container-local controller root for the nonformal consumer bootstrap lane."""

from __future__ import annotations

import base64
import json
import re
import shlex

from . import common as c


SCHEMA = "stwo.gpu-lab.lease-local-root.v2"
PERSISTENCE_SCOPE = "lease-local-container-disk"
PROVIDER_MOUNT = "/runpod-volume"
QUARANTINED_ROOT = f"{PROVIDER_MOUNT}/gpu-lab"
TARGET = "/workspace/gpu-lab"
MARKER = "LEASE_LOCAL_ROOT.json"
LEGACY_MARKER = "NETWORK_VOLUME_ID"
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

LEGACY_VOLUME_PREFLIGHT = r"""
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
        raise RuntimeError(f"quarantined legacy root mismatch: {identity(root_info)}")

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
            raise RuntimeError(
                f"quarantined legacy marker mismatch: {identity(marker_info)}"
            )
        if os.read(marker_fd, len(expected_marker) + 1) != expected_marker:
            raise RuntimeError("quarantined legacy marker content mismatch")
    finally:
        os.close(marker_fd)
finally:
    os.close(root_fd)
"""

LOCAL_MARKER_WORKER = r"""
import base64
import json
import os
import stat
import sys

target, encoded, boot_id, action = sys.argv[1:5]
document = json.loads(base64.b64decode(encoded, validate=True))
document["boot_id"] = boot_id
expected = (
    json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
).encode()
root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    root = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root.st_mode)
        or (root.st_uid, root.st_gid, stat.S_IMODE(root.st_mode)) != (0, 0, 0o755)
    ):
        raise RuntimeError("lease-local controller root identity mismatch")
    try:
        marker_fd = os.open(
            "LEASE_LOCAL_ROOT.json",
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
    except FileNotFoundError:
        if action != "create":
            raise
        marker_fd = os.open(
            "LEASE_LOCAL_ROOT.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=root_fd,
        )
        with os.fdopen(marker_fd, "wb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
        os.fsync(root_fd)
        marker_fd = os.open(
            "LEASE_LOCAL_ROOT.json",
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
    try:
        marker = os.fstat(marker_fd)
        path = os.stat(
            "LEASE_LOCAL_ROOT.json", dir_fd=root_fd, follow_symlinks=False
        )
        identity = (
            marker.st_dev, marker.st_ino, marker.st_uid, marker.st_gid,
            stat.S_IMODE(marker.st_mode), marker.st_nlink,
        )
        path_identity = (
            path.st_dev, path.st_ino, path.st_uid, path.st_gid,
            stat.S_IMODE(path.st_mode), path.st_nlink,
        )
        if (
            not stat.S_ISREG(marker.st_mode)
            or identity != path_identity
            or (marker.st_uid, marker.st_gid, stat.S_IMODE(marker.st_mode),
                marker.st_nlink) != (0, 0, 0o400, 1)
            or os.read(marker_fd, len(expected) + 1) != expected
        ):
            raise RuntimeError("lease-local controller marker mismatch")
    finally:
        os.close(marker_fd)
finally:
    os.close(root_fd)
"""


def marker_payload(pod_id: str, volume_id: str) -> bytes:
    if not c.ID_RE.fullmatch(pod_id) or not c.ID_RE.fullmatch(volume_id):
        raise ValueError("unsafe lease-local marker identity")
    document = {
        "persistence_scope": PERSISTENCE_SCOPE,
        "pod_id": pod_id,
        "provider_mount": PROVIDER_MOUNT,
        "qualification": False,
        "quarantined_root": QUARANTINED_ROOT,
        "schema_version": SCHEMA,
        "target": TARGET,
        "volume_id": volume_id,
    }
    return (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _checks(
    pod_id: str,
    volume_id: str,
    *,
    expected_boot_id: str | None = None,
    guards: bool,
    emit: bool,
) -> str:
    encoded = base64.b64encode(marker_payload(pod_id, volume_id)).decode()
    arguments = " ".join(
        shlex.quote(value) for value in (QUARANTINED_ROOT, volume_id, "0", "0")
    )
    expected_boot = (
        f"test \"$BOOT_ID\" = {shlex.quote(expected_boot_id)}"
        if expected_boot_id is not None
        else ":"
    )
    guard_checks = ""
    if guards:
        guard_checks = f"""
for path in {shlex.quote(c.HEARTBEAT_PATH)} \
  /usr/local/bin/stwo-lab-ttl /usr/local/bin/stwo-lab-idle \
  /var/run/stwo-lab-ttl.pid /var/run/stwo-lab-idle.pid; do
  test ! -L "$path"
  test -f "$path"
done
test "$(stat -c '%u:%g:%a:%h' {shlex.quote(c.HEARTBEAT_PATH)})" = 0:0:600:1
for program in stwo-lab-ttl stwo-lab-idle; do
  test "$(stat -c '%u:%g:%a:%h' "/usr/local/bin/$program")" = 0:0:700:1
  PID_FILE="/var/run/$program.pid"
  test "$(stat -c '%u:%g:%a:%h' "$PID_FILE")" = 0:0:600:1
  PID=$(cat "$PID_FILE")
  case "$PID" in ''|*[!0-9]*) exit 1 ;; esac
  test "$PID" -gt 1
  kill -0 "$PID"
  tr '\\0' '\\n' < "/proc/$PID/cmdline" | grep -Fx "/usr/local/bin/$program" >/dev/null
done
"""
    receipt = (
        f"printf 'LABCTL_LEASE_LOCAL_ATTESTED={pod_id}\\n'"
        if emit
        else ":"
    )
    return f"""
test "$(id -u):$(id -un)" = 0:root
command -v findmnt >/dev/null
command -v mountpoint >/dev/null
test ! -L {shlex.quote(PROVIDER_MOUNT)}
test -d {shlex.quote(PROVIDER_MOUNT)}
mountpoint -q {shlex.quote(PROVIDER_MOUNT)}
python3 - {arguments} <<'PY'
{LEGACY_VOLUME_PREFLIGHT}
PY
test ! -L /workspace
test -d /workspace
test "$(stat -c '%u:%g' /workspace)" = 0:0
WORKSPACE_MODE=$(stat -c '%a' /workspace)
test "$((8#$WORKSPACE_MODE & 8#022))" -eq 0
test "$((8#$WORKSPACE_MODE & 8#700))" -eq "$((8#700))"
test "$((8#$WORKSPACE_MODE & 8#001))" -eq "$((8#001))"
test ! -L {shlex.quote(TARGET)}
test -d {shlex.quote(TARGET)}
test "$(stat -c '%u:%g:%a' {shlex.quote(TARGET)})" = 0:0:755
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id)
python3 - "$BOOT_ID" <<'PY'
import re
import sys
if not re.fullmatch(
    r"[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}",
    sys.argv[1],
):
    raise SystemExit("invalid boot id")
PY
{expected_boot}
python3 - {shlex.quote(TARGET)} {shlex.quote(encoded)} "$BOOT_ID" verify <<'PY'
{LOCAL_MARKER_WORKER}
PY
CONTAINER_MOUNT_ID=$(findmnt -n -o ID --target /)
CONTAINER_DEVICE=$(findmnt -n -o MAJ:MIN --target /)
WORKSPACE_MOUNT_ID=$(findmnt -n -o ID --target /workspace)
WORKSPACE_DEVICE=$(findmnt -n -o MAJ:MIN --target /workspace)
QUARANTINE_MOUNT_ID=$(findmnt -n -o ID --target {shlex.quote(PROVIDER_MOUNT)})
QUARANTINE_DEVICE=$(findmnt -n -o MAJ:MIN --target {shlex.quote(PROVIDER_MOUNT)})
LEGACY_MOUNT_ID=$(findmnt -n -o ID --target {shlex.quote(QUARANTINED_ROOT)})
LEGACY_DEVICE=$(findmnt -n -o MAJ:MIN --target {shlex.quote(QUARANTINED_ROOT)})
TARGET_MOUNT_ID=$(findmnt -n -o ID --target {shlex.quote(TARGET)})
TARGET_DEVICE=$(findmnt -n -o MAJ:MIN --target {shlex.quote(TARGET)})
test "$WORKSPACE_MOUNT_ID:$WORKSPACE_DEVICE" = \
  "$CONTAINER_MOUNT_ID:$CONTAINER_DEVICE"
test "$TARGET_MOUNT_ID:$TARGET_DEVICE" = \
  "$CONTAINER_MOUNT_ID:$CONTAINER_DEVICE"
test "$QUARANTINE_MOUNT_ID" != "$CONTAINER_MOUNT_ID"
test "$QUARANTINE_DEVICE" != "$CONTAINER_DEVICE"
test "$LEGACY_MOUNT_ID:$LEGACY_DEVICE" = \
  "$QUARANTINE_MOUNT_ID:$QUARANTINE_DEVICE"
{guard_checks}
{receipt}
"""


def command(pod_id: str, volume_id: str) -> str:
    """Quarantine the provider volume and create a controller root on lease disk."""
    encoded = base64.b64encode(marker_payload(pod_id, volume_id)).decode()
    preflight_arguments = " ".join(
        shlex.quote(value) for value in (QUARANTINED_ROOT, volume_id, "0", "0")
    )
    marker_arguments = " ".join(
        shlex.quote(value) for value in (TARGET, encoded)
    )
    checks = _checks(pod_id, volume_id, guards=False, emit=False)
    return f"""
set -euE
trap 'rc=$?; trap - ERR; printf "LABCTL_LEASE_LOCAL_ROOT_ERROR_LINE=%s RC=%s\\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
test "$(id -u):$(id -un)" = 0:root
test ! -L {shlex.quote(PROVIDER_MOUNT)}
test -d {shlex.quote(PROVIDER_MOUNT)}
mountpoint -q {shlex.quote(PROVIDER_MOUNT)}
python3 - {preflight_arguments} <<'PY'
{LEGACY_VOLUME_PREFLIGHT}
PY
test ! -L /workspace
test -d /workspace
test "$(stat -c '%u:%g' /workspace)" = 0:0
WORKSPACE_MODE=$(stat -c '%a' /workspace)
test "$((8#$WORKSPACE_MODE & 8#022))" -eq 0
test "$((8#$WORKSPACE_MODE & 8#700))" -eq "$((8#700))"
test "$((8#$WORKSPACE_MODE & 8#001))" -eq "$((8#001))"
CONTAINER_BEFORE=$(findmnt -n -o ID,MAJ:MIN --target /)
WORKSPACE_BEFORE=$(findmnt -n -o ID,MAJ:MIN --target /workspace)
test "$WORKSPACE_BEFORE" = "$CONTAINER_BEFORE"
test ! -L {shlex.quote(TARGET)}
if test -e {shlex.quote(TARGET)}; then
  test -d {shlex.quote(TARGET)}
  test "$(stat -c '%u:%g:%a' {shlex.quote(TARGET)})" = 0:0:755
  test "$(findmnt -n -o ID,MAJ:MIN --target {shlex.quote(TARGET)})" = \
    "$CONTAINER_BEFORE"
else
  install -d -m 0755 -o root -g root {shlex.quote(TARGET)}
fi
test -z "$(find {shlex.quote(TARGET)} -mindepth 1 -maxdepth 1 \
  ! -name {shlex.quote(MARKER)} -print -quit)"
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id)
python3 - {marker_arguments} "$BOOT_ID" create <<'PY'
{LOCAL_MARKER_WORKER}
PY
{checks}
printf 'LABCTL_LEASE_LOCAL_ROOT={SCHEMA}\\n'
printf 'LABCTL_LEASE_LOCAL_TARGET={TARGET}\\n'
printf 'LABCTL_LEASE_LOCAL_QUARANTINE={QUARANTINED_ROOT}\\n'
printf 'LABCTL_LEASE_LOCAL_MARKER={TARGET}/{MARKER}\\n'
printf 'LABCTL_LEASE_LOCAL_BOOT_ID=%s\\n' "$BOOT_ID"
printf 'LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID=%s\\n' "$CONTAINER_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_CONTAINER_DEVICE=%s\\n' "$CONTAINER_DEVICE"
printf 'LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID=%s\\n' "$WORKSPACE_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE=%s\\n' "$WORKSPACE_DEVICE"
printf 'LABCTL_LEASE_LOCAL_QUARANTINE_MOUNT_ID=%s\\n' "$QUARANTINE_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE=%s\\n' "$QUARANTINE_DEVICE"
printf 'LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID=%s\\n' "$TARGET_MOUNT_ID"
printf 'LABCTL_LEASE_LOCAL_TARGET_DEVICE=%s\\n' "$TARGET_DEVICE"
printf 'LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE={PERSISTENCE_SCOPE}\\n'
printf 'LABCTL_LEASE_LOCAL_QUALIFICATION=0\\n'
"""


def attestation_command(
    pod_id: str, volume_id: str, boot_id: str, *, guards: bool
) -> str:
    if not BOOT_ID_RE.fullmatch(boot_id):
        raise ValueError(f"unsafe lease-local boot id: {boot_id!r}")
    return f"""
set -euE
trap 'rc=$?; trap - ERR; printf "LABCTL_LEASE_LOCAL_ATTEST_ERROR_LINE=%s RC=%s\\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
{_checks(pod_id, volume_id, expected_boot_id=boot_id, guards=guards, emit=True)}
"""


def _parse(output: str, pod_id: str) -> dict[str, object]:
    expected_keys = {
        "LABCTL_LEASE_LOCAL_ROOT",
        "LABCTL_LEASE_LOCAL_TARGET",
        "LABCTL_LEASE_LOCAL_QUARANTINE",
        "LABCTL_LEASE_LOCAL_MARKER",
        "LABCTL_LEASE_LOCAL_BOOT_ID",
        "LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_CONTAINER_DEVICE",
        "LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE",
        "LABCTL_LEASE_LOCAL_QUARANTINE_MOUNT_ID",
        "LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE",
        "LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID",
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
        "LABCTL_LEASE_LOCAL_QUARANTINE": QUARANTINED_ROOT,
        "LABCTL_LEASE_LOCAL_MARKER": f"{TARGET}/{MARKER}",
        "LABCTL_LEASE_LOCAL_PERSISTENCE_SCOPE": PERSISTENCE_SCOPE,
        "LABCTL_LEASE_LOCAL_QUALIFICATION": "0",
    }
    if any(values[key] != value for key, value in exact.items()):
        raise RuntimeError("lease-local root evidence did not match the requested lease")
    boot_id = values["LABCTL_LEASE_LOCAL_BOOT_ID"]
    if not BOOT_ID_RE.fullmatch(boot_id):
        raise RuntimeError("lease-local root emitted an invalid boot id")
    mount_keys = tuple(key for key in expected_keys if key.endswith("_MOUNT_ID"))
    device_keys = tuple(key for key in expected_keys if key.endswith("_DEVICE"))
    if any(not re.fullmatch(r"[1-9][0-9]*", values[key]) for key in mount_keys):
        raise RuntimeError("lease-local root emitted an invalid mount id")
    if any(not re.fullmatch(r"[0-9]+:[0-9]+", values[key]) for key in device_keys):
        raise RuntimeError("lease-local root emitted an invalid device identity")
    container_mount = int(values["LABCTL_LEASE_LOCAL_CONTAINER_MOUNT_ID"])
    workspace_mount = int(values["LABCTL_LEASE_LOCAL_WORKSPACE_MOUNT_ID"])
    target_mount = int(values["LABCTL_LEASE_LOCAL_TARGET_MOUNT_ID"])
    quarantine_mount = int(values["LABCTL_LEASE_LOCAL_QUARANTINE_MOUNT_ID"])
    container_device = values["LABCTL_LEASE_LOCAL_CONTAINER_DEVICE"]
    workspace_device = values["LABCTL_LEASE_LOCAL_WORKSPACE_DEVICE"]
    target_device = values["LABCTL_LEASE_LOCAL_TARGET_DEVICE"]
    quarantine_device = values["LABCTL_LEASE_LOCAL_QUARANTINE_DEVICE"]
    if (
        container_mount != workspace_mount
        or container_mount != target_mount
        or quarantine_mount == container_mount
        or container_device != workspace_device
        or container_device != target_device
        or quarantine_device == container_device
    ):
        raise RuntimeError("lease-local root mount authority is inconsistent")
    return {
        "container_device": container_device,
        "container_mount_id": container_mount,
        "boot_id": boot_id,
        "marker": f"{TARGET}/{MARKER}",
        "persistence_scope": PERSISTENCE_SCOPE,
        "pod_id": pod_id,
        "provider_mount": PROVIDER_MOUNT,
        "qualification": False,
        "quarantine_device": quarantine_device,
        "quarantine_mount_id": quarantine_mount,
        "quarantined_root": QUARANTINED_ROOT,
        "schema_version": SCHEMA,
        "target": TARGET,
        "target_device": target_device,
        "target_mount_id": target_mount,
        "volume_id": None,
        "workspace_device": workspace_device,
        "workspace_mount_id": workspace_mount,
    }


def install(ep: c.Endpoint, pod_id: str, volume_id: str) -> dict[str, object]:
    """Install and return authenticated lease-local controller-root evidence."""
    rc, output = c.ssh_capture(ep, command(pod_id, volume_id), timeout=30)
    if rc:
        raise RuntimeError(f"lease-local gpu-lab root failed: {output[-500:]}")
    result = _parse(output, pod_id)
    result["volume_id"] = volume_id
    return result


def attest(ep: c.Endpoint, pod_id: str, volume_id: str, boot_id: str) -> None:
    """Reject a restarted or mutated lease-local controller before remote work."""
    rc, output = c.ssh_capture(
        ep,
        attestation_command(pod_id, volume_id, boot_id, guards=True),
        timeout=30,
    )
    expected = f"LABCTL_LEASE_LOCAL_ATTESTED={pod_id}"
    if rc or output.strip() != expected:
        raise RuntimeError(
            "lease-local controller/guard attestation failed: " + output[-500:]
        )
