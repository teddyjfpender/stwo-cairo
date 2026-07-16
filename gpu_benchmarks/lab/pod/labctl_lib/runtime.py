"""Lease guards, reconciliation, active-state checks, and watchdog support."""

from __future__ import annotations

import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time

from . import common as c
from . import persistence
from . import provider


BUSY_PROCESS_RE = (
    "[g]pu-lab|[c]ompute-sanitizer|[n]cu|[n]sys|[c]argo|[r]ustc|"
    "[c]make|[n]inja|[n]vcc|[p]txas|[r]sync"
)


def _seal_command(pod_id: str, volume_id: str, *, reason: str) -> str:
    return persistence.command(pod_id, volume_id, freeze=True, reason=reason)


def _guard_command(
    pod_id: str, volume_id: str, ttl_seconds: int, idle_seconds: int
) -> str:
    """Install independent credential-free TTL and activity guards."""
    ttl_seal = _seal_command(pod_id, volume_id, reason="remote-ttl-guard")
    idle_seal = _seal_command(pod_id, volume_id, reason="remote-idle-guard")
    local_root = persistence.local_root(pod_id)
    return f"""
set -eu
POD_ID={shlex.quote(pod_id)}
VOLUME_ID={shlex.quote(volume_id)}
mountpoint -q /workspace
export POD_ID VOLUME_ID
LOCAL_ROOT={shlex.quote(local_root)}
DEV_UID=$(id -u dev)
DEV_GID=$(id -g dev)
test "$DEV_UID" = 1000
test "$DEV_GID" = 1000
for path in \
  /tmp/stwo-gpu-lab "$LOCAL_ROOT" "$LOCAL_ROOT/records" \
  "$LOCAL_ROOT/build" "$LOCAL_ROOT/fixtures"; do
  test ! -L "$path"
  test ! -e "$path" || test -d "$path"
done
install -d -m 0711 -o root -g root /tmp/stwo-gpu-lab
install -d -m 0710 -o root -g dev "$LOCAL_ROOT"
install -d -m 0700 -o root -g root "$LOCAL_ROOT/records"
install -d -m 0700 -o dev -g dev "$LOCAL_ROOT/build" "$LOCAL_ROOT/fixtures"
WORKSPACE_MOUNT=$(findmnt -n -o SOURCE,FSTYPE,TARGET --target /workspace)
LOCAL_MOUNT=$(findmnt -n -o SOURCE,FSTYPE,TARGET --target "$LOCAL_ROOT")
WORKSPACE_DEVICE=$(findmnt -n -o MAJ:MIN,SOURCE,FSTYPE --target /workspace)
LOCAL_DEVICE=$(findmnt -n -o MAJ:MIN,SOURCE,FSTYPE --target "$LOCAL_ROOT")
test -n "$WORKSPACE_MOUNT"
test -n "$LOCAL_MOUNT"
test -n "$WORKSPACE_DEVICE"
test -n "$LOCAL_DEVICE"
test "$WORKSPACE_MOUNT" != "$LOCAL_MOUNT"
test "$WORKSPACE_DEVICE" != "$LOCAL_DEVICE"
export LOCAL_ROOT WORKSPACE_MOUNT LOCAL_MOUNT WORKSPACE_DEVICE LOCAL_DEVICE
python3 - <<'PY'
import json
import os
import tempfile
from pathlib import Path

root = Path("/workspace/gpu-lab")
root.mkdir(parents=True, exist_ok=True)
marker = root / "NETWORK_VOLUME_ID"
expected = os.environ["VOLUME_ID"] + "\\n"
if marker.exists():
    if marker.read_text() != expected:
        raise SystemExit(f"network-volume marker mismatch: {{marker}}")
else:
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(expected)
        out.flush()
        os.fsync(out.fileno())
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

local = Path(os.environ["LOCAL_ROOT"])
document = {{
    "schema_version": "stwo.gpu-lab.local-root.v1",
    "pod_id": os.environ["POD_ID"],
    "volume_id": os.environ["VOLUME_ID"],
    "local_root": str(local),
    "local_mount": os.environ["LOCAL_MOUNT"],
    "local_device": os.environ["LOCAL_DEVICE"],
    "workspace_mount": os.environ["WORKSPACE_MOUNT"],
    "workspace_device": os.environ["WORKSPACE_DEVICE"],
}}
payload = json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\\n"
marker = local / "LOCAL_ROOT.json"
if marker.exists():
    if marker.is_symlink() or marker.read_text() != payload:
        raise SystemExit(f"local-root marker mismatch: {{marker}}")
else:
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())

layout = {{
    "schema_version": "stwo.gpu-lab.dev-layout.v1",
    "dev_uid": 1000,
    "dev_gid": 1000,
    "local_root": str(local),
    "local_root_mode": "0710",
    "records_root": str(local / "records"),
    "records_uid": 0,
    "records_gid": 0,
    "records_mode": "0700",
    "build_root": str(local / "build"),
    "build_uid": 1000,
    "build_gid": 1000,
    "build_mode": "0700",
    "fixtures_root": str(local / "fixtures"),
    "fixtures_uid": 1000,
    "fixtures_gid": 1000,
    "fixtures_mode": "0700",
}}
layout_payload = json.dumps(layout, allow_nan=False, indent=2, sort_keys=True) + "\\n"
layout_marker = local / "DEV_LAYOUT.json"
if layout_marker.exists():
    if layout_marker.is_symlink() or layout_marker.read_text() != layout_payload:
        raise SystemExit(f"dev-layout marker mismatch: {{layout_marker}}")
else:
    descriptor = os.open(layout_marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as output:
        output.write(layout_payload)
        output.flush()
        os.fsync(output.fileno())

expected = {{
    local: (0, 1000, 0o710),
    local / "records": (0, 0, 0o700),
    local / "build": (1000, 1000, 0o700),
    local / "fixtures": (1000, 1000, 0o700),
    marker: (0, 0, 0o444),
    layout_marker: (0, 0, 0o444),
}}
for path, identity in expected.items():
    info = path.lstat()
    actual = (info.st_uid, info.st_gid, info.st_mode & 0o7777)
    if path.is_symlink() or actual != identity:
        raise SystemExit(f"unsafe dev-layout identity for {{path}}: {{actual}}")
active = Path("/tmp/stwo-gpu-lab/ACTIVE_ROOT")
temporary = active.with_name(".ACTIVE_ROOT.tmp")
temporary.write_text(str(local) + "\\n")
os.replace(temporary, active)
PY
cat > /etc/profile.d/stwo-gpu-lab-local.sh <<EOF
export GPU_LAB_LOCAL_ROOT={shlex.quote(local_root)}
EOF
chmod 644 /etc/profile.d/stwo-gpu-lab-local.sh
HB={shlex.quote(c.HEARTBEAT_PATH)}
touch "$HB"
cat > /usr/local/bin/stwo-lab-ttl <<'LABTTL'
#!/bin/sh
set -eu
sleep {ttl_seconds}
( {ttl_seal}
) >> /var/log/stwo-lab-ttl.log 2>&1 || {{
  echo "persistence failed; refusing unsealed TTL exit" >> /var/log/stwo-lab-ttl.log
  exit 1
}}
# This credential-free guard only exits the container. The local watchdog owns
# account-level termination; neither half is a server-side guarantee.
kill -TERM 1 2>/dev/null || true
sleep 10
kill -KILL 1 2>/dev/null || true
LABTTL
chmod 700 /usr/local/bin/stwo-lab-ttl
cat > /usr/local/bin/stwo-lab-idle <<'LABIDLE'
#!/bin/sh
set -eu
HB={shlex.quote(c.HEARTBEAT_PATH)}
IDLE_S={idle_seconds}
while :; do
  sleep 30
  now=$(date +%s)
  last=$(stat -c %Y "$HB" 2>/dev/null || echo 0)
  age=$((now - last))
  [ "$age" -lt "$IDLE_S" ] && continue
  busy=0
  for pid in $(pgrep -f '{BUSY_PROCESS_RE}' || true); do
    [ "$pid" = "$$" ] || busy=1
  done
  if [ "$busy" -eq 1 ]; then
    touch "$HB"
    continue
  fi
  ROOT=/workspace/gpu-lab/leases/{shlex.quote(pod_id)}
  mkdir -p "$ROOT"
  date -u +%FT%TZ > "$ROOT/IDLE_TIMEOUT_AT"
  ( {idle_seal}
  ) >> /var/log/stwo-lab-idle.log 2>&1 || {{
    echo "persistence failed; refusing unsealed idle exit" >> /var/log/stwo-lab-idle.log
    exit 1
  }}
  kill -TERM 1 2>/dev/null || true
  sleep 10
  kill -KILL 1 2>/dev/null || true
  exit 0
done
LABIDLE
chmod 700 /usr/local/bin/stwo-lab-idle
if [ -f /var/run/stwo-lab-ttl.pid ]; then
  kill "$(cat /var/run/stwo-lab-ttl.pid)" 2>/dev/null || true
fi
if [ -f /var/run/stwo-lab-idle.pid ]; then
  kill "$(cat /var/run/stwo-lab-idle.pid)" 2>/dev/null || true
fi
nohup /usr/local/bin/stwo-lab-ttl </dev/null >/var/log/stwo-lab-ttl.log 2>&1 &
echo $! > /var/run/stwo-lab-ttl.pid
nohup /usr/local/bin/stwo-lab-idle </dev/null >/var/log/stwo-lab-idle.log 2>&1 &
echo $! > /var/run/stwo-lab-idle.pid
sleep 1
kill -0 "$(cat /var/run/stwo-lab-ttl.pid)"
kill -0 "$(cat /var/run/stwo-lab-idle.pid)"
mkdir -p /workspace/gpu-lab/leases/{shlex.quote(pod_id)}/records
"""


def _touch_heartbeat(ep: c.Endpoint) -> None:
    command = (
        f"test -f {shlex.quote(c.HEARTBEAT_PATH)} && "
        f"touch {shlex.quote(c.HEARTBEAT_PATH)}"
    )
    if c.ssh_run(ep, command, timeout=30):
        raise RuntimeError("remote idle heartbeat is absent or could not be renewed")


def _remote_is_idle(ep: c.Endpoint, idle_seconds: int) -> bool:
    """Read authenticated remote activity for provider-level idle enforcement."""
    command = f"""
set -eu
HB={shlex.quote(c.HEARTBEAT_PATH)}
test -f "$HB"
now=$(date +%s)
last=$(stat -c %Y "$HB")
busy=0
for pid in $(pgrep -f '{BUSY_PROCESS_RE}' || true); do
  [ "$pid" = "$$" ] || busy=1
done
printf 'LABCTL_IDLE age=%s busy=%s\\n' "$((now - last))" "$busy"
"""
    rc, output = c.ssh_capture(ep, command, timeout=30)
    match = re.fullmatch(r"LABCTL_IDLE age=([0-9]+) busy=([01])", output.strip())
    if rc or not match:
        raise RuntimeError(f"could not authenticate remote idle state: {output[-200:]}")
    age, busy = map(int, match.groups())
    return age >= idle_seconds + 60 and busy == 0


def _mark_terminated(state: dict, *, reason: str) -> None:
    state["phase"] = "terminated"
    state["terminated_at"] = time.time()
    state["termination_reason"] = reason
    c._write_state(state)


def _persist_for_termination(state: dict, pod: c.api.PodInfo, *, reason: str) -> None:
    """Freeze durable outputs before the first provider mutation."""
    if state.get("phase") != "open" or not state.get("remote_guard_installed_at"):
        return
    if pod.id != state.get("pod_id"):
        return
    if pod.status != "RUNNING":
        state["persistence_note"] = "remote already exited; credential-free guard owned sealing"
        c._write_state(state)
        return
    if not pod.ssh_host:
        raise RuntimeError("running lease has no SSH endpoint; refusing unsealed termination")
    try:
        result, _ = persistence.persist_remote(
            c.Endpoint.of(pod), state, freeze=True, reason=reason
        )
    except Exception as error:
        state["persistence_error"] = str(error)
        state["persistence_reason"] = reason
        c._write_state(state)
        raise
    state["last_persistence"] = result
    state.pop("persistence_error", None)
    state.pop("persistence_reason", None)
    c._write_state(state)


def _terminate_state_pod(state: dict, pod: c.api.PodInfo | None, *, reason: str) -> None:
    if pod:
        _persist_for_termination(state, pod, reason=reason)
        provider._terminate_pod_once(pod.id)
        c.ledger.append("terminate", pod_id=pod.id, gpu=pod.gpu, note=reason)
    _mark_terminated(state, reason=reason)


def _terminate_all_state_candidates(state: dict, *, reason: str) -> list[str]:
    """Reconcile reservation, known ids, and account name before cleanup."""
    candidates = {}
    for candidate_id in filter(None, [state.get("pod_id"), *state.get("orphan_ids", [])]):
        candidate = c.api.get_pod(candidate_id)
        if candidate:
            candidates[candidate.id] = candidate
    for candidate in c.api.list_pods():
        if candidate.name == state["lease_name"]:
            candidates[candidate.id] = candidate
    for candidate in candidates.values():
        _persist_for_termination(state, candidate, reason=reason)
    failures = []
    for candidate in candidates.values():
        try:
            provider._terminate_pod_once(candidate.id)
            c.ledger.append("terminate", pod_id=candidate.id, gpu=candidate.gpu, note=reason)
        except Exception as error:
            failures.append(f"{candidate.id}: {error}")
    if failures:
        state["phase"] = "cleanup_failed"
        state["orphan_ids"] = [candidate.id for candidate in candidates.values()]
        state["termination_error"] = "; ".join(failures)
        state["termination_reason"] = reason
        c._write_state(state)
        raise RuntimeError(state["termination_error"])
    _mark_terminated(state, reason=reason)
    return list(candidates)


def _elapsed_spend(state: dict, now: float, rate: float | None = None) -> float:
    usd_hr = state["usd_hr"] if rate is None else rate
    if not math.isfinite(float(usd_hr)) or float(usd_hr) < 0:
        raise RuntimeError(f"invalid live hourly price: {usd_hr}")
    return max(now - state["created_at"], 0) / 3600 * float(usd_hr)


def _check_live_budget(state: dict, pod: c.api.PodInfo) -> None:
    ttl_hours = (state["expires_at"] - state["created_at"]) / 3600
    c._check_budget(
        pod.cost_per_hr, ttl_hours, state["max_usd_hr"], state["max_total_usd"]
    )


def _spawn_watchdog(state: dict) -> int:
    env = os.environ.copy()
    env["LABCTL_STATE"] = str(c.STATE)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(c.ENTRYPOINT),
            "_watch",
            "--lease-name",
            state["lease_name"],
            "--created-at",
            str(state["created_at"]),
            "--expires-at",
            str(state["expires_at"]),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    time.sleep(0.05)
    if proc.poll() is not None:
        raise RuntimeError("detached local expiry watchdog exited during startup")
    return proc.pid


def _cancel_watchdog(state: dict) -> None:
    pid = state.get("watchdog_pid")
    if isinstance(pid, int) and pid > 1 and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _require_open_state(state: dict) -> None:
    if state.get("phase") != "open" or not state.get("pod_id"):
        raise RuntimeError(
            f"lease is {state.get('phase', 'invalid')}; close/reconcile it before use"
        )


def _active() -> tuple[dict, c.api.PodInfo, c.Endpoint]:
    state = c._read_state()
    _require_open_state(state)
    pod = c.api.get_pod(state["pod_id"])
    if not pod:
        raise RuntimeError(f"lease pod no longer exists: {state['pod_id']}")
    now = time.time()
    if now >= state["expires_at"]:
        _terminate_all_state_candidates(state, reason="local-expiry-enforcement")
        raise RuntimeError(f"lease {pod.id} expired and was terminated")
    try:
        _check_live_budget(state, pod)
    except RuntimeError as error:
        _terminate_state_pod(state, pod, reason="live-budget-ceiling-exceeded")
        raise RuntimeError(f"live budget invalid; terminated {pod.id}: {error}") from error
    if pod.status != "RUNNING":
        raise RuntimeError(f"lease pod is {pod.status}, not RUNNING")
    return state, pod, c.Endpoint.of(pod)
