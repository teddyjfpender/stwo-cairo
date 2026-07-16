"""Pod-side operations: ssh/rsync, readiness, health check, bootstrap, deadman.

The deadman design places NO credentials on the pod: a container stops itself by
killing PID 1 (RunPod then marks the pod Exited and GPU billing stops; disk
persists). Two independent triggers, both installed at bootstrap:
  * TTL: hard wall-clock cap for the whole pod (default 6h).
  * idle: no heartbeat touch AND no prover process for N minutes (default 45).
The manifest runner touches the heartbeat before every step; long steps hold a
`flock` on it so in-flight work never reads as idle.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import DEFAULT_IDLE_STOP_MIN, DEFAULT_TTL_HOURS, HEARTBEAT_PATH
from .api import PodInfo, get_pod


SSH_KEY_NAMES = ("runpodctl-ssh-key", "RunPod-Key-Go", "RunPod-Key-Ed25519")
SSH_BASE_OPTS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=6",
    "-o", "LogLevel=ERROR",
)


def _private_key() -> Path:
    configured = os.environ.get("RUNPOD_SSH_KEY", "").strip()
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        root = Path.home() / ".runpod" / "ssh"
        candidates = [
            root / name for name in SSH_KEY_NAMES if os.path.lexists(root / name)
        ]
        if not candidates:
            raise ValueError(
                "no SSH private key found; set RUNPOD_SSH_KEY to one exact file"
            )
        if len(candidates) != 1:
            raise ValueError(
                "multiple SSH private keys found; set RUNPOD_SSH_KEY to one exact file"
            )

    key = Path(os.path.abspath(candidates[0]))
    try:
        identity = key.lstat()
    except OSError as error:
        raise ValueError(f"SSH private key is unavailable: {key}") from error
    if stat.S_ISLNK(identity.st_mode) or not stat.S_ISREG(identity.st_mode):
        raise ValueError("SSH key must name one regular private-key file, not a symlink")
    if identity.st_uid != os.getuid():
        raise ValueError("SSH private key must be owned by the current user")
    mode = stat.S_IMODE(identity.st_mode)
    if mode & 0o077:
        raise ValueError("SSH private key must have no group or world permissions")
    if not mode & stat.S_IRUSR:
        raise ValueError("SSH private key must be readable by its owner")
    return key


def _ssh_opts() -> list[str]:
    key = _private_key()
    return [
        "-i", str(key),
        "-o", "IdentitiesOnly=yes",
        *SSH_BASE_OPTS,
    ]


class _LazySSHOptions:
    """Resolve the exact identity only when a transport is actually constructed."""

    def __iter__(self):
        return iter(_ssh_opts())


SSH_OPTS = _LazySSHOptions()


@dataclass
class Endpoint:
    host: str
    port: int
    user: str = "root"

    @classmethod
    def of(cls, pod: PodInfo) -> "Endpoint":
        if not pod.ssh_host or not pod.ssh_port:
            raise RuntimeError(
                f"pod {pod.id} has no public ssh endpoint (status {pod.status})"
            )
        return cls(pod.ssh_host, pod.ssh_port)


def ssh_run(
    ep: Endpoint,
    cmd: str,
    *,
    timeout: float | None = None,
    log_file: Path | None = None,
) -> int:
    """Run one remote command; stream combined output to log_file (and return rc)."""
    argv = ["ssh", *_ssh_opts(), "-p", str(ep.port), f"{ep.user}@{ep.host}", cmd]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab") as f:
            # Sentinel so criteria evaluation can separate the echoed COMMAND from
            # its OUTPUT (a must_not_match pattern must never trip on the command
            # text that defines it — learned live).
            f.write(f"\n$ {cmd}\n--8<-- output --8<--\n".encode())
            f.flush()
            proc = subprocess.run(
                argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
    else:
        proc = subprocess.run(argv, timeout=timeout, check=False)
    return proc.returncode


def ssh_capture(ep: Endpoint, cmd: str, *, timeout: float = 120) -> tuple[int, str]:
    argv = ["ssh", *_ssh_opts(), "-p", str(ep.port), f"{ep.user}@{ep.host}", cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def rsync(
    ep: Endpoint, src: str, dst: str, *, pull: bool = False, timeout: float = 1800
) -> int:
    ssh_opts = _ssh_opts()
    remote = f"{ep.user}@{ep.host}:"
    a, b = (remote + src, dst) if pull else (src, remote + dst)
    # No --info/--append-verify: macOS ships rsync 2.6.9 (learned lesson).
    argv = [
        "rsync", "-az", "--partial", "--stats",
        "-e", "ssh " + " ".join(shlex.quote(o) for o in ssh_opts) + f" -p {ep.port}",
        a, b,
    ]
    return subprocess.run(argv, timeout=timeout, check=False).returncode


def wait_ready(pod_id: str, *, timeout_s: float = 600) -> PodInfo:
    """Poll until the pod has a public ssh port that answers `true`."""
    deadline = time.time() + timeout_s
    last = "provisioning"
    while time.time() < deadline:
        pod = get_pod(pod_id)
        if pod and pod.ssh_host and pod.ssh_port:
            try:
                rc, _ = ssh_capture(Endpoint.of(pod), "true", timeout=20)
                if rc == 0:
                    return pod
                last = f"ssh rc={rc}"
            except Exception as e:  # transient connect failures while booting
                last = str(e)[:80]
        elif pod:
            last = f"status={pod.status}, no public ssh yet"
        time.sleep(10)
    raise TimeoutError(f"pod {pod_id} not ssh-ready after {timeout_s:.0f}s ({last})")


# ------------------------------- health ---------------------------------------------

HEALTH_CMD = r"""
set -o pipefail
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
  --format=csv,noheader 2>/dev/null | head -1)"
echo "cuda=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
echo "vcpu=$(nproc)"
echo "ram_gb=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
echo "disk_free_gb=$(df -BG /workspace 2>/dev/null | awk 'NR==2{print $4}' | tr -d G)"
echo "rustc=$( (~/.cargo/bin/rustc --version 2>/dev/null || rustc --version 2>/dev/null) | head -1)"
"""


def health_check(ep: Endpoint) -> dict[str, str]:
    rc, out = ssh_capture(ep, HEALTH_CMD, timeout=60)
    info: dict[str, str] = {"reachable": str(rc == 0)}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
    return info


# ------------------------------- deadman --------------------------------------------


def deadman_script(ttl_hours: float, idle_min: int) -> str:
    """The on-pod watchdog (POSIX sh; no credentials, self-stop via `kill 1`)."""
    ttl_s = int(ttl_hours * 3600)
    idle_s = int(idle_min * 60)
    return f"""
mkdir -p /var/log
touch {HEARTBEAT_PATH}
cat > /usr/local/bin/gpufleet-deadman.sh <<'DEADMAN'
#!/bin/sh
# gpufleet deadman: stop this pod (kill PID 1 -> container exits -> GPU billing
# stops) on TTL expiry or on idleness (stale heartbeat AND no prover process).
TTL_S={ttl_s}
IDLE_S={idle_s}
HB={HEARTBEAT_PATH}
START=$(date +%s)
while :; do
  sleep 60
  NOW=$(date +%s)
  if [ $((NOW - START)) -ge "$TTL_S" ]; then
    echo "$(date -u) deadman: TTL reached, stopping pod" >> /var/log/gpufleet-deadman.log
    kill 1; exit 0
  fi
  HB_AGE=$((NOW - $(stat -c %Y "$HB" 2>/dev/null || echo 0)))
  BUSY=$(pgrep -f 'gpu_bench|cargo|rustc|nvcc|ptxas|rsync' | head -1)
  if [ "$HB_AGE" -ge "$IDLE_S" ] && [ -z "$BUSY" ]; then
    echo "$(date -u) deadman: idle ${{HB_AGE}}s and no work, stopping pod" \
      >> /var/log/gpufleet-deadman.log
    kill 1; exit 0
  fi
done
DEADMAN
chmod +x /usr/local/bin/gpufleet-deadman.sh
# Replace a previous instance via PIDFILE — never pkill by pattern: this very
# shell's cmdline contains the script text (heredoc), so any -f pattern that can
# find the old daemon also finds and kills THIS session (learned the hard way).
if [ -f /var/run/gpufleet-deadman.pid ]; then
  kill "$(cat /var/run/gpufleet-deadman.pid)" 2>/dev/null || true
fi
nohup /usr/local/bin/gpufleet-deadman.sh >/dev/null 2>&1 &
echo $! > /var/run/gpufleet-deadman.pid
echo "deadman installed: ttl={ttl_hours}h idle={idle_min}min pid=$(cat /var/run/gpufleet-deadman.pid)"
"""


BOOTSTRAP_CMD = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
command -v rsync >/dev/null && command -v tmux >/dev/null && command -v jq >/dev/null || {
  apt-get update -qq && apt-get install -y -qq rsync tmux jq bc >/dev/null; }
command -v cargo >/dev/null 2>&1 || [ -x ~/.cargo/bin/cargo ] || {
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null; }
mkdir -p /workspace
echo bootstrap-ok
"""


def bootstrap(
    ep: Endpoint,
    *,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    idle_min: int = DEFAULT_IDLE_STOP_MIN,
    log_file: Path | None = None,
) -> bool:
    rc = ssh_run(ep, BOOTSTRAP_CMD, timeout=900, log_file=log_file)
    if rc != 0:
        return False
    rc = ssh_run(ep, deadman_script(ttl_hours, idle_min), timeout=60, log_file=log_file)
    return rc == 0


def touch_heartbeat(ep: Endpoint) -> None:
    ssh_capture(ep, f"touch {HEARTBEAT_PATH}", timeout=20)
