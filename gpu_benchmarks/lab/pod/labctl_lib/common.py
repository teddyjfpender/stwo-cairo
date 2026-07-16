"""Shared paths, state, locks, validation, and budget policy."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


POD_DIR = Path(__file__).resolve().parent.parent
ENTRYPOINT = POD_DIR / "labctl"
GPU_BENCH = POD_DIR.parents[1]
STWO_CAIRO = GPU_BENCH.parent
STWO = STWO_CAIRO.parent / "stwo"
FLEET = GPU_BENCH / "fleet"
sys.path.insert(0, str(FLEET))

from gpufleet import DEFAULT_DISK_GB, VOLUME_MOUNT, api, ledger  # noqa: E402
from gpufleet.podctl import (  # noqa: E402
    SSH_OPTS,
    Endpoint,
    rsync,
    ssh_capture,
    ssh_run,
    wait_ready,
)


GPU_IDS = {
    "a5000": "NVIDIA RTX A5000",
    "3090": "NVIDIA GeForce RTX 3090",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
    "a40": "NVIDIA A40",
    "l40s": "NVIDIA L40S",
    "h100": "NVIDIA H100 80GB HBM3",
}
MAX_TTL_HOURS = 6.0
HEARTBEAT_PATH = "/tmp/stwo-gpu-lab.heartbeat"
IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
SYNC_EXCLUDES = (
    "target/",
    "gpu_benchmarks/results/",
    "gpu_benchmarks/loop/results/",
    "gpu_benchmarks/fleet/results/",
)
STATE = Path(
    os.environ.get(
        "LABCTL_STATE", Path.home() / ".cache" / "stwo-gpu-lab" / "lease.json"
    )
)


def _lock_path() -> Path:
    return STATE.with_suffix(STATE.suffix + ".lock")


def _operation_lock_path() -> Path:
    return STATE.with_suffix(STATE.suffix + ".operation.lock")


@contextlib.contextmanager
def _lease_lock(*, blocking: bool = True):
    """Serialize local lease transitions before any lifecycle mutation."""
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as lock:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(lock, flags)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def _operation_lock():
    """Serialize remote operations without delaying the expiry watchdog."""
    path = _operation_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_state() -> dict:
    if not STATE.exists():
        raise RuntimeError("no active lab lease; run `labctl open` first")
    state = json.loads(STATE.read_text())
    if not isinstance(state, dict) or not state.get("phase"):
        raise RuntimeError(f"invalid lease state: {STATE}")
    return state


def _write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    payload = json.dumps(state, allow_nan=False, indent=2, sort_keys=True) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, STATE)
    directory = os.open(STATE.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _token(prefix: str, plan: dict) -> str:
    body = json.dumps(
        plan, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return f"{prefix}-{hashlib.sha256(body).hexdigest()[:12]}"


def _check_budget(price: float, ttl_hours: float, hourly: float, total: float) -> None:
    values = {
        "price": price,
        "ttl": ttl_hours,
        "hourly ceiling": hourly,
        "total ceiling": total,
    }
    for label, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f"{label} must be finite")
    if not 0.25 <= ttl_hours <= MAX_TTL_HOURS:
        raise RuntimeError(f"TTL must be 0.25..{MAX_TTL_HOURS:g} hours")
    if hourly <= 0 or total <= 0:
        raise RuntimeError("budget ceilings must be positive")
    if price <= 0 or price > hourly:
        raise RuntimeError(f"${price:.2f}/hr exceeds hourly ceiling ${hourly:.2f}")
    lease_cost = price * ttl_hours
    if lease_cost > total:
        raise RuntimeError(
            f"lease ceiling ${lease_cost:.2f} exceeds total ceiling ${total:.2f}"
        )


def _validate_open_args(args: argparse.Namespace) -> None:
    if not args.volume_id or not ID_RE.fullmatch(args.volume_id):
        raise RuntimeError("--volume-id is required and must be an exact RunPod id")
    if not args.volume_dc or not ID_RE.fullmatch(args.volume_dc):
        raise RuntimeError("--volume-dc is required to bind volume placement")
    if not args.image or args.image.count("@") != 1 or not IMAGE_RE.fullmatch(args.image):
        raise RuntimeError("--image must end in an exact lowercase @sha256:<64 hex> digest")
    name = args.name or f"stwo-gpu-lab-{args.gpu.lower()}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,47}", name):
        raise RuntimeError(
            "lease name must be 1..48 alphanumeric/hyphen characters; "
            "pass --name for arbitrary GPU ids"
        )
    if args.min_vcpu < 1 or args.min_mem_gb < 1:
        raise RuntimeError("minimum CPU and memory must be positive")
    if (
        not isinstance(args.idle_min, int)
        or isinstance(args.idle_min, bool)
        or not 5 <= args.idle_min <= 120
    ):
        raise RuntimeError("--idle-min must be an integer from 5 through 120")
    if args.idle_min * 60 >= args.ttl_hours * 3600 - 30:
        raise RuntimeError("--idle-min must leave at least 30 seconds before TTL")
    if not math.isfinite(args.ready_timeout) or args.ready_timeout < 30:
        raise RuntimeError("--ready-timeout must be finite and at least 30 seconds")
    if args.ready_timeout >= args.ttl_hours * 3600 - 30:
        raise RuntimeError("--ready-timeout must leave at least 30 seconds before TTL")
    if args.ready_timeout + args.idle_min * 60 >= args.ttl_hours * 3600 - 30:
        raise RuntimeError("readiness plus idle window must end at least 30 seconds before TTL")
