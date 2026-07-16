"""Strict recipe-bound provider lease policy for ``pod_run.sh``."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

PREFIX = "# pod_run: lease "
FIELDS = {
    "one_shot",
    "final_action",
    "gpu",
    "gpu_count",
    "min_vcpu",
    "min_mem_gb",
    "max_usd_hr",
    "name_prefix",
    "ttl_hours",
    "idle_min",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
# Leave room for the eight-hex-character lease suffix while respecting the
# provider's 64-character pod-name limit.
NAME_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,54}-")


@dataclass(frozen=True)
class LeasePolicy:
    one_shot: bool
    final_action: str
    gpu: str
    gpu_count: int
    min_vcpu: int
    min_mem_gb: int
    max_usd_hr: float
    name_prefix: str
    ttl_hours: float
    idle_min: int

    def env_lines(self) -> list[str]:
        return [
            f"ONE_SHOT={int(self.one_shot)}",
            f"FINAL_ACTION={self.final_action}",
            f"GPU={self.gpu}",
            f"GPU_COUNT={self.gpu_count}",
            f"MIN_VCPU={self.min_vcpu}",
            f"MIN_MEM_GB={self.min_mem_gb}",
            f"MAX_USD_HR={self.max_usd_hr}",
            f"NAME_PREFIX={self.name_prefix}",
            f"TTL_HOURS={self.ttl_hours}",
            f"IDLE_MIN={self.idle_min}",
        ]


def _positive_int(values: dict[str, str], name: str) -> int:
    try:
        value = int(values[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"lease {name} must be an integer") from error
    if value <= 0 or str(value) != values[name]:
        raise ValueError(f"lease {name} must be a canonical positive integer")
    return value


def _positive_float(values: dict[str, str], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"lease {name} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"lease {name} must be finite and positive")
    return value


def load(path: Path) -> LeasePolicy:
    lines = [line for line in path.read_text().splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"recipe must contain exactly one {PREFIX.strip()!r} line")
    values: dict[str, str] = {}
    for token in lines[0][len(PREFIX):].split():
        if token.count("=") != 1:
            raise ValueError(f"malformed lease token: {token!r}")
        name, value = token.split("=", 1)
        if name not in FIELDS:
            raise ValueError(f"unknown lease field: {name!r}")
        if name in values:
            raise ValueError(f"duplicate lease field: {name!r}")
        if not value:
            raise ValueError(f"empty lease field: {name!r}")
        values[name] = value
    missing = FIELDS - values.keys()
    if missing:
        raise ValueError(f"missing lease fields: {sorted(missing)}")

    if values["one_shot"] not in {"true", "false"}:
        raise ValueError("lease one_shot must be true or false")
    one_shot = values["one_shot"] == "true"
    final_action = values["final_action"]
    if final_action not in {"stop", "terminate"}:
        raise ValueError("lease final_action must be stop or terminate")
    if one_shot != (final_action == "terminate"):
        raise ValueError("one-shot leases must terminate; reusable leases must stop")
    if not TOKEN_RE.fullmatch(values["gpu"]):
        raise ValueError("lease gpu is malformed")
    if not NAME_PREFIX_RE.fullmatch(values["name_prefix"]):
        raise ValueError("lease name_prefix must be a safe prefix ending in '-'")

    gpu_count = _positive_int(values, "gpu_count")
    if gpu_count != 1:
        raise ValueError("pod_run supports exactly one GPU per worker")
    return LeasePolicy(
        one_shot=one_shot,
        final_action=final_action,
        gpu=values["gpu"],
        gpu_count=gpu_count,
        min_vcpu=_positive_int(values, "min_vcpu"),
        min_mem_gb=_positive_int(values, "min_mem_gb"),
        max_usd_hr=_positive_float(values, "max_usd_hr"),
        name_prefix=values["name_prefix"],
        ttl_hours=_positive_float(values, "ttl_hours"),
        idle_min=_positive_int(values, "idle_min"),
    )
