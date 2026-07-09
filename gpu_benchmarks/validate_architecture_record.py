#!/usr/bin/env python3
"""Fail closed unless a gpu_bench record proves the typed CUDA PCS path ran."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARCHITECTURE = "cuda-typed-pcs-driver-v1"
STAGES = (
    "OodsEvaluation",
    "QuotientAndCompaction",
    "FriCommitAndFold",
    "ProofOfWork",
    "FriQueryAndDecommit",
    "TreeDecommit",
    "Assembly",
)
RUNTIME_MODES = {
    "detached-eager": "DetachedEager",
    "arena-graph": "ArenaGraph",
}


def load_main_record(path: Path) -> dict[str, Any]:
    record = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "program" in candidate and "backend" in candidate:
                record = candidate
    if record is None:
        raise ValueError("no main gpu_bench record found")
    return record


def validate_record(record: dict[str, Any], required_mode: str) -> list[str]:
    errors: list[str] = []
    expected_mode = RUNTIME_MODES.get(required_mode)
    if expected_mode is None:
        return [f"unsupported required runtime mode: {required_mode}"]

    required = {
        "backend": "cuda",
        "engine": "gpu-native",
        "gpu_pcs_driver_architecture": ARCHITECTURE,
        "gpu_pcs_runtime_mode": expected_mode,
        "gpu_pcs_batched_tree_decommit": True,
        "gpu_pcs_driver_complete": True,
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": required_mode,
        "gpu_native_architecture_gate_passed": True,
    }
    for field, expected in required.items():
        if record.get(field) != expected:
            errors.append(f"{field}: expected {expected!r}, got {record.get(field)!r}")

    for field in ("gpu_pcs_stage_started", "gpu_pcs_stage_finished"):
        counts = record.get(field)
        if not isinstance(counts, dict):
            errors.append(f"{field}: expected an exact seven-stage count map, got {counts!r}")
            continue
        if set(counts) != set(STAGES):
            errors.append(f"{field}: expected stages {list(STAGES)!r}, got {sorted(counts)!r}")
        for stage in STAGES:
            if counts.get(stage) != 1:
                errors.append(f"{field}.{stage}: expected 1, got {counts.get(stage)!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="gpu_bench stdout file")
    parser.add_argument(
        "--runtime-mode",
        choices=tuple(RUNTIME_MODES),
        default="detached-eager",
        help="runtime mode the benchmark requested (default: detached-eager)",
    )
    args = parser.parse_args()
    try:
        record = load_main_record(args.record)
    except (OSError, ValueError) as error:
        print(f"GPU-native architecture contract: {error}", file=sys.stderr)
        return 1
    errors = validate_record(record, args.runtime_mode)
    if errors:
        print("GPU-native architecture contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
