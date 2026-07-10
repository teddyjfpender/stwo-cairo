#!/usr/bin/env python3
"""Fail closed unless a gpu_bench record proves the typed CUDA PCS path ran."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_cuda_soundness_gate import (
    GATES as CUDA_SOUNDNESS_GATE_COMMANDS,
    gates_for_runtime_mode,
)

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
SOUNDNESS_GATES = {
    name: required for name, _command, required in CUDA_SOUNDNESS_GATE_COMMANDS
}
SOUNDNESS_COMMANDS = {
    name: list(command) for name, command, _required in CUDA_SOUNDNESS_GATE_COMMANDS
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


def load_soundness_gate(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    if not isinstance(artifact, dict):
        raise ValueError("CUDA soundness artifact is not an object")
    return artifact


def validate_soundness_gate(
    artifact: dict[str, Any], required_mode: str | None = None
) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema") != "stwo.cuda.soundness-gate.v2":
        errors.append(f"soundness schema: got {artifact.get('schema')!r}")
    if artifact.get("passed") is not True:
        errors.append("soundness artifact did not pass")
    runtime_mode = artifact.get("runtime_mode")
    try:
        expected_manifest = gates_for_runtime_mode(runtime_mode)
    except ValueError:
        errors.append(f"soundness artifact runtime_mode: got {runtime_mode!r}")
        expected_manifest = ()
    if required_mode is not None and runtime_mode != required_mode:
        errors.append(
            f"soundness artifact runtime_mode: expected {required_mode!r}, "
            f"got {runtime_mode!r}"
        )
    expected_gates = {
        name: required for name, _command, required in expected_manifest
    }
    expected_commands = {
        name: list(command) for name, command, _required in expected_manifest
    }
    gates = artifact.get("gates")
    if not isinstance(gates, list) or not gates:
        return errors + ["soundness artifact has no executed gates"]
    for field in ("stwo_git_head", "stwo_cairo_git_head"):
        value = artifact.get(field)
        if not isinstance(value, str) or len(value) != 40:
            errors.append(f"soundness artifact {field}: expected a 40-character revision")
    for field in ("stwo_worktree_hash", "stwo_cairo_worktree_hash"):
        value = artifact.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"soundness artifact {field}: expected a 64-character source hash")
    names: set[str] = set()
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"soundness gate {index}: expected object")
            continue
        name = gate.get("name")
        if not isinstance(name, str) or not name or name in names:
            errors.append(f"soundness gate {index}: invalid or duplicate name {name!r}")
        else:
            names.add(name)
        executed = gate.get("executed_tests")
        required = gate.get("required_tests")
        if (
            not isinstance(executed, int)
            or isinstance(executed, bool)
            or not isinstance(required, int)
            or isinstance(required, bool)
            or required < 1
            or executed != required
        ):
            errors.append(
                f"soundness gate {name!r}: executed={executed!r}, required={required!r}"
            )
        if gate.get("exit_code") != 0 or gate.get("passed") is not True:
            errors.append(f"soundness gate {name!r}: command did not pass")
        expected_required = expected_gates.get(name)
        if expected_required is not None and required != expected_required:
            errors.append(
                f"soundness gate {name!r}: manifest requires {expected_required}, "
                f"artifact claimed {required!r}"
            )
        expected_command = expected_commands.get(name)
        if expected_command is not None and gate.get("command") != expected_command:
            errors.append(f"soundness gate {name!r}: command does not match manifest")
    missing = set(expected_gates) - names
    unexpected = names - set(expected_gates)
    if missing:
        errors.append(f"soundness artifact missing gates: {sorted(missing)!r}")
    if unexpected:
        errors.append(f"soundness artifact has unexpected gates: {sorted(unexpected)!r}")
    return errors


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
        "gpu_aot_misses": 0,
        "gpu_aot_runtime_loads": 0,
        "gpu_aot_runtime_cache_hits": 0,
        "gpu_aot_strict_rejections": 0,
        "gpu_aot_provenance_gate_passed": True,
    }
    for field, expected in required.items():
        if record.get(field) != expected:
            errors.append(f"{field}: expected {expected!r}, got {record.get(field)!r}")

    performance_admissible = required_mode == "arena-graph"
    if record.get("performance_claim_admissible") is not performance_admissible:
        errors.append(
            "performance_claim_admissible: expected "
            f"{performance_admissible!r}, got "
            f"{record.get('performance_claim_admissible')!r}"
        )
    performance_fields = (
        "steps_per_s",
        "mhz",
        "useful_mhz",
        "mhz_median",
        "useful_mhz_median",
        "mhz_at_warm_p95",
        "useful_mhz_at_warm_p95",
    )
    if performance_admissible:
        for field in ("steps_per_s", "mhz", "useful_mhz"):
            value = record.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                errors.append(
                    f"{field}: expected a positive ArenaGraph performance value, "
                    f"got {value!r}"
                )
    else:
        for field in performance_fields:
            if record.get(field) is not None:
                errors.append(
                    f"{field}: DetachedEager is correctness-only and must report null, "
                    f"got {record.get(field)!r}"
                )

    for field in (
        "gpu_aot_misses",
        "gpu_aot_runtime_loads",
        "gpu_aot_runtime_cache_hits",
        "gpu_aot_strict_rejections",
    ):
        count = record.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count != 0:
            errors.append(f"{field}: expected integer 0, got {count!r}")

    for field in ("gpu_aot_loads", "gpu_aot_cache_hits"):
        count = record.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append(f"{field}: expected a non-negative reported count, got {count!r}")
    manifest_hash = record.get("gpu_aot_manifest_hash")
    if not isinstance(manifest_hash, int) or isinstance(manifest_hash, bool) or manifest_hash <= 0:
        errors.append(
            f"gpu_aot_manifest_hash: expected a non-zero embedded-pack identity, got {manifest_hash!r}"
        )

    for field in ("gpu_pcs_stage_started", "gpu_pcs_stage_finished"):
        counts = record.get(field)
        if not isinstance(counts, dict):
            errors.append(f"{field}: expected an exact seven-stage count map, got {counts!r}")
            continue
        if set(counts) != set(STAGES):
            errors.append(f"{field}: expected stages {list(STAGES)!r}, got {sorted(counts)!r}")
        for stage in STAGES:
            count = counts.get(stage)
            if not isinstance(count, int) or isinstance(count, bool) or count != 1:
                errors.append(f"{field}.{stage}: expected integer 1, got {count!r}")
    if required_mode == "arena-graph":
        exact = {
            "gpu_host_syncs": 1,
            "gpu_hot_h2d_bytes": 0,
            "gpu_hot_allocations": 0,
            "gpu_graph_a_setup_gate_passed": True,
            "gpu_setup_base_migration_copies": 0,
            "gpu_setup_lookup_host_copies": 0,
            "gpu_setup_legacy_witness_fallbacks": 0,
        }
        for field, expected in exact.items():
            value = record.get(field)
            valid = (
                value is expected
                if isinstance(expected, bool)
                else isinstance(value, int)
                and not isinstance(value, bool)
                and value == expected
            )
            if not valid:
                errors.append(f"{field}: expected integer {expected}, got {value!r}")
        ingest_syncs = record.get("gpu_witness_ingest_syncs")
        if (
            not isinstance(ingest_syncs, int)
            or isinstance(ingest_syncs, bool)
            or not 0 <= ingest_syncs <= 1
        ):
            errors.append(
                "gpu_witness_ingest_syncs: expected integer in [0, 1], "
                f"got {ingest_syncs!r}"
            )
        for field in (
            "gpu_execution_tables_ingest_compact_h2d_bytes",
            "gpu_execution_tables_ingest_descriptor_h2d_bytes",
        ):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{field}: expected a positive setup byte count, got {value!r}")
        compact_copies = record.get(
            "gpu_execution_tables_ingest_compact_h2d_copies"
        )
        if (
            not isinstance(compact_copies, int)
            or isinstance(compact_copies, bool)
            or not 1 <= compact_copies <= 3
        ):
            errors.append(
                "gpu_execution_tables_ingest_compact_h2d_copies: "
                f"expected integer in [1, 3], got {compact_copies!r}"
            )
        descriptor_copies = record.get(
            "gpu_execution_tables_ingest_descriptor_h2d_copies"
        )
        if (
            not isinstance(descriptor_copies, int)
            or isinstance(descriptor_copies, bool)
            or descriptor_copies != 2
        ):
            errors.append(
                "gpu_execution_tables_ingest_descriptor_h2d_copies: "
                f"expected integer 2, got {descriptor_copies!r}"
            )
        execution_table_syncs = record.get("gpu_execution_tables_ingest_syncs")
        if (
            not isinstance(execution_table_syncs, int)
            or isinstance(execution_table_syncs, bool)
            or execution_table_syncs != 1
        ):
            errors.append(
                "gpu_execution_tables_ingest_syncs: "
                f"expected integer 1, got {execution_table_syncs!r}"
            )
        for field in ("gpu_graph_launches", "gpu_kernel_launches"):
            value = record.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value < 100
            ):
                errors.append(f"{field}: expected integer in [1, 100), got {value!r}")
        d2h = record.get("gpu_hot_d2h_bytes")
        if not isinstance(d2h, int) or isinstance(d2h, bool) or d2h <= 0:
            errors.append(f"gpu_hot_d2h_bytes: expected one positive final bundle, got {d2h!r}")
        max_gap = record.get("gpu_max_graph_submit_gap_ms")
        if (
            not isinstance(max_gap, (int, float))
            or isinstance(max_gap, bool)
            or not 0 <= max_gap < 50
        ):
            errors.append(
                f"gpu_max_graph_submit_gap_ms: expected value in [0, 50), got {max_gap!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="gpu_bench stdout file")
    parser.add_argument(
        "--runtime-mode",
        choices=tuple(RUNTIME_MODES),
        default="arena-graph",
        help="runtime mode the benchmark requested (default: arena-graph)",
    )
    parser.add_argument(
        "--soundness-gate",
        type=Path,
        required=True,
        help="counted CUDA differential-test artifact from run_cuda_soundness_gate.py",
    )
    args = parser.parse_args()
    try:
        record = load_main_record(args.record)
    except (OSError, ValueError) as error:
        print(f"GPU-native architecture contract: {error}", file=sys.stderr)
        return 1
    errors = validate_record(record, args.runtime_mode)
    try:
        soundness = load_soundness_gate(args.soundness_gate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"CUDA soundness artifact: {error}")
    else:
        errors.extend(validate_soundness_gate(soundness, args.runtime_mode))
    if errors:
        print("GPU-native architecture contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
