#!/usr/bin/env python3
"""Fail-closed resident-reuse checks shared by SN2 checkpoint modes."""

from typing import Optional


def _require_exact_int(record: dict[str, object], field: str, expected: int) -> None:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise SystemExit(f"{field}: expected integer {expected}, got {value!r}")


def require_resident_reuse(
    record: dict[str, object],
    reps: int,
    *,
    max_host_preparation_ns: Optional[int] = None,
) -> None:
    if max_host_preparation_ns is not None and max_host_preparation_ns <= 0:
        raise ValueError("max_host_preparation_ns must be positive")
    if record.get("gpu_host_plan_cache_materialization") != "reused":
        raise SystemExit("final repetition did not reuse the replacement host plan")
    if record.get("gpu_shape_executable_materialization") != "reused":
        raise SystemExit("final repetition did not reuse the shape executable")
    for field, expected in {
        "gpu_host_plan_cache_hits": reps - 1,
        "gpu_host_plan_cache_misses": 1,
        "gpu_host_plan_cache_compilations": 1,
        "gpu_host_plan_cache_evictions": 0,
        "gpu_host_plan_cache_collisions": 0,
        "gpu_shape_executable_cache_hits": reps - 1,
        "gpu_shape_executable_cache_misses": 1,
        "gpu_shape_executable_cache_compilations": 1,
        "gpu_shape_executable_cache_source_generation_passes": 1,
        "gpu_shape_executable_cache_binding_recipe_compilations": 1,
        "gpu_shape_executable_cache_capacity_rejections": 0,
        "gpu_hot_allocations": 0,
    }.items():
        _require_exact_int(record, field, expected)
    if record.get("gpu_workspace_materialization") != "reused":
        raise SystemExit("final repetition did not reuse the resident workspace")
    if max_host_preparation_ns is not None:
        total = record.get("gpu_host_preparation_total_ns")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise SystemExit(
                f"gpu_host_preparation_total_ns: expected nonnegative integer, got {total!r}"
            )
        if total > max_host_preparation_ns:
            raise SystemExit(
                "warm replacement host preparation exceeded budget: "
                f"{total}ns > {max_host_preparation_ns}ns"
            )
