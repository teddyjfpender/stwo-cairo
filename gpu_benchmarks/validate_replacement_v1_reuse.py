#!/usr/bin/env python3
"""Fail-closed resident-reuse checks shared by SN2 checkpoint modes."""


def _require_exact_int(record: dict[str, object], field: str, expected: int) -> None:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise SystemExit(f"{field}: expected integer {expected}, got {value!r}")


def require_resident_reuse(record: dict[str, object], reps: int) -> None:
    if record.get("gpu_shape_executable_materialization") != "reused":
        raise SystemExit("final repetition did not reuse the shape executable")
    for field, expected in {
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
