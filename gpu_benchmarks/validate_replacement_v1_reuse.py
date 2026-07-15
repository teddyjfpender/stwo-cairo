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
    if record.get("gpu_prepared_runtime_materialization") != "reused":
        raise SystemExit("final repetition did not reuse the prepared CUDA runtime")
    if record.get("gpu_prepared_runtime_capture_ready_at_entry") is not True:
        raise SystemExit("final repetition did not enter with a complete captured topology")
    if record.get("gpu_prepared_runtime_capture_ready_at_exit") is not True:
        raise SystemExit("final repetition did not exit with a complete captured topology")
    if record.get("gpu_statement_refresh_present") is not True:
        raise SystemExit("final repetition did not refresh statement-varying CUDA inputs")
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
        "gpu_execution_tables_ingest_descriptor_h2d_copies": 0,
        "gpu_composition_direct_split_graphs": 1,
        "gpu_composition_precomputed_compact_commitments": 1,
        "gpu_composition_coefficient_commit_paths": 0,
        "gpu_composition_split_fused_d2d_nodes": 0,
        "gpu_hot_allocations": 0,
        "gpu_hot_allocation_bytes": 0,
        "gpu_hot_frees": 0,
        "gpu_hot_d2d_bytes": 0,
        "gpu_hot_memset_bytes": 0,
        "gpu_hot_fill_words": 0,
        "gpu_hot_capture_begins": 0,
        "gpu_hot_capture_finishes": 0,
        "gpu_hot_capture_aborts": 0,
        "gpu_hot_lane_forks": 0,
        "gpu_hot_lane_joins": 0,
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
