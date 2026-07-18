#!/usr/bin/env python3
"""Validate one indicative SN2 compiled-Composition vertical checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PCS_FIELDS = (
    "gpu_pcs_driver_architecture",
    "gpu_pcs_runtime_mode",
    "gpu_pcs_stage_started",
    "gpu_pcs_stage_finished",
    "gpu_pcs_batched_tree_decommit",
    "gpu_pcs_driver_complete",
    "gpu_host_syncs",
    "gpu_graph_launches",
    "gpu_kernel_launches",
    "gpu_expected_graph_launches",
    "gpu_expected_kernel_launches",
    "gpu_hot_h2d_bytes",
    "gpu_hot_d2h_bytes",
    "gpu_hot_allocations",
    "gpu_hot_allocation_bytes",
    "gpu_hot_frees",
    "gpu_hot_d2d_bytes",
    "gpu_hot_memset_bytes",
    "gpu_hot_fill_words",
    "gpu_hot_capture_begins",
    "gpu_hot_capture_finishes",
    "gpu_hot_capture_aborts",
    "gpu_hot_lane_forks",
    "gpu_hot_lane_joins",
    "gpu_graph_submit_gap_ns_total",
    "gpu_max_graph_submit_gap_ms",
    "gpu_graph_submit_gap_strict_gate_passed",
)


class CheckpointError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _load_primary(path: Path) -> dict[str, object]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("program") == "SN_PIE_2.zip"
            and value.get("backend") == "cuda"
        ):
            records.append(value)
    _require(len(records) == 1, f"expected one SN2 CUDA record, got {len(records)}")
    return records[0]


def validate_checkpoint(
    stdout_path: Path,
    proof_path: Path,
    hardware_path: Path,
    *,
    reps: int = 6,
) -> dict[str, object]:
    record = _load_primary(stdout_path)
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    proof = proof_path.read_bytes()
    _require(bool(proof), "proof artifact is empty")

    exact = {
        "engine": "gpu-native",
        "n": 1,
        "cycle_count": 7_977_397,
        "pie_n_steps": 7_706_864,
        "reps": reps,
        "verified_reps": reps,
        "warm_sample_count": reps - 1,
        "gpu_resident_backend_requested": "replacement-v1",
        "gpu_resident_backend": "replacement-v1",
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": "arena-graph",
        "gpu_graph_a_setup_gate_passed": True,
        "benchmark_diagnostic_mode": True,
        "benchmark_diagnostic_reason": "compiled-composition-vertical-checkpoint",
        "benchmark_graph_submit_capture_mode": False,
        "compiled_composition_vertical_checkpoint": True,
        "compiled_composition_vertical_checkpoint_gate_passed": True,
        "compiled_composition_vertical_checkpoint_timing_scope":
            "public-prove-call-warm-end-to-end",
        "performance_claim_class": "indicative-non-formal",
        "performance_claim_admissible": False,
        "performance_measurement_available": True,
        "gpu_pcs_architecture_gate_applicable": False,
        "gpu_native_architecture_gate_passed": None,
        "fleet_pow_enabled": False,
        "proof_byte_equal_required": True,
        "proof_comparison_applicable": True,
        "proof_byte_equal": True,
        "simd_reference_required": True,
        "simd_reference_comparison_applicable": True,
        "simd_reference_fresh": True,
        "simd_reference_byte_equal": True,
        "proof_mutation_required": True,
        "proof_mutation_kind": "interaction_claim.memory_id_to_big.claimed_sum_plus_one",
        "proof_mutation_rejected": True,
        "proof_mutation_error_class": "invalid_logup_sum",
        "gpu_aot_provenance_gate_passed": True,
    }
    for field, expected in exact.items():
        _require(record.get(field) == expected,
                 f"{field}: expected {expected!r}, got {record.get(field)!r}")

    for field in PCS_FIELDS:
        _require(record.get(field) is None,
                 f"{field}: eager checkpoint must report null PCS telemetry")
    for field in (
        "gpu_graph_replay_intervals",
        "gpu_graph_replay_interval_total_ns",
        "gpu_graph_replay_interval_count",
        "gpu_graph_replay_semantic_count",
        "gpu_graph_replay_timing_scope",
    ):
        _require(record.get(field) is None,
                 f"{field}: vertical checkpoint must not enable graph diagnostics")

    for field in (
        "gpu_aot_misses",
        "gpu_aot_runtime_loads",
        "gpu_aot_runtime_cache_hits",
        "gpu_aot_strict_rejections",
    ):
        _require(record.get(field) == 0, f"{field}: strict AOT gate did not hold")
    for field in ("gpu_aot_loads", "gpu_aot_cache_hits"):
        value = record.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"{field}: invalid AOT activity")
    manifest_hash = record.get("gpu_aot_manifest_hash")
    _require(isinstance(manifest_hash, int) and not isinstance(manifest_hash, bool)
             and manifest_hash > 0, "embedded AOT manifest identity is absent")
    _require(record.get("gpu_policy_kernel_manifest_hash") == manifest_hash,
             "runtime and policy AOT identities differ")

    gpu_digest = record.get("gpu_proof_blake3")
    _require(isinstance(gpu_digest, str) and len(gpu_digest) == 64,
             "GPU proof digest is invalid")
    _require(record.get("simd_reference_blake3") == gpu_digest,
             "SIMD and GPU proof digests differ")
    for field in (
        "prove_s_warm_median",
        "prove_s_warm_p95",
        "useful_mhz_median",
        "useful_mhz_at_warm_p95",
        "verify_ms",
    ):
        _require(_positive_number(record.get(field)), f"{field}: invalid indicative metric")

    _require(
        hardware.get("schema") == "stwo.replacement-v1-sn2.hardware-identity.v2"
        and "H100" in str(hardware.get("name"))
        and hardware.get("compute_capability") == "9.0"
        and isinstance(hardware.get("memory_mib"), int)
        and hardware["memory_mib"] >= 79_000,
        "hardware receipt is not one admitted H100",
    )
    if "formal_promotion_eligible" in record:
        _require(record["formal_promotion_eligible"] is False,
                 "vertical checkpoint cannot be formal-promotion eligible")

    return {
        "schema": "stwo.sn2-compiled-vertical-checkpoint.v1",
        "verdict": "PASS",
        "checkpoint_class": "indicative-non-formal",
        "formal_promotion_eligible": False,
        "program": "SN_PIE_2.zip",
        "gpu": hardware["name"],
        "gpu_uuid": hardware["uuid"],
        "reps": reps,
        "proof_sha256": hashlib.sha256(proof).hexdigest(),
        "proof_bytes": len(proof),
        "proof_blake3": gpu_digest,
        "prove_s_warm_median": record["prove_s_warm_median"],
        "useful_mhz_median": record["useful_mhz_median"],
        "proof_byte_equal": True,
        "simd_reference_byte_equal": True,
        "verified_reps": reps,
        "mutation_rejected": True,
        "strict_session_gate": True,
        "strict_aot_gate": True,
        "pcs_telemetry": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=6)
    args = parser.parse_args()
    try:
        result = validate_checkpoint(
            args.stdout, args.proof, args.hardware, reps=args.reps
        )
    except (CheckpointError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
