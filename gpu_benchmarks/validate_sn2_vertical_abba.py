#!/usr/bin/env python3
"""Validate and aggregate one SN2 old/new/new/old vertical checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

try:
    from gpu_benchmarks.validate_replacement_v1_reuse import require_resident_reuse
    from gpu_benchmarks.validate_sn2_vertical_checkpoint import (
        CheckpointError,
        validate_checkpoint,
    )
except ModuleNotFoundError:
    from validate_replacement_v1_reuse import require_resident_reuse
    from validate_sn2_vertical_checkpoint import CheckpointError, validate_checkpoint


ORDER = (("old", 1), ("new", 1), ("new", 2), ("old", 2))
PIE_STEPS = 7_706_864
MUTATION_KIND = "interaction_claim.memory_id_to_big.claimed_sum_plus_one"
EXPECTED_STAGES = {
    "Assembly",
    "FriCommitAndFold",
    "FriQueryAndDecommit",
    "OodsEvaluation",
    "ProofOfWork",
    "QuotientAndCompaction",
    "TreeDecommit",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def _positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _round3(value: float) -> float:
    return round(value, 3)


def _primary(path: Path) -> dict[str, object]:
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
    _require(len(records) == 1, f"{path.name}: expected one SN2 CUDA record")
    return records[0]


def _argument_vector(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    _require(values and all(values), f"{path.name}: empty argument vector")
    return values


def _common_arguments(pie: Path, reps: int) -> list[str]:
    return [
        "--pie",
        str(pie),
        "--backend",
        "cuda",
        "--engine",
        "gpu-native",
        "--resident-backend",
        "replacement-v1",
        "--require-gpu-native-architecture",
        "--require-gpu-pcs-runtime-mode",
        "arena-graph",
        "--reuse-input",
        "--reps",
        str(reps),
        "--require-proof-byte-equal",
        "--require-simd-reference-byte-equal",
        "--require-proof-mutation-rejected",
    ]


def _validate_common_record(record: dict[str, object], reps: int) -> list[float]:
    exact = {
        "program": "SN_PIE_2.zip",
        "backend": "cuda",
        "engine": "gpu-native",
        "n": 1,
        "cycle_count": 7_977_397,
        "pie_n_steps": PIE_STEPS,
        "security_bits": 96,
        "n_queries": 70,
        "pow_bits": 26,
        "fold_step": 3,
        "reps": reps,
        "verified_reps": reps,
        "warm_sample_count": reps - 1,
        "gpu_resident_backend_requested": "replacement-v1",
        "gpu_resident_backend": "replacement-v1",
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": "arena-graph",
        "gpu_graph_a_setup_gate_passed": True,
        "fleet_pow_enabled": False,
        "proof_byte_equal_required": True,
        "proof_comparison_applicable": True,
        "proof_byte_equal": True,
        "simd_reference_required": True,
        "simd_reference_comparison_applicable": True,
        "simd_reference_fresh": True,
        "simd_reference_byte_equal": True,
        "proof_mutation_required": True,
        "proof_mutation_kind": MUTATION_KIND,
        "proof_mutation_rejected": True,
        "proof_mutation_error_class": "invalid_logup_sum",
        "gpu_aot_provenance_gate_passed": True,
        "performance_measurement_available": True,
        "throughput_distribution_applicable": True,
    }
    for field, expected in exact.items():
        _require(
            record.get(field) == expected,
            f"{field}: expected {expected!r}, got {record.get(field)!r}",
        )
    for field in (
        "gpu_aot_misses",
        "gpu_aot_runtime_loads",
        "gpu_aot_runtime_cache_hits",
        "gpu_aot_strict_rejections",
        "gpu_setup_base_migration_copies",
        "gpu_setup_lookup_host_copies",
        "gpu_setup_legacy_witness_fallbacks",
    ):
        _require(record.get(field) == 0, f"{field}: fallback or setup drift")
    manifest_hash = record.get("gpu_aot_manifest_hash")
    _require(
        isinstance(manifest_hash, int)
        and not isinstance(manifest_hash, bool)
        and manifest_hash > 0
        and record.get("gpu_policy_kernel_manifest_hash") == manifest_hash,
        "runtime and policy AOT identities differ",
    )
    gpu_digest = record.get("gpu_proof_blake3")
    _require(
        isinstance(gpu_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", gpu_digest) is not None
        and record.get("simd_reference_blake3") == gpu_digest,
        "GPU/SIMD proof digest identity is invalid",
    )
    samples = record.get("prove_s_warm_samples_raw")
    _require(
        isinstance(samples, list)
        and len(samples) == reps - 1
        and all(_positive(sample) for sample in samples),
        "warm proof samples are missing or invalid",
    )
    median = statistics.median(samples)
    reported_median = record.get("prove_s_warm_median")
    _require(
        _positive(reported_median)
        and math.isclose(reported_median, _round3(median), abs_tol=1e-9),
        "reported warm median differs from raw samples",
    )
    useful_mhz = PIE_STEPS / median / 1e6
    _require(
        _positive(record.get("useful_mhz_median"))
        and math.isclose(
            record["useful_mhz_median"], _round3(useful_mhz), abs_tol=0.001
        ),
        "reported useful MHz differs from raw samples",
    )
    return [float(sample) for sample in samples]


def _validate_old(record: dict[str, object], reps: int) -> None:
    exact = {
        "compiled_composition_vertical_checkpoint": False,
        "compiled_composition_vertical_checkpoint_gate_passed": None,
        "compiled_composition_vertical_checkpoint_timing_scope": None,
        "benchmark_diagnostic_mode": False,
        "benchmark_diagnostic_reason": None,
        "benchmark_graph_submit_capture_mode": False,
        "performance_claim_class": None,
        "performance_claim_admissible": True,
        "gpu_native_architecture_gate_passed": True,
        "gpu_pcs_architecture_gate_applicable": True,
        "gpu_pcs_driver_architecture": "cuda-typed-pcs-driver-v1",
        "gpu_pcs_runtime_mode": "ArenaGraph",
        "gpu_pcs_driver_complete": True,
        "gpu_pcs_batched_tree_decommit": True,
    }
    for field, expected in exact.items():
        _require(
            record.get(field) == expected,
            f"old {field}: expected {expected!r}, got {record.get(field)!r}",
        )
    for field in ("gpu_pcs_stage_started", "gpu_pcs_stage_finished"):
        stages = record.get(field)
        _require(
            isinstance(stages, dict)
            and set(stages) == EXPECTED_STAGES
            and all(bool(value) for value in stages.values()),
            f"old {field} is incomplete",
        )
    try:
        require_resident_reuse(record, reps)
    except SystemExit as error:
        raise CheckpointError(f"old resident reuse failed: {error}") from error


def aggregate_abba(
    records: list[dict[str, object]],
    proofs: list[bytes],
    arguments: list[list[str]],
    pie: Path,
    *,
    reps: int = 6,
) -> dict[str, object]:
    _require(len(records) == len(proofs) == len(arguments) == 4, "ABBA needs four runs")
    common_arguments = _common_arguments(pie, reps)
    warm: dict[str, list[float]] = {"old": [], "new": []}
    per_process_medians: dict[str, list[float]] = {"old": [], "new": []}
    digests = set()
    loop_starts = set()

    for (variant, ordinal), record, proof, argv in zip(
        ORDER, records, proofs, arguments, strict=True
    ):
        expected_arguments = common_arguments + (
            ["--compiled-composition-vertical-checkpoint"] if variant == "new" else []
        )
        _require(argv == expected_arguments, f"{variant}{ordinal}: argument vector drifted")
        samples = _validate_common_record(record, reps)
        if variant == "old":
            _validate_old(record, reps)
        else:
            _require(
                record.get("compiled_composition_vertical_checkpoint") is True,
                f"new{ordinal}: compiled Composition was not selected",
            )
        warm[variant].extend(samples)
        per_process_medians[variant].append(statistics.median(samples))
        digests.add(record["gpu_proof_blake3"])
        start = record.get("gpu_proof_loop_started_unix_ns")
        _require(
            isinstance(start, int) and not isinstance(start, bool) and start > 0,
            f"{variant}{ordinal}: invalid proof-loop start",
        )
        loop_starts.add(start)
        _require(bool(proof), f"{variant}{ordinal}: proof is empty")

    _require(len(loop_starts) == 4, "ABBA runs were not four distinct prover processes")
    _require(len(digests) == 1, "ABBA GPU proof digests differ")
    proof_hashes = {hashlib.sha256(proof).hexdigest() for proof in proofs}
    _require(len(proof_hashes) == 1 and len(set(proofs)) == 1, "ABBA proof bytes differ")

    medians = {variant: statistics.median(samples) for variant, samples in warm.items()}
    useful_mhz = {
        variant: PIE_STEPS / median / 1e6 for variant, median in medians.items()
    }
    speedup = medians["old"] / medians["new"]
    return {
        "schema": "stwo.sn2-compiled-vertical-abba.v1",
        "verdict": "PASS",
        "checkpoint_class": "indicative-non-formal",
        "formal_promotion_eligible": False,
        "performance_claim_admissible": False,
        "program": "SN_PIE_2.zip",
        "order": ["old", "new", "new", "old"],
        "reps_per_process": reps,
        "warm_samples_per_variant": len(warm["old"]),
        "proof_sha256": next(iter(proof_hashes)),
        "proof_blake3": next(iter(digests)),
        "proof_byte_equal_across_processes": True,
        "simd_reference_byte_equal_all_processes": True,
        "mutation_rejected_all_processes": True,
        "old": {
            "prove_s_warm_median": _round3(medians["old"]),
            "useful_mhz": _round3(useful_mhz["old"]),
            "process_medians_s": [_round3(value) for value in per_process_medians["old"]],
        },
        "new": {
            "prove_s_warm_median": _round3(medians["new"]),
            "useful_mhz": _round3(useful_mhz["new"]),
            "process_medians_s": [_round3(value) for value in per_process_medians["new"]],
        },
        "new_over_old_speedup": round(speedup, 6),
        "new_minus_old_useful_mhz": _round3(useful_mhz["new"] - useful_mhz["old"]),
    }


def validate_abba(
    run_root: Path,
    prefix: str,
    hardware: Path,
    source: Path,
    pie: Path,
    bootloader: Path,
    input_manifest: Path,
    aot_manifest: Path,
    expected_source: dict[str, dict[str, str]],
    *,
    reps: int = 6,
) -> dict[str, object]:
    records, proofs, arguments = [], [], []
    for variant, ordinal in ORDER:
        stem = run_root / f"{prefix}.{variant}{ordinal}"
        stdout = Path(f"{stem}.stdout.txt")
        proof = Path(f"{stem}.proof.bin")
        records.append(_primary(stdout))
        proofs.append(proof.read_bytes())
        arguments.append(_argument_vector(Path(f"{stem}.args.txt")))
        if variant == "new":
            validate_checkpoint(
                stdout,
                proof,
                hardware,
                source,
                pie,
                bootloader,
                input_manifest,
                aot_manifest,
                expected_source,
                reps=reps,
            )
    result = aggregate_abba(records, proofs, arguments, pie, reps=reps)
    hardware_record = json.loads(hardware.read_text(encoding="utf-8"))
    result["gpu"] = hardware_record.get("name")
    result["gpu_uuid"] = hardware_record.get("uuid")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--pie", required=True, type=Path)
    parser.add_argument("--bootloader", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--aot-manifest", required=True, type=Path)
    parser.add_argument("--stwo-head", required=True)
    parser.add_argument("--stwo-worktree", required=True)
    parser.add_argument("--stwo-cairo-head", required=True)
    parser.add_argument("--stwo-cairo-worktree", required=True)
    parser.add_argument("--reps", type=int, default=6)
    args = parser.parse_args()
    try:
        result = validate_abba(
            args.run_root,
            args.prefix,
            args.hardware,
            args.source,
            args.pie,
            args.bootloader,
            args.input_manifest,
            args.aot_manifest,
            {
                "stwo": {
                    "head": args.stwo_head,
                    "worktree_sha256": args.stwo_worktree,
                },
                "stwo_cairo": {
                    "head": args.stwo_cairo_head,
                    "worktree_sha256": args.stwo_cairo_worktree,
                },
            },
            reps=args.reps,
        )
    except (CheckpointError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
