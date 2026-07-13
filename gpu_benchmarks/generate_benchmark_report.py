#!/usr/bin/env python3
"""Generate a hash-bound Markdown and JSON report from one qualification round.

The qualification artifact is the authority.  This tool recomputes published
statistics from raw warm samples and refuses to turn incomplete, mutated, or
failed qualification data into a report.  Actual-SN SIMD identity is reported
only when every fixed SN record carries the required fresh byte-equality oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BENCHMARK_DIR = str(Path(__file__).resolve().parent)
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from validate_architecture_record import (  # noqa: E402
    AOT_ADMISSION_KEYS,
    AOT_MANIFEST_ENTRY_KEYS,
    AOT_PREFLIGHT_KEYS,
    ARCHITECTURE,
    FLAGS_OFF_POLICY,
    HEADLINE_POLICY,
    PREFLIGHT_CAP_BYTES,
    QUALIFICATION_FLAGS,
    QUALIFICATION_PROFILES,
    RETAINED_BUDGET_FLAG,
    STAGES,
    UNIVERSAL_POLICY,
    _aot_occurrence_sort_key,
    _valid_aot_occurrence,
    validate_record,
    validate_qualification_soundness_gate,
)

CANONICAL_RAW_MANIFEST_PATH = Path(BENCHMARK_DIR) / "pie" / "SHA256SUMS"
CANONICAL_ADAPTED_MANIFEST_PATH = (
    Path(BENCHMARK_DIR) / "pie" / "ADAPTED_SHA256SUMS"
)


QUALIFICATION_SCHEMA = "stwo.qualification-round.v4"
REPORT_SCHEMA = "stwo.benchmark-report.v1"
EMPTY_WORKTREE_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_SECURITY = {
    "security_bits": 96,
    "n_queries": 70,
    "pow_bits": 26,
    "fold_step": 3,
}
SN_NAMES = tuple(f"SN_PIE_{number}" for number in range(1, 5))
RAW_INPUT_NAMES = (*tuple(f"{name}.zip" for name in SN_NAMES), "simple_bootloader_compiled.json")
ADAPTED_INPUT_NAMES = tuple(f"{name}.adapted.bin" for name in SN_NAMES)
REMOTE_INPUT_PATHS = {
    "gate": "/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_2.zip",
    "bootloader": "/workspace/bench_inputs/simple_bootloader_compiled.json",
    **{
        name: f"/workspace/stwo-cairo/gpu_benchmarks/pie/sn/{name}.zip"
        for name in SN_NAMES
    },
}
TOP_PHASE_COUNT = 10
GPU_TELEMETRY_SCHEMA = "stwo.gpu-telemetry.csv.v1"
GPU_TELEMETRY_COLUMNS = (
    "timestamp_unix_ns",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
    "power_draw_w",
    "clock_sm_mhz",
    "clock_memory_mhz",
    "temperature_gpu_c",
    "driver_version",
    "power_limit_w",
    "clock_max_sm_mhz",
    "clock_max_memory_mhz",
)
GPU_TELEMETRY_METRICS = GPU_TELEMETRY_COLUMNS[1:8]
GPU_TELEMETRY_INTERVAL_MS = 250
GPU_TELEMETRY_MAX_GAP_NS = 2_000_000_000
GPU_TELEMETRY_METADATA_KEYS = {
    "schema",
    "columns",
    "path",
    "sha256",
    "remote_sha256",
    "size_bytes",
    "remote_size_bytes",
    "sample_count",
    "proof_window_sample_count",
    "sampler_complete",
    "sample_interval_ms",
    "max_gap_ns",
    "transport_equal",
}
NCU_PROFILE_KEYS = {
    "schema",
    "path",
    "sha256",
    "bytes",
    "kernel_regex",
    "launch_count",
    "set",
    "ncu_version",
    "synthetic",
    "remote_sha256",
    "remote_bytes",
    "remote_import_validated",
    "import_output_path",
    "import_output_sha256",
    "import_output_bytes",
    "remote_import_output_sha256",
    "remote_import_output_bytes",
    "profiled_kernel_rows",
    "profiled_proof_sha256",
}
NCU_RELEASE_KERNEL_REGEX = "relation_fused|relation_scan|stream_leaf_update"
NCU_RELEASE_LAUNCH_COUNT = 10
NCU_RELEASE_SET = "full"
QUALIFICATION_NORMALIZED_STATES = {
    "flags_off": {
        **{flag: 0 for flag in QUALIFICATION_FLAGS},
        RETAINED_BUDGET_FLAG: FLAGS_OFF_POLICY["retained_lde_budget_bytes"],
    },
    "universal_sn1_sn4": {
        **{flag: int(flag in {
            "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
            "STWO_CUDA_B2N_STAGE_FUSED",
        }) for flag in QUALIFICATION_FLAGS},
        RETAINED_BUDGET_FLAG: UNIVERSAL_POLICY["retained_lde_budget_bytes"],
    },
    "sn2_headline": {
        **{flag: 1 for flag in QUALIFICATION_FLAGS},
        RETAINED_BUDGET_FLAG: HEADLINE_POLICY["retained_lde_budget_bytes"],
    },
}


class ReportError(ValueError):
    """The qualification artifact is not safe to publish."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _is_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_head(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_artifact(round_dir: Path, raw_path: Any, label: str) -> Path:
    _require(isinstance(raw_path, str) and raw_path, f"{label}: missing artifact path")
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, round_dir / raw]
    candidates.append(round_dir / raw.name)

    # Qualification paths are sometimes repo-relative.  When the report is run
    # elsewhere, preserve the suffix following the round directory's name.
    parts = raw.parts
    if round_dir.name in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index(round_dir.name)
        suffix = parts[index + 1 :]
        if suffix:
            candidates.append(round_dir.joinpath(*suffix))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = [candidate for candidate in round_dir.rglob(raw.name) if candidate.is_file()]
    _require(len(matches) <= 1, f"{label}: artifact basename is ambiguous: {raw.name}")
    if matches:
        return matches[0].resolve()
    raise ReportError(f"{label}: artifact does not exist: {raw_path}")


def _verified_artifact(
    round_dir: Path, reference: Any, label: str, hashes: dict[str, str]
) -> Path:
    _require(isinstance(reference, dict), f"{label}: expected path/hash object")
    expected = reference.get("sha256")
    _require(_is_sha256(expected), f"{label}: invalid sha256")
    path = _resolve_artifact(round_dir, reference.get("path"), label)
    actual = _sha256(path)
    _require(actual == expected, f"{label}: sha256 mismatch")
    hashes[label] = actual
    return path


def _round3(value: float) -> float:
    # Measurement values are non-negative; this matches Rust f64::round().
    return math.floor(value * 1000.0 + 0.5) / 1000.0


def _quantile(samples: list[float], q: float) -> float:
    """R-7 linearly interpolated quantile, matching gpu_bench."""

    ordered = sorted(samples)
    rank = q * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _require_rounded(record: dict[str, Any], field: str, expected: float, label: str) -> None:
    actual = record.get(field)
    _require(
        _is_number(actual) and abs(float(actual) - _round3(expected)) <= 1e-9,
        f"{label}.{field}: expected {_round3(expected)!r}, got {actual!r}",
    )


def _validate_measurement(
    record: Any,
    *,
    expected_program: str,
    expected_gpu: str,
    label: str,
    require_fresh_simd_reference: bool = False,
) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{label}: missing measurement record")
    _require(record.get("program") == expected_program, f"{label}: wrong program")
    _require(record.get("backend") == "cuda", f"{label}: backend must be cuda")
    _require(record.get("engine") == "gpu-native", f"{label}: engine must be gpu-native")
    _require(record.get("gpu") == expected_gpu, f"{label}: wrong GPU")
    architecture_errors = validate_record(record, "arena-graph")
    _require(
        not architecture_errors,
        f"{label}: resident architecture contract failed: {'; '.join(architecture_errors)}",
    )

    reps = record.get("reps")
    _require(reps == 6 and not isinstance(reps, bool),
             f"{label}.reps: qualification requires exactly 6")
    _require(record.get("verified_reps") == reps, f"{label}: not every proof verified")
    _require(record.get("warm_sample_count") == reps - 1, f"{label}: warm sample count drift")
    samples_raw = record.get("prove_s_warm_samples_raw")
    _require(
        isinstance(samples_raw, list)
        and len(samples_raw) == reps - 1
        and all(_is_number(sample, positive=True) for sample in samples_raw),
        f"{label}: warm samples must be exact finite positive values",
    )
    samples = [float(sample) for sample in samples_raw]

    for field, expected in EXPECTED_SECURITY.items():
        _require(record.get(field) == expected, f"{label}.{field}: expected {expected}")
    for field in ("cycle_count", "pie_n_steps"):
        value = record.get(field)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                 f"{label}.{field}: expected positive integer")
    _require(record.get("throughput_distribution_applicable") is True,
             f"{label}: throughput distribution is not applicable")
    _require(record.get("performance_claim_admissible") is True,
             f"{label}: runtime measurement is not performance-capable")
    for field in (
        "proof_comparison_applicable",
        "proof_byte_equal_required",
        "proof_byte_equal",
    ):
        _require(record.get(field) is True, f"{label}.{field}: expected true")
    simd_reference = None
    if require_fresh_simd_reference:
        for field in (
            "simd_reference_required",
            "simd_reference_comparison_applicable",
            "simd_reference_byte_equal",
            "simd_reference_fresh",
        ):
            _require(record.get(field) is True, f"{label}.{field}: expected true")
        gpu_digest = record.get("gpu_proof_blake3")
        digest = record.get("simd_reference_blake3")
        _require(_is_sha256(gpu_digest),
                 f"{label}.gpu_proof_blake3: expected lowercase hex digest")
        _require(_is_sha256(digest),
                 f"{label}.simd_reference_blake3: expected lowercase hex digest")
        _require(gpu_digest == digest,
                 f"{label}: GPU proof and SIMD reference BLAKE3 digests differ")
        elapsed = record.get("simd_reference_s")
        _require(_is_number(elapsed, positive=True),
                 f"{label}.simd_reference_s: expected finite positive duration")
        simd_reference = {
            "required": True,
            "comparison_applicable": True,
            "byte_equal": True,
            "fresh": True,
            "blake3": digest,
            "gpu_proof_blake3": gpu_digest,
            "elapsed_s": float(elapsed),
        }

    median = float(statistics.median(samples))
    p95 = _quantile(samples, 0.95)
    cycle_count = record["cycle_count"]
    useful_steps = record["pie_n_steps"]
    _require_rounded(record, "prove_s_warm_median", median, label)
    _require_rounded(record, "prove_s_warm_p95", p95, label)
    _require_rounded(record, "mhz_median", cycle_count / median / 1e6, label)
    _require_rounded(record, "useful_mhz_median", useful_steps / median / 1e6, label)
    _require_rounded(record, "mhz_at_warm_p95", cycle_count / p95 / 1e6, label)
    _require_rounded(record, "useful_mhz_at_warm_p95", useful_steps / p95 / 1e6, label)

    for field in ("prove_s_cold", "proof_kb"):
        _require(_is_number(record.get(field), positive=True), f"{label}.{field}: invalid value")
    for field in (
        "verify_ms",
        "peak_rss_gb",
        "vram_end_gb",
        "vram_peak_gb",
        "pool_used_high_gb",
        "pool_reserved_high_gb",
    ):
        _require(_is_number(record.get(field)), f"{label}.{field}: invalid value")
        _require(float(record[field]) >= 0, f"{label}.{field}: must be non-negative")
    _require(
        float(record["pool_reserved_high_gb"]) >= float(record["pool_used_high_gb"]),
        f"{label}: reserved pool high-water is below used high-water",
    )

    return {
        "program": expected_program,
        "reps": reps,
        "verified_reps": reps,
        "warm_samples_s": samples,
        "warm_median_s": _round3(median),
        "warm_p95_s": _round3(p95),
        "cycle_count": cycle_count,
        "pie_n_steps": useful_steps,
        "mhz_median": _round3(cycle_count / median / 1e6),
        "useful_mhz_median": _round3(useful_steps / median / 1e6),
        "mhz_at_warm_p95": _round3(cycle_count / p95 / 1e6),
        "useful_mhz_at_warm_p95": _round3(useful_steps / p95 / 1e6),
        "verify_ms": float(record["verify_ms"]),
        "proof_kb": float(record["proof_kb"]),
        "peak_rss_gb": float(record["peak_rss_gb"]),
        "vram_peak_gb": float(record["vram_peak_gb"]),
        "pool_used_high_gb": float(record["pool_used_high_gb"]),
        "pool_reserved_high_gb": float(record["pool_reserved_high_gb"]),
        "gpu_repeat_byte_equal": True,
        "simd_reference": simd_reference,
        "security": dict(EXPECTED_SECURITY),
    }


def _parse_checksum_manifest(path: Path, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        _require(len(fields) == 2, f"{label}:{line_number}: malformed checksum line")
        digest, name = fields
        _require(_is_sha256(digest), f"{label}:{line_number}: invalid sha256")
        _require(name not in values, f"{label}:{line_number}: duplicate {name}")
        values[name] = digest
    return values


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReportError(f"{label}:{line_number}: invalid JSON: {error.msg}") from error
        _require(isinstance(entry, dict), f"{label}:{line_number}: expected object")
        entries.append(entry)
    return entries


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _round3(statistics.fmean(values)),
        "p95": _round3(_quantile(values, 0.95)),
        "max": _round3(max(values)),
    }


def _validate_gpu_telemetry(
    qualification: dict[str, Any],
    benchmark_records: dict[str, Any],
    round_dir: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    retained = qualification.get("gpu_telemetry")
    _require(isinstance(retained, dict) and set(retained) == set(SN_NAMES),
             "gpu_telemetry: expected exactly SN1-SN4")
    results = {}
    for name in SN_NAMES:
        label = f"gpu_telemetry.{name}"
        metadata = retained[name]
        _require(isinstance(metadata, dict) and set(metadata) == GPU_TELEMETRY_METADATA_KEYS,
                 f"{label}: metadata shape differs from the fixed v1 contract")
        _require(metadata.get("schema") == GPU_TELEMETRY_SCHEMA,
                 f"{label}: wrong schema")
        _require(metadata.get("columns") == list(GPU_TELEMETRY_COLUMNS),
                 f"{label}: wrong declared columns")
        _require(metadata.get("sampler_complete") is True,
                 f"{label}: sampler did not complete")
        _require(metadata.get("sample_interval_ms") == GPU_TELEMETRY_INTERVAL_MS,
                 f"{label}: expected a declared 250ms interval")
        _require(metadata.get("max_gap_ns") == GPU_TELEMETRY_MAX_GAP_NS,
                 f"{label}: wrong declared maximum gap")
        _require(metadata.get("transport_equal") is True,
                 f"{label}: remote/local transport equality is false")

        path = _resolve_artifact(round_dir, metadata.get("path"), label)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        size = len(payload)
        _require(size > 0, f"{label}: empty CSV")
        _require(digest == metadata.get("sha256") == metadata.get("remote_sha256"),
                 f"{label}: local/remote SHA256 mismatch")
        _require(size == metadata.get("size_bytes") == metadata.get("remote_size_bytes"),
                 f"{label}: local/remote byte-size mismatch")
        hashes[label] = digest

        try:
            rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ReportError(f"{label}: invalid CSV: {error}") from error
        _require(rows and list(rows[0]) == list(GPU_TELEMETRY_COLUMNS),
                 f"{label}: CSV header differs from the fixed v1 schema")
        _require(metadata.get("sample_count") == len(rows),
                 f"{label}: sample count mismatch")

        parsed = []
        last_timestamp = 0
        for index, row in enumerate(rows, 1):
            _require(None not in row and set(row) == set(GPU_TELEMETRY_COLUMNS)
                     and all(row[field] is not None for field in GPU_TELEMETRY_COLUMNS),
                     f"{label}: row {index} has the wrong column count")
            try:
                timestamp = int(row["timestamp_unix_ns"])
                values = {
                    field: float(row[field])
                    for field in (*GPU_TELEMETRY_METRICS, "power_limit_w",
                                  "clock_max_sm_mhz", "clock_max_memory_mhz")
                }
            except (TypeError, ValueError) as error:
                raise ReportError(f"{label}: row {index} has invalid numeric data") from error
            _require(timestamp > last_timestamp,
                     f"{label}: timestamps are not strictly increasing")
            if last_timestamp:
                _require(timestamp - last_timestamp <= GPU_TELEMETRY_MAX_GAP_NS,
                         f"{label}: cadence gap exceeds 2 seconds")
            last_timestamp = timestamp
            _require(all(math.isfinite(value) and value >= 0 for value in values.values()),
                     f"{label}: row {index} has a non-finite or negative value")
            _require(values["utilization_gpu_pct"] <= 100
                     and values["utilization_memory_pct"] <= 100,
                     f"{label}: row {index} has utilization above 100 percent")
            driver = row["driver_version"].strip()
            _require(driver, f"{label}: row {index} has an empty driver version")
            parsed.append((timestamp, values, driver))

        record = benchmark_records[name]
        start = record.get("gpu_proof_loop_started_unix_ns")
        finish = record.get("gpu_proof_loop_finished_unix_ns")
        _require(isinstance(start, int) and not isinstance(start, bool)
                 and isinstance(finish, int) and not isinstance(finish, bool)
                 and 0 < start < finish,
                 f"{label}: benchmark has an invalid proof-loop envelope")
        pre = [sample for sample in parsed if sample[0] < start]
        inside = [sample for sample in parsed if start <= sample[0] <= finish]
        post = [sample for sample in parsed if sample[0] > finish]
        _require(pre and len(inside) >= 2 and post,
                 f"{label}: telemetry lacks pre/in/post proof-loop coverage")
        _require(inside[0][0] - pre[-1][0] <= GPU_TELEMETRY_MAX_GAP_NS
                 and post[0][0] - inside[-1][0] <= GPU_TELEMETRY_MAX_GAP_NS,
                 f"{label}: telemetry does not cover both proof-loop edges")
        _require(metadata.get("proof_window_sample_count") == len(inside),
                 f"{label}: proof-window sample count mismatch")

        drivers = {sample[2] for sample in inside}
        _require(len(drivers) == 1, f"{label}: driver version changed inside proof loop")
        hardware = {}
        for field in ("power_limit_w", "clock_max_sm_mhz", "clock_max_memory_mhz"):
            values = {sample[1][field] for sample in inside}
            _require(len(values) == 1, f"{label}: {field} changed inside proof loop")
            hardware[field] = next(iter(values))
        results[name] = {
            "schema": GPU_TELEMETRY_SCHEMA,
            "path": str(path),
            "sha256": digest,
            "size_bytes": size,
            "sample_interval_ms": GPU_TELEMETRY_INTERVAL_MS,
            "max_gap_ns": GPU_TELEMETRY_MAX_GAP_NS,
            "proof_loop_started_unix_ns": start,
            "proof_loop_finished_unix_ns": finish,
            "proof_loop_sample_count": len(inside),
            "metrics": {
                field: _summary([sample[1][field] for sample in inside])
                for field in GPU_TELEMETRY_METRICS
            },
            "hardware": {"driver_version": next(iter(drivers)), **hardware},
            "scope": (
                "Samples filtered to the benchmark proof-loop envelope; this includes "
                "per-repetition input clones and phase-report gaps and is not exact GPU-only time"
            ),
        }
    return results


def _validate_ncu_profile(
    qualification: dict[str, Any],
    *,
    dry_run: bool,
    qualified_sn2_proof: str,
    round_dir: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    profiling = qualification.get("profiling")
    _require(isinstance(profiling, dict) and set(profiling) == {"ncu"},
             "profiling: expected exactly the required NCU profile")
    wrapper = profiling.get("ncu")
    _require(isinstance(wrapper, dict) and set(wrapper) == {
        "required", "requested", "attempted", "status", "profile"
    }, "profiling.ncu: invalid wrapper shape")
    _require(wrapper.get("required") is True and wrapper.get("requested") is True
             and wrapper.get("attempted") is True and wrapper.get("status") == "validated",
             "profiling.ncu: required capture was not validated")
    profile = wrapper.get("profile")
    _require(isinstance(profile, dict) and set(profile) == NCU_PROFILE_KEYS,
             "profiling.ncu.profile: invalid metadata shape")
    _require(profile.get("schema") == "stwo.ncu-profile.v1",
             "profiling.ncu.profile: wrong schema")
    _require(profile.get("synthetic") is dry_run,
             "profiling.ncu.profile: synthetic status disagrees with qualification")
    _require(profile.get("remote_import_validated") is True,
             "profiling.ncu.profile: remote CSV import was not validated")
    _require(profile.get("profiled_proof_sha256") == qualified_sn2_proof,
             "profiling.ncu.profile: profiled proof differs from qualified SN2")
    _require(profile.get("kernel_regex") == NCU_RELEASE_KERNEL_REGEX
             and profile.get("launch_count") == NCU_RELEASE_LAUNCH_COUNT
             and profile.get("set") == NCU_RELEASE_SET,
             "profiling.ncu.profile: capture settings differ from release contract")
    _require(isinstance(profile.get("ncu_version"), str)
             and profile["ncu_version"].startswith(
                 "NVIDIA (R) Nsight Compute Command Line Profiler"
             ),
             "profiling.ncu.profile: missing profiler version")
    launch_count = profile.get("launch_count")
    try:
        kernel_pattern = re.compile(profile["kernel_regex"])
    except re.error as error:
        raise ReportError(f"profiling.ncu.profile: invalid kernel regex: {error}") from error

    report_path = _resolve_artifact(round_dir, profile.get("path"), "profiling.ncu.report")
    report_size = report_path.stat().st_size
    report_sha = _sha256(report_path)
    with report_path.open("rb") as stream:
        report_prefix = stream.read(8192)
    _require(report_size > 0, "profiling.ncu.report: empty artifact")
    _require(report_sha == profile.get("sha256") == profile.get("remote_sha256"),
             "profiling.ncu.report: local/remote SHA256 mismatch")
    _require(report_size == profile.get("bytes") == profile.get("remote_bytes"),
             "profiling.ncu.report: local/remote byte-size mismatch")
    synthetic_prefix = b"STWO synthetic Nsight Compute report v1\n"
    if dry_run:
        _require(report_prefix.startswith(synthetic_prefix),
                 "profiling.ncu.report: dry run lacks the synthetic marker")
    else:
        _require(not report_prefix.startswith(synthetic_prefix)
                 and report_size >= 8
                 and report_prefix[:4] in (b"NVP\0", b"NVR\0"),
                 "profiling.ncu.report: real qualification lacks NVIDIA report magic")
    hashes["profiling.ncu.report"] = report_sha

    csv_path = _resolve_artifact(
        round_dir, profile.get("import_output_path"), "profiling.ncu.import_csv"
    )
    csv_payload = csv_path.read_bytes()
    csv_sha = hashlib.sha256(csv_payload).hexdigest()
    _require(csv_payload, "profiling.ncu.import_csv: empty artifact")
    _require(csv_sha == profile.get("import_output_sha256")
             == profile.get("remote_import_output_sha256"),
             "profiling.ncu.import_csv: local/remote SHA256 mismatch")
    _require(len(csv_payload) == profile.get("import_output_bytes")
             == profile.get("remote_import_output_bytes"),
             "profiling.ncu.import_csv: local/remote byte-size mismatch")
    try:
        csv_rows = list(csv.reader(csv_payload.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ReportError(f"profiling.ncu.import_csv: invalid CSV: {error}") from error
    kernel_index = None
    header_width = None
    profiled_kernel_rows = 0
    for row in csv_rows:
        if "Kernel Name" in row:
            kernel_index = row.index("Kernel Name")
            header_width = len(row)
            continue
        if (kernel_index is not None and header_width is not None
                and len(row) == header_width and row[kernel_index].strip()):
            _require(kernel_pattern.search(row[kernel_index]) is not None,
                     "profiling.ncu.import_csv: kernel outside declared regex")
            profiled_kernel_rows += 1
    _require(profiled_kernel_rows > 0,
             "profiling.ncu.import_csv: no profiled kernel data rows")
    _require(profiled_kernel_rows == profile.get("profiled_kernel_rows"),
             "profiling.ncu.import_csv: profiled kernel row count mismatch")
    synthetic_csv_prefix = b"STWO synthetic ncu import validation v1\n"
    _require(csv_payload.startswith(synthetic_csv_prefix) if dry_run
             else not csv_payload.startswith(synthetic_csv_prefix),
             "profiling.ncu.import_csv: synthetic marker disagrees with qualification")
    hashes["profiling.ncu.import_csv"] = csv_sha
    return {
        "status": "validated",
        "required": True,
        "requested": True,
        "attempted": True,
        "schema": profile["schema"],
        "synthetic": dry_run,
        "kernel_regex": profile["kernel_regex"],
        "launch_count": launch_count,
        "set": profile["set"],
        "ncu_version": profile["ncu_version"],
        "profiled_proof_sha256": qualified_sn2_proof,
        "profiled_kernel_rows": profiled_kernel_rows,
        "report": {
            "path": str(report_path), "sha256": report_sha, "size_bytes": report_size
        },
        "import_csv": {
            "path": str(csv_path), "sha256": csv_sha, "size_bytes": len(csv_payload)
        },
    }


def _validate_phase_ledger(
    path: Path,
    benchmark_records: dict[str, Any],
    *,
    expected_env: str,
    expected_gpu: str,
    expected_proof_hashes: dict[str, str],
    expected_telemetry: dict[str, Any],
    expected_soundness_sha256: str,
    expected_target: dict[str, Any],
    expected_target_sha256: str,
    expected_projection: dict[str, Any],
) -> dict[str, Any]:
    entries = _load_jsonl(path, "ledger.bench")
    _require(Counter(entry.get("run_name") for entry in entries) == Counter({
        "gate_correctness": 2,
        **{name: 1 for name in SN_NAMES},
    }), "ledger.bench: entry set is incomplete, duplicated, or unexpected")
    for entry in entries:
        _require(entry.get("pod_gpu") == expected_gpu
                 and entry.get("execution_target") == expected_target
                 and entry.get("execution_target_sha256") == expected_target_sha256
                 and entry.get("source_projection") == expected_projection
                 and entry.get("soundness_gate_sha256") == expected_soundness_sha256
                 and entry.get("execution_guard_passed") is True
                 and entry.get("remote_quiescence_passed") is True,
                 "ledger.bench: entry escaped the qualified execution seal")
    summaries = {}
    expected_reps = set(range(6))
    for sn_name in SN_NAMES:
        matches = [entry for entry in entries if entry.get("run_name") == sn_name]
        _require(len(matches) == 1,
                 f"ledger.bench: expected exactly one {sn_name} entry, got {len(matches)}")
        entry = matches[0]
        _require(entry.get("status") == "ok", f"ledger.bench.{sn_name}: status is not ok")
        _require(entry.get("bench_env") == expected_env
                 and entry.get("qualification_probe") is True
                 and entry.get("pod_gpu") == expected_gpu
                 and entry.get("proof_sha256") == expected_proof_hashes[sn_name]
                 and entry.get("gpu_telemetry") == expected_telemetry[sn_name],
                 f"ledger.bench.{sn_name}: environment or qualification binding drifted")
        _require(entry.get("record") == benchmark_records[sn_name],
                 f"ledger.bench.{sn_name}: measurement differs from qualification")
        phase_reps = entry.get("phase_totals")
        _require(isinstance(phase_reps, list) and len(phase_reps) == 6,
                 f"ledger.bench.{sn_name}.phase_totals: expected six repetitions")

        by_rep: dict[int, dict[str, Any]] = {}
        phase_names: set[str] | None = None
        for item in phase_reps:
            _require(isinstance(item, dict),
                     f"ledger.bench.{sn_name}.phase_totals: expected objects")
            rep = item.get("rep")
            _require(isinstance(rep, int) and not isinstance(rep, bool) and rep not in by_rep,
                     f"ledger.bench.{sn_name}.phase_totals: duplicate or invalid rep {rep!r}")
            phases = item.get("phase_totals")
            _require(isinstance(phases, dict) and phases,
                     f"ledger.bench.{sn_name}.rep{rep}: expected nonempty phase_totals")
            names = set(phases)
            _require(all(isinstance(name, str) and name.strip() for name in names),
                     f"ledger.bench.{sn_name}.rep{rep}: phase names must be nonempty")
            if phase_names is None:
                phase_names = names
            else:
                _require(names == phase_names,
                         f"ledger.bench.{sn_name}.rep{rep}: inconsistent phase set")
            for name, total in phases.items():
                _require(isinstance(total, dict),
                         f"ledger.bench.{sn_name}.rep{rep}.{name}: expected object")
                count = total.get("count")
                total_ms = total.get("total_ms")
                _require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                         f"ledger.bench.{sn_name}.rep{rep}.{name}.count: expected integer >= 0")
                _require(_is_number(total_ms) and float(total_ms) >= 0,
                         f"ledger.bench.{sn_name}.rep{rep}.{name}.total_ms: expected finite number >= 0")
            by_rep[rep] = phases
        _require(set(by_rep) == expected_reps,
                 f"ledger.bench.{sn_name}.phase_totals: expected reps 0..5")

        warm_phase_medians = {}
        for name in sorted(phase_names or ()):
            warm = [by_rep[rep][name] for rep in range(1, 6)]
            warm_phase_medians[name] = {
                "median_count": float(statistics.median(item["count"] for item in warm)),
                "median_total_ms": float(statistics.median(item["total_ms"] for item in warm)),
            }
        top_spans = [
            {"phase": name, **values}
            for name, values in sorted(
                warm_phase_medians.items(),
                key=lambda item: (-item[1]["median_total_ms"], item[0]),
            )[:TOP_PHASE_COUNT]
        ]
        summaries[sn_name] = {
            "warm_reps": list(range(1, 6)),
            "phase_medians": warm_phase_medians,
            "top_spans": top_spans,
        }
    return summaries


def _validate_source(qualification: dict[str, Any]) -> dict[str, Any]:
    source = qualification.get("source")
    _require(isinstance(source, dict), "source: missing object")
    result: dict[str, Any] = {}
    for name in ("stwo", "stwo_cairo"):
        repository = source.get(name)
        _require(isinstance(repository, dict), f"source.{name}: missing object")
        _require(_is_git_head(repository.get("head")), f"source.{name}.head: invalid")
        _require(_is_sha256(repository.get("worktree_hash")),
                 f"source.{name}.worktree_hash: invalid")
        result[name] = {
            "head": repository["head"],
            "worktree_hash": repository["worktree_hash"],
        }
    return result


def _validate_inputs(
    qualification: dict[str, Any], round_dir: Path, hashes: dict[str, str]
) -> dict[str, Any]:
    inputs = qualification.get("inputs")
    _require(isinstance(inputs, dict), "inputs: missing object")
    raw_files = inputs.get("files")
    adapted_files = inputs.get("adapted_files")
    _require(isinstance(raw_files, dict) and set(raw_files) == set(RAW_INPUT_NAMES),
             "inputs.files: expected exactly the four SN PIEs and bootloader")
    _require(isinstance(adapted_files, dict) and set(adapted_files) == set(ADAPTED_INPUT_NAMES),
             "inputs.adapted_files: expected exactly four adapted inputs")
    _require(all(_is_sha256(value) for value in raw_files.values()), "inputs.files: invalid hash")
    _require(all(_is_sha256(value) for value in adapted_files.values()),
             "inputs.adapted_files: invalid hash")
    _require(raw_files == _parse_checksum_manifest(
        CANONICAL_RAW_MANIFEST_PATH, "canonical release raw manifest"
    ), "inputs.files: differ from canonical release manifest")
    _require(adapted_files == _parse_checksum_manifest(
        CANONICAL_ADAPTED_MANIFEST_PATH, "canonical release adapted manifest"
    ), "inputs.adapted_files: differ from canonical release manifest")

    manifest = _verified_artifact(
        round_dir,
        {"path": inputs.get("manifest"), "sha256": inputs.get("manifest_sha256")},
        "inputs.raw_manifest",
        hashes,
    )
    _require(_parse_checksum_manifest(manifest, "inputs.raw_manifest") == raw_files,
             "inputs.raw_manifest: contents do not match qualification")
    adapted_manifest = _verified_artifact(
        round_dir,
        {"path": inputs.get("adapted_manifest"), "sha256": inputs.get("adapted_manifest_sha256")},
        "inputs.adapted_manifest",
        hashes,
    )
    _require(_parse_checksum_manifest(adapted_manifest, "inputs.adapted_manifest") == adapted_files,
             "inputs.adapted_manifest: contents do not match qualification")
    pinned_manifest = _verified_artifact(
        round_dir,
        {
            "path": inputs.get("pinned_adapted_manifest"),
            "sha256": inputs.get("pinned_adapted_manifest_sha256"),
        },
        "inputs.pinned_adapted_manifest",
        hashes,
    )
    _require(_parse_checksum_manifest(
        pinned_manifest, "inputs.pinned_adapted_manifest"
    ) == adapted_files,
             "inputs.pinned_adapted_manifest: contents do not match adapted inputs")
    _require(inputs.get("gate") == "SN_PIE_2.zip", "inputs.gate: expected SN_PIE_2.zip")
    return {"raw_files": dict(sorted(raw_files.items())),
            "adapted_files": dict(sorted(adapted_files.items()))}


def _validate_preflight(
    summary: Any,
    *,
    label: str,
    ceiling: Any,
    expected_policy: dict[str, Any],
    expected_input_name: str,
    adapted_hashes: dict[str, str],
    round_dir: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    _require(isinstance(summary, dict), f"{label}: missing summary")
    arena_bytes = summary.get("arena_bytes")
    _require(isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0,
             f"{label}: invalid preflight ceiling")
    _require(isinstance(arena_bytes, int) and not isinstance(arena_bytes, bool)
             and 0 < arena_bytes <= ceiling, f"{label}: arena exceeds ceiling")
    _require(summary.get("runtime_policy") == expected_policy,
             f"{label}: summary policy differs from qualification contract")
    artifact = _verified_artifact(
        round_dir,
        {"path": summary.get("artifact"), "sha256": summary.get("artifact_sha256")},
        f"{label}.artifact",
        hashes,
    )
    adapted = _resolve_artifact(round_dir, summary.get("adapted_input"), f"{label}.adapted_input")
    adapted_name = adapted.name
    _require(adapted_name == expected_input_name and adapted_name in adapted_hashes,
             f"{label}: unexpected adapted input")
    adapted_sha = _sha256(adapted)
    _require(adapted_sha == summary.get("adapted_input_sha256") == adapted_hashes[adapted_name],
             f"{label}: adapted input hash mismatch")
    hashes[f"{label}.adapted_input"] = adapted_sha

    record = json.loads(artifact.read_text(encoding="utf-8"))
    _require(record.get("pass") is True and record.get("vram_fit") is True,
             f"{label}: preflight did not pass")
    _require(record.get("vram_budget_bytes") == ceiling, f"{label}: preflight ceiling drift")
    _require((record.get("arena") or {}).get("total_bytes") == arena_bytes,
             f"{label}: preflight arena drift")
    _require(record.get("runtime_policy") == expected_policy,
             f"{label}: artifact policy differs from qualification contract")
    source = _resolve_artifact(round_dir, record.get("source"), f"{label}.source")
    _require(source == adapted, f"{label}: artifact source differs from adapted input")
    aot = record.get("aot_coverage")
    _require(isinstance(aot, dict) and set(aot) == AOT_PREFLIGHT_KEYS,
             f"{label}: AOT coverage shape mismatch")
    occurrences = aot.get("required_occurrences")
    _require(aot.get("pass") is True and aot.get("missing_occurrences") == []
             and aot.get("missing_occurrence_count") == 0,
             f"{label}: AOT coverage did not pass")
    _require(isinstance(occurrences, list) and occurrences
             and all(_valid_aot_occurrence(item) for item in occurrences)
             and occurrences == sorted(occurrences, key=_aot_occurrence_sort_key),
             f"{label}: AOT occurrence ledger is not canonical")
    unique_keys = sorted({item["cache_key"] for item in occurrences})
    _require(aot.get("required_occurrence_count") == len(occurrences)
             and aot.get("required_unique_key_count") == len(unique_keys),
             f"{label}: AOT coverage counts drifted")
    summary_aot = summary.get("aot_coverage")
    expected_summary_aot = {
        "manifest": aot.get("manifest"),
        "manifest_blake3": aot.get("manifest_blake3"),
        "manifest_entries": aot.get("manifest_entries"),
        "required_occurrence_count": len(occurrences),
        "required_unique_key_count": len(unique_keys),
        "required_occurrences_blake3": aot.get("required_occurrences_blake3"),
        "required_unique_keys_blake3": aot.get("required_unique_keys_blake3"),
        "missing_occurrence_count": 0,
    }
    _require(summary_aot == expected_summary_aot,
             f"{label}: AOT summary differs from preflight artifact")
    return {
        "arena_bytes": arena_bytes,
        "arena_gib": arena_bytes / 1024**3,
        "aot_occurrences": occurrences,
        "aot_coverage": expected_summary_aot,
    }


def _validate_qualification(
    qualification_path: Path, *, allow_dry_run: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    qualification_path = qualification_path.resolve()
    round_dir = qualification_path.parent
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"qualification: cannot read JSON: {error}") from error
    _require(isinstance(qualification, dict), "qualification: expected object")
    _require(qualification.get("schema") == QUALIFICATION_SCHEMA,
             f"qualification.schema: expected {QUALIFICATION_SCHEMA}")

    status = qualification.get("status")
    dry_run = qualification.get("dry_run")
    if status == "passed":
        _require(dry_run is False and qualification.get("performance_admissible") is True,
                 "passed qualification must be performance-admissible and non-dry")
    elif status == "dry_run":
        _require(allow_dry_run, "dry-run qualification requires --allow-dry-run")
        _require(dry_run is True and qualification.get("performance_admissible") is False,
                 "dry-run qualification must be performance-inadmissible")
    else:
        raise ReportError(f"qualification.status: refusing {status!r}")
    _require(qualification.get("runtime_mode") == "arena-graph",
             "qualification.runtime_mode: expected arena-graph")
    gpu = qualification.get("gpu")
    expected_gpu = "DRY-RUN-GPU" if dry_run else "NVIDIA H100 80GB HBM3"
    _require(gpu == expected_gpu,
             f"qualification.gpu: expected {expected_gpu}")
    source = _validate_source(qualification)
    _require(qualification["source"].get("sync") == {
        "method": "rsync checksum with target and result exclusions",
        "release_requires_clean_commits": True,
    }, "source.sync: release projection contract mismatch")
    if status == "passed":
        _require(all(repository["worktree_hash"] == EMPTY_WORKTREE_SHA256
                     for repository in source.values()),
                 "source: release qualification requires clean commits")

    hashes = {"qualification": _sha256(qualification_path)}
    inputs = _validate_inputs(qualification, round_dir, hashes)
    local_admission_path = _verified_artifact(
        round_dir, qualification.get("local_admission"), "local_admission", hashes
    )
    soundness_reference = qualification.get("soundness")
    soundness_path = _verified_artifact(
        round_dir, soundness_reference, "soundness", hashes
    )
    ledgers = qualification.get("ledgers")
    _require(isinstance(ledgers, dict) and set(ledgers) == {"bench", "ab"},
             "ledgers: expected bench and ab")
    bench_ledger_path = _verified_artifact(
        round_dir, ledgers["bench"], "ledger.bench", hashes
    )
    ab_ledger_path = _verified_artifact(
        round_dir, ledgers["ab"], "ledger.ab", hashes
    )

    target_wrapper = qualification.get("remote_execution_target")
    _require(isinstance(target_wrapper, dict), "remote_execution_target: missing")
    target = target_wrapper.get("target")
    _require(isinstance(target, dict), "remote_execution_target.target: missing")
    _require(target.get("schema") == "stwo.remote-execution-target.v1",
             "remote_execution_target: wrong schema")
    for field in ("pod_id", "boot_id", "gpu_uuid", "gpu_name"):
        _require(isinstance(target.get(field), str) and target[field],
                 f"remote_execution_target.{field}: missing")
    _require(target.get("gpu_name") == gpu, "remote execution GPU differs from benchmark GPU")
    binary = target.get("gpu_bench")
    _require(isinstance(binary, dict) and isinstance(binary.get("path"), str)
             and _is_sha256(binary.get("sha256")), "remote gpu_bench seal is invalid")
    _require(binary["path"]
             == f"/workspace/bench_loop_runs/sealed/gpu_bench.{binary['sha256']}",
             "remote gpu_bench path differs from release harness")
    target_sha = _canonical_sha256(target)
    _require(target_wrapper.get("sha256") == target_sha,
             "remote execution target canonical hash mismatch")
    _require(target_wrapper.get("postcheck") is True, "remote execution postcheck failed")
    _require(target_wrapper.get("remote_quiescence_passed") is True,
             "remote execution was not quiescent")
    _require(target_wrapper.get("ledger_entries_guarded") == 7
             and target_wrapper.get("quiescent_ledger_entries") == 7
             and target_wrapper.get("quiescent_measurement_launches") == 9,
             "remote execution guard counts differ from release round")
    expected_remote_inputs = {
        "gate": inputs["raw_files"]["SN_PIE_2.zip"],
        "bootloader": inputs["raw_files"]["simple_bootloader_compiled.json"],
        **{name: inputs["raw_files"][f"{name}.zip"] for name in SN_NAMES},
    }
    remote_inputs = target.get("inputs")
    _require(isinstance(remote_inputs, dict) and set(remote_inputs) == set(expected_remote_inputs),
             "remote execution inputs are incomplete")
    for name, expected_sha in expected_remote_inputs.items():
        item = remote_inputs[name]
        _require(isinstance(item, dict) and isinstance(item.get("path"), str)
                 and item.get("path") == REMOTE_INPUT_PATHS[name]
                 and item.get("sha256") == expected_sha,
                 f"remote execution input {name}: mismatch")

    projection = target_wrapper.get("source_projection")
    _require(isinstance(projection, dict) and projection.get("source") == source
             and projection.get("verified_after_soundness") is True,
             "remote source projection is invalid")

    soundness = json.loads(soundness_path.read_text(encoding="utf-8"))
    soundness_errors = validate_qualification_soundness_gate(
        soundness, expected_source=source, expected_dry_run=bool(dry_run)
    )
    _require(not soundness_errors,
             f"soundness qualification contract: {'; '.join(soundness_errors)}")
    _require(soundness.get("schema") == "stwo.cuda.soundness-gate.v3"
             and soundness.get("passed") is True, "soundness artifact did not pass")
    _require(soundness.get("runtime_mode") == "arena-graph"
             and soundness.get("dry_run") is dry_run, "soundness runtime/dry-run mismatch")
    _require(soundness.get("execution_target") == target
             and soundness.get("execution_target_sha256") == target_sha
             and soundness.get("execution_target_postcheck") is True,
             "soundness execution target mismatch")
    _require(soundness.get("source_projection") == projection,
             "soundness source projection mismatch")
    _require(soundness.get("synced_source") == {
        "stwo": source["stwo"],
        "stwo_cairo": source["stwo_cairo"],
        "transport": "rsync-archive-checksum",
    }, "soundness synced source mismatch")
    _require(isinstance(soundness_reference, dict)
             and soundness_reference.get("effective_stwo_env")
             == soundness.get("effective_stwo_env")
             and soundness_reference.get("stwo_worktree_hash")
             == source["stwo"]["worktree_hash"]
             and soundness_reference.get("stwo_cairo_worktree_hash")
             == source["stwo_cairo"]["worktree_hash"],
             "soundness wrapper is not bound to source and effective environment")

    admission = json.loads(local_admission_path.read_text(encoding="utf-8"))
    _require(admission.get("schema") == "stwo.local-preflight-admission.v1"
             and admission.get("passed") is True, "local admission did not pass")
    _require(admission.get("dry_run") is dry_run and admission.get("runtime_mode") == "arena-graph",
             "local admission mode mismatch")
    _require(admission.get("source") == source, "local admission source mismatch")

    profiles = qualification.get("profiles")
    _require(isinstance(profiles, dict) and set(profiles) == {
        "universal_sn1_sn4", "sn2_headline"
    }, "profiles: expected exactly universal and SN2 headline")
    universal = profiles.get("universal_sn1_sn4")
    headline = profiles.get("sn2_headline")
    _require(isinstance(universal, dict) and isinstance(headline, dict), "profiles: incomplete")
    _require(universal.get("env") == QUALIFICATION_PROFILES["universal_sn1_sn4"]
             and universal.get("effective_state")
             == QUALIFICATION_NORMALIZED_STATES["universal_sn1_sn4"],
             "profiles.universal_sn1_sn4: noncanonical environment or state")
    _require(headline.get("env") == QUALIFICATION_PROFILES["sn2_headline"]
             and headline.get("effective_state")
             == QUALIFICATION_NORMALIZED_STATES["sn2_headline"],
             "profiles.sn2_headline: noncanonical environment or state")
    _require(qualification.get("normalized_states") == QUALIFICATION_NORMALIZED_STATES,
             "normalized_states: noncanonical qualification state")
    universal_ceiling = universal.get("preflight_ceiling_bytes")
    _require(universal_ceiling == PREFLIGHT_CAP_BYTES,
             "preflight ceiling differs from qualification contract")
    universal_preflights = universal.get("preflights")
    _require(isinstance(universal_preflights, dict) and set(universal_preflights) == set(SN_NAMES),
             "universal preflights: expected SN1-SN4")
    preflights = {
        name: _validate_preflight(
            universal_preflights[name], label=f"preflight.universal.{name}",
            ceiling=universal_ceiling, expected_policy=UNIVERSAL_POLICY,
            expected_input_name=f"{name}.adapted.bin",
            adapted_hashes=inputs["adapted_files"],
            round_dir=round_dir, hashes=hashes,
        )
        for name in SN_NAMES
    }
    preflights["SN2_headline"] = _validate_preflight(
        headline.get("preflight"), label="preflight.headline.SN_PIE_2",
        ceiling=headline.get("preflight_ceiling_bytes"), expected_policy=HEADLINE_POLICY,
        expected_input_name="SN_PIE_2.adapted.bin",
        adapted_hashes=inputs["adapted_files"],
        round_dir=round_dir, hashes=hashes,
    )
    preflights["SN2_flags_off"] = _validate_preflight(
        qualification.get("flags_off_preflight"), label="preflight.flags_off.SN_PIE_2",
        ceiling=universal_ceiling, expected_policy=FLAGS_OFF_POLICY,
        expected_input_name="SN_PIE_2.adapted.bin",
        adapted_hashes=inputs["adapted_files"],
        round_dir=round_dir, hashes=hashes,
    )
    expected_preflight_hashes = {
        Path(universal_preflights[name]["artifact"]).name:
            universal_preflights[name]["artifact_sha256"]
        for name in SN_NAMES
    }
    expected_preflight_hashes.update({
        Path(headline["preflight"]["artifact"]).name:
            headline["preflight"]["artifact_sha256"],
        Path(qualification["flags_off_preflight"]["artifact"]).name:
            qualification["flags_off_preflight"]["artifact_sha256"],
    })
    _require(admission.get("preflight_ceiling_bytes") == universal_ceiling
             == headline.get("preflight_ceiling_bytes"),
             "local admission preflight ceiling mismatch")
    _require(admission.get("preflight_artifact_sha256") == expected_preflight_hashes,
             "local admission preflight hashes mismatch")
    _require(admission.get("profiles") == QUALIFICATION_PROFILES,
             "local admission profiles mismatch")
    aot_admission = admission.get("aot_coverage")
    _require(isinstance(aot_admission, dict)
             and set(aot_admission) == AOT_ADMISSION_KEYS,
             "local admission AOT coverage contract is not exact")
    _require(qualification.get("aot_coverage") == aot_admission,
             "qualification AOT coverage differs from local admission")
    aot_manifest = _resolve_artifact(
        round_dir, aot_admission.get("manifest"), "aot_coverage.manifest"
    )
    manifest_sha = _sha256(aot_manifest)
    _require(manifest_sha == aot_admission.get("manifest_sha256"),
             "AOT manifest SHA-256 mismatch")
    hashes["aot_manifest"] = manifest_sha
    try:
        manifest_entries = json.loads(aot_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"AOT manifest: cannot read JSON: {error}") from error
    _require(isinstance(manifest_entries, list) and manifest_entries,
             "AOT manifest: expected nonempty array")
    manifest_by_key = {}
    for index, entry in enumerate(manifest_entries):
        _require(isinstance(entry, dict) and set(entry) == AOT_MANIFEST_ENTRY_KEYS
                 and entry.get("kind") in {"constraint", "witness"}
                 and isinstance(entry.get("label"), str) and entry["label"]
                 and isinstance(entry.get("kernel_name"), str) and entry["kernel_name"]
                 and isinstance(entry.get("file"), str) and entry["file"]
                 and isinstance(entry.get("cache_key"), str)
                 and len(entry["cache_key"]) == 16
                 and all(char in "0123456789abcdef" for char in entry["cache_key"])
                 and isinstance(entry.get("semantic_hash"), str)
                 and len(entry["semantic_hash"]) == 16
                 and all(char in "0123456789abcdef" for char in entry["semantic_hash"]),
                 f"AOT manifest: malformed entry {index}")
        _require(entry["cache_key"] not in manifest_by_key,
                 f"AOT manifest: duplicate key {entry['cache_key']}")
        manifest_by_key[entry["cache_key"]] = entry
    _require(aot_admission.get("manifest_entry_count") == len(manifest_by_key)
             and _is_sha256(aot_admission.get("manifest_blake3")),
             "AOT manifest identity/count mismatch")

    universal_aot_artifacts = {
        Path(universal_preflights[f"SN_PIE_{number}"]["artifact"]).name:
            preflights[f"SN_PIE_{number}"]
        for number in range(1, 5)
    }
    flags_aot_artifact = Path(qualification["flags_off_preflight"]["artifact"]).name
    headline_aot_artifact = Path(headline["preflight"]["artifact"]).name
    sn2_universal_aot_artifact = Path(
        universal_preflights["SN_PIE_2"]["artifact"]
    ).name
    aot_preflights = {
        **universal_aot_artifacts,
        flags_aot_artifact: preflights["SN2_flags_off"],
        headline_aot_artifact: preflights["SN2_headline"],
    }
    occurrence_sha = {}
    occurrence_blake3 = {}
    key_blake3 = {}
    all_occurrences = []
    for artifact_name, preflight in aot_preflights.items():
        coverage = preflight["aot_coverage"]
        occurrences = preflight["aot_occurrences"]
        _require(_resolve_artifact(
            round_dir, coverage["manifest"], f"{artifact_name}.aot_manifest"
        ) == aot_manifest
                 and coverage["manifest_blake3"] == aot_admission["manifest_blake3"]
                 and coverage["manifest_entries"] == len(manifest_by_key),
                 f"{artifact_name}: AOT manifest binding mismatch")
        for occurrence in occurrences:
            entry = manifest_by_key.get(occurrence["cache_key"])
            _require(entry is not None
                     and entry["kind"] == occurrence["kind"]
                     and entry["kernel_name"] == occurrence["kernel_name"]
                     and entry["semantic_hash"] == occurrence["semantic_hash"],
                     f"{artifact_name}: uncovered AOT key {occurrence['cache_key']}")
        occurrence_sha[artifact_name] = _canonical_sha256(occurrences)
        occurrence_blake3[artifact_name] = coverage["required_occurrences_blake3"]
        key_blake3[artifact_name] = coverage["required_unique_keys_blake3"]
        all_occurrences.extend(occurrences)
    _require(occurrence_sha[flags_aot_artifact]
             == occurrence_sha[sn2_universal_aot_artifact]
             == occurrence_sha[headline_aot_artifact],
             "SN2 AOT keys vary across runtime-only profiles")
    by_record = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in all_occurrences
    }
    occurrence_union = [by_record[key] for key in sorted(by_record)]
    union_keys = sorted({item["cache_key"] for item in occurrence_union})
    _require(aot_admission.get("required_occurrence_count") == len(occurrence_union)
             and aot_admission.get("required_unique_key_count") == len(union_keys)
             and aot_admission.get("required_unique_keys") == union_keys
             and aot_admission.get("required_occurrences_sha256")
             == _canonical_sha256(occurrence_union)
             and aot_admission.get("required_unique_keys_sha256")
             == _canonical_sha256(union_keys)
             and aot_admission.get("preflight_occurrences_sha256") == occurrence_sha
             and aot_admission.get("preflight_occurrences_blake3") == occurrence_blake3
             and aot_admission.get("preflight_unique_keys_blake3") == key_blake3
             and aot_admission.get("missing_occurrences") == [],
             "local admission AOT union/hash contract mismatch")
    expected_per_sn = {}
    for number in range(1, 5):
        occurrences = preflights[f"SN_PIE_{number}"]["aot_occurrences"]
        keys = sorted({item["cache_key"] for item in occurrences})
        expected_per_sn[f"SN_PIE_{number}"] = {
            "required_occurrence_count": len(occurrences),
            "required_unique_key_count": len(keys),
            "required_occurrences_sha256": _canonical_sha256(occurrences),
            "required_unique_keys_sha256": _canonical_sha256(keys),
        }
    _require(aot_admission.get("per_sn") == expected_per_sn,
             "local admission AOT per-SN contract mismatch")
    aot_index_path = _verified_artifact(
        round_dir, qualification.get("aot_index_check"), "aot_index_check", hashes
    )
    aot_index = json.loads(aot_index_path.read_text(encoding="utf-8"))
    checker = aot_index.get("checker_binary") if isinstance(aot_index, dict) else None
    _require(isinstance(aot_index, dict)
             and aot_index.get("schema") == "stwo.aot-index-check.v1"
             and aot_index.get("pass") is True
             and aot_index.get("dry_run") is dry_run
             and aot_index.get("sm") == 90
             and aot_index.get("missing_keys") == []
             and aot_index.get("source") == source
             and aot_index.get("required_unique_key_count") == len(union_keys)
             and aot_index.get("required_unique_keys_sha256")
             == aot_admission.get("required_unique_keys_sha256")
             and isinstance(aot_index.get("loaded_manifest_hash"), str)
             and len(aot_index["loaded_manifest_hash"]) == 16
             and aot_index["loaded_manifest_hash"] != "0" * 16
             and all(char in "0123456789abcdef"
                     for char in aot_index["loaded_manifest_hash"])
             and isinstance(checker, dict)
             and isinstance(checker.get("path"), str)
             and checker["path"].endswith("/target/release/aot_index_check")
             and _is_sha256(checker.get("sha256")),
             "post-build AOT index check is invalid")
    embedded_manifest_hash = int(aot_index["loaded_manifest_hash"], 16)
    _require(admission.get("adapted_input_manifest_sha256")
             == qualification["inputs"]["adapted_manifest_sha256"],
             "local admission adapted manifest mismatch")
    reproduction = admission.get("adapter_reproduction")
    _require(isinstance(reproduction, dict)
             and reproduction == qualification.get("adapter_reproduction")
             and set(reproduction) == {
                 "byte_equal", "gpu_bench_binary_sha256",
                 "raw_input_manifest_sha256", "bootloader_sha256",
                 "pinned_adapted_manifest_sha256",
             }
             and reproduction.get("byte_equal") is True
             and reproduction.get("raw_input_manifest_sha256")
             == qualification["inputs"]["manifest_sha256"]
             and reproduction.get("bootloader_sha256")
             == inputs["raw_files"]["simple_bootloader_compiled.json"]
             and reproduction.get("pinned_adapted_manifest_sha256")
             == qualification["inputs"]["pinned_adapted_manifest_sha256"]
             and _is_sha256(reproduction.get("gpu_bench_binary_sha256"))
             and _is_sha256(reproduction.get("bootloader_sha256")),
             "local adapter reproduction binding mismatch")

    benchmarks_raw = qualification.get("benchmarks")
    _require(isinstance(benchmarks_raw, dict) and set(benchmarks_raw) == set(SN_NAMES),
             "benchmarks: expected exactly SN1-SN4")
    for name in SN_NAMES:
        record = benchmarks_raw[name]
        _require(isinstance(record, dict)
                 and record.get("gpu_aot_manifest_hash") == embedded_manifest_hash,
                 f"benchmarks.{name}: embedded AOT manifest hash mismatch")
    benchmarks = {
        name: _validate_measurement(
            benchmarks_raw[name], expected_program=f"{name}.zip", expected_gpu=gpu,
            label=f"benchmarks.{name}", require_fresh_simd_reference=True,
        )
        for name in SN_NAMES
    }
    gpu_telemetry = _validate_gpu_telemetry(
        qualification, benchmarks_raw, round_dir, hashes
    )

    proof_hashes = qualification.get("proof_sha256")
    _require(isinstance(proof_hashes, dict), "proof_sha256: missing")
    fixed_hashes = proof_hashes.get("fixed_pies")
    _require(isinstance(fixed_hashes, dict) and set(fixed_hashes) == set(SN_NAMES)
             and all(_is_sha256(value) for value in fixed_hashes.values()),
             "proof_sha256.fixed_pies: invalid")
    ab_hashes = ((proof_hashes.get("ab") or {}).get("sn2_headline") or {})
    _require(set(ab_hashes) == {"flags_off", "headline"}
             and all(_is_sha256(value) for value in ab_hashes.values())
             and ab_hashes["flags_off"] == ab_hashes["headline"],
             "proof_sha256.ab.sn2_headline: arms are absent or unequal")
    _require(fixed_hashes["SN_PIE_2"] == ab_hashes["headline"],
             "proof_sha256: fixed SN2 and A/B proofs differ")
    phase_profiles = _validate_phase_ledger(
        bench_ledger_path,
        benchmarks_raw,
        expected_env=QUALIFICATION_PROFILES["universal_sn1_sn4"],
        expected_gpu=gpu,
        expected_proof_hashes=fixed_hashes,
        expected_telemetry=qualification["gpu_telemetry"],
        expected_soundness_sha256=hashes["soundness"],
        expected_target=target,
        expected_target_sha256=target_sha,
        expected_projection=projection,
    )
    ncu_profile = _validate_ncu_profile(
        qualification,
        dry_run=bool(dry_run),
        qualified_sn2_proof=fixed_hashes["SN_PIE_2"],
        round_dir=round_dir,
        hashes=hashes,
    )

    comparisons = qualification.get("comparisons")
    comparison = (comparisons or {}).get("sn2_flags_off_vs_headline")
    _require(isinstance(comparison, dict) and comparison.get("status") == "ok"
             and comparison.get("reps") == 6, "SN2 comparison is incomplete")
    _require(comparison.get("lane") == "sn2_headline"
             and comparison.get("bench_env") == QUALIFICATION_PROFILES["flags_off"]
             and comparison.get("candidate_env") == QUALIFICATION_PROFILES["sn2_headline"]
             and comparison.get("baseline_state")
             == QUALIFICATION_NORMALIZED_STATES["flags_off"]
             and comparison.get("candidate_state")
             == QUALIFICATION_NORMALIZED_STATES["sn2_headline"],
             "SN2 comparison environment or normalized state is noncanonical")
    _require(comparison.get("provisional") is True
             and comparison.get("performance_admissible") is False
             and comparison.get("pod_gpu") == gpu
             and (comparison.get("architecture_soundness") or {}).get("sha256")
             == hashes["soundness"]
             and comparison.get("execution_target") == target
             and comparison.get("execution_target_sha256") == target_sha
             and comparison.get("source_projection") == projection
             and comparison.get("execution_guard_passed") is True
             and comparison.get("remote_quiescence_passed") is True,
             "SN2 comparison escaped the qualified execution seal")
    raw_ncu = ((qualification.get("profiling") or {}).get("ncu") or {}).get("profile")
    _require(comparison.get("ncu_profile_required") is True
             and comparison.get("ncu_profile_requested") is True
             and comparison.get("ncu_profile_attempted") is True
             and comparison.get("ncu_profile_status") == "validated"
             and comparison.get("ncu_profile") == raw_ncu,
             "SN2 comparison differs from the qualified NCU capture")
    comparison_results: dict[str, Any] = {}
    for arm_name, hash_name in (("baseline", "flags_off"), ("flagged", "headline")):
        arm = comparison.get(arm_name)
        _require(isinstance(arm, dict) and arm.get("proof_sha256") == ab_hashes[hash_name],
                 f"SN2 comparison {arm_name}: proof hash mismatch")
        _require(isinstance(arm.get("record"), dict)
                 and arm["record"].get("gpu_aot_manifest_hash")
                 == embedded_manifest_hash,
                 f"SN2 comparison {arm_name}: embedded AOT manifest hash mismatch")
        measured = _validate_measurement(
            arm.get("record"), expected_program="SN_PIE_2.zip", expected_gpu=gpu,
            label=f"comparisons.sn2.{arm_name}",
        )
        _require(arm.get("useful_mhz_median") == measured["useful_mhz_median"],
                 f"SN2 comparison {arm_name}: metric drift")
        comparison_results[arm_name] = measured
    ratio = (
        comparison_results["flagged"]["useful_mhz_median"]
        / comparison_results["baseline"]["useful_mhz_median"]
    )
    reported_ratio = comparison.get("useful_mhz_ratio")
    _require(_is_number(reported_ratio) and abs(float(reported_ratio) - ratio) <= 1e-12,
             "SN2 comparison ratio drift")
    ab_entries = _load_jsonl(ab_ledger_path, "ledger.ab")
    _require(len(ab_entries) == 1, "ledger.ab: expected exactly one SN2 A/B entry")
    _require(comparison == {**ab_entries[0], "useful_mhz_ratio": ratio},
             "SN2 comparison differs from authoritative A/B ledger")

    measured_identity = status == "passed"
    identity = {
        "established": measured_identity,
        "status": (
            "fresh_reference_byte_equal" if measured_identity else "synthetic_not_measured"
        ),
        "scope": (
            "Fresh verified SIMD proof bytes equal every GPU repetition for each actual SN1-SN4 input"
            if measured_identity
            else "Dry-run SIMD fields are synthetic contract fixtures and establish no actual correctness or identity claim"
        ),
    }
    if measured_identity:
        identity["references"] = {
            name: benchmarks[name]["simd_reference"] for name in SN_NAMES
        }

    report = {
        "schema": REPORT_SCHEMA,
        "qualification_status": status,
        "performance_admissible": status == "passed",
        "claim_basis": {
            "fixed_statements": "pie_n_steps / median(post-cold prove seconds) / 1e6",
            "warm_quantile": "R-7",
            "legacy_warm_best_is_excluded": True,
        },
        "proof_identity": {
            "gpu_repeat_determinism": {
                "established": measured_identity,
                "status": "measured_byte_equal" if measured_identity else "synthetic_not_measured",
                "scope": (
                    "Each fixed SN statement and each SN2 A/B arm repeated on one sealed GPU target"
                    if measured_identity
                    else "Dry-run repetition fields are synthetic contract fixtures"
                ),
            },
            "actual_sn_simd_byte_identity": identity,
        },
        "gpu": {
            "name": gpu,
            "uuid": target["gpu_uuid"],
            "pod_id": target["pod_id"],
            "boot_id": target["boot_id"],
            "binary_sha256": binary["sha256"],
            "execution_target_sha256": target_sha,
        },
        "source": source,
        "runtime_mode": "arena-graph",
        "security": dict(EXPECTED_SECURITY),
        "inputs": inputs,
        "benchmarks": benchmarks,
        "sn2_flags_off_vs_headline": {
            "flags_off": comparison_results["baseline"],
            "headline": comparison_results["flagged"],
            "useful_mhz_ratio": ratio,
            "proof_sha256": ab_hashes["headline"],
        },
        "proof_sha256": {
            "fixed_pies": dict(sorted(fixed_hashes.items())),
            "sn2_ab": ab_hashes["headline"],
        },
        "preflights": preflights,
        "profiling": {
            "status": (
                "measured_ledger_phase_totals"
                if status == "passed"
                else "synthetic_ledger_phase_totals"
            ),
            "performance_admissible": status == "passed",
            "fixed_sn": phase_profiles,
            "gpu_telemetry": gpu_telemetry,
            "ncu": ncu_profile,
            "nested_spans_warning": (
                "Tracing spans may be nested; phase medians are diagnostic and must not "
                "be summed or interpreted as proof wall time."
            ),
            "claim": (
                "Phase timing data alone imply no kernel result; the separately bound "
                "targeted NCU capture is reported independently"
            ),
        },
        "artifact_sha256": dict(sorted(hashes.items())),
    }
    return report, qualification


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _render_markdown(report: dict[str, Any]) -> str:
    dry = not report["performance_admissible"]
    lines = ["# STWO GPU Prover Benchmark Report", ""]
    if dry:
        lines.extend([
            "> **DRY RUN — PERFORMANCE-INADMISSIBLE.** These values test the reporting",
            "> contract only and must not be published as measured GPU performance.",
            "",
        ])
    lines.extend([
        f"Qualification status: `{report['qualification_status']}`  ",
        f"Runtime: `{report['runtime_mode']}`  ",
        f"GPU: `{report['gpu']['name']}` (`{report['gpu']['uuid']}`)  ",
        f"Execution target: `{report['gpu']['execution_target_sha256']}`",
        "",
        "## Fixed-statement results",
        "",
        "The claim metric is `pie_n_steps / median(post-cold prove seconds) / 1e6`.",
        "One cold repetition is excluded; all five warm repetitions are retained.",
        "",
        "| Input | Useful steps | Warm samples (s) | Median (s) | p95 (s) | Useful MHz median | Useful MHz at p95 | Verified | Peak VRAM (GB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in SN_NAMES:
        result = report["benchmarks"][name]
        samples = ", ".join(_fmt(value) for value in result["warm_samples_s"])
        lines.append(
            f"| {name} | {result['pie_n_steps']} | {samples} | "
            f"{_fmt(result['warm_median_s'])} | {_fmt(result['warm_p95_s'])} | "
            f"{_fmt(result['useful_mhz_median'])} | "
            f"{_fmt(result['useful_mhz_at_warm_p95'])} | {result['verified_reps']}/{result['reps']} | "
            f"{_fmt(result['vram_peak_gb'], 2)} |"
        )

    comparison = report["sn2_flags_off_vs_headline"]
    lines.extend([
        "",
        "## SN2 flags-off versus headline profile",
        "",
        "| Arm | Warm median (s) | Useful MHz median | Useful MHz at p95 | Peak VRAM (GB) |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in (("Flags off", "flags_off"), ("Headline", "headline")):
        result = comparison[key]
        lines.append(
            f"| {label} | {_fmt(result['warm_median_s'])} | "
            f"{_fmt(result['useful_mhz_median'])} | "
            f"{_fmt(result['useful_mhz_at_warm_p95'])} | "
            f"{_fmt(result['vram_peak_gb'], 2)} |"
        )
    lines.extend([
        "",
        f"Headline/flags-off useful-MHz ratio: **{comparison['useful_mhz_ratio']:.4f}x**.",
        "",
        "## Correctness and proof identity",
        "",
    ])
    identity = report["proof_identity"]["actual_sn_simd_byte_identity"]
    if identity["established"]:
        lines.extend([
            "- Every reported proof repetition passed the verifier.",
            "- GPU-repeat byte determinism is established for the fixed SN statements and SN2 A/B arms.",
            "- **Actual SN1-SN4 GPU-to-SIMD byte identity is established.**",
            "  Each fixed record computed and verified a fresh SIMD proof outside the timed GPU window,",
            "  and every GPU proof repetition had the same BLAKE3 digest and proof bytes.",
        ])
    else:
        lines.extend([
            "- **DRY-RUN SIMD FIELDS ARE SYNTHETIC.**",
            "  No GPU or SIMD proof execution is evidenced by these fixture records.",
            "  No actual correctness or GPU-to-SIMD identity claim is established by this report.",
        ])
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- STWO: `{report['source']['stwo']['head']}` / `{report['source']['stwo']['worktree_hash']}`",
        f"- STWO Cairo: `{report['source']['stwo_cairo']['head']}` / `{report['source']['stwo_cairo']['worktree_hash']}`",
        f"- Sealed binary: `{report['gpu']['binary_sha256']}`",
        f"- Pod/boot: `{report['gpu']['pod_id']}` / `{report['gpu']['boot_id']}`",
        f"- Security: {report['security']['security_bits']} bits, {report['security']['n_queries']} queries, "
        f"PoW {report['security']['pow_bits']}, fold step {report['security']['fold_step']}",
        "",
        "## Memory admission",
        "",
        "| Profile/input | Arena GiB |",
        "|---|---:|",
    ])
    for name, preflight in report["preflights"].items():
        gib = preflight.get("arena_gib")
        if not _is_number(gib):
            gib = preflight["arena_bytes"] / 1024**3
        lines.append(f"| {name} | {float(gib):.3f} |")
    lines.extend([
        "",
        "## Proof-loop-envelope GPU telemetry",
        "",
        "Samples are filtered strictly to each benchmark's proof-loop envelope. The envelope",
        "includes per-repetition input clones and phase-report gaps; it is not exact GPU-only time.",
        "Each metric cell is `mean / p95 / max` over retained in-envelope samples.",
        "",
        "| Input | Samples | GPU util (%) | Memory util (%) | Memory used (MiB) | Power (W) | SM clock (MHz) | Memory clock (MHz) | Temp (C) | Driver / limit / max clocks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name in SN_NAMES:
        telemetry = report["profiling"]["gpu_telemetry"][name]
        metrics = telemetry["metrics"]

        def triple(field: str) -> str:
            return " / ".join(
                _fmt(metrics[field][stat]) for stat in ("mean", "p95", "max")
            )

        hardware = telemetry["hardware"]
        lines.append(
            f"| {name} | {telemetry['proof_loop_sample_count']} | "
            f"{triple('utilization_gpu_pct')} | {triple('utilization_memory_pct')} | "
            f"{triple('memory_used_mib')} | {triple('power_draw_w')} | "
            f"{triple('clock_sm_mhz')} | {triple('clock_memory_mhz')} | "
            f"{triple('temperature_gpu_c')} | {hardware['driver_version']} / "
            f"{_fmt(hardware['power_limit_w'])} W / "
            f"{_fmt(hardware['clock_max_sm_mhz'])}, "
            f"{_fmt(hardware['clock_max_memory_mhz'])} MHz |"
        )
    lines.extend([
        "",
        "## Warm-repetition phase profile",
        "",
        report["profiling"]["nested_spans_warning"],
        "",
    ])
    for name in SN_NAMES:
        lines.extend([
            f"### {name}",
            "",
            "Top ledger spans by median `total_ms` across warm reps 1–5:",
            "",
            "| Phase | Median count | Median total (ms) |",
            "|---|---:|---:|",
        ])
        for span in report["profiling"]["fixed_sn"][name]["top_spans"]:
            lines.append(
                f"| {span['phase']} | {_fmt(span['median_count'], 1)} | "
                f"{_fmt(span['median_total_ms'])} |"
            )
        lines.append("")
    lines.extend([
        "No kernel-level result is inferred from these phase timings.",
        "",
        "## Targeted Nsight Compute profile",
        "",
    ])
    ncu = report["profiling"]["ncu"]
    lines.extend([
        f"- Status: `{ncu['status']}`; synthetic: `{str(ncu['synthetic']).lower()}`",
        f"- Capture: set `{ncu['set']}`, {ncu['launch_count']} launches, kernel regex `{ncu['kernel_regex']}`",
        f"- Profiled qualified SN2 proof: `{ncu['profiled_proof_sha256']}`",
        f"- Binary report: `{ncu['report']['path']}` (`{ncu['report']['sha256']}`)",
        f"- Raw CSV import: `{ncu['import_csv']['path']}` (`{ncu['import_csv']['sha256']}`)",
        "",
        "The NCU capture is a targeted kernel sample, not a whole-proof roofline, occupancy,",
        "or physics-bound claim.",
        "",
        "## Bound artifact hashes",
        "",
    ])
    for name, digest in sorted(report["artifact_sha256"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def generate_report(
    qualification: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str | None = None,
    *,
    allow_dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate a qualification and write report JSON, Markdown, and hashes."""

    supplied = Path(qualification).expanduser()
    qualification_path = supplied / "qualification.json" if supplied.is_dir() else supplied
    _require(qualification_path.is_file(), f"qualification does not exist: {qualification_path}")
    report, _ = _validate_qualification(qualification_path, allow_dry_run=allow_dry_run)
    destination = Path(output_dir).expanduser() if output_dir else qualification_path.parent / "report"
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "benchmark-report.json"
    markdown_path = destination / "benchmark-report.md"
    hashes_path = destination / "benchmark-report.sha256"
    for path in (json_path, markdown_path, hashes_path):
        _require(overwrite or not path.exists(), f"refusing to overwrite {path}")

    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = _render_markdown(report)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    output_hashes = {
        json_path.name: _sha256(json_path),
        markdown_path.name: _sha256(markdown_path),
    }
    hashes_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(output_hashes.items())),
        encoding="utf-8",
    )
    output_hashes[hashes_path.name] = _sha256(hashes_path)
    return {
        "performance_admissible": report["performance_admissible"],
        "files": {
            "json": str(json_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "hashes": str(hashes_path.resolve()),
        },
        "sha256": output_hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path,
                        help="qualification round directory or qualification.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-dry-run", action="store_true",
                        help="emit a visibly performance-inadmissible report for a dry fixture")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate_report(
            args.qualification,
            args.output_dir,
            allow_dry_run=args.allow_dry_run,
            overwrite=args.overwrite,
        )
    except ReportError as error:
        print(f"benchmark report rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
