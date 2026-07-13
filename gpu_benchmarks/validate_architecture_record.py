#!/usr/bin/env python3
"""Fail closed unless a gpu_bench record proves the typed CUDA PCS path ran."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from run_cuda_soundness_gate import (
    GATES as CUDA_SOUNDNESS_GATE_COMMANDS,
    QUALIFICATION_FLAGS,
    STRICT_RESIDENT_GATE,
    STRICT_RESIDENT_REQUIRED_TESTS,
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
GPU_TELEMETRY_SAMPLE_INTERVAL_MS = 250
GPU_TELEMETRY_MAX_GAP_NS = 2_000_000_000
GPU_TELEMETRY_MIN_WINDOW_SAMPLES = 2

PREFLIGHT_CAP_BYTES = 81_604_378_624
RETAINED_BUDGET_FLAG = "STWO_CUDA_RETAINED_LDE_BUDGET_BYTES"
UNIVERSAL_ENV = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 "
    "STWO_CUDA_B2N_STAGE_FUSED=1 "
    "STWO_CUDA_RETAINED_LDE_BUDGET_BYTES=4563402752"
)
SN2_HEADLINE_ENV = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 "
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION=1 "
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS=1 "
    "STWO_CUDA_B2N_STAGE_FUSED=1 "
    "STWO_CUDA_RETAINED_LDE_BUDGET_BYTES=29469326848"
)
QUALIFICATION_PROFILES = {
    "flags_off": "",
    "universal_sn1_sn4": UNIVERSAL_ENV,
    "sn2_headline": SN2_HEADLINE_ENV,
}

FLAGS_OFF_POLICY = {
    "commit_mode": "FullLifting",
    "direct_composition_retention_mode": "Disabled",
    "quotient_numerator_source_policy": "CoefficientsOnly",
    "interpolation_mode": "StageWiseCopyThenInPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 8_589_934_592,
}
UNIVERSAL_POLICY = {
    "commit_mode": "DomainProgressive",
    "direct_composition_retention_mode": "Disabled",
    "quotient_numerator_source_policy": "CoefficientsOnly",
    "interpolation_mode": "StageFusedOutOfPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 4_563_402_752,
}
HEADLINE_POLICY = {
    "commit_mode": "DomainProgressive",
    "direct_composition_retention_mode": "ExactNative",
    "quotient_numerator_source_policy": "ReuseRetainedEvaluations",
    "interpolation_mode": "StageFusedOutOfPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 29_469_326_848,
}
PREFLIGHT_SPECS = {
    "preflight_flags_off_SN2.json": ("SN_PIE_2.adapted.bin", FLAGS_OFF_POLICY),
    **{
        f"preflight_universal_SN{pie}.json": (
            f"SN_PIE_{pie}.adapted.bin",
            UNIVERSAL_POLICY,
        )
        for pie in range(1, 5)
    },
    "preflight_sn2_headline.json": ("SN_PIE_2.adapted.bin", HEADLINE_POLICY),
}
ADAPTER_REPRODUCTION_KEYS = {
    "byte_equal",
    "gpu_bench_binary_sha256",
    "raw_input_manifest_sha256",
    "bootloader_sha256",
    "pinned_adapted_manifest_sha256",
}
AOT_OCCURRENCE_KEYS = {
    "kind",
    "component",
    "instance",
    "kernel",
    "kernel_name",
    "semantic_hash",
    "cache_key",
}
AOT_PREFLIGHT_KEYS = {
    "pass",
    "manifest",
    "manifest_blake3",
    "manifest_entries",
    "required_occurrences",
    "required_occurrences_blake3",
    "required_occurrence_count",
    "required_unique_keys_blake3",
    "required_unique_key_count",
    "missing_occurrences",
    "missing_occurrence_count",
}
AOT_ADMISSION_KEYS = {
    "manifest",
    "manifest_sha256",
    "manifest_blake3",
    "manifest_entry_count",
    "required_occurrence_count",
    "required_unique_key_count",
    "required_unique_keys",
    "required_occurrences_sha256",
    "required_unique_keys_sha256",
    "preflight_occurrences_sha256",
    "preflight_occurrences_blake3",
    "preflight_unique_keys_blake3",
    "per_sn",
    "missing_occurrences",
}
AOT_MANIFEST_ENTRY_KEYS = {
    "kind", "label", "kernel_name", "cache_key", "semantic_hash", "file"
}
RAW_INPUT_NAMES = {
    *(f"SN_PIE_{pie}.zip" for pie in range(1, 5)),
    "simple_bootloader_compiled.json",
}
ADAPTED_INPUT_NAMES = {f"SN_PIE_{pie}.adapted.bin" for pie in range(1, 5)}


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


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_sha256(value: object) -> bool:
    return _is_lower_hex(value, 64)


def _valid_source_identity(source: object) -> bool:
    if not isinstance(source, dict) or set(source) != {"stwo", "stwo_cairo"}:
        return False
    for repo in ("stwo", "stwo_cairo"):
        value = source.get(repo)
        if (
            not isinstance(value, dict)
            or set(value) != {"head", "worktree_hash"}
            or not _is_lower_hex(value.get("head"), 40)
            or not _is_sha256(value.get("worktree_hash"))
        ):
            return False
    return True


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_aot_occurrence(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == AOT_OCCURRENCE_KEYS
        and value.get("kind") in {"constraint", "witness"}
        and isinstance(value.get("component"), str)
        and bool(value["component"])
        and isinstance(value.get("instance"), int)
        and not isinstance(value["instance"], bool)
        and value["instance"] >= 0
        and isinstance(value.get("kernel"), int)
        and not isinstance(value["kernel"], bool)
        and value["kernel"] >= 0
        and isinstance(value.get("kernel_name"), str)
        and bool(value["kernel_name"])
        and _is_lower_hex(value.get("semantic_hash"), 16)
        and _is_lower_hex(value.get("cache_key"), 16)
    )


def _aot_occurrence_sort_key(value: dict[str, Any]) -> tuple[object, ...]:
    return (
        value["kind"],
        value["component"],
        value["instance"],
        value["kernel"],
        value["kernel_name"],
        value["semantic_hash"],
        value["cache_key"],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_file_hash(
    path: Path, expected: object, label: str, errors: list[str]
) -> str | None:
    if not _is_sha256(expected):
        errors.append(f"{label}: invalid SHA-256")
        return None
    try:
        actual = _sha256_file(path)
    except OSError as error:
        errors.append(f"{label}: {error}")
        return None
    if actual != expected:
        errors.append(f"{label}: SHA-256 mismatch")
    return actual


def _manifest(
    path: Path, expected_names: set[str], label: str, errors: list[str]
) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"{label}: {error}")
        return entries
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 2 or not _is_sha256(fields[0]) or fields[1] in entries:
            errors.append(f"{label}: invalid entry at line {line_number}")
            continue
        entries[fields[1]] = fields[0]
    if set(entries) != expected_names:
        errors.append(f"{label}: file set is not exact")
    return entries


def _resolved_reference(path: object, base: Path) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    reference = Path(path)
    return (reference if reference.is_absolute() else base / reference).resolve()


def validate_local_admission(
    admission: dict[str, Any],
    admission_path: Path,
    *,
    expected_source: dict[str, Any],
    expected_dry_run: bool,
    gpu_bench_binary: Path,
    raw_input_manifest: Path,
    bootloader: Path,
    pinned_adapted_manifest: Path,
    aot_manifest: Path,
    required_runtime_mode: str = "arena-graph",
) -> list[str]:
    """Validate the complete local capacity/input admission before pod contact."""

    errors: list[str] = []
    if required_runtime_mode != "arena-graph":
        errors.append("local admission: release runtime must be arena-graph")
    if not _valid_source_identity(expected_source):
        errors.append("local admission: expected source identity is invalid")
    if admission.get("schema") != "stwo.local-preflight-admission.v1":
        errors.append("local admission: wrong schema")
    if admission.get("passed") is not True:
        errors.append("local admission: did not pass")
    if admission.get("dry_run") is not expected_dry_run:
        errors.append("local admission: dry-run state mismatch")
    if admission.get("runtime_mode") != required_runtime_mode:
        errors.append("local admission: runtime mode mismatch")
    if admission.get("source") != expected_source:
        errors.append("local admission: source mismatch")
    if admission.get("profiles") != QUALIFICATION_PROFILES:
        errors.append("local admission: profile map is not exact")
    if admission.get("preflight_ceiling_bytes") != PREFLIGHT_CAP_BYTES:
        errors.append("local admission: preflight ceiling mismatch")

    reproduction = admission.get("adapter_reproduction")
    if not isinstance(reproduction, dict):
        errors.append("local admission: adapter reproduction contract is not exact")
        reproduction = {}
    elif set(reproduction) != ADAPTER_REPRODUCTION_KEYS:
        errors.append("local admission: adapter reproduction contract is not exact")
    if reproduction.get("byte_equal") is not True:
        errors.append("local admission: adapter reproduction is not byte-equal")

    aot = admission.get("aot_coverage")
    if not isinstance(aot, dict) or set(aot) != AOT_ADMISSION_KEYS:
        errors.append("local admission: AOT coverage contract is not exact")
        aot = {}
    expected_aot_manifest = aot_manifest.resolve()
    admission_aot_manifest = _resolved_reference(
        aot.get("manifest"), admission_path.resolve().parent
    )
    if admission_aot_manifest != expected_aot_manifest:
        errors.append("local admission: AOT manifest path mismatch")
    _check_file_hash(
        expected_aot_manifest,
        aot.get("manifest_sha256"),
        "AOT manifest",
        errors,
    )
    try:
        manifest_value = json.loads(expected_aot_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"AOT manifest: {error}")
        manifest_value = []
    manifest_by_key: dict[str, dict[str, Any]] = {}
    if not isinstance(manifest_value, list) or not manifest_value:
        errors.append("AOT manifest: expected a nonempty array")
    else:
        for index, entry in enumerate(manifest_value):
            if (
                not isinstance(entry, dict)
                or set(entry) != AOT_MANIFEST_ENTRY_KEYS
                or entry.get("kind") not in {"constraint", "witness"}
                or not isinstance(entry.get("label"), str)
                or not entry["label"]
                or not isinstance(entry.get("kernel_name"), str)
                or not entry["kernel_name"]
                or not _is_lower_hex(entry.get("cache_key"), 16)
                or not _is_lower_hex(entry.get("semantic_hash"), 16)
                or not isinstance(entry.get("file"), str)
                or not entry["file"]
            ):
                errors.append(f"AOT manifest: malformed entry {index}")
                continue
            if entry["cache_key"] in manifest_by_key:
                errors.append(f"AOT manifest: duplicate key {entry['cache_key']}")
            manifest_by_key[entry["cache_key"]] = entry
    if aot.get("manifest_entry_count") != len(manifest_by_key):
        errors.append("local admission: AOT manifest entry count mismatch")
    if not _is_lower_hex(aot.get("manifest_blake3"), 64):
        errors.append("local admission: invalid AOT manifest BLAKE3")

    root = admission_path.resolve().parent
    preflight_hashes = admission.get("preflight_artifact_sha256")
    if not isinstance(preflight_hashes, dict):
        errors.append("local admission: preflight artifact set is not exact")
        preflight_hashes = {}
    elif set(preflight_hashes) != set(PREFLIGHT_SPECS):
        errors.append("local admission: preflight artifact set is not exact")

    adapted_manifest_path = _resolved_reference(
        admission.get("adapted_input_manifest"), root
    )
    if adapted_manifest_path is None:
        errors.append("local admission: adapted manifest path is absent")
        adapted_manifest: dict[str, str] = {}
    else:
        _check_file_hash(
            adapted_manifest_path,
            admission.get("adapted_input_manifest_sha256"),
            "adapted manifest",
            errors,
        )
        adapted_manifest = _manifest(
            adapted_manifest_path,
            ADAPTED_INPUT_NAMES,
            "adapted manifest",
            errors,
        )

    _check_file_hash(
        raw_input_manifest,
        reproduction.get("raw_input_manifest_sha256"),
        "raw input manifest",
        errors,
    )
    raw_manifest = _manifest(
        raw_input_manifest, RAW_INPUT_NAMES, "raw input manifest", errors
    )
    bootloader_sha = _check_file_hash(
        bootloader,
        reproduction.get("bootloader_sha256"),
        "bootloader",
        errors,
    )
    if bootloader_sha is not None and raw_manifest.get(bootloader.name) != bootloader_sha:
        errors.append("raw input manifest: bootloader digest mismatch")

    _check_file_hash(
        pinned_adapted_manifest,
        reproduction.get("pinned_adapted_manifest_sha256"),
        "pinned adapted manifest",
        errors,
    )
    pinned_manifest = _manifest(
        pinned_adapted_manifest,
        ADAPTED_INPUT_NAMES,
        "pinned adapted manifest",
        errors,
    )
    if adapted_manifest != pinned_manifest:
        errors.append("adapted manifest: does not match pinned manifest")

    _check_file_hash(
        gpu_bench_binary,
        reproduction.get("gpu_bench_binary_sha256"),
        "gpu_bench binary",
        errors,
    )

    preflight_aot_sha256: dict[str, str] = {}
    preflight_aot_blake3: dict[str, str] = {}
    preflight_key_blake3: dict[str, str] = {}
    preflight_occurrences: dict[str, list[dict[str, Any]]] = {}
    all_occurrences: list[dict[str, Any]] = []
    for artifact_name, (input_name, expected_policy) in PREFLIGHT_SPECS.items():
        artifact_path = root / artifact_name
        _check_file_hash(
            artifact_path,
            preflight_hashes.get(artifact_name),
            artifact_name,
            errors,
        )
        try:
            record = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append(f"{artifact_name}: {error}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{artifact_name}: artifact is not an object")
            continue
        arena = record.get("arena") or {}
        if not isinstance(arena, dict):
            errors.append(f"{artifact_name}: arena is not an object")
            arena = {}
        arena_bytes = arena.get("total_bytes")
        if (
            record.get("pass") is not True
            or record.get("vram_fit") is not True
            or record.get("vram_budget_bytes") != PREFLIGHT_CAP_BYTES
            or not isinstance(arena_bytes, int)
            or isinstance(arena_bytes, bool)
            or not 0 <= arena_bytes <= PREFLIGHT_CAP_BYTES
            or record.get("runtime_policy") != expected_policy
        ):
            errors.append(f"{artifact_name}: preflight contract mismatch")
        coverage = record.get("aot_coverage")
        if not isinstance(coverage, dict) or set(coverage) != AOT_PREFLIGHT_KEYS:
            errors.append(f"{artifact_name}: AOT coverage shape mismatch")
            coverage = {}
        coverage_manifest = _resolved_reference(
            coverage.get("manifest"), artifact_path.parent
        )
        missing = coverage.get("missing_occurrences")
        occurrences = coverage.get("required_occurrences")
        if (
            coverage.get("pass") is not True
            or coverage_manifest != expected_aot_manifest
            or coverage.get("manifest_blake3") != aot.get("manifest_blake3")
            or coverage.get("manifest_entries") != len(manifest_by_key)
            or missing != []
            or coverage.get("missing_occurrence_count") != 0
            or not isinstance(occurrences, list)
            or not occurrences
            or coverage.get("required_occurrence_count") != len(occurrences or [])
            or not _is_lower_hex(coverage.get("required_occurrences_blake3"), 64)
            or not _is_lower_hex(coverage.get("required_unique_keys_blake3"), 64)
        ):
            errors.append(f"{artifact_name}: AOT coverage did not pass")
            occurrences = []
        elif (
            any(not _valid_aot_occurrence(item) for item in occurrences)
            or occurrences != sorted(occurrences, key=_aot_occurrence_sort_key)
        ):
            errors.append(f"{artifact_name}: AOT occurrence ledger is not canonical")
            occurrences = []
        if occurrences:
            unique_keys = {item["cache_key"] for item in occurrences}
            if coverage.get("required_unique_key_count") != len(unique_keys):
                errors.append(f"{artifact_name}: AOT unique-key count mismatch")
            for occurrence in occurrences:
                entry = manifest_by_key.get(occurrence["cache_key"])
                if entry is None or (
                    entry["kind"] != occurrence["kind"]
                    or entry["kernel_name"] != occurrence["kernel_name"]
                    or entry["semantic_hash"] != occurrence["semantic_hash"]
                ):
                    errors.append(
                        f"{artifact_name}: uncovered AOT key {occurrence['cache_key']}"
                    )
            preflight_occurrences[artifact_name] = occurrences
            preflight_aot_sha256[artifact_name] = _json_sha256(occurrences)
            preflight_aot_blake3[artifact_name] = coverage[
                "required_occurrences_blake3"
            ]
            preflight_key_blake3[artifact_name] = coverage[
                "required_unique_keys_blake3"
            ]
            all_occurrences.extend(occurrences)
        source = _resolved_reference(record.get("source"), artifact_path.parent)
        if source is None or source.name != input_name:
            errors.append(f"{artifact_name}: adapted input mismatch")
            continue
        _check_file_hash(
            source,
            adapted_manifest.get(input_name),
            f"{artifact_name} adapted input",
            errors,
        )

    if set(preflight_occurrences) != set(PREFLIGHT_SPECS):
        errors.append("local admission: incomplete AOT preflight occurrence set")
        return errors
    if not (
        preflight_aot_sha256["preflight_flags_off_SN2.json"]
        == preflight_aot_sha256["preflight_universal_SN2.json"]
        == preflight_aot_sha256["preflight_sn2_headline.json"]
    ):
        errors.append("local admission: SN2 AOT keys vary across runtime-only profiles")
    union_by_record = {_canonical_json(item): item for item in all_occurrences}
    occurrence_union = [union_by_record[key] for key in sorted(union_by_record)]
    union_keys = sorted({item["cache_key"] for item in occurrence_union})
    if aot.get("required_occurrence_count") != len(occurrence_union):
        errors.append("local admission: AOT union occurrence count mismatch")
    if aot.get("required_unique_key_count") != len(union_keys):
        errors.append("local admission: AOT union key count mismatch")
    if aot.get("required_unique_keys") != union_keys:
        errors.append("local admission: AOT union key set mismatch")
    if aot.get("required_occurrences_sha256") != _json_sha256(occurrence_union):
        errors.append("local admission: AOT union occurrence hash mismatch")
    if aot.get("required_unique_keys_sha256") != _json_sha256(union_keys):
        errors.append("local admission: AOT union key hash mismatch")
    if aot.get("preflight_occurrences_sha256") != preflight_aot_sha256:
        errors.append("local admission: AOT preflight SHA-256 map mismatch")
    if aot.get("preflight_occurrences_blake3") != preflight_aot_blake3:
        errors.append("local admission: AOT preflight BLAKE3 map mismatch")
    if aot.get("preflight_unique_keys_blake3") != preflight_key_blake3:
        errors.append("local admission: AOT preflight key BLAKE3 map mismatch")
    expected_per_sn = {}
    for pie in range(1, 5):
        occurrences = preflight_occurrences[f"preflight_universal_SN{pie}.json"]
        keys = sorted({item["cache_key"] for item in occurrences})
        expected_per_sn[f"SN_PIE_{pie}"] = {
            "required_occurrence_count": len(occurrences),
            "required_unique_key_count": len(keys),
            "required_occurrences_sha256": _json_sha256(occurrences),
            "required_unique_keys_sha256": _json_sha256(keys),
        }
    if aot.get("per_sn") != expected_per_sn:
        errors.append("local admission: AOT per-SN coverage mismatch")
    if aot.get("missing_occurrences") != []:
        errors.append("local admission: AOT keys are missing")
    return errors


def _validate_remote_execution_target(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target = artifact.get("execution_target")
    if not isinstance(target, dict) or set(target) != {
        "schema", "pod_id", "boot_id", "gpu_uuid", "gpu_name", "gpu_bench", "inputs"
    }:
        errors.append("soundness artifact: invalid remote execution target shape")
        target = {}
    if target.get("schema") != "stwo.remote-execution-target.v1":
        errors.append("soundness artifact: invalid remote execution target schema")
    for field in ("pod_id", "boot_id", "gpu_uuid", "gpu_name"):
        value = target.get(field)
        if not isinstance(value, str) or not value or "\n" in value:
            errors.append(f"soundness artifact execution target: invalid {field}")
    binary = target.get("gpu_bench")
    if (
        not isinstance(binary, dict)
        or set(binary) != {"path", "sha256"}
        or not isinstance(binary.get("path"), str)
        or not binary.get("path", "").startswith("/")
        or not _is_sha256(binary.get("sha256"))
        or not binary.get("path", "").endswith(f"/sealed/gpu_bench.{binary.get('sha256')}")
    ):
        errors.append("soundness artifact execution target: invalid sealed gpu_bench")
    inputs = target.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        errors.append("soundness artifact execution target: no inputs")
    else:
        for label, value in inputs.items():
            if (
                not isinstance(label, str)
                or not label
                or not isinstance(value, dict)
                or set(value) != {"path", "sha256"}
                or not isinstance(value.get("path"), str)
                or not value.get("path", "").startswith("/")
                or not _is_sha256(value.get("sha256"))
            ):
                errors.append(f"soundness artifact execution target: invalid input {label!r}")
    encoded = json.dumps(
        target, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if artifact.get("execution_target_sha256") != hashlib.sha256(encoded).hexdigest():
        errors.append("soundness artifact: execution target hash mismatch")
    if artifact.get("execution_target_postcheck") is not True:
        errors.append("soundness artifact: execution target postcheck did not pass")
    projection = artifact.get("source_projection")
    if (
        not isinstance(projection, dict)
        or set(projection) != {"method", "verified_after_soundness", "source"}
        or projection.get("method") != "rsync-archive-checksum-dry-run-clean"
        or projection.get("verified_after_soundness") is not True
        or not _valid_source_identity(projection.get("source"))
    ):
        errors.append("soundness artifact: source projection was not verified")
    return errors


def validate_soundness_gate(
    artifact: dict[str, Any], required_mode: str | None = None
) -> list[str]:
    errors: list[str] = []
    schema = artifact.get("schema")
    if schema not in {"stwo.cuda.soundness-gate.v2", "stwo.cuda.soundness-gate.v3"}:
        errors.append(f"soundness schema: got {schema!r}")
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
    expected_order = [name for name, _command, _required in expected_manifest]
    actual_order = [
        gate.get("name") if isinstance(gate, dict) else None for gate in gates
    ]
    if actual_order != expected_order:
        errors.append("soundness artifact gate order does not match manifest")
    for field in ("stwo_git_head", "stwo_cairo_git_head"):
        value = artifact.get(field)
        if not isinstance(value, str) or len(value) != 40:
            errors.append(f"soundness artifact {field}: expected a 40-character revision")
    for field in ("stwo_worktree_hash", "stwo_cairo_worktree_hash"):
        value = artifact.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"soundness artifact {field}: expected a 64-character source hash")
    if schema == "stwo.cuda.soundness-gate.v3":
        errors.extend(_validate_remote_execution_target(artifact))
        synced = artifact.get("synced_source")
        if (
            not isinstance(synced, dict)
            or set(synced) != {"stwo", "stwo_cairo", "transport"}
            or synced.get("transport") != "rsync-archive-checksum"
            or not _valid_source_identity(
                {key: synced.get(key) for key in ("stwo", "stwo_cairo")}
            )
        ):
            errors.append("soundness artifact: invalid synced source identity")
        else:
            source = {key: synced[key] for key in ("stwo", "stwo_cairo")}
            projection = artifact.get("source_projection") or {}
            if projection.get("source") != source:
                errors.append("soundness artifact: projection disagrees with synced source")
            if any(
                artifact.get(field) != source[repo][key]
                for field, repo, key in (
                    ("stwo_git_head", "stwo", "head"),
                    ("stwo_worktree_hash", "stwo", "worktree_hash"),
                    ("stwo_cairo_git_head", "stwo_cairo", "head"),
                    (
                        "stwo_cairo_worktree_hash",
                        "stwo_cairo",
                        "worktree_hash",
                    ),
                )
            ):
                errors.append("soundness artifact: top-level source identity mismatch")
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
        if gate.get("stub_skip_detected") is not False:
            errors.append(f"soundness gate {name!r}: CUDA stub skip was not excluded")
        expected_required = expected_gates.get(name)
        if expected_required is not None and required != expected_required:
            errors.append(
                f"soundness gate {name!r}: manifest requires {expected_required}, "
                f"artifact claimed {required!r}"
            )
        expected_command = expected_commands.get(name)
        if expected_command is not None and gate.get("command") != expected_command:
            errors.append(f"soundness gate {name!r}: command does not match manifest")
        if name == STRICT_RESIDENT_GATE:
            expected_names = list(STRICT_RESIDENT_REQUIRED_TESTS)
            if gate.get("required_test_names") != expected_names:
                errors.append(f"soundness gate {name!r}: required test names drifted")
            executed_names = gate.get("executed_test_names")
            if (
                not isinstance(executed_names, list)
                or len(executed_names) != len(expected_names)
                or set(executed_names) != set(expected_names)
            ):
                errors.append(f"soundness gate {name!r}: executed test names drifted")
    missing = set(expected_gates) - names
    unexpected = names - set(expected_gates)
    if missing:
        errors.append(f"soundness artifact missing gates: {sorted(missing)!r}")
    if unexpected:
        errors.append(f"soundness artifact has unexpected gates: {sorted(unexpected)!r}")
    return errors


def validate_qualification_soundness_gate(
    artifact: dict[str, Any],
    *,
    expected_source: dict[str, Any],
    expected_dry_run: bool,
    required_mode: str = "arena-graph",
) -> list[str]:
    """Validate the counted gate plus its exact headline qualification environment."""

    errors = validate_soundness_gate(artifact, required_mode)
    if artifact.get("schema") != "stwo.cuda.soundness-gate.v3":
        errors.append("soundness artifact: qualification requires sealed v3 schema")
    if required_mode != "arena-graph":
        errors.append("soundness artifact: qualification runtime must be arena-graph")
    if not _valid_source_identity(expected_source):
        errors.append("soundness artifact: expected source identity is invalid")
    if artifact.get("dry_run") is not expected_dry_run:
        errors.append("soundness artifact: dry-run state mismatch")
    source_identity = expected_source if isinstance(expected_source, dict) else {}
    expected_synced = {**source_identity, "transport": "rsync-archive-checksum"}
    if artifact.get("synced_source") != expected_synced:
        errors.append("soundness artifact: synced source mismatch")
    projection = artifact.get("source_projection") or {}
    if not isinstance(projection, dict) or projection.get("source") != expected_source:
        errors.append("soundness artifact: source projection identity mismatch")
    if artifact.get("qualification_flags") != {
        flag: 1 for flag in QUALIFICATION_FLAGS
    }:
        errors.append("soundness artifact: headline flags mismatch")
    stwo = source_identity.get("stwo") or {}
    stwo_cairo = source_identity.get("stwo_cairo") or {}
    if not isinstance(stwo, dict):
        stwo = {}
    if not isinstance(stwo_cairo, dict):
        stwo_cairo = {}
    expected_env = {
        "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
        "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
        "STWO_PARITY_REF_STWO_HEAD": stwo.get("head"),
        "STWO_PARITY_REF_STWO_WORKTREE_HASH": stwo.get("worktree_hash"),
        "STWO_PARITY_REF_STWO_CAIRO_HEAD": stwo_cairo.get("head"),
        "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": stwo_cairo.get(
            "worktree_hash"
        ),
        **{flag: "1" for flag in QUALIFICATION_FLAGS},
        RETAINED_BUDGET_FLAG: "29469326848",
    }
    if artifact.get("effective_stwo_env") != expected_env:
        errors.append("soundness artifact: effective headline environment mismatch")
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
        graph_launches = record.get("gpu_graph_launches")
        if (
            not isinstance(graph_launches, int)
            or isinstance(graph_launches, bool)
            or graph_launches != 29
        ):
            errors.append(
                f"gpu_graph_launches: expected integer 29, got {graph_launches!r}"
            )
        kernel_launches = record.get("gpu_kernel_launches")
        if (
            not isinstance(kernel_launches, int)
            or isinstance(kernel_launches, bool)
            or not 29 <= kernel_launches < 100_000
        ):
            errors.append(
                "gpu_kernel_launches: expected integer in [29, 100000), "
                f"got {kernel_launches!r}"
            )
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


def validate_benchmark_measurement(
    record: dict[str, Any],
    *,
    expected_program: str,
    expected_reps: int,
    expected_gpu: str | None = None,
    require_fresh_simd_reference: bool = False,
) -> list[str]:
    """Bind the published warm median to one exact input and its raw samples."""

    errors: list[str] = []
    if require_fresh_simd_reference:
        for field in (
            "simd_reference_required",
            "simd_reference_comparison_applicable",
            "simd_reference_byte_equal",
            "simd_reference_fresh",
        ):
            if record.get(field) is not True:
                errors.append(f"{field}: expected true, got {record.get(field)!r}")
        digests = {}
        for field in ("gpu_proof_blake3", "simd_reference_blake3"):
            value = record.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                errors.append(f"{field}: expected a 64-character lowercase hex digest")
            else:
                digests[field] = value
        if (
            len(digests) == 2
            and digests["gpu_proof_blake3"] != digests["simd_reference_blake3"]
        ):
            errors.append("GPU and fresh SIMD proof BLAKE3 digests differ")
        reference_s = record.get("simd_reference_s")
        if (
            not isinstance(reference_s, (int, float))
            or isinstance(reference_s, bool)
            or not math.isfinite(reference_s)
            or reference_s <= 0
        ):
            errors.append(
                f"simd_reference_s: expected a finite positive duration, got {reference_s!r}"
            )
    if record.get("program") != expected_program:
        errors.append(
            f"program: expected {expected_program!r}, got {record.get('program')!r}"
        )
    if expected_gpu is not None and record.get("gpu") != expected_gpu:
        errors.append(f"gpu: expected {expected_gpu!r}, got {record.get('gpu')!r}")
    if not isinstance(expected_reps, int) or isinstance(expected_reps, bool) or expected_reps < 2:
        return errors + ["expected_reps: must be an integer >= 2"]

    for field, expected in (
        ("reps", expected_reps),
        ("verified_reps", expected_reps),
        ("warm_sample_count", expected_reps - 1),
    ):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            errors.append(f"{field}: expected integer {expected}, got {value!r}")

    window = {}
    for field in (
        "gpu_proof_loop_started_unix_ns",
        "gpu_proof_loop_finished_unix_ns",
    ):
        value = record.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1_000_000_000_000_000_000
        ):
            errors.append(f"{field}: expected a Unix nanosecond integer, got {value!r}")
        else:
            window[field] = value
    if len(window) == 2 and (
        window["gpu_proof_loop_finished_unix_ns"]
        <= window["gpu_proof_loop_started_unix_ns"]
    ):
        errors.append("GPU proof-loop finish must be after its start")

    samples = record.get("prove_s_warm_samples_raw")
    if (
        not isinstance(samples, list)
        or len(samples) != expected_reps - 1
        or any(
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(sample)
            or sample <= 0
            for sample in samples
        )
    ):
        errors.append(
            "prove_s_warm_samples_raw: expected exact finite positive warm samples"
        )
        return errors

    raw_median = float(statistics.median(samples))
    ordered = sorted(float(sample) for sample in samples)
    rank = 0.95 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    raw_p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    def rounded3(value: float) -> float:
        # All measurement values are positive; this matches Rust f64::round().
        return math.floor(value * 1000.0 + 0.5) / 1000.0

    def require_rounded(field: str, expected: float) -> None:
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or abs(float(value) - rounded3(expected)) > 1e-9
        ):
            errors.append(
                f"{field}: expected rounded measurement {rounded3(expected)!r}, got {value!r}"
            )

    require_rounded("prove_s_warm_median", raw_median)
    require_rounded("prove_s_warm_p95", raw_p95)
    for work_field, rate_field in (
        ("cycle_count", "mhz_median"),
        ("pie_n_steps", "useful_mhz_median"),
    ):
        work = record.get(work_field)
        if not isinstance(work, int) or isinstance(work, bool) or work <= 0:
            errors.append(f"{work_field}: expected a positive integer, got {work!r}")
        else:
            require_rounded(rate_field, work / raw_median / 1e6)
            p95_rate_field = (
                "mhz_at_warm_p95"
                if work_field == "cycle_count"
                else "useful_mhz_at_warm_p95"
            )
            require_rounded(p95_rate_field, work / raw_p95 / 1e6)
    for field, expected in (
        ("security_bits", 96),
        ("n_queries", 70),
        ("pow_bits", 26),
        ("fold_step", 3),
    ):
        if record.get(field) != expected:
            errors.append(f"{field}: expected {expected}, got {record.get(field)!r}")
    for field in ("prove_s_cold", "verify_ms", "proof_kb"):
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"{field}: expected a finite positive value, got {value!r}")
    cold = record.get("prove_s_cold")
    if (
        len(window) == 2
        and isinstance(cold, (int, float))
        and not isinstance(cold, bool)
        and math.isfinite(cold)
        and cold > 0
    ):
        loop_seconds = (
            window["gpu_proof_loop_finished_unix_ns"]
            - window["gpu_proof_loop_started_unix_ns"]
        ) / 1e9
        recorded_prove_seconds = float(cold) + sum(float(sample) for sample in samples)
        if loop_seconds + 0.01 < recorded_prove_seconds:
            errors.append(
                "GPU proof-loop wall duration is shorter than the summed prove samples"
            )
    for field in (
        "peak_rss_gb",
        "vram_end_gb",
        "vram_peak_gb",
        "pool_used_high_gb",
        "pool_reserved_high_gb",
    ):
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            errors.append(f"{field}: expected a finite non-negative value, got {value!r}")
    pool_used = record.get("pool_used_high_gb")
    pool_reserved = record.get("pool_reserved_high_gb")
    if (
        isinstance(pool_used, (int, float))
        and not isinstance(pool_used, bool)
        and math.isfinite(pool_used)
        and isinstance(pool_reserved, (int, float))
        and not isinstance(pool_reserved, bool)
        and math.isfinite(pool_reserved)
        and pool_reserved < pool_used
    ):
        errors.append("pool_reserved_high_gb: expected at least pool_used_high_gb")
    for field in (
        "proof_comparison_applicable",
        "proof_byte_equal_required",
        "proof_byte_equal",
        "performance_claim_admissible",
    ):
        if record.get(field) is not True:
            errors.append(f"{field}: expected true, got {record.get(field)!r}")
    if record.get("throughput_distribution_applicable") is not True:
        errors.append("throughput_distribution_applicable: expected true")
    return errors


def validate_gpu_telemetry_artifact(
    record: dict[str, Any], metadata: dict[str, Any]
) -> list[str]:
    """Bind raw nvidia-smi samples to the proof-loop envelope.

    The envelope includes per-repetition input clones and phase-report gaps; it ends
    before the fresh SIMD reference and is not claimed to be pure kernel time.
    """

    errors: list[str] = []
    expected_columns = list(GPU_TELEMETRY_COLUMNS)
    if metadata.get("schema") != GPU_TELEMETRY_SCHEMA:
        errors.append(f"telemetry schema: expected {GPU_TELEMETRY_SCHEMA!r}")
    if metadata.get("columns") != expected_columns:
        errors.append("telemetry columns do not match the fixed v1 schema")
    if metadata.get("sampler_complete") is not True:
        errors.append("telemetry sampler did not remain alive for the complete process")
    if metadata.get("sample_interval_ms") != GPU_TELEMETRY_SAMPLE_INTERVAL_MS:
        errors.append(
            f"telemetry sample_interval_ms: expected {GPU_TELEMETRY_SAMPLE_INTERVAL_MS}"
        )
    if metadata.get("max_gap_ns") != GPU_TELEMETRY_MAX_GAP_NS:
        errors.append(f"telemetry max_gap_ns: expected {GPU_TELEMETRY_MAX_GAP_NS}")
    if metadata.get("transport_equal") is not True:
        errors.append("telemetry transport equality was not established")

    path_value = metadata.get("path")
    if not isinstance(path_value, str) or not path_value:
        return errors + ["telemetry path is missing"]
    path = Path(path_value)
    try:
        payload = path.read_bytes()
    except OSError as error:
        return errors + [f"telemetry artifact: {error}"]
    actual_digest = hashlib.sha256(payload).hexdigest()
    if metadata.get("sha256") != actual_digest:
        errors.append("telemetry SHA256 does not match retained CSV bytes")
    if metadata.get("remote_sha256") != actual_digest:
        errors.append("remote telemetry SHA256 does not match retained CSV bytes")
    if metadata.get("size_bytes") != len(payload) or not payload:
        errors.append("telemetry size does not match retained CSV bytes")
    if metadata.get("remote_size_bytes") != len(payload):
        errors.append("remote telemetry size does not match retained CSV bytes")

    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames != expected_columns:
            errors.append("telemetry CSV header does not match the fixed v1 schema")
            return errors
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        return errors + [f"telemetry CSV: {error}"]
    if not rows:
        return errors + ["telemetry CSV has no samples"]
    if metadata.get("sample_count") != len(rows):
        errors.append("telemetry sample_count does not match retained CSV rows")

    start = record.get("gpu_proof_loop_started_unix_ns")
    finish = record.get("gpu_proof_loop_finished_unix_ns")
    window_samples = 0
    last_timestamp = 0
    timestamps = []
    bounded_fields = {
        "utilization_gpu_pct": (0.0, 100.0),
        "utilization_memory_pct": (0.0, 100.0),
        "memory_used_mib": (0.0, math.inf),
        "power_draw_w": (0.0, math.inf),
        "clock_sm_mhz": (0.0, math.inf),
        "clock_memory_mhz": (0.0, math.inf),
        "temperature_gpu_c": (0.0, math.inf),
        "power_limit_w": (0.0, math.inf),
        "clock_max_sm_mhz": (0.0, math.inf),
        "clock_max_memory_mhz": (0.0, math.inf),
    }
    for index, row in enumerate(rows, start=1):
        if None in row or any(row.get(field) is None for field in GPU_TELEMETRY_COLUMNS):
            errors.append(f"telemetry row {index}: column count does not match schema")
            continue
        try:
            timestamp = int(row["timestamp_unix_ns"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"telemetry row {index}: invalid Unix-ns timestamp")
            continue
        if timestamp <= 0 or timestamp < last_timestamp:
            errors.append(f"telemetry row {index}: timestamps are not monotonic")
        last_timestamp = timestamp
        timestamps.append(timestamp)
        if isinstance(start, int) and isinstance(finish, int) and start <= timestamp <= finish:
            window_samples += 1
        if not row.get("driver_version", "").strip():
            errors.append(f"telemetry row {index}: driver_version is empty")
        for field, (lower, upper) in bounded_fields.items():
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"telemetry row {index}: {field} is not numeric")
                continue
            if not math.isfinite(value) or value < lower or value > upper:
                errors.append(f"telemetry row {index}: {field} is out of range")
    if window_samples < GPU_TELEMETRY_MIN_WINDOW_SAMPLES:
        errors.append(
            "telemetry has fewer than "
            f"{GPU_TELEMETRY_MIN_WINDOW_SAMPLES} samples inside the GPU proof-loop window"
        )
    if isinstance(start, int) and timestamps and timestamps[0] > start:
        errors.append("telemetry sampling began after the GPU proof loop")
    if isinstance(finish, int) and timestamps and timestamps[-1] < finish:
        errors.append("telemetry sampling ended before the GPU proof loop")
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous > GPU_TELEMETRY_MAX_GAP_NS:
            errors.append(
                "telemetry cadence gap exceeds "
                f"{GPU_TELEMETRY_MAX_GAP_NS}ns: {current - previous}ns"
            )
            break
    if metadata.get("proof_window_sample_count") != window_samples:
        errors.append("telemetry proof_window_sample_count does not match CSV samples")
    return errors


def _print_contract_errors(label: str, errors: list[str]) -> int:
    if not errors:
        return 0
    print(f"{label} failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, nargs="?", help="gpu_bench stdout file")
    parser.add_argument(
        "--runtime-mode",
        choices=tuple(RUNTIME_MODES),
        default="arena-graph",
        help="runtime mode the benchmark requested (default: arena-graph)",
    )
    parser.add_argument(
        "--soundness-gate",
        type=Path,
        help="counted CUDA differential-test artifact from run_cuda_soundness_gate.py",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--soundness-only",
        action="store_true",
        help="validate the complete headline soundness artifact before pod work",
    )
    modes.add_argument(
        "--local-admission",
        type=Path,
        help="validate the complete local input/capacity admission before pod work",
    )
    parser.add_argument("--gpu-bench-binary", type=Path)
    parser.add_argument("--raw-input-manifest", type=Path)
    parser.add_argument("--bootloader", type=Path)
    parser.add_argument("--pinned-adapted-manifest", type=Path)
    parser.add_argument("--aot-manifest", type=Path)
    parser.add_argument("--expected-dry-run", type=int, choices=(0, 1))
    parser.add_argument("--expected-program")
    parser.add_argument("--expected-reps", type=int)
    parser.add_argument("--expected-gpu")
    parser.add_argument("--require-fresh-simd-reference", action="store_true")
    parser.add_argument("--gpu-telemetry-csv", type=Path)
    parser.add_argument("--gpu-telemetry-remote-sha256")
    parser.add_argument("--gpu-telemetry-remote-size", type=int)
    parser.add_argument("--stwo-head")
    parser.add_argument("--stwo-worktree-hash")
    parser.add_argument("--stwo-cairo-head")
    parser.add_argument("--stwo-cairo-worktree-hash")
    args = parser.parse_args(argv)

    pre_pod_mode = args.local_admission is not None or args.soundness_only
    source_values = (
        args.stwo_head,
        args.stwo_worktree_hash,
        args.stwo_cairo_head,
        args.stwo_cairo_worktree_hash,
    )
    if pre_pod_mode and (not all(source_values) or args.expected_dry_run is None):
        parser.error(
            "pre-pod validation requires both source heads, both worktree hashes, "
            "and --expected-dry-run"
        )
    expected_source = {
        "stwo": {
            "head": args.stwo_head,
            "worktree_hash": args.stwo_worktree_hash,
        },
        "stwo_cairo": {
            "head": args.stwo_cairo_head,
            "worktree_hash": args.stwo_cairo_worktree_hash,
        },
    }

    if args.local_admission is not None:
        required_paths = {
            "--gpu-bench-binary": args.gpu_bench_binary,
            "--raw-input-manifest": args.raw_input_manifest,
            "--bootloader": args.bootloader,
            "--pinned-adapted-manifest": args.pinned_adapted_manifest,
            "--aot-manifest": args.aot_manifest,
        }
        missing = [name for name, value in required_paths.items() if value is None]
        if missing:
            parser.error(f"--local-admission requires {', '.join(missing)}")
        try:
            admission = json.loads(args.local_admission.read_text(encoding="utf-8"))
            if not isinstance(admission, dict):
                raise ValueError("artifact is not an object")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return _print_contract_errors("Local admission contract", [str(error)])
        errors = validate_local_admission(
            admission,
            args.local_admission,
            expected_source=expected_source,
            expected_dry_run=bool(args.expected_dry_run),
            gpu_bench_binary=args.gpu_bench_binary,
            raw_input_manifest=args.raw_input_manifest,
            bootloader=args.bootloader,
            pinned_adapted_manifest=args.pinned_adapted_manifest,
            aot_manifest=args.aot_manifest,
            required_runtime_mode=args.runtime_mode,
        )
        return _print_contract_errors("Local admission contract", errors)

    if args.soundness_only:
        if args.soundness_gate is None:
            parser.error("--soundness-only requires --soundness-gate")
        try:
            soundness = load_soundness_gate(args.soundness_gate)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return _print_contract_errors("CUDA soundness contract", [str(error)])
        errors = validate_qualification_soundness_gate(
            soundness,
            expected_source=expected_source,
            expected_dry_run=bool(args.expected_dry_run),
            required_mode=args.runtime_mode,
        )
        return _print_contract_errors("CUDA soundness contract", errors)

    if args.record is None or args.soundness_gate is None:
        parser.error("record validation requires RECORD and --soundness-gate")
    try:
        record = load_main_record(args.record)
    except (OSError, ValueError) as error:
        print(f"GPU-native architecture contract: {error}", file=sys.stderr)
        return 1
    errors = validate_record(record, args.runtime_mode)
    if (args.expected_program is None) != (args.expected_reps is None):
        parser.error("record measurement validation requires both --expected-program and --expected-reps")
    if args.require_fresh_simd_reference and args.expected_program is None:
        parser.error("--require-fresh-simd-reference requires measurement validation")
    if args.gpu_telemetry_csv is not None and args.expected_program is None:
        parser.error("--gpu-telemetry-csv requires measurement validation")
    if args.gpu_telemetry_csv is not None and (
        args.gpu_telemetry_remote_sha256 is None
        or args.gpu_telemetry_remote_size is None
    ):
        parser.error(
            "--gpu-telemetry-csv requires remote SHA256 and size transport metadata"
        )
    if args.expected_program is not None and args.expected_reps is not None:
        errors.extend(
            validate_benchmark_measurement(
                record,
                expected_program=args.expected_program,
                expected_reps=args.expected_reps,
                expected_gpu=args.expected_gpu,
                require_fresh_simd_reference=args.require_fresh_simd_reference,
            )
        )
        if args.gpu_telemetry_csv is not None:
            try:
                payload = args.gpu_telemetry_csv.read_bytes()
                sample_count = max(0, len(payload.decode("utf-8").splitlines()) - 1)
            except (OSError, UnicodeDecodeError) as error:
                errors.append(f"telemetry artifact: {error}")
            else:
                start = record.get("gpu_proof_loop_started_unix_ns")
                finish = record.get("gpu_proof_loop_finished_unix_ns")
                proof_window_sample_count = 0
                try:
                    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
                    proof_window_sample_count = sum(
                        isinstance(start, int)
                        and isinstance(finish, int)
                        and start <= int(row["timestamp_unix_ns"]) <= finish
                        for row in rows
                    )
                except (csv.Error, KeyError, TypeError, ValueError):
                    pass
                errors.extend(
                    validate_gpu_telemetry_artifact(
                        record,
                        {
                            "schema": GPU_TELEMETRY_SCHEMA,
                            "columns": list(GPU_TELEMETRY_COLUMNS),
                            "path": str(args.gpu_telemetry_csv),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "remote_sha256": args.gpu_telemetry_remote_sha256,
                            "size_bytes": len(payload),
                            "remote_size_bytes": args.gpu_telemetry_remote_size,
                            "sample_count": sample_count,
                            "proof_window_sample_count": proof_window_sample_count,
                            "sampler_complete": True,
                            "sample_interval_ms": GPU_TELEMETRY_SAMPLE_INTERVAL_MS,
                            "max_gap_ns": GPU_TELEMETRY_MAX_GAP_NS,
                            "transport_equal": (
                                args.gpu_telemetry_remote_sha256
                                == hashlib.sha256(payload).hexdigest()
                                and args.gpu_telemetry_remote_size == len(payload)
                            ),
                        },
                    )
                )
    try:
        soundness = load_soundness_gate(args.soundness_gate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"CUDA soundness artifact: {error}")
    else:
        errors.extend(validate_soundness_gate(soundness, args.runtime_mode))
    return _print_contract_errors("GPU-native architecture contract", errors)


if __name__ == "__main__":
    raise SystemExit(main())
