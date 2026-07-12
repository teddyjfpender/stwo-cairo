#!/usr/bin/env python3
"""Validate and describe one retained Nsight Compute report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import BinaryIO


VERSION_PREFIX = "NVIDIA (R) Nsight Compute Command Line Profiler"
SYNTHETIC_REPORT_PREFIX = b"STWO synthetic Nsight Compute report v1\n"
SYNTHETIC_IMPORT_PREFIX = b"STWO synthetic ncu import validation v1\n"
OBSERVATION_KEYS = {
    "schema",
    "report_sha256",
    "report_bytes",
    "import_output_sha256",
    "import_output_bytes",
    "import_validated",
    "profiled_proof_sha256",
}


class ProfileError(ValueError):
    pass


def _fingerprint(stream: BinaryIO) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    while chunk := stream.read(1024 * 1024):
        if len(prefix) < 8192:
            prefix += chunk[: 8192 - len(prefix)]
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size, prefix


def _fingerprint_file(path: Path, label: str) -> tuple[str, int, bytes]:
    try:
        with path.open("rb") as stream:
            return _fingerprint(stream)
    except OSError as error:
        raise ProfileError(f"cannot read {label}: {error}") from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_report_shape(prefix: bytes, size: int, *, synthetic: bool) -> None:
    if synthetic:
        if not prefix.startswith(SYNTHETIC_REPORT_PREFIX):
            raise ProfileError("synthetic ncu report marker is invalid")
        return
    if len(prefix) < 8 or prefix[:4] not in (b"NVP\0", b"NVR\0"):
        raise ProfileError("ncu report has an invalid NVIDIA report magic")
    header_bytes = int.from_bytes(prefix[4:8], "little")
    if header_bytes <= 0 or size < 8 + header_bytes + 4:
        raise ProfileError("ncu report is structurally truncated")


def _load_observation(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > 16_384:
            raise ProfileError("remote ncu observation is unexpectedly large")
        observed = json.loads(path.read_text(encoding="utf-8"))
    except ProfileError:
        raise
    except (OSError, ValueError) as error:
        raise ProfileError(f"cannot read remote ncu observation: {error}") from error
    if not isinstance(observed, dict) or set(observed) != OBSERVATION_KEYS:
        raise ProfileError("remote ncu observation has an invalid shape")
    if observed.get("schema") != "stwo.ncu-remote-observation.v1":
        raise ProfileError("remote ncu observation has an invalid schema")
    return observed


def _profiled_kernel_rows(path: Path, kernel_pattern: re.Pattern[str]) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            kernel_index = None
            header_width = None
            rows = 0
            for row in reader:
                if "Kernel Name" in row:
                    kernel_index = row.index("Kernel Name")
                    header_width = len(row)
                    continue
                if (
                    kernel_index is not None
                    and header_width is not None
                    and len(row) == header_width
                    and row[kernel_index].strip()
                ):
                    if kernel_pattern.search(row[kernel_index]) is None:
                        raise ProfileError(
                            "ncu import output contains a kernel outside the declared regex"
                        )
                    rows += 1
    except (OSError, UnicodeError, csv.Error) as error:
        raise ProfileError(f"cannot parse ncu import output: {error}") from error
    if rows == 0:
        raise ProfileError("ncu import output has no profiled kernel data rows")
    return rows


def validate_profile(
    report: Path,
    version_file: Path,
    import_output: Path,
    observation_file: Path,
    *,
    kernel_regex: str,
    launch_count: int,
    set_name: str,
    profiled_proof_sha256: str,
    synthetic: bool = False,
) -> dict[str, object]:
    report_sha, report_bytes, report_prefix = _fingerprint_file(report, "ncu report")
    if report_bytes == 0:
        raise ProfileError("ncu report is empty")
    _validate_report_shape(report_prefix, report_bytes, synthetic=synthetic)

    import_sha, import_bytes, import_prefix = _fingerprint_file(
        import_output, "ncu import output"
    )
    if import_bytes == 0 or b'"Kernel Name"' not in import_prefix:
        raise ProfileError("ncu import output is empty or lacks the raw CSV header")
    if synthetic and not import_prefix.startswith(SYNTHETIC_IMPORT_PREFIX):
        raise ProfileError("synthetic ncu import marker is invalid")

    try:
        if version_file.stat().st_size > 16_384:
            raise ProfileError("ncu version is unexpectedly large")
        version = version_file.read_text(encoding="utf-8").strip()
    except ProfileError:
        raise
    except (OSError, UnicodeError) as error:
        raise ProfileError(f"cannot read ncu version: {error}") from error
    if version.partition("\n")[0] != VERSION_PREFIX:
        raise ProfileError("ncu version does not have the NVIDIA Nsight Compute prefix")
    if not kernel_regex or any(ord(char) < 32 for char in kernel_regex):
        raise ProfileError("kernel regex is empty or contains control characters")
    try:
        kernel_pattern = re.compile(kernel_regex)
    except re.error as error:
        raise ProfileError(f"kernel regex is invalid: {error}") from error
    profiled_kernel_rows = _profiled_kernel_rows(import_output, kernel_pattern)
    if isinstance(launch_count, bool) or launch_count <= 0 or launch_count > 1_000_000:
        raise ProfileError("launch count must be between 1 and 1000000")
    if not set_name or any(ord(char) < 32 for char in set_name):
        raise ProfileError("set name is empty or contains control characters")
    if not _is_sha256(profiled_proof_sha256):
        raise ProfileError("profiled proof SHA256 is invalid")

    observed = _load_observation(observation_file)
    if observed.get("import_validated") is not True:
        raise ProfileError("remote ncu import was not validated")
    comparisons = {
        "report_sha256": report_sha,
        "report_bytes": report_bytes,
        "import_output_sha256": import_sha,
        "import_output_bytes": import_bytes,
        "profiled_proof_sha256": profiled_proof_sha256,
    }
    for field, local_value in comparisons.items():
        if observed.get(field) != local_value:
            raise ProfileError(f"remote/local ncu {field} mismatch")

    return {
        "schema": "stwo.ncu-profile.v1",
        "path": str(report),
        "sha256": report_sha,
        "bytes": report_bytes,
        "kernel_regex": kernel_regex,
        "launch_count": launch_count,
        "set": set_name,
        "ncu_version": version,
        "synthetic": synthetic,
        "remote_sha256": observed["report_sha256"],
        "remote_bytes": observed["report_bytes"],
        "remote_import_validated": True,
        "import_output_path": str(import_output),
        "import_output_sha256": import_sha,
        "import_output_bytes": import_bytes,
        "remote_import_output_sha256": observed["import_output_sha256"],
        "remote_import_output_bytes": observed["import_output_bytes"],
        "profiled_kernel_rows": profiled_kernel_rows,
        "profiled_proof_sha256": profiled_proof_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--version-file", required=True, type=Path)
    parser.add_argument("--import-output", required=True, type=Path)
    parser.add_argument("--observation-file", required=True, type=Path)
    parser.add_argument("--kernel-regex", required=True)
    parser.add_argument("--launch-count", required=True, type=int)
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--profiled-proof-sha256", required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    try:
        metadata = validate_profile(
            args.report,
            args.version_file,
            args.import_output,
            args.observation_file,
            kernel_regex=args.kernel_regex,
            launch_count=args.launch_count,
            set_name=args.set_name,
            profiled_proof_sha256=args.profiled_proof_sha256,
            synthetic=args.synthetic,
        )
    except ProfileError as error:
        parser.error(str(error))
    print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
