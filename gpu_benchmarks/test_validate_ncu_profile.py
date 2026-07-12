#!/usr/bin/env python3
"""Adversarial tests for validate_ncu_profile.py."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    from gpu_benchmarks.validate_ncu_profile import ProfileError, validate_profile
except ModuleNotFoundError:
    from validate_ncu_profile import ProfileError, validate_profile


PROOF_SHA = "a" * 64
VERSION = "NVIDIA (R) Nsight Compute Command Line Profiler\nVersion 2022.3.0.0\n"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Fixture:
    def __init__(self, root: Path, *, synthetic: bool = False):
        self.synthetic = synthetic
        self.report = root / "capture.ncu-rep"
        self.version = root / "capture.version"
        self.import_output = root / "capture.import.csv"
        self.observation = root / "capture.observed.json"
        if synthetic:
            self.report.write_bytes(
                b"STWO synthetic Nsight Compute report v1\nlane=test\n"
            )
            import_prefix = b"STWO synthetic ncu import validation v1\n"
        else:
            file_header = b"\x08\x07"
            block = struct.pack("<I", 1) + b"x"
            self.report.write_bytes(
                b"NVP\0" + struct.pack("<I", len(file_header)) + file_header + block
            )
            import_prefix = b""
        self.version.write_text(VERSION, encoding="utf-8")
        self.import_output.write_bytes(
            import_prefix
            + b'"ID","Kernel Name","Metric Name"\n'
            + b'"1","relation_scan_kernel","metric"\n'
        )
        self.write_observation()

    def write_observation(self, **changes: object) -> None:
        observed = {
            "schema": "stwo.ncu-remote-observation.v1",
            "report_sha256": sha(self.report),
            "report_bytes": self.report.stat().st_size,
            "import_output_sha256": sha(self.import_output),
            "import_output_bytes": self.import_output.stat().st_size,
            "import_validated": True,
            "profiled_proof_sha256": PROOF_SHA,
        }
        observed.update(changes)
        self.observation.write_text(json.dumps(observed), encoding="utf-8")

    def validate(self) -> dict[str, object]:
        return validate_profile(
            self.report,
            self.version,
            self.import_output,
            self.observation,
            kernel_regex="relation_scan|stream_leaf_update",
            launch_count=10,
            set_name="full",
            profiled_proof_sha256=PROOF_SHA,
            synthetic=self.synthetic,
        )


class ValidateNcuProfileTests(unittest.TestCase):
    def test_actual_metadata_is_content_and_proof_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            metadata = fixture.validate()
            self.assertEqual(metadata["sha256"], sha(fixture.report))
            self.assertEqual(metadata["remote_sha256"], sha(fixture.report))
            self.assertEqual(metadata["remote_bytes"], fixture.report.stat().st_size)
            self.assertTrue(metadata["remote_import_validated"])
            self.assertEqual(metadata["import_output_path"], str(fixture.import_output))
            self.assertEqual(
                metadata["remote_import_output_sha256"],
                metadata["import_output_sha256"],
            )
            self.assertEqual(
                metadata["remote_import_output_bytes"],
                metadata["import_output_bytes"],
            )
            self.assertEqual(metadata["profiled_kernel_rows"], 1)
            self.assertFalse(metadata["synthetic"])
            self.assertEqual(metadata["profiled_proof_sha256"], PROOF_SHA)

    def test_synthetic_artifact_is_explicit_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), synthetic=True)
            metadata = fixture.validate()
            self.assertTrue(metadata["synthetic"])
            self.assertEqual(metadata["sha256"], sha(fixture.report))
            with self.assertRaisesRegex(ProfileError, "invalid NVIDIA report magic"):
                validate_profile(
                    fixture.report,
                    fixture.version,
                    fixture.import_output,
                    fixture.observation,
                    kernel_regex="kernel",
                    launch_count=1,
                    set_name="full",
                    profiled_proof_sha256=PROOF_SHA,
                )

    def test_arbitrary_and_truncated_reports_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.report.write_bytes(b"arbitrary bytes")
            fixture.write_observation()
            with self.assertRaisesRegex(ProfileError, "invalid NVIDIA report magic"):
                fixture.validate()
            fixture.report.write_bytes(b"NVP\0" + struct.pack("<I", 100) + b"short")
            fixture.write_observation()
            with self.assertRaisesRegex(ProfileError, "structurally truncated"):
                fixture.validate()

    def test_remote_observation_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            mutations = (
                ("report_sha256", "b" * 64, "report_sha256 mismatch"),
                ("report_bytes", 1, "report_bytes mismatch"),
                ("import_output_sha256", "b" * 64, "import_output_sha256 mismatch"),
                ("import_output_bytes", 1, "import_output_bytes mismatch"),
                ("profiled_proof_sha256", "b" * 64, "profiled_proof_sha256 mismatch"),
                ("import_validated", False, "import was not validated"),
            )
            for field, value, error in mutations:
                with self.subTest(field=field):
                    fixture.write_observation(**{field: value})
                    with self.assertRaisesRegex(ProfileError, error):
                        fixture.validate()

    def test_version_import_and_proof_contracts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.version.write_text("ncu 2022.3\n", encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "NVIDIA Nsight Compute prefix"):
                fixture.validate()
            fixture.version.write_text(VERSION, encoding="utf-8")
            fixture.import_output.write_bytes(b'"ID","Kernel Name","Metric Name"\n')
            fixture.write_observation()
            with self.assertRaisesRegex(ProfileError, "no profiled kernel data rows"):
                fixture.validate()
            fixture = Fixture(Path(directory))
            fixture.import_output.write_bytes(
                b'"ID","Kernel Name","Metric Name"\n'
                b'"1","unrelated_kernel","metric"\n'
            )
            fixture.write_observation()
            with self.assertRaisesRegex(ProfileError, "outside the declared regex"):
                fixture.validate()
            fixture = Fixture(Path(directory))
            fixture.import_output.write_bytes(b"not csv")
            fixture.write_observation()
            with self.assertRaisesRegex(ProfileError, "lacks the raw CSV header"):
                fixture.validate()
            fixture = Fixture(Path(directory))
            with self.assertRaisesRegex(ProfileError, "proof SHA256 is invalid"):
                validate_profile(
                    fixture.report,
                    fixture.version,
                    fixture.import_output,
                    fixture.observation,
                    kernel_regex="kernel",
                    launch_count=1,
                    set_name="full",
                    profiled_proof_sha256="not-a-hash",
                )


if __name__ == "__main__":
    unittest.main()
