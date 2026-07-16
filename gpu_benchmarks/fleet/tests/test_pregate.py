from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpufleet import pregate


class PregateInputAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stwo = self.root / "stwo"
        self.stwo_cairo = self.root / "stwo-cairo"
        self.inputs = self.root / "inputs"
        self.stwo.mkdir()
        self.inputs.mkdir()
        (self.stwo_cairo / "gpu_benchmarks/pie").mkdir(parents=True)
        self.digests = {}
        for index, name in enumerate(pregate.SN_INPUT_NAMES, 1):
            payload = f"sealed-sn-{index}".encode()
            (self.inputs / name).write_bytes(payload)
            self.digests[name] = hashlib.sha256(payload).hexdigest()
        self._write_manifest()

    @property
    def manifest(self) -> Path:
        return self.stwo_cairo / "gpu_benchmarks/pie/ADAPTED_SHA256SUMS"

    def _write_manifest(self, lines: list[str] | None = None) -> None:
        if lines is None:
            lines = [f"{self.digests[name]}  {name}" for name in pregate.SN_INPUT_NAMES]
        self.manifest.write_text("\n".join(lines) + "\n")

    def _admit(self):
        with mock.patch.dict(os.environ, {pregate.SN_INPUT_ENV: str(self.inputs)}):
            return pregate._admit_sn_inputs(self.stwo_cairo)

    def test_exact_inputs_build_one_absolute_four_sn_union_command(self) -> None:
        paths, observed, manifest_hash = self._admit()
        self.assertEqual([path.name for path in paths], list(pregate.SN_INPUT_NAMES))
        self.assertTrue(all(path.is_absolute() for path in paths))
        self.assertEqual(observed, self.digests)
        self.assertEqual(manifest_hash, hashlib.sha256(self.manifest.read_bytes()).hexdigest())

        checks = pregate._init_checks(self.stwo, self.stwo_cairo, paths)
        command = next(argv for name, argv, _ in checks if name.startswith("kernel_emit"))
        self.assertEqual(command.count("--input-bincode"), 4)
        self.assertEqual(command[command.index("--stwo-root") + 1], str(self.stwo.resolve()))
        input_paths = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--input-bincode"
        ]
        self.assertEqual(input_paths, [str(path) for path in paths])

    def test_missing_environment_fails_closed(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, pregate.SN_INPUT_ENV):
                pregate._admit_sn_inputs(self.stwo_cairo)

    def test_manifest_shape_rejects_duplicate_extra_missing_and_malformed_rows(self) -> None:
        valid = [f"{self.digests[name]}  {name}" for name in pregate.SN_INPUT_NAMES]
        cases = {
            "duplicate": valid + [valid[0]],
            "extra": valid + [f"{'0' * 64}  SN_PIE_5.adapted.bin"],
            "missing": valid[:-1],
            "malformed": ["not-a-digest  SN_PIE_1.adapted.bin", *valid[1:]],
        }
        for label, lines in cases.items():
            with self.subTest(label=label):
                self._write_manifest(lines)
                with self.assertRaises(ValueError):
                    self._admit()

    def test_missing_file_and_digest_drift_fail_closed(self) -> None:
        missing = self.inputs / pregate.SN_INPUT_NAMES[-1]
        payload = missing.read_bytes()
        missing.unlink()
        with self.assertRaises(OSError):
            self._admit()
        missing.write_bytes(payload + b"drift")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._admit()

    def test_failed_admission_overwrites_prior_green_stamp(self) -> None:
        stamp = self.root / "pregate.json"
        stamp.write_text(json.dumps({"ok": True}))
        with (
            mock.patch.object(pregate, "STAMP", stamp),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertFalse(pregate.run(self.stwo, self.stwo_cairo))
        self.assertFalse(json.loads(stamp.read_text())["ok"])

    def test_fresh_receipt_is_bound_to_source_and_input_identities(self) -> None:
        stamp = self.root / "pregate.json"
        paths, input_hashes, manifest_hash = self._admit()
        source_identity = {
            "stwo": {"head": "a", "worktree_sha256": "b"},
            "stwo_cairo": {"head": "c", "worktree_sha256": "d"},
        }
        input_identity = {
            "manifest_sha256": manifest_hash,
            "inputs": input_hashes,
        }
        with mock.patch.object(pregate, "STAMP", stamp):
            pregate._write_stamp(True, [], source_identity, input_identity)
            with (
                mock.patch.object(pregate, "_source_identity", return_value=source_identity),
                mock.patch.object(
                    pregate,
                    "_admit_sn_inputs",
                    return_value=(paths, input_hashes, manifest_hash),
                ),
            ):
                self.assertTrue(pregate.is_fresh(self.stwo, self.stwo_cairo))
            changed = {**source_identity, "stwo": {"head": "changed"}}
            with (
                mock.patch.object(pregate, "_source_identity", return_value=changed),
                mock.patch.object(
                    pregate,
                    "_admit_sn_inputs",
                    return_value=(paths, input_hashes, manifest_hash),
                ),
            ):
                self.assertFalse(pregate.is_fresh(self.stwo, self.stwo_cairo))


if __name__ == "__main__":
    unittest.main()
