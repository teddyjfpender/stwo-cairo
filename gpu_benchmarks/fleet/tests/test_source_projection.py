from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpufleet.source_projection import projection_identity

STAGER = Path(__file__).resolve().parents[2] / "loop/stage_source_projection.sh"


class SourceProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self._git("init", "-q")
        self._git("config", "user.name", "Test")
        self._git("config", "user.email", "test@example.invalid")
        (self.repo / "tracked.txt").write_text("base\n")
        self._git("add", "tracked.txt")
        self._git("commit", "-qm", "base")

    def _git(self, *args: str) -> bytes:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True
        ).stdout

    def test_hash_binds_exact_tracked_and_untracked_projection(self) -> None:
        (self.repo / "tracked.txt").write_text("changed\n")
        (self.repo / "plain.bin").write_bytes(b"plain")
        executable = self.repo / "run.sh"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        os.symlink("tracked.txt", self.repo / "link")
        excluded = self.repo / "gpu_benchmarks/loop/results/run"
        excluded.mkdir(parents=True)
        (excluded / "large.log").write_bytes(b"ignored runtime evidence")

        diff = self._git(
            "diff",
            "--binary",
            "HEAD",
            "--",
            ".",
            ":(exclude)gpu_benchmarks/loop/results/**",
            ":(exclude)gpu_benchmarks/loop/ledger.jsonl",
            ":(exclude)gpu_benchmarks/loop/pod.conf",
            ":(exclude)gpu_benchmarks/pie/sn/**",
            ":(exclude)gpu_benchmarks/pie/*.zip",
            ":(exclude)gpu_benchmarks/results/**",
            ":(exclude)gpu_benchmarks/fleet/results/**",
            ":(exclude)gpu_benchmarks/fleet/fleet_report.json",
            ":(exclude)gpu_benchmarks/fleet/fleet.conf",
            ":(exclude)gpu_benchmarks/fleet/ledger_costs.jsonl",
            ":(exclude)gpu_benchmarks/fleet/pods.conf*",
        )
        stream = bytearray(diff)
        records = {
            b"link": (b"symlink", hashlib.sha256(b"tracked.txt").hexdigest().encode()),
            b"plain.bin": (b"regular", hashlib.sha256(b"plain").hexdigest().encode()),
            b"run.sh": (
                b"executable",
                hashlib.sha256(b"#!/bin/sh\n").hexdigest().encode(),
            ),
        }
        for path in sorted(records):
            kind, digest = records[path]
            stream.extend(b"untracked-" + kind + b"\0" + path + b"\0" + digest + b"\0")

        observed = projection_identity(self.repo)
        self.assertEqual(observed["head"], self._git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(observed["worktree_sha256"], hashlib.sha256(stream).hexdigest())

    def test_excluded_runtime_content_does_not_change_hash(self) -> None:
        paths = [
            self.repo / "gpu_benchmarks/pie/sn/input.bin",
            self.repo / "gpu_benchmarks/results/bootstrap/pod.log",
            self.repo / "gpu_benchmarks/fleet/results/run.json",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"one")
        before = projection_identity(self.repo)
        for path in paths:
            path.write_bytes(b"two")
        self.assertEqual(projection_identity(self.repo), before)

    def test_tracked_controller_state_does_not_invalidate_projection(self) -> None:
        paths = [
            self.repo / "gpu_benchmarks/fleet/ledger_costs.jsonl",
            self.repo / "gpu_benchmarks/fleet/pods.conf",
            self.repo / "gpu_benchmarks/loop/pod.conf",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("before\n")
        self._git("add", *(str(path.relative_to(self.repo)) for path in paths))
        self._git("commit", "-qm", "controller state")
        before = projection_identity(self.repo)
        for path in paths:
            path.write_text("after\n")
        self.assertEqual(projection_identity(self.repo), before)

    def test_stager_omits_the_same_controller_runtime_paths(self) -> None:
        source = self.repo / "source.py"
        source.write_text("kept = True\n")
        excluded = [
            self.repo / "gpu_benchmarks/fleet/ledger_costs.jsonl",
            self.repo / "gpu_benchmarks/fleet/pods.conf",
            self.repo / "gpu_benchmarks/loop/pod.conf",
            self.repo / "gpu_benchmarks/results/bootstrap/pod.log",
        ]
        for path in excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("runtime\n")
        self._git(
            "add",
            "source.py",
            *(str(path.relative_to(self.repo)) for path in excluded),
        )
        self._git("commit", "-qm", "projection fixture")

        with tempfile.TemporaryDirectory() as parent:
            destination = Path(parent) / "projection"
            subprocess.run(
                [str(STAGER), str(self.repo), str(destination)],
                check=True,
                capture_output=True,
            )
            self.assertEqual((destination / "source.py").read_text(), "kept = True\n")
            for path in excluded:
                self.assertFalse((destination / path.relative_to(self.repo)).exists())


if __name__ == "__main__":
    unittest.main()
