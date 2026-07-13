#!/usr/bin/env python3
"""Regression tests for generated benchmark shell launchers."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ShellLauncherTests(unittest.TestCase):
    def test_generated_heredocs_are_not_captured_by_command_substitution(self) -> None:
        source = (ROOT / "loop" / "perf_gates.sh").read_text(encoding="utf-8")
        self.assertNotIn('="$(cat <<EOF', source)
        ncu_start = source.index("    IFS= read -r -d '' ncu_body <<EOF || true")
        body_start = source.index("\n", ncu_start) + 1
        body_end = source.index("\nEOF\n", body_start)
        template = source[body_start:body_end]

        # macOS Bash 3.2 closes a command substitution at an unescaped ')'
        # inside a nested heredoc. Read the heredoc directly, then validate the
        # same expanded launcher that perf_gates executes.
        script = f'''IFS= read -r -d '' body <<EOF || true
{template}
EOF
printf '%s\n' "$body" | bash -n
'''
        result = subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_qualification_probe_reuses_one_soundness_artifact(self) -> None:
        source = (ROOT / "loop" / "bench_loop.sh").read_text(encoding="utf-8")
        self.assertIn(
            'LOCAL_SOUNDNESS_GATE="$REUSE_SOUNDNESS_GATE"',
            source,
        )
        self.assertNotIn(
            'cp "$REUSE_SOUNDNESS_GATE" "$LOCAL_SOUNDNESS_GATE"',
            source,
        )

    def test_reset_container_is_bootstrapped_before_content_only_sync(self) -> None:
        source = (ROOT / "loop" / "bench_loop.sh").read_text(encoding="utf-8")
        pod_run = (ROOT / "loop" / "pod_run.sh").read_text(encoding="utf-8")
        orchestration = source.index("# Orchestration")
        bootstrap = source.index("  bootstrap_pod", orchestration)
        sync = source.index("  sync_repos", orchestration)
        self.assertLess(bootstrap, sync)
        self.assertIn("command -v rsync", source)
        self.assertIn("apt-get install -y -qq build-essential", source)
        self.assertIn("gcc g++ make ar ld", source)
        self.assertIn("/workspace/.cargo-persist", source)
        self.assertIn("rustup toolchain install >> '${POD_BUILD_LOG}' 2>&1 &&", source)

        projection_body = source[
            source.index("verify_remote_source_projection()") : source.index(
                "seal_source_projection()"
            )
        ]
        sync_body = source[
            source.index("sync_repos()") : source.index("build_pod()")
        ]
        self.assertEqual(projection_body.count("--no-perms"), 2)
        self.assertEqual(sync_body.count("--no-perms"), 2)
        self.assertEqual(pod_run.count("--no-perms"), 2)
        preserved_inputs = "--exclude='gpu_benchmarks/pie/sn/'"
        self.assertEqual(projection_body.count(preserved_inputs), 1)
        self.assertEqual(sync_body.count(preserved_inputs), 1)
        self.assertEqual(pod_run.count(preserved_inputs), 1)
        self.assertNotIn("gpu_benchmarks/pie/sn/*.zip", source)
        self.assertNotIn("gpu_benchmarks/pie/sn/*.zip", pod_run)


if __name__ == "__main__":
    unittest.main()
