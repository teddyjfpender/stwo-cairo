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


if __name__ == "__main__":
    unittest.main()
