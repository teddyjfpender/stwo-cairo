#!/usr/bin/env python3
"""Regression tests for generated benchmark shell launchers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from validate_replacement_v1_reuse import require_resident_reuse


ROOT = Path(__file__).resolve().parent


class ShellLauncherTests(unittest.TestCase):
    def test_replacement_sn2_reuse_gate_rejects_each_mutation(self) -> None:
        source = (
            ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        ).read_text(encoding="utf-8")
        validator = source.index("checkpoint_validate_sn2()")
        valid = {
            "gpu_shape_executable_materialization": "reused",
            "gpu_shape_executable_cache_hits": 1,
            "gpu_shape_executable_cache_misses": 1,
            "gpu_shape_executable_cache_compilations": 1,
            "gpu_shape_executable_cache_source_generation_passes": 1,
            "gpu_shape_executable_cache_binding_recipe_compilations": 1,
            "gpu_shape_executable_cache_capacity_rejections": 0,
            "gpu_workspace_materialization": "reused",
            "gpu_hot_allocations": 0,
        }
        require_resident_reuse(valid, 2)
        require_resident_reuse(
            {**valid, "gpu_shape_executable_cache_hits": 5}, 6
        )
        self.assertIn(
            "from validate_replacement_v1_reuse import require_resident_reuse",
            source[validator:],
        )
        self.assertIn("require_resident_reuse(r, reps)", source[validator:])

        for field, expected in valid.items():
            mutations = (
                ("compiled", None)
                if field == "gpu_shape_executable_materialization"
                else ("materialized", None)
                if field == "gpu_workspace_materialization"
                else (expected + 1, bool(expected), None)
            )
            for mutation in mutations:
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaises(SystemExit):
                        require_resident_reuse({**valid, field: mutation}, 2)

    def test_source_projection_cache_exclusions_only_cover_ignored_files(self) -> None:
        projection_files = []
        for command in (
            ["git", "ls-files", "-z"],
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        ):
            projection_files.extend(
                subprocess.run(
                    command,
                    cwd=ROOT.parent,
                    check=True,
                    capture_output=True,
                ).stdout.decode().split("\0")
            )
        bytecode = [
            path
            for path in projection_files
            if "__pycache__" in Path(path).parts
            or Path(path).suffix in {".pyc", ".pyo"}
        ]
        self.assertEqual(bytecode, [])

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

    def test_python_caches_are_symmetric_and_cannot_shadow_soundness(self) -> None:
        source = (ROOT / "loop" / "bench_loop.sh").read_text(encoding="utf-8")
        soundness = source.split("run_cuda_soundness_gate() {", 1)[1].split(
            "# Synthetic run output", 1
        )[0]
        launcher = soundness.split("run_ssh \"cat > '${gate_sh}'\" <<EOF", 1)[1].split(
            "\nEOF\n", 1
        )[0]
        bytecode_guard = "export PYTHONDONTWRITEBYTECODE=1"
        runner = "python3 gpu_benchmarks/run_cuda_soundness_gate.py"
        self.assertIn(bytecode_guard, launcher)
        self.assertLess(launcher.index(bytecode_guard), launcher.index(runner))
        cache_dir_cleanup = "-type d -name __pycache__ -prune -exec rm -rf {} + &&"
        cache_file_cleanup = (
            "-type f \\( -name '*.pyc' -o -name '*.pyo' \\) -exec rm -f {} + &&"
        )
        self.assertIn(cache_dir_cleanup, launcher)
        self.assertIn(cache_file_cleanup, launcher)
        self.assertEqual(launcher.count("find '${CAIRO_POD}/gpu_benchmarks'"), 2)
        self.assertNotIn("find '${STWO_POD}' '${CAIRO_POD}'", launcher)
        self.assertLess(launcher.index(cache_dir_cleanup), launcher.index(runner))
        self.assertLess(launcher.index(cache_file_cleanup), launcher.index(runner))

        projection = source.split("verify_remote_source_projection() {", 1)[1].split(
            "seal_source_projection() {", 1
        )[0]
        exclusion_guard = source.split(
            "verify_projection_exclusions_are_ignored() {", 1
        )[1].split("verify_remote_source_projection() {", 1)[0]
        sync = source.split("sync_repos() {", 1)[1].split("build_pod() {", 1)[0]
        source_guard = "source projection cannot exclude tracked or unignored Python bytecode"
        self.assertIn(source_guard, exclusion_guard)
        self.assertIn('git -C "$repo" ls-files -z', exclusion_guard)
        self.assertIn(
            'git -C "$repo" ls-files --others --exclude-standard -z',
            exclusion_guard,
        )
        self.assertIn("while IFS= read -r -d '' file", exclusion_guard)
        guard_call = "verify_projection_exclusions_are_ignored"
        self.assertIn(guard_call, sync)
        self.assertIn(guard_call, projection)
        self.assertLess(sync.index(guard_call), sync.index("rsync stwo -> pod"))
        self.assertLess(projection.index(guard_call), projection.index("run_rsync"))
        for cache_exclusion in (
            "--exclude='__pycache__/'",
            "--exclude='*.py[co]'",
        ):
            self.assertEqual(sync.count(cache_exclusion), 2)
            self.assertEqual(projection.count(cache_exclusion), 2)

    def test_python_cache_rsync_filters_match_delete_and_dry_run_semantics(self) -> None:
        rsync = shutil.which("rsync")
        if rsync is None:
            self.skipTest("rsync is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "pkg" / "__pycache__").mkdir(parents=True)
            (destination / "pkg" / "__pycache__").mkdir(parents=True)
            (source / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (source / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"local")
            (source / "pkg" / "legacy.pyo").write_bytes(b"local")
            stale_cache = destination / "pkg" / "__pycache__" / "stale.pyc"
            stale_cache.write_bytes(b"remote")

            filters = ["--exclude=__pycache__/", "--exclude=*.py[co]"]
            subprocess.run(
                [rsync, "-ac", "--delete", *filters, f"{source}/", f"{destination}/"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((destination / "pkg" / "module.py").is_file())
            self.assertFalse(
                (destination / "pkg" / "__pycache__" / "module.pyc").exists()
            )
            self.assertFalse((destination / "pkg" / "legacy.pyo").exists())
            self.assertTrue(stale_cache.is_file())

            verify = subprocess.run(
                [
                    rsync,
                    "-acn",
                    "--delete",
                    "--itemize-changes",
                    *filters,
                    f"{source}/",
                    f"{destination}/",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.stdout, "")

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
