#!/usr/bin/env python3
"""Regression tests for generated benchmark shell launchers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from validate_replacement_v1_reuse import require_resident_reuse


ROOT = Path(__file__).resolve().parent
NUMERATOR_SCHEMA = "stwo.sn3_quotient_numerator_hybrid.host_wall.v5"
SOURCE_SHA = "12" * 32
MODULE_SHA = "34" * 32


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sn3_numerator_record() -> dict[str, object]:
    digest = "56" * 32
    identity = {
        "topology_fixture_blake3": "ea31e3ff054c8d12d32d5b84a3d712987b31bb1fd3fb044fb27758453b49fbda",
        "input_recipe_blake3": "e4c2f871c2d05b81588a5407f06cb49c7ed76834d2e363d2214bd34e7defcf31",
        "timed_sample_index": 0,
        "timed_sample_causally_validated": True,
        "capture_revalidated": True,
        "post_timing_revalidated": True,
    }
    for field in (
        "eager_legacy_blake3",
        "eager_hybrid_blake3",
        "captured_legacy_blake3",
        "captured_hybrid_blake3",
        "timed_legacy_blake3",
        "timed_hybrid_blake3",
        "post_timing_legacy_blake3",
        "post_timing_hybrid_blake3",
    ):
        identity[field] = digest
    return {
        "schema": NUMERATOR_SCHEMA,
        "topology": {
            "group_logs": [23, 19, 20, 6, 16, 18, 8, 7, 21, 14, 17, 11, 23, 15, 10, 4, 13, 12, 22],
            "groups": 19,
            "eligible_groups": 18,
            "legacy_groups": 1,
            "coefficient_columns": 161,
            "coefficient_sources": 152,
            "total_batches": 74,
            "coefficient_batches": 71,
            "terms": 6_341,
        },
        "bytes": {
            "legacy_logical_output": 59_993_989_376,
            "hybrid_logical_output": 20_266_867_968,
            "validated_numerator_output": 402_644_224,
            "validated_auxiliary_output": 912,
            "validated_canonical_output": 402_645_136,
            "shared_data_dual_workspace_arena": 41_889_121_376,
            "workspace_span_each": 67_901_168,
            "second_workspace_arena_delta": 67_901_152,
        },
        "device_memory": {
            "total": 85_000_000_000,
            "free_before_arena": 80_000_000_000,
            "free_after_arena": 38_000_000_000,
            "isolated_pool_used_after_arena": 41_889_121_376,
            "isolated_pool_reserved_after_arena": 42_000_000_000,
        },
        "identity": identity,
        "artifact_identity": {
            "identity_complete": True,
            "source_projection_sha256": SOURCE_SHA,
            "cuda_module_sha256": MODULE_SHA,
            "cuda_build_mode": "cuda",
        },
        "warmups": 3,
        "iterations": 5,
        "minimum_iterations": 5,
        "samples_ms": {
            "legacy": [10.0, 11.0, 12.0, 13.0, 14.0],
            "hybrid": [5.0, 6.0, 7.0, 8.0, 9.0],
        },
        "host_wall_ms": {
            "legacy": {"p50": 12.0, "p95": 14.0},
            "hybrid": {"p50": 7.0, "p95": 9.0},
        },
        "speedup": {"p50": 1.714285714, "p95": 1.555555556},
    }


class ShellLauncherTests(unittest.TestCase):
    def test_replacement_sn2_numerator_v5_seals_diagnostic(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        source = common.read_text(encoding="utf-8")
        self.assertIn(f"CHECKPOINT_NUMERATOR_SCHEMA={NUMERATOR_SCHEMA}", source)
        self.assertEqual(source.count(NUMERATOR_SCHEMA), 1)
        self.assertNotIn(NUMERATOR_SCHEMA.removesuffix("5") + "4", source)
        self.assertIn(
            'checkpoint_validate_numerator_record "$out" "$source_sha" "$module_sha"',
            source,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_bench = root / "gpu_bench"
            aot_check = root / "aot_index_check"
            aot_manifest = root / "aot_manifest.json"
            proof = root / "fixture.proof.bin"
            for path, payload in (
                (gpu_bench, b"gpu-bench-v1"),
                (aot_check, b"aot-check-v1"),
                (aot_manifest, b"[]\n"),
                (proof, b"proof-v1"),
            ):
                path.write_bytes(payload)

            gpu_bench_sha = file_sha256(gpu_bench)
            aot_check_sha = file_sha256(aot_check)
            aot_manifest_sha = file_sha256(aot_manifest)
            proof_sha = file_sha256(proof)
            artifacts = {
                "source_input_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.source-input-identity.v1",
                    "source": {"stwo": "head-a", "stwo_cairo": "head-b"},
                    "inputs": {"SN_PIE_2.zip": "input-a"},
                },
                "hardware_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.hardware-identity.v2",
                    "name": "test H100",
                },
                "build_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.build-identity.v1",
                    "gpu_bench_sha256": gpu_bench_sha,
                    "aot_index_check_sha256": aot_check_sha,
                    "aot_manifest_sha256": aot_manifest_sha,
                },
                "adapted_input_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.adapted-input-identity.v1",
                    "sha256": "adapted-v1",
                    "adapter_binary_sha256": gpu_bench_sha,
                },
                "fp256_carry_oracles.json": {"pass": True},
                "numerator_ab.json": valid_sn3_numerator_record(),
                "aot_identity.json": {
                    "gpu_bench_sha256": gpu_bench_sha,
                    "checker_binary_sha256": aot_check_sha,
                    "manifest_sha256": aot_manifest_sha,
                    "loaded_manifest_hash": "loaded-v1",
                },
                "record.json": {
                    "checkpoint_validation": {
                        "verdict": "PASS",
                        "mode": "diagnostic",
                    },
                    "gpu_proof_blake3": "ab" * 32,
                    "gpu_protocol_key": "protocol-v1",
                    "gpu_shape_executable_topology_digest": "topology-v1",
                    "gpu_prepared_numerator_eligible_groups": 18,
                    "gpu_prepared_numerator_legacy_groups": 1,
                },
            }
            for name, record in artifacts.items():
                (root / f"fixture.{name}").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            (root / "fixture.proof.sha256.txt").write_text(
                f"{proof_sha}  {proof.name}\n", encoding="utf-8"
            )

            seal = root / "checkpoint.seal.json"
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                "STWO": str(root / "stwo"),
                "RUN": str(root),
                "COMMON": str(common),
                "TEST_GPU_BENCH": str(gpu_bench),
                "TEST_AOT_CHECK": str(aot_check),
                "TEST_AOT_MANIFEST": str(aot_manifest),
                "TEST_SEAL": str(seal),
                "TEST_SOURCE_SHA": SOURCE_SHA,
                "TEST_MODULE_SHA": MODULE_SHA,
            }
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    'CHECKPOINT_GPU_BENCH="$TEST_GPU_BENCH"; '
                    'CHECKPOINT_AOT_CHECK="$TEST_AOT_CHECK"; '
                    'CHECKPOINT_AOT_MANIFEST="$TEST_AOT_MANIFEST"; '
                    'CHECKPOINT_SEAL="$TEST_SEAL"; '
                    'checkpoint_validate_numerator_record '
                    '"$RUN/fixture.numerator_ab.json" "$TEST_SOURCE_SHA" "$TEST_MODULE_SHA"; '
                    "checkpoint_seal_diagnostic",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            sealed = json.loads(seal.read_text(encoding="utf-8"))
            self.assertTrue(sealed["diagnostic_pass"])
            self.assertEqual(
                sealed["receipts_sha256"]["numerator"],
                file_sha256(root / "fixture.numerator_ab.json"),
            )

    def test_replacement_sn2_aot_identity_derives_exact_key_count(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "aot_manifest.json"
            manifest.write_text(
                json.dumps([{"cache_key": char * 16} for char in "a12"]) + "\n",
                encoding="utf-8",
            )
            checker = root / "aot_index_check"
            checker.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '{\"pass\":true,\"sm\":90,"
                "\"loaded_manifest_hash\":\"1234567890abcdef\","
                "\"required_unique_key_count\":'\"${TEST_COUNT:-3}\"',"
                "\"missing_keys\":[]}'\n",
                encoding="utf-8",
            )
            checker.chmod(0o755)
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "RUN": str(root),
                "COMMON": str(common),
                "TEST_MANIFEST": str(manifest),
                "TEST_MANIFEST_SHA": file_sha256(manifest),
                "TEST_CHECKER": str(checker),
            }
            command = (
                'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                'CHECKPOINT_AOT_MANIFEST="$TEST_MANIFEST"; '
                'CHECKPOINT_AOT_MANIFEST_SHA256="$TEST_MANIFEST_SHA"; '
                'CHECKPOINT_GPU_BENCH="$TEST_CHECKER"; '
                'CHECKPOINT_AOT_CHECK="$TEST_CHECKER"; checkpoint_aot_identity'
            )

            accepted = subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            identity = json.loads(
                (root / "fixture.aot_identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(identity["schema"], "stwo.replacement-v1-sn2.aot-identity.v1")
            self.assertEqual(identity["required_unique_key_count"], 3)

            rejected = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True,
                env={**env, "TEST_COUNT": "4"},
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("identity/coverage failed", rejected.stderr)

    def test_replacement_sn2_numerator_validator_rejects_mutations(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "numerator.json"
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                "STWO": str(root / "stwo"),
                "RUN": str(root),
                "COMMON": str(common),
                "RECORD": str(record_path),
                "TEST_SOURCE_SHA": SOURCE_SHA,
                "TEST_MODULE_SHA": MODULE_SHA,
            }

            def validate(record: dict[str, object]) -> subprocess.CompletedProcess[str]:
                record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$COMMON"; checkpoint_validate_numerator_record '
                        '"$RECORD" "$TEST_SOURCE_SHA" "$TEST_MODULE_SHA"',
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )

            valid = valid_sn3_numerator_record()
            accepted = validate(valid)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn('"exact_numerator_ab": "PASS"', accepted.stdout)

            mutations = (
                ("schema", ("schema",), NUMERATOR_SCHEMA.removesuffix("5") + "4"),
                ("group logs", ("topology", "group_logs", 0), 22),
                ("group count", ("topology", "eligible_groups"), 17),
                ("batch count", ("topology", "total_batches"), 73),
                ("legacy modeled bytes", ("bytes", "legacy_logical_output"), 59_993_989_375),
                ("hybrid modeled bytes", ("bytes", "hybrid_logical_output"), 20_266_867_967),
                (
                    "validated numerator bytes",
                    ("bytes", "validated_numerator_output"),
                    402_644_223,
                ),
                ("validated auxiliary bytes", ("bytes", "validated_auxiliary_output"), 913),
                ("validated split", ("bytes", "validated_canonical_output"), 402_645_137),
                ("warmups", ("warmups",), 2),
                ("iterations", ("iterations",), 6),
                ("causal flag", ("identity", "timed_sample_causally_validated"), False),
                ("nonpositive sample", ("samples_ms", "legacy", 0), 0.0),
                ("nonfinite sample", ("samples_ms", "hybrid", 0), float("nan")),
                ("nearest-rank p50", ("host_wall_ms", "legacy", "p50"), 11.0),
                ("nearest-rank p95", ("host_wall_ms", "hybrid", "p95"), 8.0),
                ("speedup", ("speedup", "p95"), 2.0),
                ("free-memory order", ("device_memory", "free_after_arena"), 81_000_000_000),
                (
                    "pool-memory order",
                    ("device_memory", "isolated_pool_reserved_after_arena"),
                    41_000_000_000,
                ),
            )
            for label, path, value in mutations:
                mutated = copy.deepcopy(valid)
                target = mutated
                for key in path[:-1]:
                    target = target[key]  # type: ignore[index]
                target[path[-1]] = value  # type: ignore[index]
                with self.subTest(label=label):
                    rejected = validate(mutated)
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout)

    def test_replacement_sn2_ecc_policy_is_explicit_and_sealed(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        source = common.read_text(encoding="utf-8")
        fields = [
            "NVIDIA H100 80GB HBM3",
            "GPU-00000000-0000-0000-0000-000000000001",
            "00000000:01:00.0",
            "81559",
            "550.54.15",
            "9.0",
            "Enabled",
            "Disabled",
            "{ecc}",
            "Default",
            "700.00",
            "1980",
            "1593",
        ]
        records = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$TEST_GPU_ROW\"\n",
                encoding="utf-8",
            )
            nvidia_smi.chmod(0o755)
            for index, ecc in enumerate(("Enabled", "Disabled", "N/A", "", "Unknown")):
                out = root / f"hardware-{index}.json"
                env = {
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "REPLACEMENT_SN2_MODE": "diagnostic",
                    "CAIRO": "/workspace/stwo-cairo/stwo_cairo_prover",
                    "STWO": "/workspace/stwo",
                    "RUN": str(root),
                    "COMMON": str(common),
                    "OUT": str(out),
                    "TEST_GPU_ROW": ",".join(fields).format(ecc=ecc),
                }
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$COMMON"; checkpoint_capture_hardware_identity "$OUT"',
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if ecc in {"Enabled", "Disabled"}:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    records[ecc] = json.loads(out.read_text(encoding="utf-8"))
                    self.assertEqual(records[ecc]["ecc_mode"], ecc)
                else:
                    self.assertNotEqual(result.returncode, 0, ecc)
                    self.assertIn("unstable GPU policy", result.stderr)

        self.assertNotEqual(records["Enabled"], records["Disabled"])
        self.assertIn(
            'if hardware != seal.get("hardware"):\n'
            '    raise SystemExit("timing GPU identity/policy differs from the diagnostic")',
            source,
        )

    def test_replacement_sn2_reuse_gate_rejects_each_mutation(self) -> None:
        source = (
            ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        ).read_text(encoding="utf-8")
        validator = source.index("checkpoint_validate_sn2()")
        valid = {
            "gpu_host_plan_cache_materialization": "reused",
            "gpu_host_plan_cache_hits": 1,
            "gpu_host_plan_cache_misses": 1,
            "gpu_host_plan_cache_compilations": 1,
            "gpu_host_plan_cache_evictions": 0,
            "gpu_host_plan_cache_collisions": 0,
            "gpu_host_preparation_total_ns": 80_000_000,
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
        require_resident_reuse(valid, 2, max_host_preparation_ns=120_000_000)
        require_resident_reuse(
            {
                **valid,
                "gpu_host_plan_cache_hits": 5,
                "gpu_shape_executable_cache_hits": 5,
            },
            6,
            max_host_preparation_ns=120_000_000,
        )
        self.assertIn(
            "from validate_replacement_v1_reuse import require_resident_reuse",
            source[validator:],
        )
        self.assertIn(
            "require_resident_reuse(r, reps, max_host_preparation_ns=120_000_000)",
            source[validator:],
        )

        for field, expected in valid.items():
            mutations = (
                ("compiled", None)
                if field in {
                    "gpu_host_plan_cache_materialization",
                    "gpu_shape_executable_materialization",
                }
                else ("materialized", None)
                if field == "gpu_workspace_materialization"
                else (120_000_001, True, None)
                if field == "gpu_host_preparation_total_ns"
                else (expected + 1, bool(expected), None)
            )
            for mutation in mutations:
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaises(SystemExit):
                        require_resident_reuse(
                            {**valid, field: mutation},
                            2,
                            max_host_preparation_ns=120_000_000,
                        )

        with self.assertRaises(SystemExit):
            require_resident_reuse(
                {**valid, "gpu_host_preparation_total_ns": 120_000_001},
                2,
                max_host_preparation_ns=120_000_000,
            )
        with self.assertRaises(ValueError):
            require_resident_reuse(valid, 2, max_host_preparation_ns=0)

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
