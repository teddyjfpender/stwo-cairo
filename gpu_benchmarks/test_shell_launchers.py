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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_stage4_native_receipt(stwo_head: str) -> dict[str, object]:
    digest = "56" * 32
    return {
        "schema": "stwo.replacement-stage4-native.v1",
        "passed": True,
        "failure": None,
        "git_commit": stwo_head,
        "executable_blake3": digest,
        "source_blake3": digest,
        "cuda_device": "NVIDIA H100 80GB HBM3, GPU-test, 9.0, 550.54.15",
        "nvcc_version": "Cuda compilation tools, release 12.4, V12.4.131",
        "requested_cuda_arch": "sm_90",
        "total_memory_bytes": 85_000_000_000,
        "free_memory_before_bytes": 80_000_000_000,
        "free_memory_after_bytes": 79_000_000_000,
        "performance_requested": False,
        "performance_failure": None,
        "performance": [],
        "fixtures": [
            {
                "name": "staged-packed-quotient-mixed-topology",
                "production_apis": [
                    "quotient_numerator_staged_single_write_plan_with_overflow_capacities",
                    "PreparedQuotientNumeratorGraph::prepare_staged_packed_single_write",
                ],
                "cases": 2,
                "arena_bytes": 4096,
                "checks": {
                    "eager_reference": True,
                    "legacy_candidate_byte_identity": True,
                    "captured_graph_mutation": True,
                    "source_preservation": True,
                    "guard_preservation": True,
                },
                "hashes": {
                    "eager_outputs": digest,
                    "mutated_graph_outputs": digest,
                },
            },
            {
                "name": "mode-a-domain-cooperative-commit",
                "production_apis": [
                    "CommitProgram::bind",
                    "DomainCooperativeProgram::compile_mode_a",
                    "DomainCooperativeProgram::bind",
                ],
                "cases": 9,
                "arena_bytes": 8192,
                "checks": {
                    "raw_prefix_boundary_identity": True,
                    "eager_reference": True,
                    "legacy_candidate_byte_identity": True,
                    "captured_graph_mutation": True,
                    "source_preservation": True,
                    "guard_preservation": True,
                },
                "hashes": {
                    "raw_prefix_states": digest,
                    "raw_prefix_hashes": digest,
                    "eager_root_and_retained": digest,
                    "mutated_graph_root_and_retained": digest,
                },
            },
        ],
    }


class ShellLauncherTests(unittest.TestCase):
    def test_fleet_dry_run_seals_replacement_backend(self) -> None:
        fleet = ROOT / "fleet" / "fleet.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "fleet.conf"
            report = root / "fleet_report.json"
            config.write_text(
                "dry-pod | RTX 4090 | 0.69 | | | | 1\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", str(fleet)],
                cwd=ROOT.parent,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "FLEET_CONF": str(config),
                    "RESULTS_DIR": str(root / "results"),
                    "REPORT": str(report),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--resident-backend replacement-v1", result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["aggregate"]["n_pods_ok"], 1)
            worker = payload["pods"][0]
            self.assertEqual(
                worker["gpu_resident_backend_requested"], "replacement-v1"
            )
            self.assertEqual(worker["gpu_resident_backend"], "replacement-v1")

    def test_replacement_sn2_stage4_native_seals_diagnostic(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        source = common.read_text(encoding="utf-8")
        self.assertIn("checkpoint_validate_stage4_native_receipt", source)
        self.assertIn("replacement_stage4_native_bytes_match", source)
        self.assertNotIn("sn3_hybrid_graph_host_wall_benchmark", source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stwo_head = "ab" * 20
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
            counter_raw = root / "fixture.counter_acceptance.csv"
            counter_log = root / "fixture.counter_acceptance.txt"
            counter_raw.write_text(
                'ID,Process ID,Kernel Name,Metric Name,Metric Value\n'
                '1,7,"checkpoint_counter_kernel",sm__cycles_elapsed.avg,123.0\n',
                encoding="utf-8",
            )
            counter_log.write_text("CHECKPOINT_COUNTER_KERNEL_RESULT=1\n", encoding="utf-8")
            artifacts = {
                "source_input_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.source-input-identity.v1",
                    "source": {
                        "stwo": {"head": stwo_head, "worktree_sha256": "12" * 32},
                        "stwo_cairo": {"head": "cd" * 20, "worktree_sha256": "34" * 32},
                    },
                    "inputs": {"SN_PIE_2.zip": "input-a"},
                },
                "hardware_identity.json": {
                    "schema": "stwo.replacement-v1-sn2.hardware-identity.v2",
                    "name": "test H100",
                    "uuid": "GPU-test",
                },
                "counter_acceptance.json": {
                    "schema": "stwo.replacement-v1.counter-acceptance.v1",
                    "pass": True,
                    "kernel": "checkpoint_counter_kernel",
                    "metric": "sm__cycles_elapsed.avg",
                    "metric_value": 123.0,
                    "gpu_uuid": "GPU-test",
                    "raw_csv_sha256": file_sha256(counter_raw),
                    "acceptance_log_sha256": file_sha256(counter_log),
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
                "stage4_native.json": valid_stage4_native_receipt(stwo_head),
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
                        "counter_profile_admissible": True,
                    },
                    "gpu_proof_blake3": "ab" * 32,
                    "gpu_protocol_key": "protocol-v1",
                    "gpu_shape_executable_topology_digest": "topology-v1",
                    "gpu_prepared_numerator_schedule": "staged-packed-single-write",
                    "gpu_prepared_numerator_packed_output_rows": 20_971_472,
                    "gpu_composition_part_count": 153,
                    "gpu_composition_wave_count": 18,
                    "gpu_graph_submit_gap_ns_max_samples": [1_000_000, 2_000_000],
                    "gpu_graph_submit_gap_ns_total_samples": [13_000_000, 26_000_000],
                    "gpu_graph_submit_gap_ns_average_samples": [1_000_000.0, 2_000_000.0],
                    "gpu_graph_submit_launches_samples": [14, 14],
                    "gpu_graph_submit_gap_strict_gate_passed": True,
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
                "STWO_PARITY_REF_STWO_HEAD": stwo_head,
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
                    'checkpoint_validate_stage4_native_receipt '
                    '"$RUN/fixture.stage4_native.json"; '
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
                sealed["receipts_sha256"]["stage4"],
                file_sha256(root / "fixture.stage4_native.json"),
            )
            self.assertEqual(
                sealed["receipts_sha256"]["counter"],
                file_sha256(root / "fixture.counter_acceptance.json"),
            )
            self.assertEqual(
                sealed["shape_receipt"],
                {
                    "protocol_key": "protocol-v1",
                    "topology_digest": "topology-v1",
                    "numerator_schedule": "staged-packed-single-write",
                    "numerator_packed_output_rows": 20_971_472,
                    "composition_part_count": 153,
                    "composition_wave_count": 18,
                },
            )

            counter_path = root / "fixture.counter_acceptance.json"
            valid_counter = json.loads(counter_path.read_text(encoding="utf-8"))
            counter_path.write_text(
                json.dumps({**valid_counter, "metric_value": -1}) + "\n",
                encoding="utf-8",
            )
            forged = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    'CHECKPOINT_GPU_BENCH="$TEST_GPU_BENCH"; '
                    'CHECKPOINT_AOT_CHECK="$TEST_AOT_CHECK"; '
                    'CHECKPOINT_AOT_MANIFEST="$TEST_AOT_MANIFEST"; '
                    'CHECKPOINT_SEAL="$TEST_SEAL"; checkpoint_seal_diagnostic',
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(forged.returncode, 0)
            counter_path.write_text(json.dumps(valid_counter) + "\n", encoding="utf-8")

            counter_log.write_text(
                "CHECKPOINT_COUNTER_KERNEL_RESULT=1\nERR_NVGPUCTRPERM\n",
                encoding="utf-8",
            )
            contaminated_counter = {
                **valid_counter,
                "acceptance_log_sha256": file_sha256(counter_log),
            }
            counter_path.write_text(
                json.dumps(contaminated_counter) + "\n", encoding="utf-8"
            )
            contaminated = subprocess.run(
                [
                    "bash", "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    'CHECKPOINT_GPU_BENCH="$TEST_GPU_BENCH"; '
                    'CHECKPOINT_AOT_CHECK="$TEST_AOT_CHECK"; '
                    'CHECKPOINT_AOT_MANIFEST="$TEST_AOT_MANIFEST"; '
                    'CHECKPOINT_SEAL="$TEST_SEAL"; checkpoint_seal_diagnostic',
                ],
                check=False, capture_output=True, text=True, env=env,
            )
            self.assertNotEqual(contaminated.returncode, 0)

            counter_raw.write_text("ERR_NVGPUCTRPERM\n", encoding="utf-8")
            counter_log.write_text(
                "CHECKPOINT_COUNTER_KERNEL_RESULT=1\nERR_NVGPUCTRPERM\n",
                encoding="utf-8",
            )
            counter_path.write_text(
                json.dumps(
                    {
                        "schema": "stwo.replacement-v1.counter-availability.v1",
                        "status": "UNAVAILABLE",
                        "pass": False,
                        "error_code": "ERR_NVGPUCTRPERM",
                        "command_return_code": 1,
                        "gpu_uuid": "GPU-test",
                        "raw_csv_sha256": file_sha256(counter_raw),
                        "acceptance_log_sha256": file_sha256(counter_log),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            record_path = root / "fixture.record.json"
            diagnostic = json.loads(record_path.read_text(encoding="utf-8"))
            diagnostic["counter_profile_admissible"] = False
            diagnostic["checkpoint_validation"]["counter_profile_admissible"] = False
            record_path.write_text(json.dumps(diagnostic) + "\n", encoding="utf-8")
            timing_only = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    'CHECKPOINT_GPU_BENCH="$TEST_GPU_BENCH"; '
                    'CHECKPOINT_AOT_CHECK="$TEST_AOT_CHECK"; '
                    'CHECKPOINT_AOT_MANIFEST="$TEST_AOT_MANIFEST"; '
                    'CHECKPOINT_SEAL="$TEST_SEAL"; checkpoint_seal_diagnostic',
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**env, "REPLACEMENT_SN2_COUNTER_POLICY": "timing-only"},
            )
            self.assertEqual(timing_only.returncode, 0, timing_only.stderr)
            timing_only_seal = json.loads(seal.read_text(encoding="utf-8"))
            self.assertEqual(
                timing_only_seal["schema"],
                "stwo.replacement-v1-sn2.timing-only-seal.v1",
            )
            self.assertEqual(timing_only_seal["counter_policy"], "timing-only")
            self.assertFalse(timing_only_seal["counter_profile_admissible"])
            self.assertEqual(timing_only_seal["counter_status"], "UNAVAILABLE")

    def test_replacement_sn2_aot_identity_derives_exact_key_count(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "aot_manifest.json"
            entries = [
                {
                    "kind": "constraint",
                    "label": f"constraint_{index}",
                    "cache_key": f"{index:016x}",
                }
                for index in range(219)
            ]
            entries.extend(
                {
                    "kind": "constraint",
                    "label": f"wave_log_{index}",
                    "cache_key": f"{219 + index:016x}",
                }
                for index in range(119)
            )
            entries.extend(
                {
                    "kind": "witness",
                    "label": f"witness_{index}",
                    "cache_key": f"{338 + index:016x}",
                }
                for index in range(35)
            )
            manifest.write_text(
                json.dumps(entries) + "\n",
                encoding="utf-8",
            )
            checker = root / "aot_index_check"
            checker.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '{\"pass\":true,\"sm\":90,"
                "\"loaded_manifest_hash\":\"1234567890abcdef\","
                "\"required_unique_key_count\":'\"${TEST_COUNT:-373}\"',"
                "\"embedded_entry_count\":'\"${TEST_EMBEDDED_COUNT:-373}\"',"
                "\"embedded_arch_entry_count\":'\"${TEST_EMBEDDED_ARCH_COUNT:-373}\"',"
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
            self.assertEqual(identity["required_unique_key_count"], 373)
            self.assertEqual(identity["manifest_entry_count"], 373)
            self.assertEqual(identity["manifest_witness_count"], 35)
            self.assertEqual(identity["manifest_ordinary_constraint_count"], 219)
            self.assertEqual(identity["manifest_composition_wave_count"], 119)

            rejected = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True,
                env={**env, "TEST_COUNT": "374"},
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("identity/coverage failed", rejected.stderr)

            for field in ("TEST_EMBEDDED_COUNT", "TEST_EMBEDDED_ARCH_COUNT"):
                rejected = subprocess.run(
                    ["bash", "-c", command],
                    capture_output=True,
                    text=True,
                    env={**env, field: "374"},
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("identity/coverage failed", rejected.stderr)

            mutated = copy.deepcopy(entries)
            mutated[-1]["kind"] = "unknown"
            manifest.write_text(json.dumps(mutated) + "\n", encoding="utf-8")
            rejected = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                env={**env, "TEST_MANIFEST_SHA": file_sha256(manifest)},
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("replacement AOT pack shape drifted", rejected.stderr)

    def test_replacement_sn2_stage4_validator_rejects_mutations(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "stage4.json"
            stwo_head = "ab" * 20
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                "STWO": str(root / "stwo"),
                "RUN": str(root),
                "COMMON": str(common),
                "RECORD": str(record_path),
                "STWO_PARITY_REF_STWO_HEAD": stwo_head,
            }

            def validate(record: dict[str, object]) -> subprocess.CompletedProcess[str]:
                record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$COMMON"; checkpoint_validate_stage4_native_receipt '
                        '"$RECORD"',
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )

            valid = valid_stage4_native_receipt(stwo_head)
            accepted = validate(valid)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn('"replacement_stage4_native": "PASS"', accepted.stdout)

            mutations = (
                ("schema", ("schema",), "stwo.replacement-stage4-native.v0"),
                ("pass", ("passed",), False),
                ("commit", ("git_commit",), "cd" * 20),
                ("architecture", ("requested_cuda_arch",), "sm_89"),
                ("performance mode", ("performance_requested",), True),
                ("memory", ("total_memory_bytes",), 0),
                ("quotient cases", ("fixtures", 0, "cases"), 1),
                ("mode-a API", ("fixtures", 1, "production_apis", 0), "wrong"),
                ("failed check", ("fixtures", 0, "checks", "legacy_candidate_byte_identity"), False),
                ("invalid hash", ("fixtures", 1, "hashes", "raw_prefix_states"), "invalid"),
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
            for fixtures in (
                [*copy.deepcopy(valid["fixtures"]), copy.deepcopy(valid["fixtures"][0])],  # type: ignore[index]
                [*copy.deepcopy(valid["fixtures"]), "hidden-extra"],  # type: ignore[index]
            ):
                mutated = copy.deepcopy(valid)
                mutated["fixtures"] = fixtures
                self.assertNotEqual(validate(mutated).returncode, 0)

    def test_replacement_sn2_sealed_phase_order_is_fail_closed_then_profiles(self) -> None:
        recipe = (
            ROOT / "loop" / "recipes" / "replacement_v1_sn2_sealed.phases"
        ).read_text(encoding="utf-8")
        phases = [
            line.split()[1]
            for line in recipe.splitlines()
            if line.startswith("phase ")
        ]
        self.assertIn("REPLACEMENT_SN2_COUNTER_POLICY=required", recipe)
        self.assertEqual(
            phases,
            [
                "ambient_override_gate", "source_input_identity", "hardware_identity",
                "counter_permission_acceptance", "nsys_tool_identity", "build",
                "adapted_input_identity",
                "fp256_carry_oracles", "replacement_stage4_native", "aot_identity",
                "sn2_diagnostic", "sn2_diagnostic_validate", "seal_passing_diagnostic",
                "timing_ambient_override_gate", "timing_sealed_diagnostic_identity",
                "timing_sn2", "timing_sn2_validate", "nsys_profile", "ncu_profile",
                "promotion_verdict",
            ],
        )

        common = (
            ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('CHECKPOINT_AOT_TOTAL=373', common)
        self.assertIn('CHECKPOINT_AOT_WITNESS=35', common)
        self.assertIn('CHECKPOINT_AOT_ORDINARY_CONSTRAINT=219', common)
        self.assertIn('CHECKPOINT_AOT_COMPOSITION_WAVE=119', common)
        self.assertIn(
            'CHECKPOINT_AOT_MANIFEST_SHA256=1ff3089cf9c6c9284ddfbdcfd8258d3d329a115d1285ee8c066f563175005fa4',
            common,
        )
        self.assertIn('STWO_STAGE4_GIT_COMMIT="$STWO_PARITY_REF_STWO_HEAD"', common)
        self.assertIn('r.get("gpu_prepared_numerator_eligible_groups") is None', common)
        self.assertIn('r.get("gpu_prepared_numerator_legacy_groups") == 0', common)
        self.assertIn('r.get("gpu_policy_composition_launch_mode") == "wave"', common)
        self.assertIn('args+=(--capture-slow-graph-submit)', common)
        self.assertIn('benchmark_graph_submit_capture_mode', common)
        self.assertIn('return 0\n}', common[common.index("checkpoint_nsys_profile()"):])
        gpu_bench = (
            ROOT.parent
            / "stwo_cairo_prover"
            / "crates"
            / "gpu-prover"
            / "src"
            / "bin"
            / "gpu_bench.rs"
        ).read_text(encoding="utf-8")
        self.assertIn('flag("--capture-slow-graph-submit")', gpu_bench)
        self.assertIn('gpu_graph_submit_gap_ns_max_samples', gpu_bench)
        self.assertIn('gpu_graph_submit_gap_ns_total_samples', gpu_bench)
        self.assertIn('claimed_graph_submit_gap_ns(&samples)', gpu_bench)
        self.assertIn('graph_submit_gap_average_ns(*sample)', gpu_bench)
        self.assertIn('gap_count = launches - 1', common)
        self.assertGreaterEqual(
            gpu_bench.count(
                "--capture-slow-graph-submit is supported only by the standard serial benchmark"
            ),
            3,
        )

    def test_replacement_sn2_timing_only_lane_seals_exact_counter_denial(self) -> None:
        recipe = (
            ROOT / "loop" / "recipes" / "replacement_v1_sn2_timing_only.phases"
        ).read_text(encoding="utf-8")
        phases = [
            line.split()[1] for line in recipe.splitlines() if line.startswith("phase ")
        ]
        self.assertIn("REPLACEMENT_SN2_COUNTER_POLICY='timing-only'", recipe)
        self.assertEqual(
            phases,
            [
                "ambient_override_gate", "source_input_identity", "hardware_identity",
                "counter_permission_receipt", "nsys_tool_identity", "build",
                "adapted_input_identity",
                "fp256_carry_oracles", "replacement_stage4_native", "aot_identity",
                "sn2_diagnostic", "sn2_diagnostic_validate", "seal_passing_diagnostic",
                "timing_sealed_diagnostic_identity", "timing_sn2",
                "timing_sn2_validate", "nsys_profile", "timing_only_verdict",
            ],
        )

        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.hardware_identity.json").write_text(
                json.dumps(
                    {
                        "schema": "stwo.replacement-v1-sn2.hardware-identity.v2",
                        "uuid": "GPU-test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "REPLACEMENT_SN2_COUNTER_POLICY": "timing-only",
                "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                "STWO": str(root / "stwo"),
                "RUN": str(root),
                "COMMON": str(common),
            }
            command = r'''
source "$COMMON"
CHECKPOINT_PREFIX=fixture
checkpoint_counter_acceptance() {
  printf '%s\n' CHECKPOINT_COUNTER_KERNEL_RESULT=1 ERR_NVGPUCTRPERM >"$(checkpoint_artifact counter_acceptance.txt)"
  printf '%s\n' ERR_NVGPUCTRPERM >"$(checkpoint_artifact counter_acceptance.csv)"
  return 13
}
checkpoint_counter_timing_only
'''
            accepted = subprocess.run(
                ["bash", "-c", command], check=False, capture_output=True, text=True, env=env
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            receipt = json.loads(
                (root / "fixture.counter_acceptance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "UNAVAILABLE")
            self.assertFalse(receipt["pass"])
            self.assertEqual(receipt["command_return_code"], 13)
            self.assertEqual(receipt["gpu_uuid"], "GPU-test")
            self.assertEqual(
                receipt["raw_csv_sha256"],
                file_sha256(root / "fixture.counter_acceptance.csv"),
            )
            self.assertEqual(
                receipt["acceptance_log_sha256"],
                file_sha256(root / "fixture.counter_acceptance.txt"),
            )

            rejected = subprocess.run(
                ["bash", "-c", command.replace("ERR_NVGPUCTRPERM", "generic failure")],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(rejected.returncode, 0)
            failed = json.loads(
                (root / "fixture.counter_acceptance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed["status"], "FAIL")

            (root / "fixture.record.json").write_text(
                json.dumps(
                    {
                        "checkpoint_validation": {"verdict": "PASS"},
                        "performance_measurement_available": True,
                        "gpu_graph_submit_gap_strict_gate_passed": True,
                        "gpu_host_preparation_total_ns": 100_000_000,
                        "prove_s_warm_median": 0.5,
                        "prove_s_warm_p95": 0.6,
                        "useful_mhz_median": 15.0,
                        "useful_mhz_at_warm_p95": 13.0,
                        "gpu_max_graph_submit_gap_ms": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "fixture.nsys_profile.json").write_text(
                json.dumps({"status": "PASS"}) + "\n", encoding="utf-8"
            )
            (root / "replacement_v1_sn2_timing_only_checkpoint.seal.json").write_text(
                json.dumps(
                    {
                        "schema": "stwo.replacement-v1-sn2.timing-only-seal.v1",
                        "counter_policy": "timing-only",
                        "counter_profile_admissible": False,
                        "counter_status": "UNAVAILABLE",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verdict_result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    "checkpoint_assess_sn2_timing_only",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**env, "REPLACEMENT_SN2_MODE": "timing"},
            )
            self.assertEqual(verdict_result.returncode, 0, verdict_result.stderr)
            verdict = json.loads(
                (root / "fixture.timing_only_verdict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["verdict"], "INCOMPLETE")
            self.assertFalse(verdict["formal_promotion_eligible"])
            self.assertEqual(verdict["thresholds"]["useful_mhz_median_min"], 5.0)
            self.assertIn("nsys_profile_passed", verdict["failed_completion_checks"])
            self.assertEqual(
                verdict["profile_status"]["ncu"], "OMITTED_COUNTER_UNAVAILABLE"
            )

    def test_replacement_sn2_iteration_is_fast_and_non_promotable(self) -> None:
        recipe_path = ROOT / "loop" / "recipes" / "replacement_v1_sn2_iteration.phases"
        recipe = recipe_path.read_text(encoding="utf-8")
        phases = [
            line.split()[1] for line in recipe.splitlines() if line.startswith("phase ")
        ]
        self.assertNotIn("# pod_run: require_clean_sources", recipe)
        self.assertIn("checkpoint_source_input_identity iteration", recipe)
        self.assertEqual(
            phases,
            [
                "ambient_override_gate", "source_input_identity", "hardware_identity",
                "build", "adapted_input_identity", "aot_identity", "sn2_iteration",
                "sn2_iteration_validate",
            ],
        )
        self.assertNotIn("carry", " ".join(phases))
        self.assertNotIn("stage4", " ".join(phases))
        self.assertNotIn("profile", " ".join(phases))

        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        source = common.read_text(encoding="utf-8")
        self.assertIn('mode in ("timing", "iteration")', source)
        self.assertIn('r["formal_promotion_eligible"] = False', source)
        self.assertIn('"formal_promotion_eligible"] = False', source)
        self.assertIn('source_receipt.get("source_policy") == "iteration"', source)
        self.assertIn('"proof_dump_sha256": hashlib.sha256', source)
        self.assertIn('"gpu_bench_sha256": os.environ["GPU_BENCH_SHA"]', source)

        env = {
            **os.environ,
            "REPLACEMENT_SN2_MODE": "iteration",
            "REPLACEMENT_SN2_COUNTER_POLICY": "timing-only",
            "CAIRO": "/workspace/stwo-cairo/stwo_cairo_prover",
            "STWO": "/workspace/stwo",
            "RUN": "/tmp",
            "COMMON": str(common),
            "STWO_PARITY_REF_STWO_HEAD": "ab" * 20,
            "STWO_PARITY_REF_STWO_CAIRO_HEAD": "cd" * 20,
            "STWO_PARITY_REF_STWO_WORKTREE_HASH": "12" * 32,
            "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": "34" * 32,
        }
        result = subprocess.run(
            [
                "bash", "-c",
                'source "$COMMON"; checkpoint_require_source_identity; '
                "if checkpoint_require_clean_source_identity 2>/dev/null; then exit 9; fi",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        forbidden_env = {
            name: value
            for name, value in env.items()
            if name not in {"STWO_CUDA_NVCC", "STWO_CUDA_NVCC_FLAGS",
                            "STWO_CUDA_HOST_COMPILER",
                            "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                            "NVCC_CCBIN", "CUDAHOSTCXX"}
        }
        forbidden_env["NVCC_APPEND_FLAGS"] = "-lineinfo"
        rejected = subprocess.run(
            ["bash", "-c", 'source "$COMMON"; checkpoint_reject_ambient_overrides'],
            check=False,
            capture_output=True,
            text=True,
            env=forbidden_env,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("checkpoint rejects ambient override NVCC_APPEND_FLAGS", rejected.stderr)

        wrapper = (ROOT / "loop" / "quick_sn2.sh").read_text(encoding="utf-8")
        self.assertIn("replacement_v1_sn2_iteration.phases", wrapper)
        self.assertIn('exec "$loop_dir/pod_run.sh"', wrapper)
        self.assertIn('POD_RUN_POLL_INTERVAL:-2', wrapper)
        self.assertIn('rustc --edition=2021 -D warnings --test', wrapper)
        self.assertIn('crates/backend-cuda-kernels/build.rs', wrapper)
        pod_run = (ROOT / "loop" / "pod_run.sh").read_text(encoding="utf-8")
        self.assertIn('sleep "$POD_RUN_POLL_INTERVAL"', pod_run)
        self.assertNotIn("    sleep 30\n", pod_run)

    def test_replacement_sn2_profiles_fail_before_build_without_nsys(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_nsys = root / "nsys-fallback"
            fake_nsys.write_text("#!/bin/sh\necho 'Nsight Systems 2024.6.2'\n", encoding="utf-8")
            fake_nsys.chmod(0o755)
            env = {
                **os.environ,
                "PATH": "/usr/bin:/bin",
                "REPLACEMENT_SN2_MODE": "diagnostic",
                "CAIRO": "/workspace/stwo-cairo/stwo_cairo_prover",
                "STWO": "/workspace/stwo",
                "RUN": str(root),
                "COMMON": str(common),
                "FAKE_NSYS": str(fake_nsys),
            }
            command = (
                'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                'CHECKPOINT_NSYS_FALLBACK="$FAKE_NSYS"; checkpoint_nsys_tool_identity'
            )
            accepted = subprocess.run(
                ["bash", "-c", command], check=False, capture_output=True, text=True, env=env
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            identity = json.loads(
                (root / "fixture.nsys_tool_identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(identity["path"], str(fake_nsys))
            self.assertIn("2024.6.2", identity["version"])

            fake_nsys.unlink()
            rejected = subprocess.run(
                ["bash", "-c", command], check=False, capture_output=True, text=True, env=env
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("required Nsight Systems executable is absent", rejected.stderr)

    def test_replacement_sn2_iteration_forces_publication_admissibility_false(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "iteration.json"
            valid = {
                "performance_claim_admissible": True,
                "gpu_graph_submit_gap_strict_gate_passed": True,
                "iteration_only": True,
                "formal_promotion_eligible": False,
                "checkpoint_validation": {
                    "verdict": "PASS",
                    "mode": "iteration",
                    "counter_profile_admissible": False,
                    "formal_promotion_eligible": False,
                },
            }
            record_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "iteration",
                "REPLACEMENT_SN2_COUNTER_POLICY": "timing-only",
                "CAIRO": "/workspace/stwo-cairo/stwo_cairo_prover",
                "STWO": "/workspace/stwo",
                "RUN": str(root),
                "COMMON": str(common),
                "RECORD": str(record_path),
            }
            command = 'source "$COMMON"; checkpoint_mark_iteration_non_promotable "$RECORD"'
            accepted = subprocess.run(
                ["bash", "-c", command], check=False, capture_output=True, text=True, env=env
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            marked = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertFalse(marked["performance_claim_admissible"])
            self.assertTrue(marked["iteration_timing_gate_passed"])
            self.assertFalse(marked["formal_promotion_eligible"])

            invalid = copy.deepcopy(valid)
            invalid["formal_promotion_eligible"] = True
            record_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            rejected = subprocess.run(
                ["bash", "-c", command], check=False, capture_output=True, text=True, env=env
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(json.loads(record_path.read_text())["formal_promotion_eligible"], True)

    def test_replacement_sn2_promotion_thresholds_fail_soft(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {
                "record.json": {
                    "checkpoint_validation": {"verdict": "PASS"},
                    "counter_profile_admissible": True,
                    "performance_claim_admissible": True,
                    "gpu_graph_submit_gap_strict_gate_passed": True,
                    "gpu_host_preparation_total_ns": 120_000_001,
                    "useful_mhz_median": 4.999,
                    "useful_mhz_at_warm_p95": 4.5,
                    "gpu_max_graph_submit_gap_ms": 1.0,
                },
                "nsys_profile.json": {"status": "FAIL"},
                "ncu_profile.json": {"status": "PASS"},
            }
            for name, record in records.items():
                (root / f"fixture.{name}").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$COMMON"; CHECKPOINT_PREFIX=fixture; '
                    "checkpoint_assess_sn2_promotion",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "REPLACEMENT_SN2_MODE": "timing",
                    "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                    "STWO": str(root / "stwo"),
                    "RUN": str(root),
                    "COMMON": str(common),
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            verdict = json.loads(
                (root / "fixture.promotion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["verdict"], "FAIL")
            self.assertTrue(verdict["soft_failure"])
            self.assertEqual(verdict["thresholds"]["useful_mhz_median_min"], 5.0)
            self.assertEqual(
                set(verdict["failed_checks"]),
                {
                    "host_preparation_within_budget",
                    "useful_mhz_at_or_above_floor",
                    "nsys_profile_passed",
                    "ncu_profile_passed",
                },
            )

    def test_replacement_sn2_ncu_receipt_requires_exact_launch_topology(self) -> None:
        common = ROOT / "loop" / "recipes" / "replacement_v1_sn2_common.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / "proof.bin"
            proof.write_bytes(b"proof")
            proof_sha = file_sha256(proof)
            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            stdout = root / "stdout.txt"
            stdout.write_text(
                json.dumps(
                    {
                        "program": "SN_PIE_2.zip",
                        "backend": "cuda",
                        "engine": "gpu-native",
                        "gpu_resident_backend": "replacement-v1",
                        "gpu_pcs_runtime_mode": "ArenaGraph",
                        "gpu_aot_provenance_gate_passed": True,
                        "gpu_prepared_numerator_schedule": "staged-packed-single-write",
                        "gpu_composition_part_count": 153,
                        "gpu_composition_wave_count": 18,
                        "gpu_proof_blake3": "ab" * 32,
                        "verified_reps": 2,
                        "proof_byte_equal": True,
                        "simd_reference_byte_equal": True,
                        "proof_mutation_rejected": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stderr = root / "stderr.txt"
            stderr.write_text("", encoding="utf-8")
            gpu_bench = root / "gpu_bench"
            gpu_bench.write_bytes(b"gpu-bench")
            seal = root / "seal.json"
            seal.write_text(
                json.dumps(
                    {
                        "schema": "stwo.replacement-v1-sn2.checkpoint-seal.v3",
                        "diagnostic_pass": True,
                        "gpu_bench_sha256": file_sha256(gpu_bench),
                        "proof_dump_sha256": proof_sha,
                        "proof_blake3": "ab" * 32,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            table = root / "profile.csv"
            out = root / "receipt.json"
            env = {
                **os.environ,
                "REPLACEMENT_SN2_MODE": "timing",
                "CAIRO": str(root / "stwo-cairo" / "stwo_cairo_prover"),
                "STWO": str(root / "stwo"),
                "RUN": str(root),
                "COMMON": str(common),
                "TEST_GPU_BENCH": str(gpu_bench),
                "PROOF": str(proof),
                "REPORT": str(report),
                "TABLE": str(table),
                "STDOUT": str(stdout),
                "STDERR": str(stderr),
                "SEAL": str(seal),
                "OUT": str(out),
            }
            command = (
                'source "$COMMON"; CHECKPOINT_GPU_BENCH="$TEST_GPU_BENCH"; '
                'CHECKPOINT_SEAL="$SEAL"; checkpoint_write_profile_receipt ncu 0 '
                '"$PROOF" "$REPORT" "$TABLE" "$STDOUT" "$STDERR" "$OUT"'
            )

            def validate(kernels: list[str]) -> dict[str, object]:
                rows = ["ID,Process ID,Kernel Name,Metric Name,Metric Value"]
                rows.extend(
                    f'{index},7,"{kernel}",sm__cycles_elapsed.avg,100'
                    for index, kernel in enumerate(kernels, 1)
                )
                table.write_text("\n".join(rows) + "\n", encoding="utf-8")
                result = subprocess.run(
                    ["bash", "-c", command],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(out.read_text(encoding="utf-8"))

            waves = [f"stwo_composition_wave_{index:016x}" for index in range(18)]
            numerator = "stwo_quotient_numerator_packed_single_write_kernel"
            accepted = validate([*waves, numerator])
            self.assertEqual(accepted["status"], "PASS")
            self.assertEqual(
                accepted["ncu_launch_topology"],
                {
                    "selected_launch_count": 19,
                    "composition_wave_launch_count": 18,
                    "distinct_composition_wave_kernel_count": 18,
                    "packed_numerator_launch_count": 1,
                },
            )
            gpu_bench.write_bytes(b"substituted-gpu-bench")
            substituted = validate([*waves, numerator])
            self.assertEqual(substituted["status"], "FAIL")
            self.assertIn(
                "profile binary differs from sealed diagnostic",
                substituted["soft_failure_reasons"],
            )
            gpu_bench.write_bytes(b"gpu-bench")
            for kernels in (
                [*waves[:-1], numerator],
                [*waves[:-1], waves[0], numerator],
                [*waves, numerator, numerator],
            ):
                with self.subTest(kernels=len(kernels)):
                    self.assertEqual(validate(kernels)["status"], "FAIL")

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
            "gpu_prepared_runtime_materialization": "reused",
            "gpu_prepared_runtime_capture_ready_at_entry": True,
            "gpu_prepared_runtime_capture_ready_at_exit": True,
            "gpu_statement_refresh_present": True,
            "gpu_shape_executable_cache_hits": 1,
            "gpu_shape_executable_cache_misses": 1,
            "gpu_shape_executable_cache_compilations": 1,
            "gpu_shape_executable_cache_source_generation_passes": 1,
            "gpu_shape_executable_cache_binding_recipe_compilations": 1,
            "gpu_shape_executable_cache_capacity_rejections": 0,
            "gpu_workspace_materialization": "reused",
            "gpu_execution_tables_ingest_descriptor_h2d_copies": 0,
            "gpu_composition_direct_split_graphs": 1,
            "gpu_composition_precomputed_compact_commitments": 1,
            "gpu_composition_coefficient_commit_paths": 0,
            "gpu_composition_split_fused_d2d_nodes": 0,
            "gpu_hot_allocations": 0,
            "gpu_hot_allocation_bytes": 0,
            "gpu_hot_frees": 0,
            "gpu_hot_d2d_bytes": 0,
            "gpu_hot_memset_bytes": 0,
            "gpu_hot_fill_words": 0,
            "gpu_hot_capture_begins": 0,
            "gpu_hot_capture_finishes": 0,
            "gpu_hot_capture_aborts": 0,
            "gpu_hot_lane_forks": 0,
            "gpu_hot_lane_joins": 0,
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
            "require_resident_reuse(r, reps)",
            source[validator:],
        )
        validator_body = source[validator:source.index("checkpoint_seal_diagnostic()")]
        self.assertNotIn("max_host_preparation_ns", validator_body)
        promotion = source[source.index("checkpoint_assess_sn2_promotion()") :]
        self.assertIn("gpu_host_preparation_total_ns", promotion)
        self.assertIn("CHECKPOINT_PROMOTION_HOST_PREPARATION_NS", source)

        for field, expected in valid.items():
            mutations = (
                ("compiled", None)
                if field in {
                    "gpu_host_plan_cache_materialization",
                    "gpu_shape_executable_materialization",
                    "gpu_prepared_runtime_materialization",
                }
                else (False, None)
                if field
                in {
                    "gpu_prepared_runtime_capture_ready_at_entry",
                    "gpu_prepared_runtime_capture_ready_at_exit",
                    "gpu_statement_refresh_present",
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

    def test_pod_run_source_projection_is_exact_and_executable(self) -> None:
        helper = ROOT / "loop" / "stage_source_projection.sh"
        self.assertTrue(os.access(helper, os.X_OK))
        self.assertNotEqual(helper.stat().st_mode & 0o111, 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            projection = root / "projection"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "projection@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Projection Test"],
                cwd=repository,
                check=True,
            )

            (repository / ".gitignore").write_text("ignored.cache\n", encoding="utf-8")
            (repository / "unchanged.txt").write_text("unchanged\n", encoding="utf-8")
            (repository / "modified.txt").write_text("before\n", encoding="utf-8")
            (repository / "deleted.txt").write_text("delete me\n", encoding="utf-8")
            executable = repository / "executable.sh"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o644)

            runtime_files = (
                repository / "gpu_benchmarks" / "loop" / "results" / "run.json",
                repository / "gpu_benchmarks" / "loop" / "ledger.jsonl",
                repository / "gpu_benchmarks" / "pie" / "sn" / "SN_PIE_2.zip",
            )
            for path in runtime_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("runtime only\n", encoding="utf-8")

            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "projection fixture"],
                cwd=repository,
                check=True,
            )

            (repository / "modified.txt").write_text("after\n", encoding="utf-8")
            (repository / "deleted.txt").unlink()
            executable.chmod(0o751)
            untracked = repository / "untracked.txt"
            untracked.write_text("untracked\n", encoding="utf-8")
            untracked.chmod(0o600)
            (repository / "ignored.cache").write_text("ignored\n", encoding="utf-8")
            (repository / "gpu_benchmarks" / "pie" / "SN_PIE_3.zip").write_text(
                "runtime only\n", encoding="utf-8"
            )
            (repository / "modified-link").symlink_to("modified.txt")

            subprocess.run(
                [str(helper), str(repository), str(projection)],
                check=True,
                capture_output=True,
                text=True,
            )

            expected = {
                ".gitignore",
                "executable.sh",
                "modified-link",
                "modified.txt",
                "unchanged.txt",
                "untracked.txt",
            }
            projected_files = {
                str(path.relative_to(projection))
                for path in projection.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            self.assertEqual(projected_files, expected)
            self.assertEqual((projection / "unchanged.txt").read_text(), "unchanged\n")
            self.assertEqual((projection / "modified.txt").read_text(), "after\n")
            self.assertEqual((projection / "untracked.txt").read_text(), "untracked\n")
            self.assertTrue((projection / "modified-link").is_symlink())
            self.assertEqual(os.readlink(projection / "modified-link"), "modified.txt")
            for relative in expected - {"modified-link"}:
                source = repository / relative
                copied = projection / relative
                self.assertTrue(copied.is_file() and not copied.is_symlink())
                self.assertEqual(copied.stat().st_mode & 0o7777, source.stat().st_mode & 0o7777)

        pod_run = (ROOT / "loop" / "pod_run.sh").read_text(encoding="utf-8")
        self.assertEqual(pod_run.count("--perms"), 2)
        self.assertNotIn("--no-perms", pod_run)

    def test_cheap_5mhz_pod_run_binds_roots_and_immutable_recipe_snapshot(
        self,
    ) -> None:
        pod_run = (ROOT / "loop" / "pod_run.sh").read_text(encoding="utf-8")
        self.assertIn(
            'CANONICAL_CAIRO_LOCAL="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"',
            pod_run,
        )
        self.assertIn(
            '[[ "$(resolve_path "$CAIRO_LOCAL")" == "$CANONICAL_CAIRO_LOCAL" &&',
            pod_run,
        )
        self.assertIn('PHASES_SHA256="$(file_sha256 "$PHASES_SNAPSHOT")"', pod_run)
        self.assertIn(
            'lease-policy --recipe "$PHASES_SNAPSHOT"',
            pod_run,
        )
        self.assertIn('--recipe "$PHASES_FILE"', pod_run)
        self.assertIn('cat "$PHASES_SNAPSHOT"', pod_run)
        resume = pod_run.index('"$FLEET_CTL" "${RESUME_ARGS[@]}"')
        verify_before_resume = pod_run.rindex("verify_phases_snapshot", 0, resume)
        admitted_endpoint = pod_run.index('ADMITTED_ENDPOINT="$(', resume)
        external_endpoint = pod_run.index('runpodctl ssh info "$POD_ID"', resume)
        endpoint_match = pod_run.index(
            '[[ "$HOST" == "$ADMITTED_HOST" && "$PORT" == "$ADMITTED_PORT" ]]',
            external_endpoint,
        )
        upload = pod_run.index('cat "$PHASES_SNAPSHOT"')
        verify_after_sync = pod_run.rindex("verify_phases_snapshot", resume, upload)
        self.assertLess(verify_before_resume, resume)
        self.assertLess(resume, admitted_endpoint)
        self.assertLess(admitted_endpoint, external_endpoint)
        self.assertLess(external_endpoint, endpoint_match)
        self.assertLess(resume, verify_after_sync)
        self.assertLess(verify_after_sync, upload)

    def test_cheap_5mhz_pod_run_rejects_ambient_source_roots_before_provider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cairo = root / "stwo-cairo"
            stwo = root / "stwo"
            cairo.mkdir()
            stwo.mkdir()
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "loop" / "pod_run.sh"),
                    str(ROOT / "loop/recipes/sn2_5mhz_cheap_gpu_ab.phases"),
                    "root-override-test",
                ],
                env={
                    **os.environ,
                    "CAIRO_LOCAL": str(cairo),
                    "STWO_LOCAL": str(stwo),
                },
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("requires canonical stwo/stwo-cairo roots", result.stderr)
        self.assertNotIn("admitting provider lease", result.stdout + result.stderr)

    def test_pod_run_retries_one_whitelisted_evidence_sync(self) -> None:
        pod_run = (ROOT / "loop" / "pod_run.sh").read_text(encoding="utf-8")
        section = pod_run[
            pod_run.index("# --- 8. fetch evidence ---") :
            pod_run.index("# --- 9. confirmed final lifecycle action")
        ]
        helper = pod_run[
            pod_run.index("fetch_evidence() {") :
            pod_run.index("source_head() {")
        ]
        self.assertIn("for attempt in 1 2 3; do", helper)
        self.assertIn("rsync -azc --partial", helper)
        for pattern in (
            "/*.log",
            "/*.secs",
            "/*.rc",
            "/*.bin",
            "/*.csv",
            "/*.json",
            "/*.ncu-rep",
            "/*.nsys-rep",
            "/*.qdrep",
            "/*.sqlite",
            "/*.txt",
            "/*.xml",
            "/divergence/***",
        ):
            self.assertIn(f"--include='{pattern}'", helper)
        self.assertIn("--exclude='*'", helper)
        self.assertNotIn("scp ", helper)
        self.assertNotIn("compgen", helper)
        self.assertNotIn("scp ", section)
        self.assertNotIn("compgen", section)
        self.assertIn(
            'fetch_evidence || { note "ERROR: complete evidence fetch failed"; RUN_RC=1; }',
            section,
        )
        transfer = section.index("fetch_evidence")
        verify = section.index("ERROR: missing local evidence")
        self.assertLess(transfer, verify)

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

    def test_content_only_rsync_preserves_mtime_and_copies_changed_bytes(self) -> None:
        rsync = shutil.which("rsync")
        if rsync is None:
            self.skipTest("rsync is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            source_file = source / "input.rs"
            destination_file = destination / "input.rs"
            source_file.write_text("same bytes\n", encoding="utf-8")
            destination_file.write_text("same bytes\n", encoding="utf-8")
            os.utime(source_file, ns=(1_500_000_000_000_000_000,) * 2)
            os.utime(destination_file, ns=(1_600_000_000_000_000_000,) * 2)
            destination_mtime = destination_file.stat().st_mtime_ns

            command = [rsync, "-ac", "--no-times", f"{source}/", f"{destination}/"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(destination_file.stat().st_mtime_ns, destination_mtime)

            source_file.write_text("changed bytes\n", encoding="utf-8")
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(destination_file.read_text(encoding="utf-8"), "changed bytes\n")

            verify = subprocess.run(
                [
                    rsync,
                    "-acn",
                    "--no-times",
                    "--itemize-changes",
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
        self.assertEqual(pod_run.count("--perms"), 2)
        self.assertNotIn("--no-perms", pod_run)
        self.assertEqual(projection_body.count("--no-times"), 2)
        self.assertEqual(sync_body.count("--no-times"), 2)
        self.assertEqual(pod_run.count("--no-times"), 2)
        preserved_inputs = "--exclude='gpu_benchmarks/pie/sn/'"
        self.assertEqual(projection_body.count(preserved_inputs), 1)
        self.assertEqual(sync_body.count(preserved_inputs), 1)
        self.assertEqual(pod_run.count(preserved_inputs), 1)
        self.assertNotIn("gpu_benchmarks/pie/sn/*.zip", source)
        self.assertNotIn("gpu_benchmarks/pie/sn/*.zip", pod_run)


if __name__ == "__main__":
    unittest.main()
