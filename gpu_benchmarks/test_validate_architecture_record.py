#!/usr/bin/env python3

import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_cuda_soundness_gate import (
    QUALIFICATION_FLAGS,
    STRICT_RESIDENT_GATE,
    STRICT_RESIDENT_REQUIRED_TESTS,
    gates_for_runtime_mode,
    run_gate,
)
from validate_architecture_record import (
    ARCHITECTURE,
    PREFLIGHT_CAP_BYTES,
    PREFLIGHT_SPECS,
    QUALIFICATION_PROFILES,
    RAW_INPUT_NAMES,
    RETAINED_BUDGET_FLAG,
    SOUNDNESS_COMMANDS,
    SOUNDNESS_GATES,
    STAGES,
    main,
    validate_local_admission,
    validate_benchmark_measurement,
    validate_qualification_soundness_gate,
    validate_record,
    validate_soundness_gate,
)


def source_roots() -> tuple[Path, Path]:
    stwo_cairo = Path(__file__).resolve().parents[1]
    stwo = Path(os.environ.get("STWO_LOCAL", stwo_cairo.parent / "stwo")).resolve()
    return stwo, stwo_cairo


def valid_record() -> dict:
    return {
        "program": "SN_PIE_2.zip",
        "backend": "cuda",
        "engine": "gpu-native",
        "gpu_pcs_driver_architecture": ARCHITECTURE,
        "gpu_pcs_runtime_mode": "DetachedEager",
        "gpu_pcs_stage_started": {stage: 1 for stage in STAGES},
        "gpu_pcs_stage_finished": {stage: 1 for stage in STAGES},
        "gpu_pcs_batched_tree_decommit": True,
        "gpu_pcs_driver_complete": True,
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": "detached-eager",
        "gpu_native_architecture_gate_passed": True,
        "gpu_aot_loads": 2,
        "gpu_aot_cache_hits": 5,
        "gpu_aot_manifest_hash": 0xC0DA,
        "gpu_aot_misses": 0,
        "gpu_aot_runtime_loads": 0,
        "gpu_aot_runtime_cache_hits": 0,
        "gpu_aot_strict_rejections": 0,
        "gpu_aot_provenance_gate_passed": True,
        "performance_claim_admissible": False,
        "steps_per_s": None,
        "mhz": None,
        "useful_mhz": None,
        "mhz_median": None,
        "useful_mhz_median": None,
        "mhz_at_warm_p95": None,
        "useful_mhz_at_warm_p95": None,
        "gpu_host_syncs": None,
        "gpu_graph_launches": None,
        "gpu_kernel_launches": None,
        "gpu_hot_h2d_bytes": None,
        "gpu_hot_d2h_bytes": None,
        "gpu_hot_allocations": None,
        "gpu_max_graph_submit_gap_ms": None,
        "gpu_graph_a_setup_gate_passed": None,
        "gpu_setup_base_migration_copies": None,
        "gpu_setup_lookup_host_copies": None,
        "gpu_setup_legacy_witness_fallbacks": None,
        "gpu_execution_tables_ingest_compact_h2d_bytes": None,
        "gpu_execution_tables_ingest_compact_h2d_copies": None,
        "gpu_execution_tables_ingest_descriptor_h2d_bytes": None,
        "gpu_execution_tables_ingest_descriptor_h2d_copies": None,
        "gpu_execution_tables_ingest_syncs": None,
        "gpu_witness_ingest_syncs": None,
    }


def valid_arena_graph_record() -> dict:
    record = valid_record()
    record.update(
        {
            "gpu_pcs_runtime_mode": "ArenaGraph",
            "gpu_pcs_required_runtime_mode": "arena-graph",
            "performance_claim_admissible": True,
            "steps_per_s": 11_000_000,
            "mhz": 11.4,
            "useful_mhz": 10.6,
            "gpu_host_syncs": 1,
            "gpu_graph_launches": 8,
            "gpu_kernel_launches": 72,
            "gpu_hot_h2d_bytes": 0,
            "gpu_hot_d2h_bytes": 1024,
            "gpu_hot_allocations": 0,
            "gpu_max_graph_submit_gap_ms": 1.25,
            "gpu_graph_a_setup_gate_passed": True,
            "gpu_setup_base_migration_copies": 0,
            "gpu_setup_lookup_host_copies": 0,
            "gpu_setup_legacy_witness_fallbacks": 0,
            "gpu_execution_tables_ingest_compact_h2d_bytes": 4096,
            "gpu_execution_tables_ingest_compact_h2d_copies": 3,
            "gpu_execution_tables_ingest_descriptor_h2d_bytes": 64,
            "gpu_execution_tables_ingest_descriptor_h2d_copies": 2,
            "gpu_execution_tables_ingest_syncs": 1,
            "gpu_witness_ingest_syncs": 1,
        }
    )
    return record


def valid_measurement_record() -> dict:
    record = valid_arena_graph_record()
    record.update(
        {
            "program": "SN_PIE_2.zip",
            "gpu": "NVIDIA H100 80GB HBM3",
            "cycle_count": 9_900_000,
            "pie_n_steps": 9_000_000,
            "reps": 6,
            "verified_reps": 6,
            "warm_sample_count": 5,
            "prove_s_warm_samples_raw": [0.7, 0.8, 0.9, 1.0, 1.1],
            "prove_s_warm_median": 0.9,
            "mhz_median": 11.0,
            "useful_mhz_median": 10.0,
            "throughput_distribution_applicable": True,
        }
    )
    return record


def valid_soundness_artifact(runtime_mode: str = "detached-eager") -> dict:
    gates = gates_for_runtime_mode(runtime_mode)
    binary_sha = "b" * 64
    execution_target = {
        "schema": "stwo.remote-execution-target.v1",
        "pod_id": "dry-run-pod",
        "boot_id": "00000000-0000-4000-8000-000000000001",
        "gpu_uuid": "GPU-00000000-0000-4000-8000-000000000001",
        "gpu_name": "DRY-RUN-GPU",
        "gpu_bench": {
            "path": f"/workspace/bench_loop_runs/sealed/gpu_bench.{binary_sha}",
            "sha256": binary_sha,
        },
        "inputs": {
            "gate": {"path": "/workspace/SN_PIE_2.zip", "sha256": "c" * 64}
        },
    }
    target_sha = hashlib.sha256(
        json.dumps(execution_target, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = {
        "schema": "stwo.cuda.soundness-gate.v3",
        "stwo_git_head": "1" * 40,
        "stwo_cairo_git_head": "2" * 40,
        "stwo_worktree_hash": "3" * 64,
        "stwo_cairo_worktree_hash": "4" * 64,
        "runtime_mode": runtime_mode,
        "execution_target": execution_target,
        "execution_target_sha256": target_sha,
        "execution_target_postcheck": True,
        "source_projection": {
            "method": "rsync-archive-checksum-dry-run-clean",
            "verified_after_soundness": True,
            "source": {
                "stwo": {"head": "1" * 40, "worktree_hash": "3" * 64},
                "stwo_cairo": {"head": "2" * 40, "worktree_hash": "4" * 64},
            },
        },
        "passed": True,
        "gates": [
            {
                "name": name,
                "command": SOUNDNESS_COMMANDS[name],
                "exit_code": 0,
                "executed_tests": required,
                "required_tests": required,
                "passed": True,
            }
            for name, command, required in gates
        ],
    }
    for gate in artifact["gates"]:
        if gate["name"] == "strict_resident_whole_proof_simd_byte_identity":
            gate["required_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
            gate["executed_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
    return artifact


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualification_source() -> dict:
    return {
        "stwo": {"head": "1" * 40, "worktree_hash": "3" * 64},
        "stwo_cairo": {"head": "2" * 40, "worktree_hash": "4" * 64},
    }


def valid_qualification_soundness_artifact() -> dict:
    artifact = valid_soundness_artifact("arena-graph")
    source = qualification_source()
    artifact.update(
        {
            "dry_run": False,
            "synced_source": {**source, "transport": "rsync-archive-checksum"},
            "qualification_flags": {flag: 1 for flag in QUALIFICATION_FLAGS},
            "effective_stwo_env": {
                "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
                "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
                "STWO_PARITY_REF_STWO_HEAD": source["stwo"]["head"],
                "STWO_PARITY_REF_STWO_WORKTREE_HASH": source["stwo"][
                    "worktree_hash"
                ],
                "STWO_PARITY_REF_STWO_CAIRO_HEAD": source["stwo_cairo"]["head"],
                "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": source["stwo_cairo"][
                    "worktree_hash"
                ],
                **{flag: "1" for flag in QUALIFICATION_FLAGS},
                RETAINED_BUDGET_FLAG: "29469326848",
            },
        }
    )
    return artifact


def write_manifest(path: Path, entries: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())),
        encoding="utf-8",
    )


def local_admission_fixture(root: Path) -> tuple[dict, Path, dict[str, Path]]:
    adapted_dir = root / "adapted_inputs"
    adapted_dir.mkdir()
    adapted_entries = {}
    for pie in range(1, 5):
        name = f"SN_PIE_{pie}.adapted.bin"
        path = adapted_dir / name
        path.write_bytes(f"adapted-{pie}".encode())
        adapted_entries[name] = sha256_file(path)

    adapted_manifest = root / "adapted_sha256s"
    pinned_adapted_manifest = root / "ADAPTED_SHA256SUMS"
    write_manifest(adapted_manifest, adapted_entries)
    write_manifest(pinned_adapted_manifest, adapted_entries)

    bootloader = root / "simple_bootloader_compiled.json"
    bootloader.write_text('{"bootloader":true}\n', encoding="utf-8")
    raw_entries = {
        name: hashlib.sha256(name.encode()).hexdigest() for name in RAW_INPUT_NAMES
    }
    raw_entries[bootloader.name] = sha256_file(bootloader)
    raw_manifest = root / "SHA256SUMS"
    write_manifest(raw_manifest, raw_entries)

    gpu_bench = root / "gpu_bench"
    gpu_bench.write_bytes(b"current-adapter-binary")
    preflight_hashes = {}
    for artifact_name, (input_name, runtime_policy) in PREFLIGHT_SPECS.items():
        artifact_path = root / artifact_name
        artifact_path.write_text(
            json.dumps(
                {
                    "pass": True,
                    "vram_fit": True,
                    "vram_budget_bytes": PREFLIGHT_CAP_BYTES,
                    "arena": {"total_bytes": PREFLIGHT_CAP_BYTES - 1},
                    "runtime_policy": runtime_policy,
                    "source": str(adapted_dir / input_name),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        preflight_hashes[artifact_name] = sha256_file(artifact_path)

    admission = {
        "schema": "stwo.local-preflight-admission.v1",
        "passed": True,
        "dry_run": False,
        "runtime_mode": "arena-graph",
        "source": qualification_source(),
        "profiles": QUALIFICATION_PROFILES,
        "preflight_ceiling_bytes": PREFLIGHT_CAP_BYTES,
        "preflight_artifact_sha256": preflight_hashes,
        "adapted_input_manifest": str(adapted_manifest),
        "adapted_input_manifest_sha256": sha256_file(adapted_manifest),
        "adapter_reproduction": {
            "byte_equal": True,
            "gpu_bench_binary_sha256": sha256_file(gpu_bench),
            "raw_input_manifest_sha256": sha256_file(raw_manifest),
            "bootloader_sha256": sha256_file(bootloader),
            "pinned_adapted_manifest_sha256": sha256_file(
                pinned_adapted_manifest
            ),
        },
    }
    admission_path = root / "local_admission.json"
    admission_path.write_text(json.dumps(admission) + "\n", encoding="utf-8")
    paths = {
        "gpu_bench_binary": gpu_bench,
        "raw_input_manifest": raw_manifest,
        "bootloader": bootloader,
        "pinned_adapted_manifest": pinned_adapted_manifest,
        "adapted_manifest": adapted_manifest,
        "adapted_input": adapted_dir / "SN_PIE_1.adapted.bin",
        "preflight": root / "preflight_universal_SN1.json",
    }
    return admission, admission_path, paths


class ArchitectureRecordTest(unittest.TestCase):
    def test_accepts_complete_typed_cuda_record(self) -> None:
        self.assertEqual(validate_record(valid_record(), "detached-eager"), [])

    def test_rejects_legacy_simd_missing_partial_and_duplicate_records(self) -> None:
        mutations = (
            ("engine", "legacy"),
            ("backend", "simd"),
            ("gpu_pcs_driver_architecture", None),
            ("gpu_pcs_driver_complete", False),
            ("gpu_pcs_batched_tree_decommit", False),
            ("gpu_aot_misses", 1),
            ("gpu_aot_runtime_loads", 1),
            ("gpu_aot_runtime_cache_hits", 1),
            ("gpu_aot_strict_rejections", 1),
            ("gpu_aot_provenance_gate_passed", False),
            ("gpu_aot_manifest_hash", 0),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                record = valid_record()
                record[field] = value
                self.assertTrue(validate_record(record, "detached-eager"))

        partial = valid_record()
        partial["gpu_pcs_stage_finished"][STAGES[2]] = 0
        self.assertTrue(validate_record(partial, "detached-eager"))

        duplicate = valid_record()
        duplicate["gpu_pcs_stage_started"][STAGES[4]] = 2
        self.assertTrue(validate_record(duplicate, "detached-eager"))

        boolean_count = valid_record()
        boolean_count["gpu_pcs_stage_started"][STAGES[0]] = True
        boolean_count["gpu_aot_misses"] = False
        self.assertTrue(validate_record(boolean_count, "detached-eager"))

    def test_arena_graph_requirement_rejects_detached_record(self) -> None:
        errors = validate_record(valid_record(), "arena-graph")
        self.assertTrue(any("gpu_pcs_runtime_mode" in error for error in errors))

    def test_detached_gpu_native_record_cannot_publish_performance(self) -> None:
        record = valid_record()
        record["performance_claim_admissible"] = True
        record["useful_mhz"] = 1.3
        errors = validate_record(record, "detached-eager")
        self.assertTrue(any("performance_claim_admissible" in error for error in errors))
        self.assertTrue(any("useful_mhz" in error for error in errors))

    def test_arena_graph_requires_one_sync_bounded_launches_and_no_hot_setup(self) -> None:
        record = valid_arena_graph_record()
        self.assertEqual(validate_record(record, "arena-graph"), [])
        record["gpu_kernel_launches"] = 100
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_kernel_launches"] = 72
        record["gpu_host_syncs"] = 0
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_host_syncs"] = 1
        record["gpu_setup_lookup_host_copies"] = 1
        self.assertTrue(validate_record(record, "arena-graph"))

    def test_arena_graph_fails_closed_without_execution_table_ingest(self) -> None:
        record = valid_arena_graph_record()
        self.assertEqual(validate_record(record, "arena-graph"), [])

        for field in (
            "gpu_execution_tables_ingest_compact_h2d_bytes",
            "gpu_execution_tables_ingest_compact_h2d_copies",
            "gpu_execution_tables_ingest_descriptor_h2d_bytes",
            "gpu_execution_tables_ingest_descriptor_h2d_copies",
            "gpu_execution_tables_ingest_syncs",
        ):
            with self.subTest(field=field):
                missing = record.copy()
                missing[field] = None
                self.assertTrue(validate_record(missing, "arena-graph"))

                bypassed = record.copy()
                bypassed[field] = 0
                self.assertTrue(validate_record(bypassed, "arena-graph"))

        extra_sync = record.copy()
        extra_sync["gpu_execution_tables_ingest_syncs"] = 2
        self.assertTrue(validate_record(extra_sync, "arena-graph"))

    def test_benchmark_measurement_binds_input_samples_and_headline_math(self) -> None:
        record = valid_measurement_record()
        arguments = {
            "expected_program": "SN_PIE_2.zip",
            "expected_reps": 6,
            "expected_gpu": "NVIDIA H100 80GB HBM3",
        }
        self.assertEqual(validate_benchmark_measurement(record, **arguments), [])
        mutations = (
            ("program", lambda value: value.update({"program": "SN_PIE_1.zip"})),
            ("gpu", lambda value: value.update({"gpu": "NVIDIA A100-SXM4-80GB"})),
            ("reps", lambda value: value.update({"reps": 5})),
            ("verified", lambda value: value.update({"verified_reps": 5})),
            ("sample count", lambda value: value["prove_s_warm_samples_raw"].pop()),
            ("nonfinite sample", lambda value: value["prove_s_warm_samples_raw"].__setitem__(0, float("inf"))),
            ("median", lambda value: value.update({"prove_s_warm_median": 0.8})),
            ("headline", lambda value: value.update({"useful_mhz_median": 12.0})),
            ("cycle rate", lambda value: value.update({"mhz_median": 12.0})),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(record)
                mutate(candidate)
                self.assertTrue(validate_benchmark_measurement(candidate, **arguments))

    def test_soundness_gate_requires_nonzero_execution_for_every_command(self) -> None:
        artifact = valid_soundness_artifact()
        self.assertEqual(validate_soundness_gate(artifact), [])
        artifact["gates"][0]["executed_tests"] = 0
        self.assertTrue(validate_soundness_gate(artifact))
        artifact["gates"][0]["executed_tests"] = artifact["gates"][0]["required_tests"]
        artifact["gates"][0]["passed"] = False
        self.assertTrue(validate_soundness_gate(artifact))

    def test_soundness_gate_rejects_unreviewed_extra_test_execution(self) -> None:
        artifact = valid_soundness_artifact()
        artifact["gates"][0]["executed_tests"] += 1
        self.assertTrue(validate_soundness_gate(artifact))

    def test_soundness_gate_rejects_remote_execution_target_drift(self) -> None:
        mutations = (
            lambda value: value["execution_target"].update({"pod_id": "other-pod"}),
            lambda value: value.update({"execution_target_postcheck": False}),
            lambda value: value["execution_target"]["gpu_bench"].update(
                {"path": "/workspace/target/release/gpu_bench"}
            ),
            lambda value: value["execution_target"]["inputs"]["gate"].update(
                {"sha256": "not-a-hash"}
            ),
            lambda value: value["source_projection"].update(
                {"verified_after_soundness": False}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                artifact = valid_soundness_artifact()
                mutate(artifact)
                self.assertTrue(validate_soundness_gate(artifact))

    def test_soundness_gate_rejects_truncated_or_forged_manifest(self) -> None:
        artifact = valid_soundness_artifact()
        artifact["gates"].pop()
        self.assertTrue(validate_soundness_gate(artifact))
        artifact["gates"][-1]["required_tests"] = 0
        self.assertTrue(validate_soundness_gate(artifact))
        artifact["gates"][-1]["required_tests"] = SOUNDNESS_GATES[
            artifact["gates"][-1]["name"]
        ]
        artifact["gates"][-1]["command"] = ["true"]
        self.assertTrue(validate_soundness_gate(artifact))

    def test_soundness_manifest_is_scoped_to_the_requested_runtime(self) -> None:
        detached = valid_soundness_artifact("detached-eager")
        detached_names = {gate["name"] for gate in detached["gates"]}
        self.assertNotIn(
            "strict_resident_whole_proof_simd_byte_identity", detached_names
        )
        self.assertEqual(
            validate_soundness_gate(detached, "detached-eager"), []
        )
        self.assertTrue(validate_soundness_gate(detached, "arena-graph"))

        arena = valid_soundness_artifact("arena-graph")
        arena_names = {gate["name"] for gate in arena["gates"]}
        self.assertIn("strict_resident_whole_proof_simd_byte_identity", arena_names)
        self.assertEqual(validate_soundness_gate(arena, "arena-graph"), [])
        arena["gates"] = [
            gate
            for gate in arena["gates"]
            if gate["name"] != "strict_resident_whole_proof_simd_byte_identity"
        ]
        self.assertTrue(validate_soundness_gate(arena, "arena-graph"))

    def test_strict_resident_named_manifest_is_exact(self) -> None:
        artifact = valid_soundness_artifact("arena-graph")
        self.assertEqual(validate_soundness_gate(artifact, "arena-graph"), [])
        strict = next(
            gate
            for gate in artifact["gates"]
            if gate["name"] == "strict_resident_whole_proof_simd_byte_identity"
        )
        for mutation in ("missing", "extra", "duplicate"):
            with self.subTest(mutation=mutation):
                candidate = valid_soundness_artifact("arena-graph")
                gate = next(
                    item for item in candidate["gates"] if item["name"] == strict["name"]
                )
                if mutation == "missing":
                    gate["executed_test_names"].pop()
                elif mutation == "extra":
                    gate["executed_test_names"].append("unreviewed_test")
                else:
                    gate["executed_test_names"][-1] = gate["executed_test_names"][0]
                self.assertTrue(validate_soundness_gate(candidate, "arena-graph"))

        detached = valid_soundness_artifact("detached-eager")
        detached["gates"][0]["executed_tests"] = detached["gates"][0]["required_tests"]
        self.assertEqual(validate_soundness_gate(detached, "detached-eager"), [])

    def test_strict_runner_accepts_nocapture_interleaving_only_when_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stwo = Path(directory)
            source = stwo / "crates/gpu-prover/tests/resident_parity_native.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                'const STRICT_RESIDENT_FIXTURE: &str = "fixture.zip";\n',
                encoding="utf-8",
            )
            lines = [f"running {len(STRICT_RESIDENT_REQUIRED_TESTS)} tests"]
            for name in STRICT_RESIDENT_REQUIRED_TESTS:
                lines.extend((f"test {name} ... diagnostic output", "ok"))
            lines.append(
                f"test result: ok. {len(STRICT_RESIDENT_REQUIRED_TESTS)} passed; "
                "0 failed; 0 ignored; 0 measured; 0 filtered out"
            )
            output = "\n".join(lines)

            def run(candidate: str, returncode: int = 0) -> dict:
                completed = mock.Mock(stdout=candidate, returncode=returncode)
                with mock.patch(
                    "run_cuda_soundness_gate.subprocess.run", return_value=completed
                ), mock.patch("run_cuda_soundness_gate.sys.stderr"):
                    return run_gate(
                        stwo,
                        STRICT_RESIDENT_GATE,
                        ("cargo", "test"),
                        len(STRICT_RESIDENT_REQUIRED_TESTS),
                    )

            record = run(output)
            self.assertTrue(record["passed"])
            self.assertEqual(
                record["executed_test_names"], list(STRICT_RESIDENT_REQUIRED_TESTS)
            )
            self.assertFalse(
                run(output.replace(STRICT_RESIDENT_REQUIRED_TESTS[-1], "mutated", 1))[
                    "passed"
                ]
            )
            self.assertFalse(
                run(
                    output.replace(
                        f"{len(STRICT_RESIDENT_REQUIRED_TESTS)} passed",
                        f"{len(STRICT_RESIDENT_REQUIRED_TESTS) - 1} passed",
                    )
                )["passed"]
            )
            self.assertFalse(run(output, returncode=1)["passed"])

    def test_soundness_manifest_covers_cfg_native_targets_with_exact_counts(self) -> None:
        stwo, stwo_cairo = source_roots()
        test_roots = (
            (stwo / "crates/backend-cuda/tests", "stwo-backend-cuda"),
            (
                stwo_cairo / "stwo_cairo_prover/crates/gpu-prover/tests",
                "stwo-cairo-gpu-prover",
            ),
            (
                stwo_cairo / "stwo_cairo_prover/crates/prover/tests",
                "stwo-cairo-prover",
            ),
        )
        for root, _ in test_roots:
            self.assertTrue(root.is_dir(), f"native-test root is absent: {root}")
        discovered = {}
        for root, package in test_roots:
            for path in root.glob("*_native.rs"):
                source = path.read_text(encoding="utf-8")
                if "#![cfg(stwo_cuda_link)]" in source:
                    discovered[(package, path.stem)] = len(
                        re.findall(r"(?m)^#\[test\]\s*$", source)
                    )

        manifested = {}
        for name, command in SOUNDNESS_COMMANDS.items():
            if "--test" not in command:
                continue
            target = command[command.index("--test") + 1]
            if not target.endswith("_native"):
                continue
            package = command[command.index("-p") + 1]
            key = (package, target)
            self.assertNotIn(key, manifested)
            manifested[key] = SOUNDNESS_GATES[name]

        self.assertEqual(manifested, discovered)
        self.assertEqual(
            SOUNDNESS_GATES[
                "prepared_execution_tables_eager_capture_mutated_content_reference"
            ],
            2,
        )
        self.assertEqual(
            SOUNDNESS_GATES[
                "prepared_fixed_table_eager_capture_mutated_content_reference"
            ],
            1,
        )
        self.assertEqual(
            SOUNDNESS_GATES["prepared_numerator_eager_capture_reference"],
            2,
        )
        self.assertEqual(
            SOUNDNESS_GATES["strict_resident_whole_proof_simd_byte_identity"],
            len(STRICT_RESIDENT_REQUIRED_TESTS),
        )
        composition_command = SOUNDNESS_COMMANDS[
            "prepared_composition_eager_capture_cpu_reference"
        ]
        self.assertIn("--features", composition_command)
        self.assertEqual(
            composition_command[composition_command.index("--features") + 1],
            "direct-retention-test-api",
        )

    def test_reference_cache_tests_cannot_be_absorbed_by_strict_resident_target(self) -> None:
        _, stwo_cairo = source_roots()
        tests = stwo_cairo / "stwo_cairo_prover/crates/gpu-prover/tests"
        strict = (tests / "resident_parity_native.rs").read_text(encoding="utf-8")
        common = (tests / "common/reference_cache.rs").read_text(encoding="utf-8")
        host = (tests / "reference_cache_host.rs").read_text(encoding="utf-8")
        self.assertIn('#[path = "common/reference_cache.rs"]', strict)
        self.assertIsNone(re.search(r"#\[(?:test|cfg\(test\))\]", common))
        self.assertEqual(len(re.findall(r"(?m)^#\[test\]\s*$", host)), 8)


class PrePodValidationTest(unittest.TestCase):
    def validate_fixture(
        self, admission: dict, admission_path: Path, paths: dict[str, Path]
    ) -> list[str]:
        return validate_local_admission(
            admission,
            admission_path,
            expected_source=qualification_source(),
            expected_dry_run=False,
            gpu_bench_binary=paths["gpu_bench_binary"],
            raw_input_manifest=paths["raw_input_manifest"],
            bootloader=paths["bootloader"],
            pinned_adapted_manifest=paths["pinned_adapted_manifest"],
        )

    def test_accepts_complete_local_admission_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            admission, admission_path, paths = local_admission_fixture(Path(directory))
            self.assertEqual(self.validate_fixture(admission, admission_path, paths), [])
            source = qualification_source()
            self.assertEqual(
                main(
                    [
                        "--local-admission",
                        str(admission_path),
                        "--gpu-bench-binary",
                        str(paths["gpu_bench_binary"]),
                        "--raw-input-manifest",
                        str(paths["raw_input_manifest"]),
                        "--bootloader",
                        str(paths["bootloader"]),
                        "--pinned-adapted-manifest",
                        str(paths["pinned_adapted_manifest"]),
                        "--expected-dry-run",
                        "0",
                        "--stwo-head",
                        source["stwo"]["head"],
                        "--stwo-worktree-hash",
                        source["stwo"]["worktree_hash"],
                        "--stwo-cairo-head",
                        source["stwo_cairo"]["head"],
                        "--stwo-cairo-worktree-hash",
                        source["stwo_cairo"]["worktree_hash"],
                    ]
                ),
                0,
            )

    def test_local_admission_requires_exact_contract(self) -> None:
        mutations = (
            ("extra profile", lambda value: value["profiles"].update({"extra": "X=1"})),
            (
                "capacity drift",
                lambda value: value.update(
                    {"preflight_ceiling_bytes": PREFLIGHT_CAP_BYTES + 1}
                ),
            ),
            (
                "missing preflight",
                lambda value: value["preflight_artifact_sha256"].pop(
                    "preflight_flags_off_SN2.json"
                ),
            ),
            (
                "non-byte-equal adapter",
                lambda value: value["adapter_reproduction"].update(
                    {"byte_equal": False}
                ),
            ),
            (
                "extra adapter field",
                lambda value: value["adapter_reproduction"].update({"extra": True}),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            admission, admission_path, paths = local_admission_fixture(Path(directory))
            for name, mutate in mutations:
                with self.subTest(name=name):
                    candidate = copy.deepcopy(admission)
                    mutate(candidate)
                    self.assertTrue(
                        self.validate_fixture(candidate, admission_path, paths)
                    )

    def test_local_admission_rehashes_every_referenced_artifact(self) -> None:
        targets = (
            "gpu_bench_binary",
            "raw_input_manifest",
            "bootloader",
            "pinned_adapted_manifest",
            "adapted_manifest",
            "adapted_input",
            "preflight",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                admission, admission_path, paths = local_admission_fixture(Path(directory))
                path = paths[target]
                path.write_bytes(path.read_bytes() + b"stale")
                self.assertTrue(self.validate_fixture(admission, admission_path, paths))

    def test_local_admission_validates_preflight_policy_and_source(self) -> None:
        for mutation in ("policy", "source", "capacity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                admission, admission_path, paths = local_admission_fixture(Path(directory))
                artifact = paths["preflight"]
                record = json.loads(artifact.read_text(encoding="utf-8"))
                if mutation == "policy":
                    record["runtime_policy"]["commit_mode"] = "FullLifting"
                elif mutation == "source":
                    record["source"] = str(Path(directory) / "wrong.bin")
                else:
                    record["arena"]["total_bytes"] = PREFLIGHT_CAP_BYTES + 1
                artifact.write_text(json.dumps(record) + "\n", encoding="utf-8")
                admission["preflight_artifact_sha256"][artifact.name] = sha256_file(
                    artifact
                )
                self.assertTrue(self.validate_fixture(admission, admission_path, paths))

    def test_qualification_soundness_requires_full_manifest_and_headline_env(self) -> None:
        artifact = valid_qualification_soundness_artifact()
        self.assertEqual(
            validate_qualification_soundness_gate(
                artifact,
                expected_source=qualification_source(),
                expected_dry_run=False,
            ),
            [],
        )
        mutations = (
            ("truncated gates", lambda value: value["gates"].pop()),
            (
                "wrong flag",
                lambda value: value["qualification_flags"].update(
                    {QUALIFICATION_FLAGS[0]: 0}
                ),
            ),
            (
                "extra environment",
                lambda value: value["effective_stwo_env"].update(
                    {"STWO_UNAPPROVED": "1"}
                ),
            ),
            (
                "wrong source",
                lambda value: value["synced_source"]["stwo"].update(
                    {"head": "f" * 40}
                ),
            ),
            ("wrong dry state", lambda value: value.update({"dry_run": True})),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(artifact)
                mutate(candidate)
                self.assertTrue(
                    validate_qualification_soundness_gate(
                        candidate,
                        expected_source=qualification_source(),
                        expected_dry_run=False,
                    )
                )

    def test_accepts_soundness_only_cli(self) -> None:
        artifact = valid_qualification_soundness_artifact()
        source = qualification_source()
        with tempfile.TemporaryDirectory() as directory:
            soundness_path = Path(directory) / "soundness.json"
            soundness_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--soundness-only",
                        "--soundness-gate",
                        str(soundness_path),
                        "--expected-dry-run",
                        "0",
                        "--stwo-head",
                        source["stwo"]["head"],
                        "--stwo-worktree-hash",
                        source["stwo"]["worktree_hash"],
                        "--stwo-cairo-head",
                        source["stwo_cairo"]["head"],
                        "--stwo-cairo-worktree-hash",
                        source["stwo_cairo"]["worktree_hash"],
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
