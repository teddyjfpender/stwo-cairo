#!/usr/bin/env python3

import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_cuda_soundness_gate import (
    QUALIFICATION_FLAGS,
    REFERENCE_CACHE_SOURCE_ENV,
    STRICT_RESIDENT_GATE,
    STRICT_RESIDENT_REQUIRED_TESTS,
    gates_for_runtime_mode,
    main as run_soundness_main,
    run_gate,
)
from validate_architecture_record import (
    ARCHITECTURE,
    GPU_TELEMETRY_COLUMNS,
    GPU_TELEMETRY_MAX_GAP_NS,
    GPU_TELEMETRY_SAMPLE_INTERVAL_MS,
    GPU_TELEMETRY_SCHEMA,
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
    validate_gpu_telemetry_artifact,
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
        "gpu_expected_graph_launches": None,
        "gpu_expected_kernel_launches": None,
        "gpu_transcript_segments": None,
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
            "gpu_graph_launches": 29,
            "gpu_kernel_launches": 7_859,
            "gpu_expected_graph_launches": 29,
            "gpu_expected_kernel_launches": 7_859,
            "gpu_transcript_segments": 30,
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
            "simd_reference_required": True,
            "simd_reference_comparison_applicable": True,
            "simd_reference_byte_equal": True,
            "gpu_proof_blake3": "5" * 64,
            "simd_reference_blake3": "5" * 64,
            "simd_reference_fresh": True,
            "simd_reference_s": 1.001,
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
            "gpu_proof_loop_started_unix_ns": 1_700_000_000_000_000_000,
            "gpu_proof_loop_finished_unix_ns": 1_700_000_060_000_000_000,
            "prove_s_warm_samples_raw": [0.7, 0.8, 0.9, 1.0, 1.1],
            "prove_s_warm_median": 0.9,
            "prove_s_warm_p95": 1.08,
            "mhz_median": 11.0,
            "useful_mhz_median": 10.0,
            "mhz_at_warm_p95": 9.167,
            "useful_mhz_at_warm_p95": 8.333,
            "prove_s_cold": 1.2,
            "verify_ms": 42.0,
            "proof_kb": 210.5,
            "peak_rss_gb": 18.2,
            "vram_end_gb": 6.1,
            "vram_peak_gb": 11.3,
            "pool_used_high_gb": 5.0,
            "pool_reserved_high_gb": 6.0,
            "security_bits": 96,
            "n_queries": 70,
            "pow_bits": 26,
            "fold_step": 3,
            "proof_comparison_applicable": True,
            "proof_byte_equal_required": True,
            "proof_byte_equal": True,
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
        "synced_source": {
            "stwo": {"head": "1" * 40, "worktree_hash": "3" * 64},
            "stwo_cairo": {"head": "2" * 40, "worktree_hash": "4" * 64},
            "transport": "rsync-archive-checksum",
        },
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
                "stub_skip_detected": False,
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
    occurrence = {
        "kind": "constraint",
        "component": "fixture_component",
        "instance": 0,
        "kernel": 0,
        "kernel_name": "stwo_jit_fused_1111111111111111",
        "semantic_hash": "1111111111111111",
        "cache_key": "2222222222222222",
    }
    occurrences = [occurrence]
    aot_manifest = root / "aot_manifest.json"
    aot_manifest.write_text(json.dumps([{
        "kind": "constraint",
        "label": "fixture_component",
        "kernel_name": occurrence["kernel_name"],
        "cache_key": occurrence["cache_key"],
        "semantic_hash": occurrence["semantic_hash"],
        "file": "constraint_fixture_component_2222222222222222.cu",
    }]) + "\n", encoding="utf-8")
    preflight_hashes = {}
    preflight_occurrence_sha256 = {}
    preflight_occurrence_blake3 = {}
    preflight_key_blake3 = {}
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
                    "aot_coverage": {
                        "pass": True,
                        "manifest": str(aot_manifest),
                        "manifest_blake3": "a" * 64,
                        "manifest_entries": 1,
                        "required_occurrences": occurrences,
                        "required_occurrences_blake3": "b" * 64,
                        "required_occurrence_count": 1,
                        "required_unique_keys_blake3": "c" * 64,
                        "required_unique_key_count": 1,
                        "missing_occurrences": [],
                        "missing_occurrence_count": 0,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        preflight_hashes[artifact_name] = sha256_file(artifact_path)
        preflight_occurrence_sha256[artifact_name] = hashlib.sha256(
            json.dumps(occurrences, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        preflight_occurrence_blake3[artifact_name] = "b" * 64
        preflight_key_blake3[artifact_name] = "c" * 64

    occurrence_sha = hashlib.sha256(
        json.dumps(occurrences, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    key_sha = hashlib.sha256(b'["2222222222222222"]').hexdigest()
    per_sn = {
        f"SN_PIE_{pie}": {
            "required_occurrence_count": 1,
            "required_unique_key_count": 1,
            "required_occurrences_sha256": occurrence_sha,
            "required_unique_keys_sha256": key_sha,
        }
        for pie in range(1, 5)
    }

    admission = {
        "schema": "stwo.local-preflight-admission.v1",
        "passed": True,
        "dry_run": False,
        "runtime_mode": "arena-graph",
        "source": qualification_source(),
        "profiles": QUALIFICATION_PROFILES,
        "preflight_ceiling_bytes": PREFLIGHT_CAP_BYTES,
        "preflight_artifact_sha256": preflight_hashes,
        "aot_coverage": {
            "manifest": str(aot_manifest),
            "manifest_sha256": sha256_file(aot_manifest),
            "manifest_blake3": "a" * 64,
            "manifest_entry_count": 1,
            "required_occurrence_count": 1,
            "required_unique_key_count": 1,
            "required_unique_keys": ["2222222222222222"],
            "required_occurrences_sha256": occurrence_sha,
            "required_unique_keys_sha256": key_sha,
            "preflight_occurrences_sha256": preflight_occurrence_sha256,
            "preflight_occurrences_blake3": preflight_occurrence_blake3,
            "preflight_unique_keys_blake3": preflight_key_blake3,
            "per_sn": per_sn,
            "missing_occurrences": [],
        },
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
        "aot_manifest": aot_manifest,
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

    def test_arena_graph_requires_one_sync_exact_captured_topology_and_no_hot_setup(
        self,
    ) -> None:
        record = valid_arena_graph_record()
        self.assertEqual(validate_record(record, "arena-graph"), [])
        record["gpu_graph_launches"] = 28
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_graph_launches"] = 29
        record["gpu_expected_graph_launches"] = 0
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_expected_graph_launches"] = 29
        record["gpu_transcript_segments"] = 29
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_transcript_segments"] = True
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_transcript_segments"] = None
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_transcript_segments"] = 30
        record["gpu_kernel_launches"] = 0
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_kernel_launches"] = 7_859
        record["gpu_expected_kernel_launches"] = 28
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_expected_kernel_launches"] = 100_000
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_kernel_launches"] = 100_000
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_kernel_launches"] = 7_859
        record["gpu_expected_kernel_launches"] = 7_859
        record["gpu_host_syncs"] = 0
        self.assertTrue(validate_record(record, "arena-graph"))
        record["gpu_host_syncs"] = 1
        record["gpu_setup_lookup_host_copies"] = 1
        self.assertTrue(validate_record(record, "arena-graph"))

    def test_arena_graph_accepts_production_fold_step_three_topology(self) -> None:
        record = valid_arena_graph_record()
        record.update(
            {
                "gpu_graph_launches": 14,
                "gpu_expected_graph_launches": 14,
                "gpu_transcript_segments": 15,
                "gpu_kernel_launches": 2_530,
                "gpu_expected_kernel_launches": 2_530,
            }
        )
        self.assertEqual(validate_record(record, "arena-graph"), [])

    def test_graph_gap_diagnostic_softens_only_submit_timing(self) -> None:
        record = valid_arena_graph_record()
        record.update(
            {
                "benchmark_diagnostic_mode": True,
                "benchmark_diagnostic_reason": "graph-submit-gap-only",
                "performance_measurement_available": True,
                "performance_claim_admissible": False,
                "gpu_max_graph_submit_gap_ms": 918.959783,
                "gpu_graph_submit_gap_strict_gate_passed": False,
            }
        )
        self.assertEqual(
            validate_record(record, "arena-graph", graph_gap_diagnostic=True), []
        )
        self.assertTrue(validate_record(record, "arena-graph"))

        structural_regression = record.copy()
        structural_regression["gpu_graph_launches"] = 13
        self.assertTrue(
            validate_record(
                structural_regression,
                "arena-graph",
                graph_gap_diagnostic=True,
            )
        )

        dishonest_strict_result = record.copy()
        dishonest_strict_result["gpu_graph_submit_gap_strict_gate_passed"] = True
        self.assertTrue(
            validate_record(
                dishonest_strict_result,
                "arena-graph",
                graph_gap_diagnostic=True,
            )
        )

    def test_arena_graph_topology_fields_fail_closed_without_type_errors(self) -> None:
        for field in (
            "gpu_graph_launches",
            "gpu_expected_graph_launches",
            "gpu_transcript_segments",
            "gpu_kernel_launches",
            "gpu_expected_kernel_launches",
        ):
            for value in (None, "14", True):
                with self.subTest(field=field, value=value):
                    record = valid_arena_graph_record()
                    record[field] = value
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
            "require_fresh_simd_reference": True,
        }
        self.assertEqual(validate_benchmark_measurement(record, **arguments), [])
        mutations = (
            ("program", lambda value: value.update({"program": "SN_PIE_1.zip"})),
            ("gpu", lambda value: value.update({"gpu": "NVIDIA A100-SXM4-80GB"})),
            ("reps", lambda value: value.update({"reps": 5})),
            ("verified", lambda value: value.update({"verified_reps": 5})),
            (
                "missing proof window",
                lambda value: value.pop("gpu_proof_loop_started_unix_ns"),
            ),
            (
                "reversed proof window",
                lambda value: value.update(
                    {"gpu_proof_loop_finished_unix_ns": 1_699_999_999_000_000_000}
                ),
            ),
            (
                "proof window shorter than samples",
                lambda value: value.update(
                    {"gpu_proof_loop_finished_unix_ns": 1_700_000_001_000_000_000}
                ),
            ),
            ("sample count", lambda value: value["prove_s_warm_samples_raw"].pop()),
            ("nonfinite sample", lambda value: value["prove_s_warm_samples_raw"].__setitem__(0, float("inf"))),
            ("median", lambda value: value.update({"prove_s_warm_median": 0.8})),
            ("p95", lambda value: value.update({"prove_s_warm_p95": 0.9})),
            ("headline", lambda value: value.update({"useful_mhz_median": 12.0})),
            ("cycle rate", lambda value: value.update({"mhz_median": 12.0})),
            ("p95 rate", lambda value: value.update({"useful_mhz_at_warm_p95": 12.0})),
            ("security", lambda value: value.update({"security_bits": 95})),
            ("cold timing", lambda value: value.update({"prove_s_cold": 0.0})),
            ("proof size", lambda value: value.update({"proof_kb": float("nan")})),
            ("proof equality", lambda value: value.update({"proof_byte_equal": False})),
            (
                "pool high-water",
                lambda value: value.update(
                    {"pool_used_high_gb": 7.0, "pool_reserved_high_gb": 6.0}
                ),
            ),
            (
                "missing SIMD equality",
                lambda value: value.update({"simd_reference_byte_equal": False}),
            ),
            (
                "SIMD reference not required",
                lambda value: value.update({"simd_reference_required": False}),
            ),
            (
                "SIMD comparison inapplicable",
                lambda value: value.update(
                    {"simd_reference_comparison_applicable": False}
                ),
            ),
            (
                "invalid SIMD reference digest",
                lambda value: value.update({"simd_reference_blake3": "not-a-digest"}),
            ),
            (
                "GPU and SIMD digest mismatch",
                lambda value: value.update({"gpu_proof_blake3": "6" * 64}),
            ),
            (
                "cached SIMD reference",
                lambda value: value.update({"simd_reference_fresh": False}),
            ),
            (
                "invalid SIMD reference duration",
                lambda value: value.update({"simd_reference_s": float("nan")}),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(record)
                mutate(candidate)
                self.assertTrue(validate_benchmark_measurement(candidate, **arguments))

    def test_gpu_telemetry_is_byte_bound_and_overlaps_proof_window(self) -> None:
        record = valid_measurement_record()
        start = 1_700_000_000_000_000_000
        finish = start + 4_000_000_000
        record["gpu_proof_loop_started_unix_ns"] = start
        record["gpu_proof_loop_finished_unix_ns"] = finish
        header = ",".join(GPU_TELEMETRY_COLUMNS)

        def row(timestamp: int) -> str:
            return ",".join(
                (
                    str(timestamp),
                    "98",
                    "42",
                    "12000",
                    "680.5",
                    "1980",
                    "2619",
                    "64",
                    "570.86.15",
                    "700",
                    "1980",
                    "2619",
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SN_PIE_2.telemetry.csv"

            def metadata_for(proof_window_sample_count: int) -> dict:
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                return {
                    "schema": GPU_TELEMETRY_SCHEMA,
                    "columns": list(GPU_TELEMETRY_COLUMNS),
                    "path": str(path),
                    "sha256": digest,
                    "remote_sha256": digest,
                    "size_bytes": len(payload),
                    "remote_size_bytes": len(payload),
                    "sample_count": len(payload.decode("utf-8").splitlines()) - 1,
                    "proof_window_sample_count": proof_window_sample_count,
                    "sampler_complete": True,
                    "sample_interval_ms": GPU_TELEMETRY_SAMPLE_INTERVAL_MS,
                    "max_gap_ns": GPU_TELEMETRY_MAX_GAP_NS,
                    "transport_equal": True,
                }

            timestamps = range(start - 1_000_000_000, finish + 1_000_000_001, 1_000_000_000)
            path.write_text(
                header + "\n" + "\n".join(row(timestamp) for timestamp in timestamps) + "\n",
                encoding="utf-8",
            )
            metadata = metadata_for(5)
            self.assertEqual(validate_gpu_telemetry_artifact(record, metadata), [])
            for name, mutate in (
                ("digest", lambda value: value.update({"sha256": "0" * 64})),
                ("size", lambda value: value.update({"size_bytes": 1})),
                (
                    "truncated remote digest",
                    lambda value: value.update({"remote_sha256": "abc"}),
                ),
                (
                    "truncated remote size",
                    lambda value: value.update({"remote_size_bytes": 1}),
                ),
                (
                    "transport mismatch",
                    lambda value: value.update({"transport_equal": False}),
                ),
                (
                    "schema",
                    lambda value: value.update({"schema": "stwo.gpu-telemetry.csv.v0"}),
                ),
                ("incomplete", lambda value: value.update({"sampler_complete": False})),
                (
                    "wrong interval",
                    lambda value: value.update({"sample_interval_ms": 1000}),
                ),
                (
                    "no window sample",
                    lambda value: value.update({"proof_window_sample_count": 0}),
                ),
            ):
                with self.subTest(name=name):
                    candidate = copy.deepcopy(metadata)
                    mutate(candidate)
                    self.assertTrue(validate_gpu_telemetry_artifact(record, candidate))

            sparse = (start - 1_000_000_000, start + 500_000_000,
                      start + 3_500_000_000, finish + 1_000_000_000)
            path.write_text(
                header + "\n" + "\n".join(row(timestamp) for timestamp in sparse) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(validate_gpu_telemetry_artifact(record, metadata_for(2)))

            path.write_text(
                f"{header}\n{row(start - 1_000_000_000)}\n"
                f"{start + 1_000_000_000},98,42\n{row(finish + 1_000_000_000)}\n",
                encoding="utf-8",
            )
            self.assertTrue(validate_gpu_telemetry_artifact(record, metadata_for(1)))

            outside = (start - 2_000_000_000, start - 1_000_000_000)
            path.write_text(
                header + "\n" + "\n".join(row(timestamp) for timestamp in outside) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(validate_gpu_telemetry_artifact(record, metadata_for(0)))

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

    def test_bench_loop_verifies_source_projection_after_sync_and_soundness(self) -> None:
        script = (Path(__file__).parent / "loop/bench_loop.sh").read_text(
            encoding="utf-8"
        )
        orchestration = script.split("# Orchestration", 1)[1]
        self.assertLess(
            orchestration.index("sync_repos"),
            orchestration.index("verify_remote_source_projection"),
        )
        self.assertLess(
            orchestration.index("verify_remote_source_projection"),
            orchestration.index("build_pod"),
        )
        soundness = script.split("run_cuda_soundness_gate() {", 1)[1].split(
            "# Synthetic run output", 1
        )[0]
        self.assertLess(
            soundness.index("verify_remote_source_projection"),
            soundness.index('seal_source_projection "$LOCAL_SOUNDNESS_GATE"'),
        )

    def test_bench_loop_drains_and_byte_checks_telemetry_transport(self) -> None:
        script = (Path(__file__).parent / "loop/bench_loop.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("pod_telemetry_stop=", script)
        self.assertIn("pod_telemetry_meta=", script)
        self.assertNotIn('kill "\\$telemetry_pid"', script)
        self.assertLess(
            script.index('sample="\\$(timeout --signal=KILL 2s nvidia-smi'),
            script.index('timestamp="\\$(date +%s%N)"'),
        )
        self.assertLess(
            script.index("printf 'stop\\n' > '${pod_telemetry_stop}'"),
            script.index('wait "\\$telemetry_pid"'),
        )
        self.assertLess(
            script.index('wait "\\$telemetry_pid"'),
            script.index("sha256sum '${pod_telemetry}'"),
        )
        self.assertIn(
            'if ! run_ssh "cat \'${pod_telemetry}\'" > "$LAST_TELEMETRY_PATH"; then',
            script,
        )

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

        reordered = valid_soundness_artifact("arena-graph")
        reordered["gates"][-2], reordered["gates"][-1] = (
            reordered["gates"][-1],
            reordered["gates"][-2],
        )
        self.assertTrue(validate_soundness_gate(reordered, "arena-graph"))

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
        arena_gates = gates_for_runtime_mode("arena-graph")
        self.assertEqual(len(arena_gates), 27)
        self.assertEqual(sum(required for _name, _command, required in arena_gates), 48)
        self.assertEqual(
            [name for name, _command, _required in arena_gates[-4:]],
            [
                "prepared_final_fri_and_pow_eager_capture_reference",
                "prepared_decommit_eager_capture_reference",
                "device_transcript_eager_capture_reference",
                STRICT_RESIDENT_GATE,
            ],
        )
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
                    self.assertIsNone(
                        re.search(
                            r'(?m)^#\[ignore(?:\s*=\s*"[^"]*"|\([^]]*\))?\]\s*$',
                            source,
                        ),
                        f"manifested native test is ignored: {path}",
                    )
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

    def test_runner_preserves_existing_libtest_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = mock.Mock(
                stdout=(
                    "test captured ... ok\n"
                    "test result: ok. 1 passed; 0 failed; 0 ignored; "
                    "0 measured; 0 filtered out\n"
                ),
                returncode=0,
            )
            with mock.patch(
                "run_cuda_soundness_gate.subprocess.run", return_value=completed
            ) as run:
                record = run_gate(
                    Path(directory),
                    "captured",
                    ("cargo", "test", "captured", "--", "--ignored"),
                    1,
                )

            self.assertTrue(record["passed"])
            command = run.call_args.args[0]
            self.assertEqual(command.count("--"), 1)
            self.assertEqual(command[-3:], ("--", "--ignored", "--nocapture"))

    def test_runner_rejects_a_counted_stub_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = mock.Mock(
                stdout=(
                    "test captured ... strict-AOT captured row: SKIPPED (stub build)\n"
                    "ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; "
                    "0 measured; 0 filtered out\n"
                ),
                returncode=0,
            )
            with mock.patch(
                "run_cuda_soundness_gate.subprocess.run", return_value=completed
            ):
                record = run_gate(
                    Path(directory), "captured", ("cargo", "test", "captured"), 1
                )

            self.assertFalse(record["passed"])
            self.assertTrue(record["stub_skip_detected"])

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
            aot_manifest=paths["aot_manifest"],
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
                        "--aot-manifest",
                        str(paths["aot_manifest"]),
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
            (
                "AOT union hash drift",
                lambda value: value["aot_coverage"].update(
                    {"required_occurrences_sha256": "0" * 64}
                ),
            ),
            (
                "AOT missing key",
                lambda value: value["aot_coverage"].update(
                    {"missing_occurrences": [{"cache_key": "2" * 16}]}
                ),
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
            "aot_manifest",
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

    def test_local_admission_rejects_self_consistent_aot_identity_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            admission, admission_path, paths = local_admission_fixture(Path(directory))
            manifest = json.loads(paths["aot_manifest"].read_text(encoding="utf-8"))
            manifest[0]["semantic_hash"] = "3" * 16
            paths["aot_manifest"].write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            admission["aot_coverage"]["manifest_sha256"] = sha256_file(
                paths["aot_manifest"]
            )
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

    def test_unsealed_runner_rejects_synced_identity_before_cache_environment(self) -> None:
        source = qualification_source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stwo = root / "stwo"
            stwo_cairo = root / "stwo-cairo"
            (stwo / "Cargo.toml").parent.mkdir(parents=True)
            (stwo / "Cargo.toml").touch()
            cairo_manifest = stwo_cairo / "stwo_cairo_prover/Cargo.toml"
            cairo_manifest.parent.mkdir(parents=True)
            cairo_manifest.touch()

            def fake_git(command, **kwargs):
                text_mode = kwargs.get("text", False)
                if tuple(command[1:3]) == ("rev-parse", "HEAD"):
                    stdout = source[
                        "stwo_cairo" if kwargs.get("cwd") == stwo_cairo else "stwo"
                    ]["head"]
                else:
                    stdout = "" if text_mode else b""
                return subprocess.CompletedProcess(command, 0, stdout, None)

            argv = [
                "run_cuda_soundness_gate.py",
                "--stwo",
                str(stwo),
                "--stwo-cairo",
                str(stwo_cairo),
                "--synced-stwo-head",
                source["stwo"]["head"],
                "--synced-stwo-worktree-hash",
                source["stwo"]["worktree_hash"],
                "--synced-stwo-cairo-head",
                source["stwo_cairo"]["head"],
                "--synced-stwo-cairo-worktree-hash",
                source["stwo_cairo"]["worktree_hash"],
                "--output",
                str(root / "soundness.json"),
            ]
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "run_cuda_soundness_gate.subprocess.run", side_effect=fake_git
            ), mock.patch("sys.argv", argv):
                with self.assertRaisesRegex(
                    SystemExit, "synced source identity is valid only for sealed execution"
                ):
                    run_soundness_main()
                self.assertTrue(
                    all(key not in os.environ for key in REFERENCE_CACHE_SOURCE_ENV)
                )

    def test_sealed_nogit_projection_binds_final_soundness_identity(self) -> None:
        source = qualification_source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stwo = root / "stwo"
            stwo_cairo = root / "stwo-cairo"
            (stwo / "Cargo.toml").parent.mkdir(parents=True)
            (stwo / "Cargo.toml").touch()
            cairo_manifest = stwo_cairo / "stwo_cairo_prover/Cargo.toml"
            cairo_manifest.parent.mkdir(parents=True)
            cairo_manifest.touch()
            strict_source = (
                cairo_manifest.parent
                / "crates/gpu-prover/tests/resident_parity_native.rs"
            )
            strict_source.parent.mkdir(parents=True)
            strict_source.write_text(
                'const STRICT_RESIDENT_FIXTURE: &str = "SN_PIE_2.zip";\n',
                encoding="utf-8",
            )

            binary_contents = b"sealed binary"
            binary_sha = hashlib.sha256(binary_contents).hexdigest()
            gpu_bench = root / f"sealed/gpu_bench.{binary_sha}"
            gpu_bench.parent.mkdir()
            gpu_bench.write_bytes(binary_contents)
            gate_input = root / "SN_PIE_2.zip"
            gate_input.write_bytes(b"input")
            output = root / "soundness.json"

            def gate_target(command) -> str:
                marker = "--test" if "--test" in command else "--lib"
                return command[command.index(marker) + 1]

            expected_by_target = {
                gate_target(command): required
                for _name, command, required in gates_for_runtime_mode("arena-graph")
            }

            def fake_run(command, **kwargs):
                if command[0] == "nvidia-smi":
                    value = (
                        "GPU-00000000-0000-4000-8000-000000000001\n"
                        if command[1] == "--query-gpu=uuid"
                        else "NVIDIA H100 80GB HBM3\n"
                    )
                    return subprocess.CompletedProcess(command, 0, value, "")
                if command[0] == "git":
                    stdout = "" if kwargs.get("text") else b""
                    return subprocess.CompletedProcess(command, 128, stdout, None)
                target = gate_target(command)
                required = expected_by_target[target]
                lines = []
                if target == "resident_parity_native":
                    lines.extend(
                        f"test {name} ... ok"
                        for name in STRICT_RESIDENT_REQUIRED_TESTS
                    )
                lines.append(f"test result: ok. {required} passed;")
                return subprocess.CompletedProcess(command, 0, "\n".join(lines), None)

            environment = {
                "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
                "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
                RETAINED_BUDGET_FLAG: "29469326848",
                **{flag: "1" for flag in QUALIFICATION_FLAGS},
            }
            argv = [
                "run_cuda_soundness_gate.py",
                "--stwo",
                str(stwo),
                "--stwo-cairo",
                str(stwo_cairo),
                "--runtime-mode",
                "arena-graph",
                "--synced-stwo-head",
                source["stwo"]["head"],
                "--synced-stwo-worktree-hash",
                source["stwo"]["worktree_hash"],
                "--synced-stwo-cairo-head",
                source["stwo_cairo"]["head"],
                "--synced-stwo-cairo-worktree-hash",
                source["stwo_cairo"]["worktree_hash"],
                "--pod-id",
                "pod-id",
                "--gpu-bench",
                str(gpu_bench),
                "--input-artifact",
                f"gate={gate_input}",
                "--output",
                str(output),
            ]
            real_read_text = Path.read_text

            def read_text(path, *args, **kwargs):
                if str(path) == "/proc/sys/kernel/random/boot_id":
                    return "00000000-0000-4000-8000-000000000001\n"
                return real_read_text(path, *args, **kwargs)

            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "run_cuda_soundness_gate.subprocess.run", side_effect=fake_run
            ), mock.patch("pathlib.Path.read_text", read_text), mock.patch(
                "sys.argv", argv
            ), mock.patch("builtins.print"):
                self.assertEqual(run_soundness_main(), 0)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            artifact["source_projection"] = {
                "method": "rsync-archive-checksum-dry-run-clean",
                "verified_after_soundness": True,
                "source": source,
            }
            self.assertEqual(
                validate_qualification_soundness_gate(
                    artifact, expected_source=source, expected_dry_run=False
                ),
                [],
            )
            for field in (
                "stwo_git_head",
                "stwo_worktree_hash",
                "stwo_cairo_git_head",
                "stwo_cairo_worktree_hash",
            ):
                with self.subTest(field=field):
                    candidate = copy.deepcopy(artifact)
                    candidate[field] = "f" * len(candidate[field])
                    self.assertTrue(
                        validate_qualification_soundness_gate(
                            candidate, expected_source=source, expected_dry_run=False
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
