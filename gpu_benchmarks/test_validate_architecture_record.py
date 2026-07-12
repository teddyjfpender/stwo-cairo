#!/usr/bin/env python3

import re
import unittest
from pathlib import Path

from run_cuda_soundness_gate import STRICT_RESIDENT_REQUIRED_TESTS, gates_for_runtime_mode
from validate_architecture_record import (
    ARCHITECTURE,
    SOUNDNESS_COMMANDS,
    SOUNDNESS_GATES,
    STAGES,
    validate_record,
    validate_soundness_gate,
)


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


def valid_soundness_artifact(runtime_mode: str = "detached-eager") -> dict:
    gates = gates_for_runtime_mode(runtime_mode)
    artifact = {
        "schema": "stwo.cuda.soundness-gate.v2",
        "stwo_git_head": "1" * 40,
        "stwo_cairo_git_head": "2" * 40,
        "stwo_worktree_hash": "3" * 64,
        "stwo_cairo_worktree_hash": "4" * 64,
        "runtime_mode": runtime_mode,
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

    def test_soundness_manifest_covers_cfg_native_targets_with_exact_counts(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        test_roots = (
            (workspace / "stwo/crates/backend-cuda/tests", "stwo-backend-cuda"),
            (
                workspace / "stwo-cairo/stwo_cairo_prover/crates/gpu-prover/tests",
                "stwo-cairo-gpu-prover",
            ),
            (
                workspace / "stwo-cairo/stwo_cairo_prover/crates/prover/tests",
                "stwo-cairo-prover",
            ),
        )
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

    def test_reference_cache_tests_cannot_be_absorbed_by_strict_resident_target(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        tests = workspace / "stwo-cairo/stwo_cairo_prover/crates/gpu-prover/tests"
        strict = (tests / "resident_parity_native.rs").read_text(encoding="utf-8")
        common = (tests / "common/reference_cache.rs").read_text(encoding="utf-8")
        host = (tests / "reference_cache_host.rs").read_text(encoding="utf-8")
        self.assertIn('#[path = "common/reference_cache.rs"]', strict)
        self.assertIsNone(re.search(r"#\[(?:test|cfg\(test\))\]", common))
        self.assertEqual(len(re.findall(r"(?m)^#\[test\]\s*$", host)), 5)


if __name__ == "__main__":
    unittest.main()
