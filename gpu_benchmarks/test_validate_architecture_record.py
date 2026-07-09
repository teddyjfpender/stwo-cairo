#!/usr/bin/env python3

import unittest

from validate_architecture_record import ARCHITECTURE, STAGES, validate_record


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
    }


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


if __name__ == "__main__":
    unittest.main()
