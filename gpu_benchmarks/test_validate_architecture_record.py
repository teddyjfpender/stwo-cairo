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

    def test_arena_graph_requirement_rejects_detached_record(self) -> None:
        errors = validate_record(valid_record(), "arena-graph")
        self.assertTrue(any("gpu_pcs_runtime_mode" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
