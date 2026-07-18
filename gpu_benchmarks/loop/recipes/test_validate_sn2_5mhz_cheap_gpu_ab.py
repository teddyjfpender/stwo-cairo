#!/usr/bin/env python3
"""Synthetic schema checks for the cheap-GPU receipt validator."""

from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest

from validate_sn2_5mhz_cheap_gpu_ab import (
    QUOTIENT_CHECKS,
    RELATION_CHECKS,
    Checks,
    composition,
    quotient,
    relation,
    validate_run,
)


HEAD = "1" * 40


def loaded_resource(role: str = "") -> dict:
    symbols = {
        "baseline_staged_packed": "stwo_quotient_numerator_packed_single_write_kernel",
        "candidate_prepacked_prepare": "stwo_prepare_quotient_numerator_prepacked_terms_kernel",
        "candidate_prepacked_validate": "stwo_validate_quotient_numerator_prepacked_terms_kernel",
        "candidate_prepacked_hot": "stwo_quotient_numerator_prepacked_single_write_kernel",
    }
    return {
        "role": role,
        "symbol": symbols.get(role, ""),
        "launch_threads": 256,
        "dynamic_shared_bytes": 0,
        "abi_version": 1,
        "reserved": 0,
        "max_threads_per_block": 256,
        "registers_per_thread": 40,
        "binary_version": 86,
        "ptx_version": 86,
        "local_bytes": 0,
        "static_shared_bytes": 0,
    }


def relation_timing(speedup: float = 1.1) -> dict:
    stats = {
        "samples_ms": [1.0] * 30,
        "median_ms": 1.0,
        "p10_ms": 0.9,
        "p90_ms": 1.1,
    }
    return {
        "warmups": 5,
        "iterations": 30,
        "baseline": stats,
        "candidate": stats,
        "candidate_speedup": speedup,
    }


def relation_receipt() -> dict:
    return {
        "schema": "stwo.prepared-relation.same-binary-ab.v2",
        "passed": True,
        "git_commit": HEAD,
        "baseline_source_commit": "0016f4b5",
        "baseline": "pre_adaptive_columns_le_512_one_read",
        "candidate": "adaptive_tuple_width_gt_32_one_read",
        "ordering": "alternating_baseline_candidate_then_candidate_baseline",
        "percentiles": "sorted_linear_index",
        "output_blake3": "8" * 64,
        "static": {
            "source_identity": "9" * 64,
            "module_build_identity": "a" * 64,
            "target_sms": [86],
        },
        "selector_fixture": {
            "adaptive_lane_change_batches": 1,
            "shared_one_read_batches": 1,
        },
        "zero_denominator_fixture": {
            "batch_index": 0,
            "instance_index": 0,
            "column_index": 0,
            "source_index": 0,
            "tuple_class": "at_most_32_words",
            "columns": 8,
            "max_tuple_words": 16,
            "poisoned_use_tuple_words": 16,
            "baseline_lane": "one_read",
            "adaptive_lane": "suffix_recompute",
        },
        "checks": {name: True for name in RELATION_CHECKS},
        "loaded_functions": {
            "adaptive_relation_fused_kernel": loaded_resource(),
            "all_one_read_test_kernel": loaded_resource(),
        },
        "adaptive_resource_policy": {
            "mode": "first_characterization_report_only",
            "binary_version": 86,
            "ceiling_enforced": False,
        },
        "eager": relation_timing(),
        "captured": relation_timing(),
    }


def quotient_timing() -> dict:
    return {
        "warmups": 5,
        "iterations": 30,
        "median_ms": 1.0,
        "p10_ms": 0.9,
        "p90_ms": 1.1,
    }


def quotient_performance(mode: str, log: int, speedup: float = 1.1) -> dict:
    roles = (
        "baseline_staged_packed",
        "candidate_prepacked_prepare",
        "candidate_prepacked_validate",
        "candidate_prepacked_hot",
    )
    return {
        "name": f"staged-prepacked-quotient-{mode}-log{log}",
        "parameters": {
            "lifting_log_size": log,
            "groups": 4,
            "terms": 20,
            "prepacked_used_words": 157,
            "stream_count": 1,
            "source_buffer_sets": 1,
            "candidate_status_observations": 35,
        },
        "arena_bytes": {"shared_single_stream": 4096},
        "traffic_bytes": {
            "candidate_status_fence_d2h_bytes_per_replay_outside_kernel_time": 4,
            "candidate_status_fence_d2h_bytes_total_outside_kernel_time": 140,
        },
        "loaded_functions": [loaded_resource(role) for role in roles],
        "baseline": quotient_timing(),
        "candidate": quotient_timing(),
        "speedup": speedup,
    }


def quotient_receipt() -> dict:
    return {
        "schema": "stwo.replacement-stage4-native.v1",
        "passed": True,
        "git_commit": HEAD,
        "fixture_filter": "staged-prepacked-quotient",
        "requested_cuda_arch": "sm_86",
        "performance_requested": True,
        "performance_failure": None,
        "executable_blake3": "2" * 64,
        "source_blake3": "3" * 64,
        "fixtures": [
            {
                "name": "staged-prepacked-quotient-boundary",
                "cases": 4,
                "checks": {name: True for name in QUOTIENT_CHECKS},
            }
        ],
        "performance": [
            quotient_performance(mode, log)
            for log in (18, 20)
            for mode in ("eager", "captured")
        ],
    }


def composition_resource() -> dict:
    return {
        "max_threads_per_block": 128,
        "registers_per_thread": 64,
        "target_sm": 86,
        "binary_version": 86,
        "ptx_version": 86,
        "local_bytes": 0,
        "static_shared_bytes": 0,
        "dynamic_shared_bytes": 0,
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
        "source_identity": "4" * 64,
        "cubin_identity": "5" * 64,
        "component": 0,
        "kernel": 0,
        "cache_key": "0x0000000000000001",
        "semantic_hash": "0x0000000000000002",
    }


def composition_receipt() -> dict:
    timing = {
        "samples_per_arm": 6,
        "wave_source_jit_median_ms": 1.0,
        "installed_stripes_median_ms": 2.0,
        "observed_candidate_over_wave": 2.0,
    }
    return {
        "schema": "stwo.composition.stripe-direct-diagnostic.v1",
        "passed": True,
        "diagnostic_only": True,
        "promotion_credit": False,
        "real_sn2_coverage": False,
        "target_useful_mhz": 5.0,
        "stripe_count": 1,
        "correctness_checks": {
            "multiple_evaluation_domains": [7, 13, 24],
            "runtime_strict_rejections": 0,
            "direct_split_output": True,
            "shared_source_inputs": True,
            "single_process_device_arena_context_main_stream": True,
            "all_retained_bytes_equal_eager": True,
            "all_retained_bytes_equal_mutated_capture_replay": True,
            "mutated_replay_digest_changed": True,
            "installed_identity_and_resources_present": True,
        },
        "eager_retained_blake3": "6" * 64,
        "mutated_replay_retained_blake3": "7" * 64,
        "installed_functions": [composition_resource()],
        "timing": {
            "order": "ABBA",
            "same_exec_context": True,
            "promotion_credit": False,
            "eager": timing,
            "captured": timing,
        },
    }


class ReceiptValidationTests(unittest.TestCase):
    def test_relation_accepts_exact_shape_and_rejects_poison_policy_and_speedup(self) -> None:
        checks = Checks()
        self.assertEqual(set(relation(checks, relation_receipt(), HEAD)), {"eager", "captured"})
        self.assertEqual(checks.errors, [])

        value = relation_receipt()
        value["zero_denominator_fixture"]["batch_index"] = 1
        value["adaptive_resource_policy"]["mode"] = "qualified_sm_90_envelope"
        value["eager"]["candidate_speedup"] = 0.9
        checks = Checks()
        admitted = relation(checks, value, HEAD)
        self.assertGreaterEqual(len(checks.errors), 3)
        self.assertNotIn("eager", admitted)

    def test_quotient_rejects_failure_missing_cell_and_nonpositive_speedup(self) -> None:
        checks = Checks()
        self.assertEqual(len(quotient(checks, quotient_receipt(), HEAD)), 4)
        self.assertEqual(checks.errors, [])

        value = quotient_receipt()
        value["performance_failure"] = "kernel launch failed"
        value["performance"].pop()
        value["performance"][0]["speedup"] = 0.9
        checks = Checks()
        admitted = quotient(checks, value, HEAD)
        self.assertGreaterEqual(len(checks.errors), 3)
        self.assertEqual(len(admitted), 2)

    def test_composition_is_diagnostic_and_does_not_require_speedup(self) -> None:
        checks = Checks()
        self.assertEqual(set(composition(checks, composition_receipt())), {"eager", "captured"})
        self.assertEqual(checks.errors, [])

        value = copy.deepcopy(composition_receipt())
        value["diagnostic_only"] = False
        value["promotion_credit"] = True
        value["correctness_checks"]["runtime_strict_rejections"] = 1
        value["installed_functions"][0]["registers_per_thread"] = 129
        value["installed_functions"][0]["local_bytes"] = 4
        checks = Checks()
        composition(checks, value)
        self.assertGreaterEqual(len(checks.errors), 5)

    def test_raw_failure_is_aggregated_and_never_grants_promotion_credit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            for name in ("relation", "composition", "environment"):
                (run / f"{name}.raw.rc").write_text("0\n", encoding="ascii")
            (run / "quotient.raw.rc").write_text("17\n", encoding="ascii")
            record = validate_run(run)
        self.assertFalse(record["passed"])
        self.assertTrue(any("quotient: raw command failed" in error for error in record["errors"]))
        self.assertEqual(set(record["promotion_credit"].values()), {False})


if __name__ == "__main__":
    unittest.main()
