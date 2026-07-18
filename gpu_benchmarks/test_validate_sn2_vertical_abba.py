from __future__ import annotations

import copy
import unittest
from pathlib import Path

try:
    from gpu_benchmarks.validate_sn2_vertical_abba import (
        CheckpointError,
        _common_arguments,
        aggregate_abba,
    )
except ModuleNotFoundError:
    from validate_sn2_vertical_abba import (
        CheckpointError,
        _common_arguments,
        aggregate_abba,
    )


def record(variant: str, start: int, samples: list[float]) -> dict[str, object]:
    median = sorted(samples)[len(samples) // 2]
    value: dict[str, object] = {
        "program": "SN_PIE_2.zip", "backend": "cuda", "engine": "gpu-native",
        "n": 1, "cycle_count": 7_977_397, "pie_n_steps": 7_706_864,
        "security_bits": 96, "n_queries": 70, "pow_bits": 26, "fold_step": 3,
        "reps": 6, "verified_reps": 6, "warm_sample_count": 5,
        "gpu_resident_backend_requested": "replacement-v1",
        "gpu_resident_backend": "replacement-v1",
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": "arena-graph",
        "gpu_graph_a_setup_gate_passed": True, "fleet_pow_enabled": False,
        "proof_byte_equal_required": True, "proof_comparison_applicable": True,
        "proof_byte_equal": True, "simd_reference_required": True,
        "simd_reference_comparison_applicable": True,
        "simd_reference_fresh": True, "simd_reference_byte_equal": True,
        "proof_mutation_required": True,
        "proof_mutation_kind":
            "interaction_claim.memory_id_to_big.claimed_sum_plus_one",
        "proof_mutation_rejected": True,
        "proof_mutation_error_class": "invalid_logup_sum",
        "gpu_aot_provenance_gate_passed": True, "gpu_aot_misses": 0,
        "gpu_aot_runtime_loads": 0, "gpu_aot_runtime_cache_hits": 0,
        "gpu_aot_strict_rejections": 0, "gpu_setup_base_migration_copies": 0,
        "gpu_setup_lookup_host_copies": 0, "gpu_setup_legacy_witness_fallbacks": 0,
        "gpu_aot_manifest_hash": 7, "gpu_policy_kernel_manifest_hash": 7,
        "performance_measurement_available": True,
        "throughput_distribution_applicable": True,
        "gpu_proof_blake3": "ab" * 32, "simd_reference_blake3": "ab" * 32,
        "prove_s_warm_samples_raw": samples,
        "prove_s_warm_median": round(median, 3),
        "useful_mhz_median": round(7_706_864 / median / 1e6, 3),
        "gpu_proof_loop_started_unix_ns": start,
        "compiled_composition_vertical_checkpoint": variant == "new",
    }
    if variant == "old":
        value.update({
            "compiled_composition_vertical_checkpoint_gate_passed": None,
            "compiled_composition_vertical_checkpoint_timing_scope": None,
            "benchmark_diagnostic_mode": False, "benchmark_diagnostic_reason": None,
            "benchmark_graph_submit_capture_mode": False,
            "performance_claim_class": None, "performance_claim_admissible": True,
            "gpu_native_architecture_gate_passed": True,
            "gpu_pcs_architecture_gate_applicable": True,
            "gpu_pcs_driver_architecture": "cuda-typed-pcs-driver-v1",
            "gpu_pcs_runtime_mode": "ArenaGraph",
            "gpu_pcs_driver_complete": True,
            "gpu_pcs_batched_tree_decommit": True,
            "gpu_pcs_stage_started": {name: 1 for name in (
                "Assembly", "FriCommitAndFold", "FriQueryAndDecommit",
                "OodsEvaluation", "ProofOfWork", "QuotientAndCompaction",
                "TreeDecommit")},
            "gpu_pcs_stage_finished": {name: True for name in (
                "Assembly", "FriCommitAndFold", "FriQueryAndDecommit",
                "OodsEvaluation", "ProofOfWork", "QuotientAndCompaction",
                "TreeDecommit")},
            "gpu_host_plan_cache_materialization": "reused",
            "gpu_shape_executable_materialization": "reused",
            "gpu_prepared_runtime_materialization": "reused",
            "gpu_prepared_runtime_capture_ready_at_entry": True,
            "gpu_prepared_runtime_capture_ready_at_exit": True,
            "gpu_statement_refresh_present": True,
            "gpu_workspace_materialization": "reused",
        })
        for field, expected in {
            "gpu_host_plan_cache_hits": 5, "gpu_host_plan_cache_misses": 1,
            "gpu_host_plan_cache_compilations": 1, "gpu_host_plan_cache_evictions": 0,
            "gpu_host_plan_cache_collisions": 0, "gpu_shape_executable_cache_hits": 5,
            "gpu_shape_executable_cache_misses": 1,
            "gpu_shape_executable_cache_compilations": 1,
            "gpu_shape_executable_cache_source_generation_passes": 1,
            "gpu_shape_executable_cache_binding_recipe_compilations": 1,
            "gpu_shape_executable_cache_capacity_rejections": 0,
            "gpu_execution_tables_ingest_descriptor_h2d_copies": 0,
            "gpu_composition_direct_split_graphs": 1,
            "gpu_composition_precomputed_compact_commitments": 1,
            "gpu_composition_coefficient_commit_paths": 0,
            "gpu_composition_split_fused_d2d_nodes": 0,
            "gpu_hot_allocations": 0, "gpu_hot_allocation_bytes": 0,
            "gpu_hot_frees": 0, "gpu_hot_d2d_bytes": 0,
            "gpu_hot_memset_bytes": 0, "gpu_hot_fill_words": 0,
            "gpu_hot_capture_begins": 0, "gpu_hot_capture_finishes": 0,
            "gpu_hot_capture_aborts": 0, "gpu_hot_lane_forks": 0,
            "gpu_hot_lane_joins": 0,
        }.items():
            value[field] = expected
    return value


class VerticalAbbaTests(unittest.TestCase):
    def evidence(self):
        pie = Path("/sealed/SN_PIE_2.zip")
        records = [
            record("old", 1, [2.0] * 5),
            record("new", 2, [1.0] * 5),
            record("new", 3, [1.2] * 5),
            record("old", 4, [2.2] * 5),
        ]
        common = _common_arguments(pie, 6)
        arguments = [
            common,
            common + ["--compiled-composition-vertical-checkpoint"],
            common + ["--compiled-composition-vertical-checkpoint"],
            common,
        ]
        return pie, records, [b"proof"] * 4, arguments

    def test_aggregates_ten_warm_samples_per_variant(self) -> None:
        pie, records, proofs, arguments = self.evidence()
        result = aggregate_abba(records, proofs, arguments, pie)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["order"], ["old", "new", "new", "old"])
        self.assertEqual(result["warm_samples_per_variant"], 10)
        self.assertEqual(result["new_over_old_speedup"], 1.909091)

    def test_fails_closed_on_arguments_proofs_and_soundness(self) -> None:
        for mutation in ("arguments", "proof", "simd", "mutation"):
            with self.subTest(mutation=mutation):
                pie, records, proofs, arguments = self.evidence()
                if mutation == "arguments":
                    arguments[0] = arguments[0] + [
                        "--compiled-composition-vertical-checkpoint"
                    ]
                elif mutation == "proof":
                    proofs[3] = b"different"
                elif mutation == "simd":
                    records[2]["simd_reference_byte_equal"] = False
                else:
                    records[1]["proof_mutation_rejected"] = False
                with self.assertRaises(CheckpointError):
                    aggregate_abba(records, proofs, arguments, pie)


if __name__ == "__main__":
    unittest.main()
