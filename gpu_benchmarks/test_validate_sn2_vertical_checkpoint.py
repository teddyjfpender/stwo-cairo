from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from gpu_benchmarks.validate_sn2_vertical_checkpoint import (
        PCS_FIELDS,
        CheckpointError,
        validate_checkpoint,
    )
except ModuleNotFoundError:
    from validate_sn2_vertical_checkpoint import (
        PCS_FIELDS,
        CheckpointError,
        validate_checkpoint,
    )


def valid_record() -> dict[str, object]:
    record: dict[str, object] = {
        "program": "SN_PIE_2.zip", "backend": "cuda", "engine": "gpu-native",
        "n": 1, "cycle_count": 7_977_397, "pie_n_steps": 7_706_864,
        "reps": 6, "verified_reps": 6, "warm_sample_count": 5,
        "gpu_resident_backend_requested": "replacement-v1",
        "gpu_resident_backend": "replacement-v1",
        "gpu_native_architecture_required": True,
        "gpu_pcs_required_runtime_mode": "arena-graph",
        "gpu_graph_a_setup_gate_passed": True,
        "benchmark_diagnostic_mode": True,
        "benchmark_diagnostic_reason": "compiled-composition-vertical-checkpoint",
        "benchmark_graph_submit_capture_mode": False,
        "compiled_composition_vertical_checkpoint": True,
        "compiled_composition_vertical_checkpoint_gate_passed": True,
        "compiled_composition_vertical_checkpoint_timing_scope":
            "public-prove-call-warm-end-to-end",
        "performance_claim_class": "indicative-non-formal",
        "performance_claim_admissible": False,
        "performance_measurement_available": True,
        "gpu_pcs_architecture_gate_applicable": False,
        "gpu_native_architecture_gate_passed": None,
        "fleet_pow_enabled": False,
        "proof_byte_equal_required": True, "proof_comparison_applicable": True,
        "proof_byte_equal": True, "simd_reference_required": True,
        "simd_reference_comparison_applicable": True,
        "simd_reference_fresh": True, "simd_reference_byte_equal": True,
        "gpu_proof_blake3": "ab" * 32, "simd_reference_blake3": "ab" * 32,
        "proof_mutation_required": True,
        "proof_mutation_kind": "interaction_claim.memory_id_to_big.claimed_sum_plus_one",
        "proof_mutation_rejected": True,
        "proof_mutation_error_class": "invalid_logup_sum",
        "gpu_aot_provenance_gate_passed": True,
        "gpu_aot_misses": 0, "gpu_aot_runtime_loads": 0,
        "gpu_aot_runtime_cache_hits": 0, "gpu_aot_strict_rejections": 0,
        "gpu_aot_loads": 1, "gpu_aot_cache_hits": 5,
        "gpu_aot_manifest_hash": 7, "gpu_policy_kernel_manifest_hash": 7,
        "gpu_composition_split_launch_mode": "fused-first-forward",
        "gpu_composition_split_executed_kernel_launches": 5,
        "gpu_composition_split_executed_logical_bytes": 4_026_531_840,
        "prove_s_warm_median": 2.0, "prove_s_warm_p95": 2.1,
        "useful_mhz_median": 3.8, "useful_mhz_at_warm_p95": 3.6,
        "verify_ms": 1.0,
        "gpu_graph_replay_intervals": None,
        "gpu_graph_replay_interval_total_ns": None,
        "gpu_graph_replay_interval_count": None,
        "gpu_graph_replay_semantic_count": None,
        "gpu_graph_replay_timing_scope": None,
    }
    record.update(dict.fromkeys(PCS_FIELDS))
    return record


class VerticalCheckpointValidatorTests(unittest.TestCase):
    def run_validation(
        self, record: dict[str, object], receipt_mutator=None
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.txt"
            proof = root / "proof.bin"
            hardware = root / "hardware.json"
            source = root / "source.json"
            pie = root / "SN_PIE_2.zip"
            bootloader = root / "simple_bootloader_compiled.json"
            input_manifest = root / "SHA256SUMS"
            aot_manifest = root / "aot_manifest.json"
            stdout.write_text(json.dumps(record) + "\n", encoding="utf-8")
            proof.write_bytes(b"proof")
            pie.write_bytes(b"sn2")
            bootloader.write_bytes(b"bootloader")
            input_hashes = {
                "SN_PIE_2.zip": hashlib.sha256(pie.read_bytes()).hexdigest(),
                "simple_bootloader_compiled.json": hashlib.sha256(
                    bootloader.read_bytes()
                ).hexdigest(),
            }
            input_manifest.write_text(
                "".join(f"{digest}  {name}\n" for name, digest in input_hashes.items()),
                encoding="utf-8",
            )
            entries = [
                {"cache_key": "01" * 8, "kind": "witness", "label": "witness"},
                {"cache_key": "02" * 8, "kind": "constraint", "label": "ordinary"},
                {"cache_key": "03" * 8, "kind": "constraint", "label": "wave_log_21"},
            ]
            aot_manifest.write_text(
                json.dumps(entries) + "\n", encoding="utf-8"
            )
            expected_source = {
                "stwo": {"head": "ab" * 20, "worktree_sha256": "12" * 32},
                "stwo_cairo": {"head": "cd" * 20, "worktree_sha256": "34" * 32},
            }
            source_receipt = {
                "schema": "stwo.replacement-v1-sn2.source-input-identity.v1",
                "source_policy": "iteration",
                "source": copy.deepcopy(expected_source),
                "inputs": input_hashes,
                "aot_pack": {
                    "manifest_sha256": hashlib.sha256(aot_manifest.read_bytes()).hexdigest(),
                    "total": 3,
                    "witness": 1,
                    "ordinary_constraint": 1,
                    "composition_wave": 1,
                },
            }
            if receipt_mutator is not None:
                receipt_mutator(source_receipt)
            source.write_text(json.dumps(source_receipt), encoding="utf-8")
            hardware.write_text(json.dumps({
                "schema": "stwo.replacement-v1-sn2.hardware-identity.v2",
                "name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-test",
                "compute_capability": "9.0", "memory_mib": 81_559,
            }), encoding="utf-8")
            return validate_checkpoint(
                stdout,
                proof,
                hardware,
                source,
                pie,
                bootloader,
                input_manifest,
                aot_manifest,
                expected_source,
            )

    def test_accepts_only_explicit_non_formal_vertical(self) -> None:
        result = self.run_validation(valid_record())
        self.assertEqual(result["schema"], "stwo.sn2-compiled-vertical-checkpoint.v2")
        self.assertEqual(result["verdict"], "PASS")
        self.assertFalse(result["formal_promotion_eligible"])
        self.assertEqual(result["composition_split_launch_mode"], "fused-first-forward")
        self.assertEqual(result["composition_split_executed_kernel_launches"], 5)
        self.assertEqual(
            result["composition_split_executed_logical_bytes"], 4_026_531_840
        )
        for field, bad in (
            ("performance_claim_class", "formal"),
            ("performance_claim_admissible", True),
            ("benchmark_diagnostic_reason", None),
            ("fleet_pow_enabled", True),
            ("simd_reference_fresh", False),
            ("proof_byte_equal", False),
            ("gpu_graph_a_setup_gate_passed", False),
            ("gpu_aot_misses", 1),
        ):
            with self.subTest(field=field):
                record = copy.deepcopy(valid_record())
                record[field] = bad
                with self.assertRaises(CheckpointError):
                    self.run_validation(record)

    def test_requires_exact_fused_log24_split_receipt(self) -> None:
        for field, bad in (
            ("gpu_composition_split_launch_mode", "terminal-fallback"),
            ("gpu_composition_split_executed_kernel_launches", 6),
            ("gpu_composition_split_executed_logical_bytes", 5_100_273_664),
        ):
            with self.subTest(field=field):
                record = valid_record()
                record[field] = bad
                with self.assertRaises(CheckpointError):
                    self.run_validation(record)

    def test_rejects_any_pcs_telemetry(self) -> None:
        for field in PCS_FIELDS:
            with self.subTest(field=field):
                record = valid_record()
                record[field] = "unexpected"
                with self.assertRaises(CheckpointError):
                    self.run_validation(record)

    def test_rejects_mutated_source_input_and_aot_receipts(self) -> None:
        mutations = {
            "source": lambda receipt: receipt["source"]["stwo"].update(
                head="ef" * 20
            ),
            "input": lambda receipt: receipt["inputs"].update(
                {"SN_PIE_2.zip": "00" * 32}
            ),
            "aot_hash": lambda receipt: receipt["aot_pack"].update(
                manifest_sha256="00" * 32
            ),
            "aot_shape": lambda receipt: receipt["aot_pack"].update(total=4),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises(CheckpointError):
                self.run_validation(valid_record(), mutate)


if __name__ == "__main__":
    unittest.main()
