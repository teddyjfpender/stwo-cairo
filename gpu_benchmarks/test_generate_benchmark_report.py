#!/usr/bin/env python3
"""Mutation tests for generate_benchmark_report.py."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    from gpu_benchmarks.generate_benchmark_report import (
        GPU_TELEMETRY_COLUMNS,
        ReportError,
        generate_report,
    )
except ModuleNotFoundError:  # Direct execution from gpu_benchmarks/.
    from generate_benchmark_report import GPU_TELEMETRY_COLUMNS, ReportError, generate_report


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def round3(value: float) -> float:
    return math.floor(value * 1000 + 0.5) / 1000


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    rank = q * (len(ordered) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


class Fixture:
    def __init__(self, root: Path, *, dry_run: bool):
        self.root = root
        self.dry_run = dry_run
        self.source = {
            "stwo": {"head": "1" * 40, "worktree_hash": "2" * 64},
            "stwo_cairo": {"head": "3" * 40, "worktree_hash": "4" * 64},
        }
        self.gpu = "NVIDIA H100 80GB HBM3"
        self.raw_hashes = {
            **{f"SN_PIE_{number}.zip": f"{number}" * 64 for number in range(1, 5)},
            "simple_bootloader_compiled.json": "5" * 64,
        }
        self.adapted_hashes: dict[str, str] = {}
        adapted = root / "adapted_inputs"
        adapted.mkdir()
        for number in range(1, 5):
            path = adapted / f"SN_PIE_{number}.adapted.bin"
            path.write_bytes(f"adapted-{number}".encode())
            self.adapted_hashes[path.name] = sha(path)

        self.raw_manifest = root / "raw.sha256"
        self.raw_manifest.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in self.raw_hashes.items())
        )
        self.adapted_manifest = root / "adapted.sha256"
        self.adapted_manifest.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in self.adapted_hashes.items())
        )
        self.pinned_manifest = root / "pinned.sha256"
        self.pinned_manifest.write_text(self.adapted_manifest.read_text())

        ceiling = 76 * 1024**3
        self.preflight_summaries = {}
        for label, number, arena in [
            *( (f"universal_SN{n}", n, 20 * 1024**3 + n) for n in range(1, 5) ),
            ("headline_SN2", 2, 30 * 1024**3),
            ("flags_off_SN2", 2, 40 * 1024**3),
        ]:
            adapted_path = adapted / f"SN_PIE_{number}.adapted.bin"
            policy = {"profile": label}
            artifact = root / f"preflight_{label}.json"
            artifact.write_text(json.dumps({
                "pass": True,
                "vram_fit": True,
                "vram_budget_bytes": ceiling,
                "arena": {"total_bytes": arena},
                "runtime_policy": policy,
            }))
            self.preflight_summaries[label] = {
                "artifact": str(artifact),
                "artifact_sha256": sha(artifact),
                "adapted_input": str(adapted_path),
                "adapted_input_sha256": sha(adapted_path),
                "arena_bytes": arena,
                "arena_gib": arena / 1024**3,
                "runtime_policy": policy,
            }

        remote_inputs = {
            "gate": {"path": "/remote/SN_PIE_2.zip", "sha256": self.raw_hashes["SN_PIE_2.zip"]},
            "bootloader": {"path": "/remote/bootloader.json", "sha256": self.raw_hashes["simple_bootloader_compiled.json"]},
            **{
                f"SN_PIE_{number}": {
                    "path": f"/remote/SN_PIE_{number}.zip",
                    "sha256": self.raw_hashes[f"SN_PIE_{number}.zip"],
                }
                for number in range(1, 5)
            },
        }
        self.target = {
            "schema": "stwo.remote-execution-target.v1",
            "pod_id": "pod-qualification-test",
            "boot_id": "boot-qualification-test",
            "gpu_uuid": "GPU-qualification-test",
            "gpu_name": self.gpu,
            "gpu_bench": {"path": "/remote/gpu_bench", "sha256": "a" * 64},
            "inputs": remote_inputs,
        }
        self.projection = {
            "method": "rsync-archive-checksum-dry-run-clean",
            "verified_after_soundness": True,
            "source": self.source,
        }
        self.soundness = root / "soundness.json"
        self.soundness.write_text(json.dumps({
            "schema": "stwo.cuda.soundness-gate.v3",
            "passed": True,
            "runtime_mode": "arena-graph",
            "dry_run": dry_run,
            "synced_source": {
                "stwo": self.source["stwo"],
                "stwo_cairo": self.source["stwo_cairo"],
                "transport": "rsync-archive-checksum",
            },
            "execution_target": self.target,
            "execution_target_sha256": canonical_sha(self.target),
            "execution_target_postcheck": True,
            "source_projection": self.projection,
        }))
        self.local_admission = root / "local_admission.json"
        self.local_admission.write_text(json.dumps({
            "schema": "stwo.local-preflight-admission.v1",
            "passed": True,
            "dry_run": dry_run,
            "runtime_mode": "arena-graph",
            "source": self.source,
            "profiles": {
                "flags_off": "",
                "universal_sn1_sn4": "STWO_UNIVERSAL=1",
                "sn2_headline": "STWO_HEADLINE=1",
            },
            "preflight_ceiling_bytes": ceiling,
            "preflight_artifact_sha256": {
                Path(summary["artifact"]).name: summary["artifact_sha256"]
                for summary in self.preflight_summaries.values()
            },
            "adapted_input_manifest_sha256": sha(self.adapted_manifest),
            "adapter_reproduction": {
                "byte_equal": True,
                "gpu_bench_binary_sha256": "c" * 64,
                "raw_input_manifest_sha256": sha(self.raw_manifest),
                "bootloader_sha256": "5" * 64,
                "pinned_adapted_manifest_sha256": sha(self.pinned_manifest),
            },
        }))
        self.bench_ledger = root / "bench.jsonl"
        self.ab_ledger = root / "sn2_ab.jsonl"
        self.ab_ledger.write_text('{"status":"ok"}\n')

        fixed = {
            f"SN_PIE_{number}": self.measurement(
                f"SN_PIE_{number}.zip", steps=12_000_000 + number,
                cycles=15_000_000 + number, samples=[1.0, 2.0, 3.0, 4.0, 5.0],
                simd=True,
            )
            for number in range(1, 5)
        }
        self.telemetry = {
            name: self.make_telemetry(name, fixed[name], number)
            for number, name in enumerate(fixed, 1)
        }
        baseline = self.measurement(
            "SN_PIE_2.zip", steps=12_000_002, cycles=15_000_002,
            samples=[3.0, 4.0, 5.0, 6.0, 7.0],
        )
        headline = self.measurement(
            "SN_PIE_2.zip", steps=12_000_002, cycles=15_000_002,
            samples=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        ab_hash = "b" * 64
        proof_hashes = {name: str(number) * 64 for number, name in enumerate(fixed, 1)}
        proof_hashes["SN_PIE_2"] = ab_hash
        self.bench_entries = [
            {
                "run_name": name,
                "status": "ok",
                "record": copy.deepcopy(fixed[name]),
                "phase_totals": [
                    {
                        "rep": rep,
                        "phase_totals": {
                            "witness_generation": {
                                "count": 1,
                                "total_ms": 1000 + number * 10 + rep,
                            },
                            "fri": {
                                "count": 2,
                                "total_ms": 500 + number * 5 + rep * 2,
                            },
                        },
                    }
                    for rep in range(6)
                ],
            }
            for number, name in enumerate(fixed, 1)
        ]
        self.write_bench_ledger()
        self.ncu_report = root / "sn2_headline.ncu-rep"
        self.ncu_import = root / "sn2_headline.ncu-rep.import.csv"
        if dry_run:
            self.ncu_report.write_bytes(
                b"STWO synthetic Nsight Compute report v1\nfixture\n"
            )
            self.ncu_import.write_text(
                'STWO synthetic ncu import validation v1\n'
                '"ID","Kernel Name","Metric Name"\n'
                '"1","relation_fused","synthetic"\n'
            )
        else:
            self.ncu_report.write_bytes(b"NVP\0" + (4).to_bytes(4, "little") + b"headtail")
            self.ncu_import.write_text(
                '"ID","Kernel Name","Metric Name"\n'
                '"1","relation_fused","dram__bytes"\n'
            )
        self.ncu_profile = {
            "schema": "stwo.ncu-profile.v1",
            "path": str(self.ncu_report),
            "sha256": sha(self.ncu_report),
            "bytes": self.ncu_report.stat().st_size,
            "kernel_regex": "relation_fused",
            "launch_count": 10,
            "set": "full",
            "ncu_version": "NVIDIA (R) Nsight Compute Command Line Profiler",
            "synthetic": dry_run,
            "remote_sha256": sha(self.ncu_report),
            "remote_bytes": self.ncu_report.stat().st_size,
            "remote_import_validated": True,
            "import_output_path": str(self.ncu_import),
            "import_output_sha256": sha(self.ncu_import),
            "import_output_bytes": self.ncu_import.stat().st_size,
            "remote_import_output_sha256": sha(self.ncu_import),
            "remote_import_output_bytes": self.ncu_import.stat().st_size,
            "profiled_kernel_rows": 1,
            "profiled_proof_sha256": ab_hash,
        }
        ratio = headline["useful_mhz_median"] / baseline["useful_mhz_median"]
        self.qualification = {
            "schema": "stwo.qualification-round.v4",
            "status": "dry_run" if dry_run else "passed",
            "dry_run": dry_run,
            "performance_admissible": not dry_run,
            "runtime_mode": "arena-graph",
            "gpu": self.gpu,
            "source": self.source,
            "profiles": {
                "universal_sn1_sn4": {
                    "env": "STWO_UNIVERSAL=1",
                    "preflight_ceiling_bytes": ceiling,
                    "preflights": {
                        f"SN_PIE_{number}": self.preflight_summaries[f"universal_SN{number}"]
                        for number in range(1, 5)
                    },
                },
                "sn2_headline": {
                    "env": "STWO_HEADLINE=1",
                    "preflight_ceiling_bytes": ceiling,
                    "preflight": self.preflight_summaries["headline_SN2"],
                },
            },
            "flags_off_preflight": self.preflight_summaries["flags_off_SN2"],
            "local_admission": {"path": str(self.local_admission), "sha256": sha(self.local_admission)},
            "soundness": {"path": str(self.soundness), "sha256": sha(self.soundness)},
            "remote_execution_target": {
                "target": self.target,
                "sha256": canonical_sha(self.target),
                "postcheck": True,
                "source_projection": self.projection,
                "remote_quiescence_passed": True,
            },
            "inputs": {
                "manifest": str(self.raw_manifest),
                "manifest_sha256": sha(self.raw_manifest),
                "files": self.raw_hashes,
                "adapted_manifest": str(self.adapted_manifest),
                "adapted_manifest_sha256": sha(self.adapted_manifest),
                "pinned_adapted_manifest": str(self.pinned_manifest),
                "pinned_adapted_manifest_sha256": sha(self.pinned_manifest),
                "adapted_files": self.adapted_hashes,
                "gate": "SN_PIE_2.zip",
            },
            "proof_sha256": {
                "fixed_pies": proof_hashes,
                "ab": {"sn2_headline": {"flags_off": ab_hash, "headline": ab_hash}},
            },
            "benchmarks": fixed,
            "gpu_telemetry": self.telemetry,
            "profiling": {
                "ncu": {
                    "required": True,
                    "requested": True,
                    "attempted": True,
                    "status": "validated",
                    "profile": self.ncu_profile,
                }
            },
            "comparisons": {
                "sn2_flags_off_vs_headline": {
                    "status": "ok",
                    "reps": 6,
                    "baseline": {
                        "proof_sha256": ab_hash,
                        "useful_mhz_median": baseline["useful_mhz_median"],
                        "record": baseline,
                    },
                    "flagged": {
                        "proof_sha256": ab_hash,
                        "useful_mhz_median": headline["useful_mhz_median"],
                        "record": headline,
                    },
                    "useful_mhz_ratio": ratio,
                }
            },
            "ledgers": {
                "bench": {"path": str(self.bench_ledger), "sha256": sha(self.bench_ledger)},
                "ab": {"path": str(self.ab_ledger), "sha256": sha(self.ab_ledger)},
            },
        }
        self.write()

    def make_telemetry(self, name: str, record: dict, number: int) -> dict:
        path = self.root / f"{name}.gpu-telemetry.csv"
        start = record["gpu_proof_loop_started_unix_ns"]
        finish = record["gpu_proof_loop_finished_unix_ns"]
        lines = [",".join(GPU_TELEMETRY_COLUMNS)]
        for timestamp in range(start - 250_000_000, finish + 250_000_001, 250_000_000):
            inside = start <= timestamp <= finish
            lines.append(",".join((
                str(timestamp),
                str(90 + number if inside else 0),
                str(40 + number if inside else 0),
                str(12_000 + number if inside else 1),
                str(650 + number if inside else 10),
                str(1_900 + number if inside else 100),
                "2619",
                str(60 + number if inside else 20),
                "570.86.15",
                "700",
                "1980",
                "2619",
            )))
        path.write_text("\n".join(lines) + "\n")
        payload = path.read_bytes()
        digest = sha(path)
        return {
            "schema": "stwo.gpu-telemetry.csv.v1",
            "columns": list(GPU_TELEMETRY_COLUMNS),
            "path": str(path),
            "sha256": digest,
            "remote_sha256": digest,
            "size_bytes": len(payload),
            "remote_size_bytes": len(payload),
            "sample_count": len(lines) - 1,
            "proof_window_sample_count": 17,
            "sampler_complete": True,
            "sample_interval_ms": 250,
            "max_gap_ns": 2_000_000_000,
            "transport_equal": True,
        }

    def write_bench_ledger(self) -> None:
        self.bench_ledger.write_text(
            "".join(json.dumps(entry) + "\n" for entry in self.bench_entries)
        )

    def measurement(
        self,
        program: str,
        *,
        steps: int,
        cycles: int,
        samples: list[float],
        simd: bool = False,
    ) -> dict:
        median = sorted(samples)[2]
        p95 = quantile(samples, 0.95)
        record = {
            "program": program,
            "backend": "cuda",
            "engine": "gpu-native",
            "gpu": self.gpu,
            "reps": 6,
            "gpu_proof_loop_started_unix_ns": 1_700_000_000_000_000_000,
            "gpu_proof_loop_finished_unix_ns": 1_700_000_004_000_000_000,
            "verified_reps": 6,
            "warm_sample_count": 5,
            "prove_s_warm_samples_raw": samples,
            "prove_s_cold": 8.0,
            "prove_s_warm_median": round3(median),
            "prove_s_warm_p95": round3(p95),
            "cycle_count": cycles,
            "pie_n_steps": steps,
            "mhz_median": round3(cycles / median / 1e6),
            "useful_mhz_median": round3(steps / median / 1e6),
            "mhz_at_warm_p95": round3(cycles / p95 / 1e6),
            "useful_mhz_at_warm_p95": round3(steps / p95 / 1e6),
            "throughput_distribution_applicable": True,
            "performance_claim_admissible": True,
            "proof_comparison_applicable": True,
            "proof_byte_equal_required": True,
            "proof_byte_equal": True,
            "security_bits": 96,
            "n_queries": 70,
            "pow_bits": 26,
            "fold_step": 3,
            "verify_ms": 42.0,
            "proof_kb": 250.0,
            "peak_rss_gb": 20.0,
            "vram_end_gb": 8.0,
            "vram_peak_gb": 30.0,
            "pool_used_high_gb": 28.0,
            "pool_reserved_high_gb": 29.0,
        }
        if simd:
            record.update({
                "simd_reference_required": True,
                "simd_reference_comparison_applicable": True,
                "simd_reference_byte_equal": True,
                "simd_reference_blake3": "d" * 64,
                "gpu_proof_blake3": "d" * 64,
                "simd_reference_fresh": True,
                "simd_reference_s": 12.345,
            })
        return record

    def write(self) -> None:
        self.path = self.root / "qualification.json"
        self.path.write_text(json.dumps(self.qualification, sort_keys=True))


class GenerateBenchmarkReportTests(unittest.TestCase):
    def test_passed_report_recomputes_metrics_and_marks_simd_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.qualification["profiles"]["universal_sn1_sn4"]["preflights"][
                "SN_PIE_1"
            ]["arena_gib"] = 999.0
            fixture.write()
            result = generate_report(fixture.root, fixture.root / "report")
            self.assertTrue(result["performance_admissible"])
            report = json.loads(Path(result["files"]["json"]).read_text())
            self.assertEqual(report["benchmarks"]["SN_PIE_1"]["warm_p95_s"], 4.8)
            self.assertEqual(report["benchmarks"]["SN_PIE_1"]["useful_mhz_median"], 4.0)
            identity = report["proof_identity"]["actual_sn_simd_byte_identity"]
            self.assertTrue(identity["established"])
            self.assertEqual(identity["references"]["SN_PIE_1"]["blake3"], "d" * 64)
            profile = report["profiling"]["fixed_sn"]["SN_PIE_1"]
            self.assertEqual(profile["warm_reps"], [1, 2, 3, 4, 5])
            self.assertEqual(
                profile["phase_medians"]["witness_generation"],
                {"median_count": 1.0, "median_total_ms": 1013.0},
            )
            self.assertEqual(profile["top_spans"][0]["phase"], "witness_generation")
            telemetry = report["profiling"]["gpu_telemetry"]["SN_PIE_1"]
            self.assertEqual(telemetry["proof_loop_sample_count"], 17)
            self.assertEqual(
                telemetry["metrics"]["utilization_gpu_pct"],
                {"mean": 91.0, "p95": 91.0, "max": 91.0},
            )
            self.assertEqual(telemetry["hardware"]["driver_version"], "570.86.15")
            ncu = report["profiling"]["ncu"]
            self.assertFalse(ncu["synthetic"])
            self.assertEqual(ncu["profiled_proof_sha256"], "b" * 64)
            self.assertEqual(
                report["preflights"]["SN_PIE_1"]["arena_gib"],
                fixture.preflight_summaries["universal_SN1"]["arena_bytes"] / 1024**3,
            )
            self.assertEqual(report["artifact_sha256"]["profiling.ncu.report"], sha(fixture.ncu_report))
            markdown = Path(result["files"]["markdown"]).read_text()
            self.assertIn("GPU-to-SIMD byte identity is established", markdown)
            self.assertIn("Tracing spans may be nested", markdown)
            self.assertIn("| witness_generation | 1.0 | 1013.000 |", markdown)
            self.assertIn("Proof-loop-envelope GPU telemetry", markdown)
            self.assertIn("not exact GPU-only time", markdown)
            self.assertIn("Targeted Nsight Compute profile", markdown)
            self.assertIn(str(fixture.ncu_import), markdown)
            for path_key, path in result["files"].items():
                self.assertEqual(result["sha256"][Path(path).name], sha(Path(path)), path_key)

    def test_dry_run_requires_explicit_permission_and_is_watermarked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=True)
            with self.assertRaisesRegex(ReportError, "--allow-dry-run"):
                generate_report(fixture.path, fixture.root / "rejected")
            result = generate_report(
                fixture.path, fixture.root / "allowed", allow_dry_run=True
            )
            self.assertFalse(result["performance_admissible"])
            report = json.loads(Path(result["files"]["json"]).read_text())
            self.assertFalse(report["performance_admissible"])
            identity = report["proof_identity"]["actual_sn_simd_byte_identity"]
            self.assertFalse(identity["established"])
            self.assertEqual(identity["status"], "synthetic_not_measured")
            self.assertNotIn("references", identity)
            self.assertFalse(report["proof_identity"]["gpu_repeat_determinism"]["established"])
            self.assertFalse(report["profiling"]["performance_admissible"])
            self.assertEqual(report["profiling"]["status"], "synthetic_ledger_phase_totals")
            self.assertTrue(report["profiling"]["ncu"]["synthetic"])
            self.assertIn("PERFORMANCE-INADMISSIBLE", Path(result["files"]["markdown"]).read_text())
            self.assertIn(
                "No actual correctness or GPU-to-SIMD identity claim is established",
                Path(result["files"]["markdown"]).read_text(),
            )

    def test_mutated_measurement_proof_gpu_and_structure_fail_closed(self) -> None:
        mutations = {
            "median": lambda q: q["benchmarks"]["SN_PIE_1"].update({"prove_s_warm_median": 99}),
            "proof equality": lambda q: q["benchmarks"]["SN_PIE_1"].update({"proof_byte_equal": False}),
            "proof hash": lambda q: q["proof_sha256"]["fixed_pies"].update({"SN_PIE_1": "bad"}),
            "GPU seal": lambda q: q["remote_execution_target"]["target"].update({"gpu_name": "other"}),
            "missing SN": lambda q: q["benchmarks"].pop("SN_PIE_4"),
            "security": lambda q: q["benchmarks"]["SN_PIE_1"].update({"n_queries": 3}),
            "SIMD stale": lambda q: q["benchmarks"]["SN_PIE_1"].update({"simd_reference_fresh": False}),
            "SIMD unequal": lambda q: q["benchmarks"]["SN_PIE_1"].update({"simd_reference_byte_equal": False}),
            "SIMD hash": lambda q: q["benchmarks"]["SN_PIE_1"].update({"simd_reference_blake3": "D" * 64}),
            "GPU/SIMD digest mismatch": lambda q: q["benchmarks"]["SN_PIE_1"].update({"gpu_proof_blake3": "e" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.qualification)
                fixture.write()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

    def test_mutated_bound_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.bench_ledger.write_text('{"status":"mutated"}\n')
            with self.assertRaisesRegex(ReportError, "ledger.bench: sha256 mismatch"):
                generate_report(fixture.path, fixture.root / "report")

    def test_phase_ledger_mutations_fail_closed(self) -> None:
        def duplicate(entries: list[dict]) -> None:
            entries.append(copy.deepcopy(entries[0]))

        def missing(entries: list[dict]) -> None:
            entries.pop(0)

        def missing_rep(entries: list[dict]) -> None:
            entries[0]["phase_totals"].pop()

        def duplicate_rep(entries: list[dict]) -> None:
            entries[0]["phase_totals"][-1]["rep"] = 4

        def phase_set_drift(entries: list[dict]) -> None:
            entries[0]["phase_totals"][-1]["phase_totals"].pop("fri")

        def empty_phase(entries: list[dict]) -> None:
            phases = entries[0]["phase_totals"][0]["phase_totals"]
            phases[""] = phases.pop("fri")

        def invalid_count(entries: list[dict]) -> None:
            entries[0]["phase_totals"][0]["phase_totals"]["fri"]["count"] = True

        def invalid_total(entries: list[dict]) -> None:
            entries[0]["phase_totals"][0]["phase_totals"]["fri"]["total_ms"] = math.inf

        def record_drift(entries: list[dict]) -> None:
            entries[0]["record"]["proof_kb"] += 1

        mutations = {
            "duplicate fixed SN": duplicate,
            "missing fixed SN": missing,
            "missing rep": missing_rep,
            "duplicate rep": duplicate_rep,
            "phase set drift": phase_set_drift,
            "empty phase name": empty_phase,
            "invalid count": invalid_count,
            "invalid total": invalid_total,
            "measurement drift": record_drift,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.bench_entries)
                fixture.write_bench_ledger()
                fixture.qualification["ledgers"]["bench"]["sha256"] = sha(
                    fixture.bench_ledger
                )
                fixture.write()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

    def test_telemetry_mutation_and_deletion_fail_closed(self) -> None:
        metadata_mutations = {
            "remote digest": lambda value: value.update({"remote_sha256": "0" * 64}),
            "remote size": lambda value: value.update({"remote_size_bytes": 1}),
            "interval": lambda value: value.update({"sample_interval_ms": 1000}),
            "window count": lambda value: value.update({"proof_window_sample_count": 1}),
            "schema": lambda value: value.update({"schema": "wrong"}),
        }
        for name, mutate in metadata_mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.qualification["gpu_telemetry"]["SN_PIE_1"])
                fixture.write()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

        for operation in ("mutate", "delete"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                path = Path(fixture.telemetry["SN_PIE_1"]["path"])
                if operation == "mutate":
                    path.write_bytes(path.read_bytes() + b"mutated")
                else:
                    path.unlink()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            metadata = fixture.telemetry["SN_PIE_1"]
            path = Path(metadata["path"])
            lines = path.read_text().splitlines()
            # Retain valid hashes/counts while introducing a three-second cadence gap.
            path.write_text("\n".join((lines[0], lines[1], lines[2], lines[14], lines[-1])) + "\n")
            payload = path.read_bytes()
            digest = sha(path)
            metadata.update({
                "sha256": digest,
                "remote_sha256": digest,
                "size_bytes": len(payload),
                "remote_size_bytes": len(payload),
                "sample_count": 4,
                "proof_window_sample_count": 2,
            })
            fixture.write()
            with self.assertRaisesRegex(ReportError, "gap exceeds 2 seconds"):
                generate_report(fixture.path, fixture.root / "report")

    def test_ncu_mutation_and_deletion_fail_closed(self) -> None:
        mutations = {
            "status": lambda q: q["profiling"]["ncu"].update({"status": "failed"}),
            "schema": lambda q: q["profiling"]["ncu"]["profile"].update({"schema": "wrong"}),
            "synthetic": lambda q: q["profiling"]["ncu"]["profile"].update({"synthetic": True}),
            "proof": lambda q: q["profiling"]["ncu"]["profile"].update({"profiled_proof_sha256": "e" * 64}),
            "remote CSV hash": lambda q: q["profiling"]["ncu"]["profile"].update({"remote_import_output_sha256": "e" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.qualification)
                fixture.write()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

        for artifact_name in ("ncu_report", "ncu_import"):
            for operation in ("mutate", "delete"):
                with self.subTest(artifact=artifact_name, operation=operation), \
                        tempfile.TemporaryDirectory() as temporary:
                    fixture = Fixture(Path(temporary), dry_run=False)
                    path = getattr(fixture, artifact_name)
                    if operation == "mutate":
                        path.write_bytes(path.read_bytes() + b"mutated")
                    else:
                        path.unlink()
                    with self.assertRaises(ReportError):
                        generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.ncu_import.write_text(
                '"ID","Kernel Name","Metric Name"\n', encoding="utf-8"
            )
            profile = fixture.qualification["profiling"]["ncu"]["profile"]
            digest = sha(fixture.ncu_import)
            profile.update({
                "import_output_sha256": digest,
                "remote_import_output_sha256": digest,
                "import_output_bytes": fixture.ncu_import.stat().st_size,
                "remote_import_output_bytes": fixture.ncu_import.stat().st_size,
                "profiled_kernel_rows": 0,
            })
            fixture.write()
            with self.assertRaisesRegex(ReportError, "no profiled kernel data rows"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.ncu_import.write_text(
                '"ID","Kernel Name","Metric Name"\n'
                '"1","unrelated_kernel","metric"\n',
                encoding="utf-8",
            )
            profile = fixture.qualification["profiling"]["ncu"]["profile"]
            digest = sha(fixture.ncu_import)
            profile.update({
                "import_output_sha256": digest,
                "remote_import_output_sha256": digest,
                "import_output_bytes": fixture.ncu_import.stat().st_size,
                "remote_import_output_bytes": fixture.ncu_import.stat().st_size,
            })
            fixture.write()
            with self.assertRaisesRegex(ReportError, "kernel outside declared regex"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=True)
            fixture.ncu_report.write_bytes(b"NVP\0real-looking")
            profile = fixture.qualification["profiling"]["ncu"]["profile"]
            digest = sha(fixture.ncu_report)
            profile.update({
                "sha256": digest,
                "remote_sha256": digest,
                "bytes": fixture.ncu_report.stat().st_size,
                "remote_bytes": fixture.ncu_report.stat().st_size,
            })
            fixture.write()
            with self.assertRaisesRegex(ReportError, "synthetic marker"):
                generate_report(
                    fixture.path, fixture.root / "report", allow_dry_run=True
                )

    def test_observability_artifacts_resolve_after_round_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            relocated = fixture.root / "relocated_observability"
            relocated.mkdir()
            moved = [
                *(Path(metadata["path"]) for metadata in fixture.telemetry.values()),
                fixture.ncu_report,
                fixture.ncu_import,
            ]
            for path in moved:
                path.rename(relocated / path.name)
            # Qualification retains the old absolute paths; basename resolution is round-local.
            result = generate_report(fixture.path, fixture.root / "report")
            report = json.loads(Path(result["files"]["json"]).read_text())
            self.assertEqual(
                Path(report["profiling"]["ncu"]["import_csv"]["path"]).parent,
                relocated.resolve(),
            )
            self.assertTrue(all(
                Path(value["path"]).parent == relocated.resolve()
                for value in report["profiling"]["gpu_telemetry"].values()
            ))

    def test_refuses_failed_qualification_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            original = copy.deepcopy(fixture.qualification)
            fixture.qualification["status"] = "failed"
            fixture.write()
            with self.assertRaisesRegex(ReportError, "refusing"):
                generate_report(fixture.path, fixture.root / "report")
            fixture.qualification = original
            fixture.write()
            generate_report(fixture.path, fixture.root / "report")
            with self.assertRaisesRegex(ReportError, "overwrite"):
                generate_report(fixture.path, fixture.root / "report")


if __name__ == "__main__":
    unittest.main()
