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
from unittest import mock

try:
    from gpu_benchmarks.generate_benchmark_report import (
        ARCHITECTURE,
        FLAGS_OFF_POLICY,
        GPU_TELEMETRY_COLUMNS,
        HEADLINE_POLICY,
        NCU_RELEASE_KERNEL_REGEX,
        PREFLIGHT_CAP_BYTES,
        QUALIFICATION_FLAGS,
        QUALIFICATION_NORMALIZED_STATES,
        QUALIFICATION_PROFILES,
        ReportError,
        RETAINED_BUDGET_FLAG,
        STAGES,
        UNIVERSAL_POLICY,
        generate_report,
    )
    from gpu_benchmarks.run_cuda_soundness_gate import (
        STRICT_RESIDENT_GATE,
        STRICT_RESIDENT_REQUIRED_TESTS,
        gates_for_runtime_mode,
    )
except ModuleNotFoundError:  # Direct execution from gpu_benchmarks/.
    from generate_benchmark_report import (
        ARCHITECTURE,
        FLAGS_OFF_POLICY,
        GPU_TELEMETRY_COLUMNS,
        HEADLINE_POLICY,
        NCU_RELEASE_KERNEL_REGEX,
        PREFLIGHT_CAP_BYTES,
        QUALIFICATION_FLAGS,
        QUALIFICATION_NORMALIZED_STATES,
        QUALIFICATION_PROFILES,
        ReportError,
        RETAINED_BUDGET_FLAG,
        STAGES,
        UNIVERSAL_POLICY,
        generate_report,
    )
    from run_cuda_soundness_gate import (
        STRICT_RESIDENT_GATE,
        STRICT_RESIDENT_REQUIRED_TESTS,
        gates_for_runtime_mode,
    )


_REAL_GENERATE_REPORT = generate_report
_TEST_CANONICAL_MANIFESTS = None


def generate_report(*args, **kwargs):
    assert _TEST_CANONICAL_MANIFESTS is not None
    raw, adapted = _TEST_CANONICAL_MANIFESTS
    with mock.patch.dict(_REAL_GENERATE_REPORT.__globals__, {
        "CANONICAL_RAW_MANIFEST_PATH": raw,
        "CANONICAL_ADAPTED_MANIFEST_PATH": adapted,
    }):
        return _REAL_GENERATE_REPORT(*args, **kwargs)


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
    def __init__(
        self,
        root: Path,
        *,
        dry_run: bool,
        canonical_target: bool = True,
        gpu: str = "",
        unclean_source: bool = False,
    ):
        self.root = root
        self.dry_run = dry_run
        clean_hash = hashlib.sha256(b"").hexdigest()
        self.source = {
            "stwo": {
                "head": "1" * 40,
                "worktree_hash": "2" * 64 if dry_run or unclean_source else clean_hash,
            },
            "stwo_cairo": {
                "head": "3" * 40,
                "worktree_hash": "4" * 64 if dry_run or unclean_source else clean_hash,
            },
        }
        self.gpu = gpu or ("DRY-RUN-GPU" if dry_run else "NVIDIA H100 80GB HBM3")
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
        self.canonical_raw_manifest = root / "canonical-raw.sha256"
        self.canonical_raw_manifest.write_text(self.raw_manifest.read_text())
        self.canonical_adapted_manifest = root / "canonical-adapted.sha256"
        self.canonical_adapted_manifest.write_text(self.adapted_manifest.read_text())
        global _TEST_CANONICAL_MANIFESTS
        _TEST_CANONICAL_MANIFESTS = (
            self.canonical_raw_manifest,
            self.canonical_adapted_manifest,
        )

        ceiling = PREFLIGHT_CAP_BYTES
        self.aot_occurrences = [{
            "kind": "constraint",
            "component": "fixture_component",
            "instance": 0,
            "kernel": 0,
            "kernel_name": "stwo_jit_fused_1111111111111111",
            "semantic_hash": "1111111111111111",
            "cache_key": "2222222222222222",
        }]
        self.aot_manifest = root / "aot_manifest.json"
        self.aot_manifest.write_text(json.dumps([{
            "kind": "constraint",
            "label": "fixture_component",
            "kernel_name": "stwo_jit_fused_1111111111111111",
            "cache_key": "2222222222222222",
            "semantic_hash": "1111111111111111",
            "file": "constraint_fixture_component_2222222222222222.cu",
        }]) + "\n")
        self.preflight_summaries = {}
        for label, number, arena in [
            *( (f"universal_SN{n}", n, 20 * 1024**3 + n) for n in range(1, 5) ),
            ("headline_SN2", 2, 30 * 1024**3),
            ("flags_off_SN2", 2, 40 * 1024**3),
        ]:
            adapted_path = adapted / f"SN_PIE_{number}.adapted.bin"
            if label.startswith("universal_"):
                policy = copy.deepcopy(UNIVERSAL_POLICY)
            elif label == "headline_SN2":
                policy = copy.deepcopy(HEADLINE_POLICY)
            else:
                policy = copy.deepcopy(FLAGS_OFF_POLICY)
            artifact = root / f"preflight_{label}.json"
            artifact.write_text(json.dumps({
                "pass": True,
                "vram_fit": True,
                "vram_budget_bytes": ceiling,
                "arena": {"total_bytes": arena},
                "runtime_policy": policy,
                "source": str(adapted_path),
                "aot_coverage": {
                    "pass": True,
                    "manifest": str(self.aot_manifest),
                    "manifest_blake3": "a" * 64,
                    "manifest_entries": 1,
                    "required_occurrences": self.aot_occurrences,
                    "required_occurrences_blake3": "b" * 64,
                    "required_occurrence_count": 1,
                    "required_unique_keys_blake3": "c" * 64,
                    "required_unique_key_count": 1,
                    "missing_occurrences": [],
                    "missing_occurrence_count": 0,
                },
            }))
            self.preflight_summaries[label] = {
                "artifact": str(artifact),
                "artifact_sha256": sha(artifact),
                "adapted_input": str(adapted_path),
                "adapted_input_sha256": sha(adapted_path),
                "arena_bytes": arena,
                "arena_gib": arena / 1024**3,
                "runtime_policy": policy,
                "aot_coverage": {
                    "manifest": str(self.aot_manifest),
                    "manifest_blake3": "a" * 64,
                    "manifest_entries": 1,
                    "required_occurrence_count": 1,
                    "required_unique_key_count": 1,
                    "required_occurrences_blake3": "b" * 64,
                    "required_unique_keys_blake3": "c" * 64,
                    "missing_occurrence_count": 0,
                },
            }

        pie_root = (
            "/workspace/stwo-cairo/gpu_benchmarks/pie/sn"
            if canonical_target else "/remote"
        )
        remote_inputs = {
            "gate": {"path": f"{pie_root}/SN_PIE_2.zip", "sha256": self.raw_hashes["SN_PIE_2.zip"]},
            "bootloader": {
                "path": (
                    "/workspace/bench_inputs/simple_bootloader_compiled.json"
                    if canonical_target else "/remote/bootloader.json"
                ),
                "sha256": self.raw_hashes["simple_bootloader_compiled.json"],
            },
            **{
                f"SN_PIE_{number}": {
                    "path": f"{pie_root}/SN_PIE_{number}.zip",
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
            "gpu_bench": {
                "path": (
                    f"/workspace/bench_loop_runs/sealed/gpu_bench.{'a' * 64}"
                    if canonical_target else f"/remote/sealed/gpu_bench.{'a' * 64}"
                ),
                "sha256": "a" * 64,
            },
            "inputs": remote_inputs,
        }
        self.projection = {
            "method": "rsync-archive-checksum-dry-run-clean",
            "verified_after_soundness": True,
            "source": self.source,
        }
        effective_stwo_env = {
            "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
            "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
            "STWO_PARITY_REF_STWO_HEAD": self.source["stwo"]["head"],
            "STWO_PARITY_REF_STWO_WORKTREE_HASH": self.source["stwo"]["worktree_hash"],
            "STWO_PARITY_REF_STWO_CAIRO_HEAD": self.source["stwo_cairo"]["head"],
            "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": self.source["stwo_cairo"]["worktree_hash"],
            **{flag: "1" for flag in QUALIFICATION_FLAGS},
            RETAINED_BUDGET_FLAG: "29469326848",
        }
        gates = []
        for name, command, required in gates_for_runtime_mode("arena-graph"):
            gate = {
                "name": name,
                "command": list(command),
                "exit_code": 0,
                "executed_tests": required,
                "required_tests": required,
                "stub_skip_detected": False,
                "passed": True,
            }
            if name == STRICT_RESIDENT_GATE:
                gate["required_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
                gate["executed_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
            gates.append(gate)
        self.soundness_payload = {
            "schema": "stwo.cuda.soundness-gate.v3",
            "stwo_git_head": self.source["stwo"]["head"],
            "stwo_cairo_git_head": self.source["stwo_cairo"]["head"],
            "stwo_worktree_hash": self.source["stwo"]["worktree_hash"],
            "stwo_cairo_worktree_hash": self.source["stwo_cairo"]["worktree_hash"],
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
            "qualification_flags": {flag: 1 for flag in QUALIFICATION_FLAGS},
            "effective_stwo_env": effective_stwo_env,
            "gates": gates,
        }
        self.soundness = root / "soundness.json"
        self.soundness.write_text(json.dumps(self.soundness_payload))
        self.adapter_reproduction = {
            "byte_equal": True,
            "gpu_bench_binary_sha256": "c" * 64,
            "raw_input_manifest_sha256": sha(self.raw_manifest),
            "bootloader_sha256": "5" * 64,
            "pinned_adapted_manifest_sha256": sha(self.pinned_manifest),
        }
        occurrence_sha = canonical_sha(self.aot_occurrences)
        key_sha = canonical_sha(["2222222222222222"])
        preflight_occurrence_sha = {
            Path(summary["artifact"]).name: occurrence_sha
            for summary in self.preflight_summaries.values()
        }
        self.aot_admission = {
            "manifest": str(self.aot_manifest),
            "manifest_sha256": sha(self.aot_manifest),
            "manifest_blake3": "a" * 64,
            "manifest_entry_count": 1,
            "required_occurrence_count": 1,
            "required_unique_key_count": 1,
            "required_unique_keys": ["2222222222222222"],
            "required_occurrences_sha256": occurrence_sha,
            "required_unique_keys_sha256": key_sha,
            "preflight_occurrences_sha256": preflight_occurrence_sha,
            "preflight_occurrences_blake3": {
                name: "b" * 64 for name in preflight_occurrence_sha
            },
            "preflight_unique_keys_blake3": {
                name: "c" * 64 for name in preflight_occurrence_sha
            },
            "per_sn": {
                f"SN_PIE_{number}": {
                    "required_occurrence_count": 1,
                    "required_unique_key_count": 1,
                    "required_occurrences_sha256": occurrence_sha,
                    "required_unique_keys_sha256": key_sha,
                }
                for number in range(1, 5)
            },
            "missing_occurrences": [],
        }
        self.aot_index_check = root / "aot-index-check.json"
        self.aot_index_check.write_text(json.dumps({
            "schema": "stwo.aot-index-check.v1",
            "dry_run": dry_run,
            "pass": True,
            "sm": 90,
            "loaded_manifest_hash": "000000000000c0da",
            "required_unique_key_count": 1,
            "required_unique_keys_sha256": key_sha,
            "missing_keys": [],
            "checker_binary": {
                "path": "/workspace/stwo-cairo/stwo_cairo_prover/target/release/aot_index_check",
                "sha256": "d" * 64,
            },
            "source": self.source,
        }))
        self.local_admission = root / "local_admission.json"
        self.local_admission.write_text(json.dumps({
            "schema": "stwo.local-preflight-admission.v1",
            "passed": True,
            "dry_run": dry_run,
            "runtime_mode": "arena-graph",
            "source": self.source,
            "profiles": {
                **QUALIFICATION_PROFILES,
            },
            "preflight_ceiling_bytes": ceiling,
            "preflight_artifact_sha256": {
                Path(summary["artifact"]).name: summary["artifact_sha256"]
                for summary in self.preflight_summaries.values()
            },
            "aot_coverage": self.aot_admission,
            "adapted_input_manifest_sha256": sha(self.adapted_manifest),
            "adapter_reproduction": self.adapter_reproduction,
        }))
        self.bench_ledger = root / "bench.jsonl"
        self.ab_ledger = root / "sn2_ab.jsonl"

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
        ledger_seal = {
            "execution_target": self.target,
            "execution_target_sha256": canonical_sha(self.target),
            "source_projection": self.projection,
            "soundness_gate_sha256": sha(self.soundness),
            "execution_guard_passed": True,
            "remote_quiescence_passed": True,
        }
        self.bench_entries = [
            {
                "run_name": "gate_correctness",
                "pod_gpu": self.gpu,
                **copy.deepcopy(ledger_seal),
            },
            {
                "run_name": "gate_correctness",
                "pod_gpu": self.gpu,
                **copy.deepcopy(ledger_seal),
            },
            *[
            {
                "run_name": name,
                "status": "ok",
                "bench_env": QUALIFICATION_PROFILES["universal_sn1_sn4"],
                "qualification_probe": True,
                "pod_gpu": self.gpu,
                "proof_sha256": proof_hashes[name],
                "gpu_telemetry": copy.deepcopy(self.telemetry[name]),
                **copy.deepcopy(ledger_seal),
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
            ],
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
            "kernel_regex": NCU_RELEASE_KERNEL_REGEX,
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
        self.ab_entry = {
            "status": "ok",
            "lane": "sn2_headline",
            "bench_env": QUALIFICATION_PROFILES["flags_off"],
            "candidate_env": QUALIFICATION_PROFILES["sn2_headline"],
            "baseline_state": copy.deepcopy(
                QUALIFICATION_NORMALIZED_STATES["flags_off"]
            ),
            "candidate_state": copy.deepcopy(
                QUALIFICATION_NORMALIZED_STATES["sn2_headline"]
            ),
            "provisional": True,
            "performance_admissible": False,
            "pod_gpu": self.gpu,
            "architecture_soundness": {"sha256": sha(self.soundness)},
            "execution_target": copy.deepcopy(self.target),
            "execution_target_sha256": canonical_sha(self.target),
            "source_projection": copy.deepcopy(self.projection),
            "execution_guard_passed": True,
            "remote_quiescence_passed": True,
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
            "ncu_profile_required": True,
            "ncu_profile_requested": True,
            "ncu_profile_attempted": True,
            "ncu_profile_status": "validated",
            "ncu_profile": copy.deepcopy(self.ncu_profile),
        }
        self.ab_ledger.write_text(json.dumps(self.ab_entry) + "\n")
        self.qualification = {
            "schema": "stwo.qualification-round.v4",
            "status": "dry_run" if dry_run else "passed",
            "dry_run": dry_run,
            "performance_admissible": not dry_run,
            "runtime_mode": "arena-graph",
            "gpu": self.gpu,
            "source": {
                **self.source,
                "sync": {
                    "method": "rsync checksum with target and result exclusions",
                    "release_requires_clean_commits": True,
                },
            },
            "profiles": {
                "universal_sn1_sn4": {
                    "env": QUALIFICATION_PROFILES["universal_sn1_sn4"],
                    "effective_state": copy.deepcopy(
                        QUALIFICATION_NORMALIZED_STATES["universal_sn1_sn4"]
                    ),
                    "preflight_ceiling_bytes": ceiling,
                    "preflights": {
                        f"SN_PIE_{number}": self.preflight_summaries[f"universal_SN{number}"]
                        for number in range(1, 5)
                    },
                },
                "sn2_headline": {
                    "env": QUALIFICATION_PROFILES["sn2_headline"],
                    "effective_state": copy.deepcopy(
                        QUALIFICATION_NORMALIZED_STATES["sn2_headline"]
                    ),
                    "preflight_ceiling_bytes": ceiling,
                    "preflight": self.preflight_summaries["headline_SN2"],
                },
            },
            "normalized_states": copy.deepcopy(QUALIFICATION_NORMALIZED_STATES),
            "flags_off_preflight": self.preflight_summaries["flags_off_SN2"],
            "adapter_reproduction": copy.deepcopy(self.adapter_reproduction),
            "aot_coverage": copy.deepcopy(self.aot_admission),
            "aot_index_check": {
                "path": str(self.aot_index_check),
                "sha256": sha(self.aot_index_check),
            },
            "local_admission": {"path": str(self.local_admission), "sha256": sha(self.local_admission)},
            "soundness": {
                "path": str(self.soundness),
                "sha256": sha(self.soundness),
                "effective_stwo_env": effective_stwo_env,
                "stwo_worktree_hash": self.source["stwo"]["worktree_hash"],
                "stwo_cairo_worktree_hash": self.source["stwo_cairo"]["worktree_hash"],
            },
            "remote_execution_target": {
                "target": self.target,
                "sha256": canonical_sha(self.target),
                "postcheck": True,
                "source_projection": self.projection,
                "ledger_entries_guarded": 7,
                "remote_quiescence_passed": True,
                "quiescent_ledger_entries": 7,
                "quiescent_measurement_launches": 9,
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
                    **copy.deepcopy(self.ab_entry),
                    "useful_mhz_ratio": ratio,
                },
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
        reps: int = 6,
    ) -> dict:
        median = sorted(samples)[len(samples) // 2]
        p95 = quantile(samples, 0.95)
        record = {
            "program": program,
            "backend": "cuda",
            "engine": "gpu-native",
            "gpu": self.gpu,
            "gpu_pcs_driver_architecture": ARCHITECTURE,
            "gpu_pcs_runtime_mode": "ArenaGraph",
            "gpu_pcs_stage_started": {stage: 1 for stage in STAGES},
            "gpu_pcs_stage_finished": {stage: 1 for stage in STAGES},
            "gpu_pcs_batched_tree_decommit": True,
            "gpu_pcs_driver_complete": True,
            "gpu_native_architecture_required": True,
            "gpu_pcs_required_runtime_mode": "arena-graph",
            "gpu_native_architecture_gate_passed": True,
            "gpu_aot_loads": 2,
            "gpu_aot_cache_hits": 5,
            "reps": reps,
            "gpu_proof_loop_started_unix_ns": 1_700_000_000_000_000_000,
            "gpu_proof_loop_finished_unix_ns": 1_700_000_004_000_000_000,
            "verified_reps": reps,
            "warm_sample_count": reps - 1,
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
            "gpu_aot_manifest_hash": 0xC0DA,
            "gpu_aot_misses": 0,
            "gpu_aot_runtime_loads": 0,
            "gpu_aot_runtime_cache_hits": 0,
            "gpu_aot_strict_rejections": 0,
            "gpu_aot_provenance_gate_passed": True,
            "steps_per_s": steps / median,
            "mhz": cycles / median / 1e6,
            "useful_mhz": steps / median / 1e6,
            "gpu_host_syncs": 1,
            "gpu_graph_launches": 14,
            "gpu_kernel_launches": 2_473,
            "gpu_expected_graph_launches": 14,
            "gpu_expected_kernel_launches": 2_473,
            "gpu_transcript_segments": 15,
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
            "resident runtime": lambda q: q["benchmarks"]["SN_PIE_1"].pop(
                "gpu_pcs_runtime_mode"
            ),
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

    def test_noncanonical_profiles_states_and_admission_fail_closed(self) -> None:
        mutations = {
            "universal env": lambda q: q["profiles"]["universal_sn1_sn4"].update(
                {"env": "STWO_UNIVERSAL=1"}
            ),
            "headline env": lambda q: q["profiles"]["sn2_headline"].update(
                {"env": "STWO_HEADLINE=1"}
            ),
            "universal state": lambda q: q["profiles"]["universal_sn1_sn4"][
                "effective_state"
            ].update({"STWO_CUDA_B2N_STAGE_FUSED": 0}),
            "headline state": lambda q: q["profiles"]["sn2_headline"][
                "effective_state"
            ].update({"STWO_CUDA_COMPOSITION_DIRECT_RETENTION": 0}),
            "normalized flags-off state": lambda q: q["normalized_states"][
                "flags_off"
            ].update({"STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE": 1}),
            "preflight ceiling": lambda q: q["profiles"]["universal_sn1_sn4"].update(
                {"preflight_ceiling_bytes": PREFLIGHT_CAP_BYTES - 1}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.qualification)
                fixture.write()
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            admission = json.loads(fixture.local_admission.read_text())
            admission["profiles"]["sn2_headline"] = "STWO_HEADLINE=1"
            fixture.local_admission.write_text(json.dumps(admission))
            fixture.qualification["local_admission"]["sha256"] = sha(
                fixture.local_admission
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "local admission profiles mismatch"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.qualification.pop("adapter_reproduction")
            fixture.write()
            with self.assertRaisesRegex(ReportError, "adapter reproduction binding"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            with self.assertRaisesRegex(ReportError, "canonical release manifest"):
                _REAL_GENERATE_REPORT(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            entries = fixture.pinned_manifest.read_text().splitlines()
            entries[0] = f"{'0' * 64}  SN_PIE_1.adapted.bin"
            fixture.pinned_manifest.write_text("\n".join(entries) + "\n")
            digest = sha(fixture.pinned_manifest)
            fixture.qualification["inputs"]["pinned_adapted_manifest_sha256"] = digest
            fixture.qualification["adapter_reproduction"][
                "pinned_adapted_manifest_sha256"
            ] = digest
            admission = json.loads(fixture.local_admission.read_text())
            admission["adapter_reproduction"][
                "pinned_adapted_manifest_sha256"
            ] = digest
            fixture.local_admission.write_text(json.dumps(admission))
            fixture.qualification["local_admission"]["sha256"] = sha(
                fixture.local_admission
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "pinned_adapted_manifest"):
                generate_report(fixture.path, fixture.root / "report")

    def test_release_gpu_source_and_sync_are_exact(self) -> None:
        cases = {
            "GPU": {"gpu": "NVIDIA A100-SXM4-80GB"},
            "clean source": {"unclean_source": True},
        }
        for name, options in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False, **options)
                with self.assertRaises(ReportError):
                    generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.qualification["source"]["sync"][
                "release_requires_clean_commits"
            ] = False
            fixture.write()
            with self.assertRaisesRegex(ReportError, "source.sync"):
                generate_report(fixture.path, fixture.root / "report")

    def test_every_qualification_measurement_requires_six_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            prior = fixture.qualification["benchmarks"]["SN_PIE_1"]
            reduced = fixture.measurement(
                "SN_PIE_1.zip",
                steps=prior["pie_n_steps"],
                cycles=prior["cycle_count"],
                samples=[2.0],
                simd=True,
                reps=2,
            )
            fixture.qualification["benchmarks"]["SN_PIE_1"] = reduced
            entry = next(
                item for item in fixture.bench_entries
                if item.get("run_name") == "SN_PIE_1"
            )
            entry["record"] = copy.deepcopy(reduced)
            fixture.write_bench_ledger()
            fixture.qualification["ledgers"]["bench"]["sha256"] = sha(
                fixture.bench_ledger
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "requires exactly 6"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            prior = fixture.ab_entry["flagged"]["record"]
            reduced = fixture.measurement(
                "SN_PIE_2.zip",
                steps=prior["pie_n_steps"],
                cycles=prior["cycle_count"],
                samples=[2.0],
                reps=2,
            )
            for comparison in (
                fixture.ab_entry,
                fixture.qualification["comparisons"]["sn2_flags_off_vs_headline"],
            ):
                comparison["flagged"]["record"] = copy.deepcopy(reduced)
                comparison["flagged"]["useful_mhz_median"] = reduced[
                    "useful_mhz_median"
                ]
            qualified = fixture.qualification["comparisons"][
                "sn2_flags_off_vs_headline"
            ]
            qualified["useful_mhz_ratio"] = (
                reduced["useful_mhz_median"]
                / qualified["baseline"]["useful_mhz_median"]
            )
            fixture.ab_ledger.write_text(json.dumps(fixture.ab_entry) + "\n")
            fixture.qualification["ledgers"]["ab"]["sha256"] = sha(fixture.ab_ledger)
            fixture.write()
            with self.assertRaisesRegex(ReportError, "requires exactly 6"):
                generate_report(fixture.path, fixture.root / "report")

    def test_preflight_policy_must_match_canonical_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            summary = fixture.qualification["profiles"]["universal_sn1_sn4"][
                "preflights"
            ]["SN_PIE_1"]
            artifact = Path(summary["artifact"])
            record = json.loads(artifact.read_text())
            noncanonical = {"profile": "self-consistent-but-unqualified"}
            record["runtime_policy"] = noncanonical
            artifact.write_text(json.dumps(record))
            summary["runtime_policy"] = noncanonical
            summary["artifact_sha256"] = sha(artifact)
            admission = json.loads(fixture.local_admission.read_text())
            admission["preflight_artifact_sha256"][artifact.name] = sha(artifact)
            fixture.local_admission.write_text(json.dumps(admission))
            fixture.qualification["local_admission"]["sha256"] = sha(
                fixture.local_admission
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "qualification contract"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            summary = fixture.qualification["profiles"]["universal_sn1_sn4"][
                "preflights"
            ]["SN_PIE_1"]
            artifact = Path(summary["artifact"])
            record = json.loads(artifact.read_text())
            record["source"] = fixture.preflight_summaries["universal_SN2"][
                "adapted_input"
            ]
            artifact.write_text(json.dumps(record))
            summary["artifact_sha256"] = sha(artifact)
            admission = json.loads(fixture.local_admission.read_text())
            admission["preflight_artifact_sha256"][artifact.name] = sha(artifact)
            fixture.local_admission.write_text(json.dumps(admission))
            fixture.qualification["local_admission"]["sha256"] = sha(
                fixture.local_admission
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "artifact source"):
                generate_report(fixture.path, fixture.root / "report")

    def test_aot_manifest_identity_is_rederived_from_preflight_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            manifest = json.loads(fixture.aot_manifest.read_text())
            manifest[0]["semantic_hash"] = "3" * 16
            fixture.aot_manifest.write_text(json.dumps(manifest) + "\n")
            admission = json.loads(fixture.local_admission.read_text())
            admission["aot_coverage"]["manifest_sha256"] = sha(fixture.aot_manifest)
            fixture.local_admission.write_text(json.dumps(admission))
            fixture.qualification["aot_coverage"] = copy.deepcopy(
                admission["aot_coverage"]
            )
            fixture.qualification["local_admission"]["sha256"] = sha(
                fixture.local_admission
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "uncovered AOT key"):
                generate_report(fixture.path, fixture.root / "report")

    def test_embedded_aot_manifest_hash_must_match_every_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            index = json.loads(fixture.aot_index_check.read_text())
            index["loaded_manifest_hash"] = "deadbeefdeadbeef"
            fixture.aot_index_check.write_text(json.dumps(index))
            fixture.qualification["aot_index_check"]["sha256"] = sha(
                fixture.aot_index_check
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "embedded AOT manifest hash mismatch"):
                generate_report(fixture.path, fixture.root / "report")

    def test_full_soundness_contract_and_wrapper_are_bound(self) -> None:
        mutations = {
            "qualification flag": lambda value: value["qualification_flags"].update(
                {"STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE": 0}
            ),
            "effective environment": lambda value: value["effective_stwo_env"].update(
                {RETAINED_BUDGET_FLAG: "1"}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary), dry_run=False)
                mutate(fixture.soundness_payload)
                fixture.soundness.write_text(json.dumps(fixture.soundness_payload))
                reference = fixture.qualification["soundness"]
                reference["sha256"] = sha(fixture.soundness)
                reference["effective_stwo_env"] = fixture.soundness_payload[
                    "effective_stwo_env"
                ]
                fixture.write()
                with self.assertRaisesRegex(ReportError, "soundness qualification contract"):
                    generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.qualification["soundness"]["effective_stwo_env"] = {"unbound": "1"}
            fixture.write()
            with self.assertRaisesRegex(ReportError, "soundness wrapper"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(
                Path(temporary), dry_run=False, canonical_target=False
            )
            with self.assertRaisesRegex(ReportError, "release harness"):
                generate_report(fixture.path, fixture.root / "report")

    def test_embedded_comparison_must_equal_authoritative_ab_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            comparison = fixture.qualification["comparisons"][
                "sn2_flags_off_vs_headline"
            ]
            prior = comparison["flagged"]["record"]
            forged_seconds = prior["pie_n_steps"] / 6.1e6
            forged = fixture.measurement(
                "SN_PIE_2.zip",
                steps=prior["pie_n_steps"],
                cycles=prior["cycle_count"],
                samples=[forged_seconds] * 5,
            )
            self.assertEqual(forged["useful_mhz_median"], 6.1)
            comparison["flagged"]["record"] = forged
            comparison["flagged"]["useful_mhz_median"] = 6.1
            comparison["useful_mhz_ratio"] = (
                6.1 / comparison["baseline"]["useful_mhz_median"]
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "authoritative A/B ledger"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.ab_entry["unbound"] = "ledger changed after embedding"
            fixture.ab_ledger.write_text(json.dumps(fixture.ab_entry) + "\n")
            fixture.qualification["ledgers"]["ab"]["sha256"] = sha(fixture.ab_ledger)
            fixture.write()
            with self.assertRaisesRegex(ReportError, "authoritative A/B ledger"):
                generate_report(fixture.path, fixture.root / "report")

    def test_raw_ab_ncu_profile_must_equal_qualified_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.ab_entry["ncu_profile"]["kernel_regex"] = "relation_fused"
            fixture.ab_ledger.write_text(json.dumps(fixture.ab_entry) + "\n")
            fixture.qualification["ledgers"]["ab"]["sha256"] = sha(fixture.ab_ledger)
            comparison = fixture.qualification["comparisons"][
                "sn2_flags_off_vs_headline"
            ]
            comparison["ncu_profile"]["kernel_regex"] = "relation_fused"
            fixture.write()
            with self.assertRaisesRegex(ReportError, "qualified NCU capture"):
                generate_report(fixture.path, fixture.root / "report")

    def test_universal_bench_ledger_environment_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            entry = next(
                item for item in fixture.bench_entries
                if item.get("run_name") == "SN_PIE_1"
            )
            entry["bench_env"] = "STWO_UNIVERSAL=1"
            fixture.write_bench_ledger()
            fixture.qualification["ledgers"]["bench"]["sha256"] = sha(
                fixture.bench_ledger
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "qualification binding drifted"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            fixture.bench_entries[0]["pod_gpu"] = "NVIDIA A100-SXM4-80GB"
            fixture.write_bench_ledger()
            fixture.qualification["ledgers"]["bench"]["sha256"] = sha(
                fixture.bench_ledger
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "execution seal"):
                generate_report(fixture.path, fixture.root / "report")

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), dry_run=False)
            entry = next(
                item for item in fixture.bench_entries
                if item.get("run_name") == "SN_PIE_1"
            )
            entry.pop("gpu_telemetry")
            fixture.write_bench_ledger()
            fixture.qualification["ledgers"]["bench"]["sha256"] = sha(
                fixture.bench_ledger
            )
            fixture.write()
            with self.assertRaisesRegex(ReportError, "qualification binding drifted"):
                generate_report(fixture.path, fixture.root / "report")

    def test_phase_ledger_mutations_fail_closed(self) -> None:
        def fixed(entries: list[dict]) -> dict:
            return next(
                entry for entry in entries
                if entry.get("run_name", "").startswith("SN_PIE_")
            )

        def duplicate(entries: list[dict]) -> None:
            entries.append(copy.deepcopy(entries[0]))

        def missing(entries: list[dict]) -> None:
            entries.pop(0)

        def missing_rep(entries: list[dict]) -> None:
            fixed(entries)["phase_totals"].pop()

        def duplicate_rep(entries: list[dict]) -> None:
            fixed(entries)["phase_totals"][-1]["rep"] = 4

        def phase_set_drift(entries: list[dict]) -> None:
            fixed(entries)["phase_totals"][-1]["phase_totals"].pop("fri")

        def empty_phase(entries: list[dict]) -> None:
            phases = fixed(entries)["phase_totals"][0]["phase_totals"]
            phases[""] = phases.pop("fri")

        def invalid_count(entries: list[dict]) -> None:
            fixed(entries)["phase_totals"][0]["phase_totals"]["fri"]["count"] = True

        def invalid_total(entries: list[dict]) -> None:
            fixed(entries)["phase_totals"][0]["phase_totals"]["fri"]["total_ms"] = math.inf

        def record_drift(entries: list[dict]) -> None:
            fixed(entries)["record"]["proof_kb"] += 1

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
            "broad kernel subset": lambda q: q["profiling"]["ncu"]["profile"].update(
                {"kernel_regex": "relation_fused"}
            ),
            "launch count": lambda q: q["profiling"]["ncu"]["profile"].update(
                {"launch_count": 11}
            ),
            "metric set": lambda q: q["profiling"]["ncu"]["profile"].update(
                {"set": "basic"}
            ),
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
