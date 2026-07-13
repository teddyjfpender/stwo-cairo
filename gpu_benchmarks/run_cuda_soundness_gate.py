#!/usr/bin/env python3
"""Run the mandatory CUDA differential gates and emit execution evidence.

Exit status alone is insufficient: cfg-gated integration tests can succeed with
zero executed tests. Each command therefore has an exact expected test count,
and the resulting JSON is the artifact consumed by the hardware-run validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RESULT = re.compile(r"test result: ok\. (\d+) passed;")
TEST_STARTED = re.compile(r"(?m)^test ([^ ]+) \.\.\.(?: |$)")

RESIDENT_FIXTURE = re.compile(r'STRICT_RESIDENT_FIXTURE: &str = "([^"]+)"')

STRICT_RESIDENT_REQUIRED_TESTS = (
    "strict_resident_cold_and_warm_proofs_match_simd_bytes",
    "strict_resident_same_shape_changed_memory_matches_second_simd_proof",
    "strict_resident_same_workspace_changed_base_params_matches_second_simd_proof",
    "strict_resident_poseidon_graph_a_matches_simd_bytes",
    "strict_resident_transcript_mirror_diagnostic_once",
    "strict_resident_mirrored_transcript_matches_host_channel",
)

QUALIFICATION_FLAGS = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
    "STWO_CUDA_B2N_STAGE_FUSED",
)

REFERENCE_CACHE_SOURCE_ENV = (
    "STWO_PARITY_REF_STWO_HEAD",
    "STWO_PARITY_REF_STWO_WORKTREE_HASH",
    "STWO_PARITY_REF_STWO_CAIRO_HEAD",
    "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH",
)

REMOTE_EXECUTION_TARGET_SCHEMA = "stwo.remote-execution-target.v1"


GATES = (
    (
        "cuda_link_required",
        ("cargo", "test", "-p", "stwo-backend-cuda", "--test", "cuda_required"),
        3,
    ),
    (
        "fp256_poseidon_deduce_kinds_4_11",
        (
            "env",
            "STWO_CUDA_SOUNDNESS_REQUIRED=1",
            "cargo",
            "test",
            "--manifest-path",
            "../stwo-cairo/stwo_cairo_prover/Cargo.toml",
            "-p",
            "stwo-cairo-prover",
            "--lib",
            "stwo_wit_deduce_oracle_matches_fast_deduction",
        ),
        1,
    ),
    (
        "poseidon_partial_strict_aot_captured_row",
        (
            "env",
            "STWO_CUDA_SOUNDNESS_REQUIRED=1",
            "STWO_CUDA_WITNESS_JIT_MAX_INSTRS=8192",
            "cargo",
            "test",
            "--manifest-path",
            "../stwo-cairo/stwo_cairo_prover/Cargo.toml",
            "-p",
            "stwo-cairo-prover",
            "--lib",
            "poseidon_combination_37_strict_aot_captured_row",
            "--",
            "--ignored",
            "--test-threads=1",
        ),
        1,
    ),
    (
        "backend_conformance_both_channels",
        ("cargo", "test", "-p", "stwo-backend-cuda", "--test", "conformance"),
        2,
    ),
    (
        "prepared_commit_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_commit_native",
        ),
        2,
    ),
    (
        "prepared_progressive_commit_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_progressive_commit_native",
        ),
        1,
    ),
    (
        "prepared_merkle_from_progressive_leaves_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_merkle_from_leaves_native",
        ),
        1,
    ),
    (
        "prepared_fri_eager_capture_reference",
        ("cargo", "test", "-p", "stwo-backend-cuda", "--test", "prepared_fri_native"),
        1,
    ),
    (
        "prepared_oods_eager_capture_reference",
        ("cargo", "test", "-p", "stwo-backend-cuda", "--test", "prepared_oods_native"),
        1,
    ),
    (
        "prepared_numerator_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_quotient_numerator_native",
        ),
        2,
    ),
    (
        "prepared_quotient_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_quotient_native",
        ),
        1,
    ),
    (
        "prepared_composition_ext_params_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_composition_ext_params_native",
        ),
        1,
    ),
    (
        "prepared_interpolation_fused_exact_alias_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_interpolation_native",
        ),
        # Supported logs 3..30, mixed aliased/distinct columns, both launch
        # modes, eager/capture/mutation.
        1,
    ),
    (
        "prepared_relation_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_relation_native",
        ),
        # 2x2 launch-mode matrix plus compact fused implicit-launch parity;
        # every case covers eager/captured/mutated replay byte identity.
        5,
    ),
    (
        "prepared_witness_eager_capture_cpu_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_witness_native",
        ),
        1,
    ),
    (
        "prepared_execution_tables_eager_capture_mutated_content_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_execution_tables_native",
        ),
        2,
    ),
    (
        "prepared_ec_op_eager_capture_mutated_statement_reference",
        (
            "cargo",
            "test",
            "--manifest-path",
            "../stwo-cairo/stwo_cairo_prover/Cargo.toml",
            "-p",
            "stwo-cairo-prover",
            "--test",
            "prepared_ec_op_native",
        ),
        1,
    ),
    (
        "prepared_fixed_table_eager_capture_mutated_content_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_fixed_table_native",
        ),
        1,
    ),
    (
        "prepared_witness_feed_eager_capture_cpu_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_witness_feed_native",
        ),
        2,
    ),
    (
        "prepared_witness_input_gather_and_compact_eager_capture_cpu_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_witness_input_native",
        ),
        2,
    ),
    (
        "resident_custom_witness_arena_parity",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "resident_custom_witness_native",
        ),
        2,
    ),
    (
        "prepared_composition_eager_capture_cpu_reference",
        (
            "cargo",
            "test",
            "--manifest-path",
            "../stwo-cairo/stwo_cairo_prover/Cargo.toml",
            "-p",
            "stwo-cairo-gpu-prover",
            "--features",
            "direct-retention-test-api",
            "--test",
            "prepared_composition_native",
        ),
        3,
    ),
    (
        "prepared_final_fri_and_pow_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_fri_final_pow_native",
        ),
        2,
    ),
    (
        "prepared_decommit_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_decommit_native",
        ),
        1,
    ),
    (
        "device_transcript_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "device_transcript_native",
        ),
        1,
    ),
    (
        "strict_resident_whole_proof_simd_byte_identity",
        (
            "cargo",
            "test",
            "--manifest-path",
            "../stwo-cairo/stwo_cairo_prover/Cargo.toml",
            "-p",
            "stwo-cairo-gpu-prover",
            "--test",
            "resident_parity_native",
        ),
        len(STRICT_RESIDENT_REQUIRED_TESTS),
    ),
)

RUNTIME_MODES = ("detached-eager", "arena-graph")
STRICT_RESIDENT_GATE = "strict_resident_whole_proof_simd_byte_identity"


def gates_for_runtime_mode(runtime_mode: str):
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError(f"unsupported CUDA runtime mode: {runtime_mode}")
    if runtime_mode == "arena-graph":
        return GATES
    return tuple(gate for gate in GATES if gate[0] != STRICT_RESIDENT_GATE)


def run_gate(stwo: Path, name: str, command: tuple[str, ...], expected: int) -> dict:
    cwd = stwo
    actual_command = command
    if "--manifest-path" in command:
        manifest_index = command.index("--manifest-path")
        manifest = (stwo / command[manifest_index + 1]).resolve()
        cwd = manifest.parent
        actual_command = command[:manifest_index] + command[manifest_index + 2 :]
    env = dict(os.environ)
    env.setdefault("RUST_MIN_STACK", str(16 * 1024 * 1024))
    # The whole-proof gate builds full resident sessions (SN2-profile arenas
    # are ~15 GiB each plus the pedersen points table); four concurrent
    # sessions exhaust an 80 GiB device, so that target runs serially.
    extra_args: tuple[str, ...] = (
        ("--nocapture",) if "--" in actual_command else ("--", "--nocapture")
    )
    if name == STRICT_RESIDENT_GATE:
        extra_args = ("--", "--nocapture", "--test-threads=1")
    process = subprocess.run(
        actual_command + extra_args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    counts = [int(match.group(1)) for match in RESULT.finditer(process.stdout)]
    executed = max(counts, default=0)
    required_test_names = (
        STRICT_RESIDENT_REQUIRED_TESTS if name == STRICT_RESIDENT_GATE else None
    )
    executed_test_names = tuple(TEST_STARTED.findall(process.stdout))
    names_match = required_test_names is None or (
        set(executed_test_names) == set(required_test_names)
        and len(executed_test_names) == len(required_test_names)
    )
    stub_skip_detected = "SKIPPED (stub build)" in process.stdout
    passed = (
        process.returncode == 0
        and executed == expected
        and names_match
        and not stub_skip_detected
    )
    record = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "rust_min_stack": int(env["RUST_MIN_STACK"]),
        "exit_code": process.returncode,
        "executed_tests": executed,
        "required_tests": expected,
        "stub_skip_detected": stub_skip_detected,
        "passed": passed,
    }
    if required_test_names is not None:
        record["required_test_names"] = list(required_test_names)
        record["executed_test_names"] = list(executed_test_names)
    if name == "strict_resident_whole_proof_simd_byte_identity":
        # Record which fixture the whole-proof gate ran on, read from the test
        # source so the provenance cannot drift from the code.
        test_source = cwd / "crates" / "gpu-prover" / "tests" / "resident_parity_native.rs"
        fixture = RESIDENT_FIXTURE.search(test_source.read_text(encoding="utf-8"))
        if fixture:
            record["fixture"] = fixture.group(1)
    if not passed:
        sys.stderr.write(process.stdout)
        record["output_tail"] = process.stdout.splitlines()[-200:]
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stwo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "stwo",
    )
    parser.add_argument(
        "--stwo-cairo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--runtime-mode",
        choices=RUNTIME_MODES,
        default="arena-graph",
    )
    parser.add_argument("--synced-stwo-head")
    parser.add_argument("--synced-stwo-worktree-hash")
    parser.add_argument("--synced-stwo-cairo-head")
    parser.add_argument("--synced-stwo-cairo-worktree-hash")
    parser.add_argument("--pod-id")
    parser.add_argument("--gpu-bench", type=Path)
    parser.add_argument(
        "--input-artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="remote benchmark input to hash into the execution target",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stwo = args.stwo.resolve()
    stwo_cairo = args.stwo_cairo.resolve()
    if not (stwo / "Cargo.toml").is_file():
        raise SystemExit(f"not a stwo workspace: {stwo}")
    if not (stwo_cairo / "stwo_cairo_prover" / "Cargo.toml").is_file():
        raise SystemExit(f"not a stwo-cairo workspace: {stwo_cairo}")

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def gpu_property(name: str) -> str:
        value = subprocess.run(
            ("nvidia-smi", f"--query-gpu={name}", "--format=csv,noheader"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.splitlines()
        if not value or not value[0].strip():
            raise SystemExit(f"nvidia-smi did not report {name}")
        return value[0].strip()

    sealed_mode = bool(args.pod_id or args.gpu_bench or args.input_artifact)
    if sealed_mode and (not args.pod_id or args.gpu_bench is None or not args.input_artifact):
        raise SystemExit(
            "sealed execution requires --pod-id, --gpu-bench, and --input-artifact"
        )
    gpu_bench = args.gpu_bench.resolve() if args.gpu_bench is not None else None
    if sealed_mode and (gpu_bench is None or not gpu_bench.is_file()):
        raise SystemExit(f"gpu_bench binary does not exist: {gpu_bench}")
    input_paths: dict[str, Path] = {}
    for item in args.input_artifact:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or label in input_paths:
            raise SystemExit(f"invalid or duplicate --input-artifact: {item!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"input artifact does not exist: {path}")
        input_paths[label] = path
    if sealed_mode and not input_paths:
        raise SystemExit("at least one --input-artifact is required")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")

    def capture_execution_target() -> dict[str, object]:
        if gpu_bench is None or args.pod_id is None:
            raise RuntimeError("sealed execution target is not configured")
        boot_id = boot_id_path.read_text(encoding="utf-8").strip()
        if not boot_id:
            raise RuntimeError(f"empty pod boot id: {boot_id_path}")
        inputs = {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(input_paths.items())
        }
        return {
            "schema": REMOTE_EXECUTION_TARGET_SCHEMA,
            "pod_id": args.pod_id,
            "boot_id": boot_id,
            "gpu_uuid": gpu_property("uuid"),
            "gpu_name": gpu_property("name"),
            "gpu_bench": {"path": str(gpu_bench), "sha256": sha256_file(gpu_bench)},
            "inputs": inputs,
        }

    def canonical_sha256(value: object) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    execution_target = capture_execution_target() if sealed_mode else None

    def git_state(repo: Path) -> tuple[str | None, bool, str]:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
        digest = hashlib.sha256()
        tracked_diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--"),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
        digest.update(b"tracked-diff\0")
        digest.update(tracked_diff)
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.split(b"\0")
        for relative_bytes in sorted(path for path in untracked if path):
            relative = relative_bytes.decode("utf-8", errors="surrogateescape")
            path = repo / relative
            digest.update(b"untracked\0")
            digest.update(relative_bytes)
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(path.readlink().as_posix().encode("utf-8"))
            elif path.is_file():
                digest.update(path.read_bytes())
        return head or None, bool(status.strip()), digest.hexdigest()

    stwo_head, stwo_dirty, stwo_worktree_hash = git_state(stwo)
    stwo_cairo_head, stwo_cairo_dirty, stwo_cairo_worktree_hash = git_state(stwo_cairo)

    synced_values = (
        args.synced_stwo_head,
        args.synced_stwo_worktree_hash,
        args.synced_stwo_cairo_head,
        args.synced_stwo_cairo_worktree_hash,
    )
    if any(synced_values) and not all(synced_values):
        raise SystemExit("synced source identity requires both heads and both worktree hashes")
    if all(synced_values) and not sealed_mode:
        raise SystemExit("synced source identity is valid only for sealed execution")
    if sealed_mode and not all(synced_values):
        raise SystemExit("sealed execution requires a complete synced source identity")
    inherited_source_env = {
        key: os.environ[key] for key in REFERENCE_CACHE_SOURCE_ENV if key in os.environ
    }
    if all(synced_values):
        for value, length in zip(synced_values, (40, 64, 40, 64)):
            if len(value) != length or any(char not in "0123456789abcdef" for char in value):
                raise SystemExit("synced source identity must use lowercase hex heads and hashes")
        source_env = dict(zip(REFERENCE_CACHE_SOURCE_ENV, synced_values))
        if any(inherited_source_env.get(key, value) != value for key, value in source_env.items()):
            raise SystemExit("ambient reference-cache source identity disagrees with synced source")
        os.environ.update(source_env)
    elif inherited_source_env:
        raise SystemExit("reference-cache source identity requires runner-validated synced source")

    # Sealed pod trees are checksum projections with .git deliberately excluded.
    # Their controller-supplied identity is bound again by the post-soundness
    # rsync projection check before this artifact is admitted.
    if sealed_mode and all(synced_values):
        stwo_head = args.synced_stwo_head
        stwo_worktree_hash = args.synced_stwo_worktree_hash
        stwo_cairo_head = args.synced_stwo_cairo_head
        stwo_cairo_worktree_hash = args.synced_stwo_cairo_worktree_hash

    artifact = {
        "schema": (
            "stwo.cuda.soundness-gate.v3"
            if sealed_mode
            else "stwo.cuda.soundness-gate.v2"
        ),
        "dry_run": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stwo": str(stwo),
        "stwo_git_head": stwo_head,
        "stwo_git_dirty": stwo_dirty,
        "stwo_worktree_hash": stwo_worktree_hash,
        "stwo_cairo": str(stwo_cairo),
        "stwo_cairo_git_head": stwo_cairo_head,
        "stwo_cairo_git_dirty": stwo_cairo_dirty,
        "stwo_cairo_worktree_hash": stwo_cairo_worktree_hash,
        "runtime_mode": args.runtime_mode,
        "qualification_flags": {
            key: int(os.environ.get(key) == "1") for key in QUALIFICATION_FLAGS
        },
        "effective_stwo_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("STWO_")
        },
        "synced_source": (
            {
                "stwo": {
                    "head": args.synced_stwo_head,
                    "worktree_hash": args.synced_stwo_worktree_hash,
                },
                "stwo_cairo": {
                    "head": args.synced_stwo_cairo_head,
                    "worktree_hash": args.synced_stwo_cairo_worktree_hash,
                },
                "transport": "rsync-archive-checksum",
            }
            if all(synced_values)
            else None
        ),
        "gates": [],
        "passed": False,
    }
    if sealed_mode:
        artifact.update(
            {
                "execution_target": execution_target,
                "execution_target_sha256": canonical_sha256(execution_target),
                "execution_target_postcheck": False,
                "source_projection": None,
            }
        )
    try:
        for name, command, expected in gates_for_runtime_mode(args.runtime_mode):
            gate = run_gate(stwo, name, command, expected)
            artifact["gates"].append(gate)
            if not gate["passed"]:
                raise RuntimeError(
                    f"CUDA gate {name!r} failed: exit={gate['exit_code']}, "
                    f"executed={gate['executed_tests']}, required={expected}"
                )
        artifact["passed"] = True
    finally:
        if sealed_mode:
            try:
                post_target = capture_execution_target()
                artifact["execution_target_postcheck"] = post_target == execution_target
                if not artifact["execution_target_postcheck"]:
                    artifact["passed"] = False
                    artifact["execution_target_postcheck_error"] = (
                        "pod, GPU, sealed binary, or input identity changed during soundness"
                    )
            except Exception as error:  # preserve an artifact for infrastructure failures.
                artifact["passed"] = False
                artifact["execution_target_postcheck_error"] = str(error)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    if sealed_mode and artifact["execution_target_postcheck"] is not True:
        raise RuntimeError(artifact["execution_target_postcheck_error"])
    print(json.dumps(artifact, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
