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

RESIDENT_FIXTURE = re.compile(r'STRICT_RESIDENT_FIXTURE: &str = "([^"]+)"')


GATES = (
    (
        "cuda_link_required",
        ("cargo", "test", "-p", "stwo-backend-cuda", "--test", "cuda_required"),
        3,
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
        1,
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
        "prepared_relation_eager_capture_reference",
        (
            "cargo",
            "test",
            "-p",
            "stwo-backend-cuda",
            "--test",
            "prepared_relation_native",
        ),
        1,
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
        1,
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
            "--test",
            "prepared_composition_native",
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
        4,
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
    process = subprocess.run(
        actual_command + ("--", "--nocapture"),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    counts = [int(match.group(1)) for match in RESULT.finditer(process.stdout)]
    executed = max(counts, default=0)
    passed = process.returncode == 0 and executed == expected
    record = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "rust_min_stack": int(env["RUST_MIN_STACK"]),
        "exit_code": process.returncode,
        "executed_tests": executed,
        "required_tests": expected,
        "passed": passed,
    }
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stwo = args.stwo.resolve()
    stwo_cairo = args.stwo_cairo.resolve()
    if not (stwo / "Cargo.toml").is_file():
        raise SystemExit(f"not a stwo workspace: {stwo}")
    if not (stwo_cairo / "stwo_cairo_prover" / "Cargo.toml").is_file():
        raise SystemExit(f"not a stwo-cairo workspace: {stwo_cairo}")

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

    artifact = {
        "schema": "stwo.cuda.soundness-gate.v2",
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
        "gates": [],
        "passed": False,
    }
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
