"""The local no-GPU battery — run before ANY pod money moves.

Everything provable on a laptop is proven here: both repos' unit suites (which
include the prove-accessor parity gate — the local stand-in for the GPU parity
run), the pie-bench binary compile, and format checks. Spend-bearing commands
refuse to start or resume compute unless this passed recently for the exact source.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .source_projection import projection_identity

STAMP = Path(__file__).resolve().parent.parent / ".pregate_ok.json"
FRESH_S = 6 * 3600
SN_INPUT_ENV = "STWO_SN_ADAPTED_DIR"
SN_INPUT_NAMES = tuple(f"SN_PIE_{index}.adapted.bin" for index in range(1, 5))
SN2_VERTICAL_SCOPE = "sn2-vertical-indicative-v1"
SN2_VERTICAL_RECIPE = Path(
    "gpu_benchmarks/loop/recipes/sn2_compiled_vertical_checkpoint.phases"
)
SN2_VERTICAL_ABBA_RECIPE = Path(
    "gpu_benchmarks/loop/recipes/sn2_compiled_vertical_abba.phases"
)
_SN2_VERTICAL_RECIPES = {
    SN2_VERTICAL_RECIPE: {
        "phases": (
            "phase ambient_override_gate checkpoint_reject_ambient_overrides",
            "phase source_input_identity checkpoint_source_input_identity iteration",
            "phase hardware_identity vertical_hardware_identity",
            "phase build vertical_build",
            "phase sn2_compiled_vertical vertical_run",
            "phase sn2_compiled_vertical_validate vertical_validate",
        ),
        "required": ("validate_sn2_vertical_checkpoint.py",),
        "test": "gpu_benchmarks.test_validate_sn2_vertical_checkpoint",
        "control": (
            "gpu_benchmarks/validate_sn2_vertical_checkpoint.py",
            "gpu_benchmarks/test_validate_sn2_vertical_checkpoint.py",
        ),
    },
    SN2_VERTICAL_ABBA_RECIPE: {
        "phases": (
            "phase ambient_override_gate checkpoint_reject_ambient_overrides",
            "phase source_input_identity checkpoint_source_input_identity iteration",
            "phase hardware_identity vertical_abba_hardware_identity",
            "phase composition_address_free_test vertical_abba_focused_test",
            "phase build vertical_abba_build",
            "phase composition_split_log24_native vertical_abba_split_native_gate",
            "phase composition_split_commit_log24_native vertical_abba_split_commit_native_gate",
            "phase fused_compiled_vertical_pregate vertical_abba_run new pregate 2",
            "phase fused_compiled_vertical_pregate_validate vertical_abba_validate_fused_pregate",
            "phase abba_old_1 vertical_abba_run old 1",
            "phase abba_new_1 vertical_abba_run new 1",
            "phase abba_new_2 vertical_abba_run new 2",
            "phase abba_old_2 vertical_abba_run old 2",
            "phase abba_proof_identity vertical_abba_proof_identity",
            "phase abba_validate vertical_abba_validate",
        ),
        "required": (
            "cmp -s",
            "validate_sn2_vertical_abba.py",
            "address_free_structure_ignores_only_rebound_source_slots",
            "STWO_COMPOSITION_SPLIT_LOGS=24",
            "composition_split_native",
            "STWO_COMPOSITION_COMMIT_LOGS=24",
            "composition_split_commit_native",
            "STWO_COMPOSITION_SPLIT_NATIVE_RECEIPT_JSON",
            "STWO_COMPOSITION_SPLIT_COMMIT_NATIVE_RECEIPT_JSON",
            "--reps 2",
        ),
        "test": "gpu_benchmarks.test_validate_sn2_vertical_abba",
        "control": (
            "gpu_benchmarks/validate_replacement_v1_reuse.py",
            "gpu_benchmarks/validate_sn2_vertical_checkpoint.py",
            "gpu_benchmarks/test_validate_sn2_vertical_checkpoint.py",
            "gpu_benchmarks/validate_sn2_vertical_abba.py",
            "gpu_benchmarks/test_validate_sn2_vertical_abba.py",
        ),
    },
}
_SN2_VERTICAL_REQUIRED = (
    "export REPLACEMENT_SN2_MODE=vertical",
    "--compiled-composition-vertical-checkpoint",
    "--require-proof-byte-equal",
    "--require-simd-reference-byte-equal",
    "--require-proof-mutation-rejected",
)
_SN2_VERTICAL_CAIRO_CONTROL = (
    "gpu_benchmarks/fleet/gpufleet/pregate.py",
    "gpu_benchmarks/fleet/gpufleet/__main__.py",
    "gpu_benchmarks/fleet/gpufleet/source_projection.py",
    "gpu_benchmarks/fleet/gpufleet/lease_policy.py",
    "gpu_benchmarks/loop/pod_run.sh",
    "gpu_benchmarks/loop/recipes/replacement_v1_sn2_common.sh",
    "gpu_benchmarks/pie/SHA256SUMS",
    "gpu_benchmarks/pie/ADAPTED_SHA256SUMS",
)
_SN2_VERTICAL_STWO_CONTROL = (
    "crates/backend-cuda-kernels/cuda/generated/aot_manifest.json",
)


def _write_stamp(
    ok: bool,
    results: list[dict],
    source_identity: dict | None = None,
    input_identity: dict | None = None,
) -> None:
    receipt = {
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "ok": ok,
        "results": results,
    }
    if source_identity is not None:
        receipt["source_identity"] = source_identity
    if input_identity is not None:
        receipt["input_identity"] = input_identity
    STAMP.write_text(json.dumps(receipt, indent=2) + "\n")


def _repo_identity(repository: Path) -> dict[str, str]:
    return projection_identity(repository)


def _source_identity(stwo: Path, stwo_cairo: Path) -> dict[str, dict[str, str]]:
    return {
        "stwo": _repo_identity(stwo),
        "stwo_cairo": _repo_identity(stwo_cairo),
    }


def is_sn2_vertical_recipe(recipe: Path, stwo_cairo: Path) -> bool:
    """True only for a canonical non-formal vertical checkpoint recipe."""
    try:
        resolved = recipe.expanduser().resolve(strict=True)
        return any(
            resolved == (stwo_cairo / candidate).resolve(strict=True)
            for candidate in _SN2_VERTICAL_RECIPES
        )
    except OSError:
        return False


def _tracked_control_is_clean(repository: Path, paths: tuple[str, ...]) -> bool:
    commands = (
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
    )
    return all(
        subprocess.run(
            command, cwd=repository, capture_output=True, check=False
        ).returncode
        == 0
        for command in commands
    )


def _sn2_vertical_identity(
    recipe: Path, stwo: Path, stwo_cairo: Path
) -> dict[str, object]:
    if not is_sn2_vertical_recipe(recipe, stwo_cairo):
        raise ValueError("not the canonical SN2 vertical checkpoint recipe")
    resolved = recipe.expanduser().resolve(strict=True)
    relative = next(
        candidate
        for candidate in _SN2_VERTICAL_RECIPES
        if resolved == (stwo_cairo / candidate).resolve(strict=True)
    )
    spec = _SN2_VERTICAL_RECIPES[relative]
    cairo_control = (
        *_SN2_VERTICAL_CAIRO_CONTROL,
        relative.as_posix(),
        *spec["control"],
    )
    if not (
        _tracked_control_is_clean(stwo_cairo, cairo_control)
        and _tracked_control_is_clean(stwo, _SN2_VERTICAL_STWO_CONTROL)
    ):
        raise ValueError("SN2 vertical control plane differs from tracked HEAD")
    raw = recipe.read_bytes()
    source = raw.decode("utf-8")
    phases = tuple(line for line in source.splitlines() if line.startswith("phase "))
    if phases != spec["phases"]:
        raise ValueError("SN2 vertical checkpoint phase contract drifted")
    required = (*_SN2_VERTICAL_REQUIRED, *spec["required"])
    if any(fragment not in source for fragment in required):
        raise ValueError("SN2 vertical checkpoint safety contract is incomplete")
    return {
        "scope": SN2_VERTICAL_SCOPE,
        "recipe": {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "source_identity": _source_identity(stwo, stwo_cairo),
    }


def admit_sn2_vertical(
    recipe: Path, stwo: Path, stwo_cairo: Path
) -> dict[str, object] | None:
    """Run the bounded no-Cargo admission for the indicative SN2 checkpoint."""
    try:
        identity = _sn2_vertical_identity(recipe, stwo, stwo_cairo)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"[pregate] FAIL SN2 vertical identity: {error}")
        return None

    common = (
        stwo_cairo
        / "gpu_benchmarks/loop/recipes/replacement_v1_sn2_common.sh"
    )
    checks = (
        ("SN2 vertical shell syntax", ["bash", "-n", str(common), str(recipe)]),
        (
            "SN2 vertical result validator",
            [
                sys.executable,
                "-m",
                "unittest",
                _SN2_VERTICAL_RECIPES[
                    Path(identity["recipe"]["path"])
                ]["test"],
            ],
        ),
    )
    for name, argv in checks:
        result = subprocess.run(
            argv, cwd=stwo_cairo, capture_output=True, text=True, check=False
        )
        if result.returncode:
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
            print(f"[pregate] FAIL {name}\n{tail}")
            return None
        print(f"[pregate] PASS {name}")
    try:
        stable = identity == _sn2_vertical_identity(recipe, stwo, stwo_cairo)
    except (OSError, UnicodeDecodeError, ValueError):
        stable = False
    if not stable:
        print("[pregate] FAIL SN2 vertical recipe or source changed during admission")
        return None
    print("[pregate] ALL GREEN SN2 vertical indicative admission")
    return identity


def sn2_vertical_is_current(
    receipt: dict[str, object], recipe: Path, stwo: Path, stwo_cairo: Path
) -> bool:
    try:
        return receipt == _sn2_vertical_identity(recipe, stwo, stwo_cairo)
    except (OSError, UnicodeDecodeError, ValueError):
        return False


def _admit_sn_inputs(stwo_cairo: Path) -> tuple[list[Path], dict[str, str], str]:
    configured = os.environ.get(SN_INPUT_ENV)
    if not configured:
        raise ValueError(
            f"set {SN_INPUT_ENV} to the sealed SN1-SN4 adapted-input directory"
        )
    directory = Path(configured).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"{SN_INPUT_ENV} is not a directory: {directory}")

    manifest = stwo_cairo / "gpu_benchmarks/pie/ADAPTED_SHA256SUMS"
    rows: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValueError(f"malformed adapted-input manifest line {line_number}")
        digest, name = fields
        if name in rows:
            raise ValueError(f"duplicate adapted-input manifest entry: {name}")
        rows[name] = digest
    if set(rows) != set(SN_INPUT_NAMES):
        raise ValueError(
            f"adapted-input manifest names must be exactly {list(SN_INPUT_NAMES)}"
        )

    paths = []
    observed = {}
    for name in SN_INPUT_NAMES:
        path = (directory / name).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"adapted input is not a regular file: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != rows[name]:
            raise ValueError(f"adapted-input SHA-256 mismatch: {name}")
        paths.append(path)
        observed[name] = digest
    return paths, observed, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _init_checks(
    stwo: Path, stwo_cairo: Path, sn_inputs: list[Path]
) -> list[tuple[str, list[str], Path]]:
    prover = stwo_cairo / "stwo_cairo_prover"
    kernel_emit = [
        "cargo",
        "run",
        "--profile",
        "witness-opt-1",
        "-p",
        "stwo-cairo-gpu-prover",
        "--bin",
        "kernel_emit",
        "--features",
        "emit-tools",
        "--",
        "--stwo-root",
        str(stwo.resolve()),
    ]
    for path in sn_inputs:
        kernel_emit.extend(("--input-bincode", str(path)))
    kernel_emit.append("--check")
    return [
        (
            "stwo-backend-cuda host-safe library tests",
            ["cargo", "test", "-p", "stwo-backend-cuda", "--lib"],
            stwo,
        ),
        (
            "stwo-cairo prover lib tests (40: incl. 8-test differential battery "
            "+ prove-accessor parity gate)",
            ["cargo", "test", "-p", "stwo-cairo-prover", "--release", "--lib"],
            prover,
        ),
        (
            "kernel_emit --check (sealed SN1-SN4 AOT union drift gate)",
            kernel_emit,
            prover,
        ),
        (
            "schedule_emit --check (generated schedule table drift gate)",
            ["cargo", "run", "--manifest-path", "tools/schedule_emit/Cargo.toml",
             "--", "--prover-root", ".", "--check"],
            prover,
        ),
        (
            "gpu-prover lib tests (schedule/flags) + parity gate compile",
            ["cargo", "test", "-p", "stwo-cairo-gpu-prover", "--release", "--lib"],
            prover,
        ),
        (
            "gpu_bench compiles (pie-bench)",
            ["cargo", "check", "--release", "-p", "stwo-cairo-gpu-prover",
             "--bin", "gpu_bench", "--features", "pie-bench"],
            prover,
        ),
        (
            "stwo fmt",
            ["cargo", "fmt", "--check", "-p", "stwo-backend-cuda", "-p", "stwo"],
            stwo,
        ),
    ]


def run(stwo: Path, stwo_cairo: Path) -> bool:
    results: list[dict] = []
    # Invalidate any prior green receipt before admission or a long subprocess.
    _write_stamp(False, results)
    admission_started = time.time()
    try:
        sn_inputs, input_hashes, manifest_hash = _admit_sn_inputs(stwo_cairo)
    except (OSError, ValueError) as error:
        results.append(
            {
                "name": "sealed SN1-SN4 input admission",
                "ok": False,
                "seconds": round(time.time() - admission_started, 1),
                "error": str(error),
            }
        )
        _write_stamp(False, results)
        print(f"[pregate] FAIL sealed SN1-SN4 input admission: {error}")
        return False
    results.append(
        {
            "name": "sealed SN1-SN4 input admission",
            "ok": True,
            "seconds": round(time.time() - admission_started, 1),
            "manifest_sha256": manifest_hash,
            "inputs": input_hashes,
        }
    )
    input_identity = {
        "manifest_sha256": manifest_hash,
        "inputs": input_hashes,
    }
    try:
        source_identity = _source_identity(stwo, stwo_cairo)
    except (OSError, ValueError) as error:
        results.append(
            {"name": "source projection admission", "ok": False, "error": str(error)}
        )
        _write_stamp(False, results, input_identity=input_identity)
        print(f"[pregate] FAIL tracked source admission: {error}")
        return False

    ok_all = True
    for name, argv, cwd in _init_checks(stwo, stwo_cairo, sn_inputs):
        t0 = time.time()
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True,
            env={
                **os.environ,
                "RUST_MIN_STACK": "33554432",
                "CARGO_PROFILE_WITNESS_OPT_1_DEBUG": "0",
            },
            check=False,
        )
        ok = proc.returncode == 0
        ok_all &= ok
        secs = time.time() - t0
        print(f"[pregate] {'PASS' if ok else 'FAIL'} ({secs:5.1f}s) {name}")
        if not ok:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            print(tail)
        results.append({"name": name, "ok": ok, "seconds": round(secs, 1)})
    try:
        final_source_identity = _source_identity(stwo, stwo_cairo)
        _, final_input_hashes, final_manifest_hash = _admit_sn_inputs(stwo_cairo)
        final_input_identity = {
            "manifest_sha256": final_manifest_hash,
            "inputs": final_input_hashes,
        }
        stable = (
            final_source_identity == source_identity
            and final_input_identity == input_identity
        )
    except (OSError, ValueError) as error:
        stable = False
        results.append(
            {"name": "pregate identity recheck", "ok": False, "error": str(error)}
        )
    if not stable:
        ok_all = False
        if not any(
            result["name"] == "pregate identity recheck" for result in results
        ):
            results.append(
                {
                    "name": "pregate identity recheck",
                    "ok": False,
                    "error": "source projection or sealed input identity changed during pregate",
                }
            )
    else:
        results.append({"name": "pregate identity recheck", "ok": True})
    _write_stamp(ok_all, results, source_identity, input_identity)
    print(f"[pregate] {'ALL GREEN' if ok_all else 'FAILED'} -> {STAMP}")
    return ok_all


def is_fresh(stwo: Path, stwo_cairo: Path) -> bool:
    if not STAMP.exists():
        return False
    try:
        data = json.loads(STAMP.read_text())
        ts = _dt.datetime.fromisoformat(data["ts"])
        age = (_dt.datetime.now(_dt.UTC) - ts).total_seconds()
        if not data.get("ok") or age >= FRESH_S:
            return False
        _, input_hashes, manifest_hash = _admit_sn_inputs(stwo_cairo)
        return (
            data.get("source_identity") == _source_identity(stwo, stwo_cairo)
            and data.get("input_identity") == {
                "manifest_sha256": manifest_hash,
                "inputs": input_hashes,
            }
        )
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        return False
