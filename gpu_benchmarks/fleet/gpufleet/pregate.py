"""The local no-GPU battery — run before ANY pod money moves.

Everything provable on a laptop is proven here: both repos' unit suites (which
include the prove-accessor parity gate — the local stand-in for the GPU parity
run), the pie-bench binary compile, and format checks. `gpufleet run` refuses
to provision unless this passed recently (override with --skip-pregate).
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time
from pathlib import Path

STAMP = Path(__file__).resolve().parent.parent / ".pregate_ok.json"
FRESH_S = 6 * 3600

CHECKS: list[tuple[str, list[str], Path]] = []


def _init_checks(stwo: Path, stwo_cairo: Path) -> list[tuple[str, list[str], Path]]:
    prover = stwo_cairo / "stwo_cairo_prover"
    return [
        (
            "stwo-backend-cuda tests (37: isa/codegen/recording/registry)",
            ["cargo", "test", "-p", "stwo-backend-cuda"],
            stwo,
        ),
        (
            "stwo-cairo prover lib tests (40: incl. 8-test differential battery "
            "+ prove-accessor parity gate)",
            ["cargo", "test", "-p", "stwo-cairo-prover", "--release", "--lib"],
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
    results = []
    ok_all = True
    for name, argv, cwd in _init_checks(stwo, stwo_cairo):
        t0 = time.time()
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True,
            env={"RUST_MIN_STACK": "4194304", **__import__("os").environ},
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
    STAMP.write_text(json.dumps({
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "ok": ok_all,
        "results": results,
    }, indent=2))
    print(f"[pregate] {'ALL GREEN' if ok_all else 'FAILED'} -> {STAMP}")
    return ok_all


def is_fresh() -> bool:
    if not STAMP.exists():
        return False
    try:
        data = json.loads(STAMP.read_text())
        ts = _dt.datetime.fromisoformat(data["ts"])
        age = (_dt.datetime.now(_dt.UTC) - ts).total_seconds()
        return bool(data.get("ok")) and age < FRESH_S
    except (json.JSONDecodeError, KeyError, ValueError):
        return False
