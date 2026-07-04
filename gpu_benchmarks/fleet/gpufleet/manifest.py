"""Declarative run manifests: every GPU session is a TOML file with steps,
timeouts, artifacts, and machine-checked pass/fail criteria — so the expensive
part (a live pod) never waits on a human deciding what to type next, and a
"green" session means the CRITERIA passed, not that the logs looked plausible.

Manifest shape (TOML):

    [meta]
    name = "jit-witness-gate"
    stop_on_failure = true          # default true

    [env]                            # exported for every step
    STWO_POD = "1"

    [[step]]
    name = "selftest"
    cmd = "cd /workspace/... && ./gpu_bench --selftest"
    timeout_s = 1800
    gpu_bound = true                 # monitor flags idle GPU during this step
    artifacts = ["/workspace/out/selftest.json"]
    must_match = ["ALL LEGS OK"]     # regex, remote log
    must_not_match = ["mismatch=[1-9]"]
    [[step.assert]]                  # JSON assertions on a pulled artifact
    file = "selftest.json"
    path = "legs.0.n_mismatch"       # dotted path, list indices allowed
    op = "eq"                        # eq | ne | ge | le | exists | contains
    value = 0

Execution: heartbeat → run (streamed to results/<run>/<step>.log with the GPU
monitor sampling in parallel) → pull artifacts → evaluate criteria → next.
Summary lands in results/<run>/summary.json with full provenance.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .monitor import GpuMonitor
from .podctl import Endpoint, rsync, ssh_run, touch_heartbeat


@dataclass
class StepResult:
    name: str
    rc: int
    seconds: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    gpu_idle_alarm: bool = False


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(dotted)
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def _check_assert(spec: dict, artifact_dir: Path) -> str | None:
    """None = pass; else a human-readable failure reason."""
    fname, path, op = spec["file"], spec["path"], spec.get("op", "eq")
    want = spec.get("value")
    fpath = artifact_dir / fname
    if not fpath.exists():
        return f"assert {fname}:{path}: artifact missing"
    try:
        text = fpath.read_text()
        try:
            doc = json.loads(text)
        except ValueError:
            # Multi-line artifact (progress lines etc.): use the last JSON line.
            lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]
            doc = json.loads(lines[-1]) if lines else json.loads(text)
        got = _dig(doc, path)
    except (KeyError, IndexError, ValueError) as e:
        if op == "exists":
            return f"assert {fname}:{path}: missing ({e})"
        return f"assert {fname}:{path}: unreadable ({e})"
    ok = {
        "eq": lambda: got == want,
        "ne": lambda: got != want,
        "ge": lambda: float(got) >= float(want),
        "le": lambda: float(got) <= float(want),
        "exists": lambda: True,
        "contains": lambda: str(want) in str(got),
    }.get(op)
    if ok is None:
        return f"assert {fname}:{path}: unknown op {op!r}"
    if not ok():
        return f"assert {fname}:{path}: got {got!r}, want {op} {want!r}"
    return None


def _provenance(repos: dict[str, Path]) -> dict:
    out = {}
    for name, root in repos.items():
        try:
            rev = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout.strip()
            diff = subprocess.run(
                ["git", "-C", str(root), "diff", "HEAD"],
                capture_output=True, timeout=60, check=False,
            ).stdout
            out[name] = {
                "rev": rev,
                "diff_sha256": hashlib.sha256(diff).hexdigest()[:16],
                "dirty": bool(diff.strip()),
            }
        except Exception as e:
            out[name] = {"error": str(e)[:120]}
    return out


class ManifestRun:
    def __init__(self, manifest_path: Path, results_root: Path, repos: dict[str, Path]):
        self.spec = tomllib.loads(manifest_path.read_text())
        self.name = self.spec.get("meta", {}).get("name", manifest_path.stem)
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}.{self.name}"
        self.dir = results_root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.repos = repos
        self.stop_on_failure = bool(self.spec.get("meta", {}).get("stop_on_failure", True))

    def _env_prefix(self) -> str:
        env = self.spec.get("env", {})
        return "".join(f"export {k}={v!s}; " for k, v in env.items())

    def execute(self, ep: Endpoint, pod_meta: dict) -> dict:
        results: list[StepResult] = []
        t_run = time.time()
        for step in self.spec.get("step", []):
            name = step["name"]
            log = self.dir / f"{name}.log"
            timeout = float(step.get("timeout_s", 3600))
            print(f"[gpufleet] step {name} (timeout {timeout:.0f}s) -> {log}")
            touch_heartbeat(ep)
            mon = GpuMonitor(ep, self.dir / f"{name}.gpu.csv",
                             expect_busy=bool(step.get("gpu_bound", False)))
            mon.start()
            t0 = time.time()
            try:
                rc = ssh_run(
                    ep,
                    f"set -o pipefail; {self._env_prefix()}{step['cmd']}",
                    timeout=timeout,
                    log_file=log,
                )
            except subprocess.TimeoutExpired:
                rc = -9
            secs = time.time() - t0
            idle_alarm = mon.stop()
            touch_heartbeat(ep)

            res = StepResult(name=name, rc=rc, seconds=round(secs, 1),
                             passed=(rc == 0), gpu_idle_alarm=idle_alarm)
            if rc != 0:
                res.reasons.append(f"rc={rc}" + (" (timeout)" if rc == -9 else ""))

            # Pull artifacts before evaluating JSON asserts.
            for remote in step.get("artifacts", []):
                base = Path(remote).name
                if rsync(ep, remote, str(self.dir / base), pull=True) == 0:
                    res.artifacts.append(base)
                else:
                    res.passed = False
                    res.reasons.append(f"artifact pull failed: {remote}")

            text = log.read_text(errors="replace") if log.exists() else ""
            # Criteria see OUTPUT only, never the echoed command text.
            if "--8<-- output --8<--" in text:
                text = text.split("--8<-- output --8<--", 1)[1]
            for pat in step.get("must_match", []):
                if not re.search(pat, text, re.M):
                    res.passed = False
                    res.reasons.append(f"must_match failed: {pat!r}")
            for pat in step.get("must_not_match", []):
                if re.search(pat, text, re.M):
                    res.passed = False
                    res.reasons.append(f"must_not_match hit: {pat!r}")
            for spec in step.get("assert", []):
                why = _check_assert(spec, self.dir)
                if why:
                    res.passed = False
                    res.reasons.append(why)
            if idle_alarm and step.get("gpu_bound", False):
                res.reasons.append("GPU idle alarm during gpu_bound step")

            results.append(res)
            state = "PASS" if res.passed else "FAIL"
            print(f"[gpufleet]   {state} in {secs:.0f}s"
                  + (f" — {'; '.join(res.reasons)}" if res.reasons else ""))
            if not res.passed and self.stop_on_failure:
                break

        summary = {
            "run_id": self.run_id,
            "manifest": self.name,
            "pod": pod_meta,
            "provenance": _provenance(self.repos),
            "steps": [vars(r) for r in results],
            "passed": all(r.passed for r in results) and bool(results),
            "wall_s": round(time.time() - t_run, 1),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
