"""Cost ledger: every lifecycle event and run appends one JSONL row, so
"what did we spend and what did it buy" is a query, not an archaeology dig.

Rows: {ts, event, pod_id, gpu, usd_hr, purpose, run_id?, note?}
Events: create | resume | stop | terminate | run
`report` reconstructs billable intervals per pod (create/resume -> stop/terminate)
and joins run rows for $/session; the bench ledger (loop/ledger.jsonl) holds the
MHz side of $/MHz-hr.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "ledger_costs.jsonl"


def append(event: str, **kw) -> None:
    row = {"ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"), "event": event}
    row.update({k: v for k, v in kw.items() if v is not None})
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")


def _rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def report() -> str:
    rows = _rows()
    if not rows:
        return f"ledger empty ({LEDGER})"
    open_at: dict[str, tuple[_dt.datetime, float, str]] = {}
    per_pod_usd: dict[str, float] = {}
    per_pod_hrs: dict[str, float] = {}
    gpu_of: dict[str, str] = {}
    now = _dt.datetime.now(_dt.UTC)
    still_open: list[str] = []

    for r in rows:
        ts = _dt.datetime.fromisoformat(r["ts"])
        pid = r.get("pod_id", "?")
        gpu_of.setdefault(pid, r.get("gpu", "?"))
        ev = r["event"]
        if ev in ("create", "resume"):
            open_at[pid] = (ts, float(r.get("usd_hr", 0.0)), r.get("purpose", ""))
        elif ev in ("stop", "terminate") and pid in open_at:
            t0, rate, _ = open_at.pop(pid)
            hrs = (ts - t0).total_seconds() / 3600
            per_pod_hrs[pid] = per_pod_hrs.get(pid, 0.0) + hrs
            per_pod_usd[pid] = per_pod_usd.get(pid, 0.0) + hrs * rate
    for pid, (t0, rate, _) in open_at.items():
        hrs = (now - t0).total_seconds() / 3600
        per_pod_hrs[pid] = per_pod_hrs.get(pid, 0.0) + hrs
        per_pod_usd[pid] = per_pod_usd.get(pid, 0.0) + hrs * rate
        still_open.append(pid)

    lines = [f"{'pod':<16} {'gpu':<26} {'hours':>7} {'usd':>8}  open?"]
    for pid in sorted(per_pod_usd, key=per_pod_usd.get, reverse=True):
        lines.append(
            f"{pid:<16} {gpu_of.get(pid, '?'):<26} {per_pod_hrs[pid]:>7.2f} "
            f"{per_pod_usd[pid]:>8.2f}  {'OPEN+BILLING' if pid in still_open else ''}"
        )
    total = sum(per_pod_usd.values())
    lines.append(f"{'TOTAL':<16} {'':<26} {sum(per_pod_hrs.values()):>7.2f} {total:>8.2f}")
    n_runs = sum(1 for r in rows if r["event"] == "run")
    lines.append(f"runs recorded: {n_runs}; ledger: {LEDGER}")
    return "\n".join(lines)
