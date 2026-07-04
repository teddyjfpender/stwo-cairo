#!/usr/bin/env python3
"""ledger_report.py — read loop/ledger.jsonl and print a human-readable table.

Columns: ts, revs (stwo/cairo short + dirty marker), run_name ('!' = run had a
non-default BENCH_ENV — debug/bisect, never compare with clean numbers), useful_mhz,
vram_peak_gb, delta-vs-prev (same run_name AND same pod_gpu AND same bench_env — the
only meaningful comparison; community-host variance makes cross-pod deltas noise).

For a pipelined (fleet/rotate) entry the reported MHz is sustained_useful_mhz.

Usage:
  ./ledger_report.py [--ledger PATH] [--run RUN_NAME] [--phases RUN_NAME]

  (no args)        table of every entry
  --run NAME       restrict the table to one run_name
  --phases NAME    per-phase total_ms trend across entries for one run_name
                   (sums total_ms over that entry's reps)

Stdlib only.
"""
import argparse
import json
import os
import sys

DEFAULT_LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")


def load(path):
    entries = []
    if not os.path.exists(path):
        sys.exit(f"ledger not found: {path}")
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {n}: {e}", file=sys.stderr)
    return entries


def metric_of(entry):
    """The comparison metric: sustained useful MHz for a pipelined run, else the
    per-run useful MHz from the main record."""
    pipe = entry.get("pipeline")
    if pipe and pipe.get("sustained_useful_mhz") is not None:
        return pipe["sustained_useful_mhz"], True
    rec = entry.get("record") or {}
    return rec.get("useful_mhz"), False


def short(rev):
    return (rev or "?")[:8]


def revs_col(e):
    s = short(e.get("stwo_rev"))
    if e.get("stwo_dirty") not in (None, "clean"):
        s += "*"
    c = short(e.get("cairo_rev"))
    if e.get("cairo_dirty") not in (None, "clean"):
        c += "*"
    return f"{s}/{c}"


def table(entries):
    # delta computed per (run_name, pod_gpu, bench_env) in chronological (file) order:
    # cross-pod deltas are host variance, and a debug-env number (BENCH_ENV set) must
    # never be compared with a clean one.
    prev = {}
    rows = []
    for e in entries:
        run = e.get("run_name", "?")
        gpu = e.get("pod_gpu", "?")
        benv = e.get("bench_env") or ""
        status = e.get("status", "ok")
        m, sustained = metric_of(e)
        rec = e.get("record") or {}
        vram = rec.get("vram_peak_gb")
        key = (run, gpu, benv)
        if status != "ok":
            delta = f"[{status}]"
        elif m is None:
            delta = "n/a"
        elif key not in prev:
            delta = "—"
        else:
            p = prev[key]
            delta = f"{(m - p) / p * 100.0:+.1f}%" if p else "—"
        if status == "ok" and m is not None:
            prev[key] = m
        rows.append({
            "ts": e.get("ts", "?"),
            "revs": revs_col(e),
            "run": run + ("!" if benv else ""),
            "mhz": ("" if m is None else f"{m:.3f}") + ("~" if sustained else ""),
            "vram": "" if vram is None else f"{vram:.2f}",
            "delta": delta,
        })

    hdr = ("ts", "revs", "run_name", "useful_mhz", "vram_gb", "delta")
    widths = [max(len(hdr[i]), *(len(str(r[k])) for r in rows)) if rows else len(hdr[i])
              for i, k in enumerate(["ts", "revs", "run", "mhz", "vram", "delta"])]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*hdr))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(r["ts"], r["revs"], r["run"], r["mhz"], r["vram"], r["delta"]))
    print("\n(~ = sustained_useful_mhz from a pipelined run; * = dirty working tree;")
    print(" ! = non-default BENCH_ENV (debug/bisect run) — never compare with clean numbers;")
    print(" delta is vs the previous SAME-run_name SAME-pod SAME-bench_env entry only.)")


def phases(entries, run):
    rows = []
    names = []
    for e in entries:
        if e.get("run_name") != run:
            continue
        agg = {}
        for rep in e.get("phase_totals", []):
            for name, v in (rep.get("phase_totals") or {}).items():
                agg[name] = agg.get(name, 0.0) + float(v.get("total_ms", 0.0))
        if not agg:
            continue
        rows.append((e.get("ts", "?"), revs_col(e), agg))
        for n in agg:
            if n not in names:
                names.append(n)
    if not rows:
        sys.exit(f"no phase_totals found for run_name '{run}'")
    names.sort()
    print(f"per-phase total_ms trend for run_name '{run}' (summed over reps):\n")
    w = max(len(n) for n in names)
    header = f"{'phase':<{w}}  " + "  ".join(f"{ts[-9:]:>12}" for ts, _, _ in rows)
    print(header)
    print(f"{'revs':<{w}}  " + "  ".join(f"{rv:>12}" for _, rv, _ in rows))
    print("-" * len(header))
    for name in names:
        cells = "  ".join(f"{agg.get(name, 0.0):>12.1f}" for _, _, agg in rows)
        print(f"{name:<{w}}  {cells}")


def main():
    ap = argparse.ArgumentParser(description="Report the bench_loop ledger.")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--run", help="restrict the table to one run_name")
    ap.add_argument("--phases", metavar="RUN_NAME", help="per-phase total_ms trend for a run_name")
    args = ap.parse_args()

    entries = load(args.ledger)
    if args.phases:
        phases(entries, args.phases)
    else:
        if args.run:
            entries = [e for e in entries if e.get("run_name") == args.run]
        table(entries)


if __name__ == "__main__":
    main()
