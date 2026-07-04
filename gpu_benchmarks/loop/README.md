# Stwo GPU proving — fast-feedback benchmark loop

One command turns a code change into trustworthy, fully-provenanced SN-PIE benchmark
numbers on the RunPod pod, appended to a persistent ledger. Minutes, not hours, and
impossible to confuse which code produced which number.

```
cd gpu_benchmarks/loop
./bench_loop.sh                 # gate + benchmark SN_PIE_2, 2 reps, on the pod
./ledger_report.py              # read the ledger back as a table
```

## What one run does

```
(0) resolve pod  the pod address is NEVER hardcoded: `runpodctl ssh info $POD_ID`
                 resolves ip/port/key at runtime (pod id from BENCH_POD_ID or
                 pod.conf); falls back to pod.conf FALLBACK_* with a loud warning
(a) provenance   both repos' git HEAD + sha256 of the working diff  (see below)
(b) sync         rsync stwo + stwo-cairo to the pod (fast delta), then re-apply
                 the pod's Cargo.toml [patch] rewrite that rsync just clobbered
(c) build        incremental `cargo build ... gpu_bench --features pie-bench`;
                 a compile error aborts LOUDLY with the tail of the build log and
                 writes NO ledger entry
(d) GATE         10-transfer PIE, CUDA prove + verify, FIRST — under the SAME
                 BENCH_ENV as the benchmarks. A failed verify or a crash aborts
                 the whole run and writes a `gate_failed` ledger entry
(e) benchmark    the selected PIE(s), STWO_BENCH_TRACE=json, STWO_JIT_LOG=1
                 (always on — cold JIT-compile visibility catches hangs),
                 RUST_MIN_STACK=4M, --reuse-input, launched detached
                 (nohup+setsid) on the pod and polled with STALL DETECTION
(f) pull         each run's stdout (main record + per-rep phase_totals) back
(g) ledger       one JSON line per run appended to `ledger.jsonl` (incl. bench_env)
(h) summary      useful_mhz per run + delta vs the previous ledger entry for the
                 SAME run_name, SAME pod_gpu, and SAME bench_env
```

## Usage

```
./bench_loop.sh [--pie {1|2|3|4|10t}] [--reps N] [--full] [--simd]
                [--skip-sync] [--gate-only]
```

| Flag          | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `--pie SEL`   | Which PIE to benchmark: `1..4` = `SN_PIE_<n>.zip`, `10t` = 10-transfer. Default `2`. |
| `--reps N`    | Reps per run (warm-best reported). Default `2`.                         |
| `--full`      | Also benchmark `SN_PIE_1/3/4` (CUDA) **and** run the rotate-mode fleet (pipelined stream over all four PIEs — the production one-pod-proving-a-block-stream shape). |
| `--simd`      | Add a same-host SIMD run of the selected PIE (CPU baseline).            |
| `--skip-sync` | Skip rsync **and** build; benchmark the binary already on the pod.      |
| `--gate-only` | Run only the correctness gate, then exit.                              |

### Environment

| Var             | Default    | Meaning                                                     |
|-----------------|------------|-------------------------------------------------------------|
| `BENCH_POD_ID`  | (pod.conf) | Pod id, overrides `POD_ID` in `pod.conf`.                   |
| `BENCH_ENV`     | (empty)    | `"K=V K=V ..."` exported verbatim into **every** gpu_bench invocation (gate included) and **recorded in every ledger entry**. For debug bisects: `STWO_CUDA_DISABLE_STREAMS`, `STWO_CUDA_MEMORY_WITNESS`, `STWO_CUDA_DEBUG_SYNC`, `CUDA_LAUNCH_BLOCKING`, `STWO_JIT_LOG`, ... Values must not contain spaces. |
| `DRY_RUN=1`     | `0`        | Echo every ssh/rsync instead of executing, and fabricate run output so the provenance → ledger → summary path still runs for real. Use to trace logic offline. |
| `FAKE_STALL`    | (unset)    | (DRY_RUN only) name of a run to simulate as stalled — exercises the stall → evidence → ledger → abort path. |
| `POLL_INTERVAL` | `15`       | Seconds between pod poll checks.                            |
| `STALL_SECS`    | `600`      | Stall detector window (see below).                          |
| `MAX_WAIT`      | `10800`    | Hard cap on waiting for one run (3h backstop).              |
| `FLEET_REPS` / `FLEET_DEPTH` / `FLEET_PRODUCERS` | `8` / `3` / `4` | Fleet (rotate) pipeline knobs. |

## Pod resolution (`pod.conf`)

Community pods churn; a stale IP in a script is a silent foot-gun. The pod address is
resolved at runtime: `runpodctl ssh info $POD_ID` → ip/port/key. The pod id comes from
`BENCH_POD_ID` (env) or `POD_ID` in `pod.conf`. If runpodctl fails, the script falls
back to the `FALLBACK_HOST/PORT/KEY` values in `pod.conf` **with a loud warning** that
they may be stale. When the pod moves: update `POD_ID` in `pod.conf` (and refresh the
FALLBACK_* values while you're there).

## Stall detection

Observed failure mode: process alive forever, zero output (e.g. a JIT-compile hang).
Waiting for `MAX_WAIT` is too slow for a fast loop. Each poll (one ssh round-trip)
reports the run's stderr size and the GPU utilization; if **both are unchanged for
`STALL_SECS`** (default 600 s) the run is declared stalled:

1. evidence is captured — `/proc/<pid>/task/*/status` thread states + the last 30
   stderr lines — into `results/<stamp>.<run>.stall.txt`,
2. the whole process group is killed (TERM, then KILL),
3. a `status:"stalled"` ledger entry is written **with the evidence embedded**
   (`stall_evidence`), and
4. the loop aborts.

A mystery hang becomes a self-documenting ledger row.

## Reading the ledger

```
./ledger_report.py                     # full table
./ledger_report.py --run SN_PIE_2      # one run_name
./ledger_report.py --phases SN_PIE_2   # per-phase total_ms trend across entries
```

Table columns: `ts`, `revs` (stwo/cairo short + `*` when the tree was dirty),
`run_name` (`!` = the run had a non-default `BENCH_ENV` — a debug/bisect number, never
compare it with clean numbers), `useful_mhz` (a trailing `~` means it is
`sustained_useful_mhz` from a pipelined fleet run), `vram_gb`, and `delta` vs the
previous **same-run, same-pod, same-bench_env** entry.

## Provenance model — why a number is never ambiguous

Every ledger entry records, at the moment of the run:

- `stwo_rev`, `cairo_rev` — `git rev-parse HEAD` of each repo.
- `stwo_dirty`, `cairo_dirty` — `"clean"` if the working tree matches HEAD, otherwise
  the first 16 hex of `sha256(git diff HEAD)`. This makes a run reproducible **even with
  uncommitted work**: the same rev + same dirty hash == the same source. A changed
  number with an unchanged (rev, dirty) pair on the same pod is a real signal; a changed
  dirty hash tells you the source moved.
- `pod_gpu` — the GPU reported by the pod (currently `NVIDIA GeForce RTX 3090`).
- `bench_env` — the exact `BENCH_ENV` the run executed under (empty for clean runs).
  A number produced with debug sync or disabled streams is permanently marked as such.
- `record` — the harness's own self-describing JSON (security config, host fingerprint,
  timings, VRAM, `useful_mhz`, …). `phase_totals` — the per-rep `STWO_BENCH_TRACE=json`
  span breakdown. Fleet runs additionally carry a `pipeline` object (sustained numbers).
- `status` — `ok`, `gate_failed`, `run_failed`, or `stalled` (with `stall_evidence`).

**Caveat:** `git diff` does not capture *untracked* files. Add new files to the index
(`git add -N`) if you need them reflected in the dirty hash.

## The gate-first rule

The correctness gate (10-transfer PIE, CUDA prove **and** verify) always runs before any
performance benchmark, **under the same `BENCH_ENV` as the benchmarks** — a kill switch
that changes prover behavior must be correctness-gated too. A failed verify or a crash
aborts the run and records a `gate_failed` entry. **Performance is never reported from a
build that failed the gate** — a fast but wrong prover is worthless, and the ledger must
never contain a misleading fast number attached to a broken build.

## ⚠️ Only same-pod comparisons are meaningful

`useful_mhz` depends heavily on the host (GPU model, CPU, memory bandwidth, and shared
tenancy on a community pod). The delta computed by both `bench_loop.sh` and
`ledger_report.py` is deliberately scoped to the **same `run_name` AND the same
`pod_gpu` AND the same `bench_env`**. Do **not** read a delta across different GPUs — it
is host variance, not a code effect. When the pod changes, start a fresh comparison
baseline.

## Pod / environment specifics

- Pod identity: `pod.conf` (`POD_ID` + fallback endpoint); resolution via `runpodctl`.
- Repos on the pod: `/workspace/stwo`, `/workspace/stwo-cairo`.
- The pod's `stwo_cairo_prover/Cargo.toml` `[patch]` points at `/workspace/stwo`; the
  local copy points at `/Users/...`. rsync overwrites it every sync, so the script
  re-applies the `sed` rewrite immediately after every sync.
- PIE inputs (`SN_PIE_*.zip`, the 10-transfer zip) already live on the pod and are
  **never** synced (rsync excludes them).
- `target/` and `.git/` are excluded from rsync; the build is incremental (~1-2 min).
- Run scratch (launcher scripts, stdout/err, pid/pgid/rc sentinels, build log) lives in
  `/workspace/bench_loop_runs` on the pod, outside the repo tree.

Do not run `bench_loop.sh` while another benchmark round is active on the pod — the
sync + build would disturb the in-flight round's source tree.

## Files

| File               | Role                                                            |
|--------------------|-----------------------------------------------------------------|
| `bench_loop.sh`    | The one command (steps 0 + a–h above).                          |
| `ledger_report.py` | Read `ledger.jsonl` as a table / phase trend (stdlib only).     |
| `pod.conf`         | Pod id + fallback endpoint (edit when the pod moves).           |
| `ledger.jsonl`     | The persistent ledger (created on first run; one JSON per line).|
| `results/`         | Raw per-run stdout/err/rc + stall evidence pulled from the pod. |
