# Fleet orchestrator — the 10–20 MHz aggregate target

This fleet proves **whole, independent SN PIEs** on each worker. It scales block-stream
throughput by assigning complete proofs to separate GPUs; it does not split one proof
across cards, and aggregate useful MHz is not a single-proof latency claim. ZisK,
Airbender, and other EVM-prover fleets are architectural reference points only: their
workloads and published results are not evidence about this SN PIE prover.

The latest replacement-v1 SN2 diagnostic measured **3.962227 useful MHz on one H100**.
That does not qualify a consumer-card rate. This directory measures whether a fleet of
independently admitted workers can reach the **10–20 MHz aggregate target**, along with
total \$/hr and **\$/MHz-hr**; it does not assume that target has already been reached.

It implements no prover arithmetic, but it does select and launch `gpu_bench`, so the
backend selection and emitted record are sealed explicitly. Its correctness dependency
is the standard per-pod correctness gate (the loop's 10-transfer PIE CUDA prove+verify),
which is ON by default.

```
cd gpu_benchmarks/fleet
./pod_provision.sh create --gpu "NVIDIA GeForce RTX 4090" --count 6   # stand up pods
$EDITOR fleet.conf                                                    # add their ids + $/hr
./fleet.sh --prep                                                     # sync+build+gate, then benchmark
cat fleet_report.json                                                 # the aggregate number
```

## The three files

| File                | Role                                                                       |
|---------------------|----------------------------------------------------------------------------|
| `fleet.sh`          | Launch a rotate-mode stream on every pod concurrently, poll with per-pod stall detection, aggregate useful MHz + \$/hr + \$/MHz-hr, write `fleet_report.json` + a human table. |
| `pod_provision.sh`  | `runpodctl` create/list/terminate/ssh-info helpers, GPU-type arg, and the **CPU-cores-first** procurement note. |
| `fleet.conf`        | The pod roster: `id \| gpu \| usd_per_hr \| fb_host \| fb_port \| fb_key \| enabled` (one pod per line). |

## What one `fleet.sh` run does

```
(0) roster       parse fleet.conf -> enabled pods. `--only <id>` or the per-pod
                 `enabled=0` column toggle lanes (this is the bisect switch).
(a) provenance   both repos' git HEAD + sha256 of the working diff (once, shared).
(b) prep [opt]   `--prep` runs loop/bench_loop.sh --gate-only per pod (sync+build+
                 gate). Reuses the loop wholesale; a pod that fails prep is dropped.
(c) resolve      `runpodctl ssh info <id>` per pod -> ip/port/key; falls back to the
                 fb_* columns with a loud warning (may be stale) — never hardcoded.
(d) GATE         unless --skip-gate: 10-transfer PIE gpu-native CUDA prove+verify on EVERY pod
                 concurrently. A pod that fails is dropped and marked gate_failed;
                 its performance is NEVER reported (gate-first, like the loop). The
                 gate proves twice and requires byte-identical proofs; the emitted
                 JSON must also prove the typed CUDA PCS architecture completed.
(e) benchmark    one rotate-mode pipelined stream per pod (--pie a,b,c,d --pipeline D
                 --producers P --pie-mode rotate), launched detached (nohup+setsid)
                 on all pods at once, polled together. STWO_BENCH_TRACE=json,
                 STWO_JIT_LOG=1, RUST_MIN_STACK=4M.
(f) collect      pull each pod's stdout (main record + pipeline record).
(g) aggregate    sum sustained useful MHz, sum $/hr, compute $/MHz-hr, and the
                 pods-needed for the 10 and 20 MHz targets -> fleet_report.json.
(h) table        human summary to stdout.
```

The **axis of concurrency** is the one difference from `loop/bench_loop.sh`: the loop
runs N runs on ONE pod sequentially; the fleet runs ONE run on N pods concurrently. It
deliberately reuses the loop's discipline — runtime pod resolution, git provenance,
gate-first, detached launch, and the exact stall-detection contract (stderr size + GPU
util both frozen for `STALL_SECS` ⇒ evidence captured, process group killed, pod
dropped) — and does **not** modify `bench_loop.sh`.

## Usage

```
./fleet.sh [--conf PATH] [--pies LIST] [--reps N] [--depth D] [--producers P]
           [--only POD_ID] [--prep] [--skip-gate]
```

| Flag           | Meaning                                                                  |
|----------------|--------------------------------------------------------------------------|
| `--conf PATH`  | Pod roster (default `fleet/fleet.conf`).                                  |
| `--pies LIST`  | Comma list of PIE selectors for the rotate stream (`1\|2\|3\|4\|10t`; default `1,2,3,4`). |
| `--reps N`     | Reps (proofs) per pod's stream (default `FLEET_REPS=8`).                  |
| `--depth D`    | Pipeline depth per pod (default `FLEET_DEPTH=3`).                         |
| `--producers P`| Host producer threads per pod (default `FLEET_PRODUCERS=4`).              |
| `--only ID`    | Run exactly one roster pod (bisect a single lane).                       |
| `--prep`       | Run `bench_loop.sh --gate-only` per pod first (sync+build+gate).         |
| `--skip-gate`  | Skip fleet's own gate (bisect escape hatch); report marked **ungated**.  |

### Environment

| Var                                        | Default | Meaning |
|--------------------------------------------|---------|---------|
| `FLEET_CONF` / `FLEET_ONLY`                | (—)     | Roster path / single pod id (overridden by `--conf` / `--only`). |
| `FLEET_REPS` / `FLEET_DEPTH` / `FLEET_PRODUCERS` | `8`/`3`/`4` | Rotate stream knobs. |
| `BENCH_ENV`                                | (empty) | `"K=V K=V ..."` exported into every `gpu_bench` invocation (gate + benchmark) on every pod, and recorded in the report. Debug bisects (`STWO_CUDA_DISABLE_STREAMS=1`, ...). No spaces in values. |
| `GPU_PCS_RUNTIME_MODE`                     | `arena-graph` | Required typed CUDA PCS runtime mode. `detached-eager` remains migration diagnostics only. |
| `POD_BOOTLOADER_JSON`                      | `/workspace/bench_inputs/simple_bootloader_compiled.json` | Stable remote bootloader path, manifest-pinned and SHA-256 preflighted on every participating pod, then exported for every launch. |
| `DRY_RUN=1`                                | `0`     | Echo every ssh instead of executing; fabricate per-pod output so the whole roster→launch→poll→aggregate→report path runs offline. |
| `FAKE_STALL`                               | (unset) | (DRY_RUN only) pod id to simulate as stalled — exercises the stall→drop path. |
| `POLL_INTERVAL` / `STALL_SECS` / `MAX_WAIT`| `15`/`600`/`10800` | Poll cadence / stall window / hard cap (same semantics as the loop). |

## Kill switch / lane toggles (how the integration agent bisects)

The backend is not a fleet toggle: gate and benchmark launches are fixed to
`replacement-v1`. The supported toggles control *which pods participate*, which is the
intended bisect surface:

- **per-pod** — set `enabled=0` on any roster line to drop that lane without touching
  anything else, or `--only <id>` to run exactly one pod.
- **gate** — `--skip-gate` disables the correctness gate (escape hatch); the report is
  then stamped `"gated": "ungated"` so a number produced without the gate can never be
  mistaken for a trusted one.
- **debug env** — `BENCH_ENV=...` (e.g. `STWO_CUDA_DISABLE_STREAMS=1`) is recorded in
  the report and, per the loop's rule, is never comparable with a clean number.

A pod that fails its gate, fails to launch, or stalls is **dropped from the aggregate**
and recorded with its status — never silently averaged into the fleet number.

Every CUDA gate and performance invocation carries `--engine gpu-native`,
`--resident-backend replacement-v1`, and `--require-gpu-native-architecture`. After
pulling stdout, the fleet independently runs
`gpu_benchmarks/validate_architecture_record.py`; a stale binary or partial record is
dropped unless it reports the exact `cuda-typed-pcs-driver-v1` tag, the required runtime
mode, `replacement-v1` as both the requested and executed resident backend, all seven
starts and finishes exactly once, batched tree decommit, and complete telemetry. It also
rejects any AOT miss, runtime load/cache hit, or strict rejection;
AOT loads and AOT cache hits remain explicit counters, including legitimate zeroes when
no generated kernel ran, and the embedded AOT manifest hash must be non-zero.

## Reading `fleet_report.json`

```jsonc
{
  "ts": "...", "stwo_rev": "...", "cairo_rev": "...", "gated": "gated",
  "pies": "1,2,3,4", "reps": 8, "pipeline_depth": 3, "producers": 4, "pie_mode": "rotate",
  "pods": [
    { "id": "...", "gpu": "RTX 4090", "usd_per_hr": 0.44, "status": "ok",
      "useful_mhz": 1.10, "mhz_basis": "sustained_useful_mhz",
      "gpu_resident_backend_requested": "replacement-v1",
      "gpu_resident_backend": "replacement-v1",
      "feed_starved_s": 0.0, "vram_peak_gb": 36.2, "usd_per_mhz_hr": 0.40 }
  ],
  "aggregate": {
    "n_pods_ok": 6, "n_pods_total": 6,
    "aggregate_useful_mhz": 6.60, "total_usd_per_hr": 2.64, "usd_per_mhz_hr": 0.40,
    "mean_pod_useful_mhz": 1.10,
    "meets_target_lo": false, "pods_needed_for_target_lo": 10,
    "meets_target_hi": false, "pods_needed_for_target_hi": 19
  }
}
```

- **per-pod `useful_mhz`** is `sustained_useful_mhz` from the pod's pipeline record
  (the sustained, feed-fed rate — the honest fleet capacity), or the single-prove
  `useful_mhz_median` if a pod somehow produced no pipeline record. Legacy warm-best
  `useful_mhz` is never an aggregate or ranking fallback.
- **`aggregate_useful_mhz`** = Σ per-pod useful MHz over pods with `status:"ok"` only.
- **`total_usd_per_hr`** = Σ `usd_per_hr` over those same ok pods (so `$/MHz-hr` is
  honest — you do not pay for a stalled pod's MHz because it contributed none).
- **`usd_per_mhz_hr`** = `total_usd_per_hr / aggregate_useful_mhz` — the headline cost
  metric.
- **`pods_needed_for_target_*`** = ⌈target / mean_pod_useful_mhz⌉ — how many more of
  this class of pod reach 10 / 20 MHz.

## The fleet math

The only admitted fleet arithmetic is based on workers measured in the same run:
`aggregate_useful_mhz = Σ worker_useful_mhz` and
`usd_per_mhz_hr = Σ worker_usd_per_hr / aggregate_useful_mhz`. The report derives the
worker count for the 10 and 20 MHz targets from that run's measured mean. It must not
substitute the 3.962227 H100 diagnostic for an unmeasured 3090, 4090, 5090, A40, or
48-GiB-class rate. Each hardware class needs its own complete physical-memory admission,
correctness gate, and sustained whole-SN-PIE measurement first.

## Provenance & discipline (inherited from the loop)

- Every `fleet_report.json` records both repos' `git rev` + a `sha256` of the working
  diff (`*` marks a dirty tree), the `bench_env`, and `"gated"` — a number is never
  ambiguous about which source, on which pods, under which env produced it.
- **Only same-pod comparisons are meaningful.** The aggregate is a SUM across
  heterogeneous pods — a fleet-capacity number, not a per-pod delta. To track a code
  change on one pod over time, use `loop/ledger_report.py`; the fleet report is a
  snapshot of capacity, not a regression tracker.
- `n_queries=70` / `pow_bits=26` (96-bit) config throughout — never compare against
  NitrooZK's `n_queries=3` figures (KNOWN_ISSUES.md).
- Raw per-pod stdout/err and any stall evidence land in `fleet/results/`.
- A copied binary never depends on a builder-local Cargo cache path:
  `build_and_push.sh` builds with the stable `POD_BOOTLOADER_JSON`, preflights or seeds
  its pinned file on the builder, copies it with the binary, and verifies it on each
  destination. Set `BOOTLOADER_JSON_SOURCE` when the builder needs seeding.

## Files

| File                | Role                                                            |
|---------------------|-----------------------------------------------------------------|
| `fleet.sh`          | The concurrent orchestrator (steps 0 + a–h above).              |
| `build_and_push.sh` | Build once; distribute the portable binary, pinned bootloader, and JIT cache. |
| `pod_provision.sh`  | `runpodctl` create/list/terminate/ssh-info helpers.             |
| `fleet.conf`        | Pod roster (edit to add/remove/disable pods).                   |
| `fleet_report.json` | Latest aggregate report (overwritten each run).                 |
| `results/`          | Raw per-pod stdout/err + stall evidence + run manifest.         |

## gpufleet — the formalized front door (replaces ad-hoc pod handling)

Stdlib-only Python (`fleet/gpufleet/`, shim `./gpufleet.sh`). Typed RunPod
GraphQL client, budget-guarded provisioning, on-pod deadman (TTL + idle
self-stop via `kill 1` — **no credentials ever placed on a pod**), GPU-idle
alarms during runs, declarative manifests with machine-checked criteria, and a
cost ledger. `pods.conf` is now GENERATED from the live roster (`status`/`up`
refresh it); build_and_push.sh and the loop scripts consume it unchanged.

    STWO_SN_ADAPTED_DIR=/path/to/sealed/adapted_inputs \
      ./gpufleet.sh pregate                    # local no-GPU battery — required before spend
    ./gpufleet.sh offers --gpu 4090 h100       # live $/hr + stock
    ./gpufleet.sh check-manifest manifests/jit_witness_gate.toml
    ./gpufleet.sh run manifests/jit_witness_gate.toml --auto 4090 --push \
        --purpose jit-witness-gate             # provision → push → gates → stop, one command
    ./gpufleet.sh status | ledger              # pods+billing warnings | spend report
    ./gpufleet.sh resume --pod ID              # warm restart (disk kept on stop)

Discipline encoded (not advisory):
  * `pregate` hashes the manifest-pinned SN1-SN4 adapted inputs before work, then
    drift-checks their single 373-kernel production AOT union; fixture-only emission
    cannot classify production composition waves as stale.
  * `run` refuses to provision unless `pregate` passed within 6 h.
  * `up` refuses above `--max-usd-hr` (default $3.00).
  * Every pod self-stops at TTL (default 6 h; `--ttl-hours 24` for planned
    overnight loops) or after 45 min idle (no heartbeat AND no
    gpu_bench/cargo/nvcc process — manual tmux benches are auto-covered).
  * A `gpu_bound = true` manifest step with a ~0% device for >3 min raises an
    idle alarm in the step result (the stopped-H100-shaped failure mode).
  * `run` stops the pod on completion by default (`--keep` bills, says so).
