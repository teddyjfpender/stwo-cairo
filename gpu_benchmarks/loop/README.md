# Stwo GPU proving — fast-feedback benchmark loop

One command turns a code change into trustworthy, fully-provenanced SN-PIE benchmark
numbers on the RunPod pod, appended to a persistent ledger. Minutes, not hours, and
impossible to confuse which code produced which number.

```
cd gpu_benchmarks/loop
SN_PIE_SOURCE_DIR=/path/to/raw-zips \
BOOTLOADER_JSON_SOURCE=/path/to/simple_bootloader_compiled.json \
./qualification_round.sh
                                # local admission, release gate, SN2 headline A/B, all fixed PIEs
./ledger_report.py              # read the ledger back as a table
```

## What one run does

```
(0) resolve pod  the pod address is NEVER hardcoded: `runpodctl ssh info $POD_ID`
                 resolves ip/port/key at runtime (pod id from BENCH_POD_ID or
                 pod.conf); falls back to pod.conf FALLBACK_* with a loud warning
(a) provenance   both repos' git HEAD + sha256 of the working diff  (see below)
(b) sync         rsync stwo + stwo-cairo to the pod (fast delta); Cargo's relative
                 [patch] resolves the sibling /workspace/stwo checkout directly
(c) build        incremental `cargo build ... gpu_bench --features pie-bench`;
                 a compile error aborts LOUDLY with the tail of the build log and
                 writes NO ledger entry
(d) GATE         10-transfer PIE, two gpu-native CUDA proofs + verify + proof-byte
                 equality + typed PCS architecture contract, FIRST — under the SAME
                 BENCH_ENV as the benchmarks. A failure writes `gate_failed`
(e) benchmark    the selected PIE(s), STWO_BENCH_TRACE=json, STWO_JIT_LOG=1
                 (always on — cold JIT-compile visibility catches hangs),
                 RUST_MIN_STACK=4M, --reuse-input, launched detached
                 (nohup+setsid) on the pod and polled with STALL DETECTION
(f) pull         each run's stdout (main record + per-rep phase_totals) back
(g) ledger       one JSON line per run appended to `ledger.jsonl` (incl. bench_env)
(h) summary      useful_mhz_median for fixed statements (sustained_useful_mhz for
                 pipelines) + delta vs the previous SAME-run/host/env entry
```

## Usage

```
./bench_loop.sh [--pie {1|2|3|4|10t}] [--reps N] [--all-pies|--full] [--simd]
                [--skip-sync] [--gate-only]
```

For the ordinary replacement-backend edit loop, use the smaller end-to-end SN2
checkpoint instead of rerunning promotion qualification:

```bash
BENCH_POD_ID=<pod-id> ./quick_sn2.sh
```

It first runs the CPU-only launcher/provenance tests locally, before starting a paid
pod. It accepts dirty worktrees only because both benchmark-relevant source
projections are content-hashed and synced exactly; ignored caches, runtime receipts,
provider ledgers/rosters, bootstrap/run results, and persistent PIE fixtures are not
source. It reuses the persistent Cargo/CUDA
caches, rebuilds the changed release binary, checks the pinned input and AOT
identities, then runs six verifier-backed SN2 proofs with fresh SIMD byte equality
and deterministic mutation rejection. The result is explicitly `iteration_only`
and never formal-promotion eligible. Carry-oracle, Stage-4 native, NCU, and Nsight
gates remain mandatory in the sealed qualification recipes, but no longer consume
every edit cycle.

The 2026-07-15 H100 receipt bounds this lane at roughly **7–10 minutes** for a
changed full-SN2 candidate on a prepared pod: 364 s release build, 10 s input
adaptation, about 2 s AOT identity, 56 s proof checkpoint, plus sync/bootstrap.
That is about 2.5x shorter than the 18.8-minute qualification core. An unchanged
sealed binary can rerun the timing checkpoint in about one minute. This is the
full-proof integration lane, not the seconds-scale kernel lab. Current SN2 peaks
at 50.018 decimal GB VRAM, so use an 80 GB GPU here; use cheaper GPUs for the
kernel/transcript-segment replay lane, not this full proof.

For the direct Blake-G CUDA admission gate, use the much smaller A40 lane instead
of building or proving an SN PIE:

```bash
DRY_RUN=1 BENCH_POD_ID=dry-run-placeholder \
  ./pod_run.sh recipes/direct_blake_g_native.phases direct_blake_g_sm86_dry_run

BENCH_POD_ID=<secure-a40-pod-id> POD_RUN_POLL_INTERVAL=2 MAX_WAIT=3600 \
  POD_RUN_FINAL_ACTION=terminate \
  ./pod_run.sh recipes/direct_blake_g_native.phases direct_blake_g_sm86
```

The recipe pins `sm_86`, CUDA 11.8, the Stwo head, test source, lockfile and
toolchain; it requires one real eager-plus-graph-replay native test with zero
ignored tests, hashes the resulting executable and evidence, and confirms provider
termination on every exit. The A40 lane compiles the real ordinary CUDA archive
containing the Blake-G witness/relation kernels, but its explicit test-only feature
skips all unrelated generated AOT cubins and requires the log to attest zero AOT
entries. A pass promotes that exact source projection to the later target-sm90 gate;
it is not valid for resident proving, an SN-PIE benchmark, an H100 result, or a
proving-MHz claim. Other `pod_run.sh` recipes default to confirmed stop and retain
their warm attached disk; set `POD_RUN_FINAL_ACTION=terminate` only for one-shot
leases whose disk must be released.

Every non-dry `pod_run.sh` invocation rechecks the fresh, source-bound local
pregate immediately before starting or resuming compute. Keep
`STWO_SN_ADAPTED_DIR` pointed at the sealed SN1-SN4 adapted-input directory; a
source/input change or missing receipt fails before billing can start, and the
exit trap still confirms the configured final pod state.

| Flag          | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `--pie SEL`   | Which PIE to benchmark: `1..4` = `SN_PIE_<n>.zip`, `10t` = 10-transfer. Default `2`. |
| `--reps N`    | Fixed-statement proofs (minimum 2). Published default `6`: one cold + five warm; the warm median is reported. |
| `--all-pies`  | Benchmark `SN_PIE_1/2/3/4` without the rotate-mode fleet.          |
| `--full`      | Also benchmark `SN_PIE_1/3/4` (CUDA) **and** run the rotate-mode fleet (pipelined stream over all four PIEs — the production one-pod-proving-a-block-stream shape). |
| `--simd`      | Add a same-host SIMD run of the selected PIE (CPU baseline).            |
| `--skip-sync` | Internal qualification continuation only; requires its source-bound soundness artifact. |
| `--gate-only` | Run only the correctness gate, then exit; still requires local preflight admission. |

### Environment

| Var             | Default    | Meaning                                                     |
|-----------------|------------|-------------------------------------------------------------|
| `BENCH_POD_ID`  | (pod.conf) | Pod id, overrides `POD_ID` in `pod.conf`.                   |
| `BENCH_ENV`     | (empty)    | Shell-safe `"K=V K=V ..."` tokens exported into **every** gpu_bench invocation (gate included) and **recorded in every ledger entry**. `STWO_BOOTLOADER_JSON` is reserved; use `POD_BOOTLOADER_JSON`. |
| `QUALIFICATION_ARTIFACT` | (unset) | Required for ordinary publishable performance runs; must be a passed `qualification_round.sh` artifact matching the exact source, environment, and runtime. |
| `LOCAL_PREFLIGHT_ADMISSION` | (unset) | Required before any pod work. Generated by `qualification_round.sh`; binds source hashes, flags-off plus both exact profiles, current raw-to-adapted byte identity, six preflight artifacts, and the 76 GiB ceiling. |
| `GPU_PCS_RUNTIME_MODE` | `arena-graph` | Required typed CUDA PCS runtime mode for every CUDA gate and performance run. `detached-eager` remains migration diagnostics only. |
| `DRY_RUN=1`     | `0`        | Echo every ssh/rsync instead of executing, and fabricate run output so the provenance → ledger → summary path still runs for real. Use to trace logic offline. |
| `FAKE_STALL`    | (unset)    | (DRY_RUN only) name of a run to simulate as stalled — exercises the stall → evidence → ledger → abort path. |
| `POLL_INTERVAL` | `15`       | Seconds between pod poll checks.                            |
| `STALL_SECS`    | `600`      | Stall detector window (see below).                          |
| `MAX_WAIT`      | `10800`    | Hard cap on waiting for one run (3h backstop).              |
| `FLEET_REPS` / `FLEET_DEPTH` / `FLEET_PRODUCERS` | `8` / `3` / `4` | Fleet (rotate) pipeline knobs. |
| `GATE_PIE` | 10-transfer PIE on pod | Explicit remote gate fixture. Use `/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_2.zip` when the smaller fixture is unavailable; verification still runs. |
| `POD_BOOTLOADER_JSON` | `/workspace/bench_inputs/simple_bootloader_compiled.json` | Stable remote path. Its manifest-pinned file is required and SHA-256 preflighted, exported during build, and exported for every launch. |
| `SN_PIE_SOURCE_DIR` / `GATE_PIE_SOURCE` / `BOOTLOADER_JSON_SOURCE` | (unset) | Local sources to checksum and print explicit seed commands for. Raw SN PIEs and the bootloader are required by `qualification_round.sh` so the current adapter can reproduce every preflight input byte; they are never uploaded automatically. |

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
compare it with clean numbers), `claim_mhz`, its explicit `mhz_basis`
(`useful_mhz_median` for fixed statements or `sustained_useful_mhz` for pipelines),
`vram_gb`, and `delta` vs the previous **same-run, same-pod, same-bench_env** entry.
Legacy warm-best `useful_mhz` remains in raw output for compatibility and is never
used for a claim, ranking, or delta.

## Provenance model — why a number is never ambiguous

Every ledger entry records, at the moment of the run:

- `stwo_rev`, `cairo_rev` — `git rev-parse HEAD` of each repo.
- `stwo_dirty`, `cairo_dirty` — `"clean"` if the working tree matches HEAD, otherwise
  the full SHA-256 over tracked changes and untracked path/content. This makes a run reproducible **even with
  uncommitted work**: the same rev + same dirty hash == the same source. A changed
  number with an unchanged (rev, dirty) pair on the same pod is a real signal; a changed
  dirty hash tells you the source moved.
- `pod_gpu` — the GPU reported by the pod (currently `NVIDIA GeForce RTX 3090`).
- `bench_env` — the exact `BENCH_ENV` the run executed under (empty for clean runs).
  A number produced with debug sync or disabled streams is permanently marked as such.
- `record` — the harness's own self-describing JSON (security config, host fingerprint,
  timings, VRAM, `useful_mhz_median`, proof-byte equality, typed CUDA PCS architecture,
  runtime mode, and exact seven-stage start/finish counts). `phase_totals` — the per-rep `STWO_BENCH_TRACE=json`
  span breakdown. Fleet runs additionally carry a `pipeline` object (sustained numbers).
- `status` — `ok`, `gate_failed`, `run_failed`, or `stalled` (with `stall_evidence`).

The dirty hash includes tracked changes plus untracked paths and file contents, so
new generated CUDA sources are bound without modifying the local Git index.

## The gate-first rule

Before the PIE correctness gate, a counted native CUDA suite must pass for the
requested runtime. `detached-eager` requires every live and prepared-operation
conformance target; `arena-graph` additionally requires strict resident
whole-proof byte identity. A detached benchmark therefore cannot hide a live-path
CUDA failure, while an unfinished resident graph cannot masquerade as admitted.

The correctness gate (configured PIE, two gpu-native CUDA proofs, verify, required
proof-byte equality, and typed PCS architecture) always runs before any
performance benchmark, **under the same `BENCH_ENV` as the benchmarks** — a kill switch
that changes prover behavior must be correctness-gated too. The harness validates the
output contract (`verified_reps=2`, equality applicable/required/true) and independently
runs `validate_architecture_record.py`. That validator requires `backend=cuda`,
`engine=gpu-native`, architecture `cuda-typed-pcs-driver-v1`, the selected runtime mode,
all seven stage starts and finishes exactly once, batched tree decommit, and complete
telemetry. Strict AOT provenance additionally requires `gpu_aot_misses`, runtime loads,
runtime cache hits, and strict rejections all to be zero; AOT loads and cache hits are
always reported (and may both be zero when no generated kernel was invoked), and the
embedded AOT manifest hash must be non-zero. The validator is also applied to every
CUDA performance record, closing the stale-binary
case where an unknown CLI flag is silently ignored. A failed contract, verify, or crash
aborts the run. **Performance is never reported from a build that failed the gate.**

`qualification_round.sh` is the only release admission flow. Its internal A/B and
fixed-PIE records are marked provisional until the final manifest binds their hashes.
The passed manifest is the promoted performance artifact: it embeds the validated
SN1–SN4 records and the six-repetition flags-off/headline SN2 comparison.
Every remote launcher first removes ambient `STWO_*` and blocking CUDA overrides,
then exports only its recorded policy state and fixed harness variables.
`bench_loop.sh` and `perf_gates.sh` refuse pod work without the source-, profile-,
runtime-, input-, and capacity-bound local admission artifact. `bench_loop.sh` also
refuses an ordinary publishable run without the final passed qualification manifest.

## ⚠️ Only same-pod comparisons are meaningful

`useful_mhz_median` depends heavily on the host (GPU model, CPU, memory bandwidth, and shared
tenancy on a community pod). The delta computed by both `bench_loop.sh` and
`ledger_report.py` is deliberately scoped to the **same `run_name` AND the same
`pod_gpu` AND the same `bench_env`**. Do **not** read a delta across different GPUs — it
is host variance, not a code effect. When the pod changes, start a fresh comparison
baseline.

## Pod / environment specifics

- Pod identity: `pod.conf` (`POD_ID` + fallback endpoint); resolution via `runpodctl`.
- Repos on the pod: `/workspace/stwo`, `/workspace/stwo-cairo`.
- `pod_run.sh` keeps Rustup and Cargo state on the persistent volume at
  `/workspace/.rustup-persist` and `/workspace/.cargo-persist`; override these remote
  paths with `POD_RUSTUP_HOME` and `POD_CARGO_HOME` when needed.
- `stwo_cairo_prover/Cargo.toml` uses portable relative `[patch]` paths to the sibling
  `stwo` checkout; the same manifest resolves locally and under `/workspace` on a pod.
- PIE inputs (`SN_PIE_*.zip`, the 10-transfer zip) already live on the pod and are
  **never** synced (rsync excludes them).
- The bootloader lives at stable `POD_BOOTLOADER_JSON`; its pinned checksum must match
  `gpu_benchmarks/pie/SHA256SUMS` before sync/build/run. If absent, provide
  `BOOTLOADER_JSON_SOURCE` to print the explicit seed command.
- `target/` and `.git/` are excluded from rsync; the build is incremental (~1-2 min).
- Run scratch (launcher scripts, stdout/err, pid/pgid/rc sentinels, build log) lives in
  `/workspace/bench_loop_runs` on the pod, outside the repo tree.

Do not run `bench_loop.sh` while another benchmark round is active on the pod — the
sync + build would disturb the in-flight round's source tree.

## Files

| File               | Role                                                            |
|--------------------|-----------------------------------------------------------------|
| `bench_loop.sh`    | The one command (steps 0 + a–h above).                          |
| `qualification_round.sh` | One-build post-admission qualification, SN2 A/B, and all-four fixed round. |
| `../validate_architecture_record.py` | Fail-closed validator for pulled gpu_bench architecture evidence. |
| `ledger_report.py` | Read `ledger.jsonl` as a table / phase trend (stdlib only).     |
| `pod.conf`         | Pod id + fallback endpoint (edit when the pod moves).           |
| `ledger.jsonl`     | The persistent ledger (created on first run; one JSON per line).|
| `results/`         | Raw per-run stdout/err/rc + stall evidence pulled from the pod. |
