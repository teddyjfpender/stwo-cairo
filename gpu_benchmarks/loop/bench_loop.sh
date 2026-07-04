#!/usr/bin/env bash
#
# bench_loop.sh — the one command for the Stwo GPU proving optimization loop.
#
# After any code change to the stwo (CUDA backend) or stwo-cairo (prover + gpu_bench)
# repos, this produces trustworthy SN-PIE benchmark numbers, appended to a persistent,
# fully-provenanced ledger — in minutes, and impossible to confuse which code produced
# which number.
#
# Pipeline:
#   (0) resolve pod        : `runpodctl ssh info $POD_ID` (pod id from BENCH_POD_ID or
#                            loop/pod.conf); fall back to pod.conf values with a loud
#                            warning if runpodctl fails. Never a hardcoded address.
#   (a) capture provenance : both repos' git rev + sha256 of the working diff
#   (b) sync to pod        : fast rsync delta, then re-apply the pod's Cargo.toml [patch]
#   (c) incremental build  : cargo build gpu_bench; abort loudly with the log tail
#   (d) CORRECTNESS GATE   : 10-transfer PIE CUDA prove+verify FIRST, under the SAME
#                            BENCH_ENV as the benchmarks. A failed verify or crash
#                            aborts and writes a "gate_failed" ledger entry.
#                            Performance is NEVER reported from a build that failed here.
#   (e) benchmark          : selected PIE(s) — nohup on pod + poll with STALL DETECTION
#                            (stderr size + GPU util both frozen for STALL_SECS =>
#                            capture /proc task states + stderr tail, kill the process,
#                            record status:"stalled", abort). STWO_BENCH_TRACE=json,
#                            STWO_JIT_LOG=1 always on.
#   (f) pull results       : fetch each run's stdout (main record + phase_totals)
#   (g) append to ledger   : one JSON line per run to loop/ledger.jsonl, including
#                            bench_env (a debug-env number is never confused with a
#                            clean one)
#   (h) human summary      : useful_mhz per run + delta vs the previous ledger entry
#                            for the SAME run_name, SAME pod_gpu, SAME bench_env.
#
# Usage:
#   ./bench_loop.sh [--pie {1|2|3|4|10t}] [--reps N] [--full] [--simd]
#                   [--skip-sync] [--gate-only] [--help]
#
# Flags:
#   --pie SEL     Which SN PIE to benchmark: 1|2|3|4 (SN_PIE_<n>.zip) or 10t
#                 (the 10-transfer PIE). Default: 2.
#   --reps N      Repetitions per run (warm-best is reported). Default: 2.
#   --full        Also benchmark SN_PIE_1/3/4 (CUDA) and run the rotate-mode fleet
#                 (pipelined stream over all four PIEs).
#   --simd        Add a same-host SIMD run of the selected PIE (CPU baseline).
#   --skip-sync   Skip rsync AND build; benchmark the binary already on the pod.
#   --gate-only   Run only the correctness gate (sync+build still happen unless
#                 --skip-sync) and exit.
#
# Environment:
#   BENCH_POD_ID  Pod id (overrides POD_ID in loop/pod.conf).
#   BENCH_ENV     "K=V K=V ..." exported verbatim into EVERY gpu_bench invocation
#                 (gate included) and recorded in each ledger entry as "bench_env".
#                 For debug bisects: STWO_CUDA_DISABLE_STREAMS, STWO_CUDA_MEMORY_WITNESS,
#                 STWO_CUDA_DEBUG_SYNC, CUDA_LAUNCH_BLOCKING, ... (no spaces in values).
#   DRY_RUN=1     Echo every ssh/rsync instead of executing; fabricate run output so
#                 the provenance -> ledger -> summary path runs for real offline.
#   FAKE_STALL    (DRY_RUN only) name of a run to simulate as stalled, to exercise
#                 the stall -> evidence -> ledger -> abort path.
#   POLL_INTERVAL Seconds between pod poll checks (default 15).
#   STALL_SECS    Declare a run stalled when its stderr size AND the GPU utilization
#                 are both unchanged for this long (default 600).
#   MAX_WAIT      Hard cap on waiting for one run (default 10800 = 3h).
#
# NOTE: only same-pod comparisons are meaningful (community-host variance). See README.md.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all paths as variables, up top)
# ---------------------------------------------------------------------------
# Local repos + outputs.
STWO_LOCAL="/Users/theodorepender/code/personal/stwo"
CAIRO_LOCAL="/Users/theodorepender/code/personal/stwo-cairo"
LOOP_DIR="${CAIRO_LOCAL}/gpu_benchmarks/loop"
RESULTS_DIR="${LOOP_DIR}/results"
LEDGER="${LOOP_DIR}/ledger.jsonl"
POD_CONF="${LOOP_DIR}/pod.conf"

# Pod repos.
STWO_POD="/workspace/stwo"
CAIRO_POD="/workspace/stwo-cairo"
POD_PROVER_DIR="${CAIRO_POD}/stwo_cairo_prover"
POD_CARGO_TOML="${POD_PROVER_DIR}/Cargo.toml"
BIN="target/release/gpu_bench"                      # relative to POD_PROVER_DIR
POD_USER="root"

# PIE inputs on the pod (already present, hash-verified — never synced).
POD_SN_DIR="${CAIRO_POD}/gpu_benchmarks/pie/sn"
POD_GATE_PIE="${CAIRO_POD}/gpu_benchmarks/pie/cairo_pie_10_transfers_with_6_ecop.zip"

# Pod scratch (outside the repo tree so rsync never touches it).
POD_RUN_DIR="/workspace/bench_loop_runs"
POD_BUILD_LOG="${POD_RUN_DIR}/build.log"

# The [patch] path rewrite the pod copy needs (local Cargo.toml carries /Users/... paths).
SED_PATCH='s|/Users/theodorepender/code/personal/stwo|/workspace/stwo|g'

# Prover knobs.
RUST_MIN_STACK_VAL=4194304
BUILD_RUSTFLAGS="-C target-cpu=native"
BENCH_ENV="${BENCH_ENV:-}"          # debug/bisect env, recorded in every ledger entry

# Fleet (rotate) run parameters.
FLEET_REPS="${FLEET_REPS:-8}"
FLEET_DEPTH="${FLEET_DEPTH:-3}"
FLEET_PRODUCERS="${FLEET_PRODUCERS:-4}"

# Poll behavior.
DRY_RUN="${DRY_RUN:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
STALL_SECS="${STALL_SECS:-600}"
MAX_WAIT="${MAX_WAIT:-10800}"

# Flag defaults.
PIE_SEL="2"
REPS="2"
FULL=0
SIMD=0
SKIP_SYNC=0
GATE_ONLY=0

# Set by resolve_pod().
POD_HOST=""
POD_PORT=""
POD_KEY=""
SSH_OPTS=()
SSH_E=""

# Set by run_bench().
LAST_OUT=""
LAST_RC=""
LAST_STALL_FILE=""

# ---------------------------------------------------------------------------
# Logging helpers ( logs -> stderr, human summary -> stdout )
# ---------------------------------------------------------------------------
log()  { echo "[bench_loop] $*" >&2; }
dry()  { echo "[DRY_RUN] $*" >&2; }
warn() { echo "[bench_loop][WARN] $*" >&2; }
die()  { echo "[bench_loop][FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,66p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------------------------------------------------------
# (0) Pod resolution — never trust a hardcoded address; community pods churn.
# ---------------------------------------------------------------------------
resolve_pod() {
  local pod_id="" fb_host="" fb_port="" fb_key=""
  if [[ -f "$POD_CONF" ]]; then
    # pod.conf is shell-sourceable: POD_ID / FALLBACK_HOST / FALLBACK_PORT / FALLBACK_KEY
    local POD_ID="" FALLBACK_HOST="" FALLBACK_PORT="" FALLBACK_KEY=""
    # shellcheck source=/dev/null
    . "$POD_CONF"
    pod_id="$POD_ID"; fb_host="$FALLBACK_HOST"; fb_port="$FALLBACK_PORT"; fb_key="$FALLBACK_KEY"
  fi
  pod_id="${BENCH_POD_ID:-$pod_id}"   # env wins over pod.conf
  [[ -n "$pod_id" ]] || die "no pod id — set BENCH_POD_ID or POD_ID in ${POD_CONF}"

  if command -v runpodctl >/dev/null 2>&1; then
    local info parsed
    if info="$(runpodctl ssh info "$pod_id" 2>/dev/null)" &&
       parsed="$(RP_INFO="$info" python3 -c '
import json, os, sys
d = json.loads(os.environ["RP_INFO"])
ip = d.get("ip") or d.get("host")
port = d.get("port") or d.get("sshPort")
k = d.get("ssh_key")
key = k.get("path") if isinstance(k, dict) else k
if not (ip and port and key):
    sys.exit(1)
print(ip, port, key)
' 2>/dev/null)"; then
      read -r POD_HOST POD_PORT POD_KEY <<<"$parsed"
      log "pod ${pod_id} resolved via runpodctl: ${POD_HOST}:${POD_PORT}"
    fi
  fi

  if [[ -z "$POD_HOST" || -z "$POD_PORT" || -z "$POD_KEY" ]]; then
    if [[ -n "$fb_host" && -n "$fb_port" && -n "$fb_key" ]]; then
      warn "runpodctl resolution FAILED for pod '${pod_id}'."
      warn "Falling back to ${POD_CONF} values: ${fb_host}:${fb_port} — these may be"
      warn "STALE if the pod restarted. Refresh pod.conf if the connection fails."
      POD_HOST="$fb_host"; POD_PORT="$fb_port"; POD_KEY="$fb_key"
    else
      die "cannot resolve pod '${pod_id}': runpodctl failed and no FALLBACK_* in ${POD_CONF}"
    fi
  fi

  SSH_OPTS=(
    -p "$POD_PORT"
    -i "$POD_KEY"
    -o ConnectTimeout=20
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=4
    -o StrictHostKeyChecking=accept-new
    -o BatchMode=yes
  )
  SSH_E="ssh -p ${POD_PORT} -i ${POD_KEY} -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
}

# ---------------------------------------------------------------------------
# Remote transport wrappers (DRY_RUN-aware)
# ---------------------------------------------------------------------------
run_ssh() {
  if [[ "$DRY_RUN" == "1" ]]; then dry "ssh: $*"; return 0; fi
  ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "$@"
}

run_rsync() {
  if [[ "$DRY_RUN" == "1" ]]; then dry "rsync $*"; return 0; fi
  rsync "$@"
}

# ---------------------------------------------------------------------------
# PIE selector -> pod path / run name
# ---------------------------------------------------------------------------
pie_path() {
  case "$1" in
    1|2|3|4) echo "${POD_SN_DIR}/SN_PIE_$1.zip" ;;
    10t)     echo "${POD_GATE_PIE}" ;;
    *)       die "invalid --pie '$1' (want 1|2|3|4|10t)" ;;
  esac
}
pie_name() {
  case "$1" in
    10t) echo "PIE_10t" ;;
    *)   echo "SN_PIE_$1" ;;
  esac
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pie)       PIE_SEL="${2:?--pie needs a value}"; shift 2 ;;
    --reps)      REPS="${2:?--reps needs a value}"; shift 2 ;;
    --full)      FULL=1; shift ;;
    --simd)      SIMD=1; shift ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --gate-only) GATE_ONLY=1; shift ;;
    -h|--help)   usage 0 ;;
    *)           die "unknown flag: $1 (try --help)" ;;
  esac
done

[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -ge 1 ]] || die "--reps must be a positive integer"
pie_path "$PIE_SEL" >/dev/null   # validates selector early

# Validate BENCH_ENV shape early: every token must be K=V (no spaces in values).
if [[ -n "$BENCH_ENV" ]]; then
  for kv in $BENCH_ENV; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] \
      || die "BENCH_ENV token '$kv' is not K=V (values must not contain spaces)"
  done
fi

resolve_pod

# ---------------------------------------------------------------------------
# (a) Provenance capture
# ---------------------------------------------------------------------------
git_rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }
# sha256 of the working-tree diff vs HEAD (staged + unstaged, tracked files). "clean"
# when there is no diff. Makes every ledger entry traceable even with uncommitted work.
git_dirty() {
  local repo="$1"
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then echo "NOGIT"; return; fi
  if git -C "$repo" diff --quiet HEAD 2>/dev/null; then echo "clean"; return; fi
  git -C "$repo" diff HEAD 2>/dev/null | shasum -a 256 | cut -c1-16
}

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STWO_REV="$(git_rev "$STWO_LOCAL")"
STWO_DIRTY="$(git_dirty "$STWO_LOCAL")"
CAIRO_REV="$(git_rev "$CAIRO_LOCAL")"
CAIRO_DIRTY="$(git_dirty "$CAIRO_LOCAL")"

mkdir -p "$RESULTS_DIR"

log "provenance: stwo=${STWO_REV:0:12} dirty=${STWO_DIRTY} | cairo=${CAIRO_REV:0:12} dirty=${CAIRO_DIRTY}"
[[ -n "$BENCH_ENV" ]] && log "bench_env: ${BENCH_ENV} (recorded in every ledger entry)"

# ---------------------------------------------------------------------------
# Discover pod GPU (read-only; drives the same-host comparison guard in the ledger)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  POD_GPU="DRY-RUN-GPU"
else
  POD_GPU="$(run_ssh "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" || true)"
  [[ -n "$POD_GPU" ]] || POD_GPU="unknown"
fi
log "pod GPU: ${POD_GPU}"

# ---------------------------------------------------------------------------
# (b) Sync + re-apply Cargo.toml patch
# ---------------------------------------------------------------------------
sync_repos() {
  log "rsync stwo -> pod (excludes target/.git)"
  run_rsync -az --partial \
    --exclude=target --exclude=.git \
    -e "$SSH_E" \
    "${STWO_LOCAL}/" "${POD_USER}@${POD_HOST}:${STWO_POD}/"

  log "rsync stwo-cairo -> pod (excludes target/.git/PIE zips/ledger)"
  run_rsync -az --partial \
    --exclude=target --exclude=.git \
    --exclude='gpu_benchmarks/pie/sn/*.zip' \
    --exclude='gpu_benchmarks/pie/*.zip' \
    --exclude='gpu_benchmarks/loop/results' \
    --exclude='gpu_benchmarks/loop/ledger.jsonl' \
    -e "$SSH_E" \
    "${CAIRO_LOCAL}/" "${POD_USER}@${POD_HOST}:${CAIRO_POD}/"

  # The rsync just overwrote the pod's patched Cargo.toml with the /Users/... local copy.
  # Re-apply the pod path rewrite so the [patch] resolves to /workspace/stwo.
  log "re-applying Cargo.toml [patch] rewrite on pod"
  run_ssh "sed -i '${SED_PATCH}' '${POD_CARGO_TOML}'"
}

# ---------------------------------------------------------------------------
# (c) Incremental build (abort loudly, with the log tail)
# ---------------------------------------------------------------------------
build_pod() {
  log "incremental build on pod (gpu_bench, --features pie-bench)"
  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "build: cargo build --release -p stwo-cairo-prover --bin gpu_bench --features pie-bench"
    return 0
  fi
  local out
  out="$(run_ssh "cd '${POD_PROVER_DIR}' && . \$HOME/.cargo/env 2>/dev/null; \
      PATH=/usr/local/cuda/bin:\$PATH RUSTFLAGS='${BUILD_RUSTFLAGS}' \
      cargo build --release -p stwo-cairo-prover --bin gpu_bench --features pie-bench \
      > '${POD_BUILD_LOG}' 2>&1; echo BUILD_EXIT=\$?")"
  local code
  code="$(printf '%s\n' "$out" | sed -n 's/.*BUILD_EXIT=\([0-9][0-9]*\).*/\1/p' | tail -1)"
  if [[ "$code" != "0" ]]; then
    warn "BUILD FAILED (exit=${code:-?}). Tail of ${POD_BUILD_LOG}:"
    run_ssh "tail -n 40 '${POD_BUILD_LOG}'" >&2 || true
    die "compile error — aborting before any benchmark. No ledger entry written."
  fi
  log "build OK"
}

# ---------------------------------------------------------------------------
# Synthetic run output for DRY_RUN (exercises the ledger/summary path offline)
# ---------------------------------------------------------------------------
synth_out() {
  # $1 name  $2 args  $3 destfile — deterministic-ish mock numbers keyed off the name.
  local name="$1" args="$2" dest="$3"
  local backend="cuda"; [[ "$args" == *"--backend simd"* ]] && backend="simd"
  local seed=$(( $(printf '%s' "$name" | cksum | cut -d' ' -f1) % 40 ))
  local um; um="$(awk -v s="$seed" 'BEGIN{printf "%.3f", 1.4 + s/100.0}')"
  {
    echo "{\"rep\":0,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":1234.5},\"fri\":{\"count\":1,\"total_ms\":567.8}}}"
    echo "{\"rep\":1,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":1201.2},\"fri\":{\"count\":1,\"total_ms\":560.1}}}"
    echo "{\"program\":\"${name}.zip\",\"backend\":\"${backend}\",\"n\":1,\"cycle_count\":14600000,\"pie_n_steps\":12000000,\"bootloader_overhead_pct\":21.6,\"prove_s_cold\":9.9,\"prove_s_warm\":8.1,\"verify_ms\":42.0,\"proof_kb\":210.5,\"peak_rss_gb\":18.2,\"vram_end_gb\":6.1,\"vram_peak_gb\":11.3,\"steps_per_s\":1802469.0,\"mhz\":1.802,\"useful_mhz\":${um},\"vm_s\":30.2,\"adapt_s\":5.1,\"security_bits\":96,\"n_queries\":70,\"pow_bits\":26,\"fold_step\":3,\"gpu\":\"${POD_GPU}\",\"nproc\":32,\"host_mem_gb\":125.6}"
    if [[ "$args" == *"--pipeline"* ]]; then
      echo "{\"pipeline\":${FLEET_DEPTH},\"producers\":${FLEET_PRODUCERS},\"pie_mode\":\"rotate\",\"reps\":${FLEET_REPS},\"total_s\":80.5,\"feed_starved_s\":2.1,\"sustained_steps_per_s\":1450000.0,\"sustained_mhz\":1.45,\"sustained_useful_mhz\":$(awk -v s="$seed" 'BEGIN{printf "%.3f",1.1+s/100.0}')}"
    fi
  } > "$dest"
}

# ---------------------------------------------------------------------------
# Launch one gpu_bench run detached on the pod, poll to completion (with stall
# detection), fetch stdout.
# Sets globals: LAST_OUT (local stdout path), LAST_RC ("0", exit code, TIMEOUT,
# or STALLED), LAST_STALL_FILE (evidence file, when stalled).
# ---------------------------------------------------------------------------
run_bench() {
  local name="$1" args="$2"
  local base="${RESULTS_DIR}/${STAMP}.${name}"
  local pod_out="${POD_RUN_DIR}/${STAMP}.${name}.out"
  local pod_err="${POD_RUN_DIR}/${STAMP}.${name}.err"
  local pod_rc="${POD_RUN_DIR}/${STAMP}.${name}.rc"
  local pod_sh="${POD_RUN_DIR}/${STAMP}.${name}.sh"
  local pod_pid="${POD_RUN_DIR}/${STAMP}.${name}.pid"
  local pod_pgid="${POD_RUN_DIR}/${STAMP}.${name}.pgid"
  LAST_STALL_FILE=""

  log "run '${name}': gpu_bench ${args}${BENCH_ENV:+  [env: ${BENCH_ENV}]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    dry "launch on pod: ${pod_sh} -> ${pod_out} (rc ${pod_rc})"
    if [[ "${FAKE_STALL:-}" == "$name" ]]; then
      dry "simulating STALL for run '${name}'"
      {
        echo "gpu_bench pid: 4242"
        echo "--- /proc/4242/task/4242/status"
        echo "Name: gpu_bench"
        echo "State: S (sleeping)"
        echo "--- last 30 stderr lines ---"
        echo "rep=0 (no further output for ${STALL_SECS}s)"
      } > "${base}.stall.txt"
      : > "${base}.out"; : > "${base}.err"
      LAST_OUT="${base}.out"; LAST_RC="STALLED"; LAST_STALL_FILE="${base}.stall.txt"
      return 0
    fi
    synth_out "$name" "$args" "${base}.out"
    : > "${base}.err"
    echo 0 > "${base}.rc"
    LAST_OUT="${base}.out"; LAST_RC=0
    return 0
  fi

  # Write a detached launcher to the pod scratch dir (avoids nested-quote hell), then
  # start it with setsid+nohup so it survives ssh channel close. The launcher records
  # its own pid (== process-group id under setsid, used to kill a stalled run) and the
  # gpu_bench pid (used to read /proc task states). STWO_JIT_LOG is ALWAYS on — cold
  # JIT-compile visibility is how hangs get caught. BENCH_ENV (if any) is exported
  # verbatim for every run, gate included.
  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  run_ssh "cat > '${pod_sh}'" <<EOF
#!/usr/bin/env bash
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
echo \$\$ > '${pod_pgid}'
./${BIN} ${args} > '${pod_out}' 2> '${pod_err}' &
GB_PID=\$!
echo \$GB_PID > '${pod_pid}'
wait \$GB_PID
echo \$? > '${pod_rc}'
EOF
  run_ssh "rm -f '${pod_rc}' '${pod_pid}' '${pod_pgid}'; nohup setsid bash '${pod_sh}' >/dev/null 2>&1 & echo LAUNCHED"

  # Poll until the rc sentinel appears. Each poll is ONE ssh round-trip that reports
  # completion, the run's stderr size, and the GPU utilization. Today's failure mode
  # was "process alive forever, zero output": if (stderr size, gpu util) is frozen for
  # STALL_SECS, the run is declared stalled — evidence captured, process group killed,
  # LAST_RC=STALLED. MAX_WAIT stays as the hard backstop.
  local waited=0 last_sig="__init__" stall_at now st err_size gpu_util sig
  stall_at="$(date +%s)"
  while true; do
    st="$(run_ssh "if [ -f '${pod_rc}' ]; then echo DONE; fi; \
                   echo SIZE=\$(stat -c %s '${pod_err}' 2>/dev/null || echo 0); \
                   echo GPU=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); true" || true)"
    if printf '%s\n' "$st" | grep -q '^DONE$'; then break; fi

    err_size="$(printf '%s\n' "$st" | sed -n 's/^SIZE=//p' | head -1)"
    gpu_util="$(printf '%s\n' "$st" | sed -n 's/^GPU=//p' | head -1)"
    now="$(date +%s)"
    sig="${err_size}:${gpu_util}"
    if [[ "$sig" != "$last_sig" ]]; then last_sig="$sig"; stall_at="$now"; fi

    if (( now - stall_at >= STALL_SECS )); then
      warn "run '${name}' STALLED: stderr size and GPU util unchanged for ${STALL_SECS}s."
      local stall_file="${base}.stall.txt"
      # Evidence: thread states of the hung process + the last stderr lines.
      run_ssh "pid=\$(cat '${pod_pid}' 2>/dev/null); echo \"gpu_bench pid: \${pid:-unknown}\"; \
               if [ -n \"\$pid\" ] && [ -d \"/proc/\$pid\" ]; then \
                 for t in /proc/\$pid/task/*/status; do \
                   echo \"--- \$t\"; grep -E '^(Name|State)' \"\$t\" 2>/dev/null; done; \
               else echo '(process directory already gone)'; fi; \
               echo '--- last 30 stderr lines ---'; tail -n 30 '${pod_err}' 2>/dev/null; true" \
        > "$stall_file" || true
      # Kill the whole process group (launcher is the setsid session leader).
      run_ssh "pgid=\$(cat '${pod_pgid}' 2>/dev/null); \
               if [ -n \"\$pgid\" ]; then kill -TERM -\"\$pgid\" 2>/dev/null; sleep 3; \
               kill -KILL -\"\$pgid\" 2>/dev/null; fi; true" || true
      run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
      run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
      LAST_OUT="${base}.out"; LAST_RC="STALLED"; LAST_STALL_FILE="$stall_file"
      log "stall evidence written to ${stall_file}"
      return 0
    fi

    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "run '${name}' exceeded MAX_WAIT (${MAX_WAIT}s). Fetching partial output."
      break
    fi
    sleep "$POLL_INTERVAL"
  done

  # (f) pull results.
  run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
  run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
  run_ssh "cat '${pod_rc}'  2>/dev/null" > "${base}.rc"  || true
  LAST_RC="$(cat "${base}.rc" 2>/dev/null || echo TIMEOUT)"
  [[ -n "$LAST_RC" ]] || LAST_RC="TIMEOUT"
  LAST_OUT="${base}.out"
  log "run '${name}' finished (rc=${LAST_RC})"
}

# ---------------------------------------------------------------------------
# (g)+(h) Append a ledger line and print the human summary + delta.
# status: ok | gate_failed | run_failed | stalled
# ---------------------------------------------------------------------------
append_ledger() {
  local run_name="$1" out_file="$2" err_file="$3" status="$4"
  LB_TS="$TS" LB_STWO_REV="$STWO_REV" LB_STWO_DIRTY="$STWO_DIRTY" \
  LB_CAIRO_REV="$CAIRO_REV" LB_CAIRO_DIRTY="$CAIRO_DIRTY" LB_GPU="$POD_GPU" \
  LB_RUN="$run_name" LB_STATUS="$status" LB_OUT="$out_file" LB_ERR="$err_file" \
  LB_BENCH_ENV="$BENCH_ENV" LB_STALL="${LAST_STALL_FILE:-}" \
  LB_LEDGER="$LEDGER" python3 - <<'PY'
import os, json

led   = os.environ["LB_LEDGER"]
run   = os.environ["LB_RUN"]
gpu   = os.environ["LB_GPU"]
status= os.environ["LB_STATUS"]
benv  = os.environ.get("LB_BENCH_ENV", "")

def parse_lines(path):
    record, pipeline, phases = None, None, []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "phase_totals" in obj:
                    phases.append(obj)
                elif "pipeline" in obj:
                    pipeline = obj
                elif "program" in obj and "backend" in obj:
                    record = obj
    except FileNotFoundError:
        pass
    return record, pipeline, phases

record, pipeline, phases = parse_lines(os.environ.get("LB_OUT", ""))

# The comparison metric: sustained useful MHz for a pipelined (fleet) run, else the
# per-run useful MHz from the main record.
def metric_of(rec, pipe):
    if pipe and pipe.get("sustained_useful_mhz") is not None:
        return pipe["sustained_useful_mhz"]
    if rec:
        return rec.get("useful_mhz")
    return None

new_metric = metric_of(record, pipeline)

# Find the previous OK entry for the SAME run_name, SAME pod_gpu, and SAME bench_env.
# Only same-pod comparisons are meaningful, and a number produced with debug env
# (disabled streams, debug sync, ...) must never be compared with a clean number.
prev_metric = None
try:
    with open(led) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (e.get("run_name") == run and e.get("pod_gpu") == gpu
                    and e.get("status", "ok") == "ok"
                    and e.get("bench_env", "") == benv):
                m = metric_of(e.get("record"), e.get("pipeline"))
                if m is not None:
                    prev_metric = m
except FileNotFoundError:
    pass

entry = {
    "ts": os.environ["LB_TS"],
    "stwo_rev": os.environ["LB_STWO_REV"],
    "stwo_dirty": os.environ["LB_STWO_DIRTY"],
    "cairo_rev": os.environ["LB_CAIRO_REV"],
    "cairo_dirty": os.environ["LB_CAIRO_DIRTY"],
    "pod_gpu": gpu,
    "run_name": run,
    "status": status,
    "bench_env": benv,
    "record": record,
    "phase_totals": phases,
}
if pipeline:
    entry["pipeline"] = pipeline
if status != "ok":
    tail = ""
    try:
        with open(os.environ.get("LB_ERR", "")) as f:
            tail = "".join(f.readlines()[-15:])
    except (FileNotFoundError, KeyError):
        pass
    entry["error_tail"] = tail
stall_path = os.environ.get("LB_STALL", "")
if status == "stalled" and stall_path:
    try:
        with open(stall_path) as f:
            entry["stall_evidence"] = f.read()
    except FileNotFoundError:
        pass

os.makedirs(os.path.dirname(led), exist_ok=True)
with open(led, "a") as f:
    f.write(json.dumps(entry) + "\n")

# --- human summary (stdout); '!' marks a non-default BENCH_ENV run ---
disp = run + ("!" if benv else "")
if status != "ok":
    detail = "stall_evidence + " if status == "stalled" else ""
    print(f"  {disp:<21} status={status}  (see ledger {detail}error_tail)")
elif new_metric is None:
    print(f"  {disp:<21} useful_mhz=n/a")
else:
    label = "sustained_useful_mhz" if (pipeline and pipeline.get('sustained_useful_mhz') is not None) else "useful_mhz"
    if prev_metric is None:
        delta = "(no prior same-pod same-env run)"
    else:
        pct = (new_metric - prev_metric) / prev_metric * 100.0 if prev_metric else 0.0
        delta = f"{pct:+.1f}% vs {prev_metric:.3f}"
    vram = (record or {}).get("vram_peak_gb", "?")
    print(f"  {disp:<21} {label}={new_metric:.3f}  vram_peak_gb={vram}  {delta}")
PY
}

# ===========================================================================
# Orchestration
# ===========================================================================
log "=== bench_loop start (pie=${PIE_SEL} reps=${REPS} full=${FULL} simd=${SIMD} skip_sync=${SKIP_SYNC} gate_only=${GATE_ONLY} dry=${DRY_RUN}) ==="

if [[ "$SKIP_SYNC" == "1" ]]; then
  log "--skip-sync: reusing the binary already on the pod (no repo sync, no build)"
else
  sync_repos
  build_pod
fi

# Abort helper for a stalled run: self-documenting ledger entry with evidence, then die.
abort_stalled() {
  local nm="$1"
  append_ledger "$nm" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.${nm}.err" "stalled"
  die "run '${nm}' stalled — process killed; evidence in ${LAST_STALL_FILE} and the ledger entry."
}

# (d) CORRECTNESS GATE — always first, under the SAME BENCH_ENV as the benchmarks
# (a kill switch that changes behavior must be gated too; the launcher exports
# BENCH_ENV for every run including this one).
GATE_ARGS="--pie ${POD_GATE_PIE} --backend cuda --reps 1 --reuse-input"
log "=== CORRECTNESS GATE: 10-transfer PIE CUDA prove+verify ==="
run_bench "gate_10t" "$GATE_ARGS"
if [[ "$LAST_RC" == "STALLED" ]]; then
  abort_stalled "gate_10t"
fi
if [[ "$LAST_RC" != "0" ]]; then
  warn "GATE FAILED (rc=${LAST_RC}) — verify failed or the prover crashed."
  append_ledger "gate_failed" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.gate_10t.err" "gate_failed"
  die "correctness gate failed — refusing to report any performance from this build."
fi
append_ledger "gate_10t" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.gate_10t.err" "ok"
log "gate PASSED"

if [[ "$GATE_ONLY" == "1" ]]; then
  log "--gate-only: done."
  exit 0
fi

# Build the benchmark run list.
declare -a RUN_NAMES RUN_ARGS
add_run() { RUN_NAMES+=("$1"); RUN_ARGS+=("$2"); }

SEL_PATH="$(pie_path "$PIE_SEL")"
SEL_NAME="$(pie_name "$PIE_SEL")"
add_run "$SEL_NAME" "--pie ${SEL_PATH} --backend cuda --reps ${REPS} --reuse-input"

if [[ "$SIMD" == "1" ]]; then
  add_run "${SEL_NAME}_simd" "--pie ${SEL_PATH} --backend simd --reps ${REPS} --reuse-input"
fi

if [[ "$FULL" == "1" ]]; then
  for s in 1 3 4; do
    nm="$(pie_name "$s")"
    [[ "$nm" == "$SEL_NAME" ]] && continue   # already queued as the selected PIE
    add_run "$nm" "--pie $(pie_path "$s") --backend cuda --reps ${REPS} --reuse-input"
  done
  FLEET_LIST="${POD_SN_DIR}/SN_PIE_1.zip,${POD_SN_DIR}/SN_PIE_2.zip,${POD_SN_DIR}/SN_PIE_3.zip,${POD_SN_DIR}/SN_PIE_4.zip"
  add_run "SN_fleet_rotate" "--pie ${FLEET_LIST} --backend cuda --reps ${FLEET_REPS} --pipeline ${FLEET_DEPTH} --producers ${FLEET_PRODUCERS} --pie-mode rotate"
fi

log "=== benchmarking ${#RUN_NAMES[@]} run(s) ==="
FAILED=0
echo "=== bench_loop summary (${TS}) — pod ${POD_GPU} ==="
echo "    revs: stwo=${STWO_REV:0:8}${STWO_DIRTY:+(${STWO_DIRTY})} cairo=${CAIRO_REV:0:8}${CAIRO_DIRTY:+(${CAIRO_DIRTY})}"
[[ -n "$BENCH_ENV" ]] && echo "    bench_env(!): ${BENCH_ENV}  — NOT comparable with clean runs"
for idx in "${!RUN_NAMES[@]}"; do
  nm="${RUN_NAMES[$idx]}"; ar="${RUN_ARGS[$idx]}"
  run_bench "$nm" "$ar"
  if [[ "$LAST_RC" == "STALLED" ]]; then
    abort_stalled "$nm"
  fi
  if [[ "$LAST_RC" != "0" ]]; then
    warn "run '${nm}' failed (rc=${LAST_RC})"
    append_ledger "$nm" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.${nm}.err" "run_failed"
    FAILED=1
    continue
  fi
  append_ledger "$nm" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.${nm}.err" "ok"
done

echo "=== ledger: ${LEDGER} ==="
log "=== bench_loop done ==="
[[ "$FAILED" == "0" ]] || die "one or more benchmark runs failed (see warnings + ledger)"
exit 0
