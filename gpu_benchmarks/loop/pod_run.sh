#!/usr/bin/env bash
#
# pod_run.sh — durable, parameterized RunPod session driver for the strict
# GPU-resident Stwo-Cairo prover. Generalizes the ad-hoc drivers used during
# the 2026-07-11 divergence hunt (build/adapt/preflight/smoke, diagnostics,
# trace-audit + pow-gate) into one reusable tool.
#
# What it does, every run, in order:
#   1. START the pod (idempotent) and RE-RESOLVE its ssh endpoint. Community
#      pods return a NEW host/port on each resume — never trust a cached one.
#   2. BOOTSTRAP the reset container layer: apt rsync + rustup (the /workspace
#      volume persists target/ + caches; the container layer does NOT).
#   3. rsync BOTH repos (stwo, stwo-cairo) with the exact excludes bench_loop
#      uses, so a later bench_loop sync is a no-op.
#   4. Upload a PHASES fragment (your file) into a detached, setsid on-pod
#      session with per-phase rc/secs sentinels (an ssh drop can't kill it).
#   5. POLL each phase in order; print a per-phase rc + duration.
#   6. Grep a standard evidence pattern set from every phase log.
#   7. Fetch all phase logs (+ an optional divergence dir) to results/<label>/.
#   8. STOP the pod on EVERY exit path (trap) — failed rounds cost cents, not
#      an idle-pod bleed.
#
# Usage:
#   ./pod_run.sh <phases_file> [label]
#
#   <phases_file>  a bash fragment (see PHASE CONTRACT below).
#   [label]        results subdir name (default: pod_run_<UTC stamp>).
#
# Pod identity comes from loop/pod.conf (POD_ID=...). Override with
# BENCH_POD_ID=... in the environment. SSH key from pod.conf FALLBACK_KEY or
# resolved by `runpodctl ssh info`.
#
# PHASE CONTRACT — the phases_file runs ON THE POD with these available:
#   * function `phase NAME CMD...` : run CMD (never aborts siblings), record
#       $RUN/NAME.rc and $RUN/NAME.secs. The driver polls exactly the NAMEs it
#       finds by scanning your file for lines beginning `phase `.
#   * exported env: cargo/cuda on PATH, STWO_CUDA_OBJ_CACHE,
#       STWO_PARITY_REF_CACHE, STWO_SMOKE_DIVERGENCE_DIR, RUST_MIN_STACK=32Mi,
#       STWO_BOOTLOADER_JSON.
#   * vars: $CAIRO (=/workspace/stwo-cairo/stwo_cairo_prover),
#           $STWO  (=/workspace/stwo),
#           $RUN   (=/workspace/bench_loop_runs/pod_run).
#   Write extra artifacts under $RUN to have them fetched.
#
# Example phases_file (trace audit + pow gate, the round-4 recipe):
#   cd "$CAIRO"
#   phase trace_audit cargo test -p stwo-cairo-gpu-prover --test resident_trace_audit -- --nocapture --test-threads=1
#   cd "$STWO"
#   phase pow_gate cargo test -p stwo-backend-cuda --test prepared_fri_final_pow_native -- --nocapture
#
# DRY_RUN=1 prints the plan (endpoint resolve, rsync, phase names) without
# starting the pod.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
POD_CONF="${POD_CONF:-${SCRIPT_DIR}/pod.conf}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

PHASES_FILE="${1:?usage: pod_run.sh <phases_file> [label]}"
[[ -f "$PHASES_FILE" ]] || { echo "phases file not found: $PHASES_FILE" >&2; exit 2; }
LABEL="${2:-pod_run_$(date -u +%Y%m%dT%H%M%SZ)}"

# --- pod identity ---
POD_ID="${BENCH_POD_ID:-}"
KEY=""
if [[ -f "$POD_CONF" ]]; then
  [[ -z "$POD_ID" ]] && POD_ID="$(sed -n 's/^POD_ID=//p' "$POD_CONF" | head -1)"
  KEY="$(sed -n 's/^FALLBACK_KEY=//p' "$POD_CONF" | head -1)"
fi
: "${POD_ID:?set POD_ID in pod.conf or BENCH_POD_ID=...}"
KEY="${RUNPOD_SSH_KEY:-$KEY}"

STWO_POD=/workspace/stwo
CAIRO_POD=/workspace/stwo-cairo
RUN=/workspace/bench_loop_runs/pod_run
SSH_OPTS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=20)

note() { echo "[pod_run $(date -u +%H:%M:%S)] $*"; }
pssh() { ssh "${SSH_OPTS[@]}" -i "$KEY" -p "$PORT" "root@$HOST" "$@"; }

PHASE_NAMES="$(awk '$1=="phase"{print $2}' "$PHASES_FILE")"
[[ -n "$PHASE_NAMES" ]] || { echo "no 'phase NAME ...' lines in $PHASES_FILE" >&2; exit 2; }
note "phases: $(echo "$PHASE_NAMES" | tr '\n' ' ')"
note "label:  $LABEL"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  note "DRY_RUN: pod=$POD_ID key=$KEY"
  note "DRY_RUN: would rsync $STWO_LOCAL and $CAIRO_LOCAL, run the phases above, fetch to $RESULTS_DIR/$LABEL, stop the pod."
  exit 0
fi

POD_STOPPED=0
stop_pod() {
  [[ "$POD_STOPPED" == 1 ]] && return 0
  POD_STOPPED=1
  note "stopping pod $POD_ID"
  runpodctl pod stop "$POD_ID" 2>/dev/null || runpodctl stop pod "$POD_ID" 2>/dev/null \
    || note "WARN: pod stop FAILED — run 'runpodctl pod stop $POD_ID' manually!"
}
trap stop_pod EXIT

# --- 1. start + resolve endpoint (new port on every resume) ---
note "starting pod $POD_ID"
runpodctl pod start "$POD_ID" 2>/dev/null || runpodctl start pod "$POD_ID" 2>/dev/null || true
HOST=""; PORT=""
for _ in $(seq 1 40); do
  info="$(runpodctl ssh info "$POD_ID" 2>/dev/null)"
  HOST="$(printf '%s' "$info" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("ip",""))' 2>/dev/null)"
  PORT="$(printf '%s' "$info" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("port",""))' 2>/dev/null)"
  if [[ -n "$HOST" && -n "$PORT" ]] && ssh "${SSH_OPTS[@]}" -i "$KEY" -p "$PORT" "root@$HOST" true 2>/dev/null; then
    break
  fi
  HOST=""; PORT=""
  sleep 15
done
[[ -n "$HOST" && -n "$PORT" ]] || { note "FAILED to reach pod ssh endpoint"; exit 1; }
note "endpoint: $HOST:$PORT"

# --- 2. bootstrap the reset container layer ---
note "bootstrap (rsync + rustup)"
pssh 'set -e
  command -v rsync >/dev/null 2>&1 || { apt-get update -qq >/dev/null && apt-get install -y -qq rsync >/dev/null; }
  if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null
  fi
  . "$HOME/.cargo/env"
  cd /workspace/stwo-cairo/stwo_cairo_prover && rustup toolchain install 2>/dev/null || true
  rustc --version' || { note "BOOTSTRAP FAILED"; exit 1; }

# --- 3. rsync both repos (bench_loop-identical excludes) ---
note "rsync stwo"
rsync -azc --delete --partial --no-owner --no-group --exclude=target --exclude=.git \
  -e "ssh ${SSH_OPTS[*]} -i $KEY -p $PORT" \
  "${STWO_LOCAL}/" "root@${HOST}:${STWO_POD}/" || { note "SYNC stwo FAILED"; exit 1; }
note "rsync stwo-cairo"
rsync -azc --delete --partial --no-owner --no-group --exclude=target --exclude=.git \
  --exclude='gpu_benchmarks/pie/sn/*.zip' --exclude='gpu_benchmarks/pie/*.zip' \
  --exclude='gpu_benchmarks/loop/results' --exclude='gpu_benchmarks/loop/ledger.jsonl' \
  -e "ssh ${SSH_OPTS[*]} -i $KEY -p $PORT" \
  "${CAIRO_LOCAL}/" "root@${HOST}:${CAIRO_POD}/" || { note "SYNC stwo-cairo FAILED"; exit 1; }

# --- 4. upload + launch the detached session with rc sentinels ---
note "upload + launch session"
{
  cat <<'PROLOGUE'
#!/usr/bin/env bash
set -u
. "$HOME/.cargo/env" 2>/dev/null
export PATH=/usr/local/cuda/bin:$PATH
export STWO_CUDA_OBJ_CACHE=/workspace/.cuda_obj_cache
export STWO_PARITY_REF_CACHE=/workspace/.parity_ref_cache
export STWO_SMOKE_DIVERGENCE_DIR=/workspace/bench_loop_runs/pod_run/divergence
export STWO_BOOTLOADER_JSON=/workspace/bench_inputs/simple_bootloader_compiled.json
export RUST_MIN_STACK=33554432
CAIRO=/workspace/stwo-cairo/stwo_cairo_prover
STWO=/workspace/stwo
RUN=/workspace/bench_loop_runs/pod_run
mkdir -p "$RUN"
phase() {
  local name="$1"; shift
  local t0=$SECONDS
  ( "$@" ) > "$RUN/$name.log" 2>&1
  local rc=$?
  echo $((SECONDS-t0)) > "$RUN/$name.secs"
  echo "$rc" > "$RUN/$name.rc"
}
PROLOGUE
  cat "$PHASES_FILE"
  echo 'echo done > "$RUN/session.done"'
} | pssh "mkdir -p '$RUN' && cat > '$RUN/session.sh'"
# shellcheck disable=SC2016
pssh "cd '$RUN' && rm -rf divergence *.log *.rc *.secs session.done && nohup setsid -f bash '$RUN/session.sh' </dev/null > session.out 2>&1 && echo LAUNCHED" \
  || { note "LAUNCH FAILED"; exit 1; }

# --- 5. poll phases in order ---
DEADLINE=$((SECONDS + ${MAX_WAIT:-10800}))
for p in $PHASE_NAMES; do
  note "waiting on phase: $p"
  while true; do
    if (( SECONDS > DEADLINE )); then note "TIMEOUT on $p"; break; fi
    rc="$(pssh "cat '$RUN/$p.rc' 2>/dev/null" 2>/dev/null || true)"
    if [[ -n "$rc" ]]; then
      secs="$(pssh "cat '$RUN/$p.secs' 2>/dev/null" 2>/dev/null || true)"
      note "phase $p rc=$rc (${secs:-?}s)"
      break
    fi
    alive="$(pssh "pgrep -f '$RUN/[s]ession.sh' >/dev/null && echo yes || echo no" 2>/dev/null || echo unknown)"
    [[ "$alive" == "no" ]] && { note "session died before $p"; pssh "tail -30 '$RUN/session.out'" || true; break; }
    sleep 30
  done
done

# --- 6. standard evidence grep ---
for p in $PHASE_NAMES; do
  note "--- $p evidence ---"
  pssh "grep -E 'verdict|DIVERGENCE|GEOMETRY|ORDER|PADDING|CONTENT|smoke proof section|smoke divergence|first mismatch|panicked|drifted|useful_mhz|\"pass\"|test result' '$RUN/$p.log' 2>/dev/null | head -60" || true
done

# --- 7. fetch evidence ---
note "fetching evidence to $RESULTS_DIR/$LABEL"
mkdir -p "$RESULTS_DIR/$LABEL"
scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" "root@${HOST}:$RUN/*.log" "root@${HOST}:$RUN/*.secs" \
  "$RESULTS_DIR/$LABEL/" 2>/dev/null || note "WARN: log fetch failed"
pssh "test -d '$RUN/divergence'" 2>/dev/null \
  && scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" -r "root@${HOST}:$RUN/divergence" "$RESULTS_DIR/$LABEL/" 2>/dev/null

# --- 8. stop (also via trap) ---
stop_pod
note "done — results in $RESULTS_DIR/$LABEL"
