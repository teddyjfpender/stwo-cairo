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
#      volume persists target/, Rustup/Cargo homes, and caches; the container
#      layer does NOT).
#   3. Stage and rsync the exact benchmark-relevant source projection of BOTH
#      repos (tracked plus non-ignored untracked files, excluding run state).
#   4. Install and verify the repo-pinned Rust toolchain after the toolchain
#      manifest exists on the pod.
#   5. Upload a PHASES fragment (your file) into a detached, setsid on-pod
#      session with per-phase rc/secs sentinels (an ssh drop can't kill it).
#   6. POLL each phase in order; print a per-phase rc + duration and stop on
#      the first failure.
#   7. Grep a standard evidence pattern set from every phase log.
#   8. Fetch all phase logs (+ an optional divergence dir) to results/<label>/.
#   9. Confirm the configured final lifecycle state on EVERY exit path (trap):
#      EXITED by default, or provider absence for one-shot termination.
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
#   * exact directive `# pod_run: require_clean_sources` rejects either local
#       repo unless its source hash is SHA-256(empty), before the pod is started.
#   * function `phase NAME CMD...` : run CMD, record $RUN/NAME.rc and
#       $RUN/NAME.secs, and abort the session on failure. The driver polls
#       exactly the NAMEs it finds by scanning your file for lines beginning
#       `phase `.
#   * exported env: cargo/cuda on PATH, RUSTUP_HOME, CARGO_HOME, STWO_CUDA_OBJ_CACHE,
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
# POD_RUN_POLL_INTERVAL controls phase-sentinel polling (default 30 seconds;
# quick_sn2.sh uses 2 seconds so short gates do not add minutes of idle time).
# The phases file must contain one strict `# pod_run: lease ...` provider
# contract. Ambient lifecycle settings may not weaken that recipe-bound policy.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
POD_CONF="${POD_CONF:-${SCRIPT_DIR}/pod.conf}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
SOURCE_PROJECTION_TOOL="${SCRIPT_DIR}/stage_source_projection.sh"
FLEET_CTL="${SCRIPT_DIR}/../fleet/gpufleet.sh"
POD_RUSTUP_HOME="${POD_RUSTUP_HOME:-/workspace/.rustup-persist}"
POD_CARGO_HOME="${POD_CARGO_HOME:-/workspace/.cargo-persist}"
POD_RUN_POLL_INTERVAL="${POD_RUN_POLL_INTERVAL:-30}"
[[ "$POD_RUN_POLL_INTERVAL" =~ ^[1-9][0-9]*$ && "$POD_RUN_POLL_INTERVAL" -le 60 ]] \
  || { echo "POD_RUN_POLL_INTERVAL must be an integer from 1 to 60 seconds" >&2; exit 2; }
printf -v POD_RUSTUP_HOME_Q '%q' "$POD_RUSTUP_HOME"
printf -v POD_CARGO_HOME_Q '%q' "$POD_CARGO_HOME"

PHASES_FILE="${1:?usage: pod_run.sh <phases_file> [label]}"
[[ -f "$PHASES_FILE" ]] || { echo "phases file not found: $PHASES_FILE" >&2; exit 2; }
[[ -x "$FLEET_CTL" ]] \
  || { echo "fleet lifecycle tool is absent or not executable: $FLEET_CTL" >&2; exit 2; }

LEASE_ONE_SHOT="" LEASE_FINAL_ACTION="" LEASE_GPU="" LEASE_GPU_COUNT=""
LEASE_MIN_VCPU="" LEASE_MIN_MEM_GB="" LEASE_MAX_USD_HR=""
LEASE_NAME_PREFIX="" LEASE_TTL_HOURS="" LEASE_IDLE_MIN=""
while IFS='=' read -r name value; do
  case "$name" in
    ONE_SHOT) LEASE_ONE_SHOT="$value" ;;
    FINAL_ACTION) LEASE_FINAL_ACTION="$value" ;;
    GPU) LEASE_GPU="$value" ;;
    GPU_COUNT) LEASE_GPU_COUNT="$value" ;;
    MIN_VCPU) LEASE_MIN_VCPU="$value" ;;
    MIN_MEM_GB) LEASE_MIN_MEM_GB="$value" ;;
    MAX_USD_HR) LEASE_MAX_USD_HR="$value" ;;
    NAME_PREFIX) LEASE_NAME_PREFIX="$value" ;;
    TTL_HOURS) LEASE_TTL_HOURS="$value" ;;
    IDLE_MIN) LEASE_IDLE_MIN="$value" ;;
    *) echo "unexpected lease-policy output: $name" >&2; exit 2 ;;
  esac
done < <("$FLEET_CTL" lease-policy --recipe "$PHASES_FILE") \
  || { echo "invalid or missing recipe lease policy" >&2; exit 2; }
for value in "$LEASE_ONE_SHOT" "$LEASE_FINAL_ACTION" "$LEASE_GPU" \
  "$LEASE_GPU_COUNT" "$LEASE_MIN_VCPU" "$LEASE_MIN_MEM_GB" \
  "$LEASE_MAX_USD_HR" "$LEASE_NAME_PREFIX" "$LEASE_TTL_HOURS" "$LEASE_IDLE_MIN"; do
  [[ -n "$value" ]] || { echo "incomplete recipe lease policy" >&2; exit 2; }
done
if [[ -n "${POD_RUN_FINAL_ACTION:-}" &&
      "$POD_RUN_FINAL_ACTION" != "$LEASE_FINAL_ACTION" ]]; then
  echo "POD_RUN_FINAL_ACTION conflicts with the recipe lease policy" >&2
  exit 2
fi
POD_RUN_FINAL_ACTION="$LEASE_FINAL_ACTION"
LABEL="${2:-pod_run_$(date -u +%Y%m%dT%H%M%SZ)}"
REQUIRE_CLEAN_SOURCES=0
grep -Fqx '# pod_run: require_clean_sources' "$PHASES_FILE" && REQUIRE_CLEAN_SOURCES=1

# --- pod identity ---
POD_ID="${BENCH_POD_ID:-}"
KEY=""
if [[ -f "$POD_CONF" ]]; then
  if [[ "$LEASE_ONE_SHOT" != 1 && -z "$POD_ID" ]]; then
    POD_ID="$(sed -n 's/^POD_ID=//p' "$POD_CONF" | head -1)"
  fi
  KEY="$(sed -n 's/^FALLBACK_KEY=//p' "$POD_CONF" | head -1)"
fi
if [[ "$LEASE_ONE_SHOT" == 1 && -z "$POD_ID" ]]; then
  echo "one-shot recipes require an explicit BENCH_POD_ID" >&2
  exit 2
fi
: "${POD_ID:?set POD_ID in pod.conf or BENCH_POD_ID=...}"
KEY="${RUNPOD_SSH_KEY:-$KEY}"

STWO_POD=/workspace/stwo
CAIRO_POD=/workspace/stwo-cairo
RUN=/workspace/bench_loop_runs/pod_run
SSH_OPTS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=20)

note() { echo "[pod_run $(date -u +%H:%M:%S)] $*"; }
pssh() { ssh "${SSH_OPTS[@]}" -i "$KEY" -p "$PORT" "root@$HOST" "$@"; }

source_head() {
  git -C "$1" rev-parse HEAD 2>/dev/null
}

# Match the transported source projection: the head is recorded separately,
# while this hash binds tracked changes plus non-ignored untracked paths/content.
# Runtime receipts and large PIE fixtures persist remotely but are not source.
source_hash() {
  local repo="$1"
  (
    cd "${SCRIPT_DIR}/../fleet" || exit 1
    ./gpufleet.sh source-hash --repo "$repo"
  )
}

valid_source_identity() {
  [[ ${#1} -eq 40 && "$1" != *[!0-9a-f]* &&
     ${#2} -eq 64 && "$2" != *[!0-9a-f]* ]]
}

[[ -x "$SOURCE_PROJECTION_TOOL" ]] \
  || { echo "source projection tool is absent or not executable: $SOURCE_PROJECTION_TOOL" >&2; exit 2; }

STWO_HEAD="$(source_head "$STWO_LOCAL")" \
  || { echo "cannot resolve stwo source head: $STWO_LOCAL" >&2; exit 2; }
STWO_WORKTREE_HASH="$(source_hash "$STWO_LOCAL")" \
  || { echo "cannot hash stwo worktree: $STWO_LOCAL" >&2; exit 2; }
CAIRO_HEAD="$(source_head "$CAIRO_LOCAL")" \
  || { echo "cannot resolve stwo-cairo source head: $CAIRO_LOCAL" >&2; exit 2; }
CAIRO_WORKTREE_HASH="$(source_hash "$CAIRO_LOCAL")" \
  || { echo "cannot hash stwo-cairo worktree: $CAIRO_LOCAL" >&2; exit 2; }
valid_source_identity "$STWO_HEAD" "$STWO_WORKTREE_HASH" \
  || { echo "invalid stwo source identity" >&2; exit 2; }
valid_source_identity "$CAIRO_HEAD" "$CAIRO_WORKTREE_HASH" \
  || { echo "invalid stwo-cairo source identity" >&2; exit 2; }
EMPTY_SOURCE_HASH=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
if [[ "$REQUIRE_CLEAN_SOURCES" == 1 &&
      ( "$STWO_WORKTREE_HASH" != "$EMPTY_SOURCE_HASH" ||
        "$CAIRO_WORKTREE_HASH" != "$EMPTY_SOURCE_HASH" ) ]]; then
  echo "phase contract requires clean stwo and stwo-cairo sources" >&2
  exit 2
fi

PHASE_NAMES="$(awk '$1=="phase"{print $2}' "$PHASES_FILE")"
[[ -n "$PHASE_NAMES" ]] || { echo "no 'phase NAME ...' lines in $PHASES_FILE" >&2; exit 2; }
PHASE_DUPLICATES="$(printf '%s\n' "$PHASE_NAMES" | sort | uniq -d)"
[[ -z "$PHASE_DUPLICATES" ]] \
  || { echo "duplicate phase names: $PHASE_DUPLICATES" >&2; exit 2; }
INVALID_PHASE_NAMES="$(printf '%s\n' "$PHASE_NAMES" | awk '$0 !~ /^[A-Za-z0-9][A-Za-z0-9._-]*$/')"
[[ -z "$INVALID_PHASE_NAMES" ]] \
  || { echo "invalid phase names: $INVALID_PHASE_NAMES" >&2; exit 2; }
note "phases: $(echo "$PHASE_NAMES" | tr '\n' ' ')"
note "label:  $LABEL"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  note "DRY_RUN: pod=$POD_ID key=$KEY"
  note "DRY_RUN: RUSTUP_HOME=$POD_RUSTUP_HOME CARGO_HOME=$POD_CARGO_HOME"
  note "DRY_RUN: lease gpu=$LEASE_GPU count=$LEASE_GPU_COUNT name_prefix=$LEASE_NAME_PREFIX max_usd_hr=$LEASE_MAX_USD_HR final_action=$POD_RUN_FINAL_ACTION"
  note "DRY_RUN: source stwo=${STWO_HEAD}:${STWO_WORKTREE_HASH} stwo-cairo=${CAIRO_HEAD}:${CAIRO_WORKTREE_HASH}"
  note "DRY_RUN: would bootstrap, stage and rsync exact source projections from $STWO_LOCAL and $CAIRO_LOCAL, install the pinned toolchain, run the phases above, fetch to $RESULTS_DIR/$LABEL, then confirm pod action=$POD_RUN_FINAL_ACTION."
  exit 0
fi

POD_FINALIZE_ATTEMPTED=0
POD_FINALIZE_RC=1
POD_LIFECYCLE_OWNED=0
PROJECTION_ROOT=""
finalize_pod() {
  [[ "$POD_LIFECYCLE_OWNED" == 1 ]] || return 0
  [[ "$POD_FINALIZE_ATTEMPTED" == 0 ]] || return "$POD_FINALIZE_RC"
  POD_FINALIZE_ATTEMPTED=1
  note "requesting and confirming pod action=$POD_RUN_FINAL_ACTION for $POD_ID"
  if "$FLEET_CTL" "$POD_RUN_FINAL_ACTION" --pod "$POD_ID"; then
    POD_FINALIZE_RC=0
    return 0
  fi
  note "ERROR: pod $POD_RUN_FINAL_ACTION was not confirmed"
  return 1
}
cleanup() {
  local rc=$?
  trap - EXIT
  [[ -z "$PROJECTION_ROOT" || ! -d "$PROJECTION_ROOT" ]] \
    || rm -rf -- "$PROJECTION_ROOT"
  if ! finalize_pod; then
    [[ "$rc" != 0 ]] || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT

# --- 1. provider admission + guarded resume/adoption + endpoint resolution ---
[[ -n "$KEY" && -f "$KEY" ]] \
  || { note "RUNPOD SSH private key is missing: $KEY"; exit 1; }
printf -v KEY_Q '%q' "$KEY"
RESUME_ARGS=(
  resume --pod "$POD_ID" --gpu "$LEASE_GPU"
  --name-prefix "$LEASE_NAME_PREFIX"
  --max-usd-hr "$LEASE_MAX_USD_HR"
  --min-vcpu "$LEASE_MIN_VCPU" --min-mem-gb "$LEASE_MIN_MEM_GB"
  --ttl-hours "$LEASE_TTL_HOURS" --idle-min "$LEASE_IDLE_MIN"
  --failure-action "$POD_RUN_FINAL_ACTION"
  --purpose "pod-run-$LABEL"
)
[[ "$LEASE_ONE_SHOT" == 1 ]] && RESUME_ARGS+=(--one-shot)
note "admitting provider lease and installing deadman before proof work"
RUNPOD_SSH_KEY="$KEY" "$FLEET_CTL" "${RESUME_ARGS[@]}" \
  || { note "PROVIDER ADMISSION/RESUME FAILED"; exit 1; }
POD_LIFECYCLE_OWNED=1

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
pssh "set -e
  export RUSTUP_HOME=$POD_RUSTUP_HOME_Q
  export CARGO_HOME=$POD_CARGO_HOME_Q
  export PATH=\"\$CARGO_HOME/bin:\$PATH\"
  mkdir -p \"\$RUSTUP_HOME\" \"\$CARGO_HOME\"
  command -v rsync >/dev/null 2>&1 || { apt-get update -qq >/dev/null && apt-get install -y -qq rsync >/dev/null; }
  if [ ! -x \"\$CARGO_HOME/bin/rustup\" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null
  fi" || { note "BOOTSTRAP FAILED"; exit 1; }

# --- 3. stage the exact hashed source projection, then rsync both repos ---
PROJECTION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/stwo-pod-run.XXXXXX")"
note "stage content-hashed source projections"
"$SOURCE_PROJECTION_TOOL" "$STWO_LOCAL" "$PROJECTION_ROOT/stwo" \
  || { note "STAGE stwo source projection FAILED"; exit 1; }
"$SOURCE_PROJECTION_TOOL" "$CAIRO_LOCAL" "$PROJECTION_ROOT/stwo-cairo" \
  || { note "STAGE stwo-cairo source projection FAILED"; exit 1; }
note "rsync stwo"
rsync -azc --delete --partial --no-owner --no-group --perms --no-times \
  --exclude=target --exclude=.git \
  -e "ssh ${SSH_OPTS[*]} -i $KEY_Q -p $PORT" \
  "$PROJECTION_ROOT/stwo/" "root@${HOST}:${STWO_POD}/" \
  || { note "SYNC stwo FAILED"; exit 1; }
note "rsync stwo-cairo"
rsync -azc --delete --partial --no-owner --no-group --perms --no-times \
  --exclude=target --exclude=.git \
  --exclude='gpu_benchmarks/pie/sn/' --exclude='gpu_benchmarks/pie/*.zip' \
  --exclude='gpu_benchmarks/loop/results' --exclude='gpu_benchmarks/loop/ledger.jsonl' \
  -e "ssh ${SSH_OPTS[*]} -i $KEY_Q -p $PORT" \
  "$PROJECTION_ROOT/stwo-cairo/" "root@${HOST}:${CAIRO_POD}/" \
  || { note "SYNC stwo-cairo FAILED"; exit 1; }

# The cache fallback below is valid only for the exact source tree transported
# by this run. Reject a local edit racing the checksum sync.
[[ "$STWO_HEAD" == "$(source_head "$STWO_LOCAL")" &&
   "$STWO_WORKTREE_HASH" == "$(source_hash "$STWO_LOCAL")" &&
   "$CAIRO_HEAD" == "$(source_head "$CAIRO_LOCAL")" &&
   "$CAIRO_WORKTREE_HASH" == "$(source_hash "$CAIRO_LOCAL")" ]] \
  || { note "LOCAL SOURCES CHANGED DURING SYNC"; exit 1; }

# --- 4. install + verify both repo-pinned Rust toolchains ---
note "install pinned Rust toolchains"
pssh "set -e
  export RUSTUP_HOME=$POD_RUSTUP_HOME_Q
  export CARGO_HOME=$POD_CARGO_HOME_Q
  export PATH=\"\$CARGO_HOME/bin:\$PATH\"
  for repo in '$STWO_POD' '$CAIRO_POD/stwo_cairo_prover'; do
    cd \"\$repo\"
    rustup toolchain install
    printf 'toolchain[%s]=' \"\$repo\"
    rustc --version
  done" || { note "TOOLCHAIN INSTALL FAILED"; exit 1; }

# --- 5. upload + launch the detached session with rc sentinels ---
note "upload + launch session"
{
  cat <<'PROLOGUE_HEADER'
#!/usr/bin/env bash
set -euo pipefail
PROLOGUE_HEADER
  printf 'export RUSTUP_HOME=%q\nexport CARGO_HOME=%q\n' "$POD_RUSTUP_HOME" "$POD_CARGO_HOME"
  printf 'export STWO_PARITY_REF_STWO_HEAD=%q\n' "$STWO_HEAD"
  printf 'export STWO_PARITY_REF_STWO_WORKTREE_HASH=%q\n' "$STWO_WORKTREE_HASH"
  printf 'export STWO_PARITY_REF_STWO_CAIRO_HEAD=%q\n' "$CAIRO_HEAD"
  printf 'export STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH=%q\n' "$CAIRO_WORKTREE_HASH"
  cat <<'PROLOGUE'
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
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
  local rc
  set +e
  ( set -e; "$@" ) > "$RUN/$name.log" 2>&1
  rc=$?
  set -e
  echo $((SECONDS-t0)) > "$RUN/$name.secs"
  echo "$rc" > "$RUN/$name.rc"
  return "$rc"
}
PROLOGUE
  cat "$PHASES_FILE"
  # shellcheck disable=SC2016 # $RUN expands in the generated pod-side script.
  echo 'echo done > "$RUN/session.done"'
} | pssh "mkdir -p '$RUN' && cat > '$RUN/session.sh'" \
  || { note "UPLOAD FAILED"; exit 1; }
pssh "cd '$RUN' && rm -rf divergence *.bin *.log *.rc *.secs *.csv *.json *.ncu-rep *.nsys-rep *.qdrep *.sqlite *.txt *.xml session.done && nohup setsid -f bash '$RUN/session.sh' </dev/null > session.out 2>&1 && echo LAUNCHED" \
  || { note "LAUNCH FAILED"; exit 1; }

# --- 6. poll phases in order ---
DEADLINE=$((SECONDS + ${MAX_WAIT:-10800}))
RUN_RC=0
OBSERVED_PHASES=""
for p in $PHASE_NAMES; do
  PHASE_FAILED=0
  note "waiting on phase: $p"
  while true; do
    if (( SECONDS > DEADLINE )); then
      note "TIMEOUT on $p"
      RUN_RC=1
      PHASE_FAILED=1
      break
    fi
    rc="$(pssh "cat '$RUN/$p.rc' 2>/dev/null" 2>/dev/null || true)"
    if [[ -n "$rc" ]]; then
      secs="$(pssh "cat '$RUN/$p.secs' 2>/dev/null" 2>/dev/null || true)"
      note "phase $p rc=$rc (${secs:-?}s)"
      OBSERVED_PHASES="${OBSERVED_PHASES}${p}"$'\n'
      if [[ ! "$rc" =~ ^[0-9]+$ ]] || (( rc > 255 )); then
        note "invalid phase rc for $p: $rc"
        RUN_RC=1
        PHASE_FAILED=1
      elif [[ "$rc" != 0 ]]; then
        RUN_RC="$rc"
        PHASE_FAILED=1
      fi
      break
    fi
    alive="$(pssh "pgrep -f '$RUN/[s]ession.sh' >/dev/null && echo yes || echo no" 2>/dev/null || echo unknown)"
    if [[ "$alive" == "no" ]]; then
      note "session died before $p"
      pssh "tail -30 '$RUN/session.out'" || true
      RUN_RC=1
      PHASE_FAILED=1
      break
    fi
    sleep "$POD_RUN_POLL_INTERVAL"
  done
  (( PHASE_FAILED == 0 )) || break
done

# A phase writes its rc before the phase fragment has fully returned. A run is
# successful only after the detached session reaches its final sentinel.
if [[ "$RUN_RC" == 0 ]]; then
  note "waiting on successful session completion"
  while true; do
    done_value="$(pssh "cat '$RUN/session.done' 2>/dev/null" 2>/dev/null || true)"
    [[ "$done_value" == "done" ]] && break
    if (( SECONDS > DEADLINE )); then
      note "TIMEOUT waiting for session completion"
      RUN_RC=1
      break
    fi
    alive="$(pssh "pgrep -f '$RUN/[s]ession.sh' >/dev/null && echo yes || echo no" 2>/dev/null || echo unknown)"
    if [[ "$alive" == no ]]; then
      note "session died without successful completion"
      pssh "tail -30 '$RUN/session.out'" || true
      RUN_RC=1
      break
    fi
    sleep 2
  done
fi

# --- 7. standard evidence grep ---
for p in $PHASE_NAMES; do
  note "--- $p evidence ---"
  pssh "grep -E 'verdict|DIVERGENCE|GEOMETRY|ORDER|PADDING|CONTENT|smoke proof section|smoke divergence|first mismatch|panicked|drifted|useful_mhz|\"pass\"|test result' '$RUN/$p.log' 2>/dev/null | head -60" || true
done

# --- 8. fetch evidence ---
note "fetching evidence to $RESULTS_DIR/$LABEL"
mkdir -p "$RESULTS_DIR/$LABEL"
scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" "root@${HOST}:$RUN/*.log" "root@${HOST}:$RUN/*.secs" \
  "$RESULTS_DIR/$LABEL/" 2>/dev/null \
  || { note "ERROR: phase log/secs fetch failed"; RUN_RC=1; }
scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" "root@${HOST}:$RUN/*.rc" \
  "$RESULTS_DIR/$LABEL/" 2>/dev/null \
  || { note "ERROR: phase rc fetch failed"; RUN_RC=1; }
# Profiling recipes and formal checkpoints keep raw reports/proofs alongside
# the phase logs. Fetch every present optional artifact before the pod stops;
# absence is expected for ordinary benchmark recipes.
for suffix in bin csv json ncu-rep nsys-rep qdrep sqlite txt xml; do
  if pssh "compgen -G '$RUN/*.$suffix' >/dev/null" 2>/dev/null; then
    scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" "root@${HOST}:$RUN/*.$suffix" \
      "$RESULTS_DIR/$LABEL/" 2>/dev/null \
      || { note "ERROR: optional *.$suffix artifact fetch failed"; RUN_RC=1; }
  fi
done
for p in $OBSERVED_PHASES; do
  for suffix in log secs rc; do
    [[ -f "$RESULTS_DIR/$LABEL/$p.$suffix" ]] \
      || { note "ERROR: missing local evidence $p.$suffix"; RUN_RC=1; }
  done
done
pssh "test -d '$RUN/divergence'" 2>/dev/null \
  && scp "${SSH_OPTS[@]}" -i "$KEY" -P "$PORT" -r "root@${HOST}:$RUN/divergence" "$RESULTS_DIR/$LABEL/" 2>/dev/null

# --- 9. confirmed final lifecycle action (also via trap) ---
finalize_pod || RUN_RC=1
if [[ "$RUN_RC" != 0 ]]; then
  note "FAILED (rc=$RUN_RC) — results in $RESULTS_DIR/$LABEL"
  exit "$RUN_RC"
fi
note "done — results in $RESULTS_DIR/$LABEL"
