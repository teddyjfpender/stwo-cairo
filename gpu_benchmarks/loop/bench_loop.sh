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
#   (b) sync to pod        : fast rsync delta; the portable Cargo.toml [patch] resolves
#                            the sibling /workspace/stwo checkout directly
#   (c) incremental build  : cargo build gpu_bench; abort loudly with the log tail
#   (d) CORRECTNESS GATE   : configured gate PIE CUDA prove+verify FIRST, under the SAME
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
#   (h) human summary      : useful_mhz_median per fixed-statement run (sustained
#                            useful MHz for pipelines) + same-host delta
#                            for the SAME run_name, SAME pod_gpu, SAME bench_env.
#
# Usage:
#   ./bench_loop.sh [--pie {1|2|3|4|10t}] [--reps N] [--all-pies|--full] [--simd]
#                   [--skip-sync] [--gate-only] [--help]
#
# Flags:
#   --pie SEL     Which SN PIE to benchmark: 1|2|3|4 (SN_PIE_<n>.zip) or 10t
#                 (the 10-transfer PIE). Default: 2.
#   --reps N      Repetitions per fixed-statement run. Default: 6 = one cold +
#                 five warm samples; published claims use the warm median.
#   --full        Also benchmark SN_PIE_1/3/4 (CUDA) and run the rotate-mode fleet
#                 (pipelined stream over all four PIEs).
#   --all-pies    Benchmark SN_PIE_1/2/3/4 without the rotate-mode fleet.
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
#                 STWO_CUDA_DEBUG_SYNC, CUDA_LAUNCH_BLOCKING, ... (shell-safe values only).
#   DRY_RUN=1     Echo every ssh/rsync instead of executing; fabricate run output so
#                 the provenance -> ledger -> summary path runs for real offline.
#   FAKE_STALL    (DRY_RUN only) name of a run to simulate as stalled, to exercise
#                 the stall -> evidence -> ledger -> abort path.
#   POLL_INTERVAL Seconds between pod poll checks (default 15).
#   STALL_SECS    Declare a run stalled when its stderr size AND the GPU utilization
#                 are both unchanged for this long (default 600).
#   MAX_WAIT      Hard cap on waiting for one run (default 10800 = 3h).
#   SN_PIE_SOURCE_DIR      Optional local directory containing SN_PIE_<n>.zip files.
#                 Files are hash-checked and an explicit pod seed command is printed;
#                 they are never uploaded automatically.
#   GATE_PIE_SOURCE        Optional local gate PIE source, handled likewise.
#   GATE_PIE       Remote correctness-gate PIE path. If the 10-transfer fixture is
#                  unavailable, explicitly use .../pie/sn/SN_PIE_2.zip; verification
#                  still runs on that larger fixture and is never silently skipped.
#   BOOTLOADER_JSON_SOURCE Optional local simple_bootloader_compiled.json source.
#                 Hash-checked and included in explicit seed commands only.
#   POD_BOOTLOADER_JSON    Stable remote bootloader path exported for build and every
#                 run. Default: /workspace/bench_inputs/simple_bootloader_compiled.json.
#   GPU_PCS_RUNTIME_MODE   Required typed CUDA PCS mode for every CUDA run:
#                 arena-graph (default) or detached-eager (migration diagnostics).
#
# NOTE: only same-pod comparisons are meaningful (community-host variance). See README.md.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all paths as variables, up top)
# ---------------------------------------------------------------------------
# Local repos + outputs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
LOOP_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${RESULTS_DIR:-${LOOP_DIR}/results}"
LEDGER="${LEDGER:-${LOOP_DIR}/ledger.jsonl}"
POD_CONF="${POD_CONF:-${LOOP_DIR}/pod.conf}"
INPUT_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/SHA256SUMS"
PINNED_ADAPTED_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/ADAPTED_SHA256SUMS"
ARCHITECTURE_CHECK="${CAIRO_LOCAL}/gpu_benchmarks/validate_architecture_record.py"
SOUNDNESS_RUNNER="${CAIRO_LOCAL}/gpu_benchmarks/run_cuda_soundness_gate.py"

# Optional local input sources. They are validation/seed hints only: this script
# never uploads fixtures or the bootloader implicitly.
SN_PIE_SOURCE_DIR="${SN_PIE_SOURCE_DIR:-}"
GATE_PIE_SOURCE="${GATE_PIE_SOURCE:-}"
BOOTLOADER_JSON_SOURCE="${BOOTLOADER_JSON_SOURCE:-}"

# Pod repos.
STWO_POD="/workspace/stwo"
CAIRO_POD="/workspace/stwo-cairo"
POD_PROVER_DIR="${CAIRO_POD}/stwo_cairo_prover"
BIN="target/release/gpu_bench"                      # relative to POD_PROVER_DIR
POD_USER="root"

# PIE inputs on the pod (already present, hash-verified — never synced).
POD_SN_DIR="${CAIRO_POD}/gpu_benchmarks/pie/sn"
POD_GATE_PIE="${GATE_PIE:-${POD_SN_DIR}/SN_PIE_2.zip}"
POD_BOOTLOADER_JSON="${POD_BOOTLOADER_JSON:-/workspace/bench_inputs/simple_bootloader_compiled.json}"

# Pod scratch (outside the repo tree so rsync never touches it).
POD_RUN_DIR="/workspace/bench_loop_runs"
POD_BUILD_LOG="${POD_RUN_DIR}/build.log"
POD_SOUNDNESS_GATE="${POD_RUN_DIR}/cuda-soundness-gate.json"

# Prover knobs.
RUST_MIN_STACK_VAL=4194304
BUILD_RUSTFLAGS="-C target-cpu=native"
BENCH_ENV="${BENCH_ENV:-}"          # debug/bisect env, recorded in every ledger entry
QUALIFICATION_PROBE="${QUALIFICATION_PROBE:-0}"
QUALIFICATION_ARTIFACT="${QUALIFICATION_ARTIFACT:-}"
BENCH_PROOF_HASHES="${BENCH_PROOF_HASHES:-0}"
REUSE_SOUNDNESS_GATE="${REUSE_SOUNDNESS_GATE:-}"
LOCAL_PREFLIGHT_ADMISSION="${LOCAL_PREFLIGHT_ADMISSION:-}"
GPU_PCS_RUNTIME_MODE="${GPU_PCS_RUNTIME_MODE:-arena-graph}"
EXPECTED_POD_GPU="${EXPECTED_POD_GPU:-}"
GPU_NATIVE_ARGS="--engine gpu-native --require-gpu-native-architecture --require-gpu-pcs-runtime-mode ${GPU_PCS_RUNTIME_MODE}"
GPU_TELEMETRY_COLUMNS="timestamp_unix_ns,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,power_draw_w,clock_sm_mhz,clock_memory_mhz,temperature_gpu_c,driver_version,power_limit_w,clock_max_sm_mhz,clock_max_memory_mhz"
GPU_TELEMETRY_SAMPLE_INTERVAL_SECONDS="0.25"

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
REPS="6"
FULL=0
ALL_PIES=0
SIMD=0
SKIP_SYNC=0
GATE_ONLY=0

# Set by resolve_pod().
POD_HOST=""
POD_PORT=""
POD_KEY=""
POD_ID_RESOLVED=""
SSH_OPTS=()
SSH_E=""

# Set by run_bench().
LAST_OUT=""
LAST_RC=""
LAST_STALL_FILE=""
LAST_PROOF_SHA=""
LAST_REMOTE_QUIESCENCE_PASSED=false
LOCAL_SOUNDNESS_GATE_SHA=""
SEALED_BOOT_ID=""
SEALED_GPU_UUID=""
SEALED_GPU_NAME=""
SEALED_GPU_BENCH_PATH=""
SEALED_GPU_BENCH_SHA=""

# ---------------------------------------------------------------------------
# Logging helpers ( logs -> stderr, human summary -> stdout )
# ---------------------------------------------------------------------------
log()  { echo "[bench_loop] $*" >&2; }
dry()  { echo "[DRY_RUN] $*" >&2; }
warn() { echo "[bench_loop][WARN] $*" >&2; }
die()  { echo "[bench_loop][FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,77p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

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
  if [[ "$DRY_RUN" == "1" ]]; then
    POD_ID_RESOLVED="DRY-RUN-POD"
    POD_HOST="dry-run.invalid"; POD_PORT="22"; POD_KEY="/dev/null"
    SSH_OPTS=(-p "$POD_PORT" -i "$POD_KEY")
    SSH_E="ssh -p ${POD_PORT} -i ${POD_KEY}"
    return 0
  fi
  [[ -n "$pod_id" ]] || die "no pod id — set BENCH_POD_ID or POD_ID in ${POD_CONF}"
  POD_ID_RESOLVED="$pod_id"

  if command -v runpodctl >/dev/null 2>&1; then
    local info parsed
    if info="$(runpodctl ssh info "$pod_id" 2>/dev/null)" &&
       parsed="$(RP_INFO="$info" python3 -c '
import hashlib, json, os, sys
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

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    LC_ALL=C shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else LC_ALL=C shasum -a 256; fi
}

expected_sha256() {
  awk -v file="$(basename "$1")" '$2 == file { print $1; exit }' "$INPUT_SHA256SUMS"
}

check_local_input() {
  local path="$1" label="$2" actual expected
  [[ -f "$path" ]] || die "$label missing: $path"
  actual="$(sha256_file "$path")"
  expected="$(expected_sha256 "$path")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || die "$label is not pinned in $INPUT_SHA256SUMS: $(basename "$path")"
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: $path (expected $expected, got $actual)"
  log "$label: $path sha256=$actual${expected:+ (manifest match)}"
}

required_sn_selectors() {
  if [[ "$FULL" == "1" || "$ALL_PIES" == "1" ]]; then
    echo "1 2 3 4"
  elif [[ "$PIE_SEL" != "10t" ]]; then
    echo "$PIE_SEL"
  fi
}

preflight_local_sources() {
  local sel path
  if [[ -n "$SN_PIE_SOURCE_DIR" ]]; then
    [[ -d "$SN_PIE_SOURCE_DIR" ]] || die "SN_PIE_SOURCE_DIR not found: $SN_PIE_SOURCE_DIR"
    for sel in $(required_sn_selectors); do
      path="${SN_PIE_SOURCE_DIR}/SN_PIE_${sel}.zip"
      check_local_input "$path" "local PIE source"
    done
  fi
  if [[ -n "$GATE_PIE_SOURCE" ]]; then
    check_local_input "$GATE_PIE_SOURCE" "local gate PIE source"
  fi
  if [[ -n "$BOOTLOADER_JSON_SOURCE" ]]; then
    check_local_input "$BOOTLOADER_JSON_SOURCE" "local bootloader source"
  fi
}

print_cmd() { printf '%q ' "$@"; }

print_seed_commands() {
  local target="${POD_USER}@${POD_HOST}" sel
  local -a pie_sources=()
  for sel in $(required_sn_selectors); do
    [[ -n "$SN_PIE_SOURCE_DIR" ]] && pie_sources+=("${SN_PIE_SOURCE_DIR}/SN_PIE_${sel}.zip")
  done
  if [[ ${#pie_sources[@]} -eq 0 && -z "$GATE_PIE_SOURCE" && -z "$BOOTLOADER_JSON_SOURCE" ]]; then return; fi

  log "validated local inputs are NOT uploaded automatically. Explicit seed commands:"
  log "  $(print_cmd ssh "${SSH_OPTS[@]}" "$target" "mkdir -p '$POD_SN_DIR' '$(dirname "$POD_GATE_PIE")' '$(dirname "$POD_BOOTLOADER_JSON")'")"
  if [[ ${#pie_sources[@]} -gt 0 ]]; then
    log "  $(print_cmd rsync -av -e "$SSH_E" "${pie_sources[@]}" "${target}:${POD_SN_DIR}/")"
  fi
  [[ -n "$GATE_PIE_SOURCE" ]] && \
    log "  $(print_cmd rsync -av -e "$SSH_E" "$GATE_PIE_SOURCE" "${target}:${POD_GATE_PIE}")"
  if [[ -n "$BOOTLOADER_JSON_SOURCE" ]]; then
    log "  $(print_cmd rsync -av -e "$SSH_E" "$BOOTLOADER_JSON_SOURCE" "${target}:${POD_BOOTLOADER_JSON}")"
  fi
}

preflight_pod_inputs() {
  local sel path expected quoted_path quoted_expected
  local remote_cmd='missing=0; hash_input() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d" " -f1; else LC_ALL=C shasum -a 256 "$1" | cut -d" " -f1; fi; };'
  local -a paths=("$POD_GATE_PIE" "$POD_BOOTLOADER_JSON")
  for sel in $(required_sn_selectors); do paths+=("$(pie_path "$sel")"); done
  for path in "${paths[@]}"; do
    expected="$(expected_sha256 "$path")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
      || die "required pod input is not pinned in $INPUT_SHA256SUMS: $(basename "$path")"
    printf -v quoted_path '%q' "$path"
    printf -v quoted_expected '%q' "$expected"
    remote_cmd+=" path=${quoted_path}; expected=${quoted_expected}; if [ ! -f \"\$path\" ]; then echo \"MISSING required input: \$path\" >&2; missing=1; else actual=\$(hash_input \"\$path\"); echo \"\$actual  \$path\"; if [ \"\$actual\" != \"\$expected\" ]; then echo \"SHA-256 mismatch: \$path (expected \$expected, got \$actual)\" >&2; missing=1; fi; fi;"
  done
  remote_cmd+=' exit $missing'

  log "preflight: required gate/benchmark fixtures and pinned bootloader"
  if ! run_ssh "$remote_cmd"; then
    die "required pod input missing or unpinned; seed it explicitly or update the configured fixture paths, then retry"
  fi
}

soundness_input_specs() {
  local sel path
  path="$POD_GATE_PIE"
  printf 'gate|%s|%s\n' "$path" "$(expected_sha256 "$path")"
  path="$POD_BOOTLOADER_JSON"
  printf 'bootloader|%s|%s\n' "$path" "$(expected_sha256 "$path")"
  for sel in $(required_sn_selectors); do
    path="$(pie_path "$sel")"
    printf 'SN_PIE_%s|%s|%s\n' "$sel" "$path" "$(expected_sha256 "$path")"
  done
}

soundness_input_cli() {
  local label path expected quoted
  while IFS='|' read -r label path expected; do
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
      || die "soundness input is not manifest-pinned: $path"
    printf -v quoted '%q' "${label}=${path}"
    printf ' --input-artifact %s' "$quoted"
  done < <(soundness_input_specs)
}

load_execution_target() {
  local artifact="$1" key value
  SEALED_BOOT_ID=""; SEALED_GPU_UUID=""; SEALED_GPU_NAME=""
  SEALED_GPU_BENCH_PATH=""; SEALED_GPU_BENCH_SHA=""
  while IFS=$'\t' read -r key value; do
    case "$key" in
      pod_id) [[ "$value" == "$POD_ID_RESOLVED" ]] \
        || die "soundness target pod mismatch: expected $POD_ID_RESOLVED, got $value" ;;
      boot_id) SEALED_BOOT_ID="$value" ;;
      gpu_uuid) SEALED_GPU_UUID="$value" ;;
      gpu_name) SEALED_GPU_NAME="$value" ;;
      gpu_bench_path) SEALED_GPU_BENCH_PATH="$value" ;;
      gpu_bench_sha256) SEALED_GPU_BENCH_SHA="$value" ;;
    esac
  done < <(EXPECTED_INPUT_SPECS="$(soundness_input_specs)" python3 - "$artifact" <<'PY'
import hashlib, json, os, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    artifact = json.load(stream)
target = artifact.get("execution_target") or {}
canonical = json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
if artifact.get("schema") != "stwo.cuda.soundness-gate.v3":
    raise SystemExit("invalid soundness schema for remote execution target")
if target.get("schema") != "stwo.remote-execution-target.v1":
    raise SystemExit("invalid remote execution target schema")
if artifact.get("execution_target_sha256") != hashlib.sha256(canonical).hexdigest():
    raise SystemExit("remote execution target hash mismatch")
if artifact.get("execution_target_postcheck") is not True:
    raise SystemExit("remote execution target changed during soundness")
expected_inputs = {}
for line in os.environ["EXPECTED_INPUT_SPECS"].splitlines():
    label, path, digest = line.split("|", 2)
    # dict() avoids a macOS Bash 3.2 brace-expansion bug when this heredoc is
    # nested inside the process substitution consumed by the shell loop.
    expected_inputs[label] = dict(path=path, sha256=digest)
if target.get("inputs") != dict(sorted(expected_inputs.items())):
    raise SystemExit("remote execution target inputs do not match the pinned manifest")
binary = target.get("gpu_bench") or {}
for key, value in (
    ("pod_id", target.get("pod_id")),
    ("boot_id", target.get("boot_id")),
    ("gpu_uuid", target.get("gpu_uuid")),
    ("gpu_name", target.get("gpu_name")),
    ("gpu_bench_path", binary.get("path")),
    ("gpu_bench_sha256", binary.get("sha256")),
):
    if not isinstance(value, str) or not value or "\t" in value or "\n" in value:
        raise SystemExit(f"invalid execution target field: {key}")
    print(f"{key}\t{value}")
PY
  )
  [[ "$SEALED_GPU_BENCH_SHA" =~ ^[0-9a-f]{64}$ ]] \
    || die "soundness target binary hash is invalid"
  [[ "$SEALED_GPU_BENCH_PATH" == "${POD_RUN_DIR}/sealed/gpu_bench.${SEALED_GPU_BENCH_SHA}" ]] \
    || die "soundness target is not the expected content-addressed gpu_bench: $SEALED_GPU_BENCH_PATH"
  [[ "$SEALED_GPU_NAME" == "$POD_GPU" ]] \
    || die "soundness target GPU name mismatch: expected $POD_GPU, got $SEALED_GPU_NAME"
}

sealed_input_sha() {
  python3 - "$LOCAL_SOUNDNESS_GATE" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    inputs = (json.load(stream).get("execution_target") or {}).get("inputs") or {}
matches = {value.get("sha256") for value in inputs.values()
           if isinstance(value, dict) and value.get("path") == sys.argv[2]}
if len(matches) != 1:
    raise SystemExit(f"sealed input path is absent or ambiguous: {sys.argv[2]}")
value = matches.pop()
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit(f"sealed input hash is invalid: {sys.argv[2]}")
print(value)
PY
}

sealed_input_paths() {
  python3 - "$LOCAL_SOUNDNESS_GATE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    inputs = (json.load(stream).get("execution_target") or {}).get("inputs") or {}
for path in sorted({value.get("path") for value in inputs.values()
                    if isinstance(value, dict) and isinstance(value.get("path"), str)}):
    print(path)
PY
}

remote_execution_guard() {
  local paths_csv="$1" script quoted expected path
  printf -v quoted '%q' "$SEALED_BOOT_ID"
  script="actual=\$(cat /proc/sys/kernel/random/boot_id 2>/dev/null); [[ \"\$actual\" == ${quoted} ]] || { echo 'remote seal: pod boot id changed' >&2; exit 96; };"
  printf -v quoted '%q' "$SEALED_GPU_UUID"
  script+=" actual=\$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 | tr -d '\\r'); [[ \"\$actual\" == ${quoted} ]] || { echo 'remote seal: GPU UUID changed' >&2; exit 96; };"
  printf -v quoted '%q' "$SEALED_GPU_NAME"
  script+=" actual=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\\r'); [[ \"\$actual\" == ${quoted} ]] || { echo 'remote seal: GPU name changed' >&2; exit 96; };"
  script+=" hash_sealed() { if command -v sha256sum >/dev/null 2>&1; then sha256sum \"\$1\" | cut -d' ' -f1; else LC_ALL=C shasum -a 256 \"\$1\" | cut -d' ' -f1; fi; };"
  printf -v quoted '%q' "$SEALED_GPU_BENCH_PATH"
  printf -v expected '%q' "$SEALED_GPU_BENCH_SHA"
  script+=" exec 9<${quoted} || { echo 'remote seal: gpu_bench missing' >&2; exit 96; }; actual=\$(hash_sealed /proc/self/fd/9); [[ \"\$actual\" == ${expected} ]] || { echo 'remote seal: gpu_bench changed' >&2; exit 96; };"
  local old_ifs="$IFS"
  IFS=','
  for path in $paths_csv; do
    [[ -n "$path" ]] || continue
    expected="$(sealed_input_sha "$path")" || die "input is absent from the remote seal: $path"
    printf -v quoted '%q' "$path"
    printf -v expected '%q' "$expected"
    script+=" actual=\$(hash_sealed ${quoted}); [[ \"\$actual\" == ${expected} ]] || { echo 'remote seal: input changed: ${quoted}' >&2; exit 96; };"
  done
  IFS="$old_ifs"
  printf '%s\n' "$script"
}

remote_quiescence_guard() {
  local quoted_uuid quoted_binary
  printf -v quoted_uuid '%q' "$SEALED_GPU_UUID"
  printf -v quoted_binary '%q' "$SEALED_GPU_BENCH_PATH"
  cat <<EOF
sealed_inode=\$(stat -Lc '%d:%i' ${quoted_binary} 2>/dev/null) || { echo 'remote isolation: cannot stat sealed gpu_bench' >&2; exit 97; }
gpu_pids=\$(nvidia-smi --id=${quoted_uuid} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || { echo 'remote isolation: compute-process query failed' >&2; exit 97; }
[[ -z "\$(printf '%s' "\$gpu_pids" | tr -d '[:space:]')" ]] || { echo "remote isolation: GPU compute PIDs present: \$gpu_pids" >&2; exit 97; }
offenders=''
for proc in /proc/[0-9]*; do
  [[ -r "\$proc/comm" ]] || continue
  stat_line=\$(cat "\$proc/stat" 2>/dev/null) || continue
  stat_rest=\${stat_line##*)}; set -- \$stat_rest
  [[ "\${1:-}" != Z ]] || continue
  comm=\$(cat "\$proc/comm" 2>/dev/null) || continue
  exe=\$(readlink "\$proc/exe" 2>/dev/null || true)
  exe_name=\${exe##*/}
  proc_inode=\$(stat -Lc '%d:%i' "\$proc/exe" 2>/dev/null || true)
  case "\$comm" in cargo|rustc|nvcc|ptxas|gpu_bench|gpu_bench.*) offenders="\$offenders \${proc##*/}:\$comm" ;; esac
  case "\$exe_name" in cargo|rustc|nvcc|ptxas|gpu_bench|gpu_bench.*) offenders="\$offenders \${proc##*/}:\$exe_name" ;; esac
  [[ -z "\$proc_inode" || "\$proc_inode" != "\$sealed_inode" ]] || offenders="\$offenders \${proc##*/}:sealed-gpu_bench"
done
[[ -z "\$offenders" ]] || { echo "remote isolation: worker PIDs present:\$offenders" >&2; exit 97; }
EOF
}

remote_process_group_gone() {
  local pgid_file="$1" quoted
  printf -v quoted '%q' "$pgid_file"
  run_ssh "pgid=\$(cat ${quoted} 2>/dev/null) || exit 1; [[ \"\$pgid\" =~ ^[0-9]+\$ ]] && (( pgid > 1 )) || exit 1; \
    for attempt in \$(seq 1 30); do members=''; \
      for proc in /proc/[0-9]*; do stat=\$(cat \"\$proc/stat\" 2>/dev/null) || continue; \
        rest=\${stat##*)}; set -- \$rest; [[ \"\${1:-}\" != Z && \"\${3:-}\" == \"\$pgid\" ]] && members=\"\$members \${proc##*/}\"; done; \
      [[ -z \"\$members\" ]] && exit 0; sleep 0.1; done; \
    echo \"remote isolation: process group \$pgid still has PIDs:\$members\" >&2; exit 1"
}

remote_detached_startup_ok() {
  local pgid_file="$1" rc_file="$2" quoted_pgid quoted_rc
  printf -v quoted_pgid '%q' "$pgid_file"
  printf -v quoted_rc '%q' "$rc_file"
  run_ssh "for ((attempt=0; attempt<50; attempt++)); do \
      [[ -f ${quoted_rc} ]] && exit 0; \
      pgid=\$(cat ${quoted_pgid} 2>/dev/null || true); \
      if [[ \"\$pgid\" =~ ^[0-9]+\$ ]] && (( pgid > 1 )); then \
        stat_line=\$(cat \"/proc/\$pgid/stat\" 2>/dev/null || true); \
        if [[ -n \"\$stat_line\" ]]; then rest=\${stat_line##*)}; set -- \$rest; \
          [[ \"\${1:-}\" != Z && \"\${3:-}\" == \"\$pgid\" && \"\${4:-}\" == \"\$pgid\" \
             && \"\$(stat -c %u \"/proc/\$pgid\" 2>/dev/null)\" == \"\$(id -u)\" ]] && exit 0; \
        fi; \
      fi; \
      sleep 0.1; \
    done; exit 1"
}

kill_remote_process_group() {
  local pgid_file="$1" quoted
  printf -v quoted '%q' "$pgid_file"
  run_ssh "pgid=\$(cat ${quoted} 2>/dev/null) || exit 1; [[ \"\$pgid\" =~ ^[0-9]+\$ ]] && (( pgid > 1 )) || exit 1; \
    stat=\$(cat \"/proc/\$pgid/stat\" 2>/dev/null) || exit 1; rest=\${stat##*)}; set -- \$rest; \
    [[ \"\${1:-}\" != Z && \"\${3:-}\" == \"\$pgid\" && \"\${4:-}\" == \"\$pgid\" ]] || exit 1; \
    [[ \"\$(stat -c %u \"/proc/\$pgid\" 2>/dev/null)\" == \"\$(id -u)\" ]] || exit 1; \
    kill -TERM -- -\"\$pgid\" 2>/dev/null || true; sleep 3; kill -KILL -- -\"\$pgid\" 2>/dev/null || true"
  remote_process_group_gone "$pgid_file"
}

validate_remote_execution_target() {
  local all_paths seal_guard
  load_execution_target "$LOCAL_SOUNDNESS_GATE"
  if [[ "$DRY_RUN" == "1" ]]; then
    [[ -z "${FAKE_REMOTE_TARGET_MISMATCH:-}" ]] \
      || die "remote execution target mismatch (synthetic ${FAKE_REMOTE_TARGET_MISMATCH})"
    return 0
  fi
  all_paths="$(sealed_input_paths | paste -sd, -)"
  seal_guard="$(remote_execution_guard "$all_paths")" \
    || die "could not construct the sealed remote-execution guard"
  run_ssh "$seal_guard" \
    || die "remote pod/GPU/binary/input target no longer matches the counted soundness gate"
}

verify_remote_source_projection() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local stwo_changes cairo_changes
  stwo_changes="$(run_rsync -azcnO --delete --itemize-changes --no-owner --no-group --no-perms \
    --exclude=target --exclude=.git -e "$SSH_E" \
    "${STWO_LOCAL}/" "${POD_USER}@${POD_HOST}:${STWO_POD}/")" || return 1
  cairo_changes="$(run_rsync -azcnO --delete --itemize-changes --no-owner --no-group --no-perms \
    --exclude=target --exclude=.git \
    --exclude='gpu_benchmarks/pie/sn/' \
    --exclude='gpu_benchmarks/pie/*.zip' \
    --exclude='gpu_benchmarks/loop/results' \
    --exclude='gpu_benchmarks/loop/ledger.jsonl' \
    -e "$SSH_E" "${CAIRO_LOCAL}/" "${POD_USER}@${POD_HOST}:${CAIRO_POD}/")" || return 1
  [[ -z "$stwo_changes" && -z "$cairo_changes" ]] || {
    warn "remote source projection changed after sync/soundness"
    printf '%s\n%s\n' "$stwo_changes" "$cairo_changes" >&2
    return 1
  }
}

seal_source_projection() {
  local artifact="$1"
  SP_STWO_HEAD="$STWO_REV" SP_STWO_HASH="$STWO_WORKTREE_HASH" \
  SP_CAIRO_HEAD="$CAIRO_REV" SP_CAIRO_HASH="$CAIRO_WORKTREE_HASH" \
  python3 - "$artifact" <<'PY'
import json, os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    artifact = json.load(stream)
artifact["source_projection"] = {
    "method": "rsync-archive-checksum-dry-run-clean",
    "verified_after_soundness": True,
    "source": {
        "stwo": {"head": os.environ["SP_STWO_HEAD"], "worktree_hash": os.environ["SP_STWO_HASH"]},
        "stwo_cairo": {"head": os.environ["SP_CAIRO_HEAD"], "worktree_hash": os.environ["SP_CAIRO_HASH"]},
    },
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(artifact, stream, sort_keys=True)
    stream.write("\n")
PY
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pie)       PIE_SEL="${2:?--pie needs a value}"; shift 2 ;;
    --reps)      REPS="${2:?--reps needs a value}"; shift 2 ;;
    --full)      FULL=1; shift ;;
    --all-pies)  ALL_PIES=1; shift ;;
    --simd)      SIMD=1; shift ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --gate-only) GATE_ONLY=1; shift ;;
    -h|--help)   usage 0 ;;
    *)           die "unknown flag: $1 (try --help)" ;;
  esac
done

[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -ge 2 ]] || die "--reps must be at least 2 so proof-byte equality can be checked"
pie_path "$PIE_SEL" >/dev/null   # validates selector early
if [[ "$SKIP_SYNC" == "1" ]]; then
  [[ "$QUALIFICATION_PROBE" == "1" && -n "$REUSE_SOUNDNESS_GATE" ]] \
    || die "--skip-sync is restricted to the source-bound internal qualification continuation"
fi

git -C "$CAIRO_LOCAL" rev-parse --git-dir >/dev/null 2>&1 \
  || die "missing stwo-cairo checkout: $CAIRO_LOCAL"
git -C "$STWO_LOCAL" rev-parse --git-dir >/dev/null 2>&1 \
  || die "missing sibling stwo checkout: $STWO_LOCAL"
[[ -f "$INPUT_SHA256SUMS" ]] || die "input checksum manifest missing: $INPUT_SHA256SUMS"
[[ -f "$ARCHITECTURE_CHECK" ]] || die "architecture record validator missing: $ARCHITECTURE_CHECK"
[[ -f "$SOUNDNESS_RUNNER" ]] || die "CUDA soundness runner missing: $SOUNDNESS_RUNNER"
[[ -n "$(expected_sha256 "$POD_BOOTLOADER_JSON")" ]] \
  || die "pinned bootloader checksum missing from $INPUT_SHA256SUMS: $(basename "$POD_BOOTLOADER_JSON")"
STWO_LOCAL="$(cd "$STWO_LOCAL" && pwd)"

case "$GPU_PCS_RUNTIME_MODE" in
  detached-eager|arena-graph) ;;
  *) die "GPU_PCS_RUNTIME_MODE must be detached-eager or arena-graph (got '$GPU_PCS_RUNTIME_MODE')" ;;
esac

# Validate BENCH_ENV before interpolating it into a remote shell.
if [[ -n "$BENCH_ENV" ]]; then
  for kv in $BENCH_ENV; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./,:+-]*$ ]] \
      || die "BENCH_ENV token '$kv' is not a shell-safe K=V token"
    [[ "${kv%%=*}" != "STWO_BOOTLOADER_JSON" ]] \
      || die "STWO_BOOTLOADER_JSON is reserved; set POD_BOOTLOADER_JSON instead"
  done
fi

[[ -n "$LOCAL_PREFLIGHT_ADMISSION" && -f "$LOCAL_PREFLIGHT_ADMISSION" ]] \
  || die "LOCAL_PREFLIGHT_ADMISSION from qualification_round is required before pod work"
if ! LP_ENV="$BENCH_ENV" LP_RUNTIME="$GPU_PCS_RUNTIME_MODE" LP_DRY_RUN="$DRY_RUN" \
  python3 - "$LOCAL_PREFLIGHT_ADMISSION" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    admission = json.load(stream)
profiles = admission.get("profiles") or {}
if (admission.get("schema") != "stwo.local-preflight-admission.v1"
        or admission.get("passed") is not True
        or admission.get("runtime_mode") != os.environ["LP_RUNTIME"]
        or admission.get("dry_run") is not (os.environ["LP_DRY_RUN"] == "1")
        or os.environ["LP_ENV"] not in set(profiles.values())):
    raise SystemExit(1)
PY
then
  die "local preflight admission does not authorize this profile/runtime"
fi

preflight_local_sources

# ---------------------------------------------------------------------------
# (a) Provenance capture
# ---------------------------------------------------------------------------
git_rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }
# sha256 of tracked changes plus untracked paths/content. Newly generated CUDA
# files are part of the identity even before they are staged.
git_worktree_hash() {
  local repo="$1"
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then echo "NOGIT"; return; fi
  (
    git -C "$repo" diff --binary HEAD -- . ':(exclude)gpu_benchmarks/loop/results' 2>/dev/null || exit 1
    git -C "$repo" ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' path; do
        [[ "$path" == gpu_benchmarks/loop/results/* ]] && continue
        if [[ -L "$repo/$path" ]]; then
          link_hash="$(readlink -n "$repo/$path" | sha256_stream | cut -d' ' -f1)" || exit 1
          printf 'untracked-symlink\0%s\0%s\0' "$path" "$link_hash"
        elif [[ -f "$repo/$path" ]]; then
          file_kind=regular
          [[ -x "$repo/$path" ]] && file_kind=executable
          file_hash="$(sha256_file "$repo/$path")" || exit 1
          printf 'untracked-%s\0%s\0%s\0' "$file_kind" "$path" "$file_hash"
        else
          echo "unsupported untracked source path: $repo/$path" >&2
          exit 1
        fi
      done
  ) | sha256_stream | cut -d' ' -f1
}

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STWO_REV="$(git_rev "$STWO_LOCAL")"
STWO_WORKTREE_HASH="$(git_worktree_hash "$STWO_LOCAL")"
STWO_DIRTY="$STWO_WORKTREE_HASH"
[[ -n "$(git -C "$STWO_LOCAL" status --porcelain)" ]] || STWO_DIRTY="clean"
CAIRO_REV="$(git_rev "$CAIRO_LOCAL")"
CAIRO_WORKTREE_HASH="$(git_worktree_hash "$CAIRO_LOCAL")"
CAIRO_DIRTY="$CAIRO_WORKTREE_HASH"
[[ -n "$(git -C "$CAIRO_LOCAL" status --porcelain -- . ':(exclude)gpu_benchmarks/loop/results')" ]] \
  || CAIRO_DIRTY="clean"

python3 "$ARCHITECTURE_CHECK" \
  --local-admission "$LOCAL_PREFLIGHT_ADMISSION" \
  --gpu-bench-binary "${CAIRO_LOCAL}/stwo_cairo_prover/target/debug/gpu_bench" \
  --raw-input-manifest "$INPUT_SHA256SUMS" \
  --bootloader "$BOOTLOADER_JSON_SOURCE" \
  --pinned-adapted-manifest "$PINNED_ADAPTED_SHA256SUMS" \
  --runtime-mode "$GPU_PCS_RUNTIME_MODE" --expected-dry-run "$DRY_RUN" \
  --stwo-head "$STWO_REV" --stwo-worktree-hash "$STWO_WORKTREE_HASH" \
  --stwo-cairo-head "$CAIRO_REV" --stwo-cairo-worktree-hash "$CAIRO_WORKTREE_HASH" \
  || die "full local input/capacity admission is stale or incomplete"

validate_counted_soundness_gate() {
  local artifact="$1"
  python3 "$ARCHITECTURE_CHECK" --soundness-only --soundness-gate "$artifact" \
    --runtime-mode "$GPU_PCS_RUNTIME_MODE" --expected-dry-run "$DRY_RUN" \
    --stwo-head "$STWO_REV" --stwo-worktree-hash "$STWO_WORKTREE_HASH" \
    --stwo-cairo-head "$CAIRO_REV" --stwo-cairo-worktree-hash "$CAIRO_WORKTREE_HASH"
}

validate_reused_soundness_gate() {
  validate_counted_soundness_gate "$REUSE_SOUNDNESS_GATE"
}

if [[ -n "$REUSE_SOUNDNESS_GATE" ]]; then
  [[ "$QUALIFICATION_PROBE" == "1" && -f "$REUSE_SOUNDNESS_GATE" ]] \
    || die "REUSE_SOUNDNESS_GATE is restricted to qualification-internal probes"
  validate_reused_soundness_gate \
    || die "reused soundness artifact is not bound to the current source/runtime"
fi

if [[ "$GATE_ONLY" == "0" && "$QUALIFICATION_PROBE" != "1" ]]; then
  [[ -n "$QUALIFICATION_ARTIFACT" && -f "$QUALIFICATION_ARTIFACT" ]] \
    || die "publishable performance requires QUALIFICATION_ARTIFACT from a passed qualification_round"
  QA_STWO_REV="$STWO_REV" QA_STWO_HASH="$STWO_WORKTREE_HASH" \
  QA_CAIRO_REV="$CAIRO_REV" QA_CAIRO_HASH="$CAIRO_WORKTREE_HASH" \
  QA_ENV="$BENCH_ENV" QA_RUNTIME="$GPU_PCS_RUNTIME_MODE" \
  QA_PIE="$PIE_SEL" QA_ALL_PIES="$ALL_PIES" QA_FULL="$FULL" QA_SIMD="$SIMD" \
  python3 - "$QUALIFICATION_ARTIFACT" <<'PY' || die "qualification artifact does not match this source/env/runtime"
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    artifact = json.load(stream)
if artifact.get("schema") != "stwo.qualification-round.v4":
    raise SystemExit(f"qualification schema: expected v4, got {artifact.get('schema')!r}")
profiles = artifact.get("profiles") or {}
if set(profiles) != {"universal_sn1_sn4", "sn2_headline"}:
    raise SystemExit("qualification profiles are incomplete or unexpected")
allowed_envs = {profile.get("env") for profile in profiles.values() if isinstance(profile, dict)}
if os.environ["QA_ENV"] not in allowed_envs:
    raise SystemExit("BENCH_ENV is not one of the two exactly qualified profiles")
profile_name = next(
    name for name, profile in profiles.items() if profile.get("env") == os.environ["QA_ENV"]
)
selector = {
    "pie": os.environ["QA_PIE"],
    "all_pies": os.environ["QA_ALL_PIES"] == "1",
    "full": os.environ["QA_FULL"] == "1",
    "simd": os.environ["QA_SIMD"] == "1",
}
if selector["full"] or selector["simd"] or selector["pie"] not in {"1", "2", "3", "4"}:
    raise SystemExit("qualification covers only GPU-native fixed SN PIE selectors")
if profile_name == "sn2_headline" and (selector["pie"] != "2" or selector["all_pies"]):
    raise SystemExit("SN2 headline profile is qualified only for the single SN_PIE_2 selector")
required = {
    ("status",): "passed",
    ("performance_admissible",): True,
    ("runtime_mode",): os.environ["QA_RUNTIME"],
    ("source", "stwo", "head"): os.environ["QA_STWO_REV"],
    ("source", "stwo", "worktree_hash"): os.environ["QA_STWO_HASH"],
    ("source", "stwo_cairo", "head"): os.environ["QA_CAIRO_REV"],
    ("source", "stwo_cairo", "worktree_hash"): os.environ["QA_CAIRO_HASH"],
    ("remote_execution_target", "remote_quiescence_passed"): True,
}
for path, expected in required.items():
    value = artifact
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    mismatch = value is not expected if isinstance(expected, bool) else value != expected
    if mismatch:
        raise SystemExit(f"qualification {'.'.join(path)}: expected {expected!r}, got {value!r}")
PY
fi

# Do not resolve or contact a paid pod until every local source, admission, and
# qualification binding has passed.
resolve_pod
print_seed_commands
preflight_pod_inputs

mkdir -p "$RESULTS_DIR"

log "provenance: stwo=${STWO_REV:0:12} dirty=${STWO_DIRTY} | cairo=${CAIRO_REV:0:12} dirty=${CAIRO_DIRTY}"
[[ -n "$BENCH_ENV" ]] && log "bench_env: ${BENCH_ENV} (recorded in every ledger entry)"
log "bootloader: ${POD_BOOTLOADER_JSON} (pinned, exported for build and runs)"
log "GPU-native architecture gate: required mode=${GPU_PCS_RUNTIME_MODE}"

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
[[ -z "$EXPECTED_POD_GPU" || "$POD_GPU" == "$EXPECTED_POD_GPU" ]] \
  || die "qualification requires GPU '${EXPECTED_POD_GPU}', got '${POD_GPU}'"

# Reset containers do not retain apt packages or $HOME, while the Rust/Cargo
# homes and build caches live on /workspace. Recreate that small bridge before
# rsync so release qualification is self-contained after every pod resume.
bootstrap_pod() {
  log "bootstrap reset pod container (rsync + persistent Rust homes)"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "bootstrap: install rsync if absent; bind /workspace Rust/Cargo homes; validate CUDA, NCU, and launcher tools"
    return 0
  fi
  run_ssh "set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    command -v rsync >/dev/null 2>&1 || { apt-get update -qq >/dev/null && apt-get install -y -qq rsync >/dev/null; }
    command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 && \
      command -v make >/dev/null 2>&1 && command -v ar >/dev/null 2>&1 && \
      command -v ld >/dev/null 2>&1 || apt-get install -y -qq build-essential >/dev/null
    mkdir -p /workspace/.rustup-persist /workspace/.cargo-persist \"\$HOME/.cargo\"
    if [ ! -x /workspace/.cargo-persist/bin/rustup ]; then
      command -v curl >/dev/null 2>&1
      RUSTUP_HOME=/workspace/.rustup-persist CARGO_HOME=/workspace/.cargo-persist \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        RUSTUP_HOME=/workspace/.rustup-persist CARGO_HOME=/workspace/.cargo-persist \
        sh -s -- -y --default-toolchain none >/dev/null
    fi
    printf '%s\\n' \
      'export RUSTUP_HOME=/workspace/.rustup-persist' \
      'export CARGO_HOME=/workspace/.cargo-persist' \
      'export PATH=\"\$CARGO_HOME/bin:/usr/local/cuda/bin:\$PATH\"' \
      > \"\$HOME/.cargo/env\"
    . \"\$HOME/.cargo/env\"
    for tool in rsync cargo rustc rustup nvcc ncu python3 sha256sum stat setsid nohup timeout nvidia-smi gcc g++ make ar ld; do
      command -v \"\$tool\" >/dev/null 2>&1 || { echo \"missing remote prerequisite: \$tool\" >&2; exit 1; }
    done" || die "pod bootstrap failed before source sync"
}

# ---------------------------------------------------------------------------
# (b) Sync repositories
# ---------------------------------------------------------------------------
sync_repos() {
  log "rsync stwo -> pod (excludes target/.git; --delete: stale kernels break the auto-collecting build)"
  run_rsync -azc --delete --partial --no-owner --no-group --no-perms \
    --exclude=target --exclude=.git \
    -e "$SSH_E" \
    "${STWO_LOCAL}/" "${POD_USER}@${POD_HOST}:${STWO_POD}/"

  log "rsync stwo-cairo -> pod (excludes target/.git/PIE zips/ledger)"
  run_rsync -azc --delete --partial --no-owner --no-group --no-perms \
    --exclude=target --exclude=.git \
    --exclude='gpu_benchmarks/pie/sn/' \
    --exclude='gpu_benchmarks/pie/*.zip' \
    --exclude='gpu_benchmarks/loop/results' \
    --exclude='gpu_benchmarks/loop/ledger.jsonl' \
    -e "$SSH_E" \
    "${CAIRO_LOCAL}/" "${POD_USER}@${POD_HOST}:${CAIRO_POD}/"

}

# ---------------------------------------------------------------------------
# (c) Incremental build (abort loudly, with the log tail)
# ---------------------------------------------------------------------------
build_pod() {
  log "incremental build on pod (gpu_bench, --features pie-bench)"
  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "build: STWO_CUDA_BUILD_JOBS=16 STWO_BOOTLOADER_JSON=${POD_BOOTLOADER_JSON} cargo build --release -p stwo-cairo-gpu-prover --bin gpu_bench --features pie-bench"
    return 0
  fi
  local out
  out="$(run_ssh "cd '${POD_PROVER_DIR}' && . \$HOME/.cargo/env 2>/dev/null; \
      for stwo_name in \$(env | sed -n 's/^\(STWO_[A-Za-z0-9_]*\)=.*/\1/p'); do unset \"\$stwo_name\"; done; \
      unset CUDA_LAUNCH_BLOCKING CUDA_DEVICE_MAX_CONNECTIONS; \
      : > '${POD_BUILD_LOG}'; \
      rustup toolchain install >> '${POD_BUILD_LOG}' 2>&1 && \
      PATH=/usr/local/cuda/bin:\$PATH RUSTFLAGS='${BUILD_RUSTFLAGS}' \
      STWO_CUDA_OBJ_CACHE=/workspace/.cuda_obj_cache \
      STWO_CUDA_BUILD_JOBS=16 \
      STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}' \
      cargo build --release -p stwo-cairo-gpu-prover --bin gpu_bench --features pie-bench \
      >> '${POD_BUILD_LOG}' 2>&1; echo BUILD_EXIT=\$?")"
  local code
  code="$(printf '%s\n' "$out" | sed -n 's/.*BUILD_EXIT=\([0-9][0-9]*\).*/\1/p' | tail -1)"
  if [[ "$code" != "0" ]]; then
    warn "BUILD FAILED (exit=${code:-?}). Tail of ${POD_BUILD_LOG}:"
    run_ssh "tail -n 40 '${POD_BUILD_LOG}'" >&2 || true
    die "compile error — aborting before any benchmark. No ledger entry written."
  fi
  log "build OK"
}

seal_gpu_bench() {
  if [[ "$DRY_RUN" == "1" ]]; then
    SEALED_GPU_BENCH_SHA="$(printf 'b%.0s' {1..64})"
    SEALED_GPU_BENCH_PATH="${POD_RUN_DIR}/sealed/gpu_bench.${SEALED_GPU_BENCH_SHA}"
    dry "seal gpu_bench -> ${SEALED_GPU_BENCH_PATH}"
    return 0
  fi
  local source_path="${POD_PROVER_DIR}/${BIN}" digest sealed_dir sealed_path
  digest="$(run_ssh "sha256sum '${source_path}'" | cut -d' ' -f1)"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "could not hash built gpu_bench"
  sealed_dir="${POD_RUN_DIR}/sealed"
  sealed_path="${sealed_dir}/gpu_bench.${digest}"
  run_ssh "set -e; mkdir -p '${sealed_dir}'; \
    if [ ! -f '${sealed_path}' ]; then \
      install -m 0555 '${source_path}' '${sealed_path}.tmp'; \
      mv -f '${sealed_path}.tmp' '${sealed_path}'; \
    fi; \
    actual=\$(sha256sum '${sealed_path}' | cut -d' ' -f1); \
    [ \"\$actual\" = '${digest}' ]; chmod 0555 '${sealed_path}'"
  SEALED_GPU_BENCH_PATH="$sealed_path"
  SEALED_GPU_BENCH_SHA="$digest"
  log "sealed gpu_bench: ${digest}"
}

# Native cfg-gated tests can exit zero after executing nothing. Run the counted
# differential suite on the pod and retain its JSON artifact before any proof or
# performance claim from this build.
run_cuda_soundness_gate() {
  if [[ -n "$REUSE_SOUNDNESS_GATE" ]]; then
    [[ "$QUALIFICATION_PROBE" == "1" && -f "$REUSE_SOUNDNESS_GATE" ]] \
      || die "REUSE_SOUNDNESS_GATE is restricted to qualification-internal probes"
    validate_reused_soundness_gate \
      || die "reused soundness artifact is not bound to the current source/runtime"
    LOCAL_SOUNDNESS_GATE="$REUSE_SOUNDNESS_GATE"
    validate_remote_execution_target
    LOCAL_SOUNDNESS_GATE_SHA="$(sha256_file "$LOCAL_SOUNDNESS_GATE")"
    log "CUDA soundness gate: reusing source/env/remote-target-bound qualification artifact ${REUSE_SOUNDNESS_GATE}"
    return 0
  fi
  LOCAL_SOUNDNESS_GATE="${RESULTS_DIR}/${STAMP}.cuda-soundness-gate.json"
  log "CUDA soundness gate: counted native differential targets"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "python3 gpu_benchmarks/run_cuda_soundness_gate.py --stwo ${STWO_POD} --runtime-mode ${GPU_PCS_RUNTIME_MODE} --pod-id ${POD_ID_RESOLVED} --gpu-bench ${SEALED_GPU_BENCH_PATH}$(soundness_input_cli) --output ${POD_SOUNDNESS_GATE}"
    PYTHONPATH="$(dirname "$ARCHITECTURE_CHECK")" \
      STWO_HEAD="$STWO_REV" STWO_HASH="$STWO_WORKTREE_HASH" \
      STWO_CAIRO_HEAD="$CAIRO_REV" STWO_CAIRO_HASH="$CAIRO_WORKTREE_HASH" \
      RUNTIME_MODE="$GPU_PCS_RUNTIME_MODE" POD_ID_VALUE="$POD_ID_RESOLVED" \
      GPU_NAME_VALUE="$POD_GPU" GPU_BENCH_PATH="$SEALED_GPU_BENCH_PATH" \
      GPU_BENCH_SHA="$SEALED_GPU_BENCH_SHA" \
      TARGET_INPUT_SPECS="$(soundness_input_specs)" \
      python3 - "$LOCAL_SOUNDNESS_GATE" <<'PY'
import hashlib, json, os, sys
from run_cuda_soundness_gate import QUALIFICATION_FLAGS, gates_for_runtime_mode

runtime_mode = os.environ["RUNTIME_MODE"]
gates = gates_for_runtime_mode(runtime_mode)
inputs = {}
for line in os.environ["TARGET_INPUT_SPECS"].splitlines():
    label, path, digest = line.split("|", 2)
    inputs[label] = {"path": path, "sha256": digest}

execution_target = {
    "schema": "stwo.remote-execution-target.v1",
    "pod_id": os.environ["POD_ID_VALUE"],
    "boot_id": "00000000-0000-4000-8000-000000000001",
    "gpu_uuid": "GPU-00000000-0000-4000-8000-000000000001",
    "gpu_name": os.environ["GPU_NAME_VALUE"],
    "gpu_bench": {"path": os.environ["GPU_BENCH_PATH"], "sha256": os.environ["GPU_BENCH_SHA"]},
    "inputs": dict(sorted(inputs.items())),
}
target_sha = hashlib.sha256(json.dumps(
    execution_target, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()

artifact = {
    "schema": "stwo.cuda.soundness-gate.v3",
    "dry_run": True,
    "stwo_git_head": ("f" * 40 if os.environ.get("FAKE_SOUNDNESS_HEAD_MISMATCH") == "1" else os.environ["STWO_HEAD"]),
    "stwo_git_dirty": False,
    "stwo_cairo_git_head": os.environ["STWO_CAIRO_HEAD"],
    "stwo_cairo_git_dirty": False,
    "stwo_worktree_hash": os.environ["STWO_HASH"],
    "stwo_cairo_worktree_hash": os.environ["STWO_CAIRO_HASH"],
    "synced_source": {
        "stwo": {
            "head": ("f" * 40 if os.environ.get("FAKE_SOUNDNESS_HEAD_MISMATCH") == "1" else os.environ["STWO_HEAD"]),
            "worktree_hash": os.environ["STWO_HASH"],
        },
        "stwo_cairo": {
            "head": os.environ["STWO_CAIRO_HEAD"],
            "worktree_hash": os.environ["STWO_CAIRO_HASH"],
        },
        "transport": "rsync-archive-checksum",
    },
    "runtime_mode": runtime_mode,
    "execution_target": execution_target,
    "execution_target_sha256": target_sha,
    "execution_target_postcheck": True,
    "source_projection": {
        "method": "rsync-archive-checksum-dry-run-clean",
        "verified_after_soundness": True,
        "source": {
            "stwo": {"head": os.environ["STWO_HEAD"], "worktree_hash": os.environ["STWO_HASH"]},
            "stwo_cairo": {"head": os.environ["STWO_CAIRO_HEAD"], "worktree_hash": os.environ["STWO_CAIRO_HASH"]},
        },
    },
    "qualification_flags": {
        key: int(dict(token.split("=", 1) for token in os.environ.get("BENCH_ENV", "").split()).get(key) == "1")
        for key in QUALIFICATION_FLAGS
    },
    "effective_stwo_env": {
        "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
        "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
        "STWO_PARITY_REF_STWO_HEAD": os.environ["STWO_HEAD"],
        "STWO_PARITY_REF_STWO_WORKTREE_HASH": os.environ["STWO_HASH"],
        "STWO_PARITY_REF_STWO_CAIRO_HEAD": os.environ["STWO_CAIRO_HEAD"],
        "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": os.environ["STWO_CAIRO_HASH"],
        **{
            key: value
            for key, value in dict(
                token.split("=", 1) for token in os.environ.get("BENCH_ENV", "").split()
            ).items()
            if key.startswith("STWO_")
        },
    },
    "passed": True,
    "gates": [
        {"name": name, "command": list(command), "exit_code": 0,
         "executed_tests": required,
         "required_tests": required, "passed": True}
        for name, command, required in gates
    ],
}
for gate in artifact["gates"]:
    if gate["name"] == "strict_resident_whole_proof_simd_byte_identity":
        from run_cuda_soundness_gate import STRICT_RESIDENT_REQUIRED_TESTS
        gate["required_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
        gate["executed_test_names"] = list(STRICT_RESIDENT_REQUIRED_TESTS)
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(artifact, stream)
    stream.write("\n")
PY
    validate_counted_soundness_gate "$LOCAL_SOUNDNESS_GATE" \
      || die "synthetic counted soundness artifact failed its full contract"
    validate_remote_execution_target
    LOCAL_SOUNDNESS_GATE_SHA="$(sha256_file "$LOCAL_SOUNDNESS_GATE")"
    return 0
  fi
  # Detached launcher + rc sentinel, same pattern as the benchmark step below:
  # the gate builds and runs many cargo test targets for tens of minutes, and a
  # synchronous ssh channel dying mid-run kills the remote gate with it
  # (observed 2026-07-10: a transient drop aborted the arena gate and a stale
  # artifact was fetched). The pod-side artifact is removed up front so a
  # stale file can never be mistaken for this run's result.
  local gate_sh="${POD_RUN_DIR}/soundness_gate.sh"
  local gate_rc="${POD_RUN_DIR}/soundness_gate.rc"
  local gate_log="${POD_RUN_DIR}/soundness_gate.log"
  local gate_pgid="${POD_RUN_DIR}/soundness_gate.pgid"
  local input_cli
  input_cli="$(soundness_input_cli)"
  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  run_ssh "cat > '${gate_sh}'" <<EOF
#!/usr/bin/env bash
echo \$\$ > '${gate_pgid}'
cd '${CAIRO_POD}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
while IFS='=' read -r stwo_name _; do
  case "\$stwo_name" in STWO_*) unset "\$stwo_name" ;; esac
done < <(env)
unset CUDA_LAUNCH_BLOCKING CUDA_DEVICE_MAX_CONNECTIONS
export STWO_CUDA_OBJ_CACHE=/workspace/.cuda_obj_cache
export STWO_PARITY_REF_CACHE=/workspace/.parity_ref_cache
${BENCH_ENV:+export ${BENCH_ENV}}
python3 gpu_benchmarks/run_cuda_soundness_gate.py \
  --stwo '${STWO_POD}' --runtime-mode '${GPU_PCS_RUNTIME_MODE}' \
  --synced-stwo-head '${STWO_REV}' \
  --synced-stwo-worktree-hash '${STWO_WORKTREE_HASH}' \
  --synced-stwo-cairo-head '${CAIRO_REV}' \
  --synced-stwo-cairo-worktree-hash '${CAIRO_WORKTREE_HASH}' \
  --pod-id '${POD_ID_RESOLVED}' \
  --gpu-bench '${SEALED_GPU_BENCH_PATH}' \
  ${input_cli} \
  --output '${POD_SOUNDNESS_GATE}'
echo \$? > '${gate_rc}'
EOF
  run_ssh "rm -f '${gate_rc}' '${gate_pgid}' '${POD_SOUNDNESS_GATE}'; \
    nohup setsid bash '${gate_sh}' > '${gate_log}' 2>&1 & echo LAUNCHED"
  if ! remote_detached_startup_ok "$gate_pgid" "$gate_rc"; then
    warn "CUDA soundness gate failed its detached-launch startup contract"
    kill_remote_process_group "$gate_pgid" \
      || remote_process_group_gone "$gate_pgid" \
      || warn "CUDA soundness startup left an unverified process group"
    run_ssh "cat '${POD_SOUNDNESS_GATE}' 2>/dev/null" > "$LOCAL_SOUNDNESS_GATE" || true
    run_ssh "cat '${gate_log}' 2>/dev/null" > "${RESULTS_DIR}/${STAMP}.cuda-soundness-gate.startup.log" || true
    return 1
  fi
  local waited=0 code=""
  while true; do
    code="$(run_ssh "cat '${gate_rc}' 2>/dev/null" 2>/dev/null || true)"
    [[ -n "$code" ]] && break
    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "CUDA soundness gate exceeded MAX_WAIT (${MAX_WAIT}s); log tail:"
      run_ssh "tail -n 20 '${gate_log}' 2>/dev/null" || true
      kill_remote_process_group "$gate_pgid" \
        || warn "CUDA soundness gate process group remained after TERM/KILL"
      run_ssh "cat '${POD_SOUNDNESS_GATE}' 2>/dev/null" > "$LOCAL_SOUNDNESS_GATE" || true
      run_ssh "cat '${gate_log}' 2>/dev/null" > "${RESULTS_DIR}/${STAMP}.cuda-soundness-gate.timeout.log" || true
      return 1
    fi
    sleep "$POLL_INTERVAL"
  done
  run_ssh "cat '${POD_SOUNDNESS_GATE}' 2>/dev/null" > "$LOCAL_SOUNDNESS_GATE" || true
  if [[ "$code" != "0" ]]; then
    warn "CUDA soundness gate failed (exit=${code:-?}); artifact: ${LOCAL_SOUNDNESS_GATE}"
    run_ssh "tail -n 40 '${gate_log}' 2>/dev/null" || true
    return 1
  fi
  verify_remote_source_projection \
    || die "remote source no longer equals the checksum-synced local source"
  seal_source_projection "$LOCAL_SOUNDNESS_GATE"
  validate_counted_soundness_gate "$LOCAL_SOUNDNESS_GATE" \
    || die "counted soundness artifact failed its full contract"
  validate_remote_execution_target
  LOCAL_SOUNDNESS_GATE_SHA="$(sha256_file "$LOCAL_SOUNDNESS_GATE")"
  log "CUDA soundness gate PASSED: ${LOCAL_SOUNDNESS_GATE}"
}

# ---------------------------------------------------------------------------
# Synthetic run output for DRY_RUN (exercises the ledger/summary path offline)
# ---------------------------------------------------------------------------
synth_out() {
  # $1 name  $2 args  $3 destfile — deterministic-ish mock numbers keyed off the name.
  local name="$1" args="$2" dest="$3"
  local backend="cuda"; [[ "$args" == *"--backend simd"* ]] && backend="simd"
  local architecture_fields='"engine":"legacy","gpu_pcs_driver_architecture":null,"gpu_pcs_runtime_mode":null,"gpu_pcs_stage_started":null,"gpu_pcs_stage_finished":null,"gpu_pcs_batched_tree_decommit":null,"gpu_pcs_driver_complete":null,"gpu_native_architecture_required":false,"gpu_pcs_required_runtime_mode":null,"gpu_native_architecture_gate_passed":null,"gpu_aot_loads":null,"gpu_aot_cache_hits":null,"gpu_aot_manifest_hash":null,"gpu_aot_misses":null,"gpu_aot_runtime_loads":null,"gpu_aot_runtime_cache_hits":null,"gpu_aot_strict_rejections":null,"gpu_aot_provenance_gate_passed":null'
  local proof_blake3
  proof_blake3="$(printf 'c%.0s' {1..64})"
  local simd_reference_fields='"simd_reference_required":false,"simd_reference_comparison_applicable":false,"simd_reference_byte_equal":null,"simd_reference_blake3":null,"simd_reference_fresh":null,"simd_reference_s":null'
  if [[ "$args" == *"--require-simd-reference-byte-equal"* ]]; then
    simd_reference_fields='"simd_reference_required":true,"simd_reference_comparison_applicable":true,"simd_reference_byte_equal":true,"simd_reference_blake3":"'"${proof_blake3}"'","simd_reference_fresh":true,"simd_reference_s":1.001'
  fi
  if [[ "$args" == *"--require-gpu-native-architecture"* ]]; then
    local runtime_report="DetachedEager"
    [[ "$GPU_PCS_RUNTIME_MODE" == "arena-graph" ]] && runtime_report="ArenaGraph"
    local stage_counts='{"OodsEvaluation":1,"QuotientAndCompaction":1,"FriCommitAndFold":1,"ProofOfWork":1,"FriQueryAndDecommit":1,"TreeDecommit":1,"Assembly":1}'
    # Dry-only topology fixture: use the sealed H100 count only for the exact
    # four-policy profile; other synthetic profiles keep the generic fixture.
    local synthetic_kernel_launches=7859
    if [[ " $BENCH_ENV " == *" STWO_CUDA_B2N_STAGE_FUSED=1 "* \
       && " $BENCH_ENV " == *" STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 "* \
       && " $BENCH_ENV " == *" STWO_CUDA_COMPOSITION_DIRECT_RETENTION=1 "* \
       && " $BENCH_ENV " == *" STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS=1 "* ]]; then
      synthetic_kernel_launches=2473
    fi
    architecture_fields='"engine":"gpu-native","gpu_pcs_driver_architecture":"cuda-typed-pcs-driver-v1","gpu_pcs_runtime_mode":"'"$runtime_report"'","gpu_pcs_stage_started":'"$stage_counts"',"gpu_pcs_stage_finished":'"$stage_counts"',"gpu_pcs_batched_tree_decommit":true,"gpu_pcs_driver_complete":true,"gpu_native_architecture_required":true,"gpu_pcs_required_runtime_mode":"'"$GPU_PCS_RUNTIME_MODE"'","gpu_native_architecture_gate_passed":true,"gpu_aot_loads":2,"gpu_aot_cache_hits":5,"gpu_aot_manifest_hash":49370,"gpu_aot_misses":0,"gpu_aot_runtime_loads":0,"gpu_aot_runtime_cache_hits":0,"gpu_aot_strict_rejections":0,"gpu_aot_provenance_gate_passed":true,"performance_claim_admissible":true,"gpu_host_syncs":1,"gpu_graph_launches":29,"gpu_kernel_launches":'"$synthetic_kernel_launches"',"gpu_hot_h2d_bytes":0,"gpu_hot_d2h_bytes":1024,"gpu_hot_allocations":0,"gpu_max_graph_submit_gap_ms":1.25,"gpu_graph_a_setup_gate_passed":true,"gpu_setup_base_migration_copies":0,"gpu_setup_lookup_host_copies":0,"gpu_setup_legacy_witness_fallbacks":0,"gpu_execution_tables_ingest_compact_h2d_bytes":4096,"gpu_execution_tables_ingest_compact_h2d_copies":3,"gpu_execution_tables_ingest_descriptor_h2d_bytes":64,"gpu_execution_tables_ingest_descriptor_h2d_copies":2,"gpu_execution_tables_ingest_syncs":1,"gpu_witness_ingest_syncs":1'
  fi
  local seed=$(( $(printf '%s' "$name" | cksum | cut -d' ' -f1) % 40 ))
  local um um_median mhz_median warm_s warm_rounded raw_samples program
  local verified_reps=1 warm_count rep
  if [[ "$args" =~ --reps[[:space:]]+([0-9]+) ]]; then
    verified_reps="${BASH_REMATCH[1]}"
  fi
  warm_count=$((verified_reps - 1))
  warm_s="$(awk -v s="$seed" 'BEGIN{printf "%.9f", 8.0 + s/100.0}')"
  warm_rounded="$(awk -v s="$warm_s" 'BEGIN{printf "%.3f", s}')"
  um_median="$(awk -v s="$warm_s" 'BEGIN{printf "%.3f", 12.0/s}')"
  mhz_median="$(awk -v s="$warm_s" 'BEGIN{printf "%.3f", 14.6/s}')"
  um="$um_median"
  raw_samples="["
  for ((rep = 0; rep < warm_count; rep++)); do
    [[ "$rep" == "0" ]] || raw_samples+=","
    raw_samples+="$warm_s"
  done
  raw_samples+="]"
  program="${name}.zip"
  [[ "$name" == "gate_correctness" ]] && program="SN_PIE_2.zip"
  [[ "${FAKE_MISSING_QUALIFICATION_METRICS:-}" == "$name" ]] && um_median="null"
  {
    for ((rep = 0; rep < verified_reps; rep++)); do
      echo "{\"rep\":${rep},\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":1234.5},\"fri\":{\"count\":1,\"total_ms\":567.8}}}"
    done
    echo "{\"program\":\"${program}\",\"backend\":\"${backend}\",${architecture_fields},${simd_reference_fields},\"n\":1,\"cycle_count\":14600000,\"pie_n_steps\":12000000,\"bootloader_overhead_pct\":21.6,\"reps\":${verified_reps},\"gpu_proof_loop_started_unix_ns\":1700000000000000000,\"gpu_proof_loop_finished_unix_ns\":1700000060000000000,\"warm_sample_count\":${warm_count},\"prove_s_warm_samples_raw\":${raw_samples},\"prove_s_warm_samples_rounded\":${raw_samples},\"prove_s_cold\":9.9,\"prove_s_warm\":${warm_rounded},\"prove_s_warm_median\":${warm_rounded},\"prove_s_warm_p95\":${warm_rounded},\"verify_ms\":42.0,\"verified_reps\":${verified_reps},\"proof_kb\":210.5,\"peak_rss_gb\":18.2,\"vram_end_gb\":6.1,\"vram_peak_gb\":11.3,\"pool_used_high_gb\":5.0,\"pool_reserved_high_gb\":6.0,\"steps_per_s\":$(awk -v s="$warm_s" 'BEGIN{printf "%.0f",14600000/s}'),\"mhz\":${mhz_median},\"mhz_median\":${mhz_median},\"mhz_at_warm_p95\":${mhz_median},\"useful_mhz\":${um},\"useful_mhz_median\":${um_median},\"useful_mhz_at_warm_p95\":${um_median},\"throughput_distribution_applicable\":true,\"proof_comparison_applicable\":true,\"proof_byte_equal\":true,\"proof_byte_equal_required\":true,\"gpu_proof_blake3\":\"${proof_blake3}\",\"vm_s\":30.2,\"adapt_s\":5.1,\"security_bits\":96,\"n_queries\":70,\"pow_bits\":26,\"fold_step\":3,\"gpu\":\"${POD_GPU}\",\"nproc\":32,\"host_mem_gb\":125.6}"
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
  local name="$1" args="$2" input_paths="${3:-}"
  local base="${RESULTS_DIR}/${STAMP}.${name}"
  local pod_out="${POD_RUN_DIR}/${STAMP}.${name}.out"
  local pod_err="${POD_RUN_DIR}/${STAMP}.${name}.err"
  local pod_rc="${POD_RUN_DIR}/${STAMP}.${name}.rc"
  local pod_sh="${POD_RUN_DIR}/${STAMP}.${name}.sh"
  local pod_pid="${POD_RUN_DIR}/${STAMP}.${name}.pid"
  local pod_pgid="${POD_RUN_DIR}/${STAMP}.${name}.pgid"
  local pod_quiescence="${POD_RUN_DIR}/${STAMP}.${name}.quiescence"
  local pod_launcher_log="${POD_RUN_DIR}/${STAMP}.${name}.launcher.log"
  local telemetry_required=0
  [[ "$name" == SN_PIE_[1-4] ]] && telemetry_required=1
  local pod_telemetry="${POD_RUN_DIR}/${STAMP}.${name}.gpu-telemetry.csv"
  local pod_telemetry_complete="${POD_RUN_DIR}/${STAMP}.${name}.gpu-telemetry.complete"
  local pod_telemetry_stop="${POD_RUN_DIR}/${STAMP}.${name}.gpu-telemetry.stop"
  local pod_telemetry_meta="${POD_RUN_DIR}/${STAMP}.${name}.gpu-telemetry.remote-meta"
  LAST_STALL_FILE=""
  LAST_PROOF_SHA=""
  LAST_REMOTE_QUIESCENCE_PASSED=false
  LAST_TELEMETRY_PATH=""
  LAST_TELEMETRY_SAMPLER_COMPLETE=false
  LAST_TELEMETRY_REMOTE_SHA=""
  LAST_TELEMETRY_REMOTE_SIZE=""
  LAST_TELEMETRY_TRANSPORT_EQUAL=false
  local pod_proof="${POD_RUN_DIR}/${STAMP}.${name}.proof"
  local proof_export=""
  [[ "$BENCH_PROOF_HASHES" == "1" ]] && proof_export="export STWO_DUMP_PROOF='${pod_proof}'"

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
    if [[ "$telemetry_required" == "1" ]]; then
      LAST_TELEMETRY_PATH="${base}.gpu-telemetry.csv"
      {
        echo "$GPU_TELEMETRY_COLUMNS"
        local timestamp utilization
        for ((timestamp = 1699999999000000000;
              timestamp <= 1700000061000000000;
              timestamp += 250000000)); do
          utilization=0
          ((timestamp >= 1700000000000000000 && timestamp <= 1700000060000000000)) \
            && utilization=98
          printf '%s,%s,42,12000,680.5,1980,2619,64,570.86.15,700,1980,2619\n' \
            "$timestamp" "$utilization"
        done
      } > "$LAST_TELEMETRY_PATH"
      LAST_TELEMETRY_SAMPLER_COMPLETE=true
      LAST_TELEMETRY_REMOTE_SHA="$(sha256_file "$LAST_TELEMETRY_PATH")"
      LAST_TELEMETRY_REMOTE_SIZE="$(wc -c < "$LAST_TELEMETRY_PATH" | tr -d '[:space:]')"
      LAST_TELEMETRY_TRANSPORT_EQUAL=true
    fi
    : > "${base}.err"
    echo 0 > "${base}.rc"
    LAST_OUT="${base}.out"; LAST_RC=0
    LAST_REMOTE_QUIESCENCE_PASSED=true
    [[ "$BENCH_PROOF_HASHES" == "1" ]] \
      && LAST_PROOF_SHA="$(printf 'dryrun-proof:%s' "$name" | sha256_stream | cut -d' ' -f1)"
    return 0
  fi

  local seal_guard quiescence_guard
  seal_guard="$(remote_execution_guard "${input_paths}${input_paths:+,}${POD_BOOTLOADER_JSON}")" \
    || die "could not construct the sealed guard for run '${name}'"
  quiescence_guard="$(remote_quiescence_guard)" \
    || die "could not construct the remote quiescence guard for run '${name}'"

  # Write a detached launcher to the pod scratch dir (avoids nested-quote hell), then
  # start it with setsid+nohup so it survives ssh channel close. The launcher records
  # its own pid (== process-group id under setsid, used to kill a stalled run) and the
  # gpu_bench pid (used to read /proc task states). STWO_JIT_LOG is ALWAYS on — cold
  # JIT-compile visibility is how hangs get caught. BENCH_ENV (if any) is exported
  # verbatim for every run, gate included.
  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  run_ssh "cat > '${pod_sh}'" <<EOF
#!/usr/bin/env bash
trap 'rc=\$?; printf "%s\\n" "\$rc" > "${pod_rc}"' EXIT
set -euo pipefail
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
while IFS='=' read -r stwo_name _; do
  case "\$stwo_name" in STWO_*) unset "\$stwo_name" ;; esac
done < <(env)
unset CUDA_LAUNCH_BLOCKING CUDA_DEVICE_MAX_CONNECTIONS
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
export STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}'
${proof_export}
${seal_guard}
${quiescence_guard}
printf 'pre-passed\n' > '${pod_quiescence}'
echo \$\$ > '${pod_pgid}'
telemetry_pid=''
if [[ '${telemetry_required}' == '1' ]]; then
  printf '%s\n' '${GPU_TELEMETRY_COLUMNS}' > '${pod_telemetry}'
  (
    while [[ ! -e '${pod_telemetry_stop}' ]]; do
      sample="\$(timeout --signal=KILL 2s nvidia-smi --id='${SEALED_GPU_UUID}' \
        --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,driver_version,power.limit,clocks.max.sm,clocks.max.memory \
        --format=csv,noheader,nounits | head -1)"
      timestamp="\$(date +%s%N)"
      printf '%s,%s\n' "\$timestamp" "\$sample"
      sleep '${GPU_TELEMETRY_SAMPLE_INTERVAL_SECONDS}'
    done
    ) >> '${pod_telemetry}' &
  telemetry_pid=\$!
fi
/proc/self/fd/9 ${args} > '${pod_out}' 2> '${pod_err}' &
GB_PID=\$!
echo \$GB_PID > '${pod_pid}'
set +e
wait \$GB_PID
gb_rc=\$?
set -e
telemetry_ok=1
if [[ '${telemetry_required}' == '1' ]]; then
  printf 'stop\n' > '${pod_telemetry_stop}'
  set +e
  wait "\$telemetry_pid"
  telemetry_rc=\$?
  set -e
  [[ "\$telemetry_rc" == '0' ]] || telemetry_ok=0
  if [[ "\$telemetry_ok" == '1' ]]; then
    telemetry_sha="\$(sha256sum '${pod_telemetry}' | cut -d' ' -f1)"
    telemetry_size="\$(stat -c %s '${pod_telemetry}')"
    printf '%s %s\n' "\$telemetry_sha" "\$telemetry_size" > '${pod_telemetry_meta}'
    printf 'complete\n' > '${pod_telemetry_complete}'
  fi
fi
[[ "\$gb_rc" == '0' ]] || exit "\$gb_rc"
[[ "\$telemetry_ok" == '1' ]] || exit 98
EOF
  run_ssh "rm -f '${pod_rc}' '${pod_pid}' '${pod_pgid}' '${pod_quiescence}' '${pod_launcher_log}' \
    '${pod_telemetry}' '${pod_telemetry_complete}' '${pod_telemetry_stop}' '${pod_telemetry_meta}'; \
    nohup setsid bash '${pod_sh}' >'${pod_launcher_log}' 2>&1 & echo LAUNCHED"
  if ! remote_detached_startup_ok "$pod_pgid" "$pod_rc"; then
    warn "run '${name}' failed its detached-launch startup contract"
    kill_remote_process_group "$pod_pgid" \
      || remote_process_group_gone "$pod_pgid" \
      || warn "run '${name}' startup left an unverified process group"
    run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
    run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
    run_ssh "cat '${pod_launcher_log}' 2>/dev/null" > "${base}.startup.log" || true
    LAST_OUT="${base}.out"; LAST_RC=STARTUP_CONTRACT
    return 0
  fi

  # Poll until the rc sentinel appears. Each poll is ONE ssh round-trip that reports
  # completion, the run's stderr size, and the GPU utilization. Today's failure mode
  # was "process alive forever, zero output": if (stderr size, gpu util) is frozen for
  # STALL_SECS, the run is declared stalled — evidence captured, process group killed,
  # LAST_RC=STALLED. MAX_WAIT stays as the hard backstop.
  local waited=0 last_sig="__init__" stall_at now st err_size gpu_util sig timed_out=0
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
      kill_remote_process_group "$pod_pgid" \
        || warn "run '${name}' process group remained after TERM/KILL"
      run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
      run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
      LAST_OUT="${base}.out"; LAST_RC="STALLED"; LAST_STALL_FILE="$stall_file"
      log "stall evidence written to ${stall_file}"
      return 0
    fi

    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "run '${name}' exceeded MAX_WAIT (${MAX_WAIT}s). Fetching partial output."
      timed_out=1
      kill_remote_process_group "$pod_pgid" \
        || warn "run '${name}' process group remained after TERM/KILL"
      break
    fi
    sleep "$POLL_INTERVAL"
  done

  # (f) pull results.
  run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
  run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
  run_ssh "cat '${pod_rc}'  2>/dev/null" > "${base}.rc"  || true
  if [[ "$telemetry_required" == "1" ]]; then
    local telemetry_transport_ok=1 telemetry_remote_meta telemetry_remote_extra=""
    local telemetry_local_sha telemetry_local_size telemetry_complete
    LAST_TELEMETRY_PATH="${base}.gpu-telemetry.csv"
    if ! telemetry_remote_meta="$(run_ssh "cat '${pod_telemetry_meta}'")"; then
      telemetry_transport_ok=0
    elif ! read -r LAST_TELEMETRY_REMOTE_SHA LAST_TELEMETRY_REMOTE_SIZE telemetry_remote_extra \
        <<< "$telemetry_remote_meta" \
        || [[ ! "$LAST_TELEMETRY_REMOTE_SHA" =~ ^[0-9a-f]{64}$ ]] \
        || [[ ! "$LAST_TELEMETRY_REMOTE_SIZE" =~ ^[0-9]+$ ]] \
        || [[ -n "$telemetry_remote_extra" ]]; then
      telemetry_transport_ok=0
    fi
    if ! run_ssh "cat '${pod_telemetry}'" > "$LAST_TELEMETRY_PATH"; then
      telemetry_transport_ok=0
    fi
    telemetry_local_sha="$(sha256_file "$LAST_TELEMETRY_PATH")"
    telemetry_local_size="$(wc -c < "$LAST_TELEMETRY_PATH" | tr -d '[:space:]')"
    if [[ "$telemetry_local_sha" != "$LAST_TELEMETRY_REMOTE_SHA" \
          || "$telemetry_local_size" != "$LAST_TELEMETRY_REMOTE_SIZE" ]]; then
      telemetry_transport_ok=0
    fi
    if telemetry_complete="$(run_ssh "cat '${pod_telemetry_complete}'")" \
        && [[ "$telemetry_complete" == "complete" ]]; then
      LAST_TELEMETRY_SAMPLER_COMPLETE=true
    else
      telemetry_transport_ok=0
    fi
    [[ "$telemetry_transport_ok" == "1" ]] && LAST_TELEMETRY_TRANSPORT_EQUAL=true
  fi
  LAST_RC="$(cat "${base}.rc" 2>/dev/null || echo TIMEOUT)"
  [[ "$timed_out" == "0" ]] || LAST_RC=TIMEOUT
  [[ -n "$LAST_RC" ]] || LAST_RC="TIMEOUT"
  if [[ "$LAST_RC" == "0" && "$telemetry_required" == "1" \
        && "$LAST_TELEMETRY_TRANSPORT_EQUAL" != "true" ]]; then
    LAST_RC=TELEMETRY_TRANSPORT_CONTRACT
  fi
  LAST_OUT="${base}.out"
  if [[ "$LAST_RC" == "0" ]] \
     && [[ "$(run_ssh "cat '${pod_quiescence}' 2>/dev/null" 2>/dev/null || true)" == "pre-passed" ]] \
     && remote_process_group_gone "$pod_pgid" \
     && run_ssh "$(remote_quiescence_guard)"; then
    LAST_REMOTE_QUIESCENCE_PASSED=true
  elif [[ "$LAST_RC" == "0" ]]; then
    LAST_RC=REMOTE_QUIESCENCE_CONTRACT
  fi
  if [[ "$BENCH_PROOF_HASHES" == "1" && "$LAST_RC" == "0" ]]; then
    LAST_PROOF_SHA="$(run_ssh "sha256sum '${pod_proof}'" | cut -d' ' -f1)"
  fi
  log "run '${name}' finished (rc=${LAST_RC})"
}

# A stale pre-contract binary silently ignores unknown CLI flags. Exit status alone
# therefore cannot prove that all gate repetitions were verified and compared.
gate_contract_ok() {
  python3 - "$1" "${2:-2}" "${3:-0}" <<'PY'
import json
import math
import sys

record = None
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "verified_reps" in candidate:
                record = candidate
except OSError as error:
    print(f"gate contract: cannot read output: {error}", file=sys.stderr)
    raise SystemExit(1)

required = {
    "verified_reps": int(sys.argv[2]),
    "proof_comparison_applicable": True,
    "proof_byte_equal_required": True,
    "proof_byte_equal": True,
}
if sys.argv[3] == "1":
    required.update({
        "simd_reference_required": True,
        "simd_reference_comparison_applicable": True,
        "simd_reference_byte_equal": True,
        "simd_reference_fresh": True,
    })
if record is None or any(record.get(key) != value for key, value in required.items()):
    print(f"gate contract missing or false: expected {required}, got {record}", file=sys.stderr)
    raise SystemExit(1)
if sys.argv[3] == "1":
    gpu_digest = record.get("gpu_proof_blake3")
    simd_digest = record.get("simd_reference_blake3")
    reference_s = record.get("simd_reference_s")
    valid_digest = lambda value: (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
    if (not valid_digest(gpu_digest) or not valid_digest(simd_digest)
            or gpu_digest != simd_digest
            or not isinstance(reference_s, (int, float))
            or isinstance(reference_s, bool)
            or not math.isfinite(reference_s) or reference_s <= 0):
        print("gate contract SIMD/GPU proof digest binding is invalid", file=sys.stderr)
        raise SystemExit(1)
PY
}

# The binary accepts flags by manual lookup, so an older binary can ignore an
# unknown architecture flag and still exit zero. Validate the primary record too.
architecture_contract_ok() {
  local out="$1" expected_program="${2:-}" expected_reps="${3:-}" expected_gpu="${4:-}" require_simd="${5:-0}" telemetry_csv="${6:-}"
  local -a measurement_args=()
  if [[ -n "$expected_program" ]]; then
    measurement_args=(--expected-program "$expected_program" --expected-reps "$expected_reps")
    [[ -z "$expected_gpu" ]] || measurement_args+=(--expected-gpu "$expected_gpu")
    [[ "$require_simd" != "1" ]] || measurement_args+=(--require-fresh-simd-reference)
    if [[ -n "$telemetry_csv" ]]; then
      measurement_args+=(
        --gpu-telemetry-csv "$telemetry_csv"
        --gpu-telemetry-remote-sha256 "$LAST_TELEMETRY_REMOTE_SHA"
        --gpu-telemetry-remote-size "$LAST_TELEMETRY_REMOTE_SIZE"
      )
    fi
  fi
  if [[ "$require_simd" == "1" && -z "$telemetry_csv" ]]; then
    echo "fixed-SN benchmark is missing its retained GPU telemetry CSV" >&2
    return 1
  fi
  if [[ -n "$telemetry_csv" && "$LAST_TELEMETRY_SAMPLER_COMPLETE" != "true" ]]; then
    echo "GPU telemetry sampler did not cover the complete benchmark process" >&2
    return 1
  fi
  if [[ -n "$telemetry_csv" && "$LAST_TELEMETRY_TRANSPORT_EQUAL" != "true" ]]; then
    echo "GPU telemetry remote/local transport equality was not established" >&2
    return 1
  fi
  python3 "$ARCHITECTURE_CHECK" "$out" --runtime-mode "$GPU_PCS_RUNTIME_MODE" \
    --soundness-gate "$LOCAL_SOUNDNESS_GATE" "${measurement_args[@]}"
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
  LB_PROOF_SHA="$LAST_PROOF_SHA" LB_PROBE="$QUALIFICATION_PROBE" \
  LB_REMOTE_QUIESCENCE="$LAST_REMOTE_QUIESCENCE_PASSED" \
  LB_DRY_RUN="$DRY_RUN" \
  LB_SOUNDNESS_PATH="${LOCAL_SOUNDNESS_GATE:-}" LB_SOUNDNESS_SHA="$LOCAL_SOUNDNESS_GATE_SHA" \
  LB_TELEMETRY_PATH="${LAST_TELEMETRY_PATH:-}" \
  LB_TELEMETRY_COMPLETE="${LAST_TELEMETRY_SAMPLER_COMPLETE:-false}" \
  LB_TELEMETRY_REMOTE_SHA="${LAST_TELEMETRY_REMOTE_SHA:-}" \
  LB_TELEMETRY_REMOTE_SIZE="${LAST_TELEMETRY_REMOTE_SIZE:-}" \
  LB_TELEMETRY_TRANSPORT_EQUAL="${LAST_TELEMETRY_TRANSPORT_EQUAL:-false}" \
  LB_VALIDATOR_DIR="$(dirname "$ARCHITECTURE_CHECK")" \
  LB_LEDGER="$LEDGER" python3 - <<'PY'
import csv, hashlib, json, os, sys

sys.path.insert(0, os.environ["LB_VALIDATOR_DIR"])
from validate_architecture_record import (
    GPU_TELEMETRY_COLUMNS,
    GPU_TELEMETRY_MAX_GAP_NS,
    GPU_TELEMETRY_SAMPLE_INTERVAL_MS,
    GPU_TELEMETRY_SCHEMA,
    validate_gpu_telemetry_artifact,
)

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
telemetry = None
telemetry_path = os.environ.get("LB_TELEMETRY_PATH", "")
if telemetry_path and status == "ok":
    with open(telemetry_path, "rb") as stream:
        payload = stream.read()
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    start = (record or {}).get("gpu_proof_loop_started_unix_ns")
    finish = (record or {}).get("gpu_proof_loop_finished_unix_ns")
    proof_window_sample_count = sum(
        isinstance(start, int)
        and isinstance(finish, int)
        and start <= int(row["timestamp_unix_ns"]) <= finish
        for row in rows
    )
    telemetry = {
        "schema": GPU_TELEMETRY_SCHEMA,
        "columns": list(GPU_TELEMETRY_COLUMNS),
        "path": telemetry_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "remote_sha256": os.environ.get("LB_TELEMETRY_REMOTE_SHA", ""),
        "size_bytes": len(payload),
        "remote_size_bytes": int(os.environ.get("LB_TELEMETRY_REMOTE_SIZE", "-1")),
        "sample_count": len(rows),
        "proof_window_sample_count": proof_window_sample_count,
        "sampler_complete": os.environ.get("LB_TELEMETRY_COMPLETE") == "true",
        "sample_interval_ms": GPU_TELEMETRY_SAMPLE_INTERVAL_MS,
        "max_gap_ns": GPU_TELEMETRY_MAX_GAP_NS,
        "transport_equal": os.environ.get("LB_TELEMETRY_TRANSPORT_EQUAL") == "true",
    }
    telemetry_errors = validate_gpu_telemetry_artifact(record or {}, telemetry)
    if telemetry_errors:
        raise SystemExit(f"invalid retained GPU telemetry: {telemetry_errors}")
execution_target, source_projection = None, None
execution_target_sha256 = None
soundness_path = os.environ.get("LB_SOUNDNESS_PATH", "")
if soundness_path:
    with open(soundness_path, encoding="utf-8") as stream:
        soundness = json.load(stream)
    execution_target = soundness.get("execution_target")
    execution_target_sha256 = soundness.get("execution_target_sha256")
    source_projection = soundness.get("source_projection")
is_provisional = os.environ.get("LB_PROBE") == "1" or run in ("gate_correctness", "gate_failed")
performance_admissible = (
    not is_provisional and status == "ok" and os.environ.get("LB_DRY_RUN") != "1"
)

# The comparison metric: sustained useful MHz for a pipelined (fleet) run, else the
# warm-sample median for a fixed statement. Legacy warm-best useful_mhz is retained
# in raw records for compatibility but is never a ranking or claim basis.
def metric_of(rec, pipe):
    if pipe and pipe.get("sustained_useful_mhz") is not None:
        return pipe["sustained_useful_mhz"], "sustained_useful_mhz"
    if rec and rec.get("useful_mhz_median") is not None:
        return rec["useful_mhz_median"], "useful_mhz_median"
    return None, None

new_metric, new_basis = metric_of(record, pipeline)
if not performance_admissible:
    new_metric, new_basis = None, None

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
                    and e.get("execution_target") == execution_target
                    and e.get("status", "ok") == "ok"
                    and not e.get("provisional", False)
                    and e.get("bench_env", "") == benv):
                m, _ = metric_of(e.get("record"), e.get("pipeline"))
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
    "proof_sha256": os.environ.get("LB_PROOF_SHA", ""),
    "qualification_probe": os.environ.get("LB_PROBE") == "1",
    "provisional": is_provisional,
    "performance_admissible": performance_admissible,
    "soundness_gate_path": os.environ.get("LB_SOUNDNESS_PATH", ""),
    "soundness_gate_sha256": os.environ.get("LB_SOUNDNESS_SHA", ""),
    "execution_target": execution_target,
    "execution_target_sha256": execution_target_sha256,
    "source_projection": source_projection,
    "execution_guard_passed": status == "ok",
    "remote_quiescence_passed": status == "ok" and os.environ.get("LB_REMOTE_QUIESCENCE") == "true",
    "mhz_basis": new_basis,
    "record": record,
    "phase_totals": phases,
}
if telemetry is not None:
    entry["gpu_telemetry"] = telemetry
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
    print(f"  {disp:<21} claim_mhz=n/a")
else:
    if prev_metric is None:
        delta = "(no prior same-pod same-env run)"
    else:
        pct = (new_metric - prev_metric) / prev_metric * 100.0 if prev_metric else 0.0
        delta = f"{pct:+.1f}% vs {prev_metric:.3f}"
    vram = (record or {}).get("vram_peak_gb", "?")
    print(f"  {disp:<21} {new_basis}={new_metric:.3f}  vram_peak_gb={vram}  {delta}")
PY
}

# ===========================================================================
# Orchestration
# ===========================================================================
log "=== bench_loop start (pie=${PIE_SEL} reps=${REPS} all_pies=${ALL_PIES} full=${FULL} simd=${SIMD} skip_sync=${SKIP_SYNC} gate_only=${GATE_ONLY} dry=${DRY_RUN}) ==="

if [[ "$SKIP_SYNC" == "1" ]]; then
  log "--skip-sync: reusing the binary already on the pod (no repo sync, no build)"
else
  bootstrap_pod
  sync_repos
  verify_remote_source_projection \
    || die "remote source does not equal the checksum-synced local source"
  build_pod
  seal_gpu_bench
fi

# Keep this as a simple command: invoking a function under `if !` disables
# errexit inside it, which could turn a failed remote launch into a 3h poll.
run_cuda_soundness_gate

# Abort helper for a stalled run: self-documenting ledger entry with evidence, then die.
abort_stalled() {
  local nm="$1"
  append_ledger "$nm" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.${nm}.err" "stalled"
  die "run '${nm}' stalled — process killed; evidence in ${LAST_STALL_FILE} and the ledger entry."
}

# (d) CORRECTNESS GATE — always first, under the SAME BENCH_ENV as the benchmarks
# (a kill switch that changes behavior must be gated too; the launcher exports
# BENCH_ENV for every run including this one).
GATE_ARGS="--pie ${POD_GATE_PIE} --backend cuda ${GPU_NATIVE_ARGS} --reps 2 --reuse-input --require-proof-byte-equal"
log "=== CORRECTNESS GATE: ${POD_GATE_PIE} CUDA prove+verify ==="
run_bench "gate_correctness" "$GATE_ARGS" "$POD_GATE_PIE"
validate_remote_execution_target
if [[ "$LAST_RC" == "STALLED" ]]; then
  abort_stalled "gate_correctness"
fi
if [[ "$LAST_RC" == "0" ]] && ! gate_contract_ok "$LAST_OUT"; then
  LAST_RC="GATE_CONTRACT"
fi
if [[ "$LAST_RC" == "0" ]] \
   && ! architecture_contract_ok "$LAST_OUT" "SN_PIE_2.zip" 2 "$POD_GPU"; then
  LAST_RC="ARCHITECTURE_CONTRACT"
fi
if [[ "$LAST_RC" != "0" ]]; then
  warn "GATE FAILED (rc=${LAST_RC}) — typed CUDA architecture plus two verified, byte-identical proofs were not demonstrated."
  append_ledger "gate_failed" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.gate_correctness.err" "gate_failed"
  die "correctness gate failed — refusing to report any performance from this build."
fi
append_ledger "gate_correctness" "$LAST_OUT" "${RESULTS_DIR}/${STAMP}.gate_correctness.err" "ok"
log "gate PASSED"

if [[ "$GATE_ONLY" == "1" ]]; then
  log "--gate-only: done."
  exit 0
fi

# Build the benchmark run list.
declare -a RUN_NAMES RUN_ARGS RUN_INPUTS
add_run() { RUN_NAMES+=("$1"); RUN_ARGS+=("$2"); RUN_INPUTS+=("$3"); }

SEL_PATH="$(pie_path "$PIE_SEL")"
SEL_NAME="$(pie_name "$PIE_SEL")"
add_run "$SEL_NAME" "--pie ${SEL_PATH} --backend cuda ${GPU_NATIVE_ARGS} --reps ${REPS} --reuse-input --require-proof-byte-equal --require-simd-reference-byte-equal" "$SEL_PATH"

if [[ "$SIMD" == "1" ]]; then
  add_run "${SEL_NAME}_simd" "--pie ${SEL_PATH} --backend simd --reps ${REPS} --reuse-input --require-proof-byte-equal" "$SEL_PATH"
fi

if [[ "$ALL_PIES" == "1" || "$FULL" == "1" ]]; then
  for s in 1 3 4; do
    nm="$(pie_name "$s")"
    [[ "$nm" == "$SEL_NAME" ]] && continue   # already queued as the selected PIE
    pie="$(pie_path "$s")"
    add_run "$nm" "--pie ${pie} --backend cuda ${GPU_NATIVE_ARGS} --reps ${REPS} --reuse-input --require-proof-byte-equal --require-simd-reference-byte-equal" "$pie"
  done
  if [[ "$FULL" == "1" ]]; then
    FLEET_LIST="${POD_SN_DIR}/SN_PIE_1.zip,${POD_SN_DIR}/SN_PIE_2.zip,${POD_SN_DIR}/SN_PIE_3.zip,${POD_SN_DIR}/SN_PIE_4.zip"
    add_run "SN_fleet_rotate" "--pie ${FLEET_LIST} --backend cuda ${GPU_NATIVE_ARGS} --reps ${FLEET_REPS} --pipeline ${FLEET_DEPTH} --producers ${FLEET_PRODUCERS} --pie-mode rotate" "$FLEET_LIST"
  fi
fi

log "=== benchmarking ${#RUN_NAMES[@]} run(s) ==="
FAILED=0
echo "=== bench_loop summary (${TS}) — pod ${POD_GPU} ==="
echo "    revs: stwo=${STWO_REV:0:8}${STWO_DIRTY:+(${STWO_DIRTY})} cairo=${CAIRO_REV:0:8}${CAIRO_DIRTY:+(${CAIRO_DIRTY})}"
[[ -n "$BENCH_ENV" ]] && echo "    bench_env(!): ${BENCH_ENV}  — NOT comparable with clean runs"
for idx in "${!RUN_NAMES[@]}"; do
  nm="${RUN_NAMES[$idx]}"; ar="${RUN_ARGS[$idx]}"; input_paths="${RUN_INPUTS[$idx]}"
  run_bench "$nm" "$ar" "$input_paths"
  validate_remote_execution_target
  if [[ "$LAST_RC" == "STALLED" ]]; then
    abort_stalled "$nm"
  fi
  if [[ "$LAST_RC" == "0" && "$ar" == *"--require-gpu-native-architecture"* ]]; then
    if [[ "$nm" == SN_PIE_[1-4] ]]; then
      architecture_contract_ok "$LAST_OUT" "${nm}.zip" "$REPS" "$POD_GPU" 1 "$LAST_TELEMETRY_PATH" \
        || LAST_RC="ARCHITECTURE_CONTRACT"
    elif ! architecture_contract_ok "$LAST_OUT"; then
      LAST_RC="ARCHITECTURE_CONTRACT"
    fi
  fi
  if [[ "$LAST_RC" == "0" && "$ar" == *"--require-proof-byte-equal"* ]] \
     && ! gate_contract_ok "$LAST_OUT" "$REPS" "$([[ "$ar" == *"--require-simd-reference-byte-equal"* ]] && echo 1 || echo 0)"; then
    LAST_RC="PROOF_CONTRACT"
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
