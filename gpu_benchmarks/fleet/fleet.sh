#!/usr/bin/env bash
#
# fleet.sh — the harness for the 10–20 MHz AGGREGATE proving demonstration.
#
# Given a roster of RunPod pods (fleet.conf), this launches one rotate-mode pipelined
# gpu_bench stream on EVERY pod concurrently, polls all of them (with per-pod stall
# detection), collects each pod's self-describing JSON record, and computes the fleet
# aggregate: total useful MHz, total $/hr, and $/MHz-hr — the numbers that turn "one
# consumer card does ~1–2.5 useful MHz" into "N cards do 10–20 MHz at $X/MHz-hr".
# It writes one fleet_report.json + a human table.
#
# This script runs NO prover code and cannot change proof bytes (pure orchestration).
# Its only correctness dependency is the standard per-pod 10-transfer PIE CUDA
# prove+verify gate (loop's gate, ON by default) — a pod that fails its gate is
# EXCLUDED from the aggregate, never silently averaged in. Performance is never
# reported for a pod whose gate failed.
#
# Relationship to loop/bench_loop.sh: fleet.sh mirrors its discipline (runtime pod
# resolution, git provenance, gate-first, nohup+setsid detach, stall detection,
# DRY_RUN) and delegates the sync+build+gate PREP to it per pod (`--prep`). It does
# NOT modify bench_loop.sh. The difference is the axis of concurrency: bench_loop
# runs N runs on ONE pod sequentially; fleet runs ONE run on N pods concurrently.
#
# Pipeline:
#   (0) roster       : parse fleet.conf -> enabled pods (id/gpu/$hr + fallbacks).
#                      `--only <id>` or the per-pod `enabled=0` column toggle lanes.
#   (a) provenance   : both repos' git rev + sha256 of the working diff (once, shared).
#   (b) prep [opt]   : `--prep` runs `bench_loop.sh --gate-only` per pod (sync+build+
#                      gate). Reuses the loop wholesale. A pod that fails prep is
#                      dropped from the fleet.
#   (c) resolve      : `runpodctl ssh info <id>` per pod -> host/port/key; fall back to
#                      the fb_* columns with a loud warning (may be stale).
#   (d) GATE         : unless `--skip-gate`, a 10-transfer PIE CUDA prove+verify on
#                      EVERY pod concurrently. Pods that fail are dropped; the report
#                      records them as gate_failed. (`--prep` implies the gate already
#                      ran; the explicit gate stage is then skipped.)
#   (e) benchmark    : one rotate-mode pipelined stream per pod, launched detached
#                      (nohup+setsid) on all pods at once, polled together with
#                      per-pod stall detection. STWO_BENCH_TRACE=json, STWO_JIT_LOG=1.
#   (f) collect      : pull each pod's stdout (main record + pipeline record).
#   (g) aggregate    : sum useful MHz, sum $/hr, compute $/MHz-hr; pods-needed for the
#                      10 and 20 MHz targets; write fleet_report.json.
#   (h) table        : human summary to stdout.
#
# Usage:
#   ./fleet.sh [--conf PATH] [--pies LIST] [--reps N] [--depth D] [--producers P]
#              [--only POD_ID] [--prep] [--skip-gate] [--help]
#
# Flags:
#   --conf PATH    Pod roster (default: fleet/fleet.conf).
#   --pies LIST    Comma list of PIE selectors for the rotate stream: any of
#                  1|2|3|4|10t (default "1,2,3,4" — the production block-stream shape).
#   --reps N       Reps (proofs) per pod's stream (default FLEET_REPS=8).
#   --depth D      Pipeline depth per pod (default FLEET_DEPTH=3).
#   --producers P  Host producer threads per pod (default FLEET_PRODUCERS=4).
#   --only POD_ID  Run exactly one pod from the roster (bisect a single lane).
#   --prep         Run `bench_loop.sh --gate-only` per pod first (sync+build+gate).
#                  Without it, pods are assumed already built (binary present).
#   --skip-gate    Skip fleet's own correctness gate (bisect escape hatch). The
#                  report is then marked "ungated" — never trust its numbers.
#
# Environment:
#   FLEET_CONF     Roster path (overridden by --conf).
#   FLEET_ONLY     Single pod id (overridden by --only).
#   FLEET_REPS / FLEET_DEPTH / FLEET_PRODUCERS   Rotate knobs (8 / 3 / 4).
#   BENCH_ENV      "K=V K=V ..." exported verbatim into every gpu_bench invocation
#                  (gate + benchmark) on every pod, and recorded in the report. For
#                  debug bisects (STWO_CUDA_DISABLE_STREAMS=1, ...). No spaces in values.
#   DRY_RUN=1      Echo every ssh instead of executing; fabricate per-pod output so the
#                  roster -> launch -> poll -> aggregate -> report path runs offline.
#   FAKE_STALL     (DRY_RUN only) pod id to simulate as stalled in the benchmark stage.
#   POLL_INTERVAL  Seconds between poll sweeps (default 15).
#   STALL_SECS     Declare a pod stalled when its stderr size AND GPU util are both
#                  frozen for this long (default 600).
#   MAX_WAIT       Hard cap on waiting for the fleet (default 10800 = 3h).
#   GATE_PIE       Remote correctness-gate PIE path. Explicitly point this at
#                  .../pie/sn/SN_PIE_2.zip if the smaller 10-transfer PIE is absent.
#   POD_BOOTLOADER_JSON Stable remote bootloader path preflighted and exported for
#                  every gate/benchmark. Default: /workspace/bench_inputs/
#                  simple_bootloader_compiled.json.
#   GPU_PCS_RUNTIME_MODE Required typed CUDA PCS mode for every run: detached-eager
#                  (default) or arena-graph (strict future gate).
#
# NOTE: only same-pod comparisons are meaningful (community-host variance). The
# aggregate is a SUM across heterogeneous pods — a fleet capacity number, not a
# per-pod delta. See README.md for the fleet math (RESULTS round 8).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (all paths as variables, up top)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
FLEET_DIR="${SCRIPT_DIR}"
LOOP_DIR="${CAIRO_LOCAL}/gpu_benchmarks/loop"
BENCH_LOOP="${LOOP_DIR}/bench_loop.sh"
RESULTS_DIR="${RESULTS_DIR:-${FLEET_DIR}/results}"
FLEET_CONF="${FLEET_CONF:-${FLEET_DIR}/fleet.conf}"
REPORT="${REPORT:-${FLEET_DIR}/fleet_report.json}"
INPUT_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/SHA256SUMS"
ARCHITECTURE_CHECK="${CAIRO_LOCAL}/gpu_benchmarks/validate_architecture_record.py"

# Pod repos + binary (identical layout to the loop).
CAIRO_POD="/workspace/stwo-cairo"
POD_PROVER_DIR="${CAIRO_POD}/stwo_cairo_prover"
BIN="target/release/gpu_bench"                      # relative to POD_PROVER_DIR
POD_USER="root"

# PIE inputs on the pod (already present, hash-verified — never synced).
POD_SN_DIR="${CAIRO_POD}/gpu_benchmarks/pie/sn"
POD_GATE_PIE="${GATE_PIE:-${CAIRO_POD}/gpu_benchmarks/pie/cairo_pie_10_transfers_with_6_ecop.zip}"
POD_BOOTLOADER_JSON="${POD_BOOTLOADER_JSON:-/workspace/bench_inputs/simple_bootloader_compiled.json}"

# Pod scratch (outside the repo tree so no sync ever touches it).
POD_RUN_DIR="/workspace/fleet_runs"

# Prover knobs.
RUST_MIN_STACK_VAL=4194304
BENCH_ENV="${BENCH_ENV:-}"
GPU_PCS_RUNTIME_MODE="${GPU_PCS_RUNTIME_MODE:-detached-eager}"
GPU_NATIVE_ARGS="--engine gpu-native --require-gpu-native-architecture --require-gpu-pcs-runtime-mode ${GPU_PCS_RUNTIME_MODE}"

# Rotate (pipeline) parameters.
FLEET_REPS="${FLEET_REPS:-8}"
FLEET_DEPTH="${FLEET_DEPTH:-3}"
FLEET_PRODUCERS="${FLEET_PRODUCERS:-4}"

# Poll behavior (same knobs/semantics as the loop).
DRY_RUN="${DRY_RUN:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
STALL_SECS="${STALL_SECS:-600}"
MAX_WAIT="${MAX_WAIT:-10800}"

# Flag defaults.
PIES="1,2,3,4"
ONLY="${FLEET_ONLY:-}"
PREP=0
SKIP_GATE=0

# The 10 / 20 MHz targets this harness exists to demonstrate.
TARGET_LO_MHZ=10
TARGET_HI_MHZ=20

# ---------------------------------------------------------------------------
# Per-pod state (parallel arrays, indexed by roster position)
# ---------------------------------------------------------------------------
declare -a POD_IDS POD_GPUS POD_USD POD_HOSTS POD_PORTS POD_KEYS
declare -a POD_STATUS          # pending | ok | gate_failed | run_failed | stalled
declare -a POD_OUT POD_STALL   # local out file / stall-evidence file (last stage)
declare -a POD_RC              # last stage rc: 0 | <code> | STALLED | TIMEOUT

# ---------------------------------------------------------------------------
# Logging helpers ( logs -> stderr, human summary -> stdout )
# ---------------------------------------------------------------------------
log()  { echo "[fleet] $*" >&2; }
dry()  { echo "[DRY_RUN] $*" >&2; }
warn() { echo "[fleet][WARN] $*" >&2; }
die()  { echo "[fleet][FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,84p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

[[ -d "$CAIRO_LOCAL/.git" ]] || die "missing stwo-cairo checkout: $CAIRO_LOCAL"
[[ -d "$STWO_LOCAL/.git" ]] || die "missing sibling stwo checkout: $STWO_LOCAL"
[[ -f "$INPUT_SHA256SUMS" ]] || die "input checksum manifest missing: $INPUT_SHA256SUMS"
[[ -f "$ARCHITECTURE_CHECK" ]] || die "architecture record validator missing: $ARCHITECTURE_CHECK"
STWO_LOCAL="$(cd "$STWO_LOCAL" && pwd)"

case "$GPU_PCS_RUNTIME_MODE" in
  detached-eager|arena-graph) ;;
  *) die "GPU_PCS_RUNTIME_MODE must be detached-eager or arena-graph (got '$GPU_PCS_RUNTIME_MODE')" ;;
esac

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf)      FLEET_CONF="${2:?--conf needs a value}"; shift 2 ;;
    --pies)      PIES="${2:?--pies needs a value}"; shift 2 ;;
    --reps)      FLEET_REPS="${2:?--reps needs a value}"; shift 2 ;;
    --depth)     FLEET_DEPTH="${2:?--depth needs a value}"; shift 2 ;;
    --producers) FLEET_PRODUCERS="${2:?--producers needs a value}"; shift 2 ;;
    --only)      ONLY="${2:?--only needs a value}"; shift 2 ;;
    --prep)      PREP=1; shift ;;
    --skip-gate) SKIP_GATE=1; shift ;;
    -h|--help)   usage 0 ;;
    *)           die "unknown flag: $1 (try --help)" ;;
  esac
done

for n in FLEET_REPS FLEET_DEPTH FLEET_PRODUCERS; do
  v="${!n}"
  [[ "$v" =~ ^[0-9]+$ && "$v" -ge 1 ]] || die "$n must be a positive integer (got '$v')"
done

# Validate BENCH_ENV shape early: every token must be K=V (no spaces in values).
if [[ -n "$BENCH_ENV" ]]; then
  for kv in $BENCH_ENV; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] \
      || die "BENCH_ENV token '$kv' is not K=V (values must not contain spaces)"
    [[ "${kv%%=*}" != "STWO_BOOTLOADER_JSON" ]] \
      || die "STWO_BOOTLOADER_JSON is reserved; set POD_BOOTLOADER_JSON instead"
  done
fi

# ---------------------------------------------------------------------------
# PIE selector -> pod path (same mapping as the loop)
# ---------------------------------------------------------------------------
pie_path() {
  case "$1" in
    1|2|3|4) echo "${POD_SN_DIR}/SN_PIE_$1.zip" ;;
    10t)     echo "${POD_GATE_PIE}" ;;
    *)       die "invalid PIE selector '$1' (want 1|2|3|4|10t)" ;;
  esac
}

# Build the comma-separated pod PIE path list for the rotate stream from --pies.
build_pie_list() {
  local out="" sel
  IFS=',' read -r -a sels <<<"$PIES"
  for sel in "${sels[@]}"; do
    [[ -n "$sel" ]] || continue
    out="${out:+${out},}$(pie_path "$sel")"
  done
  [[ -n "$out" ]] || die "--pies produced an empty PIE list"
  echo "$out"
}

# ---------------------------------------------------------------------------
# (0) Roster parse
# ---------------------------------------------------------------------------
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; echo "$s"; }

parse_roster() {
  [[ -f "$FLEET_CONF" ]] || die "roster not found: ${FLEET_CONF}"
  local line id gpu usd fbh fbp fbk en
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%$'\r'}"                       # tolerate CRLF
    [[ -z "$(trim "$line")" ]] && continue
    [[ "$(trim "$line")" == \#* ]] && continue
    IFS='|' read -r id gpu usd fbh fbp fbk en <<<"$line"
    id="$(trim "$id")"; gpu="$(trim "$gpu")"; usd="$(trim "$usd")"
    fbh="$(trim "${fbh:-}")"; fbp="$(trim "${fbp:-}")"; fbk="$(trim "${fbk:-}")"
    en="$(trim "${en:-1}")"; [[ -n "$en" ]] || en=1
    [[ -n "$id" ]]  || die "roster line missing pod id: '${line}'"
    [[ -n "$gpu" ]] || die "roster line for '${id}' missing gpu label"
    [[ "$usd" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "roster line for '${id}' has non-numeric usd_per_hr '${usd}'"
    if [[ -n "$ONLY" && "$id" != "$ONLY" ]]; then continue; fi   # --only filter
    if [[ "$en" != "1" ]]; then
      log "roster: pod '${id}' (${gpu}) disabled (enabled=${en}) — skipping"
      continue
    fi
    POD_IDS+=("$id"); POD_GPUS+=("$gpu"); POD_USD+=("$usd")
    POD_HOSTS+=("$fbh"); POD_PORTS+=("$fbp"); POD_KEYS+=("$fbk")
    POD_STATUS+=("pending"); POD_OUT+=(""); POD_STALL+=(""); POD_RC+=("")
  done < "$FLEET_CONF"
  [[ ${#POD_IDS[@]} -gt 0 ]] || die "no enabled pods in ${FLEET_CONF}${ONLY:+ (matching --only ${ONLY})}"
}

# ---------------------------------------------------------------------------
# (c) Per-pod resolution — runtime `runpodctl ssh info`, fall back to fb_* loudly.
# Sets POD_HOSTS[idx]/POD_PORTS[idx]/POD_KEYS[idx] in place. Returns 1 if unresolved.
# ---------------------------------------------------------------------------
resolve_pod() {
  local idx="$1" id="${POD_IDS[$1]}"
  local host="" port="" key=""

  if [[ "$DRY_RUN" == "1" ]]; then
    POD_HOSTS[$idx]="${POD_HOSTS[$idx]:-dry-host}"
    POD_PORTS[$idx]="${POD_PORTS[$idx]:-22}"
    POD_KEYS[$idx]="${POD_KEYS[$idx]:-/dev/null}"
    return 0
  fi

  if command -v runpodctl >/dev/null 2>&1; then
    local info parsed
    if info="$(runpodctl ssh info "$id" 2>/dev/null)" &&
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
      read -r host port key <<<"$parsed"
      POD_HOSTS[$idx]="$host"; POD_PORTS[$idx]="$port"; POD_KEYS[$idx]="$key"
      log "pod ${id} resolved via runpodctl: ${host}:${port}"
      return 0
    fi
  fi

  if [[ -n "${POD_HOSTS[$idx]}" && -n "${POD_PORTS[$idx]}" && -n "${POD_KEYS[$idx]}" ]]; then
    warn "runpodctl resolution FAILED for pod '${id}' — using fb_* columns"
    warn "  ${POD_HOSTS[$idx]}:${POD_PORTS[$idx]} — these may be STALE if the pod restarted."
    return 0
  fi
  warn "cannot resolve pod '${id}': runpodctl failed and no fallback host/port/key in roster"
  return 1
}

# ---------------------------------------------------------------------------
# Per-pod ssh wrapper (DRY_RUN-aware; no eval — builds the option array inline).
# ---------------------------------------------------------------------------
pssh() {
  local idx="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then dry "ssh[${POD_IDS[$idx]}]: $*"; return 0; fi
  ssh -p "${POD_PORTS[$idx]}" -i "${POD_KEYS[$idx]}" \
    -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    "${POD_USER}@${POD_HOSTS[$idx]}" "$@"
}

expected_sha256() {
  awk -v file="$(basename "$1")" '$2 == file { print $1; exit }' "$INPUT_SHA256SUMS"
}

[[ -n "$(expected_sha256 "$POD_BOOTLOADER_JSON")" ]] \
  || die "pinned bootloader checksum missing from $INPUT_SHA256SUMS: $(basename "$POD_BOOTLOADER_JSON")"

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else LC_ALL=C shasum -a 256; fi
}

preflight_pod_inputs() {
  local idx="$1" path expected quoted_path quoted_expected
  local remote_cmd='missing=0; hash_input() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d" " -f1; else LC_ALL=C shasum -a 256 "$1" | cut -d" " -f1; fi; };'
  local -a paths=("$POD_BOOTLOADER_JSON") bench_paths=()
  [[ "$SKIP_GATE" == "1" ]] || paths+=("$POD_GATE_PIE")
  IFS=',' read -r -a bench_paths <<<"$PIE_LIST"
  paths+=("${bench_paths[@]}")

  for path in "${paths[@]}"; do
    expected="$(expected_sha256 "$path")"
    printf -v quoted_path '%q' "$path"
    printf -v quoted_expected '%q' "$expected"
    remote_cmd+=" path=${quoted_path}; expected=${quoted_expected}; if [ ! -f \"\$path\" ]; then echo \"MISSING required input: \$path\" >&2; missing=1; else actual=\$(hash_input \"\$path\"); echo \"\$actual  \$path\"; if [ -n \"\$expected\" ] && [ \"\$actual\" != \"\$expected\" ]; then echo \"SHA-256 mismatch: \$path (expected \$expected, got \$actual)\" >&2; missing=1; fi; fi;"
  done
  remote_cmd+=' exit $missing'

  log "preflight[${POD_IDS[$idx]}]: required gate/benchmark fixtures and pinned bootloader"
  pssh "$idx" "$remote_cmd"
}

# ---------------------------------------------------------------------------
# (a) Provenance capture (shared across the whole fleet run)
# ---------------------------------------------------------------------------
git_rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }
git_dirty() {
  local repo="$1"
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then echo "NOGIT"; return; fi
  if git -C "$repo" diff --quiet HEAD 2>/dev/null; then echo "clean"; return; fi
  git -C "$repo" diff HEAD 2>/dev/null | sha256_stream | cut -c1-16
}

# ---------------------------------------------------------------------------
# Scratch path helpers (deterministic from STAMP + sanitized pod id + stage)
# ---------------------------------------------------------------------------
podtag() { echo "${STAMP}.$(echo "${POD_IDS[$1]}" | tr -c 'A-Za-z0-9_.-' '_').$2"; }

# ---------------------------------------------------------------------------
# Launch one detached gpu_bench run on a pod (no polling here). Mirrors the loop's
# detached launcher: setsid+nohup so it survives ssh close; records its own pgid,
# the gpu_bench pid, and an rc sentinel; STWO_JIT_LOG always on; BENCH_ENV verbatim.
# ---------------------------------------------------------------------------
launch_pod() {
  local idx="$1" stage="$2" args="$3"
  local tag; tag="$(podtag "$idx" "$stage")"
  local pod_sh="${POD_RUN_DIR}/${tag}.sh"
  local pod_out="${POD_RUN_DIR}/${tag}.out"
  local pod_err="${POD_RUN_DIR}/${tag}.err"
  local pod_rc="${POD_RUN_DIR}/${tag}.rc"
  local pod_pid="${POD_RUN_DIR}/${tag}.pid"
  local pod_pgid="${POD_RUN_DIR}/${tag}.pgid"

  log "launch[${POD_IDS[$idx]}] ${stage}: gpu_bench ${args}${BENCH_ENV:+  [env: ${BENCH_ENV}]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    dry "launch on pod ${POD_IDS[$idx]}: ${pod_sh} -> ${pod_out}"
    return 0
  fi

  pssh "$idx" "mkdir -p '${POD_RUN_DIR}'"
  pssh "$idx" "cat > '${pod_sh}'" <<EOF
#!/usr/bin/env bash
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
export STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}'
echo \$\$ > '${pod_pgid}'
./${BIN} ${args} > '${pod_out}' 2> '${pod_err}' &
GB_PID=\$!
echo \$GB_PID > '${pod_pid}'
wait \$GB_PID
echo \$? > '${pod_rc}'
EOF
  pssh "$idx" "rm -f '${pod_rc}' '${pod_pid}' '${pod_pgid}'; \
               nohup setsid bash '${pod_sh}' >/dev/null 2>&1 & echo LAUNCHED"
}

# ---------------------------------------------------------------------------
# Poll ALL launched pods for one stage concurrently (one ssh round-trip per pod per
# sweep). Per-pod stall detection: (stderr size, GPU util) frozen for STALL_SECS =>
# capture evidence, kill the pod's process group, mark it STALLED. MAX_WAIT backstop.
# Consumes: the "active" idx list in ACTIVE[]. Sets POD_RC/POD_STALL per pod.
# ---------------------------------------------------------------------------
poll_stage() {
  local stage="$1"; shift
  local -a active=("$@")
  local -a finished sig_at last_sig
  local idx tag pod_rc pod_err pod_pid pod_pgid st err_size gpu_util sig now waited=0
  local now0; now0="$(date +%s)"
  for idx in "${active[@]}"; do finished[$idx]=0; sig_at[$idx]="$now0"; last_sig[$idx]="__init__"; done

  # DRY_RUN: nothing is really running — resolve immediately (stall simulated at fetch).
  if [[ "$DRY_RUN" == "1" ]]; then
    for idx in "${active[@]}"; do dry "poll[${POD_IDS[$idx]}] ${stage}: (fabricated complete)"; done
    return 0
  fi

  while true; do
    local remaining=0
    for idx in "${active[@]}"; do
      [[ "${finished[$idx]}" == "1" ]] && continue
      remaining=1
      tag="$(podtag "$idx" "$stage")"
      pod_rc="${POD_RUN_DIR}/${tag}.rc"
      pod_err="${POD_RUN_DIR}/${tag}.err"
      pod_pid="${POD_RUN_DIR}/${tag}.pid"
      pod_pgid="${POD_RUN_DIR}/${tag}.pgid"
      st="$(pssh "$idx" "if [ -f '${pod_rc}' ]; then echo DONE; fi; \
             echo SIZE=\$(stat -c %s '${pod_err}' 2>/dev/null || echo 0); \
             echo GPU=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); true" || true)"
      if printf '%s\n' "$st" | grep -q '^DONE$'; then finished[$idx]=1; continue; fi

      err_size="$(printf '%s\n' "$st" | sed -n 's/^SIZE=//p' | head -1)"
      gpu_util="$(printf '%s\n' "$st" | sed -n 's/^GPU=//p' | head -1)"
      now="$(date +%s)"
      sig="${err_size}:${gpu_util}"
      if [[ "$sig" != "${last_sig[$idx]}" ]]; then last_sig[$idx]="$sig"; sig_at[$idx]="$now"; fi

      if (( now - sig_at[$idx] >= STALL_SECS )); then
        warn "pod '${POD_IDS[$idx]}' STALLED in ${stage}: stderr size + GPU util frozen for ${STALL_SECS}s."
        local base="${RESULTS_DIR}/${tag}" stall_file
        stall_file="${RESULTS_DIR}/${tag}.stall.txt"
        pssh "$idx" "pid=\$(cat '${pod_pid}' 2>/dev/null); echo \"gpu_bench pid: \${pid:-unknown}\"; \
                 if [ -n \"\$pid\" ] && [ -d \"/proc/\$pid\" ]; then \
                   for t in /proc/\$pid/task/*/status; do echo \"--- \$t\"; \
                     grep -E '^(Name|State)' \"\$t\" 2>/dev/null; done; \
                 else echo '(process directory already gone)'; fi; \
                 echo '--- last 30 stderr lines ---'; tail -n 30 '${pod_err}' 2>/dev/null; true" \
          > "$stall_file" || true
        pssh "$idx" "pgid=\$(cat '${pod_pgid}' 2>/dev/null); \
                 if [ -n \"\$pgid\" ]; then kill -TERM -\"\$pgid\" 2>/dev/null; sleep 3; \
                 kill -KILL -\"\$pgid\" 2>/dev/null; fi; true" || true
        POD_RC[$idx]="STALLED"; POD_STALL[$idx]="$stall_file"
        finished[$idx]=1
        log "stall evidence for '${POD_IDS[$idx]}' written to ${stall_file}"
      fi
    done
    (( remaining == 0 )) && break
    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "fleet stage '${stage}' exceeded MAX_WAIT (${MAX_WAIT}s) — collecting partial output."
      break
    fi
    sleep "$POLL_INTERVAL"
  done
}

# ---------------------------------------------------------------------------
# Fetch one pod's stage output; set POD_OUT[idx] and POD_RC[idx] (unless STALLED,
# already set by the poller). DRY_RUN fabricates the record so the whole path runs.
# ---------------------------------------------------------------------------
fetch_pod() {
  local idx="$1" stage="$2" args="$3"
  local tag; tag="$(podtag "$idx" "$stage")"
  local base="${RESULTS_DIR}/${tag}"
  local pod_out="${POD_RUN_DIR}/${tag}.out"
  local pod_err="${POD_RUN_DIR}/${tag}.err"
  local pod_rc="${POD_RUN_DIR}/${tag}.rc"
  POD_OUT[$idx]="${base}.out"

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$stage" == "bench" && "${FAKE_STALL:-}" == "${POD_IDS[$idx]}" ]]; then
      dry "simulating STALL for pod '${POD_IDS[$idx]}'"
      { echo "gpu_bench pid: 4242"; echo "State: S (sleeping)"; \
        echo "--- last 30 stderr lines ---"; echo "rep=0 (no further output for ${STALL_SECS}s)"; \
      } > "${base}.stall.txt"
      : > "${base}.out"
      POD_RC[$idx]="STALLED"; POD_STALL[$idx]="${base}.stall.txt"
      return 0
    fi
    synth_pod_out "$idx" "$stage" "$args" "${base}.out"
    POD_RC[$idx]=0
    return 0
  fi

  pssh "$idx" "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
  pssh "$idx" "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
  if [[ "${POD_RC[$idx]}" == "STALLED" ]]; then return 0; fi
  local rc; rc="$(pssh "$idx" "cat '${pod_rc}' 2>/dev/null" || echo TIMEOUT)"
  [[ -n "$rc" ]] || rc="TIMEOUT"
  POD_RC[$idx]="$rc"
}

# A stale pre-contract binary silently ignores unknown CLI flags. Exit status alone
# therefore cannot prove that all gate repetitions were verified and compared.
gate_contract_ok() {
  python3 - "$1" <<'PY'
import json
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
    "verified_reps": 2,
    "proof_comparison_applicable": True,
    "proof_byte_equal_required": True,
    "proof_byte_equal": True,
}
if record is None or any(record.get(key) != value for key, value in required.items()):
    print(f"gate contract missing or false: expected {required}, got {record}", file=sys.stderr)
    raise SystemExit(1)
PY
}

architecture_contract_ok() {
  python3 "$ARCHITECTURE_CHECK" "$1" --runtime-mode "$GPU_PCS_RUNTIME_MODE"
}

# ---------------------------------------------------------------------------
# Synthetic per-pod output for DRY_RUN (main record + pipeline record for the
# rotate stream, so aggregation has real numbers to chew on offline).
# ---------------------------------------------------------------------------
synth_pod_out() {
  local idx="$1" stage="$2" args="$3" dest="$4"
  local id="${POD_IDS[$idx]}"
  local seed=$(( $(printf '%s' "$id" | cksum | cut -d' ' -f1) % 60 ))
  local sum; sum="$(awk -v s="$seed" 'BEGIN{printf "%.3f", 1.4 + s/60.0}')"   # ~1.4–2.4
  local runtime_report="DetachedEager"
  [[ "$GPU_PCS_RUNTIME_MODE" == "arena-graph" ]] && runtime_report="ArenaGraph"
  local stage_counts='{"OodsEvaluation":1,"QuotientAndCompaction":1,"FriCommitAndFold":1,"ProofOfWork":1,"FriQueryAndDecommit":1,"TreeDecommit":1,"Assembly":1}'
  local architecture_fields='"engine":"gpu-native","gpu_pcs_driver_architecture":"cuda-typed-pcs-driver-v1","gpu_pcs_runtime_mode":"'"$runtime_report"'","gpu_pcs_stage_started":'"$stage_counts"',"gpu_pcs_stage_finished":'"$stage_counts"',"gpu_pcs_batched_tree_decommit":true,"gpu_pcs_driver_complete":true,"gpu_native_architecture_required":true,"gpu_pcs_required_runtime_mode":"'"$GPU_PCS_RUNTIME_MODE"'","gpu_native_architecture_gate_passed":true,"gpu_aot_loads":2,"gpu_aot_cache_hits":5,"gpu_aot_manifest_hash":49370,"gpu_aot_misses":0,"gpu_aot_runtime_loads":0,"gpu_aot_runtime_cache_hits":0,"gpu_aot_strict_rejections":0,"gpu_aot_provenance_gate_passed":true'
  if [[ "$stage" == "gate" ]]; then
    echo "{\"program\":\"gate_10t\",\"backend\":\"cuda\",${architecture_fields},\"verify_ms\":41.0,\"verified_reps\":2,\"proof_kb\":2897.5,\"proof_comparison_applicable\":true,\"proof_byte_equal\":true,\"proof_byte_equal_required\":true}" > "$dest"
    return 0
  fi
  {
    echo "{\"rep\":0,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":17250.0}}}"
    echo "{\"program\":\"SN_PIE_2.zip\",\"backend\":\"cuda\",${architecture_fields},\"n\":1,\"cycle_count\":7980000,\"pie_n_steps\":7706864,\"prove_s_warm\":31.6,\"prove_s_warm_median\":33.2,\"verify_ms\":24.0,\"proof_kb\":3006.6,\"peak_rss_gb\":29.0,\"vram_peak_gb\":36.2,\"mhz\":0.244,\"mhz_median\":0.232,\"useful_mhz\":${sum},\"useful_mhz_median\":$(awk -v m="$sum" 'BEGIN{printf "%.3f", m*0.95}'),\"security_bits\":96,\"n_queries\":70,\"pow_bits\":26,\"gpu\":\"${POD_GPUS[$idx]}\"}"
    echo "{\"pipeline\":${FLEET_DEPTH},\"producers\":${FLEET_PRODUCERS},\"pie_mode\":\"rotate\",\"reps\":${FLEET_REPS},\"total_s\":202.8,\"feed_starved_s\":0.0,\"sustained_steps_per_s\":$(awk -v m="$sum" 'BEGIN{printf "%.1f", m*1e6}'),\"sustained_mhz\":$(awk -v m="$sum" 'BEGIN{printf "%.3f", m*1.03}'),\"sustained_useful_mhz\":${sum}}"
  } > "$dest"
}

# ---------------------------------------------------------------------------
# Run one stage across every ACTIVE pod concurrently: launch all, poll all, fetch
# all, then set POD_STATUS from the rc. `fail_status` is what a nonzero rc becomes.
# Pods already out of "pending" (dropped earlier) are skipped.
# ---------------------------------------------------------------------------
run_stage() {
  local stage="$1" args="$2" fail_status="$3"
  local -a active=()
  local idx
  for idx in "${!POD_IDS[@]}"; do
    [[ "${POD_STATUS[$idx]}" == "pending" || "${POD_STATUS[$idx]}" == "ok" ]] || continue
    active+=("$idx")
  done
  [[ ${#active[@]} -gt 0 ]] || die "no active pods left for stage '${stage}'"

  log "=== stage '${stage}' on ${#active[@]} pod(s) concurrently ==="
  for idx in "${active[@]}"; do launch_pod "$idx" "$stage" "$args"; done
  poll_stage "$stage" "${active[@]}"
  for idx in "${active[@]}"; do
    fetch_pod "$idx" "$stage" "$args"
    if [[ "$stage" == "gate" && "${POD_RC[$idx]}" == "0" ]] \
       && ! gate_contract_ok "${POD_OUT[$idx]}"; then
      POD_RC[$idx]="GATE_CONTRACT"
    fi
    if [[ "${POD_RC[$idx]}" == "0" && "$args" == *"--require-gpu-native-architecture"* ]] \
       && ! architecture_contract_ok "${POD_OUT[$idx]}"; then
      POD_RC[$idx]="ARCHITECTURE_CONTRACT"
    fi
    case "${POD_RC[$idx]}" in
      0)       POD_STATUS[$idx]="ok" ;;
      STALLED) POD_STATUS[$idx]="stalled";
               warn "pod '${POD_IDS[$idx]}' stalled in ${stage} — dropped from the fleet aggregate." ;;
      *)       POD_STATUS[$idx]="$fail_status";
               warn "pod '${POD_IDS[$idx]}' ${fail_status} in ${stage} (rc=${POD_RC[$idx]}) — dropped." ;;
    esac
  done
}

# ===========================================================================
# Orchestration
# ===========================================================================
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS_DIR"

parse_roster
PIE_LIST="$(build_pie_list)"

STWO_REV="$(git_rev "$STWO_LOCAL")";  STWO_DIRTY="$(git_dirty "$STWO_LOCAL")"
CAIRO_REV="$(git_rev "$CAIRO_LOCAL")"; CAIRO_DIRTY="$(git_dirty "$CAIRO_LOCAL")"

log "=== fleet start: ${#POD_IDS[@]} pod(s), pies=${PIES}, reps=${FLEET_REPS} depth=${FLEET_DEPTH} producers=${FLEET_PRODUCERS}, dry=${DRY_RUN} ==="
log "provenance: stwo=${STWO_REV:0:12} dirty=${STWO_DIRTY} | cairo=${CAIRO_REV:0:12} dirty=${CAIRO_DIRTY}"
[[ -n "$BENCH_ENV" ]] && log "bench_env: ${BENCH_ENV} (recorded in the report; not comparable with clean runs)"
log "bootloader: ${POD_BOOTLOADER_JSON} (pinned, exported for every launch)"
log "GPU-native architecture gate: required mode=${GPU_PCS_RUNTIME_MODE}"
[[ "$SKIP_GATE" == "1" ]] && warn "--skip-gate: correctness gate DISABLED — report will be marked UNGATED."

# (b) Optional prep: reuse bench_loop.sh --gate-only per pod (sync + build + gate).
if [[ "$PREP" == "1" ]]; then
  [[ -x "$BENCH_LOOP" ]] || die "--prep needs an executable ${BENCH_LOOP}"
  log "=== prep: bench_loop.sh --gate-only per pod (sync+build+gate) ==="
  for idx in "${!POD_IDS[@]}"; do
    log "prep[${POD_IDS[$idx]}] ..."
    if BENCH_POD_ID="${POD_IDS[$idx]}" DRY_RUN="$DRY_RUN" BENCH_ENV="$BENCH_ENV" \
         GPU_PCS_RUNTIME_MODE="$GPU_PCS_RUNTIME_MODE" \
         POD_BOOTLOADER_JSON="$POD_BOOTLOADER_JSON" \
         "$BENCH_LOOP" --gate-only >&2; then
      log "prep[${POD_IDS[$idx]}] OK"
    else
      warn "prep[${POD_IDS[$idx]}] FAILED — dropping pod from the fleet."
      POD_STATUS[$idx]="gate_failed"
    fi
  done
fi

# (c) Resolve every still-active pod (host/port/key).
for idx in "${!POD_IDS[@]}"; do
  [[ "${POD_STATUS[$idx]}" == "pending" || "${POD_STATUS[$idx]}" == "ok" ]] || continue
  if ! resolve_pod "$idx"; then
    warn "pod '${POD_IDS[$idx]}' unresolved — dropping from the fleet."
    POD_STATUS[$idx]="run_failed"
  fi
done

for idx in "${!POD_IDS[@]}"; do
  [[ "${POD_STATUS[$idx]}" == "pending" || "${POD_STATUS[$idx]}" == "ok" ]] || continue
  preflight_pod_inputs "$idx" \
    || die "required input missing on pod '${POD_IDS[$idx]}'; seed it explicitly or update GATE_PIE/--pies"
done

# (d) Correctness GATE on every pod concurrently (unless --skip-gate or --prep already
#     gated). Gate-first discipline: perf is never reported for a pod that failed here.
GATED="gated"
if [[ "$SKIP_GATE" == "1" ]]; then
  GATED="ungated"
elif [[ "$PREP" == "1" ]]; then
  GATED="gated(prep)"
  log "=== gate: already run by --prep; skipping the standalone gate stage ==="
else
  GATE_ARGS="--pie ${POD_GATE_PIE} --backend cuda ${GPU_NATIVE_ARGS} --reps 2 --reuse-input --require-proof-byte-equal"
  run_stage "gate" "$GATE_ARGS" "gate_failed"
fi

# (e) Benchmark: one rotate-mode pipelined stream per pod, all pods at once.
BENCH_ARGS="--pie ${PIE_LIST} --backend cuda ${GPU_NATIVE_ARGS} --reps ${FLEET_REPS} --pipeline ${FLEET_DEPTH} --producers ${FLEET_PRODUCERS} --pie-mode rotate"
# Reset survivors to "pending" so run_stage re-activates exactly the gate survivors.
for idx in "${!POD_IDS[@]}"; do [[ "${POD_STATUS[$idx]}" == "ok" ]] && POD_STATUS[$idx]="pending"; done
run_stage "bench" "$BENCH_ARGS" "run_failed"

# For any pod that stalled, record a self-documenting ledger-style entry is implicit in
# the report (status + stall file path). Continue to aggregation regardless.

# ---------------------------------------------------------------------------
# (g)+(h) Aggregate + report. A manifest (one tab-separated line per pod) hands the
# per-pod metadata + output file to python, which parses each pod's pipeline record,
# sums the fleet, writes fleet_report.json, and prints the human table.
# ---------------------------------------------------------------------------
MANIFEST="${RESULTS_DIR}/${STAMP}.manifest.tsv"
: > "$MANIFEST"
for idx in "${!POD_IDS[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${POD_IDS[$idx]}" "${POD_GPUS[$idx]}" "${POD_USD[$idx]}" \
    "${POD_STATUS[$idx]}" "${POD_OUT[$idx]:-}" "${POD_STALL[$idx]:-}" >> "$MANIFEST"
done

FL_TS="$TS" FL_STWO_REV="$STWO_REV" FL_STWO_DIRTY="$STWO_DIRTY" \
FL_CAIRO_REV="$CAIRO_REV" FL_CAIRO_DIRTY="$CAIRO_DIRTY" \
FL_MANIFEST="$MANIFEST" FL_REPORT="$REPORT" FL_BENCH_ENV="$BENCH_ENV" \
FL_GPU_PCS_RUNTIME_MODE="$GPU_PCS_RUNTIME_MODE" \
FL_PIES="$PIES" FL_REPS="$FLEET_REPS" FL_DEPTH="$FLEET_DEPTH" FL_PRODUCERS="$FLEET_PRODUCERS" \
FL_GATED="$GATED" FL_TARGET_LO="$TARGET_LO_MHZ" FL_TARGET_HI="$TARGET_HI_MHZ" \
python3 - <<'PY'
import json, os

manifest = os.environ["FL_MANIFEST"]
report   = os.environ["FL_REPORT"]

def parse_out(path):
    """Return (main_record, pipeline_record) from a gpu_bench stdout file."""
    rec, pipe = None, None
    if not path or not os.path.exists(path):
        return rec, pipe
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pipeline" in obj:
                pipe = obj
            elif "program" in obj and "backend" in obj:
                rec = obj
    return rec, pipe

def pod_mhz(rec, pipe):
    """Per-pod claim MHz: sustained pipeline rate, else fixed-statement median."""
    if pipe and pipe.get("sustained_useful_mhz") is not None:
        return pipe["sustained_useful_mhz"], "sustained_useful_mhz"
    if rec and rec.get("useful_mhz_median") is not None:
        return rec.get("useful_mhz_median"), "useful_mhz_median"
    return None, None

pods = []
with open(manifest) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        while len(parts) < 6:
            parts.append("")
        pid, gpu, usd, status, out, stall = parts[:6]
        rec, pipe = parse_out(out)
        mhz, basis = pod_mhz(rec, pipe)
        try:
            usd_f = float(usd)
        except ValueError:
            usd_f = None
        entry = {
            "id": pid, "gpu": gpu, "usd_per_hr": usd_f, "status": status,
            "useful_mhz": mhz, "mhz_basis": basis,
            "feed_starved_s": (pipe or {}).get("feed_starved_s"),
            "vram_peak_gb": (rec or {}).get("vram_peak_gb"),
            "gpu_pcs_driver_architecture": (rec or {}).get("gpu_pcs_driver_architecture"),
            "gpu_pcs_runtime_mode": (rec or {}).get("gpu_pcs_runtime_mode"),
            "gpu_native_architecture_gate_passed": (rec or {}).get("gpu_native_architecture_gate_passed"),
            "gpu_aot_loads": (rec or {}).get("gpu_aot_loads"),
            "gpu_aot_cache_hits": (rec or {}).get("gpu_aot_cache_hits"),
            "gpu_aot_manifest_hash": (rec or {}).get("gpu_aot_manifest_hash"),
            "gpu_aot_provenance_gate_passed": (rec or {}).get("gpu_aot_provenance_gate_passed"),
            "usd_per_mhz_hr": (round(usd_f / mhz, 4) if (usd_f and mhz) else None),
            "out_file": out or None,
        }
        if status == "stalled" and stall:
            entry["stall_file"] = stall
        pods.append(entry)

ok = [p for p in pods if p["status"] == "ok" and p["useful_mhz"] is not None]
agg_mhz = round(sum(p["useful_mhz"] for p in ok), 3) if ok else 0.0
# Only pods that actually contributed a number count toward $/hr for $/MHz-hr honesty.
agg_usd = round(sum(p["usd_per_hr"] for p in ok if p["usd_per_hr"] is not None), 4) if ok else 0.0
usd_per_mhz_hr = round(agg_usd / agg_mhz, 4) if agg_mhz else None
mean_pod_mhz = (agg_mhz / len(ok)) if ok else None

def pods_needed(target):
    if not mean_pod_mhz:
        return None
    import math
    return math.ceil(target / mean_pod_mhz)

target_lo = float(os.environ["FL_TARGET_LO"])
target_hi = float(os.environ["FL_TARGET_HI"])

aggregate = {
    "n_pods_total": len(pods),
    "n_pods_ok": len(ok),
    "aggregate_useful_mhz": agg_mhz,
    "total_usd_per_hr": agg_usd,
    "usd_per_mhz_hr": usd_per_mhz_hr,
    "mean_pod_useful_mhz": (round(mean_pod_mhz, 3) if mean_pod_mhz else None),
    "target_lo_mhz": target_lo,
    "target_hi_mhz": target_hi,
    "meets_target_lo": (agg_mhz >= target_lo),
    "meets_target_hi": (agg_mhz >= target_hi),
    "pods_needed_for_target_lo": pods_needed(target_lo),
    "pods_needed_for_target_hi": pods_needed(target_hi),
}

report_obj = {
    "ts": os.environ["FL_TS"],
    "stwo_rev": os.environ["FL_STWO_REV"], "stwo_dirty": os.environ["FL_STWO_DIRTY"],
    "cairo_rev": os.environ["FL_CAIRO_REV"], "cairo_dirty": os.environ["FL_CAIRO_DIRTY"],
    "bench_env": os.environ.get("FL_BENCH_ENV", ""),
    "gpu_pcs_runtime_mode_required": os.environ["FL_GPU_PCS_RUNTIME_MODE"],
    "gated": os.environ["FL_GATED"],
    "pies": os.environ["FL_PIES"],
    "reps": int(os.environ["FL_REPS"]),
    "pipeline_depth": int(os.environ["FL_DEPTH"]),
    "producers": int(os.environ["FL_PRODUCERS"]),
    "pie_mode": "rotate",
    "pods": pods,
    "aggregate": aggregate,
}
with open(report, "w") as f:
    json.dump(report_obj, f, indent=2)
    f.write("\n")

# --- human table (stdout) ---
def fmt(v, w, prec=None):
    if v is None:
        s = "—"
    elif prec is not None:
        s = f"{v:.{prec}f}"
    else:
        s = str(v)
    return f"{s:<{w}}"

print(f"=== fleet report ({report_obj['ts']}) ===")
print(f"    revs: stwo={report_obj['stwo_rev'][:8]}{'*' if report_obj['stwo_dirty'] not in ('clean','NOGIT') else ''}"
      f" cairo={report_obj['cairo_rev'][:8]}{'*' if report_obj['cairo_dirty'] not in ('clean','NOGIT') else ''}"
      f"  gated={report_obj['gated']}"
      + (f"  bench_env(!): {report_obj['bench_env']}" if report_obj['bench_env'] else ""))
print(f"    stream: pie_mode=rotate pies={report_obj['pies']} reps={report_obj['reps']}"
      f" depth={report_obj['pipeline_depth']} producers={report_obj['producers']}")
print()
hdr = ("pod_id", "gpu", "status", "useful_mhz", "$/hr", "$/MHz-hr", "vram_gb")
print(f"  {hdr[0]:<16}{hdr[1]:<12}{hdr[2]:<12}{hdr[3]:<12}{hdr[4]:<8}{hdr[5]:<10}{hdr[6]:<8}")
print("  " + "-" * 76)
for p in pods:
    print("  "
          + fmt(p["id"], 16)
          + fmt(p["gpu"], 12)
          + fmt(p["status"], 12)
          + fmt(p["useful_mhz"], 12, 3)
          + fmt(p["usd_per_hr"], 8, 2)
          + fmt(p["usd_per_mhz_hr"], 10, 4)
          + fmt(p["vram_peak_gb"], 8, 1))
print("  " + "-" * 76)
a = aggregate
print(f"  AGGREGATE  useful_mhz={a['aggregate_useful_mhz']:.3f}"
      f"  total_$/hr={a['total_usd_per_hr']:.2f}"
      + (f"  $/MHz-hr={a['usd_per_mhz_hr']:.4f}" if a['usd_per_mhz_hr'] is not None else "  $/MHz-hr=—")
      + f"  ({a['n_pods_ok']}/{a['n_pods_total']} pods ok)")
lo, hi = a["target_lo_mhz"], a["target_hi_mhz"]
print(f"  TARGET     {lo:.0f} MHz: {'MET' if a['meets_target_lo'] else 'not met'}"
      + (f" (need ~{a['pods_needed_for_target_lo']} such pods)" if a['pods_needed_for_target_lo'] else "")
      + f"   |   {hi:.0f} MHz: {'MET' if a['meets_target_hi'] else 'not met'}"
      + (f" (need ~{a['pods_needed_for_target_hi']} such pods)" if a['pods_needed_for_target_hi'] else ""))
print()
print(f"=== fleet_report.json: {report} ===")
PY

# Overall exit status: fail loudly if NO pod produced a usable number.
OK_COUNT=0
for idx in "${!POD_IDS[@]}"; do [[ "${POD_STATUS[$idx]}" == "ok" ]] && OK_COUNT=$(( OK_COUNT + 1 )); done
log "=== fleet done: ${OK_COUNT}/${#POD_IDS[@]} pod(s) ok ==="
[[ "$OK_COUNT" -gt 0 ]] || die "no pod produced a usable benchmark record (see ${REPORT})"
exit 0
