#!/usr/bin/env bash
#
# perf_gates.sh — one command per opt-in CUDA perf lane, runnable the moment
# bench_loop.sh's whole-proof correctness gate goes green.
#
# For each lane flag (default: STWO_CUDA_RELATION_FUSED, STWO_CUDA_RELATION_SCAN_TAIL,
# STWO_CUDA_BLAKE2S_LEAF_ILP; extra flag names accepted as positional arguments):
#
#   (a) A/B pair of SN_PIE_2 gpu_bench runs on the pod — baseline env vs
#       <FLAG>=1 — each `--reps 4 --reuse-input --require-proof-byte-equal`,
#       launched with the SAME detached nohup+setsid + rc-sentinel + stall-
#       detection pattern as bench_loop.sh. Each arm dumps its rep-0 proof
#       (STWO_DUMP_PROOF) and the two proofs' sha256 are compared ON THE POD:
#       a flagged proof that differs from baseline FAILS CLOSED (ledger entry
#       status "proof_mismatch", nonzero exit) — every lane is documented
#       byte-identical, so a mismatch is a soundness alarm, not a perf note.
#   (b) useful_mhz_median (the claim metric, warm-sample median) + per-phase
#       total_ms deltas appended as one JSON line per lane to
#       loop/perf_gates.jsonl (same provenance fields as bench_loop's ledger:
#       revs, dirty hashes, pod_gpu, bench_env).
#   (c) optionally (--ncu) a targeted Nsight Compute profile of the lane's
#       kernels with the flag ON. No literal ncu command is documented in-tree;
#       this invocation encodes the documented constraints
#       (GPU_RESIDENT_PROVER_DESIGN.md §11d.1: pod ships ncu 2022.3 / CUDA 11.8
#       only, full-prove capture is impractical, so warm-cache TARGETED capture:
#       kernel-name regex + launch-count cap). Default kernel regex per lane:
#       relation_fused | relation_scan | stream_leaf_update (blake2s.cu /
#       relation_*.cu kernel names). The A/B runs in (a) warm the on-disk JIT
#       cache first, as §11d.1 requires.
#
# --parity-only skips benchmarking entirely and just reruns the two counted
# native differential suites (prepared_relation_native, prepared_commit_native
# — the gates run_cuda_soundness_gate.py counts) on the pod stwo checkout with
# each lane flag exported, failing closed unless BOTH suites exit 0 AND report
# a nonzero executed-test count (cfg-gated tests can pass while running nothing).
#
# This script never syncs or builds: it measures the binary the green gate just
# certified. Run `./bench_loop.sh --gate-only` (or a full loop) first.
#
# Usage:
#   ./perf_gates.sh [--lanes CSV] [--reps N] [--ncu] [--ncu-kernels REGEX]
#                   [--parity-only] [--help] [EXTRA_FLAG ...]
#
# One command per lane (the runbook lines):
#   ./perf_gates.sh --lanes STWO_CUDA_RELATION_FUSED
#   ./perf_gates.sh --lanes STWO_CUDA_RELATION_SCAN_TAIL
#   ./perf_gates.sh --lanes STWO_CUDA_BLAKE2S_LEAF_ILP
#   ./perf_gates.sh --lanes SOME_NEW_FLAG            # or: ./perf_gates.sh SOME_NEW_FLAG
#
# Flags:
#   --lanes CSV       Replace the default lane set with a comma-separated list.
#   --reps N          Reps per arm (default 4; >=2 so in-run byte-equal is real).
#   --ncu             After a lane's A/B pair passes, capture a targeted ncu
#                     profile of that lane's kernels (flag ON). Report is
#                     fetched to loop/results/ and its pod path recorded.
#   --ncu-kernels RE  Override the lane's default kernel-name regex.
#   --parity-only     Only rerun prepared_relation_native + prepared_commit_native
#                     with each lane flag exported. No benchmarks, no ledger
#                     perf entry (a parity entry is still appended).
#   EXTRA_FLAG ...    Additional lane env-flag names appended to the lane set.
#
# Environment (same plumbing as bench_loop.sh):
#   BENCH_POD_ID   Pod id (overrides POD_ID in loop/pod.conf).
#   BENCH_ENV      "K=V K=V ..." exported into EVERY run of BOTH arms (recorded
#                  per ledger entry). Lane flags must not appear here.
#   DRY_RUN=1      Echo every ssh instead of executing; fabricate run output so
#                  the A/B -> proof-compare -> ledger path runs for real offline.
#   FAKE_PROOF_MISMATCH  (DRY_RUN only) lane name whose flagged proof is
#                  fabricated DIFFERENT, to exercise the fail-closed path.
#   POLL_INTERVAL / STALL_SECS / MAX_WAIT   As in bench_loop.sh.
#   NCU_LAUNCH_COUNT  Kernel launches captured per ncu profile (default 10).
#   POD_BOOTLOADER_JSON / GPU_PCS_RUNTIME_MODE   As in bench_loop.sh.
#
# NOTE: only same-pod, same-bench_env comparisons are meaningful; the A/B pair
# is always same-pod same-binary back-to-back, which is the point.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (mirrors bench_loop.sh)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
LOOP_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${RESULTS_DIR:-${LOOP_DIR}/results}"
PERF_LEDGER="${PERF_LEDGER:-${LOOP_DIR}/perf_gates.jsonl}"
POD_CONF="${POD_CONF:-${LOOP_DIR}/pod.conf}"

STWO_POD="/workspace/stwo"
CAIRO_POD="/workspace/stwo-cairo"
POD_PROVER_DIR="${CAIRO_POD}/stwo_cairo_prover"
BIN="target/release/gpu_bench"                      # relative to POD_PROVER_DIR
POD_USER="root"

POD_SN_DIR="${CAIRO_POD}/gpu_benchmarks/pie/sn"
POD_PIE="${POD_SN_DIR}/SN_PIE_2.zip"                # the lane-gate statement
POD_BOOTLOADER_JSON="${POD_BOOTLOADER_JSON:-/workspace/bench_inputs/simple_bootloader_compiled.json}"

# Scratch outside the repo tree (bench_loop's rsync never touches it).
POD_RUN_DIR="/workspace/perf_gate_runs"

RUST_MIN_STACK_VAL=4194304
BENCH_ENV="${BENCH_ENV:-}"
GPU_PCS_RUNTIME_MODE="${GPU_PCS_RUNTIME_MODE:-arena-graph}"
GPU_NATIVE_ARGS="--engine gpu-native --require-gpu-native-architecture --require-gpu-pcs-runtime-mode ${GPU_PCS_RUNTIME_MODE}"

DRY_RUN="${DRY_RUN:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
STALL_SECS="${STALL_SECS:-600}"
MAX_WAIT="${MAX_WAIT:-10800}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-10}"

# Flag defaults.
REPS="4"
DO_NCU=0
PARITY_ONLY=0
NCU_KERNELS_OVERRIDE=""
DEFAULT_LANES=(STWO_CUDA_RELATION_FUSED STWO_CUDA_RELATION_SCAN_TAIL STWO_CUDA_BLAKE2S_LEAF_ILP)
LANES=()
EXTRA_LANES=()

# Set by resolve_pod().
POD_HOST=""
POD_PORT=""
POD_KEY=""
SSH_OPTS=()

# Set by run_pod_job().
LAST_OUT=""
LAST_ERR=""
LAST_RC=""
LAST_STALL_FILE=""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log()  { echo "[perf_gates] $*" >&2; }
dry()  { echo "[DRY_RUN] $*" >&2; }
warn() { echo "[perf_gates][WARN] $*" >&2; }
die()  { echo "[perf_gates][FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,83p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------------------------------------------------------
# Pod resolution + transport (same pattern as bench_loop.sh)
# ---------------------------------------------------------------------------
resolve_pod() {
  local pod_id="" fb_host="" fb_port="" fb_key=""
  if [[ -f "$POD_CONF" ]]; then
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
}

run_ssh() {
  if [[ "$DRY_RUN" == "1" ]]; then dry "ssh: $*"; return 0; fi
  ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "$@"
}

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else LC_ALL=C shasum -a 256; fi
}

# ---------------------------------------------------------------------------
# Lane -> default ncu kernel-name regex (blake2s.cu / relation_*.cu kernels)
# ---------------------------------------------------------------------------
lane_kernel_regex() {
  case "$1" in
    STWO_CUDA_RELATION_FUSED)     echo "relation_fused" ;;
    STWO_CUDA_RELATION_SCAN_TAIL) echo "relation_scan" ;;
    STWO_CUDA_BLAKE2S_LEAF_ILP)   echo "stream_leaf_update" ;;
    *)                            echo "relation_fused|relation_scan|stream_leaf_update" ;;
  esac
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lanes)       IFS=',' read -r -a LANES <<<"${2:?--lanes needs a CSV value}"; shift 2 ;;
    --reps)        REPS="${2:?--reps needs a value}"; shift 2 ;;
    --ncu)         DO_NCU=1; shift ;;
    --ncu-kernels) NCU_KERNELS_OVERRIDE="${2:?--ncu-kernels needs a regex}"; shift 2 ;;
    --parity-only) PARITY_ONLY=1; shift ;;
    -h|--help)     usage 0 ;;
    --*)           die "unknown flag: $1 (try --help)" ;;
    *)             EXTRA_LANES+=("$1"); shift ;;
  esac
done

if [[ ${#LANES[@]} -eq 0 ]]; then LANES=("${DEFAULT_LANES[@]}"); fi
if [[ ${#EXTRA_LANES[@]} -gt 0 ]]; then LANES+=("${EXTRA_LANES[@]}"); fi

[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -ge 2 ]] \
  || die "--reps must be >= 2 so in-run proof-byte equality is a real check"

case "$GPU_PCS_RUNTIME_MODE" in
  detached-eager|arena-graph) ;;
  *) die "GPU_PCS_RUNTIME_MODE must be detached-eager or arena-graph (got '$GPU_PCS_RUNTIME_MODE')" ;;
esac

for lane in "${LANES[@]}"; do
  [[ "$lane" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "lane '$lane' is not a valid env flag name"
done

# Validate BENCH_ENV shape and keep lane flags out of it (an arm's identity
# must come from this script, never ambient env).
if [[ -n "$BENCH_ENV" ]]; then
  for kv in $BENCH_ENV; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] \
      || die "BENCH_ENV token '$kv' is not K=V (values must not contain spaces)"
    [[ "${kv%%=*}" != "STWO_BOOTLOADER_JSON" ]] \
      || die "STWO_BOOTLOADER_JSON is reserved; set POD_BOOTLOADER_JSON instead"
    [[ "${kv%%=*}" != "STWO_DUMP_PROOF" ]] \
      || die "STWO_DUMP_PROOF is reserved; perf_gates manages proof dumps"
    for lane in "${LANES[@]}"; do
      [[ "${kv%%=*}" != "$lane" ]] \
        || die "lane flag ${lane} must not appear in BENCH_ENV — perf_gates sets it per arm"
    done
  done
fi

resolve_pod

# ---------------------------------------------------------------------------
# Provenance (same fields as bench_loop's ledger)
# ---------------------------------------------------------------------------
git_rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }
git_dirty() {
  local repo="$1"
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then echo "NOGIT"; return; fi
  if git -C "$repo" diff --quiet HEAD 2>/dev/null &&
     [[ -z "$(git -C "$repo" ls-files --others --exclude-standard | head -1)" ]]; then
    echo "clean"
    return
  fi
  (
    git -C "$repo" diff --binary HEAD -- 2>/dev/null
    git -C "$repo" ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' path; do
        printf 'untracked\0%s\0' "$path"
        cat "$repo/$path"
      done
  ) | sha256_stream | cut -c1-16
}

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STWO_REV="$(git_rev "$STWO_LOCAL")"
STWO_DIRTY="$(git_dirty "$STWO_LOCAL")"
CAIRO_REV="$(git_rev "$CAIRO_LOCAL")"
CAIRO_DIRTY="$(git_dirty "$CAIRO_LOCAL")"

mkdir -p "$RESULTS_DIR"

log "provenance: stwo=${STWO_REV:0:12} dirty=${STWO_DIRTY} | cairo=${CAIRO_REV:0:12} dirty=${CAIRO_DIRTY}"
[[ -n "$BENCH_ENV" ]] && log "bench_env: ${BENCH_ENV} (recorded per entry; NOT comparable with clean runs)"
log "lanes: ${LANES[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  POD_GPU="DRY-RUN-GPU"
else
  POD_GPU="$(run_ssh "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" || true)"
  [[ -n "$POD_GPU" ]] || POD_GPU="unknown"
fi
log "pod GPU: ${POD_GPU}"

# ---------------------------------------------------------------------------
# DRY_RUN synthetic gpu_bench output (exercises the A/B -> ledger path offline)
# ---------------------------------------------------------------------------
synth_out() {
  # $1 name  $2 destfile — deterministic mock numbers keyed off the name so the
  # flagged arm's delta is nonzero.
  local name="$1" dest="$2"
  local seed=$(( $(printf '%s' "$name" | cksum | cut -d' ' -f1) % 40 ))
  local um_median
  um_median="$(awk -v s="$seed" 'BEGIN{printf "%.3f", 1.3 + s/100.0}')"
  {
    echo "{\"rep\":0,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":$((1200 + seed * 3)).5},\"fri\":{\"count\":1,\"total_ms\":$((560 + seed)).1}}}"
    echo "{\"rep\":1,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":$((1190 + seed * 3)).2},\"fri\":{\"count\":1,\"total_ms\":$((555 + seed)).8}}}"
    echo "{\"program\":\"SN_PIE_2.zip\",\"backend\":\"cuda\",\"engine\":\"gpu-native\",\"verified_reps\":${REPS},\"proof_byte_equal\":true,\"proof_byte_equal_required\":true,\"proof_comparison_applicable\":true,\"useful_mhz_median\":${um_median},\"vram_peak_gb\":11.3,\"gpu\":\"${POD_GPU}\"}"
  } > "$dest"
}

# ---------------------------------------------------------------------------
# Detached pod job: nohup+setsid launcher + rc sentinel + stall detection
# (same pattern as bench_loop.sh run_bench). $1 job name, $2 launcher body
# (runs on the pod; must NOT write the rc file — the wrapper does).
# Sets LAST_OUT/LAST_ERR (local copies), LAST_RC (0|code|TIMEOUT|STALLED),
# LAST_STALL_FILE.
# ---------------------------------------------------------------------------
run_pod_job() {
  local name="$1" body="$2"
  local base="${RESULTS_DIR}/${STAMP}.${name}"
  local pod_out="${POD_RUN_DIR}/${STAMP}.${name}.out"
  local pod_err="${POD_RUN_DIR}/${STAMP}.${name}.err"
  local pod_rc="${POD_RUN_DIR}/${STAMP}.${name}.rc"
  local pod_sh="${POD_RUN_DIR}/${STAMP}.${name}.sh"
  local pod_pgid="${POD_RUN_DIR}/${STAMP}.${name}.pgid"
  LAST_STALL_FILE=""
  LAST_OUT="${base}.out"
  LAST_ERR="${base}.err"

  if [[ "$DRY_RUN" == "1" ]]; then
    dry "write launcher ${pod_sh}:"
    printf '%s\n' "$body" | sed 's/^/[DRY_RUN]   | /' >&2
    dry "ssh: rm -f '${pod_rc}'; nohup setsid bash '${pod_sh}' > '${pod_out}' 2> '${pod_err}' & echo LAUNCHED"
    dry "poll: cat '${pod_rc}' until present (stall: stderr size + GPU util frozen ${STALL_SECS}s)"
    : > "${base}.err"
    LAST_RC=0
    return 0
  fi

  run_ssh "mkdir -p '${POD_RUN_DIR}'"
  run_ssh "cat > '${pod_sh}'" <<EOF
#!/usr/bin/env bash
echo \$\$ > '${pod_pgid}'
${body}
echo \$? > '${pod_rc}'
EOF
  run_ssh "rm -f '${pod_rc}' '${pod_pgid}'; nohup setsid bash '${pod_sh}' > '${pod_out}' 2> '${pod_err}' & echo LAUNCHED"

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
      warn "job '${name}' STALLED: stderr size and GPU util unchanged for ${STALL_SECS}s."
      local stall_file="${base}.stall.txt"
      run_ssh "echo '--- last 30 stderr lines ---'; tail -n 30 '${pod_err}' 2>/dev/null; true" \
        > "$stall_file" || true
      run_ssh "pgid=\$(cat '${pod_pgid}' 2>/dev/null); \
               if [ -n \"\$pgid\" ]; then kill -TERM -\"\$pgid\" 2>/dev/null; sleep 3; \
               kill -KILL -\"\$pgid\" 2>/dev/null; fi; true" || true
      run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
      run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
      LAST_RC="STALLED"; LAST_STALL_FILE="$stall_file"
      log "stall evidence written to ${stall_file}"
      return 0
    fi

    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "job '${name}' exceeded MAX_WAIT (${MAX_WAIT}s). Fetching partial output."
      break
    fi
    sleep "$POLL_INTERVAL"
  done

  run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
  run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
  run_ssh "cat '${pod_rc}'  2>/dev/null" > "${base}.rc"  || true
  LAST_RC="$(cat "${base}.rc" 2>/dev/null || echo TIMEOUT)"
  [[ -n "$LAST_RC" ]] || LAST_RC="TIMEOUT"
  log "job '${name}' finished (rc=${LAST_RC})"
}

# gpu_bench launcher body for one A/B arm. $1 extra "export K=V" lines,
# $2 proof dump path, $3 gpu_bench args.
bench_body() {
  local extra_exports="$1" proof_dump="$2" args="$3"
  cat <<EOF
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
export STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}'
export STWO_DUMP_PROOF='${proof_dump}'
${extra_exports}
./${BIN} ${args}
EOF
}

# sha256 of a pod file (fabricated under DRY_RUN so the compare path runs).
pod_sha256() {
  local path="$1" fake="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "ssh: sha256sum '${path}'"
    echo "$fake"
    return 0
  fi
  run_ssh "if command -v sha256sum >/dev/null 2>&1; then sha256sum '${path}'; else shasum -a 256 '${path}'; fi" \
    | cut -d' ' -f1
}

# ---------------------------------------------------------------------------
# Ledger append + summary. status: ok | run_failed | stalled | proof_mismatch
#                                | parity_ok | parity_failed
# ---------------------------------------------------------------------------
append_perf_ledger() {
  local lane="$1" status="$2" base_out="$3" flag_out="$4" base_sha="$5" flag_sha="$6" ncu_report="$7"
  PG_TS="$TS" PG_LANE="$lane" PG_STATUS="$status" \
  PG_STWO_REV="$STWO_REV" PG_STWO_DIRTY="$STWO_DIRTY" \
  PG_CAIRO_REV="$CAIRO_REV" PG_CAIRO_DIRTY="$CAIRO_DIRTY" \
  PG_GPU="$POD_GPU" PG_BENCH_ENV="$BENCH_ENV" PG_REPS="$REPS" \
  PG_BASE_OUT="$base_out" PG_FLAG_OUT="$flag_out" \
  PG_BASE_SHA="$base_sha" PG_FLAG_SHA="$flag_sha" \
  PG_NCU="$ncu_report" PG_LEDGER="$PERF_LEDGER" python3 - <<'PY'
import json, os

def parse_lines(path):
    record, phases = None, {}
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
                    for name, tot in obj["phase_totals"].items():
                        phases[name] = phases.get(name, 0.0) + float(tot.get("total_ms", 0.0))
                elif "program" in obj and "backend" in obj:
                    record = obj
    except FileNotFoundError:
        pass
    return record, phases

base_rec, base_ph = parse_lines(os.environ.get("PG_BASE_OUT", ""))
flag_rec, flag_ph = parse_lines(os.environ.get("PG_FLAG_OUT", ""))

def metric(rec):
    return rec.get("useful_mhz_median") if rec else None

bm, fm = metric(base_rec), metric(flag_rec)
delta_pct = None
if bm and fm:
    delta_pct = (fm - bm) / bm * 100.0

phase_deltas = {}
for name in sorted(set(base_ph) | set(flag_ph)):
    b, f = base_ph.get(name), flag_ph.get(name)
    d = ((f - b) / b * 100.0) if (b and f is not None) else None
    phase_deltas[name] = {"baseline_ms": b, "flagged_ms": f,
                          "delta_pct": round(d, 2) if d is not None else None}

entry = {
    "ts": os.environ["PG_TS"],
    "lane": os.environ["PG_LANE"],
    "status": os.environ["PG_STATUS"],
    "stwo_rev": os.environ["PG_STWO_REV"],
    "stwo_dirty": os.environ["PG_STWO_DIRTY"],
    "cairo_rev": os.environ["PG_CAIRO_REV"],
    "cairo_dirty": os.environ["PG_CAIRO_DIRTY"],
    "pod_gpu": os.environ["PG_GPU"],
    "bench_env": os.environ.get("PG_BENCH_ENV", ""),
    "reps": int(os.environ["PG_REPS"]),
    "pie": "SN_PIE_2",
    "baseline": {"useful_mhz_median": bm,
                 "proof_sha256": os.environ.get("PG_BASE_SHA", ""),
                 "record": base_rec},
    "flagged": {"useful_mhz_median": fm,
                "proof_sha256": os.environ.get("PG_FLAG_SHA", ""),
                "record": flag_rec},
    "useful_mhz_median_delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
    "phase_deltas": phase_deltas,
}
if os.environ.get("PG_NCU"):
    entry["ncu_report"] = os.environ["PG_NCU"]

led = os.environ["PG_LEDGER"]
os.makedirs(os.path.dirname(led), exist_ok=True)
with open(led, "a") as f:
    f.write(json.dumps(entry) + "\n")

lane = os.environ["PG_LANE"]
status = os.environ["PG_STATUS"]
if status != "ok":
    print(f"  {lane:<32} status={status}")
elif bm is None or fm is None:
    print(f"  {lane:<32} useful_mhz_median=n/a (see ledger)")
else:
    print(f"  {lane:<32} baseline={bm:.3f}  flagged={fm:.3f}  delta={delta_pct:+.2f}%")
    worst = sorted((v["delta_pct"], k) for k, v in phase_deltas.items()
                   if v["delta_pct"] is not None)
    for d, k in (worst[:2] + worst[-2:] if len(worst) > 4 else worst):
        print(f"    phase {k:<28} {d:+.2f}%")
PY
}

# Parity ledger entry (no perf record, just the counted parity outcome).
append_parity_ledger() {
  local lane="$1" status="$2" detail="$3"
  PG_TS="$TS" PG_LANE="$lane" PG_STATUS="$status" PG_DETAIL="$detail" \
  PG_STWO_REV="$STWO_REV" PG_STWO_DIRTY="$STWO_DIRTY" \
  PG_CAIRO_REV="$CAIRO_REV" PG_CAIRO_DIRTY="$CAIRO_DIRTY" \
  PG_GPU="$POD_GPU" PG_BENCH_ENV="$BENCH_ENV" PG_LEDGER="$PERF_LEDGER" python3 - <<'PY'
import json, os
entry = {
    "ts": os.environ["PG_TS"],
    "lane": os.environ["PG_LANE"],
    "status": os.environ["PG_STATUS"],
    "mode": "parity-only",
    "detail": os.environ.get("PG_DETAIL", ""),
    "stwo_rev": os.environ["PG_STWO_REV"],
    "stwo_dirty": os.environ["PG_STWO_DIRTY"],
    "cairo_rev": os.environ["PG_CAIRO_REV"],
    "cairo_dirty": os.environ["PG_CAIRO_DIRTY"],
    "pod_gpu": os.environ["PG_GPU"],
    "bench_env": os.environ.get("PG_BENCH_ENV", ""),
}
led = os.environ["PG_LEDGER"]
os.makedirs(os.path.dirname(led), exist_ok=True)
with open(led, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"  {os.environ['PG_LANE']:<32} parity status={os.environ['PG_STATUS']}"
      f"  {os.environ.get('PG_DETAIL','')}")
PY
}

# ---------------------------------------------------------------------------
# Parity-only: rerun the counted native differential suite(s) that actually
# exercise the lane flag. A suite that never reads the flag makes
# --parity-only vacuous, so every lane maps to ITS OWN suite(s); unknown lanes
# keep the historical relation+commit pair rather than silently passing.
# Exit 0 alone is insufficient (cfg-gated tests can run nothing) — require a
# "test result: ok. N passed" line with N >= 1 for EVERY mapped suite.
# ---------------------------------------------------------------------------
# Emits one "workdir|package|test_target" line per suite for the given lane.
parity_suite_specs() {
  local lane="$1"
  case "$lane" in
    STWO_CUDA_RELATION_*)
      echo "${STWO_POD}|stwo-backend-cuda|prepared_relation_native"
      ;;
    STWO_CUDA_BLAKE2S_LEAF_ILP|STWO_CUDA_BLAKE2S_INTERIOR_FUSED|STWO_CUDA_NTT_LEAF_FUSED)
      echo "${STWO_POD}|stwo-backend-cuda|prepared_commit_native"
      ;;
    STWO_CUDA_FRI_FOLD_FUSED)
      echo "${STWO_POD}|stwo-backend-cuda|prepared_fri_native"
      ;;
    STWO_CUDA_FEED_PRIVATIZED)
      echo "${STWO_POD}|stwo-backend-cuda|prepared_witness_feed_native"
      ;;
    STWO_CUDA_COMPOSITION_WIDE)
      echo "${POD_PROVER_DIR}|stwo-cairo-gpu-prover|prepared_composition_native"
      ;;
    *)
      echo "${STWO_POD}|stwo-backend-cuda|prepared_relation_native"
      echo "${STWO_POD}|stwo-backend-cuda|prepared_commit_native"
      ;;
  esac
}

run_parity_lane() {
  local lane="$1"
  local specs n_suites
  specs="$(parity_suite_specs "$lane")"
  n_suites="$(printf '%s\n' "$specs" | grep -c .)"
  # Chain suites with && so any failing suite fails the pod job.
  local cmds="" names="" dir pkg target
  while IFS='|' read -r dir pkg target; do
    [[ -z "$dir" ]] && continue
    [[ -n "$cmds" ]] && cmds+=" && "
    cmds+="cd '${dir}' && cargo test -p ${pkg} --test ${target}"
    names+="${target} + "
  done <<<"$specs"
  names="${names% + }"
  local body
  body="$(cat <<EOF
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
${BENCH_ENV:+export ${BENCH_ENV}}
export ${lane}=1
${cmds}
EOF
)"
  log "=== parity lane ${lane}: ${names} (${lane}=1) ==="
  run_pod_job "parity_${lane}" "$body"

  if [[ "$DRY_RUN" == "1" ]]; then
    append_parity_ledger "$lane" "parity_ok" "dry-run: ${n_suites} suites fabricated"
    return 0
  fi
  if [[ "$LAST_RC" != "0" ]]; then
    append_parity_ledger "$lane" "parity_failed" "rc=${LAST_RC}"
    return 1
  fi
  # Counted execution evidence: every mapped suite must report >= 1 passed.
  local suites_ok
  suites_ok="$(grep -c -E 'test result: ok\. [1-9][0-9]* passed' "$LAST_ERR" "$LAST_OUT" 2>/dev/null \
    | awk -F: '{ s += $NF } END { print s+0 }')"
  if [[ "${suites_ok:-0}" -lt "$n_suites" ]]; then
    append_parity_ledger "$lane" "parity_failed" \
      "exit 0 but only ${suites_ok:-0}/${n_suites} suites show a nonzero passed count (cfg-gated no-op?)"
    return 1
  fi
  append_parity_ledger "$lane" "parity_ok" "${n_suites} suites (${names}), nonzero executed counts"
}

# ---------------------------------------------------------------------------
# One lane: A/B pair + fail-closed proof compare (+ optional ncu profile)
# ---------------------------------------------------------------------------
run_lane() {
  local lane="$1"
  local args="--pie ${POD_PIE} --backend cuda ${GPU_NATIVE_ARGS} --reps ${REPS} --reuse-input --require-proof-byte-equal"
  local base_proof="${POD_RUN_DIR}/${STAMP}.${lane}.baseline.proof"
  local flag_proof="${POD_RUN_DIR}/${STAMP}.${lane}.flagged.proof"

  log "=== lane ${lane}: A/B SN_PIE_2 pair (reps=${REPS}) ==="

  # Arm A: baseline (lane flag explicitly OFF — unset, matching default).
  run_pod_job "${lane}.baseline" "$(bench_body "unset ${lane}" "$base_proof" "$args")"
  [[ "$DRY_RUN" == "1" ]] && synth_out "${lane}.baseline" "$LAST_OUT"
  local base_out="$LAST_OUT" base_rc="$LAST_RC" base_stall="$LAST_STALL_FILE"
  if [[ "$base_rc" != "0" ]]; then
    append_perf_ledger "$lane" "$([[ "$base_rc" == "STALLED" ]] && echo stalled || echo run_failed)" \
      "$base_out" "" "" "" ""
    warn "lane ${lane}: baseline arm failed (rc=${base_rc}${base_stall:+, stall evidence ${base_stall}})"
    return 1
  fi

  # Arm B: flagged (<lane>=1).
  run_pod_job "${lane}.flagged" "$(bench_body "export ${lane}=1" "$flag_proof" "$args")"
  [[ "$DRY_RUN" == "1" ]] && synth_out "${lane}.flagged" "$LAST_OUT"
  local flag_out="$LAST_OUT" flag_rc="$LAST_RC" flag_stall="$LAST_STALL_FILE"
  if [[ "$flag_rc" != "0" ]]; then
    append_perf_ledger "$lane" "$([[ "$flag_rc" == "STALLED" ]] && echo stalled || echo run_failed)" \
      "$base_out" "$flag_out" "" "" ""
    warn "lane ${lane}: flagged arm failed (rc=${flag_rc}${flag_stall:+, stall evidence ${flag_stall}})"
    return 1
  fi

  # FAIL CLOSED: cross-arm proof-byte identity. Every lane is documented
  # byte-identical; a mismatch is treated as a correctness failure of the lane.
  local fake_flag_sha="dryrun-proof-sha"
  [[ "${FAKE_PROOF_MISMATCH:-}" == "$lane" ]] && fake_flag_sha="dryrun-proof-sha-DIFFERENT"
  local base_sha flag_sha
  base_sha="$(pod_sha256 "$base_proof" "dryrun-proof-sha")"
  flag_sha="$(pod_sha256 "$flag_proof" "$fake_flag_sha")"
  if [[ -z "$base_sha" || -z "$flag_sha" ]]; then
    append_perf_ledger "$lane" "run_failed" "$base_out" "$flag_out" "$base_sha" "$flag_sha" ""
    warn "lane ${lane}: missing proof dump — cannot certify byte identity; failing closed"
    return 1
  fi
  if [[ "$base_sha" != "$flag_sha" ]]; then
    append_perf_ledger "$lane" "proof_mismatch" "$base_out" "$flag_out" "$base_sha" "$flag_sha" ""
    warn "lane ${lane}: PROOF BYTES DIFFER baseline=${base_sha} flagged=${flag_sha}"
    warn "lane ${lane}: the lane is documented byte-identical — this is a correctness alarm."
    return 1
  fi
  log "lane ${lane}: proof bytes identical across arms (sha256=${base_sha:0:16}…)"

  # Optional targeted ncu profile of the lane's kernels, flag ON. The A/B pair
  # above already warmed the on-disk JIT cache (the §11d.1 precondition).
  local ncu_report=""
  if [[ "$DO_NCU" == "1" ]]; then
    local kregex="${NCU_KERNELS_OVERRIDE:-$(lane_kernel_regex "$lane")}"
    local pod_rep="${POD_RUN_DIR}/${STAMP}.${lane}.ncu"
    local ncu_args="--pie ${POD_PIE} --backend cuda ${GPU_NATIVE_ARGS} --reps 2 --reuse-input --require-proof-byte-equal"
    local ncu_body
    ncu_body="$(cat <<EOF
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
export STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}'
export ${lane}=1
ncu --target-processes all \
    --kernel-name 'regex:${kregex}' \
    --launch-count ${NCU_LAUNCH_COUNT} \
    --set full -f -o '${pod_rep}' \
    ./${BIN} ${ncu_args}
EOF
)"
    log "lane ${lane}: ncu targeted capture (kernels ~ /${kregex}/, ${NCU_LAUNCH_COUNT} launches)"
    run_pod_job "${lane}.ncu" "$ncu_body"
    if [[ "$LAST_RC" == "0" ]]; then
      ncu_report="${RESULTS_DIR}/${STAMP}.${lane}.ncu-rep"
      if [[ "$DRY_RUN" == "1" ]]; then
        dry "ssh: cat '${pod_rep}.ncu-rep' -> ${ncu_report}"
        : > "$ncu_report"
      else
        run_ssh "cat '${pod_rep}.ncu-rep' 2>/dev/null" > "$ncu_report" || true
        [[ -s "$ncu_report" ]] || { warn "lane ${lane}: ncu report missing/empty on pod"; ncu_report=""; }
      fi
      [[ -n "$ncu_report" ]] && log "lane ${lane}: ncu report fetched -> ${ncu_report}"
    else
      warn "lane ${lane}: ncu capture failed (rc=${LAST_RC}) — perf numbers above still stand"
    fi
  fi

  append_perf_ledger "$lane" "ok" "$base_out" "$flag_out" "$base_sha" "$flag_sha" "$ncu_report"
}

# ===========================================================================
# Orchestration
# ===========================================================================
log "=== perf_gates start (lanes=${#LANES[@]} reps=${REPS} ncu=${DO_NCU} parity_only=${PARITY_ONLY} dry=${DRY_RUN}) ==="
echo "=== perf_gates summary (${TS}) — pod ${POD_GPU} ==="
echo "    revs: stwo=${STWO_REV:0:8}(${STWO_DIRTY}) cairo=${CAIRO_REV:0:8}(${CAIRO_DIRTY})"
[[ -n "$BENCH_ENV" ]] && echo "    bench_env(!): ${BENCH_ENV}  — NOT comparable with clean runs"

FAILED=0
for lane in "${LANES[@]}"; do
  if [[ "$PARITY_ONLY" == "1" ]]; then
    run_parity_lane "$lane" || FAILED=1
  else
    run_lane "$lane" || FAILED=1
  fi
done

echo "=== ledger: ${PERF_LEDGER} ==="
log "=== perf_gates done ==="
[[ "$FAILED" == "0" ]] || die "one or more lanes failed (see warnings + ${PERF_LEDGER})"
exit 0
