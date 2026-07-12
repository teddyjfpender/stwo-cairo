#!/usr/bin/env bash
#
# perf_gates.sh — one command per opt-in CUDA perf lane, runnable the moment
# bench_loop.sh's whole-proof correctness gate goes green.
#
# For each approved single-flag lane (STWO_CUDA_RELATION_SCAN_TAIL or
# STWO_CUDA_BLAKE2S_LEAF_ILP):
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
#   ./perf_gates.sh [--lanes CSV | --bundle NAME --candidate-env "K=V ..."]
#                   [--reps N] [--ncu] [--ncu-kernels REGEX]
#                   [--parity-only] [--help]
#
# One command per lane (the runbook lines):
#   ./perf_gates.sh --lanes STWO_CUDA_RELATION_SCAN_TAIL
#   ./perf_gates.sh --lanes STWO_CUDA_BLAKE2S_LEAF_ILP
#
# Flags:
#   --lanes CSV       Select approved single-flag lanes as a comma-separated list.
#   --bundle NAME     Compare one named multi-flag candidate against flags-off.
#   --candidate-env   Candidate K=V tokens used with --bundle.
#   --reps N          Reps per arm (default 4; >=2 so in-run byte-equal is real).
#   --ncu             After a lane's A/B pair passes, capture a targeted ncu
#                     profile of that lane's kernels (flag ON). Report is
#                     fetched to loop/results/ and its pod path recorded.
#   --ncu-kernels RE  Override the lane's default kernel-name regex.
#   --parity-only     Only rerun prepared_relation_native + prepared_commit_native
#                     with each lane flag exported. No benchmarks, no ledger
#                     perf entry (a parity entry is still appended).
#
# Environment (same plumbing as bench_loop.sh):
#   BENCH_POD_ID   Pod id (overrides POD_ID in loop/pod.conf).
#   BENCH_ENV      "K=V K=V ..." exported into EVERY run of BOTH arms (recorded
#                  per ledger entry). Lane flags must not appear here.
#   DRY_RUN=1      Echo every ssh instead of executing; fabricate run output so
#                  the A/B -> proof-compare -> ledger path runs for real offline.
#   FAKE_PROOF_MISMATCH  (DRY_RUN only) lane name whose flagged proof is
#                  fabricated DIFFERENT, to exercise the fail-closed path.
#   FAKE_REMOTE_QUIESCENCE_FAILURE  (DRY_RUN only) fail the remote-isolation
#                  check before any timed arm, for fail-closed harness testing.
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
ARCHITECTURE_CHECK="${CAIRO_LOCAL}/gpu_benchmarks/validate_architecture_record.py"
ARCHITECTURE_SOUNDNESS_GATE="${ARCHITECTURE_SOUNDNESS_GATE:-}"
LOCAL_PREFLIGHT_ADMISSION="${LOCAL_PREFLIGHT_ADMISSION:-}"
INPUT_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/SHA256SUMS"
PINNED_ADAPTED_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/ADAPTED_SHA256SUMS"
BOOTLOADER_JSON_SOURCE="${BOOTLOADER_JSON_SOURCE:-}"

STWO_POD="/workspace/stwo"
CAIRO_POD="/workspace/stwo-cairo"
POD_PROVER_DIR="${CAIRO_POD}/stwo_cairo_prover"
POD_USER="root"

POD_SN_DIR="${CAIRO_POD}/gpu_benchmarks/pie/sn"
POD_PIE="${POD_SN_DIR}/SN_PIE_2.zip"                # the lane-gate statement
POD_BOOTLOADER_JSON="${POD_BOOTLOADER_JSON:-/workspace/bench_inputs/simple_bootloader_compiled.json}"

# Scratch outside the repo tree (bench_loop's rsync never touches it).
POD_RUN_DIR="/workspace/perf_gate_runs"

RUST_MIN_STACK_VAL=4194304
BENCH_ENV="${BENCH_ENV:-}"
GPU_PCS_RUNTIME_MODE="${GPU_PCS_RUNTIME_MODE:-arena-graph}"
EXPECTED_POD_GPU="${EXPECTED_POD_GPU:-}"
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
# The resident Cairo plan seals compact relation execution to Fused, so toggling
# STWO_CUDA_RELATION_FUSED here would benchmark identical graph topologies.
APPROVED_SINGLE_LANES=(STWO_CUDA_RELATION_SCAN_TAIL STWO_CUDA_BLAKE2S_LEAF_ILP)
DEFAULT_LANES=("${APPROVED_SINGLE_LANES[@]}")
LANES=()
BUNDLE_NAME=""
CANDIDATE_ENV=""
APPROVED_BUNDLE_BOOLEAN_FLAGS=(
  STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE
  STWO_CUDA_COMPOSITION_DIRECT_RETENTION
  STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS
  STWO_CUDA_B2N_STAGE_FUSED
)
RETAINED_BUDGET_POLICY="STWO_CUDA_RETAINED_LDE_BUDGET_BYTES"
DEFAULT_RETAINED_BUDGET_BYTES="8589934592"
BASELINE_STATE="{}"
CANDIDATE_STATE="{}"

# Set by resolve_pod().
POD_HOST=""
POD_PORT=""
POD_KEY=""
POD_ID_RESOLVED=""
SSH_OPTS=()
SEALED_BOOT_ID=""
SEALED_GPU_UUID=""
SEALED_GPU_NAME=""
SEALED_GPU_BENCH_PATH=""
SEALED_GPU_BENCH_SHA=""

# Set by run_pod_job().
LAST_OUT=""
LAST_ERR=""
LAST_RC=""
LAST_STALL_FILE=""
LAST_PGID_FILE=""
BASELINE_QUIESCENCE_PASSED=0
FLAGGED_QUIESCENCE_PASSED=0

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
  if [[ "$DRY_RUN" == "1" ]]; then
    POD_ID_RESOLVED="DRY-RUN-POD"
    POD_HOST="dry-run.invalid"; POD_PORT="22"; POD_KEY="/dev/null"
    SSH_OPTS=(-p "$POD_PORT" -i "$POD_KEY")
    return 0
  fi
  [[ -n "$pod_id" ]] || die "no pod id — set BENCH_POD_ID or POD_ID in ${POD_CONF}"
  POD_ID_RESOLVED="$pod_id"

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
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else LC_ALL=C shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

expected_sha256() {
  awk -v file="$(basename "$1")" '$2 == file { print $1; exit }' "$INPUT_SHA256SUMS"
}

load_execution_target() {
  local key value
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
  done < <(python3 - "$ARCHITECTURE_SOUNDNESS_GATE" "$POD_PIE" \
    "$(expected_sha256 "$POD_PIE")" "$POD_BOOTLOADER_JSON" \
    "$(expected_sha256 "$POD_BOOTLOADER_JSON")" <<'PY'
import hashlib, json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    artifact = json.load(stream)
target = artifact.get("execution_target") or {}
canonical = json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
if artifact.get("schema") != "stwo.cuda.soundness-gate.v3":
    raise SystemExit("invalid soundness schema")
if target.get("schema") != "stwo.remote-execution-target.v1":
    raise SystemExit("invalid execution target schema")
if artifact.get("execution_target_sha256") != hashlib.sha256(canonical).hexdigest():
    raise SystemExit("execution target hash mismatch")
if artifact.get("execution_target_postcheck") is not True:
    raise SystemExit("execution target postcheck failed")
inputs = target.get("inputs") or {}
for label, path, digest in (
    ("SN_PIE_2", sys.argv[2], sys.argv[3]),
    ("bootloader", sys.argv[4], sys.argv[5]),
):
    # dict() avoids a macOS Bash 3.2 brace-expansion bug when this heredoc is
    # nested inside the process substitution consumed by the shell loop.
    if inputs.get(label) != dict(path=path, sha256=digest):
        raise SystemExit(f"execution target {label} is not manifest-pinned")
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
  [[ "$SEALED_GPU_BENCH_PATH" == "/workspace/bench_loop_runs/sealed/gpu_bench.${SEALED_GPU_BENCH_SHA}" ]] \
    || die "soundness target binary is not content-addressed"
  [[ "$SEALED_GPU_NAME" == "$POD_GPU" ]] \
    || die "soundness target GPU name mismatch: expected $POD_GPU, got $SEALED_GPU_NAME"
}

sealed_input_sha() {
  python3 - "$ARCHITECTURE_SOUNDNESS_GATE" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    inputs = (json.load(stream).get("execution_target") or {}).get("inputs") or {}
matches = {value.get("sha256") for value in inputs.values()
           if isinstance(value, dict) and value.get("path") == sys.argv[2]}
if len(matches) != 1:
    raise SystemExit(f"sealed input path is absent or ambiguous: {sys.argv[2]}")
print(matches.pop())
PY
}

sealed_input_paths() {
  python3 - "$ARCHITECTURE_SOUNDNESS_GATE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    inputs = (json.load(stream).get("execution_target") or {}).get("inputs") or {}
for path in sorted({value.get("path") for value in inputs.values()
                    if isinstance(value, dict) and isinstance(value.get("path"), str)}):
    print(path)
PY
}

remote_execution_guard() {
  local paths_csv="$1" script quoted expected path old_ifs
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
  old_ifs="$IFS"; IFS=','
  for path in $paths_csv; do
    [[ -n "$path" ]] || continue
    expected="$(sealed_input_sha "$path")" || die "input is absent from remote seal: $path"
    printf -v quoted '%q' "$path"; printf -v expected '%q' "$expected"
    script+=" actual=\$(hash_sealed ${quoted}); [[ \"\$actual\" == ${expected} ]] || { echo 'remote seal: input changed: ${quoted}' >&2; exit 96; };"
  done
  IFS="$old_ifs"
  printf '%s\n' "$script"
}

validate_remote_execution_target() {
  local all_paths seal_guard
  load_execution_target
  if [[ "$DRY_RUN" == "1" ]]; then
    [[ -z "${FAKE_REMOTE_TARGET_MISMATCH:-}" ]] \
      || die "remote execution target mismatch (synthetic ${FAKE_REMOTE_TARGET_MISMATCH})"
    return 0
  fi
  all_paths="$(sealed_input_paths | paste -sd, -)"
  seal_guard="$(remote_execution_guard "$all_paths")" \
    || die "could not construct the sealed remote-execution guard"
  run_ssh "$seal_guard" \
    || die "remote pod/GPU/binary/input target no longer matches counted soundness"
}

# Fail closed unless the sealed GPU has no compute clients and the pod has no
# exact known prover/build worker. With a pgid file, also wait for the detached
# launcher's process group to disappear before accepting a completed arm.
remote_quiescence_guard() {
  local pgid_file="${1:-}" quoted_uuid quoted_binary quoted_pgid="''"
  printf -v quoted_uuid '%q' "$SEALED_GPU_UUID"
  printf -v quoted_binary '%q' "$SEALED_GPU_BENCH_PATH"
  [[ -z "$pgid_file" ]] || printf -v quoted_pgid '%q' "$pgid_file"
  cat <<EOF
if [[ -n ${quoted_pgid} ]]; then
  perf_pgid=\$(cat ${quoted_pgid} 2>/dev/null) || { echo 'remote isolation: missing process-group sentinel' >&2; exit 97; }
  [[ "\$perf_pgid" =~ ^[1-9][0-9]*$ && "\$perf_pgid" -gt 1 ]] \
    || { echo 'remote isolation: invalid process-group sentinel' >&2; exit 97; }
  perf_group_has_live_member() {
    local perf_member perf_stat perf_rest perf_state perf_ppid perf_pgrp perf_sid
    for perf_member in /proc/[0-9]*; do
      perf_stat=\$(cat "\$perf_member/stat" 2>/dev/null) || continue
      perf_rest=\${perf_stat##*) }
      read -r perf_state perf_ppid perf_pgrp perf_sid _ <<<"\$perf_rest"
      [[ "\$perf_pgrp" != "\$perf_pgid" || "\$perf_state" == Z ]] || return 0
    done
    return 1
  }
  if [[ -r "/proc/\$perf_pgid/stat" ]]; then
    perf_stat=\$(cat "/proc/\$perf_pgid/stat") || exit 97
    perf_rest=\${perf_stat##*) }
    read -r perf_state perf_ppid perf_pgrp perf_sid _ <<<"\$perf_rest"
    [[ "\$perf_pgrp" == "\$perf_pgid" && "\$perf_sid" == "\$perf_pgid" ]] \
      || { echo 'remote isolation: process-group sentinel is not its session leader' >&2; exit 97; }
  fi
  perf_group_live=1
  for perf_wait in 1 2 3 4 5; do
    if ! perf_group_has_live_member; then perf_group_live=0; break; fi
    sleep 1
  done
  [[ "\$perf_group_live" == 0 ]] || { echo "remote isolation: process group \$perf_pgid remains alive" >&2; exit 97; }
fi
perf_gpu_pids=\$(nvidia-smi --id=${quoted_uuid} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) \
  || { echo 'remote isolation: could not query sealed GPU compute clients' >&2; exit 97; }
perf_gpu_pids=\$(printf '%s\n' "\$perf_gpu_pids" | awk 'NF { print }')
[[ -z "\$perf_gpu_pids" ]] \
  || { echo "remote isolation: sealed GPU has active compute clients: \$perf_gpu_pids" >&2; exit 97; }
perf_binary_inode=\$(stat -Lc '%d:%i' ${quoted_binary} 2>/dev/null) \
  || { echo 'remote isolation: sealed gpu_bench is missing' >&2; exit 97; }
perf_workers=''
for perf_proc in /proc/[0-9]*; do
  [[ -r "\$perf_proc/comm" ]] || continue
  perf_stat=\$(cat "\$perf_proc/stat" 2>/dev/null) || continue
  perf_rest=\${perf_stat##*) }
  read -r perf_state perf_ppid perf_pgrp perf_sid _ <<<"\$perf_rest"
  [[ "\$perf_state" != Z ]] || continue
  IFS= read -r perf_comm < "\$perf_proc/comm" || continue
  perf_exe=\$(readlink "\$perf_proc/exe" 2>/dev/null || true)
  perf_base=\${perf_exe##*/}
  perf_inode=\$(stat -Lc '%d:%i' "\$perf_proc/exe" 2>/dev/null || true)
  perf_is_worker=0
  case "\$perf_comm" in
    gpu_bench|ncu|cargo|rustc|rustdoc|build-script-bu|nvcc|ptxas|cicc|cudafe++|fatbinary|nvlink|ninja|make|cmake|cc|cc1|cc1plus|gcc|g++|c++|clang|clang++|ld|ld.lld|mold|rsync) perf_is_worker=1 ;;
  esac
  case "\$perf_base" in
    gpu_bench|ncu|cargo|rustc|rustdoc|build-script-build|nvcc|ptxas|cicc|cudafe++|fatbinary|nvlink|ninja|make|cmake|cc|cc1|cc1plus|gcc|g++|c++|clang|clang++|ld|ld.lld|mold|rsync) perf_is_worker=1 ;;
  esac
  [[ "\$perf_exe" != ${quoted_binary} && "\$perf_inode" != "\$perf_binary_inode" ]] || perf_is_worker=1
  [[ "\$perf_is_worker" == 0 ]] || perf_workers+="\${perf_proc##*/}:\${perf_comm}:\${perf_exe} "
done
[[ -z "\$perf_workers" ]] \
  || { echo "remote isolation: active prover/build workers: \$perf_workers" >&2; exit 97; }
EOF
}

assert_remote_quiescent() {
  local guard
  guard="$(remote_quiescence_guard "${1:-}")" \
    || die "could not construct remote-isolation guard"
  if [[ "$DRY_RUN" == "1" && -n "${FAKE_REMOTE_QUIESCENCE_FAILURE:-}" ]]; then
    warn "remote isolation failed (synthetic ${FAKE_REMOTE_QUIESCENCE_FAILURE})"
    return 1
  fi
  run_ssh "$guard"
}

terminate_remote_process_group() {
  local pgid_file="$1" name="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    dry "terminate and verify detached process group from '${pgid_file}' (${name})"
    return 0
  fi
  run_ssh "pgid=\$(cat '${pgid_file}' 2>/dev/null) || exit 1; \
    case \"\$pgid\" in ''|*[!0-9]*) exit 1 ;; esac; [ \"\$pgid\" -gt 1 ] || exit 1; \
    group_has_live_member() { \
      for member in /proc/[0-9]*; do \
        stat=\$(cat \"\$member/stat\" 2>/dev/null) || continue; rest=\${stat##*) }; \
        read -r state ppid pgrp sid remainder <<<\"\$rest\"; \
        [ \"\$pgrp\" != \"\$pgid\" ] || [ \"\$state\" = Z ] || return 0; \
      done; return 1; \
    }; \
    if [ -r \"/proc/\$pgid/stat\" ]; then \
      stat=\$(cat \"/proc/\$pgid/stat\") || exit 1; rest=\${stat##*) }; \
      read -r state ppid pgrp sid remainder <<<\"\$rest\"; \
      [ \"\$pgrp\" = \"\$pgid\" ] && [ \"\$sid\" = \"\$pgid\" ] || exit 1; \
    elif ! group_has_live_member; then exit 0; else exit 1; fi; \
    kill -TERM -- -\"\$pgid\" 2>/dev/null || true; sleep 3; \
    kill -KILL -- -\"\$pgid\" 2>/dev/null || true; \
    for wait_step in 1 2 3 4 5; do \
      group_has_live_member || exit 0; sleep 1; \
    done; exit 1"
}

validate_completed_arm() {
  validate_remote_execution_target
  assert_remote_quiescent "$LAST_PGID_FILE" \
    || die "remote GPU/build-worker isolation failed after completed arm"
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
    --bundle)      BUNDLE_NAME="${2:?--bundle needs a name}"; shift 2 ;;
    --candidate-env) CANDIDATE_ENV="${2:?--candidate-env needs K=V tokens}"; shift 2 ;;
    --reps)        REPS="${2:?--reps needs a value}"; shift 2 ;;
    --ncu)         DO_NCU=1; shift ;;
    --ncu-kernels) NCU_KERNELS_OVERRIDE="${2:?--ncu-kernels needs a regex}"; shift 2 ;;
    --parity-only) PARITY_ONLY=1; shift ;;
    -h|--help)     usage 0 ;;
    --*)           die "unknown flag: $1 (try --help)" ;;
    *)             die "positional lane flags are not accepted; use --lanes with an approved lane" ;;
  esac
done

[[ -z "$BUNDLE_NAME" || ${#LANES[@]} -eq 0 ]] \
  || die "--bundle and --lanes are mutually exclusive"
[[ -z "$BUNDLE_NAME" || -n "$CANDIDATE_ENV" ]] \
  || die "--bundle requires --candidate-env"
[[ -n "$BUNDLE_NAME" || -z "$CANDIDATE_ENV" ]] \
  || die "--candidate-env requires --bundle"
if [[ -n "$BUNDLE_NAME" ]]; then
  [[ "$BUNDLE_NAME" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] \
    || die "bundle name '$BUNDLE_NAME' is not a safe artifact label"
  [[ "$PARITY_ONLY" == "0" ]] || die "--bundle does not support --parity-only"
  LANES=("$BUNDLE_NAME")
elif [[ ${#LANES[@]} -eq 0 ]]; then
  LANES=("${DEFAULT_LANES[@]}")
fi
[[ "$REPS" =~ ^[0-9]+$ && "$REPS" -ge 2 ]] \
  || die "--reps must be >= 2 so in-run proof-byte equality is a real check"

case "$GPU_PCS_RUNTIME_MODE" in
  detached-eager|arena-graph) ;;
  *) die "GPU_PCS_RUNTIME_MODE must be detached-eager or arena-graph (got '$GPU_PCS_RUNTIME_MODE')" ;;
esac

for lane in "${LANES[@]}"; do
  if [[ -z "$BUNDLE_NAME" ]]; then
    [[ "$lane" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "lane '$lane' is not a valid env flag name"
    approved=0
    for approved_lane in "${APPROVED_SINGLE_LANES[@]}"; do
      [[ "$lane" != "$approved_lane" ]] || approved=1
    done
    [[ "$approved" == "1" ]] \
      || die "lane '$lane' is not approved for single-flag paid execution; use an exact admitted bundle"
  fi
done

CANDIDATE_KEYS_SEEN="|"
for kv in $CANDIDATE_ENV; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./,:+-]*$ ]] \
    || die "candidate token '$kv' is not a shell-safe K=V token"
  key="${kv%%=*}"
  value="${kv#*=}"
  [[ "$CANDIDATE_KEYS_SEEN" != *"|${key}|"* ]] \
    || die "candidate policy ${key} appears more than once"
  CANDIDATE_KEYS_SEEN+="${key}|"
  for common in $BENCH_ENV; do
    [[ "${common%%=*}" != "$key" ]] \
      || die "candidate policy ${key} must not also appear in BENCH_ENV"
  done
  approved=0
  for flag in "${APPROVED_BUNDLE_BOOLEAN_FLAGS[@]}"; do
    if [[ "$key" == "$flag" ]]; then
      [[ "$value" == "1" ]] \
        || die "bundle boolean ${key} must be set exactly to 1"
      approved=1
      break
    fi
  done
  if [[ "$key" == "$RETAINED_BUDGET_POLICY" ]]; then
    [[ "$value" =~ ^[0-9]+$ ]] \
      || die "bundle numeric policy ${key} must be a nonnegative decimal integer"
    if ! python3 - "$value" <<'PY'
import sys

value = int(sys.argv[1], 10)
raise SystemExit(0 if value <= (1 << 64) - 1 else 1)
PY
    then
      die "bundle numeric policy ${key} exceeds the unsigned 64-bit range"
    fi
    approved=1
  fi
  [[ "$approved" == "1" ]] \
    || die "bundle candidate token '$kv' is not an approved qualification policy"
done
if [[ -n "$BUNDLE_NAME" ]]; then
  BASELINE_STATE='{"STWO_CUDA_B2N_STAGE_FUSED":0,"STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE":0,"STWO_CUDA_COMPOSITION_DIRECT_RETENTION":0,"STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS":0,"STWO_CUDA_RETAINED_LDE_BUDGET_BYTES":8589934592}'
  CANDIDATE_STATE="$(CANDIDATE_ENV="$CANDIDATE_ENV" DEFAULT_RETAINED_BUDGET_BYTES="$DEFAULT_RETAINED_BUDGET_BYTES" RETAINED_BUDGET_POLICY="$RETAINED_BUDGET_POLICY" python3 - <<'PY'
import json, os
flags = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
    "STWO_CUDA_B2N_STAGE_FUSED",
)
settings = dict(token.split("=", 1) for token in os.environ["CANDIDATE_ENV"].split())
state = {flag: int(settings.get(flag) == "1") for flag in flags}
budget_policy = os.environ["RETAINED_BUDGET_POLICY"]
state[budget_policy] = int(
    settings.get(budget_policy, os.environ["DEFAULT_RETAINED_BUDGET_BYTES"]), 10
)
print(json.dumps(state, sort_keys=True))
PY
)"
fi

# Validate BENCH_ENV shape and keep lane flags out of it (an arm's identity
# must come from this script, never ambient env).
if [[ -n "$BENCH_ENV" ]]; then
  for kv in $BENCH_ENV; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./,:+-]*$ ]] \
      || die "BENCH_ENV token '$kv' is not a shell-safe K=V token"
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

# ---------------------------------------------------------------------------
# Provenance (same fields as bench_loop's ledger)
# ---------------------------------------------------------------------------
git_rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }
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

[[ -n "$LOCAL_PREFLIGHT_ADMISSION" && -f "$LOCAL_PREFLIGHT_ADMISSION" ]] \
  || die "LOCAL_PREFLIGHT_ADMISSION from qualification_round is required before pod work"
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
if ! PA_STWO_HEAD="$STWO_REV" PA_STWO_HASH="$STWO_WORKTREE_HASH" \
  PA_CAIRO_HEAD="$CAIRO_REV" PA_CAIRO_HASH="$CAIRO_WORKTREE_HASH" \
  PA_RUNTIME="$GPU_PCS_RUNTIME_MODE" PA_DRY_RUN="$DRY_RUN" \
  PA_BASELINE_ENV="$BENCH_ENV" PA_CANDIDATE_ENV="$CANDIDATE_ENV" \
  PA_IS_BUNDLE="$([[ -n "$BUNDLE_NAME" ]] && echo 1 || echo 0)" \
  python3 - "$LOCAL_PREFLIGHT_ADMISSION" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    admission = json.load(stream)
expected_source = {
    "stwo": {"head": os.environ["PA_STWO_HEAD"], "worktree_hash": os.environ["PA_STWO_HASH"]},
    "stwo_cairo": {
        "head": os.environ["PA_CAIRO_HEAD"],
        "worktree_hash": os.environ["PA_CAIRO_HASH"],
    },
}
profiles = admission.get("profiles") or {}
admitted_envs = set(profiles.values())
required_envs = {os.environ["PA_BASELINE_ENV"]}
if os.environ["PA_IS_BUNDLE"] == "1":
    required_envs.add(os.environ["PA_CANDIDATE_ENV"])
if (admission.get("schema") != "stwo.local-preflight-admission.v1"
        or admission.get("passed") is not True
        or admission.get("runtime_mode") != os.environ["PA_RUNTIME"]
        or admission.get("dry_run") is not (os.environ["PA_DRY_RUN"] == "1")
        or admission.get("source") != expected_source
        or not required_envs.issubset(admitted_envs)):
    raise SystemExit(1)
PY
then
  die "local preflight admission does not authorize current source/profile/runtime"
fi

[[ -n "$ARCHITECTURE_SOUNDNESS_GATE" && -f "$ARCHITECTURE_SOUNDNESS_GATE" ]] \
  || die "ARCHITECTURE_SOUNDNESS_GATE must name the current counted gate before pod work"
python3 "$ARCHITECTURE_CHECK" --soundness-only \
  --soundness-gate "$ARCHITECTURE_SOUNDNESS_GATE" \
  --runtime-mode "$GPU_PCS_RUNTIME_MODE" --expected-dry-run "$DRY_RUN" \
  --stwo-head "$STWO_REV" --stwo-worktree-hash "$STWO_WORKTREE_HASH" \
  --stwo-cairo-head "$CAIRO_REV" --stwo-cairo-worktree-hash "$CAIRO_WORKTREE_HASH" \
  || die "soundness artifact is not fully bound to current source/runtime/gates"

resolve_pod
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
[[ -z "$EXPECTED_POD_GPU" || "$POD_GPU" == "$EXPECTED_POD_GPU" ]] \
  || die "qualification requires GPU '${EXPECTED_POD_GPU}', got '${POD_GPU}'"
validate_remote_execution_target

# ---------------------------------------------------------------------------
# DRY_RUN synthetic gpu_bench output (exercises the A/B -> ledger path offline)
# ---------------------------------------------------------------------------
synth_out() {
  # $1 name  $2 destfile — deterministic mock numbers keyed off the name so the
  # flagged arm's delta is nonzero.
  local name="$1" dest="$2"
  local seed=$(( $(printf '%s' "$name" | cksum | cut -d' ' -f1) % 40 ))
  local um_median warm_s warm_rounded mhz_median raw_samples="[" rep
  um_median="$(awk -v s="$seed" 'BEGIN{printf "%.3f", 1.3 + s/100.0}')"
  warm_s="$(awk -v m="$um_median" 'BEGIN{printf "%.9f", 7.706864/m}')"
  warm_rounded="$(awk -v s="$warm_s" 'BEGIN{printf "%.3f", s}')"
  mhz_median="$(awk -v s="$warm_s" 'BEGIN{printf "%.3f", 7.977397/s}')"
  for ((rep = 1; rep < REPS; rep++)); do
    [[ "$rep" == "1" ]] || raw_samples+=","
    raw_samples+="$warm_s"
  done
  raw_samples+="]"
  local stages='{"OodsEvaluation":1,"QuotientAndCompaction":1,"FriCommitAndFold":1,"ProofOfWork":1,"FriQueryAndDecommit":1,"TreeDecommit":1,"Assembly":1}'
  local architecture='"gpu_pcs_driver_architecture":"cuda-typed-pcs-driver-v1","gpu_pcs_runtime_mode":"ArenaGraph","gpu_pcs_stage_started":'"$stages"',"gpu_pcs_stage_finished":'"$stages"',"gpu_pcs_batched_tree_decommit":true,"gpu_pcs_driver_complete":true,"gpu_native_architecture_required":true,"gpu_pcs_required_runtime_mode":"arena-graph","gpu_native_architecture_gate_passed":true,"gpu_aot_loads":2,"gpu_aot_cache_hits":5,"gpu_aot_manifest_hash":49370,"gpu_aot_misses":0,"gpu_aot_runtime_loads":0,"gpu_aot_runtime_cache_hits":0,"gpu_aot_strict_rejections":0,"gpu_aot_provenance_gate_passed":true,"performance_claim_admissible":true,"steps_per_s":1000000,"mhz":1.0,"useful_mhz":1.0,"gpu_host_syncs":1,"gpu_graph_launches":8,"gpu_kernel_launches":72,"gpu_hot_h2d_bytes":0,"gpu_hot_d2h_bytes":1024,"gpu_hot_allocations":0,"gpu_max_graph_submit_gap_ms":1.25,"gpu_graph_a_setup_gate_passed":true,"gpu_setup_base_migration_copies":0,"gpu_setup_lookup_host_copies":0,"gpu_setup_legacy_witness_fallbacks":0,"gpu_execution_tables_ingest_compact_h2d_bytes":4096,"gpu_execution_tables_ingest_compact_h2d_copies":3,"gpu_execution_tables_ingest_descriptor_h2d_bytes":64,"gpu_execution_tables_ingest_descriptor_h2d_copies":2,"gpu_execution_tables_ingest_syncs":1,"gpu_witness_ingest_syncs":1'
  {
    echo "{\"rep\":0,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":$((1200 + seed * 3)).5},\"fri\":{\"count\":1,\"total_ms\":$((560 + seed)).1}}}"
    echo "{\"rep\":1,\"phase_totals\":{\"witness_generation\":{\"count\":1,\"total_ms\":$((1190 + seed * 3)).2},\"fri\":{\"count\":1,\"total_ms\":$((555 + seed)).8}}}"
    echo "{\"program\":\"SN_PIE_2.zip\",\"backend\":\"cuda\",\"engine\":\"gpu-native\",${architecture},\"cycle_count\":7977397,\"pie_n_steps\":7706864,\"reps\":${REPS},\"warm_sample_count\":$((REPS - 1)),\"prove_s_warm_samples_raw\":${raw_samples},\"prove_s_warm_samples_rounded\":${raw_samples},\"prove_s_warm_median\":${warm_rounded},\"mhz_median\":${mhz_median},\"throughput_distribution_applicable\":true,\"verified_reps\":${REPS},\"proof_byte_equal\":true,\"proof_byte_equal_required\":true,\"proof_comparison_applicable\":true,\"useful_mhz_median\":${um_median},\"vram_peak_gb\":11.3,\"gpu\":\"${POD_GPU}\"}"
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
  rm -f "${base}.rc"
  LAST_STALL_FILE=""
  LAST_RC=""
  LAST_PGID_FILE="$pod_pgid"
  LAST_OUT="${base}.out"
  LAST_ERR="${base}.err"

  if [[ "$DRY_RUN" == "1" ]]; then
    dry "write launcher ${pod_sh}:"
    printf '%s\n' "$body" | sed 's/^/[DRY_RUN]   | /' >&2
    dry "ssh: rm -f '${pod_rc}'; nohup setsid bash '${pod_sh}' > '${pod_out}' 2> '${pod_err}' & echo LAUNCHED"
    dry "startup contract: within 15s require '${pod_rc}' or an owned live session leader from '${pod_pgid}'"
    dry "poll: cat '${pod_rc}' until present (stall: stderr size + GPU util frozen ${STALL_SECS}s)"
    : > "${base}.err"
    LAST_RC=0
    return 0
  fi

  run_ssh "mkdir -p '${POD_RUN_DIR}'" || return 1
  run_ssh "cat > '${pod_sh}'" <<EOF || return 1
#!/usr/bin/env bash
trap 'rc=\$?; printf "%s\\n" "\$rc" > "${pod_rc}"' EXIT
set -euo pipefail
echo \$\$ > '${pod_pgid}'
${body}
EOF
  run_ssh "rm -f '${pod_rc}' '${pod_pgid}'; nohup setsid bash '${pod_sh}' > '${pod_out}' 2> '${pod_err}' & echo LAUNCHED" || return 1

  # Do not spend MAX_WAIT polling a detached launcher that never established
  # either its exit sentinel or its owned setsid process group.
  local startup_result="" startup_rc=0
  startup_result="$(run_ssh "startup_reason='no rc or pgid sentinel'; \
    for startup_wait in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
      if [ -f '${pod_rc}' ]; then echo READY=rc; exit 0; fi; \
      if [ -f '${pod_pgid}' ]; then \
        pgid=\$(cat '${pod_pgid}' 2>/dev/null) || { startup_reason='unreadable pgid sentinel'; sleep 1; continue; }; \
        case \"\$pgid\" in ''|*[!0-9]*) echo \"INVALID=\$pgid\"; exit 97 ;; esac; \
        [ \"\$pgid\" -gt 1 ] || { echo \"INVALID=\$pgid\"; exit 97; }; \
        if [ -r \"/proc/\$pgid/stat\" ]; then \
          stat=\$(cat \"/proc/\$pgid/stat\") || { startup_reason='unreadable leader stat'; sleep 1; continue; }; \
          rest=\${stat##*) }; read -r state ppid pgrp sid remainder <<<\"\$rest\"; \
          if [ \"\$pgrp\" = \"\$pgid\" ] && [ \"\$sid\" = \"\$pgid\" ] && [ \"\$state\" != Z ]; then \
            echo \"READY=pgid:\$pgid\"; exit 0; \
          fi; \
          if [ \"\$pgrp\" != \"\$pgid\" ] || [ \"\$sid\" != \"\$pgid\" ]; then \
            echo \"INVALID=pgid:\$pgid:pgrp:\$pgrp:sid:\$sid\"; exit 97; \
          fi; \
          startup_reason='session leader is a zombie'; \
        else startup_reason='session leader is absent'; fi; \
      fi; \
      sleep 1; \
    done; \
    echo \"STARTUP_TIMEOUT=\$startup_reason\"; exit 97")" || startup_rc=$?
  if [[ "$startup_rc" != 0 ]]; then
    local startup_file="${base}.startup.txt"
    {
      printf '%s\n' "$startup_result"
      run_ssh "echo '--- pgid sentinel ---'; cat '${pod_pgid}' 2>/dev/null || echo '(missing)'; \
               echo '--- rc sentinel ---'; cat '${pod_rc}' 2>/dev/null || echo '(missing)'; \
               echo '--- last 30 stderr lines ---'; tail -n 30 '${pod_err}' 2>/dev/null; true" || true
    } > "$startup_file"
    terminate_remote_process_group "$pod_pgid" "$name" \
      || warn "startup failure for '${name}' exposed no safely owned process group to terminate"
    run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
    run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
    LAST_RC="STARTUP_FAILED"
    LAST_STALL_FILE="$startup_file"
    warn "job '${name}' failed its detached startup contract; evidence: ${startup_file}"
    return 0
  fi

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
      terminate_remote_process_group "$pod_pgid" "$name" \
        || die "could not terminate stalled process group for '${name}'"
      run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
      run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
      LAST_RC="STALLED"; LAST_STALL_FILE="$stall_file"
      log "stall evidence written to ${stall_file}"
      return 0
    fi

    waited=$(( waited + POLL_INTERVAL ))
    if (( waited >= MAX_WAIT )); then
      warn "job '${name}' exceeded MAX_WAIT (${MAX_WAIT}s). Killing its detached process group."
      terminate_remote_process_group "$pod_pgid" "$name" \
        || die "could not terminate timed-out process group for '${name}'"
      run_ssh "cat '${pod_out}' 2>/dev/null" > "${base}.out" || true
      run_ssh "cat '${pod_err}' 2>/dev/null" > "${base}.err" || true
      LAST_RC="TIMEOUT"
      log "job '${name}' timed out and was terminated (rc=${LAST_RC})"
      return 0
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
  local extra_exports="$1" proof_dump="$2" args="$3" seal_guard quiescence_guard
  seal_guard="$(remote_execution_guard "${POD_PIE},${POD_BOOTLOADER_JSON}")" || return 1
  quiescence_guard="$(remote_quiescence_guard)" || return 1
  cat <<EOF
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
export STWO_DUMP_PROOF='${proof_dump}'
${extra_exports}
${seal_guard}
${quiescence_guard}
/proc/self/fd/9 ${args}
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

proof_contract_ok() {
  python3 - "$1" "$REPS" <<'PY'
import json, sys
record = None
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "verified_reps" in candidate:
            record = candidate
required = {
    "verified_reps": int(sys.argv[2]),
    "proof_comparison_applicable": True,
    "proof_byte_equal_required": True,
    "proof_byte_equal": True,
}
raise SystemExit(0 if record is not None and all(record.get(k) == v for k, v in required.items()) else 1)
PY
}

architecture_contract_ok() {
  [[ -n "$ARCHITECTURE_SOUNDNESS_GATE" && -f "$ARCHITECTURE_SOUNDNESS_GATE" ]] || return 1
  python3 "$ARCHITECTURE_CHECK" "$1" --runtime-mode "$GPU_PCS_RUNTIME_MODE" \
    --soundness-gate "$ARCHITECTURE_SOUNDNESS_GATE" \
    --expected-program SN_PIE_2.zip --expected-reps "$REPS" --expected-gpu "$POD_GPU"
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
  PG_CANDIDATE_ENV="$CANDIDATE_ENV" \
  PG_BASELINE_STATE="$BASELINE_STATE" PG_CANDIDATE_STATE="$CANDIDATE_STATE" \
  PG_ARCH_SOUNDNESS="$ARCHITECTURE_SOUNDNESS_GATE" \
  PG_ARCH_SOUNDNESS_SHA="$(sha256_file "$ARCHITECTURE_SOUNDNESS_GATE")" \
  PG_BASE_OUT="$base_out" PG_FLAG_OUT="$flag_out" \
  PG_BASE_SHA="$base_sha" PG_FLAG_SHA="$flag_sha" \
  PG_REMOTE_QUIESCENCE="$([[ "$BASELINE_QUIESCENCE_PASSED" == 1 && "$FLAGGED_QUIESCENCE_PASSED" == 1 ]] && echo 1 || echo 0)" \
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
with open(os.environ["PG_ARCH_SOUNDNESS"], encoding="utf-8") as stream:
    soundness = json.load(stream)
execution_target = soundness.get("execution_target")

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
    "candidate_env": os.environ.get("PG_CANDIDATE_ENV", ""),
    "baseline_state": json.loads(os.environ["PG_BASELINE_STATE"]),
    "candidate_state": json.loads(os.environ["PG_CANDIDATE_STATE"]),
    "mode": "qualification_probe",
    "provisional": True,
    "performance_admissible": False,
    "architecture_soundness": {
        "path": os.environ["PG_ARCH_SOUNDNESS"],
        "sha256": os.environ["PG_ARCH_SOUNDNESS_SHA"],
        "scope": "source and runtime architecture only; policy state is normalized per arm",
    },
    "execution_target": execution_target,
    "execution_target_sha256": soundness.get("execution_target_sha256"),
    "source_projection": soundness.get("source_projection"),
    "execution_guard_passed": os.environ["PG_STATUS"] == "ok",
    "remote_quiescence_passed": (
        os.environ["PG_STATUS"] == "ok"
        and os.environ["PG_REMOTE_QUIESCENCE"] == "1"
    ),
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
# Emits one "workdir|package|test_target|features" line per suite for the given lane.
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
      echo "${POD_PROVER_DIR}|stwo-cairo-gpu-prover|prepared_composition_native|direct-retention-test-api"
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
  local cmds="" names="" dir pkg target features feature_args
  while IFS='|' read -r dir pkg target features; do
    [[ -z "$dir" ]] && continue
    feature_args=""
    [[ -z "$features" ]] || feature_args=" --features ${features}"
    [[ -n "$cmds" ]] && cmds+=" && "
    cmds+="cd '${dir}' && cargo test -p ${pkg}${feature_args} --test ${target}"
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
  run_pod_job "parity_${lane}" "$body" || return 1

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
  local body
  BASELINE_QUIESCENCE_PASSED=0
  FLAGGED_QUIESCENCE_PASSED=0

  log "=== lane ${lane}: A/B SN_PIE_2 pair (reps=${REPS}) ==="

  local baseline_exports="unset ${lane}" candidate_exports="export ${lane}=1"
  if [[ -n "$BUNDLE_NAME" ]]; then
    baseline_exports="unset ${APPROVED_BUNDLE_BOOLEAN_FLAGS[*]} ${RETAINED_BUDGET_POLICY};"
    candidate_exports="$baseline_exports export ${CANDIDATE_ENV}"
  fi

  # Arm A: baseline flags explicitly OFF, matching production defaults.
  BASELINE_QUIESCENCE_PASSED=0
  body="$(bench_body "$baseline_exports" "$base_proof" "$args")" \
    || die "could not construct the sealed baseline launcher for ${lane}"
  run_pod_job "${lane}.baseline" "$body" || return 1
  validate_completed_arm
  BASELINE_QUIESCENCE_PASSED=1
  [[ "$DRY_RUN" == "1" ]] && synth_out "${lane}.baseline" "$LAST_OUT"
  local base_out="$LAST_OUT" base_rc="$LAST_RC" base_stall="$LAST_STALL_FILE"
  if [[ "$base_rc" != "0" ]]; then
    append_perf_ledger "$lane" "$([[ "$base_rc" == "STALLED" ]] && echo stalled || echo run_failed)" \
      "$base_out" "" "" "" ""
    warn "lane ${lane}: baseline arm failed (rc=${base_rc}${base_stall:+, stall evidence ${base_stall}})"
    return 1
  fi
  if ! proof_contract_ok "$base_out" || ! architecture_contract_ok "$base_out"; then
    append_perf_ledger "$lane" "run_failed" "$base_out" "" "" "" ""
    warn "lane ${lane}: baseline output contract failed"
    return 1
  fi

  # Arm B: one lane flag or the complete named candidate bundle.
  FLAGGED_QUIESCENCE_PASSED=0
  body="$(bench_body "$candidate_exports" "$flag_proof" "$args")" \
    || die "could not construct the sealed candidate launcher for ${lane}"
  run_pod_job "${lane}.flagged" "$body" || return 1
  validate_completed_arm
  FLAGGED_QUIESCENCE_PASSED=1
  [[ "$DRY_RUN" == "1" ]] && synth_out "${lane}.flagged" "$LAST_OUT"
  local flag_out="$LAST_OUT" flag_rc="$LAST_RC" flag_stall="$LAST_STALL_FILE"
  if [[ "$flag_rc" != "0" ]]; then
    append_perf_ledger "$lane" "$([[ "$flag_rc" == "STALLED" ]] && echo stalled || echo run_failed)" \
      "$base_out" "$flag_out" "" "" ""
    warn "lane ${lane}: flagged arm failed (rc=${flag_rc}${flag_stall:+, stall evidence ${flag_stall}})"
    return 1
  fi
  if ! proof_contract_ok "$flag_out" || ! architecture_contract_ok "$flag_out"; then
    append_perf_ledger "$lane" "run_failed" "$base_out" "$flag_out" "" "" ""
    warn "lane ${lane}: candidate output contract failed"
    return 1
  fi

  # FAIL CLOSED: cross-arm proof-byte identity. Every lane is documented
  # byte-identical; a mismatch is treated as a correctness failure of the lane.
  local fake_flag_sha="dryrun-SN_PIE_2-proof-sha256"
  [[ "${FAKE_PROOF_MISMATCH:-}" == "$lane" ]] \
    && fake_flag_sha="dryrun-SN_PIE_2-proof-sha256-DIFFERENT"
  local base_sha flag_sha
  base_sha="$(pod_sha256 "$base_proof" "dryrun-SN_PIE_2-proof-sha256")"
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
    local ncu_body ncu_exports="export ${lane}=1" ncu_guard ncu_quiescence_guard
    if [[ -n "$BUNDLE_NAME" ]]; then
      ncu_exports="unset ${APPROVED_BUNDLE_BOOLEAN_FLAGS[*]} ${RETAINED_BUDGET_POLICY}; export ${CANDIDATE_ENV}"
    fi
    ncu_guard="$(remote_execution_guard "${POD_PIE},${POD_BOOTLOADER_JSON}")" \
      || die "could not construct the sealed ncu launcher for ${lane}"
    ncu_quiescence_guard="$(remote_quiescence_guard)" \
      || die "could not construct the ncu isolation guard for ${lane}"
    ncu_body="$(cat <<EOF
cd '${POD_PROVER_DIR}'
. "\$HOME/.cargo/env" 2>/dev/null || true
export PATH=/usr/local/cuda/bin:\$PATH
export RUST_MIN_STACK=${RUST_MIN_STACK_VAL}
export STWO_BENCH_TRACE=json
export STWO_JIT_LOG=1
${BENCH_ENV:+export ${BENCH_ENV}}
export STWO_BOOTLOADER_JSON='${POD_BOOTLOADER_JSON}'
${ncu_exports}
${ncu_guard}
${ncu_quiescence_guard}
ncu --target-processes all \
    --kernel-name 'regex:${kregex}' \
    --launch-count ${NCU_LAUNCH_COUNT} \
    --set full -f -o '${pod_rep}' \
    /proc/self/fd/9 ${ncu_args}
EOF
)"
    log "lane ${lane}: ncu targeted capture (kernels ~ /${kregex}/, ${NCU_LAUNCH_COUNT} launches)"
    if ! run_pod_job "${lane}.ncu" "$ncu_body"; then
      warn "lane ${lane}: ncu remote launch failed — perf numbers above still stand"
    else
      validate_completed_arm
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

validate_remote_execution_target

echo "=== ledger: ${PERF_LEDGER} ==="
log "=== perf_gates done ==="
[[ "$FAILED" == "0" ]] || die "one or more lanes failed (see warnings + ${PERF_LEDGER})"
exit 0
