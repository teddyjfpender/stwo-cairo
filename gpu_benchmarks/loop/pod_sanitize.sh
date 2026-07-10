#!/usr/bin/env bash
#
# pod_sanitize.sh — run a native CUDA test under compute-sanitizer on the pod.
#
# The standard verification for device-side memory faults (CUDA 719 class):
# memcheck localizes the exact faulting kernel, thread and address instead of
# the sticky error surfacing at a later API call. Detached (nohup+rc sentinel),
# so SSH drops cannot kill the run.
#
# Usage:
#   BENCH_POD_ID=<id> ./pod_sanitize.sh <test-target> [test-filter] [--racecheck]
# Example:
#   BENCH_POD_ID=9wx... ./pod_sanitize.sh resident_parity_native strict_resident_cold
#
# Results: /workspace/bench_loop_runs/sanitize.log (+ .rc sentinel) on the pod.

set -euo pipefail

TARGET="${1:?usage: pod_sanitize.sh <test-target> [filter] [--racecheck]}"
FILTER="${2:-}"
TOOL="memcheck"
[[ "${3:-}" == "--racecheck" || "${2:-}" == "--racecheck" ]] && TOOL="racecheck"
[[ "${2:-}" == "--racecheck" ]] && FILTER=""

POD_ID="${BENCH_POD_ID:?BENCH_POD_ID required}"
INFO=$(runpodctl ssh info "$POD_ID")
HOST=$(printf '%s' "$INFO" | sed -n 's/.*"ip": "\([^"]*\)".*/\1/p')
PORT=$(printf '%s' "$INFO" | sed -n 's/.*"port": \([0-9]*\).*/\1/p')
KEY=$(printf '%s' "$INFO" | sed -n 's/.*"path": "\([^"]*\)".*/\1/p' | head -1)

RUN_DIR=/workspace/bench_loop_runs
ssh -i "$KEY" -o ConnectTimeout=10 "root@${HOST}" -p "$PORT" "
  cd /workspace/stwo-cairo/stwo_cairo_prover && rm -f ${RUN_DIR}/sanitize.rc &&
  nohup bash -c '
    . \$HOME/.cargo/env
    export PATH=/usr/local/cuda/bin:\$PATH RUST_MIN_STACK=16777216 \
           STWO_CUDA_OBJ_CACHE=/workspace/.cuda_obj_cache RUST_BACKTRACE=1
    cargo test -p stwo-cairo-gpu-prover --test ${TARGET} --no-run 2>&1 | tail -2
    BIN=\$(ls -t target/debug/deps/${TARGET}-* | grep -v \"\\.\" | head -1)
    compute-sanitizer --tool ${TOOL} --launch-timeout 120 --error-exitcode 42 \
      \"\$BIN\" ${FILTER} --test-threads=1 --nocapture
    echo EXIT=\$? > ${RUN_DIR}/sanitize.rc
  ' > ${RUN_DIR}/sanitize.log 2>&1 &
  echo LAUNCHED"
echo "[pod_sanitize] detached; poll ${RUN_DIR}/sanitize.rc, log ${RUN_DIR}/sanitize.log"
