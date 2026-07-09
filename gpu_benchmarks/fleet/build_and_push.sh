#!/bin/bash
# Build ONCE on the designated builder pod, push the portable binary to every
# fleet pod. Kills the per-pod-rebuild tax (7-15 min x N pods -> one build +
# seconds of datacenter-link copies). Binaries are x86-64-v3 (portable across
# fleet hosts — native SIGILLs, see pods.conf rules); the JIT PTX/cubin caches
# are pushed too (sm-tagged, arch-safe).
#
# Usage: ./build_and_push.sh [--builder POD_ID] [--skip-sync]
#   1. rsync local stwo + stwo-cairo to the builder (fast delta)
#   2. cargo build gpu_bench (v3, pie-bench) on the builder's cores
#   3. preflight/seed a pinned bootloader at one stable remote path
#   4. push gpu_bench + bootloader + ~/.cache/stwo-jit to every pod in pods.conf
#   5. stamp provenance (git revs + dirty hashes) alongside the binary on each pod
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="${CAIRO_LOCAL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STWO_LOCAL="${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}"
INPUT_SHA256SUMS="${CAIRO_LOCAL}/gpu_benchmarks/pie/SHA256SUMS"
POD_BOOTLOADER_JSON="${POD_BOOTLOADER_JSON:-/workspace/bench_inputs/simple_bootloader_compiled.json}"
BOOTLOADER_JSON_SOURCE="${BOOTLOADER_JSON_SOURCE:-}"
[[ -d "$CAIRO_LOCAL/.git" ]] || { echo "missing stwo-cairo checkout: $CAIRO_LOCAL" >&2; exit 1; }
[[ -d "$STWO_LOCAL/.git" ]] || { echo "missing sibling stwo checkout: $STWO_LOCAL" >&2; exit 1; }
[[ -f "$INPUT_SHA256SUMS" ]] || { echo "input checksum manifest missing: $INPUT_SHA256SUMS" >&2; exit 1; }
STWO_LOCAL="$(cd "$STWO_LOCAL" && pwd)"
cd "$SCRIPT_DIR"
KEY="${POD_KEY:-${HOME}/.runpod/ssh/runpodctl-ssh-key}"
BUILDER_ID="${2:-}"; [ "${1:-}" = "--builder" ] || BUILDER_ID=""
SKIP_SYNC=0; for a in "$@"; do [ "$a" = "--skip-sync" ] && SKIP_SYNC=1; done

# Roster: ID|HOST|PORT|GPU|... — first line is the default builder (4090, 64 vCPU).
# (while-read, not mapfile: macOS /bin/bash is 3.2.)
PODS=()
while IFS= read -r line; do PODS+=("$line"); done < <(grep -v "^#" pods.conf | grep "|")
[ ${#PODS[@]} -gt 0 ] || { echo "no pods in pods.conf"; exit 1; }
BUILDER="${PODS[0]}"
if [ -n "$BUILDER_ID" ]; then
  for p in "${PODS[@]}"; do [[ "$p" == "$BUILDER_ID|"* ]] && BUILDER="$p"; done
fi
BHOST=$(echo "$BUILDER" | cut -d"|" -f2); BPORT=$(echo "$BUILDER" | cut -d"|" -f3)
SSHB="ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$BHOST"

echo "== builder: $(echo "$BUILDER" | cut -d"|" -f1,4) =="
if [ "$SKIP_SYNC" = 0 ]; then
  rsync -rlptz --partial --no-owner --no-group -e "ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=target --exclude=.git \
    "$STWO_LOCAL/" "root@$BHOST:/workspace/stwo/"
  rsync -rlptz --partial --no-owner --no-group -e "ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=target --exclude=.git \
    --exclude="gpu_benchmarks/pie/sn/*.zip" \
    "$CAIRO_LOCAL/" "root@$BHOST:/workspace/stwo-cairo/"
  # rsync -t preserves LOCAL mtimes; if they predate the pod's build artifacts,
  # cargo sees the path-dep source as up-to-date and SKIPS the rebuild (served a
  # stale binary for hours — 2026-07-06). Bump mtimes past the artifacts so cargo
  # always recompiles changed crates.
  $SSHB 'find /workspace/stwo/crates /workspace/stwo-cairo/stwo_cairo_prover/crates -name "*.rs" -newermt "1970-01-01" -exec touch {} + 2>/dev/null; true'
fi

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else LC_ALL=C shasum -a 256; fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    LC_ALL=C shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

expected_sha256() {
  awk -v file="$(basename "$1")" '$2 == file { print $1; exit }' "$INPUT_SHA256SUMS"
}

BOOTLOADER_SHA="$(expected_sha256 "$POD_BOOTLOADER_JSON")"
[[ -n "$BOOTLOADER_SHA" ]] \
  || { echo "pinned bootloader checksum missing from $INPUT_SHA256SUMS: $(basename "$POD_BOOTLOADER_JSON")" >&2; exit 1; }
BOOTLOADER_DIR="$(dirname "$POD_BOOTLOADER_JSON")"

echo "== bootloader: $POD_BOOTLOADER_JSON ($BOOTLOADER_SHA) =="
if [[ -n "$BOOTLOADER_JSON_SOURCE" ]]; then
  [[ -f "$BOOTLOADER_JSON_SOURCE" ]] \
    || { echo "bootloader source missing: $BOOTLOADER_JSON_SOURCE" >&2; exit 1; }
  BOOTLOADER_ACTUAL="$(sha256_file "$BOOTLOADER_JSON_SOURCE")"
  [[ "$BOOTLOADER_ACTUAL" == "$BOOTLOADER_SHA" ]] \
    || { echo "bootloader source SHA-256 mismatch: expected $BOOTLOADER_SHA, got $BOOTLOADER_ACTUAL" >&2; exit 1; }
  $SSHB "mkdir -p '$BOOTLOADER_DIR'"
  rsync -rlpt --no-owner --no-group \
    -e "ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$BOOTLOADER_JSON_SOURCE" "root@$BHOST:$POD_BOOTLOADER_JSON"
fi
$SSHB "path='$POD_BOOTLOADER_JSON'; expected='$BOOTLOADER_SHA'; \
  if [ ! -f \"\$path\" ]; then echo \"missing pinned bootloader: \$path (set BOOTLOADER_JSON_SOURCE to seed it)\" >&2; exit 1; fi; \
  if command -v sha256sum >/dev/null 2>&1; then actual=\$(sha256sum \"\$path\" | cut -d' ' -f1); else actual=\$(LC_ALL=C shasum -a 256 \"\$path\" | cut -d' ' -f1); fi; \
  [ \"\$actual\" = \"\$expected\" ] || { echo \"bootloader SHA-256 mismatch: \$path (expected \$expected, got \$actual)\" >&2; exit 1; }"

STWO_REV=$(git -C "$STWO_LOCAL" rev-parse --short HEAD)
CAIRO_REV=$(git -C "$CAIRO_LOCAL" rev-parse --short HEAD)
STWO_DIRTY=$(git -C "$STWO_LOCAL" diff | sha256_stream | cut -c1-12)
CAIRO_DIRTY=$(git -C "$CAIRO_LOCAL" diff | sha256_stream | cut -c1-12)

echo "== building (v3, pie-bench) =="
# STWO_CUDA_ARCH: multi-arch fatbin so ONE binary runs device code on every fleet
# card (sm_86 = 3090/A40, sm_89 = 4090, sm_90 = H100). Without it, detect_arch()
# targets only the builder's GPU and other cards die with cudaErrorSymbolNotFound
# (measured 2026-07-02). Extend the list when a new arch joins the roster
# (5090 = sm_120).
# Resolve the arch list LOCALLY (a remote ${VAR:-default} in a single-quoted ssh
# command ignores the caller's env — builds the wrong fatbin silently).
ARCH_LIST="${STWO_CUDA_ARCH:-sm_86,sm_89,sm_90}"
echo "   arch: $ARCH_LIST"
# Locally-resolved extra nvcc flags (e.g. -DSTWO_LEAF_MIN_BLOCKS=3 to sweep an
# occupancy hint). Resolved here so the remote build sees the caller's env (a bare
# remote ${VAR} in the single-quoted ssh body would ignore it). Forces a rebuild of
# the touched .cu (rsync's source touch already invalidates), so a flag change takes.
NVCC_FLAGS="${STWO_CUDA_NVCC_FLAGS:-}"
[ -n "$NVCC_FLAGS" ] && echo "   nvcc_flags: $NVCC_FLAGS"
$SSHB '. $HOME/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH
cd /workspace/stwo-cairo/stwo_cairo_prover
set -o pipefail
STWO_BOOTLOADER_JSON='"$POD_BOOTLOADER_JSON"' STWO_CUDA_NVCC_FLAGS='"$NVCC_FLAGS"' STWO_CUDA_ARCH='"$ARCH_LIST"' RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p stwo-cairo-gpu-prover --bin gpu_bench --features pie-bench 2>&1 | grep -E "error|Finished" | tail -20
# Surface a build failure instead of swallowing it behind a stale binary (the
# prior `| tail -1` hid a parallel-feature compile error for hours).
if [ "${PIPESTATUS[0]:-0}" != 0 ]; then echo "BUILD FAILED"; exit 1; fi
ls -la target/release/gpu_bench'

echo "== pushing to fleet =="
PUSH_PIDS=()
for p in "${PODS[@]}"; do
  ID=$(echo "$p" | cut -d"|" -f1); HOST=$(echo "$p" | cut -d"|" -f2); PORT=$(echo "$p" | cut -d"|" -f3)
  [ "$HOST" = "$BHOST" ] && { echo "  $ID: builder (in place)"; continue; }
  (
    # Builder pushes directly pod-to-pod via the account key already in the agent
    # (run this script under ssh -A; nothing is persisted on any pod).
    if ! ssh -A -i "$KEY" -p "$BPORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$BHOST" \
      "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$HOST \
        'mkdir -p /workspace/stwo-cairo/stwo_cairo_prover/target/release /root/.cache/stwo-jit $BOOTLOADER_DIR' && \
       rsync -rlptz --no-owner --no-group -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT' \
        /workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench \
        root@$HOST:/workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench && \
       rsync -rlpt --no-owner --no-group -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT' \
        '$POD_BOOTLOADER_JSON' root@$HOST:'$POD_BOOTLOADER_JSON' && \
       rsync -rlptz --no-owner --no-group -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT' \
        /root/.cache/stwo-jit/ root@$HOST:/root/.cache/stwo-jit/"; then
      echo "  $ID: binary/bootloader/JIT push failed" >&2
      exit 1
    fi
    echo "  $ID: binary + pinned bootloader + jit cache pushed"
    ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$HOST" \
      "path='$POD_BOOTLOADER_JSON'; expected='$BOOTLOADER_SHA'; \
       if command -v sha256sum >/dev/null 2>&1; then actual=\$(sha256sum \"\$path\" | cut -d' ' -f1); else actual=\$(LC_ALL=C shasum -a 256 \"\$path\" | cut -d' ' -f1); fi; \
       [ \"\$actual\" = \"\$expected\" ] || { echo \"bootloader SHA-256 mismatch after push: \$path\" >&2; exit 1; }; \
       echo 'stwo=$STWO_REV+$STWO_DIRTY cairo=$CAIRO_REV+$CAIRO_DIRTY built=$(date -u +%FT%TZ) flags=v3 bootloader_sha256=$BOOTLOADER_SHA' > /workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench.provenance"
  ) &
  PUSH_PIDS+=("$!")
done
PUSH_FAILED=0
for pid in "${PUSH_PIDS[@]}"; do
  wait "$pid" || PUSH_FAILED=1
done
[[ "$PUSH_FAILED" == "0" ]] || { echo "one or more fleet pushes failed" >&2; exit 1; }
echo "== fleet updated: one build, every pod current. Run gates via loop/bench_loop.sh --skip-sync =="
