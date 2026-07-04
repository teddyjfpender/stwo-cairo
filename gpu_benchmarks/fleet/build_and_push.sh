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
#   3. push target/release/gpu_bench + ~/.cache/stwo-jit to every pod in pods.conf
#   4. stamp provenance (git revs + dirty hashes) alongside the binary on each pod
set -euo pipefail
cd "$(dirname "$0")"
KEY="${POD_KEY:-/Users/theodorepender/.runpod/ssh/runpodctl-ssh-key}"
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
  rsync -az --partial -e "ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=target --exclude=.git \
    /Users/theodorepender/code/personal/stwo/ "root@$BHOST:/workspace/stwo/"
  rsync -az --partial -e "ssh -i $KEY -p $BPORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=target --exclude=.git \
    --exclude="gpu_benchmarks/pie/sn/*.zip" \
    /Users/theodorepender/code/personal/stwo-cairo/ "root@$BHOST:/workspace/stwo-cairo/"
  $SSHB 'sed -i "s|/Users/theodorepender/code/personal/stwo|/workspace/stwo|g" /workspace/stwo-cairo/stwo_cairo_prover/Cargo.toml'
fi

STWO_REV=$(git -C /Users/theodorepender/code/personal/stwo rev-parse --short HEAD)
CAIRO_REV=$(git -C /Users/theodorepender/code/personal/stwo-cairo rev-parse --short HEAD)
STWO_DIRTY=$(git -C /Users/theodorepender/code/personal/stwo diff | shasum -a 256 | cut -c1-12)
CAIRO_DIRTY=$(git -C /Users/theodorepender/code/personal/stwo-cairo diff | shasum -a 256 | cut -c1-12)

echo "== building (v3, pie-bench) =="
# STWO_CUDA_ARCH: multi-arch fatbin so ONE binary runs device code on every fleet
# card (sm_86 = 3090/A40, sm_89 = 4090). Without it, detect_arch() targets only the
# builder's GPU and other cards die with cudaErrorSymbolNotFound (measured 2026-07-02).
# Extend the list when a new arch joins the roster (5090 = sm_120).
# Resolve the arch list LOCALLY (a remote ${VAR:-default} in a single-quoted ssh
# command ignores the caller's env — builds the wrong fatbin silently).
ARCH_LIST="${STWO_CUDA_ARCH:-sm_86,sm_89}"
echo "   arch: $ARCH_LIST"
$SSHB '. $HOME/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH
cd /workspace/stwo-cairo/stwo_cairo_prover
STWO_CUDA_ARCH='"$ARCH_LIST"' RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p stwo-cairo-prover --bin gpu_bench --features pie-bench 2>&1 | tail -1
ls -la target/release/gpu_bench'

echo "== pushing to fleet =="
for p in "${PODS[@]}"; do
  ID=$(echo "$p" | cut -d"|" -f1); HOST=$(echo "$p" | cut -d"|" -f2); PORT=$(echo "$p" | cut -d"|" -f3)
  [ "$HOST" = "$BHOST" ] && { echo "  $ID: builder (in place)"; continue; }
  (
    # Builder pushes directly pod-to-pod via the account key already in the agent
    # (run this script under ssh -A; nothing is persisted on any pod).
    ssh -A -i "$KEY" -p "$BPORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$BHOST" \
      "rsync -az -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT' \
        /workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench \
        root@$HOST:/workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench && \
       rsync -az -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT' \
        /root/.cache/stwo-jit/ root@$HOST:/root/.cache/stwo-jit/" \
      && echo "  $ID: binary + jit cache pushed"
    ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$HOST" \
      "echo 'stwo=$STWO_REV+$STWO_DIRTY cairo=$CAIRO_REV+$CAIRO_DIRTY built=$(date -u +%FT%TZ) flags=v3' > /workspace/stwo-cairo/stwo_cairo_prover/target/release/gpu_bench.provenance"
  ) &
done
wait
echo "== fleet updated: one build, every pod current. Run gates via loop/bench_loop.sh --skip-sync =="
