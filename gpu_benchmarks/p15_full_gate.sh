#!/bin/bash
# Combined P1–P5 validation + measurement round (the ROAD_TO_10MHZ gates).
#
# Run on a CUDA pod (H100-class for the scaling curve; any CUDA card for the
# gates). Everything logs to /root/p15.log. Order matters: correctness gates
# first, measurement only after they pass.
#
#   GATE A   stwo CUDA conformance suite (covers the P3 stream-pool lanes)
#   GATE B   P1 witness differential (STWO_CUDA_WITNESS_VERIFY=1) during a full
#            CUDA e2e prove — trace columns, rc count tables, interaction
#            columns + sums, all byte-compared against the host writers
#   GATE C   clean CUDA e2e: proof byte-identical to SIMD (the decisive gate)
#   GATE D   repeated-prove determinism (the P3 silent-corruption defense),
#            plus a STWO_CUDA_DISABLE_STREAMS isolation run
#   BENCH    post-P1 fib scaling curve (1M..8M) + P5 sustained throughput
#   M1       ncu roofline trace at 1M — ranks the P2 kernel worklist
set -uo pipefail
exec > >(tee /root/p15.log) 2>&1
export STWO_CUDA_NVCC=${STWO_CUDA_NVCC:-/usr/local/cuda/bin/nvcc}
export STWO_JIT_LOG=1
source "$HOME/.cargo/env" 2>/dev/null || true

STWO_REV=d9db63fcfdf2cfd1cb8fe82b0f66a473f38634e3
CAIRO_BRANCH=generic-backend

echo "=== SETUP ==="
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
  source "$HOME/.cargo/env"
fi
cd /root
[ -d stwo ] || git clone https://github.com/teddyjfpender/stwo -b perf-optimizations
[ -d stwo-cairo ] || git clone --depth 5 https://github.com/teddyjfpender/stwo-cairo -b "$CAIRO_BRANCH"
(cd stwo && git fetch origin && git checkout "$STWO_REV")
(cd stwo-cairo && git fetch origin "$CAIRO_BRANCH" && git reset --hard FETCH_HEAD)
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== GATE A: stwo CUDA conformance (incl. P3 stream pool) ==="
cd /root/stwo
cargo test -p stwo-backend-cuda --release 2>&1 | tail -15

echo "=== GATE B: P1 witness differential during full CUDA e2e ==="
cd /root/stwo-cairo/stwo_cairo_prover
STWO_CUDA_WITNESS_VERIFY=1 cargo test -p stwo-cairo-prover --release --features slow-tests \
  test_prove_verify_all_opcode_components_cuda -- --nocapture 2>&1 | tail -30

echo "=== GATE C: clean CUDA e2e — proof byte-identical to SIMD ==="
cargo test -p stwo-cairo-prover --release --features slow-tests \
  test_prove_verify_all_opcode_components_cuda 2>&1 | tail -8

echo "=== GATE D: repeated-prove determinism (P3) ==="
for i in 1 2 3; do
  echo "--- repeat $i ---"
  cargo test -p stwo-cairo-prover --release --features slow-tests \
    test_prove_verify_all_opcode_components_cuda 2>&1 | tail -3
done

cargo build --release -p stwo-cairo-prover --bin gpu_bench
BIN=$(pwd)/target/release/gpu_bench
FIB=/root/stwo-cairo/gpu_benchmarks/fib/compiled.json

echo "=== GATE D2: streams-off isolation cross-check ==="
STWO_CUDA_DISABLE_STREAMS=1 "$BIN" --program "$FIB" --iterations 1000000 --backend cuda --reps 2

echo "=== BENCH: post-P1 scaling curve ==="
for n in 1000000 2000000 4000000 8000000; do
  "$BIN" --program "$FIB" --iterations "$n" --backend cuda --reps 3 || break
done

echo "=== BENCH: memory-witness off (P1 isolation delta) ==="
STWO_CUDA_MEMORY_WITNESS=0 "$BIN" --program "$FIB" --iterations 1000000 --backend cuda --reps 3

echo "=== P5: sustained pipelined throughput ==="
"$BIN" --program "$FIB" --iterations 1000000 --backend cuda --reps 6 --pipeline

echo "=== M1: ncu roofline trace (ranks the P2 worklist) ==="
if command -v ncu >/dev/null; then
  STWO_BENCH_TRACE=0 ncu --set basic --launch-count 300 -f -o /root/m1_profile \
    "$BIN" --program "$FIB" --iterations 1000000 --backend cuda --reps 1 || true
  ncu --import /root/m1_profile.ncu-rep --csv --page details 2>/dev/null | head -100 || true
else
  echo "ncu not available; install nsight-compute for the M1 measurement"
fi

echo "=== DONE ==="
