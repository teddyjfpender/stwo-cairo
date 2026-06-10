# Cairo e2e GPU proving benchmarks (stwo-book format)

Methodology mirrors [zksecurity/zkvm-benchmarks](https://github.com/zksecurity/zkvm-benchmarks)'
stwo runner (the source of the [stwo-book benchmark tables](https://zksecurity.github.io/stwo-book/benchmarks/index.html)):
same Cairo programs and `program_input` hint, Cairo VM in proof mode, secure prover
configuration (`pow_bits=26`, blowup 1, 70 FRI queries — ~96 bits), preprocessed trace
`CanonicalWithoutPedersen`, proof size via bincode, cycle count = sum of opcode counts.
Harness: `crates/prover/src/bin/gpu_bench.rs`.

**Hardware**: RunPod secure-cloud **H100 SXM 80 GB**, 208-vCPU host, CUDA 11.8.
**Warm** times exclude the per-process NVRTC compile of the JIT constraint kernels
(~20–40 s once per process per statement shape); **cold** includes it. The SIMD rows
ran on the same host's 208 vCPUs for a like-for-like comparison. stwo-book CPU rows
(48-vCPU EPYC-Rome, July 2025 stwo) included for reference.

## Fibonacci (`n` iterations)

| n | backend | cycle count | prove warm (s) | prove cold (s) | verify (ms) | proof (KB) | peak RSS (GB) | peak VRAM (GB) | steps/s |
|---|---|---|---|---|---|---|---|---|---|
| 65,536 | **CUDA H100** | 458,768 | **13.57** | 35.9 | 6.0 | 1,097 | 1.7 | 4.8 | 33.8 k |
| 1,048,576 | **CUDA H100** | 7,340,048 | **30.23** | 52.0 | 6.2 | 1,242 | 10.2 | 10.9 | 242.8 k |
| 4,194,304 | **CUDA H100** | 29,360,144 | **98.78** | 120.7 | 6.7 | 1,375 | 31.9 | 36.9 | 297.2 k |
| 1,048,576 | SIMD (208 vCPU, same host) | 7,340,048 | 13.11 | 13.9 | 5.9 | 1,242 | 15.2 | — | 560.0 k |
| 65,536 | stwo-book CPU (48 vCPU) | — | 11.37 | — | — | — | — | — | — |
| 1,048,576 | stwo-book CPU (48 vCPU) | — | 18.61 | — | — | — | — | — | — |
| 4,194,304 | stwo-book CPU (48 vCPU) | — | 60.94 | — | — | — | — | — | — |

## Matrix multiplication (`n`×`n`)

| n | backend | cycle count | prove warm (s) | prove cold (s) | verify (ms) | proof (KB) | peak RSS (GB) | peak VRAM (GB) | steps/s |
|---|---|---|---|---|---|---|---|---|---|
| 32 | **CUDA H100** | 705,179 | **13.77** | 34.6 | 5.8 | 1,087 | 1.8 | 4.8 | 51.2 k |
| 64 | **CUDA H100** | 5,440,763 | **19.58** | 41.5 | 5.8 | 1,165 | 6.1 | 6.4 | 277.8 k |
| 64 | SIMD (208 vCPU, same host) | 5,440,763 | 7.41 | 8.0 | 5.8 | 1,165 | 10.3 | — | 734.3 k |

## EC add, secp256k1 (`n` operations)

| n | backend | cycle count | prove warm (s) | prove cold (s) | verify (ms) | proof (KB) | peak RSS (GB) | peak VRAM (GB) | steps/s |
|---|---|---|---|---|---|---|---|---|---|
| 256 | **CUDA H100** | 59,938 | **16.55** | 55.7 | 7.1 | 1,189 | 1.0 | 4.8 | 3.6 k |
| 1,024 | **CUDA H100** | 239,650 | **17.01** | 55.2 | 6.5 | 1,197 | 1.4 | 4.8 | 14.1 k |
| 1,024 | SIMD (208 vCPU, same host) | 239,650 | 5.22 | 5.4 | 6.4 | 1,197 | 6.1 | — | 45.9 k |

## SHA2-chain (`n` chained hashes)

| n | backend | cycle count | prove warm (s) | prove cold (s) | verify (ms) | proof (KB) | peak RSS (GB) | peak VRAM (GB) | steps/s |
|---|---|---|---|---|---|---|---|---|---|
| 64 | **CUDA H100** | 23,058 | **15.98** | 52.9 | 7.4 | 1,221 | 1.0 | 4.8 | 1.4 k |

## Honest reading

- **The v1 CUDA lane does not yet beat a strong CPU on real Cairo e2e proving.** On
  this 208-vCPU host, SIMD is 2.3–3.3× faster across these workloads; the stwo-book's
  48-vCPU rows also beat the H100 at every fib size. Real Cairo proofs are dominated by
  the ~46-component pipeline's per-component overheads (per-launch synchronization,
  host roundtrips, witness on CPU + transfer), not by the bulk math the GPU wins at —
  the same conclusion as the backend's microbenchmarks, where the GPU *does* win 2×+
  against weaker hosts on single-component AIRs.
- **GPU throughput scales with size** (fib: 34 k → 243 k → 297 k steps/s from 65 k to
  4 M cycles) while its absolute floor (~13.5 s at small n, all programs) is fixed
  overhead — small workloads are entirely floor. The crossover vs strong CPUs lies
  beyond these sizes and/or after the documented v1 headroom is removed (stream
  pipelining, witness-on-GPU, batched commits, cumsum parameterization to kill the
  20–40 s cold NVRTC cost).
- **Memory**: fib 4 M peaked at 36.9 GB VRAM (an 80 GB card is comfortable; 24 GB
  consumer cards would need the low-memory mode) and ~32 GB host RSS. Proof sizes and
  ~6 ms verification match the CPU backend byte-for-byte, as gated by the conformance
  suite.

## Utilization investigation (fib 1M, RTX 3090 pod)

`nvidia-smi dmon` during the CUDA prove: **median SM utilization 0%, max 100%** — the
GPU is idle most of the prove. Per-phase span totals (2 reps) confirm the bulk math
already wins on GPU and two host-bound phases dominate:

| phase (fib 1M, per 2 reps) | CUDA | SIMD (same host) | verdict |
|---|---|---|---|
| OODS column evaluation | 22.4 s | 4.7 s | **host-roundtrip barycentric — top fix: GPU batch OODS** |
| Composition (constraint eval) | 33.6 s (incl. ~17 s one-time NVRTC) | 16.1 s | cumsum param kills the NVRTC share |
| Commitments (Merkle+NTT) | 1.2 s | 5.2 s | GPU wins 4× |
| FRI quotients | 0.2 s | 0.7 s | GPU wins |
| Interpolation | 0.1 s | 0.8 s | GPU wins |
| PoW grind (pow_bits 26) | 0.04 s | 2.2 s | GPU wins ~60× |

Ranked fixes: (1) batch OODS evaluation on GPU (the staged `batch_eval_at_point`
kernels; NitrooZK report 67× on this exact phase) — removes ~11 s/prove and alone puts
CUDA ahead of same-host SIMD; (2) parameterize the logup cumsum to stop per-statement
NVRTC recompiles (~17 s cold); (3) stream pipelining to remove per-launch
synchronization; (4) witness-on-GPU / pinned transfers.

## Optimized results — round 1 (RTX 3090 pod, same-host CUDA vs SIMD)

After the OODS weights fix (parallel + batch inversion) and statement-independent
NVRTC kernels — proofs byte-identical, all gates green. Community-pod host CPUs vary
wildly between sessions, so only same-pod comparisons are meaningful:

| program | n | cycles | CUDA warm (s) | SIMD warm (s) | verdict |
|---|---|---|---|---|---|
| fib | 1,048,576 | 7.34 M | **19.1** | 30.2 | **GPU 1.6×** |
| mat_mul | 64 | 5.44 M | **13.3** | 16.6 | **GPU 1.25×** |
| fib | 65,536 | 0.46 M | 9.8 | 8.1 | SIMD (floor-bound) |
| ec | 1,024 | 0.24 M | 12.4 | 10.3 | SIMD (floor-bound) |
| mat_mul | 32 | 0.71 M | 9.6 | ~8 | SIMD (floor-bound) |

**Where the remaining GPU time goes** (optimized fib 1M, per rep): Composition 13.5 s
(now dominant — the JIT kernels are suspected register-bound from full unrolling;
`__launch_bounds__`/register budgeting is the round-2 fix), OODS 4.7 s (down from
11.2 s — now equal to SIMD's), base+interaction witness on CPU 5.0 s (witness-on-GPU /
pinned transfers), commits ~2 s.

**Pattern**: the GPU wins where proving time actually hurts (multi-million-cycle
workloads) and loses small workloads to its ~10 s fixed floor — dominated by
preprocessed-tree construction and first-touch costs that NitrooZK eliminate with
prove-cycle caches (their warm small-PIE is 0.25 s). Porting that caching layer (with
explicit content keys — their implicit-key cache has an aliasing hazard) is the
known fix for the floor.

**Known issue**: fib at n=4,194,304 fails on this backend with "Error copying memory:
invalid argument" — a u32 length overflow in the flat-trace device copy at log≥23
sizes; fix queued (u64 lengths or chunked copies).

**Normalized vs the NitrooZK base** (their RTX 5090: ~0.9 s per 1M VM steps warm):
ours is now ~2.6 s per 1M steps on an RTX 3090 — within ~1.5–2× after accounting for
the GPU generation gap, with composition register pressure, witness-on-GPU, batch NTT,
and warm caches as the remaining levers to close and pass it.

## Optimized results — round 2 (RTX 3090 pod, warm incl. prove-cycle caches)

Round-2 levers: JIT **register compaction** (linear-scan reuse; the recorder's
monotonic SSA allocation was spilling large kernels to local memory) +
`__launch_bounds__(128)`; **pointer-table trace ABI** (no flatten copies, no u32
overflow); **prove-cycle caches with explicit content keys** (twiddles, preprocessed
tree). All gates green (conformance byte-equality, differential verify, Cairo e2e).

| program | n | R1 CUDA (s) | **R2 CUDA (s)** | R2 SIMD same-host (s) | verdict |
|---|---|---|---|---|---|
| fib | 1,048,576 | 19.1 | **13.2** | 24.3 | **GPU 1.84×** |
| mat_mul | 64 | 13.3 | **9.4** | 11.7 | **GPU 1.24×** |
| mat_mul | 32 | 9.6 | **6.7** | — | — |
| fib | 65,536 | 9.8 | **6.9** | 2.7 | SIMD 2.5× |
| ec | 1,024 | 12.4 | **8.8** | 5.2 | SIMD 1.7× |
| ec | 256 | 12.0 | **8.5** | — | — |

Notes:
- CUDA improved ~30% across every program; verify dropped to ~4 ms.
- The caches are backend-generic, so **SIMD's small-n floor collapsed too** (fib 65k:
  8.1 → 2.7 s) — honest accounting: the cache lever lifted both backends.
- The remaining GPU small-n floor (~6.8 s) is per-launch synchronization, per-column
  OODS launches, and CPU witness + transfer — the queued round-3 items (stream
  pipelining, batched OODS, witness-on-GPU, parallel NVRTC for cold).
- fib 4M exceeds 24 GB VRAM on the 3090 (peaks in the quotient/FRI phase, which tree
  compaction does not cover) — needs the full host-spill streaming (L1) or a ≥40 GB
  card; the H100 ran it in 36.9 GB.
- 5090/4090 were out of stock for this session; the multi-arch fatbin binary makes a
  5090 rerun a minutes-long job when stock returns.

**Journey on fib 1M (same GPU class, same-host SIMD)**: v1 30.2 s CUDA-loses →
round 1 19.1 s (GPU 1.6×) → round 2 **13.2 s (GPU 1.84×)** ≈ 1.8 s per 1M VM steps —
at NitrooZK's published 5090 base (~0.9 s/1M steps) once adjusted for the GPU
generation gap.
