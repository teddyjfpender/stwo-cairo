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

## Round 3: the stream-ordered rebuild (+ saturation pass)

Implements the full 5-point plan from `docs/gpu-architecture-analysis.md` in the stwo
fork (commits `25798d20`..`5d1d851e`): **stream-ordered execution** (all 82 wrapper
device-syncs removed; allocator without per-alloc private-stream churn; host reads
fenced by synchronous `cudaMemcpy` by construction), **statement-independent JIT
kernels + on-disk PTX cache** (all ext constants hoisted to runtime params; kernels
shared across statements, inputs, and processes), **batched `evaluate_polynomials`**
(one multi-column NTT per size group), **pinned witness staging** (parallel zero-copy
packing from the SIMD columns), and the **waste bundle** (fused in-kernel accumulate,
device-side split/join/twiddle-extract, no uninitialized-clone FRI buffers, borrowed
quotient twiddles). Gates green at every step: conformance byte-equality + repeated
prove on both channels, Cairo e2e proof byte-identical to SIMD.

**Same-host interleaved A/B (old = round-2 code, new = rebuild; RTX 3090, 16 vCPU):**

| program | n | warm old → new | cold old → new | cold speedup |
|---|---|---|---|---|
| fib | 1,048,576 | 17.2 → 16.8 s | 37.4 → 18.7 s | **2.0×** |
| fib | 65,536 | 9.7 → 9.4 s | 36.7 → 10.2 s | **3.6×** |
| ec | 1,024 | 12.8 → 12.4 s | 47.7 → 13.5 s | **3.5×** |

**The headline is cold-start: 2–4.7× faster.** First-prove latency is now warm+1–3 s
instead of +20–50 s — the disk PTX cache means a fresh process (or a fresh statement,
or a different input) reuses compiled kernels (`STWO_JIT_LOG=1` shows 1–3 ms
disk-cache hits vs ~1.8 s NVRTC compiles). Warm proves improved a real but modest
2–9%: at these sizes on this hardware the warm path is dominated by GPU compute and
host witness phases, not launch latency — the sync-census prediction overweighted
steady-state and underweighted cold.

Honest host caveat: round-2 and round-3 ran on *different* 3090 community hosts. This
host's CPU is ~2× faster (SIMD fib 1M: 12.3 s vs 24.3 s), which compresses the
CUDA-vs-SIMD gap here (SIMD wins warm on this host; CUDA won on the round-2 host).
The cross-backend verdict is host-dependent at 3090 scale; the rebuild's wins
(cold-start, byte-equal correctness, removed PCIe roundtrips) hold on both.

Saturated final numbers (this host, warm/cold): fib 1M 16.6/19.3 s, fib 65k
9.5/10.6 s, mat_mul 64 12.0/13.9 s, ec 1024 12.5/12.9 s. Saturation pass adds
zero-copy witness packing and borrowed quotient twiddles; surveyed-and-rejected
levers (with reasoning) are in stwo commit `5d1d851e`.

## Round 4: trace-driven — the floor was decommit, then OODS weights

A phase-trace profile (`STWO_BENCH_TRACE=1` spans + 500 ms GPU-utilization sampling)
attributed the warm prove precisely: the GPU was idle 75-93% of the time, and the
dominant cost was an UNINSTRUMENTED post-grind phase — decommit — at 6.7-9.7 s,
nearly size-independent (the long-suspected "GPU floor"). Second was OODS sampling
(3.8 s at fib 1M). Composition was already solved (78 ms vs SIMD's 1,990 ms — the
JIT lane is 25x SIMD there).

Two fixes, each gated by conformance + Cairo e2e byte-equality:

1. **Batched decommit gathers** (stwo `ffafce22`): the dense decommit read queried
   values via `Column::at` per (column, row) — one 4-byte synchronous PCIe roundtrip
   each, ~100k per prove. Now: `Column::gather_unreduced` (one gather kernel + one
   D2H per column) feeding the existing sparse `decommit_gathered` path.
2. **Device-side OODS weights** (stwo `345c68c3`): barycentric weights per unique
   (log_size, point) generated and inverted millions of circle points ON THE HOST.
   Now the whole pipeline runs on device, reusing the quotient kernels'
   conformance-proven point generator. (The first kernel attempt used coordinate-wise
   point subtraction instead of the circle group law — the conformance differential
   rejected it instantly; the gate works.)

**Same host (RTX 3090, the round-3 trace pod), warm / cold:**

| program | n | start of session | + decommit fix | + OODS fix | total speedup |
|---|---|---|---|---|---|
| fib | 1,048,576 | 14.7 / 40.0 s | 8.0 / 10.0 s | **5.37 / 8.26 s** | **2.7x warm** |
| fib | 65,536 | 8.1 s | 1.68 s | **0.87 / 1.94 s** | **9.3x warm** |
| ec | 1,024 | 11.2 s | 1.90 s | **1.36 / 2.34 s** | **8.2x warm** |

- fib 1M: **1.37 MHz** (1,367,453 steps/s) — **2.1x same-host SIMD** (11.3 s / 0.65
  MHz), and at **0.73 s per 1M VM steps on a 3090 this clears NitrooZK's published
  0.9 s/1M-steps from an RTX 5090**.
- The small-n floor collapsed: fib 65k warm 0.87 s, ec 1024 warm 1.36 s — CUDA now
  wins every workload measured, at every size, on this host.
- **Prove STARKs core: 10.7 s -> 813 ms (13x).** The warm prove is now dominated by
  Cairo witness generation on the host (trace writing + adapt) — the next frontier
  is witness-born-on-GPU, an XL item that would go beyond NitrooZK (their witness is
  also host-generated).
- VRAM at fib 1M rose to ~18.8 GB peak (cached OODS weight columns per (log_size,
  point)); fine on 24 GB, worth watching at log 23+.

**Journey on fib 1M, same GPU class**: v1 30.2 s (CUDA loses) -> R1 19.1 s -> R2
13.2 s -> R3 rebuild (cold 2-4.7x, warm parity) -> **R4 5.37 s, 1.37 MHz, 2.1x over
same-host SIMD**. Measure, then optimize: the two fixes that mattered most were
invisible until the phase trace.

## Round 5: witness-on-GPU W1 — device-finalized interaction trace

First witness phase moved onto the device (design: `WITNESS_ON_GPU.md`; this is the
step beyond NitrooZK, whose witness is fully host-generated). All 66 uniform
generated writers now emit raw (numerator, denominator) fractions on the host
(parallel, unchanged) and the finalize — batched inversion, fraction chain, claimed
sums, cumsum shift, per-coordinate prefix sums — runs on the prove backend:
on-device for CUDA, with the interaction columns **born on device** (the
interaction tree's `from_simd_evals` transfer is gone; commit span 396 -> 202 ms).
Claims build from the finalized sums in the same fixed order before channel mixing —
the Fiat-Shamir transcript is unchanged. `memory_id_to_big` stays eager for now
(multi-segment writer; bridged, order preserved).

Gates at every step: the raw-vs-eager unit differential, the device-vs-SIMD finalize
differential on hardware (also the first qualification of the CUB prefix-sum lane),
the SIMD all-opcode e2e locally, and the **CUDA e2e proof byte-identical to SIMD**.

| program | n | round 4 | **round 5 (W1)** | total session arc |
|---|---|---|---|---|
| fib | 1,048,576 | 5.37 s / 1.37 MHz | **4.67 s / 1.57 MHz** | 14.7 -> 4.67 s (**3.1x**) |
| ec | 1,024 | 1.36 s | **1.10 s** | 11.2 -> 1.10 s (**10.2x**) |
| fib | 65,536 | 0.87 s | 0.92 s (noise; tiny trace) | 8.1 -> 0.9 s (**9x**) |

fib 1M: **0.64 s per 1M VM steps on an RTX 3090** — NitrooZK's published 5090 figure
(0.9 s/1M) now exceeded by 40% on a two-generations-older card. The predicted W1 win
(0.5-0.7 s) landed exactly. Remaining warm profile: adapt 2.8 s + cairo run 1.6 s
(host, outside the prove span), base/interaction host write loops ~2.5 s, STARK core
0.84 s — next levers per the design doc: W4 (adapt parallelization), W2 (streamed
base upload), W3 (codegen witness on GPU, the road to ~3 MHz).

### Round 5 completion: all 67 writers device-finalized + first RTX 4090 data

`memory_id_to_big` (the memory table — multi-segment big values + small table, among
the largest interaction traces) joined the raw path: per-segment finalize on the
device in the eager extension order, big-claim total as the field sum of segment
sums. **W1 is complete: the interaction tree has zero `from_simd` transfers and zero
host finalize math left.** Gates: memory component constraint tests + SIMD e2e
locally; conformance + logup differential + **CUDA e2e byte-equality on an RTX 4090**
(first Ada validation — the JIT/PTX lane re-qualified on sm_89 from cold).

4090 numbers (different host than the round-5 3090 — not directly comparable):
fib 1M warm 5.51 s / 1.33 MHz, fib 65k 0.94 s, ec 1024 1.09 s.

**The instructive result: the 4090 is no faster than the 3090 host.** The STARK core
(~0.8 s at fib 1M) is now a minority of the prove; the rest is host-side witness
writing and the VM/adapter. GPU generation has stopped mattering — which is exactly
where this work was trying to get. The next speedups live in W4 (adapt, 2.8 s of
wall clock), W2 (streamed base upload), and W3 (codegen witness on GPU) — host-side
and host-to-device work, per `WITNESS_ON_GPU.md`.

## Round 6: W4 (parallel adapter) + W2 (streamed base upload) + W3 blueprint

All gates green on an RTX 4090 (conformance, device-finalize differential, **CUDA
e2e proof byte-identical**). What landed:

- **W4 — adapter**: dedup maps on hashbrown/foldhash (ids depend on first-encounter
  order only — hash function invisible to the proof), relocation loops parallel with
  order preserved by indexed collection. Gated by the new `STWO_DUMP_INPUT`
  ProverInput byte-diff (which also surfaced a pre-existing serde nondeterminism in
  the instance-counter HashMaps — content identical, iteration order not; verified
  same-binary). Local M-series: MemoryBuilder -46%, relocate_trace -88%, adapt -23%.
  NOTE: adapt sits OUTSIDE the prove span — this is a wall-clock win, invisible in
  `prove_s`.
- **W2 — streamed upload**: every component's columns convert to the prove backend
  inside its generation task; H2D transfers overlap later components' generation and
  the base tree's bulk `from_simd_evals` is gone. The sequential deduction-phase
  components (incl. the big memory tables) convert inline — their overlap arrives
  with W3.
- **W3 — blueprint locked** (see WITNESS_ON_GPU.md): the memory_id_to_big vertical
  slice (device limb-split, device rc-count table, device denominators feeding the
  W1 finalize), with the key coupling identified: device-born base columns require
  device denominators, or readbacks eat the win. Per-component differential gating,
  NitrooZK-lesson fallbacks.

4090 numbers (host variance applies): fib 1M warm 5.15 s / 1.42 MHz, fib 65k
0.94 s, ec 1024 1.13 s. The prove span is now dominated by the host write loops
that W3 removes; wall clock additionally gained the adapter speedup.

## Round 7: the big-n scaling curve (H100 80GB) — the road to 10 MHz, measured

First-ever runs past the 24 GB ceiling (fib 4M was the round-1 blocker):

| fib n | cycles | warm prove | **MHz** | VRAM |
|---|---|---|---|---|
| 1M | 7.34M | 3.86 s | **1.90** | 9.8 GB |
| 2M | 14.7M | 6.68 s | **2.20** | 17.3 GB |
| 4M | 29.4M | 13.8 s | 2.13 | 32.6 GB |
| 8M | 58.7M | 28.9 s | 2.04 | 62.9 GB |
| 16M | — | OOM (~125 GB working set; retry with STWO_CAIRO_LOW_MEMORY) | — | — |

**Verdict: the curve plateaus at ~2.2 MHz by 2M steps.** Fixed costs are fully
amortized there; beyond it the prove scales linearly at ~0.45-0.5 us/step. Big
traces alone do NOT reach 10 MHz — the per-step cost is the wall, split roughly:
host witness writes ~40%, commits (NTT+Merkle) ~25%, STARK core ~20%, rest ~15%.

Gap analysis to 10 MHz (0.1 us/step):
1. **W3 witness-on-GPU** (spec ready, formula level): removes the ~40% host write
   share -> ~3.5-4 MHz at scale. The expected-3MHz goal is covered by W3 alone.
2. **Core kernel round**: the remaining ~0.25 us/step is GPU compute (NTT, Merkle
   blake2s, quotients, OODS) — needs a dedicated kernel-optimization pass
   (occupancy/ILP via ncu, fusion, possibly warp-specialized NTT) for the next ~2x.
3. **Beyond**: multi-GPU tree/phase parallelism or newer silicon (5090/B200) for the
   final stretch. 10 MHz = W3 x kernel round x hardware, all three.

H100 vs 4090 at 1M: 3.86 vs 5.15 s — the bigger card helps the GPU phases and the
better host helps the witness, but neither changes the plateau; only removing
per-step work does.

## Round 8: P1 on hardware — every gate green, one real deadlock, plateau 2.2 → 4.3 MHz

Hardware: RunPod secure-cloud H100 SXM 80 GB, 208-vCPU host. First round on the
prebuilt pod image (`ghcr.io/teddyjfpender/stwo-pod`, CI-built): pod boot to
first gate in ~1 minute — no rustup, no cold build, kernels fat-compiled
sm_80/86/89/90.

**Gates (all green):**
- A — stwo CUDA conformance, including the `stwo_cuda_link` unit tests repaired
  this round (they had NEVER compiled on an nvcc machine; the image build
  exposed them — stwo @ 2e62f896).
- B — P1 witness differential (`STWO_CUDA_WITNESS_VERIFY=1`) inside a full CUDA
  e2e prove: pass (panics on any mismatch).
- C — clean CUDA e2e **proof byte-identical to SIMD** with P1 + P3 live.
- D — 3× repeated-prove determinism + streams-off cross-check.
- Explicit device-path confirmation at bench scale (fib 1M): `trace columns OK`,
  `rc_9_9 count tables OK`, `interaction columns + sums OK`.

**P3 deadlock found (the gate class earned its keep):** streams-on prove hung at
fib 2M rep 2 — all threads asleep (23 futex, 2 in CUDA poll), GPU 0%, no Xid —
after ≥12 clean streams-on proves including the same size. Timing-dependent,
exactly the silent-failure class ROAD_TO_10MHZ predicted for P3.
`STWO_CUDA_DISABLE_STREAMS=1` engaged for the rest of the round. **Open item:**
audit the pool-stream event bridges (incl. cross-thread launches from the rayon
scope) before P3 re-enables by default.

**The scaling curve (streams OFF — add ~5% when P3 is fixed):**

| fib n | warm prove | **MHz** | VRAM | round 7 |
|---|---|---|---|---|
| 1M | 2.20 s | **3.18** (3.35 streams-on) | 7.5 GB | 1.90 |
| 2M | 3.47 s | **4.04** | 12.9 GB | 2.20 |
| 4M | 6.72 s | **4.17** | 23.6 GB | 2.13 |
| 8M | 12.9 s | **4.34** | 45.1 GB | 2.04 |

The plateau moved 2.2 → **~4.3 MHz and is still rising at 8M** (round 7 fell
past 2M; fixed costs now amortize further out). P1 isolation at 1M:
3.18 vs 2.28 MHz with `STWO_CUDA_MEMORY_WITNESS=0` — **1.40× from the memory
slice alone**, right on the model's prediction.

**P5 sustained (1M, prefetch depth 1): 2.14 MHz** vs ~1.3 serial — pipelining
works and exposes the next ceiling: VM + adapt (~3.2 s/proof) now exceeds the
prove (2.2 s), so sustained throughput is **VM-bound**. Fix: prefetch depth ≥ 2
(independent VM workers) — pure orchestration, next round.

**Cost-per-MHz (RunPod list prices, bandwidth-scaled from measured anchors):**
community 3090/4090 deliver ~4–6 MHz per $/hr vs ~1.0–1.25 for H100 SXM —
consumer farms win sustained-throughput economics ~3–5× while 24 GB caps single
proofs at ~2M steps; datacenter cards keep single-proof latency and big traces.
Strategy: optimize on H100, price the product on sharded consumer fleets.

M1 ncu trace captured at 1M steady-state (400 launches past warmup) — the P2
kernel ranking comes from it next round.

Next: P3 deadlock audit, witness-on-GPU for the remaining component cohort
(address_to_id, rc families, opcodes — the 4.3 → ~6 MHz leg), M1-ranked P2
kernel round, P5 prefetch depth.

## Round 9: the cheap-fleet round — 13.66 MHz sustained at $1.00/hr

Directive: 10 MHz on consumer cards, not H100s. Everything below ran on
RunPod COMMUNITY 4090s ($0.34/hr) and 3090s ($0.22/hr) from the prebuilt
image (consumer archs sm_86/89 only now — datacenter archs dropped).

**W3 slice 2 — memory_address_to_id on device** (stwo addr_to_id_pair_logup
kernel @ 3f1d0453, integration @ 58bbca43): flat chunk-major id/mult buffers
upload once, every trace column is a device slice, SPLIT/2 pair-batched logup
columns + device finalize. Differential green on the 4090
(`memory_address_to_id trace columns OK / interaction columns + sums OK`),
proof byte-identical to SIMD. Isolation at 1M: +2.7% (2.236 → 2.177 s warm).
Small component, small win — the opcode cohort is where the rest lives.

**P5 prefetch depth N** (gpu_bench `--prefetch`, default 2): VM/adapt worker
pool feeds a bounded queue. 4090 sustained ladder at 1M: depth 1 = 2.29,
depth 2 = 2.70, depth 3 = 2.94 MHz ≈ the single-proof rate — sustained
throughput is GPU-bound again.

**GPU sharing (new lever, zero code):** N pipelined prover processes share one
card; each prover's host phases leave the GPU idle and the others fill them.
4090 @1M: single 2.84 → dual 3.61 (+27%) → triple 4.02 MHz (+42%, 22 GB VRAM).
3090 @1M dual: ~4.1 MHz on strong hosts (2×2.03) — community 3090 hosts with
good CPUs are the value sweet spot.

**M1 on consumer (nsys; ncu blocked by ERR_NVGPUCTRPERM on community pods):**
H2D memcpy = 46.8% of ALL GPU time (4,698 copies / 2 proves) — witness upload
traffic dominates; every W3 slice deletes it directly. Kernel ranking:
commit_on_first_layer 36.7%, NTT family ~33%, barycentric_eval_partial 7.5%
across 1,780 launches (launch-storm candidate). P2 worklist, in order:
(1) finish W3, (2) first-layer Merkle commit, (3) NTT batches, (4) OODS
launch batching. Trace archived: stwo-things/m1_4090.nsys-rep.

**P3 disposition:** root cause identified at code level — the per-pool-stream
bridge events are shared and record/wait pairs were not atomic across
threads (P1 made concurrent pool use reachable); fix = mutex around each
pair (stwo @ 9c510755), provably safe, negligible cost. Empirical repro on
the 4090: 36 streams-on proves at 2M at the UNFIXED rev, zero hangs — the
round-8 deadlock is H100-timing-correlated. Streams stay off by default
until an H100 re-check; nothing in this round used them.

**The fleet demo (all four simultaneously, synchronized window):**

| member | card | $/hr | provers | sustained MHz |
|---|---|---|---|---|
| m1 | 4090 community | 0.34 | 3 | 3.83 |
| m2 | 3090 community | 0.22 | 2 | 4.06 |
| m3 | 3090 community | 0.22 | 2 | 1.61 (weak host CPU) |
| m4 | 3090 community | 0.22 | 2 | 4.16 |
| **total** | | **$1.00/hr** | 9 | **13.66 MHz** |

Every proof verified (verify_ms in every receipt; the pipeline is the
byte-equality-gated one). Excluding the weak host: 12.05 MHz at $0.78/hr.
Versus round 8's H100 (4.3 MHz @ $3.29/hr): **~10× MHz per dollar**, and the
whole fleet cold-starts in ~2 minutes from ghcr.io/teddyjfpender/stwo-pod.
Host-CPU variance is the fleet's real risk: 1 of 4 community hosts delivered
40% of the others — a production farm needs host screening (nproc + a 10s
cargo-free CPU probe before committing a member).

Single-proof on consumer after this round: 4090 3.22 MHz @1M / 3.34 @2M.
10 MHz single-proof remains an H100+P2 story; 10 MHz **sustained** is now a
$1/hr commodity.

## Round 10: the four levers — async copies, batched OODS, word-native Blake, batched gather

Directive: design/engineer/benchmark (1) witness-on-GPU completion, (2) async
copies, (3) the P2 kernel round, (4) launch-storm + overlap — toward the
physical limits. All validation on community 3090s/4090s ($0.22-0.34/hr).
Every lever is byte-equality-gated: the CUDA proof remained byte-identical to
SIMD after each change, with all witness differentials green.

**(2) Async upload lane** (stwo a096dfe3): pinned ping-pong staging halves on a
DEDICATED copy stream; destination buffers allocated stream-ordered on that
stream (uploads never trail pending legacy compute); one closing bridge orders
consumers; record/wait pairs share the bridge mutex (the round-9 lesson).
Applied to `from_simd_evals` (base trace) and `finalize_raw_logup` (raw logup
pairs — previously pageable, synchronous, allocating). Kill switch
`STWO_CUDA_SYNC_UPLOADS=1`. A/B on one 3090: 2.185 -> 2.098 s warm (+4.2%);
kill-switch run restores baseline (clean attribution). H2D share of GPU ops
46.8% -> 38.2%, now overlapped.

**(4a) Batched OODS** (stwo b8c5348f): `PolyOps::barycentric_eval_many` — the
pcs prover flattens all (column, point) jobs; CUDA enqueues every
partial+reduce chain with NO intermediate synchronization into one device
results buffer, single readback (was: 1,780 launches EACH followed by a
16-byte stream-draining readback). Default impl keeps SIMD's parallel loop.

**(3) P2, slice 1 — word-native Blake2s commits** (stwo 68784bce): the commit
kernels fed u32 words through a byte-buffer incremental API (local-memory
64-byte buffer, memcpy per word, byte->word repack per compress). Rewritten to
16-word register blocks compressing directly, exact last-block semantics.
`commit_on_first_layer`: 472 -> 145 ms / 2 proves on the 3090 — **3.25x on the
biggest kernel** (40.4% -> 17.5% of kernel time). Wall impact small at 1M
because the kernel already overlapped host work — the savings pay out in
GPU-sharing fleet mode and as host work shrinks. NTT family (~32%) is the
next P2 slice (documented, not yet attempted).

**(4b) Batched decommit gather** (stwo 8edf8e3d): the queried-values loop read
one element per (column, query) through `Column::at` — 61,726 synchronous
4-byte readbacks per 2 proves (measured). `Column::at_many` +
`ColumnAccess::values_at` batch it per column through the existing device
gather kernel (~70x fewer roundtrips). Byte-equality + differentials GREEN at
this rev; its bench delta was lost when the community host's GPU fell off the
bus mid-round (driver death, "No devices were found" — gates had already
passed; the pod was torn down).

**(1) W3 completion — design** (`gpu_benchmarks/WITNESS_CODEGEN.md`): the
constraint-JIT record-once recipe does NOT transfer to witness writers (no
generic seam; branchy; side-effecting sub-component feeds). Mechanisms ranked:
upstream a witness-IR emitter in stwo-air-infra (right answer, private repo),
else hand-port by traffic share (verify_instruction -> loop-body opcodes -> rc
families) with the proven P1/addr recipe. Memory tables + address_to_id are
already device-born; the cohort port is the next multi-session program.

**Cumulative, same 3090 host, single-proof 1M:** 2.185 s -> **1.998 s warm
(3.50 MHz, +9.4%)** across the stack — and the freed GPU time (blake 3.25x,
H2D overlapped) raises the GPU-sharing ceiling that round 9's fleet exploits.
Phase overlap beyond the upload lane (commit N over witness N+1) is deferred
pending the H100 streams-on recheck of the bridge-mutex rev.

Costs this round: ~2.5 h of one community 3090 (~$0.55). One community GPU
died mid-round (host fault, not reproducible by our code; the same binary
passed all gates seconds earlier) — fleet host-screening remains the lesson.

## Round 11: the generic witness lane, streams settled, MPS, and the EC record

**(1) W3 — the generic witness lane + verify_instruction (the beachhead).**
Every generated interaction writer reduces to two kernel shapes; the lane ships
them once (`tuple_pair_logup`, `tuple_single_logup` with 1/column/Enabler
multiplicities and sign, `tuple_count` with per-slot LUT bit packing), so each
cohort port now only supplies a base-trace kernel. verify_instruction is the
first component through it: decode kernel + 3 staged combination columns,
device rc_7_2_5/rc_4_3 count feeds, host memory feeds from the same uploaded
arrays. Differential green on hardware (`verify_instruction trace columns OK /
interaction columns + sums OK`, both rc deltas), proof byte-identical to SIMD.
fib delta ~0 as predicted (dedup'd-tiny component) — the value is the proven
lane; the loop-body opcodes ride it next (multi-session by nature; mechanism
doc: WITNESS_CODEGEN.md).

**(2) NTT P2 slice — honest verdict.** The n2b/b2n family is already a tuned
NitrooZK-lineage implementation (warp-shuffle butterflies, smem transposes,
fused 6/8-stage blocks, poly batching). Further gains need ncu-driven
change-measure iteration on secure cloud (perf counters are blocked on
community pods) — queued as its own work item, not blind edits.

**(3) Streams settled on H100.** At the bridge-mutex rev: byte-equality green
streams-ON, then **12 repro rounds (36 proves) at the exact round-8 hang
conditions: zero hangs** — the deadlock class is closed. The on/off bench pair
shows no streams gain at the current pipeline (off even reads faster within
warm-up ordering noise), so streams stay off by default until the
commit-of-tree-N-over-witness-of-N+1 scheduler gives them real work; the async
upload lane already provides the copy/compute overlap.

**(4) The tail.**
- The 61.7k D2H reads are NOT the queried-values loop (that gather landed) —
  they are `node_hash`'s per-(column, leaf) `raw_value` reads in the
  pruned-tree recompute (`vcs_lifted/prover.rs:441`). Fix designed: per-leaf
  row-gather kernel, or recompute queried leaf hashes fully on device via the
  commit kernel with an index list (~1 launch). Next tail item.
- **MPS beats raw process sharing by +11%**: dual provers on one 3090 =
  **5.38 MHz aggregate** (2.70 + 2.67) vs 4.84 raw — and that raw dual is
  itself +19% over round 9's (the round-10 levers freed GPU time exactly as
  predicted). Fleet math: two MPS-dual 3090s ≈ **10.8 MHz at $0.44/hr** — the
  10 MHz fleet cost halved since round 9.
- Host-screening probe shipped (`gpu_benchmarks/pod/screen_host.sh`).
- CUDA graphs over the JIT constraint launches: assessed, deferred (per-prove
  pointer churn makes capture/update fiddly; the OODS batching already removed
  the worst launch-storm cost).

**EC benchmark — first record** (3090 community, current stack, secure
config): 5,000 ec-double iterations = 1.17M cycles, **warm prove 1.033 s**
(1.13 MHz single; sustained 1.07 — GPU-bound, pipeline saturated). EC cycles
drag the wide builtin traces (pedersen/ec_op/mod), so MHz reads lower than fib
by construction; the per-proof second is the comparable number.

**Headline single-proof numbers at this rev (fib 1M/2M):**
- 3090 community ($0.22): 1.70 s / 4.11 MHz @1M — a $0.22 card now beats
  round-8's H100.
- H100 SXM: 1.79 s / 3.92 @1M, 2.87 s / **4.87 MHz @2M** (+21% vs round 8) —
  and the strong-host 3090 outruns it at 1M: the workload is host-bound, which
  is the whole W3 thesis in one line.

Costs: ~1.2 h H100 (~$4) + ~3 h of 3090s (~$0.70). One earlier community 3090
GPU death (round 10) remains the only hardware casualty.

## Round 12 — 2026-06-12 (cont.): the opcode cohort lands (W3 essentially complete for fib)

Per `ROUND12_SPEC.md`, executed 4 → 2 → 1; item 3 (ncu NTT) still queued for a
secure pod. All numbers from one community 3090 ($0.22/hr, 64-core host —
slower baseline than round 11's host, so compare within-host). Every change
byte-equality-gated; every opcode port differential-gated (trace columns,
feeds vs sub_component_inputs, interaction columns + sums).

**(4) Overlap v1** (`638487ca`): `B::finalize_raw_logup` moved inside all 64
component rayon spawns (generalizing the memory/vi in-scope pattern); evals +
claims stay post-scope in fixed order — Fiat-Shamir untouched. Gates green.
1M 2.681 → 2.618 s (+2.4%); 2M flat on this host.

**(2) Device leaf-hash recompute** (stwo `89bfd0a9`):
`MerkleOpsLifted::leaf_hashes_at` hook + `commit_on_first_layer_lifted_indexed`
kernel — the pruned-tree decommit's unretained leaf hashes now come from one
indexed launch + one D2H instead of 61.7k sync 4-byte `raw_value` reads.
Unit differential (device vs CPU `build_leaves` at scattered indices) green on
hardware; kill switch `STWO_CUDA_LEAF_HASHES=0`. 1M 2.618 → 2.540 s (+3.1%),
2M +3.7%. (nsys was unavailable on this pod; the D2H-count receipt rides on a
future profiled round.)

**(1) THE WHALE: six opcode base-trace kernels on the generic lane.**
Lane upgrades first: tuple kernels now take a folded combine base + repacked
per-live-column alphas (`TupleSlot::{Col, Const}` — constant-heavy opcode
tuples cost nothing on device); memory deduce fused as device gathers
(`mem_addr_to_id`, `mem_id_to_limbs` — tag/val decode + 9-bit split identical
to `deduce_output`); prove-wide `DeviceMemTables` uploaded once. Then the
ports: **ret** (pilot, main session), **add_opcode_small, jnz_opcode_taken,
add_opcode, call_opcode_rel_imm, assert_eq_opcode** (delegated transcriptions,
integrated via union merges). Per-component kill switches
`STWO_CUDA_{RET,ADD_SMALL,JNZ_TAKEN,ADD_OPCODE,CALL_REL_IMM,ASSERT_EQ}_WITNESS=0`.

The differential EARNED ITS KEEP twice:
- add_opcode_small's id_to_big tuples were missing the leading id slot —
  caught at "interaction col 4 MISMATCH row 0" precision (trace cols green,
  ret/jnz green in the same run), fixed in one line-set.
- The first full-cohort round REGRESSED (1M 3.74 s) while gates stayed green:
  the host-side feed loops were sequential per row and jnz decoded via a
  16-lane PackedFelt252 broadcast per row. Fix: rayon-parallel feed loops +
  scalar raw-table decode. (Lesson recorded: port the feeds with the same care
  as the kernels — they are the new host critical path.)

**Final validated stack** (stwo-cairo `0fe90916` / stwo `d52454fe`):
- fib 1M: **1.566 s warm / 4.47 MHz** (stable ×6: 1.579 s) — vs 2.681 s
  baseline on this host = **+71%** in one round; beats every previous
  single-proof number on any host including round-11's faster-host 3090
  (1.70 s) and round-11's H100 @2M MHz.
- fib 2M: **3.036 s warm / 4.61 MHz**.
- Opcode-cohort lever isolated (all six off → on): 2.281 → 1.566 s = +46%.
- peak host RSS halved (5.8 → 2.5 GB @1M): the witness really left the host.

**Process note:** `crates/prover/Cargo.toml` pins `stwo-backend-cuda-kernels`
separately from the workspace manifest — bump BOTH (the round-12a leaf-hash
gates were silently lost to a duplicate-archive link failure; rerun clean).

**MPS dual at the final stack: 3.82 + 3.60 = 7.43 MHz aggregate on ONE
community 3090** — two MPS-dual 3090s ≈ 14.9 MHz at $0.44/hr; the 10 MHz
fleet now quotes at ~$0.30/hr (vs $1.00 in round 9, $0.44 in round 11).
Sustained single-process (pipeline, prefetch 3) 3.64 MHz — VM+adapt bound on
this host; the MPS pair is the throughput configuration.

Receipts: `stwo-things/round12{a,b,d,e,f}_*.log`. Remaining round-12 debt:
item 3 (ncu NTT on secure cloud), nsys D2H receipt, vi-feed device path, and
the next ranked queue (rc/builtin cohort ports on the same lane, device feeds
via index_count, commit-over-witness scheduler before streams re-enable).

### Round 12, item 3 — the NTT measurement verdict (secure 4090, 2026-06-12)

**ncu counters are blocked on RunPod, secure cloud included**: probe returned
`ERR_NVGPUCTRPERM`; `/proc/driver/nvidia/params` shows `RmProfilingAdminOnly: 1`
— a host driver policy no container can override. Documented per the spec's
contingency; real ncu iteration needs Lambda/own box.

**M1 fallback executed** (nsys per-instance kernel durations + grid/block
geometry from the sqlite export; byte model = one read + one write of each
launch's covered elements, `threads x vals_per_thread x 8B` — twiddle traffic
uncounted, so %peak is a mild underestimate; variant-relative ranking robust):

fib 1M/2M, RTX 4090 (1008 GB/s peak), aggregate over 2 proves:
- **NTT family total: ~444 GB/s ≈ 44% of DRAM peak — NOT at roofline.**
  The round-11 "already NitrooZK-tuned" hypothesis is refuted by measurement.
- Block variants are near roofline: `b2n_noinit<3>` **92%**, `b2n_noinit<4>`
  69%, `n2b_nofinal<3>` 55%, `<4>` 46%.
- The warp variants are the laggards and hold ~45% of family time:
  `n2b_final_warp<2>/<3>` 32-33%, `b2n_init_warp<2>/<3>` 26-30%,
  `n2b_final_block_warp` 14-28%.
- **Quantified headroom: ~1.6x on the family** if the warp variants reach the
  block variants' efficiency.
- One-knob check (dispatch/config tables, `LAUNCH_N2B_CONFIG_20_27`):
  computed against the measured per-variant bandwidths, alternative stage
  splits at the hot sizes (log 21-23) net out neutral-to-worse — fewer passes
  trade into intrinsically slower final kernels. The win is INSIDE the
  warp-final/init kernels (strided global access -> vectorized uint4 loads,
  smem bank conflicts in the shuffle stages) — a kernel-internal P2 slice,
  now data-ranked for the next round.

NTT share context at the current stack: ~107 ms/prove @1M (4090) out of
~2.0 s — the family is ~5% of wall today, so the 1.6x family fix buys ~3% of
wall; rank it accordingly against the rc/builtin witness cohort.

Receipts: `stwo-things/round12g_*.{log,txt}`. Secure pod torn down; zero pods
billing.
