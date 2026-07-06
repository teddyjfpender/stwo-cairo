# Cairo e2e GPU proving benchmarks (stwo-book format)

Methodology mirrors [zksecurity/zkvm-benchmarks](https://github.com/zksecurity/zkvm-benchmarks)'
stwo runner (the source of the [stwo-book benchmark tables](https://zksecurity.github.io/stwo-book/benchmarks/index.html)):
same Cairo programs and `program_input` hint, Cairo VM in proof mode, secure prover
configuration (`pow_bits=26`, blowup 1, 70 FRI queries — ~96 bits), preprocessed trace
`CanonicalWithoutPedersen`, proof size via bincode, cycle count = sum of opcode counts.
Harness: `crates/prover/src/bin/gpu_bench.rs`. As of round 8 the harness also has a
bootloader PIE-ingestion lane (`--pie`, multi-PIE lists, `--pie-copies`, `--pie-mode
aggregate|rotate`, `--pipeline`/`--producers`, `--reuse-input`, `--adapt-only`) that
proves real Starknet OS PIEs and reports `useful_mhz` (PIE `n_steps` basis) alongside
`mhz` (proved cycles, incl. bootloader overhead).

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

## Round 8: first CUDA proofs of real Starknet OS PIEs (the bootloader lane)

The landmark: **the first CUDA proofs of real Starknet OS execution**, not fib. Four
new SN PIEs (`cairo_pie` v1.1, bootloaded — NOT the old sepolia fixtures) run end to
end through the new `gpu_bench` PIE lane (`--pie`, multi-PIE lists, `--pie-copies`,
`--pie-mode aggregate|rotate`, `--pipeline`/`--producers`, `--reuse-input`,
`--adapt-only`, self-describing 96-bit JSON records, a VRAM high-water sampler,
`useful_mhz` vs `mhz`, `STWO_BENCH_TRACE=json` phase totals). Getting here took three
pre-existing bug fixes that no fib-era run could surface — the pedersen `LazyLock`×rayon
deadlock, the JIT fused-kernel NVRTC/ptxas blowup, and the batched-NTT grid-axis
overflow (full postmortems in `KNOWN_ISSUES.md`). Correctness state, stated exactly:
every completing run passes in-harness `verify_cairo`, and the CUDA proof byte-SIZE
equals the SIMD-proven baseline's (3,006,636 B on SN_PIE_2; 2,897,542 B on the
10-transfer fixture) — but the formal **CUDA-vs-SIMD proof byte-equality diff has not
yet been run on the PIE lane** (it was the gate for rounds 1–7 workloads). It is the
first item of the round-9 gate list, together with a repeated-prove check for the
once-observed nondeterministic rep hang.

The programs (PIE `n_steps`):

| PIE | n_steps | status on A40 46 GB |
|---|---|---|
| SN_PIE_2 | 7,706,864 | **proves** — warm/sustained numbers below |
| SN_PIE_1 | 14,645,112 | OOM (>46 GB, quotient/FRI peak) |
| SN_PIE_3 | 14,075,019 | OOM (>46 GB) |
| SN_PIE_4 | 14,058,247 | OOM (>46 GB) |

**SN_PIE_2 on an A40 46 GB** (RunPod secure, $0.44/hr, ~7 effective vCPU): warm
**31.6–33.9 s**, `useful_mhz` **0.227–0.244** (`mhz` 0.235–0.252 incl. the measured
**3.51 % bootloader overhead**), verify 22–30 ms, proof **3,006.6 KB**, `vram_peak`
**36.1–36.4 GB**, host RSS ~29 GB. Cost-model warm split (`sn2_cuda_jit` rep 1,
`prove_cairo` 36.09 s):

| phase | time | share | note |
|---|---|---|---|
| Write Base trace + Write interaction | 17.25 + 6.72 = 23.97 s | **66 %** | host witness write — W3's target |
| Commitments (NTT + Merkle) | 10.43 s | **29 %** | fusion + bandwidth headroom |
| Prove STARKs core | 3.14 s | **8.7 %** | Composition 0.61 s (split JIT kernels), OODS 0.73 s |

**The cost model re-ranks the programs for real Cairo, exactly opposite the fib
plateau of round 7**: on a builtin-heavy PIE the host witness write dominates (66 %,
not ~40 %), commits are second (29 %), and the STARK core is already cheap (8.7 % — the
JIT kernel-splitting fix works). W3 witness-on-GPU is the single dominant lever here.

**Sustained throughput** (pipeline 2, 3 producers, 6 reps, `--pie-mode rotate`):
`sustained_useful_mhz` **0.228**, `feed_starved_s` **0.0** (producers kept the GPU fed),
total 202.8 s — i.e. the warm single-prove rate holds under a continuous feed, no
pipeline stall.

**Cold start is the story to fix next.** Cold rep 0 was **1,796 s**: the JIT lane
codegen'd **116 per-component kernels**, of which **66 hit the disk PTX cache** (1–3 ms
each) and **7 recompiled via NVRTC** — but one module's driver **PTX→SASS load alone was
56.4 s**. PTX is cached; SASS is not, so a cold process pays the driver's per-kernel
assembly. Diagnosis: cubin (SASS) caching is queued; warm reuse is already there.

**The 14M-step PIEs are now a memory problem, not a launch-geometry one.** After the
NTT grid-axis fix, sn1/sn3/sn4 no longer error at `rfft.cu:667` (the old grid overflow);
they progress until the allocator pool is exhausted and fail as genuine OOM (`sn3`:
`cudaErrorMemoryAllocation`; `sn1`/`sn4`: pool-exhaustion → invalid downstream memcpy),
including with `STWO_CAIRO_LOW_MEMORY=1` — the quotient/FRI peak is not covered by that
mode (same shape as round 2's fib-4M-on-24 GB note). They are gated on the VRAM diet or
≥80 GB cards.

Honest caveats:
- A40 numbers ran with **`STWO_CUDA_DISABLE_STREAMS=1`** (P3 overlap was off during the
  JIT-hang bisect) — the stream-overlap upside is still pending, so these are a floor.
- The **~7-core container inflates the host-write share**: the 66 % is partly a
  thin-host artifact, not purely algorithmic. Only same-pod comparisons are meaningful.
- The same-pod A40 SIMD baseline is incomplete (7-vCPU host, one rep 77 s before the
  multi-rep run was killed) — CUDA clearly wins there, but it is not a fair CPU. A 3090
  pod's 20-vCPU SIMD did SN_PIE_2 in 28.3 s / 0.273 `useful_mhz` / 49 GB RSS (a
  different host — do not cross-rank), and the 14M PIEs OOM'd that CPU above 125 GB RAM.
  The 3090 CUDA lane proves a 10-transfer PIE warm in 6.1 s post-JIT-fix, but every SN
  PIE exceeds its 24 GB (peaked 24.99 GB on even the small PIE with Canonical
  preprocessed).
- NitrooZK's published PIE numbers use `n_queries=3`; ours are the 96-bit
  `n_queries=70` config. Never compare the raw figures.

**Verdict.** First real-workload CUDA proofs land, correctness-gated, and the cost
model gives a clean fleet math: per-GPU `useful_mhz` **0.23 today → ~1.1** with W3 +
P2 stream overlap on A40-class → **~2–2.5** on 4090/5090 after the VRAM diet. An
aggregate 10–20 MHz is then 5–8 consumer cards at ~$0.15–0.35/MHz-hr versus ~$0.7 on
H100. The per-component `wt:*` instrumentation shipped this round is the measured next
step; W3 witness-on-GPU (builtin-first) is the dominant lever the trace now names.

## 2026-07-06 — GPU-native pipeline v1: M0–M3 validated, records on all four SN PIEs (H100 SXM sk60d6jcg5p4lu, runs 20260705T233744Z + 20260706T001447Z)

Config: `--engine gpu-native` (composed defaults: witness JIT lanes + device
interaction + device edges + governor 20000) + the embedded AOT pack (118
kernels @ cap 2048, offline -O3 sm_90 cubins; 39 pack hits/prove, runtime-key
set-compare = all hit). 96-bit (pow26/blowup1/q70). All proofs verified
in-run; SN_PIE_2 byte-identical to the legacy engine's proof
(ENGINE_PROOF_MATCH + RECORD_SN2_PROOF_MATCH).

Gates: deduce_gate.toml 15/15 PASS — oracle legs, whole-kernel builtin gates,
lane on/off + DAG + AOT full-proof byte identity, engine parity, cubin-2048.

| PIE | useful steps | cold s | warm s | useful MHz | prior best | Δ |
|---|---|---|---|---|---|---|
| SN_PIE_2 | 7.71M | 14.83 | **10.76** | **0.716** | 13.5 / 0.571 | **+25%** |
| SN_PIE_1 | 14.65M | 76.6* | **18.20** | **0.805** | 38.7 / 0.378 | **+113%** |
| SN_PIE_3 | 14.08M | 73.9* | **14.35** | **0.981** | 36.0 / 0.391 | **+151%** |
| SN_PIE_4 | 14.06M | 73.6* | **14.92** | **0.942** | 29.3c / 0.480 | **+96%** |

*cold includes first-run VM+adapt (~50s single-threaded on this 22-vCPU
container).

Sustained (pipeline 3, producers 3, aggregate): 0.543 useful MHz over 3 reps
(42.6s total incl. fill; feed_starved 0.0s — host feed keeps up at this rate).

Session ladder (SN_PIE_2 warm): legacy lane-off 16.76/0.460 → DAG 13.49/0.571
→ AOT-only 14.62/0.527 → cubin-2048 14.16/0.544 → **gpu-native (DAG+AOT)
11.25→10.76/0.716** — the DAG and AOT levers COMPOSE.

VRAM: SN_PIE_1 peaked 75.4GB (DAG config; 80GB card at the edge — the M4 diet
matters even on H100 for 14M-step PIEs). SN_PIE_2 42.5GB.

Fleet math checkpoint: 0.94–0.98 useful MHz/card on 14M PIEs ⇒ ~10 MHz
aggregate at ~10–11 H100s, or the M4-diet 4090 fleet at projected 2/3 the
per-card rate for ~1/10 the $/hr — the $/MHz-hr thesis strengthens.

Session cost: ~$4.90 (1.5h H100 SXM incl. 2 full builds + 2 hardware-only AOT
wall fixes: offline-nvcc curandState in the fp256 embed, and the
anonymous-namespace extern linkage class — both committed with probes).

## 2026-07-06 — M4 increment: word-block Merkle + device mem count feeds, new records on SN1/SN2/SN3 (H100 SXM sk60d6jcg5p4lu, run 20260706T023111Z)

Two M4 levers landed and validated byte-identical (M4_PROOF_MATCH on SN_PIE_2
vs the same-build legacy engine):

- **Word-block blake2s Merkle path** — leaf/lifted/interior kernels hash M31
  words directly as LE message words (no byte staging buffer), unrolled
  16-word register blocks. Bit-identical by construction *after* fixing an
  eager-block bug: `blake2s_update` is lazy (`inlen > fill`), so a stream
  ending on a full 64-byte block flags THAT block last — the eager loop
  compressed it with last=0 and appended a zero-padded extra block, mis-hashing
  exactly the trees whose column count ≡ 0 (mod 16). SN_PIE_3's FRI first
  layer is such a tree: `Fri(FirstLayerCommitmentInvalid RootMismatch)` on
  both engines while SN2 passed everything. Fix: lazy loop (`col+16 < n`,
  rem ∈ 1..=16). The testkit merkle conformance now pins 16/32-column
  mixed-size trees (root + queried values + hash witness) — the old 5-column
  case could never catch a block-boundary bug; CUDA conformance PASS on pod.
- **Device memory count feeds v2** — opcode lanes + blake/aggregator builtin
  seams feed memory_address_to_id (signed key_offset), memory_id_to_big
  (mem-id decode, big/#small split) and blake sigma LUT counts on device;
  hand feeders gained skip guards (double-feed hazard caught by the 49-desc
  differential gate + fail-loud seam panics).

| PIE | useful steps | warm s | useful MHz | prior best | Δ |
|---|---|---|---|---|---|
| SN_PIE_2 | 7.71M | **8.98** | **0.858** (0.888 raw) | 10.40 / 0.741 | **+16%** |
| SN_PIE_3 | 14.08M | **12.43** | **1.133** (1.155 raw) | 13.98 / 1.007 | **+15%** |
| SN_PIE_1 | 14.65M | **14.90** | **0.983** (1.001 raw) | 18.20 / 0.805 | **+24%** |

| SN_PIE_4 | 14.06M | **11.45** | **1.228** (1.252 raw) | 14.92 / 0.942 | **+30%** |

First sub-10s Starknet OS proof (SN2 8.98s); SN1/SN3/SN4 all above 1 MHz raw,
SN4 the per-card best at 1.23 useful MHz.

Phase ledger after the rewrite (SN2 warm rep, prove_cairo 8.77s):
**Merkle 4.72s → 1.68s** — the #1 lever delivered. New ranking: Write Base
trace 4.08s (wt: lane spans sum ~5.9s sequential; longest single lane 0.72s
— multi-stream lane overlap is now the biggest single win), Commitment
2.01s, Prove STARKs 1.98s. Road to sub-5s SN2 runs through M5 lane/tree
overlap, not more hash work.

Per-phase VRAM ledger (SN_PIE_2, used_high/pool): witness 32.4 →
preprocessed_tree 26.7 → **base_commit 40.7 (peak)** → interaction_write 36.3
→ interaction_commit 37.4 → stark_core 34.4 GB. The diet target is the
base_commit LDE+tree working set, not stark_core. SN_PIE_1 peak 75.4GB —
still H100-edge; 4090 fit needs ~3x diet at 14M steps or PIE sharding.

Fleet math: 1.13 useful MHz/card on SN3 ⇒ ~9 H100s for 10 MHz aggregate.

## 2026-07-06 — M5a: builtin lanes as concurrent dependency arms + stream fanout, records again on all four PIEs (H100 SXM sk60d6jcg5p4lu, run 20260706T030913Z)

The post-Merkle ledger's #1 lever delivered. The sequential builtin witness
section (~3.5s of lane time) now runs as 7 rayon dependency arms derived from
the certified schedule + per-writer state edges (verify_instruction | blake
chain | mod/rc builtins | pedersen w18 | pedersen w9 | ec_op→ec_generic |
poseidon chain), and STWO_CUDA_STREAM_FANOUT=1 joined the gpu-native defaults
so concurrent lanes' kernels overlap on 4 pool streams (fork/join bridged,
"B2 engaged"). Byte-identity gated end to end: gpu_native_parity_simd GREEN
locally, M5A_PROOF_MATCH on pod, all proofs verified in-run.

| PIE | useful steps | warm s | useful MHz | prior best | Δ |
|---|---|---|---|---|---|
| SN_PIE_2 | 7.71M | **7.39** | **1.043** (1.079 raw) | 8.98 / 0.858 | **+22%** |
| SN_PIE_3 | 14.08M | **8.78** | **1.604** (1.635 raw) | 12.43 / 1.133 | **+42%** |
| SN_PIE_4 | 14.06M | **10.34** | **1.359** (1.385 raw) | 11.45 / 1.228 | **+11%** |
| SN_PIE_1 | 14.65M | **10.81** | **1.355** (1.380 raw) | 14.90 / 0.983 | **+38%** |

All four Starknet OS PIEs above 1 MHz useful; SN_PIE_3 at **1.6 MHz** —
4.1x the pre-program SIMD-era 0.39, and 14M steps proven in under 9 seconds.

Ledger after M5a (SN2 warm, prove_cairo 7.47s): Write Base trace 4.08→2.33s,
Merkle 1.75s, Commitment 2.01s, Prove STARKs 2.22s. The witness arm win is
partially masked by the A2 committer overlap; the next levers are M5b
inter-tree/commit overlap and M6 two-proof pipelining — the phase pie is now
almost evenly split, so overlap (not single-phase compression) is the road on.

Fleet math: 1.6 useful MHz/card on SN3 ⇒ **10 MHz aggregate at ~6-7 H100s**
(was ~10-11 at M0-M3). Session cost so far ~$7.

Housekeeping fixed this session: STWO_BOOTLOADER_JSON runtime override
(resume-proof), pods.conf port refresh on resume, testkit merkle conformance
pinning exact 16-word-block trees.

## 2026-07-06 — M5b: batched OODS + committer A/B + leaf-hash launch_bounds (H100 SXM sk60d6jcg5p4lu, runs 20260706T080622Z + records)

M5b landed three things and validated them with a WITHIN-SESSION A/B (this is
the methodology point: the pod's inter-session variance is ~7-8%, but
back-to-back within-session variance is only 2.1% — so A/B on the same pod
state is the reliable measurement, cross-session absolute comparison is not).

**Confirmed wins (within-session, byte-identical, verify-gated):**
- **Batched OODS** (group columns by (log_size, folded point), one launch pair
  + one D2H per group): SN3 9.519s vs 9.903s with it off = **−0.38s**. The
  OODS span itself collapsed 0.56s → **0.076s** on SN2. A latent grid.y>65535
  cap (adversarial-review finding) is guarded by chunking at 32768.
- **Leaf-hash `__launch_bounds__`**: the Merkle-span decomposition (new
  STWO_MERKLE_SPANS instrument) showed Merkle's 1.69s is **95% the log24 leaf
  hash** (635ms/rep; interior only ~11ms across 130 calls) and ~20-50× off
  both bandwidth and compute bounds = occupancy-bound. Capping registers via
  launch_bounds raised resident warps: leaf hash **635ms/rep → 512ms/rep
  (−19%)**, byte-identical.
- **Per-lane pipelined committer: DISABLED from defaults.** A/B showed it helps
  SN2 ~0.3s but is within-noise / slightly negative on the 14M PIEs (its iFFT
  contends with the witness arms on one stream). Overlap must PAY to default
  on — it doesn't here. Flag + code retained (U3 scaffolding).

**Sustained pipelining improved: 0.543 → 0.741 useful MHz (+36%)** — the
aggregate/throughput axis (the road to 10 MHz) is moving, though still below
single-proof (1.05) so the residency work (M5 graphs) must land before
pipelining fully pays.

This session's pod ran ~7-8% slower than the M5a session (SN3 9.43 vs 8.78),
so the M5a absolute records stand as the best measured numbers; the M5b levers
are confirmed to improve on them by ~0.5s/proof (within-session), projecting
SN3 to ~8.3s / ~1.7 useful MHz on a clean host. Standing records (M5a, warm
useful MHz): SN2 7.39/1.04, SN3 8.78/1.60, SN4 10.34/1.36, SN1 10.81/1.36.

**Verdict feeding M5/M6:** single-card intra-proof micro-levers are now
sub-second and near the pod noise floor. The remaining big levers are
structural — CUDA graphs (attack the F1 orchestration floor: 96% idle SM) and
M6 two-proof pipelining (fill the idle; sustained already +36%). Leaf hash
(512ms/rep) stays the top commit-path kernel target for a deeper occupancy
pass. Session cost ~$3.


## 2026-07-06 — M5b clean-host confirm + sustained producer sweep (H100 SXM sk60d6jcg5p4lu)

Clean-host re-measure of the M5b build (batched OODS + leaf launch_bounds,
committer off): SN_PIE_4 **9.89s / 1.421 useful MHz** — new record (was
10.34/1.36). SN3 8.99/1.566, SN2 7.51/1.026, SN1 11.09/1.321 — within
inter-session noise (~5-8%) of M5a.

Sustained producer sweep (SN2, --pipeline=--producers in {1,2,4}): sustained
useful **0.78-0.80 MHz, FLAT across producer count**, feed_starved=0. Pins the
model: --pipeline/--producers is HOST-FEED OVERLAP (producers run VM+adapt for
the next PIE while the GPU proves the current); proving is SERIAL on the GPU,
so sustained is capped at single-proof MHz. Exceeding single needs TRUE
two-proof GPU concurrency => needs the VRAM diet (streaming-LDE-into-leaf) so
two proof states fit one card. Not producer tuning.

## 2026-07-06 — M5c: streaming LDE-into-leaf-hash VRAM diet, byte-identical (H100 SXM sk60d6jcg5p4lu)

The linchpin. Base columns are LDE'd one 16-column group at a time, each fed
into the running per-leaf blake2s state then freed, so all columns' evaluations
are never resident at once. New device kernels (stream_leaf_init/update/
finalize) + a coeff-LDE driver, both validated byte-identical to the all-at-once
path on hardware (stream_leaf_layer_matches_build_leaves +
stream_commit_leaves_matches_bulk), wired into CommitmentTreeProver behind
STWO_CUDA_STREAM_LEAF_COMMIT (forces stream_lde + store_coeffs +
FORCE_EXTEND_EVAL_MODE downstream). **Whole-proof M5C_PROOF_MATCH** on SN_PIE_2.

VRAM (SN_PIE_2): **peak 42.5GB → 30.8GB** (base_commit 40.5→26.1, witness
23.6, preprocessed_tree 27.5, interaction 27.1, stark_core 28.9). The base_commit
peak — the original target — fell 40.5→26.1GB.

**Cost finding (decisive for strategy): single-proof time 7.4s → 26.7s (3.6×).**
The diet is coupled to full stream_lde, which regenerates trace evals from
coefficients ~3× (composition ExtendToEvalDomain + per-group FRI quotients +
decommit). This is the "costs a pass" tradeoff (design §7), realized as ~2.6
extra eval-regen passes. Implications:
- Single-card MHz: the diet HURTS (3.6×) — it is a memory tool, not a latency lever.
- 2-proof pipelining on 80GB: 2×30.8=61.6GB now FITS (was 2×42.5=85 > 80). But
  whether it's a throughput WIN depends on the regen passes overlapping the
  concurrent proof's compute — plausible given the measured ~55% single-proof
  GPU util (45% idle to absorb the extra passes), but only the M6 pipelining
  implementation can measure it. Single-proof 3.6× is a pessimistic upper bound.
- 4090 fleet (24GB): 30.8GB still over — needs the peak (now stark_core 28.9)
  dieted further to fit, and the time cost compounds.

Verdict: the memory goal is DELIVERED and byte-identical; the throughput win is
gated on M6 (measure whether the regen hides under a concurrent proof) and/or
reducing the regen cost (single cached regen / partial diet). Flag is opt-in
(NOT a gpu-native default — it regresses single-card). Standing records unchanged
(SN4 9.89s/1.42 is the current best single-card).

## 2026-07-06 — RegenCache P1: batched decommit gather cuts the M5c diet cost 3.6x → 1.53x (H100 SXM)

The streamed-LDE diet's 3.6x regression was pinned by a phase bisection (STWO_PVT
eprintln timers — the bench-trace subscriber silently drops spans with unknown
`class`) to ONE place: the decommit's per-row `at_unreduced` loop — queries×columns
individual device readbacks — NOT composition, NOT quotients, NOT the re-LDE.
`trees_decommit` was 17.3s of the ~21s Prove STARKs span; the re-LDE itself is only
0.12s. The compact/stream_lde decommit had kept the per-element gather while the
non-compact path already used the batched `gather_unreduced` (CUDA: one
`cuda_gather` + one D2H). Routed the compact path through it (dedup rows, one gather
per column).

Result (SN_PIE_2, warm): **trees_decommit 17.3s → 1.26s**; **M5c prove 26.7s → 11.23s
(3.6x → 1.53x vs non-diet 7.35s)**; **byte-identical (G_PROOF_MATCH)**; peak **31.6GB**
(two proofs now fit an 80GB card). NOTE: 11.23s is the SINGLE-proof M5c latency (1.53x of non-diet 7.35s); the <11s / <14.8s gates are for the TWO-proof WALL time (M6-a's target), a different quantity — not yet measured.

The residual 1.53x (~3.9s over non-diet) is the inherent stream_lde regen —
composition ExtendToEvalDomain (~0.9s), FRI-quotient Coeffs regen (~0.55s), the
batched decommit re-LDE+gather (~0.7s) — the "regen because released" cost. Further
reduction is the shared-lease (regen once across composition→quotients under a VRAM
budget); this batched-gather fix is the biggest single lever and is landed.

INFRASTRUCTURE FIX (critical): build_and_push had been serving a STALE binary for
hours — rsync -t preserved local mtimes so cargo skipped rebuilding the stwo path-dep,
and a parallel-feature compile error was hidden by the script's `| tail`. Now touches
synced source + surfaces build failures. (Earlier "batched decommit didn't help" and
"PVT absent" results were the stale binary; corrected here.)

## 2026-07-06 — M6-a increment 1: --resident-pipeline harness + two-proof SEQUENTIAL baseline (H100 SXM)

The M6 gates are two-proof-WALL metrics; the existing --pipeline can't measure them
(it overlaps host-feed of a serial prover, not two provers). New --resident-pipeline N
harness proves N full proofs and reports twoproof_wall_s / per_proof_s /
sustained_useful_mhz / vram_peak_gb / feed_starved_s / proof_byte_equal.

Increment 1 (SEQUENTIAL baseline, M5c diet ON, G=32): two SN_PIE_2 proofs
**twoproof_wall_s = 25.24s**, per_proof 12.59s, sustained_useful_mhz 0.611,
vram_peak 31.6GB (two proofs fit an 80GB card), proof_byte_equal=true. Also
confirmed G=32 streaming-commit group default byte-identical (H_G32_MATCH; single
warm 11.64s ≈ G=16's 11.23s — group tuning within pod noise, neutral).

This 25.24s is the baseline the stream-explicit concurrent scheduler (M6-a
increment 2) must beat. Gates for the TWO-proof wall: <14.8s (beats non-diet
throughput), <11s (meaningful), <8s (strong 10 MHz signal). The concurrent
scheduler (CudaExecContext: per-proof streams + pool namespaces + event deps +
priorities, threading past the current stream-0-centric backend; round-27 flagged
pool concurrency as never-validated) is the next major multi-session build.

## 2026-07-06 — M6-a negative control: two-host-threads-calling-prove is UNSAFE (decisive)

`--resident-concurrent 2` (N host threads, each its own GpuCairoProver, bypassing the
GPU_NATIVE_CUDA singleton mutex) on SN_PIE_2 / H100 SXM, M5c diet on:
**one thread PANICKED** — `partial_ec_mul_window_bits_18.rs:38` (`assert!(!packed_inputs
.is_empty())`) — the component received ZERO inputs under concurrent proves. So the
HOST witness-generation/feed path is NOT re-entrant across two concurrent proofs
(the other proof completed, 15.06s). This is not mere serialization — it is a hard
correctness failure. Fully vindicates the reviewer directive: two host threads
calling prove is a NEGATIVE CONTROL, not M6.

Sequential baseline re-confirmed same session: --resident-pipeline 2 = 26.32s wall
(14.29 + 11.99, first proof cold-tainted), proof_byte_equal=true, 30.9GB peak; warm
single 12.33s. Consistent with the 25.24s prior baseline.

ARCHITECTURAL IMPLICATION (drives the roadmap): the host witness path (the fp256/EC
family — partial_ec_mul/blake_round/pedersen — is still HOST) is non-reentrant, so a
throughput scheduler CANNOT overlap two proofs' host witness generation even with
per-proof streams. Therefore:
  - The resident scheduler must run ONE host orchestration thread scheduling DEVICE
    work from two DeviceProofStates (device kernels overlap on separate streams), NOT
    two host prove() paths.
  - Moving the witness to device (component #1, device witness DAG) is a PREREQUISITE
    for meaningful two-proof throughput, not just for sub-2s single-proof latency.
  - Keystone-first (#2 fused commit + #1 device witness) is the right sequencing;
    stream plumbing alone is capped by the non-reentrant host witness path.
