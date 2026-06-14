# JIT constraint-kernel compilation: the SN-PIE proving blocker (2026-06-14)

Authoritative writeup of the round-13 investigation into why the first GPU proof
of a real Starknet block (the four `stwo-things/SN_PIEs/`) appeared to stall for
~30 minutes. Method: live `tracing` span trace + `STWO_CUDA_CONSTRAINT_LOG` lane
log on an H100, cross-referenced with the JIT source. Conclusion up front: the
prover is fully GPU-capable for real Starknet; the blocker is one-time NVRTC
kernel **compilation**, which is an engineering/operational problem, not an
algorithmic one.

## 1. What actually happens during an SN-PIE prove

Pipeline (warm, from the span trace, SN_PIE_2 = 7.71M steps → 7.83M cycles):

| phase | time | lane |
|---|---|---|
| bootloader VM + adapt | ~8 s | host |
| Write Base trace (witness) | **38.6 s** | host (unported builtins + opcodes) |
| base trace commitment | ~0.8 s | GPU |
| Write interaction trace | 3.3 s | host + device finalize |
| interaction commitment | ~0.6 s | GPU |
| Composition (constraint eval) | **see §2** | GPU JIT — once compiled |
| FRI + decommit | (pending warm measurement) | GPU |

The constraint (composition-polynomial) evaluation dispatches per component
through `backend-cuda/src/backend/constraint_eval.rs::evaluate_constraint_quotients`,
which has three lanes:

1. **Precompiled GPU kernels** (`evaluate_constraint_quotients_on_domain`,
   FNV1a name-hash dispatch) — **DISABLED by default**. Comment in
   constraint_eval.rs: the NitrooZK kernel set was generated against stwo v2.1.1
   + their AIR rev; differential verification against this stack shows 100% of
   rows mismatched from row 0 (eval-struct layout / AIR-rev skew). No kernel is
   qualified.
2. **JIT lane** (`backend-cuda/src/backend/jit/`) — generated from THIS build's
   AIR via the recording evaluator, NVRTC-compiled, "consistent by
   construction." This is the lane that actually runs.
3. **CPU fallback** (`accumulate_pointwise_cpu`) — single-threaded reference.

`STWO_CUDA_CONSTRAINT_LOG=1` on the H100 confirmed: **all 29 components — every
opcode AND every builtin (pedersen, poseidon, range_check, ec_op,
partial_ec_mul, bitwise) — take `lane=JIT`. Zero CPU fallback.** Constraint eval
is fully on GPU.

## 2. The blocker: one-time NVRTC compilation

The JIT compiles one fused `__global__` kernel per component on first use
(`jit/cuda_codegen.rs::compile_v1_to_cuda_source` → `runtime_jit.cu::compile_kernel`
→ `nvrtcCompileProgram`), caching the PTX:
- in-memory: `JitCache { mutex, unordered_map<hash, CUfunction> }`
- on disk: `$HOME/.cache/stwo-jit/sm{major}{minor}_{hash}.ptx`, keyed by a
  content hash of the program mixed with `CODEGEN_VERSION`.

Because the cache key is the **AIR structure**, not the data, the kernels are
**identical across all four PIEs and every Starknet block** — compile once per
(GPU arch, codegen version), reuse forever.

Measured cost on a cold cache (H100, sm_90): **~90–120 minutes**, single-threaded,
**GPU 100% idle the whole time**, ~40 GB VRAM held. It is dominated by a handful
of MONSTER kernels in the EC/Pedersen/Poseidon family — `partial_ec_mul_generic`
alone took **20+ minutes** and never finished in a 20-min window; the other big
builtins are similar. These components have huge straight-line constraint
programs (the EC double-and-add ladder, poseidon rounds), and NVRTC's NVVM
optimizer is superlinear in program size. The opcodes compile fast (~1–2 min
each). fib never exposed this — it has few, simple components.

NVRTC is invoked with bare options (`runtime_jit.cu:102`):
`--gpu-architecture=compute_XX --std=c++14` — no optimization-level control.

GPU-idle + one busy thread for 30 min is **indistinguishable from a deadlock**
without observability — this cost real diagnosis time. `STWO_JIT_LOG=1` exists
(per-kernel timing) but was off by default.

## 3. Fixes, ranked by leverage

1. **Eliminate runtime compilation.** Two routes, do at least one:
   - **Pre-bake the PTX into the pod image**: compile the component set once
     offline (per arch + codegen version), ship the `.ptx` into
     `$HOME/.cache/stwo-jit`. Production cold-start → 0. Packaging only, no code
     risk. (Round-13 is harvesting this seed; partial 35-kernel seed already in
     `stwo-things/ptx-seed-sm90-partial-30.tar.gz`.)
   - **Regenerate the precompiled kernel set against THIS stwo-cairo AIR rev** so
     lane 1 qualifies — kernels baked into the binary, zero NVRTC, no PTX cache
     at all. The cleanest long-term answer if the generator is available.
2. **Parallelize NVRTC compilation.** Two serialization points:
   (a) `stwo_cuda_jit_eval_fused` holds `cache.mutex` across the *entire*
   compile — concurrent callers serialize; (b) the composition loop in
   `stwo/crates/stwo/src/prover/air/component_prover.rs::compute_composition_polynomial`
   is sequential. Fix: a compile-only FFI that compiles OUTSIDE the lock
   (lock only the map insert; double-compile is benign + idempotent) + a
   parallel pre-compile pass over components before the sequential
   accumulation loop. Turns ~90 min → ~max-single-kernel (~20 min) on any fresh
   machine. SAFE: compilation is deterministic + cached, and the lazy sequential
   path remains as fallback, so this can only speed things up, never corrupt.
3. **Shrink/cheapen the monster kernels.** `cuda_codegen.rs` emits one fused
   kernel with the whole constraint program inlined — pathological for EC/hash
   components, which run over TINY domains (581–25k instances) where compile
   cost dwarfs runtime. Lower NVRTC opt for them, or split into multiple
   kernels, or cap inlining. `STWO_CUDA_JIT_SKIP=<names>` (shipped, stwo
   26a2fca4) routes named components to CPU as a stopgap.
4. **Make the CPU constraint fallback rayon-parallel.** `accumulate_pointwise_cpu`
   is single-threaded; skipping even a low-instance component to CPU costs more
   than expected (observed ~minutes for the EC/hash family on SN_PIE_2) because
   it evals all constraints per row over the blown-up component domain on one
   thread. SimdBackend's composition is parallel; the Cuda CPU-fallback should be
   too.
5. **Steady-state lever, once compile is solved: the 38.6 s host witness write.**
   The unported builtins + `assert_eq_double_deref` (1.79M rows, 19 cols),
   `assert_eq_imm`, `add_ap`, `mul_small`, `jump_rel_imm`, `mul`, `jnz_non_taken`
   opcodes — all small, all the proven round-12 lane recipe (Item C).

## 4. Operational / setup lessons

- **Benchmark on representative workloads from day one.** fib masked a
  90-minute production wall and a totally different component/witness profile.
  The "4.47 MHz on fib" result does not transfer to Starknet blocks. The four
  SN_PIEs are the benchmark now.
- **The iteration loop is too expensive** (~$6–7 / pod round on $3.29/hr H100):
  9.5-min nvcc kernel rebuild + 90-min cold compile every time. Pre-baked image
  (warm PTX + a stable kernels artifact so a Rust-only change doesn't
  re-fingerprint and rebuild the nvcc archive) would make iterations minutes.
  Decouple the kernels-crate rebuild (ccache for nvcc, or commit the archive).
- **Default-on observability** in the bench harness: `STWO_BENCH_TRACE=1` (span
  breakdown), `STWO_CUDA_CONSTRAINT_LOG=1` (lane per component), `STWO_JIT_LOG=1`
  (per-kernel compile timing). These localize a stall in seconds.
- **Profiling is blocked on RunPod** (`ncu` → `ERR_NVGPUCTRPERM`,
  `RmProfilingAdminOnly=1`, secure cloud included); nsys works. SASS
  occupancy / register-pressure work on the constraint kernels needs a
  counter-capable box (own machine / Lambda).
- **SN_PIE facts**: keccak=ecdsa=rc96=add_mod=mul_mod=0 in all four → the
  bootloader missing-builtin simulation is trivial (~+1.6% cycle inflation); the
  v0.14 simple-bootloader path runs them unchanged. pie_bench at stwo-cairo
  7159d6ef; STWO_CUDA_JIT_SKIP at stwo 26a2fca4 (pinned at dae13425).

## 4b. UPDATE (round 13e, H100 validation) — codegen is a SECOND bottleneck

Implemented + validated the parallel pre-compile (stwo d807e7a4; prelude in
`compute_composition_polynomial` + `stwo_cuda_jit_compile` FFI that compiles
outside the cache mutex). Result: **the parallel COMPILE works — 28 of ~29
kernels compiled in ~111 s vs ~60 min sequential**, and the full set (46 PTX,
incl. partial_ec_mul) warmed in ~14 min. Complete sm_90 PTX seed harvested:
`stwo-things/ptx-seed-sm90-complete-46.tar.gz` (the pre-bake artifact).

BUT a warm-cache traced prove (cache hot, zero NVRTC) still spent ~15-30 min
**single-threaded in composition before the eval loop even started** (lanes=0,
1 running thread, GPU idle, only ~10 cache-hit "ready in" lines). So the
residual cost is NOT compilation — it is the Rust-side **lowering/codegen**
(`lower_framework_eval_to_v1_with_logup` + `compile_v1_to_cuda_source`), which:
- is single-threaded and expensive for the constraint-heavy components
  (partial_ec_mul / pedersen / poseidon: thousands of symbolic constraint ops);
- runs **every prove** (it computes the semantic hash that keys the PTX cache),
  so even pre-baked PTX does NOT avoid it;
- was **doubled** by the always-on prelude (codegen in the prelude AND again in
  the lazy eval lane). The prelude is now gated behind
  `STWO_CUDA_PARALLEL_JIT_WARMUP` (default OFF, stwo 7232f49a) so the default
  warm path is unregressed.

**NEW #1 fix: cache the lowered program per component** (key on the component's
type + structural params, stable across proves) so codegen runs ONCE — reused
by both the prelude and the eval lane, and across all proves/PIEs. This is what
actually makes warm SN-PIE proving fast; the parallel-compile + PTX-seed work
only addressed the (separate) NVRTC cost. No warm SN-PIE MHz number yet — the
codegen bottleneck blocks it; it is the immediate next task.

## 4c. UPDATE (round 13f, H100/sm_90) — the REAL bottleneck: load-time ptxas

§4b's "codegen is the bottleneck" conclusion was **WRONG**, and this round's
per-phase instrumentation proves it. I added a process-global codegen cache
(stwo 930d3f02: cache CUDA source/name/PTX-key per (component, log_size); the
eval lane reuses it, the prelude skips lowering on a hit) AND split timing for
`lower` (recording + compaction) vs `source_gen` (`compile_v1_to_cuda_source`)
vs module load. With a pre-seeded cache on SN_PIE_2 (STWO_JIT_LOG=1):

- **Codegen is essentially free: `lower=0-2ms source_gen=0-2ms` for EVERY
  component**, including the monsters (partial_ec_mul: 12625 base + 5503 ext
  insts → still `lower=2ms source_gen=2ms`). So the codegen cache, while correct
  and harmless, is NOT the fix. §4b mis-attributed the per-kernel "ready in" time
  (which is the MODULE LOAD) to "codegen" because it had no phase split.

- **The real cost is the CUDA driver assembling PTX→SASS (ptxas) inside
  `cuModuleLoadDataEx`**, paid on EVERY cold process — even on a warm disk PTX
  cache hit, because PTX is an IR and the driver JITs it at load. Measured
  per-kernel `ready in` (SN_PIE_2, sm_90):

  | kernel | load (ptxas) | source |
  |---|---|---|
  | (builtin) | **164 s** | disk PTX cache hit |
  | (builtin) | 75 s | disk PTX |
  | ec_op | 43 s | NVRTC |
  | (builtin) | 37 s | disk PTX |
  | (builtin) | 29 s | disk PTX |
  | (next ~6 kernels) | 2.1 s → 0.5 s | mixed |
  | **partial_ec_mul_generic** | **>24 min, never finished** | NVRTC |

  ~350 s for the builtin family PLUS the partial_ec_mul marathon (>24 min of
  ptxas for ONE kernel — the EC double-and-add ladder, 12625+5503 insts fully
  inlined), GPU 100% idle throughout. THIS is the "15-30 min single-threaded
  composition stall," and it is why a warm PTX seed did not help. After the top
  5-6 monsters the load times fall off a cliff (~2 s and below).

**partial_ec_mul is expensive on BOTH lanes** (round-13f, confirmed): routing it
to CPU via `STWO_CUDA_JIT_SKIP=partial_ec_mul_generic` did not rescue the run —
`accumulate_pointwise_cpu` then spent 8+ min on it (2^18-row blown-up domain ×
the full ladder, even rayon-parallel). So a clean all-GPU warm SN-PIE number is
**blocked on partial_ec_mul** until the cubin fix makes its (one-time) load fast;
on CPU it dominates instead. No representative warm MHz this round — correctly
deferred to the cubin-validation round (which bakes its cubin once anyway).

**THE FIX (stwo b3d48fbb): cache CUBIN, not PTX.** Compile with a real arch
(`--gpu-architecture=sm_XX`, not virtual `compute_XX`) and cache the final cubin
via `nvrtcGetCUBIN`. `cuModuleLoadDataEx` on a cubin loads SASS directly — no
JIT, milliseconds. The one-time ptxas cost moves into the (disk-cached, and
prelude-parallelizable) compile. Cubins are arch-specific; the `sm%d%d` cache
key already separates them per GPU. Implication: the pre-bake **seed must ship
cubins** (per arch), not PTX — a PTX seed re-JITs on every load and is nearly
worthless. The codegen cache (930d3f02) stays (cheap, helps long-running
multi-prove processes) but is not load-bearing for the stall.

## 4d. UPDATE (round 13g, H100/sm_90) — cubin fix VALIDATED + first warm number

The cubin cache + ptxas opt control (stwo 5e27c30b: NVRTC->PTX + cuLink with
`CU_JIT_OPTIMIZATION_LEVEL`, `-O1` for kernels with source >150KB, cached as
`.cubin`) is validated end-to-end on SN_PIE_2 (7,833,306 cycles):

- **ptxas opt control kills the bake blowup.** partial_ec_mul + 5 other EC/hash
  kernels exceeded 150KB and baked at -O1; partial_ec_mul's cubin compiled in
  ~2 min (vs >24 min, unbounded, at -O3). 6 of 50 kernels went -O1, the rest -O3.
  Full cold bake of all 50 cubins: ~9.5 min (rep0 prove_s 575s includes it).
- **cubin cache kills the load-time ptxas.** Fresh process with a warm disk cubin
  cache: the kernels that took **164s / 75s / 37s as PTX now load in 3ms / 3ms /
  2ms** (cuModuleLoadDataEx on a cubin = direct SASS load). ~50,000x. 50 cubins =
  18 MB on disk (3.4 MB gzipped: `stwo-things/cubin-seed-sm90-complete.tar.gz`).
- **Composition collapsed from 15-30 min to 1.9 s** ("Prove STARKs" span, incl.
  FRI). The stall is GONE.

**FIRST WARM SN-PIE NUMBERS (SN_PIE_2):**
- warm (in-process repeat, all kernels on GPU): **18.4 s = 0.426 MHz**
- cold process (warm disk cubin cache, production single-shot): **36.8 s = 0.213 MHz**
- verify 16 ms, proof 2.99 MB, peak RSS ~30 GB, VRAM 44.5 GB.

**The new bottleneck is the host witness write**, `Write Base trace = 28.8 s`
(cold) / ~10 s (warm) — the unported builtins + opcodes (assert_eq_double_deref
1.79M, assert_eq_imm, add_ap, mul_small, jump_rel_imm, mul, ...). Composition is
no longer on the critical path. Item C (port the witness whales, the round-12
lane recipe) is now THE lever toward higher MHz.

## 5. The plan from here

1. **Item C: port the SN-PIE witness whales** to cut the 28.8 s Write Base trace
   — now the dominant cost. assert_eq_double_deref (1.79M, the new whale) first.
2. **Re-bake the cubin seed per arch** (sm_86/89 in addition to sm_90) and ship
   in the pod image so cold production proves load in ms with zero bake. The
   sm_90 seed is harvested; sm_86/89 need a bake on those GPUs.
3. Tune the 150KB opt-level threshold if any -O3 kernel still bakes slowly, and
   consider feeding the instruction count from Rust instead of a source-size proxy.
4. THEN chase MHz on the real workload (Item B device feeds / scheduler) once the
   witness write is down.
