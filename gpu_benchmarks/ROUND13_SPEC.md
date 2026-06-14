# Round 13 specification — Starknet PIEs become the benchmark; the measured road from 4.5 to 10 MHz

## REPRIORITIZED 2026-06-14 (measured on H100 with the 4 `stwo-things/SN_PIEs/`)

The four SN_PIEs (Starknet OS, 7.7M–14.6M steps, keccak=ecdsa=0) run end-to-end
on the existing simple-bootloader path; bootloader inflation is only ~+1.6%.
A live span trace of the first GPU prove found the real bottleneck — and it is
NOT what items A–C below assumed:

1. **#1 — JIT constraint-kernel compile is the blocker.** Every one of the 29
   components (all opcodes AND all builtins: pedersen, poseidon, range_check,
   ec_op, partial_ec_mul, bitwise) evaluates constraints on the GPU **JIT lane**
   (`backend-cuda/src/backend/jit/`); ZERO fall back to CPU. But the JIT
   compiles one fused NVRTC kernel per component on first use, cached on disk at
   `$HOME/.cache/stwo-jit/sm{arch}_{hash}.ptx` keyed by AIR-content hash. The
   first prove on a cold cache stalls ~30 min single-threaded, GPU idle —
   dominated by a FEW giant kernels (`partial_ec_mul_generic` alone = 20+ min
   NVRTC; the EC double-and-add ladder is a huge straight-line program, and
   NVRTC's optimizer is superlinear). NVRTC opts are bare (`runtime_jit.cu:102`,
   no opt-level). fib never hit this (few simple components). The cache is shared
   across all 4 PIEs and every Starknet block.
   FIX (do first, it gates everything): (a) **pre-bake** the component PTX into
   the pod image — compile the ~29 kernels once offline (a warmup run or a build
   step), ship the `.ptx`, so production cold proves pay ZERO NVRTC; (b)
   **parallelize** the NVRTC compiles (currently sequential); (c) shipped stopgap
   `STWO_CUDA_JIT_SKIP=<names>` (stwo 26a2fca4) routes named giant/low-instance
   components (e.g. partial_ec_mul_generic, 581 instances) to the cheap CPU lane
   so a cold-cache prove completes in minutes. Consider also splitting the giant
   components' kernels or lowering NVRTC opt for them.

2. **#2 — host witness write is the steady-state lever.** With a warm cache the
   PIE_2 prove is dominated by the ~38.6s "Write Base trace" (host witness gen
   for the unported builtins + double_deref/imm/add_ap/mul opcodes). This is the
   Item-C lane-port work, now ranked by SN-PIE counts (below). Interaction write
   3.3s, all commits <1s each, composition (warm JIT) + FRI pending a warm-cache
   measurement.

3. Then items A–C below (A's bootloader is DONE; C is the witness ports, ranked
   by the measured SN-PIE opcode/builtin counts in the memory note). Get a clean
   warm-cache baseline number FIRST (deploy the skip-env or pre-bake), then port.

Receipts: stwo-things/round13b_*.log. pie_bench clone fix at stwo-cairo 7159d6ef;
skip-env at stwo 26a2fca4 (not yet pinned). Metal WIP parked in
stwo-things/metal-wip-backup/.

---

*Authoritative work order. Process formalities unchanged (they have held for four
rounds): local SIMD gates -> push + rev bump -> watchdogged pod round ->
STWO_CUDA_WITNESS_VERIFY differential where applicable -> CUDA-vs-SIMD proof
byte-equality (decisive) -> A/B + nsys -> RESULTS entry with receipts archived
to stwo-things/. Soundness rules per CLAUDE.md. Two standing lessons from round
12: (a) host FEEDS are part of every port — write them with kernel-level care
(rayon-parallel, scalar table decodes, never per-row packed broadcasts);
(b) bump BOTH stwo pin sites (workspace + crates/prover/Cargo.toml) or the test
link dies silently on duplicate kernel archives.*

## 0. Benchmark refocus (the WHY of this round)

fib was the right vehicle while the witness whale was opcode-shaped. It is now
device-resident (+46% lever) and fib risks overfitting: real Starknet blocks
are BUILTIN-shaped. New primary benchmark: the six Sepolia PIE bundles in
`~/Downloads/sepolia_near_step_target_pies_10m_to_60m` (12.45M -> 60.1M
n_steps; per-PIE builtin counters in its README — range_check 0.64M->3.2M,
pedersen 61k->321k, bitwise 44k->270k, poseidon 15k->84k, rc96/add_mod/mul_mod
present, keccak 37->153, ecdsa 0). fib 1M/2M stays as the regression canary
(one line per pod round), NOT the optimization target.

Proving model (CORRECTED after investigation): PIE -> the cairo-lang
v0.14 SIMPLE bootloader program (proof mode, `all_cairo_stwo` layout) ->
`adapt(&runner)` -> `prove_cairo`. Key findings that shaped this:
- The FULL bootloader does NOT simulate missing builtins; only the v0.14
  simple_bootloader.cairo main calls handle_uninitialized_{keccak,ecdsa,ec_op}
  (verify_builtins.cairo) — required because every Sepolia PIE carries a real
  keccak builtin segment and stwo has no keccak AIR / layout slot.
- The stone-era bootloader (0.13.0, 8 builtins) and even 0.13.3 (11 builtins,
  no simulation) CANNOT run these PIEs. The python bootloader runner modules
  (objects/utils) are unpublished — no python escape hatch.
- Rust cairo-vm has no dynamic auto-deduction rules — but CairoPie tasks do
  not need them: load_cairo_pie loads ALL builtin cells (inputs AND outputs)
  from the PIE memory, so re-execution only reads pre-loaded values; the
  auto-deduction-registration hints are sound NO-OPS for PIE tasks (guarded:
  error on RunProgramTask). Soundness of simulated builtins comes from the
  bootloader's pure-Cairo verification, which the proof covers.
- Keccak simulation inflates n_steps (the Cairo keccak-f per instance);
  measure the inflation (PIE n_steps vs adapted cycle_count).

## Item A — PIE proving path (bootloader integration + pie_bench)

1. Vendor Moonsong-Labs/cairo-bootloader into
   `stwo_cairo_prover/crates/bootloader` ported to cairo-vm 3.2 (workspace
   member `stwo-cairo-bootloader`; resources/bootloader-0.13.0.json embedded;
   `pub fn run_pie_with_bootloader(&Path) -> Result<CairoRunner>` mirroring
   gpu_bench's proof-mode CairoRunConfig). IN FLIGHT (delegated).
   Soundness note: the hint port is consensus-critical — every hint
   semantically faithful; the e2e gate is that adapt+prove+verify succeeds and
   the CUDA proof is byte-identical to SIMD on the same PIE.
2. `crates/prover/src/bin/pie_bench.rs` (drafted): gpu_bench-format JSON +
   vm_s/adapt_s split, `--counts-only` ranking dump (opcodes + builtin
   segments), full Canonical preprocessed trace (PIEs use pedersen; fib's
   bench drops the points tables — do NOT copy that).
3. Local smoke: tiny self-generated PIE through the path on SIMD.
4. First hardware baselines, in order:
   a. 10M PIE (`target10m_...zip`, 12.45M steps). Sizing estimate: 12.45M +
      bootloader overhead + keccak emulation (37 inst) ~ 14-15M steps ->
      padded 2^24 rows — fib-2M scale. Try H100-80 SXM first (known-good
      infra), capture peak VRAM; then test 24GB viability (4090, low-mem
      `STWO_CAIRO_LOW_MEMORY=1` if needed) — the consumer-card answer drives
      the $/MHz story.
   b. STWO_BENCH_TRACE=1 span breakdown + `--counts-only` component ranking
      on the same run: THE ranking data for Item C.
   c. 20M/30M PIEs on H100-80/H200 for the scale curve (2^25 rows); 60M
      documents the VRAM wall + the low-mem/sharding requirement; do not
      grind it this round.
   Receipts: per-PIE JSON + spans archived; RESULTS gets a PIE baseline table
   (MHz here = PIE n_steps / prove_s; report both raw-PIE steps and
   bootloader-inflated cycle_count — the honest denominator is the raw PIE
   n_steps, the prover does strictly more work).

## Item B — fib 10 MHz gap: measure, then close (generalizes to PIEs)

Current: 1.566 s @1M (4.47), 3.036 s @2M (4.61) on a $0.22 3090. 10 MHz needs
0.70/1.40 s. The remaining ~0.87/1.6 s, by bucket (estimates to be REPLACED by
a span re-trace at the final stack — first step, zero new code):

1. **Re-trace** (M-step): STWO_BENCH_TRACE span breakdown fib 1M/2M + 10M PIE,
   same pod round as Item A.4. Everything below re-ranks on this data.
2. **Device feeds (`index_count` + rc LUT counts on device)**: the host feed
   loops of the 6 ported opcodes + memory components are rayon-parallel but
   still host (vi dashmap feed = the data-dependent one; addr/id mults and rc
   counts are device-friendly atomics — the lane's tuple_count already proves
   the pattern). Expected: removes most of the remaining witness-side host
   wall; also cuts the per-port host-feed tax on every future builtin port.
3. **rc + small-component cohort on the lane**: the ~55 host components left
   (rc_6..rc_3_3_3_3_3, bitwise xor tables, etc.) — each is small but the
   long tail sums; the lane recipe makes these mechanical (most have LUT
   count tables already proven by rc_9_9/_4_3/_7_2_5).
4. **Warp-NTT kernel slice**: round12g data — family at 44% of roofline;
   n2b_final_warp<2>/<3> + b2n_init_warp<2>/<3> at 26-33% holding ~45% of
   family time; 1.6x family headroom = ~3% of today's wall (rank accordingly).
   Kernel-internal: vectorized (uint4) global access + smem bank-conflict
   elimination in the shuffle stages. Needs a counter-capable box for real
   iteration (RunPod blocks ncu everywhere: RmProfilingAdminOnly=1) — Lambda
   or local; otherwise iterate on nsys duration deltas only.
5. **Commit-over-witness scheduler, then streams**: the prerequisite for
   re-enabling the (validated-safe) stream pool with real work: schedule tree
   N's commit concurrently with component-group N+1's witness writes
   (Fiat-Shamir fixes commit ORDER only). This is the biggest structural
   overlap lever left; design doc before code.
6. **P4 reality check**: after 2+3, if the span trace shows >60% of wall in
   bandwidth-bound GPU kernels, the 3090's 936 GB/s is the wall and 10 MHz
   single-proof needs 4090/5090-class bandwidth — quote $/MHz across tiers
   (the fleet/MPS answer may remain the cost-optimal 10 MHz path regardless).

## Item C — builtin witness cohort (the PIE whale, ranked by Item A data)

The opcode lane recipe applies as-is (per-component base kernel + TupleSlot
interaction + parallel scalar feeds + differential + kill switch). Expected
ranking from the README counters (validate with A.4.b): range_check builtin
(0.64-3.2M instances), pedersen (its partial_ec_mul/aggregator/points-table
component family is the single biggest AIR), bitwise, poseidon chains, rc96,
add_mod/mul_mod. Pilot ONE (range_check_builtin: simplest, biggest count) in
the main session; delegate the rest with the round-12 delegation formalities
(+ the two standing lessons above). memory components already device-resident
feed these for free.

## Order of execution

1. A.1-A.3 (bootloader lands locally, smoke green) — prerequisite.
2. One pod round: A.4 baselines + B.1 re-trace (fib + 10M PIE) -> rank.
3. B.2 device feeds + C pilot (range_check builtin) — same pod round to gate.
4. C fan-out (delegated) + B.3 small-component tail — batched validation.
5. B.5 scheduler design doc; B.4 NTT slice when a counter-capable box exists.
6. Re-quote $/MHz on PIEs: single-card, MPS-dual, fleet; 24GB viability.

10 MHz definition going forward: PIE n_steps / wall second, single proof, on
the cheapest card that holds it — with the fib canary preventing regression.
