# Known issues surfaced by the SN-PIE benchmark round (round 8)

Both issues below are pre-existing and were invisible to every fib-era benchmark
(rounds 1–7): they require components only real Starknet workloads instantiate.

## 1. Pedersen points-table `LazyLock` × rayon deadlock (liveness, production-relevant)

`crates/common/src/preprocessed_columns/pedersen.rs` defines `PEDERSEN_TABLE_9` /
`PEDERSEN_TABLE_18` as `LazyLock` statics whose initializer runs rayon parallel
iteration. If the first `LazyLock::force` happens **on a rayon pool worker** (as it
does when pedersen-heavy witness generation races table init), the worker blocks on
the `Once` while the initializer needs pool workers → deadlock (~0% CPU, prove hangs
forever). Reproduced twice on multi-task PIE runs; confirmed by stack sampling.

- Harness workaround: `gpu_bench` calls `prewarm_pedersen_tables()` (main-thread
  `LazyLock::force` before proving) whenever the `Canonical` preprocessed variant is
  selected.
- Real fix (upstream, prover team): initialize the tables outside rayon context in
  `prove_cairo` itself, or build them without rayon inside the `LazyLock` initializer.
  Any production caller whose first pedersen proof runs witness gen cold is exposed.

## 2. JIT fused constraint kernel: unbounded NVRTC compile (cold-start liveness)

The JIT constraint lane fuses each component's constraint evaluation into one CUDA
kernel compiled by NVRTC (options hardcoded: `--gpu-architecture`, `--std=c++14`;
`backend-cuda-kernels/cuda/runtime_jit.cu`). On the first SN-PIE CUDA prove
(RTX 3090, CUDA 11.8), 27 kernels compiled/cached normally, but one PIE-only
component (suspected pedersen `partial_ec_mul`-class or a poseidon round chain)
produced source large enough that `nvrtcCompileProgram` pegged one core for 45+
minutes without terminating — deterministic, silent (the JIT log only prints AFTER
success), and reproduced across three runs. The prove appears "hung at 0% GPU" while
the base-trace and interaction commitments have already completed on the GPU.

- Diagnosis trail: `STWO_BENCH_TRACE=1` streaming spans place the stall after
  "Compute interaction trace commitment" (i.e. entering composition);
  `/proc/<pid>/stat` utime shows exactly one core of user CPU; all 15 other threads
  parked in futex/poll. `STWO_CUDA_DISABLE_STREAMS` and `STWO_CUDA_MEMORY_WITNESS=0`
  do NOT affect it (exonerating the P3 stream pool and the W3 memory-witness lane).
- **FIXED (2026-07-02, stwo fork)**: (a) memoized the recorder's O(n²) zero-const
  rescan (codegen for the 20,334-instr `partial_ec_mul_generic` dropped to 3ms);
  (b) size governor splits oversized fused kernels into K sequential kernels with
  bit-identical accumulation (default cap 2048 instrs — calibrated: NVRTC is fine at
  8192/0.39MB source (~9s) but **ptxas ran >20 min on the resulting 6.4MB PTX**; at
  2048 the component becomes 11 kernels compiling in seconds, +4% instruction
  duplication); (c) `STWO_JIT_NVRTC_OPTS` + per-kernel optimization relief;
  (d) CODEGEN_VERSION 2→3. Validated on RTX 3090: 10-transfer PIE CUDA prove+verify
  passes, warm 6.1s; the governor also caught `partial_ec_mul_window_bits_18`
  (8,904 instrs → 5 kernels), which would have hung next.
- Note: NitrooZK never JIT-fused pedersen constraint evaluation (handwritten kernel
  lane) — consistent with them never hitting this.

## 3. Batched NTT: column-batch axis on grid.y/grid.z overflows at >65,535 columns

`rfft.cu`/`ifft.cu` batched multi-column NTT launchers map the same-log-size column
group count onto `grid.y`/`grid.z` (CUDA cap: 65,535). The 14M-step SN PIEs' Canonical
column mix produces a small-log column family larger than that →
`cudaErrorInvalidValue` at `rfft.cu:667` (the launch-error check after the
`ntt_n2b_stage_batch` launch). fib never hit it: few huge columns, never >65k in one
size bucket; the older single-column `batch_rfft` kernels correctly used grid.x.
**FIXED (2026-07-02)**: batch axis tiled into ≤65,535-column chunks in both the Rust
dispatchers (`poly.rs`, unit-tested) and the CUDA entry points (defense in depth);
`num_poly` is not used for indexing in any kernel body, so chunking is provably
result-identical; single-chunk path is byte-identical to before. Hardware validation:
sn1/3/4 rerun pending.

## Benchmarking discipline reminders

- NitrooZK published PIE numbers use `n_queries=3` (~29-bit); ours use the secure
  96-bit config (`pow_bits=26`, blowup 1, 70 queries, fold_step 3 — fold_step is
  security-neutral). Never compare without normalizing.
- Only same-pod comparisons are meaningful (community host variance).
- `useful_mhz` (PIE `n_steps` basis) is the business metric; `mhz` (proved cycles,
  incl. bootloader overhead — measured 3.51% on SN_PIE_2 single-task) keeps
  continuity with the fib-era tables.

## 4. CUDA proof bytes nondeterministic in the FRI-decommit tail (soundness-neutral, breaks the byte-equality gate)

Two identical lane-OFF CUDA proves of SN_PIE_2 produce proofs differing from byte
2,291,852 onward (~0.79MB tail); the 2.29MB commitment/Fiat-Shamir prefix is
byte-identical (SHA256-verified). Both proofs verify. Hypothesis (unconfirmed): the
GPU PoW grind's host loop reads `result_low` when the found-flag first appears,
racing in-flight `atomicMin` updates within the batch → a valid but non-minimal
nonce → different query positions → different decommit tail. Soundness-neutral (any
qualifying nonce verifies) but it DEFEATS the repo's primary correctness instrument
(proof byte-equality vs SIMD). Fix candidates: synchronize the batch fully before
reading the result, or derive queries deterministically regardless (min-nonce
reduction). Until fixed, PIE-lane gates use: in-harness verify + byte-identical
commitment prefix hash + differential witness compare (the round-8/9 formulation).

### Issue 4 CORRECTION (round-9 session-1 falsification)
The grind is EXONERATED: instrumented nonces are byte-identical across runs
(8,590,189,642 @pow24; 107,374,471,375 @pow26); the added drain is byte-neutral
(kept for future stream safety). The divergence is BACKEND-INDEPENDENT (SIMD×2
diverges at the same offset: 10-transfer @2,194,060; SN_PIE_2 @2,291,852),
lives in the post-commitment value tail, and both proofs verify against
identical roots — consistent with SERIALIZATION-ORDER nondeterminism of
identical content (cf. the round-6 adapter HashMap-order finding), not value
nondeterminism. Next diagnostic: diff the two proofs structurally (deserialize
both, compare field-by-field) to locate the reordered collection; fix is a
deterministic ordering (BTreeMap/sort) in the proof struct — needs prover-team
review (proof serialization is consensus-adjacent). Until then: commitment-
prefix sha256 + verify remains the gate.

### Cubin cache verdict (session 1, measured)
nvrtcGetCUBIN cold = 11× the PTX path (1273.7s vs 115.7s, 10-transfer, A40) —
full SASS gen at compile time; warm fresh-process prove with populated cache =
13.8s total (JIT 0.5s, zero NVRTC/ptxas). Default flipped OFF; the fleet
pattern is CACHE SHIPPING: pre-populate once per arch (sm-tagged) during pod
provisioning (fleet/pod_provision.sh), transfer like the PTX cache. Parallel
compile stays default ON (~1.5× cold, correctness-neutral, A/B/C prefix-hash
identical `eae44db2d35303fe`).

### Issue 4 CONFIRMED (architect analysis, 2026-07-02)
Direct multiset analysis of the two divergent SN_PIE_2 proofs (offset 2,291,851,
tail 786,944 B): tail BYTE multiset equal, tail U32-WORD multiset equal, 16B-chunk
multiset NOT equal, 652,293/786,944 bytes differ. The tails are the SAME u32 words
in a DIFFERENT order → serialization-order nondeterminism of identical content
(per-process HashMap RandomState in a decommit/witness collection, serialized in
iteration order). Both verify because the verifier consumes it as the same logical
map. Fix: canonical ordering (BTreeMap or sort at serialize) in the proof-struct
serialization path — reorder-only, no format change, but prover-team review since
it is consensus-adjacent. Restores the full byte-equality gate when landed.

## 5. Prove-lane review finds (round-10, caught pre-hardware): the parity-gate lesson

Two proof-breaking bugs shipped in the "compiles clean" JIT-witness prove wiring and
were caught by hostile re-review BEFORE any GPU hour, not by the gates then in place:

* **Sub-input feed truncated at n_real.** The host writer feeds the FULL padded
  extent (`mults_0 = 1` on every row — the enabler only gates `mults_1`/opcodes).
  A lane feeding fewer rows than its interaction trace emits uses for unbalances
  logup → claimed_sum ≠ 0 → verify fails. Root cause: padding semantics inferred
  from the enabler's meaning instead of read from the generated writer.
* **Word-major accessors over row-major kernel stores.** The prove accessors and
  the codegen were written in different sessions against a doc comment that said
  "word-major" while the emitter wrote row-major. The comment was the only
  contract. Fix flipped codegen to word-major (coalesced + contiguous PackedM31
  repack) and bumped WITNESS_CODEGEN_VERSION so no disk PTX cache serves stale
  layout.

Postmortem: the hardware selftest verified COLUMNS only — lookup/sub words were
D2H'd nowhere and compared never; "hardware-proven" silently meant "one third of
the data contract". Fixes: (a) selftest now byte-checks cols + lookup + sub flats;
(b) a LOCAL parity gate (`add_opcode_prove_accessors_match_host`) replays the
exact prove data flow through the reference interpreter — padding rows, enabler,
word-major flats, production accessors — against the host writer, no GPU needed;
(c) the prove launch fails CLOSED (unresolved pc, mult tables, >2048 instrs, shape
mismatch vs the recording) and logs every fallback reason. RULE going forward: a
lane ships with its local parity gate IN THE SAME CHANGE, and "hardware-proven"
claims must name which buffers were compared.
