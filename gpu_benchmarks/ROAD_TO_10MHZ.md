# The road to 10 MHz+: full-scope design

*The complete engineering inventory for delivering 10 MHz proving frequency
(0.1 µs/cycle) in one round — with the cost model that says what is required, what
is sufficient, and what the risks are. Supersedes the three-factor sketch in
RESULTS.md round 7; the verdict below is that the three factors are necessary but
NOT sufficient — five programs are, plus one measurement gate.*

## 0. Definitions and the target

- **MHz** = VM cycles / `prove_cairo` wall seconds (the prover span; the VM and
  adapter are outside it, but Program 5 puts them back on the table for sustained
  throughput).
- Operating point: fib 2M–8M on H100-class hardware, where round 7 measured the
  amortization plateau (~2.2 MHz, ~0.45–0.5 µs/cycle). All budgets below are at
  the 8M point (58.7M cycles): **10 MHz ⇔ total prove ≤ 5.9 s** (today: 28.9 s).

## 1. The per-cycle cost model (measured, H100, round-7 build)

| phase | today (8M, est. from spans × curve) | µs/cycle | share |
|---|---|---|---|
| host witness writes (base + interaction loops) | ~11.6 s | 0.198 | 40% |
| commits: NTT + Merkle + pack (GPU) | ~7.2 s | 0.123 | 25% |
| STARK core: composition, OODS, FRI, decommit (GPU) | ~5.8 s | 0.099 | 20% |
| residue: finalize chains, channel ops, readbacks, claim plumbing | ~4.3 s | 0.073 | 15% |
| **total** | **28.9 s** | **0.49** | |

10 MHz budget: **0.10 µs/cycle**. The plan must remove ~0.39 µs/cycle.

## 2. The five programs

### P1 — Witness-on-GPU, complete (removes ~0.19 µs/cycle → ~3.9 MHz)
Phase-1 spec (memory_id_to_big vertical slice) is at formula level in
WITNESS_ON_GPU.md. The COMPLETE program, in dependency order:

1. **Adapter output up once**: dedup'd memory tables + packed per-opcode input
   arrays (StateTransitions) — ~1–3 GB at 8M, pinned, one transfer.
2. **Table components** (memory_id_to_big, memory_address_to_id, range-check
   families, verify_instruction): pure per-row kernels from device tables;
   device count tables (atomic) replace host AtomicMultiplicityColumns; the
   rc input→row LUT uploads once (it is a table-layout map, NOT closed-form).
3. **Opcode components** (the per-program hot set; fib needs ~6, full Cairo ~20
   for >95% row coverage): per-row decode+arithmetic kernels reading the device
   memory tables directly (the host `deduce_output` becomes a device table
   lookup); their input pushes to sub-components become device atomic counts.
4. **Interaction write loops on device**: per component, a denominators kernel
   (`combine` over device-resident base columns / lookup tuples — the combine
   arithmetic is already byte-equal-proven on device by the constraint-JIT lane)
   + device numerators, feeding `finalize_raw_logup` through a
   `DeviceRawLogupColumn` variant (no H2D pair upload). This deletes
   `lookup_data` from the host entirely and flips W1's traffic theorem fully
   GPU-side.
5. **Per-component gates**: `STWO_CUDA_WITNESS_VERIFY=1` differential (device vs
   host writer, column byte-compare) + SIMD fallback per component; promotion on
   0 mismatches; Cairo e2e byte-equality as the global gate.

Coverage strategy: components not yet ported keep today's host path (W2 streams
their upload) — correctness never gated on coverage; MHz grows with it.

### P2 — Core kernel round (target: commits+core 13 s → ≤4 s; +~0.15 µs/cycle back)
**Gated by the measurement program (M1) below — optimize what ncu says, not what
we guess.** Candidate inventory by expected yield:
- **NTT**: stage fusion (more butterfly layers per launch via shared memory),
  M31-specialized arithmetic (32-bit, no Montgomery), register-resident small
  NTTs, batched cross-column launches (already grouped — deepen to multi-stage).
- **Merkle blake2s**: protocol-fixed hash (soundness — NOT negotiable); wins are
  memory-side: hash leaves directly from NTT output (fuse the pack pass),
  layer-pair hashing in one kernel, occupancy tuning.
- **Quotients/OODS/FRI**: quotient-combine + first FRI fold fusion; fold chain at
  small sizes via one persistent kernel (or CUDA graph) instead of per-layer
  launches.
- **JIT constraint kernels**: register pressure audit per component (`ncu
  --section Occupancy`), `__launch_bounds__` per-kernel instead of global 128.

### P3 — Multi-stream overlap (removes much of the 15% residue; required for P2's
fusion to land fully)
The stream-ordered rewrite serialized everything on the legacy stream — correct
and fast to build, but single-stream. The Fiat-Shamir chain forces commit
ORDER, not GPU serialization between independent column groups:
- Per-tree column groups NTT/hash on separate streams, event-joined before root
  mixing.
- The 4 secure coordinates (quotients, FRI, prefix sums) are independent — 4
  streams.
- H2D witness uploads (P1's leftovers + adapter tables) overlap compute via the
  pinned staging lane (already pinned; add async + events).
Discipline: events at every host-read and every cross-stream dependency — the
Metal lesson, now with a written stream contract per kernel family. This is the
highest-risk program (the bug class is silent corruption); it lands behind the
repeated-prove byte-equality gate which is specifically sensitive to it.

### P4 — Hardware envelope (multiplier on everything above)
- If post-P1/P2 the profile is bandwidth-bound (likely: witness kernels + NTT are
  streaming passes), the ceiling is HBM: H100 3.35 TB/s → B200 ~8 TB/s ≈ 2.4×.
  A 5090 (1.8 TB/s, 32 GB) plateaus lower but cheap-validates the consumer story.
- Multi-GPU (last resort, real design cost): tree-level split (preprocessed/base/
  interaction commits on different devices, FRI on one) — only if single-device
  stalls short of target.
- VRAM at scale: 8M fits 80 GB; 16M needs the low-memory mode revalidated under
  P1-P3 (it exists and is byte-equality-proven; cost was +13% at log 22).

### P5 — Throughput pipelining (the "+ beyond" lever)
For sustained proving (proofs/stream), the VM run + adapter of proof N+1 execute
on host CPUs while proof N occupies the GPU — they share no state. A two-deep
pipeline turns the (out-of-span but real) 10+ s of VM/adapt at 8M into ~zero
amortized cost and lifts *sustained* MHz ~1.3–1.6× beyond single-proof MHz.
Implementation: an orchestrator API around prove_cairo (no prover changes), so
zero soundness surface.

### M1 — The measurement gate (before P2/P3 effort is spent)
- `ncu` roofline on the top-10 kernels at log 24 (compute vs memory bound, %
  of peak). If pods block profiling counters (`--cap-add` denied), fall back to
  nsys timeline + occupancy from launch configs + bandwidth math per kernel.
- Re-trace spans at 8M post-P1-phase-1 to re-rank.

## 3. Does it reach 10 MHz? The arithmetic

| program | µs/cycle removed | running µs/cycle | MHz |
|---|---|---|---|
| today | — | 0.49 | 2.0 |
| P1 complete | −0.19 (writes) −0.04 (residue share) | 0.26 | 3.8 |
| P2 (commits+core to roofline, conservative 2×) | −0.11 | 0.15 | 6.7 |
| P3 (overlap: residue + idle gaps, −0.03) | −0.03 | 0.12 | 8.3 |
| P4 (B200-class bandwidth, ×~2 on the bw-bound 70%) | −0.04 effective | **~0.08** | **~12** |

**Verdict**: P1+P2+P3 on H100 lands ~7–8.5 MHz; **10 MHz+ single-proof requires
P4's bandwidth jump OR a >2× P2 outcome** (possible if ncu shows kernels far from
roofline — unmeasured today, hence M1 gates it). P5 pushes *sustained* throughput
past 10 MHz even on H100. Risks, ranked: P3 silent-corruption class (mitigated by
the repeated-prove gate), P2 yield uncertainty (mitigated by M1 first), P1 opcode
long-tail effort (mitigated by per-component fallback — coverage is incremental
by construction even if the program isn't).

**Round-12 update (2026-06-12):** the W3 opcode cohort landed (six fib opcodes
device-resident). Measured on a community 3090: fib 1M **0.224 µs/cycle
(4.47 MHz)**, 2M **0.217 µs/cycle (4.61 MHz)** — the P1-complete row of the
table is REAL and overshot (the table predicted 3.8 on H100-class). The
opcode-cohort lever alone is +46% (kill-switch isolated). Remaining µs/cycle
now sits in commits + STARK core (P2/ncu round, still gated on secure-cloud
counters), host feeds (device `index_count` next), and the rc/builtin
component long tail. Measured MPS dual at the final
stack: **7.43 MHz aggregate on one 3090** → two cards ≈ 14.9 MHz at $0.44/hr;
**10 MHz ≈ $0.30/hr** (round 9: $1.00; round 11: $0.44).

## 4. Execution order for the round

1. P1 phase-1 (memory slice, spec ready) → gate → re-trace (feeds M1).
2. M1 measurement session (same pod) → P2 worklist ranked by data.
3. P1 phase-2 (remaining tables + fib's opcode set) in parallel with P2 items.
4. P3 streams behind the repeated-prove gate.
5. P4/P5 validation runs (B200/5090 stock permitting; pipeline orchestrator).
