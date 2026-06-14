# L3 sharding design proposal: sound near-linear multi-GPU/cluster proving (task #40)

**Status: research design proposal for expert/human review. NOT a validated sound
protocol. Soundness is absolute (an accepted invalid proof is unrecoverable); the
soundness obligations in §5 MUST be discharged by a cryptographer before any code.**

This is the architecture for the 20×-and-beyond goal: a single large PIE's proof
is split into N shards proved in parallel (on N GPUs / a cluster) and merged
recursively. The adversarial review (`SCALING_ARCHITECTURE_REVIEW.md` §3) showed
the deck's "N independent shards" framing is unsound; this proposal fixes the
specific hole it identified.

## 1. Why sharding is the 20× lever
Single-GPU full witness offload (ports + builtins + device feeds) gets SN_PIE_2
to roughly the deck's fully-GPU per-step rate (~1 MHz, ~2–3× from today's 0.45).
The remaining ~10× to reach ~9 MHz requires parallelism that the prove pipeline's
strict Fiat-Shamir chain forbids *within* one proof (review §2). Sharding gives it
*across* proofs: N shards prove concurrently with no inter-shard Fiat-Shamir
dependency, then a cheap recursive merge. Throughput AND single-block latency
scale ~linearly in N (minus the merge tail). It is also the only way a single
large block ever proves on a 24 GB 3090 (each shard fits).

## 2. The soundness hole to close (recap)
Cairo correctness rests on ONE global check: `lookup_sum(...) == 0`
(`cairo-air/src/verifier.rs:340`) summed over every component, including the
global memory table (`claims.rs:1559-1570`). Naively cutting the execution at
step k makes shard B read an address written in shard A; B's local table has no
yield for it, so to self-balance the prover FABRICATES the value. Two locally
"valid" shards compose to an execution that never ran. The register state already
chains (`PublicData{initial_state, final_state}`, `air.rs:89-94,129-147`); **only
memory + the other global lookups lack a boundary.**

## 3. The enabling property: Cairo memory is WRITE-ONCE
Cairo's memory is a *nondeterministic but immutable* partial function
address→value: each address is assigned at most once over the whole run (enforced
by the memory argument; the adapter builds one dense `address_to_id` table,
`adapter/src/memory.rs:84,133`). This is the simplification that makes sharding
tractable — there is no versioning/ordering of writes to reconcile, only a single
global address→value relation that all shards must agree on. (Contrast a mutable
RAM machine, where sharding needs a full read/write-ordering permutation argument.)

## 4. Proposed architecture: shared committed memory + sharded execution + recursive merge

**4a. Factor memory into a shared commitment.** Commit ONCE to the full
address→value memory table M (the relocated run's memory) as a public input —
the same table the monolithic prover already builds, just hoisted to a shared
artifact with its own commitment `C_M`. Every shard receives `C_M` as public
input. The per-address LOGUP *yields* (today emitted by the memory components) are
proved ONCE against `C_M` (a "memory proof"), not per shard. Each shard's
opcode/builtin rows still emit their memory-access LOGUP *uses* — but now combined
against the SAME shared lookup elements and reconciled globally (§4c). Because
memory is write-once, `C_M` is a single immutable relation; no shard can disagree
about `M[x]` without breaking the commitment.

**4b. Shard the execution with register + memory carry.** Split the trace into N
contiguous step-ranges. Shard i proves: starting from `initial_state_i`
(pc/ap/fp), the VM transitions are valid, ending at `final_state_i`; and every
memory access it makes is a LOGUP *use* against `C_M`. Register chaining is the
existing mechanism (`final_state_i == initial_state_{i+1}`). The builtins
(pedersen/poseidon/range_check/ec_op/bitwise) also accumulate global lookups —
each such table is factored like memory: shared committed table + per-shard uses,
OR (for the ones whose multiplicities are cheap) a partial-sum carry (§4c).

**4c. Reconcile the global LOGUP balance via partial-sum carry.** Do NOT require
each shard's `lookup_sum == 0`. Instead each shard exposes its partial
`claimed_sum` (already a per-component public scalar) as a public output. The
recursive aggregator (§4d) sums the N shards' partial sums + the shared memory/
builtin-table proof's yields and checks the TOTAL `== 0` — exactly the global
balance, just additively decomposed. The lookup elements (Fiat-Shamir challenges)
must be shared across shards: drawn from a transcript seeded by `C_M` + all shard
trace commitments, so they are common (this is the one cross-shard Fiat-Shamir
coupling — handled by a two-phase protocol: commit all shard traces → derive
common elements → each shard finishes its interaction/FRI independently).

**4d. Recursive merge via the Cairo verifier.** An aggregation Cairo program
verifies the N shard proofs + the memory/table proofs (using the existing
single-proof `verify_cairo`, `cairo_verifier/src/lib.cairo:4`, run N+ times),
checks register chaining across shards, and checks the partial-sum reconciliation
`== 0`. Its output is itself proved → one aggregate proof (on-chain ready). Depth
can be a tree for large N.

## 5. Soundness obligations (MUST be discharged before code)
1. **Completeness of the boundary.** Prove that register-carry + `C_M` +
   per-table partial-sum carry capture EVERY cross-shard dependency — i.e. there
   is no global argument balanced by `lookup_sum==0` that lacks a carry. Enumerate
   ALL components contributing to `lookup_sum` (`claims.rs:1336-1644`): memory(×3),
   range checks, bitwise/xor tables, pedersen/poseidon point tables, the builtin
   aggregators, public_data. Each needs either a shared-commitment factoring or a
   partial-sum carry. A single missed global lookup = a soundness hole.
2. **Shared-challenge binding.** Prove the two-phase common-element derivation is
   sound (no shard can adaptively choose its trace after seeing the common
   elements — the commit-then-derive ordering must bind).
3. **Memory-commitment consistency.** Prove `C_M` + the write-once property forces
   a single consistent `M`, and that a shard cannot use an `(addr, value)` not in
   `M`. (The LOGUP use-vs-yield balance against the shared table should give this,
   but it must be proven, including the multiplicity accounting across shards.)
4. **Public-data / boundary segments.** The program code, output, builtin segment
   ranges (`air.rs:488-525`) are anchored to initial/final state — verify they
   compose correctly across shards.
5. **Aggregator soundness.** The recursion's own memory argument (the aggregation
   program is itself a Cairo execution) must not reintroduce the same hole.

## 6. Cost / scaling reality (vs the deck's "near-linear")
- Shard proving: ~linear in N (independent). 
- The shared memory/table proof is ~size of the global memory (not free; ~the
  memory-component cost of the monolithic prove, paid once).
- Reconciliation + recursive aggregation grows with N and with the cross-shard
  boundary (number of distinct addresses referenced across shard cuts). For a
  good cut (minimize cross-boundary addresses — cut at low-memory-traffic points)
  this is sublinear in trace size but non-trivial. **Realistic scaling: strongly
  sublinear-overhead near-linear for modest N (4–16), with the merge tail and the
  shared-memory proof as the Amdahl floor.** 20× needs N≈16–32 shards + the
  single-GPU witness offload — i.e. this is a cluster story, not a 2-GPU story.

## 7. Incremental build path (de-risked, value at each step)
1. **[stepping stone, NO soundness research — task #38] Data-parallel witness
   generation.** Spread ONE proof's host witness `write_trace` across GPUs (each
   device computes a slice of the trace columns), uploaded into ONE normal single
   prove. This attacks the dominant cost (the ~10s base trace) with multi-GPU and
   needs ZERO protocol change (the proof is bit-identical to single-GPU). Delivers
   the first real multi-GPU speedup safely while §5 is being worked. DO THIS FIRST.
2. **Shared-memory factoring (single proof).** Refactor the memory argument into a
   shared-committed-table form WITHOUT sharding yet (prove it's proof-equivalent).
   Establishes 4a in isolation, reviewable.
3. **2-shard PoC** with register+memory carry + a hand-built aggregator; differential
   vs the monolithic proof on a small program; full soundness review of §5.
4. **N-shard + tree aggregation + cluster distribution.**

## 8. Verdict
Sharding is the correct 20× architecture and Cairo's write-once memory makes it
*tractable* (unlike a mutable-RAM zkVM). But it is a soundness-critical protocol
with a real proof obligation (§5) — multi-quarter, expert-reviewed, not the deck's
8–12 weeks. **The pragmatic sequence for "20× ASAP" is: single-GPU full witness
offload (#36/#31/#30) for ~2–3× NOW, then data-parallel witness (#38, §7.1) for
the next multi-GPU multiplier with no protocol risk, and sharding (this doc) as
the funded research track for the final near-linear scaling.**
