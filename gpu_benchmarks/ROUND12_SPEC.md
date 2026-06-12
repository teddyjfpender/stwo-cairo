# Round 12 specification — opcode lane ports, device leaf recompute, ncu NTT, overlap v1

*Authoritative work order for the next optimization round. Process formalities
(every item): local SIMD gates -> push both repos + rev bump -> pod round with
per-minute watchdog -> STWO_CUDA_WITNESS_VERIFY differential where applicable ->
CUDA-vs-SIMD proof byte-equality (the decisive gate) -> bench A/B + nsys
before/after -> RESULTS.md entry with receipts archived to stwo-things/.
Soundness rules per CLAUDE.md: no Fiat-Shamir order changes, no constraint
edits; every device port must be value-identical with a per-component kill
switch and a differential. Pods: prebuilt image ghcr.io/teddyjfpender/stwo-pod
(consumer archs), templates tu2emjek43 (community) / 2rfb4wwdqp (secure);
screen hosts with pod/screen_host.sh; MPS for co-residency.*

---

## Item 1 — Opcode base-trace kernels on the generic witness lane (THE WHALE)

**Objective.** Move the per-step opcode witness writers (the fib loop body) to
device. Expected: the dominant share of the ~1.4 s/1M host time + its H2D
traffic. Target: 3090 1M single-proof 1.70 s -> ~1.1-1.3 s this round.

**Step 0 — rank by measured mass.** Dump fib's per-opcode counts (gpu_bench
already computes `casm_states_by_opcode.counts()`; print it under
STWO_BENCH_TRACE or a one-line eprintln) and each component's N_TRACE_COLUMNS.
Rank by rows x columns. Expect jnz_opcode_taken / add_ap_opcode /
add_opcode_small / jump variants to dominate. Port the top 1 as the pilot
(main session), then delegate the rest per-component to subagents — each
subagent gets: the generated file, the recipe below, and MUST hand back a
differential-green diff. Never let a subagent touch the lane primitives.

**Per-component recipe (proven on verify_instruction; generalize):**
1. Read `witness/components/<op>.rs`: inputs (PackedCasmState rows: pc/ap/fp),
   the row math in write_trace_simd, SubComponentInputs (which states it
   feeds), LookupData tuples, the interaction writer's column shapes (opcodes
   use Enabler(n_rows) multiplicities — `Mult::Enabler` is already in the lane).
2. `into_parts(self)` on its ClaimGenerator: the padded input vectors +
   n_rows (opcode padding repeats inputs[0]; Enabler handles mult semantics —
   VERIFY the exact padding per component against the host writer).
3. Base-trace kernel in stwo `witness_logup.cu` (or a new opcodes.cu): the row
   math transcribed u32-for-u32. RULES: PackedUInt16/32 ops -> plain u32 ops;
   every PackedM31 add/sub/mul -> the fields.cuh modular ops (NEVER raw + on
   values only bounded by P); stage every lookup-tuple expression that is not
   a plain trace column as an extra staged column; trailing constant-zero
   tuple terms may be truncated (exact field identity).
4. Memory deduce on device: the opcode reads `addr_to_id.deduce_output(addr)`
   and `id_to_big.deduce_output(id)` per row. The tables are ALREADY device-
   resident: addr->id = the addr witness's flat chunk-major ids buffer
   (index addr-1 with chunk layout: chunk=idx/size, see
   memory_witness_backend.rs slice math); id->limbs = P1's per-segment limb
   columns (decode id: big = sequential index -> segment idx>>MAX_SEQ_LOG,
   row = idx & mask; small via LARGE_MEMORY_VALUE_ID_BASE branch). Add two
   gather kernels to the lane: `addr_to_id_gather` and `id_to_limbs_gather`.
   These require the memory components' device state to be THREADED to the
   opcode write (the claim-generator stanza already holds the states — extend
   DeviceMemoryWitness/DeviceAddrWitness with accessor methods; opcode
   write_trace happens in the scope BEFORE memory write_trace, so the device
   tables must be built EARLIER: move the memory table upload into
   ClaimGenerator::new-time or a lazy device-cache on the states. DESIGN
   DECISION REQUIRED — simplest correct v1: upload the raw tables (addr ids,
   f252 values) once lazily on first opcode use, independent of the memory
   component's own segmented buffers).
5. Sub-component feeds: rc families via `tuple_count` (+ add lut/merge methods
   to each rc state with `dense_input_to_row_lut` — width-1 tables use slot
   bits [k]); memory mults via a new `index_count` kernel (counts[idx] += 1,
   no LUT); verify_instruction feed STAYS HOST v1 (its dedup dashmap is data-
   dependent; feed from the host input arrays exactly like the host writer's
   sub_component_inputs loop — measure its cost; optimize in a later round).
6. Interaction via the lane: tuple_pair/tuple_single with Enabler mults.
7. Trait + wiring: follow VerifyInstructionWitness exactly (trait per
   component or generalize to one OpcodeWitness trait keyed by component —
   prefer ONE trait with an enum/impl per opcode to stop the bound explosion
   in prover.rs; refactor the three memory-era traits into it opportunistically).
8. Gates: per-component differential (trace cols, every count delta,
   interaction + sums), kill switch `STWO_CUDA_<OP>_WITNESS=0`, byte-equality.

**Risks.** Padding semantics per opcode differ (REVIEW each against the host);
the deduce gathers are the new soundness surface — differential catches value
errors, but layout-index bugs can alias silently within a table: include an
index-bounds debug assert kernel under the verify env.

---

## Item 2 — Device leaf-hash recompute (kills the 61.7k D2H)

**Objective.** The pruned-tree decommit recomputes queried/witness leaf hashes
by reading EVERY (column, leaf) element through `raw_value` -> one sync 4-byte
D2H each (61,726 per 2 proves, measured). Recompute the needed leaf hashes ON
DEVICE in one launch.

**Where.** stwo `prover/vcs_lifted/prover.rs`: `decommit_inner` recompute loop
+ `node_hash` (line ~441); CUDA `blake2s.cu`.

**Design (preferred).** Collect the needed prev-level leaf indices host-side
(they are known before hashing: queried pairs + witness siblings, layer
leaf_log_size only — the higher layers hash from node_memo, not columns). Add
`commit_on_first_layer_lifted_indexed(indices*, n, data**, col_log_sizes*,
lifting_log_size, out_hashes*)` — the word-native kernel body with the index
taken from the list. One launch + one D2H of n hashes -> seed node_memo.
Plumb via a `ColumnAccess::leaf_hashes_at(indices) -> Option<Vec<H::Hash>>`
hook: default None (host path unchanged — CPU/SIMD and GatheredColumns),
DenseColumns<CudaBackend> returns Some. NOTE the hook is hash-generic; gate
the device path to Blake2s towers only (match on H type or specialize at the
backend boundary).

**Gates.** decommitment.hash_witness + aux.all_node_values byte-identical
(covered by proof byte-equality); add a unit differential in the conformance
suite (device leaf hashes vs CPU node_hash on random columns).

**Expected.** ~75 ms serialized D2H + host leaf hashing per 2 proves removed;
decommit phase mostly disappears from the host timeline.

---

## Item 3 — ncu NTT round (measured iteration, secure cloud)

**Objective.** The n2b/b2n family (~32% of kernel time) is NitrooZK-tuned;
find what the profiler says before touching anything.

**Process.** (1) Secure-cloud pod (H100 or secure 4090); verify counters:
`ncu --query-metrics` then a 10-launch capture — if ERR_NVGPUCTRPERM persists
on secure cloud, stop and document (provider driver policy; alternatives:
Lambda/own box). (2) Capture `--set full` on the five hot variants
(n2b_nofinal<4>/<3>, n2b_final_warp<2>/<3>, b2n_init_warp<3>) at 1M and 2M.
(3) Read: achieved occupancy, smem bank conflicts, DRAM %-of-roofline, warp
stall reasons. (4) Change-measure loop, ONE knob at a time, conformance +
duration diff per change: block dims / LOG_VALS_PER_THREAD thresholds,
twiddle loads via __ldg/const path, vectorized (uint4) global loads, skipping
the smem round-trip for small log sizes. (5) Byte-equality after the final
state. Budget: 2-3 pod hours. Success: >=1.2x on the family or a documented
"at roofline" verdict (which would itself justify P4 hardware reasoning).

---

## Item 4 — Overlap v1: per-component in-scope logup finalize

**Objective.** The honest intra-proof overlap surface is narrow (Fiat-Shamir:
interaction depends on the base root). The legal, measurable win: the
post-scope SERIAL `B::finalize_raw_logup` loop over ~60 components currently
runs after ALL host interaction writes; move each component's finalize INTO
its rayon completion (exactly what the memory components already do) so GPU
finalize work overlaps the host's remaining write loops.

**Where.** stwo-cairo `cairo_claim_generator.rs::write_interaction_trace`:
each scope spawn currently stores `(raw, build_claim)`; change to store
`(finalized_trace, claimed_sum, build_claim)` by calling B::finalize inside
the spawn. Eval extension and claim construction STAY post-scope in fixed
order (Fiat-Shamir untouched). Precedent: the memory/vi hooks already
finalize in-scope — this generalizes that to every component stanza
(mechanical sed-able edit across ~60 stanzas + the post-scope loop).

**Caveat.** Finalize launches CUDA from many rayon threads concurrently —
the bridge-mutex rev is validated for this (12x clean H100), and the legacy
stream serializes the math; the win is host/GPU overlap, not GPU/GPU.
Streams stay off by default; revisit pool streams only after this lands and
nsys shows remaining gaps.

**Gates.** Byte-equality (the finalize math is order-independent; eval order
unchanged); repeated-prove x3.

**Expected.** The interaction phase's GPU tail hides under host writes:
~5-10% of wall at 1M, more at 2M.

---

## Order of execution

1. Item 4 (smallest, immediate, de-risks concurrent finalize before Item 1
   adds more in-scope device work).
2. Item 2 (contained, kills a measured cost).
3. Item 1 pilot opcode -> then parallel delegated ports, one validation pod
   round per batch of 2-3 components.
4. Item 3 on a secure pod whenever one is up for other reasons (piggyback).

Re-measure after each: 1M/2M single, MPS-dual aggregate, and the fleet quote
($/hr for 10 MHz). Update ROAD_TO_10MHZ arithmetic with each landing.
