# The Next GPU Optimizations — from "GPU-assisted" to a GPU-resident prover

Companion of `ENDGAME_ARCHITECTURE.md` (the bandwidth-floor argument and §-numbered
workstreams) and `DEDUCE_DESIGN.md` (the computed-deduce lane). This document does
three things: (1) states the AS-BUILT architecture precisely — where every phase
runs today and what crosses the PCIe/host boundary; (2) itemizes the taxes that
keep measured utilization far below the hardware floor; (3) specifies the target
architecture for maximum parallel proving speed — near-zero CPU participation —
and ranks the concrete moves that get there.

Grounding: every "measured" number below is from the results ledger
(H100 NVL sm_90 unless noted). SN_PIE_2 = 7.68M bootloader-counted steps.
Best warm prove today: 13.5s = 0.571 useful MHz, after the round-17 composition
NVRTC fix (+20%, the only non-flat optimization so far — it moved CPU work to
the GPU, which is the recurring lesson of the whole ledger).

---

## 1. Why utilization, not kernels, is the problem

The duty-cycle samples say it plainly: pre-round-17, **96% of 1s dmon samples
showed <5% SM utilization** during a full SN_PIE_2 prove. The GPU saturates in
bursts and starves between them. Meanwhile the bandwidth-floor estimate in
ENDGAME §1 (total trace traffic ÷ HBM bandwidth) puts the *physics* ceiling in
the **tens of MHz per H100-class card** and single-digit MHz per consumer card.
We are ~1-2 orders of magnitude below the floor, and the gap is composed of:

1. **Host-resident compute inside the prove** (the fp256/EC witness family —
   being deleted by the D′ deduce lane right now; previously also the EC
   composition kernels, fixed round-17).
2. **Host round-trips inside device phases** (sub-word D2H → host rebuild →
   host feeds → re-upload; igen rebuilds; staging copies for host-born columns).
3. **Serialization on one stream + launch gaps** (thousands of small launches
   with ~5-10µs host-side gaps each, one legacy stream for almost everything).
4. **Fiat-Shamir barriers** (host channel absorbs a 32B root, draws elements —
   microseconds of compute that drain the entire device pipeline each phase).
5. **Fixed per-proof overheads** (adapter ingest, table uploads, proof
   serialization) that only amortize with bigger PIEs or proof pipelining.

Items 1-2 dominate today (they are why dmon shows idle); 3-4 become dominant
exactly when 1-2 are gone. 5 is a systems-level lever independent of the rest.

---

## 2. The as-built architecture (what actually runs where)

A CudaBackend `prove_cairo` on an SN PIE, phase by phase. **[H]** = host CPU,
**[D]** = device, **[H→D]/[D→H]** = transfer.

### 2.0 Ingest (per PIE, once)
- **[H]** cairo-vm re-execution / PIE adaptation → `ProverInput` (dedup'd memory
  tables, casm states by opcode, builtin segments).
- **[H→D]** `DeviceExecutionTables::upload`: `addr_to_id` (raw u32), f252 value
  words, small values; device limb-split kernels expand them into 28+8 limb
  columns (once per memory, process-cached).
- **[H→D]** pedersen points table: GPU-generated on demand
  (`initialize_pedersen_table`, owned mode — ~1.9GB at W18 padding).

### 2.1 Witness ("Write Base trace") — the former wall
- Components run under a **rayon parallel scope** (host threads), each choosing
  its lane:
  - **Opcode family (14 components) [D]**: witness-JIT lane. The transformer
    recording is compiled by NVRTC into ONE kernel per component (one thread per
    row; reads the execution tables for memory deduces; writes committed columns
    device-resident; emits word-major lookup/sub flats; atomically bumps device
    mult tables). Stage B′ fans these launches across **4 pool streams**
    (fork/join event bridges around the legacy stream).
  - **Builtins (pedersen_aggregator, blake_round) [D as of this round]**: the
    same lane via slot-layout input columns (`[flat words | enabler | iota |
    mults]`) and computed deduces — blake g/sigma inline device fns; the W18 EC
    round on the fp256 chain (`stwo_wit_deduce.cuh`), with per-CUmodule pedersen
    table globals filled at module load.
  - **Still [H]**: `partial_ec_mul_window_bits_18` + `partial_ec_mul_generic`
    (~1.7s of the warm ledger — step-4 target: ~29 felt binops away), plus the
    long tail of small writers (range checks, verify_instruction — <150ms
    total, measured).
- **The lane's hidden tax [D→H→D]**: every device-laned component still D2Hs
  its **sub-input words** (aggregator: 2,023 words/row!), rebuilds packed
  vectors on host, and **feeds** downstream claim generators (DashMap /
  count-table `add_inputs`) — because downstream components build *their*
  inputs from those host structures. Lookup words either D2H + host igen
  rebuild, or (opcode family, §6a ON) stay device-resident.
- **A″ pipelined commit [D‖H]**: a committer thread starts iNTT on the opcode
  column prefix while the serial host-heavy remainder still generates.

### 2.2 Base commit
- **[D]** per-column iNTT + LDE (`rfft.cu`, column-batched with 65,535-column
  grid chunks), bit-reverse, Merkle tree via `blake2s.cu` (first-layer +
  layer-from-previous kernels; a fused two-layers-per-launch variant exists).
- **[D→H]** the 32B root → **[H]** Fiat-Shamir absorb (stwo core
  `Blake2sChannel`).
- Twiddles: computed on demand, process-cached per (backend, log_size) —
  warm reps reuse.

### 2.3 Interaction (logup)
- **[H]** channel draws `CommonLookupElements`.
- **[D]** §6a: for stash-enabled components, `logup_pairs.cu` builds all logup
  columns from the device-resident lookup words in ONE launch per component +
  the proven device `finalize_raw_logup` (measured: 1M rows × 5 cols in 2ms vs
  ~300ms host). **[H]** for everything else: SIMD `write_interaction_trace`,
  then staged H2D upload of the interaction columns.
- Builtins currently take the host-flats igen path (§6a not yet extended).
- Then the same commit path as 2.2 (iNTT/LDE/Merkle/root/absorb).

### 2.4 Composition
- **[H]** OODS point / accumulation coefficients from channel.
- **[D]** constraint-JIT: NVRTC-compiled kernels per component, split into
  ≤512-instruction pieces at -O3 (the sm_90 ptxas cliff; the 2048-instr fused
  variant loads in 240-290s per module via driver JIT — measured, hence 512),
  accumulating into the composition polynomial in place (single stream — the
  split kernels share coordinate columns; fan-out would lose updates).
  ~90+ kernels for the big EC components alone. Disk PTX/cubin cache
  (`STWO_JIT_CUBIN_CACHE`) makes rep-2+ and fresh processes cheap.
- Commit as above.

### 2.5 FRI + PoW + decommit
- **[D]** `fold_circle_into_line` / fold chain per layer; per-layer Merkle
  commit; **[H]** channel absorb per layer (a barrier per fold layer).
- **[D]** PoW grind (`grind_blake2s` — device, byte-exact search order).
- **[H]** query positions from channel; **[D→H]** batched Merkle node gathers
  (`cuda_multi_layer_batch_get_blake_2s_hash`, one gather for thousands of 32B
  reads — A′ item) + FRI leaf gathers; **[H]** opening hashing + proof assembly
  + canonical serialization (BTreeMap ordering — Stage 0).

### 2.6 Orchestration substrate
- **One legacy stream** carries everything except the B′ witness fan-out (4
  pool streams) and the A″ committer thread.
- **Launch storm**: composition (hundreds), NTT stages × trees, Merkle layers
  (~2× tree height per tree thanks to the fused variant), FRI folds — each
  launch enqueued from host between host-side bookkeeping.
- **Memory**: never-release device pool (arena); pinned staging slab (Mutex'd)
  for H2D of host-born columns; committed-LDE retention ≈ 47.6GB peak on
  SN_PIE_2 (the VRAM-diet axis for consumer cards).

---

## 3. The taxes, itemized

| # | Tax | Mechanism | Evidence | Deleted by |
|---|-----|-----------|----------|------------|
| T1 | Host witness writers | fp256/EC family runs on CPU while GPU idles | 96% idle samples; spans: w18 1.09s + generic 0.60s (+ agg 0.79 + blake 0.93 pre-this-round) | D′ lane (§4.1) |
| T2 | Sub-word round-trip | D2H flats → host packed rebuild → DashMap feeds → downstream re-derives | aggregator = 2,023 words/row; opcode lane precedent: kernel 340-400ms but wt: span 980ms | device component DAG (§4.2) |
| T3 | Host igen rebuilds | lookup flats D2H + host repack for non-§6a components (all builtins today) | §6a measured 2ms vs 300ms host per family | §6a everywhere (§4.2) |
| T4 | Interaction upload | host-born interaction/base columns staged H2D | ~2,500 staging copies pre-lane; shrinking as lanes land | full residency (§4.2) |
| T5 | Launch gaps + one stream | ~5-10µs host gap per launch × thousands; phases serialized on legacy | flat A′/A″/B′ = the gaps aren't the wall *yet*; they are once T1-T3 die | CUDA graphs (§4.3) |
| T6 | ptxas 512-instr split | 4× the composition launches to dodge driver-JIT cliff | 2048-instr module_load 240-290s driver-JIT; relax(-O0) 2.6× WORSE | offline cubins at 2048/-O3 (§4.4) |
| T7 | Fiat-Shamir drains | host channel absorb/draw between every phase + per FRI layer | inherent to STARK ordering | overlap + device channel (§4.5) |
| T8 | Fixed per-proof cost | adapter, uploads, serialization, cold JIT | 14M-step PIEs already amortize: SN_PIE_4 0.480 cold vs SN_PIE_2 0.42 cold | multi-proof pipelining (§4.6) |

---

## 4. The target architecture — a GPU-resident prover

The organizing principle: **the proof transcript is the only thing the host
should compute; device memory is the only place witness data should live.**
Host's steady-state role = issue graphs, absorb 32-byte roots, serialize the
final proof.

### 4.1 Finish witness residency (in flight — the current round + step 4)
Every component's base trace born on device via the automated lane. Remaining:
the two `partial_ec_mul`s (felt add/sub/mul/div as DeduceKinds 4-7 — the census
says ~29 binops + W27 regroup + u32 idioms), then the small-writer tail only if
the ledger says it matters (<150ms today). **Expected effect**: deletes T1
(~1.7s remaining), duty cycle structurally changes because the serial host
block between rayon scope and commit disappears.

### 4.2 The device component DAG — kill the feed round-trip (the big one)
Today's component graph is: kernel → D2H flats → host rebuild → host DashMap →
downstream write_trace re-reads → (re)upload. The target:

- **Sub-feeds become device edges.** A component's sub-input words stay in
  device buffers; downstream *mults* accumulate by device atomics into count
  tables (the witness kernel ABI already has `mult_counts` atomics — extend the
  same mechanism to range-check/table components); downstream *input lists*
  (w18's 72-word instances, blake_g's 6-word instances) are consumed directly
  as the downstream kernel's input columns — device-to-device feeding, the
  `blake_round → blake_g` seam generalized. Chained deduces (the aggregator's
  28 EC rounds) already avoid this via computed deduces IN-kernel — the DAG
  edge pattern is only needed for flat producer→consumer shapes.
- **§6a for every component** (builtins included): lookup words never leave the
  device; interaction = one pair-kernel + shared finalize per component.
- **Ordering**: dependency-sorted component schedule on N streams (the sort
  exists implicitly in cairo_claim_generator's spawn order; make it explicit),
  events only at true data edges. The rayon host threads stop being compute
  and become launch issuers.

**Expected effect**: deletes T2/T3/T4. The witness phase becomes a pure kernel
DAG whose wall-clock is max-path kernel time — on H100 measured kernel times,
hundreds of ms, not seconds. This is the single largest remaining structural
win after T1.

### 4.3 Graph-per-phase execution (Stage C, now justified)
Once phases are kernel DAGs with no host compute inside, per-launch host gaps
are the residual. Capture each phase as a **CUDA graph** keyed by the size
vector: witness DAG graph, commit graph per tree (NTT stages + Merkle chain),
composition graph, FRI fold graph. Replay per rep/per proof. Prerequisites
already staged: launches parameterized by stream (done for the witness JIT;
thread through the remaining ~82 externs), full precompile before capture
(cubin cache), arena slots addressed by column identity (pool addresses are
completion-order nondeterministic), capture-origin event bridges (the fresh-
event fork/join must not record on legacy during capture). **Expected
effect**: deletes T5 — thousands of launches become ~5 graph launches per
proof; phase gaps shrink to event waits.

### 4.4 Commit-path fusion (§3, re-scoped by data)
- **Offline cubins for 2048-instr fused composition kernels at -O3** (T6): the
  cubin cache already ships; raising `STWO_JIT_MAX_KERNEL_INSTRS` back to 2048
  with cached SASS gives 4× fewer composition launches WITHOUT the -O0 SASS
  regression (force-relax falsified at 2.6× worse) and without the driver-JIT
  cliff (no ptxas at load).
- **Hash-from-registers Merkle**: fuse leaf hashing into the last NTT/LDE
  stage output (blake2s state assembled in registers/shared before ever
  hitting HBM) and extend the existing two-layers-per-launch variant with a
  **single-block tail kernel** for the top ~12 levels (one launch replaces ~10
  tiny ones). The base-tree commit measured 75× off the streaming bound on
  A40 — this is a bandwidth-shaping problem, not a compute one.
- **sm_90 specifics** (H100 dev iron): TMA/cp.async bulk staging for the NTT
  tiles, L2 persistence window for twiddles + pedersen/limb tables (they are
  re-read by every kernel in the phase).

### 4.5 Barrier discipline (T7)
Fiat-Shamir order is fixed — but the drain need not idle the GPU:
- **Overlap independent trees**: interaction columns of tree k can NTT while
  tree k-1's Merkle top levels finish (they only serialize at the channel
  absorb).
- **Device-side channel option**: blake2s absorb/squeeze as a 1-block kernel so
  root→elements never leaves device; host mirrors the transcript for proof
  assembly. Grind already runs device-side; this closes the remaining 32B
  ping-pong per phase and (with graphs) lets consecutive phases chain on-device
  via conditional/dependent launches. Verifier compatibility is untouched — the
  transcript bytes are identical; only WHERE they're computed changes. This is
  soundness-adjacent (channel code), so it lands behind byte-equality of the
  full transcript and prover-team review, per CLAUDE.md.
- **FRI folds**: per-layer absorb makes the fold chain barrier-heavy; with the
  device channel + graphs the whole FRI phase becomes one graph.

### 4.6 Multi-proof pipelining (P5 — the utilization backstop)
Even a perfect single-proof pipeline has drains (ingest, serialization, channel
points). Two proofs in flight (independent transcripts, shared caches — cubins,
twiddles, points table) let proof N+1's witness DAG fill proof N's barrier
gaps. The producer/consumer harness already measured sustained ≈ single-proof
MHz with 3 producers and zero feed starvation — post-residency, expect
sustained > single-proof because the gaps get filled. On the fleet target
(24GB cards), this composes with the VRAM diet: per-consumer LDE regeneration
trades compute for memory, and the pipeline hides the regen.

### 4.7 What stays on the host, permanently
- PIE parse + cairo-vm adaptation (per-proof, overlappable with the previous
  proof's GPU phases).
- Transcript bookkeeping (µs) unless/until 4.5's device channel lands.
- Proof assembly + serialization (~100ms class, overlappable).
- Nothing else. "Zero CPU" operationally = no host compute inside any phase's
  critical path.

---

## 5. Ranked plan (mechanism → expected effect on SN_PIE_2 warm)

Baseline today: 13.5s / 0.571 useful MHz. Bands are honest engineering
estimates, not promises; each lands behind byte-identity gates.

1. **D′ builtins live** (this pod session): agg+blake spans 1.72s → lane cost
   0.4-0.8s ⇒ **~12.3-12.8s / ~0.60-0.63**.
2. **Step 4-5: felt DeduceKinds → both partial_ec_muls** (~29 binops + W27):
   deletes the last ~1.7s host block ⇒ **~10.5-11s / ~0.70-0.73**.
3. **§4.2 device DAG (feeds + §6a-for-builtins)**: deletes the D2H/rebuild/feed
   tax the lane still pays ⇒ **~9-10s / ~0.77-0.85**.
4. **§4.4 offline-cubin 2048 fusion + Merkle tail/fusion**: commit 5.5s →
   ~3.5-4s ⇒ **~7.5-8.5s / ~0.9-1.0**.
5. **§4.3 graphs per phase**: launch-gap floor removed; witness+FRI storms →
   graph replays ⇒ **~6-7s / ~1.1-1.3**.
6. **§4.5 barrier overlap (+ device channel if approved)** ⇒ **~5-6s /
   ~1.3-1.5** single-proof.
7. **§4.6 two proofs in flight** ⇒ sustained ≥ single-proof; fleet math takes
   over (8 × consumer cards ≥ 10 MHz aggregate — the north star — with
   single-card continuing toward the bandwidth floor).

The floor argument still holds: after 1-6 the prover is bandwidth-shaped, and
the remaining distance to tens-of-MHz is measured (nsys, now meaningful) rather
than asserted — candidates at that point are pure kernel engineering (NTT
radix/fusion, hash throughput) on a saturated device.

## 6. Falsifiable predictions (so the next ledger entry can grade this doc)

- P1: after step 2, `Write Base trace` span < 1.5s and dmon idle-sample share
  drops below 60% for the first time.
- P2: after step 3, wt: spans for laned builtins are within 2× of their raw
  kernel times (the D2H/feed majority is gone).
- P3: after step 4, Commitment span scales ≥2× better than linear when moving
  H100→A40 (bandwidth-shaped, not launch-shaped).
- P4: graphs (step 5) are FLAT if attempted before steps 2-3 (the A′/A″/B′
  lesson generalizes: don't compress gaps between work the CPU is creating).

---

## 7. Implementation status (living section — updated per increment)

Updated 2026-07-05. Every landed item is committed, byte-identity-gated, and
stub-safe on macOS; nothing below has pod validation yet (the single validating
pod session runs after §4/§5's implementable set is complete, per the program
directive).

### Landed

**§4.1 witness residency — COMPLETE (code + local gates)**
- Borrowed pedersen table (the oracle falsified the GPU-generated table —
  144/256 sampled rows vs host, run 20260705T113615Z; the host table is now the
  only permitted source, uploaded column-streamed in borrowed mode; generation
  quarantined, fill paths fail closed).
- Felt DeduceKinds 4-7 (device = `ec_add_affine`'s proven Montgomery operand
  pattern; the host `Felt252` compensation algebra verified canonical-value
  in/out before transcription), full u32 trait lane, W27 lane (SIMD = the
  production conversion pair; recording = the exact 27→9 regroup).
- The ENTIRE fp256/EC + poseidon-fp256 family through the automated lane with
  ZERO poisons: pedersen_aggregator (5,005 instrs), blake_round (2,178),
  partial_ec_mul_w18 (7,917), partial_ec_mul_generic (17,000), cube_252,
  range_check_252_width_27; u32-cohort unlock: mul_opcode, add_ap_opcode,
  blake_g. All byte-identical + interpreter-identical on real fixtures;
  pod-gated device legs wired into the interp gates.

**§4.2 device DAG — count-feed core COMPLETE**
- Transformer-emitted `JIT_LOOKUP_FIELDS` / igen accessors / `SUB_FEED_LAYOUT`
  for all 27 lane components (hand accessor data entry deleted).
- `witness_feed_counts.cu` (descriptor-driven generalization of the certified
  blake_g count feed) + FFI; launches return the DEVICE sub buffer.
- Consumer surfaces (`add_count_tables` + `input_to_row_lut` where tuple-keyed)
  for all ten count families; verified `COUNT_RELATIONS` registry.
- **The count gate**: on the real w18 fixture, the device-feed path (emitted
  layout → descriptors → fold/LUT → merge) is byte-identical to the consumers'
  own `add_input` feeds — 127 descriptors, first run, zero hardware.
- Prove-path split: `DeviceFeedPlan` in the builtin lane — count relations
  merge from device counts, input-list relations stay host, fail-closed both
  ways. Live seams: aggregator (rc_8), blake_round (rc_7_2_5).

### Landed (continued — the C/D/E increments)

- **B3 device edges**: aggregator→w18 (transactional pair: producer stashes
  device sub buffer + HOST flat mirror, skips its w18 feed; consumer gathers
  72 device columns and launches from device pointers; any failure rebuilds on
  CPU from the stashed flat — exactly-once in every combination) and
  blake_round→blake_g (row-major interleave straight into the CERTIFIED hand
  kernel's ABI). Both edge gates green (pure-Rust kernel mirrors vs the host
  feeds, real fixtures).
- **C1**: resolved as configuration + gate — the code's governor default is
  already 2048; the 512 was the sm_90 driver-JIT override, which the cubin
  path (SASS at compile time) bypasses. `sn2-cubin-2048` A/Bs it with byte
  identity.
- **C2**: `MerkleOpsLifted::build_top_layers` (defaulted — CPU/SIMD unchanged;
  vcs_lifted 15/15) + the fused-tail handoff (TAIL_LEVELS=12, retained-only) +
  the CUDA override running one `stwo_blake2s_tail` launch (same hash routine
  — scheduling only; flagged for prover-team review per the
  security-critical-file precedent).
- **E2**: the P5 pipeline harness (`--pipeline/--producers`) predates this
  program; the manifest's `sn2-sustained-dag` measures sustained throughput
  with the DAG lanes on.

### Deferred WITH REASONS (pod-gated by the program's own evidence)

- **Hash-from-registers leaf fusion (C)**: the deepest blind-CUDA surgery of
  the program (rfft output stage + blake2s state assembly), soundness-adjacent,
  and a bandwidth-shaping win that only matters after the launch floor drops.
  Revisit with nsys after the pod session.
- **Phase-graph capture (D2) + inter-tree overlap (E1)**: gap-compression
  class. A′, A″, and B′ — the same class — each measured FLAT because CPU work
  dominated the gaps; the landed A/B phases delete that CPU work, so these may
  finally pay — but WHICH of them pays is exactly what the post-DAG duty-cycle
  and phase ledger measure. Building them blind before that measurement
  inverts the program's evidence discipline (P4 codifies this). The mechanical
  enabler that is NOT deferred: the witness path is already
  stream-parameterized (the graphs prerequisite for the phase that matters
  most post-DAG).
- **Full stream-ABI sweep (D1)**: dead plumbing until D2/E1 consume it; lands
  with them.
- **chains→cube_252 edge (B3)**: the poseidon chains are HOST writers today
  (no device sub buffer exists to edge from); the edge waits for their own
  lane cycle, and their measured span share is small.
- **Overlapped concurrent proves (E2+)**: two prover threads sharing the
  device (pool/caches are process-global but never validated under concurrent
  proves) — pod-gated experimentation after the sustained numbers land.

### Outstanding (execution order)

1. **B2 tail**: count gates for generic/cube_252/aggregator/blake (clone the
   w18 gate); prove seams for w18/generic/cube_252; rc_252_width_27 registry
   entry (consumer shape differs — verify first); mem-table + sigma count
   families.
2. **B3 device edges**: producer sub buffer → consumer input columns for the
   input-list feeds (aggregator→w18 28×72-word instances, blake_round→blake_g,
   chains→cube_252) via a gather kernel; deletes the last host feed volume.
3. **C commit fusion**: 2048-instr composition via the cubin path (no
   load-time ptxas, -O3 kept); single-block Merkle tail kernel;
   hash-from-registers leaf fusion; sm_90 cp.async/L2-persistence.
4. **D phase graphs**: stream param through the remaining externs;
   capture-safe event bridges; arena slots by column identity; per-phase
   capture (witness DAG / commit / FRI).
5. **E**: inter-tree overlap across channel absorbs; two-proof pipelining
   (device Fiat-Shamir channel EXCLUDED pending human approval — barrier
   overlap does not touch channel code).
6. **The single pod session**: extended `deduce_gate.toml` (oracle legs →
   whole-kernel gates → count-feed gate → lane-ON proofs with engagement →
   whole-proof byte identity → perf) validating the whole program. Expectation
   per §5's ranked trajectory (measured, not asserted): the landed steps 1-3
   band lands around ~9-10s / ~0.8 useful MHz on SN_PIE_2 warm, with C/D/E
   carrying the band toward ~6s / ~1.2+ before the bandwidth floor becomes the
   measured frontier.

Toolchain note: the prover lib-test crate requires `RUST_MIN_STACK=16777216`
to COMPILE (rustc SIGBUS below; the emitted components keep growing).

### ROUND-28 pod session results (2026-07-05, H100 NVL, SN_PIE_2 — measured)

**Validated on hardware, first proofs of the full program:**
- Oracle legs (deduce kinds 2+3): PASS — the fp256/EC device functions and the
  borrowed host pedersen table are byte-exact on hardware.
- Whole-kernel device gates: blake_round PASS; pedersen_aggregator PASS after
  the NVRTC wall (below) — every committed column, lookup word and sub word
  byte-identical to the host writer.
- Lane-off baseline prove: PASS after the C2 tail fix (below) — the fused
  Merkle tail is live in the default commit path. Warm 17.49 s / 0.441 useful
  MHz (NOTE: this manifest measures the deduce lane in ISOLATION — no
  ASYNC_SPINE / PIPELINED_COMMIT — so it is not comparable to the composed
  15.53 s round-13 number).
- Lane-on prove (aggregator+blake, cap 6000): engaged, PASS, and
  **whole-proof byte-identity vs lane-off: PASS** — the program's core
  soundness gate on a real Starknet OS PIE proof.
- Lane-on perf (same partial config): warm 17.86 s — the two lanes alone do
  not buy wall time yet; the blake device witness costs ~2.0 s where the host
  span was 0.93 s (input-column upload + JIT witness at log 20). The win
  thesis rests on the DAG (feeds+edges deleting host work), not on lane
  substitution alone — consistent with §5's ranking, unproven until the
  full-DAG perf numbers land.
- DAG prove: blake + aggregator + w18 + cube_252 all engaged with device
  count feeds (1/1/3/2 relation families) and both edge stashes; proof
  completed. partial_ec_mul_generic silently fell back — root-caused and
  fixed (below); its re-validation did not run (session cut short).

**Three hardware-only walls found and fixed (none reachable by local gates):**
1. NVRTC JIT-mode compatibility: the fp256 embed chain was offline-nvcc-only —
   24 errors (pure-__host__ fns, one-arg static_asserts under --std=c++14,
   UINT32_MAX/curandState, a compound literal). Fixed offline-invariant
   (!__CUDACC_RTC__ guards + RTC-only prelude macros); probe-validated via a
   30-line libnvrtc harness before any Rust rebuild. stwo 90591283.
2. C2 tail kernel launch: 1024-thread block exceeds the SM register file with
   the inlined blake2s → launch failed in the BASELINE prove. 256-thread
   bounded block (scheduling-only). stwo 6cbb4da2.
3. NEEDS_PEDERSEN_TABLE semantics: the generic lane runs BEFORE the aggregator;
   its felt-deduce kernel embeds fp256 (module declares the table globals) and
   the fail-closed load fill rejected it — silent fallback + ~30 s wasted NVRTC.
   The flag now tracks the EMBED, not table READS (generic + cube_252 = true),
   and the shape pin asserts flag == embeds-fp256 for all five lanes.
   cairo e7211293.

**Session cut short: RunPod balance exhausted mid-run** (pod terminated by the
provider; volumeless /workspace lost — next session pays full re-sync/build).
Remaining pod items, in order: re-run deduce_gate.toml end-to-end (generic
engagement + DAG byte-identity + cubin-2048 A/B + sustained-DAG), then ONE
composed-config run (ASYNC_SPINE + PIPELINED_COMMIT + STREAM_FANOUT + full DAG
+ 2048 cubin, identity-gated) for the honest current-best number.
