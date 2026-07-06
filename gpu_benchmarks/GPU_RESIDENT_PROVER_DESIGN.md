# The GPU-Native Stwo Prover — Total Overhaul: Architecture & Systems Design

*2026-07-05 (v3 — correctness/residency review pass; v2 added the overhaul
framing per user decisions). This is the
governing systems-design document for the TOTAL OVERHAUL of Cairo proving on
CUDA: a new GPU-native pipeline that owns orchestration end-to-end, replacing
the GPU-assisted host prover. It supersedes `NEXT_GPU_OPTIMIZATIONS.md` §4
(the incremental target) where they conflict; the bandwidth-floor argument of
`ENDGAME_ARCHITECTURE.md` §1 is incorporated unchanged. Every "measured"
number is from the results ledger (H100 NVL sm_90 unless noted; SN_PIE_2 =
7.68M bootloader-counted steps). Baseline: best composed warm prove 13.5s /
0.571 useful MHz; the DAG lanes are code-complete and byte-identity-gated but
not yet perf-validated (round-28 session cut by pod billing).*

---

## 0. Program decisions (user-ratified 2026-07-05)

| # | decision | consequence |
|---|---|---|
| U1 | **New GPU-native pipeline** — a new crate owns orchestration end-to-end (device-resident state, DAG scheduler, graphs, pipelining); the Backend-trait path stays as the reference until parity, then retires | §3 |
| U2 | **Byte-identity at milestones** — during development: proofs must verify + per-kernel differential gates; at every integration milestone: full byte-identity to the SIMD proof before anything defaults ON | §9 |
| U3 | **Fallbacks: keep during migration, then delete** — the end-state pipeline is single-path GPU; device failure = loud prove failure; fleet-level retry is the resilience layer | §3.4 |
| U4 | **Device Fiat-Shamir channel: in scope, with review** — built behind full transcript byte-equality vs the host channel; lands only after human review (security-critical class) | §5.7 |
| U5 | **North star: fleet $/MHz-hr** — 10–20 MHz aggregate on consumer cards at ~$0.15–0.35/MHz-hr; H100 is dev iron only. The VRAM diet is critical-path (M4), and a 4090 joins the milestone manifests from M2 | §7, §11 |
| U6 | **Fork product, upstream later** — build freely on the fork; Teddy is the approver for security-critical changes, with StarkWare review sought before any production/mainnet use; every soundness-adjacent divergence documented (KNOWN_ISSUES / divergence-log discipline) so upstream PRs stay possible | §9 |
| U7 | **Pod cadence: local-first, one validating session per milestone** (~$5–15 each; balance top-up ~$50–100 covers M0–M6); persistent pods only if a kernel-heavy milestone measurably demands iteration | §12 |
| U8 | **No hard deadline; sustained pace** — done = the M6 exit (pure GPU pipeline, fallbacks deleted, sustained fleet numbers) + the results writeup | §11 |

### The review verdicts these decisions rest on (condensed; evidence in the ledger)

**F1 — The prover is orchestration-bound, not kernel-bound.** 96% of dmon
samples <5% SM util during a full prove; every measured kernel is at or near
bandwidth pace when it runs (logup family 2ms device vs 300ms host). The
15–17s is host compute inside phases, host↔device round-trips between
components, launch gaps, and Fiat-Shamir drains. Measured four times (A′
async spine, A″ pipelined commit, B′ streams, force-relax): adding
parallelism to a starved GPU is flat; deleting host residency pays (+20%
round-17 composition fix; −5% device interaction).

**F2 — The JIT is two things; only one is right.** The recording→ISA→codegen
front-end is the program's core correctness asset: kernels generated from
THIS build's AIR by construction, statement-independent by design (the
lowering hoists every statement constant into parameters), validated by
interpreter/truth-oracle/byte-identity gates. It stays. The *runtime
compilation vehicle* (NVRTC + driver ptxas at prove time) goes: the kernel
set is fixed per (AIR revision, codegen version), so prove-time compilation
buys nothing and costs 11× cold start, the sm_90 driver-JIT cliff (240–290s
module loads at 2048 instrs → forced 512-instr splits → 4× composition
launches), NVRTC dialect walls (rounds 17, 28), and a compile-during-capture
hazard blocking CUDA graphs. → build-time AOT (§4).

**F3 — The component graph is the last host structure.** Witness kernels are
device-resident but components still communicate through host memory
(sub-word D2H → host rebuild → DashMap feeds → re-upload). The B-phase work
(count feeds, device edges) is the correct fix and is code-complete; the new
pipeline makes the device DAG the *only* transport (§5.2).

**F4 — The commit path is 75× off the streaming bound** (A40, base tree):
pass-count-heavy (unfused NTT stage pairs, separate pack, per-layer Merkle,
twiddle loads). Bandwidth-shaping work, second-largest win after the DAG (§5.4).

**F5 — A single proof has a serial transcript spine.** Beyond saturation,
parallelism must come from: rows within kernels (done), components within
phases (DAG), independent work across barriers (overlap), proofs across the
device (pipelining), PIEs across the fleet. The design provides all five axes.

---

## 1. Goal and first-principles budget

**Goal:** the fastest sound Cairo prover the silicon permits — a pure GPU
prover in which the host's steady-state role is: run the VM/adapter, bind
parameters, launch graphs, mirror the transcript, serialize the proof.
Measured as useful MHz on the SN PIE set; correctness contract per U2.

Physics (ENDGAME §1): ~8–10GB committed working set, ~6–10 fundamental DRAM
passes → 60–150GB traffic/proof. Floors: 4090 ≈ 50–120 useful MHz, A40 ≈
35–75; H100's compute/launch floor binds before its 3.9TB/s does. Nobody is
within 50× of the wall today; the gap is architecture.

Phase budget this design manages toward (SN_PIE_2, H100, spans from the
round-13/28 ledgers):

| phase | today | target | mechanism |
|---|---|---|---|
| ingest + tables | 1.5–2s cold | off critical path | §5.1, §8 |
| witness (Write Base) | 6.5–7.0s | **0.2–0.4s** | device DAG §5.2 |
| interaction | 2.1–2.6s | **~0.1s** | §6a everywhere §5.3 |
| commits (×3 trees) | 5.5s | **0.6–1.0s** | pass fusion §5.4 + graphs §6 |
| composition | 3.1s | **0.4–0.6s** | AOT fused kernels §4/§5.5 |
| FRI + PoW + decommit | ~1s | **0.2–0.4s** | single-graph FRI (device channel) §5.6–5.7 |
| **single proof** | **13.5–17.5s** | **~1.5–2.5s ≈ 3–5 useful MHz** | |
| sustained (2 in flight) | = single | **> single** | §8 |

Beyond ~1.5s the frontier is measured (nsys on a *saturated* device) kernel
engineering — NTT radix depth, hash throughput — not architecture. 4090-class
fleet cards project 2–3 MHz/card → the 10–20 MHz aggregate NORTH STAR at 5–8
cards; single-card 10 MHz is H100-class post-floor-work.

### 1.1 The residency contract (the formal definition of "maximum GPU residency")

Residency is not a vibe; it is a whitelist plus five ledger metrics. Anything
outside the whitelist is a violation the ledger must show shrinking to zero
by M6.

**Permitted host compute** (exhaustive): VM execution + PIE adaptation;
parameter binding + graph/kernel launching; transcript mirror arithmetic
(µs-class blake2s over 32B values); proof assembly + canonical serialization.
Nothing else — no witness math, no feeds, no igen, no repacking, no hashing
of column data.

**Permitted PCIe traffic** (exhaustive): H2D — execution tables, packed
states, borrowed pedersen table, per-proof scalars (drawn elements until the
device channel lands). D2H — 32B commitment roots (until the device channel),
decommit openings + queried values + proof aux at proof end. Nothing else —
no column ever crosses in either direction.

**The R-metrics** (recorded in every benchmark ledger entry from M1 on):

| metric | definition | M6 target (SN_PIE_2) |
|---|---|---|
| R1 | PCIe bytes/proof, each direction, excluding permitted ingest | **0** outside the whitelist; whitelist ≈ tables in + MBs out |
| R2 | host-compute ms on the critical path (excl. VM/adapter overlap) | **< 50ms** |
| R3 | kernel/graph launches per proof | **< 100** |
| R4 | dmon idle-sample share (<5% SM) during prove | **< 20%** |
| R5 | VRAM high-water by slot class | **≤ 20GB** (diet, 24GB cards) |

R1/R2 are the residency metrics proper; R3/R4 measure the orchestration
floor; R5 gates the fleet. The P-predictions (§11) are checkpoints of these
same metrics at specific milestones.

---

## 2. What survives from the as-built backend, and what retires

**Survives (proven load-bearing; the new pipeline consumes these as-is):**
- The recording/ISA/interpreter/codegen front-end for witness AND constraint
  kernels, and the entire gate ladder (truth-oracle selftests, differential
  gates whose reference does not share the implementation's model, whole-proof
  byte identity, canonical serialization). Three proof-breaking bugs and two
  falsified tables were caught pre-hardware by these gates.
- Device execution tables + computed deduces (ISA-V3) + borrowed pedersen
  table (GPU generation stays quarantined — falsified by the oracle leg).
- The device-DAG mechanisms: §6a interaction, descriptor-driven count feeds,
  edge gather kernels, transactional-pair exactly-once semantics.
- All hand kernels behind conformance gates: rfft/ifft, blake2s (+ fused
  layer-pair + single-block tail), quotients, FRI folds, grind, logup
  finalize, batched decommit gathers.
- The arena/pool, pinned staging (ingest-only in the end state), fatbin
  multi-arch distribution, gpufleet manifest discipline, phase ledger +
  pool high-water instrumentation.
- Async-spine + stream-parameterized launches + fresh-event fork/join (flat
  as standalone levers, prerequisites for graphs/overlap).

**Retires (with the overhaul):**
- `prove_cairo`'s SIMD-shaped orchestration as the GPU path (stays as the
  reference/parity oracle until M6, then CUDA support in it is deleted).
- Runtime NVRTC as the production execution vehicle (demoted to dev lane, §4).
- The NitrooZK precompiled constraint lane (`constraint_eval.rs` FNV dispatch,
  ~100 kernels): disqualified by its own header — raw `FrameworkEval`-struct
  ABI, 100%-row mismatches on AIR-rev skew, zero qualified components. The
  AOT lane is its correct replacement. Delete at M3.
- Host fallback lanes (host writers, host igen, CPU constraint lane) — per U3,
  deleted at M6 after parity; until then they are the migration safety net.
- The hand blake_g/pedersen witness kernels once the emitted lane matches
  them on hardware (one certified lane, not two).

---

## 3. The new pipeline (U1): crate, state, scheduler

### 3.1 Crate and boundaries

New crate `stwo_cairo_prover/crates/gpu-prover` (`stwo-cairo-gpu-prover`).
It is Cairo-specific by design — the component DAG, lane specs, and phase
schedule ARE the Cairo AIR's shape — and sits above `stwo` (core types,
verifier-shared proof structs, channel reference) and `stwo-backend-cuda`
(kernels, columns, arena). It does NOT implement stwo's `Backend` traits:
those encode the SIMD pipeline's control flow (synchronous per-phase calls,
host-owned values) and are exactly what the overhaul removes. The old
CudaBackend keeps compiling throughout the migration as the parity oracle.

```
gpu-prover/
  src/
    prover.rs        GpuCairoProver: persistent per-device context + prove()
    state.rs         DeviceProofState: every buffer of an in-flight proof
    schedule.rs      the component/phase DAG as data; topo levels; stream plan
    graphs.rs        capture, instantiate, param-rebind, replay, eager fallback
    transcript.rs    TranscriptEngine: host channel + device channel, mirrored
    phases/
      ingest.rs      tables/states upload, preprocessed tree
      witness.rs     the witness DAG (nodes = AOT lane kernels; edges = device)
      commit.rs      iNTT/LDE/Merkle per tree (fused path)
      interaction.rs logup pairs + finalize (§6a as the only lane)
      composition.rs fused AOT constraint kernels
      fri.rs         quotients + fold chain + PoW + decommit gathers
    proof.rs         assembly + canonical serialization (stwo core types)
  tests/             parity gates vs prove_cairo (per-phase + whole-proof)
```

`gpu_bench` gains `--engine gpu-native|legacy` (`--pipeline` was already taken
by pipelining depth) so every manifest step can A/B the
two pipelines on the same PIE from day one. The full two-repo directory map,
placement rules, code-design rules, the strict CUDA standard, the normative
API signatures, and the codegen tool inventory are §13–§17.

### 3.2 Persistent context vs per-proof state

`GpuCairoProver` (one per device, lives across proofs): AOT kernel registry
(§4), graph cache keyed by statement shape + size vector, arena with
identity-slot map (§6), twiddle cache, execution-table/pedersen-table caches
keyed by program, stream set, pinned staging ring for ingest.

`DeviceProofState` (one per in-flight proof; two exist under pipelining §8):
every column/tree/flat buffer of the proof, named by identity slot — witness
columns, sub/lookup buffers, count tables, the three commitment trees +
retained Merkle layers, composition accumulator, FRI layers, transcript
mirror, proof-aux staging. No host copy of any column, ever (U3 end state).

### 3.3 The schedule is data

The Cairo component DAG (today implicit in `cairo_claim_generator` spawn
order + the stash/edge seams) becomes an explicit table in `schedule.rs`:
per component — lane program id, input edges (producer, word range), output
edges, count relations, mult tables, log-size source. The scheduler derives:
topological levels → stream assignment (K streams, K≈4 measured sufficient)
→ event edges → the capture order for the witness graph. One table drives
witness, interaction, and igen; adding a component is a schedule row +
emitted kernel (the transformer already emits the per-component metadata:
`SUB_FEED_LAYOUT`, `JIT_LOOKUP_FIELDS`, slot layouts).

### 3.4 Failure model

Migration (M0–M5): every phase falls back to the reference path (that is what
keeps partial states shippable), loudly logged; a fallback engaged in a
manifest run is a regression. End state (M6): fallbacks deleted; any device
error fails the prove loudly; the fleet layer retries the block on another
node. No silent degradation in either regime.

---

## 4. Execution artifacts: recording-driven AOT (retiring runtime NVRTC)

The recording front-end stays the source of truth; compilation moves to build
time:

```
AIR rev → recording → ISA program → CUDA source → nvcc -O3 at BUILD time
             │             └ semantic hash ────────────┐
             └ interpreter (correctness oracle)        ▼
                                      fatbin + manifest keyed by
                                      (semantic_hash, codegen_version, sm)
```

- **Emit**: `tools/kernel_emit` walks every lane component + constraint
  component, records, lowers WITHOUT a split cap (one fused kernel per
  component; offline nvcc handles 17k-instr kernels fine — the cliff was
  load-time driver ptxas, which AOT deletes), emits `.cu` + `manifest.rs`
  (name, semantic hash, cache key, launch shape) into
  `backend-cuda-kernels/cuda/generated/`. Checked in, reviewed like any
  generated code — the repo's existing emit+`--check` drift idiom.
- **Why AOT is sound for witness kernels**: the recorded programs are
  straight-line — data-dependent control is lowered to 0/1 field ops (masks,
  select = m·a+(1−m)·b) and table reads, so a program depends only on the
  component's SHAPE, never on input values. Recording once (on the gate
  fixtures) captures the kernel for every statement.
- **Drift gate, two layers**: (1) CI: `kernel_emit --check` regenerates and
  byte-compares `generated/` against the AIR sources on every PR — staleness
  cannot merge. (2) Prove time, for free: the lane records anyway, and the
  recording's semantic hash IS the registry lookup key — a lookup miss is
  by definition drift (kernel ≠ this build's AIR) and fails closed (NVRTC
  dev lane or abort, by `strict` flag). No separate startup re-record pass
  is needed. This preserves the exact property that disqualified NitrooZK's
  kernels and justified the JIT — "generated from this build's AIR" —
  without prove-time compilation.
- **NVRTC remains** for the dev loop (new component before its kernel is
  checked in), the drift fallback, and exploration. The 512/2048 governor
  becomes NVRTC-lane-only.
- **Effects** (falsifiable): composition returns to fused kernels at -O3
  (force-relax proved SASS quality matters: -O0 was 2.6× worse; AOT gives
  fused AND -O3, which no runtime option could); cold == warm (today:
  ~23s aggregator + ~8s w18 + minutes for generic of NVRTC per cold cache per
  arch); graphs lose their hardest prerequisite (no compile during capture,
  structurally).

---

## 5. Subsystem designs

### 5.0 Target dataflow

**Rule: witness data lives on device from birth to Merkle root; the host
never holds a column.**

```
HOST                                DEVICE
────                                ──────
VM run + adapter (overlapped with
previous proof's GPU phases §8)
  ├─ H2D once (copy stream): exec tables, states, pedersen (borrowed),
  │                          preprocessed columns ────────────┐
  │  bind params, launch graphs                               ▼
  │                                witness DAG        (graph #1)
  │                                interaction        (graph #2, after draw)
  │                                commit per tree    (graphs #3a/b/c)
  │  transcript mirror ◄─ 32B ──   composition        (graph #4)
  │  (device channel: absorb/draw  quotients+FRI+PoW  (graph #5 — ONE graph
  │   on device, host mirrors and                      with the device channel)
  │   byte-checks §5.7)            decommit gathers ── D2H (only bulk D2H)
  └─ proof assembly + canonical serialization
```

Per-proof host↔device traffic: tables/states H2D once (~1–3GB, overlapped);
32B roots + drawn elements (or, with the device channel, nothing inside FRI);
decommit openings D2H at the end (MBs).

### 5.1 Ingest and tables
As built (upload once per memory, process-cached, device limb-split) +
table upload on the copy stream overlapped with the previous proof's tail
(first proof: overlapped with VM adaptation). Preprocessed columns are
**generated on device** (`gen_preprocessed_columns_on_gpu.cu` exists —
prefer generation over upload for residency, byte-gated vs the host
generator like every lane) and join the identity-slot map so graphs can
reference them.

### 5.2 Witness — the device component DAG (completes B2/B3, then owns it)
- **Nodes**: emitted AOT witness kernels for all 27+ lane components. The
  remaining host writers — the poseidon chains (full/partial, span share
  never separately quantified: MEASURE in M2's phase ledger before deciding
  their lane priority) and the small range-check/verify tail (<150ms
  measured) — run on host threads concurrently with the device DAG during
  migration (no device dependents; must finish before interaction); ALL of
  them join the lane before M6 (U3: the end state has no host writers, and
  the chains→cube_252 device edge — deferred round-27 because chains had no
  device sub buffer — unlocks when they do).
- **Edges**: count relations via descriptor-driven atomics (landed,
  gate-green); input-list relations via producer-sub → gather → consumer
  input columns (landed for agg→w18, blake→blake_g; the schedule table
  extends it to every producer/consumer pair as producers gain lanes);
  memory/table reads via execution tables (landed).
- **Mult tables**: device atomics; igen for mult columns emitted per §4 and
  reads device counts — no D2H.
- **Schedule**: explicit DAG (§3.3) → K streams → the witness graph.
- Expected: Write Base 6.5–7.0s → max-path kernel time ~0.2–0.4s (largest
  components measured at 340–400ms per 651k rows, and the family overlaps).

### 5.3 Interaction — §6a as the only lane
`logup_pairs` + device finalize for every component (opcodes landed; builtins
via the stash; memory/table components when their witness lands). Interaction
columns are born device-resident and feed commit directly. Measured basis:
2ms vs 300ms per family → whole-phase ~0.1s.

### 5.4 Commit — pass-count fusion
In leverage order:
1. **Stage-fused NTT** (radix-8/16 in shared memory/registers): a log-23
   column in 2 DRAM passes instead of one per stage pair. Pure
   `rfft.cu`/`ifft.cu` work behind per-size byte-equality conformance gates.
2. **Twiddle regeneration in-kernel**: M31 twiddles are shift/add-cheap;
   loading them spends bandwidth to save ALU. Also deletes the twiddle-tree
   size coupling that forced A″'s warm-cache dance.
3. **Hash-from-registers leaf layer**: last NTT/LDE stage keeps its tile in
   shared memory and feeds blake2s leaf compression — the pack pass and one
   full LDE read-write disappear. Soundness-adjacent → review path (C2
   precedent).
4. **Merkle interior**: layer-pair kernel wired for real (the round-13
   finding: its switch was dead) + the landed single-block tail (top 12
   levels, one launch).
5. **sm_90**: TMA/cp.async NTT staging; L2-persistence window for exec/
   pedersen tables during witness.
Target: 5.5s → 0.6–1.0s across three trees (~30GB commit traffic ≈ 10ms/pass
at H100 BW; budget ≈ 20 effective passes).

### 5.5 Composition
Fused AOT kernels at -O3, one/few per component, single accumulation pass,
in-place (unchanged semantics). Keep single-stream in-place accumulation —
split kernels share coordinate columns; fan-out loses updates (measured
constraint). Parallelism here is rows-within-kernel and across the graph.
Target 0.4–0.6s.

### 5.6 OODS, quotients, FRI, PoW, decommit
**OODS mask evaluation on device**: the sampled values at the OODS point
(the proof's `sampled_values` + the quotient numerators) evaluate via the
existing `eval_at_point`/`barycentric` kernels — only the resulting scalars
D2H (they are proof data and transcript input; permitted by §1.1). Quotient
combine already device. Fold chain static per layer schedule. PoW grind
device (byte-exact search order). Decommit via batched gathers, one D2H.
**Quotient→first-fold fusion stays deferred** (fri.rs:446 third-consumer
problem — soundness-critical, own reviewed design; its VRAM motivation is
served by the diet §7). With the device channel (§5.7) the whole phase —
folds, per-layer absorbs, PoW, query draw — becomes ONE graph.

### 5.7 The transcript engine and the device channel (U4)
`TranscriptEngine` owns Fiat-Shamir with two implementations:
- **Host channel** (today's `Blake2sChannel`) — the reference, always
  available, the migration default.
- **Device channel**: blake2s absorb/squeeze as single-block kernels
  operating on a device transcript state; drawn elements written to device
  buffers consumed in-graph by the next phase (fold alphas, OODS point,
  query positions); roots absorbed device-to-device from the Merkle tail.
  The host *mirrors* the transcript (cheap: recompute from the same 32B
  roots/values it receives for proof assembly) and **byte-checks the mirror
  against the device transcript state at every phase boundary** — a
  self-verifying transcript during migration; the check relaxes to
  debug-only after M5 soaks.
- **Contract (U4/U6)**: verifier compatibility is untouched — transcript bytes
  identical, only WHERE they are computed changes. Security-critical class:
  lands only behind full-transcript byte-equality vs the host channel across
  the whole manifest ladder AND explicit approval by Teddy before default-ON;
  StarkWare review before any production/mainnet use; the divergence
  documented per the log discipline.
- **What it buys**: FRI as one graph (the ~20 per-layer absorb barriers
  gone), commit→draw→next-phase chaining without host round-trips, and
  (with graphs) a proof whose host interaction is: bind, launch ~5 graphs,
  read back proof data.

---

## 6. Orchestration: graphs, streams, arena

- **Graphs per phase family**, captured once per (statement shape, size
  vector), instantiated in the graph cache, replayed per proof with
  parameter rebind (`cudaGraphExecUpdate` for table/state pointers).
  Prerequisites all land with §4/§3: no compile during capture (AOT),
  stream-parameterized launches everywhere (witness done; thread the
  remaining ~82 externs during the M2 launch-table rewrite), capture-safe
  event bridges (capture-local variant of the fresh-event fork/join),
  pointer stability via identity slots.
- **Identity-slot arena**: slot = (proof namespace, tree, column id,
  purpose), assigned at schedule time. Solves graph pointer stability AND
  gives pipelining its two clean namespaces (§8). High-water instrumentation
  per slot class replaces the pool-wide probe.
- **Streams**: K witness streams inside graph #1; copy stream (H2D ingest);
  D2H stream (roots/openings); graph-internal event topology encodes the
  §5.7 overlaps (tree k iNTT while tree k−1 Merkle top finishes — legal, the
  transcript orders only the absorbs).
- **Eager mode**: every graph has an eager launch path (same schedule, no
  capture) — the migration fallback and the debugging surface. Graph replay
  must produce byte-identical proofs to eager (manifest A/B step).
- **CUDA-version gate**: the fleet is heterogeneous (the H100 NVL pod runs
  CUDA 11.8; consumer pods vary). `cudaGraphExecUpdate` semantics and limits
  differ across 11.x/12.x, and conditional graph nodes (device-side loop
  control, the FRI-chain sweetener) are 12.4+. The graphs module probes the
  driver/runtime version at init and records it in every ledger entry; any
  version-dependent feature sits behind a capability check with the eager
  path as the below-version behavior — never a hard version floor (the
  NVRTC `--dopt=off` 11.8 wall is the precedent for what unprobed version
  assumptions cost).

---

## 7. Memory and the VRAM budget

Measured decomposition (round-9 ladder, SN_PIE_2, 47.6GB peak): witness evals
9.6 + preproc commit 7.6 + base commit 11.1 + interaction commit 6.8 + core
≈ +0 (pool reuse). Driver = committed-LDE retention.
- H100/dev iron: everything fits (14M-step PIEs measured at 64GB peak). No
  diet needed to hit this design's targets there.
- Consumer fleet (24GB): the diet design stands — compact each tree
  post-commit; regenerate LDE per consumer (composition per component group,
  quotients per query batch, decommit per position). Projected ~19–20GB.
  Regen nodes are shape-static → they join the graphs; pipelining hides
  their cost. §5.4's fusion pays for the extra pass twice over.
  **Fleet-first (U5): the diet lands WITH commit fusion at M4** (it is
  commit-path work — compaction and regeneration live in the same loop the
  fusion rebuilds). A 4090 joins the milestone manifests from M2 on fixtures
  that fit pre-diet (fib + the smallest PIE workloads); SN-PIE-class proofs
  on the 4090 start at M4 — the fleet card's trajectory is measured
  continuously, not discovered at the end.
- Pinned staging shrinks to ingest-only; the 2,500-copy staging path dies
  with full residency.

---

## 8. Multi-proof pipelining and the VM feed

Two proofs in flight as the standard operating mode: independent transcripts
and slot namespaces; shared immutable state (AOT kernels, graph cache,
twiddles-if-any, exec/pedersen tables when proving the same program). Proof
N+1's ingest + witness graph fills proof N's barrier drains and tails.

**VRAM admission control** (the constraint depth-2 must respect): two FULL
proof states do not fit a 24GB card (~19–20GB each post-diet) — depth-2
there is **phase-staggered**, not fully concurrent: the scheduler admits
proof N+1's next phase only when the arena can bind its slots, which in
practice means N+1's ingest+witness overlaps N's composition/FRI tail
(whose slot classes are small) — precisely the overlap that fills the
barriers, at a fraction of a second working set. On big-VRAM iron both
states fit and admission is a no-op. The admission check is
`Arena::bind` failing softly to a wait, never an OOM abort; the ledger
records admission stalls so the overlap's realized depth is measured. The
P5 harness measured sustained ≈ single-proof with 3 producers and zero feed
starvation; post-residency the same harness must show sustained > single
(prediction P8). Pool concurrency is the one unvalidated surface (round-27)
— the namespace split makes it explicit; validate under the manifest before
default-ON. The VM feed is the endgame bottleneck (ENDGAME §6): 2–4 host
cores per GPU at 3–5 MHz; procurement stays cores-first; at fleet scale the
answer is more nodes — the $/MHz-hr thesis.

---

## 9. Correctness architecture (U2) — how a total overhaul stays sound

- **Development invariant** (every PR): proofs verify (Rust verifier; Cairo
  verifier in the milestone manifests) + per-kernel differential gates
  (interpreter/truth-oracle/conformance) + the drift gate (§4).
- **Milestone invariant** (M1–M6, before anything defaults ON): whole-proof
  **byte-identity to the SIMD backend** on the SN-PIE manifest, plus
  determinism (two proves byte-identical), plus verify. Byte-identity is
  achievable at every milestone because the pipeline reorders WHERE and WHEN
  values are computed, never WHAT they are: the transcript pins all
  committed values; canonical serialization (Stage 0) pins the bytes.
- **The reference lives**: `prove_cairo`+SimdBackend (and the legacy CUDA
  path until M6) as the parity oracle in CI and manifests.
- **Ladder**: local gates → pod manifest (oracle legs → whole-kernel gates →
  lane/graph A/Bs each byte-identical → composed run) — unchanged discipline,
  new steps `gpu-native-parity`, `graphs-identity`, `transcript-mirror`,
  `sustained-2proof`.

---

## 10. Rejected alternatives

- **Runtime JIT as production execution** — F2; front-end stays, prove-time
  compilation goes.
- **Persistent megakernel**: abandons per-kernel differential gates, fights
  occupancy variance; graphs deliver the launch win without it.
- **Multi-GPU single proof**: throughput comes from block parallelism;
  intra-proof distribution = unneeded latency at 3× complexity + soundness
  surface.
- **Tensor-core/FP NTT**: M31 integer butterflies don't map; bandwidth, not
  ALU, is the binding resource.
- **Quotient→FRI fusion now**: fri.rs:446 third consumer — soundness-critical,
  own reviewed design later.
- **In-place refactor of the Backend traits** (considered as the overhaul
  shape, rejected per U1): the trait contract encodes synchronous host-owned
  phases — residency, graphs, and pipelining all fight it; the strangler
  crate is cheaper than bending it.
- **Verify-only correctness during the whole overhaul** (rejected per U2):
  subtle soundness bugs can pass verify; byte-identity at milestones keeps
  the net.

---

## 11. The overhaul plan — strangler milestones

Each milestone: implement → dev-invariant green locally → pod manifest with
the milestone invariant (§9) → default-ON in the composed config. Bands are
engineering estimates against the 13.5s composed baseline; the ledger grades
them. M0 is unblocked (balance $19.86 as of 2026-07-05 covers one session;
the resume checklist is in the program memory) — top-up to ~$50–100 before
M2 to cover the remaining milestones (U7).

| M | contents | exit gate | expected composed SN_PIE_2 |
|---|---|---|---|
| **M0** | Finish round-28 validation: generic engagement, DAG byte-identity, cubin-2048 A/B, sustained-DAG, composed run | existing 13-step manifest green | honest current-best (~12–13s est.) |
| **M1** | `gpu-prover` crate scaffold: GpuCairoProver + DeviceProofState + schedule table + phases calling the EXISTING lane/commit code; `--engine gpu-native` A/B (gpu_bench moves to the gpu-prover crate — the harness sits above both engines) | **parity**: byte-identical to legacy pipeline + SIMD | = M0 (structure, not speed) |
| **M2** | Witness DAG owned by the scheduler: **schedule_emit** generates the ComponentNode table + COUNT_RELATIONS; all count feeds + edges via the schedule; §6a for builtins; memory/table components into the lane; stream ABI sweep via **ffi_emit** | milestone invariant + P1 (idle < 60%) + Schedule::validate in CI | **~9–10s** |
| **M3** | AOT: **kernel_emit**, checked-in generated .cu, drift gate, fused -O3 composition; ffi_emit owns raw.rs/stubs.rs; DELETE NitrooZK lane | drift gate green; cold==warm (P5); `make codegen --check` in CI | **~7.5–8.5s** |
| **M4** | Commit fusion: stage-fused NTT, twiddle regen, layer-pair, then hash-from-registers (review path); **VRAM diet lands here (U5)** — SN_PIE_2 proves on the 4090 | conformance + identity + P6 (BW-scaling) + 4090 fit | **~5–6s** H100; first real 4090 number |
| **M5** | Graphs per phase + device channel (review + transcript mirror) + inter-tree overlap | graphs A/B + transcript byte-equality + P7 (<100 launches), P9 (idle <20%) | **~3–4s** |
| **M6** | Two-proof pipelining; small-writer tail into the lane; **DELETE host fallback lanes + legacy CUDA path** | P8 (sustained ≥ +15%); fleet manifest | single **~2.5–3.5s / ~2.2–3 MHz**; fleet **10–20 MHz agg** |
| **M7** | Floor work on the saturated device (nsys-ranked: NTT radix, hash throughput, TMA/L2) | perf ledger | toward **~1.5–2s / 4–5 MHz** H100 |

Falsifiable predictions (grading the doc, extending NEXT_GPU_OPTIMIZATIONS §6):
P1 idle <60% after M2 · P5 cold==warm ±5% after M3 · P6 commit spans scale
with memory BW ±20% across H100/4090/A40 after M4 · P7 launches/proof <100
after M5 · P8 sustained ≥ single +15% after M6 · P9 idle-sample share <20%
after M5.

## 11b. Implementation status (living section)

Updated 2026-07-05. Everything below is committed, locally gated, and stub-safe
on macOS; pod validation batched into one session (running).

- **M1 COMPLETE**: `crates/gpu-prover` — GpuCairoProver (persistent twiddle +
  preprocessed-tree caches), phases/{ingest,witness,commit,interaction,stark},
  CairoBackend traits, flags registry, schedule types. Parity gate GREEN
  locally (byte-identical to prove_cairo, cold + warm). gpu_bench moved here
  (+ `--engine`); fleet scripts updated.
- **M2a COMPLETE**: schedule_emit → 36-node generated feed DAG (zero hand
  data); certified edges pinned; --check in pregate.
- **M2b COMPLETE**: §6a for builtins. JIT_LOGUP_DESCS emitted facts (7-form
  closed grammar parsed from every generated writer — pairing is arbitrary
  order, signs vary, mults ∈ {flats, one, enabler}); generalized device
  descriptors (MultSrc + explicit signs + n_real); host mirror
  finalize-identical to the writers on all three shape classes (LOCAL gates,
  zero CUDA); stash + collection branches for the five builtin lanes.
- **M3 LOCAL-COMPLETE**: aot emit surface; kernel_emit (118 kernels @ the
  cap-aligned 2048, coverage-fenced fixture matrix, --check in pregate);
  build.rs per-arch -O3 cubin pack embedded + runtime tier-0 lookup (miss =
  drift = NVRTC fallback); NitrooZK lane DELETED; manifest cap 512→2048 +
  sn2-aot-pack gate. gpu-native engine defaults = the composed device config.
- **M2 tail deferred**: ffi_emit + stream sweep, memory/table components into
  the lane, witness-phase-owns-schedule — after the pod session's numbers.
- **Pod session COMPLETE (2026-07-06, 15/15 gates + records, $4.90)**: every
  gate byte-identical; the DAG and AOT levers COMPOSE. Records (warm, useful
  MHz): SN_PIE_2 10.76s/0.716 (+25%), SN_PIE_1 18.2s/0.805 (+113%),
  SN_PIE_3 14.35s/0.981 (+151%), SN_PIE_4 14.92s/0.942 (+96%); sustained
  0.543 with zero feed starvation. Two hardware-only AOT walls fixed
  (offline-nvcc curandState in the fp256 embed; anonymous-namespace extern
  linkage — the C13 class). VRAM: SN_PIE_1 DAG config peaked 75.4GB — the
  M4 diet is now also an H100 requirement for 14M PIEs, not only a fleet one.
- **M4 increment VALIDATED (2026-07-06, run 20260706T023111Z)**: word-block
  blake2s Merkle (leaf/lifted/interior hash M31 words directly, register
  blocks, no byte buffer) + device memory count feeds v2 (opcode lanes +
  blake/aggregator seams: addr −1 offset, mem-id decode with big/#small
  split, sigma LUT; skip-guarded hand feeders). Byte-identical on SN_PIE_2
  (M4_PROOF_MATCH). One soundness-relevant bug found AND fenced: the eager
  word-block loop mis-hashed streams that end on a full 64-byte block
  (column count ≡ 0 mod 16 — SN_PIE_3's FRI first layer; RootMismatch on
  internal verify, both engines). Fix = lazy loop matching blake2s_update;
  the testkit merkle conformance now pins 16/32-column trees, CUDA
  conformance green on pod. Records (warm, useful MHz): SN_PIE_2
  **8.98s/0.858** (first sub-10s OS proof), SN_PIE_3 **12.43s/1.133**,
  SN_PIE_1 **14.90s/0.983**. Per-phase VRAM ledger (STWO_VRAM_PHASES):
  **base_commit is the peak** (40.7GB of SN2's 42.5; witness 32.4,
  stark_core 34.4) — the diet targets the base LDE+tree working set.
  Remaining M4 tail: diet design from the ledger, twiddle regen-in-kernel,
  stage-fused NTT audit, 4090 fit.
- **M5a VALIDATED (2026-07-06, run 20260706T030913Z)**: the builtin witness
  section runs as 7 concurrent dependency arms (edges from the certified
  schedule + per-writer state arguments; cross-arm states are atomic
  commutative counts; consumers stay in the sequential tail; evals order
  restored canonically from slots) and STWO_CUDA_STREAM_FANOUT=1 joined the
  gpu-native defaults (lanes overlap on 4 pool streams, fork/join bridged).
  Byte-identity: gpu_native_parity_simd GREEN local, M5A_PROOF_MATCH pod.
  Records (warm, useful MHz): SN_PIE_2 **7.39s/1.043**, SN_PIE_3
  **8.78s/1.604**, SN_PIE_4 **10.34s/1.359**, SN_PIE_1 **10.81s/1.355** —
  all four PIEs above 1 MHz useful. Ledger: Write Base trace 4.08→2.33s;
  prove_cairo 7.47s ≈ witness 2.3 + commit 2.0 + STARKs 2.2 — the pie is
  even, so M5b inter-tree overlap and M6 pipelining are the road on.
  Fleet math: 10 MHz aggregate ≈ 6-7 H100s.

- **M5b VALIDATED + FINALIZED (2026-07-06)**: batched OODS (group by
  (log_size, folded point), one launch pair + one D2H per group) — within-
  session A/B −0.38s on SN3, OODS span 0.56→0.076s, byte-identical; guarded
  at grid.y>65535 (adversarial-review finding). Leaf-hash __launch_bounds__:
  Merkle-span decomposition showed 95% of Merkle is the log24 LEAF hash
  (635ms/rep, occupancy-bound), cut to 512ms/rep (−19%) by capping registers.
  Per-lane committer DISABLED from defaults (A/B: helps SN2 ~0.3s, within-
  noise/negative on 14M PIEs — overlap must pay; flag+code retained).
  Sustained pipelining 0.543→0.741 useful MHz (+36%). Methodology: pod
  inter-session variance ~7-8%, within-session 2.1% — A/B on one pod state is
  the reliable measurement. Adversarial workflow: 4/5 findings refuted.
- **10 MHz THESIS (converged)**: single-card intra-proof micro-levers are
  exhausted (sub-second, near noise). The linchpin is the VRAM DIET at the
  base_commit peak (40.7GB): sustained pipelining is below single (0.74<1.05)
  because 2× SN2 (85GB) don't fit an 80GB card → concurrent proofs serialize
  on VRAM. stream_lde only cuts POST-commit retention, not the peak (all ~700
  LDE'd base columns must be resident for the leaf hash). The real diet =
  STREAM LDE COLUMN-GROUPS INTO THE LEAF HASH incrementally (LDE a group →
  update every leaf's running blake2s state → free the group), cutting the
  peak from all-columns to one-group + leaf states. Byte-identical (blake2s
  is streaming), soundness-adjacent (VCS path → conformance gate + review).
  Then 2 proofs fit → M6 pipelining pays (sustained → ~1.5-2× single) → fleet
  → 10 MHz aggregate. The diet is thus the H100 pipelining gate, not just the
  4090 gate.

## 11c. Next-lever ranking (adversarial-review workflow, 2026-07-06)

An 11-agent adversarial workflow reviewed the M5b changes (4/5 findings
refuted; 1 confirmed = a latent OODS grid.y>65535 robustness cap, guarded)
and ranked the remaining levers toward 10 MHz, grounded in the M5a SN2
ledger (prove_cairo 7.47s ≈ Write Base 2.33 + Commitment 2.01 (Merkle 1.75)
+ Prove STARKs 2.22 (OODS ~0.56) + ~0.9 interaction/ingest). The
load-bearing constraint (F5): a single proof's Fiat-Shamir spine is serial —
intra-proof overlap only hides work independent up to the next absorb;
filling the idle 96% SM needs a second proof (M6). Overlap and pipelining
are complementary, not redundant.

1. **Inter-tree / cross-phase overlap (M5b + extension)** — commit hidden
   under witness arms; batched OODS off the STARK-core path; tree k iNTT ‖
   tree k−1 Merkle tail. ~7.47→5.5–6.0s at M5b, toward 3–4s with more
   overlap. Low risk, soundness-neutral, no gate, pod-validated.
2. **CUDA graphs** — capture each phase family per (shape, size vector),
   replay with pointer rebind; attacks the F1 launch-gap floor (R3<100,
   R4<20%). ~1–2s if idle is launch-dominated. Covers witness/commit/
   composition; NOT FRI without #4. Medium risk; capability probe (pod is
   CUDA 11.8; cudaGraphExecUpdate/conditional-node limits vary) + eager
   fallback. No soundness gate.
3. **Commit-path kernel throughput** — stage-fused radix-8/16 NTT (~2 DRAM
   passes vs per-stage-pair), in-kernel twiddle regen, Merkle interior
   layer-pair + tail. Commitment 2.01→0.6–1.0s (~1.0–1.4s); real headroom
   because F4 measured commit 75× off the streaming bound. Conformance-gated
   per size EXCEPT hash-from-registers (soundness-adjacent → own review).
4. **Device Fiat-Shamir channel** — blake2s absorb/squeeze on device, drawn
   elements in device buffers, roots absorbed D2D. ~0.3–0.6s standalone but
   a MULTIPLIER for #1/#2 (turns FRI+PoW+decommit into one graph). HARD gate:
   U4/U6 full-transcript byte-equality + Teddy approval before default-ON,
   StarkWare before mainnet. The host mirror byte-check is the safety net.
5. **M6 two-proof pipelining** — two DeviceProofStates, N+1 ingest/witness
   fills N's barrier drains. Sustained-throughput lever, not single-proof
   latency; currently NEGATIVE (M5a sustained 0.543 < single 1.043) because
   the device isn't residency-clean — turns positive only AFTER #1 lands.
   For the fleet $/MHz-hr north star (U5) it is THE axis, gated behind #1–#4.

VRAM diet: not a single-card MHz lever (can cost an LDE-regen pass) but a
FEASIBILITY requirement — SN_PIE_1 peaked 75.4GB, base_commit is the 40.7GB
SN2 peak — and the gate for the 4090 fleet. Rides in with #3 at M4 (compaction
lives in the commit loop the NTT fusion rebuilds).

Cross-cutting caveat: #1/#2/#4 interlock (graphs want the channel to cover
FRI; the channel wants graphs for chaining; both want the overlap schedule as
their capture topology) — expect sub-additive gains where they touch the same
phase boundaries. The ledger grades each; every one stays behind whole-proof
byte-identity before default-ON.

## 12. Effort and risk

| workstream | size | risk | pod-dependent? |
|---|---|---|---|
| M0 validation | 1 session | billing only | yes |
| M1 scaffold | ~1 week | low (moves code, byte-gated) | parity run only |
| M2 DAG completion | ~1 week | low (pattern proven ×3) | gates |
| M3 AOT | ~1 week | low (emit+check idiom exists) | build validation |
| M4 NTT fusion | 1–2 weeks | medium (deep kernels, strong gates) | iterated |
| M4 hash-from-regs | ~1 week | med-high (review path) | iterated |
| M5 graphs+channel | 1–2 weeks | medium (capture semantics; channel review) | iterated |
| M6 pipelining+deletion | ~1 week | medium (pool concurrency) | soak |

Cadence (U7): everything engineered and gated locally; ONE validating pod
session per milestone (~$5–15 on H100/4090 spot; top-up ~$50–100 covers
M0–M6; check `clientBalance` BEFORE resuming pods — round-28 was terminated
mid-run at −$0.13). A persistent dev pod (with a network volume) is warranted
only if M4/M5 kernel iteration measurably stalls on the session cadence.

Risk control is the program's evidence discipline: nothing defaults ON
without byte-identity on the manifest; every fallback is loud; the reference
pipeline survives until M6; and the ledger — not this document — decides each
next lever.

---

## 13. Code organization — the normative directory structure

### 13.1 Repository and crate map (both forks)

```
stwo/                              fork teddyjfpender/stwo, branch perf-optimizations
  crates/
    stwo/                          core: verifier-shared proof types, channel
                                   reference, canonical serialization. The
                                   soundness-critical surface — changes gated
                                   per CLAUDE.md. gpu-prover CONSUMES it; it is
                                   never forked further.
    constraint-framework/          the recording EvalAtRow surface (front-end input)
    backend-cuda-kernels/          ★ THE ONLY CUDA TRANSLATION UNITS ★
      build.rs                     offline nvcc fatbin (sm_86,sm_89,sm_90) + macOS stub
      cuda/
        *.cu, *.cuh                hand kernels: rfft/ifft, blake2s (+layer-pair,
                                   single-block tail), quotients, FRI folds, grind,
                                   logup pairs/finalize, feed counts, edge gather,
                                   exec/pedersen tables, mem pool, fp256/ec_ops chain
        generated/                 ★ NEW (M3): AOT-emitted witness + constraint
                                   kernels + manifest — machine-written by
                                   kernel_emit, NEVER hand-edited, CI drift-checked
        runtime_jit.cu             NVRTC dev lane (demoted, kept)
      src/raw.rs                   every extern "C" decl, 1:1 with kernels
      src/stubs.rs                 no-CUDA stubs (CUDA_KERNELS_BUILT=false)
    backend-cuda/                  safe Rust over the kernels crate
      src/columns/                 device buffer types (BaseFieldVec, SecureFieldVec,
                                   UploadedUint32Vec, UploadedDevicePointerVec, …)
      src/backend/                 legacy Backend-trait impl — PARITY ORACLE until
                                   M6, then CUDA support deleted
      src/backend/jit_witness/     recording/ISA/interpreter/codegen front-end (KEEP)
      src/backend/jit/             constraint recording/lowering front-end (KEEP)

stwo-cairo/                        fork, branch generic-backend
  stwo_cairo_prover/crates/
    gpu-prover/                    ★ THE NEW PIPELINE (tree in §3.1) ★
    prover/                        legacy prove_cairo + witness components
                                   (reference pipeline + the writers the
                                   transformer reads)
    adapter/                       VM → ProverInput (host, permanent)
  stwo_cairo_prover/tools/       (the codegen chain — policy + inventory §17)
    witness_genericize/            transformer (KEEP: emits generic writers,
                                   SUB_FEED_LAYOUT, JIT_LOOKUP_FIELDS, igen)
    kernel_emit/                   ★ NEW (M3): records + lowers + emits
                                   generated/ + manifest; --check = drift gate
    schedule_emit/                 ★ NEW (M2): metadata + feed-map parse →
                                   schedule.rs ComponentNode table + COUNT_RELATIONS
    ffi_emit/                      ★ NEW (M2-M3): kernel exports → raw.rs +
                                   stubs.rs + typed launch wrappers
  gpu_benchmarks/                  docs (this file), manifests, fleet/, RESULTS ledger
```

### 13.2 Placement rules (enforced at review + CI)

- **CUDA text exists only in `backend-cuda-kernels/cuda/`.** No kernel source
  strings anywhere else (the NVRTC dev lane consumes files from that tree).
  No `extern "C"` declarations outside `raw.rs`/`stubs.rs`; the two must stay
  mirror-complete (a CI check compares symbol lists).
- **`gpu-prover` never touches raw FFI.** It consumes backend-cuda safe types
  and kernels via the `KernelRegistry` (§16.4) and typed phase wrappers. If a
  phase needs a new kernel surface, the safe wrapper lands in backend-cuda
  first.
- **`generated/` is written only by `kernel_emit`.** Hand edits are reverted
  by CI (`kernel_emit --check`, byte-compare, comment-insensitive — the
  existing idiom).
- **Layering inside gpu-prover** (downward deps only, no cycles):
  `prover.rs` → `phases/*` → `schedule.rs` (data) + `graphs.rs` +
  `transcript.rs` (mechanisms) → backend-cuda. Phases never call each other;
  the prover sequences them. `state.rs` (DeviceProofState) is passed down,
  never reached up for.

---

## 14. Code design rules

- **R1 — Ownership.** Every device allocation is owned by `DeviceProofState`
  under a `SlotId` (§16.2). Raw device pointers cross module boundaries only
  into launch calls, never stored outside the arena. Streams, events, and
  graphs are RAII types (the `StreamFork` acquire/Drop precedent); no naked
  create/destroy pairs.
- **R2 — Errors.** `Result<_, GpuError>` end to end, typed causes (`Launch`,
  `Alloc`, `Capture`, `Drift`, `TranscriptMismatch`, `Fallback(reason)`).
  Fail-closed always; every migration fallback logs its reason at engage
  (round-9 rule: no silently idle GPUs). `panic!` is reserved for states
  where no sound fallback exists (the split-kernel
  partially-accumulated-launch-failure precedent, `jit/mod.rs`).
- **R3 — Determinism.** Device results must be completion-order-independent:
  atomics only for commutative integer sums (counts/mults); everything
  reaching the proof serializes canonically (the Stage-0 BTreeMap rule). Any
  order dependence is a bug; the two-prove byte-identity gate is its
  detector.
- **R4 — Flags.** Every feature sits behind a default-OFF env flag registered
  in one table (`gpu-prover/src/flags.rs`): name, default, owner, gate step,
  **deletion milestone**. Unregistered `std::env` reads in gpu-prover are
  lint-banned. Flags are migration scaffolding, not configuration — U3 means
  most of the table deletes at M6.
- **R5 — Emitted code.** Marked, never hand-edited, drift-gated. Hand
  additions live outside emitted blocks (round-20 lesson: the emit template
  must absorb any hand-added gate code before a re-emit, or it silently
  reverts).
- **R6 — Tests land with the code.** Unit-pinned structural rules (descriptor
  pairing, slot layouts, schedule validation); differential gates whose
  reference does NOT share the implementation's model (round-10 rule — the
  selftest that compared kernel≡interpreter under the same wrong model passed
  for two sessions); parity gates vs the reference pipeline; manifest e2e.
- **R7 — No permanent dual paths.** A fallback lane is created WITH its
  deletion milestone (U3). Post-M6 the pipeline is single-path; resilience is
  fleet-level retry.
- **R8 — Generate, don't handwrite.** Anything derivable from the AIR,
  emitted metadata, or a repetitive schema is produced by a codegen tool with
  a `--check` drift gate (§17). Hand-writing only for gate-covered primitive
  kernels and frozen 1:1 host transcriptions.

---

## 15. CUDA engineering standard (strict; violations block merge)

Each rule cites the measured incident or precedent that makes it
non-negotiable. Reviews check against this list explicitly.

**Memory**
- **C1** Structure-of-arrays / column-major for all trace data; word-major
  for multi-word flats. (Codegen v1→v2: the row-major store was one of the
  two proof-breaking pilot bugs; word-major is also the coalesced layout.)
- **C2** All arena allocations 128-byte aligned; kernels may assume it.
- **C3** No per-element transfers. D2H/H2D is batched (the
  `cuda_multi_layer_batch_get` gather replaced thousands of 32B stalls) and
  asynchronous on named streams with event fences (A′ primitives).
- **C4** Steady-state device memory comes only from the arena; raw
  `cudaMalloc/cudaFree` inside a prove is forbidden. Per-slot-class
  high-water stats in every benchmark record (the pool probe precedent).
- **C5** Kernel pointer parameters are `const ... * __restrict__` wherever
  semantically true.

**Execution**
- **C6** No legacy/default stream anywhere in gpu-prover: every launch takes
  an explicit stream parameter. (Graphs prerequisite; the witness launch
  already migrated — the remaining ~82 externs migrate at M2.)
- **C7** `cudaDeviceSynchronize` is debug-only; production synchronization is
  event waits at true data edges.
- **C8** Block sizes are validated against register pressure at init
  (occupancy query + the C2 tail lesson: 1024 threads × inlined blake2s
  exceeded the register file and failed only on hardware). Launch shapes
  live in the AOT manifest, not in code literals.
- **C9** Atomics only for order-independent accumulation — integer adds
  exclusively; floating-point atomics are banned outright (non-associative →
  completion-order nondeterminism → breaks R3 and the byte-identity gates;
  nothing in an M31 prover needs them). In-place accumulation kernel
  families stay single-stream (composition split kernels share coordinate
  columns — fan-out loses updates, measured constraint).
- **C10** No dynamic parallelism, no in-kernel malloc, no printf in
  production kernels, no grid-wide sync primitives — the single-block tail
  kernel is the sanctioned pattern for small cross-block dependencies.

**Compilation and ABI**
- **C11** Production SASS is offline `nvcc -O3` fatbin (sm_86, sm_89, sm_90).
  Driver JIT and NVRTC are dev-lane only (F2). `-Xptxas -v`
  register/spill reports are archived per kernel in the manifest — a spill
  regression is reviewable like a perf regression.
- **C12** Explicit C ABI only: pointers + scalars, lengths as explicit
  `u64`, no struct-by-value, no Rust layout crossing the boundary (the
  NitrooZK disqualification), names `stwo_<area>_<verb>`, stream parameter
  LAST, `bool` success + last-error query for detail.
- **C13** Any emitted-source change bumps the codegen version mixed into
  every cache key (the `WITNESS_CODEGEN_VERSION` rule — stale-cache
  collisions are impossible by construction). `extern "C"` declarations at
  global scope only (round-28: anonymous-namespace extern "C" gets internal
  linkage and fails at rust-lld).
- **C14** Every CUDA driver/runtime/NVRTC call is checked; failures surface
  `cuGetErrorString`/the NVRTC log verbatim (round-17: the surfaced log is
  what falsified the wrong "offline cubins" fix-guess).

**Observability**
- **C15** NVTX range per phase and per component; the phase ledger
  (`STWO_BENCH_TRACE`) and dmon duty-cycle sampling ship in every benchmark
  record; nsys is the arbiter for post-saturation (M7) decisions.

**Graphs era (M5+)**
- **C16** Nothing in a capturable path may: compile, allocate outside the
  arena, touch the legacy stream, or record events on out-of-capture
  streams. Graph replay must byte-match the eager path (manifest A/B step).

---

## 16. Formal interfaces (normative)

Signatures are normative in shape (names, ownership, error flow); field
lists may grow. Types referenced from stwo core (`CairoProof`,
`SecureField`, `Blake2sHash`) and backend-cuda (`BaseFieldVec`, …) are the
existing ones.

### 16.1 Prover API (the public surface)

```rust
pub struct GpuProverConfig {
    pub device: u32,
    pub vram_budget: Option<usize>,      // None = card total; drives diet mode
    pub pipeline_depth: usize,           // 1 | 2 (§8)
    pub channel: ChannelMode,            // Host | DeviceMirrored (§5.7, U4)
    pub strict: bool,                    // true (post-M6): no fallbacks, fail loud
}

pub struct GpuCairoProver { /* registry, graph cache, arena, streams, caches */ }

impl GpuCairoProver {
    pub fn new(cfg: GpuProverConfig) -> Result<Self, GpuError>;
    /// Load AOT kernels, run the drift gate, pre-capture graphs for a shape.
    pub fn warmup(&mut self, shape: &StatementShape) -> Result<(), GpuError>;
    pub fn prove(&mut self, input: ProverInput) -> Result<CairoProof, GpuError>;
    /// Depth-2 pipelining: proofs stream out in order; ingest/witness of N+1
    /// overlaps barriers of N.
    pub fn prove_stream<'a>(
        &'a mut self,
        inputs: impl Iterator<Item = ProverInput> + 'a,
    ) -> impl Iterator<Item = Result<CairoProof, GpuError>> + 'a;
}
```

### 16.2 Arena and identity slots (§6)

```rust
pub struct SlotId { pub ns: ProofNs, pub tree: TreeId, pub column: ColumnId, pub purpose: Purpose }
// Purpose ∈ {WitnessEval, LdeCoeffs, LdeEvals, MerkleLayer(u8), SubWords,
//            LookupWords, CountTable(FamilyId), Accumulator, FriLayer(u8), …}

impl Arena {
    /// Address is stable for the lifetime of a capture epoch (graph validity).
    pub fn bind(&self, slot: SlotId, len: usize) -> Result<DeviceBuf, GpuError>;
    pub fn resolve(&self, slot: SlotId) -> Option<DeviceBuf>;
    pub fn compact(&mut self, tree: TreeId) -> Result<(), GpuError>;   // diet (M4)
    pub fn high_water(&self) -> ArenaStats;                            // per Purpose class
}
```

### 16.3 The schedule as data (§3.3)

```rust
pub struct ComponentNode {
    pub id: ComponentId,
    pub kernel: KernelKey,                  // AOT manifest key (semantic hash + version)
    pub log_size: LogSizeSource,            // FromStates | FromProducer(ComponentId) | Fixed(u32)
    pub inputs: &'static [InputEdge],       // Producer{of, word_base, words} | ExecTables
                                            //   | DeviceTable(TableId) | HostFixture (migration-only)
    pub outputs: &'static [OutputEdge],     // sub-word ranges consumed downstream
    pub counts: &'static [CountRelation],   // device-atomic families (registry-checked)
    pub slots: SlotLayout,                  // [flat 0..K | enabler | iota | mults..] — pinned
}

pub struct Schedule { pub nodes: &'static [ComponentNode] }
impl Schedule {
    pub fn levels(&self) -> Vec<Vec<ComponentId>>;          // topological, → stream plan
    /// Rejects: cycles, dangling edges, slot/width mismatches vs the emitted
    /// metadata, count families missing from COUNT_RELATIONS. Runs in CI.
    pub fn validate(&self) -> Result<(), ScheduleError>;
}
```

### 16.4 AOT kernel registry (§4)

```rust
pub struct KernelRegistry { /* fatbin modules + embedded manifest */ }
pub struct KernelHandle { /* CUfunction, launch shape, abi_version */ }

impl KernelRegistry {
    pub fn load() -> Result<Self, GpuError>;                       // from the fatbin
    /// The drift gate (§4): re-recorded semantic hashes vs the manifest.
    /// Mismatch → Err(Drift) → NVRTC dev lane or abort (strict mode).
    pub fn drift_check(&self, recorded: &[SemanticHash]) -> Result<(), GpuError>;
    pub fn get(&self, key: KernelKey) -> Result<KernelHandle, GpuError>;
}
// Only backend-cuda calls KernelHandle::launch (unsafe, explicit stream);
// gpu-prover uses typed per-phase wrappers that own parameter marshaling.
```

### 16.5 Transcript engine (§5.7, U4)

```rust
pub trait Transcript {
    fn absorb_root(&mut self, root: Blake2sHash);
    fn draw_felts(&mut self, n: usize) -> Vec<SecureField>;
    fn draw_queries(&mut self, bounds: QueryBounds) -> Vec<usize>;
    fn state_digest(&self) -> [u8; 32];          // the mirror-check surface
}

pub struct HostTranscript(Blake2sChannel);        // the reference, always available
pub struct DeviceTranscript {                     // U4; security-critical review path
    state: DeviceBuf,                             // device channel state
    mirror: HostTranscript,                       // host recomputation
}
impl DeviceTranscript {
    /// Absorb a DEVICE-resident root (e.g. straight from the Merkle tail
    /// kernel's output) without a D2H round-trip — the residency point of
    /// the device channel. The mirror receives the 32B copy on the D2H
    /// stream asynchronously (it is proof data anyway, §1.1-permitted).
    pub fn absorb_root_device(&mut self, root: DeviceBuf);
    /// Byte-compares device state vs mirror; called at EVERY phase boundary
    /// during migration (debug-only after M5 soak). Mismatch = abort.
    pub fn check_mirror(&self) -> Result<(), GpuError>;
    /// Device-resident drawn values for in-graph consumption (fold alphas,
    /// OODS point, query positions) — the single-graph-FRI enabler.
    pub fn device_drawn(&self, slot: SlotId) -> DeviceBuf;
}
// The `Transcript` trait's host-value methods are the MIRROR's surface (and
// HostTranscript's whole interface); the device impl's primary path is the
// device-resident pair (absorb_root_device / device_drawn).
```

### 16.6 Phase graphs (§6)

```rust
pub struct PhaseGraph { /* graph exec + topology + rebind table */ }
impl PhaseGraph {
    pub fn capture(sched: &Schedule, state: &DeviceProofState, streams: &StreamSet)
        -> Result<Self, GpuError>;
    /// Per-proof pointer/scalar rebind via cudaGraphExecUpdate; falls back to
    /// re-capture on topology-invalidating shape change.
    pub fn rebind(&mut self, params: &ProofParams) -> Result<(), GpuError>;
    pub fn replay(&self, stream: Stream) -> Result<(), GpuError>;
    /// Same schedule, no capture — the migration fallback and debug surface.
    /// Replay and eager must produce byte-identical proofs (manifest A/B).
    pub fn eager(&self, streams: &StreamSet) -> Result<(), GpuError>;
}
```

### 16.7 C ABI conventions (kernels crate boundary)

`extern "C" bool stwo_<area>_<verb>(…)`; pointer + scalar parameters only;
explicit `u64` lengths; `void* stream` LAST; `bool` success with detail via
the last-error query; declarations in `raw.rs` mirrored 1:1 in `stubs.rs`
(generated by `ffi_emit`, §17 — the mirror is correct by construction);
no varargs, no callbacks, no struct-by-value (C12/C13 govern versioning and
linkage).

---

## 17. Codegen tooling — the generate-don't-handwrite policy

**Policy (R8): anything derivable from the AIR, from emitted metadata, or
from a repetitive schema MUST be generated by a tool with a `--check` drift
gate — never hand-written, never hand-mirrored.** The program has already
paid for this rule twice: hand accessor field lists (396/850-word data entry)
were replaced by transformer emission after being identified as "error-prone
data entry the tool should own" (round-24), and the hand-added gate code
inside emitted blocks was silently reverted by a re-emit until the template
absorbed it (round-20). Hand-writing is permitted ONLY for:
(a) bandwidth-critical primitive kernels with no derivable source (NTT,
blake2s, folds, quotients, gathers, pool) — each behind a byte-equality
conformance gate; and
(b) 1:1 transcriptions of a host reference (the fp256/EC deduce device
functions) — frozen once landed, validated by truth-oracle legs, and any new
deduce kind ships WITH its oracle leg.
Litmus test: **if writing the second instance feels like data entry, stop
and build (or extend) the emitter.**

### 17.1 Tool inventory (normative)

| tool | status | input | output | drift/correctness gate |
|---|---|---|---|---|
| **`witness_genericize`** (transformer) | exists | AIR-generated witness writers (`prover/src/witness/components/`) | generic writers (Simd/Recording-polymorphic), `SUB_FEED_LAYOUT`, `JIT_LOOKUP_FIELDS`, igen fns, slot-layout consts, felt/u32/W27 lowerings | `--check` (byte, comment-insensitive) + per-component parity gates vs the original writer on real fixtures |
| **`kernel_emit`** (new, M3) | specified §4 | the recordings (witness programs via the lane on the gate fixtures — sound because recorded programs are straight-line and value-independent, §4; constraint programs via the framework recorder, fixture-free) for every component | `generated/*.cu` (fused, uncapped), `manifest.rs` (KernelKey = semantic hash + codegen version, launch shapes, ABI version), archived `ptxas -v` register/spill reports | `--check` (regenerate + byte-compare) in CI; at prove time the live recording's hash IS the registry key — a miss is drift, fail closed (§4/§16.4) |
| **`schedule_emit`** (new, M2) | NEW | transformer metadata (slot layouts, `SUB_FEED_LAYOUT`) + feed-map parsing (`parse_feed_map` — both feed-loop shapes already parsed, zero UNMAPPED) + the spawn-order dependency scan | `schedule.rs`: the `ComponentNode` table (§16.3) — edges, count relations, slot layouts, log-size sources — plus `COUNT_RELATIONS` registry entries | `Schedule::validate()` in CI (cycles, dangling edges, width mismatches vs emitted metadata) + the existing count/edge byte-gates on fixtures |
| **`ffi_emit`** (new, M2–M3) | NEW | export annotations in the `.cu` sources (or the kernel manifest) — one declaration site per symbol | `raw.rs` + `stubs.rs` (mirror-complete by construction) + typed launch-wrapper/`KernelParams` marshaling code in backend-cuda | `--check` in CI; replaces the hand-mirroring that produced the round-28 anonymous-namespace linkage class of bug; symbol-list CI check becomes a tool invariant |
| **stwo-air-infra** (external, AIR team) | upstream | AIR definitions | the generated witness writers + constraint `Eval`s this whole chain consumes | out of scope; its output is our input — never hand-edited (CLAUDE.md) |

### 17.2 Regeneration discipline

- One command regenerates everything: `make codegen` = witness_genericize →
  kernel_emit → schedule_emit → ffi_emit, in that dependency order; CI runs
  the same chain with `--check` on every PR.
- An AIR revision bump is a **codegen event, not an editing event**: re-run
  the chain, review the generated diff like any code review, re-run the gate
  ladder. No hand-patching of generated output, ever (R5).
- Every generator stamps its output (tool name, tool version, input hashes)
  so provenance is greppable; every generated file carries the
  machine-written marker the `--check` gates key on.
- Generators are themselves gated: unit tests pin the tricky parses
  (declaration-driven sub shapes, sibling-file type aliases, both feed-loop
  shapes — the round-21 lessons), and a generator change that alters emitted
  bytes bumps the relevant codegen version (C13).
