# The endgame GPU architecture: what the hardware actually permits on SN PIEs

*Round-9 architecture verdict. Grounded in measured round-8/9 data (SN_PIE_2 =
7.71M useful steps, 96-bit config, A40 warm 30.6s = 0.252 useful MHz; splits:
witness 20.1s / commits 10.4s / core 3.3s). This document is the target picture
every session steers toward; ROAD_TO_10MHZ remains the program inventory.*

## 1. The bandwidth floor — how far away we actually are

The STARK prover is a streaming computation over the trace. For SN_PIE_2 the
committed working set (base + interaction + preprocessed, pre-LDE) is ~8–10GB
(consistent with the 36.4GB pool peak once LDE ×2, quotient/FRI working set and
Merkle layers are added). The fundamental pipeline needs ~6–10 full DRAM passes
over that set (witness write, iNTT, NTT@2x, hash read, composition read,
quotient+fold, decommit gathers):

    ~60–150GB of DRAM traffic per proof
    A40   (696 GB/s): 0.10–0.22s   → 35–75 useful MHz
    4090  (1008 GB/s): 0.06–0.15s  → 50–120 useful MHz

**The hardware floor for this PIE on one consumer GPU is tens of MHz.** Today's
30.6s sits 150–300× above it. Even NitrooZK's published pace (~7s equivalent at
n_queries=3) is ~50× above. Nobody is near the wall; the gap is architecture,
not silicon. Decomposition of our 30.6s against the floor:

| slice | today | floor-ish | gap source |
|---|---|---|---|
| witness 20.1s | host CPU (7 cores here) | ~0.05s (10GB write) | not GPU work at all yet |
| commits 10.4s | ~37% of A40 peak BW | ~0.3s | pass count (unfused NTT stages, pack, per-layer Merkle, twiddle loads), zero overlap |
| core 3.3s | JIT composition already 0.61s | ~0.3s | OODS/extension/residue launches |

## 2. The keystone: device-resident execution tables

Workstream A's blocker analysis identified the single dependency that gates ALL
witness work: the generated writers call `deduce_output` (host HashMap) and are
monomorphic host code. The fix is ROAD P1 items 1–3, promoted to keystone:

**Upload the adapter's dedup'd memory + instruction tables + packed
StateTransitions once per proof (~1–3GB, pinned, async), and make
`deduce_output` a device table read.** After that:
- every opcode/table component's witness is a per-row device kernel (JIT'd for
  the M31/u32 tail; hand-written fp256 for the pedersen EC family — the JIT ISA
  should stay 32-bit);
- multiplicities are device atomics (pattern already proven);
- interaction columns already finalize on device (W1);
- the ONLY per-proof PCIe traffic left is the table upload (overlappable with
  the previous proof's tail at P5 depth 2) — witness cost collapses from 20.1s
  of host loops to ~0.1s of GPU streaming + upload overlap.

## 3. The commit path at the pass-count limit

10.4s → ~0.3s requires cutting PASSES, not tuning kernels:
1. **Stage-fused NTT**: radix-8/16 butterflies in shared memory/registers — a
   log-23 column in 2 passes instead of one pass per stage-pair.
2. **Twiddle regeneration in-kernel**: M31 twiddles are shift/add-cheap;
   loading them wastes the scarcest resource (bandwidth) to save the most
   abundant (integer ALU).
3. **Hash-from-registers Merkle leaves**: the final NTT stage keeps its output
   in registers/shared and feeds blake2s leaf compression directly — the pack
   pass and one full LDE read-write round trip disappear. Layer-pair hashing
   halves the remaining tree passes.
4. Commit ORDER is Fiat–Shamir-fixed; between-tree and within-tree column
   groups are not — streams overlap them (P3, currently disabled).

## 4. Quotient/FRI fusion is the VRAM diet is the fleet unlock

Fusing quotient-combine into the first FRI fold (plus the OODS weight-cache
cap) means the full secure-field quotient LDE is never materialized. That is
simultaneously: a speed item (one fewer giant pass), the reason SN_PIE_2
(36.4GB today) fits a 24GB 4090 (projected ~12–16GB), and the reason
14M-step blocks (OOM at 46GB today) fit 32–48GB cards. Fleet scheduling then
routes blocks by size: 7–8M blocks → 24GB cards, 14M blocks → 32/48GB cards,
or aggregate-mode batches sized to the card.

## 5. Orchestration: CUDA graphs, then streams

The prove issues ~10k launches; at 1–2s total prove time launch/latency tax
becomes a first-order term. The FRI fold chain and the witness kernel storm are
static per statement-shape → capture once as a CUDA graph, replay per proof
(near-zero launch overhead, and graph capture enforces the determinism the
byte-equality gate wants). Streams (P3) then overlap witness-upload/commit/
interaction phases that Fiat–Shamir does not order.

## 6. The final boss is the CPU again: the VM feed

At 5–8 MHz per GPU, cairo-vm (measured ~4.3M steps/s/core) + adapter needs
~2–4 dedicated host cores per GPU just to keep it fed; P5 multi-producer
already implements this and `feed_starved_s` measures it. Fleet procurement
rule: cores-per-GPU is a first-class spec (24+ vCPU community 4090s beat
7-core secure A40s twice over). At 10+ MHz/GPU nodes, VM execution — not
proving — becomes the aggregate bottleneck; the fleet answer is more nodes,
not bigger GPUs, which is exactly the $/MHz-hr thesis.

## 7. Budget to the goal (4090-class, 96-bit, SN_PIE_2)

| phase | today (A40) | endgame | via |
|---|---|---|---|
| witness | 20.1s | ~0.15s | §2 keystone + JIT tail + fp256 hand kernels |
| commits | 10.4s | ~0.3s | §3 fusion |
| core (comp/OODS/FRI) | 3.3s | ~0.4s | quotient-fold fusion, batched OODS, graphs |
| orchestration/residue | (hidden) | ~0.15s | graphs + streams |
| **total** | **30.6s / 0.25 MHz** | **~1.0–1.5s / 5–8 MHz** | |

Fleet: **2–3 such nodes = 10–20 MHz aggregate** at ~$0.35/hr each →
**~$0.10–0.20 per MHz-hour**, an order of magnitude under big-iron. The goal
does not require exotic hardware or heroic single-GPU numbers — it requires
finishing §2–§5 behind the existing gate ladder.

## 8. Explicitly rejected

- **Multi-GPU single-proof** (deck L2/L3): throughput comes from block
  parallelism; intra-proof distribution buys latency we don't need at 3×
  complexity and new soundness surface.
- **Persistent megakernels**: CUDA graphs give the launch-overhead win without
  abandoning the per-kernel differential gates.
- **FP/tensor-core NTT**: M31 integer NTT is the right tool; tensor cores
  don't map to 31-bit modular butterflies at these sizes.
- **Raising the JIT kernel cap** on new CUDA versions without re-measuring the
  ptxas cliff (KNOWN_ISSUES 2).

## §4 CORRECTION (round-9, measured): the VRAM driver is committed-LDE retention

The A40 track disproved §4 as written for SN PIEs: the true SN_PIE_2 peak is
**~47.6GB** (the 25ms harness sampler under-reported 36.4 — add a pool
high-water probe), the quotient LDE is only ~0.1–0.3GB of it, and the OODS
cache is not the driver (unlike fib). The diet is therefore **committed-LDE
spill/regenerate** (the STWO_CAIRO_LOW_MEMORY machinery, coverage extended) —
validation pending. Quotient-fold fusion is DEFERRED: fusing soundly requires
regenerating first-layer queried values at FRI decommit (fri.rs:446 is a third
LDE consumer the fusion seam missed) — a soundness-critical change that needs
its own reviewed design, and finding A removed its VRAM justification.

## §4 FINAL (round-9, per-phase pool ladder): the diet design, measured

The 35.2GB decomposes exactly (pool_gb_at_close instrumentation, SN_PIE_2, A40):
witness evals 9.6 → +preprocessed commit 7.6 → +base commit 11.1 → +interaction
commit 6.8 = 35.2GB; the ENTIRE STARK core then adds +0.0 (runs in reserved pool).
Existing low-memory compaction fires only AFTER FRI quotients (pcs/mod.rs:285) —
after the peak — hence its measured zero effect at 2.2x cost. The real fix:
compact each tree immediately post-commit; regenerate LDE per CONSUMER —
composition per component group (its JIT kernels already read per-component),
quotients/OODS per query batch, decommit per position (machinery exists).
Projected peak ~19-20GB (fits 24GB 4090 whose 64-core host then projects
SN_PIE_2 at ~10-12s ≈ 0.7 useful MHz before any new witness kernels). Cost ≈ one
extra NTT pass. Byte-equality-gated, soundness-supervised (commitment scheme).

## 9. Prove-lane readiness audit (round-10 review) + the collapse math

A hostile re-review of the witness-JIT prove integration found and fixed TWO
proof-breaking bugs before any GPU hour was spent, plus the hazards around them.
Both would have surfaced remotely as "verify failed", each costing a
provision+build+run cycle to even localize:

1. **Sub-input feed truncated at `n_real`.** The host writer feeds the FULL padded
   extent downstream (`add_inputs` over `len*N_LANES`; `mults_0 = 1` on every row —
   only `mults_1` is the enabler). Feeding fewer rows than the interaction trace
   emits uses for unbalances logup: claimed_sum ≠ 0, invalid proof.
2. **Flat-layout mismatch.** The accessors read word-major; the codegen stored
   row-major. Fixed by flipping codegen to word-major — which is also the RIGHT
   layout: coalesced stores (adjacent threads → adjacent addresses) and each
   16-lane PackedM31 repack is one contiguous 64B run. `WITNESS_CODEGEN_VERSION`
   bumped 1→2 so no pod's disk PTX cache can serve stale kernels.

Hardening landed with them: the hardware selftest now byte-checks the FULL data
contract (committed columns + lookup words + sub words — the latter two were
previously dropped unread); the prove launch fails CLOSED on unresolvable pcs,
mult-table programs, >4-input programs, and >2048-instr programs (the NVRTC/ptxas
cliff governor, so a future big component can never stall a prove — it falls back
and logs); the lane validates the recording's shape (cols/lookup/sub/poison)
against per-component expectations before launching; every fallback logs its
reason (no more silently-idle GPUs); the full address table is no longer copied
per component (deduped pc→id pairs instead — was ~0.8GB of host churn per
component at SN scale); trace columns are consumed, not device-copied.

**The keystone addition: a LOCAL parity gate** (`witness_eval::differential_test::
add_opcode_prove_accessors_match_host`). It replays the exact prove-path data flow
— interpret every PADDED row with real enabler semantics → word-major flats (the
launch's D2H format) → the production accessors → byte-compare `LookupData` and
all sub-input feeds against the host writer. Combined with the hardware selftest
(kernel ≡ interpreter), GPU risk collapses to launch mechanics only. Fixture
verified to exercise 24 real padding rows (asserted, so a fixture swap can't
silently weaken the gate).

### Can this reach 10s of MHz? The staged collapse, grounded in measurements

SN_PIE_2 (7.71M useful steps): A40 33s / 0.23 useful MHz → H100 14.26s / 0.54.
Phase split (A40 basis): host witness writes 66%, commits 29%, core 8.7%; on H100
commits shrank only 1.9× on 4.8× bandwidth (launch-bound), and today's 3090
small-PIE profile shows commits at 88% of prove — the launch tax in its purest
form. The §1 bandwidth floor says tens of MHz per card is what the silicon
permits (H100 ~30–50, 4090 ~10–15 ideal). The gap is architectural, not silicon:

| stage | what | expected SN_PIE_2 (H100) | useful MHz |
|---|---|---|---|
| today | host witness + launch-bound commits | 14.26 s | 0.54 |
| W-pilot | add_opcode JIT lane (this round; wt: ~2s → ~0.2s once measured) | ~12.5 s | ~0.62 |
| W-cohort | LaneSpec engine × every transformed CasmState opcode (same 40-line impl each; NO new kernels) | ~8–9 s | ~0.9 |
| §3+§5 | fused-stage NTT, hash-from-registers Merkle, CUDA graphs (commits 3.5–4 s → ~1 s) | ~5–6 s | ~1.4 |
| W-heavy + §6a | fp256 cohort (ISA-V2: inverse/eq; mult-table launch support) + interaction trace ON DEVICE (logup from the already-device-resident lookup words — today they round-trip D2H only because interaction is host SIMD) | ~2.5–3 s | ~2.7 |
| fleet | N cards × independent PIEs (embarrassingly parallel; the NORTH STAR is aggregate $/MHz-hr) | 8×4090 ≈ 12–20 MHz agg | 10–20 ✓ |
| endgame | full device residency at the §1 floor | ~0.5–0.8 s/card | 10–15/card |

The honest statement: **single-card 10 MHz = the full endgame architecture; the
10–20 MHz NORTH STAR arrives earlier as a fleet aggregate** once per-card ~1.5–2.5
MHz lands (W-cohort + §3/§5), at ~$0.35–0.55/hr per 4090. Nothing in the current
integration blocks any later stage: the seam hands device-resident columns to the
committed tree, keeps lookup words on device until the (still-host) interaction
consumes them, and the per-component cost of joining the lane is a LaneSpec + two
accessors — the marginal cost of a component is now ~40 lines and zero CUDA.

### Next hardware session = one manifest, in order
1. stwo+stwo-cairo test sweep on pod (37+8 incl. parity gate), then selftest
   (`STWO_WITNESS_JIT_SOURCE=emitted`) — now checks cols+lookup+sub on hardware.
2. SN_PIE_2 lane OFF baseline (record prefix-hash), then lane ON
   (`STWO_CUDA_WITNESS_JIT_PROVE=1`): prefix-hash identity, verify, repeated-prove,
   `wt:add_opcode` span delta + `jit_prove` log lines.
3. If green: flip assert_eq/jnz_taken through the engine next session; begin §3/§5.

## 9b. ROUND-10 RESULT (2026-07-03): the witness lane is PRODUCTION-GREEN on hardware

Full gate ladder PASS (run 20260703T121708Z, H100 NVL, 16 vCPU host, all six
machine-checked gates): truth-oracle selftest (kernel vs REAL memory semantics —
151.9M values on SN_PIE_2's own states, 0 mismatches), SN_PIE_2 lane OFF baseline
(warm 16.01s / 0.481 useful MHz / proof 3,006.636 KB — the SIMD-proven size),
lane ON with **PREFIX_MATCH** (proofs byte-identical through 2.29MB of 3.08MB;
divergence only in the KNOWN_ISSUES-4 tail-reorder region), verify green, 3-rep
stability, span accounting. `wt:add_opcode` lane-ON = **980ms total** (device
kernel ~340-400ms for 651,967 rows + D2H of 545MB flats + host igen/feeds) vs
~1.5-2.5s host writer — and the D2H+repack majority of that 980ms is exactly the
part the future device-interaction lane deletes.

Two more production bugs were found and fixed ON THE WAY to green, both invisible
to the previous gates:
1. **The kernel deduced dst/op0/op1 through the decode-only flat LUT** (pc →
   instruction limbs) — full-width recordings deduce over the WHOLE memory. The
   in-process SHADOW DIFF (`STWO_JIT_PROVE_SHADOW=1`, byte-diff vs the host writer
   with coordinates) localized it in one 40s run: trace col 14 host 0x4000000E
   (encoded id) vs device 0. Fix = the §2 keystone finishing its job: codegen
   TableLimb now implements the REAL table semantics (t0 = raw id at address;
   t1 = tag-dispatched value limbs over 28 big + 8 small columns, mirroring the
   proven exec_deduce_output kernel), launches pass DeviceExecutionTables
   (37-pointer ABI + clamp lengths), one upload per memory (process-cached),
   WITNESS_CODEGEN_VERSION 2→3.
2. **The hardware selftest's per-component legs compared kernel vs interpreter
   under the SAME incomplete flat model** — consistent-but-wrong passed for two
   sessions. Legs now interpret against the truth oracle (real tables, host
   deduce semantics), so kernel==interp means kernel==host-writer. Rule recorded:
   a differential gate must include a surface whose reference does NOT share the
   implementation's model.

Also calibrated on sm_90: the driver JIT's module_load hits 240-290s/kernel on
2048-instr fused constraint kernels (sm_86-calibrated cap); STWO_JIT_MAX_KERNEL_INSTRS=512
restores ~45-55ms loads (5000x). In the ladder env permanently.

NEXT: stamp assert_eq/jnz_taken through the LaneSpec engine (accessors + ~40
lines each), extend the transformer cohort, then §3/§5 per the priority
directive — with gpufleet manifests as the only way GPU sessions run.

## 10. ROUND-11 (2026-07-03, all-local engineering sprint): the witness block dissolved

Everything below compiled + locally gated (stwo 40/40, stwo-cairo 75/75 incl. 33 new
cohort gates) with ZERO GPU minutes — the next pod session is pure validation.

1. **Cohort: 14 components through the prove lane** (was 1). The per-component trait
   became ONE backend trait (`OpcodeJitBackend`) over per-component specs
   (`OpcodeLaneSpec` via `opcode_lane_spec!`); accessors are macro-generated from a
   field-width list (`jit_lookup_accessor!`/`jit_sub_accessors!`). New components =
   data entry. Stamped: add, assert_eq, jnz_taken, add_small, assert_eq_imm,
   assert_eq_dd, call_abs, call_rel_imm, jnz_non_taken, jump_abs/dd/rel/rel_imm,
   ret. Each carries gate (a) byte-identity, gate (b) recording/interpreter, and the
   prove-accessor parity gate. Deferred: mul_small + qm31_add_mul (range-check
   feeds → 4th state), mul + add_ap (u32 trait extension) — censused.
2. **ISA-V2**: `M31Inverse` (P-2 square-and-multiply, schedule == CPU, inverse(0)=0,
   unit-pinned) + `M31Eq` (0/1); mask combinators LOWER to field ops on 0/1
   registers (and=mul, as/from_m31=identity, select=m·a+(1-m)·b) — jnz_taken went
   from 4 poisoned columns to FULLY recorded, byte-verified vs host. CODEGEN v4.
   Per-component kill switches: STWO_CUDA_WITNESS_JIT_PROVE_<COMP>=0.
3. **§3 audit**: the fused-commit lanes (NttHash, LayerPair, TwiddleRegen) were
   ALREADY BUILT behind STWO_CUDA_FUSED_COMMIT (default OFF) and never validated;
   NTTs are already column-batched. Manifest step 6 (sn2-fused-commit) now A/Bs
   them with prefix-identity + verify. §5 remainder (graphs, streams, staging
   batch) ranks AFTER that data.
4. **§6a engine**: `logup_pairs.cu` computes combine + fraction pairs for ALL logup
   columns in ONE launch straight from the word-major flats (coalesced), feeding
   the proven device finalize (dense chain, claimed sum, cumsum) —
   `device_interaction_from_flats` returns device-resident interaction columns.
   The descriptor rule (sequential pairs; opcodes*→mults_1; trailing solo negated)
   is unit-pinned against the generated writers. Remaining: keep the lane's device
   lookup buffer (drop the D2H), pod differential, prove-path branch.

Bandwidth-floor arithmetic after this round validates: witness (66% host block) →
device for the entire CasmState opcode family; interaction write+finalize → device;
commits → fused lanes. The measured phases then reduce to NTT+hash passes over
committed bytes — the regime where the §1 floor (tens of MHz/card) is the binding
constraint, which is exactly the claim the next benchmark round must prove or refute
with the phase ledger.
