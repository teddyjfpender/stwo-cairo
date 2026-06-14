# Large-PIE / multi-GPU scaling: adversarial architecture review (2026-06-14)

Adversarial, code-grounded review of the three-layer scaling roadmap (L1 spill,
L2 pipeline-parallel, L3 shard) proposed in `~/Desktop/simd_vs_cuda_performance.pptx`
(Feb 2026, RTX 5090, ~1M-step PIE). Method: three independent red-teams, each
told to find why the layer fails and to cite the actual code. Conclusion up
front: **the deck's roadmap does not survive contact with the codebase.** All
three layers are mis-scoped; one (L3) is unsound as written; the real lever is
elsewhere (single-GPU witness offload + independent-proof throughput).

## 0. Why this matters (the VRAM wall)

SN_PIE_2 (7.83M cycles) measured **44.5 GB peak VRAM on an H100, low-mem OFF**.
A 3090 = 24 GB, 4090 = 24 GB, 5090 = 32 GB. So a single full Starknet-block PIE
does **not** fit a consumer card today. The deck proposes L1/L2/L3 to fix this.
The 13.66 MHz round-9 fleet result was **independent proofs across cards**
(throughput), each fitting on one card — NOT one proof split across GPUs.

Where the 44.5 GB goes (verified, `backend-cuda/src/backend/poly.rs:544-547`,
`pcs/mod.rs:489`, `prover.rs:488` store_coeffs=false → only 2×-blown-up evals
kept resident through OODS/FRI/decommit, freed only at `pcs/mod.rs:419-427`):
1. **Preprocessed LDE ~8.2 GB** (the +8224 MB commit jump) — Canonical Pedersen
   (`pedersen.rs:25,58-60`) + Poseidon/bitwise/blake/range-check tables, 2×, all
   resident. **Fixed by the program, independent of cycle count.**
2. Composition + FRI-quotient working set ~4.5 GB (secure 16× cols on top of all
   three resident trace trees).
3. Base-trace LDE ~3.5 GB; interaction/LogUp LDE ~2.0 GB.

## 1. L1 — Memory Spill + Streaming (single GPU). VERDICT: broken as specified.

**Deck claim:** spill committed evals to CPU RAM per group, keep coeffs on GPU,
reload for quotient/decommit → peak ~27-29 GB, "fits a 32 GB 5090, 2 weeks, P0."

**KILLER FLAW (fatal): freeing device memory does not lower reserved VRAM.** The
CUDA mempool is configured to never release: `cudaMemPoolAttrReleaseThreshold =
UINT64_MAX` (`backend-cuda-kernels/cuda/cuda_mem_pool.cu:22-23`); there is **no
`cudaMemPoolTrimTo` anywhere**, and `cuda_mem_pool_destroy()` is a no-op
(`:40-42`). Every device free routes to `cudaFreeAsync` → returns memory only to
the pool's free list, not the OS. Worse, the Rust `BaseColumnPool::give_back`
pushes buffers onto a `Vec` and never drops them (`mempool.rs:59-62`). So a spill
that "frees" device memory releases it into two layers that both hold it forever
— **peak reserved VRAM stays ~44.5 GB and still OOMs**. The design's core action
is a no-op against the OOM ceiling until a `cudaMemPoolTrimTo` is added AND the
Rust pool is taught to drop.

**The existing `set_low_memory` is NOT this design.** It is on-device coefficient
halving (`compact_tree_columns`, `pcs/mod.rs:615-648`): after FRI it interpolates
each column, keeps a half-size coeff column if the upper half is zero, re-evals
transiently at decommit (`:653-704`). It never moves a column to host RAM; there
is no spill/streaming machinery (env grep: only `STWO_CAIRO_LOW_MEMORY`,
`STWO_BENCH_LOW_MEMORY`, `STWO_CUDA_DISABLE_STREAMS`, `STWO_N_POOL_STREAMS`).
Default OFF (`pcs/mod.rs:53,68`); the 44.5 GB was measured with it OFF.

**Other findings:** (a) no peak is ever measured — `backend-cuda/README.md:70-71`
admits the probe reports end-of-run pool footprint, not the in-flight peak; the
one real number is −14% on a 16-col non-Cairo AIR. (b) The compaction already
pays a hidden blocking full-D2H per column (`pcs/mod.rs:636` `to_cpu().all()`);
the backend has **no async download lane** (README:95-104), so any spill/reload
serializes against compute on the legacy stream — seconds of added wall at 44 GB
over PCIe. (c) Proof-identity is tested only on a 64-row SIMD Debug-string match
(`examples/.../wide_fibonacci/mod.rs:191-254`), not PIE-scale CUDA.

**Verdict:** even in the impossible best case, coefficient-halving cannot bring
44.5 GB under 24 GB — the +8.2 GB preprocessed peak is built *before* compaction
and is program-fixed. **L1 does not help a 24 GB 3090; it is at most a 32 GB-card
story, and only after the mempool-trim flaw is fixed.** "2 weeks/P0" is
unrealistic: it requires (1) `cudaMemPoolTrimTo` + Rust-pool drop, (2) a real
host-spill path, (3) an async download lane, (4) spilling *during* commit
build-up (not just after FRI), (5) a PIE-scale CUDA byte-identity gate.

## 2. L2 — Pipeline Parallelism (2-4 GPUs). VERDICT: kill it; throughput already solved.

**Deck claim:** multi-process, 1 GPU/process, parallel stages (trace/Merkle/
quotient/PoW), data via CPU RAM → 1.5× @2GPU, 2× @4GPU, "2 weeks after L1, P1."

**The prove is a strict Fiat-Shamir serial chain — stages of one proof CANNOT
overlap.** Verified `prover.rs:276-398` + `prover/mod.rs:54,79,83` +
`pcs/mod.rs:326-378` + `fri.rs:140-234`: base-commit → mix_root → interaction
elements drawn from channel (`prover.rs:341-343`) → interaction-commit →
random_coeff drawn → composition-commit → OODS point drawn → quotient coeff drawn
→ per-layer FRI draw/fold/commit → PoW → **query positions drawn after PoW**
→ decommit. Every stage consumes channel state from the prior commit; the channel
is a single `&mut` serial transcript (`core/channel/mod.rs:21-47`, not even
`Serialize` — no IPC path). Overlapping stages would require modifying
Fiat-Shamir, which the priority contract forbids.

**On the real workload L2 touches almost nothing.** Measured SN_PIE_2 (H100,
`round13b_snpie2_spans.log`): **Write Base trace 38.6 s** (HOST CPU, single-process
rayon `cairo_claim_generator.rs:794-1041`) + interaction 3.3 s + run/adapt ~7.4 s
≈ 49 s host; the entire GPU commit+prove tail is ~2 s. **GPU idle ~92%+.** L2
parallelizes GPU stages → speeds up a ~2 s tail. Amdahl with p≈0.04 →
**ceiling ~1.04× for ANY number of GPUs.** Plus GB-scale cross-process tree
transfers (witness cells [543M, 2069M, 1629M] = >10 GB) over PCIe+CPU-RAM likely
make single-proof L2 a **net regression**.

**Throughput is already solved, simpler and better:** independent proofs per GPU
(the fleet, `fleet_m*.log`), MPS co-residency (`mps_3090.log`, ~1.3×), and the
shipped single-GPU host/GPU overlap `gpu_bench.rs --pipeline` (`:191-240`, "pure
orchestration, zero soundness surface"). N independent proves give ~N× throughput
with no peer copies, no channel IPC, no new code. **L2's stage-splitting is
strictly dominated.** The only multi-GPU work worth funding is data-parallel
**witness generation** (sharding the 38.6 s host write across devices) — a
different design (parallelize the dominant stage, not the trivial tail).

## 3. L3 — Shard Partitioning. VERDICT: unsound as written; multi-quarter research.

**Deck claim:** split execution into N **independent** shards, each a full STARK
proof, "near-linear scaling," recursive merge via Cairo verifier, "8-12 weeks, P2."

**SOUNDNESS LANDMINE.** Cairo memory correctness is ONE global scalar check:
`lookup_sum(...) != 0 → reject` (`cairo-air/src/verifier.rs:340`), summing
`claimed_sum` over every component incl. the three memory components
(`claims.rs:1336-1644`, memory at 1559-1570). The memory table is a single dense
address→value vector over the whole relocated run (`adapter/src/memory.rs:84,133`);
multiplicities are global (`memory_id_to_big.rs:150-156,476`). **Cut at step k:
shard B reads address x written in shard A; B's local table has no yield for x, so
to make B's `lookup_sum==0` the prover FABRICATES x's value. Two independently
"valid" shard proofs compose to an execution that never happened.** Total break of
the memory argument — and the same holds for every global lookup (range checks,
xor/point tables, builtin aggregators).

**No boundary mechanism exists.** The only carry today is CPU registers
(`PublicData{initial_state, final_state}`, `air.rs:89-94,129-147`) — pc/ap/fp
chaining is expressible, but **memory has no carry**: no memory-delta, no
carry-in/out digest, no multiplicity reconciliation (`get_entries` `air.rs:488-525`
exposes only code/output/builtin ranges). No continuation/segment model:
"segment" = builtin memory segment, not execution continuation; `ProverInput`
holds the whole run (`adapter/src/lib.rs:28-39`, no carry fields); the bootloader
packs many tasks into ONE proof (`bootloader/src/lib.rs`) — the opposite of
sharding. The Cairo verifier is single-proof (`cairo_verifier/src/lib.cairo:4`
`main(proof: CairoProof)`) — no aggregation, no reconciliation.

**Verdict:** the hardest unsolved piece is a **sound cross-shard memory + lookup
boundary argument** (committed carry-in/out digest + a reconciliation proof that
N partial LOGUP balances compose to one globally-consistent table) — a new
soundness-critical protocol, not a refactor, and it must cover every global
lookup. The 8-12 week figure covers only the mechanical parts (run the existing
verifier N times, chain registers). The load-bearing piece is **multi-quarter
research with a hard soundness-proof obligation.** "Independent shards" is
outright wrong; scaling is sublinear (bounded by reconciliation + aggregation,
which grow with N and boundary size).

## 4. The real picture + recommended sequencing

The deck inverted the priorities. The dominant cost is the **host witness write
(38.6 s)**, not GPU memory or GPU stage scheduling. Ranked, code-grounded levers:

1. **[P0, in flight — task #36] Single-GPU witness offload (whale ports).** Cuts
   the 38.6 s directly; arch-agnostic; soundness-gated by the differential +
   byte-equality. assert_eq_imm done; cohort next. THIS is "as fast as possible."
2. **[P0] Independent-proofs-per-GPU throughput.** Already works (fleet + MPS +
   `--pipeline`). This is the real "multi-GPU" / cost-per-MHz answer for fitting
   workloads. No new code; document + harden it as the deployment model.
3. **[P1, prerequisite for any VRAM work] mempool-trim + a true high-water-mark
   VRAM probe.** Until `cudaMemPoolTrimTo` + Rust-pool drop land and a peak probe
   exists, NO spill/low-mem change can reduce the OOM ceiling or even be measured.
   Small, foundational, unblocks L1.
4. **[P2] Data-parallel witness generation across GPUs** (the sound replacement
   for L2's intent): shard the host `write_trace` rayon work across devices, each
   producing a slice of the SAME proof's trace, uploaded to one prove. Parallelizes
   the actual bottleneck; no Fiat-Shamir change.
5. **[P2, only for 32 GB+] L1-proper** (host-spill + async download lane +
   spill-during-commit), gated on #3. Does NOT help 24 GB 3090s.
6. **[P3, research] L3 cross-shard memory-boundary soundness design.** The only
   path to a single huge PIE on a 24 GB card, but it is a multi-quarter,
   soundness-critical protocol design — schedule as research, not engineering.

**Bottom line for the 3090 target:** a single 7.83M-cycle SN-PIE does NOT fit a
24 GB 3090 and cannot be made to (even a fixed L1 is a 32 GB story; the ~8 GB
preprocessed tables are program-fixed). The 3090 fleet's value is **throughput of
fitting workloads** (#2) + **faster per-proof via witness offload** (#1). Proving
a single large block on 3090s requires L3 sharding (#6, research) or proving
smaller units. This should reset expectations vs. the deck's "L1 fixes it" framing.

(Folded into the next pod round: a `STWO_CAIRO_LOW_MEMORY=1` SN_PIE_2 VRAM
measurement + an nvidia-smi peak read — to empirically confirm the mempool-trim
finding, i.e. whether low-mem moves the OS-reserved peak at all.)
