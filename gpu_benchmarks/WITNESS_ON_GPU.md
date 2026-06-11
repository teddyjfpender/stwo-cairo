# Witness-born-on-GPU: analysis & design

*Status: design + W1 lane implementation (stwo fork). The phase-trace round (RESULTS.md
round 4) left the warm CUDA prove witness-bound: the STARK core is 813 ms at fib 1M
while witness phases and host plumbing hold the other ~4.5 s. This document is the
code-level analysis of where those seconds live and the staged design for moving the
witness onto the device — the step that goes beyond NitrooZK (their base/interaction
witness is also host-generated).*

## 1. Pipeline anatomy and measured cost (fib 1M warm, RTX 3090, round-4 build)

| phase | time | where | what it does |
|---|---|---|---|
| cairo run | 1.34 s | host (cairo-vm) | VM execution, trace+memory output |
| adapt | 3.0 s | host | VM output → ProverInput: memory builder, relocation, opcode classification (`MemoryBuilder::from_iter` historically the bulk) |
| Write Base trace | ~1.4 s | host SIMD | 67 generated per-component writers: decompositions, flags, memory tuples → PackedM31 columns + `lookup_data` extraction |
| base trace upload + commit | 0.58 s | PCIe + GPU | `from_simd_evals` (pinned, batched) + NTT + Merkle |
| Write interaction trace | ~1.2 s | host SIMD | per-component logup: `combine()` per row, batch inverse, fraction chain, prefix-sum |
| interaction upload + commit | 0.40 s | PCIe + GPU | same as base |
| Prove STARKs | 0.81 s | GPU | composition (JIT), OODS, FRI, decommit — solved |

Note `cairo run` + `adapt` (4.3 s) sit *outside* `prove_cairo`; they are user-visible
wall clock but not part of the prove span. Everything below targets the 3.6 s of
witness work inside the prove plus the 4.3 s outside it.

## 2. The interaction trace, dissected (W1 — the implemented lane)

Every one of the 67 generated `write_interaction_trace` functions follows one rigid
pattern (`add_opcode.rs` is representative):

```text
for each logup column (pair-batched):
    parallel for each packed row:
        denom_k = lookup_elements.combine(lookup_data.X_k[row])   // z - Σ αⁱ·vᵢ
        write_frac(numerator(denoms, mults), denom0 * denom1)
    finalize_col():            // HOST: batch-inverse denoms, num·inv, += prev column
finalize_last():               // HOST: claimed_sum, shift, 4× inclusive prefix sum
```

### Where the host time goes
- **write loops**: `combine()` dot products + numerator algebra — embarrassingly
  parallel rayon/SIMD, reads the fat `lookup_data` arrays.
- **finalize_col**: `batch_inverse_packed_qm31` (~6 muls/elem amortized) + multiply +
  chain-add — pure column math.
- **finalize_last**: column sum + shift + four M31 inclusive prefix sums.

### The traffic theorem (why the split is what it is)
Moving the *write loops* to the GPU requires uploading `lookup_data`, which is wider
than the output (add_opcode: ~60+ words/packed-row of lookup tuples vs 20 words of
interaction columns). Uploading inputs to save host math **loses on PCIe**. Moving
only the *finalize* math requires uploading (numerator, denominator) pairs:
8 words/row vs the 4 words/row the finalized column costs via `from_simd_evals` —
but the finalized device columns then need **no** `from_simd` upload at all. Net
PCIe: **8 vs 4+4 — traffic-neutral**, while all finalize math (inversion chains,
fraction chain, prefix sums, claimed sums) moves to the device and the host keeps
only the cheap write loops.

A registration-hook variant that downloads device results back into SimdBackend
columns (to avoid touching generated code) was rejected: upload 8 + download 4 +
re-upload 4 = 16 words/row, ~3× traffic — the PCIe cost eats the host saving.

### W1 architecture: deferred finalize
1. **`RawLogupTraceGenerator`** (stwo fork, `constraint-framework/src/prover/
   logup_raw.rs`): same call shape as `LogupTraceGenerator` (`new_col` /
   `write_frac` / `finalize_col` / `finalize_last`), but finalize stores the raw
   (numerator coords, denominator) columns instead of computing. `finalize_on_simd()`
   replays the raws through the real `LogupTraceGenerator` — **identical output by
   construction** (it is the same code path), unit-gated.
2. **`CudaBackend::finalize_raw_logup`** (backend-cuda): per column, upload the
   8 words/row of raw pairs (pinned staging); on device: existing
   `batch_inverse_secure_field` → new fused `num·inv + prev_column` kernel → for the
   last column: sum-reduction (claimed_sum, 16-byte readback), broadcast-shift
   kernel, and four `inclusive_prefix_sum` calls (the CUB lane staged from NitrooZK —
   FFI wired and conformance-gated now; its M31 operator+ under `cub::InclusiveSum`
   must byte-match the SIMD scan, which the unit gate proves). Returns
   `CircleEvaluation<CudaBackend>` columns directly — born on device.
3. **Byte-equality argument**: field adds are associative/commutative (any reduction
   or scan order yields the same element — the eager code already relies on this for
   its parallel sum); inverses are unique; the per-column chain order is preserved.
   The gates: a raw-vs-eager unit differential (host), a CUDA-vs-SIMD finalize
   differential (pod), and the Cairo e2e proof byte-equality after integration.

### Integration (staged, mechanical — the next session)
The 67 generated writers + the master aggregator (`witness/cairo.rs`) + `prover.rs`:
- writers: `LogupTraceGenerator` → `RawLogupTraceGenerator`; return the raw trace
  instead of `(evals, InteractionClaim)`. **Ordering constraint**: `claimed_sum` is
  only known after finalize, and `interaction_claim.mix_into(channel)` must happen
  *before* the interaction commit — so `prover.rs` finalizes (per backend) right
  after `write_interaction_trace`, then builds the claims from the returned sums in
  component order, then mixes and commits. Fiat-Shamir order is unchanged.
- `B::finalize_raw_logup` becomes a backend trait hook (FromSimdColumns-style):
  SimdBackend = `finalize_on_simd` (today's bytes), CudaBackend = device lane.
- ~66 files change with one sed-able pattern; the aggregate claim struct keeps its
  shape (sums arrive in the same order).

Expected win: ~0.5–0.7 s of host finalize at fib 1M, the interaction `from_simd`
upload disappears (overlapped by the pair uploads happening per column as the host
writes the *next* column), and `Write interaction trace` drops to the write loops.

## 3. Base trace on GPU (W3 — the XL frontier)

The 67 base writers (`write_trace_simd`) are ~86K LoC of *generated* SIMD code:
per-row pure functions from adapted VM data (instruction decode fields, memory
tuples, range-check decompositions) into N_TRACE_COLUMNS PackedM31 columns +
`lookup_data` side arrays. Three viable paths, in increasing order of effort:

1. **Stream the upload (W2, S/M)**: generation stays host but columns upload as each
   component finishes (today: generate-all-then-upload-all). Hides most of the
   ~0.3–0.5 s upload behind generation. Needs a streaming seam in the claim
   generator; no math changes.
2. **Codegen-on-GPU (XL, the real thing)**: the writers are generated by stwo-cairo's
   AIR codegen from the same component IR that emits constraint evaluators. The JIT
   constraint lane already proves the recipe: record/emit per-component CUDA from the
   IR with an explicit C ABI, gate per component with a differential harness
   (`STWO_CUDA_CONSTRAINT_VERIFY` pattern), fall back to SIMD per component. Inputs
   (adapted VM data) upload once; columns are born on device; `lookup_data` never
   exists on host (the interaction write loops would also move, changing the W1
   traffic calculus in GPU's favor). This obsoletes W1's host write loops — but W1's
   device finalize is a strict prerequisite component of it, so nothing is wasted.
3. **Hybrid by component family**: memory/range-check components (regular, tuple-
   shaped) first; opcode components (decode-heavy) later. The per-component fallback
   makes this incrementally shippable.

## 4. Host plumbing (W4 — outside the prove span but 45% of wall clock)

`adapt` (3.0 s) is hashmap-building and relocation (`MemoryBuilder::from_iter`,
`get_relocated_memory` 0.35 s, `relocate_trace`). No soundness surface (gated by the
e2e proof). Levers: rayon the memory builder's per-segment passes, replace the
id-dedup hashmap with a sorted/sharded build, and overlap `adapt` with `cairo run`
output streaming. `cairo run` itself (1.34 s) is the VM — out of scope (a GPU VM is
a research project, not an optimization).

## 5. Order of attack and expected end-state (fib 1M warm)

| step | effort | saves | running total (prove span) |
|---|---|---|---|
| today | — | — | 5.37 s |
| W1 device logup finalize | M (lane done; integration staged) | ~0.5–0.7 s | ~4.7 s |
| W2 streamed base upload | S/M | ~0.3–0.5 s | ~4.3 s |
| W4 adapt parallelization | M (wall clock, not prove span) | ~1.5–2 s wall | — |
| W3 codegen witness on GPU | XL | ~1.5–2 s (both write phases) | **~2.5 s** |

End-state: a ~2.5 s fib-1M prove (≈3 MHz) on a 3090-class card, witness-born-on-GPU,
with the host reduced to the VM, the adapter, and Fiat-Shamir orchestration.

## W3 phase-1 blueprint: memory_id_to_big, the full vertical slice

Chosen first because it is (a) among the largest single components, (b) in the
sequential deduction phase where W2's streaming cannot overlap it — its generation
AND upload sit on the critical path, and (c) a pure function of the adapter's
dedup'd tables (no cross-component mutation on its inputs).

The coupling that defines the cut: if the base limb columns are device-born, the
interaction writer must not read them back — so the component's denominators move
to device with them. The slice:

1. **Inputs up once**: the dedup'd f252 value table (8 u32/value — half the bytes of
   the 28 limb output columns) and the multiplicity counts.
2. **Limb-split kernel**: f252 words -> 9-bit limb columns (the `gen_big_memory_traces`
   body), columns born on device; same for the small-value table.
3. **Range-check feed on device**: the rc_9_9 input accumulation becomes one kernel
   of atomic adds into a 2^18 device count table; download (1 MB) and add into the
   host state's AtomicU32 table before rc components write. Counts are
   order-independent — byte-equality preserved by construction.
4. **Device denominators**: the interaction writer's `combine(limbs at row)` reads
   the device-resident limb columns directly (the constraint-JIT recording already
   proves the combine arithmetic byte-equal on device); numerators are the
   (device-resident) multiplicities. Feeds the existing `finalize_raw_logup` device
   pipeline — the component never materializes columns on the host at all.
5. **Gate**: a per-component differential (host writer vs device writer, column
   byte-compare — the `STWO_CUDA_CONSTRAINT_VERIFY` harness pattern) + the Cairo
   e2e. Fallback per component, NitrooZK-lesson style.

The same slice shape then applies component-by-component: memory_address_to_id,
the range-check table families, verify_instruction, then the opcode cohort (which
additionally uploads their packed VM inputs and keeps their sub-component input
pushes as device count tables). Each lands independently behind its differential.

Status: blueprint ready; W4 + W2 landed first (this session); phase-1 implementation
is the next session's opening move.

### W3 phase-1 implementation spec (formula level)

Extracted from the generated code so the port is mechanical:

- **Limb split** (`split` in `common/src/prover_types/felt.rs`): walk the 8 (big) or
  4 (small) u32 words LSB-first, emitting 28 (resp. 8) limbs of
  `FELT252_BITS_PER_WORD = 9` bits: keep a bit-buffer; while >= 9 bits remain in the
  current word emit `word & 0x1FF` and shift; on word exhaustion OR in the next
  word's low bits. Device kernel: thread per element, unrolled 28-limb emit, columns
  written column-major (coalesced). Padding rows (beyond `n_values` up to the
  power-of-two column length) are zeros; multiplicity column uploads as-is.
- **rc_9_9 feed**: the state holds `mults: [AtomicMultiplicityColumn; 8]` (one per
  relation_index = pair position i%8) and maps inputs via `input_to_row:
  HashMap<(M31,M31), row>` derived from the preprocessed table layout — NOT a
  closed-form index. Device port: upload the input->row table once as a dense
  2^18 u32 LUT (content = preprocessed layout, content-keyed), kernel does
  `atomicAdd(&counts[rel][lut[a * 512 + b]], 1)` over the limb pairs, then the 8
  count tables (8 x 1 MB) download and add into the host atomics before rc
  components write. Counts are order-independent: byte-equality by construction.
- **Denominators**: per segment, per row: `combine(chain([MEMORY_ID_TO_BIG_RELATION_ID],
  [offset + row], limbs[0..28]))` with the channel-drawn (z, alpha-powers) passed as
  kernel params; numerators are `-multiplicity`. Output feeds `finalize_raw_logup`'s
  device pipeline directly (a device-resident variant of `RawLogupColumn` — add a
  `DeviceRawLogupColumn` alongside, consumed by the same chain kernels without the
  H2D pair upload).
- **Gate**: differential harness `STWO_CUDA_WITNESS_VERIFY=1` — run host and device
  writers, byte-compare all columns + lookup sums (the constraint-verify harness
  pattern); per-component opt-in until 0 mismatches, SIMD fallback wired first.

Estimated surface: one kernel file (4 kernels), FFI x4 layers, the component's two
writers restructured behind the hook, claim plumbing unchanged (sums still arrive
in order). One focused session.
