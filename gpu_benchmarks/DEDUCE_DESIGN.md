# Computed-deduce design — the fp256/EC witness family on device (D′, G5)

Status: DESIGN (device side not implemented; the host side of everything below is
landed and gated). Companion of `WITNESS_ON_GPU.md` and `ENDGAME_ARCHITECTURE.md`.

## Where this sits

The automated witness lane (transformer → `WitnessEval` → recording → CUDA codegen)
now covers the fp256/EC family's **M31 bodies** end to end. What does NOT record yet
is the family's **computed deduces** — calls into `witness/fast_deduction` that run a
sub-computation (an EC round, a points-table read) rather than a memory-table lookup:

| deduce (fast_deduction) | sites | signature | host cost share |
|---|---|---|---|
| `PackedPartialEcMulWindowBits18::deduce_output` | 28 (pedersen_aggregator) + chained inside w18 | `(M31, M31, ([M31;14], [F252;2])) -> same` | the EC round function |
| `PackedPartialEcMulGeneric::deduce_output` | 252 (partial_ec_mul_generic) | `(M31, M31, (W27, [F252;2], [F252;2], M31)) -> same` (boxed) | the generic EC round |
| `PackedPedersenPointsTableWindowBits18/9::deduce_output` | 1 (partial_ec_mul_w18) | `([M31;1]) -> [F252;2]` | table read |
| `PackedBlakeG::deduce_output` | 8 (blake_round) | `([U32;6]) -> [U32;4]` | blake g-function |
| `PackedBlakeRoundSigma::deduce_output` | 1 (blake_round) | `(M31) -> [M31;16]` | sigma table read |

On the host lane these are REAL trait calls (`WitnessEval::deduce_*` — the SIMD
evaluator calls the exact fast_deduction function the original writer calls, so the
generic writer stays byte-identical). On the recording lane they are ALL-POISON
results censused in `poison_ops` — the pinned manifest of exactly what the device
still needs.

**Separate, harder fact (not solved by this design):** `partial_ec_mul_generic`'s
writer also does OPAQUE fp256 arithmetic inline (`+`, `-`, `*`, `/` on `Felt252` —
the EC slope division). That is body arithmetic, not a deduce; it needs fp256 ISA
ops backed by the same device primitives, or that component keeps a hand-written
kernel. Everything else in the family is M31 + deduces.

## Option A — computed-deduce ISA ops backed by device functions (RECOMMENDED)

Extend the witness ISA with one op per deduce signature (`DeducePartialEcMulW18 = 26`,
`DeducePointsTableW18 = 27`, `DeduceBlakeG = 28`, `DeduceBlakeSigma = 29`, …), each
lowering in the CUDA codegen to a call into a `__device__` function implemented once
in the kernels crate:

- The fp256/EC primitives ALREADY EXIST: `cuda/ec_ops.cuh` (~17k lines: point add,
  scalar field ops), `cuda/fp256_carry_chain.cuh` / `fp256_config.cuh` /
  `fp256_dispatch_st.cuh`, `cuda/pedersen_table.cuh` (+ the device points table init
  in `pedersen_table_init.cu`). The device function is a TRANSCRIPTION of
  `fast_deduction/pedersen.rs::PartialEcMul::<N>::deduce_output` onto those
  primitives — small, testable, and validated by the existing truth-oracle pattern
  (`ExecTablesOracle`-style leg: device deduce vs host fast_deduction over real PIE
  rows, byte-compare).
- Register pressure: the EC round works on 2×fp256 affine points + a 252-bit scalar
  window — well inside what ec_ops.cuh's existing kernels already do per thread.
- The multi-word values flow through the recording as felt handles
  (`RecFelt::Limbs`); in the kernel the op consumes/produces 9-bit-limb registers,
  converting to/from the fp256 internal representation at the op boundary (the
  canonical-limb contract both sides already carry).
- Points-table deduces read the device-resident `PEDERSEN_TABLE_18/9` (already
  init-able on device via `pedersen_table_init.cu`) — a `TableLimb`-class read with a
  new table id, not even a computed op.

Cost: one ISA op + one device fn + one oracle leg per signature; codegen emits a
call, not inlined SASS (keeps the JIT kernels within the 512-instr governor). The
blake pair reuses the u32 lane work (BlakeG is 6→4 u32 words; the hand-written
`blake_witness.cu` already contains a validated `blake_g` device function to call).

## Option B — device-to-device component feeding

Run the SOURCE component's device lane first and feed its outputs as extra INPUT
columns to the consumer (the aggregator's 28 deduces are exactly the w18 component's
row outputs). This is the `blake_round → blake_g` seam documented in
`blake_round_witness_backend.rs`.

- Pro: no ISA change; the device work is the w18 component's own lane (needed anyway).
- Con: the aggregator's deduces are CHAINED (round r+1 consumes round r's output
  within the same row body), so feeding them as inputs requires either 28 separate
  kernel passes with intermediate buffers (launch storm + sync points — the exact
  duty-cycle tax we are removing) or a fused multi-round kernel (= hand-written,
  back to square one). Feeding works for FLAT producer→consumer shapes (points
  table → w18), not for chained rounds.

**Verdict:** Option A for the EC rounds and blake; the points tables as device LUTs
(new table ids). Option B only where the dataflow is flat.

## Input ABI (G6) — required for every builtin

The witness kernel launch caps `n_inputs` at 4 (`exec_tables.rs` guard AND the
generated kernel's input-pointer table). The builtin slot layout is
`[flat input words 0..K | enabler K | iota K+1 | mults K+2+j]`:

- pedersen_aggregator: K=3 → 3+1+1+1 = 6 input columns → cap must rise to ≥8.
  Raise BOTH the guard and the codegen input table together (relaxing only the guard
  launches a kernel that reads garbage slots).
- partial_ec_mul_generic: K≈125 flattened words → per-word input columns stop making
  sense; move to a device-buffer input table (one pointer + row stride — the same
  pattern as the execution tables) when that component lands.

## Validation ladder (unchanged discipline)

1. Host: generic-vs-original byte parity (landed; the battery).
2. Recording: pinned poison manifest — the ONLY poisons are the deduce ops; any new
   poison fails the test (landed).
3. Device (pod): truth-oracle leg per deduce op (device fn vs fast_deduction over
   real PIE rows), then the component's `STWO_CUDA_WITNESS_VERIFY` differential,
   then e2e proof byte-equality. Default-OFF kill switch per component throughout.
