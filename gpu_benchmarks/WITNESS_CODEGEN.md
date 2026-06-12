# W3 completion: the opcode cohort — design and the honest mechanism

*Status: design (this round implemented the async upload lane + batched OODS,
which attack the same measured H2D cost from the transfer side; memory tables
and memory_address_to_id are already device-born and byte-equality-validated).*

## Why this is the biggest remaining lever

M1 (nsys, 4090, fib 1M, 2 proves): **H2D memcpy = 46.8% of all GPU-op time**
(645 ms / 4,698 transfers), and the kernel-side total is only ~28% of the
prove wall — the GPU starves while the host writes witnesses and ships them.
The traffic is (a) the base-trace columns of the ~60 generated opcode/builtin
components and (b) their raw-logup pairs (8 words/row). Both exist because the
per-row witness math runs on host SIMD.

## Why the constraint-JIT recipe does NOT transfer directly

The constraint lane records the component's constraint tree through the
generic `EvalAtRow` trait — one polymorphic seam every component already
implements — lowers to V1 bytecode, content-hashes, NVRTC-compiles, caches.
That works because constraint evaluation is a PURE EXPRESSION over masked
trace cells.

The witness writers (`write_trace_simd`) have no such seam:
- they are concrete generated Rust over `PackedM31` lanes with data-dependent
  branches (decode paths, small/big value splits);
- they SIDE-EFFECT sub-components mid-row (`add_packed_inputs` into rc/memory
  multiplicity tables — the device analogue is the rc99-count pattern from P1);
- their inputs are component-specific packed structs (CasmState rows, memory
  accessor closures), not a uniform mask.

So "record once and emit CUDA" needs a *witness recorder seam that does not
exist*, and the writers are emitted by stwo-air-infra (private). Three
mechanisms, ranked:

1. **Upstream witness-IR emission (the right long-term answer).** stwo-air-infra
   already holds the component IR that emits BOTH the Rust witness writer and
   the constraint evaluator. A CUDA witness emitter there is mechanical —
   same IR, different printer — and inherits regeneration safety. Requires
   access to the private repo; out of scope for this fork alone. ACTION:
   propose upstream; meanwhile (2).
2. **Hand-port by measured traffic share (the P1/addr pattern, proven twice).**
   Each port follows the established recipe: `into_parts` on the claim
   generator, device kernels for the row math, count-table merges for
   sub-component feeds, `STWO_CUDA_WITNESS_VERIFY` differential, per-component
   kill switch, proof byte-equality. Ranking for fib-shaped workloads (by
   trace cells x columns):
   a. `verify_instruction` (dedup'd instructions; pure decode per row)
   b. `add_ap_opcode` / `add_opcode_small` / `jnz_opcode_taken` (the loop body)
   c. `range_check` table families (mult columns + pair logups; the counts are
      already device-computable — P1 produces them)
   d. the long tail only via (1).
3. **Symbolic lane tracing (rejected).** Branchy writers make trace-once
   replay-many unsound; per-branch recording explodes.

## What this round shipped instead (same cost, transfer side)

- **Async upload lane**: pinned ping-pong staging, dedicated copy stream,
  stream-ordered destination allocs, one closing bridge. Host never blocks on
  PCIe; uploads overlap enqueued compute. Applied to `from_simd_evals` AND the
  raw-logup pair path. Kill switch `STWO_CUDA_SYNC_UPLOADS=1`.
- **Batched OODS** (`barycentric_eval_many`): all (column, point) jobs enqueue
  without intermediate syncs, one readback (was: 1,780 launches each followed
  by a 16-byte stream-draining readback).

These convert the H2D share from serialized time into overlapped time; (1)/(2)
above then remove the bytes themselves.
