# Device Blake2s transcript [SUPERVISED, Fiat-Shamir channel]

Status: foundation implemented in STWO commit `b36f6734`; now a live resident-runtime
caller. Native CUDA differential execution remains a mandatory pod gate.

## Ground truth and supervised surface

`Blake2sTranscriptSchedule` is a typed, versioned list of exact channel operations.
`PreparedBlake2sTranscript` binds that schedule to stable arena inputs, outputs,
state and snapshots, then launches explicit-stream CUDA kernels. Cairo supplies a
separate pure schedule (`stwo-cairo.blake2s.transcript.schedule.v1`) derived from the
canonical proof flow. The current schedule is split at both PoW seeds so grinding
occurs between state production and nonce absorption.

This is a prover-side reimplementation of Fiat-Shamir state transition and is
soundness-critical even though the verifier is unchanged.

## Invariants

1. Every operation has a stable semantic boundary and input/output ID.
2. Segment ranges are contiguous, generation-bound, and cannot be skipped,
   duplicated, resumed from the wrong schedule, or initialized twice.
3. Device state/output/boundary snapshots equal ordinary host Blake2s replay.
4. M31 inputs are canonical; invalid words poison status and prevent advancement.
5. Interaction PoW is computed after the base root and before z/alpha; query PoW is
   computed after LinePoly absorption and before query draws.
6. Replay allocates, transfers, synchronizes, and touches the default stream zero times.

## Gates

- Pure schedule/reference tests run on every host.
- `native_device_vectors_match_reference_in_eager_and_graph_modes` compares every
  boundary, output word, final digest, and captured replay to host reference.
- Whole-proof conformance on both channels is the outer transcript/proof gate.
- `run_cuda_soundness_gate.py` requires the native transcript test count to be one;
  a CPU-only cfg no-op is failure.
- The strict architecture record must show no hot-path host sync/H2D/D2H boundary.

## Rollback and failure class

Non-strict migration can use the host/reference PCS path. Strict resident mode has
no transcript fallback: any binding, status, mirror, ordering, or CUDA error aborts
the proof. A bad device transcript should produce a rejected proof; whole-proof byte
identity is required before treating it as authoritative.

## Hardware admission

Run the counted native transcript gate first, then eager-versus-captured whole-proof
byte equality on a real SN PIE. Preserve the mirror artifact and operation/segment
counts with the benchmark record. Do not admit wall-clock measurements from a run
where the native test count is zero.
