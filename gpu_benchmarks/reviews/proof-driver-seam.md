# PCS proof-driver seam [SUPERVISED, PCS/FRI]

Status: implemented in STWO commit `4e5582c3`; retroactive review package. Native
hardware admission remains required for the current dirty-tree aggregate.

## Ground truth and supervised surface

`CommitmentSchemeProver::prove_values` still owns the public PCS entrypoint. Its
former body was moved into consuming typed states in
`crates/stwo/src/prover/pcs/proof_driver.rs`; the CPU/SIMD backend dispatch calls
`prove_values_reference`, which sequences those same states. The verifier under
`crates/stwo/src/core/` is unchanged. The supervised surface is therefore the
prover transcript/order and proof-field assembly, not verifier semantics.

The seven transcript-visible stages are pinned by `PcsProofStage::ALL`:
OODS evaluation, quotient/compaction, FRI commit/fold, PoW, FRI query/decommit,
tree decommit, and assembly. The states consume `self`, so a stage cannot be
duplicated or reordered accidentally without changing the type-level call chain.

## Invariants

1. The moved reference implementation performs the same channel calls and proof
   field construction in the same order as the pre-move body.
2. `ProofOfWork` completes before queries are drawn.
3. Each stage starts and finishes exactly once.
4. Default CPU/SIMD dispatch remains the reference orchestration.
5. No verifier code or accepted-proof predicate changes.

## Gates

- `quotient_ops::tests::backend_driver_uses_all_typed_stages_in_order` compares the
  recorded sequence to `PcsProofStage::ALL` and fails with
  `backend driver changed the transcript-stage order`.
- `backend-cuda/tests/conformance.rs` requires whole-proof byte equality to
  `CpuBackend` on ordinary and M31-output Blake2s channels.
- The pod gate requires two conformance tests to have actually executed; zero
  cfg-selected tests is failure.
- The architecture record requires all seven stage start/finish counters to be one.

## Rollback and failure class

The backend trait default calls `prove_values_reference`; a backend can therefore
drop its override without altering the shared typed implementation. A prover-side
error should yield a verifier rejection (liveness failure). Transcript-order drift
is the exceptional risk, guarded by the stage-order and byte-identity gates.

## Hardware admission

Run `gpu_benchmarks/run_cuda_soundness_gate.py` on the target CUDA build, verify the
two conformance executions and seven-by-one stage counters, then verify the emitted
proof with the unchanged STWO verifier. No performance result is admissible before
those checks pass.
