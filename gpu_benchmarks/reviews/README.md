# Retroactive supervised review index

The July 9 `bump` commits bundled protocol-adjacent changes too broadly. These
packages restore the review boundary required by §21 of
`GPU_RESIDENT_PROVER_DESIGN.md`. They document the ground truth, exact supervised
surface, invariants, fail-closed gates, rollback, and hardware admission for each
change independently:

1. [PCS proof-driver seam](proof-driver-seam.md)
2. [CUDA typed-driver default](typed-driver-default.md)
3. [Device Blake2s transcript](device-transcript.md)
4. [Batched tree decommit gather](batched-tree-decommit.md)

Native CUDA gates in these packages are not considered executed merely because a
CPU-only `cargo test` exits successfully. `run_cuda_soundness_gate.py` requires the
exact reviewed test count for every cfg-gated target and writes that count into the
pod artifact. A CPU-side manifest audit matches every `*_native.rs` target carrying
`#![cfg(stwo_cuda_link)]` to its source test count, so adding or omitting a native
test fails before pod admission.

The resident closeout gates explicitly include one execution-table parity test, one
fixed-table parity test, and both strict whole-proof tests. The second whole-proof
test reuses one prover/workspace with the same memory shape but different compact
memory content, compares each CUDA proof byte-for-byte to SIMD, and rejects stale
execution-table cache reuse.
