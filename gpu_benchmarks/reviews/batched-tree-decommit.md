# Batched tree decommit gather [SUPERVISED, Merkle opening]

Status: live CUDA behavior introduced in STWO commit `4e5582c3`; audited and given a
same-binary rollback in the current dirty tree.

## Ground truth and supervised surface

`CudaBackend` overrides `MerkleOpsLifted::batch_gather_column_rows`. The override
uploads one pointer/offset/index descriptor set, launches one device gather, and
copies one flattened result instead of issuing a transfer per column/row. The result
feeds the existing Merkle leaf reconstruction and authentication-path logic for every
CUDA proof. This is the only July 9 change that directly altered the then-live tree
decommit data path.

## Invariants

1. Column order, row-list order, duplicates, and empty columns are preserved exactly.
2. Indices and lengths are bounds-checked; non-empty null device columns fail closed.
3. Gathered values retain raw device `u32` representation, including the unreduced
   word `P`; reduction occurs only when the typed queried value is consumed.
4. Leaf hashing consumes raw words, so root, hash witness, and auxiliary-node maps
   equal the CPU reference.
5. A device launch/copy error aborts rather than returning partial openings.

## Gates

- Backend-testkit compares mixed-width and exact 16-word boundary column groups.
- The raw-`P` fixture compares Merkle root, queried values, hash witness, and complete
  auxiliary node maps against `CpuBackend`.
- Whole-proof conformance requires byte equality on both Blake2s channel variants.
- The CUDA soundness runner requires both conformance tests to execute on GPU.
- The later resident `prepared_decommit_native` gate separately compares dynamic
  query normalization, all trace/FRI sections, and compact assembly in eager/capture.

## Rollback

`STWO_CUDA_DECOMMIT_GATHER_REFERENCE=1` selects the trait-equivalent per-column
`gather_unreduced` path while preserving raw-word semantics. Strict GPU-native mode
rejects this variable, preventing a benchmark from claiming the batched/resident
architecture while using the rollback.

## Hardware admission

Run the counted conformance and prepared-decommit native targets. For the live proof,
require proof byte equality, verifier success, matching aux maps, and no gather error
across repeated warm proofs. Record tree-decommit time and transfers, but treat them
as performance evidence only after all equality checks pass.
