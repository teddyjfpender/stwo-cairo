# CUDA typed-driver default [SUPERVISED, PCS orchestration]

Status: implemented in STWO commit `4e5582c3`; migration rollback added in the
current dirty tree.

## Ground truth and supervised surface

The two CUDA `BackendForChannel` implementations select
`CudaPcsDriverConfig::detached_eager()` by default. Detached eager is an
orchestration wrapper over the shared typed stage methods; it does not replace the
PCS primitives. `ArenaGraph` is a separate explicit runtime binding and must carry
real hooks plus a stable arena. The verifier is unchanged.

## Invariants

1. Detached eager and the reference orchestration emit identical proof and aux bytes.
2. Both Blake2s channel variants retain their distinct hash/finalization semantics.
3. Every typed stage begins and finishes once; graph hooks cannot skip a stage.
4. `ArenaGraph` cannot be selected without an arena and complete hook table.
5. Strict GPU-native mode never silently falls back to the reference driver.

## Gates

- Whole-proof byte equality against `CpuBackend` on both channels.
- `CudaPcsDriverTelemetry` stage arrays equal one for every `PcsProofStage::ALL`
  entry and report the selected runtime mode.
- Strict architecture validation requires `ArenaGraph`, the exact architecture
  tag, and zero AOT misses/runtime loads/strict rejections.
- The CUDA soundness runner requires the conformance test count to be exactly two.

## Rollback

`STWO_CUDA_PCS_REFERENCE=1` routes both CUDA channel implementations through
`prove_values_reference` for a same-binary A/B or emergency migration rollback.
`GpuCairoProver::new` rejects that variable when `strict=true`, so a production
resident benchmark cannot claim the new architecture while exercising the escape.

## Hardware admission

For the first pod round after any typed-driver change, run default and reference
escape paths from the same binary and require equal serialized proof bytes and
successful verification. Then run strict ArenaGraph with the escape unset and
require seven stages exactly once plus zero forbidden AOT events.
