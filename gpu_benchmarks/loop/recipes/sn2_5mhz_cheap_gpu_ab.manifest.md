# SN2 5 MHz cheap-GPU A/B manifest

Status: **BLOCKED — do not provision from this manifest.**

This is the fail-closed contract for the last cheap-hardware checkpoint before
an SN2 H100 promotion run. It is an A40 `sm_86` microbenchmark proxy, not an
SN2 proof-time claim. The target remains:

- SN2 useful steps: `7,706,864`
- 5 MHz ceiling: `1.5413728 s`
- internal 5.2 MHz margin: `1.4820892 s`

## Provider and identity contract

When the three rows below are runnable, the phase recipe must use:

```text
# pod_run: lease one_shot=true final_action=terminate gpu=a40 gpu_count=1 min_vcpu=16 min_mem_gb=62 max_usd_hr=0.50 name_prefix=stwo-sn2-5mhz-ab-a40- ttl_hours=1.5 idle_min=15
```

`pod_run.sh` must transport one content-hashed source projection and record:

- `STWO_PARITY_REF_STWO_HEAD`
- `STWO_PARITY_REF_STWO_WORKTREE_HASH`
- `STWO_PARITY_REF_STWO_CAIRO_HEAD`
- `STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH`
- exact test-binary SHA-256 and AOT-manifest SHA-256
- GPU name, UUID, PCI bus, `sm_86`, memory, driver, power limit, and maximum
  SM/memory clocks
- `nvcc --version`, `rustc -Vv`, and `cargo -V`

Every A/B pair must execute in one process from one binary, on one CUDA stream,
with identical fixtures. Baseline must run before candidate and candidate
before baseline on alternating samples; report median, p10, and p90 CUDA-event
milliseconds. Eager and captured replay are separate measurements.

## Candidate readiness

| Candidate | Same-source correctness | Same-source timing | Remaining blocker |
|---|---|---|---|
| Restored adaptive relation lane | `prepared_relation_native::{eager_capture_and_mutated_replay_match_cairo_reference,fused_eager_capture_and_mutated_replay_match_cairo_reference,fused_zero_denominator_poison_is_fail_closed}` covers host bytes, eager, capture, mutated replay, guards, and zero-denominator failure | **No** | The current binary cannot select the pre-change all-one-read fused strategy. It also emits no CUDA-event A/B or loaded-resource receipt. Comparing commits would not be same-source. |
| Resource-bounded Composition stripes | `prepared_composition_native::mixed_direct_fallback_duplicate_reuse_and_all_direct_zero_lde_are_native_safe` compares wrapper and installed stripes eagerly, compares both with CPU, replays a captured mutation, and rejects registers `>128`, local/static shared memory, SM drift, and cubin/authority drift | **No** | The test emits no machine-readable receipt or CUDA-event eager/captured A/B, and does not mutate live installed-function facts. |
| Prepacked quotient | `replacement_stage4_native::replacement_stage4_native_bytes_match` covers exact plan identity, eager bytes, captured mutated replay, four-byte status observation, invalid-descriptor stale-output protection, status reset/recovery, and guards | **Captured only** | `STWO_STAGE4_NATIVE_PERF=1` times the staged and prepacked captured graphs, but not eager launches; the test also runs unrelated Stage-4 candidates and emits no exact kernel-resource receipt. |

Therefore there is no honest all-three `sn2_5mhz_cheap_gpu_ab.phases` yet.
Creating one now would either compare different source revisions, omit required
measurements, or over-credit a candidate.

## Existing commands to preserve

These are the authoritative partial gates; the eventual recipe should call
them rather than create another harness:

```bash
cd "$STWO"
cargo test --release --locked -p stwo-backend-cuda \
  --test prepared_relation_native \
  fused_eager_capture_and_mutated_replay_match_cairo_reference \
  -- --exact --nocapture --test-threads=1
cargo test --release --locked -p stwo-backend-cuda \
  --test prepared_relation_native \
  fused_zero_denominator_poison_is_fail_closed \
  -- --exact --nocapture --test-threads=1

cd "$CAIRO"
cargo test --release --locked -p stwo-cairo-gpu-prover \
  --features direct-retention-test-api \
  --test prepared_composition_native \
  mixed_direct_fallback_duplicate_reuse_and_all_direct_zero_lde_are_native_safe \
  -- --exact --nocapture --test-threads=1

cd "$STWO"
STWO_CUDA_ARCH=sm_86 \
STWO_STAGE4_GIT_COMMIT="$STWO_PARITY_REF_STWO_HEAD" \
STWO_STAGE4_NATIVE_PERF=1 \
STWO_STAGE4_NATIVE_PERF_LOGS=18,20 \
STWO_STAGE4_NATIVE_PERF_WARMUPS=5 \
STWO_STAGE4_NATIVE_PERF_ITERATIONS=30 \
STWO_STAGE4_NATIVE_RECEIPT="$RUN/sn2_5mhz_stage4.receipt.json" \
cargo test --release --locked -p stwo-backend-cuda \
  --test replacement_stage4_native replacement_stage4_native_bytes_match \
  -- --exact --nocapture --test-threads=1
```

## Minimum unblock, in order

1. Relation: add one test-only same-binary baseline strategy and emit one
   correctness/resource/timing receipt. Production selection stays unchanged.
2. Composition: extend the existing native fixture with alternating CUDA-event
   eager/captured timing and one JSON receipt containing every installed
   function's identity and resources.
3. Quotient: add eager timing and resource facts to its existing receipt, plus
   a fixture filter so only staged-versus-prepacked is timed.
4. Add one A40 phase recipe that runs each row independently, validates the
   receipt schema and hashes, then terminates the pod on every exit path.

Promotion requires all three receipts to prove byte equality, mutation
observability, guard preservation, fail-closed status, exact loaded resources,
and a positive median speedup. Their measured deltas may then update the SN2
5 MHz budget. No A40 result is itself an H100 or end-to-end SN2 number.
