# SN2 5 MHz cheap-GPU A/B manifest

Status: **READY FOR DRY-RUN — no pod has been provisioned.**

This is the fail-closed contract for the last cheap-hardware checkpoint before
an SN2 H100 promotion run. It is an A40 `sm_86` microbenchmark proxy, not an
SN2 proof-time claim. The target remains:

- SN2 useful steps: `7,706,864`
- 5 MHz ceiling: `1.5413728 s`
- internal 5.2 MHz margin: `1.4820892 s`

## Provider and identity contract

The phase recipe uses:

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
| Restored adaptive relation lane | **Ready.** `stwo` commits `b79eb515` and `93c59874` provide a same-binary historical selector, host/eager/captured/mutated bytes, selector-changing zero-denominator poison, raw guards, and exact loaded-function facts | **Ready.** Schema `stwo.prepared-relation.same-binary-ab.v2` reports eager and captured alternating CUDA-event samples and must show `>1.0×` median speedup | A40 is first-characterization-only for resources; its result grants no SN2/H100 timing credit |
| Resource-bounded Composition stripes | **Ready as a diagnostic proxy.** `stwo-cairo` commits `6713163d`, `ba1e5bef`, and `12a13606` compare direct-retained eager and mutated captured bytes on one arena/context/stream, bind exact installed functions, and require zero strict-AOT rejections | **Ready as diagnostic only.** Eager and captured ABBA timing is recorded, but the source-JIT Wave arm is not a promotable production baseline | Real SN2/153 with an installed same-pack Wave baseline remains required before any budget credit |
| Prepacked quotient | **Ready.** `stwo` commit `7bc2fc00` selects only the staged/prepacked boundary and covers independent CPU bytes, eager, captured mutation, status reset/recovery, stale-output rejection, sources, and guards | **Ready.** Exactly eager/captured × logs 18/20, one stream/source set, status fence outside timing, four exact loaded functions, and `>1.0×` median speedup | A40 result is a same-source candidate screen, not an SN2/H100 timing delta |

## Execution and verdict

The authoritative lane is
`sn2_5mhz_cheap_gpu_ab.phases`. It compiles each exact integration-test
binary once, records its SHA-256, and runs these exact tests:

```text
prepared_relation_native::fused_same_binary_selector_ab_receipt
replacement_stage4_native::replacement_stage4_native_bytes_match
prepared_composition_stripes_direct_native::multidomain_direct_split_wave_and_installed_stripes_match_eager_and_replay
```

Each candidate phase is fail-soft: its raw exit status, phase log, binary
identity, and any receipt survive even when it fails. All three candidate
phases therefore run. `validate_sn2_5mhz_cheap_gpu_ab.py` is the only
fail-closed phase and rejects any raw failure, missing receipt, schema/check
omission, nonpositive relation/quotient speedup, resource/SM drift, identity
drift, swallowed quotient performance failure, or dishonest Composition
promotion label.

Static admission:

```bash
bash -n gpu_benchmarks/loop/recipes/sn2_5mhz_cheap_gpu_ab.phases
shellcheck gpu_benchmarks/loop/recipes/sn2_5mhz_cheap_gpu_ab.phases
python3 gpu_benchmarks/loop/recipes/test_validate_sn2_5mhz_cheap_gpu_ab.py
python3 gpu_benchmarks/loop/recipes/validate_sn2_5mhz_cheap_gpu_ab.py self-test
BENCH_POD_ID=dry-run-a40 DRY_RUN=1 \
  ./gpu_benchmarks/loop/pod_run.sh \
  gpu_benchmarks/loop/recipes/sn2_5mhz_cheap_gpu_ab.phases \
  sn2_5mhz_static_dry_run
```

A real run additionally requires an explicitly leased one-shot A40 ID:

```bash
BENCH_POD_ID=<fresh-a40-id> \
  ./gpu_benchmarks/loop/pod_run.sh \
  gpu_benchmarks/loop/recipes/sn2_5mhz_cheap_gpu_ab.phases \
  sn2_5mhz_a40_ab
```

The final verdict records candidate admission and observed same-source
speedups, but sets direct promotion credit to false for every row. No A40
result is an end-to-end SN2 proof-time number or evidence that the 5 MHz wall
has been met. It only decides whether these candidates are safe and useful
enough to carry into the exact SN2 integration gate.
