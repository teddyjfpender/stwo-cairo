# SN2 5 MHz cheap-GPU A/B manifest

Status: **EXECUTED FAIL-CLOSED — Relation and Quotient rejected; Composition
unmeasured.**

This is the fail-closed contract and retained outcome of the first cheap
hardware screen for the 5 MHz checkpoint. It is an A40 `sm_86`
microbenchmark proxy, not an SN2 proof-time claim. No H100 run follows until a
new candidate passes its cheap same-source gate. The target remains:

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
- `nvcc --version`, plus `rustc -Vv` and `cargo -V` from each repo's pinned
  working directory

Every A/B pair must execute in one process from one binary, on one CUDA stream,
with identical fixtures. Baseline must run before candidate and candidate
before baseline on alternating samples; report median, p10, and p90 CUDA-event
milliseconds. Eager and captured replay are separate measurements.

## Candidate readiness

| Candidate | Same-source correctness | Same-source timing | Remaining blocker |
|---|---|---|---|
| Restored adaptive relation lane | **Passed.** Every host/device byte, eager/captured mutation, zero-denominator, guard, ABI and resource-policy check passed | **Rejected.** `0.925714×` eager and `0.925448×` captured; about 8.0% slower | Do not promote on `sm_86`; remove its modeled 100–120 ms credit |
| Resource-bounded Composition stripes | **Unmeasured.** The test binary did not compile because two `stwo-cairo` exhaustive consumers lacked the hidden prepacked schedule | **Unmeasured.** No CUDA launch or timing sample occurred | Drift repaired in `8ca6975c`; run a Composition-only cheap differential, then real SN2/153 if it passes |
| Prepacked quotient | **Passed.** All seven correctness checks passed over four cases, with mutation, guards and recovery intact | **Rejected.** `0.273083–0.318238×` across eager/captured logs 18/20; 3.14–3.66× slower | Leave dormant; remove its assumed 80.459 ms planning credit |

## Execution and verdict

The authoritative lane is
`sn2_5mhz_cheap_gpu_ab.phases`. It compiles each exact integration-test
binary once, records its SHA-256, and runs these exact tests:

```text
prepared_relation_native::fused_same_binary_selector_ab_receipt
replacement_stage4_native::replacement_stage4_native_bytes_match
prepared_composition_stripes_direct_native::multidomain_direct_split_wave_and_installed_stripes_match_eager_and_replay
```

Relation and quotient deliberately use the test-only empty generated-AOT pack;
their compared functions are static CUDA kernels. Composition alone builds the
generated ordinary-stripe pack it installs and attests. This avoids rebuilding
the unrelated full pack for all three feature sets on a cold one-shot pod.

Each candidate phase is fail-soft: its raw exit status, phase log, binary
identity, and any receipt survive even when it fails. All three candidate
phases therefore run. `validate_sn2_5mhz_cheap_gpu_ab.py` is the only
fail-closed phase and rejects any raw failure, missing receipt, schema/check
omission, relation/quotient speedup not above one, resource/SM drift, identity
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

The executed bundle is
`gpu_benchmarks/loop/results/sn2_5mhz_a40_ab_20260718/`. The one-shot pod
`72ajgrz5n590t8` was absence-confirmed after 21m39s, at an estimated $0.159.
The detailed interpretation is in the
[A40 differential report](../../../../evidence/gpu-prover-backend-redesign-2026-07-13/stage4/A40-SN2-5MHZ-CANDIDATE-DIFFERENTIAL-2026-07-18.md).
