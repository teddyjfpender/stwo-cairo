# A40 local-retention manifest — 2026-07-19

This index closes the A40 development round without restarting a GPU. It records the local
evidence corpus, the source lineages that produced it, the provider lifecycle state, and the exact
limit of byte-for-byte artifact retention.

## Closure

- Read-only RunPod status on 2026-07-19 reported `wn3nd7oe1mzczc` (`cuda-stwo-dev`, one A40,
  $0.44/hour) as `EXITED`, with no runtime or SSH endpoint and therefore no active compute charge.
- Prior one-shot A40 pods `72ajgrz5n590t8` and `5n1ct5ie4990yf` were terminated and are now absent
  from the provider API.
- No pod was started, resumed, stopped, or mutated during this audit.
- No A40 source change is remote-only. Every observed source commit is present in the local Git
  object stores and is an ancestor of a pushed `origin` branch.

## Local corpus

The paths below are relative to the shared `stark-proving` workspace that contains the checkout.
The tree digest is SHA-256 over sorted lines of
`<file SHA-256><two spaces><workspace-relative path><newline>`.

| Local path | Files | Bytes | Tree digest |
|---|---:|---:|---|
| `evidence/gpu-prover-performance-delivery-2026-07-19/a40/` | 257 | 51,317,747 | `7522b3d9b03dd4a79535bd98d8b08431a42c7e56d78cae0152698d67da6043af` |
| `evidence/gpu-prover-backend-redesign-2026-07-13/stage4/A40-SN2-5MHZ-CANDIDATE-DIFFERENTIAL-2026-07-18.md` | 1 | 10,083 | `3bbfad10a035eadf57d23b139b130dc1351e44d03a8132ed17da38c3d4aea94c` |
| `stwo-cairo/gpu_benchmarks/loop/results/sn2_5mhz_a40_ab_20260718/` | 14 | 77,601 | `79d2cd9ca3249423350bb3dd97941611a5ddcd6023c064f4ae4c6368d4443c5d` |
| `stwo-cairo/gpu_benchmarks/loop/results/sn2_5mhz_a40_ab_repaired_20260718/` | 27 | 332,037 | `f8cc54b99780d76ad97c55d9d45438bac5b23005f00676425ce4b358f30778e2` |
| `scratchpad/fixed-image-a40-implementation/` | 3 | 153,795 | `cdfb5a6d21afb61f8683a32da18cc19d21d42b921804d9010d5d6c438499f107` |
| **Combined** | **302** | **51,891,263** | `9edc4c67c714eeae072d65164aa0c6ee15b50b15da8d6d5e752643ecd1f92b57` |

The corpus includes:

- four raw Nsight Systems reports and their four SQLite exports;
- exported kernel, memory, CUDA API, grid, and block statistics;
- ordinary and archive-LTO SASS, ELF/resource listings, linked device objects, and checksums;
- CUDA-event ABBA samples, GPU/host telemetry, clock-lock receipts, and environment versions;
- exact-output, mutation, source-preservation, fallback, graph-replay, and guard receipts;
- memcheck, racecheck, and synccheck logs;
- fixed-image SN1–SN4 policy matrices and the original and repaired A40 differential logs; and
- build logs, source projections, Git logs/status, artifact manifests, and decision records.

## Source retention

### `stwo`

The development pod advertised:

- `runpod-a40/codex-a40-performance-20260719` at
  `0d7b4cc2d6c18a29ffb7f16d3062d2304adb4cd9`; and
- `runpod/retained-run-sum-formal-20260719` at
  `495af876bf7b5c7a54a9c7042a9f3bfc11de18a5`.

Both refs and all A40 evidence commits are present locally. They are ancestors of pushed
`origin/codex/remove-redundant-ingest-hashes-20260719`, whose audited source head is
`526b489db96a23fb9d044dc3fe61f839a6062f67`.

The three files under `scratchpad/fixed-image-a40-implementation/` are archival snapshots, not
uncommitted source:

| Snapshot | Git blob | Retained history |
|---|---|---|
| `exec_context.rs` | `a941e0fb7bae497195ab8b27ef2b9495df7614b8` | commit `07e5a0f3bd7b180bd3e309fe578d3470c9a866cf` |
| `prepared_quotient_numerator.rs` | `c4b559e45f43894e60c332dc23519e85a96d9847` | commit `465bc7a9d11422dd4781f6bbe25dd631c0c205ae` |
| `prepared_quotient_numerator_sn3_bench_native.rs` | `5438cae5b71709734e867a4a76a4e05709c10cd3` | commit `3a97d092ea24ae2c0a69eab2af743c4437b5646b` |

Each introducing commit is also an ancestor of the pushed branch.

### `stwo-cairo`

The fixed-image policy evidence names
`726265bc5f20c9d77b32d98cc61e4c65d39fba8b` and
`bcd7629aca80a6616cb41cbaf05a0c4946faafc4`. Both commits are present locally and are ancestors of
pushed `origin/codex/adaptive-run-sum-resident-20260719`; the integrated source head audited with
the subsequent H100 checkpoint is `be3e7e550f892909a912e04ebc0ac8fb65ed40cc`.

## Hardware and tool identity

The July 19 production-shaped A40 evidence used:

| Item | Value |
|---|---|
| GPU | NVIDIA A40, `GPU-559575fd-a1c7-588c-a2df-9b0e95d5528b` |
| Compute capability | 8.6 |
| Driver | 580.159.04 |
| CUDA compiler | 13.3.73 |
| Observed graphics / memory clock | 1,740 / 7,251 MHz |
| Board power limit | 300 W |

The earlier July 18 one-shot differential used a separate A40 lane with driver 570.195.03 and
CUDA 11.8.89. Results across those two lanes are not treated as same-device comparisons.

Nsight Compute 2026.2.1 attached successfully but hardware performance counters were denied with
`ERR_NVGPUCTRPERM`. The retained counter directory therefore proves the permission boundary; it
does not contain fabricated roofline counters.

## Performance and correctness findings retained

| Boundary | Exact observed result | Decision boundary |
|---|---|---|
| Adaptive Relation, repaired repeat | 0.9279x eager / 0.9274x captured; byte-correct | reject on A40 |
| Prepacked Quotient, repaired repeat | 0.273–0.320x across four cells; byte-correct | reject on A40 |
| SN3 group-direct numerator | 783.362 → 592.740 ms p50; 1.3240x geomean; 20/20 wins; 402,645,136 exact bytes | keep component |
| Archive LTO, numerator→FRI boundary | conservative 1.1614x; at least 87.550 ms saved; four sealed runs | keep build mode |
| Paired-row boundary | 542.368 → 484.210 ms mean p50; 1.1201x; 670+ MB exact comparison | keep component |
| All-retained FixedImage boundary | 481.283 → 335.178 ms p50; 141 → 49 kernels; 92 LDE nodes removed | keep topology evidence |
| Native-domain run-sum, formal ABBA | 335.355 → 92.794 ms p50; 3.6140x; 20/20 wins; 671,080,592 exact bytes | component promotion passed |
| Production dispatch checkpoint | 335.602 → 92.991 ms p50; 3.6090x; fallback gate passed | component/forced-schedule only |

The run-sum SASS receipt records 38 and 40 registers/thread, zero local bytes, zero stack bytes,
zero shared bytes, and no `CALL`/`JCAL` instructions for the two promoted kernels. Memcheck,
racecheck, and synccheck were clean. These are component and transcript-boundary results, not an
end-to-end SN block useful-MHz claim.

## Byte-retention boundary

All source code, patches, decision data, timing samples, exact-output receipts, SASS needed for
resource review, and raw timeline profiles are local.

Five unique hashes in the evidence manifests do not have a second standalone copy in the local
evidence corpus:

| Content | Retention |
|---|---|
| `ptx.cuh` at SHA-256 `94cc65af…` | retained as committed Git source |
| retained benchmark executable at SHA-256 `a436d2e3…` | hash/provenance only |
| `libstwo_cuda_kernels.a` at SHA-256 `a24895c5…` | hash/provenance only; the linked device object it contains is retained |
| `aot_pack.bin` at SHA-256 `704f7917…` | hash/provenance only |
| generated `aot_index.rs` at SHA-256 `511b6beb…` | hash/provenance only |

The July 18 Relation and Quotient test executables are likewise represented by exact SHA-256,
source heads, manifests, receipts, and complete logs rather than copied executable bytes.
Composition produced no executable in that run.

Those missing bytes are reproducible build products rather than unique source or measurement
output. Recovering the original byte instances would require restarting an exited provider volume,
which this closure deliberately did not do. They must not be confused with missing source work.

## Audit commands

```bash
PYTHONPATH=gpu_benchmarks/fleet \
  /opt/homebrew/bin/python3.12 -m gpufleet status

git -C stwo merge-base --is-ancestor \
  0d7b4cc2d6c18a29ffb7f16d3062d2304adb4cd9 \
  origin/codex/remove-redundant-ingest-hashes-20260719

git -C stwo-cairo merge-base --is-ancestor \
  726265bc5f20c9d77b32d98cc61e4c65d39fba8b \
  origin/codex/adaptive-run-sum-resident-20260719
```

This index is intentionally small and tracked. The 49.49 MiB local raw corpus remains outside Git
to avoid committing generated Nsight databases, device objects, and duplicated logs.
