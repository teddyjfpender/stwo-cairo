# H100 SN2 statement-upload scan removal checkpoint

This receipt records the first complete SN2 result after removing redundant
replay-time BLAKE3 scans from the production statement-upload receipts.

| Metric | Previous checkpoint | This checkpoint | Delta |
|---|---:|---:|---:|
| Raw warm median | 1.597542116 s | **1.479474910 s** | **-118.067206 ms** |
| Reported useful MHz | 4.824 | **5.209** | **+7.98%** |
| Host preparation | 154.478049 ms | **38.393575 ms** | **-116.084474 ms** |
| Session preparation | 150.767157 ms | **34.584541 ms** | **-116.182616 ms** |

The five warm samples were 1.479474910, 1.467947317, 1.479955339,
1.477843392, and 1.482114914 seconds. The exact 5-MHz wall is 1.5413728
seconds, so the measured median clears it by **61.897890 ms**.

The production path selected `staged-group-direct`, completed six of six
verified repetitions, matched a fresh SIMD proof byte-for-byte, and rejected
the structured mutation. The fetched proof was:

- 3,078,795 bytes;
- SHA-256
  `99bf0cd0863658742ada152caee238d888f5c901dc7e4df66f2748d49cea98da`;
- BLAKE3
  `253519717a2a742a211c994a57e0c2ff3eb5ad0a956933bd18235292d7bbc9f2`.

The exact source identity was:

- `stwo` `526b489db96a23fb9d044dc3fe61f839a6062f67`;
- `stwo-cairo` `be3e7e550f892909a912e04ebc0ac8fb65ed40cc`;
- `gpu_bench` SHA-256
  `0a4051aa83fb3625ec8cc3a5803b9de841ae78a588164cf529ab1db2bb8cc504`;
- 340/340 required SM90 AOT entries present.

The H100 was an 80 GB HBM3 SXM device, UUID
`GPU-80ef4dbd-d519-8e2c-5371-e896869030cb`, using driver 580.126.09,
CUDA 11.8, a 700 W power limit, and maximum 1,980 MHz SM / 2,619 MHz memory
clocks.

This is an iteration checkpoint, not a formal promotion result. It used the
timing-only policy, did not collect performance counters, and the physical
memory ledger remains incomplete. The previous 4.824-MHz checkpoint was on a
different physical H100, so its comparison is directional rather than a
same-device balanced A/B. The standalone correctness and timing result is
nevertheless complete.

The measured attribution is unusually tight: 116.084474 ms disappeared from
host preparation while complete proof wall fell by 118.067206 ms. That is
consistent with removing the two full statement-payload scans and leaves only
38.393575 ms of total host preparation. The next large owner is therefore no
longer the host receipt layer; it is the GPU slab/pass structure, beginning
with Composition's high-register waves.

See [`summary.json`](summary.json) for the machine-readable receipt.
