# H100 SN2 adaptive versus packed checkpoint

This receipt records the first current-source, byte-correct end-to-end H100
checkpoint for the adaptive `ReplacementV1` numerator.

| Path | Actual schedule | Raw warm median | Reported useful MHz |
|---|---|---:|---:|
| Production adaptive | `staged-group-direct` | **1.597542116 s** | **4.824** |
| Packed control | `staged-packed-single-write` | 1.903122776 s | 4.050 |

The adaptive path saved **305.580660 ms** and was **1.191282x** faster than the
same-binary packed control. Relative to the prior qualified 1.946-second
checkpoint it saved **348.457884 ms**. The exact 5-MHz wall is 1.5413728
seconds, so **56.169316 ms** remains.

Both paths completed six verified repetitions, matched the freshly generated
SIMD proof under the strict runtime gate, rejected the structured mutation, and
produced identical fetched proof bytes:

- 3,078,795 bytes;
- SHA-256
  `99bf0cd0863658742ada152caee238d888f5c901dc7e4df66f2748d49cea98da`;
- BLAKE3
  `253519717a2a742a211c994a57e0c2ff3eb5ad0a956933bd18235292d7bbc9f2`.

The source identity was `stwo`
`0d7b4cc2d6c18a29ffb7f16d3062d2304adb4cd9` and `stwo-cairo`
`7312a192b90993b67710e12ea18612552f4d69d1`. The exact `gpu_bench`
binary SHA-256 was
`35769a50c20b5d71d3de4e8b455ee7d6314a8a577fc98f16220c219c2a770e07`;
all 340 required SM90 AOT entries were present.

This is an iteration result, not formal promotion: the order was adaptive then
packed rather than balanced ABBA, performance counters were unavailable, and
the physical-memory ledger is incomplete. The defensible current headline is
therefore **4.824 useful MHz median**.

The next production-code owner is recurring session preparation:
154.478049 ms on adaptive and 156.135752 ms on packed. The shared cost is far
larger than the remaining 56.169316-ms 5-MHz gap.

See [`summary.json`](summary.json) for the machine-readable receipt.
