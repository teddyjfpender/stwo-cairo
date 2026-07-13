# Canonical Pedersen fixture exporter

This host-only tool derives `witness_pedersen_builtin` outputs from the untouched production SIMD
writer, then checks every row against the candidate-free scalar slab semantics. It never
initializes CUDA and never accepts candidate-device bytes as an oracle.

## Tiny compatibility gate

From the `stwo-cairo` repository root, use the pinned toolchain explicitly. A plain `cargo` command
from that directory selects the workspace's stable toolchain and cannot compile Stwo.

```bash
CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR=stwo_cairo_prover/target \
RUSTFLAGS='-C debuginfo=0' \
  cargo +nightly-2025-06-23 run --locked --offline \
  --manifest-path gpu_benchmarks/lab/fixture-export/Cargo.toml -- \
  --fixture ../stwo/gpu-lab/cases/tiny/witness_pedersen_builtin.semantic.json \
  --check \
  --output /tmp/witness_pedersen_builtin.host-oracle.json
```

The inline `stwo.gpu-lab.semantic-fixture.v1` path remains the strict 32-row compatibility gate.
Inline fixtures are deliberately capped at 1 MiB, 4,096 rows, and 1,048,576 table words so
production data cannot accidentally fall back to whole-JSON materialization.

## Production chunk/index path

Production cases use `stwo.gpu-lab.semantic-fixture-index.v2`. The index retains the exact source
ProverInput, semantic, boundary, exporter-executable, scalar-golden, and production-SIMD identities.
The reviewed oracle version is `independent-scalar-plus-production-simd-v3`; inline arrays are
replaced with content-addressed binary references:

```json
{
  "semantic_payload": {
    "field": "M31",
    "row_count": 65536,
    "address_to_id": {
      "encoding": "m31-le-u32-row-major-v1",
      "path": "chunks/sha256/<sha256>.m31le",
      "sha256": "<sha256>",
      "element_count": 1048576,
      "byte_len": 4194304
    },
    "input_chunks": [
      {
        "encoding": "m31-le-u32-row-major-v1",
        "path": "chunks/sha256/<sha256>.m31le",
        "sha256": "<sha256>",
        "row_start": 0,
        "row_count": 65536,
        "words_per_row": 3,
        "byte_len": 786432
      }
    ],
    "expected_chunks": [
      {
        "encoding": "m31-le-u32-row-major-v1",
        "path": "chunks/sha256/<sha256>.m31le",
        "sha256": "<sha256>",
        "row_start": 0,
        "row_count": 65536,
        "words_per_row": 23,
        "byte_len": 6029312
      }
    ]
  }
}
```

Input rows are `[segment_start, enabler, iota]`. Expected/output rows are three trace words,
fourteen lookup words, then six sub-input words. Every word is canonical M31 encoded as little-
endian `u32`. Input and expected chunks must have identical contiguous row boundaries.

Build the exporter once, hash that exact executable, and generate a production fixture directly
from a hash-pinned ProverInput. A v2 index cannot be supplied through `--fixture` and cannot be
checked independently of its source ProverInput.

From the `stwo-cairo` repository root:

```bash
export EXPORTER=stwo_cairo_prover/target/debug/stwo-gpu-lab-fixture-export
export PROVER_INPUT=/absolute/path/to/prover-input.bin
export PROVER_INPUT_SHA256=<64-lowercase-hex-prover-input-sha256>
export EXPORTER_SHA256="$(python3 -c \
  'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$EXPORTER")"

"$EXPORTER" \
  --prover-input "$PROVER_INPUT" \
  --prover-input-sha256 "$PROVER_INPUT_SHA256" \
  --expected-exporter-sha256 "$EXPORTER_SHA256" \
  --fixture-class representative \
  --fixture-index ../scratchpad/gpu-lab-stage0-production-fixture/sn2/fixture-index.json \
  --output ../scratchpad/gpu-lab-stage0-production-fixture/sn2/host-oracle-index.json
```

The executable digest is checked before any fixture write and again after generation. The
ProverInput is copied to a private bounded snapshot and its required digest is checked before
deserialization. `--fixture-class` must be `representative`, `stress`, or `unseen-same-class`.
`--fixture-index` and `--output` are mandatory and must not alias the ProverInput or each other.

The exporter writes canonical chunks beside the compact fixture and oracle indexes:

```text
host-oracle-index.json
chunks/sha256/<canonical-output-sha256>.m31le
```

After reviewing the fixture and raw oracle artifact, bind them to the exact stable exporter binary
with the GPU-lab sealer. From the `stwo` repository root:

```bash
python3 gpu-lab/tools/lab.py seal-oracle \
  --fixture ../scratchpad/gpu-lab-stage0-production-fixture/sn2/fixture-index.json \
  --artifact ../scratchpad/gpu-lab-stage0-production-fixture/sn2/host-oracle-index.json \
  --exporter ../stwo-cairo/stwo_cairo_prover/target/debug/stwo-gpu-lab-fixture-export \
  --output ../scratchpad/gpu-lab-stage0-production-fixture/sn2/oracle-wrapper.json
```

`seal-oracle` accepts only a validated v2 fixture/artifact inside the shared workspace, rehashes
the exporter before and after sealing, verifies the embedded production cross-check closure, and
installs the wrapper without clobbering different evidence.

The exporter reads one bounded input chunk, computes one bounded output chunk, compares expected
words as a stream, and releases the chunk before advancing. It records independent chunk hashes
and a hash of the complete logical row stream. Chunks are fsynced and installed without clobbering
an existing content-address; the fixture and oracle JSON documents are also fsynced and installed
without replacing different evidence. A validly rehashed expected-word mutation is tested to fail
the internal checked-export gate.

## Explicit bounds and residual limit

- ProverInput snapshot: at most 256 MiB; its digest is checked before bincode deserialization.
- Exporter executable: at most 128 MiB and pinned by SHA-256.
- Index JSON: at most 16 MiB and 65,536 paired chunks.
- Input/output chunk: at most 65,536 rows (0.75/5.75 MiB of payload).
- Production SIMD segment: one power-of-two segment with `16..=2^20` rows.
- Address table: at most `2^28` M31 words.
- Conservative aggregate exporter estimate: at most 1 GiB before table/SIMD materialization.

The output and expected payloads are genuinely streamed and never transposed or assembled in
full. The address table is still resident because the canonical production
`memory_address_to_id::ClaimGenerator` requires random-access `Memory`; it is explicitly bounded
rather than described as streamed. Likewise, the canonical writer materializes one bounded
power-of-two SIMD segment. Removing those two residual bounds requires
a production host-evaluator API that can borrow a mapped table and emit trace rows incrementally;
this exporter does not invent replacement arithmetic.

The exporter fails closed if a row cannot be represented by the canonical SIMD writer. Address
zero, an address outside the memory table, and an M31 wrap into address zero remain transport/
kernel-safety cases rather than Cairo witness semantics.

## Development checks

Run these from the `stwo-cairo` repository root. Reuse the repository target directory and disable
incremental/debug artifacts; otherwise this tiny exporter can accidentally create a second
multi-gigabyte build of the full prover dependency graph.

```bash
CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR=stwo_cairo_prover/target \
RUSTFLAGS='-C debuginfo=0' \
  cargo +nightly-2025-06-23 fmt --manifest-path gpu_benchmarks/lab/fixture-export/Cargo.toml \
  --all --check

CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR=stwo_cairo_prover/target \
RUSTFLAGS='-C debuginfo=0' \
  cargo +nightly-2025-06-23 test --locked --offline \
  --manifest-path gpu_benchmarks/lab/fixture-export/Cargo.toml \
  --bin stwo-gpu-lab-fixture-export

CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR=stwo_cairo_prover/target \
RUSTFLAGS='-C debuginfo=0' \
  cargo +nightly-2025-06-23 clippy --locked --offline \
  --manifest-path gpu_benchmarks/lab/fixture-export/Cargo.toml \
  --bin stwo-gpu-lab-fixture-export -- -D warnings
```

Code is split by durable responsibility: `main` routes schemas, `model` owns semantic identity,
`oracle` owns canonical evaluation, `legacy` owns the bounded v1 gate, `streaming` owns v2 chunk
flow, and `artifact_io` owns content-addressing and atomic persistence.
