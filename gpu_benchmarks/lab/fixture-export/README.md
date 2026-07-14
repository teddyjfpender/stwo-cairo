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

## FRI round-6 replay fixture

The FRI exporter has two deliberately separate namespaces:

- `synthetic-layout` is a deterministic CPU self-test for the 1,936-byte binary layout. It is
  always indexed with `production_admissible: false` and cannot create an `sn2` artifact.
- `captured-unsealed` consumes a hash-pinned
  `stwo.gpu-lab.fri-round6-capture-seed.v1`. The exporter rebuilds the Cairo transcript topology
  and independently checks the device protocol key, Cairo segment key, C32-C36 prefix chains,
  cursor32 controls, log-6 evaluation, root6, alpha6, fold result, root7, alpha7, and
  cursor34-cursor36 states. It is still always `production_admissible: false`.

The distinction is soundness-critical. Hashing a capture proves which bytes were consumed, but an
observer can still supply an arbitrary self-consistent cursor32 digest and therefore manufacture a
different alpha6. Topology-chain agreement does not authenticate that digest against the start of
the real Cairo transcript. Consequently this exporter does not emit an `sn2` family today.

Production admission remains blocked until one seal binds all of the following:

1. the complete serialized reference Cairo proof and its full source SN2 PIE SHA-256;
2. canonical verification of that proof and independent transcript replay through cursor32;
3. root6/root7 from the proof to the recomputed log-6 entry and three-fold CPU oracle roots;
4. a reviewed PIE-to-ProverInput adapter seal (the currently expected adapted input is SHA-256
   `78b0995483a76e850c61cf7cb51861850f746ddf927344088014492b6752844c`, 162,102,412 bytes);
5. proof-shape identity recomputed from the verified proof rather than supplied by the observer.

The normative capture shape is
[`semantics/fri_round6_capture_seed.v1.schema.json`](semantics/fri_round6_capture_seed.v1.schema.json).
The Rust reader additionally enforces control-word and chain relationships that JSON Schema cannot
express. A capture seed is bounded to 1 MiB, rejects unknown fields, and is always supplied with its
exact SHA-256. Do not hand-author a capture seed: its source must identify
`stwo-cairo.production-simd-fri-observer.v1`, the source ProverInput SHA-256 and byte size, and the
observer-claimed proof-shape identity. Those declarations are evidence metadata, not the missing
reference-proof provenance seal.

The first provenance layer is identity-only and remains non-admissible. A reviewed manifest using
[`semantics/fri_round6_provenance.v1.schema.json`](semantics/fri_round6_provenance.v1.schema.json)
must be supplied with its out-of-band SHA-256:

```bash
"$EXPORTER" \
  --preflight-fri-round6-provenance /absolute/path/to/fri_round6_provenance.v1.json \
  --fri-round6-provenance-sha256 <64-lowercase-hex-manifest-sha256>
```

This preflight hashes bounded PIE, adapted-input, proof, transport, tool, invocation and source
closure files; rejects unknown fields, aliases, path-component symlinks and ambiguous relative
paths; and requires the adapter run record to cross-bind its exact inputs, output, invocation,
executable and source closure. Its success line deliberately says `IDENTITY_PREFLIGHT=PASS` plus
`production_admissible=false`, `proof_verification=pending`,
`adapter_execution_attestation=pending` and `verifier_closure_match=pending`. A hash-pinned run
record is a reviewed identity claim, not proof that the adapter executable produced the output.
Only the later proof deserialization, canonical reserialization, verifier, current-source closure,
proof-shape and transcript-prefix gates may close those pending labels.

The first causal layer is a separate, still non-admitting proof preflight over the same pinned v1
manifest:

```bash
"$EXPORTER" \
  --preflight-fri-round6-proof /absolute/path/to/fri_round6_provenance.v1.json \
  --fri-round6-provenance-sha256 <64-lowercase-hex-manifest-sha256>
```

It requires exact bincode decoding and byte-for-byte reserialization, byte identity with the
canonical Cairo felt transport, successful panic-safe Rust verification, and proof-shape identity
recomputed from the decoded proof. Its success line still says `production_admissible=false`:
adapter execution, verifier-source closure, and canonical verifier-to-capture matching remain
independent pending seals. The identity-only v1 command and schema retain their original meaning.
Before loading either bulk artifact, this causal mode caps the combined sealed proof-bincode and
canonical-transport payloads at 256 MiB. It compares bincode output and transport felts directly
against those loaded bytes instead of materializing duplicate byte buffers; the decoded proof and
Cairo felt vector are the remaining transient representations. Before the first artifact read, the
standalone proof process irreversibly caps data memory at 2 GiB, address space at 4 GiB, and CPU
time at 60 seconds. Only exit zero plus the one exact expected status line is accepted. Any signal,
timeout, or nonzero exit invalidates all captured text, even if a PASS fragment or full line was
written before termination. bincode 1.3 slice deserialization discards configured byte limits, so
the implementation does not claim `.with_limit` as an allocation boundary; the sealed artifact
caps and OS limits are that boundary. Callers must additionally impose a 90-second wall timeout and
use bounded or null output capture. Successful proof mode emits exactly one bounded status line.
For the two shape digests not otherwise encoded by the proof format, canonicalization is explicit:
the preprocessed variant hash covers its bincode-v1 bytes, and the enable-bit hash covers one byte
per component slot (`0` or `1`). `trace_column_log_sizes` contains the preprocessed tree first,
followed by the claim-derived base and interaction trees in verifier order.
`fri_witness_counts` contains the first layer followed by each inner layer in proof order.
`sampled_value_counts` and `queried_value_counts` retain the `CommitmentSchemeProof` tree order and
each tree's column order.

The durable cheap-host procedure builds the exporter, runs only the fully qualified ignored test,
and refuses to continue unless its summary says exactly one test passed. That test generates four
disposable, complete, hash-pinned v1 bundles. The procedure then invokes the public proof-preflight
CLI under a 90-second wall timeout and bounded stdout/stderr for one positive case and independently
resealed proof, transport, and shape mutations. It accepts exactly one expected non-admitting status
line from the positive case; every mutation must exit nonzero without an accepted status:

```bash
./gpu_benchmarks/lab/fixture-export/scripts/cheap_host_proof_boundary.sh
```

The exact gated test name used by the procedure is
`fri_round6_proof::tests::real_cairo_proof_crosses_the_full_validation_boundary`.

The headerless payload is fourteen consecutive little-endian-u32 chunks:

| Chunk | Offset | Bytes |
| --- | ---: | ---: |
| `entry_pong` | 0 | 1,024 |
| `inverse_twiddles` | 1,024 | 224 |
| `alpha6` | 1,248 | 16 |
| `entry_state` | 1,264 | 64 |
| `expected_final_ping` | 1,328 | 128 |
| `expected_root` | 1,456 | 32 |
| `expected_exit_state` | 1,488 | 64 |
| `expected_challenge7` | 1,552 | 16 |
| `expected_retained` | 1,568 | 128 |
| `expected_leaves` | 1,696 | 64 |
| `expected_mix_input` | 1,760 | 32 |
| `expected_draw_output` | 1,792 | 16 |
| `expected_boundary_mix` | 1,808 | 64 |
| `expected_boundary_draw` | 1,872 | 64 |

Each index binds the exact payload size and SHA-256, every chunk offset/length/SHA-256, normalized
twiddle offsets `0/32/48`, their full log-24 offsets, semantic IDs, transcript keys and chains,
capture/ProverInput declarations, and the exact exporter executable SHA-256. The hostile case is a
deterministic mutation of the captured entry evaluation and must change root and challenge without
changing the transcript graph. It is robustness evidence only and can never be an admissible SN2
witness, even after primary-prefix provenance is sealed.

Build the exporter once and authenticate that exact executable:

```bash
export EXPORTER=stwo_cairo_prover/target/debug/stwo-gpu-lab-fixture-export
export EXPORTER_SHA256="$(python3 -c \
  'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$EXPORTER")"
export CAPTURE=/absolute/path/to/fri-round6-capture-seed.v1.json
export CAPTURE_SHA256=<64-lowercase-hex-capture-sha256>
export FRI_CASE_DIR=/absolute/path/to/fri-round6-captured-unsealed
```

Captured-unsealed export has no default path and requires both hash-pinned inputs:

```bash
"$EXPORTER" \
  --fri-round6-capture "$CAPTURE" \
  --fri-round6-capture-sha256 "$CAPTURE_SHA256" \
  --expected-exporter-sha256 "$EXPORTER_SHA256" \
  --fri-round6-captured-unsealed-output-dir "$FRI_CASE_DIR"
```

It installs immutable
`fri-round6-captured-unsealed-{primary,hostile}.{payload.bin,index.json}` files. Before using those
bytes in a CUDA lab experiment, invoke the fail-closed capture validation gate:

```bash
"$EXPORTER" \
  --validate-fri-round6-captured-unsealed-dir "$FRI_CASE_DIR" \
  --fri-round6-capture "$CAPTURE" \
  --fri-round6-capture-sha256 "$CAPTURE_SHA256" \
  --expected-exporter-sha256 "$EXPORTER_SHA256"
```

Validation does not trust an index merely because its declared hashes match. It rehashes
the capture, rehashes the exporter before and after, regenerates both artifacts with the canonical
CPU fold/Merkle/transcript oracle, and compares every index and payload byte. Success emits one
`FRI_ROUND6_CAPTURE_VALIDATE=PASS production_admissible=false` line; any missing, symlinked,
renamed, stale, or altered artifact exits nonzero. This is not a production runner admission gate:
production code must reject this family and its false admission bit until the reference-proof seal
above exists.

For layout development only:

```bash
"$EXPORTER" --fri-round6-synthetic-layout-output-dir /tmp/fri-round6-layout
```

Those files use `fri-round6-synthetic-layout-*` names and are never production-admissible.

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
