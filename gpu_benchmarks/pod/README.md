# Prebuilt GPU-pod image

Paid pod minutes should go to the GPU, not to `rustup` and a cold `cargo
build`. This image bakes everything: CUDA 12.4 toolkit, the pinned Rust
toolchains, both repos (`stwo` @ the manifest-pinned rev on
`perf-optimizations`, `stwo-cairo` @ `generic-backend`), and **prebuilt**
release artifacts — `gpu_bench`, the e2e slow-test binaries, and the stwo
conformance test binaries — with kernels fat-compiled for `sm_80/86/89/90`
(A100 / 3090 / 4090 / H100).

## Use

Point the pod provider at the image, or pull it:

```bash
docker run --gpus all -it ghcr.io/teddyjfpender/stwo-pod:latest \
  bash /root/stwo-cairo/gpu_benchmarks/p15_full_gate.sh
```

The gate script detects the warm target dirs, refreshes sources
(`git fetch + reset`), and recompiles only changed crates — the GPU is busy
within a minute or two of boot instead of after a 20+ minute build.

## Refresh

The `pod-image` GitHub Actions workflow rebuilds and pushes
`ghcr.io/teddyjfpender/stwo-pod:{latest,<sha>}`:

- automatically when `gpu_benchmarks/pod/**` changes on `generic-backend`;
- manually via *Run workflow* (optionally overriding the stwo rev) — do this
  after pushing big prover changes so the next pod session starts warm.

Rebuilds reuse the GHA layer cache; only the cargo layers after the changed
source re-run.
