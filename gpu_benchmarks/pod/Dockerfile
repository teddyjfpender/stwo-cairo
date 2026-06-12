# Prebuilt GPU-pod image: CUDA toolkit + pinned Rust toolchains + both repos
# fully compiled (gpu_bench, the slow-test binaries, the stwo conformance test
# binaries), with kernels fat-compiled for A100/3090/4090/H100.
#
# Purpose: paid pod time goes to the GPU, not to rustup + a 20-minute cold
# build. A pod that boots this image runs `p15_full_gate.sh` immediately; a
# source refresh (`git fetch + reset`) recompiles only changed crates against
# the warm target dirs.
#
# Built by .github/workflows/pod-image.yaml -> ghcr.io/teddyjfpender/stwo-pod.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain none -q
ENV PATH=/root/.cargo/bin:$PATH

# Pinned revisions: pass the current tips as build args so a rebuild picks up
# new pushes (the workflow passes them explicitly; defaults match the P1-P5
# round).
ARG STWO_REV=b8c5348fd238a4bab5346f481a7ed1586ad8d1d2
ARG CAIRO_REF=generic-backend

# nvcc compiles the kernels without a GPU. Consumer-fleet focus: 3090 (sm_86)
# and 4090 (sm_89) only — datacenter archs dropped to halve the kernel compile
# (5090/B200 need a CUDA >=12.8 base image before sm_120/sm_100 can be added).
ENV STWO_CUDA_NVCC=/usr/local/cuda/bin/nvcc \
    STWO_CUDA_ARCH=sm_86,sm_89

# ssh sessions don't inherit docker ENV: scripts source this file instead, so
# on-pod incremental builds keep the exact env the image's warm target dirs
# were built with (a mismatch makes cargo recompile the kernels from scratch).
RUN printf 'export STWO_CUDA_NVCC=/usr/local/cuda/bin/nvcc\nexport STWO_CUDA_ARCH=sm_86,sm_89\nsource /root/.cargo/env\n' > /root/stwo-env.sh

RUN git clone https://github.com/teddyjfpender/stwo /root/stwo \
    && cd /root/stwo && git checkout "$STWO_REV"
RUN git clone https://github.com/teddyjfpender/stwo-cairo /root/stwo-cairo \
    && cd /root/stwo-cairo && git checkout "$CAIRO_REF"

# Prebuild the stwo conformance test binaries (GATE A).
WORKDIR /root/stwo
RUN cargo test -p stwo-backend-cuda --release --no-run

# Prebuild gpu_bench + the e2e slow-test binaries (GATES B-D + BENCH).
WORKDIR /root/stwo-cairo/stwo_cairo_prover
RUN cargo build --release -p stwo-cairo-prover --bin gpu_bench \
    && cargo test -p stwo-cairo-prover --release --features slow-tests --no-run

# Test fixtures (large files are fetched best-effort; the e2e test compiles its
# own program from test_data which is in the repo).
RUN ./scripts/fetch_large_files.sh || true

WORKDIR /root
CMD ["/bin/bash"]
