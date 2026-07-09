//! The GPU-native Cairo prover pipeline.
//!
//! Governing design: `gpu_benchmarks/GPU_RESIDENT_PROVER_DESIGN.md` (§3: the new
//! pipeline; §16: the normative interfaces). This crate OWNS proof orchestration
//! end-to-end — it does not implement stwo's `Backend` traits (those encode the
//! SIMD pipeline's synchronous, host-owned control flow, which is exactly what the
//! overhaul removes). The legacy `prove_cairo` path remains the parity oracle
//! until M6: at every milestone the pipeline here must produce proofs
//! byte-identical to it (design §9).
//!
//! M1 scope (this crate's first landing): the strangler shell — the same phase
//! sequence as `prove_cairo`/`prove_cairo_common`, decomposed into `phases/*` and
//! sequenced by [`prover::GpuCairoProver`], with the process-global caches the
//! legacy path kept in statics owned by the prover instance instead. Byte-identity
//! is the exit gate; speed work starts at M2 (witness DAG), M3 (AOT kernels),
//! M4 (commit fusion), M5 (graphs + device channel), M6 (pipelining).

pub mod arena_plan;
pub mod flags;
pub mod graphs;
pub mod phases;
pub mod plan;
pub mod protocol_plan;
pub mod prover;
pub mod relation;
pub mod relation_execution;
pub mod relation_table;
pub mod schedule;
pub mod schedule_table;
pub mod state;

pub use prover::{CairoBackend, GpuCairoProver, GpuProverConfig};
pub use stwo_backend_cuda::{CudaPcsDriverTelemetry, CudaPcsRuntimeMode};
