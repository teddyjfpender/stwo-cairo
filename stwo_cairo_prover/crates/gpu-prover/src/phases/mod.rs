//! Phase modules (design §3.1 layering: `prover.rs` → `phases/*` → backend).
//!
//! Phases never call each other; [`crate::prover::GpuCairoProver::prove`]
//! sequences them and owns the transcript spine (every channel operation is
//! visible in one place — the Fiat-Shamir order IS the proof, so it lives at the
//! top). Tracing span names are kept IDENTICAL to the legacy pipeline so the
//! phase-ledger tooling (`STWO_BENCH_TRACE` parsing) works unchanged across
//! engines.
//!
//! M1: `stark` covers composition+FRI+decommit via stwo's `prove_ex` — the split
//! into `composition.rs`/`fri.rs` (design §3.1's end-state tree) happens when the
//! pipeline takes orchestration below that API (M4/M5).

pub mod commit;
pub mod ingest;
pub mod interaction;
pub mod stark;
pub mod witness;
