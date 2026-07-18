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
pub mod compiled_proof;
pub mod composition_plan;
pub mod composition_wave;
pub mod direct_composition_retention;
pub mod fixed_table;
pub mod fixed_table_materializer;
pub mod fixed_table_table;
pub mod flags;
pub mod fleet_barrier;
pub mod fleet_plan;
pub mod fleet_pow;
pub mod fleet_pow_replay;
pub mod fleet_pow_runtime;
pub mod fleet_pow_unix;
pub mod fleet_pow_worker;
pub mod fleet_spill;
pub mod graphs;
pub mod interaction_pow_vertical;
pub mod memory_ledger;
pub mod multiplicity_pipeline;
pub mod phases;
pub mod plan;
pub mod prepared_composition;
mod program_image;
pub mod proof_bundle;
pub mod protocol_discovery;
pub mod protocol_plan;
pub mod prover;
pub mod range_allocator;
mod range_arena;
#[cfg(test)]
mod raw_replacement_oracle_tests;
#[cfg(test)]
mod raw_replacement_sn_matrix_tests;
pub mod recorded_witness_inputs;
pub mod relation;
pub mod relation_execution;
pub mod relation_table;
pub mod replacement_host_cache;
mod resident_composition;
mod resident_direct_commit;
pub mod resident_input;
mod resident_oods;
pub mod resident_runtime;
pub mod resident_session;
pub mod resident_shape;
pub mod resident_sources;
pub mod resident_witness;
pub mod schedule;
pub mod schedule_table;
pub mod shape_executable;
pub mod source_ownership;
pub mod state;
pub mod transcript_plan;
pub mod workspace_cache;

pub use prepared_composition::{
    composition_workspace_requirements, composition_workspace_requirements_with_mode,
    default_composition_launch_mode, pack_composition_wide_groups, CompositionArenaSlotRequirement,
    CompositionCoefficientSource, CompositionDeviceInputs, CompositionExtParamBinding,
    CompositionLaunchMode, CompositionOutputSlots, CompositionSourceRetention,
    CompositionTraceTopology, CompositionWideGroup, CompositionWorkspaceRequirements,
    CompositionWorkspaceSlots, PreparedCompositionError, PreparedCompositionGraph,
    COMPOSITION_POINTER_ALIGNMENT_WORDS, COMPOSITION_WIDE_SMALL_MAX_EVALUATION_LOG,
};
pub use prover::{
    CairoBackend, GpuCairoProver, GpuProverConfig, MirroredResidentBlake2sProof,
    ResidentTranscriptMirrorTelemetry,
};
pub use resident_composition::ResidentCompositionError;
pub use resident_oods::{ResidentOodsError, ResidentQuotientNumeratorReceipt};
pub use resident_session::{
    ResidentExecutionReadiness, ResidentPreparationState, ResidentSessionError,
    ResidentSessionTelemetry,
};
pub use stwo_backend_cuda::{CudaPcsDriverTelemetry, CudaPcsRuntimeMode};
pub use workspace_cache::{
    PreparedRuntimeMaterialization, WorkspaceCache, WorkspaceCacheError, WorkspaceCacheTelemetry,
    WorkspaceKey, WorkspaceMaterialization,
};
