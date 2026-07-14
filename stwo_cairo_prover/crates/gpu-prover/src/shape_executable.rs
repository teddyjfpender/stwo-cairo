//! Persistent host control plane for one exact proof topology.
//!
//! The cache boundary is the full typed [`TopologyKey`], not its digest. The
//! digest is an external identity; exact structural equality is the admission
//! rule. A hit re-records only proof-varying composition parameters and proves
//! that they still lower to the installed AOT kernel identities.

use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use cairo_air::claims::CairoClaim;
use cairo_air::relations::CommonLookupElements;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::ArenaSlotSpec;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};
use stwo_cairo_prover::witness::proof_shape::{ProofShape, RowResolution, TracePartId};

use crate::arena_plan::{
    ArenaBinding, ArenaPlanError, ExecutionTableGeometry, LogicalBuffer, ProofArenaPlan,
};
use crate::composition_plan::{
    bind_cairo_composition, compile_cairo_composition_binding_plan, plan_cairo_composition,
    CompositionBindingPlan, CompositionPlan, CompositionPlanError, CompositionProofBindings,
};
use crate::plan::ProofPlan;
use crate::protocol_discovery::{
    discover_protocol_transcript_shape, schema_zero_interaction_claim_for_composition,
    ProtocolDiscoveryError, ProtocolTranscriptDiscovery,
};
use crate::protocol_plan::{plan_protocol_geometry, ProtocolPlanError, ProtocolPlanPolicy};
use crate::transcript_plan::{
    claim_public_data_felt_count, plan_cairo_blake2s_transcript, CairoBlake2sTranscriptPlan,
    TranscriptPlanError,
};
use crate::workspace_cache::WorkspaceKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PcsTopology {
    pow_bits: u32,
    log_blowup_factor: u32,
    log_last_layer_degree_bound: u32,
    n_queries: usize,
    fold_step: u32,
    lifting_log_size: Option<u32>,
}

impl From<PcsConfig> for PcsTopology {
    fn from(value: PcsConfig) -> Self {
        Self {
            pow_bits: value.pow_bits,
            log_blowup_factor: value.fri_config.log_blowup_factor,
            log_last_layer_degree_bound: value.fri_config.log_last_layer_degree_bound,
            n_queries: value.fri_config.n_queries,
            fold_step: value.fri_config.fold_step,
            lifting_log_size: value.lifting_log_size,
        }
    }
}

/// Complete typed identity for every input allowed to change a compiled DAG,
/// arena layout, launch topology, transcript schedule or proof layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyKey {
    shape: ProofShape,
    relation_graph_hash: u64,
    component_enable_bits: Vec<bool>,
    component_log_sizes: Vec<u32>,
    claim_log_sizes: Vec<Vec<u32>>,
    claim_public_data_felts: u32,
    preprocessed_trace_variant: PreProcessedTraceVariant,
    preprocessed_columns: Vec<(String, u32)>,
    pcs: PcsTopology,
    include_all_preprocessed_columns: bool,
    execution_tables: Option<ExecutionTableGeometry>,
    policy: ProtocolPlanPolicy,
    digest: [u8; 32],
}

impl TopologyKey {
    fn new(
        claim: &CairoClaim,
        proof_plan: &ProofPlan,
        preprocessed_trace: &PreProcessedTrace,
        pcs: PcsConfig,
        include_all_preprocessed_columns: bool,
        execution_tables: Option<ExecutionTableGeometry>,
        policy: ProtocolPlanPolicy,
    ) -> Result<Self, ShapeExecutableError> {
        let preprocessed_columns = canonical_preprocessed_columns(preprocessed_trace)?;
        let (component_enable_bits, component_log_sizes) = claim.component_topology();
        let mut key = Self {
            shape: proof_plan.proof_shape().clone(),
            relation_graph_hash: proof_plan.relation_graph_hash,
            component_enable_bits,
            component_log_sizes,
            claim_log_sizes: claim.log_sizes().0,
            claim_public_data_felts: claim_public_data_felt_count(claim)?,
            preprocessed_trace_variant: preprocessed_trace.variant,
            preprocessed_columns,
            pcs: pcs.into(),
            include_all_preprocessed_columns,
            execution_tables,
            policy,
            digest: [0; 32],
        };
        key.digest = key.compute_digest();
        Ok(key)
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(b"stwo-cairo-shape-executable-v5\0");
        feed_u64(&mut hash, self.relation_graph_hash);
        for component in self.shape.components() {
            feed_bytes(&mut hash, component.id.as_bytes());
            match &component.rows {
                RowResolution::Absent => {
                    hash.update(&[0]);
                }
                RowResolution::Resolved(parts) => {
                    hash.update(&[1]);
                    feed_usize(&mut hash, parts.len());
                    for part in parts {
                        match part.part {
                            TracePartId::Main => {
                                hash.update(&[0]);
                            }
                            TracePartId::MemoryBig(index) => {
                                hash.update(&[1]);
                                hash.update(&index.to_le_bytes());
                            }
                            TracePartId::MemorySmall => {
                                hash.update(&[2]);
                            }
                        };
                        feed_u64(&mut hash, part.n_real_rows);
                        feed_u64(&mut hash, part.padded_rows);
                    }
                }
                // A shape executable is capture-ready by construction. Keep
                // these encodings total so a malformed caller still has a
                // deterministic identity before planning rejects it.
                RowResolution::Pending {
                    observed_n_real_rows,
                    ..
                } => {
                    hash.update(&[2]);
                    feed_u64(&mut hash, *observed_n_real_rows);
                }
                RowResolution::Bounded { bound, .. } => {
                    hash.update(&[3]);
                    feed_u64(&mut hash, bound.observed_rows);
                    feed_u64(&mut hash, bound.max_rows);
                    feed_u64(&mut hash, bound.padded_capacity);
                }
            }
        }
        feed_usize(&mut hash, self.component_enable_bits.len());
        for &enabled in &self.component_enable_bits {
            hash.update(&[u8::from(enabled)]);
        }
        feed_usize(&mut hash, self.component_log_sizes.len());
        for &log_size in &self.component_log_sizes {
            hash.update(&log_size.to_le_bytes());
        }
        for tree in &self.claim_log_sizes {
            feed_usize(&mut hash, tree.len());
            for &log_size in tree {
                hash.update(&log_size.to_le_bytes());
            }
        }
        hash.update(&self.claim_public_data_felts.to_le_bytes());
        hash.update(&[preprocessed_variant_tag(self.preprocessed_trace_variant)]);
        for (id, log_size) in &self.preprocessed_columns {
            feed_bytes(&mut hash, id.as_bytes());
            hash.update(&log_size.to_le_bytes());
        }
        hash.update(&self.pcs.pow_bits.to_le_bytes());
        hash.update(&self.pcs.log_blowup_factor.to_le_bytes());
        hash.update(&self.pcs.log_last_layer_degree_bound.to_le_bytes());
        feed_usize(&mut hash, self.pcs.n_queries);
        hash.update(&self.pcs.fold_step.to_le_bytes());
        hash.update(&self.pcs.lifting_log_size.unwrap_or(u32::MAX).to_le_bytes());
        hash.update(&[u8::from(self.include_all_preprocessed_columns)]);
        match self.execution_tables {
            Some(geometry) => {
                hash.update(&[1]);
                for value in [
                    geometry.n_addrs,
                    geometry.n_big,
                    geometry.n_small,
                    geometry.public_memory_entries,
                ] {
                    feed_usize(&mut hash, value);
                }
            }
            None => {
                hash.update(&[0]);
            }
        }
        feed_u64(&mut hash, self.policy.channel_tag);
        feed_u64(&mut hash, self.policy.kernel_manifest_hash);
        feed_usize(&mut hash, self.policy.composition_max_kernel_instrs);
        hash.update(&[self.policy.decommit_strategy as u8]);
        feed_usize(&mut hash, self.policy.retained_lde_budget_bytes);
        hash.update(&self.policy.unretained_bottom_layers.to_le_bytes());
        hash.update(&self.policy.max_fused_tail_levels.to_le_bytes());
        hash.update(&[self.policy.commit_mode as u8]);
        hash.update(&[self.policy.direct_composition_retention_mode as u8]);
        hash.update(&[self.policy.quotient_numerator_source_policy as u8]);
        hash.update(&[self.policy.interpolation_mode as u8]);
        hash.update(&[u8::from(self.policy.blake2s_interior_fused)]);
        hash.update(&[self.policy.composition_launch_mode as u8]);
        hash.update(&[self.policy.relation_tail_mode as u8]);
        hash.update(&[self.policy.fri_fold_launch_mode as u8]);
        hash.update(&[self.policy.witness_feed_launch_mode as u8]);
        hash.update(&[self.policy.resident_backend as u8]);
        hash.update(&[self.policy.quotient_numerator_schedule as u8]);
        *hash.finalize().as_bytes()
    }
}

fn preprocessed_variant_tag(variant: PreProcessedTraceVariant) -> u8 {
    match variant {
        PreProcessedTraceVariant::Canonical => 0,
        PreProcessedTraceVariant::CanonicalWithoutPedersen => 1,
        PreProcessedTraceVariant::CanonicalSmall => 2,
    }
}

fn ordered_preprocessed_columns(trace: &PreProcessedTrace) -> Vec<(String, u32)> {
    trace
        .ids()
        .into_iter()
        .zip(trace.log_sizes())
        .map(|(id, log_size)| (id.id, log_size))
        .collect()
}

fn canonical_preprocessed_columns(
    supplied: &PreProcessedTrace,
) -> Result<Vec<(String, u32)>, ShapeExecutableError> {
    let supplied_columns = ordered_preprocessed_columns(supplied);
    static CANONICAL_COLUMNS: OnceLock<[Vec<(String, u32)>; 3]> = OnceLock::new();
    let expected_columns = &CANONICAL_COLUMNS.get_or_init(|| {
        PreProcessedTraceVariant::ALL_VARIANTS
            .map(|variant| ordered_preprocessed_columns(&variant.to_preprocessed_trace()))
    })[usize::from(preprocessed_variant_tag(supplied.variant))];
    if &supplied_columns != expected_columns {
        let first_mismatch = supplied_columns
            .iter()
            .zip(expected_columns)
            .position(|(supplied, expected)| supplied != expected)
            .unwrap_or_else(|| supplied_columns.len().min(expected_columns.len()));
        return Err(ShapeExecutableError::NonCanonicalPreprocessedGeometry {
            variant: supplied.variant,
            supplied_columns: supplied_columns.len(),
            expected_columns: expected_columns.len(),
            first_mismatch,
        });
    }
    Ok(supplied_columns)
}

/// Exact structural identity of the stable device allocation compiled for one
/// topology. The short workspace key remains useful for telemetry, but never
/// participates in cache admission by itself.
#[derive(Clone, Debug, Eq, PartialEq)]
struct WorkspaceLayoutIdentity {
    total_words: usize,
    logical: Vec<LogicalBuffer>,
    bindings: Vec<ArenaBinding>,
    slots: Vec<ArenaSlotSpec>,
}

impl WorkspaceLayoutIdentity {
    fn from_plan(plan: &ProofArenaPlan) -> Result<Self, ShapeExecutableError> {
        let physical = plan
            .bindings()
            .iter()
            .map(|binding| binding.physical)
            .collect::<BTreeSet<_>>();
        let slots = physical
            .into_iter()
            .map(|id| {
                plan.layout()
                    .slot(id)
                    .ok_or(ShapeExecutableError::MissingArenaSlot(id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if slots.len() != plan.range_view_count() {
            return Err(ShapeExecutableError::ArenaViewCountMismatch {
                bound: slots.len(),
                planned: plan.range_view_count(),
            });
        }
        Ok(Self {
            total_words: plan.total_words(),
            logical: plan.logical_buffers().to_vec(),
            bindings: plan.bindings().to_vec(),
            slots,
        })
    }

    fn matches_plan(&self, plan: &ProofArenaPlan) -> bool {
        self.total_words == plan.total_words()
            && self.logical == plan.logical_buffers()
            && self.bindings == plan.bindings()
            && self.slots.len() == plan.range_view_count()
            && self
                .slots
                .iter()
                .all(|slot| plan.layout().slot(slot.id) == Some(*slot))
    }
}

/// Full typed admission proof carried from host compilation into the CUDA
/// workspace cache. Equality compares canonical topology and physical layout;
/// the two-u64 [`WorkspaceKey`] is diagnostic metadata only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceAdmission {
    key: WorkspaceKey,
    topology: Arc<TopologyKey>,
    layout: Arc<WorkspaceLayoutIdentity>,
}

impl WorkspaceAdmission {
    fn compile(
        topology: Arc<TopologyKey>,
        plan: &ProofArenaPlan,
    ) -> Result<Self, ShapeExecutableError> {
        Ok(Self {
            key: WorkspaceKey::from_plan(plan),
            topology,
            layout: Arc::new(WorkspaceLayoutIdentity::from_plan(plan)?),
        })
    }

    pub const fn workspace_key(&self) -> WorkspaceKey {
        self.key
    }

    pub(crate) fn matches_plan(&self, plan: &ProofArenaPlan) -> bool {
        self.key == WorkspaceKey::from_plan(plan) && self.layout.matches_plan(plan)
    }
}

fn feed_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    feed_usize(hash, bytes.len());
    hash.update(bytes);
}

fn feed_usize(hash: &mut blake3::Hasher, value: usize) {
    feed_u64(hash, value as u64);
}

fn feed_u64(hash: &mut blake3::Hasher, value: u64) {
    hash.update(&value.to_le_bytes());
}

/// Host half of the durable shape executable. CUDA workspace ownership remains
/// in `WorkspaceCache`; both use the same exact topology and arena plan.
pub struct ShapeExecutable {
    admission: WorkspaceAdmission,
    discovery: ProtocolTranscriptDiscovery,
    transcript: CairoBlake2sTranscriptPlan,
    composition: CompositionPlan,
    composition_bindings: CompositionBindingPlan,
    arena: Arc<ProofArenaPlan>,
}

impl ShapeExecutable {
    pub fn topology(&self) -> &TopologyKey {
        &self.admission.topology
    }

    pub fn arena(&self) -> &Arc<ProofArenaPlan> {
        &self.arena
    }

    pub fn workspace_key(&self) -> WorkspaceKey {
        self.admission.workspace_key()
    }

    pub fn workspace_admission(&self) -> &WorkspaceAdmission {
        &self.admission
    }

    pub(crate) fn discovery(&self) -> &ProtocolTranscriptDiscovery {
        &self.discovery
    }

    pub(crate) fn transcript(&self) -> &CairoBlake2sTranscriptPlan {
        &self.transcript
    }

    pub(crate) fn composition(&self) -> &CompositionPlan {
        &self.composition
    }

    pub(crate) fn protocol_policy(&self) -> ProtocolPlanPolicy {
        self.admission.topology.policy
    }

    fn composition_bindings(&self) -> &CompositionBindingPlan {
        &self.composition_bindings
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShapeExecutableCacheTelemetry {
    pub hits: u64,
    pub misses: u64,
    pub compilations: u64,
    /// Cold composition-planning passes that may invoke one or more emitters.
    /// This is deliberately not presented as an emitter-call count.
    pub source_generation_passes: u64,
    /// Cold compilations of the fail-closed statement binding recipe. A warm
    /// cache hit must not record or lower a Cairo evaluator.
    pub binding_recipe_compilations: u64,
    pub capacity_rejections: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeExecutableMaterialization {
    Compiled,
    Reused,
}

pub struct ShapeExecutableSelection {
    pub executable: Arc<ShapeExecutable>,
    pub bindings: CompositionProofBindings,
    pub materialization: ShapeExecutableMaterialization,
}

pub struct ShapeCompileRequest<'a> {
    pub claim: &'a CairoClaim,
    pub proof_plan: &'a ProofPlan,
    pub preprocessed_trace: &'a PreProcessedTrace,
    pub pcs: PcsConfig,
    pub include_all_preprocessed_columns: bool,
    pub execution_tables: Option<ExecutionTableGeometry>,
    pub policy: ProtocolPlanPolicy,
}

pub struct ShapeExecutableCache {
    capacity: usize,
    entries: Vec<Arc<ShapeExecutable>>,
    telemetry: ShapeExecutableCacheTelemetry,
}

impl ShapeExecutableCache {
    pub fn new(capacity: usize) -> Result<Self, ShapeExecutableError> {
        if capacity == 0 {
            return Err(ShapeExecutableError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            telemetry: ShapeExecutableCacheTelemetry::default(),
        })
    }

    pub const fn telemetry(&self) -> ShapeExecutableCacheTelemetry {
        self.telemetry
    }

    pub fn compile_or_bind(
        &mut self,
        request: ShapeCompileRequest<'_>,
    ) -> Result<ShapeExecutableSelection, ShapeExecutableError> {
        let topology = TopologyKey::new(
            request.claim,
            request.proof_plan,
            request.preprocessed_trace,
            request.pcs,
            request.include_all_preprocessed_columns,
            request.execution_tables,
            request.policy,
        )?;
        if let Some(executable) = self
            .entries
            .iter()
            .find(|executable| executable.topology() == &topology)
            .cloned()
        {
            let bindings =
                bind_cairo_composition(request.claim, executable.composition_bindings())?;
            self.telemetry.hits += 1;
            return Ok(ShapeExecutableSelection {
                executable,
                bindings,
                materialization: ShapeExecutableMaterialization::Reused,
            });
        }
        self.telemetry.misses += 1;
        if self.entries.len() == self.capacity {
            self.telemetry.capacity_rejections += 1;
            return Err(ShapeExecutableError::AtCapacity {
                capacity: self.capacity,
                requested: topology.digest,
            });
        }
        let executable = Arc::new(compile_shape_executable(topology, &request)?);
        let bindings = bind_cairo_composition(request.claim, executable.composition_bindings())?;
        self.entries.push(Arc::clone(&executable));
        self.telemetry.compilations += 1;
        self.telemetry.source_generation_passes += 1;
        self.telemetry.binding_recipe_compilations += 1;
        Ok(ShapeExecutableSelection {
            executable,
            bindings,
            materialization: ShapeExecutableMaterialization::Compiled,
        })
    }
}

fn compile_shape_executable(
    topology: TopologyKey,
    request: &ShapeCompileRequest<'_>,
) -> Result<ShapeExecutable, ShapeExecutableError> {
    let max_trace_log = request
        .claim
        .log_sizes()
        .iter()
        .flatten()
        .copied()
        .max()
        .ok_or(ShapeExecutableError::EmptyTraceGeometry)?;
    let required = max_trace_log
        .checked_add(request.pcs.fri_config.log_blowup_factor.max(1))
        .ok_or(ShapeExecutableError::SizeOverflow)?;
    let lifting_log_size = request.pcs.lifting_log_size.unwrap_or(required);
    if lifting_log_size < required {
        return Err(ShapeExecutableError::InvalidLiftingLogSize {
            lifting: lifting_log_size,
            required,
        });
    }
    let discovery = discover_protocol_transcript_shape(
        request.claim,
        request.proof_plan,
        request.preprocessed_trace,
        &request.pcs,
        lifting_log_size,
        request.include_all_preprocessed_columns,
    )?;
    let transcript = plan_cairo_blake2s_transcript(
        request.claim,
        request.pcs,
        discovery.lifting_log_size,
        discovery.dynamic_transcript_shape(),
    )?;
    let interaction = schema_zero_interaction_claim_for_composition(request.claim)?;
    let composition = plan_cairo_composition(
        request.claim,
        &CommonLookupElements::dummy(),
        &interaction,
        &request.preprocessed_trace.ids(),
        request.policy.composition_max_kernel_instrs,
    )?;
    let composition_bindings = compile_cairo_composition_binding_plan(
        request.claim,
        &CommonLookupElements::dummy(),
        &interaction,
        &request.preprocessed_trace.ids(),
        &composition,
    )?;
    let protocol = plan_protocol_geometry(
        request.proof_plan,
        request.claim,
        request.preprocessed_trace,
        &request.pcs,
        request.include_all_preprocessed_columns,
        request.policy,
        &transcript,
        &discovery,
        &composition,
    )?;
    let arena = Arc::new(match request.execution_tables {
        Some(geometry) => ProofArenaPlan::build_with_execution_tables(
            request.proof_plan,
            &protocol,
            &composition,
            geometry,
        )?,
        None => ProofArenaPlan::build(request.proof_plan, &protocol, &composition)?,
    });
    let admission = WorkspaceAdmission::compile(Arc::new(topology), &arena)?;
    Ok(ShapeExecutable {
        admission,
        discovery,
        transcript,
        composition,
        composition_bindings,
        arena,
    })
}

#[derive(Debug)]
pub enum ShapeExecutableError {
    ZeroCapacity,
    AtCapacity {
        capacity: usize,
        requested: [u8; 32],
    },
    EmptyTraceGeometry,
    SizeOverflow,
    InvalidLiftingLogSize {
        lifting: u32,
        required: u32,
    },
    NonCanonicalPreprocessedGeometry {
        variant: PreProcessedTraceVariant,
        supplied_columns: usize,
        expected_columns: usize,
        first_mismatch: usize,
    },
    Discovery(ProtocolDiscoveryError),
    Transcript(TranscriptPlanError),
    Composition(CompositionPlanError),
    Protocol(ProtocolPlanError),
    Arena(ArenaPlanError),
    MissingArenaSlot(stwo_backend_cuda::ArenaSlotId),
    ArenaViewCountMismatch {
        bound: usize,
        planned: usize,
    },
}

impl core::fmt::Display for ShapeExecutableError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "shape executable preparation failed: {self:?}")
    }
}

impl std::error::Error for ShapeExecutableError {}

macro_rules! convert_error {
    ($from:ty, $variant:ident) => {
        impl From<$from> for ShapeExecutableError {
            fn from(value: $from) -> Self {
                Self::$variant(value)
            }
        }
    };
}

convert_error!(ProtocolDiscoveryError, Discovery);
convert_error!(TranscriptPlanError, Transcript);
convert_error!(CompositionPlanError, Composition);
convert_error!(ProtocolPlanError, Protocol);
convert_error!(ArenaPlanError, Arena);

#[cfg(test)]
#[path = "shape_executable_tests.rs"]
mod tests;
