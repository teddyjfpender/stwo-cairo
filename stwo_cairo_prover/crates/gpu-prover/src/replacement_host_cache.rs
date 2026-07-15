//! Bounded immutable host planning cache for ReplacementV1.
//!
//! The cache key contains every structural value and canonical derived geometry
//! that can change the proof plan. Raw contents, PublicData, execution-memory
//! pointers, builtin starts, and transcript values are rebound from the current
//! owner.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use cairo_air::air::PublicData;
use cairo_air::claims::CairoClaim;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_adapter::builtins::MemorySegmentAddresses;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};

use crate::arena_plan::ExecutionTableGeometry;
use crate::plan::{ProofPlan, ProofPlanError};
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::recorded_witness_inputs::{
    PlannedRecordedWitnessInputs, RawRecordedWitnessTemplate, RecordedWitnessPlanError,
};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::resident_input::ResidentProverInputOwner;
use crate::resident_shape::{
    raw_replacement_compacted_geometry, raw_replacement_proof_plan, RawCompactedGeometry,
    RawResidentShapeError,
};
use crate::resident_witness::{
    planned_cairo_claim_from_public_data, require_strict_resident_witness_coverage,
    ResidentWitnessPlanError,
};
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::shape_executable::{
    ShapeCompileRequest, ShapeExecutable, ShapeExecutableCache, ShapeExecutableDynamicIdentity,
    ShapeExecutableError, ShapeExecutableSelection,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplacementHostCacheTelemetry {
    pub hits: u64,
    pub misses: u64,
    pub compilations: u64,
    pub evictions: u64,
    pub collisions: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementHostMaterialization {
    Compiled,
    Reused,
}

impl ReplacementHostMaterialization {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compiled => "compiled",
            Self::Reused => "reused",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementHostCacheProofAudit {
    pub materialization: ReplacementHostMaterialization,
    pub identity_ns: u128,
    pub select_ns: u128,
    pub telemetry: ReplacementHostCacheTelemetry,
}

pub struct ReplacementHostSelection {
    pub template: Arc<ReplacementHostTemplate>,
    pub audit: ReplacementHostCacheProofAudit,
}

pub struct ReplacementHostTemplate {
    identity: RawReplacementTopologyIdentity,
    preprocessed_trace: Arc<PreProcessedTrace>,
    capacity_plan: Arc<ProofPlan>,
    exact_plan: Arc<ProofPlan>,
    claim: RawCairoClaimTemplate,
    recorded: RawRecordedWitnessTemplate,
    shape_executable: Mutex<Option<ReplacementShapeExecutableHandle>>,
}

#[derive(Clone)]
struct ReplacementShapeExecutableHandle {
    dynamic: ShapeExecutableDynamicIdentity,
    executable: Arc<ShapeExecutable>,
}

/// Typed split of a Cairo claim at the only dynamic boundary exposed by the
/// canonical planner. `public_data` contains initial/final state, program and
/// output memory, safe-call ids, and every public segment range. Every other
/// claim field is presence/log-size geometry derived solely from `ProofPlan`.
struct RawCairoClaimTemplate {
    static_claim: CairoClaim,
}

impl RawCairoClaimTemplate {
    fn compile(exact_plan: &ProofPlan) -> Result<Self, ResidentWitnessPlanError> {
        Ok(Self {
            static_claim: planned_cairo_claim_from_public_data(&PublicData::default(), exact_plan)?,
        })
    }

    fn bind(&self, public_data: &PublicData) -> CairoClaim {
        let mut claim = self.static_claim.clone();
        claim.public_data = public_data.clone();
        claim
    }
}

impl ReplacementHostTemplate {
    pub fn preprocessed_trace(&self) -> &Arc<PreProcessedTrace> {
        &self.preprocessed_trace
    }

    pub fn capacity_plan(&self) -> &Arc<ProofPlan> {
        &self.capacity_plan
    }

    pub fn exact_plan(&self) -> &Arc<ProofPlan> {
        &self.exact_plan
    }

    pub fn recorded_lane_count(&self) -> usize {
        self.recorded.lane_count()
    }

    pub fn bind_claim(&self, public_data: &PublicData) -> CairoClaim {
        self.claim.bind(public_data)
    }

    pub fn bind_recorded(
        &self,
        owner: &ResidentProverInputOwner,
    ) -> Result<PlannedRecordedWitnessInputs, RecordedWitnessPlanError> {
        self.recorded.bind(owner)
    }

    /// Select the shape executable without rebuilding its full static topology
    /// on the ordinary warm path. Exact dynamic equality and exact live cache
    /// ownership are both required; every mismatch falls back to the full
    /// typed admission path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select_shape_executable(
        &self,
        cache: &mut ShapeExecutableCache,
        claim: &CairoClaim,
        preprocessed_trace: &Arc<PreProcessedTrace>,
        pcs: PcsConfig,
        include_all_preprocessed_columns: bool,
        execution_tables: Option<ExecutionTableGeometry>,
        policy: ProtocolPlanPolicy,
    ) -> Result<ShapeExecutableSelection, ShapeExecutableError> {
        let dynamic = ShapeExecutableDynamicIdentity::new(
            claim,
            pcs,
            include_all_preprocessed_columns,
            execution_tables,
            policy,
        )?;
        let template_preprocessed = Arc::ptr_eq(preprocessed_trace, &self.preprocessed_trace);
        let handle = if template_preprocessed {
            let handle_started = Instant::now();
            let handle = lock_handle(&self.shape_executable).clone();
            cache.record_replacement_handle_lock(handle_started.elapsed().as_nanos());
            handle
        } else {
            None
        };
        if let Some(handle) = handle.filter(|handle| handle.dynamic == dynamic) {
            if let Some(selection) = cache.bind_installed(&handle.executable, claim)? {
                return Ok(selection);
            }
        }

        let selection = cache.compile_or_bind(ShapeCompileRequest {
            claim,
            proof_plan: &self.exact_plan,
            preprocessed_trace,
            pcs,
            include_all_preprocessed_columns,
            execution_tables,
            policy,
        })?;
        if template_preprocessed {
            let update_started = Instant::now();
            *lock_handle(&self.shape_executable) = Some(ReplacementShapeExecutableHandle {
                dynamic,
                executable: Arc::clone(&selection.executable),
            });
            cache.record_replacement_handle_lock(update_started.elapsed().as_nanos());
        }
        Ok(selection)
    }
}

#[derive(Debug)]
pub enum ReplacementHostCacheError {
    ZeroCapacity,
    RawShape(RawResidentShapeError),
    ExactPlan(ProofPlanError),
    Claim(ResidentWitnessPlanError),
    Recorded(RecordedWitnessPlanError),
    DigestCollision { digest: [u8; 32] },
}

impl core::fmt::Display for ReplacementHostCacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "replacement host cache rejected: {self:?}")
    }
}

impl std::error::Error for ReplacementHostCacheError {}

impl From<RawResidentShapeError> for ReplacementHostCacheError {
    fn from(value: RawResidentShapeError) -> Self {
        Self::RawShape(value)
    }
}

impl From<ProofPlanError> for ReplacementHostCacheError {
    fn from(value: ProofPlanError) -> Self {
        Self::ExactPlan(value)
    }
}

impl From<ResidentWitnessPlanError> for ReplacementHostCacheError {
    fn from(value: ResidentWitnessPlanError) -> Self {
        Self::Claim(value)
    }
}

impl From<RecordedWitnessPlanError> for ReplacementHostCacheError {
    fn from(value: RecordedWitnessPlanError) -> Self {
        Self::Recorded(value)
    }
}

struct CacheEntry {
    digest: [u8; 32],
    template: Arc<ReplacementHostTemplate>,
}

pub struct ReplacementHostCache {
    capacity: usize,
    entries: Vec<CacheEntry>,
    telemetry: ReplacementHostCacheTelemetry,
}

impl ReplacementHostCache {
    pub fn new(capacity: usize) -> Result<Self, ReplacementHostCacheError> {
        if capacity == 0 {
            return Err(ReplacementHostCacheError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            telemetry: ReplacementHostCacheTelemetry::default(),
        })
    }

    pub const fn telemetry(&self) -> ReplacementHostCacheTelemetry {
        self.telemetry
    }

    pub fn compile_or_bind(
        &mut self,
        owner: &ResidentProverInputOwner,
        variant: PreProcessedTraceVariant,
        opt_n_id_to_big_components: Option<usize>,
    ) -> Result<ReplacementHostSelection, ReplacementHostCacheError> {
        let identity_started = Instant::now();
        let preprocessed_trace = self
            .entries
            .iter()
            .find(|entry| entry.template.preprocessed_trace.variant == variant)
            .map(|entry| Arc::clone(&entry.template.preprocessed_trace))
            .unwrap_or_else(|| Arc::new(variant.to_preprocessed_trace()));
        let identity = RawReplacementTopologyIdentity::new(
            owner,
            Arc::clone(&preprocessed_trace),
            opt_n_id_to_big_components,
        )?;
        let identity_ns = identity_started.elapsed().as_nanos();
        let digest = identity.digest();
        self.select(
            owner,
            preprocessed_trace,
            opt_n_id_to_big_components,
            identity,
            digest,
            identity_ns,
        )
    }

    #[cfg(test)]
    pub(crate) fn compile_or_bind_with_digest_for_test(
        &mut self,
        owner: &ResidentProverInputOwner,
        variant: PreProcessedTraceVariant,
        opt_n_id_to_big_components: Option<usize>,
        digest: [u8; 32],
    ) -> Result<ReplacementHostSelection, ReplacementHostCacheError> {
        let identity_started = Instant::now();
        let preprocessed_trace = self
            .entries
            .iter()
            .find(|entry| entry.template.preprocessed_trace.variant == variant)
            .map(|entry| Arc::clone(&entry.template.preprocessed_trace))
            .unwrap_or_else(|| Arc::new(variant.to_preprocessed_trace()));
        let identity = RawReplacementTopologyIdentity::new(
            owner,
            Arc::clone(&preprocessed_trace),
            opt_n_id_to_big_components,
        )?;
        let identity_ns = identity_started.elapsed().as_nanos();
        self.select(
            owner,
            preprocessed_trace,
            opt_n_id_to_big_components,
            identity,
            digest,
            identity_ns,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &mut self,
        owner: &ResidentProverInputOwner,
        preprocessed_trace: Arc<PreProcessedTrace>,
        opt_n_id_to_big_components: Option<usize>,
        identity: RawReplacementTopologyIdentity,
        digest: [u8; 32],
        identity_ns: u128,
    ) -> Result<ReplacementHostSelection, ReplacementHostCacheError> {
        let select_started = Instant::now();
        if let Some(index) = self.entries.iter().position(|entry| entry.digest == digest) {
            if self.entries[index].template.identity != identity {
                self.telemetry.collisions += 1;
                return Err(ReplacementHostCacheError::DigestCollision { digest });
            }
            let entry = self.entries.remove(index);
            let template = Arc::clone(&entry.template);
            self.entries.push(entry);
            self.telemetry.hits += 1;
            return Ok(ReplacementHostSelection {
                template,
                audit: ReplacementHostCacheProofAudit {
                    materialization: ReplacementHostMaterialization::Reused,
                    identity_ns,
                    select_ns: select_started.elapsed().as_nanos(),
                    telemetry: self.telemetry,
                },
            });
        }

        self.telemetry.misses += 1;
        let template = Arc::new(compile_template(
            owner,
            preprocessed_trace,
            opt_n_id_to_big_components,
            identity,
        )?);
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
            self.telemetry.evictions += 1;
        }
        self.entries.push(CacheEntry {
            digest,
            template: Arc::clone(&template),
        });
        self.telemetry.compilations += 1;
        Ok(ReplacementHostSelection {
            template,
            audit: ReplacementHostCacheProofAudit {
                materialization: ReplacementHostMaterialization::Compiled,
                identity_ns,
                select_ns: select_started.elapsed().as_nanos(),
                telemetry: self.telemetry,
            },
        })
    }
}

fn compile_template(
    owner: &ResidentProverInputOwner,
    preprocessed_trace: Arc<PreProcessedTrace>,
    opt_n_id_to_big_components: Option<usize>,
    identity: RawReplacementTopologyIdentity,
) -> Result<ReplacementHostTemplate, ReplacementHostCacheError> {
    let capacity_plan = Arc::new(raw_replacement_proof_plan(
        owner,
        Arc::clone(&preprocessed_trace),
        opt_n_id_to_big_components,
    )?);
    let exact_plan =
        Arc::new(capacity_plan.strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)?);
    require_strict_resident_witness_coverage(&exact_plan)?;
    let claim = RawCairoClaimTemplate::compile(&exact_plan)?;
    let recorded = RawRecordedWitnessTemplate::compile(owner, &exact_plan)?;
    Ok(ReplacementHostTemplate {
        identity,
        preprocessed_trace,
        capacity_plan,
        exact_plan,
        claim,
        recorded,
        shape_executable: Mutex::new(None),
    })
}

fn lock_handle(
    handle: &Mutex<Option<ReplacementShapeExecutableHandle>>,
) -> MutexGuard<'_, Option<ReplacementShapeExecutableHandle>> {
    // The critical section only clones or replaces a fully formed value, so a
    // prior panic cannot leave a partially initialized invariant to reject.
    handle
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawReplacementTopologyIdentity {
    variant: u8,
    opt_n_id_to_big_components: Option<usize>,
    pc_count: usize,
    casm_rows: Vec<usize>,
    generic_rows: usize,
    builtin_cells: [Option<usize>; 8],
    address_to_id_len: usize,
    f252_values_len: usize,
    small_values_len: usize,
    small_max: u128,
    log_small_value_capacity: u32,
    compacted_geometry: [RawCompactedGeometry; 3],
}

impl RawReplacementTopologyIdentity {
    fn new(
        owner: &ResidentProverInputOwner,
        preprocessed_trace: Arc<PreProcessedTrace>,
        opt_n_id_to_big_components: Option<usize>,
    ) -> Result<Self, RawResidentShapeError> {
        let builtin = owner.builtin_segments();
        let builtin_cells = [
            checked_cells("add_mod_builtin", builtin.add_mod_builtin)?,
            checked_cells("bitwise_builtin", builtin.bitwise_builtin)?,
            checked_cells("mul_mod_builtin", builtin.mul_mod_builtin)?,
            checked_cells("pedersen_builtin", builtin.pedersen_builtin)?,
            checked_cells("poseidon_builtin", builtin.poseidon_builtin)?,
            checked_cells("range_check96_builtin", builtin.range_check96_builtin)?,
            checked_cells("range_check_builtin", builtin.range_check_builtin)?,
            checked_cells("ec_op_builtin", builtin.ec_op_builtin)?,
        ];
        let memory = owner.execution_memory();
        Ok(Self {
            variant: match preprocessed_trace.variant {
                PreProcessedTraceVariant::Canonical => 0,
                PreProcessedTraceVariant::CanonicalWithoutPedersen => 1,
                PreProcessedTraceVariant::CanonicalSmall => 2,
            },
            opt_n_id_to_big_components,
            pc_count: owner.pc_count(),
            casm_rows: owner
                .casm_inputs()
                .map(|input| input.states.len())
                .collect(),
            generic_rows: owner.direct_input_rows("generic_opcode").unwrap_or(0),
            builtin_cells,
            address_to_id_len: memory.address_to_id.len(),
            f252_values_len: memory.f252_values.len(),
            small_values_len: memory.small_values.len(),
            small_max: memory.config.small_max,
            log_small_value_capacity: memory.config.log_small_value_capacity,
            compacted_geometry: raw_replacement_compacted_geometry(owner, preprocessed_trace)?,
        })
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"stwo-replacement-host-topology-v2");
        hasher.update(&[self.variant]);
        hash_optional_usize(&mut hasher, self.opt_n_id_to_big_components);
        hash_usize(&mut hasher, self.pc_count);
        hash_usize(&mut hasher, self.casm_rows.len());
        for &rows in &self.casm_rows {
            hash_usize(&mut hasher, rows);
        }
        hash_usize(&mut hasher, self.generic_rows);
        for cells in self.builtin_cells {
            hash_optional_usize(&mut hasher, cells);
        }
        hash_usize(&mut hasher, self.address_to_id_len);
        hash_usize(&mut hasher, self.f252_values_len);
        hash_usize(&mut hasher, self.small_values_len);
        hasher.update(&self.small_max.to_le_bytes());
        hasher.update(&self.log_small_value_capacity.to_le_bytes());
        for geometry in self.compacted_geometry {
            hash_usize(&mut hasher, geometry.component.len());
            hasher.update(geometry.component.as_bytes());
            match geometry.rows {
                Some(rows) => {
                    hasher.update(&[1]);
                    hasher.update(&rows.n_real_rows.to_le_bytes());
                    hasher.update(&rows.padded_rows.to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            };
        }
        *hasher.finalize().as_bytes()
    }
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn hash_optional_usize(hasher: &mut blake3::Hasher, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_usize(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn checked_cells(
    component: &'static str,
    segment: Option<MemorySegmentAddresses>,
) -> Result<Option<usize>, RawResidentShapeError> {
    segment
        .map(|segment| {
            segment
                .stop_ptr
                .checked_sub(segment.begin_addr)
                .ok_or(RawResidentShapeError::InvalidBuiltinSegment(component))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::{RawReplacementTopologyIdentity, ReplacementHostCache, ReplacementHostCacheError};
    use crate::resident_shape::{RawCompactedGeometry, RawCompactedRows};

    fn identity() -> RawReplacementTopologyIdentity {
        RawReplacementTopologyIdentity {
            variant: 0,
            opt_n_id_to_big_components: None,
            pc_count: 17,
            casm_rows: vec![16, 32],
            generic_rows: 0,
            builtin_cells: [Some(16), None, Some(32), None, None, None, None, None],
            address_to_id_len: 64,
            f252_values_len: 8,
            small_values_len: 12,
            small_max: 99,
            log_small_value_capacity: 20,
            compacted_geometry: [
                RawCompactedGeometry {
                    component: "verify_instruction",
                    rows: Some(RawCompactedRows {
                        n_real_rows: 17,
                        padded_rows: 32,
                    }),
                },
                RawCompactedGeometry {
                    component: "pedersen_aggregator_window_bits_18",
                    rows: None,
                },
                RawCompactedGeometry {
                    component: "poseidon_aggregator",
                    rows: None,
                },
            ],
        }
    }

    #[test]
    fn topology_digest_covers_every_identity_field() {
        let base = identity();
        assert_eq!(base.digest(), base.clone().digest());

        let mut variants = Vec::new();
        let mut value = base.clone();
        value.variant = 1;
        variants.push(value);
        let mut value = base.clone();
        value.opt_n_id_to_big_components = Some(3);
        variants.push(value);
        let mut value = base.clone();
        value.pc_count += 1;
        variants.push(value);
        let mut value = base.clone();
        value.casm_rows[0] += 1;
        variants.push(value);
        let mut value = base.clone();
        value.generic_rows = 1;
        variants.push(value);
        let mut value = base.clone();
        value.builtin_cells[1] = Some(16);
        variants.push(value);
        let mut value = base.clone();
        value.address_to_id_len += 1;
        variants.push(value);
        let mut value = base.clone();
        value.f252_values_len += 1;
        variants.push(value);
        let mut value = base.clone();
        value.small_values_len += 1;
        variants.push(value);
        let mut value = base.clone();
        value.small_max += 1;
        variants.push(value);
        let mut value = base.clone();
        value.log_small_value_capacity += 1;
        variants.push(value);
        let mut value = base.clone();
        value.compacted_geometry[0]
            .rows
            .as_mut()
            .unwrap()
            .n_real_rows += 1;
        variants.push(value);
        let mut value = base.clone();
        value.compacted_geometry[0]
            .rows
            .as_mut()
            .unwrap()
            .padded_rows *= 2;
        variants.push(value);
        let mut value = base.clone();
        value.compacted_geometry[0].rows = None;
        variants.push(value);

        for (index, value) in variants.into_iter().enumerate() {
            assert_ne!(base.digest(), value.digest(), "identity field {index}");
        }
    }

    #[test]
    fn cache_capacity_is_never_unbounded() {
        assert!(matches!(
            ReplacementHostCache::new(0),
            Err(ReplacementHostCacheError::ZeroCapacity)
        ));
    }
}
