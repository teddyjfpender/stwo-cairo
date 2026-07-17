//! Closed-set runtime admission for the compiled Base witness prefix.
//!
//! Semantic authority remains address-free. This module admits one exact
//! prepared arena inventory, retains process-local binding evidence, and keeps
//! statement-varying host-ingress receipts separately refreshable.

use std::collections::BTreeMap;

use prepared_graphs::{
    bind_clear, bind_ec_op_segment, bind_execution_tables, bind_generic_feed,
    bind_public_memory_seed, LoadedClear, LoadedEcOpSegment, LoadedExecutionTables,
    LoadedGenericFeed, LoadedPublicMemorySeed,
};
use statement_state::{
    SourceAttemptCounter, StatementSourceKind, StatementSourceState, StatementSourceStateError,
};
use stwo_backend_cuda::{
    DeviceArena, EcOpSegmentStartReceipt, ExecutionTablesIngestReceipt, PreparedEcOpGraph,
    PreparedExecutionTablesGraph, PreparedWitnessFeedGraph, WitnessFeedSourceUploadReceipt,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::blake_g_direct_execution_authority::{
    NativeBlakeGDirectExecutionAuthority, NativeBlakeGDirectLinkedModuleAuthority,
};
use super::ec_op_execution_authority::NativeEcOpLinkedModuleAuthority;
use super::producer_prefix::{
    BaseProducerAuthority, PreparedBaseProducerInventory, PreparedGenericMultiplicityFeed,
    PreparedRecordedKernel, SemanticBaseProducer,
};
use super::{
    planned_recorded_component, BaseProducerAuthorityError, BaseProducerCatalog,
    InvocationShapeError,
};
use crate::arena_plan::ProofArenaPlan;

mod prepared_graphs;
mod registered_fixed_sources;
mod statement_state;

pub(crate) use registered_fixed_sources::PreparedRegisteredFixedSource;

impl From<StatementSourceStateError> for InvocationShapeError {
    fn from(_: StatementSourceStateError) -> Self {
        Self::InvalidProductionBaseAuthority
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProducerKey {
    component: &'static str,
    part: ProducerPartKey,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ProducerPartKey {
    Main,
    MemoryBig(u32),
    MemorySmall,
}

impl From<TracePartId> for ProducerPartKey {
    fn from(part: TracePartId) -> Self {
        match part {
            TracePartId::Main => Self::Main,
            TracePartId::MemoryBig(ordinal) => Self::MemoryBig(ordinal),
            TracePartId::MemorySmall => Self::MemorySmall,
        }
    }
}

fn exact_key(component: &'static str, part: TracePartId) -> ProducerKey {
    ProducerKey {
        component,
        part: part.into(),
    }
}

pub(crate) struct LoadedBaseProducerAuthority {
    execution_tables: Option<LoadedExecutionTables>,
    _clear: LoadedClear,
    public_memory_seed: Option<LoadedPublicMemorySeed>,
    _recorded: Vec<super::loaded_authority::LoadedRecordedWitnessAuthority>,
    _generic_feeds: Vec<LoadedGenericFeed>,
    _native_blake_g_direct: Option<NativeBlakeGDirectExecutionAuthority>,
    ec_op_segment: Option<LoadedEcOpSegment>,
    witness_inputs: Option<WitnessInputUploadGeneration>,
    static_transcript_inputs: Option<StaticTranscriptInputUploadGeneration>,
    reserved_witness_input: Option<WitnessInputUploadGeneration>,
    reserved_static_transcript_input: Option<StaticTranscriptInputUploadGeneration>,
    witness_input_attempts: SourceAttemptCounter,
    static_transcript_input_attempts: SourceAttemptCounter,
    // Address/source association; `_recorded` separately retains each exact
    // module publication/context/completion receipt.
    _registered_module_globals: registered_fixed_sources::LoadedRegisteredModuleGlobals,
    statement: StatementSourceState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WitnessInputUploadGeneration(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StaticTranscriptInputUploadGeneration(u64);

impl LoadedBaseProducerAuthority {
    /// Start a fresh transaction and invalidate every prior statement receipt.
    pub(crate) fn begin_statement_sources(&mut self) {
        self.clear_receipts();
        self.statement.begin();
    }

    /// Explicitly close a transaction after any later input refresh fails.
    pub(crate) fn invalidate_statement_sources(&mut self) {
        self.clear_receipts();
        self.statement.invalidate();
    }

    /// Burn and reserve a fresh witness attempt without admitting its source.
    /// A panic before post-fence completion therefore cannot become publishable.
    pub(crate) fn begin_witness_input_upload_attempt(
        &mut self,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = (|| {
            let generation = WitnessInputUploadGeneration(self.witness_input_attempts.begin()?);
            self.statement.reserve(StatementSourceKind::WitnessInputs)?;
            self.reserved_witness_input = Some(generation);
            Ok::<_, StatementSourceStateError>(())
        })();
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    pub(crate) fn complete_witness_input_upload(
        &mut self,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = (|| {
            let generation = self
                .reserved_witness_input
                .take()
                .ok_or(StatementSourceStateError)?;
            self.statement
                .admit(StatementSourceKind::WitnessInputs, generation.0)?;
            self.witness_inputs = Some(generation);
            Ok::<_, StatementSourceStateError>(())
        })();
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    /// Burn and reserve the final source only after every earlier source is
    /// exact. Admission and publication happen after its setup fence drains.
    pub(crate) fn begin_static_transcript_input_upload_attempt(
        &mut self,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = (|| {
            let generation = StaticTranscriptInputUploadGeneration(
                self.static_transcript_input_attempts.begin()?,
            );
            self.statement
                .reserve_final(StatementSourceKind::StaticTranscriptInputs)?;
            self.reserved_static_transcript_input = Some(generation);
            Ok::<_, StatementSourceStateError>(())
        })();
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    pub(crate) fn complete_static_transcript_input_upload(
        &mut self,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = (|| {
            let generation = self
                .reserved_static_transcript_input
                .take()
                .ok_or(StatementSourceStateError)?;
            self.statement
                .admit(StatementSourceKind::StaticTranscriptInputs, generation.0)?;
            self.static_transcript_inputs = Some(generation);
            self.statement.publish()
        })();
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    pub(crate) fn refresh_execution_tables(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedExecutionTablesGraph<'_>,
        receipt: ExecutionTablesIngestReceipt,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = self.refresh_execution_tables_inner(arena, graph, receipt);
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    fn refresh_execution_tables_inner(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedExecutionTablesGraph<'_>,
        receipt: ExecutionTablesIngestReceipt,
    ) -> Result<(), InvocationShapeError> {
        {
            let loaded = self
                .execution_tables
                .as_mut()
                .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
            loaded.receipt = None;
            loaded.validate_candidate(arena, graph, &receipt)?;
        }
        self.statement
            .admit(StatementSourceKind::ExecutionTables, receipt.generation())?;
        self.execution_tables
            .as_mut()
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?
            .receipt = Some(receipt);
        Ok(())
    }

    pub(crate) fn refresh_public_memory_seed(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedWitnessFeedGraph<'_>,
        receipt: WitnessFeedSourceUploadReceipt,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = self.refresh_public_memory_seed_inner(arena, graph, receipt);
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    fn refresh_public_memory_seed_inner(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedWitnessFeedGraph<'_>,
        receipt: WitnessFeedSourceUploadReceipt,
    ) -> Result<(), InvocationShapeError> {
        {
            let loaded = self
                .public_memory_seed
                .as_mut()
                .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
            loaded.receipt = None;
            loaded.validate_candidate(arena, graph, &receipt)?;
        }
        self.statement
            .admit(StatementSourceKind::PublicMemorySeed, receipt.generation())?;
        self.public_memory_seed
            .as_mut()
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?
            .receipt = Some(receipt);
        Ok(())
    }

    pub(crate) fn refresh_ec_op_segment_start(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedEcOpGraph<'_>,
        receipt: EcOpSegmentStartReceipt,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = self.refresh_ec_op_segment_start_inner(arena, graph, receipt);
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    fn refresh_ec_op_segment_start_inner(
        &mut self,
        arena: &DeviceArena,
        graph: &PreparedEcOpGraph<'_>,
        receipt: EcOpSegmentStartReceipt,
    ) -> Result<(), InvocationShapeError> {
        {
            let loaded = self
                .ec_op_segment
                .as_mut()
                .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
            loaded.receipt = None;
            loaded.validate_candidate(arena, graph, &receipt)?;
        }
        self.statement
            .admit(StatementSourceKind::EcOpSegmentStart, receipt.generation())?;
        self.ec_op_segment
            .as_mut()
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?
            .receipt = Some(receipt);
        Ok(())
    }

    /// Capture admission is intentionally non-consuming.
    pub(crate) fn validate_statement_sources(
        &mut self,
        arena: &DeviceArena,
        execution_tables: Option<&PreparedExecutionTablesGraph<'_>>,
        public_memory_seed: Option<&PreparedWitnessFeedGraph<'_>>,
        ec_op_segment: Option<&PreparedEcOpGraph<'_>>,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = self.validate_statement_sources_inner(
            arena,
            execution_tables,
            public_memory_seed,
            ec_op_segment,
        );
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    /// Validate every admitted source while the final transcript source is
    /// still absent. A panic here cannot leave the statement launch-ready.
    pub(crate) fn validate_before_static_transcript_upload(
        &mut self,
        arena: &DeviceArena,
        execution_tables: Option<&PreparedExecutionTablesGraph<'_>>,
        public_memory_seed: Option<&PreparedWitnessFeedGraph<'_>>,
        ec_op_segment: Option<&PreparedEcOpGraph<'_>>,
    ) -> Result<(), BaseProducerAuthorityError> {
        let result = self
            .statement
            .validate_before_final(
                StatementSourceKind::StaticTranscriptInputs,
                self.source_generations(),
            )
            .map_err(InvocationShapeError::from)
            .and_then(|_| {
                self.validate_source_candidates(
                    arena,
                    execution_tables,
                    public_memory_seed,
                    ec_op_segment,
                )
            });
        if result.is_err() {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    fn validate_statement_sources_inner(
        &self,
        arena: &DeviceArena,
        execution_tables: Option<&PreparedExecutionTablesGraph<'_>>,
        public_memory_seed: Option<&PreparedWitnessFeedGraph<'_>>,
        ec_op_segment: Option<&PreparedEcOpGraph<'_>>,
    ) -> Result<(), InvocationShapeError> {
        self.statement.validate_ready(self.source_generations())?;
        self.validate_source_candidates(arena, execution_tables, public_memory_seed, ec_op_segment)
    }

    fn validate_source_candidates(
        &self,
        arena: &DeviceArena,
        execution_tables: Option<&PreparedExecutionTablesGraph<'_>>,
        public_memory_seed: Option<&PreparedWitnessFeedGraph<'_>>,
        ec_op_segment: Option<&PreparedEcOpGraph<'_>>,
    ) -> Result<(), InvocationShapeError> {
        match (&self.execution_tables, execution_tables) {
            (Some(loaded), Some(graph)) => {
                let receipt = loaded
                    .receipt
                    .as_ref()
                    .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
                loaded.validate_candidate(arena, graph, receipt)?;
            }
            (None, None) => {}
            _ => return Err(InvocationShapeError::InvalidProductionBaseAuthority),
        }
        match (&self.public_memory_seed, public_memory_seed) {
            (Some(loaded), Some(graph)) => {
                let receipt = loaded
                    .receipt
                    .as_ref()
                    .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
                loaded.validate_candidate(arena, graph, receipt)?;
            }
            (None, None) => {}
            _ => return Err(InvocationShapeError::InvalidProductionBaseAuthority),
        }
        match (&self.ec_op_segment, ec_op_segment) {
            (Some(loaded), Some(graph)) => {
                let receipt = loaded
                    .receipt
                    .as_ref()
                    .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
                loaded.validate_candidate(arena, graph, receipt)?;
            }
            (None, None) => {}
            _ => return Err(InvocationShapeError::InvalidProductionBaseAuthority),
        }
        Ok(())
    }

    /// Validate all live graphs, then atomically consume this statement.
    pub(crate) fn consume_statement_sources(
        &mut self,
        arena: &DeviceArena,
        execution_tables: Option<&PreparedExecutionTablesGraph<'_>>,
        public_memory_seed: Option<&PreparedWitnessFeedGraph<'_>>,
        ec_op_segment: Option<&PreparedEcOpGraph<'_>>,
    ) -> Result<(), BaseProducerAuthorityError> {
        let generations = self.source_generations();
        let result = self
            .validate_statement_sources_inner(
                arena,
                execution_tables,
                public_memory_seed,
                ec_op_segment,
            )
            .and_then(|_| {
                self.statement
                    .consume(generations)
                    .map_err(InvocationShapeError::from)
            });
        if result.is_ok() {
            self.clear_receipts();
        } else {
            self.invalidate_statement_sources();
        }
        result.map_err(|_| BaseProducerAuthorityError)
    }

    fn source_generations(&self) -> [Option<u64>; 5] {
        [
            self.execution_tables
                .as_ref()
                .and_then(|loaded| loaded.receipt.map(|receipt| receipt.generation())),
            self.public_memory_seed
                .as_ref()
                .and_then(|loaded| loaded.receipt.map(|receipt| receipt.generation())),
            self.ec_op_segment
                .as_ref()
                .and_then(|loaded| loaded.receipt.map(|receipt| receipt.generation())),
            self.witness_inputs.map(|generation| generation.0),
            self.static_transcript_inputs.map(|generation| generation.0),
        ]
    }

    fn clear_receipts(&mut self) {
        if let Some(loaded) = &mut self.execution_tables {
            loaded.receipt = None;
        }
        if let Some(loaded) = &mut self.public_memory_seed {
            loaded.receipt = None;
        }
        if let Some(loaded) = &mut self.ec_op_segment {
            loaded.receipt = None;
        }
        self.witness_inputs = None;
        self.static_transcript_inputs = None;
        self.reserved_witness_input = None;
        self.reserved_static_transcript_input = None;
    }
}

pub(super) fn bind(
    authority: &BaseProducerAuthority,
    plan: &ProofArenaPlan,
    prepared: PreparedBaseProducerInventory<'_, '_>,
    device_ordinal: u32,
    sm_major: u32,
    sm_minor: u32,
    active_sm: u32,
) -> Result<LoadedBaseProducerAuthority, InvocationShapeError> {
    validate_presence(authority, &prepared)?;
    let registered_module_globals = registered_fixed_sources::bind_for_base(
        authority,
        prepared.registered_fixed_sources,
        sm_major,
        sm_minor,
    )?;
    let catalog = BaseProducerCatalog::compile(plan)?;
    let execution_tables = authority
        .execution_tables
        .as_ref()
        .zip(prepared.execution_tables)
        .map(|(lowered, graph)| bind_execution_tables(prepared.arena, lowered, graph, active_sm))
        .transpose()?;
    let clear = bind_clear(
        prepared.arena,
        &authority.multiplicity.clear,
        prepared.clear,
        active_sm,
    )?;
    let public_memory_seed = authority
        .multiplicity
        .public_memory_seed
        .as_ref()
        .zip(prepared.public_memory_seed)
        .map(|(lowered, supplied)| {
            bind_public_memory_seed(prepared.arena, lowered, supplied, active_sm)
        })
        .transpose()?;

    let mut recorded = index_recorded(prepared.recorded)?;
    let mut feeds = index_feeds(prepared.generic_feeds)?;
    let mut direct = prepared.blake_g_direct;
    let mut ec_op_segment = prepared.ec_op_segment;
    let mut loaded_recorded = Vec::with_capacity(recorded.len());
    let mut loaded_feeds = Vec::with_capacity(feeds.len());
    let mut loaded_direct = None;
    let mut loaded_ec_op_segment = None;

    for (producer, feed) in authority
        .producers
        .iter()
        .zip(&authority.multiplicity.after_producer)
    {
        match (producer, feed) {
            (SemanticBaseProducer::Recorded(writer), Some(lowered_feed)) => {
                let key = producer_key(writer.producer.component, writer.producer.part)?;
                let prepared_writer = recorded
                    .remove(&key)
                    .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
                let prepared_feed = feeds
                    .remove(&key)
                    .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
                if !core::ptr::eq(prepared_writer.arena, prepared.arena) {
                    return Err(InvocationShapeError::InvalidProductionBaseAuthority);
                }
                let planned = planned_recorded_component(plan, writer.producer)?;
                loaded_recorded.push(super::loaded_authority::require_prepared(
                    &writer.source,
                    &catalog,
                    planned,
                    prepared.arena,
                    prepared_writer.writer,
                    device_ordinal,
                    sm_major,
                    sm_minor,
                )?);
                loaded_feeds.push(bind_generic_feed(
                    prepared.arena,
                    lowered_feed,
                    prepared_writer,
                    prepared_feed,
                    active_sm,
                )?);
            }
            (SemanticBaseProducer::NativeBlakeGDirect { contract, .. }, None) => {
                if loaded_direct.is_some() {
                    return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
                }
                let supplied = direct
                    .take()
                    .ok_or(InvocationShapeError::MissingNativeBlakeGDirectAuthority)?;
                if !core::ptr::eq(supplied.arena, prepared.arena) {
                    return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
                }
                let linked = NativeBlakeGDirectLinkedModuleAuthority::bind_linked(
                    &contract.authority,
                    active_sm,
                )?
                .ok_or(InvocationShapeError::MissingNativeBlakeGDirectAuthority)?;
                loaded_direct =
                    Some(linked.bind_prepared(&contract.authority, contract, plan, supplied)?);
            }
            (SemanticBaseProducer::NativeEcOp { contract, .. }, None) => {
                if loaded_ec_op_segment.is_some() {
                    return Err(InvocationShapeError::InvalidNativeEcOpAuthority);
                }
                let supplied = ec_op_segment
                    .take()
                    .ok_or(InvocationShapeError::InvalidNativeEcOpBinding)?;
                let linked = NativeEcOpLinkedModuleAuthority::bind_linked(&contract.authority)?
                    .ok_or(InvocationShapeError::InvalidNativeEcOpAuthority)?;
                linked.validate_active_sm(active_sm)?;
                let execution = linked.bind_lowered(
                    &contract.authority,
                    &contract.invocation,
                    &contract.effect,
                )?;
                loaded_ec_op_segment = Some(bind_ec_op_segment(
                    prepared.arena,
                    &catalog,
                    contract,
                    execution,
                    supplied,
                )?);
            }
            _ => return Err(InvocationShapeError::InvalidProductionBaseAuthority),
        }
    }
    if !recorded.is_empty() || !feeds.is_empty() || direct.is_some() || ec_op_segment.is_some() {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let expected_sources = [
        execution_tables.is_some(),
        public_memory_seed.is_some(),
        loaded_ec_op_segment.is_some(),
        true,
        true,
    ];
    let mut loaded = LoadedBaseProducerAuthority {
        execution_tables,
        _clear: clear,
        public_memory_seed,
        _recorded: loaded_recorded,
        _generic_feeds: loaded_feeds,
        _native_blake_g_direct: loaded_direct,
        ec_op_segment: loaded_ec_op_segment,
        witness_inputs: None,
        static_transcript_inputs: None,
        reserved_witness_input: None,
        reserved_static_transcript_input: None,
        witness_input_attempts: SourceAttemptCounter::default(),
        static_transcript_input_attempts: SourceAttemptCounter::default(),
        _registered_module_globals: registered_module_globals,
        statement: StatementSourceState::new(expected_sources),
    };
    for (kind, generation) in [
        (
            StatementSourceKind::ExecutionTables,
            loaded.source_generations()[0],
        ),
        (
            StatementSourceKind::PublicMemorySeed,
            loaded.source_generations()[1],
        ),
        (
            StatementSourceKind::EcOpSegmentStart,
            loaded.source_generations()[2],
        ),
    ] {
        if let Some(generation) = generation {
            loaded.statement.admit(kind, generation)?;
        }
    }
    // Witness and static transcript bytes have not crossed their setup fences.
    // Construction therefore returns a collecting, never launch-ready, binding.
    Ok(loaded)
}

pub(super) fn prepare_registered_fixed_sources(
    authority: &BaseProducerAuthority,
) -> Result<Vec<PreparedRegisteredFixedSource>, InvocationShapeError> {
    registered_fixed_sources::prepare_for_base(authority)
}

fn validate_presence(
    authority: &BaseProducerAuthority,
    prepared: &PreparedBaseProducerInventory<'_, '_>,
) -> Result<(), InvocationShapeError> {
    if authority.execution_tables.is_some() != prepared.execution_tables.is_some()
        || authority.multiplicity.public_memory_seed.is_some()
            != prepared.public_memory_seed.is_some()
        || authority.producers.len() != authority.multiplicity.after_producer.len()
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let mut expected_recorded = 0usize;
    let mut expected_feeds = 0usize;
    let mut expected_direct = 0usize;
    let mut expected_ec_op = 0usize;
    for (producer, feed) in authority
        .producers
        .iter()
        .zip(&authority.multiplicity.after_producer)
    {
        match (producer, feed) {
            (SemanticBaseProducer::Recorded(_), Some(_)) => {
                expected_recorded += 1;
                expected_feeds += 1;
            }
            (SemanticBaseProducer::NativeBlakeGDirect { .. }, None) => expected_direct += 1,
            (SemanticBaseProducer::NativeEcOp { .. }, None) => expected_ec_op += 1,
            _ => return Err(InvocationShapeError::InvalidProductionBaseAuthority),
        }
    }
    if expected_recorded != prepared.recorded.len()
        || expected_feeds != prepared.generic_feeds.len()
        || expected_direct != usize::from(prepared.blake_g_direct.is_some())
        || expected_direct > 1
        || expected_ec_op != usize::from(prepared.ec_op_segment.is_some())
        || expected_ec_op > 1
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(())
}

fn index_recorded<'prepared, 'arena>(
    prepared: &[PreparedRecordedKernel<'prepared, 'arena>],
) -> Result<BTreeMap<ProducerKey, PreparedRecordedKernel<'prepared, 'arena>>, InvocationShapeError>
{
    let mut indexed = BTreeMap::new();
    for &writer in prepared {
        insert_unique(
            &mut indexed,
            exact_key(writer.component, writer.part),
            writer,
        )?;
    }
    Ok(indexed)
}

fn index_feeds<'prepared, 'arena>(
    prepared: &[PreparedGenericMultiplicityFeed<'prepared, 'arena>],
) -> Result<
    BTreeMap<ProducerKey, PreparedGenericMultiplicityFeed<'prepared, 'arena>>,
    InvocationShapeError,
> {
    let mut indexed = BTreeMap::new();
    for &feed in prepared {
        insert_unique(&mut indexed, exact_key(feed.component, feed.part), feed)?;
    }
    Ok(indexed)
}

fn producer_key(
    component: &'static str,
    part: Option<TracePartId>,
) -> Result<ProducerKey, InvocationShapeError> {
    part.map(|part| exact_key(component, part))
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)
}

fn insert_unique<K: Ord, V>(
    indexed: &mut BTreeMap<K, V>,
    key: K,
    value: V,
) -> Result<(), InvocationShapeError> {
    if indexed.contains_key(&key) {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    indexed.insert(key, value);
    Ok(())
}

#[cfg(test)]
#[path = "loaded_base_binding/tests.rs"]
mod tests;
