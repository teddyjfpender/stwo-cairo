//! Exact process-local bindings for prepared static Base graphs.

use stwo_backend_cuda::{
    ArenaSlice, DeviceArena, EcOpCompositeContract, EcOpSegmentStartReceipt,
    ExecutionTablesContract, ExecutionTablesIngestReceipt, ExecutionTablesLinkedContract,
    ExecutionTablesStage, PreparedEcOpGraph, PreparedExecutionTablesGraph,
    PreparedWitnessFeedClearGraph, PreparedWitnessFeedGraph, WitnessFeedClearContract,
    WitnessFeedClearLinkedContract, WitnessFeedContract, WitnessFeedLinkedContract,
    WitnessFeedSourceUploadReceipt,
};

use super::super::ec_op_execution_authority::NativeEcOpCompositeExecutionAuthority;
use super::super::ec_op_prefix::LoweredNativeEcOpContract;
use super::super::multiplicity_feed::{LoweredMultiplicityFeed, MultiplicityFeedOwner};
use super::super::producer_prefix::{
    PreparedEcOpSegment, PreparedGenericMultiplicityFeed, PreparedPublicMemorySeed,
    PreparedRecordedKernel,
};
use super::super::{ArenaCatalogRange, BaseProducerCatalog, InvocationShapeError};
use crate::arena_plan::ArenaBinding;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SliceBinding {
    slot: stwo_backend_cuda::ArenaSlotId,
    pointer: usize,
    words: usize,
}

fn exact_slice(
    arena: &DeviceArena,
    expected: ArenaBinding,
    actual: ArenaSlice,
) -> Result<SliceBinding, InvocationShapeError> {
    let canonical = arena
        .bind(expected.physical)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    if expected.len_words == 0
        || canonical.len_words() < expected.len_words
        || !actual.belongs_to(arena.context())
        || actual.id() != expected.physical
        || actual.len_words() != expected.len_words
        || actual.as_u32_ptr() != canonical.as_u32_ptr()
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(SliceBinding {
        slot: actual.id(),
        pointer: actual.as_u32_ptr() as usize,
        words: actual.len_words(),
    })
}

fn snapshot_slice(
    arena: &DeviceArena,
    actual: ArenaSlice,
) -> Result<SliceBinding, InvocationShapeError> {
    if !actual.belongs_to(arena.context()) {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    Ok(SliceBinding {
        slot: actual.id(),
        pointer: actual.as_u32_ptr() as usize,
        words: actual.len_words(),
    })
}

fn exact_slices(
    arena: &DeviceArena,
    expected: impl IntoIterator<Item = ArenaBinding>,
    actual: &[ArenaSlice],
) -> Result<Vec<SliceBinding>, InvocationShapeError> {
    let expected = expected.into_iter().collect::<Vec<_>>();
    if expected.len() != actual.len() {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    expected
        .into_iter()
        .zip(actual)
        .map(|(expected, &actual)| exact_slice(arena, expected, actual))
        .collect()
}

fn snapshot_slices(
    arena: &DeviceArena,
    actual: &[ArenaSlice],
) -> Result<Vec<SliceBinding>, InvocationShapeError> {
    actual
        .iter()
        .copied()
        .map(|slice| snapshot_slice(arena, slice))
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExecutionTablesBinding {
    raw: [SliceBinding; 3],
    big: Vec<SliceBinding>,
    small: Vec<SliceBinding>,
    table_pointers: SliceBinding,
    table_strides: SliceBinding,
}

impl ExecutionTablesBinding {
    fn exact(
        arena: &DeviceArena,
        lowered: &super::super::execution_tables::LoweredExecutionTables,
        graph: &PreparedExecutionTablesGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) || graph.contract() != &lowered.contract {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        let big = lowered
            .stages
            .iter()
            .find(|stage| stage.stage == ExecutionTablesStage::Big)
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        let small = lowered
            .stages
            .iter()
            .find(|stage| stage.stage == ExecutionTablesStage::Small)
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        Ok(Self {
            raw: [
                exact_slice(arena, lowered.host_ingress[0].arena, graph.raw_addr_to_id())?,
                exact_slice(arena, lowered.host_ingress[1].arena, graph.raw_f252_words())?,
                exact_slice(
                    arena,
                    lowered.host_ingress[2].arena,
                    graph.raw_small_words(),
                )?,
            ],
            big: exact_slices(
                arena,
                big.outputs.iter().map(|output| output.arena),
                graph.big_limbs(),
            )?,
            small: exact_slices(
                arena,
                small.outputs.iter().map(|output| output.arena),
                graph.small_limbs(),
            )?,
            table_pointers: exact_slice(
                arena,
                lowered.relocations.table_pointers,
                graph.table_pointers(),
            )?,
            table_strides: exact_slice(
                arena,
                lowered.relocations.table_strides,
                graph.table_strides(),
            )?,
        })
    }

    fn snapshot(
        arena: &DeviceArena,
        graph: &PreparedExecutionTablesGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        Ok(Self {
            raw: [
                snapshot_slice(arena, graph.raw_addr_to_id())?,
                snapshot_slice(arena, graph.raw_f252_words())?,
                snapshot_slice(arena, graph.raw_small_words())?,
            ],
            big: snapshot_slices(arena, graph.big_limbs())?,
            small: snapshot_slices(arena, graph.small_limbs())?,
            table_pointers: snapshot_slice(arena, graph.table_pointers())?,
            table_strides: snapshot_slice(arena, graph.table_strides())?,
        })
    }
}

pub(super) struct LoadedExecutionTables {
    contract: ExecutionTablesContract,
    linked: ExecutionTablesLinkedContract,
    binding: ExecutionTablesBinding,
    pub(super) receipt: Option<ExecutionTablesIngestReceipt>,
}

impl LoadedExecutionTables {
    pub(super) fn validate_candidate(
        &self,
        arena: &DeviceArena,
        graph: &PreparedExecutionTablesGraph<'_>,
        receipt: &ExecutionTablesIngestReceipt,
    ) -> Result<(), InvocationShapeError> {
        let linked = graph
            .contract()
            .bind_static_build(self.linked.target_sm())
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        if graph.contract() != &self.contract
            || linked != self.linked
            || ExecutionTablesBinding::snapshot(arena, graph)? != self.binding
            || !graph.ingest_is_current(receipt)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        Ok(())
    }
}

pub(super) fn bind_execution_tables(
    arena: &DeviceArena,
    lowered: &super::super::execution_tables::LoweredExecutionTables,
    graph: &PreparedExecutionTablesGraph<'_>,
    active_sm: u32,
) -> Result<LoadedExecutionTables, InvocationShapeError> {
    let linked = lowered
        .contract
        .bind_static_build(active_sm)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let receipt = graph
        .ingest_receipt()
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
    let loaded = LoadedExecutionTables {
        contract: lowered.contract.clone(),
        linked,
        binding: ExecutionTablesBinding::exact(arena, lowered, graph)?,
        receipt: Some(receipt),
    };
    loaded.validate_candidate(arena, graph, &receipt)?;
    Ok(loaded)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClearBinding {
    destinations: Vec<SliceBinding>,
    destination_pointers: SliceBinding,
    destination_lengths: SliceBinding,
}

pub(super) struct LoadedClear {
    _contract: WitnessFeedClearContract,
    _linked: WitnessFeedClearLinkedContract,
    _binding: ClearBinding,
}

pub(super) fn bind_clear(
    arena: &DeviceArena,
    lowered: &super::super::multiplicity_clear::LoweredMultiplicityClear,
    graph: &PreparedWitnessFeedClearGraph<'_>,
    active_sm: u32,
) -> Result<LoadedClear, InvocationShapeError> {
    if !graph.belongs_to(arena) || graph.contract() != &lowered.contract {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let linked = lowered
        .contract
        .bind_static_build(active_sm)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LoadedClear {
        _contract: lowered.contract.clone(),
        _linked: linked,
        _binding: ClearBinding {
            destinations: exact_slices(
                arena,
                lowered
                    .destinations
                    .iter()
                    .map(|destination| destination.arena),
                graph.destinations(),
            )?,
            destination_pointers: exact_slice(
                arena,
                lowered.relocations.destination_pointers,
                graph.destination_pointers(),
            )?,
            destination_lengths: exact_slice(
                arena,
                lowered.relocations.destination_lengths,
                graph.destination_lengths(),
            )?,
        },
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FeedBinding {
    source: SliceBinding,
    descriptors: SliceBinding,
    luts: Vec<SliceBinding>,
    lut_pointers: SliceBinding,
    destinations: Vec<SliceBinding>,
    multiplicity_pointers: SliceBinding,
}

impl FeedBinding {
    fn exact(
        arena: &DeviceArena,
        lowered: &LoweredMultiplicityFeed,
        graph: &PreparedWitnessFeedGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) || graph.contract() != &lowered.contract {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        Ok(Self {
            source: exact_slice(arena, lowered.source.arena, graph.source())?,
            descriptors: exact_slice(
                arena,
                lowered.relocations.descriptor_workspace,
                graph.descriptors(),
            )?,
            luts: exact_slices(
                arena,
                lowered.luts.iter().map(|lut| lut.input.arena),
                graph.lut_tables(),
            )?,
            lut_pointers: exact_slice(
                arena,
                lowered.relocations.lut_pointers,
                graph.lut_pointers(),
            )?,
            destinations: exact_slices(
                arena,
                lowered
                    .destinations
                    .iter()
                    .map(|destination| destination.arena),
                graph.multiplicity_destinations(),
            )?,
            multiplicity_pointers: exact_slice(
                arena,
                lowered.relocations.multiplicity_pointers,
                graph.multiplicity_pointers(),
            )?,
        })
    }

    fn snapshot(
        arena: &DeviceArena,
        graph: &PreparedWitnessFeedGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        Ok(Self {
            source: snapshot_slice(arena, graph.source())?,
            descriptors: snapshot_slice(arena, graph.descriptors())?,
            luts: snapshot_slices(arena, graph.lut_tables())?,
            lut_pointers: snapshot_slice(arena, graph.lut_pointers())?,
            destinations: snapshot_slices(arena, graph.multiplicity_destinations())?,
            multiplicity_pointers: snapshot_slice(arena, graph.multiplicity_pointers())?,
        })
    }
}

pub(super) struct LoadedGenericFeed {
    _contract: WitnessFeedContract,
    _linked: WitnessFeedLinkedContract,
    _binding: FeedBinding,
}

pub(super) fn bind_generic_feed(
    arena: &DeviceArena,
    lowered: &LoweredMultiplicityFeed,
    writer: PreparedRecordedKernel<'_, '_>,
    feed: PreparedGenericMultiplicityFeed<'_, '_>,
    active_sm: u32,
) -> Result<LoadedGenericFeed, InvocationShapeError> {
    let expected_owner = MultiplicityFeedOwner::Recorded {
        component: feed.component,
        part: feed.part,
    };
    if lowered.owner != expected_owner
        || writer.component != feed.component
        || writer.part != feed.part
        || graph_slice(feed.graph.source()) != graph_slice(writer.writer.sub_words())
        || feed.graph.source_upload_receipt().is_some()
    {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let linked = lowered
        .contract
        .bind_static_build(active_sm)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    Ok(LoadedGenericFeed {
        _contract: lowered.contract.clone(),
        _linked: linked,
        _binding: FeedBinding::exact(arena, lowered, feed.graph)?,
    })
}

pub(super) struct LoadedPublicMemorySeed {
    contract: WitnessFeedContract,
    linked: WitnessFeedLinkedContract,
    binding: FeedBinding,
    pub(super) receipt: Option<WitnessFeedSourceUploadReceipt>,
}

impl LoadedPublicMemorySeed {
    pub(super) fn validate_candidate(
        &self,
        arena: &DeviceArena,
        graph: &PreparedWitnessFeedGraph<'_>,
        receipt: &WitnessFeedSourceUploadReceipt,
    ) -> Result<(), InvocationShapeError> {
        let linked = graph
            .contract()
            .bind_static_build(self.linked.target_sm())
            .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
        if graph.contract() != &self.contract
            || linked != self.linked
            || FeedBinding::snapshot(arena, graph)? != self.binding
            || !graph.source_upload_is_current(receipt)
        {
            return Err(InvocationShapeError::InvalidProductionBaseAuthority);
        }
        Ok(())
    }
}

pub(super) fn bind_public_memory_seed(
    arena: &DeviceArena,
    lowered: &LoweredMultiplicityFeed,
    supplied: PreparedPublicMemorySeed<'_, '_>,
    active_sm: u32,
) -> Result<LoadedPublicMemorySeed, InvocationShapeError> {
    if lowered.owner != MultiplicityFeedOwner::PublicMemorySeed {
        return Err(InvocationShapeError::InvalidProductionBaseAuthority);
    }
    let linked = lowered
        .contract
        .bind_static_build(active_sm)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
        .ok_or(InvocationShapeError::InvalidProductionBaseAuthority)?;
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?;
    let loaded = LoadedPublicMemorySeed {
        contract: lowered.contract.clone(),
        linked,
        binding: FeedBinding::exact(arena, lowered, supplied.graph)?,
        receipt: Some(supplied.receipt),
    };
    loaded.validate_candidate(arena, supplied.graph, &supplied.receipt)?;
    Ok(loaded)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EcOpBinding {
    execution_table_pointers: SliceBinding,
    segment_start: SliceBinding,
    trace_columns: Vec<SliceBinding>,
    lookup_words: SliceBinding,
    partial_input_columns: Vec<SliceBinding>,
    multiplicity_destinations: Vec<SliceBinding>,
}

impl EcOpBinding {
    fn exact(
        arena: &DeviceArena,
        catalog: &BaseProducerCatalog,
        lowered: &LoweredNativeEcOpContract,
        graph: &PreparedEcOpGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) || graph.contract() != &lowered.authority {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        Ok(Self {
            execution_table_pointers: exact_slice(
                arena,
                range_binding(catalog, &lowered.invocation.execution_table_pointers)?,
                graph.execution_table_pointers(),
            )?,
            segment_start: exact_slice(
                arena,
                range_binding(catalog, &lowered.invocation.segment_start.value)?,
                graph.segment_start_source(),
            )?,
            trace_columns: exact_slices(
                arena,
                lowered
                    .invocation
                    .trace_columns
                    .iter()
                    .map(|binding| range_binding(catalog, &binding.value))
                    .collect::<Result<Vec<_>, _>>()?,
                graph.trace_columns(),
            )?,
            lookup_words: exact_slice(
                arena,
                range_binding(catalog, &lowered.invocation.lookup_words.value)?,
                graph.lookup_words(),
            )?,
            partial_input_columns: exact_slices(
                arena,
                lowered
                    .invocation
                    .partial_input_columns
                    .iter()
                    .map(|binding| range_binding(catalog, &binding.value))
                    .collect::<Result<Vec<_>, _>>()?,
                graph.partial_input_columns(),
            )?,
            multiplicity_destinations: exact_slices(
                arena,
                lowered
                    .invocation
                    .multiplicities
                    .iter()
                    .map(|binding| range_binding(catalog, &binding.value))
                    .collect::<Result<Vec<_>, _>>()?,
                &graph.multiplicity_destinations(),
            )?,
        })
    }

    fn snapshot(
        arena: &DeviceArena,
        graph: &PreparedEcOpGraph<'_>,
    ) -> Result<Self, InvocationShapeError> {
        if !graph.belongs_to(arena) {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        Ok(Self {
            execution_table_pointers: snapshot_slice(arena, graph.execution_table_pointers())?,
            segment_start: snapshot_slice(arena, graph.segment_start_source())?,
            trace_columns: snapshot_slices(arena, graph.trace_columns())?,
            lookup_words: snapshot_slice(arena, graph.lookup_words())?,
            partial_input_columns: snapshot_slices(arena, graph.partial_input_columns())?,
            multiplicity_destinations: snapshot_slices(arena, &graph.multiplicity_destinations())?,
        })
    }
}

fn range_binding(
    catalog: &BaseProducerCatalog,
    range: &ArenaCatalogRange,
) -> Result<ArenaBinding, InvocationShapeError> {
    let value = catalog.value(range.value)?;
    if range.value_words.start != 0
        || range.value_words.end == 0
        || range.value_words.end > value.words
    {
        return Err(InvocationShapeError::InvalidNativeEcOpBinding);
    }
    Ok(ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: range.value_words.end,
    })
}

pub(super) struct LoadedEcOpSegment {
    contract: EcOpCompositeContract,
    _execution: NativeEcOpCompositeExecutionAuthority,
    binding: EcOpBinding,
    pub(super) receipt: Option<EcOpSegmentStartReceipt>,
}

impl LoadedEcOpSegment {
    pub(super) fn validate_candidate(
        &self,
        arena: &DeviceArena,
        graph: &PreparedEcOpGraph<'_>,
        receipt: &EcOpSegmentStartReceipt,
    ) -> Result<(), InvocationShapeError> {
        if graph.contract() != &self.contract
            || EcOpBinding::snapshot(arena, graph)? != self.binding
            || !graph.segment_start_is_current(receipt)
        {
            return Err(InvocationShapeError::InvalidNativeEcOpBinding);
        }
        Ok(())
    }
}

pub(super) fn bind_ec_op_segment(
    arena: &DeviceArena,
    catalog: &BaseProducerCatalog,
    lowered: &LoweredNativeEcOpContract,
    execution: NativeEcOpCompositeExecutionAuthority,
    supplied: PreparedEcOpSegment<'_, '_>,
) -> Result<LoadedEcOpSegment, InvocationShapeError> {
    execution.validate(&lowered.authority, &lowered.invocation, &lowered.effect)?;
    let loaded = LoadedEcOpSegment {
        contract: lowered.authority.clone(),
        _execution: execution,
        binding: EcOpBinding::exact(arena, catalog, lowered, supplied.graph)?,
        receipt: Some(supplied.receipt),
    };
    loaded.validate_candidate(arena, supplied.graph, &supplied.receipt)?;
    Ok(loaded)
}

fn graph_slice(slice: ArenaSlice) -> (stwo_backend_cuda::ArenaSlotId, usize, usize) {
    (slice.id(), slice.as_u32_ptr() as usize, slice.len_words())
}
