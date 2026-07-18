//! Loaded admission for the executable fixed-table stage.
//!
//! The prepared graph owns every pointer/descriptor workspace. This binder
//! snapshots no device address into compiled identity; it only revalidates the
//! live graph, process registration and linked static build before launch.

use stwo_backend_cuda::{
    ArenaSlice, ArenaSlotId, DeviceArena, FixedTableSourceColumn, PreparedFixedTableGraph,
};

use super::*;
use crate::program_image::lower_compiled::loaded_base_binding::PreparedRegisteredFixedSource;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedFixedTableStage {
    lowered: LoweredFixedTableStage,
    linked: Vec<FixedTableMaterializerLinkedContract>,
    registered: Vec<PreparedRegisteredFixedSource>,
}

pub(super) fn bind_loaded(
    lowered: &LoweredFixedTableStage,
    target_sm: u32,
    arena: &DeviceArena,
    graphs: &[PreparedFixedTableGraph<'_>],
    registered: &[PreparedRegisteredFixedSource],
) -> Result<LoadedFixedTableStage, InvocationShapeError> {
    validate_internal(lowered)?;
    let expected_registered = usize::from(lowered.tables().iter().any(|table| {
        table
            .sources()
            .iter()
            .any(|source| matches!(source, LoweredFixedTableSource::Registered { .. }))
    }));
    if graphs.len() != lowered.tables().len() || registered.len() != expected_registered {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    let linked = lowered
        .tables()
        .iter()
        .zip(graphs)
        .map(|(table, graph)| {
            validate_graph(table, arena, graph, registered.first())?;
            table
                .contract()
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?
                .ok_or(InvocationShapeError::InvalidFixedTableAuthority)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LoadedFixedTableStage {
        lowered: lowered.clone(),
        linked,
        registered: registered.to_vec(),
    })
}

impl LoadedFixedTableStage {
    /// Revalidate every live binding, then enqueue the exact prepared sequence.
    /// Each graph launch remains allocation/copy/sync-free.
    pub(super) fn launch_all(
        &self,
        arena: &DeviceArena,
        graphs: &[PreparedFixedTableGraph<'_>],
    ) -> Result<(), InvocationShapeError> {
        if graphs.len() != self.lowered.tables().len()
            || self.linked.len() != self.lowered.tables().len()
        {
            return Err(InvocationShapeError::InvalidFixedTableBinding);
        }
        for ((table, linked), graph) in self.lowered.tables().iter().zip(&self.linked).zip(graphs) {
            linked
                .validate(table.contract())
                .map_err(|_| InvocationShapeError::InvalidFixedTableAuthority)?;
            validate_graph(table, arena, graph, self.registered.first())?;
        }
        for graph in graphs {
            graph
                .launch()
                .map_err(|_| InvocationShapeError::InvalidFixedTableBinding)?;
        }
        Ok(())
    }
}

fn validate_graph(
    table: &LoweredFixedTableMaterialization,
    arena: &DeviceArena,
    graph: &PreparedFixedTableGraph<'_>,
    registered: Option<&PreparedRegisteredFixedSource>,
) -> Result<(), InvocationShapeError> {
    if !graph.belongs_to(arena)
        || graph.contract() != table.contract()
        || graph.requirements() != table.contract().requirements()
        || graph.source_columns().len() != table.sources().len()
        || !graph.multiplicity_columns().is_empty()
        || !graph.lookup_outputs().is_empty()
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    for (source, actual) in table.sources().iter().zip(graph.source_columns()) {
        match (source, actual) {
            (
                LoweredFixedTableSource::Arena {
                    arena: expected, ..
                },
                FixedTableSourceColumn::Arena(actual),
            ) if slice_matches_binding(arena, *actual, *expected) => {}
            (
                LoweredFixedTableSource::Registered { read, .. },
                FixedTableSourceColumn::RegisteredPedersen(actual),
            ) => registered
                .ok_or(InvocationShapeError::InvalidFixedTableBinding)?
                .validate_read_column(read, *actual)?,
            _ => return Err(InvocationShapeError::InvalidFixedTableBinding),
        }
    }
    if !graph
        .multiplicity_slab()
        .is_some_and(|slice| slice_matches_binding(arena, slice, table.multiplicity()))
        || graph.trace_outputs().len() != table.trace_outputs().len()
        || graph
            .trace_outputs()
            .iter()
            .zip(table.trace_outputs())
            .any(|(&actual, &expected)| !slice_matches_binding(arena, actual, expected))
        || !graph
            .lookup_output_slab()
            .is_some_and(|slice| slice_matches_binding(arena, slice, table.lookup_output()))
    {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }

    let slots = table.workspace();
    let requirements = table.contract().requirements();
    if !optional_workspace_matches(
        arena,
        graph.source_pointers(),
        slots.source_pointers,
        requirements.source_pointer_words,
    ) || !workspace_matches(
        arena,
        graph.multiplicity_pointers(),
        slots.multiplicity_pointers,
        requirements.multiplicity_pointer_words,
    ) || !workspace_matches(
        arena,
        graph.trace_multiplicity_columns(),
        slots.trace_multiplicity_columns,
        requirements.trace_mapping_words,
    ) || !workspace_matches(
        arena,
        graph.trace_output_pointers(),
        slots.trace_output_pointers,
        requirements.trace_pointer_words,
    ) || !workspace_matches(
        arena,
        graph.lookup_descriptors(),
        slots.lookup_descriptors,
        requirements.lookup_descriptor_words,
    ) || !workspace_matches(
        arena,
        graph.lookup_output_pointers(),
        slots.lookup_output_pointers,
        requirements.lookup_pointer_words,
    ) {
        return Err(InvocationShapeError::InvalidFixedTableBinding);
    }
    Ok(())
}

fn optional_workspace_matches(
    arena: &DeviceArena,
    actual: Option<ArenaSlice>,
    expected: Option<ArenaSlotId>,
    words: usize,
) -> bool {
    match (actual, expected, words) {
        (None, None, 0) => true,
        (Some(actual), Some(expected), words) if words != 0 => {
            workspace_matches(arena, actual, expected, words)
        }
        _ => false,
    }
}

fn workspace_matches(
    arena: &DeviceArena,
    actual: ArenaSlice,
    expected: ArenaSlotId,
    words: usize,
) -> bool {
    arena.bind(expected).is_ok_and(|owner| {
        actual.id() == expected
            && actual.len_words() == words
            && actual.as_u32_ptr() == owner.as_u32_ptr()
            && actual.belongs_to(arena.context())
    })
}

fn slice_matches_binding(arena: &DeviceArena, actual: ArenaSlice, expected: ArenaBinding) -> bool {
    expected.len_words != 0
        && workspace_matches(arena, actual, expected.physical, expected.len_words)
}
