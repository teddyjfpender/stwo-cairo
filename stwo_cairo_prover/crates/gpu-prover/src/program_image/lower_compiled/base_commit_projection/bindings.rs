//! Exact arena-role inventory for the BaseCommit authority.

use std::collections::BTreeMap;

use stwo_backend_cuda::{
    ArenaSlotId, BaseCommitDependencyRange, BaseCommitDependencyRole, BaseCommitProgramAuthority,
    BaseCommitValueRole, ModeAwareCommitWorkspaceSlots, ProgressiveCommitStorageMode,
};

use super::*;
use crate::arena_plan::{
    BufferLifetime, BufferPurpose, CommitmentColumnSource, PlannedCommitment, ProofEpoch,
};
use crate::compiled_proof::ElementRange;

#[derive(Clone, Debug)]
struct ExactArenaValue {
    catalog: ArenaCatalogValueId,
    arena: ArenaBinding,
}

#[derive(Clone, Debug)]
pub(super) struct BaseCommitInventory {
    roles: BTreeMap<BaseCommitValueRole, ExactArenaValue>,
    inverse_twiddles: ExactArenaValue,
    forward_twiddles: ExactArenaValue,
    pointer_tables: BTreeMap<BaseCommitDependencyRole, ArenaSlotId>,
    state_alignment_words: usize,
}

impl BaseCommitInventory {
    pub(super) fn compile(
        arena: &ProofArenaPlan,
        planned: &PlannedCommitment,
        authority: &BaseCommitProgramAuthority,
    ) -> Result<Self, InvocationShapeError> {
        if planned.storage_mode != ProgressiveCommitStorageMode::InPlaceSlab
            || planned.grouped_column_sources.len() != planned.grouped_column_log_sizes.len()
            || planned.grouped_column_sources.len() != planned.evaluation_output_groups.len()
        {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        let catalog = BaseProducerCatalog::compile(arena)?;
        let mut roles = source_and_retained_roles(arena, &catalog, planned)?;
        append_retained_hash_roles(arena, &catalog, planned, authority, &mut roles)?;
        let (state, pointer_tables) = state_and_pointer_tables(arena, &catalog, planned)?;
        append_state_backed_roles(authority, &mut roles, &state)?;
        let (inverse_twiddles, forward_twiddles) = twiddles(arena, &catalog, planned)?;
        let state_slot = arena
            .layout()
            .slot(state.arena.physical)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
        let inventory = Self {
            roles,
            inverse_twiddles,
            forward_twiddles,
            pointer_tables,
            state_alignment_words: state_slot.alignment_words,
        };
        inventory.validate_layouts(arena, authority)?;
        Ok(inventory)
    }

    pub(super) fn role(
        &self,
        role: BaseCommitValueRole,
    ) -> Result<(ArenaCatalogValueId, ArenaBinding), InvocationShapeError> {
        let exact = self
            .roles
            .get(&role)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
        Ok((exact.catalog, exact.arena))
    }

    pub(super) const fn state_alignment_words(&self) -> usize {
        self.state_alignment_words
    }

    pub(super) fn install_external_inputs(
        &self,
        values: &mut adapter::SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        for (&role, exact) in &self.roles {
            if matches!(role, BaseCommitValueRole::SourceEvaluation { .. }) {
                values.version(exact.catalog)?;
            }
        }
        values.extend_ordered([self.inverse_twiddles.catalog, self.forward_twiddles.catalog])
    }

    pub(super) fn installed_value(
        &self,
        role: BaseCommitDependencyRole,
        range: BaseCommitDependencyRange,
    ) -> Result<(ArenaCatalogValueId, ArenaBinding, ElementRange), InvocationShapeError> {
        let exact = match role {
            BaseCommitDependencyRole::InverseTwiddles => &self.inverse_twiddles,
            BaseCommitDependencyRole::ForwardTwiddles => &self.forward_twiddles,
            _ => return Err(InvocationShapeError::InvalidBaseCommitBinding),
        };
        let elements = dependency_elements(exact.arena.len_words, range)?;
        Ok((exact.catalog, exact.arena, elements))
    }

    pub(super) fn validate_pointer_table(
        &self,
        arena: &ProofArenaPlan,
        role: BaseCommitDependencyRole,
        range: BaseCommitDependencyRange,
    ) -> Result<(), InvocationShapeError> {
        let slot = *self
            .pointer_tables
            .get(&role)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
        let capacity = arena
            .layout()
            .slot(slot)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?
            .len_words;
        dependency_elements(capacity, range).map(|_| ())
    }

    fn validate_layouts(
        &self,
        arena: &ProofArenaPlan,
        authority: &BaseCommitProgramAuthority,
    ) -> Result<(), InvocationShapeError> {
        for layout in authority.layouts() {
            let (_, binding) = self.role(layout.role)?;
            let retained_exact = matches!(
                layout.role,
                BaseCommitValueRole::SourceEvaluation { .. }
                    | BaseCommitValueRole::RetainedStageTwo { .. }
                    | BaseCommitValueRole::RetainedEvaluation { .. }
            ) || authority
                .retained_layers_bottom_up()
                .iter()
                .any(|retained| retained.role == layout.role);
            if (retained_exact && binding.len_words != layout.logical_words)
                || (!retained_exact && binding.len_words < layout.logical_words)
            {
                return Err(InvocationShapeError::InvalidBaseCommitBinding);
            }
            let slot = arena
                .layout()
                .slot(binding.physical)
                .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
            if slot.len_words < binding.len_words
                || slot.alignment_words < layout.alignment_words
                || slot.offset_words % layout.alignment_words != 0
            {
                return Err(InvocationShapeError::InvalidBaseCommitBinding);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn source_catalogs(&self) -> Vec<ArenaCatalogValueId> {
        self.roles
            .iter()
            .filter_map(|(&role, exact)| {
                matches!(role, BaseCommitValueRole::SourceEvaluation { .. })
                    .then_some(exact.catalog)
            })
            .collect()
    }
}

fn source_and_retained_roles(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedCommitment,
) -> Result<BTreeMap<BaseCommitValueRole, ExactArenaValue>, InvocationShapeError> {
    let mut roles = BTreeMap::new();
    let mut canonical = 0u32;
    for ((sources, logs), outputs) in planned
        .grouped_column_sources
        .iter()
        .zip(&planned.grouped_column_log_sizes)
        .zip(&planned.evaluation_output_groups)
    {
        let outputs = outputs
            .as_ref()
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
        if sources.len() != logs.len() || sources.len() != outputs.len() {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        for ((&source, &log_size), &output) in sources.iter().zip(logs).zip(outputs) {
            let source = source_evaluation(arena, catalog, source, log_size)?;
            let retained_words = pow2(
                log_size
                    .checked_add(planned.config.log_blowup_factor)
                    .ok_or(InvocationShapeError::SizeOverflow)?,
            )?;
            let retained = exact_binding(
                arena,
                catalog,
                output,
                BufferPurpose::CommitRetainedEvaluation,
                retained_words,
            )?;
            insert_role(
                &mut roles,
                BaseCommitValueRole::SourceEvaluation {
                    canonical_column: canonical,
                },
                source,
            )?;
            insert_role(
                &mut roles,
                BaseCommitValueRole::RetainedStageTwo {
                    canonical_column: canonical,
                },
                retained.clone(),
            )?;
            insert_role(
                &mut roles,
                BaseCommitValueRole::RetainedEvaluation {
                    canonical_column: canonical,
                },
                retained,
            )?;
            canonical = canonical
                .checked_add(1)
                .ok_or(InvocationShapeError::SizeOverflow)?;
        }
    }
    Ok(roles)
}

fn append_retained_hash_roles(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedCommitment,
    authority: &BaseCommitProgramAuthority,
    roles: &mut BTreeMap<BaseCommitValueRole, ExactArenaValue>,
) -> Result<(), InvocationShapeError> {
    if authority.retained_layers_bottom_up().len() != planned.retained_layers_bottom_up.len() {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    for (retained, &binding) in authority
        .retained_layers_bottom_up()
        .iter()
        .zip(&planned.retained_layers_bottom_up)
    {
        if !matches!(retained.role, BaseCommitValueRole::HashLayer { .. }) {
            return Err(InvocationShapeError::InvalidBaseCommitBinding);
        }
        let exact = exact_binding(
            arena,
            catalog,
            binding,
            BufferPurpose::RetainedMerkleLayers,
            retained.words,
        )?;
        insert_role(roles, retained.role, exact)?;
    }
    let root = roles
        .get(&authority.root())
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if root.arena != planned.root {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(())
}

/// Materialize the authority's exact transient State/Hash role set. There is
/// no wildcard state-slab fallback: a future role must first appear in the
/// validated authority layout and is then bound explicitly here.
fn append_state_backed_roles(
    authority: &BaseCommitProgramAuthority,
    roles: &mut BTreeMap<BaseCommitValueRole, ExactArenaValue>,
    state: &ExactArenaValue,
) -> Result<(), InvocationShapeError> {
    for layout in authority.layouts() {
        match layout.role {
            BaseCommitValueRole::State { .. } => {
                insert_role(roles, layout.role, state.clone())?;
            }
            BaseCommitValueRole::HashLayer { .. } if !roles.contains_key(&layout.role) => {
                insert_role(roles, layout.role, state.clone())?;
            }
            BaseCommitValueRole::SourceEvaluation { .. }
            | BaseCommitValueRole::RetainedStageTwo { .. }
            | BaseCommitValueRole::RetainedEvaluation { .. }
            | BaseCommitValueRole::HashLayer { .. } => {}
        }
    }
    Ok(())
}

fn state_and_pointer_tables(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedCommitment,
) -> Result<
    (
        ExactArenaValue,
        BTreeMap<BaseCommitDependencyRole, ArenaSlotId>,
    ),
    InvocationShapeError,
> {
    let ModeAwareCommitWorkspaceSlots::DomainProgressive(slots) = &planned.slots else {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    };
    let mut state_candidates = arena.logical_buffers().iter().filter_map(|logical| {
        let binding = arena.binding(logical.id)?;
        (logical.purpose == BufferPurpose::CommitProgressiveStatePing
            && logical.lifetime == BufferLifetime::at(ProofEpoch::BaseCommit)
            && binding.physical == slots.leaves.state_ping)
            .then_some(binding)
    });
    let state_binding = state_candidates
        .next()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if state_candidates.next().is_some() {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    let state = exact_binding(
        arena,
        catalog,
        state_binding,
        BufferPurpose::CommitProgressiveStatePing,
        state_binding.len_words,
    )?;
    let mut pointer_tables = BTreeMap::new();
    for (batch_index, batch) in slots.leaves.batches.iter().enumerate() {
        let batch_index =
            u32::try_from(batch_index).map_err(|_| InvocationShapeError::SizeOverflow)?;
        for (role, slot) in [
            (
                BaseCommitDependencyRole::BatchSourcePointerTable { batch_index },
                batch.coefficient_ptrs,
            ),
            (
                BaseCommitDependencyRole::BatchRetainedPointerTable { batch_index },
                batch.output_ptrs,
            ),
        ] {
            if pointer_tables.insert(role, slot).is_some() {
                return Err(InvocationShapeError::InvalidBaseCommitBinding);
            }
        }
    }
    Ok((state, pointer_tables))
}

fn twiddles(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    planned: &PlannedCommitment,
) -> Result<(ExactArenaValue, ExactArenaValue), InvocationShapeError> {
    let (_, inverse) = arena
        .find(None, None, BufferPurpose::InverseTwiddles, 0)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    let (_, forward) = arena
        .find(None, None, BufferPurpose::ForwardTwiddles, 0)
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if forward != planned.twiddles || inverse.physical == forward.physical {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok((
        exact_binding(
            arena,
            catalog,
            inverse,
            BufferPurpose::InverseTwiddles,
            inverse.len_words,
        )?,
        exact_binding(
            arena,
            catalog,
            forward,
            BufferPurpose::ForwardTwiddles,
            forward.len_words,
        )?,
    ))
}

fn source_evaluation(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    source: CommitmentColumnSource,
    log_size: u32,
) -> Result<ExactArenaValue, InvocationShapeError> {
    let CommitmentColumnSource::Trace {
        component,
        part,
        purpose: BufferPurpose::BaseCoefficients,
        ordinal,
    } = source
    else {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    };
    let expected_words = pow2(log_size)?;
    let mut matches = catalog.values.iter().filter(|value| {
        value.component == Some(component)
            && value.part == Some(part)
            && value.purpose == BufferPurpose::BaseTrace
            && value.ordinal == ordinal
            && value.words == expected_words
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    exact_binding(
        arena,
        catalog,
        arena
            .binding(value.logical)
            .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?,
        BufferPurpose::BaseTrace,
        expected_words,
    )
}

fn exact_binding(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    binding: ArenaBinding,
    purpose: BufferPurpose,
    words: usize,
) -> Result<ExactArenaValue, InvocationShapeError> {
    let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
    if value.logical != binding.logical
        || value.physical != binding.physical
        || value.purpose != purpose
        || value.words != words
        || binding.len_words != words
        || arena.binding(binding.logical) != Some(binding)
    {
        return Err(InvocationShapeError::InvalidBaseCommitBinding);
    }
    Ok(ExactArenaValue {
        catalog: value.id,
        arena: binding,
    })
}

fn insert_role(
    roles: &mut BTreeMap<BaseCommitValueRole, ExactArenaValue>,
    role: BaseCommitValueRole,
    value: ExactArenaValue,
) -> Result<(), InvocationShapeError> {
    roles
        .insert(role, value)
        .is_none()
        .then_some(())
        .ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}

fn dependency_elements(
    capacity: usize,
    range: BaseCommitDependencyRange,
) -> Result<ElementRange, InvocationShapeError> {
    let (start, end) = match range {
        BaseCommitDependencyRange::Whole { words } if words == capacity => (0, words),
        BaseCommitDependencyRange::Suffix { words } => (
            capacity
                .checked_sub(words)
                .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?,
            capacity,
        ),
        BaseCommitDependencyRange::Slice { first_word, words } => (
            first_word,
            first_word
                .checked_add(words)
                .filter(|&end| end <= capacity)
                .ok_or(InvocationShapeError::InvalidBaseCommitBinding)?,
        ),
        _ => return Err(InvocationShapeError::InvalidBaseCommitBinding),
    };
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidBaseCommitBinding)
}

fn pow2(log_size: u32) -> Result<usize, InvocationShapeError> {
    1usize
        .checked_shl(log_size)
        .ok_or(InvocationShapeError::SizeOverflow)
}
