//! Exact Relation role-to-arena projection.

use stwo_backend_cuda::{
    RelationChallengeExpansionAuthority, RelationExecutionAuthority, RelationPointerTableKind,
    RelationValueLayout, RelationValueRole,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::program_image::lower_compiled) struct RelationArenaRange {
    pub(super) catalog: ArenaCatalogValueId,
    pub(super) arena: ArenaBinding,
    pub(super) start_word: usize,
    pub(super) words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RelationInventory {
    roles: Vec<(RelationValueRole, RelationArenaRange)>,
    drawn: RelationArenaRange,
}

impl RelationInventory {
    pub(super) fn compile(
        arena: &ProofArenaPlan,
        authority: &RelationExecutionAuthority,
        challenge: &RelationChallengeExpansionAuthority,
    ) -> Result<Self, InvocationShapeError> {
        let planned = arena.relation();
        if authority.program() != planned.execution.kernel_program()
            || authority.requirements() != &planned.requirements
            || challenge.max_alpha_powers() != authority.program().max_alpha_powers
            || authority.instances().len() != planned.source_plan.len()
        {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        let catalog = BaseProducerCatalog::compile(arena)?;
        let roles = authority
            .values()
            .iter()
            .map(|layout| role_range(arena, &catalog, authority, *layout))
            .collect::<Result<Vec<_>, _>>()?;
        if roles.len() != authority.values().len()
            || roles
                .iter()
                .enumerate()
                .any(|(index, (role, _))| *role != authority.values()[index].role)
        {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        let output_id = crate::transcript_plan::CairoTranscriptOutput::CommonLookupElements
            .id()
            .map_err(|_| InvocationShapeError::InvalidRelationBinding)?;
        let drawn_binding = arena
            .transcript()
            .outputs
            .iter()
            .find_map(|&(id, binding)| (id == output_id).then_some(binding))
            .ok_or(InvocationShapeError::InvalidRelationBinding)?;
        let drawn = exact(
            arena,
            &catalog,
            None,
            None,
            BufferPurpose::TranscriptOutput,
            output_id.0,
            Some(drawn_binding.physical),
            0,
            challenge.drawn_words(),
        )?;
        let inventory = Self { roles, drawn };
        inventory.validate_challenge_ranges(challenge)?;
        Ok(inventory)
    }

    pub(super) fn role(
        &self,
        role: RelationValueRole,
    ) -> Result<RelationArenaRange, InvocationShapeError> {
        let mut matches = self
            .roles
            .iter()
            .filter_map(|(candidate, range)| (*candidate == role).then_some(*range));
        let exact = matches
            .next()
            .ok_or(InvocationShapeError::InvalidRelationBinding)?;
        if matches.next().is_some() {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        Ok(exact)
    }

    pub(super) const fn drawn(&self) -> RelationArenaRange {
        self.drawn
    }

    fn validate_challenge_ranges(
        &self,
        challenge: &RelationChallengeExpansionAuthority,
    ) -> Result<(), InvocationShapeError> {
        if self.drawn.words != challenge.drawn_words()
            || self.role(RelationValueRole::AlphaPowers)?.words
                != challenge
                    .alpha_words()
                    .map_err(|_| InvocationShapeError::InvalidRelationAuthority)?
            || self.role(RelationValueRole::ChallengeZ)?.words != challenge.z_words()
        {
            return Err(InvocationShapeError::InvalidRelationBinding);
        }
        Ok(())
    }
}

fn role_range(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    authority: &RelationExecutionAuthority,
    layout: RelationValueLayout,
) -> Result<(RelationValueRole, RelationArenaRange), InvocationShapeError> {
    let planned = arena.relation();
    let global = |purpose, ordinal, binding, start| {
        exact(
            arena,
            catalog,
            None,
            None,
            purpose,
            ordinal,
            Some(binding),
            start,
            layout.words,
        )
    };
    let range = match layout.role {
        RelationValueRole::Descriptors => global(
            BufferPurpose::RelationDescriptors,
            0,
            planned.slots.descriptors,
            0,
        )?,
        RelationValueRole::AlphaPowers => global(
            BufferPurpose::RelationAlphaPowers,
            0,
            planned.slots.alphas,
            0,
        )?,
        RelationValueRole::ChallengeZ => global(BufferPurpose::RelationZ, 0, planned.slots.z, 0)?,
        RelationValueRole::DispatchPointers(table) => {
            let table_index: usize = match table {
                RelationPointerTableKind::Sources => 0,
                RelationPointerTableKind::Descriptors => 1,
                RelationPointerTableKind::Outputs => 2,
                RelationPointerTableKind::DenominatorsUnused => 3,
                RelationPointerTableKind::ClaimedSums => 4,
            };
            let start = table_index
                .checked_mul(layout.words)
                .ok_or(InvocationShapeError::SizeOverflow)?;
            global(
                BufferPurpose::RelationFractionPointers,
                0,
                planned.slots.fraction_pointers,
                start,
            )?
        }
        RelationValueRole::Geometry => global(
            BufferPurpose::RelationFractionGeometry,
            0,
            planned.slots.fraction_geometry,
            0,
        )?,
        RelationValueRole::InstanceSourcePointers { batch, instance } => instance_global(
            arena,
            catalog,
            authority,
            batch,
            instance,
            BufferPurpose::RelationSourcePointers,
            layout.words,
        )?,
        RelationValueRole::InstanceSource {
            batch,
            instance,
            source,
        } => source_range(
            arena,
            catalog,
            authority,
            batch,
            instance,
            source,
            layout.words,
        )?,
        RelationValueRole::InstanceOutputPointers { batch, instance } => instance_global(
            arena,
            catalog,
            authority,
            batch,
            instance,
            BufferPurpose::RelationOutputPointers,
            layout.words,
        )?,
        RelationValueRole::OutputCoordinate {
            batch,
            instance,
            coordinate,
        } => output_range(
            arena,
            catalog,
            authority,
            batch,
            instance,
            coordinate,
            layout.words,
        )?,
        RelationValueRole::DenominatorSentinelUnused { batch, instance } => instance_global(
            arena,
            catalog,
            authority,
            batch,
            instance,
            BufferPurpose::RelationDenominators,
            layout.words,
        )?,
        RelationValueRole::ClaimedSum { batch, instance } => instance_global(
            arena,
            catalog,
            authority,
            batch,
            instance,
            BufferPurpose::RelationClaimedSum,
            layout.words,
        )?,
        RelationValueRole::InverseScratchUnused => global(
            BufferPurpose::RelationInverseScratch,
            0,
            planned.slots.inverse_scratch,
            0,
        )?,
        RelationValueRole::ReductionPartials => global(
            BufferPurpose::RelationReductionA,
            0,
            planned.slots.reduction_a,
            0,
        )?,
        RelationValueRole::ScanBlockSums => global(
            BufferPurpose::RelationReductionB,
            0,
            planned.slots.reduction_b,
            0,
        )?,
        RelationValueRole::ScanEvalScratchUnused => global(
            BufferPurpose::RelationScanEvalScratch,
            0,
            planned.slots.scan_eval_scratch,
            0,
        )?,
        RelationValueRole::ScanTempScratchUnused => global(
            BufferPurpose::RelationScanTempScratch,
            0,
            planned.slots.scan_temp_scratch,
            0,
        )?,
        RelationValueRole::ScanDescriptorsUnused => global(
            BufferPurpose::RelationScanDescriptors,
            0,
            planned.slots.scan_descriptors,
            0,
        )?,
    };
    if range.words != layout.words {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok((layout.role, range))
}

fn instance_index(
    authority: &RelationExecutionAuthority,
    batch: u32,
    instance: u32,
) -> Result<usize, InvocationShapeError> {
    let mut matches = authority
        .instances()
        .iter()
        .enumerate()
        .filter_map(|(index, exact)| {
            (exact.batch_index == batch && exact.instance_index == instance).then_some(index)
        });
    let index = matches
        .next()
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok(index)
}

fn instance_global(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    authority: &RelationExecutionAuthority,
    batch: u32,
    instance: u32,
    purpose: BufferPurpose,
    words: usize,
) -> Result<RelationArenaRange, InvocationShapeError> {
    let index = instance_index(authority, batch, instance)?;
    exact(
        arena,
        catalog,
        None,
        None,
        purpose,
        u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
        None,
        0,
        words,
    )
}

fn source_range(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    authority: &RelationExecutionAuthority,
    batch: u32,
    instance: u32,
    source: u32,
    words: usize,
) -> Result<RelationArenaRange, InvocationShapeError> {
    let index = instance_index(authority, batch, instance)?;
    let plan = arena
        .relation()
        .source_plan
        .get(index)
        .filter(|plan| {
            arena.relation().execution.batches.get(batch as usize) == Some(&plan.batch)
                && plan.instance_index == instance as usize
                && authority.instances()[index].batch_index == batch
                && authority.instances()[index].instance_index == instance
        })
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    let purpose = match plan.plane {
        crate::relation_execution::RelationSourcePlane::LookupWords => BufferPurpose::LookupInputs,
        crate::relation_execution::RelationSourcePlane::BaseTrace => BufferPurpose::BaseTrace,
        crate::relation_execution::RelationSourcePlane::WitnessInput => BufferPurpose::WitnessInput,
    };
    exact(
        arena,
        catalog,
        Some(plan.batch.component),
        Some(plan.part),
        purpose,
        source,
        None,
        0,
        words,
    )
}

fn output_range(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    authority: &RelationExecutionAuthority,
    batch: u32,
    instance: u32,
    coordinate: u32,
    words: usize,
) -> Result<RelationArenaRange, InvocationShapeError> {
    let index = instance_index(authority, batch, instance)?;
    let plan = arena
        .relation()
        .source_plan
        .get(index)
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    exact(
        arena,
        catalog,
        Some(plan.batch.component),
        Some(plan.part),
        BufferPurpose::InteractionTrace,
        coordinate,
        None,
        0,
        words,
    )
}

#[allow(clippy::too_many_arguments)]
fn exact(
    arena: &ProofArenaPlan,
    catalog: &BaseProducerCatalog,
    component: Option<&str>,
    part: Option<TracePartId>,
    purpose: BufferPurpose,
    ordinal: u32,
    expected: Option<stwo_backend_cuda::ArenaSlotId>,
    start_word: usize,
    words: usize,
) -> Result<RelationArenaRange, InvocationShapeError> {
    let (logical, binding) = arena
        .find(component, part, purpose, ordinal)
        .ok_or(InvocationShapeError::InvalidRelationBinding)?;
    if expected.is_some_and(|expected| expected != binding.physical)
        || binding.logical != logical.id
        || binding.len_words != logical.len_words
        || start_word
            .checked_add(words)
            .is_none_or(|end| end > binding.len_words)
        || words == 0
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    let catalog_id = ArenaCatalogValueId(logical.id.0);
    let value = catalog.value(catalog_id)?;
    if value.logical != logical.id
        || value.physical != binding.physical
        || value.words != binding.len_words
    {
        return Err(InvocationShapeError::InvalidRelationBinding);
    }
    Ok(RelationArenaRange {
        catalog: catalog_id,
        arena: binding,
        start_word,
        words,
    })
}

pub(super) fn hash_role(hasher: &mut blake3::Hasher, role: RelationValueRole) {
    match role {
        RelationValueRole::Descriptors => hash_u32s(hasher, &[1]),
        RelationValueRole::AlphaPowers => hash_u32s(hasher, &[2]),
        RelationValueRole::ChallengeZ => hash_u32s(hasher, &[3]),
        RelationValueRole::DispatchPointers(table) => hash_u32s(hasher, &[4, table as u32]),
        RelationValueRole::Geometry => hash_u32s(hasher, &[5]),
        RelationValueRole::InstanceSourcePointers { batch, instance } => {
            hash_u32s(hasher, &[6, batch, instance])
        }
        RelationValueRole::InstanceSource {
            batch,
            instance,
            source,
        } => hash_u32s(hasher, &[7, batch, instance, source]),
        RelationValueRole::InstanceOutputPointers { batch, instance } => {
            hash_u32s(hasher, &[8, batch, instance])
        }
        RelationValueRole::OutputCoordinate {
            batch,
            instance,
            coordinate,
        } => hash_u32s(hasher, &[9, batch, instance, coordinate]),
        RelationValueRole::DenominatorSentinelUnused { batch, instance } => {
            hash_u32s(hasher, &[10, batch, instance])
        }
        RelationValueRole::ClaimedSum { batch, instance } => {
            hash_u32s(hasher, &[11, batch, instance])
        }
        RelationValueRole::InverseScratchUnused => hash_u32s(hasher, &[12]),
        RelationValueRole::ReductionPartials => hash_u32s(hasher, &[13]),
        RelationValueRole::ScanBlockSums => hash_u32s(hasher, &[14]),
        RelationValueRole::ScanEvalScratchUnused => hash_u32s(hasher, &[15]),
        RelationValueRole::ScanTempScratchUnused => hash_u32s(hasher, &[16]),
        RelationValueRole::ScanDescriptorsUnused => hash_u32s(hasher, &[17]),
    }
}

fn hash_u32s(hasher: &mut blake3::Hasher, values: &[u32]) {
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}
