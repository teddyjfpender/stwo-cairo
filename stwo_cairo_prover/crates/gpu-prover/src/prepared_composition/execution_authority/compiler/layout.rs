use stwo_backend_cuda::ArenaSlotId;

use super::*;
use crate::arena_plan::{BufferPurpose, LogicalBuffer};

impl Compiler<'_> {
    pub(super) fn register_component_descriptors(
        &mut self,
        component: usize,
    ) -> Result<(), CompositionAuthorityError> {
        use CompositionDescriptorRole as Descriptor;
        let descriptors = self.descriptor_binding()?;
        let requirement = &self.requirements.components[component];
        let layout = &self.requirements.component_descriptors[component];
        let evaluation_words = requirement.sources.len() * POINTER_WORDS;
        if evaluation_words != 0 {
            self.insert_relocation(
                CompositionRelocationRole::EvaluationPointers {
                    component: to_u32(component)?,
                },
                descriptors,
                layout.evaluation_pointers,
                evaluation_words,
                POINTER_WORDS,
            )?;
        }
        for (kind, first, len, alignment) in [
            (
                Descriptor::InteractionOffsets,
                layout.interaction_offsets,
                TRACE_TREES,
                1,
            ),
            (
                Descriptor::DenominatorInverses,
                layout.denominator_inverses,
                requirement.denominator_words,
                1,
            ),
            (
                Descriptor::BaseParams,
                layout.base_params,
                requirement.base_param_words,
                1,
            ),
        ] {
            if len != 0 {
                self.insert_binding(
                    CompositionValueRole::Descriptor {
                        kind,
                        index: to_u32(component)?,
                    },
                    descriptors,
                    first,
                    len,
                    alignment,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn register_accumulator(
        &mut self,
        accumulator: CompositionAccumulatorRequirements,
        generation: u8,
    ) -> Result<(), CompositionAuthorityError> {
        let binding = self.binding_for_slot(
            BufferPurpose::CompositionAccumulators,
            self.plan.composition().slots.accumulators,
        )?;
        let rows = 1usize
            .checked_shl(accumulator.log_size)
            .ok_or(CompositionAuthorityError::SizeOverflow)?;
        if accumulator.len_words != rows * 4 {
            return Err(CompositionAuthorityError::ShapeDrift("accumulator extent"));
        }
        for coordinate in 0..4 {
            self.insert_binding(
                CompositionValueRole::Accumulator {
                    log_size: accumulator.log_size,
                    coordinate,
                    generation,
                },
                binding,
                accumulator.offset_words + coordinate as usize * rows,
                rows,
                1,
            )?;
        }
        Ok(())
    }

    pub(super) fn insert_binding(
        &mut self,
        role: CompositionValueRole,
        binding: ArenaBinding,
        first: usize,
        len: usize,
        alignment: usize,
    ) -> Result<(), CompositionAuthorityError> {
        if first
            .checked_add(len)
            .is_none_or(|end| end > binding.len_words)
        {
            return Err(CompositionAuthorityError::ShapeDrift(
                "role exceeds logical binding",
            ));
        }
        self.roles
            .insert(role, binding.logical, first, len, alignment)
    }

    pub(super) fn insert_relocation(
        &mut self,
        role: CompositionRelocationRole,
        binding: ArenaBinding,
        first: usize,
        len: usize,
        alignment: usize,
    ) -> Result<(), CompositionAuthorityError> {
        if first
            .checked_add(len)
            .is_none_or(|end| end > binding.len_words)
        {
            return Err(CompositionAuthorityError::ShapeDrift(
                "relocation exceeds logical binding",
            ));
        }
        self.relocations
            .insert(role, binding.logical, first, len, alignment)
    }

    pub(super) fn descriptor_binding(&self) -> Result<ArenaBinding, CompositionAuthorityError> {
        self.binding_for_slot(
            BufferPurpose::CompositionDescriptors,
            self.plan.composition().slots.descriptors,
        )
    }

    pub(super) fn binding_for_slot(
        &self,
        purpose: BufferPurpose,
        slot: ArenaSlotId,
    ) -> Result<ArenaBinding, CompositionAuthorityError> {
        let mut matches = self.plan.logical_buffers().iter().filter_map(|logical| {
            (logical.purpose == purpose)
                .then(|| self.plan.binding(logical.id))
                .flatten()
                .filter(|binding| binding.physical == slot)
        });
        let binding = matches
            .next()
            .ok_or(CompositionAuthorityError::MissingLogicalRole(
                "arena slot purpose",
            ))?;
        if matches.next().is_some() {
            return Err(CompositionAuthorityError::AmbiguousLogicalRole(
                "arena slot purpose",
            ));
        }
        Ok(binding)
    }

    pub(super) fn find(
        &self,
        purpose: BufferPurpose,
        ordinal: u32,
    ) -> Result<(&LogicalBuffer, ArenaBinding), CompositionAuthorityError> {
        self.plan.find(None, None, purpose, ordinal).ok_or(
            CompositionAuthorityError::MissingLogicalRole("arena purpose and ordinal"),
        )
    }

    pub(super) fn claimed_sum_binding(
        &self,
        component: usize,
    ) -> Result<ArenaBinding, CompositionAuthorityError> {
        let plan_component = &self.plan.composition().plan.components[component];
        let (name, part, instance) = crate::resident_composition::relation_claimed_sum_key(
            plan_component.component,
            plan_component.instance,
        );
        let batch = self
            .plan
            .relation()
            .execution
            .batches
            .iter()
            .position(|candidate| candidate.component == name && candidate.trace_part == part)
            .ok_or(CompositionAuthorityError::MissingLogicalRole(
                "relation claimed-sum batch",
            ))?;
        let ordinal = self
            .plan
            .relation()
            .requirements
            .instances
            .iter()
            .position(|candidate| {
                candidate.batch_index == batch && candidate.instance_index == instance
            })
            .ok_or(CompositionAuthorityError::MissingLogicalRole(
                "relation claimed-sum instance",
            ))?;
        Ok(self
            .find(BufferPurpose::RelationClaimedSum, to_u32(ordinal)?)?
            .1)
    }

    pub(super) fn generation(&self, log_size: u32) -> Result<u8, CompositionAuthorityError> {
        self.accumulator_generation.get(&log_size).copied().ok_or(
            CompositionAuthorityError::ShapeDrift("accumulator generation"),
        )
    }
}
