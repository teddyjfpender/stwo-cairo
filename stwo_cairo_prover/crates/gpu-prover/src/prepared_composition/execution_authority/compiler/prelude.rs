use super::*;
use crate::composition_plan::CompositionExtParamSource;

impl Compiler<'_> {
    pub(super) fn register_inputs(&mut self) -> Result<(), CompositionAuthorityError> {
        let composition = self.plan.composition();
        self.insert_binding(
            CompositionValueRole::RandomCoefficient,
            composition.random_coefficient,
            0,
            SECURE_WORDS,
            SECURE_WORDS,
        )?;
        self.insert_binding(
            CompositionValueRole::ForwardTwiddles,
            composition.forward_twiddles,
            0,
            self.requirements.forward_twiddle_words,
            1,
        )?;
        self.insert_binding(
            CompositionValueRole::InverseTwiddles,
            composition.inverse_twiddles,
            0,
            self.requirements.inverse_twiddle_words,
            1,
        )?;
        let powers = self.binding_for_slot(
            BufferPurpose::CompositionRandomCoefficientPowers,
            composition.slots.random_coefficient_powers,
        )?;
        self.insert_binding(
            CompositionValueRole::RandomCoefficientPowers,
            powers,
            0,
            self.requirements.random_power_words,
            SECURE_WORDS,
        )?;
        let z = self.find(BufferPurpose::RelationZ, 0)?.1;
        let alpha = self.find(BufferPurpose::RelationAlphaPowers, 0)?.1;
        self.insert_binding(
            CompositionValueRole::RelationZ,
            z,
            0,
            SECURE_WORDS,
            SECURE_WORDS,
        )?;
        self.insert_binding(
            CompositionValueRole::RelationAlphaPowers,
            alpha,
            0,
            alpha.len_words,
            SECURE_WORDS,
        )?;
        for (&plan_column, &binding) in &self.direct.clone() {
            self.insert_binding(
                CompositionValueRole::DirectEvaluation {
                    plan_column: to_u32(plan_column)?,
                },
                binding,
                0,
                binding.len_words,
                1,
            )?;
        }
        for (component, params) in composition.ext_params.iter().enumerate() {
            let Some(binding) = params.binding else {
                if !params.sources.is_empty() {
                    return Err(CompositionAuthorityError::MissingLogicalRole(
                        "composition ext params",
                    ));
                }
                continue;
            };
            if binding.len_words != params.sources.len() * SECURE_WORDS {
                return Err(CompositionAuthorityError::ShapeDrift(
                    "extension parameter extent",
                ));
            }
            for slot in 0..params.sources.len() {
                self.insert_binding(
                    CompositionValueRole::ExtParam {
                        component: to_u32(component)?,
                        slot: to_u32(slot)?,
                    },
                    binding,
                    slot * SECURE_WORDS,
                    SECURE_WORDS,
                    SECURE_WORDS,
                )?;
            }
        }
        for accumulator in self.requirements.accumulators.clone() {
            self.register_accumulator(accumulator, 0)?;
        }
        Ok(())
    }

    pub(super) fn compile_materialization(&mut self) -> Result<(), CompositionAuthorityError> {
        use CompositionDescriptorRole as Descriptor;
        let descriptors = self.descriptor_binding()?;
        let count = self.requirements.dynamic_ext_param_count;
        let claimed_count = self.requirements.claimed_sum_count;
        for (role, first, len) in [
            (
                CompositionRelocationRole::DynamicDestinations,
                self.requirements.dynamic_destination_pointers,
                count * POINTER_WORDS,
            ),
            (
                CompositionRelocationRole::ClaimedSumPointers,
                self.requirements.claimed_sum_pointers,
                claimed_count * POINTER_WORDS,
            ),
        ] {
            if len != 0 {
                self.insert_relocation(role, descriptors, first, len, POINTER_WORDS)?;
            }
        }
        let descriptor_roles = [
            (
                Descriptor::DynamicSourceKinds,
                self.requirements.dynamic_source_kinds,
                count,
                1,
            ),
            (
                Descriptor::DynamicSourceIndices,
                self.requirements.dynamic_source_indices,
                count,
                1,
            ),
            (
                Descriptor::DynamicScales,
                self.requirements.dynamic_scales,
                count,
                1,
            ),
        ];
        for (kind, first, len, alignment) in descriptor_roles {
            if len != 0 {
                self.insert_binding(
                    CompositionValueRole::Descriptor { kind, index: 0 },
                    descriptors,
                    first,
                    len,
                    alignment,
                )?;
            }
        }
        let mut specs = descriptor_roles
            .into_iter()
            .filter(|(_, _, len, _)| *len != 0)
            .map(|(kind, _, len, _)| {
                AccessSpec::read(CompositionValueRole::Descriptor { kind, index: 0 }, len)
            })
            .collect::<Vec<_>>();
        specs.push(AccessSpec::read(
            CompositionValueRole::RelationZ,
            SECURE_WORDS,
        ));
        let mut dynamic_outputs = Vec::with_capacity(count);
        let mut claimed_outputs = Vec::new();
        let mut claimed_component = BTreeMap::new();
        for (component, params) in self.plan.composition().ext_params.iter().enumerate() {
            for (slot, source) in params.sources.iter().enumerate() {
                let role = CompositionValueRole::ExtParam {
                    component: to_u32(component)?,
                    slot: to_u32(slot)?,
                };
                match source {
                    CompositionExtParamSource::Constant(_) => continue,
                    CompositionExtParamSource::LookupZ => {}
                    CompositionExtParamSource::LookupAlphaPower(power)
                    | CompositionExtParamSource::LookupAlphaPowerScaled { power, .. } => {
                        let start = usize::try_from(*power)
                            .map_err(|_| CompositionAuthorityError::SizeOverflow)?
                            * SECURE_WORDS;
                        specs.push(AccessSpec::read_range(
                            CompositionValueRole::RelationAlphaPowers,
                            start,
                            start + SECURE_WORDS,
                        ));
                    }
                    CompositionExtParamSource::ClaimedSumScaled => {
                        let claimed = *claimed_component
                            .entry(component)
                            .or_insert_with(|| claimed_outputs.len());
                        if claimed == claimed_outputs.len() {
                            claimed_outputs.push(component);
                        }
                    }
                }
                specs.push(AccessSpec::write(role, SECURE_WORDS));
                dynamic_outputs.push(role);
            }
        }
        if dynamic_outputs.len() != count || claimed_outputs.len() != claimed_count {
            return Err(CompositionAuthorityError::ShapeDrift(
                "dynamic parameter counts",
            ));
        }
        for component in claimed_outputs.iter().copied() {
            let binding = self.claimed_sum_binding(component)?;
            let role = CompositionValueRole::ClaimedSum {
                component: to_u32(component)?,
            };
            self.insert_binding(role, binding, 0, SECURE_WORDS, SECURE_WORDS)?;
            specs.push(AccessSpec::read(role, SECURE_WORDS));
        }
        let effect = self.roles.effect(specs)?;
        let index = |role, destination| {
            access_index(&effect, role, destination)
                .map(to_u32)
                .and_then(|value| value)
        };
        let destination_pointees = dynamic_outputs
            .iter()
            .map(|&role| index(role, true).map(Some))
            .collect::<Result<Vec<_>, _>>()?;
        let claimed_pointees = claimed_outputs
            .iter()
            .map(|&component| {
                index(
                    CompositionValueRole::ClaimedSum {
                        component: to_u32(component)?,
                    },
                    false,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let descriptor_index =
            |kind| index(CompositionValueRole::Descriptor { kind, index: 0 }, false);
        let child = child(
            "materialize_ext_params",
            static_geometry(count, STATIC_BLOCK)?,
            vec![("count", to_u32(count)?)],
            effect.clone(),
        )?;
        self.operations.push(operation(
            self.source_identity,
            CompositionOperationKind::MaterializeExtParams {
                count: to_u32(count)?,
                alpha_power_count: to_u32(
                    self.roles
                        .layout(CompositionValueRole::RelationAlphaPowers)?
                        .word_len
                        / SECURE_WORDS,
                )?,
                claimed_sum_count: to_u32(claimed_count)?,
            },
            CompositionAbi::MaterializeExtParamsV1,
            vec![
                (
                    "destinations",
                    CompositionInvocationValue::PointerTable {
                        pointee_accesses: destination_pointees,
                    },
                ),
                (
                    "source_kinds",
                    CompositionInvocationValue::Access(descriptor_index(
                        Descriptor::DynamicSourceKinds,
                    )?),
                ),
                (
                    "source_indices",
                    CompositionInvocationValue::Access(descriptor_index(
                        Descriptor::DynamicSourceIndices,
                    )?),
                ),
                (
                    "scales",
                    CompositionInvocationValue::Access(descriptor_index(
                        Descriptor::DynamicScales,
                    )?),
                ),
                ("count", CompositionInvocationValue::U32(to_u32(count)?)),
                (
                    "z",
                    CompositionInvocationValue::Access(index(
                        CompositionValueRole::RelationZ,
                        false,
                    )?),
                ),
                (
                    "alpha_powers",
                    CompositionInvocationValue::Access(first_source_index(
                        &child.effect,
                        CompositionValueRole::RelationAlphaPowers,
                    )?),
                ),
                (
                    "alpha_power_count",
                    CompositionInvocationValue::U32(to_u32(
                        self.plan.relation().requirements.alpha_words / SECURE_WORDS,
                    )?),
                ),
                (
                    "claimed_sums",
                    CompositionInvocationValue::PointerTable {
                        pointee_accesses: claimed_pointees,
                    },
                ),
                (
                    "claimed_sum_count",
                    CompositionInvocationValue::U32(to_u32(claimed_count)?),
                ),
                ("stream", CompositionInvocationValue::ExecutionStream),
            ],
            Vec::new(),
            vec![child],
            None,
        )?);
        Ok(())
    }

    pub(super) fn compile_powers(&mut self) -> Result<(), CompositionAuthorityError> {
        let count = self.requirements.total_constraints;
        let effect = self.roles.effect([
            AccessSpec::read(CompositionValueRole::RandomCoefficient, SECURE_WORDS),
            AccessSpec::write(
                CompositionValueRole::RandomCoefficientPowers,
                self.requirements.random_power_words,
            ),
        ])?;
        let child = child(
            "generate_descending_powers",
            static_geometry(count, STATIC_BLOCK)?,
            vec![("count", to_u32(count)?)],
            effect,
        )?;
        self.operations.push(operation(
            self.source_identity,
            CompositionOperationKind::GenerateDescendingPowers {
                count: to_u32(count)?,
            },
            CompositionAbi::GenerateDescendingPowersV1,
            vec![
                ("random_coefficient", CompositionInvocationValue::Access(0)),
                ("powers", CompositionInvocationValue::Access(1)),
                ("count", CompositionInvocationValue::U32(to_u32(count)?)),
                ("stream", CompositionInvocationValue::ExecutionStream),
            ],
            Vec::new(),
            vec![child],
            None,
        )?);
        Ok(())
    }
}
