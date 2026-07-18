use std::collections::BTreeMap;

use stwo_backend_cuda::aot::COMPOSITION_WAVE_THREADS_PER_BLOCK;

use super::super::{
    CompositionAccumulatorRequirements, CompositionOutputPlan, CompositionWorkspaceRequirements,
    TRACE_TREES,
};
use super::semantic::{child, operation, AccessSpec, RelocationCatalog, RoleCatalog};
use super::*;
use crate::arena_plan::{ArenaBinding, BufferPurpose, CommitmentTreeId};
use crate::compiled_proof::InPlaceDiscipline;

mod layout;
mod prelude;
mod split;

const SECURE_WORDS: usize = 4;
const POINTER_WORDS: usize = core::mem::size_of::<*mut u32>().div_ceil(4);
const WAVE_PART_WORDS: usize = 12;
const STATIC_BLOCK: u32 = 256;

pub(super) struct Compiler<'a> {
    plan: &'a ProofArenaPlan,
    requirements: &'a CompositionWorkspaceRequirements,
    wave_program: CompositionWaveProgram,
    source_identity: [u8; 32],
    roles: RoleCatalog,
    relocations: RelocationCatalog,
    operations: Vec<CompositionOperation>,
    waves: Vec<CompositionWaveShardAuthority>,
    direct: BTreeMap<usize, ArenaBinding>,
    accumulator_generation: BTreeMap<u32, u8>,
    outputs: Option<[CompositionValueRole; 8]>,
}

impl<'a> Compiler<'a> {
    pub(super) fn new(
        plan: &'a ProofArenaPlan,
        source_identity: [u8; 32],
    ) -> Result<Self, CompositionAuthorityError> {
        let composition = plan.composition();
        let requirements = &composition.requirements;
        if requirements.mode != CompositionLaunchMode::Wave {
            return Err(CompositionAuthorityError::UnsupportedLaunchMode(
                requirements.mode,
            ));
        }
        if !matches!(
            composition.output_plan,
            CompositionOutputPlan::DirectRetainedEvaluations(_)
        ) {
            return Err(CompositionAuthorityError::UnsupportedOutputMode);
        }
        if requirements.dynamic_ext_param_count == 0 {
            return Err(CompositionAuthorityError::ShapeDrift(
                "materialization prelude is absent",
            ));
        }
        let wave_program = CompositionWaveProgram::from_plan(&composition.plan)?;
        let receipt = requirements
            .execution_receipt
            .ok_or(CompositionAuthorityError::ShapeDrift("wave receipt"))?;
        if receipt.part_count != wave_program.parts().len()
            || receipt.wave_count != wave_program.waves().len()
            || requirements.waves.len() != receipt.wave_count
            || requirements.accumulators.len() != receipt.wave_count
        {
            return Err(CompositionAuthorityError::ShapeDrift(
                "canonical wave counts",
            ));
        }
        let mut direct = BTreeMap::new();
        for binding in &composition.direct_bindings {
            let evaluation =
                binding
                    .evaluation
                    .ok_or(CompositionAuthorityError::CoefficientFallback {
                        component: binding.consumer,
                        count: 1,
                    })?;
            match direct.insert(binding.plan_column, evaluation) {
                Some(previous) if previous != evaluation => {
                    return Err(CompositionAuthorityError::ShapeDrift(
                        "direct evaluation alias",
                    ));
                }
                _ => {}
            }
        }
        for (component, requirement) in requirements.components.iter().enumerate() {
            if requirement.fallback_count != 0
                || requirement.source_retention.len() != requirement.sources.len()
                || requirement.source_retention.iter().any(|source| {
                    !source.direct
                        || source.fallback_ordinal.is_some()
                        || !direct.contains_key(&source.plan_column)
                })
            {
                return Err(CompositionAuthorityError::CoefficientFallback {
                    component,
                    count: requirement.fallback_count,
                });
            }
        }
        Ok(Self {
            plan,
            requirements,
            wave_program,
            source_identity,
            roles: RoleCatalog::new(),
            relocations: RelocationCatalog::new(),
            operations: Vec::new(),
            waves: Vec::new(),
            direct,
            accumulator_generation: requirements
                .accumulators
                .iter()
                .map(|accumulator| (accumulator.log_size, 0))
                .collect(),
            outputs: None,
        })
    }

    pub(super) fn compile(&mut self) -> Result<(), CompositionAuthorityError> {
        self.register_inputs()?;
        self.compile_materialization()?;
        self.compile_powers()?;
        self.compile_waves()?;
        self.compile_lifts()?;
        self.compile_split()?;
        Ok(())
    }

    pub(super) fn finish(
        self,
    ) -> Result<
        (
            Vec<CompositionLayout>,
            Vec<CompositionRelocationLayout>,
            Vec<CompositionOperation>,
            Vec<CompositionWaveShardAuthority>,
            [CompositionValueRole; 8],
        ),
        CompositionAuthorityError,
    > {
        let wrapper_count = 2usize
            .checked_add(self.wave_program.waves().len())
            .and_then(|count| count.checked_add(self.requirements.accumulators.len() - 1))
            .and_then(|count| count.checked_add(2))
            .ok_or(CompositionAuthorityError::SizeOverflow)?;
        let child_count = self
            .operations
            .iter()
            .try_fold(0usize, |count, operation| {
                count.checked_add(operation.children.len())
            })
            .ok_or(CompositionAuthorityError::SizeOverflow)?;
        // Every ordinary wrapper owns one child. Split A contributes two
        // additional children (3 total) and split B one (2 total): generated
        // SN2 is therefore 31 host wrappers and 34 device launches.
        if self.operations.len() != wrapper_count
            || child_count != wrapper_count + 3
            || self.waves.len() != self.wave_program.waves().len()
            || self
                .requirements
                .waves
                .iter()
                .map(|wave| wave.parts.len())
                .sum::<usize>()
                != self.wave_program.parts().len()
        {
            return Err(CompositionAuthorityError::InvalidExecution);
        }
        Ok((
            self.roles.finish(),
            self.relocations.finish(),
            self.operations,
            self.waves,
            self.outputs
                .ok_or(CompositionAuthorityError::InvalidExecution)?,
        ))
    }

    fn compile_waves(&mut self) -> Result<(), CompositionAuthorityError> {
        for wave_index in 0..self.requirements.waves.len() {
            self.compile_wave(wave_index)?;
        }
        Ok(())
    }

    fn compile_wave(&mut self, wave_index: usize) -> Result<(), CompositionAuthorityError> {
        use CompositionDescriptorRole as Descriptor;
        let requirement = self.requirements.waves[wave_index].clone();
        let part_ordinals = self.wave_program.waves()[wave_index].part_ordinals.clone();
        let kernel = self.plan.composition().plan.wave_kernels[wave_index].clone();
        let descriptors = self.descriptor_binding()?;
        let part_role = CompositionValueRole::Descriptor {
            kind: Descriptor::WaveParts,
            index: to_u32(wave_index)?,
        };
        self.insert_binding(
            part_role,
            descriptors,
            requirement.descriptor_offset_words,
            requirement.parts.len() * WAVE_PART_WORDS,
            POINTER_WORDS,
        )?;
        let mut specs = vec![AccessSpec::read(
            part_role,
            requirement.parts.len() * WAVE_PART_WORDS,
        )];
        for &ordinal in &part_ordinals {
            let part = self.wave_program.parts()[ordinal].clone();
            let component = self.requirements.components[part.component_index].clone();
            self.register_component_descriptors(part.component_index)?;
            for (kind, len) in [
                (Descriptor::InteractionOffsets, TRACE_TREES),
                (Descriptor::DenominatorInverses, component.denominator_words),
                (Descriptor::BaseParams, component.base_param_words),
            ] {
                if len != 0 {
                    specs.push(AccessSpec::read(
                        CompositionValueRole::Descriptor {
                            kind,
                            index: to_u32(part.component_index)?,
                        },
                        len,
                    ));
                }
            }
            for source in &component.source_retention {
                let role = CompositionValueRole::DirectEvaluation {
                    plan_column: to_u32(source.plan_column)?,
                };
                specs.push(AccessSpec::read(role, self.roles.layout(role)?.word_len));
            }
            for slot in 0..self.plan.composition().ext_params[part.component_index]
                .sources
                .len()
            {
                specs.push(AccessSpec::read(
                    CompositionValueRole::ExtParam {
                        component: to_u32(part.component_index)?,
                        slot: to_u32(slot)?,
                    },
                    SECURE_WORDS,
                ));
            }
            specs.push(AccessSpec::read_range(
                CompositionValueRole::RandomCoefficientPowers,
                part.coefficient_start * SECURE_WORDS,
                part.coefficient_end * SECURE_WORDS,
            ));
        }
        let rows = requirement.row_count;
        for coordinate in 0..4 {
            specs.push(AccessSpec::write(
                CompositionValueRole::Accumulator {
                    log_size: requirement.evaluation_log_size,
                    coordinate,
                    generation: 0,
                },
                rows,
            ));
        }
        let effect = self.roles.effect(specs)?;
        let embedded_pointer_tables = part_ordinals
            .iter()
            .map(|&ordinal| {
                let part = &self.wave_program.parts()[ordinal];
                self.requirements.components[part.component_index]
                    .source_retention
                    .iter()
                    .map(|source| {
                        access_index(
                            &effect,
                            CompositionValueRole::DirectEvaluation {
                                plan_column: to_u32(source.plan_column)?,
                            },
                            false,
                        )
                        .and_then(to_u32)
                        .map(Some)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|pointee_accesses| CompositionEmbeddedPointerTable { pointee_accesses })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coordinate_bindings = std::array::from_fn(|coordinate| {
            access_binding(
                &effect,
                CompositionValueRole::Accumulator {
                    log_size: requirement.evaluation_log_size,
                    coordinate: coordinate as u8,
                    generation: 0,
                },
                true,
            )
            .expect("four registered accumulator writes")
        });
        let accumulator = accumulator_for_log(
            self.requirements.accumulators.as_slice(),
            requirement.evaluation_log_size,
        )?;
        let shard = CompositionWaveShardAuthority::derive(
            &self.plan.composition().plan,
            &self.wave_program,
            wave_index,
            &requirement,
            accumulator,
            &effect.contract,
            coordinate_bindings,
        )?;
        let powers_access =
            first_source_index(&effect, CompositionValueRole::RandomCoefficientPowers)?;
        let child = child(
            kernel.kernel_name.clone(),
            CompositionLaunchGeometry {
                grid: [
                    to_u32(rows)?.div_ceil(to_u32(COMPOSITION_WAVE_THREADS_PER_BLOCK)?),
                    1,
                    1,
                ],
                block: [to_u32(COMPOSITION_WAVE_THREADS_PER_BLOCK)?, 1, 1],
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
            vec![
                ("wave_index", to_u32(wave_index)?),
                ("part_count", to_u32(requirement.parts.len())?),
                ("full_domain_rows", to_u32(rows)?),
                ("shard_start", 0),
                ("shard_rows", to_u32(rows)?),
            ],
            effect,
        )?;
        let mut args = vec![
            ("parts", CompositionInvocationValue::Access(0)),
            (
                "random_coefficient_powers",
                CompositionInvocationValue::Access(powers_access),
            ),
        ];
        for coordinate in 0..4 {
            args.push((
                match coordinate {
                    0 => "coord_0",
                    1 => "coord_1",
                    2 => "coord_2",
                    _ => "coord_3",
                },
                CompositionInvocationValue::Access(to_u32(access_index(
                    &child.effect,
                    CompositionValueRole::Accumulator {
                        log_size: requirement.evaluation_log_size,
                        coordinate,
                        generation: 0,
                    },
                    true,
                )?)?),
            ));
        }
        args.extend([
            (
                "full_domain_rows",
                CompositionInvocationValue::U32(to_u32(rows)?),
            ),
            ("shard_start", CompositionInvocationValue::U32(0)),
            ("shard_rows", CompositionInvocationValue::U32(to_u32(rows)?)),
            ("stream", CompositionInvocationValue::ExecutionStream),
        ]);
        self.operations.push(operation(
            self.source_identity,
            CompositionOperationKind::Wave {
                wave_index: to_u32(wave_index)?,
                part_count: to_u32(requirement.parts.len())?,
                evaluation_log_size: requirement.evaluation_log_size,
                row_count: to_u32(rows)?,
            },
            CompositionAbi::WaveV2,
            args,
            embedded_pointer_tables,
            vec![child],
            Some(*shard.kernel_source_identity()),
        )?);
        self.waves.push(shard);
        Ok(())
    }

    fn compile_lifts(&mut self) -> Result<(), CompositionAuthorityError> {
        let accumulators = self.requirements.accumulators.clone();
        for (lift_index, pair) in accumulators.windows(2).enumerate() {
            let previous = pair[0];
            let current = pair[1];
            let previous_generation = self.generation(previous.log_size)?;
            let current_generation = self.generation(current.log_size)?;
            self.register_accumulator(current, current_generation + 1)?;
            let mut specs = Vec::with_capacity(8);
            for coordinate in 0..4 {
                specs.push(AccessSpec::read(
                    CompositionValueRole::Accumulator {
                        log_size: previous.log_size,
                        coordinate,
                        generation: previous_generation,
                    },
                    1usize << previous.log_size,
                ));
            }
            for coordinate in 0..4 {
                specs.push(AccessSpec::required_alias(
                    CompositionValueRole::Accumulator {
                        log_size: current.log_size,
                        coordinate,
                        generation: current_generation,
                    },
                    CompositionValueRole::Accumulator {
                        log_size: current.log_size,
                        coordinate,
                        generation: current_generation + 1,
                    },
                    1usize << current.log_size,
                    InPlaceDiscipline::ElementWiseReadBeforeWrite,
                ));
            }
            let child = child(
                "lift_accumulate_coordinates",
                static_geometry(1usize << current.log_size, STATIC_BLOCK)?,
                vec![
                    ("previous_log_size", previous.log_size),
                    ("current_log_size", current.log_size),
                    ("lift", current.log_size - previous.log_size),
                ],
                self.roles.effect(specs)?,
            )?;
            self.operations.push(operation(
                self.source_identity,
                CompositionOperationKind::LiftAccumulate {
                    lift_index: to_u32(lift_index)?,
                    previous_log_size: previous.log_size,
                    current_log_size: current.log_size,
                },
                CompositionAbi::LiftAccumulateV1,
                vec![
                    (
                        "previous_coordinates",
                        CompositionInvocationValue::Access(0),
                    ),
                    (
                        "previous_log_size",
                        CompositionInvocationValue::U32(previous.log_size),
                    ),
                    ("current_coordinates", CompositionInvocationValue::Access(4)),
                    (
                        "current_log_size",
                        CompositionInvocationValue::U32(current.log_size),
                    ),
                    ("stream", CompositionInvocationValue::ExecutionStream),
                ],
                Vec::new(),
                vec![child],
                None,
            )?);
            self.accumulator_generation
                .insert(current.log_size, current_generation + 1);
        }
        Ok(())
    }
}

fn accumulator_for_log(
    accumulators: &[CompositionAccumulatorRequirements],
    log_size: u32,
) -> Result<&CompositionAccumulatorRequirements, CompositionAuthorityError> {
    let mut matches = accumulators
        .iter()
        .filter(|value| value.log_size == log_size);
    let value = matches
        .next()
        .ok_or(CompositionAuthorityError::ShapeDrift("wave accumulator"))?;
    if matches.next().is_some() {
        return Err(CompositionAuthorityError::ShapeDrift(
            "duplicate wave accumulator",
        ));
    }
    Ok(value)
}

fn static_geometry(
    elements: usize,
    block: u32,
) -> Result<CompositionLaunchGeometry, CompositionAuthorityError> {
    let elements = to_u32(elements)?;
    Ok(CompositionLaunchGeometry {
        grid: [elements.div_ceil(block), 1, 1],
        block: [block, 1, 1],
        dynamic_shared_bytes: 0,
        cooperative: false,
    })
}

fn access_index(
    effect: &CompositionEffect,
    role: CompositionValueRole,
    destination: bool,
) -> Result<usize, CompositionAuthorityError> {
    let mut matches = effect
        .accesses
        .iter()
        .enumerate()
        .filter_map(|(index, access)| {
            let candidate = if destination {
                access.destination
            } else {
                access.source
            };
            (candidate == Some(role)).then_some(index)
        });
    let index = matches
        .next()
        .ok_or(CompositionAuthorityError::InvalidInvocation)?;
    if matches.next().is_some() {
        return Err(CompositionAuthorityError::InvalidInvocation);
    }
    Ok(index)
}

fn first_source_index(
    effect: &CompositionEffect,
    role: CompositionValueRole,
) -> Result<u32, CompositionAuthorityError> {
    effect
        .accesses
        .iter()
        .position(|access| access.source == Some(role))
        .map(to_u32)
        .transpose()?
        .ok_or(CompositionAuthorityError::InvalidInvocation)
}

fn access_binding(
    effect: &CompositionEffect,
    role: CompositionValueRole,
    destination: bool,
) -> Result<EffectBindingId, CompositionAuthorityError> {
    Ok(effect.accesses[access_index(effect, role, destination)?].binding)
}

fn to_u32(value: usize) -> Result<u32, CompositionAuthorityError> {
    u32::try_from(value).map_err(|_| CompositionAuthorityError::SizeOverflow)
}
