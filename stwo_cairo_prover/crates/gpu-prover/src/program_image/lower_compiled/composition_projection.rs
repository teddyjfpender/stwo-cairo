//! Proof-wide semantic projection of the exact Composition wave kernels.
//!
//! The source authority owns wave order, generated kernel identity, descriptor
//! traversal, and row partitioning. This module only replaces its local value
//! ordinals with the proof-wide allocator and emits the raw nine-argument AOT
//! kernel contract consumed by `CompiledProof`.

use std::collections::{BTreeMap, BTreeSet};

use crate::arena_plan::{LogicalBufferId, ProofArenaPlan};
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, BoundValueRange,
    DeviceRecordPointerBinding, DeviceRecordPointerFieldBinding, EffectAccess, EffectBindingId,
    EffectContract, ElementRange, LaunchGeometry, ValueRange, ValueVersion,
};
use crate::composition_wave::{CompositionWaveProgram, CompositionWaveShardAuthority};
use crate::prepared_composition::{
    CompositionAccessKind, CompositionDescriptorRole, CompositionExecutionAuthority,
    CompositionLayout, CompositionOperation, CompositionOperationKind, CompositionValueRole,
};

use super::{adapter, ArenaCatalogValueId, InvocationShapeError};

const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-composition-waves.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionBinding {
    pub(super) binding: EffectBindingId,
    pub(super) role: CompositionValueRole,
    pub(super) arena: CompositionLayout,
    pub(super) version: ValueVersion,
    pub(super) elements: ElementRange,
    pub(super) kind: CompositionAccessKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionWave {
    operation_ordinal: u32,
    operation: CompositionOperation,
    shard: CompositionWaveShardAuthority,
    invocation: AotInvocation,
    effect: EffectContract,
    launch: LaunchGeometry,
    bindings: Vec<LoweredCompositionBinding>,
}

impl LoweredCompositionWave {
    pub(super) const fn operation_ordinal(&self) -> u32 {
        self.operation_ordinal
    }

    pub(super) const fn operation(&self) -> &CompositionOperation {
        &self.operation
    }

    pub(super) const fn shard(&self) -> &CompositionWaveShardAuthority {
        &self.shard
    }

    pub(super) const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }

    pub(super) const fn launch(&self) -> LaunchGeometry {
        self.launch
    }

    pub(super) fn bindings(&self) -> &[LoweredCompositionBinding] {
        &self.bindings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionWaves {
    authority: CompositionExecutionAuthority,
    waves: Vec<LoweredCompositionWave>,
    digest: [u8; 32],
}

impl LoweredCompositionWaves {
    pub(super) const fn authority(&self) -> &CompositionExecutionAuthority {
        &self.authority
    }

    pub(super) fn waves(&self) -> &[LoweredCompositionWave] {
        &self.waves
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Lower every generated wave or leave the proof-wide allocator untouched.
///
/// Composition materialization and random-power generation must already have
/// installed their arena values in `values`; this slice never invents them as
/// external inputs.
pub(super) fn lower_waves(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionWaves, InvocationShapeError> {
    let authority = CompositionExecutionAuthority::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let program = CompositionWaveProgram::from_plan(&arena.composition().plan)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let layouts = authority
        .layouts()
        .iter()
        .map(|layout| (layout.role, *layout))
        .collect::<BTreeMap<_, _>>();
    if layouts.len() != authority.layouts().len() {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }

    let mut next_values = values.clone();
    let mut waves = Vec::with_capacity(authority.waves().len());
    for (operation_ordinal, operation) in authority.operations().iter().enumerate() {
        let CompositionOperationKind::Wave { wave_index, .. } = operation.kind else {
            continue;
        };
        let wave_index =
            usize::try_from(wave_index).map_err(|_| InvocationShapeError::SizeOverflow)?;
        if wave_index != waves.len() {
            return Err(InvocationShapeError::InvalidCompositionAuthority);
        }
        waves.push(lower_wave(
            arena,
            &program,
            &layouts,
            operation_ordinal,
            wave_index,
            operation,
            &mut next_values,
        )?);
    }
    if waves.len() != authority.waves().len() {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    let digest = receipt_digest(&authority, &waves)?;
    let lowered = LoweredCompositionWaves {
        authority,
        waves,
        digest,
    };
    validate_receipt(arena, &lowered)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredCompositionWaves,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_waves(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidCompositionBinding)
    }
}

pub(super) fn wave_input_catalogs(
    arena: &ProofArenaPlan,
) -> Result<Vec<ArenaCatalogValueId>, InvocationShapeError> {
    let authority = CompositionExecutionAuthority::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    let layouts = authority
        .layouts()
        .iter()
        .map(|layout| (layout.role, layout))
        .collect::<BTreeMap<_, _>>();
    let mut catalogs = BTreeSet::new();
    for operation in authority.operations() {
        if !matches!(operation.kind, CompositionOperationKind::Wave { .. }) {
            continue;
        }
        let [child] = operation.children.as_slice() else {
            return Err(InvocationShapeError::InvalidCompositionAuthority);
        };
        for access in &child.effect.accesses {
            let Some(role) = access.source else {
                continue;
            };
            let layout = layouts
                .get(&role)
                .ok_or(InvocationShapeError::InvalidCompositionBinding)?;
            catalogs.insert(ArenaCatalogValueId(layout.logical.0));
        }
    }
    Ok(catalogs.into_iter().collect())
}

#[allow(clippy::too_many_arguments)]
fn lower_wave(
    arena: &ProofArenaPlan,
    program: &CompositionWaveProgram,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
    operation_ordinal: usize,
    wave_index: usize,
    operation: &CompositionOperation,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionWave, InvocationShapeError> {
    let requirement = arena
        .composition()
        .requirements
        .waves
        .get(wave_index)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let [child] = operation.children.as_slice() else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    let CompositionOperationKind::Wave {
        wave_index: declared_wave,
        part_count,
        evaluation_log_size,
        row_count,
    } = operation.kind
    else {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    };
    if declared_wave as usize != wave_index
        || part_count as usize != requirement.parts.len()
        || evaluation_log_size != requirement.evaluation_log_size
        || row_count as usize != requirement.row_count
        || child.launch.grid.contains(&0)
        || child.launch.block.contains(&0)
    {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }

    let mut builder = WaveBindingBuilder::new(layouts, values);
    let parts_role = CompositionValueRole::Descriptor {
        kind: CompositionDescriptorRole::WaveParts,
        index: u32::try_from(wave_index).map_err(|_| InvocationShapeError::SizeOverflow)?,
    };
    let root = builder.read_full(parts_role)?;
    let records = requirement
        .parts
        .iter()
        .map(|part| builder.record(arena, part.component))
        .collect::<Result<Vec<_>, _>>()?;
    let power_ranges = child
        .effect
        .accesses
        .iter()
        .filter_map(|access| {
            (access.source == Some(CompositionValueRole::RandomCoefficientPowers))
                .then_some(access.elements)
        })
        .map(|elements| builder.read(CompositionValueRole::RandomCoefficientPowers, elements))
        .collect::<Result<Vec<_>, _>>()?;
    if power_ranges.is_empty() {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }

    let coordinate_roles: [CompositionValueRole; 4] =
        std::array::from_fn(|coordinate| CompositionValueRole::Accumulator {
            log_size: requirement.evaluation_log_size,
            coordinate: coordinate as u8,
            generation: 0,
        });
    let coordinate_bindings: [EffectBindingId; 4] = coordinate_roles
        .map(|role| builder.write_full(role))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?;
    let effect = builder.finish_effect()?;
    validate_projected_roles(child, &effect, &builder.bindings)?;

    let invocation = AotInvocation {
        arguments: vec![
            argument(
                0,
                AotArgumentValue::DeviceRecordPointerGraphValue { root, records },
            ),
            argument(
                1,
                AotArgumentValue::DevicePointerRangeSetValue {
                    ranges: power_ranges,
                },
            ),
            argument(
                2,
                AotArgumentValue::DevicePointer(Some(coordinate_bindings[0])),
            ),
            argument(
                3,
                AotArgumentValue::DevicePointer(Some(coordinate_bindings[1])),
            ),
            argument(
                4,
                AotArgumentValue::DevicePointer(Some(coordinate_bindings[2])),
            ),
            argument(
                5,
                AotArgumentValue::DevicePointer(Some(coordinate_bindings[3])),
            ),
            argument(6, AotArgumentValue::U32(row_count)),
            argument(7, AotArgumentValue::U32(0)),
            argument(8, AotArgumentValue::U32(row_count)),
        ],
    };
    validate_invocation_bindings(&invocation, &effect)?;

    let accumulator = arena
        .composition()
        .requirements
        .accumulators
        .iter()
        .find(|accumulator| accumulator.log_size == requirement.evaluation_log_size)
        .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
    let shard = CompositionWaveShardAuthority::derive(
        &arena.composition().plan,
        program,
        wave_index,
        requirement,
        accumulator,
        &effect,
        coordinate_bindings,
    )
    .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    if shard.partition().kind() == &crate::compiled_proof::PartitionAuthorityKind::Monolithic {
        return Err(InvocationShapeError::InvalidCompositionAuthority);
    }
    Ok(LoweredCompositionWave {
        operation_ordinal: u32::try_from(operation_ordinal)
            .map_err(|_| InvocationShapeError::SizeOverflow)?,
        operation: operation.clone(),
        shard,
        invocation,
        effect,
        launch: LaunchGeometry {
            grid: child.launch.grid,
            block: child.launch.block,
            cluster: None,
            dynamic_shared_bytes: child.launch.dynamic_shared_bytes,
            cooperative: child.launch.cooperative,
        },
        bindings: builder.bindings,
    })
}

struct WaveBindingBuilder<'a> {
    layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
    values: &'a mut adapter::SemanticValueMap,
    outputs: BTreeMap<CompositionValueRole, ValueVersion>,
    accesses: Vec<EffectAccess>,
    bindings: Vec<LoweredCompositionBinding>,
}

impl<'a> WaveBindingBuilder<'a> {
    fn new(
        layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
        values: &'a mut adapter::SemanticValueMap,
    ) -> Self {
        Self {
            layouts,
            values,
            outputs: BTreeMap::new(),
            accesses: Vec::new(),
            bindings: Vec::new(),
        }
    }

    fn record(
        &mut self,
        arena: &ProofArenaPlan,
        component: usize,
    ) -> Result<DeviceRecordPointerBinding, InvocationShapeError> {
        let requirements = arena
            .composition()
            .requirements
            .components
            .get(component)
            .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
        let params = arena
            .composition()
            .ext_params
            .get(component)
            .ok_or(InvocationShapeError::InvalidCompositionAuthority)?;
        let component = u32::try_from(component).map_err(|_| InvocationShapeError::SizeOverflow)?;
        let trace = requirements
            .source_retention
            .iter()
            .map(|source| {
                self.read_full(CompositionValueRole::DirectEvaluation {
                    plan_column: u32::try_from(source.plan_column)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                })
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let interaction = vec![Some(self.read_full(CompositionValueRole::Descriptor {
            kind: CompositionDescriptorRole::InteractionOffsets,
            index: component,
        })?)];
        let base = if requirements.base_param_words == 0 {
            Vec::new()
        } else {
            vec![Some(self.read_full(CompositionValueRole::Descriptor {
                kind: CompositionDescriptorRole::BaseParams,
                index: component,
            })?)]
        };
        let ext = (0..params.sources.len())
            .map(|slot| {
                self.read_full(CompositionValueRole::ExtParam {
                    component,
                    slot: u32::try_from(slot).map_err(|_| InvocationShapeError::SizeOverflow)?,
                })
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let denominator = vec![Some(self.read_full(CompositionValueRole::Descriptor {
            kind: CompositionDescriptorRole::DenominatorInverses,
            index: component,
        })?)];
        Ok(DeviceRecordPointerBinding {
            fields: [trace, interaction, base, ext, denominator]
                .map(|entries| DeviceRecordPointerFieldBinding { entries })
                .into(),
        })
    }

    fn read_full(
        &mut self,
        role: CompositionValueRole,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let words = self.layout(role)?.word_len;
        self.read(
            role,
            ElementRange {
                start: 0,
                end: words,
            },
        )
    }

    fn read(
        &mut self,
        role: CompositionValueRole,
        elements: ElementRange,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = *self.layout(role)?;
        let semantic = offset_range(arena, elements)?;
        let version = self.values.version(ArenaCatalogValueId(arena.logical.0))?;
        self.push(role, arena, version, semantic, CompositionAccessKind::Read)
    }

    fn write_full(
        &mut self,
        role: CompositionValueRole,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = *self.layout(role)?;
        let version = match self.outputs.get(&role).copied() {
            Some(version) => version,
            None => {
                let version = self.values.allocate_ephemeral()?;
                self.outputs.insert(role, version);
                version
            }
        };
        self.push(
            role,
            arena,
            version,
            ElementRange {
                start: 0,
                end: arena.word_len,
            },
            CompositionAccessKind::Write,
        )
    }

    fn push(
        &mut self,
        role: CompositionValueRole,
        arena: CompositionLayout,
        version: ValueVersion,
        elements: ElementRange,
        kind: CompositionAccessKind,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let binding = EffectBindingId(
            u32::try_from(self.bindings.len()).map_err(|_| InvocationShapeError::SizeOverflow)?,
        );
        let bound = BoundValueRange {
            binding,
            value: ValueRange { version, elements },
        };
        self.accesses.push(match kind {
            CompositionAccessKind::Read => EffectAccess::Read { source: bound },
            CompositionAccessKind::Write => EffectAccess::Write { destination: bound },
            CompositionAccessKind::ReadWriteRequired => {
                return Err(InvocationShapeError::InvalidCompositionBinding)
            }
        });
        self.bindings.push(LoweredCompositionBinding {
            binding,
            role,
            arena,
            version,
            elements,
            kind,
        });
        Ok(binding)
    }

    fn layout(
        &self,
        role: CompositionValueRole,
    ) -> Result<&CompositionLayout, InvocationShapeError> {
        self.layouts
            .get(&role)
            .ok_or(InvocationShapeError::InvalidCompositionBinding)
    }

    fn finish_effect(&self) -> Result<EffectContract, InvocationShapeError> {
        EffectContract::new(self.accesses.clone(), Vec::new())
            .map_err(|_| InvocationShapeError::InvalidCompositionBinding)
    }
}

fn offset_range(
    layout: CompositionLayout,
    relative: ElementRange,
) -> Result<ElementRange, InvocationShapeError> {
    if relative.is_empty() || relative.end > layout.word_len {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(ElementRange {
        start: layout
            .first_word
            .checked_add(relative.start)
            .ok_or(InvocationShapeError::SizeOverflow)?,
        end: layout
            .first_word
            .checked_add(relative.end)
            .ok_or(InvocationShapeError::SizeOverflow)?,
    })
}

fn validate_projected_roles(
    child: &crate::prepared_composition::CompositionChildLaunch,
    effect: &EffectContract,
    bindings: &[LoweredCompositionBinding],
) -> Result<(), InvocationShapeError> {
    let expected_reads = count_role_ranges(
        child
            .effect
            .accesses
            .iter()
            .filter_map(|access| access.source.map(|role| (role, access.elements))),
    );
    let actual_reads = bindings
        .iter()
        .filter(|binding| binding.kind == CompositionAccessKind::Read)
        .map(relative_read_range)
        .collect::<Result<Vec<_>, _>>()?;
    let actual_reads = count_role_ranges(actual_reads.into_iter());
    let expected_writes = count_role_ranges(
        child
            .effect
            .accesses
            .iter()
            .filter_map(|access| access.destination.map(|role| (role, access.elements))),
    );
    let actual_writes = count_role_ranges(
        bindings
            .iter()
            .filter(|binding| binding.kind == CompositionAccessKind::Write)
            .map(|binding| (binding.role, binding.elements)),
    );
    if expected_reads != actual_reads
        || expected_writes != actual_writes
        || child.effect.accesses.len() != bindings.len()
        || effect.accesses().len() != bindings.len()
    {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(())
}

fn relative_read_range(
    binding: &LoweredCompositionBinding,
) -> Result<(CompositionValueRole, ElementRange), InvocationShapeError> {
    Ok((
        binding.role,
        ElementRange {
            start: binding
                .elements
                .start
                .checked_sub(binding.arena.first_word)
                .ok_or(InvocationShapeError::InvalidCompositionBinding)?,
            end: binding
                .elements
                .end
                .checked_sub(binding.arena.first_word)
                .ok_or(InvocationShapeError::InvalidCompositionBinding)?,
        },
    ))
}

fn count_role_ranges(
    roles: impl Iterator<Item = (CompositionValueRole, ElementRange)>,
) -> BTreeMap<(CompositionValueRole, ElementRange), usize> {
    let mut counts = BTreeMap::new();
    for role in roles {
        *counts.entry(role).or_default() += 1;
    }
    counts
}

fn validate_invocation_bindings(
    invocation: &AotInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for argument in &invocation.arguments {
        match &argument.value {
            AotArgumentValue::DeviceRecordPointerGraphValue { root, records } => {
                insert_once(&mut actual, *root)?;
                for record in records {
                    for field in &record.fields {
                        for &binding in field.entries.iter().flatten() {
                            insert_once(&mut actual, binding)?;
                        }
                    }
                }
            }
            AotArgumentValue::DevicePointerRangeSetValue { ranges } => {
                for &binding in ranges {
                    insert_once(&mut actual, binding)?;
                }
            }
            AotArgumentValue::DevicePointer(Some(binding)) => {
                insert_once(&mut actual, *binding)?;
            }
            AotArgumentValue::U32(_) => {}
            _ => return Err(InvocationShapeError::InvalidCompositionBinding),
        }
    }
    if actual != expected
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| ordinal != usize::from(argument.ordinal))
    {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(())
}

fn insert_once(
    bindings: &mut BTreeSet<EffectBindingId>,
    binding: EffectBindingId,
) -> Result<(), InvocationShapeError> {
    bindings
        .insert(binding)
        .then_some(())
        .ok_or(InvocationShapeError::InvalidCompositionBinding)
}

fn argument(ordinal: u8, value: AotArgumentValue) -> AotArgumentBinding {
    AotArgumentBinding { ordinal, value }
}

fn validate_receipt(
    arena: &ProofArenaPlan,
    lowered: &LoweredCompositionWaves,
) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    if lowered.waves.len() != lowered.authority.waves().len()
        || lowered.waves.iter().enumerate().any(|(index, wave)| {
            wave.shard.wave_index() != index
                || wave.shard.effect() != wave.effect.id()
                || wave.shard.partition().kind()
                    == &crate::compiled_proof::PartitionAuthorityKind::Monolithic
        })
        || receipt_digest(&lowered.authority, &lowered.waves)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(())
}

fn receipt_digest(
    authority: &CompositionExecutionAuthority,
    waves: &[LoweredCompositionWave],
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hasher.update(
        &u64::try_from(waves.len())
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    for wave in waves {
        hasher.update(&wave.operation_ordinal.to_le_bytes());
        hasher.update(&wave.operation.identity);
        hasher.update(wave.effect.id().as_bytes());
        hasher.update(
            wave.invocation
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?
                .as_bytes(),
        );
        hasher.update(wave.shard.digest());
    }
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
#[path = "composition_projection_tests.rs"]
mod tests;
