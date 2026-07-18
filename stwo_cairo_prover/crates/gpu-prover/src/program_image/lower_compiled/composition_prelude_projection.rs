//! Proof-wide semantic projection of the exact Composition prelude.
//!
//! The prepared-composition compiler remains the sole authority for operation
//! order, ABI, effects, and launch geometry. This module only replaces its
//! local role ordinals with proof-wide semantic versions. Missing transcript
//! or relation values fail transactionally; no proof value is fabricated.

use std::collections::{BTreeMap, BTreeSet};

use stwo::core::fields::m31::M31;

use crate::arena_plan::ProofArenaPlan;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, BoundValueRange, EffectAccess,
    EffectBindingId, EffectContract, ElementRange, LaunchGeometry, ValueRange, ValueVersion,
};
use crate::composition_plan::CompositionExtParamSource;
use crate::prepared_composition::{
    CompositionAbi, CompositionAccessKind, CompositionDescriptorRole,
    CompositionExecutionAuthority, CompositionInvocationValue, CompositionLayout,
    CompositionOperation, CompositionOperationKind, CompositionValueRole,
};
use crate::transcript_plan::CairoTranscriptOutput;

use super::{adapter, ArenaCatalogValueId, InvocationShapeError};

mod validation;
use validation::{offset_range, receipt_digest, validate_invocation_bindings, validate_receipt};

const RECEIPT_DOMAIN: &[u8] = b"stwo-cairo.lowered-composition-prelude.v1\0";
const SECURE_WORDS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LoweredCompositionPreludeValueKind {
    CatalogInput(ArenaCatalogValueId),
    Fixed,
    DynamicOutput,
    CatalogOutput(ArenaCatalogValueId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionPreludeValue {
    pub(super) role: CompositionValueRole,
    pub(super) arena: CompositionLayout,
    pub(super) version: ValueVersion,
    pub(super) kind: LoweredCompositionPreludeValueKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionPreludeBinding {
    pub(super) binding: EffectBindingId,
    pub(super) role: CompositionValueRole,
    pub(super) arena: CompositionLayout,
    pub(super) version: ValueVersion,
    pub(super) role_elements: ElementRange,
    pub(super) elements: ElementRange,
    pub(super) kind: CompositionAccessKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionPreludeOperation {
    operation_ordinal: u32,
    operation: CompositionOperation,
    invocation: AotInvocation,
    effect: EffectContract,
    launch: LaunchGeometry,
    bindings: Vec<LoweredCompositionPreludeBinding>,
}

impl LoweredCompositionPreludeOperation {
    pub(super) const fn operation_ordinal(&self) -> u32 {
        self.operation_ordinal
    }

    pub(super) const fn operation(&self) -> &CompositionOperation {
        &self.operation
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

    pub(super) fn bindings(&self) -> &[LoweredCompositionPreludeBinding] {
        &self.bindings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredCompositionPrelude {
    authority: CompositionExecutionAuthority,
    operations: Vec<LoweredCompositionPreludeOperation>,
    values: BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue>,
    digest: [u8; 32],
}

impl LoweredCompositionPrelude {
    pub(super) const fn authority(&self) -> &CompositionExecutionAuthority {
        &self.authority
    }

    pub(super) fn operations(&self) -> &[LoweredCompositionPreludeOperation] {
        &self.operations
    }

    pub(super) fn values(&self) -> &BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue> {
        &self.values
    }

    pub(super) const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub(super) fn wave_version(
        &self,
        role: CompositionValueRole,
    ) -> Result<ValueVersion, InvocationShapeError> {
        Ok(self.wave_value(role)?.version)
    }

    pub(super) fn wave_value(
        &self,
        role: CompositionValueRole,
    ) -> Result<LoweredCompositionPreludeValue, InvocationShapeError> {
        if !matches!(
            role,
            CompositionValueRole::ExtParam { .. } | CompositionValueRole::RandomCoefficientPowers
        ) {
            return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
        }
        self.values
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)
    }

    pub(super) fn validate_against_values(
        &self,
        arena: &ProofArenaPlan,
        values: &adapter::SemanticValueMap,
    ) -> Result<(), InvocationShapeError> {
        validate_receipt(arena, self)?;
        let allocated = values.allocated_versions().collect::<BTreeSet<_>>();
        for value in self.values.values() {
            if !allocated.contains(&value.version) {
                return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
            }
            match value.kind {
                LoweredCompositionPreludeValueKind::CatalogInput(catalog)
                | LoweredCompositionPreludeValueKind::CatalogOutput(catalog)
                    if values.version(catalog)? != value.version =>
                {
                    return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Lower both exact prelude operations or leave `values` untouched.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredCompositionPrelude, InvocationShapeError> {
    let authority = exact_authority(arena)?;
    let layouts = exact_layouts(&authority)?;
    validate_random_coefficient_catalog(arena, &layouts)?;
    let descriptors = DescriptorWords::derive(arena)?;

    let mut next_values = values.clone();
    let mut builder = PreludeBuilder::new(arena, &layouts, descriptors, &mut next_values);
    builder.install_constant_ext_params()?;
    let operations = authority
        .operations()
        .iter()
        .take(2)
        .enumerate()
        .map(|(ordinal, operation)| builder.lower_operation(ordinal, operation))
        .collect::<Result<Vec<_>, _>>()?;
    builder.validate_wave_inputs()?;
    let projected_values = builder.projected;
    let digest = receipt_digest(&authority, &operations, &projected_values)?;
    let lowered = LoweredCompositionPrelude {
        authority,
        operations,
        values: projected_values,
        digest,
    };
    lowered.validate_against_values(arena, &next_values)?;
    *values = next_values;
    Ok(lowered)
}

pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &LoweredCompositionPrelude,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidCompositionPreludeBinding)
    }
}

pub(super) fn required_upstream_catalogs(
    arena: &ProofArenaPlan,
) -> Result<Vec<ArenaCatalogValueId>, InvocationShapeError> {
    let authority = exact_authority(arena)?;
    let layouts = exact_layouts(&authority)?;
    validate_random_coefficient_catalog(arena, &layouts)?;
    let mut catalogs = BTreeSet::new();
    for operation in authority.operations().iter().take(2) {
        let [child] = operation.children.as_slice() else {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        };
        for access in &child.effect.accesses {
            let Some(role) = access.source else {
                continue;
            };
            if matches!(
                role,
                CompositionValueRole::Descriptor { .. } | CompositionValueRole::RandomCoefficient
            ) {
                continue;
            }
            catalogs.insert(ArenaCatalogValueId(
                layouts
                    .get(&role)
                    .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)?
                    .logical
                    .0,
            ));
        }
    }
    Ok(catalogs.into_iter().collect())
}

fn exact_authority(
    arena: &ProofArenaPlan,
) -> Result<CompositionExecutionAuthority, InvocationShapeError> {
    let authority = CompositionExecutionAuthority::compile(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    let [materialize, powers, ..] = authority.operations() else {
        return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
    };
    if !matches!(
        materialize.kind,
        CompositionOperationKind::MaterializeExtParams { .. }
    ) || materialize.abi != CompositionAbi::MaterializeExtParamsV1
        || !matches!(
            powers.kind,
            CompositionOperationKind::GenerateDescendingPowers { .. }
        )
        || powers.abi != CompositionAbi::GenerateDescendingPowersV1
    {
        return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
    }
    for (operation, argument_count) in [(materialize, 11), (powers, 4)] {
        if operation.children.len() != 1
            || operation.invocation.arguments.len() != argument_count
            || !matches!(
                operation
                    .invocation
                    .arguments
                    .last()
                    .map(|argument| &argument.value),
                Some(CompositionInvocationValue::ExecutionStream)
            )
        {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        }
    }
    Ok(authority)
}

fn exact_layouts(
    authority: &CompositionExecutionAuthority,
) -> Result<BTreeMap<CompositionValueRole, CompositionLayout>, InvocationShapeError> {
    let layouts = authority
        .layouts()
        .iter()
        .map(|layout| (layout.role, *layout))
        .collect::<BTreeMap<_, _>>();
    if layouts.len() != authority.layouts().len() {
        return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
    }
    Ok(layouts)
}

fn validate_random_coefficient_catalog(
    arena: &ProofArenaPlan,
    layouts: &BTreeMap<CompositionValueRole, CompositionLayout>,
) -> Result<(), InvocationShapeError> {
    let output_id = CairoTranscriptOutput::CompositionRandomCoefficient
        .id()
        .map_err(|_| InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    let transcript = arena
        .transcript()
        .outputs
        .iter()
        .find_map(|(id, binding)| (*id == output_id).then_some(*binding))
        .ok_or(InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    let layout = layouts
        .get(&CompositionValueRole::RandomCoefficient)
        .ok_or(InvocationShapeError::InvalidCompositionPreludeAuthority)?;
    if layout.logical != transcript.logical
        || layout.first_word != 0
        || layout.word_len != transcript.len_words
        || layout.word_len != SECURE_WORDS
    {
        return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
    }
    Ok(())
}

struct DescriptorWords {
    kinds: Vec<u32>,
    indices: Vec<u32>,
    scales: Vec<u32>,
}

impl DescriptorWords {
    fn derive(arena: &ProofArenaPlan) -> Result<Self, InvocationShapeError> {
        let composition = arena.composition();
        if composition.ext_params.len() != composition.plan.components.len() {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        }
        let mut words = Self {
            kinds: Vec::with_capacity(composition.requirements.dynamic_ext_param_count),
            indices: Vec::with_capacity(composition.requirements.dynamic_ext_param_count),
            scales: Vec::with_capacity(composition.requirements.dynamic_ext_param_count),
        };
        let mut claimed_index = 0u32;
        for (params, component) in composition
            .ext_params
            .iter()
            .zip(&composition.plan.components)
        {
            if params.sources != component.ext_param_sources {
                return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
            }
            for source in &params.sources {
                let (kind, index, scale) = match *source {
                    CompositionExtParamSource::Constant(_) => continue,
                    CompositionExtParamSource::LookupZ => (0, 0, 1),
                    CompositionExtParamSource::LookupAlphaPower(power) => (1, power, 1),
                    CompositionExtParamSource::LookupAlphaPowerScaled { power, scale } => {
                        (1, power, scale.0)
                    }
                    CompositionExtParamSource::ClaimedSumScaled => {
                        let denominator = 1u32
                            .checked_shl(component.trace_log_size)
                            .ok_or(InvocationShapeError::SizeOverflow)?;
                        let index = claimed_index;
                        claimed_index = claimed_index
                            .checked_add(1)
                            .ok_or(InvocationShapeError::SizeOverflow)?;
                        (2, index, M31::from_u32_unchecked(denominator).inverse().0)
                    }
                };
                words.kinds.push(kind);
                words.indices.push(index);
                words.scales.push(scale);
            }
        }
        if words.kinds.len() != composition.requirements.dynamic_ext_param_count
            || words.indices.len() != words.kinds.len()
            || words.scales.len() != words.kinds.len()
            || claimed_index as usize != composition.requirements.claimed_sum_count
        {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        }
        Ok(words)
    }

    fn for_role(&self, role: CompositionValueRole) -> Result<Vec<u32>, InvocationShapeError> {
        match role {
            CompositionValueRole::Descriptor {
                kind: CompositionDescriptorRole::DynamicSourceKinds,
                index: 0,
            } => Ok(self.kinds.clone()),
            CompositionValueRole::Descriptor {
                kind: CompositionDescriptorRole::DynamicSourceIndices,
                index: 0,
            } => Ok(self.indices.clone()),
            CompositionValueRole::Descriptor {
                kind: CompositionDescriptorRole::DynamicScales,
                index: 0,
            } => Ok(self.scales.clone()),
            _ => Err(InvocationShapeError::InvalidCompositionPreludeBinding),
        }
    }
}

struct PreludeBuilder<'a> {
    arena: &'a ProofArenaPlan,
    layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
    descriptors: DescriptorWords,
    semantic: &'a mut adapter::SemanticValueMap,
    projected: BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue>,
}

impl<'a> PreludeBuilder<'a> {
    fn new(
        arena: &'a ProofArenaPlan,
        layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
        descriptors: DescriptorWords,
        semantic: &'a mut adapter::SemanticValueMap,
    ) -> Self {
        Self {
            arena,
            layouts,
            descriptors,
            semantic,
            projected: BTreeMap::new(),
        }
    }

    fn install_constant_ext_params(&mut self) -> Result<(), InvocationShapeError> {
        for (component, params) in self.arena.composition().ext_params.iter().enumerate() {
            for (slot, source) in params.sources.iter().enumerate() {
                let CompositionExtParamSource::Constant(value) = source else {
                    continue;
                };
                let role = CompositionValueRole::ExtParam {
                    component: u32::try_from(component)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                    slot: u32::try_from(slot).map_err(|_| InvocationShapeError::SizeOverflow)?,
                };
                let words = value.to_m31_array().map(|coordinate| coordinate.0).to_vec();
                let version = self.semantic.register_fixed_u32(words)?;
                self.insert(role, version, LoweredCompositionPreludeValueKind::Fixed)?;
            }
        }
        Ok(())
    }

    fn lower_operation(
        &mut self,
        ordinal: usize,
        operation: &CompositionOperation,
    ) -> Result<LoweredCompositionPreludeOperation, InvocationShapeError> {
        let [child] = operation.children.as_slice() else {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        };
        let mut bindings = Vec::with_capacity(child.effect.accesses.len());
        let mut accesses = Vec::with_capacity(child.effect.accesses.len());
        for (index, access) in child.effect.accesses.iter().enumerate() {
            let binding = EffectBindingId(
                u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
            );
            if access.binding != binding {
                return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
            }
            let (projected, effect) = self.project_access(*access)?;
            bindings.push(projected);
            accesses.push(effect);
        }
        let effect = EffectContract::new(accesses, Vec::new())
            .map_err(|_| InvocationShapeError::InvalidCompositionPreludeBinding)?;
        let invocation = project_invocation(operation, &bindings, &self.projected)?;
        validate_invocation_bindings(&invocation, &effect)?;
        let launch = LaunchGeometry {
            grid: child.launch.grid,
            block: child.launch.block,
            cluster: None,
            dynamic_shared_bytes: child.launch.dynamic_shared_bytes,
            cooperative: child.launch.cooperative,
        };
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        }
        Ok(LoweredCompositionPreludeOperation {
            operation_ordinal: u32::try_from(ordinal)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            operation: operation.clone(),
            invocation,
            effect,
            launch,
            bindings,
        })
    }

    fn project_access(
        &mut self,
        access: crate::prepared_composition::CompositionValueAccess,
    ) -> Result<(LoweredCompositionPreludeBinding, EffectAccess), InvocationShapeError> {
        let (role, kind) = match (access.kind, access.source, access.destination) {
            (CompositionAccessKind::Read, Some(role), None) => {
                self.ensure_source(role)?;
                (role, CompositionAccessKind::Read)
            }
            (CompositionAccessKind::Write, None, Some(role)) => {
                self.ensure_destination(role)?;
                (role, CompositionAccessKind::Write)
            }
            _ => return Err(InvocationShapeError::InvalidCompositionPreludeBinding),
        };
        let value = *self
            .projected
            .get(&role)
            .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)?;
        if access.elements.is_empty() || access.elements.end > value.arena.word_len {
            return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
        }
        let elements = match value.kind {
            LoweredCompositionPreludeValueKind::Fixed
            | LoweredCompositionPreludeValueKind::DynamicOutput => access.elements,
            LoweredCompositionPreludeValueKind::CatalogInput(_)
            | LoweredCompositionPreludeValueKind::CatalogOutput(_) => {
                offset_range(value.arena, access.elements)?
            }
        };
        let bound = BoundValueRange {
            binding: access.binding,
            value: ValueRange {
                version: value.version,
                elements,
            },
        };
        let effect = match kind {
            CompositionAccessKind::Read => EffectAccess::Read { source: bound },
            CompositionAccessKind::Write => EffectAccess::Write { destination: bound },
            CompositionAccessKind::ReadWriteRequired => unreachable!(),
        };
        Ok((
            LoweredCompositionPreludeBinding {
                binding: access.binding,
                role,
                arena: value.arena,
                version: value.version,
                role_elements: access.elements,
                elements,
                kind,
            },
            effect,
        ))
    }

    fn ensure_source(&mut self, role: CompositionValueRole) -> Result<(), InvocationShapeError> {
        if self.projected.contains_key(&role) {
            return Ok(());
        }
        let layout = self.layout(role)?;
        if matches!(role, CompositionValueRole::Descriptor { .. }) {
            let words = self.descriptors.for_role(role)?;
            if words.len() != layout.word_len {
                return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
            }
            let version = self.semantic.register_fixed_u32(words)?;
            self.insert(role, version, LoweredCompositionPreludeValueKind::Fixed)
        } else {
            let catalog = ArenaCatalogValueId(layout.logical.0);
            let version = self.semantic.version(catalog)?;
            self.insert(
                role,
                version,
                LoweredCompositionPreludeValueKind::CatalogInput(catalog),
            )
        }
    }

    fn ensure_destination(
        &mut self,
        role: CompositionValueRole,
    ) -> Result<(), InvocationShapeError> {
        if self.projected.contains_key(&role) {
            return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
        }
        let layout = self.layout(role)?;
        match role {
            CompositionValueRole::ExtParam { .. } => {
                let version = self.semantic.allocate_ephemeral()?;
                self.insert(
                    role,
                    version,
                    LoweredCompositionPreludeValueKind::DynamicOutput,
                )
            }
            CompositionValueRole::RandomCoefficientPowers => {
                let catalog = ArenaCatalogValueId(layout.logical.0);
                let version = self.semantic.allocate_output(catalog)?;
                self.insert(
                    role,
                    version,
                    LoweredCompositionPreludeValueKind::CatalogOutput(catalog),
                )
            }
            _ => Err(InvocationShapeError::InvalidCompositionPreludeBinding),
        }
    }

    fn insert(
        &mut self,
        role: CompositionValueRole,
        version: ValueVersion,
        kind: LoweredCompositionPreludeValueKind,
    ) -> Result<(), InvocationShapeError> {
        let arena = self.layout(role)?;
        if self
            .projected
            .insert(
                role,
                LoweredCompositionPreludeValue {
                    role,
                    arena,
                    version,
                    kind,
                },
            )
            .is_some()
        {
            return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
        }
        Ok(())
    }

    fn layout(
        &self,
        role: CompositionValueRole,
    ) -> Result<CompositionLayout, InvocationShapeError> {
        self.layouts
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)
    }

    fn validate_wave_inputs(&self) -> Result<(), InvocationShapeError> {
        for &role in self.layouts.keys() {
            if matches!(
                role,
                CompositionValueRole::ExtParam { .. }
                    | CompositionValueRole::RandomCoefficientPowers
            ) && !self.projected.contains_key(&role)
            {
                return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
            }
        }
        Ok(())
    }
}

fn project_invocation(
    operation: &CompositionOperation,
    bindings: &[LoweredCompositionPreludeBinding],
    values: &BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue>,
) -> Result<AotInvocation, InvocationShapeError> {
    let mut arguments = Vec::with_capacity(operation.invocation.arguments.len() - 1);
    for source in &operation.invocation.arguments {
        if matches!(source.value, CompositionInvocationValue::ExecutionStream) {
            if source.ordinal as usize + 1 != operation.invocation.arguments.len() {
                return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
            }
            continue;
        }
        if source.ordinal as usize != arguments.len() {
            return Err(InvocationShapeError::InvalidCompositionPreludeAuthority);
        }
        let value = match &source.value {
            CompositionInvocationValue::Access(index) => {
                project_access_argument(*index, bindings, values)?
            }
            CompositionInvocationValue::PointerTable { pointee_accesses } => {
                AotArgumentValue::DevicePointerTable(
                    pointee_accesses
                        .iter()
                        .map(|index| {
                            index
                                .map(|index| {
                                    bindings
                                        .get(index as usize)
                                        .filter(|binding| binding.binding.0 == index)
                                        .map(|binding| binding.binding)
                                        .ok_or(
                                            InvocationShapeError::InvalidCompositionPreludeBinding,
                                        )
                                })
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            CompositionInvocationValue::U32(value) => AotArgumentValue::U32(*value),
            CompositionInvocationValue::ExecutionStream => unreachable!(),
        };
        arguments.push(AotArgumentBinding {
            ordinal: source.ordinal,
            value,
        });
    }
    Ok(AotInvocation { arguments })
}

fn project_access_argument(
    index: u32,
    bindings: &[LoweredCompositionPreludeBinding],
    values: &BTreeMap<CompositionValueRole, LoweredCompositionPreludeValue>,
) -> Result<AotArgumentValue, InvocationShapeError> {
    let selected = bindings
        .get(index as usize)
        .filter(|binding| binding.binding.0 == index)
        .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)?;
    let value = values
        .get(&selected.role)
        .ok_or(InvocationShapeError::InvalidCompositionPreludeBinding)?;
    if value.kind == LoweredCompositionPreludeValueKind::Fixed {
        return Ok(AotArgumentValue::DeviceFixedU32 {
            value: selected.version,
            binding: selected.binding,
        });
    }
    let reached = bindings
        .iter()
        .filter(|binding| binding.role == selected.role && binding.kind == selected.kind)
        .map(|binding| binding.binding)
        .collect::<Vec<_>>();
    if reached.len() > 1 {
        if reached.first() != Some(&selected.binding) {
            return Err(InvocationShapeError::InvalidCompositionPreludeBinding);
        }
        Ok(AotArgumentValue::DevicePointerRangeSetValue { ranges: reached })
    } else {
        Ok(AotArgumentValue::DevicePointer(Some(selected.binding)))
    }
}

#[cfg(test)]
#[path = "composition_prelude_projection_tests.rs"]
mod tests;
