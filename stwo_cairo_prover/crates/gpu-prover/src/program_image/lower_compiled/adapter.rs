//! One-way adapter into the canonical `CompiledProof` invocation IR.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::compiled_proof::{
    AotArgumentBinding, AotArgumentValue, AotInvocation, BoundValueRange, ConstantId, EffectAccess,
    EffectBindingId, EffectContract, ElementRange, ElementType, FixedValueDesc, LayoutAxis, Region,
    ValueDesc, ValueLayout, ValueOrigin, ValueRange, ValueVersion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct FixedU32Value {
    constant: ConstantId,
    value: ValueVersion,
    words: Vec<u32>,
}

/// Explicit semantic authority. Arena catalog IDs are storage inventory and
/// are never cast or inferred into immutable semantic versions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SemanticValueMap {
    current: BTreeMap<ArenaCatalogValueId, ValueVersion>,
    allocations: Vec<(ArenaCatalogValueId, ValueVersion)>,
    fixed_u32_by_content: BTreeMap<Vec<u32>, usize>,
    fixed_u32: Vec<FixedU32Value>,
    next_value: u32,
}

impl SemanticValueMap {
    pub(super) fn new(
        entries: impl IntoIterator<Item = (ArenaCatalogValueId, ValueVersion)>,
    ) -> Result<Self, InvocationShapeError> {
        let mut current = BTreeMap::new();
        let mut versions = BTreeSet::new();
        let mut allocations = Vec::new();
        for (catalog, version) in entries {
            if current.insert(catalog, version).is_some() || !versions.insert(version) {
                return Err(InvocationShapeError::InvalidProgramRole);
            }
            allocations.push((catalog, version));
        }
        if versions
            .iter()
            .enumerate()
            .any(|(index, version)| version.0 as usize != index)
        {
            return Err(InvocationShapeError::InvalidProgramRole);
        }
        Ok(Self {
            current,
            allocations,
            fixed_u32_by_content: BTreeMap::new(),
            fixed_u32: Vec::new(),
            next_value: u32::try_from(versions.len())
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
        })
    }

    pub(super) fn allocate_ordered(
        catalogs: impl IntoIterator<Item = ArenaCatalogValueId>,
    ) -> Result<Self, InvocationShapeError> {
        let mut values = Self::new(std::iter::empty::<(ArenaCatalogValueId, ValueVersion)>())?;
        values.extend_ordered(catalogs)?;
        Ok(values)
    }

    pub(super) fn extend_ordered(
        &mut self,
        catalogs: impl IntoIterator<Item = ArenaCatalogValueId>,
    ) -> Result<(), InvocationShapeError> {
        for catalog in catalogs {
            if self.current.contains_key(&catalog) {
                continue;
            }
            let version = self.allocate_version()?;
            self.current.insert(catalog, version);
            self.allocations.push((catalog, version));
        }
        Ok(())
    }

    pub(super) fn transition(
        &mut self,
        catalog: ArenaCatalogValueId,
    ) -> Result<(ValueVersion, ValueVersion), InvocationShapeError> {
        let source = self.version(catalog)?;
        let destination = self.allocate_version()?;
        self.current.insert(catalog, destination);
        self.allocations.push((catalog, destination));
        Ok((source, destination))
    }

    fn allocate_version(&mut self) -> Result<ValueVersion, InvocationShapeError> {
        let version = ValueVersion(self.next_value);
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        Ok(version)
    }

    pub(super) fn register_fixed_u32(
        &mut self,
        words: Vec<u32>,
    ) -> Result<ValueVersion, InvocationShapeError> {
        if words.is_empty() {
            return Err(InvocationShapeError::InvalidProgramRole);
        }
        if let Some(&index) = self.fixed_u32_by_content.get(words.as_slice()) {
            let fixed = self
                .fixed_u32
                .get(index)
                .ok_or(InvocationShapeError::InvalidProgramRole)?;
            return Ok(fixed.value);
        }
        let constant = ConstantId(
            u32::try_from(self.fixed_u32.len()).map_err(|_| InvocationShapeError::SizeOverflow)?,
        );
        let value = self.allocate_version()?;
        let index = self.fixed_u32.len();
        self.fixed_u32_by_content.insert(words.clone(), index);
        self.fixed_u32.push(FixedU32Value {
            constant,
            value,
            words,
        });
        Ok(value)
    }

    pub(super) fn version(
        &self,
        catalog: ArenaCatalogValueId,
    ) -> Result<ValueVersion, InvocationShapeError> {
        self.current
            .get(&catalog)
            .copied()
            .ok_or(InvocationShapeError::MissingSemanticValueMap(catalog))
    }

    /// Semantic lineage of one reusable arena value in allocation order.
    pub(super) fn versions_for(
        &self,
        catalog: ArenaCatalogValueId,
    ) -> impl Iterator<Item = ValueVersion> + '_ {
        self.allocations
            .iter()
            .filter_map(move |&(candidate, version)| (candidate == catalog).then_some(version))
    }

    pub(super) fn fixed_values(&self) -> Vec<FixedValueDesc> {
        self.fixed_u32
            .iter()
            .map(|fixed| {
                FixedValueDesc::inline_u32(fixed.constant, fixed.value, fixed.words.clone())
            })
            .collect()
    }

    pub(super) fn fixed_value_versions(&self) -> Vec<ValueDesc> {
        self.fixed_u32
            .iter()
            .map(|fixed| ValueDesc {
                version: fixed.value,
                layout: ValueLayout {
                    element: ElementType::U32,
                    axes: vec![LayoutAxis {
                        tag: 0,
                        extent: fixed.words.len(),
                        stride_bytes: core::mem::size_of::<u32>(),
                    }],
                },
                alignment: core::mem::align_of::<u32>(),
                origin: ValueOrigin::Constant(fixed.constant),
                region: Region::FixedData,
            })
            .collect()
    }

    pub(super) fn allocated_versions(&self) -> impl Iterator<Item = ValueVersion> + '_ {
        let catalogs = self.allocations.iter().map(|(_, version)| *version);
        let fixed = self.fixed_u32.iter().map(|fixed| fixed.value);
        catalogs.chain(fixed)
    }

    pub(super) fn allocation_classes(
        &self,
    ) -> (
        BTreeSet<ValueVersion>,
        BTreeSet<ValueVersion>,
        BTreeSet<ValueVersion>,
    ) {
        let mut catalogs = BTreeSet::new();
        let mut catalog_first = BTreeSet::new();
        let mut transitions = BTreeSet::new();
        for &(catalog, version) in &self.allocations {
            if catalogs.insert(catalog) {
                catalog_first.insert(version);
            } else {
                transitions.insert(version);
            }
        }
        let fixed = self.fixed_u32.iter().map(|fixed| fixed.value).collect();
        (catalog_first, transitions, fixed)
    }

    #[cfg(test)]
    pub(super) fn entries(&self) -> impl Iterator<Item = (ArenaCatalogValueId, ValueVersion)> + '_ {
        self.allocations.iter().copied()
    }

    #[cfg(test)]
    pub(super) fn with_unconsumed_transition_for_test(&self) -> Result<Self, InvocationShapeError> {
        let mut changed = self.clone();
        let catalog = changed
            .allocations
            .first()
            .map(|(catalog, _)| *catalog)
            .ok_or(InvocationShapeError::InvalidProgramRole)?;
        changed.transition(catalog)?;
        Ok(changed)
    }
}

/// Descriptor ValueIds stay in the checked source receipt for ShapeExecutable
/// relocation. The compiled effect names only ranges the kernel dereferences.
pub(super) fn compile(
    source: &[SourceArgument],
    values: &mut SemanticValueMap,
) -> Result<(AotInvocation, EffectContract), InvocationShapeError> {
    let mut required = BTreeSet::new();
    for argument in source {
        match argument {
            SourceArgument::PointerTable { entries, .. } => {
                required.extend(
                    entries
                        .iter()
                        .filter(|entry| entry.target.access != InvocationAccess::Inactive)
                        .map(|entry| entry.target.value),
                );
            }
            SourceArgument::DirectPointer { target, .. }
                if target.access != InvocationAccess::Inactive =>
            {
                required.insert(target.value);
            }
            SourceArgument::ScalarArray { .. }
            | SourceArgument::U32 { .. }
            | SourceArgument::DirectPointer { .. } => {}
        }
    }
    if let Some(&missing) = required
        .iter()
        .find(|value| !values.current.contains_key(value))
    {
        return Err(InvocationShapeError::MissingSemanticValueMap(missing));
    }
    let mut accesses = Vec::new();
    let mut next_binding = 0u32;
    let mut arguments = Vec::with_capacity(source.len());
    for (ordinal, argument) in source.iter().enumerate() {
        let expected_ordinal =
            u8::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?;
        let (actual_ordinal, value) = match argument {
            SourceArgument::PointerTable {
                ordinal,
                descriptor_access,
                entries,
                ..
            } => {
                let expected_access = if entries
                    .iter()
                    .any(|entry| entry.target.access != InvocationAccess::Inactive)
                {
                    InvocationAccess::Read
                } else {
                    InvocationAccess::Inactive
                };
                if *descriptor_access != expected_access
                    || entries
                        .iter()
                        .enumerate()
                        .any(|(index, entry)| entry.entry as usize != index)
                {
                    return Err(InvocationShapeError::InvalidProgramRole);
                }
                let entries = entries
                    .iter()
                    .map(|entry| {
                        bind_target(&entry.target, values, &mut next_binding, &mut accesses)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (*ordinal, AotArgumentValue::DevicePointerTable(entries))
            }
            SourceArgument::ScalarArray {
                ordinal, entries, ..
            } => {
                if entries
                    .iter()
                    .enumerate()
                    .any(|(index, entry)| entry.index as usize != index)
                {
                    return Err(InvocationShapeError::InvalidProgramRole);
                }
                let element_count = entries.len();
                let words = entries.iter().map(|entry| entry.value).collect::<Vec<_>>();
                let value = values.register_fixed_u32(words)?;
                let binding = EffectBindingId(next_binding);
                next_binding = next_binding
                    .checked_add(1)
                    .ok_or(InvocationShapeError::SizeOverflow)?;
                let elements = ElementRange::new(0, element_count)
                    .ok_or(InvocationShapeError::InvalidProgramRole)?;
                accesses.push(EffectAccess::Read {
                    source: BoundValueRange {
                        binding,
                        value: ValueRange {
                            version: value,
                            elements,
                        },
                    },
                });
                (
                    *ordinal,
                    AotArgumentValue::DeviceFixedU32 { value, binding },
                )
            }
            SourceArgument::DirectPointer { ordinal, target } => (
                *ordinal,
                AotArgumentValue::DevicePointer(bind_target(
                    target,
                    values,
                    &mut next_binding,
                    &mut accesses,
                )?),
            ),
            SourceArgument::U32 { ordinal, value } => (*ordinal, AotArgumentValue::U32(*value)),
        };
        if actual_ordinal != expected_ordinal {
            return Err(InvocationShapeError::InvalidStructuredAbi);
        }
        arguments.push(AotArgumentBinding {
            ordinal: actual_ordinal,
            value,
        });
    }
    let effect = EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidAdapterEffect)?;
    Ok((AotInvocation { arguments }, effect))
}

fn bind_target(
    target: &InvocationTarget,
    values: &SemanticValueMap,
    next_binding: &mut u32,
    accesses: &mut Vec<EffectAccess>,
) -> Result<Option<EffectBindingId>, InvocationShapeError> {
    if target.access == InvocationAccess::Inactive {
        return Ok(None);
    }
    let elements = ElementRange::new(target.elements.start, target.elements.end)
        .ok_or(InvocationShapeError::InvalidCatalogRange(target.value))?;
    let binding = EffectBindingId(*next_binding);
    *next_binding = next_binding
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let bound = BoundValueRange {
        binding,
        value: ValueRange {
            version: values.version(target.value)?,
            elements,
        },
    };
    accesses.push(match target.access {
        InvocationAccess::Read => EffectAccess::Read { source: bound },
        InvocationAccess::Write => EffectAccess::Write { destination: bound },
        InvocationAccess::Inactive => unreachable!(),
    });
    Ok(Some(binding))
}
