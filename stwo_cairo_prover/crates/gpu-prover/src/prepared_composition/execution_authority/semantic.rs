use std::collections::{BTreeMap, BTreeSet};

use super::encoding::{
    child_identity, effect_identity, invocation_identity, operation_identities,
    static_wrapper_source_identity,
};
use super::*;
use crate::compiled_proof::{
    BoundValueRange, EffectAccess, InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement,
    InPlaceDiscipline, ValueRange, ValueVersion,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct AccessSpec {
    pub kind: CompositionAccessKind,
    pub source: Option<CompositionValueRole>,
    pub destination: Option<CompositionValueRole>,
    pub elements: ElementRange,
    pub discipline: Option<InPlaceDiscipline>,
}

impl AccessSpec {
    pub(super) const fn read(role: CompositionValueRole, words: usize) -> Self {
        Self {
            kind: CompositionAccessKind::Read,
            source: Some(role),
            destination: None,
            elements: ElementRange {
                start: 0,
                end: words,
            },
            discipline: None,
        }
    }

    pub(super) const fn write(role: CompositionValueRole, words: usize) -> Self {
        Self {
            kind: CompositionAccessKind::Write,
            source: None,
            destination: Some(role),
            elements: ElementRange {
                start: 0,
                end: words,
            },
            discipline: None,
        }
    }

    pub(super) const fn read_range(role: CompositionValueRole, start: usize, end: usize) -> Self {
        Self {
            kind: CompositionAccessKind::Read,
            source: Some(role),
            destination: None,
            elements: ElementRange { start, end },
            discipline: None,
        }
    }

    pub(super) const fn required_alias(
        source: CompositionValueRole,
        destination: CompositionValueRole,
        words: usize,
        discipline: InPlaceDiscipline,
    ) -> Self {
        Self {
            kind: CompositionAccessKind::ReadWriteRequired,
            source: Some(source),
            destination: Some(destination),
            elements: ElementRange {
                start: 0,
                end: words,
            },
            discipline: Some(discipline),
        }
    }
}

pub(super) struct RoleCatalog {
    layouts: Vec<CompositionLayout>,
    indices: BTreeMap<CompositionValueRole, usize>,
}

pub(super) struct RelocationCatalog {
    layouts: Vec<CompositionRelocationLayout>,
    indices: BTreeMap<CompositionRelocationRole, usize>,
}

impl RelocationCatalog {
    pub(super) fn new() -> Self {
        Self {
            layouts: Vec::new(),
            indices: BTreeMap::new(),
        }
    }

    pub(super) fn insert(
        &mut self,
        role: CompositionRelocationRole,
        logical: LogicalBufferId,
        first_word: usize,
        word_len: usize,
        alignment_words: usize,
    ) -> Result<(), CompositionAuthorityError> {
        if word_len == 0 || alignment_words == 0 || first_word % alignment_words != 0 {
            return Err(CompositionAuthorityError::ShapeDrift("relocation layout"));
        }
        let layout = CompositionRelocationLayout {
            role,
            logical,
            first_word,
            word_len,
            alignment_words,
        };
        match self.indices.get(&role).copied() {
            Some(index) if self.layouts[index] == layout => Ok(()),
            Some(_) => Err(CompositionAuthorityError::ShapeDrift(
                "relocation alias drift",
            )),
            None => {
                self.indices.insert(role, self.layouts.len());
                self.layouts.push(layout);
                Ok(())
            }
        }
    }

    pub(super) fn finish(self) -> Vec<CompositionRelocationLayout> {
        self.layouts
    }
}

impl RoleCatalog {
    pub(super) fn new() -> Self {
        Self {
            layouts: Vec::new(),
            indices: BTreeMap::new(),
        }
    }

    pub(super) fn insert(
        &mut self,
        role: CompositionValueRole,
        logical: LogicalBufferId,
        first_word: usize,
        word_len: usize,
        alignment_words: usize,
    ) -> Result<(), CompositionAuthorityError> {
        if word_len == 0 || alignment_words == 0 || first_word % alignment_words != 0 {
            return Err(CompositionAuthorityError::ShapeDrift("role layout"));
        }
        let layout = CompositionLayout {
            role,
            logical,
            first_word,
            word_len,
            alignment_words,
        };
        match self.indices.get(&role).copied() {
            Some(index) if self.layouts[index] == layout => Ok(()),
            Some(_) => Err(CompositionAuthorityError::ShapeDrift("role alias drift")),
            None => {
                self.indices.insert(role, self.layouts.len());
                self.layouts.push(layout);
                Ok(())
            }
        }
    }

    pub(super) fn layout(
        &self,
        role: CompositionValueRole,
    ) -> Result<&CompositionLayout, CompositionAuthorityError> {
        self.indices
            .get(&role)
            .and_then(|&index| self.layouts.get(index))
            .ok_or(CompositionAuthorityError::MissingLogicalRole(
                "composition semantic role",
            ))
    }

    fn version(
        &self,
        role: CompositionValueRole,
    ) -> Result<ValueVersion, CompositionAuthorityError> {
        let index =
            *self
                .indices
                .get(&role)
                .ok_or(CompositionAuthorityError::MissingLogicalRole(
                    "composition value version",
                ))?;
        Ok(ValueVersion(
            u32::try_from(index).map_err(|_| CompositionAuthorityError::SizeOverflow)?,
        ))
    }

    pub(super) fn finish(self) -> Vec<CompositionLayout> {
        self.layouts
    }

    pub(super) fn effect(
        &self,
        specs: impl IntoIterator<Item = AccessSpec>,
    ) -> Result<CompositionEffect, CompositionAuthorityError> {
        let mut seen = BTreeSet::new();
        let specs = specs
            .into_iter()
            .filter(|spec| seen.insert(*spec))
            .collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(CompositionAuthorityError::InvalidEffect);
        }
        let mut accesses = Vec::with_capacity(specs.len());
        let mut semantic = Vec::with_capacity(specs.len());
        let mut alias = 0u32;
        for (index, spec) in specs.into_iter().enumerate() {
            if spec.elements.is_empty() {
                return Err(CompositionAuthorityError::InvalidEffect);
            }
            for role in [spec.source, spec.destination].into_iter().flatten() {
                if !(ElementRange {
                    start: 0,
                    end: self.layout(role)?.word_len,
                })
                .contains(spec.elements)
                {
                    return Err(CompositionAuthorityError::InvalidEffect);
                }
            }
            let binding = EffectBindingId(
                u32::try_from(index).map_err(|_| CompositionAuthorityError::SizeOverflow)?,
            );
            let bound = |role| -> Result<BoundValueRange, CompositionAuthorityError> {
                Ok(BoundValueRange {
                    binding,
                    value: ValueRange {
                        version: self.version(role)?,
                        elements: spec.elements,
                    },
                })
            };
            let compiled = match spec.kind {
                CompositionAccessKind::Read => EffectAccess::Read {
                    source: bound(
                        spec.source
                            .ok_or(CompositionAuthorityError::InvalidEffect)?,
                    )?,
                },
                CompositionAccessKind::Write => EffectAccess::Write {
                    destination: bound(
                        spec.destination
                            .ok_or(CompositionAuthorityError::InvalidEffect)?,
                    )?,
                },
                CompositionAccessKind::ReadWriteRequired => {
                    let in_place = InPlaceAliasAuthority {
                        id: InPlaceAliasId(alias),
                        requirement: InPlaceAliasRequirement::Required,
                        discipline: spec
                            .discipline
                            .ok_or(CompositionAuthorityError::InvalidEffect)?,
                    };
                    alias = alias
                        .checked_add(1)
                        .ok_or(CompositionAuthorityError::SizeOverflow)?;
                    EffectAccess::ReadWrite {
                        source: bound(
                            spec.source
                                .ok_or(CompositionAuthorityError::InvalidEffect)?,
                        )?,
                        destination: bound(
                            spec.destination
                                .ok_or(CompositionAuthorityError::InvalidEffect)?,
                        )?,
                        in_place: Some(in_place),
                    }
                }
            };
            accesses.push(compiled);
            semantic.push(CompositionValueAccess {
                binding,
                kind: spec.kind,
                source: spec.source,
                destination: spec.destination,
                elements: spec.elements,
            });
        }
        let contract = EffectContract::new(accesses, Vec::new())?;
        let identity = effect_identity(&semantic, &contract)?;
        if identity == ZERO_IDENTITY {
            return Err(CompositionAuthorityError::InvalidIdentity);
        }
        Ok(CompositionEffect {
            accesses: semantic,
            contract,
            identity,
        })
    }
}

pub(super) fn child(
    symbol: impl Into<Box<str>>,
    launch: CompositionLaunchGeometry,
    parameters: Vec<(&'static str, u32)>,
    effect: CompositionEffect,
) -> Result<CompositionChildLaunch, CompositionAuthorityError> {
    if launch.grid.contains(&0) || launch.block.contains(&0) {
        return Err(CompositionAuthorityError::InvalidExecution);
    }
    let symbol = symbol.into();
    let identity = child_identity(&symbol, launch, &parameters, &effect)?;
    Ok(CompositionChildLaunch {
        symbol,
        launch,
        parameters,
        effect,
        identity,
    })
}

pub(super) fn operation(
    source_root: [u8; 32],
    kind: CompositionOperationKind,
    abi: CompositionAbi,
    argument_specs: Vec<(&'static str, CompositionInvocationValue)>,
    embedded_pointer_tables: Vec<CompositionEmbeddedPointerTable>,
    children: Vec<CompositionChildLaunch>,
    generated_source: Option<[u8; 32]>,
) -> Result<CompositionOperation, CompositionAuthorityError> {
    let total_accesses = children
        .iter()
        .try_fold(0usize, |total, child| {
            total.checked_add(child.effect.accesses.len())
        })
        .ok_or(CompositionAuthorityError::SizeOverflow)?;
    let mut arguments = Vec::with_capacity(argument_specs.len());
    for (ordinal, (name, value)) in argument_specs.into_iter().enumerate() {
        let valid = match &value {
            CompositionInvocationValue::Access(index) => (*index as usize) < total_accesses,
            CompositionInvocationValue::PointerTable { pointee_accesses } => {
                valid_pointer_table(pointee_accesses, total_accesses)
            }
            CompositionInvocationValue::U32(_) | CompositionInvocationValue::ExecutionStream => {
                true
            }
        };
        if !valid {
            return Err(CompositionAuthorityError::InvalidInvocation);
        }
        arguments.push(CompositionInvocationArgument {
            ordinal: u8::try_from(ordinal).map_err(|_| CompositionAuthorityError::SizeOverflow)?,
            name,
            value,
        });
    }
    if embedded_pointer_tables
        .iter()
        .any(|table| !valid_pointer_table(&table.pointee_accesses, total_accesses))
    {
        return Err(CompositionAuthorityError::InvalidInvocation);
    }
    let invocation = CompositionInvocation {
        identity: invocation_identity(abi, &arguments, &embedded_pointer_tables)?,
        arguments,
        embedded_pointer_tables,
    };
    let source_identity = match generated_source {
        Some(identity) if identity != ZERO_IDENTITY => identity,
        Some(_) => return Err(CompositionAuthorityError::InvalidIdentity),
        None => static_wrapper_source_identity(source_root, abi, &children)?,
    };
    let (abi_identity, effect_identity, execution_identity, identity) =
        operation_identities(&kind, abi, &invocation, &children, source_identity)?;
    Ok(CompositionOperation {
        kind,
        abi,
        invocation,
        children,
        source_identity,
        abi_identity,
        effect_identity,
        execution_identity,
        identity,
    })
}

fn valid_pointer_table(entries: &[Option<u32>], total_accesses: usize) -> bool {
    !entries.is_empty()
        && entries
            .iter()
            .flatten()
            .all(|index| (*index as usize) < total_accesses)
}
