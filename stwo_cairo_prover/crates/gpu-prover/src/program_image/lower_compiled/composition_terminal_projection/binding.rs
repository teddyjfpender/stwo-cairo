//! Semantic-version and effect binding for terminal Composition wrappers.

use super::*;

pub(super) struct TerminalBuilder<'a> {
    layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
    versions: &'a mut BTreeMap<CompositionValueRole, ValueVersion>,
    values: &'a mut adapter::SemanticValueMap,
    accesses: Vec<EffectAccess>,
    bindings: Vec<LoweredCompositionTerminalBinding>,
    next_alias: u32,
}

impl<'a> TerminalBuilder<'a> {
    pub(super) fn new(
        layouts: &'a BTreeMap<CompositionValueRole, CompositionLayout>,
        versions: &'a mut BTreeMap<CompositionValueRole, ValueVersion>,
        values: &'a mut adapter::SemanticValueMap,
    ) -> Self {
        Self {
            layouts,
            versions,
            values,
            accesses: Vec::new(),
            bindings: Vec::new(),
            next_alias: 0,
        }
    }

    pub(super) fn read_ephemeral(
        &mut self,
        role: CompositionValueRole,
        elements: ElementRange,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = self.layout(role)?;
        let version = self.version(role)?;
        self.push(
            role,
            None,
            arena,
            Some(version),
            None,
            elements,
            elements,
            None,
        )
    }

    pub(super) fn read_catalog(
        &mut self,
        role: CompositionValueRole,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = self.layout(role)?;
        let elements = full(arena)?;
        let version = self.values.version(catalog(arena))?;
        if self
            .versions
            .insert(role, version)
            .is_some_and(|old| old != version)
        {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        self.push(
            role,
            None,
            arena,
            Some(version),
            None,
            elements,
            offset(arena, elements)?,
            None,
        )
    }

    pub(super) fn write_catalog(
        &mut self,
        role: CompositionValueRole,
        elements: ElementRange,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = self.layout(role)?;
        let version = self.values.allocate_output(catalog(arena))?;
        if self.versions.insert(role, version).is_some() {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        self.push(
            role,
            Some(role),
            arena,
            None,
            Some(version),
            elements,
            offset(arena, elements)?,
            None,
        )
    }

    pub(super) fn transition_ephemeral(
        &mut self,
        source: CompositionValueRole,
        destination: CompositionValueRole,
        elements: ElementRange,
        discipline: InPlaceDiscipline,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = self.same_layout(source, destination)?;
        let source_version = self.version(source)?;
        let destination_version = self.values.allocate_ephemeral()?;
        if self
            .versions
            .insert(destination, destination_version)
            .is_some()
        {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        self.push(
            source,
            Some(destination),
            arena,
            Some(source_version),
            Some(destination_version),
            elements,
            elements,
            Some(discipline),
        )
    }

    pub(super) fn transition_catalog(
        &mut self,
        source: CompositionValueRole,
        destination: CompositionValueRole,
        elements: ElementRange,
        discipline: InPlaceDiscipline,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let arena = self.same_layout(source, destination)?;
        let source_version = self.version(source)?;
        let (mapped_source, destination_version) = self.values.transition(catalog(arena))?;
        if mapped_source != source_version
            || self
                .versions
                .insert(destination, destination_version)
                .is_some()
        {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        self.push(
            source,
            Some(destination),
            arena,
            Some(source_version),
            Some(destination_version),
            elements,
            offset(arena, elements)?,
            Some(discipline),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        source_role: CompositionValueRole,
        destination_role: Option<CompositionValueRole>,
        arena: CompositionLayout,
        source_version: Option<ValueVersion>,
        destination_version: Option<ValueVersion>,
        role_elements: ElementRange,
        elements: ElementRange,
        discipline: Option<InPlaceDiscipline>,
    ) -> Result<EffectBindingId, InvocationShapeError> {
        let binding = EffectBindingId(
            u32::try_from(self.bindings.len()).map_err(|_| InvocationShapeError::SizeOverflow)?,
        );
        let source = source_version.map(|version| bound(binding, version, elements));
        let destination = destination_version.map(|version| bound(binding, version, elements));
        let kind = match (source, destination, discipline) {
            (Some(source), None, None) => {
                self.accesses.push(EffectAccess::Read { source });
                CompositionAccessKind::Read
            }
            (None, Some(destination), None) => {
                self.accesses.push(EffectAccess::Write { destination });
                CompositionAccessKind::Write
            }
            (Some(source), Some(destination), Some(discipline)) => {
                let alias = InPlaceAliasAuthority {
                    id: InPlaceAliasId(self.next_alias),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline,
                };
                self.next_alias = self
                    .next_alias
                    .checked_add(1)
                    .ok_or(InvocationShapeError::SizeOverflow)?;
                self.accesses.push(EffectAccess::ReadWrite {
                    source,
                    destination,
                    in_place: Some(alias),
                });
                CompositionAccessKind::ReadWriteRequired
            }
            _ => return Err(InvocationShapeError::InvalidCompositionBinding),
        };
        self.bindings.push(LoweredCompositionTerminalBinding {
            binding,
            source_role: source_version.map(|_| source_role),
            destination_role,
            arena,
            source_version,
            destination_version,
            role_elements,
            elements,
            kind,
        });
        Ok(binding)
    }

    fn version(&self, role: CompositionValueRole) -> Result<ValueVersion, InvocationShapeError> {
        self.versions
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionBinding)
    }

    fn layout(
        &self,
        role: CompositionValueRole,
    ) -> Result<CompositionLayout, InvocationShapeError> {
        self.layouts
            .get(&role)
            .copied()
            .ok_or(InvocationShapeError::InvalidCompositionBinding)
    }

    fn same_layout(
        &self,
        source: CompositionValueRole,
        destination: CompositionValueRole,
    ) -> Result<CompositionLayout, InvocationShapeError> {
        let source = self.layout(source)?;
        let destination = self.layout(destination)?;
        if source.logical != destination.logical
            || source.first_word != destination.first_word
            || source.word_len != destination.word_len
            || source.alignment_words != destination.alignment_words
        {
            return Err(InvocationShapeError::InvalidCompositionBinding);
        }
        Ok(source)
    }

    pub(super) fn finish(
        self,
        ordinal: usize,
        operation: &CompositionOperation,
        invocation: AotInvocation,
    ) -> Result<LoweredCompositionTerminalOperation, InvocationShapeError> {
        let effect = EffectContract::new(self.accesses, Vec::new())
            .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?;
        validate_invocation_bindings(&invocation, &effect)?;
        Ok(LoweredCompositionTerminalOperation {
            operation_ordinal: u32::try_from(ordinal)
                .map_err(|_| InvocationShapeError::SizeOverflow)?,
            operation: operation.clone(),
            invocation,
            effect,
            bindings: self.bindings,
        })
    }
}
