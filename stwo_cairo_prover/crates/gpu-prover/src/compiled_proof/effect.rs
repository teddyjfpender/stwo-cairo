use std::collections::BTreeSet;

use super::*;

const MODULE_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.module.v2\0";
const EFFECT_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.effect.v2\0";
const KERNEL_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.kernel.v2\0";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModuleIdentity {
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl ModuleIdentity {
    pub fn new(canonical_encoding: Vec<u8>) -> Result<Self, CompiledProofError> {
        if canonical_encoding.is_empty() {
            return Err(CompiledProofError::EmptyModuleIdentity);
        }
        let digest = digest(MODULE_DOMAIN, &canonical_encoding)?;
        Ok(Self {
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            digest,
        })
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectContractId([u8; 32]);

impl EffectContractId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}

impl ByteRange {
    pub const fn new(start: usize, end: usize) -> Option<Self> {
        if start < end {
            Some(Self { start, end })
        } else {
            None
        }
    }

    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueRange {
    pub version: ValueVersion,
    pub elements: ElementRange,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BoundValueRange {
    pub binding: EffectBindingId,
    pub value: ValueRange,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InPlaceAliasRequirement {
    Permitted,
    Required,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InPlaceDiscipline {
    ElementWiseReadBeforeWrite,
    BlockBarrierPhases,
    CooperativeGridPhases,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InPlaceAliasAuthority {
    pub id: InPlaceAliasId,
    pub requirement: InPlaceAliasRequirement,
    pub discipline: InPlaceDiscipline,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AtomicOperation {
    /// Exact wrapping CUDA `atomicAdd` on one `u32` element. Deterministic
    /// canonical reduction still requires explicit partial versions + merge
    /// operations; this authority only describes the audited memory effect.
    AddU32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectAccess {
    Read {
        source: BoundValueRange,
    },
    Write {
        destination: BoundValueRange,
    },
    ReadWrite {
        source: BoundValueRange,
        destination: BoundValueRange,
        /// `None` forbids physical aliasing. An authority permits or requires
        /// only this exact source/destination transition.
        in_place: Option<InPlaceAliasAuthority>,
    },
    Atomic {
        source: BoundValueRange,
        destination: BoundValueRange,
        operation: AtomicOperation,
        /// Atomic access is admitted only as an audited required in-place
        /// transition between distinct semantic versions.
        in_place: InPlaceAliasAuthority,
    },
}

impl EffectAccess {
    pub const fn source(&self) -> Option<&BoundValueRange> {
        match self {
            Self::Read { source }
            | Self::ReadWrite { source, .. }
            | Self::Atomic { source, .. } => Some(source),
            Self::Write { .. } => None,
        }
    }

    pub fn source_mut(&mut self) -> Option<&mut BoundValueRange> {
        match self {
            Self::Read { source }
            | Self::ReadWrite { source, .. }
            | Self::Atomic { source, .. } => Some(source),
            Self::Write { .. } => None,
        }
    }

    pub const fn destination(&self) -> Option<&BoundValueRange> {
        match self {
            Self::Write { destination }
            | Self::ReadWrite { destination, .. }
            | Self::Atomic { destination, .. } => Some(destination),
            Self::Read { .. } => None,
        }
    }

    pub const fn in_place(&self) -> Option<&InPlaceAliasAuthority> {
        match self {
            Self::ReadWrite { in_place, .. } => in_place.as_ref(),
            Self::Atomic { in_place, .. } => Some(in_place),
            Self::Read { .. } | Self::Write { .. } => None,
        }
    }

    fn first_binding(&self) -> EffectBindingId {
        match (self.source(), self.destination()) {
            (Some(source), Some(destination)) => source.binding.min(destination.binding),
            (Some(source), None) => source.binding,
            (None, Some(destination)) => destination.binding,
            (None, None) => unreachable!("every effect has a range"),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModuleGlobalEffect {
    pub module: ModuleIdentity,
    pub symbol: Box<[u8]>,
    pub bytes: ByteRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectContract {
    id: EffectContractId,
    accesses: Box<[EffectAccess]>,
    module_globals: Box<[ModuleGlobalEffect]>,
    canonical_encoding: Box<[u8]>,
}

impl EffectContract {
    pub fn new(
        accesses: Vec<EffectAccess>,
        module_globals: Vec<ModuleGlobalEffect>,
    ) -> Result<Self, CompiledProofError> {
        validate_accesses(&accesses)?;
        validate_module_globals(&module_globals)?;
        let canonical_encoding = encode_effect(&accesses, &module_globals)?;
        let id = EffectContractId(digest(EFFECT_DOMAIN, &canonical_encoding)?);
        Ok(Self {
            id,
            accesses: accesses.into_boxed_slice(),
            module_globals: module_globals.into_boxed_slice(),
            canonical_encoding: canonical_encoding.into_boxed_slice(),
        })
    }

    pub const fn id(&self) -> EffectContractId {
        self.id
    }

    pub fn accesses(&self) -> &[EffectAccess] {
        &self.accesses
    }

    pub fn module_globals(&self) -> &[ModuleGlobalEffect] {
        &self.module_globals
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub fn in_place_alias(&self, id: InPlaceAliasId) -> Option<&EffectAccess> {
        self.accesses
            .iter()
            .find(|access| access.in_place().is_some_and(|alias| alias.id == id))
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        let canonical = encode_effect(&self.accesses, &self.module_globals)?;
        Ok(canonical == self.canonical_encoding.as_ref()
            && self.id.0 == digest(EFFECT_DOMAIN, &canonical)?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AotKernelAuthority {
    id: AotKernelId,
    module: ModuleIdentity,
    semantic_encoding: Box<[u8]>,
    execution_build_encoding: Box<[u8]>,
    accepted_effects: Box<[EffectContractId]>,
    canonical_encoding: Box<[u8]>,
    digest: [u8; 32],
}

impl AotKernelAuthority {
    pub fn new(
        id: AotKernelId,
        module: ModuleIdentity,
        semantic_encoding: Vec<u8>,
        execution_build_encoding: Vec<u8>,
        accepted_effects: Vec<EffectContractId>,
    ) -> Result<Self, CompiledProofError> {
        if id.0 == 0 || semantic_encoding.is_empty() || execution_build_encoding.is_empty() {
            return Err(CompiledProofError::EmptyKernelIdentity(id));
        }
        if accepted_effects.is_empty() {
            return Err(CompiledProofError::EmptyKernelEffectAuthority(id));
        }
        if accepted_effects.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CompiledProofError::NonCanonicalKernelEffects(id));
        }
        let canonical_encoding = encode_kernel(
            id,
            &module,
            &semantic_encoding,
            &execution_build_encoding,
            &accepted_effects,
        )?;
        let digest = digest(KERNEL_DOMAIN, &canonical_encoding)?;
        Ok(Self {
            id,
            module,
            semantic_encoding: semantic_encoding.into_boxed_slice(),
            execution_build_encoding: execution_build_encoding.into_boxed_slice(),
            accepted_effects: accepted_effects.into_boxed_slice(),
            canonical_encoding: canonical_encoding.into_boxed_slice(),
            digest,
        })
    }

    pub const fn id(&self) -> AotKernelId {
        self.id
    }

    pub const fn module(&self) -> &ModuleIdentity {
        &self.module
    }

    pub fn semantic_encoding(&self) -> &[u8] {
        &self.semantic_encoding
    }

    pub fn execution_build_encoding(&self) -> &[u8] {
        &self.execution_build_encoding
    }

    pub fn accepted_effects(&self) -> &[EffectContractId] {
        &self.accepted_effects
    }

    pub fn canonical_encoding(&self) -> &[u8] {
        &self.canonical_encoding
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub(crate) fn has_valid_identity(&self) -> Result<bool, CompiledProofError> {
        let canonical = encode_kernel(
            self.id,
            &self.module,
            &self.semantic_encoding,
            &self.execution_build_encoding,
            &self.accepted_effects,
        )?;
        Ok(canonical == self.canonical_encoding.as_ref()
            && self.digest == digest(KERNEL_DOMAIN, &canonical)?)
    }
}

fn validate_accesses(accesses: &[EffectAccess]) -> Result<(), CompiledProofError> {
    if accesses.is_empty()
        || accesses
            .windows(2)
            .any(|pair| pair[0].first_binding() >= pair[1].first_binding())
    {
        return Err(CompiledProofError::NonCanonicalEffectBindings);
    }
    let mut bindings = BTreeSet::new();
    let mut aliases = BTreeSet::new();
    for access in accesses {
        let local = [access.source(), access.destination()]
            .into_iter()
            .flatten()
            .map(|range| range.binding)
            .collect::<BTreeSet<_>>();
        if local.iter().any(|binding| !bindings.insert(*binding)) {
            return Err(CompiledProofError::NonCanonicalEffectBindings);
        }
        for range in [access.source(), access.destination()]
            .into_iter()
            .flatten()
        {
            if range.value.elements.is_empty() {
                return Err(CompiledProofError::InvalidEffectRange);
            }
        }
        match access {
            EffectAccess::Read { .. } | EffectAccess::Write { .. } => {}
            EffectAccess::ReadWrite {
                source,
                destination,
                in_place,
            } => {
                validate_transition(*source, *destination)?;
                if source.binding == destination.binding
                    && !in_place
                        .is_some_and(|alias| alias.requirement == InPlaceAliasRequirement::Required)
                {
                    return Err(CompiledProofError::InvalidValueTransition);
                }
                if let Some(alias) = in_place {
                    if !aliases.insert(alias.id) {
                        return Err(CompiledProofError::NonCanonicalEffectAliases);
                    }
                }
            }
            EffectAccess::Atomic {
                source,
                destination,
                in_place,
                ..
            } => {
                validate_transition(*source, *destination)?;
                if source.binding != destination.binding
                    || in_place.requirement != InPlaceAliasRequirement::Required
                    || !aliases.insert(in_place.id)
                {
                    return Err(CompiledProofError::InvalidValueTransition);
                }
            }
        }
    }
    if bindings
        .iter()
        .enumerate()
        .any(|(index, binding)| binding.0 as usize != index)
    {
        return Err(CompiledProofError::NonCanonicalEffectBindings);
    }
    if aliases
        .iter()
        .enumerate()
        .any(|(index, alias)| alias.0 as usize != index)
    {
        return Err(CompiledProofError::NonCanonicalEffectAliases);
    }
    Ok(())
}

fn validate_transition(
    source: BoundValueRange,
    destination: BoundValueRange,
) -> Result<(), CompiledProofError> {
    if source.value.version == destination.value.version
        || source.value.elements.len() != destination.value.elements.len()
    {
        return Err(CompiledProofError::InvalidValueTransition);
    }
    Ok(())
}

fn validate_module_globals(globals: &[ModuleGlobalEffect]) -> Result<(), CompiledProofError> {
    for global in globals {
        if global.symbol.is_empty()
            || global.symbol.contains(&0)
            || global.bytes.start >= global.bytes.end
        {
            return Err(CompiledProofError::InvalidModuleGlobalEffect);
        }
    }
    if globals.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CompiledProofError::NonCanonicalModuleGlobals);
    }
    Ok(())
}

fn encode_effect(
    accesses: &[EffectAccess],
    globals: &[ModuleGlobalEffect],
) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Encoder::new(EFFECT_DOMAIN);
    out.count(accesses.len())?;
    for access in accesses {
        out.access(access)?;
    }
    out.count(globals.len())?;
    for global in globals {
        out.bytes(global.module.canonical_encoding())?;
        out.raw(global.module.digest());
        out.bytes(&global.symbol)?;
        out.byte_range(global.bytes)?;
        out.byte(0); // immutable module-global Read
    }
    Ok(out.finish())
}

fn encode_kernel(
    id: AotKernelId,
    module: &ModuleIdentity,
    semantics: &[u8],
    build: &[u8],
    effects: &[EffectContractId],
) -> Result<Vec<u8>, CompiledProofError> {
    let mut out = Encoder::new(KERNEL_DOMAIN);
    out.u32(id.0);
    out.bytes(module.canonical_encoding())?;
    out.raw(module.digest());
    out.bytes(semantics)?;
    out.bytes(build)?;
    out.count(effects.len())?;
    for effect in effects {
        out.raw(effect.as_bytes());
    }
    Ok(out.finish())
}

fn digest(domain: &[u8], canonical: &[u8]) -> Result<[u8; 32], CompiledProofError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(
        &u64::try_from(canonical.len())
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(canonical);
    Ok(*hasher.finalize().as_bytes())
}

struct Encoder(Vec<u8>);

impl Encoder {
    fn new(domain: &[u8]) -> Self {
        Self(domain.to_vec())
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn size(&mut self, value: usize) -> Result<(), CompiledProofError> {
        self.raw(
            &u64::try_from(value)
                .map_err(|_| CompiledProofError::SizeOverflow)?
                .to_le_bytes(),
        );
        Ok(())
    }

    fn count(&mut self, value: usize) -> Result<(), CompiledProofError> {
        self.size(value)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CompiledProofError> {
        self.count(bytes.len())?;
        self.raw(bytes);
        Ok(())
    }

    fn elements(&mut self, range: ElementRange) -> Result<(), CompiledProofError> {
        self.size(range.start)?;
        self.size(range.end)
    }

    fn byte_range(&mut self, range: ByteRange) -> Result<(), CompiledProofError> {
        self.size(range.start)?;
        self.size(range.end)
    }

    fn bound(&mut self, range: BoundValueRange) -> Result<(), CompiledProofError> {
        self.u32(range.binding.0);
        self.u32(range.value.version.0);
        self.elements(range.value.elements)
    }

    fn access(&mut self, access: &EffectAccess) -> Result<(), CompiledProofError> {
        match access {
            EffectAccess::Read { source } => {
                self.byte(0);
                self.bound(*source)?;
            }
            EffectAccess::Write { destination } => {
                self.byte(1);
                self.bound(*destination)?;
            }
            EffectAccess::ReadWrite {
                source,
                destination,
                in_place,
            } => {
                self.byte(2);
                self.bound(*source)?;
                self.bound(*destination)?;
                self.alias(in_place.as_ref());
            }
            EffectAccess::Atomic {
                source,
                destination,
                operation,
                in_place,
            } => {
                self.byte(3);
                self.bound(*source)?;
                self.bound(*destination)?;
                self.byte(match operation {
                    AtomicOperation::AddU32 => 0,
                });
                self.alias(Some(in_place));
            }
        }
        Ok(())
    }

    fn alias(&mut self, alias: Option<&InPlaceAliasAuthority>) {
        let Some(alias) = alias else {
            self.byte(0);
            return;
        };
        self.byte(1);
        self.u32(alias.id.0);
        self.byte(match alias.requirement {
            InPlaceAliasRequirement::Permitted => 0,
            InPlaceAliasRequirement::Required => 1,
        });
        self.byte(match alias.discipline {
            InPlaceDiscipline::ElementWiseReadBeforeWrite => 0,
            InPlaceDiscipline::BlockBarrierPhases => 1,
            InPlaceDiscipline::CooperativeGridPhases => 2,
        });
    }
}
