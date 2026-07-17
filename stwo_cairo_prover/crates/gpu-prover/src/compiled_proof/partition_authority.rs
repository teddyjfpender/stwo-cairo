use std::collections::BTreeSet;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionEffectProjection {
    ReplicatedRead { binding: EffectBindingId },
    ContiguousAxisSlice { binding: EffectBindingId },
}

impl PartitionEffectProjection {
    pub const fn binding(self) -> EffectBindingId {
        match self {
            Self::ReplicatedRead { binding } | Self::ContiguousAxisSlice { binding } => binding,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionGridAxis {
    X,
    Y,
    Z,
}

impl PartitionGridAxis {
    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    const fn tag(self) -> u8 {
        self.index() as u8
    }
}

/// Exact transformation from one full-domain invocation to a shard invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionLaunchDerivation {
    start_argument: u8,
    length_argument: u8,
    grid_axis: PartitionGridAxis,
    elements_per_grid_unit: usize,
}

impl PartitionLaunchDerivation {
    pub fn new(
        start_argument: u8,
        length_argument: u8,
        grid_axis: PartitionGridAxis,
        elements_per_grid_unit: usize,
    ) -> Result<Self, CompiledProofError> {
        if start_argument == length_argument || elements_per_grid_unit == 0 {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }
        Ok(Self {
            start_argument,
            length_argument,
            grid_axis,
            elements_per_grid_unit,
        })
    }

    pub const fn start_argument(self) -> u8 {
        self.start_argument
    }

    pub const fn length_argument(self) -> u8 {
        self.length_argument
    }

    pub const fn grid_axis(self) -> PartitionGridAxis {
        self.grid_axis
    }

    pub const fn elements_per_grid_unit(self) -> usize {
        self.elements_per_grid_unit
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalPartitionJoin {
    AscendingDisjointRangeUnion,
}

/// Typed proof that one operation can be instantiated over aligned contiguous
/// shards without changing its semantic value versions or canonical result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactPartitionAuthority {
    axis: u16,
    domain: ElementRange,
    granularity: usize,
    alignment_bytes: usize,
    projections: Box<[PartitionEffectProjection]>,
    launch: PartitionLaunchDerivation,
    join: CanonicalPartitionJoin,
}

/// One exact shard derived from the validated full-domain AOT launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactShardInvocation {
    invocation: AotInvocation,
    launch: LaunchGeometry,
}

impl ExactShardInvocation {
    pub const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }

    pub const fn launch(&self) -> LaunchGeometry {
        self.launch
    }
}

impl ExactPartitionAuthority {
    pub fn new(
        axis: u16,
        domain: ElementRange,
        granularity: usize,
        alignment_bytes: usize,
        projections: Vec<PartitionEffectProjection>,
        launch: PartitionLaunchDerivation,
    ) -> Result<Self, CompiledProofError> {
        let authority = Self {
            axis,
            domain,
            granularity,
            alignment_bytes,
            projections: projections.into_boxed_slice(),
            launch,
            join: CanonicalPartitionJoin::AscendingDisjointRangeUnion,
        };
        authority.validate_structure()?;
        Ok(authority)
    }

    pub const fn axis(&self) -> u16 {
        self.axis
    }

    pub const fn domain(&self) -> ElementRange {
        self.domain
    }

    pub const fn granularity(&self) -> usize {
        self.granularity
    }

    pub const fn alignment_bytes(&self) -> usize {
        self.alignment_bytes
    }

    pub fn projections(&self) -> &[PartitionEffectProjection] {
        &self.projections
    }

    pub const fn launch(&self) -> PartitionLaunchDerivation {
        self.launch
    }

    pub const fn join(&self) -> CanonicalPartitionJoin {
        self.join
    }

    fn materialize_shard(
        &self,
        full_invocation: &AotInvocation,
        full_launch: LaunchGeometry,
        shard: ElementRange,
    ) -> Result<ExactShardInvocation, CompiledProofError> {
        self.validate_structure()?;
        validate_full_launch_and_abi(full_invocation, full_launch, self)?;
        if shard.is_empty() || !self.domain.contains(shard) {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }
        let start = shard
            .start
            .checked_sub(self.domain.start)
            .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
        let end = shard
            .end
            .checked_sub(self.domain.start)
            .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
        if start % self.granularity != 0 || end % self.granularity != 0 {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }

        let shard_start = u32::try_from(shard.start)
            .map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
        let shard_length = u32::try_from(shard.len())
            .map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
        u32::try_from(shard.end).map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
        let grid_units = checked_grid_units(shard.len(), self.launch.elements_per_grid_unit)?;

        let mut invocation = full_invocation.clone();
        invocation.arguments[usize::from(self.launch.start_argument)].value =
            AotArgumentValue::U32(shard_start);
        invocation.arguments[usize::from(self.launch.length_argument)].value =
            AotArgumentValue::U32(shard_length);
        let mut launch = full_launch;
        launch.grid[self.launch.grid_axis.index()] = grid_units;
        Ok(ExactShardInvocation { invocation, launch })
    }

    pub(super) fn validate_structure(&self) -> Result<(), CompiledProofError> {
        if self.domain.is_empty()
            || self.domain.start != 0
            || u32::try_from(self.domain.end).is_err()
            || self.granularity == 0
            || self.granularity > self.domain.len()
            || self.domain.len() % self.granularity != 0
            || self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
            || self.projections.is_empty()
            || self
                .projections
                .windows(2)
                .any(|pair| pair[0].binding() >= pair[1].binding())
            || self.launch.start_argument == self.launch.length_argument
            || self.launch.elements_per_grid_unit == 0
            || self.granularity % self.launch.elements_per_grid_unit != 0
        {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }
        Ok(())
    }
}

impl CompiledProof {
    /// Derive one aligned shard from this validated proof's canonical AOT operation.
    pub fn materialize_exact_shard(
        &self,
        operation: OpId,
        shard: ElementRange,
    ) -> Result<ExactShardInvocation, CompiledProofError> {
        let operation = self
            .operation(operation)
            .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
        let partition = self
            .partitions()
            .iter()
            .find(|partition| partition.id() == operation.partition)
            .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
        let PartitionAuthorityKind::Exact(authority) = partition.kind() else {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        };
        let ExecutionPrimitive::AotKernel { launch, .. } = operation.primitive else {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        };
        let invocation = operation
            .invocation
            .as_ref()
            .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
        authority.materialize_shard(invocation, launch, shard)
    }
}

pub(super) fn encode_exact_partition(
    out: &mut Vec<u8>,
    authority: &ExactPartitionAuthority,
) -> Result<(), CompiledProofError> {
    authority.validate_structure()?;
    out.extend_from_slice(&authority.axis.to_le_bytes());
    push_size(out, authority.domain.start)?;
    push_size(out, authority.domain.end)?;
    push_size(out, authority.granularity)?;
    push_size(out, authority.alignment_bytes)?;
    push_size(out, authority.projections.len())?;
    for projection in authority.projections.iter().copied() {
        match projection {
            PartitionEffectProjection::ReplicatedRead { binding } => {
                out.push(0);
                out.extend_from_slice(&binding.0.to_le_bytes());
            }
            PartitionEffectProjection::ContiguousAxisSlice { binding } => {
                out.push(1);
                out.extend_from_slice(&binding.0.to_le_bytes());
            }
        }
    }
    out.push(authority.launch.start_argument);
    out.push(authority.launch.length_argument);
    out.push(authority.launch.grid_axis.tag());
    push_size(out, authority.launch.elements_per_grid_unit)?;
    match authority.join {
        CanonicalPartitionJoin::AscendingDisjointRangeUnion => out.push(0),
    }
    Ok(())
}

pub(super) fn validate_exact_partition(
    input: &CompiledProofInput,
    operation: &OpNode,
    effect: &EffectContract,
    authority: &ExactPartitionAuthority,
) -> Result<(), CompiledProofError> {
    authority.validate_structure()?;
    let ExecutionPrimitive::AotKernel { launch, .. } = &operation.primitive else {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    };
    if launch.cooperative || launch.cluster.is_some() {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    validate_launch_and_abi(operation, *launch, authority)?;

    for access in effect.accesses() {
        match access {
            EffectAccess::Atomic { .. } => {
                return Err(CompiledProofError::InvalidPartitionAuthority);
            }
            EffectAccess::ReadWrite {
                source,
                destination,
                in_place: Some(alias),
            } if alias.discipline != InPlaceDiscipline::ElementWiseReadBeforeWrite
                || [source.binding, destination.binding]
                    .into_iter()
                    .any(|binding| {
                        !authority.projections.iter().any(|projection| {
                            matches!(
                                projection,
                                PartitionEffectProjection::ContiguousAxisSlice {
                                    binding: sliced
                                } if *sliced == binding
                            )
                        })
                    }) =>
            {
                return Err(CompiledProofError::InvalidPartitionAuthority);
            }
            _ => {}
        }
    }
    for global in effect.module_globals() {
        if input
            .module_global_initializers
            .get(global.initializer.0 as usize)
            .is_none_or(|initializer| {
                initializer.id() != global.initializer || !initializer.immutable()
            })
        {
            return Err(CompiledProofError::InvalidPartitionAuthority);
        }
    }

    let expected = effect
        .accesses()
        .iter()
        .flat_map(|access| [access.source(), access.destination()])
        .flatten()
        .map(|range| range.binding)
        .collect::<BTreeSet<_>>();
    let actual = authority
        .projections
        .iter()
        .map(|projection| projection.binding())
        .collect::<BTreeSet<_>>();
    if actual.len() != authority.projections.len() || actual != expected {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }

    let mut sliced_destination = false;
    for projection in authority.projections.iter().copied() {
        let binding = projection.binding();
        let mut source = None;
        let mut destination = None;
        for access in effect.accesses() {
            if access
                .source()
                .is_some_and(|range| range.binding == binding)
            {
                if source.replace(*access.source().unwrap()).is_some() {
                    return Err(CompiledProofError::InvalidPartitionAuthority);
                }
            }
            if access
                .destination()
                .is_some_and(|range| range.binding == binding)
            {
                if destination
                    .replace(*access.destination().unwrap())
                    .is_some()
                {
                    return Err(CompiledProofError::InvalidPartitionAuthority);
                }
            }
        }
        match projection {
            PartitionEffectProjection::ReplicatedRead { .. } => {
                if source.is_none() || destination.is_some() {
                    return Err(CompiledProofError::InvalidPartitionAuthority);
                }
            }
            PartitionEffectProjection::ContiguousAxisSlice { .. } => {
                sliced_destination |= destination.is_some();
                let ranges = [source, destination]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                if ranges.is_empty() {
                    return Err(CompiledProofError::InvalidPartitionAuthority);
                }
                for range in ranges {
                    validate_contiguous_slice(input, range, authority)?;
                }
            }
        }
    }
    if !sliced_destination {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    Ok(())
}

fn validate_launch_and_abi(
    operation: &OpNode,
    launch: LaunchGeometry,
    authority: &ExactPartitionAuthority,
) -> Result<(), CompiledProofError> {
    let invocation = operation
        .invocation
        .as_ref()
        .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
    validate_full_launch_and_abi(invocation, launch, authority)
}

fn validate_full_launch_and_abi(
    invocation: &AotInvocation,
    launch: LaunchGeometry,
    authority: &ExactPartitionAuthority,
) -> Result<(), CompiledProofError> {
    if launch.cooperative
        || launch.cluster.is_some()
        || invocation
            .arguments
            .iter()
            .enumerate()
            .any(|(ordinal, argument)| usize::from(argument.ordinal) != ordinal)
    {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    let start = u32::try_from(authority.domain.start)
        .map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
    let length = u32::try_from(authority.domain.len())
        .map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
    u32::try_from(authority.domain.end)
        .map_err(|_| CompiledProofError::InvalidPartitionAuthority)?;
    require_u32_argument(invocation, authority.launch.start_argument, start)?;
    require_u32_argument(invocation, authority.launch.length_argument, length)?;

    let units = checked_grid_units(
        authority.domain.len(),
        authority.launch.elements_per_grid_unit,
    )?;
    if launch.grid[authority.launch.grid_axis.index()] != units {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    Ok(())
}

fn checked_grid_units(
    elements: usize,
    elements_per_grid_unit: usize,
) -> Result<u32, CompiledProofError> {
    if elements == 0 || elements_per_grid_unit == 0 {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    let units = (elements / elements_per_grid_unit)
        .checked_add(usize::from(elements % elements_per_grid_unit != 0))
        .ok_or(CompiledProofError::SizeOverflow)?;
    u32::try_from(units).map_err(|_| CompiledProofError::InvalidPartitionAuthority)
}

fn require_u32_argument(
    invocation: &AotInvocation,
    ordinal: u8,
    expected: u32,
) -> Result<(), CompiledProofError> {
    let argument = invocation
        .arguments
        .get(usize::from(ordinal))
        .filter(|argument| argument.ordinal == ordinal)
        .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
    if argument.value != AotArgumentValue::U32(expected) {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    Ok(())
}

fn validate_contiguous_slice(
    input: &CompiledProofInput,
    range: BoundValueRange,
    authority: &ExactPartitionAuthority,
) -> Result<(), CompiledProofError> {
    let value = input
        .values
        .get(range.value.version.0 as usize)
        .filter(|value| value.version == range.value.version)
        .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
    let axis_index = value
        .layout
        .axes
        .iter()
        .position(|axis| axis.tag == authority.axis)
        .ok_or(CompiledProofError::InvalidPartitionAuthority)?;
    let axis = value.layout.axes[axis_index];
    if authority.domain
        != (ElementRange {
            start: 0,
            end: axis.extent,
        })
        || value.layout.axes[axis_index + 1..]
            .iter()
            .any(|axis| axis.extent != 1)
        || value.alignment < value.layout.element.bytes
        || authority.alignment_bytes < value.layout.element.bytes
        || value.alignment < authority.alignment_bytes
    {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }

    let inner_elements = value.layout.axes[..axis_index]
        .iter()
        .try_fold(1usize, |product, axis| product.checked_mul(axis.extent))
        .ok_or(CompiledProofError::SizeOverflow)?;
    let start = authority
        .domain
        .start
        .checked_mul(inner_elements)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let end = authority
        .domain
        .end
        .checked_mul(inner_elements)
        .ok_or(CompiledProofError::SizeOverflow)?;
    if range.value.elements != (ElementRange { start, end }) {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }

    let bytes_per_axis_element = inner_elements
        .checked_mul(value.layout.element.bytes)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let start_bytes = authority
        .domain
        .start
        .checked_mul(bytes_per_axis_element)
        .ok_or(CompiledProofError::SizeOverflow)?;
    let granularity_bytes = authority
        .granularity
        .checked_mul(bytes_per_axis_element)
        .ok_or(CompiledProofError::SizeOverflow)?;
    if start_bytes % authority.alignment_bytes != 0
        || granularity_bytes % authority.alignment_bytes != 0
    {
        return Err(CompiledProofError::InvalidPartitionAuthority);
    }
    Ok(())
}

fn push_size(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledProofError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}
