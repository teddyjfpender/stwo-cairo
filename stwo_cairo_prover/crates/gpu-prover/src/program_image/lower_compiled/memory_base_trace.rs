//! Exact post-witness lowering for the prepared Cairo memory Base traces.
//!
//! This is one sealed semantic receipt over the proof-wide value allocator.
//! Its steps mirror the real Rust prepared graph: address, then value/rc9_9
//! pairs for each big part, then the small value/rc9_9 pair.

use stwo_backend_cuda::{
    MemoryBaseTraceContract, MemoryBaseTraceEffectRole, MemoryBaseTraceLinkedContract,
    MemoryBaseTraceStepContract, MemoryBaseTraceStepKind,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::ArenaBinding;
use crate::compiled_proof::{
    AotInvocation, EffectBindingId, EffectContract, ElementRange, InPlaceAliasAuthority,
    StaticCudaLaunchIdentity, StaticCudaWrapperAuthority, StaticCudaWrapperId, ValueVersion,
};

mod bindings;
mod semantic;

use bindings::{exact_inventory, requirements};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MemoryBaseTraceValueBinding {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MemoryBaseTraceTransition {
    pub(super) arena: ArenaBinding,
    pub(super) value: ArenaCatalogValueId,
    pub(super) elements: ElementRange,
    pub(super) binding: EffectBindingId,
    pub(super) source: ValueVersion,
    pub(super) destination: ValueVersion,
    pub(super) alias: InPlaceAliasAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredMemoryBaseTraceStep {
    contract_ordinal: u32,
    kind: MemoryBaseTraceStepKind,
    part: TracePartId,
    pub(super) reads: Vec<MemoryBaseTraceValueBinding>,
    pub(super) writes: Vec<MemoryBaseTraceValueBinding>,
    pub(super) atomic: Option<MemoryBaseTraceTransition>,
    invocation: AotInvocation,
    effect: EffectContract,
}

impl LoweredMemoryBaseTraceStep {
    pub(super) const fn kind(&self) -> MemoryBaseTraceStepKind {
        self.kind
    }

    pub(super) const fn part(&self) -> TracePartId {
        self.part
    }

    pub(super) const fn invocation(&self) -> &AotInvocation {
        &self.invocation
    }

    pub(super) const fn effect(&self) -> &EffectContract {
        &self.effect
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredMemoryBaseTrace {
    contract: MemoryBaseTraceContract,
    steps: Vec<LoweredMemoryBaseTraceStep>,
}

impl LoweredMemoryBaseTrace {
    pub(super) const fn contract(&self) -> &MemoryBaseTraceContract {
        &self.contract
    }

    pub(super) fn steps(&self) -> &[LoweredMemoryBaseTraceStep] {
        &self.steps
    }
}

#[derive(Clone)]
struct ExactValue {
    catalog: BaseCatalogValue,
    arena: ArenaBinding,
}

struct ExactPart {
    part: TracePartId,
    sources: Vec<ExactValue>,
    counts: ExactValue,
    outputs: Vec<ExactValue>,
}

struct ExactInventory {
    raw_address: ExactValue,
    address_counts: ExactValue,
    address_outputs: Vec<ExactValue>,
    big_parts: Vec<ExactPart>,
    small_part: ExactPart,
    rc99_lut: ExactValue,
    rc99_counts: ExactValue,
}

/// Lower the optional memory graph transactionally after all witness writers.
pub(super) fn lower_stage(
    arena: &ProofArenaPlan,
    values: &mut adapter::SemanticValueMap,
) -> Result<Option<LoweredMemoryBaseTrace>, InvocationShapeError> {
    let Some(workspace) = arena
        .multiplicity()
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?
        .memory_traces
        .as_ref()
    else {
        return Ok(None);
    };
    let requirements = requirements(arena, workspace)?;
    let contract = MemoryBaseTraceContract::compile(&requirements)
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    let catalog = BaseProducerCatalog::compile(arena)?;
    let inventory = exact_inventory(arena, &catalog, workspace)?;
    let mut next_values = values.clone();
    let steps = lower_steps(&contract, inventory, &mut next_values)?;
    let lowered = LoweredMemoryBaseTrace { contract, steps };
    validate_internal(&lowered)?;
    *values = next_values;
    Ok(Some(lowered))
}

/// Re-lower from the exact pre-stage allocator and compare both receipt and
/// final allocator. This is the staged-cursor equality gate.
pub(super) fn validate_from(
    arena: &ProofArenaPlan,
    before: &adapter::SemanticValueMap,
    after: &adapter::SemanticValueMap,
    supplied: &Option<LoweredMemoryBaseTrace>,
) -> Result<(), InvocationShapeError> {
    let mut exact_values = before.clone();
    let exact = lower_stage(arena, &mut exact_values)?;
    if &exact == supplied && &exact_values == after {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidMemoryBaseTraceBinding)
    }
}

pub(super) fn resolve_static_wrapper(
    id: StaticCudaWrapperId,
    target_sm: u32,
    lowered: &LoweredMemoryBaseTrace,
    step_ordinal: usize,
) -> Result<Option<StaticCudaWrapperAuthority>, InvocationShapeError> {
    validate_internal(lowered)?;
    let Some(linked) = lowered
        .contract
        .bind_static_build(target_sm)
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?
    else {
        return Ok(None);
    };
    project_static_wrapper(id, &linked, lowered, step_ordinal).map(Some)
}

fn project_static_wrapper(
    id: StaticCudaWrapperId,
    linked: &MemoryBaseTraceLinkedContract,
    lowered: &LoweredMemoryBaseTrace,
    step_ordinal: usize,
) -> Result<StaticCudaWrapperAuthority, InvocationShapeError> {
    linked
        .validate(&lowered.contract)
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    let lowered_step = lowered
        .steps
        .get(step_ordinal)
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    let contract_step = lowered
        .contract
        .steps()
        .get(step_ordinal)
        .filter(|step| {
            lowered_step.contract_ordinal as usize == step_ordinal
                && lowered_step.kind == step.kind()
        })
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)?;
    let launch = contract_step.launch();
    let launch = StaticCudaLaunchIdentity::new(
        contract_step.abi().kernel_symbol().as_bytes().to_vec(),
        crate::compiled_proof::LaunchGeometry {
            grid: launch.grid,
            block: launch.block,
            cluster: launch.cluster,
            dynamic_shared_bytes: launch.dynamic_shared_bytes,
            cooperative: launch.cooperative,
        },
    )
    .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    StaticCudaWrapperAuthority::new(
        id,
        linked.module_build_identity(),
        linked.target_sm(),
        contract_step.abi().entry_symbol().as_bytes().to_vec(),
        contract_step.abi_identity(),
        contract_step.effect_identity(),
        contract_step.identity(),
        linked.identity(),
        vec![launch],
        lowered_step
            .invocation()
            .contract_id()
            .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?,
        lowered_step.effect.id(),
    )
    .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)
}

fn lower_steps(
    contract: &MemoryBaseTraceContract,
    inventory: ExactInventory,
    values: &mut adapter::SemanticValueMap,
) -> Result<Vec<LoweredMemoryBaseTraceStep>, InvocationShapeError> {
    let mut contracts = contract.steps().iter().enumerate();
    let mut steps = Vec::with_capacity(contract.steps().len());

    let (ordinal, address) = contracts
        .next()
        .filter(|(_, step)| step.kind() == MemoryBaseTraceStepKind::Address)
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    steps.push(lower_write_step(
        ordinal,
        address,
        TracePartId::Main,
        vec![inventory.raw_address, inventory.address_counts],
        inventory.address_outputs,
        values,
    )?);

    for part in inventory.big_parts {
        let (value_ordinal, value) = contracts
            .next()
            .filter(|(_, step)| step.kind() == MemoryBaseTraceStepKind::BigValue)
            .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
        let value = lower_write_step(
            value_ordinal,
            value,
            part.part,
            value_read_values(value, &part.sources, &part.counts)?,
            part.outputs,
            values,
        )?;
        let (rc_ordinal, rc) = contracts
            .next()
            .filter(|(_, step)| step.kind() == MemoryBaseTraceStepKind::BigRc99)
            .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
        ensure_immutable(&inventory.rc99_lut, values)?;
        let rc_reads = rc_read_values(&value, rc, &inventory.rc99_lut, values)?;
        steps.push(value);
        steps.push(lower_atomic_step(
            rc_ordinal,
            rc,
            part.part,
            rc_reads,
            &inventory.rc99_counts,
            values,
        )?);
    }

    let part = inventory.small_part;
    let (value_ordinal, value) = contracts
        .next()
        .filter(|(_, step)| step.kind() == MemoryBaseTraceStepKind::SmallValue)
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    let value = lower_write_step(
        value_ordinal,
        value,
        part.part,
        value_read_values(value, &part.sources, &part.counts)?,
        part.outputs,
        values,
    )?;
    let (rc_ordinal, rc) = contracts
        .next()
        .filter(|(_, step)| step.kind() == MemoryBaseTraceStepKind::SmallRc99)
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    ensure_immutable(&inventory.rc99_lut, values)?;
    let rc_reads = rc_read_values(&value, rc, &inventory.rc99_lut, values)?;
    steps.push(value);
    steps.push(lower_atomic_step(
        rc_ordinal,
        rc,
        part.part,
        rc_reads,
        &inventory.rc99_counts,
        values,
    )?);
    if contracts.next().is_some() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority);
    }
    Ok(steps)
}

fn rc_read_values(
    value: &LoweredMemoryBaseTraceStep,
    contract: &MemoryBaseTraceStepContract,
    rc99_lut: &ExactValue,
    values: &adapter::SemanticValueMap,
) -> Result<Vec<MemoryBaseTraceValueBinding>, InvocationShapeError> {
    let limb_count = usize::try_from(contract.limb_or_pair_count())
        .map_err(|_| InvocationShapeError::SizeOverflow)?
        .checked_mul(2)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if value.writes.len() < limb_count || contract.reads().len() != limb_count + 1 {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority);
    }
    let mut reads = value.writes[..limb_count].to_vec();
    for (ordinal, read) in reads.iter_mut().enumerate() {
        read.binding = EffectBindingId(
            u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
        );
    }
    reads.push(bind_read(
        rc99_lut,
        &contract.reads()[limb_count],
        EffectBindingId(u32::try_from(limb_count).map_err(|_| InvocationShapeError::SizeOverflow)?),
        values,
    )?);
    Ok(reads)
}

fn value_read_values(
    contract: &MemoryBaseTraceStepContract,
    sources: &[ExactValue],
    counts: &ExactValue,
) -> Result<Vec<ExactValue>, InvocationShapeError> {
    contract
        .reads()
        .iter()
        .map(|effect| match effect.role {
            MemoryBaseTraceEffectRole::ValueSource => sources
                .get(effect.ordinal as usize)
                .cloned()
                .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding),
            MemoryBaseTraceEffectRole::ValueMultiplicity if effect.ordinal == 0 => {
                Ok(counts.clone())
            }
            _ => Err(InvocationShapeError::InvalidMemoryBaseTraceAuthority),
        })
        .collect()
}

fn lower_write_step(
    contract_ordinal: usize,
    contract: &MemoryBaseTraceStepContract,
    part: TracePartId,
    sources: Vec<ExactValue>,
    outputs: Vec<ExactValue>,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMemoryBaseTraceStep, InvocationShapeError> {
    if sources.len() != contract.reads().len() || outputs.len() != contract.writes().len() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    let reads = sources
        .iter()
        .zip(contract.reads())
        .enumerate()
        .map(|(index, (value, effect))| {
            bind_read(
                value,
                effect,
                EffectBindingId(
                    u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
                values,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for output in &outputs {
        if values.version(output.catalog.id).is_ok() {
            return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
        }
    }
    values.extend_ordered(outputs.iter().map(|output| output.catalog.id))?;
    let writes = outputs
        .iter()
        .zip(contract.writes())
        .enumerate()
        .map(|(index, (value, effect))| {
            bind_read(
                value,
                effect,
                EffectBindingId(
                    u32::try_from(reads.len() + index)
                        .map_err(|_| InvocationShapeError::SizeOverflow)?,
                ),
                values,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    semantic::compile(contract_ordinal, contract, part, reads, writes, None)
}

fn lower_atomic_step(
    contract_ordinal: usize,
    contract: &MemoryBaseTraceStepContract,
    part: TracePartId,
    reads: Vec<MemoryBaseTraceValueBinding>,
    destination: &ExactValue,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredMemoryBaseTraceStep, InvocationShapeError> {
    let geometry = contract
        .atomic()
        .ok_or(InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    let binding = EffectBindingId(
        u32::try_from(reads.len()).map_err(|_| InvocationShapeError::SizeOverflow)?,
    );
    let (source, output) = values.transition(destination.catalog.id)?;
    let elements = exact_elements(geometry.start_words, geometry.len_words)?;
    let atomic = MemoryBaseTraceTransition {
        arena: destination.arena,
        value: destination.catalog.id,
        elements,
        binding,
        source,
        destination: output,
        alias: semantic::atomic_alias(),
    };
    semantic::compile(
        contract_ordinal,
        contract,
        part,
        reads,
        Vec::new(),
        Some(atomic),
    )
}

fn bind_read(
    exact: &ExactValue,
    geometry: &stwo_backend_cuda::MemoryBaseTraceEffectAccess,
    binding: EffectBindingId,
    values: &adapter::SemanticValueMap,
) -> Result<MemoryBaseTraceValueBinding, InvocationShapeError> {
    let elements = exact_elements(geometry.start_words, geometry.len_words)?;
    if geometry
        .start_words
        .checked_add(geometry.len_words)
        .is_none_or(|end| end > exact.catalog.words)
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    Ok(MemoryBaseTraceValueBinding {
        arena: exact.arena,
        value: exact.catalog.id,
        elements,
        binding,
        version: values.version(exact.catalog.id)?,
    })
}

fn exact_elements(start: usize, len: usize) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(
        start,
        start
            .checked_add(len)
            .ok_or(InvocationShapeError::SizeOverflow)?,
    )
    .ok_or(InvocationShapeError::InvalidMemoryBaseTraceBinding)
}

fn ensure_immutable(
    exact: &ExactValue,
    values: &mut adapter::SemanticValueMap,
) -> Result<(), InvocationShapeError> {
    if values.version(exact.catalog.id).is_err() {
        values.extend_ordered([exact.catalog.id])?;
    }
    let lineage = values.versions_for(exact.catalog.id).collect::<Vec<_>>();
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    if lineage.len() != 1
        || !catalog_first.contains(&lineage[0])
        || transitions.contains(&lineage[0])
        || fixed.contains(&lineage[0])
    {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    Ok(())
}

fn validate_internal(lowered: &LoweredMemoryBaseTrace) -> Result<(), InvocationShapeError> {
    lowered
        .contract
        .validate()
        .map_err(|_| InvocationShapeError::InvalidMemoryBaseTraceAuthority)?;
    if lowered.steps.len() != lowered.contract.steps().len() {
        return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
    }
    for (ordinal, (supplied, contract)) in lowered
        .steps
        .iter()
        .zip(lowered.contract.steps())
        .enumerate()
    {
        if supplied.contract_ordinal as usize != ordinal
            || supplied.kind != contract.kind()
            || semantic::rebuild(supplied, contract)?
                != (supplied.invocation.clone(), supplied.effect.clone())
        {
            return Err(InvocationShapeError::InvalidMemoryBaseTraceBinding);
        }
    }
    Ok(())
}
