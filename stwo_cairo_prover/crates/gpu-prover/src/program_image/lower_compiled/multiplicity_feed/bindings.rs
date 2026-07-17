//! Exact catalog roles, writer ownership and relocation bindings.

use super::*;
use crate::arena_plan::{BufferPurpose, PlannedRecordedMultiplicityFeedGraph};
use crate::compiled_proof::{EffectAccess, ValueRange};
use crate::resident_runtime::producer_schedule::WitnessProducerKind;

const PUBLIC_SEED_STATES: [&str; 3] = [
    "memory_address_to_id_state",
    "memory_id_to_big_state",
    "memory_id_to_big_state#small",
];
const PUBLIC_SEED_DESTINATIONS: [&str; 3] = [
    "memory_address_to_id",
    "memory_id_to_big",
    "memory_id_to_big#small",
];

pub(super) struct ExactFeed<'a> {
    pub(super) source: &'a BaseCatalogValue,
    pub(super) luts: Vec<&'a BaseCatalogValue>,
    pub(super) destinations: Vec<ExactDestination<'a>>,
    pub(super) relocations: MultiplicityFeedRelocations,
}

pub(super) struct ExactDestination<'a> {
    pub(super) name: &'static str,
    pub(super) value: &'a BaseCatalogValue,
}

pub(super) fn require_recorded_membership(
    arena: &ProofArenaPlan,
    supplied: &PlannedGenericRecordedMultiplicityFeedGraph,
) -> Result<(), InvocationShapeError> {
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::MultiplicityNeedsSemanticVersions)?;
    let mut matches = multiplicity.feeds.iter().filter_map(|feed| match feed {
        PlannedRecordedMultiplicityFeedGraph::Generic(candidate)
            if same_feed(candidate, supplied) =>
        {
            Some(())
        }
        PlannedRecordedMultiplicityFeedGraph::Generic(_)
        | PlannedRecordedMultiplicityFeedGraph::BlakeGFused { .. } => None,
    });
    if matches.next().is_some() && matches.next().is_none() {
        Ok(())
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

pub(super) fn require_public_seed_membership(
    arena: &ProofArenaPlan,
    supplied: &PlannedGenericRecordedMultiplicityFeedGraph,
) -> Result<(), InvocationShapeError> {
    let exact = arena
        .multiplicity()
        .and_then(|multiplicity| multiplicity.public_memory_seed.as_ref())
        .filter(|planned| same_feed(planned, supplied))
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if exact.plan.producer != "__public_memory__"
        || exact.plan.lut_families.len() != 0
        || !exact.slots.lut_tables.is_empty()
        || exact.plan.destination_components.as_slice() != PUBLIC_SEED_STATES
        || exact.slots.multiplicity_destinations.len() != 3
        || exact.plan.requirements.multiplicity_words.len() != 3
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(())
}

pub(super) fn ordered_lut_words(
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    canonical: &BTreeMap<&'static str, Vec<u32>>,
) -> Result<Vec<Vec<u32>>, InvocationShapeError> {
    if plan.plan.lut_families.len() != plan.plan.requirements.lut_words.len()
        || plan.plan.lut_families.len() != plan.slots.lut_tables.len()
        || plan
            .plan
            .lut_families
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != plan.plan.lut_families.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    plan.plan
        .lut_families
        .iter()
        .zip(&plan.plan.requirements.lut_words)
        .map(|(&family, &words)| {
            canonical
                .get(family)
                .filter(|lut| lut.len() == words)
                .cloned()
                .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
        })
        .collect()
}

pub(super) fn exact<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    owner: MultiplicityFeedOwner,
    contract: &WitnessFeedContract,
) -> Result<ExactFeed<'a>, InvocationShapeError> {
    if contract.requirements() != &plan.plan.requirements
        || contract
            .requirements()
            .arena_slot_requirements(&plan.slots)
            .is_err()
    {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    let (component, part) = match owner {
        MultiplicityFeedOwner::Recorded { component, part } => (Some(component), Some(part)),
        MultiplicityFeedOwner::PublicMemorySeed => (None, None),
    };
    let source_purpose = match owner {
        MultiplicityFeedOwner::Recorded { .. } => BufferPurpose::SubcomponentInputs,
        MultiplicityFeedOwner::PublicMemorySeed => BufferPurpose::PublicMemoryMultiplicitySeed,
    };
    let source = exact_binding_value(
        arena,
        catalog,
        plan.source,
        source_purpose,
        component,
        part,
        Some(0),
    )?;
    if source.words != contract.effect_geometry().source.read_len_words {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }

    let descriptor = exact_physical_value(
        arena,
        catalog,
        plan.slots.descriptors,
        BufferPurpose::WitnessFeedDescriptors,
        component,
        part,
        contract.requirements().descriptor_words,
    )?;
    let lut_pointers = exact_physical_value(
        arena,
        catalog,
        plan.slots.lut_pointers,
        BufferPurpose::WitnessFeedLutPointers,
        component,
        part,
        contract.requirements().lut_pointer_words,
    )?;
    let multiplicity_pointers = exact_physical_value(
        arena,
        catalog,
        plan.slots.multiplicity_pointers,
        BufferPurpose::WitnessFeedMultiplicityPointers,
        component,
        part,
        contract.requirements().multiplicity_pointer_words,
    )?;
    if descriptor.ordinal != lut_pointers.ordinal
        || descriptor.ordinal != multiplicity_pointers.ordinal
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let relocations = MultiplicityFeedRelocations {
        descriptor_workspace: arena_binding(arena, descriptor)?,
        lut_pointers: arena_binding(arena, lut_pointers)?,
        multiplicity_pointers: arena_binding(arena, multiplicity_pointers)?,
    };

    let luts = plan
        .slots
        .lut_tables
        .iter()
        .zip(&plan.plan.requirements.lut_words)
        .map(|(&physical, &words)| {
            exact_physical_value(
                arena,
                catalog,
                physical,
                BufferPurpose::WitnessFeedLut,
                None,
                None,
                words,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if luts
        .iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>()
        .len()
        != luts.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }

    let names: &[&'static str] = match owner {
        MultiplicityFeedOwner::Recorded { .. } => &plan.plan.destination_components,
        MultiplicityFeedOwner::PublicMemorySeed => &PUBLIC_SEED_DESTINATIONS,
    };
    let destinations = exact_destinations(arena, catalog, plan, names)?;
    let occupied = std::iter::once(source.physical)
        .chain(luts.iter().map(|value| value.physical))
        .chain(
            destinations
                .iter()
                .map(|destination| destination.value.physical),
        )
        .chain([
            relocations.descriptor_workspace.physical,
            relocations.lut_pointers.physical,
            relocations.multiplicity_pointers.physical,
        ])
        .collect::<BTreeSet<_>>();
    let expected_occupied = 1 + luts.len() + destinations.len() + 3;
    if occupied.len() != expected_occupied {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(ExactFeed {
        source,
        luts,
        destinations,
        relocations,
    })
}

pub(super) fn bind_source(
    arena: &ProofArenaPlan,
    exact: &ExactFeed<'_>,
    owner: MultiplicityFeedOwner,
    writer: Option<&LoweredRecordedWitnessProducer>,
    values: &mut adapter::SemanticValueMap,
    binding: EffectBindingId,
) -> Result<MultiplicityFeedValueBinding, InvocationShapeError> {
    match (owner, writer) {
        (MultiplicityFeedOwner::PublicMemorySeed, None) => {
            values.extend_ordered([exact.source.id])?;
        }
        (MultiplicityFeedOwner::Recorded { .. }, Some(writer)) => {
            require_writer_output(writer, exact.source, values)?;
        }
        _ => return Err(InvocationShapeError::InvalidScheduledProducerBinding),
    }
    let input = bind_immutable(arena, exact.source, 0, exact.source.words, binding, values)?;
    Ok(input)
}

pub(super) fn bind_immutable(
    arena: &ProofArenaPlan,
    value: &BaseCatalogValue,
    start: usize,
    len: usize,
    binding: EffectBindingId,
    values: &adapter::SemanticValueMap,
) -> Result<MultiplicityFeedValueBinding, InvocationShapeError> {
    let elements = range(start, len, value.words)?;
    let version = values.version(value.id)?;
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    if !catalog_first.contains(&version)
        || transitions.contains(&version)
        || fixed.contains(&version)
        || values.versions_for(value.id).collect::<Vec<_>>() != [version]
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let arena = arena_binding(arena, value)?;
    Ok(MultiplicityFeedValueBinding {
        arena,
        value: value.id,
        elements,
        binding,
        version,
    })
}

pub(super) fn require_immutable_luts(
    values: &adapter::SemanticValueMap,
    luts: &[MultiplicityFeedLutBinding],
) -> Result<(), InvocationShapeError> {
    let unique = luts
        .iter()
        .map(|lut| (lut.family, lut.input.value, lut.input.version))
        .collect::<BTreeSet<_>>();
    if unique.len() != luts.len()
        || luts.iter().any(|lut| {
            lut.content_identity == [0; 32]
                || values.version(lut.input.value) != Ok(lut.input.version)
        })
    {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    } else {
        Ok(())
    }
}

fn require_writer_output(
    writer: &LoweredRecordedWitnessProducer,
    source: &BaseCatalogValue,
    values: &adapter::SemanticValueMap,
) -> Result<(), InvocationShapeError> {
    if writer.producer.kind != WitnessProducerKind::Recorded
        || writer.producer.component != source.component.unwrap_or_default()
        || writer.producer.part != source.part
        || writer
            .produced
            .iter()
            .filter(|&&id| id == source.id)
            .count()
            != 1
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let current = values.version(source.id)?;
    let elements =
        ElementRange::new(0, source.words).ok_or(InvocationShapeError::InvalidStructuredAbi)?;
    let writes = writer
        .effect
        .accesses()
        .iter()
        .filter(|access| {
            matches!(
                access,
                EffectAccess::Write { destination }
                    if destination.value
                        == ValueRange {
                            version: current,
                            elements,
                        }
            )
        })
        .count();
    if writes != 1 {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    if !catalog_first.contains(&current)
        || transitions.contains(&current)
        || fixed.contains(&current)
        || values.versions_for(source.id).collect::<Vec<_>>() != [current]
    {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    } else {
        Ok(())
    }
}

fn exact_destinations<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    plan: &PlannedGenericRecordedMultiplicityFeedGraph,
    names: &[&'static str],
) -> Result<Vec<ExactDestination<'a>>, InvocationShapeError> {
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::MultiplicityNeedsSemanticVersions)?;
    if names.len() != plan.slots.multiplicity_destinations.len()
        || names.len() != plan.plan.requirements.multiplicity_words.len()
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    names
        .iter()
        .zip(&plan.slots.multiplicity_destinations)
        .zip(&plan.plan.requirements.multiplicity_words)
        .map(|((&name, &physical), &words)| {
            let mut matches = multiplicity
                .multiplicities
                .iter()
                .filter(|(candidate, binding)| {
                    *candidate == name && binding.physical == physical && binding.len_words == words
                });
            let (_, binding) = matches
                .next()
                .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
            if matches.next().is_some() {
                return Err(InvocationShapeError::InvalidScheduledProducerBinding);
            }
            let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
            validate_destination_role(value, name)?;
            Ok(ExactDestination { name, value })
        })
        .collect()
}

fn validate_destination_role(
    value: &BaseCatalogValue,
    name: &'static str,
) -> Result<(), InvocationShapeError> {
    let valid = match name {
        "memory_address_to_id" => {
            value.purpose == BufferPurpose::RuntimeMultiplicity
                && value.component == Some("memory_address_to_id")
                && value.part == Some(TracePartId::Main)
        }
        "memory_id_to_big" | "memory_id_to_big#small" => {
            value.purpose == BufferPurpose::RuntimeMultiplicity
                && value.component == Some("memory_id_to_big")
                && value.part.is_none()
        }
        _ => {
            value.purpose == BufferPurpose::FixedMultiplicity
                && value.component == Some(name)
                && value.part == Some(TracePartId::Main)
                && value.ordinal == 0
        }
    };
    valid
        .then_some(())
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)
}

fn exact_binding_value<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    binding: ArenaBinding,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    ordinal: Option<u32>,
) -> Result<&'a BaseCatalogValue, InvocationShapeError> {
    let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
    if value.logical != binding.logical
        || value.physical != binding.physical
        || value.words != binding.len_words
        || value.purpose != purpose
        || value.component != component
        || value.part != part
        || ordinal.is_some_and(|ordinal| value.ordinal != ordinal)
        || arena.binding(binding.logical) != Some(binding)
    {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    Ok(value)
}

fn exact_physical_value<'a>(
    arena: &ProofArenaPlan,
    catalog: &'a BaseProducerCatalog,
    physical: stwo_backend_cuda::ArenaSlotId,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    words: usize,
) -> Result<&'a BaseCatalogValue, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.physical == physical
            && value.purpose == purpose
            && value.component == component
            && value.part == part
            && value.words == words
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidScheduledProducerBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidScheduledProducerBinding);
    }
    arena_binding(arena, value)?;
    Ok(value)
}

fn arena_binding(
    arena: &ProofArenaPlan,
    value: &BaseCatalogValue,
) -> Result<ArenaBinding, InvocationShapeError> {
    let binding = ArenaBinding {
        logical: value.logical,
        physical: value.physical,
        len_words: value.words,
    };
    if arena.binding(value.logical) == Some(binding) {
        Ok(binding)
    } else {
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    }
}

fn range(start: usize, len: usize, available: usize) -> Result<ElementRange, InvocationShapeError> {
    let end = start
        .checked_add(len)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    if end > available {
        return Err(InvocationShapeError::InvalidStructuredAbi);
    }
    ElementRange::new(start, end).ok_or(InvocationShapeError::InvalidStructuredAbi)
}

fn same_feed(
    left: &PlannedGenericRecordedMultiplicityFeedGraph,
    right: &PlannedGenericRecordedMultiplicityFeedGraph,
) -> bool {
    left.plan == right.plan && left.source == right.source && left.slots == right.slots
}
