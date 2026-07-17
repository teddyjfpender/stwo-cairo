//! Exact semantic binding for the native direct Blake-G producer/feed body.
//!
//! The callable entry is a linked host FFI wrapper, not a recorded-witness AOT
//! kernel. Its five shared count slabs make this V1 contract monolithic: every
//! range is the full `0..padded_rows` domain and no row-shard authority exists.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    ArenaSlotId, BlakeGDirectCompositeAbi, BlakeGDirectCompositeContract, BlakeGDirectEffectAbi,
    BlakeGDirectRowDomain, BLAKE_G_DIRECT_COUNT_ORDER, BLAKE_G_DIRECT_LUT_ORDER,
};
use stwo_cairo_prover::witness::proof_shape::TracePartId;

use super::*;
use crate::arena_plan::{
    ArenaBinding, BlakeGWitnessContract, BufferPurpose, PlannedRecordedMultiplicityFeedGraph,
    PlannedWitnessComponent, ProofArenaPlan,
};
use crate::compiled_proof::{
    AtomicOperation, BoundValueRange, EffectAccess, EffectBindingId, EffectContract, ElementRange,
    InPlaceAliasAuthority, InPlaceAliasId, InPlaceAliasRequirement, InPlaceDiscipline, ValueRange,
    ValueVersion,
};
use crate::multiplicity_pipeline::blake_g_fused_feed_binding;
use crate::resident_runtime::producer_schedule::{WitnessProducer, WitnessProducerKind};

const INPUTS: usize = 6;
const TRACES: usize = 53;
const LUTS: usize = 4;
const COUNTS: usize = 5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BlakeGDirectRangeBinding {
    pub(super) value: ArenaCatalogRange,
    pub(super) binding: EffectBindingId,
    pub(super) version: ValueVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BlakeGDirectAtomicBinding {
    pub(super) value: ArenaCatalogRange,
    pub(super) binding: EffectBindingId,
    pub(super) source: ValueVersion,
    pub(super) destination: ValueVersion,
    pub(super) alias: InPlaceAliasAuthority,
}

/// Semantic arguments of the seven-argument host wrapper. The CUDA stream is
/// execution context, so it has no proof-value binding here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StaticBlakeGDirectInvocation {
    pub(super) inputs: [BlakeGDirectRangeBinding; INPUTS],
    pub(super) n_real_rows: u32,
    pub(super) padded_rows: u32,
    pub(super) traces: [BlakeGDirectRangeBinding; TRACES],
    pub(super) luts: [BlakeGDirectRangeBinding; LUTS],
    pub(super) counts: [BlakeGDirectAtomicBinding; COUNTS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoweredNativeBlakeGDirectContract {
    pub(super) authority: BlakeGDirectCompositeContract,
    pub(super) invocation: StaticBlakeGDirectInvocation,
    pub(super) effect: EffectContract,
}

#[derive(Clone, Debug)]
pub(super) struct PendingNativeBlakeGDirectContract {
    authority: BlakeGDirectCompositeContract,
    inputs: [ArenaCatalogRange; INPUTS],
    traces: [ArenaCatalogRange; TRACES],
    luts: [ArenaCatalogRange; LUTS],
    counts: [ArenaCatalogRange; COUNTS],
}

impl PendingNativeBlakeGDirectContract {
    pub(super) fn catalog_order(&self) -> impl Iterator<Item = ArenaCatalogValueId> + '_ {
        self.inputs
            .iter()
            .chain(&self.traces)
            .chain(&self.luts)
            .chain(&self.counts)
            .map(|range| range.value)
    }
}

pub(super) fn prepare(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    producer: WitnessProducer,
) -> Result<PendingNativeBlakeGDirectContract, InvocationShapeError> {
    let component = unique_direct_component(arena, producer)?;
    let selection = component
        .blake_g_contract
        .direct()
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    if selection.batch.component != "blake_g"
        || selection.part != TracePartId::Main
        || selection.input_ordinals != [0, 1, 2, 3, 4, 5]
        || selection.enabler_ordinal != 6
        || selection.instance_index != 0
        || selection.retired_lookup_words_per_row != component.program.n_lookup_words
        || arena.relation().blake_g_inputs.as_ref() != Some(selection)
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }

    let authority = BlakeGDirectCompositeContract::compile(
        &component.program,
        component.n_real_rows,
        component.requirements.row_count,
    )
    .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?;
    validate_backend_contract(&authority)?;

    if !component_geometry_is_exact(component, &authority) {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }

    let inputs = exact_component_ranges(
        catalog,
        component,
        &component.slots.input_columns,
        BufferPurpose::WitnessInput,
        authority.input_column_words(),
    )?;
    let traces = exact_component_ranges(
        catalog,
        component,
        &component.slots.output_columns,
        BufferPurpose::BaseTrace,
        authority.trace_column_words(),
    )?;
    let (feed_plan, lut_tables, count_destinations) = unique_direct_feed(arena)?;
    let canonical = blake_g_fused_feed_binding(feed_plan)
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    if canonical
        .lut_indices
        .map(|index| feed_plan.lut_families[index])
        != [
            "verify_bitwise_xor_8_state",
            "verify_bitwise_xor_4_state",
            "verify_bitwise_xor_7_state",
            "verify_bitwise_xor_9_state",
        ]
        || canonical
            .destination_indices
            .map(|index| feed_plan.destination_components[index])
            != [
                "verify_bitwise_xor_8",
                "verify_bitwise_xor_12",
                "verify_bitwise_xor_4",
                "verify_bitwise_xor_7",
                "verify_bitwise_xor_9",
            ]
        || feed_plan.row_count != authority.padded_rows()
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    let luts = exact_feed_ranges(
        catalog,
        lut_tables,
        BufferPurpose::WitnessFeedLut,
        &[None; LUTS],
        &[None; LUTS],
        &authority.lut_words(),
    )?;
    let count_components = canonical
        .destination_indices
        .map(|index| Some(feed_plan.destination_components[index]));
    let counts = exact_feed_ranges(
        catalog,
        count_destinations,
        BufferPurpose::FixedMultiplicity,
        &count_components,
        &[Some(TracePartId::Main); COUNTS],
        &authority.count_words(),
    )?;

    if catalog.values.iter().any(|value| {
        value.component == Some("blake_g")
            && value.part == Some(TracePartId::Main)
            && matches!(
                value.purpose,
                BufferPurpose::LookupInputs | BufferPurpose::SubcomponentInputs
            )
    }) {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    require_distinct(
        catalog,
        arena,
        inputs.iter().chain(&traces).chain(&luts).chain(&counts),
    )?;
    Ok(PendingNativeBlakeGDirectContract {
        authority,
        inputs,
        traces,
        luts,
        counts,
    })
}

pub(super) fn component_geometry_is_exact(
    component: &PlannedWitnessComponent,
    authority: &BlakeGDirectCompositeContract,
) -> bool {
    component.slots.input_columns.len() == INPUTS
        && component.slots.output_columns.len() == TRACES
        && component.requirements.input_column_words.len() == INPUTS + 1
        && component.requirements.output_column_words.as_slice() == authority.trace_column_words()
        && component
            .requirements
            .input_column_words
            .iter()
            .all(|&words| words == authority.padded_rows())
        && component.input_gather.as_ref().is_some_and(|gather| {
            gather.slots.consumer_input_columns == component.slots.input_columns
                && gather.requirements.consumer_input_column_words.as_slice()
                    == authority.input_column_words()
        })
        && component.input_seed.is_none()
        && component.input_compact.is_none()
        && component.input_casm.is_none()
}

pub(super) fn lower(
    pending: PendingNativeBlakeGDirectContract,
    values: &mut adapter::SemanticValueMap,
) -> Result<LoweredNativeBlakeGDirectContract, InvocationShapeError> {
    values.extend_ordered(pending.catalog_order())?;
    let mut next_binding = 0u32;
    let inputs = bind_ranges(pending.inputs, values, &mut next_binding)?;
    let traces = bind_ranges(pending.traces, values, &mut next_binding)?;
    let luts = bind_ranges(pending.luts, values, &mut next_binding)?;
    let counts = pending
        .counts
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let binding = take_binding(&mut next_binding)?;
            let (source, destination) = values.transition(value.value)?;
            Ok(BlakeGDirectAtomicBinding {
                value,
                binding,
                source,
                destination,
                alias: InPlaceAliasAuthority {
                    id: InPlaceAliasId(
                        u32::try_from(index).map_err(|_| InvocationShapeError::SizeOverflow)?,
                    ),
                    requirement: InPlaceAliasRequirement::Required,
                    discipline: InPlaceDiscipline::ElementWiseReadBeforeWrite,
                },
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    let invocation = StaticBlakeGDirectInvocation {
        inputs,
        n_real_rows: u32::try_from(pending.authority.n_real_rows())
            .map_err(|_| InvocationShapeError::SizeOverflow)?,
        padded_rows: u32::try_from(pending.authority.padded_rows())
            .map_err(|_| InvocationShapeError::SizeOverflow)?,
        traces,
        luts,
        counts,
    };
    let effect = exact_effect(&invocation)?;
    validate_lowered(&pending.authority, &invocation, &effect)?;
    Ok(LoweredNativeBlakeGDirectContract {
        authority: pending.authority,
        invocation,
        effect,
    })
}

pub(super) fn exact_effect(
    invocation: &StaticBlakeGDirectInvocation,
) -> Result<EffectContract, InvocationShapeError> {
    let mut accesses = Vec::with_capacity(INPUTS + TRACES + LUTS + COUNTS);
    accesses.extend(invocation.inputs.iter().map(|binding| EffectAccess::Read {
        source: bound_range(binding),
    }));
    accesses.extend(invocation.traces.iter().map(|binding| EffectAccess::Write {
        destination: bound_range(binding),
    }));
    accesses.extend(invocation.luts.iter().map(|binding| EffectAccess::Read {
        source: bound_range(binding),
    }));
    for binding in &invocation.counts {
        let elements = element_range(&binding.value)?;
        accesses.push(EffectAccess::Atomic {
            source: bound(binding.binding, binding.source, elements),
            destination: bound(binding.binding, binding.destination, elements),
            operation: AtomicOperation::AddU32,
            in_place: binding.alias,
        });
    }
    EffectContract::new(accesses, Vec::new())
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

pub(super) fn validate_lowered(
    authority: &BlakeGDirectCompositeContract,
    invocation: &StaticBlakeGDirectInvocation,
    effect: &EffectContract,
) -> Result<(), InvocationShapeError> {
    validate_backend_contract(authority)?;
    if invocation.n_real_rows as usize != authority.n_real_rows()
        || invocation.padded_rows as usize != authority.padded_rows()
        || !ranges_match(&invocation.inputs, authority.input_column_words())
        || !ranges_match(&invocation.traces, authority.trace_column_words())
        || !ranges_match(&invocation.luts, &authority.lut_words())
        || !atomic_ranges_match(&invocation.counts, &authority.count_words())
        || invocation
            .inputs
            .iter()
            .chain(&invocation.traces)
            .chain(&invocation.luts)
            .enumerate()
            .any(|(index, binding)| binding.binding.0 as usize != index)
        || invocation
            .counts
            .iter()
            .enumerate()
            .any(|(index, binding)| {
                binding.binding.0 as usize != INPUTS + TRACES + LUTS + index
                    || binding.source == binding.destination
                    || binding.alias.id.0 as usize != index
                    || binding.alias.requirement != InPlaceAliasRequirement::Required
                    || binding.alias.discipline != InPlaceDiscipline::ElementWiseReadBeforeWrite
            })
        || exact_effect(invocation)? != *effect
        || !effect
            .has_valid_identity()
            .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectAuthority)?
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    Ok(())
}

fn validate_backend_contract(
    authority: &BlakeGDirectCompositeContract,
) -> Result<(), InvocationShapeError> {
    if authority.abi() != BlakeGDirectCompositeAbi::FusedDirectV1
        || authority.abi().arguments().len() != 7
        || authority.effect()
            != BlakeGDirectEffectAbi::SixInputsFourLutsFiftyThreeTraceFiveCountsSynthesizedEnablerV1
        || authority.row_domain() != BlakeGDirectRowDomain::MonolithicFullPaddedRowsV1
        || authority.lut_order() != BLAKE_G_DIRECT_LUT_ORDER
        || authority.count_order() != BLAKE_G_DIRECT_COUNT_ORDER
        || authority.abi().entry_symbol() != "blake_g_write_trace_fused_direct_into_on"
        || authority.wrapper_launch().audited_internal_kernel_symbol()
            != "blake_g_write_trace_fused_scalar_kernel"
        || authority.wrapper_launch().block != [256, 1, 1]
        || authority.wrapper_launch().grid
            != [
                u32::try_from(authority.padded_rows())
                    .map_err(|_| InvocationShapeError::SizeOverflow)?
                    .div_ceil(256),
                1,
                1,
            ]
        || authority.wrapper_launch().dynamic_shared_bytes != 0
        || authority.wrapper_launch().cooperative
        || [
            authority.source_identity(),
            authority.abi_identity(),
            authority.effect_identity(),
            authority.launch_identity(),
            authority.identity(),
        ]
        .contains(&[0; 32])
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectAuthority);
    }
    Ok(())
}

fn unique_direct_component<'a>(
    arena: &'a ProofArenaPlan,
    producer: WitnessProducer,
) -> Result<&'a PlannedWitnessComponent, InvocationShapeError> {
    if producer.kind != WitnessProducerKind::BlakeGDirect
        || producer.component != "blake_g"
        || producer.part != Some(TracePartId::Main)
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    let mut matches = arena.witness().components.iter().filter(|component| {
        component.component == producer.component && Some(component.part) == producer.part
    });
    let component = matches
        .next()
        .filter(|component| matches!(component.blake_g_contract, BlakeGWitnessContract::Direct(_)))
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    Ok(component)
}

fn unique_direct_feed(
    arena: &ProofArenaPlan,
) -> Result<
    (
        &crate::multiplicity_pipeline::PlannedRecordedMultiplicityFeed,
        &[ArenaBinding; LUTS],
        &[ArenaBinding; COUNTS],
    ),
    InvocationShapeError,
> {
    let multiplicity = arena
        .multiplicity()
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    let mut matches = multiplicity.feeds.iter().filter(|feed| match feed {
        PlannedRecordedMultiplicityFeedGraph::Generic(feed) => feed.plan.producer == "blake_g",
        PlannedRecordedMultiplicityFeedGraph::BlakeGFused { plan, .. } => {
            plan.producer == "blake_g"
        }
    });
    let feed = matches
        .next()
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    match feed {
        PlannedRecordedMultiplicityFeedGraph::BlakeGFused {
            plan,
            lut_tables,
            multiplicity_destinations,
        } => Ok((plan, lut_tables, multiplicity_destinations)),
        PlannedRecordedMultiplicityFeedGraph::Generic(_) => {
            Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding)
        }
    }
}

fn exact_component_ranges<const N: usize>(
    catalog: &BaseProducerCatalog,
    component: &PlannedWitnessComponent,
    slots: &[ArenaSlotId],
    purpose: BufferPurpose,
    words: &[usize; N],
) -> Result<[ArenaCatalogRange; N], InvocationShapeError> {
    if slots.len() != N {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    slots
        .iter()
        .copied()
        .zip(words)
        .enumerate()
        .map(|(ordinal, (slot, &words))| {
            exact_role_range(
                catalog,
                slot,
                purpose,
                Some(component.component),
                Some(component.part),
                u32::try_from(ordinal).map_err(|_| InvocationShapeError::SizeOverflow)?,
                words,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

fn exact_feed_ranges<const N: usize>(
    catalog: &BaseProducerCatalog,
    bindings: &[ArenaBinding; N],
    purpose: BufferPurpose,
    components: &[Option<&'static str>; N],
    parts: &[Option<TracePartId>; N],
    words: &[usize; N],
) -> Result<[ArenaCatalogRange; N], InvocationShapeError> {
    let mut ranges = Vec::with_capacity(N);
    for index in 0..N {
        ranges.push(exact_arena_binding(
            catalog,
            bindings[index],
            purpose,
            components[index],
            parts[index],
            words[index],
        )?);
    }
    ranges
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

fn exact_arena_binding(
    catalog: &BaseProducerCatalog,
    binding: ArenaBinding,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    words: usize,
) -> Result<ArenaCatalogRange, InvocationShapeError> {
    let value = catalog.value(ArenaCatalogValueId(binding.logical.0))?;
    if value.logical != binding.logical
        || value.physical != binding.physical
        || value.words != binding.len_words
        || value.words != words
        || value.purpose != purpose
        || value.component != component
        || value.part != part
    {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    Ok(ArenaCatalogRange {
        value: value.id,
        value_words: 0..words,
    })
}

fn exact_role_range(
    catalog: &BaseProducerCatalog,
    slot: ArenaSlotId,
    purpose: BufferPurpose,
    component: Option<&'static str>,
    part: Option<TracePartId>,
    ordinal: u32,
    words: usize,
) -> Result<ArenaCatalogRange, InvocationShapeError> {
    let mut matches = catalog.values.iter().filter(|value| {
        value.physical == slot
            && value.purpose == purpose
            && value.component == component
            && value.part == part
            && value.ordinal == ordinal
            && value.words == words
    });
    let value = matches
        .next()
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
    if matches.next().is_some() {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    Ok(ArenaCatalogRange {
        value: value.id,
        value_words: 0..words,
    })
}

fn require_distinct<'a>(
    catalog: &BaseProducerCatalog,
    arena: &ProofArenaPlan,
    ranges: impl IntoIterator<Item = &'a ArenaCatalogRange>,
) -> Result<(), InvocationShapeError> {
    let mut logical = BTreeSet::new();
    let mut physical = BTreeSet::new();
    let mut intervals = Vec::new();
    for range in ranges {
        let value = catalog.value(range.value)?;
        if !logical.insert(range.value) || !physical.insert(value.physical) {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
        }
        let slot = arena
            .layout()
            .slot(value.physical)
            .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)?;
        let start = slot
            .offset_words
            .checked_add(range.value_words.start)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let end = slot
            .offset_words
            .checked_add(range.value_words.end)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        let slot_end = slot
            .offset_words
            .checked_add(slot.len_words)
            .ok_or(InvocationShapeError::SizeOverflow)?;
        if end > slot_end {
            return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
        }
        intervals.push((start, end));
    }
    if !intervals_are_disjoint(&mut intervals) {
        return Err(InvocationShapeError::InvalidNativeBlakeGDirectBinding);
    }
    Ok(())
}

pub(super) fn intervals_are_disjoint(intervals: &mut [(usize, usize)]) -> bool {
    intervals.sort_unstable();
    intervals.iter().all(|&(start, end)| start < end)
        && intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0)
}

fn bind_ranges<const N: usize>(
    ranges: [ArenaCatalogRange; N],
    values: &adapter::SemanticValueMap,
    next_binding: &mut u32,
) -> Result<[BlakeGDirectRangeBinding; N], InvocationShapeError> {
    ranges
        .into_iter()
        .map(|value| {
            Ok(BlakeGDirectRangeBinding {
                binding: take_binding(next_binding)?,
                version: values.version(value.value)?,
                value,
            })
        })
        .collect::<Result<Vec<_>, InvocationShapeError>>()?
        .try_into()
        .map_err(|_| InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

fn take_binding(next: &mut u32) -> Result<EffectBindingId, InvocationShapeError> {
    let binding = EffectBindingId(*next);
    *next = next
        .checked_add(1)
        .ok_or(InvocationShapeError::SizeOverflow)?;
    Ok(binding)
}

fn bound_range(binding: &BlakeGDirectRangeBinding) -> BoundValueRange {
    bound(
        binding.binding,
        binding.version,
        ElementRange {
            start: binding.value.value_words.start,
            end: binding.value.value_words.end,
        },
    )
}

const fn bound(
    binding: EffectBindingId,
    version: ValueVersion,
    elements: ElementRange,
) -> BoundValueRange {
    BoundValueRange {
        binding,
        value: ValueRange { version, elements },
    }
}

fn element_range(value: &ArenaCatalogRange) -> Result<ElementRange, InvocationShapeError> {
    ElementRange::new(value.value_words.start, value.value_words.end)
        .ok_or(InvocationShapeError::InvalidNativeBlakeGDirectBinding)
}

fn ranges_match<const N: usize>(
    ranges: &[BlakeGDirectRangeBinding; N],
    words: &[usize; N],
) -> bool {
    ranges
        .iter()
        .zip(words)
        .all(|(binding, &words)| exact_full_range(&binding.value, words))
}

fn atomic_ranges_match<const N: usize>(
    ranges: &[BlakeGDirectAtomicBinding; N],
    words: &[usize; N],
) -> bool {
    ranges
        .iter()
        .zip(words)
        .all(|(binding, &words)| exact_full_range(&binding.value, words))
}

fn exact_full_range(range: &ArenaCatalogRange, words: usize) -> bool {
    range.value_words.start == 0 && range.value_words.end == words
}
