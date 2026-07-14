//! Pure Graph-A multiplicity/feed/materializer planning.

use std::collections::BTreeMap;

use stwo_backend_cuda::{WitnessFeedWorkspaceRequirements, WITNESS_FEED_DESCRIPTOR_WORDS};
use stwo_cairo_prover::witness::device_feed::{
    build_feed_descriptors_sized, CountRelation, COUNT_RELATIONS,
};
use stwo_cairo_prover::witness::jit_prove_backend::{
    all_lane_sub_feed_layouts, RecordedSubFeedLayout,
};
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};

use crate::fixed_table_materializer::{
    compile_cairo_fixed_table_materializations, CompiledFixedTableMaterialization,
    FixedTableMaterializerError,
};
use crate::plan::ProofPlan;
use crate::schedule::{WitnessWriterKind, WitnessWriterSpec};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MultiplicityFeedBlockerKind {
    RuntimeSizedRelation,
    DependentTuple,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiplicityFeedBlocker {
    pub producer: &'static str,
    pub state_param: &'static str,
    pub kind: MultiplicityFeedBlockerKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedMultiplicityCoverageGap {
    pub fixed_component: &'static str,
    pub producer: &'static str,
    pub expected_instances: u32,
    pub prepared_instances: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedCanonicalLut {
    pub state_param: &'static str,
    pub words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedRecordedMultiplicityFeed {
    pub producer: &'static str,
    pub row_count: usize,
    pub sub_words_per_row: usize,
    pub descriptors: Vec<u32>,
    pub lut_families: Vec<&'static str>,
    pub destination_components: Vec<&'static str>,
    pub requirements: WitnessFeedWorkspaceRequirements,
}

/// Exact pointer order consumed by the native blake_g producer/feed fusion.
/// Construction fail-closes unless the transformer-emitted 16-descriptor
/// source topology is still byte-for-byte canonical.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlakeGFusedFeedBinding {
    /// xor8, xor4, xor7, xor9.
    pub lut_indices: [usize; 4],
    /// xor8, xor12, xor4, xor7, xor9.
    pub destination_indices: [usize; 5],
}

pub fn blake_g_fused_feed_binding(
    feed: &PlannedRecordedMultiplicityFeed,
) -> Option<BlakeGFusedFeedBinding> {
    let layout = all_lane_sub_feed_layouts()
        .into_iter()
        .find(|layout| layout.component == "blake_g")?;
    let (descriptors, lut_families, destination_states, multiplicity_words) =
        build_feed_descriptors_sized(layout.entries, COUNT_RELATIONS, &|_| None);
    let destination_components = destination_states
        .iter()
        .map(|state| destination_for_state(state).ok())
        .collect::<Option<Vec<_>>>()?;
    if feed.producer != "blake_g"
        || feed.sub_words_per_row != 48
        || feed.descriptors != descriptors
        || feed.lut_families != lut_families
        || feed.destination_components != destination_components
        || feed.requirements.row_count != feed.row_count
        || feed.requirements.sub_words_per_row != feed.sub_words_per_row
        || feed.requirements.descriptor_words != descriptors.len()
        || feed.requirements.descriptor_count != 16
        || feed.requirements.lut_pointer_words != pointer_words(lut_families.len()).ok()?
        || feed.requirements.multiplicity_pointer_words
            != pointer_words(destination_components.len()).ok()?
        || feed.requirements.source_words != feed.row_count.checked_mul(48)?
        || feed.requirements.lut_words != [1 << 16, 1 << 8, 1 << 14, 1 << 18]
        || feed.requirements.multiplicity_words != multiplicity_words
    {
        return None;
    }
    let unique_index = |values: &[&'static str], needle: &'static str| {
        let mut matches = values
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == needle).then_some(index));
        let index = matches.next()?;
        matches.next().is_none().then_some(index)
    };
    Some(BlakeGFusedFeedBinding {
        lut_indices: [
            unique_index(&feed.lut_families, "verify_bitwise_xor_8_state")?,
            unique_index(&feed.lut_families, "verify_bitwise_xor_4_state")?,
            unique_index(&feed.lut_families, "verify_bitwise_xor_7_state")?,
            unique_index(&feed.lut_families, "verify_bitwise_xor_9_state")?,
        ],
        destination_indices: [
            unique_index(&feed.destination_components, "verify_bitwise_xor_8")?,
            unique_index(&feed.destination_components, "verify_bitwise_xor_12")?,
            unique_index(&feed.destination_components, "verify_bitwise_xor_4")?,
            unique_index(&feed.destination_components, "verify_bitwise_xor_7")?,
            unique_index(&feed.destination_components, "verify_bitwise_xor_9")?,
        ],
    })
}

pub fn plan_public_memory_multiplicity_seed(
    row_count: usize,
    address_words: usize,
    big_words: usize,
    small_words: usize,
) -> Result<PlannedRecordedMultiplicityFeed, GraphAMultiplicityPlanError> {
    if row_count == 0 {
        return Err(GraphAMultiplicityPlanError::SizeOverflow);
    }
    const LAYOUT: &[(&str, usize, &str, u32, usize, usize)] = &[
        ("address", 0, "memory_address_to_id_state", 0, 0, 1),
        ("id", 0, "memory_id_to_big_state", 0, 1, 1),
    ];
    let (descriptors, lut_families, destination_components, multiplicity_words) =
        build_feed_descriptors_sized(LAYOUT, COUNT_RELATIONS, &|state| match state {
            "memory_address_to_id_state" => Some((address_words, 0)),
            "memory_id_to_big_state" => Some((big_words, small_words)),
            _ => None,
        });
    let source_words = row_count
        .checked_mul(2)
        .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
    Ok(PlannedRecordedMultiplicityFeed {
        producer: "__public_memory__",
        row_count,
        sub_words_per_row: 2,
        requirements: WitnessFeedWorkspaceRequirements {
            row_count,
            sub_words_per_row: 2,
            source_words,
            descriptor_words: descriptors.len(),
            descriptor_count: descriptors.len() / WITNESS_FEED_DESCRIPTOR_WORDS,
            lut_pointer_words: pointer_words(0)?,
            multiplicity_pointer_words: pointer_words(destination_components.len())?,
            lut_words: Vec::new(),
            multiplicity_words,
        },
        descriptors,
        lut_families,
        destination_components,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedFixedMultiplicity {
    pub component: &'static str,
    pub row_count: usize,
    pub columns: usize,
    pub slab_words: usize,
    pub materializer: CompiledFixedTableMaterialization,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedRuntimeMultiplicity {
    pub destination: &'static str,
    pub words: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedMemoryTracePart {
    pub part: TracePartId,
    pub row_count: usize,
    pub source_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedMemoryBaseTraces {
    pub address_rows: usize,
    pub address_count_words: usize,
    pub big_count_words: usize,
    pub small_count_words: usize,
    pub rc99_lut_words: usize,
    pub rc99_count_words: usize,
    pub big_parts: Vec<PlannedMemoryTracePart>,
    pub small_part: PlannedMemoryTracePart,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphAMultiplicityPlan {
    pub fixed: Vec<PlannedFixedMultiplicity>,
    pub runtime: Vec<PlannedRuntimeMultiplicity>,
    pub memory_traces: Option<PlannedMemoryBaseTraces>,
    pub feeds: Vec<PlannedRecordedMultiplicityFeed>,
    pub luts: Vec<PlannedCanonicalLut>,
    pub coverage_gaps: Vec<FixedMultiplicityCoverageGap>,
    pub blockers: Vec<MultiplicityFeedBlocker>,
    pub topology_hash: u64,
}

impl GraphAMultiplicityPlan {
    pub fn coverage_complete(&self) -> bool {
        self.coverage_gaps.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphAMultiplicityPlanError {
    FixedTable(FixedTableMaterializerError),
    MissingProofComponent(&'static str),
    InvalidRows(&'static str),
    MissingRecording(&'static str),
    DuplicateRecording(&'static str),
    DuplicateFixedComponent(&'static str),
    UnknownFixedDestination(&'static str),
    FixedMultiplicityMismatch {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    UnexpectedPreparedProducer {
        fixed_component: &'static str,
        producer: &'static str,
    },
    SizeOverflow,
    InvalidMemoryGeometry(&'static str),
}

impl core::fmt::Display for GraphAMultiplicityPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Graph-A multiplicity plan rejected: {self:?}")
    }
}

impl std::error::Error for GraphAMultiplicityPlanError {}

impl From<FixedTableMaterializerError> for GraphAMultiplicityPlanError {
    fn from(value: FixedTableMaterializerError) -> Self {
        Self::FixedTable(value)
    }
}

pub fn plan_graph_a_multiplicities(
    proof: &ProofPlan,
) -> Result<GraphAMultiplicityPlan, GraphAMultiplicityPlanError> {
    let compiled = compile_cairo_fixed_table_materializations()?;
    let mut fixed = Vec::with_capacity(compiled.len());
    let mut fixed_by_component = BTreeMap::new();
    for materializer in compiled {
        let component = proof_component(proof, materializer.component())?;
        if !component.runtime.is_present() {
            continue;
        }
        let row_count = one_main_row_count(component)?;
        if row_count != materializer.config().row_count {
            return Err(GraphAMultiplicityPlanError::InvalidRows(
                materializer.component(),
            ));
        }
        let columns = materializer.config().multiplicity_column_count;
        let slab_words = row_count
            .checked_mul(columns)
            .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
        let index = fixed.len();
        if fixed_by_component
            .insert(materializer.component(), index)
            .is_some()
        {
            return Err(GraphAMultiplicityPlanError::DuplicateFixedComponent(
                materializer.component(),
            ));
        }
        fixed.push(PlannedFixedMultiplicity {
            component: materializer.component(),
            row_count,
            columns,
            slab_words,
            materializer,
        });
    }

    let memory_traces = plan_memory_base_traces(proof)?;
    let runtime = memory_traces
        .as_ref()
        .map(|memory| {
            vec![
                PlannedRuntimeMultiplicity {
                    destination: "memory_address_to_id",
                    words: memory.address_count_words,
                },
                PlannedRuntimeMultiplicity {
                    destination: "memory_id_to_big",
                    words: memory.big_count_words,
                },
                PlannedRuntimeMultiplicity {
                    destination: "memory_id_to_big#small",
                    words: memory.small_count_words,
                },
            ]
        })
        .unwrap_or_default();

    let mut layouts = BTreeMap::<&'static str, RecordedSubFeedLayout>::new();
    for layout in all_lane_sub_feed_layouts() {
        if layouts.insert(layout.component, layout).is_some() {
            return Err(GraphAMultiplicityPlanError::DuplicateRecording(
                layout.component,
            ));
        }
    }

    let mut feeds = Vec::new();
    let mut blockers = Vec::new();
    let mut actual = BTreeMap::<(&'static str, &'static str), u32>::new();
    for component in proof.components.iter().filter(|component| {
        component.runtime.is_present()
            && component.node.facts.witness_writer.kind == WitnessWriterKind::RecordedAot
    }) {
        let layout =
            layouts
                .get(component.node.id)
                .ok_or(GraphAMultiplicityPlanError::MissingRecording(
                    component.node.id,
                ))?;
        for &(_, _, state_param, ..) in layout.entries {
            let kind = match count_relation(state_param) {
                Some(relation)
                    if relation.table_size == 0
                        && runtime_relation_sizes(memory_traces.as_ref(), state_param)
                            .is_none() =>
                {
                    Some(MultiplicityFeedBlockerKind::RuntimeSizedRelation)
                }
                Some(_) => None,
                None if state_param.starts_with("verify_bitwise_xor_") => {
                    Some(MultiplicityFeedBlockerKind::DependentTuple)
                }
                None => None,
            };
            if let Some(kind) = kind {
                let blocker = MultiplicityFeedBlocker {
                    producer: component.node.id,
                    state_param,
                    kind,
                };
                if !blockers.contains(&blocker) {
                    blockers.push(blocker);
                }
            }
        }

        let (descriptors, lut_families, destination_states, multiplicity_words) =
            build_feed_descriptors_sized(layout.entries, COUNT_RELATIONS, &|state| {
                runtime_relation_sizes(memory_traces.as_ref(), state)
            });
        let destination_components = destination_states
            .iter()
            .map(|state| destination_for_state(state))
            .collect::<Result<Vec<_>, _>>()?;
        for &(_, _, state_param, ..) in layout.entries {
            let Some(relation) = count_relation(state_param) else {
                continue;
            };
            if relation.table_size == 0 {
                continue;
            }
            let destination = fixed_component_for_state(state_param)?;
            let entries = actual.entry((destination, component.node.id)).or_default();
            *entries = entries
                .checked_add(1)
                .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
        }
        if descriptors.is_empty() {
            continue;
        }
        let row_count = one_main_row_count(component)?;
        let sub_words_per_row =
            component
                .node
                .facts
                .sub_words
                .ok_or(GraphAMultiplicityPlanError::MissingRecording(
                    component.node.id,
                ))? as usize;
        let lut_words = lut_families
            .iter()
            .map(|state| {
                count_relation(state)
                    .map(|relation| relation.table_size)
                    .ok_or(GraphAMultiplicityPlanError::UnknownFixedDestination(state))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (&destination, &words) in destination_components.iter().zip(&multiplicity_words) {
            let expected_words = fixed_by_component
                .get(destination)
                .map(|&index| fixed[index].slab_words)
                .or_else(|| {
                    runtime
                        .iter()
                        .find(|entry| entry.destination == destination)
                        .map(|entry| entry.words)
                })
                .ok_or(GraphAMultiplicityPlanError::UnknownFixedDestination(
                    destination,
                ))?;
            if words != expected_words {
                return Err(GraphAMultiplicityPlanError::FixedMultiplicityMismatch {
                    component: destination,
                    expected: expected_words,
                    actual: words,
                });
            }
        }
        let source_words = row_count
            .checked_mul(sub_words_per_row)
            .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
        feeds.push(PlannedRecordedMultiplicityFeed {
            producer: component.node.id,
            row_count,
            sub_words_per_row,
            requirements: WitnessFeedWorkspaceRequirements {
                row_count,
                sub_words_per_row,
                source_words,
                descriptor_words: descriptors.len(),
                descriptor_count: descriptors.len() / WITNESS_FEED_DESCRIPTOR_WORDS,
                lut_pointer_words: pointer_words(lut_words.len())?,
                multiplicity_pointer_words: pointer_words(multiplicity_words.len())?,
                lut_words,
                multiplicity_words,
            },
            descriptors,
            lut_families,
            destination_components,
        });
    }

    // ec_op_builtin is a native arena producer: its precompiled kernel writes
    // both range-check values directly into range_check_8's cleared slab.  It
    // deliberately has no row-major sub buffer for the generic feed kernel.
    if proof.components.iter().any(|component| {
        component.node.id == "ec_op_builtin"
            && component.runtime.is_present()
            && component.node.facts.witness_writer.kind == WitnessWriterKind::NativeCuda
            && component.node.facts.witness_writer.is_capture_safe()
    }) {
        actual.insert(("range_check_8", "ec_op_builtin"), 2);
    }
    let memory_native_coverage = memory_traces.is_some()
        && native_memory_rc99_coverage(
            proof_component(proof, "memory_id_to_big")?
                .node
                .facts
                .witness_writer,
        );
    if memory_native_coverage {
        actual.insert(("range_check_9_9", "memory_id_to_big"), 18);
    }

    let mut expected = BTreeMap::new();
    for table in &fixed {
        let component = proof_component(proof, table.component)?;
        let relation = count_relation_for_fixed(table.component);
        if let Some(relation) = relation {
            if relation.n_relations != table.columns {
                return Err(GraphAMultiplicityPlanError::FixedMultiplicityMismatch {
                    component: table.component,
                    expected: table.columns,
                    actual: relation.n_relations,
                });
            }
        }
        for capacity in component.node.capacity_inputs {
            let producer = proof_component(proof, capacity.from)?;
            if !producer.runtime.is_present() {
                continue;
            }
            expected.insert((table.component, capacity.from), capacity.n_instances);
        }
    }
    if memory_traces.is_some() {
        expected.insert(("range_check_9_9", "memory_id_to_big"), 18);
    }
    let coverage_gaps = validate_coverage(&expected, &actual)?;

    let mut luts = BTreeMap::new();
    for feed in &feeds {
        for (&state_param, &words) in feed.lut_families.iter().zip(&feed.requirements.lut_words) {
            match luts.insert(state_param, words) {
                Some(previous) if previous != words => {
                    return Err(GraphAMultiplicityPlanError::SizeOverflow)
                }
                _ => {}
            }
        }
    }
    if let Some(memory) = &memory_traces {
        match luts.insert("range_check_9_9_state", memory.rc99_lut_words) {
            Some(previous) if previous != memory.rc99_lut_words => {
                return Err(GraphAMultiplicityPlanError::SizeOverflow)
            }
            _ => {}
        }
    }
    let luts = luts
        .into_iter()
        .map(|(state_param, words)| PlannedCanonicalLut { state_param, words })
        .collect::<Vec<_>>();
    let topology_hash = topology_hash(
        &fixed,
        &runtime,
        memory_traces.as_ref(),
        &feeds,
        &luts,
        &coverage_gaps,
        &blockers,
    );
    Ok(GraphAMultiplicityPlan {
        fixed,
        runtime,
        memory_traces,
        feeds,
        luts,
        coverage_gaps,
        blockers,
        topology_hash,
    })
}

fn validate_coverage(
    expected: &BTreeMap<(&'static str, &'static str), u32>,
    actual: &BTreeMap<(&'static str, &'static str), u32>,
) -> Result<Vec<FixedMultiplicityCoverageGap>, GraphAMultiplicityPlanError> {
    if let Some((&(fixed_component, producer), _)) =
        actual.iter().find(|(edge, _)| !expected.contains_key(edge))
    {
        return Err(GraphAMultiplicityPlanError::UnexpectedPreparedProducer {
            fixed_component,
            producer,
        });
    }
    Ok(expected
        .iter()
        .filter_map(|(&(fixed_component, producer), &expected_instances)| {
            let prepared_instances = actual
                .get(&(fixed_component, producer))
                .copied()
                .unwrap_or(0);
            (prepared_instances != expected_instances).then_some(FixedMultiplicityCoverageGap {
                fixed_component,
                producer,
                expected_instances,
                prepared_instances,
            })
        })
        .collect())
}

fn proof_component<'a>(
    proof: &'a ProofPlan,
    component: &'static str,
) -> Result<&'a crate::plan::ComponentPlan, GraphAMultiplicityPlanError> {
    proof
        .components
        .iter()
        .find(|candidate| candidate.node.id == component)
        .ok_or(GraphAMultiplicityPlanError::MissingProofComponent(
            component,
        ))
}

fn one_main_row_count(
    component: &crate::plan::ComponentPlan,
) -> Result<usize, GraphAMultiplicityPlanError> {
    let RowResolution::Resolved(parts) = &component.runtime.rows else {
        return Err(GraphAMultiplicityPlanError::InvalidRows(component.node.id));
    };
    let [part] = parts.as_slice() else {
        return Err(GraphAMultiplicityPlanError::InvalidRows(component.node.id));
    };
    if part.part != TracePartId::Main || part.padded_rows == 0 {
        return Err(GraphAMultiplicityPlanError::InvalidRows(component.node.id));
    }
    usize::try_from(part.padded_rows).map_err(|_| GraphAMultiplicityPlanError::SizeOverflow)
}

fn plan_memory_base_traces(
    proof: &ProofPlan,
) -> Result<Option<PlannedMemoryBaseTraces>, GraphAMultiplicityPlanError> {
    let address = proof_component(proof, "memory_address_to_id")?;
    let values = proof_component(proof, "memory_id_to_big")?;
    match (address.runtime.is_present(), values.runtime.is_present()) {
        (false, false) => return Ok(None),
        (true, true) => {}
        _ => {
            return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
                "memory tables must be present together",
            ))
        }
    }
    let address_rows = one_main_row_count(address)?;
    let address_count_words = address_rows
        .checked_mul(cairo_air::components::memory_address_to_id::MEMORY_ADDRESS_TO_ID_SPLIT)
        .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
    let RowResolution::Resolved(parts) = &values.runtime.rows else {
        return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
            "memory_id_to_big rows are not resolved",
        ));
    };
    let mut big_parts = Vec::new();
    let mut big_count_words = 0usize;
    let mut small_part = None;
    for part in parts {
        let row_count = usize::try_from(part.padded_rows)
            .map_err(|_| GraphAMultiplicityPlanError::SizeOverflow)?;
        if row_count == 0 {
            return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
                "memory trace part is empty",
            ));
        }
        match part.part {
            TracePartId::MemoryBig(_) => {
                let source_offset = big_count_words;
                big_count_words = big_count_words
                    .checked_add(row_count)
                    .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?;
                big_parts.push(PlannedMemoryTracePart {
                    part: part.part,
                    row_count,
                    source_offset,
                });
            }
            TracePartId::MemorySmall if small_part.is_none() => {
                small_part = Some(PlannedMemoryTracePart {
                    part: part.part,
                    row_count,
                    source_offset: 0,
                });
            }
            _ => {
                return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
                    "unexpected or duplicate memory trace part",
                ))
            }
        }
    }
    if big_parts.is_empty() {
        return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
            "memory_id_to_big has no big part",
        ));
    }
    let small_part = small_part.ok_or(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
        "memory_id_to_big has no small part",
    ))?;
    let rc99 = count_relation("range_check_9_9_state").ok_or(
        GraphAMultiplicityPlanError::InvalidMemoryGeometry("missing canonical rc9_9 relation"),
    )?;
    if !rc99.needs_lut || rc99.n_relations != 8 || rc99.table_size == 0 {
        return Err(GraphAMultiplicityPlanError::InvalidMemoryGeometry(
            "canonical rc9_9 relation geometry drifted",
        ));
    }
    Ok(Some(PlannedMemoryBaseTraces {
        address_rows,
        address_count_words,
        big_count_words,
        small_count_words: small_part.row_count,
        rc99_lut_words: rc99.table_size,
        rc99_count_words: rc99
            .table_size
            .checked_mul(rc99.n_relations)
            .ok_or(GraphAMultiplicityPlanError::SizeOverflow)?,
        big_parts,
        small_part,
    }))
}

fn runtime_relation_sizes(
    memory: Option<&PlannedMemoryBaseTraces>,
    state_param: &'static str,
) -> Option<(usize, usize)> {
    let memory = memory?;
    match state_param {
        "memory_address_to_id_state" => Some((memory.address_count_words, 0)),
        "memory_id_to_big_state" => Some((memory.big_count_words, memory.small_count_words)),
        _ => None,
    }
}

fn count_relation(state_param: &str) -> Option<&'static CountRelation> {
    COUNT_RELATIONS
        .iter()
        .find(|relation| relation.state_param == state_param)
}

fn native_memory_rc99_coverage(writer: WitnessWriterSpec) -> bool {
    writer.kind == WitnessWriterKind::NativeCuda && writer.is_capture_safe()
}

fn count_relation_for_fixed(component: &str) -> Option<&'static CountRelation> {
    COUNT_RELATIONS.iter().find(|relation| {
        relation.table_size != 0
            && relation
                .state_param
                .strip_suffix("_state")
                .is_some_and(|candidate| candidate == component)
    })
}

fn fixed_component_for_state(
    state_param: &'static str,
) -> Result<&'static str, GraphAMultiplicityPlanError> {
    state_param
        .strip_suffix("_state")
        .ok_or(GraphAMultiplicityPlanError::UnknownFixedDestination(
            state_param,
        ))
}

fn destination_for_state(
    state_param: &'static str,
) -> Result<&'static str, GraphAMultiplicityPlanError> {
    match state_param {
        "memory_address_to_id_state" => Ok("memory_address_to_id"),
        "memory_id_to_big_state" => Ok("memory_id_to_big"),
        "memory_id_to_big_state#small" => Ok("memory_id_to_big#small"),
        _ => fixed_component_for_state(state_param),
    }
}

fn pointer_words(count: usize) -> Result<usize, GraphAMultiplicityPlanError> {
    count
        .max(1)
        .checked_mul(core::mem::size_of::<*mut u32>().div_ceil(core::mem::size_of::<u32>()))
        .ok_or(GraphAMultiplicityPlanError::SizeOverflow)
}

fn topology_hash(
    fixed: &[PlannedFixedMultiplicity],
    runtime: &[PlannedRuntimeMultiplicity],
    memory: Option<&PlannedMemoryBaseTraces>,
    feeds: &[PlannedRecordedMultiplicityFeed],
    luts: &[PlannedCanonicalLut],
    gaps: &[FixedMultiplicityCoverageGap],
    blockers: &[MultiplicityFeedBlocker],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    };
    feed(b"stwo-cairo-graph-a-multiplicity-v2\0");
    for table in fixed {
        feed(table.component.as_bytes());
        feed(&[0]);
        feed(&(table.row_count as u64).to_le_bytes());
        feed(&(table.columns as u64).to_le_bytes());
        for identity in table.materializer.preprocessed_sources() {
            feed(identity.as_bytes());
            feed(&[0]);
        }
        for column in table
            .materializer
            .requirements()
            .trace_multiplicity_columns()
        {
            feed(&column.to_le_bytes());
        }
        for word in table.materializer.requirements().lookup_descriptors() {
            feed(&word.to_le_bytes());
        }
    }
    for destination in runtime {
        feed(destination.destination.as_bytes());
        feed(&[0]);
        feed(&(destination.words as u64).to_le_bytes());
    }
    if let Some(memory) = memory {
        feed(&(memory.address_rows as u64).to_le_bytes());
        feed(&(memory.rc99_lut_words as u64).to_le_bytes());
        feed(&(memory.rc99_count_words as u64).to_le_bytes());
        for part in memory.big_parts.iter().chain([&memory.small_part]) {
            feed(&match part.part {
                TracePartId::Main => [0, 0, 0, 0],
                TracePartId::MemoryBig(index) => index.to_le_bytes(),
                TracePartId::MemorySmall => u32::MAX.to_le_bytes(),
            });
            feed(&(part.row_count as u64).to_le_bytes());
            feed(&(part.source_offset as u64).to_le_bytes());
        }
    }
    for planned in feeds {
        feed(planned.producer.as_bytes());
        feed(&[0]);
        for word in &planned.descriptors {
            feed(&word.to_le_bytes());
        }
        for family in &planned.lut_families {
            feed(family.as_bytes());
            feed(&[0]);
        }
        for destination in &planned.destination_components {
            feed(destination.as_bytes());
            feed(&[0]);
        }
    }
    for lut in luts {
        feed(lut.state_param.as_bytes());
        feed(&[0]);
        feed(&(lut.words as u64).to_le_bytes());
    }
    for gap in gaps {
        feed(gap.fixed_component.as_bytes());
        feed(&[0]);
        feed(gap.producer.as_bytes());
        feed(&[0]);
        feed(&gap.expected_instances.to_le_bytes());
        feed(&gap.prepared_instances.to_le_bytes());
    }
    for blocker in blockers {
        feed(blocker.producer.as_bytes());
        feed(&[0]);
        feed(blocker.state_param.as_bytes());
        feed(&[match blocker.kind {
            MultiplicityFeedBlockerKind::RuntimeSizedRelation => 0,
            MultiplicityFeedBlockerKind::DependentTuple => 1,
        }]);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::jit_prove_backend::{
        BlakeGRecordedLane, BuiltinLaneSpec,
    };
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

    use super::*;

    fn canonical_blake_g_feed(rows: usize) -> PlannedRecordedMultiplicityFeed {
        let layout = all_lane_sub_feed_layouts()
            .into_iter()
            .find(|layout| layout.component == "blake_g")
            .unwrap();
        let (descriptors, lut_families, destination_states, multiplicity_words) =
            build_feed_descriptors_sized(layout.entries, COUNT_RELATIONS, &|_| None);
        let lut_words = lut_families
            .iter()
            .map(|family| count_relation(family).unwrap().table_size)
            .collect::<Vec<_>>();
        let destination_components = destination_states
            .iter()
            .map(|&state| destination_for_state(state).unwrap())
            .collect::<Vec<_>>();
        PlannedRecordedMultiplicityFeed {
            producer: "blake_g",
            row_count: rows,
            sub_words_per_row: 48,
            requirements: WitnessFeedWorkspaceRequirements {
                row_count: rows,
                sub_words_per_row: 48,
                source_words: rows * 48,
                descriptor_words: descriptors.len(),
                descriptor_count: descriptors.len() / WITNESS_FEED_DESCRIPTOR_WORDS,
                lut_pointer_words: pointer_words(lut_families.len()).unwrap(),
                multiplicity_pointer_words: pointer_words(destination_components.len()).unwrap(),
                lut_words,
                multiplicity_words,
            },
            descriptors,
            lut_families,
            destination_components,
        }
    }

    #[test]
    fn blake_g_fusion_admits_only_the_exact_recorded_feed_topology() {
        let feed = canonical_blake_g_feed(1 << 24);
        assert_eq!(
            blake_g_fused_feed_binding(&feed),
            Some(BlakeGFusedFeedBinding {
                lut_indices: [0, 1, 2, 3],
                destination_indices: [0, 1, 2, 3, 4],
            })
        );

        let mut changed = feed.clone();
        changed.descriptors[0] ^= 1;
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed.clone();
        changed.lut_families.swap(0, 1);
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed.clone();
        changed.destination_components.swap(0, 1);
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed.clone();
        changed.requirements.source_words -= 1;
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed.clone();
        changed.requirements.descriptor_words -= 1;
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed.clone();
        changed.requirements.lut_pointer_words += 1;
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
        let mut changed = feed;
        changed.requirements.row_count -= 1;
        assert_eq!(blake_g_fused_feed_binding(&changed), None);
    }

    #[test]
    fn blake_g_fusion_pins_sn3_sn4_capacity_and_traffic_retirement() {
        for (rows, expected_capacity_bytes, expected_traffic_bytes) in [
            (1usize << 23, 1_610_612_736usize, 3_221_225_472usize),
            (1usize << 24, 3_221_225_472usize, 6_442_450_944usize),
        ] {
            let feed = canonical_blake_g_feed(rows);
            assert!(blake_g_fused_feed_binding(&feed).is_some());
            let capacity_bytes = feed.requirements.source_words * core::mem::size_of::<u32>();
            assert_eq!(capacity_bytes, expected_capacity_bytes);
            assert_eq!(capacity_bytes * 2, expected_traffic_bytes);
        }
    }

    #[test]
    fn blake_g_native_identity_rejects_semantic_drift() {
        let recording = BlakeGRecordedLane::record();
        assert!(recording.poisoned_cols.is_empty());
        assert!(recording.poisoned_lookup_words.is_empty());
        assert!(recording.poisoned_sub_words.is_empty());
        assert!(recording.poison_ops.is_empty());
        assert!(stwo_backend_cuda::blake_g_fusion_program_is_exact(
            &recording.program
        ));

        let mut drifted = recording.program;
        drifted.insts[0].imm ^= 1;
        assert!(!stwo_backend_cuda::blake_g_fusion_program_is_exact(
            &drifted
        ));
    }

    #[test]
    fn blake_g_host_abis_prove_lookup_and_sub_require_distinct_permutations() {
        let layout = all_lane_sub_feed_layouts()
            .into_iter()
            .find(|layout| layout.component == "blake_g")
            .unwrap();
        let sub_fields = layout
            .entries
            .iter()
            .map(|&(field, instance, ..)| (field, instance))
            .collect::<Vec<_>>();
        assert_eq!(
            sub_fields,
            [
                ("verify_bitwise_xor_8", 0),
                ("verify_bitwise_xor_8", 1),
                ("verify_bitwise_xor_8", 2),
                ("verify_bitwise_xor_8", 3),
                ("verify_bitwise_xor_8_b", 0),
                ("verify_bitwise_xor_8_b", 1),
                ("verify_bitwise_xor_8_b", 2),
                ("verify_bitwise_xor_8_b", 3),
                ("verify_bitwise_xor_12", 0),
                ("verify_bitwise_xor_12", 1),
                ("verify_bitwise_xor_4", 0),
                ("verify_bitwise_xor_4", 1),
                ("verify_bitwise_xor_7", 0),
                ("verify_bitwise_xor_7", 1),
                ("verify_bitwise_xor_9", 0),
                ("verify_bitwise_xor_9", 1),
            ]
        );
        assert_eq!(
            &BlakeGRecordedLane::lookup_fields()[..16],
            &[
                ("verify_bitwise_xor_8_0", 4),
                ("verify_bitwise_xor_8_1", 4),
                ("verify_bitwise_xor_8_b_2", 4),
                ("verify_bitwise_xor_8_b_3", 4),
                ("verify_bitwise_xor_12_4", 4),
                ("verify_bitwise_xor_4_5", 4),
                ("verify_bitwise_xor_12_6", 4),
                ("verify_bitwise_xor_4_7", 4),
                ("verify_bitwise_xor_8_8", 4),
                ("verify_bitwise_xor_8_9", 4),
                ("verify_bitwise_xor_8_b_10", 4),
                ("verify_bitwise_xor_8_b_11", 4),
                ("verify_bitwise_xor_7_12", 4),
                ("verify_bitwise_xor_9_13", 4),
                ("verify_bitwise_xor_7_14", 4),
                ("verify_bitwise_xor_9_15", 4),
            ]
        );

        // These are the two independently generated host ABI projections in
        // terms of the native writer's live c[] values. The old CUDA writer
        // used LOOKUP_COLUMNS for both destinations, which permuted eight of
        // sixteen generic-feed tuples. Lookup order itself remains unchanged.
        const LOOKUP_COLUMNS: [[u8; 3]; 16] = [
            [53, 55, 18],
            [14, 16, 19],
            [54, 56, 20],
            [15, 17, 21],
            [57, 59, 28],
            [24, 26, 29],
            [58, 60, 30],
            [25, 27, 31],
            [61, 63, 38],
            [34, 36, 39],
            [62, 64, 40],
            [35, 37, 41],
            [65, 67, 48],
            [44, 46, 49],
            [66, 68, 50],
            [45, 47, 51],
        ];
        const SUB_COLUMNS: [[u8; 3]; 16] = [
            [53, 55, 18],
            [14, 16, 19],
            [61, 63, 38],
            [34, 36, 39],
            [54, 56, 20],
            [15, 17, 21],
            [62, 64, 40],
            [35, 37, 41],
            [57, 59, 28],
            [58, 60, 30],
            [24, 26, 29],
            [25, 27, 31],
            [65, 67, 48],
            [66, 68, 50],
            [44, 46, 49],
            [45, 47, 51],
        ];
        const SUB_TO_LOOKUP: [usize; 16] = [
            0, 1, 8, 9, 2, 3, 10, 11, 4, 6, 5, 7, 12, 14, 13, 15,
        ];
        assert_ne!(SUB_COLUMNS, LOOKUP_COLUMNS);
        for (sub, lookup) in SUB_TO_LOOKUP.into_iter().enumerate() {
            assert_eq!(SUB_COLUMNS[sub], LOOKUP_COLUMNS[lookup]);
        }
    }

    type SparseCounts = BTreeMap<(usize, usize), u32>;

    fn increment(counts: &mut SparseCounts, destination: usize, offset: usize) {
        let count = counts.entry((destination, offset)).or_default();
        *count = count.wrapping_add(1);
    }

    fn generic_blake_g_sparse_counts(
        sub: &[u32],
        rows: usize,
        descriptors: &[u32],
        luts: &[Vec<u32>],
    ) -> SparseCounts {
        let mut counts = SparseCounts::new();
        for descriptor in descriptors.chunks_exact(WITNESS_FEED_DESCRIPTOR_WORDS) {
            let word_base = descriptor[0] as usize;
            let bits = descriptor[2];
            let relation = descriptor[7] as usize;
            let table_size = descriptor[8] as usize;
            let destination = descriptor[10] as usize;
            for row in 0..rows {
                let a = sub[word_base * rows + row];
                let b = sub[(word_base + 1) * rows + row];
                let xor = sub[(word_base + 2) * rows + row];
                if xor != a ^ b {
                    continue;
                }
                let offset = match descriptor[11] {
                    2 if (a | b | xor) < (1 << bits) => {
                        let key = ((a << bits) | b) as usize;
                        let index = luts[descriptor[9] as usize][key] as usize;
                        (index < table_size).then_some(relation * table_size + index)
                    }
                    3 if (a | b | xor) < (1 << 12) => {
                        let column = ((a >> 10) << 2) | (b >> 10);
                        let table_row = ((a & 0x3ff) << 10) | (b & 0x3ff);
                        Some(column as usize * table_size + table_row as usize)
                    }
                    _ => None,
                };
                if let Some(offset) = offset {
                    increment(&mut counts, destination, offset);
                }
            }
        }
        counts
    }

    fn fused_blake_g_sparse_counts(columns: &[[u32; 73]], luts: &[Vec<u32>]) -> SparseCounts {
        let mut counts = SparseCounts::new();
        let lut_pairs = [
            (0, 0, 8, [53, 14, 61, 34], [55, 16, 63, 36]),
            (0, 1, 8, [54, 15, 62, 35], [56, 17, 64, 37]),
            (2, 0, 4, [24, 25, 0, 0], [26, 27, 0, 0]),
            (3, 0, 7, [65, 66, 0, 0], [67, 68, 0, 0]),
            (4, 0, 9, [44, 45, 0, 0], [46, 47, 0, 0]),
        ];
        for column in columns {
            for (family, &(destination, relation, bits, a_columns, b_columns)) in
                lut_pairs.iter().enumerate()
            {
                let pairs = if destination == 0 { 4 } else { 2 };
                let lut = &luts[[0, 0, 1, 2, 3][family]];
                let table_size = 1usize << (2 * bits);
                for pair in 0..pairs {
                    let a = column[a_columns[pair]];
                    let b = column[b_columns[pair]];
                    let index = lut[((a << bits) | b) as usize] as usize;
                    increment(&mut counts, destination, relation * table_size + index);
                }
            }
            for (&a_column, &b_column) in [57, 58].iter().zip([59, 60].iter()) {
                let a = column[a_column];
                let b = column[b_column];
                let relation = ((a >> 10) << 2) | (b >> 10);
                let row = ((a & 0x3ff) << 10) | (b & 0x3ff);
                increment(&mut counts, 1, ((relation << 20) | row) as usize);
            }
        }
        counts
    }

    #[test]
    fn blake_g_fused_count_oracle_matches_generic_words_at_lut_boundaries() {
        // Reverse LUTs make key zero map to the last row and the maximum key
        // map to row zero, exercising both destination boundaries.
        let luts = [8, 4, 7, 9].map(|bits| {
            (0..1usize << (2 * bits))
                .rev()
                .map(|row| row as u32)
                .collect()
        });
        let mut columns = vec![[0u32; 73]; 3];
        // Exact SubComponentInputs declaration order. LookupData contains the
        // same tuples in a different interaction-column order.
        let tuple_columns = [
            53, 55, 18, 14, 16, 19, 61, 63, 38, 34, 36, 39, 54, 56, 20, 15, 17, 21, 62, 64, 40, 35,
            37, 41, 57, 59, 28, 58, 60, 30, 24, 26, 29, 25, 27, 31, 65, 67, 48, 66, 68, 50, 44, 46,
            49, 45, 47, 51,
        ];
        for (row, column) in columns.iter_mut().enumerate() {
            for &(bits, a_columns, b_columns) in &[
                (
                    8,
                    &[53, 14, 61, 34, 54, 15, 62, 35][..],
                    &[55, 16, 63, 36, 56, 17, 64, 37][..],
                ),
                (12, &[57, 58][..], &[59, 60][..]),
                (4, &[24, 25][..], &[26, 27][..]),
                (7, &[65, 66][..], &[67, 68][..]),
                (9, &[44, 45][..], &[46, 47][..]),
            ] {
                let mask = (1u32 << bits) - 1;
                for (pair, (&a_column, &b_column)) in a_columns.iter().zip(b_columns).enumerate() {
                    let (a, b) = match row {
                        0 => (0, 0),
                        1 => (mask, mask),
                        _ => ((pair as u32 + 1) & mask, mask - pair as u32),
                    };
                    column[a_column] = a;
                    column[b_column] = b;
                }
            }
        }
        let rows = columns.len();
        for column in &mut columns {
            for tuple in tuple_columns.chunks_exact(3) {
                column[tuple[2]] = column[tuple[0]] ^ column[tuple[1]];
            }
        }
        let mut sub = vec![0u32; tuple_columns.len() * rows];
        for (word, &column) in tuple_columns.iter().enumerate() {
            for row in 0..rows {
                sub[word * rows + row] = columns[row][column];
            }
        }
        let feed = canonical_blake_g_feed(rows);
        let generic = generic_blake_g_sparse_counts(&sub, rows, &feed.descriptors, &luts);
        let fused = fused_blake_g_sparse_counts(&columns, &luts);
        assert_eq!(fused, generic);
        for &(destination, last) in &[
            (0, (2 << 16) - 1),
            (1, (16 << 20) - 1),
            (2, (1 << 8) - 1),
            (3, (1 << 14) - 1),
            (4, (1 << 18) - 1),
        ] {
            assert!(generic
                .keys()
                .any(|&(slot, offset)| slot == destination && offset == 0));
            assert!(generic
                .keys()
                .any(|&(slot, offset)| slot == destination && offset == last));
        }
    }

    #[test]
    fn public_memory_seed_preserves_duplicate_address_and_id_multiplicities() {
        use stwo_cairo_prover::witness::device_feed::host_feed_counts;

        let seed = plan_public_memory_multiplicity_seed(3, 16, 8, 4).unwrap();
        assert!(seed.lut_families.is_empty());
        assert_eq!(
            seed.destination_components,
            [
                "memory_address_to_id_state",
                "memory_id_to_big_state",
                "memory_id_to_big_state#small"
            ]
        );
        let source = vec![1, 3, 3, 1 << 30, 1, (1 << 30) | 2];
        let mut counts = vec![vec![0; 16], vec![0; 8], vec![0; 4]];
        host_feed_counts(&source, 3, &seed.descriptors, &[], &mut counts);
        assert_eq!(counts[0][0], 1);
        assert_eq!(counts[0][2], 2);
        assert_eq!(counts[1][0], 1);
        assert_eq!(counts[1][2], 1);
        assert_eq!(counts[2][1], 1);
        assert_eq!(
            counts
                .iter()
                .map(|slot| slot.iter().sum::<u32>())
                .sum::<u32>(),
            6
        );
    }
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule::WitnessWriterReadiness;
    use crate::schedule_table::CAIRO_SCHEDULE;

    #[test]
    fn exact_coverage_is_required_before_materialization() {
        let expected = BTreeMap::from([(("range_check_9_9", "cube_252"), 42)]);
        assert!(validate_coverage(&expected, &expected).unwrap().is_empty());

        assert_eq!(
            validate_coverage(&expected, &BTreeMap::new()).unwrap(),
            [FixedMultiplicityCoverageGap {
                fixed_component: "range_check_9_9",
                producer: "cube_252",
                expected_instances: 42,
                prepared_instances: 0,
            }]
        );

        let unexpected = BTreeMap::from([(("range_check_9_9", "unknown"), 1)]);
        assert!(matches!(
            validate_coverage(&expected, &unexpected),
            Err(GraphAMultiplicityPlanError::UnexpectedPreparedProducer { .. })
        ));
    }

    #[test]
    fn memory_rc99_native_coverage_requires_capture_safe_cuda() {
        assert!(native_memory_rc99_coverage(WitnessWriterSpec {
            kind: WitnessWriterKind::NativeCuda,
            readiness: WitnessWriterReadiness::CaptureSafe,
        }));
        assert!(!native_memory_rc99_coverage(WitnessWriterSpec {
            kind: WitnessWriterKind::NativeCuda,
            readiness: WitnessWriterReadiness::ArenaDestination,
        }));
        assert!(!native_memory_rc99_coverage(WitnessWriterSpec::HOST));
    }

    #[test]
    fn recording_and_layout_registries_are_one_to_one() {
        let recordings = stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings()
            .into_iter()
            .map(|(component, _)| component)
            .collect::<BTreeSet<_>>();
        let layouts = all_lane_sub_feed_layouts()
            .into_iter()
            .map(|layout| layout.component)
            .collect::<BTreeSet<_>>();
        assert_eq!(recordings, layouts);
        assert_eq!(recordings.len(), 35);
        assert_eq!(
            compile_cairo_fixed_table_materializations().unwrap().len(),
            22
        );
    }

    #[test]
    fn memory_offsets_follow_padded_host_id_segments_through_partial_and_padding_parts() {
        let default_shape = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let mut components = default_shape.components().to_vec();
        *components
            .iter_mut()
            .find(|component| component.id == "memory_address_to_id")
            .unwrap() = RuntimeComponentShape::uniform("memory_address_to_id", 5, 16).unwrap();
        *components
            .iter_mut()
            .find(|component| component.id == "memory_id_to_big")
            .unwrap() = RuntimeComponentShape::parts(
            "memory_id_to_big",
            vec![
                TracePartShape {
                    part: TracePartId::MemoryBig(0),
                    n_real_rows: 32,
                    padded_rows: 32,
                },
                TracePartShape {
                    part: TracePartId::MemoryBig(1),
                    n_real_rows: 17,
                    padded_rows: 32,
                },
                // An explicitly requested post-final padding component. The
                // host advances its id offset past Big(1)'s padded 32 rows.
                TracePartShape {
                    part: TracePartId::MemoryBig(2),
                    n_real_rows: 16,
                    padded_rows: 16,
                },
                TracePartShape {
                    part: TracePartId::MemorySmall,
                    n_real_rows: 11,
                    padded_rows: 16,
                },
            ],
        )
        .unwrap();
        let shape = ProofShape::new(components).unwrap();
        let proof =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &shape).unwrap();
        let memory = plan_memory_base_traces(&proof).unwrap().unwrap();
        assert_eq!(memory.address_rows, 16);
        assert_eq!(memory.address_count_words, 16 * 16);
        assert_eq!(memory.big_count_words, 32 + 32 + 16);
        assert_eq!(memory.small_count_words, 16);
        assert_eq!(memory.rc99_lut_words, 1 << 18);
        assert_eq!(memory.rc99_count_words, 8 << 18);
        assert_eq!(
            memory
                .big_parts
                .iter()
                .map(|part| (part.part, part.source_offset, part.row_count))
                .collect::<Vec<_>>(),
            [
                (TracePartId::MemoryBig(0), 0, 32),
                (TracePartId::MemoryBig(1), 32, 32),
                (TracePartId::MemoryBig(2), 64, 16),
            ]
        );
        let multiplicity = plan_graph_a_multiplicities(&proof).unwrap();
        assert!(multiplicity
            .luts
            .iter()
            .any(|lut| { lut.state_param == "range_check_9_9_state" && lut.words == 1 << 18 }));
        assert!(!multiplicity.coverage_gaps.iter().any(|gap| {
            gap.fixed_component == "range_check_9_9" && gap.producer == "memory_id_to_big"
        }));
        let mut changed_memory = multiplicity.memory_traces.clone().unwrap();
        changed_memory.rc99_count_words += 1;
        assert_ne!(
            multiplicity.topology_hash,
            topology_hash(
                &multiplicity.fixed,
                &multiplicity.runtime,
                Some(&changed_memory),
                &multiplicity.feeds,
                &multiplicity.luts,
                &multiplicity.coverage_gaps,
                &multiplicity.blockers,
            )
        );
    }
}
