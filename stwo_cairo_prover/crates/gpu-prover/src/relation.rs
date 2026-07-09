//! Pure, generated LogUp plan consumed by the GPU interaction scheduler.
//!
//! The table is emitted from witness interaction writers; this module only owns
//! its compact IR, validation, and deterministic identity. For ordinary
//! components, relation ids are the numeric first words of the generated
//! `LookupData` tuples (`M31_<id>`), not hashes recomputed from field names. The
//! three computed-layout writers use their exact `cairo_air::relations::*_RELATION_ID`
//! constants. Generation also checks the two sources agree wherever a named AIR
//! constant exists.

use std::collections::{BTreeMap, BTreeSet};

use crate::schedule::{ComponentId, ComponentRowSource, Schedule};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ChallengeEpoch {
    BeforeBaseCommitment = 0,
    /// `CommonLookupElements` drawn after the base-trace commitment.
    CommonLookupAfterBaseCommitment = 1,
}

pub const COMMON_LOOKUP_CHALLENGE_EPOCH: ChallengeEpoch =
    ChallengeEpoch::CommonLookupAfterBaseCommitment;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelationIdentity {
    pub name: &'static str,
    pub id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelationTracePart {
    Component,
    EachMemoryBig,
    MemorySmall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TupleSource {
    LookupWords { word_offset: u32 },
    MemoryAddressChunk { chunk: u32 },
    MemoryBigLimbs { first_limb: u32 },
    MemoryBigValue,
    MemorySmallLimbs { first_limb: u32 },
    MemorySmallValue,
    BitwiseXor12 { multiplicity_column: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TupleLayout {
    pub source: TupleSource,
    /// Total denominator tuple words, including the canonical relation id at
    /// `relation_id_word`.
    pub words: u32,
    pub relation_id_word: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenominatorLayout {
    pub tuple: TupleLayout,
    /// `CommonLookupElements::combine` starts at alpha^0.
    pub alpha_power_start: u32,
    /// The common lookup denominator subtracts the drawn `z`.
    pub subtract_z: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MultiplicitySource {
    One,
    Enabler,
    LookupWord { word_offset: u32 },
    MemoryAddressChunk { chunk: u32 },
    MemoryBig,
    MemorySmall,
    BitwiseXor12 { multiplicity_column: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MultiplicitySign {
    Positive = 0,
    Negative = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedMultiplicity {
    pub source: MultiplicitySource,
    pub sign: MultiplicitySign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelationUse {
    pub relation: RelationIdentity,
    pub challenge_epoch: ChallengeEpoch,
    pub denominator: DenominatorLayout,
    pub multiplicity: SignedMultiplicity,
}

#[derive(Clone, Copy, Debug)]
pub struct LogupColumnPlan {
    pub output_column: u32,
    pub uses: &'static [RelationUse],
}

#[derive(Clone, Copy, Debug)]
pub struct RelationTracePlan {
    pub part: RelationTracePart,
    pub output_columns: u32,
    pub columns: &'static [LogupColumnPlan],
}

#[derive(Clone, Copy, Debug)]
pub struct ComponentRelationPlan {
    pub component: ComponentId,
    pub row_source: ComponentRowSource,
    pub lookup_words: Option<u32>,
    pub traces: &'static [RelationTracePlan],
}

pub struct RelationGraph {
    pub relations: &'static [RelationIdentity],
    pub components: &'static [ComponentRelationPlan],
    pub expected_hash: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RelationPlanError {
    DuplicateComponent(ComponentId),
    MissingScheduleComponent(ComponentId),
    MissingRelationComponent(ComponentId),
    RowSourceMismatch(ComponentId),
    DuplicateRelationName(&'static str),
    DuplicateRelationId(u32),
    UnknownRelation(RelationIdentity),
    InvalidRelationId(RelationIdentity),
    WrongChallengeEpoch(ComponentId),
    InvalidTraceParts(ComponentId),
    OutputColumnCountMismatch(ComponentId),
    OutputColumnsNotContiguous(ComponentId),
    InvalidColumnArity(ComponentId, u32),
    DuplicateTupleUse(ComponentId),
    InvalidTupleLayout(ComponentId),
    LookupTupleOutOfBounds(ComponentId),
    LookupMultiplicityOutOfBounds(ComponentId),
    HashMismatch { expected: u64, actual: u64 },
}

impl std::fmt::Display for RelationPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RelationPlanError {}

pub struct RelationPlan<'a> {
    graph: &'a RelationGraph,
    relation_graph_hash: u64,
}

impl RelationPlan<'_> {
    pub fn relation_graph_hash(&self) -> u64 {
        self.relation_graph_hash
    }

    pub fn components(&self) -> &'static [ComponentRelationPlan] {
        self.graph.components
    }

    pub fn component(&self, id: ComponentId) -> Option<&'static ComponentRelationPlan> {
        self.graph
            .components
            .iter()
            .find(|component| component.component == id)
    }
}

impl RelationGraph {
    pub fn plan<'a>(&'a self, schedule: &Schedule) -> Result<RelationPlan<'a>, RelationPlanError> {
        self.validate(schedule)?;
        Ok(RelationPlan {
            graph: self,
            relation_graph_hash: self.relation_graph_hash(),
        })
    }

    pub fn validate(&self, schedule: &Schedule) -> Result<(), RelationPlanError> {
        let mut names = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for relation in self.relations {
            if relation.id == 0 || relation.id >= 0x7fff_ffff {
                return Err(RelationPlanError::InvalidRelationId(*relation));
            }
            if !names.insert(relation.name) {
                return Err(RelationPlanError::DuplicateRelationName(relation.name));
            }
            if !ids.insert(relation.id) {
                return Err(RelationPlanError::DuplicateRelationId(relation.id));
            }
        }
        let catalog: BTreeMap<_, _> = self
            .relations
            .iter()
            .map(|relation| ((relation.name, relation.id), *relation))
            .collect();

        for (index, component) in self.components.iter().enumerate() {
            if self.components[..index]
                .iter()
                .any(|other| other.component == component.component)
            {
                return Err(RelationPlanError::DuplicateComponent(component.component));
            }
            let schedule_node = schedule
                .nodes
                .iter()
                .find(|node| node.id == component.component)
                .ok_or(RelationPlanError::MissingScheduleComponent(
                    component.component,
                ))?;
            if schedule_node.facts.row_source != component.row_source {
                return Err(RelationPlanError::RowSourceMismatch(component.component));
            }
            validate_trace_parts(component)?;
            let mut tuple_uses = BTreeSet::new();
            for trace in component.traces {
                if trace.output_columns as usize != trace.columns.len() {
                    return Err(RelationPlanError::OutputColumnCountMismatch(
                        component.component,
                    ));
                }
                for (output_column, column) in trace.columns.iter().enumerate() {
                    if column.output_column != output_column as u32 {
                        return Err(RelationPlanError::OutputColumnsNotContiguous(
                            component.component,
                        ));
                    }
                    if !(1..=2).contains(&column.uses.len()) {
                        return Err(RelationPlanError::InvalidColumnArity(
                            component.component,
                            column.output_column,
                        ));
                    }
                    for relation_use in column.uses {
                        if relation_use.challenge_epoch != COMMON_LOOKUP_CHALLENGE_EPOCH {
                            return Err(RelationPlanError::WrongChallengeEpoch(
                                component.component,
                            ));
                        }
                        if !catalog
                            .contains_key(&(relation_use.relation.name, relation_use.relation.id))
                        {
                            return Err(RelationPlanError::UnknownRelation(relation_use.relation));
                        }
                        let tuple = relation_use.denominator.tuple;
                        if tuple.words == 0
                            || tuple.relation_id_word != 0
                            || relation_use.denominator.alpha_power_start != 0
                            || !relation_use.denominator.subtract_z
                        {
                            return Err(RelationPlanError::InvalidTupleLayout(component.component));
                        }
                        if !tuple_uses.insert((trace.part, tuple.source)) {
                            return Err(RelationPlanError::DuplicateTupleUse(component.component));
                        }
                        if let TupleSource::LookupWords { word_offset } = tuple.source {
                            let Some(lookup_words) = component.lookup_words else {
                                return Err(RelationPlanError::LookupTupleOutOfBounds(
                                    component.component,
                                ));
                            };
                            if word_offset
                                .checked_add(tuple.words)
                                .is_none_or(|end| end > lookup_words)
                            {
                                return Err(RelationPlanError::LookupTupleOutOfBounds(
                                    component.component,
                                ));
                            }
                        }
                        if let MultiplicitySource::LookupWord { word_offset } =
                            relation_use.multiplicity.source
                        {
                            if component
                                .lookup_words
                                .is_none_or(|lookup_words| word_offset >= lookup_words)
                            {
                                return Err(RelationPlanError::LookupMultiplicityOutOfBounds(
                                    component.component,
                                ));
                            }
                        }
                    }
                }
                if trace.part == RelationTracePart::Component
                    && schedule_node.facts.logup_columns != Some(trace.output_columns)
                {
                    return Err(RelationPlanError::OutputColumnCountMismatch(
                        component.component,
                    ));
                }
            }
        }
        for node in schedule.nodes {
            if !self
                .components
                .iter()
                .any(|component| component.component == node.id)
            {
                return Err(RelationPlanError::MissingRelationComponent(node.id));
            }
        }
        let actual = self.relation_graph_hash();
        if actual != self.expected_hash {
            return Err(RelationPlanError::HashMismatch {
                expected: self.expected_hash,
                actual,
            });
        }
        Ok(())
    }

    /// Stable FNV-1a projection of all proof-relevant relation metadata.
    pub fn relation_graph_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for relation in self.relations {
            hash_str(&mut hash, relation.name);
            hash_u32(&mut hash, relation.id);
        }
        for component in self.components {
            hash_str(&mut hash, component.component);
            let (row_source_tag, row_source_payload) = row_source_identity(component.row_source);
            hash_u32(&mut hash, row_source_tag);
            hash_u32(&mut hash, row_source_payload);
            hash_u32(&mut hash, component.lookup_words.unwrap_or(u32::MAX));
            for trace in component.traces {
                hash_u32(&mut hash, trace_part_tag(trace.part));
                hash_u32(&mut hash, trace.output_columns);
                for column in trace.columns {
                    hash_u32(&mut hash, column.output_column);
                    hash_u32(&mut hash, column.uses.len() as u32);
                    for relation_use in column.uses {
                        hash_str(&mut hash, relation_use.relation.name);
                        hash_u32(&mut hash, relation_use.relation.id);
                        hash_u32(&mut hash, relation_use.challenge_epoch as u32);
                        hash_u32(
                            &mut hash,
                            tuple_source_tag(relation_use.denominator.tuple.source),
                        );
                        hash_u32(
                            &mut hash,
                            tuple_source_index(relation_use.denominator.tuple.source),
                        );
                        hash_u32(&mut hash, relation_use.denominator.tuple.words);
                        hash_u32(&mut hash, relation_use.denominator.tuple.relation_id_word);
                        hash_u32(&mut hash, relation_use.denominator.alpha_power_start);
                        hash_u32(&mut hash, relation_use.denominator.subtract_z as u32);
                        hash_u32(
                            &mut hash,
                            multiplicity_source_tag(relation_use.multiplicity.source),
                        );
                        hash_u32(
                            &mut hash,
                            multiplicity_source_index(relation_use.multiplicity.source),
                        );
                        hash_u32(&mut hash, relation_use.multiplicity.sign as u32);
                    }
                }
            }
        }
        hash
    }
}

fn validate_trace_parts(component: &ComponentRelationPlan) -> Result<(), RelationPlanError> {
    let parts: Vec<_> = component.traces.iter().map(|trace| trace.part).collect();
    let valid = match component.row_source {
        ComponentRowSource::MemoryIdToBig => {
            parts
                == [
                    RelationTracePart::EachMemoryBig,
                    RelationTracePart::MemorySmall,
                ]
        }
        _ => parts == [RelationTracePart::Component],
    };
    valid
        .then_some(())
        .ok_or(RelationPlanError::InvalidTraceParts(component.component))
}

fn hash_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn hash_str(hash: &mut u64, value: &str) {
    hash_u32(hash, value.len() as u32);
    for byte in value.bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn trace_part_tag(part: RelationTracePart) -> u32 {
    match part {
        RelationTracePart::Component => 0,
        RelationTracePart::EachMemoryBig => 1,
        RelationTracePart::MemorySmall => 2,
    }
}

fn row_source_identity(source: ComponentRowSource) -> (u32, u32) {
    match source {
        ComponentRowSource::DirectInputs => (0, 0),
        ComponentRowSource::StoredLogSize => (1, 0),
        ComponentRowSource::FixedLogSize(log_size) => (2, log_size),
        ComponentRowSource::WitnessRelationFeeds => (3, 0),
        ComponentRowSource::MemoryAddress => (4, 0),
        ComponentRowSource::MemoryIdToBig => (5, 0),
    }
}

fn tuple_source_tag(source: TupleSource) -> u32 {
    match source {
        TupleSource::LookupWords { .. } => 0,
        TupleSource::MemoryAddressChunk { .. } => 1,
        TupleSource::MemoryBigLimbs { .. } => 2,
        TupleSource::MemoryBigValue => 3,
        TupleSource::MemorySmallLimbs { .. } => 4,
        TupleSource::MemorySmallValue => 5,
        TupleSource::BitwiseXor12 { .. } => 6,
    }
}

fn tuple_source_index(source: TupleSource) -> u32 {
    match source {
        TupleSource::LookupWords { word_offset } => word_offset,
        TupleSource::MemoryAddressChunk { chunk } => chunk,
        TupleSource::MemoryBigLimbs { first_limb } => first_limb,
        TupleSource::MemorySmallLimbs { first_limb } => first_limb,
        TupleSource::BitwiseXor12 {
            multiplicity_column,
        } => multiplicity_column,
        TupleSource::MemoryBigValue | TupleSource::MemorySmallValue => 0,
    }
}

fn multiplicity_source_tag(source: MultiplicitySource) -> u32 {
    match source {
        MultiplicitySource::One => 0,
        MultiplicitySource::Enabler => 1,
        MultiplicitySource::LookupWord { .. } => 2,
        MultiplicitySource::MemoryAddressChunk { .. } => 3,
        MultiplicitySource::MemoryBig => 4,
        MultiplicitySource::MemorySmall => 5,
        MultiplicitySource::BitwiseXor12 { .. } => 6,
    }
}

fn multiplicity_source_index(source: MultiplicitySource) -> u32 {
    match source {
        MultiplicitySource::LookupWord { word_offset } => word_offset,
        MultiplicitySource::MemoryAddressChunk { chunk } => chunk,
        MultiplicitySource::BitwiseXor12 {
            multiplicity_column,
        } => multiplicity_column,
        MultiplicitySource::One
        | MultiplicitySource::Enabler
        | MultiplicitySource::MemoryBig
        | MultiplicitySource::MemorySmall => 0,
    }
}
