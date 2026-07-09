//! Exact transcript-size discovery before interaction execution.
//!
//! OODS sample topology depends on the Cairo component set, trace shape,
//! preprocessed-column binding and PCS lifting degree, but not on the values of
//! lookup challenges or claimed sums. This pass therefore constructs a
//! schema-derived zero interaction claim and runs the canonical STWO
//! `Components::mask_points` topology without generating an interaction trace.

use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::{CairoClaim, CairoInteractionClaim};
use cairo_air::relations::CommonLookupElements;
use num_traits::Zero;
use serde_json::{Map, Value};
use stwo::core::air::Components;
use stwo::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::utils::try_get_lifting_log_size;
use stwo::core::pcs::{PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::plan::ProofPlan;
use crate::relation_execution::{RelationExecutionError, RelationExecutionPlan};
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::transcript_plan::DynamicTranscriptShape;

const TRACE_TREE_COUNT: usize = 3;
const COMPOSITION_SAMPLE_FELTS: usize = 2 * SECURE_EXTENSION_DEGREE;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolTranscriptDiscovery {
    pub interaction_claim_felts: usize,
    pub oods_sampled_value_felts: usize,
    pub sampled_value_felts_by_tree: Vec<usize>,
    /// Coefficient-domain log sizes of the prepared partial numerators, in the
    /// exact sample-point order consumed by `compute_quotients_and_combine`.
    pub partial_numerator_log_sizes: Vec<u32>,
    pub lifting_log_size: u32,
    pub max_log_degree_bound: u32,
}

impl ProtocolTranscriptDiscovery {
    pub fn dynamic_transcript_shape(&self) -> DynamicTranscriptShape {
        DynamicTranscriptShape {
            interaction_claim_felts: Some(self.interaction_claim_felts),
            oods_sampled_values_felts: Some(self.oods_sampled_value_felts),
        }
    }
}

#[derive(Debug)]
pub enum ProtocolDiscoveryError {
    ProofPlanNotExact,
    Schema(String),
    MissingClaimField(&'static str),
    UnknownClaimField(String),
    ClaimPlanPresenceMismatch(&'static str),
    ClaimPlanLogSizeMismatch {
        component: &'static str,
        claim: u32,
        plan: u32,
    },
    InvalidPlanTraceParts(&'static str),
    MissingMemoryIdToBig,
    InvalidMemoryClaim,
    MemoryPartCountMismatch {
        claim_big: usize,
        plan_big: usize,
        claim_small: bool,
        plan_small: bool,
    },
    InteractionClaimFeltMismatch {
        claim: usize,
        relation_plan: usize,
    },
    InvalidLiftingLogSize {
        lifting: u32,
        required: u32,
    },
    InvalidMaskTreeCount(usize),
    InvalidMaskColumnCount {
        tree: usize,
        points: usize,
        log_sizes: usize,
    },
    SampledColumnExceedsLifting {
        coefficient_log_size: u32,
        max_log_degree_bound: u32,
    },
    ComponentConstruction,
    SizeOverflow,
    Relation(RelationExecutionError),
}

impl core::fmt::Display for ProtocolDiscoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid Cairo protocol discovery: {self:?}")
    }
}

impl std::error::Error for ProtocolDiscoveryError {}

impl From<RelationExecutionError> for ProtocolDiscoveryError {
    fn from(value: RelationExecutionError) -> Self {
        Self::Relation(value)
    }
}

/// Discover the exact two dynamic transcript mix lengths before lookup
/// interaction execution. `split_composition_log_size` is the committed split
/// composition-tree size passed to STWO's `try_get_lifting_log_size` in
/// `prove_ex`; this preserves the reference path when explicit lifting is set.
pub fn discover_protocol_transcript_shape(
    claim: &CairoClaim,
    proof_plan: &ProofPlan,
    preprocessed_column_ids: &[PreProcessedColumnId],
    pcs: &PcsConfig,
    split_composition_log_size: u32,
    include_all_preprocessed_columns: bool,
) -> Result<ProtocolTranscriptDiscovery, ProtocolDiscoveryError> {
    if !proof_plan.capture_ready() {
        return Err(ProtocolDiscoveryError::ProofPlanNotExact);
    }
    let (interaction_claim, claim_fields) = schema_zero_interaction_claim(claim)?;
    validate_claim_against_plan(&claim_fields, proof_plan)?;

    let interaction_claim_felts = interaction_claim.flatten_interaction_claim().len();
    let relation_plan = RelationExecutionPlan::from_proof_plan(proof_plan, &CAIRO_RELATION_GRAPH)?;
    let relation_claim_felts = relation_plan.requirements()?.instances.len();
    if interaction_claim_felts != relation_claim_felts {
        return Err(ProtocolDiscoveryError::InteractionClaimFeltMismatch {
            claim: interaction_claim_felts,
            relation_plan: relation_claim_felts,
        });
    }

    let lifting_log_size =
        try_get_lifting_log_size(pcs, split_composition_log_size).map_err(|error| {
            ProtocolDiscoveryError::InvalidLiftingLogSize {
                lifting: error.lifting_log_size,
                required: error.min_log_size,
            }
        })?;
    let max_log_degree_bound = lifting_log_size
        .checked_sub(pcs.fri_config.log_blowup_factor)
        .ok_or(ProtocolDiscoveryError::SizeOverflow)?;
    let topology = protocol_mask_topology(
        claim,
        &interaction_claim,
        preprocessed_column_ids,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    )?;
    let sampled_value_felts_by_tree = topology
        .sample_points
        .0
        .iter()
        .map(|tree| tree.iter().map(Vec::len).sum())
        .collect::<Vec<_>>();
    let oods_sampled_value_felts = sampled_value_felts_by_tree
        .iter()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or(ProtocolDiscoveryError::SizeOverflow)?;
    let partial_numerator_log_sizes = partial_numerator_log_sizes(
        &topology.sample_points,
        &topology.coefficient_log_sizes,
        lifting_log_size,
        pcs.fri_config.log_blowup_factor,
    )?;

    Ok(ProtocolTranscriptDiscovery {
        interaction_claim_felts,
        oods_sampled_value_felts,
        sampled_value_felts_by_tree,
        partial_numerator_log_sizes,
        lifting_log_size,
        max_log_degree_bound,
    })
}

struct ProtocolMaskTopology {
    sample_points: TreeVec<Vec<Vec<CirclePoint<SecureField>>>>,
    coefficient_log_sizes: TreeVec<Vec<u32>>,
}

fn protocol_mask_topology(
    claim: &CairoClaim,
    interaction_claim: &CairoInteractionClaim,
    preprocessed_column_ids: &[PreProcessedColumnId],
    max_log_degree_bound: u32,
    include_all_preprocessed_columns: bool,
) -> Result<ProtocolMaskTopology, ProtocolDiscoveryError> {
    let (mut sample_points, mut coefficient_log_sizes) = catch_unwind(AssertUnwindSafe(|| {
        let cairo_components = CairoComponents::new(
            claim,
            &CommonLookupElements::dummy(),
            interaction_claim,
            preprocessed_column_ids,
        );
        let components = Components {
            components: cairo_components.components(),
            n_preprocessed_columns: preprocessed_column_ids.len(),
        };
        (
            components.mask_points(
                SECURE_FIELD_CIRCLE_GEN,
                max_log_degree_bound,
                include_all_preprocessed_columns,
            ),
            components.column_log_sizes(),
        )
    }))
    .map_err(|_| ProtocolDiscoveryError::ComponentConstruction)?;
    if sample_points.len() != TRACE_TREE_COUNT || coefficient_log_sizes.len() != TRACE_TREE_COUNT {
        return Err(ProtocolDiscoveryError::InvalidMaskTreeCount(
            sample_points.len(),
        ));
    }
    for (tree, (points, log_sizes)) in sample_points
        .iter()
        .zip(coefficient_log_sizes.iter())
        .enumerate()
    {
        if points.len() != log_sizes.len() {
            return Err(ProtocolDiscoveryError::InvalidMaskColumnCount {
                tree,
                points: points.len(),
                log_sizes: log_sizes.len(),
            });
        }
    }
    sample_points.push(vec![
        vec![SECURE_FIELD_CIRCLE_GEN];
        COMPOSITION_SAMPLE_FELTS
    ]);
    coefficient_log_sizes.push(vec![max_log_degree_bound; COMPOSITION_SAMPLE_FELTS]);
    Ok(ProtocolMaskTopology {
        sample_points,
        coefficient_log_sizes,
    })
}

#[derive(Clone, Copy)]
struct PartialNumeratorCandidate {
    point: CirclePoint<SecureField>,
    coefficient_log_size: u32,
}

/// Mirrors the shape-only part of STWO's quotient preparation:
/// `build_samples_with_randomness_and_periodicity`, grouping within each
/// evaluation-domain log, then the final `accumulations_per_sample_point`
/// lift. The lifted vector for a point has the largest coefficient-domain log
/// among columns sampled at that point.
fn partial_numerator_log_sizes(
    sample_points: &TreeVec<Vec<Vec<CirclePoint<SecureField>>>>,
    coefficient_log_sizes: &TreeVec<Vec<u32>>,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<Vec<u32>, ProtocolDiscoveryError> {
    if sample_points.len() != coefficient_log_sizes.len() {
        return Err(ProtocolDiscoveryError::InvalidMaskTreeCount(
            sample_points.len(),
        ));
    }
    let lifting_domain_generator = CanonicCoset::new(lifting_log_size).step();
    let max_log_degree_bound = lifting_log_size
        .checked_sub(log_blowup_factor)
        .ok_or(ProtocolDiscoveryError::SizeOverflow)?;
    let mut candidates = Vec::new();

    for (tree, (points_by_column, logs_by_column)) in sample_points
        .iter()
        .zip(coefficient_log_sizes.iter())
        .enumerate()
    {
        if points_by_column.len() != logs_by_column.len() {
            return Err(ProtocolDiscoveryError::InvalidMaskColumnCount {
                tree,
                points: points_by_column.len(),
                log_sizes: logs_by_column.len(),
            });
        }
        for (points, &coefficient_log_size) in points_by_column.iter().zip(logs_by_column) {
            if points.is_empty() {
                continue;
            }
            if coefficient_log_size > max_log_degree_bound {
                return Err(ProtocolDiscoveryError::SampledColumnExceedsLifting {
                    coefficient_log_size,
                    max_log_degree_bound,
                });
            }
            let evaluation_log_size = coefficient_log_size
                .checked_add(log_blowup_factor)
                .ok_or(ProtocolDiscoveryError::SizeOverflow)?;

            // STWO adds this periodicity sample for exactly two mask points,
            // including when it aliases the OODS point at the maximal size.
            if let [_, point] = points.as_slice() {
                let period_generator =
                    lifting_domain_generator.repeated_double(evaluation_log_size);
                candidates.push(PartialNumeratorCandidate {
                    point: *point + period_generator.into_ef(),
                    coefficient_log_size,
                });
            }
            candidates.extend(
                points
                    .iter()
                    .copied()
                    .map(|point| PartialNumeratorCandidate {
                        point,
                        coefficient_log_size,
                    }),
            );
        }
    }

    candidates.sort_by_key(|candidate| (candidate.point.x, candidate.point.y));
    let mut grouped: Vec<(CirclePoint<SecureField>, u32)> = Vec::new();
    for candidate in candidates {
        if let Some((point, log_size)) = grouped.last_mut() {
            if *point == candidate.point {
                *log_size = (*log_size).max(candidate.coefficient_log_size);
                continue;
            }
        }
        grouped.push((candidate.point, candidate.coefficient_log_size));
    }
    Ok(grouped.into_iter().map(|(_, log_size)| log_size).collect())
}

/// Build every enabled interaction field from the serialized CairoClaim
/// schema. This intentionally has no hand-maintained component list: a new or
/// renamed claim field changes the key set and fails the round-trip coverage
/// check until the upstream interaction schema changes with it.
fn schema_zero_interaction_claim(
    claim: &CairoClaim,
) -> Result<(CairoInteractionClaim, Map<String, Value>), ProtocolDiscoveryError> {
    let Value::Object(mut claim_fields) = serde_json::to_value(claim)
        .map_err(|error| ProtocolDiscoveryError::Schema(error.to_string()))?
    else {
        return Err(ProtocolDiscoveryError::Schema(
            "CairoClaim did not serialize as an object".into(),
        ));
    };
    claim_fields
        .remove("public_data")
        .ok_or_else(|| ProtocolDiscoveryError::Schema("missing public_data".into()))?;
    let claim_schema = claim_fields.clone();
    let zero = serde_json::to_value(SecureField::zero())
        .map_err(|error| ProtocolDiscoveryError::Schema(error.to_string()))?;
    for (field, value) in &mut claim_fields {
        if value.is_null() {
            continue;
        }
        let mut interaction = Map::new();
        if field == "memory_id_to_big" {
            let big_count = value
                .get("big_log_sizes")
                .and_then(Value::as_array)
                .ok_or(ProtocolDiscoveryError::InvalidMemoryClaim)?
                .len();
            interaction.insert(
                "big_claimed_sums".into(),
                Value::Array(vec![zero.clone(); big_count]),
            );
        }
        interaction.insert("claimed_sum".into(), zero.clone());
        *value = Value::Object(interaction);
    }
    if claim_fields
        .get("memory_id_to_big")
        .is_none_or(Value::is_null)
    {
        return Err(ProtocolDiscoveryError::MissingMemoryIdToBig);
    }

    let interaction_claim: CairoInteractionClaim =
        serde_json::from_value(Value::Object(claim_fields.clone()))
            .map_err(|error| ProtocolDiscoveryError::Schema(error.to_string()))?;
    let Value::Object(roundtrip) = serde_json::to_value(&interaction_claim)
        .map_err(|error| ProtocolDiscoveryError::Schema(error.to_string()))?
    else {
        return Err(ProtocolDiscoveryError::Schema(
            "CairoInteractionClaim did not serialize as an object".into(),
        ));
    };
    let claim_keys = claim_schema
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let interaction_keys = roundtrip
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if claim_keys != interaction_keys
        || claim_schema.iter().any(|(field, claim_value)| {
            roundtrip.get(field).is_none_or(|interaction_value| {
                claim_value.is_null() != interaction_value.is_null()
            })
        })
    {
        return Err(ProtocolDiscoveryError::Schema(
            "Cairo claim and interaction-claim field coverage diverged".into(),
        ));
    }
    Ok((interaction_claim, claim_schema))
}

fn validate_claim_against_plan(
    claim_fields: &Map<String, Value>,
    proof_plan: &ProofPlan,
) -> Result<(), ProtocolDiscoveryError> {
    for component in &proof_plan.components {
        if component.node.id == "memory_id_to_big" {
            validate_memory_claim(claim_fields, &component.runtime.rows)?;
            continue;
        }
        let claim_value = claim_fields
            .get(component.node.id)
            .ok_or(ProtocolDiscoveryError::MissingClaimField(component.node.id))?;
        let claim_present = !claim_value.is_null();
        if claim_present != component.runtime.is_present() {
            return Err(ProtocolDiscoveryError::ClaimPlanPresenceMismatch(
                component.node.id,
            ));
        }
        if !claim_present {
            continue;
        }
        let RowResolution::Resolved(parts) = &component.runtime.rows else {
            return Err(ProtocolDiscoveryError::ProofPlanNotExact);
        };
        if parts.len() != 1 || parts[0].part != TracePartId::Main {
            return Err(ProtocolDiscoveryError::InvalidPlanTraceParts(
                component.node.id,
            ));
        }
        if let Some(claim_log_size) = claim_value.get("log_size").and_then(Value::as_u64) {
            let claim_log_size =
                u32::try_from(claim_log_size).map_err(|_| ProtocolDiscoveryError::SizeOverflow)?;
            let plan_log_size = exact_log_size(component.node.id, parts[0].padded_rows)?;
            if claim_log_size != plan_log_size {
                return Err(ProtocolDiscoveryError::ClaimPlanLogSizeMismatch {
                    component: component.node.id,
                    claim: claim_log_size,
                    plan: plan_log_size,
                });
            }
        }
    }

    for field in claim_fields.keys() {
        if field == "memory_id_to_small" {
            continue;
        }
        if !proof_plan
            .components
            .iter()
            .any(|component| component.node.id == field)
        {
            return Err(ProtocolDiscoveryError::UnknownClaimField(field.clone()));
        }
    }
    Ok(())
}

fn validate_memory_claim(
    claim_fields: &Map<String, Value>,
    rows: &RowResolution,
) -> Result<(), ProtocolDiscoveryError> {
    let big_claim = claim_fields
        .get("memory_id_to_big")
        .filter(|value| !value.is_null())
        .ok_or(ProtocolDiscoveryError::MissingMemoryIdToBig)?;
    let big_logs = big_claim
        .get("big_log_sizes")
        .and_then(Value::as_array)
        .ok_or(ProtocolDiscoveryError::InvalidMemoryClaim)?;
    let small_claim = claim_fields
        .get("memory_id_to_small")
        .ok_or(ProtocolDiscoveryError::InvalidMemoryClaim)?;
    let claim_small = !small_claim.is_null();
    let RowResolution::Resolved(parts) = rows else {
        return Err(ProtocolDiscoveryError::ProofPlanNotExact);
    };
    let mut big_parts = parts
        .iter()
        .filter_map(|part| match part.part {
            TracePartId::MemoryBig(index) => Some((index, part.padded_rows)),
            _ => None,
        })
        .collect::<Vec<_>>();
    big_parts.sort_unstable_by_key(|(index, _)| *index);
    let small_part = parts
        .iter()
        .find(|part| part.part == TracePartId::MemorySmall);
    if big_logs.len() != big_parts.len() || claim_small != small_part.is_some() {
        return Err(ProtocolDiscoveryError::MemoryPartCountMismatch {
            claim_big: big_logs.len(),
            plan_big: big_parts.len(),
            claim_small,
            plan_small: small_part.is_some(),
        });
    }
    for (expected_index, ((actual_index, rows), log)) in big_parts.iter().zip(big_logs).enumerate()
    {
        if usize::try_from(*actual_index).ok() != Some(expected_index) {
            return Err(ProtocolDiscoveryError::InvalidMemoryClaim);
        }
        let claim_log = log
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(ProtocolDiscoveryError::InvalidMemoryClaim)?;
        let plan_log = exact_log_size("memory_id_to_big", *rows)?;
        if claim_log != plan_log {
            return Err(ProtocolDiscoveryError::ClaimPlanLogSizeMismatch {
                component: "memory_id_to_big",
                claim: claim_log,
                plan: plan_log,
            });
        }
    }
    if let Some(part) = small_part {
        let claim_log = small_claim
            .get("log_size")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(ProtocolDiscoveryError::InvalidMemoryClaim)?;
        let plan_log = exact_log_size("memory_id_to_big", part.padded_rows)?;
        if claim_log != plan_log {
            return Err(ProtocolDiscoveryError::ClaimPlanLogSizeMismatch {
                component: "memory_id_to_big",
                claim: claim_log,
                plan: plan_log,
            });
        }
    }
    Ok(())
}

fn exact_log_size(
    component: &'static str,
    padded_rows: u64,
) -> Result<u32, ProtocolDiscoveryError> {
    if padded_rows == 0 || !padded_rows.is_power_of_two() {
        return Err(ProtocolDiscoveryError::InvalidPlanTraceParts(component));
    }
    Ok(padded_rows.ilog2())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_traits::One;
    use serde_json::json;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::quotients::{build_samples_with_randomness_and_periodicity, PointSample};
    use stwo::core::pcs::PcsConfig;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTrace;
    use stwo_cairo_prover::witness::cairo_claim_generator::CairoClaimGenerator;
    use stwo_cairo_prover::witness::proof_shape::{
        ProofShape, RuntimeComponentShape, TracePartShape,
    };

    use super::*;
    use crate::relation_table::CAIRO_RELATION_GRAPH;
    use crate::schedule_table::CAIRO_SCHEDULE;

    fn representative_plan(n_big: usize) -> ProofPlan {
        let initial = CairoClaimGenerator::default().proof_shape(None).unwrap();
        let capacity =
            ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &initial).unwrap();
        let components = capacity
            .components
            .iter()
            .map(|component| {
                if component.node.id == "memory_id_to_big" {
                    let mut parts = (0..n_big)
                        .map(|index| TracePartShape {
                            part: TracePartId::MemoryBig(index as u32),
                            n_real_rows: 1 << (4 + index),
                            padded_rows: 1 << (4 + index),
                        })
                        .collect::<Vec<_>>();
                    parts.push(TracePartShape {
                        part: TracePartId::MemorySmall,
                        n_real_rows: 16,
                        padded_rows: 16,
                    });
                    return RuntimeComponentShape::parts(component.node.id, parts).unwrap();
                }
                match &component.runtime.rows {
                    RowResolution::Absent => RuntimeComponentShape::absent(component.node.id),
                    RowResolution::Resolved(parts) => {
                        RuntimeComponentShape::parts(component.node.id, parts.clone()).unwrap()
                    }
                    RowResolution::Bounded { bound, .. } => RuntimeComponentShape::uniform(
                        component.node.id,
                        bound.max_rows,
                        bound.padded_capacity,
                    )
                    .unwrap(),
                    RowResolution::Pending { .. } => {
                        panic!("capacity plan must resolve pending rows")
                    }
                }
            })
            .collect();
        let exact = ProofShape::new(components).unwrap();
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &exact).unwrap()
    }

    fn claim_for_plan(plan: &ProofPlan) -> CairoClaim {
        let mut fields = Map::new();
        fields.insert(
            "public_data".into(),
            serde_json::to_value(cairo_air::air::PublicData::default()).unwrap(),
        );
        for component in &plan.components {
            let RowResolution::Resolved(parts) = &component.runtime.rows else {
                continue;
            };
            if component.node.id == "memory_id_to_big" {
                let mut big = parts
                    .iter()
                    .filter_map(|part| match part.part {
                        TracePartId::MemoryBig(index) => Some((index, part.padded_rows.ilog2())),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                big.sort_unstable_by_key(|(index, _)| *index);
                fields.insert(
                    "memory_id_to_big".into(),
                    json!({"big_log_sizes": big.into_iter().map(|(_, log)| log).collect::<Vec<_>>() }),
                );
                let small = parts
                    .iter()
                    .find(|part| part.part == TracePartId::MemorySmall)
                    .unwrap();
                fields.insert(
                    "memory_id_to_small".into(),
                    json!({"log_size": small.padded_rows.ilog2()}),
                );
            } else {
                fields.insert(
                    component.node.id.into(),
                    json!({"log_size": parts[0].padded_rows.ilog2()}),
                );
            }
        }
        serde_json::from_value(Value::Object(fields)).unwrap()
    }

    fn pcs() -> PcsConfig {
        PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: Some(26),
        }
    }

    fn reference_partial_numerator_log_sizes(
        sample_points: &TreeVec<Vec<Vec<CirclePoint<SecureField>>>>,
        coefficient_log_sizes: &TreeVec<Vec<u32>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
    ) -> Vec<u32> {
        let samples = TreeVec(
            sample_points
                .iter()
                .map(|tree| {
                    tree.iter()
                        .map(|column| {
                            column
                                .iter()
                                .copied()
                                .map(|point| PointSample {
                                    point,
                                    value: SecureField::zero(),
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        );
        let evaluation_log_sizes = coefficient_log_sizes
            .iter()
            .map(|tree| {
                tree.iter()
                    .map(|&log_size| log_size + log_blowup_factor)
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect::<Vec<_>>();
        let expanded = build_samples_with_randomness_and_periodicity(
            &samples,
            evaluation_log_sizes,
            lifting_log_size,
            SecureField::one(),
        );

        let mut grouped = BTreeMap::new();
        for (expanded_tree, logs_tree) in expanded.iter().zip(coefficient_log_sizes.iter()) {
            for (expanded_column, &coefficient_log_size) in expanded_tree.iter().zip(logs_tree) {
                for (sample, _) in expanded_column {
                    grouped
                        .entry((sample.point.x, sample.point.y))
                        .and_modify(|log_size: &mut u32| {
                            *log_size = (*log_size).max(coefficient_log_size)
                        })
                        .or_insert(coefficient_log_size);
                }
            }
        }
        grouped.into_values().collect()
    }

    #[test]
    fn discovery_matches_actual_mask_point_flattening_for_split_shapes() {
        let preprocessed = PreProcessedTrace::canonical();
        for n_big in [1, 3] {
            let plan = representative_plan(n_big);
            let claim = claim_for_plan(&plan);
            let (interaction_claim, _) = schema_zero_interaction_claim(&claim).unwrap();
            for include_all in [false, true] {
                let discovered = discover_protocol_transcript_shape(
                    &claim,
                    &plan,
                    &preprocessed.ids(),
                    &pcs(),
                    26,
                    include_all,
                )
                .unwrap();

                let cairo_components = CairoComponents::new(
                    &claim,
                    &CommonLookupElements::dummy(),
                    &interaction_claim,
                    &preprocessed.ids(),
                );
                let components = Components {
                    components: cairo_components.components(),
                    n_preprocessed_columns: preprocessed.ids().len(),
                };
                let mut sample_points = components.mask_points(
                    SECURE_FIELD_CIRCLE_GEN,
                    discovered.max_log_degree_bound,
                    include_all,
                );
                sample_points.push(vec![
                    vec![SECURE_FIELD_CIRCLE_GEN];
                    COMPOSITION_SAMPLE_FELTS
                ]);
                let actual_count = sample_points.clone().flatten_cols().len();
                let actual_by_tree = sample_points
                    .0
                    .iter()
                    .map(|tree| tree.iter().map(Vec::len).sum())
                    .collect::<Vec<usize>>();
                assert_eq!(discovered.oods_sampled_value_felts, actual_count);
                assert_eq!(discovered.sampled_value_felts_by_tree, actual_by_tree);
                let mut coefficient_log_sizes = components.column_log_sizes();
                coefficient_log_sizes.push(vec![
                    discovered.max_log_degree_bound;
                    COMPOSITION_SAMPLE_FELTS
                ]);
                assert_eq!(
                    discovered.partial_numerator_log_sizes,
                    reference_partial_numerator_log_sizes(
                        &sample_points,
                        &coefficient_log_sizes,
                        discovered.lifting_log_size,
                        pcs().fri_config.log_blowup_factor,
                    )
                );
                assert_eq!(
                    discovered.interaction_claim_felts,
                    interaction_claim.flatten_interaction_claim().len()
                );
                assert_eq!(
                    discovered
                        .dynamic_transcript_shape()
                        .interaction_claim_felts,
                    Some(discovered.interaction_claim_felts)
                );
            }
        }
    }

    #[test]
    fn discovery_rejects_a_claim_plan_presence_mismatch() {
        let plan = representative_plan(1);
        let mut claim = claim_for_plan(&plan);
        claim.add_opcode = None;
        assert!(matches!(
            discover_protocol_transcript_shape(
                &claim,
                &plan,
                &PreProcessedTrace::canonical().ids(),
                &pcs(),
                26,
                false,
            ),
            Err(ProtocolDiscoveryError::ClaimPlanPresenceMismatch(
                "add_opcode"
            ))
        ));
    }
}
