//! Exact current-plan receipt for the address-free composition-wave candidate.
//!
//! Historical H100 counts are comparison metadata only. Every current count and
//! logical byte is derived from the exact `CompositionPlan` passed by preflight.

use serde_json::{json, Value};
use stwo_cairo_gpu_prover::composition_plan::CompositionPlan;
use stwo_cairo_gpu_prover::composition_wave::{CompositionWaveProgram, CompositionWaveTraffic};

const OLD_H100_CONSTRAINT_LAUNCHES: usize = 81;
const OLD_H100_WAVE_DOMAINS: usize = 17;
const OLD_H100_RAW_CONSTRAINT_NS: u64 = 248_080_296;
const REMAINING_OLD_H100_FAMILY_CEILING_NS: u64 = 304_575_796;

pub(crate) fn json(plan: &CompositionPlan) -> Result<Value, String> {
    let program = CompositionWaveProgram::from_plan(plan).map_err(|error| error.to_string())?;
    program
        .validate_against(plan)
        .map_err(|error| error.to_string())?;
    let traffic = program.traffic().map_err(|error| error.to_string())?;
    validate_program(&program, traffic)?;

    let parts = program
        .parts()
        .iter()
        .map(|part| {
            json!({
                "ordinal": part.ordinal,
                "component_index": part.component_index,
                "kernel_index": part.kernel_index,
                "component": part.component,
                "instance": part.instance,
                "evaluation_log_size": part.evaluation_log_size,
                "row_count": part.row_count,
                "cache_key": format!("{:016x}", part.cache_key),
                "semantic_hash": format!("{:016x}", part.semantic_hash),
                "coefficient_start": part.coefficient_start,
                "coefficient_end": part.coefficient_end,
                "coefficient_count": part.coefficient_end - part.coefficient_start,
            })
        })
        .collect::<Vec<_>>();
    let waves = program
        .waves()
        .iter()
        .map(|wave| {
            let coefficient_ranges = wave
                .part_ordinals
                .iter()
                .map(|&ordinal| {
                    let part = &program.parts()[ordinal];
                    json!([part.coefficient_start, part.coefficient_end])
                })
                .collect::<Vec<_>>();
            json!({
                "evaluation_log_size": wave.evaluation_log_size,
                "row_count": wave.row_count,
                "part_count": wave.part_ordinals.len(),
                "part_ordinals": wave.part_ordinals,
                "coefficient_ranges": coefficient_ranges,
            })
        })
        .collect::<Vec<_>>();
    let program_bytes = serde_json::to_vec(&json!({
        "source_plan_key": format!("{:016x}", program.source_plan_key()),
        "total_constraints": program.total_constraints(),
        "parts": parts,
        "waves": waves,
    }))
    .map_err(|error| format!("composition wave receipt serialization failed: {error}"))?;
    let program_blake3 = blake3::hash(&program_bytes).to_hex().to_string();

    Ok(json!({
        "schema": "stwo-current-plan-composition-wave-frontier-v1",
        "status": "address-free-current-plan-model-only-native-not-implemented",
        "source": "report.arena.composition().plan",
        "source_plan_key": format!("{:016x}", program.source_plan_key()),
        "program_blake3": program_blake3,
        "claim_boundary": "current plan structure and logical accumulator traffic only; no native execution or duration claim",
        "native_implementation_present": false,
        "h100_timing_credit_ns": 0,
        "physical_arena_credit_bytes": 0,
        "current_plan": {
            "components": plan.components.len(),
            "total_constraints": program.total_constraints(),
            "constraint_part_launches": traffic.legacy_launches,
            "wave_domain_launches": traffic.wave_launches,
            "parts": parts,
            "waves": waves,
        },
        "hard_current_plan_facts": {
            "legacy_row_passes": traffic.legacy_row_passes,
            "distinct_accumulator_rows": traffic.distinct_accumulator_rows,
            "legacy_clear_bytes": traffic.legacy_clear_bytes,
            "legacy_read_modify_write_bytes": traffic.legacy_read_modify_write_bytes,
            "legacy_total_accumulator_bytes": traffic.legacy_total_bytes,
            "candidate_time_credit_ns": 0,
        },
        "conditional_modeled_frontier": {
            "wave_final_write_bytes": traffic.wave_final_write_bytes,
            "eliminated_logical_accumulator_bytes": traffic.eliminated_bytes,
            "eliminated_fraction_ppm": fraction_ppm(
                traffic.eliminated_bytes,
                traffic.legacy_total_bytes,
            )?,
            "native_byte_identity_required": true,
            "candidate_time_credit_ns": 0,
        },
        "historical_old_h100_reference": {
            "observed_constraint_launches": OLD_H100_CONSTRAINT_LAUNCHES,
            "observed_wave_domains": OLD_H100_WAVE_DOMAINS,
            "observed_raw_constraint_ns": OLD_H100_RAW_CONSTRAINT_NS,
            "observed_counts_match_current": traffic.legacy_launches == OLD_H100_CONSTRAINT_LAUNCHES
                && traffic.wave_launches == OLD_H100_WAVE_DOMAINS,
            "bound_to_current_source_plan_key": false,
            "admitted_as_current": false,
            "candidate_time_credit_ns": 0,
            "reason": "historical trace has no current CompositionPlan key; its 81-to-17 counts never populate current fields",
        },
        "uncredited_potential": {
            "remaining_normalized_old_h100_family_ceiling_ns":
                REMAINING_OLD_H100_FAMILY_CEILING_NS,
            "interpretation": "family ceiling only; not a duration prediction for this candidate",
            "candidate_time_credit_ns": 0,
        },
        "admission_gates": {
            "current_plan_rederived": true,
            "canonical_coefficient_order_validated": true,
            "part_partition_validated": true,
            "native_lowering_present": false,
            "native_byte_identity": false,
            "sm90_resource_receipt": false,
            "sanitizer": false,
            "bounded_same_h100_ab": false,
            "timing_claim_admitted": false,
        },
    }))
}

fn validate_program(
    program: &CompositionWaveProgram,
    traffic: CompositionWaveTraffic,
) -> Result<(), String> {
    if traffic.legacy_launches != program.parts().len() {
        return Err("composition wave legacy launch count drift".to_owned());
    }
    if traffic.wave_launches != program.waves().len() {
        return Err("composition wave domain launch count drift".to_owned());
    }
    let mut coefficient_cursor = 0usize;
    for (ordinal, part) in program.parts().iter().enumerate() {
        if part.ordinal != ordinal || part.coefficient_start != coefficient_cursor {
            return Err(format!(
                "composition wave canonical coefficient order drift at part {ordinal}"
            ));
        }
        if part.coefficient_end <= part.coefficient_start {
            return Err(format!(
                "composition wave empty coefficient span at part {ordinal}"
            ));
        }
        coefficient_cursor = part.coefficient_end;
    }
    if coefficient_cursor != program.total_constraints() {
        return Err("composition wave coefficient coverage drift".to_owned());
    }

    let mut seen = vec![false; program.parts().len()];
    let mut previous_log = None;
    for wave in program.waves() {
        if previous_log.is_some_and(|previous| previous >= wave.evaluation_log_size) {
            return Err("composition wave log order drift".to_owned());
        }
        previous_log = Some(wave.evaluation_log_size);
        let expected_rows = 1u64
            .checked_shl(wave.evaluation_log_size)
            .ok_or_else(|| "composition wave log does not fit row count".to_owned())?;
        if wave.row_count != expected_rows || wave.part_ordinals.is_empty() {
            return Err(format!(
                "composition wave log {} geometry drift",
                wave.evaluation_log_size
            ));
        }
        let mut previous_ordinal = None;
        for &ordinal in &wave.part_ordinals {
            let part = program
                .parts()
                .get(ordinal)
                .ok_or_else(|| format!("composition wave part {ordinal} is out of range"))?;
            if std::mem::replace(&mut seen[ordinal], true) {
                return Err(format!("composition wave part {ordinal} is duplicated"));
            }
            if previous_ordinal.is_some_and(|previous| previous >= ordinal)
                || part.evaluation_log_size != wave.evaluation_log_size
                || part.row_count != wave.row_count
            {
                return Err(format!("composition wave part {ordinal} ownership drift"));
            }
            previous_ordinal = Some(ordinal);
        }
    }
    if seen.iter().any(|seen| !seen) {
        return Err("composition wave part partition is incomplete".to_owned());
    }
    Ok(())
}

fn fraction_ppm(numerator: u64, denominator: u64) -> Result<u64, String> {
    numerator
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_div(denominator))
        .ok_or_else(|| "composition wave fraction overflow or zero denominator".to_owned())
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo_cairo_gpu_prover::composition_plan::{
        CompositionComponentPlan, CompositionKernelPart,
    };

    use super::*;

    fn component(
        component: &'static str,
        instance: usize,
        log_size: u32,
        constraints: usize,
        coefficient_offset: usize,
        parts: &[(u64, u32)],
    ) -> CompositionComponentPlan {
        CompositionComponentPlan {
            component,
            instance,
            trace_locations: Vec::new(),
            preprocessed_column_indices: Vec::new(),
            trace_log_size: log_size - 1,
            evaluation_log_size: log_size,
            n_constraints: constraints,
            random_coefficient_offset: coefficient_offset,
            denominator_inverses: vec![BaseField::from(1)],
            base_param_values: Vec::new(),
            ext_param_values: Vec::new(),
            ext_param_sources: Vec::new(),
            kernels: parts
                .iter()
                .map(|&(id, rc_base)| CompositionKernelPart {
                    kernel_name: format!("receipt_test_{id}"),
                    cache_key: id,
                    semantic_hash: id + 10_000,
                    source: format!("source_{id}"),
                    rc_base,
                })
                .collect(),
        }
    }

    fn small_plan() -> CompositionPlan {
        CompositionPlan {
            max_kernel_instrs: 192,
            total_constraints: 9,
            max_evaluation_log_size: 8,
            components: vec![
                component("a", 0, 8, 5, 0, &[(1, 0), (2, 2)]),
                component("b", 0, 6, 4, 5, &[(3, 0), (4, 1)]),
            ],
        }
    }

    #[test]
    fn receipt_uses_current_plan_counts_and_zero_credit() {
        let receipt = json(&small_plan()).unwrap();
        assert_eq!(receipt["current_plan"]["constraint_part_launches"], 4);
        assert_eq!(receipt["current_plan"]["wave_domain_launches"], 2);
        assert_eq!(receipt["current_plan"]["total_constraints"], 9);
        assert_eq!(receipt["h100_timing_credit_ns"], 0);
        assert_eq!(receipt["physical_arena_credit_bytes"], 0);
        assert_eq!(
            receipt["historical_old_h100_reference"]["admitted_as_current"],
            false
        );
    }

    #[test]
    fn malformed_current_plan_fails_closed() {
        let mut plan = small_plan();
        plan.components[1].random_coefficient_offset += 1;
        assert!(json(&plan).is_err());
    }

    #[test]
    fn matching_historical_counts_still_receive_zero_credit() {
        let components = (0..OLD_H100_CONSTRAINT_LAUNCHES)
            .map(|ordinal| {
                component(
                    "historical_count_collision",
                    ordinal,
                    2 + u32::try_from(ordinal % OLD_H100_WAVE_DOMAINS).unwrap(),
                    1,
                    ordinal,
                    &[(u64::try_from(ordinal).unwrap() + 1, 0)],
                )
            })
            .collect();
        let plan = CompositionPlan {
            max_kernel_instrs: 192,
            total_constraints: OLD_H100_CONSTRAINT_LAUNCHES,
            max_evaluation_log_size: 18,
            components,
        };
        let receipt = json(&plan).unwrap();
        let historical = &receipt["historical_old_h100_reference"];
        assert_eq!(historical["observed_counts_match_current"], true);
        assert_eq!(historical["bound_to_current_source_plan_key"], false);
        assert_eq!(historical["admitted_as_current"], false);
        assert_eq!(historical["candidate_time_credit_ns"], 0);
        assert_eq!(receipt["h100_timing_credit_ns"], 0);
    }
}
