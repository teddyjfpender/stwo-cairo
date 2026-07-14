//! Scratch-only emitter for the pinned v13 witness phase-split resource experiment.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use stwo_backend_cuda::jit_witness::codegen::phase_plan::{
    BoundarySource, WitnessPhasePlan, PHASE_CODEGEN_VERSION,
};
use stwo_backend_cuda::jit_witness::codegen::phases::{
    compile_witness_to_phase_sources, EmittedWitnessPhaseProgram,
};
use stwo_backend_cuda::jit_witness::codegen::WITNESS_CODEGEN_VERSION;
use stwo_backend_cuda::jit_witness::isa::WitnessProgram;
use stwo_cairo_prover::witness::jit_prove_backend::all_lane_recordings;

#[derive(Clone, Copy)]
struct Target {
    label: &'static str,
    after_deduce: usize,
    boundary: usize,
    constants: usize,
    inputs: usize,
    outputs: usize,
    moved_stores: usize,
    scratch_words_per_row: u32,
}

struct ValidatedTarget {
    target: Target,
    plan: WitnessPhasePlan,
    emitted: EmittedWitnessPhaseProgram,
    constants: usize,
    inputs: usize,
    outputs: usize,
}

const TARGETS: [Target; 3] = [
    Target {
        label: "partial_ec_mul_generic",
        after_deduce: 9,
        boundary: 98,
        constants: 13,
        inputs: 57,
        outputs: 28,
        moved_stores: 28,
        scratch_words_per_row: 0,
    },
    Target {
        label: "poseidon_3_partial_rounds_chain",
        after_deduce: 16,
        boundary: 80,
        constants: 11,
        inputs: 19,
        outputs: 50,
        moved_stores: 50,
        scratch_words_per_row: 0,
    },
    Target {
        label: "poseidon_3_partial_rounds_chain",
        after_deduce: 31,
        boundary: 72,
        constants: 11,
        inputs: 1,
        outputs: 60,
        moved_stores: 60,
        scratch_words_per_row: 0,
    },
];

fn output_dir(args: &[String]) -> Result<PathBuf, &'static str> {
    match args {
        [_, option, path]
            if option == "--output-dir" && !path.is_empty() && !path.starts_with("--") =>
        {
            Ok(PathBuf::from(path))
        }
        _ => Err("usage: witness_phase_emit --output-dir <fresh-scratch-directory>"),
    }
}

fn admit_fresh_directory(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    let mut entries =
        std::fs::read_dir(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if entries.next().is_some() {
        return Err(format!("{} must be absent or empty", path.display()));
    }
    Ok(())
}

fn validate_target(programs: &[(&'static str, WitnessProgram)], target: Target) -> ValidatedTarget {
    let program = programs
        .iter()
        .find(|(label, _)| *label == target.label)
        .map(|(_, program)| program)
        .unwrap_or_else(|| panic!("missing witness program {}", target.label));
    let plan = WitnessPhasePlan::after_deduce(program, target.after_deduce)
        .unwrap_or_else(|error| panic!("plan {}: {error:?}", target.label));
    let constants = plan
        .boundary
        .iter()
        .filter(|value| matches!(value.source, BoundarySource::Constant(_)))
        .count();
    let inputs = plan
        .boundary
        .iter()
        .filter(|value| matches!(value.source, BoundarySource::Input(_)))
        .count();
    let outputs = plan
        .boundary
        .iter()
        .filter(|value| matches!(value.source, BoundarySource::Output(_)))
        .count();
    assert_eq!(plan.after_deduce_ordinal, Some(target.after_deduce));
    assert_eq!(
        plan.boundary.len(),
        target.boundary,
        "{} boundary",
        target.label
    );
    assert_eq!(constants, target.constants, "{} constants", target.label);
    assert_eq!(
        inputs, target.inputs,
        "{} input rematerializations",
        target.label
    );
    assert_eq!(outputs, target.outputs, "{} output reloads", target.label);
    assert_eq!(
        plan.moved_stores.len(),
        target.moved_stores,
        "{} moved stores",
        target.label
    );
    assert_eq!(
        plan.scratch_words_per_row, target.scratch_words_per_row,
        "{} scratch",
        target.label
    );

    // Emission independently recomputes and compares the canonical plan. These
    // identity checks also make a stale phase source impossible to name as this plan.
    let emitted = compile_witness_to_phase_sources(program, &plan)
        .unwrap_or_else(|error| panic!("emit {}: {error:?}", target.label));
    assert_eq!(emitted.parent_semantic_hash, plan.parent_semantic_hash);
    assert_eq!(emitted.plan_hash, plan.plan_hash);
    assert_eq!(emitted.scratch_words_per_row, plan.scratch_words_per_row);
    assert_eq!(emitted.phases.len(), 2);
    for (ordinal, phase) in emitted.phases.iter().enumerate() {
        let ordinal = ordinal as u32;
        assert_eq!(phase.ordinal, ordinal);
        assert_eq!(phase.kernel_name, plan.phase_kernel_name(ordinal));
        assert_eq!(phase.cache_key, plan.phase_cache_key(ordinal));
    }
    ValidatedTarget {
        target,
        plan,
        emitted,
        constants,
        inputs,
        outputs,
    }
}

fn main() -> ExitCode {
    let args = std::env::args().collect::<Vec<_>>();
    let out = match output_dir(&args) {
        Ok(out) => out,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = admit_fresh_directory(&out) {
        eprintln!("witness_phase_emit: refusing output: {error}");
        return ExitCode::FAILURE;
    }
    let programs = all_lane_recordings();
    let targets = TARGETS.map(|target| validate_target(&programs, target));
    std::fs::create_dir_all(&out).expect("create phase output directory");
    let mut manifest = Vec::new();

    for validated in targets {
        let ValidatedTarget {
            target,
            plan,
            emitted,
            constants,
            inputs,
            outputs,
        } = validated;
        let mut phase_records = Vec::new();
        for phase in emitted.phases {
            let file = format!(
                "witness_phase_{}_after_deduce_{}_plan_{:016x}_p{}_{:016x}.cu",
                target.label, target.after_deduce, plan.plan_hash, phase.ordinal, phase.cache_key,
            );
            std::fs::write(out.join(&file), &phase.source).expect("write phase source");
            phase_records.push(serde_json::json!({
                "ordinal": phase.ordinal,
                "kernel_name": phase.kernel_name,
                "cache_key": format!("{:016x}", phase.cache_key),
                "file": file,
                "source_bytes": phase.source.len(),
                "source_lines": phase.source.lines().count(),
            }));
        }
        manifest.push(serde_json::json!({
            "label": target.label,
            "witness_codegen_version": WITNESS_CODEGEN_VERSION,
            "phase_codegen_version": PHASE_CODEGEN_VERSION,
            "parent_semantic_hash": format!("{:016x}", plan.parent_semantic_hash),
            "after_deduce_ordinal": target.after_deduce,
            "cut_instruction": plan.cut_instruction,
            "plan_hash": format!("{:016x}", plan.plan_hash),
            "boundary_values": plan.boundary.len(),
            "constant_rematerializations": constants,
            "input_rematerializations": inputs,
            "output_materializations": outputs,
            "global_reload_words_per_row": inputs + outputs,
            "requested_boundary_bytes_per_row": (inputs + outputs) * std::mem::size_of::<u32>(),
            "moved_stores": plan.moved_stores.len(),
            "scratch_words_per_row": plan.scratch_words_per_row,
            "phases": phase_records,
        }));
    }
    std::fs::write(
        out.join("phase_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .expect("write phase manifest");
    println!(
        "witness_phase_emit: {} plans, {} kernels, zero scratch -> {}",
        manifest.len(),
        manifest.len() * 2,
        out.display()
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_program_phase_targets_match_pinned_contracts() {
        let programs = all_lane_recordings();
        for target in TARGETS {
            validate_target(&programs, target);
        }
    }

    #[test]
    #[should_panic(expected = "boundary")]
    fn real_program_phase_target_rejects_contract_mutation() {
        let programs = all_lane_recordings();
        let mut target = TARGETS[0];
        target.boundary += 1;
        validate_target(&programs, target);
    }

    #[test]
    fn output_requires_one_exact_argument_pair() {
        let valid = ["bin", "--output-dir", "scratch"].map(str::to_owned);
        assert_eq!(output_dir(&valid).unwrap(), PathBuf::from("scratch"));
        for invalid in [
            vec!["bin"],
            vec!["bin", "--output-dir"],
            vec!["bin", "--output-dir", ""],
            vec!["bin", "--output-dir", "--check"],
            vec!["bin", "--output-dir", "scratch", "extra"],
        ] {
            assert!(
                output_dir(&invalid.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
            );
        }
    }
}
