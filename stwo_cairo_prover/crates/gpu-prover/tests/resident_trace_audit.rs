//! Diagnostic base-trace divergence auditor for the strict resident CUDA path.
//!
//! This is a DEBUGGING INSTRUMENT, not a qualification gate. The manifest
//! validator (`gpu_benchmarks/test_validate_architecture_record.py`)
//! auto-discovers `*_native.rs` test targets and requires exact counted-gate
//! registration for each; this file deliberately does NOT match that glob so it
//! never enters the soundness manifest. Whole-proof qualification remains owned
//! by `tests/resident_parity_native.rs`.
//!
//! What it does: the sn2-profile fixture's whole-proof byte oracle diverges at
//! the BASE-TRACE COMMITMENT root (claim and preprocessed root match exactly),
//! so some component's device-generated base-trace witness CONTENT differs
//! from the SIMD writer's. This runner names the culprit(s) exactly: it enters
//! the strict resident session, captures, and replays the diagnostic
//! witness-only prefix while its ingest inputs are still live. It reads every
//! component's base-trace evaluation columns D2H, independently realizes
//! the SIMD base trace for the same fixture input on the host, and compares
//! per component, per column, over the full padded extent. Each component part
//! gets a verdict:
//!
//! - `MATCH`     — byte-identical.
//! - `GEOMETRY`  — column count / padded rows / real rows disagree (cells not compared).
//! - `PADDING`   — every mismatching cell sits at row >= n_real_rows.
//! - `ORDER`     — real rows mismatch but both sides hold the same multiset of full rows (128-bit
//!   row-hash multiset), i.e. the rows were written in a different order — the prime suspicion for
//!   the device-compacted consumers.
//! - `CONTENT`   — real-row values genuinely differ.
//!
//! The report prints, for every non-matching component: the number of
//! mismatching columns, the first mismatching (column, row) with expected
//! (SIMD) vs actual (device) u32 values, and the mismatching row range, then a
//! final component -> verdict summary table. The test fails if any verdict is
//! not MATCH, after printing the full table.

#![cfg(stwo_cuda_link)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::arena_plan::{CommitmentColumnSource, CommitmentTreeId};
use stwo_cairo_gpu_prover::plan::ProofPlan;
use stwo_cairo_gpu_prover::protocol_plan::trace_commitment_layout;
use stwo_cairo_gpu_prover::relation_table::CAIRO_RELATION_GRAPH;
use stwo_cairo_gpu_prover::resident_runtime::{ResidentGraphRuntime, ResidentRuntimeError};
use stwo_cairo_gpu_prover::schedule_table::CAIRO_SCHEDULE;
use stwo_cairo_gpu_prover::{phases, GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};
use stwo_cairo_prover::witness::base_trace::BaseTrace;
use stwo_cairo_prover::witness::exec_context::WitnessExecContext;
use stwo_cairo_prover::witness::proof_shape::{RowResolution, TracePartId};

/// Same fixture as the strict resident qualification gate and the smoke
/// runner: the input on which the base-trace commitment root diverges.
const STRICT_RESIDENT_FIXTURE: &str = "test_prove_verify_sn2_profile";

/// Production replay generation: capture consumes generation 1, the first warm
/// replay is generation 2 (same literal as `tests/resident_smoke.rs`).

/// Opt-in row capture consumed by the deduce-oracle hardware reproducer.
const DUMP_POSEIDON_KIND11_ROW0_ENV: &str = "STWO_TRACE_AUDIT_DUMP_POSEIDON_KIND11_ROW0";
const POSEIDON_KIND11_REPRO_PATH_ENV: &str = "STWO_POSEIDON_KIND11_REPRO_PATH";

fn resident_input() -> ProverInput {
    run_and_adapt(
        &get_compiled_cairo_program_path(STRICT_RESIDENT_FIXTURE),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap()
}

fn resident_params() -> ProverParameters {
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig::default(),
        preprocessed_trace: PreProcessedTraceVariant::Canonical,
        channel_salt: 0,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

/// One component part's base trace, column-major, full padded extent — the
/// shared shape for both the device readback and the SIMD realization.
struct AuditPart {
    component: &'static str,
    part: TracePartId,
    n_real_rows: u64,
    padded_rows: u64,
    /// `columns[ordinal][row]`, `padded_rows` u32 words per column.
    columns: Vec<Vec<u32>>,
}

impl AuditPart {
    fn label(&self) -> String {
        format!("{}[{:?}]", self.component, self.part)
    }

    fn bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|column| column.len() * core::mem::size_of::<u32>())
            .sum()
    }
}

/// Canonical part order used by `trace_commitment_layout`, so both planes
/// enumerate split-memory parts identically.
fn part_order_key(part: TracePartId) -> (u8, u32) {
    match part {
        TracePartId::Main => (0, 0),
        TracePartId::MemoryBig(index) => (1, index),
        TracePartId::MemorySmall => (2, 0),
    }
}

fn resolved_rows(plan: &ProofPlan, component: &str, part: TracePartId) -> (u64, u64) {
    let shape = plan
        .proof_shape()
        .component(component)
        .unwrap_or_else(|| panic!("component {component} missing from the exact proof shape"));
    let RowResolution::Resolved(parts) = &shape.rows else {
        panic!("component {component} is not exact in the strict resident plan");
    };
    let entry = parts
        .iter()
        .find(|candidate| candidate.part == part)
        .unwrap_or_else(|| panic!("component {component} has no {part:?} part"));
    (entry.n_real_rows, entry.padded_rows)
}

/// D2H every present component part's BaseTrace columns through the
/// diagnostic-only runtime seam. Runs immediately after the witness-only
/// diagnostic replay — see the buffer-lifetime note in the module doc.
fn collect_device_parts(
    runtime: &ResidentGraphRuntime<'_>,
    proof_plan: &ProofPlan,
) -> Result<Vec<AuditPart>, ResidentRuntimeError> {
    let mut out = Vec::new();
    for component in &proof_plan.components {
        let mut parts = match &component.runtime.rows {
            RowResolution::Absent => continue,
            RowResolution::Resolved(parts) => parts.clone(),
            other => panic!(
                "exact resident plan left {} unresolved: {other:?}",
                component.node.id
            ),
        };
        parts.sort_unstable_by_key(|shape| part_order_key(shape.part));
        for shape in parts {
            let columns =
                runtime.read_base_trace_columns_for_diagnostics(component.node.id, shape.part)?;
            out.push(AuditPart {
                component: component.node.id,
                part: shape.part,
                n_real_rows: shape.n_real_rows,
                padded_rows: shape.padded_rows,
                columns,
            });
        }
    }
    Ok(out)
}

/// Realize the SIMD base trace for the same fixture input and slice the flat
/// canonical column vector back into component parts via the exact plan's
/// claim-order layout (`trace_commitment_layout` — the same checked hand-off
/// production staging validates against, see
/// `resident_sources::inspect_base_trace_residency`).
fn simd_reference_parts() -> Vec<AuditPart> {
    let ingest = phases::ingest::run(resident_input(), PreProcessedTraceVariant::Canonical, None);
    let exact_plan = ingest
        .proof_plan
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .expect("strict resident exact plan must resolve on the sealed ingest plan");
    let layout = trace_commitment_layout(&exact_plan)
        .expect("exact plan must yield a claim-order trace commitment layout")
        .base;

    // Same SIMD witness drive as the differential-oracle tests in
    // `tests/fixture_profile.rs`: capacity-bounded ledger in, realized exact
    // shape out.
    let capacity_shape = ingest.generator.proof_shape(None).unwrap();
    let capacity_plan =
        ProofPlan::from_schedule(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH, &capacity_shape).unwrap();
    let exec_context = WitnessExecContext::planned_with_shape(
        Arc::new(CAIRO_SCHEDULE.artifact_plan().unwrap()),
        capacity_plan.proof_shape().clone(),
    );
    let (trace, _claim, _interaction_generator) =
        ingest
            .generator
            .write_trace::<SimdBackend>(&exec_context, None, None);
    let realized = exec_context.seal_final_proof_shape().unwrap();
    // Audit premise: both planes share one geometry. The fixture-profile
    // differential oracle already pins this; if it ever breaks, shape — not
    // content — is the bug to chase first.
    assert_eq!(
        exact_plan.proof_shape(),
        &realized,
        "strict resident plan shape diverges from the SIMD-realized exact shape"
    );

    let BaseTrace::Evals(evals) = trace else {
        panic!("write_trace without pipelined twiddles must return BaseTrace::Evals");
    };
    assert_eq!(
        layout.len(),
        evals.len(),
        "claim-order layout and realized SIMD base trace disagree on column count"
    );

    let mut out: Vec<AuditPart> = Vec::new();
    for (column, eval) in layout.iter().zip(evals) {
        let CommitmentColumnSource::Trace {
            component,
            part,
            ordinal,
            ..
        } = column.source
        else {
            panic!("base commitment layout contains a non-trace source");
        };
        assert_eq!(
            column.log_size,
            eval.domain.log_size(),
            "layout log size disagrees with the realized SIMD evaluation domain \
             for {component}[{part:?}] ordinal {ordinal}"
        );
        let values: Vec<u32> = eval
            .values
            .to_cpu()
            .into_iter()
            .map(|felt| felt.0)
            .collect();
        if out.last().map(|last| (last.component, last.part)) != Some((component, part)) {
            let (n_real_rows, padded_rows) = resolved_rows(&exact_plan, component, part);
            out.push(AuditPart {
                component,
                part,
                n_real_rows,
                padded_rows,
                columns: Vec::new(),
            });
        }
        let entry = out.last_mut().unwrap();
        assert_eq!(
            entry.columns.len(),
            ordinal as usize,
            "claim-order layout ordinals must be consecutive within {component}[{part:?}]"
        );
        entry.columns.push(values);
    }
    out
}

/// Emit the failing component row's exact compact kind-11 arguments. The
/// generated component writes `(chain, round, 4 x 10 W27 state)` into columns
/// 0..42. This captures the component input, not internal intermediates such as
/// `combination_37` (trace column 114).
fn dump_poseidon_kind11_row0(parts: &[AuditPart]) {
    let output_path = std::env::var_os(POSEIDON_KIND11_REPRO_PATH_ENV);
    if std::env::var_os(DUMP_POSEIDON_KIND11_ROW0_ENV).is_none() && output_path.is_none() {
        return;
    }
    let part = parts
        .iter()
        .find(|part| {
            part.component == "poseidon_3_partial_rounds_chain" && part.part == TracePartId::Main
        })
        .expect("poseidon_3_partial_rounds_chain[Main] missing from SIMD trace");
    assert!(
        part.columns.len() >= 42,
        "kind-11 input requires 42 columns"
    );
    let args: Vec<u32> = part.columns[..42].iter().map(|column| column[0]).collect();
    if let Some(path) = output_path {
        let words = args
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(&path, format!("{words}\n"))
            .unwrap_or_else(|error| panic!("failed to write {path:?}: {error}"));
        eprintln!("audit: wrote kind-11 row-0 input to {path:?}");
    }
    eprintln!(
        "audit POSEIDON_KIND11_INPUT component={} part={:?} row=0 args_u32={args:?}",
        part.component, part.part
    );
}

/// 128-bit FNV-1a over one full row (all columns, LE bytes). Used only for the
/// ORDER verdict's sorted row-multiset comparison; 2^-64-scale collision odds
/// are irrelevant for a diagnostic verdict.
fn row_hashes(columns: &[Vec<u32>], rows: usize) -> Vec<u128> {
    (0..rows)
        .map(|row| {
            let mut hash: u128 = 0x6c62272e07bb014262b821756295c58d;
            for column in columns {
                for byte in column[row].to_le_bytes() {
                    hash ^= u128::from(byte);
                    hash = hash.wrapping_mul(0x0000000001000000000000000000013B);
                }
            }
            hash
        })
        .collect()
}

/// Compare one component part cell-for-cell and classify the divergence.
/// Scan order is column-major (columns in ordinal order, rows within), so
/// "first mismatch" is deterministic and names a (column, row) cell.
fn audit_part(device: &AuditPart, simd: &AuditPart) -> (&'static str, Option<String>) {
    // GEOMETRY: shapes must agree before any cell is meaningful.
    if device.columns.len() != simd.columns.len() {
        return (
            "GEOMETRY",
            Some(format!(
                "device has {} columns, SIMD has {}",
                device.columns.len(),
                simd.columns.len()
            )),
        );
    }
    if device.n_real_rows != simd.n_real_rows || device.padded_rows != simd.padded_rows {
        return (
            "GEOMETRY",
            Some(format!(
                "device rows {}/{} vs SIMD rows {}/{} (real/padded)",
                device.n_real_rows, device.padded_rows, simd.n_real_rows, simd.padded_rows
            )),
        );
    }
    for (ordinal, (device_column, simd_column)) in
        device.columns.iter().zip(&simd.columns).enumerate()
    {
        if device_column.len() != simd_column.len()
            || device_column.len() as u64 != device.padded_rows
        {
            return (
                "GEOMETRY",
                Some(format!(
                    "column {ordinal}: device extent {} words, SIMD extent {} words, \
                     planned padded rows {}",
                    device_column.len(),
                    simd_column.len(),
                    device.padded_rows
                )),
            );
        }
    }

    // Cell scan over the full padded extent.
    let rows = device.padded_rows as usize;
    let mut mismatching_columns = 0usize;
    let mut mismatching_cells = 0usize;
    let mut mismatch_rows: BTreeSet<usize> = BTreeSet::new();
    let mut first: Option<(usize, usize, u32, u32)> = None;
    for (ordinal, (device_column, simd_column)) in
        device.columns.iter().zip(&simd.columns).enumerate()
    {
        let mut column_hit = false;
        for row in 0..rows {
            if device_column[row] != simd_column[row] {
                column_hit = true;
                mismatching_cells += 1;
                mismatch_rows.insert(row);
                if first.is_none() {
                    first = Some((ordinal, row, simd_column[row], device_column[row]));
                }
            }
        }
        if column_hit {
            mismatching_columns += 1;
        }
    }
    let Some((first_column, first_row, expected, actual)) = first else {
        return ("MATCH", None);
    };

    let n_real = usize::try_from(device.n_real_rows).unwrap();
    let padding_only = mismatch_rows.iter().all(|&row| row >= n_real);
    // ORDER is only meaningful once real rows mismatch: same full-row multiset
    // on both sides means the rows were written in a different order (the
    // compacted-consumer signature), not with different values.
    let row_multiset_equal = if padding_only {
        false
    } else {
        let mut device_hashes = row_hashes(&device.columns, rows);
        let mut simd_hashes = row_hashes(&simd.columns, rows);
        device_hashes.sort_unstable();
        simd_hashes.sort_unstable();
        device_hashes == simd_hashes
    };
    let verdict = if padding_only {
        "PADDING"
    } else if row_multiset_equal {
        "ORDER"
    } else {
        "CONTENT"
    };
    let detail = format!(
        "{mismatching_columns}/{} mismatching columns, {mismatching_cells} cells across {} rows; \
         first mismatch at (column {first_column}, row {first_row}): \
         expected(SIMD) {expected} ({expected:#010x}) actual(device) {actual} ({actual:#010x}); \
         mismatching rows {}..={} with n_real_rows {n_real}, padded_rows {}; \
         padding_only={padding_only} row_multiset_equal={row_multiset_equal}",
        device.columns.len(),
        mismatch_rows.len(),
        mismatch_rows.first().unwrap(),
        mismatch_rows.last().unwrap(),
        device.padded_rows,
    );
    (verdict, Some(detail))
}

/// One strict resident session replayed through the witness-only diagnostic
/// prefix before base interpolation; every component's device base-trace
/// columns are read D2H and compared against an independent host SIMD
/// realization of the same fixture. Diagnostic
/// instrument: per-boundary sync and bulk D2H deliberately violate the
/// resident hot-path budget, which is why this never asserts it.
#[test]
fn audit_resident_base_trace_against_simd_reference() {
    let params = resident_params();

    // Device plane first, so a device fault surfaces before the ~minutes-long
    // host SIMD realization runs.
    let device_start = Instant::now();
    let mut config = GpuProverConfig::default();
    config.strict = true;
    let mut prover = GpuCairoProver::<Blake2sMerkleChannel>::new(config).unwrap();
    let (device_parts, _telemetry) = prover
        .with_strict_resident_session(resident_input(), params, |runtime, artifacts| {
            runtime.require_prepared_witness_coverage()?;
            runtime.capture_all_prepared_subgraphs()?;
            runtime.replay_witness_only_for_diagnostics()?;
            eprintln!("audit: witness-only prefix replayed; reading device base trace");
            let device_parts = collect_device_parts(runtime, artifacts.proof_plan)?;
            Ok(device_parts)
        })
        .expect("strict resident session failed");
    drop(prover);
    let device_bytes: usize = device_parts.iter().map(AuditPart::bytes).sum();
    eprintln!(
        "audit: device plane done: {} component parts, {} columns, {:.1} MiB D2H, {:.3} s",
        device_parts.len(),
        device_parts
            .iter()
            .map(|part| part.columns.len())
            .sum::<usize>(),
        device_bytes as f64 / (1024.0 * 1024.0),
        device_start.elapsed().as_secs_f64()
    );

    // Host plane: independent SIMD realization of the same input.
    let simd_start = Instant::now();
    let simd_parts = simd_reference_parts();
    dump_poseidon_kind11_row0(&simd_parts);
    eprintln!(
        "audit: SIMD plane done: {} component parts, {} columns, {:.1} MiB, {:.3} s",
        simd_parts.len(),
        simd_parts
            .iter()
            .map(|part| part.columns.len())
            .sum::<usize>(),
        simd_parts.iter().map(AuditPart::bytes).sum::<usize>() as f64 / (1024.0 * 1024.0),
        simd_start.elapsed().as_secs_f64()
    );

    // Compare per component part; collect (label, verdict, detail).
    let mut reports: Vec<(String, &'static str, Option<String>)> = Vec::new();
    for device in &device_parts {
        match simd_parts
            .iter()
            .find(|simd| simd.component == device.component && simd.part == device.part)
        {
            Some(simd) => {
                let (verdict, detail) = audit_part(device, simd);
                reports.push((device.label(), verdict, detail));
            }
            None => reports.push((
                device.label(),
                "GEOMETRY",
                Some("present on device, absent from the SIMD reference layout".to_string()),
            )),
        }
    }
    for simd in &simd_parts {
        if !device_parts
            .iter()
            .any(|device| device.component == simd.component && device.part == simd.part)
        {
            reports.push((
                simd.label(),
                "GEOMETRY",
                Some(
                    "present in the SIMD reference layout, absent from the device plan".to_string(),
                ),
            ));
        }
    }

    // Divergence details first, then the full summary table.
    for (label, verdict, detail) in &reports {
        if *verdict != "MATCH" {
            eprintln!(
                "audit DIVERGENCE {label}: {verdict}: {}",
                detail.as_deref().unwrap_or("")
            );
        }
    }
    eprintln!("audit summary ({} component parts):", reports.len());
    for (label, verdict, _) in &reports {
        eprintln!("  {label:<48} {verdict}");
    }
    let divergent = reports
        .iter()
        .filter(|(_, verdict, _)| *verdict != "MATCH")
        .count();
    assert_eq!(
        divergent, 0,
        "{divergent} component part(s) diverge from the SIMD base trace \
         (verdicts and first-cell evidence above)"
    );
}
