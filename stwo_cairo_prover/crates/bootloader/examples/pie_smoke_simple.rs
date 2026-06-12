//! End-to-end smoke test for the v0.14 **simple** bootloader port.
//!
//! Mirrors `pie_smoke.rs` but feeds the PIE to `run_pie_with_simple_bootloader`
//! (cairo-lang v0.14.0 simple_bootloader.cairo) instead of the full 0.13.3
//! bootloader, then adapts the resulting runner.
//!
//! Run with:
//!   cargo run -p stwo-cairo-bootloader --example pie_smoke_simple --release

use std::path::PathBuf;

use cairo_vm::cairo_run::{cairo_run_program, CairoRunConfig};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;

use stwo_cairo_bootloader::run_pie_with_simple_bootloader;

const OUTPUT_PROGRAM: &[u8] = include_bytes!("output_program.json");

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init()
        .ok();

    // --- Step 1+2: run the inner program (non-proof mode) and build a PIE. ---
    println!("[pie_smoke_simple] running inner output program in non-proof mode...");
    let program = Program::from_bytes(OUTPUT_PROGRAM, Some("main"))?;

    let mut hint_processor = BuiltinHintProcessor::new_empty();
    let config = CairoRunConfig {
        entrypoint: "main",
        trace_enabled: true,
        relocate_mem: true,
        layout: LayoutName::small,
        proof_mode: false,
        ..Default::default()
    };
    let inner_runner = cairo_run_program(&program, &config, &mut hint_processor)
        .map_err(|e| anyhow::anyhow!("inner VM run failed: {e}"))?;

    let pie = inner_runner
        .get_cairo_pie()
        .map_err(|e| anyhow::anyhow!("get_cairo_pie failed: {e}"))?;
    println!(
        "[pie_smoke_simple] built PIE: {} execution steps, builtins={:?}",
        pie.execution_resources.n_steps, pie.metadata.program.builtins
    );

    // --- Step 3: serialize the PIE to a zip. ---
    let tmp_dir = std::env::temp_dir();
    let pie_path: PathBuf = tmp_dir.join("stwo_simple_bootloader_smoke_pie.zip");
    pie.write_zip_file(&pie_path, false)
        .map_err(|e| anyhow::anyhow!("write_zip_file failed: {e}"))?;
    println!("[pie_smoke_simple] wrote PIE zip to {}", pie_path.display());

    // --- Step 4: wrap in the simple bootloader and run in proof mode. ---
    println!("[pie_smoke_simple] running PIE through simple bootloader (proof mode, all_cairo_stwo)...");
    let runner = run_pie_with_simple_bootloader(&pie_path)?;
    println!(
        "[pie_smoke_simple] simple bootloader run complete: {} trace steps",
        runner.get_relocatable_trace().map(|t| t.len()).unwrap_or(0)
    );

    // --- Step 5: adapt for the stwo prover. ---
    println!("[pie_smoke_simple] adapting runner with stwo_cairo_adapter::adapt...");
    let prover_input = stwo_cairo_adapter::adapter::adapt(&runner)
        .map_err(|e| anyhow::anyhow!("adapt failed: {e}"))?;
    println!(
        "[pie_smoke_simple] adapt succeeded: {} state transitions blocks",
        prover_input
            .state_transitions
            .casm_states_by_opcode
            .counts()
            .len()
    );

    if std::env::var("KEEP_PIE").is_err() {
        let _ = std::fs::remove_file(&pie_path);
    }

    println!("[pie_smoke_simple] PASS");
    Ok(())
}
