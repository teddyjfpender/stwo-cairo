//! End-to-end smoke test for the vendored cairo-bootloader port.
//!
//! 1. Compile-time-embedded minimal Cairo0 program that uses the `output`
//!    builtin (`examples/output_program.json`).
//! 2. Run it through cairo-vm in NON-proof mode (layout `small`, which supports
//!    the output builtin) and extract a `CairoPie` via `get_cairo_pie()`.
//! 3. Serialize the PIE to a zip with `write_zip_file`.
//! 4. Feed that zip to `run_pie_with_bootloader` (proof mode, `all_cairo_stwo`).
//! 5. Adapt the resulting runner with `stwo_cairo_adapter::adapt` and assert it
//!    succeeds.
//!
//! Run with:
//!   cargo run -p stwo-cairo-bootloader --example pie_smoke --release

use std::path::PathBuf;

use cairo_vm::cairo_run::{cairo_run_program, CairoRunConfig};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;

use stwo_cairo_bootloader::run_pie_with_bootloader;

const OUTPUT_PROGRAM: &[u8] = include_bytes!("output_program.json");

fn main() -> anyhow::Result<()> {
    // Show the adapter's `info!("Opcode counts: ...")` line.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init()
        .ok();

    // --- Step 1+2: run the inner program (non-proof mode) and build a PIE. ---
    println!("[pie_smoke] running inner output program in non-proof mode...");
    let program = Program::from_bytes(OUTPUT_PROGRAM, Some("main"))?;

    let mut hint_processor = BuiltinHintProcessor::new_empty();
    let config = CairoRunConfig {
        entrypoint: "main",
        // Non-proof-mode run with a layout that supports the output builtin.
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
        "[pie_smoke] built PIE: {} execution steps, builtins={:?}",
        pie.execution_resources.n_steps,
        pie.metadata.program.builtins
    );

    // --- Step 3: serialize the PIE to a zip. ---
    let tmp_dir = std::env::temp_dir();
    let pie_path: PathBuf = tmp_dir.join("stwo_bootloader_smoke_pie.zip");
    pie.write_zip_file(&pie_path, false)
        .map_err(|e| anyhow::anyhow!("write_zip_file failed: {e}"))?;
    println!("[pie_smoke] wrote PIE zip to {}", pie_path.display());

    // --- Step 4: wrap in the bootloader and run in proof mode. ---
    println!("[pie_smoke] running PIE through bootloader (proof mode, all_cairo_stwo)...");
    let runner = run_pie_with_bootloader(&pie_path)?;
    println!(
        "[pie_smoke] bootloader run complete: {} trace steps",
        runner.get_relocatable_trace().map(|t| t.len()).unwrap_or(0)
    );

    // --- Step 5: adapt for the stwo prover. ---
    println!("[pie_smoke] adapting runner with stwo_cairo_adapter::adapt...");
    let prover_input = stwo_cairo_adapter::adapter::adapt(&runner)
        .map_err(|e| anyhow::anyhow!("adapt failed: {e}"))?;
    println!(
        "[pie_smoke] adapt succeeded: {} state transitions blocks",
        prover_input.state_transitions.casm_states_by_opcode.counts().len()
    );

    // Clean up the temp PIE.
    if std::env::var("KEEP_PIE").is_err() {
        let _ = std::fs::remove_file(&pie_path);
    }

    println!("[pie_smoke] PASS");
    Ok(())
}
