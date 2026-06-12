//! # stwo-cairo-bootloader
//!
//! Vendored & ported fork of the Cairo bootloader.
//!
//! NOTICE
//! ------
//! This crate is derived from **Moonsong-Labs/cairo-bootloader**
//! (<https://github.com/Moonsong-Labs/cairo-bootloader>), licensed under the
//! Apache License, Version 2.0. The original copyright and license are retained
//! in the `LICENSE` file shipped alongside this crate.
//!
//! Changes made in this fork:
//!   * Ported from the upstream `cairo-vm` git pin to the workspace's
//!     `cairo-vm 3.2.0` (crates.io) — hint-processor trait signatures
//!     (no `constants` argument under `extensive_hints`), the 3-argument
//!     `OutputBuiltinRunner::new_state(base, base_offset, included)`, etc.
//!   * Added the high-level [`run_pie_with_bootloader`] entry point that wraps a
//!     single Starknet Cairo PIE into the bootloader program and runs it in proof
//!     mode with the `all_cairo_stwo` layout, returning a [`CairoRunner`] ready to
//!     be handed to `stwo_cairo_adapter::adapt`.
//!
//! The hint logic itself (program loading, PIE memory relocation, program-hash
//! chain, fact topologies, builtin selection) is a faithful port of the upstream
//! consensus-critical behaviour.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use cairo_vm::cairo_run::{cairo_run_program_with_initial_scope, CairoRunConfig};
use cairo_vm::serde::deserialize_program::Identifier;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::vm::runners::cairo_pie::CairoPie;
use cairo_vm::vm::runners::cairo_runner::CairoRunner;
use cairo_vm::Felt252;

pub use hints::*;

pub mod bootloaders;
mod hints;

/// Inserts the bootloader input in the execution scopes.
pub fn insert_bootloader_input(
    exec_scopes: &mut ExecutionScopes,
    bootloader_input: BootloaderInput,
) {
    exec_scopes.insert_value(BOOTLOADER_INPUT, bootloader_input);
}

/// Loads a Starknet Cairo PIE from a zip file and runs it wrapped in the
/// bootloader program, in **proof mode**, ready to be proven with the stwo
/// prover.
///
/// The PIE is loaded via [`CairoPie::read_zip_file`], wrapped as a single
/// `CairoPie` task inside a [`BootloaderInput`] (single-page fact topology),
/// inserted into the execution scopes via [`insert_bootloader_input`], and the
/// `bootloader-0.13.0` program (vendored in `resources/`) is executed with:
///   * `trace_enabled = true`
///   * `relocate_trace = false`
///   * `disable_trace_padding = true`
///   * `fill_holes = true`
///   * `proof_mode = true`
///   * `layout = all_cairo_stwo`
///
/// mirroring the `CairoRunConfig` used by `gpu_bench::run_vm`.
///
/// The returned [`CairoRunner`] can be passed directly to
/// `stwo_cairo_adapter::adapt(&runner)`.
pub fn run_pie_with_bootloader(pie_path: &Path) -> anyhow::Result<CairoRunner> {
    let pie = CairoPie::read_zip_file(pie_path)
        .with_context(|| format!("failed to read Cairo PIE zip: {}", pie_path.display()))?;

    run_cairo_pie_with_bootloader(pie)
}

/// Same as [`run_pie_with_bootloader`] but takes an already-loaded [`CairoPie`].
pub fn run_cairo_pie_with_bootloader(pie: CairoPie) -> anyhow::Result<CairoRunner> {
    let bootloader_program = bootloaders::load_bootloader()
        .map_err(|e| anyhow::anyhow!("failed to load bootloader program: {e}"))?;

    // A single CairoPie task. `single_page = false` mirrors the upstream
    // `run_program` example's fact topology; the bootloader emits two output
    // words per task plus the task's own pages.
    let tasks = vec![TaskSpec {
        use_poseidon: false,
        task: Task::Pie(pie),
    }];

    let bootloader_input = BootloaderInput {
        simple_bootloader_input: SimpleBootloaderInput {
            fact_topologies_path: None,
            single_page: false,
            tasks,
        },
        bootloader_config: BootloaderConfig {
            simple_bootloader_program_hash: Felt252::from(0),
            supported_cairo_verifier_program_hashes: vec![],
        },
        packed_outputs: vec![PackedOutput::Plain(vec![])],
    };

    let mut hint_processor = BootloaderHintProcessor::new();

    let cairo_run_config = CairoRunConfig {
        entrypoint: "main",
        trace_enabled: true,
        relocate_trace: false,
        relocate_mem: false,
        layout: LayoutName::all_cairo_stwo,
        proof_mode: true,
        secure_run: None,
        disable_trace_padding: true,
        fill_holes: true,
        allow_missing_builtins: None,
        ..Default::default()
    };

    let mut exec_scopes = ExecutionScopes::new();
    insert_bootloader_input(&mut exec_scopes, bootloader_input);

    // The `execute_task` CairoPie hint reads `bootloader_program_identifiers`
    // from the top-level execution scope to resolve the `ret_pc_label` /
    // `call_task` label offsets when loading a PIE task. Populate it from the
    // bootloader program's own identifiers (the upstream `run_program` example
    // only ran Program tasks, so it never needed this).
    let identifiers: HashMap<String, Identifier> = bootloader_program
        .iter_identifiers()
        .map(|(name, id)| (name.to_string(), id.clone()))
        .collect();
    exec_scopes.insert_value(BOOTLOADER_PROGRAM_IDENTIFIERS, identifiers);

    let runner = cairo_run_program_with_initial_scope(
        &bootloader_program,
        &cairo_run_config,
        &mut hint_processor,
        exec_scopes,
    )
    .map_err(|e| anyhow::anyhow!("bootloader VM run failed: {e}"))?;

    Ok(runner)
}
