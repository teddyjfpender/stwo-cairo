use std::collections::HashMap;
use std::path::PathBuf;

use cairo_vm::serde::deserialize_program::Identifier;
use cairo_vm::types::errors::program_errors::ProgramError;
use cairo_vm::types::program::Program;
use cairo_vm::vm::runners::cairo_pie::{CairoPie, StrippedProgram};
use cairo_vm::Felt252;
use serde::Deserialize;

pub type BootloaderVersion = u64;

pub(crate) type ProgramIdentifiers = HashMap<String, Identifier>;

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct BootloaderConfig {
    pub simple_bootloader_program_hash: Felt252,
    pub supported_cairo_verifier_program_hashes: Vec<Felt252>,
}

#[derive(Deserialize, Debug, Default, Clone, PartialEq)]
pub struct CompositePackedOutput {
    pub outputs: Vec<Felt252>,
    pub subtasks: Vec<PackedOutput>,
}

impl CompositePackedOutput {
    pub fn elements_for_hash(&self) -> &Vec<Felt252> {
        &self.outputs
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub enum PackedOutput {
    Plain(Vec<Felt252>),
    Composite(CompositePackedOutput),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Task {
    Program(Program),
    Pie(CairoPie),
}

impl Task {
    pub fn get_program(&self) -> Result<StrippedProgram, ProgramError> {
        // TODO: consider whether declaring a struct similar to StrippedProgram
        //       but that uses a reference to data to avoid copying is worth the effort.
        match self {
            Task::Program(program) => program.get_stripped_program(),
            Task::Pie(cairo_pie) => Ok(cairo_pie.metadata.program.clone()),
        }
    }
}

/// A program-hash function. In the v0.14 simple bootloader, `Task.use_poseidon`
/// (a bool) was replaced by `Task.program_hash_function`, an integer selecting
/// the hash used to compute the program hash in `compute_program_hash`
/// (execute_task.cairo): `PEDERSEN_HASH = 0`, `POSEIDON_HASH = 1`,
/// `BLAKE_HASH = 2`. The `tempvar program_hash_function = nondet
/// %{ task.program_hash_function %}` hint pushes this integer to the AP.
///
/// For the pedersen-hashed Starknet PIEs we run, the value is `Pedersen` (0).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashFunc {
    Pedersen = 0,
    Poseidon = 1,
    Blake = 2,
}

impl Default for HashFunc {
    fn default() -> Self {
        HashFunc::Pedersen
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskSpec {
    pub task: Task,
    /// Mirrors the python Task.use_poseidon flag (0.13.3 bootloader): selects
    /// the poseidon program-hash chain instead of pedersen. PIE proving uses
    /// the pedersen chain (false). Retained for the existing full-bootloader
    /// (0.13.3) path.
    pub use_poseidon: bool,
    /// The v0.14 simple bootloader program-hash function selector. For PIE
    /// proving this is `HashFunc::Pedersen` (0).
    pub program_hash_function: HashFunc,
}

impl TaskSpec {
    pub fn load_task(&self) -> &Task {
        &self.task
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleBootloaderInput {
    pub fact_topologies_path: Option<PathBuf>,
    pub single_page: bool,
    pub tasks: Vec<TaskSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootloaderInput {
    pub simple_bootloader_input: SimpleBootloaderInput,
    pub bootloader_config: BootloaderConfig,
    pub packed_outputs: Vec<PackedOutput>,
}
