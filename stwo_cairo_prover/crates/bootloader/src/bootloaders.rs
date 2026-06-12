use cairo_vm::types::errors::program_errors::ProgramError;
use cairo_vm::types::program::Program;

pub use crate::hints::*;

const BOOTLOADER_V0_13_3: &[u8] = include_bytes!("../resources/bootloader-0.13.3.json");

const SIMPLE_BOOTLOADER_V0_14_0: &[u8] =
    include_bytes!("../resources/simple_bootloader-v0.14.0.json");

/// Loads the bootloader and returns it as a Cairo VM `Program` object.
pub fn load_bootloader() -> Result<Program, ProgramError> {
    Program::from_bytes(BOOTLOADER_V0_13_3, Some("main"))
}

/// Loads the cairo-lang v0.14.0 *simple* bootloader program and returns it as a
/// Cairo VM `Program` object. This is the `simple_bootloader.cairo` `main`,
/// which executes the tasks directly (no outer packed-output wrapping) and, for
/// the `all_cairo_stwo` layout, simulates the missing ecdsa/keccak/ec_op
/// builtins entirely in Cairo (`verify_builtins.cairo`).
pub fn load_simple_bootloader() -> Result<Program, ProgramError> {
    Program::from_bytes(SIMPLE_BOOTLOADER_V0_14_0, Some("main"))
}
