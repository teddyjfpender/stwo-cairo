//! Source-bound authority for recorded computed deduces.
//!
//! Most deduces are self-contained device functions. The two Pedersen kinds
//! require module-local table pointers configured after module load and remain
//! outside the semantic prefix until that effect has a typed contract.

use stwo_backend_cuda::jit_witness::isa::{
    DeduceKind, DeduceModuleState, WitnessOp, WitnessProgram,
};

use super::InvocationShapeError;

const PEDERSEN_STATE_MARKER: &str = "#define STWO_WIT_NEEDS_PEDERSEN 1\n";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SelfContainedDeduceAuthority {
    pub(super) kinds: Vec<DeduceKind>,
    pub(super) source_identity: [u8; 32],
}

#[cfg(test)]
pub(super) fn empty_for_test() -> SelfContainedDeduceAuthority {
    SelfContainedDeduceAuthority {
        kinds: Vec::new(),
        source_identity: [8; 32],
    }
}

pub(super) fn bind(
    program: &WitnessProgram,
    emitted_source: &str,
) -> Result<SelfContainedDeduceAuthority, InvocationShapeError> {
    let mut kinds = Vec::new();
    let mut first_stateful = None;
    for instruction in &program.insts {
        if WitnessOp::from_raw(instruction.op) != Some(WitnessOp::DeduceCall) {
            continue;
        }
        let kind = DeduceKind::from_raw(instruction.imm)
            .ok_or(InvocationShapeError::InvalidProgramRole)?;
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
        match kind.module_state() {
            DeduceModuleState::SelfContained => {}
            DeduceModuleState::PedersenTableColumnsAndRowsV1 => {
                first_stateful.get_or_insert(kind);
            }
        }
    }

    let marker_count = emitted_source.matches(PEDERSEN_STATE_MARKER).count();
    if marker_count != usize::from(first_stateful.is_some()) {
        return Err(InvocationShapeError::SourceEmitterRejected);
    }
    if let Some(kind) = first_stateful {
        return Err(InvocationShapeError::UnsupportedModuleGlobals(kind));
    }
    let source_identity = stwo_backend_cuda::aot::emitted_source_identity(emitted_source);
    if source_identity == [0; 32] {
        return Err(InvocationShapeError::SourceEmitterRejected);
    }
    Ok(SelfContainedDeduceAuthority {
        kinds,
        source_identity,
    })
}
