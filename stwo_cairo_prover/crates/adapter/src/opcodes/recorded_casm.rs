use stwo_cairo_common::prover_types::cpu::CasmState;

use super::CasmStatesByOpcode;

/// Stable identity for every opcode source consumed by ReplacementV1's
/// row-major Casm ingress. The enum makes downstream routing exhaustive while
/// [`RECORDED_CASM_DESCRIPTORS`] remains the single label/source/geometry
/// registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecordedCasmKind {
    Add,
    AssertEq,
    JnzTaken,
    AddSmall,
    AssertEqImm,
    AssertEqDoubleDeref,
    CallAbs,
    CallRelImm,
    JnzNonTaken,
    JumpAbs,
    JumpDoubleDeref,
    JumpRel,
    JumpRelImm,
    Ret,
    AddAp,
    Mul,
    MulSmall,
    BlakeCompress,
    Qm31AddMul,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecordedCasmDescriptor {
    pub kind: RecordedCasmKind,
    pub label: &'static str,
    pub include_iota: bool,
}

impl RecordedCasmDescriptor {
    /// Borrows the exact adapter-owned row-major source selected by this
    /// descriptor. No column materialization or generator is involved.
    pub fn states<'a>(&self, sources: &'a CasmStatesByOpcode) -> &'a [CasmState] {
        match self.kind {
            RecordedCasmKind::Add => &sources.add_opcode,
            RecordedCasmKind::AssertEq => &sources.assert_eq_opcode,
            RecordedCasmKind::JnzTaken => &sources.jnz_opcode_taken,
            RecordedCasmKind::AddSmall => &sources.add_opcode_small,
            RecordedCasmKind::AssertEqImm => &sources.assert_eq_opcode_imm,
            RecordedCasmKind::AssertEqDoubleDeref => &sources.assert_eq_opcode_double_deref,
            RecordedCasmKind::CallAbs => &sources.call_opcode_abs,
            RecordedCasmKind::CallRelImm => &sources.call_opcode_rel_imm,
            RecordedCasmKind::JnzNonTaken => &sources.jnz_opcode_non_taken,
            RecordedCasmKind::JumpAbs => &sources.jump_opcode_abs,
            RecordedCasmKind::JumpDoubleDeref => &sources.jump_opcode_double_deref,
            RecordedCasmKind::JumpRel => &sources.jump_opcode_rel,
            RecordedCasmKind::JumpRelImm => &sources.jump_opcode_rel_imm,
            RecordedCasmKind::Ret => &sources.ret_opcode,
            RecordedCasmKind::AddAp => &sources.add_ap_opcode,
            RecordedCasmKind::Mul => &sources.mul_opcode,
            RecordedCasmKind::MulSmall => &sources.mul_opcode_small,
            RecordedCasmKind::BlakeCompress => &sources.blake_compress_opcode,
            RecordedCasmKind::Qm31AddMul => &sources.qm_31_add_mul_opcode,
        }
    }
}

/// Canonical dispatch order for the 19 recorded row-major Casm sources. Keep
/// downstream iteration in this order so coverage/order drift fails locally.
pub const RECORDED_CASM_DESCRIPTORS: [RecordedCasmDescriptor; 19] = [
    descriptor(RecordedCasmKind::Add, "add_opcode", false),
    descriptor(RecordedCasmKind::AssertEq, "assert_eq_opcode", false),
    descriptor(RecordedCasmKind::JnzTaken, "jnz_opcode_taken", false),
    descriptor(RecordedCasmKind::AddSmall, "add_opcode_small", false),
    descriptor(RecordedCasmKind::AssertEqImm, "assert_eq_opcode_imm", false),
    descriptor(
        RecordedCasmKind::AssertEqDoubleDeref,
        "assert_eq_opcode_double_deref",
        false,
    ),
    descriptor(RecordedCasmKind::CallAbs, "call_opcode_abs", false),
    descriptor(RecordedCasmKind::CallRelImm, "call_opcode_rel_imm", false),
    descriptor(RecordedCasmKind::JnzNonTaken, "jnz_opcode_non_taken", false),
    descriptor(RecordedCasmKind::JumpAbs, "jump_opcode_abs", false),
    descriptor(
        RecordedCasmKind::JumpDoubleDeref,
        "jump_opcode_double_deref",
        false,
    ),
    descriptor(RecordedCasmKind::JumpRel, "jump_opcode_rel", false),
    descriptor(RecordedCasmKind::JumpRelImm, "jump_opcode_rel_imm", false),
    descriptor(RecordedCasmKind::Ret, "ret_opcode", false),
    descriptor(RecordedCasmKind::AddAp, "add_ap_opcode", false),
    descriptor(RecordedCasmKind::Mul, "mul_opcode", false),
    descriptor(RecordedCasmKind::MulSmall, "mul_opcode_small", false),
    descriptor(
        RecordedCasmKind::BlakeCompress,
        "blake_compress_opcode",
        true,
    ),
    descriptor(RecordedCasmKind::Qm31AddMul, "qm_31_add_mul_opcode", false),
];

const fn descriptor(
    kind: RecordedCasmKind,
    label: &'static str,
    include_iota: bool,
) -> RecordedCasmDescriptor {
    RecordedCasmDescriptor {
        kind,
        label,
        include_iota,
    }
}

pub fn recorded_casm_descriptor(label: &str) -> Option<&'static RecordedCasmDescriptor> {
    RECORDED_CASM_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.label == label)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use stwo::core::fields::m31::M31;

    use super::*;

    fn state(pc: u32) -> CasmState {
        CasmState {
            pc: M31(pc),
            ap: M31(pc + 100),
            fp: M31(pc + 200),
        }
    }

    #[test]
    fn registry_is_unique_ordered_and_selects_all_19_sources() {
        let sources = CasmStatesByOpcode {
            add_opcode: vec![state(1)],
            assert_eq_opcode: vec![state(2)],
            jnz_opcode_taken: vec![state(3)],
            add_opcode_small: vec![state(4)],
            assert_eq_opcode_imm: vec![state(5)],
            assert_eq_opcode_double_deref: vec![state(6)],
            call_opcode_abs: vec![state(7)],
            call_opcode_rel_imm: vec![state(8)],
            jnz_opcode_non_taken: vec![state(9)],
            jump_opcode_abs: vec![state(10)],
            jump_opcode_double_deref: vec![state(11)],
            jump_opcode_rel: vec![state(12)],
            jump_opcode_rel_imm: vec![state(13)],
            ret_opcode: vec![state(14)],
            add_ap_opcode: vec![state(15)],
            mul_opcode: vec![state(16)],
            mul_opcode_small: vec![state(17)],
            blake_compress_opcode: vec![state(18)],
            qm_31_add_mul_opcode: vec![state(19)],
            ..Default::default()
        };
        let expected_labels = [
            "add_opcode",
            "assert_eq_opcode",
            "jnz_opcode_taken",
            "add_opcode_small",
            "assert_eq_opcode_imm",
            "assert_eq_opcode_double_deref",
            "call_opcode_abs",
            "call_opcode_rel_imm",
            "jnz_opcode_non_taken",
            "jump_opcode_abs",
            "jump_opcode_double_deref",
            "jump_opcode_rel",
            "jump_opcode_rel_imm",
            "ret_opcode",
            "add_ap_opcode",
            "mul_opcode",
            "mul_opcode_small",
            "blake_compress_opcode",
            "qm_31_add_mul_opcode",
        ];

        assert_eq!(
            RECORDED_CASM_DESCRIPTORS.map(|descriptor| descriptor.label),
            expected_labels
        );
        assert_eq!(
            RECORDED_CASM_DESCRIPTORS
                .iter()
                .map(|descriptor| descriptor.label)
                .collect::<HashSet<_>>()
                .len(),
            19
        );
        for (index, descriptor) in RECORDED_CASM_DESCRIPTORS.iter().enumerate() {
            assert_eq!(descriptor.states(&sources), &[state(index as u32 + 1)]);
            assert_eq!(recorded_casm_descriptor(descriptor.label), Some(descriptor));
            assert_eq!(
                descriptor.include_iota,
                descriptor.label == "blake_compress_opcode"
            );
        }
        assert!(recorded_casm_descriptor("generic_opcode").is_none());
    }
}
