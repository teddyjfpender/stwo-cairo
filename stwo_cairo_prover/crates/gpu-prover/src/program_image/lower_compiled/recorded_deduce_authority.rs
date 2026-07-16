//! Source-bound authority for recorded computed deduces.
//!
//! Most deduces are self-contained device functions. The two Pedersen kinds
//! additionally carry the exact logical table resource and the two relocations
//! every loaded module must publish before its first launch. This is semantic
//! authority only: it is not evidence that a particular `CUmodule` was loaded
//! or initialized.

use stwo_backend_cuda::jit_witness::isa::{
    DeduceKind, DeduceModuleState, WitnessOp, WitnessProgram,
};

use super::InvocationShapeError;
#[cfg(test)]
use crate::fixed_table_materializer::pedersen_points_18_column_index;
use crate::fixed_table_materializer::{
    PEDERSEN_POINTS_18_COLUMN_COUNT, PEDERSEN_POINTS_18_ROW_COUNT,
};

const PEDERSEN_STATE_MARKER: &str = "#define STWO_WIT_NEEDS_PEDERSEN 1\n";
const CUDA_DEVICE_POINTER_BYTES: u32 = 8;
const PEDERSEN_POINTS_18_UNPADDED_ROWS: u32 = (2 * (252 / 18)) << 18;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PedersenSemanticPaddingV1 {
    RepeatRowZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PedersenUploaderExtraPaddingV1 {
    ZeroFill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PedersenContentAuthorityV1 {
    /// The digest computed and checked by stwo-cairo's
    /// `try_ensure_device_pedersen_table` is required; this source recipe is
    /// intentionally insufficient as content authority.
    RequiredFromCanonicalRegistration,
}

/// Address-free source recipe for the only table admissible to W18 recorded
/// deduces. `source_recipe` is a semantic name, not a content digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PedersenTableResourceV1 {
    pub(super) source_recipe: &'static str,
    pub(super) column_identity_prefix: &'static str,
    pub(super) first_column: u32,
    pub(super) columns: u32,
    pub(super) coordinate_limbs: u32,
    pub(super) semantic_real_rows: u32,
    pub(super) registered_source_rows: u32,
    pub(super) registered_padded_rows: u32,
    pub(super) semantic_padding: PedersenSemanticPaddingV1,
    pub(super) uploader_extra_padding: PedersenUploaderExtraPaddingV1,
    pub(super) uploader_extra_rows: u32,
    pub(super) element_bytes: u32,
    pub(super) content_authority: PedersenContentAuthorityV1,
}

/// Relocate the registered table's 56 device column addresses into one module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PedersenColumnPointersRelocationV1 {
    pub(super) symbol: &'static str,
    pub(super) symbol_bytes: u32,
    pub(super) alignment_bytes: u32,
    pub(super) pointer_bytes: u32,
    pub(super) entries: u32,
}

/// Publish the exact padded row count used by both device-side masking sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PedersenRowCountRelocationV1 {
    pub(super) symbol: &'static str,
    pub(super) symbol_bytes: u32,
    pub(super) alignment_bytes: u32,
    pub(super) value: u32,
}

/// Deterministic resource/relocation recipe. A later loaded-module authority
/// must bind this recipe to the registered table digest, addresses, generation,
/// and a completed publication event; this value deliberately contains none of
/// those process-local facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PedersenTableColumnsAndRowsV1 {
    pub(super) resource: PedersenTableResourceV1,
    pub(super) column_pointers: PedersenColumnPointersRelocationV1,
    pub(super) row_count: PedersenRowCountRelocationV1,
}

impl PedersenTableColumnsAndRowsV1 {
    pub(super) const CANONICAL: Self = Self {
        resource: PedersenTableResourceV1 {
            source_recipe: "stwo-cairo.pedersen-points.window-bits-18.host.repeat-row-zero.v1",
            column_identity_prefix: "pedersen_points_",
            first_column: 0,
            columns: PEDERSEN_POINTS_18_COLUMN_COUNT as u32,
            coordinate_limbs: 28,
            semantic_real_rows: PEDERSEN_POINTS_18_UNPADDED_ROWS,
            // Cairo's source builder repeats row zero up to 2^23 before the
            // backend uploader sees the table. Its generic zero-fill resize is
            // therefore a no-op for this canonical resource.
            registered_source_rows: PEDERSEN_POINTS_18_ROW_COUNT as u32,
            registered_padded_rows: PEDERSEN_POINTS_18_ROW_COUNT as u32,
            semantic_padding: PedersenSemanticPaddingV1::RepeatRowZero,
            uploader_extra_padding: PedersenUploaderExtraPaddingV1::ZeroFill,
            uploader_extra_rows: 0,
            element_bytes: core::mem::size_of::<u32>() as u32,
            content_authority: PedersenContentAuthorityV1::RequiredFromCanonicalRegistration,
        },
        column_pointers: PedersenColumnPointersRelocationV1 {
            symbol: "g_stwo_wit_pedersen_cols",
            symbol_bytes: PEDERSEN_POINTS_18_COLUMN_COUNT as u32 * CUDA_DEVICE_POINTER_BYTES,
            alignment_bytes: CUDA_DEVICE_POINTER_BYTES,
            pointer_bytes: CUDA_DEVICE_POINTER_BYTES,
            entries: PEDERSEN_POINTS_18_COLUMN_COUNT as u32,
        },
        row_count: PedersenRowCountRelocationV1 {
            symbol: "g_stwo_wit_pedersen_n_rows",
            symbol_bytes: core::mem::size_of::<u32>() as u32,
            alignment_bytes: core::mem::align_of::<u32>() as u32,
            value: PEDERSEN_POINTS_18_ROW_COUNT as u32,
        },
    };

    pub(super) fn validate_exact(self) -> Result<(), InvocationShapeError> {
        if self != Self::CANONICAL
            || self.resource.semantic_real_rows.next_power_of_two()
                != self.resource.registered_source_rows
            || self.resource.registered_source_rows != self.resource.registered_padded_rows
            || self.resource.uploader_extra_rows != 0
        {
            return Err(InvocationShapeError::InvalidProgramRole);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RecordedDeduceAuthority {
    pub(super) kinds: Vec<DeduceKind>,
    pub(super) source_identity: [u8; 32],
    /// Private source/relocation frontier only. It must not be converted into a
    /// module-global `EffectContract` until a loaded-module publication receipt
    /// binds the canonical registration's digest and process-local addresses.
    pub(super) module_state: Option<PedersenTableColumnsAndRowsV1>,
}

#[cfg(test)]
pub(super) fn empty_for_test() -> RecordedDeduceAuthority {
    RecordedDeduceAuthority {
        kinds: Vec::new(),
        source_identity: [8; 32],
        module_state: None,
    }
}

pub(super) fn bind(
    program: &WitnessProgram,
    emitted_source: &str,
) -> Result<RecordedDeduceAuthority, InvocationShapeError> {
    let mut kinds = Vec::new();
    let mut needs_pedersen_table = false;
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
                needs_pedersen_table = true;
            }
        }
    }

    let marker_count = emitted_source.matches(PEDERSEN_STATE_MARKER).count();
    if marker_count != usize::from(needs_pedersen_table) {
        return Err(InvocationShapeError::SourceEmitterRejected);
    }
    let module_state = needs_pedersen_table.then_some(PedersenTableColumnsAndRowsV1::CANONICAL);
    if let Some(state) = module_state {
        state.validate_exact()?;
    }
    let source_identity = stwo_backend_cuda::aot::emitted_source_identity(emitted_source);
    if source_identity == [0; 32] {
        return Err(InvocationShapeError::SourceEmitterRejected);
    }
    Ok(RecordedDeduceAuthority {
        kinds,
        source_identity,
        module_state,
    })
}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::jit_witness::isa::{WitnessInst, WitnessProgram};
    use stwo_cairo_common::preprocessed_columns::pedersen::PedersenPoints;
    use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedColumn;

    use super::*;

    fn program(kinds: &[DeduceKind]) -> WitnessProgram {
        WitnessProgram {
            label: "deduce-state-contract-test".into(),
            insts: kinds
                .iter()
                .enumerate()
                .map(|(index, kind)| {
                    WitnessInst::new(
                        WitnessOp::DeduceCall,
                        index as u16,
                        0,
                        kind.shape().1 as u32,
                        *kind as u32,
                    )
                })
                .collect(),
            n_regs: 72,
            n_inputs: 0,
            n_cols: 0,
            n_mult_tables: 0,
            n_lookup_words: 0,
            n_sub_words: 0,
        }
    }

    #[test]
    fn both_pedersen_deduce_kinds_bind_the_same_exact_resource() {
        for kind in [
            DeduceKind::PartialEcMulW18,
            DeduceKind::PedersenPointsTableW18,
        ] {
            let authority = bind(&program(&[kind]), PEDERSEN_STATE_MARKER).unwrap();
            assert_eq!(authority.kinds, vec![kind]);
            assert_eq!(
                authority.module_state,
                Some(PedersenTableColumnsAndRowsV1::CANONICAL)
            );
        }
        let authority = bind(
            &program(&[
                DeduceKind::PartialEcMulW18,
                DeduceKind::PedersenPointsTableW18,
            ]),
            PEDERSEN_STATE_MARKER,
        )
        .unwrap();
        let resource = authority.module_state.unwrap().resource;
        assert_eq!(resource.semantic_real_rows, 7_340_032);
        assert_eq!(resource.registered_source_rows, 1 << 23);
        assert_eq!(resource.registered_padded_rows, 1 << 23);
        assert_eq!(resource.uploader_extra_rows, 0);
        assert_eq!(
            resource.content_authority,
            PedersenContentAuthorityV1::RequiredFromCanonicalRegistration
        );
    }

    #[test]
    fn source_recipe_geometry_and_order_match_the_canonical_column_contract() {
        let resource = PedersenTableColumnsAndRowsV1::CANONICAL.resource;
        // `log_size` computes the canonical source recipe without forcing the
        // roughly 1.9 GiB lazy table into host memory.
        assert_eq!(PedersenPoints::<18>::new(0).log_size(), 23);
        assert_eq!(
            resource.semantic_real_rows.next_power_of_two(),
            resource.registered_source_rows
        );
        assert_eq!(
            resource.registered_source_rows,
            PEDERSEN_POINTS_18_ROW_COUNT as u32
        );
        assert!((0..resource.columns).all(|offset| {
            let index = resource.first_column + offset;
            pedersen_points_18_column_index(&format!(
                "{}{}",
                resource.column_identity_prefix, index
            )) == Some(index as usize)
        }));
    }

    #[test]
    fn source_marker_must_match_the_exhaustive_module_state() {
        assert_eq!(
            bind(&program(&[DeduceKind::PartialEcMulW18]), ""),
            Err(InvocationShapeError::SourceEmitterRejected)
        );
        assert_eq!(
            bind(
                &program(&[DeduceKind::PedersenPointsTableW18]),
                &PEDERSEN_STATE_MARKER.repeat(2),
            ),
            Err(InvocationShapeError::SourceEmitterRejected)
        );
        assert_eq!(
            bind(&program(&[DeduceKind::FeltMul]), PEDERSEN_STATE_MARKER),
            Err(InvocationShapeError::SourceEmitterRejected)
        );
        assert!(bind(&program(&[DeduceKind::FeltMul]), "self-contained").is_ok());
    }

    #[test]
    fn resource_and_relocation_recipe_reject_every_tested_drift() {
        let mutations: &[fn(&mut PedersenTableColumnsAndRowsV1)] = &[
            |state| state.resource.source_recipe = "foreign-recipe",
            |state| state.resource.column_identity_prefix = "pedersen_points_small_",
            |state| state.resource.first_column = 1,
            |state| state.resource.columns -= 1,
            |state| state.resource.coordinate_limbs -= 1,
            |state| state.resource.semantic_real_rows -= 1,
            |state| state.resource.registered_source_rows -= 1,
            |state| state.resource.registered_padded_rows -= 1,
            |state| state.resource.uploader_extra_rows = 1,
            |state| state.resource.element_bytes *= 2,
            |state| state.column_pointers.symbol = "g_stwo_wit_pedersen_n_rows",
            |state| state.column_pointers.symbol_bytes -= CUDA_DEVICE_POINTER_BYTES,
            |state| state.column_pointers.alignment_bytes /= 2,
            |state| state.column_pointers.pointer_bytes /= 2,
            |state| state.column_pointers.entries -= 1,
            |state| state.row_count.symbol = "g_stwo_wit_pedersen_cols",
            |state| state.row_count.symbol_bytes *= 2,
            |state| state.row_count.alignment_bytes *= 2,
            |state| state.row_count.value -= 1,
        ];
        for mutate in mutations {
            let mut state = PedersenTableColumnsAndRowsV1::CANONICAL;
            mutate(&mut state);
            assert_eq!(
                state.validate_exact(),
                Err(InvocationShapeError::InvalidProgramRole)
            );
        }
    }
}
