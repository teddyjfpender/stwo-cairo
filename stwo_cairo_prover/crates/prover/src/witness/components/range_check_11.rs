// This file was created by the AIR team.

#![allow(unused_parens)]
use cairo_air::components::range_check_11::{Claim, InteractionClaim, LOG_SIZE, N_TRACE_COLUMNS};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{RawLogupTrace, RawLogupTraceGenerator};

use crate::witness::prelude::*;

pub type InputType = [M31; 1];
pub type PackedInputType = [PackedM31; 1];

pub struct ClaimGenerator {
    pub mults: [AtomicMultiplicityColumn; 1],
    preprocessed_trace: Arc<PreProcessedTrace>,
}

impl ClaimGenerator {
    pub fn new(preprocessed_trace: Arc<PreProcessedTrace>) -> Self {
        let mults = from_fn(|_| AtomicMultiplicityColumn::new(1 << LOG_SIZE));
        Self {
            mults,
            preprocessed_trace,
        }
    }

    /// Merge relation-indexed device count tables (layout `counts[relation_index
    /// * table_size + row]`, row = the value/row key `add_input` uses directly)
    /// into the multiplicity columns. Zero counts skipped; `add_at` merges are
    /// identical to the same number of `increase_at` calls — the device-DAG
    /// count-feed consumer surface (see `witness/device_feed.rs`).
    pub fn add_count_tables(&self, counts: &[u32]) {
        let table_size = counts.len() / 1;
        assert_eq!(counts.len(), 1 * table_size);
        for (relation_index, table) in counts.chunks_exact(table_size).enumerate() {
            for (row, &count) in table.iter().enumerate() {
                if count != 0 {
                    self.mults[relation_index].add_at(row as u32, count);
                }
            }
        }
    }

    pub fn write_trace(
        self,
    ) -> (
        ComponentTrace<N_TRACE_COLUMNS>,
        Claim,
        InteractionClaimGenerator,
    ) {
        let mults = self
            .mults
            .into_iter()
            .map(|v| v.into_simd_vec())
            .collect::<Vec<_>>();

        let (trace, lookup_data) = write_trace_simd(&self.preprocessed_trace, mults);

        (trace, Claim {}, InteractionClaimGenerator { lookup_data })
    }
}

impl AddInputs for ClaimGenerator {
    type PackedInputType = PackedInputType;
    type InputType = InputType;

    fn add_packed_inputs(&self, packed_inputs: &[PackedInputType], relation_index: usize) {
        packed_inputs.into_par_iter().for_each(|packed_input| {
            for [idx] in packed_input.unpack() {
                self.mults[relation_index].increase_at(idx.0);
            }
        });
    }
    fn add_input(&self, input: &InputType, relation_index: usize) {
        self.mults[relation_index].increase_at(input[0].0);
    }
}

#[allow(clippy::useless_conversion)]
#[allow(unused_variables)]
#[allow(clippy::double_parens)]
#[allow(non_snake_case)]
fn write_trace_simd(
    preprocessed_trace: &PreProcessedTrace,
    mults: Vec<Vec<PackedM31>>,
) -> (ComponentTrace<N_TRACE_COLUMNS>, LookupData) {
    let log_n_packed_rows = LOG_SIZE - LOG_N_LANES;
    let (mut trace, mut lookup_data) = unsafe {
        (
            ComponentTrace::<N_TRACE_COLUMNS>::uninitialized(LOG_SIZE),
            LookupData::uninitialized(log_n_packed_rows),
        )
    };

    let M31_991608089 = PackedM31::broadcast(M31::from(991608089));
    let seq_11 = preprocessed_trace.get_column(&PreProcessedColumnId {
        id: "seq_11".to_owned(),
    });

    (trace.par_iter_mut(), lookup_data.par_iter_mut())
        .into_par_iter()
        .enumerate()
        .for_each(|(row_index, (row, lookup_data))| {
            let seq_11 = seq_11.packed_at(row_index);
            let multiplicity_0_col0 = *mults[0].get(row_index).unwrap_or(&PackedM31::zero());
            *row[0] = multiplicity_0_col0;
            *lookup_data.range_check_11_0 = [M31_991608089, seq_11];
            *lookup_data.mults_0 = multiplicity_0_col0;
        });

    (trace, lookup_data)
}

#[derive(Uninitialized, IterMut, ParIterMut)]
struct LookupData {
    range_check_11_0: Vec<[PackedM31; 2]>,
    mults_0: Vec<PackedM31>,
}

pub struct InteractionClaimGenerator {
    lookup_data: LookupData,
}
// === BEGIN relation_lookup_source_codegen ===
crate::relation_lookup_source! {
    range_check_11_0: 2,
    mults_0: scalar,
}
// === END relation_lookup_source_codegen ===
impl InteractionClaimGenerator {
    pub fn write_interaction_trace(
        self,
        common_lookup_elements: &relations::CommonLookupElements,
    ) -> (RawLogupTrace, impl FnOnce(SecureField) -> InteractionClaim) {
        let mut logup_gen = unsafe { RawLogupTraceGenerator::uninitialized(LOG_SIZE) };

        // Sum last logup term.
        let mut col_gen = logup_gen.new_col();
        (
            col_gen.par_iter_mut(),
            &self.lookup_data.range_check_11_0,
            self.lookup_data.mults_0,
        )
            .into_par_iter()
            .for_each(|(writer, values, mult)| {
                let denom = common_lookup_elements.combine(values);
                writer.write_frac((-mult).into(), denom);
            });
        col_gen.finalize_col();

        (logup_gen.into_raw(), |claimed_sum| InteractionClaim {
            claimed_sum,
        })
    }
}
