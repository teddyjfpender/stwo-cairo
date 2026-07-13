//! Read-only bridge from the untouched production SIMD writer to the oracle package.

use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuLabOracleRow {
    pub columns: [u32; 3],
    pub lookup_words: [u32; 14],
    pub sub_words: [u32; 6],
}

pub fn gpu_lab_visit_simd_rows<F>(
    log_size: u32,
    pedersen_builtin_segment_start: u32,
    memory_address_to_id_state: &memory_address_to_id::ClaimGenerator,
    mut visit: F,
) -> Result<(), String>
where
    F: FnMut(usize, GpuLabOracleRow) -> Result<(), String>,
{
    let aggregator_state = pedersen_aggregator_window_bits_18::ClaimGenerator::new();
    let (trace, lookup, sub) = write_trace_simd(
        log_size,
        pedersen_builtin_segment_start,
        memory_address_to_id_state,
        &aggregator_state,
    );
    let rows = 1usize
        .checked_shl(log_size)
        .ok_or_else(|| format!("SIMD oracle log_size {log_size} exceeds host usize"))?;
    for row in 0..rows {
        let packed = row / N_LANES;
        let lane = row % N_LANES;
        let word = |value: PackedM31| value.to_array()[lane].0;
        let columns = trace.row_at(row).map(|value| value.0);
        let lookup_words = [
            word(lookup.memory_address_to_id_0[packed][0]),
            word(lookup.memory_address_to_id_0[packed][1]),
            word(lookup.memory_address_to_id_0[packed][2]),
            word(lookup.memory_address_to_id_1[packed][0]),
            word(lookup.memory_address_to_id_1[packed][1]),
            word(lookup.memory_address_to_id_1[packed][2]),
            word(lookup.memory_address_to_id_2[packed][0]),
            word(lookup.memory_address_to_id_2[packed][1]),
            word(lookup.memory_address_to_id_2[packed][2]),
            word(lookup.pedersen_aggregator_window_bits_18_3[packed][0]),
            word(lookup.pedersen_aggregator_window_bits_18_3[packed][1]),
            word(lookup.pedersen_aggregator_window_bits_18_3[packed][2]),
            word(lookup.pedersen_aggregator_window_bits_18_3[packed][3]),
            word(lookup.mults_0[packed]),
        ];
        let aggregator = &sub.pedersen_aggregator_window_bits_18[0][packed];
        let sub_words = [
            word(sub.memory_address_to_id[0][packed]),
            word(sub.memory_address_to_id[1][packed]),
            word(sub.memory_address_to_id[2][packed]),
            word(aggregator.0[0]),
            word(aggregator.0[1]),
            word(aggregator.1),
        ];
        visit(
            row,
            GpuLabOracleRow {
                columns,
                lookup_words,
                sub_words,
            },
        )?;
    }
    Ok(())
}
