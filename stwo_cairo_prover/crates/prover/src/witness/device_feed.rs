//! Device-DAG count feeds (GPU-resident program, B2): consume a witness
//! kernel's DEVICE-resident sub-input buffer with `witness_feed_counts.cu`
//! instead of D2H → host packed rebuild → DashMap `add_input` storms.
//!
//! Scope: COUNT-STYLE relations only — consumers whose feed is a multiplicity
//! increment keyed by the tuple's preprocessed row (`range_check_*`,
//! `verify_bitwise_xor_*`, `pedersen_points_table_*`). Input-list consumers
//! (verify_instruction, component-to-component instance feeds like
//! aggregator→w18) keep the host path until the B3 device edges land.
//!
//! Equivalence: the same multiset of increments the host `add_input` calls
//! produce, merged commutatively into the consumer's `AtomicMultiplicityColumn`
//! via `add_count_tables` — the certified blake_g count-feed argument.
//!
//! Driven entirely by the transformer-emitted `SUB_FEED_LAYOUT` facts:
//! `(field, instance, state param, relation_index, word_base, words/instance)`.

/// One count-style relation family the driver knows how to feed: how to key it
/// (per-word bit widths folded MSB-first, exactly the host tuple order) and
/// whether the key needs the consumer's `input_to_row` LUT (the actual
/// preprocessed layout) or is identity-keyed (points table: key = row).
#[derive(Clone, Debug)]
pub struct CountRelation {
    /// The downstream state param name (`range_check_9_9_state`, …) — the join
    /// key against `SUB_FEED_LAYOUT`.
    pub state_param: &'static str,
    /// Per-word fold widths (bits), tuple order. `key = ((key << b) | word)`.
    pub word_bits: &'static [u32],
    /// Consumer preprocessed size (rows per relation slot) = `1 << sum(bits)`
    /// unless the consumer pads differently.
    pub table_size: usize,
    /// Number of relation slots (the consumer's `mults` array length).
    pub n_relations: usize,
    /// Whether keys map through the consumer's `input_to_row` LUT.
    pub needs_lut: bool,
}

/// Descriptor stride of `witness_feed_counts.cu` (flat u32 ABI).
pub const WFC_DESC_STRIDE: usize = 11;
pub const WFC_NO_LUT: u32 = u32::MAX;

/// Build the flat device-kernel descriptors for one component's layout,
/// returning `(descs, lut_slots, counts_slots)` where the slots name the
/// relation families in first-use order (the caller uploads one LUT and one
/// zeroed count buffer per named family and passes the pointer arrays in that
/// order). Entries whose state param is not a known count relation are skipped
/// (they stay on the host feed path).
pub fn build_feed_descriptors(
    layout: &[(&str, usize, &str, u32, usize, usize)],
    relations: &[CountRelation],
) -> (Vec<u32>, Vec<&'static str>, Vec<&'static str>) {
    let mut descs: Vec<u32> = Vec::new();
    let mut lut_slots: Vec<&'static str> = Vec::new();
    let mut counts_slots: Vec<&'static str> = Vec::new();
    for &(_field, _instance, state, rel_index, base, words) in layout {
        let Some(rel) = relations.iter().find(|r| r.state_param == state) else {
            continue; // input-list or host-fed relation
        };
        assert_eq!(
            words,
            rel.word_bits.len(),
            "SUB_FEED_LAYOUT width disagrees with the relation family for {state}"
        );
        assert!(rel.word_bits.len() <= 5, "feed kernel key fold cap");
        assert!(
            (rel_index as usize) < rel.n_relations,
            "relation_index {rel_index} out of range for {state}"
        );
        let counts_index = counts_slots
            .iter()
            .position(|s| *s == rel.state_param)
            .unwrap_or_else(|| {
                counts_slots.push(rel.state_param);
                counts_slots.len() - 1
            }) as u32;
        let lut_index = if rel.needs_lut {
            lut_slots
                .iter()
                .position(|s| *s == rel.state_param)
                .unwrap_or_else(|| {
                    lut_slots.push(rel.state_param);
                    lut_slots.len() - 1
                }) as u32
        } else {
            WFC_NO_LUT
        };
        let mut e = [0u32; WFC_DESC_STRIDE];
        e[0] = base as u32;
        e[1] = words as u32;
        for (i, b) in rel.word_bits.iter().enumerate() {
            e[2 + i] = *b;
        }
        e[7] = rel_index;
        e[8] = rel.table_size as u32;
        e[9] = lut_index;
        e[10] = counts_index;
        descs.extend_from_slice(&e);
    }
    (descs, lut_slots, counts_slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RC99: CountRelation = CountRelation {
        state_param: "range_check_9_9_state",
        word_bits: &[9, 9],
        table_size: 1 << 18,
        n_relations: 8,
        needs_lut: true,
    };
    const PTS: CountRelation = CountRelation {
        state_param: "pedersen_points_table_window_bits_18_state",
        word_bits: &[23],
        table_size: 1 << 23,
        n_relations: 1,
        needs_lut: false,
    };

    #[test]
    fn descriptors_follow_layout_and_share_slots() {
        let layout: &[(&str, usize, &str, u32, usize, usize)] = &[
            (
                "pts",
                0,
                "pedersen_points_table_window_bits_18_state",
                0,
                0,
                1,
            ),
            ("rc", 0, "range_check_9_9_state", 0, 1, 2),
            ("rc_b", 0, "range_check_9_9_state", 3, 3, 2),
            ("vi", 0, "verify_instruction_state", 0, 5, 7), // host-fed: skipped
        ];
        let (descs, luts, counts) = build_feed_descriptors(layout, &[RC99, PTS]);
        assert_eq!(descs.len(), 3 * WFC_DESC_STRIDE);
        assert_eq!(
            counts,
            vec![
                "pedersen_points_table_window_bits_18_state",
                "range_check_9_9_state"
            ]
        );
        assert_eq!(luts, vec!["range_check_9_9_state"]);
        // pts: identity key, counts slot 0.
        assert_eq!(&descs[0..2], &[0, 1]);
        assert_eq!(descs[9], WFC_NO_LUT);
        assert_eq!(descs[10], 0);
        // rc: 2-word fold, lut slot 0, counts slot 1, rel 0 then rel 3.
        let rc0 = &descs[WFC_DESC_STRIDE..2 * WFC_DESC_STRIDE];
        assert_eq!(rc0[0], 1);
        assert_eq!(rc0[1], 2);
        assert_eq!(&rc0[2..4], &[9, 9]);
        assert_eq!(rc0[7], 0);
        assert_eq!(rc0[9], 0);
        assert_eq!(rc0[10], 1);
        let rc3 = &descs[2 * WFC_DESC_STRIDE..3 * WFC_DESC_STRIDE];
        assert_eq!(rc3[7], 3);
    }

    #[test]
    #[should_panic(expected = "width disagrees")]
    fn width_mismatch_is_loud() {
        let layout: &[(&str, usize, &str, u32, usize, usize)] =
            &[("rc", 0, "range_check_9_9_state", 0, 0, 3)];
        let _ = build_feed_descriptors(layout, &[RC99]);
    }
}
