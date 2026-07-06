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
    /// unless the consumer pads differently. `0` = RUNTIME-SIZED (the memory
    /// tables): the caller supplies sizes via `build_feed_descriptors_sized`.
    pub table_size: usize,
    /// Number of relation slots (the consumer's `mults` array length).
    pub n_relations: usize,
    /// Whether keys map through the consumer's `input_to_row` LUT.
    pub needs_lut: bool,
    /// Descriptor kind: 0 = fold(+offset)(+LUT); 1 = MEM-ID DECODE
    /// (memory_id_to_big: tag = id >> 30 → big/small tables, val = low 30 bits;
    /// DEFAULT_ID skipped; the small table is a SECOND counts slot named
    /// `"<state_param>#small"`).
    pub kind: u32,
    /// Signed key offset applied after the fold (memory_address_to_id feeds
    /// addresses; rows are `addr - 1`).
    pub key_offset: i64,
}

/// The count-relation registry: every family the device feed serves, with
/// VERIFIED shapes (LOG_SIZE from cairo-air; mults lengths from the consumer
/// ClaimGenerators; key packing = the consumer's own `add_input` semantics —
/// direct value/row for single-word families, MSB-first tuple fold through the
/// generator's `input_to_row_lut` for multi-word ones). The local count gate
/// (`differential_test`) fences every entry against the consumer's real feeds.
pub const COUNT_RELATIONS: &[CountRelation] = &[
    CountRelation {
        state_param: "range_check_8_state",
        word_bits: &[8],
        table_size: 1 << 8,
        n_relations: 1,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_11_state",
        word_bits: &[11],
        table_size: 1 << 11,
        n_relations: 1,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_18_state",
        word_bits: &[18],
        table_size: 1 << 18,
        n_relations: 2,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_20_state",
        word_bits: &[20],
        table_size: 1 << 20,
        n_relations: 8,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_9_9_state",
        word_bits: &[9, 9],
        table_size: 1 << 18,
        n_relations: 8,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_4_4_state",
        word_bits: &[4, 4],
        table_size: 1 << 8,
        n_relations: 1,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_4_4_4_4_state",
        word_bits: &[4, 4, 4, 4],
        table_size: 1 << 16,
        n_relations: 1,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_3_3_3_3_3_state",
        word_bits: &[3, 3, 3, 3, 3],
        table_size: 1 << 15,
        n_relations: 1,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "range_check_7_2_5_state",
        word_bits: &[7, 2, 5],
        table_size: 1 << 14,
        n_relations: 1,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
    CountRelation {
        state_param: "pedersen_points_table_window_bits_18_state",
        word_bits: &[23],
        table_size: 1 << 23,
        n_relations: 1,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
    },
    // ---- The memory-table families (T2 at scale: every opcode feeds these
    // per row). Runtime-sized (address/id spaces are per-statement).
    CountRelation {
        state_param: "memory_address_to_id_state",
        word_bits: &[31],
        table_size: 0, // runtime: padded address space
        n_relations: 1,
        needs_lut: false,
        kind: 0,
        key_offset: -1, // add_input does increase_at(addr - 1)
    },
    CountRelation {
        state_param: "memory_id_to_big_state",
        word_bits: &[31],
        table_size: 0, // runtime: (big rows, small rows) via the sizes fn
        n_relations: 1,
        needs_lut: false,
        kind: 1,
        key_offset: 0,
    },
    CountRelation {
        state_param: "blake_round_sigma_state",
        word_bits: &[4],
        table_size: 16,
        n_relations: 1,
        needs_lut: true,
        kind: 0,
        key_offset: 0,
    },
];

/// Pure-Rust mirror of `witness_feed_counts_kernel` — the SAME descriptors,
/// fold, LUT indirection, and bounds behavior, over the word-major flats. The
/// local count gate runs THIS against consumer-fed states, so a keying bug is
/// caught without hardware; the CUDA kernel is then structurally identical.
pub fn host_feed_counts(
    sub_flat: &[u32],
    n_rows: usize,
    descs: &[u32],
    luts: &[Vec<u32>],
    counts: &mut [Vec<u32>],
) {
    for e in descs.chunks_exact(WFC_DESC_STRIDE) {
        let (word_base, n_words) = (e[0] as usize, e[1] as usize);
        let table_size = e[8] as usize;
        let kind = e[11];
        for row in 0..n_rows {
            if kind == 1 {
                // MEM-ID DECODE — see the kernel; e[12] = small size, e[13] =
                // small counts slot; DEFAULT_ID (empty) skipped defensively.
                let v = sub_flat[word_base * n_rows + row];
                if v == (1u32 << 30) - 1 {
                    continue;
                }
                let (tag, val) = (v >> 30, (v & 0x3FFF_FFFF) as usize);
                match tag {
                    1 if val < table_size => {
                        counts[e[10] as usize][e[7] as usize * table_size + val] += 1;
                    }
                    0 if val < e[12] as usize => {
                        let small = e[12] as usize;
                        counts[e[13] as usize][e[7] as usize * small + val] += 1;
                    }
                    _ => {}
                }
                continue;
            }
            let mut key: u64 = 0;
            for i in 0..n_words {
                key = (key << e[2 + i]) | u64::from(sub_flat[(word_base + i) * n_rows + row]);
            }
            let keyed = key as i64 + i64::from(e[12] as i32);
            // Key domain check BEFORE any LUT deref — mirrors the kernel; an
            // out-of-width tuple (impossible on a valid trace, where the host
            // feed would panic) is dropped, never an OOB access.
            if keyed < 0 || keyed as usize >= table_size {
                continue;
            }
            let key = keyed as usize;
            let idx = if e[9] == WFC_NO_LUT {
                key
            } else {
                luts[e[9] as usize][key] as usize
            };
            if idx < table_size {
                counts[e[10] as usize][e[7] as usize * table_size + idx] += 1;
            }
        }
    }
}

/// Descriptor stride of `witness_feed_counts.cu` (flat u32 ABI).
pub const WFC_DESC_STRIDE: usize = 14;
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
    let (descs, luts, counts, _sizes) = build_feed_descriptors_sized(layout, relations, &|_| None);
    (descs, luts, counts)
}

/// The full builder: `sizes(state_param)` supplies `(table_size, small_size)`
/// for RUNTIME-SIZED families (`table_size == 0` in the registry — the memory
/// tables). A runtime family without a size is SKIPPED (stays host-fed) — the
/// caller opts in per seam; a fixed-size family ignores `sizes`.
pub fn build_feed_descriptors_sized(
    layout: &[(&str, usize, &str, u32, usize, usize)],
    relations: &[CountRelation],
    sizes: &dyn Fn(&'static str) -> Option<(usize, usize)>,
) -> (
    Vec<u32>,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<usize>, // per counts-slot BUFFER length (n_relations * table rows)
) {
    let mut descs: Vec<u32> = Vec::new();
    let mut lut_slots: Vec<&'static str> = Vec::new();
    let mut counts_slots: Vec<&'static str> = Vec::new();
    let mut counts_sizes: Vec<usize> = Vec::new();
    let slot = |slots: &mut Vec<&'static str>, name: &'static str| -> u32 {
        slots.iter().position(|s| *s == name).unwrap_or_else(|| {
            slots.push(name);
            slots.len() - 1
        }) as u32
    };
    for &(_field, _instance, state, rel_index, base, words) in layout {
        let Some(rel) = relations.iter().find(|r| r.state_param == state) else {
            continue; // input-list or host-fed relation
        };
        let (table_size, small_size) = if rel.table_size == 0 {
            match sizes(rel.state_param) {
                Some(sz) => sz,
                None => continue, // runtime-sized, caller didn't opt in: host-fed
            }
        } else {
            (rel.table_size, 0)
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
        let counts_index = slot(&mut counts_slots, rel.state_param);
        let buf_len = rel.n_relations * table_size;
        if counts_index as usize == counts_sizes.len() {
            counts_sizes.push(buf_len);
        } else {
            assert_eq!(
                counts_sizes[counts_index as usize], buf_len,
                "count family {state} sized inconsistently across entries"
            );
        }
        let lut_index = if rel.needs_lut {
            slot(&mut lut_slots, rel.state_param)
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
        e[8] = table_size as u32;
        e[9] = lut_index;
        e[10] = counts_index;
        e[11] = rel.kind;
        if rel.kind == 1 {
            // Mem-id decode: the small table is its own counts slot; leak the
            // composed name once (bounded by the registry size).
            e[12] = small_size as u32;
            let small_name: &'static str =
                Box::leak(format!("{}#small", rel.state_param).into_boxed_str());
            // Reuse an existing "#small" slot if present (same family twice).
            let existing = counts_slots
                .iter()
                .position(|s| s.ends_with("#small") && s.starts_with(rel.state_param));
            e[13] = match existing {
                Some(i) => i as u32,
                None => {
                    let i = slot(&mut counts_slots, small_name);
                    counts_sizes.push(rel.n_relations * small_size);
                    i
                }
            };
        } else {
            e[12] = rel.key_offset as i32 as u32;
        }
        descs.extend_from_slice(&e);
    }
    (descs, lut_slots, counts_slots, counts_sizes)
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
        kind: 0,
        key_offset: 0,
    };
    const PTS: CountRelation = CountRelation {
        state_param: "pedersen_points_table_window_bits_18_state",
        word_bits: &[23],
        table_size: 1 << 23,
        n_relations: 1,
        needs_lut: false,
        kind: 0,
        key_offset: 0,
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
