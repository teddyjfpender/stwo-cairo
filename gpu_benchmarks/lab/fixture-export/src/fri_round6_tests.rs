use super::*;

#[test]
fn exact_layout_and_hostile_propagation() {
    let context = synthetic_context("11".repeat(32)).unwrap();
    let primary = build(&context, Case::Primary).unwrap();
    let hostile = build(&context, Case::Hostile).unwrap();
    validate_pair(&primary, &hostile).unwrap();
    assert_eq!(primary.payload.len(), PAYLOAD_BYTES);
    assert_eq!(primary.index.chunks.len(), 14);
    assert_eq!(primary.index.chunks[1].word_count, 56);
    assert_eq!(
        primary.index.shape.normalized_twiddle_offsets_words,
        [0, 32, 48]
    );
    assert_eq!(primary.index.predecessor_check.status, "PASS");
    assert_eq!(primary.index.predecessor_check.root6_leaf_count, 16);
    assert!(!primary.index.production_admissible);
    assert_eq!(
        primary.index.chunks.last().unwrap().offset_bytes + 64,
        PAYLOAD_BYTES
    );
}

#[test]
fn generation_is_byte_deterministic() {
    let context = synthetic_context("11".repeat(32)).unwrap();
    let left = build(&context, Case::Primary).unwrap();
    let right = build(&context, Case::Primary).unwrap();
    assert_eq!(left.payload, right.payload);
    assert_eq!(
        serde_json::to_vec(&left.index).unwrap(),
        serde_json::to_vec(&right.index).unwrap()
    );
}
