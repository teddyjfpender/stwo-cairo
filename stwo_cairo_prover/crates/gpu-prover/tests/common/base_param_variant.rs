use stwo_cairo_adapter::builtins::MemorySegmentAddresses;
use stwo_cairo_adapter::ProverInput;

/// Build an equivalent valid statement whose bitwise and EC-op segments occupy
/// each other's contiguous relocation slots. The permutation preserves every
/// table length and trace shape, but changes both components' hoisted BASE
/// parameters. Rebase every address and pointer in the moved span so Cairo
/// memory semantics remain unchanged.
pub(crate) fn swap_bitwise_and_ec_op_segments(mut input: ProverInput) -> ProverInput {
    let bitwise = input
        .builtin_segments
        .bitwise_builtin
        .expect("strict resident fixture has a bitwise segment");
    let ec_op = input
        .builtin_segments
        .ec_op_builtin
        .expect("strict resident fixture has an EC-op segment");
    assert_eq!(
        bitwise.stop_ptr, ec_op.begin_addr,
        "fixture bitwise and EC-op segments stopped being contiguous"
    );
    let bitwise_len = bitwise.stop_ptr - bitwise.begin_addr;
    let ec_op_len = ec_op.stop_ptr - ec_op.begin_addr;
    let rebase = |address: usize| {
        if (bitwise.begin_addr..bitwise.stop_ptr).contains(&address) {
            address + ec_op_len
        } else if (ec_op.begin_addr..ec_op.stop_ptr).contains(&address) {
            address - bitwise_len
        } else {
            address
        }
    };
    let rebase_small =
        |value: u128| usize::try_from(value).map_or(value, |address| rebase(address) as u128);

    // Fail closed if a future fixture introduces an ordinary small felt in the
    // relocation span. Today these are exactly the 51 bitwise pointers (stride
    // five) and 51 EC-op pointers (stride seven). The half-open segments are
    // contiguous, but padding leaves the bitwise pointer lattice below the
    // shared boundary, so the two lattices contain 102 distinct values.
    let moved_small_values = input
        .memory
        .small_values
        .iter()
        .filter(|value| (bitwise.begin_addr as u128..ec_op.stop_ptr as u128).contains(value))
        .collect::<Vec<_>>();
    let expected_moved_small_values = (0..=50)
        .map(|index| (bitwise.begin_addr + index * 5) as u128)
        .chain((0..=50).map(|index| (ec_op.begin_addr + index * 7) as u128))
        .collect::<std::collections::BTreeSet<_>>();
    let actual_moved_small_values = moved_small_values
        .iter()
        .map(|value| **value)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        moved_small_values.len(),
        actual_moved_small_values.len(),
        "relocation span contains duplicate compact small values"
    );
    assert_eq!(
        actual_moved_small_values, expected_moved_small_values,
        "relocation span contains a non-pointer compact small value"
    );
    for (id, value) in input.memory.small_values.iter().enumerate() {
        if rebase_small(*value) != *value {
            assert!(
                input
                    .memory
                    .address_to_id
                    .iter()
                    .enumerate()
                    .filter(|(_, encoded)| encoded.0 == id as u32)
                    .all(|(address, _)| address < bitwise.begin_addr),
                "moved small value {value} is not confined to execution pointers"
            );
        }
    }

    let original_address_to_id = input.memory.address_to_id.clone();
    for address in bitwise.begin_addr..ec_op.stop_ptr {
        input.memory.address_to_id[rebase(address)] = original_address_to_id[address];
    }
    for value in &mut input.memory.small_values {
        *value = rebase_small(*value);
    }
    for address in &mut input.public_memory_addresses {
        *address = rebase(*address as usize) as u32;
    }
    input.builtin_segments.ec_op_builtin = Some(MemorySegmentAddresses {
        begin_addr: bitwise.begin_addr,
        stop_ptr: bitwise.begin_addr + ec_op_len,
    });
    input.builtin_segments.bitwise_builtin = Some(MemorySegmentAddresses {
        begin_addr: bitwise.begin_addr + ec_op_len,
        stop_ptr: ec_op.stop_ptr,
    });
    input
}
