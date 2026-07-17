use std::collections::BTreeMap;

use super::super::producer_prefix::BaseProducerAuthority;

pub(in super::super) fn assert_witness_writer_def_use_is_ordered(
    authority: &BaseProducerAuthority,
) {
    let mut destination_producer = BTreeMap::new();
    for (producer_index, producer) in authority.producers.iter().enumerate() {
        for destination in producer
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.destination())
        {
            if let Some(previous) =
                destination_producer.insert(destination.value.version, producer_index)
            {
                assert_eq!(
                    previous, producer_index,
                    "one semantic version cannot be produced by different witness writers"
                );
            }
        }
    }
    for (producer_index, producer) in authority.producers.iter().enumerate() {
        for source in producer
            .effect()
            .accesses()
            .iter()
            .filter_map(|access| access.source())
        {
            if let Some(&source_producer) = destination_producer.get(&source.value.version) {
                assert!(
                    source_producer < producer_index,
                    "writer-produced source version must come from an earlier writer"
                );
            }
        }
    }
}
