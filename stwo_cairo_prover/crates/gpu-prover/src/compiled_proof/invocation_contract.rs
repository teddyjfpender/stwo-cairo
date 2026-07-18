use super::*;

const INVOCATION_DOMAIN: &[u8] = b"stwo-cairo.compiled-proof.invocation.v1\0";

/// Address-free identity of one complete ordered AOT invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InvocationContractId([u8; 32]);

impl InvocationContractId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl AotInvocation {
    pub fn contract_id(&self) -> Result<InvocationContractId, CompiledProofError> {
        let mut canonical = Vec::from(INVOCATION_DOMAIN);
        encode_invocation_payload(&mut canonical, self)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(INVOCATION_DOMAIN);
        hasher.update(
            &u64::try_from(canonical.len())
                .map_err(|_| CompiledProofError::SizeOverflow)?
                .to_le_bytes(),
        );
        hasher.update(&canonical);
        Ok(InvocationContractId(*hasher.finalize().as_bytes()))
    }
}

/// Single canonical payload encoder shared by invocation authority and whole
/// proof identity. It contains no device or host address.
pub(crate) fn encode_invocation_payload(
    out: &mut Vec<u8>,
    invocation: &AotInvocation,
) -> Result<(), CompiledProofError> {
    push_size(out, invocation.arguments.len())?;
    for argument in &invocation.arguments {
        out.push(argument.ordinal);
        match &argument.value {
            AotArgumentValue::U32(value) => {
                out.push(0);
                out.extend_from_slice(&value.to_le_bytes());
            }
            AotArgumentValue::DevicePointer(binding) => {
                out.push(1);
                encode_binding(out, *binding);
            }
            AotArgumentValue::DevicePointerTable(entries) => {
                out.push(2);
                push_size(out, entries.len())?;
                for &entry in entries {
                    encode_binding(out, entry);
                }
            }
            AotArgumentValue::DeviceFixedU32 { value, binding } => {
                out.push(3);
                out.extend_from_slice(&value.0.to_le_bytes());
                out.extend_from_slice(&binding.0.to_le_bytes());
            }
            AotArgumentValue::Usize(value) => {
                out.push(4);
                out.extend_from_slice(&value.to_le_bytes());
            }
            AotArgumentValue::DeviceRegisteredFixedSourcePointerTable(reads) => {
                out.push(5);
                push_size(out, reads.len())?;
                for read in reads {
                    encode_registered_read(out, read)?;
                }
            }
            AotArgumentValue::DeviceMixedFixedSourcePointerTable(entries) => {
                out.push(6);
                push_size(out, entries.len())?;
                for entry in entries {
                    match entry {
                        FixedSourcePointerEntry::EffectBinding(binding) => {
                            out.push(0);
                            out.extend_from_slice(&binding.0.to_le_bytes());
                        }
                        FixedSourcePointerEntry::Registered(read) => {
                            out.push(1);
                            encode_registered_read(out, read)?;
                        }
                    }
                }
            }
            AotArgumentValue::DevicePointerTableValue(table) => {
                out.push(7);
                encode_pointer_table(out, table)?;
            }
            AotArgumentValue::DeviceNestedPointerTableValue { entries } => {
                out.push(8);
                push_size(out, entries.len())?;
                for entry in entries {
                    encode_pointer_table(out, entry)?;
                }
            }
            AotArgumentValue::HostFixedU32(words) => {
                out.push(9);
                push_size(out, words.len())?;
                for word in words {
                    out.extend_from_slice(&word.to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

fn encode_pointer_table(
    out: &mut Vec<u8>,
    table: &DevicePointerTableBinding,
) -> Result<(), CompiledProofError> {
    push_size(out, table.entries.len())?;
    for &entry in &table.entries {
        encode_binding(out, entry);
    }
    Ok(())
}

fn encode_binding(out: &mut Vec<u8>, binding: Option<EffectBindingId>) {
    match binding {
        Some(binding) => {
            out.push(1);
            out.extend_from_slice(&binding.0.to_le_bytes());
        }
        None => out.push(0),
    }
}

fn encode_registered_read(
    out: &mut Vec<u8>,
    read: &RegisteredFixedSourceRead,
) -> Result<(), CompiledProofError> {
    push_bytes(out, read.source().canonical_encoding())?;
    out.extend_from_slice(read.source().identity());
    push_size(out, read.column())?;
    push_size(out, read.elements().start)?;
    push_size(out, read.elements().end)
}

fn push_size(out: &mut Vec<u8>, value: usize) -> Result<(), CompiledProofError> {
    out.extend_from_slice(
        &u64::try_from(value)
            .map_err(|_| CompiledProofError::SizeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), CompiledProofError> {
    push_size(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invocation(value: AotArgumentValue) -> AotInvocation {
        AotInvocation {
            arguments: vec![AotArgumentBinding { ordinal: 0, value }],
        }
    }

    fn registered_read(column: usize) -> RegisteredFixedSourceRead {
        RegisteredFixedSourceRead::new(
            RegisteredFixedSourceAuthority::new(
                [9; 32],
                2,
                2,
                4,
                vec![b"first".to_vec(), b"second".to_vec()],
            )
            .unwrap(),
            column,
            ElementRange::new(0, 2).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn digest_binds_scalar_and_fixed_version_binding() {
        assert_ne!(
            invocation(AotArgumentValue::U32(7)).contract_id().unwrap(),
            invocation(AotArgumentValue::U32(8)).contract_id().unwrap()
        );
        assert_ne!(
            invocation(AotArgumentValue::DeviceFixedU32 {
                value: ValueVersion(3),
                binding: EffectBindingId(4),
            })
            .contract_id()
            .unwrap(),
            invocation(AotArgumentValue::DeviceFixedU32 {
                value: ValueVersion(4),
                binding: EffectBindingId(3),
            })
            .contract_id()
            .unwrap()
        );
    }

    #[test]
    fn usize_payload_is_exactly_one_canonical_u64() {
        let value = u64::from(u32::MAX) + 7;
        let invocation = invocation(AotArgumentValue::Usize(value));
        let mut encoded = Vec::new();
        encode_invocation_payload(&mut encoded, &invocation).unwrap();
        assert_eq!(&encoded[encoded.len() - 8..], &value.to_le_bytes());
    }

    #[test]
    fn digest_binds_pointer_table_none_some_and_order() {
        let exact = invocation(AotArgumentValue::DevicePointerTable(vec![
            Some(EffectBindingId(0)),
            None,
            Some(EffectBindingId(1)),
        ]));
        for changed in [
            invocation(AotArgumentValue::DevicePointerTable(vec![
                Some(EffectBindingId(1)),
                None,
                Some(EffectBindingId(0)),
            ])),
            invocation(AotArgumentValue::DevicePointerTable(vec![
                Some(EffectBindingId(0)),
                Some(EffectBindingId(1)),
                None,
            ])),
            invocation(AotArgumentValue::DevicePointerTable(vec![
                Some(EffectBindingId(0)),
                Some(EffectBindingId(1)),
                Some(EffectBindingId(1)),
            ])),
        ] {
            assert_ne!(exact.contract_id().unwrap(), changed.contract_id().unwrap());
        }
    }

    #[test]
    fn digest_binds_device_resident_pointer_graph_shape_order_nulls_and_leaves() {
        fn inner<const N: usize>(leaves: [Option<u32>; N]) -> DevicePointerTableBinding {
            DevicePointerTableBinding {
                entries: leaves
                    .into_iter()
                    .map(|leaf| leaf.map(EffectBindingId))
                    .collect(),
            }
        }
        let exact = invocation(AotArgumentValue::DeviceNestedPointerTableValue {
            entries: vec![inner([Some(0), None, Some(1)]), inner([Some(2), Some(3)])],
        });
        for changed in [
            invocation(AotArgumentValue::DeviceNestedPointerTableValue {
                entries: vec![inner([Some(2), Some(3)]), inner([Some(0), None, Some(1)])],
            }),
            invocation(AotArgumentValue::DeviceNestedPointerTableValue {
                entries: vec![inner([Some(0), Some(1), None]), inner([Some(2), Some(3)])],
            }),
            invocation(AotArgumentValue::DeviceNestedPointerTableValue {
                entries: vec![inner([Some(0), None, Some(4)]), inner([Some(2), Some(3)])],
            }),
            invocation(AotArgumentValue::DevicePointerTableValue(inner([
                Some(0),
                None,
                Some(1),
                Some(2),
                Some(3),
            ]))),
        ] {
            assert_ne!(exact.contract_id().unwrap(), changed.contract_id().unwrap());
        }
    }

    #[test]
    fn digest_binds_host_fixed_words_by_value() {
        assert_ne!(
            invocation(AotArgumentValue::HostFixedU32(vec![1, 2, 3]))
                .contract_id()
                .unwrap(),
            invocation(AotArgumentValue::HostFixedU32(vec![1, 3, 2]))
                .contract_id()
                .unwrap()
        );
    }

    #[test]
    fn digest_binds_mixed_registered_and_effect_entry_order() {
        let read = registered_read(1);
        let exact = invocation(AotArgumentValue::DeviceMixedFixedSourcePointerTable(vec![
            FixedSourcePointerEntry::EffectBinding(EffectBindingId(0)),
            FixedSourcePointerEntry::Registered(read.clone()),
        ]));
        let swapped = invocation(AotArgumentValue::DeviceMixedFixedSourcePointerTable(vec![
            FixedSourcePointerEntry::Registered(read),
            FixedSourcePointerEntry::EffectBinding(EffectBindingId(0)),
        ]));
        assert_ne!(exact.contract_id().unwrap(), swapped.contract_id().unwrap());
    }
}
