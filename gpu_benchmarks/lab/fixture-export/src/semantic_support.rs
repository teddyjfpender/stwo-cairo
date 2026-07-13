//! Small, explicit dependency closure for canonical slab semantics.

use sha2::{Digest, Sha256};

pub const M31_P: u32 = 2_147_483_647;

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn canonical_value_hash(value: &serde_json::Value) -> Result<String, String> {
    let mut canonical = Vec::new();
    canonical_json(value, &mut canonical)?;
    Ok(sha256_hex(&canonical))
}

fn canonical_json(value: &serde_json::Value, output: &mut Vec<u8>) -> Result<(), String> {
    match value {
        serde_json::Value::Object(map) => {
            output.push(b'{');
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|error| format!("serialize canonical JSON key: {error}"))?;
                output.push(b':');
                canonical_json(value, output)?;
            }
            output.push(b'}');
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                canonical_json(value, output)?;
            }
            output.push(b']');
        }
        _ => serde_json::to_writer(output, value)
            .map_err(|error| format!("serialize canonical JSON value: {error}"))?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_key_order_does_not_change_hash() {
        let left: serde_json::Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        let right: serde_json::Value = serde_json::from_str(r#"{"a":1,"b":2}"#).unwrap();
        assert_eq!(canonical_value_hash(&left), canonical_value_hash(&right));
    }
}
