//! Host-only membership check for the AOT index embedded in a release binary.
//! No CUDA context is created: every requested `(cache_key, sm)` is resolved
//! directly against the build-time pack index.

use std::collections::BTreeSet;
use std::process::ExitCode;

fn values(name: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut args = std::env::args();
    while let Some(current) = args.next() {
        if current == name {
            if let Some(value) = args.next() {
                values.push(value);
            }
        }
    }
    values
}

fn value(name: &str) -> Option<String> {
    values(name).into_iter().next()
}

fn parse_key(value: &str) -> Result<u64, String> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("AOT key must be 16 lowercase hex digits: {value}"));
    }
    u64::from_str_radix(value, 16).map_err(|error| format!("invalid AOT key {value}: {error}"))
}

fn fail(error: String) -> ExitCode {
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({"pass": false, "error": error})).unwrap()
    );
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let sm = match value("--sm").and_then(|value| value.parse::<u32>().ok()) {
        Some(sm) if (10..=999).contains(&sm) => sm,
        _ => return fail("--sm <major*10+minor> is required".to_owned()),
    };
    let keys = match values("--key")
        .into_iter()
        .map(|key| parse_key(&key))
        .collect::<Result<BTreeSet<_>, _>>()
    {
        Ok(keys) if !keys.is_empty() => keys,
        Ok(_) => return fail("at least one --key is required".to_owned()),
        Err(error) => return fail(error),
    };
    let manifest_hash = stwo_backend_cuda::aot::loaded_manifest_hash();
    let missing = keys
        .iter()
        .copied()
        .filter(|key| !stwo_backend_cuda::aot::contains_loaded_kernel(*key, sm / 10, sm % 10))
        .map(|key| format!("{key:016x}"))
        .collect::<Vec<_>>();
    let pass = manifest_hash != 0 && missing.is_empty();
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "pass": pass,
            "sm": sm,
            "loaded_manifest_hash": format!("{manifest_hash:016x}"),
            "required_unique_key_count": keys.len(),
            "missing_keys": missing,
        }))
        .unwrap()
    );
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::parse_key;

    #[test]
    fn cache_keys_are_fixed_lower_hex() {
        assert_eq!(parse_key("0123456789abcdef"), Ok(0x0123_4567_89ab_cdef));
        for invalid in ["123", "0123456789ABCDEF", "zzzzzzzzzzzzzzzz"] {
            assert!(parse_key(invalid).is_err());
        }
    }
}
