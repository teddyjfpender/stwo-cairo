//! Deterministically thread `WitnessExecContext` through the AIR-generated Cairo
//! aggregate witness writer.
//!
//! The upstream generator is private, so repository-owned GPU seams are applied as
//! a strict post-processing step. Every transformed call family has an expected
//! count: upstream shape drift fails loudly instead of producing a partial edit.
//!
//! Usage:
//!   witness_exec_context_codegen [--prover-root PATH] [--check]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TARGET: &str = "crates/prover/src/witness/cairo_claim_generator.rs";
const GENERATED_HEADER: &str = "// This file was created by the AIR team.";
const CONTEXT_IMPORT: &str =
    "use crate::witness::exec_context::WitnessExecContext; // witness_exec_context_codegen";

fn parse_args() -> (PathBuf, bool) {
    let mut root = None;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prover-root" => {
                root = Some(PathBuf::from(
                    args.next().expect("--prover-root requires a path"),
                ));
            }
            "--check" => check = true,
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    (root.unwrap_or_else(find_prover_root), check)
}

fn find_prover_root() -> PathBuf {
    let cwd = std::env::current_dir().expect("current directory");
    for ancestor in cwd.ancestors() {
        if ancestor.join(TARGET).is_file() {
            return ancestor.to_path_buf();
        }
        let nested = ancestor.join("stwo_cairo_prover");
        if nested.join(TARGET).is_file() {
            return nested;
        }
    }
    eprintln!("could not locate stwo_cairo_prover; pass --prover-root");
    std::process::exit(2);
}

fn insert_once(source: &mut String, anchor: &str, addition: &str) {
    let count = source.matches(anchor).count();
    assert_eq!(count, 1, "expected one anchor `{anchor}`, found {count}");
    let at = source.find(anchor).unwrap();
    if source[..at].ends_with(addition) {
        return;
    }
    source.insert_str(at, addition);
}

fn inject_first_argument(source: &mut String, prefix: &str, expected: usize, turbofish: bool) {
    let count = source.matches(prefix).count();
    assert_eq!(
        count, expected,
        "generated call-family drift for `{prefix}`: expected {expected}, found {count}"
    );

    let mut from = 0;
    while let Some(relative) = source[from..].find(prefix) {
        let call_start = from + relative;
        let open = if turbofish {
            let relative_open = source[call_start + prefix.len()..]
                .find(">(")
                .unwrap_or_else(|| panic!("missing call open after `{prefix}`"));
            call_start + prefix.len() + relative_open + 1
        } else {
            call_start + prefix.len() - 1
        };
        let after_open = open + 1;
        let next = source[after_open..]
            .find(|c: char| !c.is_whitespace())
            .map(|offset| after_open + offset)
            .expect("call has a closing delimiter");
        if source[next..].starts_with("exec_context") {
            from = next + "exec_context".len();
            continue;
        }

        let insertion = if source.as_bytes()[next] == b')' {
            "exec_context".to_owned()
        } else if source[after_open..next].contains('\n') {
            let line_start = source[..call_start].rfind('\n').map_or(0, |at| at + 1);
            let indent: String = source[line_start..call_start]
                .chars()
                .take_while(|c| c.is_whitespace())
                .collect();
            format!("\n{indent}    exec_context,")
        } else {
            "exec_context, ".to_owned()
        };
        source.insert_str(after_open, &insertion);
        from = after_open + insertion.len();
    }
}

fn transform(mut source: String) -> String {
    assert!(
        source.starts_with(GENERATED_HEADER),
        "refusing to rewrite a file without the AIR-generated header"
    );

    // rustfmt sorts this import before `jit_prove_backend`; normalize there so
    // generation followed by formatting remains a fixed point.
    let context_import_line = format!("{CONTEXT_IMPORT}\n");
    source = source.replace(&context_import_line, "");
    insert_once(
        &mut source,
        "use crate::witness::jit_prove_backend::{",
        &context_import_line,
    );
    insert_once(
        &mut source,
        "        opt_n_id_to_big_components: Option<usize>,",
        "        exec_context: &WitnessExecContext,\n",
    );
    insert_once(
        &mut source,
        "        common_lookup_elements: &CommonLookupElements,",
        "        exec_context: &WitnessExecContext,\n",
    );

    for (prefix, expected) in [
        ("<B as OpcodeJitBackend>::lane_write_trace::<", 14),
        ("<B as OpcodeJitBackend>::device_interaction_pending::<", 14),
        (
            "<B as OpcodeJitBackend>::builtin_device_interaction_pending::<",
            5,
        ),
        ("<B as OpcodeJitBackend>::device_interaction::<", 14),
        ("<B as OpcodeJitBackend>::builtin_device_interaction::<", 5),
    ] {
        inject_first_argument(&mut source, prefix, expected, true);
    }

    for prefix in [
        "<B as BlakeRoundWitness>::write_trace(",
        "<B as BlakeGWitness>::write_trace(",
        "<B as PedersenAggregatorWindowBits18Witness>::write_trace(",
        "<B as PartialEcMulWindowBits18Witness>::write_trace(",
        "<B as PartialEcMulGenericWitness>::write_trace(",
        "<B as Cube252Witness>::write_trace(",
    ] {
        inject_first_argument(&mut source, prefix, 1, false);
    }

    source
}

fn main() -> ExitCode {
    let (root, check) = parse_args();
    let target = root.join(Path::new(TARGET));
    let source = std::fs::read_to_string(&target).expect("read cairo_claim_generator.rs");
    let generated = transform(source.clone());

    if check {
        if generated == source {
            println!("witness_exec_context_codegen --check: OK (no drift)");
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "witness_exec_context_codegen --check: DRIFT — regenerate {}",
                target.display()
            );
            ExitCode::FAILURE
        }
    } else {
        if generated != source {
            std::fs::write(&target, generated).expect("write cairo_claim_generator.rs");
            println!("witness_exec_context_codegen: rewrote {}", target.display());
        } else {
            println!("witness_exec_context_codegen: already current");
        }
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_aggregate_has_no_generated_drift() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let prover_root = manifest.parent().unwrap().parent().unwrap();
        let target = prover_root.join(TARGET);
        let source = std::fs::read_to_string(target).unwrap();
        assert_eq!(transform(source.clone()), source);
    }
}
