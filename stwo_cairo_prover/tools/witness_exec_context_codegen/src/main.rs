//! Deterministically thread `WitnessExecContext` through the AIR-generated Cairo
//! aggregate witness writer.
//!
//! The upstream generator is private, so repository-owned GPU seams are applied as
//! a strict post-processing step. Every transformed call family has an expected
//! count: upstream shape drift fails loudly instead of producing a partial edit.
//!
//! Usage:
//!   witness_exec_context_codegen [--prover-root PATH] [--check]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TARGET: &str = "crates/prover/src/witness/cairo_claim_generator.rs";
const GENERATED_HEADER: &str = "// This file was created by the AIR team.";
const CONTEXT_IMPORT: &str =
    "use crate::witness::exec_context::WitnessExecContext; // witness_exec_context_codegen";
const COMPONENT_DIR: &str = "crates/prover/src/witness/components";
const RELATION_BEGIN: &str = "// === BEGIN relation_lookup_source_codegen ===";
const RELATION_END: &str = "// === END relation_lookup_source_codegen ===";
const INTERACTION_COMPONENTS: usize = 67;

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

fn claim_component_names(source: &str) -> BTreeSet<String> {
    claim_component_names_in_order(source).into_iter().collect()
}

fn claim_component_names_in_order(source: &str) -> Vec<String> {
    let start = source
        .find("pub struct CairoClaimGenerator {")
        .expect("CairoClaimGenerator struct");
    let body_start = start + "pub struct CairoClaimGenerator {".len();
    let end = source[body_start..]
        .find("\n}\n\nimpl CairoClaimGenerator")
        .map(|offset| body_start + offset)
        .expect("end of CairoClaimGenerator struct");
    let fields: Vec<_> = source[body_start..end]
        .lines()
        .filter_map(|line| {
            let field = line.trim().strip_prefix("pub ")?.split(':').next()?;
            (!matches!(field, "public_data" | "jit_memory")).then(|| field.to_owned())
        })
        .collect();
    assert_eq!(
        fields.len(),
        INTERACTION_COMPONENTS,
        "Cairo interaction component count drift"
    );
    let unique: BTreeSet<_> = fields.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        fields.len(),
        "duplicate Cairo component field"
    );
    fields
}

fn interaction_component_names_in_order(source: &str) -> Vec<String> {
    let start = source
        .find("pub struct CairoInteractionClaimGenerator<")
        .expect("CairoInteractionClaimGenerator struct");
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("CairoInteractionClaimGenerator open brace");
    let mut depth = 0usize;
    let close = source[open..]
        .char_indices()
        .find_map(|(offset, ch)| match ch {
            '{' => {
                depth += 1;
                None
            }
            '}' => {
                depth -= 1;
                (depth == 0).then_some(open + offset)
            }
            _ => None,
        })
        .expect("CairoInteractionClaimGenerator close brace");
    let fields: Vec<_> = source[open + 1..close]
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("pub ")?
                .split(':')
                .next()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(
        fields.len(),
        INTERACTION_COMPONENTS,
        "CairoInteractionClaimGenerator field-count drift"
    );
    let unique: BTreeSet<_> = fields.iter().cloned().collect();
    assert_eq!(unique.len(), fields.len(), "duplicate interaction field");
    fields
}

fn replace_generated_block(source: &mut String, block: &str, anchor: &str) {
    match (source.find(RELATION_BEGIN), source.find(RELATION_END)) {
        (Some(begin), Some(end)) => {
            assert!(begin < end, "relation source marker order");
            assert_eq!(source.matches(RELATION_BEGIN).count(), 1);
            assert_eq!(source.matches(RELATION_END).count(), 1);
            let mut end = end + RELATION_END.len();
            while source.as_bytes().get(end) == Some(&b'\n') {
                end += 1;
            }
            source.replace_range(begin..end, block);
        }
        (None, None) => insert_once(source, anchor, block),
        _ => panic!("unpaired relation source codegen markers"),
    }
}

fn render_aggregate_relation_export(fields: &[String]) -> String {
    let mut block = format!(
        "{RELATION_BEGIN}\nimpl<B: MemoryIdToBigWitness + BlakeGWitness> CairoInteractionClaimGenerator<B> {{\n    /// Consumes every active interaction state into typed relation sources.\n    /// This is generated from the aggregate fields; component additions cannot\n    /// silently bypass the GPU-native relation layer.\n    pub fn into_relation_lookup_sources(\n        self,\n        exec_context: &WitnessExecContext,\n    ) -> Result<\n        crate::witness::relation_sources::CairoRelationSourceSet,\n        crate::witness::relation_sources::RelationSourceError,\n    > {{\n        use crate::witness::relation_sources::RelationLookupSourceExport;\n\n        let mut sources = Vec::new();\n"
    );
    for field in fields {
        block.push_str(&format!(
            "        if let Some(gen) = self.{field} {{\n            sources.extend(gen.export_relation_lookup_sources(\n                \"{field}\",\n                exec_context,\n            )?);\n        }}\n"
        ));
    }
    block.push_str(
        "        exec_context.assert_interaction_drained();\n        crate::witness::relation_sources::CairoRelationSourceSet::new(sources)\n    }\n}\n",
    );
    block.push_str(RELATION_END);
    block.push_str("\n\n");
    block
}

fn lookup_fields(source: &str) -> Vec<(String, String)> {
    let struct_start = source.find("struct LookupData").expect("LookupData struct");
    let open = source[struct_start..]
        .find('{')
        .map(|offset| struct_start + offset)
        .expect("LookupData open brace");
    let mut depth = 0usize;
    let close = source[open..]
        .char_indices()
        .find_map(|(offset, ch)| match ch {
            '{' => {
                depth += 1;
                None
            }
            '}' => {
                depth -= 1;
                (depth == 0).then_some(open + offset)
            }
            _ => None,
        })
        .expect("LookupData close brace");
    source[open + 1..close]
        .split(',')
        .filter_map(|field| {
            let field = field.trim();
            if field.is_empty() {
                return None;
            }
            let (name, ty) = field
                .split_once(':')
                .unwrap_or_else(|| panic!("invalid LookupData field `{field}`"));
            let ty = ty.trim();
            let width = if ty == "Vec<PackedM31>" {
                "scalar".to_owned()
            } else {
                ty.strip_prefix("Vec<[PackedM31; ")
                    .and_then(|rest| rest.strip_suffix("]>"))
                    .unwrap_or_else(|| panic!("unsupported LookupData type `{ty}`"))
                    .to_owned()
            };
            Some((name.trim().to_owned(), width))
        })
        .collect()
}

fn render_component_relation_export(component: &str, source: &str) -> String {
    let invocation = match component {
        "memory_address_to_id" => {
            "crate::relation_lookup_source_memory_address_to_id!();\n".to_owned()
        }
        "memory_id_to_big" => "crate::relation_lookup_source_memory_id_to_big!();\n".to_owned(),
        "verify_bitwise_xor_12" => "crate::relation_lookup_source_xor12!();\n".to_owned(),
        _ => {
            let fields = lookup_fields(source);
            assert!(!fields.is_empty(), "empty LookupData for {component}");
            let mut invocation = "crate::relation_lookup_source! {\n".to_owned();
            for (field, width) in fields {
                invocation.push_str(&format!("    {field}: {width},\n"));
            }
            invocation.push_str("}\n");
            invocation
        }
    };
    format!("{RELATION_BEGIN}\n{invocation}{RELATION_END}\n")
}

fn transform_component(component: &str, mut source: String) -> String {
    assert!(
        source.contains("pub struct InteractionClaimGenerator"),
        "refusing to rewrite unexpected component source: {component}"
    );
    let block = render_component_relation_export(component, &source);
    replace_generated_block(&mut source, &block, "impl InteractionClaimGenerator {");
    source
}

fn inject_final_shape_recording(source: &mut String) {
    const PREFIX: &str = "tracing::info_span!(\"wt:";
    const ENTERED: &str = ".entered()";
    const MARKER: &str = "// final_shape_ledger_codegen";

    let expected = claim_component_names(source);
    let mut observed = BTreeSet::new();
    let mut insertions = Vec::new();
    let mut from = 0;
    while let Some(relative) = source[from..].find(PREFIX) {
        let start = from + relative;
        let name_start = start + PREFIX.len();
        let name_end = source[name_start..]
            .find('"')
            .map(|offset| name_start + offset)
            .expect("unterminated wt component span");
        let name = &source[name_start..name_end];
        assert!(
            observed.insert(name.to_owned()),
            "duplicate witness component span: {name}"
        );
        let entered_end = source[name_end..]
            .find(ENTERED)
            .map(|offset| name_end + offset + ENTERED.len())
            .unwrap_or_else(|| panic!("missing span terminator for witness component {name}"));
        let semicolon = source[entered_end..]
            .find(|c: char| !c.is_whitespace())
            .map(|offset| entered_end + offset)
            .expect("witness component span statement ends with a semicolon");
        assert_eq!(source.as_bytes()[semicolon], b';');
        let statement_end = semicolon + 1;
        let next_line_end = source[statement_end..]
            .find('\n')
            .map_or(source.len(), |offset| statement_end + offset);
        if !source[statement_end..next_line_end].contains(MARKER)
            && !source[statement_end..].trim_start().starts_with(
                "exec_context.record_final_component(&gen, opt_n_id_to_big_components);",
            )
        {
            let line_start = source[..start].rfind('\n').map_or(0, |at| at + 1);
            let indent: String = source[line_start..start]
                .chars()
                .take_while(|c| c.is_whitespace())
                .collect();
            insertions.push((
                statement_end,
                format!(
                    "\n{indent}exec_context.record_final_component(&gen, opt_n_id_to_big_components); {MARKER}"
                ),
            ));
        }
        from = statement_end;
    }
    assert_eq!(
        observed, expected,
        "witness final-shape spans must match every CairoClaimGenerator component"
    );
    for (at, insertion) in insertions.into_iter().rev() {
        source.insert_str(at, &insertion);
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

    inject_final_shape_recording(&mut source);

    let fields = interaction_component_names_in_order(&source);
    assert_eq!(
        fields.iter().cloned().collect::<BTreeSet<_>>(),
        claim_component_names(&source),
        "base and interaction aggregate component sets diverged"
    );
    let block = render_aggregate_relation_export(&fields);
    replace_generated_block(
        &mut source,
        &block,
        "impl<B: MemoryIdToBigWitness + BlakeGWitness + OpcodeJitBackend> CairoInteractionClaimGenerator<B> {",
    );

    source
}

fn component_outputs(root: &Path, aggregate_source: &str) -> Vec<(PathBuf, String, String)> {
    interaction_component_names_in_order(aggregate_source)
        .into_iter()
        .map(|component| {
            let path = root.join(COMPONENT_DIR).join(format!("{component}.rs"));
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let generated = transform_component(&component, source.clone());
            (path, source, generated)
        })
        .collect()
}

fn main() -> ExitCode {
    let (root, check) = parse_args();
    let target = root.join(Path::new(TARGET));
    let source = std::fs::read_to_string(&target).expect("read cairo_claim_generator.rs");
    let generated = transform(source.clone());
    let components = component_outputs(&root, &generated);

    if check {
        let drifted: Vec<_> = components
            .iter()
            .filter(|(_, source, generated)| source != generated)
            .map(|(path, ..)| path)
            .collect();
        if generated == source && drifted.is_empty() {
            println!("witness_exec_context_codegen --check: OK (67 relation exporters, no drift)");
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "witness_exec_context_codegen --check: DRIFT — regenerate {}",
                target.display()
            );
            for path in drifted {
                eprintln!("  {}", path.display());
            }
            ExitCode::FAILURE
        }
    } else {
        if generated != source {
            std::fs::write(&target, generated).expect("write cairo_claim_generator.rs");
            println!("witness_exec_context_codegen: rewrote {}", target.display());
        } else {
            println!("witness_exec_context_codegen: already current");
        }
        let mut rewritten = 0usize;
        for (path, source, generated) in components {
            if generated != source {
                std::fs::write(&path, generated)
                    .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
                rewritten += 1;
            }
        }
        println!(
            "witness_exec_context_codegen: 67 relation exporters ({rewritten} component files rewritten)"
        );
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
        for (path, source, generated) in component_outputs(prover_root, &source) {
            assert_eq!(generated, source, "component drift: {}", path.display());
        }
    }

    #[test]
    fn aggregate_relation_export_has_every_component_arm() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let prover_root = manifest.parent().unwrap().parent().unwrap();
        let source = std::fs::read_to_string(prover_root.join(TARGET)).unwrap();
        let fields = interaction_component_names_in_order(&source);
        let block = render_aggregate_relation_export(&fields);
        assert_eq!(fields.len(), INTERACTION_COMPONENTS);
        assert_eq!(block.matches("export_relation_lookup_sources(").count(), 67);
    }
}
