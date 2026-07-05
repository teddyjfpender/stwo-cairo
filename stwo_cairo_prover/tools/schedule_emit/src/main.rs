//! schedule_emit — generate the gpu-prover component schedule table from the
//! transformer-emitted metadata (GPU_RESIDENT_PROVER_DESIGN.md §17, rule R8:
//! the table is derivable; hand-writing it is the data-entry bug class).
//!
//! Ground truth consumed (parse-only, no hand data):
//!  - `SUB_FEED_LAYOUT` in every `crates/prover/src/witness/components/*.rs`
//!    (emitted by witness_genericize; DECLARATION order): one entry per
//!    `SubComponentInputs` field instance —
//!    (field, instance, downstream state param, relation_index, word base,
//!    words per instance).
//!  - `COUNT_RELATIONS` in `crates/prover/src/witness/device_feed.rs`
//!    (state_param, n_relations): the device-atomic count families.
//!
//! Derivation: a layout entry whose state param is a count family becomes a
//! `CountFeed`; every other entry is a producer→consumer feed edge, coalesced
//! per (producer, consumer, words-per-instance) into an `OutputEdge` with an
//! instance count, and INVERTED into the consumer's `Producer` input edge — so
//! `Schedule::validate()`'s width agreement holds by construction and the
//! topological levels emerge from real feed data.
//!
//! Usage:
//!   schedule_emit --prover-root <stwo_cairo_prover> [--check]
//!
//! Writes (or, with --check, byte-compares against)
//! `crates/gpu-prover/src/schedule_table.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One parsed SUB_FEED_LAYOUT entry.
#[derive(Debug, Clone)]
struct FeedEntry {
    state_param: String,
    relation_index: u32,
    word_base: usize,
    words_per_instance: usize,
}

fn parse_args() -> (PathBuf, bool) {
    let mut root: Option<PathBuf> = None;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
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
    let root = root.unwrap_or_else(|| {
        // Default: run from anywhere inside the repo; find stwo_cairo_prover.
        let cwd = std::env::current_dir().unwrap();
        let mut dir = cwd.as_path();
        loop {
            let candidate = dir.join("stwo_cairo_prover");
            if candidate.join("crates/prover/src/witness/components").is_dir() {
                return candidate;
            }
            if dir.join("crates/prover/src/witness/components").is_dir() {
                return dir.to_path_buf();
            }
            match dir.parent() {
                Some(parent) => dir = parent,
                None => {
                    eprintln!("could not locate stwo_cairo_prover; pass --prover-root");
                    std::process::exit(2);
                }
            }
        }
    });
    (root, check)
}

/// Extract string/int literals from a syn expression tree.
fn lit_str(e: &syn::Expr) -> Option<String> {
    if let syn::Expr::Lit(l) = e {
        if let syn::Lit::Str(s) = &l.lit {
            return Some(s.value());
        }
    }
    None
}

fn lit_int(e: &syn::Expr) -> Option<u64> {
    if let syn::Expr::Lit(l) = e {
        if let syn::Lit::Int(i) = &l.lit {
            return i.base10_parse::<u64>().ok();
        }
    }
    None
}

/// Parse `SUB_FEED_LAYOUT` from a component file, if present.
fn parse_sub_feed_layout(file: &syn::File) -> Option<Vec<FeedEntry>> {
    for item in &file.items {
        let syn::Item::Const(c) = item else { continue };
        if c.ident != "SUB_FEED_LAYOUT" {
            continue;
        }
        let syn::Expr::Reference(r) = c.expr.as_ref() else {
            continue;
        };
        let syn::Expr::Array(arr) = r.expr.as_ref() else {
            continue;
        };
        let mut entries = Vec::new();
        for el in &arr.elems {
            let syn::Expr::Tuple(t) = el else {
                eprintln!("SUB_FEED_LAYOUT entry is not a tuple");
                std::process::exit(1);
            };
            let e: Vec<&syn::Expr> = t.elems.iter().collect();
            assert_eq!(e.len(), 6, "SUB_FEED_LAYOUT tuple arity changed");
            entries.push(FeedEntry {
                state_param: lit_str(e[2]).expect("state param"),
                relation_index: lit_int(e[3]).expect("relation index") as u32,
                word_base: lit_int(e[4]).expect("word base") as usize,
                words_per_instance: lit_int(e[5]).expect("words per instance") as usize,
            });
        }
        return Some(entries);
    }
    None
}

/// Parse `COUNT_RELATIONS` (state_param → n_relations) from device_feed.rs.
fn parse_count_relations(file: &syn::File) -> BTreeMap<String, u32> {
    fn scan_items(items: &[syn::Item], out: &mut BTreeMap<String, u32>) {
        for item in items {
            match item {
                syn::Item::Const(c) if c.ident == "COUNT_RELATIONS" => {
                    let syn::Expr::Reference(r) = c.expr.as_ref() else {
                        continue;
                    };
                    let syn::Expr::Array(arr) = r.expr.as_ref() else {
                        continue;
                    };
                    for el in &arr.elems {
                        let syn::Expr::Struct(s) = el else { continue };
                        let mut param = None;
                        let mut n_rel = None;
                        for f in &s.fields {
                            let syn::Member::Named(name) = &f.member else {
                                continue;
                            };
                            if name == "state_param" {
                                param = lit_str(&f.expr);
                            } else if name == "n_relations" {
                                n_rel = lit_int(&f.expr);
                            }
                        }
                        if let (Some(p), Some(n)) = (param, n_rel) {
                            out.insert(p, n as u32);
                        }
                    }
                }
                syn::Item::Mod(m) => {
                    if let Some((_, items)) = &m.content {
                        scan_items(items, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = BTreeMap::new();
    scan_items(&file.items, &mut out);
    assert!(
        !out.is_empty(),
        "COUNT_RELATIONS not found in device_feed.rs — registry moved?"
    );
    out
}

/// A coalesced feed edge: `n_instances` × `words_per_instance` words @ `word_base`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    word_base: usize,
    words_per_instance: usize,
    n_instances: usize,
}

#[derive(Default, Debug)]
struct Node {
    /// consumer component → coalesced edge (from this node's sub buffer).
    outputs: BTreeMap<String, Edge>,
    /// count family (state-param name) → number of distinct relation indices fed.
    counts: BTreeMap<String, u32>,
    /// producer component → edge (inverted from producers' outputs).
    inputs: BTreeMap<String, Edge>,
    /// whether this component has its own emitted lane metadata (vs. existing only
    /// as a feed target).
    has_layout: bool,
}

fn main() -> ExitCode {
    let (root, check) = parse_args();
    let components_dir = root.join("crates/prover/src/witness/components");
    let device_feed = root.join("crates/prover/src/witness/device_feed.rs");
    let out_path = root.join("crates/gpu-prover/src/schedule_table.rs");

    let count_families = parse_count_relations(
        &syn::parse_file(&std::fs::read_to_string(&device_feed).expect("read device_feed.rs"))
            .expect("parse device_feed.rs"),
    );

    let mut nodes: BTreeMap<String, Node> = BTreeMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&components_dir)
        .expect("read components dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();

    for path in &files {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        if stem == "mod" {
            continue;
        }
        let src = std::fs::read_to_string(path).expect("read component");
        let file = match syn::parse_file(&src) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("parse {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        let Some(entries) = parse_sub_feed_layout(&file) else {
            continue;
        };
        let node = nodes.entry(stem.clone()).or_default();
        node.has_layout = true;

        // Group entries per state param.
        let mut per_target: BTreeMap<String, Vec<&FeedEntry>> = BTreeMap::new();
        for e in &entries {
            per_target.entry(e.state_param.clone()).or_default().push(e);
        }
        for (param, group) in per_target {
            if count_families.contains_key(&param) {
                // Count family: record the number of distinct relation indices fed.
                let distinct: BTreeSet<u32> =
                    group.iter().map(|e| e.relation_index).collect();
                node.counts.insert(param, distinct.len() as u32);
            } else {
                // Feed edge to another component: coalesce. Instances must share a
                // width and tile contiguously from the minimal base (the gather
                // kernel's addressing model) — assert, don't assume.
                let target = param
                    .strip_suffix("_state")
                    .unwrap_or_else(|| {
                        eprintln!("state param without _state suffix: {param}");
                        std::process::exit(1);
                    })
                    .to_string();
                let words = group[0].words_per_instance;
                assert!(
                    group.iter().all(|e| e.words_per_instance == words),
                    "{stem}→{target}: mixed instance widths"
                );
                let mut bases: Vec<usize> = group.iter().map(|e| e.word_base).collect();
                bases.sort_unstable();
                let base = bases[0];
                for (i, b) in bases.iter().enumerate() {
                    assert_eq!(
                        *b,
                        base + i * words,
                        "{stem}→{target}: non-contiguous instance tiling"
                    );
                }
                node.outputs.insert(
                    target,
                    Edge {
                        word_base: base,
                        words_per_instance: words,
                        n_instances: bases.len(),
                    },
                );
            }
        }
    }

    // Materialize nodes for every referenced consumer, then invert edges.
    let producers: Vec<(String, BTreeMap<String, Edge>)> = nodes
        .iter()
        .map(|(id, n)| (id.clone(), n.outputs.clone()))
        .collect();
    for (producer, outputs) in producers {
        for (consumer, edge) in outputs {
            nodes
                .entry(consumer)
                .or_default()
                .inputs
                .insert(producer.clone(), edge);
        }
    }

    // Emit.
    let mut s = String::new();
    s.push_str(
        "//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT (design R8/§17).\n\
         //! Regenerate: `cargo run --manifest-path tools/schedule_emit/Cargo.toml`;\n\
         //! CI drift gate: same command with `--check`.\n\
         //!\n\
         //! The Cairo component feed DAG, derived from the transformer-emitted\n\
         //! SUB_FEED_LAYOUT consts + the COUNT_RELATIONS registry. Node order is\n\
         //! alphabetical (stable emission); EXECUTION order comes from\n\
         //! `Schedule::levels()`; trace COLLECTION order stays the claim\n\
         //! generator's (Fiat-Shamir-fixed) and is not this table's concern.\n\n\
         use crate::schedule::{\n    ComponentNode, CountFeed, InputEdge, LogSizeSource, OutputEdge, Schedule,\n};\n\n\
         pub static CAIRO_SCHEDULE: Schedule = Schedule { nodes: NODES };\n\n",
    );
    s.push_str("static NODES: &[ComponentNode] = &[\n");
    for (id, node) in &nodes {
        s.push_str("    ComponentNode {\n");
        s.push_str(&format!("        id: \"{id}\",\n"));
        s.push_str("        kernel: None,\n");
        s.push_str("        log_size: LogSizeSource::FromStates,\n");
        // Inputs: inverted producer edges (ExecTables/DeviceTable refinement is a
        // consumption-time concern, M2d).
        if node.inputs.is_empty() {
            s.push_str("        inputs: &[],\n");
        } else {
            s.push_str("        inputs: &[\n");
            for (of, e) in &node.inputs {
                s.push_str(&format!(
                    "            InputEdge::Producer {{\n                of: \"{of}\",\n                word_base: {},\n                words_per_instance: {},\n                n_instances: {},\n            }},\n",
                    e.word_base, e.words_per_instance, e.n_instances
                ));
            }
            s.push_str("        ],\n");
        }
        if node.outputs.is_empty() {
            s.push_str("        outputs: &[],\n");
        } else {
            s.push_str("        outputs: &[\n");
            for (to, e) in &node.outputs {
                s.push_str(&format!(
                    "            OutputEdge {{\n                to: \"{to}\",\n                word_base: {},\n                words_per_instance: {},\n                n_instances: {},\n            }},\n",
                    e.word_base, e.words_per_instance, e.n_instances
                ));
            }
            s.push_str("        ],\n");
        }
        if node.counts.is_empty() {
            s.push_str("        counts: &[],\n");
        } else {
            s.push_str("        counts: &[\n");
            for (family, n) in &node.counts {
                s.push_str(&format!(
                    "            CountFeed {{ family: \"{family}\", n_relations: {n} }},\n"
                ));
            }
            s.push_str("        ],\n");
        }
        s.push_str("        slots: None,\n");
        s.push_str("    },\n");
    }
    s.push_str("];\n");

    if check {
        let on_disk = std::fs::read_to_string(&out_path).unwrap_or_default();
        if on_disk == s {
            println!("schedule_emit --check: OK (no drift)");
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "schedule_emit --check: DRIFT — regenerate {} (components metadata changed)",
                out_path.display()
            );
            ExitCode::FAILURE
        }
    } else {
        std::fs::write(&out_path, &s).expect("write schedule_table.rs");
        println!(
            "schedule_emit: wrote {} ({} nodes)",
            out_path.display(),
            nodes.len()
        );
        ExitCode::SUCCESS
    }
}
