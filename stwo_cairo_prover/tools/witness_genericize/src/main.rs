//! `witness_genericize` — a deterministic, idempotent, re-runnable transformer that
//! rewrites an AIR-generated monomorphic `write_trace_simd` witness writer into a generic
//! per-row body over the `WitnessEval` trait (see the prover crate's
//! `witness/witness_eval/mod.rs`), plus a SimdBackend driver (byte-identical to the
//! original) and a RecordingEvaluator driver (per-row scalar-SSA JIT bytecode).
//!
//! It inserts a MARKED block (`// === BEGIN witness_genericize ... END ===`) into the
//! generated file, leaving the original `write_trace_simd` untouched as the byte-equality
//! baseline. Re-running strips any prior marked block and re-emits — so it is idempotent
//! and survives upstream regeneration by stwo-air-infra.
//!
//! DESIGN LAW: this is a PATTERN REWRITER, not a Rust compiler. It implements a finite
//! rewrite table over the known generated idioms (see `Lowerer`). ANY unmatched construct
//! is a LOUD per-file skip that quotes the exact construct — never a best-effort emission.
//!
//! Emitted-block surface (mirrors the retired hand-written add_opcode shape-spec):
//!   * `fn <comp>_row_body<E: WitnessEval>(eval: &mut E)` — the mechanical body.
//!   * module-private `fn write_trace_generic_simd(...)` — same signature as the writer.
//!   * `impl ClaimGenerator { pub(crate) fn write_trace_generic(...) }`.
//!   * `pub(crate) fn record_<comp>() -> RecordingOutput`.
//!   * `#[cfg(test)]` private `lookup_data_flat` / `sub_inputs_flat` +
//!     `pub(crate) struct GenericSimdDiff` + `pub(crate) fn generic_simd_diff(...)`.
//!
//! Flat-word layouts are DECLARATION ORDER (LookupData field order × widths;
//! SubComponentInputs field order × array lengths × per-field scalar shape).
//!
//! Modes:
//!   --census                parse + classify every file; print coverage + skip census.
//!   --emit-dir <dir>        write transformed full-file copies into <dir> (no in-place).
//!   --in-place              strip + re-insert the marked block in the real file.
//!   --check                 verify the on-disk block equals a freshly generated one.
//!
//! Emitted code is run through `rustfmt --edition 2021`. Iteration orders are stable
//! (BTreeMap / Vec), so output is byte-stable across runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::{
    BinOp, Expr, ExprArray, ExprAssign, ExprBinary, ExprCall, ExprField, ExprIndex,
    ExprMethodCall, ExprParen, ExprPath, ExprTuple, ExprUnary, Fields, FnArg, Ident, Item,
    ItemFn, Lit, Local, Member, Pat, Stmt, Type, UnOp,
};

/// Marker delimiting the generated block inside a component file (for idempotent re-run).
pub const BEGIN_MARKER: &str = "// === BEGIN witness_genericize (generated; re-runnable) ===";
pub const END_MARKER: &str = "// === END witness_genericize ===";

// ======================================================================================
// Types (the type map of the rewrite table)
// ======================================================================================

/// Bottom-up inferred type of a value in the per-row body's single-assignment let graph.
#[derive(Clone, Debug, PartialEq)]
enum Ty {
    M31,
    U16,
    /// u32 family — CENSUS-ONLY: typing these ops classifies files as "matched (needs
    /// u32 trait extension)"; they are never emitted.
    U32,
    Mask,
    Felt,
    ConstM31(u32),
    ConstU16(u32),
    ConstU32(u32),
    Tuple(Vec<Ty>),
    Array(Box<Ty>, usize),
    Input,
    Unknown,
}

impl Ty {
    fn is_m31(&self) -> bool {
        matches!(self, Ty::M31 | Ty::ConstM31(_))
    }
    fn is_u16(&self) -> bool {
        matches!(self, Ty::U16)
    }
    fn is_u32(&self) -> bool {
        matches!(self, Ty::U32 | Ty::ConstU32(_))
    }
    fn is_mask(&self) -> bool {
        matches!(self, Ty::Mask)
    }
}

/// A hoisted broadcast constant binding at the top of `write_trace_simd`.
#[derive(Clone, Copy, Debug)]
enum ConstKind {
    M31,
    U16,
    U32,
}

#[derive(Clone, Copy, Debug)]
struct ConstVal {
    kind: ConstKind,
    value: u32,
}

/// A shape tree of a `sub_component_inputs` assignment RHS (for flatten + reconstruction).
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    Scalar,
    Tuple(Vec<Shape>),
    Array(Vec<Shape>),
}

impl Shape {
    fn scalar_count(&self) -> usize {
        match self {
            Shape::Scalar => 1,
            Shape::Tuple(v) | Shape::Array(v) => v.iter().map(Shape::scalar_count).sum(),
        }
    }
}

/// One `(field, index)` slot of `SubComponentInputs`, with its flat base word index
/// (DECLARATION order) and value shape.
#[derive(Clone, Debug)]
struct SubSlot {
    field: String,
    index: usize,
    shape: Shape,
    base: usize,
}

/// One declared `LookupData` field: `Vec<[PackedM31; width]>` (width>1) or `Vec<PackedM31>`
/// (width==1, scalar).
#[derive(Clone, Debug)]
struct LookupField {
    name: String,
    width: usize,
    scalar: bool,
    base: usize,
}

/// A loud, quoted reason a file (or a construct in it) is not rewritable.
#[derive(Clone, Debug)]
struct Skip {
    category: &'static str,
    detail: String,
}

// ======================================================================================
// CLI
// ======================================================================================

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Census,
    EmitDir,
    InPlace,
    Check,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: witness_genericize <MODE> <file-or-dir> [more...]\n\
             MODES:\n\
             \x20 --census            parse + classify all files; print coverage + census.\n\
             \x20 --emit-dir <dir>    write transformed full-file copies into <dir>.\n\
             \x20 --in-place          strip + re-insert the marked block in the real file.\n\
             \x20 --check             verify on-disk block == freshly generated block.\n\
             A directory argument expands to its *.rs files (excluding mod.rs)."
        );
        return ExitCode::from(2);
    }

    let mut mode = Mode::Census;
    let mut emit_dir: Option<PathBuf> = None;
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--census" => mode = Mode::Census,
            "--in-place" => mode = Mode::InPlace,
            "--check" => mode = Mode::Check,
            "--emit-dir" => {
                mode = Mode::EmitDir;
                i += 1;
                if i >= args.len() {
                    eprintln!("--emit-dir requires a directory argument");
                    return ExitCode::from(2);
                }
                emit_dir = Some(PathBuf::from(&args[i]));
            }
            other if other.starts_with("--") => {
                eprintln!("unknown flag: {other}");
                return ExitCode::from(2);
            }
            other => positionals.push(other.to_string()),
        }
        i += 1;
    }

    let files = expand_files(&positionals);
    if files.is_empty() {
        eprintln!("no input .rs files found");
        return ExitCode::from(2);
    }

    match mode {
        Mode::Census => run_census(&files),
        Mode::EmitDir => run_emit_dir(&files, emit_dir.as_ref().unwrap()),
        Mode::InPlace => run_in_place(&files),
        Mode::Check => run_check(&files),
    }
}

fn expand_files(positionals: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in positionals {
        let path = PathBuf::from(p);
        if path.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&path) {
                let mut entries: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().map(|e| e == "rs").unwrap_or(false))
                    .filter(|p| p.file_name().map(|n| n != "mod.rs").unwrap_or(true))
                    .collect();
                entries.sort();
                out.extend(entries);
            }
        } else {
            out.push(path);
        }
    }
    out
}

// ======================================================================================
// File analysis
// ======================================================================================

struct FileAnalysis {
    component: String,
    has_writer: bool,
    skeleton_ok: bool,
    /// Non-empty when the whole file is skipped (skeleton-level reason).
    file_skip: Option<Skip>,
    /// Per-construct skips collected during a full walk (census backlog data).
    skips: Vec<Skip>,
    /// Fully rewritable by the emit table (skips empty AND no u32 sites).
    matched: bool,
    /// Rewrite table matches ONLY via the census-only u32 rules — needs the u32 trait
    /// extension before it can be emitted.
    matched_u32: bool,
    u32_sites: usize,
    n_cols: usize,
    n_lookup_words: usize,
    n_sub_words: usize,
    /// Deduce-output receiver -> count within this file (all writer files).
    deduce_hits: BTreeMap<String, usize>,
    /// rustfmt'd generated block (only when requested + matched).
    block: Option<String>,
}

fn analyze_file(path: &Path, build_block: bool) -> FileAnalysis {
    let component = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut fa = FileAnalysis {
        component: component.clone(),
        has_writer: false,
        skeleton_ok: false,
        file_skip: None,
        skips: Vec::new(),
        matched: false,
        matched_u32: false,
        u32_sites: 0,
        n_cols: 0,
        n_lookup_words: 0,
        n_sub_words: 0,
        deduce_hits: BTreeMap::new(),
        block: None,
    };

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            fa.file_skip = Some(Skip {
                category: "io",
                detail: format!("read: {e}"),
            });
            return fa;
        }
    };
    let file = match syn::parse_file(&src) {
        Ok(f) => f,
        Err(e) => {
            fa.file_skip = Some(Skip {
                category: "parse",
                detail: format!("syn: {e}"),
            });
            return fa;
        }
    };

    let writer = file.items.iter().find_map(|it| match it {
        Item::Fn(f) if f.sig.ident == "write_trace_simd" => Some(f),
        _ => None,
    });
    let Some(writer) = writer else {
        fa.file_skip = Some(Skip {
            category: "skeleton",
            detail: "no `fn write_trace_simd`".to_string(),
        });
        return fa;
    };
    fa.has_writer = true;

    // Global deduce-output census (independent of skeleton match).
    {
        let mut v = DeduceVisitor { hits: Vec::new() };
        syn::visit::Visit::visit_item_fn(&mut v, writer);
        for r in v.hits {
            *fa.deduce_hits.entry(r).or_insert(0) += 1;
        }
    }

    // The mem-state param idents (for deduce_output receiver matching).
    let (addr_state, big_state) = mem_state_param_names(writer);

    // Collect hoisted constants + locate the `for_each` closure.
    let mut consts: BTreeMap<String, ConstVal> = BTreeMap::new();
    for st in &writer.block.stmts {
        if let Stmt::Local(local) = st {
            if let (Some(name), Some(cv)) = (local_ident(local), local_const(local)) {
                consts.insert(name, cv);
            }
        }
    }

    let Some(closure) = find_for_each_closure(writer) else {
        fa.file_skip = Some(Skip {
            category: "skeleton",
            detail: "no `.for_each(|(row_index, (...))| {..})` closure".to_string(),
        });
        return fa;
    };
    let binders = match closure_binders(&closure.inputs) {
        Some(b) => b,
        None => {
            fa.file_skip = Some(Skip {
                category: "skeleton",
                detail: format!("unrecognized closure binder: `{}`", tok_str(&closure.inputs[0])),
            });
            return fa;
        }
    };
    if binders.len() != 4 {
        fa.file_skip = Some(Skip {
            category: "skeleton",
            detail: format!(
                "unsupported skeleton: {}-tuple closure `({})` (preprocessed/table \
                 iterate or no per-row input)",
                binders.len(),
                binders.join(", ")
            ),
        });
        return fa;
    }
    let row_name = binders[0].clone();
    let lookup_name = binders[1].clone();
    let sub_name = binders[2].clone();
    let input_name = binders[3].clone();
    if !input_name.ends_with("_input") {
        fa.file_skip = Some(Skip {
            category: "skeleton",
            detail: format!("4th closure binder `{input_name}` is not `<name>_input`"),
        });
        return fa;
    }
    fa.skeleton_ok = true;

    // Parse the LookupData layout (declaration order).
    let lookup_fields = match parse_lookup_data(&file) {
        Ok(f) => f,
        Err(s) => {
            fa.file_skip = Some(s);
            return fa;
        }
    };
    fa.n_lookup_words = lookup_fields.iter().map(|f| f.width).sum();

    // Closure body statements.
    let body_stmts: &[Stmt] = match &*closure.body {
        Expr::Block(b) => &b.block.stmts,
        _ => {
            fa.file_skip = Some(Skip {
                category: "skeleton",
                detail: "closure body is not a block".to_string(),
            });
            return fa;
        }
    };

    // Derive the SubComponentInputs DECLARATION-ORDER flat layout: struct fields ×
    // array lengths × the value shape observed at this file's assignment sites.
    let sub_slots = match build_sub_layout(&file, body_stmts, &sub_name) {
        Ok(l) => l,
        Err(s) => {
            fa.file_skip = Some(s);
            return fa;
        }
    };
    fa.n_sub_words = sub_slots.iter().map(|s| s.shape.scalar_count()).sum();

    // Run the lowering (collects skips + builds SSA).
    let mut lw = Lowerer::new(
        consts,
        addr_state,
        big_state,
        input_name.clone(),
        row_name,
        lookup_name,
        sub_name,
        lookup_fields.clone(),
        sub_slots,
    );
    lw.lower_body(body_stmts);

    fa.n_cols = lw.max_col.map(|m| m + 1).unwrap_or(0);
    fa.u32_sites = lw.u32_sites;
    fa.skips = lw.skips.clone();
    fa.matched = fa.skeleton_ok && fa.skips.is_empty() && lw.u32_sites == 0;
    fa.matched_u32 = fa.skeleton_ok && fa.skips.is_empty() && lw.u32_sites > 0;

    if fa.matched && build_block {
        let block = build_marked_block(&component, &fa, &lw, writer, &file);
        fa.block = Some(rustfmt_block(&block));
    }

    fa
}

/// Extract the `&memory_address_to_id::ClaimGenerator` / `&memory_id_to_big::ClaimGenerator`
/// parameter identifiers from the `write_trace_simd` signature.
fn mem_state_param_names(f: &ItemFn) -> (Option<String>, Option<String>) {
    let mut addr = None;
    let mut big = None;
    for arg in &f.sig.inputs {
        if let FnArg::Typed(pt) = arg {
            let tystr = tok_str(&pt.ty);
            let name = match &*pt.pat {
                Pat::Ident(pi) => pi.ident.to_string(),
                _ => continue,
            };
            if tystr.contains("memory_address_to_id :: ClaimGenerator") {
                addr = Some(name);
            } else if tystr.contains("memory_id_to_big :: ClaimGenerator") {
                big = Some(name);
            }
        }
    }
    (addr, big)
}

fn find_for_each_closure(f: &ItemFn) -> Option<syn::ExprClosure> {
    for st in &f.block.stmts {
        let expr = match st {
            Stmt::Expr(e, _) => e,
            Stmt::Local(l) => match &l.init {
                Some(init) => &init.expr,
                None => continue,
            },
            _ => continue,
        };
        if let Some(c) = search_for_each(expr) {
            return Some(c);
        }
    }
    None
}

fn search_for_each(expr: &Expr) -> Option<syn::ExprClosure> {
    if let Expr::MethodCall(mc) = expr {
        if mc.method == "for_each" {
            if let Some(Expr::Closure(c)) = mc.args.first() {
                return Some(c.clone());
            }
        }
        return search_for_each(&mc.receiver);
    }
    None
}

/// Match `|(row_index, (a, b, c, d))|` → returns the inner binder names [a, b, c, d].
fn closure_binders(inputs: &syn::punctuated::Punctuated<Pat, syn::token::Comma>) -> Option<Vec<String>> {
    let first = inputs.first()?;
    let outer = match first {
        Pat::Tuple(t) => t,
        _ => return None,
    };
    if outer.elems.len() != 2 {
        return None;
    }
    // outer.elems[0] is row_index; outer.elems[1] is the inner tuple.
    let inner = match &outer.elems[1] {
        Pat::Tuple(t) => t,
        _ => return None,
    };
    let mut names = Vec::new();
    for e in &inner.elems {
        match e {
            Pat::Ident(pi) => names.push(pi.ident.to_string()),
            _ => return None,
        }
    }
    Some(names)
}

fn parse_lookup_data(file: &syn::File) -> Result<Vec<LookupField>, Skip> {
    let st = file.items.iter().find_map(|it| match it {
        Item::Struct(s) if s.ident == "LookupData" => Some(s),
        _ => None,
    });
    let Some(st) = st else {
        return Err(Skip {
            category: "skeleton",
            detail: "no `struct LookupData`".to_string(),
        });
    };
    let named = match &st.fields {
        Fields::Named(n) => n,
        _ => {
            return Err(Skip {
                category: "skeleton",
                detail: "LookupData is not a named struct".to_string(),
            })
        }
    };
    let mut fields = Vec::new();
    let mut base = 0usize;
    for f in &named.named {
        let name = f.ident.as_ref().unwrap().to_string();
        let (width, scalar) = lookup_field_width(&f.ty).ok_or_else(|| Skip {
            category: "skeleton",
            detail: format!("LookupData.{name}: unrecognized field type `{}`", tok_str(&f.ty)),
        })?;
        fields.push(LookupField {
            name,
            width,
            scalar,
            base,
        });
        base += width;
    }
    Ok(fields)
}

/// `Vec<[PackedM31; N]>` → (N, false); `Vec<PackedM31>` → (1, true).
fn lookup_field_width(ty: &Type) -> Option<(usize, bool)> {
    let s = tok_str(ty);
    if let Some(rest) = s.strip_prefix("Vec < [PackedM31 ;") {
        let n: usize = rest.trim().trim_end_matches("] >").trim().parse().ok()?;
        return Some((n, false));
    }
    if s == "Vec < PackedM31 >" {
        return Some((1, true));
    }
    None
}

/// Parse `struct SubComponentInputs` field declarations: (name, array_len) in order.
/// Field types are `[Vec<...>; N]`.
fn parse_sub_struct(file: &syn::File) -> Result<Vec<(String, usize)>, Skip> {
    let st = file.items.iter().find_map(|it| match it {
        Item::Struct(s) if s.ident == "SubComponentInputs" => Some(s),
        _ => None,
    });
    let Some(st) = st else {
        return Err(Skip {
            category: "skeleton",
            detail: "no `struct SubComponentInputs`".to_string(),
        });
    };
    let named = match &st.fields {
        Fields::Named(n) => n,
        _ => {
            return Err(Skip {
                category: "skeleton",
                detail: "SubComponentInputs is not a named struct".to_string(),
            })
        }
    };
    let mut out = Vec::new();
    for f in &named.named {
        let name = f.ident.as_ref().unwrap().to_string();
        let Type::Array(arr) = &f.ty else {
            return Err(Skip {
                category: "skeleton",
                detail: format!(
                    "SubComponentInputs.{name}: not an array type `{}`",
                    tok_str(&f.ty)
                ),
            });
        };
        let Some(len) = expr_usize(&arr.len) else {
            return Err(Skip {
                category: "skeleton",
                detail: format!("SubComponentInputs.{name}: non-literal array length"),
            });
        };
        out.push((name, len));
    }
    Ok(out)
}

/// Syntactic value shape of a sub-input assignment RHS (tuple/array nesting only —
/// leaves are scalars; type-checking of the leaves happens during lowering).
fn syntactic_shape(expr: &Expr) -> Shape {
    match strip_parens(expr) {
        Expr::Tuple(ExprTuple { elems, .. }) => {
            Shape::Tuple(elems.iter().map(syntactic_shape).collect())
        }
        Expr::Array(ExprArray { elems, .. }) => {
            Shape::Array(elems.iter().map(syntactic_shape).collect())
        }
        _ => Shape::Scalar,
    }
}

/// Pre-scan the closure body's top-level statements for
/// `*<sub_name>.<field>[k] = rhs;` and derive the DECLARATION-ORDER flat layout.
fn build_sub_layout(
    file: &syn::File,
    body_stmts: &[Stmt],
    sub_name: &str,
) -> Result<Vec<SubSlot>, Skip> {
    // Collect (field, k, shape) at every assignment site (file order).
    let mut seen: BTreeMap<(String, usize), Shape> = BTreeMap::new();
    for st in body_stmts {
        let Stmt::Expr(Expr::Assign(a), _) = st else {
            continue;
        };
        let Expr::Unary(ExprUnary {
            op: UnOp::Deref(_),
            expr: place,
            ..
        }) = strip_parens(&a.left)
        else {
            continue;
        };
        let Expr::Index(ExprIndex { expr: base, index, .. }) = strip_parens(place) else {
            continue;
        };
        let Expr::Field(ExprField {
            base: fb,
            member: Member::Named(m),
            ..
        }) = strip_parens(base)
        else {
            continue;
        };
        if !is_path_named(fb, sub_name) {
            continue;
        }
        let field = m.to_string();
        let Some(k) = expr_usize(index) else {
            return Err(Skip {
                category: "effect",
                detail: format!("sub-input index not a literal: `{}`", tok_str(index)),
            });
        };
        let shape = syntactic_shape(&a.right);
        if seen.insert((field.clone(), k), shape).is_some() {
            return Err(Skip {
                category: "effect",
                detail: format!("sub-input `{field}[{k}]` assigned more than once"),
            });
        }
    }

    if seen.is_empty() {
        // No sub-input writes in this body (some components have an empty struct).
        return Ok(Vec::new());
    }

    let decl = parse_sub_struct(file)?;
    // Every observed field must be declared; every declared (field,k) must be assigned.
    let declared: BTreeSet<&String> = decl.iter().map(|(n, _)| n).collect();
    for (field, k) in seen.keys() {
        if !declared.contains(field) {
            return Err(Skip {
                category: "effect",
                detail: format!("sub-input `{field}[{k}]` not declared in SubComponentInputs"),
            });
        }
    }
    let mut slots = Vec::new();
    let mut base = 0usize;
    for (field, len) in &decl {
        for k in 0..*len {
            let Some(shape) = seen.get(&(field.clone(), k)) else {
                return Err(Skip {
                    category: "effect",
                    detail: format!("sub-input `{field}[{k}]` declared but never assigned"),
                });
            };
            let count = shape.scalar_count();
            slots.push(SubSlot {
                field: field.clone(),
                index: k,
                shape: shape.clone(),
                base,
            });
            base += count;
        }
    }
    Ok(slots)
}

// ======================================================================================
// Deduce-output census visitor
// ======================================================================================

struct DeduceVisitor {
    hits: Vec<String>,
}

impl<'ast> syn::visit::Visit<'ast> for DeduceVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if node.method == "deduce_output" {
            self.hits.push(receiver_label(&node.receiver));
        }
        syn::visit::visit_expr_method_call(self, node);
    }
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(p) = &*node.func {
            if let Some(last) = p.path.segments.last() {
                if last.ident == "deduce_output" {
                    let segs: Vec<String> = p
                        .path
                        .segments
                        .iter()
                        .take(p.path.segments.len() - 1)
                        .map(|s| s.ident.to_string())
                        .collect();
                    self.hits.push(format!("{}::deduce_output", segs.join("::")));
                }
            }
        }
        syn::visit::visit_expr_call(self, node);
    }
}

fn receiver_label(recv: &Expr) -> String {
    match strip_parens(recv) {
        Expr::Path(p) => format!("{}.deduce_output", tok_str(&p.path)),
        other => format!("{}.deduce_output", tok_str(other)),
    }
}

// ======================================================================================
// Lowerer — the finite rewrite table (SSA flattening + type inference + effects)
// ======================================================================================

enum Target {
    Temp,
    Named(Ident),
}

struct Lowerer {
    consts: BTreeMap<String, ConstVal>,
    addr_state: Option<String>,
    big_state: Option<String>,
    input_name: String,
    row_name: String,
    lookup_name: String,
    sub_name: String,
    lookup_fields: Vec<LookupField>,
    /// Declaration-order sub-input slots ((field, k) → flat base).
    sub_slots: Vec<SubSlot>,
    sub_base: BTreeMap<(String, usize), usize>,

    env: BTreeMap<String, Ty>,
    out: Vec<TokenStream>,
    referenced_m31: BTreeSet<u32>,
    used_slots: BTreeSet<&'static str>,
    skips: Vec<Skip>,
    u32_sites: usize,
    counter: usize,

    max_col: Option<usize>,
}

impl Lowerer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        consts: BTreeMap<String, ConstVal>,
        addr_state: Option<String>,
        big_state: Option<String>,
        input_name: String,
        row_name: String,
        lookup_name: String,
        sub_name: String,
        lookup_fields: Vec<LookupField>,
        sub_slots: Vec<SubSlot>,
    ) -> Self {
        let sub_base = sub_slots
            .iter()
            .map(|s| ((s.field.clone(), s.index), s.base))
            .collect();
        Self {
            consts,
            addr_state,
            big_state,
            input_name,
            row_name,
            lookup_name,
            sub_name,
            lookup_fields,
            sub_slots,
            sub_base,
            env: BTreeMap::new(),
            out: Vec::new(),
            referenced_m31: BTreeSet::new(),
            used_slots: BTreeSet::new(),
            skips: Vec::new(),
            u32_sites: 0,
            counter: 0,
            max_col: None,
        }
    }

    fn skip(&mut self, category: &'static str, detail: String) {
        self.skips.push(Skip { category, detail });
    }

    /// Census-only u32-family site: type-checks (so inference proceeds) but the file is
    /// classified "matched (needs u32 trait extension)" and never emitted.
    fn u32_site(&mut self, ty: Ty) -> (Ty, TokenStream) {
        self.u32_sites += 1;
        (ty, quote! { WG_U32_CENSUS_ONLY })
    }

    fn fresh(&mut self) -> Ident {
        let id = Ident::new(&format!("wg_v{}", self.counter), Span::call_site());
        self.counter += 1;
        id
    }

    /// Bind `rhs` to `target` (Named or a fresh temp); push the `let`, return the value tok.
    fn bind(&mut self, target: Target, rhs: TokenStream) -> TokenStream {
        let name = match target {
            Target::Named(n) => n,
            Target::Temp => self.fresh(),
        };
        self.out.push(quote! { let #name = #rhs; });
        quote! { #name }
    }

    // ---- statement-level -----------------------------------------------------------

    fn lower_body(&mut self, stmts: &[Stmt]) {
        for st in stmts {
            self.lower_stmt(st);
        }
    }

    fn lower_stmt(&mut self, st: &Stmt) {
        match st {
            Stmt::Local(local) => self.lower_local(local),
            Stmt::Expr(Expr::Assign(a), _) => self.lower_assign(a),
            Stmt::Expr(e, _) => {
                self.skip("stmt", format!("unexpected expression statement: `{}`", tok_str(e)));
            }
            Stmt::Macro(m) => {
                self.skip("macro", format!("macro in body: `{}`", tok_str(&m.mac.path)));
            }
            Stmt::Item(_) => self.skip("stmt", "nested item in body".to_string()),
        }
    }

    fn lower_local(&mut self, local: &Local) {
        let Some(name) = local_ident(local) else {
            self.skip("stmt", format!("unsupported `let` pattern: `{}`", tok_str(&local.pat)));
            return;
        };
        let Some(init) = &local.init else {
            self.skip("stmt", format!("`let {name}` without initializer"));
            return;
        };
        let name_ident = Ident::new(&name, Span::call_site());
        let expr = strip_parens(&init.expr);
        let ty = match expr {
            Expr::Tuple(_) | Expr::Array(_) => {
                let (ty, toks) = self.lower_aggregate(expr);
                self.out.push(quote! { let #name_ident = #toks; });
                ty
            }
            _ => {
                let (ty, _tok) = self.lower_node(expr, Target::Named(name_ident));
                ty
            }
        };
        self.env.insert(name, ty);
    }

    fn lower_assign(&mut self, a: &ExprAssign) {
        // LHS must be `*<place>`.
        let deref = match strip_parens(&a.left) {
            Expr::Unary(ExprUnary {
                op: UnOp::Deref(_),
                expr,
                ..
            }) => strip_parens(expr),
            other => {
                self.skip("effect", format!("assignment to non-deref place: `{}`", tok_str(other)));
                return;
            }
        };
        match deref {
            // *row[i] = v;
            Expr::Index(ExprIndex { expr: base, index, .. })
                if is_path_named(base, &self.row_name) =>
            {
                let Some(col) = expr_usize(index) else {
                    self.skip("effect", format!("row index not a literal: `{}`", tok_str(index)));
                    return;
                };
                let (ty, v) = self.lower_node(strip_parens(&a.right), Target::Temp);
                self.require_m31(&ty, "set_col value", &a.right);
                let cl = usize_lit(col);
                self.out.push(quote! { eval.set_col(#cl, #v); });
                self.max_col = Some(self.max_col.map_or(col, |m| m.max(col)));
            }
            // *sub_component_inputs.field[k] = <tuple/array/scalar>;
            Expr::Index(ExprIndex { expr: base, index, .. }) => {
                let field = match strip_parens(base) {
                    Expr::Field(ExprField {
                        base: fb,
                        member: Member::Named(m),
                        ..
                    }) if is_path_named(fb, &self.sub_name) => m.to_string(),
                    _ => {
                        self.skip("effect", format!("unrecognized sub-input place: `{}`", tok_str(deref)));
                        return;
                    }
                };
                let Some(k) = expr_usize(index) else {
                    self.skip("effect", format!("sub-input index not a literal: `{}`", tok_str(index)));
                    return;
                };
                let Some(base_idx) = self.sub_base.get(&(field.clone(), k)).copied() else {
                    self.skip("effect", format!("sub-input `{field}[{k}]` missing from layout"));
                    return;
                };
                let leaves = self.flatten_sub(strip_parens(&a.right));
                for (j, leaf) in leaves.iter().enumerate() {
                    let w = usize_lit(base_idx + j);
                    self.out.push(quote! { eval.set_sub_input_word(#w, #leaf); });
                }
            }
            // *lookup_data.field = <array or scalar>;
            Expr::Field(ExprField {
                base,
                member: Member::Named(m),
                ..
            }) if is_path_named(base, &self.lookup_name) => {
                let field = m.to_string();
                let Some(lf) = self.lookup_fields.iter().find(|f| f.name == field).cloned() else {
                    self.skip("effect", format!("lookup field not in LookupData: `{field}`"));
                    return;
                };
                let rhs = strip_parens(&a.right);
                if lf.scalar {
                    let (ty, v) = self.lower_node(rhs, Target::Temp);
                    self.require_m31(&ty, "lookup word", rhs);
                    let w = usize_lit(lf.base);
                    self.out.push(quote! { eval.set_lookup_word(#w, #v); });
                } else {
                    let elems = match rhs {
                        Expr::Array(ExprArray { elems, .. }) => elems,
                        _ => {
                            self.skip(
                                "effect",
                                format!("lookup field `{field}` RHS not an array: `{}`", tok_str(rhs)),
                            );
                            return;
                        }
                    };
                    if elems.len() != lf.width {
                        self.skip(
                            "effect",
                            format!(
                                "lookup field `{field}` width {} != RHS len {}",
                                lf.width,
                                elems.len()
                            ),
                        );
                        return;
                    }
                    for (j, e) in elems.iter().enumerate() {
                        let (ty, v) = self.lower_node(strip_parens(e), Target::Temp);
                        self.require_m31(&ty, "lookup word", e);
                        let w = usize_lit(lf.base + j);
                        self.out.push(quote! { eval.set_lookup_word(#w, #v); });
                    }
                }
            }
            other => {
                self.skip("effect", format!("unrecognized effect place: `{}`", tok_str(other)));
            }
        }
    }

    /// Effect values must be M31-typed (Unknown means an inner skip already fired).
    fn require_m31(&mut self, ty: &Ty, what: &str, expr: &Expr) {
        if !ty.is_m31() && *ty != Ty::Unknown {
            self.skip("effect", format!("{what} is {ty:?}, not M31: `{}`", tok_str(expr)));
        }
    }

    // ---- aggregate (kept-verbatim tuples/arrays) -----------------------------------

    fn lower_aggregate(&mut self, expr: &Expr) -> (Ty, TokenStream) {
        match strip_parens(expr) {
            Expr::Tuple(ExprTuple { elems, .. }) => {
                let mut tys = Vec::new();
                let mut toks = Vec::new();
                for e in elems {
                    let (t, k) = self.lower_agg_elem(e);
                    tys.push(t);
                    toks.push(k);
                }
                (Ty::Tuple(tys), quote! { ( #(#toks),* ) })
            }
            Expr::Array(ExprArray { elems, .. }) => {
                let mut tys = Vec::new();
                let mut toks = Vec::new();
                for e in elems {
                    let (t, k) = self.lower_agg_elem(e);
                    tys.push(t);
                    toks.push(k);
                }
                let et = tys.first().cloned().unwrap_or(Ty::Unknown);
                (Ty::Array(Box::new(et), toks.len()), quote! { [ #(#toks),* ] })
            }
            other => self.lower_node(other, Target::Temp),
        }
    }

    fn lower_agg_elem(&mut self, e: &Expr) -> (Ty, TokenStream) {
        match strip_parens(e) {
            Expr::Tuple(_) | Expr::Array(_) => self.lower_aggregate(e),
            other => self.lower_node(other, Target::Temp),
        }
    }

    // ---- expression-level (the op routing table) -----------------------------------

    /// Lower `expr`, emitting SSA temps for every eval op. Returns the inferred type and a
    /// SIMPLE token (ident / literal / projection) naming the value.
    fn lower_node(&mut self, expr: &Expr, target: Target) -> (Ty, TokenStream) {
        let expr = strip_parens(expr);
        match expr {
            Expr::Path(p) => self.lower_path(p, target),
            Expr::Field(f) => self.lower_field(f, target),
            Expr::Index(ix) => self.lower_index(ix, target),
            Expr::MethodCall(mc) => self.lower_method(mc, target),
            Expr::Call(call) => self.lower_call(call, target),
            Expr::Binary(b) => self.lower_binary(b, target),
            other => {
                self.skip("expr", format!("unsupported expression: `{}`", tok_str(other)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
        }
    }

    /// Leaf value: return its token, aliasing into `target` only when Named.
    fn leaf(&mut self, target: Target, ty: Ty, tok: TokenStream) -> (Ty, TokenStream) {
        match target {
            Target::Named(_) => {
                let out = self.bind(target, tok);
                (ty, out)
            }
            Target::Temp => (ty, tok),
        }
    }

    fn lower_path(&mut self, p: &ExprPath, target: Target) -> (Ty, TokenStream) {
        let name = tok_str(&p.path);
        if let Some(cv) = self.classify_const(&name) {
            match cv.kind {
                ConstKind::M31 => {
                    self.referenced_m31.insert(cv.value);
                    let id = Ident::new(&format!("m31_{}", cv.value), Span::call_site());
                    return self.leaf(target, Ty::ConstM31(cv.value), quote! { #id });
                }
                ConstKind::U16 => {
                    // A u16 const as a bare value only occurs in shift/mask (peeked) or
                    // additive (specially lowered) position — never materialized here.
                    return (Ty::ConstU16(cv.value), quote! { WG_U16_CONST });
                }
                ConstKind::U32 => {
                    return (Ty::ConstU32(cv.value), quote! { WG_U32_CONST });
                }
            }
        }
        if name == self.input_name {
            self.skip("expr", format!("bare use of input struct `{name}`"));
            return (Ty::Input, quote! { WG_SKIP });
        }
        if let Some(ty) = self.env.get(&name).cloned() {
            let id = Ident::new(&name, Span::call_site());
            return self.leaf(target, ty, quote! { #id });
        }
        self.skip("expr", format!("unknown identifier `{name}`"));
        (Ty::Unknown, quote! { WG_SKIP })
    }

    fn lower_field(&mut self, f: &ExprField, target: Target) -> (Ty, TokenStream) {
        match &f.member {
            Member::Named(m) => {
                // <name>_input.pc / .ap / .fp
                if is_path_named(&f.base, &self.input_name) {
                    let slot: &'static str = match m.to_string().as_str() {
                        "pc" => "SLOT_PC",
                        "ap" => "SLOT_AP",
                        "fp" => "SLOT_FP",
                        other => {
                            self.skip("input_field", format!("input.{other} (unsupported input field)"));
                            return (Ty::Unknown, quote! { WG_SKIP });
                        }
                    };
                    self.used_slots.insert(slot);
                    let slot_id = Ident::new(slot, Span::call_site());
                    return self.emit_op(target, Ty::M31, quote! { eval.input(#slot_id) });
                }
                self.skip("expr", format!("field access `.{m}` on non-input base `{}`", tok_str(&f.base)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
            Member::Unnamed(idx) => {
                // Tuple projection x.0 / x.1 ...
                let (bt, btok) = self.lower_node(&f.base, Target::Temp);
                let i = idx.index as usize;
                let elem_ty = match &bt {
                    Ty::Tuple(v) if i < v.len() => v[i].clone(),
                    _ => {
                        self.skip("expr", format!("tuple projection .{i} on non-tuple `{}`", tok_str(&f.base)));
                        Ty::Unknown
                    }
                };
                let lit = Literal::usize_unsuffixed(i);
                self.leaf(target, elem_ty, quote! { #btok.#lit })
            }
        }
    }

    fn lower_index(&mut self, ix: &ExprIndex, target: Target) -> (Ty, TokenStream) {
        let (bt, btok) = self.lower_node(&ix.expr, Target::Temp);
        let Some(i) = expr_usize(&ix.index) else {
            self.skip("expr", format!("non-literal index: `{}`", tok_str(&ix.index)));
            return (Ty::Unknown, quote! { WG_SKIP });
        };
        let elem_ty = match &bt {
            Ty::Array(e, _) => (**e).clone(),
            _ => {
                self.skip("expr", format!("index [{i}] on non-array `{}`", tok_str(&ix.expr)));
                Ty::Unknown
            }
        };
        let lit = usize_lit(i);
        self.leaf(target, elem_ty, quote! { #btok[#lit] })
    }

    fn lower_method(&mut self, mc: &ExprMethodCall, target: Target) -> (Ty, TokenStream) {
        let method = mc.method.to_string();
        match method.as_str() {
            "get_m31" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                let Some(i) = mc.args.first().and_then(expr_usize) else {
                    self.skip("expr", "get_m31 without literal index".to_string());
                    return (Ty::Unknown, quote! { WG_SKIP });
                };
                if rt != Ty::Felt {
                    self.skip("expr", format!("get_m31 on non-Felt `{}`", tok_str(&mc.receiver)));
                }
                let lit = usize_lit(i);
                self.emit_op(target, Ty::M31, quote! { eval.felt_get_m31(&#rtok, #lit) })
            }
            "as_m31" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                if rt.is_u16() {
                    self.emit_op(target, Ty::M31, quote! { eval.u16_as_m31(#rtok) })
                } else if rt.is_mask() {
                    self.emit_op(target, Ty::M31, quote! { eval.mask_as_m31(#rtok) })
                } else {
                    self.skip("expr", format!("as_m31 on {:?} `{}`", rt, tok_str(&mc.receiver)));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            // u32 family (census-only): .low()/.high() split a u32 into u16 halves.
            "low" | "high" => {
                let (rt, _rtok) = self.lower_node(&mc.receiver, Target::Temp);
                if rt.is_u32() {
                    self.u32_site(Ty::U16)
                } else {
                    self.skip("method", format!(".{method}() on `{}`", tok_str(&mc.receiver)));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            "eq" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                let (_at, atok) = self.lower_arg(mc.args.first());
                if !rt.is_m31() {
                    self.skip("expr", format!("eq on non-M31 `{}`", tok_str(&mc.receiver)));
                }
                self.emit_op(target, Ty::Mask, quote! { eval.m31_eq(#rtok, #atok) })
            }
            "inverse" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                if !rt.is_m31() {
                    self.skip("expr", format!("inverse on non-M31 `{}`", tok_str(&mc.receiver)));
                }
                self.emit_op(target, Ty::M31, quote! { eval.m31_inverse(#rtok) })
            }
            "deduce_output" => {
                let recv = tok_str(strip_parens(&mc.receiver));
                let (_at, atok) = self.lower_arg(mc.args.first());
                if Some(&recv) == self.addr_state.as_ref() {
                    self.emit_op(target, Ty::M31, quote! { eval.mem_addr_to_id(#atok) })
                } else if Some(&recv) == self.big_state.as_ref() {
                    self.emit_op(target, Ty::Felt, quote! { eval.mem_id_to_value(#atok) })
                } else {
                    self.skip("deduce_output", format!("{recv}.deduce_output"));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            "packed_at" => {
                if is_path_named(&mc.receiver, "enabler_col") {
                    self.emit_op(target, Ty::M31, quote! { eval.enabler() })
                } else {
                    // preprocessed column .packed_at(row_index) etc.
                    self.skip("method", format!("{}.packed_at (non-enabler)", tok_str(&mc.receiver)));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            other => {
                // Recurse into args for census completeness, then skip.
                for a in &mc.args {
                    let _ = self.lower_node(a, Target::Temp);
                }
                self.skip("method", format!(".{other}() on `{}`", tok_str(&mc.receiver)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
        }
    }

    fn lower_call(&mut self, call: &ExprCall, target: Target) -> (Ty, TokenStream) {
        let path = match &*call.func {
            Expr::Path(p) => tok_str(&p.path),
            other => {
                self.skip("call", format!("call of non-path `{}`", tok_str(other)));
                return (Ty::Unknown, quote! { WG_SKIP });
            }
        };
        match path.as_str() {
            "PackedUInt16 :: from_m31" => {
                let (_t, a) = self.lower_arg(call.args.first());
                self.emit_op(target, Ty::U16, quote! { eval.u16_from_m31(#a) })
            }
            "PackedBool :: from_m31" => {
                let (_t, a) = self.lower_arg(call.args.first());
                self.emit_op(target, Ty::Mask, quote! { eval.mask_from_m31(#a) })
            }
            "PackedFelt252 :: from_limbs" => {
                // Single array argument of 28 M31 exprs.
                let arr = match call.args.first().map(strip_parens) {
                    Some(Expr::Array(ExprArray { elems, .. })) => elems,
                    _ => {
                        self.skip("call", "from_limbs without array arg".to_string());
                        return (Ty::Unknown, quote! { WG_SKIP });
                    }
                };
                let mut toks = Vec::new();
                for e in arr {
                    let (_t, k) = self.lower_node(strip_parens(e), Target::Temp);
                    toks.push(k);
                }
                self.emit_op(target, Ty::Felt, quote! { eval.felt_from_limbs([ #(#toks),* ]) })
            }
            // u32 family (census-only).
            "PackedUInt32 :: from_m31" => {
                let (_t, _a) = self.lower_arg(call.args.first());
                self.u32_site(Ty::U32)
            }
            "PackedUInt32 :: from_limbs" => {
                if let Some(Expr::Array(ExprArray { elems, .. })) =
                    call.args.first().map(strip_parens)
                {
                    for e in elems {
                        let _ = self.lower_node(strip_parens(e), Target::Temp);
                    }
                }
                self.u32_site(Ty::U32)
            }
            p if p.ends_with(":: deduce_output") => {
                for a in &call.args {
                    let _ = self.lower_node(a, Target::Temp);
                }
                self.skip("deduce_output", p.to_string());
                (Ty::Unknown, quote! { WG_SKIP })
            }
            other => {
                for a in &call.args {
                    let _ = self.lower_node(a, Target::Temp);
                }
                self.skip("call", format!("call `{other}(..)`"));
                (Ty::Unknown, quote! { WG_SKIP })
            }
        }
    }

    fn lower_binary(&mut self, b: &ExprBinary, target: Target) -> (Ty, TokenStream) {
        match b.op {
            BinOp::Shl(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                if let Some(k) = self.peek_const_u16(&b.right) {
                    if lt.is_u16() {
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U16, quote! { eval.u16_shl(#ltok, #kl) });
                    }
                    self.skip("binop", format!("`<<` on non-U16 `{}`", tok_str(&b.left)));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if self.peek_const_u32(&b.right).is_some() {
                    if lt.is_u32() {
                        return self.u32_site(Ty::U32);
                    }
                    self.skip("binop", format!("`<<` (u32) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let (rt, _rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip("binop", format!("`<<` by non-const `{}`", tok_str(&b.right)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
            BinOp::Shr(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                if let Some(k) = self.peek_const_u16(&b.right) {
                    if lt.is_u16() {
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U16, quote! { eval.u16_shr(#ltok, #kl) });
                    }
                    self.skip("binop", format!("`>>` on non-U16 `{}`", tok_str(&b.left)));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if self.peek_const_u32(&b.right).is_some() {
                    if lt.is_u32() {
                        return self.u32_site(Ty::U32);
                    }
                    self.skip("binop", format!("`>>` (u32) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let (rt, _rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip("binop", format!("`>>` by non-const `{}`", tok_str(&b.right)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
            BinOp::BitAnd(_) => {
                // `&` is either `u16 & CONST_MASK` (const on EITHER side — AND commutes),
                // `mask & mask`, or the census-only u32 form.
                if let Some(k) = self.peek_const_u16(&b.right) {
                    let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                    if lt.is_u16() {
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U16, quote! { eval.u16_and(#ltok, #kl) });
                    }
                    self.skip("binop", format!("`&` (mask) on non-U16 `{}`", tok_str(&b.left)));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if let Some(k) = self.peek_const_u16(&b.left) {
                    let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                    if rt.is_u16() {
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U16, quote! { eval.u16_and(#rtok, #kl) });
                    }
                    self.skip("binop", format!("`&` (mask) on non-U16 `{}`", tok_str(&b.right)));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if self.peek_const_u32(&b.right).is_some() {
                    let (lt, _ltok) = self.lower_node(&b.left, Target::Temp);
                    if lt.is_u32() {
                        return self.u32_site(Ty::U32);
                    }
                    self.skip("binop", format!("`&` (u32 mask) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if self.peek_const_u32(&b.left).is_some() {
                    let (rt, _rtok) = self.lower_node(&b.right, Target::Temp);
                    if rt.is_u32() {
                        return self.u32_site(Ty::U32);
                    }
                    self.skip("binop", format!("`&` (u32 mask) on {:?}", rt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_mask() && rt.is_mask() {
                    return self.emit_op(target, Ty::Mask, quote! { eval.mask_and(#ltok, #rtok) });
                }
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip("binop", format!("`&` on {:?}/{:?}", lt, rt));
                (Ty::Unknown, quote! { WG_SKIP })
            }
            BinOp::BitXor(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_u16() && rt.is_u16() {
                    return self.emit_op(target, Ty::U16, quote! { eval.u16_xor(#ltok, #rtok) });
                }
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip("binop", format!("`^` on {:?}/{:?}", lt, rt));
                (Ty::Unknown, quote! { WG_SKIP })
            }
            BinOp::Add(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_m31() && rt.is_m31() {
                    self.emit_op(target, Ty::M31, quote! { eval.m31_add(#ltok, #rtok) })
                } else if lt.is_u16() && rt.is_u16() {
                    self.emit_op(target, Ty::U16, quote! { eval.u16_add(#ltok, #rtok) })
                } else if lt.is_u16() {
                    if let Ty::ConstU16(k) = rt {
                        let c = self.materialize_u16_const(k);
                        self.emit_op(target, Ty::U16, quote! { eval.u16_add(#ltok, #c) })
                    } else {
                        self.skip("binop", format!("`+` on U16/{:?}", rt));
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                } else if rt.is_u16() {
                    if let Ty::ConstU16(k) = lt {
                        let c = self.materialize_u16_const(k);
                        self.emit_op(target, Ty::U16, quote! { eval.u16_add(#c, #rtok) })
                    } else {
                        self.skip("binop", format!("`+` on {:?}/U16", lt));
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                } else if lt.is_u32() && rt.is_u32() {
                    self.u32_site(Ty::U32)
                } else {
                    self.skip("binop", format!("`+` on {:?}/{:?}", lt, rt));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            BinOp::Sub(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_m31() && rt.is_m31() {
                    self.emit_op(target, Ty::M31, quote! { eval.m31_sub(#ltok, #rtok) })
                } else if lt.is_u32() && rt.is_u32() {
                    self.u32_site(Ty::U32)
                } else {
                    self.skip("binop", format!("`-` on {:?}/{:?}", lt, rt));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            BinOp::Mul(_) => {
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_m31() && rt.is_m31() {
                    self.emit_op(target, Ty::M31, quote! { eval.m31_mul(#ltok, #rtok) })
                } else {
                    self.skip("binop", format!("`*` on {:?}/{:?}", lt, rt));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            other => {
                let _ = self.lower_node(&b.left, Target::Temp);
                let _ = self.lower_node(&b.right, Target::Temp);
                self.skip("binop", format!("unsupported binary op `{}`", tok_str_op(&other)));
                (Ty::Unknown, quote! { WG_SKIP })
            }
        }
    }

    /// Emit an eval-op RHS bound to `target`. Returns (ty, value-token).
    fn emit_op(&mut self, target: Target, ty: Ty, rhs: TokenStream) -> (Ty, TokenStream) {
        let tok = self.bind(target, rhs);
        (ty, tok)
    }

    fn lower_arg(&mut self, arg: Option<&Expr>) -> (Ty, TokenStream) {
        match arg {
            Some(e) => self.lower_node(strip_parens(e), Target::Temp),
            None => {
                self.skip("expr", "missing argument".to_string());
                (Ty::Unknown, quote! { WG_SKIP })
            }
        }
    }

    /// Lower a `sub_component_inputs` RHS into ordered scalar leaf tokens (source order,
    /// which is exactly the shape's scalar order).
    fn flatten_sub(&mut self, expr: &Expr) -> Vec<TokenStream> {
        match strip_parens(expr) {
            Expr::Tuple(ExprTuple { elems, .. }) | Expr::Array(ExprArray { elems, .. }) => {
                let mut leaves = Vec::new();
                for e in elems {
                    leaves.extend(self.flatten_sub(e));
                }
                leaves
            }
            other => {
                let (ty, tok) = self.lower_node(other, Target::Temp);
                self.require_m31(&ty, "sub-input word", other);
                vec![tok]
            }
        }
    }

    fn materialize_u16_const(&mut self, k: u32) -> TokenStream {
        self.referenced_m31.insert(k);
        let c = Ident::new(&format!("m31_{k}"), Span::call_site());
        let t = self.fresh();
        self.out.push(quote! { let #t = eval.u16_from_m31(#c); });
        quote! { #t }
    }

    fn classify_const(&self, name: &str) -> Option<ConstVal> {
        if let Some(cv) = self.consts.get(name) {
            return Some(*cv);
        }
        // Fallback: parse `M31_<k>` / `UInt16_<k>` / `UInt32_<k>` from the name.
        for (prefix, kind) in [
            ("M31_", ConstKind::M31),
            ("UInt16_", ConstKind::U16),
            ("UInt32_", ConstKind::U32),
        ] {
            if let Some(rest) = name.strip_prefix(prefix) {
                if let Ok(v) = rest.parse::<u32>() {
                    return Some(ConstVal { kind, value: v });
                }
            }
        }
        None
    }

    fn peek_const_u16(&self, expr: &Expr) -> Option<u32> {
        self.peek_const_kind(expr, |k| matches!(k, ConstKind::U16))
    }
    fn peek_const_u32(&self, expr: &Expr) -> Option<u32> {
        self.peek_const_kind(expr, |k| matches!(k, ConstKind::U32))
    }
    fn peek_const_kind(&self, expr: &Expr, want: impl Fn(&ConstKind) -> bool) -> Option<u32> {
        match strip_parens(expr) {
            Expr::Path(p) => {
                let name = tok_str(&p.path);
                match self.classify_const(&name) {
                    Some(ConstVal { kind, value }) if want(&kind) => Some(value),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// ======================================================================================
// Block emission (mirrors the retired hand-written add_opcode shape-spec)
// ======================================================================================

fn build_marked_block(
    component: &str,
    fa: &FileAnalysis,
    lw: &Lowerer,
    writer: &ItemFn,
    file: &syn::File,
) -> String {
    let mut seg: Vec<String> = Vec::new();
    seg.push(BEGIN_MARKER.to_string());
    seg.push(header_comment(component, fa, lw));

    // Imports (bare names inside the block, mirroring the shape-spec).
    seg.push(
        "use crate::witness::witness_eval::recording::{RecordingOutput, RecordingWitnessEval};"
            .to_string(),
    );
    seg.push("use crate::witness::witness_eval::simd::SimdWitnessEval;".to_string());
    let slots: Vec<&str> = lw.used_slots.iter().copied().collect();
    seg.push(format!(
        "use crate::witness::witness_eval::{{WitnessEval{}}};",
        slots.iter().map(|s| format!(", {s}")).collect::<String>()
    ));
    seg.push(String::new());
    seg.push(format!("const N_LOOKUP_WORDS: usize = {};", fa.n_lookup_words));
    seg.push(format!("const N_SUB_INPUT_WORDS: usize = {};", fa.n_sub_words));
    seg.push(String::new());

    // 1. The generic per-row body.
    seg.push(format!(
        "/// The per-row `{component}` base-trace body, routed through `WitnessEval`.\n\
         /// Mechanical transcription of `write_trace_simd`'s per-row closure (baseline above)."
    ));
    seg.push(render(&row_body_tokens(component, lw)));
    seg.push(String::new());

    // 2. write_trace_generic_simd — same signature, generic driver, module-private.
    seg.push(format!(
        "/// Generic SIMD driver: same allocation as `write_trace_simd`, but each row runs\n\
         /// `{component}_row_body` on a per-row `SimdWitnessEval`, then reconstructs the concrete\n\
         /// `LookupData` / `SubComponentInputs` from the eval's flat scratch. Module-private (it\n\
         /// returns the module-private `LookupData` / `SubComponentInputs`; wider visibility would\n\
         /// be E0446 and force a change OUTSIDE this block). External callers use the `pub(crate)`\n\
         /// `write_trace_generic` method or the `#[cfg(test)]` `generic_simd_diff` harness."
    ));
    seg.push(render(&generic_simd_tokens(component, lw, writer)));
    seg.push(String::new());

    // 3. impl ClaimGenerator { write_trace_generic } — mirrors write_trace.
    if let Some(method_toks) = write_trace_generic_tokens(file) {
        seg.push("impl ClaimGenerator {".to_string());
        seg.push(
            "/// Generic-path counterpart of [`ClaimGenerator::write_trace`]: identical shape, but\n\
             /// the base trace is produced by `write_trace_generic_simd`."
                .to_string(),
        );
        seg.push(render(&method_toks));
        seg.push("}".to_string());
        seg.push(String::new());
    }

    // 4. record_<component>().
    let record_fn = Ident::new(&format!("record_{component}"), Span::call_site());
    let row_body_fn = Ident::new(&format!("{component}_row_body"), Span::call_site());
    seg.push(format!(
        "/// Record the `{component}` per-row body into witness-JIT bytecode\n\
         /// (statement-independent — recorded once). EXTENDED ops (if any) surface in\n\
         /// `RecordingOutput::poison_ops` — the honest ISA-V2 census, not a failure."
    ));
    seg.push(render(&quote! {
        #[allow(dead_code)]
        pub(crate) fn #record_fn() -> RecordingOutput {
            let mut eval = RecordingWitnessEval::new(#component);
            #row_body_fn(&mut eval);
            eval.finish()
        }
    }));
    seg.push(String::new());

    // 5. Test-only surface: flats + GenericSimdDiff + generic_simd_diff.
    seg.push(
        "// ---- Test-only surface for the byte-equality gate ---------------------------------"
            .to_string(),
    );
    seg.push(String::new());
    seg.push(render(&lookup_flat_tokens(lw)));
    seg.push(String::new());
    seg.push(render(&sub_flat_tokens(lw)));
    seg.push(String::new());
    seg.push("/// Byte-comparison bundle (only public types cross the module boundary).".to_string());
    seg.push(render(&generic_simd_diff_struct_tokens()));
    seg.push(String::new());
    seg.push(
        "/// Run BOTH SIMD writers on the same (pure-read) states and return public compare data."
            .to_string(),
    );
    seg.push(render(&generic_simd_diff_fn_tokens(writer)));
    seg.push(END_MARKER.to_string());

    seg.join("\n")
}

/// Deterministic generated header: provenance + the derived flat layouts.
fn header_comment(component: &str, fa: &FileAnalysis, lw: &Lowerer) -> String {
    let mut lines = Vec::new();
    lines.push("//".to_string());
    lines.push(format!(
        "// GENERATED by tools/witness_genericize for `{component}` — mechanical rewrite of"
    ));
    lines.push(
        "// `write_trace_simd`'s per-row closure into a generic body over `WitnessEval`. Do not"
            .to_string(),
    );
    lines.push(
        "// edit by hand: re-run the tool after upstream regeneration (this block is stripped and"
            .to_string(),
    );
    lines.push(
        "// re-emitted idempotently). The original `write_trace_simd` above is the untouched"
            .to_string(),
    );
    lines.push("// byte-equality baseline (see `witness_eval::differential_test`).".to_string());
    lines.push("//".to_string());
    lines.push("// Flat layouts (derived, DECLARATION order):".to_string());
    lines.push("//   LOOKUP words:".to_string());
    for f in &lw.lookup_fields {
        if f.width == 1 {
            lines.push(format!("//     {} {}", f.name, f.base));
        } else {
            lines.push(format!(
                "//     {}[{}] {}..{}",
                f.name,
                f.width,
                f.base,
                f.base + f.width - 1
            ));
        }
    }
    lines.push(format!("//     ({} words)", fa.n_lookup_words));
    lines.push("//   SUB-INPUT words:".to_string());
    for s in &lw.sub_slots {
        let count = s.shape.scalar_count();
        if count == 1 {
            lines.push(format!("//     {}[{}] {}", s.field, s.index, s.base));
        } else {
            lines.push(format!(
                "//     {}[{}] {}..{}",
                s.field,
                s.index,
                s.base,
                s.base + count - 1
            ));
        }
    }
    lines.push(format!("//     ({} words)", fa.n_sub_words));
    lines.join("\n")
}

fn row_body_tokens(component: &str, lw: &Lowerer) -> TokenStream {
    let row_body_fn = Ident::new(&format!("{component}_row_body"), Span::call_site());
    let mut const_lets: Vec<TokenStream> = Vec::new();
    for v in &lw.referenced_m31 {
        let id = Ident::new(&format!("m31_{v}"), Span::call_site());
        let vl = u32_lit(*v);
        const_lets.push(quote! { let #id = eval.m31_const(#vl); });
    }
    let body_stmts = &lw.out;
    quote! {
        #[allow(clippy::identity_op)]
        #[allow(clippy::erasing_op)]
        #[allow(unused_variables)]
        #[allow(dead_code)]
        fn #row_body_fn<E: WitnessEval>(eval: &mut E) {
            #(#const_lets)*
            #(#body_stmts)*
        }
    }
}

fn generic_simd_tokens(component: &str, lw: &Lowerer, writer: &ItemFn) -> TokenStream {
    let row_body_fn = Ident::new(&format!("{component}_row_body"), Span::call_site());

    // Transcribe signature inputs + output verbatim from write_trace_simd.
    let inputs = &writer.sig.inputs;
    let output = &writer.sig.output;

    // Preamble: keep the leading locals that are NOT broadcast constants.
    let mut preamble: Vec<TokenStream> = Vec::new();
    for st in &writer.block.stmts {
        match st {
            Stmt::Local(local) => {
                if local_const(local).is_some() {
                    continue; // materialized inside the row body via m31_const
                }
                preamble.push(quote! { #st });
            }
            _ => break, // stop at the rayon expression
        }
    }

    let addr = lw
        .addr_state
        .clone()
        .unwrap_or_else(|| "memory_address_to_id_state".to_string());
    let big = lw
        .big_state
        .clone()
        .unwrap_or_else(|| "memory_id_to_big_state".to_string());
    let addr_id = Ident::new(&addr, Span::call_site());
    let big_id = Ident::new(&big, Span::call_site());
    let input_id = Ident::new(&lw.input_name, Span::call_site());
    let row_id = Ident::new(&lw.row_name, Span::call_site());
    let lookup_id = Ident::new(&lw.lookup_name, Span::call_site());
    let sub_id = Ident::new(&lw.sub_name, Span::call_site());

    let reconstruct_lookup = reconstruct_lookup(lw, &lookup_id);
    let reconstruct_sub = reconstruct_sub(lw, &sub_id);

    quote! {
        #[allow(clippy::type_complexity)]
        #[allow(unused_variables)]
        #[allow(dead_code)]
        fn write_trace_generic_simd(#inputs) #output {
            #(#preamble)*

            (
                trace.par_iter_mut(),
                #lookup_id.par_iter_mut(),
                #sub_id.par_iter_mut(),
                inputs.into_par_iter(),
            )
                .into_par_iter()
                .enumerate()
                .for_each(|(row_index, (#row_id, #lookup_id, #sub_id, #input_id))| {
                    let mut eval = SimdWitnessEval::new(
                        #row_id,
                        #addr_id,
                        #big_id,
                        #input_id,
                        row_index,
                        &enabler_col,
                        N_LOOKUP_WORDS,
                        N_SUB_INPUT_WORDS,
                    );
                    #row_body_fn(&mut eval);

                    let lw = eval.lookup_scratch();
                    #(#reconstruct_lookup)*

                    let sw = eval.sub_scratch();
                    #(#reconstruct_sub)*
                });

            (trace, #lookup_id, #sub_id)
        }
    }
}

fn reconstruct_lookup(lw: &Lowerer, lookup_id: &Ident) -> Vec<TokenStream> {
    let mut out = Vec::new();
    for lf in &lw.lookup_fields {
        let field = Ident::new(&lf.name, Span::call_site());
        if lf.scalar {
            let b = usize_lit(lf.base);
            out.push(quote! { *#lookup_id.#field = lw[#b]; });
        } else {
            let idxs: Vec<TokenStream> = (0..lf.width)
                .map(|j| {
                    let b = usize_lit(lf.base + j);
                    quote! { lw[#b] }
                })
                .collect();
            out.push(quote! { *#lookup_id.#field = [ #(#idxs),* ]; });
        }
    }
    out
}

fn reconstruct_sub(lw: &Lowerer, sub_id: &Ident) -> Vec<TokenStream> {
    let mut out = Vec::new();
    for sa in &lw.sub_slots {
        let field = Ident::new(&sa.field, Span::call_site());
        let k = usize_lit(sa.index);
        let mut idx = sa.base;
        let value = rebuild_shape(&sa.shape, &mut idx);
        out.push(quote! { *#sub_id.#field[#k] = #value; });
    }
    out
}

fn rebuild_shape(shape: &Shape, idx: &mut usize) -> TokenStream {
    match shape {
        Shape::Scalar => {
            let i = usize_lit(*idx);
            *idx += 1;
            quote! { sw[#i] }
        }
        Shape::Tuple(v) => {
            let parts: Vec<TokenStream> = v.iter().map(|s| rebuild_shape(s, idx)).collect();
            quote! { ( #(#parts),* ) }
        }
        Shape::Array(v) => {
            let parts: Vec<TokenStream> = v.iter().map(|s| rebuild_shape(s, idx)).collect();
            quote! { [ #(#parts),* ] }
        }
    }
}

/// Clone the component's `write_trace` method as `pub(crate) fn write_trace_generic`
/// (same receiver + params + body), retargeting the `write_trace_simd` call.
fn write_trace_generic_tokens(file: &syn::File) -> Option<TokenStream> {
    let method = file.items.iter().find_map(|it| match it {
        Item::Impl(im) => im.items.iter().find_map(|ii| match ii {
            syn::ImplItem::Fn(f) if f.sig.ident == "write_trace" => Some(f.clone()),
            _ => None,
        }),
        _ => None,
    })?;

    let mut method = method;
    method.sig.ident = Ident::new("write_trace_generic", Span::call_site());
    method.vis = syn::parse_quote!(pub(crate));
    method.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    struct CallRewriter;
    impl syn::visit_mut::VisitMut for CallRewriter {
        fn visit_ident_mut(&mut self, id: &mut Ident) {
            if *id == "write_trace_simd" {
                *id = Ident::new("write_trace_generic_simd", id.span());
            }
        }
    }
    syn::visit_mut::visit_impl_item_fn_mut(&mut CallRewriter, &mut method);
    Some(quote! { #method })
}

fn lookup_flat_tokens(lw: &Lowerer) -> TokenStream {
    let mut parts: Vec<TokenStream> = Vec::new();
    for lf in &lw.lookup_fields {
        let field = Ident::new(&lf.name, Span::call_site());
        if lf.scalar {
            parts.push(quote! { ld.#field.clone() });
        } else {
            parts.push(quote! { ld.#field.iter().flatten().copied().collect() });
        }
    }
    quote! {
        #[cfg(test)]
        fn lookup_data_flat(ld: &LookupData) -> Vec<Vec<PackedM31>> {
            vec![ #(#parts),* ]
        }
    }
}

fn sub_flat_tokens(lw: &Lowerer) -> TokenStream {
    let mut parts: Vec<TokenStream> = Vec::new();
    for sa in &lw.sub_slots {
        let field = Ident::new(&sa.field, Span::call_site());
        let k = usize_lit(sa.index);
        match &sa.shape {
            Shape::Scalar => parts.push(quote! { sci.#field[#k].clone() }),
            _ => {
                let t: Ident = Ident::new("t", Span::call_site());
                let scalars = shape_projection(&sa.shape, quote! { #t });
                parts.push(quote! {
                    sci.#field[#k]
                        .iter()
                        .flat_map(|#t| vec![ #(#scalars),* ])
                        .collect::<Vec<_>>()
                });
            }
        }
    }
    quote! {
        #[cfg(test)]
        fn sub_inputs_flat(sci: &SubComponentInputs) -> Vec<Vec<PackedM31>> {
            vec![ #(#parts),* ]
        }
    }
}

/// Scalar projections of a shaped `PackedInputType` value. `base` navigates from an
/// `&PackedInputType` via `.N` / `[j]`; each leaf is a Copy `PackedM31` via auto-deref.
fn shape_projection(shape: &Shape, base: TokenStream) -> Vec<TokenStream> {
    match shape {
        Shape::Scalar => vec![quote! { #base }],
        Shape::Tuple(v) => {
            let mut out = Vec::new();
            for (i, s) in v.iter().enumerate() {
                let lit = Literal::usize_unsuffixed(i);
                out.extend(shape_projection(s, quote! { #base.#lit }));
            }
            out
        }
        Shape::Array(v) => {
            let mut out = Vec::new();
            for (j, s) in v.iter().enumerate() {
                let lit = usize_lit(j);
                out.extend(shape_projection(s, quote! { #base[#lit] }));
            }
            out
        }
    }
}

/// The component-independent compare bundle (verbatim from the shape-spec).
fn generic_simd_diff_struct_tokens() -> TokenStream {
    quote! {
        #[cfg(test)]
        pub(crate) struct GenericSimdDiff {
            pub log_size: u32,
            pub orig_rows: Vec<[M31; N_TRACE_COLUMNS]>,
            pub gen_rows: Vec<[M31; N_TRACE_COLUMNS]>,
            pub orig_lookup: Vec<Vec<PackedM31>>,
            pub gen_lookup: Vec<Vec<PackedM31>>,
            pub orig_sub: Vec<Vec<PackedM31>>,
            pub gen_sub: Vec<Vec<PackedM31>>,
            pub orig_interaction_cols: Vec<Vec<M31>>,
            pub gen_interaction_cols: Vec<Vec<M31>>,
            pub orig_claimed_sum: SecureField,
            pub gen_claimed_sum: SecureField,
        }
    }
}

/// `generic_simd_diff(...)`: same params as `write_trace_simd`; runs both writers and
/// packages the compare bundle (verbatim body from the shape-spec).
fn generic_simd_diff_fn_tokens(writer: &ItemFn) -> TokenStream {
    let inputs = &writer.sig.inputs;
    // Argument names in order; the first must be `inputs` (cloned into the first call).
    let mut names: Vec<Ident> = Vec::new();
    for arg in inputs {
        if let FnArg::Typed(pt) = arg {
            if let Pat::Ident(pi) = &*pt.pat {
                names.push(pi.ident.clone());
            }
        }
    }
    let rest = &names[1..];
    quote! {
        #[cfg(test)]
        pub(crate) fn generic_simd_diff(#inputs) -> GenericSimdDiff {
            let (trace_o, ld_o, sci_o) = write_trace_simd(inputs.clone(), #(#rest),*);
            let (trace_g, ld_g, sci_g) = write_trace_generic_simd(inputs, #(#rest),*);

            let log_size = trace_o.log_size();
            let orig_rows = (0..(1usize << log_size))
                .map(|r| trace_o.row_at(r))
                .collect();
            let gen_rows = (0..(1usize << log_size))
                .map(|r| trace_g.row_at(r))
                .collect();

            let orig_lookup = lookup_data_flat(&ld_o);
            let gen_lookup = lookup_data_flat(&ld_g);
            let orig_sub = sub_inputs_flat(&sci_o);
            let gen_sub = sub_inputs_flat(&sci_g);

            let common = relations::CommonLookupElements::dummy();
            let (raw_o, _) = InteractionClaimGenerator {
                log_size,
                lookup_data: ld_o,
            }
            .write_interaction_trace(&common);
            let (raw_g, _) = InteractionClaimGenerator {
                log_size,
                lookup_data: ld_g,
            }
            .write_interaction_trace(&common);
            let (cols_o, orig_claimed_sum) = raw_o.finalize_on_simd();
            let (cols_g, gen_claimed_sum) = raw_g.finalize_on_simd();
            let orig_interaction_cols = cols_o.iter().map(|c| c.values.to_cpu()).collect();
            let gen_interaction_cols = cols_g.iter().map(|c| c.values.to_cpu()).collect();

            GenericSimdDiff {
                log_size,
                orig_rows,
                gen_rows,
                orig_lookup,
                gen_lookup,
                orig_sub,
                gen_sub,
                orig_interaction_cols,
                gen_interaction_cols,
                orig_claimed_sum,
                gen_claimed_sum,
            }
        }
    }
}

fn render(t: &TokenStream) -> String {
    t.to_string()
}

// ======================================================================================
// Small syn helpers
// ======================================================================================

fn strip_parens(mut e: &Expr) -> &Expr {
    while let Expr::Paren(ExprParen { expr, .. }) = e {
        e = expr;
    }
    e
}

fn local_ident(local: &Local) -> Option<String> {
    match &local.pat {
        Pat::Ident(pi) => Some(pi.ident.to_string()),
        Pat::Type(pt) => match &*pt.pat {
            Pat::Ident(pi) => Some(pi.ident.to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// If `local` is `let IDENT = PackedM31::broadcast(M31::from(K));` (or UInt16/UInt32),
/// return its ConstVal.
fn local_const(local: &Local) -> Option<ConstVal> {
    let init = local.init.as_ref()?;
    let call = match strip_parens(&init.expr) {
        Expr::Call(c) => c,
        _ => return None,
    };
    let path = match &*call.func {
        Expr::Path(p) => tok_str(&p.path),
        _ => return None,
    };
    let kind = match path.as_str() {
        "PackedM31 :: broadcast" => ConstKind::M31,
        "PackedUInt16 :: broadcast" => ConstKind::U16,
        "PackedUInt32 :: broadcast" => ConstKind::U32,
        _ => return None,
    };
    // arg: M31::from(K) / UInt16::from(K) / UInt32::from(K)
    let inner = match call.args.first().map(strip_parens) {
        Some(Expr::Call(c)) => c,
        _ => return None,
    };
    let v = match inner.args.first().map(strip_parens) {
        Some(Expr::Lit(el)) => match &el.lit {
            Lit::Int(i) => i.base10_parse::<u32>().ok()?,
            _ => return None,
        },
        _ => return None,
    };
    Some(ConstVal { kind, value: v })
}

fn is_path_named(e: &Expr, name: &str) -> bool {
    matches!(strip_parens(e), Expr::Path(p) if p.path.is_ident(name))
}

fn expr_usize(e: &Expr) -> Option<usize> {
    match strip_parens(e) {
        Expr::Lit(el) => match &el.lit {
            Lit::Int(i) => i.base10_parse::<usize>().ok(),
            _ => None,
        },
        _ => None,
    }
}

fn usize_lit(v: usize) -> Literal {
    Literal::usize_unsuffixed(v)
}
fn u32_lit(v: u32) -> Literal {
    Literal::u32_unsuffixed(v)
}

fn tok_str<T: quote::ToTokens>(t: &T) -> String {
    quote! { #t }.to_string()
}

fn tok_str_op(op: &BinOp) -> String {
    quote! { #op }.to_string()
}

// ======================================================================================
// rustfmt
// ======================================================================================

fn rustfmt_block(block: &str) -> String {
    // The block is a set of top-level items; rustfmt formats it as a standalone file.
    let dir = std::env::temp_dir();
    let tmp = dir.join(format!("wg_block_{}.rs", std::process::id()));
    if std::fs::write(&tmp, block).is_err() {
        return block.to_string();
    }
    let rustfmt = rustfmt_bin();
    let ok = std::process::Command::new(&rustfmt)
        .arg("--edition")
        .arg("2021")
        .arg(&tmp)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let out = if ok {
        std::fs::read_to_string(&tmp).unwrap_or_else(|_| block.to_string())
    } else {
        block.to_string()
    };
    let _ = std::fs::remove_file(&tmp);
    out
}

fn rustfmt_bin() -> String {
    if let Ok(out) = std::process::Command::new("rustup")
        .args(["which", "rustfmt"])
        .output()
    {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                return p;
            }
        }
    }
    "rustfmt".to_string()
}

// ======================================================================================
// Block insert / strip (idempotent)
// ======================================================================================

/// Remove any existing marked block (BEGIN..END inclusive) from `src`.
fn strip_existing_block(src: &str) -> String {
    let Some(bstart) = src.find(BEGIN_MARKER) else {
        return src.to_string();
    };
    let Some(erel) = src[bstart..].find(END_MARKER) else {
        return src.to_string();
    };
    let eend = bstart + erel + END_MARKER.len();
    // Trim the line containing BEGIN back to the start of its line, and consume one
    // trailing newline after END.
    let line_start = src[..bstart].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let mut after = eend;
    if src[after..].starts_with('\n') {
        after += 1;
    }
    let mut s = String::new();
    s.push_str(&src[..line_start]);
    s.push_str(&src[after..]);
    s
}

/// Insert `block` immediately before the `LookupData` struct (and its attributes),
/// separated by exactly one blank line on each side. Trimming surrounding newlines makes
/// strip+reinsert round-trip to identical bytes regardless of blank-line drift.
fn insert_block(src: &str, block: &str) -> Option<String> {
    let anchor = src.find("struct LookupData")?;
    // Walk back over the attribute/comment lines directly preceding the struct.
    let mut line_start = src[..anchor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    loop {
        if line_start == 0 {
            break;
        }
        let prev_line_start = src[..line_start - 1].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let prev = src[prev_line_start..line_start - 1].trim_start();
        if prev.starts_with("#[") || prev.starts_with("///") || prev.starts_with("//!") {
            line_start = prev_line_start;
        } else {
            break;
        }
    }
    let before = src[..line_start].trim_end_matches('\n');
    let after = &src[line_start..];
    let block = block.trim_matches('\n');
    Some(format!("{before}\n\n{block}\n\n{after}"))
}

/// Extract the current on-disk marked block (for --check), if present.
fn extract_block(src: &str) -> Option<String> {
    let bstart = src.find(BEGIN_MARKER)?;
    let erel = src[bstart..].find(END_MARKER)?;
    let eend = bstart + erel + END_MARKER.len();
    Some(src[bstart..eend].to_string())
}

// ======================================================================================
// Modes
// ======================================================================================

fn run_census(files: &[PathBuf]) -> ExitCode {
    let mut analyses: Vec<(PathBuf, FileAnalysis)> = Vec::new();
    for f in files {
        analyses.push((f.clone(), analyze_file(f, false)));
    }

    let n = analyses.len();
    let writers = analyses.iter().filter(|(_, a)| a.has_writer).count();
    let matched: Vec<&(PathBuf, FileAnalysis)> = analyses.iter().filter(|(_, a)| a.matched).collect();
    let matched_u32: Vec<&(PathBuf, FileAnalysis)> =
        analyses.iter().filter(|(_, a)| a.matched_u32).collect();

    println!("======================================================================");
    println!("witness_genericize CENSUS");
    println!("======================================================================");
    println!("Files scanned:                          {n}");
    println!("  with write_trace_simd:                {writers}");
    println!("  MATCHED (rewritable):                 {}", matched.len());
    println!("  MATCHED (needs u32 trait extension):  {}", matched_u32.len());
    println!(
        "  skipped:                              {}",
        n - matched.len() - matched_u32.len()
    );
    println!();

    println!("--- MATCHED files (rewritable now) ---");
    for (_p, a) in &matched {
        println!(
            "  {:<34} cols={:<4} lookup_words={:<4} sub_words={}",
            a.component, a.n_cols, a.n_lookup_words, a.n_sub_words
        );
    }
    println!();

    println!("--- MATCHED files (needs u32 trait extension; census-only, NOT emitted) ---");
    for (_p, a) in &matched_u32 {
        println!(
            "  {:<34} cols={:<4} lookup_words={:<4} sub_words={:<4} u32_sites={}",
            a.component, a.n_cols, a.n_lookup_words, a.n_sub_words, a.u32_sites
        );
    }
    println!();

    println!("--- SKIPPED files (loud reasons) ---");
    for (_p, a) in analyses.iter().filter(|(_, a)| !a.matched && !a.matched_u32) {
        if let Some(fs) = &a.file_skip {
            println!("  {:<34} [{}] {}", a.component, fs.category, fs.detail);
        } else {
            let mut by_cat: BTreeMap<&'static str, usize> = BTreeMap::new();
            for s in &a.skips {
                *by_cat.entry(s.category).or_insert(0) += 1;
            }
            let summ: Vec<String> = by_cat.iter().map(|(c, n)| format!("{c}×{n}")).collect();
            println!(
                "  {:<34} skeleton OK; {} unmatched constructs ({}){}",
                a.component,
                a.skips.len(),
                summ.join(", "),
                if a.u32_sites > 0 {
                    format!(" + {} u32 sites", a.u32_sites)
                } else {
                    String::new()
                }
            );
            let mut seen: BTreeSet<String> = BTreeSet::new();
            for s in &a.skips {
                let key = format!("[{}] {}", s.category, s.detail);
                if seen.insert(key.clone()) {
                    println!("        {key}");
                }
                if seen.len() >= 4 {
                    println!(
                        "        ... ({} more distinct)",
                        distinct_skips(&a.skips) - seen.len()
                    );
                    break;
                }
            }
        }
    }
    println!();

    // deduce_output backlog table (the device-kernel / ISA-V2 backlog).
    println!("--- deduce_output census (device-kernel backlog) ---");
    let mut recv_count: BTreeMap<String, usize> = BTreeMap::new();
    let mut recv_files: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (_p, a) in &analyses {
        for (recv, c) in &a.deduce_hits {
            *recv_count.entry(recv.clone()).or_insert(0) += c;
            recv_files
                .entry(recv.clone())
                .or_default()
                .insert(a.component.clone());
        }
    }
    let handled: BTreeSet<&str> = [
        "memory_address_to_id_state.deduce_output",
        "memory_id_to_big_state.deduce_output",
    ]
    .into_iter()
    .collect();
    let mut rows: Vec<(String, usize, usize)> = recv_count
        .iter()
        .map(|(r, c)| (r.clone(), *c, recv_files[r].len()))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (recv, count, nfiles) in &rows {
        let tag = if handled.contains(recv.as_str()) {
            "HANDLED "
        } else {
            "BACKLOG "
        };
        println!("  {tag}{recv:<48} {count:>5} sites  in {nfiles} files");
    }
    println!();

    // Construct-skip census grouped across files (non-deduce), normalized so specific
    // identifiers/values collapse into one backlog row per construct KIND.
    println!("--- unmatched-construct census (grouped by kind across files, excl. deduce_output) ---");
    let mut group: BTreeMap<(&'static str, String), (usize, BTreeSet<String>)> = BTreeMap::new();
    for (_p, a) in &analyses {
        if let Some(fs) = &a.file_skip {
            let key = normalize_detail(fs.category, &fs.detail);
            let ent = group.entry((fs.category, key)).or_default();
            ent.0 += 1;
            ent.1.insert(a.component.clone());
        }
        for s in &a.skips {
            if s.category == "deduce_output" {
                continue;
            }
            let key = normalize_detail(s.category, &s.detail);
            let ent = group.entry((s.category, key)).or_default();
            ent.0 += 1;
            ent.1.insert(a.component.clone());
        }
    }
    let mut gvec: Vec<((&'static str, String), (usize, BTreeSet<String>))> =
        group.into_iter().collect();
    gvec.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(&b.0)));
    let shown = gvec.len().min(70);
    for ((cat, detail), (count, fileset)) in gvec.iter().take(shown) {
        println!("  [{cat}] {detail}  — {count} sites in {} files", fileset.len());
    }
    if gvec.len() > shown {
        println!("  ... ({} more construct kinds)", gvec.len() - shown);
    }

    ExitCode::SUCCESS
}

/// Collapse a per-site skip detail into a construct-KIND key for the grouped census:
/// specific identifiers (after the first backtick) are dropped; numeric const payloads
/// are stripped, so e.g. `ConstU16(7)` and `ConstU16(1)` group together.
fn normalize_detail(category: &str, detail: &str) -> String {
    match category {
        // The backtick payload IS the key (callee path / macro path).
        "call" | "macro" => detail.to_string(),
        "binop" => strip_paren_nums(detail),
        // Everything else: keep the reason head, drop the quoted specific identifier.
        _ => detail
            .split('`')
            .next()
            .unwrap_or(detail)
            .trim()
            .to_string(),
    }
}

/// Remove `(<digits>)` groups (e.g. `ConstU16(7)` → `ConstU16`).
fn strip_paren_nums(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(open) = rest.find('(') {
        if let Some(close_rel) = rest[open + 1..].find(')') {
            let inner = &rest[open + 1..open + 1 + close_rel];
            if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit()) {
                out.push_str(&rest[..open]);
                rest = &rest[open + 1 + close_rel + 1..];
                continue;
            }
        }
        out.push_str(&rest[..=open]);
        rest = &rest[open + 1..];
    }
    out.push_str(rest);
    out
}

fn distinct_skips(skips: &[Skip]) -> usize {
    skips
        .iter()
        .map(|s| format!("[{}] {}", s.category, s.detail))
        .collect::<BTreeSet<_>>()
        .len()
}

fn not_emittable_reason(a: &FileAnalysis) -> String {
    if a.matched_u32 {
        return format!(
            "matched via u32 census rules only ({} u32 sites) — needs u32 trait \
             extension; not emitted",
            a.u32_sites
        );
    }
    a.file_skip
        .as_ref()
        .map(|s| format!("[{}] {}", s.category, s.detail))
        .unwrap_or_else(|| {
            format!(
                "{} unmatched constructs (first: {})",
                a.skips.len(),
                a.skips
                    .first()
                    .map(|s| format!("[{}] {}", s.category, s.detail))
                    .unwrap_or_default()
            )
        })
}

fn run_emit_dir(files: &[PathBuf], dir: &Path) -> ExitCode {
    if std::fs::create_dir_all(dir).is_err() {
        eprintln!("cannot create emit dir {}", dir.display());
        return ExitCode::from(2);
    }
    let mut emitted = 0;
    let mut skipped = 0;
    for f in files {
        let a = analyze_file(f, true);
        if !a.matched {
            skipped += 1;
            eprintln!("SKIP {}: {}", a.component, not_emittable_reason(&a));
            continue;
        }
        let Some(block) = &a.block else {
            eprintln!("SKIP {}: matched but no block built", a.component);
            skipped += 1;
            continue;
        };
        let src = std::fs::read_to_string(f).unwrap();
        let cleaned = strip_existing_block(&src);
        let Some(full) = insert_block(&cleaned, block) else {
            eprintln!("SKIP {}: no `struct LookupData` anchor for insert", a.component);
            skipped += 1;
            continue;
        };
        let out_path = dir.join(f.file_name().unwrap());
        if std::fs::write(&out_path, &full).is_ok() {
            emitted += 1;
            println!("EMIT {} -> {}", a.component, out_path.display());
        } else {
            eprintln!("SKIP {}: write failed", a.component);
            skipped += 1;
        }
    }
    println!("witness_genericize --emit-dir: {emitted} emitted, {skipped} skipped");
    ExitCode::SUCCESS
}

fn run_in_place(files: &[PathBuf]) -> ExitCode {
    let mut changed = 0;
    let mut skipped = 0;
    for f in files {
        let a = analyze_file(f, true);
        if !a.matched {
            skipped += 1;
            eprintln!("SKIP {}: {}", a.component, not_emittable_reason(&a));
            continue;
        }
        let Some(block) = &a.block else {
            skipped += 1;
            continue;
        };
        let src = std::fs::read_to_string(f).unwrap();
        let cleaned = strip_existing_block(&src);
        let Some(full) = insert_block(&cleaned, block) else {
            skipped += 1;
            continue;
        };
        if full != src {
            if std::fs::write(f, &full).is_ok() {
                changed += 1;
                println!("REWROTE {}", a.component);
            }
        } else {
            println!("UNCHANGED {} (already up to date)", a.component);
        }
    }
    println!("witness_genericize --in-place: {changed} rewritten, {skipped} skipped");
    ExitCode::SUCCESS
}

fn run_check(files: &[PathBuf]) -> ExitCode {
    let mut drift = 0;
    for f in files {
        let a = analyze_file(f, true);
        if !a.matched {
            continue;
        }
        let Some(block) = &a.block else { continue };
        let src = std::fs::read_to_string(f).unwrap();
        match extract_block(&src) {
            Some(on_disk) => {
                let want = block.trim_end();
                if on_disk.trim_end() != want {
                    drift += 1;
                    eprintln!("DRIFT {}: on-disk block differs from generated", a.component);
                }
            }
            None => {
                drift += 1;
                eprintln!("MISSING {}: matched file has no on-disk block", a.component);
            }
        }
    }
    if drift == 0 {
        println!("witness_genericize --check: OK (no drift)");
        ExitCode::SUCCESS
    } else {
        eprintln!("witness_genericize --check: {drift} files drifted");
        ExitCode::from(1)
    }
}

// ======================================================================================
// Tests
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_snippet_with_slots(
        consts: &[(&str, ConstKind, u32)],
        body: &str,
        sub_slots: Vec<SubSlot>,
    ) -> Lowerer {
        let mut cmap = BTreeMap::new();
        for (n, k, v) in consts {
            cmap.insert(n.to_string(), ConstVal { kind: *k, value: *v });
        }
        let mut lw = Lowerer::new(
            cmap,
            Some("memory_address_to_id_state".to_string()),
            Some("memory_id_to_big_state".to_string()),
            "add_opcode_input".to_string(),
            "row".to_string(),
            "lookup_data".to_string(),
            "sub_component_inputs".to_string(),
            vec![],
            sub_slots,
        );
        let block: syn::Block = syn::parse_str(&format!("{{ {body} }}")).unwrap();
        lw.lower_body(&block.stmts);
        lw
    }

    fn lower_snippet(consts: &[(&str, ConstKind, u32)], body: &str) -> Lowerer {
        lower_snippet_with_slots(consts, body, vec![])
    }

    #[test]
    fn infer_input_fields() {
        let lw = lower_snippet(&[], "let a = add_opcode_input.pc; let b = add_opcode_input.fp;");
        assert_eq!(lw.env["a"], Ty::M31);
        assert_eq!(lw.env["b"], Ty::M31);
        assert!(lw.skips.is_empty());
        assert!(lw.used_slots.contains("SLOT_PC"));
        assert!(lw.used_slots.contains("SLOT_FP"));
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("eval . input (SLOT_PC)"));
        assert!(s.contains("eval . input (SLOT_FP)"));
    }

    #[test]
    fn infer_u16_bit_ops() {
        let lw = lower_snippet(
            &[("UInt16_127", ConstKind::U16, 127), ("UInt16_9", ConstKind::U16, 9)],
            "let f = memory_id_to_big_state.deduce_output(memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let x = ((PackedUInt16::from_m31(f.get_m31(1))) & (UInt16_127)) << (UInt16_9); \
             let y = x.as_m31();",
        );
        assert_eq!(lw.env["f"], Ty::Felt);
        assert_eq!(lw.env["x"], Ty::U16);
        assert_eq!(lw.env["y"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        assert_eq!(lw.u32_sites, 0);
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("u16_and"));
        assert!(s.contains("u16_shl"));
        assert!(s.contains("u16_as_m31"));
        assert!(s.contains("mem_addr_to_id"));
        assert!(s.contains("mem_id_to_value"));
        assert!(s.contains("felt_get_m31"));
    }

    #[test]
    fn u16_const_mask_on_left_of_and() {
        // add_opcode's sub_p_bit idiom: (UInt16_1) & (xor_chain)
        let lw = lower_snippet(
            &[("UInt16_1", ConstKind::U16, 1)],
            "let f = memory_id_to_big_state.deduce_output(memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let x = ((UInt16_1) & ((PackedUInt16::from_m31(f.get_m31(0))) ^ (PackedUInt16::from_m31(f.get_m31(1))))); \
             let y = x.as_m31();",
        );
        assert_eq!(lw.env["x"], Ty::U16);
        assert_eq!(lw.env["y"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("u16_xor"));
        assert!(s.contains("u16_and"));
    }

    #[test]
    fn infer_m31_arith_and_consts() {
        let lw = lower_snippet(
            &[("M31_1", ConstKind::M31, 1), ("M31_8", ConstKind::M31, 8)],
            "let a = add_opcode_input.ap; let b = ((a) * (M31_8)) + ((M31_1) - (a));",
        );
        assert_eq!(lw.env["b"], Ty::M31);
        assert!(lw.skips.is_empty());
        assert!(lw.referenced_m31.contains(&1));
        assert!(lw.referenced_m31.contains(&8));
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("m31_mul"));
        assert!(s.contains("m31_add"));
        assert!(s.contains("m31_sub"));
        assert!(s.contains("m31_8"), "const binding should be named m31_8");
    }

    #[test]
    fn infer_mask_ops() {
        let lw = lower_snippet(
            &[("M31_256", ConstKind::M31, 256), ("M31_511", ConstKind::M31, 511)],
            "let f = memory_id_to_big_state.deduce_output(memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let m = f.get_m31(27).eq(M31_256); \
             let n = (f.get_m31(20).eq(M31_511)) & (m); \
             let mc = m.as_m31();",
        );
        assert_eq!(lw.env["m"], Ty::Mask);
        assert_eq!(lw.env["n"], Ty::Mask);
        assert_eq!(lw.env["mc"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("m31_eq"));
        assert!(s.contains("mask_and"));
        assert!(s.contains("mask_as_m31"));
    }

    #[test]
    fn tuple_projection_types() {
        let lw = lower_snippet(
            &[("M31_1", ConstKind::M31, 1), ("M31_0", ConstKind::M31, 0)],
            "let a = add_opcode_input.ap; \
             let t = ([a, a, a], [a, M31_1, M31_0], M31_0); \
             let u = (t.0[2]) + (t.1[1]);",
        );
        assert!(matches!(lw.env["t"], Ty::Tuple(_)));
        assert_eq!(lw.env["u"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
    }

    #[test]
    fn unknown_deduce_is_skip() {
        let lw = lower_snippet(
            &[],
            "let x = PackedPartialEcMulGeneric::deduce_output(add_opcode_input.pc);",
        );
        assert!(!lw.skips.is_empty());
        assert!(lw.skips.iter().any(|s| s.category == "deduce_output"));
    }

    #[test]
    fn u32_family_is_census_only_match() {
        let lw = lower_snippet(
            &[("UInt32_511", ConstKind::U32, 511), ("UInt32_9", ConstKind::U32, 9)],
            "let a = add_opcode_input.ap; \
             let x = PackedUInt32::from_m31(a); \
             let y = ((x) & (UInt32_511)) + ((x) << (UInt32_9)); \
             let z = y.low().as_m31(); \
             let w = y.high().as_m31();",
        );
        assert!(lw.skips.is_empty(), "u32 family must not skip: {:?}", lw.skips);
        assert!(lw.u32_sites >= 5, "expected u32 sites, got {}", lw.u32_sites);
        assert_eq!(lw.env["z"], Ty::M31);
        assert_eq!(lw.env["w"], Ty::M31);
    }

    #[test]
    fn shape_scalar_count_is_correct() {
        let s = Shape::Tuple(vec![
            Shape::Scalar,
            Shape::Array(vec![Shape::Scalar, Shape::Scalar, Shape::Scalar]),
            Shape::Array(vec![Shape::Scalar, Shape::Scalar]),
            Shape::Scalar,
        ]);
        assert_eq!(s.scalar_count(), 7);
    }

    #[test]
    fn lookup_field_width_parse() {
        let ty: Type = syn::parse_str("Vec<[PackedM31; 30]>").unwrap();
        assert_eq!(lookup_field_width(&ty), Some((30, false)));
        let ty2: Type = syn::parse_str("Vec<PackedM31>").unwrap();
        assert_eq!(lookup_field_width(&ty2), Some((1, true)));
    }

    /// Declaration-order layout: assignments interleaved in FILE order must still get
    /// bases from the struct declaration order (the shape-spec's documented layout).
    #[test]
    fn sub_layout_is_declaration_order() {
        let file: syn::File = syn::parse_str(
            "struct SubComponentInputs {\n\
                 verify_instruction: [Vec<A>; 1],\n\
                 memory_address_to_id: [Vec<B>; 2],\n\
                 memory_id_to_big: [Vec<C>; 2],\n\
             }",
        )
        .unwrap();
        let body: syn::Block = syn::parse_str(
            "{\n\
               *sub_component_inputs.verify_instruction[0] = (pc, [a, b, c], [d, e], z);\n\
               *sub_component_inputs.memory_address_to_id[0] = x0;\n\
               *sub_component_inputs.memory_id_to_big[0] = y0;\n\
               *sub_component_inputs.memory_address_to_id[1] = x1;\n\
               *sub_component_inputs.memory_id_to_big[1] = y1;\n\
             }",
        )
        .unwrap();
        let slots = build_sub_layout(&file, &body.stmts, "sub_component_inputs").unwrap();
        let got: Vec<(String, usize, usize)> = slots
            .iter()
            .map(|s| (s.field.clone(), s.index, s.base))
            .collect();
        assert_eq!(
            got,
            vec![
                ("verify_instruction".to_string(), 0, 0),
                ("memory_address_to_id".to_string(), 0, 7),
                ("memory_address_to_id".to_string(), 1, 8),
                ("memory_id_to_big".to_string(), 0, 9),
                ("memory_id_to_big".to_string(), 1, 10),
            ]
        );
        assert_eq!(slots.iter().map(|s| s.shape.scalar_count()).sum::<usize>(), 11);
    }

    #[test]
    fn sub_layout_missing_assignment_is_loud() {
        let file: syn::File =
            syn::parse_str("struct SubComponentInputs { memory_address_to_id: [Vec<B>; 2] }")
                .unwrap();
        let body: syn::Block =
            syn::parse_str("{ *sub_component_inputs.memory_address_to_id[0] = x0; }").unwrap();
        let err = build_sub_layout(&file, &body.stmts, "sub_component_inputs").unwrap_err();
        assert!(err.detail.contains("never assigned"), "{}", err.detail);
    }

    #[test]
    fn sub_words_use_declaration_order_bases() {
        let file: syn::File = syn::parse_str(
            "struct SubComponentInputs {\n\
                 memory_address_to_id: [Vec<B>; 1],\n\
                 memory_id_to_big: [Vec<C>; 1],\n\
             }",
        )
        .unwrap();
        // FILE order writes memory_id_to_big FIRST; its flat base must still be 1.
        let body_src = "*sub_component_inputs.memory_id_to_big[0] = add_opcode_input.pc;\n\
                        *sub_component_inputs.memory_address_to_id[0] = add_opcode_input.ap;";
        let block: syn::Block = syn::parse_str(&format!("{{ {body_src} }}")).unwrap();
        let slots = build_sub_layout(&file, &block.stmts, "sub_component_inputs").unwrap();
        let lw = lower_snippet_with_slots(&[], body_src, slots);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        let s = lw
            .out
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let big_pos = s.find("set_sub_input_word (1").expect("big word at flat 1");
        let addr_pos = s.find("set_sub_input_word (0").expect("addr word at flat 0");
        // File order: big (flat 1) is EMITTED before addr (flat 0).
        assert!(
            big_pos < addr_pos,
            "emission must follow file order with decl-order indices"
        );
    }

    #[test]
    fn idempotent_block_generation() {
        // Emitting from the same AST twice yields byte-identical output.
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../crates/prover/src/witness/components/assert_eq_opcode.rs"
        ));
        if !path.exists() {
            return; // skip if run outside the repo layout
        }
        let a1 = analyze_file(&path, true);
        let a2 = analyze_file(&path, true);
        assert!(a1.matched, "assert_eq_opcode should match");
        assert_eq!(a1.block, a2.block, "block generation must be deterministic");
    }
}
