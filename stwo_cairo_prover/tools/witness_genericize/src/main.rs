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
//!   * `#[cfg(test)]` private `lookup_data_flat` / `sub_inputs_flat` + `pub(crate) struct
//!     GenericSimdDiff` + `pub(crate) fn generic_simd_diff(...)`.
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
    BinOp, Expr, ExprArray, ExprAssign, ExprBinary, ExprCall, ExprField, ExprIndex, ExprMethodCall,
    ExprParen, ExprPath, ExprTuple, ExprUnary, Fields, FnArg, Ident, Item, ItemFn, Lit, Local,
    Member, Pat, Stmt, Type, UnOp,
};

/// Marker delimiting the generated block inside a component file (for idempotent re-run).
pub const BEGIN_MARKER: &str = "// === BEGIN witness_genericize (generated; re-runnable) ===";
pub const END_MARKER: &str = "// === END witness_genericize ===";

// ======================================================================================
// Types (the type map of the rewrite table)
// ======================================================================================

/// Bottom-up inferred type of a value in the per-row body's single-assignment let graph.
///
/// TWO felt types with DIFFERENT limb layouts exist in the generated writers and must
/// NEVER be conflated (a wrong width silently mis-lowers `get_m31`):
///   * `Felt252` — 28 limbs x 9 bits (`FELT252_N_WORDS`/`FELT252_BITS_PER_WORD`, common
///     `prover_types/cpu.rs`); this is the ONLY width the recording layer models (`FELT_N_LIMBS =
///     28`, `witness_eval/mod.rs`).
///   * `FeltW27` — `Felt252Width27`: 10 limbs x 27 bits (`FELT252WIDTH27_N_WORDS`); NOT
///     representable as a `WitnessEval::Felt` today, so opaque W27 values are census-only
///     (`w27_sites`).
///   * `FeltW27Limbs` — a W27 value the transformer itself assembled from 10 known M31 limb tokens
///     (bound as a `[E::M31; 10]` array); `get_m31(i)` projects `tok[i]`. Same canonical-limb
///     contract as the recording layer's `felt_from_limbs` (limbs assumed < 2^27; the per-component
///     byte-equality gate is the arbiter).
#[derive(Clone, PartialEq)]
enum Ty {
    M31,
    U16,
    /// u32 family — CENSUS-ONLY: typing these ops classifies files as "matched (needs
    /// u32 trait extension)"; they are never emitted.
    U32,
    Mask,
    /// felt252, 28 x 9-bit limbs (the recording layer's `Felt`).
    Felt252,
    /// `Felt252Width27`, 10 x 27-bit limbs — OPAQUE (from input / deduce); census-only.
    FeltW27,
    /// `Felt252Width27` whose 10 M31 limb values are transformer-known SSA tokens.
    FeltW27Limbs,
    ConstM31(u32),
    ConstU16(u32),
    ConstU32(u32),
    /// Hoisted `PackedFelt252::broadcast(Felt252::from([A,B,C,D]))` constant, decomposed
    /// at transform time into its 28 canonical 9-bit limbs (G3).
    ConstFelt252([u32; FELT252_LIMBS]),
    Tuple(Vec<Ty>),
    Array(Box<Ty>, usize),
    /// A (projection of the) builtin input binder, carrying the FLAT SLOT BASE of this
    /// subtree in the component's input-word layout (M31 leaf = 1 word, Felt252 leaf =
    /// 28, FeltW27 = 10, aggregates = sum — [`Ty::flat_width`]). Projections descend
    /// with the correct base; an M31 leaf lowers to the REAL `eval.input(<slot>)` read
    /// (the builtin lane feeds the flattened words in this exact depth-first order).
    /// Felt-typed leaves stay census-only (`input_sites`) until the lane feeds felt
    /// limbs. Never originates anywhere but [`Lowerer::input_leaf`].
    InputAt(Box<Ty>, usize),
    Unknown,
}

/// Felt252 limb shape: 28 limbs x 9 bits (see common `prover_types/cpu.rs`).
const FELT252_LIMBS: usize = 28;
const FELT252_LIMB_BITS: usize = 9;
/// Felt252Width27 limb shape: 10 limbs x 27 bits.
const FELTW27_LIMBS: usize = 10;

impl std::fmt::Debug for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::M31 => write!(f, "M31"),
            Ty::U16 => write!(f, "U16"),
            Ty::U32 => write!(f, "U32"),
            Ty::Mask => write!(f, "Mask"),
            Ty::Felt252 => write!(f, "Felt252"),
            Ty::FeltW27 => write!(f, "FeltW27"),
            Ty::FeltW27Limbs => write!(f, "FeltW27Limbs"),
            Ty::ConstM31(v) => write!(f, "ConstM31({v})"),
            Ty::ConstU16(v) => write!(f, "ConstU16({v})"),
            Ty::ConstU32(v) => write!(f, "ConstU32({v})"),
            // Payload elided: 28 limb values would flood the skip census keys.
            Ty::ConstFelt252(_) => write!(f, "ConstFelt252"),
            Ty::Tuple(v) => f.debug_tuple("Tuple").field(v).finish(),
            Ty::Array(e, n) => write!(f, "Array({e:?}, {n})"),
            Ty::InputAt(e, base) => write!(f, "InputAt({e:?}, {base})"),
            Ty::Unknown => write!(f, "Unknown"),
        }
    }
}

impl Ty {
    fn is_m31(&self) -> bool {
        matches!(self, Ty::M31 | Ty::ConstM31(_))
    }
    fn is_u16(&self) -> bool {
        matches!(self, Ty::U16)
    }
    /// Felt-shaped operand: a live `E::Felt` value or a hoisted felt constant
    /// (materialized to `felt_from_limbs` of constants at use).
    fn is_feltish(&self) -> bool {
        matches!(self, Ty::Felt252 | Ty::ConstFelt252(_))
    }
    fn is_u32(&self) -> bool {
        matches!(self, Ty::U32 | Ty::ConstU32(_))
    }
    fn is_mask(&self) -> bool {
        matches!(self, Ty::Mask)
    }

    /// FLAT INPUT-WORD width of this type in the builtin lane's input layout
    /// (depth-first): M31/U16/U32 leaves = 1 word, Felt252 = 28 limb words, FeltW27 =
    /// 10, aggregates = sum. This is the contract between the transformer's slot map,
    /// the emitted SIMD driver's input flattening, and the device lane's input columns
    /// — all three MUST agree or `input(slot)` reads the wrong word.
    fn flat_width(&self) -> usize {
        match self {
            Ty::Tuple(v) => v.iter().map(Ty::flat_width).sum(),
            Ty::Array(e, n) => e.flat_width() * n,
            Ty::Felt252 | Ty::ConstFelt252(_) => FELT252_LIMBS,
            Ty::FeltW27 | Ty::FeltW27Limbs => FELTW27_LIMBS,
            Ty::InputAt(e, _) => e.flat_width(),
            _ => 1,
        }
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
    /// A full-32-bit (`PackedUInt32`) element: ONE flat word carrying a raw u32 (the
    /// blake_g feeds). The flat transport is raw lanes, so nothing is lost.
    U32,
    /// A `PackedFelt252`-valued element: 28 flat limb words (canonical 9-bit limbs, the
    /// same `felt_get_m31` decomposition everywhere else in the lane). The driver
    /// reconstructs it with `PackedFelt252::from_limbs` — the exact inverse for
    /// canonical limbs, so the receiving component sees the identical felt value.
    Felt,
    Tuple(Vec<Shape>),
    Array(Vec<Shape>),
}

impl Shape {
    fn scalar_count(&self) -> usize {
        match self {
            Shape::Scalar | Shape::U32 => 1,
            Shape::Felt => FELT252_LIMBS,
            Shape::Tuple(v) | Shape::Array(v) => v.iter().map(Shape::scalar_count).sum(),
        }
    }
}

/// One flattened sub-input word at lowering time: its value token and whether it is a
/// full-32-bit word (stored via `set_sub_input_word_u32`) or a canonical M31 word.
struct SubLeaf {
    tok: TokenStream,
    u32: bool,
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
    /// Census-only builtin-input access sites (see `Lowerer::input_sites`).
    input_sites: usize,
    /// Census-only opaque-Width27 sites (see `Lowerer::w27_sites`).
    w27_sites: usize,
    /// Whether the body reads the row-index iota (`seq.packed_at(row_index)` — a REAL
    /// `eval.iota()` op; the record/driver assign it an input slot after the flat words).
    uses_iota: bool,
    /// Census-only KNOWN-SIGNATURE deduce sites (see `Lowerer::deduce_sites`).
    deduce_sites: usize,
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
        input_sites: 0,
        w27_sites: 0,
        uses_iota: false,
        deduce_sites: 0,
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

    // Collect hoisted constants (scalar broadcast + felt broadcast + Seq) + locate the
    // `for_each` closure.
    let mut consts: BTreeMap<String, ConstVal> = BTreeMap::new();
    let mut felt_consts: BTreeMap<String, [u32; FELT252_LIMBS]> = BTreeMap::new();
    let mut seq_idents: BTreeSet<String> = BTreeSet::new();
    for st in &writer.block.stmts {
        if let Stmt::Local(local) = st {
            if let (Some(name), Some(cv)) = (local_ident(local), local_const(local)) {
                consts.insert(name, cv);
            } else if let (Some(name), Some(words)) = (local_ident(local), local_felt_const(local))
            {
                felt_consts.insert(name, felt252_const_limbs(words));
            } else if let Some(name) = local_seq_ident(local) {
                seq_idents.insert(name);
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
    let (row_index_name, binders) = match closure_binders(&closure.inputs) {
        Some(b) => b,
        None => {
            fa.file_skip = Some(Skip {
                category: "skeleton",
                detail: format!(
                    "unrecognized closure binder: `{}`",
                    tok_str(&closure.inputs[0])
                ),
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

    // Scan-time felt recognizer for the sub-input layout: a sub element is
    // felt-valued when it is (a) a hoisted felt broadcast constant ident, or (b) a
    // projection chain rooted at a `let x = PackedX::deduce_output(..)` binding whose
    // KNOWN result type resolves to `Felt252` at that path. The flatten side re-checks
    // the LOWERED type per leaf and skips loudly on any disagreement (fail-closed) —
    // this recognizer only sets the flat WIDTH layout, never semantics.
    // Derive the SubComponentInputs DECLARATION-ORDER flat layout: struct fields ×
    // array lengths × the DECLARED element shapes (the host-typed ground truth).
    let sub_slots = match build_sub_layout(&file, body_stmts, &sub_name, path.parent()) {
        Ok(l) => l,
        Err(s) => {
            fa.file_skip = Some(s);
            return fa;
        }
    };
    fa.n_sub_words = sub_slots.iter().map(|s| s.shape.scalar_count()).sum();

    // Parse the packed-input type alias so the input binder's projections can be typed.
    let input_ty = parse_input_type(&file);

    // Run the lowering (collects skips + builds SSA).
    let mut lw = Lowerer::new(
        consts,
        felt_consts,
        seq_idents,
        addr_state,
        big_state,
        input_name.clone(),
        input_ty,
        row_index_name,
        row_name,
        lookup_name,
        sub_name,
        lookup_fields.clone(),
        sub_slots,
    );
    lw.lower_body(body_stmts);

    fa.n_cols = lw.max_col.map(|m| m + 1).unwrap_or(0);
    fa.u32_sites = lw.u32_sites;
    fa.input_sites = lw.input_sites;
    fa.w27_sites = lw.w27_sites;
    fa.uses_iota = lw.uses_iota;
    fa.deduce_sites = lw.deduce_sites;
    fa.skips = lw.skips.clone();
    // Emittable only when there are no skips AND no census-only sites (u32 / builtin
    // input / opaque Width27 / row-index / known-signature deduce). A census-only site is
    // typed correctly but has no backend op yet, so it must NEVER be emitted — an honest
    // "needs trait extension" classification.
    fa.matched = fa.skeleton_ok
        && fa.skips.is_empty()
        && lw.u32_sites == 0
        && lw.input_sites == 0
        && lw.w27_sites == 0
        && lw.deduce_sites == 0;
    fa.matched_u32 = fa.skeleton_ok && fa.skips.is_empty() && !fa.matched;

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

/// Match `|(row_index, (a, b, c, d))|` → returns (row-index binder name, inner binder
/// names [a, b, c, d]).
fn closure_binders(
    inputs: &syn::punctuated::Punctuated<Pat, syn::token::Comma>,
) -> Option<(String, Vec<String>)> {
    let first = inputs.first()?;
    let outer = match first {
        Pat::Tuple(t) => t,
        _ => return None,
    };
    if outer.elems.len() != 2 {
        return None;
    }
    // outer.elems[0] is row_index; outer.elems[1] is the inner tuple.
    let row_index = match &outer.elems[0] {
        Pat::Ident(pi) => pi.ident.to_string(),
        _ => return None,
    };
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
    Some((row_index, names))
}

/// Whether the module's `InteractionClaimGenerator` carries a real-row count
/// (`n_rows`) alongside `log_size` + `lookup_data` — selects the accessor
/// macro's ctor variant.
fn igen_has_n_rows(file: &syn::File) -> bool {
    file.items.iter().any(|it| {
        matches!(it,
            Item::Struct(s) if s.ident == "InteractionClaimGenerator"
                && matches!(&s.fields, syn::Fields::Named(f)
                    if f.named.iter().any(|fld| fld.ident.as_ref().is_some_and(|i| i == "n_rows"))))
    })
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
            detail: format!(
                "LookupData.{name}: unrecognized field type `{}`",
                tok_str(&f.ty)
            ),
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

/// Parse the component's `pub type PackedInputType = <ty>;` alias into a `Ty` tree, so the
/// 4th closure binder (`<comp>_input`) can be typed and its `.N` / `[i]` / `.get_m31(i)`
/// projections resolved. Unrecognized leaves (e.g. the opcode `PackedCasmState` struct,
/// whose fields are read via the named `input(SLOT_*)` path, not projections) map to
/// `Ty::Unknown` — an honest fallthrough, never a fabricated type.
fn parse_input_type(file: &syn::File) -> Ty {
    let alias = file.items.iter().find_map(|it| match it {
        Item::Type(t) if t.ident == "PackedInputType" => Some(&*t.ty),
        _ => None,
    });
    match alias {
        Some(ty) => syn_type_to_ty(ty),
        None => Ty::Unknown,
    }
}

/// Map a packed-input `syn::Type` to the inferred `Ty`. Only the shapes the generated
/// builtin inputs use are recognized; everything else is `Ty::Unknown` (honest skip).
fn syn_type_to_ty(ty: &Type) -> Ty {
    match ty {
        Type::Paren(p) => syn_type_to_ty(&p.elem),
        Type::Group(g) => syn_type_to_ty(&g.elem),
        Type::Tuple(t) => Ty::Tuple(t.elems.iter().map(syn_type_to_ty).collect()),
        Type::Array(a) => match expr_usize(&a.len) {
            Some(n) => Ty::Array(Box::new(syn_type_to_ty(&a.elem)), n),
            None => Ty::Unknown,
        },
        Type::Path(p) => match p.path.segments.last() {
            Some(seg) => {
                let name = seg.ident.to_string();
                if name == "PackedM31" {
                    Ty::M31
                } else if name == "PackedUInt16" {
                    Ty::U16
                } else if name == "PackedUInt32" {
                    Ty::U32
                } else if name == "PackedFelt252" {
                    // 28 x 9-bit limbs.
                    Ty::Felt252
                } else if name == "PackedFelt252Width27" {
                    // 10 x 27-bit limbs — a DIFFERENT layout; never conflate (G1).
                    Ty::FeltW27
                } else {
                    Ty::Unknown
                }
            }
            None => Ty::Unknown,
        },
        _ => Ty::Unknown,
    }
}

/// Shape of a declared sub-input element type `T` (from `[Vec<T>; N]`): the
/// DECLARATION is the ground truth for the flat layout + typed reconstruction (RHS
/// expressions cannot always be type-walked — e.g. u32 locals built via
/// `from_limbs`). Recognized leaves mirror `syn_type_to_ty`.
fn shape_from_syn_type(ty: &Type, dir: Option<&Path>) -> Option<Shape> {
    match ty {
        Type::Paren(p) => shape_from_syn_type(&p.elem, dir),
        Type::Group(g) => shape_from_syn_type(&g.elem, dir),
        Type::Tuple(t) => Some(Shape::Tuple(
            t.elems
                .iter()
                .map(|e| shape_from_syn_type(e, dir))
                .collect::<Option<Vec<_>>>()?,
        )),
        Type::Array(a) => {
            let n = expr_usize(&a.len)?;
            let e = shape_from_syn_type(&a.elem, dir)?;
            Some(Shape::Array(vec![e; n]))
        }
        Type::Path(p) => {
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            match segs.last()?.as_str() {
                "PackedM31" => Some(Shape::Scalar),
                "PackedUInt32" => Some(Shape::U32),
                "PackedFelt252" => Some(Shape::Felt),
                // `<component>::PackedInputType` — resolve by parsing the SIBLING
                // component file's alias (the transformer runs over the components
                // dir, so the sibling is on disk next to the current file).
                "PackedInputType" if segs.len() == 2 => {
                    let dir = dir?;
                    let sibling = dir.join(format!("{}.rs", segs[0]));
                    let ty = sibling_input_ty(&sibling)?;
                    ty_to_shape(&ty)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Parse (and cache) a sibling component file's `PackedInputType` alias as a `Ty`.
fn sibling_input_ty(path: &Path) -> Option<Ty> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<BTreeMap<PathBuf, Option<Ty>>>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(BTreeMap::new);
    if let Some(t) = cache.get(path) {
        return t.clone();
    }
    let ty = std::fs::read_to_string(path)
        .ok()
        .and_then(|src| syn::parse_file(&src).ok())
        .map(|file| parse_input_type(&file))
        .filter(|t| !matches!(t, Ty::Unknown));
    cache.insert(path.to_path_buf(), ty.clone());
    ty
}

/// Convert an input `Ty` tree to a flat sub-word `Shape` (leaves must be M31 / U32 /
/// Felt252; anything else — e.g. a FeltW27 — is unsupported and returns None loudly
/// upstream, never a silent width guess).
fn ty_to_shape(ty: &Ty) -> Option<Shape> {
    match ty {
        Ty::M31 => Some(Shape::Scalar),
        Ty::U32 => Some(Shape::U32),
        Ty::Felt252 => Some(Shape::Felt),
        Ty::Tuple(v) => Some(Shape::Tuple(
            v.iter().map(ty_to_shape).collect::<Option<Vec<_>>>()?,
        )),
        Ty::Array(e, n) => {
            let s = ty_to_shape(e)?;
            Some(Shape::Array(vec![s; *n]))
        }
        _ => None,
    }
}

/// Parse `struct SubComponentInputs` field declarations: (name, array_len, DECLARED
/// element shape) in order. Field types are `[Vec<T>; N]`.
fn parse_sub_struct(
    file: &syn::File,
    dir: Option<&Path>,
) -> Result<Vec<(String, usize, Shape)>, Skip> {
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
        // [Vec<T>; N] -> T's declared shape.
        let elem_shape = (|| {
            let Type::Path(p) = &*arr.elem else {
                return None;
            };
            let seg = p.path.segments.last()?;
            if seg.ident != "Vec" {
                return None;
            }
            let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
                return None;
            };
            let syn::GenericArgument::Type(t) = args.args.first()? else {
                return None;
            };
            shape_from_syn_type(t, dir)
        })();
        let Some(elem_shape) = elem_shape else {
            return Err(Skip {
                category: "skeleton",
                detail: format!(
                    "SubComponentInputs.{name}: unrecognized element type `{}`",
                    tok_str(&arr.elem)
                ),
            });
        };
        out.push((name, len, elem_shape));
    }
    Ok(out)
}

/// Match the generated multiplicity-column read idiom
/// `* mults [k] . get (row_index) . unwrap_or (& PackedM31 :: zero ())`,
/// returning `k`. Anything that deviates (a different base ident, a non-literal
/// index, a different default) does NOT match and falls through to the loud
/// unsupported-expression skip — never a silent approximation.
fn match_mults_read(expr: &Expr, row_index_name: &str) -> Option<usize> {
    let Expr::Unary(ExprUnary {
        op: UnOp::Deref(_),
        expr: inner,
        ..
    }) = strip_parens(expr)
    else {
        return None;
    };
    // .unwrap_or(&PackedM31::zero())
    let Expr::MethodCall(unwrap) = strip_parens(inner) else {
        return None;
    };
    if unwrap.method != "unwrap_or" || unwrap.args.len() != 1 {
        return None;
    }
    let default_ok = matches!(
        strip_parens(unwrap.args.first().unwrap()),
        Expr::Reference(r) if tok_str(&r.expr) == "PackedM31 :: zero ()"
    );
    if !default_ok {
        return None;
    }
    // .get(row_index)
    let Expr::MethodCall(get) = strip_parens(&unwrap.receiver) else {
        return None;
    };
    if get.method != "get"
        || get.args.len() != 1
        || !is_path_named(get.args.first().unwrap(), row_index_name)
    {
        return None;
    }
    // mults[k]
    let Expr::Index(ExprIndex {
        expr: base, index, ..
    }) = strip_parens(&get.receiver)
    else {
        return None;
    };
    if !is_path_named(base, "mults") {
        return None;
    }
    expr_usize(index)
}

/// Pre-scan the closure body's top-level statements for
/// `*<sub_name>.<field>[k] = rhs;` and derive the DECLARATION-ORDER flat layout.
fn build_sub_layout(
    file: &syn::File,
    body_stmts: &[Stmt],
    sub_name: &str,
    dir: Option<&Path>,
) -> Result<Vec<SubSlot>, Skip> {
    // Collect assigned (field, k) sites for coverage checking (file order). The slot
    // SHAPES come from the SubComponentInputs DECLARATION — the ground truth the host
    // type checker already enforces on every RHS.
    let mut seen: BTreeSet<(String, usize)> = BTreeSet::new();
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
        let Expr::Index(ExprIndex {
            expr: base, index, ..
        }) = strip_parens(place)
        else {
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
        if !seen.insert((field.clone(), k)) {
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

    let decl = parse_sub_struct(file, dir)?;
    // Every observed field must be declared; every declared (field,k) must be assigned.
    let declared: BTreeSet<&String> = decl.iter().map(|(n, ..)| n).collect();
    for (field, k) in seen.iter() {
        if !declared.contains(field) {
            return Err(Skip {
                category: "effect",
                detail: format!("sub-input `{field}[{k}]` not declared in SubComponentInputs"),
            });
        }
    }
    let mut slots = Vec::new();
    let mut base = 0usize;
    for (field, len, elem_shape) in &decl {
        for k in 0..*len {
            if !seen.contains(&(field.clone(), k)) {
                return Err(Skip {
                    category: "effect",
                    detail: format!("sub-input `{field}[{k}]` declared but never assigned"),
                });
            }
            let count = elem_shape.scalar_count();
            slots.push(SubSlot {
                field: field.clone(),
                index: k,
                shape: elem_shape.clone(),
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
                    self.hits
                        .push(format!("{}::deduce_output", segs.join("::")));
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
    /// Hoisted felt broadcast constants (G3), pre-decomposed into 28 canonical 9-bit
    /// limbs at transform time.
    felt_consts: BTreeMap<String, [u32; FELT252_LIMBS]>,
    /// Preamble `let <name> = Seq::new(..)` idents; `<name>.packed_at(row_index)` is the
    /// packed row index (census-only until the lane feeds it as an input word, G4).
    seq_idents: BTreeSet<String>,
    addr_state: Option<String>,
    big_state: Option<String>,
    input_name: String,
    /// Type of the 4th closure binder (`<comp>_input`), parsed from `PackedInputType`.
    /// Seeds input-projection typing (`.N` / `[i]` / `.get_m31(i)`).
    input_ty: Ty,
    /// The closure's outer row-index binder name (`row_index`).
    row_index_name: String,
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
    /// Census-only builtin-input access sites (`<comp>_input.N` / `[i]` / `.get_m31(i)`).
    /// Typed correctly but NOT emittable: `SimdWitnessEval`/recording model only the
    /// opcode `PackedCasmState` input (`input(SLOT_PC/AP/FP)`), so a builtin felt-tuple
    /// input has no read op yet. Any site > 0 blocks emission (like `u32_sites`).
    input_sites: usize,
    /// Census-only OPAQUE `Felt252Width27` sites: `get_m31(i)` on a W27 whose limbs the
    /// transformer does not hold (input/deduce-sourced), and the W27→Felt252 width
    /// conversion (needs `U32Shr`/`U32And` — 27-bit limbs exceed the u16 trait ops).
    /// The recording layer models only 28x9 felts (`FELT_N_LIMBS`), so these are typed
    /// correctly but never emitted. Any site > 0 blocks emission.
    w27_sites: usize,
    /// Whether the body reads the row-index iota (`seq.packed_at(row_index)`) — a REAL
    /// `eval.iota()` op (G4); the builtin record/driver assign it an input slot.
    uses_iota: bool,
    /// Multiplicity-column reads (`*mults[k].get(row_index).unwrap_or(&zero)`): REAL
    /// `eval.input(K + 2 + k)` reads — the builtin lane feeds `mults[k]` as an input
    /// column after the flat input words, the enabler and the iota. The set records
    /// which `k` the body reads (the driver emits exactly those columns).
    mults_reads: BTreeSet<usize>,
    /// Census-only KNOWN-SIGNATURE deduce sites (G5): `PackedX::deduce_output(..)` calls
    /// whose RESULT type is in [`known_deduce_output_ty`]'s table. Typing the result lets
    /// every downstream projection (`.N` / `[i]` / `.get_m31(i)`) resolve — collapsing the
    /// cascade of Unknown skips to the honest per-call deduce count — while the call
    /// itself stays census-only: it needs either a computed-deduce ISA op backed by a
    /// device function (ec_ops.cuh) or device-to-device component feeding. Blocks emission.
    deduce_sites: usize,
    counter: usize,

    max_col: Option<usize>,
}

impl Lowerer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        consts: BTreeMap<String, ConstVal>,
        felt_consts: BTreeMap<String, [u32; FELT252_LIMBS]>,
        seq_idents: BTreeSet<String>,
        addr_state: Option<String>,
        big_state: Option<String>,
        input_name: String,
        input_ty: Ty,
        row_index_name: String,
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
            felt_consts,
            seq_idents,
            addr_state,
            big_state,
            input_name,
            input_ty,
            row_index_name,
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
            input_sites: 0,
            w27_sites: 0,
            uses_iota: false,
            mults_reads: BTreeSet::new(),
            deduce_sites: 0,
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

    /// Census-only OPAQUE-Width27 site (see `w27_sites`): typing proceeds, emission is
    /// blocked. NEVER lowered to `felt_get_m31` — that op is 28x9 semantics and using it
    /// on a 10x27 value would be a silent miscompile if it ever reached emission.
    fn w27_site(&mut self, ty: Ty) -> (Ty, TokenStream) {
        self.w27_sites += 1;
        (ty, quote! { WG_W27_CENSUS_ONLY })
    }

    /// Census-only KNOWN-SIGNATURE deduce site (G5): the call's RESULT is typed from
    /// [`known_deduce_output_ty`] so downstream projections resolve, but the deduce
    /// itself has no backend op yet (device EC/blake function or device-to-device feed).
    fn deduce_site(&mut self, ty: Ty) -> (Ty, TokenStream) {
        self.deduce_sites += 1;
        (ty, quote! { WG_DEDUCE_CENSUS_ONLY })
    }

    /// Lower one expression expected to produce an `E::Felt` value: a felt-typed
    /// expression as-is, or a hoisted felt constant materialized via
    /// `felt_from_limbs` over its 28 const limbs. `None` = not a felt here.
    /// Materialize a u32-shaped (ty, tok) pair as an `E::U32` value token —
    /// hoisted broadcast constants become `eval.u32_const(v)`.
    fn u32ish_value(&mut self, ty: Ty, tok: TokenStream) -> TokenStream {
        match ty {
            Ty::ConstU32(v) => {
                let vl = u32_lit(v);
                self.bind(Target::Temp, quote! { eval.u32_const(#vl) })
            }
            _ => tok,
        }
    }

    /// Materialize a felt-shaped (ty, tok) pair as an `E::Felt` value token —
    /// constants become `felt_from_limbs` of hoisted limb constants.
    fn feltish_value(&mut self, ty: Ty, tok: TokenStream) -> TokenStream {
        match ty {
            Ty::ConstFelt252(limbs) => self.felt_const_value(Target::Temp, limbs).1,
            _ => tok,
        }
    }

    fn lower_felt_value(&mut self, e: &Expr) -> Option<TokenStream> {
        if let Some(limbs) = self.peek_felt_const(e) {
            let (_t, tok) = self.felt_const_value(Target::Temp, limbs);
            return Some(tok);
        }
        let (ty, tok) = self.lower_node(strip_parens(e), Target::Temp);
        match ty {
            Ty::Felt252 => Some(tok),
            Ty::ConstFelt252(limbs) => {
                let (_t, tok) = self.felt_const_value(Target::Temp, limbs);
                Some(tok)
            }
            _ => None,
        }
    }

    /// `PackedPartialEcMulWindowBits18::deduce_output((chain, round, ([w; 14],
    /// [acc; 2])))` — the generated literal shape — as a REAL
    /// `eval.deduce_partial_ec_mul_w18(chain, round, [w; 14], [acc0, acc1])` call.
    /// `None` when the argument is not the literal tuple shape (caller falls back to
    /// the census-only site; nothing is emitted before the shape checks pass).
    fn lower_w18_deduce(&mut self, call: &syn::ExprCall, target: Target) -> Option<TokenStream> {
        let arg = strip_parens(call.args.first()?);
        let Expr::Tuple(ExprTuple { elems, .. }) = arg else {
            return None;
        };
        let [chain_e, round_e, state_e] = elems.iter().collect::<Vec<_>>()[..] else {
            return None;
        };
        let Expr::Tuple(ExprTuple { elems: st, .. }) = strip_parens(state_e) else {
            return None;
        };
        let [wins_e, acc_e] = st.iter().collect::<Vec<_>>()[..] else {
            return None;
        };
        let Expr::Array(ExprArray { elems: wins, .. }) = strip_parens(wins_e) else {
            return None;
        };
        let Expr::Array(ExprArray { elems: accs, .. }) = strip_parens(acc_e) else {
            return None;
        };
        if wins.len() != 14 || accs.len() != 2 {
            return None;
        }
        // Shape checks passed — lower the pieces (any inner mismatch is a loud skip
        // from the piece's own lowering; the call still emits so the census stays 1:1
        // with the source, and the skip blocks emission).
        let chain = {
            let (ty, tok) = self.lower_node(strip_parens(chain_e), Target::Temp);
            self.require_m31(&ty, "w18 deduce chain", chain_e);
            tok
        };
        let round = {
            let (ty, tok) = self.lower_node(strip_parens(round_e), Target::Temp);
            self.require_m31(&ty, "w18 deduce round", round_e);
            tok
        };
        let win_toks: Vec<TokenStream> = wins
            .iter()
            .map(|w| {
                let (ty, tok) = self.lower_node(strip_parens(w), Target::Temp);
                self.require_m31(&ty, "w18 deduce window", w);
                tok
            })
            .collect();
        let acc_toks: Vec<TokenStream> = accs
            .iter()
            .map(|a| match self.lower_felt_value(a) {
                Some(tok) => tok,
                None => {
                    self.skip(
                        "deduce_output",
                        format!("w18 deduce accumulator is not a felt: `{}`", tok_str(a)),
                    );
                    quote! { WG_SKIP }
                }
            })
            .collect();
        Some(self.bind(
            target,
            quote! { eval.deduce_partial_ec_mul_w18(#chain, #round, [ #(#win_toks),* ], [ #(#acc_toks),* ]) },
        ))
    }

    /// `PackedBlakeG::deduce_output([a, b, c, d, m0, m1])` (6 full-32-bit words) as a
    /// REAL `eval.deduce_blake_g([...])` call. `None` when the literal array shape is
    /// absent (fallback: census-only).
    fn lower_blake_g_deduce(
        &mut self,
        call: &syn::ExprCall,
        target: Target,
    ) -> Option<TokenStream> {
        let arg = strip_parens(call.args.first()?);
        let Expr::Array(ExprArray { elems, .. }) = arg else {
            return None;
        };
        if elems.len() != 6 {
            return None;
        }
        let toks: Vec<TokenStream> = elems
            .iter()
            .map(|e| {
                let (ty, tok) = self.lower_node(strip_parens(e), Target::Temp);
                if !ty.is_u32() {
                    self.skip(
                        "deduce_output",
                        format!("blake_g input is not u32: `{}` ({ty:?})", tok_str(e)),
                    );
                }
                tok
            })
            .collect();
        Some(self.bind(target, quote! { eval.deduce_blake_g([ #(#toks),* ]) }))
    }

    /// `PackedPedersenPointsTableWindowBits18::deduce_output([index])` as a REAL
    /// `eval.deduce_pedersen_points_table_w18(index)` call.
    fn lower_points_table_deduce(
        &mut self,
        call: &syn::ExprCall,
        target: Target,
    ) -> Option<TokenStream> {
        let arg = strip_parens(call.args.first()?);
        let Expr::Array(ExprArray { elems, .. }) = arg else {
            return None;
        };
        if elems.len() != 1 {
            return None;
        }
        let (ty, idx) = self.lower_node(strip_parens(&elems[0]), Target::Temp);
        self.require_m31(&ty, "points-table deduce index", &elems[0]);
        Some(self.bind(
            target,
            quote! { eval.deduce_pedersen_points_table_w18(#idx) },
        ))
    }

    /// A hoisted felt broadcast constant used as a bare VALUE (not via `.get_m31(i)`,
    /// which short-circuits to the const limb): materialize it through the REAL
    /// `felt_from_limbs` op over 28 M31 constants (RecFelt::Limbs of consts — no ISA
    /// change, G3). Byte-correct: the limbs are the canonical 9-bit windows, so the SIMD
    /// impl's `from_limbs` repacks exactly the broadcast value.
    fn felt_const_value(
        &mut self,
        target: Target,
        limbs: [u32; FELT252_LIMBS],
    ) -> (Ty, TokenStream) {
        let ids: Vec<Ident> = limbs
            .iter()
            .map(|v| {
                self.referenced_m31.insert(*v);
                Ident::new(&format!("m31_{v}"), Span::call_site())
            })
            .collect();
        self.emit_op(
            target,
            Ty::ConstFelt252(limbs),
            quote! { eval.felt_from_limbs([ #(#ids),* ]) },
        )
    }

    /// Peek: is `expr` a bare path naming a hoisted felt constant? (Used by `get_m31` to
    /// avoid materializing the whole felt when only one const limb is read.)
    fn peek_felt_const(&self, expr: &Expr) -> Option<[u32; FELT252_LIMBS]> {
        match strip_parens(expr) {
            Expr::Path(p) => self.felt_consts.get(&tok_str(&p.path)).copied(),
            _ => None,
        }
    }

    /// A single known-const M31 limb value as a leaf.
    fn const_m31_leaf(&mut self, target: Target, v: u32) -> (Ty, TokenStream) {
        self.referenced_m31.insert(v);
        let id = Ident::new(&format!("m31_{v}"), Span::call_site());
        self.leaf(target, Ty::ConstM31(v), quote! { #id })
    }

    /// Builtin-input leaf: the parsed `PackedInputType`, wrapped in [`Ty::InputAt`]
    /// with flat slot base 0. Projections descend the wrapper with the correct base;
    /// an M31 leaf lowers to the REAL `eval.input(<slot>)` read ([`Self::input_slot_leaf`]).
    /// Felt-typed leaves stay census-only (`input_sites`) — the lane does not feed felt
    /// limbs yet. A bare (unprojected) use of a tuple-typed input, or an unparseable
    /// alias (opcode `PackedCasmState`), is an honest skip exactly as before.
    fn input_leaf(&mut self, target: Target) -> (Ty, TokenStream) {
        if matches!(self.input_ty, Ty::Unknown) {
            self.skip(
                "expr",
                format!("bare use of input struct `{}`", self.input_name),
            );
            return (Ty::Unknown, quote! { WG_SKIP });
        }
        let ty = Ty::InputAt(Box::new(self.input_ty.clone()), 0);
        self.input_projection(target, ty, "bare input binder")
    }

    /// Resolve an [`Ty::InputAt`]-typed value: M31 leaf → the REAL `eval.input(<slot>)`
    /// read; aggregate → pass the wrapper through for further projection (placeholder
    /// token — a bare aggregate use that reaches an op is a skip downstream); felt/other
    /// leaf → census-only `input_sites` (typed, not yet fed by the lane).
    fn input_projection(&mut self, target: Target, ty: Ty, what: &str) -> (Ty, TokenStream) {
        let Ty::InputAt(inner, base) = &ty else {
            unreachable!("input_projection on non-InputAt");
        };
        match &**inner {
            Ty::M31 => {
                let slot = u32_lit(*base as u32);
                self.emit_op(target, Ty::M31, quote! { eval.input(#slot) })
            }
            Ty::U32 => {
                // Full-32-bit input word (blake message words) — its own read op; the
                // device lane's u32 input columns carry it raw.
                let slot = u32_lit(*base as u32);
                self.emit_op(target, Ty::U32, quote! { eval.input_u32(#slot) })
            }
            Ty::Tuple(_) | Ty::Array(..) => self.leaf(target, ty.clone(), quote! { WG_INPUT_AGG }),
            Ty::Felt252 => {
                // Felt input leaf: 28 consecutive limb slots -> a REAL felt value via
                // `felt_from_limbs` over 28 input reads (existing ops; the lane feeds
                // the felt's canonical limbs as 28 input columns).
                let limb_ids: Vec<TokenStream> = (0..FELT252_LIMBS)
                    .map(|j| {
                        let slot = u32_lit((*base + j) as u32);
                        self.bind(Target::Temp, quote! { eval.input(#slot) })
                    })
                    .collect();
                self.emit_op(
                    target,
                    Ty::Felt252,
                    quote! { eval.felt_from_limbs([ #(#limb_ids),* ]) },
                )
            }
            Ty::FeltW27 => {
                // W27 input leaf: 10 consecutive word slots (27-bit values are
                // M31-safe) -> the limb-array value FeltW27Limbs carries.
                let word_ids: Vec<TokenStream> = (0..FELTW27_LIMBS)
                    .map(|j| {
                        let slot = u32_lit((*base + j) as u32);
                        self.bind(Target::Temp, quote! { eval.input(#slot) })
                    })
                    .collect();
                let tok = self.bind(target, quote! { [ #(#word_ids),* ] });
                (Ty::FeltW27Limbs, tok)
            }
            _ => {
                // U16 / other input leaves: typed but not yet fed by the lane.
                let _ = what;
                self.input_sites += 1;
                self.leaf(target, (**inner).clone(), quote! { WG_INPUT_CENSUS_ONLY })
            }
        }
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
                self.skip(
                    "stmt",
                    format!("unexpected expression statement: `{}`", tok_str(e)),
                );
            }
            Stmt::Macro(m) => {
                self.skip(
                    "macro",
                    format!("macro in body: `{}`", tok_str(&m.mac.path)),
                );
            }
            Stmt::Item(_) => self.skip("stmt", "nested item in body".to_string()),
        }
    }

    fn lower_local(&mut self, local: &Local) {
        let Some(name) = local_ident(local) else {
            self.skip(
                "stmt",
                format!("unsupported `let` pattern: `{}`", tok_str(&local.pat)),
            );
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
                self.skip(
                    "effect",
                    format!("assignment to non-deref place: `{}`", tok_str(other)),
                );
                return;
            }
        };
        match deref {
            // *row[i] = v;
            Expr::Index(ExprIndex {
                expr: base, index, ..
            }) if is_path_named(base, &self.row_name) => {
                let Some(col) = expr_usize(index) else {
                    self.skip(
                        "effect",
                        format!("row index not a literal: `{}`", tok_str(index)),
                    );
                    return;
                };
                let (ty, v) = self.lower_node(strip_parens(&a.right), Target::Temp);
                self.require_m31(&ty, "set_col value", &a.right);
                let cl = usize_lit(col);
                self.out.push(quote! { eval.set_col(#cl, #v); });
                self.max_col = Some(self.max_col.map_or(col, |m| m.max(col)));
            }
            // *sub_component_inputs.field[k] = <tuple/array/scalar>;
            Expr::Index(ExprIndex {
                expr: base, index, ..
            }) => {
                let field = match strip_parens(base) {
                    Expr::Field(ExprField {
                        base: fb,
                        member: Member::Named(m),
                        ..
                    }) if is_path_named(fb, &self.sub_name) => m.to_string(),
                    _ => {
                        self.skip(
                            "effect",
                            format!("unrecognized sub-input place: `{}`", tok_str(deref)),
                        );
                        return;
                    }
                };
                let Some(k) = expr_usize(index) else {
                    self.skip(
                        "effect",
                        format!("sub-input index not a literal: `{}`", tok_str(index)),
                    );
                    return;
                };
                let Some(base_idx) = self.sub_base.get(&(field.clone(), k)).copied() else {
                    self.skip(
                        "effect",
                        format!("sub-input `{field}[{k}]` missing from layout"),
                    );
                    return;
                };
                let leaves = self.flatten_sub(strip_parens(&a.right));
                // Fail-closed width guard: the scan-time shape fixed this slot's flat
                // word count (and every later slot's base). A lowered width that
                // disagrees (e.g. a felt the scan recognizer missed) would silently
                // corrupt the whole layout — skip loudly instead.
                let expected = self
                    .sub_slots
                    .iter()
                    .find(|s| s.field == field && s.index == k)
                    .map(|s| s.shape.scalar_count());
                if expected != Some(leaves.len()) {
                    self.skip(
                        "effect",
                        format!(
                            "sub-input `{field}[{k}]` flat width {} != scan layout {:?}",
                            leaves.len(),
                            expected
                        ),
                    );
                    return;
                }
                for (j, leaf) in leaves.iter().enumerate() {
                    let w = usize_lit(base_idx + j);
                    let tok = &leaf.tok;
                    if leaf.u32 {
                        self.out
                            .push(quote! { eval.set_sub_input_word_u32(#w, #tok); });
                    } else {
                        self.out.push(quote! { eval.set_sub_input_word(#w, #tok); });
                    }
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
                    self.skip(
                        "effect",
                        format!("lookup field not in LookupData: `{field}`"),
                    );
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
                                format!(
                                    "lookup field `{field}` RHS not an array: `{}`",
                                    tok_str(rhs)
                                ),
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
                self.skip(
                    "effect",
                    format!("unrecognized effect place: `{}`", tok_str(other)),
                );
            }
        }
    }

    /// Effect values must be M31-typed (Unknown means an inner skip already fired).
    fn require_m31(&mut self, ty: &Ty, what: &str, expr: &Expr) {
        if !ty.is_m31() && *ty != Ty::Unknown {
            self.skip(
                "effect",
                format!("{what} is {ty:?}, not M31: `{}`", tok_str(expr)),
            );
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
                (
                    Ty::Array(Box::new(et), toks.len()),
                    quote! { [ #(#toks),* ] },
                )
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
        // Multiplicity-column read (`*mults[k].get(row_index).unwrap_or(&zero)`): a
        // REAL per-row input read — the builtin lane feeds `mults[k]` as the input
        // column at slot K + 2 + k (after the flat inputs, the enabler and the iota;
        // the SIMD driver appends the same reads to its flat words, with placeholders
        // in the enabler/iota positions so the slot arithmetic is uniform).
        if let Some(k) = match_mults_read(expr, &self.row_index_name) {
            if matches!(self.input_ty, Ty::Unknown) {
                self.skip(
                    "expr",
                    format!("mults read outside a typed builtin: `{}`", tok_str(expr)),
                );
                return (Ty::Unknown, quote! { WG_SKIP });
            }
            self.mults_reads.insert(k);
            let slot = u32_lit((self.input_ty.flat_width() + 2 + k) as u32);
            return self.emit_op(target, Ty::M31, quote! { eval.input(#slot) });
        }
        match expr {
            Expr::Path(p) => self.lower_path(p, target),
            Expr::Field(f) => self.lower_field(f, target),
            Expr::Index(ix) => self.lower_index(ix, target),
            Expr::MethodCall(mc) => self.lower_method(mc, target),
            Expr::Call(call) => self.lower_call(call, target),
            Expr::Binary(b) => self.lower_binary(b, target),
            other => {
                self.skip(
                    "expr",
                    format!("unsupported expression: `{}`", tok_str(other)),
                );
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
        if let Some(limbs) = self.felt_consts.get(&name).copied() {
            return self.felt_const_value(target, limbs);
        }
        if name == self.input_name {
            return self.input_leaf(target);
        }
        if let Some(ty) = self.env.get(&name).cloned() {
            let id = Ident::new(&name, Span::call_site());
            // `E::Felt` is Clone-not-Copy: an alias binding (`let new = old;`)
            // must clone, or the original moves and later uses are E0382
            // (cube_252's unpack alias). Every other handle type is Copy.
            if ty == Ty::Felt252 {
                return self.leaf(target, ty, quote! { #id.clone() });
            }
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
                            self.skip(
                                "input_field",
                                format!("input.{other} (unsupported input field)"),
                            );
                            return (Ty::Unknown, quote! { WG_SKIP });
                        }
                    };
                    self.used_slots.insert(slot);
                    let slot_id = Ident::new(slot, Span::call_site());
                    return self.emit_op(target, Ty::M31, quote! { eval.input(#slot_id) });
                }
                self.skip(
                    "expr",
                    format!(
                        "field access `.{m}` on non-input base `{}`",
                        tok_str(&f.base)
                    ),
                );
                (Ty::Unknown, quote! { WG_SKIP })
            }
            Member::Unnamed(idx) => {
                // Tuple projection x.0 / x.1 ...
                let (bt, btok) = self.lower_node(&f.base, Target::Temp);
                let i = idx.index as usize;
                // Input-rooted projection: descend the wrapper with the flat slot base
                // advanced past the preceding elements' widths.
                if let Ty::InputAt(inner, base) = &bt {
                    if let Ty::Tuple(v) = &**inner {
                        if i < v.len() {
                            let child_base =
                                base + v[..i].iter().map(Ty::flat_width).sum::<usize>();
                            let child = Ty::InputAt(Box::new(v[i].clone()), child_base);
                            return self.input_projection(target, child, "input tuple elem");
                        }
                    }
                    self.skip(
                        "expr",
                        format!("tuple projection .{i} on input `{}`", tok_str(&f.base)),
                    );
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let elem_ty = match &bt {
                    Ty::Tuple(v) if i < v.len() => v[i].clone(),
                    _ => {
                        self.skip(
                            "expr",
                            format!("tuple projection .{i} on non-tuple `{}`", tok_str(&f.base)),
                        );
                        Ty::Unknown
                    }
                };
                let lit = Literal::usize_unsuffixed(i);
                // `E::Felt` is Clone (not Copy): reading a felt element out of a
                // runtime tuple must clone, or the emitted code moves out of the value.
                if matches!(elem_ty, Ty::Felt252) {
                    return self.leaf(target, elem_ty, quote! { #btok.#lit.clone() });
                }
                self.leaf(target, elem_ty, quote! { #btok.#lit })
            }
        }
    }

    fn lower_index(&mut self, ix: &ExprIndex, target: Target) -> (Ty, TokenStream) {
        let (bt, btok) = self.lower_node(&ix.expr, Target::Temp);
        let Some(i) = expr_usize(&ix.index) else {
            self.skip(
                "expr",
                format!("non-literal index: `{}`", tok_str(&ix.index)),
            );
            return (Ty::Unknown, quote! { WG_SKIP });
        };
        // Input-rooted projection: descend the wrapper, base advanced by whole elements.
        if let Ty::InputAt(inner, base) = &bt {
            if let Ty::Array(e, n) = &**inner {
                if i < *n {
                    let child_base = base + e.flat_width() * i;
                    let child = Ty::InputAt(Box::new((**e).clone()), child_base);
                    return self.input_projection(target, child, "input array elem");
                }
            }
            self.skip(
                "expr",
                format!("index [{i}] on input `{}`", tok_str(&ix.expr)),
            );
            return (Ty::Unknown, quote! { WG_SKIP });
        }
        let elem_ty = match &bt {
            Ty::Array(e, _) => (**e).clone(),
            _ => {
                self.skip(
                    "expr",
                    format!("index [{i}] on non-array `{}`", tok_str(&ix.expr)),
                );
                Ty::Unknown
            }
        };
        let lit = usize_lit(i);
        // `E::Felt` is Clone (not Copy): see the tuple-projection note.
        if matches!(elem_ty, Ty::Felt252) {
            return self.leaf(target, elem_ty, quote! { #btok[#lit].clone() });
        }
        self.leaf(target, elem_ty, quote! { #btok[#lit] })
    }

    fn lower_method(&mut self, mc: &ExprMethodCall, target: Target) -> (Ty, TokenStream) {
        let method = mc.method.to_string();
        match method.as_str() {
            "get_m31" => {
                // Hoisted felt const receiver: the limb is a transform-time constant
                // (G3) — no need to materialize the felt.
                if let Some(limbs) = self.peek_felt_const(&mc.receiver) {
                    let Some(i) = mc.args.first().and_then(expr_usize) else {
                        self.skip("expr", "get_m31 without literal index".to_string());
                        return (Ty::Unknown, quote! { WG_SKIP });
                    };
                    if i >= FELT252_LIMBS {
                        self.skip(
                            "expr",
                            format!("get_m31({i}) out of range for Felt252 (28 limbs)"),
                        );
                        return (Ty::Unknown, quote! { WG_SKIP });
                    }
                    return self.const_m31_leaf(target, limbs[i]);
                }
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                let Some(i) = mc.args.first().and_then(expr_usize) else {
                    self.skip("expr", "get_m31 without literal index".to_string());
                    return (Ty::Unknown, quote! { WG_SKIP });
                };
                let lit = usize_lit(i);
                // WIDTH-AWARE (G1): `felt_get_m31` is 28x9 semantics ONLY. A Width27
                // receiver must never route through it, and out-of-range indices are
                // SOURCE bugs that must skip loudly, not wrap.
                match rt {
                    Ty::Felt252 if i < FELT252_LIMBS => {
                        self.emit_op(target, Ty::M31, quote! { eval.felt_get_m31(&#rtok, #lit) })
                    }
                    Ty::Felt252 => {
                        self.skip(
                            "expr",
                            format!("get_m31({i}) out of range for Felt252 (28 limbs)"),
                        );
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                    Ty::ConstFelt252(limbs) if i < FELT252_LIMBS => {
                        self.const_m31_leaf(target, limbs[i])
                    }
                    Ty::ConstFelt252(_) => {
                        self.skip(
                            "expr",
                            format!("get_m31({i}) out of range for Felt252 (28 limbs)"),
                        );
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                    Ty::FeltW27 if i < FELTW27_LIMBS => self.w27_site(Ty::M31),
                    Ty::FeltW27Limbs if i < FELTW27_LIMBS => {
                        // The transformer holds the 10 limb tokens as an array value.
                        self.leaf(target, Ty::M31, quote! { #rtok[#lit] })
                    }
                    Ty::FeltW27 | Ty::FeltW27Limbs => {
                        self.skip(
                            "expr",
                            format!(
                                "get_m31({i}) out of range for Felt252Width27 (10 limbs) — \
                                 source bug"
                            ),
                        );
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                    _ => {
                        // Unknown/other receiver: record the skip; the RESULT of a
                        // source-level `get_m31` is always PackedM31, so type M31 to
                        // limit cascade noise (emission is blocked by the skip).
                        self.skip(
                            "expr",
                            format!("get_m31 on non-Felt `{}`", tok_str(&mc.receiver)),
                        );
                        self.emit_op(target, Ty::M31, quote! { eval.felt_get_m31(&#rtok, #lit) })
                    }
                }
            }
            "as_m31" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                if rt.is_u16() {
                    self.emit_op(target, Ty::M31, quote! { eval.u16_as_m31(#rtok) })
                } else if rt.is_mask() {
                    self.emit_op(target, Ty::M31, quote! { eval.mask_as_m31(#rtok) })
                } else {
                    self.skip(
                        "expr",
                        format!("as_m31 on {:?} `{}`", rt, tok_str(&mc.receiver)),
                    );
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            // u32 family (census-only): .low()/.high() split a u32 into u16 halves.
            // Cross-checked against `UInt32` (common `prover_types/cpu.rs`):
            //   .low()  = value & 0xFFFF  → ISA `Trunc16` (or `U32And` imm 0xFFFF)
            //   .high() = value >> 16     → ISA `U32Shr` imm 16
            //   from_limbs(low, high) = (low & 0xFFFF) | ((high & 0xFFFF) << 16)
            // Both halves are U16-typed; emission needs the u32 trait extension.
            "low" | "high" => {
                let (rt, rtok) = self.lower_node(&mc.receiver, Target::Temp);
                if rt.is_u32() {
                    // REAL trait ops now (u32 trait extension landed).
                    let op = Ident::new(
                        if method == "low" {
                            "u32_low"
                        } else {
                            "u32_high"
                        },
                        Span::call_site(),
                    );
                    self.emit_op(target, Ty::U16, quote! { eval.#op(#rtok) })
                } else {
                    self.skip(
                        "method",
                        format!(".{method}() on `{}`", tok_str(&mc.receiver)),
                    );
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
                    self.skip(
                        "expr",
                        format!("inverse on non-M31 `{}`", tok_str(&mc.receiver)),
                    );
                }
                self.emit_op(target, Ty::M31, quote! { eval.m31_inverse(#rtok) })
            }
            "deduce_output" => {
                let recv = tok_str(strip_parens(&mc.receiver));
                // Aggregate-aware: builtin deduce args are tuples; lower their leaves
                // for real (the deduce itself skips below for non-mem receivers).
                let (_at, atok) = match mc.args.first() {
                    Some(e) => self.lower_aggregate(strip_parens(e)),
                    None => {
                        self.skip("expr", "missing argument".to_string());
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                };
                if Some(&recv) == self.addr_state.as_ref() {
                    self.emit_op(target, Ty::M31, quote! { eval.mem_addr_to_id(#atok) })
                } else if Some(&recv) == self.big_state.as_ref() {
                    self.emit_op(target, Ty::Felt252, quote! { eval.mem_id_to_value(#atok) })
                } else {
                    self.skip("deduce_output", format!("{recv}.deduce_output"));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            "packed_at" => {
                if is_path_named(&mc.receiver, "enabler_col") {
                    self.emit_op(target, Ty::M31, quote! { eval.enabler() })
                } else if self
                    .seq_idents
                    .iter()
                    .any(|s| is_path_named(&mc.receiver, s))
                    && mc
                        .args
                        .first()
                        .map(|a| is_path_named(a, &self.row_index_name))
                        .unwrap_or(false)
                {
                    // `seq.packed_at(row_index)` — the packed row index (Seq is the
                    // identity sequence). A REAL trait op now: the SIMD evaluator
                    // derives it from `row_index` bit-identically to `Seq::packed_at`;
                    // the recording lane reads the designated iota input slot (G4).
                    self.uses_iota = true;
                    self.emit_op(target, Ty::M31, quote! { eval.iota() })
                } else {
                    // preprocessed column .packed_at(row_index) etc.
                    self.skip(
                        "method",
                        format!("{}.packed_at (non-enabler)", tok_str(&mc.receiver)),
                    );
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            other => {
                // Recurse into args for census completeness, then skip.
                for a in &mc.args {
                    let _ = self.lower_aggregate(a);
                }
                self.skip(
                    "method",
                    format!(".{other}() on `{}`", tok_str(&mc.receiver)),
                );
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
                // Single array argument of EXACTLY 28 M31 exprs (the trait's
                // `felt_from_limbs` takes `[M31; FELT_N_LIMBS]`; a shorter source array
                // would zero-fill on the host — not expressible, loud skip).
                let arr = match call.args.first().map(strip_parens) {
                    Some(Expr::Array(ExprArray { elems, .. })) => elems,
                    _ => {
                        self.skip("call", "from_limbs without array arg".to_string());
                        return (Ty::Unknown, quote! { WG_SKIP });
                    }
                };
                if arr.len() != FELT252_LIMBS {
                    self.skip(
                        "call",
                        format!("PackedFelt252::from_limbs with {} != 28 limbs", arr.len()),
                    );
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let mut toks = Vec::new();
                for e in arr {
                    let (_t, k) = self.lower_node(strip_parens(e), Target::Temp);
                    toks.push(k);
                }
                self.emit_op(
                    target,
                    Ty::Felt252,
                    quote! { eval.felt_from_limbs([ #(#toks),* ]) },
                )
            }
            "PackedFelt252Width27 :: from_limbs" => {
                // 10 M31 limb exprs — the transformer itself holds the limbs, so the
                // W27 value is modeled as a `[E::M31; 10]` array binding (no trait op
                // needed). Same canonical-limb contract as `felt_from_limbs` (limbs
                // assumed < 2^27); the byte-equality gate is the arbiter.
                let arr = match call.args.first().map(strip_parens) {
                    Some(Expr::Array(ExprArray { elems, .. })) => elems,
                    _ => {
                        self.skip("call", "Width27 from_limbs without array arg".to_string());
                        return (Ty::Unknown, quote! { WG_SKIP });
                    }
                };
                if arr.len() != FELTW27_LIMBS {
                    self.skip(
                        "call",
                        format!(
                            "PackedFelt252Width27::from_limbs with {} != 10 limbs",
                            arr.len()
                        ),
                    );
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let mut toks = Vec::new();
                for e in arr {
                    let (t, k) = self.lower_node(strip_parens(e), Target::Temp);
                    if !t.is_m31() && t != Ty::Unknown {
                        self.skip("call", format!("Width27 from_limbs limb is {t:?}, not M31"));
                    }
                    toks.push(k);
                }
                let tok = self.bind(target, quote! { [ #(#toks),* ] });
                (Ty::FeltW27Limbs, tok)
            }
            "PackedFelt252Width27 :: from_packed_felt252" => {
                // f252 → w27 width conversion (G2): w27[j] = f9[3j] + f9[3j+1]*2^9 +
                // f9[3j+2]*2^18 — exact (27 = 3*9; each w27 limb < 2^27 < P, so plain
                // M31 mul-by-const + add). For j = 9 only f9[27] exists (252 = 9*27+9).
                // Bit-matches `Felt252Width27::from(Felt252)` (limb reinterpretation,
                // cpu.rs) for canonical 9-bit source limbs — the same canonicity
                // contract every `felt_get_m31` use already carries.
                let (at, atok) = self.lower_arg(call.args.first());
                match at {
                    Ty::Felt252 | Ty::ConstFelt252(_) => {
                        let mut limb_toks: Vec<TokenStream> = Vec::new();
                        for j in 0..FELTW27_LIMBS {
                            let mut acc: Option<TokenStream> = None;
                            for k in 0..3usize {
                                let idx = 3 * j + k;
                                if idx >= FELT252_LIMBS {
                                    break;
                                }
                                let il = usize_lit(idx);
                                let limb = self
                                    .bind(Target::Temp, quote! { eval.felt_get_m31(&#atok, #il) });
                                let term = if k == 0 {
                                    limb
                                } else {
                                    let c = 1u32 << (FELT252_LIMB_BITS * k);
                                    self.referenced_m31.insert(c);
                                    let cid = Ident::new(&format!("m31_{c}"), Span::call_site());
                                    self.bind(Target::Temp, quote! { eval.m31_mul(#limb, #cid) })
                                };
                                acc = Some(match acc {
                                    None => term,
                                    Some(a) => {
                                        self.bind(Target::Temp, quote! { eval.m31_add(#a, #term) })
                                    }
                                });
                            }
                            limb_toks.push(acc.expect("j*3 < 28 for all j < 10"));
                        }
                        let tok = self.bind(target, quote! { [ #(#limb_toks),* ] });
                        (Ty::FeltW27Limbs, tok)
                    }
                    other => {
                        self.skip(
                            "call",
                            format!(
                                "from_packed_felt252 on {:?} `{}`",
                                other,
                                call.args.first().map(tok_str).unwrap_or_default()
                            ),
                        );
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                }
            }
            "PackedFelt252 :: from_packed_felt252width27" => {
                // w27 → f252 width conversion: the REAL `felt_from_w27_words` trait op
                // (SIMD = the production conversion pair; recording = the exact 27->9
                // regroup on raw u32 ops). Opaque W27 (no known limb tokens) stays
                // census-only.
                let (at, atok) = self.lower_arg(call.args.first());
                match at {
                    Ty::FeltW27Limbs => {
                        self.emit_op(target, Ty::Felt252, quote! { eval.felt_from_w27_words(#atok) })
                    }
                    Ty::FeltW27 => self.w27_site(Ty::Felt252),
                    other => {
                        self.skip(
                            "call",
                            format!(
                                "from_packed_felt252width27 on {:?} `{}`",
                                other,
                                call.args.first().map(tok_str).unwrap_or_default()
                            ),
                        );
                        (Ty::Unknown, quote! { WG_SKIP })
                    }
                }
            }
            "PackedFelt252 :: from_m31" => {
                // Felt252 whose VALUE is the (31-bit) M31: limbs 0..3 are 9-bit windows
                // of the value (limbs 4..28 zero) — needs `U32Shr`/`U32And`; census-only
                // under the u32 trait extension, typed Felt252.
                let (at, _a) = self.lower_arg(call.args.first());
                if at.is_m31() || at == Ty::Unknown {
                    self.u32_site(Ty::Felt252)
                } else {
                    self.skip("call", format!("PackedFelt252::from_m31 on {at:?}"));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            "PackedUInt32 :: from_m31" => {
                let (at, a) = self.lower_arg(call.args.first());
                if at.is_m31() {
                    self.emit_op(target, Ty::U32, quote! { eval.u32_from_m31(#a) })
                } else {
                    self.skip("call", format!("PackedUInt32::from_m31 on {at:?}"));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            "PackedUInt32 :: from_limbs" => {
                // `low + (high << 16)` (simd.rs:204) — a REAL trait op when the
                // literal `[low, high]` shape is present (the generated idiom).
                if let Some(Expr::Array(ExprArray { elems, .. })) =
                    call.args.first().map(strip_parens)
                {
                    if elems.len() == 2 {
                        let (lt, ltok) = self.lower_node(strip_parens(&elems[0]), Target::Temp);
                        let (ht, htok) = self.lower_node(strip_parens(&elems[1]), Target::Temp);
                        self.require_m31(&lt, "u32_from_limbs low", &elems[0]);
                        self.require_m31(&ht, "u32_from_limbs high", &elems[1]);
                        return self.emit_op(
                            target,
                            Ty::U32,
                            quote! { eval.u32_from_limbs(#ltok, #htok) },
                        );
                    }
                    for e in elems {
                        let _ = self.lower_node(strip_parens(e), Target::Temp);
                    }
                }
                self.u32_site(Ty::U32)
            }
            p if p.ends_with(":: deduce_output") => {
                // HOOKED deduces: a REAL `WitnessEval` trait call (SIMD = the exact
                // fast_deduction function the original writer calls — byte-identical;
                // recording = an all-poison result + poison_ops census = the pinned
                // manifest). Result is the KNOWN tuple type, so projections compile
                // natively on both evaluators. Falls back to the census-only site if
                // the call's argument does not match the generated literal shape.
                if p == "PackedPartialEcMulWindowBits18 :: deduce_output" {
                    if let Some(tok) = self.lower_w18_deduce(call, target) {
                        return (known_deduce_output_ty(p).expect("W18 is in the table"), tok);
                    }
                    for a in &call.args {
                        let _ = self.lower_aggregate(a);
                    }
                    return self.deduce_site(known_deduce_output_ty(p).expect("in table"));
                }
                if p == "PackedPedersenPointsTableWindowBits18 :: deduce_output" {
                    if let Some(tok) = self.lower_points_table_deduce(call, target) {
                        return (
                            known_deduce_output_ty(p).expect("PT18 is in the table"),
                            tok,
                        );
                    }
                    for a in &call.args {
                        let _ = self.lower_aggregate(a);
                    }
                    return self.deduce_site(known_deduce_output_ty(p).expect("in table"));
                }
                if p == "PackedBlakeG :: deduce_output" {
                    if let Some(tok) = self.lower_blake_g_deduce(call, target) {
                        return (known_deduce_output_ty(p).expect("BlakeG in table"), tok);
                    }
                    for a in &call.args {
                        let _ = self.lower_aggregate(a);
                    }
                    return self.deduce_site(known_deduce_output_ty(p).expect("in table"));
                }
                if p == "PackedBlakeRoundSigma :: deduce_output" {
                    let (rt, rtok) = match call.args.first() {
                        Some(e) => self.lower_node(strip_parens(e), Target::Temp),
                        None => (Ty::Unknown, quote! { WG_SKIP }),
                    };
                    self.require_m31(&rt, "sigma deduce round", &call.args[0]);
                    let tok = self.bind(target, quote! { eval.deduce_blake_round_sigma(#rtok) });
                    return (known_deduce_output_ty(p).expect("Sigma in table"), tok);
                }
                // Census-only / unknown deduces: lower the args for REAL first
                // (tuple/array shapes route through lower_aggregate, so their
                // M31/felt leaves record cleanly).
                for a in &call.args {
                    let _ = self.lower_aggregate(a);
                }
                // Known-signature deduce (G5): type the RESULT so downstream
                // projections resolve; the call stays census-only (deduce_sites).
                // The result type is transcribed from the host fast_deduction
                // signature — a WRONG shape here would silently mis-type everything
                // downstream, so entries are added only with the signature in view.
                if let Some(ty) = known_deduce_output_ty(p) {
                    return self.deduce_site(ty);
                }
                // Unknown-signature deduce: the honest skip — the quantified
                // EC/poseidon/blake deduce backlog.
                self.skip("deduce_output", p.to_string());
                (Ty::Unknown, quote! { WG_SKIP })
            }
            other => {
                for a in &call.args {
                    let _ = self.lower_aggregate(a);
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
                if let Some(k) = self.peek_const_u32(&b.right) {
                    if lt.is_u32() {
                        let l = self.u32ish_value(lt, ltok);
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U32, quote! { eval.u32_shl_imm(#l, #kl) });
                    }
                    self.skip("binop", format!("`<<` (u32) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let (rt, _rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip(
                    "binop",
                    format!("`<<` by non-const `{}`", tok_str(&b.right)),
                );
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
                if let Some(k) = self.peek_const_u32(&b.right) {
                    if lt.is_u32() {
                        let l = self.u32ish_value(lt, ltok);
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U32, quote! { eval.u32_shr_imm(#l, #kl) });
                    }
                    self.skip("binop", format!("`>>` (u32) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                let (rt, _rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_u32() && rt.is_u32() {
                    return self.u32_site(Ty::U32);
                }
                self.skip(
                    "binop",
                    format!("`>>` by non-const `{}`", tok_str(&b.right)),
                );
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
                    self.skip(
                        "binop",
                        format!("`&` (mask) on non-U16 `{}`", tok_str(&b.left)),
                    );
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if let Some(k) = self.peek_const_u16(&b.left) {
                    let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                    if rt.is_u16() {
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U16, quote! { eval.u16_and(#rtok, #kl) });
                    }
                    self.skip(
                        "binop",
                        format!("`&` (mask) on non-U16 `{}`", tok_str(&b.right)),
                    );
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if let Some(k) = self.peek_const_u32(&b.right) {
                    let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                    if lt.is_u32() {
                        let l = self.u32ish_value(lt, ltok);
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U32, quote! { eval.u32_and_imm(#l, #kl) });
                    }
                    self.skip("binop", format!("`&` (u32 mask) on {:?}", lt));
                    return (Ty::Unknown, quote! { WG_SKIP });
                }
                if let Some(k) = self.peek_const_u32(&b.left) {
                    let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                    if rt.is_u32() {
                        let r = self.u32ish_value(rt, rtok);
                        let kl = u32_lit(k);
                        return self.emit_op(target, Ty::U32, quote! { eval.u32_and_imm(#r, #kl) });
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
                    let l = self.u32ish_value(lt, ltok);
                    let r = self.u32ish_value(rt, rtok);
                    self.emit_op(target, Ty::U32, quote! { eval.u32_add(#l, #r) })
                } else if lt.is_feltish() && rt.is_feltish() {
                    let l = self.feltish_value(lt, ltok);
                    let r = self.feltish_value(rt, rtok);
                    self.emit_op(
                        target,
                        Ty::Felt252,
                        quote! { eval.felt_add(#l.clone(), #r.clone()) },
                    )
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
                    let l = self.u32ish_value(lt, ltok);
                    let r = self.u32ish_value(rt, rtok);
                    self.emit_op(target, Ty::U32, quote! { eval.u32_sub(#l, #r) })
                } else if lt.is_feltish() && rt.is_feltish() {
                    let l = self.feltish_value(lt, ltok);
                    let r = self.feltish_value(rt, rtok);
                    self.emit_op(
                        target,
                        Ty::Felt252,
                        quote! { eval.felt_sub(#l.clone(), #r.clone()) },
                    )
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
                } else if lt.is_feltish() && rt.is_feltish() {
                    let l = self.feltish_value(lt, ltok);
                    let r = self.feltish_value(rt, rtok);
                    self.emit_op(
                        target,
                        Ty::Felt252,
                        quote! { eval.felt_mul(#l.clone(), #r.clone()) },
                    )
                } else {
                    self.skip("binop", format!("`*` on {:?}/{:?}", lt, rt));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            BinOp::Div(_) => {
                // ONLY felt division exists in the writers (EC slope denominators;
                // the host `Felt252::div` panics on zero — see DeduceKind::FeltDiv).
                let (lt, ltok) = self.lower_node(&b.left, Target::Temp);
                let (rt, rtok) = self.lower_node(&b.right, Target::Temp);
                if lt.is_feltish() && rt.is_feltish() {
                    let l = self.feltish_value(lt, ltok);
                    let r = self.feltish_value(rt, rtok);
                    self.emit_op(
                        target,
                        Ty::Felt252,
                        quote! { eval.felt_div(#l.clone(), #r.clone()) },
                    )
                } else {
                    self.skip("binop", format!("`/` on {:?}/{:?}", lt, rt));
                    (Ty::Unknown, quote! { WG_SKIP })
                }
            }
            other => {
                let _ = self.lower_node(&b.left, Target::Temp);
                let _ = self.lower_node(&b.right, Target::Temp);
                self.skip(
                    "binop",
                    format!("unsupported binary op `{}`", tok_str_op(&other)),
                );
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
    fn flatten_sub(&mut self, expr: &Expr) -> Vec<SubLeaf> {
        let m31 = |tok: TokenStream| SubLeaf { tok, u32: false };
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
                match ty {
                    // Full-32-bit sub element (blake words): one raw word, stored via
                    // the u32 effect (the flat transport is raw lanes).
                    Ty::U32 => vec![SubLeaf { tok, u32: true }],
                    // Felt-valued sub element: 28 flat limb words (the canonical
                    // decomposition; the driver's `from_limbs` reconstruction is the
                    // exact inverse, so the receiver sees the identical felt).
                    Ty::Felt252 => {
                        let felt = self.bind(Target::Temp, quote! { #tok });
                        (0..FELT252_LIMBS)
                            .map(|j| {
                                let jl = usize_lit(j);
                                m31(self
                                    .bind(Target::Temp, quote! { eval.felt_get_m31(&#felt, #jl) }))
                            })
                            .collect()
                    }
                    Ty::ConstFelt252(limbs) => (0..FELT252_LIMBS)
                        .map(|j| {
                            let (_t, tok) = self.const_m31_leaf(Target::Temp, limbs[j]);
                            m31(tok)
                        })
                        .collect(),
                    _ => {
                        self.require_m31(&ty, "sub-input word", other);
                        vec![m31(tok)]
                    }
                }
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
    seg.push(format!(
        "pub(crate) const N_LOOKUP_WORDS: usize = {};",
        fa.n_lookup_words
    ));
    seg.push(format!(
        "pub(crate) const N_SUB_INPUT_WORDS: usize = {};",
        fa.n_sub_words
    ));
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
    let record_ctor: TokenStream = if matches!(lw.input_ty, Ty::Unknown) {
        quote! { RecordingWitnessEval::new(#component) }
    } else {
        // Builtin slot layout: flat input words 0..K, enabler = K, iota = K+1 (when
        // the body reads it). The device lane MUST feed its input columns in this
        // exact order.
        // Slot layout (uniform for every builtin): flat inputs 0..K, enabler K,
        // iota K+1 (reserved even when unused — no Input op records for it then),
        // multiplicity columns K+2+k. The device lane MUST feed its input columns in
        // this exact order.
        let k = u32_lit(lw.input_ty.flat_width() as u32);
        let ki = u32_lit(lw.input_ty.flat_width() as u32 + 1);
        quote! { RecordingWitnessEval::with_slots(#component, #k, Some(#ki)) }
    };
    seg.push(render(&quote! {
        #[allow(dead_code)]
        pub(crate) fn #record_fn() -> RecordingOutput {
            let mut eval = #record_ctor;
            #row_body_fn(&mut eval);
            eval.finish()
        }
    }));
    seg.push(String::new());

    // 5. Lookup-flat accessors (the witness-JIT prove / device-interaction
    // seam): JIT_LOOKUP_FIELDS + interaction_gen_from_flat_lookup_words, field
    // list in LookupData declaration order; ctor variant per the module's
    // InteractionClaimGenerator shape.
    {
        let mut inv = String::new();
        inv.push_str("crate::jit_lookup_accessor! {\n");
        if igen_has_n_rows(file) {
            inv.push_str(&format!("    with_n_rows {};\n", fa.n_lookup_words));
        } else {
            inv.push_str(&format!("    {};\n", fa.n_lookup_words));
        }
        for f in &lw.lookup_fields {
            if f.scalar {
                inv.push_str(&format!("    {}: scalar,\n", f.name));
            } else {
                inv.push_str(&format!("    {}: {},\n", f.name, f.width));
            }
        }
        inv.push('}');
        seg.push(inv);
        seg.push(String::new());
    }

    // 6. Test-only surface: flats + GenericSimdDiff + generic_simd_diff.
    seg.push(
        "// ---- Test-only surface for the byte-equality gate ---------------------------------"
            .to_string(),
    );
    seg.push(String::new());
    seg.push(render(&lookup_flat_tokens(lw)));
    seg.push(String::new());
    seg.push(
        "#[cfg(test)]\npub(crate) fn test_lookup_data_flat(ig: &InteractionClaimGenerator)          -> Vec<Vec<PackedM31>> {\n    lookup_data_flat(&ig.lookup_data)\n}"
            .to_string(),
    );
    seg.push(String::new());
    seg.push(render(&sub_flat_tokens(lw)));
    seg.push(String::new());
    seg.push(
        "/// Byte-comparison bundle (only public types cross the module boundary).".to_string(),
    );
    seg.push(render(&generic_simd_diff_struct_tokens()));
    seg.push(String::new());
    seg.push(
        "/// Run BOTH SIMD writers on the same (pure-read) states and return public compare data."
            .to_string(),
    );
    seg.push(render(&generic_simd_diff_fn_tokens(writer, &file)));
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

    // Writers without a mem-state param pass `None` (`impl Into<Option<..>>` on the
    // eval constructor keeps the opcode emitted text unchanged for present states).
    let addr_id: TokenStream = match &lw.addr_state {
        Some(name) => {
            let id = Ident::new(name, Span::call_site());
            quote! { #id }
        }
        None => quote! { None },
    };
    let big_id: TokenStream = match &lw.big_state {
        Some(name) => {
            let id = Ident::new(name, Span::call_site());
            quote! { #id }
        }
        None => quote! { None },
    };
    let input_id = Ident::new(&lw.input_name, Span::call_site());
    // Opcode writers pass their `PackedCasmState` binder straight through
    // (`impl Into<SimdInputs>` — the emitted text is unchanged from the pre-builtin
    // lane). Builtin writers flatten their typed input tuple into the flat input
    // words IN SLOT ORDER — the exact depth-first order the transformer's
    // `Ty::InputAt` slot map assigned, so `eval.input(k)` reads word k.
    let eval_input: TokenStream = if matches!(lw.input_ty, Ty::Unknown) {
        quote! { #input_id }
    } else {
        // Builtin flat input words, IN SLOT ORDER: [flattened inputs (0..K), a zero
        // placeholder at the enabler slot (K) and the iota slot (K+1) — `enabler()` /
        // `iota()` never route through `input()` on the SIMD side, but keeping the
        // positions makes the slot arithmetic identical to the recording/device
        // layout — then the multiplicity columns (K+2+k).]
        let mut words = input_flatten_tokens(&lw.input_ty, quote! { #input_id });
        if !lw.mults_reads.is_empty() {
            words.push(quote! { Simd::splat(0) });
            words.push(quote! { Simd::splat(0) });
            let max_k = *lw.mults_reads.iter().max().unwrap();
            for k in 0..=max_k {
                if lw.mults_reads.contains(&k) {
                    let kl = usize_lit(k);
                    words.push(quote! {
                        mults[#kl]
                            .get(row_index)
                            .copied()
                            .unwrap_or(PackedM31::zero())
                            .into_simd()
                    });
                } else {
                    words.push(quote! { Simd::splat(0) });
                }
            }
        }
        quote! { vec![ #(#words),* ] }
    };
    let row_id = Ident::new(&lw.row_name, Span::call_site());
    let lookup_id = Ident::new(&lw.lookup_name, Span::call_site());
    let sub_id = Ident::new(&lw.sub_name, Span::call_site());

    let reconstruct_lookup = reconstruct_lookup(lw, &lookup_id);
    let reconstruct_sub = reconstruct_sub(lw, &sub_id);

    // Writers that never multiply by the enabler (padding handled via their mults
    // column, e.g. pedersen_aggregator) have no `enabler_col` in the preamble; the
    // eval constructor still takes one. `Enabler::new` is pure, so materializing an
    // unused one is semantics-free.
    let has_enabler = preamble
        .iter()
        .any(|t| t.to_string().contains("let enabler_col"));
    let enabler_fallback: TokenStream = if has_enabler {
        quote! {}
    } else {
        // A body can only reach `eval.enabler()` through the `enabler_col.packed_at`
        // idiom, which requires the preamble binding — so when it is absent the value
        // is genuinely unused and the arity-0 construction is semantics-free.
        quote! { let enabler_col = Enabler::new(0); }
    };

    quote! {
        #[allow(clippy::type_complexity)]
        #[allow(unused_variables)]
        #[allow(dead_code)]
        fn write_trace_generic_simd(#inputs) #output {
            #(#preamble)*
            #enabler_fallback

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
                        #eval_input,
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

/// Flatten a typed builtin input value into its flat input words, IN SLOT ORDER
/// (depth-first over the `PackedInputType` tree — the same order [`Ty::InputAt`]
/// assigns slot bases, so `eval.input(k)` reads exactly word k). Only M31 leaves are
/// emitted; a file with felt/u16/u32 input leaves has `input_sites > 0` and is never
/// emitted, so this is unreachable for those (the unreachable!() is the guard).
fn input_flatten_tokens(ty: &Ty, base: TokenStream) -> Vec<TokenStream> {
    match ty {
        Ty::M31 => vec![quote! { #base.into_simd() }],
        Ty::U32 => vec![quote! { #base.simd }],
        Ty::Felt252 => (0..FELT252_LIMBS)
            .map(|j| {
                let lit = usize_lit(j);
                quote! { #base.get_m31(#lit).into_simd() }
            })
            .collect(),
        // W27 input leaf: 10 word columns (27-bit values, M31-safe raw words).
        Ty::FeltW27 => (0..FELTW27_LIMBS)
            .map(|j| {
                let lit = usize_lit(j);
                quote! { #base.get_m31(#lit).into_simd() }
            })
            .collect(),
        Ty::Tuple(v) => {
            let mut out = Vec::new();
            for (i, e) in v.iter().enumerate() {
                let lit = Literal::usize_unsuffixed(i);
                out.extend(input_flatten_tokens(e, quote! { #base.#lit }));
            }
            out
        }
        Ty::Array(e, n) => {
            let mut out = Vec::new();
            for j in 0..*n {
                let lit = usize_lit(j);
                out.extend(input_flatten_tokens(e, quote! { #base[#lit] }));
            }
            out
        }
        other => unreachable!(
            "input_flatten_tokens on non-emittable input leaf {other:?} (input_sites gate)"
        ),
    }
}

fn rebuild_shape(shape: &Shape, idx: &mut usize) -> TokenStream {
    match shape {
        Shape::Scalar => {
            let i = usize_lit(*idx);
            *idx += 1;
            // Raw lane -> canonical M31 (the store side wrote a canonical value).
            quote! { unsafe { PackedM31::from_simd_unchecked(sw[#i]) } }
        }
        Shape::U32 => {
            let i = usize_lit(*idx);
            *idx += 1;
            quote! { PackedUInt32::from_simd(sw[#i]) }
        }
        Shape::Felt => {
            // 28 consecutive limb words -> the felt value (exact inverse of the
            // canonical `felt_get_m31` decomposition the flatten side emitted).
            let limbs: Vec<TokenStream> = (0..FELT252_LIMBS)
                .map(|_| {
                    let i = usize_lit(*idx);
                    *idx += 1;
                    quote! { unsafe { PackedM31::from_simd_unchecked(sw[#i]) } }
                })
                .collect();
            quote! { PackedFelt252::from_limbs([ #(#limbs),* ]) }
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
            Shape::Scalar => parts.push(quote! {
                sci.#field[#k].iter().map(|v| v.into_simd()).collect::<Vec<_>>()
            }),
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
        fn sub_inputs_flat(sci: &SubComponentInputs) -> Vec<Vec<Simd<u32, N_LANES>>> {
            vec![ #(#parts),* ]
        }
    }
}

/// Scalar projections of a shaped `PackedInputType` value. `base` navigates from an
/// `&PackedInputType` via `.N` / `[j]`; each leaf is a Copy `PackedM31` via auto-deref.
fn shape_projection(shape: &Shape, base: TokenStream) -> Vec<TokenStream> {
    match shape {
        Shape::Scalar => vec![quote! { #base.into_simd() }],
        Shape::U32 => vec![quote! { #base.simd }],
        Shape::Felt => (0..FELT252_LIMBS)
            .map(|j| {
                let lit = usize_lit(j);
                quote! { #base.get_m31(#lit).into_simd() }
            })
            .collect(),
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
            pub orig_sub: Vec<Vec<Simd<u32, N_LANES>>>,
            pub gen_sub: Vec<Vec<Simd<u32, N_LANES>>>,
            pub orig_interaction_cols: Vec<Vec<M31>>,
            pub gen_interaction_cols: Vec<Vec<M31>>,
            pub orig_claimed_sum: SecureField,
            pub gen_claimed_sum: SecureField,
        }
    }
}

/// `generic_simd_diff(...)`: same params as `write_trace_simd`; runs both writers and
/// packages the compare bundle (verbatim body from the shape-spec).
fn generic_simd_diff_fn_tokens(writer: &ItemFn, file: &syn::File) -> TokenStream {
    // Some components' `InteractionClaimGenerator` carries an extra `n_rows` field
    // (e.g. blake_round); include it in the literal when declared (`n_rows` is a
    // writer param, in scope in the harness).
    let ig_has_n_rows = file.items.iter().any(|it| match it {
        Item::Struct(st) if st.ident == "InteractionClaimGenerator" => match &st.fields {
            Fields::Named(n) => n
                .named
                .iter()
                .any(|f| f.ident.as_ref().is_some_and(|i| i == "n_rows")),
            _ => false,
        },
        _ => false,
    });
    let ig_extra: TokenStream = if ig_has_n_rows {
        quote! { n_rows, }
    } else {
        quote! {}
    };
    let inputs = &writer.sig.inputs;
    // Argument names in order; the first must be `inputs`. BY-VALUE params (no `&` in
    // the type — e.g. the aggregator's `mults: Vec<Vec<PackedM31>>`) are cloned into
    // the FIRST call so the second still owns them; references pass through twice.
    let mut names: Vec<Ident> = Vec::new();
    let mut by_value: Vec<bool> = Vec::new();
    for arg in inputs {
        if let FnArg::Typed(pt) = arg {
            if let Pat::Ident(pi) = &*pt.pat {
                names.push(pi.ident.clone());
                by_value.push(!matches!(&*pt.ty, Type::Reference(_)));
            }
        }
    }
    let rest = &names[1..];
    let rest_first: Vec<TokenStream> = names[1..]
        .iter()
        .zip(&by_value[1..])
        .map(|(n, bv)| {
            if *bv {
                quote! { #n.clone() }
            } else {
                quote! { #n }
            }
        })
        .collect();
    quote! {
        #[cfg(test)]
        pub(crate) fn generic_simd_diff(#inputs) -> GenericSimdDiff {
            let (trace_o, ld_o, sci_o) = write_trace_simd(inputs.clone(), #(#rest_first),*);
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
                #ig_extra
                lookup_data: ld_o,
            }
            .write_interaction_trace(&common);
            let (raw_g, _) = InteractionClaimGenerator {
                log_size,
                #ig_extra
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

/// If `local` is `let IDENT = PackedFelt252::broadcast(Felt252::from([A, B, C, D]));`
/// (the hoisted felt-constant idiom, G3 — 4 LITTLE-ENDIAN u64 words), return the words.
/// A `PackedFelt252Width27::broadcast` would decompose to 10 x 27-bit limbs instead, but
/// no such method exists on the packed type today, so only the Felt252 form is parsed.
fn local_felt_const(local: &Local) -> Option<[u64; 4]> {
    let init = local.init.as_ref()?;
    let call = match strip_parens(&init.expr) {
        Expr::Call(c) => c,
        _ => return None,
    };
    let path = match &*call.func {
        Expr::Path(p) => tok_str(&p.path),
        _ => return None,
    };
    if path != "PackedFelt252 :: broadcast" {
        return None;
    }
    // arg: Felt252::from([A, B, C, D])
    let inner = match call.args.first().map(strip_parens) {
        Some(Expr::Call(c)) => c,
        _ => return None,
    };
    let inner_path = match &*inner.func {
        Expr::Path(p) => tok_str(&p.path),
        _ => return None,
    };
    if inner_path != "Felt252 :: from" {
        return None;
    }
    let arr = match inner.args.first().map(strip_parens) {
        Some(Expr::Array(ExprArray { elems, .. })) if elems.len() == 4 => elems,
        _ => return None,
    };
    let mut words = [0u64; 4];
    for (i, e) in arr.iter().enumerate() {
        match strip_parens(e) {
            Expr::Lit(el) => match &el.lit {
                Lit::Int(l) => words[i] = l.base10_parse::<u64>().ok()?,
                _ => return None,
            },
            _ => return None,
        }
    }
    Some(words)
}

/// Decompose the 4 little-endian u64 words of a `Felt252::from([u64; 4])` into the 28
/// canonical 9-bit limbs — bit-identical to `Felt252::from([u64;4])` (which masks the
/// top word to 60 bits, keeping the low 252 bits) followed by `Felt252::get_m31(i)`
/// (9-bit window at bit `9*i`; common `prover_types/cpu.rs`).
fn felt252_const_limbs(words: [u64; 4]) -> [u32; FELT252_LIMBS] {
    let mut limbs = words;
    limbs[3] &= 0x0fff_ffff_ffff_ffff; // From<[u64;4]> masks to 252 bits.
    std::array::from_fn(|i| {
        let mask = (1u64 << FELT252_LIMB_BITS) - 1;
        let shift = FELT252_LIMB_BITS * i;
        let low = shift / 64;
        let shift_low = shift & 0x3F;
        let high = (shift + FELT252_LIMB_BITS - 1) / 64;
        let v = if low == high {
            (limbs[low] >> shift_low) & mask
        } else {
            ((limbs[low] >> shift_low) | (limbs[high] << (64 - shift_low))) & mask
        };
        v as u32
    })
}

/// RESULT types of the KNOWN `PackedX::deduce_output` signatures (G5), transcribed from
/// the host `witness/fast_deduction/{pedersen,ec_op,blake}.rs` with the signatures in
/// view — a WRONG shape here would silently mis-type everything downstream of a deduce,
/// so entries are never guessed:
///   * `PackedPartialEcMul<N>::deduce_output((M31, M31, ([M31; N], [Felt252; 2])))` returns the
///     same tuple shape (pedersen.rs; WindowBits18 => N=14, WindowBits9 => N=28).
///   * `PackedPartialEcMulGeneric::deduce_output` returns `Box<(M31, M31, State)>` with `State =
///     (Felt252Width27, [Felt252; 2], [Felt252; 2], M31)` (ec_op.rs) — typed as the inner tuple;
///     source projections auto-deref through the `Box`.
///   * points tables: `([M31; 1]) -> [Felt252; 2]` (pedersen.rs).
///   * `PackedBlakeG: ([U32; 6]) -> [U32; 4]`; `PackedBlakeRoundSigma: (M31) -> [M31; 16]`
///     (blake.rs; `N_BLAKE_SIGMA_COLS = 16`).
fn known_deduce_output_ty(path: &str) -> Option<Ty> {
    let felt2 = || Ty::Array(Box::new(Ty::Felt252), 2);
    match path {
        "PackedPartialEcMulWindowBits18 :: deduce_output" => Some(Ty::Tuple(vec![
            Ty::M31,
            Ty::M31,
            Ty::Tuple(vec![Ty::Array(Box::new(Ty::M31), 14), felt2()]),
        ])),
        "PackedPartialEcMulWindowBits9 :: deduce_output" => Some(Ty::Tuple(vec![
            Ty::M31,
            Ty::M31,
            Ty::Tuple(vec![Ty::Array(Box::new(Ty::M31), 28), felt2()]),
        ])),
        "PackedPartialEcMulGeneric :: deduce_output" => Some(Ty::Tuple(vec![
            Ty::M31,
            Ty::M31,
            Ty::Tuple(vec![Ty::FeltW27, felt2(), felt2(), Ty::M31]),
        ])),
        "PackedPedersenPointsTableWindowBits18 :: deduce_output"
        | "PackedPedersenPointsTableWindowBits9 :: deduce_output" => Some(felt2()),
        "PackedBlakeG :: deduce_output" => Some(Ty::Array(Box::new(Ty::U32), 4)),
        "PackedBlakeRoundSigma :: deduce_output" => Some(Ty::Array(Box::new(Ty::M31), 16)),
        _ => None,
    }
}

/// If `local` is `let IDENT = Seq::new(...);` (the preamble row-index sequence), return
/// its name. Inside the closure, `IDENT.packed_at(row_index)` IS the packed row index
/// (an iota) — typed M31 and census-only until the builtin lane feeds it as an input
/// word (G4).
fn local_seq_ident(local: &Local) -> Option<String> {
    let name = local_ident(local)?;
    let init = local.init.as_ref()?;
    let call = match strip_parens(&init.expr) {
        Expr::Call(c) => c,
        _ => return None,
    };
    match &*call.func {
        Expr::Path(p) if tok_str(&p.path) == "Seq :: new" => Some(name),
        _ => None,
    }
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
        let prev_line_start = src[..line_start - 1]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
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
    let matched: Vec<&(PathBuf, FileAnalysis)> =
        analyses.iter().filter(|(_, a)| a.matched).collect();
    let matched_u32: Vec<&(PathBuf, FileAnalysis)> =
        analyses.iter().filter(|(_, a)| a.matched_u32).collect();

    println!("======================================================================");
    println!("witness_genericize CENSUS");
    println!("======================================================================");
    println!("Files scanned:                          {n}");
    println!("  with write_trace_simd:                {writers}");
    println!("  MATCHED (rewritable):                 {}", matched.len());
    println!(
        "  MATCHED (needs trait ext: u32/input):  {}",
        matched_u32.len()
    );
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

    println!(
        "--- MATCHED files (needs trait extension: u32/input/w27/deduce; census-only, NOT emitted) ---"
    );
    for (_p, a) in &matched_u32 {
        println!(
            "  {:<34} cols={:<4} lookup_words={:<4} sub_words={:<4} u32_sites={:<4} \
             input_sites={:<4} w27_sites={:<4} deduce_sites={}",
            a.component,
            a.n_cols,
            a.n_lookup_words,
            a.n_sub_words,
            a.u32_sites,
            a.input_sites,
            a.w27_sites,
            a.deduce_sites
        );
    }
    println!();

    println!("--- SKIPPED files (loud reasons) ---");
    for (_p, a) in analyses
        .iter()
        .filter(|(_, a)| !a.matched && !a.matched_u32)
    {
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
                census_site_suffix(a)
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
    println!(
        "--- unmatched-construct census (grouped by kind across files, excl. deduce_output) ---"
    );
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
        println!(
            "  [{cat}] {detail}  — {count} sites in {} files",
            fileset.len()
        );
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

/// " + N u32 sites + M input sites + ..." suffix for census rows (census-only sites that
/// are typed but not emittable).
fn census_site_suffix(a: &FileAnalysis) -> String {
    let mut parts = Vec::new();
    for (n, what) in [
        (a.u32_sites, "u32"),
        (a.input_sites, "input"),
        (a.w27_sites, "w27"),
        (a.deduce_sites, "deduce"),
    ] {
        if n > 0 {
            parts.push(format!(" + {n} {what} sites"));
        }
    }
    parts.concat()
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
            "matched via census-only rules ({} u32 / {} input / {} w27 / {} deduce sites) — \
             needs trait/lane extension; not emitted",
            a.u32_sites, a.input_sites, a.w27_sites, a.deduce_sites
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
            eprintln!(
                "SKIP {}: no `struct LookupData` anchor for insert",
                a.component
            );
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

/// Comparison view of a block for `--check`: comment-only lines are dropped
/// (rustfmt's `wrap_comments` reflows generated prose at `comment_width`, and for
/// long component names the reflow differs from the emitted wrapping — pure
/// noise). Code lines compare EXACTLY; the fence's teeth are unchanged.
fn check_view(block: &str) -> String {
    block
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
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
                if check_view(&on_disk) != check_view(block) {
                    drift += 1;
                    eprintln!(
                        "DRIFT {}: on-disk block differs from generated",
                        a.component
                    );
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

    fn lower_snippet_full(
        consts: &[(&str, ConstKind, u32)],
        felt_consts: BTreeMap<String, [u32; FELT252_LIMBS]>,
        input_ty: Ty,
        body: &str,
        sub_slots: Vec<SubSlot>,
    ) -> Lowerer {
        let mut cmap = BTreeMap::new();
        for (n, k, v) in consts {
            cmap.insert(
                n.to_string(),
                ConstVal {
                    kind: *k,
                    value: *v,
                },
            );
        }
        let mut lw = Lowerer::new(
            cmap,
            felt_consts,
            ["seq".to_string()].into_iter().collect(),
            Some("memory_address_to_id_state".to_string()),
            Some("memory_id_to_big_state".to_string()),
            "add_opcode_input".to_string(),
            input_ty,
            "row_index".to_string(),
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

    fn lower_snippet_with_slots(
        consts: &[(&str, ConstKind, u32)],
        body: &str,
        sub_slots: Vec<SubSlot>,
    ) -> Lowerer {
        lower_snippet_full(consts, BTreeMap::new(), Ty::Unknown, body, sub_slots)
    }

    fn lower_snippet(consts: &[(&str, ConstKind, u32)], body: &str) -> Lowerer {
        lower_snippet_with_slots(consts, body, vec![])
    }

    #[test]
    fn infer_input_fields() {
        let lw = lower_snippet(
            &[],
            "let a = add_opcode_input.pc; let b = add_opcode_input.fp;",
        );
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
        assert_eq!(lw.env["f"], Ty::Felt252);
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
        // A deduce whose signature is NOT in `known_deduce_output_ty` must stay a loud
        // skip (the fictional name keeps this test valid as the table grows).
        let lw = lower_snippet(
            &[],
            "let x = PackedNotInTheTable::deduce_output(add_opcode_input.pc);",
        );
        assert!(!lw.skips.is_empty());
        assert!(lw.skips.iter().any(|s| s.category == "deduce_output"));
        assert_eq!(lw.deduce_sites, 0);
    }

    #[test]
    fn known_deduce_types_result_and_counts_site() {
        // G5: a KNOWN-signature deduce types its result (here (M31, M31, ([M31;14],
        // [Felt252;2]))) so downstream projections resolve — .0 is M31 (usable in real
        // M31 ops), .2.1[0].get_m31(3) is a felt limb — with NO skips; the call itself
        // is census-only via `deduce_sites`, which blocks emission.
        let p = "add_opcode_input.pc";
        let windows = vec![p; 14].join(", ");
        let body = format!(
            "let f = memory_id_to_big_state.deduce_output(\
                 memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let out = PackedPartialEcMulWindowBits18::deduce_output((add_opcode_input.pc, \
             add_opcode_input.ap, ([{windows}], [f, f]))); \
             let chain = out.0; \
             let win0 = out.2.0[0]; \
             let acc_limb = out.2.1[0].get_m31(3); \
             let s = ((chain) + (win0)) + ((acc_limb) + (chain)); \
             let pt = PackedPedersenPointsTableWindowBits18::deduce_output([add_opcode_input.fp]); \
             let ptl = pt[1].get_m31(27);",
        );
        let lw = lower_snippet(&[], &body);
        // W18 + points-table are HOOKED: real trait calls, zero census-only sites.
        assert_eq!(
            lw.deduce_sites, 0,
            "hooked deduces are real ops; skips: {:?}",
            lw.skips
        );
        assert!(
            lw.skips.is_empty(),
            "projections off a typed deduce result must not skip: {:?}",
            lw.skips
        );
        let body = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(
            body.contains("eval . deduce_partial_ec_mul_w18 ("),
            "body: {body}"
        );
        assert!(
            body.contains("eval . deduce_pedersen_points_table_w18 ("),
            "body: {body}"
        );
    }

    /// An UNHOOKED known-signature deduce (W9) stays census-only: result typed (so
    /// projections resolve skip-free) but `deduce_sites` counts it and blocks emission.
    #[test]
    fn unhooked_known_deduce_is_census_only() {
        let p = "add_opcode_input.pc";
        let windows = vec![p; 28].join(", ");
        let body = format!(
            "let f = memory_id_to_big_state.deduce_output(\
                 memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let out = PackedPartialEcMulWindowBits9::deduce_output((add_opcode_input.pc, \
             add_opcode_input.ap, ([{windows}], [f, f]))); \
             let chain = out.0;",
        );
        let lw = lower_snippet(&[], &body);
        assert_eq!(lw.deduce_sites, 1, "skips: {:?}", lw.skips);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
    }

    #[test]
    fn u32_family_lowers_to_real_trait_ops() {
        let lw = lower_snippet(
            &[
                ("UInt32_511", ConstKind::U32, 511),
                ("UInt32_9", ConstKind::U32, 9),
            ],
            "let a = add_opcode_input.ap; \
             let x = PackedUInt32::from_m31(a); \
             let y = ((x) & (UInt32_511)) + ((x) << (UInt32_9)); \
             let z = y.low().as_m31(); \
             let w = y.high().as_m31();",
        );
        assert!(
            lw.skips.is_empty(),
            "u32 family must not skip: {:?}",
            lw.skips
        );
        // The whole family is REAL now (fp256-cohort u32 extension): from_m31,
        // masked and, const shl, wrapping add, low/high — zero census sites.
        assert_eq!(lw.u32_sites, 0, "u32 census sites");
        let body = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(body.contains("eval . u32_from_m31 ("), "body: {body}");
        assert!(body.contains("eval . u32_and_imm ("), "body: {body}");
        assert!(body.contains("eval . u32_shl_imm ("), "body: {body}");
        assert!(body.contains("eval . u32_add ("), "body: {body}");
        assert!(body.contains("eval . u32_low ("), "body: {body}");
        assert!(body.contains("eval . u32_high ("), "body: {body}");
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
                 verify_instruction: [Vec<(PackedM31, [PackedM31; 3], [PackedM31; 2], \
                 PackedM31)>; 1],\n\
                 memory_address_to_id: [Vec<PackedM31>; 2],\n\
                 memory_id_to_big: [Vec<PackedM31>; 2],\n\
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
        let slots = build_sub_layout(&file, &body.stmts, "sub_component_inputs", None).unwrap();
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
        assert_eq!(
            slots.iter().map(|s| s.shape.scalar_count()).sum::<usize>(),
            11
        );
    }

    #[test]
    fn sub_layout_missing_assignment_is_loud() {
        let file: syn::File = syn::parse_str(
            "struct SubComponentInputs { memory_address_to_id: [Vec<PackedM31>; 2] }",
        )
        .unwrap();
        let body: syn::Block =
            syn::parse_str("{ *sub_component_inputs.memory_address_to_id[0] = x0; }").unwrap();
        let err = build_sub_layout(&file, &body.stmts, "sub_component_inputs", None).unwrap_err();
        assert!(err.detail.contains("never assigned"), "{}", err.detail);
    }

    #[test]
    fn sub_words_use_declaration_order_bases() {
        let file: syn::File = syn::parse_str(
            "struct SubComponentInputs {\n\
                 memory_address_to_id: [Vec<PackedM31>; 1],\n\
                 memory_id_to_big: [Vec<PackedM31>; 1],\n\
             }",
        )
        .unwrap();
        // FILE order writes memory_id_to_big FIRST; its flat base must still be 1.
        let body_src = "*sub_component_inputs.memory_id_to_big[0] = add_opcode_input.pc;\n\
                        *sub_component_inputs.memory_address_to_id[0] = add_opcode_input.ap;";
        let block: syn::Block = syn::parse_str(&format!("{{ {body_src} }}")).unwrap();
        let slots = build_sub_layout(&file, &block.stmts, "sub_component_inputs", None).unwrap();
        let lw = lower_snippet_with_slots(&[], body_src, slots);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        let s = lw
            .out
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let big_pos = s.find("set_sub_input_word (1").expect("big word at flat 1");
        let addr_pos = s
            .find("set_sub_input_word (0")
            .expect("addr word at flat 0");
        // File order: big (flat 1) is EMITTED before addr (flat 0).
        assert!(
            big_pos < addr_pos,
            "emission must follow file order with decl-order indices"
        );
    }

    // ---- Step-2 front-end: felt widths, felt consts, width conversions, seq ---------

    /// Bit-window decomposition mirrors `Felt252::from([u64;4])` + `get_m31` exactly
    /// (hand-computed vectors; the tool is standalone so no differential dep on the
    /// prover types — the per-component byte-equality gate is the end-to-end arbiter).
    #[test]
    fn felt_const_limb_decomposition() {
        // value = 1 → limb0 = 1, rest 0.
        let l = felt252_const_limbs([1, 0, 0, 0]);
        assert_eq!(l[0], 1);
        assert!(l[1..].iter().all(|&v| v == 0));

        // limb0 = 0x1FF, limb1 = 3 (value = 0x1FF | 3<<9).
        let l = felt252_const_limbs([0x1FF | (3 << 9), 0, 0, 0]);
        assert_eq!((l[0], l[1]), (0x1FF, 3));

        // Word-boundary window: limb 7 spans bits 63..72 → (w0>>63) | (w1<<1).
        let l = felt252_const_limbs([1u64 << 63, 0b1010, 0, 0]);
        assert_eq!(l[7], 1 | (0b1010 << 1));

        // Top word masked to 60 bits (252-bit value): all-ones w3 gives limb27 = 511
        // and no bits beyond 252 leak in.
        let l = felt252_const_limbs([0, 0, 0, u64::MAX]);
        assert_eq!(l[27], 511);
        // Every limb is a canonical 9-bit value.
        assert!(l.iter().all(|&v| v < 512));
    }

    #[test]
    fn felt_const_get_m31_is_const_limb() {
        let mut felts = BTreeMap::new();
        // value 1 → limb0 = 1, limb5 = 0.
        felts.insert(
            "Felt252_1_0_0_0".to_string(),
            felt252_const_limbs([1, 0, 0, 0]),
        );
        let lw = lower_snippet_full(
            &[],
            felts,
            Ty::Unknown,
            "let a = Felt252_1_0_0_0.get_m31(0); let b = Felt252_1_0_0_0.get_m31(5);",
            vec![],
        );
        assert_eq!(lw.env["a"], Ty::ConstM31(1));
        assert_eq!(lw.env["b"], Ty::ConstM31(0));
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        assert!(lw.referenced_m31.contains(&1));
        // No felt materialization needed for limb reads.
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(!s.contains("felt_from_limbs"));
    }

    #[test]
    fn felt_const_bare_use_materializes_from_limbs() {
        let mut felts = BTreeMap::new();
        felts.insert(
            "Felt252_1_0_0_0".to_string(),
            felt252_const_limbs([1, 0, 0, 0]),
        );
        let lw = lower_snippet_full(&[], felts, Ty::Unknown, "let f = Felt252_1_0_0_0;", vec![]);
        assert!(matches!(lw.env["f"], Ty::ConstFelt252(_)));
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("felt_from_limbs"));
    }

    #[test]
    fn hoisted_felt_const_is_parsed() {
        let stmt: Stmt = syn::parse_str(
            "let Felt252_1_2_3_4 = PackedFelt252::broadcast(Felt252::from([1, 2, 3, 4]));",
        )
        .unwrap();
        let Stmt::Local(local) = stmt else { panic!() };
        assert_eq!(local_felt_const(&local), Some([1, 2, 3, 4]));
    }

    /// W27 get_m31: in-range is a census-only site (typed M31, blocks emission — the
    /// recording layer is 28x9 only); out-of-range is a LOUD skip (source bug).
    #[test]
    fn w27_input_get_m31_widths() {
        let lw = lower_snippet_full(
            &[],
            BTreeMap::new(),
            Ty::FeltW27,
            "let a = add_opcode_input.get_m31(9);",
            vec![],
        );
        assert_eq!(lw.env["a"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        // Real now: the input projects to 10 input reads; get_m31(9) is a plain
        // array index on the word tokens.
        assert_eq!(lw.w27_sites, 0);
        assert_eq!(lw.input_sites, 0);

        let lw = lower_snippet_full(
            &[],
            BTreeMap::new(),
            Ty::FeltW27,
            "let a = add_opcode_input.get_m31(10);",
            vec![],
        );
        assert!(
            lw.skips
                .iter()
                .any(|s| s.detail.contains("out of range for Felt252Width27")),
            "skips: {:?}",
            lw.skips
        );
    }

    /// Felt252 get_m31(i >= 28) is a loud skip, never a wrap.
    #[test]
    fn felt252_get_m31_out_of_range_is_loud() {
        let lw = lower_snippet(
            &[],
            "let f = memory_id_to_big_state.deduce_output(memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let a = f.get_m31(28);",
        );
        assert!(
            lw.skips
                .iter()
                .any(|s| s.detail.contains("out of range for Felt252")),
            "skips: {:?}",
            lw.skips
        );
    }

    /// f252 → w27 conversion (G2): 10 limbs, each `f9[3j] + f9[3j+1]*2^9 + f9[3j+2]*2^18`
    /// (j=9 → f9[27] alone); pure felt_get_m31 + m31_mul/m31_add — REAL lowering.
    #[test]
    fn from_packed_felt252_lowers_to_limb_schoolbook() {
        let lw = lower_snippet(
            &[],
            "let f = memory_id_to_big_state.deduce_output(memory_address_to_id_state.deduce_output(add_opcode_input.pc)); \
             let w = PackedFelt252Width27::from_packed_felt252(f); \
             let x = w.get_m31(0); let y = w.get_m31(9);",
        );
        assert_eq!(lw.env["w"], Ty::FeltW27Limbs);
        assert_eq!(lw.env["x"], Ty::M31);
        assert_eq!(lw.env["y"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        assert_eq!(lw.w27_sites, 0, "limb-backed W27 must not be census-only");
        let s = lw
            .out
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        // 28 limb extractions, weighted by 2^9 / 2^18.
        assert_eq!(s.matches("felt_get_m31").count(), 28);
        assert!(lw.referenced_m31.contains(&512));
        assert!(lw.referenced_m31.contains(&262144));
        // get_m31 on the limb-backed value is an array projection, not an eval op.
        assert!(s.contains("[9]"), "limb projection: {s}");
    }

    /// w27 → f252 needs 27-bit shift/mask (u32 extension) — census-only, typed Felt252.
    #[test]
    fn from_packed_felt252width27_lowers_to_real_trait_op() {
        let lw = lower_snippet_full(
            &[],
            BTreeMap::new(),
            Ty::FeltW27,
            "let f = PackedFelt252::from_packed_felt252width27(add_opcode_input); \
             let a = f.get_m31(20);",
            vec![],
        );
        assert_eq!(lw.env["f"], Ty::Felt252);
        assert_eq!(lw.env["a"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        // W27 inputs materialize as 10 real input reads; the conversion is the
        // REAL felt_from_w27_words trait op — nothing censused.
        assert_eq!(lw.w27_sites, 0);
        assert_eq!(lw.input_sites, 0);
        let body = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(body.contains("eval . felt_from_w27_words ("), "body: {body}");
        assert!(body.contains("eval . input ("), "body: {body}");
    }

    /// `seq.packed_at(row_index)` is the packed row index — a REAL `eval.iota()` op
    /// (G4); the record/driver assign its input slot after the flattened input words.
    #[test]
    fn seq_packed_at_is_real_iota() {
        let lw = lower_snippet(&[], "let s = seq.packed_at(row_index); let t = (s) * (s);");
        assert_eq!(lw.env["s"], Ty::M31);
        assert_eq!(lw.env["t"], Ty::M31);
        assert!(lw.skips.is_empty(), "skips: {:?}", lw.skips);
        assert!(lw.uses_iota);
        let s = lw.out.iter().map(|t| t.to_string()).collect::<String>();
        assert!(s.contains("eval . iota ()"), "body: {s}");
    }

    /// from_limbs with the wrong arity is a loud skip (the trait op is `[M31; 28]`).
    #[test]
    fn felt_from_limbs_arity_is_checked() {
        let lw = lower_snippet(
            &[],
            "let a = add_opcode_input.ap; let f = PackedFelt252::from_limbs([a, a, a]);",
        );
        assert!(
            lw.skips.iter().any(|s| s.detail.contains("!= 28 limbs")),
            "skips: {:?}",
            lw.skips
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
