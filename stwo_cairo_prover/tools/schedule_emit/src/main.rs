//! schedule_emit — generate the gpu-prover component schedule table from the
//! transformer-emitted metadata (GPU_RESIDENT_PROVER_DESIGN.md §17, rule R8:
//! the table is derivable; hand-writing it is the data-entry bug class).
//!
//! Ground truth consumed (parse-only, no hand data):
//!  - `SUB_FEED_LAYOUT` in every `crates/prover/src/witness/components/*.rs` (emitted by
//!    witness_genericize; DECLARATION order): one entry per `SubComponentInputs` field instance —
//!    (field, instance, downstream state param, relation_index, word base, words per instance).
//!  - `COUNT_RELATIONS` in `crates/prover/src/witness/device_feed.rs` (state_param, n_relations):
//!    the device-atomic count families.
//!  - every `CairoClaimGenerator` field and component `ClaimGenerator` row source.
//!  - every generated `SubComponentInputs` field for row-capacity multiplicities.
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
//! Writes (or, with --check, byte-compares against) the schedule table, relation
//! graph, and `crates/prover/src/witness/proof_shape_generated.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

/// One parsed SUB_FEED_LAYOUT entry.
#[derive(Debug, Clone)]
struct FeedEntry {
    state_param: String,
    relation_index: u32,
    word_base: usize,
    words_per_instance: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSource {
    DirectInputs,
    StoredLogSize,
    FixedLogSize(u32),
    WitnessPackedInputs,
    WitnessMultiplicityMap,
    MemoryAddress,
    MemoryIdToBig,
}

#[derive(Debug, Clone)]
struct StaticFacts {
    row_source: RowSource,
    lookup_words: Option<u32>,
    sub_words: Option<u32>,
    logup_columns: Option<u32>,
    recorded_witness_kernel: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LookupField {
    name: String,
    word_offset: u32,
    width: u32,
    relation_name: Option<String>,
    relation_id: Option<u32>,
}

/// One denominator use in the exact order consumed by a LogUp output column.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RelationUseFact {
    field: String,
    mult: String,
    negative: bool,
}

/// One generated LogUp output column. Standard component writers contain one or
/// two denominator uses per column.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RelationColumnFact {
    uses: Vec<RelationUseFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TracePartFact {
    Component,
    EachMemoryBig,
    MemorySmall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TupleSourceFact {
    LookupWords(u32),
    MemoryAddressChunk(u32),
    MemoryBigLimbs(u32),
    MemoryBigValue,
    MemorySmallLimbs(u32),
    MemorySmallValue,
    BitwiseXor12(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiplicitySourceFact {
    One,
    Enabler,
    LookupWord(u32),
    MemoryAddressChunk(u32),
    MemoryBig,
    MemorySmall,
    BitwiseXor12(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedUseFact {
    relation_name: String,
    relation_id: u32,
    tuple_source: TupleSourceFact,
    tuple_words: u32,
    multiplicity_source: MultiplicitySourceFact,
    negative: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedTraceFact {
    part: TracePartFact,
    columns: Vec<Vec<PlannedUseFact>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedComponentFact {
    component: String,
    row_source: RowSource,
    lookup_words: Option<u32>,
    traces: Vec<PlannedTraceFact>,
}

#[derive(Debug, Clone, Copy)]
struct CustomRelationGeometry {
    memory_address_chunks: u32,
    felt252_words: u32,
    small_memory_words: u32,
    bitwise_xor_12_columns: u32,
}

fn strip_parens(mut expr: &syn::Expr) -> &syn::Expr {
    while let syn::Expr::Paren(paren) = expr {
        expr = &paren.expr;
    }
    expr
}

fn vec_element_type(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    arguments.args.iter().find_map(|argument| match argument {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

fn lookup_field_width(ty: &syn::Type) -> Option<u32> {
    let element = vec_element_type(ty)?;
    match element {
        syn::Type::Array(array) => lit_int(&array.len).map(|width| width as u32),
        syn::Type::Path(path)
            if path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "PackedM31") =>
        {
            Some(1)
        }
        _ => None,
    }
}

fn relation_name_from_field(field: &str) -> Option<String> {
    let (name, index) = field.rsplit_once('_')?;
    index.parse::<u32>().ok().map(|_| name.to_owned())
}

fn assigned_lookup_field(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Unary(deref) = strip_parens(expr) else {
        return None;
    };
    if !matches!(deref.op, syn::UnOp::Deref(_)) {
        return None;
    }
    let syn::Expr::Field(field) = strip_parens(&deref.expr) else {
        return None;
    };
    let syn::Expr::Path(base) = strip_parens(&field.base) else {
        return None;
    };
    if !base.path.is_ident("lookup_data") {
        return None;
    }
    match &field.member {
        syn::Member::Named(name) => Some(name.to_string()),
        syn::Member::Unnamed(_) => None,
    }
}

fn numeric_m31(expr: &syn::Expr) -> Option<u32> {
    let expr = strip_parens(expr);
    match expr {
        syn::Expr::Path(path) => {
            let ident = path.path.get_ident()?.to_string();
            ident.strip_prefix("M31_")?.parse().ok()
        }
        syn::Expr::MethodCall(call) if call.method == "into" => numeric_m31(&call.receiver),
        syn::Expr::Call(call) => {
            let syn::Expr::Path(function) = strip_parens(&call.func) else {
                return None;
            };
            let last = function.path.segments.last()?.ident.to_string();
            if last != "from" && last != "from_u32_unchecked" {
                return None;
            }
            call.args
                .first()
                .and_then(lit_int)
                .map(|value| value as u32)
        }
        syn::Expr::Cast(cast) => numeric_m31(&cast.expr),
        syn::Expr::Reference(reference) => numeric_m31(&reference.expr),
        _ => None,
    }
}

/// Parse declaration-order lookup layout and prove every relation tuple starts
/// with one stable numeric relation id in the generated row writer.
fn parse_lookup_fields(file: &syn::File, component: &str) -> Option<Vec<LookupField>> {
    let lookup = file.items.iter().find_map(|item| match item {
        syn::Item::Struct(item) if item.ident == "LookupData" => Some(item),
        _ => None,
    })?;
    let syn::Fields::Named(fields) = &lookup.fields else {
        panic!("{component}: LookupData must have named fields");
    };
    let mut word_offset = 0u32;
    let mut result = Vec::with_capacity(fields.named.len());
    for field in &fields.named {
        let name = field.ident.as_ref().expect("named field").to_string();
        let Some(width) = lookup_field_width(&field.ty) else {
            if name.starts_with("mults") {
                return None;
            }
            panic!("{component}: unsupported LookupData field type for {name}");
        };
        let relation_name = (!name.starts_with("mults"))
            .then(|| relation_name_from_field(&name))
            .flatten()
            .or_else(|| {
                assert!(
                    name.starts_with("mults"),
                    "{component}: relation field {name} has no numeric suffix"
                );
                None
            });
        result.push(LookupField {
            name,
            word_offset,
            width,
            relation_name,
            relation_id: None,
        });
        word_offset = word_offset
            .checked_add(width)
            .unwrap_or_else(|| panic!("{component}: lookup word width overflow"));
    }

    struct AssignmentVisitor {
        ids: BTreeMap<String, BTreeSet<u32>>,
    }
    impl<'ast> syn::visit::Visit<'ast> for AssignmentVisitor {
        fn visit_expr_assign(&mut self, assignment: &'ast syn::ExprAssign) {
            if let Some(field) = assigned_lookup_field(&assignment.left) {
                if let syn::Expr::Array(array) = strip_parens(&assignment.right) {
                    if let Some(id) = array.elems.first().and_then(numeric_m31) {
                        self.ids.entry(field).or_default().insert(id);
                    }
                }
            }
            syn::visit::visit_expr_assign(self, assignment);
        }
    }
    let mut visitor = AssignmentVisitor {
        ids: BTreeMap::new(),
    };
    syn::visit::Visit::visit_file(&mut visitor, file);
    for field in &mut result {
        if field.relation_name.is_none() {
            assert_eq!(
                field.width, 1,
                "{component}: multiplicity {} must be scalar",
                field.name
            );
            continue;
        }
        let ids = visitor.ids.get(&field.name).unwrap_or_else(|| {
            panic!(
                "{component}: no numeric relation id assignment for {}",
                field.name
            )
        });
        assert_eq!(
            ids.len(),
            1,
            "{component}: relation field {} has unstable ids {ids:?}",
            field.name
        );
        field.relation_id = ids.iter().next().copied();
    }
    Some(result)
}

/// Parse the exact pair/solo ordering and signed multiplicities from the
/// generated interaction writer. Unknown writer algebra is a generator error.
fn parse_relation_columns(file: &syn::File, component: &str) -> Option<Vec<RelationColumnFact>> {
    use quote::ToTokens;

    fn lookup_field(expr: &syn::Expr) -> Option<String> {
        let expr = strip_parens(expr);
        let inner = if let syn::Expr::Reference(reference) = expr {
            strip_parens(&reference.expr)
        } else {
            expr
        };
        let syn::Expr::Field(field) = inner else {
            return None;
        };
        let syn::Expr::Field(base) = strip_parens(&field.base) else {
            return None;
        };
        let syn::Member::Named(member) = &base.member else {
            return None;
        };
        if member != "lookup_data" {
            return None;
        }
        match &field.member {
            syn::Member::Named(name) => Some(name.to_string()),
            syn::Member::Unnamed(_) => None,
        }
    }

    let body = file.items.iter().find_map(|item| {
        let syn::Item::Impl(item) = item else {
            return None;
        };
        item.items.iter().find_map(|item| match item {
            syn::ImplItem::Fn(function) if function.sig.ident == "write_interaction_trace" => {
                Some(&function.block)
            }
            _ => None,
        })
    })?;

    struct ColumnVisitor {
        columns: Vec<RelationColumnFact>,
        failed: Option<String>,
    }
    impl<'ast> syn::visit::Visit<'ast> for ColumnVisitor {
        fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
            syn::visit::visit_expr_method_call(self, node);
            if node.method != "for_each" || self.failed.is_some() {
                return;
            }
            let mut receiver = strip_parens(&node.receiver);
            if let syn::Expr::MethodCall(call) = receiver {
                if call.method == "enumerate" {
                    receiver = strip_parens(&call.receiver);
                }
            }
            let syn::Expr::MethodCall(par_iter) = receiver else {
                return;
            };
            if par_iter.method != "into_par_iter" {
                return;
            }
            let syn::Expr::Tuple(tuple) = strip_parens(&par_iter.receiver) else {
                return;
            };
            let mut elements = tuple.elems.iter();
            let Some(syn::Expr::MethodCall(first)) = elements.next().map(strip_parens) else {
                return;
            };
            if first.method != "par_iter_mut" {
                return;
            }
            let fields: Vec<String> = elements
                .map(lookup_field)
                .collect::<Option<_>>()
                .unwrap_or_default();
            if fields.is_empty() {
                return;
            }
            let Some(syn::Expr::Closure(closure)) = node.args.first().map(strip_parens) else {
                return;
            };
            let mut numerator = None;
            struct FractionVisitor<'a>(&'a mut Option<String>);
            impl<'ast> syn::visit::Visit<'ast> for FractionVisitor<'_> {
                fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
                    if call.method == "write_frac" {
                        if let Some(argument) = call.args.first() {
                            *self.0 = Some(
                                argument
                                    .to_token_stream()
                                    .to_string()
                                    .chars()
                                    .filter(|character| !character.is_whitespace())
                                    .collect(),
                            );
                        }
                    }
                    syn::visit::visit_expr_method_call(self, call);
                }
            }
            FractionVisitor(&mut numerator).visit_expr(&closure.body);
            let Some(numerator) = numerator else {
                return;
            };
            let use_fact = |field: &str, mult: &str, negative| RelationUseFact {
                field: field.to_owned(),
                mult: mult.to_owned(),
                negative,
            };
            let uses = match (numerator.as_str(), fields.len()) {
                ("denom0**mult1+denom1**mult0", 4) => vec![
                    use_fact(&fields[0], &fields[2], false),
                    use_fact(&fields[1], &fields[3], false),
                ],
                ("-(denom0**mult1+denom1**mult0)", 4) => vec![
                    use_fact(&fields[0], &fields[2], true),
                    use_fact(&fields[1], &fields[3], true),
                ],
                ("denom0+denom1", 2) => vec![
                    use_fact(&fields[0], "1", false),
                    use_fact(&fields[1], "1", false),
                ],
                ("denom1**mult0-denom0**mult1", 4) => vec![
                    use_fact(&fields[0], &fields[2], false),
                    use_fact(&fields[1], &fields[3], true),
                ],
                ("denom0**mult1-denom1**mult0", 4) => vec![
                    use_fact(&fields[0], &fields[2], true),
                    use_fact(&fields[1], &fields[3], false),
                ],
                ("denom0*enabler_col.packed_at(i)+denom1", 2) => vec![
                    use_fact(&fields[0], "1", false),
                    use_fact(&fields[1], "enabler", false),
                ],
                ("(-mult).into()", 2) => vec![use_fact(&fields[0], &fields[1], true)],
                ("(mult).into()", 2) => vec![use_fact(&fields[0], &fields[1], false)],
                ("-PackedQM31::one()*enabler_col.packed_at(i)", 1) => {
                    vec![use_fact(&fields[0], "enabler", true)]
                }
                _ => {
                    self.failed = Some(format!("unrecognized numerator ({fields:?}): {numerator}"));
                    return;
                }
            };
            self.columns.push(RelationColumnFact { uses });
        }
    }
    let mut visitor = ColumnVisitor {
        columns: Vec::new(),
        failed: None,
    };
    syn::visit::Visit::visit_block(&mut visitor, body);
    if let Some(error) = visitor.failed {
        panic!("{component}: {error}");
    }
    (!visitor.columns.is_empty()).then_some(visitor.columns)
}

fn parse_relation_id_catalog(file: &syn::File) -> BTreeMap<String, u32> {
    let mut catalog = BTreeMap::new();
    for item in &file.items {
        let syn::Item::Const(constant) = item else {
            continue;
        };
        let name = constant.ident.to_string();
        let Some(name) = name.strip_suffix("_RELATION_ID") else {
            continue;
        };
        let id = numeric_m31(&constant.expr)
            .unwrap_or_else(|| panic!("{} has no numeric M31 id", constant.ident));
        assert!(
            catalog.insert(name.to_ascii_lowercase(), id).is_none(),
            "duplicate relation id constant: {name}"
        );
    }
    catalog
}

fn literal_const_from_path(path: &Path, name: &str) -> u32 {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let file = syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    parse_usize_const(&file, name)
        .unwrap_or_else(|| panic!("{}: literal const {name} missing", path.display()))
}

fn parse_custom_relation_geometry(root: &Path) -> CustomRelationGeometry {
    let log_memory_address_bound = literal_const_from_path(
        &root.join("crates/common/src/memory.rs"),
        "LOG_MEMORY_ADDRESS_BOUND",
    );
    let max_sequence_log_size = literal_const_from_path(
        &root.join("crates/common/src/preprocessed_columns/preprocessed_trace.rs"),
        "MAX_SEQUENCE_LOG_SIZE",
    );
    let felt252_words = literal_const_from_path(
        &root.join("crates/common/src/prover_types/cpu.rs"),
        "FELT252_N_WORDS",
    );
    let small_memory_words = literal_const_from_path(
        &root.join("crates/common/src/memory.rs"),
        "N_M31_IN_SMALL_FELT252",
    );
    let expand_bits = literal_const_from_path(
        &root.join("crates/cairo-air/src/components/verify_bitwise_xor_12.rs"),
        "EXPAND_BITS",
    );
    CustomRelationGeometry {
        memory_address_chunks: 1u32
            .checked_shl(
                log_memory_address_bound
                    .checked_sub(max_sequence_log_size)
                    .expect("memory-address split exponent"),
            )
            .expect("memory-address split"),
        felt252_words,
        small_memory_words,
        bitwise_xor_12_columns: 1u32
            .checked_shl(expand_bits.checked_mul(2).expect("xor expand bits"))
            .expect("xor multiplicity columns"),
    }
}

fn validate_custom_writer_contract(component: &str, source: &str) {
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let required: &[&str] = match component {
        "memory_address_to_id" => &[
            "izip!(&self.ids,&self.multiplicities).tuples()",
            "MEMORY_ADDRESS_TO_ID_RELATION_ID.into()",
            "writer.write_frac(p0*(-mult1)+p1*(-mult0),p1*p0)",
        ],
        "memory_id_to_big" => &[
            "big_components_values.iter().tuples().enumerate()",
            "small_values.iter().tuples().enumerate()",
            "MEMORY_ID_TO_BIG_RELATION_ID.into()",
            "writer.write_frac(denom0+denom1,denom0*denom1)",
            "(-big_multiplicities[vec_row]).into()",
            "(-self.small_multiplicities[vec_row]).into()",
        ],
        "verify_bitwise_xor_12" => &[
            "self.lookup_data.mults.into_iter().enumerate().tuples()",
            "VERIFY_BITWISE_XOR_12_RELATION_ID.into()",
            "writer.write_frac(p0*(-mults1)+p1*(-mults0),p1*p0)",
        ],
        _ => return,
    };
    for marker in required {
        assert!(
            compact.contains(marker),
            "{component}: computed relation-writer contract changed: missing {marker}"
        );
    }
}

fn relation_id(catalog: &BTreeMap<String, u32>, name: &str) -> u32 {
    *catalog
        .get(name)
        .unwrap_or_else(|| panic!("computed relation {name} has no canonical id constant"))
}

fn computed_use(
    catalog: &BTreeMap<String, u32>,
    relation_name: &str,
    tuple_source: TupleSourceFact,
    tuple_words: u32,
    multiplicity_source: MultiplicitySourceFact,
    negative: bool,
) -> PlannedUseFact {
    PlannedUseFact {
        relation_name: relation_name.to_owned(),
        relation_id: relation_id(catalog, relation_name),
        tuple_source,
        tuple_words,
        multiplicity_source,
        negative,
    }
}

fn plan_computed_component(
    component: &str,
    row_source: RowSource,
    catalog: &BTreeMap<String, u32>,
    geometry: CustomRelationGeometry,
) -> PlannedComponentFact {
    let traces = match component {
        "memory_address_to_id" => {
            assert_eq!(geometry.memory_address_chunks % 2, 0);
            let columns = (0..geometry.memory_address_chunks)
                .step_by(2)
                .map(|chunk| {
                    vec![
                        computed_use(
                            catalog,
                            "memory_address_to_id",
                            TupleSourceFact::MemoryAddressChunk(chunk),
                            3,
                            MultiplicitySourceFact::MemoryAddressChunk(chunk),
                            true,
                        ),
                        computed_use(
                            catalog,
                            "memory_address_to_id",
                            TupleSourceFact::MemoryAddressChunk(chunk + 1),
                            3,
                            MultiplicitySourceFact::MemoryAddressChunk(chunk + 1),
                            true,
                        ),
                    ]
                })
                .collect();
            vec![PlannedTraceFact {
                part: TracePartFact::Component,
                columns,
            }]
        }
        "memory_id_to_big" => {
            assert_eq!(geometry.felt252_words % 4, 0);
            assert_eq!(geometry.small_memory_words % 4, 0);
            let relation_pairs = [
                ("range_check_9_9", "range_check_9_9_b"),
                ("range_check_9_9_c", "range_check_9_9_d"),
                ("range_check_9_9_e", "range_check_9_9_f"),
                ("range_check_9_9_g", "range_check_9_9_h"),
            ];
            let limb_columns = |words: u32, big: bool| {
                (0..words / 4)
                    .map(|column| {
                        let (first, second) =
                            relation_pairs[column as usize % relation_pairs.len()];
                        let first_limb = column * 4;
                        vec![
                            computed_use(
                                catalog,
                                first,
                                if big {
                                    TupleSourceFact::MemoryBigLimbs(first_limb)
                                } else {
                                    TupleSourceFact::MemorySmallLimbs(first_limb)
                                },
                                3,
                                MultiplicitySourceFact::One,
                                false,
                            ),
                            computed_use(
                                catalog,
                                second,
                                if big {
                                    TupleSourceFact::MemoryBigLimbs(first_limb + 2)
                                } else {
                                    TupleSourceFact::MemorySmallLimbs(first_limb + 2)
                                },
                                3,
                                MultiplicitySourceFact::One,
                                false,
                            ),
                        ]
                    })
                    .collect::<Vec<_>>()
            };
            let mut big_columns = limb_columns(geometry.felt252_words, true);
            big_columns.push(vec![computed_use(
                catalog,
                "memory_id_to_big",
                TupleSourceFact::MemoryBigValue,
                geometry.felt252_words + 2,
                MultiplicitySourceFact::MemoryBig,
                true,
            )]);
            let mut small_columns = limb_columns(geometry.small_memory_words, false);
            small_columns.push(vec![computed_use(
                catalog,
                "memory_id_to_big",
                TupleSourceFact::MemorySmallValue,
                geometry.small_memory_words + 2,
                MultiplicitySourceFact::MemorySmall,
                true,
            )]);
            vec![
                PlannedTraceFact {
                    part: TracePartFact::EachMemoryBig,
                    columns: big_columns,
                },
                PlannedTraceFact {
                    part: TracePartFact::MemorySmall,
                    columns: small_columns,
                },
            ]
        }
        "verify_bitwise_xor_12" => {
            assert_eq!(geometry.bitwise_xor_12_columns % 2, 0);
            let columns = (0..geometry.bitwise_xor_12_columns)
                .step_by(2)
                .map(|multiplicity_column| {
                    vec![
                        computed_use(
                            catalog,
                            "verify_bitwise_xor_12",
                            TupleSourceFact::BitwiseXor12(multiplicity_column),
                            4,
                            MultiplicitySourceFact::BitwiseXor12(multiplicity_column),
                            true,
                        ),
                        computed_use(
                            catalog,
                            "verify_bitwise_xor_12",
                            TupleSourceFact::BitwiseXor12(multiplicity_column + 1),
                            4,
                            MultiplicitySourceFact::BitwiseXor12(multiplicity_column + 1),
                            true,
                        ),
                    ]
                })
                .collect();
            vec![PlannedTraceFact {
                part: TracePartFact::Component,
                columns,
            }]
        }
        _ => panic!("{component}: no computed relation planner"),
    };
    PlannedComponentFact {
        component: component.to_owned(),
        row_source,
        lookup_words: None,
        traces,
    }
}

fn plan_standard_component(
    component: &str,
    row_source: RowSource,
    fields: &[LookupField],
    columns: &[RelationColumnFact],
    declared_ids: &BTreeMap<String, u32>,
) -> PlannedComponentFact {
    let by_name: BTreeMap<_, _> = fields
        .iter()
        .map(|field| (field.name.as_str(), field))
        .collect();
    let mut uses = BTreeMap::<&str, u32>::new();
    let columns = columns
        .iter()
        .map(|column| {
            column
                .uses
                .iter()
                .map(|relation_use| {
                    let field = by_name.get(relation_use.field.as_str()).unwrap_or_else(|| {
                        panic!(
                            "{component}: descriptor references unknown tuple {}",
                            relation_use.field
                        )
                    });
                    let relation_name = field.relation_name.as_deref().unwrap_or_else(|| {
                        panic!("{component}: {} is not a relation tuple", field.name)
                    });
                    let relation_id = field.relation_id.expect("relation id established");
                    if let Some(declared) = declared_ids.get(relation_name) {
                        assert_eq!(
                            *declared, relation_id,
                            "{component}: {relation_name} id disagrees with cairo_air::relations"
                        );
                    }
                    *uses.entry(field.name.as_str()).or_default() += 1;
                    let multiplicity_source = match relation_use.mult.as_str() {
                        "1" => MultiplicitySourceFact::One,
                        "enabler" => MultiplicitySourceFact::Enabler,
                        name => {
                            let mult = by_name.get(name).unwrap_or_else(|| {
                                panic!("{component}: unknown multiplicity field {name}")
                            });
                            assert!(
                                mult.relation_name.is_none() && mult.width == 1,
                                "{component}: multiplicity {name} must be scalar"
                            );
                            MultiplicitySourceFact::LookupWord(mult.word_offset)
                        }
                    };
                    PlannedUseFact {
                        relation_name: relation_name.to_owned(),
                        relation_id,
                        tuple_source: TupleSourceFact::LookupWords(field.word_offset),
                        tuple_words: field.width,
                        multiplicity_source,
                        negative: relation_use.negative,
                    }
                })
                .collect()
        })
        .collect();
    for field in fields.iter().filter(|field| field.relation_name.is_some()) {
        assert_eq!(
            uses.get(field.name.as_str()).copied(),
            Some(1),
            "{component}: relation tuple {} must occur in exactly one descriptor",
            field.name
        );
    }
    PlannedComponentFact {
        component: component.to_owned(),
        row_source,
        lookup_words: Some(fields.iter().map(|field| field.width).sum()),
        traces: vec![PlannedTraceFact {
            part: TracePartFact::Component,
            columns,
        }],
    }
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
            if candidate
                .join("crates/prover/src/witness/components")
                .is_dir()
            {
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

fn lit_bool(e: &syn::Expr) -> Option<bool> {
    if let syn::Expr::Lit(literal) = e {
        if let syn::Lit::Bool(value) = &literal.lit {
            return Some(value.value);
        }
    }
    None
}

struct JitLookupLayout {
    words: u32,
    fields: Vec<(String, u32)>,
}

impl syn::parse::Parse for JitLookupLayout {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(syn::Ident) && input.peek2(syn::LitInt) {
            let marker: syn::Ident = input.parse()?;
            if marker != "with_n_rows" {
                return Err(syn::Error::new(marker.span(), "expected with_n_rows"));
            }
        }
        let words: syn::LitInt = input.parse()?;
        input.parse::<syn::Token![;]>()?;
        let mut fields = Vec::new();
        while !input.is_empty() {
            let field: syn::Ident = input.parse()?;
            input.parse::<syn::Token![:]>()?;
            let width = if input.peek(syn::LitInt) {
                input.parse::<syn::LitInt>()?.base10_parse()?
            } else {
                let scalar: syn::Ident = input.parse()?;
                if scalar != "scalar" {
                    return Err(syn::Error::new(scalar.span(), "expected scalar"));
                }
                1
            };
            fields.push((field.to_string(), width));
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }
        Ok(Self {
            words: words.base10_parse()?,
            fields,
        })
    }
}

fn parse_jit_lookup_layout(file: &syn::File) -> Option<JitLookupLayout> {
    file.items.iter().find_map(|item| {
        let syn::Item::Macro(item) = item else {
            return None;
        };
        item.mac
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "jit_lookup_accessor")
            .then(|| syn::parse2(item.mac.tokens.clone()).expect("parse jit_lookup_accessor"))
    })
}

fn parse_jit_logup_columns(file: &syn::File) -> Option<Vec<RelationColumnFact>> {
    let constant = file.items.iter().find_map(|item| match item {
        syn::Item::Const(constant) if constant.ident == "JIT_LOGUP_DESCS" => Some(constant),
        _ => None,
    })?;
    let syn::Expr::Reference(reference) = constant.expr.as_ref() else {
        panic!("JIT_LOGUP_DESCS is not a reference");
    };
    let syn::Expr::Array(array) = reference.expr.as_ref() else {
        panic!("JIT_LOGUP_DESCS is not an array");
    };
    Some(
        array
            .elems
            .iter()
            .map(|entry| {
                let syn::Expr::Tuple(tuple) = entry else {
                    panic!("JIT_LOGUP_DESCS entry is not a tuple");
                };
                let values: Vec<_> = tuple.elems.iter().collect();
                assert_eq!(values.len(), 6, "JIT_LOGUP_DESCS tuple arity");
                let a_field = lit_str(values[0]).expect("a field");
                let a_mult = lit_str(values[1]).expect("a mult");
                let a_negative = lit_bool(values[2]).expect("a sign");
                let b_field = lit_str(values[3]).expect("b field");
                let b_mult = lit_str(values[4]).expect("b mult");
                let b_negative = lit_bool(values[5]).expect("b sign");
                let mut uses = vec![RelationUseFact {
                    field: a_field,
                    mult: a_mult,
                    negative: a_negative,
                }];
                if !b_field.is_empty() {
                    uses.push(RelationUseFact {
                        field: b_field,
                        mult: b_mult,
                        negative: b_negative,
                    });
                }
                RelationColumnFact { uses }
            })
            .collect(),
    )
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

fn type_contains_claim_generator(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segments: Vec<_> = path.path.segments.iter().collect();
    if segments
        .last()
        .is_some_and(|segment| segment.ident == "ClaimGenerator")
    {
        return segments
            .get(segments.len().checked_sub(2)?)
            .map(|segment| segment.ident.to_string());
    }
    for segment in segments {
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            continue;
        };
        for argument in &arguments.args {
            if let syn::GenericArgument::Type(inner) = argument {
                if let Some(component) = type_contains_claim_generator(inner) {
                    return Some(component);
                }
            }
        }
    }
    None
}

fn parse_claim_components(file: &syn::File) -> BTreeSet<String> {
    let aggregate = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == "CairoClaimGenerator" => Some(item),
            _ => None,
        })
        .expect("CairoClaimGenerator struct not found");
    let syn::Fields::Named(fields) = &aggregate.fields else {
        panic!("CairoClaimGenerator must have named fields");
    };
    let components: BTreeSet<_> = fields
        .named
        .iter()
        .filter_map(|field| type_contains_claim_generator(&field.ty))
        .collect();
    assert!(
        !components.is_empty(),
        "CairoClaimGenerator component set is empty"
    );
    components
}

fn parse_usize_const(file: &syn::File, name: &str) -> Option<u32> {
    file.items.iter().find_map(|item| {
        let syn::Item::Const(item) = item else {
            return None;
        };
        (item.ident == name)
            .then(|| lit_int(&item.expr).map(|value| value as u32))
            .flatten()
    })
}

fn packed_input_component(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segments: Vec<_> = path.path.segments.iter().collect();
    if segments
        .last()
        .is_some_and(|segment| segment.ident == "PackedInputType")
    {
        return segments
            .get(segments.len().checked_sub(2)?)
            .map(|segment| segment.ident.to_string());
    }
    for segment in segments {
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            continue;
        };
        for argument in &arguments.args {
            if let syn::GenericArgument::Type(inner) = argument {
                if let Some(component) = packed_input_component(inner) {
                    return Some(component);
                }
            }
        }
    }
    None
}

/// Producer row -> consumer input multiplicities from generated
/// `SubComponentInputs`, independent of device-word metadata coverage.
fn parse_capacity_outputs(file: &syn::File) -> BTreeMap<String, u32> {
    let Some(inputs) = file.items.iter().find_map(|item| match item {
        syn::Item::Struct(item) if item.ident == "SubComponentInputs" => Some(item),
        _ => None,
    }) else {
        return BTreeMap::new();
    };
    let syn::Fields::Named(fields) = &inputs.fields else {
        panic!("SubComponentInputs must have named fields");
    };
    let mut outputs = BTreeMap::<String, u32>::new();
    for field in &fields.named {
        let syn::Type::Array(array) = &field.ty else {
            panic!("SubComponentInputs field is not an array: {:?}", field.ty);
        };
        let consumer = packed_input_component(&array.elem)
            .unwrap_or_else(|| panic!("SubComponentInputs field has no PackedInputType"));
        let n_instances = lit_int(&array.len).expect("SubComponentInputs array length") as u32;
        *outputs.entry(consumer).or_default() += n_instances;
    }
    outputs
}

fn classify_row_source(component: &str, file: &syn::File, root: &Path) -> RowSource {
    if component == "memory_address_to_id" {
        return RowSource::MemoryAddress;
    }
    if component == "memory_id_to_big" {
        return RowSource::MemoryIdToBig;
    }
    let generator = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == "ClaimGenerator" => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{component}: ClaimGenerator struct not found"));
    let syn::Fields::Named(fields) = &generator.fields else {
        panic!("{component}: ClaimGenerator must have named fields");
    };
    let field = |name: &str| {
        fields
            .named
            .iter()
            .find(|field| field.ident.as_ref().is_some_and(|ident| ident == name))
    };
    if field("inputs").is_some() {
        return RowSource::DirectInputs;
    }
    if field("packed_inputs").is_some() && field("remainder_inputs").is_some() {
        return RowSource::WitnessPackedInputs;
    }
    if field("log_size").is_some() {
        return RowSource::StoredLogSize;
    }
    if let Some(mults) = field("mults") {
        return match &mults.ty {
            syn::Type::Array(_) => {
                let air_path = root.join(format!("crates/cairo-air/src/components/{component}.rs"));
                let air_source = std::fs::read_to_string(&air_path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", air_path.display()));
                let air = syn::parse_file(&air_source)
                    .unwrap_or_else(|error| panic!("parse {}: {error}", air_path.display()));
                let log_size = parse_usize_const(&air, "LOG_SIZE").unwrap_or_else(|| {
                    assert_eq!(
                        component, "verify_bitwise_xor_12",
                        "{component}: non-literal fixed LOG_SIZE needs an evaluator"
                    );
                    let elem_bits = parse_usize_const(&air, "ELEM_BITS").expect("ELEM_BITS");
                    let expand_bits = parse_usize_const(&air, "EXPAND_BITS").expect("EXPAND_BITS");
                    (elem_bits - expand_bits) * 2
                });
                RowSource::FixedLogSize(log_size)
            }
            syn::Type::Path(path)
                if path
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "DashMap") =>
            {
                RowSource::WitnessMultiplicityMap
            }
            other => panic!("{component}: unsupported multiplicity row source: {other:?}"),
        };
    }
    panic!("{component}: no mechanically supported row source")
}

fn parse_recorded_witness_labels(file: &syn::File) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    for item in &file.items {
        match item {
            syn::Item::Macro(item) if item.mac.path.is_ident("opcode_lane_spec") => {
                let tokens = item.mac.tokens.to_string();
                let mut quoted = tokens.split('"');
                let _ = quoted.next();
                let label = quoted
                    .next()
                    .unwrap_or_else(|| panic!("opcode_lane_spec missing string label: {tokens}"));
                labels.insert(label.to_owned());
            }
            syn::Item::Impl(item)
                if item.trait_.as_ref().is_some_and(|(_, path, _)| {
                    path.segments
                        .last()
                        .is_some_and(|segment| segment.ident == "BuiltinLaneSpec")
                }) =>
            {
                for impl_item in &item.items {
                    let syn::ImplItem::Const(item) = impl_item else {
                        continue;
                    };
                    if item.ident == "LABEL" {
                        labels.insert(lit_str(&item.expr).expect("BuiltinLaneSpec LABEL literal"));
                    }
                }
            }
            _ => {}
        }
    }
    assert!(!labels.is_empty(), "recorded witness lane set is empty");
    labels
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
    /// producer component -> scalar input instances per producer row.
    capacity_inputs: BTreeMap<String, u32>,
    /// whether this component has its own emitted lane metadata (vs. existing only
    /// as a feed target).
    has_layout: bool,
    static_facts: Option<StaticFacts>,
    lookup_fields: Option<Vec<LookupField>>,
    relation_columns: Option<Vec<RelationColumnFact>>,
}

fn format_rust(root: &Path, source: &str) -> Result<String, String> {
    let mut child = Command::new("rustfmt")
        .args([
            "--emit",
            "stdout",
            "--edition",
            "2021",
            "--config-path",
            "rustfmt.toml",
        ])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("run rustfmt: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped rustfmt stdin")
        .write_all(source.as_bytes())
        .map_err(|e| format!("write rustfmt stdin: {e}"))?;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait for rustfmt: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("rustfmt output is not UTF-8: {e}"))
}

fn option_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| format!("Some({value})"))
}

fn emit_trace_columns(component: &str) -> String {
    if component == "memory_id_to_big" {
        "TraceColumnCount::SplitMemory {\n            big: cairo_air::components::memory_id_to_big::BIG_N_COLUMNS as u32,\n            small: cairo_air::components::memory_id_to_small::N_TRACE_COLUMNS as u32,\n        }"
            .to_owned()
    } else {
        format!(
            "TraceColumnCount::Fixed(cairo_air::components::{component}::N_TRACE_COLUMNS as u32)"
        )
    }
}

fn emit_row_source(_component: &str, source: RowSource) -> String {
    match source {
        RowSource::DirectInputs => "ComponentRowSource::DirectInputs".to_owned(),
        RowSource::StoredLogSize => "ComponentRowSource::StoredLogSize".to_owned(),
        RowSource::FixedLogSize(log_size) => {
            format!("ComponentRowSource::FixedLogSize({log_size})")
        }
        RowSource::WitnessPackedInputs | RowSource::WitnessMultiplicityMap => {
            "ComponentRowSource::WitnessRelationFeeds".to_owned()
        }
        RowSource::MemoryAddress => "ComponentRowSource::MemoryAddress".to_owned(),
        RowSource::MemoryIdToBig => "ComponentRowSource::MemoryIdToBig".to_owned(),
    }
}

fn emit_runtime_component(component: &str, source: RowSource) -> String {
    let absent = format!("RuntimeComponentShape::absent(\"{component}\")");
    let resolved = match source {
        RowSource::DirectInputs => format!(
            "{{\n                let n_real_rows = gen.inputs.len() as u64;\n                let padded_rows = padded_rows(\"{component}\", n_real_rows, N_LANES as u64)?;\n                RuntimeComponentShape::uniform(\"{component}\", n_real_rows, padded_rows)?\n            }}"
        ),
        RowSource::StoredLogSize => format!(
            "{{\n                let rows = rows_from_log_size(\"{component}\", gen.log_size)?;\n                RuntimeComponentShape::uniform(\"{component}\", rows, rows)?\n            }}"
        ),
        RowSource::FixedLogSize(_) => format!(
            "{{\n                let rows = rows_from_log_size(\"{component}\", cairo_air::components::{component}::LOG_SIZE)?;\n                RuntimeComponentShape::uniform(\"{component}\", rows, rows)?\n            }}"
        ),
        RowSource::WitnessPackedInputs => format!(
            "{{\n                let packed_rows = gen.packed_inputs.lock().expect(\"{component} packed-input mutex poisoned\").len() as u64;\n                let remainder_rows = gen.remainder_inputs.lock().expect(\"{component} remainder-input mutex poisoned\").len() as u64;\n                let observed_n_real_rows = packed_rows\n                    .checked_mul(N_LANES as u64)\n                    .and_then(|rows| rows.checked_add(remainder_rows))\n                    .ok_or(ProofShapeError::RowCountOverflow(\"{component}\"))?;\n                RuntimeComponentShape::pending(\n                    \"{component}\",\n                    PendingRowsReason::WitnessRelationFeeds,\n                    observed_n_real_rows,\n                )\n            }}"
        ),
        RowSource::WitnessMultiplicityMap => format!(
            "RuntimeComponentShape::pending(\n                \"{component}\",\n                PendingRowsReason::WitnessRelationFeeds,\n                gen.mults.len() as u64,\n            )"
        ),
        RowSource::MemoryAddress => format!(
            "{{\n                let split = cairo_air::components::memory_address_to_id::MEMORY_ADDRESS_TO_ID_SPLIT as u64;\n                let n_real_rows = (gen.table_size() as u64) / split;\n                let padded_rows = padded_rows(\"{component}\", n_real_rows, N_LANES as u64)?;\n                RuntimeComponentShape::uniform(\"{component}\", n_real_rows, padded_rows)?\n            }}"
        ),
        RowSource::MemoryIdToBig => {
            "{\n                let max_big_rows = rows_from_log_size(\n                    \"memory_id_to_big\",\n                    stwo_cairo_common::preprocessed_columns::preprocessed_trace::MAX_SEQUENCE_LOG_SIZE,\n                )? as usize;\n                let big_rows = gen.big_table_size();\n                let required_big_components = big_rows.div_ceil(max_big_rows);\n                let n_big_components = opt_n_id_to_big_components\n                    .unwrap_or(required_big_components);\n                if n_big_components < required_big_components {\n                    return Err(ProofShapeError::InvalidMemoryComponentCount {\n                        requested: n_big_components,\n                        required: required_big_components,\n                    });\n                }\n                let mut parts = Vec::with_capacity(n_big_components + 1);\n                for index in 0..n_big_components {\n                    let n_real_rows = if index < required_big_components {\n                        big_rows.saturating_sub(index * max_big_rows).min(max_big_rows)\n                    } else {\n                        N_LANES\n                    } as u64;\n                    let padded_rows = padded_rows(\n                        \"memory_id_to_big\",\n                        n_real_rows,\n                        N_LANES as u64,\n                    )?;\n                    parts.push(TracePartShape {\n                        part: TracePartId::MemoryBig(index as u32),\n                        n_real_rows,\n                        padded_rows,\n                    });\n                }\n                let small_real_rows = gen.small_table_size() as u64;\n                let small_padded_rows = padded_rows(\n                    \"memory_id_to_big\",\n                    small_real_rows,\n                    N_LANES as u64,\n                )?;\n                parts.push(TracePartShape {\n                    part: TracePartId::MemorySmall,\n                    n_real_rows: small_real_rows,\n                    padded_rows: small_padded_rows,\n                });\n                RuntimeComponentShape::parts(\"memory_id_to_big\", parts)?\n            }"
                .to_owned()
        }
    };
    let binding = if matches!(source, RowSource::FixedLogSize(_)) {
        "_gen"
    } else {
        "gen"
    };
    format!(
        "        components.push(match &self.{component} {{\n            None => {absent},\n            Some({binding}) => {resolved},\n        }});\n"
    )
}

fn emit_proof_shape_source(nodes: &BTreeMap<String, Node>) -> String {
    let mut source = String::new();
    source.push_str(
        "//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT.\n\
         //! Complete pre-consumption row-source projection of CairoClaimGenerator.\n\n\
         use stwo::prover::backend::simd::m31::N_LANES;\n\n\
         use super::cairo_claim_generator::CairoClaimGenerator;\n\
         use super::proof_shape::{\n\
             padded_rows, rows_from_log_size, PendingRowsReason, ProofShape, ProofShapeError,\n\
             RuntimeComponentShape, TracePartId, TracePartShape,\n\
         };\n\n\
         impl CairoClaimGenerator {\n\
             pub fn proof_shape(\n\
                 &self,\n\
                 opt_n_id_to_big_components: Option<usize>,\n\
             ) -> Result<ProofShape, ProofShapeError> {\n\
                 let mut components = Vec::new();\n",
    );
    for (component, node) in nodes {
        source.push_str(&emit_runtime_component(
            component,
            node.static_facts.as_ref().unwrap().row_source,
        ));
    }
    source.push_str("        ProofShape::new(components)\n    }\n}\n");
    source
}

fn emit_tuple_source(source: TupleSourceFact) -> String {
    match source {
        TupleSourceFact::LookupWords(word_offset) => {
            format!("TupleSource::LookupWords {{ word_offset: {word_offset} }}")
        }
        TupleSourceFact::MemoryAddressChunk(chunk) => {
            format!("TupleSource::MemoryAddressChunk {{ chunk: {chunk} }}")
        }
        TupleSourceFact::MemoryBigLimbs(first_limb) => {
            format!("TupleSource::MemoryBigLimbs {{ first_limb: {first_limb} }}")
        }
        TupleSourceFact::MemoryBigValue => "TupleSource::MemoryBigValue".to_owned(),
        TupleSourceFact::MemorySmallLimbs(first_limb) => {
            format!("TupleSource::MemorySmallLimbs {{ first_limb: {first_limb} }}")
        }
        TupleSourceFact::MemorySmallValue => "TupleSource::MemorySmallValue".to_owned(),
        TupleSourceFact::BitwiseXor12(multiplicity_column) => {
            format!("TupleSource::BitwiseXor12 {{ multiplicity_column: {multiplicity_column} }}")
        }
    }
}

fn emit_multiplicity_source(source: MultiplicitySourceFact) -> String {
    match source {
        MultiplicitySourceFact::One => "MultiplicitySource::One".to_owned(),
        MultiplicitySourceFact::Enabler => "MultiplicitySource::Enabler".to_owned(),
        MultiplicitySourceFact::LookupWord(word_offset) => {
            format!("MultiplicitySource::LookupWord {{ word_offset: {word_offset} }}")
        }
        MultiplicitySourceFact::MemoryAddressChunk(chunk) => {
            format!("MultiplicitySource::MemoryAddressChunk {{ chunk: {chunk} }}")
        }
        MultiplicitySourceFact::MemoryBig => "MultiplicitySource::MemoryBig".to_owned(),
        MultiplicitySourceFact::MemorySmall => "MultiplicitySource::MemorySmall".to_owned(),
        MultiplicitySourceFact::BitwiseXor12(multiplicity_column) => format!(
            "MultiplicitySource::BitwiseXor12 {{ multiplicity_column: {multiplicity_column} }}"
        ),
    }
}

fn emit_trace_part(part: TracePartFact) -> &'static str {
    match part {
        TracePartFact::Component => "RelationTracePart::Component",
        TracePartFact::EachMemoryBig => "RelationTracePart::EachMemoryBig",
        TracePartFact::MemorySmall => "RelationTracePart::MemorySmall",
    }
}

fn fnv_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn fnv_str(hash: &mut u64, value: &str) {
    fnv_u32(hash, value.len() as u32);
    for byte in value.bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn row_source_identity(source: RowSource) -> (u32, u32) {
    match source {
        RowSource::DirectInputs => (0, 0),
        RowSource::StoredLogSize => (1, 0),
        RowSource::FixedLogSize(log_size) => (2, log_size),
        RowSource::WitnessPackedInputs | RowSource::WitnessMultiplicityMap => (3, 0),
        RowSource::MemoryAddress => (4, 0),
        RowSource::MemoryIdToBig => (5, 0),
    }
}

fn tuple_source_identity(source: TupleSourceFact) -> (u32, u32) {
    match source {
        TupleSourceFact::LookupWords(index) => (0, index),
        TupleSourceFact::MemoryAddressChunk(index) => (1, index),
        TupleSourceFact::MemoryBigLimbs(index) => (2, index),
        TupleSourceFact::MemoryBigValue => (3, 0),
        TupleSourceFact::MemorySmallLimbs(index) => (4, index),
        TupleSourceFact::MemorySmallValue => (5, 0),
        TupleSourceFact::BitwiseXor12(index) => (6, index),
    }
}

fn multiplicity_source_identity(source: MultiplicitySourceFact) -> (u32, u32) {
    match source {
        MultiplicitySourceFact::One => (0, 0),
        MultiplicitySourceFact::Enabler => (1, 0),
        MultiplicitySourceFact::LookupWord(index) => (2, index),
        MultiplicitySourceFact::MemoryAddressChunk(index) => (3, index),
        MultiplicitySourceFact::MemoryBig => (4, 0),
        MultiplicitySourceFact::MemorySmall => (5, 0),
        MultiplicitySourceFact::BitwiseXor12(index) => (6, index),
    }
}

fn relation_graph_hash(
    relations: &BTreeMap<String, u32>,
    components: &[PlannedComponentFact],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for (name, id) in relations {
        fnv_str(&mut hash, name);
        fnv_u32(&mut hash, *id);
    }
    for component in components {
        fnv_str(&mut hash, &component.component);
        let (row_source_tag, row_source_payload) = row_source_identity(component.row_source);
        fnv_u32(&mut hash, row_source_tag);
        fnv_u32(&mut hash, row_source_payload);
        fnv_u32(&mut hash, component.lookup_words.unwrap_or(u32::MAX));
        for trace in &component.traces {
            fnv_u32(
                &mut hash,
                match trace.part {
                    TracePartFact::Component => 0,
                    TracePartFact::EachMemoryBig => 1,
                    TracePartFact::MemorySmall => 2,
                },
            );
            fnv_u32(&mut hash, trace.columns.len() as u32);
            for (output_column, uses) in trace.columns.iter().enumerate() {
                fnv_u32(&mut hash, output_column as u32);
                fnv_u32(&mut hash, uses.len() as u32);
                for relation_use in uses {
                    fnv_str(&mut hash, &relation_use.relation_name);
                    fnv_u32(&mut hash, relation_use.relation_id);
                    fnv_u32(&mut hash, 1);
                    let (tuple_tag, tuple_index) = tuple_source_identity(relation_use.tuple_source);
                    fnv_u32(&mut hash, tuple_tag);
                    fnv_u32(&mut hash, tuple_index);
                    fnv_u32(&mut hash, relation_use.tuple_words);
                    fnv_u32(&mut hash, 0);
                    fnv_u32(&mut hash, 0);
                    fnv_u32(&mut hash, 1);
                    let (multiplicity_tag, multiplicity_index) =
                        multiplicity_source_identity(relation_use.multiplicity_source);
                    fnv_u32(&mut hash, multiplicity_tag);
                    fnv_u32(&mut hash, multiplicity_index);
                    fnv_u32(&mut hash, relation_use.negative as u32);
                }
            }
        }
    }
    hash
}

fn emit_relation_source(components: &[PlannedComponentFact]) -> String {
    let mut relation_ids = BTreeMap::<String, u32>::new();
    let mut relation_names = BTreeMap::<u32, String>::new();
    for component in components {
        for relation_use in component
            .traces
            .iter()
            .flat_map(|trace| &trace.columns)
            .flatten()
        {
            if let Some(previous) =
                relation_ids.insert(relation_use.relation_name.clone(), relation_use.relation_id)
            {
                assert_eq!(
                    previous, relation_use.relation_id,
                    "relation {} has unstable ids",
                    relation_use.relation_name
                );
            }
            if let Some(previous) =
                relation_names.insert(relation_use.relation_id, relation_use.relation_name.clone())
            {
                assert_eq!(
                    previous, relation_use.relation_name,
                    "relation id {} has multiple names",
                    relation_use.relation_id
                );
            }
        }
    }

    let expected_hash = relation_graph_hash(&relation_ids, components);
    let mut source = String::new();
    source.push_str(
        "//! MACHINE-WRITTEN by tools/schedule_emit — DO NOT EDIT.\n\
         //! Exact CommonLookupElements denominator graph parsed from witness interaction writers.\n\
         //! Regenerate with `cargo run --manifest-path tools/schedule_emit/Cargo.toml`.\n\n\
         use crate::relation::{\n\
             ComponentRelationPlan, DenominatorLayout, LogupColumnPlan, MultiplicitySign,\n\
             MultiplicitySource, RelationGraph, RelationIdentity, RelationTracePart,\n\
             RelationTracePlan, RelationUse, SignedMultiplicity, TupleLayout, TupleSource,\n\
             COMMON_LOOKUP_CHALLENGE_EPOCH,\n\
         };\n\
         use crate::schedule::ComponentRowSource;\n\n\
         pub static CAIRO_RELATION_GRAPH: RelationGraph = RelationGraph {\n\
             relations: RELATIONS,\n\
             components: COMPONENTS,\n\
             expected_hash: EXPECTED_HASH,\n\
         };\n\n",
    );
    source.push_str(&format!(
        "pub const EXPECTED_HASH: u64 = 0x{expected_hash:016x};\n\n"
    ));
    source.push_str(
        "const fn relation_use(\n\
             name: &'static str,\n\
             id: u32,\n\
             source: TupleSource,\n\
             words: u32,\n\
             multiplicity: MultiplicitySource,\n\
             negative: bool,\n\
         ) -> RelationUse {\n\
             RelationUse {\n\
                 relation: RelationIdentity { name, id },\n\
                 challenge_epoch: COMMON_LOOKUP_CHALLENGE_EPOCH,\n\
                 denominator: DenominatorLayout {\n\
                     tuple: TupleLayout { source, words, relation_id_word: 0 },\n\
                     alpha_power_start: 0,\n\
                     subtract_z: true,\n\
                 },\n\
                 multiplicity: SignedMultiplicity {\n\
                     source: multiplicity,\n\
                     sign: if negative { MultiplicitySign::Negative } else { MultiplicitySign::Positive },\n\
                 },\n\
             }\n\
         }\n\n",
    );
    source.push_str("static RELATIONS: &[RelationIdentity] = &[\n");
    for (name, id) in &relation_ids {
        source.push_str(&format!(
            "    RelationIdentity {{ name: \"{name}\", id: {id} }},\n"
        ));
    }
    source.push_str("];\n\nstatic COMPONENTS: &[ComponentRelationPlan] = &[\n");
    for component in components {
        source.push_str("    ComponentRelationPlan {\n");
        source.push_str(&format!(
            "        component: \"{}\",\n",
            component.component
        ));
        source.push_str(&format!(
            "        row_source: {},\n",
            emit_row_source(&component.component, component.row_source)
        ));
        source.push_str(&format!(
            "        lookup_words: {},\n",
            option_u32(component.lookup_words)
        ));
        source.push_str("        traces: &[\n");
        for trace in &component.traces {
            source.push_str("            RelationTracePlan {\n");
            source.push_str(&format!(
                "                part: {},\n",
                emit_trace_part(trace.part)
            ));
            source.push_str(&format!(
                "                output_columns: {},\n",
                trace.columns.len()
            ));
            source.push_str("                columns: &[\n");
            for (output_column, uses) in trace.columns.iter().enumerate() {
                source.push_str("                    LogupColumnPlan {\n");
                source.push_str(&format!(
                    "                        output_column: {output_column},\n"
                ));
                source.push_str("                        uses: &[\n");
                for relation_use in uses {
                    source.push_str(&format!(
                        "                            relation_use(\"{}\", {}, {}, {}, {}, {}),\n",
                        relation_use.relation_name,
                        relation_use.relation_id,
                        emit_tuple_source(relation_use.tuple_source),
                        relation_use.tuple_words,
                        emit_multiplicity_source(relation_use.multiplicity_source),
                        relation_use.negative,
                    ));
                }
                source.push_str("                        ],\n");
                source.push_str("                    },\n");
            }
            source.push_str("                ],\n");
            source.push_str("            },\n");
        }
        source.push_str("        ],\n");
        source.push_str("    },\n");
    }
    source.push_str("];\n");
    source
}

fn main() -> ExitCode {
    let (root, check) = parse_args();
    let components_dir = root.join("crates/prover/src/witness/components");
    let device_feed = root.join("crates/prover/src/witness/device_feed.rs");
    let claim_generator_path = root.join("crates/prover/src/witness/cairo_claim_generator.rs");
    let jit_backend_path = root.join("crates/prover/src/witness/jit_prove_backend.rs");
    let relations_path = root.join("crates/cairo-air/src/relations.rs");
    let out_path = root.join("crates/gpu-prover/src/schedule_table.rs");
    let relation_out = root.join("crates/gpu-prover/src/relation_table.rs");
    let proof_shape_out = root.join("crates/prover/src/witness/proof_shape_generated.rs");

    let declared_relation_ids = parse_relation_id_catalog(
        &syn::parse_file(
            &std::fs::read_to_string(&relations_path).expect("read cairo-air relations"),
        )
        .expect("parse cairo-air relations"),
    );
    let custom_geometry = parse_custom_relation_geometry(&root);

    let count_families = parse_count_relations(
        &syn::parse_file(&std::fs::read_to_string(&device_feed).expect("read device_feed.rs"))
            .expect("parse device_feed.rs"),
    );

    let claim_components = parse_claim_components(
        &syn::parse_file(
            &std::fs::read_to_string(&claim_generator_path).expect("read Cairo claim generator"),
        )
        .expect("parse Cairo claim generator"),
    );
    let recorded_witness_labels = parse_recorded_witness_labels(
        &syn::parse_file(
            &std::fs::read_to_string(&jit_backend_path).expect("read JIT prove backend"),
        )
        .expect("parse JIT prove backend"),
    );

    let mut nodes: BTreeMap<String, Node> = claim_components
        .iter()
        .map(|component| (component.clone(), Node::default()))
        .collect();
    let mut capacity_edges: Vec<(String, String, u32)> = Vec::new();
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
        if claim_components.contains(&stem) {
            validate_custom_writer_contract(&stem, &src);
            let lookup_fields = parse_lookup_fields(&file, &stem);
            let relation_columns = parse_relation_columns(&file, &stem);
            if let Some(jit_layout) = parse_jit_lookup_layout(&file) {
                let fields = lookup_fields
                    .as_ref()
                    .unwrap_or_else(|| panic!("{stem}: JIT lookup layout without LookupData"));
                let parsed: Vec<_> = fields
                    .iter()
                    .map(|field| (field.name.clone(), field.width))
                    .collect();
                assert_eq!(parsed, jit_layout.fields, "{stem}: JIT_LOOKUP_FIELDS drift");
                assert_eq!(
                    fields.iter().map(|field| field.width).sum::<u32>(),
                    jit_layout.words,
                    "{stem}: JIT lookup word count drift"
                );
                assert_eq!(
                    parse_jit_logup_columns(&file).as_ref(),
                    relation_columns.as_ref(),
                    "{stem}: JIT_LOGUP_DESCS drift from write_interaction_trace"
                );
            }
            let lookup_words = lookup_fields
                .as_ref()
                .map(|fields| fields.iter().map(|field| field.width).sum());
            let logup_columns = relation_columns
                .as_ref()
                .map(|columns| columns.len() as u32)
                .or_else(|| match stem.as_str() {
                    "memory_address_to_id" => Some(custom_geometry.memory_address_chunks / 2),
                    "verify_bitwise_xor_12" => Some(custom_geometry.bitwise_xor_12_columns / 2),
                    "memory_id_to_big" => None,
                    _ => None,
                });
            let facts = StaticFacts {
                row_source: classify_row_source(&stem, &file, &root),
                lookup_words,
                sub_words: parse_usize_const(&file, "N_SUB_INPUT_WORDS"),
                logup_columns,
                recorded_witness_kernel: recorded_witness_labels.contains(&stem),
            };
            let node = nodes
                .get_mut(&stem)
                .expect("claim component was materialized");
            assert!(
                node.static_facts.replace(facts).is_none(),
                "duplicate component source: {stem}"
            );
            node.lookup_fields = lookup_fields;
            node.relation_columns = relation_columns;
            capacity_edges.extend(
                parse_capacity_outputs(&file)
                    .into_iter()
                    .map(|(consumer, n_instances)| (stem.clone(), consumer, n_instances)),
            );
        }
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
                let distinct: BTreeSet<u32> = group.iter().map(|e| e.relation_index).collect();
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
            assert!(
                claim_components.contains(&consumer),
                "feed target {consumer} is absent from CairoClaimGenerator"
            );
            nodes
                .get_mut(&consumer)
                .expect("feed target was materialized")
                .inputs
                .insert(producer.clone(), edge);
        }
    }
    for (producer, consumer, n_instances) in capacity_edges {
        assert!(
            claim_components.contains(&consumer),
            "capacity-feed target {consumer} is absent from CairoClaimGenerator"
        );
        let previous = nodes
            .get_mut(&consumer)
            .expect("capacity-feed target was materialized")
            .capacity_inputs
            .insert(producer.clone(), n_instances);
        assert!(
            previous.is_none(),
            "duplicate capacity feed: {producer} -> {consumer}"
        );
    }
    for (component, node) in &nodes {
        assert!(
            node.static_facts.is_some(),
            "{component}: component source/static facts missing"
        );
    }
    for label in &recorded_witness_labels {
        assert!(
            claim_components.contains(label),
            "recorded witness lane {label} is absent from CairoClaimGenerator"
        );
    }

    let planned_components: Vec<PlannedComponentFact> = nodes
        .iter()
        .map(|(component, node)| {
            let facts = node
                .static_facts
                .as_ref()
                .expect("static facts established");
            match (&node.lookup_fields, &node.relation_columns) {
                (Some(fields), Some(columns)) => plan_standard_component(
                    component,
                    facts.row_source,
                    fields,
                    columns,
                    &declared_relation_ids,
                ),
                (None, None) => plan_computed_component(
                    component,
                    facts.row_source,
                    &declared_relation_ids,
                    custom_geometry,
                ),
                _ => panic!("{component}: partial relation metadata"),
            }
        })
        .collect();
    assert_eq!(planned_components.len(), claim_components.len());

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
         use crate::schedule::{\n    CapacityFeed, ComponentNode, ComponentRowSource, ComponentStaticFacts, CountFeed, InputEdge,\n    KernelIdentitySource, LogSizeSource, OutputEdge, Schedule, TraceColumnCount,\n};\n\n\
         pub static CAIRO_SCHEDULE: Schedule = Schedule { nodes: NODES };\n\n",
    );
    s.push_str("static NODES: &[ComponentNode] = &[\n");
    for (id, node) in &nodes {
        let facts = node.static_facts.as_ref().unwrap();
        s.push_str("    ComponentNode {\n");
        s.push_str(&format!("        id: \"{id}\",\n"));
        s.push_str("        facts: ComponentStaticFacts {\n");
        s.push_str(&format!(
            "            trace_columns: {},\n",
            emit_trace_columns(id)
        ));
        s.push_str(&format!(
            "            lookup_words: {},\n",
            option_u32(facts.lookup_words)
        ));
        s.push_str(&format!(
            "            sub_words: {},\n",
            option_u32(facts.sub_words)
        ));
        s.push_str(&format!(
            "            logup_columns: {},\n",
            option_u32(facts.logup_columns)
        ));
        s.push_str(&format!(
            "            row_source: {},\n",
            emit_row_source(id, facts.row_source)
        ));
        s.push_str(&format!(
            "            kernel_identity: KernelIdentitySource::{},\n",
            if facts.recorded_witness_kernel {
                "RecordedWitness"
            } else {
                "None"
            }
        ));
        s.push_str("        },\n");
        s.push_str("        kernel: None,\n");
        if matches!(facts.row_source, RowSource::FixedLogSize(_)) {
            s.push_str(&format!(
                "        log_size: LogSizeSource::Fixed(cairo_air::components::{id}::LOG_SIZE),\n"
            ));
        } else {
            s.push_str("        log_size: LogSizeSource::FromStates,\n");
        }
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
        if node.capacity_inputs.is_empty() {
            s.push_str("        capacity_inputs: &[],\n");
        } else {
            s.push_str("        capacity_inputs: &[\n");
            for (from, n_instances) in &node.capacity_inputs {
                s.push_str(&format!(
                    "            CapacityFeed {{ from: \"{from}\", n_instances: {n_instances} }},\n"
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
    let s = match format_rust(&root, &s) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("schedule_emit: {e}");
            return ExitCode::FAILURE;
        }
    };
    let proof_shape_source = match format_rust(&root, &emit_proof_shape_source(&nodes)) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("schedule_emit proof_shape: {e}");
            return ExitCode::FAILURE;
        }
    };
    let relation_source = match format_rust(&root, &emit_relation_source(&planned_components)) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("schedule_emit relation_graph: {e}");
            return ExitCode::FAILURE;
        }
    };

    if check {
        let on_disk = std::fs::read_to_string(&out_path).unwrap_or_default();
        let relation_on_disk = std::fs::read_to_string(&relation_out).unwrap_or_default();
        let proof_shape_on_disk = std::fs::read_to_string(&proof_shape_out).unwrap_or_default();
        if on_disk == s
            && relation_on_disk == relation_source
            && proof_shape_on_disk == proof_shape_source
        {
            println!("schedule_emit --check: OK (no drift)");
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "schedule_emit --check: DRIFT — regenerate {}, {}, and {} (component metadata changed)",
                out_path.display(),
                relation_out.display(),
                proof_shape_out.display(),
            );
            ExitCode::FAILURE
        }
    } else {
        std::fs::write(&out_path, &s).expect("write schedule_table.rs");
        std::fs::write(&relation_out, &relation_source).expect("write relation_table.rs");
        std::fs::write(&proof_shape_out, &proof_shape_source)
            .expect("write proof_shape_generated.rs");
        println!(
            "schedule_emit: wrote {}, {}, and {} ({} nodes)",
            out_path.display(),
            relation_out.display(),
            proof_shape_out.display(),
            nodes.len()
        );
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_rust_is_canonical() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = "use crate::schedule::{\n    ComponentNode, CountFeed, InputEdge, LogSizeSource, OutputEdge, Schedule,\n};\n";
        assert_eq!(
            format_rust(&root, source).unwrap(),
            "use crate::schedule::{ComponentNode, CountFeed, InputEdge, LogSizeSource, OutputEdge, Schedule};\n"
        );
    }
}
