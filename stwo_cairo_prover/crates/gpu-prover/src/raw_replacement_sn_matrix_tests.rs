//! Sealed SN1-SN4 host-cache and executable-handle admission matrix.
//!
//! This is intentionally distinct from the generator differential oracle. It
//! proves the replacement production seam itself: a second proof of an exact
//! shape reuses immutable planning and executable topology while rebinding all
//! current statement and execution-memory values.

#[path = "raw_replacement_sn_matrix_tests/composition_slab.rs"]
mod composition_slab;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cairo_air::claims::CairoClaim;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    direct_terminal_expand_absorb_arena_slot_requirements, quotient_numerator_run_sum_plan,
    DirectCompactTerminalBatchMode, DirectCompactTerminalProgram,
    DirectTerminalExpandAbsorbProgram, FusedCompactDomainProgram,
    ModeAwareCommitWorkspaceRequirements, ModeAwareCommitWorkspaceSlots,
    PreparedProgressiveCommitError, ProgressiveCommitStorageMode, QuotientNumeratorRunSumLiveness,
};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::{
    PreProcessedTrace, PreProcessedTraceVariant,
};

use crate::arena_plan::{
    BlakeGWitnessContract, BufferLifetime, BufferPurpose, CommitmentTreeId,
    DirectCompactTerminalPlan, ExecutionTableGeometry, ProofArenaPlan, ProofEpoch, ResidentBackend,
};
use crate::phases;
use crate::plan::ProofPlan;
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::prover::prepare_resident_ingest;
use crate::raw_replacement_oracle_tests::assert_cached_recorded_matches_fresh;
use crate::recorded_witness_inputs::{
    recorded_witness_inputs_for_raw_replacement_plan, DeviceGatherColumn,
    PlannedRecordedWitnessInputs, RecordedInputColumnProvenance,
};
use crate::relation_execution::RelationSourcePlane;
use crate::relation_table::CAIRO_RELATION_GRAPH;
use crate::replacement_host_cache::{
    ReplacementHostCache, ReplacementHostMaterialization, ReplacementHostTemplate,
};
use crate::resident_input::ResidentProverInputOwner;
use crate::resident_session::{
    direct_blake_g_route_is_exact_for_test, plan_raw_resident_preflight,
    resident_host_witness_input_route_shapes_for_test, ResidentPreWitnessInput,
    ResidentSessionError,
};
use crate::resident_shape::raw_replacement_proof_plan;
use crate::resident_sources::{
    preprocessed_commit_binding, PreprocessedCommitBinding, PreprocessedCommitSelector,
    PreprocessedCommitSlotMode,
};
use crate::resident_witness::planned_cairo_claim_from_public_data;
use crate::schedule_table::CAIRO_SCHEDULE;
use crate::shape_executable::{ShapeExecutableCache, ShapeExecutableMaterialization};

struct SealedSnFixture {
    profile: &'static str,
    file: &'static str,
    bytes: usize,
    sha256: &'static str,
    blake3: &'static str,
}

const SEALED_SN_FIXTURES: [SealedSnFixture; 4] = [
    SealedSnFixture {
        profile: "SN1",
        file: "SN_PIE_1.adapted.bin",
        bytes: 297_469_956,
        sha256: "6506b4f1a871c9f2f83d4a63c5bd288a44b60199d9e124cfcf55bf8154da9fb9",
        blake3: "0bd37b723f24b482188e69807183d82da139971a3f6f6c7522b36fec51210e36",
    },
    SealedSnFixture {
        profile: "SN2",
        file: "SN_PIE_2.adapted.bin",
        bytes: 162_102_412,
        sha256: "78b0995483a76e850c61cf7cb51861850f746ddf927344088014492b6752844c",
        blake3: "5375bd23b012fad243678af013db10498e137642c2cb273e1a1314306aa44b0d",
    },
    SealedSnFixture {
        profile: "SN3",
        file: "SN_PIE_3.adapted.bin",
        bytes: 285_299_888,
        sha256: "348dbbc4673dc9d040df3bc29805d50f583d000d6df19bd6647c11c5711c9ce5",
        blake3: "b4c70e2b46abcf81ae15047dc4d432d1526c41e6341f1c796c14c632404fa2b4",
    },
    SealedSnFixture {
        profile: "SN4",
        file: "SN_PIE_4.adapted.bin",
        bytes: 284_530_524,
        sha256: "c6c134c098f2f80cb2b99629922caf52e5917b71f1f813d23d20bb8109c5543f",
        blake3: "a750d3380159125b9a57165651009fc923b0ef351aed9203c8eaf16f0b7007ac",
    },
];

fn assert_complete_plan_eq(case: &str, cached: &ProofPlan, fresh: &ProofPlan) {
    assert_eq!(cached.shape_key, fresh.shape_key, "{case}: shape key");
    assert_eq!(
        cached.relation_graph_hash, fresh.relation_graph_hash,
        "{case}: relation graph hash"
    );
    assert_eq!(
        cached.proof_shape(),
        fresh.proof_shape(),
        "{case}: proof shape"
    );
    assert_eq!(
        cached.components.len(),
        fresh.components.len(),
        "{case}: component count"
    );
    for (cached, fresh) in cached.components.iter().zip(&fresh.components) {
        assert!(
            std::ptr::eq(cached.node, fresh.node),
            "{case}/{}: static schedule node",
            cached.node.id
        );
        assert_eq!(
            cached.runtime, fresh.runtime,
            "{case}/{}: runtime component shape",
            cached.node.id
        );
    }
}

fn fresh_plans(
    owner: &ResidentProverInputOwner,
    preprocessed: &Arc<PreProcessedTrace>,
) -> (ProofPlan, ProofPlan) {
    let capacity = raw_replacement_proof_plan(owner, Arc::clone(preprocessed), None)
        .expect("fresh raw replacement capacity plan");
    let exact = capacity
        .strict_resident_exact(&CAIRO_SCHEDULE, &CAIRO_RELATION_GRAPH)
        .expect("fresh raw replacement exact plan");
    (capacity, exact)
}

fn execution_geometry(
    owner: &ResidentProverInputOwner,
    claim: &CairoClaim,
) -> ExecutionTableGeometry {
    let public_memory_entries = claim
        .public_data
        .public_memory
        .get_entries(
            claim.public_data.initial_state.pc.0,
            claim.public_data.initial_state.ap.0,
            claim.public_data.final_state.ap.0,
        )
        .count();
    ExecutionTableGeometry::new(
        owner.execution_memory().address_to_id.len(),
        owner.execution_memory().f252_values.len(),
        owner.execution_memory().small_values.len(),
    )
    .with_public_memory_entries(public_memory_entries)
}

fn assert_sn_terminal_policy(profile: &str, arena: &ProofArenaPlan) {
    let (expected, expected_fused_materialized_rises, expected_fixed16_sinks) = match profile {
        "SN1" => (
            [
                (CommitmentTreeId::Base, 7, 12, 16_922_247_168, -7),
                (CommitmentTreeId::Interaction, 9, 10, 8_269_332_480, 1),
            ],
            20,
            16,
        ),
        "SN2" => (
            [
                (CommitmentTreeId::Base, 7, 11, 6_820_986_880, -5),
                (CommitmentTreeId::Interaction, 8, 10, 5_606_735_872, -2),
            ],
            19,
            15,
        ),
        "SN3" => (
            [
                (CommitmentTreeId::Base, 8, 10, 9_969_139_712, -6),
                (CommitmentTreeId::Interaction, 8, 10, 7_235_436_544, -6),
            ],
            18,
            16,
        ),
        "SN4" => (
            [
                (CommitmentTreeId::Base, 9, 11, 10_019_995_648, -3),
                (CommitmentTreeId::Interaction, 7, 13, 4_521_197_568, -5),
            ],
            22,
            16,
        ),
        _ => panic!("unknown sealed SN profile {profile}"),
    };
    let mut fused_materialized_rises = 0;
    let mut fixed16_sinks = 0;
    for (tree, fixed16_batches, materialized_batches, net_device_bytes, net_cuda_launches) in
        expected
    {
        let commitment = arena
            .commitment(tree)
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: commitment missing"));
        let selection = commitment
            .direct_compact_terminal
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: terminal plan missing"));
        assert_eq!(
            selection,
            &DirectCompactTerminalPlan::Materialized {
                batches: fixed16_batches + materialized_batches,
            },
            "{profile}/{tree:?}: uncredited Fixed16 candidate reached production"
        );
        assert!(
            selection.receipt().is_none(),
            "{profile}/{tree:?}: materialized selection exposed a fusion receipt"
        );
        assert!(
            commitment.direct_terminal_expand_absorb.is_none(),
            "{profile}/{tree:?}: uncredited terminal successor reached production"
        );

        // Preserve exact compiler and workspace coverage for the dormant A/B
        // candidate without allocating its larger slab in production.
        let base = commitment
            .commit_program
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: base program missing"));
        let domain = commitment
            .domain_cooperative_program
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: domain program missing"));
        let compact = commitment
            .compact_domain_program
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: compact program missing"));
        let direct = commitment
            .direct_retained_b2n_program
            .as_ref()
            .unwrap_or_else(|| panic!("{profile}/{tree:?}: direct program missing"));
        let terminal = DirectCompactTerminalProgram::compile(compact, direct)
            .unwrap_or_else(|error| panic!("{profile}/{tree:?}: terminal candidate: {error}"));
        let terminal_receipt = terminal.receipt();
        assert_eq!(
            terminal_receipt.fixed_terminal_launches, fixed16_batches,
            "{profile}/{tree:?}: exact dormant fixed16 batches"
        );
        assert_eq!(
            terminal_receipt
                .batches
                .iter()
                .filter(|batch| batch.mode == DirectCompactTerminalBatchMode::Materialized)
                .count() as u32,
            materialized_batches,
            "{profile}/{tree:?}: exact dormant materialized batches"
        );
        assert_eq!(
            terminal_receipt.net_device_bytes_removed, net_device_bytes,
            "{profile}/{tree:?}: exact dormant traffic model"
        );
        assert_eq!(
            terminal_receipt.net_cuda_launches_removed, net_cuda_launches,
            "{profile}/{tree:?}: exact dormant launch model"
        );
        assert!(!terminal_receipt.same_gpu_timing_credit_applied);

        let fused_compact = FusedCompactDomainProgram::compile(base, domain, compact)
            .unwrap_or_else(|error| panic!("{profile}/{tree:?}: fused compact candidate: {error}"));
        let successor = DirectTerminalExpandAbsorbProgram::compile(
            base,
            domain,
            compact,
            &fused_compact,
            direct,
            &terminal,
        )
        .unwrap_or_else(|error| panic!("{profile}/{tree:?}: successor candidate: {error}"));
        let successor_receipt = successor.receipt();
        let qualified = successor_receipt.qualified_slab_capacity_words;
        assert_eq!(
            qualified,
            domain.slab_words(),
            "{profile}/{tree:?}: exact dormant qualified slab"
        );
        assert!(
            qualified > compact.slab_words(),
            "{profile}/{tree:?}: dormant successor must not fit the production compact slab"
        );

        let ModeAwareCommitWorkspaceSlots::DomainProgressive(slots) = &commitment.slots else {
            panic!("{profile}/{tree:?}: dormant successor requires progressive slots");
        };
        let workspace = direct_terminal_expand_absorb_arena_slot_requirements(
            &successor,
            base,
            domain,
            compact,
            &fused_compact,
            direct,
            &terminal,
            slots,
        )
        .unwrap_or_else(|error| panic!("{profile}/{tree:?}: dormant successor slots: {error}"));
        let required_slab = workspace
            .iter()
            .find(|requirement| requirement.id == slots.leaves.state_ping)
            .unwrap_or_else(|| {
                panic!("{profile}/{tree:?}: dormant successor slab requirement missing")
            });
        assert_eq!(required_slab.len_words, qualified);
        assert!(
            arena
                .logical_buffers()
                .iter()
                .filter(|buffer| buffer.purpose == BufferPurpose::CommitProgressiveStatePing)
                .any(|buffer| {
                    buffer.len_words == compact.slab_words()
                        && arena.binding(buffer.id).is_some_and(|binding| {
                            binding.physical == slots.leaves.state_ping
                                && binding.len_words == compact.slab_words()
                        })
                }),
            "{profile}/{tree:?}: exact production compact slab owner missing"
        );

        fused_materialized_rises += successor_receipt.fused_materialized_rises;
        fixed16_sinks += successor_receipt.fixed16_batches;
        assert!(successor_receipt.fixed16_terminal_receipt_unchanged);
        assert!(successor_receipt.final_state_at_slab_zero);
        assert!(!successor_receipt.same_gpu_timing_credit_applied);
    }
    assert_eq!(
        fused_materialized_rises, expected_fused_materialized_rises,
        "{profile}: exact dormant materialized expand-absorb rises"
    );
    assert_eq!(
        fixed16_sinks, expected_fixed16_sinks,
        "{profile}: exact dormant fixed16 terminal sinks"
    );
}

fn assert_sn_blake_g_direct(profile: &str, arena: &ProofArenaPlan) {
    let direct = arena
        .witness()
        .components
        .iter()
        .filter_map(|component| {
            component
                .blake_g_contract
                .direct()
                .map(|selection| (component, selection))
        })
        .collect::<Vec<_>>();
    let [(component, selection)] = direct.as_slice() else {
        panic!("{profile}: expected exactly one direct Blake-G witness contract")
    };
    assert_eq!(component.component, "blake_g");
    assert_eq!(selection.input_ordinals, [0, 1, 2, 3, 4, 5]);
    assert_eq!(selection.enabler_ordinal, 6);
    assert_eq!(selection.retired_lookup_words_per_row, 87);
    assert_eq!(arena.relation().blake_g_inputs.as_ref(), Some(*selection));
    assert!(matches!(
        component.blake_g_contract,
        BlakeGWitnessContract::Direct(_)
    ));

    let gather = component
        .input_gather
        .as_ref()
        .unwrap_or_else(|| panic!("{profile}: direct Blake-G gather missing"));
    assert_eq!(component.requirements.input_column_words.len(), 7);
    assert_eq!(component.slots.input_columns.len(), 6);
    assert_eq!(gather.requirements.input_width, 6);
    assert!(!gather.requirements.include_enabler);
    assert!(!gather.requirements.include_iota);
    assert_eq!(gather.requirements.consumer_input_column_words.len(), 6);
    assert_eq!(
        gather.slots.consumer_input_columns,
        component.slots.input_columns
    );

    for (&ordinal, &slot) in selection
        .input_ordinals
        .iter()
        .zip(&component.slots.input_columns)
    {
        let (input, binding) = arena
            .find(
                Some("blake_g"),
                Some(selection.part),
                BufferPurpose::WitnessInput,
                ordinal,
            )
            .unwrap_or_else(|| panic!("{profile}: direct Blake-G input {ordinal} missing"));
        assert_eq!(binding.physical, slot);
        assert_eq!(input.len_words, component.requirements.row_count);
        assert_eq!(
            input.lifetime,
            BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Interaction).unwrap()
        );
    }
    assert!(
        arena
            .find(
                Some("blake_g"),
                Some(selection.part),
                BufferPurpose::WitnessInput,
                selection.enabler_ordinal,
            )
            .is_none(),
        "{profile}: retired Blake-G enabler must have no arena owner"
    );
    assert!(
        arena
            .find(
                Some("blake_g"),
                Some(selection.part),
                BufferPurpose::LookupInputs,
                0,
            )
            .is_none(),
        "{profile}: retired Blake-G lookup slab must have no arena owner"
    );
    assert_eq!(
        component.slots.multiplicity_dummy,
        Some(component.slots.lookup_words)
    );
    assert_eq!(component.slots.lookup_words, component.slots.sub_words);

    let (input_pointers, _) = arena
        .find(
            Some("blake_g"),
            Some(selection.part),
            BufferPurpose::WitnessInputPointers,
            0,
        )
        .unwrap();
    let words_per_pointer = component.requirements.input_pointer_words
        / component.requirements.input_column_words.len();
    assert_eq!(
        input_pointers.len_words,
        words_per_pointer * selection.input_ordinals.len()
    );

    let relation_source = arena
        .relation()
        .source_plan
        .iter()
        .filter(|source| source.plane == RelationSourcePlane::WitnessInput)
        .collect::<Vec<_>>();
    assert_eq!(relation_source.len(), 1);
    assert_eq!(relation_source[0].batch, selection.batch);
    assert_eq!(relation_source[0].part, selection.part);
    assert_eq!(relation_source[0].instance_index, selection.instance_index);
    assert_eq!(relation_source[0].column_count, 6);

    let base_aliases = (0..53)
        .filter(|&ordinal| {
            let (_, evaluations) = arena
                .find(
                    Some("blake_g"),
                    Some(selection.part),
                    BufferPurpose::BaseTrace,
                    ordinal,
                )
                .unwrap();
            let (_, coefficients) = arena
                .find(
                    Some("blake_g"),
                    Some(selection.part),
                    BufferPurpose::BaseCoefficients,
                    ordinal,
                )
                .unwrap();
            evaluations.physical == coefficients.physical
        })
        .count();
    assert_eq!(base_aliases, 53, "{profile}: Blake-G base alias count");

    let rows = component.requirements.row_count;
    let retired_lookup_words = rows * selection.retired_lookup_words_per_row as usize;
    let retired_enabler_words = rows;
    let retained_operand_words = rows * selection.input_ordinals.len();
    let peak_words = ProofEpoch::ALL
        .into_iter()
        .map(|epoch| arena.high_water_words(epoch))
        .max()
        .unwrap();
    eprintln!(
        "SN_BLAKE_G_DIRECT_RECEIPT {}",
        serde_json::json!({
            "schema": "stwo.sn-blake-g-direct.v1",
            "profile": profile,
            "rows": rows,
            "recorded_inputs": component.requirements.input_column_words.len(),
            "materialized_inputs": component.slots.input_columns.len(),
            "retired_enabler_ordinal": selection.enabler_ordinal,
            "retired_lookup_words": retired_lookup_words,
            "retired_enabler_words": retired_enabler_words,
            "retained_operand_words": retained_operand_words,
            "input_pointer_words": input_pointers.len_words,
            "shared_tail_dummy_words": 1,
            "base_trace_coefficient_aliases": base_aliases,
            "arena_total_words": arena.total_words(),
            "arena_peak_words": peak_words,
        })
    );
}

fn assert_current_bindings_match_fresh(
    case: &str,
    template: &ReplacementHostTemplate,
    owner: &ResidentProverInputOwner,
    preprocessed: &Arc<PreProcessedTrace>,
) -> CairoClaim {
    let (fresh_capacity, fresh_exact) = fresh_plans(owner, preprocessed);
    assert_complete_plan_eq(
        &format!("{case}/capacity"),
        template.capacity_plan(),
        &fresh_capacity,
    );
    assert_complete_plan_eq(
        &format!("{case}/exact"),
        template.exact_plan(),
        &fresh_exact,
    );

    let cached_claim = template.bind_claim(owner.public_data());
    let fresh_claim = planned_cairo_claim_from_public_data(owner.public_data(), &fresh_exact)
        .expect("fresh planned claim");
    assert_eq!(
        serde_json::to_value(&cached_claim).unwrap(),
        serde_json::to_value(&fresh_claim).unwrap(),
        "{case}: complete current claim"
    );

    let cached_recorded = template
        .bind_recorded(owner)
        .expect("cached recorded witness binding");
    let fresh_recorded = recorded_witness_inputs_for_raw_replacement_plan(owner, &fresh_exact)
        .expect("fresh recorded witness plan");
    cached_recorded
        .require_resolved()
        .unwrap_or_else(|error| panic!("{case}: cached recorded witness unresolved: {error}"));
    fresh_recorded
        .require_resolved()
        .unwrap_or_else(|error| panic!("{case}: fresh recorded witness unresolved: {error}"));
    assert_cached_recorded_matches_fresh(&cached_recorded, &fresh_recorded);
    cached_claim
}

/// Point the public bitwise-start stack cell at a different existing small
/// value. Table lengths and builtin extents stay fixed, while the executable's
/// only proof-varying parameter class is forced to bind a current value.
fn rebind_bitwise_public_start(input: &mut ProverInput, profile: &str) {
    const BITWISE_PUBLIC_SEGMENT: usize = 4;
    assert!(
        input.public_segment_context[BITWISE_PUBLIC_SEGMENT],
        "{profile}: sealed fixture must expose bitwise as a public segment"
    );
    let preceding_present = input.public_segment_context[..BITWISE_PUBLIC_SEGMENT]
        .iter()
        .filter(|present| **present)
        .count();
    let output_pointer_address = input.state_transitions.initial_state.ap.0 as usize;
    let pointer_address = output_pointer_address + preceding_present;
    let original = input.memory.get(pointer_address as u32).as_small();
    let replacement = input.memory.get(output_pointer_address as u32).as_small();
    assert_ne!(
        original, replacement,
        "{profile}: output and bitwise starts must differ"
    );
    input.memory.address_to_id[pointer_address] =
        input.memory.address_to_id[output_pointer_address];
    assert_eq!(
        input.memory.get(pointer_address as u32).as_small(),
        replacement
    );
}

fn admit_fixture_identities(directory: &Path) {
    let mut observations = Vec::with_capacity(SEALED_SN_FIXTURES.len());
    let mut mismatches = Vec::new();
    for fixture in &SEALED_SN_FIXTURES {
        let path = directory.join(fixture.file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read sealed {}: {error}", path.display()));
        let actual_bytes = bytes.len();
        let actual_blake3 = blake3::hash(&bytes).to_hex().to_string();
        drop(bytes);
        if actual_bytes != fixture.bytes || actual_blake3 != fixture.blake3 {
            mismatches.push(format!(
                "{} expected bytes={} blake3={} observed bytes={} blake3={}",
                fixture.profile, fixture.bytes, fixture.blake3, actual_bytes, actual_blake3,
            ));
        }
        observations.push(serde_json::json!({
            "profile": fixture.profile,
            "file": fixture.file,
            "expected_bytes": fixture.bytes,
            "observed_bytes": actual_bytes,
            "externally_admitted_sha256": fixture.sha256,
            "expected_blake3": fixture.blake3,
            "observed_blake3": actual_blake3,
        }));
    }
    eprintln!(
        "SN_FIXTURE_IDENTITY {}",
        serde_json::to_string(&observations).unwrap()
    );
    assert!(
        mismatches.is_empty(),
        "sealed SN fixture identity mismatch:\n{}",
        mismatches.join("\n")
    );
}

fn assert_adaptive_run_sum_receipt(
    arena: &ProofArenaPlan,
    expected_target: usize,
    expected_victim: usize,
    expected_runs: u32,
    expected_scratch_words: usize,
    expected_margin_words: usize,
) {
    let workspace = arena.quotient_numerator();
    assert_eq!(
        workspace.schedule,
        crate::arena_plan::QuotientNumeratorSchedule::StagedRunSumOrPacked
    );
    let staged = workspace
        .staged_single_write
        .as_ref()
        .expect("adaptive numerator schedule must retain the staged plan");
    let mut selected = None;
    'selection: for target in 0..staged.requirements().groups.len() {
        for victim in target + 1..staged.requirements().groups.len() {
            let capacities = workspace.destinations[victim].coordinates.map(|binding| {
                binding
                    .len_words
                    .min(staged.requirements().groups[victim].value_words)
            });
            let Ok(receipt) = quotient_numerator_run_sum_plan(
                staged,
                target,
                victim,
                QuotientNumeratorRunSumLiveness {
                    same_stream_canonical_group_order: true,
                    external_destination_ids_unique: true,
                    victim_unread_before_own_producer: true,
                    victim_fully_overwritten_by_own_producer: true,
                    downstream_consumers_after_all_group_producers: true,
                    victim_coordinate_capacity_words: capacities,
                },
            ) else {
                continue;
            };
            if receipt.add_units_saved != 0 {
                selected = Some(receipt);
                break 'selection;
            }
        }
    }
    let receipt = selected.expect("exact SN shape must admit a run-sum binding");
    assert_eq!(
        (receipt.target_group, receipt.victim_group),
        (expected_target, expected_victim)
    );
    assert_eq!(receipt.manifest.run_count, expected_runs);
    assert_eq!(receipt.scratch_words_per_coordinate, expected_scratch_words);
    assert_eq!(
        receipt.margin_words_per_coordinate,
        [expected_margin_words; 4]
    );
}

fn assert_fixture_adaptive_run_sum(
    directory: &Path,
    fixture: &SealedSnFixture,
    expected_victim: usize,
    expected_runs: u32,
    expected_scratch_words: usize,
    expected_margin_words: usize,
) {
    let path = directory.join(fixture.file);
    let encoded = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read sealed {}: {error}", path.display()));
    assert_eq!(
        encoded.len(),
        fixture.bytes,
        "{}: byte length",
        fixture.profile
    );
    assert_eq!(
        blake3::hash(&encoded).to_hex().as_str(),
        fixture.blake3,
        "{}: sealed BLAKE3",
        fixture.profile
    );
    let input: ProverInput = bincode::deserialize(&encoded)
        .unwrap_or_else(|error| panic!("decode sealed {}: {error}", path.display()));
    drop(encoded);
    let ingest = phases::ingest::run_replacement(input, PreProcessedTraceVariant::Canonical, None)
        .unwrap_or_else(|error| panic!("{} replacement ingest: {error}", fixture.profile));
    let report = plan_raw_resident_preflight(
        &ingest.input,
        &ingest.proof_plan,
        &ingest.preprocessed_trace,
        PcsConfig::default(),
        false,
    )
    .unwrap_or_else(|error| panic!("{} replacement preflight: {error:?}", fixture.profile));
    assert_adaptive_run_sum_receipt(
        &report.arena,
        0,
        expected_victim,
        expected_runs,
        expected_scratch_words,
        expected_margin_words,
    );
}

fn run_profile(directory: &Path, fixture: &SealedSnFixture) {
    let path = directory.join(fixture.file);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read sealed {}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        fixture.bytes,
        "{}: byte length",
        fixture.profile
    );
    let actual_blake3 = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        actual_blake3, fixture.blake3,
        "{}: sealed fixture BLAKE3; update only after external SHA-256 admission",
        fixture.profile
    );
    let cold_input: ProverInput = bincode::deserialize(&bytes)
        .unwrap_or_else(|error| panic!("decode sealed {}: {error}", path.display()));
    drop(bytes);
    let mut warm_input = cold_input.clone();
    warm_input.state_transitions.final_state.fp.0 ^= 1;
    rebind_bitwise_public_start(&mut warm_input, fixture.profile);

    let mut host_cache = ReplacementHostCache::new(1).unwrap();
    let cold_started = Instant::now();
    let cold = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut host_cache),
        cold_input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let cold_host_wall_ns = cold_started.elapsed().as_nanos();
    let cold_audit = cold
        .audit
        .replacement_host_cache
        .expect("replacement host-cache audit");
    assert_eq!(
        cold_audit.materialization,
        ReplacementHostMaterialization::Compiled
    );
    assert_eq!(cold.audit.claim_generator_constructions, 0);
    let cold_preprocessed = Arc::clone(&cold.preprocessed_trace);
    let ResidentPreWitnessInput::ReplacementV1 {
        input: cold_owner,
        template: cold_template,
    } = cold.input
    else {
        panic!("cold replacement dispatch returned legacy input")
    };

    let warm_started = Instant::now();
    let warm = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut host_cache),
        warm_input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let warm_host_wall_ns = warm_started.elapsed().as_nanos();
    let warm_audit = warm
        .audit
        .replacement_host_cache
        .expect("replacement host-cache audit");
    assert_eq!(
        warm_audit.materialization,
        ReplacementHostMaterialization::Reused
    );
    assert_eq!(
        (warm_audit.telemetry.hits, warm_audit.telemetry.misses),
        (1, 1)
    );
    assert_eq!(warm_audit.telemetry.compilations, 1);
    assert_eq!(warm.audit.claim_generator_constructions, 0);
    assert!(Arc::ptr_eq(&cold_preprocessed, &warm.preprocessed_trace));
    let ResidentPreWitnessInput::ReplacementV1 {
        input: warm_owner,
        template: warm_template,
    } = warm.input
    else {
        panic!("warm replacement dispatch returned legacy input")
    };
    assert!(Arc::ptr_eq(&cold_template, &warm_template));
    assert_ne!(
        cold_owner.execution_memory().address_to_id.as_ptr(),
        warm_owner.execution_memory().address_to_id.as_ptr(),
        "{}: current execution-memory allocation",
        fixture.profile
    );

    let cold_claim = assert_current_bindings_match_fresh(
        &format!("{}/cold", fixture.profile),
        &cold_template,
        &cold_owner,
        &cold_preprocessed,
    );
    let warm_claim = assert_current_bindings_match_fresh(
        &format!("{}/warm", fixture.profile),
        &warm_template,
        &warm_owner,
        &cold_preprocessed,
    );
    assert_ne!(
        serde_json::to_value(&cold_claim).unwrap(),
        serde_json::to_value(&warm_claim).unwrap(),
        "{}: current statement must be rebound",
        fixture.profile
    );
    let cold_recorded = cold_template.bind_recorded(&cold_owner).unwrap();
    let warm_recorded = warm_template.bind_recorded(&warm_owner).unwrap();

    let cold_geometry = execution_geometry(&cold_owner, &cold_claim);
    let warm_geometry = execution_geometry(&warm_owner, &warm_claim);
    assert_eq!(cold_geometry, warm_geometry);
    let pcs = PcsConfig::default();
    let policy = ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048);
    let mut shape_cache = ShapeExecutableCache::new(1).unwrap();
    let cold_shape_started = Instant::now();
    let cold_shape = cold_template
        .select_shape_executable(
            &mut shape_cache,
            &cold_claim,
            &cold_preprocessed,
            pcs,
            false,
            Some(cold_geometry),
            policy,
        )
        .unwrap();
    let cold_shape_wall_ns = cold_shape_started.elapsed().as_nanos();
    let warm_shape_started = Instant::now();
    let warm_shape = warm_template
        .select_shape_executable(
            &mut shape_cache,
            &warm_claim,
            &cold_preprocessed,
            pcs,
            false,
            Some(warm_geometry),
            policy,
        )
        .unwrap();
    let warm_shape_wall_ns = warm_shape_started.elapsed().as_nanos();
    assert_eq!(
        cold_shape.materialization,
        ShapeExecutableMaterialization::Compiled
    );
    assert_eq!(
        warm_shape.materialization,
        ShapeExecutableMaterialization::Reused
    );
    assert!(Arc::ptr_eq(&cold_shape.executable, &warm_shape.executable));
    assert_sn_terminal_policy(fixture.profile, cold_shape.executable.arena());
    assert_sn_blake_g_direct(fixture.profile, cold_shape.executable.arena());
    assert_ne!(
        cold_shape.bindings, warm_shape.bindings,
        "{}: warm handle must bind current statement values",
        fixture.profile
    );
    let shape_telemetry = shape_cache.telemetry();
    assert_eq!(shape_telemetry.topology_key_constructions, 1);
    assert_eq!((shape_telemetry.hits, shape_telemetry.misses), (1, 1));
    assert_eq!(shape_telemetry.compilations, 1);
    assert_eq!(shape_telemetry.source_generation_passes, 1);
    assert_eq!(shape_telemetry.binding_recipe_compilations, 1);
    assert_eq!(shape_telemetry.replacement_handle_lock_ops, 3);

    let arena = cold_shape.executable.arena();
    let blake_index = arena
        .witness()
        .components
        .iter()
        .position(|component| component.component == "blake_g")
        .unwrap_or_else(|| panic!("{}: Blake-G witness missing", fixture.profile));
    let blake_component = &arena.witness().components[blake_index];
    let selection = blake_component
        .blake_g_contract
        .direct()
        .unwrap_or_else(|| panic!("{}: direct Blake-G missing", fixture.profile));
    let cold_session_input = ResidentPreWitnessInput::ReplacementV1 {
        input: cold_owner,
        template: cold_template,
    };
    let warm_session_input = ResidentPreWitnessInput::ReplacementV1 {
        input: warm_owner,
        template: warm_template,
    };
    let cold_routes = resident_host_witness_input_route_shapes_for_test(
        &cold_recorded,
        arena,
        &cold_session_input,
    )
    .unwrap_or_else(|error| panic!("{}/cold route: {error}", fixture.profile));
    let warm_routes = resident_host_witness_input_route_shapes_for_test(
        &warm_recorded,
        arena,
        &warm_session_input,
    )
    .unwrap_or_else(|error| panic!("{}/warm route: {error}", fixture.profile));
    assert_eq!(cold_routes[blake_index], (0, 0, false));
    assert_eq!(warm_routes[blake_index], (0, 0, false));
    assert!(direct_blake_g_route_is_exact_for_test(
        &cold_recorded.lanes[blake_index],
        blake_component,
    ));
    let enabler = selection.enabler_ordinal as usize;
    let (blake_n_real, blake_row_count) = match &cold_recorded.lanes[blake_index].columns[enabler] {
        RecordedInputColumnProvenance::RetiredBlakeGEnabler { n_real, row_count } => {
            (*n_real, *row_count)
        }
        provenance => panic!(
            "{}: unexpected Blake-G enabler: {provenance:?}",
            fixture.profile
        ),
    };
    assert_eq!(
        (blake_n_real, blake_row_count),
        (
            cold_recorded.lanes[blake_index].n_real,
            cold_recorded.lanes[blake_index].row_count,
        )
    );
    let retired_blake_g_enabler_host_payload_bytes = blake_row_count
        .checked_mul(core::mem::size_of::<u32>())
        .expect("Blake-G enabler payload bytes");

    if fixture.profile == "SN2" {
        let assert_rejected = |recorded: &PlannedRecordedWitnessInputs, expected_ordinal: usize| {
            assert!(matches!(
                resident_host_witness_input_route_shapes_for_test(
                    recorded,
                    arena,
                    &cold_session_input,
                ),
                Err(ResidentSessionError::RecordedWitnessInputRoute {
                    component: "blake_g",
                    ordinal,
                }) if ordinal == expected_ordinal
            ));
        };
        assert!(blake_n_real > 1 && blake_n_real <= blake_row_count);

        let mut wrong_real = cold_recorded.clone();
        wrong_real.lanes[blake_index].columns[enabler] =
            RecordedInputColumnProvenance::RetiredBlakeGEnabler {
                n_real: blake_n_real - 1,
                row_count: blake_row_count,
            };
        assert_rejected(&wrong_real, enabler);

        let mut wrong_rows = cold_recorded.clone();
        wrong_rows.lanes[blake_index].columns[enabler] =
            RecordedInputColumnProvenance::RetiredBlakeGEnabler {
                n_real: blake_n_real,
                row_count: blake_row_count - 1,
            };
        assert_rejected(&wrong_rows, enabler);

        let partial_n_real = blake_n_real - 1;
        let mut partial_lane = cold_recorded.lanes[blake_index].clone();
        partial_lane.n_real = partial_n_real;
        partial_lane.columns[enabler] = RecordedInputColumnProvenance::RetiredBlakeGEnabler {
            n_real: partial_n_real,
            row_count: blake_row_count,
        };
        let mut partial_component = blake_component.clone();
        partial_component.n_real_rows = partial_n_real;
        partial_component
            .input_gather
            .as_mut()
            .unwrap()
            .requirements
            .total_real_rows = partial_n_real;
        assert!(direct_blake_g_route_is_exact_for_test(
            &partial_lane,
            &partial_component,
        ));
        for symbolic_n_real in [partial_n_real - 1, partial_n_real + 1] {
            let mut boundary_drift = partial_lane.clone();
            boundary_drift.columns[enabler] = RecordedInputColumnProvenance::RetiredBlakeGEnabler {
                n_real: symbolic_n_real,
                row_count: blake_row_count,
            };
            assert!(!direct_blake_g_route_is_exact_for_test(
                &boundary_drift,
                &partial_component,
            ));
        }

        let mut host_tag = cold_recorded.clone();
        host_tag.lanes[blake_index].columns[enabler] =
            RecordedInputColumnProvenance::Host(Vec::new());
        assert_rejected(&host_tag, enabler);

        let mut gathered_tag = cold_recorded.clone();
        gathered_tag.lanes[blake_index].columns[enabler] =
            RecordedInputColumnProvenance::DeviceGather(DeviceGatherColumn::Enabler);
        assert_rejected(&gathered_tag, enabler);

        let mut materialized_tag = cold_recorded.clone();
        materialized_tag.lanes[blake_index].columns[enabler] =
            RecordedInputColumnProvenance::StructuralEnabler(Arc::from([]));
        assert_rejected(&materialized_tag, enabler);

        let mut wrong_edge = cold_recorded.clone();
        match &mut wrong_edge.lanes[blake_index].columns[0] {
            RecordedInputColumnProvenance::DeviceEdge(edge) => edge.source_word ^= 1,
            provenance => panic!("SN2: unexpected Blake-G operand: {provenance:?}"),
        }
        assert_rejected(&wrong_edge, 0);

        let mut short_pointer_component = blake_component.clone();
        short_pointer_component
            .input_gather
            .as_mut()
            .unwrap()
            .slots
            .consumer_input_columns
            .pop();
        assert!(!direct_blake_g_route_is_exact_for_test(
            &cold_recorded.lanes[blake_index],
            &short_pointer_component,
        ));

        let mut extra_column = cold_recorded.lanes[blake_index].clone();
        extra_column
            .columns
            .push(RecordedInputColumnProvenance::RetiredBlakeGEnabler {
                n_real: blake_n_real,
                row_count: blake_row_count,
            });
        assert!(!direct_blake_g_route_is_exact_for_test(
            &extra_column,
            blake_component,
        ));

        let mut wrong_program_width = blake_component.clone();
        wrong_program_width.program.n_inputs += 1;
        assert!(!direct_blake_g_route_is_exact_for_test(
            &cold_recorded.lanes[blake_index],
            &wrong_program_width,
        ));

        let mut wrong_writer_extent = blake_component.clone();
        wrong_writer_extent.requirements.input_column_words[0] -= 1;
        assert!(!direct_blake_g_route_is_exact_for_test(
            &cold_recorded.lanes[blake_index],
            &wrong_writer_extent,
        ));

        let mut wrong_gather_extent = blake_component.clone();
        wrong_gather_extent
            .input_gather
            .as_mut()
            .unwrap()
            .requirements
            .consumer_input_column_words[0] -= 1;
        assert!(!direct_blake_g_route_is_exact_for_test(
            &cold_recorded.lanes[blake_index],
            &wrong_gather_extent,
        ));
    }

    eprintln!(
        "SN_WARM_HANDLE_METRIC {}",
        serde_json::json!({
            "schema": "stwo.sn-warm-handle-matrix.v1",
            "profile": fixture.profile,
            "fixture_bytes": fixture.bytes,
            "fixture_sha256": fixture.sha256,
            "fixture_blake3": fixture.blake3,
            "policy_manifest_hash": format!("0x{:016x}", policy.kernel_manifest_hash),
            "policy_manifest_hash_scope": "host-only-non-cuda-fixture",
            "current_cuda_module_identity": serde_json::Value::Null,
            "cold_host_wall_ns": cold_host_wall_ns,
            "warm_host_wall_ns": warm_host_wall_ns,
            "cold_host_identity_ns": cold_audit.identity_ns,
            "warm_host_identity_ns": warm_audit.identity_ns,
            "cold_host_select_ns": cold_audit.select_ns,
            "warm_host_select_ns": warm_audit.select_ns,
            "cold_shape_wall_ns": cold_shape_wall_ns,
            "warm_shape_wall_ns": warm_shape_wall_ns,
            "retired_blake_g_enabler_host_payload_bytes": retired_blake_g_enabler_host_payload_bytes,
            "retired_blake_g_enabler_former_route_words_scanned": blake_row_count,
            "retired_blake_g_enabler_current_route_words_scanned": 0,
            "topology_key_constructions": shape_telemetry.topology_key_constructions,
            "replacement_handle_lock_ns": shape_telemetry.replacement_handle_lock_ns,
            "replacement_handle_lock_ops": shape_telemetry.replacement_handle_lock_ops,
        })
    );
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_1..4.adapted.bin"]
fn raw_replacement_warm_template_and_shape_handle_on_sn1_through_sn4() {
    let directory = std::env::var("STWO_SN_ADAPTED_DIR")
        .expect("set STWO_SN_ADAPTED_DIR to the sealed adapted-input directory");
    admit_fixture_identities(Path::new(&directory));
    for fixture in &SEALED_SN_FIXTURES {
        run_profile(Path::new(&directory), fixture);
    }
}

#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_2,4.adapted.bin"]
fn raw_replacement_adaptive_run_sum_is_exact_on_sn2_and_sn4() {
    let directory = std::env::var("STWO_SN_ADAPTED_DIR")
        .expect("set STWO_SN_ADAPTED_DIR to the sealed adapted-input directory");
    let directory = Path::new(&directory);
    for (profile, victim, runs, scratch_words, margin_words) in [
        ("SN2", 13, 17, 4_194_256, 4_194_352),
        ("SN4", 12, 19, 16_777_168, 48),
    ] {
        let fixture = SEALED_SN_FIXTURES
            .iter()
            .find(|fixture| fixture.profile == profile)
            .unwrap();
        assert_fixture_adaptive_run_sum(
            directory,
            fixture,
            victim,
            runs,
            scratch_words,
            margin_words,
        );
    }
}

/// Reproduce the preprocessed one-slab admission boundary from the exact SN2
/// replacement shape without touching CUDA. The ordinary progressive
/// constructor must reject the intentional repeated slot, while the planned
/// in-place constructor must admit it. This is the host oracle for the H100
/// `AliasedSlot` failure that preceded any kernel launch.
#[test]
#[ignore = "requires STWO_SN_ADAPTED_DIR containing sealed SN_PIE_2.adapted.bin"]
fn raw_replacement_preprocessed_in_place_shape_on_sn2() {
    let directory = std::env::var("STWO_SN_ADAPTED_DIR")
        .expect("set STWO_SN_ADAPTED_DIR to the sealed adapted-input directory");
    let fixture = SEALED_SN_FIXTURES
        .iter()
        .find(|fixture| fixture.profile == "SN2")
        .unwrap();
    let path = Path::new(&directory).join(fixture.file);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read sealed {}: {error}", path.display()));
    assert_eq!(bytes.len(), fixture.bytes, "SN2 sealed byte length");
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        fixture.blake3,
        "SN2 sealed BLAKE3"
    );
    let input: ProverInput = bincode::deserialize(&bytes)
        .unwrap_or_else(|error| panic!("decode sealed {}: {error}", path.display()));
    drop(bytes);

    let mut host_cache = ReplacementHostCache::new(1).unwrap();
    let ingest = prepare_resident_ingest(
        ResidentBackend::ReplacementV1,
        Some(&mut host_cache),
        input,
        PreProcessedTraceVariant::Canonical,
        None,
    )
    .unwrap();
    let preprocessed = Arc::clone(&ingest.preprocessed_trace);
    let ResidentPreWitnessInput::ReplacementV1 {
        input: owner,
        template,
    } = ingest.input
    else {
        panic!("SN2 replacement ingest returned legacy input")
    };
    let claim =
        planned_cairo_claim_from_public_data(owner.public_data(), template.exact_plan()).unwrap();
    let geometry = execution_geometry(&owner, &claim);
    let mut shape_cache = ShapeExecutableCache::new(1).unwrap();
    let selected = template
        .select_shape_executable(
            &mut shape_cache,
            &claim,
            &preprocessed,
            PcsConfig {
                pow_bits: 26,
                fri_config: FriConfig::new(0, 1, 70, 3),
                lifting_log_size: None,
            },
            false,
            Some(geometry),
            ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048),
        )
        .unwrap();
    let arena = selected.executable.arena();
    let commitment = arena
        .commitment(CommitmentTreeId::Preprocessed)
        .expect("SN2 has a preprocessed commitment");
    assert_eq!(
        commitment.storage_mode,
        ProgressiveCommitStorageMode::InPlaceSlab
    );
    assert!(commitment.commit_program.is_some());
    assert!(commitment.domain_cooperative_program.is_none());
    assert!(commitment.compact_domain_program.is_none());
    assert!(commitment.direct_retained_b2n_program.is_none());

    let (
        ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements),
        ModeAwareCommitWorkspaceSlots::DomainProgressive(slots),
    ) = (&commitment.requirements, &commitment.slots)
    else {
        panic!("SN2 replacement preprocessed commitment is not domain-progressive")
    };
    let slab = slots.leaves.state_ping;
    assert_eq!(slots.leaves.state_pong, Some(slab));
    assert_eq!(slots.leaves.leaf_hashes, slab);
    assert_eq!(slots.merkle.leaves, slab);
    assert_eq!(slots.merkle.merkle_scratch, Some(slab));
    assert!(matches!(
        requirements.arena_slot_requirements(slots),
        Err(PreparedProgressiveCommitError::AliasedSlot(id)) if id == slab
    ));
    let admitted = requirements
        .arena_slot_requirements_in_place(slots)
        .unwrap();
    assert_eq!(
        admitted.iter().filter(|entry| entry.id == slab).count(),
        1,
        "the one-slab identity must be merged exactly once"
    );
    let program = commitment.commit_program.as_ref().unwrap();
    assert_eq!(program.requirements(), requirements);
    assert_eq!(program.identity().config, commitment.config);
    assert_eq!(
        program.identity().storage,
        ProgressiveCommitStorageMode::InPlaceSlab
    );
    let protocol = arena.protocol_identity();
    let retained_evaluation_groups = commitment
        .evaluation_output_groups
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    assert!(matches!(
        preprocessed_commit_binding(PreprocessedCommitSelector {
            tree: commitment.id,
            backend: protocol.resident_backend,
            schedule: protocol.dynamic_commitment_leaf_schedule,
            commit_mode: protocol.commit_mode,
            interior4_fused: protocol.blake2s_interior_fused,
            storage: commitment.storage_mode,
            config: commitment.config,
            grouped_column_log_sizes: &commitment.grouped_column_log_sizes,
            retained_evaluation_groups: &retained_evaluation_groups,
            requirements: &commitment.requirements,
            slot_mode: PreprocessedCommitSlotMode::DomainProgressive,
            commit_program: commitment.commit_program.as_ref(),
            has_domain_program: commitment.domain_cooperative_program.is_some(),
            has_compact_program: commitment.compact_domain_program.is_some(),
            has_direct_program: commitment.direct_retained_b2n_program.is_some(),
        }),
        Ok(PreprocessedCommitBinding::ReplacementProgressiveInPlace(_))
    ));

    eprintln!(
        "SN2_PREPROCESSED_IN_PLACE physical={slab:?} slab_words={}",
        requirements.leaves.in_place_slab_words().unwrap()
    );
}
