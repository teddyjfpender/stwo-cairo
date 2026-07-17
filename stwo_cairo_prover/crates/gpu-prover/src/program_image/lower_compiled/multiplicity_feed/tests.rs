use super::*;
use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};

use crate::arena_plan::ExecutionTableGeometry;
use crate::compiled_proof::{
    AotArgumentValue, AtomicOperation, EffectAccess, InPlaceAliasRequirement, InPlaceDiscipline,
};
use crate::protocol_plan::ProtocolPlanPolicy;
use crate::resident_witness::planned_cairo_claim;
use crate::shape_executable::{ShapeCompileRequest, ShapeExecutableCache};

struct Fixture {
    executable: std::sync::Arc<crate::shape_executable::ShapeExecutable>,
    values: adapter::SemanticValueMap,
    lowered: LoweredMultiplicityFeed,
}

fn seed_fixture() -> &'static Fixture {
    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = public_seed_executable();
        let mut values =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
                .unwrap();
        super::super::multiplicity_clear::lower_stage(executable.arena(), &mut values).unwrap();
        let seed = executable
            .arena()
            .multiplicity()
            .unwrap()
            .public_memory_seed
            .as_ref()
            .unwrap();
        let lowered = lower_public_memory_seed(executable.arena(), seed, &mut values).unwrap();
        Fixture {
            executable,
            values,
            lowered,
        }
    })
}

fn public_seed_executable() -> std::sync::Arc<crate::shape_executable::ShapeExecutable> {
    let input = run_and_adapt(
        &get_compiled_cairo_program_path("test_prove_verify_sn2_profile"),
        ProgramType::Json,
        LayoutName::all_cairo_stwo,
        None,
    )
    .unwrap();
    let ingest = crate::phases::ingest::run(input, PreProcessedTraceVariant::Canonical, None);
    let proof_plan = ingest
        .proof_plan
        .strict_resident_exact(
            &crate::schedule_table::CAIRO_SCHEDULE,
            &crate::relation_table::CAIRO_RELATION_GRAPH,
        )
        .unwrap();
    let claim = planned_cairo_claim(&ingest.generator, &proof_plan).unwrap();
    let mut cache = ShapeExecutableCache::new(1).unwrap();
    cache
        .compile_or_bind(ShapeCompileRequest {
            claim: &claim,
            proof_plan: &proof_plan,
            preprocessed_trace: &ingest.preprocessed_trace,
            pcs: PcsConfig::default(),
            include_all_preprocessed_columns: false,
            execution_tables: Some(
                ExecutionTableGeometry::new(19, 17, 5).with_public_memory_entries(3),
            ),
            policy: ProtocolPlanPolicy::replacement_v1(0x534e_0001, 2048),
        })
        .unwrap()
        .executable
}

#[test]
fn public_seed_is_exact_zero_lut_three_slab_atomic_feed() {
    let fixture = seed_fixture();
    let values = &fixture.values;
    let lowered = &fixture.lowered;
    assert_eq!(lowered.owner, MultiplicityFeedOwner::PublicMemorySeed);
    assert!(lowered.luts.is_empty());
    assert_eq!(
        lowered
            .destinations
            .iter()
            .map(|destination| destination.name)
            .collect::<Vec<_>>(),
        [
            "memory_address_to_id",
            "memory_id_to_big",
            "memory_id_to_big#small",
        ]
    );
    assert_eq!(lowered.invocation.arguments.len(), 6);
    assert_eq!(
        lowered.invocation.arguments[4].value,
        AotArgumentValue::DevicePointerTable(vec![None])
    );
    assert_eq!(lowered.effect.accesses().len(), 5);
    assert!(matches!(
        lowered.effect.accesses()[0],
        EffectAccess::Read { .. }
    ));
    assert!(matches!(
        lowered.effect.accesses()[1],
        EffectAccess::Read { .. }
    ));
    for (ordinal, (access, destination)) in lowered.effect.accesses()[2..]
        .iter()
        .zip(&lowered.destinations)
        .enumerate()
    {
        let EffectAccess::Atomic {
            source,
            destination: output,
            operation,
            in_place,
        } = access
        else {
            panic!("seed destinations must be atomic transitions");
        };
        assert_eq!(*operation, AtomicOperation::AddU32);
        assert_eq!(source.binding, output.binding);
        assert_eq!(source.value.version, destination.source);
        assert_eq!(output.value.version, destination.destination);
        assert_eq!(source.value.elements, destination.elements);
        assert_eq!(in_place.id.0 as usize, ordinal);
        assert_eq!(in_place.requirement, InPlaceAliasRequirement::Required);
        assert_eq!(
            in_place.discipline,
            InPlaceDiscipline::ElementWiseReadBeforeWrite
        );
        assert_eq!(
            values.version(destination.value).unwrap(),
            destination.destination
        );
    }
    validate_lowered(&lowered).unwrap();
}

#[test]
fn public_seed_source_is_catalog_first_but_relocations_are_not_values() {
    let fixture = seed_fixture();
    let values = &fixture.values;
    let lowered = &fixture.lowered;
    let (catalog_first, transitions, fixed) = values.allocation_classes();
    assert!(catalog_first.contains(&lowered.source.version));
    assert!(!transitions.contains(&lowered.source.version));
    assert!(!fixed.contains(&lowered.source.version));
    assert_eq!(
        values
            .versions_for(lowered.source.value)
            .collect::<Vec<_>>(),
        [lowered.source.version]
    );
    assert!(fixed.contains(&lowered.descriptor_value));
    for relocation in [
        lowered.relocations.descriptor_workspace,
        lowered.relocations.lut_pointers,
        lowered.relocations.multiplicity_pointers,
    ] {
        assert!(values
            .version(ArenaCatalogValueId(relocation.logical.0))
            .is_err());
    }
}

#[test]
fn public_seed_plan_drift_rejects_transactionally() {
    let executable = &seed_fixture().executable;
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    super::super::multiplicity_clear::lower_stage(executable.arena(), &mut values).unwrap();
    let before = values.clone();
    let mut changed = executable
        .arena()
        .multiplicity()
        .unwrap()
        .public_memory_seed
        .clone()
        .unwrap();
    changed.plan.destination_components.swap(0, 1);
    assert_eq!(
        lower_public_memory_seed(executable.arena(), &changed, &mut values),
        Err(InvocationShapeError::InvalidScheduledProducerBinding)
    );
    assert_eq!(values, before);
}

#[test]
fn abi_or_partial_slab_mutation_is_rejected() {
    let lowered = &seed_fixture().lowered;
    let mut abi = lowered.contract.abi().arguments().to_vec();
    abi[5].name = "counts_drifted";
    assert_eq!(
        semantic::invocation_for_test(&lowered, &abi),
        Err(InvocationShapeError::InvalidStructuredAbi)
    );

    let mut changed = lowered.clone();
    changed.destinations[0].elements.end -= 1;
    assert!(validate_lowered(&changed).is_err());

    let mut changed = lowered.clone();
    changed.destinations[0].alias.discipline = InPlaceDiscipline::BlockBarrierPhases;
    assert!(validate_lowered(&changed).is_err());
}

#[test]
fn linked_projection_is_exact_and_metadata_mutations_fail_closed() {
    let lowered = &seed_fixture().lowered;
    let Some(linked) = lowered.contract.bind_static_build(89).unwrap() else {
        return;
    };
    let projected = project_static_wrapper(
        crate::compiled_proof::StaticCudaWrapperId(1),
        &linked,
        &lowered,
    )
    .unwrap();
    assert_eq!(projected.wrapper.accepted_effect(), lowered.effect.id());
    assert_eq!(projected.wrapper.kernel_launches().count(), 1);
    assert_eq!(
        projected.wrapper.aggregate_contract_identity(),
        &lowered.contract.identity()
    );

    let mut changed = lowered.clone();
    changed.relocations.lut_pointers = changed.relocations.descriptor_workspace;
    assert!(project_static_wrapper(
        crate::compiled_proof::StaticCudaWrapperId(1),
        &linked,
        &changed
    )
    .is_err());

    let mut changed = lowered.clone();
    changed.source.arena.len_words -= 1;
    assert!(project_static_wrapper(
        crate::compiled_proof::StaticCudaWrapperId(1),
        &linked,
        &changed
    )
    .is_err());
}
