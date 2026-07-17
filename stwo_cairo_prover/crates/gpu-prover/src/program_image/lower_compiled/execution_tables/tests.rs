use std::sync::{Arc, OnceLock};

use super::*;
use crate::arena_plan::ExecutionTableGeometry;
use crate::compiled_proof::{AotArgumentValue, EffectAccess};
use crate::shape_executable::ShapeExecutable;

struct Fixture {
    executable: Arc<ShapeExecutable>,
    initial: adapter::SemanticValueMap,
    values: adapter::SemanticValueMap,
    lowered: LoweredExecutionTables,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let executable = super::super::tests::generated_sn2_replacement();
        let initial =
            adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
                .unwrap();
        let mut values = initial.clone();
        let lowered = lower_stage(executable.arena(), &mut values).unwrap();
        Fixture {
            executable,
            initial,
            values,
            lowered,
        }
    })
}

#[test]
fn generated_sn2_lowers_exact_ingress_then_big_and_small_splits() {
    let fixture = fixture();
    let lowered = &fixture.lowered;
    assert_eq!(
        lowered.host_ingress.each_ref().map(|ingress| ingress.role),
        [
            ExecutionTablesHostIngressRole::RawAddressToId,
            ExecutionTablesHostIngressRole::F252Values,
            ExecutionTablesHostIngressRole::SmallValues,
        ]
    );
    assert_eq!(
        lowered
            .host_ingress
            .each_ref()
            .map(|ingress| ingress.encoding),
        [
            ExecutionTablesHostIngressEncoding::RawU32,
            ExecutionTablesHostIngressEncoding::F252LittleEndianU32x8,
            ExecutionTablesHostIngressEncoding::SmallU128LittleEndianU32x4,
        ]
    );
    assert_eq!(
        lowered
            .host_ingress
            .each_ref()
            .map(|ingress| ingress.copied_words),
        lowered
            .contract
            .host_ingress()
            .fields
            .map(|field| field.copied_words)
    );
    assert_eq!(
        lowered.stages.each_ref().map(|stage| stage.stage),
        [ExecutionTablesStage::Big, ExecutionTablesStage::Small]
    );
    assert_eq!(
        lowered.stages.each_ref().map(|stage| stage.outputs.len()),
        [28, 8]
    );
    for (lowered_stage, contract_stage) in lowered.stages.iter().zip(lowered.contract.stages()) {
        assert_eq!(
            lowered_stage.effect.accesses().len(),
            lowered_stage.outputs.len() + usize::from(lowered_stage.source.is_some())
        );
        assert_eq!(
            lowered_stage.outputs.len(),
            contract_stage.effect_geometry().output_writes.len()
        );
        assert_eq!(
            lowered_stage
                .outputs
                .iter()
                .map(|output| output.elements.len())
                .collect::<Vec<_>>(),
            contract_stage
                .effect_geometry()
                .output_writes
                .iter()
                .map(|output| output.written_words as usize)
                .collect::<Vec<_>>()
        );
    }
    validate(
        fixture.executable.arena(),
        &fixture.values,
        &fixture.lowered,
    )
    .unwrap();
}

#[test]
fn host_ingress_and_outputs_are_catalog_first_but_relocations_are_not_values() {
    let fixture = fixture();
    let (catalog_first, transitions, fixed) = fixture.values.allocation_classes();
    let semantic_versions = fixture
        .lowered
        .host_ingress
        .iter()
        .filter_map(|ingress| ingress.version)
        .chain(
            fixture
                .lowered
                .stages
                .iter()
                .flat_map(|stage| stage.outputs.iter().map(|output| output.version)),
        )
        .collect::<Vec<_>>();
    assert_eq!(semantic_versions.len(), 39);
    assert!(semantic_versions.iter().all(|version| {
        catalog_first.contains(version)
            && !transitions.contains(version)
            && !fixed.contains(version)
    }));
    for relocation in [
        fixture.lowered.relocations.table_pointers,
        fixture.lowered.relocations.table_strides,
    ] {
        assert!(fixture
            .values
            .version(ArenaCatalogValueId(relocation.logical.0))
            .is_err());
    }
    assert_eq!(
        fixture.values.allocated_versions().count(),
        fixture.initial.allocated_versions().count() + semantic_versions.len()
    );
}

#[test]
fn lowering_is_idempotent_and_rejects_abi_drift() {
    let fixture = fixture();
    let mut repeated = fixture.values.clone();
    assert_eq!(
        lower_stage(fixture.executable.arena(), &mut repeated).unwrap(),
        fixture.lowered
    );
    assert_eq!(repeated, fixture.values);

    let stage = &fixture.lowered.stages[0];
    let contract_stage = &fixture.lowered.contract.stages()[0];
    let mut changed = contract_stage.abi().arguments().to_vec();
    changed[0].name = "wrong_values";
    assert_eq!(
        semantic::invocation_for_test(
            contract_stage,
            stage.source_binding,
            &stage.outputs,
            &changed,
        ),
        Err(InvocationShapeError::InvalidStructuredAbi)
    );
}

#[test]
fn empty_tables_have_no_ingress_values_and_write_only_dense_splits() {
    let executable = super::super::tests::generated_sn2_legacy_with_execution_tables(
        ExecutionTableGeometry::new(0, 0, 0),
    );
    let mut values =
        adapter::SemanticValueMap::allocate_ordered(std::iter::empty::<ArenaCatalogValueId>())
            .unwrap();
    let lowered = lower_stage(executable.arena(), &mut values).unwrap();

    assert!(lowered
        .host_ingress
        .iter()
        .all(|ingress| ingress.elements.is_none() && ingress.version.is_none()));
    for stage in &lowered.stages {
        assert_eq!(stage.source, None);
        assert_eq!(stage.source_binding, None);
        assert_eq!(
            stage.invocation.arguments[0].value,
            AotArgumentValue::DevicePointer(None)
        );
        assert!(stage
            .effect
            .accesses()
            .iter()
            .all(|access| matches!(access, EffectAccess::Write { .. })));
        assert_eq!(
            stage
                .outputs
                .iter()
                .map(|output| output.binding.0)
                .collect::<Vec<_>>(),
            (0..stage.outputs.len() as u32).collect::<Vec<_>>()
        );
    }
    assert_eq!(
        lowered.stages.each_ref().map(|stage| stage.outputs.len()),
        [28, 8]
    );
}
