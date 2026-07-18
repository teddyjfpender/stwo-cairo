use stwo_backend_cuda::{witness_casm_input_requirements, WitnessCasmInputContract};

use super::*;

const LANES: usize = 9;
const REAL_ROWS: usize = 7;

fn host_source(ordinal: u32) -> StatementHostSource {
    let requirements = witness_casm_input_requirements(REAL_ROWS, ordinal % 2 == 0).unwrap();
    let contract = WitnessCasmInputContract::compile(&requirements).unwrap();
    StatementHostSource {
        kind: StatementHostSourceKind::WitnessCasm,
        producer_ordinal: ordinal,
        component: format!("component_{ordinal}").into_boxed_str(),
        part: match ordinal % 3 {
            0 => StatementHostPart::Main,
            1 => StatementHostPart::MemoryBig(ordinal),
            _ => StatementHostPart::MemorySmall,
        },
        encoding: StatementHostEncoding::RowMajorU32,
        words: requirements.staging_words,
        real_rows: requirements.n_real_rows,
        consumer_rows: requirements.consumer_rows,
        include_iota: requirements.include_iota,
        casm_contract_identity: contract.identity(),
    }
}

fn statement_fixture() -> (Fixture, Vec<ValueVersion>) {
    let mut fixture = fixture();
    let mut input = fixture.compiled.input().clone();
    let mut assembly = input.operations.pop().unwrap();
    assembly.id = OpId(LANES as u32);
    input
        .values
        .iter_mut()
        .find(|value| value.region == Region::Output)
        .unwrap()
        .origin = ValueOrigin::OpOutput(assembly.id);

    let mut operations = Vec::with_capacity(LANES + 1);
    let mut versions = Vec::with_capacity(LANES);
    for lane in 0..LANES {
        let operation = OpId(lane as u32);
        let source = host_source(lane as u32);
        let version = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            version,
            source.words,
            4,
            ValueOrigin::OpOutput(operation),
            Region::Dynamic,
        ));
        let destination = value_range(version, source.words);
        let effect = EffectContract::new(
            vec![EffectAccess::Write {
                destination: bound(0, destination),
            }],
            vec![],
        )
        .unwrap();
        input.effects.push(effect.clone());
        operations.push(OpNode {
            id: operation,
            semantic_id: SemanticOpId(100 + lane as u32),
            primitive: ExecutionPrimitive::StatementHostIngress {
                source,
                predecessor: versions
                    .last()
                    .copied()
                    .map(|previous| value_range(previous, destination.elements.len())),
            },
            invocation: None,
            effect: effect.id(),
            partition: assembly.partition,
            stage: assembly.stage,
        });
        versions.push(version);
    }
    operations.push(assembly);
    input.operations = operations;
    input.effects.sort_unstable_by_key(EffectContract::id);
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    (fixture, versions)
}

fn compile_monolithic(fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    FleetProofPlan::compile_track_a_monolithic(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

fn compile_partitioned(mut fixture: Fixture) -> Result<FleetProofPlan, FleetCompileError> {
    fixture.placement.topology.workers.push(WorkerSpec {
        id: WorkerId(1),
        capacity_bytes: 1024,
        exchange_reserve_bytes: 0,
    });
    FleetProofPlan::compile_track_a_partitioned(
        fixture.compiled,
        fixture.shape,
        fixture.placement.topology,
        fixture.placement.pow,
        transcript(),
    )
}

fn lineage_bindings<'a>(
    plan: &'a FleetProofPlan,
    versions: &[ValueVersion],
) -> Vec<&'a FleetStoragePlacement> {
    versions
        .iter()
        .map(|version| {
            plan.placement()
                .storage_bindings
                .iter()
                .find(|binding| binding.value.version == *version)
                .unwrap()
        })
        .collect()
}

fn lineage_owners<'a>(
    plan: &'a FleetProofPlan,
    versions: &[ValueVersion],
) -> Vec<&'a FleetOwnerPlacement> {
    versions
        .iter()
        .map(|version| {
            plan.placement()
                .owners
                .iter()
                .find(|owner| owner.value.version == *version)
                .unwrap()
        })
        .collect()
}

fn assert_lineage_reuse(plan: &FleetProofPlan, versions: &[ValueVersion]) {
    let bindings = lineage_bindings(plan, versions);
    assert!(bindings.iter().all(|binding| {
        binding.storage == bindings[0].storage
            && binding.offset_bytes == 0
            && binding.value.elements.start == 0
    }));
    let owners = lineage_owners(plan, versions);
    assert!(owners.iter().all(|owner| owner.worker == WorkerId(0)));
    assert!(owners
        .windows(2)
        .all(|pair| pair[0].live.end == pair[1].live.start));
    assert_eq!(
        owners.last().unwrap().live.end,
        plan.placement().terminal_step
    );
}

fn assert_install_projection(
    plan: &FleetProofPlan,
    versions: &[ValueVersion],
) -> Result<(), FleetWorkerInstallError> {
    let install = plan.worker_install_plan(WorkerId(0))?;
    let ingress = install
        .executions()
        .iter()
        .take(LANES)
        .map(|execution| {
            execution.executables[0]
                .statement_host_ingress
                .as_ref()
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(ingress.len(), LANES);
    for (lane, projected) in ingress.iter().enumerate() {
        assert_eq!(projected.source, host_source(lane as u32));
        assert_eq!(projected.destination.version, versions[lane]);
        assert_eq!(
            projected.predecessor,
            lane.checked_sub(1)
                .map(|previous| value_range(versions[previous], projected.source.words))
        );
        if lane != 0 {
            assert_eq!(projected.window, ingress[lane - 1].window);
        }
    }
    Ok(())
}

#[test]
fn monolithic_statement_lineage_reuses_one_storage_and_preserves_install_source() {
    let (fixture, versions) = statement_fixture();
    let plan = compile_monolithic(fixture).unwrap();
    assert_lineage_reuse(&plan, &versions);
    assert_install_projection(&plan, &versions).unwrap();
}

#[test]
fn partitioned_statement_lineage_stays_monolithic_on_the_coordinator() {
    let (fixture, versions) = statement_fixture();
    let plan = compile_partitioned(fixture).unwrap();
    for operation in plan.placement().operations.iter().take(LANES) {
        assert_eq!(
            operation.executions.as_slice(),
            [FleetOperationExecution {
                worker: WorkerId(0),
                domain: OperationDomain::Monolithic,
            }]
        );
    }
    assert_lineage_reuse(&plan, &versions);
    assert_install_projection(&plan, &versions).unwrap();
    let peer = plan.worker_install_plan(WorkerId(1)).unwrap();
    assert!(peer.executions().iter().all(|execution| {
        !matches!(
            &plan
                .compiled()
                .operation(execution.operation)
                .unwrap()
                .primitive,
            ExecutionPrimitive::StatementHostIngress { .. }
        )
    }));
}

#[test]
fn worker_install_rejects_a_wrong_nonzero_casm_contract_identity() {
    let (mut fixture, _) = statement_fixture();
    let mut input = fixture.compiled.input().clone();
    let ExecutionPrimitive::StatementHostIngress { source, .. } =
        &mut input.operations[0].primitive
    else {
        unreachable!()
    };
    source.casm_contract_identity = [7; 32];
    let compiled = Arc::new(CompiledProof::compile(input, transcript()).unwrap());
    fixture.shape = shape_identity_for_test(
        b"fleet-test-topology-v2",
        b"fleet-test-workspace-v2",
        compiled.transcript_encoding(),
        compiled.identity().canonical_encoding(),
    )
    .unwrap();
    fixture.compiled = compiled;
    let plan = compile_monolithic(fixture).unwrap();
    assert_eq!(
        plan.worker_install_plan(WorkerId(0)).unwrap_err(),
        FleetWorkerInstallError::InvalidStatementHostIngress(OpId(0))
    );
}
