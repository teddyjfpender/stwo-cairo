use stwo_backend_cuda::{witness_casm_input_requirements, WitnessCasmInputContract};

use super::*;

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

fn chain_input(lanes: usize) -> (CompiledProofInput, Vec<ValueVersion>) {
    let mut input = valid_input();
    let mut assembly = input.operations.pop().unwrap();
    let assembly_id = OpId(lanes as u32);
    assembly.id = assembly_id;
    input
        .values
        .iter_mut()
        .find(|value| value.region == Region::Output)
        .unwrap()
        .origin = ValueOrigin::OpOutput(assembly_id);

    let partition = assembly.partition;
    let stage = assembly.stage;
    let mut versions = Vec::with_capacity(lanes);
    let mut operations = Vec::with_capacity(lanes + 1);
    for lane in 0..lanes {
        let op = OpId(lane as u32);
        let source = host_source(lane as u32);
        let version = ValueVersion(input.values.len() as u32);
        input.values.push(u32_value(
            version,
            source.words,
            ValueOrigin::OpOutput(op),
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
            id: op,
            semantic_id: SemanticOpId(100 + lane as u32),
            primitive: ExecutionPrimitive::StatementHostIngress {
                source,
                predecessor: versions
                    .last()
                    .copied()
                    .map(|version| value_range(version, destination.elements.len())),
            },
            invocation: None,
            effect: effect.id(),
            partition,
            stage,
        });
        versions.push(version);
    }
    operations.push(assembly);
    input.operations = operations;
    input.effects.sort_unstable_by_key(EffectContract::id);
    (input, versions)
}

fn replace_ingress_effect(input: &mut CompiledProofInput, operation: OpId, effect: EffectContract) {
    let old = input.operations[operation.0 as usize].effect;
    input.operations[operation.0 as usize].effect = effect.id();
    input.effects.retain(|candidate| candidate.id() != old);
    input.effects.push(effect);
    input.effects.sort_unstable_by_key(EffectContract::id);
}

fn add_late_assembly_read(input: &mut CompiledProofInput, source: ValueRange) {
    let operation = input.operations.last_mut().unwrap();
    let old_effect = operation.effect;
    let effect = input
        .effects
        .iter()
        .find(|candidate| candidate.id() == old_effect)
        .unwrap();
    let mut accesses = effect.accesses().to_vec();
    accesses.push(EffectAccess::Read {
        source: bound(accesses.len() as u32, source),
    });
    let effect = EffectContract::new(accesses, vec![]).unwrap();
    operation.effect = effect.id();
    operation.invocation = invocation(&effect);
    input
        .effects
        .retain(|candidate| candidate.id() != old_effect);
    input.effects.push(effect.clone());
    input.effects.sort_unstable_by_key(EffectContract::id);

    let kernel = input.kernels[0].clone();
    input.kernels[0] = AotKernelAuthority::new_with_accepted_executions(
        kernel.id(),
        kernel.module().clone(),
        kernel.semantic_encoding().to_vec(),
        kernel.execution_build_encoding().to_vec(),
        vec![(
            effect.id(),
            operation.partition,
            operation
                .invocation
                .as_ref()
                .unwrap()
                .contract_id()
                .unwrap(),
        )],
    )
    .unwrap();
}

#[test]
fn statement_host_ingress_accepts_a_nine_lane_write_only_linear_chain() {
    let (input, versions) = chain_input(9);
    let compiled = CompiledProof::compile(input, transcript()).unwrap();
    assert_eq!(versions.len(), 9);
    for (lane, operation) in compiled.operations()[..9].iter().enumerate() {
        let ExecutionPrimitive::StatementHostIngress {
            source,
            predecessor,
        } = &operation.primitive
        else {
            panic!("lane must remain first-class host ingress")
        };
        assert_eq!(source.producer_ordinal, lane as u32);
        assert_eq!(
            *predecessor,
            lane.checked_sub(1)
                .map(|previous| value_range(versions[previous], source.words))
        );
        assert!(operation.invocation.is_none());
        assert!(matches!(
            compiled.effect_for(operation.id).unwrap().accesses(),
            [EffectAccess::Write { .. }]
        ));
    }
}

#[test]
fn statement_host_ingress_shape_and_lineage_are_identity_bound() {
    let (baseline, _) = chain_input(2);
    let baseline = CompiledProof::compile(baseline, transcript()).unwrap();
    let (mut changed, _) = chain_input(2);
    let ExecutionPrimitive::StatementHostIngress {
        source,
        predecessor: _,
    } = &mut changed.operations[1].primitive
    else {
        unreachable!()
    };
    source.component = "changed_component".into();
    let changed = CompiledProof::compile(changed, transcript()).unwrap();
    assert_ne!(baseline.identity(), changed.identity());

    let (mut changed, _) = chain_input(2);
    let ExecutionPrimitive::StatementHostIngress { predecessor, .. } =
        &mut changed.operations[1].primitive
    else {
        unreachable!()
    };
    *predecessor = None;
    let changed = CompiledProof::compile(changed, transcript()).unwrap();
    assert_ne!(baseline.identity(), changed.identity());
}

#[test]
fn statement_host_ingress_rejects_non_shape_or_fake_device_effects() {
    let (mut empty_component, _) = chain_input(1);
    let ExecutionPrimitive::StatementHostIngress { source, .. } =
        &mut empty_component.operations[0].primitive
    else {
        unreachable!()
    };
    source.component = "".into();
    assert_eq!(
        CompiledProof::compile(empty_component, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostIngress { operation: OpId(0) }
    );

    let (mut zero_contract, _) = chain_input(1);
    let ExecutionPrimitive::StatementHostIngress { source, .. } =
        &mut zero_contract.operations[0].primitive
    else {
        unreachable!()
    };
    source.casm_contract_identity = [0; 32];
    assert_eq!(
        CompiledProof::compile(zero_contract, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostIngress { operation: OpId(0) }
    );

    let (mut invoked, _) = chain_input(1);
    invoked.operations[0].invocation = Some(AotInvocation { arguments: vec![] });
    assert_eq!(
        CompiledProof::compile(invoked, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostIngress { operation: OpId(0) }
    );

    let (mut fake_read, versions) = chain_input(2);
    let second = value_range(versions[1], host_source(1).words);
    replace_ingress_effect(
        &mut fake_read,
        OpId(1),
        EffectContract::new(
            vec![
                EffectAccess::Read {
                    source: bound(0, value_range(versions[0], host_source(0).words)),
                },
                EffectAccess::Write {
                    destination: bound(1, second),
                },
            ],
            vec![],
        )
        .unwrap(),
    );
    assert_eq!(
        CompiledProof::compile(fake_read, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostIngress { operation: OpId(1) }
    );

    let (mut fake_copy, _) = chain_input(1);
    fake_copy.operations[0].primitive = ExecutionPrimitive::DeviceCopyD2D {
        bytes: host_source(0).words * size_of::<u32>(),
    };
    assert_eq!(
        CompiledProof::compile(fake_copy, transcript()).unwrap_err(),
        CompiledProofError::PrimitiveEffectMismatch(OpId(0))
    );
}

#[test]
fn statement_host_ingress_rejects_fork_range_layout_and_order_drift() {
    let (mut fork, versions) = chain_input(3);
    let ExecutionPrimitive::StatementHostIngress { predecessor, .. } =
        &mut fork.operations[2].primitive
    else {
        unreachable!()
    };
    *predecessor = Some(value_range(versions[0], host_source(0).words));
    assert_eq!(
        CompiledProof::compile(fork, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(2) }
    );

    let (mut partial, versions) = chain_input(2);
    let ExecutionPrimitive::StatementHostIngress { predecessor, .. } =
        &mut partial.operations[1].primitive
    else {
        unreachable!()
    };
    *predecessor = Some(ValueRange {
        version: versions[0],
        elements: range(1, host_source(0).words),
    });
    assert_eq!(
        CompiledProof::compile(partial, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(1) }
    );

    let (mut layout, versions) = chain_input(2);
    layout.values[versions[0].0 as usize].alignment = 8;
    assert_eq!(
        CompiledProof::compile(layout, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(1) }
    );

    let (mut future, versions) = chain_input(3);
    let ExecutionPrimitive::StatementHostIngress { predecessor, .. } =
        &mut future.operations[1].primitive
    else {
        unreachable!()
    };
    *predecessor = Some(value_range(versions[2], host_source(2).words));
    assert_eq!(
        CompiledProof::compile(future, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(1) }
    );

    let (mut ordinal, _) = chain_input(2);
    let ExecutionPrimitive::StatementHostIngress { source, .. } =
        &mut ordinal.operations[1].primitive
    else {
        unreachable!()
    };
    source.producer_ordinal = 0;
    assert_eq!(
        CompiledProof::compile(ordinal, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(1) }
    );
}

#[test]
fn statement_host_ingress_rejects_a_predecessor_with_a_late_consumer() {
    let (mut input, versions) = chain_input(2);
    add_late_assembly_read(&mut input, value_range(versions[0], host_source(0).words));
    assert_eq!(
        CompiledProof::compile(input, transcript()).unwrap_err(),
        CompiledProofError::InvalidStatementHostLineage { operation: OpId(1) }
    );
}
