use core::cell::Cell;

use stwo_backend_cuda::{
    witness_casm_input_requirements, PreparedWitnessCasmInputError, WitnessCasmInputContract,
};

use super::*;
use crate::compiled_proof::{
    EffectBindingId, ElementRange, StatementHostEncoding, StatementHostSourceKind, ValueVersion,
};
use crate::fleet_plan::{
    FleetEffectBinding, FleetStatementHostIngress, OperationDomain, ScheduleRange, ScheduleStep,
    StorageId,
};

const PLAN_ID: [u8; 32] = [7; 32];
const WORKER: WorkerId = WorkerId(0);
const BASE: usize = 0x10_0000;
const CONTEXT: u64 = 41;
const GENERATION: u64 = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FakeReceipt(ReceiptFacts);

struct FakeStage {
    current: Cell<Option<FakeReceipt>>,
    fail_consume: Cell<bool>,
    consumes: Cell<u32>,
}

impl FakeStage {
    const fn new() -> Self {
        Self {
            current: Cell::new(None),
            fail_consume: Cell::new(false),
            consumes: Cell::new(0),
        }
    }

    fn publish(&self, facts: ReceiptFacts) {
        self.current.set(Some(FakeReceipt(facts)));
    }
}

impl StageAuthority for FakeStage {
    type Receipt = FakeReceipt;

    fn facts(&self, receipt: Self::Receipt) -> ReceiptFacts {
        receipt.0
    }

    fn is_current(&self, receipt: &Self::Receipt) -> bool {
        self.current.get() == Some(*receipt)
    }

    fn consume(&self, receipt: Self::Receipt) -> Result<(), PreparedWitnessCasmInputError> {
        if self.fail_consume.get() || self.current.get() != Some(receipt) {
            return Err(PreparedWitnessCasmInputError::InvalidIngressReceipt);
        }
        self.current.set(None);
        self.consumes.set(self.consumes.get() + 1);
        Ok(())
    }

    fn invalidate(&self) {
        self.current.set(None);
    }
}

fn range(version: u32, words: usize) -> ValueRange {
    ValueRange {
        version: ValueVersion(version),
        elements: ElementRange::new(0, words).unwrap(),
    }
}

fn window() -> FleetInstallWindow {
    FleetInstallWindow {
        storage: StorageId(3),
        offset_bytes: 64,
        slab_offset_bytes: 256,
        bytes: 6 * WORD_BYTES,
    }
}

fn source(lane: usize) -> StatementHostSource {
    let requirements = witness_casm_input_requirements(2, lane % 2 == 0).unwrap();
    let contract = WitnessCasmInputContract::compile(&requirements).unwrap();
    StatementHostSource {
        kind: StatementHostSourceKind::WitnessCasm,
        producer_ordinal: lane as u32,
        component: format!("component_{lane}").into_boxed_str(),
        part: match lane % 3 {
            0 => StatementHostPart::Main,
            1 => StatementHostPart::MemoryBig(lane as u32),
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

fn execution(
    operation: OpId,
    executable: super::super::FleetWorkerExecutable,
) -> FleetWorkerExecution {
    FleetWorkerExecution {
        step: ScheduleStep(operation.0 + 10),
        operation,
        domain: OperationDomain::Monolithic,
        during: ScheduleRange::new(
            ScheduleStep(operation.0 + 10),
            ScheduleStep(operation.0 + 11),
        )
        .unwrap(),
        executables: vec![executable],
    }
}

fn execution_fixture(lanes: usize) -> (Vec<FleetWorkerExecution>, Vec<OperationAuthority>) {
    let mut executions = Vec::with_capacity(lanes * 2);
    let mut authorities = Vec::with_capacity(lanes * 2);
    for lane in 0..lanes {
        let source = source(lane);
        let destination = range(100 + lane as u32, source.words);
        let predecessor = lane
            .checked_sub(1)
            .map(|previous| range(100 + previous as u32, source.words));
        let ingress = OpId((lane * 2) as u32);
        executions.push(execution(
            ingress,
            super::super::FleetWorkerExecutable {
                child_ordinal: None,
                effects: vec![FleetEffectBinding {
                    binding: EffectBindingId(0),
                    source: None,
                    destination: Some(destination),
                    window: window(),
                }],
                statement_host_ingress: Some(FleetStatementHostIngress {
                    source: source.clone(),
                    predecessor,
                    destination,
                    window: window(),
                }),
            },
        ));
        authorities.push(OperationAuthority::Ingress {
            source: source.clone(),
            predecessor,
        });

        let scatter = OpId(ingress.0 + 1);
        executions.push(execution(
            scatter,
            super::super::FleetWorkerExecutable {
                child_ordinal: None,
                effects: vec![FleetEffectBinding {
                    binding: EffectBindingId(0),
                    source: Some(destination),
                    destination: None,
                    window: window(),
                }],
                statement_host_ingress: None,
            },
        ));
        authorities.push(OperationAuthority::Scatter {
            contract_identity: source.casm_contract_identity,
        });
    }
    (executions, authorities)
}

fn expected(lanes: usize) -> Result<ExpectedSetup, FleetCasmSetupAdmissionError> {
    let (executions, authorities) = execution_fixture(lanes);
    let mut authorities = authorities.into_iter();
    expected_setup_from(PLAN_ID, WORKER, &executions, |_| {
        Ok(authorities.next().unwrap())
    })
}

fn facts(pair: &ExpectedPair) -> ReceiptFacts {
    ReceiptFacts {
        contract_identity: pair.source.casm_contract_identity,
        arena_identity: BASE,
        exec_context_token: CONTEXT,
        staging_address: BASE + pair.window.slab_offset_bytes + pair.window.offset_bytes,
        staging_words: pair.source.words,
        abi: WitnessCasmInputAbi::RowMajorStateScatterV1,
        row_domain: WitnessCasmInputRowDomain::RealPrefixWithRowZeroPaddingV1,
        state_words_per_row: WITNESS_CASM_STATE_WORDS,
        real_rows: pair.source.real_rows,
        consumer_rows: pair.source.consumer_rows,
        include_iota: pair.source.include_iota,
        generation: GENERATION,
    }
}

fn binding() -> FleetCasmRuntimeBinding {
    FleetCasmRuntimeBinding::new(PLAN_ID, WORKER, BASE, CONTEXT)
}

fn begin(expected: &ExpectedSetup) -> (FleetCasmSetupAdmissionState, Vec<FakeStage>) {
    let stages = (0..REQUIRED_PAIR_COUNT)
        .map(|_| FakeStage::new())
        .collect::<Vec<_>>();
    let mut state = FleetCasmSetupAdmissionState::default();
    let refs = stages.iter().collect::<Vec<_>>();
    state
        .begin_statement_with(PLAN_ID, GENERATION, &refs)
        .unwrap();
    for (stage, pair) in stages.iter().zip(&expected.pairs) {
        stage.publish(facts(pair));
    }
    (state, stages)
}

fn candidates<'a>(
    expected: &'a ExpectedSetup,
    stages: &'a [FakeStage],
) -> Vec<IngressCandidate<'a, FakeStage>> {
    stages
        .iter()
        .zip(&expected.pairs)
        .map(|(stage, pair)| IngressCandidate {
            source: FleetCasmSetupSource {
                producer_ordinal: pair.source.producer_ordinal,
                component: &pair.source.component,
                part: pair.source.part,
            },
            stage,
            receipt: stage.current.get().unwrap(),
        })
        .collect()
}

fn admit(
    state: &mut FleetCasmSetupAdmissionState,
    expected: &ExpectedSetup,
    stages: &[FakeStage],
) -> Result<(), FleetCasmSetupAdmissionError> {
    let refs = stages.iter().collect::<Vec<_>>();
    state.admit_expected(
        expected.clone(),
        binding(),
        GENERATION,
        &refs,
        &candidates(expected, stages),
    )
}

#[test]
fn exact_nine_pairs_publish_once_and_preserve_the_complete_schedule() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();
    let (mut state, stages) = begin(&expected);
    admit(&mut state, &expected, &stages).unwrap();
    assert!(stages.iter().all(|stage| stage.current.get().is_none()));

    let admission = state.take(PLAN_ID, GENERATION).unwrap();
    assert_eq!(
        admission.receipt_generations(),
        &[GENERATION; REQUIRED_PAIR_COUNT]
    );
    assert_eq!(
        admission.pre_executed_operations().count(),
        REQUIRED_OPERATION_COUNT
    );
    let mut complete = expected
        .pairs
        .iter()
        .flat_map(|pair| [pair.ingress.clone(), pair.scatter.clone()])
        .collect::<Vec<_>>();
    let ordinary = execution(
        OpId(99),
        super::super::FleetWorkerExecutable {
            child_ordinal: None,
            effects: vec![],
            statement_host_ingress: None,
        },
    );
    complete.push(ordinary.clone());
    let schedule_before = complete
        .iter()
        .map(|execution| (execution.operation, execution.step, execution.during))
        .collect::<Vec<_>>();
    let dispositions = admission.classify_executions(&complete).unwrap();
    assert_eq!(dispositions.len(), complete.len());
    assert_eq!(
        dispositions
            .iter()
            .filter(|&&kind| kind == FleetCasmSetupDisposition::PreExecuted)
            .count(),
        REQUIRED_OPERATION_COUNT
    );
    assert_eq!(
        dispositions.last(),
        Some(&FleetCasmSetupDisposition::Execute)
    );
    assert_eq!(
        schedule_before,
        complete
            .iter()
            .map(|execution| (execution.operation, execution.step, execution.during))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        state.take(PLAN_ID, GENERATION).unwrap_err(),
        FleetCasmSetupAdmissionError::AdmissionUnavailable
    );
}

#[test]
fn pair_compilation_rejects_missing_extra_and_operation_or_window_drift() {
    assert_eq!(
        expected(REQUIRED_PAIR_COUNT - 1).unwrap_err(),
        FleetCasmSetupAdmissionError::PairCount {
            expected: REQUIRED_PAIR_COUNT,
            actual: REQUIRED_PAIR_COUNT - 1,
        }
    );
    assert_eq!(
        expected(REQUIRED_PAIR_COUNT + 1).unwrap_err(),
        FleetCasmSetupAdmissionError::PairCount {
            expected: REQUIRED_PAIR_COUNT,
            actual: REQUIRED_PAIR_COUNT + 1,
        }
    );

    let (mut executions, authorities) = execution_fixture(REQUIRED_PAIR_COUNT);
    executions[1].operation = OpId(73);
    let mut authorities = authorities.into_iter();
    assert_eq!(
        expected_setup_from(PLAN_ID, WORKER, &executions, |_| {
            Ok(authorities.next().unwrap())
        })
        .unwrap_err(),
        FleetCasmSetupAdmissionError::InvalidPair(0)
    );

    let (mut executions, authorities) = execution_fixture(REQUIRED_PAIR_COUNT);
    executions[0].executables[0].effects[0].window.offset_bytes += WORD_BYTES;
    let mut authorities = authorities.into_iter();
    assert_eq!(
        expected_setup_from(PLAN_ID, WORKER, &executions, |_| {
            Ok(authorities.next().unwrap())
        })
        .unwrap_err(),
        FleetCasmSetupAdmissionError::InvalidPair(0)
    );
}

#[test]
fn receipt_set_rejects_missing_extra_duplicate_and_reordered_entries() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();

    for mutation in 0..4 {
        let (mut state, stages) = begin(&expected);
        let mut candidates = candidates(&expected, &stages);
        let expected_error = match mutation {
            0 => {
                candidates.pop();
                FleetCasmSetupAdmissionError::ReceiptCount {
                    expected: REQUIRED_PAIR_COUNT,
                    actual: REQUIRED_PAIR_COUNT - 1,
                }
            }
            1 => {
                candidates.push(candidates[0]);
                FleetCasmSetupAdmissionError::ReceiptCount {
                    expected: REQUIRED_PAIR_COUNT,
                    actual: REQUIRED_PAIR_COUNT + 1,
                }
            }
            2 => {
                candidates[1] = candidates[0];
                FleetCasmSetupAdmissionError::DuplicateReceipt(1)
            }
            _ => {
                candidates.swap(0, 1);
                FleetCasmSetupAdmissionError::SourceMismatch(0)
            }
        };
        assert_eq!(
            state
                .admit_expected(
                    expected.clone(),
                    binding(),
                    GENERATION,
                    &stages.iter().collect::<Vec<_>>(),
                    &candidates,
                )
                .unwrap_err(),
            expected_error
        );
        assert!(stages.iter().all(|stage| stage.current.get().is_none()));
    }
}

#[test]
fn source_stale_context_address_and_generation_mutations_fail_closed() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();
    for mutation in 0..6 {
        let (mut state, stages) = begin(&expected);
        let mut candidates = candidates(&expected, &stages);
        let expected_error = match mutation {
            0 => {
                candidates[0].source.component = "foreign";
                FleetCasmSetupAdmissionError::SourceMismatch(0)
            }
            1 => {
                stages[0].current.set(None);
                FleetCasmSetupAdmissionError::StaleReceipt(0)
            }
            2 => {
                let mut changed = candidates[0].receipt.0;
                changed.exec_context_token += 1;
                stages[0].publish(changed);
                candidates[0].receipt = stages[0].current.get().unwrap();
                FleetCasmSetupAdmissionError::ContextMismatch(0)
            }
            3 => {
                let mut changed = candidates[0].receipt.0;
                changed.staging_address += WORD_BYTES;
                stages[0].publish(changed);
                candidates[0].receipt = stages[0].current.get().unwrap();
                FleetCasmSetupAdmissionError::AddressMismatch(0)
            }
            4 => {
                let mut changed = candidates[0].receipt.0;
                changed.generation = 0;
                stages[0].publish(changed);
                candidates[0].receipt = stages[0].current.get().unwrap();
                FleetCasmSetupAdmissionError::ReceiptGenerationMismatch(0)
            }
            _ => {
                let mut changed = candidates[0].receipt.0;
                changed.staging_words += 1;
                stages[0].publish(changed);
                candidates[0].receipt = stages[0].current.get().unwrap();
                FleetCasmSetupAdmissionError::ReceiptGeometryMismatch(0)
            }
        };
        assert_eq!(
            state
                .admit_expected(
                    expected.clone(),
                    binding(),
                    GENERATION,
                    &stages.iter().collect::<Vec<_>>(),
                    &candidates,
                )
                .unwrap_err(),
            expected_error
        );
        assert!(stages.iter().all(|stage| stage.current.get().is_none()));
    }
}

#[test]
fn exact_stage_set_and_plan_binding_fail_closed() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();
    for mutation in 0..3 {
        let (mut state, stages) = begin(&expected);
        let candidates = candidates(&expected, &stages);
        let mut refs = stages.iter().collect::<Vec<_>>();
        let mut runtime_binding = binding();
        match mutation {
            0 => {
                refs.pop();
            }
            1 => refs[REQUIRED_PAIR_COUNT - 1] = refs[0],
            _ => runtime_binding.plan_identity = [8; 32],
        }
        assert_eq!(
            state
                .admit_expected(
                    expected.clone(),
                    runtime_binding,
                    GENERATION,
                    &refs,
                    &candidates,
                )
                .unwrap_err(),
            FleetCasmSetupAdmissionError::InvalidStatement
        );
        assert!(stages.iter().all(|stage| stage.current.get().is_none()));
    }
}

#[test]
fn validation_precedes_consumption_and_partial_consume_invalidates_every_stage() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();
    let (mut state, stages) = begin(&expected);
    stages[4].fail_consume.set(true);
    assert_eq!(
        admit(&mut state, &expected, &stages).unwrap_err(),
        FleetCasmSetupAdmissionError::ReceiptConsumeFailed(4)
    );
    assert_eq!(
        stages.iter().map(|stage| stage.consumes.get()).sum::<u32>(),
        4
    );
    assert!(stages.iter().all(|stage| stage.current.get().is_none()));
    assert_eq!(
        state.take(PLAN_ID, GENERATION).unwrap_err(),
        FleetCasmSetupAdmissionError::AdmissionUnavailable
    );

    let refs = stages.iter().collect::<Vec<_>>();
    state
        .begin_statement_with(PLAN_ID, GENERATION + 1, &refs)
        .unwrap();
    stages[4].fail_consume.set(false);
    let retry_generations = (0..REQUIRED_PAIR_COUNT)
        .map(|lane| GENERATION + 1 + u64::from(lane == 4))
        .collect::<Vec<_>>();
    for (lane, (stage, pair)) in stages.iter().zip(&expected.pairs).enumerate() {
        let mut retry = facts(pair);
        retry.generation = retry_generations[lane];
        stage.publish(retry);
    }
    state
        .admit_expected(
            expected.clone(),
            binding(),
            GENERATION + 1,
            &refs,
            &candidates(&expected, &stages),
        )
        .unwrap();
    assert_eq!(
        state
            .take(PLAN_ID, GENERATION + 1)
            .unwrap()
            .receipt_generations(),
        retry_generations
    );

    let (mut state, stages) = begin(&expected);
    let mut candidates = candidates(&expected, &stages);
    candidates[8].source.component = "invalid";
    assert!(state
        .admit_expected(
            expected.clone(),
            binding(),
            GENERATION,
            &stages.iter().collect::<Vec<_>>(),
            &candidates,
        )
        .is_err());
    assert_eq!(
        stages.iter().map(|stage| stage.consumes.get()).sum::<u32>(),
        0
    );
}

#[test]
fn new_statement_reset_and_take_generation_mismatch_destroy_authority() {
    let expected = expected(REQUIRED_PAIR_COUNT).unwrap();
    let (mut state, stages) = begin(&expected);
    admit(&mut state, &expected, &stages).unwrap();
    assert_eq!(
        state.take(PLAN_ID, GENERATION + 1).unwrap_err(),
        FleetCasmSetupAdmissionError::AdmissionUnavailable
    );

    let (mut state, stages) = begin(&expected);
    assert!(stages.iter().all(|stage| stage.current.get().is_some()));
    let refs = stages.iter().collect::<Vec<_>>();
    state
        .begin_statement_with(PLAN_ID, GENERATION + 1, &refs)
        .unwrap();
    assert!(stages.iter().all(|stage| stage.current.get().is_none()));
    state.reset_with(&refs);
    assert_eq!(
        state.take(PLAN_ID, GENERATION + 1).unwrap_err(),
        FleetCasmSetupAdmissionError::AdmissionUnavailable
    );
}
