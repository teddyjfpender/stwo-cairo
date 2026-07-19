//! Single-use runtime admission for eagerly completed CASM setup pairs.
//!
//! The eager resident upload/scatter path may complete before the fleet
//! scheduler starts. This module authenticates that work against the exact
//! worker install and publishes a one-shot disposition for the scheduler. It
//! never removes schedule entries: liveness and barrier cursors must advance
//! over `PreExecuted` operations exactly as they do over launched operations.

use std::collections::BTreeSet;

use stwo_backend_cuda::{
    PreparedWitnessCasmInputError, PreparedWitnessCasmInputStage, WitnessCasmInputAbi,
    WitnessCasmInputIngressReceipt, WitnessCasmInputRowDomain, WITNESS_CASM_STATE_WORDS,
};

use super::{
    FleetInstallWindow, FleetProofPlan, FleetWorkerExecution, FleetWorkerInstallPlan, WorkerId,
};
use crate::compiled_proof::{
    ExecutionPrimitive, OpId, StatementHostEncoding, StatementHostPart, StatementHostSource,
    StatementHostSourceKind, ValueRange,
};

const REQUIRED_PAIR_COUNT: usize = 9;
const REQUIRED_OPERATION_COUNT: usize = REQUIRED_PAIR_COUNT * 2;
const WORD_BYTES: usize = core::mem::size_of::<u32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetCasmRuntimeBinding {
    plan_identity: [u8; 32],
    worker: WorkerId,
    slab_base_address: usize,
    exec_context_token: u64,
}

impl FleetCasmRuntimeBinding {
    pub const fn new(
        plan_identity: [u8; 32],
        worker: WorkerId,
        slab_base_address: usize,
        exec_context_token: u64,
    ) -> Self {
        Self {
            plan_identity,
            worker,
            slab_base_address,
            exec_context_token,
        }
    }

    pub const fn plan_identity(self) -> [u8; 32] {
        self.plan_identity
    }

    pub const fn worker(self) -> WorkerId {
        self.worker
    }

    pub const fn slab_base_address(self) -> usize {
        self.slab_base_address
    }

    pub const fn exec_context_token(self) -> u64 {
        self.exec_context_token
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FleetCasmSetupSource<'a> {
    pub producer_ordinal: u32,
    pub component: &'a str,
    pub part: StatementHostPart,
}

pub struct FleetCasmSetupIngress<'stage, 'arena> {
    pub source: FleetCasmSetupSource<'stage>,
    pub stage: &'stage PreparedWitnessCasmInputStage<'arena>,
    pub receipt: WitnessCasmInputIngressReceipt,
}

impl<'stage, 'arena> FleetCasmSetupIngress<'stage, 'arena> {
    pub const fn new(
        source: FleetCasmSetupSource<'stage>,
        stage: &'stage PreparedWitnessCasmInputStage<'arena>,
        receipt: WitnessCasmInputIngressReceipt,
    ) -> Self {
        Self {
            source,
            stage,
            receipt,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FleetCasmSetupDisposition {
    Execute,
    PreExecuted,
}

#[derive(Debug, Eq, PartialEq)]
pub enum FleetCasmSetupAdmissionError {
    InvalidStatement,
    InvalidWorkerInstall,
    InvalidPair(usize),
    PairCount { expected: usize, actual: usize },
    DuplicateOperation(OpId),
    ReceiptCount { expected: usize, actual: usize },
    DuplicateReceipt(usize),
    SourceMismatch(usize),
    ReceiptGeometryMismatch(usize),
    ReceiptGenerationMismatch(usize),
    StaleReceipt(usize),
    ContextMismatch(usize),
    AddressMismatch(usize),
    ReceiptConsumeFailed(usize),
    AdmissionUnavailable,
    ExecutionDrift(OpId),
    MissingPreExecutedOperation(OpId),
    SizeOverflow,
}

impl core::fmt::Display for FleetCasmSetupAdmissionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fleet CASM setup admission rejected: {self:?}")
    }
}

impl std::error::Error for FleetCasmSetupAdmissionError {}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedPair {
    source: StatementHostSource,
    window: FleetInstallWindow,
    ingress: FleetWorkerExecution,
    scatter: FleetWorkerExecution,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedSetup {
    plan_identity: [u8; 32],
    worker: WorkerId,
    pairs: Vec<ExpectedPair>,
}

/// One authenticated, single-use scheduler discharge.
///
/// `classify_install` returns one disposition per original execution. Callers
/// must still advance their schedule/liveness/barrier cursors for
/// `PreExecuted`; only the launch itself is discharged.
#[derive(Debug)]
pub struct FleetCasmSetupAdmission {
    plan_identity: [u8; 32],
    statement_generation: u64,
    worker: WorkerId,
    receipt_generations: Vec<u64>,
    pre_executed: Vec<FleetWorkerExecution>,
}

impl FleetCasmSetupAdmission {
    pub const fn plan_identity(&self) -> [u8; 32] {
        self.plan_identity
    }

    pub const fn statement_generation(&self) -> u64 {
        self.statement_generation
    }

    pub const fn worker(&self) -> WorkerId {
        self.worker
    }

    pub fn receipt_generations(&self) -> &[u64] {
        &self.receipt_generations
    }

    pub fn pre_executed_operations(&self) -> impl Iterator<Item = OpId> + '_ {
        self.pre_executed
            .iter()
            .map(|execution| execution.operation)
    }

    pub fn classify_install(
        self,
        install: &FleetWorkerInstallPlan,
    ) -> Result<Vec<FleetCasmSetupDisposition>, FleetCasmSetupAdmissionError> {
        if install.plan_identity() != self.plan_identity
            || install.target().worker != self.worker
            || self.pre_executed.len() != REQUIRED_OPERATION_COUNT
        {
            return Err(FleetCasmSetupAdmissionError::InvalidWorkerInstall);
        }
        self.classify_executions(install.executions())
    }

    fn classify_executions(
        &self,
        executions: &[FleetWorkerExecution],
    ) -> Result<Vec<FleetCasmSetupDisposition>, FleetCasmSetupAdmissionError> {
        let mut seen = BTreeSet::new();
        let dispositions = executions
            .iter()
            .map(|execution| {
                let Some(expected) = self
                    .pre_executed
                    .iter()
                    .find(|expected| expected.operation == execution.operation)
                else {
                    return Ok(FleetCasmSetupDisposition::Execute);
                };
                if expected != execution {
                    return Err(FleetCasmSetupAdmissionError::ExecutionDrift(
                        execution.operation,
                    ));
                }
                if !seen.insert(execution.operation) {
                    return Err(FleetCasmSetupAdmissionError::DuplicateOperation(
                        execution.operation,
                    ));
                }
                Ok(FleetCasmSetupDisposition::PreExecuted)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for expected in &self.pre_executed {
            if !seen.contains(&expected.operation) {
                return Err(FleetCasmSetupAdmissionError::MissingPreExecutedOperation(
                    expected.operation,
                ));
            }
        }
        Ok(dispositions)
    }
}

#[derive(Default)]
pub struct FleetCasmSetupAdmissionState {
    statement: Option<([u8; 32], u64)>,
    admission: Option<FleetCasmSetupAdmission>,
}

impl FleetCasmSetupAdmissionState {
    /// Start a new statement before any upload. Every prior pending or
    /// published stage receipt is invalidated and any untaken admission dies.
    pub fn begin_statement<'arena>(
        &mut self,
        plan_identity: [u8; 32],
        statement_generation: u64,
        stages: &[&PreparedWitnessCasmInputStage<'arena>],
    ) -> Result<(), FleetCasmSetupAdmissionError> {
        self.begin_statement_with(plan_identity, statement_generation, stages)
    }

    pub fn reset<'arena>(&mut self, stages: &[&PreparedWitnessCasmInputStage<'arena>]) {
        self.reset_with(stages);
    }

    /// Validate all nine receipts before consuming any. Publication occurs
    /// only after every single-use stage consume succeeds.
    pub fn admit(
        &mut self,
        plan: &FleetProofPlan,
        install: &FleetWorkerInstallPlan,
        binding: FleetCasmRuntimeBinding,
        statement_generation: u64,
        stages: &[&PreparedWitnessCasmInputStage<'_>],
        ingress: &[FleetCasmSetupIngress<'_, '_>],
    ) -> Result<(), FleetCasmSetupAdmissionError> {
        let candidates = ingress
            .iter()
            .map(|candidate| IngressCandidate {
                source: candidate.source,
                stage: candidate.stage,
                receipt: candidate.receipt,
            })
            .collect::<Vec<_>>();
        let expected = match expected_setup(plan, install) {
            Ok(expected) => expected,
            Err(error) => {
                self.fail(stages, &candidates);
                return Err(error);
            }
        };
        self.admit_expected(expected, binding, statement_generation, stages, &candidates)
    }

    /// Atomically move out the exact admission. A mismatch destroys it.
    pub fn take(
        &mut self,
        plan_identity: [u8; 32],
        statement_generation: u64,
    ) -> Result<FleetCasmSetupAdmission, FleetCasmSetupAdmissionError> {
        if self.statement != Some((plan_identity, statement_generation))
            || self.admission.as_ref().is_none_or(|admission| {
                admission.plan_identity != plan_identity
                    || admission.statement_generation != statement_generation
            })
        {
            self.statement = None;
            self.admission = None;
            return Err(FleetCasmSetupAdmissionError::AdmissionUnavailable);
        }
        self.statement = None;
        self.admission
            .take()
            .ok_or(FleetCasmSetupAdmissionError::AdmissionUnavailable)
    }

    fn begin_statement_with<S: StageAuthority>(
        &mut self,
        plan_identity: [u8; 32],
        statement_generation: u64,
        stages: &[&S],
    ) -> Result<(), FleetCasmSetupAdmissionError> {
        self.reset_with(stages);
        if plan_identity == [0; 32] || statement_generation == 0 {
            return Err(FleetCasmSetupAdmissionError::InvalidStatement);
        }
        self.statement = Some((plan_identity, statement_generation));
        Ok(())
    }

    fn reset_with<S: StageAuthority>(&mut self, stages: &[&S]) {
        stages.iter().for_each(|stage| stage.invalidate());
        self.statement = None;
        self.admission = None;
    }

    fn admit_expected<S: StageAuthority>(
        &mut self,
        expected: ExpectedSetup,
        binding: FleetCasmRuntimeBinding,
        statement_generation: u64,
        stages: &[&S],
        ingress: &[IngressCandidate<'_, S>],
    ) -> Result<(), FleetCasmSetupAdmissionError> {
        self.admission = None;
        let validated = validate_receipts(
            self.statement,
            &expected,
            binding,
            statement_generation,
            ingress,
        );
        if let Err(error) = validated {
            self.fail(stages, ingress);
            return Err(error);
        }
        if !exact_stage_set(stages, ingress) {
            self.fail(stages, ingress);
            return Err(FleetCasmSetupAdmissionError::InvalidStatement);
        }
        let receipt_generations = ingress
            .iter()
            .map(|candidate| candidate.stage.facts(candidate.receipt).generation)
            .collect();

        for (index, candidate) in ingress.iter().enumerate() {
            if candidate.stage.consume(candidate.receipt).is_err() {
                self.fail(stages, ingress);
                return Err(FleetCasmSetupAdmissionError::ReceiptConsumeFailed(index));
            }
        }
        self.admission = Some(FleetCasmSetupAdmission {
            plan_identity: expected.plan_identity,
            statement_generation,
            worker: expected.worker,
            receipt_generations,
            pre_executed: expected
                .pairs
                .into_iter()
                .flat_map(|pair| [pair.ingress, pair.scatter])
                .collect(),
        });
        Ok(())
    }

    fn fail<S: StageAuthority>(&mut self, stages: &[&S], ingress: &[IngressCandidate<'_, S>]) {
        stages.iter().for_each(|stage| stage.invalidate());
        ingress
            .iter()
            .for_each(|candidate| candidate.stage.invalidate());
        self.statement = None;
        self.admission = None;
    }
}

struct IngressCandidate<'a, S: StageAuthority> {
    source: FleetCasmSetupSource<'a>,
    stage: &'a S,
    receipt: S::Receipt,
}

impl<S: StageAuthority> Copy for IngressCandidate<'_, S> {}

impl<S: StageAuthority> Clone for IngressCandidate<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReceiptFacts {
    contract_identity: [u8; 32],
    arena_identity: usize,
    exec_context_token: u64,
    staging_address: usize,
    staging_words: usize,
    abi: WitnessCasmInputAbi,
    row_domain: WitnessCasmInputRowDomain,
    state_words_per_row: usize,
    real_rows: usize,
    consumer_rows: usize,
    include_iota: bool,
    generation: u64,
}

trait StageAuthority {
    type Receipt: Copy + Eq;

    fn facts(&self, receipt: Self::Receipt) -> ReceiptFacts;
    fn is_current(&self, receipt: &Self::Receipt) -> bool;
    fn consume(&self, receipt: Self::Receipt) -> Result<(), PreparedWitnessCasmInputError>;
    fn invalidate(&self);
}

impl StageAuthority for PreparedWitnessCasmInputStage<'_> {
    type Receipt = WitnessCasmInputIngressReceipt;

    fn facts(&self, receipt: Self::Receipt) -> ReceiptFacts {
        ReceiptFacts {
            contract_identity: receipt.contract_identity(),
            arena_identity: receipt.arena_identity(),
            exec_context_token: receipt.exec_context_token(),
            staging_address: receipt.staging_address(),
            staging_words: receipt.staging_words(),
            abi: receipt.abi(),
            row_domain: receipt.row_domain(),
            state_words_per_row: receipt.state_words_per_row(),
            real_rows: receipt.real_rows(),
            consumer_rows: receipt.consumer_rows(),
            include_iota: receipt.include_iota(),
            generation: receipt.generation(),
        }
    }

    fn is_current(&self, receipt: &Self::Receipt) -> bool {
        self.ingress_is_current(receipt)
    }

    fn consume(&self, receipt: Self::Receipt) -> Result<(), PreparedWitnessCasmInputError> {
        self.consume_ingress_receipt(receipt)
    }

    fn invalidate(&self) {
        self.invalidate_ingress_receipt();
    }
}

fn exact_stage_set<S: StageAuthority>(stages: &[&S], ingress: &[IngressCandidate<'_, S>]) -> bool {
    stages.len() == REQUIRED_PAIR_COUNT
        && ingress.len() == REQUIRED_PAIR_COUNT
        && stages.iter().enumerate().all(|(index, stage)| {
            !stages[..index]
                .iter()
                .any(|prior| core::ptr::eq(*prior, *stage))
                && ingress
                    .iter()
                    .filter(|candidate| core::ptr::eq(candidate.stage, *stage))
                    .count()
                    == 1
        })
}

fn validate_receipts<S: StageAuthority>(
    statement: Option<([u8; 32], u64)>,
    expected: &ExpectedSetup,
    binding: FleetCasmRuntimeBinding,
    statement_generation: u64,
    ingress: &[IngressCandidate<'_, S>],
) -> Result<(), FleetCasmSetupAdmissionError> {
    if statement != Some((expected.plan_identity, statement_generation))
        || binding.plan_identity != expected.plan_identity
        || binding.worker != expected.worker
        || binding.slab_base_address == 0
        || binding.exec_context_token == 0
        || statement_generation == 0
    {
        return Err(FleetCasmSetupAdmissionError::InvalidStatement);
    }
    if ingress.len() != REQUIRED_PAIR_COUNT {
        return Err(FleetCasmSetupAdmissionError::ReceiptCount {
            expected: REQUIRED_PAIR_COUNT,
            actual: ingress.len(),
        });
    }
    for (index, candidate) in ingress.iter().enumerate() {
        let facts = candidate.stage.facts(candidate.receipt);
        if ingress[..index].iter().any(|prior| {
            prior.source == candidate.source
                || core::ptr::eq(prior.stage, candidate.stage)
                || prior.stage.facts(prior.receipt) == facts
        }) {
            return Err(FleetCasmSetupAdmissionError::DuplicateReceipt(index));
        }
        let pair = &expected.pairs[index];
        if candidate.source.producer_ordinal != pair.source.producer_ordinal
            || candidate.source.component != pair.source.component.as_ref()
            || candidate.source.part != pair.source.part
        {
            return Err(FleetCasmSetupAdmissionError::SourceMismatch(index));
        }
        if facts.contract_identity != pair.source.casm_contract_identity
            || facts.staging_words != pair.source.words
            || facts.abi != WitnessCasmInputAbi::RowMajorStateScatterV1
            || facts.row_domain != WitnessCasmInputRowDomain::RealPrefixWithRowZeroPaddingV1
            || facts.state_words_per_row != WITNESS_CASM_STATE_WORDS
            || facts.real_rows != pair.source.real_rows
            || facts.consumer_rows != pair.source.consumer_rows
            || facts.include_iota != pair.source.include_iota
            || facts
                .staging_words
                .checked_mul(WORD_BYTES)
                .ok_or(FleetCasmSetupAdmissionError::SizeOverflow)?
                != pair.window.bytes
        {
            return Err(FleetCasmSetupAdmissionError::ReceiptGeometryMismatch(index));
        }
        if facts.generation == 0 {
            return Err(FleetCasmSetupAdmissionError::ReceiptGenerationMismatch(
                index,
            ));
        }
        if !candidate.stage.is_current(&candidate.receipt) {
            return Err(FleetCasmSetupAdmissionError::StaleReceipt(index));
        }
        if facts.arena_identity != binding.slab_base_address
            || facts.exec_context_token != binding.exec_context_token
        {
            return Err(FleetCasmSetupAdmissionError::ContextMismatch(index));
        }
        let expected_address = binding
            .slab_base_address
            .checked_add(pair.window.slab_offset_bytes)
            .and_then(|address| address.checked_add(pair.window.offset_bytes))
            .ok_or(FleetCasmSetupAdmissionError::SizeOverflow)?;
        if facts.staging_address != expected_address {
            return Err(FleetCasmSetupAdmissionError::AddressMismatch(index));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OperationAuthority {
    Ingress {
        source: StatementHostSource,
        predecessor: Option<ValueRange>,
    },
    Scatter {
        contract_identity: [u8; 32],
    },
    Other,
}

fn expected_setup(
    plan: &FleetProofPlan,
    install: &FleetWorkerInstallPlan,
) -> Result<ExpectedSetup, FleetCasmSetupAdmissionError> {
    let worker = install.target().worker;
    let exact_install = plan
        .worker_install_plan(worker)
        .map_err(|_| FleetCasmSetupAdmissionError::InvalidWorkerInstall)?;
    if plan.identity() != install.plan_identity()
        || worker != plan.placement().topology.coordinator
        || install.coordinator().is_none()
        || exact_install != *install
    {
        return Err(FleetCasmSetupAdmissionError::InvalidWorkerInstall);
    }
    expected_setup_from(plan.identity(), worker, install.executions(), |operation| {
        let operation = plan
            .compiled()
            .operation(operation)
            .ok_or(FleetCasmSetupAdmissionError::InvalidWorkerInstall)?;
        Ok(match &operation.primitive {
            ExecutionPrimitive::StatementHostIngress {
                source,
                predecessor,
            } => OperationAuthority::Ingress {
                source: source.clone(),
                predecessor: *predecessor,
            },
            ExecutionPrimitive::StaticCudaWrapper { wrapper } => {
                let authority = plan
                    .compiled()
                    .static_wrapper(*wrapper)
                    .ok_or(FleetCasmSetupAdmissionError::InvalidWorkerInstall)?;
                OperationAuthority::Scatter {
                    contract_identity: *authority.aggregate_contract_identity(),
                }
            }
            _ => OperationAuthority::Other,
        })
    })
}

fn expected_setup_from(
    plan_identity: [u8; 32],
    worker: WorkerId,
    executions: &[FleetWorkerExecution],
    mut authority: impl FnMut(OpId) -> Result<OperationAuthority, FleetCasmSetupAdmissionError>,
) -> Result<ExpectedSetup, FleetCasmSetupAdmissionError> {
    let mut pairs = Vec::new();
    let mut operations = BTreeSet::new();
    let mut index = 0usize;
    while index < executions.len() {
        let ingress = &executions[index];
        let projected = ingress
            .executables
            .iter()
            .filter_map(|executable| executable.statement_host_ingress.as_ref())
            .collect::<Vec<_>>();
        if projected.is_empty() {
            index += 1;
            continue;
        }
        let pair_index = pairs.len();
        let [projected] = projected.as_slice() else {
            return Err(FleetCasmSetupAdmissionError::InvalidPair(pair_index));
        };
        let OperationAuthority::Ingress {
            source,
            predecessor,
        } = authority(ingress.operation)?
        else {
            return Err(FleetCasmSetupAdmissionError::InvalidPair(pair_index));
        };
        let valid_ingress_effect = matches!(
            ingress.executables.first().map(|executable| executable.effects.as_slice()),
            Some([effect])
                if effect.source.is_none()
                    && effect.destination == Some(projected.destination)
                    && effect.window == projected.window
        );
        if ingress.executables.len() != 1
            || ingress.executables[0].child_ordinal.is_some()
            || source.kind != StatementHostSourceKind::WitnessCasm
            || source.encoding != StatementHostEncoding::RowMajorU32
            || projected.source != source
            || projected.predecessor != predecessor
            || !valid_ingress_effect
        {
            return Err(FleetCasmSetupAdmissionError::InvalidPair(pair_index));
        }
        let scatter = executions
            .get(index + 1)
            .ok_or(FleetCasmSetupAdmissionError::InvalidPair(pair_index))?;
        let OperationAuthority::Scatter { contract_identity } = authority(scatter.operation)?
        else {
            return Err(FleetCasmSetupAdmissionError::InvalidPair(pair_index));
        };
        let adjacent = ingress
            .operation
            .0
            .checked_add(1)
            .is_some_and(|operation| scatter.operation == OpId(operation));
        let staging_reads = scatter
            .executables
            .first()
            .into_iter()
            .flat_map(|executable| &executable.effects)
            .filter(|effect| {
                effect.source == Some(projected.destination) && effect.window == projected.window
            })
            .count();
        if !adjacent
            || scatter.executables.len() != 1
            || scatter.executables[0].child_ordinal.is_some()
            || scatter.executables[0].statement_host_ingress.is_some()
            || staging_reads != 1
            || contract_identity != source.casm_contract_identity
        {
            return Err(FleetCasmSetupAdmissionError::InvalidPair(pair_index));
        }
        for operation in [ingress.operation, scatter.operation] {
            if !operations.insert(operation) {
                return Err(FleetCasmSetupAdmissionError::DuplicateOperation(operation));
            }
        }
        pairs.push(ExpectedPair {
            source,
            window: projected.window,
            ingress: ingress.clone(),
            scatter: scatter.clone(),
        });
        index += 2;
    }
    if pairs.len() != REQUIRED_PAIR_COUNT {
        return Err(FleetCasmSetupAdmissionError::PairCount {
            expected: REQUIRED_PAIR_COUNT,
            actual: pairs.len(),
        });
    }
    Ok(ExpectedSetup {
        plan_identity,
        worker,
        pairs,
    })
}

#[cfg(test)]
#[path = "casm_setup_admission/tests.rs"]
mod tests;
