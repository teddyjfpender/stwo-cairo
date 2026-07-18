use std::collections::BTreeMap;

use stwo_backend_cuda::{
    witness_casm_input_requirements, WitnessCasmInputContract, WITNESS_CASM_STATE_WORDS,
};

use super::*;
use crate::compiled_proof::{StatementHostEncoding, StatementHostSource, StatementHostSourceKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FleetStatementHostIngress {
    pub source: StatementHostSource,
    pub predecessor: Option<ValueRange>,
    pub destination: ValueRange,
    pub window: FleetInstallWindow,
}

pub(super) fn project(
    plan: &FleetProofPlan,
    worker: WorkerId,
    operation: &crate::compiled_proof::OpNode,
    execution: &FleetOperationExecution,
    primitive: &ExecutionPrimitive,
    storages: &BTreeMap<StorageId, FleetWorkerStorage>,
) -> Result<Option<FleetStatementHostIngress>, FleetWorkerInstallError> {
    let ExecutionPrimitive::StatementHostIngress {
        source,
        predecessor,
    } = primitive
    else {
        return Ok(None);
    };
    let invalid = || FleetWorkerInstallError::InvalidStatementHostIngress(operation.id);
    if worker != plan.placement().topology.coordinator
        || execution.domain != OperationDomain::Monolithic
        || !matches!(source.kind, StatementHostSourceKind::WitnessCasm)
        || !matches!(source.encoding, StatementHostEncoding::RowMajorU32)
    {
        return Err(invalid());
    }
    let requirements = witness_casm_input_requirements(source.real_rows, source.include_iota)
        .map_err(|_| invalid())?;
    let contract = WitnessCasmInputContract::compile(&requirements).map_err(|_| invalid())?;
    if source.words != requirements.staging_words
        || source.words
            != source
                .real_rows
                .checked_mul(WITNESS_CASM_STATE_WORDS)
                .ok_or_else(invalid)?
        || source.consumer_rows != requirements.consumer_rows
        || source.casm_contract_identity != contract.identity()
    {
        return Err(invalid());
    }

    let effect = plan
        .compiled()
        .effect_for(operation.id)
        .ok_or_else(invalid)?;
    let [crate::compiled_proof::EffectAccess::Write { destination }] = effect.accesses() else {
        return Err(invalid());
    };
    let destination =
        validate::projected_range(plan.compiled(), operation, *destination, execution)
            .map_err(|_| invalid())?;
    let window = resolve_window(plan, worker, destination, storages).map_err(|_| invalid())?;
    if let Some(predecessor) = predecessor {
        let previous =
            resolve_window(plan, worker, *predecessor, storages).map_err(|_| invalid())?;
        if previous != window {
            return Err(invalid());
        }
    }
    Ok(Some(FleetStatementHostIngress {
        source: source.clone(),
        predecessor: *predecessor,
        destination,
        window,
    }))
}
