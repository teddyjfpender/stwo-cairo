use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StatementHostReuse {
    pub operation: OpId,
    pub predecessor: ValueRange,
    pub destination: ValueRange,
}

pub(super) fn statement_host_reuses(
    compiled: &CompiledProof,
) -> Result<Vec<StatementHostReuse>, FleetCompileError> {
    compiled
        .operations()
        .iter()
        .filter_map(|operation| {
            let ExecutionPrimitive::StatementHostIngress {
                predecessor: Some(predecessor),
                ..
            } = &operation.primitive
            else {
                return None;
            };
            Some(
                statement_host_destination(compiled, operation.id).map(|destination| {
                    StatementHostReuse {
                        operation: operation.id,
                        predecessor: *predecessor,
                        destination,
                    }
                }),
            )
        })
        .collect()
}

pub(super) fn compile_components(
    value_count: usize,
    reuses: &[StatementHostReuse],
) -> Result<Vec<Option<usize>>, FleetCompileError> {
    let mut incoming = vec![None; value_count];
    let mut outgoing = vec![None; value_count];
    for (edge, reuse) in reuses.iter().enumerate() {
        let source = reuse.predecessor.version.0 as usize;
        let destination = reuse.destination.version.0 as usize;
        if source >= value_count
            || destination >= value_count
            || outgoing[source].replace(edge).is_some()
            || incoming[destination].replace(edge).is_some()
        {
            return Err(FleetCompileError::InvalidSemanticSchedule);
        }
    }

    let mut components = vec![None; value_count];
    let mut visited = vec![false; reuses.len()];
    let mut component = 0usize;
    for root in 0..value_count {
        if incoming[root].is_some() || outgoing[root].is_none() {
            continue;
        }
        let mut version = root;
        loop {
            if components[version].replace(component).is_some() {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            let Some(edge) = outgoing[version] else {
                break;
            };
            if visited[edge] {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
            visited[edge] = true;
            version = reuses[edge].destination.version.0 as usize;
            if incoming[version] != Some(edge) {
                return Err(FleetCompileError::InvalidSemanticSchedule);
            }
        }
        component = component
            .checked_add(1)
            .ok_or(FleetCompileError::SizeOverflow)?;
    }
    if visited.iter().any(|visited| !visited) {
        return Err(FleetCompileError::InvalidSemanticSchedule);
    }
    Ok(components)
}

pub(super) fn statement_host_destination(
    compiled: &CompiledProof,
    operation: OpId,
) -> Result<ValueRange, FleetCompileError> {
    let effect = compiled
        .effect_for(operation)
        .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
    match effect.accesses() {
        [crate::compiled_proof::EffectAccess::Write { destination }] => Ok(destination.value),
        _ => Err(FleetCompileError::InvalidSemanticSchedule),
    }
}
