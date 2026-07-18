use super::*;

pub(in crate::program_image::lower_compiled) fn resolve_static(
    request: StaticWrapperRequest<'_>,
    id: StaticCudaWrapperId,
    target_sm: u32,
    arena: &ProofArenaPlan,
    values: &adapter::SemanticValueMap,
) -> Result<Option<ResolvedStaticExecution>, InvocationShapeError> {
    let execution = match request {
        StaticWrapperRequest::ExecutionTable { lowered, stage } => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: execution_tables::project_static_wrapper(id, &linked, lowered, stage)?
                    .wrapper,
                invocation: lowered
                    .stages
                    .iter()
                    .find(|candidate| candidate.stage == stage)
                    .ok_or(InvocationShapeError::InvalidStructuredAbi)?
                    .invocation
                    .clone(),
            }
        }
        StaticWrapperRequest::MultiplicityClear(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: multiplicity_clear::project_static_wrapper(id, &linked, lowered)?.wrapper,
                invocation: lowered.invocation.clone(),
            }
        }
        StaticWrapperRequest::WitnessInputGather(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: witness_input_gather::project_static_wrapper(
                    arena, values, id, &linked, lowered,
                )?
                .wrapper,
                invocation: lowered.invocation.clone(),
            }
        }
        StaticWrapperRequest::WitnessInputSeed(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: witness_input_seed_compact::project_seed_static_wrapper(
                    id, &linked, lowered,
                )?
                .wrapper,
                invocation: lowered.invocation.clone(),
            }
        }
        StaticWrapperRequest::WitnessInputCompact(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            let projected =
                witness_input_seed_compact::project_compact_static_wrapper(id, &linked, lowered)?;
            ResolvedStaticExecution {
                wrapper: projected.wrapper,
                invocation: projected.invocation,
            }
        }
        StaticWrapperRequest::WitnessCasmScatter(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: witness_casm_input::project_static_wrapper(id, &linked, lowered)?.wrapper,
                invocation: lowered.invocation.clone(),
            }
        }
        StaticWrapperRequest::MultiplicityFeed(lowered) => {
            let Some(linked) = lowered
                .contract
                .bind_static_build(target_sm)
                .map_err(|_| InvocationShapeError::InvalidProductionBaseAuthority)?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: multiplicity_feed::project_static_wrapper(id, &linked, lowered)?.wrapper,
                invocation: lowered.invocation.clone(),
            }
        }
        StaticWrapperRequest::NativeBlakeGDirect(lowered) => {
            let Some(linked) =
                super::super::super::blake_g_direct_execution_authority::
                    NativeBlakeGDirectLinkedModuleAuthority::bind_linked(
                        &lowered.authority,
                        target_sm,
                    )?
            else {
                return Ok(None);
            };
            ResolvedStaticExecution {
                wrapper: static_wrapper_projection::blake_g_direct(id, &linked, lowered)?,
                invocation: static_wrapper_invocation::blake_g_direct(lowered)?,
            }
        }
        StaticWrapperRequest::NativeEcOp(lowered) => {
            let Some(linked) = super::super::super::ec_op_execution_authority::
                NativeEcOpLinkedModuleAuthority::bind_linked(&lowered.authority)?
            else {
                return Ok(None);
            };
            linked.validate_active_sm(target_sm)?;
            ResolvedStaticExecution {
                wrapper: static_wrapper_projection::ec_op(id, &linked, lowered)?,
                invocation: static_wrapper_invocation::ec_op(lowered)?,
            }
        }
    };
    Ok(Some(execution))
}
