use std::any::Any;
use std::collections::HashMap;

use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::get_integer_from_var_name;
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::ApTracking;
use cairo_vm::types::errors::math_errors::MathError;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::vm_core::VirtualMachine;
use num_traits::ToPrimitive;

use crate::hints::vars;

/// Implements
/// %{ vm_enter_scope({'n_selected_builtins': ids.n_selected_builtins}) %}
pub fn select_builtins_enter_scope(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let n_selected_builtins =
        get_integer_from_var_name(vars::N_SELECTED_BUILTINS, vm, ids_data, ap_tracking)?;
    let n_selected_builtins =
        n_selected_builtins
            .to_usize()
            .ok_or(MathError::Felt252ToUsizeConversion(Box::new(
                n_selected_builtins,
            )))?;

    exec_scopes.enter_scope(HashMap::from([(
        vars::N_SELECTED_BUILTINS.to_string(),
        Box::new(n_selected_builtins) as Box<dyn Any>,
    )]));

    Ok(())
}

