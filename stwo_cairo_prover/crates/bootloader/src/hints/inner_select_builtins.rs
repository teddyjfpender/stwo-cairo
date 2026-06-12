use cairo_vm::Felt252;
use std::collections::HashMap;

use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::{
    get_ptr_from_var_name, insert_value_from_var_name,
};
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::ApTracking;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::vm_core::VirtualMachine;

use crate::hints::vars;

/// Sets ids.select_builtin to 1 if the first builtin should be selected and 0 otherwise.
///
/// Implements
/// # A builtin should be selected iff its encoding appears in the selected encodings list
/// # and the list wasn't exhausted.
/// # Note that testing inclusion by a single comparison is possible since the lists are sorted.
/// ids.select_builtin = int(
///   n_selected_builtins > 0 and memory[ids.selected_encodings] == memory[ids.all_encodings])
/// if ids.select_builtin:
///   n_selected_builtins = n_selected_builtins - 1
pub fn select_builtin(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let n_selected_builtins: usize = exec_scopes.get(vars::N_SELECTED_BUILTINS)?;

    let select_builtin = if n_selected_builtins == 0 {
        false
    } else {
        let selected_encodings =
            get_ptr_from_var_name("selected_encodings", vm, ids_data, ap_tracking)?;
        let all_encodings = get_ptr_from_var_name("all_encodings", vm, ids_data, ap_tracking)?;

        let selected_encoding = vm.get_integer(selected_encodings)?.into_owned();
        let builtin_encoding = vm.get_integer(all_encodings)?.into_owned();

        selected_encoding == builtin_encoding
    };

    let select_builtin_felt = Felt252::from(select_builtin);
    insert_value_from_var_name(
        "select_builtin",
        select_builtin_felt,
        vm,
        ids_data,
        ap_tracking,
    )?;

    if select_builtin {
        exec_scopes.insert_value(vars::N_SELECTED_BUILTINS, n_selected_builtins - 1);
    }

    Ok(())
}

