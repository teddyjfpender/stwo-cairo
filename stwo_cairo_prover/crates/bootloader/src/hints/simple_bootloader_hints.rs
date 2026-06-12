use std::collections::HashMap;

use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::{
    get_integer_from_var_name, get_ptr_from_var_name, insert_value_from_var_name,
    insert_value_into_ap,
};
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::ApTracking;
use cairo_vm::types::errors::math_errors::MathError;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::vm_core::VirtualMachine;
use cairo_vm::Felt252;
use num_traits::ToPrimitive;
use starknet_types_core::felt::NonZeroFelt;

use crate::hints::fact_topologies::FactTopology;
use crate::hints::types::SimpleBootloaderInput;
use crate::hints::vars;

/// Implements
/// n_tasks = len(simple_bootloader_input.tasks)
/// memory[ids.output_ptr] = n_tasks
///
/// # Task range checks are located right after simple bootloader validation range checks, and
/// # this is validated later in this function.
/// ids.task_range_check_ptr = ids.range_check_ptr + ids.BuiltinData.SIZE * n_tasks
///
/// # A list of fact_toplogies that instruct how to generate the fact from the program output
/// # for each task.
/// fact_topologies = []
pub fn prepare_task_range_checks(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    // n_tasks = len(simple_bootloader_input.tasks)
    let simple_bootloader_input: &SimpleBootloaderInput =
        exec_scopes.get_ref(vars::SIMPLE_BOOTLOADER_INPUT)?;
    let n_tasks = simple_bootloader_input.tasks.len();

    // memory[ids.output_ptr] = n_tasks
    let output_ptr = get_ptr_from_var_name("output_ptr", vm, ids_data, ap_tracking)?;
    vm.insert_value(output_ptr, Felt252::from(n_tasks))?;

    // ids.task_range_check_ptr = ids.range_check_ptr + ids.BuiltinData.SIZE * n_tasks
    // BuiltinData is a struct with 11 members defined in execute_task.cairo (v0.13.3):
    // output, pedersen, range_check, ecdsa, bitwise, ec_op, keccak, poseidon,
    // range_check96, add_mod, mul_mod.
    const BUILTIN_DATA_SIZE: usize = 11;
    let range_check_ptr = get_ptr_from_var_name("range_check_ptr", vm, ids_data, ap_tracking)?;
    let task_range_check_ptr = (range_check_ptr + BUILTIN_DATA_SIZE * n_tasks)?;
    insert_value_from_var_name(
        "task_range_check_ptr",
        task_range_check_ptr,
        vm,
        ids_data,
        ap_tracking,
    )?;

    // fact_topologies = []
    let fact_topologies = Vec::<FactTopology>::new();
    exec_scopes.insert_value(vars::FACT_TOPOLOGIES, fact_topologies);

    Ok(())
}

/// Implements
/// %{ tasks = simple_bootloader_input.tasks %}
pub fn set_tasks_variable(exec_scopes: &mut ExecutionScopes) -> Result<(), HintError> {
    let simple_bootloader_input: &SimpleBootloaderInput =
        exec_scopes.get_ref(vars::SIMPLE_BOOTLOADER_INPUT)?;
    exec_scopes.insert_value(vars::TASKS, simple_bootloader_input.tasks.clone());

    Ok(())
}

/// Implements %{ ids.num // 2 %}
/// (compiled to %{ memory[ap] = to_felt_or_relocatable(ids.num // 2) %}).
pub fn divide_num_by_2(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let felt = get_integer_from_var_name("num", vm, ids_data, ap_tracking)?;
    // Unwrapping is safe in this context, 2 != 0
    let two = NonZeroFelt::try_from(Felt252::from(2)).unwrap();
    let felt_divided_by_2 = felt.floor_div(&two);

    insert_value_into_ap(vm, felt_divided_by_2)?;

    Ok(())
}

/// Implements %{ 0 %} (compiled to %{ memory[ap] = to_felt_or_relocatable(0) %}).
///
/// Stores 0 in the AP and returns.
/// Used as `tempvar use_poseidon = nondet %{ 0 %}`.
pub fn set_ap_to_zero(vm: &mut VirtualMachine) -> Result<(), HintError> {
    insert_value_into_ap(vm, Felt252::from(0))?;
    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.simple_bootloader.objects import Task
///
/// # Pass current task to execute_task.
/// task_id = len(simple_bootloader_input.tasks) - ids.n_tasks
/// task = simple_bootloader_input.tasks[task_id].load_task()
pub fn set_current_task(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let simple_bootloader_input: &SimpleBootloaderInput =
        exec_scopes.get_ref(vars::SIMPLE_BOOTLOADER_INPUT)?;
    let n_tasks_felt = get_integer_from_var_name("n_tasks", vm, ids_data, ap_tracking)?;
    let n_tasks = n_tasks_felt
        .to_usize()
        .ok_or(MathError::Felt252ToUsizeConversion(Box::new(n_tasks_felt)))?;

    let task_id = simple_bootloader_input.tasks.len() - n_tasks;
    // TODO: it's still unclear how we need to model TaskSpec/Task objects.
    //       Check if we need to keep TaskSpec, or if it needs to be implemented as a trait, etc.
    let task_spec = simple_bootloader_input.tasks[task_id].clone();
    let task = task_spec.load_task();

    exec_scopes.insert_value(vars::TASK, task.clone());
    exec_scopes.insert_value(vars::TASK_USE_POSEIDON, task_spec.use_poseidon);

    Ok(())
}

