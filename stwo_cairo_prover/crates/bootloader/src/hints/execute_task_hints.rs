use std::any::Any;
use std::collections::HashMap;

use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::{
    get_integer_from_var_name, get_ptr_from_var_name, get_relocatable_from_var_name, insert_value_from_var_name, insert_value_into_ap,
};
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::{ApTracking, Identifier};
use cairo_vm::types::builtin_name::BuiltinName;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::types::relocatable::Relocatable;
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::errors::memory_errors::MemoryError;
use cairo_vm::vm::runners::builtin_runner::{OutputBuiltinRunner, OutputBuiltinState};
use cairo_vm::vm::runners::cairo_pie::{CairoPie, StrippedProgram};
use cairo_vm::vm::vm_core::VirtualMachine;
use cairo_vm::{any_box, Felt252};
use starknet_crypto::FieldElement;

use crate::hints::fact_topologies::{get_task_fact_topology, FactTopology};
use crate::hints::load_cairo_pie::load_cairo_pie;
use crate::hints::program_hash::compute_program_hash_chain;
use crate::hints::program_loader::ProgramLoader;
use crate::hints::types::{BootloaderVersion, ProgramIdentifiers, Task};
use crate::hints::vars;

fn get_program_from_task(task: &Task) -> Result<StrippedProgram, HintError> {
    task.get_program()
        .map_err(|e| HintError::CustomHint(e.to_string().into_boxed_str()))
}

/// Implements %{ ids.program_data_ptr = program_data_base = segments.add() %}.
///
/// Creates a new segment to store the program data.
pub fn allocate_program_data_segment(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let program_data_segment = vm.add_memory_segment();
    exec_scopes.insert_value(vars::PROGRAM_DATA_BASE, program_data_segment);
    insert_value_from_var_name(
        "program_data_ptr",
        program_data_segment,
        vm,
        ids_data,
        ap_tracking,
    )?;

    Ok(())
}

fn field_element_to_felt(field_element: FieldElement) -> Felt252 {
    let bytes = field_element.to_bytes_be();
    Felt252::from_bytes_be(&bytes)
}

/// Implements
///
/// from starkware.cairo.bootloaders.simple_bootloader.utils import load_program
///
/// # Call load_program to load the program header and code to memory.
/// program_address, program_data_size = load_program(
///     task=task, memory=memory, program_header=ids.program_header,
///     builtins_offset=ids.ProgramHeader.builtin_list)
/// segments.finalize(program_data_base.segment_index, program_data_size)
pub fn load_program_hint(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let program_data_base: Relocatable = exec_scopes.get(vars::PROGRAM_DATA_BASE)?;
    let task: Task = exec_scopes.get(vars::TASK)?;
    let program = get_program_from_task(&task)?;

    let program_header_ptr = get_ptr_from_var_name("program_header", vm, ids_data, ap_tracking)?;

    // Offset of the builtin_list field in `ProgramHeader`, cf. execute_task.cairo
    let builtins_offset = 4;
    let mut program_loader = ProgramLoader::new(vm, builtins_offset);
    let bootloader_version: BootloaderVersion = 0;
    let loaded_program = program_loader
        .load_program(program_header_ptr, &program, Some(bootloader_version))
        .map_err(Into::<HintError>::into)?;

    vm.segments.finalize(
        Some(loaded_program.size),
        program_data_base.segment_index as usize,
        None,
    );

    exec_scopes.insert_value(vars::PROGRAM_ADDRESS, loaded_program.code_address);

    Ok(())
}

/// Implements the v0.14 macro hint `DETERMINE_USE_PREV_HASH`
/// (execute_task.cairo:99): sets the `use_prev_hash` local from the scope var
/// computed in `set_current_task`. Always 0 for a single PIE task.
pub fn determine_use_prev_hash(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let use_prev_hash: i32 = exec_scopes.get(vars::USE_PREV_HASH).unwrap_or(0);
    insert_value_from_var_name(
        "use_prev_hash",
        Felt252::from(use_prev_hash as i64),
        vm,
        ids_data,
        ap_tracking,
    )?;
    Ok(())
}

/// Implements the v0.14 macro hint `LOAD_PROGRAM_SEGMENT`
/// (execute_task.cairo:112). Allocates a fresh segment, sets the
/// `program_segment_ptr` local to it, loads the program header + code there, and
/// records the program address. Unlike the 0.13.3 path this combines the segment
/// allocation and `load_program` in a single hint (there is no separate
/// `program_data_ptr`/VALIDATE_HASH; the hash is computed in Cairo).
pub fn load_program_segment_hint(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let program_data_base = vm.add_memory_segment();
    insert_value_from_var_name(
        "program_segment_ptr",
        program_data_base,
        vm,
        ids_data,
        ap_tracking,
    )?;

    let task: Task = exec_scopes.get(vars::TASK)?;
    let program = get_program_from_task(&task)?;

    // In v0.14 `program_header = cast(program_segment_ptr, ProgramHeader*)`,
    // i.e. the program header is loaded at the freshly allocated segment base.
    let program_header_ptr = program_data_base;

    // Offset of the builtin_list field in `ProgramHeader`, cf. execute_task.cairo.
    let builtins_offset = 4;
    let mut program_loader = ProgramLoader::new(vm, builtins_offset);
    let bootloader_version: BootloaderVersion = 0;
    let loaded_program = program_loader
        .load_program(program_header_ptr, &program, Some(bootloader_version))
        .map_err(Into::<HintError>::into)?;

    vm.segments.finalize(
        Some(loaded_program.size),
        program_data_base.segment_index as usize,
        None,
    );

    exec_scopes.insert_value(vars::PROGRAM_DATA_BASE, program_data_base);
    exec_scopes.insert_value(vars::PROGRAM_ADDRESS, loaded_program.code_address);

    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.simple_bootloader.utils import get_task_fact_topology
///
/// # Add the fact topology of the current task to 'fact_topologies'.
/// output_start = ids.pre_execution_builtin_ptrs.output
/// output_end = ids.return_builtin_ptrs.output
/// fact_topologies.append(get_task_fact_topology(
///     output_size=output_end - output_start,
///     task=task,
///     output_builtin=output_builtin,
///     output_runner_data=output_runner_data,
/// ))
pub fn append_fact_topologies(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let task: Task = exec_scopes.get(vars::TASK)?;
    let output_runner_data: Option<OutputBuiltinState> =
        exec_scopes.get(vars::OUTPUT_RUNNER_DATA)?;
    let fact_topologies: &mut Vec<FactTopology> = exec_scopes.get_mut_ref(vars::FACT_TOPOLOGIES)?;

    let pre_execution_builtin_ptrs_addr =
        get_relocatable_from_var_name("pre_execution_builtin_ptrs", vm, ids_data, ap_tracking)?;
    let return_builtin_ptrs_addr =
        get_relocatable_from_var_name("return_builtin_ptrs", vm, ids_data, ap_tracking)?;

    // The output field is the first one in the BuiltinData struct
    let output_start = vm.get_relocatable(pre_execution_builtin_ptrs_addr)?;
    let output_end = vm.get_relocatable(return_builtin_ptrs_addr)?;
    let output_size = (output_end - output_start)?;

    let output_builtin = vm.get_output_builtin_mut()?;
    let fact_topology =
        get_task_fact_topology(output_size, &task, output_builtin, output_runner_data)
            .map_err(Into::<HintError>::into)?;
    fact_topologies.push(fact_topology);

    Ok(())
}

/// Implements
/// # Validate hash.
/// from starkware.cairo.bootloaders.hash_program import compute_program_hash_chain
///
/// assert memory[ids.output_ptr + 1] == compute_program_hash_chain(task.get_program()), \
///   'Computed hash does not match input.'";
pub fn validate_hash(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let task: Task = exec_scopes.get(vars::TASK)?;
    let program = get_program_from_task(&task)?;

    let use_poseidon = get_integer_from_var_name("use_poseidon", vm, ids_data, ap_tracking)?;
    if use_poseidon != Felt252::ZERO {
        return Err(HintError::CustomHint(
            "use_poseidon=1 program-hash chain is not implemented in this port"
                .to_string()
                .into_boxed_str(),
        ));
    }

    let output_ptr = get_ptr_from_var_name("output_ptr", vm, ids_data, ap_tracking)?;
    let program_hash_ptr = (output_ptr + 1)?;

    let program_hash = vm.get_integer(program_hash_ptr)?.into_owned();

    // Compute the hash of the program
    let computed_program_hash = compute_program_hash_chain(&program, 0).map_err(|e| {
        HintError::CustomHint(format!("Could not compute program hash: {e}").into_boxed_str())
    })?;
    let computed_program_hash = field_element_to_felt(computed_program_hash);

    if program_hash != computed_program_hash {
        return Err(HintError::AssertionFailed(
            "Computed hash does not match input"
                .to_string()
                .into_boxed_str(),
        ));
    }

    Ok(())
}

/// Implements
/// memory[ap] = to_felt_or_relocatable(1 if task.use_poseidon else 0)
pub fn task_use_poseidon(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let use_poseidon: bool = exec_scopes.get(vars::TASK_USE_POSEIDON).unwrap_or(false);
    insert_value_into_ap(vm, Felt252::from(use_poseidon as u64))
}

/// List of all builtins in the order used by the bootloader.
const ALL_BUILTINS: [BuiltinName; 11] = [
    BuiltinName::output,
    BuiltinName::pedersen,
    BuiltinName::range_check,
    BuiltinName::ecdsa,
    BuiltinName::bitwise,
    BuiltinName::ec_op,
    BuiltinName::keccak,
    BuiltinName::poseidon,
    BuiltinName::range_check96,
    BuiltinName::add_mod,
    BuiltinName::mul_mod,
];

fn check_cairo_pie_builtin_usage(
    vm: &mut VirtualMachine,
    builtin_name: &BuiltinName,
    builtin_index: usize,
    cairo_pie: &CairoPie,
    return_builtins_addr: Relocatable,
    pre_execution_builtins_addr: Relocatable,
) -> Result<(), HintError> {
    let return_builtin_value = vm.get_relocatable((return_builtins_addr + builtin_index)?)?;
    let pre_execution_builtin_value =
        vm.get_relocatable((pre_execution_builtins_addr + builtin_index)?)?;
    let expected_builtin_size = (return_builtin_value - pre_execution_builtin_value)?;

    let builtin_size = cairo_pie.metadata.builtin_segments[builtin_name].size;

    if builtin_size != expected_builtin_size {
        return Err(HintError::AssertionFailed(
            "Builtin usage is inconsistent with the CairoPie."
                .to_string()
                .into_boxed_str(),
        ));
    }

    Ok(())
}

/// Writes the updated builtin pointers after the program execution to the given return builtins
/// address.
///     
/// `used_builtins` is the list of builtins used by the program and thus updated by it.
fn write_return_builtins(
    vm: &mut VirtualMachine,
    return_builtins_addr: Relocatable,
    used_builtins: &[BuiltinName],
    used_builtins_addr: Relocatable,
    pre_execution_builtins_addr: Relocatable,
    task: &Task,
) -> Result<(), HintError> {
    let mut used_builtin_offset: usize = 0;
    for (index, builtin) in ALL_BUILTINS.iter().enumerate() {
        if used_builtins.contains(builtin) {
            let builtin_value = vm.get_relocatable((used_builtins_addr + used_builtin_offset)?)?;
            vm.insert_value((return_builtins_addr + index)?, builtin_value)?;
            used_builtin_offset += 1;

            if let Task::Pie(cairo_pie) = task {
                check_cairo_pie_builtin_usage(
                    vm,
                    builtin,
                    index,
                    cairo_pie,
                    return_builtins_addr,
                    pre_execution_builtins_addr,
                )?;
            }
        }
        // The builtin is unused, hence its value is the same as before calling the program.
        else {
            let pre_execution_builtin_addr = (pre_execution_builtins_addr + index)?;
            let pre_execution_value =
                vm.get_maybe(&pre_execution_builtin_addr).ok_or_else(|| {
                    MemoryError::UnknownMemoryCell(Box::new(pre_execution_builtin_addr))
                })?;
            vm.insert_value((return_builtins_addr + index)?, pre_execution_value)?;
        }
    }
    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.simple_bootloader.utils import write_return_builtins
///
/// # Fill the values of all builtin pointers after executing the task.
/// builtins = task.get_program().builtins
/// write_return_builtins(
///     memory=memory, return_builtins_addr=ids.return_builtin_ptrs.address_,
///     used_builtins=builtins, used_builtins_addr=ids.used_builtins_addr,
///     pre_execution_builtins_addr=ids.pre_execution_builtin_ptrs.address_, task=task)
///
/// vm_enter_scope({'n_selected_builtins': n_builtins})
///
/// This hint looks at the builtins written by the program and merges them with the stored
/// pre-execution values (stored in a struct named ids.pre_execution_builtin_ptrs) to
/// create a final BuiltinData struct for the program.
pub fn write_return_builtins_hint(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let task: Task = exec_scopes.get(vars::TASK)?;
    let n_builtins: usize = exec_scopes.get(vars::N_BUILTINS)?;

    // builtins = task.get_program().builtins
    let program = get_program_from_task(&task)?;
    let builtins = &program.builtins;

    // write_return_builtins(
    //     memory=memory, return_builtins_addr=ids.return_builtin_ptrs.address_,
    //     used_builtins=builtins, used_builtins_addr=ids.used_builtins_addr,
    //     pre_execution_builtins_addr=ids.pre_execution_builtin_ptrs.address_, task=task)
    let return_builtins_addr =
        get_relocatable_from_var_name("return_builtin_ptrs", vm, ids_data, ap_tracking)?;
    let used_builtins_addr =
        get_ptr_from_var_name("used_builtins_addr", vm, ids_data, ap_tracking)?;
    let pre_execution_builtins_addr =
        get_relocatable_from_var_name("pre_execution_builtin_ptrs", vm, ids_data, ap_tracking)?;

    write_return_builtins(
        vm,
        return_builtins_addr,
        builtins,
        used_builtins_addr,
        pre_execution_builtins_addr,
        &task,
    )?;

    // vm_enter_scope({'n_selected_builtins': n_builtins})
    let n_builtins: Box<dyn Any> = Box::new(n_builtins);
    exec_scopes.enter_scope(HashMap::from([(
        vars::N_SELECTED_BUILTINS.to_string(),
        n_builtins,
    )]));

    Ok(())
}

fn get_bootloader_identifiers(
    exec_scopes: &ExecutionScopes,
) -> Result<&ProgramIdentifiers, HintError> {
    if let Some(bootloader_identifiers) =
        exec_scopes.data[0].get(vars::BOOTLOADER_PROGRAM_IDENTIFIERS)
    {
        if let Some(program) = bootloader_identifiers.downcast_ref::<ProgramIdentifiers>() {
            return Ok(program);
        }
    }

    Err(HintError::VariableNotInScopeError(
        vars::BOOTLOADER_PROGRAM_IDENTIFIERS
            .to_string()
            .into_boxed_str(),
    ))
}

fn get_identifier(
    identifiers: &HashMap<String, Identifier>,
    name: &str,
) -> Result<usize, HintError> {
    if let Some(identifier) = identifiers.get(name) {
        if let Some(pc) = identifier.pc {
            return Ok(pc);
        }
    }

    Err(HintError::VariableNotInScopeError(
        name.to_string().into_boxed_str(),
    ))
}

/*
Implements hint:
%{
    "from starkware.cairo.bootloaders.simple_bootloader.objects import (
        CairoPieTask,
        RunProgramTask,
        Task,
    )
    from starkware.cairo.bootloaders.simple_bootloader.utils import (
        load_cairo_pie,
        prepare_output_runner,
    )

    assert isinstance(task, Task)
    n_builtins = len(task.get_program().builtins)
    new_task_locals = {}
    if isinstance(task, RunProgramTask):
        new_task_locals['program_input'] = task.program_input
        new_task_locals['WITH_BOOTLOADER'] = True

        vm_load_program(task.program, program_address)
    elif isinstance(task, CairoPieTask):
        ret_pc = ids.ret_pc_label.instruction_offset_ - ids.call_task.instruction_offset_ + pc
        load_cairo_pie(
            task=task.cairo_pie, memory=memory, segments=segments,
            program_address=program_address, execution_segment_address= ap - n_builtins,
            builtin_runners=builtin_runners, ret_fp=fp, ret_pc=ret_pc)
    else:
        raise NotImplementedError(f'Unexpected task type: {type(task).__name__}.')

    output_runner_data = prepare_output_runner(
        task=task,
        output_builtin=output_builtin,
        output_ptr=ids.pre_execution_builtin_ptrs.output)
    vm_enter_scope(new_task_locals)"
%}
*/
pub fn call_task(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    // assert isinstance(task, Task)
    let task: Task = exec_scopes.get(vars::TASK)?;

    // n_builtins = len(task.get_program().builtins)
    let n_builtins = get_program_from_task(&task)?.builtins.len();
    exec_scopes.insert_value(vars::N_BUILTINS, n_builtins);

    let mut new_task_locals = HashMap::new();

    match &task {
        // if isinstance(task, RunProgramTask):
        Task::Program(_program) => {
            let program_input = HashMap::<String, Box<dyn Any>>::new();
            // new_task_locals['program_input'] = task.program_input
            new_task_locals.insert("program_input".to_string(), any_box![program_input]);
            // new_task_locals['WITH_BOOTLOADER'] = True
            new_task_locals.insert("WITH_BOOTLOADER".to_string(), any_box![true]);

            // TODO: the content of this function is mostly useless for the Rust VM.
            //       check with SW if there is nothing of interest here.
            // vm_load_program(task.program, program_address)
        }
        // elif isinstance(task, CairoPieTask):
        Task::Pie(cairo_pie) => {
            let program_address: Relocatable = exec_scopes.get("program_address")?;

            // ret_pc = ids.ret_pc_label.instruction_offset_ - ids.call_task.instruction_offset_ + pc
            let bootloader_identifiers = get_bootloader_identifiers(exec_scopes)?;
            let ret_pc_label = get_identifier(bootloader_identifiers, "starkware.cairo.bootloaders.simple_bootloader.execute_task.execute_task.ret_pc_label")?;
            let call_task = get_identifier(
                bootloader_identifiers,
                "starkware.cairo.bootloaders.simple_bootloader.execute_task.execute_task.call_task",
            )?;

            let ret_pc_offset = ret_pc_label - call_task;
            let ret_pc = (vm.get_pc() + ret_pc_offset)?;

            // load_cairo_pie(
            //     task=task.cairo_pie, memory=memory, segments=segments,
            //     program_address=program_address, execution_segment_address= ap - n_builtins,
            //     builtin_runners=builtin_runners, ret_fp=fp, ret_pc=ret_pc)
            load_cairo_pie(
                cairo_pie,
                vm,
                program_address,
                (vm.get_ap() - n_builtins)?,
                vm.get_fp(),
                ret_pc,
            )
            .map_err(Into::<HintError>::into)?;
        }
    }

    // output_runner_data = prepare_output_runner(
    //     task=task,
    //     output_builtin=output_builtin,
    //     output_ptr=ids.pre_execution_builtin_ptrs.output)
    let pre_execution_builtin_ptrs_addr =
        get_relocatable_from_var_name(vars::PRE_EXECUTION_BUILTIN_PTRS, vm, ids_data, ap_tracking)?;
    // The output field is the first one in the BuiltinData struct
    let output_ptr = vm.get_relocatable((pre_execution_builtin_ptrs_addr + 0)?)?;
    let output_runner_data =
        util::prepare_output_runner(&task, vm.get_output_builtin_mut()?, output_ptr)?;

    exec_scopes.insert_value(vars::OUTPUT_RUNNER_DATA, output_runner_data);

    exec_scopes.enter_scope(new_task_locals);

    Ok(())
}

mod util {
    // TODO: clean up / organize
    use super::*;

    /// Prepares the output builtin if the type of task is Task, so that pages of the inner program
    /// will be recorded separately.
    /// If the type of task is CairoPie, nothing should be done, as the program does not contain
    /// hints that may affect the output builtin.
    /// The return value of this function should be later passed to get_task_fact_topology().
    pub(crate) fn prepare_output_runner(
        task: &Task,
        output_builtin: &mut OutputBuiltinRunner,
        output_ptr: Relocatable,
    ) -> Result<Option<OutputBuiltinState>, HintError> {
        match task {
            Task::Program(_) => {
                let output_state = output_builtin.get_state();
                output_builtin.new_state(output_ptr.segment_index as usize, 0, true);
                Ok(Some(output_state))
            }
            Task::Pie(_) => Ok(None),
        }
    }
}

