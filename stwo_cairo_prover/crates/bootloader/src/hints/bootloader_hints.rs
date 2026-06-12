use crate::hints::fact_topologies::{
    compute_fact_topologies, configure_fact_topologies, write_to_fact_topologies_file, FactTopology,
};
use cairo_vm::any_box;
use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::{
    get_integer_from_var_name, get_ptr_from_var_name, insert_value_from_var_name,
    insert_value_into_ap,
};
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::ApTracking;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::types::relocatable::{MaybeRelocatable, Relocatable};
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::errors::memory_errors::MemoryError;
use cairo_vm::vm::runners::builtin_runner::OutputBuiltinState;
use cairo_vm::vm::vm_core::VirtualMachine;
use num_traits::ToPrimitive;
use std::any::Any;
use std::collections::HashMap;

use crate::hints::types::{BootloaderInput, CompositePackedOutput, PackedOutput};
use crate::hints::vars;

/// Implements
/// ```no-run
/// %{
///     from starkware.cairo.bootloaders.bootloader.objects import BootloaderInput
///     bootloader_input = BootloaderInput.Schema().load(program_input)
///
///     ids.simple_bootloader_output_start = segments.add()
///
///     # Change output builtin state to a different segment in preparation for calling the
///     # simple bootloader.
///     output_builtin_state = output_builtin.get_state()
///     output_builtin.new_state(base=ids.simple_bootloader_output_start)
/// %}
/// ```
pub fn prepare_simple_bootloader_output_segment(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    // Python: bootloader_input = BootloaderInput.Schema().load(program_input)
    // -> Assert that the bootloader input has been loaded when setting up the VM
    let _bootloader_input: &BootloaderInput = exec_scopes.get_ref(vars::BOOTLOADER_INPUT)?;

    // Python: ids.simple_bootloader_output_start = segments.add()
    let new_segment_base = vm.add_memory_segment();
    insert_value_from_var_name(
        "simple_bootloader_output_start",
        new_segment_base,
        vm,
        ids_data,
        ap_tracking,
    )?;

    // Python:
    // output_builtin_state = output_builtin.get_state()
    // output_builtin.new_state(base=ids.simple_bootloader_output_start)
    let output_builtin = vm.get_output_builtin_mut()?;
    let output_builtin_state = output_builtin.get_state();
    output_builtin.new_state(new_segment_base.segment_index as usize, 0, true);
    exec_scopes.insert_value(vars::OUTPUT_BUILTIN_STATE, output_builtin_state);

    insert_value_from_var_name(
        "simple_bootloader_output_start",
        new_segment_base,
        vm,
        ids_data,
        ap_tracking,
    )?;

    Ok(())
}

/// Implements %{ simple_bootloader_input = bootloader_input %}
pub fn prepare_simple_bootloader_input(exec_scopes: &mut ExecutionScopes) -> Result<(), HintError> {
    let bootloader_input: BootloaderInput = exec_scopes.get(vars::BOOTLOADER_INPUT)?;
    exec_scopes.insert_value(
        vars::SIMPLE_BOOTLOADER_INPUT,
        bootloader_input.simple_bootloader_input,
    );

    Ok(())
}

/// Implements
/// # Restore the bootloader's output builtin state.
/// output_builtin.set_state(output_builtin_state)
pub fn restore_bootloader_output(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let output_builtin_state: OutputBuiltinState = exec_scopes.get(vars::OUTPUT_BUILTIN_STATE)?;
    vm.get_output_builtin_mut()?.set_state(output_builtin_state);

    Ok(())
}
/// Mimics the behaviour of the Python VM `gen_arg`.
///
/// Creates a new segment for each vector encountered in `args`. For each new
/// segment, the pointer to the segment will be added to the current segment.
///
/// Example: `vec![1, 2, vec![3, 4]]`
/// -> Allocates segment N, starts writing at offset 0:
/// (N, 0): 1       # Write the values of the vector one by one
/// (N, 1): 2
/// -> a vector is encountered, allocate a new segment
/// (N, 2): N+1     # Pointer to the new segment
/// (N+1, 0): 3     # Write the values of the nested vector
/// (N+1, 1): 4
fn gen_arg(vm: &mut VirtualMachine, args: &Vec<Box<dyn Any>>) -> Result<Relocatable, MemoryError> {
    let base = vm.segments.add();
    let mut ptr = base;

    for arg in args {
        if let Some(value) = arg.downcast_ref::<MaybeRelocatable>() {
            ptr = vm.segments.load_data(ptr, &vec![value.clone()])?;
        } else if let Some(vector) = arg.downcast_ref::<Vec<Box<dyn Any>>>() {
            let nested_base = gen_arg(vm, vector)?;
            ptr = vm.segments.load_data(ptr, &vec![nested_base.into()])?;
        } else {
            return Err(MemoryError::GenArgInvalidType);
        }
    }

    Ok(base)
}

/// Implements
/// from starkware.cairo.bootloaders.bootloader.objects import BootloaderConfig
/// bootloader_config: BootloaderConfig = bootloader_input.bootloader_config
///
/// ids.bootloader_config = segments.gen_arg(
///     [
///         bootloader_config.simple_bootloader_program_hash,
///         len(bootloader_config.supported_cairo_verifier_program_hashes),
///         bootloader_config.supported_cairo_verifier_program_hashes,
///     ],
/// )
pub fn load_bootloader_config(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let bootloader_input: BootloaderInput = exec_scopes.get(vars::BOOTLOADER_INPUT)?;
    let config = bootloader_input.bootloader_config;

    // Organize args as
    // [
    //     bootloader_config.simple_bootloader_program_hash,
    //     len(bootloader_config.supported_cairo_verifier_program_hashes),
    //     bootloader_config.supported_cairo_verifier_program_hashes,
    // ]
    let mut program_hashes = Vec::<Box<dyn Any>>::new();
    for program_hash in &config.supported_cairo_verifier_program_hashes {
        program_hashes.push(Box::new(MaybeRelocatable::from(program_hash)));
    }

    let args: Vec<Box<dyn Any>> = vec![
        any_box!(MaybeRelocatable::from(
            config.simple_bootloader_program_hash
        )),
        any_box!(MaybeRelocatable::from(
            config.supported_cairo_verifier_program_hashes.len()
        )),
        any_box!(program_hashes),
    ];

    // Store the args in the VM memory
    let args_segment = gen_arg(vm, &args)?;
    insert_value_from_var_name("bootloader_config", args_segment, vm, ids_data, ap_tracking)?;

    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.bootloader.objects import PackedOutput
///
/// task_id = len(packed_outputs) - ids.n_subtasks
/// packed_output: PackedOutput = packed_outputs[task_id]
///
/// vm_enter_scope(new_scope_locals=dict(packed_output=packed_output))
pub fn enter_packed_output_scope(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    // task_id = len(packed_outputs) - ids.n_subtasks
    let packed_outputs: Vec<PackedOutput> = exec_scopes.get(vars::PACKED_OUTPUTS)?;
    let n_subtasks = get_integer_from_var_name("n_subtasks", vm, ids_data, ap_tracking)
        .unwrap()
        .to_usize()
        .unwrap();
    let task_id = packed_outputs.len() - n_subtasks;
    // packed_output: PackedOutput = packed_outputs[task_id]
    let packed_output: Box<dyn Any> = Box::new(packed_outputs[task_id].clone());

    // vm_enter_scope(new_scope_locals=dict(packed_output=packed_output))
    exec_scopes.enter_scope(HashMap::from([(
        vars::PACKED_OUTPUT.to_string(),
        packed_output,
    )]));

    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.bootloader.objects import (
///     CompositePackedOutput,
///     PlainPackedOutput,
/// )
pub fn import_packed_output_schemas() -> Result<(), HintError> {
    // Nothing to do!
    Ok(())
}

/// Implements %{ isinstance(packed_output, PlainPackedOutput) %}
/// (compiled to %{ memory[ap] = to_felt_or_relocatable(isinstance(packed_output, PlainPackedOutput)) %}).
///
/// Stores the result in the `ap` register to be accessed by the program.
pub fn is_plain_packed_output(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let packed_output: PackedOutput = exec_scopes.get(vars::PACKED_OUTPUT)?;
    let result = match packed_output {
        PackedOutput::Plain(_) => 1,
        _ => 0,
    };
    insert_value_into_ap(vm, result)?;

    Ok(())
}

/// Implements
/// %{ assert isinstance(packed_output, CompositePackedOutput) %}
pub fn assert_is_composite_packed_output(
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let packed_output: PackedOutput = exec_scopes.get(vars::PACKED_OUTPUT)?;

    match packed_output {
        PackedOutput::Composite(_) => Ok(()),
        other => Err(HintError::CustomHint(
            format!("Expected composite packed output, got {:?}", other).into_boxed_str(),
        )),
    }
}

/*
Implements hint:
%{
    output_start = ids.output_ptr
%}
*/
pub fn save_output_pointer(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let output_ptr = get_ptr_from_var_name("output_ptr", vm, ids_data, ap_tracking)?;
    exec_scopes.insert_value(vars::OUTPUT_START, output_ptr);
    Ok(())
}

/*
Implements hint:
%{
    packed_outputs = bootloader_input.packed_outputs
%}
*/
pub fn save_packed_outputs(exec_scopes: &mut ExecutionScopes) -> Result<(), HintError> {
    let bootloader_input: &BootloaderInput = exec_scopes.get_ref("bootloader_input")?;
    let packed_outputs = bootloader_input.packed_outputs.clone();
    exec_scopes.insert_value("packed_outputs", packed_outputs);
    Ok(())
}

/// Implements
/// from starkware.cairo.bootloaders.bootloader.utils import compute_fact_topologies
/// from starkware.cairo.bootloaders.fact_topology import FactTopology
/// from starkware.cairo.bootloaders.simple_bootloader.utils import (
///     configure_fact_topologies,
///     write_to_fact_topologies_file,
/// )
///
/// # Compute the fact topologies of the plain packed outputs based on packed_outputs and
/// # fact_topologies of the inner tasks.
/// plain_fact_topologies: List[FactTopology] = compute_fact_topologies(
///     packed_outputs=packed_outputs, fact_topologies=fact_topologies,
/// )
///
/// # Configure the memory pages in the output builtin, based on plain_fact_topologies.
/// configure_fact_topologies(
///     fact_topologies=plain_fact_topologies, output_start=output_start,
///     output_builtin=output_builtin,
/// )
///
/// # Dump fact topologies to a json file.
/// if bootloader_input.fact_topologies_path is not None:
///     write_to_fact_topologies_file(
///         fact_topologies_path=bootloader_input.fact_topologies_path,
///         fact_topologies=plain_fact_topologies,
///     )
pub fn compute_and_configure_fact_topologies(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let packed_outputs: Vec<PackedOutput> = exec_scopes.get(vars::PACKED_OUTPUTS)?;
    let fact_topologies: Vec<FactTopology> = exec_scopes.get(vars::FACT_TOPOLOGIES)?;
    let mut output_start: Relocatable = exec_scopes.get(vars::OUTPUT_START)?;
    let output_builtin = vm.get_output_builtin_mut()?;

    let plain_fact_topologies = compute_fact_topologies(&packed_outputs, &fact_topologies)
        .map_err(Into::<HintError>::into)?;

    configure_fact_topologies(&plain_fact_topologies, &mut output_start, output_builtin)
        .map_err(Into::<HintError>::into)?;

    exec_scopes.insert_value(vars::OUTPUT_START, output_start);

    let bootloader_input: &BootloaderInput = exec_scopes.get_ref(vars::BOOTLOADER_INPUT)?;
    if let Some(path) = &bootloader_input
        .simple_bootloader_input
        .fact_topologies_path
    {
        write_to_fact_topologies_file(path.as_path(), &plain_fact_topologies)
            .map_err(Into::<HintError>::into)?;
    }

    Ok(())
}

fn unwrap_composite_output(
    packed_output: PackedOutput,
) -> Result<CompositePackedOutput, HintError> {
    match packed_output {
        PackedOutput::Plain(_) => Err(HintError::CustomHint(
            "Expected packed output to be composite"
                .to_string()
                .into_boxed_str(),
        )),
        PackedOutput::Composite(composite_packed_output) => Ok(composite_packed_output),
    }
}

/*
Implements hint:
%{
    packed_outputs = packed_output.subtasks
%}
*/
pub fn set_packed_output_to_subtasks(exec_scopes: &mut ExecutionScopes) -> Result<(), HintError> {
    let packed_output: PackedOutput = exec_scopes.get(vars::PACKED_OUTPUT)?;
    let composite_packed_output = unwrap_composite_output(packed_output)?;
    let subtasks = composite_packed_output.subtasks;
    exec_scopes.insert_value(vars::PACKED_OUTPUTS, subtasks);

    Ok(())
}

/*
Implements hint:
%{
    data = packed_output.elements_for_hash()
    ids.nested_subtasks_output_len = len(data)
    ids.nested_subtasks_output = segments.gen_arg(data)";
%}
*/
pub fn guess_pre_image_of_subtasks_output_hash(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let packed_output: PackedOutput = exec_scopes.get(vars::PACKED_OUTPUT)?;
    let composite_packed_output = unwrap_composite_output(packed_output)?;

    let data = composite_packed_output.elements_for_hash();
    insert_value_from_var_name(
        "nested_subtasks_output_len",
        data.len(),
        vm,
        ids_data,
        ap_tracking,
    )?;
    let args = data
        .iter()
        .cloned()
        .map(|x| Box::new(MaybeRelocatable::Int(x)) as Box<dyn Any>)
        .collect();
    let nested_subtasks_output = gen_arg(vm, &args)?;
    insert_value_from_var_name(
        "nested_subtasks_output",
        nested_subtasks_output,
        vm,
        ids_data,
        ap_tracking,
    )?;

    Ok(())
}

/*
Implements hint:
%{
    # Sanity check.
    assert ids.program_address == program_address"
%}
*/
pub fn assert_program_address(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let ids_program_address =
        get_ptr_from_var_name(vars::PROGRAM_ADDRESS, vm, ids_data, ap_tracking)?;
    let program_address: Relocatable = exec_scopes.get(vars::PROGRAM_ADDRESS)?;

    if ids_program_address != program_address {
        return Err(HintError::CustomHint(
            "program address is incorrect".to_string().into_boxed_str(),
        ));
    }
    Ok(())
}

