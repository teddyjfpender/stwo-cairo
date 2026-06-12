use std::collections::HashMap;

use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::{
    get_constant_from_var_name, get_integer_from_var_name, get_ptr_from_var_name,
    get_relocatable_from_var_name, insert_value_from_var_name, insert_value_into_ap,
};
use cairo_vm::hint_processor::hint_processor_definition::HintReference;
use cairo_vm::serde::deserialize_program::ApTracking;
use cairo_vm::types::errors::math_errors::MathError;
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::types::relocatable::{MaybeRelocatable, Relocatable};
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::runners::builtin_runner::{
    EcOpBuiltinRunner, KeccakBuiltinRunner, SignatureBuiltinRunner,
};
use cairo_vm::vm::vm_core::VirtualMachine;
use cairo_vm::Felt252;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use starknet_types_core::felt::NonZeroFelt;

use crate::hints::fact_topologies::FactTopology;
use crate::hints::types::{HashFunc, SimpleBootloaderInput};
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
    let task_spec = simple_bootloader_input.tasks[task_id].clone();
    let task = task_spec.load_task().clone();

    // v0.14: load_program_segment can reuse the previous task's hash/segment if
    // the programs are identical. With a single PIE task there is no previous
    // task, so use_prev_hash is always 0. We compute it faithfully nonetheless.
    let mut use_prev_hash: i32 = 0;
    if task_id > 0 {
        let prev_task = simple_bootloader_input.tasks[task_id - 1].load_task();
        let same = match (task.get_program(), prev_task.get_program()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        use_prev_hash = i32::from(same);
    }

    exec_scopes.insert_value(vars::TASK, task);
    exec_scopes.insert_value(vars::TASK_USE_POSEIDON, task_spec.use_poseidon);
    exec_scopes.insert_value(vars::USE_PREV_HASH, use_prev_hash);
    exec_scopes.insert_value(vars::PROGRAM_HASH_FUNCTION, task_spec.program_hash_function);

    Ok(())
}

/// Implements the v0.14 macro hint `SETUP_RUN_SIMPLE_BOOTLOADER_BEFORE_TASK_EXECUTION`
/// (run_simple_bootloader.cairo). Writes n_tasks to `output_ptr[0]`, sets
/// `initial_subtasks_range_check_ptr` to a fresh temporary segment (the subtask
/// range checks are relocated after the bootloader's at the end of the run), and
/// initializes the `fact_topologies` hint var to an empty list.
pub fn setup_run_simple_bootloader_before_task_execution(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let n_tasks = {
        let simple_bootloader_input: &SimpleBootloaderInput =
            exec_scopes.get_ref(vars::SIMPLE_BOOTLOADER_INPUT)?;
        simple_bootloader_input.tasks.len()
    };

    let output_ptr = get_ptr_from_var_name("output_ptr", vm, ids_data, ap_tracking)?;
    vm.insert_value(output_ptr, Felt252::from(n_tasks))?;

    let temp_segment = vm.add_temporary_segment();
    insert_value_from_var_name(
        "initial_subtasks_range_check_ptr",
        temp_segment,
        vm,
        ids_data,
        ap_tracking,
    )?;

    exec_scopes.insert_value(vars::FACT_TOPOLOGIES, Vec::<FactTopology>::new());

    Ok(())
}

/// Implements `tempvar program_hash_function = nondet %{ task.program_hash_function %}`.
/// Pushes the current task's hash-function selector (Pedersen=0/Poseidon=1/Blake=2)
/// to the AP.
pub fn program_hash_function_to_ap(
    vm: &mut VirtualMachine,
    exec_scopes: &mut ExecutionScopes,
) -> Result<(), HintError> {
    let program_hash_function: HashFunc = exec_scopes.get(vars::PROGRAM_HASH_FUNCTION)?;
    insert_value_into_ap(vm, Felt252::from(program_hash_function as usize))
}

// ---------------------------------------------------------------------------
// keccak / ec_op / ecdsa builtin simulation (verify_builtins.cairo).
//
// The `all_cairo_stwo` layout lacks the ecdsa/keccak/ec_op builtins. The v0.14
// simple bootloader allocates a *simulated* builtin segment for each one and
// verifies every used instance in Cairo after the tasks. cairo-vm 3.2.0
// implements these simulated runners natively (`vm.simulated_builtin_runners`),
// so each "register auto-deduction rule" hint maps directly to constructing a
// runner and pushing it onto the simulated runner list — the runner itself does
// the auto-deduction the python `*_auto_deduction_rule_wrapper` would have done.
// ---------------------------------------------------------------------------

/// `SIMPLE_BOOTLOADER_SIMULATE_EC_OP`: allocate a simulated ec_op builtin segment.
pub fn simple_bootloader_simulate_ec_op(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let mut ec_op_runner = EcOpBuiltinRunner::new(Some(1), false);
    ec_op_runner.initialize_segments(&mut vm.segments);
    let new_ec_op_ptr = Relocatable {
        segment_index: ec_op_runner.base() as isize,
        offset: 0,
    };
    insert_value_from_var_name("new_ec_op_ptr", new_ec_op_ptr, vm, ids_data, ap_tracking)?;
    vm.simulated_builtin_runners.push(ec_op_runner.into());
    Ok(())
}

/// `curr_m = ids.m; for i in range(ids.M_MAX_BITS): memory[ids.m_bit_unpacking + i] = curr_m % 2; curr_m >>= 1`.
pub fn simulate_ec_op_fill_mem_with_bits_of_m(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
    constants: &HashMap<String, Felt252>,
) -> Result<(), HintError> {
    let mut curr_m = get_integer_from_var_name("m", vm, ids_data, ap_tracking)?.to_biguint();
    let m_max_bits = get_constant_from_var_name("M_MAX_BITS", constants)?;
    let m_bit_unpacking = get_ptr_from_var_name("m_bit_unpacking", vm, ids_data, ap_tracking)?;
    (0..m_max_bits.to_usize().unwrap()).try_for_each(|i| {
        let bit = MaybeRelocatable::Int((&curr_m % 2u32).into());
        curr_m >>= 1;
        vm.insert_value((m_bit_unpacking + i)?, bit)
    })?;
    Ok(())
}

/// `assert False, "ec_op failed."`.
pub fn simulate_ec_op_assert_false() -> Result<(), HintError> {
    Err(HintError::CustomHint("ec_op failed.".into()))
}

/// `SIMPLE_BOOTLOADER_SIMULATE_KECCAK`: allocate a simulated keccak builtin segment.
pub fn simple_bootloader_simulate_keccak(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let mut keccak_runner = KeccakBuiltinRunner::new(Some(1), false);
    keccak_runner.initialize_segments(&mut vm.segments);
    let new_keccak_ptr = Relocatable {
        segment_index: keccak_runner.base() as isize,
        offset: 0,
    };
    insert_value_from_var_name("new_keccak_ptr", new_keccak_ptr, vm, ids_data, ap_tracking)?;
    vm.simulated_builtin_runners.push(keccak_runner.into());
    Ok(())
}

/// Packs `keccak_builtin_state.s0..s7` (each a 200-bit limb) into a 1600-bit
/// value, then splits it into 25 little-endian 64-bit words at `felt_array`.
pub fn simulate_keccak_fill_mem_with_state(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let keccak_builtin_state_addr =
        get_relocatable_from_var_name("keccak_builtin_state", vm, ids_data, ap_tracking)?;
    let felt_array = get_ptr_from_var_name("felt_array", vm, ids_data, ap_tracking)?;
    let mut full_num = (0..8).try_fold(BigUint::ZERO, |acc, i| {
        let s = vm.get_integer((keccak_builtin_state_addr + i)?)?;
        Ok::<_, HintError>(acc + (s.to_biguint() << (i * 200)))
    })?;
    let modulo = BigUint::from(1u128 << 64);
    (0..25).try_for_each(|i| {
        let felt = MaybeRelocatable::Int((&full_num % &modulo).into());
        full_num >>= 64;
        vm.insert_value((felt_array + i)?, felt)
    })?;
    Ok(())
}

/// `ids.high{index}, ids.low{index} = divmod(memory[ids.felt_array + index], 256 ** (index/3))`.
pub fn simulate_keccak_calc_high_low(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
    index: usize,
) -> Result<(), HintError> {
    let felt_array = get_ptr_from_var_name("felt_array", vm, ids_data, ap_tracking)?;
    let felt = vm.get_integer((felt_array + index)?)?;
    let x = index / 3;
    let divisor = NonZeroFelt::try_from(Felt252::from(1u64 << (x * 8))).unwrap();
    let (high_felt, low_felt) = felt.div_rem(&divisor);
    insert_value_from_var_name(&format!("high{index}"), high_felt, vm, ids_data, ap_tracking)?;
    insert_value_from_var_name(&format!("low{index}"), low_felt, vm, ids_data, ap_tracking)?;
    Ok(())
}

/// `SIMPLE_BOOTLOADER_SIMULATE_ECDSA`: allocate a simulated ecdsa/signature
/// builtin segment and register its validation rule.
pub fn simple_bootloader_simulate_ecdsa(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let mut ecdsa_runner = SignatureBuiltinRunner::new(Some(1), false);
    ecdsa_runner.initialize_segments(&mut vm.segments);
    let new_ecdsa_ptr = Relocatable {
        segment_index: ecdsa_runner.base() as isize,
        offset: 0,
    };
    insert_value_from_var_name("new_ecdsa_ptr", new_ecdsa_ptr, vm, ids_data, ap_tracking)?;
    ecdsa_runner.add_validation_rule(&mut vm.segments.memory);
    vm.simulated_builtin_runners.push(ecdsa_runner.into());
    Ok(())
}

/// `(ids.r, ids.s) = vm_ecdsa_additional_data[ids.start.address_]` — reads the
/// signature registered for the `start` pointer from the simulated ecdsa runner.
pub fn simulate_ecdsa_get_r_and_s(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let start = get_ptr_from_var_name("start", vm, ids_data, ap_tracking)?;
    let ecdsa_builtin = vm.get_signature_builtin()?;
    let (r, s) = {
        let signatures = ecdsa_builtin.signatures.borrow();
        let signature = signatures
            .get(&start)
            .ok_or_else(|| HintError::CustomHint("No signature found for start pointer.".into()))?;
        (signature.r, signature.s)
    };
    insert_value_from_var_name("r", r, vm, ids_data, ap_tracking)?;
    insert_value_from_var_name("s", s, vm, ids_data, ap_tracking)?;
    Ok(())
}

/// `order = StarkCurve.ORDER + PRIME; ids.w = pow(s, -1, order); ids.wz = w*z % order; ids.wr = w*r % order`.
pub fn simulate_ecdsa_compute_w_wr_wz(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
    constants: &HashMap<String, Felt252>,
) -> Result<(), HintError> {
    let order_const_id = "starkware.cairo.common.ec.StarkCurve.ORDER";
    let order = constants
        .get(order_const_id)
        .ok_or_else(|| HintError::MissingConstant(Box::new(order_const_id)))?;
    let order = &NonZeroFelt::from_felt_unchecked(*order);
    let s = get_integer_from_var_name("signature_s", vm, ids_data, ap_tracking)?;
    let r = get_integer_from_var_name("signature_r", vm, ids_data, ap_tracking)?;
    let message = get_integer_from_var_name("message", vm, ids_data, ap_tracking)?;
    let w = s.mod_inverse(order).unwrap();
    let wz = w.mul_mod(&message, order);
    let wr = w.mul_mod(&r, order);
    insert_value_from_var_name("w", w, vm, ids_data, ap_tracking)?;
    insert_value_from_var_name("wz", wz, vm, ids_data, ap_tracking)?;
    insert_value_from_var_name("wr", wr, vm, ids_data, ap_tracking)?;
    Ok(())
}

/// Splits `ids.num` into three little-endian 96-bit limbs at `ids.res_96_felts`.
pub fn simulate_ecdsa_fill_mem_with_felt_96_bit_limbs(
    vm: &mut VirtualMachine,
    ids_data: &HashMap<String, HintReference>,
    ap_tracking: &ApTracking,
) -> Result<(), HintError> {
    let num = get_integer_from_var_name("num", vm, ids_data, ap_tracking)?;
    let res_96_felts = get_ptr_from_var_name("res_96_felts", vm, ids_data, ap_tracking)?;
    let mut num = num.to_biguint();
    let modulo = BigUint::from(1u128 << 96);
    (0..3).try_for_each(|i| {
        let felt = MaybeRelocatable::Int((&num % &modulo).into());
        num >>= 96;
        vm.insert_value((res_96_felts + i)?, felt)
    })?;
    Ok(())
}

