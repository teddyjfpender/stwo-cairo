use std::any::Any;
use std::rc::Rc;

use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::{
    BuiltinHintProcessor, HintFunc, HintProcessorData,
};
use cairo_vm::hint_processor::builtin_hint_processor::memcpy_hint_utils::exit_scope;
use cairo_vm::hint_processor::hint_processor_definition::{HintExtension, HintProcessorLogic};
use cairo_vm::types::exec_scope::ExecutionScopes;
use cairo_vm::vm::errors::hint_errors::HintError;
use cairo_vm::vm::runners::cairo_runner::ResourceTracker;
use cairo_vm::vm::vm_core::VirtualMachine;

use crate::hints::bootloader_hints::{
    assert_is_composite_packed_output, assert_program_address,
    compute_and_configure_fact_topologies, compute_and_configure_fact_topologies_simple,
    enter_packed_output_scope, guess_pre_image_of_subtasks_output_hash, import_packed_output_schemas,
    is_plain_packed_output, load_bootloader_config, prepare_simple_bootloader_input,
    prepare_simple_bootloader_output_segment, restore_bootloader_output, save_output_pointer,
    save_packed_outputs, set_packed_output_to_subtasks,
};
use crate::hints::codes::*;
use crate::hints::types::BootloaderInput;
use crate::hints::vars;
use crate::hints::execute_task_hints::{
    allocate_program_data_segment, append_fact_topologies, call_task, determine_use_prev_hash,
    load_program_hint, load_program_segment_hint, task_use_poseidon, validate_hash,
    write_return_builtins_hint,
};
use crate::hints::inner_select_builtins::select_builtin;
use crate::hints::select_builtins::select_builtins_enter_scope;
use crate::hints::simple_bootloader_hints::{
    divide_num_by_2, prepare_task_range_checks, program_hash_function_to_ap, set_ap_to_zero,
    set_current_task, set_tasks_variable, setup_run_simple_bootloader_before_task_execution,
    simple_bootloader_simulate_ec_op, simple_bootloader_simulate_ecdsa,
    simple_bootloader_simulate_keccak, simulate_ec_op_assert_false,
    simulate_ec_op_fill_mem_with_bits_of_m, simulate_ecdsa_compute_w_wr_wz,
    simulate_ecdsa_fill_mem_with_felt_96_bit_limbs, simulate_ecdsa_get_r_and_s,
    simulate_keccak_calc_high_low, simulate_keccak_fill_mem_with_state,
};

/// A hint processor that can only execute the hints defined in this library.
/// For large projects, you may want to compose a hint processor from multiple parts
/// (ex: Starknet OS, bootloader and Cairo VM). This hint processor is as minimal as possible
/// to enable this modularity.
///
/// However, this processor is not sufficient to execute the bootloader. For this,
/// use `BootloaderHintProcessor`.
#[derive(Default)]
pub struct MinimalBootloaderHintProcessor;

impl MinimalBootloaderHintProcessor {
    pub fn new() -> Self {
        Self {}
    }
}

impl HintProcessorLogic for MinimalBootloaderHintProcessor {
    fn execute_hint(
        &mut self,
        vm: &mut VirtualMachine,
        exec_scopes: &mut ExecutionScopes,
        hint_data: &Box<dyn Any>,
    ) -> Result<(), HintError> {
        let hint_data = hint_data
            .downcast_ref::<HintProcessorData>()
            .ok_or(HintError::WrongHintData)?;

        let ids_data = &hint_data.ids_data;
        let ap_tracking = &hint_data.ap_tracking;
        let constants = &hint_data.constants;

        match hint_data.code.as_str() {
            BOOTLOADER_RESTORE_BOOTLOADER_OUTPUT => restore_bootloader_output(vm, exec_scopes),
            BOOTLOADER_PREPARE_SIMPLE_BOOTLOADER_INPUT => {
                prepare_simple_bootloader_input(exec_scopes)
            }
            BOOTLOADER_LOAD_BOOTLOADER_CONFIG => {
                load_bootloader_config(vm, exec_scopes, ids_data, ap_tracking)
            }
            BOOTLOADER_ENTER_PACKED_OUTPUT_SCOPE => {
                enter_packed_output_scope(vm, exec_scopes, ids_data, ap_tracking)
            }
            BOOTLOADER_SAVE_OUTPUT_POINTER => {
                save_output_pointer(vm, exec_scopes, ids_data, ap_tracking)
            }
            BOOTLOADER_SAVE_PACKED_OUTPUTS => save_packed_outputs(exec_scopes),
            BOOTLOADER_GUESS_PRE_IMAGE_OF_SUBTASKS_OUTPUT_HASH => {
                guess_pre_image_of_subtasks_output_hash(vm, exec_scopes, ids_data, ap_tracking)
            }
            BOOTLOADER_LOAD_INPUT => {
                // The input is inserted into exec_scopes by the caller before
                // the run; this hint just asserts it is present.
                exec_scopes
                    .get_ref::<BootloaderInput>(vars::BOOTLOADER_INPUT)
                    .map(|_| ())
            }
            BOOTLOADER_PREPARE_SIMPLE_BOOTLOADER_OUTPUT_SEGMENT => {
                prepare_simple_bootloader_output_segment(vm, exec_scopes, ids_data, ap_tracking)
            }
            BOOTLOADER_COMPUTE_FACT_TOPOLOGIES => {
                compute_and_configure_fact_topologies(vm, exec_scopes)
            }
            BOOTLOADER_SET_PACKED_OUTPUT_TO_SUBTASKS => set_packed_output_to_subtasks(exec_scopes),
            BOOTLOADER_IMPORT_PACKED_OUTPUT_SCHEMAS => import_packed_output_schemas(),
            BOOTLOADER_IS_PLAIN_PACKED_OUTPUT => is_plain_packed_output(vm, exec_scopes),
            BOOTLOADER_ASSERT_IS_COMPOSITE_PACKED_OUTPUT => {
                assert_is_composite_packed_output(exec_scopes)
            }
            SIMPLE_BOOTLOADER_PREPARE_TASK_RANGE_CHECKS => {
                prepare_task_range_checks(vm, exec_scopes, ids_data, ap_tracking)
            }
            SIMPLE_BOOTLOADER_SET_TASKS_VARIABLE => set_tasks_variable(exec_scopes),
            SIMPLE_BOOTLOADER_DIVIDE_NUM_BY_2 => divide_num_by_2(vm, ids_data, ap_tracking),
            SIMPLE_BOOTLOADER_SET_CURRENT_TASK => {
                set_current_task(vm, exec_scopes, ids_data, ap_tracking)
            }
            SIMPLE_BOOTLOADER_ZERO => set_ap_to_zero(vm),
            EXECUTE_TASK_ALLOCATE_PROGRAM_DATA_SEGMENT => {
                allocate_program_data_segment(vm, exec_scopes, ids_data, ap_tracking)
            }
            EXECUTE_TASK_LOAD_PROGRAM => load_program_hint(vm, exec_scopes, ids_data, ap_tracking),
            EXECUTE_TASK_VALIDATE_HASH => validate_hash(vm, exec_scopes, ids_data, ap_tracking),
            EXECUTE_TASK_USE_POSEIDON => task_use_poseidon(vm, exec_scopes),
            EXECUTE_TASK_ASSERT_PROGRAM_ADDRESS => {
                assert_program_address(vm, exec_scopes, ids_data, ap_tracking)
            }
            EXECUTE_TASK_CALL_TASK | EXECUTE_TASK_CALL_TASK_V14 => {
                call_task(vm, exec_scopes, ids_data, ap_tracking)
            }
            EXECUTE_TASK_WRITE_RETURN_BUILTINS => {
                write_return_builtins_hint(vm, exec_scopes, ids_data, ap_tracking)
            }
            EXECUTE_TASK_EXIT_SCOPE => exit_scope(exec_scopes),
            EXECUTE_TASK_APPEND_FACT_TOPOLOGIES => {
                append_fact_topologies(vm, exec_scopes, ids_data, ap_tracking)
            }
            SELECT_BUILTINS_ENTER_SCOPE => {
                select_builtins_enter_scope(vm, exec_scopes, ids_data, ap_tracking)
            }
            INNER_SELECT_BUILTINS_SELECT_BUILTIN => {
                select_builtin(vm, exec_scopes, ids_data, ap_tracking)
            }
            // --- v0.14 simple bootloader hints ---
            BOOTLOADER_READ_SIMPLE_BOOTLOADER_INPUT => {
                // The simple bootloader input is inserted by the caller before
                // the run; assert it is present.
                exec_scopes
                    .get_ref::<crate::hints::types::SimpleBootloaderInput>(
                        vars::SIMPLE_BOOTLOADER_INPUT,
                    )
                    .map(|_| ())
            }
            BOOTLOADER_SIMPLE_BOOTLOADER_COMPUTE_FACT_TOPOLOGIES => {
                compute_and_configure_fact_topologies_simple(vm, exec_scopes)
            }
            SETUP_RUN_SIMPLE_BOOTLOADER_BEFORE_TASK_EXECUTION => {
                setup_run_simple_bootloader_before_task_execution(
                    vm,
                    exec_scopes,
                    ids_data,
                    ap_tracking,
                )
            }
            SIMPLE_BOOTLOADER_SET_CURRENT_TASK_V14 => {
                set_current_task(vm, exec_scopes, ids_data, ap_tracking)
            }
            SIMPLE_BOOTLOADER_PROGRAM_HASH_FUNCTION => {
                program_hash_function_to_ap(vm, exec_scopes)
            }
            DETERMINE_USE_PREV_HASH => {
                determine_use_prev_hash(vm, exec_scopes, ids_data, ap_tracking)
            }
            LOAD_PROGRAM_SEGMENT => {
                load_program_segment_hint(vm, exec_scopes, ids_data, ap_tracking)
            }
            SIMPLE_BOOTLOADER_SIMULATE_EC_OP => {
                simple_bootloader_simulate_ec_op(vm, ids_data, ap_tracking)
            }
            SIMULATE_EC_OP_FILL_MEM_WITH_BITS_OF_M => {
                simulate_ec_op_fill_mem_with_bits_of_m(vm, ids_data, ap_tracking, constants)
            }
            SIMULATE_EC_OP_ASSERT_FALSE => simulate_ec_op_assert_false(),
            SIMPLE_BOOTLOADER_SIMULATE_KECCAK => {
                simple_bootloader_simulate_keccak(vm, ids_data, ap_tracking)
            }
            SIMULATE_KECCAK_FILL_MEM_WITH_STATE => {
                simulate_keccak_fill_mem_with_state(vm, ids_data, ap_tracking)
            }
            SIMULATE_KECCAK_CALC_HIGH3_LOW3 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 3)
            }
            SIMULATE_KECCAK_CALC_HIGH6_LOW6 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 6)
            }
            SIMULATE_KECCAK_CALC_HIGH9_LOW9 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 9)
            }
            SIMULATE_KECCAK_CALC_HIGH12_LOW12 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 12)
            }
            SIMULATE_KECCAK_CALC_HIGH15_LOW15 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 15)
            }
            SIMULATE_KECCAK_CALC_HIGH18_LOW18 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 18)
            }
            SIMULATE_KECCAK_CALC_HIGH21_LOW21 => {
                simulate_keccak_calc_high_low(vm, ids_data, ap_tracking, 21)
            }
            SIMPLE_BOOTLOADER_SIMULATE_ECDSA => {
                simple_bootloader_simulate_ecdsa(vm, ids_data, ap_tracking)
            }
            SIMULATE_ECDSA_GET_R_AND_S => simulate_ecdsa_get_r_and_s(vm, ids_data, ap_tracking),
            SIMULATE_ECDSA_COMPUTE_W_WR_WZ => {
                simulate_ecdsa_compute_w_wr_wz(vm, ids_data, ap_tracking, constants)
            }
            SIMULATE_ECDSA_FILL_MEM_WITH_FELT_96_BIT_LIMBS => {
                simulate_ecdsa_fill_mem_with_felt_96_bit_limbs(vm, ids_data, ap_tracking)
            }
            unknown_hint_code => Err(HintError::UnknownHint(
                unknown_hint_code.to_string().into_boxed_str(),
            )),
        }
    }
}

impl ResourceTracker for MinimalBootloaderHintProcessor {}

/// A hint processor for use cases where we only care about the bootloader hints.
///
/// When executing a hint, this hint processor will first check the hints defined in this library,
/// then the ones defined in Cairo VM.
pub struct BootloaderHintProcessor {
    bootloader_hint_processor: MinimalBootloaderHintProcessor,
    builtin_hint_processor: BuiltinHintProcessor,
}

impl Default for BootloaderHintProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BootloaderHintProcessor {
    pub fn new() -> Self {
        Self {
            bootloader_hint_processor: MinimalBootloaderHintProcessor {},
            builtin_hint_processor: BuiltinHintProcessor::new_empty(),
        }
    }

    pub fn add_hint(&mut self, hint_code: String, hint_func: Rc<HintFunc>) {
        self.builtin_hint_processor
            .extra_hints
            .insert(hint_code, hint_func);
    }
}

impl HintProcessorLogic for BootloaderHintProcessor {
    fn execute_hint(
        &mut self,
        _vm: &mut VirtualMachine,
        _exec_scopes: &mut ExecutionScopes,
        hint_data: &Box<dyn Any>,
    ) -> Result<(), HintError> {
        // This method will never be called, but must be defined for `HintProcessorLogic`.

        let hint_data = hint_data.downcast_ref::<HintProcessorData>().unwrap();
        let hint_code = &hint_data.code;
        Err(HintError::UnknownHint(hint_code.clone().into_boxed_str()))
    }

    fn execute_hint_extensive(
        &mut self,
        vm: &mut VirtualMachine,
        exec_scopes: &mut ExecutionScopes,
        hint_data: &Box<dyn Any>,
    ) -> Result<HintExtension, HintError> {
        // Cascade through the internal hint processors until we find the hint implementation.

        match self
            .bootloader_hint_processor
            .execute_hint_extensive(vm, exec_scopes, hint_data)
        {
            Err(HintError::UnknownHint(_)) => {}
            result => {
                return result;
            }
        }

        self.builtin_hint_processor
            .execute_hint_extensive(vm, exec_scopes, hint_data)
    }
}

impl ResourceTracker for BootloaderHintProcessor {}
