pub const BOOTLOADER_LOAD_INPUT: &str =
    "from starkware.cairo.bootloaders.bootloader.objects import BootloaderInput
bootloader_input = BootloaderInput.Schema().load(program_input)";

pub const BOOTLOADER_PREPARE_SIMPLE_BOOTLOADER_OUTPUT_SEGMENT: &str =
    "ids.simple_bootloader_output_start = segments.add()

# Change output builtin state to a different segment in preparation for calling the
# simple bootloader.
output_builtin_state = output_builtin.get_state()
output_builtin.new_state(base=ids.simple_bootloader_output_start)";

pub const BOOTLOADER_PREPARE_SIMPLE_BOOTLOADER_INPUT: &str =
    "simple_bootloader_input = bootloader_input";

pub const BOOTLOADER_RESTORE_BOOTLOADER_OUTPUT: &str =
    "# Restore the bootloader's output builtin state.
output_builtin.set_state(output_builtin_state)";

pub const BOOTLOADER_LOAD_BOOTLOADER_CONFIG: &str =
    "from starkware.cairo.bootloaders.bootloader.objects import BootloaderConfig
bootloader_config: BootloaderConfig = bootloader_input.bootloader_config

ids.bootloader_config = segments.gen_arg(
    [
        bootloader_config.simple_bootloader_program_hash,
        len(bootloader_config.supported_cairo_verifier_program_hashes),
        bootloader_config.supported_cairo_verifier_program_hashes,
    ],
)";

pub const BOOTLOADER_ENTER_PACKED_OUTPUT_SCOPE: &str =
    "from starkware.cairo.bootloaders.bootloader.objects import PackedOutput

task_id = len(packed_outputs) - ids.n_subtasks
packed_output: PackedOutput = packed_outputs[task_id]

vm_enter_scope(new_scope_locals=dict(packed_output=packed_output))";

pub const BOOTLOADER_IMPORT_PACKED_OUTPUT_SCHEMAS: &str =
    "from starkware.cairo.bootloaders.bootloader.objects import (
    CompositePackedOutput,
    PlainPackedOutput,
)";

// Appears as nondet %{ isinstance(packed_output, PlainPackedOutput) %} in the code.
pub const BOOTLOADER_IS_PLAIN_PACKED_OUTPUT: &str =
    "memory[ap] = to_felt_or_relocatable(isinstance(packed_output, PlainPackedOutput))";

pub const BOOTLOADER_SAVE_OUTPUT_POINTER: &str = "output_start = ids.output_ptr";

pub const BOOTLOADER_SAVE_PACKED_OUTPUTS: &str = "packed_outputs = bootloader_input.packed_outputs";

pub const BOOTLOADER_COMPUTE_FACT_TOPOLOGIES: &str = "from typing import List

from starkware.cairo.bootloaders.bootloader.utils import compute_fact_topologies
from starkware.cairo.bootloaders.fact_topology import FactTopology
from starkware.cairo.bootloaders.simple_bootloader.utils import (
    configure_fact_topologies,
    write_to_fact_topologies_file,
)

# Compute the fact topologies of the plain packed outputs based on packed_outputs and
# fact_topologies of the inner tasks.
plain_fact_topologies: List[FactTopology] = compute_fact_topologies(
    packed_outputs=packed_outputs, fact_topologies=fact_topologies,
)

# Configure the memory pages in the output builtin, based on plain_fact_topologies.
configure_fact_topologies(
    fact_topologies=plain_fact_topologies, output_start=output_start,
    output_builtin=output_builtin,
)

# Dump fact topologies to a json file.
if bootloader_input.fact_topologies_path is not None:
    write_to_fact_topologies_file(
        fact_topologies_path=bootloader_input.fact_topologies_path,
        fact_topologies=plain_fact_topologies,
    )";

pub const BOOTLOADER_GUESS_PRE_IMAGE_OF_SUBTASKS_OUTPUT_HASH: &str =
    "data = packed_output.elements_for_hash()
ids.nested_subtasks_output_len = len(data)
ids.nested_subtasks_output = segments.gen_arg(data)";

pub const BOOTLOADER_SET_PACKED_OUTPUT_TO_SUBTASKS: &str =
    "packed_outputs = packed_output.subtasks";

pub const BOOTLOADER_ASSERT_IS_COMPOSITE_PACKED_OUTPUT: &str =
    "assert isinstance(packed_output, CompositePackedOutput)";

pub const SIMPLE_BOOTLOADER_PREPARE_TASK_RANGE_CHECKS: &str =
    "n_tasks = len(simple_bootloader_input.tasks)
memory[ids.output_ptr] = n_tasks

# Task range checks are located right after simple bootloader validation range checks, and
# this is validated later in this function.
ids.task_range_check_ptr = ids.range_check_ptr + ids.BuiltinData.SIZE * n_tasks

# A list of fact_toplogies that instruct how to generate the fact from the program output
# for each task.
fact_topologies = []";

pub const SIMPLE_BOOTLOADER_SET_TASKS_VARIABLE: &str = "tasks = simple_bootloader_input.tasks";

// Appears as nondet %{ ids.num // 2 %} in the code.
pub const SIMPLE_BOOTLOADER_DIVIDE_NUM_BY_2: &str =
    "memory[ap] = to_felt_or_relocatable(ids.num // 2)";

pub const SIMPLE_BOOTLOADER_SET_CURRENT_TASK: &str =
    "from starkware.cairo.bootloaders.simple_bootloader.objects import Task

# Pass current task to execute_task.
task_id = len(simple_bootloader_input.tasks) - ids.n_tasks
task = simple_bootloader_input.tasks[task_id].load_task()";

// Appears as nondet %{ 0 %} in the code.
pub const SIMPLE_BOOTLOADER_ZERO: &str = "memory[ap] = to_felt_or_relocatable(0)";

pub const EXECUTE_TASK_ALLOCATE_PROGRAM_DATA_SEGMENT: &str =
    "ids.program_data_ptr = program_data_base = segments.add()";

pub const EXECUTE_TASK_LOAD_PROGRAM: &str =
    "from starkware.cairo.bootloaders.simple_bootloader.utils import load_program

# Call load_program to load the program header and code to memory.
program_address, program_data_size = load_program(
    task=task, memory=memory, program_header=ids.program_header,
    builtins_offset=ids.ProgramHeader.builtin_list)
segments.finalize(program_data_base.segment_index, program_data_size)";

pub const EXECUTE_TASK_VALIDATE_HASH: &str = "# Validate hash.
from starkware.cairo.bootloaders.hash_program import compute_program_hash_chain

assert memory[ids.output_ptr + 1] == compute_program_hash_chain(
    program=task.get_program(),
    use_poseidon=bool(ids.use_poseidon)), 'Computed hash does not match input.'";

pub const EXECUTE_TASK_USE_POSEIDON: &str =
    "memory[ap] = to_felt_or_relocatable(1 if task.use_poseidon else 0)";

pub const EXECUTE_TASK_ASSERT_PROGRAM_ADDRESS: &str = "# Sanity check.
assert ids.program_address == program_address";

pub const EXECUTE_TASK_CALL_TASK: &str =
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
vm_enter_scope(new_task_locals)";

/// v0.14 variant of `EXECUTE_TASK_CALL_TASK`: identical to the 0.13.3 version
/// except `load_cairo_pie(...)` takes the extra `ecdsa_additional_data=...`
/// argument. Dispatches to the same `call_task` handler (the Rust port already
/// relocates the PIE's ecdsa additional data into the signature builtin).
pub const EXECUTE_TASK_CALL_TASK_V14: &str =
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
        builtin_runners=builtin_runners, ret_fp=fp, ret_pc=ret_pc,
        ecdsa_additional_data=vm_ecdsa_additional_data)
else:
    raise NotImplementedError(f'Unexpected task type: {type(task).__name__}.')

output_runner_data = prepare_output_runner(
    task=task,
    output_builtin=output_builtin,
    output_ptr=ids.pre_execution_builtin_ptrs.output)
vm_enter_scope(new_task_locals)";

pub const EXECUTE_TASK_EXIT_SCOPE: &str = "vm_exit_scope()
# Note that bootloader_input will only be available in the next hint.";

pub const EXECUTE_TASK_WRITE_RETURN_BUILTINS: &str =
    "from starkware.cairo.bootloaders.simple_bootloader.utils import write_return_builtins

# Fill the values of all builtin pointers after executing the task.
builtins = task.get_program().builtins
write_return_builtins(
    memory=memory, return_builtins_addr=ids.return_builtin_ptrs.address_,
    used_builtins=builtins, used_builtins_addr=ids.used_builtins_addr,
    pre_execution_builtins_addr=ids.pre_execution_builtin_ptrs.address_, task=task)

vm_enter_scope({'n_selected_builtins': n_builtins})";

pub const EXECUTE_TASK_APPEND_FACT_TOPOLOGIES: &str =
    "from starkware.cairo.bootloaders.simple_bootloader.utils import get_task_fact_topology

# Add the fact topology of the current task to 'fact_topologies'.
output_start = ids.pre_execution_builtin_ptrs.output
output_end = ids.return_builtin_ptrs.output
fact_topologies.append(get_task_fact_topology(
    output_size=output_end - output_start,
    task=task,
    output_builtin=output_builtin,
    output_runner_data=output_runner_data,
))";

pub const SELECT_BUILTINS_ENTER_SCOPE: &str =
    "vm_enter_scope({'n_selected_builtins': ids.n_selected_builtins})";

// ---------------------------------------------------------------------------
// v0.14 simple bootloader hint codes.
// ---------------------------------------------------------------------------

/// Entry hint of `simple_bootloader.cairo` `main`. Normally loads
/// `simple_bootloader_input` from `program_input`; here the input is inserted
/// into the exec scope by the caller, so the hint just asserts it is present.
pub const BOOTLOADER_READ_SIMPLE_BOOTLOADER_INPUT: &str =
    "from starkware.cairo.bootloaders.simple_bootloader.objects import SimpleBootloaderInput
simple_bootloader_input = SimpleBootloaderInput.Schema().load(program_input)";

/// Final hint of `simple_bootloader.cairo` `main` — configures the output
/// builtin pages from the task fact topologies (output_start = base + 1).
pub const BOOTLOADER_SIMPLE_BOOTLOADER_COMPUTE_FACT_TOPOLOGIES: &str =
    "# Dump fact topologies to a json file.
from starkware.cairo.bootloaders.simple_bootloader.utils import (
    configure_fact_topologies,
    write_to_fact_topologies_file,
)

# The task-related output is prefixed by a single word that contains the number of tasks.
tasks_output_start = output_builtin.base + 1

if not simple_bootloader_input.single_page:
    # Configure the memory pages in the output builtin, based on fact_topologies.
    configure_fact_topologies(
        fact_topologies=fact_topologies, output_start=tasks_output_start,
        output_builtin=output_builtin,
    )

if simple_bootloader_input.fact_topologies_path is not None:
    write_to_fact_topologies_file(
        fact_topologies_path=simple_bootloader_input.fact_topologies_path,
        fact_topologies=fact_topologies,
    )";

/// Macro placeholder compiled into the v0.14 program (run_simple_bootloader.cairo):
/// writes n_tasks to output_ptr[0], sets initial_subtasks_range_check_ptr to a
/// new temporary segment, initializes fact_topologies = [].
pub const SETUP_RUN_SIMPLE_BOOTLOADER_BEFORE_TASK_EXECUTION: &str =
    "SETUP_RUN_SIMPLE_BOOTLOADER_BEFORE_TASK_EXECUTION";

/// Macro placeholder (run_simple_bootloader.cairo): loads the current task into
/// the `task` scope var and computes use_prev_hash / program_hash_function.
pub const SIMPLE_BOOTLOADER_SET_CURRENT_TASK_V14: &str = "SIMPLE_BOOTLOADER_SET_CURRENT_TASK";

/// `tempvar program_hash_function = nondet %{ task.program_hash_function %}`.
pub const SIMPLE_BOOTLOADER_PROGRAM_HASH_FUNCTION: &str =
    "memory[ap] = to_felt_or_relocatable(task.program_hash_function)";

/// Macro placeholder (execute_task.cairo:99): sets the `use_prev_hash` local.
pub const DETERMINE_USE_PREV_HASH: &str = "DETERMINE_USE_PREV_HASH";

/// Macro placeholder (execute_task.cairo:112): allocates the program segment and
/// loads the program header + code (0.14 merged allocate + load_program).
pub const LOAD_PROGRAM_SEGMENT: &str = "LOAD_PROGRAM_SEGMENT";

// --- keccak / ec_op / ecdsa builtin simulation (verify_builtins.cairo) ------

pub const SIMPLE_BOOTLOADER_SIMULATE_EC_OP: &str =
    "from starkware.cairo.lang.builtins.ec.ec_op_builtin_runner import (
    ec_op_auto_deduction_rule_wrapper,
)
ids.new_ec_op_ptr = segments.add()
vm_add_auto_deduction_rule(
    segment_index=ids.new_ec_op_ptr.segment_index,
    rule=ec_op_auto_deduction_rule_wrapper(ec_op_cache={}),
)";

pub const SIMULATE_EC_OP_FILL_MEM_WITH_BITS_OF_M: &str = "curr_m = ids.m
for i in range(ids.M_MAX_BITS):
    memory[ids.m_bit_unpacking + i] = curr_m % 2
    curr_m = curr_m >> 1";

pub const SIMULATE_EC_OP_ASSERT_FALSE: &str = "assert False, \"ec_op failed.\"";

pub const SIMPLE_BOOTLOADER_SIMULATE_KECCAK: &str =
    "from starkware.cairo.common.keccak_utils.keccak_utils import keccak_func
from starkware.cairo.lang.builtins.keccak.keccak_builtin_runner import (
    keccak_auto_deduction_rule_wrapper,
)
ids.new_keccak_ptr = segments.add()
vm_add_auto_deduction_rule(
    segment_index=ids.new_keccak_ptr.segment_index,
    rule=keccak_auto_deduction_rule_wrapper(keccak_cache={}),
)";

pub const SIMULATE_KECCAK_FILL_MEM_WITH_STATE: &str = "full_num = ids.keccak_builtin_state.s0
full_num += (2**200) * ids.keccak_builtin_state.s1
full_num += (2**400) * ids.keccak_builtin_state.s2
full_num += (2**600) * ids.keccak_builtin_state.s3
full_num += (2**800) * ids.keccak_builtin_state.s4
full_num += (2**1000) * ids.keccak_builtin_state.s5
full_num += (2**1200) * ids.keccak_builtin_state.s6
full_num += (2**1400) * ids.keccak_builtin_state.s7
for i in range(25):
    memory[ids.felt_array + i] = full_num % (2**64)
    full_num = full_num >> 64";

pub const SIMULATE_KECCAK_CALC_HIGH3_LOW3: &str =
    "ids.high3, ids.low3 = divmod(memory[ids.felt_array + 3], 256)";
pub const SIMULATE_KECCAK_CALC_HIGH6_LOW6: &str =
    "ids.high6, ids.low6 = divmod(memory[ids.felt_array + 6], 256 ** 2)";
pub const SIMULATE_KECCAK_CALC_HIGH9_LOW9: &str =
    "ids.high9, ids.low9 = divmod(memory[ids.felt_array + 9], 256 ** 3)";
pub const SIMULATE_KECCAK_CALC_HIGH12_LOW12: &str =
    "ids.high12, ids.low12 = divmod(memory[ids.felt_array + 12], 256 ** 4)";
pub const SIMULATE_KECCAK_CALC_HIGH15_LOW15: &str =
    "ids.high15, ids.low15 = divmod(memory[ids.felt_array + 15], 256 ** 5)";
pub const SIMULATE_KECCAK_CALC_HIGH18_LOW18: &str =
    "ids.high18, ids.low18 = divmod(memory[ids.felt_array + 18], 256 ** 6)";
pub const SIMULATE_KECCAK_CALC_HIGH21_LOW21: &str =
    "ids.high21, ids.low21 = divmod(memory[ids.felt_array + 21], 256 ** 7)";

pub const SIMPLE_BOOTLOADER_SIMULATE_ECDSA: &str =
    "from starkware.cairo.lang.builtins.signature.signature_builtin_runner import (
    signature_rule_wrapper,
)
from starkware.cairo.lang.vm.cairo_runner import verify_ecdsa_sig
ids.new_ecdsa_ptr = segments.add()
vm_add_validation_rule(
    segment_index=ids.new_ecdsa_ptr.segment_index,
    rule=signature_rule_wrapper(
        verify_signature_func=verify_ecdsa_sig,
        # Store signatures inside the vm's state. vm_ecdsa_additional_data is dropped
        # into the execution scope by the vm.
        signature_cache=vm_ecdsa_additional_data,
        ),
)";

pub const SIMULATE_ECDSA_GET_R_AND_S: &str =
    "(ids.r, ids.s) = vm_ecdsa_additional_data[ids.start.address_]";

pub const SIMULATE_ECDSA_COMPUTE_W_WR_WZ: &str =
    "# ids.StarkCurve.ORDER is parsed as a negative number.
order = ids.StarkCurve.ORDER + PRIME
ids.w = pow(ids.signature_s, -1, order)
ids.wz = ids.w*ids.message % order
ids.wr = ids.w*ids.signature_r % order";

pub const SIMULATE_ECDSA_FILL_MEM_WITH_FELT_96_BIT_LIMBS: &str = "num = ids.num
memory[ids.res_96_felts] = num % (2**96)
memory[ids.res_96_felts+1] = (num>>96) % (2**96)
memory[ids.res_96_felts+2] = (num>>(2*96)) % (2**96)";

pub const INNER_SELECT_BUILTINS_SELECT_BUILTIN: &str =
    "# A builtin should be selected iff its encoding appears in the selected encodings list
# and the list wasn't exhausted.
# Note that testing inclusion by a single comparison is possible since the lists are sorted.
ids.select_builtin = int(
  n_selected_builtins > 0 and memory[ids.selected_encodings] == memory[ids.all_encodings])
if ids.select_builtin:
  n_selected_builtins = n_selected_builtins - 1";
