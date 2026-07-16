%builtins output pedersen range_check ecdsa bitwise ec_op keccak poseidon range_check96 add_mod mul_mod

from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.bool import FALSE, TRUE

const COUNTER = 64;

func main{
    output_ptr,
    pedersen_ptr,
    range_check_ptr,
    ecdsa_ptr,
    bitwise_ptr,
    ec_op_ptr,
    keccak_ptr,
    poseidon_ptr,
    range_check96_ptr,
    add_mod_ptr,
    mul_mod_ptr,
}() {
    run_blake_test(is_last_block=FALSE);
    run_blake_test(is_last_block=TRUE);
    return ();
}

func run_blake_test{}(is_last_block: felt) {
    alloc_locals;

    let (local random_message) = alloc();
    assert random_message[0] = 930933030;
    assert random_message[1] = 1766240503;
    assert random_message[2] = 3660871006;
    assert random_message[3] = 388409270;
    assert random_message[4] = 1948594622;
    assert random_message[5] = 3119396969;
    assert random_message[6] = 3924579183;
    assert random_message[7] = 2089920034;
    assert random_message[8] = 3857888532;
    assert random_message[9] = 929304360;
    assert random_message[10] = 1810891574;
    assert random_message[11] = 860971754;
    assert random_message[12] = 1822893775;
    assert random_message[13] = 2008495810;
    assert random_message[14] = 2958962335;
    assert random_message[15] = 2340515744;

    let (local input_state) = alloc();
    assert input_state[0] = 0x6B08E647;
    assert input_state[1] = 0xBB67AE85;
    assert input_state[2] = 0x3C6EF372;
    assert input_state[3] = 0xA54FF53A;
    assert input_state[4] = 0x510E527F;
    assert input_state[5] = 0x9B05688C;
    assert input_state[6] = 0x1F83D9AB;
    assert input_state[7] = 0x5BE0CD19;

    let vm_output = run_blake_compress_opcode(
        is_last_block=is_last_block,
        dst=COUNTER,
        op0=input_state,
        op1=random_message,
    );

    tempvar check_nonempty = vm_output[0];
    tempvar check_nonempty = vm_output[1];
    tempvar check_nonempty = vm_output[2];
    tempvar check_nonempty = vm_output[3];
    tempvar check_nonempty = vm_output[4];
    tempvar check_nonempty = vm_output[5];
    tempvar check_nonempty = vm_output[6];
    tempvar check_nonempty = vm_output[7];

    return ();
}

func run_blake_compress_opcode(
    is_last_block: felt,
    dst: felt,
    op0: felt*,
    op1: felt*,
) -> felt* {
    alloc_locals;

    let offset0 = (2**15)-5;
    let offset1 = (2**15)-4;
    let offset2 = (2**15)-3;
    static_assert dst == [fp - 5];
    static_assert op0 == [fp - 4];
    static_assert op1 == [fp - 3];

    let flag_dst_base_fp = 1;
    let flag_op0_base_fp = 1;
    let flag_op1_imm = 0;
    let flag_op1_base_fp = 1;
    let flag_num = flag_dst_base_fp + flag_op0_base_fp*(2**1) + flag_op1_imm*(2**2) + flag_op1_base_fp*(2**3);
    let blake_compress_opcode_extension_num = 1;
    let blake_compress_last_block_opcode_extension_num = 2;
    let blake_compress_instruction_num = offset0 + offset1*(2**16) + offset2*(2**32) + flag_num*(2**48) + blake_compress_opcode_extension_num*(2**63);
    let blake_compress_last_block_instruction_num = offset0 + offset1*(2**16) + offset2*(2**32) + flag_num*(2**48) + blake_compress_last_block_opcode_extension_num*(2**63);
    static_assert blake_compress_instruction_num == 9226608988349300731;
    static_assert blake_compress_last_block_instruction_num == 18449981025204076539;

    let (local vm_output) = alloc();
    assert [ap] = cast(vm_output, felt);

    jmp last_block if is_last_block != 0;
    dw 9226608988349300731;
    return cast([ap], felt*);

    last_block:
    dw 18449981025204076539;
    return cast([ap], felt*);
}
