use stwo_circuit_air::{
    CircuitProof, CircuitVerifierConfig, VerificationOutput, get_verification_output,
    verify_circuit,
};

mod privacy_consts;

#[executable]
fn main(proof: CircuitProof) -> VerificationOutput {
    // The verifier-config constants are hardcoded for the privacy/recursion circuit
    // topology (see `privacy_consts.cairo`). Mirrors how `cairo_air::verify_cairo`
    // hardcodes its preprocessed root — the felt252 input stream contains only the
    // proof, not the config.
    let config = CircuitVerifierConfig {
        output_addresses: privacy_consts::output_addresses(),
        n_blake_gates: privacy_consts::N_BLAKE_GATES,
        preprocessed_root: privacy_consts::preprocessed_root(),
        preprocessed_column_log_sizes: privacy_consts::preprocessed_column_log_sizes(),
        lifting_log_size: privacy_consts::LIFTING_LOG_SIZE,
    };

    let verification_output = get_verification_output(proof: @proof);
    verify_circuit(:proof, config: @config);
    verification_output
}
