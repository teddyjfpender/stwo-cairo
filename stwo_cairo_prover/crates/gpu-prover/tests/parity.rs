//! The M1 exit gate (design §9): the gpu-native pipeline's proof is BYTE-IDENTICAL
//! to the legacy `prove_cairo` proof on the same input and parameters.
//!
//! This is intentionally CUDA-only now: the GPU-native prover no longer accepts a
//! generic backend, while the legacy SIMD path remains available as a separate
//! reference oracle. Warm-path parity (second prove reusing persistent caches) is
//! asserted too — the persistent context must not change bytes.

#![cfg(feature = "slow-tests")]

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo_backend_cuda::CudaBackend;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::utils::get_compiled_cairo_program_path;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_gpu_prover::{GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

fn serialize_felts<H>(proof: &cairo_air::CairoProof<H>) -> Vec<starknet_ff::FieldElement>
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    H::Hash: CairoSerialize,
{
    let mut felts = Vec::new();
    CairoSerialize::serialize(proof, &mut felts);
    felts
}

#[test]
fn gpu_native_parity_cuda() {
    if stwo_backend_cuda::gpu_memory_info() == (0, 0) {
        eprintln!("CUDA unavailable; full gpu-native parity runs in the combined hardware gate");
        return;
    }
    let compiled_program =
        get_compiled_cairo_program_path("test_prove_verify_all_opcode_components");
    let run_input = || {
        run_and_adapt(
            &compiled_program,
            ProgramType::Json,
            LayoutName::all_cairo_stwo,
            None,
        )
        .unwrap()
    };
    let params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig::default(),
        preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
        channel_salt: 0,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };

    let legacy = prove_cairo::<CudaBackend, Blake2sMerkleChannel>(run_input(), params).unwrap();
    let legacy_felts = serialize_felts(&legacy);

    let mut prover =
        GpuCairoProver::<Blake2sMerkleChannel>::new(GpuProverConfig::default()).unwrap();
    let native = prover.prove(run_input(), params).unwrap();
    assert_eq!(
        legacy_felts,
        serialize_felts(&native),
        "gpu-native proof differs from legacy proof (cold context)"
    );

    // Warm path: the persistent twiddle/preprocessed caches must not change bytes.
    let native_warm = prover.prove(run_input(), params).unwrap();
    assert_eq!(
        legacy_felts,
        serialize_felts(&native_warm),
        "gpu-native proof differs from legacy proof (warm context)"
    );
}
