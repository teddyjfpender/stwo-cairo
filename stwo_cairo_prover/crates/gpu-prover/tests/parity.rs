//! The M1 exit gate (design §9): the gpu-native pipeline's proof is BYTE-IDENTICAL
//! to the legacy `prove_cairo` proof on the same input and parameters.
//!
//! Runs on SimdBackend so the gate is LOCAL (no CUDA required); the pod manifest
//! runs the same A/B on CudaBackend via `gpu_bench --engine`. Warm-path parity
//! (second prove reusing the persistent caches) is asserted too — the persistent
//! context must not change bytes.

#![cfg(feature = "slow-tests")]

use cairo_vm::types::layout_name::LayoutName;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
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
fn gpu_native_parity_simd() {
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

    let legacy = prove_cairo::<SimdBackend, Blake2sMerkleChannel>(run_input(), params).unwrap();
    let legacy_felts = serialize_felts(&legacy);

    let mut prover =
        GpuCairoProver::<SimdBackend, Blake2sMerkleChannel>::new(GpuProverConfig::default())
            .unwrap();
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
