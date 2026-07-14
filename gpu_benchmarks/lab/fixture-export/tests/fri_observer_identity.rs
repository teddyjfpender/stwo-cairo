use std::path::PathBuf;

use cairo_air::verifier::verify_cairo;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::fri::FriCommitObserver;
use stwo::prover::line::LineEvaluation;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prover::{
    prove_cairo, prove_cairo_with_fri_observer, ChannelHash, ProverParameters,
};
use stwo_cairo_serialize::CairoSerialize;

#[derive(Default)]
struct SimdOnlyObserver {
    logs: Vec<u32>,
}

impl FriCommitObserver<SimdBackend, Blake2sMerkleChannel> for SimdOnlyObserver {
    fn observe_inner_fold(
        &mut self,
        input: &LineEvaluation<SimdBackend>,
        _alpha: SecureField,
        _root: Blake2sHash,
        _channel: &Blake2sChannel,
    ) {
        self.logs.push(input.domain().log_size());
    }
}

fn fixture() -> ProverInput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../../stwo_cairo_prover/test_data/test_prove_verify_all_opcode_components/prover_input.json",
    );
    serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
    )
    .expect("deserialize all-opcodes ProverInput")
}

fn serialized_bytes(value: &impl CairoSerialize) -> Vec<u8> {
    let mut felts = Vec::new();
    value.serialize(&mut felts);
    felts
        .into_iter()
        .flat_map(|felt| felt.to_bytes_be())
        .collect()
}

#[test]
#[ignore = "set STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY=1 on a cheap host"]
fn complete_simd_cairo_proof_is_byte_identical_with_fri_observer() {
    assert!(
        matches!(
            std::env::var("STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY").as_deref(),
            Ok("1")
        ),
        "the full-proof observer test requires its explicit cheap-host gate"
    );
    let params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config: PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: None,
        },
        preprocessed_trace: PreProcessedTraceVariant::CanonicalSmall,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };

    let ordinary = prove_cairo::<SimdBackend, Blake2sMerkleChannel>(fixture(), params).unwrap();
    let ordinary_bincode = bincode::serialize(&ordinary).unwrap();
    let ordinary_transport = serialized_bytes(&ordinary);
    verify_cairo::<Blake2sMerkleChannel>(ordinary.into()).unwrap();

    let mut observer = SimdOnlyObserver::default();
    let observed = prove_cairo_with_fri_observer::<SimdBackend, Blake2sMerkleChannel, _>(
        fixture(),
        params,
        &mut observer,
    )
    .unwrap();

    assert!(
        !observer.logs.is_empty(),
        "SIMD FRI observer was not called"
    );
    assert_eq!(
        ordinary_bincode,
        bincode::serialize(&observed).unwrap(),
        "FRI observation changed the full extended proof bytes"
    );
    assert_eq!(
        ordinary_transport,
        serialized_bytes(&observed),
        "FRI observation changed canonical Cairo proof bytes"
    );
    verify_cairo::<Blake2sMerkleChannel>(observed.into()).unwrap();
}
