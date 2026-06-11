//! Cairo e2e proving benchmark in the stwo-book / zkvm-benchmarks format.
//!
//! Methodology mirrors zksecurity/zkvm-benchmarks' stwo runner: the program is run in
//! the Cairo VM (proof mode, `iterations` delivered via the same `program_input` hint),
//! adapted in-process, and proven with the secure configuration (pow_bits=26,
//! blowup=1, 70 queries). Reported per run:
//!   proving time (cold + warm-best), proof size (bincode, as in their `utils::size`),
//!   cycle count (sum of opcode counts), peak host RSS, peak GPU VRAM, steps/second.
//!
//! Usage:
//!   gpu_bench --program path/to/compiled.json --iterations N --backend cuda|simd \
//!             [--reps 3]

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use cairo_air::verifier::verify_cairo;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use cairo_vm::cairo_run::{cairo_run_program, CairoRunConfig};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::{
    BuiltinHintProcessor, HintFunc,
};
use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::insert_value_from_var_name;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;
use cairo_vm::Felt252;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn peak_rss_gb() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let maxrss = unsafe { usage.assume_init() }.ru_maxrss as f64;
    // Linux reports KiB.
    maxrss / 1024.0 / 1024.0
}

fn run_vm(program_path: &str, iterations: u64) -> ProverInput {
    let program = Program::from_bytes(
        &std::fs::read(program_path).expect("program file"),
        Some("main"),
    )
    .expect("parse program");

    let mut hint_processor = BuiltinHintProcessor::new_empty();
    let input: HashMap<String, u64> = HashMap::from([("iterations".to_string(), iterations)]);
    hint_processor.add_hint(
        "ids.iterations = program_input['iterations']".to_string(),
        Rc::new(HintFunc(Box::new(
            move |vm, _exec_scopes, ids_data, ap_tracking, _constants| {
                insert_value_from_var_name(
                    "iterations",
                    Felt252::from(*input.get("iterations").unwrap()),
                    vm,
                    ids_data,
                    ap_tracking,
                )
                .unwrap();
                Ok(())
            },
        ))),
    );

    let config = CairoRunConfig {
        entrypoint: "main",
        trace_enabled: true,
        relocate_trace: false,
        layout: LayoutName::all_cairo_stwo,
        proof_mode: true,
        fill_holes: true,
        disable_trace_padding: true,
        ..Default::default()
    };
    let runner = cairo_run_program(&program, &config, &mut hint_processor).expect("vm run");
    let input = adapt(&runner).expect("adapt");
    // Adapter byte-equality harness: STWO_DUMP_INPUT=<path> serializes the adapted
    // ProverInput and exits — diff the dumps across adapter changes (the adapter has
    // no other content gate; ids and orders in it flow into the proof).
    if let Ok(path) = std::env::var("STWO_DUMP_INPUT") {
        let bytes = bincode::serialize(&input).expect("serialize prover input");
        std::fs::write(&path, &bytes).expect("write input dump");
        eprintln!("prover input dumped: {} bytes -> {path}", bytes.len());
        std::process::exit(0);
    }
    input
}

fn prover_params() -> ProverParameters {
    // The secure configuration used by zkvm-benchmarks' stwo runner.
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: None,
        },
        preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
        channel_salt: 0,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

fn main() {
    // STWO_BENCH_TRACE=1 prints every prover span with its duration on close;
    // aggregate externally to get the phase breakdown.
    if std::env::var("STWO_BENCH_TRACE").as_deref() == Ok("1") {
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::CLOSE)
            .with_target(false)
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .init();
    }
    let program = arg("--program").expect("--program <compiled.json>");
    let iterations: u64 = arg("--iterations")
        .expect("--iterations <n>")
        .parse()
        .unwrap();
    let backend = arg("--backend").unwrap_or_else(|| "cuda".to_string());
    let reps: usize = arg("--reps").unwrap_or_else(|| "3".to_string()).parse().unwrap();

    let input = run_vm(&program, iterations);
    let cycle_count: usize = input
        .state_transitions
        .casm_states_by_opcode
        .counts()
        .iter()
        .map(|(_, count)| *count)
        .sum();

    let mut times = Vec::new();
    let mut proof_size = 0usize;
    let mut verify_ms = 0.0f64;
    for rep in 0..reps {
        let input = run_vm(&program, iterations);
        let start = Instant::now();
        let proof = match backend.as_str() {
            "cuda" => prove_cairo::<stwo_backend_cuda::CudaBackend, Blake2sMerkleChannel>(
                input,
                prover_params(),
            )
            .unwrap(),
            "simd" => {
                prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, prover_params()).unwrap()
            }
            other => panic!("unknown backend {other}"),
        };
        let elapsed = start.elapsed().as_secs_f64();
        times.push(elapsed);
        if rep == 0 {
            proof_size = bincode::serialized_size(&proof).unwrap() as usize;
            let vstart = Instant::now();
            verify_cairo::<Blake2sMerkleChannel>(proof.into()).unwrap();
            verify_ms = vstart.elapsed().as_secs_f64() * 1000.0;
        }
        eprintln!("rep={rep} prove_s={elapsed:.3}");
    }
    let cold = times[0];
    let warm = times[1..].iter().cloned().fold(f64::INFINITY, f64::min);
    let warm = if warm.is_finite() { warm } else { cold };
    let (free, total) = stwo_backend_cuda::gpu_memory_info();
    let vram_gb = if total > 0 {
        (total - free) as f64 / 1e9
    } else {
        0.0
    };

    println!(
        "{{\"program\":\"{program}\",\"backend\":\"{backend}\",\"n\":{iterations},\
         \"cycle_count\":{cycle_count},\"prove_s_cold\":{cold:.3},\"prove_s_warm\":{warm:.3},\
         \"verify_ms\":{verify_ms:.1},\"proof_kb\":{:.1},\"peak_rss_gb\":{:.2},\
         \"vram_gb\":{vram_gb:.2},\"steps_per_s\":{:.0},\"mhz\":{:.3}}}",
        proof_size as f64 / 1024.0,
        peak_rss_gb(),
        cycle_count as f64 / warm,
        cycle_count as f64 / warm / 1e6,
    );
    // Silence unused-import warnings when only one backend path is exercised.
    let _ = CairoSerialize::serialize as fn(&u64, &mut Vec<starknet_ff::FieldElement>);
}
