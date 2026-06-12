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
//!             [--reps 3] [--pipeline]
//!
//! `--pipeline` (P5, throughput pipelining): the VM run + adapt of proof N+1
//! executes on a worker thread while proof N is being proven — pure
//! orchestration, every prove is the unmodified path on its own input, zero
//! soundness surface. Reports sustained throughput over the steady-state window
//! (after the first prove) alongside the per-proof numbers.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use cairo_air::verifier::verify_cairo;
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
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
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
    let reps: usize = arg("--reps")
        .unwrap_or_else(|| "3".to_string())
        .parse()
        .unwrap();
    let pipeline = flag("--pipeline");

    let input = run_vm(&program, iterations);
    let cycle_count: usize = input
        .state_transitions
        .casm_states_by_opcode
        .counts()
        .iter()
        .map(|(_, count)| *count)
        .sum();
    // Witness-port ranking input: per-opcode row counts (multiply by each
    // component's N_TRACE_COLUMNS to rank by mass). --counts-only skips proving.
    if std::env::var("STWO_OPCODE_COUNTS").as_deref() == Ok("1") || flag("--counts-only") {
        let mut counts = input.state_transitions.casm_states_by_opcode.counts();
        counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (name, count) in &counts {
            eprintln!("OPCODE_COUNT {name} {count}");
        }
        if flag("--counts-only") {
            return;
        }
    }

    let prove_once = |input| match backend.as_str() {
        "cuda" => prove_cairo::<stwo_backend_cuda::CudaBackend, Blake2sMerkleChannel>(
            input,
            prover_params(),
        )
        .unwrap(),
        "simd" => prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, prover_params()).unwrap(),
        other => panic!("unknown backend {other}"),
    };

    let mut times = Vec::new();
    let mut proof_size = 0usize;
    let mut verify_ms = 0.0f64;
    // P5 sustained-throughput window: wall seconds covering the last `reps - 1`
    // pipelined proves (the steady state, after the cold prove + verify).
    let mut sustained_s = None;
    if pipeline {
        // Proof N+k's VM run + adapt overlap proof N's prove: `--prefetch D`
        // (default 2) independent VM/adapt workers feed a bounded queue, so a
        // host-side input preparation slower than the prove no longer caps
        // sustained throughput. The prove path itself is untouched.
        let prefetch_depth: usize = arg("--prefetch")
            .map(|v| v.parse().expect("--prefetch <n>"))
            .unwrap_or(2)
            .max(1);
        let (tx, rx) = std::sync::mpsc::sync_channel(prefetch_depth);
        let remaining = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(reps));
        for _ in 0..prefetch_depth.min(reps) {
            let tx = tx.clone();
            let program = program.clone();
            let remaining = remaining.clone();
            std::thread::spawn(move || loop {
                // Claim one rep's input slot; stop when all are claimed.
                let claimed = remaining
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |n| n.checked_sub(1),
                    )
                    .is_ok();
                if !claimed || tx.send(run_vm(&program, iterations)).is_err() {
                    break;
                }
            });
        }
        drop(tx);

        let mut steady_start = None;
        for rep in 0..reps {
            let input = rx.recv().expect("pipelined vm/adapt");
            let start = Instant::now();
            let proof = prove_once(input);
            let elapsed = start.elapsed().as_secs_f64();
            times.push(elapsed);
            if rep == 0 {
                proof_size = bincode::serialized_size(&proof).unwrap() as usize;
                let vstart = Instant::now();
                verify_cairo::<Blake2sMerkleChannel>(proof.into()).unwrap();
                verify_ms = vstart.elapsed().as_secs_f64() * 1000.0;
                steady_start = Some(Instant::now());
            }
            eprintln!("rep={rep} prove_s={elapsed:.3} pipelined=1 prefetch={prefetch_depth}");
        }
        if reps > 1 {
            sustained_s = Some(steady_start.unwrap().elapsed().as_secs_f64());
        }
    } else {
        for rep in 0..reps {
            let input = run_vm(&program, iterations);
            let start = Instant::now();
            let proof = prove_once(input);
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

    // Sustained throughput (P5): proofs completed per wall second in the
    // steady-state pipelined window, expressed as cycles/s.
    let pipeline_fields = sustained_s
        .map(|s| {
            let sustained_mhz = (reps - 1) as f64 * cycle_count as f64 / s / 1e6;
            format!(",\"sustained_s\":{s:.3},\"sustained_mhz\":{sustained_mhz:.3}")
        })
        .unwrap_or_default();
    println!(
        "{{\"program\":\"{program}\",\"backend\":\"{backend}\",\"n\":{iterations},\
         \"cycle_count\":{cycle_count},\"prove_s_cold\":{cold:.3},\"prove_s_warm\":{warm:.3},\
         \"verify_ms\":{verify_ms:.1},\"proof_kb\":{:.1},\"peak_rss_gb\":{:.2},\
         \"vram_gb\":{vram_gb:.2},\"steps_per_s\":{:.0},\"mhz\":{:.3}{pipeline_fields}}}",
        proof_size as f64 / 1024.0,
        peak_rss_gb(),
        cycle_count as f64 / warm,
        cycle_count as f64 / warm / 1e6,
    );
    // Silence unused-import warnings when only one backend path is exercised.
    let _ = CairoSerialize::serialize as fn(&u64, &mut Vec<starknet_ff::FieldElement>);
}
