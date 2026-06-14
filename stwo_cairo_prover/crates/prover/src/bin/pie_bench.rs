//! Starknet Cairo-PIE proving benchmark — the primary benchmark from round 13.
//!
//! Runs a Cairo PIE (e.g. the Sepolia block bundles) through the vendored
//! bootloader in proof mode (builtins missing from the all_cairo_stwo layout,
//! like keccak, are emulated by the bootloader program itself), adapts the
//! runner, and proves with the chosen backend. Reports the gpu_bench JSON
//! metrics plus the VM/adapt split, so PIE and fib numbers stay comparable.
//!
//! Usage:
//!   pie_bench --pie <pie.zip> [--backend cuda|simd] [--reps N] [--counts-only]
//! Env: STWO_BENCH_TRACE=1 span timings; STWO_CAIRO_LOW_MEMORY=1 low-mem mode;
//!      STWO_OPCODE_COUNTS=1 opcode ranking dump.

use std::time::Instant;

use cairo_air::verifier::verify_cairo;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};

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
    maxrss / 1024.0 / 1024.0
}

fn prover_params() -> ProverParameters {
    // The gpu_bench secure configuration, with the FULL canonical preprocessed
    // trace: PIEs exercise the pedersen builtin, so its points tables must be
    // committed (the fib bench drops them to save VRAM).
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: None,
        },
        preprocessed_trace: PreProcessedTraceVariant::Canonical,
        channel_salt: 0,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

fn run_pie(pie_path: &str) -> (ProverInput, f64, f64) {
    let vm_start = Instant::now();
    let runner =
        stwo_cairo_bootloader::run_pie_with_simple_bootloader(std::path::Path::new(pie_path))
            .expect("bootloader run");
    let vm_s = vm_start.elapsed().as_secs_f64();
    let adapt_start = Instant::now();
    let input = adapt(&runner).expect("adapt");
    drop(runner);
    (input, vm_s, adapt_start.elapsed().as_secs_f64())
}

fn main() {
    if std::env::var("STWO_BENCH_TRACE").as_deref() == Ok("1") {
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::CLOSE)
            .with_target(false)
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .init();
    }
    let pie_path = arg("--pie").expect("--pie <pie.zip>");
    let backend = arg("--backend").unwrap_or_else(|| "cuda".to_string());
    let reps: usize = arg("--reps")
        .unwrap_or_else(|| "1".to_string())
        .parse()
        .unwrap();

    let (input, vm_s, adapt_s) = run_pie(&pie_path);

    let counts = input.state_transitions.casm_states_by_opcode.counts();
    let cycle_count: usize = counts.iter().map(|(_, count)| *count).sum();
    if std::env::var("STWO_OPCODE_COUNTS").as_deref() == Ok("1") || flag("--counts-only") {
        let mut sorted = counts.clone();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (name, count) in &sorted {
            eprintln!("OPCODE_COUNT {name} {count}");
        }
        eprintln!("BUILTIN_SEGMENTS {:?}", input.builtin_segments);
        if flag("--counts-only") {
            println!(
                "{{\"pie\":\"{pie_path}\",\"cycle_count\":{cycle_count},\"vm_s\":{vm_s:.3},\
                 \"adapt_s\":{adapt_s:.3}}}"
            );
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
    // rep 0 consumes the input already prepared above (NO clone — a multi-M-step
    // ProverInput holds the whole memory table, several GB; cloning it serially
    // is pure waste). Warm reps re-run the VM fresh.
    let mut prepared = Some(input);
    for rep in 0..reps {
        let rep_input = prepared.take().unwrap_or_else(|| run_pie(&pie_path).0);
        let start = Instant::now();
        let proof = prove_once(rep_input);
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
    // True high-water mark of reserved VRAM (peak, not end-of-run footprint).
    // With low-memory mode ON this should drop relative to OFF; the trim at the
    // spill point releases the spilled buffers back to the OS.
    let peak_vram_gb = stwo_backend_cuda::gpu_peak_vram_bytes() as f64 / 1e9;
    println!(
        "{{\"pie\":\"{pie_path}\",\"backend\":\"{backend}\",\"cycle_count\":{cycle_count},\
         \"vm_s\":{vm_s:.3},\"adapt_s\":{adapt_s:.3},\"prove_s_cold\":{cold:.3},\
         \"prove_s_warm\":{warm:.3},\"verify_ms\":{verify_ms:.1},\"proof_kb\":{:.1},\
         \"peak_rss_gb\":{:.2},\"vram_gb\":{vram_gb:.2},\"peak_vram_gb\":{peak_vram_gb:.2},\
         \"steps_per_s\":{:.0},\"mhz\":{:.3}}}",
        proof_size as f64 / 1024.0,
        peak_rss_gb(),
        cycle_count as f64 / warm,
        cycle_count as f64 / warm / 1e6,
    );
}
