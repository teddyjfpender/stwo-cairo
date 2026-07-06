//! Cairo e2e proving benchmark in the stwo-book / zkvm-benchmarks format.
//!
//! Methodology mirrors zksecurity/zkvm-benchmarks' stwo runner: the program is run in
//! the Cairo VM (proof mode, `iterations` delivered via the same `program_input` hint),
//! adapted in-process, and proven with the secure configuration (pow_bits=26,
//! blowup=1, 70 queries — 96-bit). Reported per run (one self-describing JSON line):
//!   proving time (cold + warm-best), proof size (bincode, as in their `utils::size`),
//!   proved cycle count (sum of opcode counts) + proved MHz, useful PIE steps +
//!   useful MHz (PIE sources), VM/adapt phase seconds, peak host RSS, VRAM end
//!   snapshot + in-flight high-water (25ms sampler), steps/second, and the security
//!   configuration + host fingerprint (see JSON SCHEMA below).
//!
//! Usage:
//!   gpu_bench --program path/to/compiled.json --iterations N --backend cuda|simd \
//!             [--engine legacy|gpu-native] \
//!             [--reps 3] [--pipeline <depth>] [--reuse-input] [--adapt-only]
//!   gpu_bench --pie a.zip[,b.zip,...] [--pie-copies N] [--pie-mode aggregate|rotate] \
//!             [--producers N] --backend cuda|simd ...
//!             (requires building with --features pie-bench)
//!
//! `--engine` selects the prover pipeline: `legacy` (default — `prove_cairo`, the
//! parity oracle) or `gpu-native` (`stwo-cairo-gpu-prover`, the pipeline of
//! gpu_benchmarks/GPU_RESIDENT_PROVER_DESIGN.md §3). Proofs must be byte-identical
//! across engines at every milestone (design §9); records carry an `engine` field.
//!
//! `--pie` ingests CairoPie zips via the simple bootloader (SHARP-style aggregation:
//! all listed PIEs run as tasks of one bootloader execution; `--pie-copies` replicates
//! the whole list N times). `--reuse-input` clones one adapted ProverInput across reps
//! instead of re-running VM+adapt per rep (costs one extra resident input copy).
//! `--adapt-only` stops after VM+adapt and prints cycle count + overhead.
//!
//! PIPELINING (`--pipeline <depth>`, P5 in gpu_benchmarks/ROAD_TO_10MHZ.md) measures
//! SUSTAINED throughput: producer threads run the host-only VM run + adapt for
//! upcoming proofs while the current proof occupies the prover, with up to `depth`
//! ProverInputs buffered in a bounded channel. `--producers <N>` (default 1) spawns N
//! producer threads — a single-threaded bootloader VM run of a 14.6M-step PIE takes
//! far longer than its prove at target MHz, so one producer can never feed the GPU;
//! `feed_starved_s` in the pipeline record is the time the prover spent blocked
//! waiting for input after the first item (the diagnostic separating GPU-limited from
//! host-feed-limited sustained MHz). With a multi-PIE list, `--pie-mode` selects:
//!   aggregate (default): every load proves the WHOLE task list in one bootloader run
//!     (SHARP aggregation shape).
//!   rotate: producers round-robin over the list, one PIE per proof — the production
//!     fleet stream shape (one pod proving a stream of block PIEs). Per-rep cycle
//!     counts vary; sustained numbers are computed from per-rep totals.
//!
//! TRACING: STWO_BENCH_TRACE=1 prints every prover span with its duration on close
//! (raw, unchanged). STWO_BENCH_TRACE=json instead aggregates per-span-name
//! {count, total_ms} and prints one {"rep":N,"phase_totals":{...}} JSON object after
//! each rep (totals since the previous rep's drain; producer-thread spans land in the
//! rep during which they close).
//!
//! JSON SCHEMA (main record, one line per run):
//!   program, backend, n, cycle_count (proved cycles), pie_n_steps (null for
//!   --program), bootloader_overhead_pct (null for --program), prove_s_cold,
//!   prove_s_warm, verify_ms, proof_kb, peak_rss_gb, vram_end_gb, vram_peak_gb,
//!   steps_per_s, mhz (proved basis), useful_mhz (pie_n_steps/warm; null for
//!   --program), vm_s, adapt_s (from the last load), security_bits, n_queries,
//!   pow_bits, fold_step, gpu, nproc, host_mem_gb
//! Pipeline record (second line, only with --pipeline):
//!   pipeline, producers, pie_mode, reps, total_s, feed_starved_s,
//!   sustained_steps_per_s, sustained_mhz, sustained_useful_mhz (null for --program)

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use cairo_air::verifier::verify_cairo;
use cairo_air::CairoProof;
use cairo_vm::cairo_run::{cairo_run_program, CairoRunConfig};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::{
    BuiltinHintProcessor, HintFunc,
};
use cairo_vm::hint_processor::builtin_hint_processor::hint_utils::insert_value_from_var_name;
use cairo_vm::types::layout_name::LayoutName;
use cairo_vm::types::program::Program;
use cairo_vm::Felt252;
use serde_json::json;
use stwo::core::channel::MerkleChannel;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::{CairoBackend, GpuCairoProver, GpuProverConfig};
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

/// Prover engine: `legacy` (`prove_cairo` — the parity oracle) or `gpu-native`
/// (`stwo-cairo-gpu-prover`, GPU_RESIDENT_PROVER_DESIGN.md §3).
fn engine() -> String {
    arg("--engine").unwrap_or_else(|| "legacy".to_string())
}

/// The gpu-native engine's persistent prover contexts (one per backend type): reps
/// within a bench process share twiddle/preprocessed-tree caches, exactly as the
/// legacy engine's process-global statics do — warm-rep numbers stay comparable
/// across engines.
static GPU_NATIVE_CUDA: OnceLock<
    Mutex<GpuCairoProver<stwo_backend_cuda::CudaBackend, Blake2sMerkleChannel>>,
> = OnceLock::new();
static GPU_NATIVE_SIMD: OnceLock<Mutex<GpuCairoProver<SimdBackend, Blake2sMerkleChannel>>> =
    OnceLock::new();

fn prove_gpu_native<B, MC>(
    cell: &OnceLock<Mutex<GpuCairoProver<B, MC>>>,
    input: ProverInput,
    params: ProverParameters,
) -> CairoProof<MC::H>
where
    B: CairoBackend<MC>,
    MC: MerkleChannel + 'static,
{
    let prover = cell.get_or_init(|| {
        Mutex::new(GpuCairoProver::new(GpuProverConfig::default()).expect("gpu-native config"))
    });
    let mut prover = prover.lock().unwrap();
    prover
        .prove(input, params)
        .expect("gpu-native prove failed")
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Presence check for value-less boolean flags (`arg` would look at the next token).
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn peak_rss_gb() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let maxrss = unsafe { usage.assume_init() }.ru_maxrss as f64;
    // ru_maxrss is KiB on Linux, bytes on macOS.
    #[cfg(target_os = "macos")]
    return maxrss / 1024.0 / 1024.0 / 1024.0;
    #[cfg(not(target_os = "macos"))]
    return maxrss / 1024.0 / 1024.0;
}

/// GPU device name for the host fingerprint. The CUDA backend does not expose the
/// device name, so shell out to nvidia-smi (present on every CUDA pod); empty string
/// for simd or when unavailable.
fn gpu_name(backend: &str) -> String {
    if backend != "cuda" {
        return String::new();
    }
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
        .unwrap_or_default()
}

/// Total host memory in GB from /proc/meminfo (Linux pods); 0 elsewhere (mac).
fn host_mem_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1048576.0)
        .unwrap_or(0.0)
}

fn nproc() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

/// In-flight VRAM high-water sampler: polls `gpu_memory_info` every 25ms during a
/// prove and tracks max(total-free). The end-of-run snapshot misses the transient
/// peak, and peak-fit on a 24GB card decides the consumer-card cost story. Returns 0
/// when no CUDA device reports (`total == 0`, e.g. simd).
struct VramSampler {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<f64>,
}

impl VramSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let mut peak = 0f64;
            loop {
                let (free, total) = stwo_backend_cuda::gpu_memory_info();
                if total > 0 {
                    peak = peak.max((total - free) as f64 / 1e9);
                }
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            peak
        });
        Self { stop, handle }
    }

    /// Stop sampling and return the observed peak in GB.
    fn stop(self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.join().expect("vram sampler panicked")
    }
}

/// Per-span-name {count, total_ms, pool_gb_at_close} accumulator for
/// STWO_BENCH_TRACE=json. Drained and printed after each rep so the M1 measurement
/// program consumes structured phase totals instead of scraping raw span prints.
/// pool_gb_at_close is the driver-maintained pool used-high-water read at the span's
/// LAST close: the readings are cumulative-monotone across the prove, so consecutive
/// phases' deltas attribute pool growth per phase with zero interference (no resets)
/// — the VRAM-diet decomposition instrument.
type PhaseTotals = std::collections::BTreeMap<String, (u64, f64, f64)>;
static PHASE_AGG: OnceLock<Arc<Mutex<PhaseTotals>>> = OnceLock::new();

fn install_phase_agg_layer() {
    use tracing::span::{Attributes, Id};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::util::SubscriberInitExt;

    struct SpanStart(Instant);
    struct PhaseAggLayer(Arc<Mutex<PhaseTotals>>);

    impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for PhaseAggLayer {
        fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanStart(Instant::now()));
            }
        }

        fn on_close(&self, id: Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(&id) {
                if let Some(start) = span.extensions().get::<SpanStart>() {
                    let ms = start.0.elapsed().as_secs_f64() * 1000.0;
                    let name = match span.name() {
                        "" => "(unnamed)",
                        name => name,
                    };
                    let pool_gb = stwo_backend_cuda::gpu_pool_highwater().0 as f64 / 1e9;
                    let mut totals = self.0.lock().unwrap();
                    let entry = totals.entry(name.to_string()).or_insert((0, 0.0, 0.0));
                    entry.0 += 1;
                    entry.1 += ms;
                    entry.2 = pool_gb;
                }
            }
        }
    }

    let agg = Arc::new(Mutex::new(PhaseTotals::new()));
    PHASE_AGG
        .set(Arc::clone(&agg))
        .unwrap_or_else(|_| panic!("phase agg installed twice"));
    tracing_subscriber::registry()
        .with(PhaseAggLayer(agg))
        .init();
}

/// Print and reset accumulated phase totals (no-op unless STWO_BENCH_TRACE=json).
fn emit_phase_totals(rep: usize) {
    let Some(agg) = PHASE_AGG.get() else { return };
    let totals = std::mem::take(&mut *agg.lock().unwrap());
    let phases: serde_json::Map<String, serde_json::Value> = totals
        .into_iter()
        .map(|(name, (count, total_ms, pool_gb))| {
            (
                name,
                json!({"count": count, "total_ms": (total_ms * 1000.0).round() / 1000.0,
                       "pool_gb_at_close": (pool_gb * 1000.0).round() / 1000.0}),
            )
        })
        .collect();
    println!("{}", json!({"rep": rep, "phase_totals": phases}));
}

/// A loaded ProverInput plus the host-phase timings and (for PIE sources) the useful
/// step count carried by the PIE(s) themselves.
struct LoadedInput {
    input: ProverInput,
    /// Seconds spent in the (bootloader) VM run.
    vm_s: f64,
    /// Seconds spent in `adapt`.
    adapt_s: f64,
    /// Sum of `execution_resources.n_steps` over every task in the run (paths x
    /// copies); None for --program sources.
    pie_n_steps: Option<usize>,
}

fn run_vm(program_path: &str, iterations: u64) -> LoadedInput {
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
    let vm_start = Instant::now();
    let runner = cairo_run_program(&program, &config, &mut hint_processor).expect("vm run");
    let vm_s = vm_start.elapsed().as_secs_f64();
    let adapt_start = Instant::now();
    let input = adapt(&runner).expect("adapt");
    let adapt_s = adapt_start.elapsed().as_secs_f64();
    maybe_dump_and_exit(&input);
    LoadedInput {
        input,
        vm_s,
        adapt_s,
        pie_n_steps: None,
    }
}

/// Adapter byte-equality harness: STWO_DUMP_INPUT=<path> serializes the adapted
/// ProverInput and exits — diff the dumps across adapter changes (the adapter has
/// no other content gate; ids and orders in it flow into the proof). Shared by the
/// program and PIE input paths.
fn maybe_dump_and_exit(input: &ProverInput) {
    if let Ok(path) = std::env::var("STWO_DUMP_INPUT") {
        let bytes = bincode::serialize(input).expect("serialize prover input");
        std::fs::write(&path, &bytes).expect("write input dump");
        eprintln!("prover input dumped: {} bytes -> {path}", bytes.len());
        std::process::exit(0);
    }
}

/// Load one or more CairoPie zips through the simple bootloader and adapt in-process,
/// following the reference `load_pie_input`. The bootloader task list is built from
/// `pie_paths` in order — the production SHARP pattern aggregates many PIEs per
/// bootloader run. Each unique path is loaded once and Rc-shared across repeats (PIE
/// tasks can be GBs). `copies` replicates the WHOLE list N times to scale trace size.
/// Only compiled under the `pie-bench` feature (which pulls in
/// cairo-program-runner-lib).
#[cfg(feature = "pie-bench")]
fn load_pie_input(pie_paths: &[String], copies: usize) -> LoadedInput {
    use cairo_program_runner_lib::tasks::create_pie_task;
    use cairo_program_runner_lib::types::{HashFunc, RunMode};
    use cairo_program_runner_lib::{
        cairo_run_program as bootloader_run, ProgramInput, SimpleBootloaderInput, Task, TaskSpec,
    };
    use cairo_vm::types::program::Program as VmProgram;

    assert!(copies >= 1, "--pie-copies must be >= 1");
    assert!(!pie_paths.is_empty(), "--pie requires at least one path");
    let vm_start = Instant::now();
    let mut tasks_by_path = HashMap::new();
    let task_list: Vec<_> = pie_paths
        .iter()
        .map(|path| {
            Rc::clone(tasks_by_path.entry(path.as_str()).or_insert_with(|| {
                Rc::new(
                    create_pie_task(std::path::Path::new(path))
                        .unwrap_or_else(|e| panic!("Failed to load CairoPie zip {path}: {e:?}")),
                )
            }))
        })
        .collect();
    // Useful work carried by the PIEs themselves (vs the proved cycles, which add
    // bootloader overhead): sum of each task's own n_steps, over the whole task list.
    let pie_n_steps: usize = task_list
        .iter()
        .map(|task| match task.as_ref() {
            Task::Pie(pie) => pie.execution_resources.n_steps,
            _ => unreachable!("create_pie_task always yields Task::Pie"),
        })
        .sum::<usize>()
        * copies;
    let tasks = (0..copies)
        .flat_map(|_| task_list.iter().cloned())
        .map(|task| TaskSpec {
            task,
            program_hash_function: HashFunc::Blake,
        })
        .collect();
    let bootloader_input = SimpleBootloaderInput {
        fact_topologies_path: None,
        single_page: true,
        tasks,
    };

    // Runtime override first: the compile-time path is baked at build and can
    // point into a container layer that a pod stop/resume wipes ($HOME cargo
    // checkouts); STWO_BOOTLOADER_JSON relocates it without a rebuild.
    let bootloader_path = std::env::var("STWO_BOOTLOADER_JSON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("BOOTLOADER_JSON_PATH")));
    let bootloader_program = VmProgram::from_file(bootloader_path.as_path(), Some("main"))
        .expect("Failed to load simple_bootloader_compiled.json");

    let cairo_run_config = RunMode::Proof {
        layout: LayoutName::all_cairo_stwo,
        dynamic_layout_params: None,
        disable_trace_padding: true,
        relocate_mem: false,
    }
    .create_config();

    let runner = bootloader_run(
        &bootloader_program,
        Some(ProgramInput::Value(Box::new(bootloader_input))),
        cairo_run_config,
        None,
    )
    .expect("Failed to run PIE through bootloader");
    let vm_s = vm_start.elapsed().as_secs_f64();

    let adapt_start = Instant::now();
    let input = adapt(&runner).expect("Failed to adapt runner to ProverInput");
    let adapt_s = adapt_start.elapsed().as_secs_f64();
    maybe_dump_and_exit(&input);
    LoadedInput {
        input,
        vm_s,
        adapt_s,
        pie_n_steps: Some(pie_n_steps),
    }
}

/// The source of a ProverInput: either a compiled Cairo program run with `iterations`,
/// or one or more CairoPie zips ingested via the bootloader. Cloneable + Send so it can
/// be handed to pipeline producer threads.
#[derive(Clone)]
enum InputSource {
    Program {
        path: String,
        iterations: u64,
    },
    #[cfg(feature = "pie-bench")]
    Pie {
        paths: Vec<String>,
        copies: usize,
    },
}

impl InputSource {
    fn load(&self) -> LoadedInput {
        match self {
            InputSource::Program { path, iterations } => run_vm(path, *iterations),
            #[cfg(feature = "pie-bench")]
            InputSource::Pie { paths, copies } => load_pie_input(paths, *copies),
        }
    }

    /// Value reported in the `program` JSON field. For a multi-PIE task list, joins the
    /// zip basenames with '+' to keep the field sane.
    fn label(&self) -> String {
        match self {
            InputSource::Program { path, .. } => path.clone(),
            #[cfg(feature = "pie-bench")]
            InputSource::Pie { paths, .. } => paths
                .iter()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| p.clone())
                })
                .collect::<Vec<_>>()
                .join("+"),
        }
    }

    /// Value reported in the `n` JSON field (iterations for a program, copies for a PIE).
    fn n(&self) -> u64 {
        match self {
            InputSource::Program { iterations, .. } => *iterations,
            #[cfg(feature = "pie-bench")]
            InputSource::Pie { copies, .. } => *copies as u64,
        }
    }

    /// Preprocessed-trace variant required to prove this input. The compiled-program
    /// benchmarks use no pedersen builtin, so they keep `CanonicalWithoutPedersen`
    /// (unchanged behavior). CairoPie bootloader runs (e.g. Starknet transfers) use the
    /// pedersen builtin and require the full `Canonical` trace, which carries the
    /// pedersen points.
    fn preprocessed_variant(&self) -> PreProcessedTraceVariant {
        match self {
            InputSource::Program { .. } => PreProcessedTraceVariant::CanonicalWithoutPedersen,
            #[cfg(feature = "pie-bench")]
            InputSource::Pie { .. } => PreProcessedTraceVariant::Canonical,
        }
    }
}

fn cycle_count_of(input: &ProverInput) -> usize {
    input
        .state_transitions
        .casm_states_by_opcode
        .counts()
        .iter()
        .map(|(_, count)| *count)
        .sum()
}

fn prover_params(preprocessed_trace: PreProcessedTraceVariant) -> ProverParameters {
    // The secure configuration used by zkvm-benchmarks' stwo runner (96-bit:
    // pow_bits 26 + blowup 1 * 70 queries). Do not change.
    ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        pcs_config: PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, 3),
            lifting_log_size: None,
        },
        preprocessed_trace,
        channel_salt: 0,
        // STWO_STORE_COEFFS=1: OODS evaluates from stored coefficients instead of the
        // barycentric-weights path over committed evaluations — the OODS half of the
        // streamed-LDE VRAM diet (also drops the eval-domain-sized weights columns).
        // Pairs with STWO_FORCE_EXTEND_EVAL_MODE=1 (composition half).
        store_polynomials_coefficients: std::env::var("STWO_STORE_COEFFS").as_deref() == Ok("1"),
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Force the pedersen points tables (LazyLock statics) to initialize on the main
/// thread before any parallel witness generation. Their initializer runs nested rayon
/// joins; when rayon pool workers race the underlying Once during witness gen (pool
/// workers block on the Once while the initializer's subtasks queue behind them), the
/// prove deadlocks — observed nondeterministically on multi-task PIE runs. Warming
/// from a non-pool thread is race-free. Side effect: table generation no longer counts
/// toward rep-0 prove_s_cold on the PIE path (warm-best, the reported basis, was never
/// affected — the tables are cached after first touch).
fn prewarm_pedersen_tables(variant: PreProcessedTraceVariant) {
    if matches!(variant, PreProcessedTraceVariant::Canonical) {
        use std::sync::LazyLock;

        use stwo_cairo_common::preprocessed_columns::pedersen::{
            PEDERSEN_TABLE_18, PEDERSEN_TABLE_9,
        };
        LazyLock::force(&PEDERSEN_TABLE_9);
        LazyLock::force(&PEDERSEN_TABLE_18);
    }
}

/// The self-describing tail of every benchmark record: security configuration (pods
/// are heterogeneous and published numbers have used configs as weak as n_queries=3 —
/// never compare records without these fields) and host fingerprint.
fn record_context(backend: &str) -> serde_json::Value {
    let pcs = prover_params(PreProcessedTraceVariant::Canonical).pcs_config;
    json!({
        "security_bits": pcs.security_bits(),
        "n_queries": pcs.fri_config.n_queries,
        "pow_bits": pcs.pow_bits,
        "fold_step": pcs.fri_config.fold_step,
        "engine": engine(),
        "gpu": gpu_name(backend),
        "nproc": nproc(),
        "host_mem_gb": round3(host_mem_gb()),
    })
}

/// Merge `extra` into `base` (one level).
fn merge_json(mut base: serde_json::Value, extra: serde_json::Value) -> serde_json::Value {
    let (Some(b), Some(e)) = (base.as_object_mut(), extra.as_object().cloned()) else {
        panic!("merge_json expects objects");
    };
    b.extend(e);
    base
}

struct RepOutcome {
    times: Vec<f64>,
    proof_size: usize,
    verify_ms: f64,
    vram_peak_gb: f64,
}

/// Prove `input` on `backend`, sampling VRAM in flight. Returns (proof, prove_s,
/// vram_rep_peak_gb). The proof type is fixed by the channel, not the backend.
macro_rules! prove_sampled {
    ($backend:expr, $input:expr, $variant:expr) => {{
        let sampler = VramSampler::start();
        let start = Instant::now();
        let proof = match ($backend, engine().as_str()) {
            ("cuda", "legacy") => {
                prove_cairo::<stwo_backend_cuda::CudaBackend, Blake2sMerkleChannel>(
                    $input,
                    prover_params($variant),
                )
                .unwrap()
            }
            ("simd", "legacy") => {
                prove_cairo::<SimdBackend, Blake2sMerkleChannel>($input, prover_params($variant))
                    .unwrap()
            }
            ("cuda", "gpu-native") => {
                prove_gpu_native(&GPU_NATIVE_CUDA, $input, prover_params($variant))
            }
            ("simd", "gpu-native") => {
                prove_gpu_native(&GPU_NATIVE_SIMD, $input, prover_params($variant))
            }
            (backend, engine) => panic!("unknown backend/engine {backend}/{engine}"),
        };
        let elapsed = start.elapsed().as_secs_f64();
        (proof, elapsed, sampler.stop())
    }};
}

/// Emit the main benchmark record (shared by the standard and pipelined paths).
#[allow(clippy::too_many_arguments)]
fn print_main_record(
    program: &str,
    backend: &str,
    n: u64,
    cycle_count: usize,
    pie_n_steps: Option<usize>,
    outcome: &RepOutcome,
    vm_s: f64,
    adapt_s: f64,
) {
    let cold = outcome.times[0];
    let warm = outcome.times[1..]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let warm = if warm.is_finite() { warm } else { cold };
    let (free, total) = stwo_backend_cuda::gpu_memory_info();
    let vram_end_gb = if total > 0 {
        (total - free) as f64 / 1e9
    } else {
        0.0
    };

    let record = json!({
        "program": program,
        "backend": backend,
        "n": n,
        "cycle_count": cycle_count,
        "pie_n_steps": pie_n_steps,
        "bootloader_overhead_pct": pie_n_steps.map(|s| {
            round3((cycle_count as f64 - s as f64) / s as f64 * 100.0)
        }),
        "prove_s_cold": round3(cold),
        "prove_s_warm": round3(warm),
        "verify_ms": round3(outcome.verify_ms),
        "proof_kb": round3(outcome.proof_size as f64 / 1024.0),
        "peak_rss_gb": round3(peak_rss_gb()),
        "vram_end_gb": round3(vram_end_gb),
        "vram_peak_gb": round3(outcome.vram_peak_gb),
        // Driver-maintained pool high-water marks (exact; the 25ms sampler above
        // measured up to 11GB low on SN_PIE_2). The VRAM-diet metric of record.
        "pool_used_high_gb": round3(stwo_backend_cuda::gpu_pool_highwater().0 as f64 / 1e9),
        "pool_reserved_high_gb": round3(stwo_backend_cuda::gpu_pool_highwater().1 as f64 / 1e9),
        "steps_per_s": (cycle_count as f64 / warm).round(),
        "mhz": round3(cycle_count as f64 / warm / 1e6),
        "useful_mhz": pie_n_steps.map(|s| round3(s as f64 / warm / 1e6)),
        "vm_s": round3(vm_s),
        "adapt_s": round3(adapt_s),
    });
    println!("{}", merge_json(record, record_context(backend)));
}

#[derive(Clone, Copy, PartialEq)]
enum PieMode {
    Aggregate,
    Rotate,
}

/// P5 throughput pipelining (gpu_benchmarks/ROAD_TO_10MHZ.md): overlap the host-only
/// phases (VM run + adapt) of upcoming proofs with the prove of the current one. They
/// share no state: loads are pure host work, `prove_cairo` occupies the backend.
/// `producers` threads pull rep indices from a shared counter, build the rep's input
/// (aggregate: the whole task list; rotate: one PIE, round-robin), and feed a bounded
/// channel of `depth`; the main thread proves each as it arrives. The total wall clock
/// starts before the producers spawn, so pipeline fill counts. Proof-size capture and
/// rep-0 verification run after the clock stops to keep the sustained window pure.
fn run_pipelined(
    source: &InputSource,
    backend: &str,
    reps: usize,
    depth: usize,
    producers: usize,
    pie_mode: PieMode,
) {
    assert!(depth >= 1, "--pipeline depth must be >= 1");
    assert!(producers >= 1, "--producers must be >= 1");
    let program = source.label();
    let n = source.n();
    let variant = source.preprocessed_variant();
    prewarm_pedersen_tables(variant);

    // Per-rep input source: aggregate proves the full source every rep; rotate
    // round-robins one PIE per proof (the fleet stream shape).
    let rep_source: Arc<dyn Fn(usize) -> InputSource + Send + Sync> = match (pie_mode, source) {
        (PieMode::Aggregate, _) => {
            let source = source.clone();
            Arc::new(move |_| source.clone())
        }
        #[cfg(feature = "pie-bench")]
        (PieMode::Rotate, InputSource::Pie { paths, copies }) => {
            let (paths, copies) = (paths.clone(), *copies);
            Arc::new(move |rep| InputSource::Pie {
                paths: vec![paths[rep % paths.len()].clone()],
                copies,
            })
        }
        #[allow(unreachable_patterns)]
        (PieMode::Rotate, _) => panic!("--pie-mode rotate requires a --pie source"),
    };

    let total_start = Instant::now();
    let (tx, rx) = std::sync::mpsc::sync_channel::<LoadedInput>(depth);
    let next_rep = Arc::new(AtomicUsize::new(0));
    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let tx = tx.clone();
            let next_rep = Arc::clone(&next_rep);
            let rep_source = Arc::clone(&rep_source);
            std::thread::spawn(move || loop {
                let rep = next_rep.fetch_add(1, Ordering::Relaxed);
                if rep >= reps {
                    break;
                }
                // Everything (including Rc-holding bootloader tasks) is constructed
                // inside this thread; only the Send LoadedInput crosses the channel.
                // Stop early if the proving side dropped the receiver (e.g. panicked).
                if tx.send(rep_source(rep).load()).is_err() {
                    break;
                }
            })
        })
        .collect();
    drop(tx);

    let mut rep0_cycle_count = 0usize;
    let mut rep0_pie_n_steps = None;
    let mut total_cycles = 0usize;
    let mut total_pie_n_steps = 0usize;
    let mut have_pie_steps = false;
    let mut feed_starved_s = 0.0f64;
    let mut last_vm_s = 0.0f64;
    let mut last_adapt_s = 0.0f64;
    let mut vram_peak_gb = 0.0f64;
    let mut times = Vec::new();
    let mut first_proof = None;
    for rep in 0..reps {
        let recv_start = Instant::now();
        let loaded = rx.recv().expect("producer threads died");
        if rep > 0 {
            // Time the prover spent starved for input after the pipeline fill — the
            // diagnostic separating GPU-limited from host-feed-limited sustained MHz.
            feed_starved_s += recv_start.elapsed().as_secs_f64();
        }
        let cycle_count = cycle_count_of(&loaded.input);
        total_cycles += cycle_count;
        if let Some(steps) = loaded.pie_n_steps {
            total_pie_n_steps += steps;
            have_pie_steps = true;
        }
        if rep == 0 {
            rep0_cycle_count = cycle_count;
            rep0_pie_n_steps = loaded.pie_n_steps;
        }
        last_vm_s = loaded.vm_s;
        last_adapt_s = loaded.adapt_s;
        let (proof, elapsed, rep_vram) = prove_sampled!(backend, loaded.input, variant);
        vram_peak_gb = vram_peak_gb.max(rep_vram);
        times.push(elapsed);
        if rep == 0 {
            first_proof = Some(proof);
        }
        eprintln!("rep={rep} prove_s={elapsed:.3}");
        emit_phase_totals(rep);
    }
    let total_s = total_start.elapsed().as_secs_f64();
    for handle in producer_handles {
        handle.join().expect("producer thread panicked");
    }

    // Rep-0 proof size + verification, outside the timed window.
    let proof = first_proof.expect("--reps must be >= 1");
    // Byte-equality harness: STWO_DUMP_PROOF=<path> serializes the proof so the
    // device-lane ON vs OFF proofs can be `cmp`'d (witness-on-GPU kill-switch gate).
    if let Ok(path) = std::env::var("STWO_DUMP_PROOF") {
        let bytes = bincode::serialize(&proof).expect("serialize proof");
        std::fs::write(&path, &bytes).expect("write proof dump");
        eprintln!("proof dumped: {} bytes -> {path}", bytes.len());
    }
    let proof_size = bincode::serialized_size(&proof).unwrap() as usize;
    let vstart = Instant::now();
    verify_cairo::<Blake2sMerkleChannel>(proof.into()).unwrap();
    let verify_ms = vstart.elapsed().as_secs_f64() * 1000.0;

    let outcome = RepOutcome {
        times,
        proof_size,
        verify_ms,
        vram_peak_gb,
    };
    print_main_record(
        &program,
        backend,
        n,
        rep0_cycle_count,
        rep0_pie_n_steps,
        &outcome,
        last_vm_s,
        last_adapt_s,
    );
    println!(
        "{}",
        json!({
            "pipeline": depth,
            "producers": producers,
            "pie_mode": match pie_mode { PieMode::Aggregate => "aggregate", PieMode::Rotate => "rotate" },
            "reps": reps,
            "total_s": round3(total_s),
            "feed_starved_s": round3(feed_starved_s),
            "sustained_steps_per_s": (total_cycles as f64 / total_s).round(),
            "sustained_mhz": round3(total_cycles as f64 / total_s / 1e6),
            "sustained_useful_mhz": have_pie_steps
                .then(|| round3(total_pie_n_steps as f64 / total_s / 1e6)),
        })
    );
}

/// Resolve the input source from CLI flags. `--program`/`--iterations` selects the
/// compiled-program path; `--pie` (+ optional `--pie-copies`) selects the CairoPie
/// bootloader path and accepts a comma-separated list of zips (aggregated into one
/// bootloader run, in order — the production SHARP pattern). The two are mutually
/// exclusive. `--pie` is only available when built with the `pie-bench` feature.
fn build_input_source() -> InputSource {
    let program = arg("--program");
    let pie = arg("--pie");

    #[cfg(feature = "pie-bench")]
    {
        match (program, pie) {
            (Some(_), Some(_)) => {
                panic!("--program and --pie are mutually exclusive; pass exactly one")
            }
            (None, None) => {
                panic!(
                    "provide exactly one of --program <compiled.json> or \
                     --pie <a.zip[,b.zip,...]>"
                )
            }
            (Some(path), None) => {
                let iterations: u64 = arg("--iterations")
                    .expect("--iterations <n> required with --program")
                    .parse()
                    .expect("--iterations must be a u64");
                InputSource::Program { path, iterations }
            }
            (None, Some(pie_list)) => {
                let paths: Vec<String> = pie_list
                    .split(',')
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .map(str::to_string)
                    .collect();
                assert!(!paths.is_empty(), "--pie requires at least one path");
                let copies: usize = arg("--pie-copies")
                    .unwrap_or_else(|| "1".to_string())
                    .parse()
                    .expect("--pie-copies must be a usize");
                assert!(copies >= 1, "--pie-copies must be >= 1");
                InputSource::Pie { paths, copies }
            }
        }
    }

    #[cfg(not(feature = "pie-bench"))]
    {
        if pie.is_some() {
            panic!(
                "--pie requires the `pie-bench` feature; rebuild with \
                 `--features pie-bench`"
            );
        }
        let path = program.expect("--program <compiled.json>");
        let iterations: u64 = arg("--iterations")
            .expect("--iterations <n>")
            .parse()
            .expect("--iterations must be a u64");
        InputSource::Program { path, iterations }
    }
}

/// ENDGAME §2 keystone bring-up on hardware. Extracts this PIE's dedup'd memory tables
/// and the real `(pc, ap, fp)` states of the three recorded opcodes, then runs the
/// device-execution-tables + deduce_output + witness-JIT-launch differential in
/// `stwo_backend_cuda::exec_tables`. Exits non-zero on any mismatch.
fn run_witness_jit_selftest_hook(input: &ProverInput, vm_s: f64, adapt_s: f64) {
    use stwo_cairo_common::prover_types::cpu::CasmState;

    let mem = &input.memory;
    let addr_to_id: Vec<u32> = mem.address_to_id.iter().map(|e| e.0).collect();
    let f252_values = mem.f252_values.clone();
    let small_values = mem.small_values.clone();

    // Cap states per component (bounds device buffers + D2H compare); 0 = all.
    let cap: usize = std::env::var("STWO_WITNESS_JIT_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500_000);
    let take = |v: &[CasmState]| -> Vec<(u32, u32, u32)> {
        let n = if cap == 0 { v.len() } else { cap.min(v.len()) };
        v[..n].iter().map(|s| (s.pc.0, s.ap.0, s.fp.0)).collect()
    };

    let st = &input.state_transitions.casm_states_by_opcode;
    let components: Vec<(&str, Vec<(u32, u32, u32)>)> = vec![
        ("add_opcode", take(&st.add_opcode)),
        ("assert_eq_opcode", take(&st.assert_eq_opcode)),
        ("jnz_opcode_taken", take(&st.jnz_opcode_taken)),
        ("add_opcode_small", take(&st.add_opcode_small)),
        ("assert_eq_opcode_imm", take(&st.assert_eq_opcode_imm)),
        (
            "assert_eq_opcode_double_deref",
            take(&st.assert_eq_opcode_double_deref),
        ),
        ("call_opcode_abs", take(&st.call_opcode_abs)),
        ("call_opcode_rel_imm", take(&st.call_opcode_rel_imm)),
        ("jnz_opcode_non_taken", take(&st.jnz_opcode_non_taken)),
        ("jump_opcode_abs", take(&st.jump_opcode_abs)),
        (
            "jump_opcode_double_deref",
            take(&st.jump_opcode_double_deref),
        ),
        ("jump_opcode_rel", take(&st.jump_opcode_rel)),
        ("jump_opcode_rel_imm", take(&st.jump_opcode_rel_imm)),
        ("ret_opcode", take(&st.ret_opcode)),
    ];
    let max_deduce: usize = std::env::var("STWO_WITNESS_JIT_DEDUCE_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);

    eprintln!(
        "selftest inputs: vm_s={vm_s:.3} adapt_s={adapt_s:.3} n_addr={} n_big={} n_small={} \
         add_opcode={} assert_eq_opcode={} jnz_opcode_taken={} (cap={cap})",
        addr_to_id.len(),
        f252_values.len(),
        small_values.len(),
        st.add_opcode.len(),
        st.assert_eq_opcode.len(),
        st.jnz_opcode_taken.len(),
    );

    let ok = stwo_backend_cuda::exec_tables::run_witness_jit_selftest(
        &addr_to_id,
        &f252_values,
        &small_values,
        &components,
        max_deduce,
    );
    if !ok {
        std::process::exit(1);
    }
}

fn main() {
    // STWO_CAIRO_STREAM_LDE=1: the single switch for the streamed-LDE VRAM diet.
    // Fans out to its three constituent flags (they are interdependent: streaming
    // asserts stored coefficients, and composition must run the from-coefficients
    // path once evaluations are released at commit).
    if std::env::var("STWO_CAIRO_STREAM_LDE").as_deref() == Ok("1") {
        std::env::set_var("STWO_STORE_COEFFS", "1");
        std::env::set_var("STWO_FORCE_EXTEND_EVAL_MODE", "1");
    }
    // STWO_BENCH_TRACE=1 prints every prover span with its duration on close (raw,
    // aggregate externally); STWO_BENCH_TRACE=json aggregates per-span-name totals
    // in-process and prints a phase_totals JSON object after each rep.
    match std::env::var("STWO_BENCH_TRACE").as_deref() {
        Ok("1") => {
            use tracing_subscriber::fmt::format::FmtSpan;
            tracing_subscriber::fmt()
                .with_span_events(FmtSpan::CLOSE)
                .with_target(false)
                .with_ansi(false)
                .with_writer(std::io::stderr)
                .init();
        }
        Ok("json") => install_phase_agg_layer(),
        _ => {}
    }
    let backend = arg("--backend").unwrap_or_else(|| "cuda".to_string());
    let reps: usize = arg("--reps")
        .unwrap_or_else(|| "3".to_string())
        .parse()
        .unwrap();

    let source = build_input_source();
    let program = source.label();
    let n = source.n();

    // --adapt-only: run VM + adapt, report the cycle count, and exit before proving.
    // Useful for validating that large PIEs are accepted by the bootloader/adapter
    // without paying for a full (slow) SIMD prove locally.
    if flag("--adapt-only") || std::env::var("STWO_ADAPT_ONLY").as_deref() == Ok("1") {
        let loaded = source.load();
        let cycle_count = cycle_count_of(&loaded.input);
        println!(
            "{}",
            json!({
                "program": program,
                "n": n,
                "adapt_only": true,
                "cycle_count": cycle_count,
                "pie_n_steps": loaded.pie_n_steps,
                "bootloader_overhead_pct": loaded.pie_n_steps.map(|s| {
                    round3((cycle_count as f64 - s as f64) / s as f64 * 100.0)
                }),
                "vm_s": round3(loaded.vm_s),
                "adapt_s": round3(loaded.adapt_s),
            })
        );
        return;
    }

    let pie_mode = match arg("--pie-mode").as_deref() {
        None | Some("aggregate") => PieMode::Aggregate,
        Some("rotate") => PieMode::Rotate,
        Some(other) => panic!("--pie-mode must be aggregate or rotate, got {other}"),
    };
    let producers: usize = arg("--producers")
        .unwrap_or_else(|| "1".to_string())
        .parse()
        .expect("--producers must be a usize");

    // P5 sustained-throughput mode; absent flag keeps today's behavior exactly.
    if let Some(depth) = arg("--pipeline") {
        let depth: usize = depth.parse().expect("--pipeline <depth>");
        run_pipelined(&source, &backend, reps, depth, producers, pie_mode);
        return;
    }
    assert!(
        producers == 1,
        "--producers requires --pipeline (the standard path proves serially)"
    );
    assert!(
        pie_mode == PieMode::Aggregate,
        "--pie-mode rotate requires --pipeline (the standard path has no stream to rotate over)"
    );

    let loaded = source.load();

    // STWO_WITNESS_JIT_SELFTEST=1: bring up the ENDGAME §2 keystone on hardware —
    // device execution tables + device deduce_output differential + the witness-JIT
    // launch seam for the three recorded components — using this PIE's real memory and
    // states, then exit before proving (fast iteration). See
    // stwo_backend_cuda::exec_tables.
    if std::env::var("STWO_WITNESS_JIT_SELFTEST").as_deref() == Ok("1") {
        // STWO_WITNESS_JIT_SOURCE=emitted: register the transformer-EMITTED full-width
        // writer recordings (strictly wider than the built-in hand decode-subsets:
        // e.g. add_opcode 103 columns vs 14) so the device selftest runs
        // machine-generated bytecode — the automated-witness bridge.
        if std::env::var("STWO_WITNESS_JIT_SOURCE").as_deref() == Ok("emitted") {
            for (label, rec) in stwo_cairo_prover::witness::jit_witness_hook::emitted_recordings() {
                stwo_backend_cuda::jit_witness::register_recorded_program(label, rec.program);
            }
        }
        run_witness_jit_selftest_hook(&loaded.input, loaded.vm_s, loaded.adapt_s);
        return;
    }

    // STWO_DEVICE_INTERACTION_SELFTEST=1: the §6a differential — interaction trace
    // built from the SAME device lookup words via the host path and the device
    // path (logup_pairs.cu + device finalize), byte-compared. Exits before proving.
    if std::env::var("STWO_DEVICE_INTERACTION_SELFTEST").as_deref() == Ok("1") {
        let ok = stwo_cairo_prover::witness::jit_witness_hook::run_device_interaction_selftest(
            &loaded.input,
        );
        if !ok {
            std::process::exit(1);
        }
        return;
    }

    let cycle_count: usize = cycle_count_of(&loaded.input);
    let pie_n_steps = loaded.pie_n_steps;
    let variant = source.preprocessed_variant();
    prewarm_pedersen_tables(variant);

    // --reuse-input: load once and clone the ProverInput per rep instead of re-running
    // the VM + adapt every rep. On multi-million-step PIEs the reload is minutes of
    // dead wall-clock per rep on a paid pod. Default OFF to preserve the
    // zkvm-benchmarks methodology (each rep exercises the full host path). Cost: one
    // extra resident copy of the ProverInput (peak RSS grows by roughly the input's
    // in-memory size) while a rep is proving.
    let (mut last_vm_s, mut last_adapt_s) = (loaded.vm_s, loaded.adapt_s);
    let reusable_input = if flag("--reuse-input") {
        Some(loaded.input)
    } else {
        None
    };

    let mut times = Vec::new();
    let mut proof_size = 0usize;
    let mut verify_ms = 0.0f64;
    let mut vram_peak_gb = 0.0f64;
    for rep in 0..reps {
        let input = match &reusable_input {
            Some(input) => input.clone(),
            None => {
                let loaded = source.load();
                (last_vm_s, last_adapt_s) = (loaded.vm_s, loaded.adapt_s);
                loaded.input
            }
        };
        let (proof, elapsed, rep_vram) = prove_sampled!(backend.as_str(), input, variant);
        vram_peak_gb = vram_peak_gb.max(rep_vram);
        times.push(elapsed);
        if rep == 0 {
            proof_size = bincode::serialized_size(&proof).unwrap() as usize;
            // Byte-equality harness: STWO_DUMP_PROOF=<path> serializes the proof so the
            // device-lane ON vs OFF proofs can be `cmp`'d (witness-on-GPU kill switch).
            if let Ok(path) = std::env::var("STWO_DUMP_PROOF") {
                let bytes = bincode::serialize(&proof).expect("serialize proof");
                std::fs::write(&path, &bytes).expect("write proof dump");
                eprintln!("proof dumped: {} bytes -> {path}", bytes.len());
            }
            let vstart = Instant::now();
            verify_cairo::<Blake2sMerkleChannel>(proof.into()).unwrap();
            verify_ms = vstart.elapsed().as_secs_f64() * 1000.0;
        }
        eprintln!("rep={rep} prove_s={elapsed:.3}");
        emit_phase_totals(rep);
    }
    let outcome = RepOutcome {
        times,
        proof_size,
        verify_ms,
        vram_peak_gb,
    };
    print_main_record(
        &program,
        &backend,
        n,
        cycle_count,
        pie_n_steps,
        &outcome,
        last_vm_s,
        last_adapt_s,
    );
    // Silence unused-import warnings when only one backend path is exercised.
    let _ = CairoSerialize::serialize as fn(&u64, &mut Vec<starknet_ff::FieldElement>);
}
