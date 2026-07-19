#![recursion_limit = "512"]

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
//!             [--resident-backend legacy-resident|replacement-v1] \
//!             [--packed-numerator-measurement-control] \
//!             [--compiled-composition-vertical-checkpoint] \
//!             [--reps 3] [--pipeline <depth>] [--reuse-input] [--adapt-only] \
//!             [--require-proof-byte-equal] [--require-gpu-native-architecture] \
//!             [--diagnostic-allow-slow-graph-submit] \
//!             [--capture-slow-graph-submit] \
//!             [--fleet-pow-socket /path/to/fleet-pow.sock] \
//!             [--operational-safety-reserve-bytes N] \
//!             [--require-simd-reference-byte-equal] \
//!             [--require-proof-mutation-rejected] \
//!             [--require-gpu-pcs-runtime-mode detached-eager|arena-graph]
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
//! `--require-proof-byte-equal` (or STWO_BENCH_REQUIRE_PROOF_BYTE_EQUAL=1) makes
//! same-statement repetition byte drift or an inapplicable comparison a non-zero
//! benchmark failure. Rotate-mode pipeline reps prove different statements, so their
//! comparison and per-repetition throughput distribution are reported null.
//! `--require-simd-reference-byte-equal` is a stronger, CUDA gpu-native standard-run
//! gate: outside the measured GPU window it computes and verifies one fresh SIMD
//! proof for the same adapted input and parameters, then exact-compares every
//! serialized GPU proof to it. The record reports this separately from same-backend
//! repetition determinism.
//! `--compiled-composition-vertical-checkpoint` selects the replacement backend's
//! eager end-to-end checkpoint. It is a correctness diagnostic: the default captured
//! path is unchanged, the SIMD byte oracle is mandatory, and its timing is always
//! labeled indicative/non-formal.
//! `--packed-numerator-measurement-control` selects the replacement backend's one
//! explicit packed numerator A/B baseline. Without it, replacement-v1 remains on its
//! production adaptive run-sum-or-packed policy. The control requires the same strict
//! CUDA gpu-native ArenaGraph admission as production replacement-v1.
//! `--require-proof-mutation-rejected` retains a verifier-form clone of repetition 0,
//! waits for the original to verify, adds one to the always-present memory-id
//! interaction claimed sum, and requires rejection. This is a verifier-integrity
//! gate, not arbitrary corruption of serialized transport bytes.
//! `--require-gpu-native-architecture` (or
//! STWO_BENCH_REQUIRE_GPU_NATIVE_ARCHITECTURE=1) is a fail-closed benchmark gate:
//! CUDA + gpu-native, the typed CUDA PCS driver, one start and finish for every
//! protocol stage, batched tree decommit, and complete telemetry are all required.
//! Its expected runtime mode defaults to `detached-eager` and can be selected with
//! `--require-gpu-pcs-runtime-mode` (or
//! STWO_BENCH_REQUIRE_GPU_PCS_RUNTIME_MODE). `arena-graph` is intentionally a strict
//! future gate: it rejects today's detached runtime instead of claiming graph capture.
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
//!   prove_s_warm (legacy warm-best), prove_s_warm_best,
//!   prove_s_warm_median, prove_s_warm_p95, mhz_median, useful_mhz_median,
//!   gpu_proof_loop_started_unix_ns, gpu_proof_loop_finished_unix_ns,
//!   throughput_distribution_applicable, proof_byte_equal, gpu_proof_blake3,
//!   proof_comparison_applicable, verified_reps,
//!   simd_reference_required, simd_reference_comparison_applicable,
//!   simd_reference_byte_equal, simd_reference_blake3,
//!   simd_reference_fresh, simd_reference_s,
//!   proof_mutation_required, proof_mutation_kind,
//!   proof_mutation_rejected, proof_mutation_error_class,
//!   gpu_resident_backend_requested,
//!   gpu_packed_numerator_measurement_control_requested,
//!   gpu_pcs_driver_architecture, gpu_pcs_runtime_mode,
//!   gpu_pcs_stage_started, gpu_pcs_stage_finished,
//!   gpu_pcs_batched_tree_decommit, gpu_pcs_driver_complete,
//!   gpu_native_architecture_required, gpu_pcs_required_runtime_mode,
//!   gpu_native_architecture_gate_passed,
//!   gpu_aot_loads, gpu_aot_cache_hits, gpu_aot_manifest_hash, gpu_aot_misses,
//!   gpu_aot_runtime_loads, gpu_aot_runtime_cache_hits,
//!   gpu_aot_strict_rejections, gpu_aot_provenance_gate_passed,
//!   verify_ms, proof_kb, peak_rss_gb, vram_end_gb, vram_peak_gb,
//!   steps_per_s, mhz (proved basis), useful_mhz (pie_n_steps/warm; null for
//!   --program), vm_s, adapt_s (from the last load), security_bits, n_queries,
//!   pow_bits, fold_step, gpu, nproc, host_mem_gb
//! Pipeline record (second line, only with --pipeline):
//!   pipeline, producers, pie_mode, reps, total_s, feed_starved_s,
//!   sustained_steps_per_s, sustained_mhz, sustained_useful_mhz (null for --program)

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cairo_air::verifier::{verify_cairo, CairoVerificationError};
use cairo_air::{CairoProof, CairoProofForRustVerifier};
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
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_gpu_prover::arena_plan::ResidentBackend;
use stwo_cairo_gpu_prover::fleet_pow::{FleetPowPlan, FleetPowSchedule};
use stwo_cairo_gpu_prover::fleet_pow_unix::FleetPowUnixTransport;
use stwo_cairo_gpu_prover::{
    CudaPcsDriverTelemetry, CudaPcsRuntimeMode, FleetResidentProofTelemetry, GpuCairoProver,
    GpuProverConfig, ResidentSessionTelemetry,
};
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_serialize::CairoSerialize;

#[path = "../gpu_bench_physical.rs"]
mod gpu_bench_physical;
use gpu_bench_physical::gpu_native_session_context;

type BenchProof = CairoProof<<Blake2sMerkleChannel as MerkleChannel>::H>;
type BenchVerifierProof = CairoProofForRustVerifier<<Blake2sMerkleChannel as MerkleChannel>::H>;
type AotRuntimeStats = stwo_backend_cuda::aot::RuntimeStats;

const REQUIRED_CUDA_PCS_ARCHITECTURE: &str = "cuda-typed-pcs-driver-v1";
const PROOF_MUTATION_KIND: &str = "interaction_claim.memory_id_to_big.claimed_sum_plus_one";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequiredCudaPcsRuntimeMode {
    DetachedEager,
    ArenaGraph,
}

impl RequiredCudaPcsRuntimeMode {
    fn parse(value: &str) -> Self {
        match value {
            "detached-eager" => Self::DetachedEager,
            "arena-graph" => Self::ArenaGraph,
            other => panic!(
                "--require-gpu-pcs-runtime-mode must be detached-eager or arena-graph, got {other}"
            ),
        }
    }

    const fn telemetry_mode(self) -> CudaPcsRuntimeMode {
        match self {
            Self::DetachedEager => CudaPcsRuntimeMode::DetachedEager,
            Self::ArenaGraph => CudaPcsRuntimeMode::ArenaGraph,
        }
    }

    const fn cli_name(self) -> &'static str {
        match self {
            Self::DetachedEager => "detached-eager",
            Self::ArenaGraph => "arena-graph",
        }
    }
}

fn parse_resident_backend_args<I, S>(args: I) -> Result<ResidentBackend, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let mut selected = None;
    while let Some(argument) = args.next() {
        let argument = argument.as_ref();
        if argument.starts_with("--resident-backend=") {
            return Err(
                "--resident-backend requires a separate value: legacy-resident or replacement-v1"
                    .to_owned(),
            );
        }
        if argument != "--resident-backend" {
            continue;
        }
        if selected.is_some() {
            return Err("--resident-backend may be passed only once".to_owned());
        }
        let value = args
            .next()
            .ok_or_else(|| "--resident-backend requires a value".to_owned())?;
        let value = value.as_ref();
        if value.starts_with("--") {
            return Err("--resident-backend requires a value".to_owned());
        }
        selected = Some(match value {
            "legacy-resident" => ResidentBackend::LegacyResident,
            "replacement-v1" => ResidentBackend::ReplacementV1,
            other => {
                return Err(format!(
                    "--resident-backend must be legacy-resident or replacement-v1, got {other}"
                ))
            }
        });
    }
    Ok(selected.unwrap_or_default())
}

fn parse_compiled_composition_vertical_checkpoint_args<I, S>(args: I) -> Result<bool, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    const FLAG: &str = "--compiled-composition-vertical-checkpoint";
    const VALUE_FORM: &str = "--compiled-composition-vertical-checkpoint=";
    let mut selected = false;
    for argument in args {
        let argument = argument.as_ref();
        if argument.starts_with(VALUE_FORM) {
            return Err(format!("{FLAG} is a value-less flag"));
        }
        if argument != FLAG {
            continue;
        }
        if selected {
            return Err(format!("{FLAG} may be passed only once"));
        }
        selected = true;
    }
    Ok(selected)
}

fn parse_packed_numerator_measurement_control_args<I, S>(args: I) -> Result<bool, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    const FLAG: &str = "--packed-numerator-measurement-control";
    const VALUE_FORM: &str = "--packed-numerator-measurement-control=";
    let mut selected = false;
    for argument in args {
        let argument = argument.as_ref();
        if argument.starts_with(VALUE_FORM) {
            return Err(format!("{FLAG} is a value-less flag"));
        }
        if argument != FLAG {
            continue;
        }
        if selected {
            return Err(format!("{FLAG} may be passed only once"));
        }
        selected = true;
    }
    Ok(selected)
}

fn packed_numerator_measurement_control_gate(
    enabled: bool,
    backend: &str,
    selected_engine: &str,
    resident_backend: ResidentBackend,
    architecture_required: bool,
    runtime_mode: RequiredCudaPcsRuntimeMode,
) -> Result<(), &'static str> {
    if !enabled {
        return Ok(());
    }
    if backend != "cuda" {
        return Err("--packed-numerator-measurement-control requires --backend cuda");
    }
    if selected_engine != "gpu-native" {
        return Err("--packed-numerator-measurement-control requires --engine gpu-native");
    }
    if resident_backend != ResidentBackend::ReplacementV1 {
        return Err(
            "--packed-numerator-measurement-control requires --resident-backend replacement-v1",
        );
    }
    if !architecture_required || runtime_mode != RequiredCudaPcsRuntimeMode::ArenaGraph {
        return Err(
            "--packed-numerator-measurement-control requires --require-gpu-native-architecture and --require-gpu-pcs-runtime-mode arena-graph",
        );
    }
    Ok(())
}

fn compiled_composition_vertical_checkpoint_gate(
    enabled: bool,
    resident_backend: ResidentBackend,
    simd_reference: bool,
    reuse_input: bool,
    fleet_pow: bool,
    graph_submit_diagnostic: bool,
    graph_submit_capture: bool,
) -> Result<(), &'static str> {
    if !enabled {
        return Ok(());
    }
    if resident_backend != ResidentBackend::ReplacementV1 {
        return Err(
            "--compiled-composition-vertical-checkpoint requires --resident-backend replacement-v1",
        );
    }
    if !simd_reference {
        return Err(
            "--compiled-composition-vertical-checkpoint requires --require-simd-reference-byte-equal",
        );
    }
    if !reuse_input {
        return Err("--compiled-composition-vertical-checkpoint requires --reuse-input");
    }
    if fleet_pow {
        return Err("--compiled-composition-vertical-checkpoint cannot be combined with fleet PoW");
    }
    if graph_submit_diagnostic || graph_submit_capture {
        return Err(
            "--compiled-composition-vertical-checkpoint cannot be combined with graph-submit diagnostic or capture flags",
        );
    }
    Ok(())
}

fn resident_backend_gate(
    selected: ResidentBackend,
    architecture_required: bool,
    runtime_mode: RequiredCudaPcsRuntimeMode,
) -> Result<(), &'static str> {
    if selected == ResidentBackend::ReplacementV1
        && (!architecture_required || runtime_mode != RequiredCudaPcsRuntimeMode::ArenaGraph)
    {
        return Err(
            "replacement-v1 requires --require-gpu-native-architecture and --require-gpu-pcs-runtime-mode arena-graph",
        );
    }
    Ok(())
}

fn configure_resident_backend(
    config: &mut GpuProverConfig,
    selected: ResidentBackend,
    architecture_required: bool,
    runtime_mode: RequiredCudaPcsRuntimeMode,
) -> Result<(), &'static str> {
    resident_backend_gate(selected, architecture_required, runtime_mode)?;
    config.resident_backend = selected;
    config.strict = architecture_required && runtime_mode == RequiredCudaPcsRuntimeMode::ArenaGraph;
    Ok(())
}

/// Prover engine: `legacy` (`prove_cairo` — the parity oracle) or `gpu-native`
/// (`stwo-cairo-gpu-prover`, GPU_RESIDENT_PROVER_DESIGN.md §3).
fn engine() -> String {
    arg("--engine").unwrap_or_else(|| "legacy".to_string())
}

// The GPU-native engine is deliberately CUDA-specific. SIMD remains the legacy
// reference oracle instead of masquerading as another implementation of the new
// orchestration architecture. Resident CUDA graphs borrow thread-bound execution
// state, so each proving thread owns and reuses its own prover instead of moving one
// through a process-global mutex.
thread_local! {
    static GPU_NATIVE_CUDA: RefCell<Option<GpuCairoProver<Blake2sMerkleChannel>>> =
        const { RefCell::new(None) };
    static FLEET_POW_TRANSPORT: RefCell<Option<FleetPowUnixTransport>> =
        const { RefCell::new(None) };
    static FLEET_PROOF_GENERATION: Cell<u64> = const { Cell::new(1) };
}
static LAST_GPU_NATIVE_PCS_TELEMETRY: OnceLock<Mutex<Option<CudaPcsDriverTelemetry>>> =
    OnceLock::new();
static LAST_GPU_NATIVE_AOT_STATS: OnceLock<Mutex<Option<AotRuntimeStats>>> = OnceLock::new();
static LAST_GPU_NATIVE_SESSION_TELEMETRY: OnceLock<Mutex<Option<ResidentSessionTelemetry>>> =
    OnceLock::new();
static LAST_FLEET_PROOF_TELEMETRY: OnceLock<Mutex<Option<FleetResidentProofTelemetry>>> =
    OnceLock::new();

fn record_gpu_native_pcs_telemetry(telemetry: &CudaPcsDriverTelemetry) {
    *LAST_GPU_NATIVE_PCS_TELEMETRY
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("gpu-native telemetry mutex poisoned") = Some(telemetry.clone());
}

fn clear_gpu_native_pcs_telemetry() {
    *LAST_GPU_NATIVE_PCS_TELEMETRY
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("gpu-native telemetry mutex poisoned") = None;
}

fn record_gpu_native_aot_stats(stats: AotRuntimeStats) {
    *LAST_GPU_NATIVE_AOT_STATS
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("gpu-native AOT telemetry mutex poisoned") = Some(stats);
}

fn record_gpu_native_session_telemetry(telemetry: &ResidentSessionTelemetry) {
    *LAST_GPU_NATIVE_SESSION_TELEMETRY
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("gpu-native session telemetry mutex poisoned") = Some(telemetry.clone());
}

fn record_fleet_proof_telemetry(telemetry: &FleetResidentProofTelemetry) {
    *LAST_FLEET_PROOF_TELEMETRY
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("fleet proof telemetry mutex poisoned") = Some(telemetry.clone());
}

fn prove_gpu_native(input: ProverInput, params: ProverParameters) -> BenchProof {
    GPU_NATIVE_CUDA.with(|cell| {
        let mut slot = cell.borrow_mut();
        let prover = slot.get_or_insert_with(new_gpu_native_prover);
        let proof = if let Some(socket) = fleet_pow_socket() {
            assert!(
                prover.config().strict,
                "--fleet-pow-socket requires strict resident proving"
            );
            FLEET_POW_TRANSPORT.with(|transport_cell| {
                let mut transport_slot = transport_cell.borrow_mut();
                let transport = transport_slot.get_or_insert_with(|| {
                    FleetPowUnixTransport::connect(&socket)
                        .unwrap_or_else(|error| panic!("fleet PoW connect {socket}: {error}"))
                });
                FLEET_PROOF_GENERATION.with(|generation| {
                    let current = generation.get();
                    generation.set(
                        current
                            .checked_add(1)
                            .expect("fleet proof generation overflow"),
                    );
                    let outcome = prover
                        .prove_resident_blake2s_with_fleet_pow(
                            input,
                            params,
                            fleet_pow_schedule(),
                            current,
                            transport,
                        )
                        .expect("fleet resident prove failed");
                    let pcs = CudaPcsDriverTelemetry::completed_arena_graph(
                        outcome.telemetry.execution,
                        outcome.telemetry.expected_graph_launches,
                        outcome.telemetry.expected_captured_kernel_launches,
                    );
                    record_gpu_native_pcs_telemetry(&pcs);
                    record_fleet_proof_telemetry(&outcome.telemetry);
                    Ok(outcome.proof)
                })
            })
        } else if prover.config().strict {
            prover.prove_resident_blake2s(input, params)
        } else {
            prover.prove(input, params)
        }
        .expect("gpu-native prove failed");
        if fleet_pow_socket().is_none() {
            if prover.config().compiled_composition_vertical_checkpoint {
                assert!(
                    prover.last_pcs_telemetry().is_none(),
                    "eager vertical checkpoint must not claim ArenaGraph PCS telemetry"
                );
                clear_gpu_native_pcs_telemetry();
            } else {
                let telemetry = prover
                    .last_pcs_telemetry()
                    .expect("gpu-native prove returned without CUDA PCS architecture telemetry");
                assert!(
                    telemetry.is_complete(),
                    "gpu-native CUDA PCS driver did not complete every architecture stage"
                );
                record_gpu_native_pcs_telemetry(telemetry);
            }
        }
        if let Some(session) = prover.last_resident_session_telemetry() {
            record_gpu_native_session_telemetry(session);
        }
        let aot_stats = prover
            .last_aot_stats()
            .expect("gpu-native prove returned without CUDA AOT provenance telemetry");
        if gpu_native_architecture_required() {
            validate_strict_aot_provenance(Some(&aot_stats))
                .unwrap_or_else(|error| panic!("GPU-native architecture gate failed: {error}"));
        }
        record_gpu_native_aot_stats(aot_stats);
        proof
    })
}

fn fleet_pow_socket() -> Option<String> {
    arg("--fleet-pow-socket")
}

fn fleet_pow_schedule() -> FleetPowSchedule {
    const WORKERS_PER_RANK: u32 = 1024 * 256;
    FleetPowSchedule {
        interaction: FleetPowPlan {
            workers_per_rank: WORKERS_PER_RANK,
            indices_per_attempt: 1 << 22,
        },
        query: FleetPowPlan {
            workers_per_rank: WORKERS_PER_RANK,
            indices_per_attempt: 1 << 24,
        },
    }
}

fn gpu_native_prover_config() -> GpuProverConfig {
    let mut config = GpuProverConfig::default();
    let architecture_required = gpu_native_architecture_required();
    let runtime_mode = required_gpu_pcs_runtime_mode();
    let resident_backend = requested_resident_backend();
    configure_resident_backend(
        &mut config,
        resident_backend,
        architecture_required,
        runtime_mode,
    )
    .unwrap_or_else(|error| panic!("GPU resident backend gate failed: {error}"));
    configure_graph_submit_policy(
        &mut config,
        graph_submit_gap_diagnostic(),
        graph_submit_gap_capture(),
    );
    config.compiled_composition_vertical_checkpoint = compiled_composition_vertical_checkpoint();
    config.operational_safety_reserve_bytes = gpu_bench_physical::operational_safety_reserve_bytes(
        arg("--operational-safety-reserve-bytes"),
    );
    assert!(
        !config.allow_slow_graph_submit_diagnostic || config.strict,
        "slow graph-submit capture requires the strict ArenaGraph architecture gate"
    );
    config
}

fn new_gpu_native_prover() -> GpuCairoProver<Blake2sMerkleChannel> {
    let config = gpu_native_prover_config();
    if packed_numerator_measurement_control() {
        GpuCairoProver::new_packed_numerator_measurement_control(config)
            .expect("packed numerator measurement control config")
    } else {
        GpuCairoProver::new(config).expect("gpu-native config")
    }
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

fn requested_resident_backend() -> ResidentBackend {
    parse_resident_backend_args(std::env::args()).unwrap_or_else(|error| panic!("{error}"))
}

fn compiled_composition_vertical_checkpoint() -> bool {
    parse_compiled_composition_vertical_checkpoint_args(std::env::args())
        .unwrap_or_else(|error| panic!("{error}"))
}

fn packed_numerator_measurement_control() -> bool {
    parse_packed_numerator_measurement_control_args(std::env::args())
        .unwrap_or_else(|error| panic!("{error}"))
}

fn enforce_resident_backend_invocation() {
    resident_backend_gate(
        requested_resident_backend(),
        gpu_native_architecture_required(),
        required_gpu_pcs_runtime_mode(),
    )
    .unwrap_or_else(|error| panic!("GPU resident backend gate failed: {error}"));
}

fn enforce_packed_numerator_measurement_control_invocation(backend: &str) {
    packed_numerator_measurement_control_gate(
        packed_numerator_measurement_control(),
        backend,
        &engine(),
        requested_resident_backend(),
        gpu_native_architecture_required(),
        required_gpu_pcs_runtime_mode(),
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

fn gpu_native_architecture_required() -> bool {
    flag("--require-gpu-native-architecture")
        || std::env::var("STWO_BENCH_REQUIRE_GPU_NATIVE_ARCHITECTURE").as_deref() == Ok("1")
}

fn graph_submit_gap_diagnostic() -> bool {
    flag("--diagnostic-allow-slow-graph-submit")
}

fn graph_submit_gap_capture() -> bool {
    flag("--capture-slow-graph-submit")
}

fn configure_graph_submit_policy(config: &mut GpuProverConfig, diagnostic: bool, capture: bool) {
    assert!(
        !(diagnostic && capture),
        "graph-submit diagnostic and capture modes are mutually exclusive"
    );
    config.allow_slow_graph_submit_diagnostic = diagnostic || capture;
    config.record_graph_replay_intervals_diagnostic = diagnostic;
}

fn required_gpu_pcs_runtime_mode() -> RequiredCudaPcsRuntimeMode {
    let value = arg("--require-gpu-pcs-runtime-mode")
        .or_else(|| std::env::var("STWO_BENCH_REQUIRE_GPU_PCS_RUNTIME_MODE").ok())
        .unwrap_or_else(|| "arena-graph".to_string());
    RequiredCudaPcsRuntimeMode::parse(&value)
}

fn performance_claim_admissible() -> bool {
    performance_claim_admissible_for(
        &engine(),
        gpu_native_architecture_required(),
        required_gpu_pcs_runtime_mode(),
        graph_submit_gap_diagnostic(),
        compiled_composition_vertical_checkpoint(),
    ) && fleet_pow_socket().is_none()
}

fn performance_claim_admissible_for(
    selected_engine: &str,
    architecture_required: bool,
    mode: RequiredCudaPcsRuntimeMode,
    diagnostic: bool,
    compiled_composition_vertical_checkpoint: bool,
) -> bool {
    performance_measurement_available_for(selected_engine, architecture_required, mode)
        && !diagnostic
        && !compiled_composition_vertical_checkpoint
}

fn graph_capture_claim_admissible(
    base_admissible: bool,
    capture_enabled: bool,
    observed_gate: Option<bool>,
) -> bool {
    base_admissible && (!capture_enabled || observed_gate == Some(true))
}

fn performance_measurement_available() -> bool {
    performance_measurement_available_for(
        &engine(),
        gpu_native_architecture_required(),
        required_gpu_pcs_runtime_mode(),
    )
}

fn performance_measurement_available_for(
    selected_engine: &str,
    architecture_required: bool,
    mode: RequiredCudaPcsRuntimeMode,
) -> bool {
    selected_engine != "gpu-native"
        || (architecture_required && mode == RequiredCudaPcsRuntimeMode::ArenaGraph)
}

fn last_gpu_native_pcs_telemetry() -> Option<CudaPcsDriverTelemetry> {
    LAST_GPU_NATIVE_PCS_TELEMETRY
        .get()
        .and_then(|telemetry| telemetry.lock().ok())
        .and_then(|telemetry| telemetry.clone())
}

fn last_gpu_native_aot_stats() -> Option<AotRuntimeStats> {
    LAST_GPU_NATIVE_AOT_STATS
        .get()
        .and_then(|stats| stats.lock().ok())
        .and_then(|stats| *stats)
}

fn last_gpu_native_session_telemetry() -> Option<ResidentSessionTelemetry> {
    LAST_GPU_NATIVE_SESSION_TELEMETRY
        .get()
        .and_then(|telemetry| telemetry.lock().ok())
        .and_then(|telemetry| telemetry.clone())
}

fn last_fleet_proof_telemetry() -> Option<FleetResidentProofTelemetry> {
    LAST_FLEET_PROOF_TELEMETRY
        .get()
        .and_then(|telemetry| telemetry.lock().ok())
        .and_then(|telemetry| telemetry.clone())
}

fn validate_resident_session_architecture(
    required_mode: RequiredCudaPcsRuntimeMode,
    requested_backend: ResidentBackend,
    telemetry: Option<&ResidentSessionTelemetry>,
) -> Result<(), String> {
    if required_mode == RequiredCudaPcsRuntimeMode::DetachedEager {
        return Ok(());
    }
    let telemetry =
        telemetry.ok_or_else(|| "resident Graph-A setup telemetry is missing".to_string())?;
    let actual_backend = telemetry
        .protocol_policy
        .ok_or_else(|| "resident protocol policy telemetry is missing".to_string())?
        .resident_backend;
    if actual_backend != requested_backend {
        return Err(format!(
            "requested resident backend {} but prepared {}",
            requested_backend.cli_name(),
            actual_backend.cli_name()
        ));
    }
    telemetry
        .require_strict_graph_a()
        .map_err(|error| error.to_string())
}

fn validate_strict_aot_provenance(stats: Option<&AotRuntimeStats>) -> Result<(), String> {
    let stats = stats.ok_or_else(|| "CUDA AOT provenance telemetry is missing".to_string())?;
    let rejected = [
        ("aot_misses", stats.aot_misses),
        ("runtime_loads", stats.runtime_loads),
        ("runtime_cache_hits", stats.runtime_cache_hits),
        ("strict_rejections", stats.strict_rejections),
    ];
    let nonzero = rejected
        .into_iter()
        .filter(|(_, count)| *count != 0)
        .map(|(name, count)| format!("{name}={count}"))
        .collect::<Vec<_>>();
    if nonzero.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "strict CUDA AOT provenance requires no fallback or rejection, got {}",
            nonzero.join(" ")
        ))
    }
}

fn validate_gpu_native_architecture(
    backend: &str,
    selected_engine: &str,
    expected_mode: RequiredCudaPcsRuntimeMode,
    telemetry: Option<&CudaPcsDriverTelemetry>,
) -> Result<(), String> {
    if backend != "cuda" {
        return Err(format!("backend must be cuda, got {backend}"));
    }
    if selected_engine != "gpu-native" {
        return Err(format!("engine must be gpu-native, got {selected_engine}"));
    }
    let telemetry = telemetry.ok_or_else(|| "CUDA PCS telemetry is missing".to_string())?;
    if telemetry.architecture != REQUIRED_CUDA_PCS_ARCHITECTURE {
        return Err(format!(
            "CUDA PCS architecture must be {REQUIRED_CUDA_PCS_ARCHITECTURE}, got {}",
            telemetry.architecture
        ));
    }
    if telemetry.runtime_mode != expected_mode.telemetry_mode() {
        return Err(format!(
            "CUDA PCS runtime mode must be {:?}, got {:?}",
            expected_mode.telemetry_mode(),
            telemetry.runtime_mode
        ));
    }
    for stage in stwo::prover::pcs::proof_driver::PcsProofStage::ALL {
        let index = stage.index();
        let started = telemetry.stage_started[index];
        let finished = telemetry.stage_finished[index];
        if started != 1 || finished != 1 {
            return Err(format!(
                "CUDA PCS stage {stage:?} must start and finish exactly once, got started={started} finished={finished}"
            ));
        }
    }
    if !telemetry.batched_tree_decommit {
        return Err("CUDA PCS batched tree decommit was not used".to_string());
    }
    if !telemetry.is_complete() {
        return Err("CUDA PCS telemetry did not report complete".to_string());
    }
    Ok(())
}

fn enforce_gpu_native_architecture_invocation(backend: &str) {
    if !gpu_native_architecture_required() {
        return;
    }
    assert_eq!(
        backend, "cuda",
        "GPU-native architecture gate failed: backend must be cuda"
    );
    assert_eq!(
        engine(),
        "gpu-native",
        "GPU-native architecture gate failed: engine must be gpu-native"
    );
    // Parse this before expensive input loading so an invalid future-mode request
    // fails immediately. The concrete telemetry comparison happens after proving.
    let _ = required_gpu_pcs_runtime_mode();
    let manifest_hash = stwo_backend_cuda::aot::loaded_manifest_hash();
    assert_ne!(
        manifest_hash, 0,
        "GPU-native architecture gate failed: embedded CUDA AOT kernel pack is missing"
    );
    // GpuCairoProver commits the process-wide AOT-only mode only after its full
    // fallible admission succeeds. Missing entries still fail at first use and
    // the post-proof counters remain the independent provenance contract.
}

fn reject_gpu_native_architecture_gate_without_proof(mode: &str) {
    assert!(
        !gpu_native_architecture_required(),
        "GPU-native architecture gate failed: {mode} exits without a proof or CUDA PCS telemetry"
    );
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
        .unwrap_or_else(|error| {
            panic!(
                "Failed to load bootloader JSON at {}: {error}",
                bootloader_path.display()
            )
        });

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

fn throughput_mhz(work: Option<usize>, seconds: Option<f64>, applicable: bool) -> Option<f64> {
    if !applicable {
        return None;
    }
    Some(round3(work? as f64 / seconds? / 1e6))
}

/// Linearly interpolated quantile (the common R-7 definition). Keeping this local
/// avoids pulling a statistics crate into the benchmark binary.
fn quantile(samples: &[f64], q: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    assert!((0.0..=1.0).contains(&q), "quantile must be in [0, 1]");
    assert!(
        samples.iter().all(|sample| sample.is_finite()),
        "quantile samples must be finite"
    );

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = q * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * weight)
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
    let architecture_required = gpu_native_architecture_required();
    let required_mode = architecture_required.then(required_gpu_pcs_runtime_mode);
    let telemetry = last_gpu_native_pcs_telemetry();
    let aot_stats = last_gpu_native_aot_stats();
    let session_telemetry = last_gpu_native_session_telemetry();
    let fleet_telemetry = last_fleet_proof_telemetry();
    let vertical_checkpoint = compiled_composition_vertical_checkpoint();
    if let Some(required_mode) = required_mode {
        if vertical_checkpoint {
            assert!(
                telemetry.is_none(),
                "eager vertical checkpoint must not report ArenaGraph PCS telemetry"
            );
        } else {
            validate_gpu_native_architecture(backend, &engine(), required_mode, telemetry.as_ref())
                .unwrap_or_else(|error| panic!("GPU-native architecture gate failed: {error}"));
        }
        validate_strict_aot_provenance(aot_stats.as_ref())
            .unwrap_or_else(|error| panic!("GPU-native architecture gate failed: {error}"));
        validate_resident_session_architecture(
            required_mode,
            requested_resident_backend(),
            session_telemetry.as_ref(),
        )
        .unwrap_or_else(|error| panic!("GPU-native architecture gate failed: {error}"));
    }
    let base = json!({
        "security_bits": pcs.security_bits(),
        "n_queries": pcs.fri_config.n_queries,
        "pow_bits": pcs.pow_bits,
        "fold_step": pcs.fri_config.fold_step,
        "engine": engine(),
        "gpu": gpu_name(backend),
        "nproc": nproc(),
        "host_mem_gb": round3(host_mem_gb()),
        "gpu_resident_backend_requested": requested_resident_backend().cli_name(),
        "gpu_packed_numerator_measurement_control_requested":
            packed_numerator_measurement_control(),
        "gpu_native_architecture_required": architecture_required,
        "gpu_pcs_required_runtime_mode": required_mode.map(RequiredCudaPcsRuntimeMode::cli_name),
        "gpu_native_architecture_gate_passed": (architecture_required && !vertical_checkpoint)
            .then_some(true),
        "gpu_pcs_architecture_gate_applicable": architecture_required && !vertical_checkpoint,
        "benchmark_diagnostic_mode": graph_submit_gap_diagnostic() || vertical_checkpoint,
        "benchmark_diagnostic_reason": if vertical_checkpoint {
            Some("compiled-composition-vertical-checkpoint")
        } else {
            graph_submit_gap_diagnostic().then_some("graph-submit-gap-and-replay-intervals")
        },
        "compiled_composition_vertical_checkpoint": vertical_checkpoint,
        "compiled_composition_vertical_checkpoint_gate_passed": vertical_checkpoint.then_some(true),
        "compiled_composition_vertical_checkpoint_timing_scope": vertical_checkpoint
            .then_some("public-prove-call-warm-end-to-end"),
        "performance_claim_class": vertical_checkpoint
            .then_some("indicative-non-formal"),
        "benchmark_graph_submit_capture_mode": graph_submit_gap_capture(),
        "fleet_pow_enabled": fleet_pow_socket().is_some(),
        "fleet_pow_performance_admissible": fleet_telemetry
            .as_ref()
            .map(FleetResidentProofTelemetry::performance_claim_admissible),
        "fleet_pow_proof_generation": fleet_telemetry
            .as_ref()
            .map(|telemetry| telemetry.proof_generation),
        "fleet_pow_interaction_attempt": fleet_telemetry
            .as_ref()
            .map(|telemetry| telemetry.pow.interaction.attempt_ordinal),
        "fleet_pow_query_attempt": fleet_telemetry
            .as_ref()
            .map(|telemetry| telemetry.pow.query.attempt_ordinal),
        "performance_measurement_available": performance_measurement_available(),
        "performance_claim_admissible": performance_claim_admissible()
            && !graph_submit_gap_capture(),
    });
    merge_json(
        merge_json(
            merge_json(base, gpu_native_pcs_context(telemetry.as_ref())),
            gpu_native_aot_context(aot_stats.as_ref(), architecture_required),
        ),
        gpu_native_session_context(session_telemetry.as_ref(), engine() == "gpu-native"),
    )
}

/// Architecture evidence from the concrete CUDA PCS driver. These fields land
/// in the benchmark's primary JSON record so the harness can reject a run that
/// silently re-entered reference trait dispatch or skipped a protocol stage.
fn gpu_native_pcs_context(telemetry: Option<&CudaPcsDriverTelemetry>) -> serde_json::Value {
    let Some(telemetry) = telemetry.filter(|_| engine() == "gpu-native") else {
        return json!({
            "gpu_pcs_driver_architecture": null,
            "gpu_pcs_runtime_mode": null,
            "gpu_pcs_stage_started": null,
            "gpu_pcs_stage_finished": null,
            "gpu_pcs_batched_tree_decommit": null,
            "gpu_pcs_driver_complete": null,
            "gpu_host_syncs": null,
            "gpu_graph_launches": null,
            "gpu_kernel_launches": null,
            "gpu_expected_graph_launches": null,
            "gpu_expected_kernel_launches": null,
            "gpu_hot_h2d_bytes": null,
            "gpu_hot_d2h_bytes": null,
            "gpu_hot_allocations": null,
            "gpu_hot_allocation_bytes": null,
            "gpu_hot_frees": null,
            "gpu_hot_d2d_bytes": null,
            "gpu_hot_memset_bytes": null,
            "gpu_hot_fill_words": null,
            "gpu_hot_capture_begins": null,
            "gpu_hot_capture_finishes": null,
            "gpu_hot_capture_aborts": null,
            "gpu_hot_lane_forks": null,
            "gpu_hot_lane_joins": null,
            "gpu_graph_submit_gap_ns_total": null,
            "gpu_max_graph_submit_gap_ms": null,
            "gpu_graph_submit_gap_strict_gate_passed": null,
        });
    };
    pcs_telemetry_json(telemetry)
}

fn pcs_telemetry_json(telemetry: &CudaPcsDriverTelemetry) -> serde_json::Value {
    let stages_started = stwo::prover::pcs::proof_driver::PcsProofStage::ALL
        .into_iter()
        .map(|stage| {
            (
                format!("{stage:?}"),
                json!(telemetry.stage_started[stage.index()]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let stages_finished = stwo::prover::pcs::proof_driver::PcsProofStage::ALL
        .into_iter()
        .map(|stage| (format!("{stage:?}"), json!(telemetry.completed(stage))))
        .collect::<serde_json::Map<_, _>>();
    let exec = telemetry.exec;
    json!({
        "gpu_pcs_driver_architecture": telemetry.architecture,
        "gpu_pcs_runtime_mode": format!("{:?}", telemetry.runtime_mode),
        "gpu_pcs_stage_started": stages_started,
        "gpu_pcs_stage_finished": stages_finished,
        "gpu_pcs_batched_tree_decommit": telemetry.batched_tree_decommit,
        "gpu_pcs_driver_complete": telemetry.is_complete(),
        "gpu_host_syncs": exec.map(|value| value.sync_calls),
        "gpu_graph_launches": exec.map(|value| value.graph_launches),
        "gpu_kernel_launches": exec.map(|value| value.kernel_launches),
        "gpu_expected_graph_launches": telemetry.expected_graph_launches,
        "gpu_expected_kernel_launches": telemetry.expected_kernel_launches,
        "gpu_hot_h2d_bytes": exec.map(|value| value.h2d_bytes),
        "gpu_hot_d2h_bytes": exec.map(|value| value.d2h_bytes),
        "gpu_hot_allocations": exec.map(|value| value.allocations),
        "gpu_hot_allocation_bytes": exec.map(|value| value.allocation_bytes),
        "gpu_hot_frees": exec.map(|value| value.frees),
        "gpu_hot_d2d_bytes": exec.map(|value| value.d2d_bytes),
        "gpu_hot_memset_bytes": exec.map(|value| value.memset_bytes),
        "gpu_hot_fill_words": exec.map(|value| value.fill_words),
        "gpu_hot_capture_begins": exec.map(|value| value.capture_begins),
        "gpu_hot_capture_finishes": exec.map(|value| value.capture_finishes),
        "gpu_hot_capture_aborts": exec.map(|value| value.capture_aborts),
        "gpu_hot_lane_forks": exec.map(|value| value.lane_forks),
        "gpu_hot_lane_joins": exec.map(|value| value.lane_joins),
        "gpu_graph_submit_gap_ns_total": exec.map(|value| value.graph_submit_gap_ns_total),
        "gpu_max_graph_submit_gap_ms": exec.map(|value| {
            value.graph_submit_gap_ns_max as f64 / 1_000_000.0
        }),
        "gpu_graph_submit_gap_strict_gate_passed": exec.map(|value| {
            value.graph_submit_gap_ns_max < 50_000_000
        }),
    })
}

fn gpu_native_aot_context(
    stats: Option<&AotRuntimeStats>,
    architecture_required: bool,
) -> serde_json::Value {
    let Some(stats) = stats.filter(|_| engine() == "gpu-native") else {
        return json!({
            "gpu_aot_loads": null,
            "gpu_aot_cache_hits": null,
            "gpu_aot_manifest_hash": null,
            "gpu_aot_misses": null,
            "gpu_aot_runtime_loads": null,
            "gpu_aot_runtime_cache_hits": null,
            "gpu_aot_strict_rejections": null,
            "gpu_aot_provenance_gate_passed": null,
        });
    };
    json!({
        "gpu_aot_loads": stats.aot_loads,
        "gpu_aot_cache_hits": stats.aot_cache_hits,
        "gpu_aot_manifest_hash": stwo_backend_cuda::aot::loaded_manifest_hash(),
        "gpu_aot_misses": stats.aot_misses,
        "gpu_aot_runtime_loads": stats.runtime_loads,
        "gpu_aot_runtime_cache_hits": stats.runtime_cache_hits,
        "gpu_aot_strict_rejections": stats.strict_rejections,
        "gpu_aot_provenance_gate_passed": architecture_required.then_some(true),
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
    graph_submit_samples: Vec<Option<GraphSubmitSample>>,
    proof_loop_started_unix_ns: u64,
    proof_loop_finished_unix_ns: u64,
    proof_size: usize,
    gpu_proof_blake3: String,
    verify_ms: f64,
    verified_reps: usize,
    proof_byte_equal: Option<bool>,
    simd_reference_byte_equal: Option<bool>,
    simd_reference: Option<SimdReferenceRecord>,
    proof_mutation: Option<ProofMutationRecord>,
    vram_peak_gb: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphSubmitSample {
    total_ns: u64,
    max_ns: u64,
    graph_launches: u64,
}

fn last_graph_submit_sample() -> Option<GraphSubmitSample> {
    let exec = last_gpu_native_pcs_telemetry()?.exec?;
    Some(GraphSubmitSample {
        total_ns: exec.graph_submit_gap_ns_total,
        max_ns: exec.graph_submit_gap_ns_max,
        graph_launches: exec.graph_launches,
    })
}

fn claimed_graph_submit_gap_ns(samples: &[GraphSubmitSample]) -> Option<u64> {
    let warm_start = usize::from(samples.len() > 1);
    samples[warm_start..]
        .iter()
        .map(|sample| sample.max_ns)
        .max()
}

fn graph_submit_gap_average_ns(sample: GraphSubmitSample) -> Option<f64> {
    let gap_count = sample.graph_launches.checked_sub(1)?;
    (gap_count != 0).then(|| sample.total_ns as f64 / gap_count as f64)
}

fn graph_submit_distribution_context(outcome: &RepOutcome) -> serde_json::Value {
    if outcome.graph_submit_samples.len() != outcome.times.len()
        || outcome.graph_submit_samples.iter().any(Option::is_none)
    {
        return json!({});
    }
    let samples = outcome
        .graph_submit_samples
        .iter()
        .map(|sample| sample.expect("graph-submit samples were checked above"))
        .collect::<Vec<_>>();
    let max_ns = samples
        .iter()
        .map(|sample| sample.max_ns)
        .collect::<Vec<_>>();
    let total_ns = samples
        .iter()
        .map(|sample| sample.total_ns)
        .collect::<Vec<_>>();
    let launches = samples
        .iter()
        .map(|sample| sample.graph_launches)
        .collect::<Vec<_>>();
    let averages_ns = samples
        .iter()
        .map(|sample| graph_submit_gap_average_ns(*sample))
        .collect::<Vec<_>>();
    let warm_start = usize::from(samples.len() > 1);
    let claimed = &samples[warm_start..];
    let claimed_max_ns = claimed_graph_submit_gap_ns(&samples).unwrap_or(0);
    let strict_gate_passed = claimed_max_ns < 50_000_000;
    let claim_admissible = graph_capture_claim_admissible(
        performance_claim_admissible(),
        graph_submit_gap_capture(),
        Some(strict_gate_passed),
    );
    json!({
        "performance_claim_admissible": claim_admissible,
        "gpu_graph_submit_gap_ns_max_samples": max_ns,
        "gpu_graph_submit_gap_ns_total_samples": total_ns,
        "gpu_graph_submit_gap_ns_average_samples": averages_ns,
        "gpu_graph_submit_launches_samples": launches,
        "gpu_graph_submit_warm_sample_count": claimed.len(),
        "gpu_max_graph_submit_gap_ms": claimed_max_ns as f64 / 1_000_000.0,
        "gpu_graph_submit_gap_strict_gate_passed": strict_gate_passed,
    })
}

fn unix_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos()
        .try_into()
        .expect("Unix nanosecond timestamp exceeds u64")
}

#[derive(Clone, Debug)]
struct SimdReferenceRecord {
    blake3: String,
    fresh: bool,
    elapsed_s: f64,
}

struct SimdReference {
    bytes: Vec<u8>,
    record: SimdReferenceRecord,
}

struct ProofValidation {
    proof_size: usize,
    gpu_proof_blake3: String,
    verify_ms: f64,
    verified_reps: usize,
    proof_byte_equal: Option<bool>,
    simd_reference_byte_equal: Option<bool>,
    simd_reference: Option<SimdReferenceRecord>,
    proof_mutation: Option<ProofMutationRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProofMutationRecord {
    kind: &'static str,
    rejected: bool,
    error_class: Option<&'static str>,
}

fn initial_proof_byte_equal(compare_to_rep0: bool, proof_count: usize) -> Option<bool> {
    (compare_to_rep0 && proof_count >= 2).then_some(true)
}

fn proof_byte_equal_gate_passes(required: bool, proof_byte_equal: Option<bool>) -> bool {
    !required || proof_byte_equal == Some(true)
}

fn simd_reference_required() -> bool {
    flag("--require-simd-reference-byte-equal")
        || std::env::var("STWO_BENCH_REQUIRE_SIMD_REFERENCE_BYTE_EQUAL").as_deref() == Ok("1")
}

fn simd_reference_gate_passes(required: bool, byte_equal: Option<bool>) -> bool {
    !required || byte_equal == Some(true)
}

fn simd_reference_reuse_input_gate_passes(required: bool, reuse_input: bool) -> bool {
    !required || reuse_input
}

fn proof_mutation_required() -> bool {
    flag("--require-proof-mutation-rejected")
        || std::env::var("STWO_BENCH_REQUIRE_PROOF_MUTATION_REJECTED").as_deref() == Ok("1")
}

fn proof_mutation_gate_passes(required: bool, rejected: Option<bool>) -> bool {
    !required || rejected == Some(true)
}

fn cairo_verification_error_class(error: &CairoVerificationError) -> &'static str {
    match error {
        CairoVerificationError::InvalidLogupSum => "invalid_logup_sum",
        CairoVerificationError::Stark(_) => "stark",
        CairoVerificationError::ProofOfWork => "proof_of_work",
    }
}

fn mutate_claimed_sum(claimed_sum: &mut SecureField) {
    *claimed_sum += SecureField::from(1_u32);
}

/// Mutate a verifier-semantic field of an already verified proof. The selected
/// interaction claim is mandatory for valid Cairo proofs and contributes linearly
/// to the verifier's lookup sum, so adding one deterministically violates LogUp.
fn verify_structured_proof_mutation(mut proof: BenchVerifierProof) -> ProofMutationRecord {
    let memory_id_claim = proof
        .interaction_claim
        .memory_id_to_big
        .as_mut()
        .expect("verified Cairo proof must contain memory_id_to_big interaction claim");
    mutate_claimed_sum(&mut memory_id_claim.claimed_sum);

    match verify_cairo::<Blake2sMerkleChannel>(proof) {
        Ok(()) => ProofMutationRecord {
            kind: PROOF_MUTATION_KIND,
            rejected: false,
            error_class: None,
        },
        Err(error) => ProofMutationRecord {
            kind: PROOF_MUTATION_KIND,
            rejected: true,
            error_class: Some(cairo_verification_error_class(&error)),
        },
    }
}

/// Compute and verify one fresh SIMD reference after the timed CUDA repetitions.
fn compute_simd_reference(input: ProverInput, params: ProverParameters) -> SimdReference {
    let started = Instant::now();
    let proof = prove_cairo::<SimdBackend, Blake2sMerkleChannel>(input, params)
        .expect("fresh SIMD reference proof failed");
    let bytes = bincode::serialize(&proof).expect("serialize fresh SIMD reference proof");
    verify_cairo::<Blake2sMerkleChannel>(proof.into())
        .expect("fresh SIMD reference proof failed verification");
    let blake3 = blake3::hash(&bytes).to_hex().to_string();
    SimdReference {
        bytes,
        record: SimdReferenceRecord {
            blake3,
            fresh: true,
            elapsed_s: started.elapsed().as_secs_f64(),
        },
    }
}

/// Serialize and verify every proof after the timed proving window. Exact byte
/// comparison avoids treating a non-cryptographic hash collision as determinism.
/// `dump_rep0` keeps STWO_DUMP_PROOF scoped to the standard and pipeline modes,
/// matching its historical behavior. `compare_to_rep0` is false when reps are
/// intentionally different statements (pipeline rotate mode).
fn validate_gpu_proofs(
    proofs: Vec<BenchProof>,
    dump_rep0: bool,
    compare_to_rep0: bool,
) -> (ProofValidation, Vec<Vec<u8>>) {
    assert!(!proofs.is_empty(), "at least one proof is required");

    let verified_reps = proofs.len();
    let mut serialized_proofs = Vec::with_capacity(verified_reps);
    let mut proof_size = 0;
    let mut gpu_proof_blake3 = None;
    let mut verify_ms = 0.0;
    let mut proof_byte_equal = initial_proof_byte_equal(compare_to_rep0, verified_reps);
    let mut proof_mutation = None;
    for (rep, proof) in proofs.into_iter().enumerate() {
        let bytes = bincode::serialize(&proof).expect("serialize proof");
        if rep == 0 {
            proof_size = bytes.len();
            gpu_proof_blake3 = Some(blake3::hash(&bytes).to_hex().to_string());
            if dump_rep0 {
                if let Ok(path) = std::env::var("STWO_DUMP_PROOF") {
                    std::fs::write(&path, &bytes).expect("write proof dump");
                    eprintln!("proof dumped: {} bytes -> {path}", bytes.len());
                }
            }
        } else if let Some(equal) = &mut proof_byte_equal {
            *equal &= serialized_proofs.first() == Some(&bytes);
        }

        let verify_start = Instant::now();
        let verifier_proof: BenchVerifierProof = proof.into();
        if rep == 0 && proof_mutation_required() {
            // Clone only the verifier representation (not canonical-proof aux
            // data), then leave it untouched until the original has verified.
            // This work is outside the timed proving window and occurs only when
            // explicitly requested.
            let mutation_candidate = verifier_proof.clone();
            verify_cairo::<Blake2sMerkleChannel>(verifier_proof).unwrap_or_else(|error| {
                panic!("proof repetition {rep} failed verification: {error:?}")
            });
            verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            proof_mutation = Some(verify_structured_proof_mutation(mutation_candidate));
        } else {
            verify_cairo::<Blake2sMerkleChannel>(verifier_proof).unwrap_or_else(|error| {
                panic!("proof repetition {rep} failed verification: {error:?}")
            });
            if rep == 0 {
                verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            }
        }
        serialized_proofs.push(bytes);
    }

    (
        ProofValidation {
            proof_size,
            gpu_proof_blake3: gpu_proof_blake3.expect("repetition 0 proof digest must exist"),
            verify_ms,
            verified_reps,
            proof_byte_equal,
            simd_reference_byte_equal: None,
            simd_reference: None,
            proof_mutation,
        },
        serialized_proofs,
    )
}

fn apply_simd_reference(
    validation: &mut ProofValidation,
    serialized_gpu_proofs: &[Vec<u8>],
    simd_reference: Option<&SimdReference>,
) {
    validation.simd_reference_byte_equal = simd_reference.map(|reference| {
        serialized_gpu_proofs
            .iter()
            .all(|proof| proof.as_slice() == reference.bytes.as_slice())
    });
    validation.simd_reference = simd_reference.map(|reference| reference.record.clone());
}

fn validate_proofs(
    proofs: Vec<BenchProof>,
    dump_rep0: bool,
    compare_to_rep0: bool,
    simd_reference: Option<&SimdReference>,
) -> ProofValidation {
    let (mut validation, serialized_gpu_proofs) =
        validate_gpu_proofs(proofs, dump_rep0, compare_to_rep0);
    apply_simd_reference(&mut validation, &serialized_gpu_proofs, simd_reference);
    validation
}

fn proof_byte_equal_required() -> bool {
    flag("--require-proof-byte-equal")
        || std::env::var("STWO_BENCH_REQUIRE_PROOF_BYTE_EQUAL").as_deref() == Ok("1")
}

fn enforce_proof_byte_equal(proof_byte_equal: Option<bool>) {
    assert!(
        proof_byte_equal_gate_passes(proof_byte_equal_required(), proof_byte_equal),
        "proof byte equality gate failed: comparison was unavailable or repetitions did not match repetition 0"
    );
}

fn enforce_simd_reference_byte_equal(byte_equal: Option<bool>) {
    assert!(
        simd_reference_gate_passes(simd_reference_required(), byte_equal),
        "SIMD reference byte equality gate failed: a fresh verified SIMD reference was unavailable or GPU proof bytes differed"
    );
}

fn enforce_proof_mutation_rejected(proof_mutation: Option<&ProofMutationRecord>) {
    assert!(
        proof_mutation_gate_passes(
            proof_mutation_required(),
            proof_mutation.map(|record| record.rejected),
        ),
        "proof mutation rejection gate failed: the structured verifier-relevant mutation was unavailable or was accepted"
    );
}

fn reject_proof_mutation_gate_without_proof(mode: &str) {
    assert!(
        !proof_mutation_required(),
        "--require-proof-mutation-rejected requires a proof run; unavailable in {mode}"
    );
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
                prove_gpu_native($input, prover_params($variant))
            }
            ("simd", "gpu-native") => panic!(
                "gpu-native is a concrete CUDA proof runtime; use --engine legacy --backend simd for the reference oracle"
            ),
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
    throughput_distribution_applicable: bool,
    vm_s: f64,
    adapt_s: f64,
) {
    let cold = outcome.times[0];
    let warm_samples = &outcome.times[1..];
    let warm_best = quantile(warm_samples, 0.0);
    let warm_median = quantile(warm_samples, 0.5);
    let warm_p95 = quantile(warm_samples, 0.95);
    // Compatibility: prove_s_warm has always meant the best post-cold sample,
    // falling back to cold when --reps=1.
    let warm = warm_best.unwrap_or(cold);
    // DetachedEager remains useful as a proof-byte oracle, but it is the
    // CPU-owned orchestration path. Never let its timing become a GPU-resident
    // performance claim again.
    let performance_measurement_available = performance_measurement_available();
    let performance_claim_admissible = performance_claim_admissible();
    let throughput_distribution_applicable =
        throughput_distribution_applicable && performance_measurement_available;
    let warm_samples_rounded: Vec<_> = warm_samples.iter().copied().map(round3).collect();
    let (free, total) = stwo_backend_cuda::gpu_memory_info();
    let vram_end_gb = if total > 0 {
        (total - free) as f64 / 1e9
    } else {
        0.0
    };
    let simd_reference = outcome.simd_reference.as_ref();
    let proof_mutation = outcome.proof_mutation.as_ref();

    let record = json!({
        "program": program,
        "backend": backend,
        "n": n,
        "cycle_count": cycle_count,
        "pie_n_steps": pie_n_steps,
        "bootloader_overhead_pct": pie_n_steps.map(|s| {
            round3((cycle_count as f64 - s as f64) / s as f64 * 100.0)
        }),
        "reps": outcome.times.len(),
        "warm_sample_count": warm_samples.len(),
        "prove_s_warm_samples_raw": warm_samples,
        "prove_s_warm_samples_rounded": warm_samples_rounded,
        "prove_s_cold": round3(cold),
        "prove_s_warm": round3(warm),
        "prove_s_warm_semantics": "legacy_best_post_cold_or_cold_when_no_warm_samples",
        "prove_s_warm_best": warm_best.map(round3),
        "prove_s_warm_median": warm_median.map(round3),
        "prove_s_warm_p95": warm_p95.map(round3),
        "verify_ms": round3(outcome.verify_ms),
        "verified_reps": outcome.verified_reps,
        "proof_comparison_applicable": outcome.proof_byte_equal.is_some(),
        "deterministic": outcome.proof_byte_equal,
        "proof_byte_equal": outcome.proof_byte_equal,
        "proof_byte_equal_required": proof_byte_equal_required(),
        "gpu_proof_blake3": &outcome.gpu_proof_blake3,
        "proof_mutation_required": proof_mutation_required(),
        "proof_mutation_kind": proof_mutation.map(|record| record.kind),
        "proof_mutation_rejected": proof_mutation.map(|record| record.rejected),
        "proof_mutation_error_class": proof_mutation.and_then(|record| record.error_class),
        "proof_kb": round3(outcome.proof_size as f64 / 1024.0),
        "peak_rss_gb": round3(peak_rss_gb()),
        "vram_end_gb": round3(vram_end_gb),
        "vram_peak_gb": round3(outcome.vram_peak_gb),
        // Driver-maintained pool high-water marks (exact; the 25ms sampler above
        // measured up to 11GB low on SN_PIE_2). The VRAM-diet metric of record.
        "pool_used_high_gb": round3(stwo_backend_cuda::gpu_pool_highwater().0 as f64 / 1e9),
        "pool_reserved_high_gb": round3(stwo_backend_cuda::gpu_pool_highwater().1 as f64 / 1e9),
        "performance_claim_admissible": performance_claim_admissible,
        "performance_measurement_available": performance_measurement_available,
        "steps_per_s": performance_measurement_available.then(|| (cycle_count as f64 / warm).round()),
        "mhz": performance_measurement_available.then(|| round3(cycle_count as f64 / warm / 1e6)),
        "useful_mhz": performance_measurement_available
            .then(|| pie_n_steps.map(|s| round3(s as f64 / warm / 1e6)))
            .flatten(),
        "throughput_distribution_applicable": throughput_distribution_applicable,
        "mhz_median": throughput_mhz(
            Some(cycle_count), warm_median, throughput_distribution_applicable),
        "useful_mhz_median": throughput_mhz(
            pie_n_steps, warm_median, throughput_distribution_applicable),
        "mhz_at_warm_p95": throughput_mhz(
            Some(cycle_count), warm_p95, throughput_distribution_applicable),
        "useful_mhz_at_warm_p95": throughput_mhz(
            pie_n_steps, warm_p95, throughput_distribution_applicable),
        "vm_s": round3(vm_s),
        "adapt_s": round3(adapt_s),
    });
    let record = merge_json(
        record,
        json!({
            "gpu_proof_loop_started_unix_ns": outcome.proof_loop_started_unix_ns,
            "gpu_proof_loop_finished_unix_ns": outcome.proof_loop_finished_unix_ns,
            "simd_reference_required": simd_reference_required(),
            "simd_reference_comparison_applicable": outcome.simd_reference_byte_equal.is_some(),
            "simd_reference_byte_equal": outcome.simd_reference_byte_equal,
            "simd_reference_blake3": simd_reference.map(|reference| &reference.blake3),
            "simd_reference_fresh": simd_reference.map(|reference| reference.fresh),
            "simd_reference_s": simd_reference.map(|reference| round3(reference.elapsed_s)),
        }),
    );
    let record = merge_json(record, record_context(backend));
    println!(
        "{}",
        merge_json(record, graph_submit_distribution_context(outcome))
    );
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
/// starts before the producers spawn, so pipeline fill counts. Proof serialization,
/// comparison, and verification run after the clock stops to keep the sustained
/// window pure.
/// M6-a resident two-proof throughput harness. Increment 1: prove N inputs
/// SEQUENTIALLY (pre-loaded, so feed_starved=0) and report the two-proof-wall
/// metrics the M6 gates are defined on. This is the baseline the stream-explicit
/// concurrent scheduler must beat; the metric plumbing here is reused by that
/// concurrent version (which will overlap the proves on distinct streams).
fn run_resident_pipeline(source: &InputSource, backend: &str, n: usize) {
    assert!(
        !graph_submit_gap_capture(),
        "--capture-slow-graph-submit is supported only by the standard serial benchmark"
    );
    assert!(n >= 1, "--resident-pipeline <N> must be >= 1");
    let variant = source.preprocessed_variant();
    prewarm_pedersen_tables(variant);

    // Pre-load once, clone per proof (feed is off the timed critical path).
    let loaded = source.load();
    let pie_n_steps = loaded.pie_n_steps;
    let cycle_count = cycle_count_of(&loaded.input);
    let inputs: Vec<ProverInput> = (0..n).map(|_| loaded.input.clone()).collect();

    let wall_start = Instant::now();
    let mut times = Vec::with_capacity(n);
    let mut vram_peak_gb = 0.0f64;
    let mut proofs = Vec::with_capacity(n);
    for (i, input) in inputs.into_iter().enumerate() {
        let (proof, elapsed, rep_vram) = prove_sampled!(backend, input, variant);
        vram_peak_gb = vram_peak_gb.max(rep_vram);
        times.push(elapsed);
        proofs.push(proof);
        eprintln!("resident_proof={i} prove_s={elapsed:.3}");
    }
    let wall_s = wall_start.elapsed().as_secs_f64();

    // All N proofs prove the SAME statement. Validate outside the timed window.
    let validation = validate_proofs(proofs, false, true, None);

    let per_proof_s = times.iter().sum::<f64>() / n as f64;
    let total_steps = pie_n_steps.map(|s| s * n);
    let performance_claim_admissible = performance_claim_admissible();
    let proof_mutation = validation.proof_mutation.as_ref();
    println!(
        "{}",
        merge_json(
            json!({
                "mode": "resident-pipeline",
                "concurrency": "sequential-baseline",
                "n_proofs": n,
                "twoproof_wall_s": round3(wall_s),
                "per_proof_s": round3(per_proof_s),
                "performance_claim_admissible": performance_claim_admissible,
                "sustained_steps_per_s": performance_claim_admissible
                    .then(|| total_steps.map(|s| (s as f64 / wall_s).round()))
                    .flatten(),
                "sustained_useful_mhz": performance_claim_admissible
                    .then(|| total_steps.map(|s| round3(s as f64 / wall_s / 1e6)))
                    .flatten(),
                "cycle_useful_mhz": performance_claim_admissible
                    .then(|| round3((cycle_count * n) as f64 / wall_s / 1e6)),
                "vram_peak_gb": round3(vram_peak_gb),
                "feed_starved_s": 0.0,
                "verified_reps": validation.verified_reps,
                "proof_comparison_applicable": validation.proof_byte_equal.is_some(),
                "deterministic": validation.proof_byte_equal,
                "proof_byte_equal": validation.proof_byte_equal,
                "proof_byte_equal_required": proof_byte_equal_required(),
                "proof_mutation_required": proof_mutation_required(),
                "proof_mutation_kind": proof_mutation.map(|record| record.kind),
                "proof_mutation_rejected": proof_mutation.map(|record| record.rejected),
                "proof_mutation_error_class": proof_mutation.and_then(|record| record.error_class),
                "pie_n_steps": pie_n_steps,
            }),
            record_context(backend)
        )
    );
    enforce_proof_byte_equal(validation.proof_byte_equal);
    enforce_proof_mutation_rejected(validation.proof_mutation.as_ref());
}

/// M6-a increment 2: TRUE two-proof concurrency — `N` host threads, each driving a
/// full independent proof through its OWN `GpuCairoProver` instance, so the
/// process-global singleton `Mutex` that `prove_sampled!` locks (which would
/// serialize the threads on the host) is bypassed. One VRAM sampler spans the whole
/// concurrent window, so `vram_peak_gb` is the SIMULTANEOUS resident peak (two
/// dieted SN2 proofs ~= 63GB on an 80GB card). The M5c diet stays on.
///
/// Correctness: the two proofs allocate DISJOINT pool buffers, and today every
/// kernel + pool op is ordered on the legacy default stream 0 (backend recon), so
/// their GPU work is totally ordered on one stream => byte-identical proofs (gated
/// by `proof_byte_equal`), just GPU-serialized. What overlaps is the HOST compute
/// (the ~96%-idle-GPU wall is host-bound: partial_ec_mul / blake witness gen on the
/// CPU). This measures that host overlap — the cheap lever before the
/// per-thread-default-stream build, which would additionally isolate each proof
/// onto its own stream and lift the stream-0 D2H-drain coupling.
///
/// The per-thread `set_var` races that `GpuCairoProver::{new,prove}` would trigger
/// are all pre-empted on the main thread before any spawn: the diet vars are set
/// here, and a fully admitted throwaway prover performs CUDA/AOT initialization.
/// It also installs migration defaults for the legacy resident generation;
/// replacement-v1 carries those choices in its sealed execution config.
fn run_resident_concurrent(source: &InputSource, backend: &str, n: usize) {
    assert!(
        !graph_submit_gap_capture(),
        "--capture-slow-graph-submit is supported only by the standard serial benchmark"
    );
    assert!(n >= 1, "--resident-concurrent <N> must be >= 1");
    assert_eq!(backend, "cuda", "resident-concurrent is cuda-only");
    assert_eq!(
        engine().as_str(),
        "gpu-native",
        "resident-concurrent requires --engine gpu-native (the M5c diet path)"
    );
    let variant = source.preprocessed_variant();
    prewarm_pedersen_tables(variant);

    // Pre-set the diet-implied env on the MAIN thread (single-threaded here), so
    // GpuCairoProver::prove's internal set_var is a guarded no-op and no write races
    // a sibling thread's getenv. Mirrors main()'s STREAM_LDE fan-out but also covers
    // the STREAM_LEAF_COMMIT diet used by the sequential baseline.
    let diet_on = std::env::var("STWO_CUDA_STREAM_LEAF_COMMIT").as_deref() == Ok("1")
        || std::env::var("STWO_CAIRO_STREAM_LDE").as_deref() == Ok("1");
    if diet_on {
        // SAFETY: single-threaded (before any spawn); no sibling getenv yet.
        unsafe {
            std::env::set_var("STWO_STORE_COEFFS", "1");
            std::env::set_var("STWO_FORCE_EXTEND_EVAL_MODE", "1");
        }
    }
    // Warm-up on the main thread. Legacy admission installs any unset migration
    // defaults; replacement admission is env-write-free. Both complete one-time
    // CUDA/AOT setup before per-thread construction.
    drop(new_gpu_native_prover());

    let loaded = source.load();
    let pie_n_steps = loaded.pie_n_steps;
    let cycle_count = cycle_count_of(&loaded.input);
    let inputs: Vec<ProverInput> = (0..n).map(|_| loaded.input.clone()).collect();

    let sampler = VramSampler::start();
    let wall_start = Instant::now();

    // Each thread builds its own prover (prove() takes &mut self) and proves one
    // proof; returns (prove_s, proof). thread::scope joins all before returning.
    let results: Vec<(f64, BenchProof)> = std::thread::scope(|scope| {
        let handles: Vec<_> = inputs
            .into_iter()
            .enumerate()
            .map(|(i, input)| {
                scope.spawn(move || {
                    let mut prover = new_gpu_native_prover();
                    let t = Instant::now();
                    let params = prover_params(variant);
                    let proof = if prover.config().strict {
                        prover.prove_resident_blake2s(input, params)
                    } else {
                        prover.prove(input, params)
                    }
                    .expect("concurrent prove failed");
                    let telemetry = prover
                        .last_pcs_telemetry()
                        .expect("concurrent prove returned without CUDA PCS telemetry");
                    assert!(telemetry.is_complete());
                    record_gpu_native_pcs_telemetry(telemetry);
                    let aot_stats = prover
                        .last_aot_stats()
                        .expect("concurrent prove returned without CUDA AOT telemetry");
                    if gpu_native_architecture_required() {
                        validate_strict_aot_provenance(Some(&aot_stats)).unwrap_or_else(|error| {
                            panic!("GPU-native architecture gate failed: {error}")
                        });
                    }
                    record_gpu_native_aot_stats(aot_stats);
                    let elapsed = t.elapsed().as_secs_f64();
                    eprintln!("concurrent_proof={i} prove_s={elapsed:.3}");
                    (elapsed, proof)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("prover thread panicked"))
            .collect()
    });

    let wall_s = wall_start.elapsed().as_secs_f64();
    let vram_peak_gb = sampler.stop();

    let per_proof_s = results.iter().map(|(t, _)| *t).sum::<f64>() / n as f64;
    // serial_wall = what the same N proofs cost back-to-back; overlap_speedup =
    // how much the concurrency compressed that (1.0 = fully serialized, N = perfect).
    let serial_wall: f64 = results.iter().map(|(t, _)| *t).sum();
    // Same statement proved N times: verify all and exact-compare their bytes after
    // the timed concurrent window.
    let validation = validate_proofs(
        results.into_iter().map(|(_, proof)| proof).collect(),
        false,
        true,
        None,
    );
    let total_steps = pie_n_steps.map(|s| s * n);
    let performance_claim_admissible = performance_claim_admissible();
    let proof_mutation = validation.proof_mutation.as_ref();
    println!(
        "{}",
        merge_json(
            json!({
                "mode": "resident-concurrent",
                "concurrency": "threaded-stream0",
                "n_proofs": n,
                "twoproof_wall_s": round3(wall_s),
                "per_proof_s": round3(per_proof_s),
                "serial_wall_s": round3(serial_wall),
                "overlap_speedup": round3(serial_wall / wall_s),
                "performance_claim_admissible": performance_claim_admissible,
                "sustained_steps_per_s": performance_claim_admissible
                    .then(|| total_steps.map(|s| (s as f64 / wall_s).round()))
                    .flatten(),
                "sustained_useful_mhz": performance_claim_admissible
                    .then(|| total_steps.map(|s| round3(s as f64 / wall_s / 1e6)))
                    .flatten(),
                "cycle_useful_mhz": performance_claim_admissible
                    .then(|| round3((cycle_count * n) as f64 / wall_s / 1e6)),
                "vram_peak_gb": round3(vram_peak_gb),
                "feed_starved_s": 0.0,
                "verified_reps": validation.verified_reps,
                "proof_comparison_applicable": validation.proof_byte_equal.is_some(),
                "deterministic": validation.proof_byte_equal,
                "proof_byte_equal": validation.proof_byte_equal,
                "proof_byte_equal_required": proof_byte_equal_required(),
                "proof_mutation_required": proof_mutation_required(),
                "proof_mutation_kind": proof_mutation.map(|record| record.kind),
                "proof_mutation_rejected": proof_mutation.map(|record| record.rejected),
                "proof_mutation_error_class": proof_mutation.and_then(|record| record.error_class),
                "pie_n_steps": pie_n_steps,
            }),
            record_context(backend)
        )
    );
    enforce_proof_byte_equal(validation.proof_byte_equal);
    enforce_proof_mutation_rejected(validation.proof_mutation.as_ref());
}

fn run_pipelined(
    source: &InputSource,
    backend: &str,
    reps: usize,
    depth: usize,
    producers: usize,
    pie_mode: PieMode,
) {
    assert!(
        !graph_submit_gap_capture(),
        "--capture-slow-graph-submit is supported only by the standard serial benchmark"
    );
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
    let mut proofs = Vec::with_capacity(reps);
    let proof_loop_started_unix_ns = unix_time_ns();
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
        proofs.push(proof);
        eprintln!("rep={rep} prove_s={elapsed:.3}");
        emit_phase_totals(rep);
    }
    let proof_loop_finished_unix_ns = unix_time_ns();
    let total_s = total_start.elapsed().as_secs_f64();
    for handle in producer_handles {
        handle.join().expect("producer thread panicked");
    }

    // Validate every proof outside the sustained-throughput window. Rep 0 is still
    // the proof written by STWO_DUMP_PROOF.
    let validation = validate_proofs(proofs, true, pie_mode == PieMode::Aggregate, None);

    let outcome = RepOutcome {
        times,
        graph_submit_samples: Vec::new(),
        proof_loop_started_unix_ns,
        proof_loop_finished_unix_ns,
        proof_size: validation.proof_size,
        gpu_proof_blake3: validation.gpu_proof_blake3,
        verify_ms: validation.verify_ms,
        verified_reps: validation.verified_reps,
        proof_byte_equal: validation.proof_byte_equal,
        simd_reference_byte_equal: validation.simd_reference_byte_equal,
        simd_reference: validation.simd_reference,
        proof_mutation: validation.proof_mutation,
        vram_peak_gb,
    };
    print_main_record(
        &program,
        backend,
        n,
        rep0_cycle_count,
        rep0_pie_n_steps,
        &outcome,
        pie_mode == PieMode::Aggregate,
        last_vm_s,
        last_adapt_s,
    );
    let performance_claim_admissible = performance_claim_admissible();
    println!(
        "{}",
        json!({
            "pipeline": depth,
            "producers": producers,
            "pie_mode": match pie_mode { PieMode::Aggregate => "aggregate", PieMode::Rotate => "rotate" },
            "reps": reps,
            "total_s": round3(total_s),
            "feed_starved_s": round3(feed_starved_s),
            "performance_claim_admissible": performance_claim_admissible,
            "sustained_steps_per_s": performance_claim_admissible
                .then(|| (total_cycles as f64 / total_s).round()),
            "sustained_mhz": performance_claim_admissible
                .then(|| round3(total_cycles as f64 / total_s / 1e6)),
            "sustained_useful_mhz": (performance_claim_admissible && have_pie_steps)
                .then(|| round3(total_pie_n_steps as f64 / total_s / 1e6)),
        })
    );
    enforce_proof_byte_equal(outcome.proof_byte_equal);
    enforce_proof_mutation_rejected(outcome.proof_mutation.as_ref());
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
    enforce_packed_numerator_measurement_control_invocation(&backend);
    enforce_resident_backend_invocation();
    enforce_gpu_native_architecture_invocation(&backend);
    let reuse_input = flag("--reuse-input");
    compiled_composition_vertical_checkpoint_gate(
        compiled_composition_vertical_checkpoint(),
        requested_resident_backend(),
        simd_reference_required(),
        reuse_input,
        fleet_pow_socket().is_some(),
        graph_submit_gap_diagnostic(),
        graph_submit_gap_capture(),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    if simd_reference_required() {
        assert_eq!(
            (backend.as_str(), engine().as_str()),
            ("cuda", "gpu-native"),
            "--require-simd-reference-byte-equal requires --backend cuda --engine gpu-native"
        );
        assert!(
            simd_reference_reuse_input_gate_passes(true, reuse_input),
            "--require-simd-reference-byte-equal requires --reuse-input"
        );
    }
    let reps: usize = arg("--reps")
        .unwrap_or_else(|| "3".to_string())
        .parse()
        .unwrap();
    assert!(reps >= 1, "--reps must be >= 1");

    let source = build_input_source();
    let program = source.label();
    let n = source.n();

    // --adapt-only: run VM + adapt, report the cycle count, and exit before proving.
    // Useful for validating that large PIEs are accepted by the bootloader/adapter
    // without paying for a full (slow) SIMD prove locally.
    if flag("--adapt-only") || std::env::var("STWO_ADAPT_ONLY").as_deref() == Ok("1") {
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal requires a standard proof run"
        );
        reject_gpu_native_architecture_gate_without_proof("adapt-only mode");
        reject_proof_mutation_gate_without_proof("adapt-only mode");
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

    // M6-a: resident two-proof throughput. Unlike --pipeline (host-feed overlap of
    // a SERIAL prover), this proves N full proofs and reports the TWO-PROOF-WALL
    // metrics the M6 gates are defined on (<14.8s beats non-diet throughput, <11s
    // meaningful, <8s strong). Increment 1 proves them SEQUENTIALLY (the baseline =
    // N x single ≈ 22.5s at M5c 11.23s) to establish the harness + metrics before
    // the stream-explicit concurrent scheduler lands.
    if let Some(n) = arg("--resident-pipeline") {
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal is not supported by resident-pipeline mode"
        );
        let n: usize = n.parse().expect("--resident-pipeline <N>");
        run_resident_pipeline(&source, &backend, n);
        return;
    }

    // M6-a increment 2: N proofs CONCURRENTLY (one host thread + own prover each),
    // vs --resident-pipeline's sequential baseline. Reports the same two-proof-wall
    // metrics plus overlap_speedup (serial_wall / wall). This is the throughput
    // experiment the M6 gates measure (<14.8s / <11s / <8s two-proof wall).
    if let Some(n) = arg("--resident-concurrent") {
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal is not supported by resident-concurrent mode"
        );
        let n: usize = n.parse().expect("--resident-concurrent <N>");
        run_resident_concurrent(&source, &backend, n);
        return;
    }

    // P5 sustained-throughput mode; absent flag keeps today's behavior exactly.
    if let Some(depth) = arg("--pipeline") {
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal is not supported by pipeline mode"
        );
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
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal requires a standard proof run"
        );
        reject_gpu_native_architecture_gate_without_proof("witness JIT self-test mode");
        reject_proof_mutation_gate_without_proof("witness JIT self-test mode");
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
        assert!(
            !simd_reference_required(),
            "--require-simd-reference-byte-equal requires a standard proof run"
        );
        reject_gpu_native_architecture_gate_without_proof("device interaction self-test mode");
        reject_proof_mutation_gate_without_proof("device interaction self-test mode");
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
    let mut reusable_input = reuse_input.then_some(loaded.input);

    let mut times = Vec::new();
    let mut graph_submit_samples = Vec::with_capacity(reps);
    let mut proofs = Vec::with_capacity(reps);
    let mut vram_peak_gb = 0.0f64;
    let proof_loop_started_unix_ns = unix_time_ns();
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
        graph_submit_samples.push(
            (engine() == "gpu-native")
                .then(last_graph_submit_sample)
                .flatten(),
        );
        vram_peak_gb = vram_peak_gb.max(rep_vram);
        times.push(elapsed);
        proofs.push(proof);
        eprintln!("rep={rep} prove_s={elapsed:.3}");
        emit_phase_totals(rep);
    }
    let proof_loop_finished_unix_ns = unix_time_ns();
    // Fail an invalid eager proof before paying for the slow SIMD oracle. Both
    // validation phases remain outside every reported GPU proving sample.
    let (mut validation, serialized_gpu_proofs) = validate_gpu_proofs(proofs, true, true);
    // The qualification path consumes the retained input only after the timed
    // repetitions and full GPU verification, so the SIMD proof adds neither a
    // clone nor memory pressure to the measured GPU samples.
    let simd_reference = simd_reference_required().then(|| {
        let input = reusable_input
            .take()
            .expect("SIMD reference input must remain available");
        compute_simd_reference(input, prover_params(variant))
    });
    apply_simd_reference(
        &mut validation,
        &serialized_gpu_proofs,
        simd_reference.as_ref(),
    );
    let outcome = RepOutcome {
        times,
        graph_submit_samples,
        proof_loop_started_unix_ns,
        proof_loop_finished_unix_ns,
        proof_size: validation.proof_size,
        gpu_proof_blake3: validation.gpu_proof_blake3,
        verify_ms: validation.verify_ms,
        verified_reps: validation.verified_reps,
        proof_byte_equal: validation.proof_byte_equal,
        simd_reference_byte_equal: validation.simd_reference_byte_equal,
        simd_reference: validation.simd_reference,
        proof_mutation: validation.proof_mutation,
        vram_peak_gb,
    };
    print_main_record(
        &program,
        &backend,
        n,
        cycle_count,
        pie_n_steps,
        &outcome,
        true,
        last_vm_s,
        last_adapt_s,
    );
    enforce_proof_byte_equal(outcome.proof_byte_equal);
    enforce_simd_reference_byte_equal(outcome.simd_reference_byte_equal);
    enforce_proof_mutation_rejected(outcome.proof_mutation.as_ref());
    // Silence unused-import warnings when only one backend path is exercised.
    let _ = CairoSerialize::serialize as fn(&u64, &mut Vec<starknet_ff::FieldElement>);
}

#[cfg(test)]
mod tests {
    use stwo_backend_cuda::CudaExecTelemetry;

    use crate::gpu_bench_physical::resident_session_telemetry_json;

    use super::{
        cairo_verification_error_class, claimed_graph_submit_gap_ns,
        compiled_composition_vertical_checkpoint_gate, configure_graph_submit_policy,
        configure_resident_backend, graph_capture_claim_admissible, graph_submit_gap_average_ns,
        initial_proof_byte_equal, mutate_claimed_sum, packed_numerator_measurement_control_gate,
        parse_compiled_composition_vertical_checkpoint_args,
        parse_packed_numerator_measurement_control_args, parse_resident_backend_args,
        pcs_telemetry_json, performance_claim_admissible_for, proof_byte_equal_gate_passes,
        proof_mutation_gate_passes, quantile, simd_reference_gate_passes,
        simd_reference_reuse_input_gate_passes, throughput_mhz, validate_gpu_native_architecture,
        validate_resident_session_architecture, validate_strict_aot_provenance, AotRuntimeStats,
        CairoVerificationError, CudaPcsDriverTelemetry, CudaPcsRuntimeMode, GpuProverConfig,
        GraphSubmitSample, RequiredCudaPcsRuntimeMode, ResidentBackend, ResidentSessionTelemetry,
        SecureField, REQUIRED_CUDA_PCS_ARCHITECTURE,
    };

    fn complete_telemetry(runtime_mode: CudaPcsRuntimeMode) -> CudaPcsDriverTelemetry {
        CudaPcsDriverTelemetry {
            architecture: REQUIRED_CUDA_PCS_ARCHITECTURE,
            runtime_mode,
            stage_started: [1; 7],
            stage_finished: [1; 7],
            batched_tree_decommit: true,
            exec: None,
            expected_graph_launches: None,
            expected_kernel_launches: None,
        }
    }

    #[test]
    fn quantile_handles_empty_and_singleton_samples() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[7.0], 0.0), Some(7.0));
        assert_eq!(quantile(&[7.0], 0.95), Some(7.0));
    }

    #[test]
    fn quantile_sorts_and_interpolates() {
        let samples = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&samples, 0.0), Some(1.0));
        assert_eq!(quantile(&samples, 0.5), Some(3.0));
        assert!((quantile(&samples, 0.95).unwrap() - 4.8).abs() < f64::EPSILON * 8.0);
        assert_eq!(quantile(&samples, 1.0), Some(5.0));
    }

    #[test]
    fn mixed_statement_throughput_distribution_is_not_applicable() {
        assert_eq!(throughput_mhz(Some(10_000_000), Some(2.0), false), None);
        assert_eq!(throughput_mhz(Some(10_000_000), Some(2.0), true), Some(5.0));
    }

    #[test]
    fn resident_backend_parser_is_exact_and_fail_closed() {
        assert_eq!(
            parse_resident_backend_args(["gpu_bench", "--resident-backend", "legacy-resident"])
                .unwrap(),
            ResidentBackend::LegacyResident
        );
        assert_eq!(
            parse_resident_backend_args(["gpu_bench", "--resident-backend", "replacement-v1"])
                .unwrap(),
            ResidentBackend::ReplacementV1
        );
        assert_eq!(
            parse_resident_backend_args(["gpu_bench"]).unwrap(),
            ResidentBackend::LegacyResident
        );
        assert!(
            parse_resident_backend_args(["gpu_bench", "--resident-backend", "replacement"])
                .is_err()
        );
        assert!(parse_resident_backend_args(["gpu_bench", "--resident-backend"]).is_err());
    }

    #[test]
    fn resident_backend_parser_rejects_ambiguous_cli_forms() {
        for args in [
            vec!["gpu_bench", "--resident-backend=replacement-v1"],
            vec![
                "gpu_bench",
                "--resident-backend",
                "legacy-resident",
                "--resident-backend",
                "legacy-resident",
            ],
            vec![
                "gpu_bench",
                "--resident-backend",
                "legacy-resident",
                "--resident-backend",
                "replacement-v1",
            ],
            vec![
                "gpu_bench",
                "--resident-backend",
                "--require-gpu-native-architecture",
            ],
        ] {
            assert!(parse_resident_backend_args(args).is_err());
        }
    }

    #[test]
    fn compiled_composition_vertical_checkpoint_parser_is_exact_and_default_off() {
        assert!(!parse_compiled_composition_vertical_checkpoint_args(["gpu_bench"]).unwrap());
        assert!(parse_compiled_composition_vertical_checkpoint_args([
            "gpu_bench",
            "--compiled-composition-vertical-checkpoint",
        ])
        .unwrap());
        for args in [
            vec![
                "gpu_bench",
                "--compiled-composition-vertical-checkpoint=true",
            ],
            vec![
                "gpu_bench",
                "--compiled-composition-vertical-checkpoint",
                "--compiled-composition-vertical-checkpoint",
            ],
        ] {
            assert!(parse_compiled_composition_vertical_checkpoint_args(args).is_err());
        }
    }

    #[test]
    fn packed_numerator_measurement_control_parser_is_exact_and_default_off() {
        assert!(!parse_packed_numerator_measurement_control_args(["gpu_bench"]).unwrap());
        assert!(parse_packed_numerator_measurement_control_args([
            "gpu_bench",
            "--packed-numerator-measurement-control",
        ])
        .unwrap());
        for args in [
            vec!["gpu_bench", "--packed-numerator-measurement-control=true"],
            vec![
                "gpu_bench",
                "--packed-numerator-measurement-control",
                "--packed-numerator-measurement-control",
            ],
        ] {
            assert!(parse_packed_numerator_measurement_control_args(args).is_err());
        }
    }

    #[test]
    fn packed_numerator_measurement_control_is_replacement_only() {
        assert!(packed_numerator_measurement_control_gate(
            true,
            "cuda",
            "gpu-native",
            ResidentBackend::ReplacementV1,
            true,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
        )
        .is_ok());
        assert!(packed_numerator_measurement_control_gate(
            false,
            "simd",
            "legacy",
            ResidentBackend::LegacyResident,
            false,
            RequiredCudaPcsRuntimeMode::DetachedEager,
        )
        .is_ok());
        for invalid in [
            (
                "simd",
                "gpu-native",
                ResidentBackend::ReplacementV1,
                true,
                RequiredCudaPcsRuntimeMode::ArenaGraph,
            ),
            (
                "cuda",
                "legacy",
                ResidentBackend::ReplacementV1,
                true,
                RequiredCudaPcsRuntimeMode::ArenaGraph,
            ),
            (
                "cuda",
                "gpu-native",
                ResidentBackend::LegacyResident,
                true,
                RequiredCudaPcsRuntimeMode::ArenaGraph,
            ),
            (
                "cuda",
                "gpu-native",
                ResidentBackend::ReplacementV1,
                false,
                RequiredCudaPcsRuntimeMode::ArenaGraph,
            ),
            (
                "cuda",
                "gpu-native",
                ResidentBackend::ReplacementV1,
                true,
                RequiredCudaPcsRuntimeMode::DetachedEager,
            ),
        ] {
            assert!(packed_numerator_measurement_control_gate(
                true, invalid.0, invalid.1, invalid.2, invalid.3, invalid.4,
            )
            .is_err());
        }
    }

    #[test]
    fn compiled_composition_vertical_checkpoint_requires_replacement_and_byte_oracle() {
        let valid = (
            true,
            ResidentBackend::ReplacementV1,
            true,
            true,
            false,
            false,
            false,
        );
        assert!(compiled_composition_vertical_checkpoint_gate(
            valid.0, valid.1, valid.2, valid.3, valid.4, valid.5, valid.6
        )
        .is_ok());
        assert!(compiled_composition_vertical_checkpoint_gate(
            false,
            ResidentBackend::LegacyResident,
            false,
            false,
            true,
            true,
            true,
        )
        .is_ok());
        for invalid in [
            (
                true,
                ResidentBackend::LegacyResident,
                true,
                true,
                false,
                false,
                false,
            ),
            (
                true,
                ResidentBackend::ReplacementV1,
                false,
                true,
                false,
                false,
                false,
            ),
            (
                true,
                ResidentBackend::ReplacementV1,
                true,
                false,
                false,
                false,
                false,
            ),
            (
                true,
                ResidentBackend::ReplacementV1,
                true,
                true,
                true,
                false,
                false,
            ),
            (
                true,
                ResidentBackend::ReplacementV1,
                true,
                true,
                false,
                true,
                false,
            ),
            (
                true,
                ResidentBackend::ReplacementV1,
                true,
                true,
                false,
                false,
                true,
            ),
        ] {
            assert!(compiled_composition_vertical_checkpoint_gate(
                invalid.0, invalid.1, invalid.2, invalid.3, invalid.4, invalid.5, invalid.6
            )
            .is_err());
        }
    }

    #[test]
    fn resident_backend_config_requires_strict_arena_graph_for_replacement() {
        let mut legacy = GpuProverConfig::default();
        configure_resident_backend(
            &mut legacy,
            ResidentBackend::LegacyResident,
            false,
            RequiredCudaPcsRuntimeMode::DetachedEager,
        )
        .unwrap();
        assert_eq!(legacy.resident_backend, ResidentBackend::LegacyResident);
        assert!(!legacy.strict);

        let mut replacement = GpuProverConfig::default();
        configure_resident_backend(
            &mut replacement,
            ResidentBackend::ReplacementV1,
            true,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
        )
        .unwrap();
        assert_eq!(replacement.resident_backend, ResidentBackend::ReplacementV1);
        assert!(replacement.strict);

        assert!(configure_resident_backend(
            &mut GpuProverConfig::default(),
            ResidentBackend::ReplacementV1,
            false,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
        )
        .is_err());
        assert!(configure_resident_backend(
            &mut GpuProverConfig::default(),
            ResidentBackend::ReplacementV1,
            true,
            RequiredCudaPcsRuntimeMode::DetachedEager,
        )
        .is_err());
    }

    #[test]
    fn detached_gpu_native_timing_is_never_a_performance_claim() {
        assert!(!performance_claim_admissible_for(
            "gpu-native",
            true,
            RequiredCudaPcsRuntimeMode::DetachedEager,
            false,
            false,
        ));
        assert!(!performance_claim_admissible_for(
            "gpu-native",
            false,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            false,
            false,
        ));
        assert!(performance_claim_admissible_for(
            "gpu-native",
            true,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            false,
            false,
        ));
        assert!(performance_claim_admissible_for(
            "legacy",
            false,
            RequiredCudaPcsRuntimeMode::DetachedEager,
            false,
            false,
        ));
        assert!(!performance_claim_admissible_for(
            "gpu-native",
            true,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            true,
            false,
        ));
        assert!(!performance_claim_admissible_for(
            "gpu-native",
            true,
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            false,
            true,
        ));
    }

    #[test]
    fn graph_gap_capture_admits_only_an_observed_passing_gate() {
        assert!(graph_capture_claim_admissible(true, false, None));
        assert!(graph_capture_claim_admissible(true, true, Some(true)));
        assert!(!graph_capture_claim_admissible(true, true, Some(false)));
        assert!(!graph_capture_claim_admissible(true, true, None));
        assert!(!graph_capture_claim_admissible(false, false, Some(true)));
    }

    #[test]
    #[should_panic(expected = "graph-submit diagnostic and capture modes are mutually exclusive")]
    fn graph_submit_policy_rejects_mixed_modes() {
        configure_graph_submit_policy(&mut GpuProverConfig::default(), true, true);
    }

    #[test]
    fn graph_submit_policy_instruments_only_diagnostics() {
        let assert_policy = |diagnostic, capture, allow_slow, record_intervals| {
            let mut config = GpuProverConfig::default();
            configure_graph_submit_policy(&mut config, diagnostic, capture);
            assert_eq!(config.allow_slow_graph_submit_diagnostic, allow_slow);
            assert_eq!(
                config.record_graph_replay_intervals_diagnostic,
                record_intervals
            );
        };
        assert_policy(false, false, false, false);
        assert_policy(true, false, true, true);
        assert_policy(false, true, true, false);
    }

    #[test]
    fn graph_gap_claim_uses_every_warm_repetition() {
        let sample = |max_ns| GraphSubmitSample {
            total_ns: max_ns,
            max_ns,
            graph_launches: 1,
        };
        let samples = [sample(90_000_000), sample(10_000_000), sample(60_000_000)];
        assert_eq!(claimed_graph_submit_gap_ns(&samples), Some(60_000_000));
        assert_eq!(claimed_graph_submit_gap_ns(&samples[..1]), Some(90_000_000));
        assert_eq!(claimed_graph_submit_gap_ns(&[]), None);
    }

    #[test]
    fn graph_gap_average_uses_inter_launch_gap_count() {
        let sample = |graph_launches| GraphSubmitSample {
            total_ns: 13_000_000,
            max_ns: 1_000_000,
            graph_launches,
        };
        assert_eq!(graph_submit_gap_average_ns(sample(14)), Some(1_000_000.0));
        assert_eq!(graph_submit_gap_average_ns(sample(1)), None);
        assert_eq!(graph_submit_gap_average_ns(sample(0)), None);
    }

    #[test]
    fn proof_equality_requires_two_comparable_proofs() {
        assert_eq!(initial_proof_byte_equal(true, 1), None);
        assert_eq!(initial_proof_byte_equal(false, 2), None);
        assert_eq!(initial_proof_byte_equal(true, 2), Some(true));

        assert!(proof_byte_equal_gate_passes(false, None));
        assert!(proof_byte_equal_gate_passes(true, Some(true)));
        assert!(!proof_byte_equal_gate_passes(true, None));
        assert!(!proof_byte_equal_gate_passes(true, Some(false)));
    }

    #[test]
    fn simd_reference_gate_fails_closed() {
        assert!(simd_reference_gate_passes(false, None));
        assert!(simd_reference_gate_passes(true, Some(true)));
        assert!(!simd_reference_gate_passes(true, None));
        assert!(!simd_reference_gate_passes(true, Some(false)));
    }

    #[test]
    fn simd_reference_cli_requires_reuse_input() {
        assert!(simd_reference_reuse_input_gate_passes(false, false));
        assert!(simd_reference_reuse_input_gate_passes(false, true));
        assert!(simd_reference_reuse_input_gate_passes(true, true));
        assert!(!simd_reference_reuse_input_gate_passes(true, false));
    }

    #[test]
    fn proof_mutation_gate_fails_closed() {
        assert!(proof_mutation_gate_passes(false, None));
        assert!(proof_mutation_gate_passes(true, Some(true)));
        assert!(!proof_mutation_gate_passes(true, None));
        assert!(!proof_mutation_gate_passes(true, Some(false)));
    }

    #[test]
    fn structured_mutation_changes_claimed_sum_by_one() {
        let mut claimed_sum = SecureField::from(7_u32);
        mutate_claimed_sum(&mut claimed_sum);
        assert_eq!(claimed_sum, SecureField::from(8_u32));
    }

    #[test]
    fn mutation_error_class_is_machine_readable() {
        assert_eq!(
            cairo_verification_error_class(&CairoVerificationError::InvalidLogupSum),
            "invalid_logup_sum"
        );
        assert_eq!(
            cairo_verification_error_class(&CairoVerificationError::ProofOfWork),
            "proof_of_work"
        );
    }

    #[test]
    fn pcs_architecture_telemetry_is_machine_readable() {
        let telemetry = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        let json = pcs_telemetry_json(&telemetry);
        assert_eq!(json["gpu_pcs_driver_complete"], true);
        assert_eq!(
            json["gpu_pcs_driver_architecture"],
            REQUIRED_CUDA_PCS_ARCHITECTURE
        );
        assert_eq!(json["gpu_pcs_stage_started"]["Assembly"], 1);
        assert_eq!(json["gpu_pcs_stage_finished"]["Assembly"], 1);
    }

    #[test]
    fn pcs_execution_topology_expectations_are_machine_readable() {
        let exec = CudaExecTelemetry {
            graph_launches: 14,
            kernel_launches: 2_530,
            graph_submit_gap_ns_total: 42_000_000,
            graph_submit_gap_ns_max: 3_000_000,
            ..CudaExecTelemetry::default()
        };
        let telemetry = CudaPcsDriverTelemetry::completed_arena_graph(exec, 14, 2_530);
        let json = pcs_telemetry_json(&telemetry);
        assert_eq!(json["gpu_graph_launches"], 14);
        assert_eq!(json["gpu_kernel_launches"], 2_530);
        assert_eq!(json["gpu_expected_graph_launches"], 14);
        assert_eq!(json["gpu_expected_kernel_launches"], 2_530);
        assert_eq!(json["gpu_graph_submit_gap_ns_total"], 42_000_000);
        assert_eq!(json["gpu_max_graph_submit_gap_ms"], 3.0);
    }

    #[test]
    fn execution_table_setup_telemetry_is_machine_readable() {
        let telemetry = ResidentSessionTelemetry {
            transcript_segments: 15,
            execution_tables_ingest: Some(
                stwo_backend_cuda::PreparedExecutionTablesIngestTelemetry {
                    compact_h2d_bytes: 4096,
                    compact_h2d_copies: 3,
                    descriptor_h2d_bytes: 64,
                    descriptor_h2d_copies: 2,
                    sync_calls: 1,
                },
            ),
            ..ResidentSessionTelemetry::default()
        };
        let json = resident_session_telemetry_json(&telemetry);
        assert_eq!(json["gpu_execution_tables_ingest_compact_h2d_bytes"], 4096);
        assert_eq!(json["gpu_execution_tables_ingest_compact_h2d_copies"], 3);
        assert_eq!(json["gpu_execution_tables_ingest_descriptor_h2d_bytes"], 64);
        assert_eq!(json["gpu_execution_tables_ingest_descriptor_h2d_copies"], 2);
        assert_eq!(json["gpu_execution_tables_ingest_syncs"], 1);
        assert_eq!(json["gpu_transcript_segments"], 15);
    }

    #[test]
    fn resident_session_gate_is_scoped_to_arena_graph_mode() {
        assert_eq!(
            validate_resident_session_architecture(
                RequiredCudaPcsRuntimeMode::DetachedEager,
                ResidentBackend::LegacyResident,
                None,
            ),
            Ok(())
        );
        assert!(validate_resident_session_architecture(
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            ResidentBackend::LegacyResident,
            None,
        )
        .unwrap_err()
        .contains("Graph-A setup telemetry is missing"));

        let telemetry = ResidentSessionTelemetry {
            protocol_policy: Some(
                stwo_cairo_gpu_prover::protocol_plan::ProtocolPlanPolicy::replacement_v1(
                    0x1234, 2048,
                ),
            ),
            ..ResidentSessionTelemetry::default()
        };
        assert!(validate_resident_session_architecture(
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            ResidentBackend::LegacyResident,
            Some(&telemetry),
        )
        .unwrap_err()
        .contains("requested resident backend legacy-resident but prepared replacement-v1"));
    }

    #[test]
    fn architecture_gate_accepts_only_complete_typed_cuda_telemetry() {
        let telemetry = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        assert_eq!(
            validate_gpu_native_architecture(
                "cuda",
                "gpu-native",
                RequiredCudaPcsRuntimeMode::DetachedEager,
                Some(&telemetry),
            ),
            Ok(())
        );

        for (backend, engine, telemetry, expected) in [
            (
                "simd",
                "gpu-native",
                Some(&telemetry),
                "backend must be cuda",
            ),
            (
                "cuda",
                "legacy",
                Some(&telemetry),
                "engine must be gpu-native",
            ),
            ("cuda", "gpu-native", None, "telemetry is missing"),
        ] {
            let error = validate_gpu_native_architecture(
                backend,
                engine,
                RequiredCudaPcsRuntimeMode::DetachedEager,
                telemetry,
            )
            .unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn architecture_gate_rejects_wrong_tag_mode_partial_duplicate_and_unbatched() {
        let mut telemetry = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        telemetry.architecture = "legacy-cuda-driver";
        assert!(validate_gpu_native_architecture(
            "cuda",
            "gpu-native",
            RequiredCudaPcsRuntimeMode::DetachedEager,
            Some(&telemetry),
        )
        .unwrap_err()
        .contains("architecture must be"));

        let telemetry = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        assert!(validate_gpu_native_architecture(
            "cuda",
            "gpu-native",
            RequiredCudaPcsRuntimeMode::ArenaGraph,
            Some(&telemetry),
        )
        .unwrap_err()
        .contains("runtime mode must be ArenaGraph"));

        let mut partial = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        partial.stage_finished[2] = 0;
        assert!(validate_gpu_native_architecture(
            "cuda",
            "gpu-native",
            RequiredCudaPcsRuntimeMode::DetachedEager,
            Some(&partial),
        )
        .unwrap_err()
        .contains("finish exactly once"));

        let mut duplicate = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        duplicate.stage_started[4] = 2;
        assert!(validate_gpu_native_architecture(
            "cuda",
            "gpu-native",
            RequiredCudaPcsRuntimeMode::DetachedEager,
            Some(&duplicate),
        )
        .unwrap_err()
        .contains("started=2"));

        let mut unbatched = complete_telemetry(CudaPcsRuntimeMode::DetachedEager);
        unbatched.batched_tree_decommit = false;
        assert!(validate_gpu_native_architecture(
            "cuda",
            "gpu-native",
            RequiredCudaPcsRuntimeMode::DetachedEager,
            Some(&unbatched),
        )
        .unwrap_err()
        .contains("batched tree decommit"));
    }

    #[test]
    fn strict_architecture_gate_allows_only_aot_loads_and_aot_cache_hits() {
        let clean = AotRuntimeStats {
            aot_loads: 3,
            aot_cache_hits: 9,
            ..AotRuntimeStats::default()
        };
        assert_eq!(validate_strict_aot_provenance(Some(&clean)), Ok(()));
        assert!(validate_strict_aot_provenance(None)
            .unwrap_err()
            .contains("telemetry is missing"));

        for field in [
            "aot_misses",
            "runtime_loads",
            "runtime_cache_hits",
            "strict_rejections",
        ] {
            let mut stats = clean;
            match field {
                "aot_misses" => stats.aot_misses = 1,
                "runtime_loads" => stats.runtime_loads = 1,
                "runtime_cache_hits" => stats.runtime_cache_hits = 1,
                "strict_rejections" => stats.strict_rejections = 1,
                _ => unreachable!(),
            }
            let error = validate_strict_aot_provenance(Some(&stats)).unwrap_err();
            assert!(error.contains(field), "unexpected error: {error}");
        }

        assert_eq!(
            validate_strict_aot_provenance(Some(&AotRuntimeStats::default())),
            Ok(())
        );
    }
}
