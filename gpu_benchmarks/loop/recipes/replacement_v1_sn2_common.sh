#!/usr/bin/env bash
# Shared fail-closed gates for the first replacement-v1 SN2 checkpoint.
# Sourced by the diagnostic and same-binary timing recipes; never run directly.

[[ -n "${REPLACEMENT_SN2_MODE:-}" ]] \
  || { echo "set REPLACEMENT_SN2_MODE before sourcing replacement_v1_sn2_common.sh" >&2; return 2; }

CHECKPOINT_ROOT="${CAIRO%/stwo_cairo_prover}"
CHECKPOINT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CHECKPOINT_PREFIX="replacement_v1_sn2_${REPLACEMENT_SN2_MODE}_${CHECKPOINT_STAMP}"
CHECKPOINT_PIE="$CHECKPOINT_ROOT/gpu_benchmarks/pie/sn/SN_PIE_2.zip"
CHECKPOINT_BOOTLOADER=/workspace/bench_inputs/simple_bootloader_compiled.json
CHECKPOINT_ADAPTED=/workspace/bench_inputs/SN_PIE_2.replacement_v1.adapted.bin
CHECKPOINT_INPUT_MANIFEST="$CHECKPOINT_ROOT/gpu_benchmarks/pie/SHA256SUMS"
CHECKPOINT_ADAPTED_MANIFEST="$CHECKPOINT_ROOT/gpu_benchmarks/pie/ADAPTED_SHA256SUMS"
CHECKPOINT_AOT_MANIFEST="$STWO/crates/backend-cuda-kernels/cuda/generated/aot_manifest.json"
CHECKPOINT_GPU_BENCH="$CAIRO/target/release/gpu_bench"
CHECKPOINT_AOT_CHECK="$CAIRO/target/release/aot_index_check"
CHECKPOINT_SEAL=/workspace/bench_inputs/replacement_v1_sn2_checkpoint.seal.json
CHECKPOINT_EMPTY_WORKTREE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
CHECKPOINT_AOT_MANIFEST_SHA256=3a6214cbf8417b74d7f6618d7840cc8269d2c7f69017263033f4a905866af954

# One architecture, one compiler fingerprint, and one persistent target/cache line.
export STWO_CUDA_ARCH=sm_90
export STWO_CUDA_BUILD_JOBS=16
export RUSTFLAGS='-C target-cpu=native'

checkpoint_artifact() {
  printf '%s/%s.%s' "$RUN" "$CHECKPOINT_PREFIX" "$1"
}

checkpoint_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

checkpoint_manifest_hash() {
  local manifest="$1" name="$2" value
  value="$(awk -v name="$name" '$2 == name { print $1 }' "$manifest")"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "missing or invalid pinned SHA-256 for $name in $manifest" >&2; return 1; }
  printf '%s' "$value"
}

checkpoint_require_hash() {
  local value="$1" bits="$2" label="$3"
  [[ "$value" =~ ^[0-9a-f]+$ && ${#value} -eq $((bits / 4)) ]] \
    || { echo "$label is not a lowercase SHA-$bits identity: $value" >&2; return 1; }
}

checkpoint_require_clean_source_identity() {
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_HEAD:-}" 160 stwo_head
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_CAIRO_HEAD:-}" 160 stwo_cairo_head
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_WORKTREE_HASH:-}" 256 stwo_worktree_hash
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH:-}" 256 stwo_cairo_worktree_hash
  [[ "$STWO_PARITY_REF_STWO_WORKTREE_HASH" == "$CHECKPOINT_EMPTY_WORKTREE_SHA256" ]] \
    || { echo "checkpoint requires a fully committed stwo tree" >&2; return 1; }
  [[ "$STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH" == "$CHECKPOINT_EMPTY_WORKTREE_SHA256" ]] \
    || { echo "checkpoint requires a fully committed stwo-cairo tree" >&2; return 1; }
}

checkpoint_source_input_identity() {
  local raw_expected raw_actual boot_expected boot_actual out
  checkpoint_require_clean_source_identity
  [[ "$STWO_BOOTLOADER_JSON" == "$CHECKPOINT_BOOTLOADER" ]] \
    || { echo "pod bootloader path drifted: $STWO_BOOTLOADER_JSON" >&2; return 1; }
  for path in "$CHECKPOINT_PIE" "$CHECKPOINT_BOOTLOADER" \
      "$CHECKPOINT_INPUT_MANIFEST" "$CHECKPOINT_ADAPTED_MANIFEST" "$CHECKPOINT_AOT_MANIFEST"; do
    [[ -f "$path" ]] || { echo "missing checkpoint input: $path" >&2; return 1; }
  done
  raw_expected="$(checkpoint_manifest_hash "$CHECKPOINT_INPUT_MANIFEST" SN_PIE_2.zip)"
  boot_expected="$(checkpoint_manifest_hash "$CHECKPOINT_INPUT_MANIFEST" simple_bootloader_compiled.json)"
  raw_actual="$(checkpoint_sha256 "$CHECKPOINT_PIE")"
  boot_actual="$(checkpoint_sha256 "$CHECKPOINT_BOOTLOADER")"
  [[ "$raw_actual" == "$raw_expected" ]] || { echo "SN2 PIE identity mismatch" >&2; return 1; }
  [[ "$boot_actual" == "$boot_expected" ]] || { echo "bootloader identity mismatch" >&2; return 1; }
  [[ "$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" == "$CHECKPOINT_AOT_MANIFEST_SHA256" ]] \
    || { echo "AOT manifest identity drifted; regenerate and deliberately repin this recipe" >&2; return 1; }

  # A failed diagnostic must not leave an older timing-admissible seal behind.
  rm -f "$CHECKPOINT_SEAL"
  out="$(checkpoint_artifact source_input_identity.json)"
  RAW_SHA="$raw_actual" BOOT_SHA="$boot_actual" OUT="$out" python3 - <<'PY'
import json, os
record = {
    "schema": "stwo.replacement-v1-sn2.source-input-identity.v1",
    "source": {
        "stwo": {"head": os.environ["STWO_PARITY_REF_STWO_HEAD"],
                 "worktree_sha256": os.environ["STWO_PARITY_REF_STWO_WORKTREE_HASH"]},
        "stwo_cairo": {"head": os.environ["STWO_PARITY_REF_STWO_CAIRO_HEAD"],
                       "worktree_sha256": os.environ["STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH"]},
    },
    "inputs": {
        "SN_PIE_2.zip": os.environ["RAW_SHA"],
        "simple_bootloader_compiled.json": os.environ["BOOT_SHA"],
    },
}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_hardware_identity() {
  checkpoint_capture_hardware_identity "$(checkpoint_artifact hardware_identity.json)"
}

checkpoint_capture_hardware_identity() {
  local row out="$1"
  row="$(nvidia-smi \
    --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,compute_cap,persistence_mode,mig.mode.current,ecc.mode.current,compute_mode,power.limit,clocks.max.sm,clocks.max.memory \
    --format=csv,noheader,nounits)"
  GPU_ROW="$row" OUT="$out" python3 - <<'PY'
import csv, io, json, math, os
rows = list(csv.reader(io.StringIO(os.environ["GPU_ROW"])))
if len(rows) != 1 or len(rows[0]) != 13:
    raise SystemExit("checkpoint requires exactly one queryable GPU")
name, uuid, pci_bus_id, memory_mib, driver, compute, persistence, mig, ecc, compute_mode, power_limit, max_sm, max_memory = [value.strip() for value in rows[0]]
try:
    memory_mib = int(memory_mib)
    power_limit = float(power_limit)
    max_sm = int(max_sm)
    max_memory = int(max_memory)
except ValueError as error:
    raise SystemExit(f"invalid numeric GPU identity: {error}")
if "H100" not in name or memory_mib < 79000 or compute != "9.0":
    raise SystemExit(f"replacement-v1 checkpoint requires H100 sm_90 >=79,000 MiB; got {name}, {compute}, {memory_mib}")
if persistence not in {"Enabled", "Disabled"} or mig != "Disabled" or ecc not in {"Enabled", "Disabled"} or compute_mode != "Default":
    raise SystemExit(f"unstable GPU policy: persistence={persistence}, MIG={mig}, ECC={ecc}, compute={compute_mode}")
if not all((uuid, pci_bus_id, driver)) or not math.isfinite(power_limit) or power_limit <= 0 or max_sm <= 0 or max_memory <= 0:
    raise SystemExit("GPU power/clock policy is unavailable")
record = {"schema": "stwo.replacement-v1-sn2.hardware-identity.v2", "name": name,
          "uuid": uuid, "pci_bus_id": pci_bus_id, "memory_mib": memory_mib,
          "compute_capability": compute, "driver_version": driver,
          "persistence_mode": persistence, "mig_mode": mig, "ecc_mode": ecc,
          "compute_mode": compute_mode, "power_limit_w": power_limit,
          "max_sm_clock_mhz": max_sm, "max_memory_clock_mhz": max_memory}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_build() {
  local out
  checkpoint_reject_ambient_overrides
  cd "$CAIRO"
  cargo build --release --locked -p stwo-cairo-gpu-prover \
    --bin gpu_bench --bin aot_index_check --features pie-bench
  [[ -x "$CHECKPOINT_GPU_BENCH" && -x "$CHECKPOINT_AOT_CHECK" ]] \
    || { echo "checkpoint release binaries were not produced" >&2; return 1; }
  out="$(checkpoint_artifact build_identity.json)"
  GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
  AOT_CHECK_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
  AOT_MANIFEST_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" \
  RUSTC_VERSION="$(rustc -Vv)" NVCC_VERSION="$(nvcc --version)" OUT="$out" python3 - <<'PY'
import json, os
record = {"schema": "stwo.replacement-v1-sn2.build-identity.v1",
          "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
          "aot_index_check_sha256": os.environ["AOT_CHECK_SHA"],
          "aot_manifest_sha256": os.environ["AOT_MANIFEST_SHA"],
          "cuda_arch": "sm_90", "rustc": os.environ["RUSTC_VERSION"],
          "nvcc": os.environ["NVCC_VERSION"]}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_adapted_input_identity() {
  local log expected actual bytes out
  checkpoint_reject_ambient_overrides
  log="$(checkpoint_artifact adapted_input.txt)"
  rm -f "$CHECKPOINT_ADAPTED"
  if ! env STWO_DUMP_INPUT="$CHECKPOINT_ADAPTED" "$CHECKPOINT_GPU_BENCH" \
      --pie "$CHECKPOINT_PIE" --backend simd --engine legacy --adapt-only >"$log" 2>&1; then
    cat "$log"
    return 1
  fi
  cat "$log"
  [[ -f "$CHECKPOINT_ADAPTED" ]] || { echo "adapter did not emit SN2 input" >&2; return 1; }
  expected="$(checkpoint_manifest_hash "$CHECKPOINT_ADAPTED_MANIFEST" SN_PIE_2.adapted.bin)"
  actual="$(checkpoint_sha256 "$CHECKPOINT_ADAPTED")"
  bytes="$(stat -c %s "$CHECKPOINT_ADAPTED")"
  [[ "$actual" == "$expected" ]] || { echo "fresh SN2 adapted-input identity mismatch" >&2; return 1; }
  out="$(checkpoint_artifact adapted_input_identity.json)"
  ADAPTED_SHA="$actual" ADAPTED_BYTES="$bytes" ADAPTER_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    OUT="$out" python3 - <<'PY'
import json, os
record = {"schema": "stwo.replacement-v1-sn2.adapted-input-identity.v1",
          "sha256": os.environ["ADAPTED_SHA"], "bytes": int(os.environ["ADAPTED_BYTES"]),
          "adapter_binary_sha256": os.environ["ADAPTER_SHA"], "fresh": True}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_require_one_test() {
  local log="$1" needle="$2"
  python3 - "$log" "$needle" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
started = re.findall(r"(?m)^test ([^ ]+) \.\.\.(?: |$)", text)
if len(started) != 1 or sys.argv[2] not in started[0]:
    raise SystemExit(f"expected exactly one executed {sys.argv[2]} test; got {started}")
if not re.search(r"test result: ok\. 1 passed; 0 failed;", text):
    raise SystemExit("test process did not report exactly one passing test")
PY
}

checkpoint_fp256_carry_oracles() {
  local oracle captured out
  checkpoint_reject_ambient_overrides
  oracle="$(checkpoint_artifact fp256_oracle.txt)"
  captured="$(checkpoint_artifact poseidon_captured_aot.txt)"
  cd "$CAIRO"
  if ! env STWO_CUDA_SOUNDNESS_REQUIRED=1 cargo test --release --locked \
      -p stwo-cairo-prover --lib stwo_wit_deduce_oracle_matches_fast_deduction \
      -- --nocapture --test-threads=1 >"$oracle" 2>&1; then
    cat "$oracle"
    return 1
  fi
  checkpoint_require_one_test "$oracle" stwo_wit_deduce_oracle_matches_fast_deduction
  if ! env STWO_CUDA_SOUNDNESS_REQUIRED=1 STWO_CUDA_WITNESS_JIT_MAX_INSTRS=8192 \
      cargo test --release --locked -p stwo-cairo-prover --lib \
      poseidon_combination_37_strict_aot_captured_row \
      -- --ignored --nocapture --test-threads=1 >"$captured" 2>&1; then
    cat "$captured"
    return 1
  fi
  checkpoint_require_one_test "$captured" poseidon_combination_37_strict_aot_captured_row
  cat "$oracle"
  cat "$captured"
  out="$(checkpoint_artifact fp256_carry_oracles.json)"
  ORACLE_SHA="$(checkpoint_sha256 "$oracle")" CAPTURED_SHA="$(checkpoint_sha256 "$captured")" \
    OUT="$out" python3 - <<'PY'
import json, os
record = {"schema": "stwo.replacement-v1-sn2.fp256-carry-oracles.v1", "pass": True,
          "deduce_oracle_log_sha256": os.environ["ORACLE_SHA"],
          "strict_captured_aot_log_sha256": os.environ["CAPTURED_SHA"],
          "executed_tests": 2}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_numerator_source_sha() {
  cd "$STWO"
  local files=(
    crates/backend-cuda/tests/prepared_quotient_numerator_sn3_bench_native.rs
    crates/backend-cuda/tests/support/sn3_quotient_numerator_bench.rs
    crates/backend-cuda/tests/support/sn3_quotient_topology_fixture.rs
    crates/backend-cuda/src/backend/quotient_numerator_single_write.rs
    crates/backend-cuda/src/backend/prepared_quotient_numerator.rs
    crates/backend-cuda/src/backend/prepared_quotient_numerator/plan.rs
    crates/backend-cuda/src/backend/prepared_quotient_numerator/bindings.rs
    crates/backend-cuda/src/backend/prepared_quotient_numerator/launch.rs
    crates/backend-cuda/src/backend/prepared_quotient_numerator/single_write.rs
    crates/backend-cuda-kernels/cuda/quotient_numerator_single_write.cu
    crates/backend-cuda-kernels/cuda/quotient_numerator_single_write.cuh
    crates/backend-cuda-kernels/cuda/quotients.cu
    crates/backend-cuda-kernels/cuda/quotients.cuh
  )
  sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
}

checkpoint_exact_numerator_ab() {
  local build_json build_err executable source_sha module_sha log out
  checkpoint_reject_ambient_overrides
  build_json="$(checkpoint_artifact numerator_build.jsonl.txt)"
  build_err="$(checkpoint_artifact numerator_build.stderr.txt)"
  log="$(checkpoint_artifact numerator_ab.txt)"
  out="$(checkpoint_artifact numerator_ab.json)"
  cd "$STWO"
  # Reuse the already-built Cairo release dependency graph; Cargo hashes feature
  # variants safely, while a second workspace target would rebuild the CUDA pack.
  if ! env CARGO_TARGET_DIR="$CAIRO/target" cargo test --release --locked -p stwo-backend-cuda \
      --test prepared_quotient_numerator_sn3_bench_native \
      sn3_hybrid_graph_host_wall_benchmark --no-run --message-format=json \
      >"$build_json" 2>"$build_err"; then
    cat "$build_err"
    return 1
  fi
  executable="$(python3 - "$build_json" <<'PY'
import json, sys
matches = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try: value = json.loads(line)
    except json.JSONDecodeError: continue
    target = value.get("target") or {}
    if (value.get("reason") == "compiler-artifact"
            and target.get("name") == "prepared_quotient_numerator_sn3_bench_native"
            and value.get("executable")):
        matches.append(value["executable"])
if len(matches) != 1:
    raise SystemExit(f"expected one exact numerator test executable, got {matches}")
print(matches[0])
PY
)"
  [[ -x "$executable" ]] || { echo "missing exact numerator test executable: $executable" >&2; return 1; }
  source_sha="$(checkpoint_numerator_source_sha)"
  module_sha="$(checkpoint_sha256 "$executable")"
  if ! env CARGO_TARGET_DIR="$CAIRO/target" STWO_SN3_NUMERATOR_BENCH_ITERS=5 \
      STWO_SN3_NUMERATOR_SOURCE_PROJECTION_SHA256="$source_sha" \
      STWO_SN3_NUMERATOR_CUDA_MODULE_SHA256="$module_sha" \
      cargo test --release --locked -p stwo-backend-cuda \
      --test prepared_quotient_numerator_sn3_bench_native \
      sn3_hybrid_graph_host_wall_benchmark \
      -- --ignored --exact --nocapture --test-threads=1 >"$log" 2>&1; then
    cat "$log"
    return 1
  fi
  checkpoint_require_one_test "$log" sn3_hybrid_graph_host_wall_benchmark
  SOURCE_SHA="$source_sha" MODULE_SHA="$module_sha" python3 - "$log" "$out" <<'PY'
import json, os, re, sys
decoder = json.JSONDecoder()
records = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    marker = line.find('{"schema":"stwo.sn3_quotient_numerator_hybrid.host_wall.v5"')
    if marker >= 0:
        records.append(decoder.raw_decode(line[marker:])[0])
if len(records) != 1:
    raise SystemExit(f"expected one exact numerator A/B record, got {len(records)}")
r = records[0]
top = r["topology"]
if (top["groups"], top["eligible_groups"], top["legacy_groups"], top["terms"]) != (19, 18, 1, 6341):
    raise SystemExit("exact SN3 numerator topology drifted")
byte_geometry = r["bytes"]
if (byte_geometry["shared_data_dual_workspace_arena"] != 41889121376
        or byte_geometry["workspace_span_each"] != 67901168
        or byte_geometry["second_workspace_arena_delta"] != 67901152
        or byte_geometry["validated_canonical_output"] != 402645136):
    raise SystemExit("exact SN3 numerator byte geometry drifted")
memory = r["device_memory"]
arena_bytes = byte_geometry["shared_data_dual_workspace_arena"]
if (memory["total"] < arena_bytes
        or memory["free_before_arena"] < arena_bytes
        or memory["free_after_arena"] >= memory["free_before_arena"]
        or memory["isolated_pool_used_after_arena"] < arena_bytes
        or memory["isolated_pool_reserved_after_arena"] < memory["isolated_pool_used_after_arena"]):
    raise SystemExit("exact SN3 numerator device-memory accounting is invalid")
identity = r["identity"]
if identity["topology_fixture_blake3"] != "ea31e3ff054c8d12d32d5b84a3d712987b31bb1fd3fb044fb27758453b49fbda":
    raise SystemExit("exact SN3 topology fixture identity drifted")
if identity["input_recipe_blake3"] != "e4c2f871c2d05b81588a5407f06cb49c7ed76834d2e363d2214bd34e7defcf31":
    raise SystemExit("exact SN3 input recipe identity drifted")
digests = [identity[name] for name in (
    "eager_legacy_blake3", "eager_hybrid_blake3", "captured_legacy_blake3",
    "captured_hybrid_blake3", "timed_legacy_blake3", "timed_hybrid_blake3",
    "post_timing_legacy_blake3", "post_timing_hybrid_blake3")]
if len(set(digests)) != 1 or not re.fullmatch(r"[0-9a-f]{64}", digests[0]):
    raise SystemExit("legacy/hybrid numerator outputs are not exactly identical")
artifact = r["artifact_identity"]
if (artifact["identity_complete"] is not True
        or artifact["source_projection_sha256"] != os.environ["SOURCE_SHA"]
        or artifact["cuda_module_sha256"] != os.environ["MODULE_SHA"]
        or artifact["cuda_build_mode"] != "cuda"):
    raise SystemExit("exact numerator artifact identity is incomplete")
if r["iterations"] != 5 or any(len(r["samples_ms"][arm]) != 5 for arm in ("legacy", "hybrid")):
    raise SystemExit("exact numerator A/B did not record five samples per arm")
if any(float(r["speedup"][p]) <= 0 for p in ("p50", "p95")):
    raise SystemExit("exact numerator A/B timing is invalid")
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(r, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps({"exact_numerator_ab": "PASS", "p50_speedup": r["speedup"]["p50"],
                  "p95_speedup": r["speedup"]["p95"]}, sort_keys=True))
PY
  cat "$log"
}

checkpoint_aot_identity() {
  local manifest_sha raw out key
  local -a keys key_args
  checkpoint_reject_ambient_overrides
  manifest_sha="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")"
  [[ "$manifest_sha" == "$CHECKPOINT_AOT_MANIFEST_SHA256" ]] \
    || { echo "AOT manifest SHA-256 drifted" >&2; return 1; }
  mapfile -t keys < <(python3 - "$CHECKPOINT_AOT_MANIFEST" <<'PY'
import json, re, sys
entries = json.load(open(sys.argv[1], encoding="utf-8"))
keys = sorted({entry.get("cache_key") for entry in entries})
if len(entries) != 131 or len(keys) != 131 or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{16}", k) for k in keys):
    raise SystemExit("pinned AOT manifest does not contain the expected 131 unique keys")
print("\n".join(keys))
PY
  )
  [[ ${#keys[@]} -eq 131 ]] || { echo "AOT key extraction failed" >&2; return 1; }
  key_args=()
  for key in "${keys[@]}"; do key_args+=(--key "$key"); done
  raw="$(checkpoint_artifact aot_index_raw.json)"
  "$CHECKPOINT_AOT_CHECK" --sm 90 "${key_args[@]}" >"$raw"
  out="$(checkpoint_artifact aot_identity.json)"
  MANIFEST_SHA="$manifest_sha" CHECKER_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
    GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    python3 - "$raw" "$out" <<'PY'
import json, os, re, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if (result.get("pass") is not True or result.get("sm") != 90
        or result.get("required_unique_key_count") != 131 or result.get("missing_keys") != []
        or not re.fullmatch(r"(?!0{16})[0-9a-f]{16}", result.get("loaded_manifest_hash", ""))):
    raise SystemExit(f"embedded sm_90 AOT pack identity/coverage failed: {result}")
record = {"schema": "stwo.replacement-v1-sn2.aot-identity.v1", **result,
          "manifest_sha256": os.environ["MANIFEST_SHA"],
          "checker_binary_sha256": os.environ["CHECKER_SHA"],
          "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"]}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_reject_ambient_overrides() {
  local name
  local -a forbidden=(
    STWO_CUDA_RETAINED_LDE_BUDGET_BYTES STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS
    STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE STWO_CUDA_COMPOSITION_DIRECT_RETENTION
    STWO_CUDA_B2N_STAGE_FUSED STWO_CUDA_BLAKE2S_INTERIOR_FUSED
    STWO_CUDA_COMPOSITION_WIDE STWO_CUDA_RELATION_SCAN_TAIL STWO_CUDA_FRI_FOLD_FUSED
    STWO_CUDA_FEED_PRIVATIZED STWO_CUDA_PCS_REFERENCE STWO_CUDA_DECOMMIT_GATHER_REFERENCE
    STWO_CUDA_DEVICE_INTERACTION STWO_CUDA_WITNESS_EDGES STWO_CUDA_MEM_COUNT_FEEDS
    STWO_CUDA_STREAM_FANOUT STWO_CUDA_STREAM_LEAF_COMMIT STWO_CUDA_PIPELINED_COMMIT
    STWO_CAIRO_LOW_MEMORY STWO_CAIRO_STREAM_LDE STWO_DIET_REBUILD_PREPROCESSED
    STWO_FORCE_EXTEND_EVAL_MODE STWO_STORE_COEFFS PREPROCESSED_TRACE_GPU_GENERATE
    STWO_GPU_OPERATIONAL_SAFETY_RESERVE_BYTES
    STWO_CUDA_DISABLE_JIT STWO_CUDA_WITNESS_JIT STWO_CUDA_WITNESS_JIT_PROVE
    STWO_CUDA_WITNESS_JIT_MAX_INSTRS STWO_JIT_CACHE_DIR STWO_JIT_CUBIN_CACHE
    STWO_JIT_DISABLE_SPLIT STWO_JIT_FORCE_RELAX STWO_JIT_LOG STWO_JIT_MAX_KERNEL_INSTRS
    STWO_JIT_NVRTC_OPTS STWO_JIT_PARALLEL_COMPILE STWO_WITNESS_JIT_SELFTEST
    STWO_BENCH_REQUIRE_GPU_NATIVE_ARCHITECTURE STWO_BENCH_REQUIRE_GPU_PCS_RUNTIME_MODE
    STWO_BENCH_REQUIRE_PROOF_BYTE_EQUAL STWO_BENCH_REQUIRE_SIMD_REFERENCE_BYTE_EQUAL
    STWO_BENCH_REQUIRE_PROOF_MUTATION_REJECTED GPU_PCS_RUNTIME_MODE
    CUDA_LAUNCH_BLOCKING CUDA_DEVICE_MAX_CONNECTIONS
  )
  for name in "${forbidden[@]}"; do
    [[ -z "${!name+x}" ]] || { echo "checkpoint rejects ambient override $name" >&2; return 1; }
  done
}

checkpoint_run_sn2() {
  local mode="$1" reps="$2" stdout stderr telemetry proof sampler rc
  local -a args=(
    --pie "$CHECKPOINT_PIE" --backend cuda --engine gpu-native
    --resident-backend replacement-v1
    --require-gpu-native-architecture --require-gpu-pcs-runtime-mode arena-graph
    --reuse-input --reps "$reps"
    --require-proof-byte-equal --require-simd-reference-byte-equal
    --require-proof-mutation-rejected
  )
  [[ "$mode" == diagnostic || "$mode" == timing ]] || { echo "invalid checkpoint mode: $mode" >&2; return 2; }
  if [[ "$mode" == diagnostic ]]; then args+=(--diagnostic-allow-slow-graph-submit); fi
  checkpoint_reject_ambient_overrides
  stdout="$(checkpoint_artifact stdout.txt)"
  stderr="$(checkpoint_artifact stderr.txt)"
  telemetry="$(checkpoint_artifact telemetry.csv)"
  proof="$(checkpoint_artifact proof.bin)"
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv -lms 250 >"$telemetry" 2>&1 &
  sampler=$!
  set +e
  env STWO_BENCH_TRACE=json STWO_DUMP_PROOF="$proof" \
    "$CHECKPOINT_GPU_BENCH" "${args[@]}" >"$stdout" 2>"$stderr"
  rc=$?
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  cat "$stderr"
  [[ $rc -eq 0 ]] || { echo "replacement-v1 SN2 process failed rc=$rc" >&2; return "$rc"; }
  [[ -s "$stdout" && -s "$telemetry" && -s "$proof" ]] \
    || { echo "replacement-v1 SN2 omitted a required artifact" >&2; return 1; }
  python3 - "$telemetry" <<'PY'
import sys
lines = [line for line in open(sys.argv[1], encoding="utf-8", errors="replace") if line.strip()]
if len(lines) < 2 or "timestamp" not in lines[0].lower() or any("error" in line.lower() for line in lines):
    raise SystemExit("nvidia-smi telemetry is missing or invalid")
PY
  if grep -Eiq 'cuda[_ -]error|cudaError|out[_ -]of[_ -]memory|memory allocation failed|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)' "$stdout" "$stderr"; then
    echo "replacement-v1 SN2 emitted a CUDA/OOM signature" >&2
    return 1
  fi
  printf '%s  %s\n' "$(checkpoint_sha256 "$proof")" "$(basename "$proof")" \
    >"$(checkpoint_artifact proof.sha256.txt)"
  cat "$stdout"
}

checkpoint_validate_sn2() {
  local mode="$1" reps="$2" stdout aot proof out
  stdout="$(checkpoint_artifact stdout.txt)"
  aot="$(checkpoint_artifact aot_identity.json)"
  proof="$(checkpoint_artifact proof.bin)"
  out="$(checkpoint_artifact record.json)"
  python3 - "$stdout" "$aot" "$proof" "$CHECKPOINT_SEAL" "$out" "$mode" "$reps" \
    "$CHECKPOINT_ROOT/gpu_benchmarks" <<'PY'
import hashlib, json, math, re, sys

raw_path, aot_path, proof_path, seal_path, out_path, mode, reps_text, module_path = sys.argv[1:]
sys.path.insert(0, module_path)
from validate_replacement_v1_reuse import require_resident_reuse

reps = int(reps_text)
objects = []
for line in open(raw_path, encoding="utf-8", errors="replace"):
    try: value = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value, dict): objects.append(value)
records = [value for value in objects
           if value.get("program") == "SN_PIE_2.zip" and value.get("backend") == "cuda"]
if len(records) != 1:
    raise SystemExit(f"expected one primary SN2 CUDA record, got {len(records)}")
r = records[0]
aot = json.load(open(aot_path, encoding="utf-8"))

def require(condition, message):
    if not condition: raise SystemExit(message)

def hex64(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

require(r.get("engine") == "gpu-native" and r.get("n") == 1, "wrong engine or statement multiplicity")
require(r.get("cycle_count") == 7977397 and r.get("pie_n_steps") == 7706864, "SN2 statement geometry drifted")
require((r.get("security_bits"), r.get("n_queries"), r.get("pow_bits"), r.get("fold_step")) == (96, 70, 26, 3), "security configuration drifted")
require(r.get("gpu_resident_backend_requested") == "replacement-v1", "replacement-v1 selector was not requested")
require(r.get("gpu_resident_backend") == "replacement-v1", "prepared resident backend is not replacement-v1")
require(r.get("gpu_native_architecture_required") is True, "native architecture gate was not required")
require(r.get("gpu_native_architecture_gate_passed") is True, "native architecture gate did not pass")
require(r.get("gpu_pcs_required_runtime_mode") == "arena-graph", "ArenaGraph was not required")
require(r.get("gpu_pcs_runtime_mode") == "ArenaGraph", "actual PCS runtime is not ArenaGraph")
require(r.get("gpu_pcs_driver_architecture") == "cuda-typed-pcs-driver-v1", "wrong PCS driver architecture")
require(r.get("gpu_pcs_driver_complete") is True and r.get("gpu_pcs_batched_tree_decommit") is True, "PCS driver did not complete the native path")
expected_stages = {"Assembly", "FriCommitAndFold", "FriQueryAndDecommit", "OodsEvaluation", "ProofOfWork", "QuotientAndCompaction", "TreeDecommit"}
for field in ("gpu_pcs_stage_started", "gpu_pcs_stage_finished"):
    stages = r.get(field)
    require(isinstance(stages, dict) and set(stages) == expected_stages and all(bool(value) for value in stages.values()), f"incomplete {field}")
require(r.get("gpu_graph_a_setup_gate_passed") is True, "strict Graph-A setup gate failed")
require(r.get("gpu_planned_numerator_schedule") == "hybrid-single-write", "planned numerator schedule drifted")
require(r.get("gpu_prepared_numerator_schedule") == "hybrid-single-write", "actual numerator schedule fell back")
require(isinstance(r.get("gpu_prepared_numerator_eligible_groups"), int) and r["gpu_prepared_numerator_eligible_groups"] > 0, "no hybrid numerator group executed")
require(isinstance(r.get("gpu_prepared_numerator_legacy_groups"), int) and r["gpu_prepared_numerator_legacy_groups"] >= 0, "invalid legacy numerator group count")
require(isinstance(r.get("gpu_protocol_key"), int) and not isinstance(r["gpu_protocol_key"], bool)
        and 0 < r["gpu_protocol_key"] < 2**64, "protocol key is not a nonzero u64")
topology_digest = r.get("gpu_shape_executable_topology_digest")
require(hex64(topology_digest) and topology_digest != "0" * 64,
        "topology digest is not a nonzero 256-bit identity")
require_resident_reuse(r, reps)
require(r.get("gpu_policy_retained_lde_budget_bytes") == 64 * 1024**3, "replacement-v1 LDE policy drifted")
require(r.get("gpu_policy_commit_mode") == "domain-progressive", "replacement-v1 commit policy drifted")
require(r.get("gpu_policy_direct_composition_retention") == "exact-native", "replacement-v1 composition retention drifted")
require(r.get("gpu_policy_numerator_source") == "reuse-retained-evaluations", "replacement-v1 numerator source drifted")
require(r.get("gpu_policy_interpolation_mode") == "stage-fused-out-of-place", "replacement-v1 interpolation policy drifted")
require(r.get("gpu_policy_blake2s_interior_fused") is False, "replacement-v1 Blake interior policy drifted")
require(r.get("gpu_policy_composition_launch_mode") == "serial", "replacement-v1 composition launch policy drifted")
require(r.get("gpu_policy_relation_tail_mode") == "segmented", "replacement-v1 relation-tail policy drifted")
require(r.get("gpu_policy_fri_fold_launch_mode") == "per-fold", "replacement-v1 FRI-fold policy drifted")
require(r.get("gpu_policy_witness_feed_launch_mode") == "global-atomics", "replacement-v1 witness-feed policy drifted")
for field in ("gpu_setup_base_migration_copies", "gpu_setup_lookup_host_copies", "gpu_setup_legacy_witness_fallbacks"):
    require(r.get(field) == 0, f"legacy setup/fallback executed: {field}={r.get(field)!r}")
for field in ("gpu_aot_misses", "gpu_aot_runtime_loads", "gpu_aot_runtime_cache_hits", "gpu_aot_strict_rejections"):
    require(r.get(field) == 0, f"JIT/AOT fallback executed: {field}={r.get(field)!r}")
activity = (r.get("gpu_aot_loads"), r.get("gpu_aot_cache_hits"))
require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in activity)
        and sum(activity) > 0, "runtime telemetry recorded no positive AOT activity")
require(r.get("gpu_aot_provenance_gate_passed") is True, "strict AOT provenance gate failed")
embedded_hash = int(aot["loaded_manifest_hash"], 16)
require(r.get("gpu_aot_manifest_hash") == embedded_hash and r.get("gpu_policy_kernel_manifest_hash") == embedded_hash, "runtime AOT identity differs from checked embedded pack")
require(r.get("reps") == reps and r.get("verified_reps") == reps, "not every proof repetition verified")
require(r.get("proof_byte_equal_required") is True and r.get("proof_comparison_applicable") is True and r.get("proof_byte_equal") is True, "GPU/GPU proof bytes differ")
require(r.get("simd_reference_required") is True and r.get("simd_reference_comparison_applicable") is True, "fresh SIMD comparison was not required/applicable")
require(r.get("simd_reference_fresh") is True and r.get("simd_reference_byte_equal") is True, "fresh SIMD/GPU proof bytes differ")
require(hex64(r.get("gpu_proof_blake3")) and r.get("simd_reference_blake3") == r.get("gpu_proof_blake3"), "proof digest identity is invalid")
require(r.get("proof_mutation_required") is True, "structured proof mutation was not required")
require(r.get("proof_mutation_kind") == "interaction_claim.memory_id_to_big.claimed_sum_plus_one", "wrong structured proof mutation")
require(r.get("proof_mutation_rejected") is True and r.get("proof_mutation_error_class") == "invalid_logup_sum", "structured proof mutation was not deterministically rejected")
require(isinstance(r.get("gpu_max_graph_submit_gap_ms"), (int, float)) and math.isfinite(r["gpu_max_graph_submit_gap_ms"]) and r["gpu_max_graph_submit_gap_ms"] >= 0, "missing graph-submit timing")
require(r.get("performance_measurement_available") is True, "GPU timing is unavailable")
if mode == "diagnostic":
    require(reps == 2 and r.get("benchmark_diagnostic_mode") is True, "first checkpoint must be reps=2 diagnostic")
    require(r.get("benchmark_diagnostic_reason") == "graph-submit-gap-only", "diagnostic softened more than graph-submit")
    require(r.get("performance_claim_admissible") is False, "diagnostic timing must not be admissible")
elif mode == "timing":
    require(reps >= 6 and r.get("benchmark_diagnostic_mode") is False and r.get("benchmark_diagnostic_reason") is None, "timing follow-on must have diagnostics disabled")
    require(r.get("performance_claim_admissible") is True, "timing follow-on is not admissible")
    require(r.get("gpu_graph_submit_gap_strict_gate_passed") is True, "timing follow-on exceeded strict graph-submit gap")
    require(r.get("warm_sample_count") == reps - 1, "timing follow-on lacks the expected warm samples")
    for field in ("prove_s_warm_median", "prove_s_warm_p95", "useful_mhz_median", "useful_mhz_at_warm_p95"):
        require(isinstance(r.get(field), (int, float)) and math.isfinite(r[field]) and r[field] > 0, f"invalid timing metric {field}")
    seal = json.load(open(seal_path, encoding="utf-8"))
    require(seal.get("schema") == "stwo.replacement-v1-sn2.checkpoint-seal.v2"
            and seal.get("diagnostic_pass") is True, "timing seal is not a passing diagnostic")
    require(r["gpu_proof_blake3"] == seal.get("proof_blake3"), "timing proof digest differs from diagnostic")
    proof_sha256 = hashlib.sha256(open(proof_path, "rb").read()).hexdigest()
    require(proof_sha256 == seal.get("proof_dump_sha256"), "timing proof bytes differ from diagnostic")
    shape = seal.get("shape_receipt") or {}
    require(shape == {
        "protocol_key": r["gpu_protocol_key"],
        "topology_digest": r["gpu_shape_executable_topology_digest"],
        "numerator_eligible_groups": r["gpu_prepared_numerator_eligible_groups"],
        "numerator_legacy_groups": r["gpu_prepared_numerator_legacy_groups"],
    }, "timing shape/numerator receipt differs from diagnostic")
else:
    raise SystemExit(f"unknown validation mode: {mode}")
r["checkpoint_validation"] = {"schema": "stwo.replacement-v1-sn2.record-validation.v1",
                              "verdict": "PASS", "mode": mode,
                              "graph_submit_timing_soft": mode == "diagnostic"}
with open(out_path, "w", encoding="utf-8") as stream:
    json.dump(r, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps({"replacement_v1_sn2": "PASS", "mode": mode,
                  "proof_blake3": r["gpu_proof_blake3"],
                  "useful_mhz_median": r.get("useful_mhz_median"),
                  "graph_submit_gap_ms": r["gpu_max_graph_submit_gap_ms"],
                  "graph_submit_gate": r.get("gpu_graph_submit_gap_strict_gate_passed")}, sort_keys=True))
PY
}

checkpoint_seal_diagnostic() {
  local source hardware build adapted carry numerator aot record proof proof_sha run_seal
  checkpoint_reject_ambient_overrides
  source="$(checkpoint_artifact source_input_identity.json)"
  hardware="$(checkpoint_artifact hardware_identity.json)"
  build="$(checkpoint_artifact build_identity.json)"
  adapted="$(checkpoint_artifact adapted_input_identity.json)"
  carry="$(checkpoint_artifact fp256_carry_oracles.json)"
  numerator="$(checkpoint_artifact numerator_ab.json)"
  aot="$(checkpoint_artifact aot_identity.json)"
  record="$(checkpoint_artifact record.json)"
  proof="$(checkpoint_artifact proof.bin)"
  proof_sha="$(awk '{print $1}' "$(checkpoint_artifact proof.sha256.txt)")"
  run_seal="$(checkpoint_artifact seal.json)"
  for path in "$source" "$hardware" "$build" "$adapted" "$carry" "$numerator" "$aot" "$record" "$proof"; do
    [[ -s "$path" ]] || { echo "cannot seal missing checkpoint receipt: $path" >&2; return 1; }
  done
  checkpoint_require_hash "$proof_sha" 256 proof_dump_sha256
  SOURCE="$source" HARDWARE="$hardware" BUILD="$build" ADAPTED="$adapted" CARRY="$carry" \
    NUMERATOR="$numerator" AOT="$aot" RECORD="$record" PROOF="$proof" PROOF_SHA="$proof_sha" \
    GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    AOT_CHECK_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
    AOT_MANIFEST_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" \
    OUT="$CHECKPOINT_SEAL" python3 - <<'PY'
import hashlib, json, os, tempfile
def load(name):
    with open(os.environ[name], encoding="utf-8") as stream: return json.load(stream)
def sha(name):
    return hashlib.sha256(open(os.environ[name], "rb").read()).hexdigest()
source, hardware, build, adapted = map(load, ("SOURCE", "HARDWARE", "BUILD", "ADAPTED"))
carry, numerator, aot, record = map(load, ("CARRY", "NUMERATOR", "AOT", "RECORD"))
if (source.get("schema") != "stwo.replacement-v1-sn2.source-input-identity.v1"
        or hardware.get("schema") != "stwo.replacement-v1-sn2.hardware-identity.v2"
        or build.get("schema") != "stwo.replacement-v1-sn2.build-identity.v1"
        or adapted.get("schema") != "stwo.replacement-v1-sn2.adapted-input-identity.v1"
        or carry.get("pass") is not True
        or numerator.get("schema") != "stwo.sn3_quotient_numerator_hybrid.host_wall.v4"
        or record.get("checkpoint_validation", {}).get("verdict") != "PASS"
        or record["checkpoint_validation"].get("mode") != "diagnostic"):
    raise SystemExit("diagnostic receipts are not sealable")
if (build.get("gpu_bench_sha256") != os.environ["GPU_BENCH_SHA"]
        or build.get("aot_index_check_sha256") != os.environ["AOT_CHECK_SHA"]
        or build.get("aot_manifest_sha256") != os.environ["AOT_MANIFEST_SHA"]
        or adapted.get("adapter_binary_sha256") != os.environ["GPU_BENCH_SHA"]
        or aot.get("gpu_bench_sha256") != os.environ["GPU_BENCH_SHA"]
        or aot.get("checker_binary_sha256") != os.environ["AOT_CHECK_SHA"]
        or aot.get("manifest_sha256") != os.environ["AOT_MANIFEST_SHA"]):
    raise SystemExit("diagnostic receipt/binary identity cross-check failed")
if sha("PROOF") != os.environ["PROOF_SHA"]:
    raise SystemExit("diagnostic proof hash receipt differs from proof bytes")
seal = {"schema": "stwo.replacement-v1-sn2.checkpoint-seal.v2", "diagnostic_pass": True,
        "source": source["source"], "inputs": source["inputs"], "hardware": hardware,
        "adapted_input_sha256": adapted["sha256"],
        "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
        "aot_index_check_sha256": os.environ["AOT_CHECK_SHA"],
        "aot_manifest_sha256": aot["manifest_sha256"],
        "aot_loaded_manifest_hash": aot["loaded_manifest_hash"],
        "proof_dump_sha256": os.environ["PROOF_SHA"],
        "proof_blake3": record["gpu_proof_blake3"],
        "shape_receipt": {
            "protocol_key": record["gpu_protocol_key"],
            "topology_digest": record["gpu_shape_executable_topology_digest"],
            "numerator_eligible_groups": record["gpu_prepared_numerator_eligible_groups"],
            "numerator_legacy_groups": record["gpu_prepared_numerator_legacy_groups"],
        },
        "receipts_sha256": {name.lower(): sha(name) for name in
            ("SOURCE", "HARDWARE", "BUILD", "ADAPTED", "CARRY", "NUMERATOR", "AOT", "RECORD")}}
directory = os.path.dirname(os.environ["OUT"])
fd, temporary = tempfile.mkstemp(prefix="replacement-v1-sn2-seal.", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(seal, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, os.environ["OUT"])
finally:
    if os.path.exists(temporary): os.unlink(temporary)
print(json.dumps(seal, sort_keys=True))
PY
  cp "$CHECKPOINT_SEAL" "$run_seal"
}

checkpoint_verify_sealed_diagnostic() {
  local current_hardware run_seal
  checkpoint_reject_ambient_overrides
  checkpoint_require_clean_source_identity
  [[ -s "$CHECKPOINT_SEAL" ]] || { echo "run the passing diagnostic checkpoint before timing" >&2; return 1; }
  # Re-read the embedded index without rebuilding. The timing validator consumes
  # this current-run receipt and the seal comparison prevents pack substitution.
  checkpoint_aot_identity
  current_hardware="$(checkpoint_artifact hardware_identity.json)"
  checkpoint_capture_hardware_identity "$current_hardware"
  run_seal="$(checkpoint_artifact seal.json)"
  GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
  AOT_CHECK_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
  RAW_SHA="$(checkpoint_sha256 "$CHECKPOINT_PIE")" \
  BOOT_SHA="$(checkpoint_sha256 "$CHECKPOINT_BOOTLOADER")" \
  ADAPTED_SHA="$(checkpoint_sha256 "$CHECKPOINT_ADAPTED")" \
  AOT_MANIFEST_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" \
  CURRENT_AOT="$(checkpoint_artifact aot_identity.json)" \
  CURRENT_HARDWARE="$current_hardware" python3 - "$CHECKPOINT_SEAL" <<'PY'
import json, os, sys
seal = json.load(open(sys.argv[1], encoding="utf-8"))
if seal.get("schema") != "stwo.replacement-v1-sn2.checkpoint-seal.v2" or seal.get("diagnostic_pass") is not True:
    raise SystemExit("checkpoint seal is not a passing diagnostic")
expected_source = {"stwo": {"head": os.environ["STWO_PARITY_REF_STWO_HEAD"],
                             "worktree_sha256": os.environ["STWO_PARITY_REF_STWO_WORKTREE_HASH"]},
                   "stwo_cairo": {"head": os.environ["STWO_PARITY_REF_STWO_CAIRO_HEAD"],
                                  "worktree_sha256": os.environ["STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH"]}}
if seal.get("source") != expected_source:
    raise SystemExit("timing source/commit identity differs from the diagnostic")
if seal.get("inputs") != {"SN_PIE_2.zip": os.environ["RAW_SHA"],
                           "simple_bootloader_compiled.json": os.environ["BOOT_SHA"]}:
    raise SystemExit("timing raw-input identity differs from the diagnostic")
checks = (("adapted_input_sha256", "ADAPTED_SHA"), ("gpu_bench_sha256", "GPU_BENCH_SHA"),
          ("aot_index_check_sha256", "AOT_CHECK_SHA"), ("aot_manifest_sha256", "AOT_MANIFEST_SHA"))
for field, environment in checks:
    if seal.get(field) != os.environ[environment]:
        raise SystemExit(f"timing artifact differs from diagnostic: {field}")
current_aot = json.load(open(os.environ["CURRENT_AOT"], encoding="utf-8"))
if (current_aot.get("loaded_manifest_hash") != seal.get("aot_loaded_manifest_hash")
        or current_aot.get("manifest_sha256") != seal.get("aot_manifest_sha256")
        or current_aot.get("checker_binary_sha256") != seal.get("aot_index_check_sha256")
        or current_aot.get("gpu_bench_sha256") != seal.get("gpu_bench_sha256")):
    raise SystemExit("timing embedded AOT pack differs from the diagnostic")
hardware = json.load(open(os.environ["CURRENT_HARDWARE"], encoding="utf-8"))
if hardware != seal.get("hardware"):
    raise SystemExit("timing GPU identity/policy differs from the diagnostic")
print(json.dumps({"sealed_diagnostic": "PASS", "proof_blake3": seal["proof_blake3"],
                  "gpu": hardware["name"], "uuid": hardware["uuid"]}, sort_keys=True))
PY
  cp "$CHECKPOINT_SEAL" "$run_seal"
}
