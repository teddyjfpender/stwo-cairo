#!/usr/bin/env bash
# Shared fail-closed gates for the first replacement-v1 SN2 checkpoint.
# Sourced by the diagnostic and same-binary timing recipes; never run directly.

[[ -n "${REPLACEMENT_SN2_MODE:-}" ]] \
  || { echo "set REPLACEMENT_SN2_MODE before sourcing replacement_v1_sn2_common.sh" >&2; return 2; }
REPLACEMENT_SN2_COUNTER_POLICY="${REPLACEMENT_SN2_COUNTER_POLICY:-required}"
case "$REPLACEMENT_SN2_COUNTER_POLICY" in
  required|timing-only) ;;
  *) echo "invalid replacement SN2 counter policy: $REPLACEMENT_SN2_COUNTER_POLICY" >&2; return 2 ;;
esac

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
if [[ "$REPLACEMENT_SN2_COUNTER_POLICY" == timing-only ]]; then
  CHECKPOINT_SEAL="$RUN/replacement_v1_sn2_timing_only_checkpoint.seal.json"
else
  CHECKPOINT_SEAL=/workspace/bench_inputs/replacement_v1_sn2_checkpoint.seal.json
fi
CHECKPOINT_EMPTY_WORKTREE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
CHECKPOINT_AOT_MANIFEST_SHA256=1ff3089cf9c6c9284ddfbdcfd8258d3d329a115d1285ee8c066f563175005fa4
CHECKPOINT_AOT_TOTAL=373
CHECKPOINT_AOT_WITNESS=35
CHECKPOINT_AOT_ORDINARY_CONSTRAINT=219
CHECKPOINT_AOT_COMPOSITION_WAVE=119
CHECKPOINT_SN2_PACKED_OUTPUT_ROWS=20971472
CHECKPOINT_SN2_COMPOSITION_PARTS=153
CHECKPOINT_SN2_COMPOSITION_WAVES=18
CHECKPOINT_PROMOTION_HOST_PREPARATION_NS=120000000
CHECKPOINT_PROMOTION_USEFUL_MHZ=12
CHECKPOINT_NCU_KERNEL_REGEX='regex:stwo_composition_wave_.*|stwo_quotient_numerator_packed_single_write_kernel'
CHECKPOINT_NCU_LAUNCH_COUNT=19
CHECKPOINT_NSYS_FALLBACK=/opt/nvidia/nsight-systems/2024.6.2/bin/nsys

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

checkpoint_require_aot_manifest_shape() {
  AOT_TOTAL="$CHECKPOINT_AOT_TOTAL" AOT_WITNESS="$CHECKPOINT_AOT_WITNESS" \
    AOT_ORDINARY="$CHECKPOINT_AOT_ORDINARY_CONSTRAINT" \
    AOT_WAVE="$CHECKPOINT_AOT_COMPOSITION_WAVE" \
    python3 - "$CHECKPOINT_AOT_MANIFEST" <<'PY'
import json, os, re, sys

entries = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
    raise SystemExit("replacement AOT manifest is malformed")
witness = [entry for entry in entries if entry.get("kind") == "witness"]
waves = [entry for entry in entries
         if entry.get("kind") == "constraint"
         and re.fullmatch(r"wave_log_[0-9]+", str(entry.get("label", "")))]
ordinary = [entry for entry in entries
            if entry.get("kind") == "constraint" and entry not in waves]
classified = witness + waves + ordinary
expected = {
    "total": int(os.environ["AOT_TOTAL"]),
    "witness": int(os.environ["AOT_WITNESS"]),
    "ordinary_constraint": int(os.environ["AOT_ORDINARY"]),
    "composition_wave": int(os.environ["AOT_WAVE"]),
}
actual = {
    "total": len(entries),
    "witness": len(witness),
    "ordinary_constraint": len(ordinary),
    "composition_wave": len(waves),
}
if actual != expected or len(classified) != len(entries):
    raise SystemExit(f"replacement AOT pack shape drifted: expected={expected}, actual={actual}")
keys = [entry.get("cache_key") for entry in entries]
if (any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key) for key in keys)
        or len(set(keys)) != len(keys)):
    raise SystemExit("replacement AOT pack cache keys are invalid or duplicated")
print(json.dumps({"replacement_aot_pack": "PASS", **actual}, sort_keys=True))
PY
}

checkpoint_require_source_identity() {
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_HEAD:-}" 160 stwo_head
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_CAIRO_HEAD:-}" 160 stwo_cairo_head
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_WORKTREE_HASH:-}" 256 stwo_worktree_hash
  checkpoint_require_hash "${STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH:-}" 256 stwo_cairo_worktree_hash
}

checkpoint_require_clean_source_identity() {
  checkpoint_require_source_identity
  [[ "$STWO_PARITY_REF_STWO_WORKTREE_HASH" == "$CHECKPOINT_EMPTY_WORKTREE_SHA256" ]] \
    || { echo "checkpoint requires a fully committed stwo tree" >&2; return 1; }
  [[ "$STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH" == "$CHECKPOINT_EMPTY_WORKTREE_SHA256" ]] \
    || { echo "checkpoint requires a fully committed stwo-cairo tree" >&2; return 1; }
}

checkpoint_source_input_identity() {
  local source_policy="${1:-clean}" raw_expected raw_actual boot_expected boot_actual out
  case "$source_policy" in
    clean) checkpoint_require_clean_source_identity ;;
    iteration) checkpoint_require_source_identity ;;
    *) echo "invalid checkpoint source policy: $source_policy" >&2; return 2 ;;
  esac
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
  checkpoint_require_aot_manifest_shape || return 1
  [[ "$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" == "$CHECKPOINT_AOT_MANIFEST_SHA256" ]] \
    || { echo "AOT manifest identity drifted; regenerate and deliberately repin this recipe" >&2; return 1; }

  # A failed promotion diagnostic must not leave an older admissible seal behind.
  # Iteration runs never consume or mutate a promotion seal.
  [[ "$source_policy" != clean ]] || rm -f "$CHECKPOINT_SEAL"
  out="$(checkpoint_artifact source_input_identity.json)"
  RAW_SHA="$raw_actual" BOOT_SHA="$boot_actual" SOURCE_POLICY="$source_policy" \
    OUT="$out" python3 - <<'PY'
import json, os
record = {
    "schema": "stwo.replacement-v1-sn2.source-input-identity.v1",
    "source_policy": os.environ["SOURCE_POLICY"],
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
    "aot_pack": {
        "total": 373,
        "witness": 35,
        "ordinary_constraint": 219,
        "composition_wave": 119,
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

checkpoint_nsys_tool_identity() {
  local nsys_bin version out
  nsys_bin="$(command -v nsys || true)"
  [[ -n "$nsys_bin" ]] || nsys_bin="$CHECKPOINT_NSYS_FALLBACK"
  [[ -x "$nsys_bin" ]] \
    || { echo "required Nsight Systems executable is absent: $nsys_bin" >&2; return 1; }
  version="$("$nsys_bin" --version 2>&1)" \
    || { echo "Nsight Systems version probe failed: $nsys_bin" >&2; return 1; }
  out="$(checkpoint_artifact nsys_tool_identity.json)"
  NSYS_PATH="$nsys_bin" NSYS_VERSION="$version" OUT="$out" python3 - <<'PY'
import json, os
record = {"schema": "stwo.replacement-v1-sn2.nsys-tool-identity.v1",
          "path": os.environ["NSYS_PATH"], "version": os.environ["NSYS_VERSION"]}
with open(os.environ["OUT"], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_counter_acceptance() {
  local binary raw log out rc
  binary="/tmp/stwo-replacement-counter-acceptance.$$"
  raw="$(checkpoint_artifact counter_acceptance.csv)"
  log="$(checkpoint_artifact counter_acceptance.txt)"
  out="$(checkpoint_artifact counter_acceptance.json)"
  rm -f "$binary" "$raw" "$out"
  set +e
  {
    command -v nvcc && command -v ncu && ncu --version && ncu --query-metrics >/dev/null
  } >"$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    nvcc -std=c++14 -O2 -lineinfo -arch=sm_90 -x cu -o "$binary" - >>"$log" 2>&1 <<'CU'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void checkpoint_counter_kernel(int *value) {
    if (threadIdx.x == 0) *value = 1;
}

int main() {
    int *device = nullptr;
    int host = 0;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 1;
    if (cudaMemset(device, 0, sizeof(int)) != cudaSuccess) return 2;
    checkpoint_counter_kernel<<<1, 32>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    if (cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    if (cudaFree(device) != cudaSuccess) return 5;
    if (host != 1) return 6;
    std::printf("CHECKPOINT_COUNTER_KERNEL_RESULT=1\n");
    return 0;
}
CU
    rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    "$binary" >>"$log" 2>&1
    rc=$?
  fi
  if [[ $rc -eq 0 ]]; then
    ncu --target-processes all --kernel-name regex:checkpoint_counter_kernel \
      --metrics sm__cycles_elapsed.avg --csv "$binary" >"$raw" 2>&1
    rc=$?
  fi
  [[ ! -f "$raw" ]] || cat "$raw" >>"$log"
  set -e
  rm -f "$binary"
  cat "$log"
  [[ $rc -eq 0 ]] || { echo "NCU counter-permission acceptance failed rc=$rc" >&2; return "$rc"; }
  python3 - "$raw" "$log" "$(checkpoint_artifact hardware_identity.json)" "$out" <<'PY'
import csv, hashlib, json, math, pathlib, sys

raw, log, hardware_path, out = map(pathlib.Path, sys.argv[1:])
hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
if (hardware.get("schema") != "stwo.replacement-v1-sn2.hardware-identity.v2"
        or not isinstance(hardware.get("uuid"), str) or not hardware["uuid"]):
    raise SystemExit("counter acceptance has no sealed GPU identity")
rows = csv.reader(raw.read_text(encoding="utf-8", errors="replace").splitlines())
columns = None
values = []
for row in rows:
    if all(field in row for field in ("Kernel Name", "Metric Name", "Metric Value")):
        columns = {field: row.index(field)
                   for field in ("Kernel Name", "Metric Name", "Metric Value")}
        continue
    if columns is None or max(columns.values()) >= len(row):
        continue
    if ("checkpoint_counter_kernel" not in row[columns["Kernel Name"]]
            or row[columns["Metric Name"]] != "sm__cycles_elapsed.avg"):
        continue
    try:
        value = float(row[columns["Metric Value"]].replace(",", ""))
    except ValueError:
        continue
    if math.isfinite(value) and value > 0:
        values.append(value)
text = log.read_text(encoding="utf-8", errors="replace")
if len(values) != 1 or "CHECKPOINT_COUNTER_KERNEL_RESULT=1" not in text:
    raise SystemExit(f"counter acceptance did not produce one valid metric: {values}")
record = {
    "schema": "stwo.replacement-v1.counter-acceptance.v1",
    "pass": True,
    "kernel": "checkpoint_counter_kernel",
    "metric": "sm__cycles_elapsed.avg",
    "metric_value": values[0],
    "gpu_uuid": hardware.get("uuid"),
    "raw_csv_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    "acceptance_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_counter_timing_only() {
  local rc raw log hardware out
  [[ "$REPLACEMENT_SN2_COUNTER_POLICY" == timing-only ]] \
    || { echo "counter waiver requires the timing-only policy" >&2; return 2; }
  if checkpoint_counter_acceptance; then
    echo "NCU counters are available; run the strict sealed recipe" >&2
    return 1
  else
    rc=$?
  fi
  raw="$(checkpoint_artifact counter_acceptance.csv)"
  log="$(checkpoint_artifact counter_acceptance.txt)"
  hardware="$(checkpoint_artifact hardware_identity.json)"
  out="$(checkpoint_artifact counter_acceptance.json)"
  RC="$rc" python3 - "$raw" "$log" "$hardware" "$out" <<'PY'
import csv, hashlib, json, math, os, pathlib, sys

raw, log, hardware_path, out = map(pathlib.Path, sys.argv[1:])
hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
text = log.read_text(encoding="utf-8", errors="replace")
raw_text = raw.read_text(encoding="utf-8", errors="replace") if raw.is_file() else ""
raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None
columns = None
values = []
for row in csv.reader(raw_text.splitlines()):
    if all(field in row for field in ("Kernel Name", "Metric Name", "Metric Value")):
        columns = {field: row.index(field)
                   for field in ("Kernel Name", "Metric Name", "Metric Value")}
        continue
    if columns is None or max(columns.values()) >= len(row):
        continue
    if ("checkpoint_counter_kernel" not in row[columns["Kernel Name"]]
            or row[columns["Metric Name"]] != "sm__cycles_elapsed.avg"):
        continue
    try:
        value = float(row[columns["Metric Value"]].replace(",", ""))
    except ValueError:
        continue
    if math.isfinite(value) and value > 0:
        values.append(value)
unavailable = (
    raw_sha is not None
    and "CHECKPOINT_COUNTER_KERNEL_RESULT=1" in text
    and "ERR_NVGPUCTRPERM" in raw_text
    and not values
)
record = {
    "schema": "stwo.replacement-v1.counter-availability.v1",
    "status": "UNAVAILABLE" if unavailable else "FAIL",
    "pass": False,
    "error_code": "ERR_NVGPUCTRPERM" if unavailable else None,
    "command_return_code": int(os.environ["RC"]),
    "gpu_uuid": hardware.get("uuid"),
    "raw_csv_sha256": raw_sha,
    "acceptance_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
}
out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
if not unavailable:
    raise SystemExit("counter failure was not the explicit NCU permission denial")
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

checkpoint_validate_stage4_native_receipt() {
  STWO_HEAD="$STWO_PARITY_REF_STWO_HEAD" python3 - "$1" <<'PY'
import json, os, re, sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
if (record.get("schema") != "stwo.replacement-stage4-native.v1"
        or record.get("passed") is not True
        or record.get("failure") is not None
        or record.get("git_commit") != os.environ["STWO_HEAD"]
        or record.get("requested_cuda_arch") != "sm_90"
        or record.get("performance_requested") is not False
        or record.get("performance_failure") is not None
        or record.get("performance") != []):
    raise SystemExit(f"invalid replacement Stage-4 native receipt: {record}")
for field in ("executable_blake3", "source_blake3"):
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
        raise SystemExit(f"invalid Stage-4 identity field: {field}")
for field in ("cuda_device", "nvcc_version"):
    if not isinstance(record.get(field), str) or not record[field].strip():
        raise SystemExit(f"missing Stage-4 tool/device identity: {field}")
if (not isinstance(record.get("total_memory_bytes"), int)
        or record["total_memory_bytes"] <= 0
        or not isinstance(record.get("free_memory_before_bytes"), int)
        or not isinstance(record.get("free_memory_after_bytes"), int)
        or not 0 <= record["free_memory_before_bytes"] <= record["total_memory_bytes"]
        or not 0 <= record["free_memory_after_bytes"] <= record["total_memory_bytes"]):
    raise SystemExit("invalid Stage-4 device-memory identity")
fixtures = record.get("fixtures")
if not isinstance(fixtures, list):
    raise SystemExit("Stage-4 fixture list is missing")
by_name = {fixture.get("name"): fixture for fixture in fixtures if isinstance(fixture, dict)}
expected = {
    "staged-packed-quotient-mixed-topology": {
        "cases": 2,
        "production_apis": [
            "quotient_numerator_staged_single_write_plan_with_overflow_capacities",
            "PreparedQuotientNumeratorGraph::prepare_staged_packed_single_write",
        ],
        "checks": ["eager_reference", "legacy_candidate_byte_identity",
                   "captured_graph_mutation", "source_preservation", "guard_preservation"],
        "hashes": ["eager_outputs", "mutated_graph_outputs"],
    },
    "mode-a-domain-cooperative-commit": {
        "cases": 9,
        "production_apis": [
            "CommitProgram::bind",
            "DomainCooperativeProgram::compile_mode_a",
            "DomainCooperativeProgram::bind",
        ],
        "checks": ["raw_prefix_boundary_identity", "eager_reference",
                   "legacy_candidate_byte_identity", "captured_graph_mutation",
                   "source_preservation", "guard_preservation"],
        "hashes": ["raw_prefix_states", "raw_prefix_hashes",
                   "eager_root_and_retained", "mutated_graph_root_and_retained"],
    },
}
if (len(fixtures) != len(expected)
        or any(not isinstance(fixture, dict) for fixture in fixtures)
        or len(by_name) != len(fixtures)
        or set(by_name) != set(expected)):
    raise SystemExit(f"Stage-4 fixture set drifted: {sorted(map(str, by_name))}")
for name, identity in expected.items():
    fixture = by_name[name]
    checks = fixture.get("checks")
    hashes = fixture.get("hashes")
    if (fixture.get("cases") != identity["cases"]
            or fixture.get("production_apis") != identity["production_apis"]
            or not isinstance(fixture.get("arena_bytes"), int)
            or fixture["arena_bytes"] <= 0
            or not isinstance(checks, dict) or set(checks) != set(identity["checks"])
            or any(value is not True for value in checks.values())
            or not isinstance(hashes, dict) or set(hashes) != set(identity["hashes"])
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                   for value in hashes.values())):
        raise SystemExit(f"Stage-4 fixture receipt is incomplete: {name}")
print(json.dumps({"replacement_stage4_native": "PASS",
                  "fixtures": sorted(by_name)}, sort_keys=True))
PY
}

checkpoint_stage4_native() {
  local log out
  checkpoint_reject_ambient_overrides
  log="$(checkpoint_artifact stage4_native.txt)"
  out="$(checkpoint_artifact stage4_native.json)"
  rm -f "$out"
  cd "$STWO"
  if ! env CARGO_TARGET_DIR="$CAIRO/target" \
      STWO_STAGE4_GIT_COMMIT="$STWO_PARITY_REF_STWO_HEAD" \
      STWO_STAGE4_NATIVE_RECEIPT="$out" \
      cargo test --release --locked -p stwo-backend-cuda \
      --test replacement_stage4_native replacement_stage4_native_bytes_match \
      -- --exact --nocapture --test-threads=1 >"$log" 2>&1; then
    cat "$log"
    return 1
  fi
  checkpoint_require_one_test "$log" replacement_stage4_native_bytes_match
  [[ -s "$out" ]] || { echo "replacement Stage-4 native gate emitted no receipt" >&2; return 1; }
  checkpoint_validate_stage4_native_receipt "$out"
  cat "$log"
}

checkpoint_aot_identity() {
  local manifest_sha key_lines raw out key expected_key_count
  local -a keys key_args
  checkpoint_reject_ambient_overrides
  checkpoint_require_aot_manifest_shape || return 1
  manifest_sha="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")"
  [[ "$manifest_sha" == "$CHECKPOINT_AOT_MANIFEST_SHA256" ]] \
    || { echo "AOT manifest SHA-256 drifted" >&2; return 1; }
  key_lines="$(python3 - "$CHECKPOINT_AOT_MANIFEST" <<'PY'
import json, re, sys
entries = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(entries, list) or not entries or any(not isinstance(entry, dict) for entry in entries):
    raise SystemExit("pinned AOT manifest is empty or malformed")
manifest_keys = [entry.get("cache_key") for entry in entries]
if any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key)
       for key in manifest_keys):
    raise SystemExit("pinned AOT manifest contains an invalid cache key")
keys = sorted(set(manifest_keys))
if len(keys) != len(manifest_keys):
    raise SystemExit("pinned AOT manifest contains duplicate cache keys")
print("\n".join(keys))
PY
  )" || return 1
  keys=()
  while IFS= read -r key; do keys+=("$key"); done <<<"$key_lines"
  [[ ${#keys[@]} -eq $CHECKPOINT_AOT_TOTAL ]] \
    || { echo "AOT key extraction expected $CHECKPOINT_AOT_TOTAL keys, got ${#keys[@]}" >&2; return 1; }
  expected_key_count="${#keys[@]}"
  key_args=()
  for key in "${keys[@]}"; do key_args+=(--key "$key"); done
  raw="$(checkpoint_artifact aot_index_raw.json)"
  "$CHECKPOINT_AOT_CHECK" --sm 90 "${key_args[@]}" >"$raw"
  out="$(checkpoint_artifact aot_identity.json)"
  MANIFEST_SHA="$manifest_sha" CHECKER_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
    GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    EXPECTED_KEY_COUNT="$expected_key_count" AOT_WITNESS="$CHECKPOINT_AOT_WITNESS" \
    AOT_ORDINARY="$CHECKPOINT_AOT_ORDINARY_CONSTRAINT" \
    AOT_WAVE="$CHECKPOINT_AOT_COMPOSITION_WAVE" python3 - "$raw" "$out" <<'PY'
import json, os, re, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(os.environ["EXPECTED_KEY_COUNT"])
if (result.get("pass") is not True or result.get("sm") != 90
        or result.get("required_unique_key_count") != expected
        or result.get("embedded_entry_count") != expected
        or result.get("embedded_arch_entry_count") != expected
        or result.get("missing_keys") != []
        or not re.fullmatch(r"(?!0{16})[0-9a-f]{16}", result.get("loaded_manifest_hash", ""))):
    raise SystemExit(f"embedded sm_90 AOT pack identity/coverage failed: {result}")
record = {"schema": "stwo.replacement-v1-sn2.aot-identity.v1", **result,
          "manifest_sha256": os.environ["MANIFEST_SHA"],
          "checker_binary_sha256": os.environ["CHECKER_SHA"],
          "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
          "manifest_entry_count": int(os.environ["EXPECTED_KEY_COUNT"]),
          "manifest_witness_count": int(os.environ["AOT_WITNESS"]),
          "manifest_ordinary_constraint_count": int(os.environ["AOT_ORDINARY"]),
          "manifest_composition_wave_count": int(os.environ["AOT_WAVE"])}
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
    STWO_STAGE4_NATIVE_PERF STWO_STAGE4_NATIVE_PERF_LOGS
    STWO_STAGE4_NATIVE_PERF_WARMUPS STWO_STAGE4_NATIVE_PERF_ITERATIONS
    STWO_STAGE4_GIT_COMMIT STWO_STAGE4_NATIVE_RECEIPT
    STWO_BENCH_REQUIRE_GPU_NATIVE_ARCHITECTURE STWO_BENCH_REQUIRE_GPU_PCS_RUNTIME_MODE
    STWO_BENCH_REQUIRE_PROOF_BYTE_EQUAL STWO_BENCH_REQUIRE_SIMD_REFERENCE_BYTE_EQUAL
    STWO_BENCH_REQUIRE_PROOF_MUTATION_REJECTED GPU_PCS_RUNTIME_MODE
    STWO_CUDA_NVCC STWO_CUDA_NVCC_FLAGS STWO_CUDA_HOST_COMPILER
    NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_CCBIN CUDAHOSTCXX
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
  [[ "$mode" == diagnostic || "$mode" == timing || "$mode" == iteration ]] \
    || { echo "invalid checkpoint mode: $mode" >&2; return 2; }
  if [[ "$mode" == diagnostic ]]; then
    args+=(--diagnostic-allow-slow-graph-submit)
  else
    args+=(--capture-slow-graph-submit)
  fi
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

checkpoint_mark_iteration_non_promotable() {
  python3 - "$1" <<'PY'
import json, os, sys, tempfile

path = sys.argv[1]
record = json.load(open(path, encoding="utf-8"))
validation = record.get("checkpoint_validation") or {}
raw_admissible = record.get("performance_claim_admissible")
graph_gate = record.get("gpu_graph_submit_gap_strict_gate_passed")
if (validation.get("verdict") != "PASS"
        or validation.get("mode") != "iteration"
        or validation.get("formal_promotion_eligible") is not False
        or validation.get("counter_profile_admissible") is not False
        or record.get("iteration_only") is not True
        or record.get("formal_promotion_eligible") is not False
        or not isinstance(raw_admissible, bool)
        or not isinstance(graph_gate, bool)
        or raw_admissible is not graph_gate):
    raise SystemExit("iteration result is not safe to mark non-promotable")
record["iteration_timing_gate_passed"] = raw_admissible
record["performance_claim_admissible"] = False
directory = os.path.dirname(path) or "."
fd, temporary = tempfile.mkstemp(prefix="sn2-iteration.", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps({"iteration_only": True, "formal_promotion_eligible": False,
                  "performance_claim_admissible": False,
                  "iteration_timing_gate_passed": raw_admissible}, sort_keys=True))
PY
}

checkpoint_validate_sn2() {
  local mode="$1" reps="$2" stdout aot proof out
  stdout="$(checkpoint_artifact stdout.txt)"
  aot="$(checkpoint_artifact aot_identity.json)"
  proof="$(checkpoint_artifact proof.bin)"
  out="$(checkpoint_artifact record.json)"
  PACKED_ROWS="$CHECKPOINT_SN2_PACKED_OUTPUT_ROWS" \
    COMPOSITION_PARTS="$CHECKPOINT_SN2_COMPOSITION_PARTS" \
    COMPOSITION_WAVES="$CHECKPOINT_SN2_COMPOSITION_WAVES" \
    COUNTER_POLICY="$REPLACEMENT_SN2_COUNTER_POLICY" \
    SOURCE_RECEIPT="$(checkpoint_artifact source_input_identity.json)" \
    HARDWARE_RECEIPT="$(checkpoint_artifact hardware_identity.json)" \
    BUILD_RECEIPT="$(checkpoint_artifact build_identity.json)" \
    ADAPTED_RECEIPT="$(checkpoint_artifact adapted_input_identity.json)" \
    GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    python3 - "$stdout" "$aot" "$proof" "$CHECKPOINT_SEAL" "$out" "$mode" "$reps" \
    "$CHECKPOINT_ROOT/gpu_benchmarks" <<'PY'
import hashlib, json, math, os, re, sys

raw_path, aot_path, proof_path, seal_path, out_path, mode, reps_text, module_path = sys.argv[1:]
sys.path.insert(0, module_path)
from validate_replacement_v1_reuse import require_resident_reuse

reps = int(reps_text)
counter_policy = os.environ["COUNTER_POLICY"]
counter_profile_admissible = counter_policy == "required"
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
require(r.get("gpu_planned_numerator_schedule") == "staged-packed-single-write", "planned numerator schedule drifted")
require(r.get("gpu_prepared_numerator_schedule") == "staged-packed-single-write", "actual numerator schedule fell back")
require(r.get("gpu_prepared_numerator_eligible_groups") is None, "staged numerator unexpectedly reported eligible groups")
require(r.get("gpu_prepared_numerator_legacy_groups") == 0, "staged numerator executed legacy groups")
require(r.get("gpu_prepared_numerator_packed_output_rows") == int(os.environ["PACKED_ROWS"]), "staged numerator packed-row count drifted")
require(r.get("gpu_composition_part_count") == int(os.environ["COMPOSITION_PARTS"]), "composition part count drifted")
require(r.get("gpu_composition_wave_count") == int(os.environ["COMPOSITION_WAVES"]), "composition wave count drifted")
require(r.get("gpu_composition_replay_launch_mode") == "wave", "composition replay did not use the wave launch path")
require(r.get("gpu_composition_replay_wave_launches") == int(os.environ["COMPOSITION_WAVES"]), "replayed composition wave launches drifted")
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
require(r.get("gpu_policy_composition_launch_mode") == "wave", "replacement-v1 composition launch policy drifted")
require(r.get("gpu_policy_relation_tail_mode") == "segmented", "replacement-v1 relation-tail policy drifted")
require(r.get("gpu_policy_fri_fold_launch_mode") == "per-fold", "replacement-v1 FRI-fold policy drifted")
require(r.get("gpu_policy_witness_feed_launch_mode") == "global-atomics", "replacement-v1 witness-feed policy drifted")
for field in ("gpu_setup_base_migration_copies", "gpu_setup_lookup_host_copies", "gpu_setup_legacy_witness_fallbacks"):
    require(r.get(field) == 0, f"legacy setup/fallback executed: {field}={r.get(field)!r}")
for field in ("gpu_aot_misses", "gpu_aot_runtime_loads", "gpu_aot_runtime_cache_hits", "gpu_aot_strict_rejections"):
    require(r.get(field) == 0, f"JIT/AOT fallback executed: {field}={r.get(field)!r}")
activity = (r.get("gpu_aot_loads"), r.get("gpu_aot_cache_hits"))
# A fully prepared replay may perform no AOT lookup after its per-proof counters reset.
require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in activity),
        "runtime telemetry recorded invalid AOT activity")
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
max_gap_samples = r.get("gpu_graph_submit_gap_ns_max_samples")
total_gap_samples = r.get("gpu_graph_submit_gap_ns_total_samples")
average_gap_samples = r.get("gpu_graph_submit_gap_ns_average_samples")
graph_launch_samples = r.get("gpu_graph_submit_launches_samples")
require(all(isinstance(values, list) and len(values) == reps for values in
            (max_gap_samples, total_gap_samples, average_gap_samples, graph_launch_samples)),
        "graph-submit sample vectors do not match proof repetitions")
for index, (maximum, total, average, launches) in enumerate(zip(
        max_gap_samples, total_gap_samples, average_gap_samples, graph_launch_samples)):
    require(isinstance(maximum, int) and not isinstance(maximum, bool) and maximum >= 0,
            f"invalid graph-submit maximum at repetition {index}")
    require(isinstance(total, int) and not isinstance(total, bool) and total >= maximum,
            f"invalid graph-submit total at repetition {index}")
    require(isinstance(launches, int) and not isinstance(launches, bool) and launches > 1,
            f"graph-submit average needs at least two launches at repetition {index}")
    gap_count = launches - 1
    require(isinstance(average, (int, float)) and not isinstance(average, bool)
            and math.isfinite(average)
            and math.isclose(average, total / gap_count, rel_tol=1e-12, abs_tol=1e-6),
            f"invalid graph-submit average at repetition {index}")
warm_max_gap_samples = max_gap_samples[1:] if reps > 1 else max_gap_samples
claimed_max_gap_ns = max(warm_max_gap_samples)
claimed_graph_gate = claimed_max_gap_ns < 50_000_000
require(r.get("gpu_graph_submit_warm_sample_count") == len(warm_max_gap_samples),
        "graph-submit warm sample count drifted")
require(r.get("gpu_graph_submit_gap_ns_total") == total_gap_samples[-1],
        "final graph-submit total differs from repetition vector")
require(r.get("gpu_graph_launches") == graph_launch_samples[-1],
        "final graph-launch count differs from repetition vector")
require(isinstance(r.get("gpu_max_graph_submit_gap_ms"), (int, float))
        and math.isfinite(r["gpu_max_graph_submit_gap_ms"])
        and math.isclose(r["gpu_max_graph_submit_gap_ms"], claimed_max_gap_ns / 1_000_000,
                         rel_tol=0, abs_tol=1e-9),
        "aggregate graph-submit maximum differs from warm samples")
require(r.get("gpu_graph_submit_gap_strict_gate_passed") is claimed_graph_gate,
        "aggregate graph-submit gate differs from warm samples")
require(r.get("performance_measurement_available") is True, "GPU timing is unavailable")
if mode == "diagnostic":
    require(reps == 2 and r.get("benchmark_diagnostic_mode") is True, "first checkpoint must be reps=2 diagnostic")
    require(r.get("benchmark_diagnostic_reason") == "graph-submit-gap-and-replay-intervals",
            "diagnostic timing scope drifted")
    require(r.get("benchmark_graph_submit_capture_mode") is False, "diagnostic unexpectedly used timing capture mode")
    require(r.get("performance_claim_admissible") is False, "diagnostic timing must not be admissible")
    graph_rows = r.get("gpu_graph_replay_intervals")
    expected_graphs = [
        "ingest-witness-base-commit",
        "interaction-commit",
        "composition-quotient-commit",
        "oods-evaluation",
        *[f"fri-layer-{layer}" for layer in range(9)],
        "oods-queries-decommit-assemble",
    ]
    expected_semantics = [
        "bootstrap-through-base",
        "interaction-pow-and-lookup",
        "interaction-and-composition",
        "composition-and-oods",
        "oods-and-quotient",
        *[f"fri-layer-{layer}" for layer in range(8)],
        "fri-last-layer",
        "query-pow-and-positions",
    ]
    require(isinstance(graph_rows, list) and len(graph_rows) == len(expected_graphs),
            "diagnostic omitted the 14 topology-preserving graph intervals")
    require([row.get("graph_segment") for row in graph_rows] == expected_graphs,
            "diagnostic graph timing order drifted")
    require(all(isinstance(row.get("interval_elapsed_ns"), int) and row["interval_elapsed_ns"] > 0
                and isinstance(row.get("kernel_nodes"), int) and row["kernel_nodes"] > 0
                and isinstance(row.get("transcript_segments"), list)
                for row in graph_rows),
            "diagnostic graph timing rows are incomplete")
    observed_semantics = [semantic for row in graph_rows for semantic in row["transcript_segments"]]
    require(observed_semantics == expected_semantics,
            "14 graph intervals do not cover the 15 semantic transcript segments exactly once")
    require(graph_rows[12]["transcript_segments"] == [],
            "terminal FRI fold unexpectedly owns a transcript boundary")
    graph_total = sum(row["interval_elapsed_ns"] for row in graph_rows)
    require(r.get("gpu_graph_replay_interval_total_ns") == graph_total and graph_total > 0,
            "diagnostic graph timing total is invalid")
    require(r.get("gpu_graph_replay_interval_count") == 14
            and r.get("gpu_graph_replay_semantic_count") == 15,
            "diagnostic graph/semantic timing counts drifted")
    require(r.get("gpu_graph_replay_timing_scope") ==
            "main-stream-start-through-final-graph-marker; includes-host-submit-gaps; excludes-proof-bundle-readback",
            "diagnostic graph timing scope is ambiguous")
    require(sum(row["kernel_nodes"] for row in graph_rows) == r.get("gpu_kernel_launches"),
            "diagnostic timing kernel-node total differs from executed graph telemetry")
    warm_samples = r.get("prove_s_warm_samples_raw")
    require(isinstance(warm_samples, list) and len(warm_samples) == 1
            and isinstance(warm_samples[0], (int, float)) and warm_samples[0] > 0
            and graph_total <= warm_samples[0] * 1_000_000_000,
            "diagnostic graph timeline exceeds the measured warm proof wall")
elif mode in ("timing", "iteration"):
    require(reps >= 6 and r.get("benchmark_diagnostic_mode") is False and r.get("benchmark_diagnostic_reason") is None, f"{mode} run must have diagnostics disabled")
    require(r.get("benchmark_graph_submit_capture_mode") is True, f"{mode} run did not enable soft graph-gap capture")
    require(r.get("gpu_graph_replay_intervals") is None
            and r.get("gpu_graph_replay_interval_total_ns") is None,
            f"{mode} run must remain free of CUDA-event instrumentation")
    require(r.get("performance_claim_admissible") is (r.get("gpu_graph_submit_gap_strict_gate_passed") is True), f"{mode} timing admissibility disagrees with observed graph-submit gate")
    require(r.get("warm_sample_count") == reps - 1, f"{mode} run lacks the expected warm samples")
    for field in ("prove_s_warm_median", "prove_s_warm_p95", "useful_mhz_median", "useful_mhz_at_warm_p95"):
        require(isinstance(r.get(field), (int, float)) and math.isfinite(r[field]) and r[field] > 0, f"invalid timing metric {field}")
    if mode == "iteration":
        require(counter_policy == "timing-only",
                "iteration timing must remain outside the counter-qualified promotion lane")
    else:
        seal = json.load(open(seal_path, encoding="utf-8"))
        expected_seal = ("stwo.replacement-v1-sn2.checkpoint-seal.v3" if counter_profile_admissible
                         else "stwo.replacement-v1-sn2.timing-only-seal.v1")
        require(seal.get("schema") == expected_seal and seal.get("diagnostic_pass") is True,
                "timing seal is not a passing diagnostic for the selected counter policy")
        if counter_profile_admissible:
            require("counter_policy" not in seal and "counter_profile_admissible" not in seal,
                    "strict v3 seal was altered by a counter waiver")
        else:
            require(seal.get("counter_policy") == "timing-only"
                    and seal.get("counter_profile_admissible") is False
                    and seal.get("counter_status") == "UNAVAILABLE",
                    "timing-only seal omitted the counter-profile asterisk")
        require(r["gpu_proof_blake3"] == seal.get("proof_blake3"), "timing proof digest differs from diagnostic")
        proof_sha256 = hashlib.sha256(open(proof_path, "rb").read()).hexdigest()
        require(proof_sha256 == seal.get("proof_dump_sha256"), "timing proof bytes differ from diagnostic")
        shape = seal.get("shape_receipt") or {}
        require(shape == {
            "protocol_key": r["gpu_protocol_key"],
            "topology_digest": r["gpu_shape_executable_topology_digest"],
            "numerator_schedule": r["gpu_prepared_numerator_schedule"],
            "numerator_packed_output_rows": r["gpu_prepared_numerator_packed_output_rows"],
            "composition_part_count": r["gpu_composition_part_count"],
            "composition_wave_count": r["gpu_composition_wave_count"],
        }, "timing shape/numerator receipt differs from diagnostic")
else:
    raise SystemExit(f"unknown validation mode: {mode}")
r["counter_profile_admissible"] = counter_profile_admissible
if mode == "iteration":
    source_receipt = json.load(open(os.environ["SOURCE_RECEIPT"], encoding="utf-8"))
    hardware_receipt = json.load(open(os.environ["HARDWARE_RECEIPT"], encoding="utf-8"))
    build_receipt = json.load(open(os.environ["BUILD_RECEIPT"], encoding="utf-8"))
    adapted_receipt = json.load(open(os.environ["ADAPTED_RECEIPT"], encoding="utf-8"))
    require(source_receipt.get("schema") == "stwo.replacement-v1-sn2.source-input-identity.v1"
            and source_receipt.get("source_policy") == "iteration",
            "iteration source/input receipt is absent or promotable")
    require(hardware_receipt.get("schema") == "stwo.replacement-v1-sn2.hardware-identity.v2"
            and hardware_receipt.get("name") == r.get("gpu"),
            "iteration GPU differs from the hardware receipt")
    require(build_receipt.get("schema") == "stwo.replacement-v1-sn2.build-identity.v1"
            and build_receipt.get("gpu_bench_sha256") == os.environ["GPU_BENCH_SHA"]
            and adapted_receipt.get("adapter_binary_sha256") == os.environ["GPU_BENCH_SHA"]
            and aot.get("gpu_bench_sha256") == os.environ["GPU_BENCH_SHA"],
            "iteration binary identity differs across build, adapter, AOT, and execution")
    r["iteration_only"] = True
    r["formal_promotion_eligible"] = False
    r["iteration_identity"] = {
        "source": source_receipt["source"],
        "inputs": source_receipt["inputs"],
        "hardware": hardware_receipt,
        "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
        "aot_manifest_sha256": aot["manifest_sha256"],
        "adapted_input_sha256": adapted_receipt["sha256"],
        "proof_dump_sha256": hashlib.sha256(open(proof_path, "rb").read()).hexdigest(),
    }
r["checkpoint_validation"] = {"schema": "stwo.replacement-v1-sn2.record-validation.v1",
                              "verdict": "PASS", "mode": mode,
                              "graph_submit_timing_soft": mode == "diagnostic",
                              "counter_profile_admissible": counter_profile_admissible}
if mode == "iteration":
    r["checkpoint_validation"]["formal_promotion_eligible"] = False
with open(out_path, "w", encoding="utf-8") as stream:
    json.dump(r, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps({"replacement_v1_sn2": "PASS", "mode": mode,
                  "proof_blake3": r["gpu_proof_blake3"],
                  "useful_mhz_median": r.get("useful_mhz_median"),
                  "graph_submit_gap_ms": r["gpu_max_graph_submit_gap_ms"],
                  "graph_submit_gate": r.get("gpu_graph_submit_gap_strict_gate_passed")}, sort_keys=True))
PY
  [[ "$mode" != iteration ]] || checkpoint_mark_iteration_non_promotable "$out"
}

checkpoint_seal_diagnostic() {
  local source hardware counter counter_raw counter_log build adapted carry stage4 aot record proof proof_sha run_seal
  checkpoint_reject_ambient_overrides
  source="$(checkpoint_artifact source_input_identity.json)"
  hardware="$(checkpoint_artifact hardware_identity.json)"
  counter="$(checkpoint_artifact counter_acceptance.json)"
  counter_raw="$(checkpoint_artifact counter_acceptance.csv)"
  counter_log="$(checkpoint_artifact counter_acceptance.txt)"
  build="$(checkpoint_artifact build_identity.json)"
  adapted="$(checkpoint_artifact adapted_input_identity.json)"
  carry="$(checkpoint_artifact fp256_carry_oracles.json)"
  stage4="$(checkpoint_artifact stage4_native.json)"
  aot="$(checkpoint_artifact aot_identity.json)"
  record="$(checkpoint_artifact record.json)"
  proof="$(checkpoint_artifact proof.bin)"
  proof_sha="$(awk '{print $1}' "$(checkpoint_artifact proof.sha256.txt)")"
  run_seal="$(checkpoint_artifact seal.json)"
  for path in "$source" "$hardware" "$counter" "$counter_raw" "$counter_log" "$build" "$adapted" "$carry" "$stage4" "$aot" "$record" "$proof"; do
    [[ -s "$path" ]] || { echo "cannot seal missing checkpoint receipt: $path" >&2; return 1; }
  done
  checkpoint_require_hash "$proof_sha" 256 proof_dump_sha256
  SOURCE="$source" HARDWARE="$hardware" COUNTER="$counter" \
    COUNTER_RAW="$counter_raw" COUNTER_LOG="$counter_log" \
    COUNTER_POLICY="$REPLACEMENT_SN2_COUNTER_POLICY" \
    BUILD="$build" ADAPTED="$adapted" CARRY="$carry" \
    STAGE4="$stage4" AOT="$aot" RECORD="$record" PROOF="$proof" PROOF_SHA="$proof_sha" \
    GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    AOT_CHECK_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_CHECK")" \
    AOT_MANIFEST_SHA="$(checkpoint_sha256 "$CHECKPOINT_AOT_MANIFEST")" \
    OUT="$CHECKPOINT_SEAL" python3 - <<'PY' || return $?
import csv, hashlib, json, math, os, re, tempfile
def load(name):
    with open(os.environ[name], encoding="utf-8") as stream: return json.load(stream)
def sha(name):
    return hashlib.sha256(open(os.environ[name], "rb").read()).hexdigest()
source, hardware, counter, build, adapted = map(
    load, ("SOURCE", "HARDWARE", "COUNTER", "BUILD", "ADAPTED"))
carry, stage4, aot, record = map(load, ("CARRY", "STAGE4", "AOT", "RECORD"))
counter_policy = os.environ["COUNTER_POLICY"]
counter_log_text = open(os.environ["COUNTER_LOG"], encoding="utf-8", errors="replace").read()
counter_raw_text = open(os.environ["COUNTER_RAW"], encoding="utf-8", errors="replace").read()
columns = None
counter_metric_values = []
for row in csv.reader(counter_raw_text.splitlines()):
    if all(field in row for field in ("Kernel Name", "Metric Name", "Metric Value")):
        columns = {field: row.index(field)
                   for field in ("Kernel Name", "Metric Name", "Metric Value")}
        continue
    if columns is None or max(columns.values()) >= len(row):
        continue
    if ("checkpoint_counter_kernel" not in row[columns["Kernel Name"]]
            or row[columns["Metric Name"]] != "sm__cycles_elapsed.avg"):
        continue
    try:
        value = float(row[columns["Metric Value"]].replace(",", ""))
    except ValueError:
        continue
    if math.isfinite(value) and value > 0:
        counter_metric_values.append(value)
counter_hashes_match = (
    counter.get("raw_csv_sha256") == sha("COUNTER_RAW")
    and counter.get("acceptance_log_sha256") == sha("COUNTER_LOG")
    and all(re.fullmatch(r"[0-9a-f]{64}", str(counter.get(field, "")))
            for field in ("raw_csv_sha256", "acceptance_log_sha256"))
)
strict_counter = (
    counter.get("schema") == "stwo.replacement-v1.counter-acceptance.v1"
    and counter.get("pass") is True
    and counter.get("kernel") == "checkpoint_counter_kernel"
    and counter.get("metric") == "sm__cycles_elapsed.avg"
    and isinstance(counter.get("metric_value"), (int, float))
    and not isinstance(counter.get("metric_value"), bool)
    and math.isfinite(counter["metric_value"])
    and counter["metric_value"] > 0
    and counter_metric_values == [float(counter["metric_value"])]
    and counter.get("gpu_uuid") == hardware.get("uuid")
    and counter_hashes_match
    and "CHECKPOINT_COUNTER_KERNEL_RESULT=1" in counter_log_text
    and "ERR_NVGPUCTRPERM" not in counter_log_text
    and "ERR_NVGPUCTRPERM" not in counter_raw_text
)
unavailable_counter = (
    counter.get("schema") == "stwo.replacement-v1.counter-availability.v1"
    and counter.get("status") == "UNAVAILABLE"
    and counter.get("pass") is False
    and counter.get("error_code") == "ERR_NVGPUCTRPERM"
    and isinstance(counter.get("command_return_code"), int)
    and not isinstance(counter.get("command_return_code"), bool)
    and counter["command_return_code"] != 0
    and counter.get("gpu_uuid") == hardware.get("uuid")
    and counter_hashes_match
    and "CHECKPOINT_COUNTER_KERNEL_RESULT=1" in counter_log_text
    and "ERR_NVGPUCTRPERM" in counter_log_text
    and "ERR_NVGPUCTRPERM" in counter_raw_text
    and not counter_metric_values
)
counter_is_sealable = strict_counter if counter_policy == "required" else (
    counter_policy == "timing-only" and unavailable_counter
)
if (source.get("schema") != "stwo.replacement-v1-sn2.source-input-identity.v1"
        or hardware.get("schema") != "stwo.replacement-v1-sn2.hardware-identity.v2"
        or not counter_is_sealable
        or build.get("schema") != "stwo.replacement-v1-sn2.build-identity.v1"
        or adapted.get("schema") != "stwo.replacement-v1-sn2.adapted-input-identity.v1"
        or carry.get("pass") is not True
        or stage4.get("schema") != "stwo.replacement-stage4-native.v1"
        or stage4.get("passed") is not True
        or stage4.get("failure") is not None
        or stage4.get("git_commit") != source.get("source", {}).get("stwo", {}).get("head")
        or hardware.get("uuid") not in stage4.get("cuda_device", "")
        or stage4.get("requested_cuda_arch") != "sm_90"
        or stage4.get("performance_requested") is not False
        or record.get("checkpoint_validation", {}).get("verdict") != "PASS"
        or record["checkpoint_validation"].get("mode") != "diagnostic"
        or record["checkpoint_validation"].get("counter_profile_admissible")
            is not (counter_policy == "required")):
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
seal = {"schema": ("stwo.replacement-v1-sn2.checkpoint-seal.v3" if counter_policy == "required"
                   else "stwo.replacement-v1-sn2.timing-only-seal.v1"),
        "diagnostic_pass": True,
        "source": source["source"], "inputs": source["inputs"], "hardware": hardware,
        "adapted_input_sha256": adapted["sha256"],
        "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
        "aot_index_check_sha256": os.environ["AOT_CHECK_SHA"],
        "aot_manifest_sha256": aot["manifest_sha256"],
        "aot_loaded_manifest_hash": aot["loaded_manifest_hash"],
        "proof_dump_sha256": os.environ["PROOF_SHA"],
        "proof_blake3": record["gpu_proof_blake3"],
        "diagnostic_graph_submit_receipt": {
            "max_ns_samples": record["gpu_graph_submit_gap_ns_max_samples"],
            "total_ns_samples": record["gpu_graph_submit_gap_ns_total_samples"],
            "average_ns_samples": record["gpu_graph_submit_gap_ns_average_samples"],
            "graph_launches_samples": record["gpu_graph_submit_launches_samples"],
            "strict_gate_passed": record["gpu_graph_submit_gap_strict_gate_passed"],
        },
        "shape_receipt": {
            "protocol_key": record["gpu_protocol_key"],
            "topology_digest": record["gpu_shape_executable_topology_digest"],
            "numerator_schedule": record["gpu_prepared_numerator_schedule"],
            "numerator_packed_output_rows": record["gpu_prepared_numerator_packed_output_rows"],
            "composition_part_count": record["gpu_composition_part_count"],
            "composition_wave_count": record["gpu_composition_wave_count"],
        },
        "receipts_sha256": {name.lower(): sha(name) for name in
            ("SOURCE", "HARDWARE", "COUNTER", "BUILD", "ADAPTED", "CARRY", "STAGE4", "AOT", "RECORD")}}
if counter_policy == "timing-only":
    seal.update({"counter_policy": counter_policy, "counter_profile_admissible": False,
                 "counter_status": "UNAVAILABLE"})
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
  COUNTER_POLICY="$REPLACEMENT_SN2_COUNTER_POLICY" \
  CURRENT_AOT="$(checkpoint_artifact aot_identity.json)" \
  CURRENT_HARDWARE="$current_hardware" python3 - "$CHECKPOINT_SEAL" <<'PY'
import json, os, sys
seal = json.load(open(sys.argv[1], encoding="utf-8"))
counter_policy = os.environ["COUNTER_POLICY"]
expected_schema = ("stwo.replacement-v1-sn2.checkpoint-seal.v3" if counter_policy == "required"
                   else "stwo.replacement-v1-sn2.timing-only-seal.v1")
if seal.get("schema") != expected_schema or seal.get("diagnostic_pass") is not True:
    raise SystemExit("checkpoint seal is not a passing diagnostic")
if counter_policy == "required":
    if "counter_policy" in seal or "counter_profile_admissible" in seal:
        raise SystemExit("strict v3 checkpoint seal contains a counter waiver")
elif (seal.get("counter_policy") != "timing-only"
      or seal.get("counter_profile_admissible") is not False
      or seal.get("counter_status") != "UNAVAILABLE"):
    raise SystemExit("timing-only checkpoint seal omitted its counter-profile asterisk")
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

checkpoint_write_profile_receipt() {
  local profiler="$1" rc="$2" proof="$3" report="$4" table="$5"
  local stdout="$6" stderr="$7" out="$8"
  PROFILE="$profiler" RC="$rc" GPU_BENCH_SHA="$(checkpoint_sha256 "$CHECKPOINT_GPU_BENCH")" \
    COUNTER_POLICY="$REPLACEMENT_SN2_COUNTER_POLICY" \
    NCU_LAUNCH_COUNT="$CHECKPOINT_NCU_LAUNCH_COUNT" \
    COMPOSITION_PARTS="$CHECKPOINT_SN2_COMPOSITION_PARTS" \
    COMPOSITION_WAVES="$CHECKPOINT_SN2_COMPOSITION_WAVES" \
    python3 - "$proof" "$report" "$table" "$stdout" "$stderr" "$CHECKPOINT_SEAL" "$out" <<'PY'
import csv, hashlib, json, os, pathlib, sys

proof, report, table, stdout, stderr, seal_path, out = map(pathlib.Path, sys.argv[1:])
profile = os.environ["PROFILE"]
counter_policy = os.environ["COUNTER_POLICY"]
counter_profile_admissible = counter_policy == "required"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

reasons = []
try:
    rc = int(os.environ["RC"])
except ValueError:
    rc = -1
    reasons.append("invalid profiler return code")
if rc != 0:
    reasons.append(f"{profile} exited with status {rc}")
for label, path in (("profiler report", report), ("imported counter table", table),
                    ("profile proof", proof)):
    if not path.is_file() or path.stat().st_size == 0:
        reasons.append(f"missing {label}")
try:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    seal = {}
    reasons.append(f"unreadable diagnostic seal: {error}")
expected_schema = ("stwo.replacement-v1-sn2.checkpoint-seal.v3" if counter_profile_admissible
                   else "stwo.replacement-v1-sn2.timing-only-seal.v1")
if seal.get("schema") != expected_schema or seal.get("diagnostic_pass") is not True:
    reasons.append("profile did not consume the selected policy's passing diagnostic seal")
elif counter_profile_admissible:
    if "counter_policy" in seal or "counter_profile_admissible" in seal:
        reasons.append("strict v3 diagnostic seal contains a counter waiver")
elif (seal.get("counter_policy") != "timing-only"
      or seal.get("counter_profile_admissible") is not False
      or seal.get("counter_status") != "UNAVAILABLE"):
    reasons.append("timing-only diagnostic seal omitted its counter-profile asterisk")
if seal.get("gpu_bench_sha256") != os.environ["GPU_BENCH_SHA"]:
    reasons.append("profile binary differs from sealed diagnostic")
proof_sha = digest(proof)
if proof_sha is not None and proof_sha != seal.get("proof_dump_sha256"):
    reasons.append("profile proof bytes differ from sealed diagnostic")
records = []
if stdout.is_file():
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(value, dict) and value.get("program") == "SN_PIE_2.zip"
                and value.get("backend") == "cuda"):
            records.append(value)
if len(records) != 1:
    reasons.append(f"expected one profiled SN2 record, got {len(records)}")
else:
    record = records[0]
    if (record.get("gpu_proof_blake3") != seal.get("proof_blake3")
            or record.get("engine") != "gpu-native"
            or record.get("gpu_resident_backend") != "replacement-v1"
            or record.get("gpu_pcs_runtime_mode") != "ArenaGraph"
            or record.get("gpu_aot_provenance_gate_passed") is not True
            or record.get("gpu_prepared_numerator_schedule") != "staged-packed-single-write"
            or record.get("gpu_composition_part_count") != int(os.environ["COMPOSITION_PARTS"])
            or record.get("gpu_composition_wave_count") != int(os.environ["COMPOSITION_WAVES"])
            or record.get("verified_reps") != 2
            or record.get("proof_byte_equal") is not True
            or record.get("simd_reference_byte_equal") is not True
            or record.get("proof_mutation_rejected") is not True):
        reasons.append("profiled execution did not reproduce the sealed correctness identity")
ncu_topology = None
if profile == "ncu" and table.is_file():
    launches = {}
    columns = None
    for row in csv.reader(table.read_text(encoding="utf-8", errors="replace").splitlines()):
        if "ID" in row and "Kernel Name" in row:
            columns = {field: row.index(field) for field in ("ID", "Kernel Name")}
            columns["Process ID"] = row.index("Process ID") if "Process ID" in row else None
            continue
        if columns is None or max(index for index in columns.values() if index is not None) >= len(row):
            continue
        kernel = row[columns["Kernel Name"]]
        if ("stwo_composition_wave_" not in kernel
                and "stwo_quotient_numerator_packed_single_write_kernel" not in kernel):
            continue
        launch_id = row[columns["ID"]]
        process_id = row[columns["Process ID"]] if columns["Process ID"] is not None else ""
        if not launch_id:
            continue
        key = (process_id, launch_id)
        if key in launches and launches[key] != kernel:
            reasons.append(f"NCU launch identity mapped to multiple kernels: {key}")
        launches[key] = kernel
    kernels = list(launches.values())
    waves = [kernel for kernel in kernels if "stwo_composition_wave_" in kernel]
    numerator = [kernel for kernel in kernels
                 if "stwo_quotient_numerator_packed_single_write_kernel" in kernel]
    ncu_topology = {
        "selected_launch_count": len(kernels),
        "composition_wave_launch_count": len(waves),
        "distinct_composition_wave_kernel_count": len(set(waves)),
        "packed_numerator_launch_count": len(numerator),
    }
    expected_launches = int(os.environ["NCU_LAUNCH_COUNT"])
    expected_waves = int(os.environ["COMPOSITION_WAVES"])
    if (len(kernels) != expected_launches or len(waves) != expected_waves
            or len(set(waves)) != expected_waves or len(numerator) != 1):
        reasons.append(f"NCU selected-kernel topology drifted: {ncu_topology}")
record = {
    "schema": "stwo.replacement-v1-sn2.profile-receipt.v1",
    "profiler": profile,
    "status": "PASS" if not reasons else "FAIL",
    "counter_profile_admissible": counter_profile_admissible,
    "soft_failure_reasons": reasons,
    "command_return_code": rc,
    "gpu_bench_sha256": os.environ["GPU_BENCH_SHA"],
    "diagnostic_seal_sha256": digest(seal_path),
    "proof_sha256": proof_sha,
    "report_bytes": report.stat().st_size if report.is_file() else 0,
    "report_sha256": digest(report),
    "table_bytes": table.stat().st_size if table.is_file() else 0,
    "table_sha256": digest(table),
    "stdout_sha256": digest(stdout),
    "stderr_sha256": digest(stderr),
    "ncu_launch_topology": ncu_topology,
}
out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY
}

checkpoint_profile_receipt_valid() {
  local profiler="$1" counter_admissible="$2" base report
  base="$(checkpoint_artifact "${profiler}_profile")"
  if [[ "$profiler" == ncu ]]; then
    report="$base.ncu-rep"
  else
    [[ "$profiler" == nsys ]] || return 2
    report="$base.nsys-rep"
  fi
  python3 - "$profiler" "$counter_admissible" \
    "$(checkpoint_artifact "${profiler}_profile.json")" "$CHECKPOINT_SEAL" \
    "$CHECKPOINT_GPU_BENCH" "$(checkpoint_artifact "${profiler}_profile.proof.bin")" \
    "$report" "$(checkpoint_artifact "${profiler}_profile.csv")" \
    "$(checkpoint_artifact "${profiler}_profile.stdout.txt")" \
    "$(checkpoint_artifact "${profiler}_profile.stderr.txt")" <<'PY'
import hashlib, json, pathlib, re, sys

profiler, admissible_text = sys.argv[1:3]
receipt_path, seal_path, binary, proof, report, table, stdout, stderr = map(
    pathlib.Path, sys.argv[3:])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
seal = json.loads(seal_path.read_text(encoding="utf-8"))
admissible = admissible_text == "true"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def hash64(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

paths = (binary, proof, report, table, stdout, stderr, seal_path)
if any(not path.is_file() for path in paths) or any(
        path.stat().st_size == 0 for path in (binary, proof, report, table, stdout, seal_path)):
    raise SystemExit("profile receipt artifacts are missing or empty")
expected_topology = ({
    "selected_launch_count": 19,
    "composition_wave_launch_count": 18,
    "distinct_composition_wave_kernel_count": 18,
    "packed_numerator_launch_count": 1,
} if profiler == "ncu" else None)
expected_seal = ("stwo.replacement-v1-sn2.checkpoint-seal.v3" if admissible
                 else "stwo.replacement-v1-sn2.timing-only-seal.v1")
checks = (
    receipt.get("schema") == "stwo.replacement-v1-sn2.profile-receipt.v1",
    receipt.get("profiler") == profiler,
    receipt.get("status") == "PASS",
    receipt.get("counter_profile_admissible") is admissible,
    isinstance(receipt.get("command_return_code"), int)
        and not isinstance(receipt.get("command_return_code"), bool)
        and receipt["command_return_code"] == 0,
    receipt.get("gpu_bench_sha256") == digest(binary) == seal.get("gpu_bench_sha256"),
    receipt.get("diagnostic_seal_sha256") == digest(seal_path),
    receipt.get("proof_sha256") == digest(proof) == seal.get("proof_dump_sha256"),
    receipt.get("report_bytes") == report.stat().st_size,
    receipt.get("table_bytes") == table.stat().st_size,
    receipt.get("report_sha256") == digest(report),
    receipt.get("table_sha256") == digest(table),
    receipt.get("stdout_sha256") == digest(stdout),
    receipt.get("stderr_sha256") == digest(stderr),
    receipt.get("ncu_launch_topology") == expected_topology,
    seal.get("schema") == expected_seal and seal.get("diagnostic_pass") is True,
    all(hash64(receipt.get(field)) for field in (
        "gpu_bench_sha256", "diagnostic_seal_sha256", "proof_sha256",
        "report_sha256", "table_sha256", "stdout_sha256", "stderr_sha256")),
)
if not all(checks):
    raise SystemExit("profile receipt does not bind the sealed execution artifacts")
if admissible:
    if "counter_policy" in seal or "counter_profile_admissible" in seal:
        raise SystemExit("strict profile consumed a waived seal")
elif (seal.get("counter_policy") != "timing-only"
      or seal.get("counter_profile_admissible") is not False
      or seal.get("counter_status") != "UNAVAILABLE"):
    raise SystemExit("timing-only profile consumed an invalid waiver")
PY
}

checkpoint_nsys_profile() {
  local base report table stdout stderr proof out rc table_rc nsys_bin
  checkpoint_reject_ambient_overrides
  base="$(checkpoint_artifact nsys_profile)"
  report="$base.nsys-rep"
  table="$(checkpoint_artifact nsys_profile.csv)"
  stdout="$(checkpoint_artifact nsys_profile.stdout.txt)"
  stderr="$(checkpoint_artifact nsys_profile.stderr.txt)"
  proof="$(checkpoint_artifact nsys_profile.proof.bin)"
  out="$(checkpoint_artifact nsys_profile.json)"
  nsys_bin="$(command -v nsys || true)"
  [[ -n "$nsys_bin" ]] || nsys_bin="$CHECKPOINT_NSYS_FALLBACK"
  rm -f "$report" "$table" "$stdout" "$stderr" "$proof" "$out"
  set +e
  "$nsys_bin" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=graph --stats=false --wait=all \
    --force-overwrite=true --output="$base" \
    env STWO_BENCH_TRACE=json STWO_DUMP_PROOF="$proof" \
    "$CHECKPOINT_GPU_BENCH" --pie "$CHECKPOINT_PIE" --backend cuda --engine gpu-native \
    --resident-backend replacement-v1 --require-gpu-native-architecture \
    --require-gpu-pcs-runtime-mode arena-graph --reuse-input --reps 2 \
    --require-proof-byte-equal --require-simd-reference-byte-equal \
    --require-proof-mutation-rejected --diagnostic-allow-slow-graph-submit \
    >"$stdout" 2>"$stderr"
  rc=$?
  if [[ -s "$report" ]]; then
    "$nsys_bin" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv "$report" \
      >"$table" 2>>"$stderr"
    table_rc=$?
    [[ $rc -ne 0 ]] || rc=$table_rc
  fi
  set -e
  checkpoint_write_profile_receipt nsys "$rc" "$proof" "$report" "$table" \
    "$stdout" "$stderr" "$out"
  return 0
}

checkpoint_ncu_profile() {
  local base report table stdout stderr proof out rc table_rc
  checkpoint_reject_ambient_overrides
  base="$(checkpoint_artifact ncu_profile)"
  report="$base.ncu-rep"
  table="$(checkpoint_artifact ncu_profile.csv)"
  stdout="$(checkpoint_artifact ncu_profile.stdout.txt)"
  stderr="$(checkpoint_artifact ncu_profile.stderr.txt)"
  proof="$(checkpoint_artifact ncu_profile.proof.bin)"
  out="$(checkpoint_artifact ncu_profile.json)"
  rm -f "$report" "$table" "$stdout" "$stderr" "$proof" "$out"
  set +e
  ncu --target-processes all --kernel-name "$CHECKPOINT_NCU_KERNEL_REGEX" \
    --launch-count "$CHECKPOINT_NCU_LAUNCH_COUNT" --set basic -f -o "$base" \
    env STWO_BENCH_TRACE=json STWO_DUMP_PROOF="$proof" \
    "$CHECKPOINT_GPU_BENCH" --pie "$CHECKPOINT_PIE" --backend cuda --engine gpu-native \
    --resident-backend replacement-v1 --require-gpu-native-architecture \
    --require-gpu-pcs-runtime-mode arena-graph --reuse-input --reps 2 \
    --require-proof-byte-equal --require-simd-reference-byte-equal \
    --require-proof-mutation-rejected --diagnostic-allow-slow-graph-submit \
    >"$stdout" 2>"$stderr"
  rc=$?
  if [[ -s "$report" ]]; then
    ncu --import "$report" --page raw --csv >"$table" 2>>"$stderr"
    table_rc=$?
    [[ $rc -ne 0 ]] || rc=$table_rc
  fi
  set -e
  checkpoint_write_profile_receipt ncu "$rc" "$proof" "$report" "$table" \
    "$stdout" "$stderr" "$out"
  return 0
}

checkpoint_assess_sn2_promotion() {
  local timing nsys ncu out nsys_valid=false ncu_valid=false
  timing="$(checkpoint_artifact record.json)"
  nsys="$(checkpoint_artifact nsys_profile.json)"
  ncu="$(checkpoint_artifact ncu_profile.json)"
  out="$(checkpoint_artifact promotion.json)"
  if checkpoint_profile_receipt_valid nsys true 2>/dev/null; then nsys_valid=true; fi
  if checkpoint_profile_receipt_valid ncu true 2>/dev/null; then ncu_valid=true; fi
  HOST_LIMIT="$CHECKPOINT_PROMOTION_HOST_PREPARATION_NS" \
    MHZ_FLOOR="$CHECKPOINT_PROMOTION_USEFUL_MHZ" \
    NSYS_VALID="$nsys_valid" NCU_VALID="$ncu_valid" \
    python3 - "$timing" "$nsys" "$ncu" "$out" <<'PY'
import hashlib, json, os, sys

evidence_paths = sys.argv[1:4]
timing, nsys, ncu = (json.load(open(path, encoding="utf-8")) for path in evidence_paths)
host_limit = int(os.environ["HOST_LIMIT"])
mhz_floor = float(os.environ["MHZ_FLOOR"])
checks = {
    "hard_checkpoint_validation_passed": timing.get("checkpoint_validation", {}).get("verdict") == "PASS",
    "counter_profile_admissible": timing.get("counter_profile_admissible") is True,
    "performance_claim_admissible": timing.get("performance_claim_admissible") is True,
    "graph_submit_gap_strict": timing.get("gpu_graph_submit_gap_strict_gate_passed") is True,
    "host_preparation_within_budget": isinstance(timing.get("gpu_host_preparation_total_ns"), int)
        and not isinstance(timing.get("gpu_host_preparation_total_ns"), bool)
        and timing["gpu_host_preparation_total_ns"] <= host_limit,
    "useful_mhz_at_or_above_floor": isinstance(timing.get("useful_mhz_median"), (int, float))
        and not isinstance(timing.get("useful_mhz_median"), bool)
        and timing["useful_mhz_median"] >= mhz_floor,
    "nsys_profile_passed": os.environ["NSYS_VALID"] == "true",
    "ncu_profile_passed": os.environ["NCU_VALID"] == "true",
}
failed = [name for name, passed in checks.items() if not passed]
record = {
    "schema": "stwo.replacement-v1-sn2.promotion-verdict.v1",
    "verdict": "PASS" if not failed else "FAIL",
    "soft_failure": bool(failed),
    "failed_checks": failed,
    "checks": checks,
    "thresholds": {
        "host_preparation_total_ns_max": host_limit,
        "useful_mhz_median_min": mhz_floor,
    },
    "measurements": {
        "gpu_host_preparation_total_ns": timing.get("gpu_host_preparation_total_ns"),
        "useful_mhz_median": timing.get("useful_mhz_median"),
        "useful_mhz_at_warm_p95": timing.get("useful_mhz_at_warm_p95"),
        "gpu_max_graph_submit_gap_ms": timing.get("gpu_max_graph_submit_gap_ms"),
    },
    "profile_status": {"nsys": nsys.get("status"), "ncu": ncu.get("status")},
    "evidence_sha256": {
        name: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for name, path in zip(("timing_record", "nsys_receipt", "ncu_receipt"), evidence_paths)
    },
}
with open(sys.argv[4], "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
  return 0
}

checkpoint_assess_sn2_timing_only() {
  local timing nsys out nsys_valid=false
  [[ "$REPLACEMENT_SN2_COUNTER_POLICY" == timing-only ]] \
    || { echo "timing-only verdict requires the timing-only counter policy" >&2; return 2; }
  timing="$(checkpoint_artifact record.json)"
  nsys="$(checkpoint_artifact nsys_profile.json)"
  out="$(checkpoint_artifact timing_only_verdict.json)"
  if checkpoint_profile_receipt_valid nsys false 2>/dev/null; then nsys_valid=true; fi
  HOST_LIMIT="$CHECKPOINT_PROMOTION_HOST_PREPARATION_NS" \
    MHZ_FLOOR="$CHECKPOINT_PROMOTION_USEFUL_MHZ" \
    NSYS_VALID="$nsys_valid" \
    python3 - "$timing" "$nsys" "$CHECKPOINT_SEAL" "$out" <<'PY'
import hashlib, json, math, os, sys

timing_path, nsys_path, seal_path, out_path = sys.argv[1:]
timing, nsys, seal = (json.load(open(path, encoding="utf-8"))
                      for path in (timing_path, nsys_path, seal_path))
host_limit = int(os.environ["HOST_LIMIT"])
mhz_floor = float(os.environ["MHZ_FLOOR"])

def positive_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)

completion_checks = {
    "hard_checkpoint_validation_passed":
        timing.get("checkpoint_validation", {}).get("verdict") == "PASS",
    "timing_measurement_available":
        timing.get("performance_measurement_available") is True
        and positive_number(timing.get("useful_mhz_median"))
        and positive_number(timing.get("prove_s_warm_median")),
    "counter_denial_sealed":
        seal.get("schema") == "stwo.replacement-v1-sn2.timing-only-seal.v1"
        and seal.get("counter_policy") == "timing-only"
        and seal.get("counter_profile_admissible") is False
        and seal.get("counter_status") == "UNAVAILABLE",
    "nsys_profile_passed": os.environ["NSYS_VALID"] == "true",
}
promotion_target_checks = {
    "graph_submit_gap_strict": timing.get("gpu_graph_submit_gap_strict_gate_passed") is True,
    "host_preparation_within_budget":
        isinstance(timing.get("gpu_host_preparation_total_ns"), int)
        and not isinstance(timing.get("gpu_host_preparation_total_ns"), bool)
        and timing["gpu_host_preparation_total_ns"] <= host_limit,
    "useful_mhz_at_or_above_floor":
        positive_number(timing.get("useful_mhz_median"))
        and timing["useful_mhz_median"] >= mhz_floor,
}
failed = [name for name, passed in completion_checks.items() if not passed]
record = {
    "schema": "stwo.replacement-v1-sn2.timing-only-verdict.v1",
    "verdict": "TIMING_ONLY" if not failed else "INCOMPLETE",
    "formal_promotion_eligible": False,
    "counter_profile_admissible": False,
    "failed_completion_checks": failed,
    "completion_checks": completion_checks,
    "promotion_target_checks": promotion_target_checks,
    "failed_promotion_targets": [name for name, passed in promotion_target_checks.items()
                                 if not passed],
    "thresholds": {"host_preparation_total_ns_max": host_limit,
                   "useful_mhz_median_min": mhz_floor},
    "measurements": {
        "gpu_host_preparation_total_ns": timing.get("gpu_host_preparation_total_ns"),
        "prove_s_warm_median": timing.get("prove_s_warm_median"),
        "prove_s_warm_p95": timing.get("prove_s_warm_p95"),
        "useful_mhz_median": timing.get("useful_mhz_median"),
        "useful_mhz_at_warm_p95": timing.get("useful_mhz_at_warm_p95"),
        "gpu_max_graph_submit_gap_ms": timing.get("gpu_max_graph_submit_gap_ms"),
    },
    "profile_status": {"nsys": nsys.get("status"),
                       "ncu": "OMITTED_COUNTER_UNAVAILABLE"},
    "evidence_sha256": {
        name: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for name, path in (("timing_record", timing_path), ("nsys_receipt", nsys_path),
                           ("timing_only_seal", seal_path))
    },
}
with open(out_path, "w", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True)
    stream.write("\n")
print(json.dumps(record, sort_keys=True))
PY
  return 0
}
