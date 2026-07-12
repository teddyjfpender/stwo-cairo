#!/usr/bin/env bash
# Fail-closed post-admission qualification and fixed-statement measurement round.
set -euo pipefail

[[ -z "${UNIVERSAL_ENV+x}" && -z "${SN2_HEADLINE_ENV+x}" ]] || {
  echo "UNIVERSAL_ENV and SN2_HEADLINE_ENV are fixed by the release harness" >&2
  exit 2
}
[[ -z "${GPU_PCS_RUNTIME_MODE+x}" || "$GPU_PCS_RUNTIME_MODE" == "arena-graph" ]] \
  || { echo "qualification runtime is fixed to arena-graph" >&2; exit 2; }
export GPU_PCS_RUNTIME_MODE=arena-graph
# Release-owned subprocess state: ambient qualification/reuse controls must not
# be able to replace the first counted native suite with a prior artifact.
unset BENCH_ENV QUALIFICATION_PROBE QUALIFICATION_ARTIFACT BENCH_PROOF_HASHES \
  REUSE_SOUNDNESS_GATE ARCHITECTURE_SOUNDNESS_GATE LOCAL_PREFLIGHT_ADMISSION \
  REQUIRE_NCU_PROFILE NCU_LAUNCH_COUNT \
  FAKE_REMOTE_TARGET_MISMATCH FAKE_SOUNDNESS_HEAD_MISMATCH \
  FAKE_MISSING_QUALIFICATION_METRICS FAKE_STALL

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="$(cd -P "${CAIRO_LOCAL:-${SCRIPT_DIR}/../..}" && pwd)"
STWO_LOCAL="$(cd -P "${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}" && pwd)"
CARGO_STWO_LOCAL="$(cd -P "$CAIRO_LOCAL/stwo_cairo_prover/../../stwo" && pwd)"
[[ "$STWO_LOCAL" == "$CARGO_STWO_LOCAL" ]] || {
  echo "STWO_LOCAL must be the backend resolved by stwo_cairo_prover/Cargo.toml: $CARGO_STWO_LOCAL" >&2
  exit 2
}
export CAIRO_LOCAL STWO_LOCAL
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/qualification_${STAMP}}"
UNIVERSAL_ENV="STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 STWO_CUDA_B2N_STAGE_FUSED=1 STWO_CUDA_RETAINED_LDE_BUDGET_BYTES=4563402752"
SN2_HEADLINE_ENV="STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 STWO_CUDA_COMPOSITION_DIRECT_RETENTION=1 STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS=1 STWO_CUDA_B2N_STAGE_FUSED=1 STWO_CUDA_RETAINED_LDE_BUDGET_BYTES=29469326848"
PREFLIGHT_VRAM_GIB=76
PREFLIGHT_VRAM_BYTES=81604378624
SN_PIE_SOURCE_DIR="${SN_PIE_SOURCE_DIR:-}"
BOOTLOADER_JSON_SOURCE="${BOOTLOADER_JSON_SOURCE:-}"
ADAPTED_INPUT_DIR="$ROUND_DIR/adapted_inputs"
ADAPTED_INPUT_MANIFEST="$ROUND_DIR/adapted_sha256s"
PINNED_ADAPTED_INPUT_MANIFEST="$CAIRO_LOCAL/gpu_benchmarks/pie/ADAPTED_SHA256SUMS"
PREFLIGHT_BIN="$CAIRO_LOCAL/stwo_cairo_prover/target/debug/arena_preflight"
GPU_BENCH_BIN="$CAIRO_LOCAL/stwo_cairo_prover/target/debug/gpu_bench"
INPUT_MANIFEST="$CAIRO_LOCAL/gpu_benchmarks/pie/SHA256SUMS"
ARCHITECTURE_CHECK="$CAIRO_LOCAL/gpu_benchmarks/validate_architecture_record.py"
export GATE_PIE="/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_2.zip"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXPECTED_POD_GPU="DRY-RUN-GPU"
else
  EXPECTED_POD_GPU="NVIDIA H100 80GB HBM3"
fi
export EXPECTED_POD_GPU

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum
  else shasum -a 256
  fi
}
source_hash() {
  local repo="$1"
  (
    git -C "$repo" diff --binary HEAD -- . ':(exclude)gpu_benchmarks/loop/results' || exit 1
    git -C "$repo" ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' path; do
        [[ "$path" == gpu_benchmarks/loop/results/* ]] && continue
        if [[ -L "$repo/$path" ]]; then
          link_hash="$(readlink -n "$repo/$path" | sha256_stream | cut -d' ' -f1)" || exit 1
          printf 'untracked-symlink\0%s\0%s\0' "$path" "$link_hash"
        elif [[ -f "$repo/$path" ]]; then
          file_kind=regular
          [[ -x "$repo/$path" ]] && file_kind=executable
          file_hash="$(sha256_file "$repo/$path")" || exit 1
          printf 'untracked-%s\0%s\0%s\0' "$file_kind" "$path" "$file_hash"
        else
          echo "unsupported untracked source path: $repo/$path" >&2
          exit 1
        fi
      done
  ) | sha256_stream | cut -d' ' -f1
}

STWO_HEAD="$(git -C "$STWO_LOCAL" rev-parse HEAD)"
CAIRO_HEAD="$(git -C "$CAIRO_LOCAL" rev-parse HEAD)"
STWO_HASH="$(source_hash "$STWO_LOCAL")"
CAIRO_HASH="$(source_hash "$CAIRO_LOCAL")"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -z "$(git -C "$STWO_LOCAL" status --porcelain)" ]] \
    || { echo "release qualification requires a clean STWO_LOCAL" >&2; exit 1; }
  [[ -z "$(git -C "$CAIRO_LOCAL" status --porcelain -- . ':(exclude)gpu_benchmarks/loop/results')" ]] \
    || { echo "release qualification requires a clean CAIRO_LOCAL" >&2; exit 1; }
fi

if [[ -e "$ROUND_DIR" ]]; then
  [[ -d "$ROUND_DIR" && -z "$(find "$ROUND_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || { echo "qualification output directory must be absent or empty: $ROUND_DIR" >&2; exit 1; }
else
  mkdir -p "$ROUND_DIR"
fi
for artifact in qualification.json local_admission.json bench.jsonl sn2_ab.jsonl universal.env sn2_headline.env \
  adapted_inputs adapted_sha256s adapter_reproduction_build.log \
  preflight_flags_off_SN2.json \
  preflight_universal_SN1.json preflight_universal_SN2.json \
  preflight_universal_SN3.json preflight_universal_SN4.json \
  preflight_sn2_headline.json; do
  [[ ! -e "$ROUND_DIR/$artifact" ]] \
    || { echo "qualification output already exists: $ROUND_DIR/$artifact" >&2; exit 1; }
done
mkdir -p "$ADAPTED_INPUT_DIR"
export RESULTS_DIR="$ROUND_DIR"
export LEDGER="$ROUND_DIR/bench.jsonl"
export PERF_LEDGER="$ROUND_DIR/sn2_ab.jsonl"
printf '%s\n' "$UNIVERSAL_ENV" > "$ROUND_DIR/universal.env"
printf '%s\n' "$SN2_HEADLINE_ENV" > "$ROUND_DIR/sn2_headline.env"

STATUS="failed"
write_failure_manifest() {
  [[ "$STATUS" == "passed" ]] && return
  STATUS_VALUE="$STATUS" UNIVERSAL_VALUE="$UNIVERSAL_ENV" \
  HEADLINE_VALUE="$SN2_HEADLINE_ENV" python3 - "$ROUND_DIR/qualification.json" <<'PY'
import json, os, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"schema": "stwo.qualification-round.v4", "status": os.environ["STATUS_VALUE"],
               "profiles": {
                   "universal_sn1_sn4": {"env": os.environ["UNIVERSAL_VALUE"]},
                   "sn2_headline": {"env": os.environ["HEADLINE_VALUE"]},
               }}, stream, sort_keys=True)
    stream.write("\n")
PY
}
trap write_failure_manifest EXIT

# Exact host-only capacity admission must pass before any pod is resolved,
# synced, built, or billed. The current source regenerates every adapted input
# from the manifest-pinned raw PIEs and bootloader; those exact bytes are then
# hashed into the final artifact and used by arena preflight.
[[ -n "$SN_PIE_SOURCE_DIR" && -d "$SN_PIE_SOURCE_DIR" ]] \
  || { echo "set SN_PIE_SOURCE_DIR to the four raw SN_PIE_*.zip files" >&2; exit 1; }
[[ -n "$BOOTLOADER_JSON_SOURCE" && -f "$BOOTLOADER_JSON_SOURCE" ]] \
  || { echo "set BOOTLOADER_JSON_SOURCE to simple_bootloader_compiled.json" >&2; exit 1; }
[[ -f "$PINNED_ADAPTED_INPUT_MANIFEST" ]] \
  || { echo "missing pinned adapted-input manifest: $PINNED_ADAPTED_INPUT_MANIFEST" >&2; exit 1; }

verify_source_inputs() {
  local path name expected actual
  for path in "$SN_PIE_SOURCE_DIR"/SN_PIE_{1,2,3,4}.zip "$BOOTLOADER_JSON_SOURCE"; do
    [[ -f "$path" ]] || { echo "missing source input: $path" >&2; return 1; }
    name="$(basename "$path")"
    expected="$(awk -v name="$name" '$2 == name { print $1 }' "$INPUT_MANIFEST")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
      || { echo "missing or invalid pinned hash for $name" >&2; return 1; }
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] \
      || { echo "source-input SHA-256 mismatch for $name" >&2; return 1; }
  done
}
verify_source_inputs

verify_adapted_inputs() {
  local pie name expected actual
  for pie in 1 2 3 4; do
    name="SN_PIE_${pie}.adapted.bin"
    expected="$(awk -v name="$name" '$2 == name { print $1 }' "$PINNED_ADAPTED_INPUT_MANIFEST")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
      || { echo "missing or invalid pinned hash for $name" >&2; return 1; }
    actual="$(sha256_file "$ADAPTED_INPUT_DIR/$name")"
    [[ "$actual" == "$expected" ]] \
      || { echo "adapted-input SHA-256 mismatch for $name" >&2; return 1; }
  done
}

(cd "$CAIRO_LOCAL/stwo_cairo_prover" &&
  RUST_MIN_STACK=33554432 CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=1 \
  CARGO_INCREMENTAL=0 RUSTFLAGS='-C debuginfo=0' \
  CARGO_TARGET_DIR="$CAIRO_LOCAL/stwo_cairo_prover/target" \
  cargo build -p stwo-cairo-gpu-prover --bin arena_preflight --features emit-tools) \
  > "$ROUND_DIR/arena_preflight_build.log" 2>&1

(cd "$CAIRO_LOCAL/stwo_cairo_prover" &&
  RUST_MIN_STACK=33554432 CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=1 \
  CARGO_INCREMENTAL=0 RUSTFLAGS='-C debuginfo=0' \
  CARGO_TARGET_DIR="$CAIRO_LOCAL/stwo_cairo_prover/target" \
  STWO_BOOTLOADER_JSON="$BOOTLOADER_JSON_SOURCE" \
  cargo build -p stwo-cairo-gpu-prover --bin gpu_bench --features pie-bench) \
  > "$ROUND_DIR/adapter_reproduction_build.log" 2>&1

# Run the current bootloader+adapter over every pinned raw PIE. The generated
# bytes are the exact preflight inputs. One process is used at a time so each
# large in-memory adapter state is released before the next PIE is loaded.
: > "$ADAPTED_INPUT_MANIFEST"
for pie in 1 2 3 4; do
  generated="$ADAPTED_INPUT_DIR/SN_PIE_${pie}.adapted.bin"
  (
    while IFS='=' read -r stwo_name _; do
      case "$stwo_name" in STWO_*) unset "$stwo_name" ;; esac
    done < <(env)
    export STWO_BOOTLOADER_JSON="$BOOTLOADER_JSON_SOURCE"
    export STWO_DUMP_INPUT="$generated"
    exec "$GPU_BENCH_BIN" --pie "$SN_PIE_SOURCE_DIR/SN_PIE_${pie}.zip" \
      --backend simd --engine legacy --adapt-only
  ) > "$ROUND_DIR/adapter_reproduction_SN${pie}.log" 2>&1
  printf '%s  %s\n' "$(sha256_file "$generated")" "$(basename "$generated")" \
    >> "$ADAPTED_INPUT_MANIFEST"
done
verify_source_inputs
verify_adapted_inputs
ADAPTER_BINARY_SHA="$(sha256_file "$GPU_BENCH_BIN")"
RAW_INPUT_MANIFEST_SHA="$(sha256_file "$INPUT_MANIFEST")"
BOOTLOADER_SOURCE_SHA="$(sha256_file "$BOOTLOADER_JSON_SOURCE")"
PINNED_ADAPTED_MANIFEST_SHA="$(sha256_file "$PINNED_ADAPTED_INPUT_MANIFEST")"

run_preflight() {
  local profile_env="$1" pie="$2" output="$3"
  (
    while IFS='=' read -r stwo_name _; do
      case "$stwo_name" in STWO_*) unset "$stwo_name" ;; esac
    done < <(env)
    # shellcheck disable=SC2086 # fixed, release-owned KEY=VALUE profile tokens.
    env $profile_env "$PREFLIGHT_BIN" \
      --input-bincode "$ADAPTED_INPUT_DIR/SN_PIE_${pie}.adapted.bin" \
      --vram-budget-gb "$PREFLIGHT_VRAM_GIB"
  ) > "$output"
}

for pie in 1 2 3 4; do
  run_preflight "$UNIVERSAL_ENV" "$pie" "$ROUND_DIR/preflight_universal_SN${pie}.json"
done
run_preflight "" 2 "$ROUND_DIR/preflight_flags_off_SN2.json"
run_preflight "$SN2_HEADLINE_ENV" 2 "$ROUND_DIR/preflight_sn2_headline.json"
verify_adapted_inputs

PREFLIGHT_CAP="$PREFLIGHT_VRAM_BYTES" PREFLIGHT_DIR="$ROUND_DIR" \
PREFLIGHT_INPUTS="$ADAPTED_INPUT_DIR" PREFLIGHT_ADAPTED_MANIFEST="$ADAPTED_INPUT_MANIFEST" \
ADMISSION_STWO_HEAD="$STWO_HEAD" ADMISSION_STWO_HASH="$STWO_HASH" \
ADMISSION_CAIRO_HEAD="$CAIRO_HEAD" ADMISSION_CAIRO_HASH="$CAIRO_HASH" \
ADMISSION_UNIVERSAL_ENV="$UNIVERSAL_ENV" ADMISSION_HEADLINE_ENV="$SN2_HEADLINE_ENV" \
ADMISSION_ADAPTER_BINARY_SHA="$ADAPTER_BINARY_SHA" \
ADMISSION_RAW_MANIFEST_SHA="$RAW_INPUT_MANIFEST_SHA" \
ADMISSION_BOOTLOADER_SHA="$BOOTLOADER_SOURCE_SHA" \
ADMISSION_PINNED_ADAPTED_MANIFEST_SHA="$PINNED_ADAPTED_MANIFEST_SHA" \
ADMISSION_DRY_RUN="${DRY_RUN:-0}" \
python3 - "$ROUND_DIR/local_admission.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

cap = int(os.environ["PREFLIGHT_CAP"])
root = Path(os.environ["PREFLIGHT_DIR"])
inputs = Path(os.environ["PREFLIGHT_INPUTS"])
profiles = [
    ("preflight_flags_off_SN2.json", "SN_PIE_2.adapted.bin", {
        "commit_mode": "FullLifting",
        "direct_composition_retention_mode": "Disabled",
        "quotient_numerator_source_policy": "CoefficientsOnly",
        "interpolation_mode": "StageWiseCopyThenInPlace",
        "relation_launch_mode": "Fused",
        "retained_lde_budget_bytes": 8589934592,
    }),
] + [
    (f"preflight_universal_SN{i}.json", f"SN_PIE_{i}.adapted.bin", {
        "commit_mode": "DomainProgressive",
        "direct_composition_retention_mode": "Disabled",
        "quotient_numerator_source_policy": "CoefficientsOnly",
        "interpolation_mode": "StageFusedOutOfPlace",
        "relation_launch_mode": "Fused",
        "retained_lde_budget_bytes": 4563402752,
    }) for i in range(1, 5)
] + [("preflight_sn2_headline.json", "SN_PIE_2.adapted.bin", {
    "commit_mode": "DomainProgressive",
    "direct_composition_retention_mode": "ExactNative",
    "quotient_numerator_source_policy": "ReuseRetainedEvaluations",
    "interpolation_mode": "StageFusedOutOfPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 29469326848,
})]
preflight_hashes = {}
for artifact_name, input_name, expected_policy in profiles:
    artifact_path = root / artifact_name
    with open(artifact_path, encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("pass") is not True or record.get("vram_fit") is not True:
        raise SystemExit(f"preflight did not pass: {artifact_name}")
    if record.get("vram_budget_bytes") != cap:
        raise SystemExit(f"preflight ceiling drift: {artifact_name}")
    arena_bytes = (record.get("arena") or {}).get("total_bytes")
    if not isinstance(arena_bytes, int) or arena_bytes > cap:
        raise SystemExit(f"preflight arena exceeds ceiling: {artifact_name}")
    if record.get("runtime_policy") != expected_policy:
        raise SystemExit(f"preflight policy drift: {artifact_name}")
    source = Path(record.get("source", ""))
    if source.name != input_name or source.resolve() != (inputs / input_name).resolve():
        raise SystemExit(f"preflight input drift: {artifact_name}")
    preflight_hashes[artifact_name] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

adapted_manifest = Path(os.environ["PREFLIGHT_ADAPTED_MANIFEST"])
admission = {
    "schema": "stwo.local-preflight-admission.v1",
    "passed": True,
    "dry_run": os.environ["ADMISSION_DRY_RUN"] == "1",
    "runtime_mode": "arena-graph",
    "source": {
        "stwo": {
            "head": os.environ["ADMISSION_STWO_HEAD"],
            "worktree_hash": os.environ["ADMISSION_STWO_HASH"],
        },
        "stwo_cairo": {
            "head": os.environ["ADMISSION_CAIRO_HEAD"],
            "worktree_hash": os.environ["ADMISSION_CAIRO_HASH"],
        },
    },
    "profiles": {
        "flags_off": "",
        "universal_sn1_sn4": os.environ["ADMISSION_UNIVERSAL_ENV"],
        "sn2_headline": os.environ["ADMISSION_HEADLINE_ENV"],
    },
    "preflight_ceiling_bytes": cap,
    "preflight_artifact_sha256": preflight_hashes,
    "adapted_input_manifest": str(adapted_manifest),
    "adapted_input_manifest_sha256": hashlib.sha256(adapted_manifest.read_bytes()).hexdigest(),
    "adapter_reproduction": {
        "byte_equal": True,
        "gpu_bench_binary_sha256": os.environ["ADMISSION_ADAPTER_BINARY_SHA"],
        "raw_input_manifest_sha256": os.environ["ADMISSION_RAW_MANIFEST_SHA"],
        "bootloader_sha256": os.environ["ADMISSION_BOOTLOADER_SHA"],
        "pinned_adapted_manifest_sha256": os.environ[
            "ADMISSION_PINNED_ADAPTED_MANIFEST_SHA"
        ],
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(admission, stream, sort_keys=True)
    stream.write("\n")
PY

export LOCAL_PREFLIGHT_ADMISSION="$ROUND_DIR/local_admission.json"

# Independent local contract test: fail before sync, build, or paid GPU work.
(cd "$CAIRO_LOCAL/gpu_benchmarks" &&
  python3 -m unittest test_validate_architecture_record.py) \
  > "$ROUND_DIR/architecture_validator.log" 2>&1

# The sole sync/build and counted native suite. The optimized environment is
# normalized by this fresh process and recorded in the soundness JSON.
BENCH_ENV="$SN2_HEADLINE_ENV" "$SCRIPT_DIR/bench_loop.sh" --gate-only --all-pies
SOUNDNESS_GATE_COUNT="$(find "$ROUND_DIR" -maxdepth 1 -name '*.cuda-soundness-gate.json' | wc -l | tr -d ' ')"
[[ "$SOUNDNESS_GATE_COUNT" == "1" ]] || {
  echo "expected exactly one counted soundness artifact, got $SOUNDNESS_GATE_COUNT" >&2
  exit 1
}
SOUNDNESS_GATE="$(find "$ROUND_DIR" -maxdepth 1 -name '*.cuda-soundness-gate.json' -print)"
export ARCHITECTURE_SOUNDNESS_GATE="$SOUNDNESS_GATE"

# One normalized SN2 flags-off/headline A/B. Migration-only intermediate
# bundles are deliberately excluded from the paid release round.
REQUIRE_NCU_PROFILE=1 BENCH_ENV="" "$SCRIPT_DIR/perf_gates.sh" \
  --bundle sn2_headline --candidate-env "$SN2_HEADLINE_ENV" --reps 6 --ncu

# Reuse the exact counted artifact: no second native-suite execution. These
# records remain qualification probes until the final bound manifest passes.
QUALIFICATION_PROBE=1 BENCH_PROOF_HASHES=1 REUSE_SOUNDNESS_GATE="$SOUNDNESS_GATE" \
  BENCH_ENV="$UNIVERSAL_ENV" "$SCRIPT_DIR/bench_loop.sh" \
  --skip-sync --pie 2 --all-pies --reps 6

# Source must remain byte-identical throughout the round.
[[ "$STWO_HEAD" == "$(git -C "$STWO_LOCAL" rev-parse HEAD)" ]]
[[ "$CAIRO_HEAD" == "$(git -C "$CAIRO_LOCAL" rev-parse HEAD)" ]]
[[ "$STWO_HASH" == "$(source_hash "$STWO_LOCAL")" ]]
[[ "$CAIRO_HASH" == "$(source_hash "$CAIRO_LOCAL")" ]]
verify_adapted_inputs
verify_source_inputs

# Revalidate the exact counted manifest at publication time, then bind every
# ledger entry below to this final artifact byte identity.
python3 "$ARCHITECTURE_CHECK" --soundness-only \
  --soundness-gate "$SOUNDNESS_GATE" \
  --runtime-mode "$GPU_PCS_RUNTIME_MODE" --expected-dry-run "${DRY_RUN:-0}" \
  --stwo-head "$STWO_HEAD" --stwo-worktree-hash "$STWO_HASH" \
  --stwo-cairo-head "$CAIRO_HEAD" --stwo-cairo-worktree-hash "$CAIRO_HASH"
FINAL_SOUNDNESS_SHA="$(sha256_file "$SOUNDNESS_GATE")"

Q_STWO_HEAD="$STWO_HEAD" Q_STWO_HASH="$STWO_HASH" \
Q_CAIRO_HEAD="$CAIRO_HEAD" Q_CAIRO_HASH="$CAIRO_HASH" \
Q_UNIVERSAL_ENV="$UNIVERSAL_ENV" Q_HEADLINE_ENV="$SN2_HEADLINE_ENV" \
Q_RUNTIME="${GPU_PCS_RUNTIME_MODE:-arena-graph}" \
Q_SOUNDNESS="$SOUNDNESS_GATE" Q_SOUNDNESS_SHA="$FINAL_SOUNDNESS_SHA" \
Q_EXPECTED_GPU="$EXPECTED_POD_GPU" Q_VALIDATOR_DIR="$(dirname "$ARCHITECTURE_CHECK")" \
Q_INPUT_MANIFEST="$INPUT_MANIFEST" Q_INPUT_MANIFEST_SHA="$(sha256_file "$INPUT_MANIFEST")" \
Q_ADAPTED_MANIFEST="$ADAPTED_INPUT_MANIFEST" \
Q_ADAPTED_MANIFEST_SHA="$(sha256_file "$ADAPTED_INPUT_MANIFEST")" \
Q_LOCAL_ADMISSION="$LOCAL_PREFLIGHT_ADMISSION" \
Q_LOCAL_ADMISSION_SHA="$(sha256_file "$LOCAL_PREFLIGHT_ADMISSION")" \
Q_ADAPTER_BINARY_SHA="$ADAPTER_BINARY_SHA" \
Q_RAW_MANIFEST_SHA="$RAW_INPUT_MANIFEST_SHA" \
Q_BOOTLOADER_SOURCE_SHA="$BOOTLOADER_SOURCE_SHA" \
Q_PINNED_ADAPTED_MANIFEST="$PINNED_ADAPTED_INPUT_MANIFEST" \
Q_PINNED_ADAPTED_MANIFEST_SHA="$PINNED_ADAPTED_MANIFEST_SHA" \
Q_PREFLIGHT_DIR="$ROUND_DIR" Q_PREFLIGHT_INPUT_DIR="$ADAPTED_INPUT_DIR" \
Q_PREFLIGHT_CAP="$PREFLIGHT_VRAM_BYTES" Q_DRY_RUN="${DRY_RUN:-0}" \
Q_BENCH="$LEDGER" Q_BENCH_SHA="$(sha256_file "$LEDGER")" \
Q_PERF="$PERF_LEDGER" Q_PERF_SHA="$(sha256_file "$PERF_LEDGER")" \
python3 - "$ROUND_DIR/qualification.json" <<'PY'
import hashlib, json, os, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.environ["Q_VALIDATOR_DIR"])
from validate_architecture_record import (
    validate_benchmark_measurement,
    validate_gpu_telemetry_artifact,
)

boolean_flags = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
    "STWO_CUDA_B2N_STAGE_FUSED",
)
budget_flag = "STWO_CUDA_RETAINED_LDE_BUDGET_BYTES"
baseline_state = {**{flag: 0 for flag in boolean_flags}, budget_flag: 8589934592}
universal_state = {
    **baseline_state,
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE": 1,
    "STWO_CUDA_B2N_STAGE_FUSED": 1,
    budget_flag: 4563402752,
}
headline_state = {**{flag: 1 for flag in boolean_flags}, budget_flag: 29469326848}
round_dry_run = os.environ["Q_DRY_RUN"] == "1"
load_jsonl = lambda path: [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

perf = load_jsonl(os.environ["Q_PERF"])
bench = load_jsonl(os.environ["Q_BENCH"])
if len(perf) != 1 or perf[0].get("status") != "ok":
    raise SystemExit("SN2 headline A/B qualification probe is incomplete")
headline_ab = perf[0]
if (headline_ab.get("lane") != "sn2_headline"
        or headline_ab.get("bench_env") != ""
        or headline_ab.get("candidate_env") != os.environ["Q_HEADLINE_ENV"]
        or headline_ab.get("baseline_state") != baseline_state
        or headline_ab.get("candidate_state") != headline_state
        or headline_ab.get("reps") != 6 or not headline_ab.get("provisional")
        or headline_ab.get("performance_admissible") is not False
        or headline_ab.get("execution_guard_passed") is not True
        or headline_ab.get("remote_quiescence_passed") is not True
        or headline_ab.get("baseline", {}).get("proof_sha256")
           != headline_ab.get("flagged", {}).get("proof_sha256")
        or not headline_ab.get("baseline", {}).get("proof_sha256")):
    raise SystemExit("invalid normalized SN2 headline A/B entry")
ncu = headline_ab.get("ncu_profile") or {}
if (headline_ab.get("ncu_profile_required") is not True
        or headline_ab.get("ncu_profile_requested") is not True
        or headline_ab.get("ncu_profile_attempted") is not True
        or headline_ab.get("ncu_profile_status") != "validated"
        or ncu.get("schema") != "stwo.ncu-profile.v1"
        or ncu.get("kernel_regex")
           != "relation_fused|relation_scan|stream_leaf_update"
        or ncu.get("launch_count") != 10
        or ncu.get("set") != "full"
        or not isinstance(ncu.get("ncu_version"), str)
        or not ncu["ncu_version"].startswith(
            "NVIDIA (R) Nsight Compute Command Line Profiler"
        )
        or ncu.get("remote_import_validated") is not True
        or not isinstance(ncu.get("profiled_kernel_rows"), int)
        or isinstance(ncu.get("profiled_kernel_rows"), bool)
        or ncu["profiled_kernel_rows"] <= 0
        or ncu.get("synthetic") is not round_dry_run
        or ncu.get("profiled_proof_sha256")
           != headline_ab["baseline"]["proof_sha256"]):
    raise SystemExit("required SN2 Nsight Compute profile contract is incomplete")
for label, path_field, sha_field, bytes_field, remote_sha_field, remote_bytes_field in (
    ("report", "path", "sha256", "bytes", "remote_sha256", "remote_bytes"),
    (
        "import output",
        "import_output_path",
        "import_output_sha256",
        "import_output_bytes",
        "remote_import_output_sha256",
        "remote_import_output_bytes",
    ),
):
    path = Path(ncu.get(path_field, ""))
    expected_sha = ncu.get(sha_field)
    expected_bytes = ncu.get(bytes_field)
    if (not path.is_file()
            or not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
            or sha256(path) != expected_sha
            or path.stat().st_size != expected_bytes
            or ncu.get(remote_sha_field) != expected_sha
            or ncu.get(remote_bytes_field) != expected_bytes):
        raise SystemExit(f"required SN2 ncu {label} artifact is not content-bound")
for arm in ("baseline", "flagged"):
    arm_record = headline_ab.get(arm, {}).get("record") or {}
    measurement_errors = validate_benchmark_measurement(
        arm_record,
        expected_program="SN_PIE_2.zip",
        expected_reps=6,
        expected_gpu=os.environ["Q_EXPECTED_GPU"],
    )
    if measurement_errors:
        raise SystemExit(f"SN2 headline {arm} measurement: {measurement_errors}")
    value = headline_ab.get(arm, {}).get("useful_mhz_median")
    if (value != arm_record.get("useful_mhz_median")
            or not isinstance(value, (int, float)) or value <= 0):
        raise SystemExit(f"SN2 headline {arm} metric is absent or non-positive")
headline_comparison = dict(headline_ab)
headline_comparison["useful_mhz_ratio"] = (
    headline_ab["flagged"]["useful_mhz_median"]
    / headline_ab["baseline"]["useful_mhz_median"]
)

expected_bench_counts = Counter(
    {"gate_correctness": 2, **{f"SN_PIE_{i}": 1 for i in range(1, 5)}}
)
if Counter(entry.get("run_name") for entry in bench) != expected_bench_counts:
    raise SystemExit("benchmark ledger entry set is incomplete, duplicated, or unexpected")
pies = {entry.get("run_name"): entry for entry in bench if entry.get("run_name", "").startswith("SN_PIE_")}
if set(pies) != {"SN_PIE_1", "SN_PIE_2", "SN_PIE_3", "SN_PIE_4"}:
    raise SystemExit("fixed-PIE qualification set is incomplete")
for name, entry in pies.items():
    record = entry.get("record") or {}
    measurement_errors = validate_benchmark_measurement(
        record,
        expected_program=f"{name}.zip",
        expected_reps=6,
        expected_gpu=os.environ["Q_EXPECTED_GPU"],
        require_fresh_simd_reference=True,
    )
    if measurement_errors:
        raise SystemExit(f"invalid fixed-PIE measurement {name}: {measurement_errors}")
    telemetry_errors = validate_gpu_telemetry_artifact(
        record, entry.get("gpu_telemetry") or {}
    )
    if telemetry_errors:
        raise SystemExit(f"invalid fixed-PIE GPU telemetry {name}: {telemetry_errors}")
    if (entry.get("status") != "ok"
            or entry.get("bench_env") != os.environ["Q_UNIVERSAL_ENV"]
            or not entry.get("qualification_probe") or not entry.get("proof_sha256")
            or entry.get("execution_guard_passed") is not True
            or entry.get("remote_quiescence_passed") is not True
            or record.get("verified_reps") != 6 or record.get("proof_byte_equal") is not True
            or record.get("proof_byte_equal_required") is not True
            or record.get("performance_claim_admissible") is not True
            or not isinstance(record.get("useful_mhz_median"), (int, float))
            or record.get("useful_mhz_median") <= 0):
        raise SystemExit(f"invalid fixed-PIE entry: {name}")
if pies["SN_PIE_2"].get("proof_sha256") != headline_ab["baseline"]["proof_sha256"]:
    raise SystemExit("universal SN2 proof bytes differ from the flags-off/headline proof")
gpus = {entry.get("pod_gpu") for entry in bench}
if gpus != {os.environ["Q_EXPECTED_GPU"]}:
    raise SystemExit("qualification records do not bind the exact release H100")

with open(os.environ["Q_SOUNDNESS"], encoding="utf-8") as stream:
    soundness = json.load(stream)
if hashlib.sha256(Path(os.environ["Q_SOUNDNESS"]).read_bytes()).hexdigest() != os.environ["Q_SOUNDNESS_SHA"]:
    raise SystemExit("counted soundness artifact changed after final validation")
if (soundness.get("passed") is not True
        or soundness.get("runtime_mode") != os.environ["Q_RUNTIME"]
        or soundness.get("dry_run") is not round_dry_run):
    raise SystemExit("counted soundness artifact did not pass")
synced = soundness.get("synced_source") or {}
expected_synced = {
    "stwo": {"head": os.environ["Q_STWO_HEAD"], "worktree_hash": os.environ["Q_STWO_HASH"]},
    "stwo_cairo": {
        "head": os.environ["Q_CAIRO_HEAD"],
        "worktree_hash": os.environ["Q_CAIRO_HASH"],
    },
    "transport": "rsync-archive-checksum",
}
if synced != expected_synced:
    raise SystemExit("counted soundness artifact source identity does not match the checksum sync")
qualified_env = soundness.get("qualification_flags")
if qualified_env != {flag: 1 for flag in boolean_flags}:
    raise SystemExit("counted soundness artifact has the wrong headline boolean environment")
expected_stwo_env = {
    "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
    "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
    "STWO_PARITY_REF_STWO_HEAD": os.environ["Q_STWO_HEAD"],
    "STWO_PARITY_REF_STWO_WORKTREE_HASH": os.environ["Q_STWO_HASH"],
    "STWO_PARITY_REF_STWO_CAIRO_HEAD": os.environ["Q_CAIRO_HEAD"],
    "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": os.environ["Q_CAIRO_HASH"],
    **{flag: "1" for flag in boolean_flags},
    budget_flag: "29469326848",
}
if soundness.get("effective_stwo_env") != expected_stwo_env:
    raise SystemExit("counted soundness artifact has an unapproved effective STWO environment")

checksums = {}
with open(os.environ["Q_INPUT_MANIFEST"], encoding="utf-8") as stream:
    for line in stream:
        digest, name = line.split()
        checksums[name] = digest
required_inputs = [f"SN_PIE_{i}.zip" for i in range(1, 5)] + ["simple_bootloader_compiled.json"]
if any(name not in checksums for name in required_inputs):
    raise SystemExit("input checksum manifest is incomplete")

execution_target = soundness.get("execution_target") or {}
target_encoded = json.dumps(
    execution_target, sort_keys=True, separators=(",", ":")
).encode()
target_sha = hashlib.sha256(target_encoded).hexdigest()
binary = execution_target.get("gpu_bench") or {}
expected_inputs = {
    "gate": {
        "path": "/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_2.zip",
        "sha256": checksums["SN_PIE_2.zip"],
    },
    "bootloader": {
        "path": "/workspace/bench_inputs/simple_bootloader_compiled.json",
        "sha256": checksums["simple_bootloader_compiled.json"],
    },
    **{
        f"SN_PIE_{i}": {
            "path": f"/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_{i}.zip",
            "sha256": checksums[f"SN_PIE_{i}.zip"],
        }
        for i in range(1, 5)
    },
}
expected_projection_source = {
    "stwo": {"head": os.environ["Q_STWO_HEAD"], "worktree_hash": os.environ["Q_STWO_HASH"]},
    "stwo_cairo": {
        "head": os.environ["Q_CAIRO_HEAD"], "worktree_hash": os.environ["Q_CAIRO_HASH"]
    },
}
projection = soundness.get("source_projection") or {}
if (soundness.get("schema") != "stwo.cuda.soundness-gate.v3"
        or execution_target.get("schema") != "stwo.remote-execution-target.v1"
        or soundness.get("execution_target_sha256") != target_sha
        or soundness.get("execution_target_postcheck") is not True
        or execution_target.get("inputs") != expected_inputs
        or not isinstance(binary.get("sha256"), str)
        or len(binary["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in binary["sha256"])
        or binary.get("path") != f"/workspace/bench_loop_runs/sealed/gpu_bench.{binary['sha256']}"
        or execution_target.get("gpu_name") != next(iter(gpus))
        or projection != {
            "method": "rsync-archive-checksum-dry-run-clean",
            "verified_after_soundness": True,
            "source": expected_projection_source,
        }):
    raise SystemExit("remote pod/GPU/binary/input/source execution seal is invalid")
if headline_ab.get("pod_gpu") != next(iter(gpus)):
    raise SystemExit("SN2 headline A/B ran on a different GPU")
if (headline_ab.get("architecture_soundness") or {}).get("sha256") != os.environ["Q_SOUNDNESS_SHA"]:
    raise SystemExit("SN2 headline A/B references a different soundness artifact")
for entry in [headline_ab, *bench]:
    if (entry.get("execution_target") != execution_target
            or entry.get("execution_target_sha256") != target_sha
            or entry.get("source_projection") != projection
            or entry.get("execution_guard_passed") is not True
            or entry.get("remote_quiescence_passed") is not True):
        raise SystemExit("qualification ledger entry escaped the sealed execution target")
for entry in bench:
    if entry.get("soundness_gate_sha256") != os.environ["Q_SOUNDNESS_SHA"]:
        raise SystemExit("benchmark ledger references a different soundness artifact")

preflight_root = Path(os.environ["Q_PREFLIGHT_DIR"])
adapted_root = Path(os.environ["Q_PREFLIGHT_INPUT_DIR"])
preflight_cap = int(os.environ["Q_PREFLIGHT_CAP"])
adapted_checksums = {}
with open(os.environ["Q_ADAPTED_MANIFEST"], encoding="utf-8") as stream:
    for line in stream:
        digest, name = line.split()
        adapted_checksums[name] = digest
if set(adapted_checksums) != {f"SN_PIE_{i}.adapted.bin" for i in range(1, 5)}:
    raise SystemExit("adapted-input checksum manifest is incomplete or unexpected")
universal_policy = {
    "commit_mode": "DomainProgressive",
    "direct_composition_retention_mode": "Disabled",
    "quotient_numerator_source_policy": "CoefficientsOnly",
    "interpolation_mode": "StageFusedOutOfPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 4563402752,
}
flags_off_policy = {
    "commit_mode": "FullLifting",
    "direct_composition_retention_mode": "Disabled",
    "quotient_numerator_source_policy": "CoefficientsOnly",
    "interpolation_mode": "StageWiseCopyThenInPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 8589934592,
}
headline_policy = {
    "commit_mode": "DomainProgressive",
    "direct_composition_retention_mode": "ExactNative",
    "quotient_numerator_source_policy": "ReuseRetainedEvaluations",
    "interpolation_mode": "StageFusedOutOfPlace",
    "relation_launch_mode": "Fused",
    "retained_lde_budget_bytes": 29469326848,
}

def preflight_summary(artifact_name, input_name, expected_policy):
    artifact_path = preflight_root / artifact_name
    input_path = adapted_root / input_name
    with open(artifact_path, encoding="utf-8") as stream:
        record = json.load(stream)
    arena = record.get("arena") or {}
    input_sha = sha256(input_path)
    if (record.get("pass") is not True or record.get("vram_fit") is not True
            or record.get("vram_budget_bytes") != preflight_cap
            or not isinstance(arena.get("total_bytes"), int)
            or arena["total_bytes"] > preflight_cap
            or record.get("runtime_policy") != expected_policy
            or Path(record.get("source", "")).resolve() != input_path.resolve()
            or input_sha != adapted_checksums[input_name]):
        raise SystemExit(f"preflight drift in {artifact_name}")
    return {
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "adapted_input": str(input_path),
        "adapted_input_sha256": input_sha,
        "arena_bytes": arena["total_bytes"],
        "arena_gib": arena["total_bytes"] / 1024**3,
        "runtime_policy": record["runtime_policy"],
    }

universal_preflights = {
    f"SN_PIE_{i}": preflight_summary(
        f"preflight_universal_SN{i}.json", f"SN_PIE_{i}.adapted.bin", universal_policy
    ) for i in range(1, 5)
}
flags_off_preflight = preflight_summary(
    "preflight_flags_off_SN2.json", "SN_PIE_2.adapted.bin", flags_off_policy
)
headline_preflight = preflight_summary(
    "preflight_sn2_headline.json", "SN_PIE_2.adapted.bin", headline_policy
)
with open(os.environ["Q_LOCAL_ADMISSION"], encoding="utf-8") as stream:
    local_admission = json.load(stream)
expected_admission_source = {
    "stwo": {"head": os.environ["Q_STWO_HEAD"], "worktree_hash": os.environ["Q_STWO_HASH"]},
    "stwo_cairo": {
        "head": os.environ["Q_CAIRO_HEAD"], "worktree_hash": os.environ["Q_CAIRO_HASH"]
    },
}
expected_preflight_hashes = {
    f"preflight_universal_SN{i}.json": universal_preflights[f"SN_PIE_{i}"]["artifact_sha256"]
    for i in range(1, 5)
}
expected_preflight_hashes["preflight_flags_off_SN2.json"] = flags_off_preflight[
    "artifact_sha256"
]
expected_preflight_hashes["preflight_sn2_headline.json"] = headline_preflight["artifact_sha256"]
if (local_admission.get("schema") != "stwo.local-preflight-admission.v1"
        or local_admission.get("passed") is not True
        or local_admission.get("dry_run") is not round_dry_run
        or local_admission.get("runtime_mode") != os.environ["Q_RUNTIME"]
        or local_admission.get("source") != expected_admission_source
        or local_admission.get("profiles") != {
            "flags_off": "",
            "universal_sn1_sn4": os.environ["Q_UNIVERSAL_ENV"],
            "sn2_headline": os.environ["Q_HEADLINE_ENV"],
        }
        or local_admission.get("preflight_ceiling_bytes") != preflight_cap
        or local_admission.get("preflight_artifact_sha256") != expected_preflight_hashes):
    raise SystemExit("local preflight admission drifted during qualification")
expected_adapter_reproduction = {
    "byte_equal": True,
    "gpu_bench_binary_sha256": os.environ["Q_ADAPTER_BINARY_SHA"],
    "raw_input_manifest_sha256": os.environ["Q_RAW_MANIFEST_SHA"],
    "bootloader_sha256": os.environ["Q_BOOTLOADER_SOURCE_SHA"],
    "pinned_adapted_manifest_sha256": os.environ["Q_PINNED_ADAPTED_MANIFEST_SHA"],
}
if local_admission.get("adapter_reproduction") != expected_adapter_reproduction:
    raise SystemExit("raw PIE adapter reproduction drifted during qualification")

artifact = {
    "schema": "stwo.qualification-round.v4",
    "status": "dry_run" if round_dry_run else "passed",
    "dry_run": round_dry_run,
    "performance_admissible": not round_dry_run,
    "profiles": {
        "universal_sn1_sn4": {
            "env": os.environ["Q_UNIVERSAL_ENV"],
            "effective_state": universal_state,
            "preflight_ceiling_bytes": preflight_cap,
            "preflights": universal_preflights,
        },
        "sn2_headline": {
            "env": os.environ["Q_HEADLINE_ENV"],
            "effective_state": headline_state,
            "preflight_ceiling_bytes": preflight_cap,
            "preflight": headline_preflight,
        },
    },
    "normalized_states": {
        "flags_off": baseline_state,
        "universal_sn1_sn4": universal_state,
        "sn2_headline": headline_state,
    },
    "flags_off_preflight": flags_off_preflight,
    "runtime_mode": os.environ["Q_RUNTIME"],
    "gpu": next(iter(gpus)),
    "source": {
        "stwo": {"head": os.environ["Q_STWO_HEAD"], "worktree_hash": os.environ["Q_STWO_HASH"]},
        "stwo_cairo": {"head": os.environ["Q_CAIRO_HEAD"], "worktree_hash": os.environ["Q_CAIRO_HASH"]},
        "sync": {"method": "rsync checksum with target and result exclusions",
                 "release_requires_clean_commits": True},
    },
    "local_admission": {
        "path": os.environ["Q_LOCAL_ADMISSION"],
        "sha256": os.environ["Q_LOCAL_ADMISSION_SHA"],
    },
    "adapter_reproduction": expected_adapter_reproduction,
    "remote_execution_target": {
        "target": execution_target,
        "sha256": target_sha,
        "postcheck": True,
        "source_projection": projection,
        "ledger_entries_guarded": len(bench) + 1,
        "remote_quiescence_passed": True,
        "quiescent_ledger_entries": len(bench) + len(perf),
        "quiescent_measurement_launches": len(bench) + 3,
    },
    "soundness": {"path": os.environ["Q_SOUNDNESS"], "sha256": os.environ["Q_SOUNDNESS_SHA"],
                  "effective_stwo_env": soundness.get("effective_stwo_env"),
                  "stwo_worktree_hash": soundness.get("stwo_worktree_hash"),
                  "stwo_cairo_worktree_hash": soundness.get("stwo_cairo_worktree_hash")},
    "inputs": {"manifest": os.environ["Q_INPUT_MANIFEST"],
               "manifest_sha256": os.environ["Q_INPUT_MANIFEST_SHA"],
               "files": {name: checksums[name] for name in required_inputs},
               "adapted_manifest": os.environ["Q_ADAPTED_MANIFEST"],
               "adapted_manifest_sha256": os.environ["Q_ADAPTED_MANIFEST_SHA"],
               "pinned_adapted_manifest": os.environ["Q_PINNED_ADAPTED_MANIFEST"],
               "pinned_adapted_manifest_sha256": os.environ["Q_PINNED_ADAPTED_MANIFEST_SHA"],
               "adapted_files": adapted_checksums,
               "gate": "SN_PIE_2.zip"},
    "proof_sha256": {
        "ab": {"sn2_headline": {
            "flags_off": headline_ab["baseline"]["proof_sha256"],
            "headline": headline_ab["flagged"]["proof_sha256"],
        }},
        "fixed_pies": {name: entry["proof_sha256"] for name, entry in sorted(pies.items())},
    },
    "benchmarks": {name: entry["record"] for name, entry in sorted(pies.items())},
    "gpu_telemetry": {
        name: entry["gpu_telemetry"] for name, entry in sorted(pies.items())
    },
    "profiling": {
        "ncu": {
            "required": True,
            "requested": True,
            "attempted": True,
            "status": "validated",
            "profile": ncu,
        },
    },
    "comparisons": {"sn2_flags_off_vs_headline": headline_comparison},
    "ledgers": {
        "bench": {"path": os.environ["Q_BENCH"], "sha256": os.environ["Q_BENCH_SHA"]},
        "ab": {"path": os.environ["Q_PERF"], "sha256": os.environ["Q_PERF_SHA"]},
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(artifact, stream, sort_keys=True)
    stream.write("\n")
PY

STATUS="passed"
echo "qualification artifacts: $ROUND_DIR"
