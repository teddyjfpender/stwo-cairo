#!/usr/bin/env bash
# Fail-closed post-admission qualification and fixed-statement measurement round.
set -euo pipefail

[[ -z "${OPTIMIZED_ENV+x}" ]] || {
  echo "OPTIMIZED_ENV is fixed by the release harness and may not be overridden" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAIRO_LOCAL="$(cd "${CAIRO_LOCAL:-${SCRIPT_DIR}/../..}" && pwd)"
STWO_LOCAL="$(cd "${STWO_LOCAL:-${CAIRO_LOCAL}/../stwo}" && pwd)"
export CAIRO_LOCAL STWO_LOCAL
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROUND_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/qualification_${STAMP}}"
OPTIMIZED_ENV="STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1 STWO_CUDA_COMPOSITION_DIRECT_RETENTION=1 STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS=1"
PROGRESSIVE_ENV="STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE=1"
DIRECT_ENV="$PROGRESSIVE_ENV STWO_CUDA_COMPOSITION_DIRECT_RETENTION=1"
INPUT_MANIFEST="$CAIRO_LOCAL/gpu_benchmarks/pie/SHA256SUMS"
export GATE_PIE="/workspace/stwo-cairo/gpu_benchmarks/pie/sn/SN_PIE_2.zip"

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

mkdir -p "$ROUND_DIR"
for artifact in qualification.json bench.jsonl sn2_ab.jsonl optimized.env; do
  [[ ! -e "$ROUND_DIR/$artifact" ]] \
    || { echo "qualification output already exists: $ROUND_DIR/$artifact" >&2; exit 1; }
done
export RESULTS_DIR="$ROUND_DIR"
export LEDGER="$ROUND_DIR/bench.jsonl"
export PERF_LEDGER="$ROUND_DIR/sn2_ab.jsonl"
printf '%s\n' "$OPTIMIZED_ENV" > "$ROUND_DIR/optimized.env"

STATUS="failed"
write_failure_manifest() {
  [[ "$STATUS" == "passed" ]] && return
  STATUS_VALUE="$STATUS" OPTIMIZED_VALUE="$OPTIMIZED_ENV" python3 - "$ROUND_DIR/qualification.json" <<'PY'
import json, os, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"schema": "stwo.qualification-round.v2", "status": os.environ["STATUS_VALUE"],
               "optimized_env": os.environ["OPTIMIZED_VALUE"]}, stream, sort_keys=True)
    stream.write("\n")
PY
}
trap write_failure_manifest EXIT

# Independent local contract test: fail before sync, build, or paid GPU work.
(cd "$CAIRO_LOCAL/gpu_benchmarks" &&
  python3 -m unittest test_validate_architecture_record.py) \
  > "$ROUND_DIR/architecture_validator.log" 2>&1

# The sole sync/build and counted native suite. The optimized environment is
# normalized by this fresh process and recorded in the soundness JSON.
BENCH_ENV="$OPTIMIZED_ENV" "$SCRIPT_DIR/bench_loop.sh" --gate-only
SOUNDNESS_GATE_COUNT="$(find "$ROUND_DIR" -maxdepth 1 -name '*.cuda-soundness-gate.json' | wc -l | tr -d ' ')"
[[ "$SOUNDNESS_GATE_COUNT" == "1" ]] || {
  echo "expected exactly one counted soundness artifact, got $SOUNDNESS_GATE_COUNT" >&2
  exit 1
}
SOUNDNESS_GATE="$(find "$ROUND_DIR" -maxdepth 1 -name '*.cuda-soundness-gate.json' -print)"
export ARCHITECTURE_SOUNDNESS_GATE="$SOUNDNESS_GATE"

# Fresh-process normalized A/B probes. These are explicitly provisional and
# cannot be consumed as ordinary publishable benchmark entries.
BENCH_ENV="" "$SCRIPT_DIR/perf_gates.sh" --bundle progressive \
  --candidate-env "$PROGRESSIVE_ENV" --reps 2
BENCH_ENV="" "$SCRIPT_DIR/perf_gates.sh" --bundle progressive_direct \
  --candidate-env "$DIRECT_ENV" --reps 2
BENCH_ENV="" "$SCRIPT_DIR/perf_gates.sh" --bundle optimized_resident \
  --candidate-env "$OPTIMIZED_ENV" --reps 6

# Reuse the exact counted artifact: no second native-suite execution. These
# records remain qualification probes until the final bound manifest passes.
QUALIFICATION_PROBE=1 BENCH_PROOF_HASHES=1 REUSE_SOUNDNESS_GATE="$SOUNDNESS_GATE" \
  BENCH_ENV="$OPTIMIZED_ENV" "$SCRIPT_DIR/bench_loop.sh" \
  --skip-sync --pie 2 --all-pies --reps 6

# Source must remain byte-identical throughout the round.
[[ "$STWO_HEAD" == "$(git -C "$STWO_LOCAL" rev-parse HEAD)" ]]
[[ "$CAIRO_HEAD" == "$(git -C "$CAIRO_LOCAL" rev-parse HEAD)" ]]
[[ "$STWO_HASH" == "$(source_hash "$STWO_LOCAL")" ]]
[[ "$CAIRO_HASH" == "$(source_hash "$CAIRO_LOCAL")" ]]

Q_STWO_HEAD="$STWO_HEAD" Q_STWO_HASH="$STWO_HASH" \
Q_CAIRO_HEAD="$CAIRO_HEAD" Q_CAIRO_HASH="$CAIRO_HASH" \
Q_ENV="$OPTIMIZED_ENV" Q_RUNTIME="${GPU_PCS_RUNTIME_MODE:-arena-graph}" \
Q_SOUNDNESS="$SOUNDNESS_GATE" Q_SOUNDNESS_SHA="$(sha256_file "$SOUNDNESS_GATE")" \
Q_INPUT_MANIFEST="$INPUT_MANIFEST" Q_INPUT_MANIFEST_SHA="$(sha256_file "$INPUT_MANIFEST")" \
Q_BENCH="$LEDGER" Q_BENCH_SHA="$(sha256_file "$LEDGER")" \
Q_PERF="$PERF_LEDGER" Q_PERF_SHA="$(sha256_file "$PERF_LEDGER")" \
python3 - "$ROUND_DIR/qualification.json" <<'PY'
import json, os, sys

flags = (
    "STWO_CUDA_COMMIT_DOMAIN_PROGRESSIVE",
    "STWO_CUDA_COMPOSITION_DIRECT_RETENTION",
    "STWO_CUDA_QUOTIENT_REUSE_RETAINED_EVALUATIONS",
)
states = [
    {flag: 0 for flag in flags},
    {flag: int(i == 0) for i, flag in enumerate(flags)},
    {flag: int(i <= 1) for i, flag in enumerate(flags)},
    {flag: 1 for flag in flags},
]
load_jsonl = lambda path: [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
perf = load_jsonl(os.environ["Q_PERF"])
bench = load_jsonl(os.environ["Q_BENCH"])
if len(perf) != 3 or any(entry.get("status") != "ok" for entry in perf):
    raise SystemExit("A/B qualification probes are incomplete")
for entry, lane, candidate, reps in zip(
        perf, ("progressive", "progressive_direct", "optimized_resident"), states[1:], (2, 2, 6)):
    if (entry.get("lane") != lane
            or entry.get("baseline_state") != states[0] or entry.get("candidate_state") != candidate
            or entry.get("reps") != reps or not entry.get("provisional")
            or entry.get("performance_admissible") is not False
            or entry.get("baseline", {}).get("proof_sha256") != entry.get("flagged", {}).get("proof_sha256")
            or not entry.get("baseline", {}).get("proof_sha256")):
        raise SystemExit(f"invalid normalized A/B entry: {entry.get('lane')}")
optimized_ab = perf[2]
for arm in ("baseline", "flagged"):
    value = optimized_ab.get(arm, {}).get("useful_mhz_median")
    if not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"optimized SN2 {arm} metric is absent or non-positive")
optimized_comparison = dict(optimized_ab)
optimized_comparison["useful_mhz_ratio"] = (
    optimized_ab["flagged"]["useful_mhz_median"]
    / optimized_ab["baseline"]["useful_mhz_median"]
)

pies = {entry.get("run_name"): entry for entry in bench if entry.get("run_name", "").startswith("SN_PIE_")}
if set(pies) != {"SN_PIE_1", "SN_PIE_2", "SN_PIE_3", "SN_PIE_4"}:
    raise SystemExit("fixed-PIE qualification set is incomplete")
for name, entry in pies.items():
    record = entry.get("record") or {}
    if (entry.get("status") != "ok" or entry.get("bench_env") != os.environ["Q_ENV"]
            or not entry.get("qualification_probe") or not entry.get("proof_sha256")
            or record.get("verified_reps") != 6 or record.get("proof_byte_equal") is not True
            or record.get("proof_byte_equal_required") is not True
            or record.get("performance_claim_admissible") is not True
            or not isinstance(record.get("useful_mhz_median"), (int, float))
            or record.get("useful_mhz_median") <= 0):
        raise SystemExit(f"invalid fixed-PIE entry: {name}")
gpus = {entry.get("pod_gpu") for entry in bench}
if len(gpus) != 1 or None in gpus or "" in gpus:
    raise SystemExit("qualification records do not bind one exact GPU")

with open(os.environ["Q_SOUNDNESS"], encoding="utf-8") as stream:
    soundness = json.load(stream)
if soundness.get("passed") is not True or soundness.get("runtime_mode") != os.environ["Q_RUNTIME"]:
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
if qualified_env != states[3]:
    raise SystemExit("counted soundness artifact has the wrong optimized environment")
expected_stwo_env = {
    "STWO_CUDA_OBJ_CACHE": "/workspace/.cuda_obj_cache",
    "STWO_PARITY_REF_CACHE": "/workspace/.parity_ref_cache",
    "STWO_PARITY_REF_STWO_HEAD": os.environ["Q_STWO_HEAD"],
    "STWO_PARITY_REF_STWO_WORKTREE_HASH": os.environ["Q_STWO_HASH"],
    "STWO_PARITY_REF_STWO_CAIRO_HEAD": os.environ["Q_CAIRO_HEAD"],
    "STWO_PARITY_REF_STWO_CAIRO_WORKTREE_HASH": os.environ["Q_CAIRO_HASH"],
    **{flag: "1" for flag in flags},
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

artifact = {
    "schema": "stwo.qualification-round.v2",
    "status": "passed",
    "performance_admissible": True,
    "optimized_env": os.environ["Q_ENV"],
    "normalized_states": states,
    "runtime_mode": os.environ["Q_RUNTIME"],
    "gpu": next(iter(gpus)),
    "source": {
        "stwo": {"head": os.environ["Q_STWO_HEAD"], "worktree_hash": os.environ["Q_STWO_HASH"]},
        "stwo_cairo": {"head": os.environ["Q_CAIRO_HEAD"], "worktree_hash": os.environ["Q_CAIRO_HASH"]},
        "sync": {"method": "rsync checksum with target and result exclusions",
                 "release_requires_clean_commits": True},
    },
    "soundness": {"path": os.environ["Q_SOUNDNESS"], "sha256": os.environ["Q_SOUNDNESS_SHA"],
                  "effective_stwo_env": soundness.get("effective_stwo_env"),
                  "stwo_worktree_hash": soundness.get("stwo_worktree_hash"),
                  "stwo_cairo_worktree_hash": soundness.get("stwo_cairo_worktree_hash")},
    "inputs": {"manifest": os.environ["Q_INPUT_MANIFEST"],
               "manifest_sha256": os.environ["Q_INPUT_MANIFEST_SHA"],
               "files": {name: checksums[name] for name in required_inputs},
               "gate": "SN_PIE_2.zip"},
    "proof_sha256": {
        "ab": {entry["lane"]: entry["baseline"]["proof_sha256"] for entry in perf},
        "fixed_pies": {name: entry["proof_sha256"] for name, entry in sorted(pies.items())},
    },
    "benchmarks": {name: entry["record"] for name, entry in sorted(pies.items())},
    "comparisons": {"sn2_flags_off_vs_optimized": optimized_comparison},
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
