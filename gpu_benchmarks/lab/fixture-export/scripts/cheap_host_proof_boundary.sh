#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$FIXTURE_DIR/../../.." && pwd)"
MANIFEST="$FIXTURE_DIR/Cargo.toml"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/stwo_cairo_prover/target}"
EXPORTER="$TARGET_DIR/debug/stwo-gpu-lab-fixture-export"
TEST_NAME='fri_round6_proof::tests::real_cairo_proof_crosses_the_full_validation_boundary'
STATUS='FRI_ROUND6_PROOF_PREFLIGHT=PASS production_admissible=false resource_limits=pass proof_bincode_roundtrip_match=pass proof_verification=pass canonical_transport_match=pass proof_shape_match=pass adapter_execution_attestation=pending verifier_closure_match=pending capture_verifier_match=pending'

for tool in cargo timeout sha256sum; do
  command -v "$tool" >/dev/null || {
    echo "cheap-host-proof-boundary: missing required tool: $tool" >&2
    exit 2
  }
done

scratch="$(mktemp -d "${TMPDIR:-/tmp}/stwo-proof-boundary.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT

export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="$TARGET_DIR"
export RUSTFLAGS="${RUSTFLAGS:--C debuginfo=0}"

timeout --signal=TERM --kill-after=5s 20m \
  cargo +nightly-2025-06-23 build --locked --jobs 1 \
  --manifest-path "$MANIFEST" \
  --bin stwo-gpu-lab-fixture-export

test_log="$scratch/ignored-test.log"
(
  STWO_GPU_LAB_RUN_REAL_PROOF_BOUNDARY=1 \
  STWO_GPU_LAB_PROOF_BUNDLE_DIR="$scratch/bundles" \
    exec timeout --signal=TERM --kill-after=5s 20m \
      cargo +nightly-2025-06-23 test --locked --jobs 1 \
      --manifest-path "$MANIFEST" \
      --bin stwo-gpu-lab-fixture-export \
      "$TEST_NAME" -- --ignored --exact
) >"$test_log" 2>&1

[[ "$(grep -Fxc "test $TEST_NAME ... ok" "$test_log" || true)" == 1 ]] || {
  echo "cheap-host-proof-boundary: exact ignored test did not pass once" >&2
  exit 1
}
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out; finished in .+$' "$test_log" || true)" == 1 ]] || {
  echo "cheap-host-proof-boundary: ignored-test summary was not exactly 1 passed" >&2
  exit 1
}
[[ "$(grep -c '^test result:' "$test_log" || true)" == 1 ]] || {
  echo "cheap-host-proof-boundary: unexpected additional test summary" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

run_bounded() {
  local stdout_path="$1"
  local stderr_path="$2"
  shift 2
  (
    ulimit -f 128
    exec timeout --signal=TERM --kill-after=5s 90s "$@"
  ) >"$stdout_path" 2>"$stderr_path"
}

run_case() {
  local name="$1"
  local case_dir="$scratch/bundles/$name"
  local manifest="$case_dir/fri_round6_provenance.v1.json"
  local stdout_path="$scratch/$name.stdout"
  local stderr_path="$scratch/$name.stderr"
  local manifest_sha
  manifest_sha="$(sha256_file "$manifest")"
  if run_bounded "$stdout_path" "$stderr_path" \
    "$EXPORTER" \
    --preflight-fri-round6-proof "$manifest" \
    --fri-round6-provenance-sha256 "$manifest_sha"; then
    RUN_STATUS=0
  else
    RUN_STATUS=$?
  fi
  RUN_STDOUT="$stdout_path"
  RUN_STDERR="$stderr_path"
  RUN_MANIFEST_SHA="$manifest_sha"
}

run_case positive
[[ "$RUN_STATUS" == 0 ]] || {
  echo "cheap-host-proof-boundary: positive case exited $RUN_STATUS" >&2
  exit 1
}
[[ ! -s "$RUN_STDERR" ]] || {
  echo "cheap-host-proof-boundary: positive case wrote stderr" >&2
  exit 1
}
expected="$STATUS manifest_sha256=$RUN_MANIFEST_SHA proof_sha256=$(sha256_file "$scratch/bundles/positive/extended-proof.bin") canonical_transport_sha256=$(sha256_file "$scratch/bundles/positive/canonical-transport.bin") proof_shape_sha256=$(tr -d '\n' <"$scratch/bundles/positive/proof_shape.sha256")"
[[ "$(wc -l <"$RUN_STDOUT" | tr -d ' ')" == 1 ]] && [[ "$(cat "$RUN_STDOUT")" == "$expected" ]] || {
  echo "cheap-host-proof-boundary: positive status was not the one exact expected line" >&2
  exit 1
}

for mutation in proof_mutation transport_mutation shape_mutation; do
  run_case "$mutation"
  [[ "$RUN_STATUS" != 0 ]] || {
    echo "cheap-host-proof-boundary: $mutation unexpectedly exited zero" >&2
    exit 1
  }
  ! grep -Fq 'FRI_ROUND6_PROOF_PREFLIGHT=PASS' "$RUN_STDOUT" "$RUN_STDERR" || {
    echo "cheap-host-proof-boundary: $mutation emitted an accepted status" >&2
    exit 1
  }
done

echo "cheap-host-proof-boundary: PASS (1 positive, 3 rejected mutations)"
