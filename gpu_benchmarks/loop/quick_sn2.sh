#!/usr/bin/env bash
set -euo pipefail

loop_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stwo_dir="${STWO_LOCAL:-$(cd "$loop_dir/../../.." && pwd)/stwo}"
label="${1:-sn2_iteration_$(date -u +%Y%m%dT%H%M%SZ)}"
export POD_RUN_POLL_INTERVAL="${POD_RUN_POLL_INTERVAL:-2}"

# Spend a few local seconds catching launcher/provenance regressions before a
# paid pod starts. This suite is CPU-only and does not build the prover.
(cd "$loop_dir/.." && python3 -m unittest test_shell_launchers)
cache_test="$(mktemp "${TMPDIR:-/tmp}/stwo-build-cache-test.XXXXXX")"
trap 'rm -f -- "$cache_test"' EXIT
rustc --edition=2021 -D warnings --test \
  "$stwo_dir/crates/backend-cuda-kernels/build.rs" -o "$cache_test"
"$cache_test" --quiet
rm -f -- "$cache_test"
trap - EXIT

exec "$loop_dir/pod_run.sh" \
  "$loop_dir/recipes/replacement_v1_sn2_iteration.phases" "$label"
