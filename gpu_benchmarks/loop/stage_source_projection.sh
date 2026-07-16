#!/usr/bin/env bash
set -euo pipefail

repo="${1:?usage: stage_source_projection.sh REPO DESTINATION}"
destination="${2:?usage: stage_source_projection.sh REPO DESTINATION}"
git -C "$repo" rev-parse --git-dir >/dev/null
[[ ! -e "$destination" ]] || { echo "projection destination already exists: $destination" >&2; exit 2; }

scratch="$(mktemp -d "${TMPDIR:-/tmp}/stwo-source-projection.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
git -C "$repo" ls-files --cached --others --exclude-standard -z -- . \
  ':(exclude)gpu_benchmarks/loop/results/**' \
  ':(exclude)gpu_benchmarks/loop/ledger.jsonl' \
  ':(exclude)gpu_benchmarks/loop/pod.conf' \
  ':(exclude)gpu_benchmarks/pie/sn/**' \
  ':(exclude)gpu_benchmarks/pie/*.zip' \
  ':(exclude)gpu_benchmarks/results/**' \
  ':(exclude)gpu_benchmarks/fleet/results/**' \
  ':(exclude)gpu_benchmarks/fleet/fleet_report.json' \
  ':(exclude)gpu_benchmarks/fleet/fleet.conf' \
  ':(exclude)gpu_benchmarks/fleet/ledger_costs.jsonl' \
  ':(exclude)gpu_benchmarks/fleet/pods.conf*' >"$scratch/candidates"
: >"$scratch/files"
while IFS= read -r -d '' path; do
  if [[ -f "$repo/$path" || -L "$repo/$path" ]]; then
    printf '%s\0' "$path" >>"$scratch/files"
  elif [[ -e "$repo/$path" ]]; then
    echo "unsupported source-projection path: $repo/$path" >&2
    exit 1
  fi
done <"$scratch/candidates"

mkdir -p "$destination"
rsync -a --from0 --files-from="$scratch/files" "$repo/" "$destination/"
