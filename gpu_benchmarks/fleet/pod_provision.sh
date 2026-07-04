#!/usr/bin/env bash
#
# pod_provision.sh — thin, auditable wrappers around `runpodctl` for standing up,
# listing, and tearing down the pods that fleet.sh drives. It prints the exact
# runpodctl command it will run, honors DRY_RUN, and never hides a destructive call.
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ PROCUREMENT NOTE — CPU CORES FIRST.                                          │
# │                                                                             │
# │ On the current (pre-W3) build the HOST witness write dominates the prove:   │
# │ round-8 measured it at 40–66% of prove_cairo on a real SN PIE, and the      │
# │ 66% figure was partly an artifact of a THIN ~7-vCPU container (RESULTS.md    │
# │ round 8, honest caveats). useful_mhz is therefore gated by host CPU cores /  │
# │ memory bandwidth as much as by the GPU until witness-on-GPU (W3) lands.      │
# │                                                                             │
# │ So when provisioning: for a given GPU, prefer the offering with the MOST     │
# │ host vCPUs and RAM. runpodctl exposes no direct --vcpu knob for GPU pods     │
# │ (vCPU/RAM scale with the host and gpu-count), so:                            │
# │   1. compare host specs in `runpodctl gpu list` / the web console before     │
# │      committing, and prefer datacenters/instances with fat hosts;           │
# │   2. after boot, VERIFY with `nproc` and `free -g` over ssh — a 7-vCPU box   │
# │      will quietly halve your useful_mhz vs a 30+-vCPU box on the same GPU;   │
# │   3. RAM must clear the working set: 8M-step ~63 GB, 14M-step SN PIEs need    │
# │      >125 GB host RAM (they OOM'd a smaller box) — see RESULTS round 7/8.    │
# │ Record the pod's real vCPU/RAM in fleet.conf's gpu label if it matters.      │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Subcommands:
#   list                          `runpodctl pod list` (all your pods).
#   gpus                          `runpodctl gpu list` (available GPU types + ids).
#   create --gpu "<id>" [opts]    Create pod(s). --count N creates N pods.
#   ssh-info <id>                 `runpodctl ssh info <id>` (ip/port/key JSON).
#   terminate <id> [<id>...]      `runpodctl pod delete <id>` for each id (DESTRUCTIVE).
#
# create options:
#   --gpu "<gpu-id>"    GPU id/name from `runpodctl gpu list` (e.g.
#                       "NVIDIA GeForce RTX 4090"). REQUIRED.
#   --count N           Number of pods to create (default 1). Each is created with
#                       a distinct name suffix.
#   --name NAME         Base pod name (default "stwo-fleet"). Suffixed -1.. with --count.
#   --gpu-count N       GPUs per pod (default 1). Higher counts usually mean a bigger
#                       host = more vCPUs (see the procurement note).
#   --image IMG         Docker image (default a CUDA 11.8 devel image).
#   --template-id ID    Use a RunPod template instead of --image.
#   --disk GB           Container disk size in GB (default 60).
#   --volume GB         Persistent volume size in GB (default 200, mounted /workspace).
#   --cloud TYPE        SECURE (default) or COMMUNITY.
#   --ports SPEC        Ports (default "22/tcp").
#   --extra "ARGS"      Extra args appended verbatim to `runpodctl pod create`.
#
# Usage:
#   ./pod_provision.sh list
#   ./pod_provision.sh gpus
#   ./pod_provision.sh create --gpu "NVIDIA GeForce RTX 4090" --count 5
#   ./pod_provision.sh ssh-info 5gw3c25kdtmtm7
#   ./pod_provision.sh terminate 5gw3c25kdtmtm7 a1b2c3d4e5f6g7
#
# Environment:
#   DRY_RUN=1   Print the runpodctl command(s) without executing them.
#
# After `create`, add each pod to fleet.conf (id | gpu | usd_per_hr | ...), sync+build
# with `loop/bench_loop.sh --gate-only` (or `fleet.sh --prep`), then run `fleet.sh`.

set -euo pipefail

log()  { echo "[provision] $*" >&2; }
warn() { echo "[provision][WARN] $*" >&2; }
die()  { echo "[provision][FATAL] $*" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"

usage() { sed -n '2,60p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

# Run a runpodctl command, or echo it under DRY_RUN. All args are passed verbatim.
rp() {
  if ! command -v runpodctl >/dev/null 2>&1; then
    die "runpodctl not found on PATH (install it, then \`runpodctl doctor\` to set your API key)"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] runpodctl $*" >&2
    return 0
  fi
  log "runpodctl $*"
  runpodctl "$@"
}

# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------
cmd_create() {
  local gpu="" count=1 name="stwo-fleet" gpu_count=1 disk=60 volume=200
  local cloud="SECURE" ports="22/tcp" template_id="" extra=""
  local image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)         gpu="${2:?--gpu needs a value}"; shift 2 ;;
      --count)       count="${2:?--count needs a value}"; shift 2 ;;
      --name)        name="${2:?--name needs a value}"; shift 2 ;;
      --gpu-count)   gpu_count="${2:?--gpu-count needs a value}"; shift 2 ;;
      --image)       image="${2:?--image needs a value}"; shift 2 ;;
      --template-id) template_id="${2:?--template-id needs a value}"; shift 2 ;;
      --disk)        disk="${2:?--disk needs a value}"; shift 2 ;;
      --volume)      volume="${2:?--volume needs a value}"; shift 2 ;;
      --cloud)       cloud="${2:?--cloud needs a value}"; shift 2 ;;
      --ports)       ports="${2:?--ports needs a value}"; shift 2 ;;
      --extra)       extra="${2:?--extra needs a value}"; shift 2 ;;
      *)             die "create: unknown option '$1'" ;;
    esac
  done
  [[ -n "$gpu" ]] || die "create: --gpu \"<gpu-id>\" is required (see: pod_provision.sh gpus)"
  [[ "$count" =~ ^[0-9]+$ && "$count" -ge 1 ]] || die "--count must be a positive integer"
  [[ "$gpu_count" =~ ^[0-9]+$ && "$gpu_count" -ge 1 ]] || die "--gpu-count must be a positive integer"

  warn "PROCUREMENT: verify host vCPU/RAM after boot (nproc / free -g). A thin host"
  warn "throttles useful_mhz on the current build — see the note at the top of this file."

  local i pod_name
  for (( i=1; i<=count; i++ )); do
    if [[ "$count" -eq 1 ]]; then pod_name="$name"; else pod_name="${name}-${i}"; fi
    # Assemble the create argv defensively (only include flags that are set).
    local -a a=( pod create --name "$pod_name" --gpu-count "$gpu_count"
                 --cloud-type "$cloud" --ports "$ports"
                 --container-disk-in-gb "$disk" --volume-in-gb "$volume"
                 --volume-mount-path "/workspace" --ssh )
    if [[ -n "$template_id" ]]; then
      a+=( --template-id "$template_id" --gpu-id "$gpu" )
    else
      a+=( --image "$image" --gpu-id "$gpu" )
    fi
    # shellcheck disable=SC2206  # intentional word-split of user-provided --extra
    [[ -n "$extra" ]] && a+=( $extra )
    log "creating pod '${pod_name}' (${gpu} x${gpu_count}) ..."
    rp "${a[@]}"
  done
  log "done. Add the new pod id(s) to fleet.conf, then prep with fleet.sh --prep."
}

# ---------------------------------------------------------------------------
# terminate (destructive — confirm intent by requiring explicit ids)
# ---------------------------------------------------------------------------
cmd_terminate() {
  [[ $# -ge 1 ]] || die "terminate: need at least one pod id"
  local id
  for id in "$@"; do
    [[ "$id" =~ ^[A-Za-z0-9_-]+$ ]] || die "terminate: implausible pod id '${id}'"
  done
  warn "TERMINATE is destructive and irreversible for: $*"
  for id in "$@"; do
    log "deleting pod '${id}' ..."
    rp pod delete "$id"
  done
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
[[ $# -ge 1 ]] || usage 1
sub="$1"; shift
case "$sub" in
  list)       rp pod list ;;
  gpus)       rp gpu list ;;
  create)     cmd_create "$@" ;;
  ssh-info)   [[ $# -eq 1 ]] || die "ssh-info: need exactly one pod id"; rp ssh info "$1" ;;
  terminate)  cmd_terminate "$@" ;;
  -h|--help)  usage 0 ;;
  *)          die "unknown subcommand '${sub}' (try --help)" ;;
esac
