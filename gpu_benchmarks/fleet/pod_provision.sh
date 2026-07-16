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
#   create [...]                  RETIRED: fails closed; use `gpufleet up --recipe`.
#   ssh-info <id>                 `runpodctl ssh info <id>` (ip/port/key JSON).
#   terminate <id> [<id>...]      `runpodctl pod delete <id>` for each id (DESTRUCTIVE).
#
# Usage:
#   ./pod_provision.sh list
#   ./pod_provision.sh gpus
#   ./gpufleet.sh up --recipe ../loop/recipes/<sealed-worker>.phases
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
  die "create is retired; use ./gpufleet.sh up --recipe <sealed-worker.phases>"
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
