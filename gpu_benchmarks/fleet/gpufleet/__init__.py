"""gpufleet — formalized RunPod orchestration for the stwo GPU proving program.

Design contract (why this exists instead of more shell):
  1. MAXIMUM VALUE PER GPU DOLLAR. Nothing provisions until `pregate` (the local
     no-GPU battery) is green; every remote session is a declarative manifest with
     machine-checked pass/fail criteria, so a pod never sits idle waiting for a
     human (or an agent) to decide the next step.
  2. NO SILENT SPEND. Every pod carries an on-pod TTL + idle deadman (self-stop,
     no credentials placed on the pod); every lifecycle event and run lands in a
     cost ledger; `up` enforces a price ceiling; `run` stops the pod when the
     manifest completes unless told otherwise.
  3. NO SILENT GPUS. The monitor samples utilization during every step and flags
     a GPU-bound step that shows an idle device (the failure mode we once caught
     only by eye).
  4. REPRODUCIBILITY. Run records carry both repos' git SHA + working-diff hash,
     the env, and artifact hashes.

Stdlib only (urllib for GraphQL, tomllib for config/manifests). Python >= 3.11.
"""

__version__ = "0.1.0"

# Provisioning conventions (mirrors fleet/pod_provision.sh, measured round 9+).
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_CLOUD = "SECURE"
DEFAULT_DISK_GB = 80
DEFAULT_VOLUME_GB = 120
VOLUME_MOUNT = "/workspace"
DEFAULT_MAX_USD_HR = 3.0
DEFAULT_TTL_HOURS = 6.0
DEFAULT_IDLE_STOP_MIN = 45
HEARTBEAT_PATH = "/tmp/gpufleet.heartbeat"

# GPU shortname -> RunPod gpuTypeId (extend as the program touches new silicon).
GPU_TYPE_IDS = {
    "3090": "NVIDIA GeForce RTX 3090",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
    "a40": "NVIDIA A40",
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-pcie": "NVIDIA H100 PCIe",
    "h200": "NVIDIA H200",
    "l40s": "NVIDIA L40S",
}
