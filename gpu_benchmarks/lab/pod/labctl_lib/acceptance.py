"""Interactive shell and readiness/profile acceptance evidence."""

from __future__ import annotations

import math
import os
import re
import shlex
import time

from . import common as c
from . import persistence
from . import runtime

NSYS_PACKAGE = "nsight-systems-2026.1.3"
NSYS_VERSION = "2026.1.3.425-261338342291v0"
NSYS_IDENTITY = f"{NSYS_PACKAGE}={NSYS_VERSION}"
DEV_LAYOUT_IDENTITY = "uid1000-gid1000-root0710-build0700-fixtures0700"


def _shell_argv(ep: c.Endpoint) -> list[str]:
    return ["ssh", *c.SSH_OPTS, "-p", str(ep.port), f"dev@{ep.host}"]


def cmd_shell(_args) -> int:
    with c._lease_lock():
        _, _, ep = runtime._active()
    runtime._touch_heartbeat(ep)
    os.execvp("ssh", _shell_argv(ep))


def cmd_heartbeat(_args) -> int:
    with c._lease_lock():
        state, _, ep = runtime._active()
    runtime._touch_heartbeat(ep)
    print(f"HEARTBEAT {state['pod_id']}; idle deadline renewed")
    return 0


COUNTER_CMD = r"""
cat > /tmp/labctl-counter.cu <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void labctl_counter(int *p) { if (threadIdx.x == 0) *p = 1; }
int main() {
  int *p = nullptr;
  int value = 0;
  if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return 1;
  if (cudaMemset(p, 0, sizeof(int)) != cudaSuccess) return 2;
  labctl_counter<<<1, 32>>>(p);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  if (cudaMemcpy(&value, p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
  cudaFree(p);
  if (value != 1) return 5;
  std::printf("LABCTL_KERNEL_RESULT=1\n");
  return 0;
}
CU
trap 'rm -f /tmp/labctl-counter /tmp/labctl-counter.cu /tmp/labctl-counter.log' EXIT
nvcc -O2 -lineinfo /tmp/labctl-counter.cu -o /tmp/labctl-counter
/tmp/labctl-counter
ncu --target-processes all --kernel-name regex:labctl_counter \
  --metrics sm__cycles_elapsed.avg --csv \
  /tmp/labctl-counter > /tmp/labctl-counter.log 2>&1
cat /tmp/labctl-counter.log
python3 - /tmp/labctl-counter.log <<'PY'
import csv
import math
import pathlib
import sys

metric = "sm__cycles_elapsed.avg"
kernel_marker = "labctl_counter"
rows = csv.reader(pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines())
values = []
columns = None
for row in rows:
    if all(name in row for name in ("Kernel Name", "Metric Name", "Metric Value")):
        columns = {
            "kernel": row.index("Kernel Name"),
            "metric": row.index("Metric Name"),
            "value": row.index("Metric Value"),
        }
        continue
    if columns is None or max(columns.values()) >= len(row):
        continue
    if row[columns["metric"]] != metric or kernel_marker not in row[columns["kernel"]]:
        continue
    try:
        value = float(row[columns["value"]].replace(",", ""))
    except ValueError:
        continue
    if math.isfinite(value) and value > 0:
        values.append(value)
if len(values) != 1:
    raise SystemExit(
        f"expected one positive numeric {metric} row for {kernel_marker}, got {len(values)}"
    )
print(f"LABCTL_NCU_METRIC={values[0]:.17g}")
PY
"""


def _accept_command(state: dict, *, profile: bool) -> str:
    mode = "profile" if profile else "correctness"
    local_root = persistence.local_root(state["pod_id"])
    root = f"{local_root}/records"
    persistent_probe = f"/workspace/gpu-lab/leases/{state['pod_id']}/.accept-io"
    body = f"""
set -eu
export PATH=/opt/cmake/bin:/opt/rust/cargo/bin:/opt/cargo/bin:/usr/local/cuda/bin:$PATH
test -z "${{RUNPOD_POD_ID:-}}" || \
  test "$RUNPOD_POD_ID" = {shlex.quote(state['pod_id'])}
mountpoint -q /workspace
test "$(cat /workspace/gpu-lab/NETWORK_VOLUME_ID)" = {shlex.quote(state['volume_id'])}
test -f {shlex.quote(local_root)}/LOCAL_ROOT.json
test ! -e {shlex.quote(local_root)}/SEALED
test "$(stat -c '%u:%g:%a' /tmp/stwo-gpu-lab)" = 0:0:711
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)})" = 0:1000:710
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)}/records)" = 0:0:700
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)}/build)" = 1000:1000:700
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)}/fixtures)" = 1000:1000:700
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)}/LOCAL_ROOT.json)" = 0:0:444
test "$(stat -c '%u:%g:%a' {shlex.quote(local_root)}/DEV_LAYOUT.json)" = 0:0:444
python3 - {shlex.quote(local_root)} <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
actual = json.loads((root / "DEV_LAYOUT.json").read_text())
expected = {{
    "schema_version": "stwo.gpu-lab.dev-layout.v1",
    "dev_uid": 1000,
    "dev_gid": 1000,
    "local_root": str(root),
    "local_root_mode": "0710",
    "records_root": str(root / "records"),
    "records_uid": 0,
    "records_gid": 0,
    "records_mode": "0700",
    "build_root": str(root / "build"),
    "build_uid": 1000,
    "build_gid": 1000,
    "build_mode": "0700",
    "fixtures_root": str(root / "fixtures"),
    "fixtures_uid": 1000,
    "fixtures_gid": 1000,
    "fixtures_mode": "0700",
}}
if actual != expected:
    raise SystemExit("active dev-layout marker differs")
PY
ROOT={shlex.quote(root)}
mkdir -p "$ROOT"
OUT=$(mktemp "$ROOT/accept-{mode}-XXXXXX.txt")
(
  set -eu
  echo LABCTL_POD_ID={shlex.quote(state['pod_id'])}
  echo LABCTL_PROVIDER_POD_ENV=${{RUNPOD_POD_ID:-unavailable}}
  echo LABCTL_IMAGE={shlex.quote(state['image'])}
  echo LABCTL_VOLUME_ID={shlex.quote(state['volume_id'])}
  echo LABCTL_DEV_LAYOUT={DEV_LAYOUT_IDENTITY}
  findmnt -n -o SOURCE,FSTYPE,TARGET /workspace
  df -BG /workspace {shlex.quote(local_root)}
  nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,compute_cap,persistence_mode,mig.mode.current,ecc.mode.current,clocks.current.graphics,clocks.current.memory,power.limit,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
  command -v nvcc
  command -v ninja
  command -v cmake
  command -v compute-sanitizer
  command -v ncu
  command -v nsys
  test "$(dpkg-query -W -f='${{Version}}' {NSYS_PACKAGE})" = {shlex.quote(NSYS_VERSION)}
  echo LABCTL_NSYS_PACKAGE={shlex.quote(NSYS_IDENTITY)}
  nsys --version
  ncu --query-metrics >/dev/null
  dd if=/dev/zero of={shlex.quote(persistent_probe)}-$$ bs=16M count=4 conv=fsync 2>&1
  rm -f {shlex.quote(persistent_probe)}-$$
  dd if=/dev/zero of={shlex.quote(local_root)}/.local-io-$$ bs=16M count=4 conv=fsync 2>&1
  rm -f {shlex.quote(local_root)}/.local-io-$$
"""
    if profile:
        body += COUNTER_CMD
    body += r"""
) > "$OUT" 2>&1 || {
  rc=$?
  sync -f "$OUT"
  chmod 400 "$OUT"
  sync -f "$ROOT"
  cat "$OUT"
  echo "LABCTL_RECORD=$OUT"
  echo "LABCTL_RECORD_SHA256=$(sha256sum "$OUT" | awk '{print $1}')"
  exit "$rc"
}
sync -f "$OUT"
chmod 400 "$OUT"
sync -f "$ROOT"
cat "$OUT"
echo "LABCTL_RECORD=$OUT"
echo "LABCTL_RECORD_SHA256=$(sha256sum "$OUT" | awk '{print $1}')"
"""
    body += persistence.command(
        state["pod_id"], state["volume_id"], freeze=False,
        reason=f"accept-{mode}",
    )
    return body


def _parse_acceptance(
    output: str, *, profile: bool, pod_id: str | None = None
) -> tuple[str, str, float | None, dict[str, object]]:
    record = re.search(r"^LABCTL_RECORD=(.+)$", output, re.MULTILINE)
    digest = re.search(
        r"^LABCTL_RECORD_SHA256=([0-9a-f]{64})$", output, re.MULTILINE
    )
    if not record or not digest:
        raise RuntimeError("acceptance did not produce append-only record identity")
    if not re.search(
        rf"^LABCTL_NSYS_PACKAGE={re.escape(NSYS_IDENTITY)}$", output, re.MULTILINE
    ):
        raise RuntimeError("acceptance did not prove the pinned Nsight Systems package")
    if not re.search(
        rf"^LABCTL_DEV_LAYOUT={re.escape(DEV_LAYOUT_IDENTITY)}$", output, re.MULTILINE
    ):
        raise RuntimeError("acceptance did not prove the unprivileged development layout")
    metric = None
    if profile:
        if not re.search(r"^LABCTL_KERNEL_RESULT=1$", output, re.MULTILINE):
            raise RuntimeError("profile acceptance did not validate the CUDA kernel result")
        match = re.search(r"^LABCTL_NCU_METRIC=([^\s]+)$", output, re.MULTILINE)
        if not match:
            raise RuntimeError("profile acceptance emitted no parsed ncu metric")
        try:
            metric = float(match.group(1))
        except ValueError as error:
            raise RuntimeError("profile acceptance emitted a nonnumeric ncu metric") from error
        if not math.isfinite(metric) or metric <= 0:
            raise RuntimeError("profile acceptance emitted a nonpositive ncu metric")
    return (record.group(1), digest.group(1), metric,
            persistence.parse(output, sealed=False, pod_id=pod_id))


def cmd_accept(args) -> int:
    with c._lease_lock():
        state, pod, ep = runtime._active()
    lease_identity = (
        state["created_at"],
        state["expires_at"],
        state["lease_name"],
        state["pod_id"],
    )
    mode = "profile" if args.profile else "correctness"
    runtime._touch_heartbeat(ep)
    rc, output = c.ssh_capture(ep, _accept_command(state, profile=args.profile), timeout=600)
    print(output)
    if rc:
        return rc
    record, digest, metric, persisted = _parse_acceptance(
        output, profile=args.profile, pod_id=state["pod_id"]
    )
    acceptance = {
        "at": time.time(),
        "mode": mode,
        "ncu_sm_cycles_elapsed_avg": metric,
        "record": record,
        "record_sha256": digest,
        "persistence": persisted,
    }
    with c._lease_lock():
        current, current_pod, _ = runtime._active()
        current_identity = (
            current["created_at"],
            current["expires_at"],
            current["lease_name"],
            current["pod_id"],
        )
        if current_identity != lease_identity or current_pod.id != pod.id:
            raise RuntimeError("lease changed while acceptance was running")
        current["accepted"].append(acceptance)
        c._write_state(current)
    c.ledger.append(
        "run",
        pod_id=pod.id,
        gpu=pod.gpu,
        usd_hr=pod.cost_per_hr,
        purpose=f"gpu-lab-accept-{mode}",
        note="PASS",
    )
    print(f"ACCEPTED {mode}: {digest} ({record})")
    return 0
