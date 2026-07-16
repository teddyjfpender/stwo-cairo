"""Fail-closed local-NVMe persistence before acceptance or termination."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from . import common as c


PERSIST_WORKER = r'''
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pod_id, volume_id, local_name, workspace_name, freeze_raw, reason = sys.argv[1:]
local = Path(local_name).resolve()
workspace = Path(workspace_name).resolve()
freeze = freeze_raw == "1"

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def regular(path, label):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")

def atomic_write(path, payload, mode=0o400):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".persist-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

def install(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(workspace):
        raise RuntimeError(f"persistent destination escapes /workspace: {destination}")
    if destination.exists():
        regular(destination, "persistent object")
        if digest(destination) != expected:
            raise RuntimeError(f"immutable persistent object changed: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=".object-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        if digest(temporary) != expected or digest(source) != expected:
            raise RuntimeError(f"source changed during persistence: {source}")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    regular(destination, "persistent object")
    if digest(destination) != expected:
        raise RuntimeError(f"post-copy verification failed: {destination}")
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

if not local.is_absolute() or not workspace.is_absolute() or workspace == local:
    raise RuntimeError("persistence roots must be distinct absolute paths")
if workspace in local.parents or local in workspace.parents:
    raise RuntimeError("local and persistent roots must not be nested")
marker_path = local / "LOCAL_ROOT.json"
regular(marker_path, "local-root marker")
def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate marker key: {key}")
        value[key] = item
    return value
marker = json.loads(marker_path.read_text(), object_pairs_hook=unique_object,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        RuntimeError(f"non-finite marker value: {value}")))
expected_marker = {
    "schema_version", "pod_id", "volume_id", "local_root", "local_mount", "local_device",
    "workspace_mount", "workspace_device",
}
if set(marker) != expected_marker or marker["schema_version"] != "stwo.gpu-lab.local-root.v1":
    raise RuntimeError("invalid local-root marker")
if marker["pod_id"] != pod_id or marker["volume_id"] != volume_id:
    raise RuntimeError("local-root marker belongs to another lease")
if Path(marker["local_root"]).resolve() != local:
    raise RuntimeError("local-root marker path mismatch")
if marker["local_mount"] == marker["workspace_mount"]:
    raise RuntimeError("local and network-volume mounts are indistinguishable")
if marker["local_device"] == marker["workspace_device"]:
    raise RuntimeError("local and network-volume devices are indistinguishable")

sealed = local / "SEALED"
if not freeze and sealed.exists():
    raise RuntimeError("local output root is sealed against new acceptance")
if freeze:
    if sealed.exists():
        regular(sealed, "local seal")
        if not sealed.read_text().startswith(("sealing\n", "sealed ")):
            raise RuntimeError("invalid local seal state")
    else:
        descriptor = os.open(sealed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "w") as output:
            output.write("sealing\n")
            output.flush()
            os.fsync(output.fileno())
    records = local / "records"
    records.mkdir(parents=True, exist_ok=True)
    closure = records / "closure.json"
    if not closure.exists():
        payload = json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "pod_id": pod_id,
            "reason": reason,
            "schema_version": "stwo.gpu-lab.closure.v1",
            "volume_id": volume_id,
        }, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
        atomic_write(closure, payload)

def snapshot():
    candidates = {("records", "LOCAL_ROOT.json"): marker_path}
    record_root = local / "records"
    if record_root.exists():
        if record_root.is_symlink():
            raise RuntimeError("local records root is a symlink")
        for path in record_root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"local records contain a symlink: {path}")
            if path.is_file():
                candidates[("records", str(path.relative_to(local)))] = path
    build_root = local / "build"
    if build_root.exists():
        if build_root.is_symlink():
            raise RuntimeError("local build root is a symlink")
        for path in build_root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"local build output contains a symlink: {path}")
            if not path.is_file() or "runs" not in path.parts:
                continue
            relative = path.relative_to(local)
            if "profiles" in relative.parts:
                category = "profiles"
            elif path.name in {"loop.json", "staging.json"}:
                category = "records"
            else:
                category = "results"
            candidates[(category, str(relative))] = path
    result = []
    for (category, relative), path in sorted(candidates.items()):
        regular(path, "local output")
        result.append((category, relative, path, path.stat().st_size, digest(path)))
    return result

before = snapshot()
if len(before) == 1:
    raise RuntimeError("no local records, results, or profiles exist to persist")
persistent = workspace / "gpu-lab" / "leases" / pod_id
entries = []
for category, relative, source, size, source_hash in before:
    destination = persistent / category / "sha256" / source_hash
    install(source, destination, source_hash)
    entries.append({
        "bytes": size,
        "category": category,
        "local_path": relative,
        "persistent_path": str(destination.relative_to(workspace)),
        "sha256": source_hash,
    })
after = snapshot()
before_identity = [(category, relative, size, value)
                   for category, relative, _path, size, value in before]
after_identity = [(category, relative, size, value)
                  for category, relative, _path, size, value in after]
if before_identity != after_identity:
    raise RuntimeError("local outputs changed while persistence was sealing them")

manifest = {
    "entries": entries,
    "freeze": freeze,
    "local_root": str(local),
    "pod_id": pod_id,
    "reason": reason,
    "schema_version": "stwo.gpu-lab.persistence.v1",
    "volume_id": volume_id,
}
canonical = json.dumps(manifest, allow_nan=False, sort_keys=True,
                       separators=(",", ":")).encode()
manifest_hash = hashlib.sha256(canonical).hexdigest()
manifest_payload = json.dumps(manifest, allow_nan=False, indent=2,
                              sort_keys=True).encode() + b"\n"
manifest_path = persistent / "persists" / f"{manifest_hash}.json"
if manifest_path.exists():
    regular(manifest_path, "persistence manifest")
    if manifest_path.read_bytes() != manifest_payload:
        raise RuntimeError("persistence manifest content-address collision")
else:
    atomic_write(manifest_path, manifest_payload)
atomic_write(persistent / "LATEST_PERSIST.sha256", (manifest_hash + "\n").encode())
if freeze:
    lines = [f"manifest {manifest_hash}"]
    lines += [f"{entry['sha256']}  {entry['category']}/{entry['local_path']}" for entry in entries]
    atomic_write(persistent / "SEAL.sha256", ("\n".join(lines) + "\n").encode())
    atomic_write(persistent / "CLOSED_AT", (datetime.now(timezone.utc).isoformat() + "\n").encode())
    atomic_write(sealed, f"sealed {manifest_hash}\n".encode())
print(f"LABCTL_PERSIST_MANIFEST={manifest_path}")
print(f"LABCTL_PERSIST_SHA256={manifest_hash}")
print(f"LABCTL_PERSIST_ENTRIES={len(entries)}")
print(f"LABCTL_SEALED={1 if freeze else 0}")
'''


def local_root(pod_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", pod_id):
        raise ValueError(f"unsafe pod id for local root: {pod_id!r}")
    return f"/tmp/stwo-gpu-lab/{pod_id}"


def command(pod_id: str, volume_id: str, *, freeze: bool, reason: str) -> str:
    root = local_root(pod_id)
    arguments = " ".join(shlex.quote(item) for item in (
        pod_id, volume_id, root, "/workspace", "1" if freeze else "0", reason,
    ))
    return f"""
set -eu
mountpoint -q /workspace
test "$(cat /workspace/gpu-lab/NETWORK_VOLUME_ID)" = {shlex.quote(volume_id)}
python3 - {arguments} <<'PY'
{PERSIST_WORKER}
PY
"""


def parse(output: str, *, sealed: bool, pod_id: str | None = None) -> dict[str, object]:
    manifest = re.search(r"^LABCTL_PERSIST_MANIFEST=(.+)$", output, re.MULTILINE)
    digest = re.search(r"^LABCTL_PERSIST_SHA256=([0-9a-f]{64})$", output, re.MULTILINE)
    entries = re.search(r"^LABCTL_PERSIST_ENTRIES=([0-9]+)$", output, re.MULTILINE)
    seal = re.search(r"^LABCTL_SEALED=([01])$", output, re.MULTILINE)
    if not all((manifest, digest, entries, seal)):
        raise RuntimeError("persistence did not emit a complete authenticated result")
    count = int(entries.group(1))
    if count < 1 or (seal.group(1) == "1") != sealed:
        raise RuntimeError("persistence result has an invalid entry/seal state")
    manifest_path = Path(manifest.group(1))
    if (not manifest_path.is_absolute()
            or manifest_path.name != digest.group(1) + ".json"):
        raise RuntimeError("persistence manifest path is not content addressed")
    if pod_id is not None:
        expected = Path(f"/workspace/gpu-lab/leases/{pod_id}/persists")
        if manifest_path.parent != expected:
            raise RuntimeError("persistence manifest belongs to another lease")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": digest.group(1),
        "entries": count,
        "sealed": sealed,
    }


def persist_remote(ep: c.Endpoint, state: dict, *, freeze: bool, reason: str,
                   timeout: int = 300) -> tuple[dict[str, object], str]:
    rc, output = c.ssh_capture(
        ep, command(state["pod_id"], state["volume_id"], freeze=freeze, reason=reason),
        timeout=timeout,
    )
    if rc:
        raise RuntimeError(f"remote persistence failed; compute retained: {output[-1000:]}")
    return parse(output, sealed=freeze, pod_id=state["pod_id"]), output


def marker_document(pod_id: str, volume_id: str, local_mount: str,
                    workspace_mount: str) -> dict[str, str]:
    return {
        "schema_version": "stwo.gpu-lab.local-root.v1",
        "pod_id": pod_id,
        "volume_id": volume_id,
        "local_root": local_root(pod_id),
        "local_mount": local_mount,
        "local_device": "0:1 local",
        "workspace_mount": workspace_mount,
        "workspace_device": "0:2 network",
    }


def canonical_marker(document: dict[str, str]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
