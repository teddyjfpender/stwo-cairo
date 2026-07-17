"""Content-addressed local-to-persistent-volume worktree synchronization."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import common as c
from . import bootstrap_profile
from . import generation
from . import lease_local_root
from . import runtime


TREE_ID_SCRIPT = r"""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
excludes = tuple(json.loads(sys.argv[2]))

def git_environment():
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "5",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.untrackedCache",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "core.preloadIndex",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "core.hooksPath",
        "GIT_CONFIG_VALUE_3": "/dev/null",
        "GIT_CONFIG_KEY_4": "core.excludesFile",
        "GIT_CONFIG_VALUE_4": "/dev/null",
    })
    return environment

def index_inventory():
    dot_git = repo / ".git"
    if dot_git.is_dir():
        index = dot_git / "index"
    else:
        prefix, location = dot_git.read_text().strip().split(": ", 1)
        if prefix != "gitdir":
            raise SystemExit("invalid Git worktree control path")
        index = (dot_git.parent / location).resolve() / "index"
    info = index.lstat()
    digest = hashlib.sha256()
    with open(index, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return (
        info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
        info.st_size, info.st_mtime_ns, digest.digest(),
    )

def git(*args):
    before = index_inventory()
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        check=False, capture_output=True, env=git_environment()
    )
    if index_inventory() != before:
        raise SystemExit(f"read-only Git command mutated index: {args}")
    result.check_returncode()
    return result.stdout

def index_matches_head():
    index = []
    for raw in git("ls-files", "--stage", "-z").split(b"\0"):
        if not raw:
            continue
        metadata, path = raw.split(b"\t", 1)
        mode, object_id, stage = metadata.split(b" ")
        if stage != b"0":
            return False
        index.append((mode, object_id, path))
    tree = []
    for raw in git("ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
        if not raw:
            continue
        metadata, path = raw.split(b"\t", 1)
        mode, _, object_id = metadata.split(b" ")
        tree.append((mode, object_id, path))
    return index == tree

if len(sys.argv) > 3 and sys.argv[3] == "require-clean-index" and not index_matches_head():
    raise SystemExit("Git index differs from HEAD")

head = git("rev-parse", "HEAD").decode().strip()
object_format = git("rev-parse", "--show-object-format").decode().strip()
if object_format not in ("sha1", "sha256"):
    raise SystemExit("unsupported Git object format")
tracked = git("ls-files", "--cached", "-z").split(b"\0")
untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
head_entries = {}
for raw in git("ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
    if not raw:
        continue
    metadata, path = raw.split(b"\t", 1)
    mode, kind, object_id = metadata.split(b" ")
    if kind not in (b"blob", b"commit"):
        raise SystemExit("unsupported Git tree object")
    head_entries[path.decode()] = (mode.decode(), object_id.decode())

def included(rel):
    return (
        "__pycache__" not in rel
        and not any(rel.startswith(prefix) for prefix in excludes)
    )

all_names = sorted({
    raw.decode()
    for raw in tracked + untracked if raw and included(raw.decode())
} | {rel for rel in head_entries if included(rel)})

snapshots = {}
def snapshot(rel):
    if rel in snapshots:
        return snapshots[rel]
    path = repo / rel
    if not os.path.lexists(path):
        result = ({"kind": "deleted", "path": rel}, None)
        snapshots[rel] = result
        return result
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        payload = os.fsencode(os.readlink(path))
        kind = "symlink"
        mode = 0o777
        git_mode = "120000"
    elif stat.S_ISREG(info.st_mode):
        mode = 0o755 if info.st_mode & 0o111 else 0o644
        git_mode = "100755" if info.st_mode & 0o111 else "100644"
        digest = hashlib.sha256()
        object_digest = hashlib.new(object_format)
        object_digest.update(f"blob {info.st_size}\0".encode())
        size = 0
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                object_digest.update(block)
                size += len(block)
        after = path.lstat()
        before_meta = (
            info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns,
        )
        after_meta = (
            after.st_ino, after.st_mode, after.st_uid, after.st_gid, after.st_nlink,
            after.st_size, after.st_mtime_ns,
        )
        if size != info.st_size or after_meta != before_meta:
            raise SystemExit(f"worktree path raced during identity: {rel}")
        result = ({
            "kind": "file",
            "mode": mode,
            "path": rel,
            "sha256": digest.hexdigest(),
            "size": info.st_size,
        }, (git_mode, object_digest.hexdigest()))
        snapshots[rel] = result
        return result
    else:
        raise SystemExit(f"unsupported changed path type: {rel}")
    object_digest = hashlib.new(object_format)
    object_digest.update(f"blob {len(payload)}\0".encode())
    object_digest.update(payload)
    result = ({
        "kind": kind,
        "mode": mode,
        "path": rel,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }, (git_mode, object_digest.hexdigest()))
    snapshots[rel] = result
    return result

dirty_names = [
    rel for rel in all_names if snapshot(rel)[1] != head_entries.get(rel)
]
def describe(rel):
    return snapshot(rel)[0]

entries = [describe(rel) for rel in dirty_names]
tree_digest = hashlib.sha256()
for rel in all_names:
    encoded_entry = json.dumps(
        describe(rel), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    tree_digest.update(len(encoded_entry).to_bytes(8, "big"))
    tree_digest.update(encoded_entry)
core = {
    "entries": entries,
    "head": head,
    "tree_content_sha256": tree_digest.hexdigest(),
}
encoded = json.dumps(core, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
core["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
print(json.dumps(core, allow_nan=False, sort_keys=True))
"""


def _parse_tree_identity(output: str, *, label: str) -> dict:
    try:
        identity = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {label} tree identity: {output[-300:]}") from error
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(identity.get("head", ""))):
        raise RuntimeError(f"invalid {label} HEAD identity")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("identity_sha256", ""))):
        raise RuntimeError(f"invalid {label} worktree identity")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("tree_content_sha256", ""))):
        raise RuntimeError(f"invalid {label} full-tree identity")
    if not isinstance(identity.get("entries"), list):
        raise RuntimeError(f"invalid {label} worktree entries")
    previous = None
    for item in identity["entries"]:
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid {label} worktree entry")
        path = item.get("path")
        kind = item.get("kind")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or kind not in {"deleted", "file", "symlink"}
            or (previous is not None and path <= previous)
        ):
            raise RuntimeError(f"unsafe/noncanonical {label} worktree entry: {item}")
        if kind != "deleted":
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                raise RuntimeError(f"invalid {label} content digest: {path}")
            if not isinstance(item.get("mode"), int) or not isinstance(item.get("size"), int):
                raise RuntimeError(f"invalid {label} content metadata: {path}")
        previous = path
    core = {
        "entries": identity["entries"],
        "head": identity["head"],
        "tree_content_sha256": identity["tree_content_sha256"],
    }
    encoded = json.dumps(
        core, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if identity["identity_sha256"] != hashlib.sha256(encoded).hexdigest():
        raise RuntimeError(f"self-inconsistent {label} worktree identity")
    return identity


def _local_tree_identity(repo: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", TREE_ID_SCRIPT, str(repo), json.dumps(c.SYNC_EXCLUDES)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_tree_identity(proc.stdout, label=str(repo))


def _remote_tree_identity(ep: c.Endpoint, repo: str) -> dict:
    identity_command = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(TREE_ID_SCRIPT),
            shlex.quote(repo),
            shlex.quote(json.dumps(c.SYNC_EXCLUDES)),
            "require-clean-index",
        )
    )
    rc, output = c.ssh_capture(ep, identity_command, timeout=180)
    if rc:
        raise RuntimeError(
            f"remote checkout missing/invalid at {repo}; seed exact HEAD first: {output}"
        )
    return _parse_tree_identity(output, label=repo)


REMOTE_RECORD_SCRIPT = r"""
import base64
import hashlib
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]
payload = base64.b64decode(sys.argv[3], validate=True)
root.mkdir(parents=True, exist_ok=True)
fd, path = tempfile.mkstemp(prefix=prefix + "-", suffix=".json", dir=root)
with os.fdopen(fd, "wb") as out:
    out.write(payload)
    out.flush()
    os.fsync(out.fileno())
os.chmod(path, 0o400)
directory = os.open(root, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(f"LABCTL_RECORD={path}")
print(f"LABCTL_RECORD_SHA256={hashlib.sha256(payload).hexdigest()}")
"""


def _append_remote_record(
    ep: c.Endpoint, state: dict, prefix: str, payload: dict
) -> tuple[str, str]:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    root = f"/workspace/gpu-lab/leases/{state['pod_id']}/records"
    command = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(REMOTE_RECORD_SCRIPT),
            shlex.quote(root),
            shlex.quote(prefix),
            shlex.quote(base64.b64encode(encoded).decode()),
        )
    )
    rc, output = c.ssh_capture(ep, command, timeout=60)
    if rc:
        raise RuntimeError(f"failed to append remote evidence: {output}")
    lines = output.splitlines()
    path_match = re.fullmatch(r"LABCTL_RECORD=(.+)", lines[0]) if len(lines) == 2 else None
    hash_match = (
        re.fullmatch(r"LABCTL_RECORD_SHA256=([0-9a-f]{64})", lines[1])
        if len(lines) == 2
        else None
    )
    if not path_match or not hash_match:
        raise RuntimeError(f"invalid remote evidence response: {output}")
    path, digest = path_match.group(1), hash_match.group(1)
    if (
        Path(path).parent != Path(root)
        or not Path(path).name.startswith(prefix + "-")
        or digest != hashlib.sha256(encoded).hexdigest()
    ):
        raise RuntimeError(f"mismatched remote evidence receipt: {output}")
    return path, digest


def _verify_remote_volume(ep: c.Endpoint, state: dict) -> None:
    if bootstrap_profile.is_lease_local_state(state):
        command = lease_local_root.attestation_command(
            state["pod_id"],
            state["volume_id"],
            state["lease_local_root"]["boot_id"],
            guards=False,
        )
        rc, output = c.ssh_capture(ep, command, timeout=30)
        expected = f"LABCTL_LEASE_LOCAL_ATTESTED={state['pod_id']}"
        if rc or output.strip() != expected:
            raise RuntimeError(
                "quarantined provider-volume verification failed: " + output[-500:]
            )
        return
    command = f"""
set -eu
mountpoint -q /workspace
test "$(cat /workspace/gpu-lab/NETWORK_VOLUME_ID)" = {shlex.quote(state['volume_id'])}
findmnt -n -o SOURCE,FSTYPE,TARGET /workspace
"""
    rc, output = c.ssh_capture(ep, command, timeout=30)
    if rc:
        raise RuntimeError(f"persistent /workspace verification failed: {output}")


def _seed_specs(state: dict) -> tuple[tuple[str, Path, str], ...]:
    seed_root = (
        lease_local_root.PROVIDER_MOUNT
        if bootstrap_profile.is_lease_local_state(state)
        else "/workspace"
    )
    return (
        ("stwo", c.STWO, f"{seed_root}/src/stwo"),
        ("stwo-cairo", c.STWO_CAIRO, f"{seed_root}/src/stwo-cairo"),
    )


def _require_same_identity(
    repo: Path, expected: dict, actual: dict, phase: str
) -> None:
    if actual != expected:
        raise RuntimeError(f"{repo.name} identity changed {phase}; refusing evidence")


def cmd_sync(_args) -> int:
    with c._lease_lock():
        state, _, ep = runtime._active()
    lease_identity = (
        state["created_at"],
        state["expires_at"],
        state["lease_name"],
        state["pod_id"],
    )
    runtime._touch_heartbeat(ep)
    _verify_remote_volume(ep, state)
    specs = _seed_specs(state)
    local_identities = {name: _local_tree_identity(repo) for name, repo, _ in specs}
    seeds = {name: remote for name, _, remote in specs}
    seed_identities = {
        name: _remote_tree_identity(ep, remote) for name, _, remote in specs
    }
    expected_current = generation.current(ep)
    total = sum(len(item["entries"]) for item in local_identities.values())
    with tempfile.TemporaryDirectory(prefix="labctl-generation-") as directory:
        bundle_dir = Path(directory)
        local_bundles = {}
        local_overlays = {}
        for name, repo, _ in specs:
            bundle = generation.create_bundle(
                repo,
                seed_identities[name]["head"],
                local_identities[name]["head"],
                bundle_dir,
                name,
            )
            if bundle:
                local_bundles[name] = bundle
            overlay = generation.create_overlay(
                repo, local_identities[name], bundle_dir, name
            )
            if overlay:
                local_overlays[name] = overlay
        estimated = generation.estimate_transaction(
            [repo for _, repo, _ in specs], local_bundles, local_overlays
        )
        _, transaction = generation.prepare(ep, *estimated)
        bundles = {
            name: generation.transfer_bundle(ep, transaction, name, bundle)
            for name, bundle in local_bundles.items()
        }
        overlays = {
            name: generation.transfer_overlay(ep, transaction, name, overlay)
            for name, overlay in local_overlays.items()
        }
        generation.stage(
            ep,
            transaction,
            seeds,
            seed_identities,
            bundles,
            {name: item["head"] for name, item in local_identities.items()},
            TREE_ID_SCRIPT,
            c.SYNC_EXCLUDES,
        )
        for name, repo, _ in specs:
            _require_same_identity(
                repo,
                local_identities[name],
                _local_tree_identity(repo),
                "before generation finalize",
            )
        generation_payload, generation_sha = generation.generation_document(
            local_identities
        )
        publication = generation.finalize(
            ep,
            generation.SOURCE_ROOT,
            transaction,
            seeds,
            seed_identities,
            local_identities,
            overlays,
            generation_payload,
            generation_sha,
            expected_current,
            TREE_ID_SCRIPT,
            c.SYNC_EXCLUDES,
        )
    for name, repo, _ in specs:
        _require_same_identity(
            repo,
            local_identities[name],
            _local_tree_identity(repo),
            "after generation publication",
        )
    generation.attest(
        ep,
        generation.SOURCE_ROOT,
        publication,
        local_identities,
        TREE_ID_SCRIPT,
        c.SYNC_EXCLUDES,
    )
    records = []
    for name, repo, _ in specs:
        records.append(
            {
                "generation_path": f"{publication['generation_path']}/repos/{name}",
                "head": local_identities[name]["head"],
                "repo": repo.name,
                "tree_content_sha256": local_identities[name]["tree_content_sha256"],
                "worktree_identity_sha256": local_identities[name]["identity_sha256"],
            }
        )
    for name, repo, _ in specs:
        _require_same_identity(
            repo,
            local_identities[name],
            _local_tree_identity(repo),
            "before evidence",
        )
    generation.attest(
        ep,
        generation.SOURCE_ROOT,
        publication,
        local_identities,
        TREE_ID_SCRIPT,
        c.SYNC_EXCLUDES,
    )
    record, digest = _append_remote_record(
        ep,
        state,
        "sync",
        {
            "at": time.time(),
            "pod_id": state["pod_id"],
            "repositories": records,
            "schema_version": "stwo.gpu-lab.sync-record.v2",
            "source_generation": publication,
        },
    )
    generation.cleanup(
        ep,
        generation.SOURCE_ROOT,
        transaction,
        publication,
        record,
        digest,
    )
    with c._lease_lock():
        current, _, _ = runtime._active()
        current_identity = (
            current["created_at"],
            current["expires_at"],
            current["lease_name"],
            current["pod_id"],
        )
        if current_identity != lease_identity:
            raise RuntimeError("lease changed while sync was running")
    print(f"synced {total} changed/deleted files")
    print(f"evidence {record} sha256={digest}")
    return 0
