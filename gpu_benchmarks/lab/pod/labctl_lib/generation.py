"""Immutable source-generation construction and atomic publication."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import tarfile
from pathlib import Path

from . import common as c
from . import generation_cleanup_scripts as cleanup_scripts
from . import generation_scripts as scripts


SOURCE_ROOT = "/workspace/gpu-lab/source-generations"


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode()


def _json64(value) -> str:
    return _encode(json.dumps(value, allow_nan=False, sort_keys=True).encode())


def _command(script: str, *args: str) -> str:
    return " ".join(
        shlex.quote(item) for item in ("python3", "-c", script, *args)
    )


def _capture(
    ep: c.Endpoint, script: str, args: list[str], timeout: int = 180
) -> tuple[int, str]:
    return c.ssh_capture(ep, _command(script, "0", "0", *args), timeout=timeout)


def current(ep: c.Endpoint, root: str = SOURCE_ROOT) -> str | None:
    rc, output = _capture(ep, scripts.CURRENT, [root], 30)
    if rc:
        raise RuntimeError(f"failed to read active source generation: {output}")
    match = re.fullmatch(
        r"LABCTL_SOURCE_CURRENT=(NONE|manifests/[0-9a-f]{64}\.json)", output
    )
    if not match:
        raise RuntimeError(f"invalid active source-generation receipt: {output}")
    return None if match.group(1) == "NONE" else match.group(1)


def reconcile(ep: c.Endpoint, root: str = SOURCE_ROOT) -> int:
    rc, output = _capture(ep, cleanup_scripts.RECONCILE, [root], 300)
    match = re.fullmatch(r"LABCTL_SOURCE_RECONCILED=([0-9]+)", output)
    if rc or not match:
        raise RuntimeError(f"stale source-transaction reconciliation failed: {output}")
    return int(match.group(1))


def prepare(
    ep: c.Endpoint, estimated_bytes: int, estimated_entries: int, root: str = SOURCE_ROOT
) -> tuple[str, str]:
    reconcile(ep, root)
    token = secrets.token_hex(32)
    rc, output = _capture(
        ep, scripts.PREPARE, [root, token, str(estimated_bytes), str(estimated_entries)], 30
    )
    transaction = f"{root}/transactions/{token}"
    if rc or output.splitlines() != [f"LABCTL_SOURCE_TRANSACTION={transaction}"]:
        raise RuntimeError(f"failed to prepare source transaction: {output}")
    return token, transaction


def _filesystem_inventory(root: Path) -> tuple[int, int]:
    size, entries, pending = 0, 0, [root]
    while pending:
        directory = pending.pop()
        entries += 1
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            else:
                entries += 1
                size += path.lstat().st_size
    return size, entries


def estimate_transaction(
    repos: list[Path], bundles: dict, overlays: dict
) -> tuple[int, int]:
    size, entries = 1024**3, 10_000
    for repo in repos:
        git_size, git_entries = _filesystem_inventory(repo / ".git")
        tracked = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        source_size, source_entries = 0, 0
        for raw in tracked:
            if not raw:
                continue
            rel = raw.decode()
            if "__pycache__" in rel or any(rel.startswith(prefix) for prefix in c.SYNC_EXCLUDES):
                continue
            path = repo / rel
            if os.path.lexists(path):
                source_size += path.lstat().st_size
                source_entries += 1
        size += 2 * (git_size + source_size)
        entries += 2 * (git_entries + source_entries)
    for item in [*bundles.values(), *overlays.values()]:
        size += 2 * item["size"]
        entries += 2
    if (
        size > cleanup_scripts.MAX_TRANSACTION_BYTES
        or entries > cleanup_scripts.MAX_TRANSACTION_ENTRIES
    ):
        raise RuntimeError("local source transaction estimate exceeds 40 GiB/1M entries")
    return size, entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_bundle(
    repo: Path, seed_head: str, target_head: str, directory: Path, name: str
) -> dict | None:
    if seed_head == target_head:
        return None
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", seed_head, target_head],
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode:
        raise RuntimeError(
            f"seed HEAD is not a local ancestor for {name}; sync before committing"
        )
    temporary = directory / f"{name}.bundle.tmp"
    subprocess.run(
        [
            "git", "-C", str(repo), "bundle", "create", str(temporary),
            "HEAD", f"^{seed_head}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    current = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if current != target_head or heads != [f"{target_head} HEAD"]:
        raise RuntimeError(f"local HEAD moved while bundling {name}")
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "verify", str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = _sha256(temporary)
    bundle = directory / f"{name}-{digest}.bundle"
    temporary.replace(bundle)
    return {"local_path": bundle, "sha256": digest, "size": bundle.stat().st_size}


def _transport(ep: c.Endpoint) -> str:
    return " ".join(
        shlex.quote(item) for item in ("ssh", *c.SSH_OPTS, "-p", str(ep.port))
    )


def _transfer_object(ep: c.Endpoint, source: dict, remote: str, label: str) -> dict:
    proc = subprocess.run(
        [
            "rsync", "-az", "--ignore-existing", "--no-owner", "--no-group",
            "--stats", "-e", _transport(ep), str(source["local_path"]),
            f"{ep.user}@{ep.host}:{remote}",
        ],
        timeout=180,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"source object transfer failed for {label}")
    return {
        "path": remote,
        "sha256": source["sha256"],
        "size": source["size"],
    }


def transfer_bundle(
    ep: c.Endpoint, transaction: str, name: str, bundle: dict
) -> dict:
    return _transfer_object(
        ep, bundle, f"{transaction}/bundles/{name}.bundle", f"bundle {name}"
    )


def stage(
    ep: c.Endpoint,
    transaction: str,
    seeds: dict,
    seed_identities: dict,
    bundles: dict,
    targets: dict,
    tree_script: str,
    excludes: tuple[str, ...],
) -> str:
    args = _stage_args(
        transaction, seeds, seed_identities, bundles, targets, tree_script, excludes
    )
    rc, output = _capture(ep, scripts.STAGE, args, 300)
    staging = f"{transaction}/staging"
    if rc or output.splitlines() != [f"LABCTL_SOURCE_STAGING={staging}"]:
        raise RuntimeError(f"source staging failed: {output}")
    return staging


def _stage_args(
    transaction: str,
    seeds: dict,
    seed_identities: dict,
    bundles: dict,
    targets: dict,
    tree_script: str,
    excludes: tuple[str, ...],
) -> list[str]:
    return [
        transaction,
        _json64(seeds),
        _json64(seed_identities),
        _json64(bundles),
        _json64(targets),
        _encode(tree_script.encode()),
        json.dumps(excludes),
    ]


def create_overlay(repo: Path, identity: dict, directory: Path, name: str) -> dict | None:
    present = [item for item in identity["entries"] if item["kind"] != "deleted"]
    if not present:
        return None
    archive = directory / f"{name}.overlay.tar"
    with tarfile.open(archive, "w", format=tarfile.GNU_FORMAT) as output:
        for item in present:
            source = repo / item["path"]
            info = tarfile.TarInfo(item["path"])
            info.mode, info.mtime, info.uid, info.gid = item["mode"], 0, 0, 0
            info.uname = info.gname = ""
            if item["kind"] == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = source.readlink().as_posix()
                output.addfile(info)
            else:
                info.size = item["size"]
                with open(source, "rb") as payload:
                    output.addfile(info, payload)
    digest = _sha256(archive)
    return {"local_path": archive, "sha256": digest, "size": archive.stat().st_size}


def transfer_overlay(
    ep: c.Endpoint, transaction: str, name: str, overlay: dict
) -> dict:
    return _transfer_object(
        ep, overlay, f"{transaction}/overlays/{name}.tar", f"overlay {name}"
    )


def generation_document(local_identities: dict) -> tuple[bytes, str]:
    repositories = []
    for name in sorted(local_identities):
        item = local_identities[name]
        repositories.append(
            {
                "head": item["head"],
                "name": name,
                "relative_path": f"repos/{name}",
                "tree_content_sha256": item["tree_content_sha256"],
                "worktree_identity_sha256": item["identity_sha256"],
            }
        )
    document = {
        "repositories": repositories,
        "schema_version": "stwo.gpu-lab.source-generation.v1",
    }
    payload = json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    return payload, hashlib.sha256(payload).hexdigest()


def finalize(
    ep: c.Endpoint,
    root: str,
    transaction: str,
    seeds: dict,
    seed_identities: dict,
    local_identities: dict,
    overlays: dict,
    generation_payload: bytes,
    generation_sha: str,
    expected_current: str | None,
    tree_script: str,
    excludes: tuple[str, ...],
) -> dict:
    args = _finalize_args(
        root,
        transaction,
        seeds,
        seed_identities,
        local_identities,
        overlays,
        generation_payload,
        generation_sha,
        expected_current,
        tree_script,
        excludes,
    )
    rc, output = _capture(ep, scripts.FINALIZE, args, 300)
    if rc:
        raise RuntimeError(f"source generation finalize failed: {output}")
    return _parse_publication(output, root, generation_sha, local_identities)


def _finalize_args(
    root: str,
    transaction: str,
    seeds: dict,
    seed_identities: dict,
    local_identities: dict,
    overlays: dict,
    generation_payload: bytes,
    generation_sha: str,
    expected_current: str | None,
    tree_script: str,
    excludes: tuple[str, ...],
) -> list[str]:
    return [
        root,
        transaction,
        _json64(seeds),
        _json64(seed_identities),
        _json64(local_identities),
        _json64(overlays),
        _encode(generation_payload),
        generation_sha,
        expected_current or "",
        _encode(tree_script.encode()),
        json.dumps(excludes),
    ]


def _pointer_sha(root: str, generation_sha: str, local_identities: dict) -> str:
    final = f"{root}/sha256/{generation_sha}"
    repositories = []
    for name in sorted(local_identities):
        item = local_identities[name]
        repositories.append(
            {
                "head": item["head"],
                "name": name,
                "path": f"{final}/repos/{name}",
                "tree_content_sha256": item["tree_content_sha256"],
                "worktree_identity_sha256": item["identity_sha256"],
            }
        )
    document = {
        "generation_path": final,
        "generation_sha256": generation_sha,
        "repositories": repositories,
        "schema_version": "stwo.gpu-lab.source-pointer.v1",
    }
    payload = json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _parse_publication(
    output: str, root: str, generation_sha: str, local_identities: dict
) -> dict:
    lines = output.splitlines()
    if len(lines) != 5:
        raise RuntimeError(f"invalid source-generation receipt: {output}")
    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError(f"ambiguous source-generation receipt: {output}")
        values[key] = value
    expected_path = f"{root}/sha256/{generation_sha}"
    expected_pointer_sha = _pointer_sha(root, generation_sha, local_identities)
    if (
        values.get("LABCTL_SOURCE_GENERATION") != expected_path
        or values.get("LABCTL_SOURCE_GENERATION_SHA256") != generation_sha
        or values.get("LABCTL_SOURCE_POINTER") != f"{root}/CURRENT"
        or values.get("LABCTL_SOURCE_POINTER_TARGET")
        != f"manifests/{expected_pointer_sha}.json"
        or values.get("LABCTL_SOURCE_POINTER_SHA256") != expected_pointer_sha
    ):
        raise RuntimeError(f"mismatched source-generation receipt: {output}")
    return {
        "generation_path": expected_path,
        "generation_sha256": generation_sha,
        "pointer_path": f"{root}/CURRENT",
        "pointer_sha256": values["LABCTL_SOURCE_POINTER_SHA256"],
        "pointer_target": values["LABCTL_SOURCE_POINTER_TARGET"],
    }


def attest(
    ep: c.Endpoint,
    root: str,
    publication: dict,
    local_identities: dict,
    tree_script: str,
    excludes: tuple[str, ...],
) -> None:
    args = _attest_args(root, publication, local_identities, tree_script, excludes)
    rc, output = _capture(ep, scripts.ATTEST, args, 180)
    expected = f"LABCTL_SOURCE_ATTESTED={publication['generation_sha256']}"
    if rc or output.splitlines() != [expected]:
        raise RuntimeError(f"active source-generation attestation failed: {output}")


def _attest_args(
    root: str,
    publication: dict,
    local_identities: dict,
    tree_script: str,
    excludes: tuple[str, ...],
) -> list[str]:
    return [
        root,
        _json64(publication),
        _json64(local_identities),
        _encode(tree_script.encode()),
        json.dumps(excludes),
    ]


def cleanup(
    ep: c.Endpoint,
    root: str,
    transaction: str,
    publication: dict,
    record: str,
    record_sha256: str,
) -> None:
    args = _cleanup_args(root, transaction, publication, record, record_sha256)
    rc, output = _capture(ep, cleanup_scripts.SUCCESS, args, 300)
    lines = output.splitlines()
    expected_prefixes = (
        f"LABCTL_SOURCE_CLEANED={transaction}",
        "LABCTL_SOURCE_CLEANED_BYTES=",
        "LABCTL_SOURCE_CLEANED_ENTRIES=",
    )
    if (
        rc
        or len(lines) != 3
        or lines[0] != expected_prefixes[0]
        or not lines[1].startswith(expected_prefixes[1])
        or not lines[2].startswith(expected_prefixes[2])
    ):
        raise RuntimeError(f"successful source-transaction cleanup failed: {output}")
    size = int(lines[1].split("=", 1)[1])
    entries = int(lines[2].split("=", 1)[1])
    if (
        size > cleanup_scripts.MAX_TRANSACTION_BYTES
        or entries > cleanup_scripts.MAX_TRANSACTION_ENTRIES
    ):
        raise RuntimeError("source cleanup receipt exceeds its safety bound")


def _cleanup_args(
    root: str,
    transaction: str,
    publication: dict,
    record: str,
    record_sha256: str,
) -> list[str]:
    return [root, transaction, _json64(publication), record, record_sha256]


def record_failure(
    ep: c.Endpoint, root: str, transaction: str, error: BaseException
) -> None:
    args = _failure_args(root, transaction, str(error))
    rc, output = _capture(ep, cleanup_scripts.FAILURE, args, 300)
    lines = output.splitlines()
    expected_path = f"{root}/failures/{Path(transaction).name}.json"
    if (
        rc
        or len(lines) != 4
        or lines[0] != f"LABCTL_SOURCE_FAILURE={expected_path}"
        or not re.fullmatch(r"LABCTL_SOURCE_FAILURE_SHA256=[0-9a-f]{64}", lines[1])
        or not re.fullmatch(r"LABCTL_SOURCE_FAILURE_BYTES=[0-9]+", lines[2])
        or not re.fullmatch(r"LABCTL_SOURCE_FAILURE_ENTRIES=[0-9]+", lines[3])
    ):
        raise RuntimeError(f"failed source-transaction compaction failed: {output}")
    size = int(lines[2].split("=", 1)[1])
    entries = int(lines[3].split("=", 1)[1])
    if (
        size > cleanup_scripts.MAX_TRANSACTION_BYTES
        or entries > cleanup_scripts.MAX_TRANSACTION_ENTRIES
    ):
        raise RuntimeError("failed-source receipt exceeds its safety bound")


def _failure_args(root: str, transaction: str, error: str) -> list[str]:
    return [root, transaction, _encode(error.encode())]
