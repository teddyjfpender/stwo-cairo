"""Identity of the exact source projection transported by ``pod_run.sh``."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

_DIFF_EXCLUSIONS = (
    ":(exclude)gpu_benchmarks/loop/results/**",
    ":(exclude)gpu_benchmarks/loop/ledger.jsonl",
    ":(exclude)gpu_benchmarks/loop/pod.conf",
    ":(exclude)gpu_benchmarks/pie/sn/**",
    ":(exclude)gpu_benchmarks/pie/*.zip",
    ":(exclude)gpu_benchmarks/results/**",
    ":(exclude)gpu_benchmarks/fleet/results/**",
    ":(exclude)gpu_benchmarks/fleet/fleet_report.json",
    ":(exclude)gpu_benchmarks/fleet/fleet.conf",
    ":(exclude)gpu_benchmarks/fleet/ledger_costs.jsonl",
    ":(exclude)gpu_benchmarks/fleet/pods.conf*",
)


def _excluded(path: bytes) -> bool:
    return (
        path.startswith(b"gpu_benchmarks/loop/results/")
        or path == b"gpu_benchmarks/loop/ledger.jsonl"
        or path == b"gpu_benchmarks/loop/pod.conf"
        or path.startswith(b"gpu_benchmarks/pie/sn/")
        or path.startswith(b"gpu_benchmarks/results/")
        or path.startswith(b"gpu_benchmarks/fleet/results/")
        or path == b"gpu_benchmarks/fleet/fleet_report.json"
        or path == b"gpu_benchmarks/fleet/fleet.conf"
        or path == b"gpu_benchmarks/fleet/ledger_costs.jsonl"
        or path.startswith(b"gpu_benchmarks/fleet/pods.conf")
        or (
            path.startswith(b"gpu_benchmarks/pie/")
            and path.endswith(b".zip")
        )
    )


def _git(repository: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, check=False
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"git {' '.join(args)} failed in {repository}: {detail}")
    return result.stdout


def _file_digest(path: bytes) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().encode()


def projection_identity(repository: Path) -> dict[str, str]:
    """Return HEAD plus the tracked-and-untracked transported-content hash."""
    repository = repository.resolve(strict=True)
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    digest = hashlib.sha256()
    digest.update(
        _git(repository, "diff", "--binary", "HEAD", "--", ".", *_DIFF_EXCLUSIONS)
    )

    root = os.fsencode(repository)
    untracked = _git(repository, "ls-files", "--others", "--exclude-standard", "-z")
    for relative in untracked.split(b"\0"):
        if not relative or _excluded(relative):
            continue
        path = os.path.join(root, relative)
        if os.path.islink(path):
            target = os.readlink(path)
            target = target if isinstance(target, bytes) else os.fsencode(target)
            kind = b"symlink"
            content_hash = hashlib.sha256(target).hexdigest().encode()
        elif os.path.isfile(path):
            kind = b"executable" if os.access(path, os.X_OK) else b"regular"
            content_hash = _file_digest(path)
        else:
            rendered = os.fsdecode(relative)
            raise ValueError(f"unsupported untracked source path: {rendered}")
        digest.update(b"untracked-" + kind + b"\0" + relative + b"\0")
        digest.update(content_hash + b"\0")

    return {"head": head, "worktree_sha256": digest.hexdigest()}
