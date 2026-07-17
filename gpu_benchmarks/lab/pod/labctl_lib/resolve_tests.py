"""Provider-free hostile checks for active source resolution."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import common as c
from . import generation_ownership_tests as ownership
from . import resolve
from . import runtime


PUBLISHER = [str(os.geteuid()), str(os.getegid())]


def _canonical(document: dict) -> bytes:
    return json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", resolve.REMOTE_RESOLVE, *PUBLISHER, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(path: Path, payload: bytes, mode: int) -> None:
    if path.exists():
        assert path.read_bytes() == payload
        assert path.stat().st_mode & 0o777 == mode
        return
    path.write_bytes(payload)
    path.chmod(mode)


def _install_pointer(root: Path, generation: Path, repositories: list[dict]) -> Path:
    pointer_repositories = [
        {
            "head": repo["head"],
            "name": repo["name"],
            "path": str(generation / repo["relative_path"]),
            "tree_content_sha256": repo["tree_content_sha256"],
            "worktree_identity_sha256": repo["worktree_identity_sha256"],
        }
        for repo in repositories
    ]
    document = {
        "generation_path": str(generation),
        "generation_sha256": generation.name,
        "repositories": pointer_repositories,
        "schema_version": "stwo.gpu-lab.source-pointer.v1",
    }
    payload = _canonical(document)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = root / "manifests" / f"{digest}.json"
    _write(manifest, payload, 0o400)
    temporary = root / ".CURRENT-test"
    temporary.unlink(missing_ok=True)
    os.symlink(f"manifests/{digest}.json", temporary)
    os.replace(temporary, root / "CURRENT")
    return manifest


def _fixture(base: Path) -> tuple[Path, Path, list[dict], Path]:
    root = base / "source-generations"
    (root / "sha256").mkdir(parents=True, mode=0o755)
    (root / "manifests").mkdir(mode=0o755)
    repositories = [
        {
            "head": character * 40,
            "name": name,
            "relative_path": f"repos/{name}",
            "tree_content_sha256": character * 64,
            "worktree_identity_sha256": character.upper().lower() * 64,
        }
        for name, character in (("stwo", "a"), ("stwo-cairo", "b"))
    ]
    generation_document = {
        "repositories": repositories,
        "schema_version": "stwo.gpu-lab.source-generation.v1",
    }
    generation_payload = _canonical(generation_document)
    generation_sha = hashlib.sha256(generation_payload).hexdigest()
    generation = root / "sha256" / generation_sha
    (generation / "repos" / "stwo").mkdir(parents=True)
    (generation / "repos" / "stwo-cairo").mkdir()
    _write(generation / "GENERATION.json", generation_payload, 0o444)
    for path in (
        generation / "repos" / "stwo",
        generation / "repos" / "stwo-cairo",
        generation / "repos",
        generation,
    ):
        path.chmod(0o555)
    pointer = _install_pointer(root, generation, repositories)
    return root, generation, repositories, pointer


def _remote_authority_checks() -> dict:
    compile(resolve.REMOTE_RESOLVE, "labctl-resolve", "exec")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        root, generation, repositories, pointer = _fixture(base)
        (root / "sha256" / "unrelated-invalid-object").write_text("ignored\n")
        before = ownership.exact_path_inventory(root)
        result = _run(root)
        assert result.returncode == 0, result.stderr
        assert result.stdout.endswith("\n") and not result.stdout.endswith("\n\n")
        document = resolve._parse(result.stdout[:-1], str(root))
        assert document["repositories"]["stwo"]["path"] == str(
            generation / "repos" / "stwo"
        )
        assert document["repositories"]["stwo-cairo"]["head"] == "b" * 40
        assert ownership.exact_path_inventory(root) == before
        assert pointer.stat().st_mode & 0o777 == 0o400

        pointer.chmod(0o444)
        rejected = _run(root)
        assert rejected.returncode != 0 and "mutable or invalid" in rejected.stderr
        pointer.chmod(0o400)

        mismatched = [dict(item) for item in repositories]
        mismatched[1]["head"] = "c" * 40
        bad_pointer = _install_pointer(root, generation, mismatched)
        rejected = _run(root)
        assert rejected.returncode != 0
        assert "identities disagree" in rejected.stderr
        assert bad_pointer.stat().st_mode & 0o777 == 0o400

        _install_pointer(root, generation, repositories)
        source = generation / "repos" / "stwo"
        generation.chmod(0o755)
        (generation / "repos").chmod(0o755)
        source.rmdir()
        os.symlink("stwo-cairo", source)
        (generation / "repos").chmod(0o555)
        generation.chmod(0o555)
        rejected = _run(root)
        assert rejected.returncode != 0
        assert "publisher-protected" in rejected.stderr

        (root / "CURRENT").unlink()
        rejected = _run(root)
        assert rejected.returncode != 0
        assert "no active source generation" in rejected.stderr
        return document


def _local_receipt_checks(document: dict) -> None:
    receipt = resolve._canonical(document)
    assert resolve._parse(receipt, document["pointer_path"].removesuffix("/CURRENT"))
    for hostile in (
        receipt + "\nwarning",
        json.dumps(document, indent=2, sort_keys=True),
        resolve._canonical({**document, "extra": True}),
        resolve._canonical({**document, "generation_sha256": 7}),
        resolve._canonical(
            {
                **document,
                "repositories": {
                    **document["repositories"],
                    "stwo": {
                        **document["repositories"]["stwo"],
                        "path": "/workspace/src/stwo",
                    },
                },
            }
        ),
    ):
        try:
            resolve._parse(
                hostile, document["pointer_path"].removesuffix("/CURRENT")
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted hostile source-resolution receipt")


def _active_lease_checks(document: dict) -> None:
    saved_active, saved_source = runtime._active, resolve.active_source
    try:
        calls = []

        def inactive():
            raise RuntimeError("lease is terminated")

        runtime._active = inactive
        resolve.active_source = lambda _ep: calls.append("transport")
        try:
            resolve.cmd_resolve(argparse.Namespace())
        except RuntimeError:
            pass
        else:
            raise AssertionError("resolved source without an active guarded lease")
        assert not calls

        runtime._active = lambda: (
            {"phase": "open"},
            object(),
            c.Endpoint("host", 22, "root"),
        )
        resolve.active_source = lambda ep: document
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert resolve.cmd_resolve(argparse.Namespace()) == 0
        assert json.loads(output.getvalue()) == document
    finally:
        runtime._active, resolve.active_source = saved_active, saved_source

    try:
        resolve.active_source(c.Endpoint("host", 22, "dev"))
    except RuntimeError as error:
        assert "root controller" in str(error)
    else:
        raise AssertionError("accepted a dev endpoint for controller resolution")


def _transport_receipt_checks(document: dict) -> None:
    saved_capture = resolve.generation._capture
    receipt = resolve._canonical(document)
    root = document["pointer_path"].removesuffix("/CURRENT")
    calls = []
    try:
        resolve.generation._capture = lambda ep, script, args, timeout: (
            calls.append((ep, script, args, timeout)) or (0, receipt)
        )
        assert resolve.active_source(c.Endpoint("host", 22, "root"), root) == document
        ep, script, args, timeout = calls.pop()
        assert ep.user == "root"
        assert script == resolve.REMOTE_RESOLVE
        assert args == [root] and timeout == 30

        for rc, output in (
            (1, "controller rejected pointer"),
            (0, receipt + "\nwarning"),
            (0, receipt + "\n"),
        ):
            resolve.generation._capture = (
                lambda *_args, rc=rc, output=output: (rc, output)
            )
            try:
                resolve.active_source(c.Endpoint("host", 22, "root"), root)
            except RuntimeError:
                pass
            else:
                raise AssertionError("accepted an invalid transport receipt")
    finally:
        resolve.generation._capture = saved_capture


def resolve_self_test() -> None:
    document = _remote_authority_checks()
    _local_receipt_checks(document)
    _active_lease_checks(document)
    _transport_receipt_checks(document)
