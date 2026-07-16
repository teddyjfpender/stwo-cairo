"""No-network interleaving tests for immutable source generations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import common as c
from . import generation
from . import generation_cleanup_scripts as cleanup_scripts
from . import generation_ownership_tests as ownership_tests
from . import generation_scripts as scripts
from . import sync

TEST_PUBLISHER = [str(os.geteuid()), str(os.getegid())]

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(script: str, args: list[str], publisher=TEST_PUBLISHER):
    environment = os.environ.copy()
    environment["LABCTL_DARWIN_READONLY_RENAME_TEST"] = "1"
    return subprocess.run(
        [sys.executable, "-c", ownership_tests.with_statvfs_shim(script), *publisher, *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _context(
    root: Path,
    seed: Path,
    local: Path,
    token: str,
    expected_current: str | None,
) -> dict:
    seed_identity = sync._local_tree_identity(seed)
    local_identity = sync._local_tree_identity(local)
    transaction = root / "transactions" / token
    prepared = _run(scripts.PREPARE, [str(root), token, str(1024**2), "1000"])
    assert prepared.returncode == 0, prepared.stderr
    bundle_dir = root.parent / f"bundles-{token}"
    bundle_dir.mkdir()
    bundle = generation.create_bundle(
        local, seed_identity["head"], local_identity["head"], bundle_dir, "repo"
    )
    bundles = {}
    if bundle:
        remote_bundle = transaction / "bundles" / "repo.bundle"
        shutil.copy2(bundle["local_path"], remote_bundle)
        bundles["repo"] = {
            "path": str(remote_bundle),
            "sha256": bundle["sha256"],
            "size": bundle["size"],
        }
    overlay = generation.create_overlay(
        local, local_identity, bundle_dir, "repo"
    )
    overlays = {}
    if overlay:
        remote_overlay = transaction / "overlays" / "repo.tar"
        shutil.copy2(overlay["local_path"], remote_overlay)
        overlays["repo"] = {
            "path": str(remote_overlay),
            "sha256": overlay["sha256"],
            "size": overlay["size"],
        }
    stage_args = generation._stage_args(
        str(transaction),
        {"repo": str(seed)},
        {"repo": seed_identity},
        bundles,
        {"repo": local_identity["head"]},
        sync.TREE_ID_SCRIPT,
        c.SYNC_EXCLUDES,
    )
    staged = _run(scripts.STAGE, stage_args)
    assert staged.returncode == 0, staged.stderr
    staging = transaction / "staging"
    payload, generation_sha = generation.generation_document(
        {"repo": local_identity}
    )
    return {
        "expected_current": expected_current,
        "generation_payload": payload,
        "generation_sha": generation_sha,
        "local_identity": local_identity,
        "overlays": overlays,
        "root": root,
        "seed": seed,
        "seed_identity": seed_identity,
        "staging": staging,
        "transaction": transaction,
    }


def _finalize(context: dict) -> subprocess.CompletedProcess[str]:
    args = generation._finalize_args(
        str(context["root"]),
        str(context["transaction"]),
        {"repo": str(context["seed"])},
        {"repo": context["seed_identity"]},
        {"repo": context["local_identity"]},
        context["overlays"],
        context["generation_payload"],
        context["generation_sha"],
        context["expected_current"],
        sync.TREE_ID_SCRIPT,
        c.SYNC_EXCLUDES,
    )
    return _run(scripts.FINALIZE, args)


def _publication(context: dict, result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    return generation._parse_publication(
        result.stdout,
        str(context["root"]),
        context["generation_sha"],
        {"repo": context["local_identity"]},
    )


def _attest(context: dict, publication: dict) -> None:
    result = _attest_result(context, publication)
    assert result.stdout.splitlines() == [
        f"LABCTL_SOURCE_ATTESTED={context['generation_sha']}"
    ]


def _attest_result(context: dict, publication: dict) -> subprocess.CompletedProcess[str]:
    return _run(
        scripts.ATTEST,
        generation._attest_args(
            str(context["root"]),
            publication,
            {"repo": context["local_identity"]},
            sync.TREE_ID_SCRIPT,
            c.SYNC_EXCLUDES,
        ),
    )


def _assert_frozen(generation_path: Path) -> None:
    for root, directories, files in os.walk(generation_path, followlinks=False):
        root_path = Path(root)
        assert root_path.lstat().st_mode & 0o222 == 0
        for name in files:
            path = root_path / name
            if not path.is_symlink():
                info = path.lstat()
                assert info.st_mode & 0o222 == 0 and info.st_nlink == 1


def _cleanup_success(context: dict, publication: dict) -> None:
    record = context["root"].parent / f"record-{context['transaction'].name}.json"
    record.write_text("evidence\n")
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    args = generation._cleanup_args(
        str(context["root"]),
        str(context["transaction"]),
        publication,
        str(record),
        digest,
    )
    rejected = _run(cleanup_scripts.SUCCESS, [*args[:-1], "0" * 64])
    assert rejected.returncode != 0 and context["transaction"].exists()
    result = _run(
        cleanup_scripts.SUCCESS,
        args,
    )
    assert result.returncode == 0, result.stderr
    assert not context["transaction"].exists()


def _reconcile(root: Path) -> None:
    result = _run(cleanup_scripts.RECONCILE, [str(root)])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LABCTL_SOURCE_RECONCILED=1"


def _failure_manifest(
    root: Path,
    transaction: Path,
    *,
    claimed_transaction: Path | None = None,
    tamper: bool = False,
    transaction_bytes: int = 0,
    transaction_entries: int = 4,
) -> tuple[Path, bytes]:
    error = "controller stopped after writing failure evidence"
    document = {
        "cleanup_policy": {
            "max_transaction_bytes": 40 * 1024**3,
            "max_transaction_entries": 1_000_000,
            "min_free_bytes_before_next_transaction": 20 * 1024**3,
        },
        "error": error,
        "error_sha256": hashlib.sha256(error.encode()).hexdigest(),
        "schema_version": "stwo.gpu-lab.source-failure.v1",
        "transaction": str(claimed_transaction or transaction),
        "transaction_bytes": transaction_bytes,
        "transaction_entries": transaction_entries,
    }
    payload = json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    if tamper:
        payload += b" "
    manifest = root / "failures" / f"{transaction.name}.json"
    manifest.write_bytes(payload)
    manifest.chmod(0o444)
    return manifest, payload


def generation_self_test() -> None:
    for name in ("PREPARE", "STAGE", "FINALIZE", "ATTEST", "CURRENT"):
        compile(getattr(scripts, name), f"generation-{name.lower()}", "exec")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        ownership_tests.controller_argument_rejections(
            _run, scripts.PREPARE, scripts.CURRENT, base, TEST_PUBLISHER, generation, c)
        rename_source, rename_destination = base / "rename-source", base / "rename-dest"
        rename_source.mkdir()
        rename_destination.mkdir()
        (rename_source / "source.txt").write_text("source\n")
        collision = _run(
            scripts.SUPPORT
            + "\nprint('RENAMED=' + str(rename_noreplace(Path(sys.argv[3]), Path(sys.argv[4]))))\n",
            [str(rename_source), str(rename_destination)],
        )
        assert collision.stdout.strip() == "RENAMED=False"
        assert (rename_source / "source.txt").is_file() and rename_destination.is_dir()
        root, seed, local = base / "source-generations", base / "seed", base / "local"

        crash_token = "c" * 64
        crash_transaction = root / "transactions" / crash_token
        prepared = _run(
            scripts.PREPARE, [str(root), crash_token, str(1024**2), "1000"]
        )
        assert prepared.returncode == 0, prepared.stderr
        crash_manifest, crash_payload = _failure_manifest(root, crash_transaction)
        _reconcile(root)
        assert not crash_transaction.exists()
        assert crash_manifest.read_bytes() == crash_payload
        assert crash_manifest.stat().st_mode & 0o777 == 0o444

        partial_token = "f" * 64
        partial_transaction = root / "transactions" / partial_token
        prepared = _run(
            scripts.PREPARE, [str(root), partial_token, str(1024**2), "1000"]
        )
        assert prepared.returncode == 0, prepared.stderr
        first_partial = partial_transaction / "bundles" / "first"
        second_partial = partial_transaction / "overlays" / "second"
        first_partial.write_bytes(b"one\n")
        second_partial.write_bytes(b"two\n")
        partial_manifest, partial_payload = _failure_manifest(
            root,
            partial_transaction,
            transaction_bytes=8,
            transaction_entries=6,
        )
        for path in (first_partial, second_partial):
            path.chmod(0o600)
        for path in (
            partial_transaction / "bundles",
            partial_transaction / "overlays",
            partial_transaction / "quarantine",
            partial_transaction,
        ):
            path.chmod(0o700)
        first_partial.unlink()
        _reconcile(root)
        assert not partial_transaction.exists()
        assert partial_manifest.read_bytes() == partial_payload

        larger_token = "a" * 64
        larger_transaction = root / "transactions" / larger_token
        prepared = _run(
            scripts.PREPARE, [str(root), larger_token, str(1024**2), "1000"]
        )
        assert prepared.returncode == 0, prepared.stderr
        larger_manifest, larger_payload = _failure_manifest(root, larger_transaction)
        (larger_transaction / "bundles" / "unexpected").write_text("larger\n")
        rejected = _run(cleanup_scripts.RECONCILE, [str(root)])
        assert rejected.returncode != 0
        assert (
            larger_transaction.is_dir()
            and larger_manifest.read_bytes() == larger_payload
        )
        larger_manifest.unlink()
        _reconcile(root)

        wrong_token = "d" * 64
        wrong_transaction = root / "transactions" / wrong_token
        prepared = _run(
            scripts.PREPARE, [str(root), wrong_token, str(1024**2), "1000"]
        )
        assert prepared.returncode == 0, prepared.stderr
        wrong_manifest, wrong_payload = _failure_manifest(
            root,
            wrong_transaction,
            claimed_transaction=root / "transactions" / ("e" * 64),
        )
        rejected = _run(cleanup_scripts.RECONCILE, [str(root)])
        assert rejected.returncode != 0
        assert wrong_transaction.is_dir() and wrong_manifest.read_bytes() == wrong_payload
        wrong_manifest.unlink()
        _reconcile(root)

        tampered_token = "e" * 64
        tampered_transaction = root / "transactions" / tampered_token
        prepared = _run(
            scripts.PREPARE, [str(root), tampered_token, str(1024**2), "1000"]
        )
        assert prepared.returncode == 0, prepared.stderr
        tampered_manifest, tampered_payload = _failure_manifest(
            root, tampered_transaction, tamper=True
        )
        rejected = _run(cleanup_scripts.RECONCILE, [str(root)])
        assert rejected.returncode != 0
        assert (
            tampered_transaction.is_dir()
            and tampered_manifest.read_bytes() == tampered_payload
        )
        tampered_manifest.unlink()
        _reconcile(root)

        seed.mkdir()
        _git(seed, "init", "-q")
        _git(seed, "config", "user.email", "labctl@test")
        _git(seed, "config", "user.name", "labctl")
        for name in ("head.txt", "modified.txt", "deleted.txt"):
            (seed / name).write_text("base\n")
        _git(seed, "add", ".")
        _git(seed, "commit", "-qm", "base")
        subprocess.run(["git", "clone", "-q", str(seed), str(local)], check=True)
        _git(local, "config", "user.email", "labctl@test")
        _git(local, "config", "user.name", "labctl")
        (local / "head.txt").write_text("committed\n")
        _git(local, "commit", "-qam", "advance")
        (local / "modified.txt").write_text("dirty\n")
        (local / "deleted.txt").unlink()
        (local / "added.txt").write_text("added\n")
        os.symlink("added.txt", local / "link.txt")
        ownership_tests.assert_global_git_config_ignored(local, sync, c.SYNC_EXCLUDES)
        seed_before = sync._local_tree_identity(seed)
        token = "0" * 64
        transaction = root / "transactions" / token
        assert _run(
            scripts.PREPARE, [str(root), token, str(1024**2), "1000"]
        ).returncode == 0
        bundle_dir = base / "seed-race-bundle"
        bundle_dir.mkdir()
        local_identity = sync._local_tree_identity(local)
        bundle = generation.create_bundle(
            local, seed_before["head"], local_identity["head"], bundle_dir, "repo"
        )
        remote_bundle = transaction / "bundles" / "repo.bundle"
        shutil.copy2(bundle["local_path"], remote_bundle)
        (seed / "head.txt").write_text("seed race\n")
        seed_race = _run(
            scripts.STAGE,
            generation._stage_args(
                str(transaction),
                {"repo": str(seed)},
                {"repo": seed_before},
                {"repo": {"path": str(remote_bundle), "sha256": bundle["sha256"], "size": bundle["size"]}},
                {"repo": local_identity["head"]},
                sync.TREE_ID_SCRIPT,
                c.SYNC_EXCLUDES,
            ),
        )
        assert seed_race.returncode != 0 and "seed changed" in seed_race.stderr
        assert (seed / "head.txt").read_text() == "seed race\n"
        (seed / "head.txt").write_text("base\n")
        _reconcile(root)

        first = _context(root, seed, local, "1" * 64, None)
        publication1 = _publication(first, _finalize(first))
        generation1 = Path(publication1["generation_path"])
        ownership_tests.assert_readonly_git_validation(
            generation1, lambda: _attest(first, publication1))
        assert _run(scripts.CURRENT, [str(root)]).stdout.splitlines() == [
            f"LABCTL_SOURCE_CURRENT={publication1['pointer_target']}"
        ]
        assert sync._local_tree_identity(generation1 / "repos" / "repo") == first[
            "local_identity"
        ]
        assert (generation1 / "repos" / "repo" / "added.txt").read_text() == "added\n"
        assert len(list((first["transaction"] / "quarantine" / "repo").iterdir())) >= 2
        _cleanup_success(first, publication1)
        assert generation1.is_dir()
        ownership_tests.physical_owner_rejection(
            generation1 / "repos/repo/added.txt",
            lambda: _attest_result(first, publication1),
            lambda: _context(root, seed, local, "b" * 64, publication1["pointer_target"]),
            _finalize,
            lambda: _reconcile(root),
        )

        _git(local, "add", "-A")
        _git(local, "commit", "-qm", "snapshot")
        (local / "head.txt").write_text("second dirty\n")
        (local / "second.txt").write_text("second\n")
        current1 = publication1["pointer_target"]

        staged_index = _context(root, seed, local, "2" * 64, current1)
        staged_repo = staged_index["staging"] / "repos" / "repo"
        committed_bytes = (staged_repo / "head.txt").read_text()
        (staged_repo / "head.txt").write_text("concurrent staged\n")
        _git(staged_repo, "add", "head.txt")
        (staged_repo / "head.txt").write_text(committed_bytes)
        failed = _finalize(staged_index)
        assert failed.returncode != 0 and "staging changed" in failed.stderr
        assert (staged_repo / "head.txt").read_text() == committed_bytes
        assert subprocess.run(
            ["git", "-C", str(staged_repo), "diff", "--cached", "--quiet", "HEAD"],
            check=False,
        ).returncode == 1
        assert generation1.is_dir()
        _reconcile(root)

        source_write = _context(root, seed, local, "3" * 64, current1)
        source_repo = source_write["staging"] / "repos" / "repo"
        (source_repo / "head.txt").write_text("concurrent write\n")
        assert _finalize(source_write).returncode != 0
        assert (source_repo / "head.txt").read_text() == "concurrent write\n"
        assert generation1.is_dir()
        _reconcile(root)

        source_delete = _context(root, seed, local, "4" * 64, current1)
        deleted_path = source_delete["staging"] / "repos" / "repo" / "head.txt"
        deleted_path.unlink()
        assert _finalize(source_delete).returncode != 0
        assert (generation1 / "repos" / "repo" / "head.txt").is_file()
        _reconcile(root)

        pointer_race = _context(root, seed, local, "5" * 64, current1)
        fake_target = "manifests/" + "f" * 64 + ".json"
        temporary = root / ".CURRENT-test-race"
        os.symlink(fake_target, temporary)
        os.replace(temporary, root / "CURRENT")
        raced = _finalize(pointer_race)
        assert raced.returncode != 0 and "pointer changed" in raced.stderr, raced.stderr
        generation2 = root / "sha256" / pointer_race["generation_sha"]
        assert generation1.is_dir() and generation2.is_dir()
        _reconcile(root)
        restore = root / ".CURRENT-test-restore"
        os.symlink(current1, restore)
        os.replace(restore, root / "CURRENT")

        second = _context(root, seed, local, "6" * 64, current1)
        publication2 = _publication(second, _finalize(second))
        _attest(second, publication2)
        assert generation1.is_dir()
        assert Path(publication2["generation_path"]).is_dir()
        assert publication2["generation_path"] != publication1["generation_path"]
        assert sync._local_tree_identity(generation1 / "repos" / "repo") == first[
            "local_identity"
        ]
        assert sync._local_tree_identity(seed) == seed_before
        _cleanup_success(second, publication2)

        calls = []
        original_run = generation.subprocess.run
        try:
            def fake_run(argv, **kwargs):
                calls.append((argv, kwargs))
                return subprocess.CompletedProcess(argv, 0)

            generation.subprocess.run = fake_run
            generation._transfer_object(
                c.Endpoint("host", 22),
                {"local_path": base / "object", "sha256": "a" * 64, "size": 1},
                "/remote/object",
                "test",
            )
        finally:
            generation.subprocess.run = original_run
        assert len(calls) == 1
        argv, _ = calls[0]
        assert "--ignore-existing" in argv
        assert "--no-owner" in argv and "--no-group" in argv
