"""Physical ownership regressions for immutable source generations."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

_STATVFS_SHIM = (
    "import os,types\n"
    "os.statvfs=lambda path:types.SimpleNamespace(f_bavail=100*1024**3,f_frsize=1)\n"
)


def with_statvfs_shim(script: str) -> str:
    return _STATVFS_SHIM + script


def exact_path_inventory(root: Path) -> tuple:
    records = []
    pending = [root]
    while pending:
        path = pending.pop()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            kind, content = "directory", b""
            pending.extend(
                sorted((Path(entry.path) for entry in os.scandir(path)), reverse=True)
            )
        elif stat.S_ISREG(info.st_mode):
            kind, content = "regular", path.read_bytes()
        elif stat.S_ISLNK(info.st_mode):
            kind, content = "symlink", os.fsencode(os.readlink(path))
        else:
            kind, content = "special", b""
        records.append((
            path.relative_to(root).as_posix(), kind, info.st_ino, info.st_mode,
            info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns,
            hashlib.sha256(content).digest(),
        ))
    return tuple(sorted(records))


def assert_readonly_git_validation(root: Path, validate: Callable[[], None]) -> None:
    before = exact_path_inventory(root)
    validate()
    assert exact_path_inventory(root) == before


def assert_global_git_config_ignored(repo: Path, sync, excludes) -> None:
    (repo / ".gitignore").write_text("repo-ignored.tmp\n")
    (repo / "repo-ignored.tmp").write_text("repository ignore remains semantic\n")
    baseline = sync._local_tree_identity(repo)
    assert not any(item["path"] == "repo-ignored.tmp" for item in baseline["entries"])
    assert any(item["path"] == "added.txt" for item in baseline["entries"])
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        excludes_file = home / "global-ignore"
        included = home / "included.gitconfig"
        excludes_file.write_text("added.txt\n")
        included.write_text(f"[core]\n\texcludesFile = {excludes_file}\n")
        (home / ".gitconfig").write_text(f"[include]\n\tpath = {included}\n")
        environment = os.environ.copy()
        environment.update({"HOME": str(home), "XDG_CONFIG_HOME": str(home)})
        result = subprocess.run(
            [sys.executable, "-c", sync.TREE_ID_SCRIPT, str(repo), json.dumps(excludes)],
            check=True, capture_output=True, text=True, env=environment,
        )
    hostile = sync._parse_tree_identity(result.stdout, label="hostile global config")
    assert hostile == baseline


def controller_argument_rejections(
    run, prepare: str, current: str, base: Path, publisher, generation, common
) -> None:
    for index, estimates in enumerate(((-1, 1), (1, -1))):
        root = base / f"negative-estimate-{index}"
        result = run(
            prepare, [str(root), str(index) * 64, *map(str, estimates)]
        )
        assert result.returncode != 0 and not root.exists()
    wrong_publisher = [str(int(publisher[0]) + 1), publisher[1]]
    assert run(
        current, [str(base / "absent-controller")], publisher=wrong_publisher
    ).returncode != 0
    calls = []
    original = generation.c.ssh_capture
    try:
        generation.c.ssh_capture = lambda ep, command, timeout: (
            calls.append(command) or (0, "LABCTL_SOURCE_CURRENT=NONE")
        )
        assert generation.current(common.Endpoint("host", 22), str(base / "absent")) is None
    finally:
        generation.c.ssh_capture = original
    argv = shlex.split(calls[0])
    assert argv[-3:-1] == ["0", "0"]


def physical_owner_rejection(
    target: Path,
    attest: Callable[[], object],
    prepare_reuse: Callable[[], dict],
    finalize: Callable[[dict], object],
    reconcile: Callable[[], None],
) -> bool:
    """Exercise a real 1000:1000 owner drift when running as Linux root."""
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        return False
    original = target.lstat()
    reuse = None
    os.chown(target, 1000, 1000, follow_symlinks=False)
    os.chmod(target, 0o444, follow_symlinks=False)
    try:
        rejected = attest()
        assert rejected.returncode != 0 and "wrong publisher owner" in rejected.stderr
        reuse = prepare_reuse()
        rejected = finalize(reuse)
        assert rejected.returncode != 0 and "wrong publisher owner" in rejected.stderr
    finally:
        os.chown(
            target, original.st_uid, original.st_gid, follow_symlinks=False
        )
        os.chmod(target, original.st_mode & 0o777, follow_symlinks=False)
        if reuse is not None:
            reconcile()
    assert attest().returncode == 0
    return True
