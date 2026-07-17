"""Provider-free hostile checks for the legacy persistent-root migration."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from . import common as c
from . import legacy_root


def _run(
    root: Path,
    volume_id: str = "volume-test",
    *,
    uid: int | None = None,
    gid: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            "-c",
            legacy_root.WORKER,
            str(root),
            volume_id,
            str(os.getuid() if uid is None else uid),
            str(os.getgid() if gid is None else gid),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _fixture(
    base: Path, *, mode: int = 0o777, marker_mode: int = 0o666
) -> tuple[Path, Path]:
    root = base / "gpu-lab"
    root.mkdir()
    root.chmod(mode)
    marker = root / legacy_root.MARKER
    marker.write_text("volume-test\n")
    marker.chmod(marker_mode)
    return root, marker


def _reject(root: Path, label: str, **kwargs) -> None:
    result = _run(root, **kwargs)
    assert result.returncode != 0, f"accepted hostile {label}: {result.stdout}"


def legacy_root_self_test() -> None:
    compile(legacy_root.WORKER, "labctl-legacy-root-worker", "exec")
    generated = legacy_root.command("volume-test")
    assert "mountpoint -q /workspace" in generated
    assert "test ! -L /workspace" in generated
    assert "os.O_NOFOLLOW" in generated
    assert "os.fchmod(root_fd, 0o700)" in generated
    assert "os.fchmod(marker_fd, 0o600)" in generated
    assert "os.fchmod(root_fd, 0o755)" in generated
    assert "os.fsync(root_fd)" in generated
    assert "time.sleep(0.05)" in generated
    assert "chmod -R" not in generated and "chown" not in generated

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root, marker = _fixture(base)
        child = root / "untouched"
        child.write_text("sentinel\n")
        child.chmod(0o666)
        before = root.stat()
        result = _run(root)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith(legacy_root.CHANGED)
        assert root.stat().st_mode & 0o777 == 0o755
        assert marker.stat().st_mode & 0o777 == 0o600
        assert marker.read_text() == "volume-test\n"
        assert child.read_text() == "sentinel\n"
        assert child.stat().st_mode & 0o777 == 0o666

        again = _run(root)
        assert again.returncode == 0, again.stderr
        assert again.stdout.strip().endswith(legacy_root.UNCHANGED)
        after = root.stat()
        assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
        )

    for root_mode, marker_mode, receipt in (
        (0o777, 0o600, legacy_root.RESUMED_SECURED_MARKER),
        (0o700, 0o666, legacy_root.RESUMED_MARKER),
        (0o700, 0o600, legacy_root.RESUMED_ROOT),
    ):
        with tempfile.TemporaryDirectory() as directory:
            root, marker = _fixture(
                Path(directory), mode=root_mode, marker_mode=marker_mode
            )
            result = _run(root)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip().endswith(receipt)
            assert root.stat().st_mode & 0o777 == 0o755
            assert marker.stat().st_mode & 0o777 == 0o600

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        _reject(base / "absent", "absent root")
        real, _ = _fixture(base)
        link = base / "root-link"
        link.symlink_to(real, target_is_directory=True)
        _reject(link, "root symlink")

    for mutation, label in (
        (lambda root, marker: root.chmod(0o775), "root mode"),
        (lambda root, marker: marker.unlink(), "absent marker"),
        (lambda root, marker: marker.chmod(0o644), "marker mode"),
        (lambda root, marker: marker.write_text("wrong\n"), "marker content"),
    ):
        with tempfile.TemporaryDirectory() as directory:
            root, marker = _fixture(Path(directory))
            mutation(root, marker)
            _reject(root, label)

    with tempfile.TemporaryDirectory() as directory:
        root, marker = _fixture(Path(directory))
        other = root / "marker-hardlink"
        os.link(marker, other)
        _reject(root, "hardlinked marker")

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = base / "gpu-lab"
        root.mkdir()
        root.chmod(0o777)
        target = base / "target"
        target.write_text("volume-test\n")
        (root / legacy_root.MARKER).symlink_to(target)
        _reject(root, "marker symlink")

    with tempfile.TemporaryDirectory() as directory:
        root, _ = _fixture(Path(directory))
        _reject(root, "root owner", uid=os.getuid() + 1)

    saved = c.ssh_capture
    try:
        c.ssh_capture = lambda *_a, **_kw: (
            0,
            f"LABCTL_LEGACY_ROOT_MIGRATION={legacy_root.CHANGED}\n",
        )
        evidence = legacy_root.migrate(c.Endpoint("host", 22), "volume-test")
        assert evidence == {
            "changed": True,
            "from_mode": "0777",
            "marker_from_mode": "0666",
            "marker_to_mode": "0600",
            "owner": "0:0",
            "resumed": False,
            "schema_version": legacy_root.SCHEMA,
            "to_mode": "0755",
            "volume_id": "volume-test",
        }
        c.ssh_capture = lambda *_a, **_kw: (0, "unexpected\n")
        try:
            legacy_root.migrate(c.Endpoint("host", 22), "volume-test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted invalid migration receipt")
    finally:
        c.ssh_capture = saved
