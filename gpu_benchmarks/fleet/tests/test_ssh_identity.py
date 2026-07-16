from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

FLEET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_DIR))

from gpufleet import podctl


class SshIdentityTests(unittest.TestCase):
    def test_no_key_is_rejected_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"RUNPOD_SSH_KEY": ""}
        ), mock.patch.object(podctl.Path, "home", return_value=Path(directory)):
            with self.assertRaisesRegex(ValueError, "no SSH private key"):
                podctl._ssh_opts()

    def test_ambiguous_default_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".runpod" / "ssh"
            root.mkdir(parents=True)
            for name in podctl.SSH_KEY_NAMES[:2]:
                key = root / name
                key.touch(mode=0o600)
            with mock.patch.dict(os.environ, {"RUNPOD_SSH_KEY": ""}), mock.patch.object(
                podctl.Path, "home", return_value=Path(directory)
            ):
                with self.assertRaisesRegex(ValueError, "multiple SSH private keys"):
                    podctl._ssh_opts()

    def test_group_or_world_key_permissions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key"
            key.touch(mode=0o644)
            with mock.patch.dict(os.environ, {"RUNPOD_SSH_KEY": str(key)}):
                with self.assertRaisesRegex(ValueError, "group or world"):
                    podctl._ssh_opts()

    def test_symlink_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key"
            link = Path(directory) / "link"
            key.touch(mode=0o600)
            link.symlink_to(key)
            with mock.patch.dict(os.environ, {"RUNPOD_SSH_KEY": str(link)}):
                with self.assertRaisesRegex(ValueError, "not a symlink"):
                    podctl._ssh_opts()

    def test_explicit_safe_key_is_exact_and_contents_are_not_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "explicit"
            key.write_text("deliberately not an SSH key\n")
            key.chmod(0o600)
            with mock.patch.dict(os.environ, {"RUNPOD_SSH_KEY": str(key)}):
                options = podctl._ssh_opts()
            self.assertEqual(options[:4], ["-i", str(key), "-o", "IdentitiesOnly=yes"])
            self.assertEqual(options.count("-i"), 1)


if __name__ == "__main__":
    unittest.main()
