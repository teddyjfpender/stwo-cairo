"""Provider-free hostile checks for the temporary bootstrap profile."""

from __future__ import annotations

import argparse
import contextlib
import io
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from . import acceptance
from . import bootstrap_profile as profile
from . import common as c
from . import legacy_root
from . import lifecycle
from . import provider
from . import runtime


def _rejected(call, label: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"bootstrap profile accepted invalid {label}")


def _args(**changes) -> argparse.Namespace:
    args = argparse.Namespace(bootstrap_profile=profile.NAME, confirm=None)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def _configuration_checks() -> argparse.Namespace:
    args = _args()
    profile.configure(args, {})
    assert args.gpu == "4090"
    assert args.image == profile.IMAGE
    assert args.volume_id == "2kpphx92fr" and args.volume_dc == "EU-RO-1"
    assert (args.ttl_hours, args.idle_min) == (6.0, 30)
    assert (args.max_usd_hr, args.max_total_usd) == (0.8, 4.8)
    assert (args.min_vcpu, args.min_mem_gb) == (8, 32)
    expected = {
        "formal": False,
        "image_digest": profile.IMAGE_DIGEST,
        "image_digest_authority": "requested-reference-not-runtime-attested",
        "lane": "consumer-development",
        "profile": profile.NAME,
        "qualification_eligible": False,
    }
    assert profile.metadata(args) == expected

    exact_mutations = {
        "gpu": "5090",
        "image": profile.IMAGE + "-changed",
        "volume_id": "other",
        "volume_dc": "US-1",
        "name": "other",
    }
    for field, value in exact_mutations.items():
        _rejected(
            lambda field=field, value=value: profile.configure(
                _args(**{field: value}), {}
            ),
            field,
        )
    for field, value in {
        "ttl_hours": 6.01,
        "idle_min": 31,
        "max_usd_hr": 0.81,
        "max_total_usd": 4.81,
        "min_vcpu": 7,
        "min_mem_gb": 31,
    }.items():
        _rejected(
            lambda field=field, value=value: profile.configure(
                _args(**{field: value}), {}
            ),
            field,
        )
    _rejected(
        lambda: profile.configure(
            _args(bootstrap_profile="consumer-3090-bootstrap"), {}
        ),
        "profile name",
    )

    ordinary = argparse.Namespace(bootstrap_profile=None, confirm=None)
    environment = {
        "LABCTL_IMAGE": "registry/formal@sha256:" + "a" * 64,
        "LABCTL_VOLUME_ID": "formal-volume",
        "LABCTL_VOLUME_DC": "EU-FORMAL-1",
    }
    profile.configure(ordinary, environment)
    assert ordinary.gpu == "4090" and ordinary.ttl_hours == 4.0
    assert ordinary.max_usd_hr == 1.0 and ordinary.max_total_usd == 4.0
    assert ordinary.image == environment["LABCTL_IMAGE"]
    assert profile.metadata(ordinary) == {}
    return args


def _command_checks(args: argparse.Namespace) -> None:
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
    command = profile._bootstrap_command(public_key)
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "LABCTL_BOOTSTRAP_ERROR_LINE=" in command
    assert "ubuntu:1000:/home/ubuntu:/bin/bash|ubuntu:|ubuntu:1000" in command
    assert "ubuntu:1000:/home/ubuntu:/bin/bash|dev:|ubuntu:1000" in command
    assert "dev:1000:/home/ubuntu:/bin/bash|dev:|dev:1000" in command
    assert "dev:1000:/home/dev:/bin/bash|dev:|dev:1000" in command
    assert "LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-identity" in command
    assert "initial|1000 4 20 24 25 27 29 30 44 46" in command
    assert "initial|1000') ACCOUNT_STATE=groups-cleared" in command
    assert "LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-groups" in command
    preflight = command.index("if pgrep -u 1000")
    groups_clear = command.index("usermod --groups '' ubuntu")
    group_rename = command.index("groupmod --new-name dev ubuntu")
    login_rename = command.index("usermod --login dev ubuntu")
    home_move = command.index("usermod --home /home/dev --move-home dev")
    assert preflight < groups_clear < group_rename < login_rename < home_move
    assert command.count("usermod --groups '' ubuntu") == 1
    assert "usermod --append" not in command
    assert "! getent" not in command and "! pgrep" not in command
    assert "groupadd" not in command and "useradd" not in command
    assert "groupmod --gid" not in command and "usermod --uid" not in command
    assert 'test "$(id -g dev)" = 1000' in command
    assert 'test "$(id -gn dev)" = dev' in command
    assert 'test "$(id -Gn dev)" = dev' in command
    assert "1000:1000:600:1" in command and "passwd -S dev" in command
    assert "99-stwo-consumer-bootstrap.conf" in command
    assert "sshd -t" in command and "authenticationmethods" in command
    assert "nohup sh -c" in command and "stwo-sshd-reload" in command
    assert "/run/stwo-lab-sshd-reload.receipt" in command
    assert "0:0:600:1" in command
    assert "BOOTSTRAP_PID=$$" in command
    parent_poll = command.index('while kill -0 "$parent"')
    hup = command.index('kill -HUP "$pid"')
    assert parent_poll < hup < command.index(
        'mv -T "$tmp" "$receipt"'
    )
    assert '"$BOOTSTRAP_PID"' in command
    assert "0:0:755" in command
    assert "/root/.ssh/authorized_keys" not in command

    state = {
        **profile.metadata(args),
        "bootstrap_key_sha256": "a" * 64,
        "image": profile.IMAGE,
        "persistent_root_migration": {
            "changed": True,
            "from_mode": "0777",
            "marker_from_mode": "0666",
            "marker_to_mode": "0600",
            "owner": "0:0",
            "resumed": False,
            "schema_version": legacy_root.SCHEMA,
            "to_mode": "0755",
            "volume_id": profile.VOLUME_ID,
        },
        "pod_id": "pod-test",
        "volume_dc": profile.VOLUME_DC,
        "volume_id": profile.VOLUME_ID,
    }
    record, path, digest = profile._record_command(state)
    subprocess.run(["bash", "-n"], input=record, text=True, check=True)
    assert path.endswith(f"/{profile.NAME}.json")
    assert len(digest) == 64 and "0:0:400:1" in record
    payload = profile._record_payload(state)
    assert payload["formal"] is False and payload["lane"] == profile.LANE
    assert payload["qualification_eligible"] is False
    assert payload["image_digest_authority"].startswith("requested-reference")
    assert payload["persistent_root_migration"]["changed"] is True

    guard = runtime._guard_command("pod-test", "volume-test", 60, 300)
    subprocess.run(["bash", "-n"], input=guard, text=True, check=True)
    assert "test ! -L /workspace" in guard
    assert "/workspace/gpu-lab/NETWORK_VOLUME_ID" in guard
    assert "identity != (0, 0, 0o600, 1)" in guard
    assert "os.O_NOFOLLOW" in guard and "stat.S_ISREG" in guard
    assert "tempfile.mkstemp" in guard and ".ACTIVE_ROOT.tmp" not in guard

    _rejected(
        lambda: runtime._require_open_state({"phase": "open", "pod_id": "pod-test"}),
        "open state without remote guard",
    )
    runtime._require_open_state(
        {"phase": "open", "pod_id": "pod-test", "remote_guard_installed_at": 1}
    )


def _account_state_machine_checks() -> None:
    """Execute the generated resolver against admitted and hostile databases."""
    command = profile._bootstrap_command(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
    )
    start = command.index("UID1000=$(awk")
    suffix = "\nesac\n\n# Name aliases"
    end = command.index(suffix, start) + len("\nesac")
    resolver = command[start:end]

    with tempfile.TemporaryDirectory() as directory:
        passwd_path = Path(directory) / "passwd"
        group_path = Path(directory) / "group"
        resolver = resolver.replace("/etc/passwd", shlex.quote(str(passwd_path)))
        resolver = resolver.replace("/etc/group", shlex.quote(str(group_path)))
        resolver += "\nprintf '%s|%s|%s\\n' \"$ACCOUNT_STATE\" \"$SOURCE_USER\" \"$SOURCE_HOME\"\n"

        def resolve(
            passwd: str, group: str, groups: str
        ) -> subprocess.CompletedProcess[str]:
            passwd_path.write_text(passwd)
            group_path.write_text(group)
            script = (
                f"id() {{ printf '%s\\n' {shlex.quote(groups)}; }}\n" + resolver
            )
            return subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, check=False
            )

        admitted = (
            (
                "ubuntu:x:1000:1000::/home/ubuntu:/bin/bash\n",
                "ubuntu:x:1000:\n",
                "1000 4 20 24 25 27 29 30 44 46",
                "initial|ubuntu|/home/ubuntu",
            ),
            (
                "ubuntu:x:1000:1000::/home/ubuntu:/bin/bash\n",
                "ubuntu:x:1000:\n",
                "1000",
                "groups-cleared|ubuntu|/home/ubuntu",
            ),
            (
                "ubuntu:x:1000:1000::/home/ubuntu:/bin/bash\n",
                "dev:x:1000:\n",
                "1000",
                "group-renamed|ubuntu|/home/ubuntu",
            ),
            (
                "dev:x:1000:1000::/home/ubuntu:/bin/bash\n",
                "dev:x:1000:\n",
                "1000",
                "user-renamed|dev|/home/ubuntu",
            ),
            (
                "dev:x:1000:1000::/home/dev:/bin/bash\n",
                "dev:x:1000:\n",
                "1000",
                "desired|dev|/home/dev",
            ),
        )
        for passwd, group, groups, expected in admitted:
            result = resolve(passwd, group, groups)
            assert result.returncode == 0 and result.stdout.strip() == expected

        hostile = (
            (
                admitted[0][0] + "peer:x:1000:1000::/home/peer:/bin/bash\n",
                admitted[0][1], admitted[0][2],
            ),
            (
                admitted[0][0] + "peer:x:1001:1000::/home/peer:/bin/bash\n",
                admitted[0][1], admitted[0][2],
            ),
            (admitted[0][0], admitted[0][1] + "peer:x:1000:\n", admitted[0][2]),
            (admitted[0][0], "ubuntu:x:1000:peer\n", admitted[0][2]),
            ("ubuntu:x:1000:1000::/home/ubuntu:/bin/sh\n", admitted[0][1], admitted[0][2]),
            ("ubuntu:x:1000:1000::/workspace:/bin/bash\n", admitted[0][1], admitted[0][2]),
        )
        for passwd, group, groups in hostile:
            result = resolve(passwd, group, groups)
            assert result.returncode != 0
            assert "LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-identity" in result.stderr

        for passwd, group, groups in (
            (admitted[0][0], admitted[0][1], "1000 4"),
            (admitted[2][0], admitted[2][1], admitted[0][2]),
        ):
            result = resolve(passwd, group, groups)
            assert result.returncode != 0
            assert "LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-groups" in result.stderr


def _explicit_negative_guard_checks() -> None:
    """The generated absence guards must reject explicitly under `set -e`."""
    command = profile._bootstrap_command(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
    )

    alias_start = command.index("if getent passwd dev >/dev/null; then")
    alias_end = command.index("\n    fi", alias_start) + len("\n    fi")
    alias_guard = command[alias_start:alias_end]
    present = subprocess.run(
        ["bash", "-c", "set -eu\ngetent() { return 0; }\n" + alias_guard],
        capture_output=True,
        text=True,
        check=False,
    )
    assert present.returncode != 0
    assert "LABCTL_BOOTSTRAP_CONFLICT=name-alias" in present.stderr
    absent = subprocess.run(
        ["bash", "-c", "set -eu\ngetent() { return 1; }\n" + alias_guard],
        capture_output=True,
        text=True,
        check=False,
    )
    assert absent.returncode == 0

    process_start = command.index("if pgrep -u 1000 >/dev/null; then")
    process_end = command.index("\n  fi", process_start) + len("\n  fi")
    process_guard = command[process_start:process_end]
    live = subprocess.run(
        ["bash", "-c", "set -eu\npgrep() { return 0; }\n" + process_guard],
        capture_output=True,
        text=True,
        check=False,
    )
    assert live.returncode != 0
    assert "LABCTL_BOOTSTRAP_CONFLICT=uid1000-process" in live.stderr
    quiet = subprocess.run(
        ["bash", "-c", "set -eu\npgrep() { return 1; }\n" + process_guard],
        capture_output=True,
        text=True,
        check=False,
    )
    assert quiet.returncode == 0


def _selected_key_checks() -> None:
    old_options, old_run = c.SSH_OPTS, profile.subprocess.run
    try:
        c.SSH_OPTS = ["-i", "/selected/key", "-o", "IdentitiesOnly=yes"]
        profile.subprocess.run = lambda argv, **_kw: SimpleNamespace(
            returncode=0,
            stdout="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest ignored-comment\n",
        )
        assert profile._public_key() == "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
        c.SSH_OPTS = ["-i", "/one", "-i", "/two"]
        _rejected(profile._public_key, "ambiguous selected key")
        c.SSH_OPTS = ["-i", "/selected/key"]
        profile.subprocess.run = lambda argv, **_kw: SimpleNamespace(
            returncode=0, stdout="not-a-key\n"
        )
        _rejected(profile._public_key, "invalid derived public key")
    finally:
        c.SSH_OPTS, profile.subprocess.run = old_options, old_run


def _reload_receipt_checks() -> None:
    saved = c.ssh_capture
    calls = []
    key_digest = "a" * 64
    try:
        def accepted(ep, command, *, timeout):
            calls.append((ep.user, command, timeout))
            if ep.user == "root":
                assert command.index("RELOAD_RECEIPT=") < command.index("POLICY=")
                assert "0:0:600:1" in command
                assert "EXPECTED_RELOAD=" in command
                assert "1000:dev" in command
                return 0, "\n".join((
                    "LABCTL_FRESH_ROOT_POLICY=publickey-only",
                    f"LABCTL_FRESH_ROOT_KEY_SHA256={key_digest}",
                ))
            assert "1000:1000:dev:dev:/home/dev" in command
            assert 'test "$(id -Gn)" = dev' in command
            return 0, ""

        c.ssh_capture = accepted
        profile.verify_ssh(c.Endpoint("host", 22), key_digest)
        assert [user for user, _command, _timeout in calls] == ["root", "dev"]

        calls.clear()
        c.ssh_capture = lambda ep, command, *, timeout: (
            calls.append(ep.user) or (1, "missing reload receipt")
        )
        _rejected(
            lambda: profile.verify_ssh(c.Endpoint("host", 22), key_digest),
            "missing reload receipt",
        )
        assert calls == ["root"]
        _rejected(
            lambda: profile.verify_ssh(c.Endpoint("host", 22), "not-a-digest"),
            "bootstrap key digest",
        )
        assert calls == ["root"]
    finally:
        c.ssh_capture = saved


def _reload_parent_exit_order_check() -> None:
    """Execute the generated worker and prove four markers precede HUP."""
    command = profile._bootstrap_command(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"
    )
    prefix = "nohup sh -c '\n"
    start = command.index(prefix) + len(prefix)
    end = command.index("\n' stwo-sshd-reload", start)
    worker = command[start:end]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        receipt = root / "reload.receipt"
        order = root / "order.log"
        worker = worker.replace(
            "tmp=$(mktemp /run/.stwo-lab-sshd-reload.XXXXXX)",
            f"tmp=$(mktemp {shlex.quote(str(root / 'reload.XXXXXX'))})",
        )
        worker = worker.replace('chown 0:0 "$tmp"', ":")
        worker = worker.replace(
            'kill -HUP "$pid"', 'printf "HUP\\n" >> "$ORDER_LOG"'
        )
        worker = worker.replace('kill -0 "$pid"', ":")
        worker = worker.replace('mv -T "$tmp" "$receipt"', 'mv "$tmp" "$receipt"')
        outer = f"""
set -eu
ORDER_LOG={shlex.quote(str(order))}
export ORDER_LOG
parent=$$
nohup sh -c {shlex.quote(worker)} stwo-test 999 \
  {shlex.quote(str(receipt))} token "$parent" </dev/null >/dev/null 2>&1 &
printf '%s\n' MARKER-1 MARKER-2 MARKER-3 MARKER-4 >> "$ORDER_LOG"
"""
        result = subprocess.run(
            ["bash", "-c", outer], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        for _ in range(50):
            if receipt.exists():
                break
            time.sleep(0.05)
        assert receipt.read_text() == "token\n"
        assert order.read_text().splitlines() == [
            "MARKER-1", "MARKER-2", "MARKER-3", "MARKER-4", "HUP"
        ]


def _ordering_checks(args: argparse.Namespace) -> None:
    old_state = c.STATE
    saved = (
        legacy_root.migrate,
        profile.bootstrap_dev,
        profile.verify_ssh,
        profile.persist_record,
        c.ssh_capture,
    )
    calls = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            legacy_root.migrate = lambda *_a: (
                calls.append("root-migration")
                or {
                    "changed": True,
                    "from_mode": "0777",
                    "owner": "0:0",
                    "schema_version": legacy_root.SCHEMA,
                    "to_mode": "0755",
                    "volume_id": profile.VOLUME_ID,
                }
            )
            profile.bootstrap_dev = lambda _ep: calls.append("bootstrap") or "a" * 64
            profile.verify_ssh = lambda _ep, _key: calls.append("fresh-root+dev")
            c.ssh_capture = lambda *_a, **_kw: (
                calls.append("guard")
                or (0, "LABCTL_REMOTE_GUARD_INSTALLED=pod-test")
            )
            profile.persist_record = (
                lambda *_a: calls.append("record")
                or ("/workspace/profile.json", "b" * 64)
            )
            state = {
                **profile.metadata(args),
                "phase": "bootstrapping",
                "pod_id": "pod-test",
                "volume_id": profile.VOLUME_ID,
            }
            lifecycle._install_remote_controls(
                state, args, c.Endpoint("host", 22), 3600
            )
            assert calls == [
                "root-migration", "bootstrap", "fresh-root+dev", "guard", "record"
            ]
            assert c._read_state()["phase"] == "bootstrapping"
            assert c._read_state()["persistent_root_migration"]["changed"] is True

            calls.clear()
            legacy_root.migrate = lambda *_a: (
                calls.append("root-migration-failed"),
                (_ for _ in ()).throw(RuntimeError("hostile legacy root")),
            )[1]
            _rejected(
                lambda: lifecycle._install_remote_controls(
                    state, args, c.Endpoint("host", 22), 3600
                ),
                "hostile legacy root",
            )
            assert calls == ["root-migration-failed"]

            calls.clear()
            legacy_root.migrate = lambda *_a: (
                calls.append("root-migration")
                or {
                    "changed": False,
                    "from_mode": "0755",
                    "owner": "0:0",
                    "schema_version": legacy_root.SCHEMA,
                    "to_mode": "0755",
                    "volume_id": profile.VOLUME_ID,
                }
            )
            profile.verify_ssh = lambda _ep, _key: (
                calls.append("fresh-root+dev-failed"),
                (_ for _ in ()).throw(RuntimeError("receipt missing")),
            )[1]
            _rejected(
                lambda: lifecycle._install_remote_controls(
                    state, args, c.Endpoint("host", 22), 3600
                ),
                "failed fresh SSH receipt",
            )
            assert calls == [
                "root-migration", "bootstrap", "fresh-root+dev-failed"
            ]

            calls.clear()
            profile.verify_ssh = lambda _ep, _key: calls.append("fresh-root+dev")
            diagnostic = "LABCTL_GUARD_ERROR phase=local-layout line=99 rc=1"
            c.ssh_capture = lambda *_a, **_kw: (
                calls.append("guard-failed") or (1, diagnostic)
            )
            try:
                lifecycle._install_remote_controls(
                    state, args, c.Endpoint("host", 22), 3600
                )
            except RuntimeError as error:
                assert diagnostic in str(error)
            else:
                raise AssertionError("failed guard was accepted")
            assert calls == [
                "root-migration", "bootstrap", "fresh-root+dev", "guard-failed"
            ]

            for invalid in (
                "",
                "LABCTL_REMOTE_GUARD_INSTALLED=other-pod",
                "LABCTL_REMOTE_GUARD_INSTALLED=pod-test\nunexpected-output",
            ):
                calls.clear()
                c.ssh_capture = lambda *_a, invalid=invalid, **_kw: (
                    calls.append("guard-invalid") or (0, invalid)
                )
                _rejected(
                    lambda: lifecycle._install_remote_controls(
                        state, args, c.Endpoint("host", 22), 3600
                    ),
                    "invalid guard completion authority",
                )
                assert calls == [
                    "root-migration", "bootstrap", "fresh-root+dev", "guard-invalid"
                ]
    finally:
        (
            legacy_root.migrate,
            profile.bootstrap_dev,
            profile.verify_ssh,
            profile.persist_record,
            c.ssh_capture,
        ) = saved
        c.STATE = old_state


def _nonformal_gate_check() -> None:
    saved = runtime._active
    try:
        runtime._active = lambda: (
            {"formal": False},
            object(),
            c.Endpoint("host", 22),
        )
        _rejected(
            lambda: acceptance.cmd_accept(argparse.Namespace(profile=True)),
            "formal acceptance",
        )
    finally:
        runtime._active = saved


def _bootstrapping_close_check() -> None:
    old_state = c.STATE
    saved = (
        c.api.get_pod,
        c.api.list_pods,
        provider._terminate_pod_once,
        runtime._persist_for_termination,
        c.ledger.append,
    )
    pod = c.api.PodInfo(
        id="pod-test", name="bootstrap", status="RUNNING", cost_per_hr=0.69,
        gpu="NVIDIA GeForce RTX 4090", dc=profile.VOLUME_DC, vcpu=8,
        mem_gb=32, ssh_host="host", ssh_port=22, raw={"gpuCount": 1},
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            c.STATE = Path(directory) / "lease.json"
            state = {
                "expires_at": 9e12,
                "lease_name": pod.name,
                "phase": "bootstrapping",
                "pod_id": pod.id,
                "volume_id": profile.VOLUME_ID,
            }
            c._write_state(state)
            c.api.get_pod = lambda _pod_id: pod
            c.api.list_pods = lambda: [pod]
            terminated = []
            provider._terminate_pod_once = terminated.append
            runtime._persist_for_termination = lambda *_a, **_kw: (_ for _ in ()).throw(
                AssertionError("bootstrapping close attempted formal persistence")
            )
            c.ledger.append = lambda *_a, **_kw: None
            token = c._token(
                "CLOSE",
                {"lease_name": pod.name, "pod_id": pod.id,
                 "volume_id": profile.VOLUME_ID},
            )
            with contextlib.redirect_stdout(io.StringIO()):
                assert lifecycle.cmd_close(argparse.Namespace(confirm=token)) == 0
            assert terminated == [pod.id] and not c.STATE.exists()
    finally:
        (
            c.api.get_pod,
            c.api.list_pods,
            provider._terminate_pod_once,
            runtime._persist_for_termination,
            c.ledger.append,
        ) = saved
        c.STATE = old_state


def bootstrap_profile_self_test() -> None:
    args = _configuration_checks()
    _selected_key_checks()
    _command_checks(args)
    _account_state_machine_checks()
    _explicit_negative_guard_checks()
    _reload_parent_exit_order_check()
    _reload_receipt_checks()
    _ordering_checks(args)
    _nonformal_gate_check()
    _bootstrapping_close_check()
