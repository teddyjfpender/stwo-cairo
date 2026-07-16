"""Offline regressions for dev profile admission and root receipt binding."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from . import acceptance
from . import common as c


def _rejected(call, label: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"accepted invalid {label}")


def _receipt() -> str:
    return "\n".join(
        (
            "LABCTL_PROFILE_USER=dev",
            "LABCTL_PROFILE_UID=1000",
            "LABCTL_PROFILE_GID=1000",
            "LABCTL_PROFILE_NVCC=/usr/local/cuda/bin/nvcc",
            "LABCTL_PROFILE_NCU=/usr/local/cuda/bin/ncu",
            "LABCTL_KERNEL_RESULT=1",
            "LABCTL_NCU_METRIC=123.5",
        )
    )


def _check_receipt_parser(profile_receipt: str) -> str:
    receipt_digest = hashlib.sha256(profile_receipt.encode()).hexdigest()
    accept_output = "\n".join(
        (
            "LABCTL_NSYS_PACKAGE=" + acceptance.NSYS_IDENTITY,
            "LABCTL_DEV_LAYOUT=" + acceptance.DEV_LAYOUT_IDENTITY,
            acceptance.PROFILE_RECEIPT_SHA256 + "=" + receipt_digest,
            acceptance.PROFILE_RECEIPT_BEGIN,
            profile_receipt,
            acceptance.PROFILE_RECEIPT_END,
            "LABCTL_RECORD=/tmp/stwo-gpu-lab/pod-test/records/accept-profile-abc.txt",
            "LABCTL_RECORD_SHA256=" + "b" * 64,
            "LABCTL_PERSIST_MANIFEST=/workspace/gpu-lab/leases/pod-test/persists/"
            + "c" * 64
            + ".json",
            "LABCTL_PERSIST_SHA256=" + "c" * 64,
            "LABCTL_PERSIST_ENTRIES=1",
            "LABCTL_SEALED=0",
        )
    )
    record, digest, metric, parsed_receipt_digest, persisted = \
        acceptance._parse_acceptance(accept_output, profile=True)
    assert record.endswith("abc.txt") and digest == "b" * 64 and metric == 123.5
    assert parsed_receipt_digest == receipt_digest
    assert persisted["manifest_sha256"] == "c" * 64 and persisted["entries"] == 1
    _rejected(
        lambda: acceptance._parse_acceptance(
            accept_output.replace("LABCTL_NCU_METRIC=123.5\n", ""), profile=True
        ),
        "missing ncu metric",
    )
    _rejected(
        lambda: acceptance._parse_acceptance(
            accept_output.replace("LABCTL_PROFILE_UID=1000", "LABCTL_PROFILE_UID=0"),
            profile=True,
        ),
        "root profile receipt",
    )
    _rejected(
        lambda: acceptance._profile_metric(
            profile_receipt + "\nLABCTL_RECORD=/tmp/injected"
        ),
        "profile record injection",
    )
    _rejected(
        lambda: acceptance._parse_acceptance(
            accept_output.replace(receipt_digest, "0" * 64), profile=True
        ),
        "changed profile receipt digest",
    )
    _rejected(
        lambda: acceptance._parse_acceptance(
            accept_output.replace(acceptance.NSYS_IDENTITY, "wrong=0"), profile=True
        ),
        "wrong Nsight Systems package",
    )
    _rejected(
        lambda: acceptance._parse_acceptance(
            accept_output.replace(acceptance.DEV_LAYOUT_IDENTITY, "wrong-layout"),
            profile=True,
        ),
        "wrong dev layout",
    )
    return receipt_digest


def _check_generated_commands(valid_image: str, receipt: str, digest: str) -> None:
    command = acceptance._accept_command(
        {"image": valid_image, "pod_id": "pod-test", "volume_id": "volume-test"},
        profile=True,
        profile_receipt=receipt,
    )
    counter = acceptance.COUNTER_CMD
    assert "mktemp -d /tmp/labctl-counter.XXXXXX" in counter
    assert "stat -c '%u:%g:%a' \"$TMP\"" in counter
    assert "1000:1000:700" in counter and "LABCTL_PROFILE_USER=dev" in counter
    assert "EXPECTED_PATH=" in counter
    assert "command -v nvcc" in counter and "command -v ncu" in counter
    assert 'realpath -e "$TMP"' in counter
    assert 'trap \'rm -rf -- "$TMP"\' EXIT' in counter
    assert 'cat > "$SOURCE"' in counter and '"$BINARY" > "$LOG"' in counter
    for unsafe_path in (
        "/tmp/labctl-counter.cu",
        "/tmp/labctl-counter.log",
        "/tmp/labctl-counter >",
    ):
        assert unsafe_path not in counter
    assert acceptance.PROFILE_RECEIPT_BEGIN in command
    assert digest in command and "ncu --target-processes" not in command
    assert "findmnt" in command and "NETWORK_VOLUME_ID" in command
    assert "LABCTL_DEV_LAYOUT=uid1000-gid1000-root0710-build0700-fixtures0700" \
        in command
    assert "DEV_LAYOUT.json" in command and "1000:1000:700" in command
    assert acceptance.NSYS_IDENTITY in command and "nsys --version" in command
    for generated in (counter, command):
        subprocess.run(["bash", "-n"], input=generated, text=True, check=True)


def _check_dev_transport(profile_receipt: str) -> None:
    original = c.ssh_capture
    calls = []
    try:
        def capture(ep, command, timeout):
            calls.append((ep, command, timeout))
            return 0, profile_receipt

        c.ssh_capture = capture
        collected, metric = acceptance._collect_dev_profile(c.Endpoint("host", 22))
        assert collected == profile_receipt and metric == 123.5
        assert len(calls) == 1 and calls[0][0].user == "dev"
        assert calls[0][1] == acceptance.COUNTER_CMD
    finally:
        c.ssh_capture = original


def _check_ncu_csv_parser() -> None:
    parser = acceptance.COUNTER_CMD.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    with tempfile.NamedTemporaryFile("w", delete=False) as ncu_csv:
        ncu_csv.write(
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"1","labctl_counter(int *)","sm__cycles_elapsed.avg",'
            '"cycle","12,345.5"\n'
        )
        path = Path(ncu_csv.name)
    try:
        parsed = subprocess.run(
            [sys.executable, "-c", parser, str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert parsed == "LABCTL_NCU_METRIC=12345.5"
    finally:
        path.unlink()


def acceptance_self_test(valid_image: str) -> None:
    shell_argv = acceptance._shell_argv(c.Endpoint("host", 22))
    assert shell_argv[-1] == "dev@host" and "root@host" not in shell_argv
    assert shell_argv.count("-i") == 1 and "IdentitiesOnly=yes" in shell_argv
    receipt = _receipt()
    digest = _check_receipt_parser(receipt)
    _check_generated_commands(valid_image, receipt, digest)
    _check_dev_transport(receipt)
    _check_ncu_csv_parser()
