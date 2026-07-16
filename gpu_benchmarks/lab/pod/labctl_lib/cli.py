"""Stable command-line surface and lock routing."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import acceptance
from . import common as c
from . import lifecycle
from . import selftest
from . import sync


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="labctl",
        description="bounded GPU lab lease; never launches a proof",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("open", help="preview/create one bounded Secure Cloud lease")
    p.add_argument(
        "--gpu",
        default="4090",
        help="3090|4090|5090|a5000|a40|l40s|h100 or RunPod id",
    )
    p.add_argument("--image", default=os.environ.get("LABCTL_IMAGE"))
    p.add_argument("--volume-id", default=os.environ.get("LABCTL_VOLUME_ID"))
    p.add_argument("--volume-dc", default=os.environ.get("LABCTL_VOLUME_DC"))
    p.add_argument("--name")
    p.add_argument("--ttl-hours", type=float, default=4.0)
    p.add_argument("--idle-min", type=int, default=30)
    p.add_argument("--max-usd-hr", type=float, default=1.0)
    p.add_argument("--max-total-usd", type=float, default=4.0)
    p.add_argument("--min-vcpu", type=int, default=8)
    p.add_argument("--min-mem-gb", type=int, default=32)
    p.add_argument("--ready-timeout", type=float, default=600)
    p.add_argument("--confirm", help="exact token printed by the preview")
    sub.add_parser("status")
    sub.add_parser("sync", help="publish an exact immutable source generation")
    sub.add_parser("heartbeat", help="renew the remote interactive-idle deadline")
    sub.add_parser("shell")
    p = sub.add_parser("accept", help="record readiness; --profile runs a real ncu kernel")
    p.add_argument("--profile", action="store_true")
    p = sub.add_parser("close", help="seal records and terminate compute")
    p.add_argument("--confirm", help="exact token printed by the preview")
    sub.add_parser("self-test", help="local guard/budget checks; no network")
    p = sub.add_parser("_watch", help=argparse.SUPPRESS)
    p.add_argument("--lease-name", required=True)
    p.add_argument("--created-at", type=float, required=True)
    p.add_argument("--expires-at", type=float, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {
        "open": lifecycle.cmd_open,
        "status": lifecycle.cmd_status,
        "sync": sync.cmd_sync,
        "heartbeat": acceptance.cmd_heartbeat,
        "shell": acceptance.cmd_shell,
        "accept": acceptance.cmd_accept,
        "close": lifecycle.cmd_close,
        "self-test": selftest.cmd_self_test,
        "_watch": lifecycle.cmd_watch,
    }
    try:
        if args.cmd in {"self-test", "_watch"}:
            return commands[args.cmd](args)
        if args.cmd in {"sync", "heartbeat", "shell", "accept"}:
            with c._operation_lock():
                return commands[args.cmd](args)
        if args.cmd == "close":
            with c._operation_lock():
                with c._lease_lock():
                    return commands[args.cmd](args)
        with c._lease_lock():
            return commands[args.cmd](args)
    except (
        c.api.ApiError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        OSError,
    ) as error:
        print(f"labctl: {error}", file=sys.stderr)
        return 1
