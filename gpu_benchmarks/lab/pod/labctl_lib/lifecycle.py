"""User-facing lease lifecycle commands."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from . import bootstrap_profile
from . import common as c
from . import provider
from . import runtime


REMOTE_OBSERVATION_FAILURE_LIMIT = 3


def _launch_plan(args: argparse.Namespace, offer: dict, volume: dict) -> dict:
    """Bind every mutable launch input into the confirmation token."""
    return {
        "cloud": "SECURE",
        "container_disk_gb": c.DEFAULT_DISK_GB,
        "gpu_count": 1,
        "qualification_eligible": True,
        "gpu_display_name": offer["display_name"],
        "gpu_type_id": offer["gpu_type_id"],
        "image": args.image,
        "idle_minutes": args.idle_min,
        "max_total_usd": args.max_total_usd,
        "max_usd_hr": args.max_usd_hr,
        "min_memory_gb": args.min_mem_gb,
        "min_vcpu_count": args.min_vcpu,
        "name_prefix": args.name or f"stwo-gpu-lab-{args.gpu.lower()}",
        "ports": "22/tcp",
        "quoted_secure_usd_hr": offer["usd_hr"],
        "ready_timeout_seconds": args.ready_timeout,
        "start_ssh": True,
        "termination_model": "local-api-idle+ttl-watchdog+remote-idle+ttl-exit",
        "ttl_hours": args.ttl_hours,
        "volume_dc": args.volume_dc,
        "volume_gb": 0,
        "volume_id": args.volume_id,
        "volume_mount": c.VOLUME_MOUNT,
        "volume_rest_attestation": volume,
        **bootstrap_profile.metadata(args),
    }


def _validate_created_pod(pod: c.api.PodInfo, lease_name: str, offer: dict, args) -> None:
    if not c.ID_RE.fullmatch(pod.id):
        raise RuntimeError(f"created pod returned an unsafe id: {pod.id!r}")
    if pod.name != lease_name:
        raise RuntimeError(f"created pod name mismatch: {pod.name!r} != {lease_name!r}")
    if pod.gpu != offer["display_name"]:
        raise RuntimeError(f"created GPU mismatch: {pod.gpu!r} != {offer['display_name']!r}")
    if pod.dc != args.volume_dc:
        raise RuntimeError(
            f"volume placement mismatch: pod dc {pod.dc!r} != {args.volume_dc!r}"
        )
    if pod.vcpu < args.min_vcpu or pod.mem_gb < args.min_mem_gb:
        raise RuntimeError(
            "created pod resource floor mismatch: "
            f"vcpu={pod.vcpu}/{args.min_vcpu}, memory_gb={pod.mem_gb}/{args.min_mem_gb}"
        )
    if int(pod.raw.get("gpuCount") or 0) != 1:
        raise RuntimeError(f"created pod GPU count mismatch: {pod.raw.get('gpuCount')!r}")


def _cleanup_failed_open(reservation: dict, lease_name: str, pod) -> None:
    """Reconcile a possibly accepted create mutation before releasing state."""
    candidates = {pod.id: pod} if pod else {}
    reconciliation_ok = False
    try:
        for match in c.api.list_pods():
            if match.name == lease_name:
                candidates[match.id] = match
        reconciliation_ok = True
    except Exception as error:
        reservation["reconciliation_error"] = str(error)
    cleanup_failed = []
    for candidate in candidates.values():
        try:
            provider._terminate_pod_once(candidate.id)
            c.ledger.append(
                "terminate", pod_id=candidate.id, gpu=candidate.gpu, note="open-failure"
            )
        except Exception as error:
            cleanup_failed.append(candidate.id)
            print(f"URGENT: cleanup failed for {candidate.id}: {error}", file=sys.stderr)
    if reconciliation_ok and not cleanup_failed:
        runtime._cancel_watchdog(reservation)
        c.STATE.unlink(missing_ok=True)
    else:
        reservation["orphan_ids"] = cleanup_failed or list(candidates)
        reservation["phase"] = "cleanup_failed" if cleanup_failed else "creating"
        c._write_state(reservation)


def _install_remote_controls(state: dict, args, ep: c.Endpoint, remaining: int) -> None:
    """Finish bootstrap before the ordinary remote guards make the lease open."""
    bootstrap = bool(bootstrap_profile.metadata(args))
    if bootstrap:
        state["bootstrap_key_sha256"] = bootstrap_profile.bootstrap_dev(ep)
        c._write_state(state)
        bootstrap_profile.verify_ssh(ep, state["bootstrap_key_sha256"])
    if c.ssh_run(
        ep,
        runtime._guard_command(
            state["pod_id"], state["volume_id"], remaining, args.idle_min * 60
        ),
        timeout=60,
    ):
        raise RuntimeError("failed to install remote TTL termination guard")
    if bootstrap:
        record, digest = bootstrap_profile.persist_record(ep, state)
        state["profile_record"] = record
        state["profile_record_sha256"] = digest


def cmd_open(args: argparse.Namespace) -> int:
    if c.STATE.exists():
        existing = c._read_state()
        if (
            existing.get("phase")
            in {"open", "creating", "bootstrapping", "cleanup_failed"}
            and time.time() >= existing.get("expires_at", math.inf)
        ):
            runtime._terminate_all_state_candidates(
                existing, reason="open-command-expiry-enforcement"
            )
        raise RuntimeError(f"active lease state already exists: {c.STATE}")
    c._validate_open_args(args)
    gpu_id = c.GPU_IDS.get(args.gpu.lower(), args.gpu)
    offer = provider._secure_offer(gpu_id)
    price = offer["usd_hr"]
    c._check_budget(price, args.ttl_hours, args.max_usd_hr, args.max_total_usd)
    volume = provider._network_volume_attestation(args.volume_id, args.volume_dc)
    plan = _launch_plan(args, offer, volume)
    token = c._token("OPEN", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(
        "TTL LIMITATION: RunPod exposes no server-side lease in this client. "
        "The local API watchdog and remote container-exit guard are independent "
        "best efforts, not a hard provider guarantee."
    )
    if args.confirm != token:
        print(f"NO POD LAUNCHED. Re-run with: --confirm {token}")
        return 2

    # Resolve and validate the one exact private key before watchdog state or a
    # billable provider mutation. SSH transport itself resolves the same lazy
    # option object again for every root, dev, and rsync invocation.
    tuple(c.SSH_OPTS)
    created = time.time()
    pod = None
    lease_name = f"{plan['name_prefix']}-{token[-8:].lower()}"
    if any(candidate.name == lease_name for candidate in c.api.list_pods()):
        raise RuntimeError(f"refusing duplicate lease name already in account: {lease_name}")
    reservation = {
        "created_at": created,
        "expires_at": created + args.ttl_hours * 3600,
        "lease_name": lease_name,
        "max_total_usd": args.max_total_usd,
        "max_usd_hr": args.max_usd_hr,
        "phase": "creating",
        "plan": plan,
        "usd_hr": price,
        "volume_id": args.volume_id,
        **bootstrap_profile.metadata(args),
    }
    c._write_state(reservation)
    try:
        reservation["watchdog_pid"] = runtime._spawn_watchdog(reservation)
    except Exception:
        c.STATE.unlink(missing_ok=True)
        raise
    c._write_state(reservation)
    try:
        pod = provider._create_pod_once(name=lease_name, gpu_id=gpu_id, args=args)
        _validate_created_pod(pod, lease_name, offer, args)
        attached_volume = provider._attest_pod_volume(
            pod.id, args.volume_id, args.volume_dc
        )
        account_matches = [
            candidate for candidate in c.api.list_pods() if candidate.name == lease_name
        ]
        if len(account_matches) != 1 or account_matches[0].id != pod.id:
            raise RuntimeError(
                f"account-level lease-name collision for {lease_name}: "
                + ",".join(candidate.id for candidate in account_matches)
            )
        actual_price = pod.cost_per_hr
        c._check_budget(actual_price, args.ttl_hours, args.max_usd_hr, args.max_total_usd)
        state = {
            "accepted": [],
            "created_at": created,
            "expires_at": reservation["expires_at"],
            "gpu": pod.gpu,
            "image": args.image,
            "idle_minutes": args.idle_min,
            "lease_name": lease_name,
            "max_total_usd": args.max_total_usd,
            "max_usd_hr": args.max_usd_hr,
            "phase": "bootstrapping",
            "qualification_eligible": True,
            "plan": plan,
            "pod_id": pod.id,
            "usd_hr": actual_price,
            "volume_dc": args.volume_dc,
            "volume_id": args.volume_id,
            "volume_rest_attestation": attached_volume,
            "watchdog_pid": reservation["watchdog_pid"],
            **bootstrap_profile.metadata(args),
        }
        c._write_state(state)
        c.ledger.append(
            "create",
            pod_id=pod.id,
            gpu=pod.gpu,
            usd_hr=actual_price,
            purpose="gpu-lab-bounded-lease",
            **bootstrap_profile.metadata(args),
        )
        pod = c.wait_ready(pod.id, timeout_s=args.ready_timeout)
        remaining = int(state["expires_at"] - time.time())
        required = args.idle_min * 60 + 30
        if remaining < required:
            raise RuntimeError(
                "lease has too little time remaining to keep idle and TTL guards separate"
            )
        ep = c.Endpoint.of(pod)
        _install_remote_controls(state, args, ep, remaining)
        state["phase"] = "open"
        state["remote_guard_installed_at"] = time.time()
        c._write_state(state)
        print(f"OPEN {pod.id} {ep.host}:{ep.port}; expires in <= {remaining}s")
        print(
            "TTL MODEL: local API watchdog + remote container-exit guard; "
            "a true hard TTL requires a RunPod server-side lease facility."
        )
        return 0
    except BaseException:
        _cleanup_failed_open(reservation, lease_name, pod)
        raise


def cmd_status(_args: argparse.Namespace) -> int:
    state = c._read_state()
    if state["phase"] != "open":
        if (
            state["phase"] in {"creating", "bootstrapping", "cleanup_failed"}
            and time.time() >= state.get("expires_at", math.inf)
        ):
            runtime._terminate_all_state_candidates(
                state, reason="status-ambiguous-expiry-enforcement"
            )
            state = c._read_state()
        out = dict(state)
        if state["phase"] in {"creating", "bootstrapping", "cleanup_failed"}:
            matches = [
                pod for pod in c.api.list_pods() if pod.name == state.get("lease_name")
            ]
            out["matching_pods"] = [pod.id for pod in matches]
        print(json.dumps(out, indent=2, sort_keys=True))
        return 1

    pod = c.api.get_pod(state["pod_id"])
    now = time.time()
    if pod:
        try:
            runtime._check_live_budget(state, pod)
        except RuntimeError as error:
            runtime._terminate_state_pod(
                state, pod, reason="status-budget-ceiling-exceeded"
            )
            raise RuntimeError(f"live budget invalid; terminated {pod.id}: {error}") from error
    if now >= state["expires_at"]:
        runtime._terminate_all_state_candidates(state, reason="status-expiry-enforcement")
        pod = None
        state = c._read_state()
    live_rate = pod.cost_per_hr if pod else state["usd_hr"]
    out = {
        **state,
        "elapsed_spend_estimate_usd": round(runtime._elapsed_spend(state, now, live_rate), 4),
        "live_usd_hr": live_rate,
        "remaining_seconds": max(0, int(state["expires_at"] - now)),
        "status": pod.status
        if pod
        else (state["phase"].upper() if state["phase"] != "open" else "MISSING"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if pod and pod.status == "RUNNING" else 1


def cmd_close(args: argparse.Namespace) -> int:
    state = c._read_state()
    if (
        state.get("phase") in {"open", "creating", "bootstrapping", "cleanup_failed"}
        and time.time() >= state.get("expires_at", math.inf)
    ):
        runtime._terminate_all_state_candidates(
            state, reason="close-command-expiry-enforcement"
        )
        state = c._read_state()
    pod_id = state.get("pod_id")
    token = c._token(
        "CLOSE",
        {
            "lease_name": state.get("lease_name"),
            "pod_id": pod_id,
            "volume_id": state["volume_id"],
        },
    )
    if args.confirm != token:
        print(f"NO CHANGE. Re-run with: --confirm {token}")
        return 2

    by_id = {}
    for candidate_id in filter(None, [pod_id, *state.get("orphan_ids", [])]):
        candidate = c.api.get_pod(candidate_id)
        if candidate:
            by_id[candidate.id] = candidate
    for candidate in c.api.list_pods():
        if candidate.name == state.get("lease_name"):
            by_id[candidate.id] = candidate
    pods = list(by_id.values())
    if state.get("phase") == "open" and len(pods) != 1:
        raise RuntimeError(
            f"open lease resolved to {len(pods)} pods; refusing unsealed termination"
        )
    pod = pods[0] if len(pods) == 1 else None
    seal_ok = state.get("phase") != "open"
    if (state.get("phase") == "open" and pod and pod.status == "RUNNING"
            and not pod.ssh_host):
        raise RuntimeError("running lease has no SSH endpoint; refusing unsealed termination")
    if state.get("phase") == "open" and pod and pod.status == "RUNNING" and pod.ssh_host:
        runtime._persist_for_termination(state, pod, reason="explicit-close")
        seal_dir = c.STATE.parent / "seals"
        seal_dir.mkdir(parents=True, exist_ok=True)
        (seal_dir / f"{pod.id}.json").write_text(
            json.dumps(state["last_persistence"], indent=2, sort_keys=True) + "\n"
        )
        seal_ok = True

    failures = []
    for candidate in pods:
        try:
            provider._terminate_pod_once(candidate.id)
            c.ledger.append(
                "terminate", pod_id=candidate.id, gpu=candidate.gpu, note="explicit-close"
            )
        except Exception as error:
            failures.append(f"{candidate.id}: {error}")
    if failures:
        state["phase"] = "cleanup_failed"
        state["orphan_ids"] = [candidate.id for candidate in pods]
        c._write_state(state)
        raise RuntimeError("; ".join(failures))
    runtime._cancel_watchdog(state)
    c.STATE.unlink(missing_ok=True)
    ids = ",".join(candidate.id for candidate in pods) or str(pod_id or "none")
    print(f"CLOSED {ids}; compute terminated; network volume retained")
    return 0 if seal_ok else 1


def _watch_stop_reason(state: dict) -> str | None:
    """Return a provider-termination reason from independent remote evidence."""
    if state.get("phase") != "open" or not state.get("remote_guard_installed_at"):
        return None
    pod = c.api.get_pod(state["pod_id"])
    if not pod or pod.status != "RUNNING":
        return "detached-local-watchdog-remote-exit"
    idle_seconds = int(state["idle_minutes"]) * 60
    if runtime._remote_is_idle(c.Endpoint.of(pod), idle_seconds):
        return "detached-local-watchdog-idle"
    return None


def _same_watch_lease(state: dict, args: argparse.Namespace) -> bool:
    return (
        state.get("phase") != "terminated"
        and state.get("lease_name") == args.lease_name
        and state.get("created_at") == args.created_at
        and state.get("expires_at") == args.expires_at
    )


def cmd_watch(args: argparse.Namespace) -> int:
    """Detached provider watchdog; remote TTL/idle processes are independent."""
    remote_failures = 0
    while True:
        remaining = args.expires_at - time.time()
        if remaining > 0:
            time.sleep(min(remaining, 60))
        with c._lease_lock():
            if not c.STATE.exists():
                return 0
            state = c._read_state()
            if not _same_watch_lease(state, args):
                return 0
            if time.time() >= args.expires_at:
                try:
                    runtime._terminate_all_state_candidates(
                        state, reason="detached-local-watchdog-expiry"
                    )
                except Exception as error:
                    state["phase"] = "cleanup_failed"
                    state["termination_error"] = str(error)
                    state["termination_reason"] = "detached-local-watchdog-expiry"
                    c._write_state(state)
                    raise
                return 0
            try:
                reason = _watch_stop_reason(state)
            except (c.api.ApiError, RuntimeError, ValueError, OSError):
                remote_failures += 1
                if remote_failures < REMOTE_OBSERVATION_FAILURE_LIMIT:
                    continue
                reason = "detached-local-watchdog-remote-unobservable"
            else:
                remote_failures = 0
            if reason:
                runtime._terminate_all_state_candidates(
                    state, reason=reason
                )
                return 0
