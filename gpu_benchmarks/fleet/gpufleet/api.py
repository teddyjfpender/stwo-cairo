"""Typed RunPod GraphQL client. Stdlib only; the API key is never logged.

Every call returns parsed data or raises ApiError with the server's message —
schema drift surfaces as a readable error, not a KeyError three layers up.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

GRAPHQL_URL = "https://api.runpod.io/graphql"


class ApiError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    cfg = pathlib.Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        data = tomllib.loads(cfg.read_text())
        key = str(data.get("apikey", "")).strip()
        if key:
            return key
    raise ApiError(
        "no RunPod API key: set RUNPOD_API_KEY or run `runpodctl config --apiKey ...`"
    )


def gql(query: str, variables: dict | None = None, retries: int = 3) -> dict:
    """POST one GraphQL operation; retry transient transport failures."""
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            GRAPHQL_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                # Cloudflare 403s Python's default UA; any explicit one passes.
                "User-Agent": "gpufleet/0.1",
                # Bearer auth keeps the key out of URLs.
                "Authorization": f"Bearer {_api_key()}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("errors"):
                msgs = "; ".join(e.get("message", "?") for e in payload["errors"])
                raise ApiError(f"GraphQL error: {msgs}")
            return payload["data"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2**attempt)
    raise ApiError(f"RunPod API unreachable after {retries} attempts: {last}")


# ------------------------------- pod model ------------------------------------------

POD_FIELDS = """
    id name desiredStatus costPerHr gpuCount vcpuCount memoryInGb
    containerDiskInGb volumeInGb machineId
    machine { gpuDisplayName dataCenterId }
    runtime {
      uptimeInSeconds
      ports { ip isIpPublic privatePort publicPort type }
      gpus { id gpuUtilPercent memoryUtilPercent }
    }
"""


@dataclass
class PodInfo:
    id: str
    name: str
    status: str
    cost_per_hr: float
    gpu: str
    dc: str
    vcpu: int
    mem_gb: int
    ssh_host: str | None = None
    ssh_port: int | None = None
    uptime_s: int = 0
    gpu_util: int | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, p: dict) -> "PodInfo":
        machine = p.get("machine") or {}
        runtime = p.get("runtime") or {}
        host = port = None
        for prt in runtime.get("ports") or []:
            if prt.get("privatePort") == 22 and prt.get("isIpPublic"):
                host, port = prt.get("ip"), prt.get("publicPort")
        gpus = runtime.get("gpus") or []
        return cls(
            id=p["id"],
            name=p.get("name") or "",
            status=p.get("desiredStatus") or "?",
            cost_per_hr=float(p.get("costPerHr") or 0.0),
            gpu=machine.get("gpuDisplayName") or "?",
            dc=machine.get("dataCenterId") or "?",
            vcpu=int(p.get("vcpuCount") or 0),
            mem_gb=int(p.get("memoryInGb") or 0),
            ssh_host=host,
            ssh_port=port,
            uptime_s=int(runtime.get("uptimeInSeconds") or 0),
            gpu_util=(int(gpus[0]["gpuUtilPercent"]) if gpus else None),
            raw=p,
        )


def list_pods() -> list[PodInfo]:
    data = gql("query { myself { pods { " + POD_FIELDS + " } } }")
    return [PodInfo.from_raw(p) for p in (data.get("myself") or {}).get("pods") or []]


def get_pod(pod_id: str) -> PodInfo | None:
    data = gql(
        "query($id: String!) { pod(input: {podId: $id}) { " + POD_FIELDS + " } }",
        {"id": pod_id},
    )
    p = data.get("pod")
    return PodInfo.from_raw(p) if p else None


# ------------------------------- offers ---------------------------------------------


def gpu_offers(gpu_type_ids: list[str]) -> list[dict]:
    """Price + capacity per GPU type (secure + community, on-demand + spot)."""
    out = []
    for tid in gpu_type_ids:
        data = gql(
            """
            query($id: String!) {
              gpuTypes(input: {id: $id}) {
                id displayName memoryInGb secureCloud communityCloud
                securePrice communityPrice
                lowestPrice(input: {gpuCount: 1}) {
                  minimumBidPrice uninterruptablePrice
                  stockStatus compliance
                }
              }
            }
            """,
            {"id": tid},
        )
        for t in data.get("gpuTypes") or []:
            lp = t.get("lowestPrice") or {}
            out.append(
                {
                    "id": t["id"],
                    "name": t.get("displayName") or t["id"],
                    "vram_gb": t.get("memoryInGb"),
                    "secure": bool(t.get("secureCloud")),
                    "community": bool(t.get("communityCloud")),
                    "od_usd_hr": lp.get("uninterruptablePrice"),
                    "spot_usd_hr": lp.get("minimumBidPrice"),
                    "stock": lp.get("stockStatus"),
                }
            )
    return out


def secure_offer(gpu_type_id: str) -> dict:
    """Return one exact Secure Cloud list price for a GPU type."""
    data = gql(
        """
        query($id: String!) {
          gpuTypes(input: {id: $id}) {
            id displayName secureCloud securePrice
          }
        }
        """,
        {"id": gpu_type_id},
    )
    matches = [
        offer
        for offer in data.get("gpuTypes") or []
        if offer.get("id") == gpu_type_id
    ]
    if len(matches) != 1:
        raise ApiError(f"cannot resolve one exact Secure Cloud offer for {gpu_type_id}")
    offer = matches[0]
    if not offer.get("secureCloud"):
        raise ApiError(f"{gpu_type_id} has no declared Secure Cloud capacity")
    try:
        price = float(offer["securePrice"])
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError(f"Secure Cloud price unavailable for {gpu_type_id}") from error
    if not math.isfinite(price) or price <= 0:
        raise ApiError(f"invalid Secure Cloud price for {gpu_type_id}: {price}")
    return {
        "display_name": offer.get("displayName") or gpu_type_id,
        "gpu_type_id": gpu_type_id,
        "usd_hr": price,
    }


# ------------------------------- lifecycle ------------------------------------------


def create_pod(
    *,
    name: str,
    gpu_type_id: str,
    image: str,
    cloud: str = "SECURE",
    disk_gb: int = 80,
    volume_gb: int = 120,
    volume_mount: str = "/workspace",
    min_vcpu: int = 16,
    min_mem_gb: int = 62,
    network_volume_id: str | None = None,
    env: dict[str, str] | None = None,
) -> PodInfo:
    q = """
    mutation($in: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $in) { %s }
    }
    """ % POD_FIELDS
    inp: dict[str, Any] = {
        "name": name,
        "gpuTypeId": gpu_type_id,
        "imageName": image,
        "cloudType": cloud,
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0 if network_volume_id else volume_gb,
        "volumeMountPath": volume_mount,
        "minVcpuCount": min_vcpu,
        "minMemoryInGb": min_mem_gb,
        "ports": "22/tcp",
        "startSsh": True,
        "env": [{"key": k, "value": v} for k, v in (env or {}).items()],
    }
    if network_volume_id:
        inp["networkVolumeId"] = network_volume_id
    # A transport retry can create a second billable pod after an ambiguous
    # response. Creation is issued once; the caller reconciles by unique name.
    data = gql(q, {"in": inp}, retries=1)
    p = data.get("podFindAndDeployOnDemand")
    if not p:
        raise ApiError(f"no capacity for {gpu_type_id} ({cloud})")
    return PodInfo.from_raw(p)


def _observe_pod(pod_id: str) -> tuple[PodInfo | None, PodInfo | None]:
    exact = get_pod(pod_id)
    listed = next((pod for pod in list_pods() if pod.id == pod_id), None)
    return exact, listed


def _wait_lifecycle(
    pod_id: str,
    *,
    action: str,
    mutation_error: Exception | None,
    timeout_s: float,
    poll_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    clean_observations = 0
    last = "not observed"
    while True:
        try:
            exact, listed = _observe_pod(pod_id)
            observed = [pod for pod in (exact, listed) if pod is not None]
            if action == "terminate":
                done = not observed
                terminal = "ABSENT"
            else:
                done = not observed or all(pod.status == "EXITED" for pod in observed)
                terminal = "ABSENT" if not observed else "EXITED"
            clean_observations = clean_observations + 1 if done else 0
            last = ", ".join(pod.status for pod in observed) or "absent"
            # If the mutation response was ambiguous, require two independent
            # reconciliations so a single stale read cannot declare success.
            if clean_observations >= (2 if mutation_error else 1):
                return terminal
        except Exception as error:
            clean_observations = 0
            last = f"observation failed: {error}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f"; mutation error: {mutation_error}" if mutation_error else ""
            raise RuntimeError(
                f"{action} not confirmed for pod {pod_id} ({last}){detail}"
            ) from mutation_error
        time.sleep(min(poll_s, remaining))


def stop_pod(
    pod_id: str, *, timeout_s: float = 120, poll_s: float = 2
) -> str:
    mutation_error = None
    try:
        data = gql(
            "mutation($id: String!) { podStop(input: {podId: $id}) "
            "{ id desiredStatus } }",
            {"id": pod_id},
            retries=1,
        )
        result = data.get("podStop") or {}
        if result.get("id") != pod_id:
            raise ApiError(f"stop returned the wrong pod: {result!r}")
    except Exception as error:
        mutation_error = error
    return _wait_lifecycle(
        pod_id,
        action="stop",
        mutation_error=mutation_error,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )


def resume_pod(pod_id: str) -> str:
    data = gql(
        "mutation($id: String!) { podResume(input: {podId: $id, gpuCount: 1}) "
        "{ id desiredStatus } }",
        {"id": pod_id},
        retries=1,
    )
    return (data.get("podResume") or {}).get("desiredStatus", "?")


def terminate_pod(
    pod_id: str, *, timeout_s: float = 120, poll_s: float = 2
) -> None:
    mutation_error = None
    try:
        gql(
            "mutation($id: String!) { podTerminate(input: {podId: $id}) }",
            {"id": pod_id},
            retries=1,
        )
    except Exception as error:
        mutation_error = error
    _wait_lifecycle(
        pod_id,
        action="terminate",
        mutation_error=mutation_error,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
