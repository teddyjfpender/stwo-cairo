"""One-shot RunPod provider reads and mutations."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request

from . import common as c
from . import lease_local_root


REST_BASE = "https://rest.runpod.io/v1"


def _rest_get(resource: str) -> dict:
    """Perform one authenticated, fail-closed RunPod REST read."""
    request = urllib.request.Request(
        f"{REST_BASE}/{resource.lstrip('/')}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {c.api._api_key()}",
            "User-Agent": "stwo-gpu-lab/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise c.api.ApiError(f"RunPod REST read failed for {resource}: {error}") from error
    if not isinstance(payload, dict):
        raise c.api.ApiError(f"RunPod REST returned a non-object for {resource}")
    return payload


def _network_volume_attestation(volume_id: str, data_center_id: str) -> dict:
    """Attest the requested persistent volume before any create mutation."""
    encoded = urllib.parse.quote(volume_id, safe="")
    volume = _rest_get(f"networkvolumes/{encoded}")
    size = volume.get("size")
    if (
        volume.get("id") != volume_id
        or volume.get("dataCenterId") != data_center_id
        or not isinstance(volume.get("name"), str)
        or not volume["name"]
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise RuntimeError(
            "network-volume REST attestation mismatch: "
            f"requested id={volume_id!r} dc={data_center_id!r}, returned={volume!r}"
        )
    return {
        "data_center_id": data_center_id,
        "id": volume_id,
        "name": volume["name"],
        "size_gb": size,
        "source": "RunPod REST /v1/networkvolumes/{id}",
    }


def _attest_pod_volume(pod_id: str, volume_id: str, data_center_id: str) -> dict:
    """Prove the created pod reports the exact requested network volume."""
    encoded = urllib.parse.quote(pod_id, safe="")
    pod = _rest_get(f"pods/{encoded}")
    volume = pod.get("networkVolume")
    if pod.get("id") != pod_id:
        raise RuntimeError(
            "created pod network-volume attestation mismatch: "
            f"pod={pod_id!r} volume={volume_id!r} dc={data_center_id!r}, returned={pod!r}"
        )
    if isinstance(volume, dict):
        matched = (
            volume.get("id") == volume_id
            and volume.get("dataCenterId") == data_center_id
        )
        source = "RunPod REST /v1/pods/{id}.networkVolume"
    else:
        # The current REST schema returns only networkVolumeId on the Pod. Bind
        # that exact id here and independently re-attest its data-center record.
        matched = pod.get("networkVolumeId") == volume_id
        if matched:
            _network_volume_attestation(volume_id, data_center_id)
        source = (
            "RunPod REST /v1/pods/{id}.networkVolumeId + "
            "/v1/networkvolumes/{id}.dataCenterId"
        )
    if not matched:
        raise RuntimeError(
            "created pod network-volume attestation mismatch: "
            f"pod={pod_id!r} volume={volume_id!r} dc={data_center_id!r}, returned={pod!r}"
        )
    return {
        "data_center_id": data_center_id,
        "id": volume_id,
        "pod_id": pod_id,
        "source": source,
    }


def _secure_offer(gpu_id: str) -> dict:
    """Return the explicit Secure Cloud list price; generic lowestPrice is unsafe."""
    data = c.api.gql(
        """
        query($id: String!) {
          gpuTypes(input: {id: $id}) {
            id displayName secureCloud securePrice
          }
        }
        """,
        {"id": gpu_id},
    )
    matches = [t for t in data.get("gpuTypes") or [] if t.get("id") == gpu_id]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve one exact Secure Cloud offer for {gpu_id}")
    offer = matches[0]
    if not offer.get("secureCloud"):
        raise RuntimeError(f"{gpu_id} has no declared Secure Cloud capacity")
    try:
        price = float(offer["securePrice"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Secure Cloud price unavailable for {gpu_id}") from error
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(f"invalid Secure Cloud price for {gpu_id}: {price}")
    return {
        "display_name": offer.get("displayName") or gpu_id,
        "gpu_type_id": gpu_id,
        "usd_hr": price,
    }


def _create_pod_once(
    *, name: str, gpu_id: str, volume_mount: str, args
) -> c.api.PodInfo:
    """Create exactly once: lifecycle mutations never inherit gql retries."""
    if volume_mount not in (c.VOLUME_MOUNT, lease_local_root.PROVIDER_MOUNT):
        raise RuntimeError(f"unsupported provider volume mount: {volume_mount!r}")
    query = """
    mutation($in: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $in) { %s }
    }
    """ % c.api.POD_FIELDS
    data = c.api.gql(
        query,
        {
            "in": {
                "name": name,
                "gpuTypeId": gpu_id,
                "imageName": args.image,
                "cloudType": "SECURE",
                "gpuCount": 1,
                "containerDiskInGb": c.DEFAULT_DISK_GB,
                "volumeInGb": 0,
                "volumeMountPath": volume_mount,
                "minVcpuCount": args.min_vcpu,
                "minMemoryInGb": args.min_mem_gb,
                "networkVolumeId": args.volume_id,
                "ports": "22/tcp",
                "startSsh": True,
                "env": [],
            }
        },
        retries=1,
    )
    raw = data.get("podFindAndDeployOnDemand")
    if not raw:
        raise c.api.ApiError(f"no capacity for {gpu_id} (SECURE)")
    return c.api.PodInfo.from_raw(raw)


def _terminate_pod_once(pod_id: str) -> None:
    """Issue one terminate mutation and prove the pod disappeared."""
    mutation_error = None
    try:
        c.api.gql(
            "mutation($id: String!) { podTerminate(input: {podId: $id}) }",
            {"id": pod_id},
            retries=1,
        )
    except Exception as error:
        mutation_error = error

    try:
        exact = c.api.get_pod(pod_id)
        account_ids = {candidate.id for candidate in c.api.list_pods()}
    except Exception as reconcile_error:
        detail = f" after mutation error {mutation_error}" if mutation_error else ""
        raise RuntimeError(
            f"termination ambiguous for {pod_id}{detail}; "
            f"reconciliation failed: {reconcile_error}"
        ) from (mutation_error or reconcile_error)

    if exact is not None or pod_id in account_ids:
        detail = f"; mutation error: {mutation_error}" if mutation_error else ""
        raise RuntimeError(
            f"termination not confirmed for still-present pod {pod_id}{detail}"
        ) from mutation_error
    if mutation_error is not None:
        try:
            if c.api.get_pod(pod_id) is not None:
                raise RuntimeError(f"termination became present again for pod {pod_id}")
        except RuntimeError:
            raise
        except Exception as reconcile_error:
            raise RuntimeError(
                f"termination second reconciliation failed for {pod_id}: "
                f"{reconcile_error}"
            ) from mutation_error
