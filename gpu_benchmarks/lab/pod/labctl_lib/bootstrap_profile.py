"""Pinned, development-only RTX 4090 bootstrap profile."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess

from . import common as c
from . import lease_local_root


NAME = "consumer-4090-bootstrap"
IMAGE = (
    "docker.io/runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404"
    "@sha256:82d75acd177789c2d84232be35f003a7e959b75c35da57ab35ab6ed68e9cad6a"
)
IMAGE_DIGEST = IMAGE.rsplit("@", 1)[1]
LANE = "consumer-development"
GPU = "4090"
VOLUME_ID = "2kpphx92fr"
VOLUME_DC = "EU-RO-1"
MAX_TTL_HOURS = 6.0
MAX_IDLE_MINUTES = 30
MAX_USD_HR = 0.80
MAX_TOTAL_USD = 4.80
MIN_VCPU = 8
MIN_MEM_GB = 32

_FORMAL_DEFAULTS = {
    "gpu": "4090",
    "image": None,
    "volume_id": None,
    "volume_dc": None,
    "name": None,
    "ttl_hours": 4.0,
    "idle_min": 30,
    "max_usd_hr": 1.0,
    "max_total_usd": 4.0,
    "min_vcpu": 8,
    "min_mem_gb": 32,
    "ready_timeout": 600.0,
}
_PROFILE_DEFAULTS = {
    **_FORMAL_DEFAULTS,
    "gpu": GPU,
    "image": IMAGE,
    "volume_id": VOLUME_ID,
    "volume_dc": VOLUME_DC,
    "name": NAME,
    "ttl_hours": MAX_TTL_HOURS,
    "idle_min": MAX_IDLE_MINUTES,
    "max_usd_hr": MAX_USD_HR,
    "max_total_usd": MAX_TOTAL_USD,
}


def configure(args: argparse.Namespace, environ: dict[str, str]) -> None:
    """Fill omitted arguments without changing the ordinary profile defaults."""
    if getattr(args, "bootstrap_profile", None) is None:
        defaults = {
            **_FORMAL_DEFAULTS,
            "image": environ.get("LABCTL_IMAGE"),
            "volume_id": environ.get("LABCTL_VOLUME_ID"),
            "volume_dc": environ.get("LABCTL_VOLUME_DC"),
        }
        for field, default in defaults.items():
            if not hasattr(args, field):
                setattr(args, field, default)
        return
    if args.bootstrap_profile != NAME:
        raise RuntimeError(f"unsupported lab profile: {args.bootstrap_profile!r}")

    exact = {
        "gpu": GPU,
        "image": IMAGE,
        "volume_id": VOLUME_ID,
        "volume_dc": VOLUME_DC,
        "name": NAME,
    }
    for field, expected in exact.items():
        if hasattr(args, field) and getattr(args, field) != expected:
            raise RuntimeError(
                f"{NAME} requires --{field.replace('_', '-')} {expected!r}"
            )
    for field, default in _PROFILE_DEFAULTS.items():
        if not hasattr(args, field):
            setattr(args, field, default)
    ceilings = {
        "ttl_hours": MAX_TTL_HOURS,
        "idle_min": MAX_IDLE_MINUTES,
        "max_usd_hr": MAX_USD_HR,
        "max_total_usd": MAX_TOTAL_USD,
    }
    for field, ceiling in ceilings.items():
        if getattr(args, field) > ceiling:
            raise RuntimeError(
                f"{NAME} requires --{field.replace('_', '-')} <= {ceiling:g}"
            )
    if args.min_vcpu < MIN_VCPU or args.min_mem_gb < MIN_MEM_GB:
        raise RuntimeError(
            f"{NAME} requires at least {MIN_VCPU} vCPU and {MIN_MEM_GB} GiB RAM"
        )


def metadata(args: argparse.Namespace) -> dict:
    if getattr(args, "bootstrap_profile", None) != NAME:
        return {}
    return {
        "formal": False,
        "image_digest": IMAGE_DIGEST,
        "image_digest_authority": "requested-reference-not-runtime-attested",
        "lane": LANE,
        "persistence_scope": lease_local_root.PERSISTENCE_SCOPE,
        "profile": NAME,
        "qualification": False,
        "qualification_eligible": False,
    }


def provider_volume_mount(args: argparse.Namespace) -> str:
    """Keep the formal mount unchanged and quarantine only the pinned dev volume."""
    if getattr(args, "bootstrap_profile", None) == NAME:
        return lease_local_root.PROVIDER_MOUNT
    return c.VOLUME_MOUNT


def is_lease_local_state(state: dict) -> bool:
    """Recognize only the complete controller-created nonformal lease contract."""
    plan = state.get("plan")
    root = state.get("lease_local_root")
    return (
        isinstance(plan, dict)
        and isinstance(root, dict)
        and state.get("profile") == NAME
        and state.get("lane") == LANE
        and state.get("formal") is False
        and state.get("qualification") is False
        and state.get("qualification_eligible") is False
        and state.get("persistence_scope") == lease_local_root.PERSISTENCE_SCOPE
        and state.get("image") == IMAGE
        and state.get("volume_id") == VOLUME_ID
        and state.get("volume_dc") == VOLUME_DC
        and plan.get("profile") == NAME
        and plan.get("lane") == LANE
        and plan.get("formal") is False
        and plan.get("persistence_scope") == lease_local_root.PERSISTENCE_SCOPE
        and plan.get("qualification") is False
        and plan.get("qualification_eligible") is False
        and plan.get("image") == IMAGE
        and plan.get("volume_id") == VOLUME_ID
        and plan.get("volume_dc") == VOLUME_DC
        and plan.get("volume_mount") == lease_local_root.PROVIDER_MOUNT
        and root.get("schema_version") == lease_local_root.SCHEMA
        and isinstance(root.get("boot_id"), str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            root["boot_id"],
        )
        is not None
        and root.get("persistence_scope") == lease_local_root.PERSISTENCE_SCOPE
        and root.get("qualification") is False
        and root.get("pod_id") == state.get("pod_id")
        and root.get("volume_id") == VOLUME_ID
        and root.get("provider_mount") == lease_local_root.PROVIDER_MOUNT
        and root.get("quarantined_root") == lease_local_root.QUARANTINED_ROOT
        and root.get("target") == lease_local_root.TARGET
        and root.get("marker")
        == f"{lease_local_root.TARGET}/{lease_local_root.MARKER}"
    )


def _public_key() -> str:
    options = tuple(c.SSH_OPTS)
    identities = [
        options[index + 1]
        for index, value in enumerate(options[:-1])
        if value == "-i"
    ]
    if len(identities) != 1:
        raise RuntimeError("root SSH transport did not bind one exact identity")
    result = subprocess.run(
        ["ssh-keygen", "-y", "-f", identities[0]],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("could not derive the authenticated root SSH public key")
    fields = result.stdout.strip().split()
    key_type = r"(?:ssh-(?:ed25519|rsa)|ecdsa-sha2-nistp(?:256|384|521))"
    if (
        len(fields) < 2
        or not re.fullmatch(key_type, fields[0])
        or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", fields[1])
    ):
        raise RuntimeError("authenticated root identity emitted an invalid public key")
    return " ".join(fields[:2])


def _bootstrap_command(public_key: str) -> str:
    policy = base64.b64encode(
        b"Match User dev\n"
        b"    AuthenticationMethods publickey\n"
        b"    PubkeyAuthentication yes\n"
        b"    PasswordAuthentication no\n"
        b"    KbdInteractiveAuthentication no\n"
    ).decode()
    return f"""
set -euE
trap 'rc=$?; trap - ERR; printf "LABCTL_BOOTSTRAP_ERROR_LINE=%s RC=%s\\n" "$LINENO" "$rc" >&2; exit "$rc"' ERR
test "$(id -u):$(id -un)" = 0:root
KEY={shlex.quote(public_key)}
test ! -L /home
test -d /home
test "$(stat -c '%u:%g:%a' /home)" = 0:0:755
command -v groupmod >/dev/null
command -v usermod >/dev/null
command -v pgrep >/dev/null

# The immutable image has one local ubuntu identity at 1000:1000.  Admit only
# that exact identity and the operation boundaries reachable below.
UID1000=$(awk -F: '$3 == 1000 {{printf "%s:%s:%s:%s\\n", $1, $4, $6, $7}}' /etc/passwd)
GID1000=$(awk -F: '$3 == 1000 {{printf "%s:%s\\n", $1, $4}}' /etc/group)
PRIMARY1000=$(awk -F: '$4 == 1000 {{printf "%s:%s\\n", $1, $3}}' /etc/passwd)
case "$UID1000|$GID1000|$PRIMARY1000" in
  'ubuntu:1000:/home/ubuntu:/bin/bash|ubuntu:|ubuntu:1000')
    ACCOUNT_STATE=initial; SOURCE_USER=ubuntu; SOURCE_HOME=/home/ubuntu ;;
  'ubuntu:1000:/home/ubuntu:/bin/bash|dev:|ubuntu:1000')
    ACCOUNT_STATE=group-renamed; SOURCE_USER=ubuntu; SOURCE_HOME=/home/ubuntu ;;
  'dev:1000:/home/ubuntu:/bin/bash|dev:|dev:1000')
    ACCOUNT_STATE=user-renamed; SOURCE_USER=dev; SOURCE_HOME=/home/ubuntu ;;
  'dev:1000:/home/dev:/bin/bash|dev:|dev:1000')
    ACCOUNT_STATE=desired; SOURCE_USER=dev; SOURCE_HOME=/home/dev ;;
  *)
    printf 'LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-identity user=%s group=%s primary=%s\n' \
      "$UID1000" "$GID1000" "$PRIMARY1000" >&2
    exit 1 ;;
esac

# The seed's exact supplementary set is part of the pinned-image contract.
# Clearing it is the first mutation; every subsequent state has primary GID only.
GROUPS1000=$(id -G "$SOURCE_USER")
case "$ACCOUNT_STATE|$GROUPS1000" in
  'initial|1000 4 20 24 25 27 29 30 44 46') ;;
  'initial|1000') ACCOUNT_STATE=groups-cleared ;;
  'group-renamed|1000'|'user-renamed|1000'|'desired|1000') ;;
  *)
    printf 'LABCTL_BOOTSTRAP_CONFLICT=unexpected-1000-groups state=%s groups=%s\n' \
      "$ACCOUNT_STATE" "$GROUPS1000" >&2
    exit 1 ;;
esac

# Name aliases, an unlocked password, a live process, or an unsafe home all fail
# before the first account mutation.
case "$ACCOUNT_STATE" in
  initial|groups-cleared)
    if getent passwd dev >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=passwd name=dev\n' >&2
      exit 1
    fi
    if getent group dev >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=group name=dev\n' >&2
      exit 1
    fi ;;
  group-renamed)
    if getent passwd dev >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=passwd name=dev\n' >&2
      exit 1
    fi
    if getent group ubuntu >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=group name=ubuntu\n' >&2
      exit 1
    fi ;;
  user-renamed|desired)
    if getent passwd ubuntu >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=passwd name=ubuntu\n' >&2
      exit 1
    fi
    if getent group ubuntu >/dev/null; then
      printf 'LABCTL_BOOTSTRAP_CONFLICT=name-alias database=group name=ubuntu\n' >&2
      exit 1
    fi ;;
esac
test "$(passwd -S "$SOURCE_USER" | awk '{{print $2}}')" = L
test ! -L "$SOURCE_HOME"
test -d "$SOURCE_HOME"
test "$(stat -c '%u:%g' "$SOURCE_HOME")" = 1000:1000
case "$(stat -c '%a' "$SOURCE_HOME")" in 700|750|755) ;; *) exit 1 ;; esac
for path in "$SOURCE_HOME/.ssh" "$SOURCE_HOME/.ssh/authorized_keys" \
  "$SOURCE_HOME/.ssh/.authorized_keys.new"; do
  test ! -L "$path"
done
if test -e "$SOURCE_HOME/.ssh"; then
  test -d "$SOURCE_HOME/.ssh"
  test "$(stat -c '%u:%g:%a' "$SOURCE_HOME/.ssh")" = 1000:1000:700
fi
test ! -e "$SOURCE_HOME/.ssh/.authorized_keys.new"
if test -e "$SOURCE_HOME/.ssh/authorized_keys"; then
  test -f "$SOURCE_HOME/.ssh/authorized_keys"
  test "$(stat -c '%u:%g:%a:%h' "$SOURCE_HOME/.ssh/authorized_keys")" = 1000:1000:600:1
  test "$(cat "$SOURCE_HOME/.ssh/authorized_keys")" = "$KEY"
  test "$(wc -l < "$SOURCE_HOME/.ssh/authorized_keys")" = 1
fi
if test "$ACCOUNT_STATE" = desired; then
  test ! -e /home/ubuntu
  test ! -L /home/ubuntu
else
  if pgrep -u 1000 >/dev/null; then
    printf 'LABCTL_BOOTSTRAP_CONFLICT=uid1000-process\n' >&2
    exit 1
  fi
  test ! -e /home/dev
  test ! -L /home/dev
fi

case "$ACCOUNT_STATE" in
  initial)
    usermod --groups '' ubuntu
    groupmod --new-name dev ubuntu
    usermod --login dev ubuntu
    usermod --home /home/dev --move-home dev ;;
  groups-cleared)
    groupmod --new-name dev ubuntu
    usermod --login dev ubuntu
    usermod --home /home/dev --move-home dev ;;
  group-renamed)
    usermod --login dev ubuntu
    usermod --home /home/dev --move-home dev ;;
  user-renamed)
    usermod --home /home/dev --move-home dev ;;
  desired) ;;
esac
test "$(getent passwd 1000 | cut -d: -f1)" = dev
test "$(getent passwd dev | cut -d: -f3-4)" = 1000:1000
test "$(getent passwd dev | cut -d: -f6-7)" = /home/dev:/bin/bash
test "$(getent group 1000 | cut -d: -f1,4)" = dev:
test "$(id -g dev)" = 1000
test "$(id -G dev)" = 1000
test "$(id -gn dev)" = dev
test "$(id -Gn dev)" = dev
test ! -e /home/ubuntu
test ! -L /home/ubuntu
for path in /home/dev /home/dev/.ssh /home/dev/.ssh/authorized_keys \
  /home/dev/.ssh/.authorized_keys.new; do
  test ! -L "$path"
done
test -d /home/dev
test "$(stat -c '%u:%g' /home/dev)" = 1000:1000
chmod 0700 /home/dev
if test -e /home/dev/.ssh; then
  test -d /home/dev/.ssh
  test "$(stat -c '%u:%g:%a' /home/dev/.ssh)" = 1000:1000:700
else
  install -d -m 0700 -o dev -g 1000 /home/dev/.ssh
fi
if test -e /home/dev/.ssh/authorized_keys; then
  test -f /home/dev/.ssh/authorized_keys
  test "$(stat -c '%u:%g:%a:%h' /home/dev/.ssh/authorized_keys)" = 1000:1000:600:1
  test "$(cat /home/dev/.ssh/authorized_keys)" = "$KEY"
  test "$(wc -l < /home/dev/.ssh/authorized_keys)" = 1
else
  test ! -e /home/dev/.ssh/.authorized_keys.new
  umask 077
  set -C
  printf '%s\n' "$KEY" > /home/dev/.ssh/.authorized_keys.new
  set +C
  chown 1000:1000 /home/dev/.ssh/.authorized_keys.new
  chmod 0600 /home/dev/.ssh/.authorized_keys.new
  mv -T /home/dev/.ssh/.authorized_keys.new /home/dev/.ssh/authorized_keys
fi
test ! -L /home/dev/.ssh/authorized_keys
test "$(stat -c '%u:%g:%a:%h' /home/dev/.ssh/authorized_keys)" = 1000:1000:600:1
test "$(passwd -S dev | cut -d' ' -f2)" = L
command -v sshd >/dev/null
for path in /etc/ssh /etc/ssh/sshd_config.d \
  /etc/ssh/sshd_config.d/99-stwo-consumer-bootstrap.conf; do
  test ! -L "$path"
done
test -d /etc/ssh
test "$(stat -c '%u:%g:%a' /etc/ssh)" = 0:0:755
if test -e /etc/ssh/sshd_config.d; then
  test -d /etc/ssh/sshd_config.d
  test "$(stat -c '%u:%g:%a' /etc/ssh/sshd_config.d)" = 0:0:755
else
  install -d -m 0755 -o root -g root /etc/ssh/sshd_config.d
fi
SSHD_PROFILE=/etc/ssh/sshd_config.d/99-stwo-consumer-bootstrap.conf
SSHD_POLICY={shlex.quote(policy)}
if test -e "$SSHD_PROFILE"; then
  test -f "$SSHD_PROFILE"
  test "$(stat -c '%u:%g:%a:%h' "$SSHD_PROFILE")" = 0:0:644:1
  test "$(base64 -w0 "$SSHD_PROFILE")" = "$SSHD_POLICY"
else
  TMP=$(mktemp /etc/ssh/sshd_config.d/.stwo-bootstrap.XXXXXX)
  trap 'rm -f -- "$TMP"' EXIT
  printf '%s' "$SSHD_POLICY" | base64 --decode > "$TMP"
  chown 0:0 "$TMP"
  chmod 0644 "$TMP"
  mv -T "$TMP" "$SSHD_PROFILE"
  trap - EXIT
fi
test ! -L "$SSHD_PROFILE"
test "$(stat -c '%u:%g:%a:%h' "$SSHD_PROFILE")" = 0:0:644:1
sshd -t
POLICY=$(sshd -T -C user=dev,host=localhost,addr=127.0.0.1)
test "$(printf '%s\n' "$POLICY" | awk '$1=="authenticationmethods" {{print $2}}')" = publickey
test "$(printf '%s\n' "$POLICY" | awk '$1=="pubkeyauthentication" {{print $2}}')" = yes
test "$(printf '%s\n' "$POLICY" | awk '$1=="passwordauthentication" {{print $2}}')" = no
test "$(printf '%s\n' "$POLICY" | awk '$1=="kbdinteractiveauthentication" {{print $2}}')" = no
SSHD_PID=$(pgrep -xo sshd || pgrep -o sshd)
test -n "$SSHD_PID"
RELOAD_RECEIPT=/run/stwo-lab-sshd-reload.receipt
test ! -L /run
test -d /run
test "$(stat -c '%u:%g:%a' /run)" = 0:0:755
test ! -L "$RELOAD_RECEIPT"
if test -e "$RELOAD_RECEIPT"; then
  test -f "$RELOAD_RECEIPT"
  test "$(stat -c '%u:%g:%a:%h' "$RELOAD_RECEIPT")" = 0:0:600:1
fi
KEY_SHA256=$(printf '%s\n' "$KEY" | sha256sum | cut -d' ' -f1)
RELOAD_TOKEN="$(cat /proc/sys/kernel/random/boot_id):$SSHD_PID:$KEY_SHA256"
rm -f -- "$RELOAD_RECEIPT"
BOOTSTRAP_PID=$$
nohup sh -c '
  set -eu
  pid=$1; receipt=$2; token=$3; parent=$4
  tmp=$(mktemp /run/.stwo-lab-sshd-reload.XXXXXX)
  trap "rm -f -- \"$tmp\"" EXIT HUP INT TERM
  printf "%s\n" "$token" > "$tmp"
  chown 0:0 "$tmp"
  chmod 0600 "$tmp"
  attempt=0
  while kill -0 "$parent" 2>/dev/null; do
    attempt=$((attempt + 1))
    test "$attempt" -lt 100 || exit 1
    sleep 0.1
  done
  kill -HUP "$pid"
  sleep 0.5
  kill -0 "$pid"
  mv -T "$tmp" "$receipt"
  trap - EXIT HUP INT TERM
' stwo-sshd-reload "$SSHD_PID" "$RELOAD_RECEIPT" "$RELOAD_TOKEN" \
  "$BOOTSTRAP_PID" \
  </dev/null >/dev/null 2>&1 &
printf 'LABCTL_BOOTSTRAP_USER=dev\n'
printf 'LABCTL_BOOTSTRAP_UID=1000\n'
printf 'LABCTL_BOOTSTRAP_GID=1000\n'
printf 'LABCTL_BOOTSTRAP_KEY_SHA256=%s\n' "$KEY_SHA256"
"""


def bootstrap_dev(ep: c.Endpoint) -> str:
    public_key = _public_key()
    expected = hashlib.sha256((public_key + "\n").encode()).hexdigest()
    rc, output = c.ssh_capture(ep, _bootstrap_command(public_key), timeout=60)
    markers = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if line.startswith("LABCTL_")
    )
    if rc or markers != {
        "LABCTL_BOOTSTRAP_USER": "dev",
        "LABCTL_BOOTSTRAP_UID": "1000",
        "LABCTL_BOOTSTRAP_GID": "1000",
        "LABCTL_BOOTSTRAP_KEY_SHA256": expected,
    }:
        raise RuntimeError(f"unprivileged dev bootstrap failed: {output[-300:]}")
    return expected


def verify_ssh(ep: c.Endpoint, expected_key_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_key_sha256):
        raise ValueError("invalid expected bootstrap key digest")
    root_probe = f'''
set -eu
test "$(id -u):$(id -un)" = 0:root
EXPECTED_KEY={shlex.quote(expected_key_sha256)}
RELOAD_RECEIPT=/run/stwo-lab-sshd-reload.receipt
attempt=0
while test ! -e "$RELOAD_RECEIPT" && test "$attempt" -lt 50; do
  sleep 0.1
  attempt=$((attempt + 1))
done
test ! -L "$RELOAD_RECEIPT"
test -f "$RELOAD_RECEIPT"
test "$(stat -c '%u:%g:%a:%h' "$RELOAD_RECEIPT")" = 0:0:600:1
SSHD_PID=$(pgrep -xo sshd || pgrep -o sshd)
test -n "$SSHD_PID"
EXPECTED_RELOAD="$(cat /proc/sys/kernel/random/boot_id):$SSHD_PID:$EXPECTED_KEY"
test "$(cat "$RELOAD_RECEIPT")" = "$EXPECTED_RELOAD"
test "$(getent passwd dev | cut -d: -f3-4,6-7)" = 1000:1000:/home/dev:/bin/bash
test "$(getent group 1000 | cut -d: -f1,4)" = dev:
test "$(id -G dev):$(id -Gn dev)" = 1000:dev
POLICY=$(sshd -T -C user=dev,host=localhost,addr=127.0.0.1)
test "$(printf '%s\n' "$POLICY" | awk '$1=="authenticationmethods" {{print $2}}')" = publickey
test "$(printf '%s\n' "$POLICY" | awk '$1=="pubkeyauthentication" {{print $2}}')" = yes
test "$(printf '%s\n' "$POLICY" | awk '$1=="passwordauthentication" {{print $2}}')" = no
test "$(printf '%s\n' "$POLICY" | awk '$1=="kbdinteractiveauthentication" {{print $2}}')" = no
printf 'LABCTL_FRESH_ROOT_POLICY=publickey-only\n'
printf 'LABCTL_FRESH_ROOT_KEY_SHA256=%s\n' \
  "$(sha256sum /home/dev/.ssh/authorized_keys | cut -d' ' -f1)"
'''
    rc, output = c.ssh_capture(ep, root_probe, timeout=30)
    if rc or output.splitlines() != [
        "LABCTL_FRESH_ROOT_POLICY=publickey-only",
        f"LABCTL_FRESH_ROOT_KEY_SHA256={expected_key_sha256}",
    ]:
        raise RuntimeError(f"fresh root SSH verification failed: {output[-300:]}")
    dev = c.Endpoint(ep.host, ep.port, "dev")
    command = r'''
set -eu
test "$(id -u):$(id -g):$(id -un):$(id -gn):$HOME" = 1000:1000:dev:dev:/home/dev
test "$(id -G)" = 1000
test "$(id -Gn)" = dev
'''
    rc, output = c.ssh_capture(dev, command, timeout=30)
    if rc or output:
        raise RuntimeError(f"fresh dev SSH verification failed: {output[-300:]}")


def _record_payload(state: dict) -> dict:
    return {
        **{key: state[key] for key in (
            "formal", "image_digest", "image_digest_authority", "lane", "profile",
            "persistence_scope", "qualification", "qualification_eligible",
            "lease_local_root",
        )},
        "image": state["image"],
        "bootstrap_key_sha256": state["bootstrap_key_sha256"],
        "pod_id": state["pod_id"],
        "schema_version": "stwo.gpu-lab.bootstrap-profile.v1",
        "volume_dc": state["volume_dc"],
        "volume_id": state["volume_id"],
    }


def _record_command(state: dict) -> tuple[str, str, str]:
    payload = json.dumps(
        _record_payload(state), allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    encoded = base64.b64encode(payload).decode()
    digest = hashlib.sha256(payload).hexdigest()
    path = f"/workspace/gpu-lab/leases/{state['pod_id']}/records/{NAME}.json"
    command = f"""
set -eu
OUT={shlex.quote(path)}
ROOT=$(dirname "$OUT")
test ! -L "$ROOT"
test "$(stat -c '%u:%g:%a' "$ROOT")" = 0:0:700
test ! -e "$OUT"
test ! -L "$OUT"
umask 077
set -C
printf '%s' {shlex.quote(encoded)} | base64 --decode > "$OUT"
set +C
chmod 0400 "$OUT"
sync -f "$OUT"
sync -f "$ROOT"
test ! -L "$OUT"
test "$(stat -c '%u:%g:%a:%h' "$OUT")" = 0:0:400:1
test "$(sha256sum "$OUT" | cut -d' ' -f1)" = {digest}
printf 'LABCTL_PROFILE_RECORD=%s\n' "$OUT"
printf 'LABCTL_PROFILE_RECORD_SHA256={digest}\n'
"""
    return command, path, digest


def persist_record(ep: c.Endpoint, state: dict) -> tuple[str, str]:
    command, expected_path, expected_digest = _record_command(state)
    rc, output = c.ssh_capture(ep, command, timeout=60)
    paths = re.findall(r"^LABCTL_PROFILE_RECORD=(.+)$", output, re.MULTILINE)
    digests = re.findall(
        r"^LABCTL_PROFILE_RECORD_SHA256=([0-9a-f]{64})$", output, re.MULTILINE
    )
    if rc or paths != [expected_path] or digests != [expected_digest]:
        raise RuntimeError(f"bootstrap profile record failed: {output[-300:]}")
    return expected_path, expected_digest
