# Bounded cheap-GPU pod controller

`labctl` opens one Secure Cloud development lease; it never launches a proof. Its default idle
deadline is 30 minutes and its default wall-clock TTL is four hours. Preview first, then repeat the
exact command with the content-bound confirmation token.

## Safety model

- Before creation, `labctl` reads RunPod's authenticated `GET /v1/networkvolumes/{id}` response and
  binds the exact volume id, data center, name, and size into the confirmation token.
- After creation, it reads `GET /v1/pods/{id}` and fails closed unless that pod reports the exact
  network-volume id. When the Pod response exposes only the current flat `networkVolumeId`, it
  independently re-attests the volume's data center through `GET /v1/networkvolumes/{id}`; the
  GraphQL Pod record must also match that data center. `/workspace/gpu-lab/NETWORK_VOLUME_ID` is
  only a durable memo; it is not the source of truth.
- A local process owns provider-level termination. It enforces the TTL and, once the remote guard is
  installed, polls authenticated heartbeat/process state and terminates an idle or exited pod via
  the provider API. Three consecutive unobservable polls also terminate fail-closed.
- Separate credential-free remote TTL and idle processes seal evidence and stop PID 1. Neither
  remote process can issue a provider mutation, so it is a fallback, not a billing guarantee.
- `open` declares `/tmp/stwo-gpu-lab/<pod-id>` as the active local root and records the distinct
  local and `/workspace` mount identities in `LOCAL_ROOT.json`. The replay loop fails before build
  if an active fixture, chunk, module, plan, replay, execution manifest, harness, or build output
  resolves outside that root.
- `accept` persists a verified content-addressed snapshot. `close`, the idle guard, and the TTL
  guard first freeze the local tree, persist records/results/profiles, re-scan for concurrent
  mutation, write a durable manifest and `SEAL.sha256`, and only then permit termination. A
  missing, changed, symlinked, or incompletely copied output leaves compute running and reports the
  exact persistence failure; it is never treated as a successful close.

The residual trust boundary is RunPod's authenticated control-plane response. A process inside the
container cannot cryptographically prove which physical block device the provider mounted. Missing,
ambiguous, or mismatched REST fields abort before useful work.

## Fast development use

Bind one exact local private key before invoking any command that constructs SSH transport. The
controller rejects symlinks, keys not owned by the current user, group/world permissions, missing
keys, and ambiguous automatic discovery; it never reads key contents. This machine has multiple
RunPod keys, so set the path explicitly rather than relying on discovery:

```bash
export RUNPOD_SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go" # path only; never key contents
chmod 0600 "$RUNPOD_SSH_KEY"
gpu_benchmarks/lab/pod/labctl self-test
gpu_benchmarks/lab/pod/labctl open --image IMAGE@sha256:DIGEST \
  --volume-id ID --volume-dc DC
gpu_benchmarks/lab/pod/labctl heartbeat
gpu_benchmarks/lab/pod/labctl sync
gpu_benchmarks/lab/pod/labctl accept --profile
```

Use the formal `stwo` consumer-development image described in
[`stwo/gpu-lab/docker/README.md`](../../../../stwo/gpu-lab/docker/README.md). Copy its published
manifest reference from the build record; a mutable tag, a locally loaded image id, or a generic
RunPod/PyTorch image is not admissible. Replace both placeholders below before previewing:

Until that image is published, one explicitly non-formal development exception is pinned in code:

```bash
gpu_benchmarks/lab/pod/labctl open \
  --bootstrap-profile consumer-4090-bootstrap
```

That profile is fixed to one digest-pinned RunPod CUDA 12.8 image, one Secure RTX 4090, network
volume `2kpphx92fr` in `EU-RO-1`, and ceilings of 6 hours, 30 idle minutes, $0.80/hour, and $4.80.
It creates and verifies only the `1000:1000` development identity before installing the ordinary
guards. Its records say `formal=false` and `qualification_eligible=false`; `labctl accept` rejects
it, so the lane cannot produce qualification or headline evidence. The recorded image digest is
the requested immutable reference, not a runtime attestation of the provider-started filesystem.
For this one known volume only, the profile also migrates its legacy `root:root` `0777` controller
root and `0666` volume-id memo to `0755`/`0600`. It first anchors both inodes without following
links, closes them to non-root mutation, verifies the memo against the provider-attested volume id,
fsyncs and re-attests both objects, and records the transition; no recursive mode repair is allowed.

```bash
gpu_benchmarks/lab/pod/labctl open \
  --gpu 4090 \
  --image '<published-registry>/stwo-consumer-dev@sha256:<published-manifest-digest>' \
  --volume-id 2kpphx92fr --volume-dc EU-RO-1 \
  --ttl-hours 6 --idle-min 30 --max-usd-hr 0.75 --max-total-usd 4.50
```

The first invocation is a non-mutating preview and prints the exact confirmation
token. Repeat it with `--confirm TOKEN` to open the lease. Match the cubin to the selected card
(`sm_86` for 3090, `sm_89` for 4090, or `sm_120` for 5090). Numbers from this lane are same-device
development evidence, not H100 or multi-GPU headline results.

Run `labctl heartbeat` periodically during Cursor/VS Code remote sessions. `sync` never edits the
persistent `/workspace/src` seed checkouts or an older published source tree. It copies seed Git
objects into a fresh controller transaction, fetches an exact minimal bundle when HEAD advanced,
stages dirty files separately, preserves replaced/deleted staging bytes in quarantine, and verifies
the complete local identity before an atomic no-replace rename into a SHA-256 generation path.
Only then does it atomically swap `source-generations/CURRENT` to an immutable pointer manifest.
Every remote source-controller program runs as root in production. The controller root, generation
tree, pointer symlink, and pointer manifest must remain root-owned; a non-root development process
cannot publish, replace, reuse, or attest those objects.
After the durable sync record binds that publication, success cleanup removes the remaining
transaction, transferred objects, and quarantine. Failed or interrupted transactions are compacted
into immutable bounded failure manifests; reconciliation validates an existing same-token manifest
and safely finishes cleanup after a crash between evidence publication and transaction removal.
Transactions are capped at 40 GiB and one million entries, must leave 20 GiB free, and the failure
store is capped at 256 manifests. Prior published generations remain intact because this slice does
not automatically garbage-collect generation history.

Build, profile, sanitizer, and benchmark commands must resolve the immutable manifest named by
`/workspace/gpu-lab/source-generations/CURRENT`, record its manifest SHA-256, and use only the exact
repository paths inside its `generation_path`. Resolve them once per run; do not infer a generation
from a directory listing or substitute `/workspace/src`:

```bash
CURRENT=/workspace/gpu-lab/source-generations/CURRENT
POINTER_MANIFEST="$(readlink -f "$CURRENT")"
POINTER_SHA256="$(sha256sum "$POINTER_MANIFEST" | cut -d' ' -f1)"
GENERATION_PATH="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["generation_path"])' \
  "$POINTER_MANIFEST")"
STWO_SOURCE="$GENERATION_PATH/repos/stwo"
STWO_CAIRO_SOURCE="$GENERATION_PATH/repos/stwo-cairo"
```

Record `POINTER_SHA256` with every result and run build, profile, sanitizer, and benchmark commands
only from `STWO_SOURCE` or `STWO_CAIRO_SOURCE`. `/workspace/src` is a read-only object seed, never a
build input. The sync evidence binds the generation path/hash, pointer path/target/hash, each exact
HEAD, and each full-tree identity. Fixture staging remains beneath the declared local root and binds
every source path, local destination, byte count, and SHA-256. Durable objects land under:

```text
/workspace/gpu-lab/leases/<pod-id>/records/sha256/
/workspace/gpu-lab/leases/<pod-id>/results/sha256/
/workspace/gpu-lab/leases/<pod-id>/profiles/sha256/
/workspace/gpu-lab/leases/<pod-id>/persists/<manifest-sha256>.json
/workspace/gpu-lab/source-generations/sha256/<generation-sha256>/
/workspace/gpu-lab/source-generations/manifests/<pointer-sha256>.json
```

Acceptance requires the image's pinned
`nsight-systems-2026.1.3=2026.1.3.425-261338342291v0`, executes
`nsys --version`, and records that package identity. Profile acceptance additionally executes one
real Nsight Compute counter collection through the authenticated `dev` SSH endpoint at UID/GID
`1000:1000`. The root control plane records the exact dev receipt and its SHA-256; neither check
launches the prover.

Known hardening follow-ups are explicit: replace disabled SSH host-key checking with a provider-bound
host identity when RunPod exposes one; make `GPU_LAB_LOCAL_ROOT` available to non-login IDE command
sessions without relying on `/etc/profile.d` (source that file explicitly today); and replace the
fixed first-install heartbeat/active-root temporary names with securely created root-owned files.
These are P2 defense-in-depth items, not accepted profile or benchmark evidence.

Tests are stdlib-only and prohibit provider, REST, SSH, rsync transport, and GPU calls. Keep
`labctl` as the tiny stable entrypoint, place behavior in its owning `labctl_lib` module, keep every
handwritten file below 500 lines, and follow the lab's canonical
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for taste, evidence, GPU work, and feedback-loop budgets.
The `labctl` shebang and local test contract are Python 3.11; use `python3.11`, not the macOS
`/usr/bin/python3` 3.9, for direct fleet test invocations.
