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
  GraphQL Pod record must also match that data center. The formal lane's
  `/workspace/gpu-lab/NETWORK_VOLUME_ID` and the bootstrap lane's quarantined
  `/runpod-volume/gpu-lab/NETWORK_VOLUME_ID` are filesystem memos; neither replaces the provider
  attestation.
- A local process owns provider-level termination. It enforces the TTL and, once the remote guard is
  installed, polls authenticated heartbeat/process state and terminates an idle or exited pod via
  the provider API. Three consecutive unobservable polls also terminate fail-closed.
- Separate credential-free remote TTL and idle processes stop PID 1. The formal lane first seals
  durable evidence. The explicitly ephemeral bootstrap lane records no durable-seal claim and
  discards its controller disk. Neither remote process can issue a provider mutation, so it is a
  fallback, not a billing guarantee.
- `open` declares `/tmp/stwo-gpu-lab/<pod-id>` as the active local root and records its mount
  identity in `LOCAL_ROOT.json`. Formal runs require that root and `/workspace` to be distinct.
  Bootstrap runs require both to resolve to the fresh container disk while `/runpod-volume`
  remains distinct and quarantined. The replay loop fails before build if an active fixture,
  chunk, module, plan, replay, execution manifest, harness, or build output resolves outside the
  active root.
- In the formal lane, `accept` persists a verified content-addressed snapshot. `close`, the idle
  guard, and the TTL guard first freeze the local tree, persist records/results/profiles, re-scan
  for concurrent mutation, write a durable manifest and `SEAL.sha256`, and only then permit
  termination. The nonformal bootstrap profile cannot be accepted and makes no persistence claim.

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
gpu_benchmarks/lab/pod/labctl resolve > /tmp/stwo-gpu-source.json
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
It mounts the provider network volume only at `/runpod-volume`; it never repairs or trusts that
volume's known legacy `root:root` `0777` `/runpod-volume/gpu-lab` controller root or `0666`
volume-id memo. It verifies that exact legacy state without following links, requires the provider
mount identity/device to differ from the container root, and requires `/workspace` to resolve to a
root-owned, non-group/world-writable container directory. It then creates and attests a fresh
`root:root` `0755` `/workspace/gpu-lab` directly on the container disk, with a root-owned `0400`
lease marker bound to the pod, volume, and observed boot. No mount syscall or bind mount is used.
Before every remote operation, `labctl` re-attests that marker, the quarantined provider mount,
heartbeat, guard programs, PID files, and live guard processes. A missing/restarted controller is
terminated through the provider API instead of silently continuing to accrue spend.

This exception records `persistence_scope=lease-local-container-disk` and `qualification=false`.
It is development-only. The network volume is a seed input used read-only by policy at
`/runpod-volume/src`; it is not a cache, source-generation, result, profile, or evidence authority
in this lane. Published source generations and everything beneath `/workspace/gpu-lab` are on the
pod's container disk and are lost when the pod terminates. Sync source again on the next lease and
copy out any informal diagnostics before closing. Formal development, acceptance, qualification,
and headline evidence still require the published consumer image and the unmodified
persistent-volume guards.

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
Transactions are capped at 40 GiB and one million entries, must leave 20 GiB free on the controller
device, and the failure store is capped at 256 manifests. Prior published generations remain intact
because this slice does not automatically garbage-collect generation history. Bounded generation
garbage collection remains a follow-up before the ephemeral lane is used for high-churn,
multi-generation sessions.

Build, profile, sanitizer, and benchmark commands must use the immutable manifest named by
`/workspace/gpu-lab/source-generations/CURRENT`, record its manifest SHA-256, and use only the exact
repository paths inside its `generation_path`. Resolve it once per run through the authenticated
root controller; never list generations, read the root-owned `0400` manifest as `dev`, or substitute
`/workspace/src`:

```bash
gpu_benchmarks/lab/pod/labctl resolve > /tmp/stwo-gpu-source.json
python3.11 -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(d["pointer_sha256"]);
print(d["repositories"]["stwo"]["path"]);
print(d["repositories"]["stwo-cairo"]["path"])' /tmp/stwo-gpu-source.json
gpu_benchmarks/lab/pod/labctl shell
```

The local JSON contains the pointer path/target/hash, canonical generation path/hash, exact `stwo`
and `stwo-cairo` paths, and each repository's HEAD, full-tree hash, and worktree-identity hash. Use
those paths in the `dev` SSH session and record `pointer_sha256` with every result. `/workspace/src`
is a read-only object seed, never a build input. The sync evidence binds the same identities.
Fixture staging remains beneath the declared local root and binds every source path, local
destination, byte count, and SHA-256. Durable objects land under:

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
