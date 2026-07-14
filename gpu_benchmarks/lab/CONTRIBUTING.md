# GPU lab contribution policy

This is the orchestration/oracle half of the same lab as `stwo/gpu-lab`. Read and follow the
canonical [`CONTRIBUTING.md`](../../../stwo/gpu-lab/CONTRIBUTING.md), including its independent-
oracle, evidence, file-size, progressive-disclosure, GPU-measurement, and feedback-SLO rules.
That document is authoritative: keep orchestration-specific guidance here and change shared policy
there rather than allowing the two halves of the replacement backend to drift.

`pod/labctl` is the stable, tiny entrypoint; its implementation lives in `pod/labctl_lib/`, split
by durable responsibility: common policy/state, provider mutations, lease runtime/lifecycle, tree
sync, acceptance/profiling, CLI routing, and offline self-tests. Keep each handwritten file below
500 lines (prefer below 350), preserve the fail-closed lease/cost controls and exact CLI, and prove
every change with the no-network self-test before any provider call. Read
[`pod/README.md`](pod/README.md) for the volume-attestation, independent TTL/idle-watchdog,
heartbeat, and exact-sync trust model.
