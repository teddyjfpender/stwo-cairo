# GPU lab baselines

Baselines are device-specific regression references, never cross-device speed claims. Each file is
an immutable envelope containing the aggregated fixture, oracle, ABI, shape, toolchain/flags,
module, timing, derived semantic throughput, environment, and transitive snapshot identities.

Promote one only through `stwo/gpu-lab/tools/accept-baseline`. The tool revalidates immutable
top-level snapshots, proves the transitive dependency set did not move during validation, requires
complete pre/post GPU environment evidence and non-exploratory 30-sample eager/graph timing, and
prints a confirmation token binding the candidate, destination, and previous accepted file.
Correctness failures cannot become baselines.

The envelope records the validated `build_recipe_hash` and the complete module-index snapshot
identity. It deliberately does not duplicate the full recipe, so offline review cannot recompute that
one hash from the envelope alone; promotion revalidates it against the immutable module-index
snapshot and the transitive source/tool closure before writing the envelope.
