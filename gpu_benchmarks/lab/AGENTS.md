# GPU lab agent guidance

Follow [`CONTRIBUTING.md`](CONTRIBUTING.md) for every change in this subtree. Preserve fail-closed
lease/cost behavior, make no provider call from tests, keep `pod/labctl` as a tiny stable
entrypoint, and place each responsibility in the matching `pod/labctl_lib/` module.
