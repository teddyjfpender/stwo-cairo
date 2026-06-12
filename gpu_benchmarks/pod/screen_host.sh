#!/bin/bash
# Fleet host-screening probe (round-9 lesson: one weak community host delivered
# 40% of its peers' throughput — RunPod pricing doesn't price CPU quality).
# Run FIRST on a candidate pod; commit the member only on PASS.
#
# Scores ~10s of work: single-core integer throughput (the VM run is
# single-threaded — the sustained-pipeline ceiling) + core count (prefetch
# workers + witness writes). Thresholds from the round-9/10 cohort: good hosts
# score >= 900 single-core; the weak one scored ~380.
set -u
CORES=$(nproc)
RAM_GB=$(free -g | awk 'NR==2{print $2}')

# Single-core probe: tight 64-bit mulxor loop, ~10s, portable (bash arithmetic
# is too slow/noisy; python3 ships in the image).
SCORE=$(python3 - <<'PY'
import time
t0 = time.time()
x = 1469598103934665603
n = 0
while time.time() - t0 < 10.0:
    for _ in range(100000):
        x = (x * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 33
    n += 1
print(n)
PY
)

echo "HOST_SCREEN cores=$CORES ram_gb=$RAM_GB single_core_score=$SCORE"
if [ "$SCORE" -ge 900 ] && [ "$CORES" -ge 16 ] && [ "$RAM_GB" -ge 32 ]; then
  echo "HOST_SCREEN_RESULT PASS"
  exit 0
fi
echo "HOST_SCREEN_RESULT FAIL (thresholds: score>=900, cores>=16, ram>=32GB)"
exit 1
