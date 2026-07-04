"""GPU utilization monitor: samples nvidia-smi during a step into a sidecar CSV
and raises an idle alarm when a step declared `gpu_bound = true` shows a dead
device — the failure mode (provisioned + billing + 0% SM) that once went
unnoticed until a human looked at the pod.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .podctl import Endpoint, ssh_capture

SAMPLE_CMD = (
    "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw "
    "--format=csv,noheader,nounits 2>/dev/null | head -1"
)
SAMPLE_PERIOD_S = 15
IDLE_THRESHOLD_PCT = 5
IDLE_ALARM_AFTER_S = 180


class GpuMonitor:
    def __init__(self, ep: Endpoint, csv_path: Path, *, expect_busy: bool):
        self.ep = ep
        self.csv_path = csv_path
        self.expect_busy = expect_busy
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle_streak_s = 0.0
        self.idle_alarm = False

    def _loop(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a") as f:
            f.write("ts,gpu_util_pct,mem_util_pct,mem_used_mib,power_w\n")
            while not self._stop.wait(SAMPLE_PERIOD_S):
                try:
                    rc, out = ssh_capture(self.ep, SAMPLE_CMD, timeout=20)
                except Exception:
                    continue
                if rc != 0 or not out:
                    continue
                f.write(f"{time.time():.0f},{out.replace(', ', ',')}\n")
                f.flush()
                try:
                    util = float(out.split(",")[0])
                except (ValueError, IndexError):
                    continue
                if util < IDLE_THRESHOLD_PCT:
                    self._idle_streak_s += SAMPLE_PERIOD_S
                    if (
                        self.expect_busy
                        and self._idle_streak_s >= IDLE_ALARM_AFTER_S
                        and not self.idle_alarm
                    ):
                        self.idle_alarm = True
                        print(
                            f"[gpufleet] ALARM: GPU idle >{IDLE_ALARM_AFTER_S}s "
                            f"during a gpu_bound step ({self.csv_path.stem})"
                        )
                else:
                    self._idle_streak_s = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=SAMPLE_PERIOD_S + 25)
        return self.idle_alarm
