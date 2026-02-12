"""
Performance logger for tracking GPU vs CPU training speed.

Logs wall-clock timing of key operations (env step, buffer sample,
network update, total epoch) and reports throughput (steps/sec, updates/sec).

Usage:
    perf = PerformanceLogger(device, log_dir="logs/Ant_v5/vanilla/seed_0")
    perf.start("env_step")
    ...
    perf.stop("env_step")
    perf.epoch_summary(epoch, global_step)
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import torch


class PerformanceLogger:
    """Track and log wall-clock timings per operation per epoch.

    Maintains running accumulators for each named timer.  At epoch boundary,
    call ``epoch_summary()`` to print & log stats and reset counters.

    Timers:
        env_step      — time in env.step()
        buffer_sample — time sampling from replay buffer
        update        — time for one gradient update batch
        epoch_total   — full wall-clock for the epoch
    """

    def __init__(
        self,
        device: torch.device,
        log_dir: str,
        csv_name: str = "perf_log.csv",
    ):
        self.device = device
        self.device_name = self._device_label(device)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, csv_name)

        # Running accumulators: timer_name -> list of durations (seconds)
        self._timers: Dict[str, float] = {}           # start timestamps
        self._accum: Dict[str, list[float]] = defaultdict(list)

        # Epoch-level wall-clock
        self._epoch_start: Optional[float] = None

        # Write CSV header if file doesn't exist
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch", "global_step", "device",
                    "epoch_wall_s",
                    "steps_per_sec",
                    "updates_per_sec",
                    "env_step_total_s", "env_step_mean_ms",
                    "buffer_sample_total_s", "buffer_sample_mean_ms",
                    "update_total_s", "update_mean_ms",
                    "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
                ])

    # ------------------------------------------------------------------
    @staticmethod
    def _device_label(device: torch.device) -> str:
        if device.type == "cuda":
            idx = device.index or 0
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = f"cuda:{idx}"
            return f"GPU ({name})"
        return "CPU"

    # ------------------------------------------------------------------
    def start_epoch(self) -> None:
        """Call at the beginning of each epoch."""
        self._accum.clear()
        self._epoch_start = time.perf_counter()

    def start(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """Stop a named timer and record the elapsed duration."""
        t0 = self._timers.pop(name, None)
        if t0 is not None:
            self._accum[name].append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def epoch_summary(
        self,
        epoch: int,
        global_step: int,
        steps_this_epoch: int = 0,
        updates_this_epoch: int = 0,
        print_summary: bool = True,
    ) -> Dict[str, float]:
        """Compute, print, and log epoch performance stats.

        Returns a dict of computed metrics.
        """
        epoch_wall = (time.perf_counter() - self._epoch_start
                      if self._epoch_start else 0.0)

        def _stats(name: str):
            vals = self._accum.get(name, [])
            total = sum(vals)
            mean_ms = (total / len(vals) * 1000) if vals else 0.0
            return total, mean_ms, len(vals)

        env_total, env_mean, env_count = _stats("env_step")
        buf_total, buf_mean, buf_count = _stats("buffer_sample")
        upd_total, upd_mean, upd_count = _stats("update")

        steps_per_sec = steps_this_epoch / epoch_wall if epoch_wall > 0 else 0.0
        updates_per_sec = updates_this_epoch / epoch_wall if epoch_wall > 0 else 0.0

        # GPU memory
        gpu_alloc_mb = 0.0
        gpu_reserved_mb = 0.0
        if self.device.type == "cuda":
            gpu_alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            gpu_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

        metrics = {
            "epoch_wall_s": epoch_wall,
            "steps_per_sec": steps_per_sec,
            "updates_per_sec": updates_per_sec,
            "env_step_total_s": env_total,
            "env_step_mean_ms": env_mean,
            "buffer_sample_total_s": buf_total,
            "buffer_sample_mean_ms": buf_mean,
            "update_total_s": upd_total,
            "update_mean_ms": upd_mean,
            "gpu_mem_allocated_mb": gpu_alloc_mb,
            "gpu_mem_reserved_mb": gpu_reserved_mb,
        }

        # Print
        if print_summary:
            print(f"  [perf] device={self.device_name}  "
                  f"epoch_time={epoch_wall:.1f}s  "
                  f"steps/s={steps_per_sec:.0f}  "
                  f"updates/s={updates_per_sec:.0f}")
            print(f"         env_step={env_total:.2f}s ({env_mean:.2f}ms/call x{env_count})  "
                  f"buf_sample={buf_total:.2f}s ({buf_mean:.2f}ms/call x{buf_count})  "
                  f"update={upd_total:.2f}s ({upd_mean:.2f}ms/call x{upd_count})")
            if self.device.type == "cuda":
                print(f"         GPU mem: alloc={gpu_alloc_mb:.0f}MB  "
                      f"reserved={gpu_reserved_mb:.0f}MB")

        # Log to CSV
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, global_step, self.device_name,
                f"{epoch_wall:.2f}",
                f"{steps_per_sec:.1f}",
                f"{updates_per_sec:.1f}",
                f"{env_total:.3f}", f"{env_mean:.3f}",
                f"{buf_total:.3f}", f"{buf_mean:.3f}",
                f"{upd_total:.3f}", f"{upd_mean:.3f}",
                f"{gpu_alloc_mb:.1f}", f"{gpu_reserved_mb:.1f}",
            ])

        return metrics
