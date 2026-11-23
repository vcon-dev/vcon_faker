"""Memory on a shared machine: peak footprint, a sampler, and a pause when free memory is low.

On Apple silicon MLX keeps model memory mostly outside the resident set size, so the honest
per-process figure is the peak memory footprint. System-wide, ``wait_for_memory`` holds the run
before each heavy step while free memory is under the floor (2 GB by default), because the
machine also runs Docker and other sessions.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time

MB = 1024 * 1024


class MemoryLow(Exception):
    pass


def peak_footprint_bytes() -> int | None:
    """This process's lifetime peak memory footprint on macOS (``proc_pid_rusage``, flavor 4).

    The footprint counts MLX's GPU buffers, which the resident set size mostly does not.
    """
    if sys.platform != "darwin":
        return None
    u = ctypes.c_uint64
    names = (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size "
        "phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time "
        "child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread "
        "diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
        "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive "
        "billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions "
        "cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time flags user_ptime "
        "system_ptime pinstructions pcycles energy_nj penergy_nj"
    ).split()

    class RUsageInfoV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), *[(n, u) for n in names], ("reserved", u * 14)]

    info = RUsageInfoV4()
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return int(info.lifetime_max_phys_footprint)


class MemorySampler(threading.Thread):
    """Peak resident memory of this process and its children, and the lowest free system memory."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        import psutil

        self.psutil = psutil
        self.me = psutil.Process()
        self.interval = interval
        self.peak_self = 0
        self.peak_children = 0
        self.min_available = psutil.virtual_memory().available
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                kids = 0
                for child in self.me.children(recursive=True):
                    try:
                        kids += child.memory_info().rss
                    except self.psutil.Error:
                        pass
                self.peak_self = max(self.peak_self, self.me.memory_info().rss)
                self.peak_children = max(self.peak_children, kids)
                self.min_available = min(self.min_available, self.psutil.virtual_memory().available)
            except self.psutil.Error:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2)


class MemoryGuard:
    """Pauses the run while free memory is under ``min_free_mb``; gives up after ``max_wait`` s."""

    def __init__(self, min_free_mb: int = 2048, max_wait: float = 1200.0, poll: float = 15.0) -> None:
        self.min_free_mb = min_free_mb
        self.max_wait = max_wait
        self.poll = poll
        self.pauses = 0
        self.paused_seconds = 0.0

    def wait(self, what: str = "") -> None:
        import psutil

        waited = 0.0
        while (free := psutil.virtual_memory().available / MB) < self.min_free_mb:
            if waited == 0:
                self.pauses += 1
                print(f"pausing{' before ' + what if what else ''}: {free:.0f} MB free, under {self.min_free_mb} MB", flush=True)
            if waited >= self.max_wait:
                raise MemoryLow(f"free memory stayed under {self.min_free_mb} MB for {waited:.0f} s")
            time.sleep(self.poll)
            waited += self.poll
        if waited:
            self.paused_seconds += waited
            print(f"resuming after {waited:.0f} s: {free:.0f} MB free", flush=True)
