# energy_utils.py
#
# Pomiar energii zużytej przez CPU podczas pojedynczego runu benchmarku.
#
# API:
#   from energy_utils import EnergyLogger
#
#   logger = EnergyLogger()
#   logger.start()
#   ... uruchom kernel ...
#   energy_j, power_w = logger.stop()
#
# Na Linux + Intel: wykorzystuje RAPL (powercap).
# Na macOS: próba użycia 'powermetrics' (jeśli dostępne).
# W innych przypadkach – zwraca NaN (pomiar niedostępny).

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import threading
import time
from glob import glob
from shutil import which


SYSTEM = platform.system()


# --------- Backend "dummy" (brak pomiaru) ---------


class _DummyBackend:
    def __init__(self) -> None:
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> tuple[float, float]:
        t1 = time.perf_counter()
        dt = t1 - (self._t0 or t1)
        # brak realnego pomiaru
        return (math.nan, math.nan)


# --------- Backend Linux RAPL ---------


class _RaplBackend:
    """
    Prostolinijny backend do RAPL:
    - zczytuje energy_uj z /sys/class/powercap/intel-rapl:*/energy_uj
    - sumuje domeny (package, dram, itd.)
    """

    def __init__(self) -> None:
        self.paths = sorted(
            glob("/sys/class/powercap/intel-rapl:*/*/energy_uj")
        ) or sorted(glob("/sys/class/powercap/intel-rapl:*/energy_uj"))

        if not self.paths:
            raise RuntimeError("Brak ścieżek RAPL w /sys/class/powercap")

        self._t0 = None
        self._e0 = None

    def _read_energy_j(self) -> float:
        total_uj = 0
        for p in self.paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    total_uj += int(f.read().strip())
            except Exception:
                continue
        # mikro-dżule -> dżule
        return total_uj / 1e6

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._e0 = self._read_energy_j()

    def stop(self) -> tuple[float, float]:
        t1 = time.perf_counter()
        e1 = self._read_energy_j()
        if self._t0 is None or self._e0 is None:
            return (math.nan, math.nan)

        # obsługa ewentualnego wrap-around
        energy_j = max(0.0, e1 - self._e0)
        dt = t1 - self._t0
        power_w = energy_j / dt if dt > 0 else math.nan
        return (energy_j, power_w)


# --------- Backend macOS powermetrics ---------


class _PowermetricsBackend:
    """
    Backend dla macOS.

    Uruchamia powermetrics w osobnym procesie i parsuje linie "CPU Power: X.YW"
    w wątku, integrując po czasie. Wymaga:
        - narzędzia 'powermetrics' w PATH
        - uprawnień root (uruchamiaj benchmarki przez 'sudo python3 ...').
    """

    def __init__(self, interval_ms: int = 50) -> None:
        if which("powermetrics") is None:
            raise RuntimeError("Brak 'powermetrics' w PATH")

        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []  # (czas, moc W)
        self._t0: float | None = None

    def start(self) -> None:
        cmd = [
            "powermetrics",
            "--samplers",
            "cpu_power",
            "-i",
            str(self.interval_ms),
            "-n",
            "1000000",  # duża liczba, przerwiemy sami
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._samples.clear()

        def reader():
            assert self._proc is not None
            for line in self._proc.stdout:
                if self._stop_event.is_set():
                    break
                if "CPU Power" in line:
                    m = re.search(r"([0-9]+(?:\.[0-9]*)?)\s*W", line)
                    if m:
                        w = float(m.group(1))
                        t = time.perf_counter()
                        self._samples.append((t, w))
            # po wyjściu z pętli spróbuj delikatnie zabić proces
            try:
                self._proc.terminate()
            except Exception:
                pass

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

        if self._t0 is None or not self._samples:
            return (math.nan, math.nan)

        samples = self._samples
        energy = 0.0
        # prosta integracja trapezami
        last_t, last_w = samples[0]
        for t, w in samples[1:]:
            dt = t - last_t
            if dt > 0:
                energy += 0.5 * (last_w + w) * dt
            last_t, last_w = t, w

        # dociągnięcie do t1 ostatnią mocą
        dt_end = t1 - last_t
        if dt_end > 0:
            energy += last_w * dt_end

        t0 = self._t0
        total_dt = t1 - t0 if t0 is not None else 0.0
        avg_power = energy / total_dt if total_dt > 0 else math.nan
        return (energy, avg_power)


# --------- Fabryka backendu + główna klasa ---------


def _create_backend():
    if SYSTEM == "Linux":
        try:
            return _RaplBackend()
        except Exception:
            return _DummyBackend()
    if SYSTEM == "Darwin":
        try:
            return _PowermetricsBackend(interval_ms=50)
        except Exception:
            return _DummyBackend()
    return _DummyBackend()


class EnergyLogger:
    """
    Główna klasa używana przez benchmarki.

    Przykład:
        logger = EnergyLogger()
        logger.start()
        ... benchmark ...
        energy_j, power_w = logger.stop()
    """

    def __init__(self) -> None:
        self._backend = _create_backend()

    def start(self) -> None:
        self._backend.start()

    def stop(self) -> tuple[float, float]:
        try:
            return self._backend.stop()
        except Exception:
            return (math.nan, math.nan)
