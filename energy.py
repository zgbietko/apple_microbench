#!/usr/bin/env python3
"""
energy.py

Warstwa abstrakcji do pomiaru energii CPU.

Obsługiwane backendy:
- Linux/x86: RAPL (/sys/class/powercap/intel-rapl:0/energy_uj
                lub /sys/class/powercap/amd-rapl:0/energy_uj)
- macOS (Apple Silicon): powermetrics --samplers cpu_power

API:
- energy_measurement_supported() -> bool
- read_energy_joules() -> Optional[float]
    Zwraca monotoniczny licznik energii [J] od "jakiegoś" punktu
    odniesienia (RAPL od restartu, powermetrics od uruchomienia
    monitora). Benchmarki biorą różnicę przed/po runie.

Jeśli pomiar jest niedostępny, zwraca None.
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import re
import shutil


# ============================================================
#  Linux: RAPL
# ============================================================

_RAPL_FILE: Optional[Path] = None
_RAPL_PROBED: bool = False


def _detect_rapl_file() -> Optional[Path]:
    """
    Szuka pliku energy_uj dla intel-rapl:0 lub amd-rapl:0.
    Wynik jest cache'owany w _RAPL_FILE.
    """
    global _RAPL_FILE, _RAPL_PROBED

    if _RAPL_PROBED:
        return _RAPL_FILE

    _RAPL_PROBED = True

    if platform.system() != "Linux":
        _RAPL_FILE = None
        return None

    candidates = [
        Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
        Path("/sys/class/powercap/amd-rapl:0/energy_uj"),
    ]

    for p in candidates:
        if p.exists():
            _RAPL_FILE = p
            break
    else:
        _RAPL_FILE = None

    return _RAPL_FILE


def _read_rapl_energy_j() -> Optional[float]:
    rapl = _detect_rapl_file()
    if rapl is None:
        return None
    try:
        text = rapl.read_text().strip()
        uj = int(text)  # mikro-dżule
        return uj / 1e6  # -> J
    except Exception:
        return None


# ============================================================
#  macOS: powermetrics --samplers cpu_power
# ============================================================

class _MacPowermetricsMonitor:
    """
    Uruchamia w tle `powermetrics --samplers cpu_power -i <ms>`
    i integruje moc CPU w czasie, aby oszacować energię [J].

    Działa w osobnym wątku, parsując linie zawierające "CPU Power".
    Uproszczenie: każdą próbkę traktujemy jako średnią moc w przedziale
    o długości interval_s (tym, który podaliśmy w -i).

    Uwaga: wymaga uruchomienia benchmarków z sudo:
        sudo python3 run_all_cpu_benchmarks.py
    """

    def __init__(self, interval_ms: int = 200) -> None:
        self.interval_ms = interval_ms
        self.interval_s = interval_ms / 1000.0
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._total_energy_j: float = 0.0
        self._running = False
        # CPU Power: 6.80 W  /  CPU Power: 16835 mW
        self._power_re = re.compile(
            r"CPU Power:\s+([0-9]+(?:\.[0-9]+)?)\s*(mW|W)"
        )

    def start(self) -> bool:
        """
        Startuje powermetrics i wątek czytający.
        Zwraca True przy sukcesie, False gdy nie można uruchomić.
        """
        cmd = [
            "powermetrics",
            "--samplers",
            "cpu_power",
            "-i",
            str(self.interval_ms),
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"[energy] Nie udało się uruchomić powermetrics: {e}")
            return False

        # Krótko poczekaj – jeśli proces od razu padł (np. brak sudo),
        # przeczytaj stderr i wyłącz backend.
        time.sleep(0.3)
        if self.proc.poll() is not None:
            try:
                err = (self.proc.stderr.read() or "").strip()
            except Exception:
                err = ""
            msg = "[energy] powermetrics zakończył się błędem."
            if err:
                msg += f" Wyjście powermetrics:\n{err}"
            else:
                msg += " (brak dodatkowego outputu)"
            print(msg)
            return False

        self._running = True

        def _reader():
            assert self.proc is not None
            assert self.proc.stdout is not None

            for line in self.proc.stdout:
                if not self._running:
                    break

                m = self._power_re.search(line)
                if not m:
                    continue

                try:
                    value = float(m.group(1))
                    unit = m.group(2)
                    if unit == "mW":
                        w = value / 1000.0
                    else:
                        w = value
                    if w <= 0.0:
                        continue  # ignorujemy zerowe/ujemne próbki
                    with self._lock:
                        self._total_energy_j += w * self.interval_s
                except Exception:
                    # jakiekolwiek dziwne formaty linii po prostu ignorujemy
                    continue

            self._running = False

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        print("[energy] macOS: uruchomiono monitor powermetrics (CPU power).")
        return True

    def get_energy_j(self) -> float:
        """
        Zwraca dotychczas zintegrowaną energię [J].
        """
        with self._lock:
            return self._total_energy_j

    def stop(self) -> None:
        self._running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass


_MAC_MONITOR: Optional[_MacPowermetricsMonitor] = None
_MAC_MONITOR_DISABLED: bool = False  # np. brak sudo / błąd


def _get_mac_monitor() -> Optional[_MacPowermetricsMonitor]:
    """
    Zwraca globalny monitor powermetrics; startuje go przy pierwszym wywołaniu.
    Jeśli nie uda się wystartować (np. brak sudo) – zwraca None
    i ustawia _MAC_MONITOR_DISABLED = True.
    """
    global _MAC_MONITOR, _MAC_MONITOR_DISABLED

    if _MAC_MONITOR_DISABLED:
        return None

    if _MAC_MONITOR is not None:
        return _MAC_MONITOR

    if platform.system() != "Darwin":
        _MAC_MONITOR_DISABLED = True
        return None

    if shutil.which("powermetrics") is None:
        print("[energy] Nie znaleziono narzędzia 'powermetrics' w PATH.")
        _MAC_MONITOR_DISABLED = True
        return None

    monitor = _MacPowermetricsMonitor(interval_ms=10)
    if not monitor.start():
        print("[energy] Wyłączam backend powermetrics (brak uprawnień lub inny błąd).")
        _MAC_MONITOR_DISABLED = True
        return None

    _MAC_MONITOR = monitor
    return _MAC_MONITOR


def _read_mac_energy_j() -> Optional[float]:
    mon = _get_mac_monitor()
    if mon is None:
        return None
    return mon.get_energy_j()


# ============================================================
#  Publiczne API
# ============================================================

def energy_measurement_supported() -> bool:
    """
    Czy *jakikolwiek* backend pomiaru energii CPU jest dostępny?

    - Linux/x86: True, jeśli jest RAPL (intel-rapl:0 / amd-rapl:0).
    - macOS: True, jeśli dostępny jest powermetrics i backend nie
      został wyłączony z powodu błędu (np. brak sudo).
    """
    if _detect_rapl_file() is not None:
        return True

    if platform.system() == "Darwin" and not _MAC_MONITOR_DISABLED:
        return shutil.which("powermetrics") is not None

    return False


def read_energy_joules() -> Optional[float]:
    """
    Zwraca bieżący licznik energii [J] dla backendu:

    - Linux: odczyt z RAPL (intel/amd) -> energy_uj.
    - macOS: energia zintegrowana z powermetrics od momentu startu
      monitora powermetrics.

    Użycie w benchmarkach:
        e_before = read_energy_joules()
        ... uruchom kernel ...
        e_after = read_energy_joules()
        if e_before is not None and e_after is not None:
            energy_run = max(0.0, e_after - e_before)

    Jeśli pomiar jest niedostępny -> None.
    """
    # Linux / RAPL
    e = _read_rapl_energy_j()
    if e is not None:
        return e

    # macOS / powermetrics
    if platform.system() == "Darwin":
        return _read_mac_energy_j()

    # Inne platformy – brak wsparcia
    return None
