# apple_microbench/run_all_gpu_benchmarks.py

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent


def run_script(script_rel_path: str, extra_args: Optional[List[str]] = None) -> bool:
    """
    Uruchamia podany skrypt Pythona (ścieżka względna względem ROOT)
    i zwraca True, jeśli zakończył się kodem 0.
    """
    script_path = ROOT / script_rel_path
    if not script_path.exists():
        print(f"[WARN] Pomijam {script_rel_path}, plik nie istnieje.")
        return True  # traktujemy jako "nie ma co robić", a nie błąd

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n=== Uruchamiam: {script_rel_path} ===")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[ERROR] Skrypt {script_rel_path} zakończył się kodem {result.returncode}")
            return False
        return True
    except FileNotFoundError:
        print(f"[ERROR] Nie udało się uruchomić {script_rel_path} (brak Pythona?).")
        return False


def list_cuda_devices_via_helper() -> None:
    """
    Opcjonalny helper: próbuje uruchomić mały skrypt, który wypisze listę urządzeń CUDA.
    Jeśli helpera nie ma – po prostu nic nie robi.
    """
    helper = ROOT / "gpu" / "cuda" / "list_cuda_devices.py"
    if not helper.exists():
        return
    print("\n=== Dostępne urządzenia CUDA (list_cuda_devices.py) ===")
    cmd = [sys.executable, str(helper)]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[WARN] Nie udało się uruchomić list_cuda_devices.py: {e}")


def run_metal_benchmarks() -> bool:
    """
    Uruchamia zestaw benchmarków GPU dla Metal (macOS, Apple Silicon).
    """
    ok = True

    # 1) Przepustowość pamięci (GPU, Metal)
    ok = ok and run_script("gpu/metal/benchmarks/run_metal_bandwidth.py")

    # 2) FMA throughput (single-thread na GPU)
    ok = ok and run_script("gpu/metal/benchmarks/run_metal_compute_fma.py")

    # 3) Peak FMA (maksymalne obciążenie GPU)
    ok = ok and run_script("gpu/metal/benchmarks/run_metal_compute_fma_peak.py")

    return ok


def run_cuda_benchmarks(device_index: int = 0) -> bool:
    """
    Uruchamia zestaw benchmarków GPU dla CUDA (NVIDIA).
    Używa poprawnych ścieżek:
      - gpu/cuda/benchmarks/run_cuda_bandwidth.py
      - gpu/cuda/benchmarks/run_cuda_compute_fma.py
      - gpu/cuda/benchmarks/run_cuda_compute_fma_peak.py
    """
    ok = True

    # Najpierw spróbujmy wypisać listę urządzeń CUDA (jeśli helper istnieje)
    list_cuda_devices_via_helper()

    cuda_args = ["--device", str(device_index)]

    # 1) Przepustowość pamięci (GPU, CUDA)
    ok = ok and run_script(
        "gpu/cuda/benchmarks/run_cuda_bandwidth.py",
        extra_args=cuda_args,
    )

    # 2) FMA throughput (single-thread / 1 kernel na GPU)
    ok = ok and run_script(
        "gpu/cuda/benchmarks/run_cuda_compute_fma.py",
        extra_args=cuda_args,
    )

    # 3) Peak FMA (maksymalne obciążenie GPU – wiele bloków / wątków)
    ok = ok and run_script(
        "gpu/cuda/benchmarks/run_cuda_compute_fma_peak.py",
        extra_args=cuda_args,
    )

    return ok


def main() -> None:
    """
    Wybiera backend GPU na podstawie systemu operacyjnego:
      - macOS (Darwin)  -> Metal
      - Linux / Windows -> CUDA (zakładamy, że jeśli jest GPU, to będzie to CUDA)
    W przyszłości można dodać wybór backendu przez argumenty CLI.
    """
    system = platform.system()
    print("=== Uruchamianie benchmarków GPU ===")
    print(f"[INFO] Wykryty system: {system}")

    if system == "Darwin":
        print("[INFO] Wykryto macOS -> uruchamiam benchmarki Metal.")
        ok = run_metal_benchmarks()
    else:
        print("[INFO] Zakładam środowisko z CUDA -> uruchamiam benchmarki CUDA (GPU NVIDIA).")
        # Domyślnie device_index = 0; w razie potrzeby można dodać CLI do wyboru GPU
        ok = run_cuda_benchmarks(device_index=0)

    if not ok:
        print("\n[ERROR] Co najmniej jeden benchmark GPU zakończył się błędem.")
        sys.exit(1)
    else:
        print("\n[OK] Wszystkie benchmarki GPU zakończone pomyślnie (lub zostały pominięte).")


if __name__ == "__main__":
    main()
