from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(rel_path: str, extra_args: list[str] | None = None) -> bool:
    """
    Uruchamia skrypt Pythona o ścieżce względnej rel_path z katalogu ROOT.
    Zwraca True, jeśli zakończył się kodem 0, w przeciwnym wypadku False.
    """
    script_path = ROOT / rel_path
    if not script_path.exists():
        print(f"[WARN] Pomijam {rel_path}, plik nie istnieje.")
        return True  # brak pliku nie przerywa całej serii

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n=== Uruchamiam: {rel_path} ===")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Skrypt {rel_path} zakończył się kodem {result.returncode}")
        return False
    return True


def run_metal_benchmarks() -> None:
    """
    Uruchamia benchmarki GPU oparte na Metal (macOS / Apple Silicon).
    """
    print("[INFO] Wykryto platformę macOS – używam backendu Metal.")
    device_index = 0  # na typowym M1/M2/M3 jest jedno urządzenie GPU

    ok = True
    ok &= run_script(
        "gpu/metal/benchmarks/run_metal_bandwidth.py",
        ["--device-index", str(device_index)],
    )
    ok &= run_script(
        "gpu/metal/benchmarks/run_metal_compute_fma.py",
        ["--device-index", str(device_index)],
    )
    ok &= run_script(
        "gpu/metal/benchmarks/run_metal_compute_fma_peak.py",
        ["--device-index", str(device_index)],
    )

    if ok:
        print("\n[INFO] Wszystkie benchmarki GPU (Metal) zakończone pomyślnie.")
    else:
        print("\n[ERROR] Co najmniej jeden benchmark GPU (Metal) zakończył się błędem.")


def run_cuda_benchmarks() -> None:
    """
    (Na przyszłość) Uruchamia benchmarki GPU oparte na CUDA.
    Na macOS tego nie używamy, ale zostawiam szkielet pod Linuksa/Windows.
    """
    print("[INFO] Próbuję użyć backendu CUDA (nie dotyczy macOS/Metal).")

    try:
        from gpu_utils import (
            load_cuda_library,
            configure_cuda_functions,
            select_cuda_device,
        )
    except ImportError as e:
        print(f"[ERROR] Nie można zaimportować gpu_utils ({e}) – brak wsparcia CUDA.")
        return

    try:
        lib, lib_path = load_cuda_library()
        configure_cuda_functions(lib)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    except OSError as e:
        print(f"[ERROR] Problem z załadowaniem biblioteki CUDA: {e}")
        return

    try:
        dev_id, dev_name = select_cuda_device(lib, preferred_index=None)
    except Exception as e:
        print(f"[ERROR] Nie udało się wybrać urządzenia CUDA: {e}")
        return

    print(f"[INFO] Używam urządzenia CUDA [{dev_id}]: {dev_name}")

    ok = True
    ok &= run_script(
        "gpu/cuda/benchmarks/run_cuda_bandwidth.py",
        ["--device", str(dev_id)],
    )
    ok &= run_script(
        "gpu/cuda/benchmarks/run_cuda_compute_fma.py",
        ["--device", str(dev_id)],
    )

    if ok:
        print("\n[INFO] Wszystkie benchmarki GPU (CUDA) zakończone pomyślnie.")
    else:
        print("\n[ERROR] Co najmniej jeden benchmark GPU (CUDA) zakończył się błędem.")


def main() -> None:
    system = platform.system()
    print(f"[INFO] Wykryto system: {system}")

    if system == "Darwin":
        # macOS – używamy Metal
        run_metal_benchmarks()
    else:
        # Linux / Windows – próbujemy CUDA
        run_cuda_benchmarks()


if __name__ == "__main__":
    main()
