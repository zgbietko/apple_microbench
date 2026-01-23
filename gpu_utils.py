from __future__ import annotations

import ctypes as ct
import os
import platform
import re
import socket
from pathlib import Path
from typing import Dict, List, Tuple

# ROOT katalogu projektu (tam, gdzie jest ten plik: apple_microbench/)
ROOT = Path(__file__).resolve().parent


# ============================================================
#  Ogólne narzędzia GPU (slug, CSV, metadane)
# ============================================================

def slugify_gpu_model(name: str) -> str:
    """
    Konwersja nazwy GPU na slug przyjazny dla systemu plików.
    Przykład: "Apple M2 Pro" -> "apple_m2_pro"
    """
    if not name:
        return "unknown_gpu"
    name = name.strip().lower()
    # Zamiana znaków nie-alfanumerycznych na podkreślenia
    name = re.sub(r"[^a-z0-9]+", "_", name)
    # Redukcja wielu podkreśleń i obcięcie z brzegów
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown_gpu"


def make_gpu_csv_path(root: Path, bench_name: str, backend: str, gpu_model: str) -> Path:
    """
    Buduje ścieżkę:
      data/gpu/{bench_name}_{backend}_{gpu_slug}.csv

    root – katalog projektu (np. ROOT = Path(__file__).resolve().parent)
    """
    slug = slugify_gpu_model(gpu_model)
    return root / "data" / "gpu" / f"{bench_name}_{backend}_{slug}.csv"


def common_gpu_metadata(backend: str, gpu_model: str, gpu_index: int) -> Dict[str, str]:
    """
    Zestaw wspólnych metadanych zapisywanych razem z wynikami (dla GPU).
    """
    return {
        "backend": backend,
        "gpu_model": gpu_model,
        "gpu_index": str(gpu_index),
        "system": platform.system(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }


# ============================================================
#  Pomocnicze narzędzia CUDA (do użycia na maszynach z NVIDIA)
#  – możesz z nich korzystać w run_gpu_*.py
# ============================================================

def load_cuda_library() -> Tuple[ct.CDLL, Path]:
    """
    Ładuje libgpubench.* z gpu/lib. Zwraca (lib, lib_path).
    Jeśli biblioteka nie istnieje, rzuca FileNotFoundError.
    """
    system = platform.system()
    lib_dir = ROOT / "gpu" / "lib"

    if system == "Darwin":
        candidates = ["libgpubench.dylib", "gpubench.dylib"]
    elif system == "Linux":
        candidates = ["libgpubench.so", "gpubench.so"]
    else:
        # Windows
        candidates = ["gpubench.dll"]

    for name in candidates:
        lib_path = lib_dir / name
        if lib_path.exists():
            lib = ct.CDLL(str(lib_path))
            return lib, lib_path

    raise FileNotFoundError(f"Nie znaleziono libgpubench w {lib_dir}")


def configure_cuda_functions(lib: ct.CDLL) -> None:
    """
    Ustawia argtypes / restype dla funkcji z gpubench.cu.
    """
    if not hasattr(lib, "gpu_get_device_count"):
        # Możliwe, że biblioteka nie jest biblioteką CUDA – wtedy nic nie robimy.
        return

    lib.gpu_get_device_count.argtypes = []
    lib.gpu_get_device_count.restype = ct.c_int

    lib.gpu_get_device_name.argtypes = [ct.c_int, ct.c_char_p, ct.c_int]
    lib.gpu_get_device_name.restype = ct.c_int

    lib.gpu_mem_copy_elapsed.argtypes = [ct.c_size_t, ct.c_int, ct.c_int]
    lib.gpu_mem_copy_elapsed.restype = ct.c_double

    lib.gpu_fma_elapsed.argtypes = [ct.c_size_t, ct.c_int, ct.c_int]
    lib.gpu_fma_elapsed.restype = ct.c_double


def list_cuda_devices(lib: ct.CDLL) -> List[Tuple[int, str]]:
    """
    Zwraca listę (device_id, device_name) dla urządzeń CUDA.
    """
    devices: List[Tuple[int, str]] = []

    if not hasattr(lib, "gpu_get_device_count"):
        return devices

    count = lib.gpu_get_device_count()
    for dev_id in range(count):
        buf = ct.create_string_buffer(256)
        rc = lib.gpu_get_device_name(dev_id, buf, ct.sizeof(buf))
        if rc == 0:
            name = buf.value.decode("utf-8", errors="replace")
        else:
            name = f"device_{dev_id}"
        devices.append((dev_id, name))

    return devices


def select_cuda_device(lib: ct.CDLL, preferred_index: int | None = None) -> Tuple[int, str]:
    """
    Wybiera urządzenie CUDA, uwzględniając:
      - preferred_index (np. z argumentu --device),
      - zmienną środowiskową GPU_DEVICE_ID,
      - domyślnie device 0.

    Zwraca (device_id, device_name).
    """
    devices = list_cuda_devices(lib)
    if not devices:
        raise RuntimeError("Brak dostępnych urządzeń CUDA (gpu_get_device_count() == 0).")

    # 1) argument --device
    if preferred_index is not None:
        if 0 <= preferred_index < len(devices):
            return devices[preferred_index]
        else:
            raise ValueError(
                f"Nieprawidłowy device index {preferred_index}, dostępne: 0..{len(devices)-1}"
            )

    # 2) zmienna środowiskowa
    env_val = os.environ.get("GPU_DEVICE_ID")
    if env_val is not None:
        try:
            dev_id = int(env_val)
            if 0 <= dev_id < len(devices):
                return devices[dev_id]
        except ValueError:
            pass  # ignorujemy błędną wartość

    # 3) jeśli tylko jedno urządzenie – wybieramy je
    if len(devices) == 1:
        return devices[0]

    # 4) wiele urządzeń, brak wskazania – domyślnie 0, ale wypisz listę
    print("Dostępne urządzenia CUDA:")
    for dev_id, name in devices:
        print(f"  [{dev_id}] {name}")
    print("Brak parametru --device / GPU_DEVICE_ID, używam domyślnie device 0.")

    return devices[0]


def make_gpu_specific_csv_path(
    benchmark_name: str,
    data_dir: Path,
    gpu_backend: str,
    gpu_name: str,
    device_id: int,
) -> Path:
    """
    Tworzy nazwę pliku CSV specyficzną dla benchmarku i modelu karty dla CUDA.
    Np. gpu_bandwidth_cuda_nvidia_rtx_4090_dev0.csv
    """
    backend_slug = slugify_gpu_model(gpu_backend)
    gpu_slug = slugify_gpu_model(gpu_name)
    filename = f"{benchmark_name}_{backend_slug}_{gpu_slug}_dev{device_id}.csv"
    return data_dir / filename
