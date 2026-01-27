from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CudaDeviceInfo:
    index: int
    name: str
    cc_major: int
    cc_minor: int
    global_mem_bytes: int

    @property
    def compute_capability(self) -> str:
        return f"{self.cc_major}.{self.cc_minor}"

    @property
    def global_mem_gb(self) -> float:
        return self.global_mem_bytes / (1024 ** 3)


@dataclass
class CudaContext:
    device_index: int
    lib: ctypes.CDLL
    info: CudaDeviceInfo


_LIB_CACHE: Optional[ctypes.CDLL] = None


def _find_library_path() -> Path:
    here = Path(__file__).resolve().parent
    lib_dir = here / "lib"

    system = platform.system()
    candidates = []
    if system == "Darwin":
        candidates.append(lib_dir / "libgpubench_cuda.dylib")
    elif system == "Linux":
        candidates.append(lib_dir / "libgpubench_cuda.so")
    elif system == "Windows":
        candidates.append(lib_dir / "gpubench_cuda.dll")
    else:
        candidates.append(lib_dir / "libgpubench_cuda.so")

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"Nie znaleziono biblioteki CUDA gpubench w {lib_dir}. "
        f"Upewnij się, że uruchomiłeś gpu/cuda/lib/build_cuda.sh."
    )


def _load_lib() -> ctypes.CDLL:
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    lib_path = _find_library_path()
    lib = ctypes.cdll.LoadLibrary(str(lib_path))

    # sygnatury funkcji C z gpubench.cu
    lib.gpu_cuda_get_device_count.restype = ctypes.c_int

    lib.gpu_cuda_get_device_name.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int
    ]
    lib.gpu_cuda_get_device_name.restype = ctypes.c_int

    lib.gpu_cuda_get_device_props.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.gpu_cuda_get_device_props.restype = ctypes.c_int

    lib.gpu_cuda_memcpy_bandwidth.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.gpu_cuda_memcpy_bandwidth.restype = ctypes.c_int

    lib.gpu_cuda_fma_throughput.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.gpu_cuda_fma_throughput.restype = ctypes.c_int

    _LIB_CACHE = lib
    return lib


def get_device_count() -> int:
    lib = _load_lib()
    count = lib.gpu_cuda_get_device_count()
    if count < 0:
        raise RuntimeError(f"gpu_cuda_get_device_count failed with code {count}")
    return count


def _get_raw_device_info(device_index: int) -> CudaDeviceInfo:
    lib = _load_lib()

    # nazwa
    buf_len = 256
    name_buf = ctypes.create_string_buffer(buf_len)
    err = lib.gpu_cuda_get_device_name(device_index, name_buf, buf_len)
    if err != 0:
        raise RuntimeError(f"gpu_cuda_get_device_name failed with code {err}")
    name = name_buf.value.decode("utf-8")

    # CC + pamięć
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    global_mem = ctypes.c_size_t()
    err = lib.gpu_cuda_get_device_props(
        device_index,
        ctypes.byref(cc_major),
        ctypes.byref(cc_minor),
        ctypes.byref(global_mem),
    )
    if err != 0:
        raise RuntimeError(f"gpu_cuda_get_device_props failed with code {err}")

    return CudaDeviceInfo(
        index=device_index,
        name=name,
        cc_major=cc_major.value,
        cc_minor=cc_minor.value,
        global_mem_bytes=global_mem.value,
    )


def init_cuda(device_index: int = 0) -> CudaContext:
    """
    Inicjalizuje backend dla danego urządzenia CUDA i zwraca CudaContext.
    """
    lib = _load_lib()
    count = get_device_count()
    if device_index < 0 or device_index >= count:
        raise ValueError(
            f"Nieprawidłowy indeks urządzenia {device_index}; dostępne 0..{count-1}"
        )
    info = _get_raw_device_info(device_index)
    return CudaContext(device_index=device_index, lib=lib, info=info)


def get_device_info(ctx: CudaContext) -> CudaDeviceInfo:
    return ctx.info


def get_device_name(ctx: CudaContext) -> str:
    return ctx.info.name


def cuda_memcpy_bandwidth(ctx: CudaContext, size_bytes: int, iters: int) -> float:
    """
    Wywołuje gpu_cuda_memcpy_bandwidth i zwraca czas [s] (łączny dla `iters` iteracji).
    """
    elapsed_ms = ctypes.c_double()
    err = ctx.lib.gpu_cuda_memcpy_bandwidth(
        ctypes.c_int(ctx.device_index),
        ctypes.c_size_t(size_bytes),
        ctypes.c_int(iters),
        ctypes.byref(elapsed_ms),
    )
    if err != 0:
        raise RuntimeError(f"gpu_cuda_memcpy_bandwidth failed with code {err}")
    return elapsed_ms.value / 1e3  # sekundy


def cuda_fma_throughput(ctx: CudaContext, n: int, iters_inner: int) -> float:
    """
    Wywołuje gpu_cuda_fma_throughput i zwraca czas [s] (łączny dla iters_inner iteracji).
    """
    elapsed_ms = ctypes.c_double()
    err = ctx.lib.gpu_cuda_fma_throughput(
        ctypes.c_int(ctx.device_index),
        ctypes.c_size_t(int(n)),
        ctypes.c_int(iters_inner),
        ctypes.byref(elapsed_ms),
    )
    if err != 0:
        raise RuntimeError(f"gpu_cuda_fma_throughput failed with code {err}")
    return elapsed_ms.value / 1e3  # sekundy


def _demo() -> None:
    count = get_device_count()
    print(f"CUDA devices: {count}")
    for i in range(count):
        ctx = init_cuda(i)
        info = ctx.info
        print(
            f"[{i}] {info.name} | CC {info.compute_capability}, "
            f"{info.global_mem_gb:.1f} GiB"
        )


if __name__ == "__main__":
    _demo()
