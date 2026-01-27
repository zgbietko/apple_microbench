from __future__ import annotations

import ctypes
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


class CudaBackend:
    """ctypes-based wrapper around the CUDA microbenchmark library.

    The library is built from gpu/cuda/lib/gpubench.cu and is expected to
    export the following symbols:

        int  gpu_cuda_get_device_count(void);
        int  gpu_cuda_get_device_name(int device_index, char* buf, int buf_len);
        int  gpu_cuda_get_device_props(int device_index,
                                       int* cc_major,
                                       int* cc_minor,
                                       size_t* global_mem_bytes);
        int  gpu_cuda_memcpy_bandwidth(int device_index,
                                       size_t bytes,
                                       int iters,
                                       double* elapsed_ms_out);
        int  gpu_cuda_fma_throughput(int device_index,
                                     size_t n,
                                     int iters_inner,
                                     double* elapsed_ms_out);
    """

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        self._lib = self._load_library(lib_path)
        self._configure_signatures()

    # ------------------------------------------------------------------
    # Low-level loading
    # ------------------------------------------------------------------
    @staticmethod
    def _default_library_candidates(base_dir: Path) -> list[Path]:
        lib_dir = base_dir / "lib"
        return [
            lib_dir / "libgpubench_cuda.so",
            lib_dir / "libgpubench_cuda.dylib",
            lib_dir / "gpubench_cuda.dll",
        ]

    def _load_library(self, lib_path: Optional[Path]) -> ctypes.CDLL:
        if lib_path is None:
            # Assume this file lives in apple_microbench/gpu/cuda/cuda_backend.py
            base_dir = Path(__file__).resolve().parent
            candidates = self._default_library_candidates(base_dir)
        else:
            candidates = [Path(lib_path)]

        for cand in candidates:
            if cand.is_file():
                return ctypes.CDLL(str(cand))

        raise RuntimeError(
            "CUDA microbenchmark library not found. "
            "Build it with gpu/cuda/lib/build_cuda.sh first."
        )

    def _configure_signatures(self) -> None:
        lib = self._lib

        # int gpu_cuda_get_device_count(void)
        lib.gpu_cuda_get_device_count.argtypes = []
        lib.gpu_cuda_get_device_count.restype = ctypes.c_int

        # int gpu_cuda_get_device_name(int device_index, char* buf, int buf_len)
        lib.gpu_cuda_get_device_name.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.gpu_cuda_get_device_name.restype = ctypes.c_int

        # int gpu_cuda_get_device_props(int device_index,
        #                               int* cc_major,
        #                               int* cc_minor,
        #                               size_t* global_mem_bytes)
        lib.gpu_cuda_get_device_props.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.gpu_cuda_get_device_props.restype = ctypes.c_int

        # int gpu_cuda_memcpy_bandwidth(int device_index,
        #                               size_t bytes,
        #                               int iters,
        #                               double* elapsed_ms_out)
        lib.gpu_cuda_memcpy_bandwidth.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.gpu_cuda_memcpy_bandwidth.restype = ctypes.c_int

        # int gpu_cuda_fma_throughput(int device_index,
        #                             size_t n,
        #                             int iters_inner,
        #                             double* elapsed_ms_out)
        lib.gpu_cuda_fma_throughput.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.gpu_cuda_fma_throughput.restype = ctypes.c_int

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def device_count(self) -> int:
        count = self._lib.gpu_cuda_get_device_count()
        if count < 0:
            raise RuntimeError(f"gpu_cuda_get_device_count failed with code {count}")
        return count

    def get_device_info(self, index: int) -> CudaDeviceInfo:
        buf_size = 256
        buf = ctypes.create_string_buffer(buf_size)

        rc_name = self._lib.gpu_cuda_get_device_name(index, buf, buf_size)
        if rc_name != 0:
            raise RuntimeError(
                f"gpu_cuda_get_device_name({index}) failed with code {rc_name}"
            )
        name = buf.value.decode("utf-8", errors="replace")

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        global_mem = ctypes.c_size_t()

        rc_props = self._lib.gpu_cuda_get_device_props(
            index,
            ctypes.byref(cc_major),
            ctypes.byref(cc_minor),
            ctypes.byref(global_mem),
        )
        if rc_props != 0:
            raise RuntimeError(
                f"gpu_cuda_get_device_props({index}) failed with code {rc_props}"
            )

        return CudaDeviceInfo(
            index=index,
            name=name,
            cc_major=cc_major.value,
            cc_minor=cc_minor.value,
            global_mem_bytes=int(global_mem.value),
        )

    # Returns elapsed kernel time in seconds.
    def memcpy_bandwidth_seconds(
        self, device_index: int, num_bytes: int, iters: int
    ) -> float:
        elapsed_ms = ctypes.c_double()
        rc = self._lib.gpu_cuda_memcpy_bandwidth(
            ctypes.c_int(device_index),
            ctypes.c_size_t(num_bytes),
            ctypes.c_int(iters),
            ctypes.byref(elapsed_ms),
        )
        if rc != 0:
            raise RuntimeError(
                f"gpu_cuda_memcpy_bandwidth failed (device={device_index}, "
                f"bytes={num_bytes}, iters={iters}) with code {rc}"
            )
        return float(elapsed_ms.value) / 1000.0

    # Returns elapsed kernel time in seconds.
    def fma_throughput_seconds(
        self, device_index: int, n_elements: int, iters_inner: int
    ) -> float:
        elapsed_ms = ctypes.c_double()
        rc = self._lib.gpu_cuda_fma_throughput(
            ctypes.c_int(device_index),
            ctypes.c_size_t(n_elements),
            ctypes.c_int(iters_inner),
            ctypes.byref(elapsed_ms),
        )
        if rc != 0:
            raise RuntimeError(
                f"gpu_cuda_fma_throughput failed (device={device_index}, "
                f"n={n_elements}, iters_inner={iters_inner}) with code {rc}"
            )
        return float(elapsed_ms.value) / 1000.0


def _demo() -> None:
    backend = CudaBackend()
    count = backend.device_count()
    print(f"CUDA devices: {count}")
    for i in range(count):
        info = backend.get_device_info(i)
        print(
            f"[{i}] {info.name} | CC {info.compute_capability}, "
            f"{info.global_mem_gb:.1f} GiB"
        )


if __name__ == "__main__":
    _demo()
