# cpu/benchmarks/run_bandwidth_mt.py
import ctypes as ct
import time
import csv
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import statistics as stats

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # .../apple_microbench
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from energy import energy_measurement_supported, read_energy_joules
from cpu_utils import make_cpu_specific_csv_path


def load_library():
    """Ładuje libmicrobench.* z cpu/lib – cross-platform."""
    system = platform.system()
    root = ROOT

    if system == "Darwin":
        lib_name = "libmicrobench.dylib"
    elif system == "Linux":
        lib_name = "libmicrobench.so"
    elif system == "Windows":
        lib_name = "microbench.dll"
    else:
        raise RuntimeError(f"Nieobsługiwany system: {system}")

    lib_path = root / "cpu" / "lib" / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(f"Nie znaleziono biblioteki: {lib_path}")

    return ct.CDLL(str(lib_path)), root


def configure_functions(lib):
    func = lib.mem_copy_kernel_mt
    func.argtypes = [
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.c_size_t,
        ct.c_int,
    ]
    func.restype = None
    return func


def detect_cpu_model() -> str:
    """
    Próbuje wykryć pełny model CPU (Apple M2 Pro, Intel(R) Core..., itd.).
    """
    system = platform.system()

    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out

        elif system == "Linux":
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()

        elif system == "Windows":
            return platform.processor() or platform.uname().processor
    except Exception:
        pass

    return platform.processor() or platform.machine()


def bench_mem_copy_mt(mem_copy_kernel_mt, bytes_per_iter: int, iters: int, num_threads: int):
    n = bytes_per_iter // 4  # float32

    src = np.random.rand(n).astype(np.float32)
    dst = np.empty_like(src)

    src_p = src.ctypes.data_as(ct.POINTER(ct.c_float))
    dst_p = dst.ctypes.data_as(ct.POINTER(ct.c_float))

    mem_copy_kernel_mt(dst_p, src_p, n, num_threads)

    energy_j = None
    e_before = None
    if energy_measurement_supported():
        e_before = read_energy_joules()

    t0 = time.perf_counter()
    for _ in range(iters):
        mem_copy_kernel_mt(dst_p, src_p, n, num_threads)
    t1 = time.perf_counter()

    if e_before is not None:
        e_after = read_energy_joules()
        if e_after is not None:
            delta = e_after - e_before
            if delta >= 0:
                energy_j = delta

    elapsed = t1 - t0
    total_bytes = bytes_per_iter * iters
    gbps = (total_bytes / elapsed) / (1024**3)

    avg_power_w = None
    if energy_j is not None and elapsed > 0:
        avg_power_w = energy_j / elapsed

    return {
        "size_bytes": bytes_per_iter,
        "bytes_per_iter": bytes_per_iter,
        "iters": iters,
        "threads": num_threads,
        "num_threads": num_threads,
        "elapsed_s": elapsed,
        "gbps": gbps,
        "energy_j": energy_j,
        "power_w": avg_power_w,
        "avg_power_w": avg_power_w,
    }


def collect_system_metadata():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": detect_cpu_model(),
        "python_version": platform.python_version(),
    }


def write_result_to_csv(
    csv_path: Path,
    result: dict,
    meta: dict,
    benchmark_name: str,
    write_header: bool,
):
    row = {
        **meta,
        "benchmark": benchmark_name,
        **result,
    }

    fieldnames = [
        "timestamp",
        "system",
        "node",
        "release",
        "version",
        "machine",
        "arch",
        "processor",
        "cpu_model",
        "python_version",
        "benchmark",
        "run_id",
        "size_bytes",
        "bytes_per_iter",
        "iters",
        "threads",
        "num_threads",
        "elapsed_s",
        "gbps",
        "energy_j",
        "power_w",
        "avg_power_w",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    lib, root = load_library()
    mem_copy_kernel_mt = configure_functions(lib)

    data_dir = root / "data" / "cpu"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path, arch, cpu_model, cpu_slug = make_cpu_specific_csv_path(
        "bandwidth_mt",
        data_dir,
    )
    if csv_path.exists():
        csv_path.unlink()  # nadpisujemy stare wyniki

    meta = collect_system_metadata()
    meta["arch"] = arch
    meta["cpu_model"] = cpu_model

    sizes_mb = [64, 256, 1024]
    threads_list = [1, 2, 4, 8]
    iters = 20
    runs = 5

    print("=== CPU memory bandwidth benchmark (mem_copy_kernel_mt, multi-thread) ===")
    print(f"runs per config : {runs}")
    print(f"iters per run   : {iters}")
    print(f"CPU model       : {meta['cpu_model']}")
    if energy_measurement_supported():
        print("Energy          : Linux RAPL, per-run energy_j / avg_power_w")
    else:
        print("Energy          : pomiar niedostępny na tej platformie")

    header_written = False

    for size_mb in sizes_mb:
        bytes_per_iter = size_mb * 1024 * 1024

        for num_threads in threads_list:
            print(f"\n--- Size: {size_mb} MB, threads: {num_threads} ---")
            gbps_values = []
            energy_values = []

            for run_id in range(runs):
                result = bench_mem_copy_mt(mem_copy_kernel_mt, bytes_per_iter, iters, num_threads)
                gbps_values.append(result["gbps"])
                if result["energy_j"] is not None:
                    energy_values.append(result["energy_j"])

                write_result_to_csv(
                    csv_path,
                    {**result, "run_id": run_id},
                    meta,
                    "mem_copy_MT",
                    write_header=not header_written,
                )
                header_written = True

                print(
                    f"run {run_id:2d}: elapsed = {result['elapsed_s']:.4f} s, "
                    f"GB/s = {result['gbps']:.2f}, "
                    f"energy = {result['energy_j'] if result['energy_j'] is not None else float('nan'):.4f} J"
                )

            mean_gbps = stats.mean(gbps_values)
            stdev_gbps = stats.pstdev(gbps_values) if len(gbps_values) > 1 else 0.0
            print(f"==> ŚREDNIA: {mean_gbps:.2f} GB/s, σ = {stdev_gbps:.2f} GB/s")

            if energy_values:
                mean_energy = stats.mean(energy_values)
                stdev_energy = stats.pstdev(energy_values) if len(energy_values) > 1 else 0.0
                print(f"    ŚREDNIA energia per run: {mean_energy:.4f} J, σ = {stdev_energy:.4f} J")

    print(f"\nWszystkie runy zapisane do: {csv_path}")


if __name__ == "__main__":
    main()
