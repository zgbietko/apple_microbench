# apple_microbench/gpu/cuda/benchmarks/run_cuda_bandwidth.py

import os
import sys
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from gpu.cuda.cuda_backend import (
    init_cuda,
    cuda_memcpy_benchmark,
    get_device_name,
)
from gpu.gpu_utils import (
    ensure_results_dir,
    common_gpu_metadata,
    make_gpu_specific_csv_path,
)
from energy_utils import EnergyLogger
from energy import energy_measurement_supported

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "gpu"


def run_bandwidth_bench(
    device_index: int,
    runs_per_size: int,
    sizes_mb: List[int],
) -> None:
    """
    Uruchamia benchmark przepustowości pamięci dla wskazanego urządzenia CUDA.

    Mierzony jest transfer host->device->host (memcpy) dla bloków o różnych rozmiarach.
    Wyniki zapisywane są do CSV specyficznego dla danej karty GPU.
    """
    ctx = init_cuda(device_index)
    gpu_name = get_device_name(ctx)
    print(f"=== CUDA GPU memory bandwidth benchmark (memcpy H->D->H) ===")
    print(f"GPU device   : {gpu_name} (index {device_index})")
    print(f"runs per size: {runs_per_size}")

    ensure_results_dir(DATA_DIR)

    csv_path = make_gpu_specific_csv_path(
        DATA_DIR, "cuda_bandwidth", backend="cuda", gpu_name=gpu_name
    )

    # przygotowanie CSV
    import csv

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "backend",
                "gpu_name",
                "device_index",
                "size_bytes",
                "size_mb",
                "run_id",
                "elapsed_s",
                "gb_per_s",
                "energy_j",
                "power_w",
            ],
        )
        if write_header:
            writer.writeheader()

        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            n_elems = size_bytes // 4  # float32

            print(
                f"\n--- Size: {size_mb:5d} MB ({size_bytes} bytes, {n_elems} elements) ---"
            )

            # alokacja i inicjalizacja
            host_buf = np.random.rand(n_elems).astype(np.float32)

            # przygotowanie loggera energii
            energy_enabled = energy_measurement_supported()
            energy_logger = EnergyLogger() if energy_enabled else None

            times: List[float] = []
            energies: List[float] = []
            avg_powers: List[float] = []

            for run_id in range(runs_per_size):
                if energy_logger is not None:
                    energy_logger.start()

                t0 = time.perf_counter()
                cuda_memcpy_benchmark(ctx, host_buf)
                t1 = time.perf_counter()

                if energy_logger is not None:
                    energy_j, power_w = energy_logger.stop()
                else:
                    energy_j, power_w = math.nan, math.nan

                elapsed = t1 - t0
                gb_s = (2.0 * size_bytes) / (elapsed * (1024**3))

                times.append(elapsed)
                energies.append(energy_j)
                avg_powers.append(power_w)

                print(
                    f"run {run_id:2d}: elapsed = {elapsed:8.4f} s, "
                    f"GB/s = {gb_s:7.2f}, energy = {energy_j:8.3f} J, "
                    f"P_avg = {power_w:8.3f} W"
                )

                writer.writerow(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "backend": "cuda",
                        "gpu_name": gpu_name,
                        "device_index": device_index,
                        "size_bytes": size_bytes,
                        "size_mb": size_mb,
                        "run_id": run_id,
                        "elapsed_s": elapsed,
                        "gb_per_s": gb_s,
                        "energy_j": energy_j,
                        "power_w": power_w,
                    }
                )

            mean_gb_s = float(np.mean(times and [(2.0 * size_bytes) / (t * (1024**3)) for t in times]))
            std_gb_s = float(np.std(times and [(2.0 * size_bytes) / (t * (1024**3)) for t in times]))
            print(f"==> MEAN: {mean_gb_s:7.2f} GB/s, sigma = {std_gb_s:7.2f} GB/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA GPU memory bandwidth benchmark (H->D->H memcpy)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Index urządzenia CUDA (domyślnie 0)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Liczba powtórzeń dla każdego rozmiaru (domyślnie 7)",
    )
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=[4, 16, 64, 256, 1024],
        help="Lista rozmiarów bloków (w MB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bandwidth_bench(
        device_index=args.device,
        runs_per_size=args.runs,
        sizes_mb=args.sizes_mb,
    )


if __name__ == "__main__":
    main()
