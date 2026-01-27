# apple_microbench/gpu/cuda/benchmarks/run_cuda_compute_fma.py

import os
import sys
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from gpu.cuda.cuda_backend import (
    init_cuda,
    cuda_fma_kernel,
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


def run_fma_bench(
    device_index: int,
    vector_len: int,
    iters_list: List[int],
    runs_per_config: int,
) -> None:
    """
    Benchmark wydajności obliczeń FMA na GPU (CUDA) dla jednego wektora
    i różnych wartości iters_inner.
    """
    ctx = init_cuda(device_index)
    gpu_name = get_device_name(ctx)

    print("=== CUDA GPU FMA compute benchmark ===")
    print(f"GPU device    : {gpu_name} (index {device_index})")
    print(f"vector_len    : {vector_len}")
    print(f"runs per iters: {runs_per_config}")

    ensure_results_dir(DATA_DIR)
    csv_path = make_gpu_specific_csv_path(
        DATA_DIR, "cuda_compute_fma", backend="cuda", gpu_name=gpu_name
    )

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
                "vector_len",
                "iters_inner",
                "run_id",
                "elapsed_s",
                "gflops",
                "energy_j",
                "power_w",
            ],
        )
        if write_header:
            writer.writeheader()

        x = np.random.rand(vector_len).astype(np.float32)
        y = np.random.rand(vector_len).astype(np.float32)

        energy_enabled = energy_measurement_supported()
        energy_logger = EnergyLogger() if energy_enabled else None

        for iters_inner in iters_list:
            print(f"\n--- iters_inner = {iters_inner} ---")

            gflops_list: List[float] = []
            energy_list: List[float] = []
            power_list: List[float] = []

            for run_id in range(runs_per_config):
                if energy_logger is not None:
                    energy_logger.start()

                t0 = time.perf_counter()
                cuda_fma_kernel(ctx, x, y, iters_inner)
                t1 = time.perf_counter()

                if energy_logger is not None:
                    energy_j, power_w = energy_logger.stop()
                else:
                    energy_j, power_w = math.nan, math.nan

                elapsed = t1 - t0
                # liczba operacji FMA: iters_inner * vector_len
                # FMA liczymy jako 2 FLOP (multiply + add)
                gflops = (2.0 * iters_inner * vector_len) / (elapsed * 1e9)

                gflops_list.append(gflops)
                energy_list.append(energy_j)
                power_list.append(power_w)

                print(
                    f"run {run_id:2d}: elapsed = {elapsed:8.4f} s, "
                    f"GFlop/s = {gflops:7.2f}, energy = {energy_j:8.3f} J, "
                    f"P_avg = {power_w:8.3f} W"
                )

                writer.writerow(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "backend": "cuda",
                        "gpu_name": gpu_name,
                        "device_index": device_index,
                        "vector_len": vector_len,
                        "iters_inner": iters_inner,
                        "run_id": run_id,
                        "elapsed_s": elapsed,
                        "gflops": gflops,
                        "energy_j": energy_j,
                        "power_w": power_w,
                    }
                )

            mean_gflops = float(np.mean(gflops_list)) if gflops_list else math.nan
            std_gflops = float(np.std(gflops_list)) if gflops_list else math.nan
            print(
                f"==> MEAN: {mean_gflops:7.2f} GFLOP/s, "
                f"sigma = {std_gflops:7.2f} GFLOP/s"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA GPU FMA compute benchmark."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Index urządzenia CUDA (domyślnie 0)",
    )
    parser.add_argument(
        "--vector-len",
        type=int,
        default=1_000_000,
        help="Długość wektora (domyślnie 1e6)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[250_000, 500_000, 1_000_000],
        help="Lista wartości iters_inner",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Liczba powtórzeń dla każdej konfiguracji (domyślnie 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fma_bench(
        device_index=args.device,
        vector_len=args.vector_len,
        iters_list=args.iters,
        runs_per_config=args.runs,
    )


if __name__ == "__main__":
    main()
