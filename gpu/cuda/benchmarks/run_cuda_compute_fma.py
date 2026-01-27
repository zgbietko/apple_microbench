from __future__ import annotations

import argparse
import csv
import math
import platform
import socket
import statistics as stats
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_utils  # type: ignore

try:
    from energy_utils import EnergyLogger  # type: ignore
except Exception:
    EnergyLogger = None  # type: ignore

from gpu.cuda.cuda_backend import (
    init_cuda,
    get_device_info,
    cuda_fma_throughput,
)


def _system_metadata() -> Dict[str, Any]:
    return {
        "backend": "cuda",
        "system": platform.system(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }


def run_fma_bench(
    device_index: int,
    n_elements: int,
    iters_inner: int,
    runs: int,
) -> None:
    ctx = init_cuda(device_index)
    info = get_device_info(ctx)
    gpu_name = info.name

    print("=== GPU FMA compute benchmark (CUDA) ===")
    print(f"GPU device : {gpu_name} (index {device_index})")
    print(f"n_elements : {n_elements}")
    print(f"iters_inner per run: {iters_inner}")
    print(f"runs      : {runs}")
    print()

    data_dir = ROOT / "data" / "gpu"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = gpu_utils.make_gpu_specific_csv_path(
        benchmark_name="gpu_compute_fma",
        data_dir=data_dir,
        gpu_backend="cuda",
        gpu_name=gpu_name,
        device_id=device_index,
    )
    header_written = csv_path.exists() and csv_path.stat().st_size > 0

    fieldnames = [
        "timestamp",
        "backend",
        "system",
        "arch",
        "hostname",
        "python_version",
        "gpu_model",
        "gpu_index",
        "n_elements",
        "iters_inner",
        "run_idx",
        "elapsed_s",
        "gflops",
        "energy_joule",
        "avg_power_watt",
    ]

    logger = EnergyLogger() if EnergyLogger is not None else None

    flops_per_iter = 2.0 * float(n_elements)  # 1 FMA = 2 FLOP
    gflops_values: List[float] = []

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()

        for run_idx in range(runs):
            energy_j = float("nan")
            power_w = float("nan")

            if logger is not None:
                logger.start()

            elapsed_s = cuda_fma_throughput(
                ctx, n=n_elements, iters_inner=iters_inner
            )

            if logger is not None:
                try:
                    energy_j, power_w = logger.stop()
                except RuntimeError:
                    energy_j = float("nan")
                    power_w = float("nan")

            total_flops = flops_per_iter * float(iters_inner)
            gflops = total_flops / elapsed_s / 1e9
            gflops_values.append(gflops)

            print(
                f"run {run_idx:2d}: elapsed = {elapsed_s:8.4f} s, "
                f"GFlop/s = {gflops:7.2f}, energy = {energy_j:7.3f} J, "
                f"P_avg = {power_w:7.3f} W"
            )

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **_system_metadata(),
                "gpu_model": gpu_name,
                "gpu_index": device_index,
                "n_elements": int(n_elements),
                "iters_inner": int(iters_inner),
                "run_idx": run_idx,
                "elapsed_s": elapsed_s,
                "gflops": gflops,
                "energy_joule": energy_j,
                "avg_power_watt": power_w,
            }
            writer.writerow(row)

    mean_gflops = stats.mean(gflops_values)
    sigma_gflops = stats.pstdev(gflops_values) if len(gflops_values) > 1 else 0.0
    print(
        f"==> MEAN: {mean_gflops:7.2f} GFLOP/s, sigma = {sigma_gflops:7.2f} GFLOP/s"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU FMA compute benchmark (CUDA)."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index urządzenia CUDA.",
    )
    parser.add_argument(
        "--n-elements",
        type=int,
        default=1_048_576,
        help="Liczba elementów wektorów (domyślnie 1M).",
    )
    parser.add_argument(
        "--iters-inner",
        type=int,
        default=1_000,
        help="Liczba iteracji pętli FMA wewnątrz kernelu.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Liczba powtórzeń benchmarku.",
    )

    args = parser.parse_args()

    run_fma_bench(
        device_index=args.device_index,
        n_elements=args.n_elements,
        iters_inner=args.iters_inner,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
