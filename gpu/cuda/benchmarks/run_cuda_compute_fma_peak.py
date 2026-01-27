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


def run_fma_peak_bench(
    device_index: int,
    n_elements_options: List[int],
    iters_inner_options: List[int],
    runs_per_config: int,
) -> None:
    ctx = init_cuda(device_index)
    info = get_device_info(ctx)
    gpu_name = info.name

    print("=== GPU Peak FMA benchmark (CUDA) ===")
    print(f"GPU device : {gpu_name} (index {device_index})")
    print(f"n_elements options : {n_elements_options}")
    print(f"iters_inner options: {iters_inner_options}")
    print(f"runs per config    : {runs_per_config}")
    print()

    data_dir = ROOT / "data" / "gpu"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = gpu_utils.make_gpu_specific_csv_path(
        benchmark_name="gpu_compute_fma_peak",
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
        "runs_per_config",
        "gflops_peak",
        "gflops_mean",
        "gflops_sigma",
        "energy_joule_mean",
        "energy_joule_sigma",
        "avg_power_watt_mean",
        "avg_power_watt_sigma",
    ]

    logger = EnergyLogger() if EnergyLogger is not None else None

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()

        for n_elems in n_elements_options:
            flops_per_iter = 2.0 * float(n_elems)  # 1 FMA = 2 FLOP

            for iters_inner in iters_inner_options:
                print(f"### n_elements = {n_elems}, iters_inner = {iters_inner} ###")

                gflops_runs: List[float] = []
                energy_runs: List[float] = []
                power_runs: List[float] = []

                for run_idx in range(runs_per_config):
                    energy_j = float("nan")
                    power_w = float("nan")

                    if logger is not None:
                        logger.start()

                    elapsed_s = cuda_fma_throughput(
                        ctx, n=n_elems, iters_inner=iters_inner
                    )

                    if logger is not None:
                        try:
                            energy_j, power_w = logger.stop()
                        except RuntimeError:
                            energy_j = float("nan")
                            power_w = float("nan")

                    total_flops = flops_per_iter * float(iters_inner)
                    gflops = total_flops / elapsed_s / 1e9

                    gflops_runs.append(gflops)
                    if not math.isnan(energy_j):
                        energy_runs.append(energy_j)
                    if not math.isnan(power_w):
                        power_runs.append(power_w)

                    print(
                        f"run {run_idx:2d}: elapsed = {elapsed_s:8.4f} s, "
                        f"GFlop/s = {gflops:7.2f}, "
                        f"energy = {energy_j:7.3f} J, P_avg = {power_w:7.3f} W"
                    )

                if gflops_runs:
                    gflops_peak = max(gflops_runs)
                    gflops_mean = stats.mean(gflops_runs)
                    gflops_sigma = (
                        stats.pstdev(gflops_runs) if len(gflops_runs) > 1 else 0.0
                    )
                else:
                    gflops_peak = gflops_mean = gflops_sigma = float("nan")

                if energy_runs:
                    e_mean = stats.mean(energy_runs)
                    e_sigma = (
                        stats.pstdev(energy_runs) if len(energy_runs) > 1 else 0.0
                    )
                else:
                    e_mean = e_sigma = float("nan")

                if power_runs:
                    p_mean = stats.mean(power_runs)
                    p_sigma = (
                        stats.pstdev(power_runs) if len(power_runs) > 1 else 0.0
                    )
                else:
                    p_mean = p_sigma = float("nan")

                print(
                    f"==> PEAK: {gflops_peak:7.2f} GFLOP/s, "
                    f"MEAN: {gflops_mean:7.2f} GFLOP/s, "
                    f"sigma = {gflops_sigma:7.2f} GFLOP/s"
                )
                print()

                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **_system_metadata(),
                    "gpu_model": gpu_name,
                    "gpu_index": device_index,
                    "n_elements": int(n_elems),
                    "iters_inner": int(iters_inner),
                    "runs_per_config": runs_per_config,
                    "gflops_peak": gflops_peak,
                    "gflops_mean": gflops_mean,
                    "gflops_sigma": gflops_sigma,
                    "energy_joule_mean": e_mean,
                    "energy_joule_sigma": e_sigma,
                    "avg_power_watt_mean": p_mean,
                    "avg_power_watt_sigma": p_sigma,
                }
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Peak FMA benchmark (CUDA)."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index urządzenia CUDA.",
    )
    parser.add_argument(
        "--n-elements",
        type=str,
        default="1048576,4194304,16777216",
        help=(
            "Lista n_elements (liczba elementów wektora), "
            "np. '1048576,4194304,16777216'."
        ),
    )
    parser.add_argument(
        "--iters-inner",
        type=str,
        default="1000,5000,10000",
        help=(
            "Lista iters_inner (liczba iteracji pętli FMA), "
            "np. '1000,5000,10000'."
        ),
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=3,
        help="Liczba runów dla każdej konfiguracji (n_elements, iters_inner).",
    )

    args = parser.parse_args()

    n_elements_options = [int(x) for x in args.n_elements.split(",") if x.strip()]
    iters_inner_options = [int(x) for x in args.iters_inner.split(",") if x.strip()]

    run_fma_peak_bench(
        device_index=args.device_index,
        n_elements_options=n_elements_options,
        iters_inner_options=iters_inner_options,
        runs_per_config=args.runs_per_config,
    )


if __name__ == "__main__":
    main()
