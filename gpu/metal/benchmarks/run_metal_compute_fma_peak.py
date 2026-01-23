from __future__ import annotations

import argparse
import csv
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path
import sys
import math

# ROOT projektu: .../apple_microbench
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_utils  # z apple_microbench/gpu_utils.py

try:
    from energy_utils import EnergyLogger  # type: ignore
except Exception:
    EnergyLogger = None  # type: ignore

from gpu.metal.metal_backend import MetalBackend  # type: ignore


def run_fma_peak_bench(
    device_index: int,
    vector_lens: list[int],
    inner_iters_list: list[int],
    runs_per_config: int,
) -> None:
    backend = MetalBackend(device_index=device_index)
    gpu_name = backend.device_name

    print("=== GPU Peak FMA compute benchmark (Metal, fma_kernel) ===")
    print(f"GPU device       : {gpu_name} (index {device_index})")
    print(f"vector_lens      : {vector_lens}")
    print(f"runs per config  : {runs_per_config}")
    print(f"inner_iters list : {inner_iters_list}")
    print()

    csv_path = gpu_utils.make_gpu_csv_path(
        ROOT, "gpu_compute_fma_peak", "metal", gpu_name
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
        "vector_len",
        "inner_iters",
        "run_idx",
        "elapsed_s",
        "throughput_gflops",
        "energy_joule",
        "avg_power_watt",
    ]

    logger = EnergyLogger() if EnergyLogger is not None else None

    best_gflops = 0.0
    best_conf = None  # (vector_len, inner_iters)

    for n in vector_lens:
        for inner_iters in inner_iters_list:
            print(f"--- vector_len = {n}, inner_iters = {inner_iters} ---")

            times: list[float] = []
            energies: list[float] = []
            powers: list[float] = []

            for run_idx in range(runs_per_config):
                energy_j = float("nan")
                avg_power_w = float("nan")

                if logger is not None:
                    logger.start()

                elapsed = backend.run_fma(n, inner_iters)

                if logger is not None:
                    try:
                        energy_j, avg_power_w = logger.stop()
                    except Exception:
                        energy_j, avg_power_w = float("nan"), float("nan")

                # 2 FLOP na FMA
                flops = 2.0 * float(n) * float(inner_iters)
                gflops = flops / elapsed / 1e9

                times.append(elapsed)
                energies.append(energy_j)
                powers.append(avg_power_w)

                print(
                    f"run {run_idx:2d}: elapsed = {elapsed:8.4f} s, "
                    f"GFLOP/s = {gflops:8.2f}, energy = {energy_j:7.4f} J, "
                    f"P_avg = {avg_power_w:7.2f} W"
                )

                row = gpu_utils.common_gpu_metadata("metal", gpu_name, device_index)
                row.update(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "vector_len": str(n),
                        "inner_iters": str(inner_iters),
                        "run_idx": str(run_idx),
                        "elapsed_s": f"{elapsed:.6f}",
                        "throughput_gflops": f"{gflops:.4f}",
                        "energy_joule": (
                            f"{energy_j:.6f}" if energy_j == energy_j else ""
                        ),
                        "avg_power_watt": (
                            f"{avg_power_w:.6f}" if avg_power_w == avg_power_w else ""
                        ),
                    }
                )

                with csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not header_written:
                        writer.writeheader()
                        header_written = True
                    writer.writerow(row)

                # śledzenie globalnego rekordu GFLOP/s
                if not math.isnan(gflops) and gflops > best_gflops:
                    best_gflops = gflops
                    best_conf = (n, inner_iters)

            # statystyki dla tej konfiguracji
            conf_gflops = [
                2.0 * float(n) * float(inner_iters) / t / 1e9 for t in times
            ]
            mean_gflops = stats.mean(conf_gflops)
            std_gflops = stats.pstdev(conf_gflops) if len(conf_gflops) > 1 else 0.0

            print(
                f"==> MEAN: {mean_gflops:8.2f} GFLOP/s, "
                f"sigma = {std_gflops:8.2f} GFLOP/s"
            )

            if energies and any(e == e for e in energies):
                valid_energies = [e for e in energies if e == e]
                valid_powers = [p for p in powers if p == p]
                if valid_energies:
                    e_mean = stats.mean(valid_energies)
                    e_std = (
                        stats.pstdev(valid_energies)
                        if len(valid_energies) > 1
                        else 0.0
                    )
                else:
                    e_mean = e_std = 0.0
                if valid_powers:
                    p_mean = stats.mean(valid_powers)
                    p_std = (
                        stats.pstdev(valid_powers)
                        if len(valid_powers) > 1
                        else 0.0
                    )
                else:
                    p_mean = p_std = 0.0

                print(
                    f"    energy: {e_mean:7.4f} J ± {e_std:7.4f} J, "
                    f"P_avg: {p_mean:7.2f} W ± {p_std:7.2f} W"
                )

            print()

    if best_conf is not None:
        n_best, it_best = best_conf
        print(
            f"=== PEAK GFLOP/s (Metal) ===\n"
            f"GPU: {gpu_name}\n"
            f"vector_len = {n_best}, inner_iters = {it_best}\n"
            f"Peak = {best_gflops:.2f} GFLOP/s\n"
        )
    else:
        print("=== PEAK GFLOP/s (Metal) ===\nBrak poprawnych pomiarów.\n")

    print(f"Wszystkie runy zapisane do: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Peak FMA compute benchmark (Metal)."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Indeks urządzenia Metal (jeśli jest ich więcej niż jedno).",
    )
    parser.add_argument(
        "--vector-lens",
        type=int,
        nargs="*",
        default=[1 << 18, 1 << 20, 1 << 22],  # 256k, 1M, 4M elementów
        help="Lista długości wektora (liczba elementów float32).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Liczba powtórzeń dla każdej konfiguracji.",
    )
    parser.add_argument(
        "--inner-iters",
        type=int,
        nargs="*",
        default=[1000, 5000, 10000],
        help="Lista wartości inner_iters (domyślnie: 1000 5000 10000).",
    )

    args = parser.parse_args()
    run_fma_peak_bench(
        device_index=args.device_index,
        vector_lens=args.vector_lens,
        inner_iters_list=args.inner_iters,
        runs_per_config=args.runs,
    )


if __name__ == "__main__":
    main()
