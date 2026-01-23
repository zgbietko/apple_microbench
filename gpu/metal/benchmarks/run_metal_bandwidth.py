from __future__ import annotations

import argparse
import csv
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path
import sys

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


def run_bandwidth_bench(
    device_index: int,
    runs_per_size: int,
    sizes_mb: list[int],
) -> None:
    backend = MetalBackend(device_index=device_index)
    gpu_name = backend.device_name

    print("=== GPU memory bandwidth benchmark (Metal, mem_copy_kernel) ===")
    print(f"GPU device   : {gpu_name} (index {device_index})")
    print(f"runs per size: {runs_per_size}")
    print()

    csv_path = gpu_utils.make_gpu_csv_path(ROOT, "gpu_bandwidth", "metal", gpu_name)
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
        "size_bytes",
        "num_elements",
        "run_idx",
        "elapsed_s",
        "throughput_gbps",
        "energy_joule",
        "avg_power_watt",
    ]

    logger = EnergyLogger() if EnergyLogger is not None else None

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32
        print(f"--- Size: {size_mb:5d} MB ({size_bytes} bytes, {num_elements} elements) ---")

        times: list[float] = []
        energies: list[float] = []
        powers: list[float] = []

        for run_idx in range(runs_per_size):
            energy_j = float("nan")
            avg_power_w = float("nan")

            if logger is not None:
                logger.start()

            elapsed = backend.run_mem_copy(num_elements)

            if logger is not None:
                try:
                    energy_j, avg_power_w = logger.stop()
                except Exception:
                    energy_j, avg_power_w = float("nan"), float("nan")

            # Przyjmujemy 2*size_bytes (odczyt + zapis) jako ruch pamięci
            bytes_total = 2.0 * size_bytes
            gbps = bytes_total / elapsed / 1e9

            times.append(elapsed)
            energies.append(energy_j)
            powers.append(avg_power_w)

            print(
                f"run {run_idx:2d}: elapsed = {elapsed:8.4f} s, "
                f"GB/s = {gbps:7.2f}, energy = {energy_j:7.4f} J, "
                f"P_avg = {avg_power_w:7.2f} W"
            )

            # Zapis pojedynczego runa do CSV
            row = gpu_utils.common_gpu_metadata("metal", gpu_name, device_index)
            row.update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": str(size_bytes),
                    "num_elements": str(num_elements),
                    "run_idx": str(run_idx),
                    "elapsed_s": f"{elapsed:.6f}",
                    "throughput_gbps": f"{gbps:.4f}",
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

        mean_gbps = stats.mean(2.0 * size_bytes / t / 1e9 for t in times)
        std_gbps = stats.pstdev(2.0 * size_bytes / t / 1e9 for t in times)

        print(
            f"==> MEAN: {mean_gbps:7.2f} GB/s, "
            f"sigma = {std_gbps:7.2f} GB/s"
        )
        if energies and any(e == e for e in energies):  # przynajmniej jedna wartość nie-NaN
            valid_energies = [e for e in energies if e == e]
            valid_powers = [p for p in powers if p == p]
            e_mean = stats.mean(valid_energies)
            e_std = stats.pstdev(valid_energies) if len(valid_energies) > 1 else 0.0
            p_mean = stats.mean(valid_powers)
            p_std = stats.pstdev(valid_powers) if len(valid_powers) > 1 else 0.0
            print(
                f"    energy: {e_mean:7.4f} J ± {e_std:7.4f} J, "
                f"P_avg: {p_mean:7.2f} W ± {p_std:7.2f} W"
            )

        print()

    print(f"Wszystkie runy zapisane do: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU memory bandwidth benchmark (Metal)."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Indeks urządzenia Metal (jeśli jest ich więcej niż jedno).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Liczba powtórzeń dla każdego rozmiaru.",
    )
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="*",
        default=[4, 16, 64, 256, 1024],
        help="Lista rozmiarów bufora w MB (domyślnie: 4 16 64 256 1024).",
    )

    args = parser.parse_args()
    run_bandwidth_bench(
        device_index=args.device_index,
        runs_per_size=args.runs,
        sizes_mb=args.sizes_mb,
    )


if __name__ == "__main__":
    main()
