from __future__ import annotations

import argparse
import csv
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path
import sys

# ROOT projektu: .../apple_microbench
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_utils  # z apple_microbench/gpu_utils.py


def run_gpu_bandwidth(
    preferred_device: int | None,
    runs_per_size: int,
    sizes_mb: list[int],
    iters_kernel: int,
) -> None:
    """
    Benchmark przepustowości pamięci na CUDA:
      - kernel mem_copy_kernel (device->device copy),
      - bytes_total = 2 * size_bytes * iters_kernel (read + write, iters razy).
    """
    lib, lib_path = gpu_utils.load_cuda_library()
    gpu_utils.configure_cuda_functions(lib)

    dev_id, dev_name = gpu_utils.select_cuda_device(lib, preferred_index=preferred_device)

    print("=== GPU memory bandwidth benchmark (CUDA, mem_copy_kernel) ===")
    print(f"CUDA library : {lib_path}")
    print(f"GPU device   : {dev_name} (id {dev_id})")
    print(f"runs per size: {runs_per_size}")
    print(f"iters_kernel : {iters_kernel}")
    print()

    csv_path = gpu_utils.make_gpu_csv_path(
        ROOT, "gpu_bandwidth", "cuda", dev_name
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
        "size_bytes",
        "num_elements",
        "run_idx",
        "elapsed_s",
        "throughput_gbps",
        "energy_joule",
        "avg_power_watt",
    ]

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32
        print(f"--- Size: {size_mb:5d} MB ({size_bytes} bytes, {num_elements} elements) ---")

        times: list[float] = []

        for run_idx in range(runs_per_size):
            elapsed = lib.gpu_mem_copy_elapsed(
                size_bytes,
                dev_id,
                iters_kernel,
            )
            if elapsed <= 0.0:
                print(
                    f"[ERROR] gpu_mem_copy_elapsed zwrócił {elapsed} s "
                    f"(run {run_idx}, size {size_mb} MB)"
                )
                continue

            # 2 * size_bytes * iters_kernel (read+write, powtórzone iters_kernel razy)
            total_bytes = 2.0 * float(size_bytes) * float(iters_kernel)
            gbps = total_bytes / elapsed / 1e9

            times.append(elapsed)

            print(
                f"run {run_idx:2d}: elapsed = {elapsed:8.4f} s, "
                f"GB/s = {gbps:7.2f}"
            )

            # Na razie energii nie mierzymy dla CUDA – zostawiamy puste pola.
            row = gpu_utils.common_gpu_metadata("cuda", dev_name, dev_id)
            row.update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": str(size_bytes),
                    "num_elements": str(num_elements),
                    "run_idx": str(run_idx),
                    "elapsed_s": f"{elapsed:.6f}",
                    "throughput_gbps": f"{gbps:.4f}",
                    "energy_joule": "",
                    "avg_power_watt": "",
                }
            )

            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)

        if times:
            mean_gbps = stats.mean(
                (2.0 * float(size_bytes) * float(iters_kernel) / t) / 1e9 for t in times
            )
            std_gbps = stats.pstdev(
                (2.0 * float(size_bytes) * float(iters_kernel) / t) / 1e9 for t in times
            ) if len(times) > 1 else 0.0

            print(
                f"==> MEAN: {mean_gbps:7.2f} GB/s, "
                f"sigma = {std_gbps:7.2f} GB/s"
            )
        print()

    print(f"Wszystkie runy (CUDA bandwidth) zapisane do: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU memory bandwidth benchmark (CUDA)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="ID urządzenia CUDA (jeśli nie podano – wybór automatyczny).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Liczba powtórzeń dla każdego rozmiaru.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Liczba iteracji w kernelu (inner loop).",
    )
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="*",
        default=[4, 16, 64, 256, 1024],
        help="Lista rozmiarów bufora w MB (domyślnie: 4 16 64 256 1024).",
    )

    args = parser.parse_args()
    run_gpu_bandwidth(
        preferred_device=args.device,
        runs_per_size=args.runs,
        sizes_mb=args.sizes_mb,
        iters_kernel=args.iters,
    )


if __name__ == "__main__":
    main()
