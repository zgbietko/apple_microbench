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


def run_gpu_fma(
    preferred_device: int | None,
    vector_len: int,
    runs_per_config: int,
    inner_iters_list: list[int],
) -> None:
    """
    Benchmark FMA na CUDA (GFLOP/s), analogiczny do GPU Metal i CPU.
    gpu_fma_elapsed(n, dev, iters) mierzy czas dla:
      - n elementów float32,
      - iters iteracji w kernelu.
    FLOP = 2 * n * iters.
    """
    lib, lib_path = gpu_utils.load_cuda_library()
    gpu_utils.configure_cuda_functions(lib)

    dev_id, dev_name = gpu_utils.select_cuda_device(lib, preferred_index=preferred_device)

    print("=== GPU FMA compute benchmark (CUDA) ===")
    print(f"CUDA library    : {lib_path}")
    print(f"GPU device      : {dev_name} (id {dev_id})")
    print(f"vector_len      : {vector_len}")
    print(f"runs per config : {runs_per_config}")
    print(f"inner_iters     : {inner_iters_list}")
    print()

    csv_path = gpu_utils.make_gpu_csv_path(
        ROOT, "gpu_compute_fma", "cuda", dev_name
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

    for inner_iters in inner_iters_list:
        print(f"--- inner_iters = {inner_iters} ---")

        times: list[float] = []

        for run_idx in range(runs_per_config):
            elapsed = lib.gpu_fma_elapsed(
                vector_len,
                dev_id,
                inner_iters,
            )
            if elapsed <= 0.0:
                print(
                    f"[ERROR] gpu_fma_elapsed zwrócił {elapsed} s "
                    f"(run {run_idx}, inner_iters {inner_iters})"
                )
                continue

            flops = 2.0 * float(vector_len) * float(inner_iters)
            gflops = flops / elapsed / 1e9

            times.append(elapsed)

            print(
                f"run {run_idx:2d}: elapsed = {elapsed:8.4f} s, "
                f"GFLOP/s = {gflops:7.2f}"
            )

            row = gpu_utils.common_gpu_metadata("cuda", dev_name, dev_id)
            row.update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "vector_len": str(vector_len),
                    "inner_iters": str(inner_iters),
                    "run_idx": str(run_idx),
                    "elapsed_s": f"{elapsed:.6f}",
                    "throughput_gflops": f"{gflops:.4f}",
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
            mean_gflops = stats.mean(
                (2.0 * float(vector_len) * float(inner_iters) / t) / 1e9
                for t in times
            )
            std_gflops = stats.pstdev(
                (2.0 * float(vector_len) * float(inner_iters) / t) / 1e9
                for t in times
            ) if len(times) > 1 else 0.0

            print(
                f"==> MEAN: {mean_gflops:7.2f} GFLOP/s, "
                f"sigma = {std_gflops:7.2f} GFLOP/s"
            )
        print()

    print(f"Wszystkie runy (CUDA FMA) zapisane do: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU FMA compute benchmark (CUDA)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="ID urządzenia CUDA (jeśli nie podano – wybór automatyczny).",
    )
    parser.add_argument(
        "--vector-len",
        type=int,
        default=1 << 20,  # 1M elementów
        help="Długość wektora (liczba elementów float32).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
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
    run_gpu_fma(
        preferred_device=args.device,
        vector_len=args.vector_len,
        runs_per_config=args.runs,
        inner_iters_list=args.inner_iters,
    )


if __name__ == "__main__":
    main()
