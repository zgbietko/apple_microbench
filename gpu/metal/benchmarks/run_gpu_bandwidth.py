# gpu/benchmarks/run_gpu_bandwidth.py
import argparse
import csv
import platform
from datetime import datetime
from pathlib import Path
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu.gpu_utils import (
    load_cuda_library,
    configure_cuda_functions,
    select_cuda_device,
    make_gpu_specific_csv_path,
)


def collect_metadata(gpu_backend: str, gpu_name: str, device_id: int):
    """
    Metadane hosta + GPU (model karty, backend, device_id).
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host_system": platform.system(),
        "host_node": platform.node(),
        "host_release": platform.release(),
        "host_version": platform.version(),
        "host_machine": platform.machine(),
        "host_arch": platform.machine(),
        "python_version": platform.python_version(),
        "gpu_backend": gpu_backend,
        "gpu_device_id": device_id,
        "gpu_name": gpu_name,
    }


def bench_gpu_mem_copy(lib, device_id: int, bytes_per_iter: int, iters: int):
    elapsed_s = lib.gpu_mem_copy_elapsed(bytes_per_iter, iters, device_id)
    total_bytes = float(bytes_per_iter) * float(iters)
    if elapsed_s > 0.0:
        gbps = (total_bytes / elapsed_s) / (1024.0**3)
    else:
        gbps = 0.0

    return {
        "size_bytes": bytes_per_iter,
        "iters": iters,
        "elapsed_s": elapsed_s,
        "gbps": gbps,
    }


def write_result_to_csv(
    csv_path: Path,
    result: dict,
    meta: dict,
    benchmark_name: str,
    run_id: int,
    write_header: bool,
):
    row = {
        **meta,
        "benchmark": benchmark_name,
        "run_id": run_id,
        **result,
    }

    fieldnames = [
        "timestamp",
        "host_system",
        "host_node",
        "host_release",
        "host_version",
        "host_machine",
        "host_arch",
        "python_version",
        "gpu_backend",
        "gpu_device_id",
        "gpu_name",
        "benchmark",
        "run_id",
        "size_bytes",
        "iters",
        "elapsed_s",
        "gbps",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="GPU memory bandwidth benchmark (CUDA)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Indeks urządzenia CUDA (domyślnie z GPU_DEVICE_ID lub 0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Liczba powtórzeń na konfigurację.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Liczba iteracji w kernelu GPU.",
    )

    args = parser.parse_args()

    try:
        lib, lib_path = load_cuda_library()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Zbuduj najpierw GPU lib: gpu/lib/build_cuda.sh")
        sys.exit(1)

    configure_cuda_functions(lib)

    device_id, gpu_name = select_cuda_device(lib, preferred_index=args.device)
    gpu_backend = "cuda"

    root = ROOT
    data_dir = root / "data" / "gpu"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = make_gpu_specific_csv_path(
        "gpu_bandwidth",
        data_dir,
        gpu_backend=gpu_backend,
        gpu_name=gpu_name,
        device_id=device_id,
    )
    if csv_path.exists():
        csv_path.unlink()

    meta = collect_metadata(gpu_backend, gpu_name, device_id)

    sizes_mb = [64, 256, 1024]  # 64 MB..1 GB
    runs = args.runs
    iters = args.iters

    print("=== GPU memory bandwidth benchmark (CUDA) ===")
    print(f"GPU backend      : {gpu_backend}")
    print(f"GPU device       : {device_id} -> {gpu_name}")
    print(f"runs per size    : {runs}")
    print(f"iters in kernel  : {iters}")
    print(f"CSV output       : {csv_path}")

    header_written = False

    for size_mb in sizes_mb:
        bytes_per_iter = size_mb * 1024 * 1024
        gbps_values = []

        print(f"\n--- Size: {size_mb} MB ---")

        for run_id in range(runs):
            result = bench_gpu_mem_copy(lib, device_id, bytes_per_iter, iters)
            gbps_values.append(result["gbps"])

            write_result_to_csv(
                csv_path,
                result,
                meta,
                benchmark_name="gpu_mem_copy",
                run_id=run_id,
                write_header=not header_written,
            )
            header_written = True

            print(
                f"run {run_id:2d}: elapsed = {result['elapsed_s']:.6f} s, "
                f"GB/s = {result['gbps']:.2f}"
            )

        mean_gbps = stats.mean(gbps_values)
        stdev_gbps = stats.pstdev(gbps_values) if len(gbps_values) > 1 else 0.0
        print(f"==> ŚREDNIA: {mean_gbps:.2f} GB/s, σ = {stdev_gbps:.2f} GB/s")

    print(f"\nWszystkie runy zapisane do: {csv_path}")


if __name__ == "__main__":
    main()
