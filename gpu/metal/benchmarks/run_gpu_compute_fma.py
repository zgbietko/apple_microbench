# gpu/benchmarks/run_gpu_compute_fma.py
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


def bench_gpu_fma(lib, device_id: int, n: int, iters: int):
    elapsed_s = lib.gpu_fma_elapsed(n, iters, device_id)
    total_ops = 2.0 * float(n) * float(iters)  # 1 mul + 1 add na iterację
    if elapsed_s > 0.0:
        gflops = (total_ops / elapsed_s) / 1e9
    else:
        gflops = 0.0

    return {
        "n": n,
        "iters_inner": iters,
        "elapsed_s": elapsed_s,
        "gflops": gflops,
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
        "n",
        "iters_inner",
        "elapsed_s",
        "gflops",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="GPU FMA compute throughput benchmark (CUDA)."
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
        "--n",
        type=int,
        default=1_000_000,
        help="Długość wektora (liczba elementów float).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[250_000, 500_000, 1_000_000],
        help="Lista wartości iters_inner dla kernela FMA.",
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
        "gpu_compute_fma",
        data_dir,
        gpu_backend=gpu_backend,
        gpu_name=gpu_name,
        device_id=device_id,
    )
    if csv_path.exists():
        csv_path.unlink()

    meta = collect_metadata(gpu_backend, gpu_name, device_id)

    runs = args.runs
    n = args.n
    iters_list = args.iters

    print("=== GPU FMA compute throughput benchmark (CUDA) ===")
    print(f"GPU backend      : {gpu_backend}")
    print(f"GPU device       : {device_id} -> {gpu_name}")
    print(f"vector length    : n = {n}")
    print(f"runs per iters   : {runs}")
    print(f"iters list       : {iters_list}")
    print(f"CSV output       : {csv_path}")

    header_written = False

    for iters_inner in iters_list:
        gflops_values = []

        print(f"\n--- iters_inner = {iters_inner} ---")

        for run_id in range(runs):
            result = bench_gpu_fma(lib, device_id, n, iters_inner)
            gflops_values.append(result["gflops"])

            write_result_to_csv(
                csv_path,
                result,
                meta,
                benchmark_name="gpu_fma",
                run_id=run_id,
                write_header=not header_written,
            )
            header_written = True

            print(
                f"run {run_id:2d}: elapsed = {result['elapsed_s']:.6f} s, "
                f"GFlop/s = {result['gflops']:.2f}"
            )

        mean_gflops = stats.mean(gflops_values)
        stdev_gflops = stats.pstdev(gflops_values) if len(gflops_values) > 1 else 0.0
        print(f"==> ŚREDNIA: {mean_gflops:.2f} GF/s, σ = {stdev_gflops:.2f} GF/s")

    print(f"\nWszystkie runy zapisane do: {csv_path}")


if __name__ == "__main__":
    main()
