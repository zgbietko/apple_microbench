from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics as stats
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# We assume this file lives in apple_microbench/gpu/cuda/benchmarks
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpu_utils import detect_cpu_model  # type: ignore
from energy_utils import EnergyLogger, energy_measurement_supported  # type: ignore
from gpu.cuda.cuda_backend import CudaBackend  # type: ignore


def slugify(text: str) -> str:
    """Simple slug for filenames (a-z0-9 and underscores only)."""
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def get_system_metadata() -> Dict[str, Any]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": detect_cpu_model(),
        "python_version": platform.python_version(),
    }


def write_row(
    csv_path: Path,
    header_written: bool,
    row: Dict[str, Any],
) -> bool:
    """Append a row to CSV, writing header if needed.

    Returns True if header is now written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    write_header_now = not header_written or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header_now:
            writer.writeheader()
        writer.writerow(row)

    return True


def run_bandwidth_bench(
    device_index: int,
    sizes_mb: List[int],
    iters_per_run: int,
    runs_per_size: int,
) -> None:
    backend = CudaBackend()
    dev_info = backend.get_device_info(device_index)

    gpu_slug = slugify(dev_info.name)
    data_dir = ROOT / "data" / "gpu"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"gpu_bandwidth_cuda_{gpu_slug}.csv"

    print("=== GPU memory bandwidth benchmark (CUDA) ===")
    print(f"GPU device   : {dev_info.name} (index {dev_info.index})")
    print(f"Compute cap. : {dev_info.compute_capability}")
    print(f"Global mem   : {dev_info.global_mem_gb:.1f} GiB")
    print(f"runs per size: {runs_per_size}")
    print(f"iters per run: {iters_per_run}")
    print()

    sys_meta = get_system_metadata()
    energy_supported = energy_measurement_supported()

    header_written = csv_path.exists() and csv_path.stat().st_size > 0

    for size_mb in sizes_mb:
        num_bytes = size_mb * 1024 * 1024

        print(
            f"--- Size: {size_mb:6d} MB ({num_bytes} bytes) ---"
        )

        gbps_values: List[float] = []
        energy_values: List[float] = []
        avg_power_values: List[float] = []

        for run_id in range(runs_per_size):
            energy_j = math.nan
            avg_power_w = math.nan
            energy_logger = None

            if energy_supported:
                energy_logger = EnergyLogger()
                energy_logger.start()

            elapsed_s = backend.memcpy_bandwidth_seconds(
                device_index=device_index,
                num_bytes=num_bytes,
                iters=iters_per_run,
            )

            if energy_logger is not None:
                try:
                    energy_j, avg_power_w = energy_logger.stop()
                except Exception:
                    # Fall back to NaNs if energy measurement fails.
                    energy_j = math.nan
                    avg_power_w = math.nan

            bytes_total = num_bytes * iters_per_run
            gbps = bytes_total / elapsed_s / (1024 ** 3)

            gbps_values.append(gbps)
            if not math.isnan(energy_j):
                energy_values.append(energy_j)
            if not math.isnan(avg_power_w):
                avg_power_values.append(avg_power_w)

            row: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **sys_meta,
                "backend": "cuda",
                "benchmark": "gpu_bandwidth",
                "gpu_name": dev_info.name,
                "gpu_index": dev_info.index,
                "gpu_compute_capability": dev_info.compute_capability,
                "gpu_global_mem_gb": dev_info.global_mem_gb,
                "run_id": run_id,
                "size_mb": size_mb,
                "bytes_per_buffer": num_bytes,
                "iters_per_run": iters_per_run,
                "bytes_total": bytes_total,
                "elapsed_s": elapsed_s,
                "gbps": gbps,
                "energy_j": None if math.isnan(energy_j) else energy_j,
                "avg_power_w": None if math.isnan(avg_power_w) else avg_power_w,
            }

            header_written = write_row(csv_path, header_written, row)

            energy_str = (
                f"{energy_j:.4f} J, P_avg = {avg_power_w:.2f} W"
                if not math.isnan(energy_j)
                else "energy = nan"
            )
            print(
                f"run {run_id:2d}: elapsed = {elapsed_s:8.4f} s, "
                f"GB/s = {gbps:8.2f}, {energy_str}"
            )

        mean_gbps = stats.mean(gbps_values)
        stdev_gbps = stats.pstdev(gbps_values) if len(gbps_values) > 1 else 0.0
        print(
            f"==> MEAN: {mean_gbps:8.2f} GB/s, "
            f"sigma = {stdev_gbps:6.2f} GB/s"
        )

        if energy_values:
            mean_e = stats.mean(energy_values)
            stdev_e = stats.pstdev(energy_values) if len(energy_values) > 1 else 0.0
            print(
                f"    MEAN energy per run: {mean_e:.4f} J, "
                f"sigma = {stdev_e:.4f} J"
            )

        if avg_power_values:
            mean_p = stats.mean(avg_power_values)
            stdev_p = (
                stats.pstdev(avg_power_values) if len(avg_power_values) > 1 else 0.0
            )
            print(
                f"    MEAN P_avg per run: {mean_p:.2f} W, "
                f"sigma = {stdev_p:.2f} W"
            )

    print(f"\nAll runs saved to: {csv_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU memory bandwidth benchmark (CUDA)"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--sizes-mb",
        type=str,
        default="4,16,64,256,1024",
        help="Comma-separated buffer sizes in MB (default: 4,16,64,256,1024)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of kernel launches per run (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        help="Number of runs per buffer size (default: 7)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    backend = CudaBackend()
    device_count = backend.device_count()
    if device_count == 0:
        raise SystemExit("No CUDA devices detected.")

    if args.device_index < 0 or args.device_index >= device_count:
        raise SystemExit(
            f"Invalid --device-index {args.device_index}; "
            f"available devices: 0..{device_count - 1}"
        )

    sizes_mb = [int(x) for x in args.sizes_mb.split(",") if x.strip()]

    run_bandwidth_bench(
        device_index=args.device_index,
        sizes_mb=sizes_mb,
        iters_per_run=args.iters,
        runs_per_size=args.runs,
    )


if __name__ == "__main__":
    main()
