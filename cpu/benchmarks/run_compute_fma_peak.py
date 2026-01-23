# cpu/benchmarks/run_compute_fma_peak.py
import ctypes as ct
import time
import csv
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]  # .../apple_microbench
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from energy import energy_measurement_supported, read_energy_joules
from cpu_utils import make_cpu_specific_csv_path


def load_library():
    """Ładuje libmicrobench.* z cpu/lib – cross-platform."""
    system = platform.system()
    root = ROOT

    if system == "Darwin":
        lib_name = "libmicrobench.dylib"
    elif system == "Linux":
        lib_name = "libmicrobench.so"
    elif system == "Windows":
        lib_name = "microbench.dll"
    else:
        raise RuntimeError(f"Nieobsługiwany system: {system}")

    lib_path = root / "cpu" / "lib" / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(f"Nie znaleziono biblioteki: {lib_path}")

    return ct.CDLL(str(lib_path)), root


def configure_functions(lib):
    func = lib.fma_peak_mt
    func.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_int,
    ]
    func.restype = None
    return func


def detect_cpu_model() -> str:
    """
    Próbuje wykryć pełny model CPU (Apple M2 Pro, Intel(R) Core..., itd.).
    """
    system = platform.system()

    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out

        elif system == "Linux":
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()

        elif system == "Windows":
            return platform.processor() or platform.uname().processor
    except Exception:
        pass

    return platform.processor() or platform.machine()


FMA_PEAK_N = 256


def bench_fma_peak(fma_peak_mt, n_per_thread: int, iters_inner: int, num_threads: int):
    energy_j = None
    e_before = None
    if energy_measurement_supported():
        e_before = read_energy_joules()

    t0 = time.perf_counter()
    fma_peak_mt(n_per_thread, iters_inner, num_threads)
    t1 = time.perf_counter()

    if e_before is not None:
        e_after = read_energy_joules()
        if e_after is not None:
            delta = e_after - e_before
            if delta >= 0:
                energy_j = delta

    elapsed = t1 - t0
    total_ops = 2 * n_per_thread * iters_inner * num_threads
    gflops = total_ops / elapsed / 1e9

    avg_power_w = None
    if energy_j is not None and elapsed > 0:
        avg_power_w = energy_j / elapsed

    return {
        "iters_inner": iters_inner,
        "threads": num_threads,
        "num_threads": num_threads,
        "n_per_thread": n_per_thread,
        "elapsed_s": elapsed,
        "gflops": gflops,
        "energy_j": energy_j,
        "power_w": avg_power_w,
        "avg_power_w": avg_power_w,
    }


def collect_system_metadata():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": detect_cpu_model(),
        "python_version": platform.python_version(),
    }


def write_result_to_csv(
    csv_path: Path,
    result: dict,
    meta: dict,
    benchmark_name: str,
    write_header: bool,
):
    row = {
        **meta,
        "benchmark": benchmark_name,
        **result,
    }

    fieldnames = [
        "timestamp",
        "system",
        "node",
        "release",
        "version",
        "machine",
        "arch",
        "processor",
        "cpu_model",
        "python_version",
        "benchmark",
        "run_id",
        "iters_inner",
        "threads",
        "num_threads",
        "n_per_thread",
        "elapsed_s",
        "gflops",
        "energy_j",
        "power_w",
        "avg_power_w",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    lib, root = load_library()
    fma_peak_mt = configure_functions(lib)

    data_dir = root / "data" / "cpu"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path, arch, cpu_model, cpu_slug = make_cpu_specific_csv_path(
        "compute_fma_peak",
        data_dir,
    )
    if csv_path.exists():
        csv_path.unlink()  # nadpisujemy stare wyniki

    meta = collect_system_metadata()
    meta["arch"] = arch
    meta["cpu_model"] = cpu_model

    threads_list = [1, 2, 4, 8]
    iters_list = [500_000, 1_000_000, 2_000_000]
    runs = 3

    print("=== CPU Peak FMA benchmark (multi-thread) ===")
    print(f"n_per_thread     : {FMA_PEAK_N}")
    print(f"runs per config  : {runs}")
    print(f"CPU model        : {meta['cpu_model']}")
    if energy_measurement_supported():
        print("Energy           : Linux RAPL, per-run energy_j / avg_power_w")
    else:
        print("Energy           : pomiar niedostępny na tej platformie")

    header_written = False

    for num_threads in threads_list:
        for iters_inner in iters_list:
            print(f"\n### num_threads = {num_threads} ###")
            print(f"\n--- iters_inner = {iters_inner} ---")

            gflops_values = []
            energy_values = []

            for run_id in range(runs):
                result = bench_fma_peak(fma_peak_mt, FMA_PEAK_N, iters_inner, num_threads)
                gflops_values.append(result["gflops"])
                if result["energy_j"] is not None:
                    energy_values.append(result["energy_j"])

                write_result_to_csv(
                    csv_path,
                    {**result, "run_id": run_id},
                    meta,
                    "fma_peak_mt",
                    write_header=not header_written,
                )
                header_written = True

                print(
                    f"run  {run_id:2d}: elapsed = {result['elapsed_s']:.4f} s, "
                    f"GFlop/s = {result['gflops']:.2f}, "
                    f"energy = {result['energy_j'] if result['energy_j'] is not None else float('nan'):.4f} J"
                )

            mean_gflops = sum(gflops_values) / len(gflops_values)
            if len(gflops_values) > 1:
                import statistics as stats

                stdev_gflops = stats.pstdev(gflops_values)
            else:
                stdev_gflops = 0.0

            print(f"==> ŚREDNIA: {mean_gflops:.2f} GF/s, σ = {stdev_gflops:.2f} GF/s")

            if energy_values:
                import statistics as stats

                mean_energy = sum(energy_values) / len(energy_values)
                stdev_energy = stats.pstdev(energy_values) if len(energy_values) > 1 else 0.0
                print(f"    ŚREDNIA energia per run: {mean_energy:.4f} J, σ = {stdev_energy:.4f} J")

    print(f"\nWszystkie runy zapisane do: {csv_path}")


if __name__ == "__main__":
    main()
