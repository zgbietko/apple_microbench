# cpu/benchmarks/run_pointer_latency.py
import ctypes as ct
import time
import csv
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import statistics as stats
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # .../apple_microbench
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from energy import energy_measurement_supported, read_energy_joules


def load_library():
    """Ładuje libmicrobench.* z cpu/lib – cross-platform."""
    root = Path(__file__).resolve().parents[2]  # -> apple_microbench/
    system = platform.system()

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
    func = lib.pointer_chase_kernel
    func.argtypes = [
        ct.POINTER(ct.c_uint32),  # buf
        ct.c_size_t,              # n
        ct.c_size_t,              # iters
    ]
    func.restype = None
    return func


def detect_cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
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


def make_random_cycle_buffer(n: int) -> np.ndarray:
    """
    Tworzy JEDEN cykl po losowej permutacji indeksów 0..n-1.
    """
    perm = np.arange(n, dtype=np.uint32)
    rng = np.random.default_rng()
    rng.shuffle(perm)

    buf = np.empty(n, dtype=np.uint32)
    for i in range(n):
        cur = perm[i]
        nxt = perm[(i + 1) % n]
        buf[cur] = nxt
    return buf


def bench_pointer_chase(pointer_chase_kernel, working_set_bytes: int, iters_inner: int):
    n = working_set_bytes // 4  # uint32

    buf = make_random_cycle_buffer(n)
    buf_p = buf.ctypes.data_as(ct.POINTER(ct.c_uint32))

    # rozgrzewka
    pointer_chase_kernel(buf_p, n, min(iters_inner, 1000))

    energy_j = None
    e_before = None
    if energy_measurement_supported():
        e_before = read_energy_joules()

    t0 = time.perf_counter()
    pointer_chase_kernel(buf_p, n, iters_inner)
    t1 = time.perf_counter()

    if e_before is not None:
        e_after = read_energy_joules()
        if e_after is not None:
            delta = e_after - e_before
            if delta >= 0:
                energy_j = delta

    elapsed = t1 - t0
    latency_ns = (elapsed / iters_inner) * 1e9
    avg_power_w = energy_j / elapsed if (energy_j is not None and elapsed > 0) else None

    return {
        "working_set_bytes": working_set_bytes,
        "iters_inner": iters_inner,
        "elapsed_s": elapsed,
        "latency_ns": latency_ns,
        "energy_j": energy_j,
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
        "processor": platform.processor(),
        "cpu_model": detect_cpu_model(),
        "python_version": platform.python_version(),
    }


def write_result_to_csv(
    csv_path: Path,
    result: dict,
    meta: dict,
    write_header: bool,
):
    row = {
        **meta,
        "benchmark": "pointer_chase",
        **result,  # run_id, energy_j, avg_power_w
    }

    fieldnames = [
        "timestamp",
        "system",
        "node",
        "release",
        "version",
        "machine",
        "processor",
        "cpu_model",
        "python_version",
        "benchmark",
        "run_id",
        "working_set_bytes",
        "iters_inner",
        "elapsed_s",
        "latency_ns",
        "energy_j",
        "avg_power_w",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    lib, root = load_library()
    pointer_chase_kernel = configure_functions(lib)
    meta = collect_system_metadata()

    data_dir = root / "data" / "cpu"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "pointer_latency_results.csv"
    if csv_path.exists():
        csv_path.unlink()  # nadpisujemy stare wyniki

    sizes_kb = [4, 16, 64, 256, 1024, 4096, 16384, 65536]  # 4KB .. 64MB
    iters_inner = 50_000_000
    runs = 5

    print("=== CPU pointer-chasing latency benchmark (random cycle) ===")
    print(f"runs per size : {runs}")
    print(f"iters per run : {iters_inner}")
    print(f"CPU model     : {meta['cpu_model']}")
    if energy_measurement_supported():
        print("Energy        : Linux RAPL, per-run energy_j / avg_power_w")
    else:
        print("Energy        : pomiar niedostępny na tej platformie")

    header_written = False

    for size_kb in sizes_kb:
        working_set_bytes = size_kb * 1024
        latency_values = []
        energy_values = []

        print(f"\n--- Working set: {size_kb} KB ---")

        for run_id in range(runs):
            result = bench_pointer_chase(pointer_chase_kernel, working_set_bytes, iters_inner)
            latency_values.append(result["latency_ns"])
            if result["energy_j"] is not None:
                energy_values.append(result["energy_j"])

            write_result_to_csv(
                csv_path,
                {**result, "run_id": run_id},
                meta,
                write_header=not header_written,
            )
            header_written = True

            line = (
                f"run {run_id:2d}: "
                f"elapsed = {result['elapsed_s']:.4f} s, "
                f"latency = {result['latency_ns']:.2f} ns"
            )
            if result["avg_power_w"] is not None:
                line += f", P_avg = {result['avg_power_w']:.2f} W"
            print(line)

        mean_lat = stats.mean(latency_values)
        stdev_lat = stats.pstdev(latency_values) if len(latency_values) > 1 else 0.0
        print(f"==> ŚREDNIA: {mean_lat:.2f} ns, σ = {stdev_lat:.2f} ns")

        if energy_values:
            mean_energy = stats.mean(energy_values)
            stdev_energy = stats.pstdev(energy_values) if len(energy_values) > 1 else 0.0
            print(f"    ŚREDNIA energia per run: {mean_energy:.4f} J, σ = {stdev_energy:.4f} J")

    print(f"\nWszystkie runy zapisane do: {csv_path}")


if __name__ == "__main__":
    main()
