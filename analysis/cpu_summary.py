# analysis/cpu_summary.py
#
# Zbiorcze podsumowanie benchmarków CPU dla WIELU procesorów.
# Wczytuje wszystkie pliki:
#   data/cpu/bandwidth_*.csv
#   data/cpu/bandwidth_mt_*.csv
#   data/cpu/pointer_latency_*.csv
#   data/cpu/compute_fma_*.csv
#   data/cpu/compute_fma_peak_*.csv
# (oraz ewentualne stare '..._results.csv', jeśli jeszcze istnieją)
#
# Następnie wypisuje:
# - jednowątkową przepustowość pamięci,
# - wielowątkową przepustowość,
# - latencję,
# - FMA (1T),
# - peak FMA (wielowątkowo),
# osobno dla każdej pary (arch, cpu_model).

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_CPU = ROOT / "data" / "cpu"


def _load_csv_multi(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    if not DATA_CPU.exists():
        return rows

    for pat in patterns:
        for path in DATA_CPU.glob(pat):
            try:
                with path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["_source_file"] = path.name
                        rows.append(row)
            except FileNotFoundError:
                continue
    return rows


def _nan_mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return (math.nan, math.nan)
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))


def summarize_bandwidth():
    rows = _load_csv_multi(["bandwidth_*.csv", "bandwidth_results.csv"])
    if not rows:
        print("=== Jednowątkowa przepustowość pamięci (brak danych) ===")
        return

    groups = defaultdict(lambda: {"gbps": [], "energy": [], "power": []})

    for r in rows:
        try:
            arch = r.get("arch", "unknown")
            cpu_model = r.get("cpu_model", "unknown")
            size_bytes = int(r["size_bytes"])
            gbps = float(r["gbps"])
        except (KeyError, ValueError):
            continue

        key = (arch, cpu_model, size_bytes)
        groups[key]["gbps"].append(gbps)

        # energia / moc opcjonalnie
        e = r.get("energy_j")
        p = r.get("power_w")
        try:
            energy = float(e) if e not in (None, "", "nan") else math.nan
        except ValueError:
            energy = math.nan
        try:
            power = float(p) if p not in (None, "", "nan") else math.nan
        except ValueError:
            power = math.nan

        groups[key]["energy"].append(energy)
        groups[key]["power"].append(power)

    print("=== Jednowątkowa przepustowość pamięci (bandwidth_*.csv) ===")
    for (arch, cpu_model, size_bytes) in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2])):
        data = groups[(arch, cpu_model, size_bytes)]
        g_mean, g_std = _nan_mean_std(data["gbps"])
        e_mean, e_std = _nan_mean_std(data["energy"])
        p_mean, p_std = _nan_mean_std(data["power"])

        size_mb = size_bytes / (1024 * 1024)
        print(f"{arch:<8} | {cpu_model:<32} | {size_mb:7.0f} MB | {g_mean:7.2f} GB/s ± {g_std:5.2f}")
        print(
            f"             energia: {e_mean:7.4f} J ± {e_std:7.4f}, "
            f"P_avg: {p_mean:6.2f} W ± {p_std:5.2f}"
        )


def summarize_bandwidth_mt():
    rows = _load_csv_multi(["bandwidth_mt_*.csv", "bandwidth_mt_results.csv"])
    if not rows:
        print("\n=== Wielowątkowa przepustowość pamięci (brak danych) ===")
        return

    groups = defaultdict(lambda: {"gbps": [], "energy": [], "power": []})
    for r in rows:
        try:
            arch = r.get("arch", "unknown")
            cpu_model = r.get("cpu_model", "unknown")
            size_bytes = int(r["size_bytes"])
            threads = int(r["threads"])
            gbps = float(r["gbps"])
        except (KeyError, ValueError):
            continue

        key = (arch, cpu_model, size_bytes, threads)
        groups[key]["gbps"].append(gbps)

        e = r.get("energy_j")
        p = r.get("power_w")
        try:
            energy = float(e) if e not in (None, "", "nan") else math.nan
        except ValueError:
            energy = math.nan
        try:
            power = float(p) if p not in (None, "", "nan") else math.nan
        except ValueError:
            power = math.nan

        groups[key]["energy"].append(energy)
        groups[key]["power"].append(power)

    print("\n=== Wielowątkowa przepustowość pamięci (bandwidth_mt_*.csv) ===")
    for (arch, cpu_model, size_bytes, threads) in sorted(
        groups.keys(), key=lambda x: (x[0], x[1], x[2], x[3])
    ):
        data = groups[(arch, cpu_model, size_bytes, threads)]
        g_mean, g_std = _nan_mean_std(data["gbps"])
        e_mean, e_std = _nan_mean_std(data["energy"])
        p_mean, p_std = _nan_mean_std(data["power"])

        size_mb = size_bytes / (1024 * 1024)
        print(
            f"{arch:<8} | {cpu_model:<24} | {size_mb:7.0f} MB | {threads:3d} th | "
            f"{g_mean:7.2f} GB/s ± {g_std:5.2f}"
        )
        print(
            f"             energia: {e_mean:7.4f} J ± {e_std:7.4f}, "
            f"P_avg: {p_mean:6.2f} W ± {p_std:5.2f}"
        )


def summarize_pointer_latency():
    rows = _load_csv_multi(["pointer_latency_*.csv", "pointer_latency_results.csv"])
    if not rows:
        print("\n=== Latencja pointer-chasing (brak danych) ===")
        return

    groups = defaultdict(lambda: {"lat": [], "energy": [], "power": []})
    for r in rows:
        try:
            arch = r.get("arch", "unknown")
            cpu_model = r.get("cpu_model", "unknown")
            size_bytes = int(r["size_bytes"])
            lat_ns = float(r["latency_ns"])
        except (KeyError, ValueError):
            continue

        key = (arch, cpu_model, size_bytes)
        groups[key]["lat"].append(lat_ns)

        e = r.get("energy_j")
        p = r.get("power_w")
        try:
            energy = float(e) if e not in (None, "", "nan") else math.nan
        except ValueError:
            energy = math.nan
        try:
            power = float(p) if p not in (None, "", "nan") else math.nan
        except ValueError:
            power = math.nan

        groups[key]["energy"].append(energy)
        groups[key]["power"].append(power)

    print("\n=== Latencja pointer-chasing (pointer_latency_*.csv) ===")
    for (arch, cpu_model, size_bytes) in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2])):
        data = groups[(arch, cpu_model, size_bytes)]
        lat_mean, lat_std = _nan_mean_std(data["lat"])
        e_mean, e_std = _nan_mean_std(data["energy"])
        p_mean, p_std = _nan_mean_std(data["power"])

        size_kb = size_bytes / 1024
        print(
            f"{arch:<8} | {cpu_model:<32} | {size_kb:8.0f} KB | "
            f"{lat_mean:7.2f} ns ± {lat_std:6.2f}"
        )
        print(
            f"             energia: {e_mean:7.4f} J ± {e_std:7.4f}, "
            f"P_avg: {p_mean:6.2f} W ± {p_std:5.2f}"
        )


def summarize_compute_fma():
    rows = _load_csv_multi(["compute_fma_*.csv", "compute_fma_results.csv"])
    if not rows:
        print("\n=== FMA compute throughput (brak danych) ===")
        return

    groups = defaultdict(lambda: {"gflops": [], "energy": [], "power": []})
    for r in rows:
        try:
            arch = r.get("arch", "unknown")
            cpu_model = r.get("cpu_model", "unknown")
            n = int(r["n"])
            iters = int(r["iters_inner"])
            gflops = float(r["gflops"])
        except (KeyError, ValueError):
            continue

        key = (arch, cpu_model, n, iters)
        groups[key]["gflops"].append(gflops)

        e = r.get("energy_j")
        p = r.get("power_w")
        try:
            energy = float(e) if e not in (None, "", "nan") else math.nan
        except ValueError:
            energy = math.nan
        try:
            power = float(p) if p not in (None, "", "nan") else math.nan
        except ValueError:
            power = math.nan

        groups[key]["energy"].append(energy)
        groups[key]["power"].append(power)

    print("\n=== FMA compute throughput (compute_fma_*.csv) ===")
    for (arch, cpu_model, n, iters) in sorted(
        groups.keys(), key=lambda x: (x[0], x[1], x[2], x[3])
    ):
        data = groups[(arch, cpu_model, n, iters)]
        g_mean, g_std = _nan_mean_std(data["gflops"])
        e_mean, e_std = _nan_mean_std(data["energy"])
        p_mean, p_std = _nan_mean_std(data["power"])

        print(
            f"{arch:<8} | {cpu_model:<32} | n={n:6d} | iters={iters:8d} | "
            f"{g_mean:7.2f} GF/s ± {g_std:5.2f}"
        )
        print(
            f"             energia: {e_mean:7.4f} J ± {e_std:7.4f}, "
            f"P_avg: {p_mean:6.2f} W ± {p_std:5.2f}"
        )


def summarize_compute_fma_peak():
    rows = _load_csv_multi(["compute_fma_peak_*.csv", "compute_fma_peak_results.csv"])
    if not rows:
        print("\n=== Peak FMA throughput (brak danych) ===")
        return

    # Dla każdej kombinacji (arch, cpu_model, n_per_thread, threads) wybieramy WIERZCHOŁEK (max GF/s)
    best = {}

    for r in rows:
        try:
            arch = r.get("arch", "unknown")
            cpu_model = r.get("cpu_model", "unknown")
            n_thr = int(r["n_per_thread"])
            threads = int(r["threads"])
            iters = int(r["iters_inner"])
            gflops = float(r["gflops"])
        except (KeyError, ValueError):
            continue

        key = (arch, cpu_model, n_thr, threads)

        e = r.get("energy_j")
        p = r.get("power_w")
        try:
            energy = float(e) if e not in (None, "", "nan") else math.nan
        except ValueError:
            energy = math.nan
        try:
            power = float(p) if p not in (None, "", "nan") else math.nan
        except ValueError:
            power = math.nan

        rec = best.get(key)
        if rec is None or gflops > rec["gflops"]:
            best[key] = {
                "gflops": gflops,
                "iters_inner": iters,
                "energy_j": energy,
                "power_w": power,
            }

    print("\n=== Peak FMA throughput (compute_fma_peak_*.csv) ===")
    for (arch, cpu_model, n_thr, threads) in sorted(
        best.keys(), key=lambda x: (x[0], x[1], x[2], x[3])
    ):
        rec = best[(arch, cpu_model, n_thr, threads)]
        g = rec["gflops"]
        iters = rec["iters_inner"]
        e = rec["energy_j"]
        p = rec["power_w"]

        print(
            f"{arch:<8} | {cpu_model:<32} | n_thr={n_thr:5d} | th={threads:2d} | "
            f"peak={g:7.2f} GF/s @ iters={iters:8d}"
        )
        print(
            f"             energia: {e:7.4f} J ± {0.0:7.4f}, "
            f"P_avg: {p:6.2f} W ± {0.0:5.2f}"
        )


def main():
    summarize_bandwidth()
    summarize_bandwidth_mt()
    summarize_pointer_latency()
    summarize_compute_fma()
    summarize_compute_fma_peak()


if __name__ == "__main__":
    main()
