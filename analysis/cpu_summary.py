#!/usr/bin/env python3
"""
Podsumowanie wyników CPU:
- bandwidth_results.csv
- bandwidth_mt_results.csv
- pointer_latency_results.csv
- compute_fma_results.csv
- compute_fma_peak_results.csv

Dodatkowo, jeśli w CSV są kolumny:
- energy_j
- avg_power_w

to liczone są średnie i odchylenia energii/mocy
i wypisywane pod każdą konfiguracją.
"""

import csv
from pathlib import Path
from statistics import mean, pstdev


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    m = mean(vals)
    s = pstdev(vals)
    return m, s


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]  # apple_microbench/


# === 1. Jednowątkowa przepustowość pamięci ===

def summarize_bandwidth():
    data_path = root_dir() / "data" / "cpu" / "bandwidth_results.csv"
    if not data_path.exists():
        print("=== Jednowątkowa przepustowość pamięci (bandwidth_results.csv) ===")
        print("Brak pliku:", data_path)
        print()
        return

    rows = list(csv.DictReader(data_path.open()))

    groups = {}
    for row in rows:
        arch = row.get("machine", "") or ""
        cpu_model = row.get("cpu_model", "") or ""
        bpi = safe_float(row.get("bytes_per_iter"))
        gbps = safe_float(row.get("gbps"))
        energy = safe_float(row.get("energy_j"))
        power = safe_float(row.get("avg_power_w"))

        if bpi is None or gbps is None:
            continue

        size_mb = int(bpi / (1024 * 1024))
        key = (arch, cpu_model, size_mb)

        if key not in groups:
            groups[key] = {"gbps": [], "energy": [], "power": []}
        groups[key]["gbps"].append(gbps)
        if energy is not None:
            groups[key]["energy"].append(energy)
        if power is not None:
            groups[key]["power"].append(power)

    print("=== Jednowątkowa przepustowość pamięci (bandwidth_results.csv) ===")
    for (arch, cpu_model, size_mb) in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2])):
        g = groups[(arch, cpu_model, size_mb)]
        m_gbps, s_gbps = mean_std(g["gbps"])
        if m_gbps is None:
            continue

        print(f"{arch:<10} | {cpu_model:<35} | {size_mb:7d} MB | {m_gbps:7.2f} GB/s ± {s_gbps:5.2f}")

        m_e, s_e = mean_std(g["energy"])
        m_p, s_p = mean_std(g["power"])
        if m_e is not None and m_p is not None:
            print(
                f"{'':10}   energia: {m_e:7.4f} J ± {s_e:6.4f}, "
                f"P_avg: {m_p:6.2f} W ± {s_p:5.2f}"
            )

    print()


# === 2. Wielowątkowa przepustowość pamięci ===

def summarize_bandwidth_mt():
    data_path = root_dir() / "data" / "cpu" / "bandwidth_mt_results.csv"
    if not data_path.exists():
        print("=== Wielowątkowa przepustowość pamięci (bandwidth_mt_results.csv) ===")
        print("Brak pliku:", data_path)
        print()
        return

    rows = list(csv.DictReader(data_path.open()))

    groups = {}
    for row in rows:
        arch = row.get("machine", "") or ""
        cpu_model = row.get("cpu_model", "") or ""
        bpi = safe_float(row.get("bytes_per_iter"))
        gbps = safe_float(row.get("gbps"))
        nth = safe_float(row.get("num_threads"))
        energy = safe_float(row.get("energy_j"))
        power = safe_float(row.get("avg_power_w"))

        if bpi is None or gbps is None or nth is None:
            continue

        size_mb = int(bpi / (1024 * 1024))
        num_threads = int(nth)
        key = (arch, cpu_model, size_mb, num_threads)

        if key not in groups:
            groups[key] = {"gbps": [], "energy": [], "power": []}
        groups[key]["gbps"].append(gbps)
        if energy is not None:
            groups[key]["energy"].append(energy)
        if power is not None:
            groups[key]["power"].append(power)

    print("=== Wielowątkowa przepustowość pamięci (bandwidth_mt_results.csv) ===")
    for (arch, cpu_model, size_mb, num_threads) in sorted(
        groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3])
    ):
        g = groups[(arch, cpu_model, size_mb, num_threads)]
        m_gbps, s_gbps = mean_std(g["gbps"])
        if m_gbps is None:
            continue

        print(
            f"{arch:<10} | {cpu_model:<27} | {size_mb:7d} MB | {num_threads:3d} th "
            f"| {m_gbps:7.2f} GB/s ± {s_gbps:5.2f}"
        )

        m_e, s_e = mean_std(g["energy"])
        m_p, s_p = mean_std(g["power"])
        if m_e is not None and m_p is not None:
            print(
                f"{'':10}   energia: {m_e:7.4f} J ± {s_e:6.4f}, "
                f"P_avg: {m_p:6.2f} W ± {s_p:5.2f}"
            )

    print()


# === 3. Latencja pointer-chasing ===

def summarize_pointer_latency():
    data_path = root_dir() / "data" / "cpu" / "pointer_latency_results.csv"
    if not data_path.exists():
        print("=== Latencja pointer-chasing (pointer_latency_results.csv) ===")
        print("Brak pliku:", data_path)
        print()
        return

    rows = list(csv.DictReader(data_path.open()))

    groups = {}
    for row in rows:
        arch = row.get("machine", "") or ""
        cpu_model = row.get("cpu_model", "") or ""
        wsb = safe_float(row.get("working_set_bytes"))
        lat_ns = safe_float(row.get("latency_ns"))
        energy = safe_float(row.get("energy_j"))
        power = safe_float(row.get("avg_power_w"))

        if wsb is None or lat_ns is None:
            continue

        size_kb = int(wsb / 1024)
        key = (arch, cpu_model, size_kb)

        if key not in groups:
            groups[key] = {"lat": [], "energy": [], "power": []}
        groups[key]["lat"].append(lat_ns)
        if energy is not None:
            groups[key]["energy"].append(energy)
        if power is not None:
            groups[key]["power"].append(power)

    print("=== Latencja pointer-chasing (pointer_latency_results.csv) ===")
    for (arch, cpu_model, size_kb) in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2])):
        g = groups[(arch, cpu_model, size_kb)]
        m_lat, s_lat = mean_std(g["lat"])
        if m_lat is None:
            continue

        print(
            f"{arch:<10} | {cpu_model:<35} | {size_kb:8d} KB | {m_lat:7.2f} ns ± {s_lat:6.2f}"
        )

        m_e, s_e = mean_std(g["energy"])
        m_p, s_p = mean_std(g["power"])
        if m_e is not None and m_p is not None:
            print(
                f"{'':10}   energia: {m_e:7.4f} J ± {s_e:6.4f}, "
                f"P_avg: {m_p:6.2f} W ± {s_p:5.2f}"
            )

    print()


# === 4. FMA compute throughput ===

def summarize_compute_fma():
    data_path = root_dir() / "data" / "cpu" / "compute_fma_results.csv"
    if not data_path.exists():
        print("=== FMA compute throughput (compute_fma_results.csv) ===")
        print("Brak pliku:", data_path)
        print()
        return

    rows = list(csv.DictReader(data_path.open()))

    groups = {}
    for row in rows:
        arch = row.get("machine", "") or ""
        cpu_model = row.get("cpu_model", "") or ""
        n = safe_float(row.get("vector_len"))
        iters = safe_float(row.get("iters_inner"))
        gflops = safe_float(row.get("gflops"))
        energy = safe_float(row.get("energy_j"))
        power = safe_float(row.get("avg_power_w"))

        if n is None or iters is None or gflops is None:
            continue

        n_int = int(n)
        it_int = int(iters)
        key = (arch, cpu_model, n_int, it_int)

        if key not in groups:
            groups[key] = {"gflops": [], "energy": [], "power": []}
        groups[key]["gflops"].append(gflops)
        if energy is not None:
            groups[key]["energy"].append(energy)
        if power is not None:
            groups[key]["power"].append(power)

    print("=== FMA compute throughput (compute_fma_results.csv) ===")
    for (arch, cpu_model, n, iters) in sorted(
        groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3])
    ):
        g = groups[(arch, cpu_model, n, iters)]
        m_gf, s_gf = mean_std(g["gflops"])
        if m_gf is None:
            continue

        print(
            f"{arch:<10} | {cpu_model:<35} | n={n:6d} | iters={iters:8d} | "
            f"{m_gf:7.2f} GF/s ± {s_gf:5.2f}"
        )

        m_e, s_e = mean_std(g["energy"])
        m_p, s_p = mean_std(g["power"])
        if m_e is not None and m_p is not None:
            print(
                f"{'':10}   energia: {m_e:7.4f} J ± {s_e:6.4f}, "
                f"P_avg: {m_p:6.2f} W ± {s_p:5.2f}"
            )

    print()


# === 5. Peak FMA throughput ===

def summarize_compute_fma_peak():
    data_path = root_dir() / "data" / "cpu" / "compute_fma_peak_results.csv"
    if not data_path.exists():
        print("=== Peak FMA throughput (compute_fma_peak_results.csv) ===")
        print("Brak pliku:", data_path)
        print()
        return

    rows = list(csv.DictReader(data_path.open()))

    # grupujemy po (arch, cpu_model, num_threads, iters_inner)
    groups = {}
    for row in rows:
        arch = row.get("machine", "") or ""
        cpu_model = row.get("cpu_model", "") or ""
        n_per_thread = safe_float(row.get("n_per_thread"))
        iters = safe_float(row.get("iters_inner"))
        nth = safe_float(row.get("num_threads"))
        gflops = safe_float(row.get("gflops"))
        energy = safe_float(row.get("energy_j"))
        power = safe_float(row.get("avg_power_w"))

        if n_per_thread is None or iters is None or nth is None or gflops is None:
            continue

        npt = int(n_per_thread)
        it_int = int(iters)
        num_threads = int(nth)
        key = (arch, cpu_model, num_threads, npt, it_int)

        if key not in groups:
            groups[key] = {"gflops": [], "energy": [], "power": []}
        groups[key]["gflops"].append(gflops)
        if energy is not None:
            groups[key]["energy"].append(energy)
        if power is not None:
            groups[key]["power"].append(power)

    # dla każdego (arch,cpu_model,num_threads) wybierz iters dające maksymalne średnie GF/s
    print("=== Peak FMA throughput (compute_fma_peak_results.csv) ===")

    # pomocnicza struktura: peaks[(arch,cpu_model,num_threads)] = (best_npt, best_iters, m_gf, s_gf, m_e, s_e, m_p, s_p)
    peaks = {}

    for (arch, cpu_model, num_threads, npt, iters) in groups.keys():
        g = groups[(arch, cpu_model, num_threads, npt, iters)]
        m_gf, s_gf = mean_std(g["gflops"])
        if m_gf is None:
            continue
        m_e, s_e = mean_std(g["energy"])
        m_p, s_p = mean_std(g["power"])

        key_base = (arch, cpu_model, num_threads)
        prev = peaks.get(key_base)
        if prev is None or m_gf > prev[2]:
            peaks[key_base] = (npt, iters, m_gf, s_gf, m_e, s_e, m_p, s_p)

    for (arch, cpu_model, num_threads) in sorted(peaks.keys(), key=lambda k: (k[0], k[1], k[2])):
        npt, iters, m_gf, s_gf, m_e, s_e, m_p, s_p = peaks[(arch, cpu_model, num_threads)]

        print(
            f"{arch:<10} | {cpu_model:<35} | n_thr={npt:4d} | th={num_threads:2d} "
            f"| peak={m_gf:7.2f} GF/s @ iters={iters:8d}"
        )

        if m_e is not None and m_p is not None:
            print(
                f"{'':10}   energia: {m_e:7.4f} J ± {s_e:6.4f}, "
                f"P_avg: {m_p:6.2f} W ± {s_p:5.2f}"
            )

    print()


def main():
    summarize_bandwidth()
    summarize_bandwidth_mt()
    summarize_pointer_latency()
    summarize_compute_fma()
    summarize_compute_fma_peak()


if __name__ == "__main__":
    main()
