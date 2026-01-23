from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(rel_path: str) -> bool:
    script_path = ROOT / rel_path
    if not script_path.exists():
        print(f"[WARN] Pomijam {rel_path}, plik nie istnieje.")
        return True
    print(f"\n=== Uruchamiam: {rel_path} ===")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Skrypt {rel_path} zakończył się kodem {result.returncode}")
        return False
    return True


def main() -> None:
    ok = True

    # 1) CPU
    ok &= run_script("run_all_cpu_benchmarks.py")

    # 2) GPU
    ok &= run_script("run_all_gpu_benchmarks.py")

    if ok:
        print("\n[INFO] Wszystkie benchmarki (CPU + GPU) zakończone pomyślnie.")
    else:
        print(
            "\n[ERROR] Co najmniej jeden benchmark (CPU lub GPU) "
            "zakończył się błędem."
        )


if __name__ == "__main__":
    main()
