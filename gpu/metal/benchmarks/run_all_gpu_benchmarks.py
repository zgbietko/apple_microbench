# gpu/run_all_gpu_benchmarks.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(rel_path: str, extra_args=None) -> bool:
    script_path = ROOT / rel_path
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n=== Uruchamiam: {rel_path} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Skrypt {rel_path} zakończył się kodem {result.returncode}")
        return False
    return True


def main():
    # Możesz tutaj przekazać np. --device 1 jeśli chcesz konkretną kartę.
    extra_args = []  # np. ["--device", "1"]

    ok = run_script("gpu/benchmarks/run_gpu_bandwidth.py", extra_args=extra_args)
    if not ok:
        sys.exit(1)

    ok = run_script("gpu/benchmarks/run_gpu_compute_fma.py", extra_args=extra_args)
    if not ok:
        sys.exit(1)

    print("\n[INFO] Wszystkie benchmarki GPU CUDA zakończone.")


if __name__ == "__main__":
    main()
