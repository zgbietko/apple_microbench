#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${SCRIPT_DIR}/gpubench.cu"
LIBDIR="${SCRIPT_DIR}/lib"
OUT="${LIBDIR}/libgpubench.so"

mkdir -p "${LIBDIR}"

echo "[INFO] Buduję bibliotekę CUDA: ${OUT}"
nvcc -O3 --compiler-options="-fPIC" -shared "${SRC}" -o "${OUT}"

echo "[INFO] Zbudowano ${OUT}"
