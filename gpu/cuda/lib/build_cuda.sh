#!/usr/bin/env bash
# Simple build script for CUDA microbenchmarks.
# Builds a shared library libgpubench_cuda.* from gpubench.cu.
#
# Usage:
#   cd apple_microbench/gpu/cuda/lib
#   ./build_cuda.sh
#
# You need CUDA toolkit (nvcc) in PATH.

set -euo pipefail

SRC="gpubench.cu"

if ! command -v nvcc >/dev/null 2>&1 ; then
    echo "ERROR: nvcc not found in PATH. Install CUDA toolkit first." >&2
    exit 1
fi

UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    Linux*)
        LIB_NAME="libgpubench_cuda.so"
        ;;
    Darwin*)
        LIB_NAME="libgpubench_cuda.dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        LIB_NAME="gpubench_cuda.dll"
        ;;
    *)
        echo "WARNING: Unknown OS ${UNAME_OUT}, defaulting to .so"
        LIB_NAME="libgpubench_cuda.so"
        ;;
esac

echo "Building ${LIB_NAME} from ${SRC} ..."

nvcc -O3 -std=c++14 -Xcompiler "-fPIC" -shared "${SRC}" -o "${LIB_NAME}"

echo "Done. Output: ${LIB_NAME}"
