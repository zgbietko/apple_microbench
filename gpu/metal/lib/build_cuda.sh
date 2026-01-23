#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[build_cuda] nvcc (CUDA toolkit) not found in PATH" >&2
  exit 1
fi

UNAME="$(uname)"
if [ "$UNAME" = "Darwin" ]; then
  OUT_LIB="libgpubench.dylib"
  nvcc -Xcompiler "-fPIC" -shared gpubench.cu -o "$OUT_LIB"
elif [ "$UNAME" = "Linux" ]; then
  OUT_LIB="libgpubench.so"
  nvcc -Xcompiler "-fPIC" -shared gpubench.cu -o "$OUT_LIB"
else
  OUT_LIB="gpubench.dll"
  nvcc -Xcompiler "-MD" -shared gpubench.cu -o "$OUT_LIB"
fi

echo "[build_cuda] Built $OUT_LIB in $SCRIPT_DIR"
