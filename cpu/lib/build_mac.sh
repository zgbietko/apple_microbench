#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_LIB="libmicrobench.dylib"

echo "Kompiluję do: $SCRIPT_DIR/$OUT_LIB"

clang -O3 -std=c11 -fPIC -dynamiclib microbench.c -o "$OUT_LIB" -pthread

echo "Gotowe. Plik:"
ls -l "$OUT_LIB"
