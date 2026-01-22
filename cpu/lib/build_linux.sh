#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Buduję libmicrobench.so dla Linuxa..."

SRC="microbench.c"
OUT="libmicrobench.so"

# Sprzątnij starą bibliotekę, jeśli istnieje
if [ -f "$OUT" ]; then
    echo "[INFO] Usuwam stare $OUT"
    rm -f "$OUT"
fi

# Kompilacja:
# -O3          : maksymalna optymalizacja
# -std=c11     : standard C11
# -fPIC        : kod współdzielony
# -shared      : budujemy bibliotekę współdzieloną .so
# -pthread     : wątkowanie (pthreads)
gcc -O3 -std=c11 -fPIC -shared "$SRC" -o "$OUT" -pthread

echo "[INFO] Zbudowano $OUT"
ls -l "$OUT"
