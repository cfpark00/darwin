#!/bin/bash
# Run all diffusion v2 experiments (with fast resource regen timescale=10)
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

echo "=== Diffusion v2: E=100 suite ==="
for r in 5 10 15 20 25 30 35 40; do
    echo "--- Resource level $r ---"
    bash scripts/diffusion_v2/r${r}.sh "$@"
done

echo "=== Diffusion v2: E=30 suite ==="
for r in 5 10 15 20 25 30 35 40; do
    echo "--- E=30, Resource level $r ---"
    bash scripts/diffusion_v2/e30_r${r}.sh "$@"
done

echo "=== Diffusion v2: E=15 suite ==="
for r in 5 10 15 20 25 30 35 40; do
    echo "--- E=15, Resource level $r ---"
    bash scripts/diffusion_v2/e15_r${r}.sh "$@"
done

echo "=== All diffusion v2 experiments complete ==="
